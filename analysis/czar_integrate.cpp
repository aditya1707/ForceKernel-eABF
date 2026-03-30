/*
 * czar_integrate.cpp
 * ==================
 * Standalone C++ tool to recover the PMF A(z) from FKERNELABF CZAR kernel files.
 *
 * Reads z-kernel snapshot files, evaluates the CZAR gradient on a regular grid
 * (Nadaraya-Watson regression + finite-difference log-density correction),
 * then integrates using abf_integrate's metadynamics-style MC random walk.
 *
 * No PLUMED dependency — compiles with any C++11 compiler:
 *   g++ -O2 -o czar_integrate czar_integrate.cpp -lm
 *
 * Usage:
 *   ./czar_integrate                       Print help and exit
 *   ./czar_integrate <output_dir>          Scan CWD for *czar_kernels_*.dat, write FELs
 *   ./czar_integrate <output_dir> [opts]   Same, with options
 *
 * Options (same conventions as abf_integrate):
 *   -n <steps>      MC integration steps (0 = auto-converge) [default: 0]
 *   -h <height>     Initial hill height                      [default: 0.01]
 *   -f <factor>     Hill reduction factor                    [default: 0.5]
 *   -t <kT>         Override kT from file (kJ/mol)           [default: from file]
 *   -g <pts>        Grid points per dimension                [default: 100]
 *   -s <nsigma>     Kernel cutoff in sigma units             [default: 4.0]
 *   -m <minpop>     Min density fraction for allowed region  [default: 1e-3]
 *   -i <file>       Process a single kernel file instead of scanning
 *   -o <file>       Output file for single-file mode         [default: FEL_czar.dat]
 *   -d <dir>        Directory to scan for snapshots          [default: .]
 *   -v              Verbose output
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <dirent.h>
#include <sys/stat.h>

// ─────────────────────── data structures ────────────────────────────────────

struct Kernel {
    double Nk;
    std::vector<double> center;
    std::vector<double> mu;
    std::vector<double> sigma;
};

struct Meta {
    int dim;
    double kT;
    std::vector<double> kappa;
    std::vector<bool> periodic;
    std::vector<double> domMin, domMax;
    std::vector<double> sigma0;  // initial bandwidth (for KDE normalization)
};

// ─────────────────────── file reader ────────────────────────────────────────

bool parse_czar_file(const char *path, Meta &meta, std::vector<Kernel> &kernels) {
    std::ifstream fh(path);
    if (!fh.is_open()) {
        fprintf(stderr, "ERROR: cannot open %s\n", path);
        return false;
    }

    meta.dim = 0;
    meta.kT = 0;
    kernels.clear();

    std::string line;
    while (std::getline(fh, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string key;
        iss >> key;

        if (key == "dim") {
            iss >> meta.dim;
        } else if (key == "kT") {
            iss >> meta.kT;
        } else if (key == "kappa") {
            meta.kappa.resize(meta.dim);
            for (int i = 0; i < meta.dim; i++) iss >> meta.kappa[i];
        } else if (key == "periodic") {
            meta.periodic.resize(meta.dim);
            for (int i = 0; i < meta.dim; i++) {
                int v; iss >> v;
                meta.periodic[i] = (v != 0);
            }
        } else if (key == "domMin") {
            meta.domMin.resize(meta.dim);
            for (int i = 0; i < meta.dim; i++) iss >> meta.domMin[i];
        } else if (key == "domMax") {
            meta.domMax.resize(meta.dim);
            for (int i = 0; i < meta.dim; i++) iss >> meta.domMax[i];
        } else if (key == "sigma0") {
            meta.sigma0.resize(meta.dim);
            for (int i = 0; i < meta.dim; i++) iss >> meta.sigma0[i];
        } else if (key == "nkernels") {
            // informational; we count from data lines
        } else {
            // Data line: Nk center[0..d-1] mu[0..d-1] sigma[0..d-1]
            if (meta.dim == 0) continue;
            std::vector<double> vals;
            // key is already the first number
            vals.push_back(std::atof(key.c_str()));
            double v;
            while (iss >> v) vals.push_back(v);
            if ((int)vals.size() != 1 + 3 * meta.dim) continue;

            Kernel k;
            k.Nk = vals[0];
            k.center.resize(meta.dim);
            k.mu.resize(meta.dim);
            k.sigma.resize(meta.dim);
            for (int i = 0; i < meta.dim; i++) {
                k.center[i] = vals[1 + i];
                k.mu[i]     = vals[1 + meta.dim + i];
                k.sigma[i]  = vals[1 + 2*meta.dim + i];
            }
            kernels.push_back(k);
        }
    }

    if (kernels.empty()) {
        fprintf(stderr, "ERROR: no kernel data found in %s\n", path);
        return false;
    }
    if (meta.kT <= 0) {
        fprintf(stderr, "ERROR: kT not found or non-positive in %s\n", path);
        return false;
    }
    return true;
}

// ─────────────────────── grid helpers ───────────────────────────────────────

static inline double periodic_delta(double delta, double period) {
    return delta - period * std::round(delta / period);
}

// ─────────────────────── CZAR gradient evaluation ──────────────────────────

void czar_on_grid(
    const Meta &meta,
    const std::vector<Kernel> &kernels,
    int grid_pts,
    double nsigma,
    // outputs (pre-allocated):
    std::vector<double> &ptilde,         // [gridTotal]
    std::vector<double> &czar_grad,      // [gridTotal * dim]
    // grid info (filled by this function):
    std::vector<int> &sizes,
    std::vector<double> &gmin,
    std::vector<double> &gmax,
    std::vector<double> &dx,
    std::vector<bool> &allowed_out,
    double minpop,
    bool verbose)
{
    const int dim = meta.dim;
    sizes.assign(dim, grid_pts);
    gmin = meta.domMin;
    gmax = meta.domMax;
    dx.resize(dim);

    int gridTotal = 1;
    for (int d = 0; d < dim; d++) {
        if (meta.periodic[d])
            dx[d] = (gmax[d] - gmin[d]) / sizes[d];
        else
            dx[d] = (gmax[d] - gmin[d]) / (sizes[d] - 1);
        gridTotal *= sizes[d];
    }

    // Period array
    std::vector<double> period(dim, 0.0);
    for (int d = 0; d < dim; d++)
        if (meta.periodic[d]) period[d] = gmax[d] - gmin[d];

    // Allocate
    ptilde.assign(gridTotal, 0.0);
    std::vector<double> sum_wkmu(gridTotal * dim, 0.0);
    czar_grad.assign(gridTotal * dim, 0.0);

    // Strides for flat indexing
    std::vector<int> strides(dim);
    strides[dim-1] = 1;
    for (int d = dim-2; d >= 0; d--) strides[d] = strides[d+1] * sizes[d+1];

    // Helper: grid coordinate
    auto grid_coord = [&](int d, int i) -> double {
        return gmin[d] + i * dx[d];
    };

    // ── Kernel loop: accumulate ptilde and NW numerator ─────────────────────
    int nk = (int)kernels.size();
    int report_every = std::max(1, nk / 10);

    for (int ki = 0; ki < nk; ki++) {
        if (verbose && ki % report_every == 0)
            printf("  Processing kernel %d/%d ...\n", ki+1, nk);

        const Kernel &kern = kernels[ki];

        // KDE normalization: alpha_k = prod(sigma0/sigma_k) ensures the NW
        // denominator is proportional to a properly normalized variable-bandwidth
        // KDE. When sigma0 is unavailable (old files), alpha defaults to 1.0.
        double alpha_k = 1.0;
        if ((int)meta.sigma0.size() == dim) {
            for (int d = 0; d < dim; d++)
                alpha_k *= meta.sigma0[d] / (kern.sigma[d] + 1e-300);
        }
        const double Nk_eff = kern.Nk * alpha_k;

        // Per-dimension index range (box cutoff)
        std::vector<int> lo_idx(dim), hi_idx(dim);
        bool skip = false;
        for (int d = 0; d < dim; d++) {
            if (meta.periodic[d]) {
                lo_idx[d] = 0;
                hi_idx[d] = sizes[d] - 1;
            } else {
                double hw = 2.0 * nsigma * kern.sigma[d];
                lo_idx[d] = std::max(0, (int)std::floor((kern.center[d] - hw - gmin[d]) / dx[d]));
                hi_idx[d] = std::min(sizes[d]-1, (int)std::ceil((kern.center[d] + hw - gmin[d]) / dx[d]));
                if (lo_idx[d] > hi_idx[d]) { skip = true; break; }
            }
        }
        if (skip) continue;

        // Iterate over the box
        // For simplicity, support 1D, 2D, 3D explicitly
        std::vector<int> idx(dim);
        for (idx[0] = lo_idx[0]; idx[0] <= hi_idx[0]; idx[0]++) {
            int id0 = meta.periodic[0] ? ((idx[0] % sizes[0]) + sizes[0]) % sizes[0] : idx[0];
            double z0 = grid_coord(0, id0);
            double dz0 = z0 - kern.center[0];
            if (meta.periodic[0] && period[0] > 0) dz0 = periodic_delta(dz0, period[0]);
            double inv4s0 = 1.0 / (4.0 * kern.sigma[0] * kern.sigma[0] + 1e-300);
            double e0 = dz0 * dz0 * inv4s0;
            if (e0 >= nsigma * nsigma) continue;

            if (dim == 1) {
                double Gk = std::exp(-e0);
                if (Gk < 1e-300) continue;
                double wk = Nk_eff * Gk;
                int g = id0;
                ptilde[g] += wk;
                sum_wkmu[g * dim + 0] += wk * kern.mu[0];
            } else {
                for (idx[1] = lo_idx[1]; idx[1] <= hi_idx[1]; idx[1]++) {
                    int id1 = meta.periodic[1] ? ((idx[1] % sizes[1]) + sizes[1]) % sizes[1] : idx[1];
                    double z1 = grid_coord(1, id1);
                    double dz1 = z1 - kern.center[1];
                    if (meta.periodic[1] && period[1] > 0) dz1 = periodic_delta(dz1, period[1]);
                    double inv4s1 = 1.0 / (4.0 * kern.sigma[1] * kern.sigma[1] + 1e-300);
                    double e1 = dz1 * dz1 * inv4s1;
                    if (e1 >= nsigma * nsigma) continue;

                    if (dim == 2) {
                        double Gk = std::exp(-(e0 + e1));
                        if (Gk < 1e-300) continue;
                        double wk = Nk_eff * Gk;
                        int g = id0 * sizes[1] + id1;
                        ptilde[g] += wk;
                        sum_wkmu[g * dim + 0] += wk * kern.mu[0];
                        sum_wkmu[g * dim + 1] += wk * kern.mu[1];
                    } else {
                        // dim == 3
                        for (idx[2] = lo_idx[2]; idx[2] <= hi_idx[2]; idx[2]++) {
                            int id2 = meta.periodic[2] ? ((idx[2] % sizes[2]) + sizes[2]) % sizes[2] : idx[2];
                            double z2 = grid_coord(2, id2);
                            double dz2 = z2 - kern.center[2];
                            if (meta.periodic[2] && period[2] > 0) dz2 = periodic_delta(dz2, period[2]);
                            double inv4s2 = 1.0 / (4.0 * kern.sigma[2] * kern.sigma[2] + 1e-300);
                            double e2 = dz2 * dz2 * inv4s2;
                            if (e2 >= nsigma * nsigma) continue;

                            double Gk = std::exp(-(e0 + e1 + e2));
                            if (Gk < 1e-300) continue;
                            double wk = Nk_eff * Gk;
                            int g = (id0 * sizes[1] + id1) * sizes[2] + id2;
                            ptilde[g] += wk;
                            sum_wkmu[g * dim + 0] += wk * kern.mu[0];
                            sum_wkmu[g * dim + 1] += wk * kern.mu[1];
                            sum_wkmu[g * dim + 2] += wk * kern.mu[2];
                        }
                    }
                }
            }
        }
    }

    // ── NW mean force ───────────────────────────────────────────────────────
    std::vector<double> mu_NW(gridTotal * dim, 0.0);
    for (int g = 0; g < gridTotal; g++) {
        if (ptilde[g] > 0) {
            for (int d = 0; d < dim; d++)
                mu_NW[g * dim + d] = sum_wkmu[g * dim + d] / ptilde[g];
        }
    }

    // ── FD gradient of ln(ptilde) ───────────────────────────────────────────
    // Matches DRR's getCountsLogDerivative: FD of ln(count)
    double pmax = *std::max_element(ptilde.begin(), ptilde.end());
    double pfloor = std::max(pmax * 1e-18, 1e-300);
    std::vector<double> ln_p(gridTotal);
    for (int g = 0; g < gridTotal; g++)
        ln_p[g] = std::log(std::max(ptilde[g], pfloor));

    std::vector<double> grad_ln_p(gridTotal * dim, 0.0);

    // Compute FD per dimension using strides
    for (int d = 0; d < dim; d++) {
        int stride_d = strides[d];
        int N_d = sizes[d];
        double h = dx[d];

        for (int g = 0; g < gridTotal; g++) {
            if (ptilde[g] <= 0) continue;

            // Compute index along dimension d
            int idx_d = (g / stride_d) % N_d;

            if (meta.periodic[d]) {
                int g_next = g + ((idx_d + 1 < N_d) ? stride_d : -(N_d - 1) * stride_d);
                int g_prev = g - ((idx_d > 0) ? stride_d : -(N_d - 1) * stride_d);
                grad_ln_p[g * dim + d] = (ln_p[g_next] - ln_p[g_prev]) / (2.0 * h);
            } else if (idx_d == 0) {
                if (N_d >= 3) {
                    // 2nd-order forward: (-3f0 + 4f1 - f2) / (2h)
                    grad_ln_p[g * dim + d] = (-3.0 * ln_p[g] + 4.0 * ln_p[g + stride_d]
                                              - ln_p[g + 2 * stride_d]) / (2.0 * h);
                } else {
                    grad_ln_p[g * dim + d] = (ln_p[g + stride_d] - ln_p[g]) / h;
                }
            } else if (idx_d == N_d - 1) {
                if (N_d >= 3) {
                    // 2nd-order backward: (3fN - 4fN1 + fN2) / (2h)
                    grad_ln_p[g * dim + d] = (3.0 * ln_p[g] - 4.0 * ln_p[g - stride_d]
                                              + ln_p[g - 2 * stride_d]) / (2.0 * h);
                } else {
                    grad_ln_p[g * dim + d] = (ln_p[g] - ln_p[g - stride_d]) / h;
                }
            } else {
                // Central difference
                grad_ln_p[g * dim + d] = (ln_p[g + stride_d] - ln_p[g - stride_d]) / (2.0 * h);
            }
        }
    }

    // ── Assemble CZAR gradient ──────────────────────────────────────────────
    // dA/dz_i = -mu_NW_i - kT * d/dz_i ln(ptilde)
    // (confirmed from DRR CZAR::getGradient)
    for (int g = 0; g < gridTotal; g++) {
        for (int d = 0; d < dim; d++) {
            czar_grad[g * dim + d] = -mu_NW[g * dim + d] - meta.kT * grad_ln_p[g * dim + d];
        }
    }

    // ── Build allowed mask ──────────────────────────────────────────────────
    double pop_thresh = minpop * pmax;
    allowed_out.resize(gridTotal);
    for (int g = 0; g < gridTotal; g++)
        allowed_out[g] = (ptilde[g] >= pop_thresh);

    if (verbose) {
        int n_pop = 0;
        for (int g = 0; g < gridTotal; g++) if (ptilde[g] > 0) n_pop++;
        int n_allowed = 0;
        for (int g = 0; g < gridTotal; g++) if (allowed_out[g]) n_allowed++;
        printf("  Grid: %d total, %d populated (%.1f%%), %d allowed (%.1f%%)\n",
               gridTotal, n_pop, 100.0 * n_pop / gridTotal,
               n_allowed, 100.0 * n_allowed / gridTotal);
    }
}

// ─────────────────────── MC integration (from abf_integrate) ───────────────

struct MCResult {
    unsigned int steps;
    unsigned long proposals;
    double acceptance;
    double rmsd;
};

MCResult mc_integrate(
    const std::vector<double> &grad,   // [gridTotal * dim]
    const std::vector<bool> &allowed,
    const std::vector<int> &sizes,
    const std::vector<double> &dx,
    const std::vector<bool> &periodic,
    double kT,
    unsigned int nsteps_in,
    double hill_init,
    double hill_factor,
    bool verbose,
    // output:
    std::vector<double> &A)
{
    const int dim = (int)sizes.size();
    int gridTotal = 1;
    for (int d = 0; d < dim; d++) gridTotal *= sizes[d];

    // Strides
    std::vector<int> strides(dim);
    strides[dim-1] = 1;
    for (int d = dim-2; d >= 0; d--) strides[d] = strides[d+1] * sizes[d+1];

    double mbeta = -1.0 / kT;
    double hill = hill_init;
    double hill_min = 0.0005;
    double convergence_limit = -0.001;

    std::vector<double> bias(gridTotal, 0.0);
    std::vector<unsigned long> histogram(gridTotal, 0);

    // Build allowed list
    std::vector<int> allowed_list;
    for (int g = 0; g < gridTotal; g++)
        if (allowed[g]) allowed_list.push_back(g);

    if (allowed_list.empty()) {
        fprintf(stderr, "WARNING: no allowed grid points\n");
        A.assign(gridTotal, 0.0);
        return {0, 0, 0.0, 0.0};
    }

    srand(time(NULL));

    // Flat to multi-index
    auto flat_to_multi = [&](int off, std::vector<int> &idx) {
        for (int d = dim-1; d >= 0; d--) {
            idx[d] = off % sizes[d];
            off /= sizes[d];
        }
    };
    auto multi_to_flat = [&](const std::vector<int> &idx) -> int {
        int off = 0;
        for (int d = 0; d < dim; d++) off = off * sizes[d] + idx[d];
        return off;
    };

    // Pick random allowed starting position
    std::vector<int> pos(dim);
    {
        int start = allowed_list[rand() % allowed_list.size()];
        flat_to_multi(start, pos);
    }

    // Convergence settings
    unsigned int nsteps = nsteps_in;
    unsigned int out_freq, scale_hill_step, hill_interval, min_steps;
    if (nsteps > 0) {
        // Fixed mode: run exactly nsteps
        out_freq = std::max(1u, nsteps / 10);
        scale_hill_step = nsteps / 2;
        hill_interval = out_freq;
        min_steps = nsteps;  // no early convergence in fixed mode
    } else {
        // Auto-converge mode:
        //   - Check RMSD every out_freq steps
        //   - Start scaling hill after scale_hill_step (separate interval)
        //   - Allow convergence after min_steps
        //   - Hard cap at max_steps to prevent infinite loops
        out_freq = std::max(100000u, (unsigned int)(10 * gridTotal));
        scale_hill_step = std::max(200000u, (unsigned int)(200 * gridTotal));
        hill_interval = std::max(1000000u, (unsigned int)(100 * gridTotal));
        min_steps = 2 * scale_hill_step;
        nsteps = 20 * scale_hill_step;    // hard cap
        if (verbose)
            printf("  MC auto-converge: check every %u, hill scale at %u (every %u), "
                   "min %u, cap %u\n", out_freq, scale_hill_step, hill_interval, min_steps, nsteps);
    }

    // ── Compute RMSD (deviation between input grad and FD of bias) ──────────
    auto compute_rmsd = [&]() -> double {
        double rmsd_sum = 0.0;
        unsigned int norm = 0;
        std::vector<int> p(dim), np2(dim);
        for (int g = 0; g < gridTotal; g++) {
            if (!allowed[g]) continue;
            flat_to_multi(g, p);
            for (int d = 0; d < dim; d++) {
                double est = 0.0;
                int c = 0;
                // Backward neighbor
                np2 = p;
                np2[d] = p[d] - 1;
                if (periodic[d]) {
                    np2[d] = ((np2[d] % sizes[d]) + sizes[d]) % sizes[d];
                    int noff = multi_to_flat(np2);
                    if (allowed[noff]) {
                        est += (bias[noff] - bias[g]) / dx[d];
                        c++;
                    }
                } else if (np2[d] >= 0) {
                    int noff = multi_to_flat(np2);
                    if (allowed[noff]) {
                        est += (bias[noff] - bias[g]) / dx[d];
                        c++;
                    }
                }
                // Forward neighbor
                np2 = p;
                np2[d] = p[d] + 1;
                if (periodic[d]) {
                    np2[d] = np2[d] % sizes[d];
                    int noff = multi_to_flat(np2);
                    if (allowed[noff]) {
                        est += (bias[g] - bias[noff]) / dx[d];
                        c++;
                    }
                } else if (np2[d] < sizes[d]) {
                    int noff = multi_to_flat(np2);
                    if (allowed[noff]) {
                        est += (bias[g] - bias[noff]) / dx[d];
                        c++;
                    }
                }
                if (c > 0) est /= (double)c;
                else continue;  // no valid neighbors — skip this point/dimension
                double dev = grad[g * dim + d] - est;
                rmsd_sum += dev * dev;
                norm++;
            }
        }
        return std::sqrt(rmsd_sum / std::max(norm, 1u));
    };

    double rmsd = compute_rmsd();
    if (verbose) printf("  MC initial gradient RMSD: %.6f\n", rmsd);

    // ── Main MC loop ────────────────────────────────────────────────────────
    unsigned long total = 0;
    unsigned int actual_steps = 0;
    std::vector<int> newpos(dim);

    for (unsigned int step = 1; step <= nsteps; step++) {
        actual_steps = step;

        // Hill scaling (separate interval from RMSD checks)
        if (hill_factor > 0 && step > scale_hill_step &&
            step % hill_interval == 0 && hill > hill_min)
            hill *= hill_factor;

        // RMSD convergence check
        if (step % out_freq == 0) {
            double rmsd_old = rmsd;
            rmsd = compute_rmsd();
            double rmsd_rel_change = (rmsd - rmsd_old) / (rmsd_old * (double)out_freq + 1e-300) * 1e6;

            if (verbose) {
                printf("  MC step %10u  RMSD=%.6f  dRMSD/1M=%.4f  hill=%.6f\n",
                       step, rmsd, rmsd_rel_change, hill);
                fflush(stdout);
            }

            if (rmsd_rel_change > convergence_limit && step >= min_steps) {
                if (verbose)
                    printf("  Converged at step %u (RMSD=%.6f)\n", step, rmsd);
                break;
            }
        }

        int offset = multi_to_flat(pos);
        histogram[offset]++;
        bias[offset] += hill;

        const double *grad_here = &grad[offset * dim];

        // Propose move
        int not_accepted = 1;
        while (not_accepted) {
            total++;
            double dA = 0.0;
            for (int d = 0; d < dim; d++) {
                int dp = (rand() % 3) - 1;
                int candidate = pos[d] + dp;
                newpos[d] = pos[d];

                if (periodic[d]) {
                    candidate = ((candidate % sizes[d]) + sizes[d]) % sizes[d];
                    newpos[d] = candidate;
                } else {
                    if (candidate < 0 || candidate >= sizes[d]) {
                        dp = 0;
                    } else {
                        newpos[d] = candidate;
                    }
                }
                if (newpos[d] == pos[d]) dp = 0;
                if (dp != 0)
                    dA += grad_here[d] * dp * dx[d];
            }

            int newoffset = multi_to_flat(newpos);
            dA += bias[newoffset] - bias[offset];

            if (allowed[newoffset] &&
                ((double)rand() / RAND_MAX) < std::exp(mbeta * dA)) {
                pos = newpos;
                not_accepted = 0;
            }
        }
    }

    double acceptance = (double)actual_steps / (double)std::max(total, 1UL);
    double final_rmsd = compute_rmsd();

    // PMF = -bias (bias fills wells, converges to -A + const)
    A.resize(gridTotal);
    double amin = 1e30;
    for (int g = 0; g < gridTotal; g++) {
        A[g] = -bias[g];
        if (allowed[g] && A[g] < amin) amin = A[g];
    }
    for (int g = 0; g < gridTotal; g++) A[g] -= amin;

    return {actual_steps, total, acceptance, final_rmsd};
}

// ─────────────────────── 1D trapezoidal integration ──────────────────────

void trapz_integrate_1d(
    const std::vector<double> &grad,   // [N] CZAR gradient
    const std::vector<bool> &allowed,
    int N,
    double dx,
    // output:
    std::vector<double> &A)
{
    A.assign(N, 0.0);

    // Cumulative trapezoidal rule
    for (int i = 1; i < N; i++) {
        double g_prev = allowed[i-1] ? grad[i-1] : 0.0;
        double g_curr = allowed[i]   ? grad[i]   : 0.0;
        A[i] = A[i-1] + 0.5 * (g_prev + g_curr) * dx;
    }

    // Shift so min over allowed region = 0
    double amin = 1e30;
    for (int i = 0; i < N; i++)
        if (allowed[i] && A[i] < amin) amin = A[i];
    for (int i = 0; i < N; i++) A[i] -= amin;
}

// ─────────────────────── output writer ──────────────────────────────────────

void write_output(const char *path,
                  const Meta &meta,
                  const std::vector<int> &sizes,
                  const std::vector<double> &gmin,
                  const std::vector<double> &dx,
                  const std::vector<double> &ptilde,
                  const std::vector<double> &czar_grad,
                  const std::vector<double> &A,
                  const std::vector<bool> &allowed)
{
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s for writing\n", path); return; }

    int dim = meta.dim;
    int gridTotal = 1;
    for (int d = 0; d < dim; d++) gridTotal *= sizes[d];

    fprintf(f, "#");
    for (int d = 0; d < dim; d++) fprintf(f, " z%d", d);
    for (int d = 0; d < dim; d++) fprintf(f, " czar_grad%d", d);
    fprintf(f, " ptilde A_czar[kJ/mol]\n");
    fprintf(f, "# kT = %.6f kJ/mol\n", meta.kT);

    std::vector<int> idx(dim);
    for (int g = 0; g < gridTotal; g++) {
        // Compute multi-index
        int tmp = g;
        for (int d = dim-1; d >= 0; d--) {
            idx[d] = tmp % sizes[d];
            tmp /= sizes[d];
        }

        // Coordinates
        for (int d = 0; d < dim; d++)
            fprintf(f, " %.8f", gmin[d] + idx[d] * dx[d]);
        // Gradient
        for (int d = 0; d < dim; d++)
            fprintf(f, " %.8f", czar_grad[g * dim + d]);
        // Density
        fprintf(f, " %.8f", ptilde[g]);
        // PMF
        if (allowed[g])
            fprintf(f, " %.8f", A[g]);
        else
            fprintf(f, " nan");
        fprintf(f, "\n");

        // Blank line between z0 slices (for gnuplot pm3d)
        if (dim >= 2 && idx[dim-1] == sizes[dim-1] - 1)
            fprintf(f, "\n");
    }
    fclose(f);
}

// ─────────────────── simple FEL writer (z0 z1 ... A) ────────────────────────

void write_simple_fel(const char *path,
                      int dim,
                      const std::vector<int> &sizes,
                      const std::vector<double> &gmin,
                      const std::vector<double> &dx,
                      const std::vector<double> &A,
                      const std::vector<bool> &allowed)
{
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s for writing\n", path); return; }

    int gridTotal = 1;
    for (int d = 0; d < dim; d++) gridTotal *= sizes[d];

    // Shift so minimum over allowed region = 0
    double amin = 1e30;
    for (int g = 0; g < gridTotal; g++)
        if (allowed[g] && A[g] < amin) amin = A[g];

    fprintf(f, "#");
    for (int d = 0; d < dim; d++) fprintf(f, " z%d", d);
    fprintf(f, " A(kJ/mol)\n");

    std::vector<int> idx(dim);
    for (int g = 0; g < gridTotal; g++) {
        int tmp = g;
        for (int d = dim-1; d >= 0; d--) {
            idx[d] = tmp % sizes[d];
            tmp /= sizes[d];
        }
        for (int d = 0; d < dim; d++)
            fprintf(f, " %.8f", gmin[d] + idx[d] * dx[d]);
        if (allowed[g])
            fprintf(f, " %.6f", A[g] - amin);
        else
            fprintf(f, " nan");
        fprintf(f, "\n");

        if (dim >= 2 && idx[dim-1] == sizes[dim-1] - 1)
            fprintf(f, "\n");
    }
    fclose(f);
}

// ─────────────────── batch processing ───────────────────────────────────────

// Extract step number from *_NNNNN...N.dat filename
static long extract_step_from_path(const std::string &path) {
    size_t dot = path.rfind(".dat");
    if (dot == std::string::npos || dot == 0) return -1;
    // Walk backward from just before ".dat" collecting digits
    size_t end = dot;
    size_t start = end;
    while (start > 0 && isdigit(path[start - 1])) --start;
    if (start == end) return -1;  // no digits found
    // Check that the character before the digits is '_'
    if (start == 0 || path[start - 1] != '_') return -1;
    return std::atol(path.substr(start, end - start).c_str());
}

// Scan a directory for *czar_kernels_NNNNN...N.dat files (any label prefix, any digit count)
static std::vector<std::string> scan_for_czar_files(const std::string &dir) {
    std::vector<std::string> results;
    DIR *dp = opendir(dir.c_str());
    if (!dp) {
        fprintf(stderr, "ERROR: cannot open directory '%s'\n", dir.c_str());
        return results;
    }

    struct dirent *entry;
    while ((entry = readdir(dp)) != NULL) {
        std::string fname(entry->d_name);
        // Match: *czar_kernels_<digits>.dat
        size_t pos = fname.find("czar_kernels_");
        if (pos == std::string::npos) continue;
        size_t dstart = pos + 13; // after "czar_kernels_"
        if (fname.size() < dstart + 5) continue; // need at least 1 digit + ".dat"
        if (fname.substr(fname.size() - 4) != ".dat") continue;
        // Everything between "czar_kernels_" and ".dat" must be digits
        std::string digits = fname.substr(dstart, fname.size() - 4 - dstart);
        if (digits.empty()) continue;
        bool all_digits = true;
        for (size_t i = 0; i < digits.size(); i++)
            if (!isdigit(digits[i])) { all_digits = false; break; }
        if (!all_digits) continue;
        results.push_back(dir + "/" + fname);
    }
    closedir(dp);

    // Sort by step number (numeric), not lexicographic.
    // Lexicographic sort puts "1000000" before "600000" because '1' < '6'.
    std::sort(results.begin(), results.end(),
        [](const std::string &a, const std::string &b) {
            return extract_step_from_path(a) < extract_step_from_path(b);
        });
    return results;
}

void process_batch(const std::string &scan_dir,
                   const std::string &out_dir,
                   int grid_pts, double nsigma, double minpop,
                   unsigned int mc_steps, double mc_hill,
                   double mc_hill_factor, bool verbose,
                   long start_step)
{
    printf("Scanning '%s' for *czar_kernels_*.dat ...\n", scan_dir.c_str());
    std::vector<std::string> snapshots = scan_for_czar_files(scan_dir);

    if (snapshots.empty()) {
        printf("  No czar_kernels snapshot files found.\n");
        return;
    }
    printf("  Found %d snapshots (sorted by step number)\n", (int)snapshots.size());

    // Filter by start_step
    if (start_step > 0) {
        std::vector<std::string> filtered;
        for (const auto &f : snapshots) {
            if (extract_step_from_path(f) >= start_step)
                filtered.push_back(f);
        }
        printf("  After --start %ld filter: %d snapshots\n",
               start_step, (int)filtered.size());
        snapshots = std::move(filtered);
    }

    if (snapshots.empty()) {
        printf("  No snapshots remaining after filter.\n");
        return;
    }

    // Create output directory
    mkdir(out_dir.c_str(), 0755);
    printf("  Output -> %s/\n\n", out_dir.c_str());

    int written = 0;
    for (size_t si = 0; si < snapshots.size(); si++) {
        const std::string &fpath = snapshots[si];
        long step = extract_step_from_path(fpath);

        Meta meta;
        std::vector<Kernel> kernels;
        if (!parse_czar_file(fpath.c_str(), meta, kernels)) {
            printf("  Skipping %s: parse error\n", fpath.c_str());
            continue;
        }
        if (kernels.empty()) continue;

        printf("  [%d/%d] step %ld  (%d kernels, %dD) ... ",
               (int)(si+1), (int)snapshots.size(), step,
               (int)kernels.size(), meta.dim);
        fflush(stdout);

        // Evaluate CZAR gradient
        std::vector<double> ptilde, czar_grad;
        std::vector<int> sizes;
        std::vector<double> gmin, gmax, dx;
        std::vector<bool> allowed;
        czar_on_grid(meta, kernels, grid_pts, nsigma,
                     ptilde, czar_grad, sizes, gmin, gmax, dx, allowed, minpop, verbose);

        // Integrate: trapezoidal for 1D, MC for 2D+
        std::vector<double> A;
        char outname[512];
        snprintf(outname, sizeof(outname), "%s/FEL_%08ld.dat", out_dir.c_str(), step);

        if (meta.dim == 1) {
            trapz_integrate_1d(czar_grad, allowed, sizes[0], dx[0], A);
            write_simple_fel(outname, meta.dim, sizes, gmin, dx, A, allowed);
            printf("trapz -> FEL_%08ld.dat\n", step);
        } else {
            MCResult mc = mc_integrate(czar_grad, allowed, sizes, dx, meta.periodic, meta.kT,
                         mc_steps, mc_hill, mc_hill_factor, verbose, A);
            write_simple_fel(outname, meta.dim, sizes, gmin, dx, A, allowed);
            printf("MC %uk %.0f%% RMSD=%.4f -> FEL_%08ld.dat\n",
                   mc.steps / 1000, 100.0 * mc.acceptance, mc.rmsd, step);
        }
        written++;
    }

    printf("\nDone. %d FEL files written to %s/\n", written, out_dir.c_str());
}

// ─────────────────────── help ──────────────────────────────────────────────

static void print_help(const char *prog) {
    printf("czar_integrate — integration of CZAR kernel gradient field\n");
    printf("                  (FK-eABF companion, abf_integrate conventions)\n");
    printf("                  1D: trapezoidal rule; 2D+: MC integration\n\n");
    printf("Usage:\n");
    printf("  %s <output_dir>             Scan CWD for *czar_kernels_*.dat, write FELs\n", prog);
    printf("  %s <output_dir> [options]   Same, with options\n", prog);
    printf("  %s -i <file> [options]      Process a single kernel file\n\n", prog);
    printf("Options:\n");
    printf("  -n <steps>      MC integration steps (0 = auto-converge) [0]\n");
    printf("  -h <height>     Initial hill height                      [0.01]\n");
    printf("  -f <factor>     Hill reduction factor                    [0.5]\n");
    printf("  -t <kT>         Override kT from file (kJ/mol)\n");
    printf("  -g <pts>        Grid points per dimension                [100]\n");
    printf("  -s <nsigma>     Kernel cutoff in sigma units             [4.0]\n");
    printf("  -m <minpop>     Min density fraction for allowed region  [1e-3]\n");
    printf("  -d <dir>        Directory to scan (default: current)     [.]\n");
    printf("  -S <step>       Start from this step number (skip earlier) [0]\n");
    printf("  -i <file>       Single-file mode (skip auto-scan)\n");
    printf("  -o <file>       Output file for single-file mode         [FEL_czar.dat]\n");
    printf("  -v              Verbose output\n\n");
    printf("Examples:\n");
    printf("  %s FEL_snapshots                     # scan ., write to FEL_snapshots/\n", prog);
    printf("  %s FEL_snapshots -n 5000000          # fixed MC steps\n", prog);
    printf("  %s FEL_snapshots -d /path/to/run     # scan another directory\n", prog);
    printf("  %s FEL_snapshots -S 5000000          # skip snapshots before step 5M\n", prog);
    printf("  %s -i fk.czar_kernels_10000000.dat -o PMF.dat  # single file\n", prog);
    printf("\nCompile: g++ -O2 -o czar_integrate czar_integrate.cpp -lm\n");
}

// ─────────────────────── main ──────────────────────────────────────────────

int main(int argc, char *argv[]) {
    // Defaults
    int grid_pts = 100;
    double nsigma = 4.0;
    unsigned int mc_steps = 0;
    double mc_hill = 0.01;
    double mc_hill_factor = 0.5;
    double minpop = 1e-3;
    double kT_override = 0.0;
    const char *output = "FEL_czar.dat";
    const char *single_input = NULL;
    const char *scan_dir = ".";
    const char *out_dir = NULL;
    bool verbose = false;
    long start_step = 0;

    if (argc < 2) {
        print_help(argv[0]);
        return 0;
    }

    // Parse command line
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-' && argv[i][1] != '\0') {
            switch (argv[i][1]) {
            case 'n': mc_steps = (unsigned int)atoi(argv[++i]); break;
            case 'h': mc_hill = atof(argv[++i]); break;
            case 'f': mc_hill_factor = atof(argv[++i]); break;
            case 't': kT_override = atof(argv[++i]); break;
            case 'g': grid_pts = atoi(argv[++i]); break;
            case 's': nsigma = atof(argv[++i]); break;
            case 'm': minpop = atof(argv[++i]); break;
            case 'i': single_input = argv[++i]; break;
            case 'o': output = argv[++i]; break;
            case 'd': scan_dir = argv[++i]; break;
            case 'S': start_step = atol(argv[++i]); break;
            case 'v': verbose = true; break;
            default:
                fprintf(stderr, "Unknown option: %s\n", argv[i]);
                return 1;
            }
        } else {
            // Positional argument = output directory
            out_dir = argv[i];
        }
    }

    // ── Single file mode ────────────────────────────────────────────────────
    if (single_input) {
        Meta meta;
        std::vector<Kernel> kernels;
        printf("Reading CZAR kernels from: %s\n", single_input);
        if (!parse_czar_file(single_input, meta, kernels)) return 1;

        if (kT_override > 0) meta.kT = kT_override;

        printf("  dim=%d  kT=%.5f kJ/mol  kernels=%d\n", meta.dim, meta.kT, (int)kernels.size());
        printf("  periodic: ");
        for (int d = 0; d < meta.dim; d++) printf("%s%s", d?",":"", meta.periodic[d]?"true":"false");
        printf("\n");
        if ((int)meta.sigma0.size() == meta.dim) {
            printf("  sigma0: ");
            for (int d = 0; d < meta.dim; d++) printf("%s%.6f", d?",":"", meta.sigma0[d]);
            printf("  (KDE normalization active)\n");
        } else {
            printf("  sigma0: not found in file (old format); KDE normalization disabled\n");
        }

        std::vector<double> ptilde, czar_grad;
        std::vector<int> sizes;
        std::vector<double> gmin, gmax, dx;
        std::vector<bool> allowed;

        printf("Evaluating CZAR gradient on %d^%d grid ...\n", grid_pts, meta.dim);
        czar_on_grid(meta, kernels, grid_pts, nsigma,
                     ptilde, czar_grad, sizes, gmin, gmax, dx, allowed, minpop, verbose);

        std::vector<double> A;
        if (meta.dim == 1) {
            printf("Integrating via trapezoidal rule (1D) ...\n");
            trapz_integrate_1d(czar_grad, allowed, sizes[0], dx[0], A);
            printf("  Trapezoidal integration complete.\n");
        } else {
            printf("Integrating via MC (%dD) ...\n", meta.dim);
            MCResult mc = mc_integrate(czar_grad, allowed, sizes, dx, meta.periodic, meta.kT,
                         mc_steps, mc_hill, mc_hill_factor, verbose, A);
            printf("  MC: %u steps, %.1f%% acceptance, RMSD=%.6f\n",
                   mc.steps, 100.0 * mc.acceptance, mc.rmsd);
        }

        printf("Writing FEL to: %s\n", output);
        write_output(output, meta, sizes, gmin, dx, ptilde, czar_grad, A, allowed);

        printf("Done.\n");
        return 0;
    }

    // ── Batch mode (default) ────────────────────────────────────────────────
    if (!out_dir) {
        fprintf(stderr, "ERROR: specify an output directory.\n");
        fprintf(stderr, "  Usage: %s <output_dir> [options]\n", argv[0]);
        fprintf(stderr, "  Run '%s' with no arguments for full help.\n", argv[0]);
        return 1;
    }

    process_batch(std::string(scan_dir), std::string(out_dir),
                  grid_pts, nsigma, minpop,
                  mc_steps, mc_hill, mc_hill_factor, verbose,
                  start_step);

    return 0;
}
