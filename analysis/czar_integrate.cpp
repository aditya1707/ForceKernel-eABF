


























/*
 * czar_integrate.cpp
 *
 * Recovers the free energy landscape (FEL) from a FKERNELABF CZAR kernel file
 * by evaluating the CZAR mean-force gradient on a regular grid and then
 * integrating it via a Monte Carlo path-integration scheme.
 *
 * CZAR (Corrected z-averaged restraint) estimator:
 *   dA/dz_d = -<mu_d>_{NW}(z) - kT * d/dz_d ln p_tilde(z)
 * where <mu_d>_{NW} is the Nadaraya-Watson estimator of the mean force and
 * p_tilde is the kernel-density estimate of the biased sampling density.
 *
 * MC integration: a walker traverses the grid driven by the CZAR gradient
 * field, accumulating a history bias (WTMetaD-like) until the potential of
 * mean force converges; the negative bias is returned as A(z).
 *
 * Usage:
 *   czar_integrate <czar_kernels_file> [options]
 *   czar_integrate <czar_kernels_file> -a -d <output_dir>   (batch snapshots)
 *
 * Compile:
 *   g++ -O2 -std=c++11 -o czar_integrate czar_integrate.cpp
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



// Per-kernel data read from the CZAR file:
//   Nk     - accumulated weight (number of restraint evaluations contributing)
//   center - restraint center z_k in collective-variable space
//   mu     - kernel mean force <f/kappa> at the center
//   sigma  - kernel bandwidth (standard deviation) per CV dimension
struct Kernel {
    double Nk;
    std::vector<double> center;
    std::vector<double> mu;
    std::vector<double> sigma;
};

// Global metadata from the CZAR file header.
// kappa holds the harmonic spring constants (kJ/mol/unit^2) per dimension;
// periodic and domMin/domMax define the CV domain boundaries.
struct Meta {
    int dim;
    double kT;
    std::vector<double> kappa;
    std::vector<bool> periodic;
    std::vector<double> domMin, domMax;
};



// Parse a FKERNELABF CZAR kernel file.
// The file contains a header (dim, kT, kappa, periodic, domMin, domMax)
// followed by one data line per kernel: Nk  center[d]  mu[d]  sigma[d].
// Returns false and prints to stderr on any fatal error.
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
        } else if (key == "nkernels") {
            
        } else {
            
            if (meta.dim == 0) continue;
            std::vector<double> vals;
            
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



// Wrap delta into [-period/2, period/2) for periodic CVs.
static inline double periodic_delta(double delta, double period) {
    return delta - period * std::round(delta / period);
}



// Evaluate the CZAR gradient field and biased density on a regular grid.
//
// For each kernel k with center z_k, weight N_k, mean force mu_k, bandwidth sigma_k,
// the Gaussian contribution G_k(z) = exp(-sum_d dz_d^2 / (4 sigma_d^2)) is accumulated:
//   p_tilde(z)    += N_k * G_k(z)
//   sum_wkmu(z,d) += N_k * G_k(z) * mu_k[d]
//
// Nadaraya-Watson mean force estimate:
//   <mu_d>_NW(z) = sum_wkmu(z,d) / p_tilde(z)
//
// CZAR gradient (= negative mean force of the FEL):
//   dA/dz_d = -<mu_d>_NW(z) - kT * d ln p_tilde / dz_d
//
// The ln p_tilde gradient is computed by central finite differences
// (one-sided 2nd-order stencil at non-periodic boundaries).
//
// Output: ptilde, czar_grad, grid geometry (sizes/gmin/gmax/dx), and a
// boolean mask (allowed_out) indicating grid points above the minpop threshold.
void czar_on_grid(
    const Meta &meta,
    const std::vector<Kernel> &kernels,
    int grid_pts,
    double nsigma,
    
    std::vector<double> &ptilde,         
    std::vector<double> &czar_grad,      
    
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

    
    std::vector<double> period(dim, 0.0);
    for (int d = 0; d < dim; d++)
        if (meta.periodic[d]) period[d] = gmax[d] - gmin[d];

    
    ptilde.assign(gridTotal, 0.0);
    std::vector<double> sum_wkmu(gridTotal * dim, 0.0);
    czar_grad.assign(gridTotal * dim, 0.0);

    
    std::vector<int> strides(dim);
    strides[dim-1] = 1;
    for (int d = dim-2; d >= 0; d--) strides[d] = strides[d+1] * sizes[d+1];

    
    auto multi_to_flat = [&](const std::vector<int> &idx) -> int {
        int off = 0;
        for (int d = 0; d < dim; d++) off += idx[d] * strides[d];
        return off;
    };

    
    auto grid_coord = [&](int d, int i) -> double {
        return gmin[d] + i * dx[d];
    };

    
    int nk = (int)kernels.size();
    int report_every = std::max(1, nk / 10);

    for (int ki = 0; ki < nk; ki++) {
        if (verbose && ki % report_every == 0)
            printf("  Processing kernel %d/%d ...\n", ki+1, nk);

        const Kernel &kern = kernels[ki];

        
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
                double wk = kern.Nk * Gk;
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
                        double wk = kern.Nk * Gk;
                        int g = id0 * sizes[1] + id1;
                        ptilde[g] += wk;
                        sum_wkmu[g * dim + 0] += wk * kern.mu[0];
                        sum_wkmu[g * dim + 1] += wk * kern.mu[1];
                    } else {
                        
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
                            double wk = kern.Nk * Gk;
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

    
    std::vector<double> mu_NW(gridTotal * dim, 0.0);
    for (int g = 0; g < gridTotal; g++) {
        if (ptilde[g] > 0) {
            for (int d = 0; d < dim; d++)
                mu_NW[g * dim + d] = sum_wkmu[g * dim + d] / ptilde[g];
        }
    }

    
    
    double pmax = *std::max_element(ptilde.begin(), ptilde.end());
    double pfloor = std::max(pmax * 1e-18, 1e-300);
    std::vector<double> ln_p(gridTotal);
    for (int g = 0; g < gridTotal; g++)
        ln_p[g] = std::log(std::max(ptilde[g], pfloor));

    std::vector<double> grad_ln_p(gridTotal * dim, 0.0);

    
    for (int d = 0; d < dim; d++) {
        int stride_d = strides[d];
        int N_d = sizes[d];
        double h = dx[d];

        for (int g = 0; g < gridTotal; g++) {
            if (ptilde[g] <= 0) continue;

            
            int idx_d = (g / stride_d) % N_d;

            if (meta.periodic[d]) {
                int g_next = g + ((idx_d + 1 < N_d) ? stride_d : -(N_d - 1) * stride_d);
                int g_prev = g - ((idx_d > 0) ? stride_d : -(N_d - 1) * stride_d);
                grad_ln_p[g * dim + d] = (ln_p[g_next] - ln_p[g_prev]) / (2.0 * h);
            } else if (idx_d == 0) {
                if (N_d >= 3) {
                    
                    grad_ln_p[g * dim + d] = (-3.0 * ln_p[g] + 4.0 * ln_p[g + stride_d]
                                              - ln_p[g + 2 * stride_d]) / (2.0 * h);
                } else {
                    grad_ln_p[g * dim + d] = (ln_p[g + stride_d] - ln_p[g]) / h;
                }
            } else if (idx_d == N_d - 1) {
                if (N_d >= 3) {
                    
                    grad_ln_p[g * dim + d] = (3.0 * ln_p[g] - 4.0 * ln_p[g - stride_d]
                                              + ln_p[g - 2 * stride_d]) / (2.0 * h);
                } else {
                    grad_ln_p[g * dim + d] = (ln_p[g] - ln_p[g - stride_d]) / h;
                }
            } else {
                
                grad_ln_p[g * dim + d] = (ln_p[g + stride_d] - ln_p[g - stride_d]) / (2.0 * h);
            }
        }
    }

    
    
    
    for (int g = 0; g < gridTotal; g++) {
        for (int d = 0; d < dim; d++) {
            czar_grad[g * dim + d] = -mu_NW[g * dim + d] - meta.kT * grad_ln_p[g * dim + d];
        }
    }

    
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



// Integrate the CZAR gradient field by Monte Carlo path integration to obtain A(z).
//
// Algorithm: a walker on the allowed grid points accumulates a local history
// bias (hill height 'hill') and proposes random single-step moves accepted by
// Metropolis-Hastings with energy dA = grad . dz + delta_bias.  The hill is
// annealed by hill_factor after scale_hill_step steps.
//
// Convergence is assessed by comparing the numerical gradient of the current
// bias field with the target CZAR gradient (RMSD criterion).
//
// If nsteps_in == 0, the walk runs until the RMSD relative change falls below
// convergence_limit (auto-converge mode).
//
// Output A is shifted so that its minimum over allowed points equals zero.
void mc_integrate(
    const std::vector<double> &grad,
    const std::vector<bool> &allowed,
    const std::vector<int> &sizes,
    const std::vector<double> &dx,
    const std::vector<bool> &periodic,
    double kT,
    unsigned int nsteps_in,
    double hill_init,
    double hill_factor,
    bool verbose,
    
    std::vector<double> &A)
{
    const int dim = (int)sizes.size();
    int gridTotal = 1;
    for (int d = 0; d < dim; d++) gridTotal *= sizes[d];

    
    std::vector<int> strides(dim);
    strides[dim-1] = 1;
    for (int d = dim-2; d >= 0; d--) strides[d] = strides[d+1] * sizes[d+1];

    double mbeta = -1.0 / kT;
    double hill = hill_init;
    double hill_min = 0.0005;
    double convergence_limit = -0.001;

    std::vector<double> bias(gridTotal, 0.0);
    std::vector<unsigned long> histogram(gridTotal, 0);

    
    std::vector<int> allowed_list;
    for (int g = 0; g < gridTotal; g++)
        if (allowed[g]) allowed_list.push_back(g);

    if (allowed_list.empty()) {
        fprintf(stderr, "WARNING: no allowed grid points\n");
        A.assign(gridTotal, 0.0);
        return;
    }

    srand(time(NULL));

    
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

    
    std::vector<int> pos(dim);
    {
        int start = allowed_list[rand() % allowed_list.size()];
        flat_to_multi(start, pos);
    }

    
    unsigned int nsteps = nsteps_in;
    unsigned int out_freq, scale_hill_step;
    bool converged;
    if (nsteps > 0) {
        out_freq = std::max(1u, nsteps / 10);
        scale_hill_step = nsteps / 2;
        converged = true; 
    } else {
        out_freq = 1000000;
        scale_hill_step = 2000 * (unsigned int)gridTotal;
        nsteps = 2 * scale_hill_step;
        converged = false;
        if (verbose) printf("  MC auto-converge: min steps = %u\n", nsteps);
    }

    
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
                double dev = grad[g * dim + d] - est;
                rmsd_sum += dev * dev;
                norm++;
            }
        }
        return std::sqrt(rmsd_sum / std::max(norm, 1u));
    };

    double rmsd = compute_rmsd();
    if (verbose) printf("  MC initial gradient RMSD: %.6f\n", rmsd);

    
    unsigned long total = 0;
    std::vector<int> newpos(dim);

    for (unsigned int step = 1; step <= nsteps || !converged; step++) {

        if (step % out_freq == 0) {
            double rmsd_old = rmsd;
            rmsd = compute_rmsd();
            double rmsd_rel_change = (rmsd - rmsd_old) / (rmsd_old * (double)out_freq + 1e-300) * 1e6;

            if (verbose) {
                printf("  MC step %10u  RMSD=%.6f  rel_change/1M=%.4f",
                       step, rmsd, rmsd_rel_change);
            }

            if (hill_factor > 0 && step > scale_hill_step && hill > hill_min) {
                hill *= hill_factor;
                if (verbose) printf("  hill->%.6f", hill);
            }
            if (verbose) printf("\n");
            fflush(stdout);

            if (rmsd_rel_change > convergence_limit && step >= nsteps)
                converged = true;
        }

        int offset = multi_to_flat(pos);
        histogram[offset]++;
        bias[offset] += hill;

        const double *grad_here = &grad[offset * dim];

        
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

    double acceptance = (double)nsteps / (double)std::max(total, 1UL);
    double final_rmsd = compute_rmsd();
    printf("  MC integration: %u steps, %lu proposals, acceptance=%.3f, final RMSD=%.6f\n",
           nsteps, total, acceptance, final_rmsd);

    
    A.resize(gridTotal);
    double amin = 1e30;
    for (int g = 0; g < gridTotal; g++) {
        A[g] = -bias[g];
        if (allowed[g] && A[g] < amin) amin = A[g];
    }
    for (int g = 0; g < gridTotal; g++) A[g] -= amin;
}



// Write full diagnostic output: grid coordinates, CZAR gradient per dimension,
// biased density (ptilde), and free energy A (NaN for unsampled points).
// A blank line is inserted between rows of the innermost grid dimension
// (gnuplot pm3d format).
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
        
        int tmp = g;
        for (int d = dim-1; d >= 0; d--) {
            idx[d] = tmp % sizes[d];
            tmp /= sizes[d];
        }

        
        for (int d = 0; d < dim; d++)
            fprintf(f, " %.8f", gmin[d] + idx[d] * dx[d]);
        
        for (int d = 0; d < dim; d++)
            fprintf(f, " %.8f", czar_grad[g * dim + d]);
        
        fprintf(f, " %.8f", ptilde[g]);
        
        if (allowed[g])
            fprintf(f, " %.8f", A[g]);
        else
            fprintf(f, " nan");
        fprintf(f, "\n");

        
        if (dim >= 2 && idx[dim-1] == sizes[dim-1] - 1)
            fprintf(f, "\n");
    }
    fclose(f);
}



// Write a minimal FEL file containing only grid coordinates and A(z) (in
// kJ/mol, shifted to zero minimum).  Used for convergence snapshot output.
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




// Split a filesystem path into directory and basename components.
static void split_path(const std::string &path, std::string &dir, std::string &base) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        dir = ".";
        base = path;
    } else {
        dir = path.substr(0, pos);
        base = path.substr(pos + 1);
    }
}



// Extract the stem and extension from a basename, stripping any trailing
// _XXXXXXXX (8-digit) step counter so snapshot files share the same stem as
// their reference file.
static void parse_stem_ext(const std::string &base, std::string &stem, std::string &ext) {
    
    size_t dot = base.rfind('.');
    if (dot == std::string::npos) {
        stem = base; ext = "";
    } else {
        ext = base.substr(dot);
        std::string noext = base.substr(0, dot);
        
        if (noext.size() >= 9 && noext[noext.size()-9] == '_') {
            bool all_digits = true;
            for (int i = 0; i < 8; i++)
                if (!isdigit(noext[noext.size()-8+i])) all_digits = false;
            if (all_digits)
                noext = noext.substr(0, noext.size() - 9);
        }
        stem = noext;
    }
}


// Enumerate all snapshot files in 'dir' matching stem_XXXXXXXX.ext,
// sorted lexicographically (which is chronological for zero-padded step numbers).
static std::vector<std::string> find_snapshots(const std::string &dir,
                                                const std::string &stem,
                                                const std::string &ext)
{
    std::vector<std::string> results;
    DIR *dp = opendir(dir.c_str());
    if (!dp) { fprintf(stderr, "ERROR: cannot open directory %s\n", dir.c_str()); return results; }

    std::string prefix = stem + "_";
    size_t expect_len = prefix.size() + 8 + ext.size();

    struct dirent *entry;
    while ((entry = readdir(dp)) != NULL) {
        std::string fname(entry->d_name);
        if (fname.size() != expect_len) continue;
        if (fname.substr(0, prefix.size()) != prefix) continue;
        if (fname.substr(fname.size() - ext.size()) != ext) continue;
        
        bool ok = true;
        for (int i = 0; i < 8; i++)
            if (!isdigit(fname[prefix.size() + i])) { ok = false; break; }
        if (!ok) continue;
        results.push_back(dir + "/" + fname);
    }
    closedir(dp);
    std::sort(results.begin(), results.end());
    return results;
}


// Extract the 8-digit step number from a snapshot filename path.
static long extract_step(const std::string &path, const std::string &ext) {
    
    size_t epos = path.size() - ext.size();
    if (epos < 9) return -1;
    std::string digits = path.substr(epos - 8, 8);
    return std::atol(digits.c_str());
}



// Process every snapshot file found alongside 'reference_path'.
// For each snapshot: parse kernels, evaluate CZAR grid, MC-integrate, and write
// a FEL_XXXXXXXX.dat file to fel_dir (or the same directory as the input).
void process_all_snapshots(const char *reference_path,
                           int grid_pts, double nsigma, double minpop,
                           unsigned int mc_steps, double mc_hill,
                           double mc_hill_factor,
                           const char *fel_dir, bool verbose)
{
    std::string ref(reference_path);
    std::string dir, base, stem, ext;
    split_path(ref, dir, base);
    parse_stem_ext(base, stem, ext);

    printf("Scanning for snapshots: %s/%s_????????%s\n", dir.c_str(), stem.c_str(), ext.c_str());
    std::vector<std::string> snapshots = find_snapshots(dir, stem, ext);

    if (snapshots.empty()) {
        printf("  No snapshot files found.\n");
        return;
    }
    printf("  Found %d snapshots\n", (int)snapshots.size());

    
    std::string outdir = fel_dir ? std::string(fel_dir) : dir;
    mkdir(outdir.c_str(), 0755);

    printf("  Processing → %s/\n\n", outdir.c_str());

    for (size_t si = 0; si < snapshots.size(); si++) {
        const std::string &fpath = snapshots[si];
        long step = extract_step(fpath, ext);

        
        Meta meta;
        std::vector<Kernel> kernels;
        if (!parse_czar_file(fpath.c_str(), meta, kernels)) {
            printf("  Skipping %s: parse error\n", fpath.c_str());
            continue;
        }
        if (kernels.empty()) continue;

        printf("  [%d/%d] step %08ld  (%d kernels) ... ",
               (int)(si+1), (int)snapshots.size(), step, (int)kernels.size());
        fflush(stdout);

        
        std::vector<double> ptilde, czar_grad;
        std::vector<int> sizes;
        std::vector<double> gmin, gmax, dx;
        std::vector<bool> allowed;
        czar_on_grid(meta, kernels, grid_pts, nsigma,
                     ptilde, czar_grad, sizes, gmin, gmax, dx, allowed, minpop, false);

        
        std::vector<double> A;
        mc_integrate(czar_grad, allowed, sizes, dx, meta.periodic, meta.kT,
                     mc_steps, mc_hill, mc_hill_factor, false, A);

        
        char outname[256];
        snprintf(outname, sizeof(outname), "%s/FEL_%08ld.dat", outdir.c_str(), step);
        write_simple_fel(outname, meta.dim, sizes, gmin, dx, A, allowed);

        printf("→ FEL_%08ld.dat\n", step);
    }

    printf("\nDone. %d FEL files written to %s/\n", (int)snapshots.size(), outdir.c_str());
}



// Entry point. Parses command-line options, then either runs batch snapshot
// processing (-a flag) or processes a single CZAR kernel file.
int main(int argc, char *argv[]) {

    int grid_pts = 100;
    double nsigma = 4.0;
    unsigned int mc_steps = 0;
    double mc_hill = 0.01;
    double mc_hill_factor = 0.5;
    double minpop = 1e-3;
    double kT_override = 0.0;
    const char *output = "FEL_czar.dat";
    const char *fel_dir = NULL;
    bool verbose = false;
    bool process_all = false;
    const char *input = NULL;

    
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-') {
            switch (argv[i][1]) {
            case 'g': grid_pts = atoi(argv[++i]); break;
            case 's': nsigma = atof(argv[++i]); break;
            case 'n': mc_steps = (unsigned int)atoi(argv[++i]); break;
            case 't': kT_override = atof(argv[++i]); break;
            case 'h': mc_hill = atof(argv[++i]); break;
            case 'f': mc_hill_factor = atof(argv[++i]); break;
            case 'm': minpop = atof(argv[++i]); break;
            case 'o': output = argv[++i]; break;
            case 'd': fel_dir = argv[++i]; break;
            case 'a': process_all = true; break;
            case 'v': verbose = true; break;
            default:
                fprintf(stderr, "Unknown option: %s\n", argv[i]);
                return 1;
            }
        } else {
            input = argv[i];
        }
    }

    if (!input) {
        fprintf(stderr, "czar_integrate: MC integration of CZAR kernel gradient field\n\n");
        fprintf(stderr, "Usage: %s <czar_kernels_file> [options]\n\n", argv[0]);
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  -g <grid_pts>      Grid points per dim (default: 100)\n");
        fprintf(stderr, "  -s <nsigma>        Kernel cutoff in sigma (default: 4.0)\n");
        fprintf(stderr, "  -n <mc_steps>      MC steps (0 = auto-converge, default: 0)\n");
        fprintf(stderr, "  -t <kT>            Override kT from file (kJ/mol)\n");
        fprintf(stderr, "  -h <hill>          Initial hill height (default: 0.01)\n");
        fprintf(stderr, "  -f <hill_factor>   Hill reduction factor (default: 0.5)\n");
        fprintf(stderr, "  -m <minpop>        Min density fraction for allowed (default: 1e-3)\n");
        fprintf(stderr, "  -o <output_file>   Output file (default: FEL_czar.dat)\n");
        fprintf(stderr, "  -a                 Process ALL snapshots (stem_XXXXXXXX.dat)\n");
        fprintf(stderr, "  -d <dir>           Output directory for -a (default: same as input)\n");
        fprintf(stderr, "  -v                 Verbose output\n");
        return 1;
    }

    
    if (process_all) {
        process_all_snapshots(input, grid_pts, nsigma, minpop,
                              mc_steps, mc_hill, mc_hill_factor,
                              fel_dir, verbose);
        return 0;
    }

    

    
    Meta meta;
    std::vector<Kernel> kernels;
    printf("Reading CZAR kernels from: %s\n", input);
    if (!parse_czar_file(input, meta, kernels)) return 1;

    if (kT_override > 0) meta.kT = kT_override;

    printf("  dim=%d  kT=%.5f kJ/mol  kernels=%d\n", meta.dim, meta.kT, (int)kernels.size());
    printf("  periodic: ");
    for (int d = 0; d < meta.dim; d++) printf("%s%s", d?",":"", meta.periodic[d]?"true":"false");
    printf("\n");

    
    std::vector<double> ptilde, czar_grad;
    std::vector<int> sizes;
    std::vector<double> gmin, gmax, dx;
    std::vector<bool> allowed;

    printf("Evaluating CZAR gradient on %d^%d grid ...\n", grid_pts, meta.dim);
    czar_on_grid(meta, kernels, grid_pts, nsigma,
                 ptilde, czar_grad, sizes, gmin, gmax, dx, allowed, minpop, verbose);

    
    std::vector<double> A;
    printf("Integrating via MC ...\n");
    mc_integrate(czar_grad, allowed, sizes, dx, meta.periodic, meta.kT,
                 mc_steps, mc_hill, mc_hill_factor, verbose, A);

    
    printf("Writing FEL to: %s\n", output);
    write_output(output, meta, sizes, gmin, dx, ptilde, czar_grad, A, allowed);

    printf("Done.\n");
    return 0;
}
