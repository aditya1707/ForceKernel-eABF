#include "bias/Bias.h"
#include "core/ActionRegister.h"
#include "core/PlumedMain.h"
#include "core/Atoms.h"
#include "tools/IFile.h"
#include "tools/OFile.h"
#include <vector>
#include <array>
#include <unordered_map>
#include <map>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <random>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace PLMD {
namespace bias {


class ForceKernelABF : public Bias {
private:
  unsigned dim_;

  // --- internal extended Lagrangian ---
  std::vector<double> kappa_;
  std::vector<double> mass_;
  std::vector<double> friction_;
  double kT_;
  std::vector<double> s_fict_;
  std::vector<double> v_fict_;
  bool firstStep_;

  // adaptive sigma warmup (active when SIGMA not user-supplied)
  bool adaptiveSigma_;
  unsigned adaptiveSigmaStride_;
  unsigned adaptiveCounter_;
  std::vector<double> av_cv_;  // Welford running mean of z
  std::vector<double> av_M2_;  // Welford running sum of squared deviations
  std::mt19937 rng_;
  std::normal_distribution<double> gauss_;

  // --- kernel / ABF parameters ---
  unsigned pace_;
  double thresh_;     // compression threshold (default 1.0)
  double nsigmaCut_;  // kernel cutoff in sigma (default 4.0, same as OPES)
  // biasFactor_ (γ) controls the density-based exploration force on λ.
  // γ = 1: pure ABF, no exploration force.
  // γ > 1: F_ex = −c·∇Z/(Z₀+Z) where c=kT(γ−1), Z is the NW denominator
  //   interpolated from nwDenominator_, and Z₀=median(Z) on the grid.  The force
  //   pushes λ away from well-sampled basins toward under-sampled regions.
  //   Both Z and F_ex are updated at GRIDPACE intervals, synchronized with
  //   the ABF mean force.
  // The CZAR estimator on z is not directly affected (exploration acts on λ only,
  // though it indirectly influences z-kernel statistics through altered sampling).
  double biasFactor_;
  double Z0_density_;   // median-Z reference for density-based exploration (denominator)
  std::vector<double> explorScale_;  // per-CV scaling of exploration force (default 1.0)
  double muxClamp_;   // per-kernel force clamp (default 500)
  double maxForce_;   // grid bias force clamp  (default 500)
  std::vector<double> sigma0_;
  std::vector<double> sigmaMin_;
  bool fixedSigma_;
  bool hasMin_;    // cached !sigmaMin_.empty(); avoids repeated .size() checks

  // -- lambda-kernels (bias driving, indexed at s_fict) --------------------------
  struct Kernel {
    std::array<double,3> center = {};
    std::array<double,3> mu = {};      // mean force estimate (clamped, exact running mean)
    std::array<double,3> sigma = {};   // effective local bandwidth (not sample variance; see addSample)
    double Nk = 0.0;
    uint64_t id = 0;
  };
  std::vector<Kernel> kernels_;
  unsigned nKernels_;
  double totalN_;
  double sumNk2_;   // sum of Nk^2; neff = totalN_^2 / sumNk2_

  // -- z-kernels (CZAR estimator, indexed at real CV z) -------------------------
  // Centers track the real collective variable (not the fictitious lambda).
  // mu stores the unclamped mean spring force kappa*(z - lambda).
  // Always uses the exact running mean.
  struct ZKernel {
    std::array<double,3> center = {};  // position in real CV space
    std::array<double,3> mu = {};      // unclamped mean of kappa(z - lambda)
    std::array<double,3> sigma = {};   // effective local bandwidth (see addZSample)
    double Nk = 0.0;
    uint64_t id = 0;
  };
  std::vector<ZKernel> zKernels_;
  unsigned nZKernels_;
  double zTotalN_;
  double zSumNk2_;  // sum of zNk^2 for z-kernel neff

  // --- stable kernel IDs and id→index maps ---
  uint64_t nextKernelId_;
  uint64_t nextZKernelId_;
  std::unordered_map<uint64_t, unsigned> idMap_;   // lambda kernel id → index
  std::unordered_map<uint64_t, unsigned> zIdMap_;  // z-kernel id → index

  // --- incremental grid update tracking (lambda-kernels only, no z-kernel grid maintained)
  // Dirty tracking is first-touch: records the delta between the last published
  // grid and the final kernel state at publish time.
  struct DirtyEntry {
    std::array<double,3> old_center, old_mu, old_sigma;
    double old_Nk;
    bool has_old;  // false = newly created this interval, no old contribution on grid
  };
  std::map<uint64_t, DirtyEntry> dirty_;
  std::vector<double> nwNumerator_;  // persistent NW numerator [gridTotal_ * dim_]
  unsigned fullRebuildInterval_;  // FULL rebuilds every N GRIDPACE events (for floating-point hygiene)
  unsigned gridRebuildCount_;     // count GRIDPACE events since last full rebuild
  bool lastRebuildWasFull_;       // diagnostic: was the last grid update a full rebuild?

  // --- neighbor list (for lambda-kernels only) ---
  bool nlist_;
  double nlistCutFactor_;
  double nlistSkinFactor_;
  std::vector<unsigned> nlistIdx_;
  std::vector<double> nlistCenter_;
  std::vector<double> nlistDev2_;
  bool nlistUpdate_;

  // --- neighbor list (for z-kernels; accelerates geometric merge search) ---
  std::vector<unsigned> znlistIdx_;
  std::vector<double> znlistCenter_;
  std::vector<double> znlistDev2_;
  bool znlistUpdate_;

  // --- grid ---
  std::vector<unsigned> gridN_;
  unsigned gridPace_, gridTotal_;
  std::vector<double> gridMin_, gridMax_, gridDx_;
  std::vector<double> meanForceGrid_;   // NW mean force grid [gridTotal_ * dim_], for direct ABF force
  std::vector<double> nwDenominator_;  // NW denominator on grid [gridTotal_], for exploration + Neff
  std::vector<double> explorationForceGrid_;    // exploration force on grid [gridTotal_ * dim_], zero when γ=1

  // --- lambda-grid output (debug: NW mean force on λ grid) ---
  // Filename derived from label: {label}.lambda_grid.dat
  // Step-stamped on each write: {label}.lambda_grid_{step:08d}.dat
  // Off by default; enabled by setting LAMBDAGRIDSTRIDE > 0.
  std::string lambdaGridFile_;
  unsigned lambdaGridStride_;

  // --- domain ---
  std::vector<double> domMin_, domMax_, domLen_;
  std::vector<bool> periodic_;

  // --- lambda-kernel dump ---
  // Filename derived from label: {label}.kernels.dat
  // Step-stamped on each write: {label}.kernels_{step:08d}.dat
  std::string kernelFile_;
  unsigned kernelStride_;

  // --- CZAR kernel file (z-kernels) ---
  // Filename derived from label: {label}.czar_kernels.dat
  // Step-stamped on each write: {label}.czar_kernels_{step:08d}.dat
  std::string czarFile_;
  unsigned czarStride_;

  // --- restart state file ---
  // Written at STATESTRIDE interval, read on RESTART.
  // Contains: fictitious particle, kernel populations (with stable IDs),
  // sigma0, Z0_density, ID counters, and RNG state.
  // The mean-force grid is fully rebuilt from kernels on RESTART.
  std::string stateFile_;
  unsigned stateStride_;

  // --- kernel diagnostics file ---
  // Appends one line per write to {label}.kernelinfo.dat.
  // Columns: step, M, zM, neff, sigma per CV dim, nlker.
  std::string kernelInfoFile_;
  unsigned kernelInfoStride_;
  OFile kernelInfoOFile_;
  bool kernelInfoFileOpen_;

  // --- state ---
  std::vector<std::string> fictNames_;

  // --- cached Silverman bandwidths (recomputed only when nKernels_ or nZKernels_ changes) ---
  std::vector<double> cachedSig_;
  std::vector<double> cachedZSig_;
  bool sigDirty_;
  bool zSigDirty_;

  // --- precomputed BAOAB thermostat constants ---
  // c1 = exp(-friction*dt), c2 = sqrt(kT/mass * (1 - c1^2))
  // Computed once at first use; changing the timestep mid-simulation is unsupported.
  std::vector<double> baoab_c1_;  // exp(-friction[i] * dt) per dimension
  std::vector<double> baoab_c2_; // sqrt(kT/mass[i] * (1 - c1[i]^2))
  bool baoabReady_;

  // --- per-step work arrays (avoid heap allocation in hot path) ---
  // Interpolation: shared lo/frac computation (max dim=3)
  double work_frac_[3];
  unsigned work_lo_[3];
  // calculate() per-step vectors
  std::vector<double> work_cv_, work_springF_;
  // addSample / addZSample work vectors
  std::vector<double> work_s_, work_f_, work_z_;
  // interpolation output work vectors
  std::vector<double> work_gf_, work_fe_, work_gf2_, work_fev_;
  // grid splat work vectors (per-dimension 1D weights)
  std::vector<std::vector<double>> work_w1d_;
  std::vector<std::vector<unsigned>> work_gi1d_;
  // gridUnflat work vector
  std::vector<unsigned> work_idx_;

  // ================ helpers ================
  double sq(double x) const { return x * x; }

  // First-touch dirty recording (for C++14-compatible try_emplace equivalent).
  // Inserts only if the key does not already exist, preserving the original snapshot.
  void dirtyRecord(uint64_t id, const DirtyEntry& entry) {
    if (dirty_.count(id) == 0) dirty_.emplace(id, entry);
  }

  double wrapToDomain(unsigned i, double x) const {
    if (!periodic_[i]) return x;
    double L = domLen_[i]; if (L <= 0) return x;
    double y = x - domMin_[i];
    y -= L * std::floor(y / L);
    return domMin_[i] + y;
  }

  double periodicDelta(unsigned i, double from, double to) const {
    double d = to - from;
    if (periodic_[i] && domLen_[i] > 0)
      d -= domLen_[i] * std::round(d / domLen_[i]);
    return d;
  }

  // Silverman bandwidth for lambda-kernels.
  // Uses n_eff = totalN_^2 / sumNk2_, i.e. the effective sample size accounting
  // for the unequal distribution of counts across kernels.  When all M kernels
  // carry equal Nk = n, n_eff = (Mn)^2 / (Mn^2) = M (the kernel count).
  // When one kernel dominates, n_eff -> 1.
  // Falls back to nKernels_ only if sumNk2_ is zero (initial state).
  const std::vector<double>& currentSigma() {
    if (!sigDirty_) return cachedSig_;
    cachedSig_ = sigma0_;
    if (!fixedSigma_ && nKernels_ > 1) {
      const double neff = (sumNk2_ > 0.0) ? (totalN_*totalN_/sumNk2_) : (double)nKernels_;
      double s_rescaling = std::pow(neff*(dim_+2.0)/4.0, -1.0/(4.0+dim_));
      for (unsigned i = 0; i < dim_; ++i) {
        cachedSig_[i] *= s_rescaling;
        if (hasMin_) cachedSig_[i] = std::max(cachedSig_[i], sigmaMin_[i]);
      }
    }
    sigDirty_ = false;
    return cachedSig_;
  }


  // Silverman bandwidth for z-kernels.
  const std::vector<double>& currentZSigma() {
    if (!zSigDirty_) return cachedZSig_;
    cachedZSig_ = sigma0_;
    if (!fixedSigma_ && nZKernels_ > 1) {
      const double neff = (zSumNk2_ > 0.0) ? (zTotalN_*zTotalN_/zSumNk2_) : (double)nZKernels_;
      double s_rescaling = std::pow(neff*(dim_+2.0)/4.0, -1.0/(4.0+dim_));
      for (unsigned i = 0; i < dim_; ++i) {
        cachedZSig_[i] *= s_rescaling;
        if (hasMin_) cachedZSig_[i] = std::max(cachedZSig_[i], sigmaMin_[i]);
      }
    }
    zSigDirty_ = false;
    return cachedZSig_;
  }

  double dist2KernelNorm(const double* s, unsigned k,
                         const std::vector<double>& sig) const {
    // Merge distance normalized by the current global Silverman bandwidth.
    // Using per-kernel sigma causes a positive feedback loop where shrinking
    // sigma = tighter merge radius = more kernels = more shrinkage.
    double acc = 0.0;
    for (unsigned i = 0; i < dim_; ++i) {
      double d = s[i] - kernels_[k].center[i];
      if (periodic_[i] && domLen_[i] > 0)
        d -= domLen_[i] * std::round(d / domLen_[i]);
      acc += sq(d) / (4.0*sq(sig[i]) + 1e-300);
    }
    return acc;
  }

  // -- lambda-kernel merge search (purely geometric) --
  int findMergeable(const double* s, const std::vector<double>& sig, int exclude = -1) {
    double r2 = 0.25*sq(thresh_); int best = -1; double bestd2 = r2;
    auto evalKernel = [&](unsigned k) {
      if ((int)k == exclude) return;
      double d2 = dist2KernelNorm(s, k, sig);
      if (d2 < bestd2) { bestd2 = d2; best = (int)k; }
    };
    if (nlist_ && !nlistIdx_.empty()) {
      for (unsigned n = 0; n < nlistIdx_.size(); ++n) evalKernel(nlistIdx_[n]);
    } else {
      for (unsigned k = 0; k < nKernels_; ++k) evalKernel(k);
    }
    return best;
  }

  // -- z-kernel merge search (purely geometric, uses global Silverman sigma) --
  int findMergeableZ(const double* s, const std::vector<double>& sig, int exclude = -1) {
    double r2 = 0.25*sq(thresh_); int best = -1; double bestd2 = r2;
    auto evalKernel = [&](unsigned k) {
      if ((int)k == exclude) return;
      double d2 = 0.0;
      for (unsigned i = 0; i < dim_; ++i) {
        double d = s[i] - zKernels_[k].center[i];
        if (periodic_[i] && domLen_[i] > 0)
          d -= domLen_[i] * std::round(d / domLen_[i]);
        d2 += sq(d) / (4.0*sq(sig[i]) + 1e-300);
      }
      if (d2 < bestd2) { bestd2 = d2; best = (int)k; }
    };

    if (nlist_ && !znlistIdx_.empty()) {
      for (unsigned idx : znlistIdx_) evalKernel(idx);
    } else {
      for (unsigned k = 0; k < nZKernels_; ++k) evalKernel(k);
    }
    return best;
  }

  // ================ neighbor list (lambda-kernels only) ================
  // The broad-phase cutoff must use the same distance metric as findMergeable
  // (global Silverman sigma), otherwise valid merge candidates can be excluded
  // when per-kernel sigma differs from the global sigma.
  void updateNlist(const std::vector<double>& cv, const std::vector<double>& sig) {
    nlistCenter_ = cv;
    nlistIdx_.clear();
    if (nKernels_ == 0) { nlistUpdate_ = false; return; }
    double cutoff2 = 0.25 * nlistCutFactor_ * sq(nsigmaCut_);
    for (unsigned k = 0; k < nKernels_; ++k) {
      double norm2 = 0.0;
      for (unsigned i = 0; i < dim_; ++i) {
        double d = periodicDelta(i, nlistCenter_[i], kernels_[k].center[i]);
        norm2 += sq(d) / (4.0*sq(sig[i]) + 1e-300);
      }
      if (norm2 <= cutoff2) nlistIdx_.push_back(k);
    }
    nlistDev2_.assign(dim_, 0.0);
    if (!nlistIdx_.empty()) {
      for (unsigned n = 0; n < nlistIdx_.size(); ++n) {
        unsigned k = nlistIdx_[n];
        for (unsigned i = 0; i < dim_; ++i) {
          double d = periodicDelta(i, nlistCenter_[i], kernels_[k].center[i]);
          nlistDev2_[i] += sq(d);
        }
      }
      for (unsigned i = 0; i < dim_; ++i)
        nlistDev2_[i] /= nlistIdx_.size();
    } else {
      for (unsigned i = 0; i < dim_; ++i) nlistDev2_[i] = sq(sig[i]);
    }
    nlistUpdate_ = false;
  }

  bool needsNlistUpdate(const std::vector<double>& cv) const {
    for (unsigned i = 0; i < dim_; ++i) {
      double d = periodicDelta(i, cv[i], nlistCenter_[i]);
      if (sq(d) > nlistSkinFactor_ * nlistDev2_[i]) return true;
    }
    return false;
  }

  // ================ neighbor list (z-kernels) ================
  // Same global-sigma metric as findMergeableZ for consistent broad-phase filtering.
  void updateZNlist(const std::vector<double>& cv, const std::vector<double>& sig) {
    znlistCenter_ = cv;
    znlistIdx_.clear();
    if (nZKernels_ == 0) { znlistUpdate_ = false; return; }

    const double cutoff2 = 0.25 * nlistCutFactor_ * sq(nsigmaCut_);
    for (unsigned k = 0; k < nZKernels_; ++k) {
      double norm2 = 0.0;
      for (unsigned i = 0; i < dim_; ++i) {
        double d = periodicDelta(i, znlistCenter_[i], zKernels_[k].center[i]);
        norm2 += sq(d) / (4.0*sq(sig[i]) + 1e-300);
      }
      if (norm2 <= cutoff2) znlistIdx_.push_back(k);
    }

    znlistDev2_.assign(dim_, 0.0);
    if (!znlistIdx_.empty()) {
      for (unsigned idx : znlistIdx_) {
        for (unsigned i = 0; i < dim_; ++i) {
          double d = periodicDelta(i, znlistCenter_[i], zKernels_[idx].center[i]);
          znlistDev2_[i] += sq(d);
        }
      }
      for (unsigned i = 0; i < dim_; ++i)
        znlistDev2_[i] /= (double)znlistIdx_.size();
    } else {
      for (unsigned i = 0; i < dim_; ++i) znlistDev2_[i] = sq(sig[i]);
    }

    znlistUpdate_ = false;
  }

  bool needsZNlistUpdate(const std::vector<double>& cv) const {
    for (unsigned i = 0; i < dim_; ++i) {
      double d = periodicDelta(i, cv[i], znlistCenter_[i]);
      if (sq(d) > nlistSkinFactor_ * znlistDev2_[i]) return true;
    }
    return false;
  }

  // ================ generic kernel-pool merge loop ================
  // Shared implementation for both lambda-kernels and z-kernels.
  // Starting from 'giver', repeatedly finds the nearest mergeable neighbor
  // and fuses the pair (weighted pairwise variance merge) until no further
  // merge candidates exist.
  //
  // Template parameters:
  //   K          -- kernel type (Kernel or ZKernel)
  //   FindFn     -- callable(center, exclude) -> int
  //   MuFn       -- callable(mu_merged) -> double: post-merge mu transform
  //   PreMergeFn -- callable(taker, giver): first-touch dirty recording
  //
  // The giver kernel is removed by swap-with-last (O(1)); the nlist index
  // array and id to index map are patched in-place to keep them consistent.
  template<typename K, typename FindFn, typename MuFn, typename PreMergeFn>
  void mergeKernelPool(unsigned giver,
                       std::vector<K>& kernels,
                       unsigned& M,
                       double& sumNk2,
                       std::vector<unsigned>& nlistIdx,
                       std::unordered_map<uint64_t, unsigned>& idMap,
                       FindFn findMerge,
                       MuFn muUpdate,
                       PreMergeFn preMerge) {
    const bool hasMin = !sigmaMin_.empty();
    int taker = findMerge(kernels[giver].center, (int)giver);
    while (taker >= 0) {
      if (M == 0)
        error("mergeKernelPool: kernel count reached zero during cascade — logic error");
      if (giver >= kernels.size() || (unsigned)taker >= kernels.size())
        error("mergeKernelPool: kernel index out of bounds (giver=" + std::to_string(giver)
              + " taker=" + std::to_string(taker) + " size=" + std::to_string(kernels.size()) + ")");

      // First-touch dirty recording before modification
      preMerge(kernels[taker], kernels[giver]);

      double Nt = kernels[taker].Nk, Ng = kernels[giver].Nk, Ntot = Nt + Ng;
      if (Ntot <= 0.0)
        error("mergeKernelPool: merged Nk total is non-positive (Nt=" + std::to_string(Nt)
              + " Ng=" + std::to_string(Ng) + ") — corrupted kernel data");
      sumNk2 -= (Nt*Nt + Ng*Ng);

      const double inv = 1.0 / Ntot;
      for (unsigned i = 0; i < dim_; ++i) {
        double ct = kernels[taker].center[i];
        double cg = kernels[giver].center[i];
        if (periodic_[i] && domLen_[i] > 0)
          cg = ct + periodicDelta(i, ct, cg);
        double c_new = (Nt*ct + Ng*cg) * inv;
        double dt = ct - c_new, dg = cg - c_new;
        double var = (Nt*(sq(kernels[taker].sigma[i]) + sq(dt)) +
                      Ng*(sq(kernels[giver].sigma[i]) + sq(dg))) * inv;
        kernels[taker].center[i] = wrapToDomain(i, c_new);
        kernels[taker].sigma[i] = std::sqrt(std::max(var,
            sq(hasMin ? sigmaMin_[i] : 1e-6)));
        double mu_new = (Nt*kernels[taker].mu[i] + Ng*kernels[giver].mu[i]) * inv;
        kernels[taker].mu[i] = muUpdate(mu_new);
      }
      kernels[taker].Nk = Ntot;
      sumNk2 += Ntot*Ntot;

      // Swap giver with the last kernel to remove it in O(1).
      // If taker was the last element, it moves into giver's slot and newGiver tracks this.
      uint64_t giver_id = kernels[giver].id;
      unsigned last = M - 1;
      unsigned newGiver = (unsigned)taker;
      if (giver != last && (unsigned)taker == last) newGiver = giver;
      if (giver != last) kernels[giver] = std::move(kernels[last]);
      kernels.resize(last);
      --M;
      // Patch id→index map: erase deleted giver, update moved kernel
      idMap.erase(giver_id);
      if (giver != last) idMap[kernels[giver].id] = giver;
      // Patch neighbor list
      if (nlist_) {
        for (auto it = nlistIdx.begin(); it != nlistIdx.end(); ) {
          if (*it == giver) { it = nlistIdx.erase(it); }
          else {
            if (*it == last) *it = giver;
            ++it;
          }
        }
      }
      giver = newGiver;
      taker = findMerge(kernels[giver].center, (int)giver);
    }
  }


  // ================ lambda-kernel compression (the kernels that drive the bias) ================
  void addSample(const std::vector<double>& s_in, const std::vector<double>& f_in) {
    for (unsigned i = 0; i < dim_; ++i) {
      work_s_[i] = wrapToDomain(i, s_in[i]);
      work_f_[i] = std::max(-muxClamp_, std::min(muxClamp_, f_in[i]));
    }

    // Rebuild lambda-kernel neighbor list if flagged or if s_fict has drifted by
    // checking if λ moved far enough relative to how spread-out the nearby kernels are
    if (nlist_ && nKernels_ > 0) {
      if (nlistUpdate_ || needsNlistUpdate(s_in))
        updateNlist(s_in, currentSigma());
    }

    const std::vector<double>& sig = currentSigma();

    int k = findMergeable(work_s_.data(), sig, -1);
    unsigned giver;

    if (k >= 0) {
      // First-touch dirty record BEFORE modifying the kernel
      dirtyRecord(kernels_[k].id, DirtyEntry{
          kernels_[k].center, kernels_[k].mu, kernels_[k].sigma, kernels_[k].Nk, true});

      double Nold = kernels_[k].Nk;
      sumNk2_ -= Nold*Nold;
      kernels_[k].Nk += 1.0;
      double Nnew = kernels_[k].Nk;
      sumNk2_ += Nnew*Nnew;
      totalN_ += 1.0;
      sigDirty_ = true;
      const double inv_Nnew = 1.0 / Nnew;

      for (unsigned i = 0; i < dim_; ++i) {
        double muold = kernels_[k].mu[i];
        double munew = (Nold*muold + work_f_[i]) * inv_Nnew;
        kernels_[k].mu[i] = std::max(-muxClamp_, std::min(muxClamp_, munew));

        double ct = kernels_[k].center[i];
        double cs = ct + periodicDelta(i, ct, work_s_[i]);
        double c_new = (Nold * ct + cs) * inv_Nnew;
        double dt = ct - c_new, ds = cs - c_new;
        // Per-kernel width update (Chan et al. pairwise merge).
        // NB: the incoming sample is treated as a Gaussian of width sigma_global,
        // NOT as a point (zero variance).  Consequently sigma_k is an effective
        // local KDE bandwidth — the combined width of all absorbed Gaussians —
        // rather than the literal spatial variance of the absorbed sample positions.
        // This ensures that even a singleton kernel has a well-defined influence
        // radius in the NW regression and prevents pathologically narrow kernels.
        double var = (Nold * (sq(kernels_[k].sigma[i]) + sq(dt)) +
                             (sq(sig[i])               + sq(ds))) * inv_Nnew;
        kernels_[k].center[i] = wrapToDomain(i, c_new);
        kernels_[k].sigma[i] = std::sqrt(std::max(var,
            sq(hasMin_ ? sigmaMin_[i] : 1e-6)));
      }
      giver = (unsigned)k;

    } else {
      Kernel nk;
      for (unsigned i = 0; i < dim_; ++i) {
        nk.center[i] = work_s_[i];
        nk.mu[i] = work_f_[i];
        nk.sigma[i] = sig[i];  // initial bandwidth = current Silverman σ_global (not zero)
      }
      nk.Nk = 1.0;
      nk.id = nextKernelId_++;
      kernels_.push_back(std::move(nk));
      ++nKernels_;
      totalN_ += 1.0;
      sumNk2_ += 1.0;
      sigDirty_ = true;
      idMap_[kernels_.back().id] = nKernels_ - 1;
      // Dirty-record as newly created (no old contribution on grid)
      dirtyRecord(kernels_.back().id, DirtyEntry{{}, {}, {}, 0.0, false});
      if (nlist_) nlistIdx_.push_back(nKernels_-1);
      giver = nKernels_ - 1;
    }

    mergeKernelPool(giver, kernels_, nKernels_, sumNk2_, nlistIdx_, idMap_,
        [this](const std::array<double,3>& c, int ex){ sigDirty_ = true; return findMergeable(c.data(), currentSigma(), ex); },
        [this](double mu){ return std::max(-muxClamp_, std::min(muxClamp_, mu)); },
        [this](const Kernel& t, const Kernel& g) {
            dirtyRecord(t.id, DirtyEntry{t.center, t.mu, t.sigma, t.Nk, true});
            dirtyRecord(g.id, DirtyEntry{g.center, g.mu, g.sigma, g.Nk, true});
        });
    sigDirty_ = true;

    if (nlist_) nlistUpdate_ = true;
  }

  // ================ z-kernel accumulation (CZAR) ================
  // z-kernels are centred at the real CV z and store the unclamped spring force
  // as a running mean.  They are NOT evaluated on a grid inside this module;
  // they are written to file (writeCZARFile) for offline CZAR integration by
  // czar_integrate.  The KDE normalization factor alpha_k = prod(sigma0/sigma_k)
  // is applied at evaluation time in czar_integrate (not stored in the kernel
  // data), so the raw Nk, center, mu, sigma written to file are unnormalized.
  // sigma0 is written to the CZAR file header to enable this correction.
  void addZSample(const std::vector<double>& z_in,
                  const std::vector<double>& f_raw) {
    for (unsigned i = 0; i < dim_; ++i)
      work_z_[i] = wrapToDomain(i, z_in[i]);

    if (nlist_ && nZKernels_ > 0) {
      const std::vector<double>& zsig = currentZSigma();
      if (znlistUpdate_ || needsZNlistUpdate(work_z_)) updateZNlist(work_z_, zsig);
    }

    const std::vector<double>& sig = currentZSigma();

    int best = findMergeableZ(work_z_.data(), sig, -1);
    unsigned giver;

    if (best >= 0) {
      double Nold = zKernels_[best].Nk;
      zSumNk2_ -= Nold*Nold;
      zKernels_[best].Nk += 1.0;
      double Nnew = zKernels_[best].Nk;
      zSumNk2_ += Nnew*Nnew;
      zTotalN_ += 1.0;
      zSigDirty_ = true;
      const double inv_Nnew = 1.0 / Nnew;

      for (unsigned i = 0; i < dim_; ++i) {
        zKernels_[best].mu[i] = (Nold * zKernels_[best].mu[i] + f_raw[i]) * inv_Nnew;

        double ct = zKernels_[best].center[i];
        double cs = ct + periodicDelta(i, ct, work_z_[i]);
        double c_new = (Nold * ct + cs) * inv_Nnew;
        double dt = ct - c_new, ds = cs - c_new;
        // Per-kernel width update: see comment in addSample — sigma_k is an
        // effective local bandwidth, not the sample spatial variance.
        double var = (Nold * (sq(zKernels_[best].sigma[i]) + sq(dt)) +
                             (sq(sig[i])                   + sq(ds))) * inv_Nnew;
        zKernels_[best].center[i] = wrapToDomain(i, c_new);
        zKernels_[best].sigma[i] = std::sqrt(std::max(var,
            sq(hasMin_ ? sigmaMin_[i] : 1e-6)));
      }
      giver = (unsigned)best;

    } else {
      ZKernel nk;
      for (unsigned i = 0; i < dim_; ++i) {
        nk.center[i] = work_z_[i];
        nk.mu[i]     = f_raw[i];
        nk.sigma[i]  = sig[i];  // initial bandwidth = current Silverman σ_global (not zero)
      }
      nk.Nk = 1.0;
      nk.id = nextZKernelId_++;
      zKernels_.push_back(std::move(nk));
      ++nZKernels_;
      zTotalN_ += 1.0;
      zSumNk2_ += 1.0;
      zSigDirty_ = true;
      zIdMap_[zKernels_.back().id] = nZKernels_ - 1;
      if (nlist_) {
        znlistIdx_.push_back(nZKernels_ - 1);
        znlistUpdate_ = true;
      }
      giver = nZKernels_ - 1;
    }

    mergeKernelPool(giver, zKernels_, nZKernels_, zSumNk2_, znlistIdx_, zIdMap_,
        [this](const std::array<double,3>& c, int ex){ zSigDirty_ = true; return findMergeableZ(c.data(), currentZSigma(), ex); },
        [](double mu){ return mu; },
        [](const ZKernel&, const ZKernel&){});  // no dirty tracking for z-kernels
    zSigDirty_ = true;

    if (nlist_) znlistUpdate_ = true;
  }

  // ================ grid indexing ================
  unsigned gridFlat(const std::vector<unsigned>& idx) const {
    unsigned f = 0;
    for (unsigned d = 0; d < dim_; ++d) f = f * gridN_[d] + idx[d];
    return f;
  }
  void gridUnflat(unsigned flat, std::vector<unsigned>& idx) const {
    for (int d = (int)dim_-1; d >= 0; --d) { idx[d] = flat % gridN_[d]; flat /= gridN_[d]; }
  }
  double gridCoord(unsigned d, unsigned idx) const {
    return gridMin_[d] + idx * gridDx_[d];
  }

  // ================ incremental grid update: add/subtract one kernel ================
  // Scatters a single kernel's Gaussian-weighted contribution onto the NW grid
  // (numerator = weight*mu, denominator = weight*Nk). Pass sign=+1 to add,
  // sign=-1 to remove. Matches the cutoff and weights used in the full rebuild.
  void splatKernel(const std::array<double,3>& center,
                   const std::array<double,3>& sigma,
                   const std::array<double,3>& mu,
                   double Nk, double sign) {
    int R[3], ic[3];
    for (unsigned d = 0; d < dim_; ++d) {
      R[d] = (int)std::ceil(nsigmaCut_ * sigma[d] / std::max(gridDx_[d], 1e-12));
      ic[d] = (int)std::round((center[d] - gridMin_[d]) / gridDx_[d]);
      if (periodic_[d])
        ic[d] = ((ic[d] % (int)gridN_[d]) + (int)gridN_[d]) % (int)gridN_[d];
      else
        ic[d] = std::max(0, std::min(ic[d], (int)gridN_[d]-1));
    }
    for (unsigned d = 0; d < dim_; ++d) {
      double inv4s2 = 1.0 / (4.0 * sq(sigma[d]) + 1e-300);
      work_w1d_[d].clear();   work_w1d_[d].reserve(2*R[d]+1);
      work_gi1d_[d].clear(); work_gi1d_[d].reserve(2*R[d]+1);
      for (int r = -R[d]; r <= R[d]; ++r) {
        int raw = ic[d] + r;
        unsigned gi;
        if (periodic_[d]) {
          gi = (unsigned)(((raw % (int)gridN_[d]) + (int)gridN_[d]) % (int)gridN_[d]);
        } else {
          if (raw < 0 || raw >= (int)gridN_[d]) continue;
          gi = (unsigned)raw;
        }
        double dd = periodicDelta(d, center[d], gridCoord(d, gi));
        double w = std::exp(-sq(dd) * inv4s2);
        if (w < 1e-300) continue;
        work_w1d_[d].push_back(w);
        work_gi1d_[d].push_back(gi);
      }
    }
    // KDE normalization: weight by alpha_k = prod(sigma0/sigma_k) so that
    // the NW denominator is proportional to a properly normalized KDE.
    // This enters both numerator and denominator of the NW ratio (cancels
    // partially) and fixes the density estimate used by the exploration force.
    double alpha = 1.0;
    for (unsigned d = 0; d < dim_; ++d)
      alpha *= sigma0_[d] / (sigma[d] + 1e-300);
    const double sNk = sign * Nk * alpha;
    // Spherical cutoff: reject when total exponent exceeds nsigmaCut_²/4.
    // Since 1D weights are exp(-e_d), the product exp(-Σe_d) < this threshold
    // means the point is outside the ellipsoidal cutoff.
    const double sphereCut = std::exp(-0.25 * nsigmaCut_ * nsigmaCut_);
    if (dim_ == 1) {
      for (unsigned a = 0; a < work_w1d_[0].size(); ++a) {
        unsigned g = work_gi1d_[0][a]; double wNk = work_w1d_[0][a] * sNk;
        nwDenominator_[g] += wNk; nwNumerator_[g] += wNk * mu[0];
      }
    } else if (dim_ == 2) {
      for (unsigned a = 0; a < work_w1d_[0].size(); ++a) {
        double wa = work_w1d_[0][a];
        for (unsigned b = 0; b < work_w1d_[1].size(); ++b) {
          double wab = wa * work_w1d_[1][b];
          if (wab < sphereCut) continue;  // ellipsoidal cutoff
          double wNk = wab * sNk;
          unsigned g = work_gi1d_[0][a]*gridN_[1] + work_gi1d_[1][b];
          nwDenominator_[g] += wNk;
          nwNumerator_[g*2+0] += wNk * mu[0];
          nwNumerator_[g*2+1] += wNk * mu[1];
        }
      }
    } else {
      for (unsigned a = 0; a < work_w1d_[0].size(); ++a) {
        double wa = work_w1d_[0][a];
        for (unsigned b = 0; b < work_w1d_[1].size(); ++b) {
          double wab = wa * work_w1d_[1][b];
          if (wab < sphereCut) continue;  // early exit: wab*wc <= wab
          unsigned gab = (work_gi1d_[0][a]*gridN_[1]+work_gi1d_[1][b])*gridN_[2];
          for (unsigned c = 0; c < work_w1d_[2].size(); ++c) {
            double wabc = wab * work_w1d_[2][c];
            if (wabc < sphereCut) continue;  // ellipsoidal cutoff
            double wNk = wabc * sNk;
            unsigned g = gab + work_gi1d_[2][c];
            nwDenominator_[g] += wNk;
            nwNumerator_[g*3+0] += wNk * mu[0];
            nwNumerator_[g*3+1] += wNk * mu[1];
            nwNumerator_[g*3+2] += wNk * mu[2];
          }
        }
      }
    }
  }

  // ================ full grid rebuild (lambda-kernels -> nwNumerator_ and nwDenominator_) ========
  // Zeroes and recomputes the persistent NW numerator (nwNumerator_) and denominator
  // (nwDenominator_) from all lambda-kernels.  Does NOT compute meanForceGrid_ (the ratio) —
  // that is done once in finalizeGrid() after either full or incremental update.
  void fullGridRebuild() {
    nwNumerator_.assign(gridTotal_ * dim_, 0.0);
    nwDenominator_.assign(gridTotal_, 0.0);
    if (nKernels_ == 0) return;

    std::vector<int> R(dim_), ic(dim_);
    const double sphereCut = std::exp(-0.25 * nsigmaCut_ * nsigmaCut_);

    for (unsigned kk = 0; kk < nKernels_; ++kk) {
      for (unsigned d = 0; d < dim_; ++d) {
        R[d] = (int)std::ceil(nsigmaCut_ * kernels_[kk].sigma[d] / std::max(gridDx_[d], 1e-12));
        ic[d] = (int)std::round((kernels_[kk].center[d] - gridMin_[d]) / gridDx_[d]);
        if (periodic_[d])
          ic[d] = ((ic[d] % (int)gridN_[d]) + (int)gridN_[d]) % (int)gridN_[d];
        else
          ic[d] = std::max(0, std::min(ic[d], (int)gridN_[d]-1));
      }

      for (unsigned d = 0; d < dim_; ++d) {
        double inv4s2 = 1.0 / (4.0 * sq(kernels_[kk].sigma[d]) + 1e-300);
        work_w1d_[d].clear();   work_w1d_[d].reserve(2*R[d]+1);
        work_gi1d_[d].clear(); work_gi1d_[d].reserve(2*R[d]+1);
        for (int r = -R[d]; r <= R[d]; ++r) {
          int raw = ic[d] + r;
          unsigned gi;
          if (periodic_[d]) {
            gi = (unsigned)(((raw % (int)gridN_[d]) + (int)gridN_[d]) % (int)gridN_[d]);
          } else {
            if (raw < 0 || raw >= (int)gridN_[d]) continue;
            gi = (unsigned)raw;
          }
          double dd = periodicDelta(d, kernels_[kk].center[d], gridCoord(d, gi));
          double w = std::exp(-sq(dd) * inv4s2);
          if (w < 1e-300) continue;
          work_w1d_[d].push_back(w);
          work_gi1d_[d].push_back(gi);
        }
      }

      // KDE normalization: alpha_k = prod(sigma0/sigma_k)
      double alpha_kk = 1.0;
      for (unsigned d = 0; d < dim_; ++d)
        alpha_kk *= sigma0_[d] / (kernels_[kk].sigma[d] + 1e-300);
      const double Nk_kk = kernels_[kk].Nk * alpha_kk;
      if (dim_ == 1) {
        for (unsigned a = 0; a < work_w1d_[0].size(); ++a) {
          unsigned g = work_gi1d_[0][a]; double wNk = work_w1d_[0][a] * Nk_kk;
          nwDenominator_[g] += wNk; nwNumerator_[g] += wNk*kernels_[kk].mu[0];
        }
      } else if (dim_ == 2) {
        for (unsigned a = 0; a < work_w1d_[0].size(); ++a) {
          double wa = work_w1d_[0][a];
          for (unsigned b = 0; b < work_w1d_[1].size(); ++b) {
            double wab = wa * work_w1d_[1][b];
            if (wab < sphereCut) continue;  // ellipsoidal cutoff
            double wNk = wab * Nk_kk;
            unsigned g = work_gi1d_[0][a]*gridN_[1]+work_gi1d_[1][b];
            nwDenominator_[g] += wNk;
            nwNumerator_[g*2+0] += wNk*kernels_[kk].mu[0];
            nwNumerator_[g*2+1] += wNk*kernels_[kk].mu[1];
          }
        }
      } else {
        for (unsigned a = 0; a < work_w1d_[0].size(); ++a) {
          double wa = work_w1d_[0][a];
          for (unsigned b = 0; b < work_w1d_[1].size(); ++b) {
            double wab = wa * work_w1d_[1][b];
            if (wab < sphereCut) continue;  // early exit
            unsigned gab = (work_gi1d_[0][a]*gridN_[1]+work_gi1d_[1][b])*gridN_[2];
            for (unsigned c = 0; c < work_w1d_[2].size(); ++c) {
              double wabc = wab*work_w1d_[2][c];
              if (wabc < sphereCut) continue;  // ellipsoidal cutoff
              double wNk = wabc * Nk_kk;
              unsigned g = gab + work_gi1d_[2][c];
              nwDenominator_[g] += wNk;
              nwNumerator_[g*3+0] += wNk*kernels_[kk].mu[0];
              nwNumerator_[g*3+1] += wNk*kernels_[kk].mu[1];
              nwNumerator_[g*3+2] += wNk*kernels_[kk].mu[2];
            }
          }
        }
      }
    }
  }

  // ================ incremental grid update (process dirty entries) ==============
  // For each dirty kernel: subtract old contribution (if any), add current (if alive).
  void incrementalGridUpdate() {
    for (auto it = dirty_.begin(); it != dirty_.end(); ++it) {
      uint64_t id = it->first;
      const DirtyEntry& entry = it->second;
      // Subtract old contribution from grid (if kernel was on the published grid)
      if (entry.has_old)
        splatKernel(entry.old_center, entry.old_sigma, entry.old_mu, entry.old_Nk, -1.0);
      // Add current contribution (if kernel still exists)
      auto kit = idMap_.find(id);
      if (kit != idMap_.end()) {
        unsigned idx = kit->second;
        splatKernel(kernels_[idx].center, kernels_[idx].sigma,
                    kernels_[idx].mu, kernels_[idx].Nk, +1.0);
      }
    }
    dirty_.clear();
  }

  // ================ grid accumulation update ====================================
  // Updates the persistent NW numerator (nwNumerator_) and denominator (nwDenominator_) either
  // incrementally (from dirty entries) or via full rebuild (for floating point hygiene).
  // On full rebuild, logs the max deviation from the incremental state as a diagnostic.
  void updateKernelAccumulationGrid() {
    ++gridRebuildCount_;
    bool doFullRebuild = (gridRebuildCount_ >= fullRebuildInterval_ || nwNumerator_.empty());

    if (doFullRebuild) {
      // Apply any pending incremental updates first so we can compare
      // the fully-caught-up incremental state against a from-scratch rebuild.
      bool hadIncrementalState = (!nwNumerator_.empty() && nwNumerator_.size() == gridTotal_ * dim_);
      if (hadIncrementalState && !dirty_.empty())
        incrementalGridUpdate();  // applies and clears dirty_

      // Save incremental state for comparison (skip on first call when grid is zero)
      double maxDevZ = 0.0, maxDevF = 0.0;
      std::vector<double> oldZ, oldF;
      if (hadIncrementalState && nKernels_ > 0) {
        oldZ = nwDenominator_;
        oldF = nwNumerator_;
      }

      // Full rebuild from scratch
      fullGridRebuild();
      dirty_.clear();
      gridRebuildCount_ = 0;
      lastRebuildWasFull_ = true;

      // Compute and log max deviation (only if we had a prior incremental state)
      if (!oldZ.empty()) {
        for (unsigned g = 0; g < gridTotal_; ++g) {
          maxDevZ = std::max(maxDevZ, std::abs(nwDenominator_[g] - oldZ[g]));
          for (unsigned d = 0; d < dim_; ++d)
            maxDevF = std::max(maxDevF, std::abs(nwNumerator_[g*dim_+d] - oldF[g*dim_+d]));
        }
        log.printf("  [FKERNELABF] Full rebuild FP drift: maxDevZ=%.2e maxDevF=%.2e\n",
                   maxDevZ, maxDevF);
      }
    } else if (!dirty_.empty()) {
      incrementalGridUpdate();
      lastRebuildWasFull_ = false;
    } else {
      lastRebuildWasFull_ = false;
    }
  }

  // ================ mean force finalization ====================================
  // Computes meanForceGrid_ = clamp(nwNumerator_ / nwDenominator_) at every grid node.
  void finalizeMeanForceGrid() {
    meanForceGrid_.assign(gridTotal_*dim_, 0.0);
    for (unsigned g = 0; g < gridTotal_; ++g) {
      if (nwDenominator_[g] > 1e-300) {
        for (unsigned d = 0; d < dim_; ++d) {
          double v = nwNumerator_[g*dim_+d] / nwDenominator_[g];
          if (maxForce_ > 0) v = std::max(-maxForce_, std::min(maxForce_, v));
          meanForceGrid_[g*dim_+d] = v;
        }
      }
    }
  }

  // ================ exploration force finalization ==============================
  // Computes Z₀ = median(Z) and explorationForceGrid_ = −c·∇Z/(Z₀+Z) via finite differences.
  // Only active when γ > 1; otherwise explorationForceGrid_ is zeroed.
  void finalizeExplorationGrid() {
    explorationForceGrid_.assign(gridTotal_*dim_, 0.0);
    if (biasFactor_ > 1.0) {
      std::vector<double> Zpop;
      Zpop.reserve(gridTotal_);
      for (unsigned g = 0; g < gridTotal_; ++g)
        if (nwDenominator_[g] > 1e-10) Zpop.push_back(nwDenominator_[g]);
      if (!Zpop.empty()) {
        size_t n = Zpop.size();
        std::nth_element(Zpop.begin(), Zpop.begin() + n/2, Zpop.end());
        if (n % 2 == 1) {
          Z0_density_ = Zpop[n/2];
        } else {
          double upper = Zpop[n/2];
          double lower = *std::max_element(Zpop.begin(), Zpop.begin() + n/2);
          Z0_density_ = 0.5 * (lower + upper);
        }
      }
      if (Z0_density_ < 1e-10) Z0_density_ = 1.0;

      unsigned strides[3]; strides[dim_-1] = 1;
      for (int dd = (int)dim_-2; dd >= 0; --dd)
        strides[dd] = strides[dd+1] * gridN_[dd+1];

      double c_ex = kT_ * (biasFactor_ - 1.0);

      for (unsigned g = 0; g < gridTotal_; ++g) {
        if (nwDenominator_[g] < 1e-300) continue;
        double denom = Z0_density_ + nwDenominator_[g];

        gridUnflat(g, work_idx_);

        for (unsigned d = 0; d < dim_; ++d) {
          double dZds = 0.0;
          unsigned N_d = gridN_[d];
          unsigned stride_d = strides[d];

          if (periodic_[d]) {
            unsigned gp = g + ((work_idx_[d]+1 < N_d) ? stride_d : -(N_d-1)*stride_d);
            unsigned gm = g - ((work_idx_[d] > 0) ? stride_d : -(N_d-1)*stride_d);
            dZds = (nwDenominator_[gp] - nwDenominator_[gm]) / (2.0 * gridDx_[d]);
          } else if (work_idx_[d] == 0) {
            if (N_d >= 3)
              dZds = (-3.0*nwDenominator_[g] + 4.0*nwDenominator_[g+stride_d] - nwDenominator_[g+2*stride_d]) / (2.0*gridDx_[d]);
            else
              dZds = (nwDenominator_[g+stride_d] - nwDenominator_[g]) / gridDx_[d];
          } else if (work_idx_[d] == N_d-1) {
            if (N_d >= 3)
              dZds = (3.0*nwDenominator_[g] - 4.0*nwDenominator_[g-stride_d] + nwDenominator_[g-2*stride_d]) / (2.0*gridDx_[d]);
            else
              dZds = (nwDenominator_[g] - nwDenominator_[g-stride_d]) / gridDx_[d];
          } else {
            dZds = (nwDenominator_[g+stride_d] - nwDenominator_[g-stride_d]) / (2.0 * gridDx_[d]);
          }

          explorationForceGrid_[g*dim_+d] = -c_ex * dZds / denom;
        }
      }
    }
  }

  // ================ top-level grid rebuild (called at GRIDPACE) ================
  // Coordinates the three phases and logs diagnostics.
  // Between rebuilds, the total force on λ is completely frozen — ideal for BAOAB.
  void reconstructBiasGrid() {
    updateKernelAccumulationGrid();
    finalizeMeanForceGrid();
    finalizeExplorationGrid();

    const std::vector<double>& sig = currentSigma();
    log.printf("  [FKERNELABF] Grid update step %lld (%s): M=%u Ntot=%.0f zM=%u zNtot=%.0f",
               (long long)getStep(), lastRebuildWasFull_ ? "full" : "incremental",
               nKernels_, totalN_, nZKernels_, zTotalN_);
    if (biasFactor_ > 1.0)
      log.printf(" Z0_density=%.2f", Z0_density_);
    log.printf(" sigma=(%.4f", sig[0]);
    for (unsigned i = 1; i < dim_; ++i) log.printf(",%.4f", sig[i]);
    log.printf(")\n");
  }

  // Shared interpolation cell computation. Writes lo/frac into member work arrays.
  void prepareInterp(const std::vector<double>& s) {
    for (unsigned d = 0; d < dim_; ++d) {
      double f = (s[d] - gridMin_[d]) / gridDx_[d];
      if (periodic_[d]) {
        f -= std::floor(f/(double)gridN_[d]) * (double)gridN_[d];
        work_lo_[d] = (unsigned)std::floor(f);
        if (work_lo_[d] >= gridN_[d]) work_lo_[d] = gridN_[d]-1;
      } else {
        if (f < 0.0) f = 0.0;
        if (f >= (double)(gridN_[d]-1)) f = (double)(gridN_[d]-1)-1e-12;
        work_lo_[d] = (unsigned)std::floor(f);
        if (work_lo_[d] >= gridN_[d]-1) work_lo_[d] = gridN_[d]-2;
      }
      work_frac_[d] = f - (double)work_lo_[d];
    }
  }

  // Multilinear interpolation of a vector-valued grid [gridTotal_ * dim_].
  // Used for both meanForceGrid_ and explorationForceGrid_.
  void interpolateVectorGrid(const std::vector<double>& s,
                             const std::vector<double>& grid,
                             std::vector<double>& force) {
    force.assign(dim_, 0.0);
    if (grid.empty()) return;
    prepareInterp(s);
    unsigned nC = 1u << dim_;
    for (unsigned c = 0; c < nC; ++c) {
      unsigned ci[3]; double w = 1.0;
      for (unsigned d = 0; d < dim_; ++d) {
        bool hi = (c >> d) & 1;
        ci[d] = periodic_[d] ? (hi ? (work_lo_[d]+1)%gridN_[d] : work_lo_[d]) : (hi ? work_lo_[d]+1 : work_lo_[d]);
        w *= hi ? work_frac_[d] : (1.0-work_frac_[d]);
      }
      unsigned g = 0;
      for (unsigned d = 0; d < dim_; ++d) g = g * gridN_[d] + ci[d];
      for (unsigned d = 0; d < dim_; ++d) force[d] += w * grid[g*dim_+d];
    }
  }

  // Multilinear interpolation of a scalar grid (used for nwDenominator_ -> V_ex).
  double interpolateScalar(const std::vector<double>& s, const std::vector<double>& grid) {
    if (grid.empty()) return 0.0;
    prepareInterp(s);
    unsigned nC = 1u << dim_;
    double val = 0.0;
    for (unsigned c = 0; c < nC; ++c) {
      unsigned ci[3]; double w = 1.0;
      for (unsigned d = 0; d < dim_; ++d) {
        bool hi = (c >> d) & 1;
        ci[d] = periodic_[d] ? (hi ? (work_lo_[d]+1)%gridN_[d] : work_lo_[d]) : (hi ? work_lo_[d]+1 : work_lo_[d]);
        w *= hi ? work_frac_[d] : (1.0-work_frac_[d]);
      }
      unsigned g = 0;
      for (unsigned d = 0; d < dim_; ++d) g = g * gridN_[d] + ci[d];
      val += w * grid[g];
    }
    return val;
  }

  // ================ lambda-grid output (DEBUG: mean force on λ grid) ========
  void writeGridFile() {
    if (lambdaGridFile_.empty()) return;
    std::string path = stampedPath(lambdaGridFile_);
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return;

    std::fprintf(f, "# NW mean force on lambda grid (DEBUG)\n");
    std::fprintf(f, "# Generated at step %lld  M=%u\n", (long long)getStep(), nKernels_);
    std::fprintf(f, "#! FIELDS");
    for (unsigned d = 0; d < dim_; ++d) std::fprintf(f, " s%u", d);
    for (unsigned d = 0; d < dim_; ++d) std::fprintf(f, " ghat%u", d);
    std::fprintf(f, " Z_nw\n");

    std::vector<unsigned> idx(dim_);
    std::vector<double> s(dim_), ghat(dim_);
    for (unsigned g = 0; g < gridTotal_; ++g) {
      gridUnflat(g, idx);
      for (unsigned d = 0; d < dim_; ++d) s[d] = gridCoord(d, idx[d]);

      double Z = 0.0;
      ghat.assign(dim_, 0.0);
      for (unsigned kk = 0; kk < nKernels_; ++kk) {
        double d2 = 0.0;
        for (unsigned i = 0; i < dim_; ++i) {
          double dd = periodicDelta(i, kernels_[kk].center[i], s[i]);
          d2 += sq(dd) / (4.0*sq(kernels_[kk].sigma[i]) + 1e-300);
        }
        if (d2 > 0.25*sq(nsigmaCut_)) continue;
        double w = std::exp(-d2);
        if (w < 1e-300) continue;
        // KDE normalization: alpha_k = prod(sigma0/sigma_k)
        double alpha_kk = 1.0;
        for (unsigned i = 0; i < dim_; ++i)
          alpha_kk *= sigma0_[i] / (kernels_[kk].sigma[i] + 1e-300);
        double wNk = w * kernels_[kk].Nk * alpha_kk;
        Z += wNk;
        for (unsigned i = 0; i < dim_; ++i) ghat[i] += wNk*kernels_[kk].mu[i];
      }
      double Z_nw = 0.0;
      if (Z > 0) {
        for (unsigned i = 0; i < dim_; ++i) ghat[i] /= Z;
        Z_nw = Z;
      }

      for (unsigned d = 0; d < dim_; ++d) std::fprintf(f, " %14.6f", s[d]);
      for (unsigned d = 0; d < dim_; ++d) std::fprintf(f, " %14.6f", ghat[d]);
      std::fprintf(f, " %14.6f\n", Z_nw);
    }
    std::fclose(f);
    log.printf("  [FKERNELABF] Lambda-grid written: %s  (M=%u, step %lld)\n",
               path.c_str(), nKernels_, (long long)getStep());
  }

  // ================ kernel diagnostics file ================
  // Appends one line per stride to {label}.kernelinfo.dat via PLUMED OFile.
  // OFile handles bck.* backups on fresh runs and append-on-restart automatically.
  // PLUMED-compatible #! FIELDS header for use with fkabf_analysis.py.
  void writeKernelInfo() {
    if (!kernelInfoFileOpen_) return;
    if (getStep() % kernelInfoStride_ != 0) return;

    const double neff = (sumNk2_ > 0.0) ? (totalN_*totalN_/sumNk2_) : (double)nKernels_;
    const std::vector<double>& sig = currentSigma();

    kernelInfoOFile_.printf("%lld %u %u %.4f", (long long)getStep(), nKernels_, nZKernels_, neff);
    for (unsigned i = 0; i < dim_; ++i)
      kernelInfoOFile_.printf(" %.6f", sig[i]);
    kernelInfoOFile_.printf(" %zu\n", nlistIdx_.size());
    kernelInfoOFile_.flush();
  }

  // ================ lambda-kernel dump ================
  // Human-readable columnar format. Each call writes a fresh step-stamped file.
  // Columns use CV names, plus derived quantities:
  //   |mu|   -- force magnitude (kJ/mol per CV unit)
  //   wt     -- Nk / totalN (fractional weight of this kernel)
  void dumpKernelsIfNeeded() {
    if (kernelFile_.empty() || kernelStride_ == 0) return;
    if (getStep() % kernelStride_ != 0) return;
    std::string path = stampedPath(kernelFile_);
    std::FILE* kfile = std::fopen(path.c_str(), "w");
    if (!kfile) return;

    const std::vector<double>& sig = currentSigma();
    const double neff_sil = (sumNk2_ > 0.0) ? (totalN_*totalN_/sumNk2_) : (double)nKernels_;
    std::fprintf(kfile, "# ========================================================\n");
    std::fprintf(kfile, "# Lambda-kernel snapshot  step=%-10lld  M=%-6u  totalN=%-8.0f\n",
                 (long long)getStep(), nKernels_, totalN_);
    std::fprintf(kfile, "#   neff(Silverman)=%-8.1f  sigma=(", neff_sil);
    for (unsigned i = 0; i < dim_; ++i) std::fprintf(kfile, "%s%.4f", i?",":"", sig[i]);
    std::fprintf(kfile, ")\n");
    std::fprintf(kfile, "# --------------------------------------------------------\n");

    std::vector<std::string> cvNames(dim_);
    for (unsigned i = 0; i < dim_; ++i) cvNames[i] = getPntrToArgument(i)->getName();

    std::fprintf(kfile, "# %5s  %8s", "k", "Nk");
    for (unsigned i = 0; i < dim_; ++i)
      std::fprintf(kfile, "  %12s", ("c_"+cvNames[i]).c_str());
    for (unsigned i = 0; i < dim_; ++i)
      std::fprintf(kfile, "  %12s", ("mu_"+cvNames[i]).c_str());
    for (unsigned i = 0; i < dim_; ++i)
      std::fprintf(kfile, "  %10s", ("sig_"+cvNames[i]).c_str());
    std::fprintf(kfile, "  %10s  %8s\n", "|mu|", "wt");

    for (unsigned kk = 0; kk < nKernels_; ++kk) {
      double mu2 = 0;
      for (unsigned i = 0; i < dim_; ++i) mu2 += sq(kernels_[kk].mu[i]);
      double mu_mag = std::sqrt(mu2);
      double wt_val = (totalN_ > 0) ? kernels_[kk].Nk / totalN_ : 0.0;

      std::fprintf(kfile, "  %5u  %8.1f", kk, kernels_[kk].Nk);
      for (unsigned i = 0; i < dim_; ++i) std::fprintf(kfile, "  %12.6f", kernels_[kk].center[i]);
      for (unsigned i = 0; i < dim_; ++i) std::fprintf(kfile, "  %12.4f", kernels_[kk].mu[i]);
      for (unsigned i = 0; i < dim_; ++i) std::fprintf(kfile, "  %10.6f", kernels_[kk].sigma[i]);
      std::fprintf(kfile, "  %10.4f  %8.5f\n", mu_mag, wt_val);
    }
    std::fclose(kfile);
    log.printf("  [FKERNELABF] Kernel file written: %s  (M=%u, step %lld)\n",
               path.c_str(), nKernels_, (long long)getStep());
  }

  // ================ stamped filename helper ================
  // Inserts _{step:08d} before the extension of any base filename.
  // e.g. "fk.czar_kernels.dat" at step 50000 -> "fk.czar_kernels_00050000.dat"
  // All three output files use this so snapshots are never overwritten.
  std::string stampedPath(const std::string& base) const {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "_%08lld", (long long)getStep());
    auto dot = base.rfind('.');
    if (dot == std::string::npos) return base + buf;
    return base.substr(0, dot) + buf + base.substr(dot);
  }

  // ================ CZAR z-kernel file ================
  // Step-stamped snapshot of all z-kernels; never overwritten between writes.
  // The file header embeds kappa, kT, periodicity, and domain so that
  // czar_integrate can reconstruct A(z) without access to the input file.
  void writeCZARFile() {
    if (czarFile_.empty() || nZKernels_ == 0) return;
    std::string path = stampedPath(czarFile_);
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return;

    std::fprintf(f, "# CZAR z-kernel file -- ForceKernelABF\n");
    std::fprintf(f, "# Generated at step %lld\n", (long long)getStep());
    std::fprintf(f, "#\n");
    std::fprintf(f, "# KEY METADATA\n");
    std::fprintf(f, "dim %u\n", dim_);
    std::fprintf(f, "kT %.15g\n", kT_);
    std::fprintf(f, "kappa");
    for (unsigned i = 0; i < dim_; ++i) std::fprintf(f, " %.15g", kappa_[i]);
    std::fprintf(f, "\n");
    std::fprintf(f, "periodic");
    for (unsigned i = 0; i < dim_; ++i) std::fprintf(f, " %d", (int)periodic_[i]);
    std::fprintf(f, "\n");
    std::fprintf(f, "domMin");
    for (unsigned i = 0; i < dim_; ++i) std::fprintf(f, " %.15g", domMin_[i]);
    std::fprintf(f, "\n");
    std::fprintf(f, "domMax");
    for (unsigned i = 0; i < dim_; ++i) std::fprintf(f, " %.15g", domMax_[i]);
    std::fprintf(f, "\n");
    std::fprintf(f, "sigma0");
    for (unsigned i = 0; i < dim_; ++i) std::fprintf(f, " %.15g", sigma0_[i]);
    std::fprintf(f, "\n");
    std::fprintf(f, "nkernels %u\n", nZKernels_);
    std::fprintf(f, "#\n");
    std::fprintf(f, "# COLUMNS: Nk");
    for (unsigned i = 0; i < dim_; ++i) std::fprintf(f, " center%u", i);
    for (unsigned i = 0; i < dim_; ++i) std::fprintf(f, " mu%u", i);
    for (unsigned i = 0; i < dim_; ++i) std::fprintf(f, " sigma%u", i);
    std::fprintf(f, "\n");
    std::fprintf(f, "# NOTE: mu stores kappa*(z - lambda) [unclamped, exact mean].\n");
    std::fprintf(f, "# CZAR gradient: dA/dz_i = -mu_NW_i(z) - kT * d/dz_i ln ptilde(z)\n");
    std::fprintf(f, "# (minus kT: derives from rho_tilde = C*exp(-beta*A)*Q(z), see DRR CZAR::getGradient)\n");
    std::fprintf(f, "#\n");
    for (unsigned k = 0; k < nZKernels_; ++k) {
      std::fprintf(f, "%.6f", zKernels_[k].Nk);
      for (unsigned i = 0; i < dim_; ++i) std::fprintf(f, " %.10f", zKernels_[k].center[i]);
      for (unsigned i = 0; i < dim_; ++i) std::fprintf(f, " %.10f", zKernels_[k].mu[i]);
      for (unsigned i = 0; i < dim_; ++i) std::fprintf(f, " %.10f", zKernels_[k].sigma[i]);
      std::fprintf(f, "\n");
    }
    std::fclose(f);
    log.printf("  [FKERNELABF] CZAR file written: %s  (%u z-kernels, step %lld)\n",
               path.c_str(), nZKernels_, (long long)getStep());
  }

  // ================ restart state I/O ================
  // The state file contains everything needed to resume a simulation:
  //   - fictitious particle position and velocity
  //   - adaptive sigma state
  //   - lambda-kernel population (center, mu, sigma, Nk, id)
  //   - z-kernel population (center, mu, sigma, Nk, id)
  //   - stable ID counters (nextKernelId, nextZKernelId)
  //   - Z0_density for exploration
  //   - RNG state for reproducible restart
  void writeState() {
    if (stateFile_.empty()) return;
    // Write to a temporary file then atomically rename to avoid corruption
    // if the process is killed mid-write.
    std::string tmpFile = stateFile_ + ".tmp";
    std::ofstream f(tmpFile.c_str());
    if (!f.is_open()) {
      log.printf("  [FKERNELABF] WARNING: could not open state file '%s' for writing.\n",
                 tmpFile.c_str());
      return;
    }
    f << std::setprecision(15);

    f << "# FKERNELABF state file\n";
    f << "# Written at step " << getStep() << "\n";
    f << "#\n";

    f << "dim " << dim_ << "\n";
    f << "kT " << kT_ << "\n";
    f << "s_fict";
    for (unsigned i = 0; i < dim_; ++i) f << " " << s_fict_[i];
    f << "\n";
    f << "v_fict";
    for (unsigned i = 0; i < dim_; ++i) f << " " << v_fict_[i];
    f << "\n";
    f << "sigma0";
    for (unsigned i = 0; i < dim_; ++i) f << " " << sigma0_[i];
    f << "\n";
    f << "adaptive_done " << (adaptiveSigma_ ? 0 : 1) << "\n";
    f << "Z0_density " << Z0_density_ << "\n";

    f << "M " << nKernels_ << "\n";
    f << "totalN " << totalN_ << "\n";
    f << "sumNk2 " << sumNk2_ << "\n";
    f << "nextKernelId " << nextKernelId_ << "\n";
    f << "# lambda-kernels: Nk center[0..d-1] mu[0..d-1] sigma[0..d-1] id\n";
    for (unsigned k = 0; k < nKernels_; ++k) {
      f << "K " << kernels_[k].Nk;
      for (unsigned i = 0; i < dim_; ++i) f << " " << kernels_[k].center[i];
      for (unsigned i = 0; i < dim_; ++i) f << " " << kernels_[k].mu[i];
      for (unsigned i = 0; i < dim_; ++i) f << " " << kernels_[k].sigma[i];
      f << " " << kernels_[k].id;
      f << "\n";
    }

    f << "zM " << nZKernels_ << "\n";
    f << "zTotalN " << zTotalN_ << "\n";
    f << "zSumNk2 " << zSumNk2_ << "\n";
    f << "nextZKernelId " << nextZKernelId_ << "\n";
    f << "# z-kernels: Nk center[0..d-1] mu[0..d-1] sigma[0..d-1] id\n";
    for (unsigned k = 0; k < nZKernels_; ++k) {
      f << "Z " << zKernels_[k].Nk;
      for (unsigned i = 0; i < dim_; ++i) f << " " << zKernels_[k].center[i];
      for (unsigned i = 0; i < dim_; ++i) f << " " << zKernels_[k].mu[i];
      for (unsigned i = 0; i < dim_; ++i) f << " " << zKernels_[k].sigma[i];
      f << " " << zKernels_[k].id;
      f << "\n";
    }

    // RNG state for reproducible restart.
    // std::mt19937 supports << / >> for full internal state serialization.
    f << "RNG " << rng_ << "\n";

    f << "END\n";

    f.flush();
    if (!f.good()) {
      log.printf("  [FKERNELABF] WARNING: I/O error writing state file '%s'.\n",
                 tmpFile.c_str());
      return;
    }
    f.close();
    if (std::rename(tmpFile.c_str(), stateFile_.c_str()) != 0)
      log.printf("  [FKERNELABF] WARNING: could not rename '%s' to '%s'.\n",
                 tmpFile.c_str(), stateFile_.c_str());
    else
      log.printf("  [FKERNELABF] State written: %s  (M=%u, zM=%u, step %lld)\n",
                 stateFile_.c_str(), nKernels_, nZKernels_, (long long)getStep());
  }

  void readState() {
    if (stateFile_.empty()) return;
    IFile f; f.link(*this);
    if (!f.FileExist(stateFile_)) {
      log.printf("  [FKERNELABF] WARNING: RESTART requested but state file '%s' not found. "
                 "Starting from scratch.\n", stateFile_.c_str());
      return;
    }
    f.open(stateFile_);
    log.printf("  [FKERNELABF] Reading state from: %s\n", stateFile_.c_str());

    // Clear existing kernel data
    kernels_.clear(); nKernels_ = 0; totalN_ = 0; sumNk2_ = 0;
    zKernels_.clear(); nZKernels_ = 0; zTotalN_ = 0; zSumNk2_ = 0;

    std::string line;
    while (f.getline(line)) {
      if (line.empty() || line[0] == '#') continue;

      std::istringstream iss(line);
      std::string key;
      iss >> key;

      if (key == "dim") {
        unsigned file_dim; iss >> file_dim;
        if (file_dim != dim_)
          error("RESTART state file has dim=" + std::to_string(file_dim)
                + " but current input has " + std::to_string(dim_) + " CVs. "
                "Cannot restart with different dimensionality.");
      } else if (key == "kT") {
        double file_kT; iss >> file_kT;
        double rel = std::abs(file_kT - kT_) / (kT_ + 1e-300);
        if (rel > 1e-6)
          error("RESTART state file has kT=" + std::to_string(file_kT)
                + " but current input gives kT=" + std::to_string(kT_)
                + ". Cannot restart with different temperature.");
      } else if (key == "s_fict") {
        for (unsigned i = 0; i < dim_; ++i) iss >> s_fict_[i];
        if (!iss) error("RESTART state file: failed to parse 's_fict' — file may be truncated or corrupted.");
      } else if (key == "v_fict") {
        for (unsigned i = 0; i < dim_; ++i) iss >> v_fict_[i];
        if (!iss) error("RESTART state file: failed to parse 'v_fict' — file may be truncated or corrupted.");
      } else if (key == "sigma0") {
        for (unsigned i = 0; i < dim_; ++i) iss >> sigma0_[i];
        if (!iss) error("RESTART state file: failed to parse 'sigma0' — file may be truncated or corrupted.");
      } else if (key == "adaptive_done") {
        int done; iss >> done;
        if (!iss) error("RESTART state file: failed to parse 'adaptive_done' — file may be truncated or corrupted.");
        if (done) {
          adaptiveSigma_ = false;
          adaptiveCounter_ = adaptiveSigmaStride_ + 1;
        }
      } else if (key == "Z0_density") {
        iss >> Z0_density_;
        if (!iss) error("RESTART state file: failed to parse 'Z0_density' — file may be truncated or corrupted.");
      } else if (key == "M") {
        iss >> nKernels_;
      } else if (key == "totalN") {
        iss >> totalN_;
        if (!iss) error("RESTART state file: failed to parse 'totalN' — file may be truncated or corrupted.");
      } else if (key == "sumNk2") {
        iss >> sumNk2_;
        if (!iss) error("RESTART state file: failed to parse 'sumNk2' — file may be truncated or corrupted.");
      } else if (key == "zM") {
        iss >> nZKernels_;
      } else if (key == "zTotalN") {
        iss >> zTotalN_;
        if (!iss) error("RESTART state file: failed to parse 'zTotalN' — file may be truncated or corrupted.");
      } else if (key == "zSumNk2") {
        iss >> zSumNk2_;
        if (!iss) error("RESTART state file: failed to parse 'zSumNk2' — file may be truncated or corrupted.");
      } else if (key == "nextKernelId") {
        iss >> nextKernelId_;
      } else if (key == "nextZKernelId") {
        iss >> nextZKernelId_;
      } else if (key == "K") {
        // Lambda-kernel data line: Nk center[0..d-1] mu[0..d-1] sigma[0..d-1] id
        Kernel nk;
        iss >> nk.Nk;
        for (unsigned i = 0; i < dim_; ++i) iss >> nk.center[i];
        for (unsigned i = 0; i < dim_; ++i) iss >> nk.mu[i];
        for (unsigned i = 0; i < dim_; ++i) iss >> nk.sigma[i];
        iss >> nk.id;
        if (!iss) error("RESTART state file: failed to parse kernel 'K' row — file may be truncated or corrupted.");
        if (nk.Nk <= 0)
          error("RESTART state file: kernel 'K' has Nk=" + std::to_string(nk.Nk)
                + " <= 0. Corrupted state file.");
        if (nk.id == 0)
          error("RESTART state file: kernel 'K' has id=0 — state file may be from an older format without stable IDs.");
        for (unsigned i = 0; i < dim_; ++i)
          if (nk.sigma[i] <= 0.0 || !std::isfinite(nk.sigma[i]))
            error("RESTART state file: kernel 'K' has sigma[" + std::to_string(i)
                  + "]=" + std::to_string(nk.sigma[i]) + " which is non-positive or non-finite. Corrupted state file.");
        for (unsigned i = 0; i < dim_; ++i)
          if (!std::isfinite(nk.center[i]) || !std::isfinite(nk.mu[i]))
            error("RESTART state file: kernel 'K' has non-finite center or mu at dimension "
                  + std::to_string(i) + ". Corrupted state file.");
        kernels_.push_back(nk);
      } else if (key == "Z") {
        // Z-kernel data line: Nk center[0..d-1] mu[0..d-1] sigma[0..d-1] id
        ZKernel nk;
        iss >> nk.Nk;
        for (unsigned i = 0; i < dim_; ++i) iss >> nk.center[i];
        for (unsigned i = 0; i < dim_; ++i) iss >> nk.mu[i];
        for (unsigned i = 0; i < dim_; ++i) iss >> nk.sigma[i];
        iss >> nk.id;
        if (!iss) error("RESTART state file: failed to parse z-kernel 'Z' row — file may be truncated or corrupted.");
        if (nk.Nk <= 0)
          error("RESTART state file: z-kernel 'Z' has Nk=" + std::to_string(nk.Nk)
                + " <= 0. Corrupted state file.");
        if (nk.id == 0)
          error("RESTART state file: z-kernel 'Z' has id=0 — state file may be from an older format without stable IDs.");
        for (unsigned i = 0; i < dim_; ++i)
          if (nk.sigma[i] <= 0.0 || !std::isfinite(nk.sigma[i]))
            error("RESTART state file: z-kernel 'Z' has sigma[" + std::to_string(i)
                  + "]=" + std::to_string(nk.sigma[i]) + " which is non-positive or non-finite. Corrupted state file.");
        for (unsigned i = 0; i < dim_; ++i)
          if (!std::isfinite(nk.center[i]) || !std::isfinite(nk.mu[i]))
            error("RESTART state file: z-kernel 'Z' has non-finite center or mu at dimension "
                  + std::to_string(i) + ". Corrupted state file.");
        zKernels_.push_back(nk);
      } else if (key == "RNG") {
        // Restore full mt19937 state for reproducible restart.
        iss >> rng_;
      } else if (key == "END") {
        break;
      }
      // Unknown keys are silently skipped for forward compatibility.
    }

    // Validate
    if (kernels_.size() != nKernels_) {
      log.printf("  [FKERNELABF] WARNING: state file M=%u but read %lu lambda-kernels. "
                 "Using actual count.\n", nKernels_, (unsigned long)kernels_.size());
      nKernels_ = kernels_.size();
    }
    if (zKernels_.size() != nZKernels_) {
      log.printf("  [FKERNELABF] WARNING: state file zM=%u but read %lu z-kernels. "
                 "Using actual count.\n", nZKernels_, (unsigned long)zKernels_.size());
      nZKernels_ = zKernels_.size();
    }
    if (nKernels_ > 0 && zKernels_.empty())
      log.printf("  [FKERNELABF] WARNING: lambda-kernels restored but no z-kernels found. "
                 "CZAR history is lost; z-kernels will be rebuilt from scratch.\n");

    // Rebuild id→index maps and advance ID counters past all loaded IDs
    idMap_.clear();
    for (unsigned k = 0; k < nKernels_; ++k) {
      idMap_[kernels_[k].id] = k;
      if (kernels_[k].id >= nextKernelId_) nextKernelId_ = kernels_[k].id + 1;
    }

    zIdMap_.clear();
    for (unsigned k = 0; k < nZKernels_; ++k) {
      zIdMap_[zKernels_[k].id] = k;
      if (zKernels_[k].id >= nextZKernelId_) nextZKernelId_ = zKernels_[k].id + 1;
    }

    // Recompute aggregate scalars from actual kernel data (robust to truncation
    // and resets any floating-point drift in the running sums).
    totalN_ = 0; sumNk2_ = 0;
    for (unsigned k = 0; k < nKernels_; ++k) {
      totalN_ += kernels_[k].Nk;
      sumNk2_ += kernels_[k].Nk * kernels_[k].Nk;
    }
    zTotalN_ = 0; zSumNk2_ = 0;
    for (unsigned k = 0; k < nZKernels_; ++k) {
      zTotalN_ += zKernels_[k].Nk;
      zSumNk2_ += zKernels_[k].Nk * zKernels_[k].Nk;
    }

    // Clear dirty tracking — force a full grid rebuild on restart
    dirty_.clear();
    gridRebuildCount_ = fullRebuildInterval_;  // next reconstructBiasGrid triggers full rebuild

    // Populate neighbor list indices for all restored kernels
    if (nlist_) {
      nlistIdx_.clear();
      for (unsigned k = 0; k < nKernels_; ++k) nlistIdx_.push_back(k);
      nlistUpdate_ = true;
      znlistIdx_.clear();
      for (unsigned k = 0; k < nZKernels_; ++k) znlistIdx_.push_back(k);
      znlistUpdate_ = true;
    }

    // If adaptive sigma was completed before checkpoint, warmup is done
    // and sigma0_ from the state file is valid. If NOT completed
    // (adaptive_done=0), restart warmup from scratch rather than using
    // the placeholder sigma0_ values.
    if (adaptiveSigma_) {
      // adaptive_done=0 was read (or never seen): warmup was interrupted.
      adaptiveCounter_ = 0;
      av_cv_.assign(dim_, 0.0);
      av_M2_.assign(dim_, 0.0);
      log.printf("  [FKERNELABF] WARNING: adaptive sigma warmup was "
                 "incomplete at checkpoint. Restarting warmup from scratch.\n");
    }

    // Mark that first step should NOT re-initialise s_fict from z
    firstStep_ = false;

    // Invalidate cached bandwidths so they are recomputed from restored state
    sigDirty_ = true;
    zSigDirty_ = true;

    log.printf("  [FKERNELABF] State restored: M=%u totalN=%.0f zM=%u zTotalN=%.0f "
               "Z0=%.2f sigma0=(%.5f",
               nKernels_, totalN_, nZKernels_, zTotalN_, Z0_density_, sigma0_[0]);
    for (unsigned i = 1; i < dim_; ++i) log.printf(",%.5f", sigma0_[i]);
    log.printf(") s_fict=(%.5f", s_fict_[0]);
    for (unsigned i = 1; i < dim_; ++i) log.printf(",%.5f", s_fict_[i]);
    log.printf(")\n");
  }

public:
  static void registerKeywords(Keywords& keys) {
    Bias::registerKeywords(keys);
    keys.use("ARG");

    // Extended Lagrangian
    keys.add("compulsory", "KAPPA",   "Spring constant(s) for extended Lagrangian.");
    keys.add("compulsory", "TAU",     "0.5", "Oscillation period(s) (determines mass).");
    keys.add("compulsory", "FRICTION","10.0","Langevin friction (1/time_unit). One value or one per CV.");
    keys.add("compulsory", "TEMP",    "300.0","Temperature (K).");

    // Kernel parameters
    keys.add("optional", "SIGMA",
             "Initial kernel bandwidth(s). If omitted, set automatically from CV "
             "fluctuations measured over ADAPTIVE_SIGMA_STRIDE unbiased steps.");
    keys.add("optional", "ADAPTIVE_SIGMA_STRIDE",
             "Steps used to measure CV variance for automatic SIGMA (default 10*PACE). "
             "Ignored when SIGMA is supplied explicitly.");
    keys.add("optional",   "SIGMA_MIN",      "Minimum bandwidth(s).");
    keys.addFlag("FIXED_SIGMA", false,        "Fix sigma; do not apply Silverman shrinkage.");
    keys.add("compulsory", "PACE",    "5",   "Accumulate force sample every PACE steps.");
    keys.add("compulsory", "THRESH",  "1.0",
             "Kernel compression threshold (sigma-normalised). "
             "Kernels closer than THRESH×σ are merged. "
             "1.0 is the OPES standard; lower = more compression, higher = more kernels.");
    keys.add("compulsory", "NSIGMACUT", "4.0",
             "Kernel cutoff in sigma units per dimension. "
             "Kernels further than NSIGMACUT×σ are ignored in NW regression. "
             "4.0 gives <2%% contribution at the cutoff boundary.");

    // Density-based exploration on λ
    keys.add("compulsory", "BIASFACTOR", "1.0",
             "Exploration factor γ (>= 1.0).  "
             "1.0 = pure ABF (no exploration).  "
             ">1 = density-based exploration: F_ex = −c·∇Z/(Z₀+Z) where "
             "c = kT·(γ−1), Z is the NW denominator on the grid, and "
             "Z₀ = median(Z).  All exploration quantities are computed "
             "on the grid at GRIDPACE intervals, synchronized with the "
             "ABF mean force.  Pushes λ away from well-sampled basins. "
             "The CZAR estimator on z is unaffected.");
    keys.add("optional", "EXPLORSCALE",
             "Per-CV scaling of the exploration force (default 1.0 for all CVs). "
             "One value or one per CV. Values in [0,1]: 1.0 = full exploration, "
             "0.0 = no exploration along that CV. The ABF cancellation force is "
             "unaffected by this scaling. Useful when one CV should be driven "
             "by exploration while another evolves naturally on the flattened surface.");

    // Force clamps (safety nets — defaults are generous for most systems)
    keys.add("compulsory", "MUXCLAMP", "500.0",
             "Per-kernel mean-force clamp (PLUMED internal units). Individual "
             "kernel mu values are hard-clamped to [-MUXCLAMP, +MUXCLAMP]. "
             "Only fires for corrupted force samples at very sparse regions.");
    keys.add("compulsory", "MAXFORCE", "500.0",
             "Grid mean-force clamp (PLUMED internal units). The NW mean force "
             "on the grid is clamped per-node before interpolation. Only fires "
             "for unphysically large force estimates.");

    // Grid
    keys.add("compulsory", "GRIDSIZE","72",  "Grid points per dimension.");
    keys.add("compulsory", "GRIDPACE","500", "Reconstruct mean-force grid every GRIDPACE steps.");
    keys.add("optional",   "GRIDMIN",        "Lower grid bound(s) for non-periodic CVs.");
    keys.add("optional",   "GRIDMAX",        "Upper grid bound(s) for non-periodic CVs.");

    // Neighbor list
    keys.addFlag("NONLIST", false, "Disable neighbor list.");
    keys.add("optional", "NLIST_PARAMETERS",
             "Two values: cutoff_factor (default=3.0) and skin_factor (default=0.5).");

    // Output files -- filenames are derived automatically from the action label:
    //   LAMBDAGRIDSTRIDE -> {label}.lambda_grid.dat   (debug: NW mean force on λ grid)
    //   KERNELSTRIDE     -> {label}.kernels.dat
    //   CZARSTRIDE       -> {label}.czar_kernels_{step}.dat
    keys.add("optional", "LAMBDAGRIDSTRIDE",
             "Write lambda mean-force debug grid every N steps. Output is the NW "
             "mean force on the fictitious variable grid -- NOT the free energy A(z). "
             "Off by default; use CZARSTRIDE + czar_integrate for FEL recovery.");
    keys.add("optional", "KERNELSTRIDE",
             "Write lambda-kernel state every N steps.");

    
    keys.add("optional", "CZARSTRIDE",
             "Write CZAR z-kernel file every N steps. Each write produces a "
             "step-stamped file {label}.czar_kernels_{step:08d}.dat. "
             "Feed the final file to czar_integrate to recover A(z).");

    // Restart state
    keys.add("optional", "STATESTRIDE",
             "Write restart state file every N steps (default: CZARSTRIDE if set, "
             "otherwise 10×GRIDPACE). The state file {label}.state.dat is overwritten "
             "in place (no backups). On RESTART, the state is read automatically.");
    keys.add("optional", "KERNELINFOSTRIDE",
             "Output stride for KERNELINFO file — always written, appending one line per "
             "write with M, zM, neff, sigma per CV dim, nlker. Default: PACE. Override "
             "to match your PRINT STRIDE (e.g. KERNELINFOSTRIDE=50).");

    // Output components
    keys.addOutputComponent("force2",    "default","squared net bias force magnitude on λ (ABF + exploration)");
    keys.addOutputComponent("wamp",      "default","exploration potential V_ex = c·ln(1+Z/Z₀) at s_fict (kJ/mol); 0 when γ=1");
    keys.addOutputComponent("_fict",     "default","fictitious variable position from extended Lagrangian");
  }

  explicit ForceKernelABF(const ActionOptions& ao)
    : PLUMED_BIAS_INIT(ao),
      dim_(0),
      kT_(0.0),
      firstStep_(true),
      adaptiveSigma_(false), adaptiveSigmaStride_(0), adaptiveCounter_(0),
      rng_(std::random_device{}()), gauss_(0.0, 1.0),
      pace_(5), thresh_(1.0), nsigmaCut_(4.0),
      biasFactor_(1.0), Z0_density_(1.0),
      muxClamp_(500), maxForce_(500),
      fixedSigma_(false),
      hasMin_(false),
      nKernels_(0), totalN_(0), sumNk2_(0.0),
      nZKernels_(0), zTotalN_(0.0), zSumNk2_(0.0),
      nextKernelId_(1), nextZKernelId_(1),
      fullRebuildInterval_(50), gridRebuildCount_(0),
      lastRebuildWasFull_(false),
      nlist_(true), nlistCutFactor_(3.0), nlistSkinFactor_(0.5),
      nlistUpdate_(true),
      znlistUpdate_(true),
      gridPace_(500), gridTotal_(0),
      lambdaGridStride_(0),
      kernelStride_(0),
      czarStride_(0),
      stateStride_(0),
      kernelInfoStride_(0),
      kernelInfoFileOpen_(false),
      sigDirty_(true), zSigDirty_(true),
      baoabReady_(false) {

    dim_ = getNumberOfArguments();
    if (dim_ < 1 || dim_ > 3)
      error("FKERNELABF supports 1 to 3 CVs.");

    // Initialize work vectors sized to dim_
    cachedSig_.resize(dim_, 0.0);
    cachedZSig_.resize(dim_, 0.0);
    work_cv_.resize(dim_); work_springF_.resize(dim_);
    work_s_.resize(dim_); work_f_.resize(dim_); work_z_.resize(dim_);
    work_gf_.resize(dim_); work_fe_.resize(dim_);
    work_gf2_.resize(dim_); work_fev_.resize(dim_);
    work_w1d_.resize(dim_); work_gi1d_.resize(dim_);
    work_idx_.resize(dim_);

    parseVector("KAPPA", kappa_);
    if (kappa_.empty()) error("KAPPA is required");
    if (kappa_.size() == 1) kappa_.assign(dim_, kappa_[0]);
    else if (kappa_.size() != dim_) error("KAPPA: one value or one per CV");

    std::vector<double> tau;
    parseVector("TAU", tau);
    if (tau.size() == 1) tau.assign(dim_, tau[0]);
    else if (tau.size() != dim_) error("TAU: one value or one per CV");

    for (unsigned i = 0; i < dim_; ++i) {
      if (kappa_[i] <= 0.0) error("KAPPA must be > 0 for all CVs.");
      if (tau[i] <= 0.0) error("TAU must be > 0 for all CVs.");
    }

    double temp = 300.0;
    parse("TEMP", temp);
    parseVector("FRICTION", friction_);
    if (friction_.size() == 1) friction_.assign(dim_, friction_[0]);
    else if (friction_.size() != dim_) error("FRICTION: one value or one per CV");

    if (temp <= 0.0) error("TEMP must be > 0.");
    for (unsigned i = 0; i < dim_; ++i)
      if (friction_[i] < 0.0) error("FRICTION must be >= 0 for all CVs.");
    mass_.resize(dim_);
    for (unsigned i = 0; i < dim_; ++i)
      mass_[i] = kappa_[i] * sq(tau[i]) / (4.0*sq(M_PI));

    for (unsigned i = 0; i < dim_; ++i) {
      if (!(mass_[i] > 0.0)) error("Derived mass is non-positive. Check KAPPA and TAU.");
    }

    double kB = plumed.getAtoms().getKBoltzmann();
    if (kB <= 0.0) kB = 8.314462618e-3;
    kT_ = kB * temp;
    if (kT_ <= 0.0) error("kT computed as non-positive. Check TEMP and Boltzmann constant.");

    s_fict_.assign(dim_, 0.0);
    v_fict_.assign(dim_, 0.0);

    // Parse kernel parameters early — PACE is needed for the ADAPTIVE_SIGMA_STRIDE default.
    parseFlag("FIXED_SIGMA", fixedSigma_);
    parse("PACE", pace_);
    parse("THRESH", thresh_);
    parse("NSIGMACUT", nsigmaCut_);
    if (thresh_ <= 0.0) error("THRESH must be > 0.");
    if (nsigmaCut_ <= 0.0) error("NSIGMACUT must be > 0.");

    parseVector("SIGMA", sigma0_);
    if (sigma0_.size() == 1) {
      sigma0_.assign(dim_, sigma0_[0]);
      adaptiveSigma_ = false;
    } else if (sigma0_.size() == dim_) {
      adaptiveSigma_ = false;
    } else if (sigma0_.empty()) {
      adaptiveSigma_ = true;
      sigma0_.assign(dim_, 1.0); // placeholder; overwritten after warmup
    } else {
      error("SIGMA: supply one value, one per CV, or omit entirely for auto-detection");
    }

    if (!adaptiveSigma_) {
      for (unsigned i = 0; i < dim_; ++i) {
        if (sigma0_[i] <= 0.0) error("SIGMA must be > 0 for all CVs.");
      }
    }
    adaptiveSigmaStride_ = 0;
    parse("ADAPTIVE_SIGMA_STRIDE", adaptiveSigmaStride_);
    if (adaptiveSigma_ && adaptiveSigmaStride_ == 0)
      adaptiveSigmaStride_ = 10 * pace_;  // 10 strides gives ~10× the per-step variance for a stable estimate
    if (!adaptiveSigma_ && adaptiveSigmaStride_ > 0)
      warning("ADAPTIVE_SIGMA_STRIDE ignored because SIGMA was provided explicitly.");

    parseVector("SIGMA_MIN", sigmaMin_);
    if (sigmaMin_.size() == 1) sigmaMin_.assign(dim_, sigmaMin_[0]);
    else if (sigmaMin_.size() > 0 && sigmaMin_.size() != dim_)
      error("SIGMA_MIN: one value or one per CV");

    if (!sigmaMin_.empty()) {
      for (unsigned i = 0; i < dim_; ++i) {
        if (sigmaMin_[i] <= 0.0) error("SIGMA_MIN must be > 0 for all CVs.");
      }
    }
    hasMin_ = !sigmaMin_.empty();


    parse("BIASFACTOR", biasFactor_);
    if (biasFactor_ < 1.0) error("BIASFACTOR must be >= 1.0 (1.0 = pure ABF).");

    parseVector("EXPLORSCALE", explorScale_);
    if (explorScale_.empty()) explorScale_.assign(dim_, 1.0);
    else if (explorScale_.size() == 1) explorScale_.assign(dim_, explorScale_[0]);
    else if (explorScale_.size() != dim_) error("EXPLORSCALE: one value or one per CV");
    for (unsigned i = 0; i < dim_; ++i) {
      if (explorScale_[i] < 0.0 || explorScale_[i] > 1.0)
        error("EXPLORSCALE must be in [0, 1] for all CVs.");
    }

    parse("MUXCLAMP", muxClamp_);
    parse("MAXFORCE", maxForce_);
    if (muxClamp_ <= 0.0) error("MUXCLAMP must be > 0.");
    if (maxForce_ <= 0.0) error("MAXFORCE must be > 0.");

    unsigned gridSize = 72;
    parse("GRIDSIZE", gridSize); parse("GRIDPACE", gridPace_);
    if (gridSize < 2) error("GRIDSIZE must be >= 2.");

    bool noNlist = false;
    parseFlag("NONLIST", noNlist);
    nlist_ = !noNlist;
    std::vector<double> nlistParam;
    parseVector("NLIST_PARAMETERS", nlistParam);
    if (nlistParam.size() >= 2) {
      nlistCutFactor_ = nlistParam[0];
      nlistSkinFactor_ = nlistParam[1];
    } else if (nlistParam.size() == 1) {
      error("NLIST_PARAMETERS requires two values: cutoff_factor, skin_factor");
    }

    // CZAR output -- filename derived from label
    czarStride_ = 0; parse("CZARSTRIDE", czarStride_);

    // Domain: getDomain() asserts the value is periodic before returning,
    // so guard with isPeriodic() first — non-periodic CVs (distances, RMSDs, etc.)
    // crash with "function should be periodic" otherwise.
    periodic_.assign(dim_, false);
    domMin_.assign(dim_, 0); domMax_.assign(dim_, 0); domLen_.assign(dim_, 0);
    for (unsigned i = 0; i < dim_; ++i) {
      if (getPntrToArgument(i)->isPeriodic()) {
        double mn, mx;
        getPntrToArgument(i)->getDomain(mn, mx);
        domMin_[i] = mn; domMax_[i] = mx;
        domLen_[i] = mx - mn;
        periodic_[i] = (domLen_[i] > 0);
      }
      // non-periodic: periodic_[i]=false, domMin/Max/Len left at 0
      // until GRIDMIN/GRIDMAX are parsed below
    }

    std::vector<double> userMin, userMax;
    parseVector("GRIDMIN", userMin);
    parseVector("GRIDMAX", userMax);
    gridMin_.resize(dim_); gridMax_.resize(dim_);
    for (unsigned i = 0; i < dim_; ++i) {
      if (periodic_[i]) {
        gridMin_[i] = domMin_[i]; gridMax_[i] = domMax_[i];
      } else {
        if (userMin.empty() || userMax.empty())
          error("GRIDMIN and GRIDMAX are required for non-periodic CV '"
                + getPntrToArgument(i)->getName() + "'.");
        if (userMin.size() != dim_ || userMax.size() != dim_)
          error("GRIDMIN and GRIDMAX must have one value per CV dimension.");
        gridMin_[i] = userMin[i]; gridMax_[i] = userMax[i];
        if (gridMax_[i] <= gridMin_[i])
          error("GRIDMAX must be > GRIDMIN for dim " + std::to_string(i));
        domMin_[i] = userMin[i]; domMax_[i] = userMax[i];
        domLen_[i] = userMax[i] - userMin[i];
      }
    }

    gridN_.assign(dim_, gridSize);
    gridDx_.resize(dim_);
    gridTotal_ = 1;
    for (unsigned d = 0; d < dim_; ++d) {
      gridDx_[d] = periodic_[d] ? domLen_[d]/(double)gridN_[d]
                                : (gridMax_[d]-gridMin_[d])/(double)(gridN_[d]-1);
      gridTotal_ *= gridN_[d];
    }
    meanForceGrid_.assign(gridTotal_*dim_, 0.0);
    nwDenominator_.assign(gridTotal_, 0.0);
    explorationForceGrid_.assign(gridTotal_*dim_, 0.0);
    nwNumerator_.assign(gridTotal_*dim_, 0.0);

    // Derive all output filenames from the action label.
    // Users only specify strides; paths are not configurable.
    std::string lbl = getLabel();
    lambdaGridFile_ = "";          // populated below only if LAMBDAGRIDSTRIDE > 0
    lambdaGridStride_ = 0; parse("LAMBDAGRIDSTRIDE", lambdaGridStride_);
    if (lambdaGridStride_ > 0) lambdaGridFile_ = lbl + ".lambda_grid.dat";

    kernelFile_ = "";
    kernelStride_ = 0; parse("KERNELSTRIDE", kernelStride_);
    if (kernelStride_ > 0) kernelFile_ = lbl + ".kernels.dat";

    kernelInfoFile_ = lbl + ".kernelinfo.dat";
    kernelInfoStride_ = 0; parse("KERNELINFOSTRIDE", kernelInfoStride_);
    if (kernelInfoStride_ == 0) kernelInfoStride_ = pace_;

    czarFile_ = lbl + ".czar_kernels.dat";  // base; stamped on each write
    stateFile_ = lbl + ".state.dat";        // overwritten at STATESTRIDE interval

    // STATESTRIDE: default to CZARSTRIDE if set, else 10×GRIDPACE
    stateStride_ = 0; parse("STATESTRIDE", stateStride_);
    if (stateStride_ == 0) {
      stateStride_ = (czarStride_ > 0) ? czarStride_ : 10 * gridPace_;
    }

    nlistCenter_.assign(dim_, 0.0);
    nlistDev2_.assign(dim_, 0.0);
    znlistCenter_.assign(dim_, 0.0);
    znlistDev2_.assign(dim_, 0.0);

    // Output components
    addComponent("force2");   componentIsNotPeriodic("force2");
    addComponent("wamp");     componentIsNotPeriodic("wamp");

    fictNames_.resize(dim_);
    for (unsigned i = 0; i < dim_; ++i) {
      fictNames_[i] = getPntrToArgument(i)->getName() + "_fict";
      addComponent(fictNames_[i]);
      componentIsNotPeriodic(fictNames_[i]);
    }

    log.printf("  [FKERNELABF] Force-kernel ABF + Kernel CZAR + density-based exploration (BAOAB, direct mean force, incremental grid)\n");
    log.printf("  [FKERNELABF] CVs: ");
    for (unsigned i = 0; i < dim_; ++i)
      log.printf("%s%s", i?", ":"", getPntrToArgument(i)->getName().c_str());
    log.printf("\n  [FKERNELABF] KAPPA=");
    for (unsigned i = 0; i < dim_; ++i) log.printf("%s%.1f", i?",":"", kappa_[i]);
    log.printf("  mass=");
    for (unsigned i = 0; i < dim_; ++i) log.printf("%s%.6f", i?",":"", mass_[i]);
    log.printf("\n  [FKERNELABF] friction=");
    for (unsigned i = 0; i < dim_; ++i) log.printf("%s%.2f", i?",":"", friction_[i]);
    log.printf("  kT=%.4f  temp=%.1f\n", kT_, temp);
    if (adaptiveSigma_) {
      log.printf("  [FKERNELABF] SIGMA=AUTO (measuring CV variance over %u unbiased steps)\n",
                 adaptiveSigmaStride_);
    } else {
      log.printf("  [FKERNELABF] SIGMA=");
      for (unsigned i = 0; i < dim_; ++i) log.printf("%s%.4f", i?",":"", sigma0_[i]);
      log.printf("\n");
    }
    if (hasMin_) {
      log.printf("  SIGMA_MIN=");
      for (unsigned i = 0; i < dim_; ++i) log.printf("%s%.4f", i?",":"", sigmaMin_[i]);
      log.printf("\n");
    }
    log.printf("  %s\n", fixedSigma_ ? "FIXED_SIGMA" : "adaptive (Silverman)");
    log.printf("  [FKERNELABF] lambda-kernel mean force: exact running mean\n");
    log.printf("  [FKERNELABF] MUXCLAMP=%.0f  MAXFORCE=%.0f\n",
               muxClamp_, maxForce_);
    if (biasFactor_ > 1.0) {
      log.printf("  [FKERNELABF] BIASFACTOR=%.2f  density-based exploration: "
                 "V_ex=c·ln(1+Z/Z₀)  c=kT·(γ−1)=%.4f kJ/mol\n",
                 biasFactor_, kT_ * (biasFactor_ - 1.0));
      log.printf("  [FKERNELABF] Z₀ = median(Z) on mean-force grid, updated each grid rebuild\n");
      // Log EXPLORSCALE if any dimension is not 1.0
      bool hasScale = false;
      for (unsigned i = 0; i < dim_; ++i) if (explorScale_[i] < 1.0) hasScale = true;
      if (hasScale) {
        log.printf("  [FKERNELABF] EXPLORSCALE=(");
        for (unsigned i = 0; i < dim_; ++i) log.printf("%s%.2f", i?",":"", explorScale_[i]);
        log.printf(")  exploration force scaled per CV\n");
      }
    } else
      log.printf("  [FKERNELABF] BIASFACTOR=1.0  pure ABF, no exploration\n");
    for (unsigned i = 0; i < dim_; ++i)
      log.printf("  [FKERNELABF] dim %u: %s grid=[%.4f, %.4f] dx=%.6f (%u pts)\n",
                 i, periodic_[i]?"periodic":"non-periodic",
                 gridMin_[i], gridMax_[i], gridDx_[i], gridN_[i]);
    log.printf("  [FKERNELABF] grid update every %u steps  THRESH=%.2f  NSIGMACUT=%.1f\n",
               gridPace_, thresh_, nsigmaCut_);
    log.printf("  [FKERNELABF] incremental grid with full rebuild every %u GRIDPACE events\n",
               fullRebuildInterval_);
    // Output file summary
    log.printf("  [FKERNELABF] Output files (label=%s):\n", lbl.c_str());
    if (czarStride_ > 0)
      log.printf("  [FKERNELABF]   CZAR z-kernels : %s_<step>.dat  every %u steps\n",
                 czarFile_.c_str(), czarStride_);
    else
      log.printf("  [FKERNELABF]   CZAR z-kernels : disabled (set CZARSTRIDE to enable)\n");
    if (kernelStride_ > 0)
      log.printf("  [FKERNELABF]   lambda-kernels : %s  every %u steps\n",
                 kernelFile_.c_str(), kernelStride_);
    else
      log.printf("  [FKERNELABF]   lambda-kernels : disabled (set KERNELSTRIDE to enable)\n");
    if (lambdaGridStride_ > 0)
      log.printf("  [FKERNELABF]   lambda-grid    : %s  every %u steps  "
                 "[DEBUG: NW mean force on λ]\n",
                 lambdaGridFile_.c_str(), lambdaGridStride_);
    else
      log.printf("  [FKERNELABF]   lambda-grid    : disabled (set LAMBDAGRIDSTRIDE to enable)\n");
    log.printf("  [FKERNELABF]   restart state  : %s  every %u steps\n",
               stateFile_.c_str(), stateStride_);
    log.printf("  [FKERNELABF]   kernel info    : %s  every %u steps%s\n",
               kernelInfoFile_.c_str(), kernelInfoStride_,
               (kernelInfoStride_ == pace_) ? " (default: PACE)" : "");
    if (nlist_)
      log.printf("  [FKERNELABF] neighbor list: cutoff_factor=%.1f skin_factor=%.2f\n",
                 nlistCutFactor_, nlistSkinFactor_);
    else
      log.printf("  [FKERNELABF] neighbor list: disabled\n");

    checkRead();

    // ── Open kernel-info OFile (PLUMED backup/restart aware) ──────────
    if (!kernelInfoFile_.empty() && kernelInfoStride_ > 0) {
      kernelInfoOFile_.link(*this);
      kernelInfoOFile_.open(kernelInfoFile_);
      // Write FIELDS header
      kernelInfoOFile_.printf("#! FIELDS step M zM neff");
      for (unsigned i = 0; i < dim_; ++i)
        kernelInfoOFile_.printf(" %s_sigma", getPntrToArgument(i)->getName().c_str());
      kernelInfoOFile_.printf(" nlker\n");
      kernelInfoOFile_.flush();
      kernelInfoFileOpen_ = true;
    }

    // ── Restart: read state from file if PLUMED restart is active ──────
    // Usage: add RESTART to the PLUMED input file, or pass --restart to
    // the MD engine.  The state file {label}.state.dat must exist.
    // After reading, the mean-force grid is reconstructed immediately.
    if (getRestart()) {
      readState();
      // Force immediate mean-force grid reconstruction from restored kernels
      if (nKernels_ > 0) reconstructBiasGrid();
    }
  }

  void calculate() override {
    double dt = getTimeStep();
    for (unsigned i = 0; i < dim_; ++i) work_cv_[i] = getArgument(i);

    if (firstStep_) {
      for (unsigned i = 0; i < dim_; ++i) {
        s_fict_[i] = work_cv_[i];
        v_fict_[i] = std::sqrt(kT_ / mass_[i]) * gauss_(rng_);
      }
      if (adaptiveSigma_) {
        av_cv_.assign(dim_, 0.0);
        av_M2_.assign(dim_, 0.0);
      }
      firstStep_ = false;
    }

    // ── Adaptive sigma warmup ─────────────────────────────────────────────────
    // During the warmup window: zero bias, no kernel deposition, collect variance.
    // s_fict_ tracks z directly so it starts at the right position when bias begins.
    // NB: The Welford update uses periodicDelta, which handles wrapping correctly
    // when the distribution is concentrated within ~half the periodic domain.
    // For distributions spanning the branch cut, the variance estimate becomes
    // path-dependent.  A post-warmup sanity check flags this case.
    if (adaptiveSigma_ && adaptiveCounter_ < adaptiveSigmaStride_) {
      adaptiveCounter_++;
      unsigned tau = adaptiveCounter_;
      for (unsigned i = 0; i < dim_; ++i) {
        double diff = periodicDelta(i, av_cv_[i], work_cv_[i]);
        av_cv_[i] += diff / (double)tau;
        av_M2_[i] += diff * periodicDelta(i, av_cv_[i], work_cv_[i]);
      }
      for (unsigned i = 0; i < dim_; ++i) setOutputForce(i, 0.0);
      setBias(0.0);
      for (unsigned i = 0; i < dim_; ++i) s_fict_[i] = work_cv_[i];
      return;
    }

    // ── Set sigma0_ once at end of warmup ─────────────────────────────────────
    if (adaptiveSigma_ && adaptiveCounter_ == adaptiveSigmaStride_) {
      adaptiveCounter_++; // guard: run this block exactly once
      for (unsigned i = 0; i < dim_; ++i) {
        double std_i = (adaptiveSigmaStride_ > 1)
            ? std::sqrt(av_M2_[i] / (adaptiveSigmaStride_ - 1))
            : 0.1;
        if (std_i < 1e-6) {
          log.printf("  [FKERNELABF] WARNING: adaptive sigma for CV %u near zero "
                     "(%.2e). CV may not be fluctuating -- using 0.1.\n", i, std_i);
          std_i = 0.1;
        }
        sigma0_[i] = std_i;
        if (!sigmaMin_.empty()) sigma0_[i] = std::max(sigma0_[i], sigmaMin_[i]);
      }
      log.printf("  [FKERNELABF] Adaptive sigma finalised after %u-step warmup: sigma=(",
                 adaptiveSigmaStride_);
      for (unsigned i = 0; i < dim_; ++i)
        log.printf("%s%.5f", i?",":"", sigma0_[i]);
      log.printf(")\n");
      // Sanity check for periodic CVs: if sigma0 > domLen/4, the Welford estimate
      // likely wrapped around the branch cut and is unreliable.  The online Welford
      // algorithm with periodicDelta is only accurate when the distribution is
      // concentrated within less than half the periodic domain.
      for (unsigned i = 0; i < dim_; ++i) {
        if (periodic_[i] && domLen_[i] > 0 && sigma0_[i] > 0.25 * domLen_[i]) {
          log.printf("  [FKERNELABF] WARNING: adaptive sigma for periodic CV %u "
                     "(%.4f) exceeds domLen/4 (%.4f). The CV may have explored across "
                     "the periodic branch cut during warmup, making the variance "
                     "estimate unreliable. Consider supplying SIGMA explicitly.\n",
                     i, sigma0_[i], 0.25 * domLen_[i]);
          // Clamp to domLen/4 as a safety measure
          sigma0_[i] = 0.25 * domLen_[i];
          log.printf("  [FKERNELABF]   -> clamped to %.5f\n", sigma0_[i]);
        }
      }
      adaptiveSigma_ = false;  // warmup complete; state file will now write adaptive_done=1
      sigDirty_ = true;
      zSigDirty_ = true;
    }

    // (A) Spring force: kappa(z - lambda) — raw, unclamped
    for (unsigned i = 0; i < dim_; ++i) {
      double d = periodicDelta(i, s_fict_[i], work_cv_[i]);  // = z - lambda
      work_springF_[i] = kappa_[i] * d;
      setOutputForce(i, -kappa_[i] * d);  // force on z = kappa(lambda - z)
    }

    // (B) Accumulate lambda-kernel sample at s_fict_ (force clamped inside addSample).
    // Accumulate z-kernel sample at real CV z (unclamped, exact running mean).
    // Both happen at the same stride so lambda-kernel and z-kernel populations grow together.
    if (pace_ > 0 && getStep() > 0 && getStep() % pace_ == 0) {
      addSample(s_fict_, work_springF_);
      addZSample(work_cv_, work_springF_);
    }

    // (D) Reconstruct NW mean-force grid from lambda-kernels
    if (gridPace_ > 0 && nKernels_ > 0 && getStep() % gridPace_ == 0)
      reconstructBiasGrid();

    // (E) Interpolate ABF cancellation and exploration forces from frozen grids.
    //
    // ABF force:     F_abf = -meanForceGrid_(λ)
    // Exploration:   F_ex  = explorationForceGrid_(λ)    [only when γ > 1]
    //
    // Both grids are updated at GRIDPACE intervals; between rebuilds the force
    // on λ is a smooth, frozen function of position (ideal for BAOAB stability).
    // Scalar diagnostics (V_ex) are computed after BAOAB at the final s_fict_
    // position so they are synchronized with the reported lambda.
    double ghatF[3] = {};
    interpolateVectorGrid(s_fict_, meanForceGrid_, work_gf_);
    for (unsigned i = 0; i < dim_; ++i) ghatF[i] = work_gf_[i];

    double F_explore[3] = {0, 0, 0};
    if (biasFactor_ > 1.0) {
      interpolateVectorGrid(s_fict_, explorationForceGrid_, work_fe_);
      for (unsigned i = 0; i < dim_; ++i) F_explore[i] = explorScale_[i] * work_fe_[i];
    }

    // (F) BAOAB Langevin dynamics for s_fict_.
    if (!baoabReady_) {
      baoab_c1_.resize(dim_);
      baoab_c2_.resize(dim_);
      for (unsigned i = 0; i < dim_; ++i) {
        baoab_c1_[i] = std::exp(-friction_[i] * dt);
        baoab_c2_[i] = std::sqrt(kT_ / mass_[i] * (1.0 - baoab_c1_[i]*baoab_c1_[i]));
      }
      baoabReady_ = true;
    }
    const double halfdt = 0.5 * dt;
    double fmag2 = 0;

    // ── B: half kick (current forces) ──────────────────────────────────────
    for (unsigned i = 0; i < dim_; ++i) {
      double F_abf   = -ghatF[i];   // direct NW mean force, no Poisson
      double F_total = work_springF_[i] + F_abf + F_explore[i];
      fmag2 += sq(F_abf + F_explore[i]);
      v_fict_[i] += halfdt * F_total / mass_[i];
    }

    // ── A: half drift ──────────────────────────────────────────────────────
    for (unsigned i = 0; i < dim_; ++i)
      s_fict_[i] += halfdt * v_fict_[i];

    // ── O: Ornstein-Uhlenbeck thermostat (full dt) ─────────────────────────
    for (unsigned i = 0; i < dim_; ++i)
      v_fict_[i] = baoab_c1_[i] * v_fict_[i] + baoab_c2_[i] * gauss_(rng_);

    // ── A: half drift ──────────────────────────────────────────────────────
    for (unsigned i = 0; i < dim_; ++i) {
      s_fict_[i] += halfdt * v_fict_[i];
      // Boundary handling (reflecting walls or periodic wrap)
      if (!std::isfinite(s_fict_[i])) {
        log.printf("  [FKERNELABF] WARNING: s_fict_[%u] is non-finite (%g); resetting to domain center.\n",
                   i, s_fict_[i]);
        s_fict_[i] = 0.5 * (gridMin_[i] + gridMax_[i]);
        v_fict_[i] = 0.0;
      }
      if (periodic_[i]) {
        s_fict_[i] = wrapToDomain(i, s_fict_[i]);
      } else {
        while (s_fict_[i] < gridMin_[i]) {
          s_fict_[i] = 2.0*gridMin_[i] - s_fict_[i];
          v_fict_[i] = -v_fict_[i];
        }
        while (s_fict_[i] > gridMax_[i]) {
          s_fict_[i] = 2.0*gridMax_[i] - s_fict_[i];
          v_fict_[i] = -v_fict_[i];
        }
      }
    }

    // ── B: half kick (forces at updated position) ──────────────────────────
    {
      interpolateVectorGrid(s_fict_, meanForceGrid_, work_gf2_);
      double fe2[3] = {0, 0, 0};
      if (biasFactor_ > 1.0) {
        interpolateVectorGrid(s_fict_, explorationForceGrid_, work_fev_);
        for (unsigned i = 0; i < dim_; ++i) fe2[i] = explorScale_[i] * work_fev_[i];
      }
      for (unsigned i = 0; i < dim_; ++i) {
        double d2 = periodicDelta(i, s_fict_[i], work_cv_[i]);
        double springF2 = kappa_[i] * d2;
        double F_abf2   = -work_gf2_[i];
        double F_total2 = springF2 + F_abf2 + fe2[i];
        v_fict_[i] += halfdt * F_total2 / mass_[i];
      }
    }

    // (G) Post-BAOAB diagnostics at the final s_fict_ position.
    //
    // V_ex and Neff are evaluated here (not before BAOAB) so that the
    // reported scalars are synchronized with the reported lambda position.
    double V_ex = 0.0;
    if (biasFactor_ > 1.0) {
      double Z_final = interpolateScalar(s_fict_, nwDenominator_);
      double c_ex = kT_ * (biasFactor_ - 1.0);
      V_ex = c_ex * std::log(1.0 + Z_final / (Z0_density_ + 1e-300));
    }

    setBias(V_ex);
    getPntrToComponent("wamp")->set(V_ex);
    getPntrToComponent("force2")->set(fmag2);
    for (unsigned i = 0; i < dim_; ++i)
      getPntrToComponent(fictNames_[i])->set(s_fict_[i]);

    // (I) Lambda-grid file output (debug: NW mean force on λ)
    if (!lambdaGridFile_.empty() && lambdaGridStride_ > 0 && nKernels_ > 0 && getStep() % lambdaGridStride_ == 0)
      writeGridFile();

    // (J) lambda-kernel dump
    dumpKernelsIfNeeded();

    // (K) CZAR z-kernel file (step-stamped snapshot; never overwritten)
    if (!czarFile_.empty() && czarStride_ > 0 && nZKernels_ > 0 && getStep() % czarStride_ == 0)
      writeCZARFile();

    // (L) Restart state (overwritten in place, no backups)
    if (stateStride_ > 0 && nKernels_ > 0 && getStep() % stateStride_ == 0)
      writeState();

    // (M) Kernel diagnostics file (appending time series)
    if (kernelInfoStride_ > 0) writeKernelInfo();
  }
};

PLUMED_REGISTER_ACTION(ForceKernelABF, "FKERNELABF")

}  
}  
