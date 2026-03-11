#include "bias/Bias.h"
#include "core/ActionRegister.h"
#include "core/PlumedMain.h"
#include "core/Atoms.h"
#include "tools/IFile.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <random>
#include <cstdio>
#include <cassert>

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
  double friction_;
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
  double nsigmaCut_;  // kernel cutoff in sigma (default 4.0)
  // biasFactor_ (γ) controls the density-based exploration force on λ.
  // γ = 1: pure ABF, no exploration force.
  // γ > 1: F_ex = −c·∇Z/(Z₀+Z) where c=kT(γ−1), Z is the NW denominator
  //   interpolated from Zgrid_, and Z₀=median(Z) on the grid.  The force
  //   pushes λ away from well-sampled basins toward under-sampled regions.
  //   Both Z and F_ex are updated at GRIDPACE intervals, synchronized with
  //   the ABF mean force.
  // The CZAR estimator on z is completely unaffected.
  double biasFactor_;
  double Z0_density_;   // median-Z reference for density-based exploration
  double muxClamp_;   // per-kernel force clamp (default 500)
  double maxForce_;   // grid bias force clamp  (default 500)
  std::vector<double> sigma0_;
  std::vector<double> sigmaMin_;
  bool fixedSigma_;
  bool hasMin_;    // cached !sigmaMin_.empty(); avoids repeated .size() checks

  // -- lambda-kernels (bias driving, indexed at s_fict) --------------------------
  struct Kernel {
    std::vector<double> center;
    std::vector<double> mu;      // mean force estimate (clamped, exact running mean)
    std::vector<double> sigma;
    double Nk;
  };
  std::vector<Kernel> kernels_;
  unsigned M_;
  double totalN_;
  double sumNk2_;   // sum of Nk^2; neff = totalN_^2 / sumNk2_

  // -- z-kernels (CZAR estimator, indexed at real CV z) -------------------------
  // Centers track the real collective variable (not the fictitious lambda).
  // mu stores the unclamped mean spring force kappa*(z - lambda).
  // Always uses the exact running mean.
  struct ZKernel {
    std::vector<double> center;  // position in real CV space
    std::vector<double> mu;      // unclamped mean of kappa(z - lambda)
    std::vector<double> sigma;
    double Nk;
  };
  std::vector<ZKernel> zKernels_;
  unsigned zM_;
  double zTotalN_;
  double zSumNk2_;  // sum of zNk^2 for z-kernel neff

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
  unsigned gridSize_;
  std::vector<double> gridMin_, gridMax_, gridDx_;
  std::vector<double> ghat_;   // NW mean force grid [gridTotal_ * dim_], for direct ABF force
  std::vector<double> Zgrid_;  // NW denominator on grid [gridTotal_], for exploration + Neff
  std::vector<double> fex_;    // exploration force on grid [gridTotal_ * dim_], zero when γ=1

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
  // Contains: fictitious particle, kernel populations, sigma0, Z0_density.
  // The mean-force grid is reconstructed immediately from kernels on RESTART
  // (not deferred to the next GRIDPACE step) so the bias is live from step 1.
  std::string stateFile_;
  unsigned stateStride_;

  // --- kernel diagnostics file ---
  // Appends one line per write to {label}.kernelinfo.dat.
  // Columns: step, M, zM, neff, sigma per CV dim, nlker.
  std::string kernelInfoFile_;
  unsigned kernelInfoStride_;

  // --- state ---
  std::vector<std::string> fictNames_;

  // ================ helpers ================
  double sq(double x) const { return x * x; }

  // Lambda-side ABF always applies the full NW mean force from the grid.
  // Density-based exploration (γ > 1) adds F_ex = −c·∇Z/(Z₀+Z) on top,
  // also interpolated from the grid (synchronized at GRIDPACE intervals).
  // Both the ABF cancellation force and the exploration force act on λ,
  // leaving z and the CZAR estimator completely clean.

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
  // Uses M_ (compressed kernel count), NOT totalN_ (raw sample count).
  // Silverman's n = number of distinct locations being represented.
  // After compression that is M_; using totalN_ >> M_ causes the bandwidth
  // to shrink as if you had far more data than you do, hitting SIGMA_MIN
  // prematurely and fragmenting the representation.
  std::vector<double> currentSigma() const {
    std::vector<double> sig = sigma0_;
    if (!fixedSigma_ && M_ > 1) {
      const double neff = (sumNk2_ > 0.0) ? (totalN_*totalN_/sumNk2_) : (double)M_;
      double s_rescaling = std::pow(neff*(dim_+2.0)/4.0, -1.0/(4.0+dim_));
      for (unsigned i = 0; i < dim_; ++i) {
        sig[i] *= s_rescaling;
        if (hasMin_) sig[i] = std::max(sig[i], sigmaMin_[i]);
      }
    }
    return sig;
  }


  // Silverman bandwidth for z-kernels -- same fix applied.
  std::vector<double> currentZSigma() const {
    std::vector<double> sig = sigma0_;
    if (!fixedSigma_ && zM_ > 1) {
      const double neff = (zSumNk2_ > 0.0) ? (zTotalN_*zTotalN_/zSumNk2_) : (double)zM_;
      double s_rescaling = std::pow(neff*(dim_+2.0)/4.0, -1.0/(4.0+dim_));
      for (unsigned i = 0; i < dim_; ++i) {
        sig[i] *= s_rescaling;
        if (hasMin_) sig[i] = std::max(sig[i], sigmaMin_[i]);
      }
    }
    return sig;
  }

  double dist2KernelNorm(const std::vector<double>& s, unsigned k,
                         const std::vector<double>& sig) const {
    // Merge distance normalized by the current global Silverman bandwidth.
    // Using per-kernel sigma causes a positive feedback loop where shrinking
    // sigma → tighter merge radius → more kernels → more shrinkage.
    double acc = 0.0;
    for (unsigned i = 0; i < dim_; ++i) {
      double d = s[i] - kernels_[k].center[i];
      if (periodic_[i] && domLen_[i] > 0)
        d -= domLen_[i] * std::round(d / domLen_[i]);
      acc += sq(d) / (4.0*sq(sig[i]) + 1e-300);
    }
    return acc;
  }

  // -- lambda-kernel merge search (purely geometric) ---------------------------
  int findMergeable(const std::vector<double>& s, int exclude = -1) {
    const std::vector<double> sig = currentSigma();
    double r2 = 0.25*sq(thresh_); int best = -1; double bestd2 = r2;
    auto evalKernel = [&](unsigned k) {
      if ((int)k == exclude) return;
      double d2 = dist2KernelNorm(s, k, sig);
      if (d2 < bestd2) { bestd2 = d2; best = (int)k; }
    };
    if (nlist_ && !nlistIdx_.empty()) {
      for (unsigned n = 0; n < nlistIdx_.size(); ++n) evalKernel(nlistIdx_[n]);
    } else {
      for (unsigned k = 0; k < M_; ++k) evalKernel(k);
    }
    return best;
  }

  // -- z-kernel merge search (purely geometric, uses global Silverman sigma) --
  int findMergeableZ(const std::vector<double>& s, int exclude = -1) {
    const std::vector<double> sig = currentZSigma();
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
      for (unsigned k = 0; k < zM_; ++k) evalKernel(k);
    }
    return best;
  }

  // ================ neighbor list (lambda-kernels only) ================
  // The broad-phase cutoff must use the same distance metric as findMergeable
  // (global Silverman sigma), otherwise valid merge candidates can be excluded
  // when per-kernel sigma differs from the global sigma.
  void updateNlist(const std::vector<double>& cv) {
    nlistCenter_ = cv;
    nlistIdx_.clear();
    if (M_ == 0) { nlistUpdate_ = false; return; }
    const std::vector<double> sig = currentSigma();
    double cutoff2 = 0.25 * nlistCutFactor_ * sq(nsigmaCut_);
    for (unsigned k = 0; k < M_; ++k) {
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
  void updateZNlist(const std::vector<double>& cv) {
    znlistCenter_ = cv;
    znlistIdx_.clear();
    if (zM_ == 0) { znlistUpdate_ = false; return; }

    const std::vector<double> sig = currentZSigma();
    const double cutoff2 = 0.25 * nlistCutFactor_ * sq(nsigmaCut_);
    for (unsigned k = 0; k < zM_; ++k) {
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
  // and fuses the pair (weighted parallel-variance merge) until no further
  // merge candidates exist.
  //
  // Template parameters:
  //   K        -- kernel type (Kernel or ZKernel)
  //   FindFn   -- callable(center, exclude) -> int: returns index of nearest
  //               mergeable neighbor, or -1 if none within threshold
  //   MuFn     -- callable(mu_merged) -> double: post-merge mu transform
  //               (clamping for lambda-kernels; identity for z-kernels)
  //
  // The giver kernel is removed by swap-with-last (O(1)); the nlist index
  // array is patched in-place to keep it consistent.
  template<typename K, typename FindFn, typename MuFn>
  void mergeKernelPool(unsigned giver,
                       std::vector<K>& kernels,
                       unsigned& M,
                       double& sumNk2,
                       std::vector<unsigned>& nlistIdx,
                       FindFn findMerge,
                       MuFn muUpdate) {
    const bool hasMin = !sigmaMin_.empty();
    int taker = findMerge(kernels[giver].center, (int)giver);
    while (taker >= 0) {
      assert(M > 0);
      assert(giver < kernels.size() && (unsigned)taker < kernels.size());

      double Nt = kernels[taker].Nk, Ng = kernels[giver].Nk, Ntot = Nt + Ng;
      assert(Ntot > 0.0);
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
      // If taker was the last element, it moves into giver's slot — newGiver tracks this.
      unsigned last = M - 1;
      unsigned newGiver = (unsigned)taker;
      if (giver != last && (unsigned)taker == last) newGiver = giver;
      if (giver != last) kernels[giver] = std::move(kernels[last]);
      kernels.resize(last);
      --M;
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


  // ================ lambda-kernel compression (bias driving) ================
  void addSample(const std::vector<double>& s_in, const std::vector<double>& f_in) {
    std::vector<double> s(dim_), f(dim_);
    for (unsigned i = 0; i < dim_; ++i) {
      s[i] = wrapToDomain(i, s_in[i]);
      // Clamping applied to lambda-kernel forces only; z-kernels are unclamped.
      f[i] = std::max(-muxClamp_, std::min(muxClamp_, f_in[i]));
    }

    // Pre-arrival bandwidth: the incoming sample is conceptually a mini-kernel
    // with the bandwidth that existed before it changed the statistics.
    // Computing this before any counter updates makes the absorption and
    // new-kernel branches use the same sigma snapshot.
    std::vector<double> sig = currentSigma();

    int k = findMergeable(s, -1);
    unsigned giver;   // index of the kernel to start recursive merging from

    if (k >= 0) {
      // ── Absorption: merge sample into existing kernel k ──────────────
      double Nold = kernels_[k].Nk;
      sumNk2_ -= Nold*Nold;
      kernels_[k].Nk += 1.0;
      double Nnew = kernels_[k].Nk;
      sumNk2_ += Nnew*Nnew;
      totalN_ += 1.0;
      const double inv_Nnew = 1.0 / Nnew;

      for (unsigned i = 0; i < dim_; ++i) {
        // mu: exact running mean
        double muold = kernels_[k].mu[i];
        double munew = (Nold*muold + f[i]) * inv_Nnew;
        kernels_[k].mu[i] = std::max(-muxClamp_, std::min(muxClamp_, munew));

        // center + sigma: parallel variance with pre-arrival mini-kernel
        double ct = kernels_[k].center[i];
        double cs = ct + periodicDelta(i, ct, s[i]);
        double c_new = (Nold * ct + cs) * inv_Nnew;
        double dt = ct - c_new, ds = cs - c_new;
        double var = (Nold * (sq(kernels_[k].sigma[i]) + sq(dt)) +
                             (sq(sig[i])               + sq(ds))) * inv_Nnew;
        kernels_[k].center[i] = wrapToDomain(i, c_new);
        kernels_[k].sigma[i] = std::sqrt(std::max(var,
            sq(hasMin_ ? sigmaMin_[i] : 1e-6)));
      }
      giver = (unsigned)k;

    } else {
      // ── New kernel at pre-arrival Silverman bandwidth ────────────────
      Kernel nk;
      nk.center.resize(dim_); nk.mu.resize(dim_); nk.sigma.resize(dim_);
      for (unsigned i = 0; i < dim_; ++i) {
        nk.center[i] = s[i];
        nk.mu[i] = f[i];
        nk.sigma[i] = sig[i];
      }
      nk.Nk = 1.0;
      kernels_.push_back(std::move(nk));
      ++M_;
      totalN_ += 1.0;
      sumNk2_ += 1.0;
      if (nlist_) nlistIdx_.push_back(M_-1);
      giver = M_ - 1;
    }

    // ── Recursive merge from giver (runs for BOTH branches) ────────────
    // After absorption the kernel's center has shifted; after new-kernel
    // creation the kernel may overlap a neighbor. In either case, check
    // for cascading merges until no more candidates remain.
    mergeKernelPool(giver, kernels_, M_, sumNk2_, nlistIdx_,
        [this](const std::vector<double>& c, int ex){ return findMergeable(c, ex); },
        [this](double mu){ return std::max(-muxClamp_, std::min(muxClamp_, mu)); });

    // Cascading merges may have shifted kernel centers; flag nlist rebuild.
    if (nlist_) nlistUpdate_ = true;
  }

  // ================ z-kernel accumulation (CZAR) ================
  // z-kernels are centred at the real CV z, store the unclamped spring force,
  // and always use the exact running mean.
  void addZSample(const std::vector<double>& z_in,
                  const std::vector<double>& f_raw) {
    std::vector<double> z(dim_);
    for (unsigned i = 0; i < dim_; ++i)
      z[i] = wrapToDomain(i, z_in[i]);
    // f_raw = kappa(z - lambda), unclamped -- do NOT clamp here.

    // Maintain a local z-kernel neighbor list around the current real CV.
    if (nlist_ && zM_ > 0) {
      if (znlistUpdate_ || needsZNlistUpdate(z)) updateZNlist(z);
    }

    // Pre-arrival bandwidth (same rationale as addSample).
    std::vector<double> sig = currentZSigma();

    int best = findMergeableZ(z, -1);
    unsigned giver;   // index to start recursive merging from

    if (best >= 0) {
      // ── Absorption: merge sample into existing z-kernel ──────────────
      double Nold = zKernels_[best].Nk;
      zSumNk2_ -= Nold*Nold;
      zKernels_[best].Nk += 1.0;
      double Nnew = zKernels_[best].Nk;
      zSumNk2_ += Nnew*Nnew;
      zTotalN_ += 1.0;
      const double inv_Nnew = 1.0 / Nnew;

      for (unsigned i = 0; i < dim_; ++i) {
        // mu: exact running mean, unclamped
        zKernels_[best].mu[i] = (Nold * zKernels_[best].mu[i] + f_raw[i]) * inv_Nnew;

        // center + sigma: parallel variance with pre-arrival mini-kernel
        double ct = zKernels_[best].center[i];
        double cs = ct + periodicDelta(i, ct, z[i]);
        double c_new = (Nold * ct + cs) * inv_Nnew;
        double dt = ct - c_new, ds = cs - c_new;
        double var = (Nold * (sq(zKernels_[best].sigma[i]) + sq(dt)) +
                             (sq(sig[i])                   + sq(ds))) * inv_Nnew;
        zKernels_[best].center[i] = wrapToDomain(i, c_new);
        zKernels_[best].sigma[i] = std::sqrt(std::max(var,
            sq(hasMin_ ? sigmaMin_[i] : 1e-6)));
      }
      giver = (unsigned)best;

    } else {
      // ── New z-kernel at pre-arrival Silverman bandwidth ──────────────
      ZKernel nk;
      nk.center.resize(dim_); nk.mu.resize(dim_); nk.sigma.resize(dim_);
      for (unsigned i = 0; i < dim_; ++i) {
        nk.center[i] = z[i];
        nk.mu[i]     = f_raw[i];   // unclamped
        nk.sigma[i]  = sig[i];
      }
      nk.Nk = 1.0;
      zKernels_.push_back(std::move(nk));
      ++zM_;
      zTotalN_ += 1.0;
      zSumNk2_ += 1.0;
      if (nlist_) {
        znlistIdx_.push_back(zM_ - 1);
        znlistUpdate_ = true;
      }
      giver = zM_ - 1;
    }

    // ── Recursive merge from giver (runs for BOTH branches) ────────────
    mergeKernelPool(giver, zKernels_, zM_, zSumNk2_, znlistIdx_,
        [this](const std::vector<double>& c, int ex){ return findMergeableZ(c, ex); },
        [](double mu){ return mu; });

    // Flag nlist rebuild after any center/sigma changes.
    if (nlist_) znlistUpdate_ = true;
  }

  // ================ grid indexing ================
  unsigned gridFlat(const std::vector<unsigned>& idx) const {
    unsigned f = 0;
    for (unsigned d = 0; d < dim_; ++d) f = f * gridN_[d] + idx[d];
    return f;
  }
  void gridUnflat(unsigned flat, std::vector<unsigned>& idx) const {
    idx.resize(dim_);
    for (int d = (int)dim_-1; d >= 0; --d) { idx[d] = flat % gridN_[d]; flat /= gridN_[d]; }
  }
  double gridCoord(unsigned d, unsigned idx) const {
    return gridMin_[d] + idx * gridDx_[d];
  }

  // ================ grid mean-force evaluation (lambda-kernels, for bias) ========
  // Pure Nadaraya-Watson regression weighted by Nk.  Returns the estimated
  // mean force ĝ(s) at each grid point, and the NW denominator Z[g] which
  // measures kernel support at each node.
  // The mean-force clamp (maxForce_) is applied downstream in reconstructBiasGrid.
  void evaluateGridMeanForces(std::vector<double>& ghat, std::vector<double>& Zgrid) {
    ghat.assign(gridTotal_ * dim_, 0.0);
    Zgrid.assign(gridTotal_, 0.0);
    if (M_ == 0) return;

    std::vector<double>& Z = Zgrid;   // alias — filled in-place
    std::vector<double> numF(gridTotal_ * dim_, 0.0);

    std::vector<int> R(dim_), ic(dim_);
    std::vector<std::vector<double>> w1d(dim_);
    std::vector<std::vector<unsigned>> gi1d(dim_);

    for (unsigned kk = 0; kk < M_; ++kk) {
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
        w1d[d].clear();   w1d[d].reserve(2*R[d]+1);
        gi1d[d].clear(); gi1d[d].reserve(2*R[d]+1);
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
          w1d[d].push_back(w);
          gi1d[d].push_back(gi);
        }
      }

      // Nadaraya-Watson regression: weight by Nk so that well-sampled
      // kernels dominate the local mean-force estimate.
      const double Nk_kk = kernels_[kk].Nk;
      if (dim_ == 1) {
        for (unsigned a = 0; a < w1d[0].size(); ++a) {
          unsigned g = gi1d[0][a]; double wNk = w1d[0][a] * Nk_kk;
          Z[g] += wNk; numF[g] += wNk*kernels_[kk].mu[0];
        }
      } else if (dim_ == 2) {
        for (unsigned a = 0; a < w1d[0].size(); ++a) {
          double wa = w1d[0][a];
          for (unsigned b = 0; b < w1d[1].size(); ++b) {
            double wNk = wa * w1d[1][b] * Nk_kk;
            unsigned g = gi1d[0][a]*gridN_[1]+gi1d[1][b];
            Z[g] += wNk;
            numF[g*2+0] += wNk*kernels_[kk].mu[0];
            numF[g*2+1] += wNk*kernels_[kk].mu[1];
          }
        }
      } else {
        for (unsigned a = 0; a < w1d[0].size(); ++a) {
          double wa = w1d[0][a];
          for (unsigned b = 0; b < w1d[1].size(); ++b) {
            double wab = wa * w1d[1][b];
            unsigned gab = (gi1d[0][a]*gridN_[1]+gi1d[1][b])*gridN_[2];
            for (unsigned c = 0; c < w1d[2].size(); ++c) {
              double wNk = wab*w1d[2][c] * Nk_kk;
              unsigned g = gab + gi1d[2][c];
              Z[g] += wNk;
              numF[g*3+0] += wNk*kernels_[kk].mu[0];
              numF[g*3+1] += wNk*kernels_[kk].mu[1];
              numF[g*3+2] += wNk*kernels_[kk].mu[2];
            }
          }
        }
      }
    }

    // Z[g] = Σ_k w_k·Nk serves as the NW denominator.
    for (unsigned g = 0; g < gridTotal_; ++g) {
      if (Z[g] > 1e-300) {
        for (unsigned d = 0; d < dim_; ++d) ghat[g*dim_+d] = numF[g*dim_+d] / Z[g];
      }
    }
  }

  // ================ bias grid reconstruction (direct mean force, DRR-style) ====
  // Computes: ghat_ (ABF mean force), Zgrid_ (NW denominator), fex_ (exploration force).
  // All three grids are updated atomically at GRIDPACE intervals.
  // Between rebuilds, the total force on λ is completely frozen — ideal for BAOAB.
  void reconstructBiasGrid() {
    std::vector<double> ghat, Zgrid;
    evaluateGridMeanForces(ghat, Zgrid);

    // Clamp and store the mean force grid for direct interpolation.
    ghat_.assign(gridTotal_*dim_, 0.0);
    for (unsigned g = 0; g < gridTotal_; ++g)
      for (unsigned d = 0; d < dim_; ++d) {
        double v = ghat[g*dim_+d];
        if (maxForce_ > 0) v = std::max(-maxForce_, std::min(maxForce_, v));
        ghat_[g*dim_+d] = v;
      }

    // Store Z grid for Neff and V_ex interpolation.
    Zgrid_ = Zgrid;

    // Update Z₀ and compute exploration force on grid.
    fex_.assign(gridTotal_*dim_, 0.0);
    if (biasFactor_ > 1.0) {
      // Z₀ = median of Z over populated nodes.
      std::vector<double> Zpop;
      Zpop.reserve(gridTotal_);
      for (unsigned g = 0; g < gridTotal_; ++g)
        if (Zgrid[g] > 1e-10) Zpop.push_back(Zgrid[g]);
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

      // Compute ∇Z via finite differences, then F_ex = -c·∇Z/(Z₀+Z) at each grid point.
      // Strides for flat indexing along each dimension.
      std::vector<unsigned> strides(dim_);
      strides[dim_-1] = 1;
      for (int dd = (int)dim_-2; dd >= 0; --dd)
        strides[dd] = strides[dd+1] * gridN_[dd+1];

      double c_ex = kT_ * (biasFactor_ - 1.0);

      std::vector<unsigned> idx(dim_);
      for (unsigned g = 0; g < gridTotal_; ++g) {
        if (Zgrid[g] < 1e-300) continue;  // no data, no force
        double denom = Z0_density_ + Zgrid[g];

        // Extract multi-index for boundary checks.
        gridUnflat(g, idx);

        for (unsigned d = 0; d < dim_; ++d) {
          double dZds = 0.0;
          unsigned N_d = gridN_[d];
          unsigned stride_d = strides[d];

          if (periodic_[d]) {
            // Central difference with periodic wrap.
            unsigned gp = g + ((idx[d]+1 < N_d) ? stride_d : -(N_d-1)*stride_d);
            unsigned gm = g - ((idx[d] > 0) ? stride_d : -(N_d-1)*stride_d);
            dZds = (Zgrid[gp] - Zgrid[gm]) / (2.0 * gridDx_[d]);
          } else if (idx[d] == 0) {
            // Forward difference at left boundary.
            if (N_d >= 3)
              dZds = (-3.0*Zgrid[g] + 4.0*Zgrid[g+stride_d] - Zgrid[g+2*stride_d]) / (2.0*gridDx_[d]);
            else
              dZds = (Zgrid[g+stride_d] - Zgrid[g]) / gridDx_[d];
          } else if (idx[d] == N_d-1) {
            // Backward difference at right boundary.
            if (N_d >= 3)
              dZds = (3.0*Zgrid[g] - 4.0*Zgrid[g-stride_d] + Zgrid[g-2*stride_d]) / (2.0*gridDx_[d]);
            else
              dZds = (Zgrid[g] - Zgrid[g-stride_d]) / gridDx_[d];
          } else {
            // Central difference.
            dZds = (Zgrid[g+stride_d] - Zgrid[g-stride_d]) / (2.0 * gridDx_[d]);
          }

          fex_[g*dim_+d] = -c_ex * dZds / denom;
        }
      }
    }

    std::vector<double> sig = currentSigma();
    log.printf("  [FKERNELABF] Grid update step %lld: M=%u Ntot=%.0f zM=%u zNtot=%.0f",
               (long long)getStep(), M_, totalN_, zM_, zTotalN_);
    if (biasFactor_ > 1.0)
      log.printf(" Z0_density=%.2f", Z0_density_);
    log.printf(" sigma=(%.4f", sig[0]);
    for (unsigned i = 1; i < dim_; ++i) log.printf(",%.4f", sig[i]);
    log.printf(")\n");
  }

  // ================ direct mean-force interpolation (DRR-style) ================
  // Multilinear interpolation of the NW mean force grid ghat_.
  // This bypasses the Poisson solver entirely: the ABF cancellation force
  // is the negative of the interpolated mean force, exactly as DRR does it.
  // No integration, no differentiation, no boundary artifacts.
  void interpolateForce(const std::vector<double>& s, std::vector<double>& force) const {
    force.assign(dim_, 0.0);
    if (ghat_.empty()) return;
    std::vector<double> frac(dim_);
    std::vector<unsigned> lo(dim_);
    for (unsigned d = 0; d < dim_; ++d) {
      double f = (s[d] - gridMin_[d]) / gridDx_[d];
      if (periodic_[d]) {
        f -= std::floor(f/(double)gridN_[d]) * (double)gridN_[d];
        lo[d] = (unsigned)std::floor(f);
        if (lo[d] >= gridN_[d]) lo[d] = gridN_[d]-1;
      } else {
        if (f < 0.0) f = 0.0;
        if (f >= (double)(gridN_[d]-1)) f = (double)(gridN_[d]-1)-1e-12;
        lo[d] = (unsigned)std::floor(f);
        if (lo[d] >= gridN_[d]-1) lo[d] = gridN_[d]-2;
      }
      frac[d] = f - (double)lo[d];
    }
    unsigned nC = 1u << dim_;
    for (unsigned c = 0; c < nC; ++c) {
      std::vector<unsigned> ci(dim_);
      double w = 1.0;
      for (unsigned d = 0; d < dim_; ++d) {
        bool hi = (c >> d) & 1;
        ci[d] = periodic_[d] ? (hi ? (lo[d]+1)%gridN_[d] : lo[d]) : (hi ? lo[d]+1 : lo[d]);
        w *= hi ? frac[d] : (1.0-frac[d]);
      }
      unsigned g = gridFlat(ci);
      for (unsigned d = 0; d < dim_; ++d) force[d] += w * ghat_[g*dim_+d];
    }
  }

  // Multilinear interpolation of a scalar grid (used for Zgrid_ -> Neff, V_ex).
  double interpolateScalar(const std::vector<double>& s, const std::vector<double>& grid) const {
    if (grid.empty()) return 0.0;
    std::vector<double> frac(dim_);
    std::vector<unsigned> lo(dim_);
    for (unsigned d = 0; d < dim_; ++d) {
      double f = (s[d] - gridMin_[d]) / gridDx_[d];
      if (periodic_[d]) {
        f -= std::floor(f/(double)gridN_[d]) * (double)gridN_[d];
        lo[d] = (unsigned)std::floor(f);
        if (lo[d] >= gridN_[d]) lo[d] = gridN_[d]-1;
      } else {
        if (f < 0.0) f = 0.0;
        if (f >= (double)(gridN_[d]-1)) f = (double)(gridN_[d]-1)-1e-12;
        lo[d] = (unsigned)std::floor(f);
        if (lo[d] >= gridN_[d]-1) lo[d] = gridN_[d]-2;
      }
      frac[d] = f - (double)lo[d];
    }
    unsigned nC = 1u << dim_;
    double val = 0.0;
    for (unsigned c = 0; c < nC; ++c) {
      std::vector<unsigned> ci(dim_);
      double w = 1.0;
      for (unsigned d = 0; d < dim_; ++d) {
        bool hi = (c >> d) & 1;
        ci[d] = periodic_[d] ? (hi ? (lo[d]+1)%gridN_[d] : lo[d]) : (hi ? lo[d]+1 : lo[d]);
        w *= hi ? frac[d] : (1.0-frac[d]);
      }
      val += w * grid[gridFlat(ci)];
    }
    return val;
  }

  // Multilinear interpolation of the exploration force grid fex_.
  void interpolateExplore(const std::vector<double>& s, std::vector<double>& force) const {
    force.assign(dim_, 0.0);
    if (fex_.empty()) return;
    std::vector<double> frac(dim_);
    std::vector<unsigned> lo(dim_);
    for (unsigned d = 0; d < dim_; ++d) {
      double f = (s[d] - gridMin_[d]) / gridDx_[d];
      if (periodic_[d]) {
        f -= std::floor(f/(double)gridN_[d]) * (double)gridN_[d];
        lo[d] = (unsigned)std::floor(f);
        if (lo[d] >= gridN_[d]) lo[d] = gridN_[d]-1;
      } else {
        if (f < 0.0) f = 0.0;
        if (f >= (double)(gridN_[d]-1)) f = (double)(gridN_[d]-1)-1e-12;
        lo[d] = (unsigned)std::floor(f);
        if (lo[d] >= gridN_[d]-1) lo[d] = gridN_[d]-2;
      }
      frac[d] = f - (double)lo[d];
    }
    unsigned nC = 1u << dim_;
    for (unsigned c = 0; c < nC; ++c) {
      std::vector<unsigned> ci(dim_);
      double w = 1.0;
      for (unsigned d = 0; d < dim_; ++d) {
        bool hi = (c >> d) & 1;
        ci[d] = periodic_[d] ? (hi ? (lo[d]+1)%gridN_[d] : lo[d]) : (hi ? lo[d]+1 : lo[d]);
        w *= hi ? frac[d] : (1.0-frac[d]);
      }
      unsigned g = gridFlat(ci);
      for (unsigned d = 0; d < dim_; ++d) force[d] += w * fex_[g*dim_+d];
    }
  }

  // ================ lambda-grid output (DEBUG: mean force on λ grid) ========
  void writeGridFile() {
    if (lambdaGridFile_.empty()) return;
    std::string path = stampedPath(lambdaGridFile_);
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return;

    std::fprintf(f, "# NW mean force on lambda grid (DEBUG)\n");
    std::fprintf(f, "# Generated at step %lld  M=%u\n", (long long)getStep(), M_);
    std::fprintf(f, "#! FIELDS");
    for (unsigned d = 0; d < dim_; ++d) std::fprintf(f, " s%u", d);
    for (unsigned d = 0; d < dim_; ++d) std::fprintf(f, " ghat%u", d);
    std::fprintf(f, " Neff\n");

    std::vector<unsigned> idx(dim_);
    std::vector<double> s(dim_), ghat(dim_);
    for (unsigned g = 0; g < gridTotal_; ++g) {
      gridUnflat(g, idx);
      for (unsigned d = 0; d < dim_; ++d) s[d] = gridCoord(d, idx[d]);

      double Z = 0.0;
      ghat.assign(dim_, 0.0);
      for (unsigned kk = 0; kk < M_; ++kk) {
        double d2 = 0.0;
        for (unsigned i = 0; i < dim_; ++i) {
          double dd = periodicDelta(i, kernels_[kk].center[i], s[i]);
          d2 += sq(dd) / (4.0*sq(kernels_[kk].sigma[i]) + 1e-300);
        }
        if (d2 > 0.25*sq(nsigmaCut_)) continue;
        double w = std::exp(-d2);
        if (w < 1e-300) continue;
        double wNk = w * kernels_[kk].Nk;
        Z += wNk;
        for (unsigned i = 0; i < dim_; ++i) ghat[i] += wNk*kernels_[kk].mu[i];
      }
      double Neff = 0.0;
      if (Z > 0) {
        for (unsigned i = 0; i < dim_; ++i) ghat[i] /= Z;
        Neff = Z;
      }

      for (unsigned d = 0; d < dim_; ++d) std::fprintf(f, " %14.6f", s[d]);
      for (unsigned d = 0; d < dim_; ++d) std::fprintf(f, " %14.6f", ghat[d]);
      std::fprintf(f, " %14.6f\n", Neff);
    }
    std::fclose(f);
    log.printf("  [FKERNELABF] Lambda-grid written: %s  (M=%u, step %lld)\n",
               path.c_str(), M_, (long long)getStep());
  }

  // ================ kernel diagnostics file ================
  // Appends one line per call to {label}.kernelinfo.dat.
  // Header written on first call (file absent or step 0).
  // PLUMED-compatible #! FIELDS header for use with fkabf_analysis.py.
  void writeKernelInfo() {
    if (kernelInfoFile_.empty()) return;
    if (getStep() % kernelInfoStride_ != 0) return;

    bool write_header = (getStep() == 0);
    if (!write_header) {
      std::ifstream test(kernelInfoFile_);
      write_header = !test.good();
    }

    std::FILE* f = std::fopen(kernelInfoFile_.c_str(), "a");
    if (!f) return;

    if (write_header) {
      std::fprintf(f, "#! FIELDS step M zM neff");
      for (unsigned i = 0; i < dim_; ++i)
        std::fprintf(f, " %s_sigma", getPntrToArgument(i)->getName().c_str());
      std::fprintf(f, " nlker\n");
    }

    const double neff = (sumNk2_ > 0.0) ? (totalN_*totalN_/sumNk2_) : (double)M_;
    const std::vector<double> sig = currentSigma();

    std::fprintf(f, "%lld %u %u %.4f", (long long)getStep(), M_, zM_, neff);
    for (unsigned i = 0; i < dim_; ++i)
      std::fprintf(f, " %.6f", sig[i]);
    std::fprintf(f, " %zu\n", nlistIdx_.size());

    std::fclose(f);
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

    std::vector<double> sig = currentSigma();
    const double neff_sil = (sumNk2_ > 0.0) ? (totalN_*totalN_/sumNk2_) : (double)M_;
    std::fprintf(kfile, "# ========================================================\n");
    std::fprintf(kfile, "# Lambda-kernel snapshot  step=%-10lld  M=%-6u  totalN=%-8.0f\n",
                 (long long)getStep(), M_, totalN_);
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

    for (unsigned kk = 0; kk < M_; ++kk) {
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
               path.c_str(), M_, (long long)getStep());
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
  // czar_integrate.py can reconstruct A(z) without access to the input file.
  void writeCZARFile() {
    if (czarFile_.empty() || zM_ == 0) return;
    std::string path = stampedPath(czarFile_);
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return;

    std::fprintf(f, "# CZAR z-kernel file -- ForceKernelABF v3.1.0\n");
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
    std::fprintf(f, "nkernels %u\n", zM_);
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
    for (unsigned k = 0; k < zM_; ++k) {
      std::fprintf(f, "%.6f", zKernels_[k].Nk);
      for (unsigned i = 0; i < dim_; ++i) std::fprintf(f, " %.10f", zKernels_[k].center[i]);
      for (unsigned i = 0; i < dim_; ++i) std::fprintf(f, " %.10f", zKernels_[k].mu[i]);
      for (unsigned i = 0; i < dim_; ++i) std::fprintf(f, " %.10f", zKernels_[k].sigma[i]);
      std::fprintf(f, "\n");
    }
    std::fclose(f);
    log.printf("  [FKERNELABF] CZAR file written: %s  (%u z-kernels, step %lld)\n",
               path.c_str(), zM_, (long long)getStep());
  }

  // ================ restart state I/O ================
  // The state file contains everything needed to resume a simulation:
  //   - fictitious particle position and velocity
  //   - adaptive sigma state
  //   - lambda-kernel population (center, mu, sigma, Nk)
  //   - z-kernel population (center, mu, sigma, Nk)
  //   - Z0_density for exploration
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

    f << "# FKERNELABF state file v3.1.0\n";
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

    f << "M " << M_ << "\n";
    f << "totalN " << totalN_ << "\n";
    f << "sumNk2 " << sumNk2_ << "\n";
    f << "# lambda-kernels: Nk center[0..d-1] mu[0..d-1] sigma[0..d-1]\n";
    for (unsigned k = 0; k < M_; ++k) {
      f << "K " << kernels_[k].Nk;
      for (unsigned i = 0; i < dim_; ++i) f << " " << kernels_[k].center[i];
      for (unsigned i = 0; i < dim_; ++i) f << " " << kernels_[k].mu[i];
      for (unsigned i = 0; i < dim_; ++i) f << " " << kernels_[k].sigma[i];
      f << "\n";
    }

    f << "zM " << zM_ << "\n";
    f << "zTotalN " << zTotalN_ << "\n";
    f << "zSumNk2 " << zSumNk2_ << "\n";
    f << "# z-kernels: Nk center[0..d-1] mu[0..d-1] sigma[0..d-1]\n";
    for (unsigned k = 0; k < zM_; ++k) {
      f << "Z " << zKernels_[k].Nk;
      for (unsigned i = 0; i < dim_; ++i) f << " " << zKernels_[k].center[i];
      for (unsigned i = 0; i < dim_; ++i) f << " " << zKernels_[k].mu[i];
      for (unsigned i = 0; i < dim_; ++i) f << " " << zKernels_[k].sigma[i];
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
                 stateFile_.c_str(), M_, zM_, (long long)getStep());
  }

  // Helper for readState(): aborts with a descriptive error if the stream
  // is in a failed state after parsing 'description'.  readState uses this
  // pattern inline for most fields; this helper exists for any future
  // consolidation where the repeated literal string would be too verbose.
  void parseCheck(const std::istringstream& iss, const std::string& description) {
    if (iss.fail())
      error("RESTART state file: failed to parse " + description
            + " — file may be truncated or corrupted.");
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
    kernels_.clear(); M_ = 0; totalN_ = 0; sumNk2_ = 0;
    zKernels_.clear(); zM_ = 0; zTotalN_ = 0; zSumNk2_ = 0;

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
        iss >> M_;
      } else if (key == "totalN") {
        iss >> totalN_;
        if (!iss) error("RESTART state file: failed to parse 'totalN' — file may be truncated or corrupted.");
      } else if (key == "sumNk2") {
        iss >> sumNk2_;
        if (!iss) error("RESTART state file: failed to parse 'sumNk2' — file may be truncated or corrupted.");
      } else if (key == "zM") {
        iss >> zM_;
      } else if (key == "zTotalN") {
        iss >> zTotalN_;
        if (!iss) error("RESTART state file: failed to parse 'zTotalN' — file may be truncated or corrupted.");
      } else if (key == "zSumNk2") {
        iss >> zSumNk2_;
        if (!iss) error("RESTART state file: failed to parse 'zSumNk2' — file may be truncated or corrupted.");
      } else if (key == "K") {
        // Lambda-kernel data line
        Kernel nk;
        nk.center.resize(dim_); nk.mu.resize(dim_); nk.sigma.resize(dim_);
        iss >> nk.Nk;
        for (unsigned i = 0; i < dim_; ++i) iss >> nk.center[i];
        for (unsigned i = 0; i < dim_; ++i) iss >> nk.mu[i];
        for (unsigned i = 0; i < dim_; ++i) iss >> nk.sigma[i];
        if (!iss) error("RESTART state file: failed to parse kernel 'K' row — file may be truncated or corrupted.");
        if (nk.Nk <= 0)
          error("RESTART state file: kernel 'K' has Nk=" + std::to_string(nk.Nk)
                + " <= 0. Corrupted state file.");
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
        // Z-kernel data line
        ZKernel nk;
        nk.center.resize(dim_); nk.mu.resize(dim_); nk.sigma.resize(dim_);
        iss >> nk.Nk;
        for (unsigned i = 0; i < dim_; ++i) iss >> nk.center[i];
        for (unsigned i = 0; i < dim_; ++i) iss >> nk.mu[i];
        for (unsigned i = 0; i < dim_; ++i) iss >> nk.sigma[i];
        if (!iss) error("RESTART state file: failed to parse z-kernel 'Z' row — file may be truncated or corrupted.");
        if (nk.Nk <= 0)
          error("RESTART state file: z-kernel 'Z' has Nk=" + std::to_string(nk.Nk)
                + " <= 0. Corrupted state file.");
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
        // The rest of the line contains the serialized state (624 words + index).
        // Old state files without this line will just use a fresh random seed.
        iss >> rng_;
      } else if (key == "END") {
        break;
      }
      // Unknown keys (from old state file versions, etc.)
      // are silently skipped for backward compatibility.
    }

    // Validate
    if (kernels_.size() != M_) {
      log.printf("  [FKERNELABF] WARNING: state file M=%u but read %lu lambda-kernels. "
                 "Using actual count.\n", M_, (unsigned long)kernels_.size());
      M_ = kernels_.size();
    }
    if (zKernels_.size() != zM_) {
      log.printf("  [FKERNELABF] WARNING: state file zM=%u but read %lu z-kernels. "
                 "Using actual count.\n", zM_, (unsigned long)zKernels_.size());
      zM_ = zKernels_.size();
    }

    // Populate neighbor list indices for all restored kernels
    if (nlist_) {
      nlistIdx_.clear();
      for (unsigned k = 0; k < M_; ++k) nlistIdx_.push_back(k);
      nlistUpdate_ = true;
      znlistIdx_.clear();
      for (unsigned k = 0; k < zM_; ++k) znlistIdx_.push_back(k);
      znlistUpdate_ = true;
    }

    // Skip adaptive sigma warmup since we restored learned sigma
    if (adaptiveSigma_) {
      adaptiveSigma_ = false;
      adaptiveCounter_ = adaptiveSigmaStride_ + 1;
    }

    // Mark that first step should NOT re-initialise s_fict from z
    firstStep_ = false;

    log.printf("  [FKERNELABF] State restored: M=%u totalN=%.0f zM=%u zTotalN=%.0f "
               "Z0=%.2f sigma0=(%.5f",
               M_, totalN_, zM_, zTotalN_, Z0_density_, sigma0_[0]);
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
    keys.add("compulsory", "FRICTION","10.0","Langevin friction (1/time_unit).");
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
    // Deprecated keywords — silently consumed so old input files still parse.
    // ETA (biased learning rate): always 1.0 (exact mean) since v2.0.
    // N0, LINEARGATE: removed in v1.1; kept here for backward compatibility only.
    // LAMBDAMAX: hardcoded to 1.0 in v3.0.
    // VCLAMP: removed in v3.0; replaced by MUXCLAMP and MAXFORCE.
    keys.add("optional", "ETA",
             "Deprecated: ignored. Lambda-kernel mean force always uses exact running mean.");
    keys.add("optional", "N0", "Deprecated: ignored. Kept for backward compatibility.");
    keys.addFlag("LINEARGATE", false, "Deprecated: ignored. Kept for backward compatibility.");
    keys.add("optional", "LAMBDAMAX", "Deprecated in v3.0: hardcoded to 1.0.");
    keys.add("optional", "VCLAMP",   "Deprecated in v3.0: removed. Use MUXCLAMP and MAXFORCE.");

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

    // Force clamps (safety nets — defaults are generous for most systems)
    keys.add("compulsory", "MUXCLAMP", "500.0",
             "Per-kernel mean-force clamp (kJ/mol/rad). Individual kernel mu "
             "values are hard-clamped to [-MUXCLAMP, +MUXCLAMP]. Only fires "
             "for corrupted force samples at very sparse regions.");
    keys.add("compulsory", "MAXFORCE", "500.0",
             "Grid mean-force clamp (kJ/mol/rad). The NW mean force on the "
             "grid is clamped per-node before interpolation. Only fires "
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
             "Feed the final file to czar_integrate.py to recover A(z).");

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
      friction_(10.0), kT_(0.0),
      firstStep_(true),
      adaptiveSigma_(false), adaptiveSigmaStride_(0), adaptiveCounter_(0),
      rng_(std::random_device{}()), gauss_(0.0, 1.0),
      pace_(5), thresh_(1.0), nsigmaCut_(4.0),
      biasFactor_(1.0), Z0_density_(1.0),
      muxClamp_(500), maxForce_(500),
      fixedSigma_(false),
      M_(0), totalN_(0), sumNk2_(0.0),
      zM_(0), zTotalN_(0.0), zSumNk2_(0.0),
      nlist_(true), nlistCutFactor_(3.0), nlistSkinFactor_(0.5),
      nlistUpdate_(true),
      znlistUpdate_(true),
      gridPace_(500), gridTotal_(0), gridSize_(72),
      lambdaGridStride_(0),
      kernelStride_(0),
      czarStride_(0),
      stateStride_(0),
      kernelInfoStride_(0) {

    dim_ = getNumberOfArguments();
    if (dim_ < 1 || dim_ > 3)
      error("FKERNELABF supports 1 to 3 CVs.");

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
    parse("FRICTION", friction_);

    if (temp <= 0.0) error("TEMP must be > 0.");
    if (friction_ < 0.0) error("FRICTION must be >= 0.");
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


    // Silently consume deprecated keywords so old input files still parse.
    {
      double eta_compat = 1.0; parse("ETA", eta_compat);
      double N0_compat  = 0.0; parse("N0",  N0_compat);
      bool   lg_compat  = false; parseFlag("LINEARGATE", lg_compat);
      if (eta_compat != 1.0)
        log.printf("  [FKERNELABF] WARNING: ETA is deprecated and ignored since v2.0. "
                   "Lambda-kernel mean force always uses the exact running mean.\n");
      if (N0_compat > 0.0 || lg_compat)
        log.printf("  [FKERNELABF] NOTE: N0 and LINEARGATE are deprecated and ignored since v2.0.\n");

      // v3.0: LAMBDAMAX hardcoded to 1.0; VCLAMP removed.
      double lmax_compat = 0; parse("LAMBDAMAX", lmax_compat);
      double vclamp_compat = 0; parse("VCLAMP", vclamp_compat);
      if (lmax_compat > 0)
        log.printf("  [FKERNELABF] NOTE: LAMBDAMAX is deprecated in v3.0 (hardcoded to 1.0).\n");
      if (vclamp_compat > 0)
        log.printf("  [FKERNELABF] NOTE: VCLAMP is deprecated in v3.0. "
                   "Use MUXCLAMP and MAXFORCE directly.\n");
    }

    parse("BIASFACTOR", biasFactor_);
    if (biasFactor_ < 1.0) error("BIASFACTOR must be >= 1.0 (1.0 = pure ABF).");

    parse("MUXCLAMP", muxClamp_);
    parse("MAXFORCE", maxForce_);
    if (muxClamp_ <= 0.0) error("MUXCLAMP must be > 0.");
    if (maxForce_ <= 0.0) error("MAXFORCE must be > 0.");

    parse("GRIDSIZE", gridSize_); parse("GRIDPACE", gridPace_);
    if (gridSize_ < 2) error("GRIDSIZE must be >= 2.");

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

    gridN_.assign(dim_, gridSize_);
    gridDx_.resize(dim_);
    gridTotal_ = 1;
    for (unsigned d = 0; d < dim_; ++d) {
      gridDx_[d] = periodic_[d] ? domLen_[d]/(double)gridN_[d]
                                : (gridMax_[d]-gridMin_[d])/(double)(gridN_[d]-1);
      gridTotal_ *= gridN_[d];
    }
    ghat_.assign(gridTotal_*dim_, 0.0);
    Zgrid_.assign(gridTotal_, 0.0);
    fex_.assign(gridTotal_*dim_, 0.0);

    // Derive all output filenames from the action label.
    // Users only specify strides; paths are not configurable.
    std::string lbl = getLabel();
    lambdaGridFile_ = "";          // populated below only if LAMBDAGRIDSTRIDE > 0
    lambdaGridStride_ = 0; parse("LAMBDAGRIDSTRIDE", lambdaGridStride_);
    if (lambdaGridStride_ > 0) lambdaGridFile_ = lbl + ".lambda_grid.dat";

    kernelFile_ = "";
    kernelStride_ = 0; parse("KERNELSTRIDE", kernelStride_);
    if (kernelStride_ > 0) kernelFile_ = lbl + ".kernels.dat";

    kernelInfoFile_ = "KERNELINFO";
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

    log.printf("  [FKERNELABF v3.1.0] Force-kernel ABF + Kernel CZAR + density-based exploration (BAOAB, direct mean force)\n");
    log.printf("  [FKERNELABF] CVs: ");
    for (unsigned i = 0; i < dim_; ++i)
      log.printf("%s%s", i?", ":"", getPntrToArgument(i)->getName().c_str());
    log.printf("\n  [FKERNELABF] KAPPA=");
    for (unsigned i = 0; i < dim_; ++i) log.printf("%s%.1f", i?",":"", kappa_[i]);
    log.printf("  mass=");
    for (unsigned i = 0; i < dim_; ++i) log.printf("%s%.6f", i?",":"", mass_[i]);
    log.printf("\n  [FKERNELABF] friction=%.2f  kT=%.4f  temp=%.1f\n", friction_, kT_, temp);
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
    } else
      log.printf("  [FKERNELABF] BIASFACTOR=1.0  pure ABF, no exploration\n");
    for (unsigned i = 0; i < dim_; ++i)
      log.printf("  [FKERNELABF] dim %u: %s grid=[%.4f, %.4f] dx=%.6f (%u pts)\n",
                 i, periodic_[i]?"periodic":"non-periodic",
                 gridMin_[i], gridMax_[i], gridDx_[i], gridN_[i]);
    log.printf("  [FKERNELABF] grid update every %u steps  THRESH=%.2f  NSIGMACUT=%.1f\n",
               gridPace_, thresh_, nsigmaCut_);
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

    // ── Restart: read state from file if PLUMED restart is active ──────
    // Usage: add RESTART to the PLUMED input file, or pass --restart to
    // the MD engine.  The state file {label}.state.dat must exist.
    // After reading, the mean-force grid is reconstructed immediately.
    if (getRestart()) {
      readState();
      // Force immediate mean-force grid reconstruction from restored kernels
      if (M_ > 0) reconstructBiasGrid();
    }
  }

  void calculate() override {
    double dt = getTimeStep();
    std::vector<double> s(dim_);
    for (unsigned i = 0; i < dim_; ++i) s[i] = getArgument(i);

    if (firstStep_) {
      for (unsigned i = 0; i < dim_; ++i) { s_fict_[i] = s[i]; v_fict_[i] = 0.0; }
      if (adaptiveSigma_) {
        av_cv_.assign(dim_, 0.0);
        av_M2_.assign(dim_, 0.0);
      }
      firstStep_ = false;
    }

    // ── Adaptive sigma warmup ─────────────────────────────────────────────────
    // During the warmup window: zero bias, no kernel deposition, collect variance.
    // s_fict_ tracks z directly so it starts at the right position when bias begins.
    if (adaptiveSigma_ && adaptiveCounter_ < adaptiveSigmaStride_) {
      adaptiveCounter_++;
      unsigned tau = adaptiveCounter_;
      for (unsigned i = 0; i < dim_; ++i) {
        double diff = periodicDelta(i, av_cv_[i], s[i]);
        av_cv_[i] += diff / (double)tau;
        av_M2_[i] += diff * periodicDelta(i, av_cv_[i], s[i]);
      }
      for (unsigned i = 0; i < dim_; ++i) setOutputForce(i, 0.0);
      setBias(0.0);
      for (unsigned i = 0; i < dim_; ++i) s_fict_[i] = s[i];
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
    }

    // (A) Spring force: kappa(z - lambda) — raw, unclamped
    std::vector<double> springF(dim_);
    for (unsigned i = 0; i < dim_; ++i) {
      double d = periodicDelta(i, s_fict_[i], s[i]);  // = z - lambda
      springF[i] = kappa_[i] * d;
      setOutputForce(i, -kappa_[i] * d);  // force on z = kappa(lambda - z)
    }

    // (B) Neighbor list
    if (nlist_ && M_ > 0) {
      if (nlistUpdate_ || needsNlistUpdate(s_fict_))
        updateNlist(s_fict_);
    }

    // (C) Accumulate lambda-kernel sample at s_fict_ (force clamped inside addSample).
    // (C2) Accumulate z-kernel sample at real CV z (unclamped, exact running mean).
    // Both happen at the same stride so lambda-kernel and z-kernel populations grow together.
    if (pace_ > 0 && getStep() % pace_ == 0) {
      addSample(s_fict_, springF);
      addZSample(s, springF);
    }

    // (D) Reconstruct NW mean-force grid from lambda-kernels
    if (gridPace_ > 0 && M_ > 0 && getStep() % gridPace_ == 0)
      reconstructBiasGrid();

    // (E) ABF cancellation + exploration forces from grids.
    //
    // All forces on λ are interpolated from grids updated at GRIDPACE intervals.
    // Between rebuilds, the total force is a smooth, frozen function of position.
    // This gives temporal stability for the BAOAB integrator and eliminates the
    // per-step O(nlist) kernel scan that previously dominated the cost.
    //
    // ABF force:   F_abf = -ghat(λ)         [from ghat_ grid]
    // Exploration:  F_ex from fex_ grid      [FD of Z, only when γ>1]
    //
    // Scalar diagnostics (V_ex, Neff) are computed after BAOAB at the final
    // s_fict_ position so they are synchronized with the reported lambda.
    double ghatF[3] = {};
    {
      std::vector<double> gf(dim_, 0.0);
      interpolateForce(s_fict_, gf);
      for (unsigned i = 0; i < dim_; ++i) ghatF[i] = gf[i];
    }

    double F_explore[3] = {0, 0, 0};
    if (biasFactor_ > 1.0) {
      std::vector<double> fe(dim_, 0.0);
      interpolateExplore(s_fict_, fe);
      for (unsigned i = 0; i < dim_; ++i) F_explore[i] = fe[i];
    }

    // (F) BAOAB Langevin dynamics for s_fict_.
    //
    // Second-order symmetric splitting (Leimkuhler & Matthews 2013):
    //   B – half kick with current force
    //   A – half drift
    //   O – Ornstein-Uhlenbeck thermostat (full dt)
    //   A – half drift
    //   B – half kick with force at updated position
    //
    // All forces (spring, ABF, exploration) are re-evaluated at the new s_fict_
    // for the closing half-kick. Since all bias forces come from frozen grids,
    // these re-evaluations are cheap (grid interpolation only).
    const double halfdt = 0.5 * dt;
    double fmag2 = 0;

    // ── B: half kick (current forces) ──────────────────────────────────────
    for (unsigned i = 0; i < dim_; ++i) {
      double F_abf   = -ghatF[i];   // direct NW mean force, no Poisson
      double F_total = springF[i] + F_abf + F_explore[i];
      fmag2 += sq(F_abf + F_explore[i]);
      v_fict_[i] += halfdt * F_total / mass_[i];
    }

    // ── A: half drift ──────────────────────────────────────────────────────
    for (unsigned i = 0; i < dim_; ++i)
      s_fict_[i] += halfdt * v_fict_[i];

    // ── O: Ornstein-Uhlenbeck thermostat (full dt) ─────────────────────────
    const double c1 = std::exp(-friction_ * dt);
    for (unsigned i = 0; i < dim_; ++i) {
      double c2 = std::sqrt(kT_ / mass_[i] * (1.0 - c1*c1));
      v_fict_[i] = c1 * v_fict_[i] + c2 * gauss_(rng_);
    }

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
    // Recompute all forces at new s_fict_ from grids (cheap: just interpolation).
    {
      std::vector<double> gf2(dim_, 0.0);
      interpolateForce(s_fict_, gf2);
      double fe2[3] = {0, 0, 0};
      if (biasFactor_ > 1.0) {
        std::vector<double> fev(dim_, 0.0);
        interpolateExplore(s_fict_, fev);
        for (unsigned i = 0; i < dim_; ++i) fe2[i] = fev[i];
      }
      for (unsigned i = 0; i < dim_; ++i) {
        double d2 = periodicDelta(i, s_fict_[i], s[i]);  // z - lambda_new
        double springF2 = kappa_[i] * d2;
        double F_abf2   = -gf2[i];
        double F_total2 = springF2 + F_abf2 + fe2[i];
        v_fict_[i] += halfdt * F_total2 / mass_[i];
      }
    }

    // (G) Post-BAOAB diagnostics at the final s_fict_ position.
    //
    // V_ex and Neff are evaluated here (not before BAOAB) so that the
    // reported scalars are synchronized with the reported lambda position.
    double V_ex = 0.0;
    double Neff = 0.0;
    if (biasFactor_ > 1.0) {
      double Z_final = interpolateScalar(s_fict_, Zgrid_);
      double c_ex = kT_ * (biasFactor_ - 1.0);
      V_ex = c_ex * std::log(1.0 + Z_final / (Z0_density_ + 1e-300));
      Neff = Z_final;
    } else {
      Neff = interpolateScalar(s_fict_, Zgrid_);
    }

    setBias(V_ex);
    getPntrToComponent("wamp")->set(V_ex);
    getPntrToComponent("force2")->set(fmag2);
    for (unsigned i = 0; i < dim_; ++i)
      getPntrToComponent(fictNames_[i])->set(s_fict_[i]);

    // (I) Lambda-grid file output (debug: NW mean force on λ)
    if (!lambdaGridFile_.empty() && lambdaGridStride_ > 0 && M_ > 0 && getStep() % lambdaGridStride_ == 0)
      writeGridFile();

    // (J) lambda-kernel dump
    dumpKernelsIfNeeded();

    // (K) CZAR z-kernel file (step-stamped snapshot; never overwritten)
    if (!czarFile_.empty() && czarStride_ > 0 && zM_ > 0 && getStep() % czarStride_ == 0)
      writeCZARFile();

    // (L) Restart state (overwritten in place, no backups)
    if (stateStride_ > 0 && M_ > 0 && getStep() % stateStride_ == 0)
      writeState();

    // (M) Kernel diagnostics file (appending time series)
    if (kernelInfoStride_ > 0) writeKernelInfo();
  }
};

PLUMED_REGISTER_ACTION(ForceKernelABF, "FKERNELABF")

}  
}  
