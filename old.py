#!/usr/bin/env python3
"""
czar_integrate.py
=================
Recover the true free energy landscape (FEL) along the real CV z from
the z-kernel CZAR file written by FKERNELABF v2+.

Theory
------
CZAR (Lesage et al. J. Phys. Chem. B 2017) gives the mean force along the
REAL CV z as:

    dA/dz_i = <kappa_i*(lambda_i - z_i)>_z  -  kT * d/dz_i ln p̃(z)

In terms of quantities stored in the z-kernel file
(mu_k stores kappa*(z - lambda), the negative of the CZAR spring term):

    dA/dz_i = -mu_NW_i(z)  −  kT * d/dz_i ln p̃(z)

    mu_NW_i is the Nadaraya-Watson mean of mu, from kernel regression.
    d/dz ln p̃ is computed via finite differences of log(ptilde) on the grid,
    matching abf_integrate / DRR (CZAR::getGradient line 484).

where
    G_k(z) = exp( -sum_d (z_d - c_kd)^2 / (4*sigma_kd^2) )
    p̃(z)  = sum_k Nk * G_k(z)
    w_k(z) = Nk * G_k(z) / p̃(z)          (normalised weights)
    mu_NW_i(z) = sum_k w_k(z) * mu_ki      (Nadaraya-Watson regression)

Both terms are evaluated analytically from the kernel list — no bins anywhere.
The kernel cutoff is applied per-dimension at NSIGMA*sigma (default 4).

After computing the gradient field, A(z) is recovered by:
  • 1-D: cumulative trapezoid integration (exact).
  • 2-D+ (default): Poisson FFT — solves ∇²A = div(grad A) with even-extension
          for non-periodic dimensions.  Instant (<1 second).
  • 2-D+ (--mc flag): metadynamics-style MC random walk on the gradient grid,
          following abf_integrate (Hénin, Colvars).  Slower but handles
          non-conservative gradient fields.  Use --mc_steps to set step count.

Notes
-----
v2.0 introduces WTM-eABF flooding: the additional force −α·∇V(λ) acts on the
fictitious variable λ only.  The z-kernels and the CZAR estimator are
completely unaffected by this change — this script requires no modification
to handle v2+ output.  The CZAR identity dA/dz_i = <κ(λ_i−z_i)>_z + kT·∂_z ln p̃
holds exactly regardless of whether the flooding component is active.

Usage
-----
    python czar_integrate.py --czar fk.czar_kernels_00100000.dat [OPTIONS]

Options
-------
    --czar FILE         CZAR kernel file (required)
    --grid N            Grid points per dimension (default: 100)
    --nsigma N          Kernel cutoff in sigma units per dimension (default: 4.0)
    --output FILE       Output FEL file (default: FEL_czar.dat)
    --min V [V ...]     Override grid min per dim (use for non-periodic if needed)
    --max V [V ...]     Override grid max per dim
    --kT VALUE          Override kT from file (kJ/mol)
    --minpop FRAC       Display mask: NaN where p̃(z) < FRAC*max(p̃) (default 1e-5)
                        Only affects output, NOT the MC integration.
    --verbose           Print extra diagnostics

Output format (space-separated):
    z0 [z1 [z2]]   czar_grad0 [czar_grad1 [czar_grad2]]   ptilde   A_czar
    (A_czar set to NaN where p̃ < minpop threshold; integration is unaffected)

Examples
--------
    # Alanine dipeptide: phi/psi (both periodic -pi..pi)
    python czar_integrate.py --czar fk.czar_kernels_01000000.dat --grid 100 --output FEL.dat

    # Non-periodic CV, override grid range
    python czar_integrate.py --czar fk.czar_kernels_01000000.dat --grid 150 \\
        --min -2.0 --max 2.0 --output FEL.dat

    # Convergence: compare all snapshots against final reference
    python czar_integrate.py --czar fk.czar_kernels_01000000.dat \\
        --convergence --conv_output czar_convergence.dat

    # Convert all snapshots to simple z0 z1 A(kJ/mol) files (like OPES)
    python czar_integrate.py --czar fk.czar_kernels_01000000.dat \\
        --process_all --fel_dir ./FEL_snapshots/
"""

import argparse
import sys
import numpy as np


# ─────────────────────── file reader ─────────────────────────────────────────

def parse_czar_file(path):
    """
    Parse the CZAR kernel file written by FKERNELABF v2+.

    Returns
    -------
    meta : dict   with keys: dim, kT, kappa, periodic, domMin, domMax, nkernels
    kernels : list of dicts  with keys: Nk, center, mu, sigma   (all 1-D arrays)
    """
    meta = {}
    kernels = []
    dim = None

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            key = parts[0]

            if key == 'dim':
                dim = int(parts[1])
                meta['dim'] = dim
            elif key == 'kT':
                meta['kT'] = float(parts[1])
            elif key == 'kappa':
                meta['kappa'] = np.array([float(x) for x in parts[1:]])
            elif key == 'periodic':
                meta['periodic'] = np.array([bool(int(x)) for x in parts[1:]])
            elif key == 'domMin':
                meta['domMin'] = np.array([float(x) for x in parts[1:]])
            elif key == 'domMax':
                meta['domMax'] = np.array([float(x) for x in parts[1:]])
            elif key == 'nkernels':
                meta['nkernels'] = int(parts[1])
            else:
                # Data line: Nk  center[0..d-1]  mu[0..d-1]  sigma[0..d-1]
                if dim is None:
                    continue
                vals = [float(x) for x in parts]
                if len(vals) != 1 + 3 * dim:
                    continue   # malformed line, skip
                Nk     = vals[0]
                center = np.array(vals[1       : 1+dim])
                mu     = np.array(vals[1+dim   : 1+2*dim])
                sigma  = np.array(vals[1+2*dim : 1+3*dim])
                kernels.append({'Nk': Nk, 'center': center, 'mu': mu, 'sigma': sigma})

    if not kernels:
        sys.exit(f"ERROR: no kernel data found in {path}")
    if 'kT' not in meta:
        sys.exit("ERROR: kT not found in CZAR file header")

    for key in ('kT', 'kappa', 'periodic', 'domMin', 'domMax'):
        if key not in meta:
            print(f"WARNING: '{key}' not found in CZAR file — using defaults may be unsafe.")

    return meta, kernels


# ─────────────────────── grid construction ────────────────────────────────────

def build_grid(meta, args):
    """
    Build the evaluation grid, honouring periodic domains and user overrides.

    Returns: coords_per_dim (list of 1-D arrays), periodic (bool array),
             gmin (array), gmax (array)
    """
    dim = meta['dim']
    periodic = meta.get('periodic', np.zeros(dim, dtype=bool))
    domMin   = meta.get('domMin',   np.zeros(dim))
    domMax   = meta.get('domMax',   np.ones(dim))

    gmin = np.array(args.min) if args.min else domMin.copy()
    gmax = np.array(args.max) if args.max else domMax.copy()

    if len(gmin) != dim or len(gmax) != dim:
        sys.exit(f"ERROR: --min/--max must have {dim} values")

    coords = []
    for d in range(dim):
        if periodic[d]:
            # For periodic CVs, use N points evenly covering the period
            # (exclude the endpoint to avoid double-counting)
            pts = np.linspace(gmin[d], gmax[d], args.grid, endpoint=False)
        else:
            pts = np.linspace(gmin[d], gmax[d], args.grid)
        coords.append(pts)

    return coords, periodic, gmin, gmax


# ─────────────────────── kernel evaluation (core) ────────────────────────────

def periodic_delta(delta, period):
    """Wrap delta into (-period/2, period/2]."""
    return delta - period * np.round(delta / period)


def czar_on_grid(coords, periodic, meta, kernels, kT, nsigma, verbose=False):
    """
    Evaluate the CZAR gradient field and biased density p̃(z) on the grid.

    Step 1: Evaluate NW mean spring force and density ptilde from kernels.
    Step 2: Compute d/dz_i ln(ptilde) via finite differences on the gridded
            log-density (matching abf_integrate's approach).
    Step 3: Assemble CZAR gradient: dA/dz_i = -mu_NW_i + kT * d/dz_i ln(ptilde)

    Parameters
    ----------
    coords   : list of 1-D arrays, one per dim
    periodic : bool array, length dim
    meta     : dict with domMin, domMax, etc.
    kernels  : list of kernel dicts
    kT       : float
    nsigma   : float, per-dimension cutoff in sigma units

    Returns
    -------
    ptilde    : ndarray, shape = (*grid_shape,)
    czar_grad : ndarray, shape = (*grid_shape, dim)
    """
    dim = len(coords)
    shape = tuple(len(c) for c in coords)
    dx = np.array([c[1] - c[0] if len(c) > 1 else 1.0 for c in coords])

    # --- Period from domain metadata (stable under grid overrides) -----------
    domMin = meta.get('domMin', np.zeros(dim))
    domMax = meta.get('domMax', np.ones(dim))
    period = np.zeros(dim)
    for d in range(dim):
        if periodic[d]:
            period[d] = domMax[d] - domMin[d]

    # --- Meshgrid of evaluation points: shape (*shape, dim) ------------------
    mg = np.meshgrid(*coords, indexing='ij')
    z_grid = np.stack(mg, axis=-1)  # (*shape, dim)

    ptilde   = np.zeros(shape)
    sum_wkmu = np.zeros((*shape, dim))

    # --- Kernel loop: accumulate ptilde and NW numerator ---------------------
    n_total = len(kernels)
    report_every = max(1, n_total // 10)

    for ki, kern in enumerate(kernels):
        if verbose and ki % report_every == 0:
            print(f"  Processing kernel {ki+1}/{n_total} ...", flush=True)

        Nk     = kern['Nk']
        center = kern['center']
        mu_k   = kern['mu']
        sigma  = kern['sigma']

        # --- Per-dimension range clipping (box cutoff) -----------------------
        slices = []
        skip = False
        for d in range(dim):
            if periodic[d] and period[d] > 0:
                slices.append(slice(None))
            else:
                halfwidth = 2.0 * nsigma * sigma[d]
                lo = center[d] - halfwidth
                hi = center[d] + halfwidth
                c = coords[d]
                idx_lo = np.searchsorted(c, lo, side='left')
                idx_hi = np.searchsorted(c, hi, side='right')
                if idx_lo >= idx_hi:
                    skip = True
                    break
                slices.append(slice(idx_lo, idx_hi))
        if skip:
            continue

        slices_tuple = tuple(slices)
        z_sub = z_grid[slices_tuple]
        sub_shape = z_sub.shape[:-1]

        dz = z_sub - center
        for d in range(dim):
            if periodic[d] and period[d] > 0:
                dz[..., d] = periodic_delta(dz[..., d], period[d])

        inv4sig2 = 1.0 / (4.0 * sigma**2 + 1e-300)
        exponent = np.einsum('...d,d->...', dz**2, inv4sig2)

        per_dim_exp = dz**2 * inv4sig2
        mask = np.all(per_dim_exp < nsigma**2, axis=-1)

        if not np.any(mask):
            continue

        Gk = np.zeros(sub_shape)
        Gk[mask] = np.exp(-exponent[mask])

        wk = Nk * Gk

        ptilde[slices_tuple]   += wk
        sum_wkmu[slices_tuple] += wk[..., np.newaxis] * mu_k

    # --- NW mean spring force ------------------------------------------------
    safe_ptilde = np.where(ptilde > 0, ptilde, 1.0)
    mu_NW = np.zeros((*shape, dim))
    for d in range(dim):
        mu_NW[..., d] = sum_wkmu[..., d] / safe_ptilde

    # --- Finite-difference gradient of ln(ptilde) ----------------------------
    # This matches abf_integrate: it computes kT * d/dx ln(histogram) as the
    # entropy correction.  We do FD on ln(ptilde) directly (not FD(ptilde)/ptilde)
    # for numerical stability at sampling boundaries.
    p_floor = max(ptilde.max() * 1e-18, 1e-300)
    ln_p = np.log(np.maximum(ptilde, p_floor))

    grad_ln_ptilde = np.zeros((*shape, dim))
    for d in range(dim):
        N = shape[d]
        if N < 2:
            continue
        h = dx[d]
        # Work on the spatial-only 2D slice for this component
        gd = grad_ln_ptilde[..., d]  # view into (*shape,) array
        if periodic[d]:
            fp1 = np.roll(ln_p, -1, axis=d)
            fm1 = np.roll(ln_p, +1, axis=d)
            gd[...] = (fp1 - fm1) / (2.0 * h)
        else:
            def _slc(s):
                idx = [slice(None)] * dim
                idx[d] = s
                return tuple(idx)
            # Central differences interior
            if N >= 3:
                gd[_slc(slice(1, -1))] = \
                    (ln_p[_slc(slice(2, None))] - ln_p[_slc(slice(None, -2))]) / (2.0 * h)
                # Boundaries: one-sided
                gd[_slc(0)]  = (ln_p[_slc(1)] - ln_p[_slc(0)]) / h
                gd[_slc(-1)] = (ln_p[_slc(-1)] - ln_p[_slc(-2)]) / h
            else:
                gd[_slc(0)] = (ln_p[_slc(1)] - ln_p[_slc(0)]) / h
                gd[_slc(1)] = gd[_slc(0)]

        # Zero out where density is truly zero
        gd[...] = np.where(ptilde > 0, gd, 0.0)

    # --- Assemble CZAR gradient ----------------------------------------------
    # From the derivation (confirmed by DRR CZAR::getGradient, line 484):
    #
    #   ρ̃(z) = C · e^{-βA(z)} · Q(z)   where Q = ∫ e^{-β[κ/2(z-λ)² + V(λ)]} dλ
    #   ∂ ln ρ̃/∂z = -β ∂A/∂z + β<κ(λ-z)>_z
    #   ∂A/∂z = <κ(λ-z)>_z − kT · ∂/∂z ln ρ̃(z)
    #         = -mu_NW      − kT · grad_ln_ptilde
    #
    # (mu stores κ(z-λ), so -mu_NW = <κ(λ-z)>_z)
    # The log-density term has a MINUS sign — it opposes the density gradient.
    czar_grad = np.zeros((*shape, dim))
    for d in range(dim):
        czar_grad[..., d] = -mu_NW[..., d] - kT * grad_ln_ptilde[..., d]

    return ptilde, czar_grad


# ─────────────────────── MC integration (à la abf_integrate) ─────────────────

def mc_integrate(czar_grad, coords, periodic, allowed, kT,
                 nsteps=0, temp_mc=500.0, hill=0.01, hill_factor=0.5,
                 hill_min=0.0005, verbose=False):
    """
    Integrate a gradient field to recover A(z), using the metadynamics-style
    Monte Carlo random walk from abf_integrate (Hénin, Colvars).

    Algorithm
    ---------
    A Metropolis MC walker moves on the gradient grid.  At each visited site
    a small "hill" is deposited into a bias array.  Moves are proposed as
    random ±1 steps along each axis, and accepted with probability

        min(1, exp(-dA / kT_mc))

    where  dA = grad · displacement  +  bias(new) − bias(old).

    The bias flattens the sampling and converges to −A(z) (up to a constant).
    Convergence is monitored via the RMSD between the input gradient and the
    finite-difference gradient of the current bias, exactly as abf_integrate
    does.

    Parameters
    ----------
    czar_grad : ndarray (*shape, dim)   – gradient field on the grid
    coords    : list of 1-D arrays      – grid coordinates per dim
    periodic  : bool array              – periodic flags per dim
    allowed   : bool ndarray (*shape,)  – mask of grid points with data
    kT        : float                   – physical kT (for gradient units)
    nsteps    : int                     – MC steps (0 = auto-converge)
    temp_mc   : float                   – MC temperature (controls acceptance)
    hill      : float                   – initial hill height
    hill_factor : float                 – hill reduction factor (0 = fixed)
    hill_min  : float                   – minimum hill height
    verbose   : bool

    Returns
    -------
    A : ndarray (*shape,)   – integrated free energy, min-shifted to 0
    """
    dim = len(coords)
    shape = tuple(len(c) for c in coords)
    sizes = np.array(shape, dtype=int)
    dx = np.array([c[1] - c[0] if len(c) > 1 else 1.0 for c in coords])
    scalar_dim = int(np.prod(sizes))

    # MC inverse temperature (in gradient units, NOT physical kT)
    kT_mc = 0.001987 * temp_mc  # kcal/mol — but gradients are in kJ/mol
    # Actually: abf_integrate uses kcal/mol internally. Our grads are in kJ/mol.
    # We use the physical kT from the simulation for the MC temperature scaling.
    # The MC temperature just needs to be high enough for good acceptance.
    # Use kT directly as the MC thermal energy unit.
    mbeta = -1.0 / kT  # -1/(kT) in same units as czar_grad

    # Flatten gradient for fast indexing: grad_flat[offset * dim + d]
    grad_flat = czar_grad.reshape(-1, dim)
    allowed_flat = allowed.ravel()

    bias = np.zeros(scalar_dim)
    histogram = np.zeros(scalar_dim, dtype=int)

    # Allowed-point list for random starting position
    allowed_indices = np.where(allowed_flat)[0]
    if len(allowed_indices) == 0:
        print("  WARNING: no allowed grid points. Returning zeros.", flush=True)
        return np.zeros(shape)

    rng = np.random.default_rng()

    # ── Helper: multi-index ↔ flat offset ────────────────────────────────────
    def flat_to_multi(offset):
        idx = np.empty(dim, dtype=int)
        tmp = offset
        for d in range(dim - 1, -1, -1):
            idx[d] = tmp % sizes[d]
            tmp //= sizes[d]
        return idx

    def multi_to_flat(idx):
        off = 0
        for d in range(dim):
            off = off * sizes[d] + idx[d]
        return off

    def wrap(val, d):
        """Wrap index for periodic, clamp for non-periodic. Return wrapped value."""
        if periodic[d]:
            return val % sizes[d]
        else:
            # Non-periodic: if out of bounds, return the edge (move will be rejected)
            if val < 0 or val >= sizes[d]:
                return -1  # sentinel: out of bounds
            return val

    # ── Pick random starting position ────────────────────────────────────────
    pos = flat_to_multi(rng.choice(allowed_indices))
    offset = multi_to_flat(pos)

    # ── Convergence settings ─────────────────────────────────────────────────
    if nsteps > 0:
        out_freq = max(1, nsteps // 10)
        scale_hill_step = nsteps // 2
        converged = True  # will run exactly nsteps
    else:
        out_freq = 1_000_000
        scale_hill_step = 2000 * scalar_dim
        nsteps = 2 * scale_hill_step
        converged = False
        if verbose:
            print(f"  MC auto-converge: min steps = {nsteps}", flush=True)

    convergence_limit = -0.001

    # ── Compute initial RMSD ─────────────────────────────────────────────────
    def compute_rmsd():
        """RMSD between input gradient and FD gradient of current bias."""
        rmsd_sum = 0.0
        norm = 0
        for off in allowed_indices:
            p = flat_to_multi(off)
            for d in range(dim):
                est = 0.0
                c = 0
                # Backward neighbor
                nm = p.copy(); nm[d] -= 1
                w = wrap(nm[d], d)
                if w >= 0:
                    nm[d] = w
                    noff = multi_to_flat(nm)
                    if allowed_flat[noff]:
                        est += (bias[noff] - bias[off]) / dx[d]
                        c += 1
                # Forward neighbor
                np2 = p.copy(); np2[d] += 1
                w = wrap(np2[d], d)
                if w >= 0:
                    np2[d] = w
                    noff = multi_to_flat(np2)
                    if allowed_flat[noff]:
                        est += (bias[off] - bias[noff]) / dx[d]
                        c += 1
                if c > 0:
                    est /= c
                dev = grad_flat[off, d] - est
                rmsd_sum += dev * dev
                norm += 1
        return np.sqrt(rmsd_sum / max(norm, 1))

    rmsd = compute_rmsd()
    if verbose:
        print(f"  MC initial gradient RMSD: {rmsd:.6f}", flush=True)

    # ── Main MC loop ─────────────────────────────────────────────────────────
    total = 0
    step = 0
    while step < nsteps or not converged:
        step += 1

        # Convergence check and hill scheduling
        if step % out_freq == 0:
            rmsd_old = rmsd
            rmsd = compute_rmsd()
            rmsd_rel_change = (rmsd - rmsd_old) / (rmsd_old * out_freq + 1e-300) * 1e6
            if verbose:
                msg = f"  MC step {step:>10d}  RMSD={rmsd:.6f}  rel_change/1M={rmsd_rel_change:.4f}"
                if hill_factor and step > scale_hill_step and hill > hill_min:
                    hill *= hill_factor
                    msg += f"  hill→{hill:.6f}"
                print(msg, flush=True)
            else:
                if hill_factor and step > scale_hill_step and hill > hill_min:
                    hill *= hill_factor
            if rmsd_rel_change > convergence_limit and step >= nsteps:
                converged = True

        offset = multi_to_flat(pos)
        histogram[offset] += 1
        bias[offset] += hill

        grad_here = grad_flat[offset]

        # Propose move
        not_accepted = True
        while not_accepted:
            total += 1
            dA = 0.0
            newpos = pos.copy()
            for d in range(dim):
                dp = rng.integers(-1, 2)  # -1, 0, or 1
                candidate = pos[d] + dp
                w = wrap(candidate, d)
                if w < 0:
                    dp = 0  # out of bounds, no move in this dim
                else:
                    newpos[d] = w
                    if newpos[d] == pos[d]:
                        dp = 0
                if dp != 0:
                    dA += grad_here[d] * dp * dx[d]

            newoffset = multi_to_flat(newpos)
            dA += bias[newoffset] - bias[offset]

            if allowed_flat[newoffset] and rng.random() < np.exp(mbeta * dA):
                pos = newpos
                not_accepted = False

    acceptance = step / max(total, 1)
    final_rmsd = compute_rmsd()
    print(f"  MC integration: {step} steps, {total} total proposals, "
          f"acceptance={acceptance:.3f}, final RMSD={final_rmsd:.6f}", flush=True)

    # PMF = -bias (bias fills wells, converging to -A + const)
    A = -bias.reshape(shape)
    if allowed.any():
        A -= A[allowed].min()

    return A


# ─────────────────────── 1-D integration (exact) ─────────────────────────────

def integrate_1d(grad, dx, is_periodic):
    """
    Cumulative trapezoid integration of a 1-D gradient.
    For 1-D the gradient field is automatically conservative, so direct
    integration is exact and preferred over MC.
    """
    N = len(grad)
    A = np.zeros(N)
    A[1:] = np.cumsum(0.5 * (grad[:-1] + grad[1:]) * dx)

    if is_periodic and N > 1:
        full_integral = A[-1] + 0.5 * (grad[-1] + grad[0]) * dx
        drift_per_pt = full_integral / N
        A -= np.arange(N) * drift_per_pt

    return A


# ─────────────────────── N-D Poisson FFT integration (default) ───────────────

def poisson_integrate(czar_grad, coords, periodic):
    """
    Recover A(z) from the gradient field via Poisson equation:
        laplacian(A) = divergence(czar_grad)

    Uses FFT (periodic) or even-extension / DCT-I trick (non-periodic).
    Runs in milliseconds for typical grids.
    """
    dim = len(coords)
    shape = tuple(len(c) for c in coords)
    dx = np.array([c[1] - c[0] for c in coords])

    def _idx(d, i):
        s = [slice(None)] * dim
        s[d] = i
        return tuple(s)

    # 1. Divergence of the gradient field
    div = np.zeros(shape)
    for d in range(dim):
        g = czar_grad[..., d]
        if periodic[d]:
            div += (np.roll(g, -1, axis=d) - np.roll(g, +1, axis=d)) / (2.0 * dx[d])
        else:
            N = shape[d]
            slc_c = [slice(None)] * dim; slc_c[d] = slice(1, -1)
            slc_p = [slice(None)] * dim; slc_p[d] = slice(2, None)
            slc_m = [slice(None)] * dim; slc_m[d] = slice(None, -2)
            div[tuple(slc_c)] += (g[tuple(slc_p)] - g[tuple(slc_m)]) / (2.0 * dx[d])
            if N >= 3:
                div[_idx(d, 0)]  += (-3*g[_idx(d, 0)] + 4*g[_idx(d, 1)] - g[_idx(d, 2)]) / (2*dx[d])
                div[_idx(d, -1)] += (3*g[_idx(d, -1)] - 4*g[_idx(d, -2)] + g[_idx(d, -3)]) / (2*dx[d])
            else:
                div[_idx(d, 0)]  += (g[_idx(d, 1)] - g[_idx(d, 0)]) / dx[d]
                div[_idx(d, -1)] += (g[_idx(d, -1)] - g[_idx(d, -2)]) / dx[d]

    # 2. Even-extension for non-periodic dimensions (DCT-I via FFT)
    ext_shape = tuple(
        shape[d] if periodic[d] else (2 * (shape[d] - 1) if shape[d] > 1 else 1)
        for d in range(dim)
    )
    div_ext = np.zeros(ext_shape)
    orig_slices = tuple(slice(0, shape[d]) for d in range(dim))
    div_ext[orig_slices] = div

    for d in range(dim):
        if not periodic[d] and shape[d] > 1:
            N = shape[d]
            n_reflect = ext_shape[d] - N
            if n_reflect > 0:
                src_idx = np.arange(N - 2, 0, -1)
                dst_idx = np.arange(N, ext_shape[d])
                src = np.take(div_ext, src_idx, axis=d)
                slc_dst = [slice(None)] * dim
                slc_dst[d] = dst_idx
                div_ext[tuple(slc_dst)] = src

    # 3. FFT solve: laplacian eigenvalues in Fourier space
    Fd = np.fft.fftn(div_ext)
    eigenvals = np.zeros(ext_shape)
    for d in range(dim):
        Nd = ext_shape[d]
        ks = np.arange(Nd)
        eig_d = (2.0 * np.cos(2.0 * np.pi * ks / Nd) - 2.0) / dx[d]**2
        bc_shape = [1] * dim
        bc_shape[d] = Nd
        eigenvals += eig_d.reshape(bc_shape)

    with np.errstate(divide='ignore', invalid='ignore'):
        Fd_sol = np.where(eigenvals != 0, Fd / eigenvals, 0.0)

    A_ext = np.fft.ifftn(Fd_sol).real

    # 4. Extract and shift
    A = A_ext[orig_slices].copy()
    A -= A.min()
    return A


# ─────────────────────── output writer ───────────────────────────────────────

def write_output(path, coords, periodic, ptilde, czar_grad, A, minpop, kT):
    """Write the FEL and gradient to a space-separated text file (vectorised)."""
    dim = len(coords)
    shape = tuple(len(c) for c in coords)

    ptilde_max = ptilde.max()
    pop_threshold = minpop * ptilde_max if ptilde_max > 0 else 0.0

    # Build header
    hdr_parts = ['#']
    for d in range(dim):
        hdr_parts.append(f'z{d}')
    for d in range(dim):
        hdr_parts.append(f'czar_grad{d}')
    hdr_parts += ['ptilde', 'A_czar[kJ/mol]']

    header_lines = [
        ' '.join(hdr_parts),
        f'# kT = {kT:.6f} kJ/mol',
        f'# Grid shape: {shape}',
        f'# minpop threshold: {pop_threshold:.4g} ({minpop} * max(ptilde))',
        f'# A_czar set to NaN where ptilde < threshold',
        '#',
    ]

    # Flatten all arrays and build output columns
    mg = np.meshgrid(*coords, indexing='ij')
    n_pts = int(np.prod(shape))
    cols = []
    for d in range(dim):
        cols.append(mg[d].ravel())
    for d in range(dim):
        cols.append(czar_grad[..., d].ravel())
    cols.append(ptilde.ravel())

    A_out = A.ravel().copy()
    A_out[ptilde.ravel() < pop_threshold] = np.nan
    cols.append(A_out)

    data = np.column_stack(cols)

    with open(path, 'w') as fh:
        for line in header_lines:
            fh.write(line + '\n')

        if dim == 1:
            # Simple: one block
            for row in data:
                vals = [f'{v:.8f}' if np.isfinite(v) else 'nan' for v in row]
                fh.write(' '.join(vals) + '\n')
        elif dim == 2:
            # Insert blank line between z0 slices for gnuplot pm3d
            n1 = shape[1]
            for i, row in enumerate(data):
                vals = [f'{v:.8f}' if np.isfinite(v) else 'nan' for v in row]
                fh.write(' '.join(vals) + '\n')
                if (i + 1) % n1 == 0:
                    fh.write('\n')
        else:
            # 3-D+: blank lines when the last two inner indices wrap
            inner_size = int(np.prod(shape[1:]))
            for i, row in enumerate(data):
                vals = [f'{v:.8f}' if np.isfinite(v) else 'nan' for v in row]
                fh.write(' '.join(vals) + '\n')
                if (i + 1) % inner_size == 0:
                    fh.write('\n')


# ─────────────────────── batch FEL processing ────────────────────────────────

def write_simple_fel(path, coords, A, minpop_frac, ptilde):
    """Write a simple z0 z1 ... A(z) file (OPES-style processed output).
    Blank lines between z0 slices for gnuplot pm3d compatibility."""
    dim = len(coords)
    shape = tuple(len(c) for c in coords)
    mg = np.meshgrid(*coords, indexing='ij')

    pop_thresh = minpop_frac * (ptilde.max() if ptilde.max() > 0 else 1.0)
    A_out = A.copy()
    A_out[ptilde < pop_thresh] = np.nan
    # Shift so minimum = 0
    finite = A_out[np.isfinite(A_out)]
    if len(finite) > 0:
        A_out -= finite.min()

    with open(path, 'w') as fh:
        hdr = ' '.join([f'z{d}' for d in range(dim)] + ['A(kJ/mol)'])
        fh.write(f'# {hdr}\n')

        if dim == 1:
            for g in range(shape[0]):
                a = A_out.ravel()[g]
                val = f'{a:.6f}' if np.isfinite(a) else 'nan'
                fh.write(f'{coords[0][g]:.8f} {val}\n')
        elif dim == 2:
            n1 = shape[1]
            for i in range(shape[0]):
                for j in range(shape[1]):
                    a = A_out[i, j]
                    val = f'{a:.6f}' if np.isfinite(a) else 'nan'
                    fh.write(f'{coords[0][i]:.8f} {coords[1][j]:.8f} {val}\n')
                fh.write('\n')
        else:
            flat = 0
            inner = int(np.prod(shape[1:]))
            idx = [0] * dim
            for g in range(int(np.prod(shape))):
                # compute multi-index
                tmp = g
                for d in range(dim-1, -1, -1):
                    idx[d] = tmp % shape[d]; tmp //= shape[d]
                parts = [f'{coords[d][idx[d]]:.8f}' for d in range(dim)]
                a = A_out.ravel()[g]
                parts.append(f'{a:.6f}' if np.isfinite(a) else 'nan')
                fh.write(' '.join(parts) + '\n')
                if (g + 1) % inner == 0:
                    fh.write('\n')


def process_all_snapshots(args, coords, periodic, meta):
    """Process every czar_kernels snapshot file and write a simple FEL .dat."""
    import glob, os, re

    czar_path = os.path.abspath(args.czar)
    czar_dir  = os.path.dirname(czar_path)
    czar_base = os.path.basename(czar_path)

    stem_match = re.match(r'^(.+?)(_\d{8})?(\.\w+)?$', czar_base)
    stem = stem_match.group(1) if stem_match else czar_base.split('.')[0]
    ext  = stem_match.group(3) if stem_match and stem_match.group(3) else '.dat'

    pattern = os.path.join(czar_dir, f'{stem}_????????{ext}')
    candidates = sorted(glob.glob(pattern))

    if not candidates:
        print(f"  No snapshot files found matching: {pattern}")
        return

    out_dir = args.fel_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = czar_dir

    print(f"  Processing {len(candidates)} snapshots → {out_dir}/", flush=True)

    for fpath in candidates:
        fname = os.path.basename(fpath)
        m = re.search(r'_(\d{8})' + re.escape(ext) + r'$', fname)
        if not m:
            continue
        step = int(m.group(1))

        try:
            meta_s, kernels_s = parse_czar_file(fpath)
        except SystemExit:
            print(f"  Skipping {fname}: parse error")
            continue
        if not kernels_s:
            continue

        A_s, mask_s, ptilde_s = compute_fel(meta_s, kernels_s, args, coords, periodic)

        out_name = f'FEL_{step:08d}.dat'
        out_path = os.path.join(out_dir, out_name)
        write_simple_fel(out_path, coords, A_s, args.minpop, ptilde_s)
        print(f"  step {step:8d} → {out_name}", flush=True)

    print(f"  Done. {len(candidates)} FEL files written.", flush=True)


# ─────────────────────── convergence time series ──────────────────────────────

def compute_fel(meta, kernels, args, coords, periodic):
    """Run CZAR evaluation + integration, return (A, mask, ptilde)."""
    kT = args.kT if args.kT is not None else meta['kT']
    ptilde, czar_grad = czar_on_grid(coords, periodic, meta, kernels, kT,
                                      nsigma=args.nsigma, verbose=False)

    allowed = ptilde > args.minpop * (ptilde.max() if ptilde.max() > 0 else 1.0)

    if len(coords) == 1:
        dx = coords[0][1] - coords[0][0]
        A = integrate_1d(czar_grad[..., 0], dx, periodic[0])
    elif getattr(args, 'mc', False):
        A = mc_integrate(czar_grad, coords, periodic, allowed, kT,
                         nsteps=args.mc_steps)
    else:
        A = poisson_integrate(czar_grad, coords, periodic)

    mask = allowed
    return A, mask, ptilde


def rmsd_convergence(args, coords, periodic, A_ref, mask_ref, meta_ref):
    """
    Scan the directory of args.czar for snapshot files matching
    czar_kernels_XXXXXXXX.dat (or the same stem with step suffix),
    compute the FEL for each, and return sorted lists (steps, rmsds).

    Following Pfaendtner & Bonomi (2015), the RMSD is restricted to
    grid points within conv_kt_window × kT of the reference minimum.
    """
    import glob, os, re

    kT = args.kT if args.kT is not None else meta_ref.get('kT', 2.494)
    kt_window = args.conv_kt_window * kT if args.conv_kt_window > 0 else 1e30

    czar_path = os.path.abspath(args.czar)
    czar_dir  = os.path.dirname(czar_path)
    czar_base = os.path.basename(czar_path)

    stem_match = re.match(r'^(.+?)(_\d{8})?(\.\w+)?$', czar_base)
    stem = stem_match.group(1) if stem_match else czar_base.split('.')[0]
    ext  = stem_match.group(3) if stem_match and stem_match.group(3) else '.dat'

    pattern = os.path.join(czar_dir, f'{stem}_????????{ext}')
    candidates = sorted(glob.glob(pattern))

    if not candidates:
        print(f"  No snapshot files found matching: {pattern}")
        return [], []

    print(f"  Found {len(candidates)} snapshot files matching '{stem}_XXXXXXXX{ext}'")

    # Build the reference mask: populated AND within kT window of minimum.
    A_ref_shifted = A_ref.copy()
    A_ref_shifted -= A_ref_shifted[mask_ref].min() if mask_ref.any() else 0
    mask_window = mask_ref & (A_ref_shifted <= kt_window)
    n_window = mask_window.sum()
    print(f"  Reference: {mask_ref.sum()} populated pts, "
          f"{n_window} within {args.conv_kt_window} kT ({kt_window:.1f} kJ/mol) of minimum")

    steps, rmsds = [], []
    for fpath in candidates:
        fname = os.path.basename(fpath)
        m = re.search(r'_(\d{8})' + re.escape(ext) + r'$', fname)
        if not m:
            continue
        step = int(m.group(1))

        try:
            meta_s, kernels_s = parse_czar_file(fpath)
        except SystemExit:
            print(f"  Skipping {fname}: parse error")
            continue

        if not kernels_s:
            print(f"  Skipping {fname}: no kernels")
            continue

        A_s, mask_s, _ = compute_fel(meta_s, kernels_s, args, coords, periodic)

        common = mask_window & mask_s
        if common.sum() < 4:
            print(f"  step {step:8d}: too few common grid points ({common.sum()}) — skipping")
            continue

        A_ref_al = A_ref.copy()
        A_s_al   = A_s.copy()
        A_ref_al -= A_ref_al[common].min()
        A_s_al   -= A_s_al[common].min()

        rmsd = float(np.sqrt(np.mean((A_ref_al[common] - A_s_al[common])**2)))
        steps.append(step)
        rmsds.append(rmsd)
        print(f"  step {step:8d}: {common.sum():5d} common pts  RMSD = {rmsd:.4f} kJ/mol")

    return steps, rmsds


# ─────────────────────── main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Recover FEL from FKERNELABF v2+ CZAR kernel file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--czar',    required=True,    help='CZAR kernel file (reference)')
    parser.add_argument('--grid',    type=int, default=100, help='Grid points per dim')
    parser.add_argument('--nsigma',  type=float, default=4.0,
                        help='Kernel cutoff in sigma units per dimension (default 4.0)')
    parser.add_argument('--output',  default='FEL_czar.dat', help='Output file')
    parser.add_argument('--min',     type=float, nargs='+', default=None,
                        help='Override grid min per dim')
    parser.add_argument('--max',     type=float, nargs='+', default=None,
                        help='Override grid max per dim')
    parser.add_argument('--kT',      type=float, default=None, help='Override kT (kJ/mol)')
    parser.add_argument('--minpop',  type=float, default=1e-3,
                        help='Allowed-region mask: grid points with ptilde < MINPOP*max(ptilde) '
                             'are excluded from display (and from MC integration if --mc). '
                             'Default 1e-3.')
    parser.add_argument('--mc', action='store_true',
                        help='Use MC integration (à la abf_integrate) instead of Poisson FFT. '
                             'Slower but handles non-conservative gradient fields. '
                             'Default: Poisson FFT (instant).')
    parser.add_argument('--mc_steps', type=int, default=0,
                        help='MC integration steps (0 = auto-converge). Only used with --mc.')
    parser.add_argument('--mc_hill', type=float, default=0.01,
                        help='Initial MC hill height (default 0.01). Only used with --mc.')
    parser.add_argument('--mc_hill_factor', type=float, default=0.5,
                        help='Hill reduction factor (default 0.5). Only used with --mc.')
    parser.add_argument('--convergence', action='store_true',
                        help='Compute RMSD time series vs reference (--czar) '
                             'using all czar_*_XXXXXXXX.dat snapshots in the same directory')
    parser.add_argument('--conv_output', default='czar_convergence.dat',
                        help='Output file for convergence RMSD time series')
    parser.add_argument('--conv_kt_window', type=float, default=20.0,
                        help='Only compare grid points within this many kT of the '
                             'reference minimum (default: 20, following Pfaendtner & '
                             'Bonomi 2015). Set 0 to disable.')
    parser.add_argument('--process_all', action='store_true',
                        help='Convert ALL czar_kernels snapshot files to simple '
                             'z0 z1 A(kJ/mol) grid files (OPES-style output). '
                             'Files written to --fel_dir as FEL_XXXXXXXX.dat.')
    parser.add_argument('--fel_dir', default=None,
                        help='Directory for processed FEL files (default: same as --czar)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    print(f"Reading reference CZAR kernels from: {args.czar}", flush=True)
    meta, kernels = parse_czar_file(args.czar)

    dim   = meta['dim']
    kT    = args.kT if args.kT is not None else meta['kT']
    nkern = len(kernels)

    print(f"  dim={dim}  kT={kT:.5f} kJ/mol  kernels={nkern}", flush=True)
    if 'periodic' in meta:
        print(f"  periodic: {meta['periodic']}", flush=True)
    if 'kappa' in meta:
        print(f"  kappa:    {meta['kappa']}", flush=True)

    coords, periodic, gmin, gmax = build_grid(meta, args)
    print(f"  Grid: {args.grid} pts/dim  "
          f"min={gmin.tolist()}  max={gmax.tolist()}", flush=True)

    # ── Reference FEL ────────────────────────────────────────────────────────
    print("Evaluating reference CZAR gradient on grid ...", flush=True)
    ptilde, czar_grad = czar_on_grid(coords, periodic, meta, kernels, kT,
                                      nsigma=args.nsigma, verbose=args.verbose)

    frac_populated = np.mean(ptilde > 0)
    print(f"  Fraction of grid populated: {frac_populated*100:.1f}%", flush=True)
    if frac_populated < 0.1:
        print("  WARNING: less than 10% of grid has kernel coverage. "
              "Consider reducing --grid or increasing SIGMA.", flush=True)

    print("Integrating reference FEL ...", flush=True)

    pop_thresh = args.minpop * (ptilde.max() if ptilde.max() > 0 else 1.0)
    allowed = ptilde >= pop_thresh

    if dim == 1:
        dx = coords[0][1] - coords[0][0]
        A_ref = integrate_1d(czar_grad[..., 0], dx, periodic[0])
    elif args.mc:
        print("  Using MC integration (abf_integrate style) ...", flush=True)
        A_ref = mc_integrate(czar_grad, coords, periodic, allowed, kT,
                             nsteps=args.mc_steps,
                             hill=args.mc_hill,
                             hill_factor=args.mc_hill_factor,
                             verbose=args.verbose)
    else:
        print("  Using Poisson FFT integration ...", flush=True)
        A_ref = poisson_integrate(czar_grad, coords, periodic)

    mask_ref = allowed

    if mask_ref.any():
        A_range = A_ref[mask_ref].max() - A_ref[mask_ref].min()
        print(f"  Reference FEL range: {A_range:.3f} kJ/mol ({A_range/kT:.2f} kT)", flush=True)
    else:
        print("  WARNING: no grid points above minpop threshold.", flush=True)

    print(f"Writing reference FEL to: {args.output}", flush=True)
    write_output(args.output, coords, periodic, ptilde, czar_grad, A_ref,
                 args.minpop, kT)

    # ── Convergence time series ───────────────────────────────────────────────
    if args.convergence:
        print(f"\nConvergence mode: scanning for snapshot files ...", flush=True)
        steps, rmsds = rmsd_convergence(args, coords, periodic, A_ref,
                                        mask_ref, meta)

        if steps:
            steps_a = np.array(steps, dtype=float)
            rmsds_a = np.array(rmsds, dtype=float)
            # RMSD² × t: if error ∝ 1/√t, this product is constant.
            # A flat plateau means the method has no systematic bias and
            # the error is purely statistical (PBMetaD, Pfaendtner & Bonomi 2015).
            rmsd2_t = rmsds_a**2 * steps_a

            with open(args.conv_output, 'w') as fh:
                fh.write(f'# CZAR FEL convergence vs reference: {args.czar}\n')
                fh.write(f'# Grid: {args.grid} pts/dim  minpop={args.minpop}\n')
                fh.write(f'# RMSD within {args.conv_kt_window} kT of reference minimum\n')
                fh.write(f'# FELs aligned to zero minimum over common region before RMSD\n')
                fh.write(f'# If error ~ 1/sqrt(t), RMSD^2*step should plateau to a constant.\n')
                fh.write(f'# (Pfaendtner & Bonomi, JCTC 2015)\n')
                fh.write('# step  RMSD_kJ/mol  RMSD^2*step\n')
                for s, r, r2t in zip(steps, rmsds, rmsd2_t):
                    fh.write(f'{s:10d}  {r:.6f}  {r2t:.4f}\n')
            print(f"\nConvergence table written to: {args.conv_output}", flush=True)

            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                fig.suptitle('CZAR FEL Convergence', fontsize=13)

                # Panel A: RMSD vs step (log-log to check 1/√t scaling)
                ax = axes[0]
                ax.loglog(steps_a, rmsds_a, 'o-', lw=1.5, ms=5, color='steelblue',
                          label='RMSD')
                ax.axhline(kT, color='orange', lw=1.2, ls='--',
                           label=f'kT = {kT:.3f} kJ/mol')
                # Reference 1/√t slope from first valid point
                if len(steps_a) > 2:
                    t0 = steps_a[0]; r0 = rmsds_a[0]
                    t_ref = np.linspace(t0, steps_a[-1], 200)
                    ax.plot(t_ref, r0 * np.sqrt(t0 / t_ref), ':', color='grey',
                            lw=1.2, label=r'$\propto 1/\sqrt{t}$')
                ax.set_xlabel('Step')
                ax.set_ylabel('RMSD vs reference (kJ/mol)')
                ax.set_title('RMSD vs step (log-log)')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3, which='both')

                # Panel B: RMSD² × step (should plateau if error ~ 1/√t)
                ax = axes[1]
                ax.plot(steps_a, rmsd2_t, 's-', lw=1.5, ms=5, color='crimson')
                # Show late-time mean as reference
                if len(rmsd2_t) > 4:
                    late = rmsd2_t[len(rmsd2_t)//2:]
                    late_mean = late.mean()
                    ax.axhline(late_mean, color='orange', lw=1.2, ls='--',
                               label=f'late mean = {late_mean:.1f}')
                ax.set_xlabel('Step')
                ax.set_ylabel(r'RMSD$^2 \times$ step  (kJ/mol)$^2 \cdot$ step')
                ax.set_title(r'RMSD$^2 \times t$ (plateau = purely statistical error)')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                conv_plot = args.conv_output.replace('.dat', '.png')
                plt.savefig(conv_plot, dpi=150)
                print(f"  Convergence plot saved: {conv_plot}", flush=True)
                plt.close()
            except ImportError:
                pass
        else:
            print("  No valid snapshots found — convergence table not written.", flush=True)

    # ── Batch process all snapshots to simple FEL files ──────────────────────
    if args.process_all:
        print(f"\nProcessing all snapshots to simple FEL files ...", flush=True)
        process_all_snapshots(args, coords, periodic, meta)

    print("Done.", flush=True)

    # ── Summary plots ─────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import CenteredNorm

        imgfile = args.output.replace('.dat', '_summary.png')

        if dim == 1:
            cv_names = meta.get('cv_names', None)
            z0_label = cv_names[0] if cv_names and len(cv_names) > 0 else 'z'

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            A_plot = np.where(mask_ref, A_ref, np.nan)
            axes[0].plot(coords[0], czar_grad[..., 0], color='steelblue')
            axes[0].set_xlabel(z0_label); axes[0].set_ylabel(f'dA/d{z0_label} (kJ/mol/rad)')
            axes[0].set_title('CZAR mean force'); axes[0].grid(True, alpha=0.3)
            axes[1].plot(coords[0], A_plot, color='darkorange')
            axes[1].set_xlabel(z0_label); axes[1].set_ylabel('A (kJ/mol)')
            axes[1].set_title('FEL via CZAR'); axes[1].grid(True, alpha=0.3)
            axes[2].plot(coords[0], np.log10(ptilde + 1e-300), color='seagreen')
            axes[2].set_xlabel(z0_label); axes[2].set_ylabel('log10(ptilde)')
            axes[2].set_title('Biased density'); axes[2].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(imgfile, dpi=150)
            print(f"  Summary plot saved: {imgfile}", flush=True)
            plt.close()

        elif dim == 2:
            c0, c1 = coords[0], coords[1]
            dx0 = c0[1] - c0[0]; dx1 = c1[1] - c1[0]
            A_plot = np.where(mask_ref, A_ref, np.nan)
            vmax_A = float(np.nanpercentile(A_plot[np.isfinite(A_plot)], 95))
            levels_A = np.linspace(0, vmax_A, 40)
            cmap_A = 'viridis'

            F0 = czar_grad[..., 0]
            F1 = czar_grad[..., 1]
            if periodic[0]:
                dF1_dz0 = (np.roll(F1, -1, axis=0) - np.roll(F1, +1, axis=0)) / (2*dx0)
                dF0_dz1 = (np.roll(F0, -1, axis=1) - np.roll(F0, +1, axis=1)) / (2*dx1)
            else:
                dF1_dz0 = np.gradient(F1, dx0, axis=0)
                dF0_dz1 = np.gradient(F0, dx1, axis=1)
            curl = dF1_dz0 - dF0_dz1
            curl_masked = np.where(mask_ref, curl, np.nan)
            curl_norm = curl_masked / kT
            curl_abs_95 = float(np.nanpercentile(
                np.abs(curl_norm[np.isfinite(curl_norm)]), 95))
            clim_curl = max(curl_abs_95, 0.01)

            curl_rms = (float(np.sqrt(np.nanmean(curl_masked[mask_ref]**2)))
                        if mask_ref.any() else np.nan)
            curl_rms_norm = curl_rms / kT
            print(f"  Curl RMS = {curl_rms:.4f} kJ/mol/rad^2  "
                  f"({curl_rms_norm:.4f} in kT/rad^2 units)", flush=True)
            if curl_rms_norm > 0.5:
                print(f"  WARNING: large curl residual suggests the force field "
                      f"is not yet integrable -- FEL may have path-dependent errors.",
                      flush=True)

            # Signed force component limits (symmetric around 0)
            F0_masked = np.where(mask_ref, F0, np.nan)
            F1_masked = np.where(mask_ref, F1, np.nan)
            f0_lim = float(np.nanpercentile(np.abs(F0_masked[np.isfinite(F0_masked)]), 95))
            f1_lim = float(np.nanpercentile(np.abs(F1_masked[np.isfinite(F1_masked)]), 95))
            f0_lim = max(f0_lim, 0.01)
            f1_lim = max(f1_lim, 0.01)

            # Get CV names from metadata if available
            cv_names = meta.get('cv_names', None)
            z0_label = cv_names[0] if cv_names and len(cv_names) > 0 else 'z0'
            z1_label = cv_names[1] if cv_names and len(cv_names) > 1 else 'z1'

            fig, axes = plt.subplots(3, 2, figsize=(13, 16))
            fig.suptitle('CZAR FEL Analysis', fontsize=14)

            # [0,0] Free energy surface
            ax = axes[0, 0]
            cs = ax.contourf(c1, c0, A_plot.T, levels=np.linspace(0,90,45), cmap=cmap_A,
                             extend='max')
            ax.contour(c1, c0, A_plot.T, levels=np.linspace(0,90,45),
                       colors='k', linewidths=0.5, alpha=0.6)
            plt.colorbar(cs, ax=ax, label='A (kJ/mol)')
            ax.set_xlabel(z1_label); ax.set_ylabel(z0_label)
            ax.set_title('Free Energy Surface')

            # [0,1] Biased density
            ax = axes[0, 1]
            ld = np.log10(ptilde + 1e-300)
            ld_masked = np.where(mask_ref, ld, np.nan)
            cs2 = ax.contourf(c1, c0, ld_masked.T, levels=20, cmap='plasma')
            plt.colorbar(cs2, ax=ax, label='log10(ptilde)')
            ax.set_xlabel(z1_label); ax.set_ylabel(z0_label)
            ax.set_title('Biased sampling density log10(ptilde)')

            # [1,0] Signed mean force component 0: dA/dz0
            ax = axes[1, 0]
            cs3 = ax.contourf(c1, c0, F0_masked.T,
                              levels=np.linspace(-f0_lim, f0_lim, 40),
                              cmap='RdBu_r', extend='both')
            ax.contour(c1, c0, A_plot.T, levels=levels_A,
                       colors='k', linewidths=0.4, alpha=0.4)
            plt.colorbar(cs3, ax=ax, label=f'dA/d{z0_label} (kJ/mol/rad)')
            ax.set_xlabel(z1_label); ax.set_ylabel(z0_label)
            ax.set_title(f'CZAR mean force dA/d{z0_label}')

            # [1,1] Signed mean force component 1: dA/dz1
            ax = axes[1, 1]
            cs4 = ax.contourf(c1, c0, F1_masked.T,
                              levels=np.linspace(-f1_lim, f1_lim, 40),
                              cmap='RdBu_r', extend='both')
            ax.contour(c1, c0, A_plot.T, levels=levels_A,
                       colors='k', linewidths=0.4, alpha=0.4)
            plt.colorbar(cs4, ax=ax, label=f'dA/d{z1_label} (kJ/mol/rad)')
            ax.set_xlabel(z1_label); ax.set_ylabel(z0_label)
            ax.set_title(f'CZAR mean force dA/d{z1_label}')

            # [2,0] Curl residual
            ax = axes[2, 0]
            cs5 = ax.contourf(c1, c0, curl_norm.T,
                              levels=np.linspace(-clim_curl, clim_curl, 40),
                              cmap='bwr', extend='both')
            ax.contour(c1, c0, A_plot.T, levels=levels_A,
                       colors='k', linewidths=0.4, alpha=0.4)
            plt.colorbar(cs5, ax=ax,
                         label='(dF1/dz0 - dF0/dz1) / kT')
            ax.set_xlabel(z1_label); ax.set_ylabel(z0_label)
            ax.set_title(f'Curl residual / kT  (RMS={curl_rms_norm:.3f})')

            # [2,1] Force magnitude (for reference)
            ax = axes[2, 1]
            grad_mag = np.sqrt(F0**2 + F1**2)
            grad_mag_masked = np.where(mask_ref, grad_mag, np.nan)
            gm_max = float(np.nanpercentile(
                grad_mag_masked[np.isfinite(grad_mag_masked)], 95))
            cs6 = ax.contourf(c1, c0, grad_mag_masked.T,
                              levels=np.linspace(0, gm_max, 40),
                              cmap='hot_r', extend='max')
            ax.contour(c1, c0, A_plot.T, levels=levels_A,
                       colors='white', linewidths=0.4, alpha=0.5)
            plt.colorbar(cs6, ax=ax, label='|grad A| (kJ/mol/rad)')
            ax.set_xlabel(z1_label); ax.set_ylabel(z0_label)
            ax.set_title('Mean force magnitude |dA/dz|')

            # Sampling boundary overlay on all panels
            for axi in axes.flat:
                axi.contour(c1, c0, mask_ref.astype(float).T,
                            levels=[0.5], colors=['grey'], linewidths=0.8,
                            linestyles=['--'])

            plt.tight_layout()
            plt.savefig(imgfile, dpi=150)
            print(f"  Summary plot saved: {imgfile}", flush=True)
            plt.close()

        # ── Kernel representation plots ───────────────────────────────────────
        # Analogous to the kernel panels in fkabf_analysis.py, but for
        # z-kernels (CZAR) rather than λ-kernels.
        if kernels and dim <= 3:
            n_kern = len(kernels)
            Nks     = np.array([k['Nk']     for k in kernels])
            centers = np.array([k['center']  for k in kernels])  # (n_kern, dim)
            mus     = np.array([k['mu']      for k in kernels])  # (n_kern, dim)
            sigmas  = np.array([k['sigma']   for k in kernels])  # (n_kern, dim)
            mu_mags = np.sqrt(np.sum(mus**2, axis=1))

            # Marker size proportional to σ_geo²
            sigma_geo = np.exp(np.mean(np.log(sigmas + 1e-300), axis=1))
            sigma_norm = sigma_geo / (np.median(sigma_geo) + 1e-300)
            sigma_norm = np.clip(sigma_norm, 0.2, 5.0)
            marker_s = (sigma_norm ** 2) * 40

            kern_imgfile = args.output.replace('.dat', '_kernels.png')

            if dim == 2:
                # ── Page: z-kernel positions coloured by |μ| and Nk ──────
                fig, axes_k = plt.subplots(1, 2, figsize=(13, 5))
                fig.suptitle(
                    f'z-Kernel Positions ({n_kern} kernels)  '
                    f'— marker area ∝ σ_geo²', fontsize=13)

                ax = axes_k[0]
                sc = ax.scatter(centers[:,0], centers[:,1], c=mu_mags,
                                cmap='hot_r', s=marker_s, alpha=0.6,
                                vmax=np.percentile(mu_mags, 95),
                                linewidths=0)
                plt.colorbar(sc, ax=ax, label='|μ| (kJ/mol/rad)')
                ax.set_xlabel(z0_label); ax.set_ylabel(z1_label)
                ax.set_title('Coloured by mean force |μ|')

                ax = axes_k[1]
                sc = ax.scatter(centers[:,0], centers[:,1], c=np.log10(Nks+1),
                                cmap='viridis', s=marker_s, alpha=0.6,
                                linewidths=0)
                plt.colorbar(sc, ax=ax, label='log₁₀(Nk+1)')
                ax.set_xlabel(z0_label); ax.set_ylabel(z1_label)
                ax.set_title('Coloured by log₁₀(Nk)')

                plt.tight_layout()
                plt.savefig(kern_imgfile, dpi=150)
                print(f"  Kernel scatter saved: {kern_imgfile}", flush=True)
                plt.close()

            elif dim == 1:
                fig, axes_k = plt.subplots(2, 1, figsize=(12, 6))
                fig.suptitle(
                    f'z-Kernel Positions ({n_kern} kernels)  '
                    f'— marker area ∝ σ²', fontsize=13)
                ax = axes_k[0]
                ax.scatter(centers[:,0], mu_mags, c=mu_mags,
                           cmap='hot_r', s=marker_s, alpha=0.6, linewidths=0)
                ax.set_xlabel(z0_label); ax.set_ylabel('|μ| (kJ/mol/rad)')
                ax.set_title('Kernel position vs |μ|')
                ax = axes_k[1]
                ax.scatter(centers[:,0], Nks, c=np.log10(Nks+1),
                           cmap='viridis', s=marker_s, alpha=0.6, linewidths=0)
                ax.set_xlabel(z0_label); ax.set_ylabel('Nk')
                ax.set_title('Kernel position vs Nk')

                plt.tight_layout()
                plt.savefig(kern_imgfile, dpi=150)
                print(f"  Kernel scatter saved: {kern_imgfile}", flush=True)
                plt.close()

            # ── Kernel statistics page ────────────────────────────────────
            stat_imgfile = args.output.replace('.dat', '_kernel_stats.png')
            fig, axes_s = plt.subplots(2, 2, figsize=(13, 8))
            fig.suptitle(f'z-Kernel Statistics ({n_kern} kernels)', fontsize=13)

            ax = axes_s[0, 0]
            ax.hist(Nks, bins=50, color='steelblue', edgecolor='none', log=True)
            ax.axvline(np.median(Nks), color='r', lw=1.5, ls='--',
                       label=f'median={np.median(Nks):.1f}')
            ax.axvline(np.mean(Nks), color='orange', lw=1.5, ls=':',
                       label=f'mean={np.mean(Nks):.1f}')
            ax.set_xlabel('Nk (samples per kernel)')
            ax.set_ylabel('Count (log)')
            ax.set_title('Nk distribution')
            ax.legend(fontsize=8)

            ax = axes_s[0, 1]
            ax.hist(mu_mags, bins=50, color='tomato', edgecolor='none')
            ax.axvline(np.median(mu_mags), color='k', lw=1.5, ls='--',
                       label=f'median={np.median(mu_mags):.1f}')
            ax.set_xlabel('|μ| (kJ/mol/rad)')
            ax.set_ylabel('Count')
            ax.set_title('Mean force magnitude distribution')
            ax.legend(fontsize=8)

            # Per-kernel sigma distribution
            ax = axes_s[1, 0]
            colors_sig = ['seagreen', 'darkorange', 'royalblue']
            cv_names_plot = meta.get('cv_names', None)
            for d in range(dim):
                lbl = cv_names_plot[d] if cv_names_plot and d < len(cv_names_plot) else f'dim{d}'
                c = colors_sig[d % len(colors_sig)]
                ax.hist(sigmas[:, d], bins=50, alpha=0.6,
                        edgecolor='none', color=c, label=f'σ({lbl})')
                ax.axvline(np.median(sigmas[:, d]), color=c, lw=1.5, ls='--')
            ax.set_xlabel('Per-kernel σ')
            ax.set_ylabel('Count')
            ax.set_title('Kernel bandwidth distribution')
            ax.legend(fontsize=8)

            # Nk vs |mu| correlation
            ax = axes_s[1, 1]
            ax.scatter(Nks, mu_mags, alpha=0.3, s=8, c='purple')
            ax.set_xlabel('Nk')
            ax.set_ylabel('|μ| (kJ/mol/rad)')
            ax.set_title('Nk vs |μ|: does confidence correlate with force?')
            ax.set_xscale('log')

            plt.tight_layout()
            plt.savefig(stat_imgfile, dpi=150)
            print(f"  Kernel stats saved: {stat_imgfile}", flush=True)
            plt.close()

    except ImportError:
        pass


if __name__ == '__main__':
    main()
