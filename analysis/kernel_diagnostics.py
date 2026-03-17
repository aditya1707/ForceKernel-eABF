#!/usr/bin/env python3
"""
FK-ABF kernel diagnostic plotter.

Reads a kernel file and produces a 4-panel diagnostic figure:
  (a) Kernel density Z(·) — NW denominator on a grid
  (b) Exploration potential V_ex(·) = c·ln(1 + Z/Z₀)   [γ > 1 only]
  (c) Kernel ellipses / scatter colored by sample count Nk
  (d) Kernel ellipses / scatter colored by |μ| (mean force magnitude)

Works for 1D (line plots) and 2D (contour + ellipse plots).

Supports both FKERNELABF output file types (auto-detected from header):

  ① Lambda-kernel dump  ({label}.kernels_{step:08d}.dat)   [KERNELSTRIDE]
       Kernels positioned in fictitious variable (λ) space.
       Domain metadata NOT embedded: supply --gridmin/--gridmax, or bounds
       will be estimated automatically from kernel extents.

  ② CZAR z-kernel file  ({label}.czar_kernels_{step:08d}.dat)  [CZARSTRIDE]
       Kernels positioned in real CV (z) space.
       Self-contained: dim, kT, kappa, domMin, domMax, periodic in header.

Usage:
    # Lambda-kernel dump (Müller-Brown, natural units):
    python kernel_diagnostics.py fk.kernels_05000000.dat \\
        --gridmin -1.5 -0.5 --gridmax 1.2 2.0 --kT 1.0 --output diag.pdf

    # CZAR z-kernel file (domain embedded in file):
    python kernel_diagnostics.py fk.czar_kernels_05000000.dat --output czar.pdf

    # Dihedral CVs (periodic, override γ for V_ex):
    python kernel_diagnostics.py fk.kernels_05000000.dat \\
        --gridmin -3.14159 -3.14159 --gridmax 3.14159 3.14159 \\
        --periodic 1 1 --gamma 10 --kT 2.479

    # Programmatic use:
    from kernel_diagnostics import plot_kernel_diagnostics
    plot_kernel_diagnostics('fk.kernels_05000000.dat',
                            domMin=[-1.5, -0.5], domMax=[1.2, 2.0])
"""

import re
import numpy as np
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
from matplotlib.ticker import AutoMinorLocator

# ─── House style ──────────────────────────────────────────────────────────
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.it'] = 'cmmi10'
plt.rcParams['font.family'] = 'DejaVu Sans'

TICK_PARAMS = dict(axis='both', which='both', direction='in',
                   top=True, right=True, length=5, width=1, labelsize=12)
TICK_PARAMS_MINOR = dict(axis='both', which='minor', direction='in',
                         top=True, right=True, length=3, width=0.7)


def panel_label(ax, label, x=-0.12, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=22, fontweight='bold', va='top', ha='left')


# ═══════════════════════════════════════════════════════════════════════════
# FILE-TYPE DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def _detect_file_type(filepath):
    """Return 'czar' or 'lambda_dump' based on the comment header."""
    with open(filepath) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if 'CZAR z-kernel file' in s:
                return 'czar'
            if 'Lambda-kernel snapshot' in s:
                return 'lambda_dump'
            # Fallback: key-value metadata lines are unique to CZAR format
            if s.startswith('#'):
                continue
            if s.split()[0] in ('dim', 'kT', 'kappa', 'periodic', 'domMin', 'domMax'):
                return 'czar'
            break   # non-comment, non-metadata line reached → probably lambda dump
    return 'lambda_dump'


# ═══════════════════════════════════════════════════════════════════════════
# CZAR FILE PARSER  ({label}.czar_kernels_{step:08d}.dat)
# ═══════════════════════════════════════════════════════════════════════════

def _parse_czar_file(filepath):
    """Parse a CZAR z-kernel file produced by FKERNELABF::writeCZARFile().

    File format
    -----------
    Comment header: key-value metadata lines (dim, kT, kappa, periodic,
                    domMin, domMax, nkernels).
    Data rows:      Nk  center[0..d-1]  mu[0..d-1]  sigma[0..d-1]
                    (1 + 3*dim floats per row)

    Returns  meta, kernels  (meta['kernel_type'] == 'z')
    """
    meta = {'biasFactor': 1.0}
    kernels = []
    dim = None

    with open(filepath) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            tokens = s.split()
            key = tokens[0]

            if key == 'dim':
                dim = int(tokens[1]); meta['dim'] = dim
            elif key == 'kT':
                meta['kT'] = float(tokens[1])
            elif key == 'kappa':
                meta['kappa'] = [float(t) for t in tokens[1:]]
            elif key == 'periodic':
                meta['periodic'] = [int(t) != 0 for t in tokens[1:]]
            elif key == 'domMin':
                meta['domMin'] = [float(t) for t in tokens[1:]]
            elif key == 'domMax':
                meta['domMax'] = [float(t) for t in tokens[1:]]
            elif key == 'nkernels':
                pass  # informational only
            else:
                # Data line: Nk center[d]... mu[d]... sigma[d]...
                try:
                    vals = [float(t) for t in tokens]
                except ValueError:
                    continue
                if dim is not None and len(vals) == 1 + 3 * dim:
                    kernels.append({
                        'Nk':     vals[0],
                        'center': vals[1:1+dim],
                        'mu':     vals[1+dim:1+2*dim],
                        'sigma':  vals[1+2*dim:1+3*dim],
                    })

    if dim is None:
        raise ValueError(f"No 'dim' line found in CZAR file: {filepath}")

    meta.setdefault('kT',       1.0)
    meta.setdefault('kappa',    [1.0] * dim)
    meta.setdefault('periodic', [False] * dim)
    meta['kernel_type'] = 'z'
    return meta, kernels


# ═══════════════════════════════════════════════════════════════════════════
# LAMBDA-KERNEL DUMP PARSER  ({label}.kernels_{step:08d}.dat)
# ═══════════════════════════════════════════════════════════════════════════

def _parse_lambda_dump(filepath, domMin=None, domMax=None,
                       periodic=None, kT=1.0):
    """Parse a lambda-kernel dump produced by FKERNELABF::dumpKernelsIfNeeded().

    File format
    -----------
    Comment header with snapshot metadata:
        # Lambda-kernel snapshot  step=50000000    M=456     totalN=12345678
        #   neff(Silverman)=1234.5  sigma=(0.1234,0.5678)
    Column-header comment:
        # k  Nk  c_cv0 [c_cv1 ...]  mu_cv0 ...  sig_cv0 ...  |mu|  wt
    Data rows (one per kernel):
        k(int)  Nk  center[0..d-1]  mu[0..d-1]  sigma[0..d-1]  |mu|  wt
        (4 + 3*dim floats per row)

    Domain metadata (domMin/domMax/periodic/kT) is NOT embedded in the file.
    Supply these via keyword arguments or they will be estimated from the
    kernel extents.

    Returns  meta, kernels  (meta['kernel_type'] == 'lambda')
    """
    meta = {'biasFactor': 1.0, 'kT': kT}
    kernels = []
    dim = None
    cv_names = []

    with open(filepath) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue

            if s.startswith('#'):
                # ── Snapshot summary comment ────────────────────────────
                # "# Lambda-kernel snapshot  step=50000000    M=456     totalN=12345678"
                if 'Lambda-kernel snapshot' in s:
                    for part in s.split():
                        if part.startswith('step='):
                            try:    meta['step'] = int(part.split('=')[1])
                            except: pass
                        elif part.startswith('M='):
                            try:    meta['M_snapshot'] = int(part.split('=')[1])
                            except: pass
                        elif part.startswith('totalN='):
                            try:    meta['totalN'] = float(part.split('=')[1])
                            except: pass

                # "#   neff(Silverman)=1234.5  sigma=(0.1234,0.5678)"
                elif 'neff(Silverman)' in s:
                    m = re.search(r'neff\(Silverman\)=([\d.]+)', s)
                    if m:
                        meta['neff_snapshot'] = float(m.group(1))
                    m = re.search(r'sigma=\(([^)]+)\)', s)
                    if m:
                        meta['sigma_snapshot'] = [float(x) for x in m.group(1).split(',')]

                # Column header:
                # "# k  Nk  c_phi  c_psi  mu_phi  mu_psi  sig_phi  sig_psi  |mu|  wt"
                elif 'Nk' in s and ('c_' in s or '|mu|' in s):
                    tokens = s.lstrip('#').split()
                    c_cols = [t for t in tokens if t.startswith('c_')]
                    if c_cols:
                        dim = len(c_cols)
                        meta['dim'] = dim
                        cv_names = [t[2:] for t in c_cols]  # strip leading 'c_'
                        meta['cv_names'] = cv_names
                continue

            # ── Data row ────────────────────────────────────────────────
            if dim is None:
                continue
            try:
                vals = [float(t) for t in s.split()]
            except ValueError:
                continue

            # Layout: k  Nk  center[dim]  mu[dim]  sigma[dim]  |mu|  wt
            # Total: 2 + 3*dim + 2 = 4 + 3*dim tokens
            if len(vals) == 4 + 3 * dim:
                kernels.append({
                    'Nk':     vals[1],
                    'center': vals[2:2+dim],
                    'mu':     vals[2+dim:2+2*dim],
                    'sigma':  vals[2+2*dim:2+3*dim],
                })

    if dim is None:
        raise ValueError(
            f"Could not determine dimensionality from: {filepath}\n"
            "Ensure the file is an FK-ABF lambda-kernel dump (KERNELSTRIDE output)."
        )

    # ── Periodicity ──────────────────────────────────────────────────────
    if periodic is not None:
        if isinstance(periodic, (bool, int)):
            meta['periodic'] = [bool(periodic)] * dim
        else:
            meta['periodic'] = [bool(p) for p in periodic]
    else:
        meta['periodic'] = [False] * dim

    meta['kT'] = kT

    # ── Domain bounds ─────────────────────────────────────────────────────
    if domMin is not None and domMax is not None:
        meta['domMin'] = list(domMin)
        meta['domMax'] = list(domMax)
    elif kernels:
        centers = np.array([k['center'] for k in kernels])
        sigmas  = np.array([k['sigma']  for k in kernels])
        margin  = 2.0
        meta['domMin'] = (centers.min(axis=0) - margin * sigmas.max(axis=0)).tolist()
        meta['domMax'] = (centers.max(axis=0) + margin * sigmas.max(axis=0)).tolist()
        print("  [kernel_diagnostics] Domain estimated from kernel extents "
              "(pass --gridmin/--gridmax for exact bounds):")
        print(f"    domMin = {[f'{v:.4f}' for v in meta['domMin']]}")
        print(f"    domMax = {[f'{v:.4f}' for v in meta['domMax']]}")
    else:
        meta['domMin'] = [0.0] * dim
        meta['domMax'] = [1.0] * dim

    meta['kernel_type'] = 'lambda'
    return meta, kernels


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED PARSER
# ═══════════════════════════════════════════════════════════════════════════

def parse_kernel_file(filepath, domMin=None, domMax=None,
                      periodic=None, kT=1.0):
    """Auto-detect and parse an FK-ABF kernel file.

    Handles both:
      ① CZAR z-kernel files    ({label}.czar_kernels_{step:08d}.dat)
      ② Lambda-kernel dumps    ({label}.kernels_{step:08d}.dat)

    Parameters
    ----------
    filepath : str or Path
    domMin, domMax : list of float or None
        Domain bounds — needed for lambda dump files when domain is not
        obvious. Ignored for CZAR files (domain is in the header).
    periodic : list of bool/int or None
        Periodicity flags per dimension. Ignored for CZAR files.
    kT : float
        Thermal energy — used for V_ex computation in lambda dump files.
        Ignored for CZAR files (kT is in the header).

    Returns
    -------
    meta : dict
        Keys always present: 'dim', 'kT', 'periodic', 'domMin', 'domMax',
        'biasFactor', 'kernel_type' ('z' or 'lambda').
        CZAR files additionally carry: 'kappa'.
        Lambda dumps additionally carry (when parseable from comments):
        'step', 'M_snapshot', 'totalN', 'neff_snapshot',
        'sigma_snapshot', 'cv_names'.
    kernels : list of dicts
        Each: {'Nk': float, 'center': list, 'mu': list, 'sigma': list}
    """
    filepath = str(filepath)
    ftype = _detect_file_type(filepath)

    if ftype == 'czar':
        return _parse_czar_file(filepath)
    else:
        return _parse_lambda_dump(filepath,
                                  domMin=domMin, domMax=domMax,
                                  periodic=periodic, kT=kT)


# ═══════════════════════════════════════════════════════════════════════════
# GRID EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_on_grid(meta, kernels, grid_pts=200, nsigma=4.0):
    """Evaluate Z(·) and ĝ(·) on a regular grid using per-kernel sigma.

    Matches the Gaussian weighting in FKERNELABF::splatKernel() /
    writeGridFile().

    Returns
    -------
    grids : list of 1-D arrays (grid coordinates per dimension)
    Z     : ndarray of shape grid_shape
    ghat  : ndarray of shape grid_shape + (dim,)
    """
    dim      = meta['dim']
    domMin   = meta['domMin']
    domMax   = meta['domMax']
    periodic = meta['periodic']

    axes = []
    for d in range(dim):
        if periodic[d]:
            dx = (domMax[d] - domMin[d]) / grid_pts
            axes.append(np.linspace(domMin[d], domMax[d] - dx, grid_pts))
        else:
            axes.append(np.linspace(domMin[d], domMax[d], grid_pts))

    shape = tuple(len(a) for a in axes)
    Z    = np.zeros(shape)
    numF = np.zeros(shape + (dim,))

    for kern in kernels:
        c   = kern['center']
        mu  = kern['mu']
        sig = kern['sigma']
        Nk  = kern['Nk']

        w1d   = []
        idx1d = []
        for d in range(dim):
            inv4s2 = 1.0 / (4.0 * sig[d]**2 + 1e-300)
            dx_d   = axes[d][1] - axes[d][0] if len(axes[d]) > 1 else 1.0
            R      = int(np.ceil(nsigma * sig[d] / max(dx_d, 1e-12)))
            ic     = int(np.round((c[d] - domMin[d]) / dx_d))
            N_d    = len(axes[d])

            ws   = []
            idxs = []
            for r in range(-R, R + 1):
                raw = ic + r
                if periodic[d]:
                    gi = raw % N_d
                else:
                    if raw < 0 or raw >= N_d:
                        continue
                    gi = raw

                delta = axes[d][gi] - c[d]
                if periodic[d]:
                    L = domMax[d] - domMin[d]
                    delta -= L * np.round(delta / L)

                w = np.exp(-delta**2 * inv4s2)
                if w < 1e-300:
                    continue
                ws.append(w)
                idxs.append(gi)

            w1d.append(np.array(ws))
            idx1d.append(np.array(idxs, dtype=int))

        if dim == 1:
            for a, wa in enumerate(w1d[0]):
                gi  = idx1d[0][a]
                wNk = wa * Nk
                Z[gi]       += wNk
                numF[gi, 0] += wNk * mu[0]
        elif dim == 2:
            for a, wa in enumerate(w1d[0]):
                for b, wb in enumerate(w1d[1]):
                    gi  = (idx1d[0][a], idx1d[1][b])
                    wNk = wa * wb * Nk
                    Z[gi]           += wNk
                    numF[gi + (0,)] += wNk * mu[0]
                    numF[gi + (1,)] += wNk * mu[1]
        else:
            for a, wa in enumerate(w1d[0]):
                for b, wb in enumerate(w1d[1]):
                    for cc, wc in enumerate(w1d[2]):
                        gi  = (idx1d[0][a], idx1d[1][b], idx1d[2][cc])
                        wNk = wa * wb * wc * Nk
                        Z[gi] += wNk
                        for d in range(dim):
                            numF[gi + (d,)] += wNk * mu[d]

    ghat    = np.zeros_like(numF)
    nonzero = Z > 1e-300
    with np.errstate(divide='ignore', invalid='ignore'):
        for d in range(dim):
            ghat[..., d] = np.where(nonzero, numF[..., d] / Z, 0.0)

    return axes, Z, ghat


def compute_Vex(Z, kT, gamma):
    """Compute V_ex = c·ln(1 + Z/Z₀), c = kT(γ−1), Z₀ = median(Z > 0).

    Matches FKERNELABF::finalizeExplorationGrid(). Returns zeros when γ ≤ 1.
    """
    if gamma <= 1.0:
        return np.zeros_like(Z)
    c    = kT * (gamma - 1.0)
    Zpop = Z[Z > 1e-10]
    Z0   = float(np.median(Zpop)) if len(Zpop) > 0 else 1.0
    if Z0 < 1e-10:
        Z0 = 1.0
    return c * np.log(1.0 + Z / (Z0 + 1e-300))


# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _make_ellipses(kernels, scale=2.0):
    """Ellipse patches from 2-D kernel centers/sigmas (scale × sigma radius)."""
    return [Ellipse(xy=(k['center'][0], k['center'][1]),
                    width=scale * 2 * k['sigma'][0],
                    height=scale * 2 * k['sigma'][1],
                    angle=0)
            for k in kernels]


def _style_2d_ax(ax, meta, xlabel=None, ylabel=None):
    ax.set_aspect('equal')
    ax.set_xlim(meta['domMin'][0], meta['domMax'][0])
    ax.set_ylim(meta['domMin'][1], meta['domMax'][1])
    ax.tick_params(**TICK_PARAMS)
    ax.tick_params(**TICK_PARAMS_MINOR)
    ax.tick_params(axis='x', pad=5)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14, labelpad=2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14, labelpad=2)


def _axis_labels(meta):
    """Return (xlabel, ylabel) using CV names when available."""
    cv_names = meta.get('cv_names', [])
    space    = r'$\lambda$' if meta.get('kernel_type') == 'lambda' else r'$z$'
    xl = fr'{space} ({cv_names[0]})' if len(cv_names) >= 1 else fr'{space} (CV 1)'
    yl = fr'{space} ({cv_names[1]})' if len(cv_names) >= 2 else fr'{space} (CV 2)'
    return xl, yl


def _kernel_label(meta):
    return 'λ-kernels' if meta.get('kernel_type') == 'lambda' else 'z-kernels (CZAR)'


# ═══════════════════════════════════════════════════════════════════════════
# 2-D DIAGNOSTIC FIGURE
# ═══════════════════════════════════════════════════════════════════════════

def plot_kernel_diagnostics_2d(meta, kernels, axes_grids, Z, ghat,
                               output=None, title_prefix=''):
    """2×2 diagnostic figure for 2-D kernel populations."""
    kT    = meta['kT']
    gamma = meta['biasFactor']
    Vex   = compute_Vex(Z, kT, gamma)

    Nks     = np.array([k['Nk'] for k in kernels])
    mu_mags = np.array([np.sqrt(sum(m**2 for m in k['mu'])) for k in kernels])

    xu, yu        = axes_grids[0], axes_grids[1]
    xlabel, ylabel = _axis_labels(meta)
    klabel        = _kernel_label(meta)
    space         = r'$\lambda$' if meta.get('kernel_type') == 'lambda' else r'$z$'

    fig, axs = plt.subplots(2, 2, figsize=(12, 10.5))

    # ── (a) Density Z ─────────────────────────────────────────────────
    ax    = axs[0, 0]
    Zplot = np.where(Z > 0, np.log10(Z + 1), 0)
    cf    = ax.contourf(xu, yu, Zplot.T, levels=30, cmap='viridis')
    cf.set_edgecolor('face')
    cb = fig.colorbar(cf, ax=ax, shrink=0.85)
    cb.set_label(r'$\log_{10}(Z + 1)$', fontsize=12)
    cb.ax.tick_params(labelsize=10)
    _style_2d_ax(ax, meta, xlabel=xlabel, ylabel=ylabel)
    ax.set_title(f'{title_prefix}Kernel density Z({space})', fontsize=13)
    panel_label(ax, 'a')

    # ── (b) Exploration potential V_ex ────────────────────────────────
    ax = axs[0, 1]
    if gamma > 1.0:
        cf = ax.contourf(xu, yu, Vex.T, levels=30, cmap='inferno')
        cf.set_edgecolor('face')
        cb = fig.colorbar(cf, ax=ax, shrink=0.85)
        cb.set_label(r'$V_{\mathrm{ex}}$ (kJ/mol)', fontsize=12)
        cb.ax.tick_params(labelsize=10)
    else:
        ax.text(0.5, 0.5, r'$\gamma = 1$: no exploration',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=13, color='grey')
    _style_2d_ax(ax, meta, xlabel=xlabel)
    ax.set_yticklabels([])
    ax.set_title(f'{title_prefix}Exploration potential', fontsize=13)
    panel_label(ax, 'b')

    # ── (c) Kernels colored by Nk ─────────────────────────────────────
    ax      = axs[1, 0]
    patches = _make_ellipses(kernels, scale=2.0)
    if patches:
        pc = PatchCollection(patches, alpha=0.6, edgecolors='none')
        pc.set_array(np.log10(Nks + 1))
        pc.set_cmap('turbo')
        ax.add_collection(pc)
        cb = fig.colorbar(pc, ax=ax, shrink=0.85)
        cb.set_label(r'$\log_{10}(N_k + 1)$', fontsize=12)
        cb.ax.tick_params(labelsize=10)
    _style_2d_ax(ax, meta, xlabel=xlabel, ylabel=ylabel)
    ax.set_title(f'{title_prefix}{klabel} by sample count ($M = {len(kernels)}$)',
                 fontsize=13)
    panel_label(ax, 'c')

    # ── (d) Kernels colored by |μ| ───────────────────────────────────
    # Note: clim is set once to [0, 95th-percentile]; magnitude is non-negative.
    ax      = axs[1, 1]
    patches = _make_ellipses(kernels, scale=2.0)
    if patches:
        vmax = float(np.percentile(mu_mags, 95)) if len(mu_mags) > 0 else 1.0
        pc   = PatchCollection(patches, alpha=0.6, edgecolors='none')
        pc.set_array(mu_mags)
        pc.set_clim(0, vmax)
        pc.set_cmap('hot_r')
        ax.add_collection(pc)
        cb = fig.colorbar(pc, ax=ax, shrink=0.85)
        cb.set_label(r'$|\boldsymbol{\mu}|$ (kJ/mol/unit)', fontsize=12)
        cb.ax.tick_params(labelsize=10)
    _style_2d_ax(ax, meta, xlabel=xlabel)
    ax.set_yticklabels([])
    ax.set_title(f'{title_prefix}{klabel} by force magnitude', fontsize=13)
    panel_label(ax, 'd')

    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=300, bbox_inches='tight')
        print(f"Saved: {output}")
    else:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 1-D DIAGNOSTIC FIGURE
# ═══════════════════════════════════════════════════════════════════════════

def plot_kernel_diagnostics_1d(meta, kernels, axes_grids, Z, ghat,
                               output=None, title_prefix=''):
    """4-panel (stacked) diagnostic figure for 1-D kernel populations."""
    kT    = meta['kT']
    gamma = meta['biasFactor']
    Vex   = compute_Vex(Z, kT, gamma)

    Nks     = np.array([k['Nk']        for k in kernels])
    centers = np.array([k['center'][0] for k in kernels])
    sigmas  = np.array([k['sigma'][0]  for k in kernels])

    x      = axes_grids[0]
    klabel = _kernel_label(meta)
    cv_names = meta.get('cv_names', [])
    xlabel   = cv_names[0] if cv_names else 'CV'
    space    = r'$\lambda$' if meta.get('kernel_type') == 'lambda' else r'$z$'

    fig, axs = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

    # (a) Density Z
    ax = axs[0]
    ax.plot(x, Z, 'k-', lw=1.5)
    ax.fill_between(x, 0, Z, alpha=0.2, color='steelblue')
    ax.set_ylabel(fr'$Z({space})$', fontsize=14)
    ax.set_title(f'{title_prefix}Kernel density', fontsize=13)
    ax.tick_params(**TICK_PARAMS)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    panel_label(ax, 'a')

    # (b) Exploration potential
    ax = axs[1]
    if gamma > 1.0:
        ax.plot(x, Vex, 'k-', lw=1.5)
        ax.fill_between(x, 0, Vex, alpha=0.2, color='orangered')
    else:
        ax.text(0.5, 0.5, r'$\gamma = 1$: no exploration',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=13, color='grey')
    ax.set_ylabel(r'$V_{\mathrm{ex}}$ (kJ/mol)', fontsize=14)
    ax.set_title(f'{title_prefix}Exploration potential', fontsize=13)
    ax.tick_params(**TICK_PARAMS)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    panel_label(ax, 'b')

    # (c) Kernel locations colored by Nk
    ax = axs[2]
    sc = ax.scatter(centers, np.zeros_like(centers), c=np.log10(Nks + 1),
                    cmap='turbo', s=20, zorder=3)
    ax.errorbar(centers, np.zeros_like(centers), xerr=sigmas,
                fmt='none', ecolor='grey', elinewidth=0.3, alpha=0.4, zorder=2)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label(r'$\log_{10}(N_k + 1)$', fontsize=11)
    ax.set_ylabel(klabel, fontsize=14)
    ax.set_title(f'{title_prefix}{klabel} ($M = {len(kernels)}$)', fontsize=13)
    ax.set_yticks([])
    ax.tick_params(**TICK_PARAMS)
    panel_label(ax, 'c')

    # (d) NW mean force ĝ
    ax = axs[3]
    ax.plot(x, ghat[:, 0], 'k-', lw=1.5)
    ax.axhline(0, color='grey', lw=0.5, ls='--')
    ax.set_ylabel(fr'$\hat{{g}}({space})$', fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_title(f'{title_prefix}NW mean force', fontsize=13)
    ax.tick_params(**TICK_PARAMS)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    panel_label(ax, 'd')

    ax.set_xlim(meta['domMin'][0], meta['domMax'][0])
    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=300, bbox_inches='tight')
        print(f"Saved: {output}")
    else:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def plot_kernel_diagnostics(filepath, grid_pts=200, nsigma=4.0,
                            output=None, title_prefix='', gamma=None,
                            swap_xy=False,
                            domMin=None, domMax=None,
                            periodic=None, kT=1.0):
    """Read an FK-ABF kernel file and produce a 4-panel diagnostic figure.

    Parameters
    ----------
    filepath : str or Path
        CZAR z-kernel file or lambda-kernel dump (auto-detected).
    grid_pts : int
        Grid resolution per dimension (default 200).
    nsigma : float
        Kernel cutoff in sigma units (default 4.0, matching C++ NSIGMACUT).
    output : str or None
        Output path. None → plt.show().
    title_prefix : str
        Prepended to all panel titles (e.g. 'Step 5M: ').
    gamma : float or None
        Override γ for V_ex. CZAR files never carry biasFactor; always
        pass gamma explicitly when γ > 1.
    swap_xy : bool
        Swap CV1/CV2 axes (2-D only).
    domMin, domMax : list of float or None
        Domain bounds — lambda dump files only.
    periodic : list of bool/int or None
        Periodicity flags — lambda dump files only.
    kT : float
        Thermal energy — lambda dump files only (default 1.0).
    """
    meta, kernels = parse_kernel_file(filepath,
                                      domMin=domMin, domMax=domMax,
                                      periodic=periodic, kT=kT)
    dim = meta['dim']
    if gamma is not None:
        meta['biasFactor'] = gamma

    ktype = meta.get('kernel_type', '?')
    print(f"Loaded {len(kernels)} {ktype}-kernels from {Path(filepath).name}  "
          f"[dim={dim}, kT={meta['kT']:.4f}, γ={meta['biasFactor']:.2f}]")
    if ktype == 'lambda':
        print(f"  step={meta.get('step','?')}  M={meta.get('M_snapshot', len(kernels))}  "
              f"domain: {[f'{v:.3f}' for v in meta['domMin']]} "
              f"→ {[f'{v:.3f}' for v in meta['domMax']]}")
    if meta.get('biasFactor', 1.0) > 1.0 and kT == 1.0 and ktype == 'lambda':
        print("  NOTE: V_ex uses kT=1.0. Pass --kT <value> if units require it.")

    axes_grids, Z, ghat = evaluate_on_grid(meta, kernels, grid_pts, nsigma)

    if swap_xy and dim == 2:
        axes_grids        = [axes_grids[1], axes_grids[0]]
        Z                 = Z.T
        ghat              = ghat.transpose(1, 0, 2)[:, :, ::-1]
        meta['domMin']    = [meta['domMin'][1],    meta['domMin'][0]]
        meta['domMax']    = [meta['domMax'][1],    meta['domMax'][0]]
        meta['periodic']  = [meta['periodic'][1],  meta['periodic'][0]]
        if 'cv_names' in meta:
            meta['cv_names'] = [meta['cv_names'][1], meta['cv_names'][0]]
        for k in kernels:
            k['center'] = [k['center'][1], k['center'][0]]
            k['sigma']  = [k['sigma'][1],  k['sigma'][0]]
            k['mu']     = [k['mu'][1],     k['mu'][0]]

    if dim == 1:
        plot_kernel_diagnostics_1d(meta, kernels, axes_grids, Z, ghat,
                                   output=output, title_prefix=title_prefix)
    elif dim == 2:
        plot_kernel_diagnostics_2d(meta, kernels, axes_grids, Z, ghat,
                                   output=output, title_prefix=title_prefix)
    else:
        # 3-D: print summary statistics only
        print(f"  3D not supported for plotting; Z grid shape = {Z.shape}")
        print(f"  Z range: {Z.min():.2f} – {Z.max():.2f}")
        print(f"  Populated nodes: {(Z > 0).sum()} / {Z.size}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='FK-ABF kernel diagnostic plotter (CZAR or lambda dump)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Lambda-kernel dump, Muller-Brown (natural units, non-periodic):
  python kernel_diagnostics.py fk.kernels_05000000.dat \\
      --gridmin -1.5 -0.5 --gridmax 1.2 2.0 --kT 1.0 -o diag.pdf

  # Alanine dipeptide (periodic dihedrals):
  python kernel_diagnostics.py fk.kernels_05000000.dat \\
      --gridmin -3.14159 -3.14159 --gridmax 3.14159 3.14159 \\
      --periodic 1 1 --kT 2.479 --gamma 10 -o diag.pdf

  # CZAR z-kernel file (all metadata in file; --gamma to show V_ex):
  python kernel_diagnostics.py fk.czar_kernels_05000000.dat \\
      --gamma 10 -o czar.pdf
""")
    parser.add_argument('filepath',
                        help='Kernel file (lambda dump or CZAR z-kernel file)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output path (default: interactive)')
    parser.add_argument('--grid', type=int, default=200,
                        help='Grid points per dimension (default: 200)')
    parser.add_argument('--nsigma', type=float, default=4.0,
                        help='Kernel cutoff in sigma units (default: 4.0)')
    parser.add_argument('--title', default='',
                        help='Title prefix prepended to all panels')
    parser.add_argument('--gamma', type=float, default=None,
                        help='Override bias factor γ for V_ex visualisation')
    parser.add_argument('--swap-xy', action='store_true',
                        help='Swap CV1/CV2 axes (2D only)')
    # Lambda-dump domain arguments (ignored for CZAR files)
    parser.add_argument('--gridmin', nargs='+', type=float, default=None,
                        help='Domain lower bounds per dim '
                             '(lambda dump only; estimated if omitted)')
    parser.add_argument('--gridmax', nargs='+', type=float, default=None,
                        help='Domain upper bounds per dim (lambda dump only)')
    parser.add_argument('--periodic', nargs='+', type=int, default=None,
                        help='Periodicity flags 0/1 per dim (lambda dump only; default: all 0)')
    parser.add_argument('--kT', type=float, default=1.0,
                        help='Thermal energy kT for V_ex (lambda dump only; default: 1.0)')
    args = parser.parse_args()

    plot_kernel_diagnostics(
        args.filepath,
        grid_pts=args.grid,
        nsigma=args.nsigma,
        output=args.output,
        title_prefix=args.title,
        gamma=args.gamma,
        swap_xy=args.swap_xy,
        domMin=args.gridmin,
        domMax=args.gridmax,
        periodic=args.periodic,
        kT=args.kT,
    )
