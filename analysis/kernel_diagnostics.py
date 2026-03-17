#!/usr/bin/env python3
"""
FK-eABF kernel diagnostic plotter.

Reads a lambda-kernel dump file and produces a 2×2 diagnostic figure:
  (a) Kernel density Z(λ) — NW denominator on a grid
  (b) Exploration potential V_ex(λ) = c·ln(1 + Z/Z₀)
  (c) Kernel ellipses colored by sample count Nk
  (d) Kernel ellipses colored by |μ| (mean force magnitude)

Works for 1D (line plots) and 2D (contour + ellipse plots).

Usage:
    from kernel_diagnostics import plot_kernel_diagnostics
    plot_kernel_diagnostics('fk.kernels_05000000.dat', output='diag.pdf')

    # Or from command line:
    python kernel_diagnostics.py fk.kernels_05000000.dat --output diag.pdf
"""

import numpy as np
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

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


# ─── Kernel file parser ──────────────────────────────────────────────────

def parse_kernel_file(filepath):
    """Parse a FK-eABF kernel dump file.

    Returns
    -------
    meta : dict with keys 'dim', 'kT', 'kappa', 'periodic', 'domMin',
           'domMax', 'biasFactor'
    kernels : list of dicts with keys 'Nk', 'center', 'mu', 'sigma'
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
                dim = int(tokens[1])
                meta['dim'] = dim
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
            elif key == 'biasFactor':
                meta['biasFactor'] = float(tokens[1])
            elif key == 'nkernels':
                pass  # informational
            else:
                # Data line: Nk center[d] mu[d] sigma[d]
                vals = [float(t) for t in tokens]
                if dim is not None and len(vals) == 1 + 3 * dim:
                    kernels.append({
                        'Nk': vals[0],
                        'center': vals[1:1+dim],
                        'mu': vals[1+dim:1+2*dim],
                        'sigma': vals[1+2*dim:1+3*dim],
                    })

    if dim is None:
        raise ValueError(f"No 'dim' line found in {filepath}")

    return meta, kernels


# ─── Grid evaluation ──────────────────────────────────────────────────────

def evaluate_on_grid(meta, kernels, grid_pts=200, nsigma=4.0):
    """Evaluate Z(s) and NW mean force on a regular grid.

    Returns
    -------
    grids : list of 1D arrays (grid coordinates per dimension)
    Z : ndarray, NW denominator (density)
    ghat : ndarray of shape (*grid_shape, dim), NW mean force
    """
    dim = meta['dim']
    domMin = meta['domMin']
    domMax = meta['domMax']
    periodic = meta['periodic']

    # Build grid
    axes = []
    for d in range(dim):
        if periodic[d]:
            dx = (domMax[d] - domMin[d]) / grid_pts
            axes.append(np.linspace(domMin[d], domMax[d] - dx, grid_pts))
        else:
            axes.append(np.linspace(domMin[d], domMax[d], grid_pts))

    shape = tuple(len(a) for a in axes)
    Z = np.zeros(shape)
    numF = np.zeros(shape + (dim,))

    # Splat each kernel onto the grid
    for kern in kernels:
        c = kern['center']
        mu = kern['mu']
        sig = kern['sigma']
        Nk = kern['Nk']

        # Per-dimension 1D weights
        w1d = []
        idx1d = []
        for d in range(dim):
            inv4s2 = 1.0 / (4.0 * sig[d]**2 + 1e-300)
            dx = axes[d][1] - axes[d][0] if len(axes[d]) > 1 else 1.0
            R = int(np.ceil(nsigma * sig[d] / max(dx, 1e-12)))
            ic = int(np.round((c[d] - domMin[d]) / dx))
            N_d = len(axes[d])

            ws = []
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

        # Outer product splat
        if dim == 1:
            for a, wa in enumerate(w1d[0]):
                gi = idx1d[0][a]
                wNk = wa * Nk
                Z[gi] += wNk
                numF[gi, 0] += wNk * mu[0]
        elif dim == 2:
            for a, wa in enumerate(w1d[0]):
                for b, wb in enumerate(w1d[1]):
                    gi = (idx1d[0][a], idx1d[1][b])
                    wNk = wa * wb * Nk
                    Z[gi] += wNk
                    numF[gi + (0,)] += wNk * mu[0]
                    numF[gi + (1,)] += wNk * mu[1]
        else:
            for a, wa in enumerate(w1d[0]):
                for b, wb in enumerate(w1d[1]):
                    for cc_idx, wc in enumerate(w1d[2]):
                        gi = (idx1d[0][a], idx1d[1][b], idx1d[2][cc_idx])
                        wNk = wa * wb * wc * Nk
                        Z[gi] += wNk
                        for dd in range(dim):
                            numF[gi + (dd,)] += wNk * mu[dd]

    # NW mean force
    ghat = np.zeros_like(numF)
    nonzero = Z > 1e-300
    with np.errstate(divide='ignore', invalid='ignore'):
        for d in range(dim):
            ghat[..., d] = np.where(nonzero, numF[..., d] / Z, 0.0)

    return axes, Z, ghat


def compute_Vex(Z, kT, gamma):
    """Compute exploration potential V_ex = c·ln(1 + Z/Z0)."""
    if gamma <= 1.0:
        return np.zeros_like(Z)

    c = kT * (gamma - 1.0)
    Zpop = Z[Z > 1e-10]
    if len(Zpop) == 0:
        Z0 = 1.0
    else:
        Z0 = np.median(Zpop)
    if Z0 < 1e-10:
        Z0 = 1.0

    Vex = c * np.log(1.0 + Z / (Z0 + 1e-300))
    return Vex


# ─── Plotting (2D) ────────────────────────────────────────────────────────

def _make_ellipses(kernels, dim, scale=2.0):
    """Create matplotlib Ellipse patches from kernel centers/sigmas.
    scale controls the displayed ellipse size (in units of sigma)."""
    patches = []
    for kern in kernels:
        if dim == 2:
            e = Ellipse(xy=(kern['center'][0], kern['center'][1]),
                        width=scale * 2 * kern['sigma'][0],
                        height=scale * 2 * kern['sigma'][1],
                        angle=0)
            patches.append(e)
    return patches


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


def plot_kernel_diagnostics_2d(meta, kernels, axes, Z, ghat, output=None,
                               title_prefix='', z_label=r'$\lambda$'):
    """2×2 diagnostic figure for 2D kernel populations."""
    kT = meta['kT']
    gamma = meta['biasFactor']
    Vex = compute_Vex(Z, kT, gamma)

    Nks = np.array([k['Nk'] for k in kernels])
    mu_mags = np.array([np.sqrt(sum(m**2 for m in k['mu'])) for k in kernels])

    xu, yu = axes[0], axes[1]
    xlabel = r'$\phi$ (rad)' if meta['domMax'][0] < 10 else 'CV 1'
    ylabel = r'$\psi$ (rad)' if meta['domMax'][1] < 10 else 'CV 2'

    fig, axs = plt.subplots(2, 2, figsize=(12, 10.5))

    # ── (a) Density Z ─────────────────────────────────────────────────
    ax = axs[0, 0]
    Zplot = np.where(Z > 0, np.log10(Z + 1), 0)
    cf = ax.contourf(xu, yu, Zplot.T, levels=30, cmap='viridis')
    cf.set_edgecolor('face')
    cb = fig.colorbar(cf, ax=ax, shrink=0.85)
    cb.set_label(r'$\log_{10}(Z + 1)$', fontsize=12)
    cb.ax.tick_params(labelsize=10)
    _style_2d_ax(ax, meta, xlabel=xlabel, ylabel=ylabel)
    ax.set_title(f'{title_prefix}Kernel density $Z(\\lambda)$', fontsize=13)
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

    # ── (c) Kernels colored by Nk ────────────────────────────────────
    ax = axs[1, 0]
    patches = _make_ellipses(kernels, 2, scale=1.0)
    if patches:
        pc = PatchCollection(patches, alpha=0.6, edgecolors='none')
        pc.set_array(np.log10(Nks + 1))
        pc.set_cmap('turbo')
        ax.add_collection(pc)
        cb = fig.colorbar(pc, ax=ax, shrink=0.85)
        cb.set_label(r'$\log_{10}(N_k + 1)$', fontsize=12)
        cb.ax.tick_params(labelsize=10)
    _style_2d_ax(ax, meta, xlabel=xlabel, ylabel=ylabel)
    ax.set_title(f'{title_prefix}Kernels by sample count ($M = {len(kernels)}$)',
                 fontsize=13)
    panel_label(ax, 'c')

    # ── (d) Kernels colored by |μ| ───────────────────────────────────
    ax = axs[1, 1]
    patches = _make_ellipses(kernels, 2, scale=1.0)
    if patches:
        pc = PatchCollection(patches, alpha=0.6, edgecolors='none')
        pc.set_array(mu_mags)
        pc.set_cmap('RdBu_r')
        vmax = np.percentile(mu_mags, 95) if len(mu_mags) > 0 else 1.0
        pc.set_clim(-vmax, vmax)
        # For magnitude, use 0 to vmax
        pc.set_clim(0, vmax)
        pc.set_cmap('hot_r')
        ax.add_collection(pc)
        cb = fig.colorbar(pc, ax=ax, shrink=0.85)
        cb.set_label(r'$|\boldsymbol{\mu}|$ (kJ/mol/rad)', fontsize=12)
        cb.ax.tick_params(labelsize=10)
    _style_2d_ax(ax, meta, xlabel=xlabel)
    ax.set_yticklabels([])
    ax.set_title(f'{title_prefix}Kernels by force magnitude', fontsize=13)
    panel_label(ax, 'd')

    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=300, bbox_inches='tight')
        print(f"Saved: {output}")
    else:
        plt.show()
    plt.close(fig)


# ─── Plotting (1D) ────────────────────────────────────────────────────────

def plot_kernel_diagnostics_1d(meta, kernels, axes, Z, ghat, output=None,
                               title_prefix=''):
    """4-panel diagnostic figure for 1D kernel populations."""
    kT = meta['kT']
    gamma = meta['biasFactor']
    Vex = compute_Vex(Z, kT, gamma)

    Nks = np.array([k['Nk'] for k in kernels])
    centers = np.array([k['center'][0] for k in kernels])
    sigmas = np.array([k['sigma'][0] for k in kernels])
    mus = np.array([k['mu'][0] for k in kernels])

    x = axes[0]
    xlabel = 'CV'

    fig, axs = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

    # (a) Density Z
    ax = axs[0]
    ax.plot(x, Z, 'k-', lw=1.5)
    ax.fill_between(x, 0, Z, alpha=0.2, color='steelblue')
    ax.set_ylabel(r'$Z(\lambda)$', fontsize=14)
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
    ax.errorbar(centers, np.zeros_like(centers),
                xerr=sigmas, fmt='none', ecolor='grey',
                elinewidth=0.3, alpha=0.4, zorder=2)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label(r'$\log_{10}(N_k + 1)$', fontsize=11)
    ax.set_ylabel('Kernels', fontsize=14)
    ax.set_title(f'{title_prefix}Kernel locations ($M = {len(kernels)}$)',
                 fontsize=13)
    ax.set_yticks([])
    ax.tick_params(**TICK_PARAMS)
    panel_label(ax, 'c')

    # (d) NW mean force
    ax = axs[3]
    ax.plot(x, ghat[:, 0], 'k-', lw=1.5)
    ax.axhline(0, color='grey', lw=0.5, ls='--')
    ax.set_ylabel(r'$\hat{g}(\lambda)$', fontsize=14)
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


# ─── Main entry point ─────────────────────────────────────────────────────

def plot_kernel_diagnostics(filepath, grid_pts=200, nsigma=4.0,
                            output=None, title_prefix='', gamma=None,
                            swap_xy=False):
    """
    Read a kernel dump file and produce a diagnostic figure.

    Parameters
    ----------
    filepath : str
        Path to kernel dump file (e.g. fk.kernels_05000000.dat).
    grid_pts : int
        Grid resolution per dimension for Z and V_ex evaluation.
    nsigma : float
        Kernel cutoff in sigma units for grid evaluation.
    output : str or None
        Save to file; otherwise plt.show().
    title_prefix : str
        Prefix for panel titles (e.g. 'Step 5M: ').
    gamma : float or None
        Override bias factor from file. If None, uses the value
        in the kernel file header (defaults to 1.0 if absent).
    swap_xy : bool
        Swap CV1/CV2 axes for display (2D only).
    """
    meta, kernels = parse_kernel_file(filepath)
    dim = meta['dim']

    if gamma is not None:
        meta['biasFactor'] = gamma

    print(f"Loaded {len(kernels)} kernels, dim={dim}, "
          f"kT={meta['kT']:.4f}, γ={meta['biasFactor']:.1f}")

    axes, Z, ghat = evaluate_on_grid(meta, kernels, grid_pts, nsigma)

    if swap_xy and dim == 2:
        axes = [axes[1], axes[0]]
        Z = Z.T
        ghat = ghat.transpose(1, 0, 2)[:, :, ::-1]  # swap spatial + force components
        meta['domMin'] = [meta['domMin'][1], meta['domMin'][0]]
        meta['domMax'] = [meta['domMax'][1], meta['domMax'][0]]
        meta['periodic'] = [meta['periodic'][1], meta['periodic'][0]]
        for k in kernels:
            k['center'] = [k['center'][1], k['center'][0]]
            k['sigma'] = [k['sigma'][1], k['sigma'][0]]
            k['mu'] = [k['mu'][1], k['mu'][0]]

    if dim == 1:
        plot_kernel_diagnostics_1d(meta, kernels, axes, Z, ghat,
                                   output=output, title_prefix=title_prefix)
    elif dim == 2:
        plot_kernel_diagnostics_2d(meta, kernels, axes, Z, ghat,
                                   output=output, title_prefix=title_prefix)
    else:
        print(f"  3D not supported for plotting; Z grid shape = {Z.shape}")
        print(f"  Z range: {Z.min():.2f} to {Z.max():.2f}")
        print(f"  Populated nodes: {(Z > 0).sum()} / {Z.size}")


# ─── CLI ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='FK-eABF kernel diagnostic plotter')
    parser.add_argument('filepath', help='Kernel dump file')
    parser.add_argument('--output', '-o', default=None,
                        help='Output file (default: show)')
    parser.add_argument('--grid', type=int, default=200,
                        help='Grid points per dimension (default: 200)')
    parser.add_argument('--nsigma', type=float, default=4.0,
                        help='Kernel cutoff in sigma (default: 4.0)')
    parser.add_argument('--title', default='',
                        help='Title prefix for panels')
    parser.add_argument('--gamma', type=float, default=None,
                        help='Override bias factor (default: read from file)')
    parser.add_argument('--swap-xy', action='store_true',
                        help='Swap CV1/CV2 axes for display')
    args = parser.parse_args()

    plot_kernel_diagnostics(args.filepath, grid_pts=args.grid,
                            nsigma=args.nsigma, output=args.output,
                            title_prefix=args.title, gamma=args.gamma,
                            swap_xy=args.swap_xy)
