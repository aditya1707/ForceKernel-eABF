#!/usr/bin/env python3
"""
FK-ABF run diagnostics: reads COLVAR and KERNELINFO files and produces
publication-quality diagnostic figures.

Figures produced:
  1. fig_trajectory.pdf       — CV and λ time series + z−λ histograms
  2. fig_bias.pdf             — V_ex, force², and bias force components
  3. fig_kernels.pdf          — M, zM, neff, sigma evolution
  4. fig_exploration.pdf      — 2D trajectory in CV space colored by time
  5. fig_phase.pdf            — z vs λ scatter per CV (coupling diagnostic)

Usage:
    For generic analysis:
    python fkabf_diagnostics.py
    
    If COLVAR/KERNELINFO have different names and replace step w/ time
    python fkabf_diagnostics.py --colvar COLVAR --kernelinfo KERNELINFO --dt 0.001

    Print every X steps
    python fkabf_diagnostics.py --colvar COLVAR --prefix fk --thinning 10

    Take differences with periodic CV
    python fkabf_diagnostics.py --periodic "phi:-pi:pi,psi:-pi:pi"
"""

import argparse
import os
import sys
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import matplotlib.gridspec as gridspec

# ─── House style ──────────────────────────────────────────────────────────
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.it'] = 'cmmi10'
plt.rcParams['font.family'] = 'DejaVu Sans'

TICK_PARAMS = dict(axis='both', which='both', direction='in',
                   top=True, right=True, length=5, width=1, labelsize=14)
TICK_PARAMS_MINOR = dict(axis='both', which='minor', direction='in',
                         top=True, right=True, length=3, width=0.7)
LABEL_SIZE = 18
TITLE_SIZE = 16
PANEL_FONT = 28

# Palette
C_BLUE    = '#2c7bb6'
C_RED     = '#d7191c'
C_GREEN   = '#1a9641'
C_ORANGE  = '#fdae61'
C_PURPLE  = '#7b3294'
C_GREY    = '#666666'
C_TEAL    = '#008080'
CV_COLORS = [C_BLUE, C_RED, C_GREEN, C_ORANGE]


def style_ax(ax, xlabel=None, ylabel=None, title=None):
    ax.tick_params(**TICK_PARAMS)
    ax.tick_params(**TICK_PARAMS_MINOR)
    ax.tick_params(axis='x', pad=7)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, labelpad=2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, labelpad=2)
    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, pad=10)


def panel_label(ax, label, x=-0.14, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=PANEL_FONT, fontweight='bold', va='top', ha='left')


# ─── File readers ─────────────────────────────────────────────────────────

def read_plumed_file(path):
    """Read a PLUMED-style file with #! FIELDS header. Returns dict of arrays."""
    header = None
    with open(path) as f:
        for line in f:
            if line.startswith('#! FIELDS'):
                header = line.strip().split()[2:]
                break
    if header is None:
        raise ValueError(f"No #! FIELDS header found in {path}")
    data = np.loadtxt(path, comments='#')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    result = {}
    for i, name in enumerate(header):
        if i < data.shape[1]:
            result[name] = data[:, i]
    return result


def detect_cv_names(colvar, prefix):
    """Detect CV names and fictitious variable names from COLVAR columns."""
    fict_suffix = '_fict'
    fict_keys = [k for k in colvar if k.endswith(fict_suffix) and k.startswith(prefix + '.')]
    cv_names = []
    fict_names = []
    for fk in sorted(fict_keys):
        # fk = "fk.d1.x_fict" → cv = "d1.x"
        cv = fk[len(prefix)+1:-len(fict_suffix)]
        if cv in colvar:
            cv_names.append(cv)
            fict_names.append(fk)
    # Fallback: try without prefix
    if not cv_names:
        fict_keys = [k for k in colvar if k.endswith(fict_suffix)]
        for fk in sorted(fict_keys):
            cv = fk[:-len(fict_suffix)]
            if cv in colvar:
                cv_names.append(cv)
                fict_names.append(fk)
    return cv_names, fict_names


def steps_from_colvar(colvar, dt):
    """Get step array. Uses 'time' column if present, else row index × PACE."""
    if 'time' in colvar:
        return colvar['time'] / dt
    return np.arange(len(next(iter(colvar.values()))))


def thin(arr, n):
    """Thin an array by factor n for plotting."""
    if n <= 1:
        return arr
    return arr[::n]


def periodic_delta(z, lam, period):
    """Minimum-image periodic difference: wraps z-lam into [-period/2, +period/2]."""
    d = z - lam
    d = d - period * np.round(d / period)
    return d


def parse_periodic(pstring):
    """Parse --periodic argument into dict: cv_name -> period.

    Format: 'cv1:period1,cv2:period2' or 'cv1:min:max,cv2:min:max'
    Examples:
        'phi:-pi:pi,psi:-pi:pi'   → {'phi': 2π, 'psi': 2π}
        'phi:6.2832,psi:6.2832'   → {'phi': 6.2832, 'psi': 6.2832}
    """
    if not pstring:
        return {}
    result = {}
    for token in pstring.split(','):
        parts = token.strip().split(':')
        name = parts[0].strip()
        if len(parts) == 3:
            # min:max format
            lo = _parse_pi(parts[1].strip())
            hi = _parse_pi(parts[2].strip())
            result[name] = hi - lo
        elif len(parts) == 2:
            # direct period
            result[name] = _parse_pi(parts[1].strip())
        else:
            raise ValueError(f"Cannot parse periodic spec: '{token}'")
    return result


def _parse_pi(s):
    """Parse a string that may contain 'pi', e.g. '-pi', '2*pi', '3.14'."""
    s = s.replace('pi', str(np.pi))
    return float(eval(s))


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: CV and λ trajectories + z−λ histograms
# ═══════════════════════════════════════════════════════════════════════════

def plot_trajectory(colvar, cv_names, fict_names, steps, thinning,
                    pdict=None, output='fig_trajectory.pdf'):
    if pdict is None:
        pdict = {}
    dim = len(cv_names)
    fig = plt.figure(figsize=(16, 4.5 * dim))
    gs = gridspec.GridSpec(dim, 3, width_ratios=[3, 3, 1.2],
                           hspace=0.35, wspace=0.35)

    step_M = thin(steps, thinning) / 1e6

    for i in range(dim):
        z_i = thin(colvar[cv_names[i]], thinning)
        lam_i = thin(colvar[fict_names[i]], thinning)

        # Minimum-image difference for periodic CVs
        period = pdict.get(cv_names[i], None)
        if period is not None:
            delta_i = periodic_delta(colvar[cv_names[i]],
                                     colvar[fict_names[i]], period)
        else:
            delta_i = colvar[cv_names[i]] - colvar[fict_names[i]]

        # (a) z and λ vs time
        ax_ts = fig.add_subplot(gs[i, 0])
        ax_ts.plot(step_M, z_i, lw=0.3, alpha=0.6, color=CV_COLORS[i % 4],
                   label=fr'$z$ ({cv_names[i]})', rasterized=True)
        ax_ts.plot(step_M, lam_i, lw=0.3, alpha=0.6, color=C_GREY,
                   label=fr'$\lambda$ ({cv_names[i]})', rasterized=True)
        style_ax(ax_ts, xlabel=r'Steps $\times\,10^6$',
                 ylabel=cv_names[i])
        ax_ts.legend(fontsize=10, frameon=True, fancybox=False,
                     edgecolor='#cccccc', loc='upper right', markerscale=3)
        ax_ts.xaxis.set_minor_locator(AutoMinorLocator())
        ax_ts.yaxis.set_minor_locator(AutoMinorLocator())
        ax_ts.grid(True, alpha=0.2, which='major')
        panel_label(ax_ts, chr(ord('a') + i*3), x=-0.10, y=1.06)

        # (b) z − λ vs time
        ax_d = fig.add_subplot(gs[i, 1])
        delta_thin = thin(delta_i, thinning)
        ax_d.plot(step_M, delta_thin, lw=0.3, alpha=0.6,
                  color=CV_COLORS[i % 4], rasterized=True)
        ax_d.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
        style_ax(ax_d, xlabel=r'Steps $\times\,10^6$',
                 ylabel=fr'$z - \lambda$ ({cv_names[i]})')
        ax_d.xaxis.set_minor_locator(AutoMinorLocator())
        ax_d.yaxis.set_minor_locator(AutoMinorLocator())
        ax_d.grid(True, alpha=0.2, which='major')
        panel_label(ax_d, chr(ord('b') + i*3), x=-0.10, y=1.06)

        # (c) histogram of z − λ
        ax_h = fig.add_subplot(gs[i, 2])
        ax_h.hist(delta_i, bins=80, orientation='horizontal',
                  density=True, color=CV_COLORS[i % 4], alpha=0.7,
                  edgecolor='none')
        ax_h.set_ylim(ax_d.get_ylim())
        ax_h.set_yticklabels([])
        style_ax(ax_h, xlabel='Density')
        ax_h.xaxis.set_minor_locator(AutoMinorLocator())
        panel_label(ax_h, chr(ord('c') + i*3), x=-0.15, y=1.06)

    fig.savefig(output, dpi=200, bbox_inches='tight')
    print(f"  -> {output}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Bias force and exploration potential
# ═══════════════════════════════════════════════════════════════════════════

def plot_bias(colvar, steps, prefix, thinning, output='fig_bias.pdf'):
    force2_key = prefix + '.force2'
    wamp_key = prefix + '.wamp'

    has_force2 = force2_key in colvar
    has_wamp = wamp_key in colvar

    nrows = sum([has_force2, has_wamp])
    if nrows == 0:
        print("  [skip] fig_bias.pdf — no force2 or wamp columns found.")
        return

    fig, axes = plt.subplots(nrows, 1, figsize=(14, 4.0 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    step_M = thin(steps, thinning) / 1e6
    idx = 0

    if has_force2:
        ax = axes[idx]; idx += 1
        f2 = thin(colvar[force2_key], thinning)
        ax.plot(step_M, np.sqrt(f2), lw=0.4, alpha=0.5, color=C_BLUE,
                rasterized=True)
        style_ax(ax, ylabel=r'$|\mathbf{F}_{\mathrm{bias}}|$ (kJ/mol/unit)')
        ax.set_ylim(bottom=0)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, alpha=0.2, which='major')
        panel_label(ax, 'a', x=-0.06, y=1.06)

    if has_wamp:
        ax = axes[idx]; idx += 1
        vex = thin(colvar[wamp_key], thinning)
        ax.plot(step_M, vex, lw=0.4, alpha=0.5, color=C_PURPLE,
                rasterized=True)
        style_ax(ax, ylabel=r'$V_{\mathrm{ex}}$ (kJ/mol)')
        ax.set_ylim(bottom=0)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, alpha=0.2, which='major')
        panel_label(ax, chr(ord('a') + idx - 1), x=-0.06, y=1.06)

    axes[-1].set_xlabel(r'Steps $\times\,10^6$', fontsize=LABEL_SIZE)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches='tight')
    print(f"  -> {output}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Kernel compression diagnostics
# ═══════════════════════════════════════════════════════════════════════════

def plot_kernels(kinfo, cv_names, output='fig_kernels.pdf'):
    step_M = kinfo['step'] / 1e6

    # Detect sigma columns
    sigma_cols = [k for k in kinfo if '_sigma' in k]
    has_Z0 = 'Z0' in kinfo
    has_Zlocal = 'Zlocal' in kinfo
    has_totalN = 'totalN' in kinfo

    nrows = 3 + (1 if has_Z0 or has_Zlocal else 0)
    fig, axes = plt.subplots(nrows, 1, figsize=(14, 3.8 * nrows), sharex=True)

    # (a) M and zM
    ax = axes[0]
    ax.plot(step_M, kinfo['M'], lw=1.5, color=C_BLUE, label=r'$M$ ($\lambda$-kernels)')
    ax.plot(step_M, kinfo['zM'], lw=1.5, color=C_RED, label=r'$M_z$ ($z$-kernels)')
    if has_totalN:
        ax2 = ax.twinx()
        ax2.plot(step_M, kinfo['totalN'], lw=1.0, ls='--', color=C_GREY, alpha=0.6,
                 label=r'$N_{\mathrm{total}}$')
        ax2.set_ylabel(r'$N_{\mathrm{total}}$', fontsize=14, color=C_GREY)
        ax2.tick_params(axis='y', labelcolor=C_GREY, labelsize=12,
                        direction='in', length=4)
    style_ax(ax, ylabel='Kernel count')
    ax.legend(fontsize=11, frameon=True, fancybox=False,
              edgecolor='#cccccc', loc='upper left')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.2, which='major')
    panel_label(ax, 'a', x=-0.06, y=1.06)

    # (b) neff and compression ratio
    ax = axes[1]
    ax.plot(step_M, kinfo['neff'], lw=1.5, color=C_GREEN, label=r'$n_{\mathrm{eff}}$')
    if has_totalN and 'M' in kinfo:
        ratio = kinfo['totalN'] / np.maximum(kinfo['M'], 1)
        ax2 = ax.twinx()
        ax2.plot(step_M, ratio, lw=1.0, ls='--', color=C_ORANGE, alpha=0.8,
                 label=r'$N/M$ (samples/kernel)')
        ax2.set_ylabel(r'$N/M$', fontsize=14, color=C_ORANGE)
        ax2.tick_params(axis='y', labelcolor=C_ORANGE, labelsize=12,
                        direction='in', length=4)
    style_ax(ax, ylabel=r'$n_{\mathrm{eff}}$')
    ax.legend(fontsize=11, frameon=True, fancybox=False,
              edgecolor='#cccccc', loc='upper left')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.2, which='major')
    panel_label(ax, 'b', x=-0.06, y=1.06)

    # (c) Silverman bandwidth per CV
    ax = axes[2]
    for j, scol in enumerate(sigma_cols):
        # Extract CV name from column name (e.g. "d1.x_sigma" → "d1.x")
        cv_label = scol.replace('_sigma', '')
        ax.plot(step_M, kinfo[scol], lw=1.5,
                color=CV_COLORS[j % len(CV_COLORS)],
                label=fr'$\sigma$ ({cv_label})')
    style_ax(ax, ylabel=r'Silverman $\sigma$')
    ax.legend(fontsize=11, frameon=True, fancybox=False,
              edgecolor='#cccccc', loc='upper right')
    ax.set_ylim(bottom=0)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.2, which='major')
    panel_label(ax, 'c', x=-0.06, y=1.06)

    # (d) Z0 and Zlocal (if available)
    if has_Z0 or has_Zlocal:
        ax = axes[3]
        if has_Z0:
            ax.plot(step_M, kinfo['Z0'], lw=1.5, color=C_TEAL,
                    label=r'$Z_0$ (median)')
        if has_Zlocal:
            ax.plot(step_M, kinfo['Zlocal'], lw=0.5, alpha=0.5,
                    color=C_PURPLE, label=r'$Z(\lambda)$', rasterized=True)
            if has_Z0:
                # Ratio on twin axis
                ax2 = ax.twinx()
                ratio = kinfo['Zlocal'] / np.maximum(kinfo['Z0'], 1e-10)
                ax2.plot(step_M, ratio, lw=0.4, alpha=0.4, color=C_ORANGE,
                         rasterized=True, label=r'$Z(\lambda)/Z_0$')
                ax2.set_ylabel(r'$Z(\lambda)/Z_0$', fontsize=14, color=C_ORANGE)
                ax2.tick_params(axis='y', labelcolor=C_ORANGE, labelsize=12,
                                direction='in', length=4)
                ax2.axhline(1, color=C_ORANGE, lw=0.8, ls=':', alpha=0.5)
                ax2.legend(fontsize=10, frameon=True, fancybox=False,
                           edgecolor='#cccccc', loc='upper right')
        ax.set_yscale('log')
        style_ax(ax, ylabel=r'$Z$ (kernel support)')
        ax.legend(fontsize=11, frameon=True, fancybox=False,
                  edgecolor='#cccccc', loc='upper left')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, alpha=0.2, which='major')
        panel_label(ax, 'd', x=-0.06, y=1.06)

    axes[-1].set_xlabel(r'Steps $\times\,10^6$', fontsize=LABEL_SIZE)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches='tight')
    print(f"  -> {output}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: 2D trajectory colored by time (only for dim >= 2)
# ═══════════════════════════════════════════════════════════════════════════

def plot_exploration(colvar, cv_names, fict_names, steps, thinning,
                     output='fig_exploration.pdf'):
    if len(cv_names) < 2:
        print("  [skip] fig_exploration.pdf — requires 2+ CVs.")
        return

    fig, (ax_z, ax_lam) = plt.subplots(1, 2, figsize=(14, 6))

    step_M = thin(steps, thinning) / 1e6

    # (a) Real CV trajectory
    z0 = thin(colvar[cv_names[0]], thinning)
    z1 = thin(colvar[cv_names[1]], thinning)
    sc = ax_z.scatter(z0, z1, c=step_M, cmap='plasma', s=4, alpha=0.5,
                      edgecolors='none', rasterized=True)
    ax_z.set_aspect('equal')
    style_ax(ax_z, xlabel=cv_names[0], ylabel=cv_names[1])
    ax_z.set_title('Real CV trajectory', fontsize=TITLE_SIZE)
    panel_label(ax_z, 'a', x=-0.12, y=1.06)

    # (b) λ trajectory
    lam0 = thin(colvar[fict_names[0]], thinning)
    lam1 = thin(colvar[fict_names[1]], thinning)
    sc2 = ax_lam.scatter(lam0, lam1, c=step_M, cmap='plasma', s=4, alpha=0.5,
                         edgecolors='none', rasterized=True)
    ax_lam.set_aspect('equal')
    style_ax(ax_lam, xlabel=fict_names[0].split('.')[-1].replace('_fict', r'$_\lambda$'),
             ylabel=fict_names[1].split('.')[-1].replace('_fict', r'$_\lambda$'))
    ax_lam.set_title(r'Extended variable $\lambda$ trajectory', fontsize=TITLE_SIZE)
    panel_label(ax_lam, 'b', x=-0.12, y=1.06)

    # Shared colorbar
    cbar = fig.colorbar(sc2, ax=[ax_z, ax_lam], shrink=0.85, pad=0.03)
    cbar.set_label(r'Steps $\times\,10^6$', fontsize=LABEL_SIZE, labelpad=8)
    cbar.ax.tick_params(direction='in', labelsize=12, length=4)

    fig.savefig(output, dpi=200, bbox_inches='tight')
    print(f"  -> {output}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: z vs λ phase plots (coupling diagnostic)
# ═══════════════════════════════════════════════════════════════════════════

def plot_phase(colvar, cv_names, fict_names, steps, thinning,
               pdict=None, output='fig_phase.pdf'):
    if pdict is None:
        pdict = {}
    dim = len(cv_names)
    fig, axes = plt.subplots(1, dim, figsize=(6.5 * dim, 6))
    if dim == 1:
        axes = [axes]

    step_M = thin(steps, thinning) / 1e6

    for i in range(dim):
        ax = axes[i]
        z_i = thin(colvar[cv_names[i]], thinning)
        lam_i = thin(colvar[fict_names[i]], thinning)

        sc = ax.scatter(lam_i, z_i, c=step_M, cmap='plasma', s=3, alpha=0.4,
                        edgecolors='none', rasterized=True)

        # Identity line
        lo = min(z_i.min(), lam_i.min())
        hi = max(z_i.max(), lam_i.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.0, alpha=0.6)

        ax.set_aspect('equal')
        style_ax(ax, xlabel=fr'$\lambda$ ({cv_names[i]})',
                 ylabel=fr'$z$ ({cv_names[i]})')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, alpha=0.15, which='major')
        panel_label(ax, chr(ord('a') + i), x=-0.12, y=1.06)

    # Colorbar on last axis
    cbar = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.03)
    cbar.set_label(r'Steps $\times\,10^6$', fontsize=LABEL_SIZE, labelpad=8)
    cbar.ax.tick_params(direction='in', labelsize=12, length=4)

    fig.savefig(output, dpi=200, bbox_inches='tight')
    print(f"  -> {output}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Nlist and neighbor-list kernel count
# ═══════════════════════════════════════════════════════════════════════════

def plot_nlist(kinfo, output='fig_nlist.pdf'):
    if 'nlker' not in kinfo:
        print("  [skip] fig_nlist.pdf — no nlker column.")
        return

    step_M = kinfo['step'] / 1e6

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(step_M, kinfo['nlker'], lw=1.0, color=C_TEAL,
            alpha=0.7, label='nlker (neighbor list size)')

    if 'M' in kinfo:
        ax2 = ax.twinx()
        frac = kinfo['nlker'] / np.maximum(kinfo['M'], 1) * 100
        ax2.plot(step_M, frac, lw=0.8, ls='--', color=C_ORANGE, alpha=0.7,
                 label='nlker / M (%)')
        ax2.set_ylabel('nlker / M (%)', fontsize=14, color=C_ORANGE)
        ax2.tick_params(axis='y', labelcolor=C_ORANGE, labelsize=12,
                        direction='in', length=4)
        ax2.set_ylim(0, 100)
        ax2.legend(fontsize=10, frameon=True, fancybox=False,
                   edgecolor='#cccccc', loc='upper right')

    style_ax(ax, xlabel=r'Steps $\times\,10^6$', ylabel='Neighbor list size')
    ax.legend(fontsize=11, frameon=True, fancybox=False,
              edgecolor='#cccccc', loc='upper left')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.2, which='major')

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches='tight')
    print(f"  -> {output}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY PRINT
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(colvar, kinfo, cv_names, fict_names, prefix, dt, pdict=None):
    """Print a text summary of the run to stdout."""
    if pdict is None:
        pdict = {}
    print("\n" + "=" * 70)
    print("  FK-ABF Run Summary")
    print("=" * 70)

    nsteps = len(next(iter(colvar.values())))
    if 'time' in colvar:
        total_time = colvar['time'][-1] - colvar['time'][0]
        total_steps = total_time / dt
    else:
        total_steps = nsteps

    print(f"  COLVAR rows:    {nsteps}")
    print(f"  Total steps:    {total_steps:.0f}")
    print(f"  CVs detected:   {len(cv_names)}")
    for i, (cv, fict) in enumerate(zip(cv_names, fict_names)):
        z = colvar[cv]
        lam = colvar[fict]
        period = pdict.get(cv, None)
        if period is not None:
            delta = periodic_delta(z, lam, period)
            pstr = f"  (periodic, period={period:.4f})"
        else:
            delta = z - lam
            pstr = ""
        print(f"    CV {i}: {cv}{pstr}")
        print(f"      z    range: [{z.min():.4f}, {z.max():.4f}]  mean={z.mean():.4f}")
        print(f"      lam  range: [{lam.min():.4f}, {lam.max():.4f}]  mean={lam.mean():.4f}")
        print(f"      z-lam std:  {delta.std():.6f}")

    if kinfo is not None:
        print(f"\n  KERNELINFO rows: {len(kinfo['step'])}")
        print(f"  Final M:         {kinfo['M'][-1]:.0f}")
        print(f"  Final zM:        {kinfo['zM'][-1]:.0f}")
        print(f"  Final neff:      {kinfo['neff'][-1]:.1f}")
        if 'totalN' in kinfo:
            print(f"  Final totalN:    {kinfo['totalN'][-1]:.0f}")
            print(f"  Compression:     {kinfo['totalN'][-1]/max(kinfo['M'][-1],1):.1f}x "
                  f"(samples/kernel)")
        sigma_cols = [k for k in kinfo if '_sigma' in k]
        for sc in sigma_cols:
            print(f"  Final {sc}: {kinfo[sc][-1]:.6f}")
        if 'nlker' in kinfo:
            print(f"  Final nlker:     {kinfo['nlker'][-1]:.0f}  "
                  f"({kinfo['nlker'][-1]/max(kinfo['M'][-1],1)*100:.1f}% of M)")

    wamp_key = prefix + '.wamp'
    force2_key = prefix + '.force2'
    if wamp_key in colvar:
        vex = colvar[wamp_key]
        print(f"\n  V_ex:  final={vex[-1]:.4f}  max={vex.max():.4f}  "
              f"mean(last 10%)={vex[int(0.9*len(vex)):].mean():.4f}")
    if force2_key in colvar:
        f2 = colvar[force2_key]
        fmag = np.sqrt(f2)
        print(f"  |F_bias|: final={fmag[-1]:.2f}  mean(last 10%)="
              f"{fmag[int(0.9*len(fmag)):].mean():.2f}")

    print("=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='FK-ABF diagnostics: COLVAR + KERNELINFO analysis')
    parser.add_argument('--colvar', default='COLVAR',
                        help='Path to COLVAR file (default: COLVAR)')
    parser.add_argument('--kernelinfo', default='KERNELINFO',
                        help='Path to KERNELINFO file (default: KERNELINFO)')
    parser.add_argument('--prefix', default=None,
                        help='PLUMED label prefix (auto-detected if not set)')
    parser.add_argument('--dt', type=float, default=0.001,
                        help='MD timestep in time units (default: 0.001)')
    parser.add_argument('--thinning', type=int, default=10,
                        help='Thinning factor for trajectory plots (default: 10)')
    parser.add_argument('--periodic', default='',
                        help='Periodic CVs: "cv1:min:max,cv2:min:max" or "cv1:period". '
                             'Supports "pi" in expressions. '
                             'Example: "phi:-pi:pi,psi:-pi:pi"')
    parser.add_argument('--outdir', default='.',
                        help='Output directory for figures (default: .)')
    args = parser.parse_args()

    # ── Read COLVAR ──────────────────────────────────────────────────────
    if not os.path.isfile(args.colvar):
        print(f"ERROR: COLVAR file not found: {args.colvar}")
        sys.exit(1)

    print(f"Reading {args.colvar} ...")
    colvar = read_plumed_file(args.colvar)
    print(f"  Columns: {list(colvar.keys())}")

    # Auto-detect prefix from _fict columns
    prefix = args.prefix
    if prefix is None:
        for k in colvar:
            if '_fict' in k and '.' in k:
                # e.g. "fk.d1.x_fict" → prefix = "fk"
                prefix = k.split('.')[0]
                break
        if prefix is None:
            prefix = 'fk'
    print(f"  Prefix: {prefix}")

    cv_names, fict_names = detect_cv_names(colvar, prefix)
    if not cv_names:
        print("ERROR: Could not detect CV / _fict column pairs.")
        print("  Available columns:", list(colvar.keys()))
        sys.exit(1)
    print(f"  CVs: {cv_names}")
    print(f"  Fict: {fict_names}")

    steps = steps_from_colvar(colvar, args.dt)

    # ── Parse periodic CV specifications ─────────────────────────────────
    pdict = parse_periodic(args.periodic)
    if pdict:
        print(f"  Periodic CVs: {pdict}")

    # ── Read KERNELINFO ──────────────────────────────────────────────────
    kinfo = None
    if os.path.isfile(args.kernelinfo):
        print(f"Reading {args.kernelinfo} ...")
        kinfo = read_plumed_file(args.kernelinfo)
        print(f"  Columns: {list(kinfo.keys())}")
    else:
        print(f"  KERNELINFO not found ({args.kernelinfo}), skipping kernel plots.")

    # ── Summary ──────────────────────────────────────────────────────────
    print_summary(colvar, kinfo, cv_names, fict_names, prefix, args.dt, pdict)

    # ── Generate figures ─────────────────────────────────────────────────
    od = args.outdir
    os.makedirs(od, exist_ok=True)
    th = args.thinning

    print("Generating figures ...")

    plot_trajectory(colvar, cv_names, fict_names, steps, th,
                    pdict=pdict,
                    output=os.path.join(od, 'fig_trajectory.pdf'))

    plot_bias(colvar, steps, prefix, th,
              output=os.path.join(od, 'fig_bias.pdf'))

    if kinfo is not None:
        plot_kernels(kinfo, cv_names,
                     output=os.path.join(od, 'fig_kernels.pdf'))
        plot_nlist(kinfo,
                   output=os.path.join(od, 'fig_nlist.pdf'))

    plot_exploration(colvar, cv_names, fict_names, steps, th,
                     output=os.path.join(od, 'fig_exploration.pdf'))

    plot_phase(colvar, cv_names, fict_names, steps, th,
               pdict=pdict,
               output=os.path.join(od, 'fig_phase.pdf'))

    print("\nDone.")


if __name__ == '__main__':
    main()
