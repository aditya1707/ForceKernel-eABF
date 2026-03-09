#!/usr/bin/env python3
"""
fkabf_analysis.py
=================
Analysis and diagnostic plots for FKERNELABF v3+ simulations.

Reads the PLUMED COLVAR file (and optionally kernel dump files) and
produces a multi-panel PDF report covering:

  Panel 1  — Real vs fictitious CV time series (one subplot per CV)
  Panel 2  — z − λ coupling gap time series + histogram per CV
  Panel 3  — Spring force magnitude time series
  Panel 4  — Bias amplitude (Vamp), exploration potential V_ex over time
  Panel 5  — Kernel counts (M, zM) and Silverman sigma over time
  Panel 6  — 2D FEL from binned −V(λ) + biased density  [dims 1–2]
  Panel 7  — Kernel scatter: center positions coloured by |mu|  [from kernels]
  Panel 8  — Kernel Nk distribution, |mu|, sigma, and Nk-vs-|mu| correlation
  Panel 9  — Kernel snapshot evolution over time

The authoritative free energy surface comes from czar_integrate.py.
The binned −V(λ) FEL (≈ A(z) at convergence) is a quick diagnostic.

Usage
-----
    python fkabf_analysis.py --colvar COLVAR [OPTIONS]

Options
-------
    --colvar FILE          PLUMED COLVAR/output file  (required)
    --kernels PATTERN      Kernel dump file(s). Accepts:
                             - a single file (legacy appended format)
                             - a glob pattern, e.g. "fk.kernels_*.dat"
                             - a directory (auto-discovers *.kernels_*.dat)
    --output FILE          Output PDF  (default: fkabf_report.pdf)
    --stride N             Thin COLVAR rows by factor N for speed  (default: 1)
    --kappa V [V ...]      κ values per CV for coupling-gap units  (default: 1.0)
    --temp K               Temperature in K for kT annotation  (default: 300)
    --cvnames N [N...]     Override CV names  (e.g. phi psi)
    --biasfactor V         BIASFACTOR γ from the simulation.  If γ > 1,
                           reweighting uses A(s) = γ·(-kT ln P) instead of
                           exp(+V/kT) weights.  (default: 1.0 = standard ABF)
    --dt PS                MD timestep in ps for time axis  (default: 0.002)
    --show                 Display figure interactively instead of saving

Examples
--------
    # With density-based exploration (γ=6)
    python fkabf_analysis.py --colvar COLVAR --kernels "fk.kernels_*.dat" \\
        --kappa 3000 3000 --cvnames phi psi --temp 300 --biasfactor 6

    # Auto-discover kernel files in a directory
    python fkabf_analysis.py --colvar COLVAR --kernels ./output/ \\
        --kappa 3000 3000 --cvnames phi psi --biasfactor 4

    # Pure ABF (no exploration, biasfactor=1)
    python fkabf_analysis.py --colvar COLVAR --kernels "fk.kernels_*.dat"

    # No kernel files
    python fkabf_analysis.py --colvar COLVAR --stride 10 --output report.pdf
"""

import argparse
import glob
import os
import re
import sys
import warnings
import numpy as np

warnings.filterwarnings('ignore')


# ─────────────────────── file parsers ────────────────────────────────────────

def parse_colvar(path, stride=1):
    """
    Parse a PLUMED COLVAR file.
    Returns (fields, data, periodic_info) where periodic_info is a dict
    {cv_name: (lo, hi)} for CVs whose SET min_/max_ lines are in the header.
    """
    fields = None
    rows = []
    set_vals = {}

    def _parse_numeric(s):
        s = s.strip()
        if s == 'pi':  return np.pi
        if s == '-pi': return -np.pi
        return float(s)

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#! FIELDS'):
                fields = line.split()[2:]
                continue
            if line.startswith('#! SET'):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        set_vals[parts[2]] = _parse_numeric(parts[3])
                    except ValueError:
                        pass
                continue
            if line.startswith('#'):
                continue
            try:
                vals = [float(x) for x in line.split()]
            except ValueError:
                continue
            if fields is None:
                fields = [f'col{i}' for i in range(len(vals))]
            if len(vals) == len(fields):
                rows.append(vals)

    # Build periodic_info from matched min_xxx / max_xxx pairs
    periodic_info = {}
    for key, lo in set_vals.items():
        if key.startswith('min_'):
            cv = key[4:]
            if 'max_' + cv in set_vals:
                periodic_info[cv] = (lo, set_vals['max_' + cv])

    if not rows:
        sys.exit(f"ERROR: no data rows found in {path}")

    data = np.array(rows[::stride])
    print(f"  Loaded {len(rows)} rows -> {len(data)} after stride={stride}")
    print(f"  Columns: {fields}")
    if periodic_info:
        print(f"  Periodic CVs detected: {list(periodic_info.keys())}")
    return fields, data, periodic_info


def _parse_single_kernel_file(path):
    """
    Parse ONE kernel dump file (either a standalone step-stamped file
    or one snapshot block from a legacy appended file).

    Handles all known formats:
      v2+: k  Nk  center×d  mu×d  sigma×d  |mu|  wt       (no boost column)
      legacy:  k  Nk  center×d  mu×d  sigma×d  |mu|  boost  wt

    Note: the snapshot header no longer emits <lambda>=... because lambdaMax
    is now a fixed constant rather than a convergence diagnostic.  lambda_mean
    will be None for v2+ files; callers should treat None as expected, not missing.

    Returns list of snapshot dicts and a list of CV names.
    Each snapshot dict contains:
      step, M, totalN, neff, sigma, vamp, lambda_mean,
      kernels: list of dicts {k, Nk, center, mu, sigma, mu_mag, wt}
    """
    snapshots = []
    current = None
    cvnames = None
    n_center = None

    with open(path) as fh:
        for line in fh:
            line = line.rstrip()

            # Snapshot header (case-insensitive; all versions write "snapshot" or "Snapshot")
            if 'snapshot' in line.lower() and 'step=' in line:
                if current is not None:
                    snapshots.append(current)
                m_step = re.search(r'step=\s*(\d+)', line)
                m_M    = re.search(r'M=\s*(\d+)',    line)
                m_totN = re.search(r'totalN=\s*([0-9.eE+\-]+)', line)
                step  = int(m_step.group(1))  if m_step else 0
                M     = int(m_M.group(1))     if m_M    else 0
                totN  = float(m_totN.group(1)) if m_totN else 0.0
                current = {'step': step, 'M': M, 'totalN': totN,
                           'neff': None, 'sigma': [], 'vamp': None,
                           'lambda_mean': None, 'kernels': []}
                cvnames = None
                continue

            if current is None:
                continue

            # Second header line: neff, sigma, Vamp, lambda
            if 'neff(Silverman)' in line:
                m = re.search(r'neff\(Silverman\)=\s*([0-9.eE+\-]+)', line)
                if m: current['neff'] = float(m.group(1))
                m = re.search(r'sigma=\(([^)]+)\)', line)
                if m: current['sigma'] = [float(x) for x in m.group(1).split(',')]
                m = re.search(r'Vamp=\s*([0-9.eE+\-]+)', line)
                if m: current['vamp'] = float(m.group(1))
                # Note: <lambda>= line removed (lambdaMax is a fixed constant).
                # Still parsed here so legacy files continue to work.
                m = re.search(r'<lambda>=\s*([0-9.eE+\-]+)', line)
                if m: current['lambda_mean'] = float(m.group(1))
                continue

            # Column header line — extract CV names and detect format
            if line.lstrip().startswith('#') and 'c_' in line:
                parts = line.lstrip('# ').split()
                cvnames = []
                for p in parts:
                    if p.startswith('c_'):
                        cvnames.append(p[2:])
                n_center = len(cvnames)
                continue

            # Data line — starts with whitespace + integer index
            if line and not line.lstrip().startswith('#') and line.strip():
                parts = line.split()
                if not parts:
                    continue
                try:
                    k = int(parts[0])
                except ValueError:
                    continue

                d = n_center if n_center else 1

                # v2+ format: k  Nk  center×d  mu×d  sigma×d  |mu|  wt
                expected_v2 = 1 + 1 + d + d + d + 1 + 1
                # Legacy format: k  Nk  center×d  mu×d  sigma×d  |mu|  boost  wt
                expected_legacy = expected_v2 + 1

                try:
                    Nk     = float(parts[1])
                    center = [float(parts[2+i])     for i in range(d)]
                    mu     = [float(parts[2+d+i])   for i in range(d)]
                    sigma  = [float(parts[2+2*d+i]) for i in range(d)]
                    mu_mag = float(parts[2+3*d])

                    if len(parts) >= expected_legacy:
                        # Legacy format: skip the boost column (index 2+3*d+1)
                        wt = float(parts[2+3*d+2])
                    elif len(parts) >= expected_v2:
                        # v2+ format: no boost column
                        wt = float(parts[2+3*d+1])
                    else:
                        continue

                    current['kernels'].append({
                        'k': k, 'Nk': Nk, 'center': center, 'mu': mu,
                        'sigma': sigma, 'mu_mag': mu_mag, 'wt': wt
                    })
                except (ValueError, IndexError):
                    continue

    if current is not None:
        snapshots.append(current)

    return snapshots, cvnames or []


def load_kernel_snapshots(kernels_arg):
    """
    Discover and load kernel snapshot files from a path argument.

    kernels_arg can be:
      - a single file path             (legacy appended file OR single file)
      - a glob pattern                 (e.g. "fk.kernels_*.dat")
      - a directory                    (auto-discovers *kernels*.dat files)

    Returns (snapshots_list, cvnames) sorted by step.
    """
    # Determine the list of files to parse
    if os.path.isdir(kernels_arg):
        # Directory: discover kernel files
        patterns = [
            os.path.join(kernels_arg, '*kernels_*.dat'),
            os.path.join(kernels_arg, '*kernels_*.dat.gz'),
        ]
        files = []
        for pat in patterns:
            files.extend(glob.glob(pat))
        if not files:
            print(f"  WARNING: no kernel files found in directory {kernels_arg}")
            return [], []
    elif '*' in kernels_arg or '?' in kernels_arg:
        # Glob pattern
        files = glob.glob(kernels_arg)
        if not files:
            print(f"  WARNING: no files matched pattern {kernels_arg}")
            return [], []
    elif os.path.isfile(kernels_arg):
        # Single file (legacy appended or single snapshot)
        files = [kernels_arg]
    else:
        print(f"  WARNING: kernel path not found: {kernels_arg}")
        return [], []

    # Sort files by the step number embedded in the filename (if present)
    def _extract_step(path):
        m = re.search(r'_(\d{6,})\.dat', os.path.basename(path))
        return int(m.group(1)) if m else 0
    files.sort(key=_extract_step)

    # Parse all files and collect snapshots
    all_snapshots = []
    cvnames = []
    for f in files:
        snaps, cvn = _parse_single_kernel_file(f)
        all_snapshots.extend(snaps)
        if cvn and not cvnames:
            cvnames = cvn

    # Sort snapshots by step (handles mixed ordering)
    all_snapshots.sort(key=lambda s: s['step'])

    # Deduplicate by step (keep latest if duplicated)
    seen_steps = {}
    for s in all_snapshots:
        seen_steps[s['step']] = s
    deduped = sorted(seen_steps.values(), key=lambda s: s['step'])

    print(f"  Loaded {len(deduped)} kernel snapshots from {len(files)} file(s)")
    if deduped:
        print(f"  Step range: {deduped[0]['step']} -> {deduped[-1]['step']}")
    return deduped, cvnames


# ─────────────────────── column finders ──────────────────────────────────────

def find_col(fields, patterns):
    """Return index of first field matching any pattern (case-insensitive substring)."""
    for pat in patterns:
        for i, f in enumerate(fields):
            if pat.lower() in f.lower():
                return i
    return None


def find_cols(fields, patterns):
    """Return list of indices matching any pattern."""
    result = []
    for i, f in enumerate(fields):
        if any(p.lower() in f.lower() for p in patterns):
            result.append(i)
    return result


# ─────────────────────── plotting ────────────────────────────────────────────

def make_report(fields, data, snapshots, cvnames_kern, args, periodic_info=None):
    import matplotlib
    matplotlib.use('Agg' if not args.show else 'TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_pdf import PdfPages

    kB = 0.008314462618  # kJ/mol/K
    kT = kB * args.temp
    if periodic_info is None:
        periodic_info = {}

    def periodic_gap(z_arr, lam_arr, cv_name):
        """Minimum-image difference z - lambda for periodic CVs."""
        raw = z_arr - lam_arr
        if cv_name in periodic_info:
            lo, hi = periodic_info[cv_name]
            period = hi - lo
            raw = raw - period * np.round(raw / period)
        return raw

    # ── identify columns ──────────────────────────────────────────────────────
    time_col = find_col(fields, ['time'])
    time = data[:, time_col] if time_col is not None else np.arange(len(data)) * args.dt

    # Real CVs: anything that looks like phi/psi or generic arg names,
    # excluding _fict, bias., force2, lambda, nkernels, vamp, wamp, neff, sigma, nlker
    exclude = ['fict','bias','force2','lambda','nkernels','nzkernels',
               'vamp','wamp','vbias','alpha','neff','sigma','nlker','time']
    cv_cols = [i for i,f in enumerate(fields)
               if not any(e in f.lower() for e in exclude)
               and i != time_col]

    # Fictitious CVs: *_fict
    fict_cols = [i for i,f in enumerate(fields) if 'fict' in f.lower()]

    # Pair real with fict by CV name.
    # Handles both plain 'phi_fict' and prefixed 'fk.phi_fict' column names.
    cv_pairs = []  # list of (cv_name, real_col, fict_col)
    for rc in cv_cols:
        cv_name = fields[rc]
        fc = next((i for i, f in enumerate(fields)
                   if 'fict' in f.lower() and cv_name.lower() in f.lower()), None)
        cv_pairs.append((cv_name, rc, fc))

    # Use user-supplied names if given
    if args.cvnames and len(args.cvnames) == len(cv_pairs):
        cv_pairs = [(args.cvnames[i], cv_pairs[i][1], cv_pairs[i][2])
                    for i in range(len(cv_pairs))]

    ndim = len(cv_pairs)
    kappa_given = bool(args.kappa and len(args.kappa) >= 1)
    kappas = list(args.kappa) if kappa_given else [None] * ndim
    # Extend to ndim if only one value given
    if kappa_given and len(kappas) < ndim:
        kappas = (kappas * ndim)[:ndim]

    # Scalar diagnostic columns
    vamp_col   = find_col(fields, ['vamp'])
    wamp_col   = find_col(fields, ['wamp'])
    vbias_col  = find_col(fields, ['vbias'])
    nker_col   = find_col(fields, ['nkernels'])
    nzker_col  = find_col(fields, ['nzkernels'])
    sig_col    = find_col(fields, ['sigma'])
    neff_col   = find_col(fields, ['neff'])
    force2_col = find_col(fields, ['force2'])

    # ── figure layout ─────────────────────────────────────────────────────────
    have_kernels = len(snapshots) > 0

    pdf_path = args.output
    with PdfPages(pdf_path) as pdf:

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 1 — Real vs fictitious CV + coupling diagnostics
        # ══════════════════════════════════════════════════════════════════════
        fig = plt.figure(figsize=(14, 4 * ndim))
        gs  = gridspec.GridSpec(ndim, 3, figure=fig, wspace=0.35, hspace=0.45)
        fig.suptitle('Real vs Fictitious CV & Coupling Quality', fontsize=13, y=1.01)

        for di, (cvn, rc, fc) in enumerate(cv_pairs):
            z   = data[:, rc]
            lam = data[:, fc] if fc is not None else None
            gap = periodic_gap(z, lam, cvn) if lam is not None else None

            # Time series
            ax = fig.add_subplot(gs[di, 0:2])
            ax.scatter(time[::10], z[::10],   s=0.5, alpha=0.8, label=f'z ({cvn})',       color='steelblue')
            if lam is not None:
                ax.scatter(time[::10], lam[::10], s=0.5, alpha=0.6, label=f'lambda ({cvn}_fict)', color='tomato')
            ax.set_xlabel('Time (ps)')
            ax.set_ylabel(f'{cvn} (rad)')
            ax.set_title(f'CV: {cvn}')
            ax.legend(fontsize=8, loc='upper right')

            # Gap histogram
            ax2 = fig.add_subplot(gs[di, 2])
            if gap is not None:
                kap = kappas[di]
                from scipy.stats import norm as spnorm
                if kap is not None:
                    plot_gap = kap * gap
                    ideal_sd_force = np.sqrt(kT * kap)
                    xlabel = f'kappa*(z-lambda) [{cvn}] (kJ/mol/rad)'
                    clip = 8.0 * ideal_sd_force
                    x_lo, x_hi = -clip, clip
                else:
                    plot_gap = gap
                    ideal_sd_force = None
                    xlabel = f'z - lambda [{cvn}] (rad)'
                    clip = 8.0 * np.std(plot_gap)
                    x_lo, x_hi = -clip, clip

                mask = (plot_gap >= x_lo) & (plot_gap <= x_hi)
                n_clipped = (~mask).sum()
                ax2.hist(plot_gap[mask], bins=80, color='mediumpurple', alpha=0.8,
                         edgecolor='none', density=True, range=(x_lo, x_hi))
                mu_g = np.mean(plot_gap[mask])
                sd_g = np.std(plot_gap[mask])
                xg = np.linspace(x_lo, x_hi, 300)
                ax2.plot(xg, spnorm.pdf(xg, mu_g, sd_g), 'k--', lw=1.5,
                         label=f'N(μ={mu_g:.3f}, σ={sd_g:.4f})')
                if ideal_sd_force is not None:
                    ax2.axvline( ideal_sd_force, color='orange', lw=1.5, ls='--',
                                label=f'ideal sd={ideal_sd_force:.3f}')
                    ax2.axvline(-ideal_sd_force, color='orange', lw=1.5, ls='--')
                ax2.set_xlim(x_lo, x_hi)
                ax2.set_xlabel(xlabel, fontsize=8)
                ax2.set_ylabel('Density')
                title = f'Coupling gap: {cvn}'
                if n_clipped > 0:
                    title += f' ({n_clipped} outliers clipped)'
                ax2.set_title(title, fontsize=9)
                ax2.legend(fontsize=7)
            else:
                ax2.text(0.5, 0.5, 'No fictitious CV\ncolumn found',
                         ha='center', va='center', transform=ax2.transAxes,
                         color='grey', fontsize=9)
                ax2.set_title(f'Coupling gap: {cvn}')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 2 — Scalar diagnostics time series
        # ══════════════════════════════════════════════════════════════════════
        # v3+: wamp = V_ex (exploration potential), alpha is always 0.
        gamma = args.biasfactor

        # V_ex time series from wamp column
        wamp_ts = data[:, wamp_col] if wamp_col is not None else None

        # ── Standard scalar rows ──────────────────────────────────────────────
        diag_items = []
        if vamp_col   is not None: diag_items.append(('V(λ) grid amplitude Vamp (kJ/mol)', vamp_col,   'darkorange'))
        if neff_col   is not None: diag_items.append(('Neff at λ position',                neff_col,   'seagreen'))
        if force2_col is not None: diag_items.append(('Bias |F|² (kJ/mol/rad)²',           force2_col, 'crimson'))
        if sig_col    is not None: diag_items.append(('Silverman σ (dim 0)',                sig_col,    'purple'))

        # Exploration row: V(λ) + V_ex on one panel (same kJ/mol scale).
        have_explore_diag = (wamp_ts is not None or vbias_col is not None)
        n_explore_rows = 1 if (gamma > 1.0 and have_explore_diag) else (
                         1 if (vbias_col is not None) else 0)

        n_rows = len(diag_items) + n_explore_rows
        if n_rows > 0:
            fig, axes = plt.subplots(n_rows, 1,
                                     figsize=(14, 2.8 * n_rows),
                                     sharex=True)
            if n_rows == 1: axes = [axes]
            fig.suptitle('Scalar Diagnostics Over Time', fontsize=13)

            def plot_ts(ax, y, color, label=None):
                ax.plot(time, y, lw=0.6, color=color, alpha=0.85, label=label)

            # Standard rows
            for ax, (label, col, color) in zip(axes, diag_items):
                plot_ts(ax, data[:, col], color)
                ax.set_ylabel(label, fontsize=9)
                ax.legend(fontsize=7, loc='upper right')
                ax.grid(True, alpha=0.3)

            row = len(diag_items)

            # Exploration panel: V(λ) and V_ex on same kJ/mol axis.
            if have_explore_diag and n_explore_rows >= 1:
                ax = axes[row]
                if vbias_col is not None:
                    plot_ts(ax, data[:, vbias_col], 'darkorange', label='V(λ) at s_fict (kJ/mol)')
                if wamp_ts is not None:
                    plot_ts(ax, wamp_ts, 'royalblue', label='V_ex (kJ/mol)')
                ax.set_ylabel('Bias potential at s_fict (kJ/mol)', fontsize=9)
                ax.legend(fontsize=7, loc='upper left')
                ax.grid(True, alpha=0.3)

            axes[-1].set_xlabel('Time (ps)')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 3 — Kernel count + sigma
        # ══════════════════════════════════════════════════════════════════════
        if nker_col is not None or nzker_col is not None or sig_col is not None:
            fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
            fig.suptitle('Kernel Population & Bandwidth', fontsize=13)

            ax = axes[0]
            if nker_col is not None:
                ax.plot(time, data[:,nker_col],  lw=0.8, color='steelblue', label='M (λ-kernels)')
            if nzker_col is not None:
                ax.plot(time, data[:,nzker_col], lw=0.8, color='tomato',    label='zM (z-kernels)', alpha=0.7)
            ax.set_ylabel('Kernel count')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            ax = axes[1]
            if sig_col is not None:
                ax.plot(time, data[:,sig_col], lw=0.8, color='purple', label='σ (dim 0)')
                if args.cvnames and len(args.cvnames) > 0:
                    sig_cols_all = find_cols(fields, ['sigma'])
                    colors = ['purple','darkorange','seagreen']
                    for j, sc in enumerate(sig_cols_all[:3]):
                        lbl = fields[sc]
                        ax.plot(time, data[:,sc], lw=0.8, color=colors[j], label=lbl, alpha=0.8)
            ax.set_ylabel('Silverman σ')
            ax.set_xlabel('Time (ps)')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 4 — 2D FEL from binned −V(λ) + biased density
        # ══════════════════════════════════════════════════════════════════════
        # At convergence, −∇V cancels ∇A, so V ≈ −A + const.
        # Binning −fk.vbias by (z₀, z₁) gives A(z) directly.
        # With tight coupling (κ large), λ ≈ z, so this works on the real CVs.
        # Empty bins are filled by periodic nearest-neighbor interpolation,
        # then a light Gaussian smooth removes bin noise.
        if ndim >= 2:
            z0 = data[:, cv_pairs[0][1]]
            z1 = data[:, cv_pairs[1][1]]
            cv0_name = cv_pairs[0][0]
            cv1_name = cv_pairs[1][0]

            n_bin = 72
            lo0 = z0.min() - 0.02; hi0 = z0.max() + 0.02
            lo1 = z1.min() - 0.02; hi1 = z1.max() + 0.02
            if cv_pairs[0][0] in periodic_info:
                lo0, hi0 = periodic_info[cv_pairs[0][0]]
            if cv_pairs[1][0] in periodic_info:
                lo1, hi1 = periodic_info[cv_pairs[1][0]]

            edges0 = np.linspace(lo0, hi0, n_bin + 1)
            edges1 = np.linspace(lo1, hi1, n_bin + 1)
            c0 = 0.5 * (edges0[:-1] + edges0[1:])
            c1 = 0.5 * (edges1[:-1] + edges1[1:])
            dx0 = (hi0 - lo0) / n_bin
            dx1 = (hi1 - lo1) / n_bin

            # Helper for periodic gap-filling (used by both FEL and density panels)
            def _nanmean(vals):
                f = vals[np.isfinite(vals)]
                return np.mean(f) if len(f) > 0 else np.nan

            FEL_smooth = None
            FEL_biased = None

            if vbias_col is not None:
                vb = data[:, vbias_col]
                sum_v = np.zeros((n_bin, n_bin))
                count_v = np.zeros((n_bin, n_bin))
                for t in range(len(z0)):
                    i = min(max(int((z0[t] - lo0) / dx0), 0), n_bin - 1)
                    j = min(max(int((z1[t] - lo1) / dx1), 0), n_bin - 1)
                    sum_v[i, j] += -vb[t]  # negate: V ≈ −A, so −V ≈ A
                    count_v[i, j] += 1

                raw = np.where(count_v > 0, sum_v / count_v, np.nan)

                # Fill empty bins via periodic nearest-neighbor, then smooth
                try:
                    from scipy.ndimage import generic_filter, gaussian_filter
                    raw_pad = np.tile(raw, (3, 3))
                    mask_nan = ~np.isfinite(raw_pad)
                    filled = raw_pad.copy()
                    for _ in range(5):
                        filled2 = generic_filter(filled, _nanmean, size=3,
                                                 mode='constant', cval=np.nan)
                        filled[mask_nan] = filled2[mask_nan]
                        mask_nan = ~np.isfinite(filled)
                        if not mask_nan.any():
                            break
                    smoothed = gaussian_filter(filled, sigma=1.5, mode='wrap')
                    FEL_smooth = smoothed[n_bin:2*n_bin, n_bin:2*n_bin]
                    FEL_smooth -= FEL_smooth.min()
                except ImportError:
                    # No scipy: just use raw with NaN
                    FEL_smooth = raw.copy()
                    FEL_smooth -= np.nanmin(FEL_smooth)

            # Biased density with smoothing
            h, _, _ = np.histogram2d(z0, z1, bins=n_bin,
                                     range=[[lo0, hi0], [lo1, hi1]])
            h_raw = np.where(h > 0, h, np.nan)
            FEL_biased_raw = -kT * np.log(h_raw / np.nanmax(h_raw))
            try:
                from scipy.ndimage import generic_filter as _gf2, gaussian_filter as _gs2
                bio_pad = np.tile(FEL_biased_raw, (3, 3))
                bio_mask = ~np.isfinite(bio_pad)
                bio_filled = bio_pad.copy()
                for _ in range(5):
                    bf2 = _gf2(bio_filled, _nanmean, size=3,
                               mode='constant', cval=np.nan)
                    bio_filled[bio_mask] = bf2[bio_mask]
                    bio_mask = ~np.isfinite(bio_filled)
                    if not bio_mask.any():
                        break
                bio_smooth = _gs2(bio_filled, sigma=1.5, mode='wrap')
                FEL_biased = bio_smooth[n_bin:2*n_bin, n_bin:2*n_bin]
                FEL_biased -= FEL_biased.min()
            except ImportError:
                FEL_biased = FEL_biased_raw

            # ── Plot ─────────────────────────────────────────────────────
            ncols = 3 if FEL_smooth is not None else 2
            fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5))
            fig.suptitle('2D Free Energy Surface', fontsize=13)

            # Panel A: biased density
            ax = axes[0]
            vmax = min(40.0, float(np.nanpercentile(
                FEL_biased[np.isfinite(FEL_biased)], 98)))
            levels = np.linspace(0, max(vmax, 0.1), 24)
            cs = ax.contourf(c1, c0, FEL_biased.T, levels=levels,
                             cmap='RdYlBu_r', extend='max')
            ax.contour(c1, c0, FEL_biased.T, levels=levels[::2],
                       colors='k', linewidths=0.4, alpha=0.5)
            plt.colorbar(cs, ax=ax, label='-kT ln p(z) (kJ/mol)')
            ax.set_xlabel(f'{cv1_name} (rad)')
            ax.set_ylabel(f'{cv0_name} (rad)')
            ax.set_title('Biased sampling −kT ln P(z)')

            if FEL_smooth is not None:
                # Panel B: smoothed −V(λ) FEL
                ax = axes[1]
                finite_d = FEL_smooth[np.isfinite(FEL_smooth)]
                vmax_d = min(80.0, float(np.nanpercentile(finite_d, 95)))
                lvl_d = np.linspace(0, vmax_d+10, 40)
                cs2 = ax.contourf(c1, c0, FEL_smooth.T, levels=lvl_d,
                                  cmap='RdYlBu_r', extend='max')
                ax.contour(c1, c0, FEL_smooth.T, levels=lvl_d[::2],
                           colors='k', linewidths=0.4, alpha=0.5)
                plt.colorbar(cs2, ax=ax, label='A(z) (kJ/mol)')
                ax.set_xlabel(f'{cv1_name} (rad)')
                ax.set_ylabel(f'{cv0_name} (rad)')
                ax.set_title('FEL from −V(λ)\n(≈ A(z) at convergence)')

                # Panel C: hexbin for coverage
                ax = axes[2]
                hb = ax.hexbin(z0, z1, gridsize=60, cmap='Blues', mincnt=1)
                plt.colorbar(hb, ax=ax, label='counts')
                ax.set_xlabel(f'{cv0_name} (rad)')
                ax.set_ylabel(f'{cv1_name} (rad)')
                ax.set_title('Sampling histogram')
            else:
                ax = axes[1]
                hb = ax.hexbin(z0, z1, gridsize=60, cmap='Blues', mincnt=1)
                plt.colorbar(hb, ax=ax, label='counts')
                ax.set_xlabel(f'{cv0_name} (rad)')
                ax.set_ylabel(f'{cv1_name} (rad)')
                ax.set_title('Sampling histogram')

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 5 — Coupling gap time series
        # ══════════════════════════════════════════════════════════════════════
        gap_pairs = [(cvn, rc, fc) for cvn,rc,fc in cv_pairs if fc is not None]
        if gap_pairs:
            fig, axes = plt.subplots(len(gap_pairs), 1,
                                     figsize=(14, 3*len(gap_pairs)), sharex=True)
            if len(gap_pairs)==1: axes=[axes]
            fig.suptitle('Coupling Gap z - lambda Over Time (periodic-corrected)', fontsize=13)
            for ax, (cvn, rc, fc) in zip(axes, gap_pairs):
                di = next(i for i,(n,r,f) in enumerate(cv_pairs) if r==rc)
                kap = kappas[di]
                gap = periodic_gap(data[:,rc], data[:,fc], cvn)
                if kap is not None:
                    plot_gap = kap * gap
                    ideal_force = np.sqrt(kT * kap)
                    ylabel = f'kappa*(z-lambda) {cvn}\n(kJ/mol/rad)'
                    ax.axhline( ideal_force, color='orange', lw=1.2, ls='--',
                                label=f'ideal sd={ideal_force:.2f}')
                    ax.axhline(-ideal_force, color='orange', lw=1.2, ls='--')
                    ax.set_ylim(-10*ideal_force, 10*ideal_force)
                else:
                    plot_gap = gap
                    ylabel = f'z - lambda {cvn} (rad)'
                ax.plot(time, plot_gap, lw=0.4, color='mediumpurple', alpha=0.7)
                ax.axhline(0, color='k', lw=0.8)
                ax.set_ylabel(ylabel, fontsize=9)
                if kap is not None: ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
            axes[-1].set_xlabel('Time (ps)')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # PAGES 6–8 — Kernel analysis (only if kernel files supplied)
        # ══════════════════════════════════════════════════════════════════════
        if have_kernels:
            last = snapshots[-1]
            kerns = last['kernels']
            n_kern = len(kerns)

            if n_kern == 0:
                print("  WARNING: last kernel snapshot has no kernel data rows — skipping kernel panels")
            else:
                Nks      = np.array([k['Nk']     for k in kerns])
                mu_mags  = np.array([k['mu_mag'] for k in kerns])
                wts      = np.array([k['wt']     for k in kerns])
                centers  = np.array([k['center'] for k in kerns])  # (n_kern, d)
                sigmas   = np.array([k['sigma']  for k in kerns])  # (n_kern, d)

                # PAGE 6 — Kernel positions with true spatial extent
                # Each kernel is drawn as an ellipse with width=2σ₀, height=2σ₁
                # in data coordinates (radians).  This shows the actual NW
                # regression footprint of each kernel on the CV axes.
                from matplotlib.collections import EllipseCollection
                from matplotlib.cm import ScalarMappable
                from matplotlib.colors import Normalize

                if centers.shape[1] >= 2:
                    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
                    fig.suptitle(
                        f'λ-Kernel Positions (step {last["step"]}, M={n_kern})  '
                        f'— ellipses show 2σ extent in data units', fontsize=12)

                    cv0n = cv_pairs[0][0] if cv_pairs else 'z0'
                    cv1n = cv_pairs[1][0] if len(cv_pairs)>1 else 'z1'

                    # Subsample if too many kernels for readability
                    max_show = 99999
                    if n_kern > max_show:
                        idx_show = np.random.choice(n_kern, max_show, replace=False)
                    else:
                        idx_show = np.arange(n_kern)
                    c_show = centers[idx_show]
                    s_show = sigmas[idx_show]
                    mu_show = mu_mags[idx_show]
                    nk_show = Nks[idx_show]

                    # Panel 1: coloured by |μ|
                    ax = axes[0]
                    norm_mu = Normalize(vmin=0, vmax=np.percentile(mu_mags, 95))
                    ec = EllipseCollection(
                        widths=2*s_show[:,0], heights=2*s_show[:,1], angles=0,
                        units='xy',
                        offsets=np.column_stack([c_show[:,0], c_show[:,1]]),
                        transOffset=ax.transData,
                        array=mu_show, cmap='hot_r', norm=norm_mu,
                        alpha=0.4, edgecolors='none')
                    ax.add_collection(ec)
                    ax.set_xlim(centers[:,0].min()-0.2, centers[:,0].max()+0.2)
                    ax.set_ylim(centers[:,1].min()-0.2, centers[:,1].max()+0.2)
                    plt.colorbar(ec, ax=ax, label='|μ| (kJ/mol/rad)')
                    ax.set_xlabel(f'λ {cv0n}'); ax.set_ylabel(f'λ {cv1n}')
                    ax.set_title('Coloured by mean force |μ|')
                    ax.set_aspect('equal')

                    # Panel 2: coloured by log₁₀(Nk)
                    ax = axes[1]
                    norm_nk = Normalize(vmin=0, vmax=np.log10(Nks.max()+1))
                    ec2 = EllipseCollection(
                        widths=2*s_show[:,0], heights=2*s_show[:,1], angles=0,
                        units='xy',
                        offsets=np.column_stack([c_show[:,0], c_show[:,1]]),
                        transOffset=ax.transData,
                        array=np.log10(nk_show+1), cmap='viridis', norm=norm_nk,
                        alpha=0.4, edgecolors='none')
                    ax.add_collection(ec2)
                    ax.set_xlim(centers[:,0].min()-0.2, centers[:,0].max()+0.2)
                    ax.set_ylim(centers[:,1].min()-0.2, centers[:,1].max()+0.2)
                    plt.colorbar(ec2, ax=ax, label='log₁₀(Nk+1)')
                    ax.set_xlabel(f'λ {cv0n}'); ax.set_ylabel(f'λ {cv1n}')
                    ax.set_title('Coloured by log₁₀(Nk)')
                    ax.set_aspect('equal')

                elif centers.shape[1] == 1:
                    # 1D: show horizontal bars of width 2σ centered at each kernel
                    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
                    fig.suptitle(
                        f'λ-Kernel Positions (step {last["step"]})  '
                        f'— bar width = 2σ', fontsize=13)
                    ax = axes[0]
                    ax.barh(centers[:,0], width=mu_mags, left=centers[:,0]-sigmas[:,0],
                            height=2*sigmas[:,0], color='tomato', alpha=0.3, linewidth=0)
                    ax.scatter(centers[:,0], mu_mags, c=mu_mags,
                               cmap='hot_r', s=8, alpha=0.6, linewidths=0)
                    ax.set_xlabel('λ'); ax.set_ylabel('|μ|')
                    ax = axes[1]
                    ax.scatter(centers[:,0], Nks, c=np.log10(Nks+1),
                               cmap='viridis', s=8, alpha=0.6, linewidths=0)
                    ax.set_xlabel('λ'); ax.set_ylabel('Nk')

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

                # PAGE 7 — Nk, |mu|, per-kernel sigma distributions + Nk-vs-|mu|
                fig, axes = plt.subplots(2, 2, figsize=(13, 8))
                fig.suptitle(f'Kernel Statistics (step {last["step"]}, M={n_kern})', fontsize=13)

                ax = axes[0, 0]
                ax.hist(Nks, bins=50, color='steelblue', edgecolor='none', log=True)
                ax.axvline(np.median(Nks), color='r', lw=1.5, ls='--',
                           label=f'median={np.median(Nks):.1f}')
                ax.axvline(np.mean(Nks),   color='orange', lw=1.5, ls=':',
                           label=f'mean={np.mean(Nks):.1f}')
                ax.set_xlabel('Nk (samples per kernel)')
                ax.set_ylabel('Count (log)')
                ax.set_title('Nk distribution')
                ax.legend(fontsize=8)

                ax = axes[0, 1]
                ax.hist(mu_mags, bins=50, color='tomato', edgecolor='none')
                ax.axvline(np.median(mu_mags), color='k', lw=1.5, ls='--',
                           label=f'median={np.median(mu_mags):.1f}')
                ax.set_xlabel('|μ| (kJ/mol/rad)')
                ax.set_ylabel('Count')
                ax.set_title('Mean force magnitude distribution')
                ax.legend(fontsize=8)

                # Per-kernel sigma distribution (replaces old boost histogram)
                ax = axes[1, 0]
                colors_sig = ['seagreen', 'darkorange', 'royalblue']
                for d in range(sigmas.shape[1]):
                    lbl = cvnames_kern[d] if d < len(cvnames_kern) else f'dim{d}'
                    c = colors_sig[d % len(colors_sig)]
                    ax.hist(sigmas[:, d], bins=50, alpha=0.6,
                            edgecolor='none', color=c, label=f'σ({lbl})')
                    ax.axvline(np.median(sigmas[:, d]), color=c, lw=1.5, ls='--')
                ax.set_xlabel('Per-kernel σ')
                ax.set_ylabel('Count')
                ax.set_title('Kernel bandwidth distribution')
                ax.legend(fontsize=8)

                ax = axes[1, 1]
                ax.scatter(Nks, mu_mags, alpha=0.3, s=8, c='purple')
                ax.set_xlabel('Nk')
                ax.set_ylabel('|μ| (kJ/mol/rad)')
                ax.set_title('Nk vs |μ|: does confidence correlate with force?')
                ax.set_xscale('log')

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

                # PAGE 8 — Kernel evolution across snapshots
                if len(snapshots) > 1:
                    snap_steps  = [s['step']    for s in snapshots if s['M']>0]
                    snap_M      = [s['M']        for s in snapshots if s['M']>0]
                    snap_neff   = [s['neff']     for s in snapshots
                                   if s['M']>0 and s['neff'] is not None]
                    snap_sigma  = [s['sigma'][0] for s in snapshots
                                   if s['M']>0 and s['sigma']]
                    snap_vamp   = [s['vamp']     for s in snapshots
                                   if s['M']>0 and s['vamp'] is not None]

                    # Wamp time series: interpolate from COLVAR wamp column at
                    # the kernel snapshot steps (wamp is per-step in COLVAR).
                    snap_wamp = []
                    if wamp_col is not None and time_col is not None:
                        colvar_steps = data[:, time_col]
                        colvar_wamp  = data[:, wamp_col]
                        for step in snap_steps:
                            # Find the COLVAR row closest to this step
                            idx = np.argmin(np.abs(colvar_steps - step * args.dt))
                            snap_wamp.append(colvar_wamp[idx])

                    # Expand to 2×3 when flooding diagnostics are available,
                    # so each quantity gets its own panel and is easy to read.
                    have_flood_cols = (vbias_col is not None or wamp_col is not None)
                    ncols = 3 if have_flood_cols else 2
                    fig, axes = plt.subplots(2, ncols, figsize=(6*ncols, 8))
                    fig.suptitle('Kernel Snapshot Evolution', fontsize=13)

                    def splot(ax, steps, vals, label, color):
                        if vals:
                            ax.plot(steps[:len(vals)], vals, 'o-', lw=1.2,
                                    ms=5, color=color, label=label)

                    splot(axes[0,0], snap_steps, snap_M,     'M (λ-kernels)', 'steelblue')
                    axes[0,0].set_ylabel('M'); axes[0,0].set_title('Kernel count')
                    axes[0,0].grid(True, alpha=0.3)

                    splot(axes[0,1], snap_steps[:len(snap_neff)], snap_neff,
                          'neff (Silverman)', 'royalblue')
                    axes[0,1].set_ylabel('neff'); axes[0,1].set_title('Effective kernel count')
                    axes[0,1].grid(True, alpha=0.3)

                    splot(axes[1,0], snap_steps[:len(snap_sigma)], snap_sigma,
                          'σ dim 0', 'purple')
                    axes[1,0].set_ylabel('σ'); axes[1,0].set_xlabel('Time (ps)')
                    axes[1,0].set_title('Silverman bandwidth')
                    axes[1,0].grid(True, alpha=0.3)

                    # Panel [1,1]: V(λ) at s_fict and V_ex — same kJ/mol scale.
                    ax = axes[1,1]
                    if vbias_col is not None:
                        ax.plot(time, data[:, vbias_col], lw=0.8,
                                color='darkorange', alpha=0.85,
                                label='V(λ) at s_fict')
                    if wamp_ts is not None:
                        ax.plot(time, wamp_ts, lw=0.8,
                                color='royalblue', alpha=0.85,
                                label='V_ex')
                    ax.set_ylabel('Bias potential at s_fict (kJ/mol)')
                    ax.set_xlabel('Time (ps)')
                    ax.set_title('V(λ) and V_ex')
                    ax.legend(fontsize=7, loc='upper left')
                    ax.grid(True, alpha=0.3)

                    if ncols == 3:
                        # Panel [0,2]: Vamp (grid amplitude) for long-range convergence view.
                        ax = axes[0,2]
                        if vamp_col is not None:
                            ax.plot(time, data[:, vamp_col], lw=0.8,
                                    color='darkorange', alpha=0.85,
                                    label='Vamp = max(V)−min(V)')
                        ax.set_ylabel('Grid amplitude (kJ/mol)')
                        ax.set_title('V(λ) grid amplitude Vamp')
                        ax.legend(fontsize=7)
                        ax.grid(True, alpha=0.3)

                        # Panel [1,2]: wamp/V_ex ratio to Vamp for convergence assessment.
                        ax = axes[1,2]
                        if wamp_ts is not None and vamp_col is not None:
                            vamp_ts = data[:, vamp_col]
                            safe_vamp = np.where(vamp_ts > 0.1, vamp_ts, np.nan)
                            ratio_ts = wamp_ts / safe_vamp
                            ax.plot(time, ratio_ts, lw=0.6, color='teal', alpha=0.85,
                                    label='V_ex / Vamp')
                        ax.set_ylabel('V_ex / Vamp')
                        ax.set_xlabel('Time (ps)')
                        ax.set_title('Exploration/ABF ratio (sweet spot: 0.3–1.0)')
                        ax.legend(fontsize=7, loc='upper right')
                        ax.grid(True, alpha=0.3)

                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # Summary page
        # ══════════════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        lines = [
            'FKERNELABF v3+ Analysis Summary',
            '─' * 50,
            f'COLVAR file:       {args.colvar}',
            f'Frames loaded:     {len(data)}',
            f'Simulation time:   {time[-1]:.1f} ps',
            f'Temperature:       {args.temp} K  (kT = {kT:.4f} kJ/mol)',
            f'CVs:               {[p[0] for p in cv_pairs]}',
            f'κ:                 {kappas}',
            f'BIASFACTOR (γ):    {args.biasfactor}'
            + (' [density-based exploration on λ]' if args.biasfactor > 1 else ' [pure ABF]'),
            '',
        ]
        coupling_verdicts = []
        for di, (cvn, rc, fc) in enumerate(cv_pairs):
            z = data[:, rc]
            lines.append(f'  {cvn}: mean={np.mean(z):.4f}  std={np.std(z):.4f}  '
                         f'range=[{z.min():.4f}, {z.max():.4f}]')
            if fc is not None:
                lam = data[:, fc]
                gap = periodic_gap(z, lam, cvn)
                kap = kappas[di]
                if kap is not None:
                    ideal = np.sqrt(kT / kap)
                    ratio = np.std(gap) / ideal if ideal > 0 else float('inf')
                    if ratio < 3:
                        verdict = 'GOOD (< 3x ideal)'
                    elif ratio < 8:
                        verdict = f'ACCEPTABLE ({ratio:.1f}x ideal)'
                    else:
                        verdict = f'WARNING: poor coupling ({ratio:.1f}x ideal) -- consider higher KAPPA or lower TAU'
                    coupling_verdicts.append((cvn, ratio, verdict))
                    lines.append(f'  {cvn} coupling gap: std={np.std(gap):.5f} rad  '
                                 f'({ratio:.1f}x ideal={ideal:.5f})  [{verdict}]')
                else:
                    lines.append(f'  {cvn} coupling gap (rad): mean={np.mean(gap):.5f}  '
                                 f'std={np.std(gap):.5f}')
        lines += ['']
        if vamp_col is not None:
            lines.append(f'Final Vamp:           {data[-1, vamp_col]:.2f} kJ/mol  [V(λ) grid amplitude]')
        if vbias_col is not None:
            lines.append(f'Final V(λ) at s_fict: {data[-1, vbias_col]:.2f} kJ/mol')
        if wamp_ts is not None:
            lines.append(f'Final V_ex:           {wamp_ts[-1]:.2f} kJ/mol')
        if nker_col is not None:
            lines.append(f'Final M:              {int(data[-1, nker_col])}')
        if nzker_col is not None:
            lines.append(f'Final zM:             {int(data[-1, nzker_col])}')
        if sig_col is not None:
            lines.append(f'Final σ (dim 0):      {data[-1, sig_col]:.5f}')
        if have_kernels:
            lines.append(f'Kernel snapshots:     {len(snapshots)}')

        ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    if not args.show:
        print(f"Report saved: {pdf_path}")
    else:
        plt.show()


# ─────────────────────── main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Analysis report for FKERNELABF v3+ simulations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--colvar',      required=True, help='PLUMED COLVAR file')
    parser.add_argument('--kernels',     default=None,
                        help='Kernel dump: single file, glob pattern '
                             '(e.g. "fk.kernels_*.dat"), or directory')
    parser.add_argument('--output',      default='fkabf_report.pdf', help='Output PDF')
    parser.add_argument('--stride',      type=int,   default=1,    help='Row stride for COLVAR')
    parser.add_argument('--kappa',       type=float, nargs='+',    help='κ per CV (kJ/mol/rad²)')
    parser.add_argument('--temp',        type=float, default=300,  help='Temperature (K)')
    parser.add_argument('--cvnames',     nargs='+',               help='CV display names')
    parser.add_argument('--biasfactor',  type=float, default=1.0,
                        help='BIASFACTOR γ used in the simulation (default 1.0 = pure ABF). '
                             'If γ > 1, activates exploration diagnostics (V_ex panel).')
    parser.add_argument('--dt',          type=float, default=0.002,help='MD timestep (ps)')
    parser.add_argument('--show',        action='store_true',      help='Show interactively')
    args = parser.parse_args()

    try:
        import matplotlib
    except ImportError as e:
        sys.exit(f"ERROR: {e}\nInstall with: pip install matplotlib")

    print(f"Reading COLVAR: {args.colvar}", flush=True)
    fields, data, periodic_info = parse_colvar(args.colvar, stride=args.stride)

    snapshots = []
    cvnames_kern = []
    if args.kernels:
        print(f"Reading kernels: {args.kernels}", flush=True)
        snapshots, cvnames_kern = load_kernel_snapshots(args.kernels)

    print("Generating report ...", flush=True)
    make_report(fields, data, snapshots, cvnames_kern, args, periodic_info)


if __name__ == '__main__':
    main()
