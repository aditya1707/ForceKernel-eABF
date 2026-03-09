
























import argparse, os, sys
import numpy as np



_A  = np.array([-200., -100., -170.,  15.])
_a  = np.array([  -1.,   -1.,  -6.5,   0.7])
_b  = np.array([   0.,    0.,  11.,    0.6])
_c  = np.array([ -10.,  -10.,  -6.5,   0.7])
_x0 = np.array([   1.,    0.,  -0.5,  -1. ])
_y0 = np.array([   0.,    0.5,  1.5,   1. ])
MB_SCALE = 0.10

def mb_V(x, y):
    return sum(_A[i]*np.exp(_a[i]*(x-_x0[i])**2
               +_b[i]*(x-_x0[i])*(y-_y0[i])
               +_c[i]*(y-_y0[i])**2) for i in range(4)) * MB_SCALE

MINIMA = {
    'A': (0.6235,  0.0280),
    'B': (-0.5582, 1.4417),
    'C': (-0.0500, 0.4667),
}


def compute_reference_fel(kT, nx=200, ny=200):
    
    xs = np.linspace(-1.6, 1.3, nx)
    ys = np.linspace(-0.3, 2.1, ny)
    X, Y = np.meshgrid(xs, ys)
    V = np.vectorize(mb_V)(X, Y)
    V_clip = np.where(V < 50, V, 50)
    p = np.exp(-V_clip / kT)
    FEL = -kT * np.log(np.where(p > 0, p / p.max(), np.nan))
    return xs, ys, FEL




def run_czar(czar_file, grid_size, minpop, kT):
    



    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import czar_integrate as ci
    except ImportError:
        print("WARNING: czar_integrate.py not importable — skipping CZAR overlay")
        return None

    
    meta, kernels = ci.parse_czar_file(czar_file)
    if not kernels:
        print("WARNING: no kernels parsed"); return None

    dim = meta.get('dim', 0)
    if dim != 2:
        print(f"WARNING: expected 2D, got {dim}D"); return None

    
    grid_ranges = [(-1.6, 1.3), (-0.3, 2.1)]
    coords = [np.linspace(lo, hi, grid_size) for (lo, hi) in grid_ranges]
    periodic = np.array([False, False])
    nsigma = 4.0

    ptilde, czar_grad = ci.czar_on_grid(coords, periodic, kernels, kT, nsigma)

    
    mask = ptilde >= minpop * ptilde.max()

    
    A = ci.poisson_integrate(czar_grad, coords, periodic)
    A = np.where(mask, A, np.nan)
    A -= np.nanmin(A)

    
    
    return coords[0], coords[1], A.T, mask.T




def reweighted_fel_from_colvar(colvar_path, kT, nx=80, ny=80):
    



    try:
        fields, data, _ = _parse_colvar_simple(colvar_path)
    except Exception as e:
        print(f"WARNING: could not parse COLVAR: {e}"); return None

    
    def fc(pat):
        for i, f in enumerate(fields):
            if pat.lower() in f.lower(): return i
        return None

    ix = fc('cv.x'); iy = fc('cv.y'); ib = fc('bias')
    if ix is None or iy is None:
        print("WARNING: could not find cv.x / cv.y in COLVAR"); return None

    x = data[:, ix]; y = data[:, iy]

    xs = np.linspace(-1.6, 1.3, nx)
    ys = np.linspace(-0.3, 2.1, ny)
    bw = 0.08   

    if ib is not None:
        V = data[:, ib]
        logw = np.clip(V / kT, -50, 50)
        weights = np.exp(logw); weights /= weights.sum()
    else:
        weights = np.ones(len(x)) / len(x)

    
    G0, G1 = np.meshgrid(xs, ys)
    grid_pts = np.vstack([G0.ravel(), G1.ravel()])
    inv2bw2 = 1.0 / (2.0 * bw**2)
    rw_dens = np.zeros(nx * ny)
    chunk = 5000
    for s in range(0, len(x), chunk):
        d0 = grid_pts[0][:, None] - x[None, s:s+chunk]
        d1 = grid_pts[1][:, None] - y[None, s:s+chunk]
        rw_dens += np.exp(-(d0**2 + d1**2) * inv2bw2) @ weights[s:s+chunk]
    rw_dens = rw_dens.reshape(ny, nx)
    rw_dens = np.where(rw_dens > 0, rw_dens, np.nan)
    FEL = -kT * np.log(rw_dens / np.nanmax(rw_dens))
    return xs, ys, FEL


def _parse_colvar_simple(path, stride=5):
    fields = None; rows = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if line.startswith('#! FIELDS'):
                fields = line.split()[2:]; continue
            if line.startswith('#'): continue
            try: vals = [float(v) for v in line.split()]
            except: continue
            if fields and len(vals)==len(fields) and i % stride == 0:
                rows.append(vals)
    return fields, np.array(rows), {}




def extract_minimum_values(xs, ys, A_grid, mask=None):
    
    vals = {}
    for name, (mx, my) in MINIMA.items():
        ix = np.argmin(np.abs(xs - mx))
        iy = np.argmin(np.abs(ys - my))
        if mask is not None and not mask[iy, ix]:
            
            found = False
            for r in range(1, 10):
                for di in range(-r, r+1):
                    for dj in range(-r, r+1):
                        ii = iy+di; jj = ix+dj
                        if 0<=ii<len(ys) and 0<=jj<len(xs):
                            if mask[ii, jj]:
                                vals[name] = A_grid[ii, jj]; found=True; break
                    if found: break
                if found: break
            if not found: vals[name] = np.nan
        else:
            vals[name] = A_grid[iy, ix]
    return vals


def print_delta_A_table(label, vals, kT, ref_vals=None):
    names = list(MINIMA.keys())
    print(f"\n{label}")
    print(f"  {'Min':4s}  {'A (kJ/mol)':12s}  {'A/kT':6s}", end='')
    if ref_vals:
        print(f"  {'err vs ref (kJ/mol)':22s}  {'err/kT':8s}", end='')
    print()
    print("  " + "─"*60)
    for n in names:
        v = vals.get(n, np.nan)
        line = f"  {n:4s}  {v:12.4f}  {v/kT:6.3f}"
        if ref_vals:
            rv = ref_vals.get(n, np.nan)
            err = v - rv
            line += f"  {err:+22.4f}  {err/kT:+8.3f}"
        print(line)
    print()
    print(f"  ΔA differences:")
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if j <= i: continue
            dA_est = vals.get(nj, np.nan) - vals.get(ni, np.nan)
            line = f"    A({nj}) - A({ni}) = {dA_est:+.4f} kJ/mol = {dA_est/kT:+.3f} kT"
            if ref_vals:
                dA_ref = ref_vals.get(nj, np.nan) - ref_vals.get(ni, np.nan)
                line += f"   [ref: {dA_ref:+.4f}, err: {dA_est-dA_ref:+.4f}]"
            print(line)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--czar',    default='mb_czar_kernels_00200000.dat')
    parser.add_argument('--grid',    type=int,   default=80)
    parser.add_argument('--minpop',  type=float, default=1e-5)
    parser.add_argument('--temp',    type=float, default=300.0)
    parser.add_argument('--colvar',  default='MB_COLVAR')
    args = parser.parse_args()

    kB  = 0.008314462618
    kT  = kB * args.temp
    print(f"Müller-Brown FKERNELABF validation")
    print(f"  kT = {kT:.4f} kJ/mol  (T = {args.temp} K)")

    
    print("\nComputing reference FEL (exact Boltzmann)...")
    xs_ref, ys_ref, A_ref = compute_reference_fel(kT, nx=200, ny=200)
    A_ref -= np.nanmin(A_ref)
    ref_vals = extract_minimum_values(xs_ref, ys_ref, A_ref)
    print_delta_A_table("Reference (exact Boltzmann)", ref_vals, kT)

    
    czar_result = None
    if os.path.exists(args.czar):
        print(f"Running CZAR integration on {args.czar}...")
        czar_result = run_czar(args.czar, args.grid, args.minpop, kT)
        if czar_result:
            xs_c, ys_c, A_czar, mask_c = czar_result
            czar_vals = extract_minimum_values(xs_c, ys_c, A_czar, mask_c)
            print_delta_A_table("CZAR estimate", czar_vals, kT, ref_vals)
        else:
            print("  CZAR integration failed.")
    else:
        print(f"  {args.czar} not found — skipping CZAR")

    
    rw_result = None
    if os.path.exists(args.colvar):
        print(f"Computing reweighted FEL from {args.colvar}...")
        rw_result = reweighted_fel_from_colvar(args.colvar, kT)
        if rw_result:
            xs_rw, ys_rw, A_rw = rw_result
            rw_result_aligned = A_rw - np.nanmin(A_rw)
            rw_vals = extract_minimum_values(xs_rw, ys_rw, rw_result_aligned)
            print_delta_A_table("Reweighted FEL (exp(+V/kT))", rw_vals, kT, ref_vals)
    else:
        print(f"  {args.colvar} not found — skipping reweighted FEL")

    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        n_panels = 1 + (1 if czar_result else 0) + (1 if rw_result else 0) + (1 if czar_result else 0)
        fig, axes = plt.subplots(2, 2, figsize=(13, 11))
        axes = axes.flat
        fig.suptitle('Müller-Brown: FKERNELABF vs Reference', fontsize=14)

        cmap = 'viridis'
        vmax = 15.0
        levels = np.linspace(0, vmax, 22)
        X_ref, Y_ref = np.meshgrid(xs_ref, ys_ref)
        A_ref_plot = np.where(A_ref < vmax*1.5, A_ref, np.nan)

        def mark_minima(ax):
            for name, (mx, my) in MINIMA.items():
                ax.plot(mx, my, 'r*', ms=14, mec='k', mew=0.5)
                ax.annotate(name, (mx+0.04, my+0.06), color='red',
                            fontsize=11, fontweight='bold')

        
        ax = next(axes)
        cs = ax.contourf(X_ref, Y_ref, A_ref_plot, levels=levels, cmap=cmap, extend='max')
        ax.contour(X_ref, Y_ref, A_ref_plot, levels=levels[::4],
                   colors='k', linewidths=0.5, alpha=0.6)
        plt.colorbar(cs, ax=ax, label='A (kJ/mol)')
        mark_minima(ax)
        ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')
        ax.set_title('Reference FEL (exact Boltzmann)')

        
        ax = next(axes)
        if czar_result:
            xs_c, ys_c, A_czar, mask_c = czar_result
            Xc, Yc = np.meshgrid(xs_c, ys_c)
            cs2 = ax.contourf(Xc, Yc, np.where(mask_c, A_czar, np.nan),
                              levels=np.linspace(0,40,20), cmap=cmap, extend='max')
            ax.contour(Xc, Yc, np.where(mask_c, A_czar, np.nan),
                       levels=np.linspace(0,40,20), colors='k', linewidths=0.5, alpha=0.6)
            plt.colorbar(cs2, ax=ax, label='A (kJ/mol)')
            mark_minima(ax)
            ax.set_title(f'CZAR FEL (grid={args.grid}²)')
        else:
            ax.text(0.5, 0.5, 'CZAR result\nnot available',
                    ha='center', va='center', transform=ax.transAxes, color='grey')
            ax.set_title('CZAR FEL')
        ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')

        
        ax = next(axes)
        if czar_result:
            from scipy.interpolate import RegularGridInterpolator
            
            czar_interp = RegularGridInterpolator(
                (ys_c, xs_c), np.where(mask_c, A_czar, np.nan),
                method='linear', bounds_error=False, fill_value=np.nan)
            
            
            pts = np.column_stack([Y_ref.ravel(), X_ref.ravel()])
            A_czar_on_ref = czar_interp(pts).reshape(X_ref.shape)
            diff = A_czar_on_ref - A_ref_plot
            diff_finite = diff[np.isfinite(diff)]
            dlim = min(5.0, float(np.nanpercentile(np.abs(diff_finite), 95))) if len(diff_finite) else 1.0
            cs3 = ax.contourf(X_ref, Y_ref, diff,
                              levels=np.linspace(-dlim, dlim, 22),
                              cmap='bwr', extend='both')
            plt.colorbar(cs3, ax=ax, label='CZAR - Reference (kJ/mol)')
            rms = float(np.sqrt(np.nanmean(diff**2)))
            ax.set_title(f'CZAR − Reference  (RMS={rms:.3f} kJ/mol)')
        else:
            ax.text(0.5, 0.5, 'CZAR result\nnot available',
                    ha='center', va='center', transform=ax.transAxes, color='grey')
            ax.set_title('CZAR − Reference')
        ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')

        
        ax = next(axes)
        if rw_result:
            xs_rw, ys_rw, A_rw_raw = rw_result
            A_rw_al = A_rw_raw - np.nanmin(A_rw_raw)
            Xrw, Yrw = np.meshgrid(xs_rw, ys_rw)
            cs4 = ax.contourf(Xrw, Yrw, np.where(A_rw_al < vmax*1.5, A_rw_al, np.nan),
                              levels=levels, cmap=cmap, extend='max')
            plt.colorbar(cs4, ax=ax, label='A (kJ/mol)')
            mark_minima(ax)
            ax.set_title('Reweighted FEL (exp(+V/kT))')
        else:
            ax.text(0.5, 0.5, 'Reweighted FEL\nnot available',
                    ha='center', va='center', transform=ax.transAxes, color='grey')
            ax.set_title('Reweighted FEL')
        ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')

        plt.tight_layout()
        plt.savefig('mb_validation.png', dpi=150)
        print("\nValidation plot saved: mb_validation.png")
        plt.close()

    except Exception as e:
        print(f"WARNING: plotting failed: {e}")
        import traceback; traceback.print_exc()

    
    with open('mb_validation.txt', 'w') as f:
        f.write("Müller-Brown FKERNELABF Validation\n")
        f.write("="*60 + "\n\n")
        f.write("Reference minimum locations and FEL values:\n")
        for name, (mx, my) in MINIMA.items():
            ix = np.argmin(np.abs(xs_ref - mx)); iy = np.argmin(np.abs(ys_ref - my))
            f.write(f"  Min {name}: ({mx:.4f}, {my:.4f})  A_ref = {A_ref[iy,ix]:.4f} kJ/mol\n")
        f.write("\nTarget ΔA values:\n")
        for ni, nj in [('A','B'), ('A','C'), ('B','C')]:
            vni = ref_vals[ni]; vnj = ref_vals[nj]
            f.write(f"  A({nj}) - A({ni}) = {vnj-vni:+.4f} kJ/mol = {(vnj-vni)/kT:+.4f} kT\n")
    print("Validation table saved: mb_validation.txt")


if __name__ == '__main__':
    main()
