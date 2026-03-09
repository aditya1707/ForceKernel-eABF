































































import argparse
import sys
import numpy as np




def parse_czar_file(path):
    







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
                
                if dim is None:
                    continue
                vals = [float(x) for x in parts]
                if len(vals) != 1 + 3 * dim:
                    continue   
                Nk    = vals[0]
                center= np.array(vals[1        : 1+dim])
                mu    = np.array(vals[1+dim    : 1+2*dim])
                sigma = np.array(vals[1+2*dim  : 1+3*dim])
                kernels.append({'Nk': Nk, 'center': center, 'mu': mu, 'sigma': sigma})

    if not kernels:
        sys.exit(f"ERROR: no kernel data found in {path}")
    if 'kT' not in meta:
        sys.exit("ERROR: kT not found in CZAR file header")

    
    for key in ('kT', 'kappa', 'periodic', 'domMin', 'domMax'):
        if key not in meta:
            print(f"WARNING: '{key}' not found in CZAR file — using defaults may be unsafe.")

    return meta, kernels




def build_grid(meta, args):
    




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
            
            
            pts = np.linspace(gmin[d], gmax[d], args.grid, endpoint=False)
        else:
            pts = np.linspace(gmin[d], gmax[d], args.grid)
        coords.append(pts)

    return coords, periodic, gmin, gmax




def periodic_delta(delta, period):
    
    return delta - period * np.round(delta / period)


def czar_on_grid(coords, periodic, kernels, kT, nsigma, verbose=False):
    















    dim = len(coords)
    shape = tuple(len(c) for c in coords)
    period = np.zeros(dim)
    for d in range(dim):
        if periodic[d]:
            period[d] = coords[d][-1] - coords[d][0]
            
            period[d] += coords[d][1] - coords[d][0]

    
    mg = np.meshgrid(*coords, indexing='ij')   
    z_grid = np.stack(mg, axis=-1)             

    ptilde    = np.zeros(shape)
    sum_wkmu  = np.zeros((*shape, dim))   
    
    sum_wkdz  = np.zeros((*shape, dim))

    n_total = len(kernels)
    report_every = max(1, n_total // 10)

    for ki, kern in enumerate(kernels):
        if verbose and ki % report_every == 0:
            print(f"  Processing kernel {ki+1}/{n_total} ...", flush=True)

        Nk     = kern['Nk']
        center = kern['center']
        mu_k   = kern['mu']
        sigma  = kern['sigma']

        
        dz = z_grid - center   
        
        for d in range(dim):
            if periodic[d] and period[d] > 0:
                dz[..., d] = periodic_delta(dz[..., d], period[d])

        
        inv4sig2 = 1.0 / (4.0 * sigma**2 + 1e-300)   
        exponent = np.einsum('...d,d->...', dz**2, inv4sig2)  

        
        mask = exponent < nsigma**2
        if not np.any(mask):
            continue

        Gk = np.zeros(shape)
        Gk[mask] = np.exp(-exponent[mask])

        norm = 1.0 / np.prod(sigma)
        wk = Nk * norm * Gk  

        ptilde   += wk
        sum_wkmu += wk[..., np.newaxis] * mu_k   

        
        inv2sig2 = 1.0 / (2.0 * sigma**2 + 1e-300)
        dz_scaled = dz * inv2sig2   
        sum_wkdz += wk[..., np.newaxis] * dz_scaled

    
    
    
    
    
    
    
    
    
    
    
    
    
    

    safe_ptilde = np.where(ptilde > 0, ptilde, 1.0)  
    czar_grad = np.zeros((*shape, dim))
    for d in range(dim):
        czar_grad[..., d] = (-sum_wkmu[..., d] - kT * sum_wkdz[..., d]) / safe_ptilde

    return ptilde, czar_grad




def integrate_1d(grad, dx, periodic):
    
    from numpy import cumsum
    A = np.zeros_like(grad)
    A[1:] = np.cumsum(0.5 * (grad[:-1] + grad[1:]) * dx)
    if periodic:
        
        N = len(A)
        drift = A[-1] / (N - 1)
        A -= np.arange(N) * drift
    return A




def poisson_integrate(czar_grad, coords, periodic):
    
















    dim = len(coords)
    shape = tuple(len(c) for c in coords)
    dx    = np.array([c[1]-c[0] for c in coords])

    
    div = np.zeros(shape)
    for d in range(dim):
        g = czar_grad[..., d]
        if periodic[d]:
            
            gp = np.roll(g, -1, axis=d)
            gm = np.roll(g, +1, axis=d)
            div += (gp - gm) / (2.0 * dx[d])
        else:
            
            slc_p = [slice(None)] * dim
            slc_m = [slice(None)] * dim
            slc   = [slice(None)] * dim

            
            slc[d]  = slice(1, -1)
            slc_p[d]= slice(2, None)
            slc_m[d]= slice(None, -2)
            div[tuple(slc)] += (g[tuple(slc_p)] - g[tuple(slc_m)]) / (2.0 * dx[d])

            
            slc_lb    = [slice(None)] * dim; slc_lb[d]  = 0
            slc_lb1   = [slice(None)] * dim; slc_lb1[d] = 1
            div[tuple(slc_lb)] += (g[tuple(slc_lb1)] - g[tuple(slc_lb)]) / dx[d]

            
            slc_rb    = [slice(None)] * dim; slc_rb[d]  = -1
            slc_rb1   = [slice(None)] * dim; slc_rb1[d] = -2
            div[tuple(slc_rb)] += (g[tuple(slc_rb)] - g[tuple(slc_rb1)]) / dx[d]

    
    div -= div.mean()

    
    ext_shape = tuple(
        shape[d] if periodic[d] else (2*(shape[d]-1) if shape[d]>1 else 1)
        for d in range(dim)
    )

    
    div_ext = np.zeros(ext_shape)
    
    orig_slices = tuple(slice(0, shape[d]) for d in range(dim))
    div_ext[orig_slices] = div

    for d in range(dim):
        if not periodic[d] and shape[d] > 1:
            
            N = shape[d]
            ext_N = ext_shape[d]
            for j in range(N, ext_N):
                j_orig = 2*(N-1) - j
                src = [slice(None)] * dim
                src[d] = j_orig
                dst = [slice(None)] * dim
                dst[d] = j
                div_ext[tuple(dst)] = div_ext[tuple(src)]

    
    Fd = np.fft.fftn(div_ext)

    
    freq_arrays = []
    for d in range(dim):
        Nd = ext_shape[d]
        ks = np.fft.fftfreq(Nd, d=1.0/Nd).astype(int)  
        eig_d = (2.0*np.cos(2.0*np.pi*ks/Nd) - 2.0) / dx[d]**2
        freq_arrays.append(eig_d)

    
    eigenvals = np.zeros(ext_shape)
    for d in range(dim):
        shape_broadcast = [1]*dim
        shape_broadcast[d] = ext_shape[d]
        eigenvals += freq_arrays[d].reshape(shape_broadcast)

    
    with np.errstate(divide='ignore', invalid='ignore'):
        Fd_sol = np.where(eigenvals != 0, Fd / eigenvals, 0.0)

    A_ext = np.fft.ifftn(Fd_sol).real

    
    A = A_ext[orig_slices]

    
    A -= A.min()

    return A




def write_output(path, coords, periodic, ptilde, czar_grad, A, minpop, kT):
    
    dim = len(coords)
    shape = tuple(len(c) for c in coords)

    ptilde_max = ptilde.max()
    pop_threshold = minpop * ptilde_max if ptilde_max > 0 else 0.0

    
    header_parts = ['#']
    for d in range(dim):
        header_parts.append(f'z{d}')
    for d in range(dim):
        header_parts.append(f'czar_grad{d}')
    header_parts += ['ptilde', 'A_czar[kJ/mol]']
    header = ' '.join(header_parts)

    lines = [header]
    lines.append(f'# kT = {kT:.6f} kJ/mol')
    lines.append(f'# Grid shape: {shape}')
    lines.append(f'# minpop threshold: {pop_threshold:.4g} ({minpop} * max(ptilde))')
    lines.append(f'# A_czar set to NaN where ptilde < threshold')
    lines.append('#')

    it = np.ndindex(*shape)
    for idx in it:
        row = []
        for d in range(dim):
            row.append(f'{coords[d][idx[d]]:.8f}')
        for d in range(dim):
            row.append(f'{czar_grad[idx][d]:.8f}')
        row.append(f'{ptilde[idx]:.6e}')
        if ptilde[idx] >= pop_threshold:
            row.append(f'{A[idx]:.8f}')
        else:
            row.append('nan')
        lines.append(' '.join(row))

        
        if dim >= 2:
            
            inner_idx = idx[1] if dim >= 2 else idx[0]
            if inner_idx == shape[1] - 1 and dim == 2:
                lines.append('')
            elif dim == 3 and idx[2] == shape[2]-1 and idx[1] == shape[1]-1:
                lines.append('')

    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')





def wls_integrate_2d(
    czar_grad,
    coords,
    periodic,
    ptilde=None,
    minpop=1e-5,
    weight_exp=2.0,
    tikhonov=1e-4,
    cg_tol=1e-10,
    cg_maxiter=20000,
):
    



















    import numpy as _np
    from scipy import sparse as _sp
    from scipy.sparse.linalg import cg as _cg

    x, y = coords
    nx, ny = len(x), len(y)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    gx = czar_grad[..., 0].astype(_np.float64, copy=False)
    gy = czar_grad[..., 1].astype(_np.float64, copy=False)

    
    if ptilde is None:
        w_node = _np.ones((nx, ny), dtype=_np.float64)
    else:
        pmax = float(_np.max(ptilde))
        if pmax <= 0:
            w_node = _np.ones((nx, ny), dtype=_np.float64)
        else:
            
            p_thresh = minpop * pmax
            w_node = ptilde / (ptilde + p_thresh + 1e-300)

    def idx(i, j):
        return i * ny + j

    N = nx * ny

    
    if periodic[0]:
        n_xe = nx * ny
        ie = _np.repeat(_np.arange(nx), ny)
        je = _np.tile(_np.arange(ny), nx)
        iR = (ie + 1) % nx
    else:
        n_xe = (nx - 1) * ny
        ie = _np.repeat(_np.arange(nx - 1), ny)
        je = _np.tile(_np.arange(ny), nx - 1)
        iR = ie + 1

    row = _np.arange(n_xe, dtype=_np.int64)
    colL = _np.fromiter((idx(int(i), int(j)) for i, j in zip(ie, je)), dtype=_np.int64)
    colR = _np.fromiter((idx(int(i), int(j)) for i, j in zip(iR, je)), dtype=_np.int64)

    data = _np.concatenate([_np.full(n_xe, -1.0/dx), _np.full(n_xe, 1.0/dx)])
    rows = _np.concatenate([row, row])
    cols = _np.concatenate([colL, colR])
    Dx = _sp.csr_matrix((data, (rows, cols)), shape=(n_xe, N))

    wx = 0.5 * (w_node[ie, je] + w_node[iR, je])
    fx = 0.5 * (gx[ie, je] + gx[iR, je])
    wx_flat = _np.power(wx, weight_exp).astype(_np.float64, copy=False)
    fx_flat = fx.astype(_np.float64, copy=False)

    
    if periodic[1]:
        n_ye = nx * ny
        iy = _np.repeat(_np.arange(nx), ny)
        jy = _np.tile(_np.arange(ny), nx)
        jR = (jy + 1) % ny
    else:
        n_ye = nx * (ny - 1)
        iy = _np.repeat(_np.arange(nx), ny - 1)
        jy = _np.tile(_np.arange(ny - 1), nx)
        jR = jy + 1

    row = _np.arange(n_ye, dtype=_np.int64)
    colL = _np.fromiter((idx(int(i), int(j)) for i, j in zip(iy, jy)), dtype=_np.int64)
    colR = _np.fromiter((idx(int(i), int(j)) for i, j in zip(iy, jR)), dtype=_np.int64)

    data = _np.concatenate([_np.full(n_ye, -1.0/dy), _np.full(n_ye, 1.0/dy)])
    rows = _np.concatenate([row, row])
    cols = _np.concatenate([colL, colR])
    Dy = _sp.csr_matrix((data, (rows, cols)), shape=(n_ye, N))

    wy = 0.5 * (w_node[iy, jy] + w_node[iy, jR])
    fy = 0.5 * (gy[iy, jy] + gy[iy, jR])
    wy_flat = _np.power(wy, weight_exp).astype(_np.float64, copy=False)
    fy_flat = fy.astype(_np.float64, copy=False)

    Wx = _sp.diags(wx_flat)
    Wy = _sp.diags(wy_flat)

    LHS = Dx.T @ Wx @ Dx + Dy.T @ Wy @ Dy
    RHS = Dx.T @ (wx_flat * fx_flat) + Dy.T @ (wy_flat * fy_flat)

    
    if tikhonov > 0:
        rows_L, cols_L, vals_L = [], [], []
        diag_main = _np.zeros(N, dtype=_np.float64)

        for i in range(nx):
            for j in range(ny):
                k = idx(i, j)
                
                neigh = []
                if periodic[0] or i > 0:      neigh.append(((i - 1) % nx, j, 1.0/dx**2))
                if periodic[0] or i < nx - 1: neigh.append(((i + 1) % nx, j, 1.0/dx**2))
                if periodic[1] or j > 0:      neigh.append((i, (j - 1) % ny, 1.0/dy**2))
                if periodic[1] or j < ny - 1: neigh.append((i, (j + 1) % ny, 1.0/dy**2))

                for (ii, jj, w) in neigh:
                    kk = idx(ii, jj)
                    rows_L.append(k); cols_L.append(kk); vals_L.append(w)
                    diag_main[k] -= w

        Lap = _sp.csr_matrix(
            (_np.array(vals_L + list(diag_main)),
             (_np.array(rows_L + list(range(N))),
              _np.array(cols_L + list(range(N))))),
            shape=(N, N)
        )
        LHS = LHS + tikhonov * (Lap.T @ Lap)

    
    pin = int(_np.argmax(w_node.ravel()))
    pin_penalty = float(LHS.diagonal().max()) * 1e4
    LHS = LHS.tolil()
    LHS[pin, pin] += pin_penalty
    LHS = LHS.tocsr()

    diag = LHS.diagonal().copy()
    diag[diag < 1e-300] = 1.0
    M_pre = _sp.diags(1.0 / diag)

    A_flat, info = _cg(LHS, RHS, tol=cg_tol, maxiter=cg_maxiter, M=M_pre)
    A = A_flat.reshape(nx, ny)
    A -= _np.nanmin(A)
    return A, {"cg_info": int(info), "cg_converged": int(info) == 0}


def compute_fel(meta, kernels, args, coords, periodic):
    
    kT = args.kT if args.kT is not None else meta['kT']
    ptilde, czar_grad = czar_on_grid(coords, periodic, kernels, kT,
                                      nsigma=args.nsigma, verbose=False)

    if len(coords) == 1:
        dx = coords[0][1] - coords[0][0]
        A = integrate_1d(czar_grad[..., 0], dx, periodic[0])
    elif len(coords) == 2 and args.integrator.lower() == "wls":
        A, info = wls_integrate_2d(
            czar_grad, coords, periodic,
            ptilde=ptilde, minpop=args.minpop,
            weight_exp=args.wexp,
            tikhonov=args.tikhonov,
            cg_tol=args.cg_tol,
            cg_maxiter=args.cg_maxiter,
        )
        if args.verbose:
            st = "converged" if info.get("cg_converged", False) else f"NOT converged (info={info.get('cg_info')})"
            print(f"[WLS] CG {st}", flush=True)
    else:
        
        A = poisson_integrate(czar_grad, coords, periodic, ptilde=ptilde, minpop=args.minpop)

    
    pmax = ptilde.max() if ptilde.max() > 0 else 1.0
    mask = ptilde >= args.minpop * pmax
    return A, mask, ptilde


def rmsd_convergence(args, coords, periodic, A_ref, mask_ref):
    




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
        return [], []

    print(f"  Found {len(candidates)} snapshot files matching '{stem}_XXXXXXXX{ext}'")

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

        
        common = mask_ref & mask_s
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




def main():
    parser = argparse.ArgumentParser(
        description='Recover FEL from FKERNELABF v5 CZAR kernel file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--czar',    required=True,    help='CZAR kernel file (reference)')
    parser.add_argument('--grid',    type=int, default=100, help='Grid points per dim')
    parser.add_argument('--nsigma',  type=float, default=4.0,
                        help='Kernel cutoff in sigma units (default 4.0)')
    parser.add_argument('--output',  default='FEL_czar.dat', help='Output file')
    parser.add_argument('--min',     type=float, nargs='+', default=None,
                        help='Override grid min per dim')
    parser.add_argument('--max',     type=float, nargs='+', default=None,
                        help='Override grid max per dim')
    parser.add_argument('--kT',      type=float, default=None, help='Override kT (kJ/mol)')
    parser.add_argument('--minpop',  type=float, default=1e-5,
                        help='Mask points with ptilde < MINPOP*max(ptilde) (default 1e-5)')

    parser.add_argument('--integrator', choices=['poisson','wls'], default='poisson',
                        help='Integration method for dim>1: poisson (default) or wls (recommended for nonperiodic).')
    parser.add_argument('--wexp', type=float, default=2.0,
                        help='WLS weight exponent (default 2.0). Only used with --integrator wls.')
    parser.add_argument('--tikhonov', type=float, default=1e-4,
                        help='WLS Tikhonov regularization strength (default 1e-4).')
    parser.add_argument('--cg_tol', type=float, default=1e-10,
                        help='WLS conjugate-gradient tolerance (default 1e-10).')
    parser.add_argument('--cg_maxiter', type=int, default=20000,
                        help='WLS conjugate-gradient max iterations (default 20000).')
    parser.add_argument('--convergence', action='store_true',
                        help='Compute RMSD time series vs reference (--czar) '
                             'using all czar_*_XXXXXXXX.dat snapshots in the same directory')
    parser.add_argument('--conv_output', default='czar_convergence.dat',
                        help='Output file for convergence RMSD time series (default: czar_convergence.dat)')
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

    
    print("Evaluating reference CZAR gradient on grid ...", flush=True)
    ptilde, czar_grad = czar_on_grid(coords, periodic, kernels, kT,
                                      nsigma=args.nsigma, verbose=args.verbose)

    frac_populated = np.mean(ptilde > 0)
    print(f"  Fraction of grid populated: {frac_populated*100:.1f}%", flush=True)
    if frac_populated < 0.1:
        print("  WARNING: less than 10% of grid has kernel coverage. "
              "Consider reducing --grid or increasing SIGMA.", flush=True)

    print("Integrating reference FEL ...", flush=True)
    shape = tuple(len(c) for c in coords)
    if dim == 1:
        dx = coords[0][1] - coords[0][0]
        A_ref = integrate_1d(czar_grad[..., 0], dx, periodic[0])
    else:
        A_ref = poisson_integrate(czar_grad, coords, periodic)

    pop_thresh = args.minpop * (ptilde.max() if ptilde.max() > 0 else 1.0)
    mask_ref = ptilde >= pop_thresh

    if mask_ref.any():
        A_range = A_ref[mask_ref].max() - A_ref[mask_ref].min()
        print(f"  Reference FEL range: {A_range:.3f} kJ/mol ({A_range/kT:.2f} kT)", flush=True)
    else:
        print("  WARNING: no grid points above minpop threshold.", flush=True)

    
    print(f"Writing reference FEL to: {args.output}", flush=True)
    write_output(args.output, coords, periodic, ptilde, czar_grad, A_ref,
                 args.minpop, kT)

    
    if args.convergence:
        print(f"\nConvergence mode: scanning for snapshot files ...", flush=True)
        steps, rmsds = rmsd_convergence(args, coords, periodic, A_ref, mask_ref)

        if steps:
            
            with open(args.conv_output, 'w') as fh:
                fh.write(f'# CZAR FEL convergence vs reference: {args.czar}\n')
                fh.write(f'# Grid: {args.grid} pts/dim  minpop={args.minpop}\n')
                fh.write(f'# RMSD computed over grid points sampled in BOTH reference and snapshot\n')
                fh.write(f'# FELs aligned to zero minimum over common region before RMSD\n')
                fh.write('# step  RMSD_kJ/mol\n')
                for s, r in zip(steps, rmsds):
                    fh.write(f'{s:10d}  {r:.6f}\n')
            print(f"\nConvergence table written to: {args.conv_output}", flush=True)

            
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                kB = 0.008314462618
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.plot(steps, rmsds, 'o-', lw=1.5, ms=5, color='steelblue')
                ax.axhline(kT, color='orange', lw=1.2, ls='--', label=f'kT = {kT:.3f} kJ/mol')
                ax.set_xlabel('Step')
                ax.set_ylabel('RMSD vs reference (kJ/mol)')
                ax.set_title('CZAR FEL convergence')
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

    print("Done.", flush=True)

    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import CenteredNorm

        imgfile = args.output.replace('.dat', '_summary.png')

        if dim == 1:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            A_plot = np.where(mask_ref, A_ref, np.nan)
            axes[0].plot(coords[0], czar_grad[..., 0], color='steelblue')
            axes[0].set_xlabel('z'); axes[0].set_ylabel('dA/dz (kJ/mol/rad)')
            axes[0].set_title('CZAR mean force'); axes[0].grid(True, alpha=0.3)
            axes[1].plot(coords[0], A_plot, color='darkorange')
            axes[1].set_xlabel('z'); axes[1].set_ylabel('A (kJ/mol)')
            axes[1].set_title('FEL via CZAR'); axes[1].grid(True, alpha=0.3)
            axes[2].plot(coords[0], np.log10(ptilde + 1e-300), color='seagreen')
            axes[2].set_xlabel('z'); axes[2].set_ylabel('log10(ptilde)')
            axes[2].set_title('Biased density'); axes[2].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(imgfile, dpi=150)
            print(f"  Summary plot saved: {imgfile}", flush=True)
            plt.close()

        elif dim == 2:
            c0, c1 = coords[0], coords[1]
            dx0 = c0[1]-c0[0]; dx1 = c1[1]-c1[0]
            A_plot = np.where(mask_ref, A_ref, np.nan)
            vmax_A = float(np.nanpercentile(A_plot[np.isfinite(A_plot)], 95))
            levels_A = np.linspace(0, vmax_A, 24)
            cmap_A = 'viridis'

            
            
            
            
            F0 = czar_grad[..., 0]   
            F1 = czar_grad[..., 1]
            per = meta.get('periodic', np.zeros(2, dtype=bool))
            
            if per[0]:
                dF1_dz0 = (np.roll(F1, -1, axis=0) - np.roll(F1, +1, axis=0)) / (2*dx0)
            else:
                dF1_dz0 = np.gradient(F1, dx0, axis=0)
            if per[1]:
                dF0_dz1 = (np.roll(F0, -1, axis=1) - np.roll(F0, +1, axis=1)) / (2*dx1)
            else:
                dF0_dz1 = np.gradient(F0, dx1, axis=1)
            curl = dF1_dz0 - dF0_dz1      
            curl_masked = np.where(mask_ref, curl, np.nan)
            
            curl_norm = curl_masked / kT
            curl_abs_95 = float(np.nanpercentile(np.abs(curl_norm[np.isfinite(curl_norm)]), 95))
            clim_curl = max(curl_abs_95, 0.01)

            
            curl_rms = float(np.sqrt(np.nanmean(curl_masked[mask_ref]**2))) if mask_ref.any() else np.nan
            curl_rms_norm = curl_rms / kT
            print(f"  Curl RMS = {curl_rms:.4f} kJ/mol/rad^2  "
                  f"({curl_rms_norm:.4f} in kT/rad^2 units)", flush=True)
            if curl_rms_norm > 0.5:
                print(f"  WARNING: large curl residual suggests the force field "
                      f"is not yet integrable -- FEL may have path-dependent errors.", flush=True)

            
            grad_mag = np.sqrt(czar_grad[...,0]**2 + czar_grad[...,1]**2)
            grad_mag_masked = np.where(mask_ref, grad_mag, np.nan)

            
            fig, axes = plt.subplots(2, 2, figsize=(13, 11))
            fig.suptitle('CZAR FEL Analysis', fontsize=14)

            
            ax = axes[0,0]
            cs = ax.contourf(c1, c0, A_plot, levels=levels_A, cmap=cmap_A, extend='max')
            ax.contour(c1, c0, A_plot, levels=levels_A[::4],
                       colors='k', linewidths=0.5, alpha=0.6)
            plt.colorbar(cs, ax=ax, label='A (kJ/mol)')
            ax.set_xlabel(f'z1 (rad)'); ax.set_ylabel(f'z0 (rad)')
            ax.set_title('Free Energy Surface')

            
            ax = axes[0,1]
            ld = np.log10(ptilde + 1e-300)
            ld_masked = np.where(mask_ref, ld, np.nan)
            cs2 = ax.contourf(c1, c0, ld_masked, levels=20, cmap='plasma')
            plt.colorbar(cs2, ax=ax, label='log10(ptilde)')
            ax.set_xlabel(f'z1 (rad)'); ax.set_ylabel(f'z0 (rad)')
            ax.set_title('Biased sampling density log10(ptilde)')

            
            ax = axes[1,0]
            gmax = float(np.nanpercentile(grad_mag_masked[np.isfinite(grad_mag_masked)], 95))
            cs3 = ax.contourf(c1, c0, grad_mag_masked,
                              levels=np.linspace(0, gmax, 20), cmap='hot_r', extend='max')
            
            ax.contour(c1, c0, A_plot, levels=levels_A[::4],
                       colors='white', linewidths=0.4, alpha=0.5)
            plt.colorbar(cs3, ax=ax, label='|grad A| (kJ/mol/rad)')
            ax.set_xlabel(f'z1 (rad)'); ax.set_ylabel(f'z0 (rad)')
            ax.set_title('Mean force magnitude |dA/dz|')

            
            ax = axes[1,1]
            cs4 = ax.contourf(c1, c0, curl_norm,
                              levels=np.linspace(-clim_curl, clim_curl, 24),
                              cmap='bwr', extend='both')
            ax.contour(c1, c0, A_plot, levels=levels_A[::4],
                       colors='k', linewidths=0.4, alpha=0.4)
            plt.colorbar(cs4, ax=ax, label='(dF1/dz0 - dF0/dz1) / kT  [rad^-2]')
            ax.set_xlabel(f'z1 (rad)'); ax.set_ylabel(f'z0 (rad)')
            ax.set_title(f'Curl residual / kT  (RMS={curl_rms_norm:.3f})')

            
            for axi in axes.flat:
                axi.contour(c1, c0, mask_ref.astype(float),
                            levels=[0.5], colors=['grey'], linewidths=0.8,
                            linestyles=['--'])

            plt.tight_layout()
            plt.savefig(imgfile, dpi=150)
            print(f"  Summary plot saved: {imgfile}", flush=True)
            plt.close()

    except ImportError:
        pass


if __name__ == '__main__':
    main()
