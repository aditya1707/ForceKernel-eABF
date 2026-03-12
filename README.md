# MB-ForceKernel

A toolkit for running Force Kernel ABF (FKERNELABF) enhanced sampling simulations in PLUMED and recovering free energy landscapes from the results.

FKERNELABF is an adaptive biasing force method that uses an extended Lagrangian (fictitious particle λ coupled to the real collective variable z) and a kernel-based mean force estimator. The CZAR estimator on the real CV z provides an unbiased free energy gradient that is integrated into a free energy landscape in post-processing.

---

## Requirements

- PLUMED (with `forcekernelabf_v5_0_0.cpp` compiled as a plugin via `LOAD`)
- A MD engine supported by PLUMED, or the built-in `pesmd` toy integrator for 2D potentials
- Python 3 with NumPy and SciPy for post-processing

---

## Workflow

### 1. Run the simulation

The plugin is loaded at runtime via the PLUMED `LOAD` directive — no recompilation of PLUMED is required. Configure your `plumed.dat` to load the plugin and define the `FKERNELABF` action:

```plumed
LOAD FILE=./forcekernelabf_v5_0_0.cpp

cv: <YOUR_CV_DEFINITION>

fk: FKERNELABF ...
    
    # CV Definition
    ARG=cv

    # Extended Lagrangian Options
    KAPPA=3000.0          # coupling spring constant (kJ/mol/nm^2)
    TAU=0.5               # fictitious particle time constant (ps)
    FRICTION=8.0          # Langevin friction (ps^-1)
    TEMP=300              # temperature (K)

    #FKABF Options
    GRIDMIN=-1.5          # CV domain lower bound
    GRIDMAX=1.5           # CV domain upper bound
    SIGMA=0.05            # initial kernel width
    SIGMA_MIN=0.01        # minimum kernel width
    PACE=5                # steps between data accumulation
    GRIDPACE=1000         # steps bewteen biasing force updates

    # Output options
    CZARSTRIDE=50000      # steps between CZAR kernel file writes
    KERNELINFOSTRIDE=500  # steps for writing kernel information  
...

PRINT FILE=COLVAR STRIDE=500 ARG=*
```

Alternatively, it is possible for FKABF to set its own options, where an adaptive sigma is determined based on 10xPACE (or set with 'ADAPTIVE_SIGMA_STRIDE'), and the default data aquisition (PACE) and force-field update frequency (GRIDPACE) are often sufficient for learning the gradient in classical simulations.

```plumed
LOAD FILE=./forcekernelabf_v5_0_0.cpp

cv: <YOUR_CV_DEFINITION>

fk: FKERNELABF ...
    
    # CV Definition
    ARG=cv

    # Extended Lagrangian Options
    KAPPA=3000.0          # coupling spring constant (kJ/mol/nm^2)
    TAU=0.5               # fictitious particle time constant (ps)
    FRICTION=8.0          # Langevin friction (ps^-1)
    TEMP=300              # temperature (K)

    #FKABF Options
    GRIDMIN=-1.5          # CV domain lower bound
    GRIDMAX=1.5           # CV domain upper bound

    # Output options
    CZARSTRIDE=50000      # steps between CZAR kernel file writes
    KERNELINFOSTRIDE=500  # steps for writing kernel information  
...

PRINT FILE=COLVAR STRIDE=500 ARG=*
```

### Compulsory Keywords

| Keyword | Default | Description |
|---------|---------|-------------|
| `ARG` | — | Collective variables (standard PLUMED). One to three CVs supported. |
| `KAPPA` | — | Spring constant(s) for the extended Lagrangian coupling (kJ/mol/unit²). One value or one per CV. Larger κ → tighter z–λ coupling, smaller fluctuations σ = √(kT/κ). |
| `TAU` | `0.5` | Oscillation period(s) of the extended variable (time units). Determines the fictitious mass as m = κτ²/(4π²). One value or one per CV. |
| `FRICTION` | `10.0` | Langevin friction coefficient for the BAOAB thermostat on λ (1/time_unit). |
| `TEMP` | `300.0` | Temperature (K). Sets kT = k_B × TEMP. |
| `PACE` | `5` | Deposit a force sample into both kernel populations every PACE MD steps. |
| `THRESH` | `1.0` | Kernel merge threshold in σ-normalised distance. Kernels closer than THRESH × σ_global are merged via parallel variance. 1.0 is the OPES standard; lower → more compression, higher → more kernels. |
| `NSIGMACUT` | `4.0` | Kernel cutoff in σ per dimension. Kernels farther than NSIGMACUT × σ are ignored in NW regression. 4.0 gives < 2% contribution at the boundary. |
| `BIASFACTOR` | `1.0` | Exploration factor γ. `1.0` = pure ABF (no exploration). `> 1.0` = density-based exploration: V_ex = c·ln(1 + Z/Z₀) where c = kT(γ−1). Pushes λ away from well-sampled basins. The CZAR estimator on z is unaffected. |
| `MUXCLAMP` | `500.0` | Per-kernel mean-force clamp (kJ/mol/unit). Individual kernel μ values are hard-clamped to ±MUXCLAMP on absorption. Safety net for sparse regions. |
| `MAXFORCE` | `500.0` | Grid mean-force clamp (kJ/mol/unit). The NW mean force on the grid is clamped per-node before interpolation. Safety net for unphysical force estimates. |
| `GRIDSIZE` | `72` | Grid points per dimension for the frozen mean-force grid. |
| `GRIDPACE` | `500` | Rebuild the mean-force grid from λ-kernels every GRIDPACE steps. Between rebuilds, bias forces are interpolated from the frozen grid. |

### Optional Keywords — Bandwidth

| Keyword | Default | Description |
|---------|---------|-------------|
| `SIGMA` | *(auto)* | Initial kernel bandwidth σ₀. One value, one per CV, or **omit entirely** for adaptive mode (measures CV variance during an unbiased warmup). |
| `SIGMA_MIN` | *(none)* | Minimum bandwidth floor. Silverman rescaling and per-kernel variance will never shrink σ below this value. One value or one per CV. |
| `ADAPTIVE_SIGMA_STRIDE` | `10 × PACE` | Number of unbiased warmup steps for automatic σ₀ determination (Welford online variance). Only used when `SIGMA` is omitted. During warmup: zero bias, no kernel deposition, λ tracks z. |
| `FIXED_SIGMA` | `false` | Flag. If set, disables Silverman bandwidth rescaling — all kernels use σ₀ permanently. |

### Optional Keywords — Grid Bounds

| Keyword | Default | Description |
|---------|---------|-------------|
| `GRIDMIN` | *(from CV)* | Lower grid bound(s) for non-periodic CVs. For periodic CVs, bounds are taken from the CV period. One per CV. |
| `GRIDMAX` | *(from CV)* | Upper grid bound(s) for non-periodic CVs. One per CV. |

### Optional Keywords — Neighbor List

| Keyword | Default | Description |
|---------|---------|-------------|
| `NONLIST` | `false` | Flag. Disables the neighbor list (brute-force kernel search). |
| `NLIST_PARAMETERS` | `3.0 0.5` | Two values: cutoff_factor and skin_factor. The neighbor list includes kernels within cutoff_factor × NSIGMACUT × σ, and rebuilds when the query point drifts by skin_factor × dev². |

### Optional Keywords — Output Files

All filenames are derived from the PLUMED action label (e.g., `fk: FKERNELABF ...` → files prefixed with `fk.`).

| Keyword | Default | Description |
|---------|---------|-------------|
| `CZARSTRIDE` | *(off)* | Write a step-stamped CZAR z-kernel snapshot every N steps → `{label}.czar_kernels_{step:08d}.dat`. Feed to `czar_integrate` to recover A(z). |
| `KERNELSTRIDE` | *(off)* | Write a step-stamped λ-kernel snapshot every N steps → `{label}.kernels_{step:08d}.dat`. |
| `LAMBDAGRIDSTRIDE` | *(off)* | Write the NW mean-force debug grid every N steps → `{label}.lambda_grid_{step:08d}.dat`. This is the bias force on the λ grid, **not** the free energy. |
| `STATESTRIDE` | `CZARSTRIDE` or `10 × GRIDPACE` | Write restart state file every N steps → `{label}.state.dat`. Overwritten in place. Read automatically on `RESTART`. |
| `KERNELINFOSTRIDE` | `PACE` | Append one line of kernel diagnostics every N steps → `KERNELINFO`. Columns: step, M, zM, neff, σ per CV, nlker. |




For the included Müller-Brown benchmark, run the exaple input using PLUMED's built-in 2D toy integrator (`pesmd`) with the provided input files:

```bash
plumed pesmd < pesmd.in
```

This runs the simulation defined in `pesmd.in` (10M steps on the 2D Müller-Brown potential) driven by `plumed.dat`, and writes CZAR kernel dump files at the configured stride.

### 2. Recover the free energy landscape

Compile 'czar_integrate.cpp':

```bash
 g++ -O2 -o czar_integrate czar_integrate.cpp -lm 
```

The executable can now be used to process the czar_kernel files: 

```bash
./czar_integrate FEL_snapshots -d /path/to/scan' 
```
will read files in the directory '-d' and will process all kenel files, depositing the resulting PMFs into the specified directory

```bash
python czar_integrate_improved.py --czar czar_kernels.dat --grid 100 --output FEL.dat
```

Key options:

Options:
  -n <steps>      MC integration steps (0 = auto-converge) [0]
  -h <height>     Initial hill height                      [0.01]
  -f <factor>     Hill reduction factor                    [0.5]
  -t <kT>         Override kT from file (kJ/mol)
  -g <pts>        Grid points per dimension                [100]
  -s <nsigma>     Kernel cutoff in sigma units             [4.0]
  -m <minpop>     Min density fraction for allowed region  [1e-3]
  -d <dir>        Directory to scan (default: current)     [.]
  -i <file>       Single-file mode (skip auto-scan)
  -o <file>       Output file for single-file mode         [FEL_czar.dat]
  -v              Verbose output

Examples:
  ./czar_integrate FEL_snapshots                 # scan ., write to FEL_snapshots/
  ./czar_integrate FEL_snapshots -n 5000000      # fixed MC steps
  ./czar_integrate FEL_snapshots -d /path/to/run # scan another directory
  ./czar_integrate -i fk.czar_kernels_10000000.dat -o PMF.dat  # single file

The output is a space-separated file with CV coordinates, CZAR gradients, biased density, and free energy (kJ/mol; plumed default). Points below the population threshold are written as NaN.





