# Force Kernel ABF

A toolkit for running Force Kernel ABF (FKERNELABF) enhanced sampling simulations in PLUMED and recovering free energy landscapes from the results.

FKERNELABF is an adaptive biasing force method that uses an extended Lagrangian (fictitious particle λ coupled to the real collective variable z) and a kernel-based mean force estimator. The CZAR estimator on the real CV z provides an unbiased free energy gradient that is integrated into a free energy landscape in post-processing.

---

## Requirements

- PLUMED (with `forcekernelabf_v5_0_0.cpp` compiled as a plugin via `LOAD`)
- A MD engine supported by PLUMED, or the built-in `pesmd` toy integrator for 2D potentials
- Python 3 with NumPy and SciPy for post-processing

---

## Workflow

### 1. Setting up a FKABF simulation

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

#### Compulsory Keywords

| Keyword | Default | Description |
|---------|---------|-------------|
| `ARG` | — | Collective variables (standard PLUMED). One to three CVs supported. |
| `KAPPA` | — | Spring constant(s) for the extended Lagrangian coupling (kJ/mol/unit²). One value or one per CV. Larger κ → tighter z–λ coupling, smaller fluctuations σ = √(kT/κ). |
| `TAU` | `0.5` | Oscillation period(s) of the extended variable (time units). Determines the fictitious mass as m = κτ²/(4π²). One value or one per CV. |
| `FRICTION` | `10.0` | Langevin friction coefficient for the BAOAB thermostat on λ (1/time_unit). Setting one option sets the FRICTION for all CVs, alternatively, it accepts one value per CV. |
| `TEMP` | `300.0` | Temperature (K). Sets kT = k_B × TEMP. |
| `PACE` | `5` | Deposit a force sample into both kernel populations every PACE MD steps. |
| `THRESH` | `1.0` | Kernel merge threshold in σ-normalised distance. Kernels closer than THRESH × σ_global are merged via parallel variance. 1.0 is the OPES standard; lower → more compression, higher → more kernels. Do not change unless you know what you are doing. |
| `NSIGMACUT` | `4.0` | Kernel cutoff in σ per dimension. Kernels farther than NSIGMACUT × σ are ignored in NW regression. 4.0 gives < 2% contribution at the boundary. Do not change unless you know what you are doing. |
| `BIASFACTOR` | `1.0` | Exploration factor γ. `1.0` = pure ABF (no exploration). `> 1.0` = density-based exploration: V_ex = c·ln(1 + Z/Z₀) where c = kT(γ−1). Pushes λ away from well-sampled basins. The CZAR estimator on z is unaffected. |
| `EXPLORSCALE` | `1.0` | Exploration scaling factor if `BIASFACTOR > 1`. `1.0` = full exploration biasing force. `< 1.0` = reduced application of force on the CV from the density-based bias, setting it `= 0` disables the density-based boost. This is an option to reduce the additional force on CVs that are more auxillary, and do not drive the transition, for example setting `= 1.0, 0.0` for a two CV system disables the boost on the second CV. |
| `MUXCLAMP` | `500.0` | Per-kernel mean-force clamp (kJ/mol/unit). Individual kernel μ values are hard-clamped to ±MUXCLAMP on absorption. Safety net for sparse regions. |
| `MAXFORCE` | `500.0` | Grid mean-force clamp (kJ/mol/unit). The NW mean force on the grid is clamped per-node before interpolation. Safety net for unphysical force estimates. |
| `GRIDSIZE` | `72` | Grid points per dimension for the frozen mean-force grid. |
| `GRIDPACE` | `500` | Rebuild the mean-force grid from λ-kernels every GRIDPACE steps. Between rebuilds, bias forces are interpolated from the frozen grid. This is a safe choice for classical simulations but should be reduced for AIMD. |

#### Optional Keywords — Bandwidth

| Keyword | Default | Description |
|---------|---------|-------------|
| `SIGMA` | *(auto)* | Initial kernel bandwidth σ₀, effectively, the radius of the kernel. Setting this to half Gaussian width one would set for Metadynamics is a safe choice. One value, one per CV, or **omit entirely** for adaptive mode (measures CV variance during an unbiased warmup). |
| `SIGMA_MIN` | *(none)* | Minimum bandwidth floor. Silverman rescaling and per-kernel variance will never shrink σ below this value. One value or one per CV. |
| `ADAPTIVE_SIGMA_STRIDE` | `10 × PACE` | Number of unbiased warmup steps for automatic σ₀ determination (Welford online variance). Only used when `SIGMA` is omitted. During warmup: zero bias, no kernel deposition, λ tracks z. |
| `FIXED_SIGMA` | `false` | Flag. If set, disables Silverman bandwidth rescaling — all kernels use σ₀ permanently. |

#### Optional Keywords — Grid Bounds

| Keyword | Default | Description |
|---------|---------|-------------|
| `GRIDMIN` | *(from CV)* | Lower grid bound(s) for non-periodic CVs. For periodic CVs, bounds are taken from the CV period. One per CV. Reflecting walls automatically applied. |
| `GRIDMAX` | *(from CV)* | Upper grid bound(s) for non-periodic CVs. One per CV. Reflecting walls automatically applied. |

#### Optional Keywords — Neighbor List

| Keyword | Default | Description |
|---------|---------|-------------|
| `NONLIST` | `false` | Flag. Disables the neighbor list (brute-force kernel search). |
| `NLIST_PARAMETERS` | `3.0 0.5` | Two values: cutoff_factor and skin_factor. The neighbor list includes kernels within cutoff_factor × NSIGMACUT × σ, and rebuilds when the query point drifts by skin_factor × dev². |

#### Optional Keywords — Output Files

All filenames are derived from the PLUMED action label (e.g., `fk: FKERNELABF ...` → files prefixed with `fk.`).

| Keyword | Default | Description |
|---------|---------|-------------|
| `CZARSTRIDE` | *(off)* | Write a step-stamped CZAR z-kernel snapshot every N steps → `{label}.czar_kernels_{step:08d}.dat`. Feed to `czar_integrate` to recover A(z). |
| `KERNELSTRIDE` | *(off)* | Write a step-stamped λ-kernel snapshot every N steps → `{label}.kernels_{step:08d}.dat`. |
| `LAMBDAGRIDSTRIDE` | *(off)* | Write the NW mean-force debug grid every N steps → `{label}.lambda_grid_{step:08d}.dat`. This is the bias force on the λ grid, **not** the free energy. |
| `STATESTRIDE` | `CZARSTRIDE` or `10 × GRIDPACE` | Write restart state file every N steps → `{label}.state.dat`. Overwritten in place. Read automatically on `RESTART`. |
| `KERNELINFOSTRIDE` | `PACE` | Append one line of kernel diagnostics every N steps → `KERNELINFO`. Columns: step, M, zM, neff, σ per CV, nlker. |

<br>

---

<br>

### 2. Running the simulation

For the included Müller-Brown benchmark, run the exaple input using PLUMED's built-in 2D toy integrator (`pesmd`) with the provided input files:

```bash
plumed pesmd < pesmd.in
```

This runs the simulation defined in `pesmd.in` (10M steps on the 2D Müller-Brown potential) driven by `plumed.dat`, and writes CZAR kernel dump files at the configured stride.

<br>

---

<br>

### 3. Recover the free energy landscape

Compile 'czar_integrate.cpp':

```bash
 g++ -O2 -o czar_integrate czar_integrate.cpp -lm 
```

The executable can now be used to process the czar_kernel files: 

```bash
./czar_integrate FEL_snapshots -d /path/to/scan
```
czar_integrate only requires one argument: the directory for depositing the PMFs. Else, czar_integrate can read files in a specified directory (-d) and for all files matching `*czar_kernels_XXXXXXXX.dat`, integrates, and writes `FEL_XXXXXXXX.dat` to the output directory.

#### Options

| Flag | Argument | Default | Description |
|------|----------|---------|-------------|
| `-n` | `<steps>` | `0` | MC integration steps. `0` = auto-converge (stop when RMSD stabilises). |
| `-h` | `<height>` | `0.01` | Initial hill height for the MC bias potential. |
| `-f` | `<factor>` | `0.5` | Hill reduction factor. Hill is multiplied by this at regular intervals after a warmup period. |
| `-t` | `<kT>` | *(from file)* | Override kT from the kernel file header (kJ/mol). |
| `-g` | `<pts>` | `100` | Grid points per dimension for the integration grid. |
| `-s` | `<nsigma>` | `4.0` | Kernel cutoff in σ units. Kernels beyond this distance are skipped. |
| `-m` | `<minpop>` | `1e-3` | Minimum density fraction for the allowed region. Grid points with density below `minpop × max(density)` are masked as NaN in the output. |
| `-d` | `<dir>` | `.` | Directory to scan for kernel snapshot files (batch mode). |
| `-i` | `<file>` | — | Process a single kernel file instead of scanning. |
| `-o` | `<file>` | `FEL_czar.dat` | Output filename (single-file mode only). |
| `-v` | — | off | Verbose: print per-step RMSD, hill scaling, and convergence diagnostics. |
| `-S` | — | off | Skip kernel files before some step number |

#### Examples

```bash
# Batch: scan current directory, write FEL snapshots
./czar_integrate FEL_snapshots

# Batch with fixed MC steps and user-specified height
./czar_integrate FEL_snapshots -n 5000000 -h 0.2

# Batch with fixed MC steps, user-specified height, skip analyzing before step 5M
./czar_integrate FEL_snapshots -n 5000000 -h 0.2 -S 5000000

# Batch: scan a different directory
./czar_integrate FEL_snapshots -d /path/to/run

# Single file processing
./czar_integrate -i fk.czar_kernels_10000000.dat -o PMF.dat

# Fine grid, verbose
./czar_integrate FEL_snapshots -g 150 -v
```

#### Output Format

**Single-file mode** (`-i`): a space-separated file with columns:

| Column | Description |
|--------|-------------|
| `z0`, `z1`, ... | Grid coordinates (one per CV dimension) |
| `czar_grad0`, `czar_grad1`, ... | CZAR free-energy gradient components (kJ/mol/unit) |
| `ptilde` | Biased density (NW denominator) at each grid point |
| `A_czar` | Free energy (kJ/mol), shifted so the minimum is zero |

Points below the population threshold are written as `nan`. For 2D+ grids, blank lines separate slices along the first dimension (gnuplot `pm3d` compatible).

<br>

---

<br>

### 4. Additional diagnosis 

As an additional diagnostic tool, fkabf_diagnostics.py can be used to process the COLVAR file and the KERNELINFO file in the current directory:
```bash
 python fkabf_diagnostics.py
```

#### Options

| Flag | Argument | Default | Description |
|------|----------|---------|-------------|
| `--colvar` | `<file>` | `COLVAR` | Path to the PLUMED COLVAR file. |
| `--kernelinfo` | `<file>` | `KERNELINFO` | Path to the KERNELINFO file. If absent, kernel diagnostic plots are skipped. |
| `--prefix` | `<label>` | *(auto)* | PLUMED action label prefix (e.g., `fk`). Auto-detected from `_fict` column names if not set. |
| `--dt` | `<float>` | `0.001` | MD timestep in time units. Used to convert the `time` column to step numbers. |
| `--thinning` | `<int>` | `10` | Thinning factor for scatter and trajectory plots. Every Nth point is plotted. |
| `--periodic` | `<spec>` | *(none)* | Periodic CV specification for minimum-image z−λ differences. Format: `"cv1:min:max,cv2:min:max"` or `"cv1:period"`. Supports `pi` in expressions. |
| `--outdir` | `<dir>` | `.` | Output directory for figures. Created if it does not exist. |

#### Examples

```bash
# Minimal: auto-detect everything in current directory
python fkabf_diagnostics.py

# Specify files and output location
python fkabf_diagnostics.py --colvar COLVAR --kernelinfo KERNELINFO --outdir plots/

# Alanine dipeptide with periodic CVs
python fkabf_diagnostics.py --periodic "phi:-pi:pi,psi:-pi:pi" --dt 0.002

# Dense trajectory, less thinning
python fkabf_diagnostics.py --thinning 2
```

#### Output Figures

| File | Contents |
|------|----------|
| `fig_trajectory.pdf` | Per-CV row with three columns: (a) z and λ time series, (b) z−λ difference over time, (c) histogram of z−λ. For periodic CVs (`--periodic`), the minimum-image difference is used. |
| `fig_bias.pdf` | (a) Bias force magnitude \|F_bias\| over time, (b) exploration potential V_ex over time. |
| `fig_kernels.pdf` | From KERNELINFO: (a) kernel counts M and M_z, (b) n_eff and compression ratio N/M, (c) Silverman σ per CV, (d) Z₀ and Z(λ) if present. |
| `fig_exploration.pdf` | Side-by-side 2D scatter: (a) real CV trajectory, (b) λ trajectory, both colored by simulation time. Only produced for 2+ CVs. |
| `fig_phase.pdf` | z vs λ scatter per CV, colored by time. Points should cluster along the identity line; spread indicates coupling width √(kT/κ). |
| `fig_nlist.pdf` | Neighbor list size over time, with nlker/M fraction on a twin axis. |

A text summary of the run (CV ranges, z−λ standard deviation, final kernel counts, compression ratio, convergence metrics) is printed to stdout before figure generation.





