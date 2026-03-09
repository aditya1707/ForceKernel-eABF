# MB-ForceKernel

A toolkit for running Force Kernel ABF (FKERNELABF) enhanced sampling simulations in PLUMED and recovering free energy landscapes from the results.

FKERNELABF is an adaptive biasing force method that uses an extended Lagrangian (fictitious particle λ coupled to the real collective variable z) and a kernel-based mean force estimator. The CZAR estimator on the real CV z provides an unbiased free energy gradient that is integrated into a free energy landscape in post-processing.

---

## Requirements

- PLUMED (with `forcekernelabf_v5_0_0.cpp` compiled as a plugin via `LOAD`)
- A MD engine supported by PLUMED, or the built-in `pesmd` toy integrator for 2D potentials
- Python 3 with NumPy (and SciPy for `--integrator wls`) for post-processing

---

## Workflow

### 1. Run the simulation

The plugin is loaded at runtime via the PLUMED `LOAD` directive — no recompilation of PLUMED is required. Configure your `plumed.dat` to load the plugin and define the `FKERNELABF` action:

```plumed
LOAD FILE=./forcekernelabf_v5_0_0.cpp

cv: <YOUR_CV_DEFINITION>

fk: FKERNELABF ...
    ARG=cv
    KAPPA=3000.0        # coupling spring constant (kJ/mol/nm^2)
    TAU=0.5             # fictitious particle time constant (ps)
    FRICTION=8.0        # Langevin friction (ps^-1)
    TEMP=300            # temperature (K)
    GRIDMIN=-1.5        # CV domain lower bound
    GRIDMAX=1.5         # CV domain upper bound
    SIGMA=0.05          # initial kernel width
    SIGMA_MIN=0.01      # minimum kernel width
    PACE=5              # steps between kernel updates
    CZARSTRIDE=50000    # steps between CZAR kernel file writes
...

PRINT FILE=COLVAR STRIDE=50 ARG=*
```

For the included Müller-Brown benchmark, run using PLUMED's built-in 2D toy integrator (`pesmd`) with the provided input files:

```bash
plumed pesmd < pesmd.in
```

This runs the simulation defined in `pesmd.in` (10M steps on the 2D Müller-Brown potential) driven by `plumed.dat`, and writes CZAR kernel dump files at the configured stride.

### 2. Recover the free energy landscape

Use `czar_integrate_improved.py` to integrate the CZAR kernel file into a free energy surface:

```bash
python czar_integrate_improved.py --czar czar_kernels.dat --grid 100 --output FEL.dat
```

Key options:

| Option | Description | Default |
|---|---|---|
| `--czar FILE` | CZAR kernel file | required |
| `--grid N` | Grid points per dimension | 100 |
| `--output FILE` | Output FEL file | FEL_czar.dat |
| `--minpop FRAC` | Mask points below FRAC × max density | 1e-5 |
| `--integrator` | `poisson` or `wls` (weighted least-squares) | poisson |
| `--convergence` | Compute RMSD time series from snapshot files | off |
| `--kT VALUE` | Override kT (kJ/mol) | from file |

The output is a space-separated file with CV coordinates, CZAR gradients, biased density, and free energy (kJ/mol). Points below the population threshold are written as NaN. A summary plot is saved alongside the data file.

Alternatively, the standalone C++ tool `czar_integrate` performs the same integration using an MC random-walk integrator and requires no Python dependencies:

```bash
g++ -O2 -o czar_integrate czar_integrate.cpp -lm
./czar_integrate czar_kernels.dat -g 100 -o FEL.dat
```

Use `-a` to batch-process all timestamped snapshot files in a directory.
