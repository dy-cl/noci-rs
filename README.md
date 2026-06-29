# noci-rs

<p align="center">
  <img src="assets/logo.png" alt="Logo" width="220">
</p>

<h1 align="center">noci-rs</h1>

`noci-rs` is a Rust code for non-orthogonal configuration interaction (NOCI), selected NOCI (SNOCI), and deterministic or stochastic NOCI-QMC calculations on molecular systems. It drives PySCF to generate one- and two-electron integrals, builds non-orthogonal SCF determinant bases, and evaluates reference and selected non-orthogonal CI spaces.

[![tests](https://github.com/dy-cl/noci-rs/actions/workflows/tests.yml/badge.svg)](https://github.com/dy-cl/noci-rs/actions/workflows/tests.yml)
[![Clippy](https://github.com/dy-cl/noci-rs/actions/workflows/clippy.yml/badge.svg?branch=main)](https://github.com/dy-cl/noci-rs/actions/workflows/clippy.yml)
[![codecov](https://codecov.io/gh/dy-cl/noci-rs/branch/main/graph/badge.svg)](https://codecov.io/gh/dy-cl/noci-rs)

## Current Capabilities

- **SCF reference generation**
  - RHF and UHF SCF solutions with DIIS acceleration.
  - MOM-style state recipes with spin-density bias, spatial-density bias, and occupied-virtual excitation seeds.
  - SCF metadynamics for discovering multiple RHF/UHF solutions.
  - Holomorphic SCF (`h-SCF`) optimisation for complex continuations of selected MOM states.
  - Geometry scans, with converged states from one geometry reused as seeds for the next.

- **Reference NOCI**
  - NOCI basis construction from selected real or holomorphic SCF states.
  - Optional excited determinant generation from user-selected excitation orders, or all available orders.
  - Hamiltonian, overlap, and Fock matrix element generation.
  - Matrix elements via generalised Slater-Condon rules, extended non-orthogonal Wick's theorem, or orthogonal shortcuts where applicable.
  - Wick intermediate storage in RAM or disk-backed cache.
  - Rayon-parallel matrix builds and MPI distribution for shared calculations.

- **Selected NOCI (SNOCI)**
  - Iterative candidate generation and determinant selection.
  - NOCI-PT2 based candidate scoring.
  - GMRES solution of projected candidate-space equations.
  - Diagonal or Woodbury preconditioning.
  - Optional imaginary shifts for complex SNOCI/NOCI-PT2.

- **NOCI-QMC propagation**
  - Deterministic imaginary-time propagation with optional dynamic shift.
  - Stochastic NOCI-QMC propagation with MPI and Rayon parallelism.
  - Uniform, heat-bath, and approximate heat-bath excitation generation.
  - Propagator choices: `unshifted`, `shifted`, `doubly-shifted`, `difference-doubly-shifted-u1`, and `difference-doubly-shifted-u2`.

- **Output and restart support**
  - Text reports for SCF states, reference NOCI, SNOCI, NOCI-PT2, deterministic propagation, stochastic propagation, and timings.
  - Optional HDF5 output for orbitals and matrices.
  - Optional deterministic coefficient and excitation histogram output.
  - Optional stochastic restart read/write files.

## Requirements

- Rust toolchain with Cargo.
- HDF5 development libraries compatible with the `hdf5` Rust crate.
- OpenBLAS/LAPACK libraries for `ndarray-linalg`.
- MPI compiler/runtime.
- Python 3 with PySCF.

## Build

```bash
git clone https://github.com/dy-cl/noci-rs
cd noci-rs
cargo build --release
```

Timing counters are available with:

```bash
cargo build --release --features timings
```

## Run

```bash
mpirun -np X ./target/release/noci-rs input.lua > output.out
```

For one MPI rank:

```bash
cargo run -- input.lua
```

Input examples live under `inputs/`.

## Minimal Example

```lua
mol = {
    basis = "cc-pVDZ",
    r = {1.50},
    unit = "Ang",
    atoms = function(r)
        return {
            string.format("H 0 0 %g", -r / 2),
            string.format("H 0 0 %g",  r / 2),
        }
    end,
}

scf = {
    max_cycle = 10000,
    e_tol = 1e-10,
    diis = {space = 8},
}

states = {
    mom = {
        {label = "RHF", noci = true},
        {label = "UHF (+, -)", spin_bias = {pattern = {1, -1}, pol = 0.75}, noci = true},
        {label = "UHF (-, +)", spin_bias = {pattern = {-1, 1}, pol = 0.75}, noci = true},
    }
}

excit = {
    orders = {1, 2},
}

wicks = {
    enabled = true,
    compare = false,
    storage = "ram",
    cachedir = ".",
}
```

## Input Tables

Input files are Lua scripts. Required top-level tables:

- `mol`: basis, unit, scan coordinate(s), and atoms.
- `states`: MOM recipes or SCF metadynamics settings.

Optional top-level tables:

- `scf`: SCF and h-SCF convergence settings.
- `excit`: determinant excitation generation.
- `prop`: shared propagation timestep and propagator.
- `det`: deterministic propagation settings.
- `qmc`: stochastic NOCI-QMC settings.
- `snoci`: selected NOCI and NOCI-PT2 settings.
- `noccmc`: NOCCMC settings.
- `write`: optional file output and restart settings.
- `wicks`: extended non-orthogonal Wick's theorem settings.

## Common Options

### Molecule

`mol.r` may be a number or a Lua table of scan points. `mol.atoms` may be a static atom-string table or a function of `r`.

```lua
mol = {
    basis = "cc-pVDZ",
    r = {0.8, 1.0, 1.2},
    unit = "Ang",
    atoms = function(r)
        return {
            string.format("H 0 0 %g", -r / 2),
            string.format("H 0 0 %g",  r / 2),
        }
    end,
}
```

### States

MOM recipes support `spin_bias`, `spatial_bias`, `excit`, `noci`, `holomorphic`, and `partner`.

```lua
states = {
    mom = {
        {label = "RHF", noci = true},
        {label = "UHF (+, -)", spin_bias = {pattern = {1, -1}, pol = 0.75}, noci = true},
        {label = "RHF excited", excit = {spin = "both", occ = -1, vir = 0}, noci = false},
        {label = "h-UHF (+, -)", holomorphic = true, partner = "UHF (+, -)", noci = true},
    }
}
```

Metadynamics is an alternative to MOM recipes:

```lua
states = {
    metadynamics = {
        nstates_rhf = 2,
        nstates_uhf = 4,
        spinpol = 0.75,
        spatialpol = 0.75,
        lambda = 0.5,
        max_attempts = 100,
    }
}
```

### Excitations

```lua
excit = {
    orders = {1, 2},
}
```

Use `excit.all = true` instead of `orders` to generate all supported excitation orders.

### Propagation

`prop` is required when `det` or `qmc` is present.

```lua
prop = {
    dt = 1e-4,
    propagator = "difference-doubly-shifted-u2",
}

det = {
    max_steps = 10000,
    dynamic_shift = true,
    dynamic_shift_alpha = 0.1,
    e_tol = 1e-10,
}

qmc = {
    initial_population = 100,
    target_population = 100000,
    shift_damping = 5e-4,
    ncycles = 10,
    nreports = 1000,
    excitation_gen = "uniform",
    seed = 12345,
}
```

### SNOCI

```lua
snoci = {
    sigma = 1e-6,
    tol = 1e-8,
    max_iter = 100,
    max_add = 1,
    max_dim = 100,
    preconditioner = "woodbury",
    imag_shift = {0.0},
    gmres = {
        max_iter = 100,
        restart = 200,
        res_tol = 1e-8,
        metric_tol = 1e-8,
        full_m = false,
    },
}
```

### NOCCMC

```lua
noccmc = {}
```

### Output

```lua
write = {
    verbose = true,
    write_dir = "outputs/",
    write_orbitals = false,
    write_matrices = false,
    write_deterministic_coeffs = false,
    write_excitation_hist = false,
    write_restart = nil,
    read_restart = nil,
}
```

### Wick Intermediates

```lua
wicks = {
    enabled = true,
    compare = false,
    storage = "ram",
    cachedir = ".",
}
```

Set `storage = "disk"` to use disk-backed Wick intermediate storage.

## Selected Defaults

- `scf.max_cycle = 10000`
- `scf.e_tol = 1e-12`
- `scf.fds_sdf_tol = 1e-8`
- `scf.d_tol = 1e-4`
- `scf.diis.space = 8`
- `scf.do_fci = false`
- `scf.h.max_cycle = 100`
- `scf.h.g_tol = 1e-10`
- `scf.h.sr1_tol = 1e-12`
- `scf.h.denom_tol = 1e-10`
- `scf.h.max_step = 0.5`
- `scf.h.line_steps = 12`
- `scf.h.line_shrink = 0.5`
- `scf.h.history = 20`
- `excit.orders = {1, 2}`
- `excit.all = false`
- `prop.dt = 1e-4`
- `prop.propagator = "unshifted"`
- `det.max_steps = 10000`
- `det.dynamic_shift = true`
- `det.dynamic_shift_alpha = 0.1`
- `det.e_tol = 1e-10`
- `qmc.initial_population = 100`
- `qmc.target_population = 100000`
- `qmc.shift_damping = 5e-4`
- `qmc.ncycles = 10`
- `qmc.nreports = 1000`
- `qmc.excitation_gen = "uniform"`
- `qmc.seed = nil`
- `snoci.sigma = 1e-6`
- `snoci.tol = 1e-8`
- `snoci.imag_shift = {0.0}`
- `snoci.max_iter = 100`
- `snoci.max_add = 1`
- `snoci.max_dim = 100`
- `snoci.preconditioner = "woodbury"`
- `snoci.gmres.max_iter = 100`
- `snoci.gmres.res_tol = 1e-8`
- `snoci.gmres.metric_tol = 1e-8`
- `snoci.gmres.restart = 200`
- `snoci.gmres.full_m = false`
- `write.verbose = true`
- `write.write_dir = "outputs/"`
- `write.write_deterministic_coeffs = false`
- `write.write_orbitals = false`
- `write.write_excitation_hist = false`
- `write.write_matrices = false`
- `write.write_restart = nil`
- `write.read_restart = nil`
- `wicks.enabled = true`
- `wicks.compare = false`
- `wicks.storage = "ram"`
- `wicks.cachedir = "."`

## References

[1] Tracy P Hamilton and Peter Pulay. Direct Inversion in the Iterative Subspace (DIIS) Optimization of Open-shell, Excited-state, and Small Multiconfiguration SCF Wavefunctions. *The Journal of Chemical Physics*, 84(10):5728-5734, 1986.

[2] Alex J. W. Thom and Martin Head-Gordon. Locating Multiple Self-consistent Field Solutions: An Approach Inspired by Metadynamics. *Physical Review Letters*, 101(19):193001, 2008.

[3] Andrew T. B. Gilbert, Nicholas A. Besley, and Peter M. W. Gill. Self-Consistent Field Calculations of Excited States Using the Maximum Overlap Method (MOM). *The Journal of Physical Chemistry A*, 112(50):13164-13171, 2008.

[4] Istvan Mayer. *Simple Theorems, Proofs, and Derivations in Quantum Chemistry*. Springer Science & Business Media, 2003.

[5] Hugh G. A. Burton. Generalized Nonorthogonal Matrix Elements. II: Extension to Arbitrary Excitations. *The Journal of Chemical Physics*, 157(20), 2022.

[6] Adam A. Holmes, Hitesh J. Changlani, and C. J. Umrigar. Efficient Heat-Bath Sampling in Fock Space. *Journal of Chemical Theory and Computation*, 12(4):1561-1571, 2016.

[7] Hugh G. A. Burton and Alex J. W. Thom. Reaching full correlation through nonorthogonal configuration interaction: A second-order perturbative approach. *Journal of Chemical Theory and Computation*, 16(9):5586-5600, 2020.
