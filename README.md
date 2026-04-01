
# noci-rs
`noci-rs` is a Rust implementation of non-orthogonal configuration interaction (NOCI) and stochastic NOCI (NOCI-QMC) for molecular systems where a non-orthogonal determinant basis is advantageous.

[![tests](https://github.com/dy-cl/noci-rs/actions/workflows/tests.yml/badge.svg)](https://github.com/dy-cl/noci-rs/actions/workflows/tests.yml)
## Features

- **Non-Orthogonal Configuration Interaction (NOCI)**
  - Generation of UHF and RHF SCF states with DIIS acceleration [1].
  - SCF Metadynamics [2] and Maximum Orbital Overlap [3] methods for finding SCF solutions.
  - Generation of excited-state NOCI bases using user-specified excitation orders.
  - NOCI matrix element generation using generalised Slater-Condon rules [4] or extended non-orthogonal Wick's theorem [5].
  - Rayon-parallelised matrix element calculations.

- **Stochastic NOCI (NOCI-QMC)**
  - Choice of various propagators with different null-space and energy shift behaviours.
  - Combined Rayon and MPI parallelism allows for computation across multiple nodes.
  - Uniform and heat-bath [6] excitation generation schemes.

- **Selected NOCI (SNOCI)**
  - Selected NOCI (SNOCI) with iterative candidate generation and determinant selection.
  - GMRES solve of the projected candidate-space equations.
  - NOCI-PT2 [7] based candidate scoring and selection.

- **Output**
  - SCF state energies, MO energies, MO coefficients, and $\langle S^2 \rangle$.
  - Reference NOCI Hamiltonian, overlap, and energies.
  - Stochastic and/or deterministic projected energy, energy shifts, walker populations, and timings.
  - Deterministic coefficient evolution history.
  - Excitation generation probability histograms.

## Installation and Compilation

### Requirements

- **Rust**
- **HDF5 (< 1.14)**
- **MPICC**
- **Python**
  - **PySCF**

### Obtaining and Building

```bash
git clone https://github.com/dy-cl/noci-rs
cd noci-rs
cargo build --release
```

## Usage

### Running Calculations

```bash
mpirun -np X ./target/release/noci-rs input.lua > output.out
```

### Example Input File

An example input for a cc-pVDZ $F_2$ NOCI-QMC calculation is shown below.

```lua
scf = {
    max_cycle = 1e4,
    e_tol = 1e-8,
    diis = {
        space = 8,
    },
    do_fci = false,
}

mol = {
    basis = "cc-pVDZ",
    r = {1.75},
    unit = "Ang",
    atoms = function(r)
        return {
            string.format("F 0 0 %g", -r / 2),
            string.format("F 0 0 %g",  r / 2),
        }
    end,
}

prop = {
    dt = 1e-6,
    max_steps = 5e6,
    propagator = "difference-doubly-shifted-u2",
}

qmc = {
    initial_population = 2e2,
    target_population = 4e5,
    shift_damping = 0.0005,
    shift_update_freq = 1,
    excitation_gen = "uniform",
}

states = {
    mom = {
        {label = "RHF (0, 0)", noci = true},
        {label = "UHF (1, -1)", spin_bias = {pattern = {1, -1}, pol = 0.75}, noci = true},
        {label = "UHF (-1, 1)", spin_bias = {pattern = {-1, 1}, pol = 0.75}, noci = true},
    }
}

excit = {
    orders = {1, 2},
}

wicks = {
    enabled = true,
    compare = false,
    storage = "ram",
    cachedir = "/path/to/cachedir",
}
```

Note that not every option is shown above, and most inputs have default values if they are not specified.

### Input Overview

The input file is a Lua file with the following top-level tables:

- `mol` — molecular geometry and basis specification
- `scf` — SCF convergence and DIIS settings
- `states` — definition of SCF states via `mom` or `metadynamics`
- `excit` — excited-state generation settings
- `prop` — propagation parameters for deterministic/QMC propagation
- `qmc` — stochastic propagation options
- `det` — deterministic propagation options
- `snoci` — selected NOCI options
- `write` — output and file-writing options
- `wicks` — extended non-orthogonal Wick's theorem options

### Excitation Input

Excited-state basis generation is controlled through:

```lua
excit = {
    orders = {1, 2},
}
```

Examples:

- `orders = {1}` generates singles only
- `orders = {2}` generates doubles only
- `orders = {1, 2}` generates singles and doubles
- `orders = {1, 2, 3}` generates singles, doubles, and triples

### Selected Defaults

If a table or option is omitted, defaults are used where available. Important defaults include:

- `scf.max_cycle = 10000`
- `scf.e_tol = 1e-12`
- `scf.diis.space = 8`
- `scf.do_fci = false`

- `excit.orders = {1, 2}`

- `prop.dt = 1e-4`
- `prop.max_steps = 5000000`
- `prop.propagator = "unshifted"`

- `qmc.initial_population = 100`
- `qmc.target_population = 100000`
- `qmc.shift_damping = 5e-4`
- `qmc.shift_update_freq = 1`
- `qmc.excitation_gen = "uniform"`

- `det.dynamic_shift = true`
- `det.dynamic_shift_alpha = 0.1`
- `det.e_tol = 1e-10`

- `write.verbose = true`
- `write.write_deterministic_coeffs = false`
- `write.write_orbitals = false`
- `write.write_excitation_hist = false`
- `write.write_matrices = false`
- `write.write_dir = "outputs/"`

- `wicks.enabled = true`
- `wicks.compare = false`
- `wicks.storage = "ram"`
- `wicks.cachedir = "."`

## References

[1] Tracy P Hamilton and Peter Pulay. Direct Inversion in the Iterative Subspace (DIIS) Optimization of Open-shell, Excited-state, and Small Multiconfiguration SCF Wavefunctions. *The Journal of Chemical Physics*, 84(10):5728–5734, 1986.

[2] Alex J. W. Thom and Martin Head-Gordon. Locating Multiple Self-consistent Field Solutions: An Approach Inspired by Metadynamics. *Physical Review Letters*, 101(19):193001, 2008.

[3] Andrew T. B. Gilbert, Nicholas A. Besley, and Peter M. W. Gill. Self-Consistent Field Calculations of Excited States Using the Maximum Overlap Method (MOM). *The Journal of Physical Chemistry A*, 112(50):13164–13171, 2008.

[4] István Mayer. *Simple Theorems, Proofs, and Derivations in Quantum Chemistry*. Springer Science & Business Media, 2003.

[5] Hugh G. A. Burton. Generalized Nonorthogonal Matrix Elements. II: Extension to Arbitrary Excitations. *The Journal of Chemical Physics*, 157(20), 2022.

[6] Adam A. Holmes, Hitesh J. Changlani, and C. J. Umrigar. Efficient Heat-Bath Sampling in Fock Space. *Journal of Chemical Theory and Computation*, 12(4):1561–1571, 2016.

[7] Hugh GA Burton and Alex JW Thom. Reaching full correlation through nonorthogonal configuration interaction: A second-order perturbative approach. Journal of Chemical Theory and Computation, 16(9):5586–5600, 2020.
