# NOCI-rs
Stochastic Non-orthogonal Configuration Interaction in Rust. 
## Features
- **Non-Orthogonal Configuration Interaction (NOCI)**
  - Generation of UHF and RHF SCF solutions with density biasing options, DIIS acceleration and maximum orbital overlap for excited-state SCF.
  - Generalised eigenvalue problem (GEVP) solution for non-orthogonal basis of SCF solutions.

- **Stochastic NOCI (NOCI-QMC)**
  - Stochastic propagation in non-orthogonal basis with up to double excitations of the non-orthogonal references.
  - Choice of various propagators with different null-space and energy shift behaviours.
  - Deterministic propagation option for debugging.
  - MPI-parallelism of Hilbert space.

- **Output**
  - Energies, shifts, populations and timings.
  - Excitation generation probability histograms.
  - Deterministic propagation coefficient history.
  - MOs of SCF solutions.

## Installation and Compilation 

### Requirements
- **Rust**
- **HDF5 (< 1.14)**
- **MPICC**
- **Python**
  - **PySCF**

### Build
```bash
git clone https://github.com/dy-cl/noci-rs
cd noci-rs
cargo build --release
```

### Running
```bash
mpirun -np X ./target/release/noci-rs input.lua
```

## Example Input File
```lua
scf = {
    max_cycle = 1e4, 
    e_tol = 1e-8, 
    diis = {space = 8,},
    do_fci = false,
}

write = {
    verbose = true,
    write_deterministic_coeffs = false,
    write_excitation_hist = true,
    write_dir = 'write_dir',
}

mol = {
    basis = 'cc-pVDZ',
    r = {1.5},
    unit = 'Ang',
    atoms = function(r)
        return {string.format("H 0 0 %g", -r / 2), string.format("H 0 0 %g",  r / 2),}
        end,
}

excit = {
    singles = true,
    doubles = true,
}

prop = {
    dt = 1e-3,
    max_steps = 1e5,
    propagator = "doubly-shifted",
}

det = {
    dynamic_shift = true,
    dynamic_shift_alpha = 0.001,
    e_tol = 1e-12, 
}

qmc = {
    initial_population = 1e3,
    target_population = 5e4,
    shift_damping = 0.05,
    shift_update_freq = 10,
    excitation_gen = 'uniform',
}

states = {
    {label = "RHF (0, 0)", noci = true},
    {label = "UHF (1, -1)", spin_bias = {pattern = {1, -1}, pol = 0.75}, noci = true},
    {label = "UHF (-1 , 1)", spin_bias = {pattern = {-1, 1}, pol = 0.75}, noci = true},
}
```
