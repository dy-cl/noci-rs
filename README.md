# NOCI-rs
Stochastic Non-orthogonal Configuration Interaction in Rust. 
## Features
- **Non-Orthogonal Configuration Interaction (NOCI)**
  - Generation of UHF and RHF SCF with DIIS acceleration [1, 2, 3].
  - SCF Metadynamics [4] and Maximum Orbital Overlap [5] methods for finding SCF solutions.
  - NOCI matrix element generation using generalised Slater-Condon rules [6] or extended non-orthogonal Wick's theorem [7].

- **Stochastic NOCI (NOCI-QMC)**
  - Stochastic propagation in non-orthogonal basis with up to double excitations of the non-orthogonal references.
  - Choice of various propagators with different null-space and energy shift behaviours.
  - Deterministic propagation option for debugging and benchmarking.
  - MPI-parallelism over Hilbert space.

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
    --MOM and metadynamics may not be used simultaneously.
    --mom = {
    --    {label = "RHF (0, 0)", noci = true},
    --    {label = "UHF (1, -1)", spin_bias = {pattern = {1, -1}, pol = 0.75}, noci = true},
    --    {label = "UHF (-1 , 1)", spin_bias = {pattern = {-1, 1}, pol = 0.75}, noci = true},
    --},
    metadynamics = {
        nstates_rhf = 1,
        nstates_uhf = 2,
        spinpol = 0.75,
        spatialpol = 0,
        lambda = 1,
        max_attempts = 5,
    },
}
```

## References

[1] Péter Pulay. Convergence Acceleration of Iterative Sequences. The Case of SCF Iteration. Chemical Physics letters, 73(2):393–398, 1980.

[2] Peter Pulay. Improved SCF Convergence Acceleration. Journal of Computational Chemistry, 3(4):556–560, 1982.

[3] Tracy P Hamilton and Peter Pulay. Direct Inversion in the Iterative Subspace (DIIS) Optimization of Open-shell, Excited-state, and Small Multiconfiguration SCF Wavefunctions. The Journal of Chemical Physics, 84(10):5728–5734, 1986.

[4] Alex JW Thom and Martin Head-Gordon. Locating Multiple Self-consistent Field Solutions: An Approach Inspired by Metadynamics. Physical Review Letters, 101(19):193001, 2008.

[5] Andrew T. B. Gilbert, Nicholas A. Besley, and Peter M. W. Gill. Self-Consistent Field Calculations of Excited States Using the Maximum Overlap Method (MOM). The Journal of Physical Chemistry A 112 (50), 13164–13171 (2008).

[6] István Mayer. Simple Theorems, Proofs, and Derivations in Quantum Chemistry. Springer Science & Business Media (2003).

[7] Hugh G. A. Burton. Generalized Nonorthogonal Matrix Elements. II: Extension to Arbitrary Excitations. The Journal of Chemical Physics 157 (20) (2022).
