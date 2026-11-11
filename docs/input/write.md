# Output: `write`

The optional `write` table sets verbosity, output files and stochastic restarts.

```lua
write = {
    -- Print progress and diagnostics, and write files under data/.
    verbose = 1,
    write_dir = "data",

    -- Write no orbitals, and write matrices as npy if requested.
    write_orbitals = false,
    write_matrices = false,
    matrix_format = "npy",

    -- Checkpoint a stochastic run every 100000 iterations, starting fresh.
    write_restart = "run.h5",
    write_restart_interval = 100000,
    read_restart = nil,
}
```

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `verbose` | integer | `1` | `0` prints results only, `1` adds progress and diagnostics, `2` adds raw arrays and matrices. Must be an integer literal. |
| `write_dir` | string | `"data"` | Output directory, created if needed. Also the default parent of the Wick cache. |
| `write_orbitals` | boolean | `false` | Write each SCF state's orbitals as HDF5 to `<write_dir>/orbitals/`. |
| `write_matrices` | boolean | `false` | Write the dense Hamiltonian and overlap matrices to `write_dir` whenever they are built, by reference NOCI and by deterministic propagation. A later build overwrites the earlier files, so with `det` they hold the extended space. |
| `matrix_format` | string | `"text"` | `"text"` writes `HAMI` and `OVLP`; `"npy"` writes `HAMI.npy` and `OVLP.npy`, much smaller and faster for large spaces. |
| `write_deterministic_coeffs` | boolean | `false` | Deterministic NOCIQMC: write the coefficient history to `<write_dir>/coefficients` and enable the relevant/null-space diagnostics. **Also starts the propagation from a uniform vector** instead of the reference NOCI state. |
| `write_excitation_hist` | boolean | `false` | Stochastic NOCIQMC: write the spawning-magnitude histogram to `<write_dir>/excitationsamples<rank>`. |
| `write_restart` | string | `RESTART.H5` in the current directory | Restart file to write. |
| `write_restart_interval` | integer | none | Write a restart every this many iterations. Must be a multiple of `qmc.ncycles`. |
| `read_restart` | string | none | Restart file to resume from. |

## Restarts

A restart holds the populations, shift and controller history, random seed and overlap weight, so
a resumed run continues where it stopped. $S$-apply and $B$-apply restarts are
interchangeable. Reading checks that the restart matches the determinant space and the MPI
partition.

:::{note}
A restart is written every `write_restart_interval` iterations, and when a `STOP` file appears
in the working directory, which also ends the run cleanly. It is **not** written when a run
finishes normally, so set `write_restart_interval` to keep a restart of a completed run.
:::

JM-NOCCMC does not use the restart options yet.
