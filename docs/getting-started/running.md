# Running calculations

## Parallel execution

`noci-rs` combines distributed-memory MPI parallelism with shared-memory Rayon parallelism
within each MPI process. On one rank:

```bash
RAYON_NUM_THREADS=8 ./target/release/noci-rs input.lua > output.out
```

Across several ranks:

```bash
RAYON_NUM_THREADS=8 mpirun -np 4 ./target/release/noci-rs input.lua > output.out
```

| Method | Rayon | MPI |
| --- | --- | --- |
| Full Hamiltonian and overlap construction | Determinant pairs | — |
| Selected NOCI and NOCI-PT2 | Matrix elements within a rank | Candidate-space operator applications |
| Stochastic NOCIQMC | Spawning within a rank | Determinant ownership and update exchange |
| Wick intermediates | — | Shared memory between ranks on each node |

Set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` when Rayon already uses every core, so that
BLAS calls do not oversubscribe the node.

## Output

All reports go to standard output. Their detail is set by `write.verbose`: `0` prints only the
results, `1` adds progress and diagnostics, and `2` adds raw numerical arrays and matrices.
Optional files are written to `write.write_dir`:

- converged orbitals in HDF5 (`write_orbitals`);
- the Hamiltonian and overlap matrices, `HAMI` and `OVLP` as text or `HAMI.npy` and `OVLP.npy`
  with `matrix_format = "npy"` (`write_matrices`);
- deterministic coefficient histories (`write_deterministic_coeffs`);
- spawning-magnitude histograms (`write_excitation_hist`).

See [the `write` table](../input/write.md) for the full list.

## Stochastic restarts

A stochastic calculation writes a restart file, `write.write_restart`, every
`write.write_restart_interval` iterations, which must be a multiple of `qmc.ncycles`. Creating a
file named `STOP` in the working directory writes a restart and ends the run cleanly. No restart
is written when a run simply finishes, so set an interval if you may want to continue it.

Restarting with `write.read_restart` restores the populations, shift and population-control
state, so the run continues where it stopped. S-apply and B-apply share one population
representation, so a restart written by one can be read by the other.

## Analysis scripts

`scripts/` holds the Python analysis tools.

`blocking.py` performs Flyvbjerg–Petersen blocking {cite}`flyvbjerg1989` of the stochastic
projected energy. It takes one or more consecutive output files, for example a run and its
restarts. Unless `--start` is given, it picks the start of the equilibrated window automatically
from the population shoulder, an MSER criterion and the flatness of the trial overlap:

```bash
python3 scripts/blocking.py run.out restart1.out
python3 scripts/blocking.py run.out --start 200000
```

`plotting.py` draws one figure per subcommand, shown on screen or written with `--save`:

| Subcommand | Plot |
| --- | --- |
| `energy` | Energies across a geometry scan |
| `projected-shift` | Projected-energy and shift correlation energies against iteration |
| `nw` | Persistent and sampled populations against iteration |
| `shoulder` | Ratio of total to reference population against population, to locate the shoulder |
| `trial-overlap` | Normalised projected-energy denominator `EProjDen/NRange` against iteration |
| `trial-overlap-relaxation` | Its relaxation relative to the late-time mean |
| `excitation-hist` | Spawning-magnitude histograms from `write_excitation_hist` |
| `deterministic-projected-shift` | Deterministic energy and shift against iteration |
| `deterministic-population` | Deterministic population of the propagated vector |
| `deterministic-coefficients` | Deterministic relevant and null-space coefficients |
| `matrix` | Heatmap of a written Hamiltonian or overlap matrix |

`scaling.py` times one input over a grid of MPI rank and Rayon thread counts.

[Practical advice](../guide/advice.md#analysing-stochastic-output) explains how to read the
stochastic output and choose the blocking window.
