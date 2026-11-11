# Practical advice

This page collects experience from running `noci-rs`. Most of it concerns NOCIQMC, where the
choices interact most.

## Choosing references

How to pick good references for the post-NOCI methods is largely unexplored, so treat these
points as observations rather than rules.

- **Spin-flipped pairs.** The stretched hydrogen chains studied so far used RHF and the
  spin-flipped pair of UHF solutions, `(+, -)` and `(-, +)`. Using both members of a pair keeps the
  spin symmetry that each one breaks on its own.
- **Watch for duplicates.** Near a coalescence point two recipes can converge to the same
  solution. Duplicates within `scf.d_tol` are removed automatically, but nearly identical solutions
  still make the overlap ill-conditioned.
- **Use holomorphic states for scans** through regions where real solutions disappear. They keep
  the NOCI basis continuous along the scan. They are supported by reference NOCI, NOCI-PT2 and
  deterministic NOCIQMC, but not by stochastic NOCIQMC.
- **More references are not always better.** Extra references enlarge the overcomplete null space
  of the extended space. In JM-NOCCMC, more references than target states cause intruders.

## Choosing a NOCIQMC propagator

- **`b-apply`** is the default choice for extended spaces. It is safe in overcomplete spaces and
  has no slowing from the overlap. Use it with `pchb` generation.
- **`s-apply`** is also safe in overcomplete spaces and works directly in the NOCI basis, with
  overlap-weighted generation. Its extra overlap action slows the small-eigenvalue directions of an
  ill-conditioned overlap, so it equilibrates more slowly than $B$-apply in such spaces.
- **`unshifted`** exists for those who are confident that their extended space has no null space,
  and who can therefore use a sparse coefficient vector. With an exact null space its population
  there grows without bound and the run diverges.

The overlap spectrum shows whether a space has a null space, but only as a diagnostic: it needs the
dense overlap and its exact diagonalisation, so it is feasible only for small spaces or smaller
analogues of a production space, never for production runs themselves. Write the matrices of a
one-step deterministic run and diagonalise the overlap:

```lua
-- One deterministic B-apply step, only to build the extended-space matrices.
prop = { dt = 1e-4, propagator = "b-apply" }
det = { max_steps = 1, dynamic_shift = false }

-- Write the Hamiltonian and overlap as npy files under data/.
write = { write_matrices = true, matrix_format = "npy" }
```

```python
import numpy as np

# Load the dense overlap of the extended space.
s = np.load("data/OVLP.npy")

# Print its smallest eigenvalues and its condition number.
w = np.linalg.eigvalsh(s)
print(w[:5], w[-1] / w[0])
```

An eigenvalue at round-off level means an exact null direction, and `unshifted` will fail.

## Choosing the time step

- **The time step is limited by stability, not noise.** Propagation is stable only if
  $\Delta\tau < 2/\lambda_\text{max}$, where $\lambda_\text{max}$ is the largest eigenvalue of the
  propagated operator, for example $H_{wx} - E_s^S S_{wx}$ for $B$-apply. Above the limit the population grows
  without control, typically to the point of exhausting memory.
- **The limit shrinks with system size and basis**, because $\lambda_\text{max}$ grows. A time
  step that is stable for a short chain can diverge for a longer one.
- **Scan for the largest stable value** with short runs, then back off a little. Do not guess
  large values.
- A larger `dt` reaches equilibrium in fewer iterations but usually needs a larger population, so
  it trades imaginary time for population rather than giving a free speedup.

## Populations and the shoulder

- **Start small.** `initial_population` of 1–10 lets the shoulder form cleanly and makes its
  height meaningful.
- **Set `target_population` above the shoulder.** Run once with a large target to find the shoulder
  height (`scripts/plotting.py shoulder`), then choose a target comfortably above it.
- **A plateau is not a stall.** Before population control starts, the population levels off at the
  shoulder for a while, often with large swings in the projected energy, and then resumes growing
  on its own. Wait it out rather than restarting or changing the shift.
- **Use `initial_shift = "noci"`** (the default). A shift close to the ground-state energy slows the
  early growth, so the population overshoots the target less before the controller takes over.
- **If the population overshoots the target badly**, increase `shift_damping` so the controller
  responds faster.

## Excitation generators and FRI

- **`pchb` reduces the shoulder of $B$-apply** compared with `uniform`, typically by a factor of 1.5–3
  in our tests, at a modest cost per spawn.
- **A generator only matters for spawns above `fri.spawn.cutoff`.** Below the cutoff, spawns are
  rounded stochastically and the rounding noise does not depend on the generator. If almost no
  spawns exceed the cutoff, changing the generator changes nothing.
- **`heat-bath`** is exact but very expensive, and is rarely worth it.
- **If the population blows up at a time step that should be stable**, increase
  `fri.pre_overlap.target_nnz`. Too small a target makes the compressed pre-overlap change too
  noisy at large time steps.

## Analysing stochastic output

- **Block the projected energy** with `scripts/blocking.py`. Pass a run and all its restarts in
  order. The automatic start uses the shoulder, an MSER criterion and the trial overlap, and
  requires the blocked window to be at least as long as the time taken to reach population
  control.
- **Check the trial overlap.** `scripts/plotting.py trial-overlap` plots `EProjDen/NRange`. It should
  be flat over the blocked window. A drift means the wavefunction is still relaxing, even if the
  projected energy looks converged.
- **Use a larger trial if the projected energy is noisy.** `n_projected` adds the determinants with
  the largest residuals to the trial state.
- **Compare with a deterministic run** for small systems. `det` gives the noise-free trajectory of
  the same propagator, so it separates time-step and slow-mode effects from stochastic bias.
- **Keep restarts.** Set `write_restart_interval`; no restart is written when a run finishes on
  its own.

## NOCI-PT2 and selected NOCI

- **Use singles and doubles**, `excit.orders = {1, 2}`, for NOCI-PT2. Higher orders are accepted,
  but the rigorous first-order interacting space is singles and doubles, and a perturbation theory
  in higher excitations would constitute a new research direction.
- **If the energy diverges along a scan**, an intruder candidate is nearly degenerate with the
  reference. Add a small imaginary shift to `snoci.imag_shift` and check how the energy depends on
  it.
- **For large candidate spaces**, keep `full_m = "none"` so products are formed on demand, and move
  the factor tables and Wick cache to disk.

## Performance

- **Keep Wick intermediates enabled.** They are much faster for excited determinants and are
  required by $B$-apply with several references and by the factorised paths.
- **Move large caches to disk** with `wicks.storage = "disk"` and point `wicks.cachedir` at fast
  local scratch, not a network file system.
- **Match `MAXEXCIT` to your excitation ranks.** Pairs within `MAXEXCIT` use specialised kernels;
  others fall back to generic code.
- **Balance MPI ranks and Rayon threads.** One rank per socket or node with Rayon threads on the
  remaining cores is a good start; `scripts/scaling.py` measures the best split for an input. Set
  `OPENBLAS_NUM_THREADS=1` so BLAS does not oversubscribe the cores.
