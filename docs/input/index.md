# Input reference

An input file is a Lua script that defines global tables. `noci-rs` runs the script, then reads
the tables below. Because the input is a full Lua program, geometries can be functions, scans can
be generated in loops, and common settings can be shared through `dofile`.

## Tables

| Table | Required | Purpose |
| --- | --- | --- |
| [`mol`](molecule.md) | Yes | Geometry, basis set, units and scan coordinate |
| [`states`](states.md) | Yes | MOM state recipes or SCF metadynamics |
| [`scf`](scf.md) | No | SCF convergence and duplicate detection |
| [`excit`](excit.md) | No | Excitation orders of the extended space |
| [`prop`](propagation.md#prop) | With `det` or `qmc` | Time step and propagator |
| [`det`](propagation.md#det) | No | Deterministic NOCIQMC |
| [`qmc`](propagation.md#qmc) | No | Stochastic NOCIQMC |
| [`snoci`](snoci.md) | No | Selected NOCI and NOCI-PT2 |
| [`reshf`](reshf.md) | No | ResHF optimisation of the references |
| [`write`](write.md) | No | Output files, verbosity and restarts |
| [`wicks`](wicks.md) | No | Nonorthogonal Wick intermediates and disk caches |
| [`gnoccmc`](nocc.md#gnoccmc) | No | GNOCC, with the `gnocc` feature |
| [`jmnoccmc`](nocc.md#jmnoccmc) | No | JM-NOCCMC, with nested `det` and `qmc` tables |

## What runs

For every geometry in `mol.r`, in order:

1. The SCF states of `states` are generated, seeded from the previous geometry.
2. If `reshf` is present, ResHF re-optimises the states with `noci = true`.
3. Reference NOCI runs over the states with `noci = true`, after duplicates are removed.
4. Every post-NOCI table present runs, in this order: `det`, `jmnoccmc`, `gnoccmc`, `snoci`,
   `qmc`.

If any recipe is holomorphic, NOCI is complex and only `det` and `snoci` run afterwards. `qmc` is
skipped with a warning, and `jmnoccmc` and `gnoccmc` are skipped silently.

## Parsing rules

:::{warning}
- **Unknown keys are ignored**, so a misspelt key silently leaves its default in place. The
  options echoed at the start of the output show the values actually used.
- **A value of the wrong type usually falls back to the default** without an error. Keys with
  explicit validation, listed on each page, are the exception.
- **An omitted boolean is `false` when its table is present.** The Lua bridge reads a missing key
  as `false` before the default can apply. This affects `states.mom[i].noci`,
  `det.dynamic_shift` and `wicks.enabled`, whose documented defaults hold only when the whole
  table is absent. Set these keys explicitly.
:::

- Integer options accept floats and truncate them, so `max_steps = 1e5` means 100000.
- Lua 5.4 distinguishes integers from floats. A scalar `mol.r` must be a float (`r = 1.0`, not
  `r = 1`), and `write.verbose` must be an integer (`1`, not `1.0`).
- Validation failures print a message and exit before any calculation starts, except where a page
  says a check happens at run time.
- `prop` is checked only when `det` or `qmc` starts, after the SCF and reference NOCI have run.
