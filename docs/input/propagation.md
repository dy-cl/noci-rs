# NOCIQMC: `prop`, `det`, `qmc`

These tables run NOCIQMC in the extended space set by [`excit`](excit.md). `prop` is shared by
both modes; `det` runs the [deterministic propagation](../methods/nociqmc.md#deterministic-propagation)
and `qmc` the [stochastic propagation](../methods/nociqmc.md#stochastic-propagation). Both may be
present, and `det` runs first.

## `prop`

```lua
prop = {
    dt = 1e-4,
    propagator = "b-apply",
}
```

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `dt` | number | `1e-4` | Imaginary time step $\Delta\tau$ in $E_\text{h}^{-1}$. |
| `propagator` | string | `"unshifted"` | `"unshifted"`, `"s-apply"` or `"b-apply"`. See [the propagators](../methods/nociqmc.md#imaginary-time-propagation-in-a-nonorthogonal-space). `"unshifted"` needs a full-rank overlap. |

`prop` is required with `det` or `qmc`, but is only checked when propagation starts, after the
SCF and reference NOCI have run.

## `det`

```lua
det = {
    max_steps = 10000,
    e_tol = 1e-10,

    dynamic_shift = true,
    dynamic_shift_alpha = 0.1,
}
```

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `max_steps` | integer | `10000` | Maximum propagation steps. If it is reached, the last vector is used. |
| `e_tol` | number | `1e-10` | Stop when the projected energy changes by less than this between steps. A change above 10 $E_\text{h}$ stops the run as divergent. |
| `dynamic_shift` | boolean | `false` when omitted | Update the shift every step from the population of the propagated vector. Without it the shift stays at the reference NOCI energy. **Set it explicitly**: see [Parsing rules](index.md#parsing-rules). |
| `dynamic_shift_alpha` | number | `0.1` | Damping $\alpha$ of the dynamic shift. |
| `projector_eps` | number | `1e-12` | Overlap eigenvalue cutoff, relative to the largest, that separates the relevant and null subspaces. Positive. |
| `canonical_states_n` | integer | `10` | Number of dominant canonical states printed. Positive. |
| `canonical_terms_m` | integer | `10` | Number of basis terms printed per canonical state. Positive. |

The relevant/null-space diagnostics and the canonical-state output need
`write.write_deterministic_coeffs = true`, which also starts the propagation from a uniform vector
instead of the reference NOCI state; `projector_eps` and the `canonical_*` keys have no effect
without it.

## `qmc`

```lua
qmc = {
    initial_population = 10,
    target_population = 1e5,
    initial_shift = "noci",
    shift_damping = 5e-4,

    ncycles = 10,
    nreports = 1000,

    excitation_gen = "pchb",
    seed = 92774801300236626,

    fri = {
        population = { cutoff = 1.0 },
        spawn = { cutoff = 0.25 },
        pre_overlap = { target_nnz = 2048 },
        shift_tangent = { target_nnz = 1024 },
    },
}
```

### Populations and shift

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `initial_population` | number | `100` | 1-norm of the starting population: the coefficients for `unshifted`, the range population $\mathbf N = \mathbf S\mathbf c$ for `s-apply` and `b-apply`. |
| `target_population` | number | `1e5` | Population at which shift control starts; with `population_restoring` > 0, also the population it restores towards. Should lie above the shoulder. |
| `initial_shift` | string or number | `"noci"` | Shift before control starts: `"noci"` for the reference NOCI energy, `"hf"` for the energy of the first reference (normally RHF), or a number in $E_\text{h}$. A restart overrides it. |
| `shift_damping` | number | `5e-4` | Damping $\zeta$ of the shift update. |
| `population_restoring` | number | `0.0` | Restoring strength $\kappa$, with $0 \le \kappa < 1$. `0` holds the population at zero growth. Used by `s-apply` and `b-apply`; the unshifted controller ignores it. |
| `momentum_beta` | number | `0.0` | Report-level heavy-ball momentum $\beta$, with $0 \le \beta < 1$. `b-apply` only. |

### Run length and estimators

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `ncycles` | integer | `10` | Iterations per report. The shift is updated once per report. |
| `nreports` | integer | `1000` | Number of reports; the run lasts `ncycles` × `nreports` iterations. |
| `n_projected` | integer | number of references | Size of the projected-energy trial state. Larger values add the determinants with the largest residuals and re-solve NOCI in the trial space. At least the number of references and at most the size of the space. |
| `seed` | integer | random | Base random seed, from which every rank's streams derive. A restart's stored seed overrides it, with a warning. |

### Excitation generation

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `excitation_gen` | string | `"overlap-weighted"` for `s-apply`, otherwise `"uniform"` | Spawning proposal; see the table below. |
| `overlap_weight` | number | `0.5` for `s-apply` with `overlap-weighted`, otherwise `0.0` | Probability $0 \le p < 1$ of the overlap branch of `overlap-weighted`; the rest is uniform. |
| `optimise_overlap_weight` | boolean | `false` | Adapt `overlap_weight` between reports from the sampled second moment. Requires `overlap-weighted`. |
| `factor_tables` | string | `"ram"` | Storage of the overlap-factor tables of the unshifted `overlap-weighted` generator: `"none"`, `"ram"` or `"disk"`. Also the default of the next two keys. |
| `sapply_factor_tables` | string | `factor_tables` | S-apply overlap factors and proposal tables. Must be `"ram"` or `"disk"` with `overlap-weighted`. |
| `bapply_factor_tables` | string | `factor_tables` | B-apply auxiliary $\mathbf B^\dagger$ tables. |

Disk-backed tables are written to [`wicks.cachedir`](wicks.md).

| Generator | `unshifted` | `s-apply` | `b-apply` |
| --- | --- | --- | --- |
| `uniform` | Default | Yes | Default |
| `overlap-weighted` | Yes | Default | No |
| `heat-bath` | Yes, very expensive | No | No |
| `pchb` | No | No | Yes |

S-apply rejects `heat-bath` because its shift tangent needs separately realised overlap elements.
The generators are described under
[Excitation generators](../methods/nociqmc.md#excitation-generators).

### FRI compression

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `fri.population.cutoff` | number | `1.0` | Fixed cutoff for sampling the persistent population each iteration. Non-negative. |
| `fri.spawn.cutoff` | number | `0.25` | Fixed cutoff for individual spawned changes. Non-negative. |
| `fri.pre_overlap.target_nnz` | integer | `2048` | Per-rank target nonzeros of each report's pre-overlap change. `s-apply` and `b-apply` only. Positive. |
| `fri.shift_tangent.target_nnz` | integer | `1024` | Per-rank target nonzeros of the shift tangent. `s-apply` and `b-apply` only. Positive. |

The cutoff sites accept only `cutoff`, and the target sites only `target_nnz`. The old root-level
keys `sampling_cutoff`, `sampling_cutoff1`, `sampling_cutoff2` and `spawn_cutoff` are rejected with
a message pointing to `qmc.fri`.

### Output columns

Each report prints one row:

| Column | Meaning |
| --- | --- |
| `Iter` | Iteration |
| `EProjNum`, `EProjDen` | Numerator and denominator of the projected energy |
| `EProj` | Projected energy |
| `ECorr` | Projected correlation energy relative to the first reference |
| `EShift` | Shift, shown as zero before population control starts |
| `NWalk`, `NRef` | `unshifted`: total and reference population |
| `NRange`, `NRangeRef` | `s-apply`, `b-apply`: range population and its reference part |
| `NSample`, `NSampleOcc` | `s-apply`, `b-apply`: sampled population and its number of occupied determinants |

### Run-time checks

- `write.write_restart_interval` must be a nonzero multiple of `ncycles`.
- `b-apply` needs orthonormal orbitals within every parent, and Wick intermediates when there is
  more than one parent.
- Stochastic NOCIQMC is skipped, with a warning, for holomorphic references.
