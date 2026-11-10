# Coupled cluster: `noccmc`, `jmnoccmc`

These tables run the experimental [nonorthogonal coupled-cluster](../methods/nocc.md) methods.
Both run only with real references.

## `noccmc`

GNOCC needs a build with `--features gnocc`. Without it the table is checked and then ignored.

```lua
noccmc = {
    active_space_tol = 1e-6,
    max_cumulant = 4,
    fois_tol = 4e-2,

    max_macro = 100,
    residual_tol = 1e-8,

    holomorphic = true,
    newton = "gmres",
    preconditioner = "linear",

    gmres = {
        max_iter = 200,
        restart = 200,
        res_tol = 1e-10,
    },
}
```

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `active_space_tol` | number | `1e-6` | Natural orbitals within this of two or zero electrons are core or virtual; the rest are active. |
| `max_cumulant` | integer | `4` | Highest cumulant rank $k$ kept in the energy and residual, the GNOCCSD($k$) truncation. Ranks 1–4 are implemented. |
| `fois_tol` | number | `4e-2` | Metric eigenvalue threshold for removing redundant excitations. |
| `max_macro` | integer | `100` | Maximum Newton iterations. |
| `residual_tol` | number | `1e-8` | Convergence threshold on the norm of the projected residual. |
| `holomorphic` | boolean | `true` | Solve with complex amplitudes, following the solution onto the complex plane where no real solution exists. |
| `newton` | string | `"gmres"` | Newton-step solver: `"gmres"`, matrix-free, or `"dense"`, which builds the projected Jacobian and suits only small active spaces. |
| `preconditioner` | string | `"linear"` | GMRES preconditioner: `"linear"`, the inverse of the exact linear Jacobian, built once; or `"diagonal"`, level-shifted orbital-energy denominators. |
| `level_shift` | number | `0.5` | Level shift in $E_\text{h}$ added to the denominators of the `"diagonal"` preconditioner. No effect with `"linear"`. |
| `gmres.max_iter` | integer | `200` | Maximum GMRES iterations per Newton step. |
| `gmres.restart` | integer | `200` | GMRES restart dimension. |
| `gmres.res_tol` | number | `1e-10` | Floor of the GMRES tolerance. Each step is solved to a tenth of the residual norm, but never more tightly than this. |

## `jmnoccmc`

Without a nested `qmc` table, JM-NOCCMC propagates deterministically on one rank. With one, it
runs stochastically on every rank.

```lua
jmnoccmc = {
    cluster_rank = 2,
    expansion_rank = nil,
    dt = 0.1,

    max_steps = 10000,
    residual_tol = 1e-9,
    e_tol = 1e-10,
    dynamic_shift = true,
    dynamic_shift_alpha = 0.1,

    diagnostics = false,
    null_eps = 1e-8,
    null_n = 10,
}
```

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `cluster_rank` | integer | `2` | Highest excitation rank of each reference's cluster operator; 2 gives singles and doubles. |
| `expansion_rank` | integer | every rank | Highest rank in the expansion of each $e^{\hat T_x}\lvert{}^x\Psi\rangle$. At least `cluster_rank` + 2. Lower values truncate only the overlaps between references. |
| `dt` | number | `0.1` | Imaginary time step, independent of `prop.dt`. |
| `max_steps` | integer | `10000` | Maximum deterministic steps. |
| `residual_tol` | number | `1e-9` | Deterministic convergence requires $\max\lvert R_{x\mu}\rvert$ below this… |
| `e_tol` | number | `1e-10` | …and the energy change per step below this. |
| `dynamic_shift` | boolean | `true` | Update the deterministic shift from the population. |
| `dynamic_shift_alpha` | number | `0.1` | Damping $\alpha$ of the deterministic shift. |
| `diagnostics` | boolean | `false` | Diagnose the Jacobian at the initial and final amplitudes and write `<write_dir>/jacobian_history`. Small systems only. |
| `null_eps` | number | `1e-8` | Relative cutoff for near-null Jacobian singular values. |
| `null_n` | integer | `10` | Number of lowest singular values printed with their external fractions. |
| `memory_effective` | number | `100` | Stochastic only: memory, in reports, of the running estimate of $G$. At least 1. |
| `qmc` | table | none | Selects stochastic propagation. |

`jmnoccmc.qmc` takes the [`qmc` keys](propagation.md#qmc) that apply to it: `seed`,
`initial_population`, `target_population`, `initial_shift`, `shift_damping`,
`population_restoring`, `ncycles`, `nreports`, `fri.population.cutoff` and `fri.spawn.cutoff`.
Only the `uniform` generator is allowed, `n_projected` must be omitted because only reference
trials give the JM-NOCCMC energy, and `momentum_beta` must be zero. JM-NOCCMC does not use `prop`,
`excit` or the restart options.
