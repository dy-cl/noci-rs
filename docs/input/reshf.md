# ResHF: `reshf`

The `reshf` table replaces the real SCF states marked `noci = true` with
[ResHF](../methods/noci.md#reshf-reference-optimisation) determinants before reference NOCI.

```lua
reshf = {
    weights = {1.0},
    cutoff = 1e-8,
}
```

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `weights` | list of numbers | `{1.0}` | State weights $w_I$ of $E_\text{SA} = \sum_I w_I E_I$, lowest root first. Non-negative and summing to one; no more weights than NOCI roots. |
| `cutoff` | number | `1e-8` | Relative floor on the singular values of each occupied-orbital overlap, keeping nearly orthogonal pairs finite. Positive. |

- Convergence uses `scf.e_tol`, `scf.fds_sdf_tol`, `scf.max_cycle` and `scf.diis.space`.
- The reported commutator norm is summed over all determinants, so it grows roughly as the square
  root of their number.
- ResHF cannot be combined with holomorphic recipes.
- In a scan, the ResHF determinants seed the SCF at the next geometry.
