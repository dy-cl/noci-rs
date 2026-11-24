# ResHF: `reshf`

The `reshf` table replaces the real SCF states marked `noci = true` with
[ResHF](../methods/noci.md#reshf-reference-optimisation) determinants before reference NOCI.

```lua
-- Optimise the references for the NOCI ground state alone.
reshf = {
    weights = {1.0},
    cutoff = 1e-8,
}
```

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `weights` | list of numbers | `{1.0}` | State weights $w_I$ of $E_\text{SA} = \sum_I w_I E_I$, lowest root first. Non-negative and summing to one; no more weights than NOCI roots. |
| `cutoff` | number | `1e-8` | Relative floor on the singular values of each occupied-orbital overlap, keeping nearly orthogonal pairs finite. Positive. |
| `space` | string | `"references"` | Determinants to optimise: `"references"`, or `"excitations"` for every reference and its excitations of the [`excit`](excit.md) orders, each with its own orbitals. |

- Convergence uses `scf.e_tol`, `scf.fds_sdf_tol`, `scf.max_cycle` and `scf.diis.space`.
- The reported commutator norm is summed over all determinants, so it grows roughly as the square
  root of their number.
- ResHF cannot be combined with holomorphic recipes.
- In a scan, the ResHF determinants seed the SCF at the next geometry.

## Optimising the excitation space

```lua
-- Optimise every determinant of the NOCISD space together.
excit = { orders = {1, 2} }
reshf = { space = "excitations" }
```

The excitations are generated as for the extended space of the post-reference methods, then
become independent determinants, so ResHF optimises the orbitals of all of them together and the
optimised space is the reference NOCI basis.

- `det`, `qmc`, `snoci`, `gnoccmc` and `jmnoccmc` would expand this basis again, and are rejected.
- Every pair of determinants is stored each iteration, so memory and time grow as the square of
  the space size.
- Pairs from the same reference are often exactly orthogonal and rely on `cutoff`.
- In a scan, the SCF states rather than the optimised determinants seed the next geometry.
