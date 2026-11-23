# Excited-state replacement: `replace`

The `replace` table fills the place of each duplicate or unconverged SCF state with a distinct
excited SCF state of a parent determinant, so the NOCI basis keeps one determinant per recipe
with `noci = true`. Without it, such a state just leaves the NOCI basis.

```lua
-- Replace missing states by excitations of RHF, trying the HOMO to LUMO double first.
replace = {
    parent = "RHF",
    excits = {
        { spin = "both", occ = -1, vir = 0 },
    },
}
```

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `parent` | string | first state with `noci = true` | Label of the state whose excitations replace missing states. |
| `excits` | list of tables | `{}` | Excitations to try first, in order. Each table has the keys of a recipe's [`excit`](states.md). |

- A state is missing when it is a duplicate of an earlier NOCI-basis state under `scf.d_tol`, or
  when its SCF does not converge. Without `replace`, an unconverged SCF stops the run.
- After the listed excitations come generated ones, `occ = -1 - o` to `vir = v` by increasing
  `o + v`: first paired doubles (`spin = "both"`), then alpha and beta singles.
- Each excitation runs as a MOM SCF seeded from the parent's converged densities at the current
  geometry. One that is a duplicate, or does not converge, is discarded and the next is tried.
  Each excitation is used at most once per geometry.
- The replacement is labelled `<parent> excit(<spin>, <occ>, <vir>)` and follows the state it
  replaces. The duplicate itself is kept, outside the NOCI basis, to seed the next geometry.
- `replace` needs MOM recipes and cannot be combined with holomorphic recipes. With `reshf`, the
  replacements are optimised together with the other states.
