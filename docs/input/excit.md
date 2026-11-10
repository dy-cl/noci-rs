# Excitations: `excit`

The optional `excit` table sets the excitation orders that build the extended space of
deterministic and stochastic NOCIQMC, and the candidate pool of selected NOCI and NOCI-PT2.
Excitations are generated from every reference NOCI determinant, within its own orbital frame,
over every $\alpha$/$\beta$ split of each order.

```lua
-- Singles and doubles of every reference, the NOCISD space.
excit = {
    orders = {1, 2},
}
```

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `orders` | list of integers | `{1, 2}` | Excitation ranks to generate. Duplicates are removed and the list is sorted. |
| `all` | boolean | `false` | Generate every order up to the largest possible, the full space of each reference. Cannot be combined with `orders`. |

`orders = {1, 2}` gives the NOCISD space and is the first-order interacting space of NOCI-PT2.
The JM-NOCCMC and GNOCC methods set their own excitation spaces and do not read `excit`.
