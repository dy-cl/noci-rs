# SCF states: `states`

The required `states` table says how SCF solutions are found. It contains exactly one of `mom`,
a list of explicit recipes, or `metadynamics`, an automatic search. See
[SCF states](../methods/scf.md) for the methods.

## `states.mom`

Each entry is a recipe for one SCF solution. Recipes are processed in order.

```lua
states = {
    mom = {
        { label = "RHF", noci = true },

        {
            label = "UHF (+, -)",
            spin_bias = { pattern = {1, -1}, pol = 0.75 },
            noci = true,
        },

        {
            label = "UHF (-, +)",
            spin_bias = { pattern = {-1, 1}, pol = 0.75 },
            noci = true,
        },

        -- Holomorphic continuation of the first UHF state.
        {
            label = "h-UHF (+, -)",
            holomorphic = true,
            partner = "UHF (+, -)",
            noci = true,
        },
    },
}
```

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `label` | string | `""` | Name of the state. It identifies the state at the next geometry, for `partner` lookups and in orbital file names. Recipes with `excit` are seeded from the previous geometry's state labelled `"RHF"`. |
| `noci` | boolean | `false` when omitted | Whether the state enters the reference NOCI basis, and is optimised by ResHF. **Set it explicitly**: see [Parsing rules](index.md#parsing-rules). |
| `mom` | boolean | `false` | Use MOM occupations for both spins in a recipe without `excit`. Recipes with `excit` always use MOM in the excited spin channel. |
| `spin_bias` | table | none | Spin polarisation of the initial guess, see [bias tables](#bias-tables). Produces spin-broken UHF states. |
| `spatial_bias` | table | none | Charge polarisation of the initial guess, see [bias tables](#bias-tables). Applied after `spin_bias` if both are given. |
| `excit` | table | none | Start from a singly excited occupation, see [`excit`](#excited-state-recipes). |
| `holomorphic` | boolean | `false` | Generate a holomorphic state by tracking from a real seed. Not allowed with `reshf`. |
| `partner` | string | the recipe's `label` | Label of the real state generated at the same geometry that seeds a new holomorphic track. Used only by holomorphic recipes. |

### Bias tables

`spin_bias` and `spatial_bias` take the same two keys.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `pattern` | list of `-1`, `0`, `1` | required | One entry per atom, in `mol.atoms` order. |
| `pol` | number | `0.0` | Polarisation strength, between 0 and 1. |

For `spin_bias`, an atom marked `1` has its $\alpha$ guess density scaled by $(1 + \text{pol})$ and
its $\beta$ density by $(1 - \text{pol})$; `-1` does the reverse. For `spatial_bias`, `1` scales
both spins by $(1 + \text{pol})$ and `-1` by $(1 - \text{pol})$. A `0` leaves the atom unchanged.

### Excited-state recipes

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `spin` | string | `"both"` | Spin channel to excite: `"alpha"`, `"beta"` or `"both"`. |
| `occ` | integer | `0` | Hole, as an offset from the number of occupied orbitals: `-1` is the HOMO, `-2` the orbital below. |
| `vir` | integer | `0` | Particle, as an offset from the number of occupied orbitals: `0` is the LUMO. |

For example, `excit = { spin = "both", occ = -1, vir = 0 }` targets the doubly excited
HOMO → LUMO solution. The default `occ = 0` points at the LUMO, which is not occupied, so always
give `occ`.

## `states.metadynamics`

```lua
states = {
    metadynamics = {
        nstates_rhf = 1,
        nstates_uhf = 2,

        spinpol = 0.75,
        spatialpol = 0.75,

        lambda = 0.5,
        max_attempts = 100,
    },
}
```

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `nstates_rhf` | integer | `0` | Number of RHF solutions to find, labelled `"RHF M k"`. |
| `nstates_uhf` | integer | `0` | Number of UHF solutions to find. They are found as spin-flipped pairs, labelled `"UHF M p A"` and `"UHF M p B"`, so use an even number: an odd count ends the search early, and `1` finds none. |
| `spinpol` | number | `0.0` | Spin-bias strength of the random UHF guesses. |
| `spatialpol` | number | `0.0` | Spatial-bias strength of the random RHF guesses. |
| `lambda` | number | `0.0` | Strength and inverse width $\lambda$ of the bias $N\lambda e^{-\lambda d^2}$ from each solution already found. |
| `max_attempts` | integer | `100` | Maximum biased SCF attempts per geometry. Reaching it prints a message and continues with the solutions found. |

Every solution found by metadynamics enters the NOCI basis. Metadynamics does not support
holomorphic states.
