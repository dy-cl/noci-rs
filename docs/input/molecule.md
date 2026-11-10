# Molecule: `mol`

The required `mol` table defines the molecule, basis set and geometry scan.

```lua
mol = {
    basis = "cc-pVDZ",
    unit = "Ang",
    r = {0.8, 1.0, 1.2},

    atoms = function(r)
        return {
            string.format("H 0 0 %g", -r / 2),
            string.format("H 0 0 %g",  r / 2),
        }
    end,
}
```

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `basis` | string | required | Basis set name, read from the Basis Set Exchange data in `BSE_DATA_DIR`. Spherical harmonics are always used. |
| `unit` | string | required | Coordinate unit: `"Ang"`, `"ANG"`, `"angstrom"` or `"Angstrom"` for Ångström; `"Bohr"`, `"BOHR"`, `"au"`, `"AU"`, `"a.u."` or `"A.U."` for Bohr. |
| `r` | float or list of numbers | required | Scan coordinate. One calculation runs per entry, in order, each seeded from the previous geometry. A scalar must be a Lua float, so write `1.0`, not `1`. |
| `atoms` | list of strings, or function of `r` | required | Atom lines `"<Element> x y z"`. A function is called once per entry of `r`; a static list is reused for every entry. |

:::{note}
There is no charge or spin input. Each spin receives half of the total nuclear charge, so only
neutral, closed-shell-count ($M_s = 0$) systems with an even number of electrons are supported.
:::

The atom order of `atoms` is the order used by the `pattern` lists of
[`spin_bias` and `spatial_bias`](states.md#bias-tables).
