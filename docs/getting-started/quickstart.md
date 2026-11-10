# Quick start

This example runs reference NOCI on stretched H{sub}`2`, where the RHF determinant and the two
spin-symmetry-broken UHF determinants are all needed to describe the bond.

## Input

Input files are Lua scripts. Save the following as `h2.lua`:

```lua
-- Stretched H2 geometry with multireference character.
mol = {
    basis = "cc-pVDZ",
    r = 1.5,
    unit = "Ang",

    atoms = function(r)
        return {
            string.format("H 0 0 %g", -r / 2),
            string.format("H 0 0 %g",  r / 2),
        }
    end,
}

-- Use MOM to obtain the RHF state and two symmetry-broken UHF states.
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
    },
}
```

`mol` gives the geometry and basis set. Each entry of `states.mom` is a recipe for one SCF
solution: the two UHF recipes bias the initial spin density towards opposite spins on the two
atoms, and `noci = true` puts the converged determinant into the reference NOCI basis.

## Run

```bash
RAYON_NUM_THREADS=4 ./target/release/noci-rs h2.lua > h2.out
```

The output reports each SCF solution, then the reference NOCI energies obtained from
$\mathbf H\mathbf c = E\mathbf S\mathbf c$ in the basis of the three determinants.

## Next steps

- Add an `excit` table and a `snoci` table to correct the reference NOCI energy with NOCI-PT2,
  or `prop` and `qmc` tables to propagate in the excited space by NOCIQMC. The
  [input reference](../input/index.md) lists every table.
- `inputs/examples/` holds a holomorphic NOCI-PT2 scan of F{sub}`2`, which produces the
  [F{sub}`2` result](../results.md).
- [Running calculations](running.md) covers parallel execution, output files and restarts.
