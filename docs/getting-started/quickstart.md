# Quick start

This example runs reference NOCI on stretched H{sub}`2` in a basis of three SCF determinants:
the RHF determinant and the two spin-symmetry-broken UHF determinants.

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

The same input runs across several MPI ranks, each with its own Rayon threads:

```bash
RAYON_NUM_THREADS=4 mpirun -np 2 ./target/release/noci-rs h2.lua > h2.out
```

The output reports each SCF solution, then the reference NOCI energies obtained from
$({}^{wx}H - E\,{}^{wx}S)c_x = 0$ in the basis of the three determinants.

[Running calculations](running.md) covers parallel execution, output files and restarts.

## More calculation types

- Add `excit`, `prop` and `qmc` tables to propagate in the excited space of the references by
  [NOCIQMC](../methods/nociqmc.md).
- Add `excit` and `snoci` tables to correct the reference NOCI energy with
  [NOCI-PT2](../methods/selected.md).
- `inputs/examples/` holds a holomorphic NOCI-PT2 scan of F{sub}`2`, which produces the
  [F{sub}`2` result](../results.md).

The [input reference](../input/index.md) lists every table.
