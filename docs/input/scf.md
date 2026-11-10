# SCF convergence: `scf`

The optional `scf` table controls convergence of the RHF and UHF calculations, and duplicate
detection between solutions.

```lua
scf = {
    -- Converge every SCF tightly in energy and commutator norm.
    max_cycle = 10000,
    e_tol = 1e-12,
    fds_sdf_tol = 1e-8,

    -- Treat solutions closer than this as duplicates.
    d_tol = 1e-4,

    -- Extrapolate from the last eight Fock matrices.
    diis = { space = 8 },
}
```

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `max_cycle` | integer | `10000` | Maximum SCF iterations. Also caps each holomorphic SCF solve and the ResHF iterations. |
| `e_tol` | number | `1e-12` | Energy-change threshold in Hartree. |
| `fds_sdf_tol` | number | `1e-8` | Threshold on the commutator norm $\lVert\mathbf F^\omega\mathbf D^\omega\mathbf S - \mathbf S\mathbf D^\omega\mathbf F^\omega\rVert$, summed over spins in the orthonormalised basis ([SCF states](../methods/scf.md#conventional-scf)). An SCF has converged when both this and `e_tol` are satisfied. |
| `d_tol` | number | `1e-4` | Density distance below which two solutions are duplicates. Duplicates are excluded from the NOCI basis, and collapsed solutions are rejected in metadynamics. |
| `diis.space` | integer | `8` | Maximum DIIS subspace size. DIIS is always used. |

ResHF uses `e_tol`, `fds_sdf_tol`, `max_cycle` and `diis.space` for its own convergence.

The holomorphic SCF solver has no input options beyond `max_cycle`: its gradient tolerance,
quasi-Newton history, step limit and line search are fixed in the code.
