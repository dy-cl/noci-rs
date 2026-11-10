# Overview of methods

Every calculation follows the same pipeline.

1. **SCF states.** Multiple real or holomorphic RHF and UHF solutions are found by
   maximum-overlap recipes or SCF metadynamics ([SCF states](scf.md)).
2. **Reference NOCI.** The selected solutions form a nonorthogonal basis
   $\{|{}^x\Psi\rangle\}$, in which the generalised eigenvalue problem
   $\mathbf H\mathbf c = E\mathbf S\mathbf c$ gives the reference NOCI states
   ([NOCI](noci.md)). ResHF can optimise the reference determinants for that NOCI energy.
3. **Extended space.** Excitations of each reference within its own orbital frame, up to the
   orders in `excit`, build the extended space in which correlation is recovered.
4. **Post-NOCI method.** NOCI-PT2 and selected NOCI ([Selected NOCI and NOCI-PT2](selected.md)),
   deterministic or stochastic NOCIQMC ([NOCIQMC](nociqmc.md)), or nonorthogonal coupled cluster
   ([NOCC](nocc.md)) treat the extended space.

Every step needs matrix elements between determinants built from different orbital frames;
[Matrix elements](elements.md) describes how these are evaluated.

## Method summary

| Method | Input tables | Deterministic | Stochastic | Holomorphic states |
| --- | --- | --- | --- | --- |
| Reference NOCI | `mol`, `states` | Yes | — | Yes |
| ResHF | `reshf` | Yes | — | No (real states only) |
| NOCI-PT2, selected NOCI | `excit`, `snoci` | Yes | — | Yes |
| NOCIQMC | `excit`, `prop`, `det` or `qmc` | `det` | `qmc` | Deterministic only |
| GNOCC | `noccmc`, with the `gnocc` feature | Yes | — | Complex amplitudes |
| JM-NOCCMC | `jmnoccmc` | Without `jmnoccmc.qmc` | With `jmnoccmc.qmc` | — |

## Notation

| Symbol | Meaning |
| --- | --- |
| $\lvert{}^x\Psi\rangle$ | Reference determinant $x$, an SCF solution |
| $w, x, y$ | Indices of determinants in the extended space |
| $H_{wx}$, $S_{wx}$ | Hamiltonian and overlap matrix elements $\langle w\lvert\hat H\rvert x\rangle$, $\langle w\vert x\rangle$ |
| $c_x$ | Coefficients of the wavefunction $\lvert\Psi\rangle = \sum_x c_x\lvert x\rangle$ |
| $N_w = S_{wx}c_x$ | Range population of the S-apply and B-apply propagators |
| $E_s$ | Population-control shift |
| $\Delta\tau$ | Imaginary time step `prop.dt` |

Repeated indices are summed.
