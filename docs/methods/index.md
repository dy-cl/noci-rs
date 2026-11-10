# Overview of methods

Every calculation follows the same pipeline.

1. **SCF states.** Multiple real or holomorphic RHF and UHF solutions are found by
   maximum-overlap recipes or SCF metadynamics ([SCF states](scf.md)). ResHF can instead optimise
   the reference determinants together with their NOCI expansion
   ([ResHF](noci.md#reshf-reference-optimisation)).
2. **Reference NOCI.** The selected solutions form a nonorthogonal basis
   $\{|{}^x\Psi\rangle\}$, in which the generalised eigenvalue problem
   $({}^{wx}H - E\,{}^{wx}S)c_x = 0$ gives the reference NOCI states ([NOCI](noci.md)).
3. **Extended space.** Excitations of each reference within its own orbital frame, up to the
   orders in `excit`, build the extended space in which correlation is recovered.
4. **Post-NOCI method.** NOCI-PT2 and selected NOCI ([Selected NOCI and NOCI-PT2](selected.md)),
   deterministic or stochastic NOCIQMC ([NOCIQMC](nociqmc.md)), or nonorthogonal coupled cluster
   ([NOCC](nocc.md)) treat the extended space.

Every step needs matrix elements between determinants built from different orbital frames. Their
evaluation forms a core part of the package and is described in [Matrix elements](elements.md).

## Method summary

| Method | Input tables | Holomorphic states |
| --- | --- | --- |
| Reference NOCI | `mol`, `states` | Yes |
| ResHF | `reshf` | No |
| NOCI-PT2, selected NOCI | `excit`, `snoci` | Yes |
| NOCIQMC | `excit`, `prop`, `det`, `qmc` | `det` only |
| GNOCC | `gnoccmc`, with the `gnocc` feature | No |
| JM-NOCCMC | `jmnoccmc` | No |

## Notation

The notation follows the tensor notation of Head-Gordon *et al.* {cite}`headgordon1998` for
nonorthogonal determinants.

| Symbol | Meaning |
| --- | --- |
| $\mu, \nu, \lambda, \sigma$ | Atomic orbitals |
| $i, j, k, l$ / $a, b, c, d$ / $p, q, r, s$ | Occupied / virtual / general molecular orbitals |
| $w, x, y, z$ | Determinants |
| $\omega \in \{\alpha, \beta\}$ | Spin |
| $N_e$, $N_\omega$ | Number of electrons, and of electrons of spin $\omega$ |
| $S_{\mu\nu}$, $h_{\mu\nu}$, $(\mu\nu\vert\lambda\sigma)$ | AO overlap, core Hamiltonian and electron-repulsion integrals |
| ${}^x c^{\mu\cdot}_{\cdot i}$ | AO coefficients of molecular orbital $\lvert{}^x\psi_i\rangle$ of determinant $x$ |
| $\lvert{}^x\Psi\rangle$ | Determinant $x$, a reference SCF solution or an excitation of one |
| $\lvert{}^x\Psi_{ij\cdots}^{ab\cdots}\rangle$ | Excitation of reference $\lvert{}^x\Psi\rangle$ within its own orbital frame |
| ${}^{wx}S$, ${}^{wx}H$ | Overlap $\langle{}^w\Psi\vert{}^x\Psi\rangle$ and Hamiltonian $\langle{}^w\Psi\vert\hat H\vert{}^x\Psi\rangle$ elements |
| ${}_{wx}S$ | Inverse metric, where the overlap is invertible |
| $S_{wx}$, $H_{wx}$ | The same elements, written as ordinary matrices where the overlap may be singular |
| $c_x$ | Covariant coefficients of $\lvert\Psi\rangle = \lvert{}^x\Psi\rangle c_x$ |
| $\tau$, $\Delta\tau$ | Imaginary time and its step `prop.dt` |
| $c_x(\tau)$, $N_w(\tau) = S_{wx}c_x(\tau)$ | Propagated coefficients and range population |
| $E_s^S(\tau)$ | Population-control shift |
| $n_\text{cyc}$ | Iterations per report, `qmc.ncycles` |
| $N_\text{tot}(\tau)$, $N_\text{ref}(\tau)$, $N_\text{target}$ | Total population, population on the references, and target population |

Repeated determinant indices are summed unless a sum is written explicitly. Once the overlap may be
singular, indices can no longer be raised or lowered, so the propagation pages write $S_{wx}$ and
$H_{wx}$ for ${}^{wx}S$ and ${}^{wx}H$.
