# SCF states

NOCI builds its basis from several distinct SCF solutions of the same molecule. Away from
equilibrium these solutions break spin or spatial symmetry, and each describes one dominant
configuration well. `noci-rs` finds them in two ways: explicit maximum-overlap recipes, or an
automatic metadynamics search.

## Conventional SCF

For each spin $\omega$, the RHF and UHF orbitals solve the Roothaan–Hall equations
{cite}`roothaan1951`

$$
\sum_\nu F^\omega_{\mu\nu}c^\omega_{\nu i} = \sum_\nu S_{\mu\nu}c^\omega_{\nu i}\varepsilon^\omega_i,
\qquad
F^\omega_{\mu\nu} = h_{\mu\nu}
+ \sum_{\lambda\sigma}\Big[(\mu\nu|\sigma\lambda)D_{\lambda\sigma}
- (\mu\lambda|\nu\sigma)D^\omega_{\lambda\sigma}\Big],
$$

with spin densities $D^\omega_{\mu\nu} = \sum_i^{N_\omega}c^\omega_{\mu i}c^{\omega *}_{\nu i}$,
total density $D_{\mu\nu} = D^\alpha_{\mu\nu} + D^\beta_{\mu\nu}$, and
$D^\alpha_{\mu\nu} = D^\beta_{\mu\nu}$ for RHF. They are solved iteratively with DIIS acceleration
{cite}`hamilton1986`, starting from a superposition of atomic densities {cite}`vanlenthe2006`. An
SCF has converged once the energy change is below `scf.e_tol` and the commutator norm

$$
\Big(\sum_\omega\big\lVert\mathbf X^\dagger\big(\mathbf F^\omega\mathbf D^\omega\mathbf S
- \mathbf S\mathbf D^\omega\mathbf F^\omega\big)\mathbf X\big\rVert_F^2\Big)^{1/2},
$$

taken in the Löwdin-orthonormalised basis $\mathbf X = \mathbf S^{-1/2}$, is below
`scf.fds_sdf_tol`.

## Maximum-overlap recipes

Each entry of `states.mom` is a recipe for one solution. It perturbs the initial guess towards the
desired state, then converges with the maximum overlap method (MOM) {cite}`gilbert2008`. At SCF
iteration $k$, MOM forms the overlap of every new orbital $j$ with the occupied orbitals $i$ of the
previous iteration,

$$
\Omega_{ij} = \sum_{\mu\nu} c^{\omega *}_{\mu i, k-1}S_{\mu\nu}c^\omega_{\nu j, k},
\qquad
p_j = \sum_i^{N_\omega}|\Omega_{ij}|,
$$

and occupies the $N_\omega$ orbitals with the largest $p_j$, rather than the lowest-energy
orbitals. MOM therefore stays on the solution the guess was steered towards, even when that
solution is not the SCF ground state.

Three kinds of perturbation are available, alone or together:

- **Spin bias** (`spin_bias`) polarises the initial $\alpha$ and $\beta$ densities in a pattern
  over the atoms, for example $(+, -)$ for the two ends of a stretched bond. This produces
  spin-symmetry-broken UHF solutions.
- **Spatial bias** (`spatial_bias`) localises the initial density over the atoms, producing
  spatial-symmetry-broken RHF solutions.
- **Excitations** (`excit`) swap occupied and virtual orbitals of the guess, targeting excited SCF
  solutions.

Only recipes with `noci = true` enter the reference NOCI basis; others can serve as seeds or
partners. Converged solutions are compared by the electron distance {cite}`thom2008` and a density
distance,

$$
d_{wx}^2 = N_e - \sum_\omega {}^xD^{\omega,\mu\nu}\,{}^wD^\omega_{\mu\nu},
\qquad
\delta_{wx}^2 = \sum_\omega\sum_{\mu\nu}
\Big|\sum_\lambda\big({}^wD^{\omega,\mu\lambda} - {}^xD^{\omega,\mu\lambda}\big)S_{\lambda\nu}\Big|^2,
$$

where ${}^wD^\omega_{\mu\nu} = \sum_{\lambda\sigma}S_{\mu\lambda}\,{}^wD^{\omega,\lambda\sigma}S_{\sigma\nu}$
is the covariant density of solution $w$. A solution with $d_{wx}^2$ or $\delta_{wx}^2$ below
`scf.d_tol` from an earlier one is a duplicate and leaves the NOCI basis, so coalesced solutions
are removed before they make it linearly dependent.

## SCF metadynamics

`states.metadynamics` searches for solutions without recipes {cite}`thom2008`. Every solution $w$
already found adds a repulsive bias to the Fock matrix, shaped like its density:

$$
\tilde F^\omega_{\mu\nu} = F^\omega_{\mu\nu}
+ \sum_w N_e\lambda\, e^{-\lambda d_{0w}^2}\,{}^wD^{\omega,\mu\nu},
$$

where $d_{0w}$ is the electron distance between the current density and solution $w$, and $\lambda$
is the input `lambda`, which sets both the strength and the inverse width of the bias. The next SCF
is therefore pushed away from every solution already found. Its initial guesses are polarised by
`spinpol` (for UHF) and `spatialpol` (for RHF), and the search runs until `nstates_rhf` RHF and
`nstates_uhf` UHF solutions are found or `max_attempts` is reached. This needs no prior knowledge
of the solutions, at the cost of less control over which ones are found.

## Holomorphic SCF

Real SCF solutions are not guaranteed to exist at every geometry: as a bond stretches or
compresses, two real solutions can coalesce and vanish. A NOCI built from them then changes
discontinuously along a scan. Holomorphic Hartree–Fock {cite}`burton2016` removes this problem by
dropping the complex conjugation from the density,

$$
\tilde D^\omega_{\mu\nu} = \sum_i^{N_\omega}c^\omega_{\mu i}c^\omega_{\nu i},
\qquad
\sum_\nu\tilde F^\omega_{\mu\nu}c^\omega_{\nu i} = \sum_\nu S_{\mu\nu}c^\omega_{\nu i}\tilde\varepsilon^\omega_i,
$$

where the holomorphic Fock matrix $\tilde F^\omega_{\mu\nu}$ is built from $\tilde D^\omega_{\mu\nu}$.
The energy is then a complex-analytic function of the orbital coefficients, and its stationary
points are the analytic continuation of the real solutions, which persist through the coalescence
points. Without an Aufbau ordering of complex orbital energies, the stationary points are found by
rotating the occupied and virtual orbitals until the holomorphic Brillouin condition holds,

$$
g_{ai} = \frac{\partial\tilde E}{\partial\theta_{ai}}
= 2\sum_{\mu\nu}c^\omega_{\mu a}\tilde F^\omega_{\mu\nu}c^\omega_{\nu i}
= 2\tilde F^\omega_{ai} = 0,
$$

by quasi-Newton steps in energy-weighted coordinates, with complex-symmetric SR1 Hessian updates
and a backtracking line search on $\lVert g\rVert$.

### Tracking holomorphic solutions

Holomorphic solutions are followed along a scan using the complex adiabatic connection
{cite}`burton2019complex`, which scales the electron–electron interaction by a complex $\lambda$,
$F^\omega_{\mu\nu}(\lambda) = h_{\mu\nu} + \lambda\big(J_{\mu\nu} - K^\omega_{\mu\nu}\big)$:

1. A converged real solution at $\lambda = 1$ is rotated in small phase steps to
   $\lambda = e^{i\theta}$, off the real axis, where solutions do not coalesce.
2. The off-axis branch is carried from geometry to geometry at fixed $\lambda$. The previous
   orbitals are re-orthonormalised in each new atomic-orbital metric before optimisation.
3. At each geometry the branch is relaxed back to $\lambda = 1$, giving the physical holomorphic
   state.

A recipe with `holomorphic = true` is initialised this way from the real solution named by its
`partner`, or by its own label if `partner` is omitted. Holomorphic states can enter reference
NOCI, NOCI-PT2 and deterministic NOCIQMC {cite}`burton2019`.

## Geometry scans

When `mol.r` is a list, converged states seed the SCF at the next geometry {cite}`burton2019`, so
each recipe follows one continuous solution along the scan.
