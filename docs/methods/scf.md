# SCF states

NOCI builds its basis from several distinct SCF solutions of the same molecule. Away from
equilibrium these solutions break spin or spatial symmetry, and each describes one dominant
configuration well. `noci-rs` finds them in two ways: explicit maximum-overlap recipes, or an
automatic metadynamics search.

## Conventional SCF

RHF and UHF equations are solved by Roothaan iteration with DIIS acceleration
{cite}`hamilton1986`, starting from a superposition of atomic densities {cite}`vanlenthe2006`.
An SCF has converged once the energy change is below `scf.e_tol` and the commutator norm
$\lVert\mathbf{FDS} - \mathbf{SDF}\rVert$ is below `scf.fds_sdf_tol`.

## Maximum-overlap recipes

Each entry of `states.mom` is a recipe for one solution. It perturbs the initial guess towards the
desired state, then converges with the maximum overlap method (MOM) {cite}`gilbert2008`, which
occupies the orbitals overlapping most with the previous iteration's occupied space rather than
the lowest-energy orbitals. MOM therefore stays on the solution the guess was steered towards,
even when that solution is not the SCF ground state.

Three kinds of perturbation are available, alone or together:

- **Spin bias** (`spin_bias`) polarises the initial $\alpha$ and $\beta$ densities in a pattern
  over the atoms, for example $(+, -)$ for the two ends of a stretched bond. This produces
  spin-symmetry-broken UHF solutions.
- **Spatial bias** (`spatial_bias`) localises the initial density over the atoms, producing
  spatial-symmetry-broken RHF solutions.
- **Excitations** (`excit`) swap occupied and virtual orbitals of the guess, targeting excited SCF
  solutions.

Only recipes with `noci = true` enter the reference NOCI basis; others can serve as seeds or
partners. Converged solutions are compared by density and electron-distance measures: two
solutions closer than `scf.d_tol` are duplicates, so coalesced solutions are removed before they
make the NOCI basis linearly dependent.

## SCF metadynamics

`states.metadynamics` searches for solutions without recipes {cite}`thom2008`. Every solution
already found adds a repulsive bias to the Fock matrix, shaped like its density and weighted by

$$
N\lambda\, e^{-\lambda d^2},
$$

where $N$ is its electron count, $d$ the electron distance between the current density and that
solution, and $\lambda$ the input `lambda`. The next SCF is therefore pushed away from every
solution already found. Its initial guesses are polarised by `spinpol` (for UHF) and `spatialpol`
(for RHF), and the search runs until `nstates_rhf` RHF and `nstates_uhf` UHF solutions are found
or `max_attempts` is reached. This needs no prior knowledge of the solutions, at the cost of less
control over which ones are found.

## Holomorphic SCF

Real SCF solutions are not guaranteed to exist at every geometry: as a bond stretches or
compresses, two real solutions can coalesce and vanish. A NOCI built from them then changes
discontinuously along a scan. Holomorphic Hartree–Fock {cite}`burton2016` removes this problem by
replacing the Hermitian density with its complex-analytic counterpart,

$$
\tilde E[\mathbf C] = E\big[\mathbf C, \mathbf C^{*} \to \mathbf C\big],
$$

whose stationary points are the analytic continuation of the real solutions and persist through
the coalescence points. The holomorphic equations are solved by quasi-Newton optimisation of the
complex orbital rotations, with SR1 updates and a backtracking line search, controlled by the
`scf.h` table.

### Tracking holomorphic solutions

Holomorphic solutions are followed along a scan using the complex adiabatic connection
{cite}`burton2019complex`, scaling the electron–electron interaction by a complex $\lambda$:

1. A converged real solution at $\lambda = 1$ is rotated in small phase steps to
   $\lambda = e^{i\theta}$, off the real axis, where solutions do not coalesce.
2. The off-axis branch is carried from geometry to geometry at fixed $\lambda$. The previous
   orbitals are re-orthonormalised in each new atomic-orbital metric before optimisation.
3. At each geometry the branch is relaxed back to $\lambda = 1$, giving the physical holomorphic
   state.

A recipe with `holomorphic = true` is initialised this way from the real solution named by its
`partner`, or by its own label if `partner` is omitted. Holomorphic states can enter reference NOCI, NOCI-PT2 and deterministic NOCIQMC
{cite}`burton2019`.

## Geometry scans

When `mol.r` is a list, converged states seed the SCF at the next geometry {cite}`burton2019`, so
each recipe follows one continuous solution along the scan.
