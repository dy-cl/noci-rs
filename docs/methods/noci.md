# Nonorthogonal configuration interaction

## Reference NOCI

Reference NOCI expands the wavefunction in the selected SCF determinants {cite}`thom2009,burton2019`,

$$
|\Psi_\text{NOCI}\rangle = |{}^x\Psi\rangle c_x,
$$

where each determinant is built from its own molecular orbitals,
$|{}^x\psi_i\rangle = \sum_\mu {}^xc^{\mu\cdot}_{\cdot i}|\phi_\mu\rangle$. Projecting the
Schrödinger equation onto $\langle{}^w\Psi|$ gives the generalised eigenvalue problem

$$
\big({}^{wx}H - E\,{}^{wx}S\big)c_x = 0,
\qquad
{}^{wx}H = \langle{}^w\Psi|\hat H|{}^x\Psi\rangle,
\quad
{}^{wx}S = \langle{}^w\Psi|{}^x\Psi\rangle.
$$

Because each determinant is a separately optimised SCF solution, the basis is nonorthogonal and
nearly linearly dependent wherever two solutions approach each other. The eigenproblem is
therefore solved by canonical orthogonalisation {cite}`lowdin1950`: eigenvectors of the overlap
with eigenvalues below a tolerance are projected out before the Hamiltonian is diagonalised in the
remaining subspace.

A handful of determinants captures the static correlation of bond breaking and other
multireference problems. The lowest root is the reference state for every post-NOCI method.
Holomorphic determinants enter in the same way, with their matrix elements continued
analytically, without complex conjugation of the orbitals {cite}`burton2019`.

## Extended determinant spaces

The post-NOCI methods work in an extended space of excitations of the reference determinants. Each
reference $|{}^x\Psi\rangle$ defines an orthonormal orbital frame, its *parent frame*, with
creation and annihilation operators ${}^x\hat b^\dagger_p$ and ${}^x\hat b_p$. The orders in
`excit` generate its excitations within that frame,

$$
|{}^x\Psi_i^a\rangle = {}^x\hat b_a^\dagger\,{}^x\hat b_i|{}^x\Psi\rangle,
\qquad
|{}^x\Psi_{ij}^{ab}\rangle
= {}^x\hat b_a^\dagger\,{}^x\hat b_b^\dagger\,{}^x\hat b_i\,{}^x\hat b_j|{}^x\Psi\rangle,
\qquad \ldots,
$$

and the extended wavefunction is

$$
|\Psi\rangle = \sum_x\Big[|{}^x\Psi\rangle c_x
+ \sum_{ia}|{}^x\Psi_i^a\rangle c_{x,i}^a
+ \frac14\sum_{ijab}|{}^x\Psi_{ij}^{ab}\rangle c_{x,ij}^{ab} + \cdots\Big].
$$

Determinants from one parent are mutually orthogonal, but determinants from different parents
overlap, so the extended space is nonorthogonal. Once it is large enough it is also
*overcomplete*. By Thouless' theorem {cite}`thouless1960`, any determinant with nonzero overlap
with a reference is an exponential of its excitations,

$$
|{}^w\Psi\rangle = \langle{}^x\Psi|{}^w\Psi\rangle\,e^{\hat Z}|{}^x\Psi\rangle,
\qquad
\hat Z = \sum_{ia}{}^{xw}Z_{ai}\,{}^x\hat b_a^\dagger\,{}^x\hat b_i,
$$

so excitations of different references span the same states. The overlap matrix then has an exact
null space, coefficient vectors with

$$
S_{wx}c_x = 0 \;\; \forall w
\quad\Rightarrow\quad
H_{wx}c_x = \langle{}^w\Psi|\hat H|\Psi_\text{ker}\rangle = 0,
\qquad
|\Psi_\text{ker}\rangle = |{}^x\Psi\rangle c_x = 0,
$$

so $\ker S \subseteq \ker H$. These redundant directions describe no state, and every NOCI-based
method has to cope with them. Where the space is not overcomplete it is typically ill-conditioned.

## ResHF reference optimisation

The SCF determinants are optimised one at a time, each for its own energy, not for the NOCI built
from them. Resonating Hartree–Fock (ResHF) {cite}`miller2025` instead optimises the orbitals of
every reference determinant together with the NOCI expansion, minimising the state-averaged energy

$$
E_{\text{SA}} = \sum_I w_I E_I,
\qquad
\big({}^{wx}H - E_I\,{}^{wx}S\big)c_{xI} = 0.
$$

Each macro-iteration alternates a CI step, which solves the NOCI problem for fixed determinants,
with a Roothaan step on each determinant's *resonating* Fock matrix ${}^xF^\omega_{\mu\nu}$. Its
occupied–virtual block is the orbital gradient of $E_\text{SA}$,

$$
\frac{\partial E_{\text{SA}}}{\partial\,{}^x\kappa^\omega_{ai}}
= 2\sum_{\mu\nu}{}^xc^{\mu\cdot *}_{\cdot a}\,{}^xF^\omega_{\mu\nu}\,{}^xc^{\nu\cdot}_{\cdot i}.
$$

One DIIS history is shared by all determinants, because they are coupled through the CI expansion
and a common set of extrapolation coefficients keeps their updates consistent.

Nearly orthogonal pairs of determinants make the transition quantities singular. Following
{cite}`miller2025`, the small singular values $s_i$ of each occupied-orbital overlap
${}^{wx}S_{ij}$ are floored at `reshf.cutoff`, so the coupling, overlap and resonating Fock
matrices stay finite and are all built from the same floored quantities. ResHF minimises this
floored energy, which equals the reference NOCI energy unless two determinants become nearly
orthogonal.

The optimised determinants replace the real SCF states marked `noci = true` and are ordinary SCF
states to every later method. In a scan they also seed the SCF at the next geometry.
