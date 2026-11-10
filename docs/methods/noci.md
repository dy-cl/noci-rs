# Nonorthogonal configuration interaction

## Reference NOCI

Reference NOCI expands the wavefunction in the selected SCF determinants,

$$
|\Psi\rangle = \sum_x c_x |{}^x\Psi\rangle,
$$

and solves the generalised eigenvalue problem

$$
\mathbf H\mathbf c = E\,\mathbf S\mathbf c,
\qquad
H_{wx} = \langle{}^w\Psi|\hat H|{}^x\Psi\rangle,
\quad
S_{wx} = \langle{}^w\Psi|{}^x\Psi\rangle
$$

{cite}`thom2009,burton2019`. Because each determinant is a separately optimised SCF solution, the
basis is nonorthogonal and nearly linearly dependent wherever two solutions approach each other.
The eigenproblem is therefore solved by canonical orthogonalisation {cite}`lowdin1950`:
eigenvectors of $\mathbf S$ with eigenvalues below a tolerance are projected out before $\mathbf H$
is diagonalised in the remaining subspace.

A handful of determinants captures the static correlation of bond breaking and other
multireference problems. The lowest root is the reference state for every post-NOCI method.
Holomorphic determinants enter in the same way, with their matrix elements continued
analytically, without complex conjugation of the orbitals {cite}`burton2019`.

## Extended determinant spaces

The post-NOCI methods work in an extended space of excitations of the reference determinants.
Each reference $|{}^x\Psi\rangle$ defines an orthonormal orbital frame, its *parent frame*, and
the orders in `excit` generate its single, double and higher excitations within that frame.
Determinants from one parent are mutually orthogonal, but determinants from different parents
overlap, so the extended space is nonorthogonal and, once it is large enough, *overcomplete*: its
overlap matrix has an exact null space. Excitations of different references span the same states,
and every NOCI-based method has to cope with these redundant directions.

## ResHF reference optimisation

The SCF determinants are optimised one at a time, each for its own energy, not for the NOCI built
from them. Resonating Hartree–Fock (ResHF) {cite}`miller2025` instead optimises the orbitals of
every reference determinant together with the NOCI expansion, minimising the state-averaged energy

$$
E_{\text{SA}} = \sum_I w_I E_I,
\qquad
\mathbf H\mathbf c_I = E_I\mathbf S\mathbf c_I.
$$

Each macro-iteration alternates a CI step, which solves the NOCI problem for fixed determinants,
with a Roothaan step on each determinant's *resonating* Fock matrix. Its occupied–virtual block is
the orbital gradient of $E_\text{SA}$,

$$
\frac{\partial E_{\text{SA}}}{\partial\kappa^A_{ai}}
= 2\big(\mathbf C^{A\dagger}_{\text{virt}}\mathbf F^{A,\omega}\mathbf C^A_{\text{occ}}\big)_{ai}.
$$

One DIIS history is shared by all determinants, because they are coupled through the CI expansion
and a common set of extrapolation coefficients keeps their updates consistent.

Nearly orthogonal pairs of determinants make the transition quantities singular. Following
{cite}`miller2025`, the small singular values of each occupied-orbital overlap are floored at
`reshf.cutoff`, so the coupling, overlap and resonating Fock matrices stay finite and are all
built from the same floored quantities. ResHF minimises this floored energy, which equals the
reference NOCI energy unless two determinants become nearly orthogonal.

The optimised determinants replace the real SCF states marked `noci = true` and are ordinary SCF
states to every later method. In a scan they also seed the SCF at the next geometry.
