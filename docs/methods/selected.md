# Selected NOCI and NOCI-PT2

Reference NOCI captures static correlation but misses most dynamic correlation. NOCI-PT2
{cite}`burton2020` recovers it by second-order perturbation theory in the excited space of the
references, and selected NOCI (SNOCI) uses the same perturbative amplitudes to decide which
excited determinants to add to the variational space.

## NOCI-PT2

Start from a NOCI state in a space $\mathcal V_N^{(k)}$ of nonorthogonal determinants,

$$
|\Psi^{(k)}\rangle = \sum_{x\in\mathcal V_N^{(k)}}|{}^x\Psi\rangle c_x^{(k)},
\qquad
\hat P^{(k)} = \sum_{w,x\in\mathcal V_N^{(k)}}|{}^w\Psi\rangle\,{}_{wx}S^{(k)}\langle{}^x\Psi|,
\qquad
\hat Q^{(k)}_\text{space} = \hat I - \hat P^{(k)},
$$

where $\hat P^{(k)}$ projects onto the current space through its inverse metric ${}_{wx}S^{(k)}$.
Every determinant may overlap the current space, so the first-order interacting space cannot be
partitioned off as in the orthogonal case. Instead the candidate excitations
$|{}^{xI}\Psi\rangle = \hat O_I|{}^x\Psi\rangle$ of the space $\mathcal A_N^{(k)}$, generated from
the orders in `excit`, are projected against it, giving states
$|{}^{xI}\Omega^{(k)}\rangle = \hat Q^{(k)}_\text{space}|{}^{xI}\Psi\rangle$ of the projected space
$\mathcal S_N^{(k)}$. Candidates with negligible projected norm are discarded. The first-order
wavefunction is

$$
|\Psi^{(k,1)}\rangle = \sum_{xI\in\mathcal S_N^{(k)}}a_{xI}^{(k)}|{}^{xI}\Omega^{(k)}\rangle,
$$

and with the state projectors $\hat M^{(k)} = |\Psi^{(k)}\rangle\langle\Psi^{(k)}|$ and
$\hat Q^{(k)}_\text{state} = \hat I - \hat M^{(k)}$ its amplitudes solve

$$
\sum_{wI\in\mathcal S_N^{(k)}}\big({}^{xJ,wI}F^{(k)} - E_0^{(k)}\,{}^{xJ,wI}S^{(k)}\big)a_{wI}^{(k)}
= -{}^{xJ}V^{(k)},
$$

$$
{}^{xJ,wI}F^{(k)} = \langle{}^{xJ}\Omega^{(k)}|\hat Q^{(k)}_\text{state}\hat H_0^{(k)}\hat Q^{(k)}_\text{state}|{}^{wI}\Omega^{(k)}\rangle,
\qquad
{}^{xJ,wI}S^{(k)} = \langle{}^{xJ}\Omega^{(k)}|\hat Q^{(k)}_\text{state}|{}^{wI}\Omega^{(k)}\rangle,
\qquad
{}^{xJ}V^{(k)} = \langle{}^{xJ}\Omega^{(k)}|\hat H - E^{(k)}|\Psi^{(k)}\rangle.
$$

The NOCI determinants have no common orbitals, so the zeroth-order Hamiltonian
$\hat H_0^{(k)} = \hat M^{(k)}\hat F\hat M^{(k)} + \hat Q^{(k)}_\text{state}\hat F\hat Q^{(k)}_\text{state}$
is built from a generalised Fock operator of the multireference density of $|\Psi^{(k)}\rangle$,

$$
F[{}^{(k)}P_\Psi]_{\mu\nu} = h_{\mu\nu}
+ \sum_{\sigma\tau}\big[(\mu\nu|\sigma\tau) - (\mu\sigma|\nu\tau)\big]{}^{(k)}P_\Psi^{\tau\sigma},
\qquad
{}^{(k)}P_\Psi^{\mu\nu} = \sum_{w,x\in\mathcal V_N^{(k)}}c_w^{(k)*}\,{}^{wx}\rho^{\mu\nu}c_x^{(k)},
$$

with co-densities ${}^{wx}\rho^{\mu\nu}$ from the generalised Slater–Condon rules, ${}^{wx}\tilde S\,{}^{wx}W^{\mu\nu}$
or ${}^{wx}\tilde S\,{}^{wx}P_i^{\mu\nu}$ ([Matrix elements](elements.md)). This makes
$|\Psi^{(k)}\rangle$ an eigenfunction of $\hat H_0^{(k)}$, and the second-order energy is

$$
E_\text{PT2}^{(k)} = \sum_{xI\in\mathcal S_N^{(k)}}a_{xI}^{(k)*}\,{}^{xI}V^{(k)}.
$$

The zeroth-order operator is not diagonal in a nonorthogonal candidate space, so the amplitude
equation is solved iteratively by restarted GMRES {cite}`saad1986`. Two preconditioners are
available:

- `diag` uses the diagonal of ${}^{xJ,wI}M^{(k)} = {}^{xJ,wI}F^{(k)} - E_0^{(k)}\,{}^{xJ,wI}S^{(k)}$.
- `woodbury` adds a low-rank correction to the diagonal through the Woodbury identity
  {cite}`woodbury1950`, capturing the projection against $|\Psi^{(k)}\rangle$ that the diagonal
  misses.

In a geometry scan, GMRES starts from the converged amplitudes of the previous geometry. The
candidates are matched by parent reference and occupation, which the orbital alignment after the
SCF ([SCF](scf.md)) keeps comparable, and candidates new to this geometry start from zero. The NOCI
state is fixed only up to a phase, which carries into both $\mathbf V^{(k)}$ and
$\mathbf a^{(k)}$, so the previous amplitudes are rescaled by the phase of
$\langle \mathbf V^{(k)}_{\text{prev}}, \mathbf V^{(k)}\rangle$. The solution is unchanged to
within the GMRES tolerance.

Products of ${}^{xJ,wI}M^{(k)}$ with the amplitudes are either evaluated on demand, or from a
stored packed copy (`full_m`). On demand, the one-body operator factorises over the spins of each
pair of references, $F = F^\alpha \otimes S^\beta + S^\alpha \otimes F^\beta$. When the candidates
of a pair of references, the same one or two different ones, have at most two replacements, each
same-spin factor is applied in contraction form rather than element by element: the occupied orbitals of the ket reference are
made biorthogonal to those of the bra, $\tilde\phi = C^x_o (C^{w\dagger}_o S C^x_o)^{-1}$, the
virtual orbitals of both are reduced to their remainders outside the occupied spaces and made
biorthogonal in turn, and the operator then acts by the orthogonal Slater–Condon rules. Every step
is a matrix product over orbital indices, so a product costs $O(o^2v^2(o+v))$ per reference pair
and spin rather than the square of the number of spin strings. Near-zero occupied overlaps are
deformed by $\epsilon$, which makes every factor a polynomial in $\epsilon$, and the factors at
$\epsilon = 0$ follow exactly from a few well-conditioned values of $\epsilon$. For a reference
paired with itself the frame is its own orthonormal one: the occupied overlap is the identity, no
pair is deformed, and the operator acts by the ordinary Slater–Condon rules, now on whole vectors.
Double replacements take two passes, all products over occupied orbitals and then all products over
virtual orbitals, one pair of holes at a time, and are read directly at the bra strings, so no
array of double replacements is ever stored in full.
The contraction form needs no Wick intermediates, and the QMC overlap products use the same one. Other pairs use
the spin-factorised tables and Wick intermediates (`factor_tables`). MPI
distributes the products across ranks and Rayon parallelises the matrix elements within each rank.

### Imaginary shifts

Where a projected candidate is nearly degenerate with the NOCI state, ${}^{xJ,wI}M^{(k)}$ is nearly
singular and the amplitudes diverge: the intruder-state problem. An imaginary level shift
$\hat H_0^{(k)} \rightarrow \hat H_0^{(k)} + i\epsilon\,\hat Q^{(k)}_\text{state}$
{cite}`forsberg1997,burton2020` moves these poles off the real axis, so the equation becomes

$$
\sum_{wI\in\mathcal S_N^{(k)}}{}^{xJ,wI}M^{(k)}(\epsilon)\,a_{wI}^{(k)}(\epsilon) = -{}^{xJ}V^{(k)},
\qquad
{}^{xJ,wI}M^{(k)}(\epsilon) = {}^{xJ,wI}F^{(k)} - E_0^{(k)}\,{}^{xJ,wI}S^{(k)} + i\epsilon\,{}^{xJ,wI}S^{(k)},
$$

and the energy is the real part of the Hylleraas functional,

$$
E_\text{PT2}^{(k)}(\epsilon) = \operatorname{Re}\Big[
\sum_{xJ,wI}a_{xJ}^{(k)*}(\epsilon)\,{}^{xJ,wI}M^{(k)}(\epsilon)\,a_{wI}^{(k)}(\epsilon)
+ \sum_{xJ}a_{xJ}^{(k)*}(\epsilon)\,{}^{xJ}V^{(k)}
+ \sum_{xJ}{}^{xJ}V^{(k)*}a_{xJ}^{(k)}(\epsilon)\Big],
$$

which reduces to $E_\text{PT2}^{(k)}$ at $\epsilon = 0$. The shift removes the divergences at the
cost of a small shift-dependent bias. `snoci.imag_shift` sets the shifts $\epsilon$ to solve for.

### First-order interacting space

Arbitrary excitation orders are accepted, but the rigorous NOCI-PT2 first-order interacting space
$\mathcal A_N^{(k)}$ consists of the single and double excitations of the references. Use
`excit.orders = {1, 2}` for NOCI-PT2 energies.

## Selected NOCI

:::{warning}
SNOCI has not been worked on for a long time and is not well developed. Its cost currently blows up
as the selected space grows. Work on it is planned.
:::

SNOCI repeats the NOCI-PT2 step, with $(k)$ labelling the selected space at iteration $k$, and
scores every candidate by its perturbative contribution,

$$
s_{xI}^{(k)} = \big|a_{xI}^{(k)}\,{}^{xI}V^{(k)}\big|.
$$

At each iteration it:

1. solves the generalised eigenvalue problem in $\mathcal V_N^{(k)}$;
2. constructs or updates the external candidate pool;
3. builds the current–candidate overlap, Hamiltonian and generalised-Fock quantities;
4. projects the candidates against the current NOCI state;
5. solves the projected NOCI-PT2 amplitude equation;
6. adds the candidates with $s_{xI}^{(k)} > \sigma$, at most `max_add` per iteration, to form
   $\mathcal V_N^{(k+1)}$, where $\sigma$ is `sigma`.

The iteration stops when $|E_\text{PT2}^{(k)}|$ falls below `tol`, when no candidate passes
$\sigma$, or after `max_iter` iterations or once the space reaches `max_dim` determinants. A
calculation limited to one iteration is a plain NOCI-PT2 calculation.

The correction is state-specific: $|\Psi^{(k)}\rangle$ may be any eigenstate of the reference
NOCI problem, not only the lowest. `roots` lists the states to correct, and each receives its own
generalised Fock operator, projector and amplitude equation. With spin-broken references the NOCI
states need not be spin-pure, so each is reported with its
$\langle S^2\rangle = \sum_{wx} c_w^* c_x \langle{}^w\Psi|\hat S^2|{}^x\Psi\rangle$; where two states
of different spin cross, listing both follows each surface through the crossing.

NOCI-PT2 and SNOCI support holomorphic SCF states.
