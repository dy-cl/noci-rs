# Selected NOCI and NOCI-PT2

Reference NOCI captures static correlation but misses most dynamic correlation. NOCI-PT2
{cite}`burton2020` recovers it by second-order perturbation theory in the excited space of the
references, and selected NOCI (SNOCI) uses the same perturbative amplitudes to decide which
excited determinants to add to the variational space.

## NOCI-PT2

The first-order wavefunction is expanded in a *candidate space* of excitations of the reference
determinants, generated from the orders in `excit`. The candidates are projected against the
current NOCI ground state $|\Psi_0\rangle$, so the first-order correction is orthogonal to it, and
those with negligible projected norm are discarded. The amplitudes $\mathbf a$ then solve the
linear system

$$
\mathbf M(\epsilon)\,\mathbf a(\epsilon) = -\mathbf V,
$$

where $\mathbf M$ is the projected zeroth-order operator, built from the generalised Fock
operator of the NOCI state shifted by its zeroth-order energy together with the candidate-space
metric, and $\mathbf V$ couples $|\Psi_0\rangle$ to the candidates. The second-order energy follows
from the amplitudes and couplings.

The zeroth-order operator is not diagonal in a nonorthogonal candidate space, so the system is
solved iteratively by restarted GMRES {cite}`saad1986`. Two preconditioners are available:

- `diag` uses the diagonal of $\mathbf M$.
- `woodbury` adds a low-rank correction to the diagonal through the Woodbury identity
  {cite}`woodbury1950`, capturing the projection against $|\Psi_0\rangle$ that the diagonal
  misses.

Candidate-space products $\mathbf M\mathbf a$ are either evaluated on demand from the factorised
tables and Wick intermediates, or from a stored packed copy of $\mathbf M$ (`full_m`). MPI
distributes the products across ranks and Rayon parallelises the matrix elements within each rank.

### Imaginary shifts

Where a candidate is nearly degenerate with the reference, $\mathbf M$ is nearly singular and the
second-order energy diverges: the intruder-state problem. An imaginary level shift $i\epsilon$
in the zeroth-order operator {cite}`burton2020` regularises these near-singularities, at the cost
of a small shift-dependent bias. `snoci.imag_shift` sets the shifts to solve for.

### First-order interacting space

Arbitrary excitation orders are accepted, but the rigorous NOCI-PT2 first-order interacting space
consists of the single and double excitations of the references. Use `excit.orders = {1, 2}` for
NOCI-PT2 energies.

## Selected NOCI

SNOCI repeats the NOCI-PT2 step and grows the variational space from it. At each iteration it:

1. solves the generalised eigenvalue problem in the current space;
2. constructs or updates the external candidate pool;
3. builds the current–candidate overlap, Hamiltonian and generalised-Fock quantities;
4. projects the candidates against the current NOCI ground state;
5. solves the projected NOCI-PT2 amplitude equation;
6. scores each candidate by its perturbative contribution and adds those above `sigma`, at most
   `max_add` per iteration, to the variational space.

The iteration stops when the perturbative correction falls below `tol`, when no candidate passes
`sigma`, or after `max_iter` iterations or once the space reaches `max_dim` determinants. A
calculation limited to one iteration is a plain NOCI-PT2 calculation.

NOCI-PT2 and SNOCI support holomorphic SCF states.
