from __future__ import annotations

from fractions import Fraction
from functools import cache
from itertools import permutations

def permutationSign(sequence: tuple[int, ...]) -> int:
    """
    Return fermionic parity of a permutation.

    Notation:
        Every swap of fermion operators contributes a factor of -1.

    Examples:
        permutationSign((1, 0)) gives -1 because a_p a_q = -a_q a_p.
        permutationSign((0, 2, 1, 3)) gives -1 because the sequence has one inversion.
    """
    inversions = 0

    for i in range(len(sequence)):
        # Compare each i with later positions j > i.
        for j in range(i + 1, len(sequence)):
            # If i is larger than any j pair is unordered so we have an inversion.
            if sequence[i] > sequence[j]:
                inversions += 1

    return -1 if inversions % 2 else 1

def invPerm(p: tuple[int, ...]) -> tuple[int, ...]:
    """
    Return the inverse of a permutation. 

    Notation:
        p maps i ---> p[i], invPerm(p) maps p[i] ---> i.

    Examples:
        p = (2, 0, 1) means that:
        0 ---> 2,
        1 ---> 0,
        2 ---> 1,
        and therefore invPerm(p) = (1, 2, 0).
    """
    out = [0] * len(p)

    for i, x in enumerate(p):
        out[x] = i

    return tuple(out)

def composePerm(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, ...]:
    """
    Return the composition of two permutation maps.

    Notation:
        p maps i ---> p[i],
        q maps i ---> q[i],
        compose(p, q) represents i ---> q[i] ---> p[q[i]].

    Examples:
        p = (2, 0, 1) means that:
        0 ---> 2,
        1 ---> 0,
        2 ---> 1.

        q = (1, 2, 0) means that:
        0 ---> 1,
        1 ---> 2,
        2 ---> 0.

        Then compose(p, q) = p after q.

        For i = 0:
        0 ---> q[0] = 1 ---> p[1] = 0.
        For i = 1:
        1 ---> q[1] = 2 ---> p[2] = 1.
        For i = 2:
        2 ---> q[2] = 0 ---> p[0] = 2.

        Therefore:
        compose(p, q) = (0, 1, 2).

    """
    return tuple(p[q[i]] for i in range(len(p)))

def cycleCountPerm(p: tuple[int, ...]) -> int:
    """
    Count the number of cycles in a permutation. A cycle is a closed chain within a permutation.
    
    Notation:
        p maps i ---> p[i].
        Starting from an index i apply p repeatedly.
        i ---> p[i] ---> p[p[i]] ---> ...
        This cycle will eventually return to an index visited previously.
        This closed chain is one cycle.

    Examples:
        p = (1, 2, 0) means:
        0 ---> 1,
        1 ---> 2, 
        2 ---> 0,

        If we start from 0: 0 ---> 1 ---> 2 ---> 0 so all indices are in the same cycle.
        Therefore cycleCount((1, 2, 0)) = 1.
        Conversley the identity permutation p = (0, 1, 2) gives cycleCount = 3.
    """
    # Track seen indices within a cycle.
    seen = [False] * len(p)
    cycles = 0
    
    # Start a cycle from all indices.
    for i in range(len(p)):
        # If this index was visited in another cycle already skip it.
        if seen[i]:
            continue
        
        # New unvisited index starts a new cycle.
        cycles += 1
        j = i
        
        # Follow permutation map until returning to a visited index.
        while not seen[j]:
            seen[j] = True
            j = p[j]

    return cycles

@cache
def spinGram(rank: int) -> tuple[tuple[Fraction, ...], ...]:
    """
    Build the spin-projection Gram matrix for lower-index permutations.

    Notation:
        G_{pq} = \operatorname{sgn}(p) \operatorname{sgn}(q) 2^{c(p^{-1}q)},
        where c(p^{-1}q) is the number of cycles in the relative permutation.

    Examples:
        For rank 2, the permutations are (0, 1), (1, 0)
        and this returns the 2 by 2 Gram matrix relating the two possible
        lower-index orderings of \Lambda^{pq}_{rs}.
    """
    perms = tuple(permutations(range(rank)))
    gram = []

    for p in perms:
        row = []

        for q in perms:
            rel = composePerm(invPerm(p), q)
            row.append(
                Fraction(
                    permutationSign(p)
                    * permutationSign(q)
                    * (2 ** cycleCountPerm(rel)),
                    1,
                )
            )

        gram.append(row)

    return tuple(tuple(row) for row in gram)

def solveConsistent(mat: list[list[Fraction]], rhs: list[Fraction]) -> list[Fraction]:
    """
    Solve a linear system A x = b that may be singular and choose a deterministic gauge. 

    Here, A is the spin-projection Gram matrix and x are the coefficients c_\pi in:
    \lambda^{p_1 \sigma_1 ... p_k \sigma_k}_{q_1 \tau_1 ... q_k \tau_k}
    = sum_{\pi in S_k} c_\pi Lambda^{p_1...p_k}_{q_{\pi(1)}...q_{\pi(k)}}.

    This system can be singular because the spin-free \Lambda permutation basis 
    is linearly dependent for rank 3 or greater. This function chooses a deterministic solve 
    by setting all free variables to zero.
    """
    # Number of equations.
    nRows = len(mat)
    # Number of unknowns (number of lower-permutations).
    nCols = len(mat[0])
    # Combine A and b into one matrix for Gaussian elimination.
    aug = [list(row) + [rhs[i]] for i, row in enumerate(mat)]
    
    # Record pivot columns, those which correspond to determined variables.
    pivotCols = []
    row = 0
    
    # Scan columns left to right and try to find a pivot in each.
    for col in range(nCols):
        pivot = None
        
        # First entry in column j with r >= row where A_{rj} \neq 0 becomes a pivot.
        for r in range(row, nRows):
            if aug[r][col] != 0:
                pivot = r
                break
        
        # If we have no pivot column j corresponds to a free variable.
        if pivot is None:
            continue
        
        # Swap pivot and row.
        if pivot != row:
            aug[row], aug[pivot] = aug[pivot], aug[row]
        
        # Normalise pivot row.
        scaleFactor = aug[row][col]
        aug[row] = [x / scaleFactor for x in aug[row]]
        
        # Eliminate the pivot column from all other rows (Gauss-Jordan elimination).
        for r in range(nRows):
            if r == row:
                continue

            factor = aug[r][col]

            if factor == 0:
                continue

            aug[r] = [
                aug[r][i] - factor * aug[row][i]
                for i in range(nCols + 1)
            ]

        pivotCols.append(col)
        row += 1
        
        # Now all rows have pivots.
        if row == nRows:
            break

    for r in range(row, nRows):
        if all(aug[r][c] == 0 for c in range(nCols)) and aug[r][-1] != 0:
            raise ValueError("inconsistent spin-projection system")
    
    # Initialise solution with all variables zero.
    sol = [Fraction(0) for _ in range(nCols)]
    
    # Insert pivot variables into solution.
    for r, col in enumerate(pivotCols):
        sol[col] = aug[r][-1]

    return sol
