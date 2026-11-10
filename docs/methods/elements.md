# Matrix elements

Every method needs overlaps, Hamiltonian and generalised-Fock matrix elements, and sometimes
transition densities, between pairs of determinants. Each pair goes to one of three engines,
depending on how the two determinants are related.

## Same orbital frame: Slater–Condon rules

Two determinants excited from the same reference share an orthonormal orbital frame. Their
overlap is zero unless they are identical, and their Hamiltonian element follows from the
standard Slater–Condon rules {cite}`mayer2003` for the excitation connecting them. This is the
cheapest case.

## Different frames: generalised Slater–Condon rules

For determinants from different frames, the occupied-orbital overlap matrix
$\mathbf S^{wx} = \mathbf C_w^{\dagger}\mathbf S_{\text{AO}}\mathbf C_x$ is diagonalised by a
singular value decomposition, giving a biorthogonal pair of orbital sets. The overlap is the
product of the singular values, and one- and two-body elements follow from the corresponding
co-densities. Zero singular values, which occur often between excited determinants, are treated
explicitly rather than inverted {cite}`burton2021`. This route costs a decomposition per pair and
scales with the number of electrons.

## Different frames: extended nonorthogonal Wick's theorem

When many excited determinants are built on the same pair of references, as in every post-NOCI
method, the extended nonorthogonal Wick's theorem {cite}`burton2021,burton2022` is much faster.
For each ordered pair of references $\langle{}^x\Psi|$ and $|{}^w\Psi\rangle$, `noci-rs`
precomputes the fundamental contractions and operator intermediates once. A matrix element
between any excitation of $|{}^x\Psi\rangle$ and any excitation of $|{}^w\Psi\rangle$ then
becomes a small determinant of contractions, whose size is set by the excitation ranks.

Singular occupied-orbital overlaps are handled by separating the product of the nonzero singular
values from the number $m$ of zero-overlap orbital pairs. Each contraction carries an assignment
$m_i\in\{0,1\}$, and the matrix element is the constrained sum

$$
\sum_{\substack{m_1,\ldots,m_L\\ \sum_i m_i = m}}
$$

over the allowed distributions of the zero-overlap pairs, with $X^{(m_i)}$ entries on and below
the diagonal of each contraction determinant and $Y^{(m_i)}$ entries above it. Once the
intermediates exist, the cost of an element depends on the excitation ranks and on the number of
allowed $m_i$ distributions, not on the number of electrons or basis functions.

Fixed-rank kernels are compiled up to the total excitation rank `MAXEXCIT` (see
[Installation](../getting-started/installation.md#fixed-rank-wick-kernels)) and packed into AVX2
or AVX-512 lanes of four or eight determinant pairs. Higher ranks use generic fallbacks.

The intermediates scale as the fourth power of the number of orbitals for every reference pair.
They can be held in memory, shared between MPI ranks on a node, or stored in a disk-backed cache
([`wicks.storage`](../input/wicks.md)). With `wicks.compare = true`, every Wick element is checked
against the generalised Slater–Condon result.

## Factorised tables

The stochastic and selected methods also use spin-factorised overlap and one-body tables. Because
a determinant is a product of $\alpha$ and $\beta$ strings, its overlap factorises into
$\alpha$ and $\beta$ parts, which are precomputed and combined on demand.
These tables (`factor_tables`, `sapply_factor_tables`, `bapply_factor_tables`) can also live in
memory or on disk.
