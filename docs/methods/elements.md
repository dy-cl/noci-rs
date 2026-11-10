# Matrix elements

Every method needs overlaps, Hamiltonian and generalised-Fock matrix elements, and sometimes
transition densities, between pairs of determinants. Each pair goes to one of three engines,
depending on how the two determinants are related. On this page the bra is built on reference
$\langle{}^x\Psi|$ and the ket on reference $|{}^w\Psi\rangle$.

## Same orbital frame: Slater–Condon rules

Two determinants excited from the same reference share its orthonormal orbital frame. Their
overlap is one if they are identical and zero otherwise, and their Hamiltonian element follows from
the standard
Slater–Condon rules {cite}`mayer2003`. In the molecular orbitals of the frame, with antisymmetrised
integrals $\langle pq\Vert rs\rangle = \langle pq|rs\rangle - \langle pq|sr\rangle$,

$$
\langle{}^x\Psi|\hat H|{}^x\Psi\rangle = \sum_i h_{ii} + \frac12\sum_{ij}\langle ij\Vert ij\rangle,
\qquad
\langle{}^x\Psi|\hat H|{}^x\Psi_i^a\rangle = h_{ia} + \sum_j\langle ij\Vert aj\rangle,
\qquad
\langle{}^x\Psi|\hat H|{}^x\Psi_{ij}^{ab}\rangle = \langle ij\Vert ab\rangle,
$$

and determinants that differ by more than two orbitals do not couple. The same rules hold between
any two determinants of the frame, with the sums running over the orbitals occupied in both and the
sign of the permutation that aligns them. This is the cheapest case.

## Different frames: generalised Slater–Condon rules

For determinants from different frames, the occupied-orbital overlap matrix

$$
{}^{xw}S_{ij} = \langle{}^x\psi_i|{}^w\psi_j\rangle
= \sum_{\mu\nu}{}^xc^{\mu\cdot *}_{\cdot i}S_{\mu\nu}\,{}^wc^{\nu\cdot}_{\cdot j}
$$

is brought to diagonal form by a singular value decomposition,
${}^{xw}\mathbf S = \mathbf U\,\operatorname{diag}(s_i)\,\mathbf V^\dagger$, whose rotations give
Löwdin-paired orbitals ${}^x\tilde c^{\mu\cdot}_{\cdot i}$ and ${}^w\tilde c^{\mu\cdot}_{\cdot i}$
{cite}`lowdin1970` with $\langle{}^x\tilde\psi_i|{}^w\tilde\psi_j\rangle = s_i\delta_{ij}$. The
overlap, the reduced overlap and the co-densities are

$$
{}^{xw}S = \det\mathbf U\det\mathbf V^\dagger\prod_i s_i,
\qquad
{}^{xw}\tilde S = \det\mathbf U\det\mathbf V^\dagger\prod_{s_i\neq0}s_i,
$$

$$
{}^{xw}P_i^{\mu\nu} = {}^w\tilde c^{\mu\cdot}_{\cdot i}\,{}^x\tilde c^{\nu\cdot *}_{\cdot i},
\qquad
{}^{xw}W^{\mu\nu} = \sum_{s_i\neq0}\frac{1}{s_i}\,{}^{xw}P_i^{\mu\nu}.
$$

The one- and two-electron elements depend on how many singular values vanish
{cite}`thom2009,burton2019`:

$$
\langle{}^x\Psi|\hat H_1|{}^w\Psi\rangle
= {}^{xw}\tilde S\sum_{\mu\nu}h_{\mu\nu}
\begin{cases}
{}^{xw}W^{\nu\mu}, & \text{no } s_i = 0,\\
{}^{xw}P_i^{\nu\mu}, & s_i = 0 \text{ for one } i,\\
0, & \text{otherwise},
\end{cases}
$$

$$
\langle{}^x\Psi|\hat H_2|{}^w\Psi\rangle
= {}^{xw}\tilde S\sum_{\mu\nu\lambda\sigma}\langle\mu\lambda\Vert\nu\sigma\rangle
\begin{cases}
\tfrac12\,{}^{xw}W^{\nu\mu}\,{}^{xw}W^{\sigma\lambda}, & \text{no } s_i = 0,\\
{}^{xw}P_i^{\nu\mu}\,{}^{xw}W^{\sigma\lambda}, & s_i = 0 \text{ for one } i,\\
{}^{xw}P_i^{\nu\mu}\,{}^{xw}P_j^{\sigma\lambda}, & s_i = s_j = 0 \text{ for one pair } i, j,\\
0, & \text{otherwise}.
\end{cases}
$$

Zero singular values, which occur often between excited determinants, are therefore treated
explicitly rather than inverted. This route costs a decomposition and a contraction with the
two-electron integrals for every pair, $\mathcal O(n_\text{SO}^4)$, so it suits only a small
number of elements.

## Different frames: extended nonorthogonal Wick's theorem

When many excited determinants are built on the same pair of references, as in every post-NOCI
method, the extended nonorthogonal Wick's theorem {cite}`burton2021,burton2022` is much faster.
Each excited determinant is a string of operators on its reference,

$$
\langle{}^x\Psi_{ij\cdots}^{ab\cdots}|\hat O|{}^w\Psi_{kl\cdots}^{cd\cdots}\rangle
= \langle{}^x\Psi|\,{}^x\hat b_i^\dagger\,{}^x\hat b_j^\dagger\cdots{}^x\hat b_b\,{}^x\hat b_a\,
\hat O\,{}^w\hat b_c^\dagger\,{}^w\hat b_d^\dagger\cdots{}^w\hat b_l\,{}^w\hat b_k|{}^w\Psi\rangle,
$$

and only full contractions of the string survive between the two vacua. With $m$ zero-overlap
orbital pairs between the references, the fundamental contractions are

$$
{}^{wx}X^{(m)}_{qp}
= \sum_{\mu\nu\sigma\tau}{}^wc^{\cdot\mu *}_{q\cdot}S_{\mu\sigma}\,{}^{xw}M_{(m)}^{\sigma\tau}S_{\tau\nu}\,{}^xc^{\nu\cdot}_{\cdot p},
\qquad
{}^{xw}Y^{(m)}_{pq} =
\begin{cases}
{}^{xw}X^{(0)}_{pq} - {}^{xw}S_{pq}, & m = 0,\\
{}^{xw}X^{(1)}_{pq}, & m = 1,
\end{cases}
$$

with ${}^{xw}M_{(0)} = {}^{xw}W + {}^{xw}P + {}^{ww}P$ and ${}^{xw}M_{(1)} = {}^{xw}P$, where
${}^{xw}P = \sum_{s_k = 0}{}^{xw}P_k$, and both contractions vanish for $m > 1$. The overlap of two
excited determinants of total excitation rank $L$ is then a determinant of contractions,

$$
\langle{}^x\Psi_{ij\cdots}^{ab\cdots}|{}^w\Psi_{kl\cdots}^{cd\cdots}\rangle
= {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_L\\ m_1 + \cdots + m_L = m}}\det\mathbf D_\text{ov},
$$

summed over the distributions of the zero-overlap pairs among its columns. The contraction matrix
$\mathbf D_\text{ov}$ holds ${}^{\chi\chi'}X^{(m_i)}$ entries on and below the diagonal and
${}^{\chi\chi'}Y^{(m_i)}$ entries above it, with $\chi, \chi' \in \{x, w\}$ the references of its
row and column operators.

A one-body operator $\hat f = \sum_{pq}{}^xf_{pq}\,{}^x\hat b_p^\dagger\,{}^x\hat b_q$ adds one
row and one column. Expanding over them absorbs the sum over $p, q$ into the intermediates

$$
{}^xF_0^{(m)} = \sum_{pq}{}^xf_{pq}\,{}^{xx}X^{(m)}_{qp},
\qquad
{}^{\chi_r\chi_z}\mathcal F_{rz}^{(m_i, m_j)}
= \sum_{pq}{}^{\chi_rx}\mathcal A^{(m_i)}_{rp}\,{}^xf_{pq}\,{}^{x\chi_z}\mathcal B^{(m_j)}_{qz},
\qquad \mathcal A, \mathcal B \in \{X, Y\},
$$

so that

$$
\langle{}^x\Psi_{i\cdots}^{a\cdots}|\hat f|{}^w\Psi_{j\cdots}^{b\cdots}\rangle
= {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_{L+1}\\ m_1 + \cdots + m_{L+1} = m}}
\Big[{}^xF_0^{(m_1)}\det\mathbf D_\text{ov}
- \sum_z\det\mathbf D_\text{ov}^{z\rightarrow\boldsymbol{\mathcal F}_z}\Big],
$$

where $\mathbf D_\text{ov}^{z\rightarrow\boldsymbol{\mathcal F}_z}$ replaces the column of
creation operator $z$ by the intermediate column $\boldsymbol{\mathcal F}_z$. A two-body operator
$\hat v = \sum_{pqrs}{}^xv_{pqrs}\,{}^x\hat b_p^\dagger\,{}^x\hat b_q^\dagger\,{}^x\hat b_s\,{}^x\hat b_r$
adds two rows and two columns. Expanding over them gives a scalar, a one-column and a two-column
replacement,

$$
{}^xV_0^{(m_1, m_2)} = \sum_{pqrs}{}^xv_{pqrs}\big({}^{xx}X^{(m_1)}_{rp}\,{}^{xx}X^{(m_2)}_{sq}
- {}^{xx}X^{(m_1)}_{rq}\,{}^{xx}X^{(m_2)}_{sp}\big),
$$

$$
{}^{\chi_\eta\chi_z}\mathcal V^{(m_1, m_2, m_3)}_{\eta z}
= \sum_{ps}{}^{\chi_\eta x}\mathcal B^{(m_1)}_{\eta p}\big({}^xJ^{(m_2)}_{ps} - {}^xK^{(m_2)}_{ps}\big)
{}^{x\chi_z}\mathcal A^{(m_3)}_{sz},
$$

$$
{}^{\chi_\eta\chi_z,\chi_\xi\chi_y}\mathcal J_{\eta z,\xi y}
= \sum_{pqrs}{}^xv_{pqrs}
\begin{vmatrix}
{}^{\chi_\eta x}\mathcal C_{\eta p} & {}^{\chi_\eta x}\mathcal D_{\eta q}\\
{}^{\chi_\xi x}\mathcal C_{\xi p} & {}^{\chi_\xi x}\mathcal D_{\xi q}
\end{vmatrix}
\begin{vmatrix}
{}^{x\chi_z}\mathcal A_{rz} & {}^{x\chi_y}\mathcal B_{ry}\\
{}^{x\chi_z}\mathcal A_{sz} & {}^{x\chi_y}\mathcal B_{sy}
\end{vmatrix},
$$

where $J$ and $K$ are Coulomb- and exchange-like contractions of the integrals with a fundamental
contraction, and the zero-overlap labels of $\mathcal J$ are suppressed. The two-body element is

$$
\langle{}^x\Psi_{i\cdots}^{a\cdots}|\hat v|{}^w\Psi_{j\cdots}^{b\cdots}\rangle
= {}^{xw}\tilde S\sum_{\substack{m_1,\ldots,m_{L+2}\\ m_1 + \cdots + m_{L+2} = m}}
\Big[{}^xV_0^{(m_1, m_2)}\det\mathbf D_\text{ov}
- 2\sum_z\det\mathbf D_\text{ov}^{z\rightarrow\boldsymbol{\mathcal V}_z}
+ \sum_{z<y}\sum_{\eta<\xi}\phi^{zy}_{\eta\xi}\,\mathcal J_{\eta z,\xi y}\det\mathbf D_\text{ov}[\eta,\xi|z,y]\Big],
$$

with $\mathbf D_\text{ov}[\eta,\xi|z,y]$ the minor without rows $\eta, \xi$ and columns $z, y$,
and $\phi^{zy}_{\eta\xi}$ its cofactor sign.

The intermediates depend only on the pair of references, so `noci-rs` builds them once for each
ordered pair. The cost of an element then depends on its excitation rank $L$ and on the number of
allowed $m_i$ distributions, not on the number of electrons or basis functions.

Fixed-rank kernels are compiled up to the total excitation rank `MAXEXCIT` (see
[Installation](../getting-started/installation.md#fixed-rank-wick-kernels)) and packed into AVX2
or AVX-512 lanes of four or eight determinant pairs. Higher ranks use generic fallbacks.

The intermediates scale as the fourth power of the number of orbitals for every reference pair.
They can be held in memory, shared between MPI ranks on a node, or stored in a disk-backed cache
([`wicks.storage`](../input/wicks.md)). With `wicks.compare = true`, every Wick element is checked
against the generalised Slater–Condon result.

## Factorised tables

The stochastic and selected methods also use spin-factorised overlap and one-body tables. All
references are spin-collinear, so a determinant $w = (a_w, b_w)$ is a product of an $\alpha$ string
$a_w$ and a $\beta$ string $b_w$, and the overlap between determinants of parents $Q$ and $P$
factorises:

$$
S_{wx} = A^{QP}_{a_w,a_x}B^{QP}_{b_w,b_x}.
$$

Determinants that share a string share its factor, so a sum over many pairs needs far fewer
same-spin factors than determinant overlaps. Arranging the source amplitudes of parent $P$ by
string, $\Delta^P_{ab}$, the overlap action contracts one spin first,

$$
\sum_{x\in P}S_{wx}\Delta_x = \sum_{ab}A^{QP}_{a_w,a}\Delta^P_{ab}B^{QP}_{b_w,b}
= \sum_b\Theta^{QP}_{a_w,b}B^{QP}_{b_w,b},
\qquad
\Theta^{QP}_{\bar a,b} = \sum_a A^{QP}_{\bar a,a}\Delta^P_{ab},
$$

or the other, choosing the cheaper order for each pair of parents. One-body elements factorise in
the same way, into same-spin one-body factors $F^{\omega,QP}$ alongside the overlap factors,

$$
F_{wx} = F^{\alpha,QP}_{a_w,a_x}B^{QP}_{b_w,b_x} + A^{QP}_{a_w,a_x}F^{\beta,QP}_{b_w,b_x}.
$$

These
tables (`factor_tables`, `sapply_factor_tables`, `bapply_factor_tables`) can live in memory or on
disk.
