# NOCIQMC

NOCIQMC finds the ground state in the extended nonorthogonal space by imaginary-time propagation,
as FCIQMC does in an orthogonal determinant space {cite}`booth2009`. Its populations are sparse,
and the matrix elements it needs are evaluated on demand, so the Hamiltonian and overlap of the
extended space are never built. A small set of references describes the static correlation, so
NOCIQMC needs far fewer populated determinants than CIQMC at the same accuracy (see
[Results](../results.md)).

## Imaginary-time propagation in a nonorthogonal space

Imaginary-time evolution $e^{-\tau(\hat H - E_s)}$ projects any starting state onto the ground
state. In a nonorthogonal basis with coefficients $\mathbf c$, its first-order step reads

$$
\mathbf S\,\frac{\mathrm d\mathbf c}{\mathrm d\tau} = -(\mathbf H - E_s\mathbf S)\,\mathbf c,
$$

which would require $\mathbf S^{-1}$ at every step. Every propagator in `noci-rs` instead
applies a first-order map whose fixed points satisfy $\mathbf H\mathbf v = E\,\mathbf S\mathbf v$,
so the stationary state is the NOCI ground state without inverting the overlap. The propagators
differ in what they store and in how they treat the overlap null space, the redundant directions
of an overcomplete space.

All three propagators use one population-control shift $E_s$, which multiplies the overlap
{cite}`baumgarten2023`.

### Unshifted

The `unshifted` propagator evolves the coefficients directly:

$$
\mathbf c' = \mathbf c - \Delta\tau(\mathbf H - E_s\mathbf S)\,\mathbf c.
$$

It never applies the overlap exactly, so its populations stay as sparse as the wavefunction.
However, a vector in $\operatorname{null}(\mathbf S)$ is left unchanged by this map, since
$\mathbf H$ and $\mathbf S$ both annihilate it, and the shift has no grip on it. Sampling noise
deposited there therefore grows without bound. The unshifted propagator is valid only when the
overlap of the determinant space has full rank, and works best when its smallest eigenvalue is
not tiny.

### S-apply

The range propagators store the *range population*

$$
\mathbf N = \mathbf S\mathbf c,
$$

which lies in the physical range of the overlap and contains no null-space component. S-apply
samples the same residual as the unshifted propagator from a sparse sample $\tilde{\mathbf N}$
of the population and applies the overlap exactly to the result:

$$
\boldsymbol\Delta \simeq -\Delta\tau(\mathbf H - E_s\mathbf S)\,\tilde{\mathbf N},
\qquad
\mathbf N' = \mathbf N + \mathbf S\boldsymbol\Delta.
$$

Every change $\mathbf S\boldsymbol\Delta$ lies in $\operatorname{range}(\mathbf S)$, and the
initial population $\mathbf N_0 = \mathbf S\mathbf c_0$ does too, so by induction the population
never acquires a null-space component, with no diagonalisation or inversion of $\mathbf S$. The
cost is the outer overlap action, which slows the small range directions of an ill-conditioned
basis.

### B-apply

B-apply removes that extra overlap action. An auxiliary factor $\mathbf B$, mapping the
nonorthogonal space into an auxiliary orthogonal determinant space indexed by $p, q$, obeys

$$
\mathbf B^\dagger\mathbf B = \mathbf S,
\qquad
\mathbf B^\dagger\hat{\mathbf H}\mathbf B = \mathbf H.
$$

B-apply samples the orthogonal-space action and maps it back exactly:

$$
\boldsymbol\chi \simeq -\Delta\tau(\hat{\mathbf H} - E_s)\,\mathbf B\tilde{\mathbf N},
\qquad
\mathbf N' = \mathbf N + \mathbf B^\dagger\boldsymbol\chi.
$$

Its changes also lie in the range of the overlap. In expectation the update is
$\mathbf N' = \mathbf N - \Delta\tau(\mathbf H - E_s\mathbf S)\mathbf N$: the unshifted map,
applied to the range population, without the slowing of S-apply's second overlap. B-apply is
usually the best choice for overcomplete spaces.

### Choosing a propagator

| | `unshifted` | `s-apply` | `b-apply` |
| --- | --- | --- | --- |
| Stored population | Coefficients $\mathbf c$ | $\mathbf N = \mathbf S\mathbf c$ | $\mathbf N = \mathbf S\mathbf c$ |
| Overcomplete spaces | No | Yes | Yes |
| Slow modes from ill-conditioning | No | Yes | No |
| Excitation generators | `uniform`, `heat-bath`, `overlap-weighted` | `uniform`, `overlap-weighted` | `uniform`, `pchb` |
| Shift controller | Logarithmic | Newton (tangent) | Newton (tangent) |

[Practical advice](../guide/advice.md#choosing-a-propagator) discusses the choice further.

## Deterministic propagation

The `det` table runs the same propagators exactly, on the full Hamiltonian and overlap of the
extended space. It is meant for small systems, validation and analysis of the propagators,
giving noise-free reference trajectories for the stochastic code. Each step applies the
expectation of the stochastic update, renormalises the vector in the overlap metric, and
optionally updates the shift from the population of the propagated vector,

$$
E_s \leftarrow E_s - \frac{\alpha}{\Delta\tau}\ln\frac{\lVert\mathbf v_{k+1}\rVert_1}{\lVert\mathbf v_k\rVert_1},
$$

with $\alpha$ = `dynamic_shift_alpha`. The run stops when the energy changes by less than
`e_tol` between steps, or after `max_steps`.

For diagnostics the overlap can be diagonalised, $\mathbf S = \mathbf U\boldsymbol\Lambda\mathbf U^\dagger$,
splitting the space into a *relevant* subspace $\mathbf U_r$, with eigenvalues above
`projector_eps`, and a *null* subspace $\mathbf U_n$. The propagator is then resolved into its
relevant–relevant, relevant–null, null–relevant and null–null blocks, for example
$\mathbf U_{rn} = -\Delta\tau\,\mathbf U_r^\dagger\mathbf A\mathbf U_n$ with
$\mathbf A = \mathbf S(\mathbf H - E_s\mathbf S)$ for S-apply and $\mathbf H - E_s\mathbf S$
otherwise. This shows directly how each propagator treats redundant null-space population.
Deterministic propagation also supports holomorphic SCF states.

## Stochastic propagation

The `qmc` table enables stochastic NOCIQMC. Populations are real-valued and distributed over MPI
ranks by determinant. Each iteration:

1. draws a sparse, conditionally unbiased sample $\tilde{\mathbf N}$ of the persistent population;
2. generates excitations from each sampled determinant with the chosen excitation generator and
   evaluates the required $H$ and $S$ elements on demand;
3. exchanges and annihilates the generated updates across ranks;
4. applies the update to the persistent population, through $\mathbf S$ or $\mathbf B^\dagger$
   for the range propagators.

A *report* is `ncycles` iterations, after which energies, populations and the shift are updated
and printed. A run lasts `nreports` reports.

### Projected energy

The energy estimator is the projected energy against a fixed trial state
$|\Psi_P\rangle = \sum_i c_i^P|i\rangle$,

$$
E_\text{proj} = \frac{\sum_w v_w H^P_w}{\sum_w v_w S^P_w},
\qquad
H^P_w = \sum_i c_i^P H_{iw},
\quad
S^P_w = \sum_i c_i^P S_{iw},
$$

where $\mathbf v$ is the coefficient or range population. The contractions $H^P_w$ and $S^P_w$
are built once, so each estimate costs one pass over the population. By default the trial is the
reference NOCI state. `n_projected` enlarges it: the references are kept and the determinants with
the largest residual $\langle w|(\hat H - E_0)|\Psi_0\rangle$ are added, up to that dimension,
then the NOCI problem is re-solved in the enlarged space. A better trial reduces the noise of
$E_\text{proj}$. The denominator, normalised as `EProjDen/NRange`, is the trial overlap, a useful
convergence diagnostic, though it does not prove convergence of every component of the
stochastic wavefunction.

### Population control

The population first grows at a fixed shift, `initial_shift`, until it reaches
`target_population`. The shift then varies to hold the population steady.

The unshifted propagator uses the logarithmic controller of FCIQMC {cite}`booth2009`,

$$
E_s \leftarrow E_s - \frac{\zeta}{A\,\delta\tau}\ln\frac{N}{N^\text{prev}},
$$

with damping $\zeta$ = `shift_damping` and $A\,\delta\tau$ the imaginary time per report
(`ncycles` × `dt`). It holds the population at zero growth from wherever control started.

The range propagators use a damped Newton update with a restoring term towards the target
$N_*$, of strength $\kappa$ = `population_restoring`. The shift enters through
$E_s\mathbf S$ rather than as a simple death term, so the derivative of the population with
respect to the shift is not just proportional to the population. S-apply and B-apply therefore
propagate the *shift tangent* $\mathbf T = \partial\mathbf N'/\partial E_s$ from the same sampled
paths, and step

$$
E_s \leftarrow E_s - \zeta\,N_M\Big(\frac{\partial N_M}{\partial E_s}\Big)^{-1}
\left[\ln\frac{N_M}{N_M^\text{prev}} + \kappa\ln\frac{N_M^\text{prev}}{N_*}\right],
\qquad
\frac{\partial N_M}{\partial E_s} = \operatorname{sign}(\mathbf N')^{\mathsf T}\mathbf T,
$$

with $N_M = \lVert\mathbf N'\rVert_1$. With $\kappa = 0$ the controller holds the population at
zero growth; positive $\kappa$ restores it towards the target.

B-apply can also add report-level heavy-ball momentum {cite}`polyak1964`,
$\mathbf V_{r+1} = \beta\mathbf V_r + \mathbf B^\dagger\boldsymbol\chi_r$, with
$\beta$ = `momentum_beta`.

### The shoulder

Before population control starts, the population passes through an *annihilation plateau*, the
shoulder: growth stalls while the sign structure of the wavefunction establishes itself, and then
resumes. The shoulder height is the population the method needs to resolve the sign problem, and
the ratio of total to reference population peaks there. `target_population` must lie above the
shoulder, or the energy is biased. The [Results](../results.md) page compares NOCIQMC and CIQMC
shoulders.

### Fast randomised iteration

Populations and updates are compressed by fast randomised iteration (FRI) {cite}`lim2017,greene2019`,
which replaces a vector $\mathbf x$ by a sparse vector $\Phi(\mathbf x)$ with
$\mathbb E[\Phi(\mathbf x)] = \mathbf x$, so the propagation stays unbiased in expectation.

- **Fixed-cutoff** compression keeps every element with $|x_i| \ge c$ and keeps smaller ones with
  probability $p_i = |x_i|/c$, rescaled to $x_i/p_i$. This is used for the persistent population
  (`fri.population.cutoff`) and for generated spawns (`fri.spawn.cutoff`).
- **Target-NNZ** compression solves $\sum_i\min(1, |x_i|/c) = m$ for the cutoff, so the result has
  at most $m$ nonzeros with minimal expected $L^2$ error among unbiased $m$-sparse approximations.
  This is used for the coalesced pre-overlap change of each report (`fri.pre_overlap.target_nnz`)
  and the shift tangent (`fri.shift_tangent.target_nnz`).

Selection uses pivotal sampling {cite}`weare2026`, which has the same coordinate-wise expected
error as independent Bernoulli sampling but reduces fluctuations in the retained count. FRI
reduces variance in many cases, but not universally.

### Excitation generators

Each sampled determinant spawns onto determinants drawn from a proposal distribution, and the
spawn is weighted by the inverse of the generation probability.

- **`uniform`** picks children uniformly from the whole determinant space.
- **`heat-bath`** draws children with probability proportional to $|H_{xw} - E_s S_{xw}|$ exactly
  {cite}`holmes2016`. It is very expensive and rarely worthwhile.
- **`overlap-weighted`** mixes uniform sampling with a factorised proposal proportional to
  $|S_{wx}|$. `overlap_weight` sets the probability $0 \le p < 1$ of the overlap branch, and
  `optimise_overlap_weight` adapts it between reports from the sampled second moment.
- **`pchb`** is B-apply's precomputed heat-bath generator within each parent's orthogonal frame
  {cite}`holmes2016,weser2023`. Doubles pick an occupied hole pair with probability $W_{ij}/Z$,
  where $W_{ij} = \sum_{ab}|\langle ai\Vert jb\rangle|$, or $|(ai|jb)|$ for $\alpha\beta$ holes,
  and then a particle pair from the empty orbitals in proportion to its integral. Singles pick a
  hole with probability $S_i/Z$, with $S_i = \sum_a s_i^a$ and the precomputed bound
  $s_i^a = |h_{ai}| + \sum_j|(ai\Vert jj)| + \sum_{\bar j}|(ai|\bar j\bar j)|$ on
  $|H_{a\leftarrow i,x}|$, and then a particle in proportion to $s_i^a$. From a parent's own
  reference determinant, $s_i^a$ is replaced by the exact $|H_{a\leftarrow i,0}|$, so singles that
  vanish, for example by Brillouin's theorem, are not sampled. The split
  $p_\text{single} = \sum_i S_i/Z$, with $Z = \sum_i S_i + \sum_{ij}W_{ij}$, is chosen
  automatically for every source. PCHB weights need $O(n^4)$ memory per parent, which is printed at
  startup.

Stochastic NOCIQMC currently supports real SCF states only.
