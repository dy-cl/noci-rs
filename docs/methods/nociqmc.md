# NOCIQMC

NOCIQMC finds the ground state in the extended nonorthogonal space by imaginary-time propagation,
as FCIQMC does in an orthogonal determinant space {cite}`booth2009`. Its populations are sparse,
and the matrix elements it needs are evaluated on demand, so the Hamiltonian and overlap of the
extended space are never built. A small set of references describes the static correlation, so
NOCIQMC needs far fewer populated determinants than CIQMC at the same accuracy (see
[Results](../results.md)).

## Imaginary-time propagation in a nonorthogonal space

The imaginary-time Schrödinger equation with a shift,
$-\partial_\tau|\Psi(\tau)\rangle = [\hat H - E_s^S(\tau)]|\Psi(\tau)\rangle$, projects any starting
state onto the ground state. Taking one first-order step of length $\Delta\tau$, expanding
$|\Psi(\tau)\rangle = |{}^x\Psi\rangle c_x(\tau)$ and projecting onto $\langle{}^w\Psi|$ gives

$$
S_{wx}c_x(\tau + \Delta\tau) = S_{wx}c_x(\tau) - \Delta\tau\big[H_{wx} - E_s^S(\tau)S_{wx}\big]c_x(\tau),
$$

which would require the inverse overlap at every step. The overlap is often singular or too
ill-conditioned to invert, and inverting it would cost as much as solving the generalised
eigenvalue problem directly. Every propagator in `noci-rs` instead applies a first-order map whose
fixed points satisfy $[H_{wx} - E_s^S S_{wx}]c_x = 0$, so the stationary state is the NOCI ground
state without inverting the overlap. The propagators differ in what they store and in how they
treat the overlap null space, the redundant directions of an overcomplete space
([Extended determinant spaces](noci.md#extended-determinant-spaces)).

All three propagators use one population-control shift $E_s^S(\tau)$, which multiplies the overlap
{cite}`baumgarten2023`.

### Unshifted

The `unshifted` propagator evolves the coefficients directly,

$$
c_w(\tau + \Delta\tau) = U^{(0,\eta)}_{wx}(\Delta\tau)\,c_x(\tau),
\qquad
U^{(0,\eta)}_{wx}(\Delta\tau) = \delta_{wx} - \Delta\tau\big[H_{wx} - E_s^S(\tau)S_{wx}\big].
$$

It never applies the overlap exactly, so its populations stay as sparse as the wavefunction.
However, a null-space vector $c_x \in \ker S$ is also in $\ker H$, so it is stationary,

$$
U^{(0,\eta)}_{wx}(\Delta\tau)\,c_x = c_w,
$$

and the shift has no grip on it. Sampling noise deposited there performs a random walk that grows
without bound. The unshifted propagator is valid only when the overlap of the determinant space
has full rank, and works best when its smallest eigenvalue is not tiny.

### $S$-apply

The range propagators store the *range population*, initialised from the reference NOCI
eigenvector $c_x(0)$,

$$
N_w(0) = S_{wx}c_x(0) = \langle{}^w\Psi|\Psi_\text{NOCI}\rangle,
$$

which lies in $\operatorname{range}(S)$ without any inversion or diagonalisation of $S$. It is
propagated as a coefficient vector, $|\Psi(\tau)\rangle = |{}^w\Psi\rangle N_w(\tau)$. $S$-apply
applies the overlap exactly to every population change:

$$
N_w(\tau + \Delta\tau) = \big[\delta_{wy} - \Delta\tau\,S_{wx}\big(H_{xy} - E_s^S(\tau)S_{xy}\big)\big]N_y(\tau),
$$

sampling the pre-overlap change $\Delta b_x = -\Delta\tau[H_{xy} - E_s^S(\tau)S_{xy}]N_y(\tau)$ and
then adding $S_{wx}\Delta b_x$. Every change lies in $\operatorname{range}(S)$, so by induction the
population never acquires a null-space component. The cost is the extra overlap action. On
$\operatorname{range}(S)$ the $S$-apply propagation matrix is similar to

$$
K_{wz} = (S^{1/2})_{wx}\big(H_{xy} - E_0S_{xy}\big)(S^{1/2})_{yz},
$$

so an overlap eigenvector with small eigenvalue $\sigma_i$ is suppressed by an extra factor of
$\sigma_i$, giving very slowly converging physical modes in an ill-conditioned space.

### $B$-apply

$B$-apply keeps the range constraint without the extra overlap action. Each parent $P$ has an
auxiliary orthogonal determinant space $\{|D^P\rangle\}$ in its own orthonormal orbitals, containing
its NOCI determinants and every determinant connected to them by $\hat H$. With $p \equiv (P, D)$,
the transformation $\mathcal B$, its same-parent block $\mathcal E$ and the block-diagonal auxiliary
Hamiltonian $\mathcal H$ are

$$
\mathcal B_{px} = \langle D^P|{}^x\Psi\rangle,
\qquad
\mathcal E_{px} = \delta_{PP(x)}\delta_{Dx},
\qquad
\mathcal H_{pq} = \delta_{PQ}\langle D^P|\hat H|D'^P\rangle,
$$

with $P(x)$ the parent of determinant $x$. They factorise the NOCI matrices,

$$
\mathcal B^*_{pw}\mathcal E_{px} = S_{wx},
\qquad
\mathcal B^*_{pw}\mathcal H_{pq}\mathcal E_{qx} = H_{wx},
$$

because $\hat H|{}^x\Psi\rangle$ lies entirely within the auxiliary space of $P(x)$. $B$-apply
generates an auxiliary population change and transforms it back into the range population,

$$
\chi_p(\tau) = -\Delta\tau\big[\mathcal H_{pq} - E_s^S(\tau)\delta_{pq}\big]\mathcal E_{qy}N_y(\tau),
\qquad
N_w(\tau + \Delta\tau) = N_w(\tau) + \mathcal B^*_{pw}\chi_p(\tau),
$$

which in expectation is

$$
N_w(\tau + \Delta\tau) = \big[\delta_{wy} - \Delta\tau\big(H_{wy} - E_s^S(\tau)S_{wy}\big)\big]N_y(\tau).
$$

The change $\mathcal B^*_{pw}\chi_p$ is a vector of overlaps with a state, so it also lies in
$\operatorname{range}(S)$, but the residual is no longer premultiplied by the overlap. Slow modes
remain where the overlap is ill-conditioned: for a normalised overlap eigenvector $v^{(i)}_x$ with
eigenvalue $\sigma_i$,

$$
0 \le v^{(i)*}_w\big(H_{wx} - E_0S_{wx}\big)v^{(i)}_x \le \sigma_i\big(E_\text{max} - E_0\big),
$$

but they carry one factor of $\sigma_i$ fewer than in $S$-apply. A further advantage is that
spawning needs only ordinary Slater–Condon elements within each parent: the coupling between
parents enters only through the overlaps of the back-transformation. $B$-apply is usually the best
choice for overcomplete spaces.

Both range propagators have the same stationary points: since
$\ker S \subseteq \ker H$, a stationary $S$-apply residual lies in both $\ker S$ and
$\operatorname{range}(S)$, and therefore vanishes, so both satisfy

$$
\big[H_{wy} - E_s^S(\tau)S_{wy}\big]N_y(\tau) = 0.
$$

### Choosing a propagator

| | `unshifted` | `s-apply` | `b-apply` |
| --- | --- | --- | --- |
| Stored population | Coefficients $c_x(\tau)$ | $N_w(\tau) = S_{wx}c_x(\tau)$ | $N_w(\tau) = S_{wx}c_x(\tau)$ |
| Overcomplete spaces | No | Yes | Yes |
| Extra slow modes from ill-conditioning | No | Yes | No |
| Excitation generators | `uniform`, `heat-bath`, `overlap-weighted` | `uniform`, `overlap-weighted` | `uniform`, `pchb` |
| Shift controller | Logarithmic | Newton (tangent) | Newton (tangent) |

[Practical advice](../guide/advice.md#choosing-a-nociqmc-propagator) discusses the choice further.

## Deterministic propagation

The `det` table runs the same propagators exactly, on the full Hamiltonian and overlap of the
extended space. It is meant for small systems, validation and analysis of the propagators, giving
noise-free reference trajectories for the stochastic code. Each step applies the expectation of
the stochastic update to the propagated vector $v_w(\tau)$, either $c_w(\tau)$ or $N_w(\tau)$,
renormalises it in the overlap metric, and optionally updates the shift from its population,

$$
E_s^S(\tau + \Delta\tau) = E_s^S(\tau)
- \frac{\alpha}{\Delta\tau}\ln\frac{\sum_w|v_w(\tau + \Delta\tau)|}{\sum_w|v_w(\tau)|},
$$

with $\alpha$ = `dynamic_shift_alpha`. The run stops when the energy changes by less than `e_tol`
between steps, or after `max_steps`.

For diagnostics the overlap can be diagonalised, $S_{wx} = \sum_k u^{(k)}_w\sigma_k u^{(k)*}_x$,
splitting its eigenvectors into a *relevant* set $r$, with $\sigma_k$ above `projector_eps`, and a
*null* set $n$. The propagator is then resolved into its relevant–relevant, relevant–null,
null–relevant and null–null blocks, for example

$$
T^{rn}_{kl} = -\Delta\tau\,u^{(k)*}_wA_{wx}u^{(l)}_x,
\qquad k \in r,\; l \in n,
$$

with $A_{wx} = S_{wy}(H_{yx} - E_s^S S_{yx})$ for $S$-apply and $A_{wx} = H_{wx} - E_s^S S_{wx}$
otherwise. This shows directly how each propagator treats redundant null-space population.
Deterministic propagation also supports holomorphic SCF states.

## Stochastic propagation

The `qmc` table enables stochastic NOCIQMC. Populations are real-valued and distributed over MPI
ranks by determinant. A *report* is $n_\text{cyc}$ = `ncycles` stochastic cycles, during which the
persistent population $N_w(\tau)$ is held fixed. In each cycle $k$:

1. a sparse, unbiased sample $\bar N_w^{(k)}(\tau)$ of the persistent population is drawn;
2. each sampled determinant updates itself exactly on the diagonal and spawns onto children drawn
   by the excitation generator, with the required matrix elements evaluated on demand;
3. the spawns are exchanged and annihilated across ranks and accumulated.

At the end of the report the accumulated change is applied to the persistent population, through
the overlap or $\mathcal B^\dagger$ for the range propagators, and the energies, populations and
shift are updated and printed. A run lasts `nreports` reports.

The off-diagonal contribution of a sampled determinant $y$ is divided over
$n^{(k)}(y) = \max[1, \lceil|\bar N_y^{(k)}|\rceil]$ attempts of weight
$n_y^{(k)} = \bar N_y^{(k)}/n^{(k)}(y)$. In $S$-apply, an attempt that draws child $x$ with
probability $P_\text{Gen}^S(x|y)$ adds

$$
\delta^{(k,r)}_{x\leftarrow y} = -\frac{\Delta\tau\big[H_{xy} - E_s^S(\tau)S_{xy}\big]n_y^{(k)}}{P_\text{Gen}^S(x|y)}
$$

to the pre-overlap change, and in $B$-apply an attempt that draws auxiliary determinant $D$ with
probability $P_\text{Gen}^B(D|x)$ adds

$$
\delta^{(k,r)}_{D\leftarrow x} = -\frac{\Delta\tau\,H_{Dx}\,n_x^{(k)}}{P_\text{Gen}^B(D|x)}
$$

to the auxiliary change.

### Projected energy

The energy estimator is the projected energy against a fixed trial state
$|\Psi_T\rangle = \sum_{i\in\mathcal T}c_i^T|{}^i\Psi\rangle$,

$$
E_\text{proj}(\tau) = \frac{\sum_x H_x^T N_x(\tau)}{\sum_x S_x^T N_x(\tau)},
\qquad
H_x^T = \sum_{i\in\mathcal T}c_i^TH_{ix},
\quad
S_x^T = \sum_{i\in\mathcal T}c_i^TS_{ix},
$$

with $N_x(\tau)$ replaced by $c_x(\tau)$ for the unshifted propagator. The contractions $H_x^T$
and $S_x^T$ are built once, so each estimate costs one pass over the population. By default the
trial is the reference NOCI state. `n_projected` enlarges it: the references are kept and the
external determinants with the largest residual

$$
R_w = \sum_{i\in r}\big[H_{wi} - E_0S_{wi}\big]c_i^{(0)}
$$

are added, up to that dimension, where $c_i^{(0)}$ and $E_0$ are the reference NOCI ground state
on the reference set $r$. The NOCI problem is then re-solved in the enlarged space. A better trial
reduces the noise of $E_\text{proj}$. The normalised trial overlap

$$
O_T(\tau) = \frac{\sum_x S_x^T N_x(\tau)}{N_\text{tot}(\tau)},
\qquad
N_\text{tot}(\tau) = \sum_w|N_w(\tau)|,
$$

printed as `EProjDen/NRange`, is a useful convergence diagnostic, though it does not prove
convergence of every component of the stochastic wavefunction.

### Population control

The shift is held at `initial_shift` while the population grows, until $N_\text{tot}(\tau)$
reaches $N_\text{target}$ = `target_population`. The shift then varies to hold the population
steady, once per report.

The unshifted propagator uses the logarithmic controller of FCIQMC {cite}`booth2009`,

$$
E_s^S(\tau + n_\text{cyc}\Delta\tau) = E_s^S(\tau)
- \frac{\zeta}{n_\text{cyc}\Delta\tau}\ln\frac{N_\text{tot}(\tau + n_\text{cyc}\Delta\tau)}{N_\text{tot}(\tau)},
$$

with damping $\zeta$ = `shift_damping`. It holds the population at zero growth from wherever
control started.

That controller assumes an orthogonal basis, in which
$\partial N_\text{tot}/\partial E_s^S = n_\text{cyc}\Delta\tau N_\text{tot}$. In the range
propagators the shift multiplies the overlap, so its effect passes through one ($B$-apply) or two
($S$-apply) overlap actions. They therefore propagate the range-population tangents over each
report from the same sampled populations,

$$
J_w^B = \frac{\partial N_w'}{\partial E_s^S} = \Delta\tau\,S_{wy}\sum_{k=1}^{n_\text{cyc}}\bar N_y^{(k)},
\qquad
J_w^S = \frac{\partial N_w'}{\partial E_s^S} = \Delta\tau\,S_{wx}S_{xy}\sum_{k=1}^{n_\text{cyc}}\bar N_y^{(k)},
$$

giving $\partial N_\text{tot}/\partial E_s^S = \sum_w\operatorname{sgn}(N_w')J_w$, and apply the
damped Newton update

$$
E_s^S(\tau + n_\text{cyc}\Delta\tau) = E_s^S(\tau)
- \zeta\,\frac{N_\text{tot}(\tau + n_\text{cyc}\Delta\tau)}{\partial N_\text{tot}/\partial E_s^S}
\left\{\ln\frac{N_\text{tot}(\tau + n_\text{cyc}\Delta\tau)}{N_\text{tot}(\tau)}
+ \kappa\ln\frac{N_\text{tot}(\tau)}{N_\text{target}}\right\},
$$

with restoring strength $\kappa$ = `population_restoring`. With $\kappa = 0$ the controller holds
the population at zero growth; positive $\kappa$ weakly restores it towards $N_\text{target}$.

$B$-apply can also add report-level heavy-ball momentum {cite}`polyak1964` to the
back-transformed change, $V_w^{(r+1)} = \beta V_w^{(r)} + \mathcal B^*_{pw}\bar\chi_p^{(r)}$ for
report $r$, with $\beta$ = `momentum_beta`.

### The shoulder

Before population control starts, the population passes through a plateau: growth stalls while
the sign structure of the wavefunction establishes itself, and then resumes. Where the plateau is
less clearly resolved it appears as a *shoulder*, a maximum of
$\rho(\tau) = N_\text{tot}(\tau)/N_\text{ref}(\tau)$ against $N_\text{tot}(\tau)$ followed by a
sharp downturn, with $N_\text{ref}(\tau) = \sum_{x\in r}|N_x(\tau)|$ the population on the
references. Its height is the population the method needs to resolve the sign problem, so
$N_\text{target}$ must lie above it, or the energy is biased. The [Results](../results.md) page
compares NOCIQMC and CIQMC shoulders.

### Fast randomised iteration

The overlap and $\mathcal B^\dagger$ columns are dense, so even a sparse change produces a dense
population. Populations and changes are therefore compressed by fast randomised iteration (FRI)
{cite}`lim2017,greene2019`. For a vector $z_i$ and a cutoff $c$, element $i$ is kept with
probability $p_i(c)$ and rescaled,

$$
p_i(c) = \min\Big(1, \frac{|z_i|}{c}\Big),
\qquad
\Phi_c(z_i) =
\begin{cases}
z_i, & p_i(c) = 1,\\
I_i\,z_i/p_i(c), & p_i(c) < 1,
\end{cases}
$$

with $I_i \in \{0, 1\}$ and $\mathbb E[I_i] = p_i(c)$. Elements with $|z_i| \ge c$ are kept
exactly, and smaller ones are set to zero or rounded up to $\pm c$. The map is unbiased,
$\mathbb E[\Phi_c(z_i)\,|\,z_i] = z_i$, with conditional variance

$$
\operatorname{Var}\big[\Phi_c(z_i)\,|\,z_i\big] =
\begin{cases}
0, & |z_i| \ge c,\\
|z_i|\big(c - |z_i|\big), & |z_i| < c.
\end{cases}
$$

Where a target number $m$ of nonzeros is given instead, the cutoff solves
$\sum_i\min(1, |z_i|/c) = m$, so at most $m$ elements are retained. The retention events are drawn
jointly by pivotal sampling {cite}`deville1998,weare2026`, which preserves every marginal $p_i(c)$
and the coordinate-wise variance of independent rounding while reducing fluctuations in the
retained count. The compressions are applied at four sites:

| Site | Compression | Input |
| --- | --- | --- |
| Persistent population, $\bar N_w^{(k)} = \Phi_{c_1}(N_w)$ each cycle | Cutoff $c_1$ | `fri.population.cutoff` |
| Individual spawns, before they are combined | Cutoff $c_s$ | `fri.spawn.cutoff` |
| Report change $\sum_k\Delta b_x^{(k)}$ or $\sum_k\chi_p^{(k)}$, before the range transformation | Target $m_\Delta$ | `fri.pre_overlap.target_nnz` |
| Shift tangent, before its range transformation | Target $m_T$ | `fri.shift_tangent.target_nnz` |

Each compression preserves the conditional expectation of its input, and the transformations that
follow are linear, so none of them changes the expected population change. FRI reduces variance in
many cases, but not universally.

### Excitation generators

Each spawning attempt draws a child from a proposal distribution and divides the spawn by its
generation probability.

- **`uniform`** picks children uniformly from the whole determinant space,
  $q_U(w|x) = 1/(N_\text{dets} - 1)$ for $w \neq x$. Same-parent pairs that differ by more than two
  orbitals are recognised and skipped without evaluating matrix elements.
- **`heat-bath`** draws children with probability proportional to $|H_{wx} - E_s^S S_{wx}|$
  exactly {cite}`holmes2016`. It is very expensive and rarely worthwhile.
- **`overlap-weighted`** mixes uniform sampling with a proposal proportional to the cross-parent
  overlap. For a source $x = (a_x, b_x)$ of parent $P$ and a target $w = (a_w, b_w)$ of parent
  $Q \neq P$, the overlap factorises into spin factors, so the target parent and strings are drawn
  without evaluating a determinant overlap:

  $$
  q_S(w|x) = \frac{|A^{QP}_{a_w,a_x}B^{QP}_{b_w,b_x}|}{Z_x} = \frac{|S_{wx}|}{Z_x},
  \qquad
  Z_x = \sum_{Q\neq P}\sum_{a_w}|A^{QP}_{a_w,a_x}|\sum_{b_w}|B^{QP}_{b_w,b_x}|,
  $$

  $$
  P_\text{Gen}^S(w|x) = p\,q_S(w|x) + (1 - p)\,q_U(w|x),
  $$

  where $0 \le p < 1$ is `overlap_weight`. The uniform branch keeps support on same-parent pairs and
  on pairs with zero overlap. `optimise_overlap_weight` adapts $p$ between reports by a Newton step
  $p \leftarrow p - \mu_2'(p)/\mu_2''(p)$ on the sampled second moment
  $\mu_2(p) = \sum_w|g_{wx}|^2/P_\text{Gen}^S(w|x)$ of the spawn numerators $g_{wx}$.
- **`pchb`** is $B$-apply's precomputed heat-bath generator within each parent's orthonormal frame
  {cite}`holmes2016,weser2023`. With $O_x$ and $V_x$ the occupied and empty orbitals of the source,
  its weights are

  $$
  w_{ij}^{ab} =
  \begin{cases}
  |\langle ai\Vert jb\rangle|, & \text{same-spin holes},\\
  |(ai|jb)|, & \alpha\beta \text{ holes},
  \end{cases}
  \qquad
  W_{ij} = \sum_{ab}w_{ij}^{ab},
  \qquad
  s_i^a = |h_{ai}| + \sum_j|(ai\Vert jj)| + \sum_{\bar j}|(ai|\bar j\bar j)|,
  $$

  with $s_i^a \ge |H_{a\leftarrow i,x}|$ a precomputed bound on the single-excitation element. A
  hole $i$ is drawn with probability $S_i/Z$, $S_i = \sum_{a\in V_x}s_i^a$, or a hole pair with
  probability $W_{ij}/Z$, normalised by $Z = \sum_{i\in O_x}S_i + \sum_{ij\in O_x}W_{ij}$. The
  particles are then drawn from $V_x$ only, with probabilities $s_i^a/\sum_{c\in V_x}s_i^c$ or
  $w_{ij}^{ab}/\sum_{cd\in V_x}w_{ij}^{cd}$. The singles probability
  $p_\text{single} = \sum_iS_i/Z$ is therefore chosen automatically for every source. From a
  parent's own reference determinant, $s_i^a$ is replaced by the exact $|H_{a\leftarrow i,0}|$, so
  singles that vanish, for example by Brillouin's theorem, are not sampled. PCHB weights need
  $\mathcal O(n_\text{SO}^4)$ memory per parent, which is printed at startup.

Stochastic NOCIQMC currently supports real SCF states only.
