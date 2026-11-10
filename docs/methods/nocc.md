# Nonorthogonal coupled cluster

:::{warning}
Both NOCC formulations are highly experimental and intended for method development rather than
production calculations. Their equations, truncations and interfaces may change substantially.
:::

`noci-rs` develops two coupled-cluster formulations on nonorthogonal references.

## Generalised-normal-ordered NOCC

GNOCC, built with the `gnocc` feature, develops coupled cluster on the correlated NOCI reference
state {cite}`kutzelnigg1997,lee2026`,

$$
|\Phi\rangle = |\Psi_\text{NOCI}\rangle = \sum_x c_x|{}^x\Psi\rangle.
$$

Generalised normal ordering treats $|\Phi\rangle$ as the vacuum and defines contractions through
its reduced density matrices and cumulants. The ansatz and working equations are

$$
|\Psi_\text{NOCC}\rangle = \{e^{\hat T}\}|\Phi\rangle,
\qquad
\hat T = \sum_\mu t_\mu\hat\tau_\mu,
$$

$$
E = \langle\Phi|\hat H\{e^{\hat T}\}|\Phi\rangle,
\qquad
R_\mu = \langle\Phi|\hat\tau_\mu^\dagger\hat H\{e^{\hat T}\}|\Phi\rangle_\text{c} = 0,
$$

with spin-free single and double excitation operators $\hat\tau_\mu$, truncated at second order in
$\hat T$. The reference enters only through its spin-free RDMs and cumulants, evaluated with the
extended nonorthogonal Wick's theorem.

- **Truncation.** `max_cumulant` = $k$ selects GNOCCSD($k$): every energy and residual term that
  contains a cumulant of rank above $k$ is dropped. The metric is always evaluated exactly.
- **Active space.** Natural orbitals within `active_space_tol` of two or zero electrons are
  treated as core or virtual, and all others as active.
- **Redundancies.** Excitations are made non-redundant by canonical orthogonalisation of the
  metric of the first-order interacting space {cite}`lee2026symmetry`, discarding metric
  directions with eigenvalues below `fois_tol`. The default `4e-2` keeps the singular values of
  the orthogonaliser below 5, the pruning of Appendix A of {cite}`lee2026symmetry`: near-null
  directions amplify the error of the cumulant truncation and stall the iteration, while removing
  them leaves the energy unchanged.
- **Solver.** The amplitude equations are solved by inexact Newton iteration. Each macro-iteration
  evaluates the energy and residual, and GMRES solves the Newton equation with the exact Jacobian
  in the orthonormal first-order interacting space, preconditioned by orbital-energy denominators
  with `level_shift` added. Each Newton step is solved to a tenth of the residual norm.
- **Holomorphic amplitudes.** With `holomorphic = true`, the default, the amplitudes are complex
  and the equations are continued analytically, without complex conjugation, from a small
  imaginary starting vector. Where the truncated equations lose their real solution, as two real
  roots merge into a complex-conjugate pair, the iteration follows the solution onto the complex
  plane and reports the real and imaginary parts of the energy. Where a real solution exists, the
  imaginary part decays to zero.

The build script generates the residual, energy, metric and zeroth-order term tables, so
compiling with `gnocc` takes much longer.

## Jeziorski–Monkhorst NOCC

JM-NOCCMC gives every nonorthogonal reference its own cluster operator $\hat T_x$
{cite}`jeziorski1981`, with excitations up to `cluster_rank` in that reference's orbital frame:

$$
|\Psi_k\rangle = \sum_x c_x^{(k)}|\Phi_x\rangle,
\qquad
|\Phi_x\rangle = e^{\hat T_x}|{}^x\Psi\rangle.
$$

The span of the correlated references is invariant under $\hat H$ when
$\hat H|\Phi_y\rangle = \sum_x G_{xy}|\Phi_x\rangle$, with

$$
G = (\mathbf S^e)^{-1}\mathbf H^e,
\qquad
S^e_{zx} = \langle{}^z\Psi|\Phi_x\rangle,
\quad
H^e_{zx} = \langle{}^z\Psi|\hat H|\Phi_x\rangle.
$$

The eigenvalues of $G$ are the state energies. With the oblique projector
$\hat P = \sum_{xz}|\Phi_x\rangle[(\mathbf S^e)^{-1}]_{xz}\langle{}^z\Psi|$, the amplitude residuals
are

$$
R_{x\mu} = \langle{}^x\Psi|\hat\tau_{x\mu}^\dagger e^{-\hat T_x}|W_x\rangle,
\qquad
|W_x\rangle = (\hat I - \hat P)\hat H|\Phi_x\rangle.
$$

### Propagation

The amplitudes $t_{x\mu} = n_{x\mu}/N_x$ are carried by excitor populations $n_{x\mu}$ and
reference populations $N_x$. With one shift $E_s$ and $g_x = \sum_y G_{xy}N_y/N_x$, each step
applies

$$
n_{x\mu} \leftarrow n_{x\mu} - \Delta\tau\,N_x R_{x\mu} - \Delta\tau(g_x - E_s)\,n_{x\mu},
\qquad
N_x \leftarrow N_x - \Delta\tau(g_x - E_s)\,N_x.
$$

The reference populations follow the power iteration of $G$, so they carry the ground-state mixing
coefficients.

- **Stochastic.** A nested `jmnoccmc.qmc` table selects coupled cluster Monte Carlo
  {cite}`thom2010`. The same-reference term is sampled by linked CCMC {cite}`franklin2016`: within
  one reference the determinants are orthonormal, and
  $\bar H_x = e^{-\hat T_x}\hat H e^{\hat T_x}$ terminates after four nested commutators. The
  coupling between references does not terminate, since the exponentials act in different orbital
  frames. It is sampled through projections of clusters of $e^{\hat T_x}$, up to `expansion_rank`,
  onto the other references' determinants. $G$ is estimated from report-block sums, forgotten over
  `memory_effective` blocks, and applied one report block late. Populations live in a sparse store
  on occupied excitors only, and no storage scales with the size of the excitor space.
- **Deterministic.** Without `qmc`, the same update runs as its noise-free counterpart with exact
  $G$, renormalising every step, with an optional dynamic shift as in deterministic NOCIQMC.

### Projection and expansion spaces

Each residual needs projections onto the reference and its excitations up to the cluster rank
only. Each $|\Phi_x\rangle$ is expanded over the determinants of its reference up to
`expansion_rank`, every rank by default. Since $\hat H$ changes the rank by at most two, any
expansion rank of at least `cluster_rank` + 2 keeps every single-reference term exact, but the
overlaps between different references are exact only for the complete expansion.

### Null space and intruders

A null of the Jacobian $J_{y\mu,x\nu} = \partial R_{y\mu}/\partial t_{x\nu}$ comes in three
types:

- **Type I**, internal mixing of the correlated references. These give the $M(M-1)$ exact nulls of
  $M$ references at the complete cluster limit; truncation generically turns them into slow modes
  with a restoring force. At the complete cluster rank nothing restores them, and sampling noise
  random-walks the amplitudes along them, so stochastic propagation is meant for truncated cluster
  ranks.
- **Type II**, an external response, when an external level meets a target root: the intruder
  problem, which the choice of references must avoid.
- **Type III**, a mixture of the two.

`diagnostics = true` diagnoses $J$ at the initial and final amplitudes. It reports the lowest
eigenvalues, where $\operatorname{Re}\lambda \le 0$ marks a growing mode such as an intruder, the
singular values below `null_eps` relative to the largest, and the lowest `null_n` singular values,
each with the external fraction of its vector: 0 for type I and 1 for type II. It also writes
`jacobian_history` to `write.write_dir`, giving $\mathbf t_k - \mathbf t_\star$ along the final
singular vectors. It builds the overlap of every expansion determinant, so use it for small
systems only.

With more references than target states, for example RHF plus the spin-reversed UHF pair in
stretched hydrogen chains, intruders appear.
