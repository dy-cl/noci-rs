# Code architecture

The diagrams below map the main code layers and execution paths. Module names refer to
`src/`; arrows in the first diagram mean **uses**, while the remaining diagrams show
**execution flow**. The [method pages](../methods/index.md) give the scientific theory,
and [STYLE.md][style] defines the code conventions.

## Method families and shared layers

```{figure} diagrams/layers.svg
:alt: NOCI and NOCC approaches use their family shared layers, elements and qmc; the common layers use determinant spaces and numerical foundations.
:width: 100%

Principal dependencies. The asterisk marks machinery enabled by the `gnocc` feature.
[Graphviz source](diagrams/layers.dot).
```

Each approach depends on its family's shared layer and the common layers below it.
Method folders do not depend on sibling approaches. `qmc/` never depends on a method
family. NOCC can also use the NOCI shared layer; this overview omits individual helper
and utility dependencies.

The [crate root][lib-src] and module `mod.rs` files expose the interfaces. The
[driver][driver-src] owns orchestration; scientific operations belong to their numerical
modules. The [NOCI shared layer][noci-src] supplies matrices, reference solutions and
factorised operators. The [NOCC shared layer][nocc-src] supplies JM equations and the
feature-gated GNOCC machinery.

## From input to results

```{figure} diagrams/calculation.svg
:alt: Lua input passes through AO integrals, SCF, reference selection, cache preparation and a reference NOCI solve before the requested propagation, selection or coupled-cluster method runs and writes results.
:width: 85%

The driver loads input once and repeats the geometry stages for a scan. Post-reference
branches run only when requested; GNOCC requires `gnocc`.
[Graphviz source](diagrams/calculation.dot).
```

Start at `driver/run.rs` for the scan, `driver/geometry.rs` for one geometry,
`driver/reference.rs` for reference preparation, and `driver/post.rs` for method selection.
`AoData` carries the molecular AO data. `PostSCFData` borrows the retained determinant
space, AO data, MO caches and tolerance for post-reference calculations.

### Determinant data passed between layers

The [determinant layer][det-src] separates a state's identity from its position in a space
and its excitation data. Occupations are interpreted within a parent orbital frame.

| Role | Determinant | Spin string |
| --- | --- | --- |
| Position in a space | `NOCIIndex` | `NOCISpinStringIndex` |
| State in a parent frame | `DeterminantState` | `SpinStringState` |
| Excitation, phase and cached labels relative to a reference | `DeterminantData` | `SpinStringData` |

`NOCISpace` owns retained topology and interns each parent's spin strings. Auxiliary spaces
have their own handles and share parent definitions. `ConnectionState` supplies excitation
data measured from its source for connections within one frame.

## Matrix-element dispatch

```{figure} diagrams/dispatch.svg
:alt: Pair dispatch first checks a shared parent, connecting rank and orthonormal cache, then selects zero, orthogonal Slater–Condon, extended Wick or generalised Slater–Condon evaluation.
:width: 90%

The routing in `elements/pairs.rs` is shared by dense and on-demand matrix-element requests.
[Graphviz source](diagrams/dispatch.dot).
```

Methods submit `DetPair` requests to [elements/pairs.rs][pairs-src]. Definitions in
[elements/operator.rs][operator-src] implement `PairOperator` for overlap, shifted Fock
and Hamiltonian/overlap evaluation. Requests are grouped by parent or ordered reference
pair before entering an engine. Density matrices are assembled in `elements/rdm/`;
higher-rank pair RDM evaluation for GNOCC is feature-gated.

The [Wick engine][wicks-src] prepares reference-pair contractions in `build.rs`, arranges
contiguous storage through `layout.rs`, `storage.rs` and `shared.rs`, and exposes borrowed
views in `view.rs`. The operator files in `eval/` gather contraction determinants and
reuse cofactor/minor buffers from `scratch.rs`. This separates reference-pair preparation
from repeated excited-pair evaluation.

[elements/batch.rs][batch-src] owns `LaneBatch`, CPU-width selection, packet padding and
output writes. Wick fixed-rank formulas use `V: Simd<N>` for one, four or eight lanes;
generic paths handle zero-overlap assignment sums and ranks outside the generated region.
Same-spin factor rows also supply the spin-factorised operators in `noci/factorise/`.
See [Matrix elements](../methods/elements.md) for the governing equations.

## Shared stochastic execution

```{figure} diagrams/stochastic.svg
:alt: NOCIQMC and JM-NOCCMC implement Propagator for the same qmc report loop, which samples cycles through persistent workers, applies one step, handles reports and restarts, then repeats or finishes.
:width: 100%

Both method families run under `qmc/run.rs`. Dashed arrows show work delegated during a
sampling cycle. [Graphviz source](diagrams/stochastic.dot).
```

The `Propagator` interface supplies `cycle`, `step`, `write_restart`, `print_row` and
`finish`. Within a report, populations stay frozen for sampling; `step` applies the
accumulated change once. The shared loop checks stop requests after that step, writes
scheduled restarts and prints report rows. A stop writes a restart and exits before the row.

Persistent Rayon workers hold random streams, matrix-element queues and reusable scratch.
They resolve requests through `elements/`. [qmc/][qmc-src] also owns MPI key ownership,
update exchange and annihilation, FRI compression, restart serialization and the common
shift update. Population representations and method-specific controllers stay in their
own stochastic folders. Rayon distributes work within a rank; MPI communicates between
ranks, and Wick storage can be shared within a node.

## Editing and validation

The SVG figures render directly in the documentation. Their editable `.dot` sources live
beside them; regenerate a figure with Graphviz, for example:

```bash
dot -Tsvg docs/development/diagrams/layers.dot \
  -o docs/development/diagrams/layers.svg
```

[build/build.rs][build-src] generates the Wick dispatch tables controlled by `MAXEXCIT`
and, with `gnocc`, prepares generated GNOCC term data. Edit generators and their source
data when changing generated expressions.

Integration tests and numerical fixtures live in [tests/][tests-src]. In particular,
`tests/wicks.rs` compares Wick and generalised Slater–Condon elements, including singular
overlaps. The [contribution guide][contributing] lists the formatting, lint, numerical-test
and documentation-build checks.

[style]: https://github.com/dy-cl/noci-rs/blob/main/STYLE.md
[contributing]: https://github.com/dy-cl/noci-rs/blob/main/CONTRIBUTING.md
[lib-src]: https://github.com/dy-cl/noci-rs/blob/main/src/lib.rs
[driver-src]: https://github.com/dy-cl/noci-rs/tree/main/src/driver
[det-src]: https://github.com/dy-cl/noci-rs/tree/main/src/determinant
[pairs-src]: https://github.com/dy-cl/noci-rs/blob/main/src/elements/pairs.rs
[operator-src]: https://github.com/dy-cl/noci-rs/blob/main/src/elements/operator.rs
[wicks-src]: https://github.com/dy-cl/noci-rs/tree/main/src/elements/nonorthogonalwicks
[batch-src]: https://github.com/dy-cl/noci-rs/blob/main/src/elements/batch.rs
[noci-src]: https://github.com/dy-cl/noci-rs/tree/main/src/noci
[nocc-src]: https://github.com/dy-cl/noci-rs/tree/main/src/nocc
[qmc-src]: https://github.com/dy-cl/noci-rs/tree/main/src/qmc
[build-src]: https://github.com/dy-cl/noci-rs/blob/main/build/build.rs
[tests-src]: https://github.com/dy-cl/noci-rs/tree/main/tests
