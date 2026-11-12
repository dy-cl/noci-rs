# Code architecture

`noci-rs` is organised around method families and shared numerical layers. The driver
connects them into a calculation; determinant spaces define the states; matrix-element
engines evaluate operators between those states; NOCI and NOCC methods use those elements
to solve or propagate a wavefunction.

This page is a map for reading and extending the implementation. The
[method pages](../methods/index.md) describe the theory, while
[STYLE.md][style] defines the conventions for organising and documenting code.

## Source layout and dependencies

```text
src/
  main.rs                 Command-line entry point
  driver/                 Input, geometry scans and calculation orchestration
  input/                  Lua input tables and their validation
  integrals/              Atomic-orbital integrals
  scf/                    Real and holomorphic SCF calculations
  basis/                  Reference selection and excitation construction
  determinant/            Parent orbital frames, state identities and spaces
  elements/               Pair operators, engine dispatch and density matrices
    orthogonal/           Slater–Condon evaluation within one orthonormal frame
    naive/                Generalised Slater–Condon evaluation
    nonorthogonalwicks/   Extended nonorthogonal Wick evaluation
  qmc/                    Shared stochastic machinery
  noci/                   NOCI matrices, reference solutions and factorised operators
    deterministic/        Exact imaginary-time propagation
    stochastic/           NOCIQMC propagators
    selected/             Selected NOCI and NOCI-PT2
    optimised/            ResHF orbital optimisation
  nocc/                   Shared nonorthogonal coupled-cluster machinery
    deterministic/        Deterministic JM-NOCC and experimental GNOCC solvers
    stochastic/           JM-NOCCMC
  maths/                  Linear algebra and scalar/SIMD numerical operations
```

The numerical method folders depend on their family's shared layer and the common layers
below it. For example, `noci/stochastic/` uses factorised operators from `noci/`, matrix
elements from `elements/`, determinant identities from `determinant/`, and sampling and
communication from `qmc/`. It does not call `noci/deterministic/` to implement its updates.
The same boundary applies between method families: shared stochastic infrastructure belongs
in `qmc/`, which never depends on `noci/` or `nocc/`.

Keep orchestration in `driver/` and scientific calculations in the modules that own them.
The [crate root][lib-src] and each module's `mod.rs` describe their interfaces and re-export
the types and functions used by their callers.

## From input to a calculation

The executable in [main.rs][main-src] calls `driver::load_config` and `driver::run`.
The Lua configuration is loaded once. For each geometry, the [driver][driver-src]
performs the following stages:

1. Prepare the molecular AO data and obtain the requested real
   or holomorphic SCF states. `AoData` carries AO integrals, electron counts and nuclear
   repulsion; the scalar traits in `scalar.rs` support the real and complex numerical paths.
2. Select the parent references and construct a `NOCISpace`. Its ordering supplies the
   determinant and spin-string indices used throughout the calculation.
3. Build MO integral caches and, when enabled, the ordered reference-pair Wick intermediates.
   Construct the reference NOCI matrices and solve the generalised eigenvalue problem.
4. Run the requested post-reference methods. `PostSCFData` borrows the AO data, retained
   space, MO caches and tolerance; `NOCIData` supplies the input and matrix-element data
   used by pair evaluation. Method-specific populations and workspaces remain with the method.
5. Report the results and write the requested outputs. Geometry scans carry selected
   converged states and supported post-reference guesses to the next geometry.

The useful entry points are `driver/run.rs` for the scan, `driver/geometry.rs` for one
geometry, `driver/reference.rs` for reference preparation, and `driver/post.rs` for the
post-reference method selection.

## Determinants: identity, position and excitation data

A determinant's orbital frame is part of its identity. Identical occupation masks in two
different parent frames represent different states. The [determinant layer][det-src]
separates three roles:

| Role | Determinant type | Spin-string type | Meaning |
| --- | --- | --- | --- |
| Handle | `NOCIIndex` | `NOCISpinStringIndex` | A position in a particular space |
| Identity | `DeterminantState` | `SpinStringState` | Parent frame and occupations, independent of a retained-space position |
| Payload | `DeterminantData` | `SpinStringData` | Holes, particles, fermionic phase and cached labels measured from a reference in that frame |

`NOCISpace` owns the retained topology and interns each parent's alpha and beta spin strings.
Subspaces can therefore be represented by collections of `NOCIIndex` handles. Auxiliary
spaces use their own handles while sharing the parent-frame definitions.

A `ConnectionState` describes a connection within one frame. It can produce excitation data
measured from its source determinant, which lets the orthogonal matrix-element kernels use
the same orbital-label and phase conventions as reference-relative evaluation.

## One matrix-element interface, three engines

Methods submit `DetPair` requests through [elements/pairs.rs][pairs-src]. Operator definitions
in [elements/operator.rs][operator-src] implement `PairOperator` for overlap, shifted Fock
and Hamiltonian/overlap evaluation. The dispatcher groups requests and selects the engine
for each pair:

| Pair | Evaluation |
| --- | --- |
| Same parent, beyond the operator's connecting excitation rank | Zero |
| Same parent, with the required Hermitian orthonormal MO cache | Orthogonal Slater–Condon kernels |
| Remaining pairs, with Wick evaluation enabled and intermediates available | Extended nonorthogonal Wick kernels |
| Remaining pairs without Wick intermediates | Generalised Slater–Condon rules |

Dense matrix construction and stochastic on-demand requests use this common evaluation
layer. Reduced density matrices are assembled in `elements/rdm/`; the higher-rank pair RDM
evaluators used by GNOCC are enabled by the `gnocc` feature. See
[Matrix elements](../methods/elements.md) for the equations and numerical conventions.

### Preparing and evaluating Wick intermediates

The [Wick engine][wicks-src] separates work associated with a reference pair from work
associated with a requested excited determinant pair:

| Stage | Files | Responsibility |
| --- | --- | --- |
| Construction | `build.rs` | Fundamental contractions and scalar, one-column and two-column operator intermediates |
| Layout and storage | `layout.rs`, `storage.rs`, `shared.rs` | Pair-specific offsets, contiguous storage, MPI sharing and disk caching |
| Read access | `view.rs` | Borrowed views of one ordered reference pair and its spin sectors |
| Working storage | `scratch.rs`, `eval/prepare.rs` | Orbital labels, contraction determinants and reusable cofactor/minor buffers |
| Evaluation | `eval/overlap.rs`, `onebody.rs`, `hamiltonian.rs`, `rdm.rs` | Operator-specific determinant and replacement formulas |

Reference-pair views hold the reduced overlap and zero-overlap count as well as the
intermediates. When occupied-orbital pairs have zero overlap, evaluation requires constrained
sums over contraction assignments. Generic evaluators handle those sums and ranks outside the generated fixed-rank
region. Equations beside the calculation blocks document the assignment ordering, phases
and prefactors.

### Shared batching and SIMD

[elements/batch.rs][batch-src] owns `LaneBatch`, CPU-width selection, packet padding and
writing the genuine output lanes. Operator families supply their scalar and packed kernels.
Groups share the ranks or orbital labels needed by one kernel; their results are scattered
back into request order after evaluation.

The Wick fixed-rank formulas are written over `V: Simd<N>`. Scalar evaluation uses the same
formula with `T::Packed1`; supported CPUs also use four- or eight-lane packets. Same-spin
factor rows in `nonorthogonalwicks/eval/batch.rs` share the overlap and one-body evaluations
used by pair requests and by the spin-factorised operators in `noci/factorise/`.

## Method families and stochastic execution

The [NOCI shared layer][noci-src] constructs full matrices, solves the reference NOCI problem,
and supplies spin-factorised overlap and one-body applications. Its method folders own
their solvers, propagators and selection algorithms. The [NOCC family][nocc-src] contains
the shared JM equations in `jm.rs`, deterministic and stochastic JM implementations, and
the feature-gated GNOCC machinery for spaces, cumulants, generated terms and equations.

Both stochastic families implement the `Propagator` interface in [qmc/run.rs][run-src].
Within one report, the shared loop calls `cycle` repeatedly with frozen populations, then
calls `step` once to apply the accumulated change. It also orders report output, stop
requests, restart intervals and finalisation. The method supplies the actual population
update and the contents of its report and restart state.

Persistent per-thread workers hold random streams, matrix-element queues and scratch
buffers. Workers batch their queued requests through `elements/`; they do not implement
their own matrix-element formulas. MPI key ownership, exchange and annihilation of updates,
FRI compression, restart serialization and the common shift update live in [qmc/][qmc-src].
The NOCI and JM methods keep their different population representations and method-specific
controllers in their own stochastic folders.

Rayon distributes work within a rank. MPI distributes work and combines updates between
ranks; Wick storage can be shared within a node. Read-only intermediates and reusable worker
scratch keep repeated evaluation separate from reference-pair preparation.

## Generated code, tests and extending the implementation

[build/build.rs][build-src] generates the fixed-rank Wick dispatch tables controlled by
`MAXEXCIT`. With `gnocc`, the build also prepares the generated GNOCC term data. Changes to
equation generation belong in the generators and their source data; generated build outputs
are consumed by the Rust implementation.

Integration tests live in [tests/][tests-src], with inputs and expected results under
`tests/fixtures/`. `tests/wicks.rs` compares Wick matrix elements with generalised
Slater–Condon evaluation, including singular-overlap cases. Propagator and restart tests
exercise the method families and their shared stochastic execution.

When extending the code, start with the layer that owns the operation: determinant identity
and topology in `determinant/`, operator evaluation in `elements/`, reusable stochastic
operations in `qmc/`, and method-specific equations and state in the appropriate family.
Follow the dependency boundaries above and keep the governing equations beside scientific
calculations, as described in [STYLE.md][style]. The [contribution guide][contributing]
lists the required formatting, lint, numerical-test and documentation-build commands.

[style]: https://github.com/dy-cl/noci-rs/blob/main/STYLE.md
[contributing]: https://github.com/dy-cl/noci-rs/blob/main/CONTRIBUTING.md
[lib-src]: https://github.com/dy-cl/noci-rs/blob/main/src/lib.rs
[main-src]: https://github.com/dy-cl/noci-rs/blob/main/src/main.rs
[driver-src]: https://github.com/dy-cl/noci-rs/tree/main/src/driver
[det-src]: https://github.com/dy-cl/noci-rs/tree/main/src/determinant
[pairs-src]: https://github.com/dy-cl/noci-rs/blob/main/src/elements/pairs.rs
[operator-src]: https://github.com/dy-cl/noci-rs/blob/main/src/elements/operator.rs
[wicks-src]: https://github.com/dy-cl/noci-rs/tree/main/src/elements/nonorthogonalwicks
[batch-src]: https://github.com/dy-cl/noci-rs/blob/main/src/elements/batch.rs
[noci-src]: https://github.com/dy-cl/noci-rs/tree/main/src/noci
[nocc-src]: https://github.com/dy-cl/noci-rs/tree/main/src/nocc
[run-src]: https://github.com/dy-cl/noci-rs/blob/main/src/qmc/run.rs
[qmc-src]: https://github.com/dy-cl/noci-rs/tree/main/src/qmc
[build-src]: https://github.com/dy-cl/noci-rs/blob/main/build/build.rs
[tests-src]: https://github.com/dy-cl/noci-rs/tree/main/tests
