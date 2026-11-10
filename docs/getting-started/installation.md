# Installation

## Requirements

- Rust 1.90.0 with Cargo. The repository's `rust-toolchain.toml` selects this toolchain
  automatically when you use `rustup`.
- HDF5 development libraries compatible with the `hdf5` Rust crate.
- OpenBLAS and LAPACK.
- An MPI compiler and runtime.
- MolSSI Basis Set Exchange data {cite}`pritchard2019` for the named basis sets used by
  libcint {cite}`sun2015`.

## Basis-set data

`noci-rs` reads named basis sets from a checkout of the Basis Set Exchange. The tested data
version is `4adaf1372c7101620ca1a9f3130be9ae97fb8f30`. Point `BSE_DATA_DIR` at its `data`
directory:

```bash
git clone https://github.com/MolSSI-BSE/basis_set_exchange.git basis_set_exchange
git -C basis_set_exchange checkout 4adaf1372c7101620ca1a9f3130be9ae97fb8f30
export BSE_DATA_DIR="$PWD/basis_set_exchange/basis_set_exchange/data"
```

:::{warning}
Without `BSE_DATA_DIR` the program aborts when it parses the basis set. Export it in every shell
or job script that runs `noci-rs`.
:::

## Build

```bash
git clone https://github.com/dy-cl/noci-rs
cd noci-rs
cargo build --release
```

The executable is `target/release/noci-rs`.

## Build options

### Fixed-rank Wick kernels

`MAXEXCIT` sets the largest total excitation rank compiled into the scalar and SIMD fixed-rank
Wick kernels. It defaults to four:

```bash
MAXEXCIT=3 cargo build --release
```

For a two-spin determinant pair, a specialised kernel is used when both
$R_{X\alpha} + R_{X\beta} \le$ `MAXEXCIT` and $R_{W\alpha} + R_{W\beta} \le$ `MAXEXCIT`.
Higher excitation ranks are still supported through generic fallbacks: `MAXEXCIT` controls
cache width and monomorphisation, not the largest excitation rank that can be evaluated. AVX2 and
AVX-512 pack four and eight determinant pairs per lane respectively; neither limits the excitation
rank.

### Timing counters

```bash
cargo build --release --features timings
```

This adds routine-level call counts and timing summaries to the output.

:::{warning}
The instrumentation slows the whole program down massively, and its timings misrepresent where
the time goes in a normal build. Use it only to see how calls are distributed across routines. To
find hot spots, profile an ordinary release build with a sampling profiler such as `perf`.
:::

### GNOCC

[GNOCC](../methods/nocc.md#generalised-normal-ordered-nocc) is behind the `gnocc` feature:

```bash
cargo build --release --features gnocc
```

Its coupled-cluster equations are not written by hand. The build script expands the energy and
residual equations, together with the metric and zeroth-order operator of the first-order
interacting space, by Wick's theorem in generalised normal order. It spin adapts the result and
stores it as *term tables*, each term a contraction of amplitudes, integrals and density cumulants,
cached under `build/generated/`. Set `WICKFORCEREGENERATE=1` to regenerate them.
Compiling with this feature takes much longer, so leave it off unless you need GNOCC. JM-NOCCMC
does not need it.
