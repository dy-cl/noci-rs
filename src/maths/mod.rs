// maths/mod.rs
//! Numerical kernels shared by the electronic-structure methods.
//!
//! This module contains reusable tensor contractions, electron-repulsion-integral
//! transformations, dense linear algebra and contraction-determinant operations. These
//! routines provide the numerical building blocks used by SCF, NOCI, the extended
//! nonorthogonal Wick's theorem, deterministic and stochastic propagation, and NOCI-PT2/SNOCI.
//!
//! The `einsum` submodule contains scalar-generic tensor contractions used throughout the
//! matrix-element implementations. The `eri` submodule transforms and contracts two-electron
//! integrals. The `linalg` submodule provides determinant, adjugate, eigensolver and matrix-
//! vector utilities. The `wick` submodule contains the low-level contraction-determinant
//! construction and column-mixing operations used by the nonorthogonal Wick evaluators.
//!
//! Numerical kernels support real or complex scalar types where required by their callers.

pub mod cpu;
#[cfg(feature = "gpu")]
pub(crate) mod gpu;

// Public mixed re-exports.
pub use cpu::*;
