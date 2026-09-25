// lib.rs
//! Build-time generator of the spin-free GNOCC residual and FOIS metric term tables.
//!
//! The equations are derived in four stages: generalised-normal-ordered Wick contraction in
//! the antisymmetrised spin-orbital basis (`so`), comparable term by term with Wick&D; spin
//! adaptation to an `SU(2)`-invariant spin ensemble (`spin`); global reduction modulo exact
//! spin relations (`reduce`); and encoding for the runtime (`emit`).

// Public submodules.
pub mod emit;
pub mod so;
pub mod target;

// Private submodules.
mod canon;
mod reduce;
mod schema;
mod specs;
mod spin;

// Public type re-exports.
pub use schema::{
    GeneratedTerm, OverlapBlockTerms, OverlapTermSet, ResidualClassTerms, TensorFactor,
};
