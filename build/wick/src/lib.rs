// lib.rs

mod canonical;
mod cluster;
mod encode;
mod gno;
mod hamiltonian;
mod progress;
mod schema;
mod specs;
mod spinsum;

pub mod ir;
pub mod overlap;
pub mod residual;
pub mod target;
pub mod wick;

pub use schema::{GeneratedTerm, OverlapBlockTerms, OverlapTermSet, ResidualClassTerms, ResidualTermSet, TensorFactor};

/// Generate all metric terms.
/// # Arguments:
/// - None.
/// # Returns:
/// - `OverlapTermSet`: Compact metric term table.
pub fn overlap_terms() -> OverlapTermSet {
    encode::overlap_terms()
}

/// Generate all zeroth-order residual terms.
/// # Arguments:
/// - None.
/// # Returns:
/// - `ResidualTermSet`: Compact zeroth-order residual term table.
pub fn r0_terms() -> ResidualTermSet {
    encode::residual_terms(0)
}

/// Generate all first-order residual terms.
/// # Arguments:
/// - None.
/// # Returns:
/// - `ResidualTermSet`: Compact first-order residual term table.
pub fn r1_terms() -> ResidualTermSet {
    encode::residual_terms(1)
}

/// Generate all second-order residual terms.
/// # Arguments:
/// - None.
/// # Returns:
/// - `ResidualTermSet`: Compact second-order residual term table.
pub fn r2_terms() -> ResidualTermSet {
    encode::residual_terms(2)
}
