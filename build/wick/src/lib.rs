// lib.rs 

mod canonical;
mod cluster;
mod encode;
mod gno;
mod hamiltonian;
mod ir;
mod schema;
mod specs;
mod spinsum;
pub mod overlap;
pub mod residual;
pub mod target;
pub mod wick;

pub use schema::OverlapTermSet;

/// Generate all metric terms.
/// # Arguments:
/// - None.
/// # Returns:
/// - `OverlapTermSet`: Compact metric term table.
pub fn overlap_terms() -> OverlapTermSet {
    encode::overlap_terms()
}
