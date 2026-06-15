// lib.rs 

mod canonical;
mod encode;
mod ir;
pub mod overlap;
mod schema;
mod specs;
mod spinsum;
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
