// noci/mod.rs
mod cache;
mod fock;
mod hs;
mod m;
mod matrix;
mod naive;
mod overlap;
mod overlaps;
mod types;
mod wicks;

pub use cache::{build_fock_mo_cache, build_mo_cache};
pub(crate) use fock::calculate_f_pair;
pub(crate) use hs::calculate_hs_pair;
pub(crate) use m::calculate_m_pair;
pub(crate) use matrix::build_noci_fock;
pub use matrix::{build_noci_hs, build_noci_s, calculate_noci_energy};
pub use naive::noci_density;
pub(crate) use naive::occ_coeffs;
#[cfg(feature = "nocc")]
pub(crate) use naive::{build_s_pair, pair_density};
pub(crate) use overlap::calculate_s_pair;
pub(crate) use overlaps::{OverlapFactor, OverlapFactorScratch};
pub(crate) use types::{DetPair, FockData};
pub use types::{FockMOCache, MOCache};
pub use types::{NOCIData, NOCIScalar};
pub use wicks::{build_wicks_shared, update_wicks_fock};
