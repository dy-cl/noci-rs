// noci/mod.rs
mod cache;
mod naive;
mod overlap;
mod fock;
mod hs;
mod wicks;
mod matrix;
mod types;

pub use types::{MOCache, FockMOCache};
pub use cache::{build_mo_cache, build_fock_mo_cache};
pub use wicks::{build_wicks_shared, update_wicks_fock};
pub use matrix::{build_noci_s, build_noci_hs, calculate_noci_energy};
pub use naive::noci_density;
pub(crate) use matrix::build_noci_fock;
pub(crate) use overlap::calculate_s_pair;
pub(crate) use fock::calculate_f_pair;
pub(crate) use hs::calculate_hs_pair;

pub(crate) use naive::occ_coeffs;
pub use types::NOCIData;
pub(crate) use types::{DetPair, FockData};
