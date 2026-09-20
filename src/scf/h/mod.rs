// scf/h/mod.rs
//! Holomorphic Hartree–Fock optimisation and complex-parameter tracking.
//!
//! # References
//!
//! - Holomorphic SCF: Burton and Thom, *J. Chem. Theory Comput.* **12**, 167 (2016),
//!   [doi:10.1021/acs.jctc.5b01005](https://doi.org/10.1021/acs.jctc.5b01005).
//! - Complex adiabatic connection: Burton, Thom, and Loos, *J. Chem. Phys.* **150**, 041103
//!   (2019), [doi:10.1063/1.5085121](https://doi.org/10.1063/1.5085121).

mod canonical;
mod finalise;
mod optimise;
mod seed;
mod step;
mod tangent;
mod track;
mod types;

// Public function re-exports.
pub use canonical::normalise_hermitian;

// Restricted function re-exports.
pub(crate) use track::{continue_hscf_track, initialise_hscf_track, physical_hscf_state};
