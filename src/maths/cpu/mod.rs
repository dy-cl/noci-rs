// maths/cpu/mod.rs
//! CPU numerical kernels shared by the electronic-structure methods.

pub mod einsum;
pub mod eri;
pub mod linalg;
pub mod wick;

// Public mixed re-exports.
pub use einsum::*;
pub use eri::*;
pub use linalg::*;
pub use wick::*;
