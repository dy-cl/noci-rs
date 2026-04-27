// snoci/mod.rs

mod candidate;
mod gmres;
mod operators;
mod step;
mod types;

pub use step::*;
pub use types::*;

pub(in crate::snoci) use gmres::gmres;
pub(in crate::snoci) use operators::*;
pub(in crate::snoci) use candidate::CandidatePool;
