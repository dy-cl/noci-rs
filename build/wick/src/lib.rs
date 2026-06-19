// lib.rs

mod canonical;
mod cluster;
mod gno;
mod hamiltonian;
mod progress;
mod schema;
mod specs;
mod spinsum;

pub mod encode;
pub mod ir;
pub mod overlap;
pub mod residual;
pub mod target;
pub mod wick;

pub use schema::{
    GeneratedTerm, OverlapBlockTerms, OverlapTermSet, ResidualClassTerms, ResidualTermSet,
    TensorFactor,
};
