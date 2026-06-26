// lib.rs

mod canonical;
mod cluster;
mod field;
mod gno;
mod gram;
mod hamiltonian;
mod int;
mod progress;
mod schema;
mod specs;

pub mod encode;
pub mod ir;
pub mod overlap;
pub mod residual;
pub mod target;
pub mod timers;
pub mod wick;

pub use schema::{
    GeneratedTerm, OverlapBlockTerms, OverlapTermSet, ResidualClassTerms, ResidualTermSet,
    TensorFactor,
};
