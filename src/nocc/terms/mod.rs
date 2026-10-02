// nocc/terms/mod.rs
//! Dense tensor-contraction evaluation of generated term tables.
//!
//! A generated table gives one element of a residual, energy or coupling block as a sum of
//! terms `c\prod_k F_k` over dummy indices. Evaluating each element separately repeats every
//! dummy loop for every element. Here each term is instead contracted over whole orbital-space
//! blocks, pairwise in the order that keeps each intermediate cheapest, so one pass yields the
//! term's contribution to every element of the block at once.
//!
//! Tables hold millions of small terms, so per-term overhead is kept off the heap: the dense
//! block of every factor is resolved once per table and cached, operand shapes are fixed-size,
//! label sets are bit masks, and intermediate buffers are reused within each worker.
//!
//! The stages are: `loader` decodes the embedded tables, `plan` resolves each table's kept
//! terms, contraction orders and shared products, `factors` builds the dense factor blocks,
//! `term` contracts one term in the reusable `workspace`, `evaluate` sums a table over its
//! terms, and `assemble` gathers tables into quantities over the raw excitation basis.

mod assemble;
mod evaluate;
mod factors;
mod loader;
mod plan;
mod schema;
mod tensors;
mod term;
mod workspace;

// Restricted type re-exports.
pub(crate) use plan::TermEvaluator;
pub(in crate::nocc) use tensors::{Amplitudes, Tensors};

// Restricted function re-exports.
pub(in crate::nocc) use assemble::{assemble_matrix, assemble_scalar, assemble_vector};
pub(in crate::nocc) use loader::{e1_terms, e2_terms, overlap_blocks, residual_classes};
