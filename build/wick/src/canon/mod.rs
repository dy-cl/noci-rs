// canon/mod.rs
//! Exact canonical forms of tensor-product terms.
//!
//! A term is a product of tensor factors over free and dummy indices. Two terms are equal when
//! they differ only by a relabelling of dummy indices and by the declared slot symmetries of
//! their factors: antisymmetric index sets (spin-orbital tensors), simultaneous column
//! permutations (spin-free cumulants, integrals and amplitudes) and unordered symmetric sets.
//!
//! Terms are encoded as vertex-coloured graphs whose unordered structure is carried by
//! unordered edges, and the graphs are labelled canonically by individualisation and
//! refinement. The resulting key is exact: two terms share a key if and only if they are equal
//! up to the declared symmetries, with the relative sign reported explicitly.

// Private submodules.
mod form;
mod graph;

// Restricted type re-exports.
pub(crate) use form::{Factor, Form, Key, Sym};

// Restricted function re-exports.
pub(crate) use form::canonical_key;
