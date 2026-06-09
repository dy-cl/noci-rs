// nocc/terms.rs

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Rational coefficient stored as `[numerator, denominator]`.
pub(crate) type Coeff = [i64; 2];

/// Kronecker-delta index pair stored as two class-local index ids.
pub(crate) type Delta = [u16; 2];

/// Compact residual term table generated from spin-free Wick expressions.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ResidualTermSet {
    /// JSON/bincode scheme version.
    pub(crate) version: u32,
    /// Residual order in the cluster amplitudes.
    pub(crate) order: u8,
    /// Orbital-space enum values used by the generated term table.
    #[serde(rename = "spaceKinds")]
    pub(crate) space_kinds: BTreeMap<String, u8>,
    /// Tensor enum values used by the generated term table.
    #[serde(rename = "tensorKinds")]
    pub(crate) tensor_kinds: BTreeMap<String, u8>,
    /// Residual terms grouped by excitation class.
    pub(crate) classes: BTreeMap<String, ResidualClassTerms>,
}

/// Compact residual terms for one excitation class.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ResidualClassTerms {
    /// Class-local symbolic indices stored as `(name, space)`.
    pub(crate) indices: Vec<(String, u8)>,
    /// Class-local ids of the residual free indices.
    pub(crate) free: Vec<u16>,
    /// Symbolic residual terms for this excitation class.
    pub(crate) terms: Vec<ResidualTerm>,
}

/// One symbolic residual term.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ResidualTerm(
    /// Rational prefactor.
    pub(crate) Coeff,
    /// Class-local dummy indices to sum over.
    pub(crate) Vec<u16>,
    /// Kronecker-delta factors.
    pub(crate) Vec<Delta>,
    /// Tensor factors.
    pub(crate) Vec<TensorFactor>,
);

/// One tensor factor in a symbolic residual term.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct TensorFactor(
    /// Tensor kind id.
    pub(crate) u8,
    /// Upper class-local index ids.
    pub(crate) Vec<u16>,
    /// Lower class-local index ids.
    pub(crate) Vec<u16>,
);
