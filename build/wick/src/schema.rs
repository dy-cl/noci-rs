// schema.rs

use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};

/// Rational coefficient.
pub type Coeff = [i64; 2];

/// Kronecker delta factor.
pub type Delta = [u16; 2];

/// Compact generated metric term table.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OverlapTermSet {
    /// Schema version.
    pub version: u32,
    /// Orbital-space ids.
    #[serde(rename = "spaceKinds")]
    pub space_kinds: BTreeMap<String, u8>,
    /// Tensor-kind ids.
    #[serde(rename = "tensorKinds")]
    pub tensor_kinds: BTreeMap<String, u8>,
    /// Metric blocks.
    pub blocks: BTreeMap<String, OverlapBlockTerms>,
}

/// Compact terms for one metric block.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OverlapBlockTerms {
    /// Left excitation class.
    pub left: String,
    /// Right excitation class.
    pub right: String,
    /// Block-local indices as `(name, space)`.
    pub indices: Vec<(String, u8)>,
    /// Left free-index ids.
    #[serde(rename = "leftFree")]
    pub left_free: Vec<u16>,
    /// Right free-index ids.
    #[serde(rename = "rightFree")]
    pub right_free: Vec<u16>,
    /// Generated terms.
    pub terms: Vec<GeneratedTerm>,
}

/// One generated term.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GeneratedTerm(
    /// Rational coefficient.
    pub Coeff,
    /// Dummy-loop index ids.
    pub Vec<u16>,
    /// Delta factors.
    pub Vec<Delta>,
    /// Tensor factors.
    pub Vec<TensorFactor>,
);

/// One tensor factor.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TensorFactor(
    /// Tensor kind id.
    pub u8,
    /// Upper index ids.
    pub Vec<u16>,
    /// Lower index ids.
    pub Vec<u16>,
);
