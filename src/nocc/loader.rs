// nocc/loader.rs

use std::collections::BTreeMap;
use std::sync::OnceLock;

use bincode::Options;

use super::terms::{OverlapTermSet, ResidualClassTerms, ResidualTermSet};

static OVERLAP_TERMS: OnceLock<OverlapTermSet> = OnceLock::new();
static R0_TERMS: OnceLock<ResidualTermSet> = OnceLock::new();
static R1_TERMS: OnceLock<ResidualTermSet> = OnceLock::new();
static R2_TERMS: OnceLock<ResidualTermSet> = OnceLock::new();

/// Decode one embedded residual class term table.
/// # Arguments:
/// - `bytes`: Bincode-encoded residual class term table.
/// # Returns:
/// - `ResidualClassTerms`: Decoded residual class terms.
fn decode_class(bytes: &[u8]) -> ResidualClassTerms {
    bincode::DefaultOptions::new()
        .with_varint_encoding()
        .deserialize(bytes)
        .expect("failed to decode residual class term table")
}

/// Decode one embedded overlap term table.
/// # Arguments:
/// - `bytes`: Bincode-encoded overlap term table.
/// # Returns:
/// - `OverlapTermSet`: Decoded overlap term table.
fn decode_overlap(bytes: &[u8]) -> OverlapTermSet {
    bincode::DefaultOptions::new()
        .with_varint_encoding()
        .deserialize(bytes)
        .expect("failed to decode overlap term table")
}

/// Return the generated space-kind table.
/// # Arguments:
/// - None.
/// # Returns:
/// - `BTreeMap<String, u8>`: Space-kind ids.
fn space_kinds() -> BTreeMap<String, u8> {
    let mut out = BTreeMap::new();

    out.insert("core".to_string(), 0);
    out.insert("active".to_string(), 1);
    out.insert("virtual".to_string(), 2);

    out
}

/// Return the generated tensor-kind table.
/// # Arguments:
/// - None.
/// # Returns:
/// - `BTreeMap<String, u8>`: Tensor-kind ids.
fn tensor_kinds() -> BTreeMap<String, u8> {
    let mut out = BTreeMap::new();

    out.insert("Gamma1".to_string(), 0);
    out.insert("Theta".to_string(), 1);
    out.insert("f".to_string(), 2);
    out.insert("g".to_string(), 3);
    out.insert("Lambda2".to_string(), 4);
    out.insert("Lambda3".to_string(), 5);
    out.insert("Lambda4".to_string(), 6);
    out.insert("t1".to_string(), 8);
    out.insert("t2".to_string(), 9);

    out
}

/// Assemble one residual term table from embedded class files.
/// # Arguments:
/// - `order`: Residual order.
/// - `items`: Class names and bincode class payloads.
/// # Returns:
/// - `ResidualTermSet`: Decoded residual term table.
fn residual_terms(
    order: u8,
    items: &[(&str, &[u8])],
) -> ResidualTermSet {
    ResidualTermSet {
        version: 1,
        order,
        space_kinds: space_kinds(),
        tensor_kinds: tensor_kinds(),
        classes: items
            .iter()
            .map(|&(name, bytes)| (name.to_string(), decode_class(bytes)))
            .collect(),
    }
}

include!(concat!(env!("OUT_DIR"), "/nocc_terms.rs"));
