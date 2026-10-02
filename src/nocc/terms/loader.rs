// nocc/terms/loader.rs
//! Decoding of the embedded term tables, once per run.

// Standard library imports.
use std::collections::BTreeMap;
use std::sync::OnceLock;

// External crate imports.
use bincode::Options;

// Crate-root imports.
use crate::nocc::space::ExcitationClass;

// Parent/sibling imports.
use super::schema::{OverlapBlockTerms, OverlapTermSet, ResidualClassTerms, ResidualTermSet};
use super::tensors::{SpaceKind, TensorKind};

/// Residual terms of one order keyed by excitation class.
pub(super) type ClassTables = BTreeMap<ExcitationClass, &'static ResidualClassTerms>;

/// Class-pair blocks of one overlap-type table, as left class, right class and block terms, in
/// the order of the generated table.
pub(super) type BlockTables = Vec<(ExcitationClass, ExcitationClass, &'static OverlapBlockTerms)>;

static E1_TERMS: OnceLock<ResidualClassTerms> = OnceLock::new();
static E2_TERMS: OnceLock<ResidualClassTerms> = OnceLock::new();
static OVERLAP_TERMS: OnceLock<OverlapTermSet> = OnceLock::new();
static R0_TERMS: OnceLock<ResidualTermSet> = OnceLock::new();
static R1_TERMS: OnceLock<ResidualTermSet> = OnceLock::new();
static R2_TERMS: OnceLock<ResidualTermSet> = OnceLock::new();
static RESIDUAL_CLASSES: OnceLock<[ClassTables; 3]> = OnceLock::new();
static OVERLAP_BLOCKS: OnceLock<BlockTables> = OnceLock::new();

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
    SpaceKind::ALL
        .into_iter()
        .map(|kind| (kind.name().to_string(), kind as u8))
        .collect()
}

/// Return the generated tensor-kind table.
/// # Arguments:
/// - None.
/// # Returns:
/// - `BTreeMap<String, u8>`: Tensor-kind ids.
fn tensor_kinds() -> BTreeMap<String, u8> {
    TensorKind::ALL
        .into_iter()
        .map(|kind| (kind.name().to_string(), kind as u8))
        .collect()
}

/// Return the residual terms of one order keyed by excitation class, converting the generated
/// class names once.
/// # Arguments:
/// - `order`: Residual order, `0`, `1` or `2`.
/// # Returns:
/// - `&'static ClassTables`: Residual class terms of that order.
/// # Panics
/// - Panics if `order` exceeds two or a generated class name is unknown.
pub(in crate::nocc) fn residual_classes(order: usize) -> &'static ClassTables {
    let tables = RESIDUAL_CLASSES.get_or_init(|| {
        [r0_terms(), r1_terms(), r2_terms()].map(|set| {
            set.classes
                .iter()
                .map(|(name, terms)| (ExcitationClass::from_name(name), terms))
                .collect()
        })
    });
    &tables[order]
}

/// Return the class-pair blocks of the FOIS metric with their excitation classes.
/// # Arguments:
/// - None.
/// # Returns:
/// - `&'static BlockTables`: Metric blocks in generated order.
/// # Panics
/// - Panics if a generated class name is unknown.
pub(in crate::nocc) fn overlap_blocks() -> &'static BlockTables {
    OVERLAP_BLOCKS.get_or_init(|| class_blocks(overlap_terms()))
}

/// Convert the class names of every block of an overlap-type table.
/// # Arguments:
/// - `set`: Decoded overlap-type table.
/// # Returns:
/// - `BlockTables`: Blocks with their left and right excitation classes.
/// # Panics
/// - Panics if a generated class name is unknown.
fn class_blocks(set: &'static OverlapTermSet) -> BlockTables {
    set.blocks
        .values()
        .map(|b| {
            let left = ExcitationClass::from_name(&b.left);
            let right = ExcitationClass::from_name(&b.right);
            (left, right, b)
        })
        .collect()
}

/// Assemble one residual term table from embedded class files.
/// # Arguments:
/// - `order`: Residual order.
/// - `items`: Class names and bincode class payloads.
/// # Returns:
/// - `ResidualTermSet`: Decoded residual term table.
fn assemble_residual_terms(
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
