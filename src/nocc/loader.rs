// nocc/loader.rs

use std::sync::OnceLock;

use bincode::Options;

use super::terms::ResidualTermSet;

static R0_TERMS: OnceLock<ResidualTermSet> = OnceLock::new();
static R1_TERMS: OnceLock<ResidualTermSet> = OnceLock::new();

/// Decode one embedded residual term table.
/// # Arguments:
/// - `bytes`: Bincode-encoded residual term table.
/// # Returns:
/// - `ResidualTermSet`: Decoded residual term table.
fn decode_terms(bytes: &[u8]) -> ResidualTermSet {
    bincode::DefaultOptions::new()
        .with_varint_encoding()
        .deserialize(bytes)
        .expect("failed to decode residual term table")
}

/// Return the zeroth-order residual term table.
/// # Arguments:
/// - None.
/// # Returns:
/// - `&'static ResidualTermSet`: Decoded zeroth-order residual term table.
pub(crate) fn r0_terms() -> &'static ResidualTermSet {
    R0_TERMS.get_or_init(|| decode_terms(include_bytes!(concat!(env!("OUT_DIR"), "/r0terms.bin"))))
}

/// Return the first-order residual term table.
/// # Arguments:
/// - None.
/// # Returns:
/// - `&'static ResidualTermSet`: Decoded first-order residual term table.
pub(crate) fn r1_terms() -> &'static ResidualTermSet {
    R1_TERMS.get_or_init(|| decode_terms(include_bytes!(concat!(env!("OUT_DIR"), "/r1terms.bin"))))
}
