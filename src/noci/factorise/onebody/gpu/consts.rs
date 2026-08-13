// noci/factorise/onebody/gpu/consts.rs
//! Constants for factorised one-body GPU contractions.

/// Maximum bytes allocated to one transient same-spin `S/F` factor-panel pair.
pub(super) const FACTOR_PANEL_BYTES: usize = 1536 * 1024 * 1024;

/// Maximum bytes allocated to one transient first-stage `S/F` intermediate pair.
pub(super) const FIRST_PANEL_BYTES: usize = 3072 * 1024 * 1024;
