// nonorthogonalwicks/mod.rs
//! Extended nonorthogonal Wick matrix elements between arbitrary excited determinants.
//!
//! For each ordered pair of nonorthogonal reference determinants
//! `\langle{}^x\Psi| and |{}^w\Psi\rangle, this module constructs the fundamental contractions`
//! and operator intermediates required to evaluate overlap, one-body, two-body and
//! transition-density matrix elements between determinants excited from the two references.
//!
//! Singular occupied-orbital overlap matrices are handled by separating the product of
//! non-zero singular values from the number m of zero-overlap occupied-orbital pairs. Each
//! `fundamental contraction carries an assignment m_i \in \{0,1\}, and a matrix element is`
//! obtained from the constrained sum
//!
//! `\sum_{\substack{m_1,\ldots,m_L\\\sum_i m_i = m}}`
//!
//! over the allowed distributions of the zero-overlap pairs. The contraction determinants
//! `contain X^{(m_i)} entries on and below the diagonal and Y^{(m_i)} entries above the`
//! diagonal.
//!
//! The implementation is separated into:
//!
//! `- Construction of X^{(m_i)}, Y^{(m_i)} and the scalar, one-column and two-column`
//!   operator intermediates;
//! - Pair-adaptive layout of the stored matrices and rank-four tensors;
//! - Shared-memory or disk-backed contiguous storage;
//! - Read-only views over pair-specific intermediates;
//! - Preparation of contraction determinants and reusable scratch storage;
//! - Specialised and general evaluators for overlap, one-body, two-body and
//!   transition-density quantities.
//!
//! Once the reference-pair intermediates have been constructed, the subsequent evaluation
//! `cost depends on the excitation ranks and the allowed m_i distributions rather than`
//! directly on the number of electrons or basis functions.

pub(crate) mod cpu;
#[cfg(feature = "gpu")]
pub(crate) mod gpu;

mod requirements;
mod types;

// Public type re-exports.
pub use cpu::{WicksShared, WicksView};

// Crate-visible type re-exports.
pub(crate) use cpu::{
    DiffSpinBuild, SameSpinBuild, WickScratchSpin, WicksDiskMeta, WicksPairView, WicksRma,
};
pub(crate) use requirements::WicksRequirements;
pub(crate) use types::{DiffSpinMeta, PairMeta, PairZeroCounts, SameSpinMeta};

// Crate-visible function re-exports.
pub(crate) use cpu::{
    assign_offsets, create_wicks_mmap, load_wicks_mmap, prepare_same, write_diff_spin,
    write_same_spin, write2t, xw_f, xw_h1, xw_h2_diff, xw_h2_same, xw_overlap, xw_overlap_same_f64,
};
#[cfg(feature = "nocc")]
pub(crate) use cpu::{xw_rdm_same_element, xw_rdm1, xw_rdm2_diff, xw_rdm2_same};
