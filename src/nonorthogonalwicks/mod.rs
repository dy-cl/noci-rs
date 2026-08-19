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

mod build;
mod eval;
mod layout;
mod scratch;
mod storage;
mod types;
mod view;

// Public type re-exports.
pub use storage::WicksShared;
pub use view::WicksView;

// Crate-visible type re-exports.
pub(crate) use build::{DiffSpinBuild, SameSpinBuild};
pub(crate) use scratch::WickScratchSpin;
pub(crate) use storage::{WicksDiskMeta, WicksRma};
pub(crate) use types::{DiffSpinMeta, PairMeta, PairZeroCounts, SameSpinMeta};
pub(crate) use view::WicksPairView;

// Crate-visible function re-exports.
pub(crate) use eval::{
    prepare_same, xw_f, xw_f_overlap_prepared, xw_hamiltonian_overlap_prepared,
    xw_hamiltonian_overlap_prepared_batched, xw_overlap, xw_overlap_same_f64,
    xw_overlap_same_f64_batched,
};
#[cfg(target_arch = "x86_64")]
pub(crate) use eval::{xw_f_overlap_m0_prepared_f64x4, xw_f_overlap_m0_prepared_f64x8};
#[cfg(feature = "nocc")]
pub(crate) use eval::{xw_rdm_same_element, xw_rdm1, xw_rdm2_diff, xw_rdm2_same};
pub(crate) use layout::{assign_offsets, write_diff_spin, write_same_spin, write2t};
pub(crate) use storage::{create_wicks_mmap, load_wicks_mmap};
