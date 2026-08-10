// nocc/mod.rs
//! Highly experimental nonorthogonal coupled-cluster and NOCCMC support.
//!
//! This feature-gated module develops coupled-cluster theory using a correlated NOCI reference
//! state
//!
//! `|\Phi\rangle = |\Psi_{\mathrm{NOCI}}\rangle = \sum_x c_x|{}^x\Psi\rangle.`
//!
//! `Generalised normal ordering treats |\Phi\rangle as the vacuum and defines contractions`
//! through its reduced density matrices and cumulants. The current formulation uses the
//! generalised-normal-ordered ansatz
//!
//! `|\Psi_{\mathrm{NOCC}}\rangle = \{e^{\hat T}\}|\Phi\rangle,`
//!
//! where
//!
//! `\hat T = \sum_\mu t_\mu\hat\tau_\mu`
//!
//! contains spin-free excitation operators. The energy and connected amplitude residuals are
//!
//! `E = \langle\Phi|\hat H\{e^{\hat T}\}|\Phi\rangle,`
//!
//! `R_\mu = \langle\Phi|\hat\tau_\mu^\dagger\hat H\{e^{\hat T}\}|\Phi\rangle_{\mathrm c},`
//!
//! `with the coupled-cluster solution satisfying R_\mu = 0 for every retained excitation.`
//!
//! The module provides the generated overlap and residual expressions, excitation spaces,
//! one- through four-body reduced density matrices and their associated cumulants required by
//! the current stochastic NOCCMC implementation.
//!
//! This implementation is highly experimental and is intended for method development rather
//! than production calculations. Its equations, truncations, interfaces and stochastic
//! propagation remain subject to substantial change and require further validation. Enabling
//! the `nocc` feature also performs extensive build-time equation generation.

mod common;
mod cumulants;
mod driver;
mod loader;
mod overlap;
mod rdm;
mod residual;
mod space;
mod terms;

// Crate-visible type re-exports.
pub(crate) use cumulants::Cumulants;
pub(crate) use rdm::{RDM1, RDM2, RDM3, RDM4};

// Crate-visible function re-exports.
pub(crate) use cumulants::cumulants;
pub(crate) use driver::run_noccmc;
pub(crate) use rdm::{rdm1, rdm2, rdm3, rdm4};
