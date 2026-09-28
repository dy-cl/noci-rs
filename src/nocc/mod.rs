// nocc/mod.rs
//! Highly experimental nonorthogonal generalised-normal-ordered coupled cluster.
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
//! The module implements deterministic GNOCC: the generated metric, residual, energy and
//! zeroth-order coupling expressions, the orbital and excitation spaces, the one- through
//! four-body cumulants of the reference, and the macro-micro amplitude solver. The reduced
//! density matrices of the reference come from [`crate::elements`]. It is organised as a
//! shared layer (`setup`, `cumulants`, `space`, `terms` and `equations`) with the approach in
//! `deterministic`; stochastic NOCCMC is planned as a sibling of `deterministic`.
//!
//! This implementation is highly experimental and is intended for method development rather
//! than production calculations. Its equations, truncations and interfaces remain subject to
//! substantial change and require further validation. Enabling the `nocc` feature also performs
//! extensive build-time equation generation.
//!
//! # References
//!
//! - Generalised normal ordering and extended Wick theory: Kutzelnigg and Mukherjee,
//!   *J. Chem. Phys.* **107**, 432 (1997), [doi:10.1063/1.474405](https://doi.org/10.1063/1.474405).
//! - Stochastic coupled cluster: Thom, *Phys. Rev. Lett.* **105**, 263004 (2010),
//!   [doi:10.1103/PhysRevLett.105.263004](https://doi.org/10.1103/PhysRevLett.105.263004).
//! - Spin-free generalised normal-ordered coupled cluster: Lee and Tew, *J. Chem. Phys.*
//!   **164**, 134118 (2026), [doi:10.1063/5.0311996](https://doi.org/10.1063/5.0311996).

mod cumulants;
mod deterministic;
mod equations;
mod setup;
mod space;
mod terms;

// Restricted type re-exports.
pub(crate) use deterministic::AmplitudeSolution;
pub(crate) use setup::ReferenceState;
pub(crate) use terms::TermEvaluator;

// Restricted function re-exports.
pub(crate) use cumulants::cumulants;
pub(crate) use deterministic::solve_amplitudes;
pub(crate) use setup::{
    noci_natural_orbitals, print_noci_natural_orbitals, reference_energy, transform_ao_data,
    transform_noci_basis,
};
pub(crate) use space::{build_excitations, build_fois_basis, build_spaces};
