// noci/stochastic/mod.rs
//! Stochastic imaginary-time propagation in nonorthogonal determinant spaces.
//!
//! This module implements NOCIQMC without constructing the complete Hamiltonian or overlap
//! matrices. Matrix elements required by the current population and excitation generator are
//! evaluated on demand using the common NOCI matrix-element layer.
//!
//! Two population representations are provided: the sparse coefficient vector of the unshifted
//! propagator and the range populations of the S-apply and B-apply propagators. All three use
//! one population-control shift `E_s`, which multiplies the overlap.
//!
//! # Unshifted propagation
//!
//! The unshifted propagator samples the residual directly on the coefficient vector,
//!
//! `\mathbf c' = \mathbf c - \Delta\tau(\mathbf H - E_s\mathbf S)\mathbf c,`
//!
//! drawing both `H` and `S` elements for every spawn, so it never applies the overlap exactly and
//! its populations stay sparse. Its sampled residual is the same as S-apply's; only the persistent
//! update differs. The shift enters only through `E_s\mathbf S`, so it has no control over
//! `\operatorname{null}(\mathbf S)`: stochastic noise there grows without bound. The propagator is
//! therefore valid only when the overlap of the determinant space has full rank, and is best when
//! its smallest eigenvalue is not tiny.
//!
//! # Range propagation
//!
//! Both SApply and BApply instead store the real range population
//!
//! `\mathbf N = \mathbf S\mathbf c,`
//!
//! which lies in the physical range of the overlap matrix. A conditionally unbiased sparse
//! sample
//!
//! `\tilde{\mathbf N} = \Phi_c(\mathbf N), \mathbb E[\tilde{\mathbf N}\mid\mathbf N] = \mathbf N,`
//!
//! generates the pre-overlap change
//!
//! `\mathbf\Delta = -dt(\mathbf H - E_s\mathbf S)\tilde{\mathbf N}.`
//!
//! The persistent population is changed only through the explicit overlap action
//!
//! `\mathbf N' = \mathbf N + \mathbf S\mathbf\Delta.`
//!
//! BApply instead samples the orthogonal-space action
//!
//! `\boldsymbol\chi \simeq -dt(\hat{\mathbf H}-E_s)\mathbf B\tilde{\mathbf N}`
//!
//! and applies the exact final map
//!
//! `\mathbf N' = \mathbf N + \mathbf B^\dagger\boldsymbol\chi.`
//!
//! Therefore
//!
//! `\mathbf N' - \mathbf N \in \operatorname{range}(\mathbf S).`
//!
//! `Since the initial population is constructed as \mathbf N_0 = \mathbf S\mathbf c_0, induction`
//! `gives \mathbf N_k \in \operatorname{range}(\mathbf S) after every persistent update. The`
//! propagation consequently prevents stochastic population from accumulating in
//! `\operatorname{null}(\mathbf S) without diagonalising or inverting the overlap matrix.`
//!
//! SApply differentiates the same sampled linear map with respect to its one physical
//! shift. With `B = \partial\Delta/\partial E_s = dt S\tilde N`, the tangent of the final
//! persistent population is `\partial N'/\partial E_s = SB`. The second overlap action is required
//! because the controlled metric is evaluated after the persistent update.
//!
//! Fast Randomised Iteration-style stochastic compression may be applied to the persistent metric
//! population, individual generated changes, the coalesced report change, and the independently
//! coalesced shift tangent. Each compression map preserves its conditional expectation.
//!
//! # Shared stochastic infrastructure
//!
//! Seeded random streams, determinant ownership, sparse update exchange and annihilation across
//! MPI ranks, pivotal FRI compression, the logarithmic shift controller, stop requests and the
//! uniform and precomputed heat-bath (PCHB) parent-frame excitation generators come from
//! [`crate::qmc`]. This module adds the NOCIQMC-specific pieces:
//!
//! - Uniform, heat-bath and overlap-weighted selection of children from the whole determinant
//!   space;
//! - On-demand Hamiltonian and overlap matrix elements;
//! - Dense population storage and FRI sampling of it;
//! - Persistent Rayon worker states and nonorthogonal Wick scratch;
//! - Projected-energy and population statistics;
//! - The shift-tangent controller of the range propagators;
//! - Report-block accumulation, output and restart support.
//!
//! The stochastic driver selects the population representation from the configured propagator:
//! `Unshifted` uses the coefficient implementation, `SApply` and `BApply` the range branches.

mod bapply;
mod common;
mod excit;
mod fri;
mod init;
mod overlapweighted;
mod propagate;
mod report;
mod sapply;
mod shift;
mod state;
mod unshifted;

// Public type re-exports.
pub use crate::qmc::excitgen::ExcitationHist;
pub use state::QMCTimings;

// Public function re-exports.
pub use propagate::qmc_step;
