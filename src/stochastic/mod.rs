// stochastic/mod.rs
//! Stochastic imaginary-time propagation in nonorthogonal determinant spaces.
//!
//! This module implements NOCIQMC without constructing the complete Hamiltonian or overlap
//! matrices. Matrix elements required by the current population and excitation generator are
//! evaluated on demand using the common NOCI matrix-element layer.
//!
//! Two population representations are provided: the signed-integer walker formulation and
//! the range-preserving direct-overlap formulation.
//!
//! # Signed-integer walker propagation
//!
//! The signed-integer formulation represents the determinant coefficient vector by populations
//! `n_x \in \mathbb Z and maintains the corresponding overlap-transformed population`
//!
//! `\tilde n_x = \sum_w S_{xw}n_w.`
//!
//! Spawning samples the off-diagonal propagator, death or cloning samples its diagonal
//! contribution, and annihilation combines changes of opposite sign on the same determinant.
//! The unshifted, shifted, doubly shifted and difference-doubly-shifted propagators are defined
//! by the residual
//!
//! `(\mathbf H - E_s^S\mathbf S)\mathbf n`
//!
//! `together with an identity shift controlled by E_s, E_s^S, or their difference. The`
//! coefficient and overlap-transformed populations therefore have separate population-control
//! shifts and population statistics.
//!
//! Because an extended NOCI basis may be overcomplete, stochastic noise can generate
//! `coefficient population in \operatorname{null}(\mathbf S). Such population represents the`
//! zero state of the Hilbert space and does not affect physical observables, but it can increase
//! the total walker population and computational cost.
//!
//! # Direct-overlap propagation
//!
//! The direct-overlap formulation instead stores the real metric population
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
//! `\mathbf\Delta = -\Delta\tau(\mathbf H - E_s^S\mathbf S)\tilde{\mathbf N}.`
//!
//! The persistent population is changed only through the explicit overlap action
//!
//! `\mathbf N' = \mathbf N + \mathbf S\mathbf\Delta.`
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
//! Fast Randomised Iteration-style stochastic compression may be applied to the persistent
//! metric population and to individual generated population changes. Both compression maps
//! preserve their conditional expectation while reducing the number or magnitude of values
//! entering the spawning and communication steps.
//!
//! # Shared stochastic infrastructure
//!
//! The stochastic implementations share:
//!
//! - Uniform and heat-bath excitation generation;
//! - On-demand Hamiltonian and overlap matrix elements;
//! - Determinant ownership and sparse population exchange across MPI ranks;
//! - Persistent Rayon worker states and nonorthogonal Wick scratch;
//! - Projected-energy and population statistics;
//! - Population-control shift updates;
//! - Report-block accumulation, output, stopping conditions and restart support.
//!
//! The stochastic driver selects the population representation from the configured propagator,
//! using direct-overlap propagation for `Propagator::DirectOverlap` and the signed-walker
//! implementation for the remaining stochastic propagators.

mod common;
mod excit;
mod init;
mod metric;
mod propagate;
mod report;
mod restart;
mod state;
mod walkers;

// Public type re-exports.
pub use state::{ExcitationHist, QMCTimings};

// Public function re-exports.
pub use propagate::qmc_step;
