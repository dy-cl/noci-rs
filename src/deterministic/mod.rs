// deterministic/mod.rs
//! Deterministic imaginary-time propagation in extended NOCI spaces.
//!
//! This module provides deterministic implementations of the propagators used to analyse and
//! validate NOCIQMC. Hamiltonian and overlap matrices are constructed explicitly, allowing
//! the complete population vector and its evolution to be examined without stochastic
//! sampling.
//!
//! Two population representations are supported: direct propagation of a coefficient vector
//! and direct-overlap propagation of a real population constrained to the range of the overlap
//! matrix.
//!
//! # Coefficient propagation
//!
//! `For a coefficient vector \mathbf c, the non-inverting propagators are constructed from the`
//! generalised-eigenvalue residual
//!
//! `\mathbf r = (\mathbf H - E_s^S\mathbf S)\mathbf c.`
//!
//! The unshifted, shifted, doubly shifted and difference-doubly-shifted propagators differ in
//! the identity contribution applied alongside this residual. The coefficient population and
//! overlap-transformed population
//!
//! `\tilde{\mathbf c} = \mathbf S\mathbf c`
//!
//! `may therefore be controlled using separate shifts E_s and E_s^S.`
//!
//! # Direct-overlap propagation
//!
//! `The direct-overlap propagator stores a real population \mathbf N initialised in`
//! `\operatorname{range}(\mathbf S), for example as`
//!
//! `\mathbf N_0 = \mathbf S\mathbf c_0.`
//!
//! One deterministic iteration applies
//!
//! `\mathbf N(\tau+\Delta\tau) = [\mathbf I - \Delta\tau\mathbf S(\mathbf H - E_s^S(\tau)\mathbf S)]\mathbf N(\tau).`
//!
//! Every population change therefore has the form
//!
//! `\Delta\mathbf N = -\Delta\tau\mathbf S(\mathbf H - E_s^S\mathbf S)\mathbf N \in \operatorname{range}(\mathbf S).`
//!
//! `Consequently, an initial population in \operatorname{range}(\mathbf S) remains in that`
//! range under every subsequent update. The propagation therefore prevents accumulation in
//! `\operatorname{null}(\mathbf S) without requiring inversion or diagonalisation of the`
//! overlap matrix.
//!
//! # Relevant and null subspaces
//!
//! For diagnostic calculations, the overlap matrix may be diagonalised as
//!
//! `\mathbf S = \mathbf U\mathbf\Lambda\mathbf U^\dagger`
//!
//! and partitioned into relevant and null subspaces according to an eigenvalue threshold. The
//! corresponding projectors are
//!
//! `\mathbf P_r = \mathbf U_r\mathbf U_r^\dagger,`
//!
//! `\mathbf P_n = \mathbf U_n\mathbf U_n^\dagger.`
//!
//! These projectors allow the coefficient vector and propagator to be resolved into
//! relevant-relevant, relevant-null, null-relevant and null-null components. This is used to
//! examine the effect of each coefficient-space propagator on redundant null-space
//! population.
//!
//! Deterministic propagation is intended principally for small calculations, validation and
//! analysis of propagator behaviour. It provides explicit reference trajectories against which
//! stochastic projected energies, populations, compression schemes and null-space behaviour
//! may be compared.

pub mod nociqmc;
mod write;

// Public type re-exports.
pub use nociqmc::{Coefficients, ProjPropagator, Projectors};

// Public function re-exports.
pub use nociqmc::{projected_energy, propagate};
