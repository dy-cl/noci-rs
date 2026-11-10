// noci/deterministic/mod.rs
//! Deterministic imaginary-time propagation in extended NOCI spaces.
//!
//! This module provides deterministic implementations of the propagators used to analyse and
//! validate NOCIQMC. Hamiltonian and overlap matrices are constructed explicitly, allowing
//! the complete population vector and its evolution to be examined without stochastic
//! sampling.
//!
//! Two population representations are supported, matching the stochastic propagators: the
//! unshifted propagator evolves the coefficient vector, and S-apply and B-apply a real population
//! constrained to the range of the overlap matrix. All three use one shift `E_s` multiplying the
//! overlap, controlled by the population `\|\cdot\|_1` of the propagated vector, as in the
//! stochastic code.
//!
//! # Unshifted propagation
//!
//! `For a coefficient vector \mathbf c, one deterministic iteration applies`
//!
//! `\mathbf c(\tau+\Delta\tau) = [\mathbf I - \Delta\tau(\mathbf H - E_s(\tau)\mathbf S)]\mathbf c(\tau).`
//!
//! `Population in \operatorname{null}(\mathbf S) is left unchanged by this map, so the`
//! propagator is suitable only for full-rank overlaps.
//!
//! # S-apply propagation
//!
//! `The S-apply propagator stores a real population \mathbf N initialised in`
//! `\operatorname{range}(\mathbf S), for example as`
//!
//! `\mathbf N_0 = \mathbf S\mathbf c_0.`
//!
//! One deterministic iteration applies
//!
//! `\mathbf N(\tau+\Delta\tau) = [\mathbf I - \Delta\tau\mathbf S(\mathbf H - E_s(\tau)\mathbf S)]\mathbf N(\tau).`
//!
//! Every population change therefore has the form
//!
//! `\Delta\mathbf N = -\Delta\tau\mathbf S(\mathbf H - E_s\mathbf S)\mathbf N \in \operatorname{range}(\mathbf S).`
//!
//! `Consequently, an initial population in \operatorname{range}(\mathbf S) remains in that`
//! range under every subsequent update. The propagation therefore prevents accumulation in
//! `\operatorname{null}(\mathbf S) without requiring inversion or diagonalisation of the`
//! overlap matrix.
//!
//! # B-apply propagation
//!
//! `The B-apply factor obeys \mathbf B^\dagger\mathbf B = \mathbf S and`
//! `\mathbf B^\dagger\hat{\mathbf H}\mathbf B = \mathbf H, so the expectation of its stochastic update is`
//!
//! `\mathbf N(\tau+\Delta\tau) = [\mathbf I - \Delta\tau(\mathbf H - E_s(\tau)\mathbf S)]\mathbf N(\tau),`
//!
//! the unshifted map applied to the range population `\mathbf N_0 = \mathbf S\mathbf c_0`. Its
//! changes also lie in the range of the overlap, and the auxiliary orthogonal space matters only
//! for how the stochastic propagator samples the update.
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
//! examine the effect of each propagator on redundant null-space population.
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
