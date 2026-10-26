// qmc/mod.rs
//! Shared stochastic-propagation layer used by every Monte Carlo method family.
//!
//! This layer holds the machinery that does not depend on how a method represents its
//! wavefunction: seeded random-number streams, MPI ownership of sparse keys, exchange and
//! annihilation of keyed population updates, Fast Randomised Iteration compression, the
//! population-control shift, stop requests, a fast deterministic hash map, and excitation
//! generators within one orthonormal orbital frame. Method subfolders such as `noci::stochastic` and `nocc::stochastic` build their
//! population storage and propagators on top of it. This layer never depends on a method family.
//!
//! # References
//!
//! - Population dynamics, annihilation and shift control: G. H. Booth, A. J. W. Thom and
//!   A. Alavi, *J. Chem. Phys.* **131**, 054106 (2009),
//!   [doi:10.1063/1.3193710](https://doi.org/10.1063/1.3193710).

pub(crate) mod excitgen;

mod exchange;
mod fri;
mod hash;
mod ownership;
mod report;
mod rng;
mod shift;

// Crate-visible type re-exports.
pub(crate) use exchange::{ExchangeScratch, KeyedUpdate};
pub(crate) use fri::FriAmplitude;
pub(crate) use hash::FastMap;
pub(crate) use rng::QmcRng;

// Crate-visible function re-exports.
pub(crate) use exchange::{coalesce, exchange_updates, gather_updates, pack_updates};
pub(crate) use fri::{
    compress_sparse, cutoff_candidate, pivotal_push, resolve_pivotal, target_cutoff,
};
pub(crate) use ownership::{owner, owner_words};
pub(crate) use report::stop_requested;
pub(crate) use rng::{rank_seed, stream};
pub(crate) use shift::update_shift;
