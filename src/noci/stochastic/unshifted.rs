// noci/stochastic/unshifted.rs
//! Unshifted propagation of real-valued coefficient populations,
//! `c' = c - \Delta\tau(H - E_s S)c`, valid only when the overlap has full rank.
//!
//! # References
//!
//! - Fermion Monte Carlo walker dynamics: Booth, Thom, and Alavi, *J. Chem. Phys.* **131**,
//!   054106 (2009), [doi:10.1063/1.3193710](https://doi.org/10.1063/1.3193710).
//! - Baumgarten, *Nonorthogonal and Overcomplete Hilbert Spaces for Quantum Monte Carlo Methods*
//!   (M.Sc. thesis, Cambridge, 2023).

// Standard library imports.
use std::path::Path;
use std::sync::Mutex;

// External crate imports.
use mpi::topology::Communicator;

// Crate-root imports.
use crate::elements::NOCIData;
use crate::input::ExcitationGen;
use crate::noci::{OverlapFactors, SpinFactorisation};
use crate::qmc::{PopulationUpdate, coalesce};

// Parent/sibling imports.
use super::common::{exchange_accumulated_updates, take_population_changes};
use super::overlapweighted::OverlapWeightedGenerator;
use super::propagate::{NOCIPropagator, NOCIRun};
use super::state::{NOCIPropagationResult, NOCIThreadPropagation, QMCRunInfo, ScratchSize};

/// Apply direct coefficient-population changes to rank-local populations.
/// # Arguments:
/// - `populations`: Rank-local coefficient population vector.
/// - `changes`: Coalesced sparse global population changes.
/// - `local_pos`: Global determinant to rank-local position map.
/// # Returns:
/// - `()`: Updates rank-local populations in place.
fn apply_population_changes(
    populations: &mut [f64],
    changes: &[PopulationUpdate],
    local_pos: &[usize],
) {
    for update in changes {
        let det = update.index as usize;
        let pos = local_pos[det];

        if pos != usize::MAX {
            populations[pos] += update.dn;
        }
    }
}

/// The unshifted propagator of coefficient populations, `c' = c - \Delta\tau(H - E_s S)c`, with
/// the logarithmic shift controller.
pub(in crate::noci::stochastic) struct Unshifted {
    /// Overlap factor tables and overlap-weighted generator, when that generator is selected.
    overlap: Option<(OverlapFactors, OverlapWeightedGenerator)>,
    /// Rank-local position of every owned determinant, `usize::MAX` for the others.
    positions: Vec<usize>,
    /// Per-thread propagation workers.
    workers: Vec<Mutex<NOCIThreadPropagation>>,
    /// Reusable gathered spawns of one cycle.
    result: NOCIPropagationResult,
    /// Coalesced report change.
    changes: Vec<PopulationUpdate>,
}

impl<C: Communicator> NOCIPropagator<C> for Unshifted {
    /// Build the optional overlap-weighted generator, the owned positions and the workers.
    /// # Arguments:
    /// - `data`: Immutable stochastic propagation data.
    /// - `run`: Rank-local topology of the run.
    /// - `sizes`: Wick scratch dimensions of the worker threads.
    /// - `world`: MPI communicator.
    /// # Returns
    /// - `Self`: Unshifted propagator.
    fn new(
        data: &NOCIData<'_, f64>,
        run: &QMCRunInfo,
        sizes: ScratchSize,
        world: &C,
    ) -> Self {
        let qmc = data.input.qmc.as_ref().unwrap();

        // Build overlap-weighted proposal factors only for the corresponding generator.
        let overlap = (qmc.excitation_gen == ExcitationGen::OverlapWeighted).then(|| {
            let factorisation = SpinFactorisation::new(data);
            let factor_cache = data.input.wicks.cachedir.as_deref().unwrap_or(".");
            let factors = factorisation.build_overlap_factors(
                data,
                Path::new(factor_cache),
                world.rank(),
                qmc.factor_tables,
                true,
            );
            let generator = OverlapWeightedGenerator::new(data, &factorisation, &factors);
            (factors, generator)
        });

        // Map global determinant indices to this rank's compact population vector.
        let mut positions = vec![usize::MAX; run.ndets];
        for (k, &det) in run.owned.iter().enumerate() {
            positions[det] = k;
        }

        let workers = (0..rayon::current_num_threads())
            .map(|tid| {
                Mutex::new(NOCIThreadPropagation::with_sizes(
                    run.rank_seed ^ tid as u64,
                    sizes.maxsame,
                    sizes.maxla,
                    sizes.maxlb,
                ))
            })
            .collect();

        Self {
            overlap,
            positions,
            workers,
            result: NOCIPropagationResult::new(),
            changes: Vec::new(),
        }
    }

    /// Spawn one cycle from the sampled populations.
    /// # Arguments:
    /// - `self`: Unshifted propagator.
    /// - `run`: Shared run state.
    /// - `iteration`: Global iteration of the cycle.
    /// # Returns
    /// - `()`: Accumulates the cycle's change.
    fn cycle(
        &mut self,
        run: &mut NOCIRun<'_, C>,
        iteration: usize,
    ) {
        let overlap = self.overlap.as_ref();
        run.spawn(
            iteration,
            (
                overlap.map(|(factors, _)| factors),
                overlap.map(|(_, generator)| generator),
            ),
            &mut self.workers,
            &mut self.result,
        );
    }

    /// Route the report change to its owners, coalesce it and add it to the populations.
    /// # Arguments:
    /// - `self`: Unshifted propagator.
    /// - `run`: Shared run state.
    /// - `_`: Report ending with this update.
    /// # Returns
    /// - `Option<&[f64]>`: `None`, as the unshifted propagator uses the logarithmic controller.
    fn apply(
        &mut self,
        run: &mut NOCIRun<'_, C>,
        _: usize,
    ) -> Option<&[f64]> {
        exchange_accumulated_updates(&mut run.state.mc, &mut run.mpi, run.world, &run.run);
        take_population_changes(&mut run.state.mc, &mut self.changes);
        self.changes.sort_unstable_by_key(|update| update.index);
        coalesce(&mut self.changes);
        apply_population_changes(
            &mut run.state.mc.populations,
            &self.changes,
            &self.positions,
        );
        None
    }
}
