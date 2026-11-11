// noci/stochastic/bapply.rs

// Standard library imports.
use std::path::Path;
use std::sync::Mutex;

// External crate imports.
use mpi::topology::Communicator;
use rand::SeedableRng;

// Crate-root imports.
use crate::determinant::{AuxiliarySpace, NOCIIndex};
use crate::elements::NOCIData;
use crate::input::ExcitationGen;
use crate::noci::{AuxiliaryOverlapScratch, OverlapFactors, OverlapScratch, SpinFactorisation};
use crate::qmc::excitgen::{ExcitationHist, OrthogonalUniformGenerator};
use crate::qmc::{
    QmcRng, coalesce, compress_sparse, exchange_updates, gather_updates, pack_updates,
    target_cutoff,
};

// Parent/sibling imports.
use super::common::propagate_iteration_auxiliary;
use super::excit::{AuxiliaryGenerator, pchb_generator};
use super::propagate::{NOCIPropagator, NOCIRun};
use super::shift::ReportTangent;
use super::state::{
    AuxiliaryMPIScratch, AuxiliaryPopulationUpdate, AuxiliaryPropagationResult,
    AuxiliaryThreadPropagation, QMCRunInfo, ScratchSize,
};

/// Accumulate one cycle's `\chi_D^P` events and retain remote events for report exchange.
/// # Arguments:
/// - `result`: Ownership-separated worker propagation results.
/// - `updates`: Rank-local report residual accumulator.
/// - `scratch`: Persistent MPI exchange scratch for remote events.
/// - `histogram`: Optional excitation-magnitude histogram.
/// # Returns
/// - `()`: Drains cycle-local auxiliary results into report accumulators.
fn accumulate_generated_updates_auxiliary(
    result: &mut AuxiliaryPropagationResult,
    updates: &mut Vec<AuxiliaryPopulationUpdate>,
    scratch: &mut AuxiliaryMPIScratch,
    histogram: &mut Option<ExcitationHist>,
) {
    updates.append(&mut result.local);
    scratch.send_ranked.append(&mut result.remote);
    if let Some(histogram) = histogram.as_mut() {
        for sample in result.samples.drain(..) {
            histogram.add(sample);
        }
    } else {
        result.samples.clear();
    }
}

/// Exchange pre-routed auxiliary events and combine them with local report updates.
/// # Arguments:
/// - `local`: Rank-owned report residual updates.
/// - `scratch`: Reusable auxiliary MPI storage.
/// - `world`: MPI communicator.
/// # Returns
/// - `()`: Places complete owner-local contributions in `scratch.recv_contig`.
fn redistribute_population_updates_auxiliary(
    local: &mut Vec<AuxiliaryPopulationUpdate>,
    scratch: &mut AuxiliaryMPIScratch,
    world: &impl Communicator,
) {
    // Send every routed event, unmerged, to its owner and append the already-owned local
    // events to the same owner-local buffer.
    pack_updates(scratch.send_counts.len(), scratch, false);
    exchange_updates(world, scratch);
    scratch.recv_contig.append(local);
    scratch.send_ranked.clear();
}

/// Report persistent BApply factor backing and lazily discovered physical spin strings.
/// # Arguments:
/// - `rank`: Current MPI rank; only rank zero writes storage output.
/// - `mode`: Requested shared `SNOCIStorage` backend.
/// - `initial`: Initial factor-table backing bytes.
/// - `factors`: Current persistent cross-parent factor tables.
/// # Returns
/// - `()`: Writes actual initial, final, and peak backing and added spin string counts.
fn print_bapply_storage(
    rank: usize,
    mode: crate::input::SNOCIStorage,
    initial: usize,
    factors: &OverlapFactors,
) {
    if rank != 0 {
        return;
    }
    let (final_bytes, _) = factors.storage_bytes();
    let (alpha, beta) = factors.added_components();
    let mib = 1024.0 * 1024.0;
    println!("BApply factor storage: {}", mode.as_str());
    println!(
        "BApply initial factor tables: {:.3} MiB",
        initial as f64 / mib
    );
    println!(
        "BApply final factor tables: {:.3} MiB",
        final_bytes as f64 / mib
    );
    println!("BApply added alpha factor columns: {alpha}");
    println!("BApply added beta factor columns: {beta}");
    println!(
        "BApply peak factor storage: {:.3} MiB",
        final_bytes as f64 / mib
    );
}

/// The BApply range propagator.
/// `\chi \simeq -dt(\hat H-E_s)B N` is sampled in parent-orthogonal determinant spaces and the
/// persistent real range population is updated as `N_{r+1} = N_r + B^\dagger\chi_r`.
pub(in crate::noci::stochastic) struct BApply {
    /// Determinant and spin-string mappings of the overlap.
    factorisation: SpinFactorisation,
    /// Auxiliary space of every parent's connections.
    auxiliary: AuxiliarySpace,
    /// Parent-orthogonal proposal, uniform or PCHB.
    generator: AuxiliaryGenerator,
    /// Overlap factor tables, grown lazily as auxiliary spin strings are discovered.
    factors: OverlapFactors,
    /// Factor-table storage in bytes before propagation.
    initial_bytes: usize,
    /// Reusable storage of one application of the overlap.
    overlap_scratch: OverlapScratch,
    /// Reusable storage of one application of `B^\dagger`.
    auxiliary_scratch: AuxiliaryOverlapScratch,
    /// Auxiliary-space MPI exchange scratch.
    mpi: AuxiliaryMPIScratch,
    /// Per-thread propagation workers.
    workers: Vec<Mutex<AuxiliaryThreadPropagation>>,
    /// Reusable gathered spawns of one cycle.
    result: AuxiliaryPropagationResult,
    /// Report change `\chi` owned by this rank.
    local: Vec<AuxiliaryPopulationUpdate>,
    /// Last FRI cutoff of `\chi`, which seeds the next search.
    hint: f64,
    /// Report shift tangent.
    tangent: ReportTangent,
}

impl<C: Communicator> NOCIPropagator<C> for BApply {
    /// Check the parent orbitals, then build the auxiliary space, the proposal, the overlap factor
    /// tables and the workers.
    /// # Arguments:
    /// - `data`: Immutable stochastic propagation data.
    /// - `run`: Rank-local topology of the run.
    /// - `_`: Wick scratch dimensions, unused by the auxiliary workers.
    /// - `world`: MPI communicator.
    /// # Returns
    /// - `Self`: BApply propagator.
    /// # Panics
    /// - Panics for nonorthogonal parent orbitals, or missing MO caches or Wick data.
    fn new(
        data: &NOCIData<'_, f64>,
        run: &QMCRunInfo,
        _: ScratchSize,
        world: &C,
    ) -> Self {
        // Validate the parent orbitals; the input parser has already checked the generator.
        let qmc = data.input.qmc.as_ref().unwrap();
        let mocache = data
            .mocache
            .expect("BApply requires parent MO integral caches");
        if mocache.iter().any(|cache| !cache.orthogonal_slater_condon) {
            panic!("BApply requires orthonormal orbitals within every parent MO basis");
        }

        // Build retained/auxiliary spin spaces and reusable overlap-factor storage.
        let factorisation = SpinFactorisation::new(data);
        let auxiliary = AuxiliarySpace::new(data.space);
        if factorisation.nparents() > 1 && data.wicks.is_none() {
            panic!("BApply cross-parent B^dagger requires Wick intermediates");
        }

        // Select the parent-orthogonal proposal; PCHB tables are built once per parent.
        let generator = if qmc.excitation_gen == ExcitationGen::Pchb {
            let pchb = pchb_generator(data, &mocache[..factorisation.nparents()]);
            if run.irank == 0 {
                println!(
                    "PCHB excitation tables: {:.3} MiB",
                    pchb.bytes() as f64 / (1024.0 * 1024.0)
                );
            }
            AuxiliaryGenerator::Pchb(pchb)
        } else {
            AuxiliaryGenerator::Uniform(OrthogonalUniformGenerator::new(
                data.space.occupations(NOCIIndex(0)),
                &mocache[data.space.parent(NOCIIndex(0))],
            ))
        };
        let factor_cache = data.input.wicks.cachedir.as_deref().unwrap_or(".");
        let factors = factorisation.build_overlap_factors(
            data,
            Path::new(factor_cache),
            world.rank(),
            qmc.bapply_factor_tables,
            false,
        );

        // Give each Rayon worker independent RNG and auxiliary propagation scratch.
        let workers = (0..rayon::current_num_threads())
            .map(|tid| {
                Mutex::new(AuxiliaryThreadPropagation::new(
                    run.rank_seed ^ tid as u64,
                    factorisation.nparents(),
                ))
            })
            .collect();

        Self {
            initial_bytes: factors.storage_bytes().0,
            overlap_scratch: factorisation.overlap_scratch(),
            auxiliary_scratch: factorisation.auxiliary_overlap_scratch(),
            factorisation,
            auxiliary,
            generator,
            factors,
            mpi: AuxiliaryMPIScratch::new(run.nranks),
            workers,
            result: AuxiliaryPropagationResult::new(),
            local: Vec::new(),
            hint: 0.0,
            tangent: ReportTangent::new(run),
        }
    }

    /// Spawn one cycle of auxiliary events `\chi^{(a)}` from the sampled populations.
    /// # Arguments:
    /// - `self`: BApply propagator.
    /// - `run`: Shared run state.
    /// - `iteration`: Global iteration of the cycle.
    /// # Returns
    /// - `()`: Accumulates the cycle's events and tangent.
    fn cycle(
        &mut self,
        run: &mut NOCIRun<'_, C>,
        iteration: usize,
    ) {
        propagate_iteration_auxiliary(
            (iteration, &run.state.mc.sampled),
            run.data,
            &run.run,
            run.shift,
            (&self.generator, &self.auxiliary),
            &mut self.workers,
            &mut self.result,
        );
        accumulate_generated_updates_auxiliary(
            &mut self.result,
            &mut self.local,
            &mut self.mpi,
            &mut run.state.mc.excitation_hist,
        );
    }

    /// Coalesce and compress `\chi` on its owners, apply `N' = N + B^\dagger\Phi(\chi)`, and
    /// propagate the compressed tangent through the overlap.
    /// # Arguments:
    /// - `self`: BApply propagator.
    /// - `run`: Shared run state.
    /// - `report`: Report ending with this update.
    /// # Returns
    /// - `Option<&[f64]>`: Propagated shift tangent `\partial N'/\partial E_s`.
    fn apply(
        &mut self,
        run: &mut NOCIRun<'_, C>,
        report: usize,
    ) -> Option<&[f64]> {
        let qmc = run.data.input.qmc.as_ref().unwrap();
        let (world, nranks) = (run.world, run.run.nranks);

        // Coalesce generated auxiliary amplitudes on their owning MPI ranks.
        self.local.sort_unstable_by_key(|update| update.det);
        coalesce(&mut self.local);
        let mut owner_updates = if nranks == 1 {
            // Preserve allocation across reports while avoiding MPI scratch and a second coalesce.
            std::mem::take(&mut self.local)
        } else {
            redistribute_population_updates_auxiliary(&mut self.local, &mut self.mpi, world);
            let mut updates = std::mem::take(&mut self.mpi.recv_contig);
            updates.sort_unstable_by_key(|update| update.det);
            coalesce(&mut updates);
            updates
        };

        // Compress the auxiliary vector before applying `B^dagger` to physical populations.
        self.hint = target_cutoff(
            &owner_updates,
            qmc.fri.pre_overlap_target_nnz,
            self.hint,
            |update| update.dn.abs(),
        );
        let mut fri_rng =
            QmcRng::seed_from_u64(run.run.rank_seed ^ 0xA0761D6478BD642F ^ report as u64);
        compress_sparse(&mut owner_updates, self.hint, &mut fri_rng);
        let physical_updates: &[AuxiliaryPopulationUpdate] = if nranks == 1 {
            &owner_updates
        } else {
            gather_updates(world, &owner_updates, &mut self.mpi)
        };

        // Apply the realised range update `N_{r+1} = N_r + B^\dagger\chi_r`.
        self.factorisation.apply_auxiliary_overlap_sparse(
            &mut run.state.mc.populations,
            &run.run.owned,
            physical_updates
                .iter()
                .map(|update| (update.index(), update.dn)),
            (run.data, &self.auxiliary),
            &mut self.factors,
            &mut self.auxiliary_scratch,
        );

        // Recycle the report buffers, emptied for the next report's cycles.
        if nranks == 1 {
            self.local = owner_updates;
        } else {
            self.mpi.recv_contig = owner_updates;
        }
        self.local.clear();
        self.mpi.send_ranked.clear();

        // `T = \partial N'/\partial E_s = dt S \sum_a\tilde N^{(a)}` for BApply, compressed on the
        // continuing `\chi` stream.
        Some(self.tangent.propagate(
            &mut self.workers,
            run,
            &mut fri_rng,
            (&self.factorisation, &self.factors),
            &mut self.overlap_scratch,
        ))
    }

    /// Report the factor-table growth of the run.
    /// # Arguments:
    /// - `self`: BApply propagator.
    /// - `run`: Shared run state.
    /// # Returns
    /// - `()`: Prints the storage on the root rank.
    fn finish(
        &mut self,
        run: &NOCIRun<'_, C>,
    ) {
        let qmc = run.data.input.qmc.as_ref().unwrap();
        print_bapply_storage(
            run.run.irank,
            qmc.bapply_factor_tables,
            self.initial_bytes,
            &self.factors,
        );
    }
}
