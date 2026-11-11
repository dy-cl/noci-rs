// noci/stochastic/sapply.rs

// Standard library imports.
use std::path::Path;
use std::sync::Mutex;

// External crate imports.
use mpi::topology::Communicator;
use mpi::traits::*;
use rand::SeedableRng;

// Crate-root imports.
use crate::elements::NOCIData;
use crate::elements::nonorthogonalwicks::WickScratchSpin;
use crate::input::ExcitationGen;
use crate::noci::{OverlapFactors, OverlapScratch, SpinFactorisation};
use crate::qmc::{QmcRng, coalesce, compress_sparse, gather_updates, target_cutoff};
use crate::time_call;

// Parent/sibling imports.
use super::common::{exchange_accumulated_updates, take_population_changes};
use super::fri::compress_dense_to_sparse;
use super::overlapweighted::OverlapWeightedGenerator;
use super::propagate::{NOCIPropagator, NOCIRun};
use super::shift::{collect_shift_tangent, exchange_shift_tangent, reduce_shift_tangent};
use super::state::{
    NOCIMPIScratch, NOCIPopulationUpdate, NOCIPropagationResult, NOCIThreadPropagation, QMCRunInfo,
    ScratchSize, ShiftTangent,
};

/// `Apply \delta N_w = \sum_\Omega S_{w\Omega}\Delta_\Omega.`
/// # Arguments:
/// - `populations`: `Rank-local persistent populations N_w.`
/// - `updates`: `Sparse pre-overlap changes \Omega, \Delta_\Omega.`
/// - `data`: Immutable NOCI data.
/// - `overlap_factor`: Precomputed determinant and spin-component mappings.
/// - `overlap_factors`: Persistent cross-parent overlap factors.
/// - `targets`: Global determinant indices for rank-local rows.
/// - `scratch`: `Reusable allocation storage for one application of S\Delta.`
/// # Returns:
/// - `()`: `Applies N_w \leftarrow N_w + \delta N_w.`
fn apply_population_changes_local<I>(
    populations: &mut [f64],
    updates: I,
    data: &NOCIData<'_, f64>,
    overlap_factor: &SpinFactorisation,
    overlap_factors: &OverlapFactors,
    targets: &[usize],
    scratch: &mut OverlapScratch,
) where
    I: IntoIterator<Item = (usize, f64)>,
{
    overlap_factor.apply_overlap_sparse(
        populations,
        targets,
        updates,
        data,
        overlap_factors,
        scratch,
    );
}

/// Apply the overlap to the global range-population change, `N \leftarrow N + S\Delta`.
/// Each rank initially owns a subset of the accumulated changes
/// `\Delta_\Omega`. The changes are gathered across MPI ranks and each
/// rank updates its locally owned persistent populations according to
/// `N_w \leftarrow N_w + \sum_\Omega S_{w\Omega}\Delta_\Omega`.
/// Since every change has the form `S\Delta`, the population vector remains in
/// `\operatorname{range}(S)` provided the initial vector is in
/// `\operatorname{range}(S)`, therefore avoiding population growth in the null space.
/// # Arguments:
/// - `changes`: Rank-local persistent populations and local determinant population changes.
/// - `data`: Immutable stochastic propagation data.
/// - `overlap`: Spin factorisation and persistent cross-parent overlap factors.
/// - `run`: Rank-local run metadata.
/// - `mpi`: MPI communicator and reusable MPI scratch storage.
/// - `scratch`: `Reusable overlap allocation storage for grouped S\Delta application.`
/// # Returns
/// - `()`: Applies the overlap to the global range-population change.
pub(in crate::noci::stochastic) fn apply_overlap_population_changes(
    changes: (&mut [f64], &[NOCIPopulationUpdate]),
    data: &NOCIData<'_, f64>,
    overlap: (&SpinFactorisation, &OverlapFactors),
    run: &QMCRunInfo,
    mpi: (&impl CommunicatorCollectives, &mut NOCIMPIScratch),
    scratch: &mut OverlapScratch,
) {
    let (populations, dlocal) = changes;
    let (overlap_factor, overlap_factors) = overlap;
    let (world, mpi) = mpi;

    time_call!(crate::timers::stochastic::add_apply_overlap_changes, {
        if run.nranks == 1 {
            time_call!(
                crate::timers::stochastic::add_apply_local_overlap_changes,
                {
                    apply_population_changes_local(
                        populations,
                        dlocal.iter().map(|up| (up.det as usize, up.dn)),
                        data,
                        overlap_factor,
                        overlap_factors,
                        &run.owned,
                        scratch,
                    );
                }
            );
            return;
        }

        // Gather every rank's updates, so each rank applies the complete change to its owned rows.
        if gather_updates(world, dlocal, mpi).is_empty() {
            return;
        }

        // Apply all report updates in one pass. This avoids regrouping local and remote updates
        // separately, which is more expensive than the small all-gather wait on these workloads.
        time_call!(
            crate::timers::stochastic::add_apply_remote_overlap_changes,
            {
                mpi.gather_recv.sort_unstable_by_key(|update| update.det);
                coalesce(&mut mpi.gather_recv);

                apply_population_changes_local(
                    populations,
                    mpi.gather_recv.iter().map(|up| (up.det as usize, up.dn)),
                    data,
                    overlap_factor,
                    overlap_factors,
                    &run.owned,
                    scratch,
                );
            }
        );
    })
}

/// The $S$-apply range-preserving propagator.
/// `The initial population is N_0 = S c_0, rescaled to the requested population 1-norm.`
/// Within each report block, the population vector is held fixed while `ncycles` independent
/// `samples \tilde N^{(a)} = \Phi_c(N) generate pre-overlap changes`
/// `\Delta^{(a)} \approx -\Delta\tau(H - E_s S)\tilde N^{(a)}.`
/// At the end of the report block, the accumulated change is applied as
/// `N'= N + S\sum_{a = 1}^{n_{\text{cycles}}}\Delta^{(a)}.`
/// The same sampled paths produce `B = \partial\Delta/\partial E_s`, and the damped Newton
/// controller uses `d||N'||_1/dE_s = sign(N')^T SB`.
/// `This update preserves N \in \range(S) and removes null-space components.`
pub(in crate::noci::stochastic) struct SApply {
    /// Determinant and spin-string mappings of the overlap.
    factorisation: SpinFactorisation,
    /// Cross-parent overlap factor tables.
    factors: OverlapFactors,
    /// Overlap-weighted generator, when selected.
    generator: Option<OverlapWeightedGenerator>,
    /// Per-thread propagation workers.
    workers: Vec<Mutex<NOCIThreadPropagation>>,
    /// Reusable gathered spawns of one cycle.
    result: NOCIPropagationResult,
    /// Reusable storage of one application of the overlap.
    overlap_scratch: OverlapScratch,
    /// Coalesced report change `\Delta`.
    changes: Vec<NOCIPopulationUpdate>,
    /// Report shift tangent `B = \partial\Delta/\partial E_s`.
    tangent: ShiftTangent,
    /// Sparse report shift tangent after compression.
    tangent_changes: Vec<NOCIPopulationUpdate>,
    /// Propagated shift tangent `SB`.
    propagated: Vec<f64>,
    /// Last FRI cutoffs of the change and the tangent, which seed the next searches.
    hints: (f64, f64),
}

impl<C: Communicator> NOCIPropagator<C> for SApply {
    /// Build the overlap factor tables, the optional overlap-weighted generator and the workers.
    /// # Arguments:
    /// - `data`: Immutable stochastic propagation data.
    /// - `run`: Rank-local topology of the run.
    /// - `sizes`: Wick scratch dimensions of the worker threads.
    /// - `world`: MPI communicator.
    /// # Returns
    /// - `Self`: $S$-apply propagator.
    fn new(
        data: &NOCIData<'_, f64>,
        run: &QMCRunInfo,
        sizes: ScratchSize,
        world: &C,
    ) -> Self {
        let qmc = data.input.qmc.as_ref().unwrap();

        // Factor tables of `S`, with proposal CDFs for the overlap-weighted generator.
        let factorisation = SpinFactorisation::new(data);
        let build_overlap_cdfs = matches!(qmc.excitation_gen, ExcitationGen::OverlapWeighted);
        let factor_cache = data.input.wicks.cachedir.as_deref().unwrap_or(".");
        let factors = factorisation.build_overlap_factors(
            data,
            Path::new(factor_cache),
            world.rank(),
            qmc.sapply_factor_tables,
            build_overlap_cdfs,
        );
        if run.irank == 0 {
            let (tables, cdfs) = factors.storage_bytes();
            let mib = 1024.0 * 1024.0;
            println!(
                "SApply factor storage: {}",
                qmc.sapply_factor_tables.as_str()
            );
            println!("SApply factor tables: {:.3} MiB", tables as f64 / mib);
            println!("SApply proposal CDFs: {:.3} MiB", cdfs as f64 / mib);
            println!(
                "SApply total factor storage: {:.3} MiB",
                (tables + cdfs) as f64 / mib
            );
        }
        let generator = build_overlap_cdfs
            .then(|| OverlapWeightedGenerator::new(data, &factorisation, &factors));

        // One worker per thread, each with its own random stream and Wick scratch.
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
        if run.irank == 0 {
            println!(
                "Size of Wick's Scratch (MiB): {}",
                std::mem::size_of::<WickScratchSpin<f64>>() as f64 / (1024.0 * 1024.0)
            );
            type ThreadState = (
                Vec<(usize, f64)>,
                Vec<NOCIPopulationUpdate>,
                Vec<f64>,
                QmcRng,
                WickScratchSpin<f64>,
            );
            println!(
                "Size of per thread state (MiB): {}",
                std::mem::size_of::<ThreadState>() as f64 / (1024.0 * 1024.0)
            );
        }

        Self {
            overlap_scratch: factorisation.overlap_scratch(),
            factorisation,
            factors,
            generator,
            workers,
            result: NOCIPropagationResult::new(),
            changes: Vec::new(),
            tangent: ShiftTangent::new(run.ndets),
            tangent_changes: Vec::new(),
            propagated: vec![0.0; run.owned.len()],
            hints: (0.0, 0.0),
        }
    }

    /// Spawn one cycle of `\Delta^{(a)}` and `B^{(a)}` from the sampled populations.
    /// # Arguments:
    /// - `self`: $S$-apply propagator.
    /// - `run`: Shared run state.
    /// - `iteration`: Global iteration of the cycle.
    /// # Returns
    /// - `()`: Accumulates the cycle's change and tangent.
    fn cycle(
        &mut self,
        run: &mut NOCIRun<'_, C>,
        iteration: usize,
    ) {
        run.spawn(
            iteration,
            (Some(&self.factors), self.generator.as_ref()),
            &mut self.workers,
            &mut self.result,
        );
    }

    /// Compress the report change and tangent, then apply `N' = N + S\Phi(\Delta)` and propagate
    /// the tangent `\partial N'/\partial E_s = S\Phi(B)`.
    /// # Arguments:
    /// - `self`: $S$-apply propagator.
    /// - `run`: Shared run state.
    /// - `report`: Report ending with this update.
    /// # Returns
    /// - `Option<&[f64]>`: Propagated shift tangent `SB`.
    fn apply(
        &mut self,
        run: &mut NOCIRun<'_, C>,
        report: usize,
    ) -> Option<&[f64]> {
        let qmc = run.data.input.qmc.as_ref().unwrap();
        let (world, nranks) = (run.world, run.run.nranks);

        // Form `B = \sum_a dB^{(a)}` once per report. The single-rank path deliberately reduces
        // dense thread vectors because B is almost fully occupied before FRI.
        if nranks == 1 {
            reduce_shift_tangent(&mut self.workers, &mut self.tangent.values);
        } else {
            collect_shift_tangent(&mut self.workers, &mut self.tangent);
        }

        exchange_accumulated_updates(&mut run.state.mc, &mut run.mpi, world, &run.run);
        if nranks > 1 {
            exchange_shift_tangent(&mut self.tangent, &mut run.mpi, world, &run.run);
        }

        take_population_changes(&mut run.state.mc, &mut self.changes);
        if nranks > 1 {
            self.tangent.take_sparse(&mut self.tangent_changes);
        }

        self.changes.sort_unstable_by_key(|update| update.det);

        // Both report vectors are now coalesced by determinant ownership. Select target-NNZ
        // cutoffs only from these complete realised vectors, before any report-level FRI draw.
        // Choose c_Delta from `E[||Phi_c(Delta)||_0] = \sum_i min(1,|Delta_i|/c)` and compress
        // the complete physical report change before the outer overlap action.
        let mut fri_rng = QmcRng::seed_from_u64(
            run.run.rank_seed
                ^ 0xA0761D6478BD642F
                ^ (report as u64).wrapping_mul(0xE7037ED1A0B428DB),
        );
        let change_cutoff = target_cutoff(
            &self.changes,
            qmc.fri.pre_overlap_target_nnz,
            self.hints.0,
            |update| update.dn.abs(),
        );
        self.hints.0 = change_cutoff;
        compress_sparse(&mut self.changes, change_cutoff, &mut fri_rng);

        // Compress B independently. `E[Phi(B)|B] = B`, hence linearity of S gives
        // `E[S Phi(B)|B] = SB`.
        let mut tangent_rng = QmcRng::seed_from_u64(
            run.run.rank_seed
                ^ 0x8EBC6AF09C88C6E3
                ^ (report as u64).wrapping_mul(0x589965CC75374CC3),
        );
        let tangent_cutoff = if nranks == 1 {
            target_cutoff(
                &self.tangent.values,
                qmc.fri.shift_tangent_target_nnz,
                self.hints.1,
                |value| value.abs(),
            )
        } else {
            target_cutoff(
                &self.tangent_changes,
                qmc.fri.shift_tangent_target_nnz,
                self.hints.1,
                |update| update.dn.abs(),
            )
        };
        self.hints.1 = tangent_cutoff;
        if nranks == 1 {
            compress_dense_to_sparse(
                &mut self.tangent.values,
                tangent_cutoff,
                &mut tangent_rng,
                &mut self.tangent_changes,
            );
        } else {
            compress_sparse(&mut self.tangent_changes, tangent_cutoff, &mut tangent_rng);
        }

        // Apply the physical report update `N' = N + S Delta`.
        apply_overlap_population_changes(
            (&mut run.state.mc.populations, &self.changes),
            run.data,
            (&self.factorisation, &self.factors),
            &run.run,
            (world, &mut run.mpi),
            &mut self.overlap_scratch,
        );

        // Propagate the tangent through the same outer overlap,
        // `\partial N'/\partial E_s = SB`.
        self.propagated.fill(0.0);
        apply_overlap_population_changes(
            (&mut self.propagated, &self.tangent_changes),
            run.data,
            (&self.factorisation, &self.factors),
            &run.run,
            (world, &mut run.mpi),
            &mut self.overlap_scratch,
        );
        Some(&self.propagated)
    }

    /// $S$-apply has no final diagnostics.
    /// # Arguments:
    /// - `self`: $S$-apply propagator.
    /// - `_`: Shared run state.
    /// # Returns
    /// - `()`: Prints nothing.
    fn finish(
        &mut self,
        _: &NOCIRun<'_, C>,
    ) {
    }
}
