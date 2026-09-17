// stochastic/bapply.rs

// Standard library imports.
use std::path::Path;
use std::sync::Mutex;

// External crate imports.
use mpi::datatype::{Partition, PartitionMut};
use mpi::topology::Communicator;
use mpi::traits::*;
use rand::SeedableRng;

// Crate-root imports.
use crate::input::ExcitationGen;
use crate::noci::{
    AuxiliaryIndex, AuxiliarySpace, NOCIData, NOCIIndex, OverlapFactors, SpinFactorisation,
};
use crate::nonorthogonalwicks::WickScratchSpin;

// Parent/sibling imports.
use super::common::{
    construct_qmc_run, population_stats_projected_energy, propagate_iteration_auxiliary,
};
use super::excit::OrthogonalUniformGenerator;
use super::fri::{compress_dense_to_sparse, compress_sparse, sample_populations, target_cutoff};
use super::init::initialise_qmc_state;
use super::report::{check_stop, print_header, print_initial_row, print_row, write_restart};
use super::shift::update_shift_tangent;
use super::state::{
    AuxiliaryMPIScratch, AuxiliaryPopulationUpdate, AuxiliaryPropagationResult,
    AuxiliaryThreadPropagation, ExcitationHist, NOCIMPIScratch, NOCIPopulationUpdate, QmcRng,
    ShiftTangent,
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
    histogram: &mut Option<super::state::ExcitationHist>,
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

/// Coalesce repeated auxiliary determinant amplitudes in place.
/// # Arguments:
/// - `updates`: Sparse auxiliary updates with arbitrary ordering.
/// # Returns
/// - `()`: Sorts by determinant key and combines repeated amplitudes.
fn coalesce_auxiliary_population_updates(updates: &mut Vec<AuxiliaryPopulationUpdate>) {
    updates.sort_unstable_by_key(|update| update.det);

    let mut out = 0usize;
    for i in 0..updates.len() {
        let update = updates[i];
        if out != 0 && updates[out - 1].det == update.det {
            updates[out - 1].dn += update.dn;
        } else {
            updates[out] = update;
            out += 1;
        }
    }
    updates.truncate(out);
    updates.retain(|update| update.dn != 0.0);
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
    scratch
        .send_ranked
        .sort_unstable_by_key(|(peer, update)| (*peer, update.det));
    scratch.send_counts.fill(0);
    scratch.send_displacements.fill(0);
    scratch.send_contig.clear();
    for &(peer, update) in &scratch.send_ranked {
        scratch.send_counts[peer] += 1;
        scratch.send_contig.push(update);
    }
    let mut nsend = 0usize;
    for peer in 0..scratch.send_counts.len() {
        scratch.send_displacements[peer] = nsend as i32;
        nsend += scratch.send_counts[peer] as usize;
    }
    world.all_to_all_into(&scratch.send_counts[..], &mut scratch.recv_counts[..]);
    let mut nrecv = 0usize;
    for peer in 0..scratch.recv_counts.len() {
        scratch.recv_displacements[peer] = nrecv as i32;
        nrecv += scratch.recv_counts[peer] as usize;
    }
    scratch.recv_contig.resize(
        nrecv,
        AuxiliaryPopulationUpdate::new(AuxiliaryIndex(0), 0.0),
    );
    let send = Partition::new(
        &scratch.send_contig[..],
        &scratch.send_counts[..],
        &scratch.send_displacements[..],
    );
    let mut recv = PartitionMut::new(
        &mut scratch.recv_contig[..],
        &scratch.recv_counts[..],
        &scratch.recv_displacements[..],
    );
    world.all_to_all_varcount_into(&send, &mut recv);
    scratch.recv_contig.append(local);
    scratch.send_ranked.clear();
}

/// Gather owner-compressed auxiliary vectors so every rank sees identical `\chi`.
/// # Arguments:
/// - `owned`: Complete compressed updates owned by this rank.
/// - `scratch`: Reusable auxiliary MPI storage.
/// - `world`: MPI communicator.
/// # Returns
/// - `&[AuxiliaryPopulationUpdate]`: Identical global compressed vector on every rank.
fn gather_all_auxiliary_updates<'a>(
    owned: &[AuxiliaryPopulationUpdate],
    scratch: &'a mut AuxiliaryMPIScratch,
    world: &impl Communicator,
) -> &'a [AuxiliaryPopulationUpdate] {
    let nsend = owned.len() as i32;
    world.all_gather_into(&nsend, &mut scratch.gather_counts[..]);
    let mut ntotal = 0usize;
    for peer in 0..scratch.gather_counts.len() {
        scratch.gather_displs[peer] = ntotal as i32;
        ntotal += scratch.gather_counts[peer] as usize;
    }
    if ntotal == 0 {
        scratch.gather_recv.clear();
        return &scratch.gather_recv;
    }
    scratch.gather_recv.resize(
        ntotal,
        AuxiliaryPopulationUpdate::new(AuxiliaryIndex(0), 0.0),
    );
    let mut recv = PartitionMut::new(
        &mut scratch.gather_recv[..],
        &scratch.gather_counts[..],
        &scratch.gather_displs[..],
    );
    world.all_gather_varcount_into(owned, &mut recv);
    &scratch.gather_recv
}

/// Report persistent BApply factor backing and lazily discovered physical components.
/// # Arguments:
/// - `rank`: Current MPI rank; only rank zero writes storage output.
/// - `mode`: Requested shared `SNOCIStorage` backend.
/// - `initial`: Initial factor-table backing bytes.
/// - `factors`: Current persistent cross-parent factor tables.
/// # Returns
/// - `()`: Writes actual initial, final, and peak backing and added component counts.
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

/// Perform BApply stochastic range propagation.
/// `\chi \simeq -dt(\hat H-E_s)B N` is sampled in parent-orthogonal determinant spaces and the
/// persistent real range population is updated exactly as `N' = N + B^\dagger\chi`.
/// # Arguments:
/// - `data`: Immutable stochastic propagation data.
/// - `c0`: Initial determinant coefficient vector.
/// - `es`: Population-control shift energy.
/// - `ref_indices`: Determinants included in the reference-population norm.
/// - `world`: MPI communicator.
/// # Returns
/// - `(f64, Option<ExcitationHist>)`: Final projected energy and optional spawning histogram.
pub fn qmc_step(
    data: &NOCIData<'_, f64>,
    c0: &[f64],
    es: &mut f64,
    ref_indices: &[usize],
    world: &impl Communicator,
) -> (f64, Option<ExcitationHist>) {
    // Validate BApply-specific generator, checkpoint, and parent-orbital requirements.
    let qmc = data.input.qmc.as_ref().unwrap();
    if qmc.excitation_gen != ExcitationGen::Uniform {
        panic!("BApply supports only excitation_gen = \"uniform\"");
    }
    if let Some(interval) = data.input.write.write_restart_interval
        && (interval == 0 || interval % qmc.ncycles != 0)
    {
        panic!("write_restart_interval must be divisible by qmc.ncycles");
    }
    let mocache = data
        .mocache
        .expect("BApply requires parent MO integral caches");
    if mocache.iter().any(|cache| !cache.orthogonal_slater_condon) {
        panic!("BApply requires orthonormal orbitals within every parent MO basis");
    }
    // Build retained/auxiliary spin spaces and reusable overlap-factor storage.
    let (isref, _, run) = construct_qmc_run(data, c0, ref_indices, world);
    let factorisation = SpinFactorisation::new(data);
    let auxiliary = AuxiliarySpace::new(data.space);
    if factorisation.nparents() > 1 && data.wicks.is_none() {
        panic!("BApply cross-parent B^dagger requires Wick intermediates");
    }
    let generator = OrthogonalUniformGenerator::new(
        data.space.occupations(NOCIIndex(0)),
        &mocache[data.space.state(NOCIIndex(0)).parent],
    );
    let factor_cache = data.input.wicks.cachedir.as_deref().unwrap_or(".");
    let mut overlap_factors = factorisation.build_overlap_factors(
        data,
        Path::new(factor_cache),
        world.rank(),
        qmc.bapply_factor_tables,
        false,
    );
    let initial_factor_bytes = overlap_factors.storage_bytes().0;
    // Allocate persistent contraction, communication, and propagation state.
    let mut overlap_scratch = factorisation.overlap_scratch();
    let mut auxiliary_scratch = factorisation.auxiliary_overlap_scratch();
    let mut mpi = NOCIMPIScratch::new(run.nranks);
    let mut auxiliary_mpi = AuxiliaryMPIScratch::new(run.nranks);
    let mut wick = WickScratchSpin::new();
    let mut state = initialise_qmc_state(c0, es, data, &run, &isref, &mut wick, (world, &mut mpi));
    let propagator = data.input.prop_ref().propagator;
    print_header(run.irank, propagator);
    print_initial_row(
        run.irank,
        state.start_report * qmc.ncycles,
        &state,
        data.space.parents[0].e,
        *es,
        propagator,
    );

    // Give each Rayon worker independent RNG and auxiliary propagation scratch.
    let mut workers = (0..rayon::current_num_threads())
        .map(|tid| {
            Mutex::new(AuxiliaryThreadPropagation::new(
                run.rank_seed ^ tid as u64,
                factorisation.nparents(),
            ))
        })
        .collect::<Vec<_>>();
    let mut propagation_result = AuxiliaryPropagationResult::new();
    let mut local_updates = Vec::new();
    let mut shift_tangent = ShiftTangent::new(run.ndets);
    let mut tangent_updates = Vec::<NOCIPopulationUpdate>::new();
    let mut propagated_tangent = vec![0.0; run.owned.len()];
    let mut sample_chunks = Vec::new();
    let mut chi_cutoff_hint = 0.0;
    let mut tangent_cutoff_hint = 0.0;

    // Advance one report block at a time so communication and compression remain amortised.
    for report in state.start_report..qmc.nreports {
        local_updates.clear();
        auxiliary_mpi.send_ranked.clear();
        for cycle in 0..qmc.ncycles {
            let iteration = report * qmc.ncycles + cycle;
            let mut rng = QmcRng::seed_from_u64(
                run.rank_seed
                    ^ 0xD1B54A32D192ED03
                    ^ (iteration as u64).wrapping_mul(0x9E3779B97F4A7C15),
            );
            sample_populations(
                &state.mc.populations,
                &mut state.mc.sampled,
                qmc.fri.population_cutoff,
                &run,
                &mut rng,
                &mut sample_chunks,
            );

            propagate_iteration_auxiliary(
                (iteration, &state.mc.sampled),
                data,
                &run,
                *es,
                (&generator, &auxiliary),
                &mut workers,
                &mut propagation_result,
            );
            accumulate_generated_updates_auxiliary(
                &mut propagation_result,
                &mut local_updates,
                &mut auxiliary_mpi,
                &mut state.mc.excitation_hist,
            );
        }
        // Coalesce generated auxiliary amplitudes on their owning MPI ranks.
        coalesce_auxiliary_population_updates(&mut local_updates);
        let mut owner_updates = if run.nranks == 1 {
            // Preserve allocation across reports while avoiding MPI scratch and a second coalesce.
            std::mem::take(&mut local_updates)
        } else {
            redistribute_population_updates_auxiliary(
                &mut local_updates,
                &mut auxiliary_mpi,
                world,
            );

            let mut updates = std::mem::take(&mut auxiliary_mpi.recv_contig);
            coalesce_auxiliary_population_updates(&mut updates);
            updates
        };
        // Compress the auxiliary vector before applying `B^dagger` to physical populations.
        let chi_cutoff = target_cutoff(
            &owner_updates,
            qmc.fri.pre_overlap_target_nnz,
            chi_cutoff_hint,
            |update| update.dn.abs(),
        );
        chi_cutoff_hint = chi_cutoff;
        let mut fri_rng = QmcRng::seed_from_u64(run.rank_seed ^ 0xA0761D6478BD642F ^ report as u64);
        compress_sparse(&mut owner_updates, chi_cutoff, &mut fri_rng);
        if run.nranks == 1 {
            factorisation.apply_auxiliary_overlap_sparse(
                &mut state.mc.populations,
                &run.owned,
                owner_updates
                    .iter()
                    .map(|update| (update.index(), update.dn)),
                (data, &auxiliary),
                &mut overlap_factors,
                &mut auxiliary_scratch,
            );
        } else {
            let gathered = gather_all_auxiliary_updates(&owner_updates, &mut auxiliary_mpi, world);

            factorisation.apply_auxiliary_overlap_sparse(
                &mut state.mc.populations,
                &run.owned,
                gathered.iter().map(|update| (update.index(), update.dn)),
                (data, &auxiliary),
                &mut overlap_factors,
                &mut auxiliary_scratch,
            );
        }
        // Collect and compress `d chi / d E_s` in the representation local to this MPI mode.
        if run.nranks == 1 {
            local_updates = owner_updates;
        } else {
            auxiliary_mpi.recv_contig = owner_updates;
        }

        if run.nranks == 1 {
            super::sapply::reduce_shift_tangent(&mut workers, &mut shift_tangent.values);
        } else {
            super::sapply::collect_shift_tangent(&mut workers, &mut shift_tangent);
            super::sapply::exchange_shift_tangent(&mut shift_tangent, &mut mpi, world, &run);
            shift_tangent.take_sparse(&mut tangent_updates);
        }
        let tangent_cutoff = if run.nranks == 1 {
            target_cutoff(
                &shift_tangent.values,
                qmc.fri.shift_tangent_target_nnz,
                tangent_cutoff_hint,
                |value| value.abs(),
            )
        } else {
            target_cutoff(
                &tangent_updates,
                qmc.fri.shift_tangent_target_nnz,
                tangent_cutoff_hint,
                |update| update.dn.abs(),
            )
        };
        tangent_cutoff_hint = tangent_cutoff;
        if run.nranks == 1 {
            compress_dense_to_sparse(
                &mut shift_tangent.values,
                tangent_cutoff,
                &mut fri_rng,
                &mut tangent_updates,
            );
        } else {
            compress_sparse(&mut tangent_updates, tangent_cutoff, &mut fri_rng);
        }
        propagated_tangent.fill(0.0);
        // `T = \partial N'/\partial E_s = dt S \sum_a\tilde N^{(a)}` for BApply.
        // The shared tangent controller therefore sees the complete physical population derivative.
        super::sapply::apply_overlap_population_changes(
            (&mut propagated_tangent, &tangent_updates),
            data,
            (&factorisation, &overlap_factors),
            &run,
            (world, &mut mpi),
            &mut overlap_scratch,
        );

        // Update projected-energy statistics and the tangent-aware population-control shift.
        let end = (report + 1) * qmc.ncycles;
        let (stats, pe) = population_stats_projected_energy(&state.mc, &isref, &run, world);
        state.pe = pe;
        state.eprojcur = state.pe.num / state.pe.den;
        state.cur_pop = stats;
        update_shift_tangent(
            &stats,
            &mut state,
            es,
            &propagated_tangent,
            &run,
            world,
            data.input,
        );
        // Honour convergence stops and periodic restart checkpoints before reporting.
        if let Some(result) = check_stop(
            report,
            &mut state,
            *es,
            &run,
            world,
            data.input.write.write_restart.as_ref(),
        ) {
            print_bapply_storage(
                run.irank,
                qmc.bapply_factor_tables,
                initial_factor_bytes,
                &overlap_factors,
            );
            return result;
        }
        if let Some(interval) = data.input.write.write_restart_interval
            && end.is_multiple_of(interval)
        {
            write_restart(
                report,
                &state,
                *es,
                &run,
                world,
                data.input.write.write_restart.as_ref(),
            );
        }
        print_row(
            run.irank,
            end,
            &state,
            &stats,
            data.space.parents[0].e,
            *es,
            propagator,
        );
    }

    // Report final factor-table growth and return the last projected estimate.
    print_bapply_storage(
        run.irank,
        qmc.bapply_factor_tables,
        initial_factor_bytes,
        &overlap_factors,
    );
    (state.eprojcur, state.mc.excitation_hist)
}
