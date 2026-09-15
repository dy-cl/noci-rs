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
    NOCIData, OrthogonalDetState, OrthogonalOverlapScratch, OverlapFactors, SpinFactorisation,
};
use crate::nonorthogonalwicks::WickScratchSpin;

// Parent/sibling imports.
use super::common::{
    construct_qmc_run, population_stats_projected_energy, propagate_iteration_orthogonal,
};
use super::excit::OrthogonalUniformGenerator;
use super::fri::{compress_dense_to_sparse, compress_sparse, sample_populations, target_cutoff};
use super::init::initialise_qmc_state;
use super::report::{check_stop, print_header, print_initial_row, print_row, write_restart};
use super::shift::update_shift_tangent;
use super::state::{
    ExcitationHist, MPIScratch, MPIScratchOrthogonal, PopulationUpdate, PopulationUpdateOrthogonal,
    PropagationResultOrthogonal, QmcRng, ShiftTangent, ThreadPropagationOrthogonal,
};

/// Accumulate one cycle's `\chi_D^P` events and retain remote events for report exchange.
/// # Arguments:
/// - `result`: Ownership-separated worker propagation results.
/// - `updates`: Rank-local report residual accumulator.
/// - `scratch`: Persistent MPI exchange scratch for remote events.
/// - `histogram`: Optional excitation-magnitude histogram.
/// # Returns
/// - `()`: Drains cycle-local results into report accumulators.
fn accumulate_generated_updates_orthogonal(
    result: &mut PropagationResultOrthogonal,
    updates: &mut Vec<PopulationUpdateOrthogonal>,
    scratch: &mut MPIScratchOrthogonal,
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

/// Coalesce repeated orthogonal determinant amplitudes in place.
/// # Arguments:
/// - `updates`: Sparse orthogonal updates with arbitrary ordering.
/// # Returns
/// - `()`: Sorts by determinant key and combines repeated amplitudes.
fn coalesce_population_updates_orthogonal(updates: &mut Vec<PopulationUpdateOrthogonal>) {
    updates.sort_unstable_by_key(|update| {
        (
            update.parent,
            update.oa_high,
            update.oa_low,
            update.ob_high,
            update.ob_low,
        )
    });
    let mut out = 0usize;
    for i in 0..updates.len() {
        let update = updates[i];
        if out != 0 && updates[out - 1].state() == update.state() {
            updates[out - 1].dn += update.dn;
        } else {
            updates[out] = update;
            out += 1;
        }
    }
    updates.truncate(out);
    updates.retain(|update| update.dn != 0.0);
}

/// Exchange pre-routed orthogonal events and combine them with local report updates.
/// # Arguments:
/// - `local`: Rank-owned report residual updates.
/// - `scratch`: Reusable orthogonal MPI storage.
/// - `world`: MPI communicator.
/// # Returns
/// - `()`: Places complete owner-local contributions in `scratch.recv`.
fn redistribute_population_updates_orthogonal(
    local: &mut Vec<PopulationUpdateOrthogonal>,
    scratch: &mut MPIScratchOrthogonal,
    world: &impl Communicator,
) {
    if world.size() == 1 {
        scratch.recv_contig.clear();
        scratch.recv_contig.append(local);
        return;
    }

    scratch.send_ranked.sort_unstable_by_key(|(peer, update)| {
        (
            *peer,
            update.parent,
            update.oa_high,
            update.oa_low,
            update.ob_high,
            update.ob_low,
        )
    });
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
        PopulationUpdateOrthogonal::new(
            OrthogonalDetState {
                parent: 0,
                oa: 0,
                ob: 0,
            },
            0.0,
        ),
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

/// Gather owner-compressed orthogonal vectors so every rank sees identical `chi`.
/// # Arguments:
/// - `owned`: Complete compressed updates owned by this rank.
/// - `scratch`: Reusable orthogonal MPI storage.
/// - `world`: MPI communicator.
/// # Returns
/// - `&[PopulationUpdateOrthogonal]`: Identical global compressed vector on every rank.
fn gather_all_populations_orthogonal<'a>(
    owned: &[PopulationUpdateOrthogonal],
    scratch: &'a mut MPIScratchOrthogonal,
    world: &impl Communicator,
) -> &'a [PopulationUpdateOrthogonal] {
    if world.size() == 1 {
        scratch.gather_recv.clear();
        scratch.gather_recv.extend_from_slice(owned);
        return &scratch.gather_recv;
    }
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
        PopulationUpdateOrthogonal::new(
            OrthogonalDetState {
                parent: 0,
                oa: 0,
                ob: 0,
            },
            0.0,
        ),
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
/// - `scratch`: Canonical physical component registry.
/// - `spin`: Retained spin-component factorisation.
/// # Returns:
/// - `()`: Writes actual initial, final, and peak backing and added component counts.
fn print_bapply_storage(
    rank: usize,
    mode: crate::input::SNOCIStorage,
    initial: usize,
    factors: &OverlapFactors,
    scratch: &OrthogonalOverlapScratch,
    spin: &SpinFactorisation,
) {
    if rank != 0 {
        return;
    }
    let (final_bytes, _) = factors.storage_bytes();
    let (alpha, beta) = scratch.added_components(spin);
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
    println!("BApply added physical alpha components: {alpha}");
    println!("BApply added physical beta components: {beta}");
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
    let (isref, _, run) = construct_qmc_run(data, c0, ref_indices, world);
    let factorisation = SpinFactorisation::new(data);
    if factorisation.nparents() > 1 && data.wicks.is_none() {
        panic!("BApply cross-parent B^dagger requires Wick intermediates");
    }
    let generator = OrthogonalUniformGenerator::new(&data.basis[0], &mocache[data.basis[0].parent]);
    let factor_cache = data.input.wicks.cachedir.as_deref().unwrap_or(".");
    let mut overlap_factors = factorisation.build_overlap_factors(
        data,
        Path::new(factor_cache),
        world.rank(),
        qmc.bapply_factor_tables,
        false,
    );
    let initial_factor_bytes = overlap_factors.storage_bytes().0;
    let mut overlap_scratch = factorisation.overlap_scratch();
    let mut orthogonal_scratch = factorisation.orthogonal_overlap_scratch(data);
    let mut mpi = MPIScratch::new(run.nranks);
    let mut orthogonal_mpi = MPIScratchOrthogonal::new(run.nranks);
    let mut wick = WickScratchSpin::new();
    let mut state = initialise_qmc_state(c0, es, data, &run, &isref, &mut wick, (world, &mut mpi));
    let propagator = data.input.prop_ref().propagator;
    print_header(run.irank, propagator);
    print_initial_row(
        run.irank,
        state.start_report * qmc.ncycles,
        &state,
        data.basis[0].e,
        *es,
        propagator,
    );

    let mut workers = (0..rayon::current_num_threads())
        .map(|tid| Mutex::new(ThreadPropagationOrthogonal::new(run.rank_seed ^ tid as u64)))
        .collect::<Vec<_>>();
    let mut propagation_result = PropagationResultOrthogonal::new();
    let mut local_updates = Vec::new();
    let mut shift_tangent = ShiftTangent::new(run.ndets);
    let mut tangent_updates = Vec::<PopulationUpdate>::new();
    let mut propagated_tangent = vec![0.0; run.owned.len()];
    let mut sample_chunks = Vec::new();
    let mut chi_cutoff_hint = 0.0;
    let mut tangent_cutoff_hint = 0.0;

    for report in state.start_report..qmc.nreports {
        local_updates.clear();
        orthogonal_mpi.send_ranked.clear();
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

            propagate_iteration_orthogonal(
                (iteration, &state.mc.sampled),
                data,
                &run,
                *es,
                &generator,
                &mut workers,
                &mut propagation_result,
            );
            accumulate_generated_updates_orthogonal(
                &mut propagation_result,
                &mut local_updates,
                &mut orthogonal_mpi,
                &mut state.mc.excitation_hist,
            );
        }
        coalesce_population_updates_orthogonal(&mut local_updates);
        redistribute_population_updates_orthogonal(&mut local_updates, &mut orthogonal_mpi, world);
        let mut owner_updates = std::mem::take(&mut orthogonal_mpi.recv_contig);
        coalesce_population_updates_orthogonal(&mut owner_updates);
        let chi_cutoff = target_cutoff(
            &owner_updates,
            qmc.fri.pre_overlap_target_nnz,
            chi_cutoff_hint,
            |update| update.dn.abs(),
        );
        chi_cutoff_hint = chi_cutoff;
        let mut fri_rng = QmcRng::seed_from_u64(run.rank_seed ^ 0xA0761D6478BD642F ^ report as u64);
        compress_sparse(&mut owner_updates, chi_cutoff, &mut fri_rng);
        let gathered =
            gather_all_populations_orthogonal(&owner_updates, &mut orthogonal_mpi, world);
        factorisation.apply_orthogonal_overlap_sparse(
            &mut state.mc.populations,
            &run.owned,
            gathered.iter().map(|update| (update.state(), update.dn)),
            data,
            &mut overlap_factors,
            &mut orthogonal_scratch,
        );
        orthogonal_mpi.recv_contig = owner_updates;

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
                &orthogonal_scratch,
                &factorisation,
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
            data.basis[0].e,
            *es,
            propagator,
        );
    }

    print_bapply_storage(
        run.irank,
        qmc.bapply_factor_tables,
        initial_factor_bytes,
        &overlap_factors,
        &orthogonal_scratch,
        &factorisation,
    );
    (state.eprojcur, state.mc.excitation_hist)
}
