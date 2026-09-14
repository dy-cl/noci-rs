// stochastic/bapply.rs

// Standard library imports.
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

// External crate imports.
use mpi::datatype::{Partition, PartitionMut};
use mpi::topology::Communicator;
use mpi::traits::*;
use rand::SeedableRng;

// Crate-root imports.
use crate::input::ExcitationGen;
use crate::noci::{
    MOCache, NOCIData, OrthogonalDetState, SpinFactorisation, calculate_h_pair_orthogonal,
};
use crate::nonorthogonalwicks::WickScratchSpin;

// Parent/sibling imports.
use super::common::{construct_qmc_run, gather_all_populations, population_stats_projected_energy};
use super::excit::pgen_orthogonal_uniform;
use super::fri::{FriAmplitude, compress_sparse, round, sample_populations, target_cutoff};
use super::init::initialise_qmc_state;
use super::report::{check_stop, print_header, print_initial_row, print_row, write_restart};
use super::shift::update_shift_tangent;
use super::state::{
    ExcitationHist, MPIScratch, PopulationUpdate, QMCRunInfo, QmcRng, SparsePopulations,
    orthogonal_owner,
};

/// Thread-local storage for BApply orthogonal-residual events.
struct BApplyThread {
    /// Uncoalesced diagonal and sampled off-diagonal residual events.
    updates: Vec<OrthogonalUpdate>,
    /// Spawning magnitudes accumulated for the optional excitation histogram.
    samples: Vec<f64>,
    /// Independent stochastic stream for orthogonal excitation generation and rounding.
    rng: QmcRng,
}

impl BApplyThread {
    /// Construct empty thread-local BApply residual storage.
    /// # Arguments:
    /// - `seed`: Initial random-number seed, replaced deterministically each cycle.
    /// # Returns
    /// - `Self`: Empty worker-local event and histogram buffers.
    fn new(seed: u64) -> Self {
        Self {
            updates: Vec::new(),
            samples: Vec::new(),
            rng: QmcRng::seed_from_u64(seed),
        }
    }
}

/// MPI representation of one sparse orthogonal-space residual amplitude.
#[repr(C)]
#[derive(Copy, Clone, Equivalence)]
struct OrthogonalUpdate {
    /// Parent reference index.
    parent: u64,
    /// Low 64 bits of the alpha occupation.
    oa_low: u64,
    /// High 64 bits of the alpha occupation.
    oa_high: u64,
    /// Low 64 bits of the beta occupation.
    ob_low: u64,
    /// High 64 bits of the beta occupation.
    ob_high: u64,
    /// Signed orthogonal-space residual amplitude.
    dn: f64,
}

impl OrthogonalUpdate {
    /// Construct an MPI update from an orthogonal determinant key and amplitude.
    /// # Arguments:
    /// - `det`: Orthogonal determinant key.
    /// - `dn`: Signed residual amplitude.
    /// # Returns
    /// - `Self`: MPI-safe sparse update.
    fn new(
        det: OrthogonalDetState,
        dn: f64,
    ) -> Self {
        Self {
            parent: det.parent as u64,
            oa_low: det.oa as u64,
            oa_high: (det.oa >> 64) as u64,
            ob_low: det.ob as u64,
            ob_high: (det.ob >> 64) as u64,
            dn,
        }
    }

    /// Recover the orthogonal determinant key carried by an MPI update.
    /// # Arguments:
    /// - `self`: Sparse MPI update.
    /// # Returns
    /// - `OrthogonalDetState`: Parent and spin occupations.
    fn state(&self) -> OrthogonalDetState {
        OrthogonalDetState {
            parent: self.parent as usize,
            oa: self.oa_low as u128 | (self.oa_high as u128) << 64,
            ob: self.ob_low as u128 | (self.ob_high as u128) << 64,
        }
    }
}

impl FriAmplitude for OrthogonalUpdate {
    /// Return one orthogonal-space residual amplitude.
    /// # Arguments:
    /// - `self`: Sparse orthogonal update.
    /// # Returns
    /// - `f64`: Signed residual amplitude.
    fn amplitude(&self) -> f64 {
        self.dn
    }

    /// Replace one orthogonal-space residual amplitude.
    /// # Arguments:
    /// - `self`: Sparse orthogonal update.
    /// - `amplitude`: Replacement residual amplitude.
    /// # Returns
    /// - `()`: Updates `dn` in place.
    fn set_amplitude(
        &mut self,
        amplitude: f64,
    ) {
        self.dn = amplitude;
    }
}

/// Reusable MPI storage for orthogonal residual redistribution and all-gather.
struct OrthogonalMPIScratch {
    /// Per-peer send counts.
    send_counts: Vec<i32>,
    /// Per-peer send displacements.
    send_displacements: Vec<i32>,
    /// Per-peer receive counts.
    recv_counts: Vec<i32>,
    /// Per-peer receive displacements.
    recv_displacements: Vec<i32>,
    /// Owner-ranked updates before contiguous packing.
    send_ranked: Vec<(usize, OrthogonalUpdate)>,
    /// Contiguous owner-redistribution send buffer.
    send: Vec<OrthogonalUpdate>,
    /// Contiguous owner-redistribution receive buffer.
    recv: Vec<OrthogonalUpdate>,
    /// Per-rank all-gather counts.
    gather_counts: Vec<i32>,
    /// Per-rank all-gather displacements.
    gather_displacements: Vec<i32>,
    /// Global gathered compressed orthogonal vector.
    gather: Vec<OrthogonalUpdate>,
}

impl OrthogonalMPIScratch {
    /// Construct reusable orthogonal-update MPI buffers.
    /// # Arguments:
    /// - `nranks`: Number of MPI ranks.
    /// # Returns
    /// - `Self`: Empty communication storage sized for `nranks`.
    fn new(nranks: usize) -> Self {
        Self {
            send_counts: vec![0; nranks],
            send_displacements: vec![0; nranks],
            recv_counts: vec![0; nranks],
            recv_displacements: vec![0; nranks],
            send_ranked: Vec::new(),
            send: Vec::new(),
            recv: Vec::new(),
            gather_counts: vec![0; nranks],
            gather_displacements: vec![0; nranks],
            gather: Vec::new(),
        }
    }
}

/// Generate one parallel stochastic cycle of the BApply orthogonal residual.
/// For each sampled source `x`, this accumulates
/// `\chi_D = -dt[<D^P|\hat H|\Phi_x^P> - E_s\delta_{Dx}]\tilde N_x`.
/// Sources are independent at fixed sampled population, so workers append uncoalesced events to
/// private sequential vectors without locking or hashing in the spawning loop.
/// # Arguments:
/// - `iteration`: Global stochastic cycle index used to seed worker streams.
/// - `sampled`: Sparse sampled NOCI populations `\tilde N`.
/// - `shift`: Current population-control shift `E_s`.
/// - `data`: Immutable stochastic propagation data.
/// - `mocache`: Parent-orthogonal MO integral caches.
/// - `run`: Rank-local propagation metadata.
/// - `workers`: Persistent thread-local BApply event buffers.
/// # Returns
/// - `()`: Appends this cycle's realised residual events to `workers`.
fn generate_orthogonal_cycle(
    iteration: usize,
    sampled: &SparsePopulations,
    shift: f64,
    data: &NOCIData<'_, f64>,
    mocache: &[MOCache<f64>],
    run: &QMCRunInfo,
    workers: &mut [Mutex<BApplyThread>],
) {
    let occupied = sampled.occ();
    if occupied.is_empty() {
        return;
    }
    let dt = data.input.prop_ref().dt;
    let spawn_cutoff = data.input.qmc.as_ref().unwrap().fri.spawn_cutoff;
    let record_samples = data.input.write.write_excitation_hist;
    let next = AtomicUsize::new(0);
    let workers_shared: &[Mutex<BApplyThread>] = workers;

    rayon::broadcast(|context| {
        let tid = context.index();
        let mut worker = workers_shared[tid].lock().unwrap();
        worker.rng = QmcRng::seed_from_u64(
            run.rank_seed ^ tid as u64 ^ (iteration as u64).wrapping_mul(0x9E3779B97F4A7C15),
        );
        loop {
            let start = next.fetch_add(8, Ordering::Relaxed);
            if start >= occupied.len() {
                break;
            }
            for &source_index in &occupied[start..(start + 8).min(occupied.len())] {
                let population = sampled.get(source_index);
                if population == 0.0 {
                    continue;
                }
                let source = &data.basis[source_index];
                let cache = &mocache[source.parent];
                let diagonal = calculate_h_pair_orthogonal(
                    data.ao,
                    cache,
                    (source.oa, source.ob),
                    (source.oa, source.ob),
                );
                let diagonal_update = -dt * (diagonal - shift) * population;
                if diagonal_update != 0.0 {
                    worker.updates.push(OrthogonalUpdate::new(
                        OrthogonalDetState {
                            parent: source.parent,
                            oa: source.oa,
                            ob: source.ob,
                        },
                        diagonal_update,
                    ));
                }

                let nattempts = population.abs().ceil().max(1.0) as usize;
                let attempted_population = population / nattempts as f64;
                for _ in 0..nattempts {
                    let Some((pgen, child)) =
                        pgen_orthogonal_uniform(source, cache, &mut worker.rng)
                    else {
                        continue;
                    };
                    let h = calculate_h_pair_orthogonal(
                        data.ao,
                        cache,
                        (child.oa, child.ob),
                        (source.oa, source.ob),
                    );
                    // Sample `D^P` with `P_\mathrm{gen}(D|x)` and accumulate the unbiased term
                    // `E[\chi_D] = -dt <D^P|\hat H|\Phi_x^P>\tilde N_x`.
                    let raw = -dt * h * attempted_population / pgen;
                    if record_samples {
                        worker.samples.push(raw.abs());
                    }
                    let dn = round(raw, spawn_cutoff, &mut worker.rng);
                    if dn != 0.0 {
                        worker.updates.push(OrthogonalUpdate::new(child, dn));
                    }
                }
            }
        }
    });
}

/// Collect report-level BApply events and optional histogram samples from all workers.
/// The concatenated update vector is deliberately left uncoalesced so one local sort combines
/// every contribution only after all stochastic cycles in the report have completed.
/// # Arguments:
/// - `workers`: Persistent thread-local BApply event buffers.
/// - `updates`: Rank-local report vector receiving every worker event.
/// - `histogram`: Optional excitation-magnitude histogram.
/// # Returns
/// - `()`: Drains worker buffers into report-level storage.
fn collect_orthogonal_updates(
    workers: &mut [Mutex<BApplyThread>],
    updates: &mut Vec<OrthogonalUpdate>,
    histogram: &mut Option<ExcitationHist>,
) {
    updates.clear();
    for worker in workers {
        let worker = worker.get_mut().unwrap();
        updates.append(&mut worker.updates);
        if let Some(histogram) = histogram.as_mut() {
            for sample in worker.samples.drain(..) {
                histogram.add(sample);
            }
        } else {
            worker.samples.clear();
        }
    }
}

/// Coalesce repeated orthogonal determinant amplitudes in place.
/// # Arguments:
/// - `updates`: Sparse orthogonal updates with arbitrary ordering.
/// # Returns
/// - `()`: Sorts by determinant key and combines repeated amplitudes.
fn coalesce_orthogonal_updates(updates: &mut Vec<OrthogonalUpdate>) {
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

/// Redistribute local orthogonal events to deterministic determinant owners.
/// # Arguments:
/// - `local`: Rank-local uncoalesced orthogonal events.
/// - `scratch`: Reusable orthogonal MPI storage.
/// - `world`: MPI communicator.
/// # Returns
/// - `()`: Places complete owner-local contributions in `scratch.recv`.
fn redistribute_orthogonal_updates(
    local: &mut Vec<OrthogonalUpdate>,
    scratch: &mut OrthogonalMPIScratch,
    world: &impl Communicator,
) {
    if world.size() == 1 {
        scratch.recv.clear();
        scratch.recv.append(local);
        return;
    }

    scratch.send_ranked.clear();
    for update in local.drain(..) {
        let peer = orthogonal_owner(&update.state(), world.size() as usize);
        scratch.send_ranked.push((peer, update));
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
    scratch.send.clear();
    for &(peer, update) in &scratch.send_ranked {
        scratch.send_counts[peer] += 1;
        scratch.send.push(update);
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
    scratch.recv.resize(
        nrecv,
        OrthogonalUpdate::new(
            OrthogonalDetState {
                parent: 0,
                oa: 0,
                ob: 0,
            },
            0.0,
        ),
    );
    let send = Partition::new(
        &scratch.send[..],
        &scratch.send_counts[..],
        &scratch.send_displacements[..],
    );
    let mut recv = PartitionMut::new(
        &mut scratch.recv[..],
        &scratch.recv_counts[..],
        &scratch.recv_displacements[..],
    );
    world.all_to_all_varcount_into(&send, &mut recv);
}

/// Gather owner-compressed orthogonal vectors so every rank sees identical `chi`.
/// # Arguments:
/// - `owned`: Complete compressed updates owned by this rank.
/// - `scratch`: Reusable orthogonal MPI storage.
/// - `world`: MPI communicator.
/// # Returns
/// - `&[OrthogonalUpdate]`: Identical global compressed orthogonal vector on every rank.
fn gather_orthogonal_updates<'a>(
    owned: &[OrthogonalUpdate],
    scratch: &'a mut OrthogonalMPIScratch,
    world: &impl Communicator,
) -> &'a [OrthogonalUpdate] {
    if world.size() == 1 {
        scratch.gather.clear();
        scratch.gather.extend_from_slice(owned);
        return &scratch.gather;
    }
    let nsend = owned.len() as i32;
    world.all_gather_into(&nsend, &mut scratch.gather_counts[..]);
    let mut ntotal = 0usize;
    for peer in 0..scratch.gather_counts.len() {
        scratch.gather_displacements[peer] = ntotal as i32;
        ntotal += scratch.gather_counts[peer] as usize;
    }
    if ntotal == 0 {
        scratch.gather.clear();
        return &scratch.gather;
    }
    scratch.gather.resize(
        ntotal,
        OrthogonalUpdate::new(
            OrthogonalDetState {
                parent: 0,
                oa: 0,
                ob: 0,
            },
            0.0,
        ),
    );
    let mut recv = PartitionMut::new(
        &mut scratch.gather[..],
        &scratch.gather_counts[..],
        &scratch.gather_displacements[..],
    );
    world.all_gather_varcount_into(owned, &mut recv);
    &scratch.gather
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
    let factor_cache = data.input.wicks.cachedir.as_deref().unwrap_or(".");
    let overlap_factors = factorisation.build_overlap_factors(
        data,
        Path::new(factor_cache),
        world.rank(),
        qmc.factor_tables,
        false,
    );
    let mut overlap_scratch = factorisation.overlap_scratch();
    let mut orthogonal_scratch = factorisation.orthogonal_overlap_scratch(data);
    let mut mpi = MPIScratch::new(run.nranks);
    let mut orthogonal_mpi = OrthogonalMPIScratch::new(run.nranks);
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
        .map(|tid| Mutex::new(BApplyThread::new(run.rank_seed ^ tid as u64)))
        .collect::<Vec<_>>();
    let mut local_updates = Vec::new();
    let mut global_chi = Vec::new();
    let mut tangent_values = vec![0.0; run.owned.len()];
    let mut tangent_updates = Vec::<PopulationUpdate>::new();
    let mut propagated_tangent = vec![0.0; run.owned.len()];
    let mut sample_chunks = Vec::new();
    let mut local_pos = vec![usize::MAX; run.ndets];
    for (position, &det) in run.owned.iter().enumerate() {
        local_pos[det] = position;
    }
    let mut chi_cutoff_hint = 0.0;
    let mut tangent_cutoff_hint = 0.0;

    for report in state.start_report..qmc.nreports {
        tangent_values.fill(0.0);
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

            for &source_index in state.mc.sampled.occ() {
                let population = state.mc.sampled.get(source_index);
                tangent_values[local_pos[source_index]] += data.input.prop_ref().dt * population;
            }
            // For each sampled source `x`, accumulate
            // `\chi_D = -dt[<D^P|\hat H|\Phi_x^P> - E_s\delta_{Dx}]\tilde N_x`.
            // Source determinants are independent within one fixed-population report cycle, so
            // orthogonal-space events are generated into thread-local sequential buffers.
            generate_orthogonal_cycle(
                iteration,
                &state.mc.sampled,
                *es,
                data,
                mocache,
                &run,
                &mut workers,
            );
        }

        collect_orthogonal_updates(
            &mut workers,
            &mut local_updates,
            &mut state.mc.excitation_hist,
        );
        coalesce_orthogonal_updates(&mut local_updates);
        redistribute_orthogonal_updates(&mut local_updates, &mut orthogonal_mpi, world);
        let mut owner_updates = std::mem::take(&mut orthogonal_mpi.recv);
        coalesce_orthogonal_updates(&mut owner_updates);
        let chi_cutoff = target_cutoff(
            &owner_updates,
            qmc.fri.pre_overlap_target_nnz,
            chi_cutoff_hint,
            |update| update.dn.abs(),
        );
        chi_cutoff_hint = chi_cutoff;
        let mut fri_rng = QmcRng::seed_from_u64(run.rank_seed ^ 0xA0761D6478BD642F ^ report as u64);
        compress_sparse(&mut owner_updates, chi_cutoff, &mut fri_rng);
        let gathered = gather_orthogonal_updates(&owner_updates, &mut orthogonal_mpi, world);
        global_chi.clear();
        global_chi.extend(gathered.iter().map(|update| (update.state(), update.dn)));
        orthogonal_mpi.recv = owner_updates;
        factorisation.apply_orthogonal_overlap_sparse(
            &mut state.mc.populations,
            &run.owned,
            &global_chi,
            data,
            &mut orthogonal_scratch,
        );

        tangent_updates.clear();
        tangent_updates.extend(
            run.owned
                .iter()
                .zip(tangent_values.iter())
                .filter(|(_, value)| **value != 0.0)
                .map(|(&det, &dn)| PopulationUpdate {
                    det: det as u64,
                    dn,
                }),
        );
        let tangent_cutoff = target_cutoff(
            &tangent_updates,
            qmc.fri.shift_tangent_target_nnz,
            tangent_cutoff_hint,
            |update| update.dn.abs(),
        );
        tangent_cutoff_hint = tangent_cutoff;
        compress_sparse(&mut tangent_updates, tangent_cutoff, &mut fri_rng);
        let global_tangent = gather_all_populations(world, &tangent_updates, &mut mpi);
        propagated_tangent.fill(0.0);
        // `T = \partial N'/\partial E_s = dt S \sum_a\tilde N^{(a)}` for BApply.
        // The shared tangent controller therefore sees the complete physical population derivative.
        factorisation.apply_overlap_sparse(
            &mut propagated_tangent,
            &run.owned,
            global_tangent
                .iter()
                .map(|update| (update.det as usize, update.dn)),
            data,
            &overlap_factors,
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

    (state.eprojcur, state.mc.excitation_hist)
}
