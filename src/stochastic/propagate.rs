// stochastic/propagate.rs
use rand::rngs::SmallRng;
use rand::SeedableRng;
use mpi::topology::Communicator;
use mpi::collective::SystemOperation;
use mpi::traits::*;
use rayon::prelude::*;

use crate::input::{Input, Propagator};
use crate::nonorthogonalwicks::WickScratchSpin;
use crate::noci::{DetPair, NOCIData};
use crate::time_call;
use crate::timers::stochastic as stochastic_timers;
use super::state::{MCState, PopulationUpdate, ExcitationHist, ProjectedEnergyUpdate, PropagationState, MPIScratch, 
                   QMCRunInfo, ScratchSize, Shifts, PropagationResult, ThreadPropagation, PopulationStats};

use crate::noci::{calculate_s_pair, calculate_hs_pair};
use crate::mpiutils::{owner, communicate_spawn_updates, gather_all_walkers};
use super::report::{print_row, print_header, print_cached_row, print_initial_row, check_stop};
use super::init::{initialise_qmc_state, max_scratch_sizes};

/// Find overlap matrix element S_{ij}.
/// # Arguments:
/// - `data`: Immutable stochastic propagation data.
/// - `i`: Index of state `i`.
/// - `j`: Index of state `j`.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `f64`: Overlap matrix element `S_{ij}`.
pub(in crate::stochastic) fn find_s(data: &NOCIData<'_>, i: usize, j: usize, scratch: &mut WickScratchSpin) -> f64 {
    // Get the sorted pair of indices 
    let (a, b) = if i <= j {(i, j)} else {(j, i)};
    calculate_s_pair(data, DetPair::new(&data.basis[a], &data.basis[b]), Some(scratch))
}

/// Find Hamiltonian and overlap matrix elements H_{ij} and S_{ij}.
/// # Arguments:
/// - `data`: Immutable stochastic propagation data.
/// - `i`: Index of state `i`.
/// - `j`: Index of state `j`.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `(f64, f64)`: Hamiltonian and overlap matrix elements `H_{ij}` and `S_{ij}`.
pub(in crate::stochastic) fn find_hs(data: &NOCIData<'_>, i: usize, j: usize, scratch: &mut WickScratchSpin) -> (f64, f64) {
    // Get the sorted pair of indices 
    let (a, b) = if i <= j {(i, j)} else {(j, i)};
    calculate_hs_pair(data, DetPair::new(&data.basis[a], &data.basis[b]), Some(scratch))
}

/// Accumulate the population change dn for a determinant i into the per-iteration delta vector.
/// Note that the actual populations stored in mc.walkers are not yet changed here.
/// # Arguments:
/// - `mc`: Contains information about the current Monte Carlo state.
/// - `i`: Index of determinant i to be updated.
/// - `dn`: Population change on determinant i. 
/// # Returns
/// - `()`: Updates the accumulated population changes in place.
fn add_delta(mc: &mut MCState, i: usize, dn: i64) {
    if dn == 0 {return;}
    // If current delta for this determinant is zero this function being called is its first
    // modification for this iteration and so we record this fact in changed.
    if mc.delta[i] == 0 {
        mc.changed.push(i);
    }
    // Add the population change to delta.
    mc.delta[i] += dn;
}

/// Apply accumulated population changes from the mc.delta vector to the actual populations stored
/// in mc.walkers.
/// # Arguments:
/// - `mc`: Contains information about the current Monte Carlo state.
/// # Returns
/// - `Vec<PopulationUpdate>`: List of applied population updates for this iteration.
fn apply_delta(mc: &mut MCState) -> Vec<PopulationUpdate> {
    time_call!(stochastic_timers::add_apply_delta, {
        if mc.changed.is_empty() {return Vec::new();}
        let mut applied = Vec::with_capacity(mc.changed.len());

        for &i in &mc.changed {
            let dn = mc.delta[i];
            mc.delta[i] = 0;
            mc.walkers.add(i, dn);
            applied.push(PopulationUpdate {det: i as u64, dn});
        }

        mc.changed.clear();
        applied
    })
}

/// Compute the off-diagonal coupling between determinants depending on which propagator is used.
/// # Arguments:
/// - `hlg`: Matrix element H_{\Lambda\Gamma}
/// - `slg`: Matrix element S_{\Lambda\Gamma}
/// - `es_s`: E_s^S(\tau) shift energy.
/// - `es`: E_s(\tau) shift energy.
/// - `prop`: Chosen propagator.
/// # Returns
/// - `f64`: Off-diagonal coupling used by the propagator.
pub(in crate::stochastic) fn coupling(hlg: f64, slg: f64, es_s: f64, es: f64, prop: &Propagator) -> f64 {
    match prop {
        Propagator::Unshifted => hlg - es_s * slg,
        Propagator::Shifted => hlg - es_s * slg,
        Propagator::DoublyShifted => hlg - es_s * slg,
        Propagator::DifferenceDoublyShiftedU1 => hlg - 0.5 * (es + es_s) * slg,
        Propagator::DifferenceDoublyShiftedU2 => hlg - es_s * slg,
    }
}

/// Incrementally update the running projected-energy state using the net walker
/// population changes applied in the current iteration.
/// # Arguments:
/// - `d`: Net population changes applied in the current iteration.
/// - `pe`: Running projected-energy state to update in place.
/// - `data`: Immutable stochastic propagation data.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// # Returns
/// - `()`: Updates `pe` in place.
fn update_projected_energy(d: &[PopulationUpdate], pe: &mut ProjectedEnergyUpdate, data: &NOCIData<'_>, world: &impl Communicator) {
    time_call!(stochastic_timers::add_update_projected_energy, {
        let iref = pe.iref;

        let (dnum_local, dden_local) = d.par_iter().fold(|| (0.0_f64, 0.0_f64, WickScratchSpin::new()), |(mut dnum, mut dden, mut scratch), up| {
            let gamma = up.det as usize;
            let dn = up.dn as f64;
            let (hgr, sgr) = find_hs(data, gamma, iref, &mut scratch);
            dnum += dn * hgr;
            dden += dn * sgr;
            (dnum, dden, scratch)
        }).map(|(dnum, dden, _)| (dnum, dden)).reduce(|| (0.0, 0.0), |(a_num, a_den), (b_num, b_den)| (a_num + b_num, a_den + b_den));

        let mut dnum = 0.0;
        let mut dden = 0.0;
        world.all_reduce_into(&dnum_local, &mut dnum, SystemOperation::sum());
        world.all_reduce_into(&dden_local, &mut dden, SystemOperation::sum());

        pe.num += dnum;
        pe.den += dden;
    })
}

/// Update the vector `p_{\Gamma} = \sum_\Omega S_{\Gamma,\Omega} N_{\Omega}`.
/// # Arguments:
/// - `plocal`: Local portion of `p_{\Gamma}` on this rank.
/// - `dlocal`: Local determinant population updates.
/// - `data`: Immutable stochastic propagation data.
/// - `run`: Rank-local run metadata.
/// - `mpi`: Reusable MPI scratch space.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// # Returns
/// - `()`: Updates `plocal` in place.
fn update_p(plocal: &mut [f64], dlocal: &[PopulationUpdate], data: &NOCIData<'_>, run: &QMCRunInfo, mpi: &mut MPIScratch, world: &impl Communicator) {
    time_call!(stochastic_timers::add_update_p, {
        let dglobal = gather_all_walkers(world, dlocal, mpi);

        plocal.par_iter_mut().enumerate().for_each_init(WickScratchSpin::new, |scratch, (k, pgamma)| {
            let gamma = run.start + k;
            let mut dp = 0.0;
            for up in dglobal {
                let omega = up.det as usize;
                dp += find_s(data, gamma, omega, scratch) * up.dn as f64;
            }
            *pgamma += dp;
        });
    })
}

/// Perform spawning and death/cloning steps over the currently occupied determinants.
/// # Arguments:
/// - `it`: Current iteration number.
/// - `mc`: Contains the current Monte Carlo state.
/// - `data`: Immutable stochastic propagation data.
/// - `run`: Rank-local run metadata.
/// - `scratchsize`: Maximum sizes required for per-thread Wick scratch space.
/// - `shifts`: Current non-overlap and overlap-transformed shifts.
/// # Returns
/// - `PropagationResult`: Local and remote population updates together with spawning probability samples.
fn propagate_iteration(it: usize, mc: &MCState, data: &NOCIData<'_>, run: &QMCRunInfo, scratchsize: &ScratchSize, shifts: Shifts) -> PropagationResult {
    time_call!(stochastic_timers::add_propagate_iteration, {

        let initialise = || -> ThreadPropagation {
            let tid = rayon::current_thread_index().unwrap_or(0) as u64;
            ThreadPropagation{
                local: Vec::new(),
                remote: Vec::new(),
                samples: Vec::new(),
                rng: SmallRng::seed_from_u64(run.rank_seed ^ tid ^ ((it as u64).wrapping_mul(0x9E3779B97F4A7C15))),
                scratch: WickScratchSpin::with_sizes(scratchsize.maxsame, scratchsize.maxla, scratchsize.maxlb),
            }
        };

        let propagate = |mut acc: ThreadPropagation, &gamma: &usize| -> ThreadPropagation {
            let ngamma = mc.walkers.get(gamma);
            if ngamma == 0 {
                return acc;
            }

            acc.death_cloning(gamma, ngamma, shifts, data);
            acc.spawning(gamma, ngamma, shifts, data, run);
            acc
        };

        let merge = |mut a: ThreadPropagation, mut b: ThreadPropagation| -> ThreadPropagation {
            a.local.append(&mut b.local);
            a.remote.append(&mut b.remote);
            a.samples.append(&mut b.samples);
            a
        };

        let acc = mc.walkers.occ().par_iter().fold(initialise, propagate).reduce(
            || ThreadPropagation {
                local: Vec::new(),
                remote: Vec::new(),
                samples: Vec::new(),
                rng: SmallRng::seed_from_u64(0),
                scratch: WickScratchSpin::new(),
            },
            merge,
        );

        PropagationResult {
            local: acc.local,
            remote: acc.remote,
            samples: acc.samples,
        }
    })
}

/// Accumulate all local population updates into the global delta vector, update the local
/// excitation histogram if requested, and pack remote updates into per-rank MPI send buffers.
/// # Arguments:
/// - `mc`: Contains the current Monte Carlo state.
/// - `send`: Per-rank MPI send buffers.
/// - `prop`: Population updates generated in the spawning and death/cloning step.
/// - `input`: User specified input options.
/// - `nranks`: Total number of MPI ranks.
/// # Returns
/// - `()`: Updates the local delta vector, optional excitation histogram, and MPI send buffers.
fn acc_pack_updates(mc: &mut MCState, send: &mut [Vec<PopulationUpdate>], prop: PropagationResult, input: &Input, nranks: usize) {
    time_call!(stochastic_timers::add_acc_pack_updates, {
        if nranks > 1 {
            for buf in send.iter_mut() {buf.clear();}
        }

        for (det, dn) in prop.local {
            add_delta(mc, det, dn);
        }

        if input.write.write_excitation_hist && let Some(hist) = mc.excitation_hist.as_mut() {
            for p in prop.samples {
                hist.add(p);
            }
        }

        if nranks > 1 {
            for up in prop.remote {
                let dest = owner(up.det as usize, mc.walkers.len(), nranks);
                send[dest].push(up);
            }
        }
    })
}

/// Exchange remote spawned walker updates between MPI ranks.
/// # Arguments:
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// - `send`: Per-rank MPI send buffers.
/// - `mpi`: Reusable MPI scratch space.
/// # Returns
/// - `&[PopulationUpdate]`: Walker updates received from remote ranks.
fn exchange_updates<'a>(world: &impl Communicator, send: &[Vec<PopulationUpdate>], mpi: &'a mut MPIScratch) -> &'a [PopulationUpdate] {
    time_call!(stochastic_timers::add_exchange_updates, {
        communicate_spawn_updates(world, send, mpi)
    })
}

/// Unpack received walker updates into the global delta vector.
/// # Arguments:
/// - `mc`: Contains information about the current Monte Carlo state.
/// - `received`: Walker updates received from remote MPI ranks.
/// # Returns
/// - `()`: Adds received walker updates into `mc.delta`.
fn unpack_received_updates(mc: &mut MCState, received: &[PopulationUpdate]) {
    time_call!(stochastic_timers::add_unpack_received_updates, {
        for up in received {
            add_delta(mc, up.det as usize, up.dn);
        }
    })
}

/// Accumulate thread-local updates into the global delta vector and exchange any remote
/// updates across MPI ranks.
/// # Arguments:
/// - `mc`: Contains the current Monte Carlo state.
/// - `send`: Per-rank MPI send buffers.
/// - `prop`: Population updates generated in the spawning and death/cloning step.
/// - `input`: User specified input options.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// - `nranks`: Total number of MPI ranks.
/// # Returns
/// - `i32`: Global indicator for whether any population changes occurred on any rank.
fn accumulate_updates(mc: &mut MCState, send: &mut [Vec<PopulationUpdate>], prop: PropagationResult, 
                      input: &Input, mpi: &mut MPIScratch, world: &impl Communicator, nranks: usize) -> i32 {
    for buf in send.iter_mut() {
        buf.clear();
    }

    acc_pack_updates(mc, send, prop, input, nranks);

    if nranks > 1 {
        for buf in send.iter_mut() {
            if buf.len() > 1 {
                compress_updates(buf);
            }
        }

        let received = exchange_updates(world, send, mpi);
        unpack_received_updates(mc, received);
    }

    let changed = (!mc.changed.is_empty()) as i32;
    let mut changedglobal = 0;
    world.all_reduce_into(&changed, &mut changedglobal, SystemOperation::max());
    changedglobal
}

/// Sort and combine repeated remote population updates in place.
/// # Arguments:
/// - `updates`: Remote population updates for a single destination rank.
/// # Returns
/// - `()`: Compresses repeated determinant updates in place.
fn compress_updates(updates: &mut Vec<PopulationUpdate>) {
    if updates.is_empty() {return;}
    updates.sort_unstable_by_key(|up| up.det);

    let mut out = 0usize;
    for i in 0..updates.len() {
        if out > 0 && updates[out - 1].det == updates[i].det {
            updates[out - 1].dn += updates[i].dn;
        } else {
            updates[out] = updates[i];
            out += 1;
        }
    }

    updates.truncate(out);
    updates.retain(|up| up.dn != 0);
}

/// Compute the current non-overlap transformed and overlap-transformed walker populations.
/// # Arguments:
/// - `mc`: Contains the current Monte Carlo state.
/// - `isref`: Boolean mask specifying which determinants are reference determinants.
/// - `run`: Rank-local run metadata.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// # Returns
/// - `PopulationStats`: Current total and reference populations in both representations.
fn compute_populations(mc: &MCState, isref: &[bool], run: &QMCRunInfo, world: &impl Communicator) -> PopulationStats {
    time_call!(stochastic_timers::add_compute_populations, {

        let nwclocal = mc.walkers.norm();
        let mut nwc = 0i64;
        world.all_reduce_into(&nwclocal, &mut nwc, SystemOperation::sum());

        let nrefc_local: i64 = mc.walkers.occ().iter().filter(|&&det| isref[det]).map(|&det| mc.walkers.get(det).abs()).sum();
        let mut nrefc = 0i64;
        world.all_reduce_into(&nrefc_local, &mut nrefc, SystemOperation::sum());

        let nwsc_local: f64 = mc.pg.iter().map(|x| x.abs()).sum();
        let mut nwsc = 0.0;
        world.all_reduce_into(&nwsc_local, &mut nwsc, SystemOperation::sum());

        let nrefsc_local: f64 = mc.pg.iter().enumerate().filter(|(k, _)| isref[run.start + *k]).map(|(_, x)| x.abs()).sum();
        let mut nrefsc = 0.0;
        world.all_reduce_into(&nrefsc_local, &mut nrefsc, SystemOperation::sum());

        let noccdetslocal = mc.walkers.occ().len() as i64;
        let mut noccdets = 0i64;
        world.all_reduce_into(&noccdetslocal, &mut noccdets, SystemOperation::sum());

        PopulationStats {nwc, nrefc, nwsc, nrefsc, noccdets}
    })
}

/// Cache the latest population statistics inside the propagation state for later
/// printing and possible early exiting.
/// # Arguments:
/// - `state`: Propagation state containing QMC stats.
/// - `stats`: Population statistics computed for the current iteration.
/// # Returns
/// - `()`: Updates cached population values in `state` in place.
fn cache_population_stats(state: &mut PropagationState, stats: &PopulationStats) {
    state.cur_pop.nwc = stats.nwc;
    state.cur_pop.nrefc = stats.nrefc;
    state.cur_pop.nwsc = stats.nwsc;
    state.cur_pop.nrefsc = stats.nrefsc;
    state.cur_pop.noccdets = stats.noccdets;
}

/// Update the shift energies according to the current walker populations and shift.
/// # Arguments:
/// - `it`: Current iteration number.
/// - `stats`: Population statistics computed for the current iteration.
/// - `state`: Propagation state containing QMC stats.
/// - `es`: Non-overlap transformed shift energy.
/// - `input`: User specified input options.
/// # Returns
/// - `()`: Updates the shift energies and associated state variables in place.
fn update_shifts(it: usize, stats: &PopulationStats, state: &mut PropagationState, es: &mut f64, input: &Input) {
    let qmc = input.qmc.as_ref().unwrap();

    if !state.reached_c && stats.nwc > qmc.target_population {
        state.reached_c = true;
        state.prev_pop.nwc = stats.nwc;
    }

    if !state.reached_sc && stats.nwsc > qmc.target_population as f64 {
        state.reached_sc = true;
        state.prev_pop.nwsc = stats.nwsc;
    }

    if state.reached_c && (it + 1).is_multiple_of(qmc.shift_update_freq) {
        *es -= (qmc.shift_damping / (input.prop_ref().dt * (qmc.shift_update_freq as f64))) * (stats.nwc as f64 / state.prev_pop.nwc as f64).ln();
        state.prev_pop.nwc = stats.nwc;
    }

    if state.reached_sc && (it + 1).is_multiple_of(qmc.shift_update_freq) {
        state.es_s -= (qmc.shift_damping / (input.prop_ref().dt * (qmc.shift_update_freq as f64))) * (stats.nwsc / state.prev_pop.nwsc).ln();
        state.prev_pop.nwsc = stats.nwsc;
    }
}

/// Propagate according to the stochastic update equations for `max_steps` iterations.
/// This routine initialises the NOCI-QMC state, performs the parallel spawning,
/// death/cloning, annihilation, population, shift, and projected-energy updates, and
/// optionally writes a restart file if a `STOP` file is detected.
/// # Arguments:
/// - `data`: Immutable stochastic propagation data.
/// - `c0`: Initial determinant coefficient vector to be translated into walker populations.
/// - `es`: Non-overlap transformed shift energy.
/// - `ref_indices`: Indices of the reference determinants in the full basis.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// # Returns
/// - `(f64, Option<ExcitationHist>)`: Final projected energy estimate and optional excitation histogram.
pub fn qmc_step(data: &NOCIData<'_>, c0: &[f64], es: &mut f64, ref_indices: &[usize], world: &impl Communicator) -> (f64, Option<ExcitationHist>) {
    let qmc = data.input.qmc.as_ref().unwrap();

    // Set up data for stochastic run.
    let irank = world.rank() as usize;
    let nranks = world.size() as usize;
    let ndets = data.basis.len();
    let start = (ndets * irank) / nranks;
    let end = (ndets * (irank + 1)) / nranks;
    // Determine which of the determinant indices are reference states.
    let mut isref = vec![false; ndets];
    for &i in ref_indices {
        isref[i] = true;
    }

    let base_seed = qmc.seed.unwrap_or_else(rand::random);
    let rank_seed = base_seed.wrapping_add((irank as u64).wrapping_mul(0x9E3779B9));

    let run = QMCRunInfo {irank, nranks, ndets, start, end, iref: 0, base_seed, rank_seed};

    let scratchsize = {
        let (maxsame, maxla, maxlb) = max_scratch_sizes(data.basis);
        ScratchSize {maxsame, maxla, maxlb}
    };

    let mut scratch = WickScratchSpin::new();
    let mut mpiscratch = MPIScratch::new(run.nranks);

    let mut state = initialise_qmc_state(c0, es, data, &run, &isref, world, &mut scratch, &mut mpiscratch);

    println!("Size of Wick's Scratch (MiB): {}", std::mem::size_of::<WickScratchSpin>() as f64 / (1024.0 * 1024.0));
    type ThreadState = (Vec<(usize, i64)>, Vec<PopulationUpdate>, Vec<f64>, SmallRng, WickScratchSpin);
    println!("Size of per thread state (MiB): {}", std::mem::size_of::<ThreadState>() as f64 / (1024.0 * 1024.0));

    let mut send: Vec<Vec<PopulationUpdate>> = (0..nranks).map(|_| Vec::new()).collect();

    print_header(irank);
    print_initial_row(irank, &state, data.basis[0].e);

    for it in state.start_iter..data.input.prop_ref().max_steps {
        let shifts = Shifts {es: *es, es_s: state.es_s};
        let prop = propagate_iteration(it, &state.mc, data, &run, &scratchsize, shifts);
        
        let changedglobal = accumulate_updates(&mut state.mc, &mut send, prop, data.input, &mut mpiscratch, world, nranks);
        if changedglobal == 0 {
            print_cached_row(irank, it + 1, &state, data.basis[0].e, *es);
            continue;
        }
        
        let d = apply_delta(&mut state.mc);
        update_p(&mut state.mc.pg, &d, data, &run, &mut mpiscratch, world);

        let stats = compute_populations(&state.mc, &isref, &run, world);
        cache_population_stats(&mut state, &stats);
        update_shifts(it, &stats, &mut state, es, data.input);
        
        update_projected_energy(&d, &mut state.pe, data, world);
        state.eprojcur = state.pe.num / state.pe.den;
        
        let stopshifts = Shifts {es: *es, es_s: state.es_s};
        if let Some(ret) = check_stop(it, &mut state, stopshifts, &run, world) {
            return ret;
        }

        print_row(irank, it + 1, &state, &stats, data.basis[0].e, *es);
    }

    (state.eprojcur, state.mc.excitation_hist)
}
