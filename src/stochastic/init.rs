// stochastic/init.rs
use mpi::topology::Communicator;
use mpi::collective::SystemOperation;
use mpi::traits::*;
use rayon::prelude::*;

use crate::noci::NOCIData;
use crate::SCFState;
use crate::nonorthogonalwicks::WickScratchSpin;
use crate::time_call;
use crate::timers::stochastic as stochastic_timers;
use super::state::{MCState, Walkers, PopulationUpdate, ProjectedEnergyUpdate, PropagationState, QMCRunInfo, ExcitationHist, PopulationStats, MPIScratch};

use crate::mpiutils::{gather_all_walkers, local_walkers};
use super::propagate::{find_s, find_hs};
use super::restart::read_restart_hdf5;

/// For each entry in initial coefficient vector c0 calculate `(c0_i / ||c||) * N_0` and assign
/// this value as the initial walker population on determinant `i`. The sign of the population is
/// given by the sign of `c0_i`.
/// # Arguments:
/// - `c0`: Initial determinant coefficient vector.
/// - `init_pop`: Number of walkers to start with.
/// - `n`: Number of determinants.
/// - `data`: Immutable stochastic propagation data.
/// - `iref`: Index of the projected reference determinant.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `Walkers`: Initial walker population.
pub(in crate::stochastic) fn initialise_walkers(c0: &[f64], init_pop: i64, n: usize, data: &NOCIData<'_>, iref: usize, scratch: &mut WickScratchSpin) -> Walkers {
    time_call!(stochastic_timers::add_initialise_walkers, {
        let mut w = Walkers::new(n);

        // Ill-conditioning threshold.
        let threshold = 1e-6; 

        // Calculate 1-norm of initial coefficient vector.
        let norm1: f64 = c0.iter().map(|x| x.abs()).sum::<f64>();

        // Assign initial populations based on c0.
        for (i, &ci) in c0.iter().enumerate() {
            let ni = ((ci.abs() / norm1) * (init_pop as f64)).round() as i64;
            // Decide sign when not zero.
            if ni != 0 {
                let sgn = if ci >= 0.0 {1} else {-1};
                w.add(i, sgn * ni);
            }
        }
        
        // Calculate projected energy denominator \Sum_{\Gamma} N_{\Gamma} S_{\Gamma, iref}. If this
        // quantity is very small we have large amounts of overcompleteness and therefore
        // ill-conditioning in the NOCI-QMC overlap matrix.
        let mut den: f64 = 0.0; 
        for &gamma in w.occ() {
            let ngamma = w.get(gamma);
            // Calculate matrix element H_{\Gamma, \text{Reference}}, S_{\Gamma,\text{Reference}} 
            let sgr = find_s(data, gamma, iref, scratch);
            den += (ngamma as f64) * sgr;
        }

        // If the NOCI-QMC overlap matrix is very ill-conditioned starting with all walkers on the RHF
        // reference rather than distributed according to c0 can prevent initial blow-ups in the
        // projected energy.
        if den.abs() < threshold {
            println!("NOCI-QMC overlap very ill-conditioned. Starting from reference index 0 (Denominator: {})", den);
            let mut w0 = Walkers::new(n);
            w0.add(0, init_pop);
            return w0;
        }
        w
    })
}

/// Initialise the running projected-energy state
/// `\frac{\sum_{\Gamma} N_{\Gamma} H_{\Gamma,\mathrm{ref}}}{\sum_{\Gamma} N_{\Gamma} S_{\Gamma,\mathrm{ref}}}`
/// by performing the full initial sweep over the occupied determinant space.
/// # Arguments:
/// - `walkers`: Object containing information about current walker distribution.
/// - `iref`: Index of the determinant onto which the energy is projected.
/// - `data`: Immutable stochastic propagation data.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// # Returns
/// - `ProjectedEnergyUpdate`: Initial projected-energy numerator and denominator.
pub(in crate::stochastic) fn init_projected_energy(walkers: &Walkers, iref: usize, data: &NOCIData<'_>, world: &impl Communicator) -> ProjectedEnergyUpdate {
    let (num_local, den_local) = walkers.occ().par_iter().fold(|| (0.0_f64, 0.0_f64, WickScratchSpin::new()), |(mut num, mut den, mut scratch), &gamma| {
        let ngamma = walkers.get(gamma) as f64;
        let (hgr, sgr) = find_hs(data, gamma, iref, &mut scratch);
        num += ngamma * hgr;
        den += ngamma * sgr;
        (num, den, scratch)
    }).map(|(num, den, _)| (num, den)).reduce(|| (0.0, 0.0), |(a_num, a_den), (b_num, b_den)| (a_num + b_num, a_den + b_den));

    let mut num = 0.0;
    let mut den = 0.0;
    world.all_reduce_into(&num_local, &mut num, SystemOperation::sum());
    world.all_reduce_into(&den_local, &mut den, SystemOperation::sum());

    ProjectedEnergyUpdate {iref, num, den}
}

/// Initialise the local portion of `p_{\Gamma} = \sum_{\Omega} S_{\Gamma\Omega} N_{\Omega}`.
/// # Arguments:
/// - `walkers`: Rank-local walker populations.
/// - `data`: Immutable stochastic propagation data.
/// - `run`: Rank-local run metadata.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// - `scratch`: Scratch space for Wick's quantities.
/// - `mpiscratch`: Reusable MPI scratch space.
/// # Returns
/// - `Vec<f64>`: Local portion of the overlap-transformed population vector `p_{\Gamma}`.
pub(in crate::stochastic) fn init_p(walkers: &Walkers, data: &NOCIData<'_>, run: &QMCRunInfo, world: &impl Communicator, 
                                    scratch: &mut WickScratchSpin, mpiscratch: &mut MPIScratch) -> Vec<f64> {
    let local: Vec<PopulationUpdate> = walkers.occ().iter().map(|&i| PopulationUpdate {det: i as u64, dn: walkers.get(i)}).collect();
    let global = gather_all_walkers(world, &local, mpiscratch);

    let mut p = vec![0.0; run.end - run.start];
    for gamma in run.start..run.end {
        //p_{\Gamma} = \sum_\Omega S_{\Gamma, \Omega} N_{\Omega}.
        let mut pgamma = 0.0;
        for entry in global {
            // Here we use dn to mean total population.
            let nomega = entry.dn as f64;
            let omega = entry.det as usize;
            let sgo = find_s(data, gamma, omega, scratch);
            pgamma += nomega * sgo;
        }
        p[gamma - run.start] = pgamma
    }
    p
}

/// Determine the maximum scratch sizes required for computation of matrix elements using extended
/// non-orthogonal Wick's theorem depending on the maximum excitation rank present in the basis.
/// # Arguments:
/// - `basis`: Full list of the NOCI-QMC basis.
/// # Returns
/// - `(usize, usize, usize)`: Maximum same-spin scratch size, alpha excitation size, and beta
///   excitation size.
pub(in crate::stochastic) fn max_scratch_sizes(basis: &[SCFState]) -> (usize, usize, usize) {
    let maxexa = basis.iter().map(|st| st.excitation.alpha.holes.len()).max().unwrap_or(0);
    let maxexb = basis.iter().map(|st| st.excitation.beta.holes.len()).max().unwrap_or(0);
    let maxsame = 2 * maxexa.max(maxexb);
    let maxla = 2 * maxexa;
    let maxlb = 2 * maxexb;
    (maxsame, maxla, maxlb)
}

/// Initialise projected-energy, walker distribution, `p_{\Gamma}` vector, and population totals
/// across ranks. Initialisation occurs either from an initial coefficient vector `c0` or from a
/// restart file.
/// # Arguments:
/// - `c0`: Initial determinant coefficient vector.
/// - `es`: Non-overlap transformed shift energy.
/// - `data`: Immutable stochastic propagation data.
/// - `run`: Rank-local run metadata.
/// - `isref`: Boolean mask specifying which determinants are reference determinants.
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// - `scratch`: Scratch space for Wick's quantities.
/// - `mpiscratch`: Reusable MPI scratch space.
/// # Returns
/// - `PropagationState`: Initialised NOCI-QMC state with required bookkeeping parameters.
pub(in crate::stochastic) fn initialise_qmc_state(c0: &[f64], es: &mut f64, data: &NOCIData<'_>, run: &QMCRunInfo, isref: &[bool], 
                                                  world: &impl Communicator, scratch: &mut WickScratchSpin, mpiscratch: &mut MPIScratch) -> PropagationState {

    let qmc = data.input.qmc.as_ref().unwrap();
    // Use restart file if avaliable.
    if let Some(path) = data.input.write.read_restart.as_deref() {
        if run.irank == 0 {
            println!("Reading restart from {path}");
        }

        let rs = read_restart_hdf5(path, world, run.ndets).unwrap();
    
        // Read shifts and populations.
        *es = rs.es;
        let es_s = rs.es_s;

        let mc = MCState {
            walkers: rs.walkers, 
            delta: vec![0; run.ndets], 
            changed: Vec::new(), 
            pg: rs.pg,
            excitation_hist: rs.excitation_hist
        };
        
        // Initialise projected-energy.
        let pe = init_projected_energy(&mc.walkers, run.iref, data, world);
        
        let reached_c = rs.nwprevc >= qmc.target_population;
        let reached_sc = rs.nwprevsc >= qmc.target_population as f64;
        let start_iter = rs.iter + 1;
        let prev_pop = PopulationStats::new(rs.nwprevc, rs.nrefprevc, rs.nwprevsc, rs.nrefprevsc, rs.noccdets);
        PropagationState::restart(mc, pe, es_s, start_iter, reached_sc, reached_c, prev_pop)
    } else {
        if run.irank == 0 {
            println!("Initialising walkers.....");
        }
        
        // Initialise walker populations from initial coefficient vector.
        let w = initialise_walkers(c0, qmc.initial_population, run.ndets, data, run.iref, scratch);
        let w = local_walkers(w, run.irank, run.nranks);

        let excitation_hist = if data.input.write.write_excitation_hist {
            Some(ExcitationHist::new(-60.0, 1e-12, 100))
        } else {
            None
        };

        let pg = init_p(&w, data, run, world, scratch, mpiscratch);
        let mc = MCState {
            walkers: w, 
            delta: vec![0; run.ndets], 
            changed: Vec::new(), 
            pg,
            excitation_hist
        };
        
        // Initialise projected-energy energy.
        let pe = init_projected_energy(&mc.walkers, run.iref, data, world);
        
        // Initialise overlap-transformed walker populations.
        let mut nwprevsc = 0.0;
        let nwprevsc_local: f64 = mc.pg.iter().map(|x| x.abs()).sum();
        world.all_reduce_into(&nwprevsc_local, &mut nwprevsc, SystemOperation::sum());
        let mut nrefprevsc = 0.0;
        let nrefprevsc_local: f64 = mc.pg.iter().enumerate().filter(|(k, _)| isref[run.start + *k]).map(|(_, x)| x.abs()).sum();
        world.all_reduce_into(&nrefprevsc_local, &mut nrefprevsc, SystemOperation::sum());
        
        // Initialise non-overlap-transformed walker populations.
        let nwprevc = qmc.initial_population;
        let mut nrefprevc = 0i64;
        let nrefprevc_local: i64 = mc.walkers.occ().iter().filter(|&&det| isref[det]).map(|&det| mc.walkers.get(det).abs()).sum();
        world.all_reduce_into(&nrefprevc_local, &mut nrefprevc, SystemOperation::sum());
        
        // Initialise number of occupied determinants.
        let noccdets_local = mc.walkers.occ().len() as i64;
        let mut noccdets = 0i64;
        world.all_reduce_into(&noccdets_local, &mut noccdets, SystemOperation::sum());

        let prev_pop = PopulationStats::new(nwprevc, nrefprevc, nwprevsc, nrefprevsc, noccdets);
        PropagationState::fresh(mc, pe, *es, prev_pop)
    }
}

