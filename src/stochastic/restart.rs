// stochastic/restart.rs
use std::fs;
use std::path::Path;

use hdf5::File;
use mpi::topology::Communicator;
use mpi::traits::*;

use super::state::{ExcitationHist, Walkers};

/// Storage for all data required to resume a stochastic run.
pub(in crate::stochastic) struct RestartState {
    /// Report at which the restart file was written.
    pub(in crate::stochastic) report: usize,
    /// Current non-overlap-transformed shift.
    pub(in crate::stochastic) es: f64,
    /// Current overlap-transformed shift.
    pub(in crate::stochastic) es_s: f64,
    /// Raw walker population stored at the previous shift update.
    pub(in crate::stochastic) nwprevc: i64,
    /// Raw reference walker population stored at the previous shift update.
    pub(in crate::stochastic) nrefprevc: i64,
    /// Overlap-transformed walker population stored at the previous shift update.
    pub(in crate::stochastic) nwprevsc: f64,
    /// Overlap-transformed reference walker population stored at the previous shift update.
    pub(in crate::stochastic) nrefprevsc: f64,
    /// Walker distribution at the moment the restart file was written.
    pub(in crate::stochastic) walkers: Walkers,
    // Number of occupied determinants at the moment the restart file was written.
    pub(in crate::stochastic) noccdets: i64,
    /// Local portion of the overlap-transformed population vector p.
    pub(in crate::stochastic) pg: Vec<f64>,
    /// Optional excitation histogram accumulated so far.
    pub(in crate::stochastic) excitation_hist: Option<ExcitationHist>,
    /// Optional base RNG seed used to initialise the stochastic run.
    pub(in crate::stochastic) base_seed: Option<u64>,
}

/// Write a restart file containing the current stochastic propagation state.
/// # Arguments:
/// - `path`: Path of the HDF5 restart file to create or overwrite.
/// - `world`: MPI communicator object.
/// - `state`: Restart state to be written to disk.
/// - `ndets`: Total number of determinants in the stochastic basis.
/// # Returns
/// - `hdf5::Result<()>`: `Ok(())` if the restart file was written successfully.
pub(in crate::stochastic) fn write_restart_hdf5(path: &str, world: &impl Communicator, state: &RestartState, ndets: usize) -> hdf5::Result<()> {
    let irank = world.rank() as usize;
    let nranks = world.size() as usize;

    if irank == 0 {
        // Ensure a parent directory exists and it it does not, create it.
        if let Some(parent) = Path::new(path).parent() && !parent.as_os_str().is_empty() {
            let _ = fs::create_dir_all(parent);
        }
        // Create HDF5 file.
        let file = File::create(path)?;
        // Create and write metadata group.
        let meta = file.create_group("meta")?;
        meta.new_dataset_builder().with_data(&[state.report as u64]).create("report")?;
        meta.new_dataset_builder().with_data(&[ndets as u64]).create("ndets")?;
        meta.new_dataset_builder().with_data(&[nranks as u64]).create("nranks")?;
        meta.new_dataset_builder().with_data(&[state.es]).create("es")?;
        meta.new_dataset_builder().with_data(&[state.es_s]).create("es_s")?;
        meta.new_dataset_builder().with_data(&[state.nwprevc]).create("nwprevc")?;
        meta.new_dataset_builder().with_data(&[state.nrefprevc]).create("nrefprevc")?;
        meta.new_dataset_builder().with_data(&[state.nwprevsc]).create("nwprevsc")?;
        meta.new_dataset_builder().with_data(&[state.nrefprevsc]).create("nrefprevsc")?;
        meta.new_dataset_builder().with_data(&[state.noccdets]).create("noccdets")?;
        // If there was a user provided RNG seed reuse it.
        if let Some(seed) = state.base_seed {
            meta.new_dataset_builder().with_data(&[seed]).create("base_seed")?;
        }
    }

    world.barrier();
    
    // Ensure each MPI rank writes its data sequentially to avoid conflict.
    for r in 0..nranks {
        if irank == r {
            let file = File::open_rw(path)?;
            // Create per MPI rank group for rank local data.
            let grp = file.create_group(&format!("rank_{irank:02}"))?;
            // Store walker distribution.
            let occ: Vec<u64> = state.walkers.occ().iter().map(|&i| i as u64).collect();
            let pop: Vec<i64> = state.walkers.occ().iter().map(|&i| state.walkers.get(i)).collect();
            grp.new_dataset_builder().with_data(&occ).create("occ")?;
            grp.new_dataset_builder().with_data(&pop).create("pop")?;
            grp.new_dataset_builder().with_data(&state.pg).create("pg")?;
            // If the excitation histogram is being recorded store this data also.
            if let Some(hist) = &state.excitation_hist {
                let h = grp.create_group("excitation_hist")?;
                h.new_dataset_builder().with_data(&[hist.logmin]).create("logmin")?;
                h.new_dataset_builder().with_data(&[hist.logmax]).create("logmax")?;
                h.new_dataset_builder().with_data(&[hist.noverflow_low]).create("noverflow_low")?;
                h.new_dataset_builder().with_data(&[hist.noverflow_high]).create("noverflow_high")?;
                h.new_dataset_builder().with_data(&[hist.nbins as u64]).create("nbins")?;
                h.new_dataset_builder().with_data(&[hist.ntotal]).create("ntotal")?;
                h.new_dataset_builder().with_data(&hist.counts).create("counts")?;
            }
        }
        world.barrier();
    }
    Ok(())
}

/// Read a restart file and reconstruct the rank-local stochastic propagation state.
/// # Arguments:
/// - `path`: Path to the HDF5 restart file.
/// - `world`: MPI communicator object.
/// - `ndets`: Total number of determinants in the stochastic basis.
/// # Returns
/// - `hdf5::Result<RestartState>`: Restart state reconstructed for the current rank.
pub(in crate::stochastic) fn read_restart_hdf5(path: &str, world: &impl Communicator, ndets: usize) -> hdf5::Result<RestartState> {
    let irank = world.rank() as usize;
    let file = File::open(path)?;

    let meta = file.group("meta")?;
    let report = meta.dataset("report")?.read_1d::<u64>()?[0] as usize;
    let es = meta.dataset("es")?.read_1d::<f64>()?[0];
    let es_s = meta.dataset("es_s")?.read_1d::<f64>()?[0];
    let nwprevc = meta.dataset("nwprevc")?.read_1d::<i64>()?[0];
    let nrefprevc = meta.dataset("nrefprevc")?.read_1d::<i64>()?[0];
    let nwprevsc = meta.dataset("nwprevsc")?.read_1d::<f64>()?[0];
    let nrefprevsc = meta.dataset("nrefprevsc")?.read_1d::<f64>()?[0];
    let noccdets = meta.dataset("noccdets").map(|d| d.read_1d::<i64>().unwrap()[0]).unwrap_or(0);
    let base_seed = meta.dataset("base_seed").ok().map(|d| d.read_1d::<u64>().unwrap()[0]);

    let grp = file.group(&format!("rank_{irank:02}"))?;
    let occ = grp.dataset("occ")?.read_1d::<u64>()?;
    let pop = grp.dataset("pop")?.read_1d::<i64>()?;
    let pg = grp.dataset("pg")?.read_1d::<f64>()?.to_vec();

    let mut walkers = Walkers::new(ndets);
    for (&i, &n) in occ.iter().zip(pop.iter()) {
        walkers.add(i as usize, n);
    }

    let excitation_hist = if let Ok(h) = grp.group("excitation_hist") {
        let logmin = h.dataset("logmin")?.read_1d::<f64>()?[0];
        let logmax = h.dataset("logmax")?.read_1d::<f64>()?[0];
        let nbins = h.dataset("nbins")?.read_1d::<u64>()?[0] as usize;
        let mut hist = ExcitationHist::new(logmin, logmax, nbins);
        hist.noverflow_low = h.dataset("noverflow_low")?.read_1d::<u64>()?[0];
        hist.noverflow_high = h.dataset("noverflow_high")?.read_1d::<u64>()?[0];
        hist.ntotal = h.dataset("ntotal")?.read_1d::<u64>()?[0];
        hist.counts = h.dataset("counts")?.read_1d::<u64>()?.to_vec();
        Some(hist)
    } else {
        None
    };

    Ok(RestartState {report, es, es_s, nwprevc, nrefprevc, nwprevsc, nrefprevsc, walkers, noccdets, pg, excitation_hist, base_seed})
}
