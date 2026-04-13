// stochastic/restart.rs
use std::fs;
use std::path::Path;

use hdf5::File;
use mpi::topology::Communicator;
use mpi::traits::*;

use super::state::{ExcitationHist, Walkers};

pub(crate) struct RestartState {
    pub(crate) iter: usize,
    pub(crate) es: f64,
    pub(crate) es_s: f64,
    pub(crate) nwprevc: i64,
    pub(crate) nrefprevc: i64,
    pub(crate) nwprevsc: f64,
    pub(crate) nrefprevsc: f64,
    pub(crate) walkers: Walkers,
    pub(crate) pg: Vec<f64>,
    pub(crate) excitation_hist: Option<ExcitationHist>,
    pub(crate) base_seed: Option<u64>,
}

pub(crate) fn write_restart_hdf5(path: &str, world: &impl Communicator, state: &RestartState, start: usize, end: usize, ndets: usize) -> hdf5::Result<()> {
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
        meta.new_dataset_builder().with_data(&[state.iter as u64]).create("iter")?;
        meta.new_dataset_builder().with_data(&[ndets as u64]).create("ndets")?;
        meta.new_dataset_builder().with_data(&[nranks as u64]).create("nranks")?;
        meta.new_dataset_builder().with_data(&[state.es]).create("es")?;
        meta.new_dataset_builder().with_data(&[state.es_s]).create("es_s")?;
        meta.new_dataset_builder().with_data(&[state.nwprevc]).create("nwprevc")?;
        meta.new_dataset_builder().with_data(&[state.nrefprevc]).create("nrefprevc")?;
        meta.new_dataset_builder().with_data(&[state.nwprevsc]).create("nwprevsc")?;
        meta.new_dataset_builder().with_data(&[state.nrefprevsc]).create("nrefprevsc")?;
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
            // Store determinant range owned by a given rank
            grp.new_dataset_builder().with_data(&[start as u64]).create("start")?;
            grp.new_dataset_builder().with_data(&[end as u64]).create("end")?;
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

pub(crate) fn read_restart_hdf5(path: &str, world: &impl Communicator, ndets: usize) -> hdf5::Result<RestartState> {
    let irank = world.rank() as usize;
    let file = File::open(path)?;

    let meta = file.group("meta")?;
    let iter = meta.dataset("iter")?.read_1d::<u64>()?[0] as usize;
    let es = meta.dataset("es")?.read_1d::<f64>()?[0];
    let es_s = meta.dataset("es_s")?.read_1d::<f64>()?[0];
    let nwprevc = meta.dataset("nwprevc")?.read_1d::<i64>()?[0];
    let nrefprevc = meta.dataset("nrefprevc")?.read_1d::<i64>()?[0];
    let nwprevsc = meta.dataset("nwprevsc")?.read_1d::<f64>()?[0];
    let nrefprevsc = meta.dataset("nrefprevsc")?.read_1d::<f64>()?[0];
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

    Ok(RestartState {iter, es, es_s, nwprevc, nrefprevc, nwprevsc, nrefprevsc, walkers, pg, excitation_hist, base_seed})
}
