// stochastic/restart.rs
use std::fs;
use std::path::Path;

use hdf5::File;
use mpi::topology::Communicator;
use mpi::traits::*;

use super::state::ExcitationHist;

/// Storage required to resume stochastic propagation.
pub(in crate::stochastic) struct RestartState {
    /// Report at which the restart was written.
    pub(in crate::stochastic) report: usize,
    /// Current population-control shift.
    pub(in crate::stochastic) shift: f64,
    /// Persistent population at the previous shift update.
    pub(in crate::stochastic) nwprev: f64,
    /// Persistent reference population at the previous shift update.
    pub(in crate::stochastic) nrefprev: f64,
    /// Rank-local persistent real populations.
    pub(in crate::stochastic) populations: Vec<f64>,
    /// Optional excitation histogram.
    pub(in crate::stochastic) excitation_hist: Option<ExcitationHist>,
    /// Optional base RNG seed.
    pub(in crate::stochastic) base_seed: Option<u64>,
}

/// Write a restart file containing the current stochastic propagation state.
/// # Arguments:
/// - `path`: Path of the HDF5 restart file.
/// - `world`: MPI communicator.
/// - `state`: Restart state to write.
/// # Returns:
/// - `hdf5::Result<()>`: Result of writing the restart file.
pub(in crate::stochastic) fn write_restart_hdf5(
    path: &str,
    world: &impl Communicator,
    state: &RestartState,
) -> hdf5::Result<()> {
    let irank = world.rank() as usize;
    let nranks = world.size() as usize;

    if irank == 0 {
        if let Some(parent) = Path::new(path).parent()
            && !parent.as_os_str().is_empty()
        {
            let _ = fs::create_dir_all(parent);
        }

        let file = File::create(path)?;
        let meta = file.create_group("meta")?;

        meta.new_dataset_builder()
            .with_data(&[state.report as u64])
            .create("report")?;

        meta.new_dataset_builder()
            .with_data(&[state.shift])
            .create("shift")?;

        meta.new_dataset_builder()
            .with_data(&[state.nwprev])
            .create("nwprev")?;

        meta.new_dataset_builder()
            .with_data(&[state.nrefprev])
            .create("nrefprev")?;

        if let Some(seed) = state.base_seed {
            meta.new_dataset_builder()
                .with_data(&[seed])
                .create("base_seed")?;
        }
    }

    world.barrier();

    for rank in 0..nranks {
        if irank == rank {
            let file = File::open_rw(path)?;
            let group = file.create_group(&format!("rank_{irank:02}"))?;

            group
                .new_dataset_builder()
                .with_data(&state.populations)
                .create("populations")?;

            if let Some(hist) = &state.excitation_hist {
                let h = group.create_group("excitation_hist")?;

                h.new_dataset_builder()
                    .with_data(&[hist.logmin])
                    .create("logmin")?;

                h.new_dataset_builder()
                    .with_data(&[hist.logmax])
                    .create("logmax")?;

                h.new_dataset_builder()
                    .with_data(&[hist.noverflow_low])
                    .create("noverflow_low")?;

                h.new_dataset_builder()
                    .with_data(&[hist.noverflow_high])
                    .create("noverflow_high")?;

                h.new_dataset_builder()
                    .with_data(&[hist.nbins as u64])
                    .create("nbins")?;

                h.new_dataset_builder()
                    .with_data(&[hist.ntotal])
                    .create("ntotal")?;

                h.new_dataset_builder()
                    .with_data(&hist.counts)
                    .create("counts")?;
            }
        }

        world.barrier();
    }

    Ok(())
}

/// Read a restart file and reconstruct the rank-local propagation state.
/// # Arguments:
/// - `path`: Path to the HDF5 restart file.
/// - `world`: MPI communicator.
/// # Returns:
/// - `hdf5::Result<RestartState>`: Rank-local restart state.
pub(in crate::stochastic) fn read_restart_hdf5(
    path: &str,
    world: &impl Communicator,
) -> hdf5::Result<RestartState> {
    let irank = world.rank() as usize;

    let file = File::open(path)?;
    let meta = file.group("meta")?;

    let report = meta.dataset("report")?.read_1d::<u64>()?[0] as usize;

    let shift = meta.dataset("shift")?.read_1d::<f64>()?[0];

    let nwprev = meta.dataset("nwprev")?.read_1d::<f64>()?[0];

    let nrefprev = meta.dataset("nrefprev")?.read_1d::<f64>()?[0];

    let base_seed = meta
        .dataset("base_seed")
        .ok()
        .map(|dataset| dataset.read_1d::<u64>().unwrap()[0]);

    let group = file.group(&format!("rank_{irank:02}"))?;

    let populations = group.dataset("populations")?.read_1d::<f64>()?.to_vec();

    let excitation_hist = if let Ok(h) = group.group("excitation_hist") {
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

    Ok(RestartState {
        report,
        shift,
        nwprev,
        nrefprev,
        populations,
        excitation_hist,
        base_seed,
    })
}
