// stochastic/restart.rs
// Standard library imports.
use std::fs;
use std::path::Path;

// External crate imports.
use hdf5::File;
use mpi::topology::Communicator;
use mpi::traits::*;

// Crate-root imports.
use crate::SCFState;

// Parent/sibling imports.
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
    /// Optional current overlap-weight mixture probability.
    pub(in crate::stochastic) overlap_weight: Option<f64>,
    /// Number of determinants in the global stochastic basis.
    pub(in crate::stochastic) ndets: usize,
    /// Deterministic hash of the ordered stochastic determinant basis.
    pub(in crate::stochastic) basis_hash: [u64; 2],
}

/// Build a deterministic hash of the ordered stochastic determinant basis.
/// The hash includes determinant order, parent indices, occupation bitstrings, excitation
/// bitstrings, phase convention, and each parent orbital convention. This is a deterministic
/// compatibility checksum rather than a cryptographic hash.
/// # Arguments:
/// - `basis`: Ordered stochastic determinant basis used by the current executable.
/// # Returns:
/// - `[u64; 2]`: Two-lane deterministic basis hash.
pub(in crate::stochastic) fn basis_hash(basis: &[SCFState]) -> [u64; 2] {
    let mut hash = [0xcbf29ce484222325, 0x84222325cbf29ce4];
    let max_parent = basis.iter().map(|det| det.parent).max().unwrap_or(0);
    let mut seen_parent = vec![false; max_parent + 1];

    let mut mix = |value: u64| {
        hash[0] ^= value;
        hash[0] = hash[0].wrapping_mul(0x00000100000001b3);
        hash[1] ^= value.rotate_left(32);
        hash[1] = hash[1].wrapping_mul(0x00000100000001b3);
    };

    mix(basis.len() as u64);
    for (i, det) in basis.iter().enumerate() {
        mix(i as u64);
        mix(det.parent as u64);
        for value in [det.oa, det.ob] {
            mix(value as u64);
            mix((value >> 64) as u64);
        }
        mix(det.pha.to_bits());
        mix(det.phb.to_bits());
        for value in [
            det.excitation.alpha.holes,
            det.excitation.alpha.parts,
            det.excitation.beta.holes,
            det.excitation.beta.parts,
        ] {
            mix(value as u64);
            mix((value >> 64) as u64);
        }
        if !seen_parent[det.parent] {
            seen_parent[det.parent] = true;
            mix(det.parent as u64);
            mix(det.ca.nrows() as u64);
            mix(det.ca.ncols() as u64);
            for &value in det.ca.iter() {
                mix(value.to_bits());
            }
            mix(det.cb.nrows() as u64);
            mix(det.cb.ncols() as u64);
            for &value in det.cb.iter() {
                mix(value.to_bits());
            }
        }
    }

    hash
}

/// Validate restart metadata against the current stochastic determinant basis.
/// Legacy restarts without schema metadata are accepted because they cannot be validated.
/// # Arguments:
/// - `meta`: HDF5 restart metadata group.
/// - `world`: MPI communicator used to compare the saved MPI rank count.
/// - `expected_ndets`: Current number of global stochastic determinants.
/// - `expected_hash`: Current deterministic basis hash.
/// # Returns:
/// - `()`: Panics if stored metadata is present and incompatible.
fn validate_restart_metadata(
    meta: &hdf5::Group,
    world: &impl Communicator,
    expected_ndets: usize,
    expected_hash: [u64; 2],
) {
    let Ok(schema) = meta.dataset("schema_version") else {
        if world.rank() == 0 {
            println!("Warning: legacy restart has no validation, proceed at own risk.");
        }
        return;
    };

    let schema = schema.read_1d::<u64>().unwrap()[0];
    let nranks = meta.dataset("nranks").unwrap().read_1d::<u64>().unwrap()[0] as usize;
    let ndets = meta.dataset("ndets").unwrap().read_1d::<u64>().unwrap()[0] as usize;
    let hash = meta
        .dataset("basis_hash")
        .unwrap()
        .read_1d::<u64>()
        .unwrap();
    let saved_hash = [hash[0], hash[1]];

    if schema != 1 {
        panic!("Restart schema version mismatch: saved {schema}, current 1.");
    }
    if nranks != world.size() as usize {
        panic!(
            "Restart MPI rank count mismatch: saved {nranks}, current {}.",
            world.size()
        );
    }
    if ndets != expected_ndets {
        panic!("Restart determinant count mismatch: saved {ndets}, current {expected_ndets}.");
    }
    if saved_hash != expected_hash {
        panic!("Restart basis hash mismatch: saved {saved_hash:x?}, current {expected_hash:x?}.");
    }
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

        if let Some(overlap_weight) = state.overlap_weight {
            meta.new_dataset_builder()
                .with_data(&[overlap_weight])
                .create("overlap_weight")?;
        }

        meta.new_dataset_builder()
            .with_data(&[1_u64])
            .create("schema_version")?;

        meta.new_dataset_builder()
            .with_data(&[nranks as u64])
            .create("nranks")?;

        meta.new_dataset_builder()
            .with_data(&[state.ndets as u64])
            .create("ndets")?;

        meta.new_dataset_builder()
            .with_data(&state.basis_hash)
            .create("basis_hash")?;
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

            group
                .new_dataset_builder()
                .with_data(&[state.populations.len() as u64])
                .create("population_len")?;

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
/// - `expected_ndets`: Current number of global stochastic determinants.
/// - `expected_hash`: Current deterministic basis hash.
/// # Returns:
/// - `hdf5::Result<RestartState>`: Rank-local restart state.
pub(in crate::stochastic) fn read_restart_hdf5(
    path: &str,
    world: &impl Communicator,
    expected_ndets: usize,
    expected_hash: [u64; 2],
) -> hdf5::Result<RestartState> {
    let irank = world.rank() as usize;

    let file = File::open(path)?;
    let meta = file.group("meta")?;
    validate_restart_metadata(&meta, world, expected_ndets, expected_hash);

    let report = meta.dataset("report")?.read_1d::<u64>()?[0] as usize;

    let shift = meta.dataset("shift")?.read_1d::<f64>()?[0];

    let nwprev = meta.dataset("nwprev")?.read_1d::<f64>()?[0];

    let nrefprev = meta.dataset("nrefprev")?.read_1d::<f64>()?[0];

    let base_seed = meta
        .dataset("base_seed")
        .ok()
        .map(|dataset| dataset.read_1d::<u64>().unwrap()[0]);

    let overlap_weight = meta
        .dataset("overlap_weight")
        .ok()
        .map(|dataset| dataset.read_1d::<f64>().unwrap()[0]);

    let group = file.group(&format!("rank_{irank:02}"))?;

    let populations = group.dataset("populations")?.read_1d::<f64>()?.to_vec();
    if let Ok(population_len) = group.dataset("population_len") {
        let population_len = population_len.read_1d::<u64>()?[0] as usize;
        if population_len != populations.len() {
            panic!(
                "Restart population length mismatch on rank {irank}: metadata {population_len}, data {}.",
                populations.len()
            );
        }
    }

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
        overlap_weight,
        ndets: expected_ndets,
        basis_hash: expected_hash,
    })
}
