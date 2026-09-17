// stochastic/restart.rs
// Standard library imports.
use std::fs;
use std::path::Path;

// External crate imports.
use hdf5::File;
use mpi::topology::Communicator;
use mpi::traits::*;

// Crate-root imports.
use crate::input::Propagator;
use crate::noci::{NOCIIndex, NOCISpace};

// Parent/sibling imports.
use super::state::{ExcitationHist, PopulationRepresentation};

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
    /// Sampled population at the previous shift update.
    pub(in crate::stochastic) nsampledprev: f64,
    /// Number of sampled-population determinants at the previous shift update.
    pub(in crate::stochastic) nsampledoprev: i64,
    /// Rank-local persistent real populations.
    pub(in crate::stochastic) populations: Vec<f64>,
    /// Optional report-level heavy-ball velocity in the rank-local population layout.
    pub(in crate::stochastic) momentum: Option<Vec<f64>>,
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
    /// Population representation, absent in legacy restart files.
    pub(in crate::stochastic) representation: Option<PopulationRepresentation>,
    /// Whether population control had activated, absent in legacy restart files.
    pub(in crate::stochastic) reached: Option<bool>,
    /// Target population used when restart was written, absent in legacy files.
    pub(in crate::stochastic) target_population: Option<f64>,
}

/// Map propagator choice to persisted population representation.
/// # Arguments:
/// - `propagator`: Stochastic propagator selected by input.
/// # Returns:
/// - `PopulationRepresentation`: Representation stored in restart metadata.
pub(in crate::stochastic) fn population_representation(
    propagator: Propagator
) -> PopulationRepresentation {
    match propagator {
        Propagator::SApply | Propagator::BApply => PopulationRepresentation::Range,
        _ => PopulationRepresentation::Coefficient,
    }
}

/// Read restart validation metadata needed before constructing RNG streams.
pub(in crate::stochastic) fn restart_base_seed(
    path: &str,
    world: &impl Communicator,
    expected_ndets: usize,
    expected_hash: [u64; 2],
    expected_representation: PopulationRepresentation,
) -> Option<u64> {
    let file = File::open(path).unwrap();
    let meta = file.group("meta").unwrap();
    validate_restart_metadata(
        &meta,
        world,
        expected_ndets,
        expected_hash,
        expected_representation,
    );
    meta.dataset("base_seed")
        .ok()
        .map(|dataset| dataset.read_1d::<u64>().unwrap()[0])
}

/// Build a deterministic hash of the ordered stochastic determinant basis.
/// The hash includes determinant order, parent indices, occupation bitstrings, excitation
/// bitstrings, phase convention, and each parent orbital convention. This is a deterministic
/// compatibility checksum rather than a cryptographic hash.
/// # Arguments:
/// - `basis`: Ordered stochastic determinant basis used by the current executable.
/// # Returns:
/// - `[u64; 2]`: Two-lane deterministic basis hash.
pub(in crate::stochastic) fn basis_hash(space: &NOCISpace<f64>) -> [u64; 2] {
    let mut hash = [0xcbf29ce484222325, 0x84222325cbf29ce4];
    let mut seen_parent = vec![false; space.parents.len()];

    let mut mix = |value: u64| {
        hash[0] ^= value;
        hash[0] = hash[0].wrapping_mul(0x00000100000001b3);
        hash[1] ^= value.rotate_left(32);
        hash[1] = hash[1].wrapping_mul(0x00000100000001b3);
    };

    mix(space.len() as u64);
    for i in 0..space.len() {
        let index = NOCIIndex(i);
        let det = space.state(index);
        let alpha = space.alpha(index);
        let beta = space.beta(index);

        mix(i as u64);
        mix(det.parent as u64);
        for value in [alpha.occupation, beta.occupation] {
            mix(value as u64);
            mix((value >> 64) as u64);
        }
        mix(alpha.reduced.phase.to_bits());
        mix(beta.reduced.phase.to_bits());
        for value in [
            alpha.excitation.holes,
            alpha.excitation.parts,
            beta.excitation.holes,
            beta.excitation.parts,
        ] {
            mix(value as u64);
            mix((value >> 64) as u64);
        }
        if !seen_parent[det.parent] {
            seen_parent[det.parent] = true;
            let parent = &space.parents[det.parent];
            mix(det.parent as u64);
            mix(parent.ca.nrows() as u64);
            mix(parent.ca.ncols() as u64);
            for &value in parent.ca.iter() {
                mix(value.to_bits());
            }
            mix(parent.cb.nrows() as u64);
            mix(parent.cb.ncols() as u64);
            for &value in parent.cb.iter() {
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
    expected_representation: PopulationRepresentation,
) {
    let Ok(schema) = meta.dataset("schema_version") else {
        if world.rank() == 0 {
            println!("Warning: legacy restart has no metadata validation; proceed at own risk.");
        }
        return;
    };

    let schema = schema.read_1d::<u64>().unwrap()[0];
    if schema == 1 {
        if world.rank() == 0 {
            println!(
                "Warning: old restart schema has no full metadata validation; proceed at own risk."
            );
        }
    } else if schema != 2 {
        panic!("Restart schema version mismatch: saved {schema}, current 2.");
    }
    let nranks = meta.dataset("nranks").unwrap().read_1d::<u64>().unwrap()[0] as usize;
    let ndets = meta.dataset("ndets").unwrap().read_1d::<u64>().unwrap()[0] as usize;
    let hash = meta
        .dataset("basis_hash")
        .unwrap()
        .read_1d::<u64>()
        .unwrap();
    let saved_hash = [hash[0], hash[1]];

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

    if schema == 2 {
        let dataset = meta
            .dataset("population_representation")
            .unwrap_or_else(|_| panic!("Restart schema-2 file lacks population representation."));
        let saved = dataset.read_1d::<u8>().unwrap();
        let saved = std::str::from_utf8(saved.as_slice().unwrap()).unwrap_or("invalid");
        if saved != expected_representation.as_str() {
            panic!(
                "Restart population representation mismatch: saved {saved}, current {}.",
                expected_representation.as_str()
            );
        }
    }
}

/// Write a restart file containing the current stochastic propagation state.
/// # Arguments:
/// - `path`: Path of the HDF5 restart file.
/// - `world`: MPI communicator.
/// - `state`: Restart state to write.
/// # Returns:
/// - `hdf5::Result<()>`: Result of writing the restart file.
/// # Errors
/// - Returns an HDF5 error if the file, groups, or datasets cannot be created or written.
pub(in crate::stochastic) fn write_restart_hdf5(
    path: &str,
    world: &impl Communicator,
    state: &RestartState,
) -> hdf5::Result<()> {
    // Resolve the rank topology used for root metadata and serialized rank-local writes.
    let irank = world.rank() as usize;
    let nranks = world.size() as usize;

    // Rank zero creates the file and writes propagation-wide restart metadata.
    if irank == 0 {
        // Create a configured parent directory when the restart path is nested.
        if let Some(parent) = Path::new(path).parent()
            && !parent.as_os_str().is_empty()
        {
            let _ = fs::create_dir_all(parent);
        }

        let file = File::create(path)?;
        let meta = file.create_group("meta")?;

        // Store the report, shift, and previous population-controller observables.
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

        meta.new_dataset_builder()
            .with_data(&[state.nsampledprev])
            .create("nsampledprev")?;

        meta.new_dataset_builder()
            .with_data(&[state.nsampledoprev])
            .create("nsampledoprev")?;

        // Preserve optional controls used by reproducible and overlap-weighted propagation.
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

        // Record schema and basis invariants required to reject incompatible restarts.
        meta.new_dataset_builder()
            .with_data(&[2_u64])
            .create("schema_version")?;

        meta.new_dataset_builder()
            .with_data(state.representation.unwrap().as_str().as_bytes())
            .create("population_representation")?;

        meta.new_dataset_builder()
            .with_data(&[u8::from(state.reached.unwrap_or(false))])
            .create("reached")?;

        meta.new_dataset_builder()
            .with_data(&[state.target_population.unwrap()])
            .create("target_population")?;

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

    // Ensure the file and global metadata exist before any rank opens its local group.
    world.barrier();

    // Serialize rank-local writes because this HDF5 file is not opened with parallel I/O.
    for rank in 0..nranks {
        if irank == rank {
            let file = File::open_rw(path)?;
            let group = file.create_group(&format!("rank_{irank:02}"))?;

            // Store the owned population shard and an explicit consistency length.
            group
                .new_dataset_builder()
                .with_data(&state.populations)
                .create("populations")?;

            group
                .new_dataset_builder()
                .with_data(&[state.populations.len() as u64])
                .create("population_len")?;

            // Store BApply report-level heavy-ball state only when momentum is active.
            if let Some(momentum) = &state.momentum {
                group
                    .new_dataset_builder()
                    .with_data(momentum)
                    .create("momentum")?;
            }

            // Persist the optional spawning histogram with all binning metadata.
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

        // Hand file ownership to the next rank only after the current handle is dropped.
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
/// - `expected_representation`: Population convention required by the active propagator.
/// # Returns:
/// - `hdf5::Result<RestartState>`: Rank-local restart state.
/// # Errors
/// - Returns an HDF5 error if required restart groups or datasets cannot be opened or read.
pub(in crate::stochastic) fn read_restart_hdf5(
    path: &str,
    world: &impl Communicator,
    expected_ndets: usize,
    expected_hash: [u64; 2],
    expected_representation: PopulationRepresentation,
) -> hdf5::Result<RestartState> {
    // Open the shared file and reject incompatible schema, basis, or population conventions.
    let irank = world.rank() as usize;

    let file = File::open(path)?;
    let meta = file.group("meta")?;
    validate_restart_metadata(
        &meta,
        world,
        expected_ndets,
        expected_hash,
        expected_representation,
    );

    // Restore required propagation-controller scalars.
    let report = meta.dataset("report")?.read_1d::<u64>()?[0] as usize;

    let shift = meta.dataset("shift")?.read_1d::<f64>()?[0];

    let nwprev = meta.dataset("nwprev")?.read_1d::<f64>()?[0];

    let nrefprev = meta.dataset("nrefprev")?.read_1d::<f64>()?[0];

    // Read fields added after the original schema with legacy-compatible defaults.
    let nsampledprev = meta
        .dataset("nsampledprev")
        .ok()
        .map(|dataset| dataset.read_1d::<f64>().unwrap()[0])
        .unwrap_or_else(|| {
            if world.rank() == 0 {
                println!("Warning: restart lacks saved sampled population; using zero.");
            }
            0.0
        });

    let nsampledoprev = meta
        .dataset("nsampledoprev")
        .ok()
        .map(|dataset| dataset.read_1d::<i64>().unwrap()[0])
        .unwrap_or(0);

    let base_seed = meta
        .dataset("base_seed")
        .ok()
        .map(|dataset| dataset.read_1d::<u64>().unwrap()[0]);

    let overlap_weight = meta
        .dataset("overlap_weight")
        .ok()
        .map(|dataset| dataset.read_1d::<f64>().unwrap()[0]);

    // Decode optional controller and population-representation metadata.
    let representation = meta
        .dataset("population_representation")
        .ok()
        .map(|dataset| {
            let bytes = dataset.read_1d::<u8>().unwrap();
            match std::str::from_utf8(bytes.as_slice().unwrap()).unwrap_or("") {
                "coefficient" => PopulationRepresentation::Coefficient,
                "range" => PopulationRepresentation::Range,
                value => panic!("Invalid restart population representation {value:?}."),
            }
        });
    let reached = meta
        .dataset("reached")
        .ok()
        .map(|dataset| dataset.read_1d::<u8>().unwrap()[0] != 0);
    let target_population = meta
        .dataset("target_population")
        .ok()
        .map(|dataset| dataset.read_1d::<f64>().unwrap()[0]);

    // Load this rank's population shard and verify its recorded length when available.
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

    // Momentum is optional so existing schema-2 restart files remain readable.
    let momentum = if let Ok(dataset) = group.dataset("momentum") {
        Some(dataset.read_1d::<f64>()?.to_vec())
    } else {
        None
    };

    // Reconstruct the optional spawning histogram from its binning state and counts.
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

    // Assemble a complete rank-local state, retaining expected basis invariants in memory.
    Ok(RestartState {
        report,
        shift,
        nwprev,
        nrefprev,
        nsampledprev,
        nsampledoprev,
        populations,
        momentum,
        excitation_hist,
        base_seed,
        overlap_weight,
        ndets: expected_ndets,
        basis_hash: expected_hash,
        representation,
        reached,
        target_population,
    })
}
