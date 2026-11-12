// qmc/restart.rs
//! Restart files of every stochastic propagator. A restart holds the report, the shift and its
//! target state, the base seed, each rank's populations as `(index, value)` pairs, and named
//! controller values of the propagator. Because the populations are keyed, a run may resume on any
//! number of ranks: every rank reads every saved pair and keeps the ones it owns.

// Standard library imports.
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

// External crate imports.
use hdf5::File;
use mpi::topology::Communicator;
use mpi::traits::*;

// Crate-root imports.
use crate::determinant::{NOCIIndex, NOCISpace};

// Parent/sibling imports.
use super::exchange::PopulationUpdate;
use super::excitgen::ExcitationHist;

/// Version of the restart layout; files of any other version are rejected.
const SCHEMA: u64 = 3;

/// One rank's part of a restart.
pub(crate) struct Restart {
    /// Last completed report.
    pub(crate) report: usize,
    /// Population-control shift `E_s`.
    pub(crate) shift: f64,
    /// Whether the population had reached its target.
    pub(crate) reached: bool,
    /// Base seed shared by every rank.
    pub(crate) seed: u64,
    /// Populations owned by this rank, keyed by flat index.
    pub(crate) populations: Vec<PopulationUpdate>,
    /// Controller values of the propagator, by name.
    pub(crate) values: BTreeMap<String, Vec<f64>>,
    /// Spawn-magnitude histogram of this rank, when recorded.
    pub(crate) histogram: Option<ExcitationHist>,
}

/// Write a restart, with metadata from rank zero and every rank's populations and values in its
/// own group. The writes are serialised, since the file is not opened for parallel I/O.
/// # Arguments:
/// - `path`: Restart file path.
/// - `identity`: Kind of propagation and hash of its determinant space.
/// - `restart`: This rank's part of the restart.
/// - `world`: MPI communicator.
/// # Returns:
/// - `hdf5::Result<()>`: Error of the first failed write.
pub(crate) fn write_restart(
    path: &str,
    identity: (&str, [u64; 2]),
    restart: &Restart,
    world: &impl Communicator,
) -> hdf5::Result<()> {
    let (kind, hash) = identity;
    let irank = world.rank() as usize;

    // Rank zero creates the file and writes the metadata shared by every rank.
    if irank == 0 {
        if let Some(parent) = Path::new(path).parent()
            && !parent.as_os_str().is_empty()
        {
            let _ = fs::create_dir_all(parent);
        }
        let meta = File::create(path)?.create_group("meta")?;
        meta.new_dataset_builder()
            .with_data(&[SCHEMA])
            .create("schema")?;
        meta.new_dataset_builder()
            .with_data(kind.as_bytes())
            .create("kind")?;
        meta.new_dataset_builder().with_data(&hash).create("hash")?;
        meta.new_dataset_builder()
            .with_data(&[
                restart.report as u64,
                restart.seed,
                u64::from(restart.reached),
            ])
            .create("state")?;
        meta.new_dataset_builder()
            .with_data(&[restart.shift])
            .create("shift")?;
        meta.new_dataset_builder()
            .with_data(&[world.size() as u64])
            .create("nranks")?;
    }

    world.barrier();

    // Each rank in turn adds its populations, values and histogram.
    for rank in 0..world.size() as usize {
        if irank == rank {
            let group = File::open_rw(path)?.create_group(&format!("rank_{irank}"))?;
            let indices = restart
                .populations
                .iter()
                .map(|u| u.index)
                .collect::<Vec<_>>();
            let values = restart.populations.iter().map(|u| u.dn).collect::<Vec<_>>();
            group
                .new_dataset_builder()
                .with_data(&indices)
                .create("indices")?;
            group
                .new_dataset_builder()
                .with_data(&values)
                .create("populations")?;

            // Named controller arrays retain each propagator's state without encoding it
            // in the shared population-key format.
            let named = group.create_group("values")?;
            for (name, value) in &restart.values {
                named
                    .new_dataset_builder()
                    .with_data(value)
                    .create(name.as_str())?;
            }

            if let Some(hist) = &restart.histogram {
                let h = group.create_group("histogram")?;
                h.new_dataset_builder()
                    .with_data(&[hist.logmin, hist.logmax])
                    .create("range")?;
                h.new_dataset_builder()
                    .with_data(&[
                        hist.nbins as u64,
                        hist.ntotal,
                        hist.noverflow_low,
                        hist.noverflow_high,
                    ])
                    .create("totals")?;
                h.new_dataset_builder()
                    .with_data(&hist.counts)
                    .create("counts")?;
            }
        }

        world.barrier();
    }

    Ok(())
}

/// Read a restart. Every rank reads every saved population and keeps the ones it owns, so the run
/// may resume on a different number of ranks; values and the histogram come from the group of the
/// same rank, or from rank zero's when fewer ranks were saved.
/// # Arguments:
/// - `path`: Restart file path.
/// - `identity`: Kind of propagation and hash of its determinant space.
/// - `owns`: Whether this rank owns a population index, given the saved report and base seed.
/// - `world`: MPI communicator.
/// # Returns:
/// - `hdf5::Result<Restart>`: This rank's part of the restart.
/// # Panics
/// - Panics if the schema, kind or hash differ from the current run's.
pub(crate) fn read_restart(
    path: &str,
    identity: (&str, [u64; 2]),
    owns: impl Fn(u64, (usize, u64)) -> bool,
    world: &impl Communicator,
) -> hdf5::Result<Restart> {
    let file = File::open(path)?;
    let meta = file.group("meta")?;
    validate(&meta, identity)?;

    // Shared report and RNG metadata determine ownership when the saved rank count differs.
    let state = meta.dataset("state")?.read_1d::<u64>()?;
    let (report, seed, reached) = (state[0] as usize, state[1], state[2] != 0);
    let shift = meta.dataset("shift")?.read_1d::<f64>()?[0];
    let saved = meta.dataset("nranks")?.read_1d::<u64>()?[0] as usize;

    // Keep this rank's populations from every saved rank.
    let mut populations = Vec::new();
    for rank in 0..saved {
        let group = file.group(&format!("rank_{rank}"))?;
        let indices = group.dataset("indices")?.read_1d::<u64>()?;
        let values = group.dataset("populations")?.read_1d::<f64>()?;
        populations.extend(
            indices
                .iter()
                .zip(values.iter())
                .filter(|&(&index, _)| owns(index, (report, seed)))
                .map(|(&index, &dn)| PopulationUpdate { index, dn }),
        );
    }

    // Values and histogram of the matching saved rank.
    let irank = world.rank() as usize;
    let group = file.group(&format!("rank_{}", if irank < saved { irank } else { 0 }))?;

    // Restore controller arrays by name, independently of the rank that owns each population.
    let named = group.group("values")?;
    let mut values = BTreeMap::new();

    for name in named.member_names()? {
        let value = named.dataset(&name)?.read_1d::<f64>()?.to_vec();
        values.insert(name, value);
    }

    // Histograms are optional diagnostics and use the same saved-rank fallback as the values.
    let histogram = match group.group("histogram") {
        Ok(h) => {
            let range = h.dataset("range")?.read_1d::<f64>()?;
            let totals = h.dataset("totals")?.read_1d::<u64>()?;
            let mut hist = ExcitationHist::new(range[0], range[1], totals[0] as usize);
            hist.ntotal = totals[1];
            hist.noverflow_low = totals[2];
            hist.noverflow_high = totals[3];
            hist.counts = h.dataset("counts")?.read_1d::<u64>()?.to_vec();
            Some(hist)
        }
        Err(_) => None,
    };

    Ok(Restart {
        report,
        shift,
        reached,
        seed,
        populations,
        values,
        histogram,
    })
}

/// Read the base seed of a restart, which the run's random streams are built from before its
/// populations are read.
/// # Arguments:
/// - `path`: Restart file path.
/// - `identity`: Kind of propagation and hash of its determinant space.
/// # Returns:
/// - `hdf5::Result<u64>`: Base seed of the saved run.
/// # Panics
/// - Panics if the schema, kind or hash differ from the current run's.
pub(crate) fn restart_seed(
    path: &str,
    identity: (&str, [u64; 2]),
) -> hdf5::Result<u64> {
    let meta = File::open(path)?.group("meta")?;
    validate(&meta, identity)?;
    Ok(meta.dataset("state")?.read_1d::<u64>()?[1])
}

/// Reject a restart of another layout, kind of propagation or determinant space.
/// # Arguments:
/// - `meta`: Metadata group of the restart.
/// - `identity`: Kind of propagation and hash of its determinant space.
/// # Returns:
/// - `hdf5::Result<()>`: Error of a failed read.
/// # Panics
/// - Panics if the schema, kind or hash differ from the current run's.
fn validate(
    meta: &hdf5::Group,
    identity: (&str, [u64; 2]),
) -> hdf5::Result<()> {
    let (kind, hash) = identity;
    let schema = meta.dataset("schema")?.read_1d::<u64>()?[0];
    if schema != SCHEMA {
        panic!("Restart schema mismatch: saved {schema}, current {SCHEMA}.");
    }

    let saved = meta.dataset("kind")?.read_1d::<u8>()?;
    let saved = std::str::from_utf8(saved.as_slice().unwrap()).unwrap_or("invalid");
    if saved != kind {
        panic!("Restart propagation mismatch: saved {saved}, current {kind}.");
    }

    let saved = meta.dataset("hash")?.read_1d::<u64>()?;
    if [saved[0], saved[1]] != hash {
        panic!("Restart determinant space mismatch: saved {saved:x?}, current {hash:x?}.");
    }

    Ok(())
}

/// Hash the structure of a determinant space, so a restart is only read into the space it was
/// written from: the determinant order, occupations, phases and excitations, and every parent's
/// orbital coefficients once.
/// # Arguments:
/// - `space`: Determinant space of the run.
/// # Returns:
/// - `[u64; 2]`: Two-lane FNV-style hash, stable across runs and rank counts.
pub(crate) fn basis_hash(space: &NOCISpace<f64>) -> [u64; 2] {
    let mut hash = [0xcbf29ce484222325, 0x84222325cbf29ce4];
    let mut seen_parent = vec![false; space.parents.len()];

    // Mix every structural word into both lanes; rotate the second lane to retain a distinct
    // ordering of each word's high and low halves.
    let mut mix = |value: u64| {
        hash[0] ^= value;
        hash[0] = hash[0].wrapping_mul(0x00000100000001b3);
        hash[1] ^= value.rotate_left(32);
        hash[1] = hash[1].wrapping_mul(0x00000100000001b3);
    };

    mix(space.len() as u64);
    for i in 0..space.len() {
        let index = NOCIIndex(i);
        let det = space.spin_string_pair(index);
        let alpha = space.alpha(index);
        let beta = space.beta(index);

        // Determinant order, occupations, phases and excitations determine the meaning and
        // sign of a saved population coordinate. Hash all of them before the shared frame.
        mix(i as u64);
        mix(det.parent as u64);
        for value in [alpha.occupation, beta.occupation] {
            mix(value as u64);
            mix((value >> 64) as u64);
        }
        mix(alpha.data.phase.to_bits());
        mix(beta.data.phase.to_bits());

        for value in [
            alpha.data.excitation.holes,
            alpha.data.excitation.parts,
            beta.data.excitation.holes,
            beta.data.excitation.parts,
        ] {
            mix(value as u64);
            mix((value >> 64) as u64);
        }

        // Parent frames are shared by many determinants, so each is mixed once.
        if !seen_parent[det.parent] {
            seen_parent[det.parent] = true;
            let parent = &space.parents[det.parent];
            mix(det.parent as u64);
            for c in [&parent.ca, &parent.cb] {
                mix(c.nrows() as u64);
                mix(c.ncols() as u64);
                for &value in c.iter() {
                    mix(value.to_bits());
                }
            }
        }
    }

    hash
}
