// qmc/exchange.rs

// External crate imports.
use mpi::datatype::{Partition, PartitionMut};
use mpi::topology::Communicator;
use mpi::traits::*;

// Crate-root imports.
use crate::time_call;

// Parent/sibling imports.
use super::fri::FriAmplitude;

/// Sparse keyed population update communicated between MPI ranks and annihilated by summing the
/// amplitudes that share a key.
pub(crate) trait KeyedUpdate: FriAmplitude + Equivalence + Default {
    /// Ordered key identifying the population the update applies to.
    type Key: Ord + Copy;

    /// Return the key of the population the update applies to.
    /// # Arguments:
    /// - `self`: Sparse update.
    /// # Returns:
    /// - `Self::Key`: Update key.
    fn key(&self) -> Self::Key;
}

/// Reusable MPI buffers for exchanging and gathering keyed population updates.
pub(crate) struct ExchangeScratch<U> {
    /// Number of sparse updates contributed by each rank for all-gather.
    pub(crate) gather_counts: Vec<i32>,
    /// Displacements for gathered sparse updates.
    pub(crate) gather_displs: Vec<i32>,
    /// Reusable receive buffer for gathered sparse updates.
    pub(crate) gather_recv: Vec<U>,
    /// Number of updates sent to each rank.
    pub(crate) send_counts: Vec<i32>,
    /// Displacements into the contiguous send buffer for each rank.
    pub(crate) send_displacements: Vec<i32>,
    /// Number of updates received from each rank.
    pub(crate) recv_counts: Vec<i32>,
    /// Displacements into the contiguous receive buffer for each rank.
    pub(crate) recv_displacements: Vec<i32>,
    /// Reusable contiguous send buffer.
    pub(crate) send_contig: Vec<U>,
    /// Reusable remote updates paired with their destination ranks.
    pub(crate) send_ranked: Vec<(usize, U)>,
    /// Reusable contiguous receive buffer.
    pub(crate) recv_contig: Vec<U>,
}

impl<U> ExchangeScratch<U> {
    /// Construct reusable MPI buffers sized for the communicator.
    /// # Arguments:
    /// - `nranks`: Number of MPI ranks.
    /// # Returns
    /// - `ExchangeScratch<U>`: Empty exchange storage.
    pub(crate) fn new(nranks: usize) -> Self {
        Self {
            gather_counts: vec![0; nranks],
            gather_displs: vec![0; nranks],
            gather_recv: Vec::new(),
            send_counts: vec![0; nranks],
            send_displacements: vec![0; nranks],
            recv_counts: vec![0; nranks],
            recv_displacements: vec![0; nranks],
            send_contig: Vec::new(),
            send_ranked: Vec::new(),
            recv_contig: Vec::new(),
        }
    }
}

/// Pack remote updates into contiguous per-destination blocks with their counts and
/// displacements. Updates are sorted by destination and key; with `merge`, repeated keys bound
/// for one rank are summed and zero sums are dropped before transmission.
/// # Arguments:
/// - `nranks`: Number of MPI ranks.
/// - `scratch`: Reusable MPI update storage holding `send_ranked`.
/// - `merge`: Whether to coalesce repeated keys before sending.
/// # Returns:
/// - `()`: Fills `send_contig`, `send_counts` and `send_displacements`.
pub(crate) fn pack_updates<U: KeyedUpdate>(
    nranks: usize,
    scratch: &mut ExchangeScratch<U>,
    merge: bool,
) {
    scratch.send_counts.fill(0);
    scratch.send_displacements.fill(0);
    scratch
        .send_ranked
        .sort_unstable_by_key(|&(peer, update)| (peer, update.key()));

    // Sum repeated keys bound for the same rank so each is transmitted once.
    if merge {
        let mut out = 0usize;
        for i in 0..scratch.send_ranked.len() {
            let (peer, update) = scratch.send_ranked[i];
            if out != 0
                && scratch.send_ranked[out - 1].0 == peer
                && scratch.send_ranked[out - 1].1.key() == update.key()
            {
                let previous = &mut scratch.send_ranked[out - 1].1;
                previous.set_amplitude(previous.amplitude() + update.amplitude());
            } else {
                scratch.send_ranked[out] = (peer, update);
                out += 1;
            }
        }
        scratch.send_ranked.truncate(out);
    }

    // Keep the transmitted updates in per-rank contiguous segments and record the counts and
    // displacements used by the collective exchange.
    scratch.send_contig.clear();
    for &(peer, update) in &scratch.send_ranked {
        if !merge || update.amplitude() != 0.0 {
            scratch.send_counts[peer] += 1;
            scratch.send_contig.push(update);
        }
    }
    let mut sent = 0usize;
    for peer in 0..nranks {
        scratch.send_displacements[peer] = sent as i32;
        sent += scratch.send_counts[peer] as usize;
    }
}

/// Exchange packed updates between every pair of MPI ranks.
/// The per-destination counts are exchanged first so each rank can size its receive segments,
/// followed by one variable-count all-to-all of the packed payloads.
/// # Arguments:
/// - `world`: `MPI communicator object (MPI_COMM_WORLD).`
/// - `scratch`: Reusable MPI storage with packed send data from `pack_updates`.
/// # Returns
/// - `&[U]`: Flat buffer of every update received from other ranks.
pub(crate) fn exchange_updates<'a, U: KeyedUpdate>(
    world: &impl CommunicatorCollectives,
    scratch: &'a mut ExchangeScratch<U>,
) -> &'a [U] {
    time_call!(
        crate::timers::stochastic::add_exchange_population_changes,
        {
            let nranks = world.size() as usize;

            // Gather every rank's per-destination message sizes, so that `recv_counts[peer]` is
            // the number of updates rank `peer` will send to this rank.
            time_call!(
                crate::timers::stochastic::add_exchange_population_change_counts,
                {
                    world.all_to_all_into(&scratch.send_counts[..], &mut scratch.recv_counts[..]);
                }
            );

            // `recv_displacements[peer]` is the start of rank `peer`'s block in `recv_contig`.
            let mut nrecv = 0usize;
            for peer in 0..nranks {
                scratch.recv_displacements[peer] = nrecv as i32;
                nrecv += scratch.recv_counts[peer] as usize;
            }

            // Reuse one contiguous receive buffer large enough for all remote updates.
            scratch.recv_contig.clear();
            scratch.recv_contig.resize(nrecv, U::default());

            let send_part = Partition::new(
                &scratch.send_contig[..],
                &scratch.send_counts[..],
                &scratch.send_displacements[..],
            );
            let mut recv_part = PartitionMut::new(
                &mut scratch.recv_contig[..],
                &scratch.recv_counts[..],
                &scratch.recv_displacements[..],
            );
            time_call!(
                crate::timers::stochastic::add_exchange_population_change_payload,
                {
                    world.all_to_all_varcount_into(&send_part, &mut recv_part);
                }
            );

            &scratch.recv_contig[..]
        }
    )
}

/// Gather variable-length updates from all ranks so every rank holds the same global list.
/// # Arguments:
/// - `world`: `MPI communicator object (MPI_COMM_WORLD).`
/// - `send`: Local updates to gather.
/// - `scratch`: Reusable MPI scratch space.
/// # Returns
/// - `&[U]`: Global gathered updates in rank order.
pub(crate) fn gather_updates<'a, U: KeyedUpdate>(
    world: &impl Communicator,
    send: &[U],
    scratch: &'a mut ExchangeScratch<U>,
) -> &'a [U] {
    time_call!(crate::timers::stochastic::add_gather_all_populations, {
        // Exchange per-rank lengths first to build displacements for the variable-count gather
        // into reusable contiguous storage.
        let nsend = send.len() as i32;
        world.all_gather_into(&nsend, &mut scratch.gather_counts[..]);

        let mut ntot = 0usize;
        for (i, &n) in scratch.gather_counts.iter().enumerate() {
            scratch.gather_displs[i] = ntot as i32;
            ntot += n as usize;
        }

        if ntot == 0 {
            scratch.gather_recv.clear();
            return &scratch.gather_recv[..];
        }

        scratch.gather_recv.resize(ntot, U::default());
        let mut recv = PartitionMut::new(
            &mut scratch.gather_recv[..],
            &scratch.gather_counts[..],
            &scratch.gather_displs[..],
        );
        world.all_gather_varcount_into(send, &mut recv);
        &scratch.gather_recv[..]
    })
}

/// Annihilate key-sorted updates in place: sum consecutive updates sharing a key and drop zero
/// sums.
/// # Arguments:
/// - `updates`: Updates sorted by key.
/// # Returns
/// - `()`: Leaves one nonzero update per key.
pub(crate) fn coalesce<U: KeyedUpdate>(updates: &mut Vec<U>) {
    let mut out = 0usize;
    for i in 0..updates.len() {
        if out > 0 && updates[out - 1].key() == updates[i].key() {
            let amplitude = updates[out - 1].amplitude() + updates[i].amplitude();
            updates[out - 1].set_amplitude(amplitude);
        } else {
            updates[out] = updates[i];
            out += 1;
        }
    }

    updates.truncate(out);
    updates.retain(|update| update.amplitude() != 0.0);
}
