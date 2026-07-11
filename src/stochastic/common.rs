// stochastic/common.rs
use mpi::datatype::{Partition, PartitionMut};
use mpi::topology::Communicator;
use mpi::traits::*;

use super::state::{MPIScratch, PopulationUpdate};
use crate::SCFState;
use crate::noci::{DetPair, NOCIData, calculate_hs_pair, calculate_s_pair};
use crate::nonorthogonalwicks::WickScratchSpin;
use crate::time_call;

/// Find overlap matrix element S_{ij}.
/// # Arguments:
/// - `data`: Immutable stochastic propagation data.
/// - `i`: Index of state `i`.
/// - `j`: Index of state `j`.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `f64`: Overlap matrix element `S_{ij}`.
pub(in crate::stochastic) fn find_s(
    data: &NOCIData<'_, f64>,
    i: usize,
    j: usize,
    scratch: &mut WickScratchSpin<f64>,
) -> f64 {
    // Get the sorted pair of indices
    let (a, b) = if i <= j { (i, j) } else { (j, i) };
    let ldet = &data.basis[a];
    let gdet = &data.basis[b];

    // If the determinants share the same parent take an orthogonal early exit.
    if ldet.parent == gdet.parent
        && let Some(mocache) = data.mocache
        && mocache[ldet.parent].orthogonal_slater_condon
    {
        if ldet.oa == gdet.oa && ldet.ob == gdet.ob {
            return (ldet.pha * gdet.pha) * (ldet.phb * gdet.phb);
        }
        return 0.0;
    }

    // Otherwise calculate normally.
    calculate_s_pair(data, DetPair::new(ldet, gdet), Some(scratch))
}

/// Find Hamiltonian and overlap matrix elements H_{ij} and S_{ij}.
/// # Arguments:
/// - `data`: Immutable stochastic propagation data.
/// - `i`: Index of state `i`.
/// - `j`: Index of state `j`.
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `(f64, f64)`: Hamiltonian and overlap matrix elements `H_{ij}` and `S_{ij}`.
pub(in crate::stochastic) fn find_hs(
    data: &NOCIData<'_, f64>,
    i: usize,
    j: usize,
    scratch: &mut WickScratchSpin<f64>,
) -> (f64, f64) {
    // Get the sorted pair of indices
    let (a, b) = if i <= j { (i, j) } else { (j, i) };

    // Calculate the matrix element.
    calculate_hs_pair(
        data,
        DetPair::new(&data.basis[a], &data.basis[b]),
        Some(scratch),
    )
}

/// Determine the maximum scratch sizes required for computation of matrix elements using extended
/// non-orthogonal Wick's theorem depending on the maximum excitation rank present in the basis.
/// # Arguments:
/// - `basis`: Full list of the NOCI-QMC basis.
/// # Returns
/// - `(usize, usize, usize)`: Maximum same-spin scratch size, alpha excitation size, and beta
///   excitation size.
pub(in crate::stochastic) fn max_scratch_sizes(basis: &[SCFState]) -> (usize, usize, usize) {
    let maxexa = basis
        .iter()
        .map(|st| st.excitation.alpha.holes.len())
        .max()
        .unwrap_or(0);
    let maxexb = basis
        .iter()
        .map(|st| st.excitation.beta.holes.len())
        .max()
        .unwrap_or(0);
    let maxsame = 2 * maxexa.max(maxexb);
    let maxla = 2 * maxexa;
    let maxlb = 2 * maxexb;
    (maxsame, maxla, maxlb)
}

/// Communicate spawned population updates between MPI ranks.
/// Remote spawn updates are stored locally as one `Vec<PopulationUpdate>` per destination rank.
/// This routine packs those per-destination buffers into one contiguous send buffer, exchanges the
/// number of updates each rank will send/receive, then performs one `MPI_Alltoallv`-style exchange
/// of the packed payloads.
/// # Arguments:
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// - `scratch`: Reusable MPI scratch space for counts, displacements, and contiguous send/recv buffers.
/// # Returns
/// - `&[PopulationUpdate]`: Flat buffer containing all spawned population updates received from other ranks.
pub(crate) fn exchange_population_changes<'a>(
    world: &impl CommunicatorCollectives,
    scratch: &'a mut MPIScratch,
) -> &'a [PopulationUpdate] {
    time_call!(
        crate::timers::stochastic::add_exchange_population_changes,
        {
            let nranks = world.size() as usize;

            // Gather every rank's per-destination message sizes.
            // After this, `recv_counts[peer]` contains how many updates rank `peer` will send to the
            // current rank.
            time_call!(
                crate::timers::stochastic::add_exchange_population_change_counts,
                {
                    world.all_to_all_into(&scratch.send_counts[..], &mut scratch.recv_counts[..]);
                }
            );

            // Build the incoming MPI metadata.
            // `recv_displacements[peer]` is the starting offset of rank `peer`'s block inside the packed
            // contiguous receive buffer `recv_contig`.
            // `nrecv` is the total number of remote updates this rank will receive.
            let mut nrecv = 0usize;
            for peer in 0..nranks {
                scratch.recv_displacements[peer] = nrecv as i32;
                nrecv += scratch.recv_counts[peer] as usize;
            }

            // Reuse one contiguous recieve buffer large enough for all remote updates.
            scratch.recv_contig.clear();
            scratch
                .recv_contig
                .resize(nrecv, PopulationUpdate { det: 0, dn: 0.0 });

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

            // Return the receive buffer for later accumulation.
            &scratch.recv_contig[..]
        }
    )
}

/// Gather variable-length population updates from all ranks into a reusable receive buffer.
/// # Arguments:
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// - `send`: Local population updates to gather.
/// - `scratch`: Reusable MPI scratch space.
/// # Returns
/// - `&[PopulationUpdate]`: Global gathered population updates.
pub(crate) fn gather_all_populations<'a>(
    world: &impl Communicator,
    send: &[PopulationUpdate],
    scratch: &'a mut MPIScratch,
) -> &'a [PopulationUpdate] {
    time_call!(crate::timers::stochastic::add_gather_all_populations, {
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

        scratch
            .gather_recv
            .resize(ntot, PopulationUpdate { det: 0, dn: 0.0 });
        let mut recv = PartitionMut::new(
            &mut scratch.gather_recv[..],
            &scratch.gather_counts[..],
            &scratch.gather_displs[..],
        );
        world.all_gather_varcount_into(send, &mut recv);
        &scratch.gather_recv[..]
    })
}

/// Combine repeated consecutive determinant updates in place.
/// # Arguments:
/// - `updates`: Population updates sorted by determinant index.
/// # Returns
/// - `()`: Compresses repeated determinant updates in place.
pub(in crate::stochastic) fn coalesce_population_updates(updates: &mut Vec<PopulationUpdate>) {
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
    updates.retain(|up| up.dn != 0.0);
}
