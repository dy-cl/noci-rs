// mpiutils.rs
use std::{ffi::c_void, ptr};

use mpi::topology::{Communicator};
use mpi::traits::*;
use mpi::point_to_point;
use mpi::datatype::{PartitionMut};
use serde::{Serialize, de::DeserializeOwned};

use crate::time_call;
use crate::stochastic::{Walkers, PopulationUpdate, MPIScratch};

pub struct Sharedffi {
    // Communicator for ranks on the same node.
    pub shared_comm: mpi::ffi::MPI_Comm,
    // MPI window is remotely accessible region of memory.
    pub win: mpi::ffi::MPI_Win,
    // Pointer to start of shared memory (across node).
    pub base: *mut u8,
    // Total size of the shared segment.
    pub nbytes: usize,   
    // Rank of current process within the same node.
    pub shared_rank: i32,  
    // Number of ranks on this node.
    pub shared_size: i32,              
}

impl Sharedffi {
     /// Allocate remote memory access region.
    /// # Arguments:
    /// - `world`: MPI Communicator object.
    /// - `nbytes`: Size of allocation.
    /// # Returns
    /// - `Sharedffi`: Shared-memory allocation metadata and pointer to the shared segment.
    pub fn allocate(world: &impl Communicator, nbytes: usize) -> Self {
        // Calling MPI API in C so needs to be unsafe.
        unsafe {
            // Create new MPI communicator and make it null.
            let mut shared_comm: mpi::ffi::MPI_Comm = mpi::ffi::RSMPI_COMM_NULL;
            // Request rank grouping by shared memory domains, that is, nodes. More comments about
            // arguments of call would prbably help.
            mpi::ffi::MPI_Comm_split_type(world.as_raw(), mpi::ffi::MPI_COMM_TYPE_SHARED as i32, 0, mpi::ffi::RSMPI_INFO_NULL, &mut shared_comm);

            // Fill in number of shared memory ranks and individual indices.
            let mut shared_rank: i32 = 0;
            let mut shared_size: i32 = 0;
            mpi::ffi::MPI_Comm_rank(shared_comm, &mut shared_rank);
            mpi::ffi::MPI_Comm_size(shared_comm, &mut shared_size);

            // Only rank 0 should allocate the required memory. All others request 0 bytes.
            let bytes: mpi::ffi::MPI_Aint = if shared_rank == 0 {nbytes as _} else {0};
            // Unit is 1 (bytes).
            let unit: i32 = 1;
            
            // Initialise null pointer which will later be filled with pointer to start of shared
            // memory.
            let mut base: *mut c_void = ptr::null_mut();
            // Allocate the shared memory.
            let mut win: mpi::ffi::MPI_Win = mpi::ffi::RSMPI_WIN_NULL;
            mpi::ffi::MPI_Win_allocate_shared(bytes, unit, mpi::ffi::RSMPI_INFO_NULL, shared_comm, (&mut base as *mut *mut c_void).cast::<c_void>(), &mut win,);
            
            let mut qsize: mpi::ffi::MPI_Aint = 0;
            let mut qdisp: i32 = 0;
            let mut qbase: *mut c_void = ptr::null_mut();
            // All ranks get a pointer to the segment of rank 0 where the data is stored in memory.
            // qbase is the pointer, qsize is the segment size, and qdisp is displacement unit (1
            // in this case).
            mpi::ffi::MPI_Win_shared_query(win, 0, &mut qsize, &mut qdisp, (&mut qbase as *mut *mut c_void).cast::<c_void>());

            Self {shared_comm, win, base: qbase as *mut u8, nbytes: qsize as usize, shared_rank, shared_size,}
        }
    }
    
    /// Synchronise ranks on the same node.
    /// # Arguments:
    /// - `self`: Shared remote memory access data.
    /// # Returns
    /// - `()`: Blocks until all ranks in the shared-memory communicator reach the barrier.
    pub fn barrier(&self) {
        unsafe {mpi::ffi::MPI_Barrier(self.shared_comm);}
    }
}

impl Drop for Sharedffi {
    /// Free the shared memory segment when Sharedffi goes out of scope.
    /// # Arguments: 
    /// - `self`: Shared remote memory access data.
    /// # Returns
    /// - `()`: Frees the MPI window and shared-memory communicator.
    fn drop(&mut self) {
        unsafe {
            let mut w = self.win;
            let _ = mpi::ffi::MPI_Win_free(&mut w);

            let mut c = self.shared_comm;
            let _ = mpi::ffi::MPI_Comm_free(&mut c);
        }
    }
}

/// Broadcast a serialisable value of arbitrary (provided it is serialisable) type T from rank 0 to
/// all MPI ranks.
/// # Arguments:
/// - `world`: MPI communicator object (MPI_COMM_WORLD). 
/// - `value`: If rank 0 this is the value to broadcast. Any other rank recieves value from 0.
/// # Returns
/// - `()`: Updates `value` in place on non-root ranks.
/// # Type Parameters:
/// - `T`: Type of value being broadcast.
pub fn broadcast<T>(world: &impl Communicator, value: &mut T)
where
    // Function can broadcast any type provided that it can be converted to bytes and consequently
    // reconstructed from bytes.
    T: Serialize + DeserializeOwned {

        let irank = world.rank();
        let root = world.process_at_rank(0);
        
        // On rank 0 convert the given value into binary, on all other ranks create an empty buffer
        // to recieve the value.
        let mut bytes: Vec<u8> = if irank == 0 {
            bincode::serialize(value).unwrap()
        } else {
            Vec::new()
        };
        
        // Broadcast the number of bytes that will be sent from rank 0.
        let mut len: u64 = bytes.len() as u64;
        root.broadcast_into(&mut len);
        
        // All ranks except 0 allocate the recieve buffer to bt ethe correct size.
        if irank != 0 {
            bytes.resize(len as usize, 0u8);
        }

        // Send value from rank 0 to all other ranks in chunks to avoid overflow.
        const CHUNK: usize = 256 * 1024 * 1024; 
        let mut off = 0usize;
        while off < bytes.len() {
            let end = (off + CHUNK).min(bytes.len());
            root.broadcast_into(&mut bytes[off..end]);
            off = end;
        }
        
        // On all ranks except 0 deserialise the value and put it back into T.
        if irank != 0 {
            *value = bincode::deserialize(&bytes).unwrap();
        }
}

/// Given a determinant index return which MPI rank owns it.
/// # Arguments 
/// - `det`: Determinant index.
/// - `nranks`: Number of MPI ranks.
/// # Returns
/// - `usize`: MPI rank that owns the determinant.
#[inline(always)]
pub fn owner(det: usize, ndets: usize, nranks: usize) -> usize {
    det * nranks / ndets
}

/// Find range of determinants owned by a given MPI rank.
/// # Arguments
/// - `irank`: Current rank index.
/// - `ndets`: Number of determinants.
/// - `nranks`: Total number of ranks. 
/// # Returns 
/// - `(usize, usize)`: Start and end index of determinants owned by this rank.
#[inline(always)]
pub fn rank_range(irank: usize, ndets: usize, nranks: usize) -> (usize, usize) {
    let start = ndets * irank / nranks;
    let end = ndets * (irank + 1) / nranks;
    (start, end)
}

/// Take initialised walker population as a full vector and remove population from the vector if a
/// given index is not owned by the thread in question. This is obviously quite wasteful keeping a
/// full vector of ndets for each thread, and initialising population just to remove it but it is
/// simple to do for now.
/// # Arguments
/// - `w`: Contains information about determinant populations, indices, and occupations.
/// - `irank`: Rank of current thread.
/// - `nranks`: Total number of threads.
/// # Returns
/// - `Walkers`: Walker population restricted to determinants owned by the current rank.
pub(crate) fn local_walkers(mut w: Walkers, irank: usize, nranks: usize) -> Walkers {
    let occ = w.occ().to_vec();
    for i in occ {
        if owner(i, w.len(), nranks) != irank {
            let ni = w.get(i);
            w.add(i, -ni)
        }
    }
    w
}

/// Communicate spawned walker updates between MPI ranks.
/// # Arguments:
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// - `send`: Per-rank spawned walker updates to send.
/// - `scratch`: Reusable MPI scratch space.
/// # Returns
/// - `&[PopulationUpdate]`: Received spawned walker updates from all ranks.
pub(crate) fn communicate_spawn_updates<'a>(world: &impl Communicator, send: &[Vec<PopulationUpdate>], scratch: &'a mut MPIScratch) -> &'a [PopulationUpdate] {
    time_call!(crate::timers::stochastic::add_communicate_spawn_updates, {
        let irank = world.rank() as usize;
        let nranks = world.size() as usize;

        for (i, buf) in send.iter().enumerate() {
            scratch.send_counts[i] = buf.len() as i32;
        }

        world.all_to_all_into(&scratch.send_counts[..], &mut scratch.recv_counts[..]);

        let nrecv: usize = scratch.recv_counts.iter().map(|&n| n as usize).sum();
        scratch.recv_contig.clear();
        scratch.recv_contig.resize(nrecv, PopulationUpdate {det: 0, dn: 0});

        mpi::request::scope(|scope| {
            let mut recv_reqs = Vec::new();
            let mut send_reqs = Vec::new();

            let mut rest = &mut scratch.recv_contig[..];
            for peer in 0..nranks {
                if peer == irank {continue;}
                let n = scratch.recv_counts[peer] as usize;
                if n == 0 {continue;}

                let (chunk, tail) = rest.split_at_mut(n);
                rest = tail;

                let proc = world.process_at_rank(peer as i32);
                recv_reqs.push(proc.immediate_receive_into(scope, chunk));
            }

            for (peer, buf) in send.iter().enumerate() {
                if peer == irank || buf.is_empty() {continue;}

                let proc = world.process_at_rank(peer as i32);
                send_reqs.push(proc.immediate_send(scope, &buf[..]));
            }

            for req in recv_reqs {
                req.wait();
            }
            for req in send_reqs {
                req.wait();
            }
        });

        &scratch.recv_contig[..]
    })
}

/// Gather variable-length walker updates from all ranks into a reusable receive buffer.
/// # Arguments:
/// - `world`: MPI communicator object (MPI_COMM_WORLD).
/// - `send`: Local walker updates to gather.
/// - `scratch`: Reusable MPI scratch space.
/// # Returns
/// - `&[PopulationUpdate]`: Global gathered walker updates.
pub(crate) fn gather_all_walkers<'a>(world: &impl Communicator, send: &[PopulationUpdate], scratch: &'a mut MPIScratch) -> &'a [PopulationUpdate] {
    time_call!(crate::timers::stochastic::add_gather_all_walkers, {
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

        scratch.gather_recv.resize(ntot, PopulationUpdate {det: 0, dn: 0});
        let mut recv = PartitionMut::new(&mut scratch.gather_recv[..], &scratch.gather_counts[..], &scratch.gather_displs[..]);
        world.all_gather_varcount_into(send, &mut recv);
        &scratch.gather_recv[..]
    })
}
