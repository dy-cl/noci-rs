// mpiutils.rs
use std::{ffi::c_void, ptr};

use mpi::topology::{Communicator};
use mpi::traits::*;
use mpi::datatype::{Partition, PartitionMut};
use serde::{Serialize, de::DeserializeOwned};

use crate::stochastic::Walkers;
use crate::stochastic::PopulationUpdate;

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
    ///     `world`: Communicator, MPI Communicator object.
    ///     `nbytes`: usize, size of allocation.
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
    ///     `self`: Sharedffi, shared remote memory access data.
    pub fn barrier(&self) {
        unsafe {mpi::ffi::MPI_Barrier(self.shared_comm);}
    }
}

impl Drop for Sharedffi {
    /// Free the shared memory segment when Sharedffi goes out of scope.
    /// # Arguments: 
    ///     `self`: Sharedffi, shared remote memory access data.
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
///     `world`: Communicator, MPI communicator object (MPI_COMM_WORLD). 
///     `value`: If rank 0 this is the value to broadcast. Any other rank recieves value from 0.
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
///     `det`: usize, determinant index.
///     `nranks`: usize, number of MPI ranks.
pub fn owner(det: usize, nranks: usize) -> usize {
    det % nranks
}

/// Take initialised walker population as a full vector and remove population from the vector if a
/// given index is not owned by the thread in question. This is obviously quite wasteful keeping a
/// full vector of ndets for each thread, and initialising population just to remove it but it is
/// simple to do for now.
/// # Arguments
///     `w`: Walkers, contains information about determinant populations, indices, and occupations.
///     `irank`: usize, rank of current thread.
///     `nranks`: usize, total number of threads.
pub fn local_walkers(mut w: Walkers, irank: usize, nranks: usize) -> Walkers {
    let occ = w.occ().to_vec();
    for i in occ {
        if owner(i, nranks) != irank {
            let ni = w.get(i);
            w.add(i, -ni)
        }
    }
    w
}

/// For any spawning procedure that spawns onto a determinant not owned by the originating rank, we
/// must communicate this change. Each rank sends a Vec<PopulationUpdate> to every other ranking
/// containing the required population changes.
/// # Arguments:
///     `world`: Communicator, MPI communicator object (MPI_COMM_WORLD). 
///     `send`: [Vec<PopulationUpdate>], per destination send buffers.
pub fn communicate_spawn_updates(world: &impl Communicator, send: &[Vec<PopulationUpdate>]) -> Vec<PopulationUpdate> {
    let nranks = world.size() as usize;

    // Assemble message to be communicated. 
    // Number of messages this rank will send to rank i.
    let mut send_counts = vec![0i32; nranks];
    // Where inside send_contig do the messages for a given rank start.
    let mut send_displacements = vec![0i32; nranks];
    // Contiguous send buffer.
    let mut send_contig: Vec<PopulationUpdate> = Vec::new();
    for i in 0..nranks {
        send_displacements[i] = send_contig.len() as i32;
        send_contig.extend_from_slice(&send[i]);
        send_counts[i] = send[i].len() as i32;
    }

    // Exchange send_counts to find out how many messages are to be recieved. 
    // Once complete recv_counts contains number of messages rank i will send to the current rank.
    let mut recv_counts = vec![0i32; nranks];
    world.all_to_all_into(&send_counts[..], &mut recv_counts[..]);

    // Construct the recv_displacements and allocate buffer for messages to be recieved.
    let mut recv_displacements = vec![0i32; nranks];
    let total_recv: usize = recv_counts.iter().map(|&x| x as usize).sum();
    for i in 1..nranks {
        recv_displacements[i] = recv_displacements[i - 1] + recv_counts[i - 1];
    }
    let mut recv_contig = vec![PopulationUpdate {det: 0, dn: 0}; total_recv];

    // Exchange messages.
    let send_part = Partition::new(&send_contig[..], send_counts, send_displacements);
    let mut recv_part = PartitionMut::new(&mut recv_contig[..], recv_counts, recv_displacements);
    world.all_to_all_varcount_into(&send_part, &mut recv_part);
    recv_contig
}

/// Gather walker populations from all ranks into a global list. Requires use of the
/// PopulationUpdate struct which strictly is for population changes (hence dn) rather than totals.
/// But it works for this routine so there is no reason to define another struct.
/// # Arguments:
///     `world`: Communicator, MPI communicator object (MPI_COMM_WORLD). 
///     `local`: [PopulationUpdate], determinant populations and indices on this rank.
pub fn gather_all_walkers(world: &impl Communicator, local: &[PopulationUpdate]) -> Vec<PopulationUpdate> {
    let nranks = world.size() as usize;
    
    // Calculate how many entries this rank will send, and recieve the count of how many entries
    // will be recieved from other ranks.
    let send_count = local.len() as i32;
    let mut recv_counts = vec![0i32; nranks];
    world.all_gather_into(&send_count, &mut recv_counts[..]);
    
    // Find displacements for where each ranks recieved message will be placed.
    let mut recv_displacements = vec![0i32; nranks];
    for i in 1..nranks {
        recv_displacements[i] = recv_displacements[i - 1] + recv_counts[i - 1]; 
    }
    // Get total number of entries to be recieved from all ranks and allocate buffer to hold.
    let total_recv: usize = recv_counts.iter().map(|&c| c as usize).sum();
    let mut recv_contig = vec![PopulationUpdate {det: 0, dn: 0}; total_recv];
    
    // Gather the local list from each rank and return the global list. 
    let mut recv_part = PartitionMut::new(&mut recv_contig[..], recv_counts, recv_displacements);
    world.all_gather_varcount_into(local, &mut recv_part);
    recv_contig
}
