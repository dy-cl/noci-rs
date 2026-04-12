// noci/wicks.rs
use std::ptr::NonNull;
use std::fs::{File, self, OpenOptions};
use std::path::{PathBuf};

use ndarray::Array2;
use mpi::topology::Communicator;
use memmap2::MmapOptions;

use crate::{AoData, SCFState};
use crate::nonorthogonalwicks::{DiffSpinBuild, DiffSpinMeta, PairMeta, SameSpinBuild, SameSpinMeta, WicksRma, WicksShared, 
           WicksView, WicksBacking, WicksDiskMeta};
use crate::mpiutils::Sharedffi;
use crate::input::Input;

use crate::nonorthogonalwicks::{write2, write_same_spin, write_diff_spin, assign_offsets};
use crate::mpiutils::{broadcast};

/// Build the Wick's per reference-pair intermediates and store in a shared memory access region (per node). 
/// This may be in RAM or on disk if requested.
/// # Arguments:
/// - `world`: MPI communicator object.
/// - `ao`: Contains AO integrals and other system data. 
/// - `noci_reference_basis`: Vector of only the reference determinants.
/// - `tol`: Tolerance for a number being zero.
/// - `input`: User input specifications.
/// # Returns:
/// - `WicksShared`: Shared-memory storage and view for precomputed Wick's intermediates.
pub fn build_wicks_shared(world: &impl Communicator, ao: &AoData, noci_reference_basis: &[SCFState], tol: f64, input: &Input,) -> WicksShared {
    let nref = noci_reference_basis.len();
    let nmo  = noci_reference_basis[0].ca.ncols();

    let (offset, tensor_len) = assign_offsets(nref, nmo);
    let nbytes = tensor_len * std::mem::size_of::<f64>();

    println!("Number of MOs: {}", ao.n);
    println!("Estimated memory required for Wick's intermediates (MiB): {}", nbytes as f64 / (1024.0 * 1024.0));

    match input.wicks.storage {
        crate::input::WicksStorage::RAM => {
            let shared = Sharedffi::allocate(world, nbytes);
            let shared_rank = shared.shared_rank;

            let tensor_ptr = shared.base as *mut f64;
            let mut meta = vec![PairMeta::default(); nref * nref];

            if shared_rank == 0 {
                let tensor: &mut [f64] = unsafe {std::slice::from_raw_parts_mut(tensor_ptr, tensor_len)};
                tensor.fill(f64::NAN);

                for i in 0..nref {
                    let ri = &noci_reference_basis[i];
                    for (j, rj) in noci_reference_basis.iter().enumerate() {
                        println!("Building intermediates for reference pair: {}, {} on world rank {}", i, j, world.rank());

                        let aa = SameSpinBuild::new(&ao.eri_coul, &ao.h, &ao.s, &rj.ca, &ri.ca, rj.oa, ri.oa, tol);
                        let bb = SameSpinBuild::new(&ao.eri_coul, &ao.h, &ao.s, &rj.cb, &ri.cb, rj.ob, ri.ob, tol);
                        let ab = DiffSpinBuild::new(&ao.eri_coul, &ao.s, &rj.ca, &rj.cb, &ri.ca, &ri.cb, rj.oa, rj.ob, ri.oa, ri.ob, tol);

                        let idx = i * nref + j;

                        meta[idx].aa = SameSpinMeta {tilde_s_prod: aa.tilde_s_prod, phase: aa.phase, m: aa.m, nmo: aa.nmo,
                                                     f0h: aa.f0h, f0f: aa.f0f, v0: aa.v0};
                        meta[idx].bb = SameSpinMeta {tilde_s_prod: bb.tilde_s_prod, phase: bb.phase, m: bb.m, nmo: bb.nmo,
                                                     f0h: bb.f0h, f0f: bb.f0f, v0: bb.v0};
                        meta[idx].ab = DiffSpinMeta {nmo: ab.vab[0][0][0].nrows() / 2, vab0: ab.vab0, vba0: ab.vba0};

                        write_same_spin(tensor, &offset[idx].aa, &aa);
                        write_same_spin(tensor, &offset[idx].bb, &bb);
                        write_diff_spin(tensor, &offset[idx].ab, &ab);
                    }
                }
            }

            shared.barrier();
            broadcast(world, &mut meta);

            let rma = WicksRma {base_ptr: shared.base, nbytes, shared};
            let slab = NonNull::new(tensor_ptr).expect("Should not be null.");
            let view = WicksView {slab, slab_len: tensor_len, nref, off: offset, meta};

            WicksShared {backing: WicksBacking::Shared(rma), view}
        }

        crate::input::WicksStorage::Disk => {
            let cache_dir = PathBuf::from(input.wicks.cachedir.clone().unwrap_or_else(|| ".".to_string()));
            fs::create_dir_all(&cache_dir).unwrap();
            
            let slab_path = cache_dir.join("wicks.bin");
            let meta_path = cache_dir.join("wicks.meta");

            // Only use shared memory machinery here to obtain a node-local rank and barrier
            // rather than to allocate memory.
            let shared = Sharedffi::allocate(world, 1);
            let shared_rank = shared.shared_rank;

            if shared_rank == 0 {
                println!("Building Wick's intermediates and writing disk cache on world rank {}: {:?}", world.rank(), cache_dir);

                let file = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&slab_path).unwrap();
                file.set_len(nbytes as u64).unwrap();

                let mut mmap = unsafe {MmapOptions::new().len(nbytes).map_mut(&file).unwrap()};
                let ptr = mmap.as_mut_ptr() as *mut f64;
                let tensor: &mut [f64] = unsafe {std::slice::from_raw_parts_mut(ptr, tensor_len)};
                tensor.fill(f64::NAN);

                let mut meta = vec![PairMeta::default(); nref * nref];

                for i in 0..nref {
                    let ri = &noci_reference_basis[i];
                    for (j, rj) in noci_reference_basis.iter().enumerate() {
                        println!("Building intermediates for reference pair: {}, {} on world rank {}", i, j, world.rank());

                        let aa = SameSpinBuild::new(&ao.eri_coul, &ao.h, &ao.s, &rj.ca, &ri.ca, rj.oa, ri.oa, tol);
                        let bb = SameSpinBuild::new(&ao.eri_coul, &ao.h, &ao.s, &rj.cb, &ri.cb, rj.ob, ri.ob, tol);
                        let ab = DiffSpinBuild::new(&ao.eri_coul, &ao.s, &rj.ca, &rj.cb, &ri.ca, &ri.cb, rj.oa, rj.ob, ri.oa, ri.ob, tol);

                        let idx = i * nref + j;

                        meta[idx].aa = SameSpinMeta {tilde_s_prod: aa.tilde_s_prod, phase: aa.phase, m: aa.m, nmo: aa.nmo,
                                                     f0h: aa.f0h, f0f: aa.f0f, v0: aa.v0};
                        meta[idx].bb = SameSpinMeta {tilde_s_prod: bb.tilde_s_prod, phase: bb.phase, m: bb.m, nmo: bb.nmo,
                                                     f0h: bb.f0h, f0f: bb.f0f, v0: bb.v0};
                        meta[idx].ab = DiffSpinMeta {nmo: ab.vab[0][0][0].nrows() / 2, vab0: ab.vab0, vba0: ab.vba0};

                        write_same_spin(tensor, &offset[idx].aa, &aa);
                        write_same_spin(tensor, &offset[idx].bb, &bb);
                        write_diff_spin(tensor, &offset[idx].ab, &ab);
                    }
                }

                mmap.flush().unwrap();

                let disk_meta = WicksDiskMeta {version: 1, nref, slab_len: tensor_len, off: offset.clone(), meta: meta.clone(),};
                std::fs::write(&meta_path, bincode::serialize(&disk_meta).unwrap()).unwrap();
            }

            shared.barrier();

            let disk_meta: WicksDiskMeta = bincode::deserialize(&std::fs::read(&meta_path).unwrap()).unwrap();
            let file = File::open(&slab_path).unwrap();
            let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
            let ptr = mmap.as_ptr() as *mut f64;

            let slab = NonNull::new(ptr).expect("Should not be null.");
            let view = WicksView {slab, slab_len: disk_meta.slab_len, nref: disk_meta.nref, off: disk_meta.off, meta: disk_meta.meta,};

            WicksShared {backing: WicksBacking::Mmap(mmap), view}
        }
    }
}

/// Update the Wick's intermediates required for fast Fock matrix element evaluation, as these
/// intermediates change per iteration of SNOCI.
/// # Arguments:
/// - `fa`: Fock matrix spin alpha.
/// - `fb`: Fock matrix spin beta. 
/// - `noci_reference_basis`: Vector of only the reference determinants.
/// - `wicks`: Shared memory Wick's intermediates storage.
/// # Returns:
/// - `()`: Updates the stored Fock-related Wick's intermediates in `wicks` in place.
pub fn update_wicks_fock(fa: &Array2<f64>, fb: &Array2<f64>, noci_reference_basis: &[SCFState], wicks: &mut WicksShared) {
    let nref = noci_reference_basis.len();

    for i in 0..nref {
        let ri = &noci_reference_basis[i];
        for j in 0..nref {
            let idx = i * nref + j;

            let (xa, ya, xb, yb, off_aa, off_bb) = {
                let view = wicks.view();
                let pair = view.pair(i, j);

                let xa = [pair.aa.x(0).to_owned(), pair.aa.x(1).to_owned()];
                let ya = [pair.aa.y(0).to_owned(), pair.aa.y(1).to_owned()];
                let xb = [pair.bb.x(0).to_owned(), pair.bb.x(1).to_owned()];
                let yb = [pair.bb.y(0).to_owned(), pair.bb.y(1).to_owned()];

                let off_aa = view.off[idx].aa;
                let off_bb = view.off[idx].bb;

                (xa, ya, xb, yb, off_aa, off_bb)
            };

            let (f0_0fa, f00fa) = SameSpinBuild::construct_f(&ri.ca, fa, &xa[0], &ya[0]);
            let (_,       f01fa) = SameSpinBuild::construct_f(&ri.ca, fa, &xa[0], &ya[1]);
            let (_,       f10fa) = SameSpinBuild::construct_f(&ri.ca, fa, &xa[1], &ya[0]);
            let (f0_1fa, f11fa) = SameSpinBuild::construct_f(&ri.ca, fa, &xa[1], &ya[1]);
            let f0fa: [f64; 2] = [f0_0fa, f0_1fa];
            let ffa: [[Array2<f64>; 2]; 2] = [[f00fa, f01fa], [f10fa, f11fa]];

            let (f0_0fb, f00fb) = SameSpinBuild::construct_f(&ri.cb, fb, &xb[0], &yb[0]);
            let (_,       f01fb) = SameSpinBuild::construct_f(&ri.cb, fb, &xb[0], &yb[1]);
            let (_,       f10fb) = SameSpinBuild::construct_f(&ri.cb, fb, &xb[1], &yb[0]);
            let (f0_1fb, f11fb) = SameSpinBuild::construct_f(&ri.cb, fb, &xb[1], &yb[1]);
            let f0fb: [f64; 2] = [f0_0fb, f0_1fb];
            let ffb: [[Array2<f64>; 2]; 2] = [[f00fb, f01fb], [f10fb, f11fb]];

            let view = wicks.view_mut();
            view.meta[idx].aa.f0f = f0fa;
            view.meta[idx].bb.f0f = f0fb;
            let slab = wicks.slab_mut();
            for mi in 0..2 {
                for mj in 0..2 {
                    write2(slab, off_aa.ff[mi][mj], &ffa[mi][mj]);
                    write2(slab, off_bb.ff[mi][mj], &ffb[mi][mj]);
                }
            }
        }
    }
}

