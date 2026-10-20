// elements/nonorthogonalwicks/shared.rs

// Standard library imports.
use std::fs::{self};
use std::path::PathBuf;
use std::ptr::NonNull;

// External crate imports.
use mpi::topology::Communicator;
use ndarray::Array2;

// Crate-root imports.
use crate::AoData;
use crate::NOCIScalar;
use crate::determinant::ParentDeterminant;
use crate::elements::nonorthogonalwicks::types::{PairBranches, PairOffset};
use crate::elements::nonorthogonalwicks::{
    DiffSpinBuild, DiffSpinMeta, PairMeta, PairZeroCounts, SameSpinBuild, SameSpinMeta,
    WICKS_DISK_CACHE_VERSION, WicksDiskMeta, WicksRma, WicksShared, WicksView,
};
use crate::elements::nonorthogonalwicks::{
    assign_offsets, create_wicks_mmap, load_wicks_mmap, write_diff_spin, write_hcol0,
    write_same_spin, write2t,
};
use crate::input::Input;
use crate::input::Spin;
use crate::mpiutils::Sharedffi;
use crate::mpiutils::broadcast;

/// Build all Wick's intermediates and metadata for one reference pair and write the
/// intermediates into the tensor slab. Every near-zero corner of each spin sector is built once,
/// and every combination of alpha- and beta-spin corners receives its own different-spin
/// intermediates and metadata.
/// # Arguments:
/// - `slab`: Contiguous tensor storage.
/// - `off`: Offset tables of every ordinary pair followed by those of the branch combinations.
/// - `branches`: Location of the branch combinations of every ordinary pair.
/// - `idx`: Flattened index `i * nref + j` of the ordered reference pair.
/// - `ao`: Contains AO integrals and other system data.
/// - `parents`: Reference determinants.
/// - `tol`: Tolerance for a number being zero.
/// # Returns
/// - `Vec<(usize, PairMeta<T>)>`: Metadata of every branch combination with its pair index.
fn build_wicks_pair_into<T: NOCIScalar>(
    slab: &mut [T],
    off: &[PairOffset],
    branches: &[PairBranches],
    idx: usize,
    ao: &AoData,
    parents: &[ParentDeterminant<T>],
    tol: f64,
) -> Vec<(usize, PairMeta<T>)> {
    let nref = parents.len();
    let (ri, rj) = (&parents[idx / nref], &parents[idx % nref]);
    let branches = &branches[idx];

    // Build `\alpha` and `\beta` contractions once per branch of each spin sector.
    let aa: Vec<_> = (0..branches.na)
        .map(|ia| SameSpinBuild::new(ao, rj, ri, Spin::Alpha, tol, ia))
        .collect();
    let bb: Vec<_> = (0..branches.nb)
        .map(|ib| SameSpinBuild::new(ao, rj, ri, Spin::Beta, tol, ib))
        .collect();
    for (ia, build) in aa.iter().enumerate() {
        write_same_spin(slab, &off[branches.index(idx, ia, 0)].aa, build);
    }
    for (ib, build) in bb.iter().enumerate() {
        write_same_spin(slab, &off[branches.index(idx, 0, ib)].bb, build);
    }

    // Build the mixed-spin contractions of every branch combination; the evaluator reads them
    // through the paired metadata below.
    let mut meta = Vec::with_capacity(branches.na * branches.nb);
    for (ia, aa) in aa.iter().enumerate() {
        for (ib, bb) in bb.iter().enumerate() {
            let k = branches.index(idx, ia, ib);
            let ab = DiffSpinBuild::new(ao, rj, ri, tol, ia, ib);
            write_diff_spin(slab, &off[k].ab, &ab);

            // The precombined one-column intermediates exist only for all-zero pairs.
            write_hcol0(slab, &off[k], aa.nmo);

            let pair = PairMeta {
                aa: SameSpinMeta {
                    tilde_s_prod: aa.tilde_s_prod,
                    phase: aa.phase,
                    m: aa.m,
                    nmo: aa.nmo,
                    nocc: ri.oa.count_ones() as usize,
                    f0h: aa.f0h,
                    f0f: aa.f0f,
                    v0: aa.v0,
                },
                bb: SameSpinMeta {
                    tilde_s_prod: bb.tilde_s_prod,
                    phase: bb.phase,
                    m: bb.m,
                    nmo: bb.nmo,
                    nocc: ri.ob.count_ones() as usize,
                    f0h: bb.f0h,
                    f0f: bb.f0f,
                    v0: bb.v0,
                },
                ab: DiffSpinMeta {
                    nmo: ab.vab[0][0][0].nrows(),
                    vab0: ab.vab0,
                    vba0: ab.vba0,
                },
            };
            meta.push((k, pair));
        }
    }
    meta
}

/// Determine pair-specific alpha and beta zero-overlap and near-zero counts needed to allocate only reachable rank-four intermediates.
/// # Arguments:
/// - `ao`: Contains AO integrals and overlap matrix.
/// - `refs`: Reference determinants in the flattened pair ordering.
/// - `tol`: Tolerance for whether a singular value is considered zero.
/// # Returns
/// - `Vec<PairZeroCounts>`: Pair-adaptive zero-overlap and near-zero counts ordered by `idx = i * nref + j`.
fn plan_wicks_pairs<T: NOCIScalar>(
    ao: &AoData,
    refs: &[ParentDeterminant<T>],
    tol: f64,
) -> Vec<PairZeroCounts> {
    let nref = refs.len();
    let mut plans = Vec::with_capacity(nref * nref);

    for ri in refs {
        for rj in refs {
            let (ma, na) = SameSpinBuild::count_zero_overlap_pairs(
                &ao.s,
                rj.ca.as_ref(),
                rj.oa,
                ri.ca.as_ref(),
                ri.oa,
                tol,
            );
            let (mb, nb) = SameSpinBuild::count_zero_overlap_pairs(
                &ao.s,
                rj.cb.as_ref(),
                rj.ob,
                ri.cb.as_ref(),
                ri.ob,
                tol,
            );
            plans.push(PairZeroCounts { ma, mb, na, nb });
        }
    }

    plans
}

/// Build the Wick's per reference-pair intermediates and store in a shared memory access region (per node).
/// This may be in RAM or on disk if requested.
/// # Arguments:
/// - `world`: MPI communicator object.
/// - `ao`: Contains AO integrals and other system data.
/// - `parents`: Selected parent orbital frames in reference order.
/// - `tol`: Tolerance for a number being zero.
/// - `input`: User input specifications.
/// # Returns:
/// - `WicksShared<T>`: Shared-memory storage and view for precomputed Wick's intermediates.
pub fn build_wicks_shared<T: NOCIScalar>(
    world: &impl Communicator,
    ao: &AoData,
    parents: &[ParentDeterminant<T>],
    tol: f64,
    input: &Input,
) -> WicksShared<T> {
    // Plan one contiguous tensor slab and immutable offsets for every parent pair.
    let nref = parents.len();
    let nmo = parents[0].ca.ncols();
    let irank = world.rank();

    let plans = plan_wicks_pairs(ao, parents, tol);
    let (offset, branches, tensor_len) = assign_offsets(&plans, nmo, ao.n);
    drop(plans);
    let nbytes = tensor_len * std::mem::size_of::<T>();

    // Report the shared storage requirement once per MPI world.
    if irank == 0 {
        println!("Number of MOs: {}", ao.n);
        println!(
            "Estimated memory required for Wick's intermediates (MiB): {}",
            nbytes as f64 / (1024.0 * 1024.0)
        );
    }

    match input.wicks.storage {
        crate::input::WicksStorage::RAM => {
            // Allocate one slab per shared-memory node; only its local leader populates it.
            let shared = Sharedffi::allocate(world, nbytes);
            let shared_rank = shared.shared_rank;

            let tensor_ptr = shared.base as *mut T;
            let mut meta = vec![PairMeta::<T>::default(); offset.len()];

            if shared_rank == 0 {
                // Build every ordered parent pair directly into its assigned slab slices.
                let tensor: &mut [T] =
                    unsafe { std::slice::from_raw_parts_mut(tensor_ptr, tensor_len) };
                tensor.fill(<T as From<f64>>::from(f64::NAN));

                for i in 0..nref {
                    for j in 0..nref {
                        println!(
                            "Building intermediates for reference pair: {}, {} on world rank {}",
                            i,
                            j,
                            world.rank()
                        );

                        let built = build_wicks_pair_into(
                            tensor,
                            &offset,
                            &branches,
                            i * nref + j,
                            ao,
                            parents,
                            tol,
                        );
                        for (k, pair) in built {
                            meta[k] = pair;
                        }
                    }
                }
            }

            // Publish completed tensor data and identical pair metadata to all ranks.
            shared.barrier();
            broadcast(world, &mut meta);

            // Couple the owning shared allocation to a typed immutable Wick view.
            let rma = WicksRma::<T> {
                base_ptr: shared.base,
                _nbytes: nbytes,
                _shared: shared,
                _marker: std::marker::PhantomData,
            };
            let slab = NonNull::new(tensor_ptr).expect("Should not be null.");
            let view = WicksView {
                slab,
                slab_len: tensor_len,
                nref,
                off: offset,
                meta,
                branches,
            };

            WicksShared::from_shared(rma, view)
        }

        crate::input::WicksStorage::Disk => {
            // Resolve stable slab and metadata paths for the file-backed cache.
            let cache_dir = PathBuf::from(
                input
                    .wicks
                    .cachedir
                    .clone()
                    .unwrap_or_else(|| ".".to_string()),
            );
            fs::create_dir_all(&cache_dir).unwrap();

            let slab_path = cache_dir.join("wicks.bin");
            let meta_path = cache_dir.join("wicks.meta");

            // Use a minimal shared allocation solely to elect and synchronize a node leader.
            let shared = Sharedffi::allocate(world, 1);
            let shared_rank = shared.shared_rank;

            if shared_rank == 0 {
                // Materialise all ordered parent pairs in the writable memory map.
                println!(
                    "Building Wick's intermediates and writing disk cache on world rank {}: {:?}",
                    world.rank(),
                    cache_dir
                );

                let mut wicks = create_wicks_mmap::<T>(
                    &slab_path,
                    nref,
                    offset.clone(),
                    branches.clone(),
                    tensor_len,
                )
                .unwrap();
                let mut meta = vec![PairMeta::<T>::default(); offset.len()];

                for i in 0..nref {
                    for j in 0..nref {
                        println!(
                            "Building intermediates for reference pair: {}, {} on world rank {}",
                            i,
                            j,
                            world.rank()
                        );

                        let built = build_wicks_pair_into(
                            wicks.slab_mut(),
                            &offset,
                            &branches,
                            i * nref + j,
                            ao,
                            parents,
                            tol,
                        );
                        for (k, pair) in built {
                            meta[k] = pair;
                        }
                    }
                }
                wicks.view_mut().meta = meta;

                // Flush numerical data before publishing the matching serialized layout metadata.
                wicks.flush_mmap().unwrap();

                let view = wicks.view();
                let disk_meta = WicksDiskMeta::<T> {
                    version: WICKS_DISK_CACHE_VERSION,
                    nref: view.nref,
                    slab_len: view.slab_len,
                    off: view.off.clone(),
                    meta: view.meta.clone(),
                    branches: view.branches.clone(),
                };
                std::fs::write(&meta_path, bincode::serialize(&disk_meta).unwrap()).unwrap();
            }

            // Open read-only views only after the node leader has completed both cache files.
            shared.barrier();
            load_wicks_mmap::<T>(&slab_path, &meta_path).unwrap()
        }
    }
}

/// Update the Wick's intermediates required for fast Fock matrix element evaluation, as these
/// intermediates change per iteration of SNOCI.
/// # Arguments:
/// - `fa`: Fock matrix spin alpha.
/// - `fb`: Fock matrix spin beta.
/// - `parents`: Selected parent orbital frames in reference order.
/// - `s`: AO overlap matrix used to update paired orbitals.
/// - `tol`: Numerical tolerance for the Wick contractions.
/// - `wicks`: Shared memory Wick's intermediates storage.
/// # Returns:
/// - `()`: Updates the stored Fock-related Wick's intermediates in `wicks` in place.
pub(crate) fn update_wicks_fock<T: NOCIScalar>(
    fa: &Array2<T>,
    fb: &Array2<T>,
    parents: &[ParentDeterminant<T>],
    s: &Array2<f64>,
    tol: f64,
    wicks: &mut WicksShared<T>,
) {
    let nref = parents.len();

    // Refresh only Fock-dependent scalars and rank-two contractions for every branch
    // combination of each ordered parent pair; overlap and Coulomb intermediates remain reusable.
    for (i, ri) in parents.iter().enumerate().take(nref) {
        for (j, rj) in parents.iter().enumerate().take(nref) {
            let idx = i * nref + j;
            let branches = wicks.view().branches[idx];

            for ia in 0..branches.na {
                for ib in 0..branches.nb {
                    let k = branches.index(idx, ia, ib);
                    let (off_aa, off_bb) = {
                        let view = wicks.view();
                        (view.off[k].aa, view.off[k].bb)
                    };

                    let (f0fa, ffa) =
                        SameSpinBuild::construct_f_scalar(s, fa, rj, ri, Spin::Alpha, tol, ia);
                    let (f0fb, ffb) =
                        SameSpinBuild::construct_f_scalar(s, fb, rj, ri, Spin::Beta, tol, ib);

                    {
                        let view = wicks.view_mut();
                        view.meta[k].aa.f0f = f0fa;
                        view.meta[k].bb.f0f = f0fb;
                    }

                    // Store all binary fundamental-contraction assignments in the
                    // existing pair slab offsets.
                    let slab = wicks.slab_mut();
                    for mi in 0..2 {
                        for mj in 0..2 {
                            write2t(slab, off_aa.ff[mi][mj], &ffa[mi][mj]);
                            write2t(slab, off_bb.ff[mi][mj], &ffb[mi][mj]);
                        }
                    }
                }
            }
        }
    }
}
