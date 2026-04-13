// nonorthogonalwicks/layout.rs
use super::types::{PairOffset, SameSpinOffset, DiffSpinOffset};
use super::build::{SameSpinBuild, DiffSpinBuild};

/// Calculate and store all the required offsets from the beginning of the large contiguous shared
/// tensor storage to a given matrix or tensor.
/// # Arguments:
/// - `nref`: Number of references.
/// - `nmo`: Number of molecular orbitals.
/// # Returns
/// - `(Vec<PairOffset>, usize)`: Per-pair offset table and total slab length in units of `f64`.
pub fn assign_offsets(nref: usize, nmo: usize) -> (Vec<PairOffset>, usize) {
    
    let n = 2 * nmo;
    let nn2 = n * n;
    let nn4 = n * n * n * n;
    let mut off = vec![PairOffset::default(); nref * nref];
    let mut i: usize = 0;
    
    // For each reference-pair we have n by n f64s or n by n by n by n f64s.
    // For every pair p we assign all offset locations for all required tensors.
    for p in off.iter_mut() {
        for mi in 0..2 {p.aa.x[mi] = i; i += nn2;}
        for mi in 0..2 {p.aa.y[mi] = i; i += nn2;}
        for mi in 0..2 { for mj in 0..2 {p.aa.fh[mi][mj] = i; i += nn2;}}
        for mi in 0..2 { for mj in 0..2 {p.aa.ff[mi][mj] = i; i += nn2;}}
        for mi in 0..2 {for mj in 0..2 {for mk in 0..2 {p.aa.v[mi][mj][mk] = i; i += nn2;}}}
        for s in 0..10 {p.aa.j[s] = i; i += nn4;}

        for mi in 0..2 {p.bb.x[mi] = i; i += nn2;}
        for mi in 0..2 {p.bb.y[mi] = i; i += nn2;}
        for mi in 0..2 {for mj in 0..2 {p.bb.fh[mi][mj] = i; i += nn2;}}
        for mi in 0..2 {for mj in 0..2 {p.bb.ff[mi][mj] = i; i += nn2;}}
        for mi in 0..2 {for mj in 0..2 {for mk in 0..2 {p.bb.v[mi][mj][mk] = i; i += nn2;}}}
        for s in 0..10 {p.bb.j[s] = i; i += nn4;}

        for ma0 in 0..2 {for mb0 in 0..2 {for mk in 0..2 {p.ab.vab[ma0][mb0][mk] = i; i += nn2;}}}
        for mb0 in 0..2 {for ma0 in 0..2 {for mk in 0..2 {p.ab.vba[mb0][ma0][mk] = i; i += nn2;}}}

        for ma0 in 0..2 {for maj in 0..2 {for mb0 in 0..2 {for mbj in 0..2 {p.ab.iiab[ma0][maj][mb0][mbj] = i; i += nn4;}}}}
    }
    (off, i) 
}

/// Fill the same-spin data owning structs with the same-spin Wick's intermediates using the
/// prescribed offsets.
/// # Arguments:
/// - `slab`: Contiguous tensor storage.
/// - `o`: Offsets into the storage.
/// - `w`: Owned Wick's intermediates.
/// # Returns
/// - `()`: Writes the same-spin intermediates into the slab.
pub fn write_same_spin(slab: &mut [f64], o: &SameSpinOffset, w: &SameSpinBuild) {
    write2(slab, o.x[0], &w.x[0]);
    write2(slab, o.x[1], &w.x[1]);
    write2(slab, o.y[0], &w.y[0]);
    write2(slab, o.y[1], &w.y[1]);
    for mi in 0..2 {for mj in 0..2 {write2t(slab, o.fh[mi][mj], &w.fh[mi][mj]);}}
    for mi in 0..2 {for mj in 0..2 {write2t(slab, o.ff[mi][mj], &w.ff[mi][mj]);}}
    for mi in 0..2 {for mj in 0..2 {for mk in 0..2 {write2t(slab, o.v[mi][mj][mk], &w.v[mi][mj][mk]);}}}
    for s in 0..10 {write4ijrc(slab, o.j[s], &w.j[s]);}
}

/// Fill the diff-spin data owning structs with the diff-spin Wick's intermediates using the
/// prescribed offsets.
/// # Arguments:
/// - `slab`: Contiguous tensor storage.
/// - `o`: Offsets into the storage.
/// - `w`: Owned Wick's intermediates.
/// # Returns
/// - `()`: Writes the diff-spin intermediates into the slab.
pub fn write_diff_spin(slab: &mut [f64], o: &DiffSpinOffset, w: &DiffSpinBuild) {
    for ma0 in 0..2 {for mb0 in 0..2 {for mk in 0..2 {write2t(slab, o.vab[ma0][mb0][mk], &w.vab[ma0][mb0][mk]);}}}
    for mb0 in 0..2 {for ma0 in 0..2 {for mk in 0..2 {write2t(slab, o.vba[mb0][ma0][mk], &w.vba[mb0][ma0][mk]);}}}
    for ma0 in 0..2 {for maj in 0..2 {for mb0 in 0..2 {for mbj in 0..2 {write4rcij(slab, o.iiab[ma0][maj][mb0][mbj], &w.iiab[ma0][maj][mb0][mbj]);}}}}
}

/// Copy matrix into tensor slab provided it is contiguous.
/// # Arguments:
/// - `slab`: Contiguous tensor storage.
/// - `off`: Offset for the start position.
/// - `a`: Matrix to copy.
/// # Returns
/// - `()`: Writes the matrix into the tensor slab.
pub fn write2(slab: &mut [f64], off: usize, a: &ndarray::Array2<f64>) {
    let src = a.as_slice().expect("Array2 must be contiguous");
    slab[off..off + src.len()].copy_from_slice(src);
}

/// Copy transpoed matrix into tensor slab provided it is contiguous.
/// # Arguments:
/// - `slab`: Contiguous tensor storage.
/// - `off`: Offset for the start position.
/// - `a`: Matrix to copy.
/// # Returns
/// - `()`: Writes the matrix into the tensor slab.
pub fn write2t(slab: &mut [f64], off: usize, a: &ndarray::Array2<f64>) {
    let (nr, nc) = a.dim();
    let src = a.as_slice().expect("Array2 must be contiguous");
    let dst = &mut slab[off..off + nr * nc];
    for r in 0..nr {
        let src_row = &src[r * nc..(r + 1) * nc];
        for c in 0..nc {
            dst[c * nr + r] = src_row[c];
        }
    }
}

/// Copy tensor in [r, c, i, j] form into tensor slab provided it is contiguous.
/// # Arguments:
/// - `slab`: Contiguous tensor storage.
/// - `off`: Offset for the start position.
/// - `a`: Tensor to copy.
/// # Returns
/// - `()`: Writes the tensor into the tensor slab.
fn write4rcij(slab: &mut [f64], off: usize, a: &ndarray::Array4<f64>) {
    let src = a.as_slice().expect("Array4 must be contiguous");
    slab[off..off + src.len()].copy_from_slice(src);
}

/// Copy tensor in [i, j, r, c] form into tensor slab provided it is contiguous.
/// # Arguments:
/// - `slab`: Contiguous tensor storage.
/// - `off`: Offset for the start position.
/// - `a`: Tensor to copy.
/// # Returns
/// - `()`: Writes the tensor into the tensor slab.
fn write4ijrc(slab: &mut [f64], off: usize, a: &ndarray::Array4<f64>) {
    let sh = a.shape();
    let nr = sh[0];
    let nc = sh[1];
    let ni = sh[2];
    let nj = sh[3];

    let src = a.as_slice().expect("Array4 must be contiguous");
    let dst = &mut slab[off..off + src.len()];

    for r in 0..nr {
        for c in 0..nc {
            for i in 0..ni {
                for j in 0..nj {
                    let src_idx = ((r * nc + c) * ni + i) * nj + j;
                    let dst_idx = ((i * nj + j) * nr + r) * nc + c;
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
}


/// Convert row and column indices into a row-major flat index.
/// # Arguments:
/// - `ncols`: Number of columns in the flattened matrix.
/// - `r`: Row index.
/// - `c`: Column index.
/// # Returns
/// - `usize`: Flat row-major index corresponding to `(r, c)`.
#[inline(always)]
pub(in crate::nonorthogonalwicks) fn idx(ncols: usize, r: usize, c: usize) -> usize {
    r * ncols + c
}

/// Convert 4D indices into a flat row-major index for an `n` x `n` x `n` x n` tensor.
/// # Arguments:
/// - `n`: Dimension of each tensor axis.
/// - `a`, `b`, `c`, `d`: Tensor indices.
/// # Returns
/// - `usize`: Flat row-major index corresponding to `(a, b, c, d)`.
#[inline(always)]
pub(in crate::nonorthogonalwicks) fn idx4(n: usize, a: usize, b: usize, c: usize, d: usize) -> usize {
    (((a * n + b) * n + c) * n) + d
}

