// nonorthogonalwicks/layout.rs
// External crate imports.
use ndarray::{Array2, Array4};

// Crate-root imports.
use crate::noci::NOCIScalar;

// Parent/sibling imports.
use super::build::{DiffSpinBuild, SAME_SPIN_J_BRANCHES, SameSpinBuild};
use super::types::{DiffSpinOffset, PairOffset, PairZeroCounts, SameSpinOffset};

/// Assign offsets into the contiguous tensor slab for every ordered reference pair (x,w).
/// Rank-four tensors are allocated only when their fundamental-contraction assignments can
/// `satisfy the corresponding m_\alpha or m_\beta constraint.`
/// # Arguments:
/// - `plans`: `Pair-specific values of m_\alpha and m_\beta.`
/// - `nmo`: Number of molecular orbitals in one reference orbital set.
/// - `nbas`: Dimension of the external RDM basis.
/// # Returns
/// - `(Vec<PairOffset>, usize)`: Per-pair offset tables and total slab length in units of `T`.
pub fn assign_offsets(
    plans: &[PairZeroCounts],
    nmo: usize,
    nbas: usize,
) -> (Vec<PairOffset>, usize) {
    let nn2 = nmo * nmo;
    let nn4 = nn2 * nn2;

    #[cfg(feature = "nocc")]
    let nbas2 = nbas * nbas;
    #[cfg(not(feature = "nocc"))]
    let _ = nbas;

    let mut off = vec![PairOffset::default(); plans.len()];
    let mut i: usize = 0;

    for (p, plan) in off.iter_mut().zip(plans.iter()) {
        // Allocate the alpha-spin X^{(m_i)} and Y^{(m_i)} fundamental contractions.
        for mi in 0..2 {
            p.aa.x[mi] = i;
            i += nn2;
        }
        for mi in 0..2 {
            p.aa.y[mi] = i;
            i += nn2;
        }
        #[cfg(feature = "nocc")]
        {
            // Allocate the corresponding alpha-spin contractions in the external RDM basis.
            for mi in 0..2 {
                p.aa.xrdm[mi] = i;
                i += nbas2;
            }
            for mi in 0..2 {
                p.aa.yrdm[mi] = i;
                i += nbas2;
            }
        }
        // Allocate the alpha-spin one-body \mathcal F and one-column \mathcal V intermediates.
        for mi in 0..2 {
            for mj in 0..2 {
                p.aa.fh[mi][mj] = i;
                i += nn2;
            }
        }
        for mi in 0..2 {
            for mj in 0..2 {
                p.aa.ff[mi][mj] = i;
                i += nn2;
            }
        }
        for mi in 0..2 {
            for mj in 0..2 {
                for mk in 0..2 {
                    p.aa.v[mi][mj][mk] = i;
                    i += nn2;
                }
            }
        }
        // Allocate only those symmetry-unique alpha-spin \mathcal J tensors whose four
        // assignments do not already exceed m_\alpha.
        p.aa.j.fill(usize::MAX);
        for (slot, branch) in SAME_SPIN_J_BRANCHES.iter().copied().enumerate() {
            if branch.0 + branch.1 + branch.2 + branch.3 <= plan.ma {
                p.aa.j[slot] = i;
                i += nn4;
            }
        }

        // Allocate the beta-spin X^{(m_i)} and Y^{(m_i)} fundamental contractions.
        for mi in 0..2 {
            p.bb.x[mi] = i;
            i += nn2;
        }
        for mi in 0..2 {
            p.bb.y[mi] = i;
            i += nn2;
        }
        #[cfg(feature = "nocc")]
        {
            // Allocate the corresponding beta-spin contractions in the external RDM basis.
            for mi in 0..2 {
                p.bb.xrdm[mi] = i;
                i += nbas2;
            }
            for mi in 0..2 {
                p.bb.yrdm[mi] = i;
                i += nbas2;
            }
        }
        // Allocate the beta-spin one-body \mathcal F and one-column \mathcal V intermediates.
        for mi in 0..2 {
            for mj in 0..2 {
                p.bb.fh[mi][mj] = i;
                i += nn2;
            }
        }
        for mi in 0..2 {
            for mj in 0..2 {
                p.bb.ff[mi][mj] = i;
                i += nn2;
            }
        }
        for mi in 0..2 {
            for mj in 0..2 {
                for mk in 0..2 {
                    p.bb.v[mi][mj][mk] = i;
                    i += nn2;
                }
            }
        }
        // Allocate only those symmetry-unique beta-spin \mathcal J tensors whose four
        // assignments do not already exceed m_\beta.
        p.bb.j.fill(usize::MAX);
        for (slot, branch) in SAME_SPIN_J_BRANCHES.iter().copied().enumerate() {
            if branch.0 + branch.1 + branch.2 + branch.3 <= plan.mb {
                p.bb.j[slot] = i;
                i += nn4;
            }
        }

        // Allocate the different-spin one-column intermediates \mathcal V^\alpha and
        // \mathcal V^\beta for every binary assignment of their three contractions.
        for ma0 in 0..2 {
            for mb0 in 0..2 {
                for mk in 0..2 {
                    p.ab.vab[ma0][mb0][mk] = i;
                    i += nn2;
                }
            }
        }
        for mb0 in 0..2 {
            for ma0 in 0..2 {
                for mk in 0..2 {
                    p.ab.vba[mb0][ma0][mk] = i;
                    i += nn2;
                }
            }
        }
        // Allocate only those \mathcal{II} tensors whose alpha- and beta-spin assignments
        // can satisfy their independent constraints.
        p.ab.iiab = [[[[usize::MAX; 2]; 2]; 2]; 2];
        for ma0 in 0..2 {
            for maj in 0..2 {
                for mb0 in 0..2 {
                    for mbj in 0..2 {
                        if ma0 + maj <= plan.ma && mb0 + mbj <= plan.mb {
                            p.ab.iiab[ma0][maj][mb0][mbj] = i;
                            i += nn4;
                        }
                    }
                }
            }
        }
    }

    (off, i)
}

/// `Write the same-spin fundamental contractions and \mathcal F, \mathcal V and \mathcal J`
/// intermediates into the contiguous tensor slab using their assigned offsets.
/// # Arguments:
/// - `slab`: Contiguous tensor storage.
/// - `o`: Same-spin offsets into the slab.
/// - `w`: Owned same-spin intermediates for one reference pair and spin sector.
/// # Returns
/// - `()`: Writes the same-spin intermediates into the slab.
pub fn write_same_spin<T: NOCIScalar>(
    slab: &mut [T],
    o: &SameSpinOffset,
    w: &SameSpinBuild<T>,
) {
    write2(slab, o.x[0], &w.x[0]);
    write2(slab, o.x[1], &w.x[1]);
    write2(slab, o.y[0], &w.y[0]);
    write2(slab, o.y[1], &w.y[1]);
    #[cfg(feature = "nocc")]
    {
        write2(slab, o.xrdm[0], &w.xrdm[0]);
        write2(slab, o.xrdm[1], &w.xrdm[1]);
        write2(slab, o.yrdm[0], &w.yrdm[0]);
        write2(slab, o.yrdm[1], &w.yrdm[1]);
    }

    // Transpose the one-column matrices so a replacement column with fixed orbital label is contiguous.
    for mi in 0..2 {
        for mj in 0..2 {
            write2t(slab, o.fh[mi][mj], &w.fh[mi][mj]);
        }
    }
    for mi in 0..2 {
        for mj in 0..2 {
            write2t(slab, o.ff[mi][mj], &w.ff[mi][mj]);
        }
    }
    for mi in 0..2 {
        for mj in 0..2 {
            for mk in 0..2 {
                write2t(slab, o.v[mi][mj][mk], &w.v[mi][mj][mk]);
            }
        }
    }
    // Reorder each \mathcal J tensor so the fixed replacement pair precedes the varying pair.
    for (slot, blk) in &w.j {
        write4ijrc(slab, o.j[*slot], blk);
    }
}

/// `Write the different-spin \mathcal V^\alpha, \mathcal V^\beta and \mathcal{II}`
/// intermediates into the contiguous tensor slab using their assigned offsets.
/// # Arguments:
/// - `slab`: Contiguous tensor storage.
/// - `o`: Different-spin offsets into the slab.
/// - `w`: Owned different-spin intermediates for one reference pair.
/// # Returns
/// - `()`: Writes the different-spin intermediates into the slab.
pub fn write_diff_spin<T: NOCIScalar>(
    slab: &mut [T],
    o: &DiffSpinOffset,
    w: &DiffSpinBuild<T>,
) {
    for ma0 in 0..2 {
        for mb0 in 0..2 {
            for mk in 0..2 {
                write2t(slab, o.vab[ma0][mb0][mk], &w.vab[ma0][mb0][mk]);
            }
        }
    }
    for mb0 in 0..2 {
        for ma0 in 0..2 {
            for mk in 0..2 {
                write2t(slab, o.vba[mb0][ma0][mk], &w.vba[mb0][ma0][mk]);
            }
        }
    }
    // The constructed and evaluator \mathcal{II} axis orders agree, so no permutation is required.
    for ((ma0, maj, mb0, mbj), blk) in &w.iiab {
        write4rcij(slab, o.iiab[*ma0][*maj][*mb0][*mbj], blk);
    }
}

/// Copy a contiguous matrix into the tensor slab without changing its row-major ordering.
/// # Arguments:
/// - `slab`: Contiguous tensor storage.
/// - `off`: Offset of the first matrix entry in units of `T`.
/// - `a`: Matrix to copy.
/// # Returns
/// - `()`: Writes the matrix into the tensor slab.
pub fn write2<T: NOCIScalar>(
    slab: &mut [T],
    off: usize,
    a: &Array2<T>,
) {
    let src = a.as_slice().expect("Array2 must be contiguous");
    slab[off..off + src.len()].copy_from_slice(src);
}

/// Store a matrix constructed in `[r,z]` order in transposed `[z,r]` order. The entries of a
/// replacement column with fixed z are then contiguous and may be read as `slice[z * n + r]`.
/// # Arguments:
/// - `slab`: Contiguous tensor storage.
/// - `off`: Offset of the first stored matrix entry in units of `T`.
/// - `a`: Matrix to transpose into the slab.
/// # Returns
/// - `()`: Writes the transposed matrix into the tensor slab.
pub fn write2t<T: NOCIScalar>(
    slab: &mut [T],
    off: usize,
    a: &Array2<T>,
) {
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

/// Copy a rank-four tensor already constructed in `[r,c,i,j]` order into the slab without
/// `permuting its axes. This is the ordering used by the different-spin \mathcal{II} evaluator.`
/// # Arguments:
/// - `slab`: Contiguous tensor storage.
/// - `off`: Offset of the first tensor entry in units of `T`.
/// - `a`: Tensor to copy.
/// # Returns
/// - `()`: Writes the tensor into the tensor slab.
fn write4rcij<T: NOCIScalar>(
    slab: &mut [T],
    off: usize,
    a: &Array4<T>,
) {
    let src = a.as_slice().expect("Array4 must be contiguous");
    slab[off..off + src.len()].copy_from_slice(src);
}

/// Permute a rank-four tensor from constructed `[r,c,i,j]` order to stored `[i,j,r,c]` order.
/// `This places the fixed \mathcal J replacement pair before the varying minor entry.`
/// # Arguments:
/// - `slab`: Contiguous tensor storage.
/// - `off`: Offset of the first stored tensor entry in units of `T`.
/// - `a`: Tensor to permute into the slab.
/// # Returns
/// - `()`: Writes the permuted tensor into the tensor slab.
fn write4ijrc<T: NOCIScalar>(
    slab: &mut [T],
    off: usize,
    a: &Array4<T>,
) {
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
pub(in crate::nonorthogonalwicks) fn idx(
    ncols: usize,
    r: usize,
    c: usize,
) -> usize {
    r * ncols + c
}

/// `Convert four tensor indices into the row-major flat index of an n \times n \times n \times n tensor.`
/// # Arguments:
/// - `n`: Dimension of each tensor axis.
/// - `a`, `b`, `c`, `d`: Tensor indices.
/// # Returns
/// - `usize`: Flat row-major index corresponding to `(a, b, c, d)`.
#[inline(always)]
pub(in crate::nonorthogonalwicks) fn idx4(
    n: usize,
    a: usize,
    b: usize,
    c: usize,
    d: usize,
) -> usize {
    (((a * n + b) * n + c) * n) + d
}
