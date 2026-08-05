// nonorthogonalwicks/eval/helpers.rs

#[cfg(feature = "nocc")]
use ndarray::{Array2, ArrayView2, s};

#[cfg(feature = "nocc")]
use crate::maths::adjoint;
use crate::maths::{
    adjugate_transpose, det, minor as build_minor, minor_adjugate_transpose, mix_columns,
};
use crate::noci::NOCIScalar;
use crate::time_call;

use super::super::layout::{idx, idx4};
use super::super::scratch::{Vec2, WickScratch};
use super::super::view::{SameSpinView, WicksPairView};

/// Orbital-index layout used to construct \mathcal J and \mathcal{II} replacement columns.
/// The determinant row and column positions are mapped through `rows` and `cols` to the
/// corresponding orbital labels carried by the stored intermediates.
#[derive(Clone, Copy)]
pub(super) struct ReplacementLayout<'a> {
    /// Orbital dimension used to flatten each intermediate-tensor axis.
    pub n: usize,
    /// Orbital label associated with each row of the full contraction determinant.
    pub rows: &'a [usize],
    /// Orbital label associated with each column of the full contraction determinant.
    pub cols: &'a [usize],
}

/// Pair of row and column indices used to identify a contraction-determinant entry or a fixed
/// pair of orbital labels, as specified by the calling replacement helper.
#[derive(Clone, Copy)]
pub(super) struct DetIndex {
    /// Row index or row orbital label.
    pub row: usize,
    /// Column index or column orbital label.
    pub col: usize,
}

/// All-m_i = 0 and all-m_i = 1 endpoint contraction determinants. A mixed
/// \mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L) is formed by selecting column i from the endpoint
/// corresponding to m_i \in \{0,1\}.
#[derive(Clone, Copy)]
pub(super) struct DetBranches<'a, T> {
    /// \mathbf D_{\mathrm{ov}}(0,\ldots,0).
    pub zero: &'a [T],
    /// \mathbf D_{\mathrm{ov}}(1,\ldots,1).
    pub one: &'a [T],
}

/// Specification of the minor \mathbf D[\eta|z] obtained by deleting one row and one column
/// from a full contraction determinant.
#[derive(Clone, Copy)]
pub(super) struct Minor {
    /// Dimension L of the full contraction determinant.
    pub l: usize,
    /// Row \eta removed from the full determinant.
    pub row: usize,
    /// Column z removed from the full determinant.
    pub col: usize,
}

/// Evaluate \det\mathbf A for a row-major square matrix.
/// # Arguments:
/// - `a`: Row-major entries of \mathbf A.
/// - `n`: Dimension of \mathbf A.
/// # Returns
/// - `Option<T>`: \det\mathbf A when the determinant routine succeeds.
#[inline(always)]
pub(super) fn det_slice<T: NOCIScalar>(
    a: &[T],
    n: usize,
) -> Option<T> {
    det(a, n)
}

/// Evaluate \det\mathbf A for a row-major square matrix, returning zero when the determinant
/// routine does not produce a value.
/// # Arguments:
/// - `a`: Row-major entries of \mathbf A.
/// - `n`: Dimension of \mathbf A.
/// # Returns
/// - `T`: \det\mathbf A, or zero when evaluation fails.
#[inline(always)]
pub(super) fn det_or_zero<T: NOCIScalar>(
    a: &[T],
    n: usize,
) -> T {
    det_slice(a, n).unwrap_or(<T as From<f64>>::from(0.0))
}

/// Extend one fundamental-contraction matrix to include the external RDM basis:
/// \mathbf D_{\mathrm{ext}}^{(m)}
/// = \begin{pmatrix}\mathbf D^{(m)} & \mathbf C_{\mathrm{row}}^\dagger
/// \mathbf D_{\mathrm{RDM}}^{(m)}\\\mathbf D_{\mathrm{RDM}}^{(m)}
/// \mathbf C_{\mathrm{col}} & \mathbf D_{\mathrm{RDM}}^{(m)}\end{pmatrix}.
/// The upper-left block acts in the compact excitation row and column spaces, the lower-right
/// block contracts two external RDM indices, and the off-diagonal blocks contract one external
/// index with one excitation-space index.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `d`: Fundamental-contraction matrix \mathbf D^{(m)} in the compact excitation spaces.
/// - `d_rdm`: Fundamental-contraction matrix \mathbf D_{\mathrm{RDM}}^{(m)} in the external RDM basis.
/// - `l_c`: Bra-reference molecular-orbital coefficients in the external RDM basis.
/// - `g_c`: Ket-reference molecular-orbital coefficients in the external RDM basis.
/// # Returns
/// - `Array2<T>`: Extended fundamental-contraction matrix used to build augmented RDM determinants.
#[inline(always)]
#[cfg(feature = "nocc")]
pub(super) fn extend_rdm_d<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    d: &ArrayView2<'_, T>,
    d_rdm: &ArrayView2<'_, T>,
    l_c: &Array2<T>,
    g_c: &Array2<T>,
) -> Array2<T> {
    // Append an external block of dimension n_{\mathrm{RDM}} to the compact n_{\mathrm{mo}}
    // contraction matrix.
    let nrdm = l_c.nrows();
    let mut out = Array2::<T>::zeros((w.nmo + nrdm, w.nmo + nrdm));

    // Construct the transformations connecting the external RDM basis to the compact row and
    // column orbital spaces.
    let (row_c, col_c) = contraction_orbitals(l_c, g_c, w.nocc);

    // Form the external-row/compact-column and compact-row/external-column contraction blocks:
    // \mathbf D_{\mathrm{RDM}}^{(m)}\mathbf C_{\mathrm{col}} and
    // \mathbf C_{\mathrm{row}}^\dagger\mathbf D_{\mathrm{RDM}}^{(m)}.
    let rdm_rows = d_rdm.dot(&col_c);
    let rdm_cols = adjoint(&row_c).dot(d_rdm);

    // Assemble the compact, cross-contraction and external-basis blocks of
    // \mathbf D_{\mathrm{ext}}^{(m)}.
    out.slice_mut(s![0..w.nmo, 0..w.nmo]).assign(d);

    out.slice_mut(s![w.nmo..w.nmo + nrdm, 0..w.nmo])
        .assign(&rdm_rows);

    out.slice_mut(s![0..w.nmo, w.nmo..w.nmo + nrdm])
        .assign(&rdm_cols);

    out.slice_mut(s![w.nmo..w.nmo + nrdm, w.nmo..w.nmo + nrdm])
        .assign(d_rdm);

    out
}

/// Map the four fundamental-contraction assignments of
/// \mathcal J_{\eta z,\xi y}^{(m_i,m_j,m_k,m_l)} to one of the ten stored tensors. Pair-exchange
/// symmetry identifies
/// \mathcal J_{\eta z,\xi y}^{(m_i,m_j,m_k,m_l)}
/// = \mathcal J_{\xi y,\eta z}^{(m_k,m_l,m_i,m_j)};
/// the returned flag records whether this exchanged access is required.
/// # Arguments:
/// - `mi`: First assignment associated with the pair (\eta,z).
/// - `mj`: Second assignment associated with the pair (\eta,z).
/// - `mk`: First assignment associated with the pair (\xi,y).
/// - `ml`: Second assignment associated with the pair (\xi,y).
/// # Returns
/// - `(usize, bool)`: Stored \mathcal J slot and whether the two index pairs must be exchanged.
pub(super) fn jslot(
    mi: usize,
    mj: usize,
    mk: usize,
    ml: usize,
) -> (usize, bool) {
    // Reduce the sixteen binary assignments to ten symmetry-unique \mathcal J tensors.
    match (mi, mj, mk, ml) {
        (0, 0, 0, 0) => (0, false),
        (0, 0, 0, 1) => (1, false),
        (0, 1, 0, 0) => (1, true),
        (0, 0, 1, 0) => (2, false),
        (1, 0, 0, 0) => (2, true),
        (0, 0, 1, 1) => (3, false),
        (1, 1, 0, 0) => (3, true),
        (0, 1, 0, 1) => (4, false),
        (0, 1, 1, 0) => (5, false),
        (1, 0, 0, 1) => (5, true),
        (0, 1, 1, 1) => (6, false),
        (1, 1, 0, 1) => (6, true),
        (1, 0, 1, 0) => (7, false),
        (1, 0, 1, 1) => (8, false),
        (1, 1, 1, 0) => (8, true),
        (1, 1, 1, 1) => (9, false),
        _ => unreachable!(),
    }
}

/// Read one entry of the \mathcal J replacement column used in the same-spin two-body term.
/// For fixed (\eta,z), the minor indices are mapped back to the full determinant to obtain
/// (\xi,y), giving either \mathcal J_{\eta z,\xi y} or its pair-exchanged stored form.
/// # Arguments:
/// - `jsl`: Flattened storage of the selected \mathcal J tensor.
/// - `layout`: Orbital dimension and full determinant row and column labels.
/// - `removed`: Row \eta and column z removed to form \mathbf D_{\mathrm{ov}}[\eta|z].
/// - `minor`: Row and column indices (\xi,y) within the resulting minor.
/// - `fixed`: Orbital labels of the fixed pair (\eta,z).
/// - `swap`: Whether the stored tensor is accessed as \mathcal J_{\xi y,\eta z}.
/// # Returns
/// - `T`: Required \mathcal J entry for the replacement column.
#[inline(always)]
pub(super) fn j_replacement<T: NOCIScalar>(
    jsl: &[T],
    layout: ReplacementLayout<'_>,
    removed: DetIndex,
    minor: DetIndex,
    fixed: DetIndex,
    swap: bool,
) -> T {
    // Restore the full-determinant row and column positions of the minor entry, then map them
    // to the corresponding orbital labels (\xi,y).
    let r_full = minor_to_full(minor.row, removed.row);
    let k_full = minor_to_full(minor.col, removed.col);
    let rr = layout.rows[r_full];
    let cc = layout.cols[k_full];

    // Read \mathcal J_{\eta z,\xi y}, or the pair-exchanged
    // \mathcal J_{\xi y,\eta z} required by the compressed storage.
    if !swap {
        jsl[idx4(layout.n, fixed.row, fixed.col, rr, cc)]
    } else {
        jsl[idx4(layout.n, rr, cc, fixed.row, fixed.col)]
    }
}

/// Read one entry of the \mathcal{II} replacement column used in the different-spin two-body term.
/// The helper combines one fixed spin-sector pair with one row and column from the other spin-sector
/// contraction determinant.
/// # Arguments:
/// - `iisl`: Flattened storage of the selected \mathcal{II} tensor.
/// - `layout`: Orbital dimension and determinant row and column labels for the varying spin sector.
/// - `entry`: Row and column positions of the varying determinant entry.
/// - `fixed`: Orbital labels of the fixed pair from the other spin sector.
/// - `ijrc`: Selects \mathcal{II}_{\mathrm{fixed},\mathrm{entry}} when true and
///   \mathcal{II}_{\mathrm{entry},\mathrm{fixed}} when false.
/// # Returns
/// - `T`: Required \mathcal{II} entry for the replacement column.
#[inline(always)]
pub(super) fn ii_replacement<T: NOCIScalar>(
    iisl: &[T],
    layout: ReplacementLayout<'_>,
    entry: DetIndex,
    fixed: DetIndex,
    ijrc: bool,
) -> T {
    // Map the varying determinant position to its orbital pair.
    let rr = layout.rows[entry.row];
    let cc = layout.cols[entry.col];

    // Preserve the alpha-beta or beta-alpha pair ordering required by the caller.
    if ijrc {
        iisl[idx4(layout.n, fixed.row, fixed.col, rr, cc)]
    } else {
        iisl[idx4(layout.n, rr, cc, fixed.row, fixed.col)]
    }
}

/// Map a row or column index of \mathbf D[\eta|z] back to the corresponding index of the full
/// contraction determinant \mathbf D.
/// # Arguments:
/// - `midx`: Index in the minor.
/// - `removed`: Index removed from the full matrix.
/// # Returns
/// - `usize`: Corresponding full-matrix index.
#[inline(always)]
pub(super) fn minor_to_full(
    midx: usize,
    removed: usize,
) -> usize {
    // Indices at or beyond the removed position are displaced by one in the full matrix.
    if midx < removed { midx } else { midx + 1 }
}

/// Extract one fundamental-contraction assignment from a packed distribution. Bit k represents
/// m_k, with an unset bit selecting m_k = 0 and a set bit selecting m_k = 1.
/// # Arguments:
/// - `bits`: Packed assignments (m_0,m_1,\ldots).
/// - `k`: Assignment index.
/// # Returns
/// - `usize`: m_k \in \{0,1\}.
#[inline(always)]
pub(super) fn bit(
    bits: u64,
    k: usize,
) -> usize {
    ((bits >> k) & 1) as usize
}

/// Evaluate the change in a determinant after replacing column c of \mathbf D by \mathbf N:
/// \Delta_c = \det\mathbf D[c\rightarrow\mathbf N]-\det\mathbf D
/// = \sum_r(N_r-D_{rc})\operatorname{cof}[\mathbf D]_{rc}.
/// Callers therefore obtain the replacement determinant as \det\mathbf D+\Delta_c.
/// # Arguments:
/// - `n`: Dimension of \mathbf D.
/// - `old`: Row-major entries of \mathbf D.
/// - `cof`: Row-major cofactor matrix \operatorname{cof}[\mathbf D].
/// - `col`: Replaced column c.
/// - `new_at`: Returns N_r for row r.
/// # Returns
/// - `T`: Determinant correction \Delta_c.
#[inline(always)]
pub(super) fn column_replacement_correction<T: NOCIScalar>(
    n: usize,
    old: &[T],
    cof: &[T],
    col: usize,
    mut new_at: impl FnMut(usize) -> T,
) -> T {
    // Contract the difference between the new and original columns with the cofactors of column c.
    match (n, col) {
        (0, _) => return <T as From<f64>>::from(0.0),
        (1, 0) => return (new_at(0) - old[0]) * cof[0],
        (2, 0) => {
            return (new_at(0) - old[0]) * cof[0] + (new_at(1) - old[2]) * cof[2];
        }
        (2, 1) => {
            return (new_at(0) - old[1]) * cof[1] + (new_at(1) - old[3]) * cof[3];
        }
        (3, 0) => {
            return (new_at(0) - old[0]) * cof[0]
                + (new_at(1) - old[3]) * cof[3]
                + (new_at(2) - old[6]) * cof[6];
        }
        (3, 1) => {
            return (new_at(0) - old[1]) * cof[1]
                + (new_at(1) - old[4]) * cof[4]
                + (new_at(2) - old[7]) * cof[7];
        }
        (3, 2) => {
            return (new_at(0) - old[2]) * cof[2]
                + (new_at(1) - old[5]) * cof[5]
                + (new_at(2) - old[8]) * cof[8];
        }
        (4, 0) => {
            return (new_at(0) - old[0]) * cof[0]
                + (new_at(1) - old[4]) * cof[4]
                + (new_at(2) - old[8]) * cof[8]
                + (new_at(3) - old[12]) * cof[12];
        }
        (4, 1) => {
            return (new_at(0) - old[1]) * cof[1]
                + (new_at(1) - old[5]) * cof[5]
                + (new_at(2) - old[9]) * cof[9]
                + (new_at(3) - old[13]) * cof[13];
        }
        (4, 2) => {
            return (new_at(0) - old[2]) * cof[2]
                + (new_at(1) - old[6]) * cof[6]
                + (new_at(2) - old[10]) * cof[10]
                + (new_at(3) - old[14]) * cof[14];
        }
        (4, 3) => {
            return (new_at(0) - old[3]) * cof[3]
                + (new_at(1) - old[7]) * cof[7]
                + (new_at(2) - old[11]) * cof[11]
                + (new_at(3) - old[15]) * cof[15];
        }
        _ => {}
    }

    let mut correction = <T as From<f64>>::from(0.0);
    for r in 0..n {
        let i = idx(n, r, col);
        correction += (new_at(r) - old[i]) * cof[i];
    }
    correction
}

/// Evaluate the determinant obtained by replacing column c of \mathbf D by \mathbf N:
/// \det\mathbf D[c\rightarrow\mathbf N]
/// = \sum_rN_r\operatorname{cof}[\mathbf D]_{rc}.
/// This is equivalent to \det\mathbf D+\sum_r(N_r-D_{rc})\operatorname{cof}[\mathbf D]_{rc},
/// but does not require the original column entries.
/// # Arguments:
/// - `n`: Dimension of \mathbf D.
/// - `cof`: Row-major cofactor matrix \operatorname{cof}[\mathbf D].
/// - `col`: Replaced column c.
/// - `new_at`: Returns N_r for row r.
/// # Returns
/// - `T`: \det\mathbf D[c\rightarrow\mathbf N].
#[inline(always)]
pub(super) fn column_replacement_det<T: NOCIScalar>(
    n: usize,
    cof: &[T],
    col: usize,
    mut new_at: impl FnMut(usize) -> T,
) -> T {
    // Apply the Laplace expansion of the replacement determinant along column c.
    match (n, col) {
        (0, _) => return <T as From<f64>>::from(0.0),
        (1, 0) => return new_at(0) * cof[0],
        (2, 0) => return new_at(0) * cof[0] + new_at(1) * cof[2],
        (2, 1) => return new_at(0) * cof[1] + new_at(1) * cof[3],
        (3, 0) => return new_at(0) * cof[0] + new_at(1) * cof[3] + new_at(2) * cof[6],
        (3, 1) => return new_at(0) * cof[1] + new_at(1) * cof[4] + new_at(2) * cof[7],
        (3, 2) => return new_at(0) * cof[2] + new_at(1) * cof[5] + new_at(2) * cof[8],
        (4, 0) => {
            return new_at(0) * cof[0]
                + new_at(1) * cof[4]
                + new_at(2) * cof[8]
                + new_at(3) * cof[12];
        }
        (4, 1) => {
            return new_at(0) * cof[1]
                + new_at(1) * cof[5]
                + new_at(2) * cof[9]
                + new_at(3) * cof[13];
        }
        (4, 2) => {
            return new_at(0) * cof[2]
                + new_at(1) * cof[6]
                + new_at(2) * cof[10]
                + new_at(3) * cof[14];
        }
        (4, 3) => {
            return new_at(0) * cof[3]
                + new_at(1) * cof[7]
                + new_at(2) * cof[11]
                + new_at(3) * cof[15];
        }
        _ => {}
    }

    let mut value = <T as From<f64>>::from(0.0);
    for r in 0..n {
        let i = idx(n, r, col);
        value += new_at(r) * cof[i];
    }
    value
}

/// Form every mixed same-spin contraction determinant required by
/// \sum_{\substack{m_1,\ldots,m_{L+p}\\m_1+\cdots+m_{L+p}=m}}
/// \det\mathbf D_{\mathrm{ov}}(m_{p+1},\ldots,m_{p+L}).
/// The first p assignments are operator-specific; the remaining L assignments select each
/// determinant column from \mathbf D_{\mathrm{ov}}(0,\ldots,0) or
/// \mathbf D_{\mathrm{ov}}(1,\ldots,1).
/// # Arguments:
/// - `w`: Same-spin view containing the number m of zero-overlap orbital pairs.
/// - `l`: Contraction-determinant dimension L.
/// - `pbits`: Number p of leading operator-specific assignments.
/// - `scratch`: Endpoint and mixed contraction-determinant storage.
/// - `f`: Called once for every allowed complete distribution.
/// # Returns
/// - `()`: Writes each mixed determinant to `scratch.det_mix` before calling `f`.
#[inline(always)]
pub(super) fn mix_dets_same<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l: usize,
    pbits: usize,
    scratch: &mut WickScratch<T>,
    mut f: impl FnMut(u64, &mut WickScratch<T>),
) {
    let mut prev_cbits: Option<u64> = None;
    // Enumerate the assignments satisfying \sum_{i=1}^{L+p}m_i = m, then remove the
    // p operator-specific assignments to obtain the L determinant-column assignments.
    for_each_m_combination(l + pbits, w.m, |bits| {
        let cbits = bits >> pbits;
        match prev_cbits {
            None => {
                // Construct the first mixed determinant by selecting every column from the
                // corresponding all-m_i = 0 or all-m_i = 1 endpoint.
                mix_columns(
                    scratch.det_mix.as_mut_slice(),
                    scratch.det0.as_slice(),
                    scratch.det1.as_slice(),
                    l,
                    cbits,
                );
            }
            Some(prev) => {
                // Consecutive complete distributions may change only part of the column assignment.
                // Update exactly the columns whose m_i values differ from the previous determinant.
                let mut changed = prev ^ cbits;
                while changed != 0 {
                    let col = changed.trailing_zeros() as usize;
                    if bit(cbits, col) == 0 {
                        let src = scratch.det0.as_slice();
                        let dst = scratch.det_mix.as_mut_slice();
                        // Restore column `col` from \mathbf D_{\mathrm{ov}}(0,\ldots,0).
                        for r in 0..l {
                            let i = idx(l, r, col);
                            dst[i] = src[i];
                        }
                    } else {
                        let src = scratch.det1.as_slice();
                        let dst = scratch.det_mix.as_mut_slice();
                        // Replace column `col` by the corresponding column of
                        // \mathbf D_{\mathrm{ov}}(1,\ldots,1).
                        for r in 0..l {
                            let i = idx(l, r, col);
                            dst[i] = src[i];
                        }
                    }
                    changed &= changed - 1;
                }
            }
        }
        prev_cbits = Some(cbits);
        f(bits, scratch);
    });
}

/// Form each allowed same-spin mixed contraction determinant, evaluate
/// \det\mathbf D_{\mathrm{ov}} and \operatorname{cof}[\mathbf D_{\mathrm{ov}}], and pass these
/// quantities to the operator-specific evaluator.
/// # Arguments:
/// - `w`: Same-spin view containing m.
/// - `l`: Contraction-determinant dimension L.
/// - `pbits`: Number of leading operator-specific assignments.
/// - `scratch`: Mixed determinant, cofactor and work storage.
/// - `tol`: Numerical threshold applied to \det\mathbf D_{\mathrm{ov}}.
/// - `f`: Receives the complete assignment, scratch storage and determinant value.
/// # Returns
/// - `()`: Calls `f` for each mixed determinant whose absolute determinant exceeds `tol`.
#[inline(always)]
pub(super) fn get_det_adjt_same<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l: usize,
    pbits: usize,
    scratch: &mut WickScratch<T>,
    tol: f64,
    mut f: impl FnMut(u64, &mut WickScratch<T>, T),
) {
    time_call!(crate::timers::nonorthogonalwicks::add_get_det_adjt_same, {
        // Build each \mathbf D_{\mathrm{ov}}(m_{p+1},\ldots,m_{p+L}) in the constrained sum.
        mix_dets_same(w, l, pbits, scratch, |bits, scratch| {
            // Evaluate its determinant and cofactor matrix; operator-specific scalar,
            // one-column and two-column terms are formed by the callback.
            if let Some(det_det) = adjugate_transpose_generic(
                scratch.adjt_det.as_mut_slice(),
                scratch.det_mix.as_slice(),
                l,
                tol,
            ) {
                f(bits, scratch, det_det);
            }
        });
    })
}

/// Independently enumerate the alpha- and beta-spin distributions required by the different-spin
/// two-body matrix element:
/// m_{\alpha0}+\sum_{z=1}^{L_\alpha}m_{\alpha z}=m_\alpha,\qquad
/// m_{\beta0}+\sum_{y=1}^{L_\beta}m_{\beta y}=m_\beta.
/// The leading assignment in each spin space belongs to the operator contraction; the remaining
/// assignments select the columns of \mathbf D_{\alpha,\mathrm{ov}} and
/// \mathbf D_{\beta,\mathrm{ov}}.
/// # Arguments:
/// - `w`: Different-spin reference-pair Wick intermediates.
/// - `rank`: Contraction-determinant dimensions (L_\alpha,L_\beta).
/// - `scratch`: Mixed determinant, cofactor and work storage for both spin spaces.
/// - `deta`: Alpha-spin all-m_i = 0 and all-m_i = 1 endpoint determinants.
/// - `detb`: Beta-spin all-m_i = 0 and all-m_i = 1 endpoint determinants.
/// - `tol`: Numerical threshold applied independently to both determinant values.
/// - `f`: Receives both complete assignments, scratch storage and the two determinant values.
/// # Returns
/// - `()`: Calls `f` when both mixed determinant magnitudes exceed `tol`.
#[inline(always)]
pub(super) fn get_det_adjt_diff<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    rank: (usize, usize),
    scratch: &mut WickScratch<T>,
    deta: DetBranches<'_, T>,
    detb: DetBranches<'_, T>,
    tol: f64,
    mut f: impl FnMut(u64, u64, &mut WickScratch<T>, T, T),
) {
    let (la, lb) = rank;

    time_call!(crate::timers::nonorthogonalwicks::add_get_det_adjt_diff, {
        // Enumerate m_{\alpha0}+\sum_zm_{\alpha z}=m_\alpha. Bit zero is
        // m_{\alpha0}; the remaining bits select alpha-spin determinant columns.
        for_each_m_combination(la + 1, w.aa.m, |bits_a| {
            let inda = bits_a >> 1;
            mix_columns(
                scratch.deta_mix.as_mut_slice(),
                deta.zero,
                deta.one,
                la,
                inda,
            );
            // Evaluate \det\mathbf D_{\alpha,\mathrm{ov}} and its cofactor matrix.
            if let Some(det_a) = adjugate_transpose_generic(
                scratch.adjt_deta.as_mut_slice(),
                scratch.deta_mix.as_slice(),
                la,
                tol,
            ) {
                // For each retained alpha-spin distribution, independently enumerate
                // m_{\beta0}+\sum_ym_{\beta y}=m_\beta.
                for_each_m_combination(lb + 1, w.bb.m, |bits_b| {
                    let indb = bits_b >> 1;
                    mix_columns(
                        scratch.detb_mix.as_mut_slice(),
                        detb.zero,
                        detb.one,
                        lb,
                        indb,
                    );
                    // Evaluate \det\mathbf D_{\beta,\mathrm{ov}} and its cofactor matrix before
                    // passing the factorised spin-sector quantities to the different-spin evaluator.
                    if let Some(det_b) = adjugate_transpose_generic(
                        scratch.adjt_detb.as_mut_slice(),
                        scratch.detb_mix.as_slice(),
                        lb,
                        tol,
                    ) {
                        f(bits_a, bits_b, scratch, det_a, det_b);
                    }
                });
            }
        });
    })
}

/// Form \mathbf D[\eta|z] by deleting row \eta and column z from a full contraction determinant,
/// then evaluate \det\mathbf D[\eta|z] and its cofactor matrix. These quantities generate the
/// two-column same-spin contribution:
/// \sum_{\xi,y}\mathcal J_{\eta z,\xi y}
/// \operatorname{cof}[\mathbf D[\eta|z]]_{\xi y}.
/// # Arguments:
/// - `full`: Row-major entries of the full contraction determinant \mathbf D.
/// - `minor`: Full dimension and removed row \eta and column z.
/// - `minorb`: Scratch storage for \mathbf D[\eta|z].
/// - `adjtb`: Scratch storage for \operatorname{cof}[\mathbf D[\eta|z]].
/// - `tol`: Numerical threshold applied to \det\mathbf D[\eta|z].
/// - `f`: Receives the minor dimension, entries, cofactor matrix and determinant.
/// # Returns
/// - `()`: Calls `f` when the minor determinant magnitude exceeds `tol`.
#[inline(always)]
pub(super) fn minor_adjt<T: NOCIScalar>(
    full: &[T],
    minor: Minor,
    minorb: &mut Vec2<T>,
    adjtb: &mut Vec2<T>,
    tol: f64,
    mut f: impl FnMut(usize, &[T], &[T], T),
) {
    let lm1 = minor.l.saturating_sub(1);
    if minor.l <= 4 {
        let mut invs = [];
        let mut lu = [];
        if let Some(det_minor) = minor_adjugate_transpose(
            adjtb.as_mut_slice(),
            minorb.as_mut_slice(),
            &mut invs,
            &mut lu,
            full,
            minor.l,
            minor.row,
            minor.col,
            tol,
        ) && det_minor.abs() > tol
        {
            f(lm1, minorb.as_slice(), adjtb.as_slice(), det_minor);
        }
    } else {
        build_minor(minorb.as_mut_slice(), full, minor.l, minor.row, minor.col);
        if let Some(det_minor) =
            adjugate_transpose_generic(adjtb.as_mut_slice(), minorb.as_slice(), lm1, tol)
        {
            f(lm1, minorb.as_slice(), adjtb.as_slice(), det_minor);
        }
    }
}

/// Enumerate every binary distribution (m_1,\ldots,m_L) satisfying
/// \sum_{i=1}^{L}m_i=m, with m_i\in\{0,1\}. Each distribution is represented by an L-bit mask
/// containing exactly m set bits.
/// # Arguments:
/// - `l`: Number L of fundamental contractions.
/// - `m`: Number of zero-overlap orbital pairs that must be assigned.
/// - `f`: Called once for each valid distribution.
/// # Returns
/// - `()`: Calls `f` exactly \binom{L}{m} times when 0\leq m\leq L.
#[inline(always)]
pub(super) fn for_each_m_combination(
    l: usize,
    m: usize,
    mut f: impl FnMut(u64),
) {
    // No allowed distribution exists when more zero-overlap orbital pairs are required than
    // there are fundamental contractions.
    if m > l {
        return;
    }
    // The unique m = 0 distribution has every m_i = 0.
    if m == 0 {
        f(0);
        return;
    }
    // The unique m = L distribution has every m_i = 1.
    if m == l {
        f((1u64 << l) - 1);
        return;
    }

    // Enumerate the \binom{L}{m} fixed-population bit masks in increasing integer order.
    let limit = 1u64 << l;
    let mut x = (1u64 << m) - 1;
    while x < limit {
        f(x);
        let c = x & x.wrapping_neg();
        let r = x + c;
        x = (((r ^ x) >> 2) / c) | r;
    }
}

/// Evaluate \det\mathbf D and its cofactor matrix:
/// \operatorname{cof}[\mathbf D]_{rc}=(-1)^{r+c}\det\mathbf D[r|c].
/// Fixed-rank determinant and cofactor kernels are used for n\leq4; larger matrices evaluate the
/// determinant first and then construct every cofactor explicitly from an (n-1)\times(n-1) minor.
/// # Arguments:
/// - `adjt`: Output row-major cofactor matrix \operatorname{cof}[\mathbf D].
/// - `full`: Row-major entries of \mathbf D.
/// - `n`: Dimension of \mathbf D.
/// - `tol`: Numerical threshold applied to |\det\mathbf D|.
/// # Returns
/// - `Option<T>`: \det\mathbf D when its magnitude exceeds `tol`.
#[inline(always)]
fn adjugate_transpose_generic<T: NOCIScalar>(
    adjt: &mut [T],
    full: &[T],
    n: usize,
    tol: f64,
) -> Option<T> {
    // Use the specialised low-rank routine for n \leq 4.
    if n <= 4 {
        let mut invs = [];
        let mut lu = [];
        let detv = adjugate_transpose(adjt, &mut invs, &mut lu, full, n, tol)?;
        if detv.abs() <= tol {
            return None;
        }
        return Some(detv);
    }

    // For larger matrices, reject determinant values below the numerical threshold before
    // constructing the cofactor matrix.
    let detv = det_slice(full, n)?;
    if detv.abs() <= tol {
        return None;
    }

    // Construct each cofactor as (-1)^{r+c}\det\mathbf D[r|c].
    let mut minor = vec![<T as From<f64>>::from(0.0); (n - 1) * (n - 1)];
    for r in 0..n {
        for c in 0..n {
            build_minor(&mut minor, full, n, r, c);
            let md = det_or_zero(&minor, n - 1);
            let sign = if ((r + c) & 1) == 0 { 1.0 } else { -1.0 };
            adjt[idx(n, r, c)] = <T as From<f64>>::from(sign) * md;
        }
    }

    Some(detv)
}
