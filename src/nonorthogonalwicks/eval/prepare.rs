// nonorthogonalwicks/eval/prepare.rs
use super::super::scratch::{IndexVec, WickScratch};
use super::super::view::SameSpinView;

use crate::ExcitationSpin;
use crate::noci::NOCIScalar;

use crate::maths::build_d;
use crate::time_call;

/// Prepare the contraction-determinant quantities shared by the same-spin overlap and Hamiltonian evaluators.
/// For total excitation rank L = L_x + L_w, the contraction determinant has elements:
/// (\mathbf D_{\mathrm{ov}})_{ij} = X_{r_i c_j}^{(m_j)} for i \geq j,
/// (\mathbf D_{\mathrm{ov}})_{ij} = Y_{r_i c_j}^{(m_j)} for i < j.
/// `scratch.det0` stores \mathbf D_{\mathrm{ov}}(0,\ldots,0); when m > 0, `scratch.det1` also stores
/// \mathbf D_{\mathrm{ov}}(1,\ldots,1). Mixed distributions are formed later by selecting each
/// column j according to m_j \in \{0,1\}.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `l_ex`: Excitation defining the bra determinant \langle{}^x\Psi_{i\cdots}^{a\cdots}|.
/// - `g_ex`: Excitation defining the ket determinant |{}^w\Psi_{j\cdots}^{b\cdots}\rangle.
/// - `scratch`: Scratch storage receiving the determinant labels and endpoint contraction determinants.
/// # Returns
/// - `()`: Writes the required contraction-determinant quantities into `scratch`.
#[inline(always)]
pub fn prepare_same<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
) {
    time_call!(crate::timers::nonorthogonalwicks::add_prepare_same, {
        // With no zero-overlap orbital pairs, the constrained sum requires only
        // \mathbf D_{\mathrm{ov}}(0,\ldots,0). Otherwise prepare both endpoint determinants.
        if w.m == 0 {
            prepare_same_m0(w, l_ex, g_ex, scratch)
        } else {
            prepare_same_gen(w, l_ex, g_ex, scratch)
        }
    })
}

/// Prepare \mathbf D_{\mathrm{ov}}(0,\ldots,0) when m = 0, so the reference pair contains
/// no zero-overlap orbital pairs and every column assignment is m_j = 0. Fixed-rank kernels are
/// used for L = 1,\ldots,6; arbitrary ranks use the general determinant builder.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage receiving the determinant labels and \mathbf D_{\mathrm{ov}}(0,\ldots,0).
/// # Returns
/// - `()`: Writes the m_j = 0 contraction determinant into `scratch.det0`.
#[inline(always)]
fn prepare_same_m0<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
) {
    time_call!(crate::timers::nonorthogonalwicks::add_prepare_same_m0, {
        // The contraction-determinant dimension is L = L_x + L_w.
        let l = l_ex.holes.len() + g_ex.holes.len();

        // For L = 0 the empty determinant has value one, so no determinant storage is required.
        if l > 0 {
            // Allocate the determinant and auxiliary buffers required by the corresponding evaluators.
            match l {
                1 => scratch.ensure_same_m0(1),
                2 => scratch.ensure_same_m0(2),
                3 => scratch.ensure_same(3),
                4 => scratch.ensure_same(4),
                5 => scratch.ensure_same(5),
                6 => scratch.ensure_same(6),
                _ => scratch.ensure_same(l),
            }

            // Order the labels as V_x followed by O_w for rows and O_x followed by V_w for columns.
            construct_determinant_indices(l_ex, g_ex, w, &mut scratch.rows, &mut scratch.cols);

            // Use direct fixed-rank construction where available.
            match l {
                1 => prepare_same_m0_l1(w, scratch),
                2 => prepare_same_m0_l2(w, scratch),
                3 => prepare_same_m0_l3(w, scratch),
                4 => prepare_same_m0_l4(w, scratch),
                5 => prepare_same_m0_l5(w, scratch),
                6 => prepare_same_m0_l6(w, scratch),
                _ => {
                    // Select the m_j = 0 X and Y fundamental contractions.
                    let x0 = w.x(0);
                    let y0 = w.y(0);

                    // Fill (\mathbf D_{\mathrm{ov}})_{ij} with X_{r_i c_j}^{(0)} for i \geq j
                    // and Y_{r_i c_j}^{(0)} for i < j.
                    build_d(
                        scratch.det0.as_mut_slice(),
                        l,
                        &x0,
                        &y0,
                        scratch.rows.as_slice(),
                        scratch.cols.as_slice(),
                    );
                }
            }
        }
    })
}

/// Prepare the fixed-rank L = 1 contraction determinant \mathbf D_{\mathrm{ov}}(0,\ldots,0).
/// Its elements are (\mathbf D_{\mathrm{ov}})_{ij} = X_{r_i c_j}^{(0)} for i \geq j and
/// (\mathbf D_{\mathrm{ov}})_{ij} = Y_{r_i c_j}^{(0)} for i < j.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared row and column labels and storage for the rank-1 contraction determinant.
/// # Returns
/// - `()`: Writes \mathbf D_{\mathrm{ov}}(0,\ldots,0) into `scratch.det0`.
#[inline(always)]
fn prepare_same_m0_l1<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) {
    time_call!(crate::timers::nonorthogonalwicks::add_prepare_same_m0_l1, {
        // For L = 1, \mathbf D_{\mathrm{ov}}(0) = [X_{r_0c_0}^{(0)}].
        let x0 = w.x(0);
        let xstr = x0.strides();
        let xptr = x0.as_ptr();
        let r0 = scratch.rows[0] as isize;
        let c0 = scratch.cols[0] as isize;
        let det0 = scratch.det0.as_mut_slice();

        // The prepared labels and rank-one buffer make the unchecked access valid.
        unsafe {
            det0[0] = *xptr.offset(r0 * xstr[0] + c0 * xstr[1]);
        }
    })
}

/// Prepare the fixed-rank L = 2 contraction determinant \mathbf D_{\mathrm{ov}}(0,\ldots,0).
/// Its elements are (\mathbf D_{\mathrm{ov}})_{ij} = X_{r_i c_j}^{(0)} for i \geq j and
/// (\mathbf D_{\mathrm{ov}})_{ij} = Y_{r_i c_j}^{(0)} for i < j.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared row and column labels and storage for the rank-2 contraction determinant.
/// # Returns
/// - `()`: Writes \mathbf D_{\mathrm{ov}}(0,\ldots,0) into `scratch.det0`.
#[inline(always)]
fn prepare_same_m0_l2<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) {
    time_call!(crate::timers::nonorthogonalwicks::add_prepare_same_m0_l2, {
        // Build \mathbf D_{\mathrm{ov}}(0,0) with X^{(0)} on and below the diagonal
        // and Y^{(0)} above the diagonal.
        let x0 = w.x(0);
        let y0 = w.y(0);
        let xstr = x0.strides();
        let ystr = y0.strides();
        let xptr = x0.as_ptr();
        let yptr = y0.as_ptr();

        let r0 = scratch.rows[0] as isize;
        let r1 = scratch.rows[1] as isize;
        let c0 = scratch.cols[0] as isize;
        let c1 = scratch.cols[1] as isize;

        let xr0 = r0 * xstr[0];
        let xr1 = r1 * xstr[0];
        let yr0 = r0 * ystr[0];

        let det0 = scratch.det0.as_mut_slice();

        // \mathbf D_{\mathrm{ov}}(0,0) =
        // \begin{pmatrix}X_{r_0c_0}^{(0)} & Y_{r_0c_1}^{(0)}\\X_{r_1c_0}^{(0)} & X_{r_1c_1}^{(0)}\end{pmatrix}.
        unsafe {
            det0[0] = *xptr.offset(xr0 + c0 * xstr[1]);
            det0[1] = *yptr.offset(yr0 + c1 * ystr[1]);
            det0[2] = *xptr.offset(xr1 + c0 * xstr[1]);
            det0[3] = *xptr.offset(xr1 + c1 * xstr[1]);
        }
    })
}

/// Prepare the fixed-rank L = 3 contraction determinant \mathbf D_{\mathrm{ov}}(0,\ldots,0).
/// Its elements are (\mathbf D_{\mathrm{ov}})_{ij} = X_{r_i c_j}^{(0)} for i \geq j and
/// (\mathbf D_{\mathrm{ov}})_{ij} = Y_{r_i c_j}^{(0)} for i < j.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared row and column labels and storage for the rank-3 contraction determinant.
/// # Returns
/// - `()`: Writes \mathbf D_{\mathrm{ov}}(0,\ldots,0) into `scratch.det0`.
#[inline(always)]
fn prepare_same_m0_l3<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) {
    time_call!(crate::timers::nonorthogonalwicks::add_prepare_same_m0_l3, {
        // Build the rank-three \mathbf D_{\mathrm{ov}}(0,0,0) from the m_j = 0
        // X and Y fundamental contractions.
        let x0 = w.x(0);
        let y0 = w.y(0);
        let xstr = x0.strides();
        let ystr = y0.strides();
        let xptr = x0.as_ptr();
        let yptr = y0.as_ptr();

        let r0 = scratch.rows[0] as isize;
        let r1 = scratch.rows[1] as isize;
        let r2 = scratch.rows[2] as isize;
        let c0 = scratch.cols[0] as isize;
        let c1 = scratch.cols[1] as isize;
        let c2 = scratch.cols[2] as isize;

        let xr0 = r0 * xstr[0];
        let xr1 = r1 * xstr[0];
        let xr2 = r2 * xstr[0];
        let yr0 = r0 * ystr[0];
        let yr1 = r1 * ystr[0];

        let det0 = scratch.det0.as_mut_slice();

        // Row i = 0 contains X_{r_0c_0}^{(0)}, Y_{r_0c_1}^{(0)}, Y_{r_0c_2}^{(0)};
        // row i = 1 contains X_{r_1c_0}^{(0)}, X_{r_1c_1}^{(0)}, Y_{r_1c_2}^{(0)};
        // row i = 2 contains X_{r_2c_0}^{(0)}, X_{r_2c_1}^{(0)}, X_{r_2c_2}^{(0)}.
        unsafe {
            det0[0] = *xptr.offset(xr0 + c0 * xstr[1]);
            det0[1] = *yptr.offset(yr0 + c1 * ystr[1]);
            det0[2] = *yptr.offset(yr0 + c2 * ystr[1]);

            det0[3] = *xptr.offset(xr1 + c0 * xstr[1]);
            det0[4] = *xptr.offset(xr1 + c1 * xstr[1]);
            det0[5] = *yptr.offset(yr1 + c2 * ystr[1]);

            det0[6] = *xptr.offset(xr2 + c0 * xstr[1]);
            det0[7] = *xptr.offset(xr2 + c1 * xstr[1]);
            det0[8] = *xptr.offset(xr2 + c2 * xstr[1]);
        }
    })
}

/// Prepare the fixed-rank L = 4 contraction determinant \mathbf D_{\mathrm{ov}}(0,\ldots,0).
/// Its elements are (\mathbf D_{\mathrm{ov}})_{ij} = X_{r_i c_j}^{(0)} for i \geq j and
/// (\mathbf D_{\mathrm{ov}})_{ij} = Y_{r_i c_j}^{(0)} for i < j.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared row and column labels and storage for the rank-4 contraction determinant.
/// # Returns
/// - `()`: Writes \mathbf D_{\mathrm{ov}}(0,\ldots,0) into `scratch.det0`.
#[inline(always)]
fn prepare_same_m0_l4<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) {
    time_call!(crate::timers::nonorthogonalwicks::add_prepare_same_m0_l4, {
        // Build the rank-four \mathbf D_{\mathrm{ov}}(0,0,0,0) from the m_j = 0
        // X and Y fundamental contractions.
        let x0 = w.x(0);
        let y0 = w.y(0);
        let xstr = x0.strides();
        let ystr = y0.strides();
        let xptr = x0.as_ptr();
        let yptr = y0.as_ptr();

        let r0 = scratch.rows[0] as isize;
        let r1 = scratch.rows[1] as isize;
        let r2 = scratch.rows[2] as isize;
        let r3 = scratch.rows[3] as isize;
        let c0 = scratch.cols[0] as isize;
        let c1 = scratch.cols[1] as isize;
        let c2 = scratch.cols[2] as isize;
        let c3 = scratch.cols[3] as isize;

        let xr0 = r0 * xstr[0];
        let xr1 = r1 * xstr[0];
        let xr2 = r2 * xstr[0];
        let xr3 = r3 * xstr[0];
        let yr0 = r0 * ystr[0];
        let yr1 = r1 * ystr[0];
        let yr2 = r2 * ystr[0];

        let det0 = scratch.det0.as_mut_slice();

        // Each row switches from X^{(0)} through the diagonal to Y^{(0)} above it:
        // (\mathbf D_{\mathrm{ov}})_{ij} = X_{r_i c_j}^{(0)} for i \geq j and Y_{r_i c_j}^{(0)} for i < j.
        unsafe {
            det0[0] = *xptr.offset(xr0 + c0 * xstr[1]);
            det0[1] = *yptr.offset(yr0 + c1 * ystr[1]);
            det0[2] = *yptr.offset(yr0 + c2 * ystr[1]);
            det0[3] = *yptr.offset(yr0 + c3 * ystr[1]);

            det0[4] = *xptr.offset(xr1 + c0 * xstr[1]);
            det0[5] = *xptr.offset(xr1 + c1 * xstr[1]);
            det0[6] = *yptr.offset(yr1 + c2 * ystr[1]);
            det0[7] = *yptr.offset(yr1 + c3 * ystr[1]);

            det0[8] = *xptr.offset(xr2 + c0 * xstr[1]);
            det0[9] = *xptr.offset(xr2 + c1 * xstr[1]);
            det0[10] = *xptr.offset(xr2 + c2 * xstr[1]);
            det0[11] = *yptr.offset(yr2 + c3 * ystr[1]);

            det0[12] = *xptr.offset(xr3 + c0 * xstr[1]);
            det0[13] = *xptr.offset(xr3 + c1 * xstr[1]);
            det0[14] = *xptr.offset(xr3 + c2 * xstr[1]);
            det0[15] = *xptr.offset(xr3 + c3 * xstr[1]);
        }
    })
}

/// Prepare the fixed-rank L = 5 contraction determinant \mathbf D_{\mathrm{ov}}(0,\ldots,0).
/// Its elements are (\mathbf D_{\mathrm{ov}})_{ij} = X_{r_i c_j}^{(0)} for i \geq j and
/// (\mathbf D_{\mathrm{ov}})_{ij} = Y_{r_i c_j}^{(0)} for i < j.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared row and column labels and storage for the rank-5 contraction determinant.
/// # Returns
/// - `()`: Writes \mathbf D_{\mathrm{ov}}(0,\ldots,0) into `scratch.det0`.
#[inline(always)]
fn prepare_same_m0_l5<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) {
    // Construct every element from the rule
    // (\mathbf D_{\mathrm{ov}})_{ij} = X_{r_i c_j}^{(0)} for i \geq j and Y_{r_i c_j}^{(0)} for i < j.
    const N: usize = 5;
    let x0 = w.x(0);
    let y0 = w.y(0);
    let xstr = x0.strides();
    let ystr = y0.strides();
    let xptr = x0.as_ptr();
    let yptr = y0.as_ptr();
    let rows = scratch.rows.as_slice();
    let cols = scratch.cols.as_slice();
    let det0 = scratch.det0.as_mut_slice();

    // The fixed rank and prepared buffers guarantee that all unchecked row, column and determinant accesses are valid.
    unsafe {
        let mut i = 0usize;
        while i < N {
            let r = *rows.get_unchecked(i) as isize;
            let xr = r * xstr[0];
            let yr = r * ystr[0];

            let mut j = 0usize;
            while j < N {
                let c = *cols.get_unchecked(j) as isize;
                *det0.get_unchecked_mut(i * N + j) = if i >= j {
                    *xptr.offset(xr + c * xstr[1])
                } else {
                    *yptr.offset(yr + c * ystr[1])
                };
                j += 1;
            }

            i += 1;
        }
    }
}

/// Prepare the fixed-rank L = 6 contraction determinant \mathbf D_{\mathrm{ov}}(0,\ldots,0).
/// Its elements are (\mathbf D_{\mathrm{ov}})_{ij} = X_{r_i c_j}^{(0)} for i \geq j and
/// (\mathbf D_{\mathrm{ov}})_{ij} = Y_{r_i c_j}^{(0)} for i < j.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `scratch`: Prepared row and column labels and storage for the rank-6 contraction determinant.
/// # Returns
/// - `()`: Writes \mathbf D_{\mathrm{ov}}(0,\ldots,0) into `scratch.det0`.
#[inline(always)]
fn prepare_same_m0_l6<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    scratch: &mut WickScratch<T>,
) {
    // Construct every element from the rule
    // (\mathbf D_{\mathrm{ov}})_{ij} = X_{r_i c_j}^{(0)} for i \geq j and Y_{r_i c_j}^{(0)} for i < j.
    const N: usize = 6;
    let x0 = w.x(0);
    let y0 = w.y(0);
    let xstr = x0.strides();
    let ystr = y0.strides();
    let xptr = x0.as_ptr();
    let yptr = y0.as_ptr();
    let rows = scratch.rows.as_slice();
    let cols = scratch.cols.as_slice();
    let det0 = scratch.det0.as_mut_slice();

    // The fixed rank and prepared buffers guarantee that all unchecked row, column and determinant accesses are valid.
    unsafe {
        let mut i = 0usize;
        while i < N {
            let r = *rows.get_unchecked(i) as isize;
            let xr = r * xstr[0];
            let yr = r * ystr[0];

            let mut j = 0usize;
            while j < N {
                let c = *cols.get_unchecked(j) as isize;
                *det0.get_unchecked_mut(i * N + j) = if i >= j {
                    *xptr.offset(xr + c * xstr[1])
                } else {
                    *yptr.offset(yr + c * ystr[1])
                };
                j += 1;
            }

            i += 1;
        }
    }
}

/// Prepare the two endpoint contraction determinants required when m > 0:
/// \mathbf D_{\mathrm{ov}}(0,\ldots,0) and \mathbf D_{\mathrm{ov}}(1,\ldots,1).
/// The evaluators subsequently construct each mixed \mathbf D_{\mathrm{ov}}(m_1,\ldots,m_L)
/// by selecting column j from the endpoint determinant corresponding to m_j \in \{0,1\}.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage receiving the determinant labels and both endpoint determinants.
/// # Returns
/// - `()`: Writes the endpoint contraction determinants into `scratch.det0` and `scratch.det1`.
pub fn prepare_same_gen<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
) {
    time_call!(crate::timers::nonorthogonalwicks::add_prepare_same_gen, {
        // The contraction-determinant dimension is L = L_x + L_w.
        let l = l_ex.holes.len() + g_ex.holes.len();
        scratch.ensure_same(l);

        // Use the same V_x \cup O_w row ordering and O_x \cup V_w column ordering for both endpoints.
        construct_determinant_indices(l_ex, g_ex, w, &mut scratch.rows, &mut scratch.cols);

        // Build \mathbf D_{\mathrm{ov}}(0,\ldots,0) from X^{(0)} and Y^{(0)}.
        let x0 = w.x(0);
        let y0 = w.y(0);
        build_d(
            scratch.det0.as_mut_slice(),
            l,
            &x0,
            &y0,
            scratch.rows.as_slice(),
            scratch.cols.as_slice(),
        );

        // Build \mathbf D_{\mathrm{ov}}(1,\ldots,1) from X^{(1)} and Y^{(1)}.
        // Mixed distributions are assembled later by selecting each column from one of these endpoints.
        let x1 = w.x(1);
        let y1 = w.y(1);
        build_d(
            scratch.det1.as_mut_slice(),
            l,
            &x1,
            &y1,
            scratch.rows.as_slice(),
            scratch.cols.as_slice(),
        );
    })
}

/// Construct the row and column labels of the overlap contraction determinant.
/// Each x-reference excitation pair contributes a particle row a \in V_x and a hole column i \in O_x.
/// Each w-reference excitation pair then contributes a hole row j \in O_w and a particle column b \in V_w.
/// The resulting row space is V_x \cup O_w and the column space is O_x \cup V_w.
/// # Arguments:
/// - `l_ex`: Excitation defining the bra determinant generated from \langle{}^x\Psi|.
/// - `g_ex`: Excitation defining the ket determinant generated from |{}^w\Psi\rangle.
/// - `w`: Same-spin intermediates containing the compact occupied and virtual block dimensions.
/// - `rows`: Output labels in V_x \cup O_w.
/// - `cols`: Output labels in O_x \cup V_w.
/// # Returns
/// - `()`: Writes the ordered contraction-determinant labels into `rows` and `cols`.
#[inline(always)]
fn construct_determinant_indices<T: NOCIScalar>(
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    w: &SameSpinView<'_, T>,
    rows: &mut IndexVec,
    cols: &mut IndexVec,
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_construct_determinant_indices,
        {
            // The number of row/column pairs is L = L_x + L_w.
            let l = l_ex.holes.len() + g_ex.holes.len();

            // Fixed-rank constructors retain the same x-reference-then-w-reference ordering.
            match l {
                1 => construct_determinant_indices_l1(l_ex, g_ex, w, rows, cols),
                2 => construct_determinant_indices_l2(l_ex, g_ex, w, rows, cols),
                3 => construct_determinant_indices_l3(l_ex, g_ex, w, rows, cols),
                4 => construct_determinant_indices_l4(l_ex, g_ex, w, rows, cols),
                _ => construct_determinant_indices_gen(l_ex, g_ex, w, rows, cols),
            }
        }
    )
}

/// Construct the row and column labels for a rank-1 contraction determinant.
/// The x-reference pairs are written first as (r_k,c_k) = (a_k,i_k), followed by the
/// w-reference pairs as (r_k,c_k) = (j_k,b_k). Thus the row and column spaces are
/// V_x \cup O_w and O_x \cup V_w respectively.
/// # Arguments:
/// - `l_ex`: Excitation defining the bra determinant generated from \langle{}^x\Psi|.
/// - `g_ex`: Excitation defining the ket determinant generated from |{}^w\Psi\rangle.
/// - `w`: Same-spin intermediates containing the compact occupied and virtual block dimensions.
/// - `rows`: Output labels in V_x \cup O_w.
/// - `cols`: Output labels in O_x \cup V_w.
/// # Returns
/// - `()`: Writes the ordered rank-1 labels into `rows` and `cols`.
#[inline(always)]
pub(super) fn construct_determinant_indices_l1<T: NOCIScalar>(
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    w: &SameSpinView<'_, T>,
    rows: &mut IndexVec,
    cols: &mut IndexVec,
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_construct_determinant_indices_l1,
        {
            // Exactly one excitation pair is present: either L_x = 1 or L_w = 1.
            let nl = l_ex.holes.len();
            let nocc = w.nocc;
            let nvirt = w.nmo - nocc;

            rows.ensure(1);
            cols.ensure(1);

            let rows = rows.as_mut_slice();
            let cols = cols.as_mut_slice();

            // Compact row labels place V_x before O_w; compact column labels place O_x before V_w.
            unsafe {
                if nl == 1 {
                    *rows.get_unchecked_mut(0) = *l_ex.parts.get_unchecked(0) - nocc;
                    *cols.get_unchecked_mut(0) = *l_ex.holes.get_unchecked(0);
                } else {
                    *rows.get_unchecked_mut(0) = nvirt + *g_ex.holes.get_unchecked(0);
                    *cols.get_unchecked_mut(0) = *g_ex.parts.get_unchecked(0);
                }
            }
        }
    )
}

/// Construct the row and column labels for a rank-2 contraction determinant.
/// The x-reference pairs are written first as (r_k,c_k) = (a_k,i_k), followed by the
/// w-reference pairs as (r_k,c_k) = (j_k,b_k). Thus the row and column spaces are
/// V_x \cup O_w and O_x \cup V_w respectively.
/// # Arguments:
/// - `l_ex`: Excitation defining the bra determinant generated from \langle{}^x\Psi|.
/// - `g_ex`: Excitation defining the ket determinant generated from |{}^w\Psi\rangle.
/// - `w`: Same-spin intermediates containing the compact occupied and virtual block dimensions.
/// - `rows`: Output labels in V_x \cup O_w.
/// - `cols`: Output labels in O_x \cup V_w.
/// # Returns
/// - `()`: Writes the ordered rank-2 labels into `rows` and `cols`.
#[inline(always)]
pub(super) fn construct_determinant_indices_l2<T: NOCIScalar>(
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    w: &SameSpinView<'_, T>,
    rows: &mut IndexVec,
    cols: &mut IndexVec,
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_construct_determinant_indices_l2,
        {
            let nl = l_ex.holes.len();
            let ng = g_ex.holes.len();
            let nocc = w.nocc;
            let nvirt = w.nmo - nocc;

            rows.ensure(2);
            cols.ensure(2);

            let rows = rows.as_mut_slice();
            let cols = cols.as_mut_slice();

            // Write (a_k,i_k) for all x-reference pairs, followed by (j_k,b_k) for all w-reference pairs.
            // The fixed total rank and ensured buffers make the unchecked accesses valid.
            unsafe {
                for k in 0..nl {
                    *rows.get_unchecked_mut(k) = *l_ex.parts.get_unchecked(k) - nocc;
                    *cols.get_unchecked_mut(k) = *l_ex.holes.get_unchecked(k);
                }
                for k in 0..ng {
                    let i = nl + k;
                    *rows.get_unchecked_mut(i) = nvirt + *g_ex.holes.get_unchecked(k);
                    *cols.get_unchecked_mut(i) = *g_ex.parts.get_unchecked(k);
                }
            }
        }
    )
}

/// Construct the row and column labels for a rank-3 contraction determinant.
/// The x-reference pairs are written first as (r_k,c_k) = (a_k,i_k), followed by the
/// w-reference pairs as (r_k,c_k) = (j_k,b_k). Thus the row and column spaces are
/// V_x \cup O_w and O_x \cup V_w respectively.
/// # Arguments:
/// - `l_ex`: Excitation defining the bra determinant generated from \langle{}^x\Psi|.
/// - `g_ex`: Excitation defining the ket determinant generated from |{}^w\Psi\rangle.
/// - `w`: Same-spin intermediates containing the compact occupied and virtual block dimensions.
/// - `rows`: Output labels in V_x \cup O_w.
/// - `cols`: Output labels in O_x \cup V_w.
/// # Returns
/// - `()`: Writes the ordered rank-3 labels into `rows` and `cols`.
#[inline(always)]
pub(super) fn construct_determinant_indices_l3<T: NOCIScalar>(
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    w: &SameSpinView<'_, T>,
    rows: &mut IndexVec,
    cols: &mut IndexVec,
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_construct_determinant_indices_l3,
        {
            let nl = l_ex.holes.len();
            let ng = g_ex.holes.len();
            let nocc = w.nocc;
            let nvirt = w.nmo - nocc;

            rows.ensure(3);
            cols.ensure(3);

            let rows = rows.as_mut_slice();
            let cols = cols.as_mut_slice();

            // Write (a_k,i_k) for all x-reference pairs, followed by (j_k,b_k) for all w-reference pairs.
            // The fixed total rank and ensured buffers make the unchecked accesses valid.
            unsafe {
                for k in 0..nl {
                    *rows.get_unchecked_mut(k) = *l_ex.parts.get_unchecked(k) - nocc;
                    *cols.get_unchecked_mut(k) = *l_ex.holes.get_unchecked(k);
                }
                for k in 0..ng {
                    let i = nl + k;
                    *rows.get_unchecked_mut(i) = nvirt + *g_ex.holes.get_unchecked(k);
                    *cols.get_unchecked_mut(i) = *g_ex.parts.get_unchecked(k);
                }
            }
        }
    )
}

/// Construct the row and column labels for a rank-4 contraction determinant.
/// The x-reference pairs are written first as (r_k,c_k) = (a_k,i_k), followed by the
/// w-reference pairs as (r_k,c_k) = (j_k,b_k). Thus the row and column spaces are
/// V_x \cup O_w and O_x \cup V_w respectively.
/// # Arguments:
/// - `l_ex`: Excitation defining the bra determinant generated from \langle{}^x\Psi|.
/// - `g_ex`: Excitation defining the ket determinant generated from |{}^w\Psi\rangle.
/// - `w`: Same-spin intermediates containing the compact occupied and virtual block dimensions.
/// - `rows`: Output labels in V_x \cup O_w.
/// - `cols`: Output labels in O_x \cup V_w.
/// # Returns
/// - `()`: Writes the ordered rank-4 labels into `rows` and `cols`.
#[inline(always)]
pub(super) fn construct_determinant_indices_l4<T: NOCIScalar>(
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    w: &SameSpinView<'_, T>,
    rows: &mut IndexVec,
    cols: &mut IndexVec,
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_construct_determinant_indices_l4,
        {
            let nl = l_ex.holes.len();
            let ng = g_ex.holes.len();
            let nocc = w.nocc;
            let nvirt = w.nmo - nocc;

            rows.ensure(4);
            cols.ensure(4);

            let rows = rows.as_mut_slice();
            let cols = cols.as_mut_slice();

            // Write (a_k,i_k) for all x-reference pairs, followed by (j_k,b_k) for all w-reference pairs.
            // The fixed total rank and ensured buffers make the unchecked accesses valid.
            unsafe {
                for k in 0..nl {
                    *rows.get_unchecked_mut(k) = *l_ex.parts.get_unchecked(k) - nocc;
                    *cols.get_unchecked_mut(k) = *l_ex.holes.get_unchecked(k);
                }
                for k in 0..ng {
                    let i = nl + k;
                    *rows.get_unchecked_mut(i) = nvirt + *g_ex.holes.get_unchecked(k);
                    *cols.get_unchecked_mut(i) = *g_ex.parts.get_unchecked(k);
                }
            }
        }
    )
}

/// Construct the row and column labels for an arbitrary-rank contraction determinant.
/// For L_x bra-reference and L_w ket-reference excitation pairs:
/// (r_k,c_k) = (a_k,i_k) for 0 \leq k < L_x,
/// (r_{L_x+k},c_{L_x+k}) = (j_k,b_k) for 0 \leq k < L_w.
/// Hence the row space is V_x \cup O_w and the column space is O_x \cup V_w.
/// # Arguments:
/// - `l_ex`: Excitation defining the bra determinant generated from \langle{}^x\Psi|.
/// - `g_ex`: Excitation defining the ket determinant generated from |{}^w\Psi\rangle.
/// - `w`: Same-spin intermediates containing the compact occupied and virtual block dimensions.
/// - `rows`: Output labels in V_x \cup O_w.
/// - `cols`: Output labels in O_x \cup V_w.
/// # Returns
/// - `()`: Writes the ordered contraction-determinant labels into `rows` and `cols`.
#[inline(always)]
pub(super) fn construct_determinant_indices_gen<T: NOCIScalar>(
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    w: &SameSpinView<'_, T>,
    rows: &mut IndexVec,
    cols: &mut IndexVec,
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_construct_determinant_indices_gen,
        {
            // L_x and L_w determine the two ordered blocks of the determinant labels.
            let nl = l_ex.holes.len();
            let ng = g_ex.holes.len();
            let need = nl + ng;
            let nocc = w.nocc;
            let nvirt = w.nmo - nocc;

            rows.ensure(need);
            cols.ensure(need);

            let rows = rows.as_mut_slice();
            let cols = cols.as_mut_slice();

            // Map the x-reference pairs to the V_x row block and O_x column block.
            for k in 0..nl {
                rows[k] = l_ex.parts[k] - nocc;
                cols[k] = l_ex.holes[k];
            }

            // Append the w-reference pairs in the O_w row block and V_w column block.
            for k in 0..ng {
                let i = nl + k;
                rows[i] = nvirt + g_ex.holes[k];
                cols[i] = g_ex.parts[k];
            }
        }
    )
}
