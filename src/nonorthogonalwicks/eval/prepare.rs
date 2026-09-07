// nonorthogonalwicks/eval/prepare.rs
// Crate-root imports.
use crate::ExcitationSpin;
use crate::maths::{build_d_const, build_d_dynamic};
use crate::noci::NOCIScalar;
use crate::time_call;

// Parent/sibling imports.
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;
use super::dispatch::{dispatch_overlap_scalar_ranks, dispatch_pair_ranks};

/// Prepare the contraction-determinant quantities shared by the same-spin overlap and Hamiltonian evaluators.
/// `For total excitation rank L = L_x + L_w, the contraction determinant has elements:`
/// `(\mathbf D_{\mathrm{ov}})_{ij} = X_{r_i c_j}^{(m_j)} for i \geq j,`
/// `(\mathbf D_{\mathrm{ov}})_{ij} = Y_{r_i c_j}^{(m_j)} for i < j.`
/// `scratch.det0 stores \mathbf D_{\mathrm{ov}}(0,\ldots,0); when m > 0, scratch.det1 also stores`
/// `\mathbf D_{\mathrm{ov}}(1,\ldots,1). Mixed distributions are formed later by selecting each`
/// `column j according to m_j \in \{0,1\}.`
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `l_ex`: `Excitation defining the bra determinant \langle{}^x\Psi_{i\cdots}^{a\cdots}|.`
/// - `g_ex`: `Excitation defining the ket determinant |{}^w\Psi_{j\cdots}^{b\cdots}\rangle.`
/// - `scratch`: Scratch storage receiving the determinant labels and required contraction determinants.
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
        // `\mathbf D_{\mathrm{ov}}(0,\ldots,0)`. Otherwise prepare both required contraction
        // determinants.
        if w.m == 0 {
            prepare_same_m0(w, l_ex, g_ex, scratch)
        } else {
            prepare_same_gen(w, l_ex, g_ex, scratch)
        }
    })
}

/// Prepare `D_ov(0,...,0)` when the reference pair contains no zero-overlap orbital pairs.
/// Fixed-rank evaluation is selected by the generated `MAXEXCIT` dispatcher; larger ranks use
/// the runtime determinant builder.
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `l_ex`: Bra excitation.
/// - `g_ex`: Ket excitation.
/// - `scratch`: Reusable Wick workspace.
/// # Returns
/// - `()`: Writes determinant labels and `D_ov(0,...,0)`.
#[inline(always)]
fn prepare_same_m0<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
) {
    time_call!(crate::timers::nonorthogonalwicks::add_prepare_same_m0, {
        let rx = l_ex.holes.count_ones() as usize;
        let rw = g_ex.holes.count_ones() as usize;

        if rx == 0 && rw == 0 {
            scratch.ensure_same(0);
            return;
        }

        dispatch_overlap_scalar_ranks!(
            (rx, rw),
            |_RX, _RW, L, _D| {
                prepare_same_m0_const::<T, L>(w, l_ex, g_ex, scratch);
            },
            {
                let l = rx + rw;
                scratch.ensure_same(l);

                construct_determinant_indices(
                    l_ex,
                    g_ex,
                    w,
                    scratch.rows.as_mut_slice(),
                    scratch.cols.as_mut_slice(),
                );

                let x0 = w.x(0);
                let y0 = w.y(0);

                build_d_dynamic(
                    scratch.det0.as_mut_slice(),
                    l,
                    &x0,
                    &y0,
                    scratch.rows.as_slice(),
                    scratch.cols.as_slice(),
                );
            },
        )
    })
}

/// `Prepare the fixed-rank L contraction determinant \mathbf D_{\mathrm{ov}}(0,\ldots,0).`
/// The determinant labels are constructed directly from the bra and ket excitations, and the
/// selected compile-time rank uses the fixed-rank determinant fill order.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage receiving the determinant labels and contraction determinant.
/// # Returns
/// - `()`: Writes the rank-`L` determinant labels and contraction determinant.
#[inline(always)]
fn prepare_same_m0_const<T: NOCIScalar, const L: usize>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_prepare_same_m0_const,
        {
            scratch.ensure_same(L);

            construct_determinant_indices(
                l_ex,
                g_ex,
                w,
                scratch.rows.as_mut_slice(),
                scratch.cols.as_mut_slice(),
            );

            let x0 = w.x(0);
            let y0 = w.y(0);

            // Prepare `\mathbf D_{\mathrm{ov}}(0,\ldots,0)` with
            // `D_{ij} = X^{(0)}_{r_i c_j}` for `i \geq j` and
            // `D_{ij} = Y^{(0)}_{r_i c_j}` for `i < j`, using the fixed-rank fill order.
            build_d_const::<T, L>(
                scratch.det0.as_mut_slice(),
                &x0,
                &y0,
                scratch.rows.as_slice(),
                scratch.cols.as_slice(),
            );
        }
    )
}

/// Prepare the two contraction determinants required when m > 0:
/// `\mathbf D_{\mathrm{ov}}(0,\ldots,0) and \mathbf D_{\mathrm{ov}}(1,\ldots,1).`
/// `Mixed distributions are formed later by selecting each column according to m_j \in \{0,1\}.`
/// # Arguments:
/// - `w`: Same-spin reference-pair Wick intermediates.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage receiving the determinant labels and both contraction determinants.
/// # Returns
/// - `()`: Writes both contraction determinants into `scratch.det0` and `scratch.det1`.
pub fn prepare_same_gen<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
) {
    time_call!(crate::timers::nonorthogonalwicks::add_prepare_same_gen, {
        // For m > 0, later GNME sums need both endpoint determinants so each mixed
        // distribution can choose every column from `\mathbf D_{\mathrm{ov}}(0,\ldots,0)` or
        // `\mathbf D_{\mathrm{ov}}(1,\ldots,1)`.
        let l = l_ex.holes.count_ones() as usize + g_ex.holes.count_ones() as usize;
        scratch.ensure_same(l);

        construct_determinant_indices(
            l_ex,
            g_ex,
            w,
            scratch.rows.as_mut_slice(),
            scratch.cols.as_mut_slice(),
        );

        let x0 = w.x(0);
        let y0 = w.y(0);
        build_d_dynamic(
            scratch.det0.as_mut_slice(),
            l,
            &x0,
            &y0,
            scratch.rows.as_slice(),
            scratch.cols.as_slice(),
        );

        let x1 = w.x(1);
        let y1 = w.y(1);
        build_d_dynamic(
            scratch.det1.as_mut_slice(),
            l,
            &x1,
            &y1,
            scratch.rows.as_slice(),
            scratch.cols.as_slice(),
        );
    })
}

/// Construct the row and column labels for an arbitrary-rank contraction determinant.
/// `For L_x bra-reference and L_w ket-reference excitation pairs:`
/// `(r_k,c_k) = (a_k,i_k) for 0 \leq k < L_x,`
/// `(r_{L_x+k},c_{L_x+k}) = (j_k,b_k) for 0 \leq k < L_w.`
/// `Hence the row space is V_x \cup O_w and the column space is O_x \cup V_w.`
/// # Arguments:
/// - `x_ex`: `Excitation defining the bra determinant generated from \langle{}^x\Psi|.`
/// - `w_ex`: `Excitation defining the ket determinant generated from |{}^w\Psi\rangle.`
/// - `w`: Same-spin intermediates containing the compact occupied and virtual block dimensions.
/// - `rows`: `Output labels in V_x \cup O_w.`
/// - `cols`: `Output labels in O_x \cup V_w.`
/// # Returns
/// - `()`: Writes the ordered contraction-determinant labels into `rows` and `cols`.
#[inline(always)]
pub(super) fn construct_determinant_indices<T: NOCIScalar>(
    x_ex: &ExcitationSpin,
    w_ex: &ExcitationSpin,
    w: &SameSpinView<'_, T>,
    rows: &mut [usize],
    cols: &mut [usize],
) {
    time_call!(
        crate::timers::nonorthogonalwicks::add_construct_determinant_indices,
        {
            // `L_x` and `L_w` determine the two ordered blocks of the determinant labels.
            let nocc = w.nocc;
            let nvirt = w.nmo - nocc;
            let mut xh = x_ex.holes;
            let mut xp = x_ex.parts;
            let mut wh = w_ex.holes;
            let mut wp = w_ex.parts;
            let mut i = 0usize;
            // Map the x-reference pairs to the `V_x` row block and `O_x` column block.
            while xh != 0 {
                let hole = xh.trailing_zeros() as usize;
                let part = xp.trailing_zeros() as usize;
                xh &= xh - 1;
                xp &= xp - 1;
                rows[i] = part - nocc;
                cols[i] = hole;
                i += 1;
            }
            // Append the w-reference pairs in the `O_w` row block and `V_w` column block.
            while wh != 0 {
                let hole = wh.trailing_zeros() as usize;
                let part = wp.trailing_zeros() as usize;
                wh &= wh - 1;
                wp &= wp - 1;
                rows[i] = nvirt + hole;
                cols[i] = part;
                i += 1;
            }
        }
    )
}
