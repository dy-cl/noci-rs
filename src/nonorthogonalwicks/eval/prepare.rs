// nonorthogonalwicks/eval/prepare.rs
// Crate-root imports.
use crate::ExcitationSpin;
use crate::maths::build_d;
use crate::noci::NOCIScalar;
use crate::time_call;

// Parent/sibling imports.
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;

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
        // \mathbf D_{\mathrm{ov}}(0,\ldots,0). Otherwise prepare both required contraction determinants.
        if w.m == 0 {
            prepare_same_m0(w, l_ex, g_ex, scratch)
        } else {
            prepare_same_gen(w, l_ex, g_ex, scratch)
        }
    })
}

/// `Prepare \mathbf D_{\mathrm{ov}}(0,\ldots,0) when m = 0, so the reference pair contains`
/// `no zero-overlap orbital pairs and every column assignment is m_j = 0. Fixed-rank kernels are`
/// `used for L = 1,\ldots,6; arbitrary ranks use the general determinant builder.`
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: `Scratch storage receiving the determinant labels and \mathbf D_{\mathrm{ov}}(0,\ldots,0).`
/// # Returns
/// - `()`: `Writes the determinant labels and m_j = 0 contraction determinant.`
#[inline(always)]
fn prepare_same_m0<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
) {
    time_call!(crate::timers::nonorthogonalwicks::add_prepare_same_m0, {
        let l = l_ex.holes.count_ones() as usize + g_ex.holes.count_ones() as usize;

        match l {
            0 => {}
            1 => prepare_same_m0_l1(w, l_ex, g_ex, scratch),
            2 => prepare_same_m0_l2(w, l_ex, g_ex, scratch),
            3 => prepare_same_m0_l3(w, l_ex, g_ex, scratch),
            4 => prepare_same_m0_l4(w, l_ex, g_ex, scratch),
            5 => prepare_same_m0_l5(w, l_ex, g_ex, scratch),
            6 => prepare_same_m0_l6(w, l_ex, g_ex, scratch),
            _ => {
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
    })
}

/// `Prepare the fixed-rank L = 1 contraction determinant \mathbf D_{\mathrm{ov}}(0).`
/// The determinant labels are constructed directly from the bra and ket excitations.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage receiving the determinant labels and contraction determinant.
/// # Returns
/// - `()`: Writes the rank-1 determinant labels and contraction determinant.
#[inline(always)]
fn prepare_same_m0_l1<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
) {
    time_call!(crate::timers::nonorthogonalwicks::add_prepare_same_m0_l1, {
        scratch.ensure_same_m0(1);
        scratch.rows.ensure(1);
        scratch.cols.ensure(1);

        construct_determinant_indices(
            l_ex,
            g_ex,
            w,
            scratch.rows.as_mut_slice(),
            scratch.cols.as_mut_slice(),
        );

        let rows = scratch.rows.as_slice();
        let cols = scratch.cols.as_slice();

        unsafe {
            let r0 = *rows.get_unchecked(0) * w.n();
            let c0 = *cols.get_unchecked(0);

            *scratch.det0.as_mut_slice().get_unchecked_mut(0) =
                *w.x_slice(0).get_unchecked(r0 + c0);
        }
    })
}

/// `Prepare the fixed-rank L = 2 contraction determinant \mathbf D_{\mathrm{ov}}(0,\ldots,0).`
/// The determinant labels are constructed directly from the bra and ket excitations.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage receiving the determinant labels and contraction determinant.
/// # Returns
/// - `()`: Writes the rank-2 determinant labels and contraction determinant.
#[inline(always)]
fn prepare_same_m0_l2<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
) {
    time_call!(crate::timers::nonorthogonalwicks::add_prepare_same_m0_l2, {
        scratch.ensure_same_m0(2);
        scratch.rows.ensure(2);
        scratch.cols.ensure(2);

        construct_determinant_indices(
            l_ex,
            g_ex,
            w,
            scratch.rows.as_mut_slice(),
            scratch.cols.as_mut_slice(),
        );

        let rows = scratch.rows.as_slice();
        let cols = scratch.cols.as_slice();
        let x0 = w.x_slice(0);
        let y0 = w.y_slice(0);
        let n = w.n();
        let det0 = scratch.det0.as_mut_slice();

        unsafe {
            let r0 = *rows.get_unchecked(0) * n;
            let r1 = *rows.get_unchecked(1) * n;

            let c0 = *cols.get_unchecked(0);
            let c1 = *cols.get_unchecked(1);

            *det0.get_unchecked_mut(0) = *x0.get_unchecked(r0 + c0);
            *det0.get_unchecked_mut(1) = *y0.get_unchecked(r0 + c1);

            *det0.get_unchecked_mut(2) = *x0.get_unchecked(r1 + c0);
            *det0.get_unchecked_mut(3) = *x0.get_unchecked(r1 + c1);
        }
    })
}

/// `Prepare the fixed-rank L = 3 contraction determinant \mathbf D_{\mathrm{ov}}(0,\ldots,0).`
/// The determinant labels are constructed directly from the bra and ket excitations.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage receiving the determinant labels and contraction determinant.
/// # Returns
/// - `()`: Writes the rank-3 determinant labels and contraction determinant.
#[inline(always)]
fn prepare_same_m0_l3<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
) {
    time_call!(crate::timers::nonorthogonalwicks::add_prepare_same_m0_l3, {
        scratch.ensure_same(3);

        construct_determinant_indices(
            l_ex,
            g_ex,
            w,
            scratch.rows.as_mut_slice(),
            scratch.cols.as_mut_slice(),
        );

        let rows = scratch.rows.as_slice();
        let cols = scratch.cols.as_slice();
        let x0 = w.x_slice(0);
        let y0 = w.y_slice(0);
        let n = w.n();
        let det0 = scratch.det0.as_mut_slice();

        unsafe {
            let r0 = *rows.get_unchecked(0) * n;
            let r1 = *rows.get_unchecked(1) * n;
            let r2 = *rows.get_unchecked(2) * n;

            let c0 = *cols.get_unchecked(0);
            let c1 = *cols.get_unchecked(1);
            let c2 = *cols.get_unchecked(2);

            *det0.get_unchecked_mut(0) = *x0.get_unchecked(r0 + c0);
            *det0.get_unchecked_mut(1) = *y0.get_unchecked(r0 + c1);
            *det0.get_unchecked_mut(2) = *y0.get_unchecked(r0 + c2);

            *det0.get_unchecked_mut(3) = *x0.get_unchecked(r1 + c0);
            *det0.get_unchecked_mut(4) = *x0.get_unchecked(r1 + c1);
            *det0.get_unchecked_mut(5) = *y0.get_unchecked(r1 + c2);

            *det0.get_unchecked_mut(6) = *x0.get_unchecked(r2 + c0);
            *det0.get_unchecked_mut(7) = *x0.get_unchecked(r2 + c1);
            *det0.get_unchecked_mut(8) = *x0.get_unchecked(r2 + c2);
        }
    })
}

/// `Prepare the fixed-rank L = 4 contraction determinant \mathbf D_{\mathrm{ov}}(0,\ldots,0).`
/// The determinant labels are constructed directly from the bra and ket excitations.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage receiving the determinant labels and contraction determinant.
/// # Returns
/// - `()`: Writes the rank-4 determinant labels and contraction determinant.
#[inline(always)]
fn prepare_same_m0_l4<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
) {
    time_call!(crate::timers::nonorthogonalwicks::add_prepare_same_m0_l4, {
        scratch.ensure_same(4);

        construct_determinant_indices(
            l_ex,
            g_ex,
            w,
            scratch.rows.as_mut_slice(),
            scratch.cols.as_mut_slice(),
        );

        let rows = scratch.rows.as_slice();
        let cols = scratch.cols.as_slice();
        let x0 = w.x_slice(0);
        let y0 = w.y_slice(0);
        let n = w.n();
        let det0 = scratch.det0.as_mut_slice();

        unsafe {
            let r0 = *rows.get_unchecked(0) * n;
            let r1 = *rows.get_unchecked(1) * n;
            let r2 = *rows.get_unchecked(2) * n;
            let r3 = *rows.get_unchecked(3) * n;

            let c0 = *cols.get_unchecked(0);
            let c1 = *cols.get_unchecked(1);
            let c2 = *cols.get_unchecked(2);
            let c3 = *cols.get_unchecked(3);

            *det0.get_unchecked_mut(0) = *x0.get_unchecked(r0 + c0);
            *det0.get_unchecked_mut(1) = *y0.get_unchecked(r0 + c1);
            *det0.get_unchecked_mut(2) = *y0.get_unchecked(r0 + c2);
            *det0.get_unchecked_mut(3) = *y0.get_unchecked(r0 + c3);

            *det0.get_unchecked_mut(4) = *x0.get_unchecked(r1 + c0);
            *det0.get_unchecked_mut(5) = *x0.get_unchecked(r1 + c1);
            *det0.get_unchecked_mut(6) = *y0.get_unchecked(r1 + c2);
            *det0.get_unchecked_mut(7) = *y0.get_unchecked(r1 + c3);

            *det0.get_unchecked_mut(8) = *x0.get_unchecked(r2 + c0);
            *det0.get_unchecked_mut(9) = *x0.get_unchecked(r2 + c1);
            *det0.get_unchecked_mut(10) = *x0.get_unchecked(r2 + c2);
            *det0.get_unchecked_mut(11) = *y0.get_unchecked(r2 + c3);

            *det0.get_unchecked_mut(12) = *x0.get_unchecked(r3 + c0);
            *det0.get_unchecked_mut(13) = *x0.get_unchecked(r3 + c1);
            *det0.get_unchecked_mut(14) = *x0.get_unchecked(r3 + c2);
            *det0.get_unchecked_mut(15) = *x0.get_unchecked(r3 + c3);
        }
    })
}

/// `Prepare the fixed-rank L = 5 contraction determinant \mathbf D_{\mathrm{ov}}(0,\ldots,0).`
/// The determinant labels are constructed directly from the bra and ket excitations.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage receiving the determinant labels and contraction determinant.
/// # Returns
/// - `()`: Writes the rank-5 determinant labels and contraction determinant.
#[inline(always)]
fn prepare_same_m0_l5<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
) {
    scratch.ensure_same(5);

    construct_determinant_indices(
        l_ex,
        g_ex,
        w,
        scratch.rows.as_mut_slice(),
        scratch.cols.as_mut_slice(),
    );

    let rows = scratch.rows.as_slice();
    let cols = scratch.cols.as_slice();
    let x0 = w.x_slice(0);
    let y0 = w.y_slice(0);
    let n = w.n();
    let det0 = scratch.det0.as_mut_slice();

    unsafe {
        let r0 = *rows.get_unchecked(0) * n;
        let r1 = *rows.get_unchecked(1) * n;
        let r2 = *rows.get_unchecked(2) * n;
        let r3 = *rows.get_unchecked(3) * n;
        let r4 = *rows.get_unchecked(4) * n;

        let c0 = *cols.get_unchecked(0);
        let c1 = *cols.get_unchecked(1);
        let c2 = *cols.get_unchecked(2);
        let c3 = *cols.get_unchecked(3);
        let c4 = *cols.get_unchecked(4);

        *det0.get_unchecked_mut(0) = *x0.get_unchecked(r0 + c0);
        *det0.get_unchecked_mut(1) = *y0.get_unchecked(r0 + c1);
        *det0.get_unchecked_mut(2) = *y0.get_unchecked(r0 + c2);
        *det0.get_unchecked_mut(3) = *y0.get_unchecked(r0 + c3);
        *det0.get_unchecked_mut(4) = *y0.get_unchecked(r0 + c4);

        *det0.get_unchecked_mut(5) = *x0.get_unchecked(r1 + c0);
        *det0.get_unchecked_mut(6) = *x0.get_unchecked(r1 + c1);
        *det0.get_unchecked_mut(7) = *y0.get_unchecked(r1 + c2);
        *det0.get_unchecked_mut(8) = *y0.get_unchecked(r1 + c3);
        *det0.get_unchecked_mut(9) = *y0.get_unchecked(r1 + c4);

        *det0.get_unchecked_mut(10) = *x0.get_unchecked(r2 + c0);
        *det0.get_unchecked_mut(11) = *x0.get_unchecked(r2 + c1);
        *det0.get_unchecked_mut(12) = *x0.get_unchecked(r2 + c2);
        *det0.get_unchecked_mut(13) = *y0.get_unchecked(r2 + c3);
        *det0.get_unchecked_mut(14) = *y0.get_unchecked(r2 + c4);

        *det0.get_unchecked_mut(15) = *x0.get_unchecked(r3 + c0);
        *det0.get_unchecked_mut(16) = *x0.get_unchecked(r3 + c1);
        *det0.get_unchecked_mut(17) = *x0.get_unchecked(r3 + c2);
        *det0.get_unchecked_mut(18) = *x0.get_unchecked(r3 + c3);
        *det0.get_unchecked_mut(19) = *y0.get_unchecked(r3 + c4);

        *det0.get_unchecked_mut(20) = *x0.get_unchecked(r4 + c0);
        *det0.get_unchecked_mut(21) = *x0.get_unchecked(r4 + c1);
        *det0.get_unchecked_mut(22) = *x0.get_unchecked(r4 + c2);
        *det0.get_unchecked_mut(23) = *x0.get_unchecked(r4 + c3);
        *det0.get_unchecked_mut(24) = *x0.get_unchecked(r4 + c4);
    }
}

/// `Prepare the fixed-rank L = 6 contraction determinant \mathbf D_{\mathrm{ov}}(0,\ldots,0).`
/// The determinant labels are constructed directly from the bra and ket excitations.
/// # Arguments:
/// - `w`: Reference-pair Wick intermediates with no zero-overlap orbital pairs.
/// - `l_ex`: Excitation defining the bra determinant.
/// - `g_ex`: Excitation defining the ket determinant.
/// - `scratch`: Scratch storage receiving the determinant labels and contraction determinant.
/// # Returns
/// - `()`: Writes the rank-6 determinant labels and contraction determinant.
#[inline(always)]
fn prepare_same_m0_l6<T: NOCIScalar>(
    w: &SameSpinView<'_, T>,
    l_ex: &ExcitationSpin,
    g_ex: &ExcitationSpin,
    scratch: &mut WickScratch<T>,
) {
    scratch.ensure_same(6);

    construct_determinant_indices(
        l_ex,
        g_ex,
        w,
        scratch.rows.as_mut_slice(),
        scratch.cols.as_mut_slice(),
    );

    let rows = scratch.rows.as_slice();
    let cols = scratch.cols.as_slice();
    let x0 = w.x_slice(0);
    let y0 = w.y_slice(0);
    let n = w.n();
    let det0 = scratch.det0.as_mut_slice();

    unsafe {
        let r0 = *rows.get_unchecked(0) * n;
        let r1 = *rows.get_unchecked(1) * n;
        let r2 = *rows.get_unchecked(2) * n;
        let r3 = *rows.get_unchecked(3) * n;
        let r4 = *rows.get_unchecked(4) * n;
        let r5 = *rows.get_unchecked(5) * n;

        let c0 = *cols.get_unchecked(0);
        let c1 = *cols.get_unchecked(1);
        let c2 = *cols.get_unchecked(2);
        let c3 = *cols.get_unchecked(3);
        let c4 = *cols.get_unchecked(4);
        let c5 = *cols.get_unchecked(5);

        *det0.get_unchecked_mut(0) = *x0.get_unchecked(r0 + c0);
        *det0.get_unchecked_mut(1) = *y0.get_unchecked(r0 + c1);
        *det0.get_unchecked_mut(2) = *y0.get_unchecked(r0 + c2);
        *det0.get_unchecked_mut(3) = *y0.get_unchecked(r0 + c3);
        *det0.get_unchecked_mut(4) = *y0.get_unchecked(r0 + c4);
        *det0.get_unchecked_mut(5) = *y0.get_unchecked(r0 + c5);

        *det0.get_unchecked_mut(6) = *x0.get_unchecked(r1 + c0);
        *det0.get_unchecked_mut(7) = *x0.get_unchecked(r1 + c1);
        *det0.get_unchecked_mut(8) = *y0.get_unchecked(r1 + c2);
        *det0.get_unchecked_mut(9) = *y0.get_unchecked(r1 + c3);
        *det0.get_unchecked_mut(10) = *y0.get_unchecked(r1 + c4);
        *det0.get_unchecked_mut(11) = *y0.get_unchecked(r1 + c5);

        *det0.get_unchecked_mut(12) = *x0.get_unchecked(r2 + c0);
        *det0.get_unchecked_mut(13) = *x0.get_unchecked(r2 + c1);
        *det0.get_unchecked_mut(14) = *x0.get_unchecked(r2 + c2);
        *det0.get_unchecked_mut(15) = *y0.get_unchecked(r2 + c3);
        *det0.get_unchecked_mut(16) = *y0.get_unchecked(r2 + c4);
        *det0.get_unchecked_mut(17) = *y0.get_unchecked(r2 + c5);

        *det0.get_unchecked_mut(18) = *x0.get_unchecked(r3 + c0);
        *det0.get_unchecked_mut(19) = *x0.get_unchecked(r3 + c1);
        *det0.get_unchecked_mut(20) = *x0.get_unchecked(r3 + c2);
        *det0.get_unchecked_mut(21) = *x0.get_unchecked(r3 + c3);
        *det0.get_unchecked_mut(22) = *y0.get_unchecked(r3 + c4);
        *det0.get_unchecked_mut(23) = *y0.get_unchecked(r3 + c5);

        *det0.get_unchecked_mut(24) = *x0.get_unchecked(r4 + c0);
        *det0.get_unchecked_mut(25) = *x0.get_unchecked(r4 + c1);
        *det0.get_unchecked_mut(26) = *x0.get_unchecked(r4 + c2);
        *det0.get_unchecked_mut(27) = *x0.get_unchecked(r4 + c3);
        *det0.get_unchecked_mut(28) = *x0.get_unchecked(r4 + c4);
        *det0.get_unchecked_mut(29) = *y0.get_unchecked(r4 + c5);

        *det0.get_unchecked_mut(30) = *x0.get_unchecked(r5 + c0);
        *det0.get_unchecked_mut(31) = *x0.get_unchecked(r5 + c1);
        *det0.get_unchecked_mut(32) = *x0.get_unchecked(r5 + c2);
        *det0.get_unchecked_mut(33) = *x0.get_unchecked(r5 + c3);
        *det0.get_unchecked_mut(34) = *x0.get_unchecked(r5 + c4);
        *det0.get_unchecked_mut(35) = *x0.get_unchecked(r5 + c5);
    }
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
        build_d(
            scratch.det0.as_mut_slice(),
            l,
            &x0,
            &y0,
            scratch.rows.as_slice(),
            scratch.cols.as_slice(),
        );

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
            // L_x and L_w determine the two ordered blocks of the determinant labels.
            let nocc = w.nocc;
            let nvirt = w.nmo - nocc;
            let mut xh = x_ex.holes;
            let mut xp = x_ex.parts;
            let mut wh = w_ex.holes;
            let mut wp = w_ex.parts;
            let mut i = 0usize;
            // Map the x-reference pairs to the V_x row block and O_x column block.
            while xh != 0 {
                let hole = xh.trailing_zeros() as usize;
                let part = xp.trailing_zeros() as usize;
                xh &= xh - 1;
                xp &= xp - 1;
                rows[i] = part - nocc;
                cols[i] = hole;
                i += 1;
            }
            // Append the w-reference pairs in the O_w row block and V_w column block.
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

/// Construct fixed-rank `L = 1` contraction labels from decoded x- and w-excitation indices.
/// # Arguments:
/// - `x_rank`: Excitation rank relative to the x reference.
/// - `x_indices`: Cached x hole then particle orbital indices, `[i0..i3,a0..a3]`.
/// - `w_indices`: Cached w hole then particle orbital indices, `[j0..j3,b0..b3]`.
/// - `w`: Same-spin intermediates containing occupied and virtual block dimensions.
/// - `rows`: Output contraction-row labels.
/// - `cols`: Output contraction-column labels.
/// # Returns
/// - `()`: Writes one ordered contraction label pair.
#[inline(always)]
pub(super) fn construct_determinant_indices_l1<T: NOCIScalar>(
    x_rank: u8,
    x_indices: &[u8; 8],
    w_indices: &[u8; 8],
    w: &SameSpinView<'_, T>,
    rows: &mut [usize; 1],
    cols: &mut [usize; 1],
) {
    let nocc = w.nocc;
    let nvirt = w.nmo - nocc;
    match x_rank {
        0 => {
            rows[0] = nvirt + usize::from(w_indices[0]);
            cols[0] = usize::from(w_indices[4]);
        }
        1 => {
            rows[0] = usize::from(x_indices[4]) - nocc;
            cols[0] = usize::from(x_indices[0]);
        }
        _ => unreachable!(),
    }
}

/// Construct fixed-rank `L = 2` contraction labels from decoded x- and w-excitation indices.
/// # Arguments:
/// - `x_rank`: Excitation rank relative to the x reference.
/// - `x_indices`: Cached x hole then particle orbital indices, `[i0..i3,a0..a3]`.
/// - `w_indices`: Cached w hole then particle orbital indices, `[j0..j3,b0..b3]`.
/// - `w`: Same-spin intermediates containing occupied and virtual block dimensions.
/// - `rows`: Output contraction-row labels.
/// - `cols`: Output contraction-column labels.
/// # Returns
/// - `()`: Writes two ordered contraction label pairs.
#[inline(always)]
pub(super) fn construct_determinant_indices_l2<T: NOCIScalar>(
    x_rank: u8,
    x_indices: &[u8; 8],
    w_indices: &[u8; 8],
    w: &SameSpinView<'_, T>,
    rows: &mut [usize; 2],
    cols: &mut [usize; 2],
) {
    let nocc = w.nocc;
    let nvirt = w.nmo - nocc;
    match x_rank {
        0 => {
            rows[0] = nvirt + usize::from(w_indices[0]);
            cols[0] = usize::from(w_indices[4]);
            rows[1] = nvirt + usize::from(w_indices[1]);
            cols[1] = usize::from(w_indices[5]);
        }
        1 => {
            rows[0] = usize::from(x_indices[4]) - nocc;
            cols[0] = usize::from(x_indices[0]);
            rows[1] = nvirt + usize::from(w_indices[0]);
            cols[1] = usize::from(w_indices[4]);
        }
        2 => {
            rows[0] = usize::from(x_indices[4]) - nocc;
            cols[0] = usize::from(x_indices[0]);
            rows[1] = usize::from(x_indices[5]) - nocc;
            cols[1] = usize::from(x_indices[1]);
        }
        _ => unreachable!(),
    }
}

/// Construct fixed-rank `L = 3` contraction labels from decoded x- and w-excitation indices.
/// # Arguments:
/// - `x_rank`: Excitation rank relative to the x reference.
/// - `x_indices`: Cached x hole then particle orbital indices, `[i0..i3,a0..a3]`.
/// - `w_indices`: Cached w hole then particle orbital indices, `[j0..j3,b0..b3]`.
/// - `w`: Same-spin intermediates containing occupied and virtual block dimensions.
/// - `rows`: Output contraction-row labels.
/// - `cols`: Output contraction-column labels.
/// # Returns
/// - `()`: Writes three ordered contraction label pairs.
#[inline(always)]
pub(super) fn construct_determinant_indices_l3<T: NOCIScalar>(
    x_rank: u8,
    x_indices: &[u8; 8],
    w_indices: &[u8; 8],
    w: &SameSpinView<'_, T>,
    rows: &mut [usize; 3],
    cols: &mut [usize; 3],
) {
    let nocc = w.nocc;
    let nvirt = w.nmo - nocc;
    match x_rank {
        0 => {
            rows[0] = nvirt + usize::from(w_indices[0]);
            cols[0] = usize::from(w_indices[4]);
            rows[1] = nvirt + usize::from(w_indices[1]);
            cols[1] = usize::from(w_indices[5]);
            rows[2] = nvirt + usize::from(w_indices[2]);
            cols[2] = usize::from(w_indices[6]);
        }
        1 => {
            rows[0] = usize::from(x_indices[4]) - nocc;
            cols[0] = usize::from(x_indices[0]);
            rows[1] = nvirt + usize::from(w_indices[0]);
            cols[1] = usize::from(w_indices[4]);
            rows[2] = nvirt + usize::from(w_indices[1]);
            cols[2] = usize::from(w_indices[5]);
        }
        2 => {
            rows[0] = usize::from(x_indices[4]) - nocc;
            cols[0] = usize::from(x_indices[0]);
            rows[1] = usize::from(x_indices[5]) - nocc;
            cols[1] = usize::from(x_indices[1]);
            rows[2] = nvirt + usize::from(w_indices[0]);
            cols[2] = usize::from(w_indices[4]);
        }
        3 => {
            rows[0] = usize::from(x_indices[4]) - nocc;
            cols[0] = usize::from(x_indices[0]);
            rows[1] = usize::from(x_indices[5]) - nocc;
            cols[1] = usize::from(x_indices[1]);
            rows[2] = usize::from(x_indices[6]) - nocc;
            cols[2] = usize::from(x_indices[2]);
        }
        _ => unreachable!(),
    }
}

