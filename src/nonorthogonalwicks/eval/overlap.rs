// nonorthogonalwicks/eval/overlap.rs
use crate::ExcitationSpin;
use crate::maths::det;
use crate::time_call;
use crate::timers::nonorthogonalwicks as wick_timers;
use super::helpers::mix_dets_same;
use super::super::scratch::WickScratch;
use super::super::view::SameSpinView;

/// Calculate overlap matrix element between two determinants |{}^\Lambda \Psi\rangle and
/// |{}^\Gamma \Psi\rangle using the extended non-orthogonal Wick's theorem prescription. Utilises
/// a sum over possible ways to distribute zeros across the columns of the L by L determinant.
/// # Arguments:
/// `w`: SameSpin: same spin Wick's reference pair intermediates.
/// - `l_ex`: Spin resolved excitation array for |{}^\Lambda \Psi\rangle.
/// - `g_ex`: Spin resolved excitation array for |{}^\Gamma \Psi\rangle. 
/// - `scratch`: Scratch space for Wick's quantities.
/// # Returns
/// - `f64`: Overlap matrix element.
pub fn lg_overlap(w: &SameSpinView, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin, scratch: &mut WickScratch) -> f64 {
    time_call!(wick_timers::add_lg_overlap, {
        let l = l_ex.holes.len() + g_ex.holes.len();

        if w.m == 0 {
            return w.phase * w.tilde_s_prod * det(scratch.det0.as_slice(), l).unwrap_or(0.0);
        }
        if w.m == l {
            return w.phase * w.tilde_s_prod * det(scratch.det1.as_slice(), l).unwrap_or(0.0);
        }

        let mut acc = 0.0;
        mix_dets_same(w, l, 0, scratch, |_, scratch| {
            if let Some(d) = det(scratch.det_mix.as_slice(), l) {
                acc += d;
            }
        });
        w.phase * w.tilde_s_prod * acc
    })
}


