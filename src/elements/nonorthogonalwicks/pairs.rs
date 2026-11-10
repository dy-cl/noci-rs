// elements/nonorthogonalwicks/pairs.rs
//! Determinant-pair matrix elements from the extended nonorthogonal Wick theorem.

// Crate-root imports.
use crate::NOCIScalar;
use crate::determinant::ExcitedState;

// Parent/sibling imports.
use super::{WickScratchSpin, WicksPairView, xw_f_overlap_prepared, xw_overlap_prepared};

/// Calculate the overlap matrix element `S_{xw} = S^\alpha_{xw} S^\beta_{xw}` between two prepared
/// determinants using the extended nonorthogonal Wick theorem.
/// # Arguments:
/// - `w`: Wick intermediates for the ordered parent pair of the determinants.
/// - `bra`: Prepared bra determinant x.
/// - `ket`: Prepared ket determinant w.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `T`: Overlap matrix element.
pub(in crate::elements) fn calculate_s_pair_wicks<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    bra: &ExcitedState,
    ket: &ExcitedState,
    scratch: &mut WickScratchSpin<T>,
) -> T {
    let zero = T::from_real(0.0);
    let (lex, gex) = (&bra.excitation, &ket.excitation);

    // A zero-overlap count larger than the available excitation rank makes the corresponding
    // Wick determinant vanish.
    let ra = lex.alpha.holes.count_ones() as usize + gex.alpha.holes.count_ones() as usize;
    let rb = lex.beta.holes.count_ones() as usize + gex.beta.holes.count_ones() as usize;
    if w.aa.m > ra || w.bb.m > rb {
        return zero;
    }

    // The determinant product factorises into alpha and beta same-spin overlaps, each carrying
    // the product of its spin phases.
    let sa = T::from_real(bra.phase_a * ket.phase_a)
        * xw_overlap_prepared(&w.aa, &lex.alpha, &gex.alpha, &mut scratch.aa);
    if sa == zero {
        return zero;
    }
    let sb = T::from_real(bra.phase_b * ket.phase_b)
        * xw_overlap_prepared(&w.bb, &lex.beta, &gex.beta, &mut scratch.bb);
    sa * sb
}

/// Calculate the Fock and overlap matrix elements between two prepared determinants using the
/// extended nonorthogonal Wick theorem. The one-body element separates over spin sectors as
/// `F_{xw} = F^\alpha_{xw} S^\beta_{xw} + S^\alpha_{xw} F^\beta_{xw}`.
/// # Arguments:
/// - `w`: Wick intermediates for the ordered parent pair of the determinants.
/// - `bra`: Prepared bra determinant x.
/// - `ket`: Prepared ket determinant w.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, T)`: Fock and overlap matrix elements.
pub(in crate::elements) fn calculate_fs_pair_wicks<T: NOCIScalar>(
    w: &WicksPairView<'_, T>,
    bra: &ExcitedState,
    ket: &ExcitedState,
    tol: f64,
    scratch: &mut WickScratchSpin<T>,
) -> (T, T) {
    let (lex, gex) = (&bra.excitation, &ket.excitation);
    let pha = T::from_real(bra.phase_a * ket.phase_a);
    let phb = T::from_real(bra.phase_b * ket.phase_b);

    // Prepared Wick contractions supply each spin's overlap and Fock factors.
    let (sa, fa) = xw_f_overlap_prepared(&w.aa, &lex.alpha, &gex.alpha, &mut scratch.aa, tol);
    let (sb, fb) = xw_f_overlap_prepared(&w.bb, &lex.beta, &gex.beta, &mut scratch.bb, tol);
    let sa = pha * sa;
    let sb = phb * sb;

    (pha * fa * sb + phb * fb * sa, sa * sb)
}
