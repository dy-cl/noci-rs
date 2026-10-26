// elements/excited.rs

// Crate-root imports.
use crate::NOCIScalar;
use crate::determinant::{ExcitedDeterminant, ExcitedState};
use crate::elements::NOCIData;
use crate::time_call;

// Parent/sibling imports.
use super::nonorthogonalwicks::{
    WickScratchSpin, xw_hamiltonian_overlap_prepared, xw_overlap_prepared,
};
use super::orthogonal::calculate_h_pair_orthogonal;

/// Evaluate `(H_{lg}, S_{lg}) = (\langle l|\hat H|g\rangle, \langle l|g\rangle)` between two
/// occupation-identified determinants. Determinants of one orthonormal parent use the
/// Slater-Condon rules, with `S_{lg} = \delta_{lg}`; every other pair uses the extended
/// nonorthogonal Wick theorem with the precomputed intermediates of its ordered parent pair.
/// # Arguments:
/// - `data`: Shared data whose space holds the parent determinants.
/// - `pair`: Bra and ket determinants.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements.
/// # Panics:
/// - Panics if a nonorthogonal pair is requested while Wick's theorem is disabled.
pub(crate) fn calculate_hs_excited<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: (&ExcitedDeterminant, &ExcitedDeterminant),
    scratch: &mut WickScratchSpin<T>,
) -> (T, T) {
    time_call!(crate::timers::noci::add_calculate_hs_pair, {
        let (l, g) = pair;

        // Same orthonormal parent: Slater-Condon rules on the occupation strings.
        if l.parent == g.parent
            && let Some(mocache) = data.mocache
            && mocache[l.parent].orthogonal_slater_condon
        {
            let h = calculate_h_pair_orthogonal(
                data.ao,
                &mocache[l.parent],
                (l.alpha, l.beta),
                (g.alpha, g.beta),
            );
            let s = if l == g {
                T::from_real(1.0)
            } else {
                T::from_real(0.0)
            };
            return (h, s);
        }

        // Nonorthogonal pair: excitations relative to each parent through the pair's Wick
        // intermediates.
        let parents = data.space.parents();
        calculate_hs_states(data, (&l.state(parents), &g.state(parents)), scratch)
    })
}

/// Evaluate `(H_{lg}, S_{lg})` between two prepared determinants with the extended
/// nonorthogonal Wick theorem, reusing their parent-relative excitations.
/// # Arguments:
/// - `data`: Shared data holding the Wick intermediates.
/// - `pair`: Prepared bra and ket determinants.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements.
/// # Panics:
/// - Panics if Wick's theorem is disabled.
pub(crate) fn calculate_hs_states<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: (&ExcitedState, &ExcitedState),
    scratch: &mut WickScratchSpin<T>,
) -> (T, T) {
    let (ls, gs) = pair;
    let wicks = data
        .wicks
        .filter(|_| data.input.wicks.enabled)
        .expect("nonorthogonal excited-determinant elements require wicks.enabled");
    xw_hamiltonian_overlap_prepared(
        &wicks.pair(ls.det.parent, gs.det.parent),
        (&ls.excitation, &gs.excitation),
        (&ls.reduced.excitation_cache, &gs.reduced.excitation_cache),
        ls.reduced.phase * gs.reduced.phase,
        data.ao.enuc,
        scratch,
        data.tol,
    )
}

/// Evaluate the overlap `S_{lg} = \langle l|g\rangle` between two prepared determinants of
/// different parents, which factorises into same-spin Wick overlaps
/// `S_{lg} = S^\alpha_{lg} S^\beta_{lg}`.
/// # Arguments:
/// - `data`: Shared data holding the Wick intermediates.
/// - `pair`: Prepared bra and ket determinants.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `T`: Overlap matrix element.
/// # Panics:
/// - Panics if Wick's theorem is disabled.
pub(crate) fn calculate_s_states<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: (&ExcitedState, &ExcitedState),
    scratch: &mut WickScratchSpin<T>,
) -> T {
    time_call!(crate::timers::noci::add_calculate_s_pair_wicks, {
        let (ls, gs) = pair;
        let zero = T::from_real(0.0);
        let wicks = data
            .wicks
            .filter(|_| data.input.wicks.enabled)
            .expect("nonorthogonal excited-determinant elements require wicks.enabled");
        let w = wicks.pair(ls.det.parent, gs.det.parent);
        let (lex, gex) = (&ls.excitation, &gs.excitation);

        // A zero-overlap count larger than the available excitation rank makes the
        // corresponding Wick determinant vanish.
        let ra = lex.alpha.holes.count_ones() as usize + gex.alpha.holes.count_ones() as usize;
        let rb = lex.beta.holes.count_ones() as usize + gex.beta.holes.count_ones() as usize;
        if w.aa.m > ra || w.bb.m > rb {
            return zero;
        }

        let sa = T::from_real(ls.phase_a * gs.phase_a)
            * xw_overlap_prepared(&w.aa, &lex.alpha, &gex.alpha, &mut scratch.aa);
        if sa == zero {
            return zero;
        }
        let sb = T::from_real(ls.phase_b * gs.phase_b)
            * xw_overlap_prepared(&w.bb, &lex.beta, &gex.beta, &mut scratch.bb);
        sa * sb
    })
}
