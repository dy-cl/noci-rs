// elements/excited.rs
//! Matrix elements between determinants identified by their occupations in a parent frame,
//! without a retained determinant space. The entry points take batches of pairs, matching
//! [`crate::elements::calculate_hs_pair`]: same-parent pairs are grouped by parent for the
//! orthogonal Slater-Condon kernels and the remaining pairs use the extended nonorthogonal Wick
//! theorem.

// Crate-root imports.
use crate::NOCIScalar;
use crate::determinant::{ExcitedDeterminant, ExcitedState};
use crate::elements::NOCIData;
use crate::time_call;

// Parent/sibling imports.
use super::nonorthogonalwicks::{
    WickScratchSpin, xw_hamiltonian_overlap_prepared, xw_overlap_prepared,
};
use super::orthogonal::{
    calculate_h_pair_orthogonal, prepare_orthogonal_state, xw_hamiltonian_orthogonal_batched,
};

/// Evaluate `(H_{lg}, S_{lg}) = (\langle l|\hat H|g\rangle, \langle l|g\rangle)` for a batch of
/// occupation-identified determinant pairs. Pairs of one orthonormal parent use the Slater-Condon
/// rules with `S_{lg} = \delta_{lg}`, grouped by parent so that the double excitations fill SIMD
/// packets; every other pair uses the extended nonorthogonal Wick theorem with the precomputed
/// intermediates of its ordered parent pair. A single pair is evaluated directly.
/// # Arguments:
/// - `data`: Shared data whose space holds the parent determinants.
/// - `pairs`: Bra and ket determinants.
/// - `scratch`: Scratch space for Wick's calculations.
/// - `out`: Hamiltonian and overlap matrix elements in the order of `pairs`.
/// # Returns:
/// - `()`: Writes every requested `(H, S)` into `out`.
/// # Panics:
/// - Panics if a nonorthogonal pair is requested while Wick's theorem is disabled.
pub(crate) fn calculate_hs_excited<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pairs: &[(ExcitedDeterminant, ExcitedDeterminant)],
    scratch: &mut WickScratchSpin<T>,
    out: &mut [(T, T)],
) {
    time_call!(crate::timers::noci::add_calculate_hs_pair, {
        // A single pair has no other requests to fill SIMD lanes, so evaluate it directly.
        if let [pair] = pairs {
            out[0] = calculate_hs_excited_single(data, (&pair.0, &pair.1), scratch);
            return;
        }

        let zero = T::from_real(0.0);
        let one = T::from_real(1.0);
        let mut groups = vec![Vec::new(); data.space.parents().len()];

        // Place every same-parent request of an orthonormal frame into its parent group, and
        // evaluate every other request immediately.
        for (output, (l, g)) in pairs.iter().enumerate() {
            if l.parent == g.parent {
                // A one- or two-body Hamiltonian cannot connect same-parent determinants
                // differing by more than two excitations (four occupation bits).
                if (l.alpha ^ g.alpha).count_ones() + (l.beta ^ g.beta).count_ones() > 4 {
                    out[output] = (zero, zero);
                    continue;
                }
                if let Some(mocache) = data.mocache
                    && mocache[l.parent].orthogonal_slater_condon
                {
                    groups[l.parent].push(output);
                    continue;
                }
            }
            out[output] = calculate_hs_excited_single(data, (l, g), scratch);
        }

        // Evaluate each parent group with the batched orthogonal kernels.
        let Some(mocache) = data.mocache else {
            return;
        };
        for (parent, outputs) in groups.iter().enumerate() {
            if outputs.is_empty() {
                continue;
            }
            let mut occupations = Vec::with_capacity(outputs.len());
            let mut states = Vec::with_capacity(outputs.len());
            let mut prepared = Vec::with_capacity(outputs.len());
            for &output in outputs {
                let (l, g) = &pairs[output];
                let s = if l == g { one } else { zero };
                match prepare_orthogonal_state((l.alpha, l.beta), (g.alpha, g.beta), 2) {
                    Some(state) => {
                        occupations.push((g.alpha, g.beta));
                        states.push(state);
                        prepared.push(output);
                    }
                    None => out[output] = (zero, s),
                }
            }
            let mut values = vec![zero; states.len()];
            xw_hamiltonian_orthogonal_batched(
                data.ao,
                &mocache[parent],
                &occupations,
                &states,
                &mut values,
            );
            for (&output, h) in prepared.iter().zip(values) {
                let (l, g) = &pairs[output];
                out[output] = (h, if l == g { one } else { zero });
            }
        }
    })
}

/// Evaluate `(H_{lg}, S_{lg})` for one occupation-identified pair.
/// # Arguments:
/// - `data`: Shared data whose space holds the parent determinants.
/// - `pair`: Bra and ket determinants.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements.
fn calculate_hs_excited_single<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: (&ExcitedDeterminant, &ExcitedDeterminant),
    scratch: &mut WickScratchSpin<T>,
) -> (T, T) {
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
    calculate_hs_states_single(data, (&l.state(parents), &g.state(parents)), scratch)
}

/// Evaluate `(H_{lg}, S_{lg})` for a batch of prepared determinant pairs with the extended
/// nonorthogonal Wick theorem, reusing their parent-relative excitations.
/// # Arguments:
/// - `data`: Shared data holding the Wick intermediates.
/// - `pairs`: Prepared bra and ket determinants.
/// - `scratch`: Scratch space for Wick's calculations.
/// - `out`: Hamiltonian and overlap matrix elements in the order of `pairs`.
/// # Returns:
/// - `()`: Writes every requested `(H, S)` into `out`.
/// # Panics:
/// - Panics if Wick's theorem is disabled.
pub(crate) fn calculate_hs_states<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pairs: &[(&ExcitedState, &ExcitedState)],
    scratch: &mut WickScratchSpin<T>,
    out: &mut [(T, T)],
) {
    for (pair, value) in pairs.iter().zip(out) {
        *value = calculate_hs_states_single(data, *pair, scratch);
    }
}

/// Evaluate `(H_{lg}, S_{lg})` between two prepared determinants with the extended
/// nonorthogonal Wick theorem.
/// # Arguments:
/// - `data`: Shared data holding the Wick intermediates.
/// - `pair`: Prepared bra and ket determinants.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, T)`: Hamiltonian and overlap matrix elements.
/// # Panics:
/// - Panics if Wick's theorem is disabled.
fn calculate_hs_states_single<T: NOCIScalar>(
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

/// Evaluate the overlaps `S_{lg} = \langle l|g\rangle` for a batch of prepared determinant pairs
/// of different parents.
/// # Arguments:
/// - `data`: Shared data holding the Wick intermediates.
/// - `pairs`: Prepared bra and ket determinants.
/// - `scratch`: Scratch space for Wick's calculations.
/// - `out`: Overlap matrix elements in the order of `pairs`.
/// # Returns:
/// - `()`: Writes every requested overlap into `out`.
/// # Panics:
/// - Panics if Wick's theorem is disabled.
pub(crate) fn calculate_s_states<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pairs: &[(&ExcitedState, &ExcitedState)],
    scratch: &mut WickScratchSpin<T>,
    out: &mut [T],
) {
    time_call!(crate::timers::noci::add_calculate_s_pair_wicks, {
        for (pair, value) in pairs.iter().zip(out) {
            *value = calculate_s_states_single(data, *pair, scratch);
        }
    })
}

/// Evaluate the overlap `S_{lg}` between two prepared determinants of different parents, which
/// factorises into same-spin Wick overlaps `S_{lg} = S^\alpha_{lg} S^\beta_{lg}`.
/// # Arguments:
/// - `data`: Shared data holding the Wick intermediates.
/// - `pair`: Prepared bra and ket determinants.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `T`: Overlap matrix element.
/// # Panics:
/// - Panics if Wick's theorem is disabled.
fn calculate_s_states_single<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: (&ExcitedState, &ExcitedState),
    scratch: &mut WickScratchSpin<T>,
) -> T {
    let (ls, gs) = pair;
    let zero = T::from_real(0.0);
    let wicks = data
        .wicks
        .filter(|_| data.input.wicks.enabled)
        .expect("nonorthogonal excited-determinant elements require wicks.enabled");
    let w = wicks.pair(ls.det.parent, gs.det.parent);
    let (lex, gex) = (&ls.excitation, &gs.excitation);

    // A zero-overlap count larger than the available excitation rank makes the corresponding
    // Wick determinant vanish.
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
}
