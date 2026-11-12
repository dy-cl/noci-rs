// elements/pairs.rs
//! Dispatch of determinant-pair matrix elements to the orthogonal, Wick and naive engines.
//!
//! Every pair is routed by its parents. Same-parent pairs connected beyond the operator rank
//! vanish, the remaining same-parent pairs of an orthonormal parent use the orthogonal
//! Slater-Condon rules, and pairs of different parents use the extended nonorthogonal Wick
//! theorem when its intermediates are enabled or the generalised Slater-Condon rules otherwise.
//! Batches are grouped by parent and by ordered reference pair so that each engine receives the
//! requests its SIMD kernels share.

// Crate-root imports.
use crate::NOCIScalar;
use crate::determinant::DeterminantState;
use crate::elements::{DetPair, NOCIData};
use crate::time_call;

// Parent/sibling imports.
use super::nonorthogonalwicks::WickScratchSpin;
use super::operator::PairOperator;
use super::orthogonal::{calculate_pairs_orthogonal_batched, connection_data};

/// Engine that evaluates one determinant pair.
enum Route {
    /// Same-parent pair connected beyond the operator rank, so the element vanishes.
    Zero,
    /// Same-parent pair of an orthonormal parent, evaluated by the orthogonal rules.
    Orthogonal,
    /// Pair evaluated by the extended nonorthogonal Wick theorem.
    Wicks,
    /// Pair evaluated by the generalised Slater-Condon rules.
    Naive,
}

/// Choose the engine that evaluates one determinant pair.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `op`: Operator whose matrix element is requested.
/// - `wicks`: Whether Wick intermediates are available for nonorthogonal pairs.
/// - `pair`: Determinant pair.
/// # Returns:
/// - `Route`: Engine for the pair.
fn route<T, Op>(
    data: &NOCIData<'_, T>,
    op: &Op,
    wicks: bool,
    pair: DetPair,
) -> Route
where
    T: NOCIScalar,
    Op: PairOperator<T>,
{
    let lp = pair.bra.parent;
    if lp == pair.ket.parent {
        // An operator of rank `R` cannot connect determinants of one orthonormal frame that
        // differ by more than `R` excitations, `2R` occupation bits.
        let l_occ = (pair.bra.alpha, pair.bra.beta);
        let g_occ = (pair.ket.alpha, pair.ket.beta);
        let differing = (l_occ.0 ^ g_occ.0).count_ones() + (l_occ.1 ^ g_occ.1).count_ones();
        if differing as usize > 2 * Op::RANK {
            return Route::Zero;
        }
        if op.orthonormal(data, lp) {
            return Route::Orthogonal;
        }
    }

    if wicks { Route::Wicks } else { Route::Naive }
}

/// Evaluate the matrix elements of an operator for a batch of determinant pairs. A single pair
/// is evaluated directly. Larger batches group same-parent pairs by parent for the orthogonal
/// kernels and Wick pairs by ordered reference pair for the nonorthogonal kernels.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `op`: Operator whose matrix elements are requested.
/// - `pairs`: Determinant pairs.
/// - `scratch`: Scratch space for Wick's calculations.
/// - `out`: Matrix elements in the same order as `pairs`.
/// # Returns:
/// - `()`: Writes every requested matrix element into `out`.
/// # Panics
/// - Panics if a Wick pair is requested without Wick scratch space.
pub(crate) fn calculate_pairs<T, Op>(
    data: &NOCIData<'_, T>,
    op: &Op,
    pairs: &[DetPair],
    scratch: Option<&mut WickScratchSpin<T>>,
    out: &mut [Op::Out],
) where
    T: NOCIScalar,
    Op: PairOperator<T>,
{
    time_call!(Op::TIMER, {
        // A single pair has no other requests to fill SIMD lanes, so evaluate it directly.
        if let [pair] = pairs {
            out[0] = calculate_pair(data, op, *pair, scratch);
            return;
        }

        let space = data.space;
        let wicks = data.wicks.filter(|_| data.input.wicks.enabled);
        let nref = wicks.map_or(0, |wicks| wicks.nref);
        let mut orthogonal = vec![Vec::new(); space.parents.len()];
        let mut nonorthogonal = vec![Vec::new(); nref * nref];

        // Place every request into exactly one parent group or ordered reference-pair group, or
        // evaluate it immediately when neither batched path applies.
        for (output, &pair) in pairs.iter().enumerate() {
            match route(data, op, wicks.is_some(), pair) {
                Route::Zero => out[output] = Op::zero(),
                Route::Orthogonal => orthogonal[pair.bra.parent].push((output, pair)),
                Route::Wicks => {
                    let group = pair.bra.parent * nref + pair.ket.parent;
                    nonorthogonal[group].push((output, pair));
                }
                Route::Naive => {
                    out[output] = op.naive(data, &pair.bra, &pair.ket);
                }
            }
        }

        // Evaluate same-parent groups with the orthogonal Slater-Condon kernels.
        for (parent, requests) in orthogonal.iter().enumerate() {
            if requests.is_empty() {
                continue;
            }
            calculate_pairs_orthogonal_batched(
                requests,
                Op::RANK,
                Op::zero(),
                out,
                |occupations, connections, values| {
                    op.orthogonal(data, parent, occupations, connections, values)
                },
            );
        }

        // Each nonempty Wick group contains only requests belonging to one reference pair.
        if let Some(wicks) = wicks {
            let scratch = scratch.expect("Wick pairs require Wick scratch space");
            for (group, requests) in nonorthogonal.iter().enumerate() {
                if requests.is_empty() {
                    continue;
                }
                let w = wicks.pair(group / nref, group % nref);
                op.wicks(data, &w, requests, scratch, out);
            }
        }
    })
}

/// Evaluate the matrix element of an operator for one determinant pair through its engine.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `op`: Operator whose matrix element is requested.
/// - `pair`: Determinant pair.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `Op::Out`: Matrix element of the pair.
/// # Panics
/// - Panics if the pair needs Wick's theorem and no Wick scratch space is given.
fn calculate_pair<T, Op>(
    data: &NOCIData<'_, T>,
    op: &Op,
    pair: DetPair,
    scratch: Option<&mut WickScratchSpin<T>>,
) -> Op::Out
where
    T: NOCIScalar,
    Op: PairOperator<T>,
{
    let wicks = data.wicks.filter(|_| data.input.wicks.enabled);
    let mut value = [Op::zero()];

    // Use the same routing as a batch, but pass a single connection or Wick request directly.
    match route(data, op, wicks.is_some(), pair) {
        Route::Zero => {}
        Route::Orthogonal => {
            let g_occ = (pair.ket.alpha, pair.ket.beta);

            // Measure the bra from the ket in their shared frame. A missing connection leaves
            // the element zero; a valid one carries both its orbital labels and fermionic phase.
            if let Some(connection) =
                connection_data((pair.bra.alpha, pair.bra.beta), g_occ, Op::RANK)
            {
                let parent = pair.bra.parent;
                op.orthogonal(data, parent, &[g_occ], &[connection], &mut value);
            }
        }
        Route::Wicks => {
            // Wick intermediates belong to an ordered parent pair, even for a single request.
            let w = wicks.unwrap().pair(pair.bra.parent, pair.ket.parent);
            let scratch = scratch.expect("Wick pairs require Wick scratch space");
            op.wicks(data, &w, &[(0, pair)], scratch, &mut value);
        }
        Route::Naive => {
            value[0] = op.naive(data, &pair.bra, &pair.ket);
        }
    }

    value[0]
}

/// Dense counterpart of `calculate_pairs`, which fills one matrix row. The ket determinants of
/// the row are pre-sorted into runs sharing a parent, so each run is passed to its engine
/// without regrouping the requests.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `op`: Operator whose matrix elements are requested.
/// - `ldet`: Bra determinant of the row.
/// - `kets`: Ket determinants of every column, and the parent and ascending column indices of
///   each run of ket determinants.
/// - `first`: First column to evaluate; earlier columns are left untouched.
/// - `scratch`: Scratch space for Wick's calculations.
/// - `out`: Matrix elements of every column of the row.
/// # Returns:
/// - `()`: Writes the requested elements of the row into `out`.
/// # Panics
/// - Panics if a Wick column is requested without Wick scratch space.
pub(crate) fn calculate_pairs_dense<T, Op>(
    data: &NOCIData<'_, T>,
    op: &Op,
    ldet: DeterminantState,
    kets: (&[DeterminantState], &[(usize, Vec<usize>)]),
    first: usize,
    mut scratch: Option<&mut WickScratchSpin<T>>,
    out: &mut [Op::Out],
) where
    T: NOCIScalar,
    Op: PairOperator<T>,
{
    time_call!(Op::TIMER, {
        let (right, runs) = kets;
        let wicks = data.wicks.filter(|_| data.input.wicks.enabled);
        let lp = ldet.parent;
        let mut orthogonal = Vec::new();
        let mut nonorthogonal = Vec::new();

        for (gp, columns) in runs {
            // Columns are ascending, so the requested ones form a suffix.
            let columns = &columns[columns.partition_point(|&j| j < first)..];

            // Route every column of the run to its engine, evaluating naive columns at once.
            orthogonal.clear();
            nonorthogonal.clear();
            for &j in columns {
                let pair = DetPair::new(ldet, right[j]);
                match route(data, op, wicks.is_some(), pair) {
                    Route::Zero => out[j] = Op::zero(),
                    Route::Orthogonal => orthogonal.push((j, pair)),
                    Route::Wicks => nonorthogonal.push((j, pair)),
                    Route::Naive => out[j] = op.naive(data, &ldet, &right[j]),
                }
            }

            // Evaluate the orthogonal and Wick columns of the run as one batch each.
            if !orthogonal.is_empty() {
                calculate_pairs_orthogonal_batched(
                    &orthogonal,
                    Op::RANK,
                    Op::zero(),
                    out,
                    |occupations, connections, values| {
                        op.orthogonal(data, lp, occupations, connections, values)
                    },
                );
            }
            if let Some(wicks) = wicks
                && !nonorthogonal.is_empty()
            {
                let scratch = scratch
                    .as_deref_mut()
                    .expect("Wick pairs require Wick scratch space");
                op.wicks(data, &wicks.pair(lp, *gp), &nonorthogonal, scratch, out);
            }
        }
    })
}

/// Compare the Wick and generalised Slater-Condon evaluations of one matrix element.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `op`: Operator whose matrix element is compared.
/// - `pair`: Determinant pair.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(Op::Out, (f64, f64))`: Wick matrix element, with the total and largest elementwise
///   discrepancy from the generalised Slater-Condon element.
/// # Panics
/// - Panics if no Wick intermediates are available.
pub(crate) fn compare_pairs_wicks_naive<T, Op>(
    data: &NOCIData<'_, T>,
    op: &Op,
    pair: DetPair,
    scratch: &mut WickScratchSpin<T>,
) -> (Op::Out, (f64, f64))
where
    T: NOCIScalar,
    Op: PairOperator<T>,
{
    let w = data
        .wicks
        .expect("comparison requires Wick intermediates")
        .pair(pair.bra.parent, pair.ket.parent);

    let mut wick = [Op::zero()];

    // Evaluate the same occupations in both engines, then compare with the operator's metric
    // (including both H and S when the output is a pair).
    op.wicks(data, &w, &[(0, pair)], scratch, &mut wick);
    let naive = op.naive(data, &pair.bra, &pair.ket);

    (wick[0], Op::discrepancy(wick[0], naive))
}
