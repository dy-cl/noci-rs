// elements/nonorthogonalwicks/pairs.rs
//! Determinant-pair overlap and Fock matrix elements from the extended nonorthogonal Wick
//! theorem.
//!
//! Both elements separate over spin sectors,
//! `S_{xw} = S^\alpha_{xw} S^\beta_{xw}` and
//! `F_{xw} = F^\alpha_{xw} S^\beta_{xw} + S^\alpha_{xw} F^\beta_{xw}`, so a batch of pairs of one
//! ordered reference pair is evaluated from its distinct same-spin factors. Each distinct pair of
//! spin components is evaluated once, and the distinct ket components of each bra component form
//! one same-spin factor row.

// Standard library imports.
use std::ops::Range;

// Crate-root imports.
use crate::NOCIScalar;
use crate::determinant::{ExcitedSpinState, NOCISpace};
use crate::elements::{DetPair, PairDeterminant};
use crate::hash::FastMap;

// Parent/sibling imports.
use super::scratch::WickScratch;
use super::view::SameSpinView;
use super::{
    OneBodyFactor, OverlapFactor, SpinFactor, SpinFactorRow, SpinSources, WickScratchSpin,
    WicksPairView, xw_spin_factor_row,
};

/// Distinct same-spin component pairs of a batch of determinant pairs, arranged as rows of one bra
/// component against its distinct ket components.
struct SpinPairRows {
    /// Bra component of each row, the slots of its factors and its distinct ket components.
    rows: Vec<(ExcitedSpinState, Range<usize>, SpinSources)>,
    /// Number of distinct component pairs over every row.
    nslot: usize,
    /// Slot of the component pair of each request.
    slots: Vec<usize>,
}

impl SpinPairRows {
    /// Group the same-spin component pairs of a batch into rows of distinct pairs.
    /// # Arguments:
    /// - `space`: Determinant space holding the parent frames.
    /// - `requests`: Output positions with their determinant pairs.
    /// - `occupations`: Same-spin bra and ket occupations of each request.
    /// - `alpha`: Whether to group the alpha rather than the beta components.
    /// # Returns
    /// - `Self`: Rows of distinct component pairs and the slot of every request.
    fn new<T, D>(
        space: &NOCISpace<T>,
        requests: &[(usize, DetPair<D>)],
        occupations: &[(u128, u128)],
        alpha: bool,
    ) -> Self
    where
        T: NOCIScalar,
        D: PairDeterminant<T>,
    {
        // Within one reference pair a spin component is identified by its occupation. Index the
        // distinct bra components as rows and, within each row, the distinct ket components,
        // preparing each component only the first time it is seen.
        let mut row_of: FastMap<u128, usize> = FastMap::default();
        let mut row_sources: Vec<(ExcitedSpinState, Vec<ExcitedSpinState>)> = Vec::new();
        let mut column_of: Vec<FastMap<u128, usize>> = Vec::new();
        let mut positions = Vec::with_capacity(requests.len());
        let mut previous: Option<(u128, usize)> = None;
        for (&(_, pair), &(bra, ket)) in requests.iter().zip(occupations) {
            // Consecutive requests usually share their bra, so check the previous row first.
            let row = match previous {
                Some((key, row)) if key == bra => row,
                _ => *row_of.entry(bra).or_insert_with(|| {
                    row_sources.push((pair.ldet.spin(space, alpha), Vec::new()));
                    column_of.push(FastMap::default());
                    row_sources.len() - 1
                }),
            };
            previous = Some((bra, row));

            let sources = &mut row_sources[row].1;
            let column = *column_of[row].entry(ket).or_insert_with(|| {
                sources.push(pair.gdet.spin(space, alpha));
                sources.len() - 1
            });
            positions.push((row, column));
        }

        // Lay the factor slots of the rows out contiguously, so each row writes one slice.
        let mut rows = Vec::with_capacity(row_sources.len());
        let mut nslot = 0;
        for (bra, kets) in row_sources {
            rows.push((bra, nslot..nslot + kets.len(), SpinSources::new(&kets)));
            nslot += kets.len();
        }
        let slots = positions
            .iter()
            .map(|&(row, column)| rows[row].1.start + column)
            .collect();

        Self { rows, nslot, slots }
    }

    /// Group the alpha and beta component pairs of a batch, reading each request's occupations
    /// once.
    /// # Arguments:
    /// - `space`: Determinant space holding the parent frames.
    /// - `requests`: Output positions with their determinant pairs.
    /// # Returns
    /// - `(Self, Self)`: Alpha and beta rows of distinct component pairs.
    fn spins<T, D>(
        space: &NOCISpace<T>,
        requests: &[(usize, DetPair<D>)],
    ) -> (Self, Self)
    where
        T: NOCIScalar,
        D: PairDeterminant<T>,
    {
        let (alpha, beta): (Vec<_>, Vec<_>) = requests
            .iter()
            .map(|(_, pair)| {
                let (la, lb) = pair.ldet.occupations(space);
                let (ga, gb) = pair.gdet.occupations(space);
                ((la, ga), (lb, gb))
            })
            .unzip();
        (
            Self::new(space, requests, &alpha, true),
            Self::new(space, requests, &beta, false),
        )
    }
}

/// Evaluate the factors of one same-spin component pair, including the excitation phases.
/// # Arguments:
/// - `factor`: Same-spin factor family.
/// - `w`: Same-spin Wick intermediates of the ordered reference pair.
/// - `pair`: Bra and ket spin components.
/// - `scratch`: Same-spin Wick scratch space.
/// # Returns:
/// - `[T; K]`: Factors of the component pair.
fn spin_factor<F: SpinFactor<K>, T: NOCIScalar, const K: usize>(
    factor: &F,
    w: &SameSpinView<'_, T>,
    pair: (ExcitedSpinState, ExcitedSpinState),
    scratch: &mut WickScratch<T>,
) -> [T; K] {
    let (bra, ket) = pair;
    let phase = T::from_real(bra.state.phase * ket.state.phase);
    factor
        .values(w, (&bra, &ket), scratch)
        .map(|value| phase * value)
}

/// Evaluate the distinct same-spin factors of a batch, including the excitation phases, one row
/// per distinct bra component.
/// # Arguments:
/// - `factor`: Same-spin factor family.
/// - `w`: Same-spin Wick intermediates of the ordered reference pair.
/// - `pairs`: Distinct same-spin component pairs of the batch.
/// - `scratch`: Same-spin Wick scratch space.
/// # Returns:
/// - `[Vec<T>; K]`: Factors of each distinct component pair, aligned with the slots of `pairs`.
fn spin_factors<F: SpinFactor<K>, T: NOCIScalar, const K: usize>(
    factor: &F,
    w: &SameSpinView<'_, T>,
    pairs: &SpinPairRows,
    scratch: &mut WickScratch<T>,
) -> [Vec<T>; K] {
    let mut out: [Vec<T>; K] = std::array::from_fn(|_| vec![T::from_real(0.0); pairs.nslot]);
    for (target, slots, sources) in &pairs.rows {
        let row = SpinFactorRow {
            target: *target,
            sources,
            target_left: true,
            out: out.each_mut().map(|factors| &mut factors[slots.clone()]),
        };
        xw_spin_factor_row(factor, w, row, scratch);
    }
    out
}

/// Calculate the overlap matrix elements `S_{xw} = S^\alpha_{xw} S^\beta_{xw}` of a batch of
/// determinant pairs of one ordered reference pair. A single pair is evaluated directly; larger
/// batches evaluate each distinct same-spin factor once in factor rows.
/// # Arguments:
/// - `w`: Wick intermediates of the ordered parent pair of every determinant pair.
/// - `space`: Determinant space holding the parent frames.
/// - `requests`: Output positions with their determinant pairs.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `Vec<T>`: Overlap matrix element of each request, in request order.
pub(in crate::elements) fn calculate_s_pairs_wicks<T, D>(
    w: &WicksPairView<'_, T>,
    space: &NOCISpace<T>,
    requests: &[(usize, DetPair<D>)],
    scratch: &mut WickScratchSpin<T>,
) -> Vec<T>
where
    T: NOCIScalar,
    D: PairDeterminant<T>,
{
    // A single pair shares no factors with other requests, so evaluate both spins directly.
    if let [(_, pair)] = requests {
        let alpha = (pair.ldet.spin(space, true), pair.gdet.spin(space, true));
        let beta = (pair.ldet.spin(space, false), pair.gdet.spin(space, false));
        let [sa] = spin_factor(&OverlapFactor, &w.aa, alpha, &mut scratch.aa);
        let [sb] = spin_factor(&OverlapFactor, &w.bb, beta, &mut scratch.bb);
        return vec![sa * sb];
    }

    // Evaluate every distinct same-spin factor once, then combine them for each request.
    let (alpha, beta) = SpinPairRows::spins(space, requests);
    let [sa] = spin_factors(&OverlapFactor, &w.aa, &alpha, &mut scratch.aa);
    let [sb] = spin_factors(&OverlapFactor, &w.bb, &beta, &mut scratch.bb);

    (0..requests.len())
        .map(|k| sa[alpha.slots[k]] * sb[beta.slots[k]])
        .collect()
}

/// Calculate the Fock and overlap matrix elements
/// `F_{xw} = F^\alpha_{xw} S^\beta_{xw} + S^\alpha_{xw} F^\beta_{xw}` and
/// `S_{xw} = S^\alpha_{xw} S^\beta_{xw}` of a batch of determinant pairs of one ordered reference
/// pair. A single pair is evaluated directly; larger batches evaluate each distinct same-spin
/// factor once in factor rows.
/// # Arguments:
/// - `w`: Wick intermediates of the ordered parent pair of every determinant pair.
/// - `space`: Determinant space holding the parent frames.
/// - `requests`: Output positions with their determinant pairs.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `Vec<(T, T)>`: Fock and overlap matrix elements of each request, in request order.
pub(in crate::elements) fn calculate_fs_pairs_wicks<T, D>(
    w: &WicksPairView<'_, T>,
    space: &NOCISpace<T>,
    requests: &[(usize, DetPair<D>)],
    tol: f64,
    scratch: &mut WickScratchSpin<T>,
) -> Vec<(T, T)>
where
    T: NOCIScalar,
    D: PairDeterminant<T>,
{
    // A single pair shares no factors with other requests, so evaluate both spins directly.
    let factor = OneBodyFactor { tol };
    if let [(_, pair)] = requests {
        let alpha = (pair.ldet.spin(space, true), pair.gdet.spin(space, true));
        let beta = (pair.ldet.spin(space, false), pair.gdet.spin(space, false));
        let [sa, fa] = spin_factor(&factor, &w.aa, alpha, &mut scratch.aa);
        let [sb, fb] = spin_factor(&factor, &w.bb, beta, &mut scratch.bb);
        return vec![(fa * sb + fb * sa, sa * sb)];
    }

    // Evaluate every distinct same-spin factor once, then combine them for each request.
    let (alpha, beta) = SpinPairRows::spins(space, requests);
    let [sa, fa] = spin_factors(&factor, &w.aa, &alpha, &mut scratch.aa);
    let [sb, fb] = spin_factors(&factor, &w.bb, &beta, &mut scratch.bb);

    (0..requests.len())
        .map(|k| {
            let (a, b) = (alpha.slots[k], beta.slots[k]);
            (fa[a] * sb[b] + fb[b] * sa[a], sa[a] * sb[b])
        })
        .collect()
}
