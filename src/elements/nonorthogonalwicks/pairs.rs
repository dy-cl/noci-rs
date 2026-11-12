// elements/nonorthogonalwicks/pairs.rs
//! Determinant-pair overlap and Fock matrix elements from the extended nonorthogonal Wick
//! theorem.
//!
//! Both elements separate over spin sectors,
//! `S_{xw} = S^\alpha_{xw} S^\beta_{xw}` and
//! `F_{xw} = F^\alpha_{xw} S^\beta_{xw} + S^\alpha_{xw} F^\beta_{xw}`, so a batch of pairs of one
//! ordered reference pair is evaluated from its distinct same-spin factors. Each distinct pair of
//! spin strings is evaluated once, and the distinct ket spin strings of each bra spin string form
//! one same-spin factor row.
//!
//! The Hamiltonian's two-body part also contains the mixed-spin cofactor coupling
//! `\sum_{\eta z,\xi y}\operatorname{cof}(D_\alpha)_{\eta z}`
//! `\mathcal{II}_{\eta z,\xi y}\operatorname{cof}(D_\beta)_{\xi y}`. It therefore uses the fused
//! two-spin evaluator in `eval/hamiltonian.rs`, which reuses determinants, cofactors and second
//! minors across H and S, rather than the independent same-spin factor rows used here.

// Standard library imports.
use std::ops::Range;

// Crate-root imports.
use crate::determinant::{NOCISpace, ParentDeterminant, SpinStringData};
use crate::elements::DetPair;
use crate::hash::FastMap;
use crate::{NOCIScalar, StateScalar};

// Parent/sibling imports.
use super::scratch::WickScratch;
use super::view::SameSpinView;
use super::{
    OneBodyFactor, OverlapFactor, SpinFactor, SpinFactorRow, SpinStringSources, WickScratchSpin,
    WicksPairView, xw_spin_factor_row,
};

/// Distinct same-spin string pairs of a batch of determinant pairs, arranged as rows of one bra
/// spin string against its distinct ket spin strings.
struct SpinPairRows {
    /// Bra spin string of each row, the slots of its factors and its distinct ket spin strings.
    rows: Vec<(SpinStringData, Range<usize>, SpinStringSources)>,
    /// Number of distinct spin-string pairs over every row.
    nslot: usize,
    /// Slot of the spin-string pair of each request.
    slots: Vec<usize>,
}

impl SpinPairRows {
    /// Group the same-spin string pairs of a batch into rows of distinct pairs.
    /// # Arguments:
    /// - `references`: Same-spin occupations of the bra and ket parents shared by the batch.
    /// - `occupations`: Same-spin bra and ket occupations of each request.
    /// # Returns
    /// - `Self`: Rows of distinct spin-string pairs and the slot of every request.
    fn new(
        references: (u128, u128),
        occupations: &[(u128, u128)],
    ) -> Self {
        // Within one reference pair a spin string is identified by its occupation. Index the
        // distinct bra spin strings as rows and, within each row, the distinct ket spin strings,
        // preparing each spin string only the first time it is seen.
        let mut row_of: FastMap<u128, usize> = FastMap::default();
        let mut row_sources: Vec<(SpinStringData, Vec<SpinStringData>)> = Vec::new();
        let mut column_of: Vec<FastMap<u128, usize>> = Vec::new();
        let mut positions = Vec::with_capacity(occupations.len());
        let mut previous: Option<(u128, usize)> = None;

        for &(bra, ket) in occupations {
            // Consecutive requests usually share their bra, so check the previous row first.
            let row = match previous {
                Some((key, row)) if key == bra => row,
                _ => *row_of.entry(bra).or_insert_with(|| {
                    row_sources.push((SpinStringData::between(references.0, bra), Vec::new()));
                    column_of.push(FastMap::default());
                    row_sources.len() - 1
                }),
            };
            previous = Some((bra, row));

            // Intern the ket within this bra row and remember where the request will read it.
            let sources = &mut row_sources[row].1;
            let column = *column_of[row].entry(ket).or_insert_with(|| {
                sources.push(SpinStringData::between(references.1, ket));
                sources.len() - 1
            });
            positions.push((row, column));
        }

        // Lay the factor slots of the rows out contiguously, so each row writes one slice.
        let mut rows = Vec::with_capacity(row_sources.len());
        let mut nslot = 0;

        for (bra, kets) in row_sources {
            rows.push((
                bra,
                nslot..nslot + kets.len(),
                SpinStringSources::new(&kets),
            ));
            nslot += kets.len();
        }

        let slots = positions
            .iter()
            .map(|&(row, column)| rows[row].1.start + column)
            .collect();

        Self { rows, nslot, slots }
    }

    /// Group the alpha and beta spin-string pairs of a batch of one ordered reference pair.
    /// # Arguments:
    /// - `references`: Bra and ket parents shared by the batch.
    /// - `requests`: Output positions with their determinant pairs.
    /// # Returns
    /// - `(Self, Self)`: Alpha and beta rows of distinct spin-string pairs.
    fn spins<T: StateScalar>(
        references: (&ParentDeterminant<T>, &ParentDeterminant<T>),
        requests: &[(usize, DetPair)],
    ) -> (Self, Self) {
        let (x, w) = references;
        let (alpha, beta): (Vec<_>, Vec<_>) = requests
            .iter()
            .map(|(_, pair)| {
                (
                    (pair.bra.alpha, pair.ket.alpha),
                    (pair.bra.beta, pair.ket.beta),
                )
            })
            .unzip();

        (
            Self::new((x.oa, w.oa), &alpha),
            Self::new((x.ob, w.ob), &beta),
        )
    }
}

/// Evaluate the factors of one same-spin string pair, including the excitation phases.
/// # Arguments:
/// - `factor`: Same-spin factor family.
/// - `w`: Same-spin Wick intermediates of the ordered reference pair.
/// - `pair`: Bra and ket spin strings.
/// - `scratch`: Same-spin Wick scratch space.
/// # Returns:
/// - `[T; K]`: Factors of the spin-string pair.
fn spin_factor<F: SpinFactor<K>, T: NOCIScalar, const K: usize>(
    factor: &F,
    w: &SameSpinView<'_, T>,
    pair: (SpinStringData, SpinStringData),
    scratch: &mut WickScratch<T>,
) -> [T; K] {
    let (bra, ket) = pair;
    let phase = T::from_real(bra.phase * ket.phase);

    factor
        .values(w, (&bra, &ket), scratch)
        .map(|value| phase * value)
}

/// Evaluate the distinct same-spin factors of a batch, including the excitation phases, one row
/// per distinct bra spin string.
/// # Arguments:
/// - `factor`: Same-spin factor family.
/// - `w`: Same-spin Wick intermediates of the ordered reference pair.
/// - `pairs`: Distinct same-spin string pairs of the batch.
/// - `scratch`: Same-spin Wick scratch space.
/// # Returns:
/// - `[Vec<T>; K]`: Factors of each distinct spin-string pair, aligned with the slots of `pairs`.
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
pub(in crate::elements) fn calculate_s_pairs_wicks<T>(
    w: &WicksPairView<'_, T>,
    space: &NOCISpace<T>,
    requests: &[(usize, DetPair)],
    scratch: &mut WickScratchSpin<T>,
) -> Vec<T>
where
    T: NOCIScalar,
{
    // A single pair shares no factors with other requests, so evaluate both spins directly.
    if let [(_, pair)] = requests {
        let (bra, ket) = (
            pair.bra.data(space.parents()),
            pair.ket.data(space.parents()),
        );
        let (alpha, beta) = ((bra.alpha, ket.alpha), (bra.beta, ket.beta));

        let [sa] = spin_factor(&OverlapFactor, &w.aa, alpha, &mut scratch.aa);
        let [sb] = spin_factor(&OverlapFactor, &w.bb, beta, &mut scratch.bb);

        return vec![sa * sb];
    }

    // Evaluate every distinct same-spin factor once, then combine them for each request.
    let (_, first) = requests[0];
    let parents = space.parents();
    let references = (&parents[first.bra.parent], &parents[first.ket.parent]);
    let (alpha, beta) = SpinPairRows::spins(references, requests);

    let [sa] = spin_factors(&OverlapFactor, &w.aa, &alpha, &mut scratch.aa);
    let [sb] = spin_factors(&OverlapFactor, &w.bb, &beta, &mut scratch.bb);

    // Recover `S_{xw} = S^\alpha_{xw} S^\beta_{xw}` in determinant-request order.
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
pub(in crate::elements) fn calculate_fs_pairs_wicks<T>(
    w: &WicksPairView<'_, T>,
    space: &NOCISpace<T>,
    requests: &[(usize, DetPair)],
    tol: f64,
    scratch: &mut WickScratchSpin<T>,
) -> Vec<(T, T)>
where
    T: NOCIScalar,
{
    // A single pair shares no factors with other requests, so evaluate both spins directly.
    let factor = OneBodyFactor { tol };

    if let [(_, pair)] = requests {
        let (bra, ket) = (
            pair.bra.data(space.parents()),
            pair.ket.data(space.parents()),
        );
        let (alpha, beta) = ((bra.alpha, ket.alpha), (bra.beta, ket.beta));

        let [sa, fa] = spin_factor(&factor, &w.aa, alpha, &mut scratch.aa);
        let [sb, fb] = spin_factor(&factor, &w.bb, beta, &mut scratch.bb);

        return vec![(fa * sb + fb * sa, sa * sb)];
    }

    // Evaluate every distinct same-spin factor once, then combine them for each request.
    let (_, first) = requests[0];
    let parents = space.parents();
    let references = (&parents[first.bra.parent], &parents[first.ket.parent]);
    let (alpha, beta) = SpinPairRows::spins(references, requests);

    let [sa, fa] = spin_factors(&factor, &w.aa, &alpha, &mut scratch.aa);
    let [sb, fb] = spin_factors(&factor, &w.bb, &beta, &mut scratch.bb);

    // Combine `F = F^\alpha S^\beta + S^\alpha F^\beta` and `S = S^\alpha S^\beta`.
    (0..requests.len())
        .map(|k| {
            let (a, b) = (alpha.slots[k], beta.slots[k]);

            (fa[a] * sb[b] + fb[b] * sa[a], sa[a] * sb[b])
        })
        .collect()
}
