// elements/nonorthogonalwicks/transition.rs
//! Spin-resolved transition overlaps and one-body transition densities of a determinant pair from
//! the extended nonorthogonal Wick theorem.

// External crate imports.
use ndarray::Array2;

// Crate-root imports.
use crate::NOCIScalar;
use crate::elements::{DetPair, NOCIData, SpinTransition};

// Parent/sibling imports.
use super::{WickScratchSpin, xw_overlap_prepared, xw_rdmk_same_prepared_batched};

/// Evaluate the spin overlaps `S^\sigma = \langle l_\sigma|r_\sigma\rangle` and the covariant AO
/// transition densities `\tilde\Gamma^\sigma_{\mu\nu}` of a determinant pair from different parents,
/// with the excitation phases included; the external-basis contractions are `\mathbf S M \mathbf S`.
/// The evaluators sum the near-zero branches of each spin sector and drop the distributions that
/// cannot survive, so no excitation-rank screening is applied here. A density enters the NOCI
/// density only through `\tilde\Gamma^\alpha S^\beta` or `S^\alpha \tilde\Gamma^\beta`, so it is
/// evaluated only when the other spin's overlap is non-zero, unless `cross` asks for both, as the
/// `\mathrm{Tr}(G^\alpha G^\beta)` term of `S^2` does.
/// # Arguments:
/// - `data`: Shared NOCI matrix-element data, with Wick intermediates.
/// - `pair`: Bra and ket determinants, of different parents.
/// - `requests`: Every AO index pair `([\mu], [\nu])` in row-major order.
/// - `scratch`: Scratch space for Wick's calculations.
/// - `cross`: Whether to evaluate each density whatever the other spin's overlap.
/// # Returns:
/// - `SpinTransition<T>`: Spin overlaps and the covariant transition densities that were evaluated.
pub(in crate::elements) fn spin_transition_pair_wicks<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair,
    requests: &[([usize; 1], [usize; 1])],
    scratch: &mut WickScratchSpin<T>,
    cross: bool,
) -> SpinTransition<T> {
    // Resolve the determinant pair, AO dimension, and ordered parent-pair Wick data.
    let parents = data.space.parents();
    let (lp, gp) = (&parents[pair.bra.parent], &parents[pair.ket.parent]);
    let (ldet, gdet) = (pair.bra.data(parents), pair.ket.data(parents));
    let n = data.ao.h.nrows();
    let w = data.wicks.unwrap().pair(pair.bra.parent, pair.ket.parent);
    let zero = <T as From<f64>>::from(0.0);

    // Excitations and phases of each spin sector.
    let (ex_la, ex_ga) = (&ldet.alpha.excitation, &gdet.alpha.excitation);
    let (ex_lb, ex_gb) = (&ldet.beta.excitation, &gdet.beta.excitation);
    let pha = <T as From<f64>>::from(ldet.alpha.phase * gdet.alpha.phase);
    let phb = <T as From<f64>>::from(ldet.beta.phase * gdet.beta.phase);

    // Spin overlaps `S^\sigma`, phased.
    let sa = pha * xw_overlap_prepared(&w.aa, ex_la, ex_ga, &mut scratch.aa);
    let sb = phb * xw_overlap_prepared(&w.bb, ex_lb, ex_gb, &mut scratch.bb);

    // Rank-one transition densities `\tilde\Gamma^\sigma_{\mu\nu}`, phased, where they are needed.
    let ga = (cross || sb.abs() > data.tol).then(|| {
        let mut g = vec![zero; requests.len()];
        xw_rdmk_same_prepared_batched::<T, 1>(
            &w.aa,
            (ex_la, ex_ga),
            (lp.ca.as_ref(), gp.ca.as_ref()),
            requests,
            &mut scratch.aa,
            data.tol,
            &mut g,
        );
        Array2::from_shape_fn((n, n), |(p, q)| pha * g[p * n + q])
    });
    let gb = (cross || sa.abs() > data.tol).then(|| {
        let mut g = vec![zero; requests.len()];
        xw_rdmk_same_prepared_batched::<T, 1>(
            &w.bb,
            (ex_lb, ex_gb),
            (lp.cb.as_ref(), gp.cb.as_ref()),
            requests,
            &mut scratch.bb,
            data.tol,
            &mut g,
        );
        Array2::from_shape_fn((n, n), |(p, q)| phb * g[p * n + q])
    });

    SpinTransition {
        s: (sa, sb),
        gamma: (ga, gb),
    }
}
