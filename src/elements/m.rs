// elements/m.rs

// Crate-root imports.
use crate::NOCIScalar;
use crate::elements::{DetPair, FockData, NOCIData};

// Parent/sibling imports.
use super::naive::calculate_m_pair_naive;
use super::nonorthogonalwicks::{WickScratchSpin, calculate_m_pair_wicks};
use super::orthogonal::{calculate_m_pair_orthogonal, calculate_m_pairs_orthogonal_batched};

/// Calculate shifted candidate-candidate matrix elements `M_{ab} = F_{ab} - E0 S_{ab}` for a batch
/// of determinant pairs without evaluating `F_{ab}` and `S_{ab}` separately. A single pair is
/// evaluated directly; larger batches group same-parent pairs for the SIMD Slater-Condon kernels.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `fock`: Fock-specific data required for Fock matrix-element evaluation.
/// - `pairs`: Pairs of determinants whose shifted matrix elements are to be evaluated.
/// - `e0`: Zeroth-order energy shift.
/// - `scratch`: Scratch space for Wick's calculations.
/// - `out`: Shifted matrix elements in the same order as `pairs`.
/// # Returns:
/// - `()`: Writes every requested shifted matrix element into `out`.
pub(crate) fn calculate_m_pair<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    fock: &FockData<'_, T>,
    pairs: &[DetPair],
    e0: f64,
    mut scratch: Option<&mut WickScratchSpin<T>>,
    out: &mut [T],
) {
    // A single pair has no other requests to fill SIMD lanes, so evaluate it directly.
    if let [pair] = pairs {
        out[0] = calculate_m_pair_single(data, fock, *pair, e0, scratch);
        return;
    }

    // Group same-parent orthonormal pairs by parent and evaluate all others immediately.
    let mut groups = vec![Vec::new(); data.space.parents.len()];
    for (output, &pair) in pairs.iter().enumerate() {
        let lp = data.space.state(pair.ldet).parent;
        let gp = data.space.state(pair.gdet).parent;
        if lp == gp && fock.fock_mocache[lp].orthogonal_slater_condon {
            groups[lp].push((output, pair));
            continue;
        }
        out[output] = calculate_m_pair_nonorthogonal(data, fock, pair, e0, scratch.as_deref_mut());
    }

    calculate_m_pairs_orthogonal_batched(fock.fock_mocache, data.space, &groups, e0, out);
}

/// Evaluate one shifted matrix element through the orthogonal, Wick, or naive path.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `fock`: Fock-specific data required for Fock matrix-element evaluation.
/// - `pair`: Pair of determinants whose shifted matrix element is to be evaluated.
/// - `e0`: Zeroth-order energy shift.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `T`: Shifted matrix element `M_{ab}`.
fn calculate_m_pair_single<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    fock: &FockData<'_, T>,
    pair: DetPair,
    e0: f64,
    scratch: Option<&mut WickScratchSpin<T>>,
) -> T {
    let lp = data.space.state(pair.ldet).parent;
    let gp = data.space.state(pair.gdet).parent;
    if lp == gp {
        let cache = &fock.fock_mocache[lp];
        if cache.orthogonal_slater_condon {
            return calculate_m_pair_orthogonal(cache, data.space, pair.ldet, pair.gdet, e0);
        }
    }

    calculate_m_pair_nonorthogonal(data, fock, pair, e0, scratch)
}

/// Evaluate one shifted matrix element with Wick's theorem or the generalised Slater-Condon rules.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `fock`: Fock-specific data required for Fock matrix-element evaluation.
/// - `pair`: Pair of determinants whose shifted matrix element is to be evaluated.
/// - `e0`: Zeroth-order energy shift.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `T`: Shifted matrix element `M_{ab}`.
fn calculate_m_pair_nonorthogonal<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    fock: &FockData<'_, T>,
    pair: DetPair,
    e0: f64,
    scratch: Option<&mut WickScratchSpin<T>>,
) -> T {
    if data.input.wicks.enabled {
        calculate_m_pair_wicks(
            data.space,
            pair.ldet,
            pair.gdet,
            data.tol,
            data.wicks.unwrap(),
            e0,
            scratch.unwrap(),
        )
    } else {
        calculate_m_pair_naive(fock, data, pair.ldet, pair.gdet, e0)
    }
}
