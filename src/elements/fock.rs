// elements/fock.rs

// Crate-root imports.
use crate::NOCIScalar;
use crate::elements::{DetPair, FockData, NOCIData};
use crate::time_call;

// Parent/sibling imports.
use super::naive::calculate_f_pair_naive;
use super::nonorthogonalwicks::{WickScratchSpin, calculate_f_pair_wicks};
use super::orthogonal::{calculate_f_pair_orthogonal, calculate_f_pairs_orthogonal_batched};

/// Wrapper function which dispatches Fock matrix-element evaluation for a batch of determinant
/// pairs depending on user input and properties of each pair. If a determinant pair has the same
/// Hermitian-orthonormal parent we may use the standard Slater-Condon rules, if not we can either
/// use generalised Slater-Condon rules or extended non-orthogonal Wick's theorem to evaluate the
/// matrix element. A single pair is evaluated directly; larger batches group same-parent pairs for
/// the SIMD Slater-Condon kernels.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `fock`: Fock-specific data required for Fock matrix-element evaluation.
/// - `pairs`: Pairs of determinants whose Fock matrix elements are to be evaluated.
/// - `scratch`: Scratch space for Wick's calculations.
/// - `out`: Fock matrix elements in the same order as `pairs`.
/// # Returns:
/// - `()`: Writes every requested Fock matrix element into `out`.
pub(crate) fn calculate_f_pair<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    fock: &FockData<'_, T>,
    pairs: &[DetPair],
    mut scratch: Option<&mut WickScratchSpin<T>>,
    out: &mut [T],
) {
    time_call!(crate::timers::noci::add_calculate_f_pair, {
        // A single pair has no other requests to fill SIMD lanes, so evaluate it directly.
        if let [pair] = pairs {
            out[0] = calculate_f_pair_single(data, fock, *pair, scratch);
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
            out[output] = calculate_f_pair_nonorthogonal(data, fock, pair, scratch.as_deref_mut());
        }

        calculate_f_pairs_orthogonal_batched(fock.fock_mocache, data.space, &groups, out);
    })
}

/// Evaluate one Fock matrix element through the orthogonal, Wick, or naive path.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `fock`: Fock-specific data required for Fock matrix-element evaluation.
/// - `pair`: Pair of determinants whose Fock matrix element is to be evaluated.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `T`: Fock matrix element between the determinant pair.
fn calculate_f_pair_single<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    fock: &FockData<'_, T>,
    pair: DetPair,
    scratch: Option<&mut WickScratchSpin<T>>,
) -> T {
    let lp = data.space.state(pair.ldet).parent;
    let gp = data.space.state(pair.gdet).parent;
    if lp == gp {
        let cache = &fock.fock_mocache[lp];
        if cache.orthogonal_slater_condon {
            return calculate_f_pair_orthogonal(cache, data.space, pair.ldet, pair.gdet);
        }
    }

    calculate_f_pair_nonorthogonal(data, fock, pair, scratch)
}

/// Evaluate one Fock matrix element with Wick's theorem or the generalised Slater-Condon rules.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `fock`: Fock-specific data required for Fock matrix-element evaluation.
/// - `pair`: Pair of determinants whose Fock matrix element is to be evaluated.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `T`: Fock matrix element between the determinant pair.
fn calculate_f_pair_nonorthogonal<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    fock: &FockData<'_, T>,
    pair: DetPair,
    scratch: Option<&mut WickScratchSpin<T>>,
) -> T {
    if data.input.wicks.enabled {
        calculate_f_pair_wicks(
            data.space,
            pair.ldet,
            pair.gdet,
            data.tol,
            data.wicks.unwrap(),
            scratch.unwrap(),
        )
    } else {
        calculate_f_pair_naive(
            fock.fa, fock.fb, data.ao, data.space, pair.ldet, pair.gdet, data.tol,
        )
    }
}

/// Compare naive and Wick's calculation of Fock matrix elements to ensure consistency.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `fock`: Fock-specific data required for Fock matrix-element evaluation.
/// - `pair`: Pair of determinants whose Fock matrix element is to be compared.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `(T, (f64, f64))`: Wick's Fock matrix element, total discrepancy from
///   the naive path, and max elementwise discrepancy.
pub(crate) fn compare_f_pair_wicks_naive<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    fock: &FockData<'_, T>,
    pair: DetPair,
    scratch: &mut WickScratchSpin<T>,
) -> (T, (f64, f64)) {
    let ldet = pair.ldet;
    let gdet = pair.gdet;

    let fnv = calculate_f_pair_naive(fock.fa, fock.fb, data.ao, data.space, ldet, gdet, data.tol);
    let fw = calculate_f_pair_wicks(
        data.space,
        ldet,
        gdet,
        data.tol,
        data.wicks.unwrap(),
        scratch,
    );

    let diff = (fnv - fw).abs();
    (fw, (diff, diff))
}
