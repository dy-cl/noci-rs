// elements/overlap.rs

// Crate-root imports.
use crate::NOCIScalar;
use crate::elements::{DetPair, NOCIData};
use crate::time_call;

// Parent/sibling imports.
use super::naive::calculate_s_pair_naive;
use super::nonorthogonalwicks::{WickScratchSpin, calculate_s_pair_wicks};
use super::orthogonal::{calculate_s_pair_orthogonal, calculate_s_pairs_orthogonal_batched};

/// Wrapper function which dispatches overlap matrix-element evaluation for a batch of determinant
/// pairs depending on user input and properties of each pair.
/// If a determinant pair has the same Hermitian-orthonormal parent we may use the standard
/// Slater-Condon rules, if not we can either use generalised Slater-Condon rules or extended
/// non-orthogonal Wick's theorem to evaluate the matrix element. A single pair is evaluated
/// directly; larger batches group same-parent pairs for the SIMD Slater-Condon kernels.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pairs`: Pairs of determinants whose overlap matrix elements are to be evaluated.
/// - `scratch`: Scratch space for Wick's calculations.
/// - `out`: Overlap matrix elements in the same order as `pairs`.
/// # Returns:
/// - `()`: Writes every requested overlap into `out`.
pub(crate) fn calculate_s_pair<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pairs: &[DetPair],
    mut scratch: Option<&mut WickScratchSpin<T>>,
    out: &mut [T],
) {
    time_call!(crate::timers::noci::add_calculate_s_pair, {
        // A single pair has no other requests to fill SIMD lanes, so evaluate it directly.
        if let [pair] = pairs {
            out[0] = calculate_s_pair_single(data, *pair, scratch);
            return;
        }

        // Group same-parent orthonormal pairs by parent and evaluate all others immediately.
        let mut groups = vec![Vec::new(); data.space.parents.len()];
        for (output, &pair) in pairs.iter().enumerate() {
            let lp = data.space.state(pair.ldet).parent;
            let gp = data.space.state(pair.gdet).parent;
            if lp == gp {
                let mocache = data
                    .mocache
                    .expect("Orthogonal overlap matrix elements require mocache.");
                if mocache[lp].orthogonal_slater_condon {
                    groups[lp].push((output, pair));
                    continue;
                }
            }
            out[output] = calculate_s_pair_nonorthogonal(data, pair, scratch.as_deref_mut());
        }

        calculate_s_pairs_orthogonal_batched(data.space, &groups, out);
    })
}

/// Evaluate one overlap matrix element through the orthogonal, Wick, or naive path.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose overlap matrix element is to be evaluated.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `T`: Overlap matrix element between the determinant pair.
fn calculate_s_pair_single<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair,
    scratch: Option<&mut WickScratchSpin<T>>,
) -> T {
    let lp = data.space.state(pair.ldet).parent;
    let gp = data.space.state(pair.gdet).parent;

    if lp == gp {
        let mocache = data
            .mocache
            .expect("Orthogonal overlap matrix elements require mocache.");
        if mocache[lp].orthogonal_slater_condon {
            return calculate_s_pair_orthogonal(data.space, pair.ldet, pair.gdet);
        }
    }

    calculate_s_pair_nonorthogonal(data, pair, scratch)
}

/// Evaluate one overlap matrix element with Wick's theorem or the generalised Slater-Condon rules.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `pair`: Pair of determinants whose overlap matrix element is to be evaluated.
/// - `scratch`: Scratch space for Wick's calculations.
/// # Returns:
/// - `T`: Overlap matrix element between the determinant pair.
fn calculate_s_pair_nonorthogonal<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    pair: DetPair,
    scratch: Option<&mut WickScratchSpin<T>>,
) -> T {
    if data.input.wicks.enabled {
        calculate_s_pair_wicks(
            data.space,
            pair.ldet,
            pair.gdet,
            data.wicks.unwrap(),
            scratch.unwrap(),
        )
    } else {
        calculate_s_pair_naive(data, pair.ldet, pair.gdet)
    }
}
