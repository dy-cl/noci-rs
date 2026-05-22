// scf/h/select.rs

use crate::input::StateRecipe;
use crate::{HSCFState, SCFState};

use super::distance::h_density_distance_from_real;

/// Select the h-SCF candidate consistent with the recipe's branch preference.
/// # Arguments:
/// - `recipe`: Holomorphic recipe used to generate the candidates.
/// - `seed`: Real SCF seed used for fresh h-SCF recovery.
/// - `d_tol`: Density tolerance used to identify collapse onto the real seed.
/// - `candidates`: Converged h-SCF candidates.
/// # Returns:
/// - `Option<HSCFState>`: Selected h-SCF state, or `None` if no valid branch was found.
pub(crate) fn select_hscf_candidate(
    recipe: &StateRecipe,
    seed: &SCFState,
    d_tol: f64,
    candidates: Vec<HSCFState>,
) -> Option<HSCFState> {
    if candidates.is_empty() {
        println!("No converged h-SCF candidate for '{}'.", recipe.label);
        return None;
    }

    if recipe.partner.is_some() {
        let best = candidates
            .iter()
            .enumerate()
            .map(|(i, st)| (i, h_density_distance_from_real(st, seed)))
            .filter(|(_, d)| *d > d_tol)
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());

        if let Some((i, d)) = best {
            println!(
                "Accepted recovered h-SCF branch '{}' with d: {:.3e}.",
                recipe.label, d
            );
            return candidates.into_iter().nth(i);
        }

        println!(
            "No recovered h-SCF branch '{}' survived collapse test.",
            recipe.label
        );
        return None;
    }

    if recipe.spin_bias.is_some() {
        candidates
            .into_iter()
            .min_by(|a, b| a.e.re.partial_cmp(&b.e.re).unwrap())
    } else if recipe.spatial_bias.is_some() {
        candidates
            .into_iter()
            .max_by(|a, b| a.e.re.partial_cmp(&b.e.re).unwrap())
    } else {
        candidates.into_iter().next()
    }
}
