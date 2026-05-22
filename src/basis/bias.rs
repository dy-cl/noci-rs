// basis/bias.rs

use ndarray::Array2;

use crate::input::StateRecipe;
use crate::{AoData, SCFState};

use super::atoms::atomao_for_labels;

/// Multiply a square sub-block of a matrix by a scalar in place.
/// # Arguments:
/// - `d`: Matrix to be modified in place.
/// - `idx`: Row and column indices defining the square sub-block.
/// - `scale`: Multiplicative factor applied to the selected sub-block.
/// # Returns:
/// - `()`: Modifies `d` in place.
fn scale_block(
    d: &mut Array2<f64>,
    idx: &[usize],
    scale: f64,
) {
    for &i in idx {
        for &j in idx {
            d[(i, j)] *= scale
        }
    }
}

/// Bias density matrices towards a spatial symmetry broken RHF guess. We will have da = db.
/// # Arguments
/// - `da`: Spin density matrix a.
/// - `db`: Spin density matrix b.
/// - `atomao`: Global AO indices of AOs belonging to atom i.
/// - `pol`: Bias strength.
/// - `pattern`: Spatial biasing pattern.
/// # Returns
/// - `()`: Modifies `da` and `db` in place.
pub(crate) fn bias_spatial(
    da: &mut Array2<f64>,
    db: &mut Array2<f64>,
    atomao: &[Vec<usize>],
    pol: f64,
    pattern: &[i8],
) {
    let up = 1.0 + pol;
    let dn = 1.0 - pol;

    for (a, &sgn) in pattern.iter().enumerate() {
        if sgn == 0 {
            continue;
        }
        let idx = &atomao[a];
        if sgn > 0 {
            scale_block(da, idx, up);
            scale_block(db, idx, up);
        } else {
            scale_block(da, idx, dn);
            scale_block(db, idx, dn);
        }
    }
}

/// Bias density matrices towards a spin symmetry-broken UHF guess. We will have da != db.
/// # Arguments
/// - `da`: Spin density matrix a.
/// - `db`: Spin density matrix b.
/// - `atomao`: Global AO indices of AOs belonging to atom i.
/// - `pol`: Bias strength.
/// - `pattern`: Spin biasing pattern.
/// # Returns
/// - `()`: Modifies `da` and `db` in place.
pub(crate) fn bias_spin(
    da: &mut Array2<f64>,
    db: &mut Array2<f64>,
    atomao: &[Vec<usize>],
    pol: f64,
    pattern: &[i8],
) {
    let up = 1.0 + pol;
    let dn = 1.0 - pol;

    for (a, &sgn) in pattern.iter().enumerate() {
        if sgn == 0 {
            continue;
        }
        let i = &atomao[a];
        if sgn > 0 {
            scale_block(da, i, up);
            scale_block(db, i, dn);
        } else {
            scale_block(da, i, dn);
            scale_block(db, i, up);
        }
    }
}

/// Build the real alpha and beta density guess for a state recipe.
/// Biases are applied only when no continuation seed is available; otherwise the previous converged density is reused directly so the branch can collapse naturally.
/// # Arguments
/// - `ao`: AO data containing the RHF density and AO labels.
/// - `recipe`: Recipe whose spin/spatial bias should be applied.
/// - `seed`: Optional previous or partner SCF state used as the base density.
/// # Returns
/// - `(Array2<f64>, Array2<f64>)`: Biased alpha and beta density guesses.
pub(crate) fn biased_density_guess(
    ao: &AoData,
    recipe: &StateRecipe,
    seed: Option<&SCFState>,
) -> (Array2<f64>, Array2<f64>) {
    let mut da = seed
        .map(|st| (*st.da).clone())
        .unwrap_or_else(|| ao.dm.clone() * 0.5);
    let mut db = seed
        .map(|st| (*st.db).clone())
        .unwrap_or_else(|| ao.dm.clone() * 0.5);

    if seed.is_none() && (recipe.spin_bias.is_some() || recipe.spatial_bias.is_some()) {
        let atomao = atomao_for_labels(&ao.labels);

        if let Some(sb) = &recipe.spin_bias {
            bias_spin(&mut da, &mut db, &atomao, sb.pol, &sb.pattern);
        }
        if let Some(spb) = &recipe.spatial_bias {
            bias_spatial(&mut da, &mut db, &atomao, spb.pol, &spb.pattern);
        }
    }

    (da, db)
}
