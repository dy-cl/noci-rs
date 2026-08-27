// basis/bias.rs

// External crate imports.
use ndarray::Array2;

// Crate-root imports.
use crate::input::StateRecipe;
use crate::maths::general_evp_x;
use crate::scf::{DensityMode, density, fock};
use crate::{AoData, SCFState};

// Parent/sibling imports.
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
            d[(i, j)] *= scale;
        }
    }
}

/// Bias density matrices towards a spatial symmetry-broken RHF guess.
/// # Arguments:
/// - `da`: Alpha-spin density matrix.
/// - `db`: Beta-spin density matrix.
/// - `atomao`: Global AO indices belonging to each atom.
/// - `pol`: Bias strength.
/// - `pattern`: Spatial biasing pattern.
/// # Returns:
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

/// Bias density matrices towards a spin symmetry-broken UHF guess.
/// # Arguments:
/// - `da`: Alpha-spin density matrix.
/// - `db`: Beta-spin density matrix.
/// - `atomao`: Global AO indices belonging to each atom.
/// - `pol`: Bias strength.
/// - `pattern`: Spin biasing pattern.
/// # Returns:
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

        let idx = &atomao[a];

        if sgn > 0 {
            scale_block(da, idx, up);
            scale_block(db, idx, dn);
        } else {
            scale_block(da, idx, dn);
            scale_block(db, idx, up);
        }
    }
}

/// Build molecular alpha and beta density guesses for a state recipe.
/// The atomic SAD is biased first, then used for one molecular Fock diagonalisation.
/// Continuation seeds are reused directly without rebuilding the initial guess.
/// # Arguments:
/// - `ao`: AO data containing the SAD density, integrals, and AO labels.
/// - `recipe`: Recipe whose spin or spatial bias should be applied.
/// - `seed`: Optional previous or partner SCF state used directly as the density guess.
/// # Returns:
/// - `(Array2<f64>, Array2<f64>)`: Molecular alpha and beta density guesses.
pub(crate) fn biased_density_guess(
    ao: &AoData,
    recipe: &StateRecipe,
    seed: Option<&SCFState>,
) -> (Array2<f64>, Array2<f64>) {
    if let Some(state) = seed {
        return ((*state.da).clone(), (*state.db).clone());
    }

    let mut da = ao.dm.clone() * 0.5;
    let mut db = ao.dm.clone() * 0.5;

    if recipe.spin_bias.is_some() || recipe.spatial_bias.is_some() {
        let atomao = atomao_for_labels(&ao.labels);

        if let Some(spin_bias) = &recipe.spin_bias {
            bias_spin(&mut da, &mut db, &atomao, spin_bias.pol, &spin_bias.pattern);
        }

        if let Some(spatial_bias) = &recipe.spatial_bias {
            bias_spatial(
                &mut da,
                &mut db,
                &atomao,
                spatial_bias.pol,
                &spatial_bias.pattern,
            );
        }
    }

    let (fa, fb) = fock(&ao.h, &ao.eri_coul, &da, &db);
    let (_, ca) = general_evp_x(&fa, &ao.x);
    let (_, cb) = general_evp_x(&fb, &ao.x);

    let da = density(&ca, ao.nelec[0] as usize, DensityMode::Hermitian);
    let db = density(&cb, ao.nelec[1] as usize, DensityMode::Hermitian);

    (da, db)
}
