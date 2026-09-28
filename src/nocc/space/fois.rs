// nocc/space/fois.rs
//! Weighted, canonically orthogonalised first-order interacting space.

// External crate imports.
use ndarray::{Array1, Array2};

// Crate-root imports.
use crate::input::{FoisWeighting, NOCCMCOptions};
use crate::maths::linalg::{block_loewdin_x, symmetric_blocks};
use crate::nocc::equations::metric_matrix;
use crate::nocc::setup::ReferenceState;
use crate::nocc::terms::TermEvaluator;

// Parent/sibling imports.
use super::excitations::Excitation;
use super::orbitals::Spaces;

/// Raw metric and orthogonalised FOIS basis used by the amplitude solver.
pub(crate) struct FoisBasis {
    /// Raw spin-free FOIS metric S.
    pub metric: Array2<f64>,
    /// Canonical FOIS transformation `Y = w\tilde X`.
    pub y: Array2<f64>,
}

/// Build the weighted FOIS basis from the full raw excitation list.
/// The weights `w_\mu` are either the Hamiltonian couplings `h_\mu` or, for coupled weighting,
/// unit weights on excitations with `|h_\mu|` above the coupling threshold and zero otherwise.
/// Both exclude spectator excitations of separated fragments, whose `h_\mu` vanish exactly.
/// # Arguments:
/// - `reference`: Normal-ordered reference state.
/// - `spaces`: Core, active, and virtual orbital-space maps.
/// - `excitations`: Raw spin-free excitation list.
/// - `evaluator`: Term-table evaluator.
/// - `options`: FOIS weighting, coupling threshold and weighted-metric eigenvalue threshold.
/// # Returns:
/// - `FoisBasis`: Raw metric and the orthogonalised FOIS basis `Y`.
/// # References
/// - Lee and Tew, arXiv:2507.13472 (2025), Eqs. (38)-(49).
pub(crate) fn build_fois_basis(
    reference: &ReferenceState<'_>,
    spaces: &Spaces,
    excitations: &[Excitation],
    evaluator: &TermEvaluator,
    options: &NOCCMCOptions,
) -> FoisBasis {
    // Raw FOIS metric `S_{\mu\nu} = \langle E_\mu^\dagger E_\nu\rangle` from its class-pair blocks.
    let s = metric_matrix(reference, spaces, excitations, evaluator);

    // Form the weighted metric `\tilde S = \operatorname{diag}(w) S \operatorname{diag}(w)`.
    let h = hamiltonian_weights(reference, spaces, excitations);
    let w = match options.fois_weighting {
        FoisWeighting::Coupled => {
            let tol = options.fois_coupling_tol;
            h.mapv(|x| if x.abs() > tol { 1.0 } else { 0.0 })
        }
        FoisWeighting::Hamiltonian => h.clone(),
    };
    let mut stilde: Array2<f64> = Array2::zeros(s.raw_dim());

    for i in 0..s.nrows() {
        for j in 0..s.ncols() {
            stilde[(i, j)] = w[i] * s[(i, j)] * w[j];
        }
    }

    // The metric is block diagonal through its Kronecker deltas, and the weights only remove
    // rows, so both are orthogonalised block by block.
    let blocks = symmetric_blocks(&s);
    let weighted_blocks = blocks
        .iter()
        .map(|b| {
            b.iter()
                .copied()
                .filter(|&mu| w[mu] != 0.0)
                .collect::<Vec<_>>()
        })
        .filter(|b| !b.is_empty())
        .collect::<Vec<_>>();

    // Löwdin orthogonalisation removes small weighted-metric eigenmodes;
    // `Y = \operatorname{diag}(w) \tilde X` maps orthogonal columns to the raw FOIS basis.
    let xtilde = block_loewdin_x(&stilde, &weighted_blocks, options.fois_tol);
    let mut y = xtilde.clone();

    for mu in 0..w.len() {
        for col in 0..y.ncols() {
            y[(mu, col)] *= w[mu];
        }
    }

    FoisBasis { metric: s, y }
}

/// Build Hamiltonian coupling weights used for the weighted FOIS metric.
/// # Arguments:
/// - `reference`: Normal-ordered reference state, whose generalised Fock matrix is the
///   spin-resolved Fock matrix of the equally split density.
/// - `spaces`: NOCC orbital spaces.
/// - `excitations`: Raw spin-free excitation list.
/// # Returns:
/// - `Array1<f64>`: One Hamiltonian weight per excitation.
fn hamiltonian_weights(
    reference: &ReferenceState<'_>,
    spaces: &Spaces,
    excitations: &[Excitation],
) -> Array1<f64> {
    let (f, eri) = (&reference.fock, &reference.ao.eri_coul);
    let mut h = Array1::zeros(excitations.len());

    // Weights are the coefficients of `\hat H = \sum_\mu h_\mu \hat\tau_\mu`: singles use `F_{qp}`
    // and doubles `(pr|qs)`, halved when the pair swap `\hat E^{qp}_{sr}` is also in the list.
    for (i, &ex) in excitations.iter().enumerate() {
        h[i] = match ex {
            Excitation::Single { p, q } => f[(q, p)],
            Excitation::Double { p, q, r, s } => {
                let same = spaces.class_of[p] == spaces.class_of[q]
                    && spaces.class_of[r] == spaces.class_of[s];
                let w = if same { 0.5 } else { 1.0 };
                w * eri[(p, r, q, s)]
            }
        };
    }

    h
}

/// Project an amplitude change onto the FOIS, `P x = Y Y^\dagger S x`, keeping the amplitudes
/// consistent with `t = Y\tilde t`.
/// # Arguments:
/// - `fois`: FOIS basis data.
/// - `x`: Vector in the raw excitation basis.
/// # Returns:
/// - `Array1<f64>`: Projected vector.
/// # References
/// - Lee and Tew, arXiv:2507.13472 (2025), Eqs. (63)-(64).
pub(in crate::nocc) fn project_onto_fois(
    fois: &FoisBasis,
    x: &Array1<f64>,
) -> Array1<f64> {
    fois.y.dot(&fois.y.t().dot(&fois.metric.dot(x)))
}
