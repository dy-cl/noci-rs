// noci/optimised/fock.rs
//! Resonating Fock matrices of ResHF, whose occupied-virtual blocks are the orbital gradient of
//! the state-averaged NOCI energy: `\partial E_{\text{SA}} / \partial \kappa^A_{ai} =
//! 2 (C^{A\dagger}_u F^{A,\omega} C^A_o)_{ai}`.

// External crate imports.
use ndarray::Array2;

// Parent/sibling imports.
use super::ci::ResHFExpansion;
use super::pair::ResHFPair;

/// Off-diagonal resonating contribution of the ordered pair `(A, B)` for one spin,
/// `\bar K^{AB,\omega} = D_{AB} s_{AB} (1 - S Q^{AB,\omega}) F^{AB,\omega} Q^{AB,\omega} S
/// + (D_{AB} H_{AB} - D^E_{AB} s_{AB}) S Q^{AB,\omega} S`. The first term is the derivative of
/// the pair energy `H_{AB} / s_{AB}` and the second that of the pair overlap.
/// # Arguments:
/// - `s_ao`: AO overlap matrix `S`.
/// - `q`: Transition density `Q^{AB,\omega}`.
/// - `f`: Transition Fock matrix `F^{AB,\omega}`.
/// - `c_f`: Coefficient `D_{AB} s_{AB}` of the pair-energy term.
/// - `c_s`: Coefficient `D_{AB} H_{AB} - D^E_{AB} s_{AB}` of the pair-overlap term.
/// # Returns
/// - `Array2<f64>`: `\bar K^{AB,\omega}`.
fn resonating_term(
    s_ao: &Array2<f64>,
    q: &Array2<f64>,
    f: &Array2<f64>,
    c_f: f64,
    c_s: f64,
) -> Array2<f64> {
    // `F Q S - S Q F Q S = (1 - S Q) F Q S`.
    let sq = s_ao.dot(q);
    let fqs = f.dot(&q.dot(s_ao));
    let projected = &fqs - &sq.dot(&fqs);

    projected * c_f + sq.dot(s_ao) * c_s
}

/// Resonating Fock matrices of every determinant, eqs. (15a), (15b) and (16) of Miller and
/// Parker with the state sum taken inside the pair weights `D_{AB}` and `D^E_{AB}`:
/// `F^{A,\omega} = D_{AA} F^{AA,\omega} + \sum_{B \ne A} (\bar K^{AB,\omega} + (\bar K^{AB,\omega})^T)`.
/// `H_{AB}` and `E_I` may both include nuclear repulsion, since `H_{AB} - s_{AB} E_I` does not
/// change. Each `F^{A,\omega}` is symmetric.
/// # Arguments:
/// - `s_ao`: AO overlap matrix `S`.
/// - `pairs`: Cutoff-floored pair quantities of the determinants.
/// - `expansion`: NOCI expansion of the same determinants, solved on the pairs' `H_{AB}` and
///   `s_{AB}`.
/// # Returns
/// - `Vec<[Array2<f64>; 2]>`: `F^{A,\alpha}` and `F^{A,\beta}` for each determinant `A`.
pub(super) fn reshf_fock(
    s_ao: &Array2<f64>,
    pairs: &Array2<ResHFPair>,
    expansion: &ResHFExpansion,
) -> Vec<[Array2<f64>; 2]> {
    let n = pairs.nrows();
    let d = &expansion.pair_weights;
    let de = &expansion.pair_energy_weights;

    (0..n)
        .map(|a| {
            // Diagonal term `D_{AA} F^{AA,\omega}`, the Fock matrix of `A`'s own density.
            let aa = &pairs[(a, a)];
            let mut fock = [&aa.f[0] * d[(a, a)], &aa.f[1] * d[(a, a)]];

            // Symmetrised resonating terms of every other determinant.
            for b in (0..n).filter(|&b| b != a) {
                let ab = &pairs[(a, b)];
                let c_f = d[(a, b)] * ab.s;
                let c_s = d[(a, b)] * ab.h - de[(a, b)] * ab.s;
                for (w, f) in fock.iter_mut().enumerate() {
                    let k = resonating_term(s_ao, &ab.q[w], &ab.f[w], c_f, c_s);
                    *f += &k;
                    *f += &k.t();
                }
            }

            fock
        })
        .collect()
}
