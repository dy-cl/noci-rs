// elements/rdm/density.rs
//! NOCI-state densities and spin-squared matrices assembled from the spin-resolved transition
//! quantities of their determinant pairs.

// External crate imports.
use ndarray::{Array1, Array2, Zip};
use rayon::prelude::*;

// Crate-root imports.
use crate::NOCIScalar;
use crate::determinant::NOCIIndex;
use crate::elements::nonorthogonalwicks::WickScratchSpin;
use crate::elements::{DetPair, NOCIData};

// Parent/sibling imports.
use super::SpinTransition;

/// Calculate the alpha and beta density matrices of a multireference NOCI state,
/// `D^\alpha = \mathbf S^{-1} \big(\sum_{ij} c_i^* c_j \tilde\Gamma^\alpha_{ij} S^\beta_{ij}\big)
/// \mathbf S^{-1}` and likewise `D^\beta` with `S^\alpha_{ij} \tilde\Gamma^\beta_{ij}`, raising the
/// summed covariant densities once at the end.
/// # Arguments:
/// - `data`: Shared NOCI matrix-element data.
/// - `states`: Determinant basis of the NOCI wavefunction.
/// - `c`: Coefficients of the NOCI wavefunction.
/// # Returns:
/// - `(Array2<T>, Array2<T>)`: Alpha and beta AO density matrices.
pub(crate) fn noci_density<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    states: &[NOCIIndex],
    c: &Array1<T>,
) -> (Array2<T>, Array2<T>) {
    let nao = data.ao.h.nrows();

    // Every AO index pair `([\mu], [\nu])` in row-major order, the rank-one Wick density requests.
    let requests: Vec<_> = (0..nao)
        .flat_map(|p| (0..nao).map(move |q| ([p], [q])))
        .collect();

    // Partition bra states across threads and reduce their local covariant densities.
    let (da, db) = (0..states.len())
        .into_par_iter()
        .map_init(WickScratchSpin::new, |scratch, i| {
            let mut da = Array2::<T>::zeros((nao, nao));
            let mut db = Array2::<T>::zeros((nao, nao));

            for (j, &ket) in states.iter().enumerate() {
                let pair = DetPair::from_space(data.space, states[i], ket);
                let t = SpinTransition::new(data, pair, &requests, scratch, false);
                let cij = c[i].conj() * c[j];

                if let Some(ga) = &t.gamma.0 {
                    da.scaled_add(cij * t.s.1, ga);
                }
                if let Some(gb) = &t.gamma.1 {
                    db.scaled_add(cij * t.s.0, gb);
                }
            }

            (da, db)
        })
        .reduce(
            || {
                (
                    Array2::<T>::zeros((nao, nao)),
                    Array2::<T>::zeros((nao, nao)),
                )
            },
            |(mut da_a, mut db_a), (da_b, db_b)| {
                da_a += &da_b;
                db_a += &db_b;

                (da_a, db_a)
            },
        );

    // `\mathbf S^{-1} = X X` from the symmetric orthogonaliser `X = \mathbf S^{-1/2}`.
    let sinv = data.ao.x.dot(&data.ao.x).mapv(<T as From<f64>>::from);

    (sinv.dot(&da).dot(&sinv), sinv.dot(&db).dot(&sinv))
}

/// Build the spin-squared matrix `\langle l|S^2|r\rangle` between two determinant lists, from which
/// a NOCI state `|\Psi\rangle = \sum_i c_i |i\rangle` has
/// `\langle S^2\rangle = \sum_{ij} c_i^* c_j \langle i|S^2|j\rangle / \sum_{ij} c_i^* c_j S_{ij}`.
/// With the raised densities `G^\sigma = \mathbf S^{-1}\tilde\Gamma^\sigma`, each determinant pair
/// contributes
/// `\langle l|S^2|r\rangle = S_z(S_z + 1) S^\alpha S^\beta + S^\alpha \mathrm{Tr}\,G^\beta - \mathrm{Tr}(G^\alpha G^\beta)`
/// from its spin-resolved transition quantities, so pairs with a zero spin overlap stay finite.
/// For `l = r` this reduces to the single-determinant formula used for SCF states.
/// # Arguments:
/// - `data`: Shared NOCI matrix-element data.
/// - `left`: Bra determinants, the rows.
/// - `right`: Ket determinants, the columns.
/// # Returns:
/// - `Array2<T>`: Spin-squared matrix `\langle l|S^2|r\rangle`.
pub(crate) fn build_noci_spin_square<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    left: &[NOCIIndex],
    right: &[NOCIIndex],
) -> Array2<T> {
    let nao = data.ao.h.nrows();
    let zero = <T as From<f64>>::from(0.0);

    // Every AO index pair `([\mu], [\nu])` in row-major order, the rank-one Wick density requests.
    let requests: Vec<_> = (0..nao)
        .flat_map(|p| (0..nao).map(move |q| ([p], [q])))
        .collect();

    // `\mathbf S^{-1} = X X` raises the covariant densities.
    let sinv = data.ao.x.dot(&data.ao.x).mapv(<T as From<f64>>::from);

    // Each row `l` of `\langle l|S^2|r\rangle` is evaluated independently.
    let rows: Vec<Vec<T>> = left
        .par_iter()
        .map_init(WickScratchSpin::new, |scratch, &bra| {
            let (oa, ob) = data.space.occupations(bra);
            let sz = 0.5 * (oa.count_ones() as f64 - ob.count_ones() as f64);

            right
                .iter()
                .map(|&ket| {
                    let pair = DetPair::from_space(data.space, bra, ket);
                    let t = SpinTransition::new(data, pair, &requests, scratch, true);
                    let (sa, sb) = t.s;

                    // `S_z(S_z + 1) S^\alpha S^\beta + S^\alpha \mathrm{Tr}\,G^\beta
                    //  - \mathrm{Tr}(G^\alpha G^\beta)`, with `\mathrm{Tr}(AB) = \sum_{\mu\nu}
                    //  A_{\mu\nu} B_{\nu\mu}`.
                    let mut value = <T as From<f64>>::from(sz * (sz + 1.0)) * sa * sb;
                    if let Some(gb) = &t.gamma.1 {
                        let gb = sinv.dot(gb);
                        value += sa * gb.diag().sum();

                        if let Some(ga) = &t.gamma.0 {
                            let ga = sinv.dot(ga);
                            value -= Zip::from(&ga)
                                .and(&gb.t())
                                .fold(zero, |acc, &a, &b| acc + a * b);
                        }
                    }

                    if value.abs() == 0.0 { zero } else { value }
                })
                .collect()
        })
        .collect();

    Array2::from_shape_fn((left.len(), right.len()), |(l, r)| rows[l][r])
}
