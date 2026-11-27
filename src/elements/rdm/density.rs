// elements/rdm/density.rs
//! Spin-resolved transition overlaps and densities of determinant pairs, and the NOCI-state
//! densities and spin-squared matrices assembled from them.

// External crate imports.
use ndarray::{Array1, Array2, Zip};
use rayon::prelude::*;

// Crate-root imports.
use crate::NOCIScalar;
use crate::determinant::{Bits, DeterminantData, NOCIIndex, SpinStringData};
use crate::elements::nonorthogonalwicks::{WickScratchSpin, spin_transition_pair_wicks};
use crate::elements::{DetPair, NOCIData, build_s_pair, occ_coeffs, pair_density};

/// Spin-resolved transition quantities of a determinant pair `\langle l|\cdot|r\rangle`, with the
/// excitation phases included: the spin overlaps `S^\sigma = \langle l_\sigma|r_\sigma\rangle` and
/// the covariant AO transition densities `\tilde\Gamma^\sigma = \mathbf S W^\sigma \mathbf S`, where
/// `W^\sigma_{\mu\nu}` has the ket orbital on the row and the conjugated bra orbital on the column and
/// carries its own spin overlap. The covariant form is the one the Wick intermediates provide, and
/// the contravariant AO density of the pair is
/// `\langle l|a^\dagger a|r\rangle^\alpha = \mathbf S^{-1}\tilde\Gamma^\alpha\mathbf S^{-1} S^\beta`.
/// A density is `None` where it vanishes or was not needed.
pub(crate) struct SpinTransition<T: NOCIScalar> {
    /// Spin overlaps `(S^\alpha, S^\beta)`.
    pub(crate) s: (T, T),
    /// Covariant transition densities `(\tilde\Gamma^\alpha, \tilde\Gamma^\beta)`.
    pub(crate) gamma: (Option<Array2<T>>, Option<Array2<T>>),
}

impl<T: NOCIScalar> SpinTransition<T> {
    /// Evaluate the spin-resolved transition quantities of one determinant pair. Determinants of
    /// one parent share orthonormal orbitals, so their quantities follow the Slater-Condon rules
    /// directly; different parents use the extended Wick theorem, or the generalised Slater-Condon
    /// rules when no Wick intermediates are available. As in the Wick evaluation, a density is
    /// formed only when the other spin's overlap is non-zero, unless `cross` asks for both.
    /// # Arguments:
    /// - `data`: Shared NOCI matrix-element data.
    /// - `pair`: Bra and ket determinants.
    /// - `requests`: Every AO index pair `([\mu], [\nu])` in row-major order.
    /// - `scratch`: Scratch space for Wick's calculations.
    /// - `cross`: Whether to form each density whatever the other spin's overlap.
    /// # Returns:
    /// - `SpinTransition<T>`: Spin overlaps and covariant transition densities.
    pub(crate) fn new(
        data: &NOCIData<'_, T>,
        pair: DetPair,
        requests: &[([usize; 1], [usize; 1])],
        scratch: &mut WickScratchSpin<T>,
        cross: bool,
    ) -> Self {
        if pair.bra.parent == pair.ket.parent {
            return Self::orthogonal(data, pair, cross);
        }
        if data.wicks.is_some() {
            return spin_transition_pair_wicks(data, pair, requests, scratch, cross);
        }
        Self::naive(data, pair)
    }

    /// Generalised Slater-Condon transition quantities of two determinants from the occupied
    /// orbitals of each spin, for pairs of different parents without Wick intermediates. The
    /// contravariant pair densities are lowered with the AO overlap.
    /// # Arguments:
    /// - `data`: Shared NOCI matrix-element data.
    /// - `pair`: Bra and ket determinants.
    /// # Returns:
    /// - `SpinTransition<T>`: Spin overlaps and covariant transition densities.
    fn naive(
        data: &NOCIData<'_, T>,
        pair: DetPair,
    ) -> Self {
        let parents = data.space.parents();
        let (lp, rp) = (&parents[pair.bra.parent], &parents[pair.ket.parent]);
        let nao = data.ao.h.nrows();
        let s = data.ao.s.mapv(<T as From<f64>>::from);
        let pa = build_s_pair(
            &occ_coeffs(&lp.ca, pair.bra.alpha),
            &occ_coeffs(&rp.ca, pair.ket.alpha),
            &data.ao.s,
            data.tol,
        );
        let pb = build_s_pair(
            &occ_coeffs(&lp.cb, pair.bra.beta),
            &occ_coeffs(&rp.cb, pair.ket.beta),
            &data.ao.s,
            data.tol,
        );
        let ga = s.dot(&pair_density(&pa, nao)).dot(&s);
        let gb = s.dot(&pair_density(&pb, nao)).dot(&s);

        Self {
            s: (pa.s, pb.s),
            gamma: (Some(ga), Some(gb)),
        }
    }

    /// Slater-Condon transition quantities of two determinants of one orthonormal parent. With the
    /// lowered orbitals `\tilde C = \mathbf S C`, per spin, identical strings have `S^\sigma = 1` and
    /// `\tilde\Gamma^\sigma_{\mu\nu} = \sum_{i} \tilde C_{\mu i} \tilde C^*_{\nu i}` over the occupied
    /// orbitals; strings differing by one replacement, the ket's `q` by the bra's `p`, have
    /// `S^\sigma = 0` and `\tilde\Gamma^\sigma_{\mu\nu} = \pm \tilde C_{\mu q} \tilde C^*_{\nu p}`; any
    /// larger difference gives zero.
    /// # Arguments:
    /// - `data`: Shared NOCI matrix-element data.
    /// - `pair`: Bra and ket determinants of the same parent.
    /// - `cross`: Whether to form each density whatever the other spin's overlap.
    /// # Returns:
    /// - `SpinTransition<T>`: Spin overlaps and covariant transition densities.
    fn orthogonal(
        data: &NOCIData<'_, T>,
        pair: DetPair,
        cross: bool,
    ) -> Self {
        let parent = &data.space.parents()[pair.bra.parent];
        let s = data.ao.s.mapv(<T as From<f64>>::from);
        let zero = <T as From<f64>>::from(0.0);
        let one = <T as From<f64>>::from(1.0);

        // The bra measured from the ket: the ket's holes `q` become the bra's particles `p`.
        let connection = DeterminantData::between(
            (pair.ket.alpha, pair.ket.beta),
            (pair.bra.alpha, pair.bra.beta),
        );
        let ra = connection.alpha.excitation.holes.count_ones();
        let rb = connection.beta.excitation.holes.count_ones();
        let sa = if ra == 0 { one } else { zero };
        let sb = if rb == 0 { one } else { zero };

        // One spin's covariant density from its orbitals, ket occupation and connection to the bra.
        let density = |c: &Array2<T>, occ: Bits, string: &SpinStringData| {
            let (holes, parts) = (string.excitation.holes, string.excitation.parts);
            match holes.count_ones() {
                0 => {
                    let lowered = s.dot(&occ_coeffs(c, occ));
                    Some(lowered.dot(&lowered.mapv(|x| x.conj()).t()))
                }
                1 => {
                    let phase = <T as From<f64>>::from(string.phase);
                    let lq = s.dot(&c.column(holes.first()));
                    let lp = s.dot(&c.column(parts.first()));
                    Some(Array2::from_shape_fn((lq.len(), lp.len()), |(mu, nu)| {
                        phase * lq[mu] * lp[nu].conj()
                    }))
                }
                _ => None,
            }
        };

        let ga = (cross || rb == 0)
            .then(|| density(&parent.ca, pair.ket.alpha, &connection.alpha))
            .flatten();
        let gb = (cross || ra == 0)
            .then(|| density(&parent.cb, pair.ket.beta, &connection.beta))
            .flatten();

        Self {
            s: (sa, sb),
            gamma: (ga, gb),
        }
    }
}

/// Every AO index pair `([\mu], [\nu])` in row-major order, the rank-one Wick density requests.
/// # Arguments:
/// - `nao`: Number of AO basis functions.
/// # Returns:
/// - `Vec<([usize; 1], [usize; 1])>`: Index pairs.
fn ao_requests(nao: usize) -> Vec<([usize; 1], [usize; 1])> {
    (0..nao)
        .flat_map(|p| (0..nao).map(move |q| ([p], [q])))
        .collect()
}

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
    let requests = ao_requests(nao);

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
/// from its spin-resolved transition quantities, so pairs with a zero spin overlap stay finite. For `l = r` this reduces to the single-determinant formula used
/// for SCF states.
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
    let requests = ao_requests(nao);
    let sinv = data.ao.x.dot(&data.ao.x).mapv(<T as From<f64>>::from);
    let zero = <T as From<f64>>::from(0.0);

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
