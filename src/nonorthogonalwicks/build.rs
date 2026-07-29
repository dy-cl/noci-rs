// nonorthogonalwicks/build.rs
use ndarray::{Array1, Array2, Array4, Axis, s};
use ndarray_linalg::{Determinant, SVD};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::input::Spin;
use crate::{AoData, DetState};

use crate::maths::{ERIAO2MOScratch, adjoint, real2_as};
use crate::noci::{NOCIScalar, occ_coeffs};

/// Symmetry-unique distributions of zero-overlap orbital pairs over the four contractions
/// in the same-spin \mathcal J intermediate. The symmetry
/// {}^{\chi_\eta\chi_z,\chi_\xi\chi_y}\mathcal J_{\eta z,\xi y}^{(m_1,m_2,m_3,m_4)}
/// = {}^{\chi_\xi\chi_y,\chi_\eta\chi_z}\mathcal J_{\xi y,\eta z}^{(m_3,m_4,m_1,m_2)}
/// reduces the 16 possible combinations with m_i \in \{0,1\} to the ten combinations stored here.
pub(crate) const SAME_SPIN_J_BRANCHES: [(usize, usize, usize, usize); 10] = [
    (0, 0, 0, 0),
    (0, 0, 0, 1),
    (0, 0, 1, 0),
    (0, 0, 1, 1),
    (0, 1, 0, 1),
    (0, 1, 1, 0),
    (0, 1, 1, 1),
    (1, 0, 1, 0),
    (1, 0, 1, 1),
    (1, 1, 1, 1),
];

/// Precomputed same-spin intermediates for a pair of non-orthogonal reference determinants.
#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(bound = "T: NOCIScalar")]
pub struct SameSpinBuild<T: NOCIScalar> {
    /// Fundamental contraction matrices X^{(m_i)} for m_i = 0 and m_i = 1.
    pub x: [Array2<T>; 2],
    /// Fundamental contraction matrices Y^{(m_i)} for m_i = 0 and m_i = 1.
    pub y: [Array2<T>; 2],
    /// Scalar Fock one-body intermediates {}^x F_0^{(m_i)}.
    pub f0f: [T; 2],
    /// Scalar Hamiltonian one-body intermediates {}^x F_0^{(m_i)}.
    pub f0h: [T; 2],
    /// Hamiltonian one-body intermediates {}^{\chi_r\chi_z}\mathcal F_{rz}^{(m_i,m_j)}.
    pub fh: [[Array2<T>; 2]; 2],
    /// Fock one-body intermediates {}^{\chi_r\chi_z}\mathcal F_{rz}^{(m_i,m_j)}.
    pub ff: [[Array2<T>; 2]; 2],
    /// Scalar same-spin two-body intermediates {}^x V_0^{(m_i,m_j)}.
    pub v0: [T; 3],
    /// Same-spin one-column intermediates {}^{\chi_\eta\chi_z}\mathcal V_{\eta z}^{(m_i,m_j,m_k)}.
    pub v: [[[Array2<T>; 2]; 2]; 2],
    /// Sparse symmetry-reduced same-spin two-column intermediates. Each entry contains the SAME_SPIN_J_BRANCHES
    /// slot and the corresponding {}^{\chi_\eta\chi_z,\chi_\xi\chi_y}\mathcal J tensor, with unreachable branches omitted.
    pub j: Vec<(usize, Array4<T>)>,
    /// Product of the non-zero singular values forming the magnitude of the reduced overlap.
    pub tilde_s_prod: f64,
    /// Phase of the reduced overlap arising from the occupied orbital rotations.
    pub phase: T,
    /// Number m of zero-overlap orbital pairs.
    pub m: usize,
    /// Number of molecular orbitals in one reference orbital set.
    pub nmo: usize,
}

impl<T: NOCIScalar> SameSpinBuild<T> {
    /// Construct the same-spin intermediates required to evaluate matrix elements between arbitrary excited determinants.
    /// For the reference determinant pair \langle{}^x\Psi| and |{}^w\Psi\rangle, the occupied orbital overlap matrix is
    /// singular-value decomposed as:
    /// {}^{xw}\mathbf S_{\mathrm{occ}} = \mathbf U{}^{xw}\tilde{\mathbf S}_{\mathrm{occ}}\mathbf V^\dagger.
    /// The occupied orbitals are rotated into the Löwdin paired basis, allowing the number m of zero-overlap orbital pairs
    /// and the reduced overlap to be identified. The reduced overlap is:
    /// {}^{xw}\tilde S = \phi^{xw}\prod_{\{i\mid{}^{xw}\tilde S_i \neq 0\}}{}^{xw}\tilde S_i.
    /// The fundamental contractions X^{(m_i)} and Y^{(m_i)} are then constructed for m_i = 0 and m_i = 1. These are used to
    /// form the one-body intermediates F, the same-spin one-column intermediates \mathcal V, and the same-spin two-column
    /// intermediates \mathcal J. Once these intermediates have been constructed, matrix elements are independent of the number of
    /// electrons and basis functions, although their cost continues to depend on the excitation rank and the number of distributions
    /// satisfying \sum_i m_i = m.
    /// # Arguments:
    /// - `ao`: AO overlap matrix and one- and two-electron integrals.
    /// - `g`: Ket reference determinant |{}^w\Psi\rangle.
    /// - `l`: Reference determinant forming the bra \langle{}^x\Psi|.
    /// - `spin`: Spin block for which the intermediates are constructed.
    /// - `tol`: Singular values satisfying |{}^{xw}\tilde S_i| \leq \mathtt{tol} are treated as zero.
    /// # Returns
    /// - `SameSpinBuild<T>`: Same-spin intermediates for the reference determinant pair.
    pub fn new(
        ao: &AoData,
        g: &DetState<T>,
        l: &DetState<T>,
        spin: Spin,
        tol: f64,
    ) -> Self {
        let eri = &ao.eri_coul;
        let h_munu = &ao.h;
        let s_munu = &ao.s;

        // Extract the selected spin orbital coefficients and numbers of occupied orbitals for
        // the bra reference x and ket reference w.
        let (g_c, go, l_c, lo) = match spin {
            Spin::Alpha => (g.ca.as_ref(), g.oa, l.ca.as_ref(), l.oa),
            Spin::Beta => (g.cb.as_ref(), g.ob, l.cb.as_ref(), l.ob),
            Spin::Both => panic!("SameSpinBuild requires either alpha or beta spin, not both."),
        };

        let nmo = g_c.ncols();
        let nbas = l_c.nrows();
        let z = <T as From<f64>>::from(0.0);

        // Construct the orbital spaces used for the rows and columns of the excitation
        // contraction determinant. The row space is V_x \cup O_w, corresponding to the
        // annihilation operators, while the column space is O_x \cup V_w, corresponding
        // to the creation operators.
        let nocc = lo.count_ones() as usize;
        let (rowc, colc) = contraction_orbitals(l_c, g_c, nocc);

        // Extract the occupied orbitals required to construct the reference-pair
        // occupied orbital overlap matrix.
        let l_c_occ = occ_coeffs(l_c, lo);
        let g_c_occ = occ_coeffs(g_c, go);

        // Singular-value decompose the occupied orbital overlap matrix and rotate the
        // occupied orbitals into the Löwdin paired basis.
        let (tilde_s_occ, g_tilde_c_occ, l_tilde_c_occ, phase) =
            Self::perform_ortho_and_svd_and_rotate(s_munu, &l_c_occ, &g_c_occ);

        // Form the reduced overlap by multiplying only the non-zero
        // singular values. The phase arising from the occupied orbital rotations is
        // stored separately.
        let tilde_s_prod = tilde_s_occ
            .iter()
            .filter(|&&x| x.abs() > tol)
            .product::<f64>();

        // Identify the m zero-overlap orbital pairs. When evaluating a matrix element,
        // each pair must be assigned to one of the fundamental contractions such that
        // the complete distribution satisfies \sum_i m_i = m.
        let zeros: Vec<usize> = tilde_s_occ
            .iter()
            .enumerate()
            .filter_map(|(k, &sk)| if sk.abs() <= tol { Some(k) } else { None })
            .collect();
        let m = zeros.len();

        // Construct the AO matrices used for the m_i = 0 and m_i = 1 fundamental contractions.
        // A fundamental contraction contains either zero or one
        // zero-overlap orbital pair, so no higher branch is required.
        let (m0, m1) = Self::construct_m(&tilde_s_occ, &l_tilde_c_occ, &g_tilde_c_occ, &zeros, tol);
        let mao: [Array2<T>; 2] = [m0, m1];

        // Transform the AO matrices into the X^{(m_i)} and Y^{(m_i)} fundamental contractions.
        let (x0, y0, x0rdm, y0rdm) = Self::construct_xy(&rowc, &colc, s_munu, &mao[0], true);
        let (x1, y1, x1rdm, y1rdm) = Self::construct_xy(&rowc, &colc, s_munu, &mao[1], false);
        let x = [x0, x1];
        let y = [y0, y1];

        #[cfg(feature = "nocc")]
        let xrdm = [x0rdm, x1rdm];
        #[cfg(feature = "nocc")]
        let yrdm = [y0rdm, y1rdm];
        #[cfg(not(feature = "nocc"))]
        drop((x0rdm, x1rdm, y0rdm, y1rdm));

        // Construct the left and right partial transformations required by the
        // \mathcal F, \mathcal V and \mathcal J intermediates. These contain the
        // appropriate X or Y form for each x- or w-reference orbital block, avoiding
        // branching over the operator origin during the later transformations.
        let mut cx: [Array2<T>; 2] = [
            Array2::<T>::zeros((nbas, nmo)),
            Array2::<T>::zeros((nbas, nmo)),
        ];
        let mut xc: [Array2<T>; 2] = [
            Array2::<T>::zeros((nbas, nmo)),
            Array2::<T>::zeros((nbas, nmo)),
        ];
        for mi in 0..2 {
            (cx[mi], xc[mi]) = DiffSpinBuild::build_cx_xc(&mao[mi], s_munu, l_c, g_c, lo, mi);
        }

        // Contract the AO two-electron integrals with {}^{xw}M^{(m_i)} to form the Coulomb-minus-exchange matrices
        // J_{st}^{(m_i)} - K_{st}^{(m_i)} = \sum_{\mu\nu} [(st|\mu\nu) - (s\mu|\nu t)] {}^{xw}M_{\mu\nu}^{(m_i)}.
        // These matrices are reused in the scalar and one-column two-body intermediates.
        let nbas = mao[0].nrows();
        let mut jkao: [Array2<T>; 2] = [
            Array2::<T>::zeros((nbas, nbas)),
            Array2::<T>::zeros((nbas, nbas)),
        ];
        for mi in 0..2 {
            let j = Self::build_j_coulomb(eri, &mao[mi]);
            let k = Self::build_k_exchange(eri, &mao[mi]);
            jkao[mi] = &j - &k;
        }

        // Construct the scalar and one-column intermediates for the one-electron
        // Hamiltonian. These correspond to:
        // {}^x F_0^{(m_i)} = \sum_{pq} {}^x h_{pq} {}^{xx}X_{qp}^{(m_i)} and
        // {}^{\chi_r\chi_z}\mathcal F_{rz}^{(m_i,m_j)}[\mathcal A,\mathcal B]
        // = \sum_{pq} {}^{\chi_r x}\mathcal A_{rp}^{(m_i)} {}^x h_{pq} {}^{x\chi_z}\mathcal B_{qz}^{(m_j)}.
        let h = real2_as::<T>(h_munu);
        let f00h = adjoint(&cx[0]).dot(&h).dot(&xc[0]);
        let f01h = adjoint(&cx[0]).dot(&h).dot(&xc[1]);
        let f10h = adjoint(&cx[1]).dot(&h).dot(&xc[0]);
        let f11h = adjoint(&cx[1]).dot(&h).dot(&xc[1]);

        let f0_0h = T::einsum_ba_ab_realop(&mao[0], h_munu);
        let f0_1h = T::einsum_ba_ab_realop(&mao[1], h_munu);

        let f0h: [T; 2] = [f0_0h, f0_1h];
        let fh: [[Array2<T>; 2]; 2] = [[f00h, f01h], [f10h, f11h]];

        // Initialise the generalised-Fock {}^x F_0^{(m_i)} and {}^{\chi_r\chi_z}\mathcal F_{rz}^{(m_i,m_j)}
        // intermediates to zero. Unlike the core-Hamiltonian intermediates, these cannot be constructed
        // when the reference-pair Wick intermediates are first built because the generalised Fock matrices
        // depend on the density of the current selected NOCI wavefunction. During each SNOCI iteration,
        // `update_wicks_fock` reconstructs these intermediates from the current alpha- and beta-spin
        // Fock matrices and overwrites these initial values.
        let f0f: [T; 2] = [z, z];
        let ff: [[Array2<T>; 2]; 2] = [
            [Array2::zeros((nmo, nmo)), Array2::zeros((nmo, nmo))],
            [Array2::zeros((nmo, nmo)), Array2::zeros((nmo, nmo))],
        ];

        // Construct the scalar same-spin two-body intermediate
        // {}^x V_0^{(m_i,m_j)} = \sum_{pqrs} {}^x v_{pqrs}
        // ({}^{xx}X_{rp}^{(m_i)}{}^{xx}X_{sq}^{(m_j)} - {}^{xx}X_{rq}^{(m_i)} {}^{xx}X_{sp}^{(m_j)}).
        // The three stored entries correspond to total assignments m_i + m_j = 0, 1 and 2.
        // The m_i + m_j = 1 entry contains both symmetry-equivalent assignments (0,1) and (1,0).
        let mut v0 = [<T as From<f64>>::from(0.0); 3];
        v0[0] = T::einsum_ba_ab(&jkao[0], &mao[0]);
        if m >= 1 {
            // The assignments (0,1) and (1,0) are equal and are therefore combined.
            v0[1] = <T as From<f64>>::from(2.0) * T::einsum_ba_ab(&jkao[0], &mao[1]);
        }
        if m >= 2 {
            // Both operator contractions contain one zero-overlap orbital pair.
            v0[2] = T::einsum_ba_ab(&jkao[1], &mao[1]);
        }

        // Construct the same-spin one-column intermediates
        // {}^{\chi_\eta\chi_z}\mathcal V_{\eta z}^{(m_1,m_2,m_3)}[\mathcal B,\mathcal A]
        // = \sum_{ps} {}^{\chi_\eta x}\mathcal B_{\eta p}^{(m_1)}
        // ({}^x J_{ps}^{(m_2)} - {}^x K_{ps}^{(m_2)}) {}^{x\chi_z}\mathcal A_{sz}^{(m_3)}.
        // The array is stored as v[m_1][m_3][m_2], with the left and right
        // fundamental-contraction assignments preceding the Coulomb-minus-exchange assignment.
        let mut v: [[[Array2<T>; 2]; 2]; 2] = std::array::from_fn(|_| {
            std::array::from_fn(|_| std::array::from_fn(|_| Array2::<T>::zeros((nmo, nmo))))
        });
        let combos: Vec<(usize, usize, usize)> = (0..2)
            .flat_map(|mi| (0..2).flat_map(move |mj| (0..2).map(move |mk| (mi, mj, mk))))
            .collect();
        let blocks: Vec<((usize, usize, usize), Array2<T>)> = combos
            .into_par_iter()
            .map(|(mi, mj, mk)| {
                let blk = adjoint(&cx[mi]).dot(&jkao[mk]).dot(&xc[mj]);
                ((mi, mj, mk), blk)
            })
            .collect();
        for ((mi, mj, mk), blk) in blocks {
            v[mi][mj][mk] = blk;
        }

        // Construct the same-spin two-column intermediates
        // {}^{\chi_\eta\chi_z,\chi_\xi\chi_y}\mathcal J_{\eta z,\xi y}^{(m_1,m_2,m_3,m_4)}
        // [\mathcal C,\mathcal A,\mathcal D,\mathcal B]. The direct and exchange AO integral
        // transformations are evaluated separately and then antisymmetrised. Only the ten symmetry-unique
        // distributions in SAME_SPIN_J_BRANCHES are stored, and branches requiring more than m assigned
        // zero-overlap orbital pairs are omitted.
        let combos: Vec<(usize, (usize, usize, usize, usize))> = SAME_SPIN_J_BRANCHES
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, branch)| branch.0 + branch.1 + branch.2 + branch.3 <= m)
            .collect();
        let j: Vec<(usize, Array4<T>)> = combos
            .into_par_iter()
            .map_init(
                || -> ERIAO2MOScratch<T> { T::new_eri_ao2mo_scratch(eri, nmo, nmo, nmo, nmo) },
                |scratch, (slot, (mi, mj, mk, ml))| {
                    let mut blk = Array4::<T>::zeros((nmo, nmo, nmo, nmo));
                    let mut exch = Array4::<T>::zeros((nmo, nmo, nmo, nmo));
                    // Transform the direct two-electron contribution.
                    T::eri_ao2mo_hermitian_into(
                        eri,
                        &cx[mi],
                        &xc[mj],
                        &cx[mk],
                        &xc[ml],
                        blk.view_mut(),
                        scratch,
                    );
                    // Transform the exchange contribution with the second pair of excitation indices interchanged.
                    T::eri_ao2mo_hermitian_into(
                        eri,
                        &cx[mi],
                        &cx[mk],
                        &xc[mj],
                        &xc[ml],
                        exch.view_mut(),
                        scratch,
                    );
                    // Antisymmetrise the transformed integrals to give the same-spin direct-minus-exchange intermediate.
                    for p in 0..nmo {
                        for q in 0..nmo {
                            for r in 0..nmo {
                                for s in 0..nmo {
                                    blk[(p, q, r, s)] -= exch[(p, r, q, s)];
                                }
                            }
                        }
                    }
                    (slot, blk)
                },
            )
            .collect();

        Self {
            x,
            y,
            #[cfg(feature = "nocc")]
            xrdm,
            #[cfg(feature = "nocc")]
            yrdm,
            f0h,
            fh,
            f0f,
            ff,
            v0,
            v,
            j,
            tilde_s_prod,
            phase,
            m,
            nmo,
        }
    }

    /// Count the zero-overlap occupied orbital pairs without constructing the Löwdin paired orbitals.
    /// The singular values of {}^{xw}\mathbf S_{\mathrm{occ}} = ({}^x\mathbf C_{\mathrm{occ}})^\dagger
    /// \mathbf S {}^w\mathbf C_{\mathrm{occ}} are sufficient to determine m. The singular vectors and rotated
    /// occupied coefficients are therefore not constructed. This is used before allocating the intermediate
    /// slab so that storage is reserved only for reachable \mathcal J and \mathcal{II} branches.
    /// # Arguments:
    /// - `s_munu`: AO overlap matrix.
    /// - `g_c`: Ket-reference MO coefficient matrix {}^w\mathbf C.
    /// - `go`: Ket-reference occupation bitstring.
    /// - `l_c`: Bra-reference MO coefficient matrix {}^x\mathbf C.
    /// - `lo`: Bra-reference occupation bitstring.
    /// - `tol`: Singular values satisfying |{}^{xw}\tilde S_i| \leq \mathtt{tol} are treated as zero.
    /// # Returns
    /// - `usize`: Number m of zero-overlap orbital pairs.
    pub(crate) fn count_zero_overlap_pairs(
        s_munu: &Array2<f64>,
        g_c: &Array2<T>,
        go: u128,
        l_c: &Array2<T>,
        lo: u128,
        tol: f64,
    ) -> usize {
        let l_c_occ = occ_coeffs(l_c, lo);
        let g_c_occ = occ_coeffs(g_c, go);
        let xw_s = Self::occupied_overlap(s_munu, &l_c_occ, &g_c_occ);
        let (_, s, _) = xw_s.svd(false, false).unwrap();
        s.iter().filter(|&&sk| sk.abs() <= tol).count()
    }

    /// Construct the occupied orbital overlap matrix between the reference determinants.
    /// The matrix is: {}^{xw}\mathbf S_{\mathrm{occ}} = ({}^x\mathbf C_{\mathrm{occ}})^\dagger
    /// \mathbf S {}^w\mathbf C_{\mathrm{occ}}.
    /// # Arguments:
    /// - `s_munu`: AO overlap matrix \mathbf S.
    /// - `l_c_occ`: Occupied coefficient matrix {}^x\mathbf C_{\mathrm{occ}} of the bra reference.
    /// - `g_c_occ`: Occupied coefficient matrix {}^w\mathbf C_{\mathrm{occ}} of the ket reference.
    /// # Returns
    /// - `Array2<T>`: Occupied orbital overlap matrix {}^{xw}\mathbf S_{\mathrm{occ}}.
    fn occupied_overlap(
        s_munu: &Array2<f64>,
        l_c_occ: &Array2<T>,
        g_c_occ: &Array2<T>,
    ) -> Array2<T> {
        let s = real2_as::<T>(s_munu);
        adjoint(l_c_occ).dot(&s).dot(g_c_occ)
    }

    /// Singular-value decompose the occupied orbital overlap matrix and construct the Löwdin paired occupied orbitals.
    /// The occupied orbital overlap matrix is decomposed as:
    /// {}^{xw}\mathbf S_{\mathrm{occ}} = \mathbf U {}^{xw}\tilde{\mathbf S}_{\mathrm{occ}} \mathbf V^\dagger.
    /// The paired occupied coefficient matrices are:
    /// {}^x\tilde{\mathbf C}_{\mathrm{occ}} = {}^x\mathbf C_{\mathrm{occ}}\mathbf U,
    /// {}^w\tilde{\mathbf C}_{\mathrm{occ}} = {}^w\mathbf C_{\mathrm{occ}}\mathbf V.
    /// These satisfy:
    /// ({}^x\tilde{\mathbf C}_{\mathrm{occ}})^\dagger \mathbf S {}^w\tilde{\mathbf C}_{\mathrm{occ}}
    /// = {}^{xw}\tilde{\mathbf S}_{\mathrm{occ}}.
    /// The phase introduced into the determinant overlap by the occupied orbital rotations is:
    /// \phi^{xw} = \det(\mathbf U)\det(\mathbf V)^*.
    /// # Arguments:
    /// - `s_munu`: AO overlap matrix \mathbf S.
    /// - `l_c_occ`: Occupied coefficient matrix {}^x\mathbf C_{\mathrm{occ}} of the bra reference.
    /// - `g_c_occ`: Occupied coefficient matrix {}^w\mathbf C_{\mathrm{occ}} of the ket reference.
    /// # Returns
    /// - `(Array1<f64>, Array2<T>, Array2<T>, T)`: Singular values, paired occupied coefficients for w, paired occupied
    ///   coefficients for x, and the phase \phi^{xw}.
    pub fn perform_ortho_and_svd_and_rotate(
        s_munu: &Array2<f64>,
        l_c_occ: &Array2<T>,
        g_c_occ: &Array2<T>,
    ) -> (Array1<f64>, Array2<T>, Array2<T>, T) {
        // Construct {}^{xw}\mathbf S_{\mathrm{occ}} in the original occupied orbital bases.
        let xw_s = Self::occupied_overlap(s_munu, l_c_occ, g_c_occ);
        // Singular-value decompose the occupied orbital overlap matrix.
        let (u, xw_tilde_s, v_dag) = xw_s.svd(true, true).unwrap();
        let u = u.unwrap();
        let v = adjoint(&v_dag.unwrap());

        // Rotate both occupied spaces into the Löwdin paired basis.
        let l_tilde_c = l_c_occ.dot(&u);
        let g_tilde_c = g_c_occ.dot(&v);

        // Calculate the phase required to recover the original determinant overlap
        // from the product of the paired-orbital singular values.
        let det_u = u.det().unwrap();
        let det_v = v.det().unwrap();
        let ph = det_u * det_v.conj();

        (xw_tilde_s, g_tilde_c, l_tilde_c, ph)
    }

    /// Construct the AO fundamental contractions {}^{xw}M^{(0)} and {}^{xw}M^{(1)}.
    /// The contribution from the non-zero singular values is:
    /// {}^{xw}W^{\mu\nu} = \sum_{\{i\mid{}^{xw}\tilde S_i \neq 0\}} {}^w\tilde c_{\cdot i}^{\mu\cdot}
    /// \frac{1}{{}^{xw}\tilde S_i} {}^x\tilde c_{\cdot i}^{*\,\nu\cdot}.
    /// The contribution from the zero-overlap orbital pairs is:
    /// {}^{xw}P^{\mu\nu} = \sum_{\{k\mid{}^{xw}\tilde S_k = 0\}} {}^w\tilde c_{\cdot k}^{\mu\cdot} {}^x\tilde c_{\cdot k}^{*\,\nu\cdot}.
    /// The corresponding same-reference contribution formed from the ket orbitals is:
    /// {}^{ww}P^{\mu\nu} = \sum_{\{k\mid{}^{xw}\tilde S_k = 0\}} {}^w\tilde c_{\cdot k}^{\mu\cdot} {}^w\tilde c_{\cdot k}^{*\,\nu\cdot}.
    /// The two stored fundamental contraction matrices are:
    /// {}^{xw}M^{\mu\nu,(0)} = {}^{xw}W^{\mu\nu} + {}^{xw}P^{\mu\nu} + {}^{ww}P^{\mu\nu},
    /// {}^{xw}M^{\mu\nu,(1)} = {}^{xw}P^{\mu\nu}.
    /// A fundamental contraction containing more than one zero-overlap orbital pair vanishes, and therefore
    /// no M^{(m_i)} with m_i > 1 is required.
    /// # Arguments:
    /// - `xw_tilde_s`: Singular values {}^{xw}\tilde S_i of the occupied orbital overlap matrix.
    /// - `l_tilde_c_occ`: Paired occupied coefficients {}^x\tilde{\mathbf C}_{\mathrm{occ}}.
    /// - `g_tilde_c_occ`: Paired occupied coefficients {}^w\tilde{\mathbf C}_{\mathrm{occ}}.
    /// - `zeros`: Indices k for which |{}^{xw}\tilde S_k| \leq \mathtt{tol}.
    /// - `tol`: Singular-value tolerance.
    /// # Returns
    /// - `(Array2<T>, Array2<T>)`: AO fundamental contraction matrices {}^{xw}M^{(0)} and {}^{xw}M^{(1)}.
    pub fn construct_m(
        xw_tilde_s: &Array1<f64>,
        l_tilde_c_occ: &Array2<T>,
        g_tilde_c_occ: &Array2<T>,
        zeros: &Vec<usize>,
        tol: f64,
    ) -> (Array2<T>, Array2<T>) {
        let nbas = g_tilde_c_occ.nrows();
        let nocc = g_tilde_c_occ.ncols();

        // Divide each ket occupied orbital by its paired non-zero singular value.
        // Columns belonging to zero-overlap orbital pairs are set to zero, since they
        // do not contribute to {}^{xw}W.
        let mut g_tilde_c_occ_scaled = g_tilde_c_occ.clone();
        for k in 0..nocc {
            let s = xw_tilde_s[k];
            if s.abs() > tol {
                let scale = <T as From<f64>>::from(1.0 / s);
                let mut col = g_tilde_c_occ_scaled.column_mut(k);
                col.mapv_inplace(|z| z * scale);
            } else {
                g_tilde_c_occ_scaled
                    .column_mut(k)
                    .fill(<T as From<f64>>::from(0.0));
            }
        }

        // Construct the inverse-weighted contribution {}^{xw}W.
        let mut xw_m0 = g_tilde_c_occ_scaled.dot(&adjoint(l_tilde_c_occ));
        let mut xw_m1 = Array2::<T>::zeros((nbas, nbas));

        // Construct the same-reference {}^{ww}P contribution from the zero-overlap ket orbitals.
        let mut gg_m0 = Array2::<T>::zeros((nbas, nbas));
        for &k in zeros {
            for mu in 0..nbas {
                for nu in 0..nbas {
                    gg_m0[(mu, nu)] += g_tilde_c_occ[(mu, k)] * g_tilde_c_occ[(nu, k)].conj();
                }
            }
        }

        // Add {}^{ww}P to the M^{(0)} matrix.
        xw_m0 += &gg_m0;

        // Construct {}^{xw}P. This gives M^{(1)} directly and also contributes to M^{(0)}.
        for &k in zeros {
            for mu in 0..nbas {
                for nu in 0..nbas {
                    let p = g_tilde_c_occ[(mu, k)] * l_tilde_c_occ[(nu, k)].conj();
                    xw_m1[(mu, nu)] += p;
                    xw_m0[(mu, nu)] += p;
                }
            }
        }

        (xw_m0, xw_m1)
    }

    /// Construct the MO fundamental contractions X^{(m_i)} and Y^{(m_i)} from the AO fundamental contraction M^{(m_i)}.
    /// The AO fundamental contractions are:
    /// X_{\mathrm{AO}}^{(m_i)} = \mathbf S M^{(m_i)}\mathbf S,
    /// Y_{\mathrm{AO}}^{(0)} = X_{\mathrm{AO}}^{(0)} - \mathbf S,
    /// Y_{\mathrm{AO}}^{(1)} = X_{\mathrm{AO}}^{(1)}.
    /// These are transformed into the orbital spaces required by the contraction determinant as:
    /// X^{(m_i)} = \mathbf C_{\mathrm{row}}^\dagger X_{\mathrm{AO}}^{(m_i)}\mathbf C_{\mathrm{col}},
    /// Y^{(m_i)} = \mathbf C_{\mathrm{row}}^\dagger Y_{\mathrm{AO}}^{(m_i)}\mathbf C_{\mathrm{col}}.
    /// The row orbitals are drawn from the x-reference virtual orbitals and w-reference occupied orbitals, while the
    /// column orbitals are drawn from the x-reference occupied orbitals and w-reference virtual orbitals. The returned
    /// matrices therefore contain the {}^{xx}X, {}^{xx}Y, {}^{xw}Y, {}^{wx}X and {}^{ww}X contractions required by
    /// the overlap contraction determinant.
    /// # Arguments:
    /// - `rowc`: Coefficients of the orbitals associated with the contraction determinant rows.
    /// - `colc`: Coefficients of the orbitals associated with the contraction determinant columns.
    /// - `s_munu`: AO overlap matrix \mathbf S.
    /// - `gl_m`: AO fundamental contraction {}^{xw}M^{(m_i)}.
    /// - `subtract`: Whether to construct Y^{(0)} by subtracting the AO overlap matrix.
    /// # Returns
    /// - `(Array2<T>, Array2<T>, Array2<T>, Array2<T>)`: MO fundamental contractions X^{(m_i)} and Y^{(m_i)},
    ///   followed by their AO forms X_{\mathrm{AO}}^{(m_i)} and Y_{\mathrm{AO}}^{(m_i)}.
    fn construct_xy(
        rowc: &Array2<T>,
        colc: &Array2<T>,
        s_munu: &Array2<f64>,
        gl_m: &Array2<T>,
        subtract: bool,
    ) -> (Array2<T>, Array2<T>, Array2<T>, Array2<T>) {
        let s = real2_as::<T>(s_munu);

        let xrdm = s.dot(gl_m).dot(&s);
        let yrdm = if subtract { &xrdm - &s } else { xrdm.clone() };

        let x = adjoint(rowc).dot(&xrdm).dot(colc);
        let y = adjoint(rowc).dot(&yrdm).dot(colc);

        (x, y, xrdm, yrdm)
    }

    /// Construct the scalar and column intermediates for a one-body operator.
    /// The scalar intermediate is:
    /// {}^x F_0^{(m_1)} = \sum_{pq} {}^x f_{pq}{}^{xx}X_{qp}^{(m_1)}.
    /// The column intermediates are:
    /// {}^{\chi_r\chi_z}\mathcal F_{rz}^{(m_i,m_1)}[\mathcal A,\mathcal B]
    /// = \sum_{pq}{}^{\chi_r x}\mathcal A_{rp}^{(m_i)}{}^x f_{pq}
    /// {}^{x\chi_z}\mathcal B_{qz}^{(m_1)},
    /// where \mathcal A,\mathcal B \in \{X,Y\} and \chi_r,\chi_z \in \{x,w\}.
    /// The appropriate X or Y contraction is selected by the reference determinant associated with the corresponding
    /// excitation operator.
    /// # Arguments:
    /// - `s_munu`: AO overlap matrix \mathbf S.
    /// - `f_munu`: AO representation of the one-body operator.
    /// - `g`: Ket reference determinant |{}^w\Psi\rangle.
    /// - `l`: Bra reference determinant \langle{}^x\Psi|.
    /// - `spin`: Spin block for which the intermediates are constructed.
    /// - `tol`: Singular values satisfying |{}^{xw}\tilde S_i| \leq \mathtt{tol} are treated as zero.
    /// # Returns
    /// - `([T; 2], [[Array2<T>; 2]; 2])`: Scalar intermediates F_0^{(m_i)} and column intermediates
    ///   \mathcal F^{(m_i,m_j)} for m_i,m_j \in \{0,1\}.
    pub fn construct_f_scalar(
        s_munu: &Array2<f64>,
        f_munu: &Array2<T>,
        g: &DetState<T>,
        l: &DetState<T>,
        spin: Spin,
        tol: f64,
    ) -> ([T; 2], [[Array2<T>; 2]; 2]) {
        // Extract the selected spin orbital coefficients and occupied orbitals for
        // the bra reference x and ket reference w.
        let (g_c, go, l_c, lo) = match spin {
            Spin::Alpha => (g.ca.as_ref(), g.oa, l.ca.as_ref(), l.oa),
            Spin::Beta => (g.cb.as_ref(), g.ob, l.cb.as_ref(), l.ob),
            Spin::Both => panic!("SameSpinBuild requires either alpha or beta spin, not both."),
        };

        let l_c_occ = occ_coeffs(l_c, lo);
        let g_c_occ = occ_coeffs(g_c, go);

        // Construct the Löwdin paired occupied orbitals and identify the
        // zero-overlap orbital pairs.
        let (tilde_s_occ, g_tilde_c_occ, l_tilde_c_occ, _) =
            Self::perform_ortho_and_svd_and_rotate(s_munu, &l_c_occ, &g_c_occ);
        let zeros: Vec<usize> = tilde_s_occ
            .iter()
            .enumerate()
            .filter_map(|(k, &sk)| if sk.abs() <= tol { Some(k) } else { None })
            .collect();

        // Construct the AO fundamental contractions M^{(0)} and M^{(1)}.
        let (m0, m1) = Self::construct_m(&tilde_s_occ, &l_tilde_c_occ, &g_tilde_c_occ, &zeros, tol);
        let mao = [&m0, &m1];

        // Construct the matrices containing the X^{(m_i)} or Y^{(m_i)}
        // contractions required on either side of the one-body operator.
        let (cx0, xc0) = DiffSpinBuild::build_cx_xc(mao[0], s_munu, l_c, g_c, lo, 0);
        let (cx1, xc1) = DiffSpinBuild::build_cx_xc(mao[1], s_munu, l_c, g_c, lo, 1);
        let cx = [&cx0, &cx1];
        let xc = [&xc0, &xc1];

        // Contract the one-body operator with M^{(m_i)} to form the scalar
        // intermediates F_0^{(m_i)}.
        let f0 = [
            T::einsum_ba_ab(mao[0], f_munu),
            T::einsum_ba_ab(mao[1], f_munu),
        ];
        // Form the column intermediates \mathcal F^{(m_i,m_j)} for every combination
        // m_i,m_j \in \{0,1\}. The allowed distributions are selected during evaluation.
        let ff = std::array::from_fn(|mi| {
            std::array::from_fn(|mj| adjoint(cx[mi]).dot(f_munu).dot(xc[mj]))
        });

        (f0, ff)
    }

    /// Construct the Coulomb contraction required by the same-spin and different-spin two-body intermediates.
    /// The AO contraction is:
    /// J_{st}^{(m_i)} = \sum_{\mu\nu}(st|\mu\nu){}^{xw}M_{\mu\nu}^{(m_i)}.
    /// # Arguments:
    /// - `eri`: Non-antisymmetrised AO two-electron integrals.
    /// - `m`: AO fundamental contraction {}^{xw}M^{(m_i)}.
    /// # Returns
    /// - `Array2<T>`: Coulomb contraction J^{(m_i)}.
    fn build_j_coulomb(
        eri: &Array4<f64>,
        m: &Array2<T>,
    ) -> Array2<T> {
        let n = m.nrows();
        let mut j = Array2::<T>::zeros((n, n));

        // Each row of J^{(m_i)} is independent and may be evaluated in parallel.
        j.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(s, mut row)| {
                for t in 0..n {
                    let mut acc = <T as From<f64>>::from(0.0);
                    for mu in 0..n {
                        for nu in 0..n {
                            acc += <T as From<f64>>::from(eri[(s, t, mu, nu)]) * m[(mu, nu)];
                        }
                    }
                    row[t] = acc;
                }
            });
        j
    }

    /// Construct the exchange contraction required by the same-spin two-body intermediates.
    /// The AO contraction is:
    /// K_{st}^{(m_i)} = \sum_{\mu\nu}(s\mu|\nu t){}^{xw}M_{\mu\nu}^{(m_i)}.
    /// # Arguments:
    /// - `eri`: Non-antisymmetrised AO two-electron integrals.
    /// - `m`: AO fundamental contraction {}^{xw}M^{(m_i)}.
    /// # Returns
    /// - `Array2<T>`: Exchange contraction K^{(m_i)}.
    fn build_k_exchange(
        eri: &Array4<f64>,
        m: &Array2<T>,
    ) -> Array2<T> {
        let n = m.nrows();
        let mut k = Array2::<T>::zeros((n, n));

        // Each row of K^{(m_i)} is independent and may be evaluated in parallel.
        k.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(s, mut row)| {
                for t in 0..n {
                    let mut acc = <T as From<f64>>::from(0.0);
                    for mu in 0..n {
                        for nu in 0..n {
                            acc += <T as From<f64>>::from(eri[(s, mu, nu, t)]) * m[(mu, nu)];
                        }
                    }
                    row[t] = acc;
                }
            });
        k
    }
}

/// Distribution (m_{\alpha 0},m_{\alpha z},m_{\beta 0},m_{\beta y}) associated with a stored
/// different-spin \mathcal{II} intermediate.
type IIMask = (usize, usize, usize, usize);
/// Different-spin \mathcal{II} intermediate stored together with its distribution of zero-overlap orbital pairs.
type IIABBlock<T> = (IIMask, Array4<T>);

/// Precomputed different-spin two-body intermediates for a pair of non-orthogonal reference determinants.
#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(bound = "T: NOCIScalar")]
pub struct DiffSpinBuild<T: NOCIScalar> {
    /// Scalar different-spin intermediates {}^x V_{\alpha\beta,0}^{(m_{\alpha 0},m_{\beta 0})}.
    pub vab0: [[T; 2]; 2],
    /// Alpha-spin column intermediates \mathcal V^\alpha stored as
    /// vab[m_{\alpha 0}][m_{\beta 0}][m_{\alpha z}].
    pub vab: [[[Array2<T>; 2]; 2]; 2],
    /// Scalar different-spin intermediates {}^x V_{\alpha\beta,0}^{(m_{\alpha 0},m_{\beta 0})} stored with the beta-spin assignment first.
    pub vba0: [[T; 2]; 2],
    /// Beta-spin column intermediates \mathcal V^\beta stored as
    /// vba[m_{\beta 0}][m_{\alpha 0}][m_{\beta y}].
    pub vba: [[[Array2<T>; 2]; 2]; 2],
    /// Different-spin \mathcal{II} intermediates indexed by
    /// (m_{\alpha 0},m_{\alpha z},m_{\beta 0},m_{\beta y}).
    pub iiab: Vec<IIABBlock<T>>,
}

impl<T: NOCIScalar> DiffSpinBuild<T> {
    /// Construct the different-spin two-body intermediates required to evaluate matrix elements between arbitrary excited determinants.
    /// For a different-spin two-body operator, the contraction determinant factorises into separate alpha- and beta-spin
    /// determinants. Multiplying the two Laplace expansions gives the scalar contribution V_{\alpha\beta,0}, the two
    /// column contributions \mathcal V^\alpha and \mathcal V^\beta, and the contribution containing one replaced column
    /// in each spin space, \mathcal{II}.
    /// The zero-overlap orbital pairs are distributed independently within each spin space according to:
    /// m_{\alpha 0} + \sum_z m_{\alpha z} = m_\alpha,
    /// m_{\beta 0} + \sum_y m_{\beta y} = m_\beta.
    /// No exchange term occurs between the alpha- and beta-spin operator pairs.
    /// # Arguments:
    /// - `ao`: AO overlap matrix and two-electron integrals.
    /// - `g`: Ket reference determinant |{}^w\Psi\rangle.
    /// - `l`: Bra reference determinant \langle{}^x\Psi|.
    /// - `tol`: Singular values satisfying |{}^{xw}\tilde S_{\sigma i}| \leq \mathtt{tol} are treated as zero.
    /// # Returns
    /// - `DiffSpinBuild<T>`: Different-spin two-body intermediates for the reference determinant pair.
    pub fn new(
        ao: &AoData,
        g: &DetState<T>,
        l: &DetState<T>,
        tol: f64,
    ) -> Self {
        let eri = &ao.eri_coul;
        let s_munu = &ao.s;

        // Extract the alpha- and beta-spin MO coefficients for the ket reference
        // |{}^w\Psi\rangle and bra reference \langle{}^x\Psi|.
        let g_ca = g.ca.as_ref();
        let g_cb = g.cb.as_ref();
        let l_ca = l.ca.as_ref();
        let l_cb = l.cb.as_ref();

        // Extract the corresponding alpha- and beta-spin occupation bitstrings.
        let goa = g.oa;
        let gob = g.ob;
        let loa = l.oa;
        let lob = l.ob;

        let nmo = g_ca.ncols();

        // Extract the occupied alpha- and beta-spin orbital coefficients used
        // to form the occupied orbital overlap matrices.
        let l_ca_occ = occ_coeffs(l_ca, loa);
        let g_ca_occ = occ_coeffs(g_ca, goa);
        let l_cb_occ = occ_coeffs(l_cb, lob);
        let g_cb_occ = occ_coeffs(g_cb, gob);

        // Construct the Löwdin paired occupied orbitals independently for the
        // alpha- and beta-spin spaces.
        let (tilde_sa_occ, g_tilde_ca_occ, l_tilde_ca_occ, _phase) =
            SameSpinBuild::perform_ortho_and_svd_and_rotate(s_munu, &l_ca_occ, &g_ca_occ);
        let (tilde_sb_occ, g_tilde_cb_occ, l_tilde_cb_occ, _phase) =
            SameSpinBuild::perform_ortho_and_svd_and_rotate(s_munu, &l_cb_occ, &g_cb_occ);

        // Identify the m_\alpha and m_\beta zero-overlap orbital pairs.
        let zerosa: Vec<usize> = tilde_sa_occ
            .iter()
            .enumerate()
            .filter_map(|(k, &sk)| if sk.abs() <= tol { Some(k) } else { None })
            .collect();
        let zerosb: Vec<usize> = tilde_sb_occ
            .iter()
            .enumerate()
            .filter_map(|(k, &sk)| if sk.abs() <= tol { Some(k) } else { None })
            .collect();
        let ma = zerosa.len();
        let mb = zerosb.len();

        // Construct M_\alpha^{(0)}, M_\alpha^{(1)}, M_\beta^{(0)} and M_\beta^{(1)}.
        let (m0a, m1a) = SameSpinBuild::construct_m(
            &tilde_sa_occ,
            &l_tilde_ca_occ,
            &g_tilde_ca_occ,
            &zerosa,
            tol,
        );
        let (m0b, m1b) = SameSpinBuild::construct_m(
            &tilde_sb_occ,
            &l_tilde_cb_occ,
            &g_tilde_cb_occ,
            &zerosb,
            tol,
        );
        let mao_a = [&m0a, &m1a];
        let mao_b = [&m0b, &m1b];

        // Construct the alpha- and beta-spin Coulomb contractions. There is no exchange contribution between different spin spaces.
        let ja = [
            SameSpinBuild::build_j_coulomb(eri, mao_a[0]),
            SameSpinBuild::build_j_coulomb(eri, mao_a[1]),
        ];
        let jb = [
            SameSpinBuild::build_j_coulomb(eri, mao_b[0]),
            SameSpinBuild::build_j_coulomb(eri, mao_b[1]),
        ];

        // Construct the scalar different-spin intermediate
        // {}^x V_{\alpha\beta,0}^{(m_{\alpha 0},m_{\beta 0})} = \sum_{pqrs}{}^x v_{pqrs}
        // {}^{xx}X_{\alpha,rp}^{(m_{\alpha 0})} {}^{xx}X_{\beta,sq}^{(m_{\beta 0})}.
        // vba0 stores the same scalar intermediate with the beta-spin assignment first for use with \mathcal V^\beta.
        let z = <T as From<f64>>::from(0.0);
        let mut vab0 = [[z; 2]; 2];
        let mut vba0 = [[z; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                vab0[i][j] = T::einsum_ba_ab(&ja[i], mao_b[j]);
                vba0[j][i] = T::einsum_ba_ab(&jb[j], mao_a[i]);
            }
        }

        // Construct the X^{(m_i)} or Y^{(m_i)} contractions required on the left- and right-hand sides of the
        // Coulomb contraction, independently for the alpha- and beta-spin spaces and for m_i = 0 and m_i = 1.
        let (cx_a0, xc_a0) = Self::build_cx_xc(mao_a[0], s_munu, l_ca, g_ca, loa, 0);
        let (cx_a1, xc_a1) = Self::build_cx_xc(mao_a[1], s_munu, l_ca, g_ca, loa, 1);
        let (cx_b0, xc_b0) = Self::build_cx_xc(mao_b[0], s_munu, l_cb, g_cb, lob, 0);
        let (cx_b1, xc_b1) = Self::build_cx_xc(mao_b[1], s_munu, l_cb, g_cb, lob, 1);
        let cx_a = [&cx_a0, &cx_a1];
        let xc_a = [&xc_a0, &xc_a1];
        let cx_b = [&cx_b0, &cx_b1];
        let xc_b = [&xc_b0, &xc_b1];

        // Construct the different-spin one-column intermediates:
        // {}^{\chi_\eta\chi_z}\mathcal V^\alpha_{\eta z}[C,A]^{(m_{\alpha 0},m_{\beta 0},m_{\alpha z})}
        // = \sum_{pqrs} {}^{\chi_\eta x}C_{\alpha,\eta p}^{(m_{\alpha 0})}{}^x v_{pqrs}
        // {}^{x\chi_z}A_{\alpha,rz}^{(m_{\alpha z})}{}^{xx}X_{\beta,sq}^{(m_{\beta 0})},
        // and the corresponding beta-spin intermediate \mathcal V^\beta. These are stored as
        // vab[m_{\alpha 0}][m_{\beta 0}][m_{\alpha z}] and vba[m_{\beta 0}][m_{\alpha 0}][m_{\beta y}].
        let mut vab: [[[Array2<T>; 2]; 2]; 2] = std::array::from_fn(|_| {
            std::array::from_fn(|_| std::array::from_fn(|_| Array2::<T>::zeros((nmo, nmo))))
        });
        let mut vba: [[[Array2<T>; 2]; 2]; 2] = std::array::from_fn(|_| {
            std::array::from_fn(|_| std::array::from_fn(|_| Array2::<T>::zeros((nmo, nmo))))
        });

        // Enumerate all combinations with m_i = 0 or m_i = 1 for the three fundamental contractions.
        let combos: Vec<(usize, usize, usize)> = (0..2)
            .flat_map(|ma0| (0..2).flat_map(move |mb0| (0..2).map(move |mk| (ma0, mb0, mk))))
            .collect();

        // Construct \mathcal V^\alpha by contracting the beta-spin Coulomb matrix between the two alpha-spin contractions.
        let vabblocks: Vec<((usize, usize, usize), Array2<T>)> = combos
            .clone()
            .into_par_iter()
            .map(|(ma0, mb0, mak)| {
                let blk = adjoint(cx_a[ma0]).dot(&jb[mb0]).dot(xc_a[mak]);
                ((ma0, mb0, mak), blk)
            })
            .collect();
        for ((ma0, mb0, mak), blk) in vabblocks {
            vab[ma0][mb0][mak] = blk;
        }

        // Construct \mathcal V^\beta by contracting the alpha-spin Coulomb matrix between the two beta-spin contractions.
        let vbablocks: Vec<((usize, usize, usize), Array2<T>)> = combos
            .into_par_iter()
            .map(|(ma0, mb0, mbk)| {
                let blk = adjoint(cx_b[mb0]).dot(&ja[ma0]).dot(xc_b[mbk]);
                ((mb0, ma0, mbk), blk)
            })
            .collect();
        for ((mb0, ma0, mbk), blk) in vbablocks {
            vba[mb0][ma0][mbk] = blk;
        }

        // Construct the different-spin two-column intermediates:
        // {}^{\chi_\eta\chi_z,\chi_\xi\chi_y}\mathcal{II}_{\eta z,\xi y}
        // [C,A,D,B]^{(m_{\alpha 0},m_{\alpha z},m_{\beta 0},m_{\beta y})}
        // = \sum_{pqrs} {}^x v_{pqrs}{}^{\chi_\eta x}C_{\alpha,\eta p}^{(m_{\alpha 0})}
        // {}^{x\chi_z}A_{\alpha,rz}^{(m_{\alpha z})}{}^{\chi_\xi x}D_{\beta,\xi q}^{(m_{\beta 0})}
        // {}^{x\chi_y}B_{\beta,sy}^{(m_{\beta y})}. There is no different-spin exchange contribution.
        let combos: Vec<(usize, usize, usize, usize)> = (0..2)
            .flat_map(|ma0| {
                (0..2).flat_map(move |maz| {
                    (0..2).flat_map(move |mb0| {
                        (0..2).filter_map(move |mby| {
                            if ma0 + maz <= ma && mb0 + mby <= mb {
                                Some((ma0, maz, mb0, mby))
                            } else {
                                None
                            }
                        })
                    })
                })
            })
            .collect();

        // Transform the AO two-electron integrals for each allowed distribution of zero-overlap orbital pairs.
        let iiab: Vec<IIABBlock<T>> = combos
            .into_par_iter()
            .map_init(
                || -> ERIAO2MOScratch<T> { T::new_eri_ao2mo_scratch(eri, nmo, nmo, nmo, nmo) },
                |scratch, (ma0, maz, mb0, mby)| {
                    let mut blk = Array4::<T>::zeros((nmo, nmo, nmo, nmo));
                    T::eri_ao2mo_hermitian_into(
                        eri,
                        cx_a[ma0],
                        xc_a[maz],
                        cx_b[mb0],
                        xc_b[mby],
                        blk.view_mut(),
                        scratch,
                    );
                    ((ma0, maz, mb0, mby), blk)
                },
            )
            .collect();

        Self {
            vab0,
            vab,
            vba0,
            vba,
            iiab,
        }
    }

    /// Construct the matrices containing the X^{(m_i)} or Y^{(m_i)} contractions required by the stored intermediates.
    /// For m_i = 0, the direct orbital contribution is subtracted from the left x-reference block and the right
    /// w-reference block, giving the required Y^{(0)} contractions. For m_i = 1, no subtraction is required because
    /// Y^{(1)} = X^{(1)}. The first returned matrix is restricted to the x-reference virtual and w-reference
    /// occupied orbitals associated with the rows of the contraction determinant. The second is restricted to the x-reference
    /// occupied and w-reference virtual orbitals associated with its columns.
    /// # Arguments:
    /// - `m`: AO fundamental contraction {}^{xw}M^{(m_i)}.
    /// - `s`: AO overlap matrix \mathbf S.
    /// - `l_c`: Bra-reference MO coefficient matrix {}^x\mathbf C.
    /// - `g_c`: Ket-reference MO coefficient matrix {}^w\mathbf C.
    /// - `lo`: Bra-reference occupation bitstring.
    /// - `i`: Zero-overlap assignment m_i, equal to zero or one.
    /// # Returns
    /// - `(Array2<T>, Array2<T>)`: Matrices containing the contractions associated with the rows and columns of the
    ///   contraction determinant.
    fn build_cx_xc(
        m: &Array2<T>,
        s: &Array2<f64>,
        l_c: &Array2<T>,
        g_c: &Array2<T>,
        lo: u128,
        i: usize,
    ) -> (Array2<T>, Array2<T>) {
        let nao = l_c.nrows();
        let nmo = l_c.ncols();

        let nocc = lo.count_ones() as usize;

        let smat = real2_as::<T>(s);
        let mut cx_full = Array2::<T>::zeros((nao, 2 * nmo));
        let mut xc_full = Array2::<T>::zeros((nao, 2 * nmo));

        let one_minus_i = <T as From<f64>>::from((1 - i) as f64);

        // Form M^{(m_i)}S and its adjoint.
        let ms = m.dot(&smat);
        let mts = adjoint(m).dot(&smat);

        // The direct orbital contribution is subtracted only for m_i = 0, for which Y^{(0)} = X^{(0)} - S.
        let l_scaled = l_c.mapv(|z| z * one_minus_i);
        let g_scaled = g_c.mapv(|z| z * one_minus_i);

        // Construct the x- and w-reference blocks required on the left- and right-hand sides of the intermediates.
        cx_full
            .slice_mut(s![.., 0..nmo])
            .assign(&(mts.dot(l_c) - &l_scaled));
        cx_full
            .slice_mut(s![.., nmo..2 * nmo])
            .assign(&mts.dot(g_c));
        xc_full.slice_mut(s![.., 0..nmo]).assign(&ms.dot(l_c));
        xc_full
            .slice_mut(s![.., nmo..2 * nmo])
            .assign(&(ms.dot(g_c) - &g_scaled));

        // Restrict the left contraction to V_x \cup O_w and the right contraction to O_x \cup V_w.
        let row_idx: Vec<_> = (nocc..nmo).chain(nmo..nmo + nocc).collect();
        let col_idx: Vec<_> = (0..nocc).chain(nmo + nocc..2 * nmo).collect();

        (
            cx_full.select(Axis(1), &row_idx),
            xc_full.select(Axis(1), &col_idx),
        )
    }
}

/// Construct the orbital coefficient matrices associated with the rows and columns of the contraction determinant.
/// For a matrix element
/// \langle{}^x\Psi_{i\cdots}^{a\cdots}|\hat O|{}^w\Psi_{j\cdots}^{b\cdots}\rangle,
/// the rows correspond to the annihilation operators {}^x\hat b_a,\ldots and {}^w\hat b_j,\ldots, while the columns
/// correspond to the creation operators {}^x\hat b_i^\dagger,\ldots and {}^w\hat b_b^\dagger,\ldots. The coefficient
/// matrices are therefore ordered as:
/// \mathbf C_{\mathrm{row}} = \begin{pmatrix}{}^x\mathbf C_{\mathrm{vir}} & {}^w\mathbf C_{\mathrm{occ}}\end{pmatrix},
/// \mathbf C_{\mathrm{col}} = \begin{pmatrix}{}^x\mathbf C_{\mathrm{occ}} & {}^w\mathbf C_{\mathrm{vir}}\end{pmatrix}.
/// # Arguments:
/// - `l_c`: Bra-reference molecular-orbital coefficients {}^x\mathbf C.
/// - `g_c`: Ket-reference molecular-orbital coefficients {}^w\mathbf C.
/// - `nocc`: Number of occupied orbitals in this spin space.
/// # Returns
/// - `(Array2<T>, Array2<T>)`: Orbital coefficient matrices associated with the contraction determinant rows and columns.
pub(in crate::nonorthogonalwicks) fn contraction_orbitals<T: NOCIScalar>(
    l_c: &Array2<T>,
    g_c: &Array2<T>,
    nocc: usize,
) -> (Array2<T>, Array2<T>) {
    let nbas = l_c.nrows();
    let nmo = l_c.ncols();
    let nvirt = nmo - nocc;

    let mut rowc = Array2::<T>::zeros((nbas, nmo));
    let mut colc = Array2::<T>::zeros((nbas, nmo));

    rowc.slice_mut(s![.., 0..nvirt])
        .assign(&l_c.slice(s![.., nocc..nmo]));
    rowc.slice_mut(s![.., nvirt..nmo])
        .assign(&g_c.slice(s![.., 0..nocc]));

    colc.slice_mut(s![.., 0..nocc])
        .assign(&l_c.slice(s![.., 0..nocc]));
    colc.slice_mut(s![.., nocc..nmo])
        .assign(&g_c.slice(s![.., nocc..nmo]));

    (rowc, colc)
}
