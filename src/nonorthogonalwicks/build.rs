// nonorthogonalwicks/build.rs 
use ndarray::{Array1, Array2, Array4, Axis, s};
use ndarray_linalg::{SVD, Determinant};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

use crate::maths::{einsum_ba_ab_real, eri_ao2mo, loewdin_x_real};
use crate::noci::occ_coeffs;

/// Owning struct for the same-spin computed intermediates.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SameSpinBuild {
    /// X[mi] contraction matrices for the two branch choices.
    pub x: [Array2<f64>; 2],
    /// Y[mi] contraction matrices for the two branch choices.
    pub y: [Array2<f64>; 2],
    /// Zeroth-order Fock one-body scalar contributions for the two branch choices.
    pub f0f: [f64; 2],
    /// Zeroth-order Hamiltonian one-body scalar contributions for the two branch choices.
    pub f0h: [f64; 2],
    /// Hamiltonian one-body F[mi][mj] intermediates.
    pub fh: [[Array2<f64>; 2]; 2],
    /// Fock one-body F[mi][mj] intermediates.
    pub ff: [[Array2<f64>; 2]; 2],
    /// Zeroth-order two-body scalar contributions for the allowed branch combinations.
    pub v0: [f64; 3],
    /// Same-spin V[mi][mj][mk] intermediates.
    pub v: [[[Array2<f64>; 2]; 2]; 2],
    /// Compressed same-spin J[mi][mj][mk][ml] tensors.
    pub j: [Array4<f64>; 10],
    /// Product of the non-zero singular values for this same-spin block.
    pub tilde_s_prod: f64,
    /// Overall phase associated with this same-spin block.
    pub phase: f64,
    /// Number of zero-overlap orbital pairs in the biorthogonal basis.
    pub m: usize,
    /// Number of molecular orbitals for this same-spin block.
    pub nmo: usize,
}

impl SameSpinBuild {
    /// Constructor for the WicksReferencePair object of SameSpin which contains the precomputed values
    /// per pair of reference determinants required to evaluate arbitrary excited determinants 
    /// in O(1) time when the excitations are of the same spin.
    /// # Arguments:
    /// - `eri`: Electron repulsion integrals. 
    /// - `h_munu`: AO core Hamiltonian.
    /// - `s_munu`: AO overlap matrix.
    /// - `g_c`: Full AO coefficient matrix of |^\Gamma\Psi\rangle.
    /// - `l_c`: Full AO coefficient matrix of |^\Lambda\Psi\rangle.
    /// - `go`: Occupancy vector for |^\Gamma\Psi\rangle.
    /// - `lo`: Occupancy vector for |^\Lambda\Psi\rangle. 
    /// - `tol`: Tolerance for whether a number is considered zero.
    /// # Returns
    /// - `SameSpinBuild`: Precomputed same-spin Wick's intermediates for the reference pair.
    pub fn new(eri: &Array4<f64>, h_munu: &Array2<f64>, s_munu: &Array2<f64>, g_c: &Array2<f64>, l_c: &Array2<f64>, go: u128, lo: u128, tol: f64) -> Self {
        let nmo = g_c.ncols();
        let nbas = l_c.nrows();

        let mut ccat = Array2::<f64>::zeros((nbas, 2 * nmo));
        ccat.slice_mut(s![.., 0..nmo]).assign(l_c);
        ccat.slice_mut(s![.., nmo..2*nmo]).assign(g_c);

        let l_c_occ = occ_coeffs(l_c, lo);
        let g_c_occ = occ_coeffs(g_c, go);

        // SVD and rotate the occupied orbitals.
        let (tilde_s_occ, g_tilde_c_occ, l_tilde_c_occ, phase) = Self::perform_ortho_and_svd_and_rotate(s_munu, &l_c_occ, &g_c_occ, 1e-20);

        // Multiply diagonal non-zero values of {}^{\Gamma\Lambda} \tilde{S} together.
        let tilde_s_prod = tilde_s_occ.iter().filter(|&&x| x.abs() > tol).product::<f64>();

        // Find indices where zeros occur in {}^{\Gamma\Lambda} \tilde{S} and count them.
        let zeros: Vec<usize> = tilde_s_occ.iter().enumerate().filter_map(|(k, &sk)| if sk.abs() <= tol {Some(k)} else {None}).collect();
        let m = zeros.len();
        
        // Construct the {}^{\Gamma\Lambda} M^{\sigma\tau, 0} and {}^{\Gamma\Lambda} M^{\sigma\tau, 1} matrices. 
        // The argument order here is swapped compared to what the function asks for, but it does not match the naive
        // implementation if we swap them.
        let (m0, m1) = Self::construct_m(&tilde_s_occ, &l_tilde_c_occ, &g_tilde_c_occ, &zeros, tol);
        let mao: [Array2<f64>; 2] = [m0, m1];

        // Construct the {}^{\Gamma\Lambda} X_{ij}^{m_k} and {}^{\Gamma\Lambda} Y_{ij}^{m_k} matrices.
        let (x0, y0) = Self::construct_xy(g_c, l_c, s_munu, &mao[0], true);
        let (x1, y1) = Self::construct_xy(g_c, l_c, s_munu, &mao[1], false);
        let x: [Array2<f64>; 2] = [x0, x1];
        let y: [Array2<f64>; 2] = [y0, y1];

        // Construct Coulomb and exchange contractions of ERIs with  {}^{\Gamma\Lambda} M^{\sigma\tau, m_k}, 
        // {}^{\Gamma\Lambda} J_{\mu\nu}^{m_k} and {}^{\Gamma\Lambda} K_{\mun\u}^{m_k}. These
        // quantities are used in many of the following intermediates so we precompute here.
        let nbas = mao[0].nrows();
        let mut jkao: [Array2<f64>; 2] = [Array2::<f64>::zeros((nbas, nbas)), Array2::<f64>::zeros((nbas, nbas))];
        for mi in 0..2 {
            let j = Self::build_j_coulomb(eri, &mao[mi]);
            let k = Self::build_k_exchange(eri, &mao[mi]);
            jkao[mi] = &j - &k;
        }
        
        // Construct the {}^{\Gamma\Lambda} F_0^{m_k} and {}^{\Lambda\Gamma} F_{ab}^{m_i, m_j}
        // intermediates required for one electron Hamiltonian matrix elements.
        let (f0_0h, f00h) = Self::construct_f(l_c, h_munu, &x[0], &y[0]);
        let (_, f01h) = Self::construct_f(l_c, h_munu, &x[0], &y[1]);
        let (_, f10h) = Self::construct_f(l_c, h_munu, &x[1], &y[0]);
        let (f0_1h, f11h) = Self::construct_f(l_c, h_munu, &x[1], &y[1]);
        let f0h: [f64; 2] = [f0_0h, f0_1h];
        let fh: [[Array2<f64>; 2]; 2] = [[f00h, f01h], [f10h, f11h]];
        
        // Initialise {}^{\Gamma\Lambda} F_0^{m_k} and {}^{\Lambda\Gamma} F_{ab}^{m_i, m_j} for 
        // Fock matrix elements to zero as when used in SNOCI these will change per iteration.
        let f0f: [f64; 2] = [0.0, 0.0];
        let ff: [[Array2<f64>; 2]; 2] = [[Array2::zeros((2 * nmo, 2 * nmo)), Array2::zeros((2 * nmo, 2 * nmo))],
                                         [Array2::zeros((2 * nmo, 2 * nmo)), Array2::zeros((2 * nmo, 2 * nmo))]];
        
        // Calculate the left and right factorisations of the {}^{\Gamma\Lambda} X_{ij}^{m_k} and 
        // {}^{\Gamma\Lambda} Y_{ij}^{m_k} matrices, {}^{\Gamma\Lambda} so as to avoid branching
        // down the line.
        let mut cx: [Array2<f64>; 2] = [Array2::<f64>::zeros((nbas, 2*nmo)), Array2::<f64>::zeros((nbas, 2*nmo))];
        let mut xc: [Array2<f64>; 2] = [Array2::<f64>::zeros((nbas, 2*nmo)), Array2::<f64>::zeros((nbas, 2*nmo))];
        for mi in 0..2 {
            (cx[mi], xc[mi]) = DiffSpinBuild::build_cx_xc(&mao[mi], s_munu, l_c, g_c, mi);
        }

        // Construct {}^{\Lambda\Gamma} V_0^{m_i, m_j} = \sum_{prqs} ({}^{\Lambda}(pr|qs) -
        // {}^{\Lambda}(ps|qr)) {}^{\Lambda\Gamma} X_{sq}^{m_i} {}^{\Lambda\Gamma}. This can be
        // rewritten (and thus calculated) as V_0^{m_i, m_j} = \sum_{pr} (J_{\mu\nu}^{m_i} - K_{\mu\nu}^{m_i}) 
        // {}^{\Gamma\Lambda} M^{\sigma\tau, m_j}.
        let mut v0 = [0.0f64; 3];
        v0[0] = einsum_ba_ab_real(&jkao[0], &mao[0]);
        if m > 1 {
            v0[1] = 2.0 * einsum_ba_ab_real(&jkao[0], &mao[1]);
            v0[2] = einsum_ba_ab_real(&jkao[1], &mao[1]);
        } else {
            v0[1] = 0.0;
        }

        // Construct {}^{\Lambda\Gamma} V_{ab}^{m_1, m_2, m_3} intermediates for two electron
        // Hamiltonian matrix elements. These are given as:
        //      {}^{\Lambda\Gamma} V_{ab}^{m_1, m_2, m_3} = \sum_{pq} {}^{\Lambda\Gamma} Y_{ap}^{m_1}
        //      (\sum_{rs} ({}^{\Lambda}(pr|qs) - {}^{\Lambda}(ps|qr)) {}^{\Lambda\Gamma} X_{sr}^{m_2}) 
        //      {}^{\Lambda\Gamma} X_{qb}^{m_3},
        // where the use of X or Y on the left and righthand sides depends on the ordering of
        // \Lambda and \Gamma. Again using our precomputed quantities we rewrite as:
        //      {}^{\Lambda\Gamma} V_{ab}^{m_1, m_2, m_3} = \sum_{pq} C_{L,ap}^{m_1} (J_{\mu\nu}^{m_2} - K_{\mu\nu}^{m_2})
        //      C_{R,qb}^{m_3}.
        let mut v: [[[Array2<f64>; 2]; 2]; 2] = std::array::from_fn(|_| {
            std::array::from_fn(|_| {
                std::array::from_fn(|_| Array2::<f64>::zeros((2 * nmo, 2 * nmo)))
            })
        });
        let combos: Vec<(usize, usize, usize)> =
            (0..2).flat_map(|mi|
            (0..2).flat_map(move |mj|
            (0..2).map(move |mk| (mi, mj, mk))
        )).collect();
        let blocks: Vec<((usize, usize, usize), Array2<f64>)> =
            combos.into_par_iter().map(|(mi, mj, mk)| {
                let blk = cx[mi].t().dot(&jkao[mk]).dot(&xc[mj]);
                ((mi, mj, mk), blk)
            }).collect();
        for ((mi, mj, mk), blk) in blocks {
            v[mi][mj][mk] = blk;
        }

        // Construct {}^{\Lambda\Gamma} J_{ab,cd}^{m_1,m_2,m_3,m_4} intermediates for two electron
        // Hamiltonian matrix elements. These are given as:
        //      {}^{\Lambda\Gamma} J_{ab,cd}^{m_1,m_2,m_3,m_4} = \sum_{prqs} ({}^{\Lambda}(pr|qs) - {}^{\Lambda}(ps|qr))
        //      {}^{\Lambda\Gamma} Y_{ap}^{m_1} {}^{\Lambda\Gamma} X_{rb}^{m_2} {}^{\Lambda\Gamma} Y_{cq}^{m_3} {}^{\Lambda\Gamma} X_{sd}^{m_4},
        // where the use of X or Y in each part depends on the ordering of \Lambda and \Gamma. Again using our quantities
        // this may instead be calculated as
        //      {}^{\Lambda\Gamma} J_{ab,cd}^{m_1, m_2, m_3, m_4} = \sum_{\mu\nu\tau\sigma} C_{L,a\mu}^{m_1} C_{R,b\nu}^{m_2}
        //      ((\mu\nu|\tau\sigma) - (\mu\sigma|\tau\nu)) C_{L,c\tau}^{m_3} C_{R,d\sigma}^{m_4},
        // which is achieved by antisymmetrising the AO integrals and transforming from AO to MO
        // basis. Only 10 of 16 4D tensors of J need be stored due to symmetry.
        let mut j: [Array4<f64>; 10] = std::array::from_fn(|_| {
            Array4::<f64>::zeros((2 * nmo, 2 * nmo, 2 * nmo, 2 * nmo))
        });
        let combos: [(usize, usize, usize, usize); 10] = [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 0, 1), 
                                                          (0, 1, 1, 0), (0, 1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 1), (1, 1, 1, 1)];
        let blocks: Vec<(usize, Array4<f64>)> = combos.into_par_iter().enumerate().map(|(s, (mi, mj, mk, ml))| {
            let mut blk = eri_ao2mo(eri, &cx[mi], &xc[mj], &cx[mk], &xc[ml]);
            let ex = blk.view().permuted_axes([0, 2, 1, 3]).to_owned().as_standard_layout().to_owned();
            blk -= &ex;
            (s, blk.as_standard_layout().to_owned())
        }).collect();
        for (s, blk) in blocks {
            j[s] = blk;
        }

        Self {x, y, f0h, fh, f0f, ff, v0, v, j, tilde_s_prod, phase, m, nmo}
    }

    /// Perform singular value decomposition on the occupied orbital overlap matrix {}^{\Gamma\Lambda} S_{ij} as:
    ///     {}^{\Gamma\Lambda} \mathbf{S} = \mathbf{U} {}^{\Gamma\Lambda} \mathbf{\tilde{S}} \mathbf{V}^\dagger,
    /// and rotate the occupied coefficients:
    ///     |{}^\Gamma \Psi_i\rangle = \sum_{\mu} {}^\Gamma c_i^\mu U_{ij} |\phi_\mu \rangle.
    ///     |{}^\Lambda \Psi_j\rangle = \sum_{\nu} {}^\Lambda c_j^\nu V_{ij} |\phi_\nu \rangle.
    /// # Arguments:
    /// - `g_c_occ`: Occupied coefficients ({}^\Gamma C^*)_i^\mu.
    /// - `l_c_occ`: Occupied coefficients ({}^\Lambda C)_j^\nu.
    /// - `tol`: Tolerance for the orthonormalisation step.
    /// # Returns
    /// - `(Array1<f64>, Array2<f64>, Array2<f64>, f64)`: Singular values, rotated occupied coefficients
    ///   for \Gamma and \Lambda, and the phase associated with the rotation.
    pub fn perform_ortho_and_svd_and_rotate(s_munu: &Array2<f64>, l_c_occ: &Array2<f64>, g_c_occ: &Array2<f64>, tol: f64) 
                                            -> (Array1<f64>, Array2<f64>, Array2<f64>, f64) {
        
        // Orthonormalise.
        let s_ll = l_c_occ.t().dot(&s_munu.dot(l_c_occ));
        let s_gg = g_c_occ.t().dot(&s_munu.dot(g_c_occ));
        let x_l = loewdin_x_real(&s_ll, true, tol);
        let x_g = loewdin_x_real(&s_gg, true, tol);
        let l_c_occ_ortho = l_c_occ.dot(&x_l);
        let g_c_occ_ortho = g_c_occ.dot(&x_g);

        let lg_s = l_c_occ_ortho.t().dot(&s_munu.dot(&g_c_occ_ortho));

        // SVD.
        let (u, lg_tilde_s, v_dag) = lg_s.svd(true, true).unwrap();
        let u = u.unwrap();
        let v = v_dag.unwrap().t().to_owned();
        
        // Rotate MOs.
        let l_tilde_c = l_c_occ_ortho.dot(&u);
        let g_tilde_c = g_c_occ_ortho.dot(&v);
        
        // Calculate phase associated with rotation.
        let det_u = u.det().unwrap();
        let det_v = v.det().unwrap();
        let ph = det_u * det_v;
        
        (lg_tilde_s, g_tilde_c, l_tilde_c, ph)
    }
    
    /// Form the matrices {}^{\Gamma\Lambda} M^{\sigma\tau, 0} and{}^{\Gamma\Lambda} M^{\sigma\tau, 1} as:
    ///     {}^{\Gamma\Lambda} M^{\sigma\tau, 0} = {}^{\Gamma\Lambda} W^{\sigma\tau} + {}^{\Gamma\Lambda} P^{\sigma\tau} + {}^{\Gamma\Gamma} P^{\sigma\tau}
    ///     {}^{\Gamma\Lambda} M^{\sigma\tau, 1} = {}^{\Gamma\Lambda} P^{\sigma\tau}.
    /// The components {}^{\Gamma\Lambda} W^{\sigma\tau}, {}^{\Gamma\Lambda} P^{\sigma\tau}, 
    /// {}^{\Gamma\Gamma} P^{\sigma\tau} are constructed sequentially and added into the correct
    /// matrix.
    /// # Arguments:
    /// - `gl_tilde_s`: Vector of diagonal single values of {}^{\Gamma\Lambda} \tilde{S}.
    /// - `g_c_tilde_occ`: Rotated occupied coefficients ({}^\Gamma \tilde{C}^*)_i^\mu.
    /// - `l_c_tilde_occ`: Rotated occupied coefficients ({}^\Lambda \tilde{C})_j^\nu.
    /// - `zeros`: Indices where zeros occur in {}^{\Gamma\Lambda} \tilde{S}. 
    /// - `tol`: Tolerance for whether a singular value is considered zero.
    /// # Returns
    /// - `(Array2<f64>, Array2<f64>)`: The M^{0} and M^{1} matrices.
    pub fn construct_m(lg_tilde_s: &Array1<f64>, g_tilde_c_occ: &Array2<f64>, l_tilde_c_occ: &Array2<f64>, zeros: &Vec<usize>, tol: f64) -> (Array2<f64>, Array2<f64>) {
        let nbas = g_tilde_c_occ.nrows();
        let nocc = g_tilde_c_occ.ncols();
        
        // Calculate {}^{\Lambda\Gamma} W^{\sigma\tau} (weighted co-density matrix) as:
        //      {}^{\Lambda\Gamma} W^{\sigma\tau} = \sum_i (^\Lambda \tilde{C})^\sigma_i (1 /
        //      {}^{\Lambda\Gamma} \tilde{S}_i) (^\Gamma \tilde{C}^*)^\tau_i
        //  for all i where {}^{\Lambda\Gamma} \tilde{S}_i != 0. This result is stored in
        //  {}^{\Lambda\Gamma} M^{\sigma\tau, 0}, where the zero indicates this quantity will be
        //  used when m_k = 0. 
        let mut l_tilde_c_occ_scaled = l_tilde_c_occ.clone();
        for k in 0..nocc {
            let s = lg_tilde_s[k];
            if s.abs() > tol {
                let scale = 1.0 / s;
            let mut col = l_tilde_c_occ_scaled.column_mut(k);
            col *= scale;
            } else {
                l_tilde_c_occ_scaled.column_mut(k).fill(0.0);
            }
        }
        let mut lg_m0 = l_tilde_c_occ_scaled.dot(&g_tilde_c_occ.t());
        let mut lg_m1 = Array2::<f64>::zeros((nbas, nbas));
        let mut ll_m0 = Array2::<f64>::zeros((nbas, nbas));
        
        // Calculate {}^{\Lambda\Lambda} P^{\sigma\tau}_k (co-density matrix) as:
        //      {}^{\Lambda\Lambda} P^{\sigma\tau}_k = ({}^\Lambda \tilde{C})^\sigma_k ({}^\Lambda
        //      \tilde{C}^*)^\tau_k
        // for all k where {}^{\Lambda\Lambda} \tilde{S}_k = 0 and sum together to form {}^{\Lambda\Lambda} P^{\sigma\tau}.
        // This result is added to {}^{\Lambda\Gamma} M^{\sigma\tau, 0} which now
        // contains contributions from the \Lambda, \Lambda co-density matrix and \Lambda \Gamma
        // weighted co-density matrix.
        for &k in zeros {
            let l_tilde_c_occ_k = l_tilde_c_occ.column(k).to_owned();
            let outer = l_tilde_c_occ_k.view().insert_axis(Axis(1)).dot(&l_tilde_c_occ_k.view().insert_axis(Axis(0)));
            ll_m0 += &outer;
        }
        lg_m0 += &ll_m0;
        
        // Calculate {}^{\Lambda\Gamma} P^{\sigma\tau}_k (co-density matrix) as:
        //      {}^{\Lambda\Gamma} P^{\sigma\tau}_k = ({}^\Lambda \tilde{C})^\sigma_k ({}^\Gamma
        //      \tilde{C}^*)^\tau_k
        // for all k where {}^{\Lambda\Gamma} \tilde{S}_k = 0 and sum together to form {}^{\Lambda\Gamma} P^{\sigma\tau}.
        // This result is added to {}^{\Lambda\Gamma} M^{\sigma\tau, 0} which now
        // contains the correct contributions to be:
        //  {}^{\Lambda\Gamma} M^{\sigma\tau, 0} = {}^{\Lambda\Gamma} W^{\sigma\tau} + {}^{\Lambda\Gamma} P^{\sigma\tau} + {}^{\Lambda\Lambda} P^{\sigma\tau} 
        //  as required. Similarly we make {}^{\Lambda\Gamma} M^{\sigma\tau, 1} = {}^{\Lambda\Gamma} P^{\sigma\tau}.  
        for &k in zeros {
            let l_tilde_c_occ_k = l_tilde_c_occ.column(k).to_owned();
            let g_tilde_c_occ_k = g_tilde_c_occ.column(k).to_owned(); 
            let outer = l_tilde_c_occ_k.view().insert_axis(Axis(1)).dot(&g_tilde_c_occ_k.view().insert_axis(Axis(0)));
            lg_m1 += &outer;
            lg_m0 += &outer;
        }

        (lg_m0, lg_m1)
    }

    /// Form the matrices {}^{\Gamma\Lambda} X^{\sigma\tau, m_k}_{ij} and {\Gamma\Lambda} Y_{ij}^{m_k} as:
    ///     {}^{\Gamma\Lambda} X_{ij}^{m_k} = \sum_{\mu\nu\sigma\tau} ({}^\Gamma C^*)_i^\mu S_{\mu\nu} 
    ///     (^{\Gammma\Lambda} M^{m_k})^{\sigma\tau} S_{\mu\nu} (^\Lambda C)_j^\nu.
    ///     {}^{\Gammma\Lambda} Y_{ij}^{0} = {\Gamma\Lambda} X_{ij}^{0} - {}^{\Gammma\Lambda} S_{ij}.
    ///     {}^{\Gammma\Lambda} Y_{ij}^{1} = {\Gamma\Lambda} X_{ij}^{1}.
    /// # Arguments:
    /// - `s_munu`: AO overlap matrix.
    /// - `g_c`: Full AO coefficient matrix of |^\Gamma\Psi\rangle.
    /// - `l_c`: Full AO coefficient matrix of |^\Lambda\Psi\rangle.
    /// - `gl_m`: M matrix{}^{\Gamma\Lambda} M^{\sigma\tau, 0} or  {}^{\Gamma\Lambda} M^{\sigma\tau, 1}. 
    /// - `subtract`: Whether to use m_k = 0 or m_k = 1. 
    /// # Returns
    /// - `(Array2<f64>, Array2<f64>)`: The X and Y matrices.
    fn construct_xy(g_c: &Array2<f64>, l_c: &Array2<f64>, s_munu: &Array2<f64>, gl_m: &Array2<f64>, subtract: bool) -> (Array2<f64>, Array2<f64>) {
        let nbas = g_c.nrows();
        let nmo = g_c.ncols();
        
        // Concatenate coefficient matrices into one.
        let mut lg_c = Array2::<f64>::zeros((nbas, 2 * nmo));
        lg_c.slice_mut(s![.., 0..nmo]).assign(l_c);
        lg_c.slice_mut(s![.., nmo..2 * nmo]).assign(g_c);

        // {}^{\Gamma\Lambda} X_{ij}^{m_k} = \sum_{\mu\nu\sigma\tau} ({}^\Gamma C^*)_i^\mu
        // S_{\mu\nu} (^{\Gammma\Lambda} M^{m_k})^{\sigma\tau} S_{\mu\nu} (^\Lambda C)_j^\nu. Note 
        // that this expression if for computing specifically the {}^{\Gamma\Lambda} quadrant of
        // X_{ij}^{m_k}. By using the concatenated coefficient matrices we compute it all in one go. 
        let sm = s_munu.dot(gl_m);
        let sms = sm.dot(s_munu);
        let x = lg_c.t().dot(&sms).dot(&lg_c);
        
        // {}^{\Gammma\Lambda} Y_{ij}^{0} = {\Gamma\Lambda} X_{ij}^{0} - {}^{\Gammma\Lambda} S_{ij}.
        // {}^{\Gammma\Lambda} Y_{ij}^{1} = {\Gamma\Lambda} X_{ij}^{1}.
        let ymiddle = if subtract {&sms - s_munu} else {sms}; 
        let y = lg_c.t().dot(&ymiddle).dot(&lg_c);

        (x, y)
    }
    
    /// Construct the {}^{\Gamma\Lambda} F_0^{m_k} and {}^{\Lambda\Gamma} F_{ab}^{m_i, m_j}
    /// intermediates required for one-body coupling as:
    ///     {}^{\Lambda\Lambda} F_0^{m_i} = \sum_{pq} {}^\Lambda f_{pq} {\Lambda\Lambda} X_{qp}^{m_i}
    /// where {}^{\Lambda} f_{pq} is the required onebody operator in the MO basis for determinant
    /// \Lambda, and:
    ///     {}^{\Lambda\Gamma} F_{ab}^{m_i, m_j} = \sum_{pq} {\Gamma\Lambda} X_{ap}^{m_i}
    ///     {\Lambda\Lambda} f_{pq} {\Lambda\Lambda} X_{qb}^{m_j},
    /// where the use of X or Y and their quadrants depends on the requested ordering of \Lambda,
    /// \Gamma in {}^{\Lambda\Gamma} F_{ab}^{m_i, m_j}.
    /// # Arguments:
    /// - `g_c`: Full AO coefficient matrix of |^\Gamma\Psi\rangle.
    /// - `l_c`: Full AO coefficient matrix of |^\Lambda\Psi\rangle.
    /// - `h_munu`: One-electron core AO hamiltonian.
    /// - `x`: {}^{\Gamma\Lambda} X_{ij}^{m_k}.
    /// - `y`: {}^{\Gamma\Lambda} Y_{ij}^{m_k}.
    /// # Returns
    /// - `(f64, Array2<f64>)`: Scalar F_0^{m_k} and matrix F_{ab}^{m_i,m_j}.
    pub fn construct_f(l_c: &Array2<f64>, h_munu: &Array2<f64>, x: &Array2<f64>, y: &Array2<f64>) -> (f64, Array2<f64>) {
        let nmo = l_c.ncols();
        let ll_h = l_c.t().dot(h_munu).dot(l_c);
        
        let ll_x = x.slice(s![0..nmo, 0..nmo]).to_owned();  
        let gl_x = x.slice(s![nmo..2 * nmo, 0..nmo]).to_owned();   
        let ll_y = y.slice(s![0..nmo, 0..nmo]).to_owned();          
        let lg_y = y.slice(s![0..nmo, nmo..2 * nmo]).to_owned(); 

        let ll_f0 = einsum_ba_ab_real(&ll_x, &ll_h);

        let ll_f = ll_y.dot(&ll_h).dot(&ll_x);
        let gl_f = gl_x.dot(&ll_h).dot(&ll_x);
        let lg_f = ll_y.dot(&ll_h).dot(&lg_y);
        let gg_f = gl_x.dot(&ll_h).dot(&lg_y);
        
        let mut f = Array2::<f64>::zeros((2 * nmo, 2 * nmo));
        f.slice_mut(s![0..nmo, 0..nmo]).assign(&ll_f);              
        f.slice_mut(s![0..nmo, nmo..2 * nmo]).assign(&lg_f);        
        f.slice_mut(s![nmo..2 * nmo, 0..nmo]).assign(&gl_f);          
        f.slice_mut(s![nmo..2 * nmo, nmo..2 * nmo]).assign(&gg_f); 

        (ll_f0, f)
    }
    
    /// Calculate the Coulomb contraction J^{m_k}_{\mu\nu} required for the two electron
    /// intermediates as:
    ///     J^{m_k}_{\mu\nu} = \sum_{\sigma\tau} ({}^{\Lambda}(\mu\nu|\sigma\tau)) {}^{\Gamma\Lambda} M^{\mu\nu, m_k}
    /// where {}^{\Gamma\Lambda} M^{m_k} is the AO-space M matrix. 
    /// # Arguments:
    /// - `eri`: AO basis ERIs (not-antisymmetrised).
    /// - `m`: {}^{\Gamma\Lambda} M^{m_k} AO-space M matrix. 
    /// # Returns
    /// - `Array2<f64>`: Coulomb contraction matrix.
    fn build_j_coulomb(eri: &Array4<f64>, m: &Array2<f64>) -> Array2<f64> {
        let n = m.nrows();
        let mut j = Array2::<f64>::zeros((n, n));

        j.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(s, mut row)| {
            for t in 0..n {
                let mut acc = 0.0;
                for mu in 0..n {
                    for nu in 0..n {
                        acc += eri[(s, t, mu, nu)] * m[(mu, nu)];
                    }
                }
                row[t] = acc;
            }
        });
        j
    }

    /// Calculate the Coulomb contraction K^{m_k}_{\mu\nu} required for the two electron
    /// intermediates as:
    ///     K^{m_k}_{\mu\nu} = \sum_{\sigma\tau} ({}^{\Lambda}(\mu\sigma|\tau\nu)) {}^{\Gamma\Lambda} M^{\mu\nu, m_k}
    /// where {}^{\Gamma\Lambda} M^{m_k} is the AO-space M matrix. 
    /// # Arguments:
    /// - `eri`: AO basis ERIs (not-antisymmetrised).
    /// - `m`: {}^{\Gamma\Lambda} M^{m_k} AO-space M matrix. 
    /// # Returns
    /// - `Array2<f64>`: Exchange contraction matrix.
    fn build_k_exchange(eri: &Array4<f64>, m: &Array2<f64>) -> Array2<f64> {
        let n = m.nrows();
        let mut k = Array2::<f64>::zeros((n, n));

        k.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(s, mut row)| {
            for t in 0..n {
                let mut acc = 0.0;
                for mu in 0..n {
                    for nu in 0..n {
                        acc += eri[(s, mu, nu, t)] * m[(mu, nu)];
                    }
                }
                row[t] = acc;
            }
        });
        k
    }
}


/// Owning struct for the diff-spin computed intermediates.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct DiffSpinBuild {
    /// Zeroth-order mixed-spin Vab scalar contributions.
    pub vab0: [[f64; 2]; 2],
    /// Mixed-spin Vab[ma0][mb0][mak] intermediates.
    pub vab: [[[Array2<f64>; 2]; 2]; 2],
    /// Zeroth-order mixed-spin Vba scalar contributions.
    pub vba0: [[f64; 2]; 2],
    /// Mixed-spin Vba[mb0][ma0][mbk] intermediates.
    pub vba: [[[Array2<f64>; 2]; 2]; 2],
    /// Mixed-spin IIab[ma0][mak][mb0][mbj] tensors.
    pub iiab: [[[[Array4<f64>; 2]; 2]; 2]; 2],
}

impl DiffSpinBuild {
    /// Constructor for the WicksReferencePair object of DiffSpin which contains the precomputed values
    /// per pair of reference determinants required to evaluate arbitrary excited determinants 
    /// in O(1) time when the excitations are of different spin. As such, the quantities here are
    /// lesser as we only need to evaluate two electron terms with these intermediates.
    /// # Arguments:
    /// - `eri`: Electron repulsion integrals. 
    /// - `h_munu`: AO core Hamiltonian.
    /// - `s_munu`: AO overlap matrix.
    /// - `g_ca`: Spin alpha AO coefficient matrix of |^\Gamma\Psi\rangle.
    /// - `g_cb`: Spin beta AO coefficient matrix of |^\Gamma\Psi\rangle.
    /// - `l_ca`: Spin alpha AO coefficient matrix of |^\Lambda\Psi\rangle.
    /// - `l_cb`: Spin beta AO coefficient matrix of |^\Lambda\Psi\rangle.
    /// - `goa`: Alpha occupancy vector for |^\Gamma\Psi\rangle.
    /// - `gob`: Beta occupancy vector for |^\Gamma\Psi\rangle.
    /// - `loa`: Alpha occupancy vector for |^\Lambda\Psi\rangle.
    /// - `lob`: Beta occupancy vector for |^\Lambda\Psi\rangle.
    /// - `tol`: Tolerance for whether a number is considered zero.
    /// # Returns
    /// - `DiffSpinBuild`: Precomputed different-spin Wick's intermediates for the reference pair.
    pub fn new(eri: &Array4<f64>, s_munu: &Array2<f64>, g_ca: &Array2<f64>, g_cb: &Array2<f64>, l_ca: &Array2<f64>, l_cb: &Array2<f64>, 
           goa: u128, gob: u128, loa: u128, lob: u128, tol: f64) -> Self {
        let nmo = g_ca.ncols();

        let l_ca_occ = occ_coeffs(l_ca, loa);
        let g_ca_occ = occ_coeffs(g_ca, goa);
        let l_cb_occ = occ_coeffs(l_cb, lob);
        let g_cb_occ = occ_coeffs(g_cb, gob);

        // SVD and rotate the occupied orbitals per spin.
        let (tilde_sa_occ, g_tilde_ca_occ, l_tilde_ca_occ, _phase) = SameSpinBuild::perform_ortho_and_svd_and_rotate(s_munu, &l_ca_occ, &g_ca_occ, 1e-20);
        let (tilde_sb_occ, g_tilde_cb_occ, l_tilde_cb_occ, _phase) = SameSpinBuild::perform_ortho_and_svd_and_rotate(s_munu, &l_cb_occ, &g_cb_occ, 1e-20);
        
        // Find indices where zeros occur in {}^{\Gamma\Lambda} \tilde{S} and count them per spin.
        // No longer writing per spin from here onwards, hopefully it is clear.
        let zerosa: Vec<usize> = tilde_sa_occ.iter().enumerate().filter_map(|(k, &sk)| if sk.abs() <= tol {Some(k)} else {None}).collect();
        let zerosb: Vec<usize> = tilde_sb_occ.iter().enumerate().filter_map(|(k, &sk)| if sk.abs() <= tol {Some(k)} else {None}).collect();

        // Construct the {}^{\Gamma\Lambda} M^{\sigma\tau, 0} and {}^{\Gamma\Lambda} M^{\sigma\tau, 1} matrices.
        let (m0a, m1a) = SameSpinBuild::construct_m(&tilde_sa_occ, &l_tilde_ca_occ, &g_tilde_ca_occ, &zerosa, tol);
        let (m0b, m1b) = SameSpinBuild::construct_m(&tilde_sb_occ, &l_tilde_cb_occ, &g_tilde_cb_occ, &zerosb, tol);
        let ma = [&m0a, &m1a];
        let mb = [&m0b, &m1b];
        
        // Construct only the Coulomb contraction of ERIs with  {}^{\Gamma\Lambda} M^{\sigma\tau, m_k}, 
        // {}^{\Gamma\Lambda} J_{\mu\nu}^{m_k}. No exchange here due to differing spins.
        let ja = [SameSpinBuild::build_j_coulomb(eri, ma[0]), SameSpinBuild::build_j_coulomb(eri, ma[1])];
        let jb = [SameSpinBuild::build_j_coulomb(eri, mb[0]), SameSpinBuild::build_j_coulomb(eri, mb[1])];

        // Construct {}^{\Lambda\Gamma} V_{ab,0}^{m_i, m_j} = \sum_{prqs} ({}^{\Lambda}(pr|qs)) X_{sq}^{m_i} {}^{\Lambda\Gamma}. 
        // This can be rewritten (and thus calculated) as V_{ab, 0}^{m_i, m_j} = \sum_{pr} (J_{\mu\nu}^{m_i}) {}^{\Gamma\Lambda} M^{\sigma\tau, m_j}.
        // This is directly analogous to {}^{\Lambda\Gamma} V_0^{m_i, m_j} in the same spin case
        // but with exchange omitted.
        let mut vab0 = [[0.0f64; 2]; 2]; 
        let mut vba0 = [[0.0f64; 2]; 2]; 
        for i in 0..2 {
            for j in 0..2 {
                vab0[i][j] = einsum_ba_ab_real(&ja[i], mb[j]); 
                vba0[j][i] = einsum_ba_ab_real(&jb[j], ma[i]); 
            }
        }
           
        // Calculate the left and right factorisations of the {}^{\Gamma\Lambda} X_{ij}^{m_k} and 
        // {}^{\Gamma\Lambda} Y_{ij}^{m_k} matrices, {}^{\Gamma\Lambda} so as to avoid branching
        // down the line. Again analogous to the same spin case but with spin resolved quantities.
        let (cx_a0, xc_a0) = Self::build_cx_xc(ma[0], s_munu, l_ca, g_ca, 0);
        let (cx_a1, xc_a1) = Self::build_cx_xc(ma[1], s_munu, l_ca, g_ca, 1);
        let (cx_b0, xc_b0) = Self::build_cx_xc(mb[0], s_munu, l_cb, g_cb, 0);
        let (cx_b1, xc_b1) = Self::build_cx_xc(mb[1], s_munu, l_cb, g_cb, 1);
        let cx_a = [&cx_a0, &cx_a1];
        let xc_a = [&xc_a0, &xc_a1];
        let cx_b = [&cx_b0, &cx_b1];
        let xc_b = [&xc_b0, &xc_b1];
        
        // Construct {}^{\Lambda\Gamma} V_{ab}^{m_1, m_2, m_3} intermediates for two electron
        // Hamiltonian matrix elements. These are given as:
        //      {}^{\Lambda\Gamma} V_{ab}^{m_1, m_2, m_3} = \sum_{pq} {}^{\Lambda\Gamma} Y_{ap}^{m_1}
        //      (\sum_{rs} ({}^{\Lambda}(pr|qs)) {}^{\Lambda\Gamma} X_{sr}^{m_2}) {}^{\Lambda\Gamma} X_{qb}^{m_3},
        // where the use of X or Y on the left and righthand sides depends on the ordering of
        // \Lambda and \Gamma. Again using our precomputed quantities we rewrite as:
        //      {}^{\Lambda\Gamma} V_{ab}^{m_1, m_2, m_3} = \sum_{pq} C_{L,ap}^{m_1} (J_{\mu\nu}^{m_2})
        //      C_{R,qb}^{m_3}.
        //  Once more this is analogous to the SameSpin case but with exchange removed. The
        //  similarities here hint at a possible generalisation of the code.
        let mut vab: [[[Array2<f64>; 2]; 2]; 2] = std::array::from_fn(|_| {std::array::from_fn(|_| std::array::from_fn(|_| Array2::<f64>::zeros((2*nmo, 2*nmo))))});
        let mut vba: [[[Array2<f64>; 2]; 2]; 2] = std::array::from_fn(|_| {std::array::from_fn(|_| std::array::from_fn(|_| Array2::<f64>::zeros((2*nmo, 2*nmo))))});
        let combos: Vec<(usize, usize, usize)> =
            (0..2).flat_map(|ma0|
            (0..2).flat_map(move |mb0|
            (0..2).map(move |mk| (ma0, mb0, mk))
        )).collect();
        let vabblocks: Vec<((usize, usize, usize), Array2<f64>)> =
            combos.clone().into_par_iter().map(|(ma0, mb0, mak)| {
                let blk = cx_a[ma0].t().dot(&jb[mb0]).dot(xc_a[mak]);
                ((ma0, mb0, mak), blk)
            }).collect();
        for ((ma0, mb0, mak), blk) in vabblocks {
            vab[ma0][mb0][mak] = blk;
        }
        let vbablocks: Vec<((usize, usize, usize), Array2<f64>)> =
            combos.into_par_iter().map(|(ma0, mb0, mbk)| {
                let blk = cx_b[mb0].t().dot(&ja[ma0]).dot(xc_b[mbk]);
                ((mb0, ma0, mbk), blk)
            }).collect();
        for ((mb0, ma0, mbk), blk) in vbablocks {
            vba[mb0][ma0][mbk] = blk;
        }

        // Construct {}^{\Lambda\Gamma} II_{ab,cd}^{m_1,m_2,m_3,m_4} intermediates for two electron
        // Hamiltonian matrix elements. These are given as:
        //      {}^{\Lambda\Gamma} II_{ab,cd}^{m_1,m_2,m_3,m_4} = \sum_{prqs} ({}^{\Lambda}(pr|qs))
        //      {}^{\Lambda\Gamma} Y_{ap}^{m_1} {}^{\Lambda\Gamma} X_{rb}^{m_2} {}^{\Lambda\Gamma} Y_{cq}^{m_3} {}^{\Lambda\Gamma} X_{sd}^{m_4},
        // where the use of X or Y in each part depends on the ordering of \Lambda and \Gamma. Again using our quantities
        // this may instead be calculated as
        //      {}^{\Lambda\Gamma} II_{ab,cd}^{m_1, m_2, m_3, m_4} = \sum_{\mu\nu\tau\sigma} C_{L,a\mu}^{m_1} C_{R,b\nu}^{m_2}
        //      ((\mu\nu|\tau\sigma)) C_{L,c\tau}^{m_3} C_{R,d\sigma}^{m_4},
        // which is achieved by antisymmetrising the AO integrals and transforming from AO to MO
        // basis. This is unsuprisngly analogous to the 4-index J tensor in SameSpin.
        let mut iiab: [[[[Array4<f64>; 2]; 2]; 2]; 2] = std::array::from_fn(|_| {
            std::array::from_fn(|_| {
                std::array::from_fn(|_| std::array::from_fn(|_| Array4::<f64>::zeros((2 * nmo, 2 * nmo, 2 * nmo, 2 * nmo))))
            })
        });

        let combos: Vec<(usize, usize, usize, usize)> =
            (0..2).flat_map(|mi| (0..2).flat_map(move |mj|
            (0..2).flat_map(move |mk| (0..2).map(move |ml| (mi, mj, mk, ml)))
        )).collect();
        
        type IIabBlock = ((usize, usize, usize, usize), Array4<f64>);

        let blocks: Vec<IIabBlock> = combos.into_par_iter().map(|(ma0, maj, mb0, mbj)| {
            let blk = eri_ao2mo(eri, cx_a[ma0], xc_a[maj], cx_b[mb0], xc_b[mbj]).as_standard_layout().to_owned();
            ((ma0, maj, mb0, mbj), blk)
        }).collect();

        for ((ma0, maj, mb0, mbj), blk) in blocks {
            iiab[ma0][maj][mb0][mbj] = blk;
        }

        Self {vab0, vab, vba0, vba, iiab}
    }
        
    /// Build the left and right factorisations of the {}^{\Gamma\Lambda} X_{ij}^{m_k} and 
    /// {}^{\Gamma\Lambda} Y_{ij}^{m_k} matrices such that our intermediates can be computed more
    /// easily.
    /// # Arguments:
    /// - `m`: {}^{\Gamma\Lambda} M^{m_k} AO-space M matrix. 
    /// - `s`: AO overlap matrix S_{\mu\nu}.
    /// - `cx`: AO coefficient matrix for \Lambda. Should be renamed.
    /// - `cw`: AO coefficient matrix for \Gamma. Should be renamed.
    /// - `i`: Selector for m being 0 or 1.
    /// # Returns
    /// - `(Array2<f64>, Array2<f64>)`: Left and right factorisation matrices.
    fn build_cx_xc(m: &Array2<f64>, s: &Array2<f64>, cx: &Array2<f64>, cw: &Array2<f64>, i: usize) -> (Array2<f64>, Array2<f64>) {
        let nao = cx.nrows();
        let nmo = cx.ncols();
        let mut cx_out = Array2::<f64>::zeros((nao, 2 * nmo));
        let mut xc_out = Array2::<f64>::zeros((nao, 2 * nmo));

        let one_minus_i = (1 - i) as f64;

        let ms = m.dot(s);
        let mts = m.t().dot(s);

        cx_out.slice_mut(s![.., 0..nmo]).assign(&(mts.dot(cx) - &(cx * one_minus_i)));
        xc_out.slice_mut(s![.., 0..nmo]).assign(&ms.dot(cx));

        cx_out.slice_mut(s![.., nmo..2 * nmo]).assign(&mts.dot(cw));
        xc_out.slice_mut(s![.., nmo..2 * nmo]).assign(&(ms.dot(cw) - &(cw * one_minus_i)));

        (cx_out, xc_out)
    }
}
