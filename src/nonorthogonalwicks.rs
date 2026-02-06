// nonorthogonalwicks.rs 
use ndarray::{Array1, Array2, Array4, Axis, s};
use ndarray_linalg::{SVD, Determinant, Solve};

use crate::{ExcitationSpin};

use crate::maths::{einsum_ba_ab_real, eri_ao2mo};
use crate::noci::occ_coeffs;

// Whether a given orbital index belongs to the bra or ket.
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Side {Gamma, Lambda}

#[derive(Debug, Copy, Clone)]
pub enum Type {Hole, Part}

pub type Label = (Side, Type, usize);

pub struct SameSpin {
    // X[mi], Y[mi]
    pub x: [Array2<f64>; 2],
    pub y: [Array2<f64>; 2],

    // F0[mi], F[mi][mj]
    pub f0: [f64; 2],
    pub f: [[Array2<f64>; 2]; 2],

    // V0[mi][mj], V[mi][mj][mk]
    pub v0: [f64; 3],
    pub v: [[[Array2<f64>; 2]; 2]; 2],

    // J[mi][mj][mk][ml]
    pub j: [[[[Array4<f64>; 2]; 2]; 2]; 2],

    pub tilde_s_prod: f64,
    pub phase: f64,
    pub m: usize,
    pub nmo: usize,
}

pub struct DiffSpin {
    pub vab0: [[f64; 2]; 2], // vab0[ma0][mb0]
    pub vab:  [[[Array2<f64>; 2]; 2]; 2], // vab[ma0][mb0][mak]

    pub vba0: [[f64; 2]; 2], // vba0[mb0][ma0]
    pub vba:  [[[Array2<f64>; 2]; 2]; 2], // vba[mb0][ma0][mbk]

    pub iiab: [[[[Array4<f64>; 2]; 2]; 2]; 2], // iiab[ma0][mak][mb0][mbj]
    pub iiba: [[[[Array4<f64>; 2]; 2]; 2]; 2], // iiba[mb0][mbk][ma0][maj]
}

pub struct WicksReferencePair {
    pub aa: SameSpin,
    pub bb: SameSpin,
    pub ab: DiffSpin,
}

impl SameSpin {
    /// Constructor for the WicksReferencePair object which contains the precomputed values
    /// per pair of reference determinants required to evaluate arbitrary excited determinants 
    /// in O(1) time.
    /// # Arguments:
    ///     `eri`: Array4, electron repulsion integrals. 
    ///     `h_munu`: Array2, AO core Hamiltonian.
    ///     `s_munu`: Array2, AO overlap matrix.
    ///     `g_c`: Array2, full AO coefficient matrix of |^\Gamma\Psi\rangle.
    ///     `l_c`: Array2, full AO coefficient matrix of |^\Lambda\Psi\rangle.
    ///     `g_c_occ`: Array2, occupied AO coefficient matrix of |^\Gamma\Psi\rangle.
    ///     `l_c_occ`: Array2, occupied AO coefficient matrix of |^\Lambda\Psi\rangle.
    pub fn new(eri: &Array4<f64>, h_munu: &Array2<f64>, s_munu: &Array2<f64>, g_c: &Array2<f64>, l_c: &Array2<f64>, go: &Array1<f64>, lo: &Array1<f64>) -> Self {
        let tol = 1e-12;
        let nmo = g_c.ncols();
        let nbas = l_c.nrows();

        let mut ccat = Array2::<f64>::zeros((nbas, 2 * nmo));
        ccat.slice_mut(s![.., 0..nmo]).assign(l_c);
        ccat.slice_mut(s![.., nmo..2*nmo]).assign(g_c);

        let l_c_occ = occ_coeffs(l_c, lo);
        let g_c_occ = occ_coeffs(g_c, go);

        // Get occupied MO overlap matrix.
        let s_occ = Self::calculate_mo_overlap_matrix(&l_c_occ, &g_c_occ, s_munu);

        // SVD and rotate the occupied orbitals.
        let (tilde_s_occ, g_tilde_c_occ, l_tilde_c_occ, phase) = Self::perform_svd_and_rotate(&s_occ, &l_c_occ, &g_c_occ);

        // Multiply diagonal non-zero values of {}^{\Gamma\Lambda} \tilde{S} together.
        let tilde_s_prod = tilde_s_occ.iter().filter(|&&x| x.abs() > tol).product::<f64>();

        // Find indices where zeros occur in {}^{\Gamma\Lambda} \tilde{S} and count them.
        let zeros: Vec<usize> = tilde_s_occ.iter().enumerate().filter_map(|(k, &sk)| if sk.abs() <= tol {Some(k)} else {None}).collect();
        let m = zeros.len();

        // Construct the {}^{\Gamma\Lambda} M^{\sigma\tau, 0} and {}^{\Gamma\Lambda} M^{\sigma\tau, 1} matrices.
        let (m0, m1) = Self::construct_m(&tilde_s_occ, &l_tilde_c_occ, &g_tilde_c_occ, &zeros);
        let mao: [Array2<f64>; 2] = [m0, m1];

        // Construct the {}^{\Gamma\Lambda} X_{ij}^{m_k} and {}^{\Gamma\Lambda} Y_{ij}^{m_k} matrices.
        let (x0, y0) = Self::construct_xy(g_c, l_c, s_munu, &mao[0], true);
        let (x1, y1) = Self::construct_xy(g_c, l_c, s_munu, &mao[1], false);

        let x: [Array2<f64>; 2] = [x0, x1];
        let y: [Array2<f64>; 2] = [y0, y1];
        
        let nbas = mao[0].nrows();
        let mut jkao: [Array2<f64>; 2] = [Array2::<f64>::zeros((nbas, nbas)), Array2::<f64>::zeros((nbas, nbas))];
        for mi in 0..2 {
            let j = Self::build_j_coulomb(eri, &mao[mi]);
            let k = Self::build_k_exchange(eri, &mao[mi]);
            jkao[mi] = &j - &k;
        }

        let (f0_0, f00) = Self::construct_f(l_c, h_munu, &x[0], &y[0]);
        let (_, f01) = Self::construct_f(l_c, h_munu, &x[0], &y[1]);
        let (_, f10) = Self::construct_f(l_c, h_munu, &x[1], &y[0]);
        let (f0_1, f11) = Self::construct_f(l_c, h_munu, &x[1], &y[1]);

        let f0: [f64; 2] = [f0_0, f0_1];
        let f: [[Array2<f64>; 2]; 2] = [[f00, f01], [f10, f11]];

        let mut cx: [Array2<f64>; 2] = [Array2::<f64>::zeros((nbas, 2*nmo)), Array2::<f64>::zeros((nbas, 2*nmo))];
        let mut xc: [Array2<f64>; 2] = [Array2::<f64>::zeros((nbas, 2*nmo)), Array2::<f64>::zeros((nbas, 2*nmo))];
        for mi in 0..2 {
            (cx[mi], xc[mi]) = DiffSpin::build_cx_xc(&mao[mi], s_munu, l_c, g_c, mi);
        }

        // Construct v0[mi][mj]
        let mut v0 = [0.0f64; 3];
        v0[0] = einsum_ba_ab_real(&jkao[0], &mao[0]);
        if m > 1 {
            v0[1] = 2.0 * einsum_ba_ab_real(&jkao[0], &mao[1]);
            v0[2] = einsum_ba_ab_real(&jkao[1], &mao[1]);
        } else {
            v0[1] = 0.0;
        }

        // Construct v[mi][mj][mk]
        let mut v: [[[Array2<f64>; 2]; 2]; 2] = std::array::from_fn(|_| {
            std::array::from_fn(|_| {
                std::array::from_fn(|_| Array2::<f64>::zeros((2 * nmo, 2 * nmo)))
            })
        });
        for mi in 0..2 {
            for mj in 0..2 {
                for mk in 0..2 {
                    v[mi][mj][mk] = cx[mi].t().dot(&jkao[mk]).dot(&xc[mj]);
                    //v[mi][mj][mk] = 2.0 * cx[mi].t().dot(&jkao[mk]).dot(&xc[mj]);
                }
            }
        }

        // Construct j[mi][mj][mk][ml]
        let mut j: [[[[Array4<f64>; 2]; 2]; 2]; 2] = std::array::from_fn(|_| {
            std::array::from_fn(|_| {
                std::array::from_fn(|_| {
                    std::array::from_fn(|_| {
                        Array4::<f64>::zeros((2 * nmo, 2 * nmo, 2 * nmo, 2 * nmo))
                    })
                })
            })
        });
        for mi in 0..2 {
            for mj in 0..2 {
                for mk in 0..2 {
                    for ml in 0..2 {
                        let mut blk = eri_ao2mo(eri, &cx[mi], &xc[mj], &cx[mk], &xc[ml]);
                        let ex = blk.view().permuted_axes([0, 2, 1, 3]).to_owned();
                        blk -= &ex;
                        j[mi][mj][mk][ml] = blk;
                    }
                }
            }
        }
        Self {x, y, f0, f, v0, v, j, tilde_s_prod, phase, m, nmo}
    }

    /// Calculate the overlap matrix between two sets of occupied orbitals as:
    ///     {}^{\Gamma\Lambda} S_{ij} = \sum_{\mu\nu} ({}^\Gamma C^*)_i^\mu S_{\mu\nu} ({}^\Lambda C)_j^\nu
    /// # Arguments:
    ///     `g_c_occ`: Array2, occupied coefficients ({}^\Gamma C^*)_i^\mu.
    ///     `l_c_occ`: Array2, occupied coefficients ({}^\Lambda C)_j^\nu 
    ///     `s_munu`: Array2, AO overlap matrix S_{\mu\nu}.
    pub fn calculate_mo_overlap_matrix(l_c_occ: &Array2<f64>, g_c_occ: &Array2<f64>, s_munu: &Array2<f64>) -> Array2<f64> {
        l_c_occ.t().dot(&s_munu.dot(g_c_occ))
    }
    
    /// Perform singular value decomposition on the occupied orbital overlap matrix {}^{\Gamma\Lambda} S_{ij} as:
    ///     {}^{\Gamma\Lambda} \mathbf{S} = \mathbf{U} {}^{\Gamma\Lambda} \mathbf{\tilde{S}} \mathbf{V}^\dagger,
    /// and rotate the occupied coefficients:
    ///     |{}^\Gamma \Psi_i\rangle = \sum_{\mu} {}^\Gamma c_i^\mu U_{ij} |\phi_\mu \rangle.
    ///     |{}^\Lambda \Psi_j\rangle = \sum_{\nu} {}^\Lambda c_j^\nu V_{ij} |\phi_\nu \rangle.
    /// # Arguments:
    ///     `gl_s`: Array2, occupied coefficient matrix {}^{\Gamma\Lambda} S_{ij}.
    ///     `g_c_occ`: Array2, occupied coefficients ({}^\Gamma C^*)_i^\mu.
    ///     `l_c_occ`: Array2, occupied coefficients ({}^\Lambda C)_j^\nu.
    pub fn perform_svd_and_rotate(lg_s: &Array2<f64>, l_c_occ: &Array2<f64>, g_c_occ: &Array2<f64>) -> (Array1<f64>, Array2<f64>, Array2<f64>, f64){
        // SVD.
        let (u, gl_tilde_s, v_dag) = lg_s.svd(true, true).unwrap();
        let u = u.unwrap();
        let v = v_dag.unwrap().t().to_owned();
        
        // Rotate MOs.
        //let g_tilde_c = g_c_occ.dot(&u);
        //let l_tilde_c = l_c_occ.dot(&v);
        let l_tilde_c = l_c_occ.dot(&u);
        let g_tilde_c = g_c_occ.dot(&v);
        
        // Calculate phase associated with rotation.
        let det_u = u.det().unwrap();
        let det_v = v.det().unwrap();
        let ph = det_u * det_v;
        
        (gl_tilde_s, g_tilde_c, l_tilde_c, ph)
    }
    
    /// Form the matrices {}^{\Gamma\Lambda} M^{\sigma\tau, 0} and{}^{\Gamma\Lambda} M^{\sigma\tau, 1} as:
    ///     {}^{\Gamma\Lambda} M^{\sigma\tau, 0} = {}^{\Gamma\Lambda} W^{\sigma\tau} + {}^{\Gamma\Lambda} P^{\sigma\tau} + {}^{\Gamma\Gamma} P^{\sigma\tau}
    ///     {}^{\Gamma\Lambda} M^{\sigma\tau, 1} = {}^{\Gamma\Lambda} P^{\sigma\tau}.
    /// The components {}^{\Gamma\Lambda} W^{\sigma\tau}, {}^{\Gamma\Lambda} P^{\sigma\tau}, 
    /// {}^{\Gamma\Gamma} P^{\sigma\tau} are constructed sequentially and added into the correct
    /// matrix.
    /// # Arguments:
    ///     `gl_tilde_s`: Array1, vector of diagonal single values of {}^{\Gamma\Lambda} \tilde{S}.
    ///     `g_c_tilde_occ`: Array2, rotated occupied coefficients ({}^\Gamma \tilde{C}^*)_i^\mu.
    ///     `l_c_tilde_occ`: Array2, rotated occupied coefficients ({}^\Lambda \tilde{C})_j^\nu.
    ///     `zeros`: Vec<usize>, indices where zeros occur in {}^{\Gamma\Lambda} \tilde{S}. 
    pub fn construct_m(lg_tilde_s: &Array1<f64>, g_tilde_c_occ: &Array2<f64>, l_tilde_c_occ: &Array2<f64>, zeros: &Vec<usize>) -> (Array2<f64>, Array2<f64>) {
        let tol = 1e-12;
        let nbas = g_tilde_c_occ.nrows();
        let nocc = g_tilde_c_occ.ncols();
        
        // Calculate {}^{\Gamma\Lambda} W^{\sigma\tau} (weighted co-density matrix) as:
        //      {}^{\Gamma\Lambda} W^{\sigma\tau} = \sum_i (^\Gamma \tilde{C})^\sigma_i (1 /
        //      {}^{\Gamma\Lambda} \tilde{S}_i) (^\Lambda \tilde{C}^*)^\tau_i
        //  for all i where {}^{\Gamma\Lambda} \tilde{S}_i != 0. This result is stored in
        //  {}^{\Gamma\Lambda} M^{\sigma\tau, 0}, where the zero indicates this quantity will be
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
        
        // Calculate {}^{\Gamma\Gamma} P^{\sigma\tau}_k (co-density matrix) as:
        //      {}^{\Gamma\Gamma} P^{\sigma\tau}_k = ({}^\Gamma \tilde{C})^\sigma_k ({}^\Gamma
        //      \tilde{C}^*)^\tau_k
        // for all k where {}^{\Gamma\Gamma} \tilde{S}_k = 0 and sum together to form {}^{\Gamma\Gamma} P^{\sigma\tau}.
        // This result is added to {}^{\Gamma\Lambda} M^{\sigma\tau, 0} which now
        // contains contributions from the \Gamma, \Gamma co-density matrix and \Gamma \Lambda
        // weighted co-density matrix.
        for &k in zeros {
            let l_tilde_c_occ_k = l_tilde_c_occ.column(k).to_owned();
            let outer = l_tilde_c_occ_k.view().insert_axis(Axis(1)).dot(&l_tilde_c_occ_k.view().insert_axis(Axis(0)));
            ll_m0 += &outer;
        }
        lg_m0 += &ll_m0;
        
        // Calculate {}^{\Gamma\Lambda} P^{\sigma\tau}_k (co-density matrix) as:
        //      {}^{\Gamma\Lambda} P^{\sigma\tau}_k = ({}^\Gamma \tilde{C})^\sigma_k ({}^\Lambda
        //      \tilde{C}^*)^\tau_k
        // for all k where {}^{\Gamma\Lambda} \tilde{S}_k = 0 and sum together to form {}^{\Gamma\Lambda} P^{\sigma\tau}.
        // This result is added to {}^{\Gamma\Lambda} M^{\sigma\tau, 0} which now
        // contains the correct contributions to be:
        //  {}^{\Gamma\Lambda} M^{\sigma\tau, 0} = {}^{\Gamma\Lambda} W^{\sigma\tau} + {}^{\Gamma\Lambda} P^{\sigma\tau} + {}^{\Gamma\Gamma} P^{\sigma\tau} 
        //  as required. Similarly we make {}^{\Gamma\Lambda} M^{\sigma\tau, 1} = {}^{\Gamma\Lambda} P^{\sigma\tau}.  
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
    ///     `s_munu`: Array2, AO overlap matrix.
    ///     `g_c`: Array2, full AO coefficient matrix of |^\Gamma\Psi\rangle.
    ///     `l_c`: Array2, full AO coefficient matrix of |^\Lambda\Psi\rangle.
    ///     `gl_m`: Array2, M matrix{}^{\Gamma\Lambda} M^{\sigma\tau, 0} or  {}^{\Gamma\Lambda} M^{\sigma\tau, 1}. 
    ///     `subtract`: bool, whether to use m_k = 0 or m_k = 1. 
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

    fn construct_f(l_c: &Array2<f64>, h_munu: &Array2<f64>, x: &Array2<f64>, y: &Array2<f64>) -> (f64, Array2<f64>) {
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

    fn build_j_coulomb(eri: &Array4<f64>, m: &Array2<f64>) -> Array2<f64> {
        let n = m.nrows();
        let mut j = Array2::<f64>::zeros((n, n));
        for s in 0..n {
            for t in 0..n {
                let mut acc = 0.0;
                for mu in 0..n {
                    for nu in 0..n {
                        acc += eri[(s, t, mu, nu)] * m[(mu, nu)];
                        
                    }
                }
                j[(s, t)] = acc;
            }
        }
        j
    }

    fn build_k_exchange(eri: &Array4<f64>, m: &Array2<f64>) -> Array2<f64> {
        let n = m.nrows();
        let mut k = Array2::<f64>::zeros((n, n));
        for s in 0..n {
            for t in 0..n {
                let mut acc = 0.0;
                for mu in 0..n {
                    for nu in 0..n {
                        acc += eri[(s, mu, nu, t)] * m[(mu, nu)];
                    }
                }
                k[(s, t)] = acc;
            }
        }
        k
    }

}

impl DiffSpin {
    pub fn new(eri: &Array4<f64>, s_munu: &Array2<f64>, g_ca: &Array2<f64>, g_cb: &Array2<f64>, l_ca: &Array2<f64>, l_cb: &Array2<f64>, 
           goa: &Array1<f64>, gob: &Array1<f64>, loa: &Array1<f64>, lob: &Array1<f64>) -> Self {
        let tol = 1e-12;
        let nmo = g_ca.ncols();

        let l_ca_occ = occ_coeffs(l_ca, loa);
        let g_ca_occ = occ_coeffs(g_ca, goa);
        let l_cb_occ = occ_coeffs(l_cb, lob);
        let g_cb_occ = occ_coeffs(g_cb, gob);
        
        let sa_occ = SameSpin::calculate_mo_overlap_matrix(&l_ca_occ, &g_ca_occ, s_munu);
        let sb_occ = SameSpin::calculate_mo_overlap_matrix(&l_cb_occ, &g_cb_occ, s_munu);

        let (tilde_sa_occ, g_tilde_ca_occ, l_tilde_ca_occ, _phase) = SameSpin::perform_svd_and_rotate(&sa_occ, &l_ca_occ, &g_ca_occ);
        let (tilde_sb_occ, g_tilde_cb_occ, l_tilde_cb_occ, _phase) = SameSpin::perform_svd_and_rotate(&sb_occ, &l_cb_occ, &g_cb_occ);

        let zerosa: Vec<usize> = tilde_sa_occ.iter().enumerate().filter_map(|(k, &sk)| if sk.abs() <= tol {Some(k)} else {None}).collect();
        let zerosb: Vec<usize> = tilde_sb_occ.iter().enumerate().filter_map(|(k, &sk)| if sk.abs() <= tol {Some(k)} else {None}).collect();

        let (m0a, m1a) = SameSpin::construct_m(&tilde_sa_occ, &l_tilde_ca_occ, &g_tilde_ca_occ, &zerosa);
        let (m0b, m1b) = SameSpin::construct_m(&tilde_sb_occ, &l_tilde_cb_occ, &g_tilde_cb_occ, &zerosb);

        let ma = [&m0a, &m1a];
        let mb = [&m0b, &m1b];

        let ja = [SameSpin::build_j_coulomb(eri, &m0a), SameSpin::build_j_coulomb(eri, &m1a)];
        let jb = [SameSpin::build_j_coulomb(eri, &m0b), SameSpin::build_j_coulomb(eri, &m1b)];
        
        let mut vab0 = [[0.0f64; 2]; 2]; 
        let mut vba0 = [[0.0f64; 2]; 2]; 
        for i in 0..2 {
            for j in 0..2 {
                vab0[i][j] = einsum_ba_ab_real(&ja[i], mb[j]); 
                vba0[j][i] = einsum_ba_ab_real(&jb[j], ma[i]); 
            }
        }

        let (cx_a0, xc_a0) = Self::build_cx_xc(&m0a, s_munu, l_ca, g_ca, 0);
        let (cx_a1, xc_a1) = Self::build_cx_xc(&m1a, s_munu, l_ca, g_ca, 1);
        let (cx_b0, xc_b0) = Self::build_cx_xc(&m0b, s_munu, l_cb, g_cb, 0);
        let (cx_b1, xc_b1) = Self::build_cx_xc(&m1b, s_munu, l_cb, g_cb, 1);

        let cx_a = [&cx_a0, &cx_a1];
        let xc_a = [&xc_a0, &xc_a1];
        let cx_b = [&cx_b0, &cx_b1];
        let xc_b = [&xc_b0, &xc_b1];

        let mut vab: [[[Array2<f64>; 2]; 2]; 2] = std::array::from_fn(|_| {std::array::from_fn(|_| std::array::from_fn(|_| Array2::<f64>::zeros((2*nmo, 2*nmo))))});
        let mut vba: [[[Array2<f64>; 2]; 2]; 2] = std::array::from_fn(|_| {std::array::from_fn(|_| std::array::from_fn(|_| Array2::<f64>::zeros((2*nmo, 2*nmo))))});
        for ma0 in 0..2 {
            for mb0 in 0..2 {
                for mak in 0..2 {
                    vab[ma0][mb0][mak] = cx_a[ma0].t().dot(&jb[mb0].dot(xc_a[mak]));
                }
                for mbk in 0..2 {
                    vba[mb0][ma0][mbk] = cx_b[mb0].t().dot(&ja[ma0].dot(xc_b[mbk]));
                }
            }
        }
        
        let mut iiab: [[[[Array4<f64>; 2]; 2]; 2]; 2] = std::array::from_fn(|_| {
            std::array::from_fn(|_| {
                std::array::from_fn(|_| std::array::from_fn(|_| Array4::<f64>::zeros((2 * nmo, 2 * nmo, 2 * nmo, 2 * nmo))))
            })
        });
        let mut iiba: [[[[Array4<f64>; 2]; 2]; 2]; 2] = std::array::from_fn(|_| {
            std::array::from_fn(|_| {
                std::array::from_fn(|_| std::array::from_fn(|_| Array4::<f64>::zeros((2 * nmo, 2 * nmo, 2 * nmo, 2 * nmo))))
            })
        });
        for (ma0, cxa0) in cx_a.iter().enumerate() {
            for (maj, xca_j) in xc_a.iter().enumerate() {
                for (mb0, cxb0) in cx_b.iter().enumerate() {
                    for (mbj, xcb_j) in xc_b.iter().enumerate() {
                        iiab[ma0][maj][mb0][mbj] = eri_ao2mo(eri, cxa0, xca_j, cxb0, xcb_j);
                    }
                }
            }
        }
        for (mb0, _cxb0) in cx_b.iter().enumerate() {
            for (mbk, _xcb_k) in xc_b.iter().enumerate() {
                for (ma0, _cxa0) in cx_a.iter().enumerate() {
                    for (maj, _xca_j) in xc_a.iter().enumerate() {
                        iiba[mb0][mbk][ma0][maj] = iiab[ma0][maj][mb0][mbk].view().permuted_axes([2, 3, 0, 1]).to_owned();
                    }
                }
            }
        }

        Self {vab0, vab, vba0, vba, iiab, iiba}
    }

    fn build_cx_xc(m: &Array2<f64>, s: &Array2<f64>, cx: &Array2<f64>, cw: &Array2<f64>, i: usize) -> (Array2<f64>, Array2<f64>) {
        let nao = cx.nrows();
        let nmo = cx.ncols();
        let mut cx_out = Array2::<f64>::zeros((nao, 2*nmo));
        let mut xc_out = Array2::<f64>::zeros((nao, 2*nmo));

        let one_minus_i = (1 - i) as f64;

        let ms = m.dot(s);
        let mts = m.t().dot(s);

        cx_out.slice_mut(s![.., 0..nmo]).assign(&(mts.dot(cx) - &(cx * one_minus_i)));
        xc_out.slice_mut(s![.., 0..nmo]).assign(&ms.dot(cx));

        cx_out.slice_mut(s![.., nmo..2*nmo]).assign(&mts.dot(cw));
        xc_out.slice_mut(s![.., nmo..2*nmo]).assign(&(ms.dot(cw) - &(cw * one_minus_i)));

        (cx_out, xc_out)
    }
}

/// Given the orbitals excited from and to  relative to the reference determinants, construct the 
/// index labels for the L by L contraction matrix. The contraction matrix is blocked by quadrant
/// as: [\Lambda\Lambda & \Lambda\Gamma \\ \Gamma\Lambda & \Gamma\Gamma]. The rows correspond to
/// operations that act as creation operators after normal ordering to the bra (\Lambda), whilst
/// columns correspond to operations that are annhilations after normal ordering to the bra.
/// # Arguments:
///     `l_ex`: Excitation, left excitations relative to reference \Lambda in the bra.
///     `g_ex`: Excitation, right excitations relative to reference \Gamma in the ket.
fn construct_determinant_lables(l_ex: &ExcitationSpin, g_ex: &ExcitationSpin) -> (Vec<Label>, Vec<Label>) {
    // Integer L is the total number of combined excitations from the bra (\Lambda) and ket (\Gamma).
    let l = l_ex.holes.len() + g_ex.holes.len();
    let nl = l_ex.holes.len();
    let ng = g_ex.holes.len();

    let mut rows = Vec::with_capacity(l);
    let mut cols = Vec:: with_capacity(l);

    if nl == 0 && ng == 0 {return (rows, cols);}
    
    // If number of holes in \Lambda determinant is non-zero and number of holes in \Gamma
    // determinant is zero we only need to consider \Lambda determinant. The ordering is:
    //  rows = [\Lambda parts], cols = [\Lambda holes].
    if nl > 0 && ng == 0 {
        for &a in &l_ex.parts {rows.push((Side::Lambda, Type::Part, a));}
        for &i in &l_ex.holes {cols.push((Side::Lambda, Type::Hole, i));}
        return (rows, cols);
    }
    
    // If number of holes in \Gamma determinant is non-zero and number of holes in \Lambda
    // determinant is zero we only need to consider \Gamma determinant. The ordering is:
    //  rows = [\Gamma holes], cols = [\Gamma parts].
    if nl == 0 && ng > 0 {
        for &i in &g_ex.holes {rows.push((Side::Gamma, Type::Hole, i));}
        for &a in &g_ex.parts {cols.push((Side::Gamma, Type::Part, a));}
        return (rows, cols);
    }

    // If both are non-zero we have the ordering:
    //  rows = [\Lambda parts ; \Gamma holes], cols = [\Lambda holes ; \Gamma parts]
    for &a in &l_ex.parts {rows.push((Side::Lambda, Type::Part, a));}
    for &i in &g_ex.holes {rows.push((Side::Gamma, Type::Hole, i));}
    for &i in &l_ex.holes {cols.push((Side::Lambda, Type::Hole, i));}
    for &a in &g_ex.parts {cols.push((Side::Gamma, Type::Part, a));}
    
    (rows, cols)
}

/// Given the total excitation rank L and the number of zero-overlap orbital couplings find all the
/// possible combinations of (m_1,..., m_L) \in {0, 1}^L which satisfy
///     m_1 + .... + m_L = m.
/// The return is some type with Iterator implemented which is not specified due to long name.
/// # Arguments:
///     `l`: usize, total excitation rank.
///     `m`: usize, number of zero-overlap orbital couplings.
fn iter_m_combinations(l: usize, m: usize) -> impl Iterator<Item = u64> {
    // If the total excitation rank is greater than 64 the below calculation will not work.
    assert!(l < 64);
    // Left shift 000....0001 (64) by L positions to get 2^L combinations of (m_1,..., m_L) \in {0, 1}^L.
    let max = 1u64 << l;
    // Iterate over all 2^L bitstrings and keep only those with m bits set to 1 as required by 
    //  m_1 + ..... + m_L = m.
    let mut bitstrings = Vec::new();
    for bitstring in 0..max {
        let ones: usize = bitstring.count_ones() as usize;
        if ones == m {bitstrings.push(bitstring)}
    }
    bitstrings.into_iter()
}

/// Convert the generated contraction determinant labels into position indices.
/// # Arguments:
///     `side`: Side, whether the label is from the bra (\Lambda) or ket (\Gamma).
///     `p`: usize, orbital index in MO basis.
///     `nmo`: usize, number of MOs.
fn label_to_idx(side: Side, p: usize, nmo: usize) -> usize {
    match side {
        Side::Lambda => p,        // First block.
        Side::Gamma  => nmo + p,  // Second block.
    }
}

pub fn lg_overlap(w: &SameSpin, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin) -> f64 {

    let (rows_label, cols_label) = construct_determinant_lables(l_ex, g_ex);

    // If the total excitation rank is less than the number of zero-singular values in
    // {}^{\Gamma\Lambda} \tilde{S} the overlap element is zero.
    let l = rows_label.len();
    if w.m > l {return 0.0;}
    
    // Convert the contraction determinant labels into actual indices.
    let rows: Vec<usize> = rows_label.into_iter().map(|(s, _t, i)| label_to_idx(s, i, w.nmo)).collect();
    let cols: Vec<usize> = cols_label.into_iter().map(|(s, _t, i)| label_to_idx(s, i, w.nmo)).collect();

    let mut acc = 0.0;
    
    // Iterate over all possible distributions of zeros amongst the columns.
    for bitstring in iter_m_combinations(l, w.m) {
        // Build two full contraction determinants each having exclusively (X0, Y0) or (X1, Y1). 
        let mut det0 = Array2::<f64>::zeros((l, l)); 
        let mut det1 = Array2::<f64>::zeros((l, l));

        for a in 0..l {
            for b in 0..l {
                let ra = rows[a];
                let cb = cols[b];
                // Lower triangle and diagonal of the determinant uses X0 or X1.
                if a >= b {
                    det0[(a, b)] = w.x[0][(ra, cb)];
                    det1[(a, b)] = w.x[1][(ra, cb)];
                // Upper triangle uses Y0 or Y1.
                } else {
                    det0[(a, b)] = w.y[0][(ra, cb)];
                    det1[(a, b)] = w.y[1][(ra, cb)];
                }
            }
        }
        
        // Iterate over columns and decide to keep the 0 determinant or exchange a column for the
        // one determinant version.
        let mut det = det0.clone();
        for b in 0..l {
            // Extract the m_k for this coulmn by moving the 1 or 0 to the rightmost point of the
            // bitstring, read it (&1) and convert into boolean (== 1). 
            let useone = ((bitstring >> b) & 1) == 1;
            // If we are  using the one determinant, exchange all entries in a column.
            if useone {
                for a in 0..l {
                    det[(a, b)] = det1[(a, b)];
                }
            }
        }
        acc += det.det().unwrap();
    }
    w.phase * w.tilde_s_prod * acc
}

pub fn lg_h1(w: &SameSpin, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin) -> f64 {
    
    let (rows_label, cols_label) = construct_determinant_lables(l_ex, g_ex);
    
    // If the total excitation rank + 1 is less than the number of zero-singular values in
    // {}^{\Gamma\Lambda} \tilde{S} the one electron matrix element is zero.
    let l = rows_label.len();
    if w.m > (l + 1) {return 0.0;}

    let mut acc = 0.0;

    // Convert the contraction determinant labels into actual indices.
    let rows: Vec<usize> = rows_label.into_iter().map(|(s,_t,i)| label_to_idx(s, i, w.nmo)).collect();
    let cols: Vec<usize> = cols_label.into_iter().map(|(s,_t,i)| label_to_idx(s, i, w.nmo)).collect();

    // Iterate over all possible distributions of zeros amongst the columns.
    for bitstring in iter_m_combinations(l + 1, w.m) {
        let m1_one = (bitstring & 1) == 1;

        let mut det0 = Array2::<f64>::zeros((l, l));
        let mut det1 = Array2::<f64>::zeros((l, l));

        for a in 0..l {
            for b in 0..l {
                let ra = rows[a];
                let cb = cols[b]; 

                if a >= b {
                    det0[(a, b)] = w.x[0][(ra, cb)];
                    det1[(a, b)] = w.x[1][(ra, cb)];
                } else {
                    det0[(a, b)] = w.y[0][(ra, cb)];
                    det1[(a, b)] = w.y[1][(ra, cb)];
                }
            }
        }

        let mut det = det0.clone();
        for b in 0..l {
            let mb_one = ((bitstring >> (b + 1)) & 1) == 1;
            if mb_one {
                for a in 0..l {
                    det[(a, b)] = det1[(a, b)]
                }
            }
        }
    
        // First term in the sum of Eqn 23.
        let mut contrib = det.det().unwrap() * w.f0[m1_one as usize];

        for b in 0..l {
            let mb_one = ((bitstring >> (b + 1)) & 1) == 1; 
            let f = &w.f[m1_one as usize][mb_one as usize];

            let mut det_f = det.clone(); 
            for a in 0..l {
                let ra = rows[a];
                let cb = cols[b];
                det_f[(a, b)] = f[(ra, cb)]
            }
            contrib -= det_f.det().unwrap();
        }
        acc += contrib;
    }
    w.phase * w.tilde_s_prod * acc
}

pub fn lg_h2_same(w: &SameSpin, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin) -> f64 {
    
    let (rows_label, cols_label) = construct_determinant_lables(l_ex, g_ex);
    let l = rows_label.len();
    if w.m > (l + 2) {return 0.0;}

    let rows: Vec<usize> = rows_label.iter().map(|(s,_t,i)| label_to_idx(*s, *i, w.nmo)).collect();
    let cols: Vec<usize> = cols_label.iter().map(|(s,_t,i)| label_to_idx(*s, *i, w.nmo)).collect();

    let det0 = build_d(&w.x[0], &w.y[0], &rows, &cols);
    let det1 = build_d(&w.x[1], &w.y[1], &rows, &cols);

    let mut acc = 0.0;

    for bits in iter_m_combinations(l + 2, w.m) {
        let m1 = (bits & 1) == 1;
        let m2 = ((bits >> 1) & 1) == 1;
        let m_col = |k: usize| ((bits >> (k + 2)) & 1) == 1;
        let ind: u64 = bits >> 2;

        let det_mix = mix_columns(&det0, &det1, ind);
        let Some((det_det, adjt_det)) = adjugate_transpose(&det_mix) else {continue;};

        let mut contrib = 0.0f64;

        // Equation 30.
        let q = (m1 as usize) + (m2 as usize);
        let x = w.v0[q] * det_det;
        contrib += x;
        
        for k in 0..l {

            let mk = m_col(k);
            
            let vcol = &w.v[m1 as usize][m2 as usize][mk as usize];
            let mut v1 = Array1::<f64>::zeros(l);
            for r in 0..l {
                v1[r] = vcol[(rows[r], cols[k])];
            }

            let v2 = det_mix.column(k).to_owned();
            let a = adjt_det.column(k).to_owned();
            
            let x = det_det + (&v1 - &v2).dot(&a);
            contrib -= 2.0 * x;
        }
        
        for i in 0..l {
            for j in 0..l {
                let phase = if ((i + j) & 1) == 0 { 1.0 } else { -1.0 };

                let det_mix2 = minor(&det_mix, i, j);
                let Some((det_det2, adjt_det2)) = adjugate_transpose(&det_mix2) else {continue;};

                let ri_fixed = rows[i];
                let cj_fixed = cols[j];

                let mj = m_col(j);

                for k2 in 0..(l - 1) {
                    let k_full = if k2 < j { k2 } else { k2 + 1 };

                    let mk = m_col(k_full);

                    let j4 = &w.j[m1 as usize][m2 as usize][mk as usize][mj as usize];
                    let jslice_full = slice_ii(j4, &rows, &cols, ri_fixed, cj_fixed, true);

                    let jslice2 = minor(&jslice_full, i, j);

                    let v1 = jslice2.column(k2).to_owned();

                    let v2 = det_mix2.column(k2).to_owned();
                    let a  = adjt_det2.column(k2).to_owned();

                    let det_repl = det_det2 + (&v1 - &v2).dot(&a);

                    let x = 1.0 * phase * det_repl;
                    contrib += x;
                }
            }
        }
        acc += contrib;
    }
    w.phase * w.tilde_s_prod * acc
}

pub fn lg_h2_diff(w: &WicksReferencePair, l_ex_a: &ExcitationSpin, g_ex_a: &ExcitationSpin, l_ex_b: &ExcitationSpin, g_ex_b: &ExcitationSpin) -> f64 {

    let (rows_a_lab, cols_a_lab) = construct_determinant_lables(l_ex_a, g_ex_a);
    let (rows_b_lab, cols_b_lab) = construct_determinant_lables(l_ex_b, g_ex_b);
    
    let rows_a: Vec<usize> = rows_a_lab.iter().map(|&lbl| lbl_to_idx(lbl, w.aa.nmo)).collect();
    let cols_a: Vec<usize> = cols_a_lab.iter().map(|&lbl| lbl_to_idx(lbl, w.aa.nmo)).collect();
    let rows_b: Vec<usize> = rows_b_lab.iter().map(|&lbl| lbl_to_idx(lbl, w.bb.nmo)).collect();
    let cols_b: Vec<usize> = cols_b_lab.iter().map(|&lbl| lbl_to_idx(lbl, w.bb.nmo)).collect();

    let la = rows_a_lab.len();
    let lb = rows_b_lab.len();

    if w.aa.m > la + 1 {return 0.0;}
    if w.bb.m > lb + 1 {return 0.0;}

    let deta0  = build_d(&w.aa.x[0], &w.aa.y[0], &rows_a, &cols_a);    
    let deta1 = build_d(&w.aa.x[1], &w.aa.y[1], &rows_a, &cols_a);
    let detb0  = build_d(&w.bb.x[0], &w.bb.y[0], &rows_b, &cols_b);
    let detb1 = build_d(&w.bb.x[1], &w.bb.y[1], &rows_b, &cols_b);
    
    let mut acc = 0.0;

    for bits_a in iter_m_combinations(la + 1, w.aa.m) {
        let ma0 = (bits_a & 1) == 1;
        let ma_col = |k: usize| ((bits_a >> (k + 1)) & 1) == 1;
        let inda: u64 = bits_a >> 1;

        let deta_mix = mix_columns(&deta0, &deta1, inda);
        let Some((det_deta, adjt_deta)) = adjugate_transpose(&deta_mix) else {continue;};

        for bits_b in iter_m_combinations(lb + 1, w.bb.m) {
            let mb0 = (bits_b & 1) == 1;
            let mb_col = |k: usize| ((bits_b >> (k + 1)) & 1) == 1;
            let indb: u64 = bits_b >> 1;

            let detb_mix = mix_columns(&detb0, &detb1, indb);
            let Some((det_detb, adjt_detb)) = adjugate_transpose(&detb_mix) else {continue;};

            let mut contrib = 0.0f64;

            // Equation 30.
            let x = w.ab.vab0[ma0 as usize][mb0 as usize] * det_deta * det_detb;
            contrib += x;
        
            for k in 0..la {

                let mak = ma_col(k);

                let vcol = &w.ab.vab[ma0 as usize][mb0 as usize][mak as usize];

                let mut v1 = Array1::<f64>::zeros(la);
                for r in 0..la {
                    v1[r] = vcol[(rows_a[r], cols_a[k])];
                }

                let v2 = deta_mix.column(k).to_owned();
                let a = adjt_deta.column(k).to_owned();
                
                let x = (det_deta + (&v1 - &v2).dot(&a)) * det_detb;
                contrib -= x;
            }

            // Equation 34b.
            for k in 0..lb {

                let mbk = mb_col(k);

                let vcol = &w.ab.vba[mb0 as usize][ma0 as usize][mbk as usize];

                let mut v1 = Array1::<f64>::zeros(lb);
                for r in 0..lb {
                    v1[r] = vcol[(rows_b[r], cols_b[k])];
                }

                let v2 = detb_mix.column(k).to_owned();
                let a = adjt_detb.column(k).to_owned();

                let x = (det_detb + (&v1 - &v2).dot(&a)) * det_deta;
                contrib -= x;
            }
            
            // Equation 38b.
            for (i, &ra) in rows_a.iter().enumerate() {
                for (j, &ca) in cols_a.iter().enumerate() {
                    let phase = if ((i + j) & 1) == 0 {1.0} else {-1.0};

                    let deta0_minor = minor(&deta0, i, j);
                    let deta1_minor = minor(&deta1, i, j);

                    let inda_cols: u64 = bits_a >> 1;        
                    let inda2: u64 = remove_bit(inda_cols, j); 

                    let deta_minor_mix = mix_columns(&deta0_minor, &deta1_minor, inda2);
                    let det_deta_minor_mix = deta_minor_mix.det().unwrap();

                    for k in 0..lb {

                        let mbk = mb_col(k);
                        let ma1 = ma_col(j);

                        let iib = &w.ab.iiba[mb0 as usize][mbk as usize][ma0 as usize][ma1 as usize];
                        let iisliceb = slice_ii(iib, &rows_b, &cols_b, ra, ca, true);

                        let mut v1 = Array1::<f64>::zeros(lb);
                        for r in 0..lb {
                            v1[r] = iisliceb[(r, k)];
                        }

                        let v2 = detb_mix.column(k).to_owned();
                        let a = adjt_detb.column(k).to_owned();

                        let x = 0.5 * phase * (det_detb + (&v1 - &v2).dot(&a)) * det_deta_minor_mix;
                        contrib += x;
                    }
                }
            }

            // Equation 38a.
            for (i, &rb) in rows_b.iter().enumerate() {
                for (j, &cb) in cols_b.iter().enumerate() {
                    let phase = if ((i + j) & 1) == 0 { 1.0 } else { -1.0 };

                    let detb0_minor = minor(&detb0, i, j);
                    let detb1_minor = minor(&detb1, i, j);

                    let indb_cols: u64 = bits_b >> 1;          
                    let indb2: u64 = remove_bit(indb_cols, j);

                    let detb_minor_mix = mix_columns(&detb0_minor, &detb1_minor, indb2);
                    let det_detb_minor_mix = detb_minor_mix.det().unwrap();
                    
                    for k in 0..la {

                        let mak = ma_col(k);
                        let mb1 = mb_col(j);

                        let iia = &w.ab.iiab[ma0 as usize][mak as usize][mb0 as usize][mb1 as usize];
                        let iislicea = slice_ii(iia, &rows_a, &cols_a, rb, cb, true);
                        
                        let mut v1 = Array1::<f64>::zeros(la);
                        for r in 0..la {
                            v1[r] = iislicea[(r, k)];
                        }

                        let v2 = deta_mix.column(k).to_owned();
                        let a = adjt_deta.column(k).to_owned();

                        let x = 0.5 * phase * (det_deta + (&v1 - &v2).dot(&a)) * det_detb_minor_mix;
                        contrib += x;
                    }
                }
            }
            acc += contrib;
        }
    }

    (w.aa.phase * w.aa.tilde_s_prod) * (w.bb.phase * w.bb.tilde_s_prod) * acc
}


// Various matrix utilities.

fn slice_ii(t: &Array4<f64>, rows: &[usize], cols: &[usize], i_fixed: usize, j_fixed: usize, first: bool) -> Array2<f64> {
    let l = rows.len();
    let mut out = Array2::<f64>::zeros((l, l));
    for r in 0..l {
        let rr = rows[r];
        for c in 0..l {
            let cc = cols[c];
            out[(r, c)] = if first {
                t[(rr, cc, i_fixed, j_fixed)]
            } else {
                t[(i_fixed, j_fixed, rr, cc)]
            };
        }
    }
    out
}

fn remove_bit(mask: u64, j: usize) -> u64 {
    let low = mask & ((1u64 << j) - 1);
    let high = mask >> (j + 1);
    low | (high << j)
}

fn lbl_to_idx(lbl: Label, nmo: usize) -> usize {
    label_to_idx(lbl.0, lbl.2, nmo)
}

fn build_d(x: &Array2<f64>, y: &Array2<f64>, rows: &[usize], cols: &[usize],) -> Array2<f64> {
    let l = rows.len();
    let mut out = Array2::<f64>::zeros((l, l));
    for i in 0..l {
        let r = rows[i];
        for j in 0..l {
            let c = cols[j];
            out[(i, j)] = if i >= j {x[(r, c)]} else {y[(r, c)]};
        }
    }
    out
}

fn mix_columns(det0: &Array2<f64>, det1: &Array2<f64>, bits: u64) -> Array2<f64> {
    let n = det0.ncols();
    let mut out = det0.clone();
    for c in 0..n {
        if ((bits >> c) & 1) == 1 {
            out.column_mut(c).assign(&det1.column(c));
        }
    }
    out
}

fn minor(m: &Array2<f64>, r_rm: usize, c_rm: usize) -> Array2<f64> {
    let n = m.nrows();
    if n == 0 {return Array2::<f64>::zeros((0, 0));}
    let mut out = Array2::<f64>::zeros((n - 1, n - 1));
    let mut ii = 0;
    for i in 0..n {
        if i == r_rm {continue;}
        let mut jj = 0;
        for j in 0..n {
            if j == c_rm {continue;}
            out[(ii, jj)] = m[(i, j)];
            jj += 1;
        }
        ii += 1;
    }
    out
}

fn adjugate_transpose(a: &Array2<f64>) -> Option<(f64, Array2<f64>)> {
    let n = a.nrows();

    if n == 0 {return Some((1.0, Array2::zeros((0, 0))));}
    if n == 1 {return Some((a[(0, 0)], Array2::from_elem((1, 1), 1.0)));}

    // Conditioning test.
    let cond = 1e12;
    let (_, s, _) = a.svd(false, false).ok()?;
    let smax = s[0];
    let smin = s[s.len() - 1];
    if smin <= 0.0 || smax / smin > cond {
        return None;
    }

    let det = a.det().unwrap();

    let mut adjt = Array2::<f64>::zeros((n, n));
    let at = a.t().to_owned();              
    for i in 0..n {
        let mut e = Array1::<f64>::zeros(n);
        e[i] = 1.0;
        let x = at.solve_into(e).unwrap();     
        adjt.column_mut(i).assign(&(&x * det)); 
    }
    Some((det, adjt))
}
