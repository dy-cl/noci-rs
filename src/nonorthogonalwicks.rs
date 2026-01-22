// nonorthogonalwicks.rs 
use ndarray::{Array1, Array2, Axis, s};
use ndarray_linalg::{SVD, Determinant};

use crate::{ExcitationSpin};

// Whether a given orbital index belongs to the bra or ket.
#[derive(PartialEq, Copy, Clone)]
pub enum Side {
    Gamma, 
    Lambda,
}

// Storage of quantities required to compute matrix elements between a given determinant pair
// \Gamma, \Lambda using non-orthogonal Wick's theorem. We use `l` as shorthand for \Lambda and 
// `g` \Gamma. 
pub struct WicksReferencePair {
    // Dimensions (2 * nmo of |{}^\Gamma \Psi\rangle x 2 * nmo of |{}^\Lambda \Psi \rangle).
    pub x0: Array2<f64>,
    pub x1: Array2<f64>,
    pub y0: Array2<f64>,
    pub y1: Array2<f64>,

    pub tilde_s_prod: f64,
    pub phase: f64,

    pub m: usize,
    pub nmo: usize,
}

impl WicksReferencePair {
    /// Constructor for the WicksReferencePair object which contains the precomputed values
    /// per pair of reference determinants required to evaluate arbitrary excited determinants 
    /// in O(1) time.
    /// # Arguments: 
    ///     `s_munu`: Array2, AO overlap matrix.
    ///     `g_c`: Array2, full AO coefficient matrix of |^\Gamma\Psi\rangle.
    ///     `l_c`: Array2, full AO coefficient matrix of |^\Lambda\Psi\rangle.
    ///     `g_c_occ`: Array2, occupied AO coefficient matrix of |^\Gamma\Psi\rangle.
    ///     `l_c_occ`: Array2, occupied AO coefficient matrix of |^\Lambda\Psi\rangle.
    pub fn new(s_munu: &Array2<f64>, g_c: &Array2<f64>, l_c: &Array2<f64>, g_c_occ: &Array2<f64>, l_c_occ: &Array2<f64>) -> Self {
        let tol = 1e-12;
        let nmo = g_c.ncols();
        
        // Get occupied MO overlap matrix.
        let s_occ = Self::calculate_mo_overlap_matrix(g_c_occ, l_c_occ, s_munu);
        // SVD and rotate the occupied orbitals.
        let (tilde_s_occ, g_tilde_c_occ, l_tilde_c_occ, phase) = Self::perform_svd_and_rotate(&s_occ, g_c_occ, l_c_occ);
        // Multiply diagonal non-zero values of {}^{\Gamma\Lambda} \tilde{S} together.
        let tilde_s_prod = tilde_s_occ.iter().filter(|&&x| x.abs() > tol).product::<f64>();
        // Find indices where zeros occur in {}^{\Gamma\Lambda} \tilde{S} and count them. 
        let zeros: Vec<usize> = tilde_s_occ.iter().enumerate().filter_map(|(k, &sk)| if sk.abs() <= tol {Some(k)} else {None}).collect();
        let m = zeros.len();
        // Construct the {}^{\Gamma\Lambda} M^{\sigma\tau, 0} and  {}^{\Gamma\Lambda} M^{\sigma\tau, 1} matrices.
        let (m0, m1) = Self::construct_m(&tilde_s_occ, &g_tilde_c_occ, &l_tilde_c_occ, &zeros);
        // Construct the {}^{\Gamma\Lambda} X_{ij}^{m_k} and {}^{\Gamma\Lambda} Y_{ij}^{m_k} matrices.
        let (x0, y0) = Self::construct_xy(g_c, l_c, s_munu, &m0, true);
        let (x1, y1) = Self::construct_xy(g_c, l_c, s_munu, &m1, false);
        
        Self {x0, y0, x1, y1, tilde_s_prod, phase, m, nmo}  
    }
    
    /// Calculate the overlap matrix between two sets of occupied orbitals as:
    ///     {}^{\Gamma\Lambda} S_{ij} = \sum_{\mu\nu} ({}^\Gamma C^*)_i^\mu S_{\mu\nu} ({}^\Lambda C)_j^\nu
    /// # Arguments:
    ///     `g_c_occ`: Array2, occupied coefficients ({}^\Gamma C^*)_i^\mu.
    ///     `l_c_occ`: Array2, occupied coefficients ({}^\Lambda C)_j^\nu 
    ///     `s_munu`: Array2, AO overlap matrix S_{\mu\nu}.
    fn calculate_mo_overlap_matrix(g_c_occ: &Array2<f64>, l_c_occ: &Array2<f64>, s_munu: &Array2<f64>) -> Array2<f64> {
        g_c_occ.t().dot(&s_munu.dot(l_c_occ))
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
    fn perform_svd_and_rotate(gl_s: &Array2<f64>, g_c_occ: &Array2<f64>, l_c_occ: &Array2<f64>) -> (Array1<f64>, Array2<f64>, Array2<f64>, f64){
        // SVD.
        let (u, gl_tilde_s, v_dag) = gl_s.svd(true, true).unwrap();
        let u = u.unwrap();
        let v = v_dag.unwrap().t().to_owned();
        
        // Rotate MOs.
        let g_tilde_c = g_c_occ.dot(&u);
        let l_tilde_c = l_c_occ.dot(&v);
        
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
    fn construct_m(gl_tilde_s: &Array1<f64>, g_tilde_c_occ: &Array2<f64>, l_tilde_c_occ: &Array2<f64>, zeros: &Vec<usize>) -> (Array2<f64>, Array2<f64>) {
        let tol = 1e-12;
        let nbas = g_tilde_c_occ.nrows();
        let nocc = g_tilde_c_occ.ncols();
        
        // Calculate {}^{\Gamma\Lambda} W^{\sigma\tau} (weighted co-density matrix) as:
        //      {}^{\Gamma\Lambda} W^{\sigma\tau} = \sum_i (^\Gamma \tilde{C})^\sigma_i (1 /
        //      {}^{\Gamma\Lambda} \tilde{S}_i) (^\Lambda \tilde{C}^*)^\tau_i
        //  for all i where {}^{\Gamma\Lambda} \tilde{S}_i != 0. This result is stored in
        //  {}^{\Gamma\Lambda} M^{\sigma\tau, 0}, where the zero indicates this quantity will be
        //  used when m_k = 0. 
        let mut g_tilde_c_occ_scaled = g_tilde_c_occ.clone();
        for k in 0..nocc {
            let s = gl_tilde_s[k];
            if s.abs() > tol {
                let scale = 1.0 / s;
            let mut col = g_tilde_c_occ_scaled.column_mut(k);
            col *= scale;
            } else {
                g_tilde_c_occ_scaled.column_mut(k).fill(0.0);
            }
        }
        let mut gl_m0 = g_tilde_c_occ_scaled.dot(&l_tilde_c_occ.t());
        let mut gl_m1 = Array2::<f64>::zeros((nbas, nbas));
        
        // Calculate {}^{\Gamma\Gamma} P^{\sigma\tau}_k (co-density matrix) as:
        //      {}^{\Gamma\Gamma} P^{\sigma\tau}_k = ({}^\Gamma \tilde{C})^\sigma_k ({}^\Gamma
        //      \tilde{C}^*)^\tau_k
        // for all k where {}^{\Gamma\Gamma} \tilde{S}_k = 0 and sum together to form {}^{\Gamma\Gamma} P^{\sigma\tau}.
        // This result is added to {}^{\Gamma\Lambda} M^{\sigma\tau, 0} which now
        // contains contributions from the \Gamma, \Gamma co-density matrix and \Gamma \Lambda
        // weighted co-density matrix.
        for &k in zeros {
            let g_tilde_c_occ_k = g_tilde_c_occ.column(k).to_owned();
            let outer = g_tilde_c_occ_k.view().insert_axis(Axis(1)).dot(&g_tilde_c_occ_k.view().insert_axis(Axis(0)));
            gl_m0 += &outer;
        }
        
        // Calculate {}^{\Gamma\Lambda} P^{\sigma\tau}_k (co-density matrix) as:
        //      {}^{\Gamma\Lambda} P^{\sigma\tau}_k = ({}^\Gamma \tilde{C})^\sigma_k ({}^\Lambda
        //      \tilde{C}^*)^\tau_k
        // for all k where {}^{\Gamma\Lambda} \tilde{S}_k = 0 and sum together to form {}^{\Gamma\Lambda} P^{\sigma\tau}.
        // This result is added to {}^{\Gamma\Lambda} M^{\sigma\tau, 0} which now
        // contains the correct contributions to be:
        //  {}^{\Gamma\Lambda} M^{\sigma\tau, 0} = {}^{\Gamma\Lambda} W^{\sigma\tau} + {}^{\Gamma\Lambda} P^{\sigma\tau} + {}^{\Gamma\Gamma} P^{\sigma\tau} 
        //  as required. Similarly we make {}^{\Gamma\Lambda} M^{\sigma\tau, 1} = {}^{\Gamma\Lambda} P^{\sigma\tau}.  
        for &k in zeros {
            let g_tilde_c_occ_k = g_tilde_c_occ.column(k).to_owned();
            let l_tilde_c_occ_k = l_tilde_c_occ.column(k).to_owned(); 
            let outer = g_tilde_c_occ_k.view().insert_axis(Axis(1)).dot(&l_tilde_c_occ_k.view().insert_axis(Axis(0)));
            gl_m1 += &outer;
            gl_m0 += &outer;
        }
        (gl_m0, gl_m1)
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
}

/// Given the orbitals excited from and to  relative to the reference determinants, construct the 
/// index labels for the L by L contraction matrix. The contraction matrix is blocked by quadrant
/// as: [\Lambda\Lambda & \Lambda\Gamma \\ \Gamma\Lambda & \Gamma\Gamma]. The rows correspond to
/// operations that act as creation operators after normal ordering to the bra (\Lambda), whilst
/// columns correspond to operations that are annhilations after normal ordering to the bra.
/// # Arguments:
///     `l_ex`: Excitation, left excitations relative to reference \Lambda in the bra.
///     `g_ex`: Excitation, right excitations relative to reference \Gamma in the ket.
fn construct_determinant_lables(l_ex: &ExcitationSpin, g_ex: &ExcitationSpin) -> (Vec<(Side, usize)>, Vec<(Side, usize)>) {
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
        for &a in &l_ex.parts {rows.push((Side::Lambda, a));}
        for &i in &l_ex.holes {cols.push((Side::Lambda, i));}
        return (rows, cols);
    }
    
    // If number of holes in \Gamma determinant is non-zero and number of holes in \Lambda
    // determinant is zero we only need to consider \Gamma determinant. The ordering is:
    //  rows = [\Gamma holes], cols = [\Gamma parts].
    if nl == 0 && ng > 0 {
        for &i in &g_ex.holes {rows.push((Side::Gamma, i));}
        for &a in &g_ex.parts {cols.push((Side::Gamma, a));}
        return (rows, cols);
    }
    
    // If both are non-zero we have the ordering:
    //  rows = [\Lambda parts ; \Gamma holes], cols = [\Lambda holes ; \Gamma parts]
    for &a in &l_ex.parts {rows.push((Side::Lambda, a));}
    for &i in &g_ex.holes {rows.push((Side::Gamma, i));}
    for &i in &l_ex.holes {cols.push((Side::Lambda, i));}
    for &a in &g_ex.parts {cols.push((Side::Gamma, a));}

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
        Side::Lambda => p,        // first block
        Side::Gamma  => nmo + p,  // second block
    }
}

/// Compute overlap \langle^\Lambda \Psi |^\Gamma \Psi \rangle between two determinants using the
/// extended non-orthogonal Wick's theorem  technique.
/// # Arguments: 
///     `w`: WicksReferencePair, precomputed values per pair of reference determinants required 
///     to evaluate arbitrary excited determinants.
///     `l_ex`, ExcitationSpin, per spin excitations of the bra relative to a reference.
///     `g_ex`, ExcitationSpin, per spin excitations of the ket relative to  a reference.
pub fn lg_overlap(w: &WicksReferencePair, l_ex: &ExcitationSpin, g_ex: &ExcitationSpin) -> f64 {

    let (rows_label, cols_label) = construct_determinant_lables(l_ex, g_ex);

    // If the total excitation rank is less than the number of zero-singular values in
    // {}^{\Gamma\Lambda} \tilde{S} the overlap element is zero.
    let l = rows_label.len();
    if w.m > l {return 0.0;}
    
    // Convert the contraction determinant labels into actual indices.
    let rows: Vec<usize> = rows_label.into_iter().map(|(s,i)| label_to_idx(s, i, w.nmo)).collect();
    let cols: Vec<usize> = cols_label.into_iter().map(|(s,i)| label_to_idx(s, i, w.nmo)).collect();

    let mut acc = 0.0;
    
    // Iterate over all possible distributions of zeros amongst the columns.
    for bitstring in iter_m_combinations(l, w.m) {
        // Build two full contraction determinants each having exclusively (X0, Y0) or (X1, Y1). 
        let mut zerodet = Array2::<f64>::zeros((l, l)); 
        let mut onedet = Array2::<f64>::zeros((l, l));

        for a in 0..l {
            for b in 0..l {
                let ra = rows[a];
                let cb = cols[b];
                // Lower triangle and diagonal of the determinant uses X0 or X1.
                if a >= b {
                    zerodet[(a, b)] = w.x0[(ra, cb)];
                    onedet[(a, b)] = w.x1[(ra, cb)];
                // Upper triangle uses Y0 or Y1.
                } else {
                    zerodet[(a, b)]    = w.y0[(ra, cb)];
                    onedet[(a, b)] = w.y1[(ra, cb)];
                }
            }
        }
        
        // Iterate over columns and decide to keep the 0 determinant or exchange a column for the
        // one determinant version.
        let mut det = zerodet.clone();
        for b in 0..l {
            // Extract the m_k for this coulmn by moving the 1 or 0 to the rightmost point of the
            // bitstring, read it (&1) and convert into boolean (== 1). 
            let useone = ((bitstring >> b) & 1) == 1;
            // If we are  using the one determinant, exchange all entries in a column.
            if useone {
                for a in 0..l {
                    det[(a, b)] = onedet[(a, b)];
                }
            }
        }
        acc += det.det().unwrap();
    }
    w.phase * w.tilde_s_prod * acc
}

