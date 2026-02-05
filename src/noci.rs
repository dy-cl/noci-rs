// noci.rs
//use std::io::{self, Write};
use std::time::{Duration, Instant};

use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::{SVD, Determinant};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::{AoData, SCFState};
use crate::nonorthogonalwicks::{lg_h2_diff, lg_h2_same, lg_h1, lg_overlap, WicksReferencePair, SameSpin, DiffSpin};

use crate::utils::print_array2;
use crate::maths::{einsum_ba_ab_real, einsum_ba_abcd_cd_real, general_evp_real};

// Storage of quantities required to compute matrix elements between determinant pairs in the naive fashion.
pub struct Pair {
    pub s: f64,
    pub tilde_s: Array1<f64>,
    pub s_red: f64,
    pub zeros: Vec<usize>,
    pub l_tilde_c_occ: Array2<f64>,
    pub g_tilde_c_occ: Array2<f64>,
    pub w: Option<Array2<f64>>,
    pub p_i: Option<Array2<f64>>,
    pub p_j: Option<Array2<f64>>,
    pub phase: f64,
}

/// Given an MO coefficient matrix and a corresponding occupancy vector, return the occupied only
/// coefficient matrix.
/// # Arguments:
///     `c`: Array2, MO coefficient matrix.
///     `occ`: Array1, occupancy vector.
pub fn occ_coeffs(c: &Array2<f64>, occ: &Array1<f64>) -> Array2<f64> {
    // occ is length nmo; treat >0.5 as occupied
    let occ_idx: Vec<usize> = occ.iter().enumerate().filter_map(|(i, &x)| if x > 0.5 {Some(i)} else {None}).collect();

    let nbas = c.nrows();
    let nocc = occ_idx.len();
    let mut c_occ = Array2::<f64>::zeros((nbas, nocc));

    for (k, &i) in occ_idx.iter().enumerate() {c_occ.column_mut(k).assign(&c.index_axis(Axis(1), i));}

    c_occ
}

/// Calculate the reduced occupied MO overlap scalar of {}^{\Lambda\Gamma} \tilde{S} as the product of
/// all non-zero singular values of the SVD'd {}^{\Lambda\Gamma} \tilde{S}.
/// # Arguments
///     `tilde_s`: Array1, vector of singular values, the diagonal of {}^{\Lambda\Gamma} \tilde{S}. 
///     `tol`: f64, tolerance up to which a number is considered zero. 
fn calculate_s_red(tilde_s: &Array1<f64>, tol: f64) -> (f64, Vec<usize>) {
    let mut prod = 1.0f64;
    let mut zeros = Vec::new();

    for (i, &v) in tilde_s.iter().enumerate() {
        if v.abs() > tol {
            prod *= v;      
        } else {
            zeros.push(i);
        }
    }
    (prod, zeros)
}

/// Calculate overlap between determinants \Lambda and \Gamma. An SVD is performed on the occupied
/// MO overlap matrix {}^{\Lambda\Gamma} S_{ij} as {}^{\Lambda\Gamma} S_{ij} = U {}^{\Lambda\Gamma}\tilde{S}_{ij} V^{\dagger}, 
/// each set of occupied MOs are rotated to form {}^\Lambda \tilde{C} and {}^\Gamma \tilde{C}, and
/// the overlap element between the determinants the product of diagonal (singular) values in {}^{\Lambda\Gamma}\tilde{S}_{ij}.
/// # Arguments:
///     `l_c_occ`: Array2, occupied MO coefficient matrix {}^\Lambda C.
///     `g_c_occ`: Array2, occupied MO coefficient matrix {}^\Gamma C.
///     `s_munu`: Array2, AO overlap matrix.
///     `tol`: f64, tolerance up to which a number is considered zero. 
fn build_s_pair(l_c_occ: &Array2<f64>, g_c_occ: &Array2<f64>, s_munu: &Array2<f64>, tol: f64) -> Pair {
    
    // Occupied MO overlap.
    let s_ij = l_c_occ.t().dot(&s_munu.dot(g_c_occ));

    // SVD the MO overlap matrix.
    let (u, tilde_s, vdag) = s_ij.svd(true, true).unwrap();
    let u = u.unwrap();
    let v = vdag.unwrap().t().to_owned();
    let phase = u.det().unwrap() * v.det().unwrap();

    // Calculate the reduced MO overlap matrix.
    let (s_red, zeros) = calculate_s_red(&tilde_s, tol);

    // Rotate occupied MO coefficients
    let l_tilde_c_occ = l_c_occ.dot(&u);
    let g_tilde_c_occ = g_c_occ.dot(&v);
    
    // Calculate only the required quantities given the number of zeros.
    let w = match zeros.len() {
        0 | 1 => Some(calculate_codensity_w_pair(&l_tilde_c_occ, &g_tilde_c_occ, &tilde_s, tol)),
        _ => None,
    };

    let (p_i, p_j) = match zeros.len() {
        1 => (
            Some(calculate_codensity_p_pair(&l_tilde_c_occ, &g_tilde_c_occ, zeros[0])), 
            None
        ),
        2 => (
            Some(calculate_codensity_p_pair(&l_tilde_c_occ, &g_tilde_c_occ, zeros[0])), 
            Some(calculate_codensity_p_pair(&l_tilde_c_occ, &g_tilde_c_occ, zeros[1]))
        ),
        _ => (None, None),
    };

    // Overlap matrix element for this pair.
    let prod: f64 = tilde_s.iter().copied().product();
    let s = phase * prod;

    Pair {s, tilde_s, s_red, zeros, l_tilde_c_occ, g_tilde_c_occ, w, p_i, p_j, phase}
}
    
/// Calculate {}^{\Lambda\Gamma}P_i^{\mu\nu} = {}^{\Lambda} \tilde{C}_i^\mu {}^{\Gamma}\tilde{C}_i^\nu* 
/// co-density matrix where {}^{\Lambda} \tilde{C}_i^\mu and {}^{\Gamma}\tilde{C}_i^\nu* are the 
/// rotated MO coefficients of determinant \Lambda and \Gamma respectively.
/// # Arguments 
///     `l_tilde_c_occ`: Array2, U rotated occupied MO coefficients for pair of states \Lambda, \Gamma. 
///     `g_tilde_c_occ`: Array2, V rotated occupied MO coefficients for pair of states \Lambda, \Gamma. 
///     `i`: usize, MO index.
fn calculate_codensity_p_pair(l_tilde_c_occ: &Array2<f64>, g_tilde_c_occ: &Array2<f64>, i: usize) -> Array2<f64> {
    let nso = l_tilde_c_occ.nrows();
    let mut munu_p_i = Array2::<f64>::zeros((nso, nso));
    for x in 0..nso {
        for y in 0..nso {
            munu_p_i[(x, y)] = l_tilde_c_occ[(x, i)] * g_tilde_c_occ[(y, i)];
        }
    }
    munu_p_i
}

/// Calculate {}^{\Lambda\Gamma}W^{\mu\nu} = \sum_{i} 1 / s_i * {}^{\Lambda}\tilde{C}_i^\mu {}^{\Gamma}\tilde{C}_i^\nu* 
/// weighted co-density  matrix where s_i are the singular values of the SVD decomposed MO overlap matrix, 
/// and {}^{\Lambda}\tilde{C}_i^\mu and {}^{\Gamma}\tilde{C}_i^\nu are the occupied rotated MO coefficients of state 
/// \Gamma and \Lambda respectively.
/// # Arguments:
///     `l_tilde_c_occ`: Array2, U rotated occupied MO coefficients for pair of states \Lambda, \Gamma. 
///     `g_tilde_c_occ`: Array2, V rotated occupied MO coefficients for pair of states \Lambda, \Gamma.
///     `s_vals`: Array1, singular values of SVD'd {}^{\Lambda\Gamma} \tilde{S}_{ij}.
///     `tol`: f64, tolerance up to which a number is considered zero. 
fn calculate_codensity_w_pair(l_tilde_c_occ: &Array2<f64>, g_tilde_c_occ: &Array2<f64>, tilde_s: &Array1<f64>, tol: f64) -> Array2<f64> {
    let mut l_tilde_c_occ_scaled = l_tilde_c_occ.to_owned();
    
    for (i, mut col) in l_tilde_c_occ_scaled.axis_iter_mut(Axis(1)).enumerate() {
        let w = if tilde_s[i].abs() > tol {1.0 / tilde_s[i]} else {0.0};
        col.mapv_inplace(|z| z * w);
    }

    l_tilde_c_occ_scaled.dot(&g_tilde_c_occ.t())
}

/// Calculate one electron and nuclear Hamiltonian matrix elements using the generalised 
/// Slater-Condon rules for a pair of determinants \Lambda and \Gamma. 
/// # Arguments:
///     `pair`: Pair struct, contains the following data concerning a pair of determinants:
///         `s`: f64, overlap matrix element for this pair.
///         `tilde_s`: Array1, singular values of SVD'd {}^{\Lambda\Gamma} \tilde{S}_{ij}.
///         `s_red`: f64, product of the non-zero values of tilde_s.
///         `zeros`: [usize], array containing zero singular value indices.
///         `l_tilde_c_occ`: Array2, rotated occupied MO coefficient matrix of determinant \Lambda.
///         `g_tilde_c_occ`: Array2, rotated occupied MO coefficient matrix of determinant \Gamma.
///         `w`: Array2, weighted codensity matrix {}^{\Lambda\Gamma}W^{\mu\nu} for this pair.
///         `p_i`: Array2, unweighted codensity matrix {}^{\Lambda\Gamma}P_i^{\mu\nu}.
///         `p_j`: Array2, unweighted codensity matrix {}^{\Lambda\Gamma}P_j^{\mu\nu}. 
///         `phase`: f64, Phase associated with determinant pair \mu \nu.
///     `ao`: AoData struct, contains AO integrals and other system data. 
fn one_electron_h(ao: &AoData, pair: &Pair) -> f64 {
    match pair.zeros.len() {
        // With no zeros (s_i != 0 for all i) we use munu_w.
        0 => pair.s_red * pair.phase * einsum_ba_ab_real(pair.w.as_ref().unwrap(), &ao.h),
        // With 1 zero (s_i = 0 for 1 i) we use P_i.
        1 => pair.s_red * pair.phase * einsum_ba_ab_real(pair.p_i.as_ref().unwrap(), &ao.h),
        // Otherwise the matrix element is zero.
        _ => 0.0,
    }
}

/// Calculate two electron Hamiltonian matrix elements for electrons of the same spin 
/// using the generalised Slater-Condon rules for a pair of determinants \Lambda and \Gamma.
/// # Arguments:
///     `pair`: Pair struct, contains the following data concerning a pair of determinants:
///         `s`: f64, overlap matrix element for this pair.
///         `tilde_s`: Array1, singular values of SVD'd {}^{\Lambda\Gamma} \tilde{S}_{ij}.
///         `s_red`: f64, product of the non-zero values of tilde_s.
///         `zeros`: [usize], array containing zero singular value indices.
///         `l_tilde_c_occ`: Array2, rotated occupied MO coefficient matrix of determinant \Lambda.
///         `g_tilde_c_occ`: Array2, rotated occupied MO coefficient matrix of determinant \Gamma.
///         `w`: Array2, weighted codensity matrix {}^{\Lambda\Gamma}W^{\mu\nu} for this pair.
///         `p_i`: Array2, unweighted codensity matrix {}^{\Lambda\Gamma}P_i^{\mu\nu}.
///         `p_j`: Array2, unweighted codensity matrix {}^{\Lambda\Gamma}P_j^{\mu\nu}. 
///         `phase`: f64, Phase associated with determinant pair \mu \nu.
///     `ao`: AoData struct, contains AO integrals and other system data. 
fn two_electron_h_same(ao: &AoData, pair: &Pair) -> f64 {
    match pair.zeros.len() {
        // With no zeros (s_i != 0 for all i) we use munu_w on both sides.
        0 => 0.5 * pair.s_red * pair.phase * einsum_ba_abcd_cd_real(pair.w.as_ref().unwrap(), &ao.eri_asym, pair.w.as_ref().unwrap()),
        // With 1 zero (s_i = 0 for one index i) we use P_i on one side.
        1 => pair.s_red * pair.phase * einsum_ba_abcd_cd_real(pair.w.as_ref().unwrap(), &ao.eri_asym, pair.p_i.as_ref().unwrap()),
        // with 2 zeros (s_i, s_j = 0 for two indices i, j) we use P_i on one side and P_j on the other.
        2 => pair.s_red * pair.phase * einsum_ba_abcd_cd_real(pair.p_i.as_ref().unwrap(), &ao.eri_asym, pair.p_j.as_ref().unwrap()),
        // Otherwise the matrix element is 0.
        _ => 0.0,
    }
}

/// Calculate two electron Hamiltonian matrix elements for electrons of opposite spin 
/// using the generalised Slater-Condon rules for a pair of determinants \Lambda and \Gamma.
/// # Arguments:
///     `pair`: Pair struct, contains the following data concerning a pair of determinants:
///         `s`: f64, overlap matrix element for this pair.
///         `tilde_s`: Array1, singular values of SVD'd {}^{\Lambda\Gamma} \tilde{S}_{ij}.
///         `s_red`: f64, product of the non-zero values of tilde_s.
///         `zeros`: [usize], array containing zero singular value indices.
///         `l_tilde_c_occ`: Array2, rotated occupied MO coefficient matrix of determinant \Lambda.
///         `g_tilde_c_occ`: Array2, rotated occupied MO coefficient matrix of determinant \Gamma.
///         `w`: Array2, weighted codensity matrix {}^{\Lambda\Gamma}W^{\mu\nu} for this pair.
///         `p_i`: Array2, unweighted codensity matrix {}^{\Lambda\Gamma}P_i^{\mu\nu}.
///         `p_j`: Array2, unweighted codensity matrix {}^{\Lambda\Gamma}P_j^{\mu\nu}. 
///         `phase`: f64, Phase associated with determinant pair \mu \nu.
///     `ao`: AoData struct, contains AO integrals and other system data. 
fn two_electron_h_diff(ao: &AoData, pa: &Pair, pb: &Pair) -> f64 {
    match (pa.zeros.len(), pb.zeros.len()) {
        // With no zeros (s_i != 0 for all i) for both spins we use munu_w on both sides.
        (0, 0) => (pa.s_red * pa.phase) * (pb.s_red * pb.phase) * einsum_ba_abcd_cd_real(pa.w.as_ref().unwrap(), &ao.eri_coul, pb.w.as_ref().unwrap()),
        // With one zero in beta spin only we use W^a and P_i^b.
        (0, 1) => (pa.s_red * pa.phase) * (pb.s_red * pb.phase) * einsum_ba_abcd_cd_real(pa.w.as_ref().unwrap(), &ao.eri_coul, pb.p_i.as_ref().unwrap()),
        // With one zero in alpha spin only we use P_i^a and W_b.
        (1, 0) => (pa.s_red * pa.phase) * (pb.s_red * pb.phase) * einsum_ba_abcd_cd_real(pa.p_i.as_ref().unwrap(), &ao.eri_coul, pb.w.as_ref().unwrap()),
        // With one zero in both spins we use P_i^a and P_i^b.
        (1, 1) => (pa.s_red * pa.phase) * (pb.s_red * pb.phase) * einsum_ba_abcd_cd_real(pa.p_i.as_ref().unwrap(), &ao.eri_coul, pb.p_i.as_ref().unwrap()),
        // Otherwise the matrix element is zero.
        _ => 0.0,
    }
}

/// Calculate the overlap matrix element between determinants \Lambda and \Gamma using 
/// generalised Slater-Condon rules.
/// # Arguments:
///     `ao`: AoData struct, contains AO integrals and other system data. 
///     `determinants`: Vec<SCFState>, vector of all the determinants in the NOCI basis.
///     `l`: usize, index of state \Lambda.
///     `g`: usize, index of state \Gamma.
pub fn calculate_s_pair(ao: &AoData, determinants: &[SCFState], l: usize, g: usize) -> f64 {

    // Tolerance for a number being non-zero.
    let tol = 1e-12;

    // Per spin occupid coefficients.
    let l_ca_occ = occ_coeffs(&determinants[l].ca, &determinants[l].oa);
    let g_ca_occ = occ_coeffs(&determinants[g].ca, &determinants[g].oa);
    let l_cb_occ = occ_coeffs(&determinants[l].cb, &determinants[l].ob);
    let g_cb_occ = occ_coeffs(&determinants[g].cb, &determinants[g].ob);

    let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &ao.s, tol);
    let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &ao.s, tol);

    // Overlap matrix element for this pair. 
    pa.s * pb.s
}

/// Calculate both the overlap and Hamiltonian matrix elements between determinants \Lambda and \Gamma 
/// using generalised Slater-Condon rules.
/// # Arguments:
///     `ao`: AoData struct, contains AO integrals and other system data. 
///     `determinants`: Vec<SCFState>, vector of all the determinants in the NOCI basis.
///     `l`: usize, index of state \Lambda.
///     `g`: usize, index of state \Gamma.
pub fn calculate_hs_pair(ao: &AoData, determinants: &[SCFState], l: usize, g: usize) -> (f64, f64, f64, f64, f64, f64, f64) {

    // Tolerance for a number being non-zero.
    let tol = 1e-12;

    // Per spin occupid coefficients.
    let l_ca_occ = occ_coeffs(&determinants[l].ca, &determinants[l].oa);
    let g_ca_occ = occ_coeffs(&determinants[g].ca, &determinants[g].oa);
    let l_cb_occ = occ_coeffs(&determinants[l].cb, &determinants[l].ob);
    let g_cb_occ = occ_coeffs(&determinants[g].cb, &determinants[g].ob);

    let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &ao.s, tol);
    let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &ao.s, tol);

    // Overlap matrix element for this pair. 
    let s = pa.s * pb.s;
    
    let hnuc = match (pa.zeros.len(), pb.zeros.len()) {
        (0, 0) => ao.enuc * s,
        _ => 0.0,
    };

    let h1a = one_electron_h(ao, &pa);
    let h1b = one_electron_h(ao, &pb);
    let h1 = pb.s * h1a + pa.s * h1b;

    let h2aa = pb.s * two_electron_h_same(ao, &pa); 
    let h2bb = pa.s * two_electron_h_same(ao, &pb); 
    let h2ab = two_electron_h_diff(ao, &pa, &pb);
    let h2 = h2aa + h2bb + h2ab;

    (hnuc + h1 + h2, h2aa, h2bb, h2ab, h1, h2, s)
}

/// Form the full Hamiltonian and overlap matrices using the generalised Slater-Condon rules.
/// # Arguments:
///     `ao`: AoData struct, contains AO integrals and other system data.
///     `determinants`: Vec<SCFState>, vector of all determinants in the NOCI basis.
///     `noci_reference_basis`: Vec<SCFState>, vector of only the reference determinants.
pub fn build_noci_matrices(ao: &AoData, determinants: &[SCFState], noci_reference_basis: &[SCFState]) -> (Array2<f64>, Array2<f64>, Duration) {
    let ndets = determinants.len();
    
    let mut h = Array2::<f64>::zeros((ndets, ndets));
    let mut s = Array2::<f64>::zeros((ndets, ndets));
    
    // Testing new matrix element calculation routine.
    let nref = noci_reference_basis.len();
    let mut wicks: Vec<Vec<WicksReferencePair>> = Vec::with_capacity(nref);
    for ri in noci_reference_basis.iter() {
        let mut row: Vec<WicksReferencePair> = Vec::with_capacity(nref);

        for rj in noci_reference_basis.iter() {
            //println!("PARENTS (AA): L: {}, G: {}", ri.parent, rj.parent);
            let aa = SameSpin::new(&ao.eri_coul, &ao.h, &ao.s, &rj.ca, &ri.ca, &rj.oa, &ri.oa);
            //println!("PARENTS (BB): L: {}, G: {}", ri.parent, rj.parent);
            let bb = SameSpin::new(&ao.eri_coul, &ao.h, &ao.s, &rj.cb, &ri.cb, &rj.ob, &ri.ob);
            //println!("PARENTS (AB): L: {}, G: {}", ri.parent, rj.parent);
            let ab = DiffSpin::new(&ao.eri_coul, &ao.s, &rj.ca, &rj.cb, &ri.ca, &ri.cb, &rj.oa, &rj.ob, &ri.oa, &ri.ob);

            row.push(WicksReferencePair { aa, bb, ab });
        }
        wicks.push(row);
    }

    // Build list of all upper-triangle and diagonal pairs \Lambda, \Gamma. 
    let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let pairs: Vec<(usize, usize)> = (0..ndets).flat_map(|mu| (mu..ndets).map(move |nu| (mu, nu))).collect();

    // Calculate Hamiltonian matrix elements in parallel.
    let t_h = Instant::now();
    let tmp: Vec<(usize, usize, f64, f64, f64)> = pool.install(|| { 
        pairs.par_iter().map(|&(l, g)| {

            // Calculate matrix elements for this pair.
            let (h, h2aa, h2bb, h2ab, h1, h2, s) = calculate_hs_pair(ao, determinants, l, g);

            let lp = determinants[l].parent;
            let gp = determinants[g].parent;

            //println!("PARENTS (AA): L: {}, G: {}", lp, gp);

            // Wicks testing.
            let w = &wicks[determinants[l].parent][determinants[g].parent];
            let ex_la = &determinants[l].excitation.alpha;
            let ex_ga = &determinants[g].excitation.alpha;
            let ex_lb = &determinants[l].excitation.beta;
            let ex_gb = &determinants[g].excitation.beta;

            let occ_la = occvec_to_bits(&noci_reference_basis[lp].oa, 1e-12);
            let occ_lb = occvec_to_bits(&noci_reference_basis[lp].ob, 1e-12);
            let occ_ga = occvec_to_bits(&noci_reference_basis[gp].oa, 1e-12);
            let occ_gb = occvec_to_bits(&noci_reference_basis[gp].ob, 1e-12);

            let ph_a = excitation_phase(occ_la, &ex_la.holes, &ex_la.parts) * excitation_phase(occ_ga, &ex_ga.holes, &ex_ga.parts);
            let ph_b = excitation_phase(occ_lb, &ex_lb.holes, &ex_lb.parts) * excitation_phase(occ_gb, &ex_gb.holes, &ex_gb.parts);

            let sa = ph_a * lg_overlap(&w.aa, ex_la, ex_ga);
            let sb = ph_b * lg_overlap(&w.bb, ex_lb, ex_gb);
            let s_w = sa * sb;

            let h1a = lg_h1(&w.aa, ex_la, ex_ga);
            let h1b = lg_h1(&w.bb, ex_lb, ex_gb);
            let h1_w = ph_a * h1a * sb + ph_b * h1b * sa;

            let h2aa_w = 0.5 * ph_a * sb * lg_h2_same(&w.aa, ex_la, ex_ga);
            let h2bb_w = 0.5 * ph_b * sa * lg_h2_same(&w.bb, ex_lb, ex_gb);
            let h2ab_w = (ph_a * ph_b) * lg_h2_diff(w, ex_la, ex_ga, ex_lb, ex_gb);
            let h2_w = h2aa_w + h2bb_w + h2ab_w;

            let hnuc_w = match (w.aa.m, w.bb.m) {
                (0, 0) => ao.enuc * s_w,
                _ => 0.0,
            };

            let h_w = hnuc_w + h1_w + h2_w;

            let dtot = (s - s_w).abs() + (h1 - h1_w).abs() + (h2 - h2_w).abs();
            //let ds = (s - s_w).abs();
            //let dh1 = (h1 - h1_w).abs();
            //let dh2aa = (h2aa - h2aa_w).abs();
            //let dh2bb = (h2bb - h2bb_w).abs();
            //let dh2ab = (h2ab - h2ab_w).abs();
            //let dh2 = (h2 - h2_w ).abs();
            
            //println!(
            //    concat!(
            //        "Lambda Alpha holes: {:?}, Lambda Alpha parts: {:?}\n",
            //        "Gamma  Alpha holes: {:?}, Gamma  Alpha parts: {:?}\n",
            //        "Lambda Beta  holes: {:?}, Lambda Beta  parts: {:?}\n",
            //        "Gamma  Beta  holes: {:?}, Gamma  Beta  parts: {:?}\n",
            //        "l:{l:>3} g:{g:>3}  | S    {s:+.10} (w {sw:+.10}) | d {ds:+.10}\n",
            //        "             | H1   {h1:+.10} (w {h1w:+.10}) | d {dh1:+.10}\n",
            //        "             | H2   {h2:+.10} (w {h2w:+.10}) | d {dh2:+.10}\n",
            //        "             | H2aa {h2aa:+.10} (w {h2aaw:+.10}) | d {dh2aa:+.10}\n",
            //        "             | H2bb {h2bb:+.10} (w {h2bbw:+.10}) | d {dh2bb:+.10}\n",
            //        "             | H2ab {h2ab:+.10} (w {h2abw:+.10}) | d {dh2ab:+.10}\n",
            //        "             | dtot    {dtot:+.10}\n",
            //    ),
            //    ex_la.holes, ex_la.parts,
            //    ex_ga.holes, ex_ga.parts,
            //    ex_lb.holes, ex_lb.parts,
            //    ex_gb.holes, ex_gb.parts,
            //    l = l, g = g,

            //    s = s,   sw = s_w,   ds = ds,
            //    h1 = h1, h1w = h1_w, dh1 = dh1,
            //    h2 = h2, h2w = h2_w, dh2 = dh2,

            //    h2aa = h2aa, h2aaw = h2aa_w, dh2aa = dh2aa,
            //    h2bb = h2bb, h2bbw = h2bb_w, dh2bb = dh2bb,
            //    h2ab = h2ab, h2abw = h2ab_w, dh2ab = dh2ab,
            //    dtot = dtot,
            //);

            (l, g, h_w, s_w, dtot)
        })
        .collect()
    });

    let d_h = t_h.elapsed();

    // Calculate discrepancy between naively calculated matrix elements and Wicks ones.
    let mut dtot_sum = 0.0_f64;
    for (_, _, _, _, dtot) in &tmp {
        dtot_sum += *dtot;
    }
    println!("Wicks vs naive discrepancy: sum(|Δ|) over all pairs: {:.6e}", dtot_sum);

    // Scatter Hamiltonian and overlap matrix elements into full matrices. 
    for (l, g, lg_h, lg_s, _) in tmp {
        h[(l, g)] = lg_h;
        s[(l, g)] = lg_s;
        // Hermitian.
        if l != g {
            h[(g, l)] = lg_h;
            s[(g, l)] = lg_s;
        }
    }
    (h, s, d_h)
}

/// Calculate NOCI energy by solving GEVP with NOCI Hamiltonian and overlap.
/// # Arguments:
///     `scfstates`: Vec<SCFState>, vector of all the calculated SCF states. 
///     `ao`: AoData struct, contains AO integrals and other system data. 
pub fn calculate_noci_energy(ao: &AoData, scfstates: &[SCFState]) -> (f64, Array1<f64>, Duration) {
    let tol = f64::EPSILON;
    let (h, s, d_h) = build_noci_matrices(ao, scfstates, scfstates);
        
    println!("NOCI-reference Hamiltonian:");
    print_array2(&h);
    println!("NOCI-reference Overlap:");
    print_array2(&s);
    println!("Shifted NOCI-reference Hamiltonian");
    let h_shift = &h.map(|z: &f64| z) - scfstates[0].e * &s;
    print_array2(&h_shift);
    let (evals, c) = general_evp_real(&h, &s, true, tol);
    println!("GEVP eigenvalues in NOCI-reference basis: {}", evals);

    // Assumes columns of c are energy ordered eigenvectors
    let c0 = c.column(0).to_owned(); 
    (evals[0], c0, d_h)
}

// Fermionic phase logic.

fn occvec_to_bits(occ: &Array1<f64>, tol: f64) -> u64 {
    let mut bits = 0u64;
    for (i, &x) in occ.iter().enumerate() {
        if x > 1.0 - tol { bits |= 1u64 << i; }
    }
    bits
}

fn excitation_phase(mut occ: u64, holes: &[usize], parts: &[usize]) -> f64 {
    fn below(bits: u64, p: usize) -> u32 {
        if p == 0 { return 0; }
        (bits & ((1u64 << p) - 1)).count_ones()
    }

    let mut ph = 1.0;

    let mut hs = holes.to_vec();
    hs.sort_unstable_by(|a,b| b.cmp(a));    
    for &i in &hs {
        if (below(occ, i) & 1) == 1 { ph = -ph; }
        occ &= !(1u64 << i);
    }

    let mut ps = parts.to_vec();
    ps.sort_unstable();                     
    for &a in &ps {
        if (below(occ, a) & 1) == 1 { ph = -ph; }
        occ |= 1u64 << a;
    }

    ph
}

