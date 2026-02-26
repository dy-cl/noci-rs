// noci.rs
use std::time::{Duration, Instant};
use std::fs::{create_dir_all};
use std::io::{Write};
use std::ptr::NonNull;

use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::{SVD, Determinant};
use rayon::prelude::*;
use mpi::topology::Communicator;

use crate::{AoData, SCFState};
use crate::nonorthogonalwicks::{DiffSpinBuild, DiffSpinMeta, PairMeta, SameSpinBuild, SameSpinMeta, WickScratch, WicksRma, WicksShared, WicksView};
use crate::mpiutils::Sharedffi;
use crate::input::Input;

use crate::utils::{excitation_phase, occvec_to_bits, print_array2};
use crate::nonorthogonalwicks::{lg_h1, lg_h2_diff, lg_h2_same, lg_overlap, write_same_spin, write_diff_spin, assign_offsets, prepare_same};
use crate::maths::{einsum_ba_ab_real, einsum_ba_abcd_cd_real, general_evp_real};
use crate::mpiutils::{broadcast};

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
pub fn calculate_s_pair_naive(ao: &AoData, determinants: &[SCFState], l: usize, g: usize, tol: f64) -> f64 {

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
pub fn calculate_hs_pair_naive(ao: &AoData, determinants: &[SCFState], l: usize, g: usize, tol: f64) -> (f64, f64) {

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

    (hnuc + h1 + h2, s)
}


/// Build the Wick's per reference-pair intermediates and store in a shared memory access region (per node).
/// # Arguments:
///     `world`: Communicator, MPI communicator object.
///     `ao`: AoData struct, contains AO integrals and other system data. 
///     `noci_reference_basis`: [SCFState], vector of only the reference determinants.
///     `tol`: f64, tolerance for a number being zero.
pub fn build_wicks_shared(world: &impl Communicator, ao: &AoData, noci_reference_basis: &[SCFState], tol: f64) -> WicksShared {

    let nref = noci_reference_basis.len();
    let nmo  = noci_reference_basis[0].ca.ncols();

    let (offset, tensor_len) = assign_offsets(nref, nmo);

    let nbytes = tensor_len * std::mem::size_of::<f64>();
    let shared = Sharedffi::allocate(world, nbytes);
    let shared_rank = shared.shared_rank;

    let tensor_ptr = shared.base as *mut f64;
    let mut meta = vec![PairMeta::default(); nref * nref];

    if shared_rank == 0 {
        let tensor: &mut [f64] = unsafe {std::slice::from_raw_parts_mut(tensor_ptr, tensor_len)};
        tensor.fill(f64::NAN);

        for i in 0..nref {
            let ri = &noci_reference_basis[i];
            for (j, rj) in noci_reference_basis.iter().enumerate() {
                println!("Building intermediates for reference pair: {}, {}", i, j);

                let aa = SameSpinBuild::new(&ao.eri_coul, &ao.h, &ao.s, &rj.ca, &ri.ca, &rj.oa, &ri.oa, tol);
                let bb = SameSpinBuild::new(&ao.eri_coul, &ao.h, &ao.s, &rj.cb, &ri.cb, &rj.ob, &ri.ob, tol);
                let ab = DiffSpinBuild::new(&ao.eri_coul, &ao.s, &rj.ca, &rj.cb, &ri.ca, &ri.cb, &rj.oa, &rj.ob, &ri.oa, &ri.ob, tol);

                let idx = i * nref + j;

                meta[idx].aa = SameSpinMeta {tilde_s_prod: aa.tilde_s_prod, phase: aa.phase, m: aa.m, nmo: aa.nmo, f0: aa.f0, v0: aa.v0};
                meta[idx].bb = SameSpinMeta {tilde_s_prod: bb.tilde_s_prod, phase: bb.phase, m: bb.m, nmo: bb.nmo, f0: bb.f0, v0: bb.v0};
                meta[idx].ab = DiffSpinMeta {nmo: ab.vab[0][0][0].nrows() / 2, vab0: ab.vab0, vba0: ab.vba0};

                write_same_spin(tensor, &offset[idx].aa, &aa);
                write_same_spin(tensor, &offset[idx].bb, &bb);
                write_diff_spin(tensor, &offset[idx].ab, &ab);
            }
        }
    }

    shared.barrier();

    broadcast(world, &mut meta);

    let rma = WicksRma {base_ptr: shared.base, nbytes, shared,};
    let slab = NonNull::new(tensor_ptr).expect("Should not be null.");
    let view = WicksView {slab, slab_len: tensor_len, nref, off: offset, meta};

    WicksShared {rma, view}
}

/// Calculate the overlap matrix element between determinants \Lambda and \Gamma using 
/// extended non-orthogonal Wick's theorem.
/// # Arguments:
///     `determinants`: Vec<SCFState>, vector of all the determinants in the NOCI basis.
///     `noci_reference_basis`: Vec<SCFState>, vector of only the reference determinants.
///     `l`: usize, index of state \Lambda.
///     `g`: usize, index of state \Gamma.
pub fn calculate_s_pair_wicks(determinants: &[SCFState], noci_reference_basis: &[SCFState], l: usize, g: usize, tol: f64, wicks: &WicksView, scratch: &mut WickScratch) -> f64 {

    let lp = determinants[l].parent;
    let gp = determinants[g].parent;

    let w = &wicks.pair(lp, gp);

    let ex_la = &determinants[l].excitation.alpha;
    let ex_ga = &determinants[g].excitation.alpha;
    let ex_lb = &determinants[l].excitation.beta;
    let ex_gb = &determinants[g].excitation.beta;

    let occ_la = occvec_to_bits(&noci_reference_basis[lp].oa, tol);
    let occ_lb = occvec_to_bits(&noci_reference_basis[lp].ob, tol);
    let occ_ga = occvec_to_bits(&noci_reference_basis[gp].oa, tol);
    let occ_gb = occvec_to_bits(&noci_reference_basis[gp].ob, tol);

    let ph_a = excitation_phase(occ_la, &ex_la.holes, &ex_la.parts) * excitation_phase(occ_ga, &ex_ga.holes, &ex_ga.parts);
    let ph_b = excitation_phase(occ_lb, &ex_lb.holes, &ex_lb.parts) * excitation_phase(occ_gb, &ex_gb.holes, &ex_gb.parts);
    
    prepare_same(&w.aa, ex_la, ex_ga, scratch);
    let sa = ph_a * lg_overlap(&w.aa, ex_la, ex_ga, scratch);
    prepare_same(&w.bb, ex_lb, ex_gb, scratch);
    let sb = ph_b * lg_overlap(&w.bb, ex_lb, ex_gb, scratch);
    sa * sb
}

/// Calculate both the overlap and Hamiltonian matrix elements between determinants \Lambda and \Gamma 
/// using extended non-orthogonal Wick's theorem.
/// # Arguments:
///     `ao`: AoData struct, contains AO integrals and other system data. 
///     `determinants`: Vec<SCFState>, vector of all the determinants in the NOCI basis.
///     `noci_reference_basis`: Vec<SCFState>, vector of only the reference determinants.
///     `l`: usize, index of state \Lambda.
///     `g`: usize, index of state \Gamma.
pub fn calculate_hs_pair_wicks(ao: &AoData, determinants: &[SCFState], noci_reference_basis: &[SCFState], wicks: &WicksView, scratch: &mut WickScratch, 
                               l: usize, g: usize, tol: f64) -> (f64, f64) {

    let lp = determinants[l].parent;
    let gp = determinants[g].parent;

    let w = wicks.pair(lp, gp);

    let ex_la = &determinants[l].excitation.alpha;
    let ex_ga = &determinants[g].excitation.alpha;
    let ex_lb = &determinants[l].excitation.beta;
    let ex_gb = &determinants[g].excitation.beta;

    let occ_la = occvec_to_bits(&noci_reference_basis[lp].oa, tol);
    let occ_lb = occvec_to_bits(&noci_reference_basis[lp].ob, tol);
    let occ_ga = occvec_to_bits(&noci_reference_basis[gp].oa, tol);
    let occ_gb = occvec_to_bits(&noci_reference_basis[gp].ob, tol);

    let ph_a = excitation_phase(occ_la, &ex_la.holes, &ex_la.parts) * excitation_phase(occ_ga, &ex_ga.holes, &ex_ga.parts);
    let ph_b = excitation_phase(occ_lb, &ex_lb.holes, &ex_lb.parts) * excitation_phase(occ_gb, &ex_gb.holes, &ex_gb.parts);
    
    prepare_same(&w.aa, ex_la, ex_ga, scratch);
    let sa = ph_a * lg_overlap(&w.aa, ex_la, ex_ga, scratch);
    let h1a = lg_h1(&w.aa, ex_la, ex_ga, scratch, tol);
    let h2aa = lg_h2_same(&w.aa, ex_la, ex_ga, scratch, tol);

    prepare_same(&w.bb, ex_lb, ex_gb, scratch);
    let sb = ph_b * lg_overlap(&w.bb, ex_lb, ex_gb, scratch);
    let h1b = lg_h1(&w.bb, ex_lb, ex_gb, scratch, tol);
    let h2bb = lg_h2_same(&w.bb, ex_lb, ex_gb,scratch, tol);

    let h2ab = lg_h2_diff(&w, ex_la, ex_ga, ex_lb, ex_gb, scratch, tol);

    let s = sa * sb;
    let h1 = ph_a * h1a * sb + ph_b * h1b * sa;
    let h2 = (0.5 * ph_a * sb * h2aa) + (0.5 * ph_b * sa * h2bb) + (ph_a * ph_b * h2ab);
    
    let hnuc = if w.aa.m == 0 && w.bb.m == 0 {ao.enuc * s} else {0.0};

    (hnuc + h1 + h2, s)
}

/// Form the full Hamiltonian and overlap matrices using the generalised Slater-Condon rules.
/// # Arguments:
///     `ao`: AoData struct, contains AO integrals and other system data.
///     `determinants`: Vec<SCFState>, vector of all determinants in the NOCI basis.
///     `noci_reference_basis`: Vec<SCFState>, vector of only the reference determinants.
///     `input`: Input, user input specifications.
pub fn build_noci_matrices(ao: &AoData, input: &Input, determinants: &[SCFState], noci_reference_basis: &[SCFState], tol: f64, wicks: Option<&WicksView>) 
                           -> (Array2<f64>, Array2<f64>, Duration) {
    
    let ndets = determinants.len();
    let mut h = Array2::<f64>::zeros((ndets, ndets));
    let mut s = Array2::<f64>::zeros((ndets, ndets));

    // Build list of all upper-triangle and diagonal pairs \Lambda, \Gamma. 
    let pairs: Vec<(usize, usize)> = (0..ndets).flat_map(|mu| (mu..ndets).map(move |nu| (mu, nu))).collect();

    let wicks = if input.wicks.enabled || input.wicks.compare {Some(wicks.expect("Wick's requested but found wicks: None"))} else {None};

    // Calculate Hamiltonian matrix elements in parallel.
    let t_hs = Instant::now();
    let tmp: Vec<(usize, usize, f64, f64, f64)> = pairs.par_iter().map_init(WickScratch::new, |scratch, &(l, g)| {
        let (h, s, d) = if input.wicks.compare {
            let (hn, sn) = calculate_hs_pair_naive(ao, determinants, l, g, tol);
            let (hw, sw) = calculate_hs_pair_wicks(ao, determinants, noci_reference_basis, wicks.unwrap(), scratch, l, g, tol);
            let d = (hn - hw).abs() + (sn - sw).abs();
            (hw, sw, d)
        } else {
            let (h, s) = if input.wicks.enabled {
                calculate_hs_pair_wicks(ao, determinants, noci_reference_basis, wicks.unwrap(), scratch, l, g, tol)
            } else {
                calculate_hs_pair_naive(ao, determinants, l, g, tol)
            };
            (h, s, 0.0)
        };
        (l, g, h, s, d)}).collect();
    let d_hs = t_hs.elapsed();

    // Scatter Hamiltonian and overlap matrix elements into full matrices.
    let mut td = 0.0;
    for (l, g, lg_h, lg_s, d) in tmp {
        h[(l, g)] = lg_h;
        s[(l, g)] = lg_s;
        // Hermitian.
        if l != g {
            h[(g, l)] = lg_h;
            s[(g, l)] = lg_s;
        }
        td += d;
    }
    
    if input.wicks.compare{println!("Total naive–wicks discrepancy: {:.6e}", td);}

    // Write out the Hamiltonian and Overlap if requested.
    if input.write.write_matrices{
        create_dir_all(&input.write.write_dir).unwrap();
        let mut fhamiltonian = std::io::BufWriter::new(std::fs::File::create(format!("{}/HAMI", input.write.write_dir)).unwrap());
        for r in 0..h.nrows() {
            for c in 0..h.ncols() {
                if c > 0 { write!(fhamiltonian, " ").unwrap(); }
                write!(fhamiltonian, "{}", h[(r,c)]).unwrap();
            }
            writeln!(fhamiltonian).unwrap();
        }

        let mut foverlap = std::io::BufWriter::new(std::fs::File::create(format!("{}/OVLP", input.write.write_dir)).unwrap());
        for r in 0..s.nrows() {
            for c in 0..s.ncols() {
                if c > 0 { write!(foverlap, " ").unwrap(); }
                write!(foverlap, "{}", s[(r,c)]).unwrap();
            }
            writeln!(foverlap).unwrap();
        }
    }

    (h, s, d_hs)
}

/// Calculate NOCI energy by solving GEVP with NOCI Hamiltonian and overlap.
/// # Arguments:
///     `scfstates`: Vec<SCFState>, vector of all the calculated SCF states. 
///     `ao`: AoData struct, contains AO integrals and other system data.
///     `input`: Input, user input specifications.
pub fn calculate_noci_energy(ao: &AoData, input: &Input, scfstates: &[SCFState], tol: f64, wicks: Option<&WicksView>) -> (f64, Array1<f64>, Duration) {
    let (h, s, d_hs) = build_noci_matrices(ao, input, scfstates, scfstates, tol, wicks);
        
    println!("NOCI-reference Hamiltonian:");
    print_array2(&h);
    println!("NOCI-reference Overlap:");
    print_array2(&s);
    println!("Shifted NOCI-reference Hamiltonian");
    let h_shift = &h.map(|z: &f64| z) - scfstates[0].e * &s;
    print_array2(&h_shift);
    let (evals, c) = general_evp_real(&h, &s, true, f64::EPSILON);
    println!("GEVP eigenvalues in NOCI-reference basis: {}", evals);

    // Assumes columns of c are energy ordered eigenvectors
    let c0 = c.column(0).to_owned(); 
    (evals[0], c0, d_hs)
}

