// noci.rs
use std::time::{Duration, Instant};
use std::ptr::NonNull;

use ndarray::{Array1, Array2, Array4, Axis};
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
use crate::write::write_hs_matrices;

// Trait which defines how returned determinant-pair quatity should be scattered into matrices.
// Used such that we can have generic scatter functions which return 1 or 2 matrices.
trait ScatterValue: Sized {
    type Output;
    /// Construct zero initialised output. 
    /// # Arguments:
    ///     `nl`: usize, length of determinant set 1.
    ///     `nr`: usize, length of determinant set 2.
    fn zeros(nl: usize, nr: usize) -> Self::Output;
    /// Write a value into the output at indices i, j.
    /// # Arguments:
    ///     `out`: Self::Output, output container to write into.
    ///     `i`: usize, row index.
    ///     `j`: usize, column index.
    ///     `val`: Self, determinant-pair value to scatter.
    fn write(out: &mut Self::Output, i: usize, j: usize, val: Self);
}

impl ScatterValue for f64 {
    type Output = Array2<f64>;
    /// Construct zero initialised output. 
    /// # Arguments:
    ///     `nl`: usize, length of determinant set 1.
    ///     `nr`: usize, length of determinant set 2.
    fn zeros(nl: usize, nr: usize) -> Self::Output {
        Array2::<f64>::zeros((nl, nr))
    }
    /// Write scalar value into matrix position (i, j).
    /// # Arguments:
    ///     `out`: Array2<f64>, output matrix.
    ///     `i`: usize, row index.
    ///     `j`: usize, column index.
    ///     `val`: f64, matrix element value.
    fn write(out: &mut Self::Output, i: usize, j: usize, val: Self) {
        out[(i, j)] = val;
    }
}

impl ScatterValue for (f64, f64) {
    type Output = (Array2<f64>, Array2<f64>);

    /// Construct zero initialised output. 
    /// # Arguments:
    ///     `nl`: usize, length of determinant set 1.
    ///     `nr`: usize, length of determinant set 2.
    fn zeros(nl: usize, nr: usize) -> Self::Output {
        (Array2::<f64>::zeros((nl, nr)), Array2::<f64>::zeros((nl, nr)))
    }

    /// Write scalar value into matrix position (i, j) in both matrices.
    /// # Arguments:
    ///     `out`: (Array2<f64>, Array2<f64>), output matrices.
    ///     `i`: usize, row index.
    ///     `j`: usize, column index.
    ///     `val`: (f64, f64), matrix element values.
    fn write(out: &mut Self::Output, i: usize, j: usize, val: Self) {
        out.0[(i, j)] = val.0;
        out.1[(i, j)] = val.1;
    }
}

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

/// Calculate pair density {}^{\Lambda\Gamma} \rho_{ij} using the generalised Slater-Condon rules.
/// # Arguments:
///     `pair`: Pair struct, contains data concerning a pair of determinants.
///     `nao`: usize, number of AOs.
fn pair_density(pair: &Pair, nao: usize) -> Array2<f64> {
    match pair.zeros.len() {
        0 => pair.w.as_ref().unwrap().mapv(|x| x * pair.s_red * pair.phase),
        1 => pair.p_i.as_ref().unwrap().mapv(|x| x * pair.s_red * pair.phase),
        _ => Array2::zeros((nao, nao)),
    }
}

/// Calculate density matrix of the multi-reference NOCI state.
/// # Arguments:
///     `ao`: AoData struct, contains AO integrals and other system data.
///     `c`: Array1, coefficients of the NOCI wavefunction.
///     `states`: [SCFState], basis of states for the wavefunction.
///     `densitites`: [Array2], pair density matrices.
pub fn noci_density(ao: &AoData, states: &[SCFState], c: &Array1<f64>, tol: f64,) -> (Array2<f64>, Array2<f64>) {
    let nao = ao.h.nrows();
    let mut da = Array2::<f64>::zeros((nao, nao));
    let mut db = Array2::<f64>::zeros((nao, nao));

    for i in 0..states.len() {
        for j in 0..states.len() {
            let ldet = &states[i];
            let gdet = &states[j];

            let l_ca_occ = occ_coeffs(&ldet.ca, &ldet.oa);
            let g_ca_occ = occ_coeffs(&gdet.ca, &gdet.oa);
            let l_cb_occ = occ_coeffs(&ldet.cb, &ldet.ob);
            let g_cb_occ = occ_coeffs(&gdet.cb, &gdet.ob);

            let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &ao.s, tol);
            let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &ao.s, tol);

            let rhoa = pair_density(&pa, nao);
            let rhob = pair_density(&pb, nao);

            let cij = c[i] * c[j];
            da.scaled_add(cij * pb.s, &rhoa);
            db.scaled_add(cij * pa.s, &rhob);
        }
    }
    (da, db)
}

/// Calculate one body matrix elements using the generalised 
/// Slater-Condon rules for a pair of determinants \Lambda and \Gamma. 
/// # Arguments:
///     `o`: Array2, operator to obtain matrix elements of.
///     `pair`: Pair struct, contains data concerning a pair of determinants.
fn one_electron(o: &Array2<f64>, pair: &Pair) -> f64 {
    match pair.zeros.len() {
        // With no zeros (s_i != 0 for all i) we use munu_w.
        0 => pair.s_red * pair.phase * einsum_ba_ab_real(pair.w.as_ref().unwrap(), o),
        // With 1 zero (s_i = 0 for 1 i) we use P_i.
        1 => pair.s_red * pair.phase * einsum_ba_ab_real(pair.p_i.as_ref().unwrap(), o),
        // Otherwise the matrix element is zero.
        _ => 0.0,
    }
}

/// Calculate two body matrix elements for electrons of the same spin 
/// using the generalised Slater-Condon rules for a pair of determinants \Lambda and \Gamma.
/// # Arguments:
///     `o`: Array2, operator to obtain matrix elements of.
///     `pair`: Pair struct, contains data concerning a pair of determinants.
fn two_electron_same(o: &Array4<f64>, pair: &Pair) -> f64 {
    match pair.zeros.len() {
        // With no zeros (s_i != 0 for all i) we use munu_w on both sides.
        0 => 0.5 * pair.s_red * pair.phase * einsum_ba_abcd_cd_real(pair.w.as_ref().unwrap(), o, pair.w.as_ref().unwrap()),
        // With 1 zero (s_i = 0 for one index i) we use P_i on one side.
        1 => pair.s_red * pair.phase * einsum_ba_abcd_cd_real(pair.w.as_ref().unwrap(), o, pair.p_i.as_ref().unwrap()),
        // with 2 zeros (s_i, s_j = 0 for two indices i, j) we use P_i on one side and P_j on the other.
        2 => pair.s_red * pair.phase * einsum_ba_abcd_cd_real(pair.p_i.as_ref().unwrap(), o, pair.p_j.as_ref().unwrap()),
        // Otherwise the matrix element is 0.
        _ => 0.0,
    }
}

/// Calculate two body matrix elements for electrons of opposite spin 
/// using the generalised Slater-Condon rules for a pair of determinants \Lambda and \Gamma.
/// # Arguments:
///     `o`: Array2, operator to obtain matrix elements of.
///     `pair`: Pair struct, contains data concerning a pair of determinants.
fn two_electron_diff(o: &Array4<f64>, pa: &Pair, pb: &Pair) -> f64 {
    match (pa.zeros.len(), pb.zeros.len()) {
        // With no zeros (s_i != 0 for all i) for both spins we use munu_w on both sides.
        (0, 0) => (pa.s_red * pa.phase) * (pb.s_red * pb.phase) * einsum_ba_abcd_cd_real(pa.w.as_ref().unwrap(), o, pb.w.as_ref().unwrap()),
        // With one zero in beta spin only we use W^a and P_i^b.
        (0, 1) => (pa.s_red * pa.phase) * (pb.s_red * pb.phase) * einsum_ba_abcd_cd_real(pa.w.as_ref().unwrap(), o, pb.p_i.as_ref().unwrap()),
        // With one zero in alpha spin only we use P_i^a and W_b.
        (1, 0) => (pa.s_red * pa.phase) * (pb.s_red * pb.phase) * einsum_ba_abcd_cd_real(pa.p_i.as_ref().unwrap(), o, pb.w.as_ref().unwrap()),
        // With one zero in both spins we use P_i^a and P_i^b.
        (1, 1) => (pa.s_red * pa.phase) * (pb.s_red * pb.phase) * einsum_ba_abcd_cd_real(pa.p_i.as_ref().unwrap(), o, pb.p_i.as_ref().unwrap()),
        // Otherwise the matrix element is zero.
        _ => 0.0,
    }
}

/// Calculate the overlap matrix element between determinants \Lambda and \Gamma using 
/// generalised Slater-Condon rules.
/// # Arguments:
///     `ao`: AoData struct, contains AO integrals and other system data. 
///     `ldet`: SCFState, state \Lambda.
///     `gdet`: SCFState, state \Gamma.
pub fn calculate_s_pair_naive(ao: &AoData, ldet: &SCFState, gdet: &SCFState, tol: f64) -> f64 {

    // Per spin occupid coefficients.
    let l_ca_occ = occ_coeffs(&ldet.ca, &ldet.oa);
    let g_ca_occ = occ_coeffs(&gdet.ca, &gdet.oa);
    let l_cb_occ = occ_coeffs(&ldet.cb, &ldet.ob);
    let g_cb_occ = occ_coeffs(&gdet.cb, &gdet.ob);

    let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &ao.s, tol);
    let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &ao.s, tol);

    // Overlap matrix element for this pair. 
    pa.s * pb.s
}

/// Calculate the Fock matrix element between determinants \Lambda and \Gamma using 
/// generalised Slater-Condon rules.
/// # Arguments:
///     `ao`: AoData struct, contains AO integrals and other system data. 
///     `ldet`: SCFState, state \Lambda.
///     `gdet`: SCFState, state \Gamma.
///     `fa`: Array2, NOCI Fock matrix spin alpha.
///     `fb`: Array2, NOCI Fock matrix spin beta.
pub fn calculate_f_pair_naive(fa: &Array2<f64>, fb: &Array2<f64>, ao: &AoData, ldet: &SCFState, gdet: &SCFState, tol: f64) -> f64 {

    // Per spin occupid coefficients.
    let l_ca_occ = occ_coeffs(&ldet.ca, &ldet.oa);
    let g_ca_occ = occ_coeffs(&gdet.ca, &gdet.oa);
    let l_cb_occ = occ_coeffs(&ldet.cb, &ldet.ob);
    let g_cb_occ = occ_coeffs(&gdet.cb, &gdet.ob);

    let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &ao.s, tol);
    let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &ao.s, tol);

    pb.s * one_electron(fa, &pa) + pa.s * one_electron(fb, &pb)
}

/// Calculate both the overlap and Hamiltonian matrix elements between determinants \Lambda and \Gamma 
/// using generalised Slater-Condon rules.
/// # Arguments:
///     `ao`: AoData struct, contains AO integrals and other system data. 
///     `ldet`: SCFState, state \Lambda.
///     `gdet`: SCFState, state \Gamma.
pub fn calculate_hs_pair_naive(ao: &AoData, ldet: &SCFState, gdet: &SCFState, tol: f64) -> (f64, f64) {

    // Per spin occupid coefficients.
    let l_ca_occ = occ_coeffs(&ldet.ca, &ldet.oa);
    let g_ca_occ = occ_coeffs(&gdet.ca, &gdet.oa);
    let l_cb_occ = occ_coeffs(&ldet.cb, &ldet.ob);
    let g_cb_occ = occ_coeffs(&gdet.cb, &gdet.ob);

    let pa = build_s_pair(&l_ca_occ, &g_ca_occ, &ao.s, tol);
    let pb = build_s_pair(&l_cb_occ, &g_cb_occ, &ao.s, tol);

    // Overlap matrix element for this pair. 
    let s = pa.s * pb.s;
    
    let hnuc = match (pa.zeros.len(), pb.zeros.len()) {
        (0, 0) => ao.enuc * s,
        _ => 0.0,
    };

    let h1a = one_electron(&ao.h, &pa);
    let h1b = one_electron(&ao.h, &pb);
    let h1 = pb.s * h1a + pa.s * h1b;

    let h2aa = pb.s * two_electron_same(&ao.eri_asym, &pa); 
    let h2bb = pa.s * two_electron_same(&ao.eri_asym, &pb); 
    let h2ab = two_electron_diff(&ao.eri_coul, &pa, &pb);
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
    let bytes = tensor_len.checked_mul(std::mem::size_of::<f64>()).unwrap();
    // Number of AOs same as MOs provided we don't freeze anything.
    println!("Number of MOs: {}", ao.n);
    println!("Estimated memory required for Wick's intermediates (MiB): {}", bytes as f64 / (1024.0 * 1024.0));

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
                println!("Building intermediates for reference pair: {}, {} on world rank {}", i, j, world.rank());

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
///     `noci_reference_basis`: Vec<SCFState>, vector of only the reference determinants.
///     `ldet`: SCFState, state \Lambda.
///     `gdet`: SCFState, state \Gamma.
pub fn calculate_s_pair_wicks(noci_reference_basis: &[SCFState], ldet: &SCFState, gdet: &SCFState, tol: f64, wicks: &WicksView, scratch: &mut WickScratch) -> f64 {

    let lp = ldet.parent;
    let gp = gdet.parent;

    let w = &wicks.pair(lp, gp);

    let ex_la = &ldet.excitation.alpha;
    let ex_ga = &gdet.excitation.alpha;
    let ex_lb = &ldet.excitation.beta;
    let ex_gb = &gdet.excitation.beta;

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
///     `noci_reference_basis`: Vec<SCFState>, vector of only the reference determinants.
///     `ldet`: SCFState, state \Lambda.
///     `gdet`: SCFState, state \Gamma.
pub fn calculate_hs_pair_wicks(ao: &AoData, noci_reference_basis: &[SCFState], ldet: &SCFState, gdet: &SCFState, tol: f64, wicks: &WicksView, scratch: &mut WickScratch) -> (f64, f64) {

    let lp = ldet.parent;
    let gp = gdet.parent;

    let w = &wicks.pair(lp, gp);

    let ex_la = &ldet.excitation.alpha;
    let ex_ga = &gdet.excitation.alpha;
    let ex_lb = &ldet.excitation.beta;
    let ex_gb = &gdet.excitation.beta;

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

    let h2ab = lg_h2_diff(w, ex_la, ex_ga, ex_lb, ex_gb, scratch, tol);

    let s = sa * sb;
    let h1 = ph_a * h1a * sb + ph_b * h1b * sa;
    let h2 = (0.5 * ph_a * sb * h2aa) + (0.5 * ph_b * sa * h2bb) + (ph_a * ph_b * h2ab);
    
    let hnuc = if w.aa.m == 0 && w.bb.m == 0 {ao.enuc * s} else {0.0};

    (hnuc + h1 + h2, s)
}

/// Evaluate arbitrary determinant pair quantity given a closure `o` which computes `T` for the
/// pair. The closure may be for example the Hamiltonian, overlap or Fock matrix element routines.
/// # Arguments:
///     `left`: [SCFState], first set of determinants.  
///     `right`: [SCFState], second set of determinants.
///     `symmetric`: bool, is the matrix to be evaluated symmetric.
///     `o`: O, closure function for matrix element routine.
///     `input`: Input, user specified input options.
/// # Type Parameters:
///     `O`: closure type implementing `Fn(&SCFState, &SCFState) -> T` and `Sync`.
///     `T`: Send, generic type for return which must be Send for parallelisation.
fn calculate_matrix_elements<T, O> (left: &[SCFState], right: &[SCFState], input: &Input, symmetric: bool, o: O) -> (Vec<(usize, usize, T)>, Duration)
    where T: Send, O: Fn(&SCFState, &SCFState, Option<&mut WickScratch>) -> T + Sync {

    let nl = left.len();
    let nr = right.len();
    
    // Build list of all upper-triangle and diagonal pairs \Lambda, \Gamma.
    let pairs: Vec<(usize, usize)> = if symmetric {(0..nl).flat_map(|i| (i..nr).map(move |j| (i, j))).collect()} 
    else {(0..nl).flat_map(|i| (0..nr).map(move |j| (i, j))).collect()};
    
    let t0 = Instant::now();
    let vals = if input.wicks.enabled {
        pairs.par_iter().map_init(WickScratch::new, |scratch, &(i, j)| {(i, j, o(&left[i], &right[j], Some(scratch)))}).collect()
    } else {
         pairs.par_iter().map(|&(i, j)| (i, j, o(&left[i], &right[j], None))).collect()
    };
    let dt = t0.elapsed();

    (vals, dt)
}

/// Scatter matrix elements into 2D Array.
/// # Arguments:
///     `vals`: Vec<(usize, usize, f64)>, matrix elements and indices.
///     `nl`: usize, length of determinant set 1.
///     `nl`: usize, length of determinant set 2.
///     `symmetric`: bool, is the matrix symmetric.
/// # Type Parameters: 
///     `T`: Matrix element value type to scatter. Can be f64 or (f64, f64).
fn scatter_matrix_elements<T>(vals: Vec<(usize, usize, T)>, nl: usize, nr: usize, symmetric: bool) -> T::Output 
    where T: ScatterValue + Copy {
    let mut out = T::zeros(nl, nr);
    for (i, j, val) in vals {
        T::write(&mut out, i, j, val);
        if symmetric && i != j {
            T::write(&mut out, j, i, val);
        }
    }
    out
}

/// Construct full NOCI Fock matrix using generalised Slater-Condon rules.
/// # Arguments:
///     `ao`: AoData struct, contains AO integrals and other system data.
///     `left`: [SCFState], first set of determinants.  
///     `right`: [SCFState], second set of determinants.
///     `fa`: Array2, NOCI Fock matrix spin alpha.
///     `fb`: Array2, NOCI Fock matrix spin beta.
///     `symmetric`: bool, is the matrix symmetric.
///     `tol`: f64, tolerance up to which a number is considered zero.
///     `input`: Input, user specified input options.
pub fn build_noci_fock(ao: &AoData, left: &[SCFState], right: &[SCFState], fa: &Array2<f64>, fb: &Array2<f64>, tol: f64, symmetric: bool, input: &Input) -> (Array2<f64>, Duration) {
    let nl = left.len();
    let nr = right.len();
    let (vals, dt) = calculate_matrix_elements(left, right, input, symmetric, |ldet, gdet, _scratch| {calculate_f_pair_naive(fb, fa, ao, ldet, gdet, tol)});
    let f = scatter_matrix_elements(vals, nl, nr, symmetric);
    (f, dt)
}

/// Form the full Hamiltonian and overlap matrices using either generalised Slater-Condon rules or
/// extended non-orthogonal Wick's theorem. 
/// # Arguments:
///     `ao`: AoData struct, contains AO integrals and other system data.
///     `left`: [SCFState], first set of determinants.  
///     `right`: [SCFState], second set of determinants.
///     `noci_reference_basis`: Vec<SCFState>, vector of only the reference determinants.
///     `input`: Input, user input specifications.
pub fn build_noci_hs(ao: &AoData, input: &Input, left: &[SCFState], right: &[SCFState], noci_reference_basis: &[SCFState], tol: f64, wicks: Option<&WicksView>, symmetric: bool) 
                     -> (Array2<f64>, Array2<f64>, Duration) {

    let nl = left.len();
    let nr = right.len();

    let wicks = if input.wicks.enabled || input.wicks.compare {Some(wicks.expect("Wick's requested but found wicks: None"))} else {None};

    if input.wicks.compare {
        let (vals, dt) = calculate_matrix_elements(left, right, input, symmetric, |ldet, gdet, scratch| {
            let (hn, sn) = calculate_hs_pair_naive(ao, ldet, gdet, tol);
            let (hw, sw) = calculate_hs_pair_wicks(ao, noci_reference_basis, ldet, gdet, tol, wicks.unwrap(), scratch.unwrap());
            ((hw, sw), (hn - hw).abs() + (sn - sw).abs())
        });

        let mut td = 0.0;
        let mut hsvals = Vec::with_capacity(vals.len());
        for (i, j, (hs, d)) in vals {
            hsvals.push((i, j, hs));
            td += d;
        }
        println!("Total naive–wicks discrepancy: {:.6e}", td);
        let (h, s) = scatter_matrix_elements(hsvals, nl, nr, symmetric);
        return (h, s, dt);
    }

    let (vals, dt) = calculate_matrix_elements(left, right, input, symmetric, |ldet, gdet, scratch| {
        if input.wicks.enabled {
            calculate_hs_pair_wicks(ao, noci_reference_basis, ldet, gdet, tol, wicks.unwrap(), scratch.unwrap())
        } else {
            calculate_hs_pair_naive(ao, ldet, gdet, tol)
        }
    });

    let (h, s) = scatter_matrix_elements(vals, nl, nr, symmetric);

    if input.write.write_matrices {
        write_hs_matrices(&input.write.write_dir, &h, &s);
    }

    (h, s, dt)
}

/// Calculate NOCI energy by solving GEVP with NOCI Hamiltonian and overlap.
/// # Arguments:
///     `scfstates`: Vec<SCFState>, vector of all the calculated SCF states. 
///     `ao`: AoData struct, contains AO integrals and other system data.
///     `input`: Input, user input specifications.
pub fn calculate_noci_energy(ao: &AoData, input: &Input, scfstates: &[SCFState], tol: f64, wicks: Option<&WicksView>) -> (f64, Array1<f64>, Duration) {
    let (h, s, d_hs) = build_noci_hs(ao, input, scfstates, scfstates, scfstates, tol, wicks, true);
    
    println!("{}", "=".repeat(100));
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

