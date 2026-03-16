//snoci.rs
use ndarray::{Array1, Array2};

use crate::{input::Input, AoData, SCFState};
use crate::nonorthogonalwicks::WicksView;

use crate::basis::generate_excited_basis;
use crate::noci::{build_noci_hs, build_noci_fock, noci_density};
use crate::maths::{general_evp_real, loewdin_x_real};
use crate::scf::form_fock_matrices;

pub struct SNOCIState {
    pub ecurrent: f64,
    pub coeffs: Array1<f64>,
    pub hcurrent: Array2<f64>,
    pub scurrent: Array2<f64>,
    pub candidates: Vec<SCFState>,
    pub selected: Vec<SCFState>,
    pub candidate_scores: Vec<f64>,
    pub channel_denoms: Vec<f64>,
    pub channel_pt2: Vec<f64>,
    pub ept2: f64,
}

pub fn snoci_step(ao: &AoData, current_space: &[SCFState], noci_reference_basis: &[SCFState], input: &Input, tol: f64, wicks: Option<&WicksView>) -> SNOCIState {

    // Solve NOCI GEVP in the current selected space.
    let (hcurrent, scurrent, _) = build_noci_hs(ao, input, current_space, current_space, noci_reference_basis, tol, wicks, true);
    let (evals, c) = general_evp_real(&hcurrent, &scurrent, true, tol);
    let ecurrent = evals[0];
    let coeffs = c.column(0).to_owned();
    
    // Generate single and double excitations of the current space.
    let include_current = false;
    let candidates = generate_excited_basis(current_space, input, include_current);

    // If the candidates list is empty we have nothing more to do. 
    if candidates.is_empty() {
        return SNOCIState {ecurrent, coeffs, hcurrent, scurrent, candidates, selected: Vec::new(), candidate_scores: Vec::new(),
                           channel_denoms: Vec::new(), channel_pt2: Vec::new(), ept2: 0.0};
    }

    // Build candidate-candidate and candidate-current matrices. We use incides i, j to refer to
    // the current state space, and a, b (or \alpha and \beta) to refer to the candidate space.
    let symmetric = true;
    let (_, sab, _) = build_noci_hs(ao, input, &candidates, &candidates, noci_reference_basis, tol, wicks, symmetric);
    let symmetric = false;
    let (hai, sai, _) = build_noci_hs(ao, input, &candidates, current_space, noci_reference_basis, tol, wicks, symmetric);
    let sia = sai.t().to_owned();

    // Find psuedoinvserse S^{ij}.
    let x = loewdin_x_real(&scurrent, true, tol);
    let sij_inv = x.dot(&x);

    // \hat P | \Phi_\alpha \rangle = \sum_{i, j} | \Psi_i \rangle S^{ij} \langle \Psi_j |
    // \Psi_\alpha \rangle = \sum_{i, j} | \Psi_i \rangle S^{ij} S_{j \alpha}. In order to form
    // matrices we left project with \langle \Phi_\beta | such that we get \langle \Phi_\beta |
    // \hat P | \Phi_\alpha \rangle = S_{\beta i} S^{ij} S_{j \alpha}.
    let sbi_ij_ja = sai.dot(&sij_inv.dot(&sia));
    // \hat Q = \hat I - \hat P. We label S with \hat Q applied with \Omega.
    let s_omega_ab = &sab - &sbi_ij_ja;

    // Calculate density of the current space NOCI wavefunction and get the generalised Fock
    // between the spaces.
    let (da, db) = noci_density(ao, current_space, &coeffs, tol);
    let (fa, fb) = form_fock_matrices(&ao.h, &ao.eri_coul, &da, &db);
    let (fii, _) = build_noci_fock(ao, current_space, current_space, &fa, &fb, tol, true, input);
    let (fai, _) = build_noci_fock(ao, &candidates, current_space, &fa, &fb, tol, false, input);
    let fia = fai.t().to_owned();
    let (fab, _) = build_noci_fock(ao, &candidates, &candidates, &fa, &fb, tol, true, input);

    // \Omega space Fock. 
    // \hat H_0$ as $\hat H_0 = \hat M \hat F \hat M + \hat Q_{\text{state}} \hat F \hat Q_{\text{state}}.
    let f_omega_ab = &fab - &fai.dot(&sij_inv.dot(&sia)) - &sai.dot(&sij_inv.dot(&fia)) + &sai.dot(&sij_inv.dot(&fii.dot(&sij_inv.dot(&sia))));

    let v_omega = (&hai - &sai.dot(&sij_inv.dot(&hcurrent))).dot(&coeffs);

    // Orthognalise projected candidate space and transform quantities into it.
    let x_omega = loewdin_x_real(&s_omega_ab, true, tol);
    let f_tilde = x_omega.t().dot(&f_omega_ab.dot(&x_omega));
    let v_tilde = x_omega.t().dot(&v_omega);

    // Diagonalise transformed Fock and re-transform.
    let n = f_tilde.nrows();
    let ident = Array2::from_diag(&Array1::ones(n));
    let (f_evals, y) = general_evp_real(&f_tilde, &ident, true, tol);
    let delta = f_evals.mapv(|x| x - ecurrent);
    let v_bar = y.t().dot(&v_tilde);
    let z = x_omega.dot(&y);

    // Channel contributions and EPT2.
    let mut channel_denoms = Vec::with_capacity(delta.len());
    let mut channel_pt2 = Vec::with_capacity(delta.len());
    for a in 0..delta.len() {
        let da = delta[a];
        let va = v_bar[a];
        channel_denoms.push(da);
        channel_pt2.push(if da.abs() > tol {-(va * va) / da} else {0.0});
    }

    let ept2: f64 = channel_pt2.iter().sum();

    // Candidate scoring.
    let mut candidate_scores = vec![0.0; candidates.len()];
    for alpha in 0..candidates.len() {
        let mut score = 0.0;
        for a in 0..channel_pt2.len() {
            score += channel_pt2[a].abs() * z[(alpha, a)].powi(2);
        }
        candidate_scores[alpha] = score;
    }

    let opts = input.snoci.as_ref().unwrap();

    let mut ranked: Vec<(SCFState, f64)> = candidates.iter().cloned().zip(candidate_scores.iter().copied()).filter(|(_, score)| *score > opts.sigma).collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    ranked.truncate(opts.max_add);
    let selected: Vec<SCFState> = ranked.into_iter().map(|(state, _)| state).collect();

    return SNOCIState {ecurrent, coeffs, hcurrent, scurrent, candidates, selected, candidate_scores, channel_denoms, channel_pt2, ept2};
}

