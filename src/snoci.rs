//snoci.rs
use std::time::{Instant, Duration};

use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, UPLO};

use crate::{input::Input, AoData, SCFState};
use crate::nonorthogonalwicks::{WicksShared};

use crate::basis::generate_excited_basis;
use crate::noci::{build_noci_fock, build_noci_hs, build_noci_s, noci_density, update_wicks_fock};
use crate::maths::{general_evp_real, loewdin_x_real, orthogonaliser_real};
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

#[derive(Default)]
pub struct SNOCIStepTimings {
    pub current_hs: Duration,
    pub current_gevp: Duration,
    pub generate_candidates: Duration,
    pub candidate_hs: Duration,
    pub psuedoinvserse: Duration,
    pub s_omega: Duration,
    pub generalised_fock: Duration,
    pub update_wicks: Duration,
    pub candidate_f: Duration,
    pub f_omega: Duration,
    pub ortho_candidate: Duration,
    pub diagonalise_fock: Duration,
    pub channel_ept2: Duration,
    pub select: Duration,
}

pub fn snoci_step(ao: &AoData, current_space: &[SCFState], noci_reference_basis: &[SCFState], input: &Input, tol: f64, mut wicks: Option<&mut WicksShared>) -> (SNOCIState, SNOCIStepTimings) {

    let mut timings = SNOCIStepTimings::default();
    let opts = input.snoci.as_ref().unwrap();
    let mut current_space = current_space.to_vec();
    let mut final_state: Option<SNOCIState> = None;

    // RHF energy from which to define correlation.
    let e0 = noci_reference_basis[0].e;
    // Print table header.
    println!("{}", "=".repeat(100));
    println!("{:^6} {:^10} {:^11} {:^11} {:^16} {:^16} {:^16} {:^16}", "iter", "# Current", "# Candidate", "# Selected", "E", "Ecorr", "EPT2", "E + EPT2");

    for it in 0..opts.max_iter {
        // Solve NOCI GEVP in the current selected space.
        let t0 = Instant::now();
        let (hcurrent, scurrent, _) = build_noci_hs(ao, input, &current_space, &current_space, noci_reference_basis, tol, wicks.as_ref().map(|ws| ws.view()), true);
        timings.current_hs += t0.elapsed();
        
        let t0 = Instant::now();
        let (evals, c) = general_evp_real(&hcurrent, &scurrent, true, tol);
        let ecurrent = evals[0];
        let coeffs = c.column(0).to_owned();
        timings.current_gevp += t0.elapsed();
        
        // Generate single and double excitations of the current space.
        let t0 = Instant::now();
        let include_current = false;
        let candidates = generate_excited_basis(&current_space, input, include_current);
        timings.generate_candidates += t0.elapsed();

        // If the candidates list is empty we have nothing more to do. 
        if candidates.is_empty() {
            return (SNOCIState {ecurrent, coeffs, hcurrent, scurrent, candidates, selected: Vec::new(), candidate_scores: Vec::new(),
                               channel_denoms: Vec::new(), channel_pt2: Vec::new(), ept2: 0.0}, timings);
        }

        // Build candidate-candidate and candidate-current matrices. We use incides i, j to refer to
        // the current state space, and a, b (or \alpha and \beta) to refer to the candidate space.
        let t0 = Instant::now();
        let symmetric = true;
        let (sab, _) = build_noci_s(ao, input, &candidates, &candidates, noci_reference_basis, tol, wicks.as_ref().map(|ws| ws.view()), symmetric);
        let symmetric = false;
        let (hai, sai, _) = build_noci_hs(ao, input, &candidates, &current_space, noci_reference_basis, tol, wicks.as_ref().map(|ws| ws.view()), symmetric);
        let sia = sai.t().to_owned();
        timings.candidate_hs += t0.elapsed();

        // Find psuedoinvserse S^{ij}.
        let t0 = Instant::now();
        let x = loewdin_x_real(&scurrent, true, tol);
        let sij_inv = x.dot(&x);
        timings.psuedoinvserse += t0.elapsed();

        // \hat P | \Phi_\alpha \rangle = \sum_{i, j} | \Psi_i \rangle S^{ij} \langle \Psi_j |
        // \Psi_\alpha \rangle = \sum_{i, j} | \Psi_i \rangle S^{ij} S_{j \alpha}. In order to form
        // matrices we left project with \langle \Phi_\beta | such that we get \langle \Phi_\beta |
        // \hat P | \Phi_\alpha \rangle = S_{\beta i} S^{ij} S_{j \alpha}.
        let t0 = Instant::now();
        let sbi_ij_ja = sai.dot(&sij_inv.dot(&sia));
        // \hat Q = \hat I - \hat P. We label S with \hat Q applied with \Omega.
        let s_omega_ab = &sab - &sbi_ij_ja;
        timings.s_omega += t0.elapsed();

        // Get multireference NOCI density and form generalised Fock matrix.
        let t0 = Instant::now();
        let (da, db) = noci_density(ao, &current_space, &coeffs, tol);
        let (fa, fb) = form_fock_matrices(&ao.h, &ao.eri_coul, &da, &db);
        timings.generalised_fock += t0.elapsed();

        // Update Wick's intermediates for Fock matrix element calculation if using. 
        let t0 = Instant::now();
        if input.wicks.enabled && let Some(ws) = wicks.as_deref_mut() {
            update_wicks_fock(&fa, &fb, noci_reference_basis, ws);
        }
        timings.update_wicks += t0.elapsed();

        // Calculate current-current, candidate-current and candidate-candidate Fock matrix elements.
        let t0 = Instant::now();
        let (fii, _) = build_noci_fock(ao, &current_space, &current_space, &fa, &fb, noci_reference_basis, wicks.as_ref().map(|ws| ws.view()), tol, true, input);
        let (fai, _) = build_noci_fock(ao, &candidates, &current_space, &fa, &fb, noci_reference_basis, wicks.as_ref().map(|ws| ws.view()), tol, false, input);
        let fia = fai.t().to_owned();
        let (fab, _) = build_noci_fock(ao, &candidates, &candidates, &fa, &fb, noci_reference_basis, wicks.as_ref().map(|ws| ws.view()), tol, true, input);
        timings.candidate_f += t0.elapsed();

        // \Omega space Fock. 
        // \hat H_0$ as $\hat H_0 = \hat M \hat F \hat M + \hat Q_{\text{state}} \hat F \hat Q_{\text{state}}.
        let t0 = Instant::now();
        let f_omega_ab = &fab - &fai.dot(&sij_inv.dot(&sia)) - &sai.dot(&sij_inv.dot(&fia)) + &sai.dot(&sij_inv.dot(&fii.dot(&sij_inv.dot(&sia))));

        let v_omega = (&hai - &sai.dot(&sij_inv.dot(&hcurrent))).dot(&coeffs);
        timings.f_omega += t0.elapsed();

        // Orthognalise projected candidate space and transform quantities into it.
        let t0 = Instant::now();
        let x_omega = orthogonaliser_real(&s_omega_ab, tol);
        let f_tilde = x_omega.t().dot(&f_omega_ab.dot(&x_omega));
        let v_tilde = x_omega.t().dot(&v_omega);
        timings.ortho_candidate += t0.elapsed();

        // Diagonalise transformed Fock and re-transform.
        let t0 = Instant::now();
        let n = f_tilde.nrows();
        let (f_evals, y) = f_tilde.eigh(UPLO::Lower).unwrap();
        let delta = f_evals.mapv(|x| x - ecurrent);
        let v_bar = y.t().dot(&v_tilde);
        let z = x_omega.dot(&y);
        timings.diagonalise_fock += t0.elapsed();

        // Channel contributions and EPT2.
        let t0 = Instant::now();
        let mut channel_denoms = Vec::with_capacity(delta.len());
        let mut channel_pt2 = Vec::with_capacity(delta.len());
        for a in 0..delta.len() {
            let da = delta[a];
            let va = v_bar[a];
            channel_denoms.push(da);
            channel_pt2.push(if da.abs() > tol {-(va * va) / da} else {0.0});
        }
        let ept2: f64 = channel_pt2.iter().sum();
        timings.channel_ept2 += t0.elapsed();

        // Candidate scoring.
        let t0 = Instant::now();
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
        timings.select += t0.elapsed();

        println!("{:<6} {:>10} {:>11} {:>11} {:>16.12} {:>16.12} {:>16.12} {:>16.12}", 
                it, current_space.len(), candidates.len(), selected.len(), ecurrent, ecurrent - e0, ept2, ecurrent + ept2);

        let state = SNOCIState {ecurrent, coeffs, hcurrent, scurrent, candidates, selected, candidate_scores, channel_denoms, channel_pt2, ept2};

        // Convergence or stopping.
        if state.selected.is_empty() {
            println!("SNOCI stopped at iteration {}: no candidates satisfied the selection threshold ({}).", it, opts.sigma);
            return (state, timings);
        }
        if state.ept2.abs() < opts.tol {
            println!("SNOCI stopped at iteration {}: |EPT2|: {:.12} fell below tolerance {:.12}.", it, state.ept2.abs(), opts.tol);
            return (state, timings);
        }

        current_space.extend(state.selected.iter().cloned());
        final_state = Some(state);
    }
    println!("SNOCI stopped: Maximum iteration was reached ({}).", opts.max_iter);
    (final_state.unwrap(), timings)
}
