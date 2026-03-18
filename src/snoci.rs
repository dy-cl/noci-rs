//snoci.rs
use std::time::{Instant, Duration};

use ndarray::{Array1, Array2};
use rayon::prelude::*;

use crate::{input::Input, AoData, SCFState};
use crate::nonorthogonalwicks::{WicksShared};

use crate::basis::generate_excited_basis;
use crate::noci::{build_noci_fock, build_noci_hs, build_noci_s, noci_density, update_wicks_fock};
use crate::maths::{general_evp_real, loewdin_x_real, parallel_matvec_real};
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
    pub davidson: Duration,
    pub channel_ept2: Duration,
    pub select: Duration,
}

fn seed(m: &Array2<f64>, s: &Array2<f64>, nroots: usize, tol: f64,) -> Vec<Array1<f64>> {
    let n = m.nrows();
    let mdiag = m.diag().to_owned();
    let sdiag = s.diag().to_owned();

    let mut order: Vec<usize> = (0..n).filter(|&i| sdiag[i] > tol).collect();
    order.sort_by(|&i, &j| {
        let qi = mdiag[i] / sdiag[i];
        let qj = mdiag[j] / sdiag[j];
        qi.partial_cmp(&qj).unwrap()
    });

    let seeddim = order.len().min((4 * nroots).max(25));
    let idx = &order[..seeddim];

    let mut m0 = Array2::<f64>::zeros((seeddim, seeddim));
    let mut s0 = Array2::<f64>::zeros((seeddim, seeddim));
    for p in 0..seeddim {
        for q in 0..seeddim {
            m0[(p, q)] = m[(idx[p], idx[q])];
            s0[(p, q)] = s[(idx[p], idx[q])];
        }
    }

    let (evals, c0) = general_evp_real(&m0, &s0, true, tol);
    let keep = nroots.min(evals.len());

    let mut basis: Vec<Array1<f64>> = Vec::new();
    let mut sbasis: Vec<Array1<f64>> = Vec::new();
    for a in 0..keep {
        let mut v = Array1::<f64>::zeros(n);
        for p in 0..seeddim {
            v[idx[p]] = c0[(p, a)];
        }
        if let Some((u, su)) = orthonormalise(v, &basis, &sbasis, s, tol) {
            basis.push(u);
            sbasis.push(su);
        }
    }
    basis
}

fn form_vector(basis: &[Array1<f64>], coeffs: &Array1<f64>) -> Array1<f64> {
    let n = basis[0].len();
    let mut v = Array1::<f64>::zeros(n);
    for (u, &c) in basis.iter().zip(coeffs.iter()) {
        v.scaled_add(c, u);
    }
    v
}

fn project_subspace(basis: &[Array1<f64>], mbasis: &[Array1<f64>], sbasis: &[Array1<f64>],) -> (Array2<f64>, Array2<f64>) {
    let k = basis.len();

    let rows: Vec<(Vec<f64>, Vec<f64>)> = (0..k).into_par_iter().map(|p| {
        let mut mrow = vec![0.0; p + 1];
        let mut srow = vec![0.0; p + 1];
        for q in 0..=p {
            mrow[q] = basis[p].dot(&mbasis[q]);
            srow[q] = basis[p].dot(&sbasis[q]);
        }
        (mrow, srow)
    }).collect();

    let mut msub = Array2::<f64>::zeros((k, k));
    let mut ssub = Array2::<f64>::zeros((k, k));

    for (p, (mrow, srow)) in rows.into_iter().enumerate() {
        for q in 0..=p {
            let mpq = mrow[q];
            let spq = srow[q];
            msub[(p, q)] = mpq;
            msub[(q, p)] = mpq;
            ssub[(p, q)] = spq;
            ssub[(q, p)] = spq;
        }
    }
    (msub, ssub)
}

fn orthonormalise(mut v: Array1<f64>, basis: &[Array1<f64>], sbasis: &[Array1<f64>], s: &Array2<f64>,tol: f64) -> Option<(Array1<f64>, Array1<f64>)> {
    let mut sv = parallel_matvec_real(s, &v);
    for _ in 0..2 {
        for (u, su) in basis.iter().zip(sbasis.iter()) {
            let proj = u.dot(&sv);
            v.scaled_add(-proj, u);
            sv.scaled_add(-proj, su);
        }
    }

    let n2 = v.dot(&sv);
    if !n2.is_finite() || n2 <= tol * tol {
        None
    } else {
        let inv = n2.sqrt().recip();
        v *= inv;
        sv *= inv;
        Some((v, sv))
    }
}

fn generalised_davidson(m: &Array2<f64>, s: &Array2<f64>, input: &Input, tol: f64) -> (Array1<f64>, Array2<f64>) {
    let n = m.nrows();
    let options = input.snoci.as_ref().unwrap();

    // Approximation of the correction `t` will use diagonals M_{\alpha \alpha}^{(k)} and S_{\alpha \alpha}^{(k)}.
    let mdiag = m.diag().to_owned();
    let sdiag = s.diag().to_owned();

    // Assemble initial trial subspace {U_{\alpha p}^{(k, 0)}}.
    // We use coordinate unit vectors in original projected candidate basis, choosing vectors 
    // which have the smallest values of  M_{\alpha \alpha}^{(k)} / S_{\alpha \alpha}^{(k)} as a 
    // guess of the eigenvalues.
    let mut order: Vec<usize> = (0..n).filter(|&i| sdiag[i] > tol).collect();
    order.sort_by(|&i, &j| {
        let qi = mdiag[i] / sdiag[i];
        let qj = mdiag[j] / sdiag[j];
        qi.partial_cmp(&qj).unwrap()
    });

    let mut basis = seed(m, s, options.davidson.nroots, tol);
    if basis.is_empty() {return (Array1::zeros(0), Array2::zeros((n, 0)));}

    // Converged eigenvectors and eigenvalues Z_{\beta a}^{(k)} and \Delta_a^{(k)}.
    let mut zfinal = Array2::<f64>::zeros((n, options.davidson.nroots));
    let mut deltafinal = Array1::<f64>::zeros(options.davidson.nroots);

    let mut mbasis: Vec<Array1<f64>> = basis.iter().map(|u| parallel_matvec_real(m, u)).collect();
    let mut sbasis: Vec<Array1<f64>> = basis.iter().map(|u| parallel_matvec_real(s, u)).collect();

    for _ in 0..options.davidson.max_iter {
        // Project M_{\alpha \beta}^{(k)} and S_{\alpha \beta}^{(k)} into the trial subspace to
        // get M_{pq}^{(k, m)} and S_{pq}^{(k, m)}.
        let (msub, ssub) = project_subspace(&basis, &mbasis, &sbasis);

        // Solve GEVP in the small trial subspace.
        let (delta, c) = general_evp_real(&msub, &ssub, true, tol);
        let keep = options.davidson.nroots.min(delta.len());

        // Ritz vector (approximate Z_{\alpha a}^{(k, m)}) and residual R_{\alpha a}^{(k, m)}.
        let mut zritz: Vec<Array1<f64>> = Vec::with_capacity(keep);
        let mut residual: Vec<Array1<f64>> = Vec::with_capacity(keep);
        let mut converged = 0usize;
        
        // For each of the lowest eigenpairs we construct full-space Ritz vector and test it with
        // the residual.
        for a in 0..keep {
            // Coefficient C_{pa}^{(k, m)} for trial vector U_{\alpha p}^{(k, m)}
            let ca = c.column(a).to_owned();
            // Z_{\alpha a}^{(k, m)} = \sum_p^m U_{\alpha p}^{(k, m)} C_{pa}^{(k, m)}
            let z = form_vector(&basis, &ca);

            // Calculate residual R_{\alpha a}^{(k, m)} = \sum_{\beta \in \mathcal{S}_N^{(k)}} M_{\alpha\beta}^{(k)} Z_{\beta a}^{(k, m)} 
            // - \Delta_a^{(k, m)} \sum_{\beta \in \mathcal{S}_N^{(k)}} S_{\alpha\beta}^{(k)} Z_{\beta a}^{(k, m)}.
            let mz = form_vector(&mbasis, &ca);
            let sz = form_vector(&sbasis, &ca);
            let mut r = mz.clone();
            r.scaled_add(-delta[a], &sz);

            // Check for convergence for this root.
            let rnorm = r.dot(&r).sqrt();
            if rnorm < options.davidson.res_tol {converged += 1;}

            // Save current best \Delta_a^{(k, m)} and Z_{\alpha a}^{(k, m)}.
            deltafinal[a] = delta[a];
            zfinal.column_mut(a).assign(&z);

            zritz.push(z);
            residual.push(r);
        }

        // If we have converged the required number of roots than return.
        if converged == keep {return (deltafinal.slice_move(ndarray::s![0..keep]), zfinal.slice_move(ndarray::s![.., 0..keep]))}

        // Construct new search directions.
        let mut dirs: Vec<Array1<f64>> = Vec::new();
        let mut dirsm: Vec<Array1<f64>> = Vec::new();
        let mut dirss: Vec<Array1<f64>> = Vec::new();
        for a in 0..keep {
            let deltaa = delta[a];
            let r = &residual[a];

            // Approximate correction T_{\alpha a}^{(k, m)}.
            let mut t = Array1::<f64>::zeros(n);
            for i in 0..n {
                let denom = mdiag[i] - deltaa * sdiag[i];
                if denom.abs() > tol {t[i] = -r[i] / denom};
            }
            
            // Orthogonalise new directions against old trial basis and already accepted new directions.
            let mut allbasis: Vec<Array1<f64>> = Vec::with_capacity(basis.len() + dirs.len());
            let mut allsbasis: Vec<Array1<f64>> = Vec::with_capacity(sbasis.len() + dirss.len());
            allbasis.extend(basis.iter().cloned());
            allbasis.extend(dirs.iter().cloned());
            allsbasis.extend(sbasis.iter().cloned());
            allsbasis.extend(dirss.iter().cloned());

            if let Some((v, sv)) = orthonormalise(t, &allbasis, &allsbasis, s, tol) {
                let mv = parallel_matvec_real(m, &v);
                dirs.push(v);
                dirsm.push(mv);
                dirss.push(sv);
            }

        }

        if dirs.is_empty() {return (deltafinal.slice_move(ndarray::s![0..keep]), zfinal.slice_move(ndarray::s![.., 0..keep]))}
        basis.extend(dirs);
        mbasis.extend(dirsm);
        sbasis.extend(dirss);

        // If the subspace gets too large we can restart only with the best trial vectors. 
        if basis.len() > options.davidson.max_subspace {
            let mut restartedbasis: Vec<Array1<f64>> = Vec::new();
            let mut restartedsbasis: Vec<Array1<f64>> = Vec::new();
            for a in 0..keep.min(options.davidson.restart_dim) {
                let z = zfinal.column(a).to_owned();
                if let Some((v, sv)) = orthonormalise(z, &restartedbasis, &restartedsbasis, s, tol) {
                    restartedbasis.push(v);
                    restartedsbasis.push(sv);
                }
            }
            basis = restartedbasis;
            sbasis = restartedsbasis;
            mbasis = basis.iter().map(|u| parallel_matvec_real(m, u)).collect();
        }
    }
    let keep = options.davidson.nroots.min(deltafinal.len());
    (deltafinal.slice_move(ndarray::s![0..keep]), zfinal.slice_move(ndarray::s![.., 0..keep]))
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

        let t0 = Instant::now();
        let m_omega_ab = &f_omega_ab - &(s_omega_ab.mapv(|x| ecurrent * x));
        let (delta, z) = generalised_davidson(&m_omega_ab, &s_omega_ab, input, tol);
        let v_bar = z.t().dot(&v_omega);
        timings.davidson += t0.elapsed();

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
