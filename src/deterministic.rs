use ndarray::{Array1, Array2};
use num_complex::Complex64;

/// Perform one deterministic update step of NOCI-QMC unshifted propagator:
/// # Arguments
///     `h`: Array2, Shifted NOCI Hamiltonian in full NOCI-QMC basis: \tilde{H} = (H - E_s * S)
///                  where E_s is the shift energy and S the NOCI overlap matrix.
///     `c`: Array1, NOCI-QMC coefficient vector.
///     `dt`: Propagation time step.
pub fn propagate_step_unshifted(h: &Array2<Complex64>, c: &Array1<Complex64>, dt: f64) -> Array1<Complex64> {
    let n = c.len();
    let mut c_new = c.clone();

    for lam in 0..n {
        let mut sum = Complex64::new(0.0, 0.0);
        for gam in 0..n {
            let h_lg = h[(lam, gam)];
            sum += h_lg * c[gam];
        }
        c_new[lam] = c[lam] - Complex64::new(dt, 0.0) * sum;
    }
    c_new
}

/// Perform one deterministic update step of NOCI-QMC shifted propagator:
/// C^\Lambda(\tau + \Delta\tau) = C^\Lambda(\tau) - \Delta\tau(H_{\Lambda\Lambda} - E_sS_{\Lambda\Lambda} - E_s)C^\Lambda(\tau) 
/// - \Delta\tau \sum_{\Gamma \neq \Lambda}(H_{\Lambda\Gamma} - E_s S_{\Lambda\Gamma})C^\Gamma(\tau).
/// # Arguments
///     `h`: Array2, Shifted NOCI Hamiltonian in full NOCI-QMC basis: \tilde{H} = (H - E_s * S)
///                  where E_s is the shift energy and S the NOCI overlap matrix.
///     `c`: Array1, NOCI-QMC coefficient vector.
///     `esc`: Scalar, Energy shift, we just use the reference NOCI energy for this here.
///     `dt`: Propagation time step.
pub fn propagate_step_shifted(h: &Array2<Complex64>, c: &Array1<Complex64>, esc: Complex64, dt: f64) -> Array1<Complex64> {
    let n = c.len();
    let mut c_new = c.clone();
    
    for lam in 0..n{
        let mut off = Complex64::new(0.0, 0.0);
        
        // Diagonal contribution.
        let h_ll = h[(lam, lam)];
        let diag = (h_ll - esc) * c[lam];

        // Off-diagonal contribution.
        for gam in 0..n {
            if gam == lam {continue;}
            let h_lg = h[(lam, gam)];
            off += h_lg * c[gam];
        }
        c_new[lam] = c[lam] - Complex64::new(dt, 0.0) * (diag + off);
    }
    c_new
}

/// Propagate nsteps number of time-step updates.
/// # Arguments
///     `h`: Array2, NOCI Hamiltonian in full NOCI-QMC basis. Shifted by E_s * S.
///     `s`: Array2, Overlap matrix in full NOCI-QMC basis.
///     `c0`: Array1, Initial NOCI-QMC coefficient vector, start from reference NOCI coefficients.
///     `es`: Scalar, Energy shift, we just use the reference HF energy for this here.
///     `max_steps`: Maximum number of time-step updates to perform.
///     `e_tol`: Energy tolerance which determines convergence.
pub fn propagate(h: &Array2<Complex64>, s: &Array2<Complex64>, c0: &Array1<Complex64>,
    es: f64, dt: f64, max_steps: usize, e_tol: f64,) -> Option<Array1<Complex64>> {

    let mut c = c0.clone();
    let mut e_prev = projected_energy(h, s, &c, es);
    let esc = Complex64::new(es, 0.0);

    let relative_de_max = 1.0;
    
    // Print table header.
    println!("{}", "=".repeat(100));
    println!("{:<6} {:>10} {:>10}, {:>10} {:>10}", "iter", "E", "|dE|", "||C||", "C^† S C");
    // Print initial.
    let c01norm = c0.iter().map(|z| z.norm()).sum::<f64>();
    let den = c.mapv(|z| z.conj()).dot(&s.dot(&c));
    println!("{:<6} {:>10.6} {:>10.3e}, {:>10.6} {:>10.6}", 0, e_prev, 0, c01norm, den);

    for it in 0..max_steps {
        //let mut c_new = propagate_step_unshifted(h, &c, dt);
        let mut c_new = propagate_step_shifted(h, &c, esc, dt);

        // Normalise.
        let norm = c_new.mapv(|z| z.conj()*z).sum().sqrt();
        c_new.mapv_inplace(|z| z / norm);

        let e = projected_energy(h, s, &c_new, es);
        let de = (e - e_prev).abs();
        let relative_de = de / e_prev.abs();
        
        // Print table rows.
        let c1norm = c_new.iter().map(|z| z.norm()).sum::<f64>();
        let den = c_new.mapv(|z| z.conj()).dot(&s.dot(&c_new));
        println!("{:<6} {:>10.6} {:>10.3e} {:>10.6} {:>10.6}", it + 1, e, de, c1norm, den);

        // If our energy change between iterations is large we likely have problems with
        // singularity and very low eigenvalues. Just skip this geometry for now.
        if relative_de > relative_de_max {
            println!("Relative energy change too large at iter {}: relative |dE| = {}. Eigenvalues invalid, S and/or H possibly singular. Fix needed.", it + 1, relative_de);
            return None
        }

        if de < e_tol {
            return Some(c_new)
        }

        c = c_new;
        e_prev = e;
    }
    Some(c)
}

/// Calculate the projected energy using a given NOCI-QMC coefficient vector as:
/// E(\tau) = \frac{C^\Lambda\langle\Psi_\Lambda|\hat H|\Psi_\Gamma\rangle C^\Gamma}{C^\Lambda\langle\Psi_\Lambda|\Psi_\Gamma\rangle C^\Gamma} 
/// = \frac{C^\Lambda H_{\Lambda\Gamma}C^{\Gamma} }{C^\Lambda S_{\Lambda\Gamma}C^\Gamma}.
/// # Arguments
///     `h`: Array2, NOCI Hamiltonian in full NOCI-QMC basis. Shifted by E_s * S.
///     `s`: Array2, Overlap matrix in full NOCI-QMC basis.
///     `c`: Array1, NOCI-QMC coefficient vector.
///     `es`: Scalar, Energy shift, we just use the reference HF energy for this here.
pub fn projected_energy(h: &Array2<Complex64>, s: &Array2<Complex64>, c: &Array1<Complex64>, es: f64) -> f64 {
    let hc = h.dot(c);
    let num = c.iter().zip(hc.iter()).map(|(ci, hci)| ci.conj() * hci).sum::<Complex64>();
    let sc = s.dot(c);
    let den = c.iter().zip(sc.iter()).map(|(ci, sci)| ci.conj() * sci).sum::<Complex64>();
    (num / den).re + es
}
