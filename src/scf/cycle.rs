// scf/cycle.rs

// Standard library imports.
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// External crate imports.
use ndarray::{Array1, Array2, Axis};

// Crate-root imports.
use crate::input::{Input, SCFExcitation, Spin, StateType};
use crate::maths::general_evp_x;
use crate::utils::print_array2_indexed;
use crate::write::write_orbitals;
use crate::{AoData, SCFState};

// Parent/sibling imports.
use super::bias::metadynamics_bias;
use super::diis::Diis;
use super::kernels::DensityMode;
use super::kernels::{density, energy, fock, orbital_gradient};
use super::occupation::{mo_occupancies, occvec_to_bits};
use super::print::{print_header, print_mos};
use super::select::{aufbau_indices, mom_select};

/// Convert spin enum to printable label.
/// # Arguments
/// - `spin`: Spin sector from input.
/// # Returns
/// - `&str`: Printable spin label.
pub(in crate::scf) fn spin_label(spin: &Spin) -> &str {
    match spin {
        Spin::Alpha => "alpha",
        Spin::Beta => "beta",
        Spin::Both => "both",
    }
}

/// Determine which spin channels should use MOM occupation selection.
/// # Arguments
/// - `use_mom`: Whether to use MOM for non-excited SCF occupation tracking.
/// - `scfexcitation`: Optional excited SCF occupation request.
/// # Returns
/// - `(bool, bool)`: Whether alpha and beta occupations should use MOM.
fn mom_flags(
    use_mom: bool,
    scfexcitation: Option<&SCFExcitation>,
) -> (bool, bool) {
    if let Some(ex) = scfexcitation {
        match ex.spin {
            Spin::Alpha => (true, false),
            Spin::Beta => (false, true),
            Spin::Both => (true, true),
        }
    } else if use_mom {
        (true, true)
    } else {
        (false, false)
    }
}

/// Build the initial excited-state occupation used to seed MOM.
/// # Arguments
/// - `nocc`: Number of occupied MOs.
/// - `ex`: Excited SCF occupation request.
/// # Returns
/// - `Vec<usize>`: Occupied MO indices.
fn seed_excited_occupation(
    nocc: usize,
    ex: &SCFExcitation,
) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..nocc).collect();
    let occ_abs = (nocc as i32 + ex.occ) as usize;
    let vir_abs = nocc + ex.vir as usize;
    idx.retain(|&k| k != occ_abs);
    idx.push(vir_abs);
    idx
}

/// Select occupied MOs and construct occupied coefficient matrix.
/// # Arguments
/// - `e`: MO energies.
/// - `c`: MO coefficient matrix.
/// - `s`: AO overlap matrix.
/// - `nocc`: Number of occupied MOs.
/// - `use_mom`: Whether to use MOM occupation selection.
/// - `c_occ_old`: Previous iteration occupied MO coefficient matrix.
/// - `scfexcitation`: Optional excited SCF occupation request.
/// # Returns
/// - `(Vec<usize>, Array2<f64>)`: Occupied MO indices and occupied MO coefficients.
fn occupy(
    e: &Array1<f64>,
    c: &Array2<f64>,
    s: &Array2<f64>,
    nocc: usize,
    use_mom: bool,
    c_occ_old: Option<&Array2<f64>>,
    scfexcitation: Option<&SCFExcitation>,
) -> (Vec<usize>, Array2<f64>) {
    let idx = if use_mom {
        match c_occ_old {
            // If some orbitals already exist use MOM on them.
            Some(c_occ_old) => mom_select(c_occ_old, c, s, nocc),
            // If they do not exist we are on first MOM iteration.
            None => match scfexcitation {
                // If there's an excitation do excited state MOM.
                Some(ex) => seed_excited_occupation(nocc, ex),
                // Otherwise take Aufbau ordering for first MOM iteration.
                None => aufbau_indices(e, nocc),
            },
        }
    } else {
        aufbau_indices(e, nocc)
    };

    let c_occ = c.select(Axis(1), &idx);
    (idx, c_occ)
}

/// Orbitals, occupations, densities, Fock matrices and energy after one Roothaan step.
pub(crate) struct SCFIterate {
    /// Alpha- and beta-spin MO energies.
    pub(crate) e_mo: (Array1<f64>, Array1<f64>),
    /// Alpha- and beta-spin MO coefficients, in eigenvalue order.
    pub(crate) c: (Array2<f64>, Array2<f64>),
    /// Alpha- and beta-spin occupied MO indices.
    pub(crate) idx: (Vec<usize>, Vec<usize>),
    /// Alpha- and beta-spin occupied MO coefficients.
    pub(crate) c_occ: (Array2<f64>, Array2<f64>),
    /// Alpha- and beta-spin density matrices.
    pub(crate) d: (Array2<f64>, Array2<f64>),
    /// Alpha- and beta-spin Fock matrices of the new densities.
    pub(crate) f: (Array2<f64>, Array2<f64>),
    /// SCF energy of the new densities.
    pub(crate) e: f64,
}

impl SCFIterate {
    /// Convert the iterate into an SCF state.
    /// # Arguments
    /// - `self`: Iterate to convert.
    /// - `label`: Label for the state.
    /// - `noci_basis`: Whether or not to use this state in the NOCI basis.
    /// # Returns
    /// - `SCFState`: State with the iterate's orbitals, occupations, densities and energy.
    pub(crate) fn into_state(
        self,
        label: &str,
        noci_basis: bool,
    ) -> SCFState {
        SCFState {
            e: self.e,
            oa: occvec_to_bits(&self.idx.0),
            ob: occvec_to_bits(&self.idx.1),
            ca: Arc::new(self.c.0),
            cb: Arc::new(self.c.1),
            da: Arc::new(self.d.0),
            db: Arc::new(self.d.1),
            label: label.to_string(),
            noci_basis,
        }
    }
}

/// Take one Roothaan step: solve `F^\sigma C^\sigma = S C^\sigma \epsilon^\sigma` for both
/// spins, select the occupied orbitals by Aufbau or MOM, and rebuild the densities, Fock
/// matrices and SCF energy.
/// # Arguments
/// - `ao`: Contains AO integrals and metadata.
/// - `f`: Alpha- and beta-spin Fock matrices to diagonalise.
/// - `nocc`: Number of alpha- and beta-spin occupied MOs.
/// - `mom`: Whether alpha- and beta-spin occupations use MOM.
/// - `c_occ_old`: Previous alpha- and beta-spin occupied MO coefficients, if any.
/// - `scfexcitation`: Optional excited SCF occupation request.
/// # Returns
/// - `SCFIterate`: New orbitals, occupations, densities, Fock matrices and energy.
pub(crate) fn roothaan_step(
    ao: &AoData,
    f: (&Array2<f64>, &Array2<f64>),
    nocc: (usize, usize),
    mom: (bool, bool),
    c_occ_old: (Option<&Array2<f64>>, Option<&Array2<f64>>),
    scfexcitation: Option<&SCFExcitation>,
) -> SCFIterate {
    let (na, nb) = nocc;

    // Solve both spin eigenproblems and select occupied orbitals by Aufbau or MOM.
    let ((ea, ca), (eb, cb)) =
        rayon::join(|| general_evp_x(f.0, &ao.x), || general_evp_x(f.1, &ao.x));

    let (idx_a, ca_occ) = occupy(&ea, &ca, &ao.s, na, mom.0, c_occ_old.0, scfexcitation);
    let (idx_b, cb_occ) = occupy(&eb, &cb, &ao.s, nb, mom.1, c_occ_old.1, scfexcitation);

    // Rebuild physical densities, Fock matrices, and the unbiased SCF energy.
    let (da, db) = rayon::join(
        || density(&ca_occ, na, DensityMode::Hermitian),
        || density(&cb_occ, nb, DensityMode::Hermitian),
    );

    let (fa, fb) = fock(&ao.h, &ao.eri_coul, &da, &db);

    let e = energy(&ao.h, ao.enuc, &da, &db, &fa, &fb);

    SCFIterate {
        e_mo: (ea, eb),
        c: (ca, cb),
        idx: (idx_a, idx_b),
        c_occ: (ca_occ, cb_occ),
        d: (da, db),
        f: (fa, fb),
        e,
    }
}

/// Calculate spin contamination expectation value.
/// # Arguments
/// - `da`: Alpha-spin density matrix.
/// - `db`: Beta-spin density matrix.
/// - `s`: AO overlap matrix.
/// # Returns
/// - `f64`: `Expectation value of S^2.`
fn spin_square(
    da: &Array2<f64>,
    db: &Array2<f64>,
    s: &Array2<f64>,
) -> f64 {
    let nas = (da.dot(s)).diag().sum();
    let nbs = (db.dot(s)).diag().sum();
    let sz = 0.5 * (nas - nbs);
    let trdasdbs = da.dot(s).dot(db).dot(s).diag().sum();
    sz * (sz + 1.0) + nbs - trdasdbs
}

/// Print and write a converged SCF state.
/// # Arguments
/// - `state`: Converged SCF state.
/// - `ea`: Alpha-spin MO energies.
/// - `eb`: Beta-spin MO energies.
/// - `ao`: Contains AO integrals and metadata.
/// - `input`: Contains user specified input data.
/// # Returns
/// - `SCFState`: Converged SCF state.
fn finalise(
    state: SCFState,
    ea: &Array1<f64>,
    eb: &Array1<f64>,
    ao: &AoData,
    input: &Input,
) -> SCFState {
    // Compute overlap-aware MO occupations once for both reporting and the
    // optional orbital file.
    let oaprint = mo_occupancies(state.ca.as_ref(), state.da.as_ref(), &ao.s);
    let obprint = mo_occupancies(state.cb.as_ref(), state.db.as_ref(), &ao.s);

    if input.write.verbose >= 1 {
        print_mos("Alpha MOs", ea, &oaprint);
        print_mos("Beta MOs", eb, &obprint);
        println!(
            "<S^2>: {}",
            spin_square(state.da.as_ref(), state.db.as_ref(), &ao.s)
        );
    }

    if input.write.verbose >= 2 {
        println!("{}", "-".repeat(100));
        println!("Coefficients ca:");
        print_array2_indexed(state.ca.as_ref());
        println!("Coefficients cb:");
        print_array2_indexed(state.cb.as_ref());
    }

    // Sanitise the state label before using it as an orbital filename.
    if input.write.write_orbitals {
        let orbitalsdir: PathBuf = Path::new(&input.write.write_dir).join("orbitals");
        let _ = fs::create_dir_all(&orbitalsdir);
        let labelstr = state
            .label
            .replace([' ', ','], "_")
            .replace(['(', ')'], "")
            .replace('/', "_");
        let fname = orbitalsdir.join(format!("{labelstr}orbitals.h5"));

        write_orbitals(
            fname.to_str().unwrap(),
            ao,
            &state.label,
            (state.ca.as_ref(), state.cb.as_ref()),
            (ea, eb),
            (oaprint.as_slice().unwrap(), obprint.as_slice().unwrap()),
            (state.da.as_ref(), state.db.as_ref()),
        );
    }

    state
}

/// Unrestricted SCF cycle.
/// Uses AO integrals from AoData struct.
/// # Arguments
/// - `d0`: Initial `\alpha`- and `\beta`-spin density matrices.
/// - `ao`: Contains AO integrals and metadata.
/// - `input`: Contains user specified input data.
/// - `label`: Label for current scf state.
/// - `noci_basis`: Whether or not to use this state in the NOCI basis.
/// - `controls`: Explicit MOM flag, requested excited state, and metadynamics bias states.
/// # Returns
/// - `Option<SCFState>`: Converged SCF state if the SCF cycle succeeds, otherwise `None`.
pub fn scf_cycle(
    d0: (&Array2<f64>, &Array2<f64>),
    ao: &AoData,
    input: &Input,
    label: &str,
    noci_basis: bool,
    controls: (bool, Option<&SCFExcitation>, Option<&[SCFState]>),
) -> Option<SCFState> {
    // Unpack controls and establish the spin occupations for this SCF branch.
    let (da0, db0) = d0;
    let (use_mom, scfexcitation, biases) = controls;

    let na = usize::try_from(ao.nelec[0]).unwrap();
    let nb = usize::try_from(ao.nelec[1]).unwrap();

    // Initialise the physical densities, convergence state, and DIIS history.
    let mut e = f64::INFINITY;
    let mut da = da0.clone();
    let mut db = db0.clone();

    let use_diis = true;
    let mut diis = Diis::new(input.scf.diis.space);

    let mut ca_occ_old: Option<Array2<f64>> = None;
    let mut cb_occ_old: Option<Array2<f64>> = None;

    print_header(input, scfexcitation);

    // Resolve optional metadynamics and maximum-overlap controls once per cycle.
    let lambda = match &input.states {
        StateType::Metadynamics(meta) => Some(meta.lambda),
        _ => None,
    };

    let (mom_a, mom_b) = mom_flags(use_mom, scfexcitation);

    // Build the unbiased physical Fock matrices from the initial densities.
    let (mut fa_phys, mut fb_phys) = fock(&ao.h, &ao.eri_coul, &da, &db);

    let mut iter = 0;
    while iter < input.scf.max_cycle {
        // Form this iteration's trial Fock, adding bias only to the diagonalisation problem.
        let mut fa_curr = fa_phys.clone();
        let mut fb_curr = fb_phys.clone();

        if let (Some(bias_states), Some(lambda)) = (biases, lambda)
            && !bias_states.is_empty()
        {
            let (ba, bb) = metadynamics_bias(&da, &db, ao, bias_states, lambda);
            fa_curr = fa_curr + ba;
            fb_curr = fb_curr + bb;
        }

        // Extrapolate the trial Fock from commutator errors when sufficient history exists.
        if use_diis {
            diis.push(&[&fa_curr, &fb_curr], &[&da, &db], &ao.s, &ao.x);
        }
        let (fa_use, fb_use) = if use_diis {
            diis.extrapolate_fock()
                .and_then(|f| <[Array2<f64>; 2]>::try_from(f).ok())
                .map_or((fa_curr.clone(), fb_curr.clone()), |[fa, fb]| (fa, fb))
        } else {
            (fa_curr.clone(), fb_curr.clone())
        };

        // Diagonalise, occupy and rebuild the densities, Fock matrices and energy.
        let step = roothaan_step(
            ao,
            (&fa_use, &fb_use),
            (na, nb),
            (mom_a, mom_b),
            (ca_occ_old.as_ref(), cb_occ_old.as_ref()),
            scfexcitation,
        );
        ca_occ_old = Some(step.c_occ.0.clone());
        cb_occ_old = Some(step.c_occ.1.clone());

        // Evaluate energy, commutator, and orbital-gradient convergence diagnostics.
        let err = if use_diis {
            diis.last_error_norm2().unwrap_or(f64::INFINITY).sqrt()
        } else {
            f64::INFINITY
        };
        let d_e = (step.e - e).abs();

        let ga = orbital_gradient(&step.c.0, &step.f.0, na, DensityMode::Hermitian);
        let gb = orbital_gradient(&step.c.1, &step.f.1, nb, DensityMode::Hermitian);
        let gnorm =
            (ga.iter().map(|z| z * z).sum::<f64>() + gb.iter().map(|z| z * z).sum::<f64>()).sqrt();

        if input.write.verbose >= 1 {
            println!(
                "{:4} {:12.6} {:12.4e} {:12.4e} {:12.4e}",
                iter, step.e, d_e, err, gnorm
            );
        }

        // Finalise immediately once both configured convergence criteria are satisfied.
        if d_e < input.scf.e_tol && err < input.scf.fds_sdf_tol {
            let (ea, eb) = step.e_mo.clone();
            let state = step.into_state(label, noci_basis);
            return Some(finalise(state, &ea, &eb, ao, input));
        }

        // Commit the physical iterate for the next cycle.
        (da, db) = step.d;
        (fa_phys, fb_phys) = step.f;
        e = step.e;
        iter += 1;
    }

    // Exhausting the iteration budget leaves this requested state unavailable.
    println!("SCF not converged.");
    None
}
