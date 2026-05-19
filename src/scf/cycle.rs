// scf/cycle.rs

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ndarray::{Array1, Array2, Axis};

use super::diis::Diis;
use super::kernels::DensityMode;
use crate::input::{Input, SCFExcitation, Spin, StateType};
use crate::{AoData, Excitation, SCFState};

use super::bias::metadynamics_bias;
use super::kernels::{density, energy, fock, orbital_gradient};
use super::occupation::{mo_occupancies, occvec_to_bits};
use super::print::{print_header, print_mos};
use super::select::{aufbau_indices, mom_select};
use crate::maths::general_evp;
use crate::utils::print_array2_indexed;
use crate::write::write_orbitals;

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
/// - `scfexcitation`: Optional excited SCF occupation request.
/// # Returns
/// - `(bool, bool)`: Whether alpha and beta occupations should use MOM.
fn mom_flags(scfexcitation: Option<&SCFExcitation>) -> (bool, bool) {
    if let Some(ex) = scfexcitation {
        match ex.spin {
            Spin::Alpha => (true, false),
            Spin::Beta => (false, true),
            Spin::Both => (true, true),
        }
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
            Some(c_occ_old) => mom_select(c_occ_old, c, s, nocc),
            None => seed_excited_occupation(
                nocc,
                scfexcitation.expect("MOM occupation requires an SCF excitation."),
            ),
        }
    } else {
        aufbau_indices(e, nocc)
    };
    let c_occ = c.select(Axis(1), &idx);
    (idx, c_occ)
}

/// Calculate spin contamination expectation value.
/// # Arguments
/// - `da`: Alpha-spin density matrix.
/// - `db`: Beta-spin density matrix.
/// - `s`: AO overlap matrix.
/// # Returns
/// - `f64`: Expectation value of S^2.
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
    let oaprint = mo_occupancies(state.ca.as_ref(), state.da.as_ref(), &ao.s);
    let obprint = mo_occupancies(state.cb.as_ref(), state.db.as_ref(), &ao.s);

    print_mos("Alpha MOs", ea, &oaprint);
    print_mos("Beta MOs", eb, &obprint);
    println!("{}", "-".repeat(100));
    println!("Coefficients ca:");
    print_array2_indexed(state.ca.as_ref());
    println!("Coefficients cb:");
    print_array2_indexed(state.cb.as_ref());
    println!(
        "<S^2>: {}",
        spin_square(state.da.as_ref(), state.db.as_ref(), &ao.s)
    );

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
/// - `da0`: Initial spin a density matrix.
/// - `db0`: Initial spin b density matrix.
/// - `ao`: Contains AO integrals and metadata.
/// - `input`: Contains user specified input data.
/// - `label`: Label for current scf state.
/// - `noci_basis`: Whether or not to use this state in the NOCI basis.
/// - `scfexcitation`: Use requested excited states.
/// - `i`: Index of the SCF state.
/// - `biases`: Previously found states when using metadynamics.
/// # Returns
/// - `Option<SCFState>`: Converged SCF state if the SCF cycle succeeds, otherwise `None`.
pub fn scf_cycle(
    d0: (&Array2<f64>, &Array2<f64>),
    ao: &AoData,
    input: &Input,
    label: &str,
    noci_basis: bool,
    i: usize,
    controls: (Option<&SCFExcitation>, Option<&[SCFState]>),
) -> Option<SCFState> {
    let (da0, db0) = d0;
    let (scfexcitation, biases) = controls;

    let h = &ao.h;
    let eri = &ao.eri_coul;
    let s = &ao.s;
    let enuc = ao.enuc;

    let na = usize::try_from(ao.nelec[0]).unwrap();
    let nb = usize::try_from(ao.nelec[1]).unwrap();

    let mut e = f64::INFINITY;
    let mut da = da0.clone();
    let mut db = db0.clone();

    let use_diis = true;
    let mut diis = Diis::new(input.scf.diis.space);

    let mut ca_occ_old: Option<Array2<f64>> = None;
    let mut cb_occ_old: Option<Array2<f64>> = None;

    print_header(input, scfexcitation);
    let lambda = match &input.states {
        StateType::Metadynamics(meta) => Some(meta.lambda),
        _ => None,
    };
    let (mom_a, mom_b) = mom_flags(scfexcitation);

    let mut iter = 0;
    while iter < input.scf.max_cycle {
        let (mut fa_curr, mut fb_curr) = fock(h, eri, &da, &db);

        if let (Some(bias_states), Some(lambda)) = (biases, lambda)
            && !bias_states.is_empty()
        {
            let (ba, bb) = metadynamics_bias(&da, &db, ao, bias_states, lambda);
            fa_curr = fa_curr + ba;
            fb_curr = fb_curr + bb;
        }

        if use_diis {
            diis.push(&fa_curr, &fb_curr, &da, &db, s);
        }
        let (fa_use, fb_use) = if use_diis {
            diis.extrapolate_fock()
                .unwrap_or((fa_curr.clone(), fb_curr.clone()))
        } else {
            (fa_curr.clone(), fb_curr.clone())
        };

        let (ea, ca) = general_evp(&fa_use, s, false, 1e-8);
        let (eb, cb) = general_evp(&fb_use, s, false, 1e-8);

        let (idx_a, ca_occ) = occupy(&ea, &ca, s, na, mom_a, ca_occ_old.as_ref(), scfexcitation);
        let (idx_b, cb_occ) = occupy(&eb, &cb, s, nb, mom_b, cb_occ_old.as_ref(), scfexcitation);
        ca_occ_old = Some(ca_occ.clone());
        cb_occ_old = Some(cb_occ.clone());

        let da_new = density(&ca_occ, na, DensityMode::Hermitian);
        let db_new = density(&cb_occ, nb, DensityMode::Hermitian);
        let (fa_new, fb_new) = fock(h, eri, &da_new, &db_new);
        let e_new = energy(h, enuc, &da_new, &db_new, &fa_new, &fb_new);

        let err = if use_diis {
            diis.last_error_norm2().unwrap_or(f64::INFINITY).sqrt()
        } else {
            f64::INFINITY
        };
        let d_e = (e_new - e).abs();

        let ga = orbital_gradient(&ca, &fa_new, na, DensityMode::Hermitian);
        let gb = orbital_gradient(&cb, &fb_new, nb, DensityMode::Hermitian);
        let gnorm =
            (ga.iter().map(|z| z * z).sum::<f64>() + gb.iter().map(|z| z * z).sum::<f64>()).sqrt();

        if input.write.verbose {
            println!(
                "{:4} {:12.6} {:12.4e} {:12.4e} {:12.4e}",
                iter, e_new, d_e, err, gnorm
            );
        }

        if d_e < input.scf.e_tol && err < input.scf.fds_sdf_tol {
            let state = SCFState {
                e: e_new,
                oa: occvec_to_bits(&idx_a),
                ob: occvec_to_bits(&idx_b),
                pha: 1.0,
                phb: 1.0,
                ca: Arc::new(ca),
                cb: Arc::new(cb),
                da: Arc::new(da_new),
                db: Arc::new(db_new),
                label: label.to_string(),
                noci_basis,
                parent: i,
                excitation: Excitation::empty(),
            };
            return Some(finalise(state, &ea, &eb, ao, input));
        }

        da = da_new;
        db = db_new;
        e = e_new;
        iter += 1;
    }

    println!("SCF not converged.");
    None
}
