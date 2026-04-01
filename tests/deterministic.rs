mod common;

use common::{assert_close, load_test};
use serial_test::serial;
use serde::Deserialize;
use ndarray::Array1;

use noci_rs::basis::{generate_excited_basis, generate_reference_noci_basis};
use noci_rs::deterministic::{projected_energy, propagate};
use noci_rs::noci::{build_noci_hs, calculate_noci_energy};

#[derive(Deserialize)]
struct ExpectedDeterministic {
    scf_energies: Vec<f64>,
    reference_noci_energy: f64,
    deterministic_noci_energy: f64,
}

/// Run SCF, reference NOCI and deterministic NOCI-QMC and compare energies with known good energies. 
/// # Arguments:
/// - `fixture`: Name of the test fixture to load.
/// # Returns
/// - `(Vec<f64>, f64, f64)`: Sorted SCF state energies, the reference NOCI energy and the
///   deterministic NOCI-QMC energy.
fn run_deterministic_fixture(fixture: &str) -> (Vec<f64>, f64, f64) {
    let (mut input, ao, _expected): (_, _, ExpectedDeterministic) = load_test(fixture);

    let states = generate_reference_noci_basis(&ao, &mut input, None);
    let mut scf_energies: Vec<f64> = states.iter().map(|s| s.e).collect();
    scf_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut noci_reference_basis: Vec<_> = states.iter().filter(|s| s.noci_basis).cloned().collect();
    for (i, st) in noci_reference_basis.iter_mut().enumerate() {
        st.parent = i;
    }

    let (e_ref, c0, _dt_hs_ref) = calculate_noci_energy(&ao, &input, &noci_reference_basis, 1e-12, None);

    let include_refs = true;
    let basis = generate_excited_basis(&noci_reference_basis, &input, include_refs);

    let symmetric = true;
    let (h, s, _d_hs) = build_noci_hs(&ao, &input, &basis, &basis, &noci_reference_basis, 1e-12, None, symmetric,);

    let n = basis.len();
    let mut c0qmc = Array1::<f64>::zeros(n);
    if !input.write.write_deterministic_coeffs {
        for (i, ref_st) in noci_reference_basis.iter().enumerate() {
            let idx = basis.iter().position(|qmc_st| qmc_st.label == ref_st.label).unwrap();
            c0qmc[idx] = c0[i];
        }
    } else {
        c0qmc = Array1::from_elem(n, 1.0 / (n as f64).sqrt());
    }

    let es = states[0].e;

    let mut coefficients = Vec::new();
    let cfinal = propagate(&h, &s, &c0qmc, es, &mut coefficients, &input).expect("deterministic propagation failed");
    let e_det = projected_energy(&h, &s, &cfinal);

    (scf_energies, e_ref, e_det)
}

/// Test that the H2 cc-pVDZ 1.5 Angstrom fixture reproduces the expected SCF state energies,
/// reference NOCI energy and deterministic NOCI-QMC energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF, reference NOCI or deterministic NOCI-QMC energy differs from known good value
///   outside tolerance.
#[test]
#[serial]
fn deterministic_h2_cc_pvdz_1_5_ang_energies() {
    let (_input, _ao, expected): (_, _, ExpectedDeterministic) = load_test("DET_H2_cc-pVDZ_1_5");
    let (got_scf, got_ref, got_det) = run_deterministic_fixture("DET_H2_cc-pVDZ_1_5");

    let mut want_scf = expected.scf_energies;
    want_scf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got_scf.len(), want_scf.len());
    for (i, (&x, &y)) in got_scf.iter().zip(want_scf.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("H2 SCF state {i}"));
    }

    assert_close(got_ref, expected.reference_noci_energy, 1e-8, "H2 reference NOCI energy");
    assert_close(got_det, expected.deterministic_noci_energy, 1e-8, "H2 deterministic energy");
}

/// Test that the LiH 6-31G 2.8 Angstrom fixture reproduces the expected SCF state energies,
/// reference NOCI energy and deterministic NOCI-QMC energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF, reference NOCI or deterministic NOCI-QMC energy differs from known good value
///   outside tolerance.
#[test]
#[serial]
fn deterministic_lih_6_31g_2_8_ang_energies() {
    let (_input, _ao, expected): (_, _, ExpectedDeterministic) = load_test("DET_LiH_6-31G_2_8");
    let (got_scf, got_ref, got_det) = run_deterministic_fixture("DET_LiH_6-31G_2_8");

    let mut want_scf = expected.scf_energies;
    want_scf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got_scf.len(), want_scf.len());
    for (i, (&x, &y)) in got_scf.iter().zip(want_scf.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("LiH SCF state {i}"));
    }

    assert_close(got_ref, expected.reference_noci_energy, 1e-8, "LiH reference NOCI energy");
    assert_close(got_det, expected.deterministic_noci_energy, 1e-8, "LiH deterministic energy",);
}

/// Test that the F2 sto-3g 1.75 Angstrom fixture reproduces the expected SCF state energies,
/// reference NOCI energy and deterministic NOCI-QMC energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF, reference NOCI or deterministic NOCI-QMC energy differs from known good value
///   outside tolerance.
#[test]
#[serial]
fn deterministic_f2_sto_3g_1_75_ang_energies() {
    let (_input, _ao, expected): (_, _, ExpectedDeterministic) = load_test("DET_F2_sto-3g_1_75");
    let (got_scf, got_ref, got_det) = run_deterministic_fixture("DET_F2_sto-3g_1_75");

    let mut want_scf = expected.scf_energies;
    want_scf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got_scf.len(), want_scf.len());
    for (i, (&x, &y)) in got_scf.iter().zip(want_scf.iter()).enumerate() {
        assert_close(x, y, 1e-7, &format!("F2 SCF state {i}"));
    }

    assert_close(got_ref, expected.reference_noci_energy, 1e-8, "F2 reference NOCI energy");
    assert_close(got_det, expected.deterministic_noci_energy, 1e-8, "F2 deterministic energy");
}
