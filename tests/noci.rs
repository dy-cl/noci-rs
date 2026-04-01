mod common;

use common::{assert_close, load_test};
use serde::Deserialize;
use serial_test::serial;

use noci_rs::noci::{calculate_noci_energy};
use noci_rs::basis::{generate_reference_noci_basis};

#[derive(Deserialize)]
struct ExpectedReferenceNoci {
    scf_energies: Vec<f64>,
    reference_noci_energy: f64,
}

/// Run SCF and reference NOCI and compare energies with known good energies. 
/// # Arguments:
/// - `fixture`: Name of the test fixture to load.
/// # Returns
/// - `(Vec<f64>, f64)`: Sorted SCF state energies and the reference NOCI energy.
fn run_reference_noci_fixture(fixture: &str) -> (Vec<f64>, f64) {
    let (mut input, ao, _expected): (_, _, ExpectedReferenceNoci) = load_test(fixture);

    let states = generate_reference_noci_basis(&ao, &mut input, None);
    let mut scf_energies: Vec<f64> = states.iter().map(|s| s.e).collect();
    scf_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let mut basis: Vec<_> = states.into_iter().filter(|s| s.noci_basis).collect();
    for (i, st) in basis.iter_mut().enumerate() {
        st.parent = i;
    }

    let (e_ref, _coeffs, _dt_hs) = calculate_noci_energy(&ao, &input, &basis, 1e-12, None);

    (scf_energies, e_ref)
}

/// Test that the H2 cc-pVDZ 1.5 Angstrom fixture reproduces the expected SCF state energies and reference
/// NOCI energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF or reference NOCI energy differs from known good value outside tolerance.
#[test]
#[serial]
fn reference_noci_h2_cc_pvdz_1_5_ang_energies() {
    let (_input, _ao, expected): (_, _, ExpectedReferenceNoci) = load_test("REF_NOCI_H2_cc-pVDZ_1_5");
    let (got_scf, got_ref) = run_reference_noci_fixture("REF_NOCI_H2_cc-pVDZ_1_5");

    let mut want_scf = expected.scf_energies;
    want_scf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got_scf.len(), want_scf.len());
    for (i, (&x, &y)) in got_scf.iter().zip(want_scf.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("H2 SCF state {i}"));
    }
    assert_close(got_ref, expected.reference_noci_energy, 1e-8, "H2 reference NOCI energy",);
}

/// Test that the LiH cc-pVDZ 2.8 Angstrom fixture reproduces the expected SCF state energies and reference
/// NOCI energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF or reference NOCI energy differs from known good value outside tolerance.
#[test]
#[serial]
fn reference_noci_lih_cc_pvdz_2_8_ang_energies() {
    let (_input, _ao, expected): (_, _, ExpectedReferenceNoci) = load_test("REF_NOCI_LiH_cc-pVDZ_2_8");
    let (got_scf, got_ref) = run_reference_noci_fixture("REF_NOCI_LiH_cc-pVDZ_2_8");

    let mut want_scf = expected.scf_energies;
    want_scf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got_scf.len(), want_scf.len());
    for (i, (&x, &y)) in got_scf.iter().zip(want_scf.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("LiH SCF state {i}"));
    }
    assert_close(got_ref, expected.reference_noci_energy, 1e-8, "LiH reference NOCI energy",);
}

/// Test that the H4 cc-pVDZ 1.75 Angstrom fixture reproduces the expected SCF state energies and reference
/// NOCI energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF or reference NOCI energy differs from known good value outside tolerance.
#[test]
#[serial]
fn reference_noci_h4_cc_pvdz_1_75_ang_energies() {
    let (_input, _ao, expected): (_, _, ExpectedReferenceNoci) = load_test("REF_NOCI_H4_cc-pVDZ_1_75");
    let (got_scf, got_ref) = run_reference_noci_fixture("REF_NOCI_H4_cc-pVDZ_1_75");

    let mut want_scf = expected.scf_energies;
    want_scf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got_scf.len(), want_scf.len());
    for (i, (&x, &y)) in got_scf.iter().zip(want_scf.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("H4 SCF state {i}"));
    }
    assert_close(got_ref, expected.reference_noci_energy, 1e-8, "H4 reference NOCI energy",);
}

/// Test that the F2 cc-pVDZ 1.75 Angstrom fixture reproduces the expected SCF state energies and reference
/// NOCI energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF or reference NOCI energy differs from known good value outside tolerance.
#[test]
#[serial]
fn reference_noci_f2_cc_pvdz_1_75_ang_energies() {
    let (_input, _ao, expected): (_, _, ExpectedReferenceNoci) = load_test("REF_NOCI_F2_cc-pVDZ_1_75");
    let (got_scf, got_ref) = run_reference_noci_fixture("REF_NOCI_F2_cc-pVDZ_1_75");

    let mut want_scf = expected.scf_energies;
    want_scf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got_scf.len(), want_scf.len());
    for (i, (&x, &y)) in got_scf.iter().zip(want_scf.iter()).enumerate() {
        assert_close(x, y, 1e-7, &format!("F2 SCF state {i}"));
    }
    assert_close(got_ref, expected.reference_noci_energy, 1e-8, "F2 reference NOCI energy",);
}
