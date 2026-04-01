mod common;

use common::{assert_close, load_test};
use serial_test::serial;

use noci_rs::noci::{calculate_noci_energy};
use noci_rs::basis::{generate_reference_noci_basis};

/// Run SCF and reference NOCI and compare energies with known good energies. 
/// # Arguments:
/// - `fixture`: Name of the test fixture to load.
/// # Returns
/// - `(Vec<f64>, f64)`: Sorted SCF state energies and the reference NOCI energy.
fn run_reference_noci_fixture(fixture: &str) -> (Vec<f64>, f64) {
    let (mut input, ao, _expected) = load_test(fixture);

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

/// Test that the H2 cc-pVDZ fixture reproduces the expected SCF state energies and reference
/// NOCI energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF or reference NOCI energy differs from known good value outside tolerance.
#[test]
#[serial]
fn h2_energies() {
    let (_input, _ao, expected) = load_test("H2_cc-pVDZ");
    let (got_scf, got_ref) = run_reference_noci_fixture("H2_cc-pVDZ");

    let mut want_scf = expected.scf_energies.unwrap();
    want_scf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got_scf.len(), want_scf.len());
    for (i, (&x, &y)) in got_scf.iter().zip(want_scf.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("H2 SCF state {i}"));
    }
    assert_close(got_ref, expected.reference_noci_energy.unwrap(), 1e-8, "H2 reference NOCI energy",);
}

/// Test that the LiH cc-pVDZ fixture reproduces the expected SCF state energies and reference
/// NOCI energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF or reference NOCI energy differs from known good value outside tolerance.
#[test]
#[serial]
fn lih_energies() {
    let (_input, _ao, expected) = load_test("LiH_cc-pVDZ");
    let (got_scf, got_ref) = run_reference_noci_fixture("LiH_cc-pVDZ");

    let mut want_scf = expected.scf_energies.unwrap();
    want_scf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got_scf.len(), want_scf.len());
    for (i, (&x, &y)) in got_scf.iter().zip(want_scf.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("LiH SCF state {i}"));
    }
    assert_close(got_ref, expected.reference_noci_energy.unwrap(), 1e-8, "LiH reference NOCI energy",);
}

/// Test that the H4 cc-pVDZ fixture reproduces the expected SCF state energies and reference
/// NOCI energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF or reference NOCI energy differs from known good value outside tolerance.
#[test]
#[serial]
fn h4_energies() {
    let (_input, _ao, expected) = load_test("H4_cc-pVDZ");
    let (got_scf, got_ref) = run_reference_noci_fixture("H4_cc-pVDZ");

    let mut want_scf = expected.scf_energies.unwrap();
    want_scf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got_scf.len(), want_scf.len());
    for (i, (&x, &y)) in got_scf.iter().zip(want_scf.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("H4 SCF state {i}"));
    }
    assert_close(got_ref, expected.reference_noci_energy.unwrap(), 1e-8, "H4 reference NOCI energy",);
}

/// Test that the F2 cc-pVDZ fixture reproduces the expected SCF state energies and reference
/// NOCI energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF or reference NOCI energy differs from known good value outside tolerance.
#[test]
#[serial]
fn f2_energies() {
    let (_input, _ao, expected) = load_test("F2_cc-pVDZ");
    let (got_scf, got_ref) = run_reference_noci_fixture("F2_cc-pVDZ");

    let mut want_scf = expected.scf_energies.unwrap();
    want_scf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got_scf.len(), want_scf.len());
    for (i, (&x, &y)) in got_scf.iter().zip(want_scf.iter()).enumerate() {
        assert_close(x, y, 1e-7, &format!("F2 SCF state {i}"));
    }
    assert_close(got_ref, expected.reference_noci_energy.unwrap(), 1e-8, "F2 reference NOCI energy",);
}
