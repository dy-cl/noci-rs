mod common;

// External crate imports.
use noci_rs::basis::generate_reference_noci_basis;
use noci_rs::determinant::NOCISpace;
use noci_rs::elements::build_mo_cache;
use noci_rs::noci::calculate_noci_energy;
use noci_rs::noci::optimised::reshf_optimise;
use serde::Deserialize;
use serial_test::serial;

// Parent/sibling imports.
use self::common::{assert_close, load_test};

/// Expected exact energies for a ResHF or reference NOCI fixture.
#[derive(Deserialize)]
struct ExpectedReferenceNoci {
    /// Expected SCF state energies, after ResHF for a ResHF fixture.
    scf_energies: Vec<f64>,
    /// Expected reference-space NOCI energy.
    reference_noci_energy: f64,
}

/// Run SCF, ResHF and reference NOCI for a fixture.
/// # Arguments:
/// - `fixture`: Name of the test fixture to load.
/// # Returns
/// - `(Vec<f64>, f64)`: Sorted ResHF determinant energies and the reference NOCI energy.
fn run_reshf_fixture(fixture: &str) -> (Vec<f64>, f64) {
    let (mut input, ao, _expected): (_, _, ExpectedReferenceNoci) = load_test(fixture);

    let basis = generate_reference_noci_basis(&ao, &mut input, None, None);
    let states = reshf_optimise(&ao, &input, &basis.states);

    let mut scf_energies: Vec<f64> = states.iter().map(|s| s.e).collect();

    scf_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let basis: Vec<_> = states.into_iter().filter(|s| s.noci_basis).collect();
    let space = NOCISpace::from_scf(&basis);

    let mocache = build_mo_cache(&ao, space.parents(), input.scf.d_tol);
    let (e_ref, _coeffs, _dt_hs) =
        calculate_noci_energy(&ao, &input, &space, 1e-12, &mocache, None);

    (scf_energies, e_ref)
}

/// Compare ResHF determinant and reference NOCI energies with a fixture's expected energies.
/// # Arguments:
/// - `fixture`: Name of the test fixture to load.
/// - `system`: System label used in failure messages.
/// # Returns
/// - `()`: Panics if any energy differs from its known good value outside tolerance.
fn check_reshf_fixture(
    fixture: &str,
    system: &str,
) {
    let (_input, _ao, expected): (_, _, ExpectedReferenceNoci) = load_test(fixture);
    let (got_scf, got_ref) = run_reshf_fixture(fixture);

    let mut want_scf = expected.scf_energies;
    want_scf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got_scf.len(), want_scf.len());
    for (i, (&x, &y)) in got_scf.iter().zip(want_scf.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("{system} ResHF determinant {i}"));
    }
    assert_close(
        got_ref,
        expected.reference_noci_energy,
        1e-8,
        &format!("{system} ResHF reference NOCI energy"),
    );
}

/// Test that the H2 3-21G 1.5 Angstrom fixture reproduces the expected ResHF determinant energies
/// and reference NOCI energy.
/// # Panics
/// - If the number of states differs from the stored reference.
/// - If a ResHF determinant or reference NOCI energy differs from known good value outside
///   tolerance.
#[test]
#[serial]
fn reshf_h2_3_21g_1_5_ang_energies() {
    check_reshf_fixture("RESHF_H2_3-21G_1_5", "H2");
}

/// Test that the H4 STO-3G 1.75 Angstrom fixture reproduces the expected ResHF determinant
/// energies and reference NOCI energy.
/// # Panics
/// - If the number of states differs from the stored reference.
/// - If a ResHF determinant or reference NOCI energy differs from known good value outside
///   tolerance.
#[test]
#[serial]
fn reshf_h4_sto_3g_1_75_ang_energies() {
    check_reshf_fixture("RESHF_H4_STO-3G_1_75", "H4");
}

/// Test that ResHF lowers the reference NOCI energy of the H4 STO-3G 1.75 Angstrom SCF states.
/// # Panics
/// - If the ResHF reference NOCI energy is not below the reference NOCI energy of the SCF states.
#[test]
#[serial]
fn reshf_h4_sto_3g_1_75_ang_lowers_reference_noci() {
    let (_input, _ao, scf): (_, _, ExpectedReferenceNoci) = load_test("REF_NOCI_H4_STO-3G_1_75");
    let (_got_scf, got_ref) = run_reshf_fixture("RESHF_H4_STO-3G_1_75");

    assert!(
        got_ref < scf.reference_noci_energy,
        "H4 ResHF reference NOCI energy {got_ref} is not below the SCF value {}",
        scf.reference_noci_energy
    );
}
