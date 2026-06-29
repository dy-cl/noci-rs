mod common;

use common::{assert_close, load_test, mpi_universe};
use serde::Deserialize;
use serial_test::serial;

use noci_rs::basis::generate_reference_noci_basis;
use noci_rs::noci::{build_mo_cache, build_wicks_shared, calculate_noci_energy};

/// Expected exact energies for a reference NOCI fixture.
#[derive(Deserialize)]
struct ExpectedReferenceNoci {
    /// Expected SCF state energies.
    scf_energies: Vec<f64>,
    /// Expected reference-space NOCI energy.
    reference_noci_energy: f64,
}

/// Run SCF and reference NOCI and compare energies with known good energies.
/// # Arguments:
/// - `fixture`: Name of the test fixture to load.
/// # Returns
/// - `(Vec<f64>, f64)`: Sorted SCF state energies and the reference NOCI energy.
fn run_reference_noci_fixture(fixture: &str) -> (Vec<f64>, f64) {
    let (mut input, ao, _expected): (_, _, ExpectedReferenceNoci) = load_test(fixture);

    let basis = generate_reference_noci_basis(&ao, &mut input, None, None);
    let states = basis.states;

    let mut scf_energies: Vec<f64> = states.iter().map(|s| s.e).collect();

    scf_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut basis: Vec<_> = states.into_iter().filter(|s| s.noci_basis).collect();
    for (i, st) in basis.iter_mut().enumerate() {
        st.parent = i;
    }

    let mocache = build_mo_cache(&ao, &basis, input.scf.d_tol);
    let (e_ref, _coeffs, _dt_hs) =
        calculate_noci_energy(&ao, &input, &basis, 1e-12, &mocache, None);

    (scf_energies, e_ref)
}

/// Run SCF and reference NOCI with Wick's intermediates and compare energies with known good energies.
/// # Arguments:
/// - `fixture`: Name of the test fixture to load.
/// # Returns
/// - `(Vec<f64>, f64)`: Sorted SCF state energies and the reference NOCI energy.
fn run_reference_noci_fixture_wicks(fixture: &str) -> (Vec<f64>, f64) {
    let (mut input, ao, _expected): (_, _, ExpectedReferenceNoci) = load_test(fixture);

    let basis = generate_reference_noci_basis(&ao, &mut input, None, None);
    let states = basis.states;

    let mut scf_energies: Vec<f64> = states.iter().map(|s| s.e).collect();

    scf_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut basis: Vec<_> = states.into_iter().filter(|s| s.noci_basis).collect();
    for (i, st) in basis.iter_mut().enumerate() {
        st.parent = i;
    }

    let (_mpi_lock, universe) = mpi_universe();
    let world = universe.world();

    let mocache = build_mo_cache(&ao, &basis, input.scf.d_tol);
    let wicks = build_wicks_shared::<f64>(&world, &ao, &basis, 1e-12, &input);
    let (e_ref, _coeffs, _dt_hs) =
        calculate_noci_energy(&ao, &input, &basis, 1e-12, &mocache, Some(wicks.view()));

    (scf_energies, e_ref)
}

/// Test that the H2 STO-3G 1.5 Angstrom fixture reproduces the expected SCF state energies and reference
/// NOCI energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF or reference NOCI energy differs from known good value outside tolerance.
#[test]
#[serial]
fn reference_noci_h2_sto_3g_1_5_ang_energies() {
    let (_input, _ao, expected): (_, _, ExpectedReferenceNoci) =
        load_test("REF_NOCI_H2_STO-3G_1_5");
    let (got_scf, got_ref) = run_reference_noci_fixture("REF_NOCI_H2_STO-3G_1_5");

    let mut want_scf = expected.scf_energies;
    want_scf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got_scf.len(), want_scf.len());
    for (i, (&x, &y)) in got_scf.iter().zip(want_scf.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("H2 SCF state {i}"));
    }
    assert_close(
        got_ref,
        expected.reference_noci_energy,
        1e-8,
        "H2 reference NOCI energy",
    );
}

/// Test that the H2 STO-3G 1.5 Angstrom fixture reproduces the expected SCF state energies and reference
/// NOCI energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF or reference NOCI energy differs from known good value outside tolerance.
#[test]
#[serial]
fn reference_noci_h2_sto_3g_1_5_ang_energies_wicks() {
    let (_input, _ao, expected): (_, _, ExpectedReferenceNoci) =
        load_test("REF_NOCI_H2_STO-3G_1_5_WICKS");
    let (got_scf, got_ref) = run_reference_noci_fixture_wicks("REF_NOCI_H2_STO-3G_1_5_WICKS");

    let mut want_scf = expected.scf_energies;
    want_scf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got_scf.len(), want_scf.len());
    for (i, (&x, &y)) in got_scf.iter().zip(want_scf.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("H2 SCF state {i}"));
    }
    assert_close(
        got_ref,
        expected.reference_noci_energy,
        1e-8,
        "H2 reference NOCI energy",
    );
}

/// Test that the H2 STO-3G 1.5 Angstrom fixture agrees with and without Wick's intermediates.
/// # Panics
/// - If the number of SCF states differs between implementations.
/// - If SCF or reference NOCI energy differs between implementations outside tolerance.
#[test]
#[serial]
fn reference_noci_h2_sto_3g_1_5_ang_energies_agree() {
    let (got_scf, got_ref) = run_reference_noci_fixture("REF_NOCI_H2_STO-3G_1_5");
    let (got_scf_wicks, got_ref_wicks) =
        run_reference_noci_fixture_wicks("REF_NOCI_H2_STO-3G_1_5_WICKS");

    assert_eq!(got_scf.len(), got_scf_wicks.len());
    for (i, (&x, &y)) in got_scf.iter().zip(got_scf_wicks.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("H2 SCF state {i} Wicks agreement"));
    }
    assert_close(
        got_ref,
        got_ref_wicks,
        1e-8,
        "H2 reference NOCI Wicks agreement",
    );
}

/// Test that the LiH 6-31G 2.8 Angstrom fixture reproduces the expected SCF state energies and reference
/// NOCI energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF or reference NOCI energy differs from known good value outside tolerance.
#[test]
#[serial]
fn reference_noci_lih_6_31g_2_8_ang_energies() {
    let (_input, _ao, expected): (_, _, ExpectedReferenceNoci) =
        load_test("REF_NOCI_LiH_6-31G_2_8");
    let (got_scf, got_ref) = run_reference_noci_fixture("REF_NOCI_LiH_6-31G_2_8");

    let mut want_scf = expected.scf_energies;
    want_scf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got_scf.len(), want_scf.len());
    for (i, (&x, &y)) in got_scf.iter().zip(want_scf.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("LiH SCF state {i}"));
    }
    assert_close(
        got_ref,
        expected.reference_noci_energy,
        1e-8,
        "LiH reference NOCI energy",
    );
}

/// Test that the LiH 6-31G 2.8 Angstrom fixture reproduces the expected SCF state energies and reference
/// NOCI energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF or reference NOCI energy differs from known good value outside tolerance.
#[test]
#[serial]
fn reference_noci_lih_6_31g_2_8_ang_energies_wicks() {
    let (_input, _ao, expected): (_, _, ExpectedReferenceNoci) =
        load_test("REF_NOCI_LiH_6-31G_2_8_WICKS");
    let (got_scf, got_ref) = run_reference_noci_fixture_wicks("REF_NOCI_LiH_6-31G_2_8_WICKS");

    let mut want_scf = expected.scf_energies;
    want_scf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got_scf.len(), want_scf.len());
    for (i, (&x, &y)) in got_scf.iter().zip(want_scf.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("LiH SCF state {i}"));
    }
    assert_close(
        got_ref,
        expected.reference_noci_energy,
        1e-8,
        "LiH reference NOCI energy",
    );
}

/// Test that the LiH 6-31G 2.8 Angstrom fixture agrees with and without Wick's intermediates.
/// # Panics
/// - If the number of SCF states differs between implementations.
/// - If SCF or reference NOCI energy differs between implementations outside tolerance.
#[test]
#[serial]
fn reference_noci_lih_6_31g_2_8_ang_energies_agree() {
    let (got_scf, got_ref) = run_reference_noci_fixture("REF_NOCI_LiH_6-31G_2_8");
    let (got_scf_wicks, got_ref_wicks) =
        run_reference_noci_fixture_wicks("REF_NOCI_LiH_6-31G_2_8_WICKS");

    assert_eq!(got_scf.len(), got_scf_wicks.len());
    for (i, (&x, &y)) in got_scf.iter().zip(got_scf_wicks.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("LiH SCF state {i} Wicks agreement"));
    }
    assert_close(
        got_ref,
        got_ref_wicks,
        1e-8,
        "LiH reference NOCI Wicks agreement",
    );
}

/// Test that the H4 STO-3G 1.75 Angstrom fixture reproduces the expected SCF state energies and reference
/// NOCI energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF or reference NOCI energy differs from known good value outside tolerance.
#[test]
#[serial]
fn reference_noci_h4_sto_3g_1_75_ang_energies() {
    let (_input, _ao, expected): (_, _, ExpectedReferenceNoci) =
        load_test("REF_NOCI_H4_STO-3G_1_75");
    let (got_scf, got_ref) = run_reference_noci_fixture("REF_NOCI_H4_STO-3G_1_75");

    let mut want_scf = expected.scf_energies;
    want_scf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got_scf.len(), want_scf.len());
    for (i, (&x, &y)) in got_scf.iter().zip(want_scf.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("H4 SCF state {i}"));
    }
    assert_close(
        got_ref,
        expected.reference_noci_energy,
        1e-8,
        "H4 reference NOCI energy",
    );
}

/// Test that the H4 STO-3G 1.75 Angstrom fixture reproduces the expected SCF state energies and reference
/// NOCI energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF or reference NOCI energy differs from known good value outside tolerance.
#[test]
#[serial]
fn reference_noci_h4_sto_3g_1_75_ang_energies_wicks() {
    let (_input, _ao, expected): (_, _, ExpectedReferenceNoci) =
        load_test("REF_NOCI_H4_STO-3G_1_75_WICKS");
    let (got_scf, got_ref) = run_reference_noci_fixture_wicks("REF_NOCI_H4_STO-3G_1_75_WICKS");

    let mut want_scf = expected.scf_energies;
    want_scf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got_scf.len(), want_scf.len());
    for (i, (&x, &y)) in got_scf.iter().zip(want_scf.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("H4 SCF state {i}"));
    }
    assert_close(
        got_ref,
        expected.reference_noci_energy,
        1e-8,
        "H4 reference NOCI energy",
    );
}

/// Test that the H4 STO-3G 1.75 Angstrom fixture agrees with and without Wick's intermediates.
/// # Panics
/// - If the number of SCF states differs between implementations.
/// - If SCF or reference NOCI energy differs between implementations outside tolerance.
#[test]
#[serial]
fn reference_noci_h4_sto_3g_1_75_ang_energies_agree() {
    let (got_scf, got_ref) = run_reference_noci_fixture("REF_NOCI_H4_STO-3G_1_75");
    let (got_scf_wicks, got_ref_wicks) =
        run_reference_noci_fixture_wicks("REF_NOCI_H4_STO-3G_1_75_WICKS");

    assert_eq!(got_scf.len(), got_scf_wicks.len());
    for (i, (&x, &y)) in got_scf.iter().zip(got_scf_wicks.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("H4 SCF state {i} Wicks agreement"));
    }
    assert_close(
        got_ref,
        got_ref_wicks,
        1e-8,
        "H4 reference NOCI Wicks agreement",
    );
}

/// Test that the F2 STO-3G 1.75 Angstrom fixture reproduces the expected SCF state energies and reference
/// NOCI energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF or reference NOCI energy differs from known good value outside tolerance.
#[test]
#[serial]
fn reference_noci_f2_sto_3g_1_75_ang_energies() {
    let (_input, _ao, expected): (_, _, ExpectedReferenceNoci) =
        load_test("REF_NOCI_F2_STO-3G_1_75");
    let (got_scf, got_ref) = run_reference_noci_fixture("REF_NOCI_F2_STO-3G_1_75");

    let mut want_scf = expected.scf_energies;
    want_scf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got_scf.len(), want_scf.len());
    for (i, (&x, &y)) in got_scf.iter().zip(want_scf.iter()).enumerate() {
        assert_close(x, y, 1e-7, &format!("F2 SCF state {i}"));
    }
    assert_close(
        got_ref,
        expected.reference_noci_energy,
        1e-8,
        "F2 reference NOCI energy",
    );
}

/// Test that the F2 STO-3G 1.75 Angstrom fixture reproduces the expected SCF state energies and reference
/// NOCI energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF or reference NOCI energy differs from known good value outside tolerance.
#[test]
#[serial]
fn reference_noci_f2_sto_3g_1_75_ang_energies_wicks() {
    let (_input, _ao, expected): (_, _, ExpectedReferenceNoci) =
        load_test("REF_NOCI_F2_STO-3G_1_75_WICKS");
    let (got_scf, got_ref) = run_reference_noci_fixture_wicks("REF_NOCI_F2_STO-3G_1_75_WICKS");

    let mut want_scf = expected.scf_energies;
    want_scf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got_scf.len(), want_scf.len());
    for (i, (&x, &y)) in got_scf.iter().zip(want_scf.iter()).enumerate() {
        assert_close(x, y, 1e-7, &format!("F2 SCF state {i}"));
    }
    assert_close(
        got_ref,
        expected.reference_noci_energy,
        1e-8,
        "F2 reference NOCI energy",
    );
}

/// Test that the F2 STO-3G 1.75 Angstrom fixture agrees with and without Wick's intermediates.
/// # Panics
/// - If the number of SCF states differs between implementations.
/// - If SCF or reference NOCI energy differs between implementations outside tolerance.
#[test]
#[serial]
fn reference_noci_f2_sto_3g_1_75_ang_energies_agree() {
    let (got_scf, got_ref) = run_reference_noci_fixture("REF_NOCI_F2_STO-3G_1_75");
    let (got_scf_wicks, got_ref_wicks) =
        run_reference_noci_fixture_wicks("REF_NOCI_F2_STO-3G_1_75_WICKS");

    assert_eq!(got_scf.len(), got_scf_wicks.len());
    for (i, (&x, &y)) in got_scf.iter().zip(got_scf_wicks.iter()).enumerate() {
        assert_close(x, y, 1e-7, &format!("F2 SCF state {i} Wicks agreement"));
    }
    assert_close(
        got_ref,
        got_ref_wicks,
        1e-8,
        "F2 reference NOCI Wicks agreement",
    );
}
