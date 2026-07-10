mod common;

use common::{assert_close, load_test, mpi_universe};
use ndarray::Array1;
use serde::Deserialize;
use serial_test::serial;

use noci_rs::basis::{generate_excited_basis, generate_reference_noci_basis};
use noci_rs::deterministic::{projected_energy, propagate};
use noci_rs::noci::{
    NOCIData, build_mo_cache, build_noci_hs, build_wicks_shared, calculate_noci_energy,
};
use noci_rs::scf::occ_first;

/// Expected exact energies for a deterministic NOCI-QMC fixture.
#[derive(Deserialize)]
struct ExpectedDeterministic {
    /// Expected SCF state energies.
    scf_energies: Vec<f64>,
    /// Expected reference-space NOCI energy.
    reference_noci_energy: f64,
    /// Expected deterministic NOCI-QMC energy.
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

    let basis = generate_reference_noci_basis(&ao, &mut input, None, None);
    let states = basis.states;

    let mut scf_energies: Vec<f64> = states.iter().map(|s| s.e).collect();
    scf_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut noci_reference_basis: Vec<_> =
        states.iter().filter(|s| s.noci_basis).cloned().collect();
    for (i, st) in noci_reference_basis.iter_mut().enumerate() {
        st.parent = i;
    }
    let noci_reference_basis: Vec<_> = noci_reference_basis.iter().map(occ_first).collect();
    let mocache = build_mo_cache(&ao, &noci_reference_basis, input.scf.d_tol);

    let (e_ref, c0, _dt_hs_ref) =
        calculate_noci_energy(&ao, &input, &noci_reference_basis, 1e-12, &mocache, None);

    let include_refs = true;
    let basis = generate_excited_basis(&noci_reference_basis, &input, include_refs);

    let symmetric = true;
    let data = NOCIData::new(&ao, &basis, &input, 1e-12, None).withmocache(&mocache);
    let (h, s, _d_hs) = build_noci_hs(&data, &basis, &basis, symmetric);

    let n = basis.len();
    let mut c0qmc = Array1::<f64>::zeros(n);
    if !input.write.write_deterministic_coeffs {
        for (i, ref_st) in noci_reference_basis.iter().enumerate() {
            let idx = basis
                .iter()
                .position(|qmc_st| qmc_st.label == ref_st.label)
                .unwrap();
            c0qmc[idx] = c0[i];
        }
    } else {
        c0qmc = Array1::from_elem(n, 1.0 / (n as f64).sqrt());
    }

    let es = states[0].e;

    let mut coefficients = Vec::new();
    let cfinal = propagate(&h, &s, &c0qmc, es, &mut coefficients, &input)
        .expect("deterministic propagation failed");
    let e_det = projected_energy(&h, &s, &cfinal);

    (scf_energies, e_ref, e_det)
}

/// Run SCF, reference NOCI and deterministic NOCI-QMC with Wick's intermediates and compare energies with known good energies.
/// # Arguments:
/// - `fixture`: Name of the test fixture to load.
/// # Returns
/// - `(Vec<f64>, f64, f64)`: Sorted SCF state energies, the reference NOCI energy and the
///   deterministic NOCI-QMC energy.
fn run_deterministic_fixture_wicks(fixture: &str) -> (Vec<f64>, f64, f64) {
    let (mut input, ao, _expected): (_, _, ExpectedDeterministic) = load_test(fixture);

    let basis = generate_reference_noci_basis(&ao, &mut input, None, None);
    let states = basis.states;

    let mut scf_energies: Vec<f64> = states.iter().map(|s| s.e).collect();
    scf_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut noci_reference_basis: Vec<_> =
        states.iter().filter(|s| s.noci_basis).cloned().collect();
    for (i, st) in noci_reference_basis.iter_mut().enumerate() {
        st.parent = i;
    }
    let noci_reference_basis: Vec<_> = noci_reference_basis.iter().map(occ_first).collect();
    let mocache = build_mo_cache(&ao, &noci_reference_basis, input.scf.d_tol);

    let (_mpi_lock, universe) = mpi_universe();
    let world = universe.world();

    let wicks = build_wicks_shared::<f64>(&world, &ao, &noci_reference_basis, 1e-12, &input);
    let wicks_view = wicks.view();

    let (e_ref, c0, _dt_hs_ref) = calculate_noci_energy(
        &ao,
        &input,
        &noci_reference_basis,
        1e-12,
        &mocache,
        Some(wicks_view),
    );

    let include_refs = true;
    let basis = generate_excited_basis(&noci_reference_basis, &input, include_refs);

    let symmetric = true;
    let data = NOCIData::new(&ao, &basis, &input, 1e-12, Some(wicks_view)).withmocache(&mocache);
    let (h, s, _d_hs) = build_noci_hs(&data, &basis, &basis, symmetric);

    let n = basis.len();
    let mut c0qmc = Array1::<f64>::zeros(n);
    if !input.write.write_deterministic_coeffs {
        for (i, ref_st) in noci_reference_basis.iter().enumerate() {
            let idx = basis
                .iter()
                .position(|qmc_st| qmc_st.label == ref_st.label)
                .unwrap();
            c0qmc[idx] = c0[i];
        }
    } else {
        c0qmc = Array1::from_elem(n, 1.0 / (n as f64).sqrt());
    }

    let es = states[0].e;

    let mut coefficients = Vec::new();
    let cfinal = propagate(&h, &s, &c0qmc, es, &mut coefficients, &input)
        .expect("deterministic propagation failed");
    let e_det = projected_energy(&h, &s, &cfinal);

    (scf_energies, e_ref, e_det)
}

/// Test that the H2 STO-3G 1.5 Angstrom fixture reproduces the expected SCF state energies,
/// reference NOCI energy and deterministic NOCI-QMC energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF, reference NOCI or deterministic NOCI-QMC energy differs from known good value
///   outside tolerance.
#[test]
#[serial]
fn deterministic_h2_sto_3g_1_5_ang_energies() {
    let (_input, _ao, expected): (_, _, ExpectedDeterministic) = load_test("DET_H2_STO-3G_1_5");
    let (got_scf, got_ref, got_det) = run_deterministic_fixture("DET_H2_STO-3G_1_5");

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
    assert_close(
        got_det,
        expected.deterministic_noci_energy,
        1e-8,
        "H2 deterministic energy",
    );
}

/// Test that the H2 STO-3G 1.5 Angstrom fixture reproduces the expected SCF state energies,
/// reference NOCI energy and deterministic NOCI-QMC energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF, reference NOCI or deterministic NOCI-QMC energy differs from known good value
///   outside tolerance.
#[test]
#[serial]
fn deterministic_h2_sto_3g_1_5_ang_energies_wicks() {
    let (_input, _ao, expected): (_, _, ExpectedDeterministic) =
        load_test("DET_H2_STO-3G_1_5_WICKS");
    let (got_scf, got_ref, got_det) = run_deterministic_fixture_wicks("DET_H2_STO-3G_1_5_WICKS");

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
    assert_close(
        got_det,
        expected.deterministic_noci_energy,
        1e-8,
        "H2 deterministic energy",
    );
}

/// Test that the H2 STO-3G 1.5 Angstrom fixture agrees with and without Wick's intermediates.
/// # Panics
/// - If the number of SCF states differs between implementations.
/// - If SCF, reference NOCI or deterministic NOCI-QMC energy differs between implementations
///   outside tolerance.
#[test]
#[serial]
fn deterministic_h2_sto_3g_1_5_ang_energies_agree() {
    let (got_scf, got_ref, got_det) = run_deterministic_fixture("DET_H2_STO-3G_1_5");
    let (got_scf_wicks, got_ref_wicks, got_det_wicks) =
        run_deterministic_fixture_wicks("DET_H2_STO-3G_1_5_WICKS");

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
    assert_close(
        got_det,
        got_det_wicks,
        1e-8,
        "H2 deterministic Wicks agreement",
    );
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

    assert_close(
        got_ref,
        expected.reference_noci_energy,
        1e-8,
        "F2 reference NOCI energy",
    );
    assert_close(
        got_det,
        expected.deterministic_noci_energy,
        1e-8,
        "F2 deterministic energy",
    );
}

/// Test that the F2 sto-3g 1.75 Angstrom fixture reproduces the expected SCF state energies,
/// reference NOCI energy and deterministic NOCI-QMC energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF, reference NOCI or deterministic NOCI-QMC energy differs from known good value
///   outside tolerance.
#[test]
#[serial]
fn deterministic_f2_sto_3g_1_75_ang_energies_wicks() {
    let (_input, _ao, expected): (_, _, ExpectedDeterministic) =
        load_test("DET_F2_sto-3g_1_75_WICKS");
    let (got_scf, got_ref, got_det) = run_deterministic_fixture_wicks("DET_F2_sto-3g_1_75_WICKS");

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
    assert_close(
        got_det,
        expected.deterministic_noci_energy,
        1e-8,
        "F2 deterministic energy",
    );
}

/// Test that the F2 sto-3g 1.75 Angstrom fixture agrees with and without Wick's intermediates.
/// # Panics
/// - If the number of SCF states differs between implementations.
/// - If SCF, reference NOCI or deterministic NOCI-QMC energy differs between implementations
///   outside tolerance.
#[test]
#[serial]
fn deterministic_f2_sto_3g_1_75_ang_energies_agree() {
    let (got_scf, got_ref, got_det) = run_deterministic_fixture("DET_F2_sto-3g_1_75");
    let (got_scf_wicks, got_ref_wicks, got_det_wicks) =
        run_deterministic_fixture_wicks("DET_F2_sto-3g_1_75_WICKS");

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
    assert_close(
        got_det,
        got_det_wicks,
        1e-8,
        "F2 deterministic Wicks agreement",
    );
}
