mod common;

use common::{assert_close, load_test};
use serde::Deserialize;
use serial_test::serial;

use noci_rs::PostSCFData;
use noci_rs::basis::generate_reference_noci_basis;
use noci_rs::noci::{build_mo_cache, calculate_noci_energy};
use noci_rs::snoci::snoci_step;

#[derive(Deserialize)]
struct ExpectedSNOCI {
    scf_energies: Vec<f64>,
    reference_noci_energy: f64,
    snoci_energy: f64,
}

/// Run SCF, reference NOCI and selected NOCI and compare energies with known good energies.
/// # Arguments:
/// - `fixture`: Name of the test fixture to load.
/// # Returns
/// - `(Vec<f64>, f64, f64)`: Sorted SCF state energies, the reference NOCI energy and the
///   selected NOCI energy.
fn run_snoci_fixture(fixture: &str) -> (Vec<f64>, f64, f64) {
    let (mut input, ao, _expected): (_, _, ExpectedSNOCI) = load_test(fixture);

    let basis = generate_reference_noci_basis(&ao, &mut input, None, None);
    let states = basis.states;

    let mut scf_energies: Vec<f64> = states.iter().map(|s| s.e).collect();
    scf_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut noci_reference_basis: Vec<_> =
        states.iter().filter(|s| s.noci_basis).cloned().collect();
    for (i, st) in noci_reference_basis.iter_mut().enumerate() {
        st.parent = i;
    }

    let mocache = build_mo_cache(&ao, &noci_reference_basis, input.scf.d_tol);

    let (e_ref, _c0, _dt_hs_ref) =
        calculate_noci_energy(&ao, &input, &noci_reference_basis, 1e-12, &mocache, None);

    let post = PostSCFData {
        ao: &ao,
        states: &states,
        noci_reference_basis: &noci_reference_basis,
        mocache: &mocache,
        tol: 1e-12,
    };

    let universe = mpi::initialize().expect("MPI initialisation failed");
    let world = universe.world();

    let result = snoci_step(&post, &noci_reference_basis, &input, None, &world);

    assert!(
        !result.pt2.is_empty(),
        "SNOCI did not produce a NOCI-PT2 result"
    );
    for pt2 in &result.pt2 {
        assert!(
            pt2.gmres_converged,
            "SNOCI GMRES failed to converge: residual {}",
            pt2.gmres_residual,
        );
    }

    (scf_energies, e_ref, result.ecurrent)
}

/// Test that the H2 cc-pVDZ 1.5 Angstrom fixture reproduces the expected SCF state energies,
/// reference NOCI energy and selected NOCI energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF, reference NOCI or selected NOCI energy differs from the known good value
///   outside tolerance.
/// - If the NOCI-PT2 GMRES solve does not converge.
#[test]
#[serial]
fn snoci_h2_cc_pvdz_1_5_ang_energies() {
    let (_input, _ao, expected): (_, _, ExpectedSNOCI) =
        load_test("SNOCI_H2_cc-pVDZ_1_5");
    let (got_scf, got_ref, got_snoci) = run_snoci_fixture("SNOCI_H2_cc-pVDZ_1_5");

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
        got_snoci,
        expected.snoci_energy,
        1e-8,
        "H2 selected NOCI energy",
    );
}
