mod common;

// External crate imports.
use noci_rs::basis::generate_reference_noci_basis;
use noci_rs::determinant::{NOCIIndex, NOCISpace};
use noci_rs::elements::{NOCIData, build_mo_cache, build_wicks_shared};
use noci_rs::nocc::{JMEquations, jm_space, propagate_populations, report_converged};
use noci_rs::noci::{build_noci_hs, calculate_noci_energy};
use noci_rs::scf::occ_first;
use serde::Deserialize;
use serial_test::serial;

// Parent/sibling imports.
use self::common::{assert_close, load_test, mpi_universe};

/// Expected exact energies for a deterministic JM-NOCCMC fixture.
#[derive(Deserialize)]
struct ExpectedJMNOCCMC {
    /// Expected SCF state energies.
    scf_energies: Vec<f64>,
    /// Expected reference-space NOCI energy.
    reference_noci_energy: f64,
    /// Expected deterministic JM-NOCCMC ground-state energy.
    jmnoccmc_energy: f64,
}

/// Run SCF, reference NOCI and deterministic JM-NOCCMC, with Wick's intermediates when the
/// fixture enables them.
/// # Arguments:
/// - `fixture`: Name of the test fixture to load.
/// # Returns
/// - `(Vec<f64>, f64, f64)`: Sorted SCF state energies, the reference NOCI energy and the
///   deterministic JM-NOCCMC ground-state energy.
fn run_jmnoccmc_fixture(fixture: &str) -> (Vec<f64>, f64, f64) {
    let (mut input, ao, _expected): (_, _, ExpectedJMNOCCMC) = load_test(fixture);

    let basis = generate_reference_noci_basis(&ao, &mut input, None, None);
    let states = basis.states;

    let mut scf_energies: Vec<f64> = states.iter().map(|s| s.e).collect();
    scf_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let noci_reference_basis: Vec<_> = states.iter().filter(|s| s.noci_basis).cloned().collect();
    let noci_reference_basis: Vec<_> = noci_reference_basis.iter().map(occ_first).collect();
    let reference_space = NOCISpace::from_scf(&noci_reference_basis);
    let mocache = build_mo_cache(&ao, reference_space.parents(), input.scf.d_tol);

    let mpi = input.wicks.enabled.then(mpi_universe);
    let wicks = mpi.as_ref().map(|(_mpi_lock, universe)| {
        build_wicks_shared::<f64>(
            &universe.world(),
            &ao,
            reference_space.parents(),
            1e-12,
            &input,
        )
    });
    let wicks_view = wicks.as_ref().map(|w| w.view());

    let (e_ref, c0, _dt_hs_ref) =
        calculate_noci_energy(&ao, &input, &reference_space, 1e-12, &mocache, wicks_view);

    // Rectangular Hamiltonian and overlap between the projection and expansion spaces.
    let options = input.jmnoccmc.as_ref().unwrap();
    let space = jm_space(&reference_space, options);
    let rows = space.rows();
    let columns = (0..space.space.len()).map(NOCIIndex).collect::<Vec<_>>();
    let data = NOCIData::new(&ao, &space.space, &input, 1e-12, wicks_view).withmocache(&mocache);
    let (h, s, _d_hs) = build_noci_hs(&data, &rows, &columns, false);

    let jm = JMEquations::new(space, h, s, None);
    let mut history = Vec::new();
    let populations =
        propagate_populations(&jm, options, (c0.as_slice().unwrap(), e_ref), &mut history)
            .expect("deterministic JM-NOCCMC propagation failed");
    let e_jm = report_converged(&jm, &populations, &history, options, &input.write.write_dir);

    (scf_energies, e_ref, e_jm)
}

/// Compare a JM-NOCCMC fixture with its known good energies.
/// # Arguments:
/// - `fixture`: Name of the test fixture to load.
/// - `label`: Description printed if an assertion fails.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF, reference NOCI or deterministic JM-NOCCMC energy differs from known good value
///   outside tolerance.
fn check_jmnoccmc_fixture(
    fixture: &str,
    label: &str,
) {
    let (_input, _ao, expected): (_, _, ExpectedJMNOCCMC) = load_test(fixture);
    let (got_scf, got_ref, got_jm) = run_jmnoccmc_fixture(fixture);

    let mut want_scf = expected.scf_energies;
    want_scf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got_scf.len(), want_scf.len());
    for (i, (&x, &y)) in got_scf.iter().zip(want_scf.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("{label} SCF state {i}"));
    }

    assert_close(
        got_ref,
        expected.reference_noci_energy,
        1e-8,
        &format!("{label} reference NOCI energy"),
    );
    assert_close(
        got_jm,
        expected.jmnoccmc_energy,
        1e-8,
        &format!("{label} deterministic JM-NOCCMC energy"),
    );
}

/// Test that the H4 STO-3G 1.5 Angstrom fixture with two UHF references reproduces, with Wick's
/// intermediates, the expected SCF state energies, reference NOCI energy and deterministic
/// JM-NOCCMC CCSD energy.
/// # Panics
/// - If any energy differs from its known good value outside tolerance.
#[test]
#[serial]
fn jmnoccmc_h4_sto_3g_1_5_ang_energies_wicks() {
    check_jmnoccmc_fixture("JM_H4_STO-3G_1_5_WICKS", "H4");
}

/// Test that the H4 STO-3G 1.5 Angstrom fixture with two UHF references at the complete cluster
/// rank reproduces the expected energies, whose JM-NOCCMC ground state is the FCI ground state.
/// # Panics
/// - If any energy differs from its known good value outside tolerance.
#[test]
#[serial]
fn jmnoccmc_h4_sto_3g_1_5_ang_energies_complete() {
    check_jmnoccmc_fixture("JM_H4_STO-3G_1_5_COMPLETE", "H4 complete");
}
