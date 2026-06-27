mod common;

use common::{assert_close, load_test};
use serde::Deserialize;
use serial_test::serial;
use rayon::ThreadPoolBuilder;

use noci_rs::basis::{generate_excited_basis, generate_reference_noci_basis};
use noci_rs::noci::{
    build_mo_cache, build_wicks_shared, calculate_noci_energy, NOCIData,
};
use noci_rs::stochastic::qmc_step;

#[derive(Deserialize)]
struct ExpectedQMC {
    scf_energies: Vec<f64>,
    reference_noci_energy: f64,
    qmc_noci_energy: f64,
}

/// Run SCF, reference NOCI and stochastic NOCI-QMC and compare energies with known good energies.
/// # Arguments:
/// - `fixture`: Name of the test fixture to load.
/// # Returns
/// - `(Vec<f64>, f64, f64)`: Sorted SCF state energies, the reference NOCI energy and the
///   stochastic NOCI-QMC energy.
fn run_qmc_fixture(fixture: &str) -> (Vec<f64>, f64, f64) {
    let (mut input, ao, _expected): (_, _, ExpectedQMC) = load_test(fixture);

    let basis = generate_reference_noci_basis(&ao, &mut input, None, None);
    let states = basis.states;

    let mut scf_energies: Vec<f64> = states.iter().map(|s| s.e).collect();
    scf_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut noci_reference_basis: Vec<_> =
        states.iter().filter(|s| s.noci_basis).cloned().collect();
    for (i, st) in noci_reference_basis.iter_mut().enumerate() {
        st.parent = i;
    }

    let universe = mpi::initialize().expect("MPI initialisation failed");
    let world = universe.world();

    let mocache = build_mo_cache(&ao, &noci_reference_basis, input.scf.d_tol);

    let wicks = if input.wicks.enabled {
        Some(build_wicks_shared::<f64>(
            &world,
            &ao,
            &noci_reference_basis,
            1e-12,
            &input,
        ))
    } else {
        None
    };
    let wicks_view = wicks.as_ref().map(|w| w.view());

    let (e_ref, c0, _dt_hs_ref) = calculate_noci_energy(
        &ao,
        &input,
        &noci_reference_basis,
        1e-12,
        &mocache,
        wicks_view,
    );

    let include_refs = true;
    let basis = generate_excited_basis(&noci_reference_basis, &input, include_refs);

    let n = basis.len();
    let mut c0qmc = vec![0.0; n];
    let mut ref_indices = Vec::with_capacity(noci_reference_basis.len());

    for (i, ref_st) in noci_reference_basis.iter().enumerate() {
        let idx = basis
            .iter()
            .position(|qmc_st| qmc_st.label == ref_st.label)
            .unwrap();

        c0qmc[idx] = c0[i];
        ref_indices.push(idx);
    }

    let data = NOCIData::new(
        &ao,
        &basis,
        &input,
        1e-12,
        wicks_view,
    )
    .withmocache(&mocache);

    let mut es = states[0].e;
    let (e_qmc, _excitation_hist) =
        qmc_step(&data, &c0qmc, &mut es, &ref_indices, &world);

    (scf_energies, e_ref, e_qmc)
}

/// Test that the H2 cc-pVDZ 1.5 Angstrom fixture reproduces the expected SCF state energies,
/// reference NOCI energy and stochastic NOCI-QMC energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF, reference NOCI or stochastic NOCI-QMC energy differs from the known good value
///   outside tolerance.
#[test]
#[serial]
#[ignore = "non-deterministic"]
fn qmc_h2_cc_pvdz_1_5_ang_wwicks_energies() {

    ThreadPoolBuilder::new()
            .stack_size(128 * 1024 * 1024)
            .num_threads(1)
            .build_global()
            .unwrap();

    let (_input, _ao, expected): (_, _, ExpectedQMC) = load_test("QMC_H2_cc-pVDZ_1_5_WICKS");
    let (got_scf, got_ref, got_qmc) = run_qmc_fixture("QMC_H2_cc-pVDZ_1_5_WICKS");

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
        got_qmc,
        expected.qmc_noci_energy,
        1e-8,
        "H2 stochastic NOCI-QMC energy",
    );
}
