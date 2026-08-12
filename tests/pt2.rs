mod common;

// External crate imports.
use noci_rs::PostSCFData;
use noci_rs::basis::generate_reference_noci_basis;
use noci_rs::input::SNOCIBackend;
use noci_rs::noci::{build_mo_cache, build_wicks_shared, calculate_noci_energy};
use noci_rs::snoci::snoci_step;
use serde::Deserialize;
use serial_test::serial;

// Parent/sibling imports.
use self::common::{assert_close, load_test, mpi_universe};

/// Expected exact energies for a NOCI-PT2 fixture.
#[derive(Deserialize)]
struct ExpectedPT2 {
    /// Expected SCF state energies.
    scf_energies: Vec<f64>,
    /// Expected reference-space NOCI energy.
    reference_noci_energy: f64,
    /// Expected total NOCI-PT2 energy.
    noci_pt2_energy: f64,
}

/// Energies and GMRES diagnostics from one backend-specific NOCI-PT2 run.
struct PT2GpuComparisonRun {
    /// Sorted SCF state energies.
    scf_energies: Vec<f64>,
    /// Reference-space NOCI energy.
    reference_noci_energy: f64,
    /// Total NOCI-PT2 energy.
    noci_pt2_energy: f64,
    /// NOCI-PT2 correction energy.
    ept2: f64,
    /// Final GMRES residual.
    gmres_residual: f64,
    /// Final GMRES iteration count.
    gmres_iterations: usize,
}

/// Run SCF, reference NOCI and NOCI-PT2 and compare energies with known good energies.
/// # Arguments:
/// - `fixture`: Name of the test fixture to load.
/// # Returns
/// - `(Vec<f64>, f64, f64)`: Sorted SCF state energies, the reference NOCI energy and the
///   total NOCI-PT2 energy.
fn run_pt2_fixture(fixture: &str) -> (Vec<f64>, f64, f64) {
    let (mut input, ao, _expected): (_, _, ExpectedPT2) = load_test(fixture);

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

    let (_mpi_lock, universe) = mpi_universe();
    let world = universe.world();

    let result = snoci_step(&post, &noci_reference_basis, &input, None, &world);

    let pt2 = result
        .pt2
        .last()
        .expect("NOCI-PT2 did not produce a PT2 result");

    assert!(
        pt2.gmres_converged,
        "NOCI-PT2 GMRES failed to converge: residual {}",
        pt2.gmres_residual,
    );

    let e_pt2 = result.ecurrent + pt2.ept2;

    (scf_energies, e_ref, e_pt2)
}

/// Run SCF, reference NOCI and NOCI-PT2 with Wick's intermediates and compare energies with known good energies.
/// # Arguments:
/// - `fixture`: Name of the test fixture to load.
/// # Returns
/// - `(Vec<f64>, f64, f64)`: Sorted SCF state energies, the reference NOCI energy and the
///   total NOCI-PT2 energy.
fn run_pt2_fixture_wicks(fixture: &str) -> (Vec<f64>, f64, f64) {
    let (mut input, ao, _expected): (_, _, ExpectedPT2) = load_test(fixture);

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

    let (_mpi_lock, universe) = mpi_universe();
    let world = universe.world();

    let mut wicks = build_wicks_shared::<f64>(&world, &ao, &noci_reference_basis, 1e-12, &input);
    let (e_ref, _c0, _dt_hs_ref) = {
        let wicks_view = wicks.view();
        calculate_noci_energy(
            &ao,
            &input,
            &noci_reference_basis,
            1e-12,
            &mocache,
            Some(wicks_view),
        )
    };

    let post = PostSCFData {
        ao: &ao,
        states: &states,
        noci_reference_basis: &noci_reference_basis,
        mocache: &mocache,
        tol: 1e-12,
    };

    let result = snoci_step(
        &post,
        &noci_reference_basis,
        &input,
        Some(&mut wicks),
        &world,
    );

    let pt2 = result
        .pt2
        .last()
        .expect("NOCI-PT2 did not produce a PT2 result");

    assert!(
        pt2.gmres_converged,
        "NOCI-PT2 GMRES failed to converge: residual {}",
        pt2.gmres_residual,
    );

    let e_pt2 = result.ecurrent + pt2.ept2;

    (scf_energies, e_ref, e_pt2)
}

/// Run a Wicks NOCI-PT2 fixture with an explicit one-body backend.
/// The input fixture supplies all numerical settings, while this helper overwrites only
/// `snoci.backend` so CPU and GPU runs compare the same matrix-free factorised path.
/// # Arguments:
/// - `fixture`: Name of the Wicks fixture to load.
/// - `backend`: One-body backend to use for the NOCI-PT2 solve.
/// # Returns
/// - `PT2BackendRun`: Energies and GMRES diagnostics from the run.
fn run_pt2_fixture_wicks_with(
    fixture: &str,
    backend: SNOCIBackend,
) -> PT2GpuComparisonRun {
    let (mut input, ao, _expected): (_, _, ExpectedPT2) = load_test(fixture);
    input
        .snoci
        .as_mut()
        .expect("NOCI-PT2 fixture must define snoci")
        .backend = backend;

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

    let (_mpi_lock, universe) = mpi_universe();
    let world = universe.world();

    let mut wicks = build_wicks_shared::<f64>(&world, &ao, &noci_reference_basis, 1e-12, &input);
    let (e_ref, _c0, _dt_hs_ref) = {
        let wicks_view = wicks.view();
        calculate_noci_energy(
            &ao,
            &input,
            &noci_reference_basis,
            1e-12,
            &mocache,
            Some(wicks_view),
        )
    };

    let post = PostSCFData {
        ao: &ao,
        states: &states,
        noci_reference_basis: &noci_reference_basis,
        mocache: &mocache,
        tol: 1e-12,
    };

    let result = snoci_step(
        &post,
        &noci_reference_basis,
        &input,
        Some(&mut wicks),
        &world,
    );
    let pt2 = result
        .pt2
        .last()
        .expect("NOCI-PT2 did not produce a PT2 result");
    assert!(
        pt2.gmres_converged,
        "NOCI-PT2 GMRES failed to converge: residual {}",
        pt2.gmres_residual,
    );

    PT2GpuComparisonRun {
        scf_energies,
        reference_noci_energy: e_ref,
        noci_pt2_energy: result.ecurrent + pt2.ept2,
        ept2: pt2.ept2,
        gmres_residual: pt2.gmres_residual,
        gmres_iterations: pt2.gmres_iterations,
    }
}

/// Test that the H2 STO-3G 1.5 Angstrom fixture reproduces the expected SCF state energies,
/// reference NOCI energy and total NOCI-PT2 energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF, reference NOCI or total NOCI-PT2 energy differs from the known good value
///   outside tolerance.
/// - If the NOCI-PT2 GMRES solve does not converge.
#[test]
#[serial]
fn pt2_h2_sto_3g_1_5_ang_energies() {
    let (_input, _ao, expected): (_, _, ExpectedPT2) = load_test("PT2_H2_STO-3G_1_5");
    let (got_scf, got_ref, got_pt2) = run_pt2_fixture("PT2_H2_STO-3G_1_5");

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
        got_pt2,
        expected.noci_pt2_energy,
        1e-8,
        "H2 NOCI-PT2 energy",
    );
}

/// Test that the H2 STO-3G 1.5 Angstrom fixture reproduces the expected SCF state energies,
/// reference NOCI energy and total NOCI-PT2 energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF, reference NOCI or total NOCI-PT2 energy differs from the known good value
///   outside tolerance.
/// - If the NOCI-PT2 GMRES solve does not converge.
#[test]
#[serial]
fn pt2_h2_sto_3g_1_5_ang_energies_wicks() {
    let (_input, _ao, expected): (_, _, ExpectedPT2) = load_test("PT2_H2_STO-3G_1_5_WICKS");
    let (got_scf, got_ref, got_pt2) = run_pt2_fixture_wicks("PT2_H2_STO-3G_1_5_WICKS");

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
        got_pt2,
        expected.noci_pt2_energy,
        1e-8,
        "H2 NOCI-PT2 energy",
    );
}

/// Test that the H2 STO-3G 1.5 Angstrom fixture agrees with and without Wick's intermediates.
/// # Panics
/// - If the number of SCF states differs between implementations.
/// - If SCF, reference NOCI or total NOCI-PT2 energy differs between implementations
///   outside tolerance.
/// - If the NOCI-PT2 GMRES solve does not converge.
#[test]
#[serial]
fn pt2_h2_sto_3g_1_5_ang_energies_agree() {
    let (got_scf, got_ref, got_pt2) = run_pt2_fixture("PT2_H2_STO-3G_1_5");
    let (got_scf_wicks, got_ref_wicks, got_pt2_wicks) =
        run_pt2_fixture_wicks("PT2_H2_STO-3G_1_5_WICKS");

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
    assert_close(got_pt2, got_pt2_wicks, 1e-8, "H2 NOCI-PT2 Wicks agreement");
}

/// Test that the H2 3-21G 1.5 Angstrom fixture reproduces the expected SCF state energies,
/// reference NOCI energy and total NOCI-PT2 energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF, reference NOCI or total NOCI-PT2 energy differs from the known good value
///   outside tolerance.
/// - If the NOCI-PT2 GMRES solve does not converge.
#[test]
#[serial]
fn pt2_h2_3_21g_1_5_ang_energies() {
    let (_input, _ao, expected): (_, _, ExpectedPT2) = load_test("PT2_H2_3-21G_1_5");
    let (got_scf, got_ref, got_pt2) = run_pt2_fixture("PT2_H2_3-21G_1_5");

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
        got_pt2,
        expected.noci_pt2_energy,
        1e-8,
        "H2 NOCI-PT2 energy",
    );
}

/// Test that the H2 3-21G 1.5 Angstrom fixture reproduces the expected SCF state energies,
/// reference NOCI energy and total NOCI-PT2 energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF, reference NOCI or total NOCI-PT2 energy differs from the known good value
///   outside tolerance.
/// - If the NOCI-PT2 GMRES solve does not converge.
#[test]
#[serial]
fn pt2_h2_3_21g_1_5_ang_energies_wicks() {
    let (_input, _ao, expected): (_, _, ExpectedPT2) = load_test("PT2_H2_3-21G_1_5_WICKS");
    let (got_scf, got_ref, got_pt2) = run_pt2_fixture_wicks("PT2_H2_3-21G_1_5_WICKS");

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
        got_pt2,
        expected.noci_pt2_energy,
        1e-8,
        "H2 NOCI-PT2 energy",
    );
}

/// Test that the H2 3-21G 1.5 Angstrom fixture agrees with and without Wick's intermediates.
/// # Panics
/// - If the number of SCF states differs between implementations.
/// - If SCF, reference NOCI or total NOCI-PT2 energy differs between implementations
///   outside tolerance.
/// - If the NOCI-PT2 GMRES solve does not converge.
#[test]
#[serial]
fn pt2_h2_3_21g_1_5_ang_energies_agree() {
    let (got_scf, got_ref, got_pt2) = run_pt2_fixture("PT2_H2_3-21G_1_5");
    let (got_scf_wicks, got_ref_wicks, got_pt2_wicks) =
        run_pt2_fixture_wicks("PT2_H2_3-21G_1_5_WICKS");

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
    assert_close(got_pt2, got_pt2_wicks, 1e-8, "H2 NOCI-PT2 Wicks agreement");
}

/// Test that the H2 3-21G GPU NOCI-PT2 backend agrees with the CPU factorised backend.
/// # Panics
/// - If CPU or GPU GMRES does not converge.
/// - If SCF, reference NOCI, PT2 correction, total NOCI-PT2 energy or GMRES diagnostics differ
///   outside tolerance.
#[cfg(feature = "gpu")]
#[test]
#[ignore = "gpu"]
#[serial]
fn pt2_h2_3_21g_1_5_ang_gpu_matches_cpu() {
    let cpu = run_pt2_fixture_wicks_with("PT2_H2_3-21G_1_5_GPU", SNOCIBackend::CPU);
    let gpu = run_pt2_fixture_wicks_with("PT2_H2_3-21G_1_5_GPU", SNOCIBackend::GPU);

    assert_eq!(cpu.scf_energies.len(), gpu.scf_energies.len());
    for (i, (&x, &y)) in cpu
        .scf_energies
        .iter()
        .zip(gpu.scf_energies.iter())
        .enumerate()
    {
        assert_close(x, y, 1e-8, &format!("H2 SCF state {i} GPU agreement"));
    }

    assert_close(
        cpu.reference_noci_energy,
        gpu.reference_noci_energy,
        1e-8,
        "H2 reference NOCI GPU agreement",
    );
    assert_close(cpu.ept2, gpu.ept2, 1e-8, "H2 EPT2 GPU agreement");
    assert_close(
        cpu.noci_pt2_energy,
        gpu.noci_pt2_energy,
        1e-8,
        "H2 NOCI-PT2 GPU agreement",
    );
    assert_close(
        cpu.gmres_residual,
        gpu.gmres_residual,
        1e-8,
        "H2 GMRES residual GPU agreement",
    );
    assert_eq!(cpu.gmres_iterations, gpu.gmres_iterations);
}
