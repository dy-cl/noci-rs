mod common;

use std::process::Command;
use std::sync::OnceLock;

use common::{assert_close, fixture_dir, load_test, mpi_universe};
use rayon::ThreadPoolBuilder;
use serde::Deserialize;
use serial_test::serial;

use noci_rs::basis::{generate_excited_basis, generate_reference_noci_basis};
use noci_rs::noci::{NOCIData, build_mo_cache, build_wicks_shared, calculate_noci_energy};
use noci_rs::stochastic::qmc_step;

/// Expected exact energies for a QMC fixture.
#[derive(Deserialize)]
struct ExpectedQMC {
    /// Expected SCF state energies.
    scf_energies: Vec<f64>,
    /// Expected reference-space NOCI energy.
    reference_noci_energy: f64,
    /// Expected stochastic NOCI-QMC energy.
    qmc_noci_energy: f64,
}

/// Expected trajectory checks for a QMC fixture.
#[derive(Deserialize)]
struct ExpectedQMCTrajectory {
    /// Minimum allowed reported QMC energy.
    min_energy: f64,
    /// Maximum allowed reported QMC energy.
    max_energy: f64,
    /// Minimum number of QMC report energies.
    min_samples: usize,
    /// Minimum difference between the largest and smallest reported QMC energies.
    min_energy_span: f64,
    /// Minimum decrease from the first to final reported QMC energy.
    min_final_drop: f64,
}

/// Initialise the global Rayon pool used by QMC tests.
/// # Arguments:
/// - None.
/// # Returns
/// - `()`: The global Rayon pool is initialised once for this process.
/// # Panics
/// - If the global Rayon thread pool cannot be built.
fn init_qmc_thread_pool() {
    static INIT: OnceLock<()> = OnceLock::new();

    INIT.get_or_init(|| {
        ThreadPoolBuilder::new()
            .stack_size(128 * 1024 * 1024)
            .num_threads(1)
            .build_global()
            .unwrap();
    });
}

/// Run SCF, reference NOCI and stochastic NOCI-QMC and compare energies with known good energies.
/// # Arguments:
/// - `fixture`: Name of the test fixture to load.
/// # Returns
/// - `(Vec<f64>, f64, f64)`: Sorted SCF state energies, the reference NOCI energy and the
///   stochastic NOCI-QMC energy.
/// # Panics
/// - If the fixture cannot be loaded.
/// - If SCF state energies cannot be sorted.
/// - If a reference determinant is missing from the QMC basis.
/// - If Wick's intermediates or NOCI-QMC propagation fail internally.
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

    let (_mpi_lock, universe) = mpi_universe();
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

    let data = NOCIData::new(&ao, &basis, &input, 1e-12, wicks_view).withmocache(&mocache);

    let mut es = states[0].e;
    let (e_qmc, _excitation_hist) = qmc_step(&data, &c0qmc, &mut es, &ref_indices, &world);

    (scf_energies, e_ref, e_qmc)
}

/// Run a QMC fixture through the binary and collect report energies from stdout.
/// # Arguments:
/// - `fixture`: Name of the test fixture to run.
/// # Returns
/// - `Vec<f64>`: QMC report energies printed by the binary.
/// # Panics
/// - If the binary cannot be found or run.
/// - If stdout or stderr are not valid UTF-8.
/// - If the binary exits with a non-zero status.
fn qmc_report_energies(fixture: &str) -> Vec<f64> {
    let exe = env!("CARGO_BIN_EXE_noci-rs");
    let input_path = fixture_dir(fixture).join("input.lua");

    let output = Command::new(exe)
        .env("RAYON_NUM_THREADS", "1")
        .arg(input_path)
        .output()
        .unwrap();

    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();

    assert!(
        output.status.success(),
        "noci-rs failed with status {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        stdout,
        stderr
    );

    stdout
        .lines()
        .filter_map(|line| {
            let cols: Vec<_> = line.split_whitespace().collect();
            if cols.len() == 10 && cols[0].parse::<usize>().is_ok() {
                cols[1].parse::<f64>().ok()
            } else {
                None
            }
        })
        .collect()
}

/// Check that a short QMC trajectory is finite, bounded and moving downward.
/// # Arguments:
/// - `fixture`: Name of the trajectory fixture to run.
/// # Returns
/// - `()`: The fixture satisfies its stored trajectory bounds.
/// # Panics
/// - If the fixture `expected.json` cannot be read or parsed.
/// - If too few report energies are printed.
/// - If any report energy is non-finite or outside the fixture bounds.
/// - If the report energies do not span or drop by the fixture minima.
fn assert_qmc_trajectory_bounds(fixture: &str) {
    let expected: ExpectedQMCTrajectory = serde_json::from_str(
        &std::fs::read_to_string(fixture_dir(fixture).join("expected.json")).unwrap(),
    )
    .unwrap();
    let energies = qmc_report_energies(fixture);

    assert!(
        energies.len() >= expected.min_samples,
        "{fixture} produced {} report energies, expected at least {}",
        energies.len(),
        expected.min_samples
    );
    for (i, &energy) in energies.iter().enumerate() {
        assert!(
            energy.is_finite(),
            "{fixture} report energy {i} is non-finite: {energy}"
        );
        assert!(
            (expected.min_energy..=expected.max_energy).contains(&energy),
            "{fixture} report energy {i} out of bounds: {energy}, expected [{}, {}]",
            expected.min_energy,
            expected.max_energy
        );
    }

    let min_energy = energies.iter().copied().reduce(f64::min).unwrap();
    let max_energy = energies.iter().copied().reduce(f64::max).unwrap();
    let energy_span = max_energy - min_energy;
    assert!(
        energy_span >= expected.min_energy_span,
        "{fixture} report energies changed by {energy_span}, expected at least {}",
        expected.min_energy_span
    );

    let final_drop = energies.first().unwrap() - energies.last().unwrap();
    assert!(
        final_drop >= expected.min_final_drop,
        "{fixture} final report energy dropped by {final_drop}, expected at least {}",
        expected.min_final_drop
    );
}

/// Test that the H2 STO-3G 1.5 Angstrom fixture reproduces the expected SCF state energies,
/// reference NOCI energy and stochastic NOCI-QMC energy.
/// # Arguments:
/// - None.
/// # Returns
/// - `()`: The fixture energies match the stored reference values.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF, reference NOCI or stochastic NOCI-QMC energy differs from the known good value
///   outside tolerance.
#[test]
#[serial]
#[ignore = "non-deterministic"]
fn qmc_h2_sto_3g_1_5_ang_energies() {
    init_qmc_thread_pool();

    let (_input, _ao, expected): (_, _, ExpectedQMC) = load_test("QMC_H2_STO-3G_1_5");
    let (got_scf, got_ref, got_qmc) = run_qmc_fixture("QMC_H2_STO-3G_1_5");

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

/// Test that the H2 STO-3G 1.5 Angstrom fixture reproduces the expected SCF state energies,
/// reference NOCI energy and stochastic NOCI-QMC energy.
/// # Arguments:
/// - None.
/// # Returns
/// - `()`: The Wick's fixture energies match the stored reference values.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF, reference NOCI or stochastic NOCI-QMC energy differs from the known good value
///   outside tolerance.
#[test]
#[serial]
#[ignore = "non-deterministic"]
fn qmc_h2_sto_3g_1_5_ang_energies_wicks() {
    init_qmc_thread_pool();

    let (_input, _ao, expected): (_, _, ExpectedQMC) = load_test("QMC_H2_STO-3G_1_5_WICKS");
    let (got_scf, got_ref, got_qmc) = run_qmc_fixture("QMC_H2_STO-3G_1_5_WICKS");

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

/// Test that the H2 STO-3G 1.5 Angstrom fixture agrees with and without Wick's intermediates.
/// # Arguments:
/// - None.
/// # Returns
/// - `()`: The Wick's and non-Wick's fixture energies agree.
/// # Panics
/// - If the number of SCF states differs between implementations.
/// - If SCF, reference NOCI or stochastic NOCI-QMC energy differs between implementations
///   outside tolerance.
#[test]
#[serial]
#[ignore = "non-deterministic"]
fn qmc_h2_sto_3g_1_5_ang_energies_agree() {
    init_qmc_thread_pool();

    let (got_scf, got_ref, got_qmc) = run_qmc_fixture("QMC_H2_STO-3G_1_5");
    let (got_scf_wicks, got_ref_wicks, got_qmc_wicks) = run_qmc_fixture("QMC_H2_STO-3G_1_5_WICKS");

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
        got_qmc,
        got_qmc_wicks,
        1e-8,
        "H2 stochastic NOCI-QMC Wicks agreement",
    );
}

/// Test that the H2 3-21G 1.5 Angstrom fixture reproduces the expected SCF state energies,
/// reference NOCI energy and stochastic NOCI-QMC energy.
/// # Arguments:
/// - None.
/// # Returns
/// - `()`: The fixture energies match the stored reference values.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF, reference NOCI or stochastic NOCI-QMC energy differs from the known good value
///   outside tolerance.
#[test]
#[serial]
#[ignore = "non-deterministic"]
fn qmc_h2_3_21g_1_5_ang_energies() {
    init_qmc_thread_pool();

    let (_input, _ao, expected): (_, _, ExpectedQMC) = load_test("QMC_H2_3-21G_1_5");
    let (got_scf, got_ref, got_qmc) = run_qmc_fixture("QMC_H2_3-21G_1_5");

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

/// Test that the H2 3-21G 1.5 Angstrom fixture reproduces the expected SCF state energies,
/// reference NOCI energy and stochastic NOCI-QMC energy.
/// # Arguments:
/// - None.
/// # Returns
/// - `()`: The Wick's fixture energies match the stored reference values.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF, reference NOCI or stochastic NOCI-QMC energy differs from the known good value
///   outside tolerance.
#[test]
#[serial]
#[ignore = "non-deterministic"]
fn qmc_h2_3_21g_1_5_ang_energies_wicks() {
    init_qmc_thread_pool();

    let (_input, _ao, expected): (_, _, ExpectedQMC) = load_test("QMC_H2_3-21G_1_5_WICKS");
    let (got_scf, got_ref, got_qmc) = run_qmc_fixture("QMC_H2_3-21G_1_5_WICKS");

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

/// Test that the H2 3-21G 1.5 Angstrom fixture agrees with and without Wick's intermediates.
/// # Arguments:
/// - None.
/// # Returns
/// - `()`: The Wick's and non-Wick's fixture energies agree.
/// # Panics
/// - If the number of SCF states differs between implementations.
/// - If SCF, reference NOCI or stochastic NOCI-QMC energy differs between implementations
///   outside tolerance.
#[test]
#[serial]
#[ignore = "non-deterministic"]
fn qmc_h2_3_21g_1_5_ang_energies_agree() {
    init_qmc_thread_pool();

    let (got_scf, got_ref, got_qmc) = run_qmc_fixture("QMC_H2_3-21G_1_5");
    let (got_scf_wicks, got_ref_wicks, got_qmc_wicks) = run_qmc_fixture("QMC_H2_3-21G_1_5_WICKS");

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
        got_qmc,
        got_qmc_wicks,
        1e-8,
        "H2 stochastic NOCI-QMC Wicks agreement",
    );
}

/// Test that a short LiH STO-3G QMC trajectory stays in a physical energy window.
/// # Arguments:
/// - None.
/// # Returns
/// - `()`: The trajectory satisfies the stored bounds.
/// # Panics
/// - If the binary run fails.
/// - If too few QMC report energies are printed.
/// - If any report energy is non-finite or outside the fixture bounds.
/// - If the trajectory does not move downward by the fixture minimum.
#[test]
#[serial]
fn qmc_lih_sto_3g_2_8_ang_trajectory_bounds() {
    assert_qmc_trajectory_bounds("QMC_LiH_STO-3G_2_8_TRAJECTORY");
}

/// Test that a short LiH 6-31G QMC trajectory stays in a physical energy window.
/// # Arguments:
/// - None.
/// # Returns
/// - `()`: The trajectory satisfies the stored bounds.
/// # Panics
/// - If the binary run fails.
/// - If too few QMC report energies are printed.
/// - If any report energy is non-finite or outside the fixture bounds.
/// - If the trajectory does not move downward by the fixture minimum.
#[test]
#[serial]
fn qmc_lih_6_31g_2_8_ang_trajectory_bounds() {
    assert_qmc_trajectory_bounds("QMC_LiH_6-31G_2_8_TRAJECTORY");
}
