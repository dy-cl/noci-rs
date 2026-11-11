mod common;

// Standard library imports.
use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

// External crate imports.
use noci_rs::basis::generate_reference_noci_basis;
use noci_rs::determinant::{NOCIIndex, NOCISpace};
use noci_rs::elements::{NOCIData, build_mo_cache, build_wicks_shared};
use noci_rs::noci::calculate_noci_energy;
use noci_rs::noci::stochastic::qmc_step;
use rayon::ThreadPoolBuilder;
use serde::Deserialize;
use serial_test::serial;

// Parent/sibling imports.
use self::common::{assert_close, fixture_dir, load_test, mpi_universe};

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
    /// Expected reported QMC energies.
    report_energies: Vec<f64>,
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

    let noci_reference_basis: Vec<_> = states.iter().filter(|s| s.noci_basis).cloned().collect();
    let reference_space = NOCISpace::from_scf(&noci_reference_basis);

    let (_mpi_lock, universe) = mpi_universe();
    let world = universe.world();

    let mocache = build_mo_cache(&ao, reference_space.parents(), input.scf.d_tol);

    let wicks = if input.wicks.enabled {
        Some(build_wicks_shared::<f64>(
            &world,
            &ao,
            reference_space.parents(),
            1e-12,
            &input,
        ))
    } else {
        None
    };
    let wicks_view = wicks.as_ref().map(|w| w.view());

    let (e_ref, c0, _dt_hs_ref) =
        calculate_noci_energy(&ao, &input, &reference_space, 1e-12, &mocache, wicks_view);

    let sources = (0..reference_space.len())
        .map(NOCIIndex)
        .collect::<Vec<_>>();
    let basis = reference_space.excited_from(&sources, &input, true);

    let n = basis.len();
    let mut c0qmc = vec![0.0; n];
    let mut ref_indices = Vec::with_capacity(noci_reference_basis.len());

    for (i, ref_st) in noci_reference_basis.iter().enumerate() {
        let idx = (0..basis.len())
            .find(|&index| basis.label(NOCIIndex(index)) == ref_st.label)
            .unwrap();

        c0qmc[idx] = c0[i];
        ref_indices.push(idx);
    }

    let data = NOCIData::new(&ao, &basis, &input, 1e-12, wicks_view).withmocache(&mocache);

    let mut es = states[0].e;
    let (e_qmc, _excitation_hist) = qmc_step(&data, &c0qmc, &mut es, &ref_indices, &world);

    (scf_energies, e_ref, e_qmc)
}

/// Run a QMC input through the binary and collect report energies from stdout.
/// # Arguments:
/// - `input`: QMC input file to run.
/// - `workdir`: Working directory of the run, which receives any restart it writes.
/// # Returns:
/// - `Vec<f64>`: QMC report energies printed by the binary.
/// # Panics:
/// - If the binary cannot be found or run.
/// - If stdout or stderr are not valid UTF-8.
/// - If the binary exits with a nonzero status.
fn qmc_report_energies(
    input: &Path,
    workdir: &Path,
) -> Vec<f64> {
    let exe = env!("CARGO_BIN_EXE_noci-rs");

    let mut command = qmc_binary_command(exe);
    let output = command
        .current_dir(workdir)
        .env("RAYON_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("OPENBLAS_CORETYPE", "CORE2")
        .env("OMP_NUM_THREADS", "1")
        .env("MKL_NUM_THREADS", "1")
        .arg(input)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}: QMC binary exited with status {:?}\nstdout:\n{}\nstderr:\n{}",
        input.display(),
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).unwrap();

    let mut eproj = None;
    let mut energies = Vec::new();
    for line in stdout.lines() {
        let columns = line.split_whitespace().collect::<Vec<_>>();
        if columns.first() == Some(&"Iter") {
            eproj = columns.iter().position(|&column| column == "EProj");
            continue;
        }

        let Some(col) = eproj else {
            continue;
        };
        if columns
            .first()
            .and_then(|column| column.parse::<usize>().ok())
            .is_none()
        {
            continue;
        }
        if let Some(energy) = columns
            .get(col)
            .and_then(|column| column.parse::<f64>().ok())
        {
            energies.push(energy);
        }
    }

    energies
}

/// Construct a command for the test binary, honouring Cargo's target runner when CI executes the
/// test harness under Intel SDE.
/// # Arguments:
/// - `exe`: Path to the compiled `noci-rs` binary.
/// # Returns:
/// - `Command`: Command that executes `exe` directly or through `CARGO_TARGET_*_RUNNER`.
fn qmc_binary_command(exe: &str) -> Command {
    let runner = env::vars()
        .filter(|(key, _)| key.starts_with("CARGO_TARGET_") && key.ends_with("_RUNNER"))
        .map(|(_, value)| value)
        .next();

    let Some(runner) = runner else {
        return Command::new(exe);
    };

    let mut parts = runner.split_whitespace();
    let Some(program) = parts.next() else {
        return Command::new(exe);
    };

    let mut command = Command::new(program);
    command.args(parts);
    command.arg(exe);
    command
}

/// Check that a short QMC trajectory matches the stored reported energies.
/// # Arguments:
/// - `fixture`: Name of the trajectory fixture to run.
/// - `workdir`: Working directory of the run, which receives any restart it writes.
/// # Returns
/// - `Vec<f64>`: Report energies of the run, which satisfy the stored trajectory.
/// # Panics
/// - If the fixture `expected.json` cannot be read or parsed.
/// - If the report trajectory length differs from the stored trajectory.
/// - If any report energy differs from the stored trajectory outside tolerance.
fn assert_qmc_trajectory(
    fixture: &str,
    workdir: &Path,
) -> Vec<f64> {
    let expected: ExpectedQMCTrajectory = serde_json::from_str(
        &fs::read_to_string(fixture_dir(fixture).join("expected.json")).unwrap(),
    )
    .unwrap();
    let input = fixture_dir(fixture)
        .canonicalize()
        .unwrap()
        .join("input.lua");
    let got = qmc_report_energies(&input, workdir);

    assert_eq!(
        got.len(),
        expected.report_energies.len(),
        "{fixture}: QMC trajectory length"
    );

    for (i, (&got, &expected)) in got.iter().zip(expected.report_energies.iter()).enumerate() {
        assert_close(got, expected, 1e-9, &format!("{fixture} QMC report {i}"));
    }
    got
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

/// Test that a short LiH STO-3G SApply QMC trajectory reproduces the stored deterministic QMC trajectory.
/// # Arguments:
/// - None.
/// # Returns
/// - `()`: The trajectory matches the stored report energies.
/// # Panics
/// - If the binary run fails.
/// - If the report trajectory length differs from the stored trajectory.
/// - If any report energy differs from the stored trajectory outside tolerance.
#[test]
#[serial]
fn qmc_lih_sto_3g_2_8_ang_trajectory_s_apply() {
    assert_qmc_trajectory("QMC_LiH_STO-3G_2_8_TRAJECTORY_S_APPLY", Path::new("."));
}

/// Test that a short LiH 6-31G SApply QMC trajectory reproduces the stored deterministic QMC trajectory.
/// # Arguments:
/// - None.
/// # Returns
/// - `()`: The trajectory matches the stored report energies.
/// # Panics
/// - If the binary run fails.
/// - If the report trajectory length differs from the stored trajectory.
/// - If any report energy differs from the stored trajectory outside tolerance.
#[test]
#[serial]
fn qmc_lih_6_31g_2_8_ang_trajectory_s_apply() {
    assert_qmc_trajectory("QMC_LiH_6-31G_2_8_TRAJECTORY_S_APPLY", Path::new("."));
}

/// Test that an H6 STO-3G SDT Wick SApply QMC trajectory reproduces the stored deterministic QMC trajectory.
/// # Arguments:
/// - None.
/// # Returns
/// - `()`: The trajectory matches the stored report energies.
/// # Panics
/// - If the binary run fails.
/// - If the report trajectory length differs from the stored trajectory.
/// - If any report energy differs from the stored trajectory outside tolerance.
#[test]
#[serial]
fn qmc_h6_sto_3g_1_5_ang_sdt_wicks_trajectory() {
    assert_qmc_trajectory(
        "QMC_H6_STO-3G_1_5_TRAJECTORY_S_APPLY_WICKS_SDT",
        Path::new("."),
    );
}

/// Test that a short H6 STO-3G overlap-weighted Wick QMC trajectory reproduces the stored
/// deterministic QMC trajectory.
/// # Arguments:
/// - None.
/// # Returns
/// - `()`: The trajectory matches the stored report energies.
/// # Panics
/// - If the binary run fails.
/// - If the report trajectory length differs from the stored trajectory.
/// - If any report energy differs from the stored trajectory outside tolerance.
#[test]
#[serial]
fn qmc_h6_sto_3g_1_5_ang_overlap_weighted_wicks_trajectory() {
    assert_qmc_trajectory(
        "QMC_H6_STO-3G_1_5_TRAJECTORY_OVERLAP_WEIGHTED_WICKS",
        Path::new("."),
    );
}

/// Test that a short H4 STO-3G stochastic JM-NOCCMC trajectory with two UHF references, through
/// population growth and into population control, reproduces the stored report energies.
/// # Arguments:
/// - None.
/// # Returns
/// - `()`: The trajectory matches the stored report energies.
/// # Panics
/// - If the binary run fails.
/// - If the report trajectory length differs from the stored trajectory.
/// - If any report energy differs from the stored trajectory outside tolerance.
#[test]
#[serial]
fn jmnoccmc_h4_sto_3g_1_5_ang_trajectory() {
    assert_qmc_trajectory("QMC_H4_STO-3G_1_5_TRAJECTORY_JMNOCCMC", Path::new("."));
}

/// Test that the H6 STO-3G SDT Wick BApply QMC trajectory with the uniform generator reproduces
/// the stored deterministic QMC trajectory.
/// # Arguments:
/// - None.
/// # Returns
/// - `()`: The trajectory matches the stored report energies.
/// # Panics
/// - If the binary run fails.
/// - If the report trajectory length differs from the stored trajectory.
/// - If any report energy differs from the stored trajectory outside tolerance.
#[test]
#[serial]
fn qmc_h6_sto_3g_1_5_ang_b_apply_wicks_trajectory() {
    assert_qmc_trajectory(
        "QMC_H6_STO-3G_1_5_TRAJECTORY_B_APPLY_WICKS_SDT",
        Path::new("."),
    );
}

/// Test that the H6 STO-3G SDT Wick BApply QMC trajectory with the PCHB generator reproduces the
/// stored deterministic QMC trajectory.
/// # Arguments:
/// - None.
/// # Returns
/// - `()`: The trajectory matches the stored report energies.
/// # Panics
/// - If the binary run fails.
/// - If the report trajectory length differs from the stored trajectory.
/// - If any report energy differs from the stored trajectory outside tolerance.
#[test]
#[serial]
fn qmc_h6_sto_3g_1_5_ang_b_apply_pchb_wicks_trajectory() {
    assert_qmc_trajectory(
        "QMC_H6_STO-3G_1_5_TRAJECTORY_B_APPLY_PCHB_WICKS_SDT",
        Path::new("."),
    );
}

/// Test that the H6 STO-3G SDT Wick unshifted QMC trajectory reproduces the stored deterministic
/// QMC trajectory.
/// # Arguments:
/// - None.
/// # Returns
/// - `()`: The trajectory matches the stored report energies.
/// # Panics
/// - If the binary run fails.
/// - If the report trajectory length differs from the stored trajectory.
/// - If any report energy differs from the stored trajectory outside tolerance.
#[test]
#[serial]
fn qmc_h6_sto_3g_1_5_ang_unshifted_wicks_trajectory() {
    assert_qmc_trajectory(
        "QMC_H6_STO-3G_1_5_TRAJECTORY_UNSHIFTED_WICKS_SDT",
        Path::new("."),
    );
}

/// Test that an H6 STO-3G SApply QMC trajectory reproduces the stored deterministic QMC trajectory,
/// and that a run resumed from the restart it writes after three iterations reproduces the last
/// reports of the uninterrupted run.
/// # Arguments:
/// - None.
/// # Returns
/// - `()`: Both runs match the stored and the uninterrupted trajectory.
/// # Panics
/// - If either binary run fails.
/// - If the uninterrupted trajectory differs from the stored trajectory.
/// - If the resumed trajectory differs from the last reports of the uninterrupted run.
#[test]
#[serial]
fn qmc_h6_sto_3g_1_5_ang_s_apply_restart_trajectory() {
    let fixture = "QMC_H6_STO-3G_1_5_TRAJECTORY_S_APPLY_RESTART";

    // Run the uninterrupted and the resumed input in one directory, which holds the restart.
    let workdir = env::temp_dir().join(format!("noci-rs-{fixture}-{}", std::process::id()));
    fs::create_dir_all(&workdir).unwrap();
    let full = assert_qmc_trajectory(fixture, &workdir);
    let resume = fixture_dir(fixture)
        .canonicalize()
        .unwrap()
        .join("resume.lua");
    let resumed = qmc_report_energies(&resume, &workdir);
    fs::remove_dir_all(&workdir).unwrap();

    // The resumed run starts from the state after three iterations and prints it as its first row.
    assert_eq!(
        resumed.len() + 3,
        full.len(),
        "{fixture}: resumed trajectory length"
    );
    for (i, (&got, &expected)) in resumed.iter().zip(&full[3..]).enumerate() {
        assert_close(
            got,
            expected,
            1e-9,
            &format!("{fixture} resumed report {i}"),
        );
    }
}
