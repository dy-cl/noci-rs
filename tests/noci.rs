mod common;

use std::fs;
use std::path::Path;
use std::process::Command;

use common::{assert_close, fixture_dir, load_test, mpi_universe};
use num_complex::Complex64;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serial_test::serial;

use noci_rs::basis::{generate_reference_noci_basis, hermitian_hnoci_basis};
use noci_rs::input::{Input, load_input};
use noci_rs::noci::{build_mo_cache, build_wicks_shared, calculate_noci_energy};
use noci_rs::read::read_integrals;
use noci_rs::{AoData, HSCFState, SCFState};

/// Expected exact energies for a reference NOCI fixture.
#[derive(Deserialize)]
struct ExpectedReferenceNoci {
    /// Expected SCF state energies.
    scf_energies: Vec<f64>,
    /// Expected reference-space NOCI energy.
    reference_noci_energy: f64,
}

/// Expected complex value in JSON fixtures.
#[derive(Deserialize)]
struct ExpectedComplex {
    /// Expected real component.
    real: f64,
    /// Expected imaginary component.
    imag: f64,
}

/// Expected exact energies for a holomorphic reference NOCI fixture.
#[derive(Deserialize)]
struct ExpectedHolomorphicReferenceNoci {
    /// Expected reference NOCI energies in geometry order.
    reference_noci_energies: Vec<f64>,
    /// Expected final-geometry real SCF energies.
    final_scf_energies: Vec<f64>,
    /// Expected final-geometry holomorphic SCF energies.
    final_hscf_energies: Vec<ExpectedComplex>,
}

/// Load a geometry-scan fixture by reading the input and generating HDF5 data for each geometry.
/// # Arguments:
/// - `name`: Name of the fixture.
/// # Returns
/// - `(Input, Vec<AoData>, Expected)`: Parsed input, generated AO data for each geometry, and
///   expected energies.
fn load_scan_test<T: DeserializeOwned>(name: &str) -> (Input, Vec<AoData>, T) {
    let dir = fixture_dir(name);
    let input = load_input(dir.join("input.lua").to_str().unwrap());

    let mut aos = Vec::with_capacity(input.mol.geoms.len());
    for i in 0..input.mol.geoms.len() {
        let fname = format!("data_{i}.h5");
        generate_data_h5_for_geometry(&dir, &input, i, &fname);
        aos.push(read_integrals(dir.join(fname).to_str().unwrap()));
    }

    let input = load_input(dir.join("input.lua").to_str().unwrap());
    let expected: T =
        serde_json::from_str(&fs::read_to_string(dir.join("expected.json")).unwrap()).unwrap();
    (input, aos, expected)
}

/// Generate an HDF5 data file for one fixture geometry.
/// # Arguments:
/// - `dir`: Fixture directory from which to run `generate.py`.
/// - `input`: User input specifications for this fixture.
/// - `geometry`: Geometry index to generate.
/// - `out`: Output HDF5 file name.
/// # Returns
/// - `()`: Writes the requested HDF5 data file.
fn generate_data_h5_for_geometry(
    dir: &Path,
    input: &Input,
    geometry: usize,
    out: &str,
) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let generate_py = root.join("scripts/generate.py");
    let atoms: Vec<String> = input.mol.geoms[geometry].clone();
    let atomsj = serde_json::to_string(&atoms).unwrap();

    Command::new("python3")
        .arg(&generate_py)
        .arg("--atoms")
        .arg(&atomsj)
        .arg("--basis")
        .arg(&input.mol.basis)
        .arg("--unit")
        .arg(&input.mol.unit)
        .arg("--out")
        .arg(out)
        .arg("--fci")
        .arg(if input.scf.do_fci { "true" } else { "false" })
        .current_dir(dir)
        .status()
        .unwrap();
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

/// Run holomorphic SCF continuation and reference NOCI for a geometry-scan fixture.
/// # Arguments:
/// - `input`: Parsed input for the scan fixture.
/// - `aos`: AO data for each geometry.
/// # Returns
/// - `(Vec<f64>, Vec<f64>, Vec<Complex64>, f64)`: Reference NOCI energies in geometry order,
///   final real SCF energies, final h-SCF energies and the largest final imaginary orbital
///   component.
fn run_holomorphic_reference_noci_fixture(
    mut input: Input,
    aos: &[AoData],
) -> (Vec<f64>, Vec<f64>, Vec<Complex64>, f64) {
    let (_mpi_lock, universe) = mpi_universe();
    let world = universe.world();

    let mut prev_states: Vec<SCFState> = Vec::new();
    let mut prev_hstates: Vec<HSCFState> = Vec::new();
    let mut energies = Vec::with_capacity(aos.len());
    let mut final_scf = Vec::new();
    let mut final_hscf = Vec::new();
    let mut final_imag = 0.0;

    for ao in aos {
        let refs = if prev_states.is_empty() && prev_hstates.is_empty() {
            generate_reference_noci_basis(ao, &mut input, None, None)
        } else {
            let prev_h = if prev_hstates.is_empty() {
                None
            } else {
                Some(prev_hstates.as_slice())
            };
            generate_reference_noci_basis(ao, &mut input, Some(&prev_states), prev_h)
        };

        let mut basis: Vec<_> = hermitian_hnoci_basis(&refs.hstates, &ao.s)
            .into_iter()
            .filter(|st| st.noci_basis)
            .collect();
        for (i, st) in basis.iter_mut().enumerate() {
            st.parent = i;
        }

        let mocache = build_mo_cache(ao, &basis, input.scf.d_tol);
        let wicks = if input.wicks.enabled || input.wicks.compare {
            Some(build_wicks_shared::<Complex64>(
                &world, ao, &basis, 1e-12, &input,
            ))
        } else {
            None
        };
        let wicks_view = wicks.as_ref().map(|w| w.view());
        let (e_ref, _c0, _dt_hs) =
            calculate_noci_energy(ao, &input, &basis, 1e-12, &mocache, wicks_view);

        energies.push(e_ref);
        final_scf = refs.states.iter().map(|st| st.e).collect();
        let holomorphic_states: Vec<_> = refs
            .hstates
            .iter()
            .filter(|st| st.label.starts_with("h-"))
            .collect();
        final_hscf = holomorphic_states.iter().map(|st| st.e).collect();
        final_imag = holomorphic_states
            .iter()
            .flat_map(|st| st.ca.iter().chain(st.cb.iter()))
            .map(|z| z.im.abs())
            .fold(0.0, f64::max);

        prev_states = refs.states;
        prev_hstates = refs.hstates;
    }

    (energies, final_scf, final_hscf, final_imag)
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

/// Test that the H2 3-21G 0.8 Angstrom holomorphic fixture reproduces expected energies.
/// # Panics
/// - If reference NOCI, final real SCF or final h-SCF energies differ from known good values
///   outside tolerance.
/// - If holomorphic continuation does not leave a non-negligible imaginary orbital component.
#[test]
#[serial]
fn reference_noci_h2_3_21g_0_8_ang_holomorphic_energies() {
    let (input, aos, expected): (_, _, ExpectedHolomorphicReferenceNoci) =
        load_scan_test("REF_NOCI_H2_3-21G_0_8");
    let (got_ref, got_scf, got_hscf, got_imag) =
        run_holomorphic_reference_noci_fixture(input, &aos);

    assert_eq!(got_ref.len(), expected.reference_noci_energies.len());
    for (i, (&x, &y)) in got_ref
        .iter()
        .zip(expected.reference_noci_energies.iter())
        .enumerate()
    {
        assert_close(x, y, 1e-8, &format!("H2 holomorphic reference NOCI {i}"));
    }

    assert_eq!(got_scf.len(), expected.final_scf_energies.len());
    for (i, (&x, &y)) in got_scf
        .iter()
        .zip(expected.final_scf_energies.iter())
        .enumerate()
    {
        assert_close(x, y, 1e-8, &format!("H2 final SCF state {i}"));
    }

    assert_eq!(got_hscf.len(), expected.final_hscf_energies.len());
    for (i, (x, y)) in got_hscf
        .iter()
        .zip(expected.final_hscf_energies.iter())
        .enumerate()
    {
        assert_close(
            x.re,
            y.real,
            1e-8,
            &format!("H2 final h-SCF state {i} real"),
        );
        assert_close(
            x.im,
            y.imag,
            1e-8,
            &format!("H2 final h-SCF state {i} imag"),
        );
    }

    assert!(got_imag > 1e-3);
    assert!(got_ref[6] < got_ref[4]);
    assert!(got_ref[6] < got_ref[0]);
}

/// Test that the H2 3-21G 0.8 Angstrom holomorphic fixture reproduces expected energies.
/// # Panics
/// - If reference NOCI, final real SCF or final h-SCF energies differ from known good values
///   outside tolerance.
/// - If holomorphic continuation does not leave a non-negligible imaginary orbital component.
#[test]
#[serial]
fn reference_noci_h2_3_21g_0_8_ang_holomorphic_energies_wicks() {
    let (input, aos, expected): (_, _, ExpectedHolomorphicReferenceNoci) =
        load_scan_test("REF_NOCI_H2_3-21G_0_8_WICKS");
    let (got_ref, got_scf, got_hscf, got_imag) =
        run_holomorphic_reference_noci_fixture(input, &aos);

    assert_eq!(got_ref.len(), expected.reference_noci_energies.len());
    for (i, (&x, &y)) in got_ref
        .iter()
        .zip(expected.reference_noci_energies.iter())
        .enumerate()
    {
        assert_close(x, y, 1e-8, &format!("H2 holomorphic reference NOCI {i}"));
    }

    assert_eq!(got_scf.len(), expected.final_scf_energies.len());
    for (i, (&x, &y)) in got_scf
        .iter()
        .zip(expected.final_scf_energies.iter())
        .enumerate()
    {
        assert_close(x, y, 1e-8, &format!("H2 final SCF state {i}"));
    }

    assert_eq!(got_hscf.len(), expected.final_hscf_energies.len());
    for (i, (x, y)) in got_hscf
        .iter()
        .zip(expected.final_hscf_energies.iter())
        .enumerate()
    {
        assert_close(
            x.re,
            y.real,
            1e-8,
            &format!("H2 final h-SCF state {i} real"),
        );
        assert_close(
            x.im,
            y.imag,
            1e-8,
            &format!("H2 final h-SCF state {i} imag"),
        );
    }

    assert!(got_imag > 1e-3);
    assert!(got_ref[6] < got_ref[4]);
    assert!(got_ref[6] < got_ref[0]);
}

/// Test that the H2 3-21G 0.8 Angstrom holomorphic fixture agrees with and without Wick's intermediates.
/// # Panics
/// - If reference NOCI, final real SCF or final h-SCF energies differ between implementations
///   outside tolerance.
#[test]
#[serial]
fn reference_noci_h2_3_21g_0_8_ang_holomorphic_energies_agree() {
    let (input, aos, _expected): (_, _, ExpectedHolomorphicReferenceNoci) =
        load_scan_test("REF_NOCI_H2_3-21G_0_8");
    let (input_wicks, aos_wicks, _expected_wicks): (_, _, ExpectedHolomorphicReferenceNoci) =
        load_scan_test("REF_NOCI_H2_3-21G_0_8_WICKS");
    let (got_ref, got_scf, got_hscf, got_imag) =
        run_holomorphic_reference_noci_fixture(input, &aos);
    let (got_ref_wicks, got_scf_wicks, got_hscf_wicks, got_imag_wicks) =
        run_holomorphic_reference_noci_fixture(input_wicks, &aos_wicks);

    assert_eq!(got_ref.len(), got_ref_wicks.len());
    for (i, (&x, &y)) in got_ref.iter().zip(got_ref_wicks.iter()).enumerate() {
        assert_close(
            x,
            y,
            1e-8,
            &format!("H2 holomorphic reference NOCI {i} Wicks agreement"),
        );
    }

    assert_eq!(got_scf.len(), got_scf_wicks.len());
    for (i, (&x, &y)) in got_scf.iter().zip(got_scf_wicks.iter()).enumerate() {
        assert_close(
            x,
            y,
            1e-8,
            &format!("H2 final SCF state {i} Wicks agreement"),
        );
    }

    assert_eq!(got_hscf.len(), got_hscf_wicks.len());
    for (i, (&x, &y)) in got_hscf.iter().zip(got_hscf_wicks.iter()).enumerate() {
        assert_close(
            x.re,
            y.re,
            1e-8,
            &format!("H2 final h-SCF state {i} real Wicks agreement"),
        );
        assert_close(
            x.im,
            y.im,
            1e-8,
            &format!("H2 final h-SCF state {i} imag Wicks agreement"),
        );
    }

    assert!(got_imag > 1e-3);
    assert!(got_imag_wicks > 1e-3);
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
