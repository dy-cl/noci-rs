// mod.rs
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::de::DeserializeOwned;

use noci_rs::AoData;
use noci_rs::input::{Input, load_input};
use noci_rs::read::read_integrals;

/// Return name of a directory containing a test fixture.
/// # Arguments:
/// - `name`: Name of the fixture.
/// # Returns
/// - `PathBuf`: Path to `tests/fixtures/<name>`.
pub fn fixture_dir(name: &str) -> PathBuf {
    Path::new("tests").join("fixtures").join(name)
}

/// Load a test fixture by reading the input and generating the HDF5 data file.
/// # Arguments:
/// - `name`: Name of the fixture.
/// # Returns
/// - `(Input, AoData, Expected)`: Parsed input, generated AO data, and expected energies.
pub fn load_test<T: DeserializeOwned>(name: &str) -> (Input, AoData, T) {
    let dir = fixture_dir(name);
    let input = load_input(dir.join("input.lua").to_str().unwrap());
    generate_data_h5(&dir, &input);
    let input = load_input(dir.join("input.lua").to_str().unwrap());
    let ao = read_integrals(dir.join("data.h5").to_str().unwrap());
    let expected: T = serde_json::from_str(&fs::read_to_string(dir.join("expected.json")).unwrap()).unwrap();
    (input, ao, expected)
}

/// Generate the `data.h5` file for a fixture which contains AO integrals.
/// # Arguments:
/// - `dir`: Fixture directory from which to run `generate.py`.
/// - `input`: User input specifications for this fixture.
fn generate_data_h5(dir: &std::path::Path, input: &Input) {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let generate_py = root.join("scripts/generate.py");
    let atoms: Vec<String> = input.mol.geoms[0].clone();
    let atomsj = serde_json::to_string(&atoms).unwrap();

    Command::new("python3").arg(&generate_py).arg("--atoms").arg(&atomsj).arg("--basis").arg(&input.mol.basis).arg("--unit")
    .arg(&input.mol.unit).arg("--out").arg("data.h5").arg("--fci").arg(if input.scf.do_fci {"true"} else {"false"}).current_dir(dir)
    .status().unwrap();
}

/// Assert that two floating point numbers agree within tolerance
/// # Arguments:
/// - `x`: Calculated value.
/// - `y`: Reference value.
/// - `tol`: Maximum allowed absolute error.
/// - `label`: Description printed if assertion fails.
pub fn assert_close(x: f64, y: f64, tol: f64, label: &str) {
    let err = (x - y).abs();
    assert!(err < tol, "{label}: expected {y}, got {x}, |Δ|: {err}, tol: {tol}");
}
