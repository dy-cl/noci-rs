// mod.rs
// Standard library imports.
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};

// External crate imports.
use ndarray::Array2;
use noci_rs::input::{Input, load_input};
use noci_rs::integrals::generate_ao_data;
use noci_rs::{AoData, NOCIScalar};
use serde::de::DeserializeOwned;

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
    let input = load_input(dir.join("input.lua")).unwrap();
    let ao = generate_ao_data(&input.mol.geoms[0], &input.mol.basis, &input.mol.unit);
    let expected: T =
        serde_json::from_str(&fs::read_to_string(dir.join("expected.json")).unwrap()).unwrap();
    (input, ao, expected)
}

/// Return a process-local lock and Universe for tests that initialise MPI.
/// # Returns
/// - `(MutexGuard<'static, ()>, &'static mpi::environment::Universe)`: Guard held while MPI is used
///   and the process-global MPI universe.
pub fn mpi_universe() -> (MutexGuard<'static, ()>, &'static mpi::environment::Universe) {
    static MPI_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    static MPI_UNIVERSE: OnceLock<mpi::environment::Universe> = OnceLock::new();

    let lock = MPI_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("MPI test lock poisoned");
    let universe =
        MPI_UNIVERSE.get_or_init(|| mpi::initialize().expect("MPI initialisation failed"));

    (lock, universe)
}

/// Assert that two floating point numbers agree within tolerance
/// # Arguments:
/// - `x`: Calculated value.
/// - `y`: Reference value.
/// - `tol`: Maximum allowed absolute error.
/// - `label`: Description printed if assertion fails.
pub fn assert_close(
    x: f64,
    y: f64,
    tol: f64,
    label: &str,
) {
    let err = (x - y).abs();
    assert!(
        err < tol,
        "{label}: expected {y}, got {x}, |Δ|: {err}, tol: {tol}"
    );
}

/// Assert that Hamiltonian and overlap matrices from the extended nonorthogonal Wick's theorem
/// agree element by element with those from the naive generalised Slater-Condon rules.
/// # Arguments:
/// - `wicks`: Hamiltonian and overlap matrices from Wick's theorem.
/// - `naive`: Hamiltonian and overlap matrices from the naive rules.
/// - `tol`: Maximum allowed absolute error.
/// - `label`: Description printed if assertion fails.
// Only the Wick agreement tests compare matrix elements.
#[allow(dead_code)]
pub fn assert_wicks_naive_agree<T: NOCIScalar>(
    wicks: (&Array2<T>, &Array2<T>),
    naive: (&Array2<T>, &Array2<T>),
    tol: f64,
    label: &str,
) {
    let (h_wicks, s_wicks) = wicks;
    let (h_naive, s_naive) = naive;
    assert_eq!(h_wicks.shape(), h_naive.shape(), "{label}: H shapes differ");
    assert_eq!(s_wicks.shape(), s_naive.shape(), "{label}: S shapes differ");
    for ((i, j), &h) in h_wicks.indexed_iter() {
        let s = s_wicks[(i, j)];
        let (hn, sn) = (h_naive[(i, j)], s_naive[(i, j)]);
        let (dh, ds) = ((h - hn).abs(), (s - sn).abs());
        assert!(
            dh < tol,
            "{label} H[{i}, {j}]: expected {hn}, got {h}, |Δ|: {dh}, tol: {tol}"
        );
        assert!(
            ds < tol,
            "{label} S[{i}, {j}]: expected {sn}, got {s}, |Δ|: {ds}, tol: {tol}"
        );
    }
}
