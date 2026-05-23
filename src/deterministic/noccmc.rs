// deterministic/noccmc.rs

use ndarray::Array1;

use crate::PostSCFData;
use crate::input::Input;
use crate::noci::{NOCIData, build_spin_free_rdms_12};
use crate::nonorthogonalwicks::{WickScratchSpin, WicksView};

/// Run NOCCMC setup work.
/// # Arguments:
/// - `post`: Data shared by post-SCF methods.
/// - `input`: User input specifications.
/// - `c0`: Reference NOCI coefficient vector.
/// - `wicks`: Optional precomputed Wick's intermediates.
/// # Returns:
/// - `()`: Calls RDM construction so `wicks.compare` compares Wick and naive matrix elements.
pub fn run_noccmc(
    post: &PostSCFData<'_, f64>,
    input: &Input,
    c0: &[f64],
    wicks: Option<&WicksView<f64>>,
) {
    println!("{}", "=".repeat(100));
    println!("Running NOCCMC spin-free RDM check....");

    let data = NOCIData::new(post.ao, post.noci_reference_basis, input, post.tol, wicks)
        .withmocache(post.mocache);
    let coeffs = Array1::from_vec(c0.to_vec());
    let mut scratch = WickScratchSpin::new();
    let scratch = if input.wicks.enabled {
        Some(&mut scratch)
    } else {
        None
    };
    let _ = build_spin_free_rdms_12(&data, &coeffs, &coeffs, scratch);
}
