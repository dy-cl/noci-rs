// driver/nocc/jmnoccmc.rs

// External crate imports.
use mpi::topology::Communicator;

// Crate-root imports.
use crate::PostSCFData;
use crate::determinant::NOCIIndex;
use crate::elements::NOCIData;
use crate::elements::nonorthogonalwicks::WicksView;
use crate::input::{InitialShift, Input};
use crate::nocc::{JMEquations, jm_qmc_step, jm_space, propagate_populations, report_converged};
use crate::noci::{build_noci_hs, build_noci_s};

/// Perform the deterministic JM-NOCCMC propagation on the reference determinants, the
/// noise-free counterpart of the stochastic propagation. Builds the projection and expansion
/// spaces of every reference, the rectangular Hamiltonian and overlap between them, propagates
/// the populations from the reference NOCI state and reports the state energies.
/// # Arguments:
/// - `post`: Data shared by post-SCF methods.
/// - `input`: User input specifications.
/// - `reference`: Reference NOCI coefficients and energy, giving the initial reference
///   populations, the trial and the initial shift.
/// - `wicks`: Optional precomputed Wick's intermediates of the reference parents.
/// # Returns:
/// - `Option<f64>`: Ground-state JM-NOCCMC energy, or `None` if the propagation failed.
/// # Panics
/// - Panics if the JM-NOCCMC options are absent.
pub(crate) fn run_jmnoccmc(
    post: &PostSCFData<'_, f64>,
    input: &Input,
    reference: (&[f64], f64),
    wicks: Option<&WicksView<f64>>,
) -> Option<f64> {
    let options = input
        .jmnoccmc
        .as_ref()
        .expect("JM-NOCCMC options are required");
    let det = options
        .det
        .as_ref()
        .expect("deterministic JM-NOCCMC options are required");

    // Expand every reference into its projection and expansion spaces.
    println!("{}", "=".repeat(100));
    println!("Building JM-NOCCMC spaces....");
    let space = jm_space(post.space, options);
    let rows = space.rows();
    let columns = (0..space.space.len()).map(NOCIIndex).collect::<Vec<_>>();
    println!(
        "Built JM-NOCCMC spaces of {} projection and {} expansion determinants over {} references.",
        rows.len(),
        columns.len(),
        post.space.len()
    );
    println!(
        "Calculating JM-NOCCMC matrix elements ({} elements)...",
        rows.len() * columns.len()
    );

    // Materialise the rectangular Hamiltonian and overlap between the two spaces.
    let data =
        NOCIData::new(post.ao, &space.space, input, post.tol, wicks).withmocache(post.mocache);
    let (h, s, _) = build_noci_hs(&data, &rows, &columns, false);
    println!("Finished calculating JM-NOCCMC matrix elements.");

    // The external fractions of the Jacobian diagnostics need the overlap between every pair of
    // expansion determinants.
    let metric = det.diagnostics.then(|| {
        println!(
            "Calculating JM-NOCCMC expansion overlap for diagnostics ({} elements)...",
            columns.len() * columns.len()
        );
        build_noci_s(&data, &columns, &columns, true).0
    });

    println!("Running deterministic JM-NOCCMC propagation....");
    let jm = JMEquations::new(space, h, s, metric);
    let mut history = Vec::new();
    let populations = propagate_populations(&jm, options.dt, det, reference, &mut history)?;
    Some(report_converged(
        &jm,
        &populations,
        &history,
        det,
        &input.write.write_dir,
    ))
}

/// Perform the stochastic JM-NOCCMC propagation on the reference determinants with every MPI
/// rank.
/// # Arguments:
/// - `post`: Data shared by post-SCF methods.
/// - `input`: User input specifications.
/// - `reference`: Reference NOCI coefficients and energy, giving the initial reference
///   populations and, with `initial_shift = "noci"`, the initial shift.
/// - `wicks`: Optional precomputed Wick's intermediates of the reference parents.
/// - `world`: MPI communicator.
/// # Returns:
/// - `f64`: Projected energy of the last report.
/// # Panics
/// - Panics if the JM-NOCCMC options are absent.
pub(crate) fn run_jmnoccmc_stochastic(
    post: &PostSCFData<'_, f64>,
    input: &Input,
    reference: (&[f64], f64),
    wicks: Option<&WicksView<f64>>,
    world: &impl Communicator,
) -> f64 {
    let options = input
        .jmnoccmc
        .as_ref()
        .expect("JM-NOCCMC options are required");
    let (c0, e_noci) = reference;

    // Start population control from the requested shift, as stochastic NOCIQMC does.
    let shift = match options.qmc.as_ref().unwrap().initial_shift {
        InitialShift::Hf => post.space.parents()[0].e,
        InitialShift::Noci => e_noci,
        InitialShift::Value(x) => x,
    };

    if world.rank() == 0 {
        println!("{}", "=".repeat(100));
        println!(
            "Running stochastic JM-NOCCMC propagation over {} references....",
            post.space.len()
        );
    }
    let data = NOCIData::new(post.ao, post.space, input, post.tol, wicks).withmocache(post.mocache);
    jm_qmc_step(&data, options, c0, shift, world)
}
