// driver/nocc.rs

// External crate imports.
use mpi::topology::Communicator;
use ndarray::Array1;

// Crate-root imports.
use crate::PostSCFData;
use crate::input::Input;
use crate::nocc::{
    AmplitudeSolution, ReferenceState, TermEvaluator, build_excitations, build_fois_basis,
    build_spaces, cumulants, noci_natural_orbitals, print_noci_natural_orbitals, rdm1, rdm2, rdm3,
    rdm4, reference_energy, solve_amplitudes, transform_ao_data, transform_noci_basis,
};
use crate::noci::{NOCIData, build_wicks_shared};
use crate::nonorthogonalwicks::{WickScratchSpin, WicksShared};

/// Run GNOCC on a NOCI reference in its natural-orbital basis.
/// Builds the NOCI natural orbitals while the reference Wick storage is available, releases that
/// storage, transforms the reference into the natural-orbital basis, builds the spin-free RDMs
/// and cumulants, partitions the natural orbitals into core, active and virtual spaces,
/// constructs the FOIS, solves the amplitude equations and reports the GNOCC energy.
/// # Arguments:
/// - `post`: Data shared by post-SCF methods.
/// - `input`: User input specifications.
/// - `c0`: Reference NOCI coefficient vector.
/// - `wicks`: Reference Wick storage, released before the natural-orbital storage is built.
/// - `world`: MPI communicator object.
/// # Returns:
/// - `()`: Prints the natural orbitals, the amplitude iterations and the GNOCC energy.
/// # Panics
/// - Panics if the NOCCMC options are absent.
pub(crate) fn run_gnocc(
    post: &PostSCFData<'_, f64>,
    input: &Input,
    c0: &[f64],
    wicks: &mut Option<WicksShared<f64>>,
    world: &impl Communicator,
) {
    let coeffs = Array1::from_vec(c0.to_vec());

    // Build natural orbitals before releasing Wick storage for the GNOCC calculation.
    let no = {
        let view = wicks.as_ref().map(|ws| ws.view());
        let data =
            NOCIData::new(post.ao, post.space, input, post.tol, view).withmocache(post.mocache);

        let tol = input.noccmc.as_ref().map_or(1e-6, |n| n.active_space_tol);
        noci_natural_orbitals(&data, &coeffs, tol, tol)
    };
    wicks.take();

    if world.rank() == 0 {
        print_noci_natural_orbitals("NOCI natural orbitals", &no);
    }

    // Transform the reference into the natural-orbital basis with its own Wick storage.
    let nobasis = transform_noci_basis(post.space, &no.c, &post.ao.s);
    let noao = transform_ao_data(post.ao, &no.c);

    let nowicks = if input.wicks.enabled {
        Some(build_wicks_shared(
            world,
            &noao,
            &nobasis.parents,
            post.tol,
            input,
        ))
    } else {
        None
    };

    let nowicksview = nowicks.as_ref().map(|w| w.view());
    let nodata = NOCIData::new(&noao, &nobasis, input, post.tol, nowicksview);

    if world.rank() == 0 {
        println!("{}", "=".repeat(100));
        println!("Running GNOCC in NOCI natural orbital basis....");
    }

    // Each RDM builds with its own Wick scratch, when Wick's theorem is enabled.
    let mut scratch = input.wicks.enabled.then(WickScratchSpin::new);
    let (_, gamma1) = rdm1(&nodata, &coeffs, &coeffs, scratch.as_mut());
    let mut scratch = input.wicks.enabled.then(WickScratchSpin::new);
    let (_, gamma2) = rdm2(&nodata, &coeffs, &coeffs, scratch.as_mut());
    let mut scratch = input.wicks.enabled.then(WickScratchSpin::new);
    let (_, gamma3) = rdm3(&nodata, &coeffs, &coeffs, &no.active, scratch.as_mut());
    let mut scratch = input.wicks.enabled.then(WickScratchSpin::new);
    let (_, gamma4) = rdm4(&nodata, &coeffs, &coeffs, &no.active, scratch.as_mut());

    let lambdas = cumulants(&gamma1, &gamma2, &gamma3, &gamma4, &no.active);

    // Orbital spaces, excitation manifold, normal-ordered reference and weighted FOIS.
    let options = input.noccmc.as_ref().expect("NOCCMC options are required");
    let tol = options.active_space_tol;
    let spaces = build_spaces(gamma1.n, &no.active, &gamma1, tol, tol);
    let excitations = build_excitations(&spaces);
    let reference = ReferenceState::new(&noao, &gamma1, &lambdas);
    let evaluator = TermEvaluator::new(options.max_cumulant, &spaces);
    let fois = build_fois_basis(&reference, &spaces, &excitations, &evaluator, options);

    if world.rank() == 0 {
        // Solve the amplitude equations and report the GNOCC energy.
        let e0 = reference_energy(&noao, &gamma1, &gamma2);
        let solution = solve_amplitudes(
            &reference,
            &spaces,
            &excitations,
            &evaluator,
            &fois,
            e0,
            options,
        );
        print_solution(e0, &solution);
    }
}

/// Print the final GNOCC amplitude solution.
/// # Arguments:
/// - `e0`: Reference energy `\langle\Phi|\hat H|\Phi\rangle`.
/// - `solution`: Final amplitude-equation state.
/// # Returns:
/// - `()`: Prints the reference, correlation and total energies.
fn print_solution(
    e0: f64,
    solution: &AmplitudeSolution,
) {
    println!("{}", "=".repeat(100));
    println!("GNOCC energy");
    println!(
        "Converged: {} after {} iterations (||Y^T R||: {:.4e})",
        solution.converged, solution.iterations, solution.residual_norm
    );
    println!("Reference energy: {:.12}", e0);
    println!("Correlation energy: {:.12}", solution.correlation_energy);
    println!("Total energy: {:.12}", e0 + solution.correlation_energy);
}
