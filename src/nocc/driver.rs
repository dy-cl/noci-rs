// nocc/driver.rs

// External crate imports.
use mpi::topology::Communicator;
use ndarray::Array1;

// Crate-root imports.
use crate::PostSCFData;
use crate::input::Input;
use crate::nocc::contract::TermEvaluator;
use crate::nocc::energy::reference_energy;
use crate::nocc::reference::ReferenceState;
use crate::nocc::solver::{AmplitudeSolution, solve_amplitudes};
use crate::nocc::space;
use crate::nocc::space::ExcitationManifold;
use crate::nocc::{cumulants, rdm1, rdm2, rdm3, rdm4};
use crate::noci::{NOCIData, build_wicks_shared};
use crate::nonorthogonalwicks::WickScratchSpin;
use crate::orbitals::{
    NOCINaturalOrbitals, print_noci_natural_orbitals, transform_ao_data, transform_noci_basis,
};

/// Run GNOCC on a NOCI reference in its natural-orbital basis.
/// Builds the spin-free RDMs and cumulants of the reference, partitions the natural orbitals into
/// core, active and virtual spaces, constructs the FOIS, solves the amplitude equations and
/// reports the GNOCC energy.
/// # Arguments:
/// - `post`: Data shared by post-SCF methods.
/// - `input`: User input specifications.
/// - `c0`: Reference NOCI coefficient vector.
/// - `no`: NOCI natural orbital basis.
/// - `world`: MPI communicator object.
/// # Returns:
/// - `()`: Prints the natural orbitals, the amplitude iterations and the GNOCC energy.
pub(crate) fn run_noccmc(
    post: &PostSCFData<'_, f64>,
    input: &Input,
    c0: &[f64],
    no: &NOCINaturalOrbitals,
    world: &impl Communicator,
) {
    let coeffs = Array1::from_vec(c0.to_vec());

    if world.rank() == 0 {
        print_noci_natural_orbitals("NOCI natural orbitals", no);
    }

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

    let mut scratch1 = WickScratchSpin::new();
    let scratch1 = if input.wicks.enabled {
        Some(&mut scratch1)
    } else {
        None
    };
    let (_, gamma1) = rdm1(&nodata, &coeffs, &coeffs, scratch1);

    let mut scratch2 = WickScratchSpin::new();
    let scratch2 = if input.wicks.enabled {
        Some(&mut scratch2)
    } else {
        None
    };
    let (_, gamma2) = rdm2(&nodata, &coeffs, &coeffs, scratch2);

    let mut scratch3 = WickScratchSpin::new();
    let scratch3 = if input.wicks.enabled {
        Some(&mut scratch3)
    } else {
        None
    };
    let (_, gamma3) = rdm3(&nodata, &coeffs, &coeffs, &no.active, scratch3);

    let mut scratch4 = WickScratchSpin::new();
    let scratch4 = if input.wicks.enabled {
        Some(&mut scratch4)
    } else {
        None
    };
    let (_, gamma4) = rdm4(&nodata, &coeffs, &coeffs, &no.active, scratch4);

    let lambdas = cumulants(&gamma1, &gamma2, &gamma3, &gamma4, &no.active);

    let options = input.noccmc.as_ref().expect("NOCCMC options are required");
    let tol = options.active_space_tol;
    let spaces = space::build_spaces(gamma1.n, &no.active, &gamma1, tol, tol);
    let excitations = space::build_excitations(&spaces);
    let reference = ReferenceState::new(&noao, &gamma1, &lambdas);
    let manifold = ExcitationManifold {
        spaces: &spaces,
        excitations: &excitations,
    };
    let evaluator = TermEvaluator::new(options.max_cumulant, &spaces);
    let fois = space::build_fois_basis(&reference, &manifold, &evaluator, options);

    if world.rank() == 0 {
        // Solve the amplitude equations and report the GNOCC energy.
        let e0 = reference_energy(&noao, &gamma1, &gamma2);
        let solution = solve_amplitudes(&reference, &manifold, &evaluator, &fois, e0, options);
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
