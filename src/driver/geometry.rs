// driver/geometry.rs

// External crate imports.
use mpi::collective::CommunicatorCollectives;
use mpi::topology::Communicator;

// Crate-root imports.
use crate::driver::post::{run_holomorphic_post_reference, run_real_post_reference};
use crate::driver::reference::{ReferenceKind, run_reference_space};
use crate::driver::scf::{
    HolomorphicReferencePrep, RealReferencePrep, generate_holomorphic_references,
    generate_real_references,
};
use crate::driver::types::{Atoms, GeometryResults};
use crate::input::{Input, StateType};
use crate::integrals::generate_ao_data;
use crate::mpiutils::broadcast;
use crate::{AoData, HSCFState, Result, SCFState, time_call, timers};

/// Decide whether this geometry needs holomorphic references.
/// # Arguments:
/// - `input`: User input specifications.
/// # Returns:
/// - `bool`: True if a holomorphic reference is requested.
pub fn should_run_holomorphic(input: &Input) -> bool {
    match &input.states {
        StateType::Mom(recipes) => recipes.iter().any(|r| r.holomorphic),
        StateType::Metadynamics(_) => false,
    }
}

/// Run one geometry calculation.
/// # Arguments:
/// - `r`: Current geometry.
/// - `atoms`: Atom types.
/// - `input`: User input specifications.
/// - `prev_states`: Converged SCF states at previous r, used for seeding.
/// - `prev_htracks`: Converged h-SCF states at previous r, used for complex branch tracking.
/// - `world`: MPI communicator object.
/// # Returns:
/// - `Result<GeometryResults>`: Calculated energies, timings, and SCF states.
/// # Errors
/// - The current geometry paths return `Ok`; no error is produced here.
/// # Panics
/// - Panics if the root-generated AO data is missing after broadcast or the generated state
///   lists violate the holomorphic-reference ordering assumptions.
pub fn run_geometry(
    r: f64,
    atoms: &Atoms,
    input: &mut Input,
    prev_states: &[SCFState],
    prev_htracks: &[HSCFState],
    world: &impl Communicator,
) -> Result<GeometryResults> {
    // Reset per-geometry diagnostics and distribute the root-generated AO data.
    let tol = 1e-8;
    timers::reset_all();

    let mut ao = if world.rank() == 0 {
        Some(time_call!(
            crate::timers::general::add_generate_integrals,
            { generate_ao_data(atoms, &input.mol.basis, &input.mol.unit) }
        ))
    } else {
        None
    };
    broadcast(world, &mut ao);
    let ao: AoData = ao.unwrap();

    // Build and distribute either holomorphic-tracking or ordinary SCF references.
    if should_run_holomorphic(input) {
        let mut prep = if world.rank() == 0 {
            generate_holomorphic_references(&ao, input, prev_states, prev_htracks)
        } else {
            HolomorphicReferencePrep {
                states: Vec::new(),
                hstates: Vec::new(),
                htracks: Vec::new(),
                basis: Vec::new(),
            }
        };
        world.barrier();
        broadcast(world, &mut prep.states);
        broadcast(world, &mut prep.hstates);

        // Use complex NOCI only when a newly generated holomorphic state survives selection.
        let holomorphic = prep.hstates[prep.states.len()..]
            .iter()
            .any(|state| state.noci_basis);

        if holomorphic {
            // Run the complex reference and post-reference paths, then snapshot global timings.
            let mut reference =
                run_reference_space(&ao, input, prep.basis, tol, ReferenceKind::Complex, world);
            let post =
                run_holomorphic_post_reference(&ao, &mut reference, input, tol, holomorphic, world);
            let timings = timers::snapshot_all_mpi(world);
            Ok(GeometryResults::from_holomorphic(
                r,
                (prep.states, prep.hstates, prep.htracks),
                reference,
                post,
                world.size() as usize,
                timings,
            ))
        } else {
            // Fall back to real NOCI while retaining holomorphic states for branch tracking.
            let mut reference = run_reference_space(
                &ao,
                input,
                prep.states.clone(),
                tol,
                ReferenceKind::Real,
                world,
            );
            let post = run_real_post_reference(&ao, &mut reference, input, tol, world);
            let timings = timers::snapshot_all_mpi(world);
            let mut results = GeometryResults::from_real(
                r,
                prep.states,
                reference,
                post,
                world.size() as usize,
                timings,
            );
            results.hstates = prep.hstates;
            results.htracks = prep.htracks;
            Ok(results)
        }
    } else {
        // Run the ordinary real-reference path when no holomorphic recipe is requested.
        let mut prep = if world.rank() == 0 {
            generate_real_references(&ao, input, prev_states)
        } else {
            RealReferencePrep {
                states: Vec::new(),
                basis: Vec::new(),
            }
        };
        world.barrier();
        broadcast(world, &mut prep.states);

        let mut reference =
            run_reference_space(&ao, input, prep.basis, tol, ReferenceKind::Real, world);
        let post = run_real_post_reference(&ao, &mut reference, input, tol, world);
        let timings = timers::snapshot_all_mpi(world);
        Ok(GeometryResults::from_real(
            r,
            prep.states,
            reference,
            post,
            world.size() as usize,
            timings,
        ))
    }
}
