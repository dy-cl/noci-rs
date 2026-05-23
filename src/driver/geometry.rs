// driver/geometry.rs

use mpi::collective::CommunicatorCollectives;
use mpi::topology::Communicator;

use crate::driver::post::{run_holomorphic_post_reference, run_real_post_reference};
use crate::driver::pyscf::run_pyscf;
use crate::driver::reference::{ReferenceKind, run_reference_space};
use crate::driver::scf::{
    HolomorphicReferencePrep, RealReferencePrep, generate_holomorphic_references,
    generate_real_references,
};
use crate::driver::types::{Atoms, GeometryResults};
use crate::input::{Input, StateType};
use crate::mpiutils::broadcast;
use crate::read::read_integrals;
use crate::{AoData, HSCFState, SCFState, timers};

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
/// - `prev_hstates`: Converged h-SCF states at previous r, used for complex branch tracking.
/// - `world`: MPI communicator object.
/// # Returns:
/// - `GeometryResults`: Calculated energies, timings, and SCF states for the current geometry.
pub fn run_geometry(
    r: f64,
    atoms: &Atoms,
    input: &mut Input,
    prev_states: &[SCFState],
    prev_hstates: &[HSCFState],
    world: &impl Communicator,
) -> GeometryResults {
    let tol = 1e-8;
    timers::reset_all();

    if world.rank() == 0 {
        run_pyscf(atoms, input);
    }
    world.barrier();

    let ao: AoData = read_integrals("data.h5");

    if should_run_holomorphic(input) {
        let mut prep = if world.rank() == 0 {
            generate_holomorphic_references(&ao, input, prev_states, prev_hstates)
        } else {
            HolomorphicReferencePrep {
                states: Vec::new(),
                hstates: Vec::new(),
                basis: Vec::new(),
            }
        };
        world.barrier();
        broadcast(world, &mut prep.states);
        broadcast(world, &mut prep.hstates);

        let holomorphic_noci_refs = matches!(
            &input.states,
            StateType::Mom(recipes) if recipes.iter().any(|st| st.holomorphic && st.noci)
        );
        let mut reference =
            run_reference_space(&ao, input, prep.basis, tol, ReferenceKind::Complex, world);
        let post = run_holomorphic_post_reference(
            &ao,
            &mut reference,
            input,
            tol,
            holomorphic_noci_refs,
            world,
        );
        let timings = timers::snapshot_all_mpi(world);
        GeometryResults::from_holomorphic(
            r,
            (prep.states, prep.hstates),
            reference,
            post,
            ao.e_fci,
            world.size() as usize,
            timings,
        )
    } else {
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
        let post = run_real_post_reference(&ao, &prep.states, &mut reference, input, tol, world);
        let timings = timers::snapshot_all_mpi(world);
        GeometryResults::from_real(
            r,
            prep.states,
            reference,
            post,
            ao.e_fci,
            world.size() as usize,
            timings,
        )
    }
}
