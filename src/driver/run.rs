// driver/run.rs

use std::time::Instant;

use mpi::collective::CommunicatorCollectives;
use mpi::topology::Communicator;
use rayon::ThreadPoolBuilder;

use crate::driver::geometry::run_geometry;
use crate::driver::report::print_report;
use crate::driver::types::Atoms;
use crate::input::Input;
use crate::write::print_input;
use crate::{HSCFState, SCFState};

/// Run the full program.
/// # Arguments:
/// - `input`: User input specifications.
/// # Returns:
/// - `()`: Runs all requested geometries and prints reports.
pub fn run(mut input: Input) {
    ThreadPoolBuilder::new()
        .stack_size(128 * 1024 * 1024)
        .build_global()
        .unwrap();

    let t_total = Instant::now();
    let mut prev_states: Vec<SCFState> = Vec::new();
    let mut prev_hstates: Vec<HSCFState> = Vec::new();

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let irank = world.rank();

    if irank == 0 {
        print_input(&input);
    }
    world.barrier();

    let rlist = input.mol.r_list.clone();
    let geoms = input.mol.geoms.clone();

    for (i, r) in rlist.iter().copied().enumerate() {
        println!("\n");
        let atoms: &Atoms = &geoms[i];
        let res = run_geometry(r, atoms, &mut input, &prev_states, &prev_hstates, &world);
        if irank == 0 {
            print_report(&res, &input);
        }
        prev_states = res.states.clone();
        prev_hstates = res.hstates.clone();
    }
    if irank == 0 {
        println!("\n Total wall time: {:?}", t_total.elapsed());
    }
}
