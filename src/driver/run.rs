// driver/run.rs

// Standard library imports.
use std::fs;
use std::time::Instant;

// External crate imports.
use mpi::collective::CommunicatorCollectives;
use mpi::topology::Communicator;
use rayon::ThreadPoolBuilder;

// Crate-root imports.
use crate::driver::geometry::run_geometry;
use crate::driver::report::print_report;
use crate::driver::types::Atoms;
use crate::input::Input;
use crate::paths::RunPaths;
use crate::write::print_input;
use crate::{Error, HSCFState, Result, SCFState};

/// Run the full program.
/// # Arguments:
/// - `input`: User input specifications.
/// # Returns:
/// - `()`: Runs all requested geometries and prints reports.
pub fn run(mut input: Input) -> Result<()> {
    ThreadPoolBuilder::new()
        .stack_size(128 * 1024 * 1024)
        .build_global()
        .unwrap();

    let paths = RunPaths::from_input(&input);
    if input.wicks.cachedir.is_none() {
        input.wicks.cachedir = Some(paths.wicks_cache_dir.to_string_lossy().into_owned());
    }
    fs::create_dir_all(&paths.output_dir).map_err(|source| {
        Error::io(
            "failed to create output directory",
            &paths.output_dir,
            source,
        )
    })?;
    if let Some(parent) = paths.integral_file.parent() {
        fs::create_dir_all(parent)
            .map_err(|source| Error::io("failed to create integral directory", parent, source))?;
    }

    let t_total = Instant::now();
    let mut prev_states: Vec<SCFState> = Vec::new();
    let mut prev_htracks: Vec<HSCFState> = Vec::new();

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
        let res = run_geometry(
            r,
            atoms,
            &mut input,
            &prev_states,
            &prev_htracks,
            &paths,
            &world,
        )?;

        if irank == 0 {
            print_report(&res, &input);
        }

        prev_states = res.states.clone();
        prev_htracks = res.htracks.clone();
    }
    if irank == 0 {
        println!("\n Total wall time: {:?}", t_total.elapsed());
    }
    Ok(())
}
