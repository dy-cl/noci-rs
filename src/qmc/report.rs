// qmc/report.rs

// Standard library imports.
use std::path::Path;

// External crate imports.
use mpi::topology::Communicator;
use mpi::traits::*;

/// Decide on every rank whether a `STOP` file requests the run to end.
/// Rank zero observes the file and broadcasts one decision so every rank leaves the propagation
/// loop, and enters any collective restart write, together.
/// # Arguments:
/// - `irank`: MPI rank.
/// - `nranks`: Number of MPI ranks.
/// - `world`: MPI communicator.
/// # Returns:
/// - `bool`: Whether the run should stop.
pub(crate) fn stop_requested(
    irank: usize,
    nranks: usize,
    world: &impl Communicator,
) -> bool {
    let mut stop = 0;

    if irank == 0 && Path::new("STOP").exists() {
        stop = 1;
    }

    if nranks > 1 {
        world.process_at_rank(0).broadcast_into(&mut stop);
    }

    stop != 0
}

/// Print the stochastic iteration table header on rank zero. The energy columns are the
/// projected-energy numerator and denominator, their ratio, its correlation part and the shift,
/// followed by four population columns named by the method.
/// # Arguments:
/// - `irank`: MPI rank.
/// - `populations`: Names of the four population columns, `-` for unused ones.
/// # Returns:
/// - `()`: Writes the header.
pub(crate) fn print_table_header(
    irank: usize,
    populations: [&str; 4],
) {
    if irank == 0 {
        let [n, nref, naux, naux_occ] = populations;
        println!("{}", "=".repeat(132));
        println!(
            "{:<8} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16}",
            "Iter", "EProjNum", "EProjDen", "EProj", "ECorr", "EShift", n, nref, naux, naux_occ,
        );
    }
}

/// Print one stochastic iteration row on rank zero, with the projected energy
/// `E_{\text{proj}} = N_{\text{num}}/N_{\text{den}}`.
/// # Arguments:
/// - `irank`: MPI rank.
/// - `iter`: Iteration number.
/// - `projected`: Projected-energy numerator and denominator.
/// - `energies`: Energy `E_0` defining the correlation energy, and the shift (zero before
///   population control starts).
/// - `populations`: Total and reference populations, and the sampled population and its number
///   of occupied entries where the method has them.
/// # Returns:
/// - `()`: Writes the row.
pub(crate) fn print_table_row(
    irank: usize,
    iter: usize,
    projected: (f64, f64),
    energies: (f64, f64),
    populations: (f64, f64, Option<(f64, i64)>),
) {
    if irank != 0 {
        return;
    }
    let (num, den) = projected;
    let (e0, shift) = energies;
    let eproj = num / den;
    let (nw, nref, sampled) = populations;
    match sampled {
        Some((nsampled, nsampledo)) => println!(
            "{:<8} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.6} {:>16.6} {:>16.6} {:>16}",
            iter,
            num,
            den,
            eproj,
            eproj - e0,
            shift,
            nw,
            nref,
            nsampled,
            nsampledo,
        ),
        None => println!(
            "{:<8} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.6} {:>16.6} {:>16} {:>16}",
            iter,
            num,
            den,
            eproj,
            eproj - e0,
            shift,
            nw,
            nref,
            "-",
            "-",
        ),
    }
}
