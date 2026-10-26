// nocc/stochastic/report.rs

/// Print the stochastic JM-NOCCMC table header on rank zero.
/// # Arguments:
/// - `irank`: MPI rank.
/// # Returns:
/// - `()`: Writes the header.
pub(super) fn print_header(irank: usize) {
    if irank != 0 {
        return;
    }
    println!("{}", "=".repeat(120));
    println!(
        "{:>8} {:>18} {:>18} {:>18} {:>18} {:>14} {:>14} {:>10}",
        "Iter", "EBlock", "EAverage", "ECorr", "EShift", "NWalk", "NRef", "NExcitors"
    );
}

/// Print one report row on rank zero.
/// # Arguments:
/// - `irank`: MPI rank.
/// - `iteration`: Iteration at the end of the report.
/// - `energies`: Report-block energy, running average, reference energy and shift, printed as
///   zero before population control starts.
/// - `populations`: Total population, reference population 1-norm and occupied excitors.
/// # Returns:
/// - `()`: Writes the row.
pub(super) fn print_row(
    irank: usize,
    iteration: usize,
    energies: (f64, f64, f64, f64),
    populations: (f64, f64, usize),
) {
    if irank != 0 {
        return;
    }
    let (block, average, reference, shift) = energies;
    let (total, nref, excitors) = populations;
    println!(
        "{:>8} {:>18.12} {:>18.12} {:>18.12} {:>18.12} {:>14.3} {:>14.3} {:>10}",
        iteration,
        block,
        average,
        block - reference,
        shift,
        total,
        nref,
        excitors
    );
}
