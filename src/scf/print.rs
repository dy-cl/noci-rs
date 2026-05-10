// scf/print.rs

use ndarray::Array1;

use crate::input::{SCFExcitation, Input};

use super::cycle::spin_label;

/// Print SCF header information.
/// # Arguments
/// - `input`: Contains user specified input data.
/// - `scfexcitation`: Optional excited SCF occupation request.
pub (in crate::scf) fn print_header(input: &Input, scfexcitation: Option<&SCFExcitation>) {
    if !input.write.verbose {return;}
    match scfexcitation {
        Some(ex) => {
            let sp = spin_label(&ex.spin);
            println!("Requested excitation: [spin: {}, from occupied: {}, to virtual: {}]", sp, ex.occ, ex.vir);
        }
        None => println!("No excitation requested."),
    }
    println!("{:>4} {:>12} {:>12} {:>12}", "i", "E", "dE", "‖FDS - SDF‖");
}

/// Print h-SCF iteration header.
/// # Arguments:
/// - `input`: User input specifications.
/// - `label`: Label for the h-SCF state.
/// # Returns:
/// - `()`: Prints header if verbose output is enabled.
pub (in crate::scf) fn print_header_h(input: &Input, label: &str) {
    if !input.write.verbose {return;}
    println!("{}Begin h-SCF{}", "=".repeat(45), "=".repeat(46));
    println!("State: {label}");
    println!("{:>4} {:>16} {:>17} {:>12}", "i", "Re(E)", "Im(E)", "||g_ov||");
}

/// Print MO occupations and energies.
/// # Arguments
/// - `title`: Title for this spin channel.
/// - `e`: MO energies.
/// - `occ`: MO occupation vector.
pub (in crate::scf) fn print_mos(title: &str, e: &Array1<f64>, occ: &Array1<f64>) {
    println!("{}", "-".repeat(100));
    let mut mos: Vec<(f64, usize, bool)> = (0..e.len()).map(|i| (e[i], i, occ[i] > 0.5)).collect();
    mos.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("{title}:"); println!("{:^5} {:^5} {:^5}", "MO", "Occ", "E");
    for (e, i, occ) in mos.iter() {println!("{:^5} {:^5.6} {:^5.6}", i, if *occ {1} else {0}, e);}
}
