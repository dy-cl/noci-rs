// driver/pyscf.rs

use std::process::Command;

use crate::driver::types::Atoms;
use crate::input::Input;
use crate::time_call;

/// Call PySCF script to get the two electron integrals and core hamiltonian.
/// # Arguments:
/// - `atoms`: Atom types.
/// - `input`: User input specifications.
/// # Returns:
/// - `()`: Runs the PySCF interface and writes the generated integrals to disk.
pub fn run_pyscf(
    atoms: &Atoms,
    input: &Input,
) {
    time_call!(crate::timers::general::add_run_pyscf, {
        let atomsj = serde_json::to_string(atoms).unwrap();

        let status = Command::new("python3")
            .arg("scripts/generate.py")
            .arg("--atoms")
            .arg(&atomsj)
            .arg("--basis")
            .arg(&input.mol.basis)
            .arg("--unit")
            .arg(&input.mol.unit)
            .arg("--out")
            .arg("data.h5")
            .arg("--fci")
            .arg(if input.scf.do_fci { "true" } else { "false" })
            .status()
            .unwrap();

        if !status.success() {
            eprintln!("Failed to generate mol with status {status}");
            std::process::exit(1);
        }
    })
}
