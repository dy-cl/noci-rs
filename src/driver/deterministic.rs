use std::fs::File;
use std::io::{BufWriter, Write};

use ndarray::Array1;

use crate::PostSCFData;
use crate::basis::generate_excited_basis;
use crate::deterministic::{projected_energy, propagate};
use crate::input::Input;
use crate::noci::{NOCIData, build_noci_hs};
use crate::nonorthogonalwicks::WicksView;
use crate::time_call;
use crate::utils::wavefunction_sparsity;

/// Perform the deterministic propagation in the NOCI-QMC space.
/// # Arguments:
/// - `post`: Data shared by post-SCF methods.
/// - `input`: User input specifications.
/// - `c0`: Initial coefficient vector of basis states.
/// - `wicks`: Optional precomputed Wick's intermediates.
/// # Returns:
/// - `f64`: Propagated energy.
pub fn run_qmc_deterministic_noci(
    post: &PostSCFData<'_, f64>,
    input: &Input,
    c0: &[f64],
    wicks: Option<&WicksView<f64>>,
) -> f64 {
    time_call!(
        crate::timers::deterministic::add_run_qmc_deterministic_noci,
        {
            println!("{}", "=".repeat(100));
            println!("Building NOCI-QMC basis....");

            let include_refs = true;
            let basis = time_call!(crate::timers::deterministic::add_generate_excited_basis, {
                generate_excited_basis(post.noci_reference_basis, input, include_refs)
            });

            let n = basis.len();
            println!("Built NOCI-QMC basis of {} determinants.", n);
            println!(
                "Calculating NOCI-QMC deterministic propagation matrix elements for {} determinants ({} elements)...",
                n,
                n * n
            );

            let symmetric = true;
            let data =
                NOCIData::new(post.ao, &basis, input, post.tol, wicks).withmocache(post.mocache);
            let (h, s, _) = time_call!(crate::timers::deterministic::add_build_noci_hs, {
                build_noci_hs(&data, &basis, &basis, symmetric)
            });
            println!("Finished calculating NOCI-QMC matrix elements.");

            let es = post.states[0].e;
            println!("Running deterministic NOCI-QMC propagation....");

            let mut c0qmc = Array1::<f64>::zeros(n);
            if !input.write.write_deterministic_coeffs {
                for (i, ref_st) in post.noci_reference_basis.iter().enumerate() {
                    let idx = basis
                        .iter()
                        .position(|qmc_st| qmc_st.label == ref_st.label)
                        .unwrap();
                    c0qmc[idx] = c0[i];
                }
            } else {
                c0qmc = Array1::from_elem(n, 1.0 / (n as f64).sqrt());
            };

            println!("Initial wavefunction ansatz (C0-QMC): {}", c0qmc);

            let ref_indices: Vec<usize> = post
                .noci_reference_basis
                .iter()
                .map(|ref_st| {
                    basis
                        .iter()
                        .position(|qmc_st| qmc_st.label == ref_st.label)
                        .unwrap()
                })
                .collect();

            let mut coefficients = Vec::new();

            let c = time_call!(crate::timers::deterministic::add_propagate, {
                propagate(&h, &s, &c0qmc, es, &mut coefficients, input)
            });

            let cfinal = match c {
                Some(c) => c,
                None => {
                    println!("Propagation failed.");
                    std::process::exit(1);
                }
            };

            let e = projected_energy(&h, &s, &cfinal);
            wavefunction_sparsity(cfinal.as_slice().unwrap(), &ref_indices);

            if input.write.write_deterministic_coeffs {
                println!("Writing coefficients to file...");
                let filepath = format!("{}/{}", input.write.write_dir, "coefficients");
                let file = File::create(filepath).unwrap();
                let mut writer = BufWriter::new(file);
                for iter in &coefficients {
                    writeln!(writer, "iter {}", iter.iter).unwrap();
                    writeln!(writer, "Full coefficients:").unwrap();
                    for (i, z) in iter.c_full.iter().enumerate() {
                        writeln!(writer, "{:4} {:.8e}", i, z).unwrap();
                    }
                    writeln!(writer, "Relevant space coefficients:").unwrap();
                    for (i, z) in iter.c_relevant.iter().enumerate() {
                        writeln!(writer, "{:4} {:.8e}", i, z).unwrap();
                    }
                    writeln!(writer, "Null space coefficients:").unwrap();
                    for (i, z) in iter.c_null.iter().enumerate() {
                        writeln!(writer, "{:4} {:.8e}", i, z).unwrap();
                    }
                    writeln!(writer).unwrap();
                }
            }

            e
        }
    )
}
