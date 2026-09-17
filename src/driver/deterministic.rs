// driver/deterministic.rs

// Standard library imports.
use std::fs::File;
use std::io::{BufWriter, Write};

// External crate imports.
use ndarray::Array1;

// Crate-root imports.
use crate::PostSCFData;
use crate::deterministic::{projected_energy, propagate};
use crate::input::Input;
use crate::noci::{NOCIData, NOCIIndex, NOCIScalar, build_noci_hs};
use crate::nonorthogonalwicks::WicksView;
use crate::time_call;
use crate::utils::wavefunction_sparsity;

/// Perform the deterministic propagation in the NOCI-QMC space.
/// # Arguments:
/// - `post`: Data shared by post-SCF methods.
/// - `input`: User input specifications.
/// - `c0`: Initial coefficient vector of basis states.
/// - `e_shift0`: Initial propagation energy shift.
/// - `wicks`: Optional precomputed Wick's intermediates.
/// # Returns:
/// - `f64`: Propagated energy.
pub fn run_qmc_deterministic_noci<T: NOCIScalar>(
    post: &PostSCFData<'_, T>,
    input: &Input,
    c0: &[T],
    e_shift0: f64,
    wicks: Option<&WicksView<T>>,
) -> f64 {
    time_call!(
        crate::timers::deterministic::add_run_qmc_deterministic_noci,
        {
            // Expand the reference space into the deterministic NOCI-QMC excitation basis.
            println!("{}", "=".repeat(100));
            println!("Building NOCI-QMC basis....");

            let references = (0..post.space.len()).map(NOCIIndex).collect::<Vec<_>>();
            let basis = time_call!(crate::timers::deterministic::add_generate_excited_basis, {
                post.space.excited_from(&references, input, true)
            });

            let n = basis.len();
            println!("Built NOCI-QMC basis of {} determinants.", n);
            println!(
                "Calculating NOCI-QMC deterministic propagation matrix elements for {} determinants ({} elements)...",
                n,
                n * n
            );

            // Materialise symmetric Hamiltonian and overlap matrices in the expanded basis.
            let symmetric = true;
            let data =
                NOCIData::new(post.ao, &basis, input, post.tol, wicks).withmocache(post.mocache);
            let indices = (0..basis.len()).map(NOCIIndex).collect::<Vec<_>>();
            let (h, s, _) = time_call!(crate::timers::deterministic::add_build_noci_hs, {
                build_noci_hs(&data, &indices, &indices, symmetric)
            });
            println!("Finished calculating NOCI-QMC matrix elements.");

            println!("Running deterministic NOCI-QMC propagation....");

            // Embed reference coefficients, or use the requested uniform diagnostic ansatz.
            let mut c0qmc = Array1::<T>::zeros(n);
            if !input.write.write_deterministic_coeffs {
                for (i, ref_st) in post.space.labels.iter().enumerate() {
                    let idx = basis
                        .labels
                        .iter()
                        .position(|qmc_st| qmc_st == ref_st)
                        .unwrap();
                    c0qmc[idx] = c0[i];
                }
            } else {
                c0qmc = Array1::from_elem(n, T::from_real(1.0 / (n as f64).sqrt()));
            };

            println!("Initial wavefunction ansatz (C0-QMC): {}", c0qmc);
            println!("{}", "=".repeat(100));

            // Locate reference determinants in the expanded basis for population diagnostics.
            let ref_indices: Vec<usize> = post
                .space
                .labels
                .iter()
                .map(|ref_st| {
                    basis
                        .labels
                        .iter()
                        .position(|qmc_st| qmc_st == ref_st)
                        .unwrap()
                })
                .collect();

            let mut coefficients = Vec::new();

            // Propagate in the nonorthogonal metric while retaining optional iteration history.
            let c = time_call!(crate::timers::deterministic::add_propagate, {
                propagate(&h, &s, &c0qmc, e_shift0, &mut coefficients, input, &basis)
            });

            let cfinal = match c {
                Some(c) => c,
                None => {
                    println!("Propagation failed.");
                    std::process::exit(1);
                }
            };

            // Evaluate the final projected energy and wavefunction sparsity.
            let e = projected_energy(&h, &s, &cfinal);
            wavefunction_sparsity(cfinal.as_slice().unwrap(), &ref_indices);

            // Write retained/null canonical coefficients only when iteration output is requested.
            if input.write.write_deterministic_coeffs {
                println!("Writing coefficients to file...");
                let filepath = format!("{}/{}", input.write.write_dir, "coefficients");
                let file = File::create(filepath).unwrap();
                let mut writer = BufWriter::new(file);
                writeln!(writer, "iter,space,state,coeff").unwrap();
                for iter in &coefficients {
                    for (i, z) in iter.c_relevant.iter().enumerate() {
                        writeln!(writer, "{},relevant,{},{}", iter.iter, i, z.re()).unwrap();
                    }
                    for (i, z) in iter.c_null.iter().enumerate() {
                        writeln!(writer, "{},null,{},{}", iter.iter, i, z.re()).unwrap();
                    }
                }
            }

            e
        }
    )
}
