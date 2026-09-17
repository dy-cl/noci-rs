// driver/stochastic.rs

// Standard library imports.
use std::fs::File;
use std::io::{BufWriter, Write};

// External crate imports.
use mpi::topology::Communicator;

// Crate-root imports.
use crate::PostSCFData;
use crate::input::Input;
use crate::noci::{NOCIData, NOCIIndex};
use crate::nonorthogonalwicks::WicksView;
use crate::stochastic::qmc_step;
use crate::time_call;

/// Perform stochastic propagation in the NOCI-QMC space.
/// # Arguments:
/// - `post`: Data shared by post-SCF methods.
/// - `input`: User input specifications.
/// - `c0`: Initial coefficient vector of basis states.
/// - `world`: MPI communicator object.
/// - `wicks`: Optional precomputed Wick's intermediates.
/// # Returns:
/// - `f64`: Stochastic energy estimate.
pub fn run_qmc_stochastic_noci(
    post: &PostSCFData<'_, f64>,
    input: &mut Input,
    c0: &[f64],
    world: &impl Communicator,
    wicks: Option<&WicksView<f64>>,
) -> f64 {
    time_call!(crate::timers::stochastic::add_run_qmc_stochastic_noci, {
        // Identify root for user-facing output while all ranks build identical basis metadata.
        let irank = world.rank();

        if irank == 0 {
            println!("{}", "=".repeat(100));
            println!("Building NOCI-QMC basis....");
        }

        // Expand reference determinants into the stochastic NOCI-QMC basis.
        let references = (0..post.space.len()).map(NOCIIndex).collect::<Vec<_>>();
        let basis = time_call!(crate::timers::stochastic::add_generate_excited_basis, {
            post.space.excited_from(&references, input, true)
        });
        let n = basis.len();

        // Locate reference states and embed their coefficients in the expanded basis.
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

        let mut c0qmc = vec![0.0_f64; n];
        for (i, ref_st) in post.space.labels.iter().enumerate() {
            let idx = basis
                .labels
                .iter()
                .position(|qmc_st| qmc_st == ref_st)
                .unwrap();
            c0qmc[idx] = c0[i];
        }

        if irank == 0 {
            println!("Built NOCI-QMC basis of {} determinants.", n);
            println!("Running stochastic NOCI-QMC propagation....");
        }

        // Construct matrix-element data and run distributed stochastic propagation.
        let mut es = basis.parents[0].e;
        let data = NOCIData::new(post.ao, &basis, input, post.tol, wicks).withmocache(post.mocache);
        let (e, local_hist) = time_call!(crate::timers::stochastic::add_qmc_step, {
            qmc_step(&data, &c0qmc, &mut es, &ref_indices, world)
        });

        // Write rank-local spawning histograms when diagnostic sampling is enabled.
        if let Some(hist) = local_hist.as_ref()
            && input.write.write_excitation_hist
        {
            if irank == 0 {
                println!("Writing excitation samples to file...");
            }
            let filepath = format!("{}/excitationsamples{}", input.write.write_dir, irank);
            let file = File::create(filepath).unwrap();
            let mut writer = BufWriter::new(file);

            writeln!(writer, "{} {} {}", hist.logmin, hist.logmax, hist.nbins).unwrap();
            writeln!(
                writer,
                "{} {} {}",
                hist.ntotal, hist.noverflow_low, hist.noverflow_high
            )
            .unwrap();

            for &c in &hist.counts {
                writeln!(writer, "{}", c).unwrap();
            }
        }

        e
    })
}
