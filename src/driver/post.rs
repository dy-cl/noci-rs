// driver/post.rs

// External crate imports.
use mpi::topology::Communicator;
use num_complex::Complex64;

// Crate-root imports.
#[cfg(feature = "gnocc")]
use crate::driver::nocc::run_gnocc;
use crate::driver::nocc::{run_jmnoccmc, run_jmnoccmc_stochastic};
use crate::driver::noci::{run_qmc_deterministic_noci, run_qmc_stochastic_noci, run_snoci};
use crate::driver::reference::ReferenceRun;
use crate::input::Input;
#[cfg(feature = "gnocc")]
use crate::nocc::AmplitudeGuess;
use crate::{AoData, PostSCFData};

/// Post-reference solutions of one geometry carried to the next as starting guesses.
#[derive(Clone, Default)]
pub struct PostGuess {
    /// Converged GNOCC amplitudes and their natural orbitals.
    #[cfg(feature = "gnocc")]
    pub nocc: Option<AmplitudeGuess>,
}

/// Results from optional post-reference calculations.
pub struct PostReferenceResults {
    /// Deterministic NOCI-QMC energy if calculated.
    pub e_noci_qmc_det: Option<f64>,
    /// Stochastic NOCI-QMC energy if calculated.
    pub e_noci_qmc_stoch: Option<f64>,
    /// Deterministic JM-NOCCMC energy if calculated.
    pub e_jmnoccmc_det: Option<f64>,
    /// Stochastic JM-NOCCMC energy if calculated.
    pub e_jmnoccmc_stoch: Option<f64>,
    /// Selected NOCI energy if calculated.
    pub e_snoci: Option<f64>,
    /// NOCI-PT2 energy if calculated.
    pub e_pt2: Option<Vec<(f64, f64)>>,
    /// Solutions carried to the next geometry.
    pub guess: PostGuess,
}

impl PostReferenceResults {
    /// Construct empty post-reference results.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `PostReferenceResults`: Empty optional result fields.
    fn empty() -> Self {
        Self {
            e_noci_qmc_det: None,
            e_noci_qmc_stoch: None,
            e_jmnoccmc_det: None,
            e_jmnoccmc_stoch: None,
            e_snoci: None,
            e_pt2: None,
            guess: PostGuess::default(),
        }
    }
}

/// Run optional real-reference post-reference calculations.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `reference`: Reference-space intermediates and solution.
/// - `input`: User input specifications.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `previous`: Post-reference solutions of the previous geometry.
/// - `world`: MPI communicator object.
/// # Returns:
/// - `PostReferenceResults`: Optional post-reference energies.
pub fn run_real_post_reference(
    ao: &AoData,
    reference: &mut ReferenceRun<f64>,
    input: &mut Input,
    tol: f64,
    #[cfg_attr(not(feature = "gnocc"), allow(unused_variables))] previous: &PostGuess,
    world: &impl Communicator,
) -> PostReferenceResults {
    // Assemble the immutable post-SCF view shared by all optional methods.
    let mut out = PostReferenceResults::empty();
    let post = PostSCFData {
        ao,
        space: &reference.space,
        mocache: &reference.mocache,
        tol,
    };

    // Deterministic NOCI-QMC is replicated only on the root rank.
    if world.rank() == 0 && input.det.is_some() {
        let wicks = reference.wicks.as_ref().map(|ws| ws.view());
        out.e_noci_qmc_det = Some(run_qmc_deterministic_noci(
            &post,
            input,
            &reference.c0,
            reference.e_noci,
            wicks,
        ));
    }

    // Deterministic JM-NOCCMC is replicated only on the root rank; the stochastic propagation,
    // selected by a nested qmc table, uses every rank.
    if let Some(options) = input.jmnoccmc.as_ref() {
        let wicks = reference.wicks.as_ref().map(|ws| ws.view());
        if options.qmc.is_some() {
            let e = run_jmnoccmc_stochastic(
                &post,
                input,
                (&reference.c0, reference.e_noci),
                wicks,
                world,
            );
            if world.rank() == 0 {
                out.e_jmnoccmc_stoch = Some(e);
            }
        } else if world.rank() == 0 {
            out.e_jmnoccmc_det = run_jmnoccmc(&post, input, wicks);
        }
    }

    #[cfg(feature = "gnocc")]
    if input.noccmc.is_some() {
        out.guess.nocc = run_gnocc(
            &post,
            input,
            &reference.c0,
            &mut reference.wicks,
            previous.nocc.as_ref(),
            world,
        );
    }

    // Select real or complex PT2 arithmetic according to the requested imaginary shifts.
    if input.snoci.is_some() {
        let snoci = input.snoci.as_ref().unwrap();

        let (e_snoci, e_pt2) = if snoci.imag_shifts.iter().any(|&x| x != 0.0) {
            run_snoci::<f64, Complex64>(
                &post,
                &reference.space,
                input,
                reference.wicks.as_mut(),
                world,
            )
        } else {
            run_snoci::<f64, f64>(
                &post,
                &reference.space,
                input,
                reference.wicks.as_mut(),
                world,
            )
        };

        // Only root publishes the replicated selected-NOCI results.
        if world.rank() == 0 {
            out.e_snoci = Some(e_snoci);
            out.e_pt2 = Some(e_pt2);
        }
    }

    // Stochastic NOCI-QMC uses all MPI ranks and returns its global estimate.
    if input.qmc.is_some() {
        let wicks = reference.wicks.as_ref().map(|ws| ws.view());
        out.e_noci_qmc_stoch = Some(run_qmc_stochastic_noci(
            &post,
            input,
            &reference.c0,
            reference.e_noci,
            world,
            wicks,
        ));
    }

    out
}

/// Run optional holomorphic-reference post-reference calculations.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `reference`: Reference-space intermediates and solution.
/// - `input`: User input specifications.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `warn_qmc`: Whether to print holomorphic QMC unsupported warning.
/// - `world`: MPI communicator object.
/// # Returns:
/// - `PostReferenceResults`: Optional post-reference energies.
pub fn run_holomorphic_post_reference(
    ao: &AoData,
    reference: &mut ReferenceRun<Complex64>,
    input: &Input,
    tol: f64,
    warn_qmc: bool,
    world: &impl Communicator,
) -> PostReferenceResults {
    // Holomorphic references currently support deterministic and selected post-reference paths.
    let mut out = PostReferenceResults::empty();

    if world.rank() == 0 && warn_qmc && input.qmc.is_some() {
        println!(
            "Holomorphic NOCI states not currently supported for stochastic propagation. Continuing with reference NOCI/deterministic NOCI-QMC/SNOCI/NOCI-PT2."
        );
    }

    // Assemble the complex post-SCF view after issuing any unsupported-QMC warning.
    let post = PostSCFData {
        ao,
        space: &reference.space,
        mocache: &reference.mocache,
        tol,
    };

    // Run deterministic NOCI-QMC only on root, as for real references.
    if world.rank() == 0 && input.det.is_some() {
        let wicks = reference.wicks.as_ref().map(|ws| ws.view());
        out.e_noci_qmc_det = Some(run_qmc_deterministic_noci(
            &post,
            input,
            &reference.c0,
            reference.e_noci,
            wicks,
        ));
    }

    // Complex references and imaginary shifts require complex selected-NOCI arithmetic.
    if input.snoci.is_some() {
        let (e_snoci, e_pt2) = run_snoci::<Complex64, Complex64>(
            &post,
            &reference.space,
            input,
            reference.wicks.as_mut(),
            world,
        );

        if world.rank() == 0 {
            out.e_snoci = Some(e_snoci);
            out.e_pt2 = Some(e_pt2);
        }
    }

    out
}
