// driver/post.rs

use mpi::topology::Communicator;
#[cfg(feature = "nocc")]
use ndarray::Array1;
use num_complex::Complex64;

use crate::driver::deterministic::run_qmc_deterministic_noci;
use crate::driver::reference::ReferenceRun;
use crate::driver::snoci::run_snoci;
use crate::driver::stochastic::run_qmc_stochastic_noci;
use crate::input::Input;
#[cfg(feature = "nocc")]
use crate::nocc::run_noccmc;
#[cfg(feature = "nocc")]
use crate::noci::NOCIData;
use crate::noci::{build_mo_cache, build_wicks_shared};
use crate::nonorthogonalwicks::WicksShared;
#[cfg(feature = "nocc")]
use crate::orbitals::noci_natural_orbitals;
use crate::{AoData, HSCFState, PostSCFData, SCFState};

/// Results from optional post-reference calculations.
pub struct PostReferenceResults {
    /// Deterministic NOCI-QMC energy if calculated.
    pub e_noci_qmc_det: Option<f64>,
    /// Stochastic NOCI-QMC energy if calculated.
    pub e_noci_qmc_stoch: Option<f64>,
    /// Selected NOCI energy if calculated.
    pub e_snoci: Option<f64>,
    /// NOCI-PT2 energy if calculated.
    pub e_pt2: Option<Vec<(f64, f64)>>,
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
            e_snoci: None,
            e_pt2: None,
        }
    }
}

/// Run optional real-reference post-reference calculations.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `states`: Real SCF states generated for this geometry.
/// - `reference`: Reference-space intermediates and solution.
/// - `input`: User input specifications.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `world`: MPI communicator object.
/// # Returns:
/// - `PostReferenceResults`: Optional post-reference energies.
pub fn run_real_post_reference(
    ao: &AoData,
    states: &[SCFState],
    reference: &mut ReferenceRun<f64>,
    input: &mut Input,
    tol: f64,
    world: &impl Communicator,
) -> PostReferenceResults {
    let mut out = PostReferenceResults::empty();
    let post = PostSCFData {
        ao,
        states,
        noci_reference_basis: &reference.basis,
        mocache: &reference.mocache,
        tol,
    };

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

    #[cfg(feature = "nocc")]
    if input.noccmc.is_some() {
        let no = {
            let wicks = reference.wicks.as_ref().map(|ws| ws.view());
            let data = NOCIData::new(post.ao, post.noci_reference_basis, input, post.tol, wicks)
                .withmocache(post.mocache);
            let coeffs = Array1::from_vec(reference.c0.clone());

            noci_natural_orbitals(&data, &coeffs, 1e-6, 1e-6)
        };

        reference.wicks = None;
        run_noccmc(&post, input, &reference.c0, &no, world);
    }

    if let Some(snoci) = input.snoci.as_ref() {
        if snoci.imag_shifts.iter().all(|&x| x == 0.0) {
            let (e_snoci, e_pt2) = run_snoci(
                &post,
                &reference.basis,
                input,
                reference.wicks.as_mut(),
                world,
            );
            if world.rank() == 0 {
                out.e_snoci = Some(e_snoci);
                out.e_pt2 = Some(e_pt2);
            }
        } else {
            let (e_snoci, e_pt2) = run_imag_shifted_complex_snoci(ao, reference, input, tol, world);

            if world.rank() == 0 {
                out.e_snoci = Some(e_snoci);
                out.e_pt2 = Some(e_pt2);
            }
        }
    }

    if input.qmc.is_some() {
        let wicks = reference.wicks.as_ref().map(|ws| ws.view());
        out.e_noci_qmc_stoch = Some(run_qmc_stochastic_noci(
            &post,
            input,
            &reference.c0,
            world,
            wicks,
        ));
    }

    out
}

/// Run imaginary-shifted SNOCI/NOCI-PT2 in complex arithmetic.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `reference`: Real reference-space intermediates and solution.
/// - `input`: User input specifications.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `world`: MPI communicator object.
/// # Returns:
/// - `(f64, Vec<(f64, f64)>)`: Current SNOCI energy and NOCI-PT2 corrections.
fn run_imag_shifted_complex_snoci(
    ao: &AoData,
    reference: &ReferenceRun<f64>,
    input: &Input,
    tol: f64,
    world: &impl Communicator,
) -> (f64, Vec<(f64, f64)>) {
    if world.rank() == 0 {
        println!("Running imaginary-shifted SNOCI/NOCI-PT2 in complex arithmetic....");
    }

    let complex_reference_basis: Vec<HSCFState> =
        reference.basis.iter().map(HSCFState::from_real).collect();

    let mut complex_wicks_shared: Option<WicksShared<Complex64>> =
        if input.wicks.enabled || input.wicks.compare {
            Some(build_wicks_shared(
                world,
                ao,
                &complex_reference_basis,
                tol,
                input,
            ))
        } else {
            None
        };

    let complex_mocache = build_mo_cache(ao, &complex_reference_basis, tol);
    let complex_post = PostSCFData {
        ao,
        states: &complex_reference_basis,
        noci_reference_basis: &complex_reference_basis,
        mocache: &complex_mocache,
        tol,
    };

    run_snoci(
        &complex_post,
        &complex_reference_basis,
        input,
        complex_wicks_shared.as_mut(),
        world,
    )
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
    let mut out = PostReferenceResults::empty();

    if world.rank() == 0 && warn_qmc && input.qmc.is_some() {
        println!(
            "Holomorphic NOCI states not currently supported for stochastic propagation. Continuing with reference NOCI/deterministic NOCI-QMC/SNOCI/NOCI-PT2."
        );
    }

    let post = PostSCFData {
        ao,
        states: &reference.basis,
        noci_reference_basis: &reference.basis,
        mocache: &reference.mocache,
        tol,
    };

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

    if input.snoci.is_some() {
        let (e_snoci, e_pt2) = run_snoci(
            &post,
            &reference.basis,
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
