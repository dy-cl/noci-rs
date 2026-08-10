// scf/h/track.rs

use num_complex::Complex64;

use crate::input::Input;
use crate::maths::complex_metric_orthonormalize;
use crate::{AoData, HSCFState, SCFState};

use super::optimise::hscf_cycle;
use super::seed::complex_orbitals_from_real;
use super::types::HSCFRunData;

/// Phase used to move two electron integrals into the complex plane.
const PHASE: f64 = std::f64::consts::PI / 20.0;
/// Number of steps used moving integrals to or from the complex plane.
const STEPS: usize = 5;

/// Initialise an off-axis h-SCF tracking state from a converged real HF solution.
/// The electron-electron interaction is rotated from the physical Hamiltonian at
/// `\lambda = 1` to the complex tracking Hamiltonian at `\lambda = \exp(i \theta_\mathrm{track})`
/// using a sequence of small phase steps.
/// # Arguments:
/// - `seed`: Converged real HF state used to initialise the holomorphic branch.
/// - `ao`: Contains AO integrals and metadata.
/// - `input`: Contains user specified input data.
/// - `label`: Label for the h-SCF state.
/// - `i`: Index of the h-SCF state.
/// # Returns:
/// - `Option<HSCFState>`: Converged off-axis h-SCF tracking state if continuation succeeds.
pub(crate) fn initialise_hscf_track(
    seed: &SCFState,
    ao: &AoData,
    input: &Input,
    label: &str,
    i: usize,
) -> Option<HSCFState> {
    // Promote orbitals from a converged real HF calculation to complex.
    // This corresponds to `lambda = 1` scaling of the two electron integrals.
    let (mut ca, mut cb) = complex_orbitals_from_real(seed, ao);

    let mut out = None;

    // Move the HF solution into the complex plane in incremenents of
    // `\lambda_k = exp(i \theta_k)`, which is given by the total phase
    // change divided by the number of steps in which to apply it.
    for step in 1..=STEPS {
        let theta = PHASE * step as f64 / STEPS as f64;
        // '\lambda = e^{i \theta}'.
        let lambda = Complex64::from_polar(1.0, theta);

        // Run holomorphic HF procedure and use the solution from
        // run '\lambda_{k - 1}' as initial coefficients.
        let state = hscf_cycle(
            &ca,
            &cb,
            ao,
            input,
            HSCFRunData {
                label,
                noci_basis: false,
                parent: i,
                lambda,
            },
        )?;

        // Update coefficients.
        ca = (*state.ca).clone();
        cb = (*state.cb).clone();
        out = Some(state);
    }

    out
}

/// Continue an off-axis h-SCF tracking state to the next molecular geometry.
/// The previous tracking orbitals are metric orthonormalised in the new AO basis
/// before optimisation at the fixed complex scaling `\lambda = \exp(i \theta_\mathrm{track})`.
/// # Arguments:
/// - `previous`: Off-axis h-SCF tracking state from the previous geometry.
/// - `ao`: Contains AO integrals and metadata for the current geometry.
/// - `input`: Contains user specified input data.
/// - `label`: Label for the h-SCF state.
/// - `i`: Index of the h-SCF state.
/// # Returns:
/// - `Option<HSCFState>`: Converged off-axis tracking state at the current geometry.
pub(crate) fn continue_hscf_track(
    previous: &HSCFState,
    ao: &AoData,
    input: &Input,
    label: &str,
    i: usize,
) -> Option<HSCFState> {
    // AO overlap changes between geometries so previous orbitals
    // must be re-orthonormalised with the new metric.
    let ca = complex_metric_orthonormalize(&previous.ca, &ao.s);
    let cb = complex_metric_orthonormalize(&previous.cb, &ao.s);

    // Same '\lambda' shift at every geometry.
    let lambda = Complex64::from_polar(1.0, PHASE);

    hscf_cycle(
        &ca,
        &cb,
        ao,
        input,
        HSCFRunData {
            label,
            noci_basis: false,
            parent: i,
            lambda,
        },
    )
}

/// Relax an off-axis h-SCF tracking state back to the physical Hamiltonian.
/// The complex electron-electron scaling is continuously reduced from
/// `\lambda = \exp(i \theta_\mathrm{track})` to `\lambda = 1`, producing the
/// physical h-SCF state associated with the tracked holomorphic branch.
/// # Arguments:
/// - `track`: Converged off-axis h-SCF tracking state.
/// - `ao`: Contains AO integrals and metadata.
/// - `input`: Contains user specified input data.
/// - `label`: Label for the h-SCF state.
/// - `i`: Index of the h-SCF state.
/// - `noci_basis`: Whether the final physical state should enter the NOCI basis.
/// # Returns:
/// - `Option<HSCFState>`: Physical `\lambda = 1` h-SCF state if continuation succeeds.
pub(crate) fn physical_hscf_state(
    track: &HSCFState,
    ao: &AoData,
    input: &Input,
    label: &str,
    i: usize,
    noci_basis: bool,
) -> Option<HSCFState> {
    // Start from the off-axis '\lambda' shifted solutions.
    let mut ca = (*track.ca).clone();
    let mut cb = (*track.cb).clone();

    let mut out = None;

    // Return to the physical Hamiltonian by reducing the '\lambda' phase
    // in incremenents of `\lambda_k = exp(i \theta_k)`, that is, the reverse
    // of the initialisation. Increment is again given by total phase to unapply
    // divided by number of steps in which to do so.
    for step in 1..=STEPS {
        let x = step as f64 / STEPS as f64;
        let theta = PHASE * (1.0 - x);

        // '\lambda = e^{i \theta}'.
        let lambda = Complex64::from_polar(1.0, theta);

        // Run holomorphic HF procedure and use the solution from
        // run '\lambda_{k - 1}' as initial coefficients.
        let state = hscf_cycle(
            &ca,
            &cb,
            ao,
            input,
            HSCFRunData {
                label,
                noci_basis: noci_basis && step == STEPS,
                parent: i,
                lambda,
            },
        )?;

        ca = (*state.ca).clone();
        cb = (*state.cb).clone();
        out = Some(state);
    }

    out
}
