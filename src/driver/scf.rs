// driver/scf.rs

// External crate imports.
use num_complex::Complex64;

// Crate-root imports.
use crate::basis::{generate_reference_noci_basis, hermitian_hnoci_basis};
use crate::input::Input;
use crate::time_call;
use crate::{AoData, DetState, HSCFState, SCFState};

/// Prepared real references for downstream reference-space NOCI.
pub struct RealReferencePrep {
    /// SCF states generated for this geometry.
    pub states: Vec<SCFState>,
    /// Real reference basis before reference-space filtering.
    pub basis: Vec<DetState<f64>>,
}

/// Prepared holomorphic references for downstream reference-space NOCI.
pub struct HolomorphicReferencePrep {
    /// Real SCF states generated for this geometry.
    pub states: Vec<SCFState>,
    /// Complex h-SCF states generated for this geometry.
    pub hstates: Vec<HSCFState>,
    /// Off axis complex h-SCF states for tracking across geometries.
    pub htracks: Vec<HSCFState>,
    /// Complex Hermitian reference basis before reference-space filtering.
    pub basis: Vec<DetState<Complex64>>,
}

/// Run SCF calculations for real reference states.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `input`: User input specifications.
/// - `prev_states`: Converged SCF states at previous r, used for seeding.
/// # Returns:
/// - `RealReferencePrep`: Generated real states and reference basis.
pub fn generate_real_references(
    ao: &AoData,
    input: &mut Input,
    prev_states: &[SCFState],
) -> RealReferencePrep {
    let states = time_call!(crate::timers::general::add_run_scf, {
        if prev_states.is_empty() {
            generate_reference_noci_basis(ao, input, None, None).states
        } else {
            generate_reference_noci_basis(ao, input, Some(prev_states), None).states
        }
    });

    RealReferencePrep {
        basis: states.clone(),
        states,
    }
}

/// Run SCF calculations for holomorphic reference states.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `input`: User input specifications.
/// - `prev_states`: Converged SCF states at previous r, used for seeding.
/// - `prev_htracks`: Converged h-SCF states at previous r, used for complex branch tracking.
/// # Returns:
/// - `HolomorphicReferencePrep`: Generated real, h-SCF, and Hermitian h-NOCI references.
pub fn generate_holomorphic_references(
    ao: &AoData,
    input: &mut Input,
    prev_states: &[SCFState],
    prev_htracks: &[HSCFState],
) -> HolomorphicReferencePrep {
    let refs = time_call!(crate::timers::general::add_run_scf, {
        if prev_states.is_empty() && prev_htracks.is_empty() {
            generate_reference_noci_basis(ao, input, None, None)
        } else {
            let prev_htracks = if prev_htracks.is_empty() {
                None
            } else {
                Some(prev_htracks)
            };
            generate_reference_noci_basis(ao, input, Some(prev_states), prev_htracks)
        }
    });
    let basis = hermitian_hnoci_basis(&refs.hstates, &ao.s);

    HolomorphicReferencePrep {
        states: refs.states,
        hstates: refs.hstates,
        htracks: refs.htracks,
        basis,
    }
}
