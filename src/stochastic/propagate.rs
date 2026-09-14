// stochastic/propagate.rs
// External crate imports.
use mpi::topology::Communicator;

// Crate-root imports.
use crate::input::Propagator;
use crate::noci::NOCIData;

// Parent/sibling imports.
use super::state::ExcitationHist;

/// Propagate according to the stochastic update equations selected by `prop.propagator`.
/// # Arguments:
/// - `data`: Immutable stochastic propagation data.
/// - `c0`: Initial determinant coefficient vector.
/// - `es`: Population-control shift energy.
/// - `ref_indices`: Indices of the reference determinants in the stochastic basis.
/// - `world`: MPI communicator.
/// # Returns:
/// - `(f64, Option<ExcitationHist>)`: Final projected energy estimate and optional
///   spawning-magnitude histogram.
pub fn qmc_step(
    data: &NOCIData<'_, f64>,
    c0: &[f64],
    es: &mut f64,
    ref_indices: &[usize],
    world: &impl Communicator,
) -> (f64, Option<ExcitationHist>) {
    match data.input.prop_ref().propagator {
        Propagator::Unshifted
        | Propagator::Shifted
        | Propagator::DoublyShifted
        | Propagator::DifferenceDoublyShiftedU1
        | Propagator::DifferenceDoublyShiftedU2 => {
            super::walkers::qmc_step(data, c0, es, ref_indices, world)
        }
        Propagator::SApply => super::sapply::qmc_step(data, c0, es, ref_indices, world),
        Propagator::BApply => super::bapply::qmc_step(data, c0, es, ref_indices, world),
    }
}
