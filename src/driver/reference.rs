use mpi::collective::CommunicatorCollectives;
use mpi::topology::Communicator;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::input::Input;
use crate::mpiutils::broadcast;
use crate::noci::{MOCache, NOCIScalar, build_mo_cache, build_wicks_shared, calculate_noci_energy};
use crate::nonorthogonalwicks::{WicksShared, WicksView};
use crate::time_call;
use crate::{AoData, DetState};

/// Reference-space NOCI intermediates owned for post-reference work.
pub struct ReferenceRun<T: NOCIScalar> {
    /// Reference determinant basis after NOCI filtering.
    pub basis: Vec<DetState<T>>,
    /// Reference-space NOCI energy.
    pub e_noci: f64,
    /// Reference-space NOCI coefficients.
    pub c0: Vec<T>,
    /// MO-basis one and two-electron integral caches.
    pub mocache: Vec<MOCache<T>>,
    /// Optional shared Wick's theorem intermediates.
    pub wicks: Option<WicksShared<T>>,
}

/// Reference-space arithmetic and print labels.
#[derive(Clone, Copy, Debug)]
pub enum ReferenceKind {
    /// Real-valued reference-space calculation.
    Real,
    /// Complex-valued reference-space calculation.
    Complex,
}

impl ReferenceKind {
    /// Return the label used for Wick's intermediate construction.
    /// # Arguments:
    /// - `self`: Reference-space arithmetic kind.
    /// # Returns:
    /// - `&'static str`: Wick's construction label.
    fn wicks_label(self) -> &'static str {
        match self {
            Self::Real => "Wick's",
            Self::Complex => "complex Wick's",
        }
    }

    /// Return the label used for MO-cache construction.
    /// # Arguments:
    /// - `self`: Reference-space arithmetic kind.
    /// # Returns:
    /// - `&'static str`: MO-cache construction label.
    fn mo_label(self) -> &'static str {
        match self {
            Self::Real => "reference NOCI",
            Self::Complex => "complex reference NOCI",
        }
    }
}

/// Run shared reference-space NOCI work for real or holomorphic references.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `input`: User input specifications.
/// - `basis`: Initial reference basis.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `kind`: Reference-space arithmetic and print labels.
/// - `world`: MPI communicator object.
/// # Returns:
/// - `ReferenceRun<T>`: Reference-space intermediates and solution.
pub fn run_reference_space<T>(
    ao: &AoData,
    input: &Input,
    basis: Vec<DetState<T>>,
    tol: f64,
    kind: ReferenceKind,
    world: &impl Communicator,
) -> ReferenceRun<T>
where
    T: NOCIScalar + Serialize + DeserializeOwned,
{
    let mut basis = filter_reference_basis(basis);

    world.barrier();
    broadcast(world, &mut basis);

    let wicks = build_reference_wicks(ao, input, &basis, tol, kind, world);

    if world.rank() == 0 {
        println!("Constructing {} MO basis....", kind.mo_label());
    }
    let mocache = build_mo_cache(ao, &basis, tol);

    let mut e_noci = 0.0;
    let mut c0 = Vec::new();
    if world.rank() == 0 {
        let wicks_view = wicks.as_ref().map(|ws| ws.view());
        let (e_ref, c0v) = solve_reference_noci(ao, input, &basis, tol, &mocache, wicks_view);
        e_noci = e_ref;
        c0 = c0v;
    }

    world.barrier();
    broadcast(world, &mut c0);
    broadcast(world, &mut e_noci);

    ReferenceRun {
        basis,
        e_noci,
        c0,
        mocache,
        wicks,
    }
}

/// Filter and reindex a reference determinant basis.
/// # Arguments:
/// - `states`: Candidate reference determinant states.
/// # Returns:
/// - `Vec<DetState<T>>`: Filtered and reindexed reference determinant basis.
fn filter_reference_basis<T>(states: Vec<DetState<T>>) -> Vec<DetState<T>>
where
    T: NOCIScalar,
{
    let mut basis: Vec<_> = states.into_iter().filter(|st| st.noci_basis).collect();

    for (i, st) in basis.iter_mut().enumerate() {
        st.parent = i;
    }

    basis
}

/// Build optional Wick's intermediates for reference-space work.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `input`: User input specifications.
/// - `basis`: Reference determinant basis.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `kind`: Reference-space arithmetic and print labels.
/// - `world`: MPI communicator object.
/// # Returns:
/// - `Option<WicksShared<T>>`: Optional shared Wick's storage.
fn build_reference_wicks<T: NOCIScalar>(
    ao: &AoData,
    input: &Input,
    basis: &[DetState<T>],
    tol: f64,
    kind: ReferenceKind,
    world: &impl Communicator,
) -> Option<WicksShared<T>> {
    time_call!(crate::timers::general::add_build_wicks_shared, {
        if input.wicks.enabled || input.wicks.compare {
            if world.rank() == 0 {
                println!("{}", "=".repeat(100));
                println!("Precomputing {} intermediates....", kind.wicks_label());
            }
            Some(build_wicks_shared(world, ao, basis, tol, input))
        } else {
            None
        }
    })
}

/// Solve reference NOCI for an already filtered basis.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `input`: User input specifications.
/// - `basis`: Filtered reference NOCI basis.
/// - `tol`: Tolerance up to which a number is considered zero.
/// - `mocache`: MO-basis one and two-electron integral caches.
/// - `wicks`: Optional precomputed Wick's intermediates.
/// # Returns:
/// - `(f64, Vec<T>)`: Reference NOCI energy and coefficients.
fn solve_reference_noci<T: NOCIScalar>(
    ao: &AoData,
    input: &Input,
    basis: &[DetState<T>],
    tol: f64,
    mocache: &[MOCache<T>],
    wicks: Option<&WicksView<T>>,
) -> (f64, Vec<T>) {
    time_call!(crate::timers::general::add_run_reference_noci, {
        let (e_noci, c0, _) = time_call!(crate::timers::general::add_calculate_noci_energy, {
            calculate_noci_energy(ao, input, basis, tol, mocache, wicks)
        });

        (e_noci, c0.to_vec())
    })
}
