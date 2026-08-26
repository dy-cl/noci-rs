// driver/snoci.rs

// External crate imports.
use mpi::topology::Communicator;
use num_complex::Complex64;

// Crate-root imports.
use crate::noci::NOCIScalar;
use crate::nonorthogonalwicks::WicksShared;
use crate::snoci::snoci_step;
use crate::time_call;
use crate::{DetState, PostSCFData};

/// Run SNOCI and solve the projected NOCI-PT2 equation
/// `M^Omega(epsilon) a(epsilon) = -V^Omega`.
/// Chemistry quantities are represented by `T`; Krylov vectors and shifted linear algebra are
/// represented by `R`.
/// # Arguments:
/// - `post`: Data shared by post-SCF methods.
/// - `initial_space`: Current determinant space used as the starting point for SNOCI.
/// - `input`: User input specifications.
/// - `wicks`: Optional shared-memory Wick's intermediates storage.
/// - `world`: MPI communicator object.
/// # Returns:
/// - `(f64, Vec<(f64, f64)>)`: Current SNOCI energy and NOCI-PT2 corrections stored as `(imag_shift, ept2)`.
pub fn run_snoci<T, R>(
    post: &PostSCFData<'_, T>,
    initial_space: &[DetState<T>],
    input: &crate::input::Input,
    wicks: Option<&mut WicksShared<T>>,
    world: &impl Communicator,
) -> (f64, Vec<(f64, f64)>)
where
    T: NOCIScalar + Into<Complex64>,
    R: NOCIScalar + From<T> + Into<Complex64>,
{
    time_call!(crate::timers::snoci::add_run_snoci, {
        let current_space = initial_space.to_vec();

        if world.rank() == 0 {
            println!("{}", "=".repeat(100));
        }

        let state = snoci_step::<T, R>(post, &current_space, input, wicks, world);

        let ept2 = state
            .pt2
            .iter()
            .map(|r| (r.imag_shift, r.ept2))
            .collect::<Vec<_>>();

        (state.ecurrent, ept2)
    })
}
