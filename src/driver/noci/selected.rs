// driver/noci/selected.rs

// External crate imports.
use mpi::topology::Communicator;
use num_complex::Complex64;

// Crate-root imports.
use crate::NOCIScalar;
use crate::PostSCFData;
use crate::determinant::NOCISpace;
use crate::elements::nonorthogonalwicks::WicksShared;
use crate::noci::selected::{NOCIPT2Energy, snoci_step};
use crate::time_call;

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
/// - `(f64, Vec<NOCIPT2Energy>)`: Current SNOCI energy of the first corrected state and the total
///   NOCI-PT2 energy of every corrected state and imaginary shift.
pub fn run_snoci<T, R>(
    post: &PostSCFData<'_, T>,
    initial_space: &NOCISpace<T>,
    input: &crate::input::Input,
    wicks: Option<&mut WicksShared<T>>,
    world: &impl Communicator,
) -> (f64, Vec<NOCIPT2Energy>)
where
    T: NOCIScalar + Into<Complex64>,
    R: NOCIScalar + From<T> + Into<Complex64>,
{
    time_call!(crate::timers::selected::add_run_snoci, {
        if world.rank() == 0 {
            println!("{}", "=".repeat(100));
        }

        let state = snoci_step::<T, R>(post, initial_space, input, wicks, world);

        let e_pt2 = state
            .pt2
            .iter()
            .map(|r| NOCIPT2Energy {
                root: r.root,
                imag_shift: r.imag_shift,
                e: r.e_noci + r.ept2,
                s2: r.s2,
            })
            .collect::<Vec<_>>();

        (state.ecurrent, e_pt2)
    })
}
