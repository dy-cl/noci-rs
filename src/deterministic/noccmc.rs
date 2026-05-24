// deterministic/noccmc.rs

use mpi::topology::Communicator;
use ndarray::Array1;

use crate::PostSCFData;
use crate::input::Input;
use crate::maths::general_evp;
use crate::noci::{NOCIData, build_noci_hs, build_spin_free_rdms_12, build_wicks_shared};
use crate::nonorthogonalwicks::WickScratchSpin;
use crate::orbitals::{
    NOCINaturalOrbitals, print_noci_natural_orbitals, transform_ao_data, transform_noci_basis,
};

/// Run NOCCMC setup work.
/// # Arguments:
/// - `post`: Data shared by post-SCF methods.
/// - `input`: User input specifications.
/// - `c0`: Reference NOCI coefficient vector.
/// - `no`: NOCI natural orbital basis.
/// - `world`: MPI communicator object.
/// # Returns:
/// - `()`: Calls RDM construction so `wicks.compare` compares Wick and naive matrix elements.
pub(crate) fn run_noccmc(
    post: &PostSCFData<'_, f64>,
    input: &Input,
    c0: &[f64],
    no: &NOCINaturalOrbitals,
    world: &impl Communicator,
) {
    let coeffs = Array1::from_vec(c0.to_vec());

    if world.rank() == 0 {
        print_noci_natural_orbitals("NOCI natural orbitals", no);
    }

    // Transform the NOCI basis using the natural NOCI orbitals.
    let nobasis = transform_noci_basis(post.noci_reference_basis, &no.c, &post.ao.s);
    // Transform the AO data using the natural NOCI orbitals.
    let noao = transform_ao_data(post.ao, &no.c);

    let nowicks = if input.wicks.enabled {
        Some(build_wicks_shared(world, &noao, &nobasis, post.tol, input))
    } else {
        None
    };

    let nowicksview = nowicks.as_ref().map(|w| w.view());
    let nodata = NOCIData::new(&noao, &nobasis, input, post.tol, nowicksview);

    let mut scratch = WickScratchSpin::new();
    let scratch = if input.wicks.enabled {
        Some(&mut scratch)
    } else {
        None
    };

    if world.rank() == 0 {
        println!("{}", "=".repeat(100));
        println!("Running NOCCMC spin-free RDM check in NOCI natural orbital basis....");
    }

    let (_, gamma1, _) = build_spin_free_rdms_12(&nodata, &coeffs, &coeffs, scratch);

    if world.rank() == 0 {
        let mut scheck = no.c.t().dot(&post.ao.s).dot(&no.c);
        for i in 0..scheck.nrows() {
            scheck[(i, i)] -= 1.0;
        }
        let serr = scheck.iter().map(|x| x.abs()).fold(0.0, f64::max);
        println!("NOCI natural orbital orthonormality error: {:.6e}", serr);

        let (h, s, _) = build_noci_hs(&nodata, nodata.basis, nodata.basis, true);
        let e_coeff = coeffs.dot(&h.dot(&coeffs)) / coeffs.dot(&s.dot(&coeffs));
        let (evals, _) = general_evp(&h, &s, true, post.tol);

        println!("NOCI energy in NO basis: {:.12}", e_coeff);
        println!("Lowest NOCI GEVP energy in NO basis: {:.12}", evals[0]);

        println!("Trace Gamma1(NO): {:.10}", gamma1.diag().sum());
    }
}
