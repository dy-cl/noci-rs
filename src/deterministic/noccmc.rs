// deterministic/noccmc.rs

use mpi::topology::Communicator;
use ndarray::Array1;

use crate::PostSCFData;
use crate::input::Input;
use crate::maths::general_evp;
use crate::noci::{NOCIData, build_noci_hs, build_wicks_shared, rdm1, rdm2, rdm3, rdm4};
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
/// - `()`: Builds and checks spin-free RDMs in the NOCI natural orbital basis.
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

    if world.rank() == 0 {
        println!("{}", "=".repeat(100));
        println!("Running NOCCMC spin-free RDM check in NOCI natural orbital basis....");
    }

    let mut scratch1 = WickScratchSpin::new();
    let scratch1 = if input.wicks.enabled {
        Some(&mut scratch1)
    } else {
        None
    };
    let (_, gamma1) = rdm1(&nodata, &coeffs, &coeffs, scratch1);

    let mut scratch2 = WickScratchSpin::new();
    let scratch2 = if input.wicks.enabled {
        Some(&mut scratch2)
    } else {
        None
    };
    let (_, gamma2) = rdm2(&nodata, &coeffs, &coeffs, scratch2);

    let mut scratch3 = WickScratchSpin::new();
    let scratch3 = if input.wicks.enabled {
        Some(&mut scratch3)
    } else {
        None
    };
    let (_, gamma3) = rdm3(&nodata, &coeffs, &coeffs, &no.active, scratch3);

    let mut scratch4 = WickScratchSpin::new();
    let scratch4 = if input.wicks.enabled {
        Some(&mut scratch4)
    } else {
        None
    };
    let (_, gamma4) = rdm4(&nodata, &coeffs, &coeffs, &no.active, scratch4);
    let _active_rank_sizes = (gamma3.n, gamma4.n);

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

        let mut e1 = 0.0;
        for a in 0..gamma1.n {
            for b in 0..gamma1.n {
                let i = b * gamma1.n + a;
                e1 += noao.h[(a, b)] * gamma1.data[i];
            }
        }

        let mut e2 = 0.0;
        for a in 0..gamma2.n {
            for b in 0..gamma2.n {
                for c in 0..gamma2.n {
                    for d in 0..gamma2.n {
                        let i = (((b * gamma2.n + c) * gamma2.n + a) * gamma2.n) + d;
                        e2 += noao.eri_coul[(a, b, c, d)] * gamma2.data[i];
                    }
                }
            }
        }

        let erdm = noao.enuc + e1 + 0.5 * e2;

        let mut trace = 0.0;
        for p in 0..gamma1.n {
            trace += gamma1.data[p * gamma1.n + p];
        }

        let mut nact_elec = 0.0;
        for &p in no.active.iter() {
            nact_elec += gamma1.data[p * gamma1.n + p];
        }

        println!("NOCI energy in NO basis: {:.12}", e_coeff);
        println!("Lowest NOCI GEVP energy in NO basis: {:.12}", evals[0]);
        println!("NOCI energy from RDMs in NO basis: {:.12}", erdm);
        println!("Trace Gamma1(NO): {:.10}", trace);
        println!("Active electron count from Gamma1(NO): {:.10}", nact_elec);
        println!(
            "Max active Gamma3(NO): {:.6e}",
            gamma3.data.iter().map(|x| x.abs()).fold(0.0, f64::max)
        );
        println!(
            "Max active Gamma4(NO): {:.6e}",
            gamma4.data.iter().map(|x| x.abs()).fold(0.0, f64::max)
        );
    }
}
