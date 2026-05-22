// scf/h/finalise.rs

use std::sync::Arc;

use ndarray::Array2;
use num_complex::Complex64;

use crate::input::Input;
use crate::scf::DensityMode;
use crate::utils::print_array2_indexed;
use crate::{AoData, Excitation, HSCFState};

use super::types::HSCFRunData;
use crate::scf::{density, energy, fock};

/// Construct final h-SCF state from optimised complex orbitals.
/// # Arguments:
/// - `ca`: Final alpha-spin MO coefficients.
/// - `cb`: Final beta-spin MO coefficients.
/// - `ao`: Contains AO integrals and metadata.
/// - `input`: User input specifications.
/// - `run`: Immutable data for this h-SCF optimisation.
/// # Returns:
/// - `HSCFState`: Final h-SCF determinant state.
pub(crate) fn finalise(
    ca: Array2<Complex64>,
    cb: Array2<Complex64>,
    ao: &AoData,
    input: &Input,
    run: HSCFRunData<'_>,
) -> HSCFState {
    let na = usize::try_from(ao.nelec[0]).unwrap();
    let nb = usize::try_from(ao.nelec[1]).unwrap();

    let da = density(&ca, na, DensityMode::Holomorphic);
    let db = density(&cb, nb, DensityMode::Holomorphic);

    let (fa, fb) = fock(&ao.h, &ao.eri_coul, &da, &db);
    let e = energy(&ao.h, ao.enuc, &da, &db, &fa, &fb);

    if input.write.verbose {
        println!("{}", "-".repeat(100));
        println!("Complex coefficients ca:");
        print_array2_indexed(&ca);
        println!("Complex coefficients cb:");
        print_array2_indexed(&cb);
    }

    if input.write.write_orbitals {
        println!("Complex h-SCF orbital HDF5 writing is not implemented yet.");
    }

    // Occupy the first `na` and `nb` orbitals because h-SCF keeps occupied orbitals first throughout.
    let oa = (0..na).fold(0u128, |bits, j| bits | (1u128 << j));
    let ob = (0..nb).fold(0u128, |bits, j| bits | (1u128 << j));

    HSCFState {
        e,
        oa,
        ob,
        pha: 1.0,
        phb: 1.0,
        ca: Arc::new(ca),
        cb: Arc::new(cb),
        da: Arc::new(da),
        db: Arc::new(db),
        label: run.label.to_string(),
        noci_basis: run.noci_basis,
        parent: run.parent,
        excitation: Excitation::empty(),
    }
}
