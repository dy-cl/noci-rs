// scf/h/finalise.rs

// Standard library imports.
use std::sync::Arc;

// External crate imports.
use ndarray::Array2;
use num_complex::Complex64;

// Crate-root imports.
use crate::input::Input;
use crate::scf::{DensityMode, density, energy, fock_lambda};
use crate::utils::print_array2_indexed;
use crate::{AoData, Excitation, HSCFState};

// Parent/sibling imports.
use super::types::HSCFRunData;

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

    // Construct Holomorphic densities in complex orthogonal convention.
    let da = density(&ca, na, DensityMode::Holomorphic);
    let db = density(&cb, nb, DensityMode::Holomorphic);

    // Construct fock matrices at the same two electron integral scaling
    // parameter that was used for finding this state, and evaluate its energy.
    let (fa, fb) = fock_lambda(&ao.h, &ao.eri_coul, &da, &db, run.lambda);
    let e = energy(&ao.h, ao.enuc, &da, &db, &fa, &fb);

    if input.write.verbose >= 2 {
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
        rank_a: 0,
        rank_b: 0,
        indices_a: [0; 8],
        indices_b: [0; 8],
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
