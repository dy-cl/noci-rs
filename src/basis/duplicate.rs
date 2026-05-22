// basis/duplicate.rs

use ndarray::Array2;

use crate::DetState;
use crate::maths::real2_as;
use crate::noci::NOCIScalar;

/// Calculate the distance between determinant states from Phys. Rev. Lett. 101, 193001 as
/// d_{wx}^2 = N - {}^w D^{\mu\nu} {}^x D_{\nu\mu} = N - Tr(D_w S D_x S).
/// # Arguments
/// - `w`: Reference state from which distance is computed.
/// - `x`: State to which distance is computed.
/// - `s`: AO overlap matrix.
/// # Returns
/// - `f64`: Electron distance between the two determinant states.
pub fn electron_distance<T: NOCIScalar>(
    w: &DetState<T>,
    x: &DetState<T>,
    s: &Array2<f64>,
) -> f64 {
    let smat = real2_as::<T>(s);

    // Calculate electron number N as Tr(D_r S).
    let na = w.da.dot(&smat).diag().sum();
    let nb = w.db.dot(&smat).diag().sum();
    let n = na + nb;

    // Calculate Tr(D_w S D_x S).
    let tr_a = w.da.dot(&smat).dot(&*x.da).dot(&smat).diag().sum();
    let tr_b = w.db.dot(&smat).dot(&*x.db).dot(&smat).diag().sum();

    // Electron distance is the difference.
    (n - (tr_a + tr_b)).abs()
}

/// Calculate a squared overlap-weighted density distance between determinant states as
/// d_D^2 = ||(D_a^w - D_a^x)S||_F^2 + ||(D_b^w - D_b^x)S||_F^2.
/// This remains a real positive duplicate-detection metric for holomorphic states.
/// # Arguments
/// - `w`: Reference state from which distance is computed.
/// - `x`: State to which distance is computed.
/// - `s`: AO overlap matrix.
/// # Returns
/// - `f64`: Squared density distance between the two determinant states.
pub fn density_distance<T: NOCIScalar>(
    w: &DetState<T>,
    x: &DetState<T>,
    s: &Array2<f64>,
) -> f64 {
    let smat = real2_as::<T>(s);

    let dsa = (&*w.da - &*x.da).dot(&smat);
    let dsb = (&*w.db - &*x.db).dot(&smat);

    dsa.iter().chain(dsb.iter()).map(|z| z.abs().powi(2)).sum()
}

/// Mark duplicate NOCI-basis determinant states by setting `noci_basis = false`.
/// States are retained for printing and branch tracking.
/// # Arguments:
/// - `states`: Determinant states to deduplicate.
/// - `s`: AO overlap matrix.
/// - `d_tol`: Tolerance below which two states are treated as duplicates.
/// # Returns:
/// - `()`: Mutates duplicate states in place by setting `noci_basis = false`.
pub(crate) fn mark_duplicate_noci_states<T: NOCIScalar>(
    states: &mut [DetState<T>],
    s: &Array2<f64>,
    d_tol: f64,
) {
    println!("{}", "=".repeat(100));
    for i in 0..states.len() {
        if !states[i].noci_basis {
            continue;
        }

        for j in 0..i {
            if !states[j].noci_basis {
                continue;
            }

            let delec2 = electron_distance(&states[j], &states[i], s);
            let dp2 = density_distance(&states[j], &states[i], s);
            println!(
                "State '{}' distance from state '{}', electron: {}, density: {}",
                states[i].label, states[j].label, delec2, dp2
            );
            if delec2 < d_tol || dp2 < d_tol {
                println!(
                    "Removed state '{}' from NOCI basis as duplicate of '{}'",
                    states[i].label, states[j].label
                );
                states[i].noci_basis = false;
                break;
            }
        }
    }
}
