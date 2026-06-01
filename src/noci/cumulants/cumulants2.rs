// noci/cumulants/cumulants2.rs

use super::common::{CumulantTensor, build_cumulant, disconnected};
use super::cumulants1::Cumulant1;
use crate::noci::rdm::RDM2;
use crate::noci::types::NOCIScalar;

pub(crate) type Cumulant2<T> = CumulantTensor<T>;

/// Build the active-space spin-free two-cumulant.
/// # Arguments:
/// - `gamma2`: Full-space spin-free two-body RDM.
/// - `lambda1`: Active-space one-cumulant.
/// - `active`: Active orbital indices.
/// # Returns:
/// - `Cumulant2<T>`: Active-space two-cumulant.
pub(crate) fn cumulants2<T: NOCIScalar>(
    gamma2: &RDM2<T>,
    lambda1: &Cumulant1<T>,
    active: &[usize],
) -> Cumulant2<T> {
    let n = active.len();
    let products = disconnected(2);

    build_cumulant(
        2,
        n,
        |upper, lower| {
            let p = active[upper[0]];
            let q = active[upper[1]];
            let r = active[lower[0]];
            let s = active[lower[1]];

            gamma2.data[(((p * gamma2.n + q) * gamma2.n + r) * gamma2.n) + s]
        },
        &[lambda1],
        &products,
    )
}
