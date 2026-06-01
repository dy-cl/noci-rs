// noci/cumulants/cumulants1.rs

use super::common::{CumulantTensor, build_cumulant};
use crate::noci::rdm::RDM1;
use crate::noci::types::NOCIScalar;

pub(crate) type Cumulant1<T> = CumulantTensor<T>;

/// Build the active-space spin-free one-cumulant.
/// # Arguments:
/// - `gamma1`: Full-space spin-free one-body RDM.
/// - `active`: Active orbital indices.
/// # Returns:
/// - `Cumulant1<T>`: Active-space one-cumulant.
pub(crate) fn cumulants1<T: NOCIScalar>(
    gamma1: &RDM1<T>,
    active: &[usize],
) -> Cumulant1<T> {
    let n = active.len();

    build_cumulant(
        1,
        n,
        |upper, lower| {
            let p = active[upper[0]];
            let q = active[lower[0]];

            gamma1.data[p * gamma1.n + q]
        },
        &[],
        &[],
    )
}
