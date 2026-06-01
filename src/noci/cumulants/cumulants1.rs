// noci/cumulants/cumulants1.rs

use super::common::CumulantTensor;
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
    let mut lambda = CumulantTensor::zeros(1, n);

    for p in 0..n {
        for q in 0..n {
            let pp = active[p];
            let qq = active[q];

            lambda.set(&[p], &[q], gamma1.data[pp * gamma1.n + qq]);
        }
    }

    lambda
}
