// noci/cumulants/cumulants3.rs

use super::common::{CumulantTensor, build_cumulant, disconnected};
use super::cumulants1::Cumulant1;
use super::cumulants2::Cumulant2;
use crate::noci::rdm::RDM3;
use crate::noci::types::NOCIScalar;

pub(crate) type Cumulant3<T> = CumulantTensor<T>;

/// Build the active-space spin-free three-cumulant.
/// # Arguments:
/// - `gamma3`: Active-space spin-free three-body RDM.
/// - `lambda1`: Active-space one-cumulant.
/// - `lambda2`: Active-space two-cumulant.
/// # Returns:
/// - `Cumulant3<T>`: Active-space three-cumulant.
pub(crate) fn cumulants3<T: NOCIScalar>(
    gamma3: &RDM3<T>,
    lambda1: &Cumulant1<T>,
    lambda2: &Cumulant2<T>,
) -> Cumulant3<T> {
    let n = gamma3.n;
    let products = disconnected(3);

    build_cumulant(
        3,
        n,
        |upper, lower| {
            let i = upper
                .iter()
                .chain(lower.iter())
                .fold(0, |acc, &p| acc * gamma3.n + p);

            gamma3.data[i]
        },
        &[lambda1, lambda2],
        &products,
    )
}
