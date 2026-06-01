// noci/cumulants/cumulants4.rs

use super::common::{CumulantTensor, build_cumulant, disconnected};
use super::cumulants1::Cumulant1;
use super::cumulants2::Cumulant2;
use super::cumulants3::Cumulant3;
use crate::noci::rdm::RDM4;
use crate::noci::types::NOCIScalar;

pub(crate) type Cumulant4<T> = CumulantTensor<T>;

/// Build the active-space spin-free four-cumulant.
/// # Arguments:
/// - `gamma4`: Active-space spin-free four-body RDM.
/// - `lambda1`: Active-space one-cumulant.
/// - `lambda2`: Active-space two-cumulant.
/// - `lambda3`: Active-space three-cumulant.
/// # Returns:
/// - `Cumulant4<T>`: Active-space four-cumulant.
pub(crate) fn cumulants4<T: NOCIScalar>(
    gamma4: &RDM4<T>,
    lambda1: &Cumulant1<T>,
    lambda2: &Cumulant2<T>,
    lambda3: &Cumulant3<T>,
) -> Cumulant4<T> {
    let n = gamma4.n;
    let products = disconnected(4);

    build_cumulant(
        4,
        n,
        |upper, lower| {
            let i = upper
                .iter()
                .chain(lower.iter())
                .fold(0, |acc, &p| acc * gamma4.n + p);

            gamma4.data[i]
        },
        &[lambda1, lambda2, lambda3],
        &products,
    )
}
