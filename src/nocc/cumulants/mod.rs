// nocc/cumulants/mod.rs

mod common;
mod cumulants1;
mod cumulants2;
mod cumulants3;
mod cumulants4;

pub(crate) use self::cumulants1::{Cumulant1, cumulants1};
pub(crate) use self::cumulants2::{Cumulant2, cumulants2};
pub(crate) use self::cumulants3::{Cumulant3, cumulants3};
pub(crate) use self::cumulants4::{Cumulant4, cumulants4};

use crate::nocc::rdm::{RDM1, RDM2, RDM3, RDM4};
use crate::noci::NOCIScalar;

/// Spin-free cumulants through rank four.
pub(crate) struct Cumulants<T: NOCIScalar> {
    /// Active-space one-cumulant Λ[p, q].
    pub lambda1: Cumulant1<T>,
    /// Active-space two-cumulant Λ[p, q, r, s].
    pub lambda2: Cumulant2<T>,
    /// Active-space three-cumulant Λ[p, q, r, s, t, u].
    pub lambda3: Cumulant3<T>,
    /// Active-space four-cumulant Λ[p, q, r, s, t, u, v, w].
    pub lambda4: Cumulant4<T>,
}

/// Build spin-free cumulants from one- to four-body spin-free RDMs.
/// # Arguments:
/// - `gamma1`: Full-space spin-free one-body RDM.
/// - `gamma2`: Full-space spin-free two-body RDM.
/// - `gamma3`: Active-space spin-free three-body RDM.
/// - `gamma4`: Active-space spin-free four-body RDM.
/// - `active`: Active orbital indices used to build `gamma3` and `gamma4`.
/// # Returns:
/// - `Cumulants<T>`: Active-space spin-free cumulants through rank four.
pub(crate) fn cumulants<T: NOCIScalar>(
    gamma1: &RDM1<T>,
    gamma2: &RDM2<T>,
    gamma3: &RDM3<T>,
    gamma4: &RDM4<T>,
    active: &[usize],
) -> Cumulants<T> {
    let lambda1 = cumulants1(gamma1, active);
    let lambda2 = cumulants2(gamma2, &lambda1, active);
    let lambda3 = cumulants3(gamma3, &lambda1, &lambda2);
    let lambda4 = cumulants4(gamma4, &lambda1, &lambda2, &lambda3);

    Cumulants {
        lambda1,
        lambda2,
        lambda3,
        lambda4,
    }
}
