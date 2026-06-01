// noci/cumulants/cumulants3.rs

use super::common::CumulantTensor;
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
    let half = <T as From<f64>>::from(0.5);
    let quarter = <T as From<f64>>::from(0.25);
    let mut lambda = CumulantTensor::zeros(3, n);

    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                for s in 0..n {
                    for t in 0..n {
                        for u in 0..n {
                            let g3i = (((((p * gamma3.n + q) * gamma3.n + r) * gamma3.n + s)
                                * gamma3.n
                                + t)
                                * gamma3.n)
                                + u;

                            let g1ps = lambda1.get(&[p], &[s]);
                            let g1pt = lambda1.get(&[p], &[t]);
                            let g1pu = lambda1.get(&[p], &[u]);

                            let g1qs = lambda1.get(&[q], &[s]);
                            let g1qt = lambda1.get(&[q], &[t]);
                            let g1qu = lambda1.get(&[q], &[u]);

                            let g1rs = lambda1.get(&[r], &[s]);
                            let g1rt = lambda1.get(&[r], &[t]);
                            let g1ru = lambda1.get(&[r], &[u]);

                            let disconnected = g1ps * g1qt * g1ru
                                - half * g1ps * g1qu * g1rt
                                - half * g1pt * g1qs * g1ru
                                - half * g1pu * g1qt * g1rs
                                + quarter * g1pt * g1qu * g1rs
                                + quarter * g1pu * g1qs * g1rt
                                + g1ps * lambda2.get(&[q, r], &[t, u])
                                - half * g1pt * lambda2.get(&[q, r], &[s, u])
                                - half * g1pu * lambda2.get(&[q, r], &[t, s])
                                + g1qt * lambda2.get(&[p, r], &[s, u])
                                - half * g1qs * lambda2.get(&[p, r], &[t, u])
                                - half * g1qu * lambda2.get(&[p, r], &[s, t])
                                + g1ru * lambda2.get(&[p, q], &[s, t])
                                - half * g1rs * lambda2.get(&[p, q], &[u, t])
                                - half * g1rt * lambda2.get(&[p, q], &[s, u]);

                            lambda.set(&[p, q, r], &[s, t, u], gamma3.data[g3i] - disconnected);
                        }
                    }
                }
            }
        }
    }

    lambda
}
