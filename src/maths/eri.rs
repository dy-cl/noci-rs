// maths/eri.rs

use ndarray::{Array2, Array4};
use num_complex::Complex64;

/// Transform ERIs from AO to real MO basis as:
///     (pq|rs) = \sum_{\mu\nu\lambda\sigma} (\mu\nu|\lambda\sigma) C_{\mu,p} C_{\nu,q} C_{\lambda,r} C_{\sigma,s}.
/// Contraction is performed one index at a time for O(n^5) work.
/// # Arguments:
/// - `eri`: AO basis ERIs.
/// - `c_mu_p`: MO coefficients C_{\mu,p}.
/// - `c_nu_q`: MO coefficients C_{\nu,q}.
/// - `c_lam_r`: MO coefficients C_{\lambda,r}.
/// - `c_sig_s`: MO coefficients C_{\sigma,s}.
/// # Returns
/// - `Array4<f64>`: ERIs transformed to the real MO basis.
pub fn eri_ao2mo(
    eri: &Array4<f64>,
    c_mu_p: &Array2<f64>,
    c_nu_q: &Array2<f64>,
    c_lam_r: &Array2<f64>,
    c_sig_s: &Array2<f64>,
) -> Array4<f64> {
    let nbas = c_mu_p.nrows();
    let nmo_p = c_mu_p.ncols();
    let nmo_q = c_nu_q.ncols();
    let nmo_r = c_lam_r.ncols();
    let nmo_s = c_sig_s.ncols();

    let mut t1 = Array4::<f64>::zeros((nbas, nbas, nbas, nmo_s));
    for mu in 0..nbas {
        for nu in 0..nbas {
            for lam in 0..nbas {
                for s in 0..nmo_s {
                    let mut acc = 0.0;
                    for sig in 0..nbas {
                        acc += eri[(mu, nu, lam, sig)] * c_sig_s[(sig, s)];
                    }
                    t1[(mu, nu, lam, s)] = acc;
                }
            }
        }
    }

    let mut t2 = Array4::<f64>::zeros((nbas, nbas, nmo_r, nmo_s));
    for mu in 0..nbas {
        for nu in 0..nbas {
            for r in 0..nmo_r {
                for s in 0..nmo_s {
                    let mut acc = 0.0;
                    for lam in 0..nbas {
                        acc += t1[(mu, nu, lam, s)] * c_lam_r[(lam, r)];
                    }
                    t2[(mu, nu, r, s)] = acc;
                }
            }
        }
    }

    let mut t3 = Array4::<f64>::zeros((nbas, nmo_q, nmo_r, nmo_s));
    for mu in 0..nbas {
        for q in 0..nmo_q {
            for r in 0..nmo_r {
                for s in 0..nmo_s {
                    let mut acc = 0.0;
                    for nu in 0..nbas {
                        acc += t2[(mu, nu, r, s)] * c_nu_q[(nu, q)];
                    }
                    t3[(mu, q, r, s)] = acc;
                }
            }
        }
    }

    let mut out = Array4::<f64>::zeros((nmo_p, nmo_q, nmo_r, nmo_s));
    for p in 0..nmo_p {
        for q in 0..nmo_q {
            for r in 0..nmo_r {
                for s in 0..nmo_s {
                    let mut acc = 0.0;
                    for mu in 0..nbas {
                        acc += t3[(mu, q, r, s)] * c_mu_p[(mu, p)];
                    }
                    out[(p, q, r, s)] = acc;
                }
            }
        }
    }

    out
}

/// Transform ERIs from AO to complex MO basis with Hermitian bra-side conjugation as:
///     (pq|rs) = \sum_{\mu\nu\lambda\sigma} (\mu\nu|\lambda\sigma) C^*_{\mu,p} C_{\nu,q} C^*_{\lambda,r} C_{\sigma,s}.
/// This is the ordinary complex MO integral transform for Hermitian matrix elements.
/// # Arguments:
/// - `eri`: AO basis ERIs.
/// - `c_mu_p`: MO coefficients C_{\mu,p}, conjugated in the transform.
/// - `c_nu_q`: MO coefficients C_{\nu,q}.
/// - `c_lam_r`: MO coefficients C_{\lambda,r}, conjugated in the transform.
/// - `c_sig_s`: MO coefficients C_{\sigma,s}.
/// # Returns
/// - `Array4<Complex64>`: ERIs transformed to the complex MO basis.
pub fn eri_ao2mo_complex_hermitian(
    eri: &Array4<f64>,
    c_mu_p: &Array2<Complex64>,
    c_nu_q: &Array2<Complex64>,
    c_lam_r: &Array2<Complex64>,
    c_sig_s: &Array2<Complex64>,
) -> Array4<Complex64> {
    let nbas = c_mu_p.nrows();
    let nmo_p = c_mu_p.ncols();
    let nmo_q = c_nu_q.ncols();
    let nmo_r = c_lam_r.ncols();
    let nmo_s = c_sig_s.ncols();

    let mut t1 = Array4::<Complex64>::zeros((nbas, nbas, nbas, nmo_s));
    for mu in 0..nbas {
        for nu in 0..nbas {
            for lam in 0..nbas {
                for s in 0..nmo_s {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for sig in 0..nbas {
                        acc += c_sig_s[(sig, s)] * eri[(mu, nu, lam, sig)];
                    }
                    t1[(mu, nu, lam, s)] = acc;
                }
            }
        }
    }

    let mut t2 = Array4::<Complex64>::zeros((nbas, nbas, nmo_r, nmo_s));
    for mu in 0..nbas {
        for nu in 0..nbas {
            for r in 0..nmo_r {
                for s in 0..nmo_s {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for lam in 0..nbas {
                        acc += t1[(mu, nu, lam, s)] * c_lam_r[(lam, r)].conj();
                    }
                    t2[(mu, nu, r, s)] = acc;
                }
            }
        }
    }

    let mut t3 = Array4::<Complex64>::zeros((nbas, nmo_q, nmo_r, nmo_s));
    for mu in 0..nbas {
        for q in 0..nmo_q {
            for r in 0..nmo_r {
                for s in 0..nmo_s {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for nu in 0..nbas {
                        acc += t2[(mu, nu, r, s)] * c_nu_q[(nu, q)];
                    }
                    t3[(mu, q, r, s)] = acc;
                }
            }
        }
    }

    let mut out = Array4::<Complex64>::zeros((nmo_p, nmo_q, nmo_r, nmo_s));
    for p in 0..nmo_p {
        for q in 0..nmo_q {
            for r in 0..nmo_r {
                for s in 0..nmo_s {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for mu in 0..nbas {
                        acc += t3[(mu, q, r, s)] * c_mu_p[(mu, p)].conj();
                    }
                    out[(p, q, r, s)] = acc;
                }
            }
        }
    }

    out
}

/// Scalar type accepted by Hermitian ERI AO-to-MO transformation dispatch.
pub trait ERIScalar: Sized {
    /// Transform ERIs from AO to MO basis with Hermitian bra-side conjugation as:
    ///     (pq|rs) = \sum_{\mu\nu\lambda\sigma} (\mu\nu|\lambda\sigma) C^*_{\mu,p} C_{\nu,q} C^*_{\lambda,r} C_{\sigma,s}.
    /// For real orbitals this is equivalent to the ordinary AO-to-MO transform.
    /// # Arguments:
    /// - `eri`: AO basis ERIs.
    /// - `c_mu_p`: MO coefficients C_{\mu,p}, conjugated in the transform.
    /// - `c_nu_q`: MO coefficients C_{\nu,q}.
    /// - `c_lam_r`: MO coefficients C_{\lambda,r}, conjugated in the transform.
    /// - `c_sig_s`: MO coefficients C_{\sigma,s}.
    /// # Returns
    /// - `Array4<Self>`: ERIs transformed to the MO basis.
    fn eri_ao2mo_hermitian(
        eri: &Array4<f64>,
        c_mu_p: &Array2<Self>,
        c_nu_q: &Array2<Self>,
        c_lam_r: &Array2<Self>,
        c_sig_s: &Array2<Self>,
    ) -> Array4<Self>;
}

impl ERIScalar for f64 {
    /// Transform ERIs from AO to real MO basis with Hermitian bra-side conjugation.
    /// For real coefficients this is equivalent to the ordinary AO-to-MO transform.
    /// # Arguments:
    /// - `eri`: AO basis ERIs.
    /// - `c_mu_p`: MO coefficients C_{\mu,p}.
    /// - `c_nu_q`: MO coefficients C_{\nu,q}.
    /// - `c_lam_r`: MO coefficients C_{\lambda,r}.
    /// - `c_sig_s`: MO coefficients C_{\sigma,s}.
    /// # Returns
    /// - `Array4<f64>`: ERIs transformed to the real MO basis.
    fn eri_ao2mo_hermitian(
        eri: &Array4<f64>,
        c_mu_p: &Array2<Self>,
        c_nu_q: &Array2<Self>,
        c_lam_r: &Array2<Self>,
        c_sig_s: &Array2<Self>,
    ) -> Array4<Self> {
        eri_ao2mo(eri, c_mu_p, c_nu_q, c_lam_r, c_sig_s)
    }
}

impl ERIScalar for Complex64 {
    /// Transform ERIs from AO to complex MO basis with Hermitian bra-side conjugation.
    /// This is the ordinary complex MO integral transform for Hermitian matrix elements.
    /// # Arguments:
    /// - `eri`: AO basis ERIs.
    /// - `c_mu_p`: MO coefficients C_{\mu,p}, conjugated in the transform.
    /// - `c_nu_q`: MO coefficients C_{\nu,q}.
    /// - `c_lam_r`: MO coefficients C_{\lambda,r}, conjugated in the transform.
    /// - `c_sig_s`: MO coefficients C_{\sigma,s}.
    /// # Returns
    /// - `Array4<Complex64>`: ERIs transformed to the complex MO basis.
    fn eri_ao2mo_hermitian(
        eri: &Array4<f64>,
        c_mu_p: &Array2<Self>,
        c_nu_q: &Array2<Self>,
        c_lam_r: &Array2<Self>,
        c_sig_s: &Array2<Self>,
    ) -> Array4<Self> {
        eri_ao2mo_complex_hermitian(eri, c_mu_p, c_nu_q, c_lam_r, c_sig_s)
    }
}

/// Transform ERIs from AO to MO basis with Hermitian bra-side conjugation using scalar dispatch.
/// # Arguments:
/// - `eri`: AO basis ERIs.
/// - `c_mu_p`: MO coefficients C_{\mu,p}, conjugated in the transform.
/// - `c_nu_q`: MO coefficients C_{\nu,q}.
/// - `c_lam_r`: MO coefficients C_{\lambda,r}, conjugated in the transform.
/// - `c_sig_s`: MO coefficients C_{\sigma,s}.
/// # Returns
/// - `Array4<T>`: ERIs transformed to the MO basis.
pub fn eri_ao2mo_hermitian_as<T: ERIScalar>(
    eri: &Array4<f64>,
    c_mu_p: &Array2<T>,
    c_nu_q: &Array2<T>,
    c_lam_r: &Array2<T>,
    c_sig_s: &Array2<T>,
) -> Array4<T> {
    T::eri_ao2mo_hermitian(eri, c_mu_p, c_nu_q, c_lam_r, c_sig_s)
}
