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

    let eri_std = eri.as_standard_layout();
    let eri_rows = eri_std
        .view()
        .into_shape((nbas * nbas * nbas, nbas))
        .unwrap();
    let t1 = eri_rows
        .dot(c_sig_s)
        .into_shape((nbas, nbas, nbas, nmo_s))
        .unwrap();

    let t1_lam_last = t1
        .view()
        .permuted_axes([0, 1, 3, 2])
        .as_standard_layout()
        .to_owned();
    let t2 = t1_lam_last
        .view()
        .into_shape((nbas * nbas * nmo_s, nbas))
        .unwrap()
        .dot(c_lam_r)
        .into_shape((nbas, nbas, nmo_s, nmo_r))
        .unwrap();

    let t2_nu_last = t2
        .view()
        .permuted_axes([0, 2, 3, 1])
        .as_standard_layout()
        .to_owned();
    let t3 = t2_nu_last
        .view()
        .into_shape((nbas * nmo_s * nmo_r, nbas))
        .unwrap()
        .dot(c_nu_q)
        .into_shape((nbas, nmo_s, nmo_r, nmo_q))
        .unwrap();

    let t3_mu_last = t3
        .view()
        .permuted_axes([1, 2, 3, 0])
        .as_standard_layout()
        .to_owned();
    t3_mu_last
        .view()
        .into_shape((nmo_s * nmo_r * nmo_q, nbas))
        .unwrap()
        .dot(c_mu_p)
        .into_shape((nmo_s, nmo_r, nmo_q, nmo_p))
        .unwrap()
        .permuted_axes([3, 2, 1, 0])
        .as_standard_layout()
        .to_owned()
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

    let eri_complex = eri.as_standard_layout().mapv(|x| Complex64::new(x, 0.0));
    let c_lam_r_conj = c_lam_r.mapv(|x| x.conj());
    let c_mu_p_conj = c_mu_p.mapv(|x| x.conj());

    let eri_rows = eri_complex
        .view()
        .into_shape((nbas * nbas * nbas, nbas))
        .unwrap();
    let t1 = eri_rows
        .dot(c_sig_s)
        .into_shape((nbas, nbas, nbas, nmo_s))
        .unwrap();

    let t1_lam_last = t1
        .view()
        .permuted_axes([0, 1, 3, 2])
        .as_standard_layout()
        .to_owned();
    let t2 = t1_lam_last
        .view()
        .into_shape((nbas * nbas * nmo_s, nbas))
        .unwrap()
        .dot(&c_lam_r_conj)
        .into_shape((nbas, nbas, nmo_s, nmo_r))
        .unwrap();

    let t2_nu_last = t2
        .view()
        .permuted_axes([0, 2, 3, 1])
        .as_standard_layout()
        .to_owned();
    let t3 = t2_nu_last
        .view()
        .into_shape((nbas * nmo_s * nmo_r, nbas))
        .unwrap()
        .dot(c_nu_q)
        .into_shape((nbas, nmo_s, nmo_r, nmo_q))
        .unwrap();

    let t3_mu_last = t3
        .view()
        .permuted_axes([1, 2, 3, 0])
        .as_standard_layout()
        .to_owned();
    t3_mu_last
        .view()
        .into_shape((nmo_s * nmo_r * nmo_q, nbas))
        .unwrap()
        .dot(&c_mu_p_conj)
        .into_shape((nmo_s, nmo_r, nmo_q, nmo_p))
        .unwrap()
        .permuted_axes([3, 2, 1, 0])
        .as_standard_layout()
        .to_owned()
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
