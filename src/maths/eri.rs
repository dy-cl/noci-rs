// maths/eri.rs

use ndarray::linalg::general_mat_mul;
use ndarray::{
    Array2, Array4, ArrayView2, ArrayView4, ArrayViewMut2, ArrayViewMut4, LinalgScalar, Zip,
};
use num_complex::Complex64;

/// Reusable allocation boundary for AO-to-MO ERI transformations.
pub struct ERIAO2MOScratch<T: ERIScalar> {
    /// Alternating contraction and permutation buffer.
    worka: Vec<T>,
    /// Alternating contraction and permutation buffer.
    workb: Vec<T>,
    /// Conjugated C_{\mu,p} buffer populated only by the complex implementation.
    c_mu_p_conj: Vec<T>,
    /// Conjugated C_{\lambda,r} buffer populated only by the complex implementation.
    c_lam_r_conj: Vec<T>,
    /// Empty for `f64`; for `Complex64`, a single complex conversion of the real AO ERIs.
    eri_as: Option<Array4<T>>,
}

/// Scalar type accepted by Hermitian ERI AO-to-MO transformation dispatch.
pub trait ERIScalar: LinalgScalar + From<f64> {
    /// Construct reusable scratch storage for AO-to-MO ERI transformations.
    /// # Arguments:
    /// - `eri`: AO basis ERIs.
    /// - `nmop`: Number of p-index MOs.
    /// - `nmoq`: Number of q-index MOs.
    /// - `nmor`: Number of r-index MOs.
    /// - `nmos`: Number of s-index MOs.
    /// # Returns
    /// - `ERIAO2MOScratch<Self>`: Reusable transformation scratch storage.
    fn new_eri_ao2mo_scratch(
        eri: &Array4<f64>,
        nmop: usize,
        nmoq: usize,
        nmor: usize,
        nmos: usize,
    ) -> ERIAO2MOScratch<Self>;

    /// Transform ERIs from AO to MO basis with Hermitian bra-side conjugation into caller storage.
    /// # Arguments:
    /// - `eri`: AO basis ERIs.
    /// - `c_mu_p`: MO coefficients C_{\mu,p}, conjugated in the transform.
    /// - `c_nu_q`: MO coefficients C_{\nu,q}.
    /// - `c_lam_r`: MO coefficients C_{\lambda,r}, conjugated in the transform.
    /// - `c_sig_s`: MO coefficients C_{\sigma,s}.
    /// - `out`: Output ERIs in [p, q, r, s] order.
    /// - `scratch`: Reusable transformation scratch storage.
    fn eri_ao2mo_hermitian_into(
        eri: &Array4<f64>,
        c_mu_p: &Array2<Self>,
        c_nu_q: &Array2<Self>,
        c_lam_r: &Array2<Self>,
        c_sig_s: &Array2<Self>,
        out: ArrayViewMut4<'_, Self>,
        scratch: &mut ERIAO2MOScratch<Self>,
    );
}

impl ERIScalar for f64 {
    /// Construct reusable scratch storage for real AO-to-MO ERI transformations.
    /// # Arguments:
    /// - `eri`: AO basis ERIs.
    /// - `nmop`: Number of p-index MOs.
    /// - `nmoq`: Number of q-index MOs.
    /// - `nmor`: Number of r-index MOs.
    /// - `nmos`: Number of s-index MOs.
    /// # Returns
    /// - `ERIAO2MOScratch<f64>`: Reusable transformation scratch storage.
    fn new_eri_ao2mo_scratch(
        eri: &Array4<f64>,
        nmop: usize,
        nmoq: usize,
        nmor: usize,
        nmos: usize,
    ) -> ERIAO2MOScratch<Self> {
        let nbas = eri.shape()[0];

        let t1len = nbas * nbas * nbas * nmos;
        let t2len = nbas * nbas * nmos * nmor;
        let t3len = nbas * nmos * nmor * nmoq;
        let t4len = nmos * nmor * nmoq * nmop;

        let worklen = t1len.max(t2len).max(t3len).max(t4len);

        ERIAO2MOScratch {
            worka: vec![0.0; worklen],
            workb: vec![0.0; worklen],
            c_mu_p_conj: Vec::new(),
            c_lam_r_conj: Vec::new(),
            eri_as: None,
        }
    }

    /// Transform ERIs from AO to real MO basis with Hermitian bra-side conjugation into caller storage.
    /// For real coefficients this is equivalent to the ordinary AO-to-MO transform.
    /// # Arguments:
    /// - `eri`: AO basis ERIs.
    /// - `c_mu_p`: MO coefficients C_{\mu,p}.
    /// - `c_nu_q`: MO coefficients C_{\nu,q}.
    /// - `c_lam_r`: MO coefficients C_{\lambda,r}.
    /// - `c_sig_s`: MO coefficients C_{\sigma,s}.
    /// - `out`: Output ERIs in [p, q, r, s] order.
    /// - `scratch`: Reusable transformation scratch storage.
    fn eri_ao2mo_hermitian_into(
        eri: &Array4<f64>,
        c_mu_p: &Array2<Self>,
        c_nu_q: &Array2<Self>,
        c_lam_r: &Array2<Self>,
        c_sig_s: &Array2<Self>,
        out: ArrayViewMut4<'_, Self>,
        scratch: &mut ERIAO2MOScratch<Self>,
    ) {
        eri_ao2mo_into(
            eri.view(),
            c_mu_p.view(),
            c_nu_q.view(),
            c_lam_r.view(),
            c_sig_s.view(),
            out,
            &mut scratch.worka,
            &mut scratch.workb,
        );
    }
}

impl ERIScalar for Complex64 {
    /// Construct reusable scratch storage for complex AO-to-MO ERI transformations.
    /// # Arguments:
    /// - `eri`: AO basis ERIs.
    /// - `nmop`: Number of p-index MOs.
    /// - `nmoq`: Number of q-index MOs.
    /// - `nmor`: Number of r-index MOs.
    /// - `nmos`: Number of s-index MOs.
    /// # Returns
    /// - `ERIAO2MOScratch<Complex64>`: Reusable transformation scratch storage.
    fn new_eri_ao2mo_scratch(
        eri: &Array4<f64>,
        nmop: usize,
        nmoq: usize,
        nmor: usize,
        nmos: usize,
    ) -> ERIAO2MOScratch<Self> {
        let nbas = eri.shape()[0];

        let t1len = nbas * nbas * nbas * nmos;
        let t2len = nbas * nbas * nmos * nmor;
        let t3len = nbas * nmos * nmor * nmoq;
        let t4len = nmos * nmor * nmoq * nmop;

        let worklen = t1len.max(t2len).max(t3len).max(t4len);

        ERIAO2MOScratch {
            worka: vec![Complex64::new(0.0, 0.0); worklen],
            workb: vec![Complex64::new(0.0, 0.0); worklen],
            c_mu_p_conj: vec![Complex64::new(0.0, 0.0); nbas * nmop],
            c_lam_r_conj: vec![Complex64::new(0.0, 0.0); nbas * nmor],
            eri_as: Some(eri.mapv(|x| Complex64::new(x, 0.0))),
        }
    }

    /// Transform ERIs from AO to complex MO basis with Hermitian bra-side conjugation into caller storage.
    /// This is the ordinary complex MO integral transform for Hermitian matrix elements.
    /// # Arguments:
    /// - `_eri`: AO basis ERIs, already converted and stored in `scratch`.
    /// - `c_mu_p`: MO coefficients C_{\mu,p}, conjugated in the transform.
    /// - `c_nu_q`: MO coefficients C_{\nu,q}.
    /// - `c_lam_r`: MO coefficients C_{\lambda,r}, conjugated in the transform.
    /// - `c_sig_s`: MO coefficients C_{\sigma,s}.
    /// - `out`: Output ERIs in [p, q, r, s] order.
    /// - `scratch`: Reusable transformation scratch storage.
    fn eri_ao2mo_hermitian_into(
        _eri: &Array4<f64>,
        c_mu_p: &Array2<Self>,
        c_nu_q: &Array2<Self>,
        c_lam_r: &Array2<Self>,
        c_sig_s: &Array2<Self>,
        out: ArrayViewMut4<'_, Self>,
        scratch: &mut ERIAO2MOScratch<Self>,
    ) {
        let nbas = c_mu_p.nrows();

        let nmop = c_mu_p.ncols();
        let nmor = c_lam_r.ncols();

        {
            let mut c_mu_p_conj =
                ArrayViewMut2::from_shape((nbas, nmop), &mut scratch.c_mu_p_conj).unwrap();
            Zip::from(&mut c_mu_p_conj)
                .and(c_mu_p)
                .for_each(|dst, &src| *dst = src.conj());
        }
        {
            let mut c_lam_r_conj =
                ArrayViewMut2::from_shape((nbas, nmor), &mut scratch.c_lam_r_conj).unwrap();
            Zip::from(&mut c_lam_r_conj)
                .and(c_lam_r)
                .for_each(|dst, &src| *dst = src.conj());
        }

        let c_mu_p_conj = ArrayView2::from_shape((nbas, nmop), &scratch.c_mu_p_conj).unwrap();
        let c_lam_r_conj = ArrayView2::from_shape((nbas, nmor), &scratch.c_lam_r_conj).unwrap();

        let eri_as = scratch.eri_as.as_ref().unwrap();

        eri_ao2mo_into(
            eri_as.view(),
            c_mu_p_conj,
            c_nu_q.view(),
            c_lam_r_conj,
            c_sig_s.view(),
            out,
            &mut scratch.worka,
            &mut scratch.workb,
        );
    }
}

/// Transform typed AO ERIs using preselected effective coefficient matrices and reusable buffers.
/// The contraction order is \sigma then \lambda then \nu then \mu.
/// The final assignment writes [p, q, r, s] into `out`.
/// # Arguments:
/// - `eri`: Typed AO ERIs.
/// - `c_mu_p`: Effective C_{\mu,p} coefficients.
/// - `c_nu_q`: Effective C_{\nu,q} coefficients.
/// - `c_lam_r`: Effective C_{\lambda,r} coefficients.
/// - `c_sig_s`: Effective C_{\sigma,s} coefficients.
/// - `out`: Output ERIs in [p, q, r, s] order.
/// - `worka`: Reusable contraction and permutation buffer.
/// - `workb`: Reusable contraction and permutation buffer.
fn eri_ao2mo_into<T: ERIScalar>(
    eri: ArrayView4<'_, T>,
    c_mu_p: ArrayView2<'_, T>,
    c_nu_q: ArrayView2<'_, T>,
    c_lam_r: ArrayView2<'_, T>,
    c_sig_s: ArrayView2<'_, T>,
    mut out: ArrayViewMut4<'_, T>,
    worka: &mut [T],
    workb: &mut [T],
) {
    let nbas = eri.shape()[0];
    let nmop = c_mu_p.ncols();
    let nmoq = c_nu_q.ncols();
    let nmor = c_lam_r.ncols();
    let nmos = c_sig_s.ncols();
    let alpha = T::from(1.0);
    let beta = T::from(0.0);

    let t1len = nbas * nbas * nbas * nmos;
    let t2len = nbas * nbas * nmos * nmor;
    let t3len = nbas * nmos * nmor * nmoq;
    let t4len = nmos * nmor * nmoq * nmop;

    let erirows = eri.into_shape((nbas * nbas * nbas, nbas)).unwrap();
    let mut t1rows =
        ArrayViewMut2::from_shape((nbas * nbas * nbas, nmos), &mut worka[..t1len]).unwrap();
    general_mat_mul(alpha, &erirows, &c_sig_s, beta, &mut t1rows);

    let src = ArrayView4::from_shape((nbas, nbas, nbas, nmos), &worka[..t1len]).unwrap();
    let mut dst = ArrayViewMut4::from_shape((nbas, nbas, nmos, nbas), &mut workb[..t1len]).unwrap();
    dst.assign(&src.permuted_axes([0, 1, 3, 2]));

    let t1rows = ArrayView2::from_shape((nbas * nbas * nmos, nbas), &workb[..t1len]).unwrap();
    let mut t2rows =
        ArrayViewMut2::from_shape((nbas * nbas * nmos, nmor), &mut worka[..t2len]).unwrap();
    general_mat_mul(alpha, &t1rows, &c_lam_r, beta, &mut t2rows);

    let src = ArrayView4::from_shape((nbas, nbas, nmos, nmor), &worka[..t2len]).unwrap();
    let mut dst = ArrayViewMut4::from_shape((nbas, nmos, nmor, nbas), &mut workb[..t2len]).unwrap();
    dst.assign(&src.permuted_axes([0, 2, 3, 1]));

    let t2rows = ArrayView2::from_shape((nbas * nmos * nmor, nbas), &workb[..t2len]).unwrap();
    let mut t3rows =
        ArrayViewMut2::from_shape((nbas * nmos * nmor, nmoq), &mut worka[..t3len]).unwrap();
    general_mat_mul(alpha, &t2rows, &c_nu_q, beta, &mut t3rows);

    let src = ArrayView4::from_shape((nbas, nmos, nmor, nmoq), &worka[..t3len]).unwrap();
    let mut dst = ArrayViewMut4::from_shape((nmos, nmor, nmoq, nbas), &mut workb[..t3len]).unwrap();
    dst.assign(&src.permuted_axes([1, 2, 3, 0]));

    let t3rows = ArrayView2::from_shape((nmos * nmor * nmoq, nbas), &workb[..t3len]).unwrap();
    let mut t4rows =
        ArrayViewMut2::from_shape((nmos * nmor * nmoq, nmop), &mut worka[..t4len]).unwrap();
    general_mat_mul(alpha, &t3rows, &c_mu_p, beta, &mut t4rows);

    let src = ArrayView4::from_shape((nmos, nmor, nmoq, nmop), &worka[..t4len]).unwrap();
    out.assign(&src.permuted_axes([3, 2, 1, 0]));
}
