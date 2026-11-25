// maths/eri.rs

// External crate imports.
use ndarray::linalg::general_mat_mul;
use ndarray::{Array2, Array4, ArrayView2, ArrayViewMut2, ArrayViewMut4, Axis};
use rayon::prelude::*;

// Crate-root imports.
use crate::StateScalar;

/// Reusable buffers for AO-to-MO ERI transformations.
pub struct ERIAO2MOScratch<T> {
    /// `Half-transformed (\mu\nu|rs) buffer.`
    worka: Vec<T>,
    /// `Three-quarter-transformed (\mu q|rs) buffer.`
    workb: Vec<T>,
}

impl<T: StateScalar> ERIAO2MOScratch<T> {
    /// Construct reusable buffers for AO-to-MO ERI transformations.
    /// # Arguments:
    /// - `eri`: AO basis ERIs, which set the number of basis functions.
    /// - `nmoq`: Number of q-index MOs.
    /// - `nmor`: Number of r-index MOs.
    /// - `nmos`: Number of s-index MOs.
    /// # Returns
    /// - `ERIAO2MOScratch<T>`: Reusable transformation buffers.
    pub fn new(
        eri: &Array4<f64>,
        nmoq: usize,
        nmor: usize,
        nmos: usize,
    ) -> Self {
        let nbas = eri.shape()[0];

        Self {
            worka: vec![T::zero(); nbas * nbas * nmor * nmos],
            workb: vec![T::zero(); nbas * nmoq * nmor * nmos],
        }
    }
}

/// Transform real AO ERIs into the MO integrals of Hermitian matrix elements,
/// `(pq|rs) = \sum_{\mu\nu\lambda\sigma} C^*_{\mu p} C_{\nu q} C^*_{\lambda r} C_{\sigma s} (\mu\nu|\lambda\sigma)`.
/// For real coefficients the conjugations have no effect. The first half-transformation forms
/// `(\mu\nu|rs) = \sum_{\lambda\sigma} C^*_{\lambda r} (\mu\nu|\lambda\sigma) C_{\sigma s}` as one small
/// matrix product per AO pair, which stays in cache. The second half contracts `\nu` and then
/// `\mu` with matrix products whose rows already follow the `[p, q, r, s]` output order, so no
/// intermediate tensor is permuted.
/// # Arguments:
/// - `eri`: AO basis ERIs in `[\mu, \nu, \lambda, \sigma]` order.
/// - `c_mu_p`: `MO coefficients C_{\mu,p}, conjugated in the transform.`
/// - `c_nu_q`: `MO coefficients C_{\nu,q}.`
/// - `c_lam_r`: `MO coefficients C_{\lambda,r}, conjugated in the transform.`
/// - `c_sig_s`: `MO coefficients C_{\sigma,s}.`
/// - `out`: Output ERIs in `[p, q, r, s]` order.
/// - `scratch`: Reusable transformation buffers.
/// # Returns
/// - `()`: Writes the `[p, q, r, s]` tensor into `out`.
/// # Panics
/// - Panics if `out` is not in standard contiguous layout.
pub fn eri_ao2mo_hermitian_into<T: StateScalar>(
    eri: &Array4<f64>,
    c_mu_p: &Array2<T>,
    c_nu_q: &Array2<T>,
    c_lam_r: &Array2<T>,
    c_sig_s: &Array2<T>,
    out: ArrayViewMut4<'_, T>,
    scratch: &mut ERIAO2MOScratch<T>,
) {
    let nbas = eri.shape()[0];
    let nmop = c_mu_p.ncols();
    let nmoq = c_nu_q.ncols();
    let nmor = c_lam_r.ncols();
    let nmos = c_sig_s.ncols();
    let alpha = T::one();
    let beta = T::zero();

    let nrs = nmor * nmos;
    let nqrs = nmoq * nrs;

    // Hermitian MO integrals conjugate the bra-side coefficients `C^*_{\mu p}` and
    // `C^*_{\lambda r}`; the transposes are the left factors of the matrix products below.
    let c_mu_t = c_mu_p.t().mapv(|z| z.conj());
    let c_nu_t = c_nu_q.t();
    let c_lam_t = c_lam_r.t().mapv(|z| z.conj());

    // First half-transformation of each AO pair,
    // `(\mu\nu|rs) = [C_\lambda^\dagger (\mu\nu|\cdot\cdot) C_\sigma]_{rs}`, stored as `[\mu, \nu, r, s]`.
    let eripairs = eri.view().into_shape((nbas * nbas, nbas * nbas)).unwrap();
    scratch.worka[..nbas * nbas * nrs]
        .par_chunks_mut(nrs)
        .enumerate()
        .for_each_init(
            || {
                (
                    Array2::<T>::zeros((nbas, nbas)),
                    Array2::<T>::zeros((nmor, nbas)),
                )
            },
            |(pair, half), (munu, dst)| {
                pair.iter_mut()
                    .zip(eripairs.row(munu))
                    .for_each(|(p, &x)| *p = T::from_real(x));
                general_mat_mul(alpha, &c_lam_t, pair, beta, half);

                let mut dst = ArrayViewMut2::from_shape((nmor, nmos), dst).unwrap();
                general_mat_mul(alpha, half, c_sig_s, beta, &mut dst);
            },
        );

    // Contract `\nu \to q` for each `\mu`, `(\mu q|rs) = \sum_\nu C_{\nu q} (\mu\nu|rs)`, stored as
    // `[\mu, q, r, s]`.
    let t2 = &scratch.worka[..nbas * nbas * nrs];
    scratch.workb[..nbas * nqrs]
        .par_chunks_mut(nqrs)
        .zip(t2.par_chunks(nbas * nrs))
        .for_each(|(dst, src)| {
            let src = ArrayView2::from_shape((nbas, nrs), src).unwrap();
            let mut dst = ArrayViewMut2::from_shape((nmoq, nrs), dst).unwrap();
            general_mat_mul(alpha, &c_nu_t, &src, beta, &mut dst);
        });

    // Contract `\mu \to p` directly into `out` viewed as `[p, (q, r, s)]`. The product is split
    // into column blocks, four per thread, which are independent products.
    let t3 = ArrayView2::from_shape((nbas, nqrs), &scratch.workb[..nbas * nqrs]).unwrap();
    let mut out = out.into_shape((nmop, nqrs)).unwrap();
    let chunk = nqrs.div_ceil(4 * rayon::current_num_threads()).max(1);
    out.axis_chunks_iter_mut(Axis(1), chunk)
        .into_par_iter()
        .zip(t3.axis_chunks_iter(Axis(1), chunk))
        .for_each(|(mut block, cols)| {
            general_mat_mul(alpha, &c_mu_t, &cols, beta, &mut block);
        });
}
