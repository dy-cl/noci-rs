// maths/gmres/device.rs
//! `DeviceKrylov`, the GPU counterpart of `HostKrylov` in `gmres.rs`: GMRES vectors resident on a
//! GPU, with their arithmetic in cuBLAS. The Arnoldi basis is one column-major
//! `n x (restart + 1)` matrix, so both classical Gram-Schmidt passes and the solution update are
//! single matrix-vector products. Real vectors only. cuBLAS writes the scalar results of inner
//! products and norms to the host, so those calls wait for the stream.

// Standard library imports.
use std::sync::Arc;

// External crate imports.
use cudarc::cublas::result::CublasError;
use cudarc::cublas::sys::{self, cublasOperation_t, cublasSideMode_t};
use cudarc::cublas::{CudaBlas, Gemv, GemvConfig};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};

// Crate-root imports.
use crate::gpu::Device;

// Parent/sibling imports.
use super::KrylovSpace;

/// GMRES vectors on a GPU.
pub(crate) struct DeviceKrylov {
    /// Stream on which every operation is issued, shared with the operator GMRES applies.
    stream: Arc<CudaStream>,
    /// cuBLAS handle bound to `stream`.
    blas: CudaBlas,
}

/// Arnoldi basis of one restart cycle on a GPU.
pub(crate) struct DeviceBasis {
    /// Basis vectors as the columns of a column-major `n x capacity` matrix.
    q: CudaSlice<f64>,
    /// Vector length.
    n: usize,
    /// Number of basis vectors held.
    count: usize,
}

impl DeviceKrylov {
    /// Bind a cuBLAS handle to a GPU's stream.
    /// # Arguments:
    /// - `device`: GPU whose stream also carries the operator GMRES applies.
    /// # Returns:
    /// - `Result<DeviceKrylov, CublasError>`: The vector space, or the cuBLAS error.
    pub(crate) fn new(device: &Device) -> Result<Self, CublasError> {
        let stream = device.stream.clone();
        let blas = CudaBlas::new(stream.clone())?;

        Ok(Self { stream, blas })
    }

    /// Copy a host vector to the GPU.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `v`: Host vector.
    /// # Returns:
    /// - `CudaSlice<f64>`: The vector on the GPU.
    pub(crate) fn upload(
        &self,
        v: &[f64],
    ) -> CudaSlice<f64> {
        self.stream
            .clone_htod(v)
            .expect("failed to copy a vector to the GPU")
    }

    /// Copy a vector back to the host.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `v`: Vector on the GPU.
    /// # Returns:
    /// - `Vec<f64>`: The vector on the host.
    pub(crate) fn download(
        &self,
        v: &CudaSlice<f64>,
    ) -> Vec<f64> {
        self.stream
            .clone_dtoh(v)
            .expect("failed to copy a vector from the GPU")
    }

    /// Elementwise product `d \odot v`, a diagonal matrix applied to a vector.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `d`: Diagonal.
    /// - `v`: Vector.
    /// # Returns:
    /// - `CudaSlice<f64>`: The product.
    pub(crate) fn multiply(
        &self,
        d: &CudaSlice<f64>,
        v: &CudaSlice<f64>,
    ) -> CudaSlice<f64> {
        let n = v.len();
        let mut y = self.zeros(n);
        {
            let (d, _d) = d.device_ptr(&self.stream);
            let (v, _v) = v.device_ptr(&self.stream);
            let (out, _out) = y.device_ptr_mut(&self.stream);
            // `diag(d) V` with `V` the `n x 1` matrix holding `v`.
            unsafe {
                sys::cublasDdgmm(
                    *self.blas.handle(),
                    cublasSideMode_t::CUBLAS_SIDE_LEFT,
                    n as i32,
                    1,
                    v as *const f64,
                    n as i32,
                    d as *const f64,
                    1,
                    out as *mut f64,
                    n as i32,
                )
            }
            .result()
            .expect("cuBLAS diagonal product failed");
        }
        y
    }
}

impl KrylovSpace<f64> for DeviceKrylov {
    type Vector = CudaSlice<f64>;
    type Basis = DeviceBasis;

    /// Zero vector.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `n`: Vector length.
    /// # Returns:
    /// - `CudaSlice<f64>`: Zero vector of length `n`.
    /// # Panics
    /// - If the GPU operation fails.
    fn zeros(
        &self,
        n: usize,
    ) -> CudaSlice<f64> {
        self.stream
            .alloc_zeros(n)
            .expect("failed to allocate a GPU vector")
    }

    /// Length of a vector.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `v`: Vector.
    /// # Returns:
    /// - `usize`: Number of entries of `v`.
    fn length(
        &self,
        v: &CudaSlice<f64>,
    ) -> usize {
        v.len()
    }

    /// Copy of a vector.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `v`: Vector to copy.
    /// # Returns:
    /// - `CudaSlice<f64>`: Independent copy of `v`.
    /// # Panics
    /// - If the GPU operation fails.
    fn copy(
        &self,
        v: &CudaSlice<f64>,
    ) -> CudaSlice<f64> {
        self.stream
            .clone_dtod(v)
            .expect("failed to copy a GPU vector")
    }

    /// Conjugated inner product `x^\dagger y` by cuBLAS, which waits for the stream.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `x`: Left vector.
    /// - `y`: Right vector.
    /// # Returns:
    /// - `f64`: Inner product `x^\dagger y`.
    /// # Panics
    /// - If the GPU operation fails.
    fn inner_product(
        &self,
        x: &CudaSlice<f64>,
        y: &CudaSlice<f64>,
    ) -> f64 {
        let mut result = 0.0;
        let (xp, _x) = x.device_ptr(&self.stream);
        let (yp, _y) = y.device_ptr(&self.stream);
        unsafe {
            sys::cublasDdot_v2(
                *self.blas.handle(),
                x.len() as i32,
                xp as *const f64,
                1,
                yp as *const f64,
                1,
                &mut result,
            )
        }
        .result()
        .expect("cuBLAS inner product failed");
        result
    }

    /// Euclidean norm by cuBLAS, which waits for the stream.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `v`: Vector to norm.
    /// # Returns:
    /// - `f64`: Euclidean norm of `v`.
    /// # Panics
    /// - If the GPU operation fails.
    fn vector_norm(
        &self,
        v: &CudaSlice<f64>,
    ) -> f64 {
        let mut result = 0.0;
        let (vp, _v) = v.device_ptr(&self.stream);
        unsafe {
            sys::cublasDnrm2_v2(
                *self.blas.handle(),
                v.len() as i32,
                vp as *const f64,
                1,
                &mut result,
            )
        }
        .result()
        .expect("cuBLAS norm failed");
        result
    }

    /// Scale a vector in place, `v \leftarrow a v`.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `a`: Scale factor.
    /// - `v`: Vector to scale.
    /// # Returns:
    /// - `()`: Scales `v` in place.
    /// # Panics
    /// - If the GPU operation fails.
    fn scale(
        &self,
        a: f64,
        v: &mut CudaSlice<f64>,
    ) {
        let n = v.len();
        let (vp, _v) = v.device_ptr_mut(&self.stream);
        unsafe { sys::cublasDscal_v2(*self.blas.handle(), n as i32, &a, vp as *mut f64, 1) }
            .result()
            .expect("cuBLAS scale failed");
    }

    /// Add a scaled vector in place, `y \leftarrow y + a x`.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `a`: Scale factor.
    /// - `x`: Vector to add.
    /// - `y`: Vector to update.
    /// # Returns:
    /// - `()`: Updates `y` in place.
    /// # Panics
    /// - If the GPU operation fails.
    fn scaled_add(
        &self,
        a: f64,
        x: &CudaSlice<f64>,
        y: &mut CudaSlice<f64>,
    ) {
        let (xp, _x) = x.device_ptr(&self.stream);
        let (yp, _y) = y.device_ptr_mut(&self.stream);
        unsafe {
            sys::cublasDaxpy_v2(
                *self.blas.handle(),
                x.len() as i32,
                &a,
                xp as *const f64,
                1,
                yp as *mut f64,
                1,
            )
        }
        .result()
        .expect("cuBLAS scaled addition failed");
    }

    /// Start the Arnoldi basis of a restart cycle with `q_0 = r / \beta`.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `r`: Residual at the start of the cycle.
    /// - `beta`: Norm of `r`.
    /// - `capacity`: Largest number of basis vectors the cycle can hold.
    /// # Returns:
    /// - `DeviceBasis`: Basis holding `q_0`.
    /// # Panics
    /// - If the GPU operation fails.
    fn basis(
        &self,
        r: &CudaSlice<f64>,
        beta: f64,
        capacity: usize,
    ) -> DeviceBasis {
        let n = r.len();
        let mut q = DeviceBasis {
            q: self.zeros(n * capacity),
            n,
            count: 0,
        };
        self.extend(&mut q, self.copy(r), beta);
        q
    }

    /// Normalise an orthogonalised Arnoldi vector and append it to the basis.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `q`: Arnoldi basis.
    /// - `w`: Orthogonalised Arnoldi vector.
    /// - `norm`: Norm of `w`.
    /// # Returns:
    /// - `()`: Appends `w / norm` to `q`.
    /// # Panics
    /// - If the GPU operation fails.
    fn extend(
        &self,
        q: &mut DeviceBasis,
        w: CudaSlice<f64>,
        norm: f64,
    ) {
        let (n, k) = (q.n, q.count);
        let mut column = q.q.slice_mut(k * n..(k + 1) * n);
        self.stream
            .memcpy_dtod(&w, &mut column)
            .expect("failed to copy a GPU basis vector");

        let scale = 1.0 / norm;
        let (cp, _c) = column.device_ptr_mut(&self.stream);
        unsafe { sys::cublasDscal_v2(*self.blas.handle(), n as i32, &scale, cp as *mut f64, 1) }
            .result()
            .expect("cuBLAS scale failed");

        q.count += 1;
    }

    /// Apply the right preconditioner to one basis vector, `z_k = M^{-1} q_k`, from a copy of the basis column.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `q`: Arnoldi basis.
    /// - `k`: Index of the basis vector.
    /// - `precondition`: Right preconditioner `M^{-1}`.
    /// # Returns:
    /// - `CudaSlice<f64>`: Preconditioned vector `z_k`.
    /// # Panics
    /// - If the GPU operation fails.
    fn precondition<P>(
        &self,
        q: &DeviceBasis,
        k: usize,
        precondition: &P,
    ) -> CudaSlice<f64>
    where
        P: Fn(&CudaSlice<f64>) -> CudaSlice<f64>,
    {
        let column = self
            .stream
            .clone_dtod(&q.q.slice(k * q.n..(k + 1) * q.n))
            .expect("failed to copy a GPU basis vector");
        precondition(&column)
    }

    /// Orthogonalise an Arnoldi vector against the first `k + 1` basis vectors by classical Gram-Schmidt applied twice; each pass is two matrix-vector products over the leading basis columns.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `q`: Arnoldi basis.
    /// - `w`: Arnoldi vector to orthogonalise in place.
    /// - `k`: Current Arnoldi iteration in the restart cycle.
    /// # Returns:
    /// - `Vec<f64>`: Hessenberg column `h_{jk}`, `j \le k`, the sum of both passes.
    /// # Panics
    /// - If the GPU operation fails.
    fn orthogonalise(
        &self,
        q: &DeviceBasis,
        w: &mut CudaSlice<f64>,
        k: usize,
    ) -> Vec<f64> {
        let (n, m) = (q.n as i32, k as i32 + 1);
        let mut h = vec![0.0; k + 1];
        let mut pass = self.zeros(k + 1);

        for _ in 0..2 {
            // `h = Q^T w`.
            let product = GemvConfig {
                trans: cublasOperation_t::CUBLAS_OP_T,
                m: n,
                n: m,
                alpha: 1.0,
                lda: n,
                incx: 1,
                beta: 0.0,
                incy: 1,
            };
            unsafe { self.blas.gemv(product, &q.q, &*w, &mut pass) }
                .expect("cuBLAS basis product failed");

            // `w \leftarrow w - Q h`.
            let update = GemvConfig {
                trans: cublasOperation_t::CUBLAS_OP_N,
                m: n,
                n: m,
                alpha: -1.0,
                lda: n,
                incx: 1,
                beta: 1.0,
                incy: 1,
            };
            unsafe { self.blas.gemv(update, &q.q, &pass, w) }.expect("cuBLAS basis update failed");

            for (hj, pj) in h.iter_mut().zip(self.download(&pass)) {
                *hj += pj;
            }
        }

        h
    }

    /// Combine basis vectors, `u = \sum_j y_j q_j` in one matrix-vector product.
    /// # Arguments:
    /// - `self`: Vector space.
    /// - `q`: Arnoldi basis.
    /// - `y`: Expansion coefficients, one per leading basis vector.
    /// - `n`: Vector length.
    /// # Returns:
    /// - `CudaSlice<f64>`: Combination `u`.
    /// # Panics
    /// - If the GPU operation fails.
    fn combine(
        &self,
        q: &DeviceBasis,
        y: &[f64],
        n: usize,
    ) -> CudaSlice<f64> {
        let coefficients = self.upload(y);
        let mut u = self.zeros(n);

        // `u = Q y` over the leading `len(y)` columns of the basis.
        let combination = GemvConfig {
            trans: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: y.len() as i32,
            alpha: 1.0,
            lda: n as i32,
            incx: 1,
            beta: 0.0,
            incy: 1,
        };
        unsafe { self.blas.gemv(combination, &q.q, &coefficients, &mut u) }
            .expect("cuBLAS basis combination failed");
        u
    }
}
