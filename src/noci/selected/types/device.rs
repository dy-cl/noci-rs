// noci/selected/types/device.rs
//! GPU counterparts of `PT2Projection` and `Preconditioner` in `types.rs`, for a real Krylov
//! scalar. A real projection has `S_{0a} = S_{a0}` and `F_{0a} = F_{a0}`, so only `S_{a0}` and
//! `F_{a0}` are uploaded.

// External crate imports.
use cudarc::driver::CudaSlice;
use ndarray::Array1;

// Crate-root imports.
use crate::NOCIScalar;
use crate::maths::gmres::{DeviceKrylov, KrylovSpace};

// Parent/sibling imports.
use super::{PT2Projection, Preconditioner};

/// Projection contractions on a GPU, the counterpart of `PT2Projection`.
pub(in crate::noci::selected) struct DevicePT2Projection {
    /// Zeroth-order NOCI generalised-Fock energy.
    pub(in crate::noci::selected) e0: f64,
    /// Candidate-reference overlap `S_{a0}`, equal to `S_{0a}`.
    pub(in crate::noci::selected) s_a0: CudaSlice<f64>,
    /// Candidate-reference Fock contraction `F_{a0}`, equal to `F_{0a}`.
    pub(in crate::noci::selected) f_a0: CudaSlice<f64>,
}

/// Rank-2 Woodbury preconditioner on a GPU, the counterpart of `Preconditioner`.
pub(in crate::noci::selected) struct DevicePreconditioner {
    /// Inverse diagonal of the unprojected candidate-candidate matrix.
    dinv: CudaSlice<f64>,
    /// First diagonal-scaled left update vector `D^{-1} u_0`.
    z0: CudaSlice<f64>,
    /// Second diagonal-scaled left update vector `D^{-1} u_1`.
    z1: CudaSlice<f64>,
    /// First right update vector.
    v0: CudaSlice<f64>,
    /// Second right update vector.
    v1: CudaSlice<f64>,
    /// `(0,0)` element of the inverse Woodbury core.
    w00: f64,
    /// `(0,1)` element of the inverse Woodbury core.
    w01: f64,
    /// `(1,0)` element of the inverse Woodbury core.
    w10: f64,
    /// `(1,1)` element of the inverse Woodbury core.
    w11: f64,
    /// Whether the rank-2 Woodbury correction is numerically safe to apply.
    active: bool,
}

impl DevicePT2Projection {
    /// Upload the real parts of the projection contractions.
    /// # Arguments:
    /// - `space`: GPU vector space.
    /// - `p`: Projection contractions of a real Krylov scalar.
    /// # Returns:
    /// - `DevicePT2Projection`: The contractions on the GPU.
    pub(in crate::noci::selected) fn new<R: NOCIScalar>(
        space: &DeviceKrylov,
        p: &PT2Projection<R>,
    ) -> Self {
        let real = |v: &Array1<R>| v.iter().map(|x| x.re()).collect::<Vec<f64>>();

        Self {
            e0: p.e0,
            s_a0: space.upload(&real(&p.s_a0)),
            f_a0: space.upload(&real(&p.f_a0)),
        }
    }
}

impl DevicePreconditioner {
    /// Upload the real parts of a host preconditioner.
    /// # Arguments:
    /// - `space`: GPU vector space.
    /// - `p`: Preconditioner of a real Krylov scalar.
    /// # Returns:
    /// - `DevicePreconditioner`: The preconditioner on the GPU.
    pub(in crate::noci::selected) fn new<R: NOCIScalar>(
        space: &DeviceKrylov,
        p: &Preconditioner<R>,
    ) -> Self {
        let real = |v: &Array1<R>| v.iter().map(|x| x.re()).collect::<Vec<f64>>();

        Self {
            dinv: space.upload(&real(&p.dinv)),
            z0: space.upload(&real(&p.z0)),
            z1: space.upload(&real(&p.z1)),
            v0: space.upload(&real(&p.v0)),
            v1: space.upload(&real(&p.v1)),
            w00: p.w00.re(),
            w01: p.w01.re(),
            w10: p.w10.re(),
            w11: p.w11.re(),
            active: p.active,
        }
    }

    /// Apply the rank-2 Woodbury preconditioner to a vector, the counterpart of
    /// `Preconditioner::apply`.
    /// # Arguments:
    /// - `self`: Preconditioner on the GPU.
    /// - `space`: GPU vector space.
    /// - `v`: Vector to precondition.
    /// # Returns:
    /// - `CudaSlice<f64>`: Approximate action of `(M^Omega)^{-1} v`.
    pub(in crate::noci::selected) fn apply(
        &self,
        space: &DeviceKrylov,
        v: &CudaSlice<f64>,
    ) -> CudaSlice<f64> {
        let mut y = space.multiply(&self.dinv, v);

        if !self.active {
            return y;
        }

        // `(D + U V^T)^{-1}v = D^{-1}v - D^{-1}U C^{-1}V^T D^{-1}v`.
        let t0 = space.inner_product(&self.v0, &y);
        let t1 = space.inner_product(&self.v1, &y);

        let c0 = self.w00 * t0 + self.w01 * t1;
        let c1 = self.w10 * t0 + self.w11 * t1;

        space.scaled_add(-c0, &self.z0, &mut y);
        space.scaled_add(-c1, &self.z1, &mut y);
        y
    }
}
