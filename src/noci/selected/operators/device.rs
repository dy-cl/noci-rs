// noci/selected/operators/device.rs
//! GPU counterpart of `apply_factorised_shifted_omega_m` in `operators.rs`, on vectors resident on
//! the GPU that holds the one-body operator. Real Krylov scalar without an imaginary shift only.

// External crate imports.
use cudarc::driver::CudaSlice;

// Crate-root imports.
use crate::maths::gmres::{DeviceKrylov, KrylovSpace};
use crate::noci::{DeviceOneBody, DeviceOneBodyScratch};

// Parent/sibling imports.
use super::super::DevicePT2Projection;

/// Apply factorised projected `(F - E_0 S)x` on the GPU, the counterpart of
/// `apply_factorised_shifted_omega_m` with `\epsilon = 0`.
/// # Arguments:
/// - `space`: GPU vector space.
/// - `p`: Projection contractions on the GPU.
/// - `one_body`: Spin-factorised one-body operator on the GPU.
/// - `scratch`: First-contraction buffers of the one-body operator.
/// - `x`: Krylov vector to apply the operator to.
/// # Returns:
/// - `CudaSlice<f64>`: Matrix-vector product `M^Omega x`.
pub(in crate::noci::selected) fn apply_device_factorised_shifted_omega_m(
    space: &DeviceKrylov,
    p: &DevicePT2Projection,
    one_body: &DeviceOneBody,
    scratch: &mut DeviceOneBodyScratch,
    x: &CudaSlice<f64>,
) -> CudaSlice<f64> {
    // `M^Omega = F - E_0 S - F_{a0} S_{0b} - S_{a0} F_{0b} + 2 E_0 S_{a0} S_{0b}`.
    let sx = space.inner_product(&p.s_a0, x);
    let fx = space.inner_product(&p.f_a0, x);
    let mut y = one_body
        .apply_one_body(x, -p.e0, scratch)
        .expect("GPU one-body product failed");

    space.scaled_add(-sx, &p.f_a0, &mut y);
    space.scaled_add(2.0 * p.e0 * sx - fx, &p.s_a0, &mut y);
    y
}
