// gpu/scalar.rs
//! GPU-private scalar helpers for CubeCL kernels.

// External crate imports.
use num_complex::Complex64;

/// Host-to-GPU scalar conversion used by accelerator packing code.
pub(crate) trait GpuScalar: Copy {
    /// GPU device-side scalar representation.
    type Device: Copy;

    /// Convert one host scalar to its GPU representation.
    /// # Arguments:
    /// - `self`: Host scalar.
    /// # Returns
    /// - `Self::Device`: GPU scalar representation.
    fn to_gpu(self) -> Self::Device;

    /// Convert one GPU scalar representation to the host scalar.
    /// # Arguments:
    /// - `value`: GPU scalar representation.
    /// # Returns
    /// - `Self`: Host scalar.
    fn from_gpu(value: Self::Device) -> Self;
}

impl GpuScalar for f64 {
    type Device = f64;

    /// Convert one host `f64` to the GPU `f64` fast path.
    /// # Arguments:
    /// - `self`: Host scalar.
    /// # Returns
    /// - `f64`: Device scalar.
    fn to_gpu(self) -> Self::Device {
        self
    }

    /// Convert one GPU `f64` to the host scalar.
    /// # Arguments:
    /// - `value`: Device scalar.
    /// # Returns
    /// - `f64`: Host scalar.
    fn from_gpu(value: Self::Device) -> Self {
        value
    }
}

/// GPU-private complex `f64` representation used when portable complex support is unavailable.
#[derive(Clone, Copy)]
pub(crate) struct GpuComplex64 {
    /// Real component.
    pub(crate) re: f64,
    /// Imaginary component.
    pub(crate) im: f64,
}

impl GpuScalar for Complex64 {
    type Device = GpuComplex64;

    /// Convert one host `Complex64` to GPU-private complex storage.
    /// # Arguments:
    /// - `self`: Host scalar.
    /// # Returns
    /// - `GpuComplex64`: Device scalar.
    fn to_gpu(self) -> Self::Device {
        GpuComplex64 {
            re: self.re,
            im: self.im,
        }
    }

    /// Convert one GPU-private complex scalar to host `Complex64`.
    /// # Arguments:
    /// - `value`: Device scalar.
    /// # Returns
    /// - `Complex64`: Host scalar.
    fn from_gpu(value: Self::Device) -> Self {
        Complex64::new(value.re, value.im)
    }
}

impl GpuComplex64 {
    /// Return `0 + 0i`.
    /// # Returns
    /// - `GpuComplex64`: Additive identity.
    pub(crate) fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    /// Return `x + 0i`.
    /// # Arguments:
    /// - `x`: Real component.
    /// # Returns
    /// - `GpuComplex64`: Complex value with zero imaginary component.
    pub(crate) fn from_real(x: f64) -> Self {
        Self { re: x, im: 0.0 }
    }

    /// Return `0 + ix`.
    /// # Arguments:
    /// - `x`: Imaginary component.
    /// # Returns
    /// - `GpuComplex64`: Pure imaginary complex value.
    pub(crate) fn from_imag(x: f64) -> Self {
        Self { re: 0.0, im: x }
    }

    /// Add two complex values, `z = x + y`.
    /// # Arguments:
    /// - `self`: Left operand.
    /// - `rhs`: Right operand.
    /// # Returns
    /// - `GpuComplex64`: Sum.
    pub(crate) fn add(
        self,
        rhs: Self,
    ) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }

    /// Subtract two complex values, `z = x - y`.
    /// # Arguments:
    /// - `self`: Left operand.
    /// - `rhs`: Right operand.
    /// # Returns
    /// - `GpuComplex64`: Difference.
    pub(crate) fn sub(
        self,
        rhs: Self,
    ) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }

    /// Multiply two complex values, `z = xy`.
    /// # Arguments:
    /// - `self`: Left operand.
    /// - `rhs`: Right operand.
    /// # Returns
    /// - `GpuComplex64`: Product.
    pub(crate) fn mul(
        self,
        rhs: Self,
    ) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }

    /// Negate one complex value, `z = -x`.
    /// # Arguments:
    /// - `self`: Operand.
    /// # Returns
    /// - `GpuComplex64`: Negated value.
    pub(crate) fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}
