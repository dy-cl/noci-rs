// noci/factorise/onebody/gpu/scalar.rs
//! GPU-private scalar helpers for CubeCL one-body kernels.

/// GPU-private complex `f64` representation used when portable complex support is unavailable.
#[derive(Clone, Copy)]
pub(crate) struct GpuComplex64 {
    /// Real component.
    pub(crate) re: f64,
    /// Imaginary component.
    pub(crate) im: f64,
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
}
