// maths/simd/mod.rs
//! Packed real and complex arithmetic for numerical and Wick kernels. The `Simd` trait and its
//! one-lane types are portable, so a kernel written once over `V: Simd<N>` evaluates a single
//! element with `N = 1`; `avx` adds the four- and eight-lane types of x86_64.

#[cfg(target_arch = "x86_64")]
mod avx;

// Public type re-exports.
#[cfg(target_arch = "x86_64")]
pub use avx::{C64x4, C64x8, F64x4, F64x8};

// Restricted type re-exports.
#[cfg(target_arch = "x86_64")]
pub(crate) use avx::Lanes;

// Restricted function re-exports.
#[cfg(target_arch = "x86_64")]
pub(crate) use avx::lanes;

// External crate imports.
use num_complex::Complex64;

/// Packed arithmetic used by numerical and matrix-element kernels. Public so that `NOCIScalar` can
/// name the packed type of each scalar type.
/// `N` is the number of independent scalar lanes represented by one packed value.
/// # Arguments:
/// - None.
/// # Returns
/// - Implementations provide packed construction, arithmetic, loading, and storage operations.
pub trait Simd<const N: usize>: Copy {
    /// Scalar value stored independently in each SIMD lane.
    type Scalar: Copy;

    /// Packed real type with the same number of lanes.
    type Real: Simd<N, Scalar = f64>;

    /// Construct packed additive zero.
    /// # Arguments:
    /// - None.
    /// # Returns
    /// - `Self`: Packed zero.
    fn zero() -> Self;

    /// Construct packed multiplicative one.
    /// # Arguments:
    /// - None.
    /// # Returns
    /// - `Self`: Packed one.
    fn one() -> Self;

    /// Broadcast one scalar value into every lane.
    /// # Arguments:
    /// - `value`: Scalar value to broadcast.
    /// # Returns
    /// - `Self`: Packed broadcast value.
    fn splat(value: Self::Scalar) -> Self;

    /// Load one value for each lane.
    /// # Arguments:
    /// - `values`: Scalar values in lane order.
    /// # Returns
    /// - `Self`: Packed SIMD value.
    fn load(values: &[Self::Scalar; N]) -> Self;

    /// Gather one scalar value for each lane from independent indices.
    /// # Arguments:
    /// - `values`: Contiguous scalar input values.
    /// - `indices`: Scalar indices in lane order.
    /// # Returns
    /// - `Self`: Packed gathered values.
    /// # Safety
    /// - Every index must be in bounds for `values`.
    unsafe fn gather(
        values: &[Self::Scalar],
        indices: &[usize; N],
    ) -> Self;

    /// Store every SIMD lane.
    /// # Arguments:
    /// - `self`: Packed value to store.
    /// - `values`: Scalar output lanes.
    /// # Returns
    /// - `()`: Writes every lane into `values`.
    fn store(
        self,
        values: &mut [Self::Scalar; N],
    );

    /// Add two packed values.
    /// # Arguments:
    /// - `lhs`: Left packed operand.
    /// - `rhs`: Right packed operand.
    /// # Returns
    /// - `Self`: `lhs + rhs`.
    fn add(
        lhs: Self,
        rhs: Self,
    ) -> Self;

    /// Subtract two packed values.
    /// # Arguments:
    /// - `lhs`: Left packed operand.
    /// - `rhs`: Right packed operand.
    /// # Returns
    /// - `Self`: `lhs - rhs`.
    fn sub(
        lhs: Self,
        rhs: Self,
    ) -> Self;

    /// Multiply two packed values.
    /// # Arguments:
    /// - `lhs`: Left packed operand.
    /// - `rhs`: Right packed operand.
    /// # Returns
    /// - `Self`: `lhs * rhs`.
    fn mul(
        lhs: Self,
        rhs: Self,
    ) -> Self;

    /// Accumulate one packed product.
    /// # Arguments:
    /// - `acc`: Packed accumulator.
    /// - `lhs`: Left product operand.
    /// - `rhs`: Right product operand.
    /// # Returns
    /// - `Self`: `acc + lhs * rhs`.
    fn madd(
        acc: Self,
        lhs: Self,
        rhs: Self,
    ) -> Self;

    /// Subtract one packed product from an accumulator.
    /// # Arguments:
    /// - `acc`: Packed accumulator.
    /// - `lhs`: Left product operand.
    /// - `rhs`: Right product operand.
    /// # Returns
    /// - `Self`: `acc - lhs * rhs`.
    fn msub(
        acc: Self,
        lhs: Self,
        rhs: Self,
    ) -> Self;

    /// Multiply every lane by one real scalar.
    /// # Arguments:
    /// - `value`: Packed value to scale.
    /// - `factor`: Real scale factor.
    /// # Returns
    /// - `Self`: `factor * value`.
    fn scale_real(
        value: Self,
        factor: f64,
    ) -> Self;

    /// Multiply each lane by its corresponding real factor.
    /// # Arguments:
    /// - `value`: Packed real or complex values.
    /// - `factors`: Independent real scale factors in lane order.
    /// # Returns
    /// - `Self`: Lane-wise products `factors[lane] * value[lane]`.
    fn multiply_real_lanes(
        value: Self,
        factors: &[f64; N],
    ) -> Self;

    /// Accumulate one packed value scaled by a real scalar.
    /// # Arguments:
    /// - `acc`: Packed accumulator.
    /// - `factor`: Real scale factor.
    /// - `rhs`: Packed operand.
    /// # Returns
    /// - `Self`: `acc + factor * rhs`.
    fn madd_real(
        acc: Self,
        factor: f64,
        rhs: Self,
    ) -> Self;

    /// Accumulate one packed value times packed real lanes.
    /// # Arguments:
    /// - `acc`: Packed accumulator.
    /// - `lhs`: Packed operand.
    /// - `rhs`: Packed real factor of every lane.
    /// # Returns
    /// - `Self`: Lane-wise `acc + lhs * rhs`.
    fn madd_real_lanes(
        acc: Self,
        lhs: Self,
        rhs: Self::Real,
    ) -> Self;
}

/// One `f64` value, the one-lane packed type of the scalar kernels.
#[derive(Clone, Copy)]
pub struct F64x1(f64);

/// One complex value, the one-lane packed type of the scalar kernels.
#[derive(Clone, Copy)]
pub struct C64x1(Complex64);

impl Simd<1> for F64x1 {
    type Scalar = f64;
    type Real = F64x1;

    /// Construct the additive zero.
    /// # Arguments:
    /// - None.
    /// # Returns
    /// - `Self`: Zero.
    #[inline(always)]
    fn zero() -> Self {
        Self(0.0)
    }

    /// Construct the multiplicative one.
    /// # Arguments:
    /// - None.
    /// # Returns
    /// - `Self`: One.
    #[inline(always)]
    fn one() -> Self {
        Self(1.0)
    }

    /// Wrap one real value.
    /// # Arguments:
    /// - `value`: Value to wrap.
    /// # Returns
    /// - `Self`: Wrapped value.
    #[inline(always)]
    fn splat(value: f64) -> Self {
        Self(value)
    }

    /// Load the real value of the lane.
    /// # Arguments:
    /// - `values`: Value of the lane.
    /// # Returns
    /// - `Self`: Wrapped value.
    #[inline(always)]
    fn load(values: &[f64; 1]) -> Self {
        Self(values[0])
    }

    /// Read one real value by index.
    /// # Arguments:
    /// - `values`: Contiguous input values.
    /// - `indices`: Index of the lane.
    /// # Returns
    /// - `Self`: Value at the index.
    /// # Safety
    /// - The index must be in bounds for `values`.
    #[inline(always)]
    unsafe fn gather(
        values: &[f64],
        indices: &[usize; 1],
    ) -> Self {
        Self(unsafe { *values.get_unchecked(indices[0]) })
    }

    /// Store the value of the lane.
    /// # Arguments:
    /// - `self`: Wrapped value.
    /// - `values`: Destination of the lane.
    /// # Returns
    /// - `()`: Writes the lane.
    #[inline(always)]
    fn store(
        self,
        values: &mut [f64; 1],
    ) {
        values[0] = self.0;
    }

    /// Add two values.
    /// # Arguments:
    /// - `lhs`: Left operand.
    /// - `rhs`: Right operand.
    /// # Returns
    /// - `Self`: Sum.
    #[inline(always)]
    fn add(
        lhs: Self,
        rhs: Self,
    ) -> Self {
        Self(lhs.0 + rhs.0)
    }

    /// Subtract two values.
    /// # Arguments:
    /// - `lhs`: Left operand.
    /// - `rhs`: Right operand.
    /// # Returns
    /// - `Self`: Difference.
    #[inline(always)]
    fn sub(
        lhs: Self,
        rhs: Self,
    ) -> Self {
        Self(lhs.0 - rhs.0)
    }

    /// Multiply two values.
    /// # Arguments:
    /// - `lhs`: Left operand.
    /// - `rhs`: Right operand.
    /// # Returns
    /// - `Self`: Product.
    #[inline(always)]
    fn mul(
        lhs: Self,
        rhs: Self,
    ) -> Self {
        Self(lhs.0 * rhs.0)
    }

    /// Accumulate one product, `acc + lhs rhs`, unfused as in the scalar recurrence.
    /// # Arguments:
    /// - `acc`: Accumulator.
    /// - `lhs`: Left factor.
    /// - `rhs`: Right factor.
    /// # Returns
    /// - `Self`: Updated accumulator.
    #[inline(always)]
    fn madd(
        acc: Self,
        lhs: Self,
        rhs: Self,
    ) -> Self {
        Self(acc.0 + lhs.0 * rhs.0)
    }

    /// Subtract one product from an accumulator, `acc - lhs rhs`.
    /// # Arguments:
    /// - `acc`: Accumulator.
    /// - `lhs`: Left factor.
    /// - `rhs`: Right factor.
    /// # Returns
    /// - `Self`: Updated accumulator.
    #[inline(always)]
    fn msub(
        acc: Self,
        lhs: Self,
        rhs: Self,
    ) -> Self {
        Self(acc.0 - lhs.0 * rhs.0)
    }

    /// Multiply by one real scalar.
    /// # Arguments:
    /// - `value`: Value to scale.
    /// - `factor`: Real factor.
    /// # Returns
    /// - `Self`: Scaled value.
    #[inline(always)]
    fn scale_real(
        value: Self,
        factor: f64,
    ) -> Self {
        Self(value.0 * factor)
    }

    /// Multiply by the real factor of the lane.
    /// # Arguments:
    /// - `value`: Value to scale.
    /// - `factors`: Real factor of the lane.
    /// # Returns
    /// - `Self`: Scaled value.
    #[inline(always)]
    fn multiply_real_lanes(
        value: Self,
        factors: &[f64; 1],
    ) -> Self {
        Self(value.0 * factors[0])
    }

    /// Accumulate one value scaled by a real scalar, `acc + factor rhs`.
    /// # Arguments:
    /// - `acc`: Accumulator.
    /// - `factor`: Real factor.
    /// - `rhs`: Value to scale.
    /// # Returns
    /// - `Self`: Updated accumulator.
    #[inline(always)]
    fn madd_real(
        acc: Self,
        factor: f64,
        rhs: Self,
    ) -> Self {
        Self(acc.0 + factor * rhs.0)
    }

    /// Accumulate one value times a real value, `acc + lhs rhs`.
    /// # Arguments:
    /// - `acc`: Accumulator.
    /// - `lhs`: Value to scale.
    /// - `rhs`: Real factor.
    /// # Returns
    /// - `Self`: Updated accumulator.
    #[inline(always)]
    fn madd_real_lanes(
        acc: Self,
        lhs: Self,
        rhs: F64x1,
    ) -> Self {
        Self(acc.0 + lhs.0 * rhs.0)
    }
}

impl Simd<1> for C64x1 {
    type Scalar = Complex64;
    type Real = F64x1;

    /// Construct the additive zero.
    /// # Arguments:
    /// - None.
    /// # Returns
    /// - `Self`: Zero.
    #[inline(always)]
    fn zero() -> Self {
        Self(Complex64::new(0.0, 0.0))
    }

    /// Construct the multiplicative one.
    /// # Arguments:
    /// - None.
    /// # Returns
    /// - `Self`: One.
    #[inline(always)]
    fn one() -> Self {
        Self(Complex64::new(1.0, 0.0))
    }

    /// Wrap one complex value.
    /// # Arguments:
    /// - `value`: Value to wrap.
    /// # Returns
    /// - `Self`: Wrapped value.
    #[inline(always)]
    fn splat(value: Complex64) -> Self {
        Self(value)
    }

    /// Load the complex value of the lane.
    /// # Arguments:
    /// - `values`: Value of the lane.
    /// # Returns
    /// - `Self`: Wrapped value.
    #[inline(always)]
    fn load(values: &[Complex64; 1]) -> Self {
        Self(values[0])
    }

    /// Read one complex value by index.
    /// # Arguments:
    /// - `values`: Contiguous input values.
    /// - `indices`: Index of the lane.
    /// # Returns
    /// - `Self`: Value at the index.
    /// # Safety
    /// - The index must be in bounds for `values`.
    #[inline(always)]
    unsafe fn gather(
        values: &[Complex64],
        indices: &[usize; 1],
    ) -> Self {
        Self(unsafe { *values.get_unchecked(indices[0]) })
    }

    /// Store the value of the lane.
    /// # Arguments:
    /// - `self`: Wrapped value.
    /// - `values`: Destination of the lane.
    /// # Returns
    /// - `()`: Writes the lane.
    #[inline(always)]
    fn store(
        self,
        values: &mut [Complex64; 1],
    ) {
        values[0] = self.0;
    }

    /// Add two values.
    /// # Arguments:
    /// - `lhs`: Left operand.
    /// - `rhs`: Right operand.
    /// # Returns
    /// - `Self`: Sum.
    #[inline(always)]
    fn add(
        lhs: Self,
        rhs: Self,
    ) -> Self {
        Self(lhs.0 + rhs.0)
    }

    /// Subtract two values.
    /// # Arguments:
    /// - `lhs`: Left operand.
    /// - `rhs`: Right operand.
    /// # Returns
    /// - `Self`: Difference.
    #[inline(always)]
    fn sub(
        lhs: Self,
        rhs: Self,
    ) -> Self {
        Self(lhs.0 - rhs.0)
    }

    /// Multiply two values.
    /// # Arguments:
    /// - `lhs`: Left operand.
    /// - `rhs`: Right operand.
    /// # Returns
    /// - `Self`: Product.
    #[inline(always)]
    fn mul(
        lhs: Self,
        rhs: Self,
    ) -> Self {
        Self(lhs.0 * rhs.0)
    }

    /// Accumulate one product, `acc + lhs rhs`, unfused as in the scalar recurrence.
    /// # Arguments:
    /// - `acc`: Accumulator.
    /// - `lhs`: Left factor.
    /// - `rhs`: Right factor.
    /// # Returns
    /// - `Self`: Updated accumulator.
    #[inline(always)]
    fn madd(
        acc: Self,
        lhs: Self,
        rhs: Self,
    ) -> Self {
        Self(acc.0 + lhs.0 * rhs.0)
    }

    /// Subtract one product from an accumulator, `acc - lhs rhs`.
    /// # Arguments:
    /// - `acc`: Accumulator.
    /// - `lhs`: Left factor.
    /// - `rhs`: Right factor.
    /// # Returns
    /// - `Self`: Updated accumulator.
    #[inline(always)]
    fn msub(
        acc: Self,
        lhs: Self,
        rhs: Self,
    ) -> Self {
        Self(acc.0 - lhs.0 * rhs.0)
    }

    /// Multiply by one real scalar.
    /// # Arguments:
    /// - `value`: Value to scale.
    /// - `factor`: Real factor.
    /// # Returns
    /// - `Self`: Scaled value.
    #[inline(always)]
    fn scale_real(
        value: Self,
        factor: f64,
    ) -> Self {
        Self(value.0 * factor)
    }

    /// Multiply by the real factor of the lane.
    /// # Arguments:
    /// - `value`: Value to scale.
    /// - `factors`: Real factor of the lane.
    /// # Returns
    /// - `Self`: Scaled value.
    #[inline(always)]
    fn multiply_real_lanes(
        value: Self,
        factors: &[f64; 1],
    ) -> Self {
        Self(value.0 * factors[0])
    }

    /// Accumulate one value scaled by a real scalar, `acc + factor rhs`.
    /// # Arguments:
    /// - `acc`: Accumulator.
    /// - `factor`: Real factor.
    /// - `rhs`: Value to scale.
    /// # Returns
    /// - `Self`: Updated accumulator.
    #[inline(always)]
    fn madd_real(
        acc: Self,
        factor: f64,
        rhs: Self,
    ) -> Self {
        Self(acc.0 + rhs.0 * factor)
    }

    /// Accumulate one value times a real value, `acc + lhs rhs`.
    /// # Arguments:
    /// - `acc`: Accumulator.
    /// - `lhs`: Value to scale.
    /// - `rhs`: Real factor.
    /// # Returns
    /// - `Self`: Updated accumulator.
    #[inline(always)]
    fn madd_real_lanes(
        acc: Self,
        lhs: Self,
        rhs: F64x1,
    ) -> Self {
        Self(acc.0 + lhs.0 * rhs.0)
    }
}
