// maths/simd.rs
//! Packed real and complex arithmetic for numerical and Wick kernels.

// Standard library imports.
use std::arch::x86_64::{
    __m256d, __m512d, _mm256_add_pd, _mm256_fmadd_pd, _mm256_fmsub_pd, _mm256_fnmadd_pd,
    _mm256_loadu_pd, _mm256_mul_pd, _mm256_set_pd, _mm256_set1_pd, _mm256_setzero_pd,
    _mm256_storeu_pd, _mm256_sub_pd, _mm512_add_pd, _mm512_fmadd_pd, _mm512_fmsub_pd,
    _mm512_fnmadd_pd, _mm512_loadu_pd, _mm512_mul_pd, _mm512_set_pd, _mm512_set1_pd,
    _mm512_setzero_pd, _mm512_storeu_pd, _mm512_sub_pd,
};

// External crate imports.
use num_complex::Complex64;

/// Packed arithmetic used by SIMD numerical and Wick kernels.
/// `N` is the number of independent scalar lanes represented by one packed value.
/// # Arguments:
/// - None.
/// # Returns
/// - Implementations provide packed construction, arithmetic, loading, and storage operations.
pub(crate) trait Simd<const N: usize>: Copy {
    /// Scalar value stored independently in each SIMD lane.
    type Scalar: Copy;
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
}

/// Four packed `f64` values in one AVX2 register.
#[derive(Clone, Copy)]
pub(crate) struct F64x4(__m256d);

/// Four packed complex values in split real/imaginary AVX2 registers.
#[derive(Clone, Copy)]
pub(crate) struct C64x4 {
    re: __m256d,
    im: __m256d,
}

/// Eight packed `f64` values in one AVX-512 register.
#[derive(Clone, Copy)]
pub(crate) struct F64x8(__m512d);

/// Eight packed complex values in split real/imaginary AVX-512 registers.
#[derive(Clone, Copy)]
pub(crate) struct C64x8 {
    re: __m512d,
    im: __m512d,
}

impl F64x4 {
    /// Construct zero packed real values.
    /// # Arguments:
    /// - None.
    /// # Returns
    /// - `F64x4`: Packed zero.
    #[inline(always)]
    pub(super) fn zero() -> Self {
        unsafe { Self(_mm256_setzero_pd()) }
    }

    /// Broadcast one real scalar to all lanes.
    /// # Arguments:
    /// - `value`: Real scalar to broadcast.
    /// # Returns
    /// - `F64x4`: Packed broadcast value.
    #[inline(always)]
    pub(super) fn splat(value: f64) -> Self {
        unsafe { Self(_mm256_set1_pd(value)) }
    }

    /// Load four real lane values.
    /// # Arguments:
    /// - `values`: Real lane values.
    /// # Returns
    /// - `F64x4`: Packed values.
    #[inline(always)]
    pub(super) fn load(values: &[f64; 4]) -> Self {
        unsafe { Self(_mm256_loadu_pd(values.as_ptr())) }
    }

    /// Store four real lane values.
    /// # Arguments:
    /// - `values`: Real output lanes.
    /// # Returns
    /// - `()`: Writes packed lanes into `values`.
    #[inline(always)]
    pub(super) fn store(
        self,
        values: &mut [f64; 4],
    ) {
        unsafe { _mm256_storeu_pd(values.as_mut_ptr(), self.0) }
    }

    /// Add packed real values.
    /// # Arguments:
    /// - `a`: Left packed operand.
    /// - `b`: Right packed operand.
    /// # Returns
    /// - `F64x4`: `a + b`.
    #[inline(always)]
    pub(super) fn add(
        a: Self,
        b: Self,
    ) -> Self {
        unsafe { Self(_mm256_add_pd(a.0, b.0)) }
    }

    /// Subtract packed real values.
    /// # Arguments:
    /// - `a`: Left packed operand.
    /// - `b`: Right packed operand.
    /// # Returns
    /// - `F64x4`: `a - b`.
    #[inline(always)]
    pub(super) fn sub(
        a: Self,
        b: Self,
    ) -> Self {
        unsafe { Self(_mm256_sub_pd(a.0, b.0)) }
    }

    /// Multiply packed real values.
    /// # Arguments:
    /// - `a`: Left packed operand.
    /// - `b`: Right packed operand.
    /// # Returns
    /// - `F64x4`: `a * b`.
    #[inline(always)]
    pub(super) fn mul(
        a: Self,
        b: Self,
    ) -> Self {
        unsafe { Self(_mm256_mul_pd(a.0, b.0)) }
    }

    /// Accumulate packed real product, `acc + a * b`.
    /// # Arguments:
    /// - `acc`: Packed accumulator.
    /// - `a`: Left packed product operand.
    /// - `b`: Right packed product operand.
    /// # Returns
    /// - `F64x4`: `acc + a * b`.
    #[inline(always)]
    pub(super) fn madd(
        acc: Self,
        a: Self,
        b: Self,
    ) -> Self {
        unsafe { Self(_mm256_fmadd_pd(a.0, b.0, acc.0)) }
    }

    /// Accumulate negative packed real product, `acc - a * b`.
    /// # Arguments:
    /// - `acc`: Packed accumulator.
    /// - `a`: Left packed product operand.
    /// - `b`: Right packed product operand.
    /// # Returns
    /// - `F64x4`: `acc - a * b`.
    #[inline(always)]
    pub(super) fn msub(
        acc: Self,
        a: Self,
        b: Self,
    ) -> Self {
        unsafe { Self(_mm256_fnmadd_pd(a.0, b.0, acc.0)) }
    }
}

impl C64x4 {
    /// Construct zero packed complex values.
    /// # Arguments:
    /// - None.
    /// # Returns
    /// - `C64x4`: Packed complex zero.
    #[inline(always)]
    pub(super) fn zero() -> Self {
        unsafe {
            Self {
                re: _mm256_setzero_pd(),
                im: _mm256_setzero_pd(),
            }
        }
    }

    /// Broadcast one complex scalar to all lanes.
    /// # Arguments:
    /// - `re`: Real part to broadcast.
    /// - `im`: Imaginary part to broadcast.
    /// # Returns
    /// - `C64x4`: Packed complex broadcast value.
    #[inline(always)]
    pub(super) fn splat(
        re: f64,
        im: f64,
    ) -> Self {
        unsafe {
            Self {
                re: _mm256_set1_pd(re),
                im: _mm256_set1_pd(im),
            }
        }
    }

    /// Construct four packed complex values from independent scalar lanes.
    /// # Arguments:
    /// - `v0`: Lane 0 value.
    /// - `v1`: Lane 1 value.
    /// - `v2`: Lane 2 value.
    /// - `v3`: Lane 3 value.
    /// # Returns
    /// - `C64x4`: Packed complex values `[v0,v1,v2,v3]`.
    #[inline(always)]
    pub(super) fn from_values(
        v0: Complex64,
        v1: Complex64,
        v2: Complex64,
        v3: Complex64,
    ) -> Self {
        unsafe {
            Self {
                re: _mm256_set_pd(v3.re, v2.re, v1.re, v0.re),
                im: _mm256_set_pd(v3.im, v2.im, v1.im, v0.im),
            }
        }
    }

    /// Store four split complex lane values.
    /// # Arguments:
    /// - `re`: Real output lanes.
    /// - `im`: Imaginary output lanes.
    /// # Returns
    /// - `()`: Writes packed complex lanes into `re` and `im`.
    #[inline(always)]
    pub(super) fn store(
        self,
        re: &mut [f64; 4],
        im: &mut [f64; 4],
    ) {
        unsafe {
            _mm256_storeu_pd(re.as_mut_ptr(), self.re);
            _mm256_storeu_pd(im.as_mut_ptr(), self.im);
        }
    }

    /// Add packed complex values.
    /// # Arguments:
    /// - `a`: Left packed operand.
    /// - `b`: Right packed operand.
    /// # Returns
    /// - `C64x4`: `a + b`.
    #[inline(always)]
    pub(super) fn add(
        a: Self,
        b: Self,
    ) -> Self {
        unsafe {
            Self {
                re: _mm256_add_pd(a.re, b.re),
                im: _mm256_add_pd(a.im, b.im),
            }
        }
    }

    /// Subtract packed complex values.
    /// # Arguments:
    /// - `a`: Left packed operand.
    /// - `b`: Right packed operand.
    /// # Returns
    /// - `C64x4`: `a - b`.
    #[inline(always)]
    pub(super) fn sub(
        a: Self,
        b: Self,
    ) -> Self {
        unsafe {
            Self {
                re: _mm256_sub_pd(a.re, b.re),
                im: _mm256_sub_pd(a.im, b.im),
            }
        }
    }

    /// Multiply packed complex values with four real products.
    /// # Arguments:
    /// - `a`: Left packed complex operand.
    /// - `b`: Right packed complex operand.
    /// # Returns
    /// - `C64x4`: `a * b`.
    #[inline(always)]
    pub(super) fn mul(
        a: Self,
        b: Self,
    ) -> Self {
        unsafe {
            Self {
                re: _mm256_fmsub_pd(a.re, b.re, _mm256_mul_pd(a.im, b.im)),
                im: _mm256_fmadd_pd(a.re, b.im, _mm256_mul_pd(a.im, b.re)),
            }
        }
    }

    /// Accumulate packed complex product, `acc + a * b`.
    /// # Arguments:
    /// - `acc`: Packed complex accumulator.
    /// - `a`: Left packed complex product operand.
    /// - `b`: Right packed complex product operand.
    /// # Returns
    /// - `C64x4`: `acc + a * b`.
    #[inline(always)]
    pub(super) fn madd(
        acc: Self,
        a: Self,
        b: Self,
    ) -> Self {
        unsafe {
            Self {
                re: _mm256_fnmadd_pd(a.im, b.im, _mm256_fmadd_pd(a.re, b.re, acc.re)),
                im: _mm256_fmadd_pd(a.im, b.re, _mm256_fmadd_pd(a.re, b.im, acc.im)),
            }
        }
    }

    /// Accumulate negative packed complex product, `acc - a * b`.
    /// # Arguments:
    /// - `acc`: Packed complex accumulator.
    /// - `a`: Left packed complex product operand.
    /// - `b`: Right packed complex product operand.
    /// # Returns
    /// - `C64x4`: `acc - a * b`.
    #[inline(always)]
    pub(super) fn msub(
        acc: Self,
        a: Self,
        b: Self,
    ) -> Self {
        unsafe {
            Self {
                re: _mm256_fmadd_pd(a.im, b.im, _mm256_fnmadd_pd(a.re, b.re, acc.re)),
                im: _mm256_fnmadd_pd(a.im, b.re, _mm256_fnmadd_pd(a.re, b.im, acc.im)),
            }
        }
    }
}

impl F64x8 {
    /// Construct zero packed real values.
    /// # Arguments:
    /// - None.
    /// # Returns
    /// - `F64x8`: Packed zero.
    #[inline(always)]
    pub(super) fn zero() -> Self {
        unsafe { Self(_mm512_setzero_pd()) }
    }

    /// Broadcast one real scalar to all lanes.
    /// # Arguments:
    /// - `value`: Real scalar to broadcast.
    /// # Returns
    /// - `F64x8`: Packed broadcast value.
    #[inline(always)]
    pub(super) fn splat(value: f64) -> Self {
        unsafe { Self(_mm512_set1_pd(value)) }
    }

    /// Load eight real lane values.
    /// # Arguments:
    /// - `values`: Real lane values.
    /// # Returns
    /// - `F64x8`: Packed values.
    #[inline(always)]
    pub(super) fn load(values: &[f64; 8]) -> Self {
        unsafe { Self(_mm512_loadu_pd(values.as_ptr())) }
    }

    /// Store eight real lane values.
    /// # Arguments:
    /// - `values`: Real output lanes.
    /// # Returns
    /// - `()`: Writes packed lanes into `values`.
    #[inline(always)]
    pub(super) fn store(
        self,
        values: &mut [f64; 8],
    ) {
        unsafe { _mm512_storeu_pd(values.as_mut_ptr(), self.0) }
    }

    /// Add packed real values.
    /// # Arguments:
    /// - `a`: Left packed operand.
    /// - `b`: Right packed operand.
    /// # Returns
    /// - `F64x8`: `a + b`.
    #[inline(always)]
    pub(super) fn add(
        a: Self,
        b: Self,
    ) -> Self {
        unsafe { Self(_mm512_add_pd(a.0, b.0)) }
    }

    /// Subtract packed real values.
    /// # Arguments:
    /// - `a`: Left packed operand.
    /// - `b`: Right packed operand.
    /// # Returns
    /// - `F64x8`: `a - b`.
    #[inline(always)]
    pub(super) fn sub(
        a: Self,
        b: Self,
    ) -> Self {
        unsafe { Self(_mm512_sub_pd(a.0, b.0)) }
    }

    /// Multiply packed real values.
    /// # Arguments:
    /// - `a`: Left packed operand.
    /// - `b`: Right packed operand.
    /// # Returns
    /// - `F64x8`: `a * b`.
    #[inline(always)]
    pub(super) fn mul(
        a: Self,
        b: Self,
    ) -> Self {
        unsafe { Self(_mm512_mul_pd(a.0, b.0)) }
    }

    /// Accumulate packed real product, `acc + a * b`.
    /// # Arguments:
    /// - `acc`: Packed accumulator.
    /// - `a`: Left packed product operand.
    /// - `b`: Right packed product operand.
    /// # Returns
    /// - `F64x8`: `acc + a * b`.
    #[inline(always)]
    pub(super) fn madd(
        acc: Self,
        a: Self,
        b: Self,
    ) -> Self {
        unsafe { Self(_mm512_fmadd_pd(a.0, b.0, acc.0)) }
    }

    /// Accumulate negative packed real product, `acc - a * b`.
    /// # Arguments:
    /// - `acc`: Packed accumulator.
    /// - `a`: Left packed product operand.
    /// - `b`: Right packed product operand.
    /// # Returns
    /// - `F64x8`: `acc - a * b`.
    #[inline(always)]
    pub(super) fn msub(
        acc: Self,
        a: Self,
        b: Self,
    ) -> Self {
        unsafe { Self(_mm512_fnmadd_pd(a.0, b.0, acc.0)) }
    }
}

impl C64x8 {
    /// Construct zero packed complex values.
    /// # Arguments:
    /// - None.
    /// # Returns
    /// - `C64x8`: Packed complex zero.
    #[inline(always)]
    pub(super) fn zero() -> Self {
        unsafe {
            Self {
                re: _mm512_setzero_pd(),
                im: _mm512_setzero_pd(),
            }
        }
    }

    /// Broadcast one complex scalar to all lanes.
    /// # Arguments:
    /// - `re`: Real part to broadcast.
    /// - `im`: Imaginary part to broadcast.
    /// # Returns
    /// - `C64x8`: Packed complex broadcast value.
    #[inline(always)]
    pub(super) fn splat(
        re: f64,
        im: f64,
    ) -> Self {
        unsafe {
            Self {
                re: _mm512_set1_pd(re),
                im: _mm512_set1_pd(im),
            }
        }
    }

    /// Construct eight packed complex values from independent scalar lanes.
    /// # Arguments:
    /// - `values`: Independent scalar lane values.
    /// # Returns
    /// - `C64x8`: Packed complex values in array order.
    #[inline(always)]
    pub(super) fn from_values(values: [Complex64; 8]) -> Self {
        let [v0, v1, v2, v3, v4, v5, v6, v7] = values;
        unsafe {
            Self {
                re: _mm512_set_pd(v7.re, v6.re, v5.re, v4.re, v3.re, v2.re, v1.re, v0.re),
                im: _mm512_set_pd(v7.im, v6.im, v5.im, v4.im, v3.im, v2.im, v1.im, v0.im),
            }
        }
    }

    /// Store eight split complex lane values.
    /// # Arguments:
    /// - `re`: Real output lanes.
    /// - `im`: Imaginary output lanes.
    /// # Returns
    /// - `()`: Writes packed complex lanes into `re` and `im`.
    #[inline(always)]
    pub(super) fn store(
        self,
        re: &mut [f64; 8],
        im: &mut [f64; 8],
    ) {
        unsafe {
            _mm512_storeu_pd(re.as_mut_ptr(), self.re);
            _mm512_storeu_pd(im.as_mut_ptr(), self.im);
        }
    }

    /// Add packed complex values.
    /// # Arguments:
    /// - `a`: Left packed operand.
    /// - `b`: Right packed operand.
    /// # Returns
    /// - `C64x8`: `a + b`.
    #[inline(always)]
    pub(super) fn add(
        a: Self,
        b: Self,
    ) -> Self {
        unsafe {
            Self {
                re: _mm512_add_pd(a.re, b.re),
                im: _mm512_add_pd(a.im, b.im),
            }
        }
    }

    /// Subtract packed complex values.
    /// # Arguments:
    /// - `a`: Left packed operand.
    /// - `b`: Right packed operand.
    /// # Returns
    /// - `C64x8`: `a - b`.
    #[inline(always)]
    pub(super) fn sub(
        a: Self,
        b: Self,
    ) -> Self {
        unsafe {
            Self {
                re: _mm512_sub_pd(a.re, b.re),
                im: _mm512_sub_pd(a.im, b.im),
            }
        }
    }

    /// Multiply packed complex values with four real products.
    /// # Arguments:
    /// - `a`: Left packed complex operand.
    /// - `b`: Right packed complex operand.
    /// # Returns
    /// - `C64x8`: `a * b`.
    #[inline(always)]
    pub(super) fn mul(
        a: Self,
        b: Self,
    ) -> Self {
        unsafe {
            Self {
                re: _mm512_fmsub_pd(a.re, b.re, _mm512_mul_pd(a.im, b.im)),
                im: _mm512_fmadd_pd(a.re, b.im, _mm512_mul_pd(a.im, b.re)),
            }
        }
    }

    /// Accumulate packed complex product, `acc + a * b`.
    /// # Arguments:
    /// - `acc`: Packed complex accumulator.
    /// - `a`: Left packed complex product operand.
    /// - `b`: Right packed complex product operand.
    /// # Returns
    /// - `C64x8`: `acc + a * b`.
    #[inline(always)]
    pub(super) fn madd(
        acc: Self,
        a: Self,
        b: Self,
    ) -> Self {
        unsafe {
            Self {
                re: _mm512_fnmadd_pd(a.im, b.im, _mm512_fmadd_pd(a.re, b.re, acc.re)),
                im: _mm512_fmadd_pd(a.im, b.re, _mm512_fmadd_pd(a.re, b.im, acc.im)),
            }
        }
    }

    /// Accumulate negative packed complex product, `acc - a * b`.
    /// # Arguments:
    /// - `acc`: Packed complex accumulator.
    /// - `a`: Left packed complex product operand.
    /// - `b`: Right packed complex product operand.
    /// # Returns
    /// - `C64x8`: `acc - a * b`.
    #[inline(always)]
    pub(super) fn msub(
        acc: Self,
        a: Self,
        b: Self,
    ) -> Self {
        unsafe {
            Self {
                re: _mm512_fmadd_pd(a.im, b.im, _mm512_fnmadd_pd(a.re, b.re, acc.re)),
                im: _mm512_fnmadd_pd(a.im, b.re, _mm512_fnmadd_pd(a.re, b.im, acc.im)),
            }
        }
    }
}

impl Simd<4> for F64x4 {
    type Scalar = f64;

    /// Construct packed additive zero.
    /// # Arguments:
    /// - None.
    /// # Returns
    /// - `Self`: Packed zero.
    #[inline(always)]
    fn zero() -> Self {
        F64x4::zero()
    }

    /// Construct packed multiplicative one.
    /// # Arguments:
    /// - None.
    /// # Returns
    /// - `Self`: Packed one.
    #[inline(always)]
    fn one() -> Self {
        F64x4::splat(1.0)
    }

    /// Broadcast one scalar value into every lane.
    /// # Arguments:
    /// - `value`: Scalar value to broadcast.
    /// # Returns
    /// - `Self`: Packed broadcast value.
    #[inline(always)]
    fn splat(value: Self::Scalar) -> Self {
        F64x4::splat(value)
    }

    /// Load one scalar value for each lane.
    /// # Arguments:
    /// - `values`: Scalar values in lane order.
    /// # Returns
    /// - `Self`: Packed SIMD value.
    #[inline(always)]
    fn load(values: &[Self::Scalar; 4]) -> Self {
        F64x4::load(values)
    }

    /// Store every packed lane.
    /// # Arguments:
    /// - `self`: Packed value to store.
    /// - `values`: Scalar output lanes.
    /// # Returns
    /// - `()`: Writes every lane into `values`.
    #[inline(always)]
    fn store(
        self,
        values: &mut [Self::Scalar; 4],
    ) {
        F64x4::store(self, values);
    }

    /// Add two packed values.
    /// # Arguments:
    /// - `lhs`: Left packed operand.
    /// - `rhs`: Right packed operand.
    /// # Returns
    /// - `Self`: `lhs + rhs`.
    #[inline(always)]
    fn add(
        lhs: Self,
        rhs: Self,
    ) -> Self {
        F64x4::add(lhs, rhs)
    }

    /// Subtract two packed values.
    /// # Arguments:
    /// - `lhs`: Left packed operand.
    /// - `rhs`: Right packed operand.
    /// # Returns
    /// - `Self`: `lhs - rhs`.
    #[inline(always)]
    fn sub(
        lhs: Self,
        rhs: Self,
    ) -> Self {
        F64x4::sub(lhs, rhs)
    }

    /// Multiply two packed values.
    /// # Arguments:
    /// - `lhs`: Left packed operand.
    /// - `rhs`: Right packed operand.
    /// # Returns
    /// - `Self`: `lhs * rhs`.
    #[inline(always)]
    fn mul(
        lhs: Self,
        rhs: Self,
    ) -> Self {
        F64x4::mul(lhs, rhs)
    }

    /// Accumulate one packed product.
    /// # Arguments:
    /// - `acc`: Packed accumulator.
    /// - `lhs`: Left product operand.
    /// - `rhs`: Right product operand.
    /// # Returns
    /// - `Self`: `acc + lhs * rhs`.
    #[inline(always)]
    fn madd(
        acc: Self,
        lhs: Self,
        rhs: Self,
    ) -> Self {
        F64x4::madd(acc, lhs, rhs)
    }

    /// Subtract one packed product from an accumulator.
    /// # Arguments:
    /// - `acc`: Packed accumulator.
    /// - `lhs`: Left product operand.
    /// - `rhs`: Right product operand.
    /// # Returns
    /// - `Self`: `acc - lhs * rhs`.
    #[inline(always)]
    fn msub(
        acc: Self,
        lhs: Self,
        rhs: Self,
    ) -> Self {
        F64x4::msub(acc, lhs, rhs)
    }

    /// Multiply every lane by one real scalar.
    /// # Arguments:
    /// - `value`: Packed value to scale.
    /// - `factor`: Real scale factor.
    /// # Returns
    /// - `Self`: `factor * value`.
    #[inline(always)]
    fn scale_real(
        value: Self,
        factor: f64,
    ) -> Self {
        F64x4::mul(value, F64x4::splat(factor))
    }
}

impl Simd<8> for F64x8 {
    type Scalar = f64;

    /// Construct packed additive zero.
    /// # Arguments:
    /// - None.
    /// # Returns
    /// - `Self`: Packed zero.
    #[inline(always)]
    fn zero() -> Self {
        F64x8::zero()
    }

    /// Construct packed multiplicative one.
    /// # Arguments:
    /// - None.
    /// # Returns
    /// - `Self`: Packed one.
    #[inline(always)]
    fn one() -> Self {
        F64x8::splat(1.0)
    }

    /// Broadcast one scalar value into every lane.
    /// # Arguments:
    /// - `value`: Scalar value to broadcast.
    /// # Returns
    /// - `Self`: Packed broadcast value.
    #[inline(always)]
    fn splat(value: Self::Scalar) -> Self {
        F64x8::splat(value)
    }

    /// Load one scalar value for each lane.
    /// # Arguments:
    /// - `values`: Scalar values in lane order.
    /// # Returns
    /// - `Self`: Packed SIMD value.
    #[inline(always)]
    fn load(values: &[Self::Scalar; 8]) -> Self {
        F64x8::load(values)
    }

    /// Store every packed lane.
    /// # Arguments:
    /// - `self`: Packed value to store.
    /// - `values`: Scalar output lanes.
    /// # Returns
    /// - `()`: Writes every lane into `values`.
    #[inline(always)]
    fn store(
        self,
        values: &mut [Self::Scalar; 8],
    ) {
        F64x8::store(self, values);
    }

    /// Add two packed values.
    /// # Arguments:
    /// - `lhs`: Left packed operand.
    /// - `rhs`: Right packed operand.
    /// # Returns
    /// - `Self`: `lhs + rhs`.
    #[inline(always)]
    fn add(
        lhs: Self,
        rhs: Self,
    ) -> Self {
        F64x8::add(lhs, rhs)
    }

    /// Subtract two packed values.
    /// # Arguments:
    /// - `lhs`: Left packed operand.
    /// - `rhs`: Right packed operand.
    /// # Returns
    /// - `Self`: `lhs - rhs`.
    #[inline(always)]
    fn sub(
        lhs: Self,
        rhs: Self,
    ) -> Self {
        F64x8::sub(lhs, rhs)
    }

    /// Multiply two packed values.
    /// # Arguments:
    /// - `lhs`: Left packed operand.
    /// - `rhs`: Right packed operand.
    /// # Returns
    /// - `Self`: `lhs * rhs`.
    #[inline(always)]
    fn mul(
        lhs: Self,
        rhs: Self,
    ) -> Self {
        F64x8::mul(lhs, rhs)
    }

    /// Accumulate one packed product.
    /// # Arguments:
    /// - `acc`: Packed accumulator.
    /// - `lhs`: Left product operand.
    /// - `rhs`: Right product operand.
    /// # Returns
    /// - `Self`: `acc + lhs * rhs`.
    #[inline(always)]
    fn madd(
        acc: Self,
        lhs: Self,
        rhs: Self,
    ) -> Self {
        F64x8::madd(acc, lhs, rhs)
    }

    /// Subtract one packed product from an accumulator.
    /// # Arguments:
    /// - `acc`: Packed accumulator.
    /// - `lhs`: Left product operand.
    /// - `rhs`: Right product operand.
    /// # Returns
    /// - `Self`: `acc - lhs * rhs`.
    #[inline(always)]
    fn msub(
        acc: Self,
        lhs: Self,
        rhs: Self,
    ) -> Self {
        F64x8::msub(acc, lhs, rhs)
    }

    /// Multiply every lane by one real scalar.
    /// # Arguments:
    /// - `value`: Packed value to scale.
    /// - `factor`: Real scale factor.
    /// # Returns
    /// - `Self`: `factor * value`.
    #[inline(always)]
    fn scale_real(
        value: Self,
        factor: f64,
    ) -> Self {
        F64x8::mul(value, F64x8::splat(factor))
    }
}

impl Simd<4> for C64x4 {
    type Scalar = Complex64;

    /// Construct packed additive zero.
    /// # Arguments:
    /// - None.
    /// # Returns
    /// - `Self`: Packed zero.
    #[inline(always)]
    fn zero() -> Self {
        C64x4::zero()
    }

    /// Construct packed multiplicative one.
    /// # Arguments:
    /// - None.
    /// # Returns
    /// - `Self`: Packed one.
    #[inline(always)]
    fn one() -> Self {
        C64x4::splat(1.0, 0.0)
    }

    /// Broadcast one scalar value into every lane.
    /// # Arguments:
    /// - `value`: Scalar value to broadcast.
    /// # Returns
    /// - `Self`: Packed broadcast value.
    #[inline(always)]
    fn splat(value: Self::Scalar) -> Self {
        C64x4::splat(value.re, value.im)
    }

    /// Load one scalar value for each lane.
    /// # Arguments:
    /// - `values`: Scalar values in lane order.
    /// # Returns
    /// - `Self`: Packed SIMD value.
    #[inline(always)]
    fn load(values: &[Self::Scalar; 4]) -> Self {
        C64x4::from_values(values[0], values[1], values[2], values[3])
    }

    /// Store every packed lane.
    /// # Arguments:
    /// - `self`: Packed value to store.
    /// - `values`: Scalar output lanes.
    /// # Returns
    /// - `()`: Writes every lane into `values`.
    #[inline(always)]
    fn store(
        self,
        values: &mut [Self::Scalar; 4],
    ) {
        let mut re = [0.0; 4];
        let mut im = [0.0; 4];
        C64x4::store(self, &mut re, &mut im);
        for lane in 0..4 {
            values[lane] = Complex64::new(re[lane], im[lane]);
        }
    }

    /// Add two packed values.
    /// # Arguments:
    /// - `lhs`: Left packed operand.
    /// - `rhs`: Right packed operand.
    /// # Returns
    /// - `Self`: `lhs + rhs`.
    #[inline(always)]
    fn add(
        lhs: Self,
        rhs: Self,
    ) -> Self {
        C64x4::add(lhs, rhs)
    }

    /// Subtract two packed values.
    /// # Arguments:
    /// - `lhs`: Left packed operand.
    /// - `rhs`: Right packed operand.
    /// # Returns
    /// - `Self`: `lhs - rhs`.
    #[inline(always)]
    fn sub(
        lhs: Self,
        rhs: Self,
    ) -> Self {
        C64x4::sub(lhs, rhs)
    }

    /// Multiply two packed values.
    /// # Arguments:
    /// - `lhs`: Left packed operand.
    /// - `rhs`: Right packed operand.
    /// # Returns
    /// - `Self`: `lhs * rhs`.
    #[inline(always)]
    fn mul(
        lhs: Self,
        rhs: Self,
    ) -> Self {
        C64x4::mul(lhs, rhs)
    }

    /// Accumulate one packed product.
    /// # Arguments:
    /// - `acc`: Packed accumulator.
    /// - `lhs`: Left product operand.
    /// - `rhs`: Right product operand.
    /// # Returns
    /// - `Self`: `acc + lhs * rhs`.
    #[inline(always)]
    fn madd(
        acc: Self,
        lhs: Self,
        rhs: Self,
    ) -> Self {
        C64x4::madd(acc, lhs, rhs)
    }

    /// Subtract one packed product from an accumulator.
    /// # Arguments:
    /// - `acc`: Packed accumulator.
    /// - `lhs`: Left product operand.
    /// - `rhs`: Right product operand.
    /// # Returns
    /// - `Self`: `acc - lhs * rhs`.
    #[inline(always)]
    fn msub(
        acc: Self,
        lhs: Self,
        rhs: Self,
    ) -> Self {
        C64x4::msub(acc, lhs, rhs)
    }

    /// Multiply every lane by one real scalar.
    /// # Arguments:
    /// - `value`: Packed value to scale.
    /// - `factor`: Real scale factor.
    /// # Returns
    /// - `Self`: `factor * value`.
    #[inline(always)]
    fn scale_real(
        value: Self,
        factor: f64,
    ) -> Self {
        C64x4::mul(value, C64x4::splat(factor, 0.0))
    }
}

impl Simd<8> for C64x8 {
    type Scalar = Complex64;

    /// Construct packed additive zero.
    /// # Arguments:
    /// - None.
    /// # Returns
    /// - `Self`: Packed zero.
    #[inline(always)]
    fn zero() -> Self {
        C64x8::zero()
    }

    /// Construct packed multiplicative one.
    /// # Arguments:
    /// - None.
    /// # Returns
    /// - `Self`: Packed one.
    #[inline(always)]
    fn one() -> Self {
        C64x8::splat(1.0, 0.0)
    }

    /// Broadcast one scalar value into every lane.
    /// # Arguments:
    /// - `value`: Scalar value to broadcast.
    /// # Returns
    /// - `Self`: Packed broadcast value.
    #[inline(always)]
    fn splat(value: Self::Scalar) -> Self {
        C64x8::splat(value.re, value.im)
    }

    /// Load one scalar value for each lane.
    /// # Arguments:
    /// - `values`: Scalar values in lane order.
    /// # Returns
    /// - `Self`: Packed SIMD value.
    #[inline(always)]
    fn load(values: &[Self::Scalar; 8]) -> Self {
        C64x8::from_values(*values)
    }

    /// Store every packed lane.
    /// # Arguments:
    /// - `self`: Packed value to store.
    /// - `values`: Scalar output lanes.
    /// # Returns
    /// - `()`: Writes every lane into `values`.
    #[inline(always)]
    fn store(
        self,
        values: &mut [Self::Scalar; 8],
    ) {
        let mut re = [0.0; 8];
        let mut im = [0.0; 8];
        C64x8::store(self, &mut re, &mut im);
        for lane in 0..8 {
            values[lane] = Complex64::new(re[lane], im[lane]);
        }
    }

    /// Add two packed values.
    /// # Arguments:
    /// - `lhs`: Left packed operand.
    /// - `rhs`: Right packed operand.
    /// # Returns
    /// - `Self`: `lhs + rhs`.
    #[inline(always)]
    fn add(
        lhs: Self,
        rhs: Self,
    ) -> Self {
        C64x8::add(lhs, rhs)
    }

    /// Subtract two packed values.
    /// # Arguments:
    /// - `lhs`: Left packed operand.
    /// - `rhs`: Right packed operand.
    /// # Returns
    /// - `Self`: `lhs - rhs`.
    #[inline(always)]
    fn sub(
        lhs: Self,
        rhs: Self,
    ) -> Self {
        C64x8::sub(lhs, rhs)
    }

    /// Multiply two packed values.
    /// # Arguments:
    /// - `lhs`: Left packed operand.
    /// - `rhs`: Right packed operand.
    /// # Returns
    /// - `Self`: `lhs * rhs`.
    #[inline(always)]
    fn mul(
        lhs: Self,
        rhs: Self,
    ) -> Self {
        C64x8::mul(lhs, rhs)
    }

    /// Accumulate one packed product.
    /// # Arguments:
    /// - `acc`: Packed accumulator.
    /// - `lhs`: Left product operand.
    /// - `rhs`: Right product operand.
    /// # Returns
    /// - `Self`: `acc + lhs * rhs`.
    #[inline(always)]
    fn madd(
        acc: Self,
        lhs: Self,
        rhs: Self,
    ) -> Self {
        C64x8::madd(acc, lhs, rhs)
    }

    /// Subtract one packed product from an accumulator.
    /// # Arguments:
    /// - `acc`: Packed accumulator.
    /// - `lhs`: Left product operand.
    /// - `rhs`: Right product operand.
    /// # Returns
    /// - `Self`: `acc - lhs * rhs`.
    #[inline(always)]
    fn msub(
        acc: Self,
        lhs: Self,
        rhs: Self,
    ) -> Self {
        C64x8::msub(acc, lhs, rhs)
    }

    /// Multiply every lane by one real scalar.
    /// # Arguments:
    /// - `value`: Packed value to scale.
    /// - `factor`: Real scale factor.
    /// # Returns
    /// - `Self`: `factor * value`.
    #[inline(always)]
    fn scale_real(
        value: Self,
        factor: f64,
    ) -> Self {
        C64x8::mul(value, C64x8::splat(factor, 0.0))
    }
}
