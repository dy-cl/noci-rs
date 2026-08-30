// nonorthogonalwicks/eval/simd.rs
//! Evaluator-local packed real and complex arithmetic.

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

/// Four packed `f64` values in one AVX2 register.
#[derive(Clone, Copy)]
pub(super) struct F64x4(__m256d);

/// Four packed complex values in split real/imaginary AVX2 registers.
#[derive(Clone, Copy)]
pub(super) struct C64x4 {
    re: __m256d,
    im: __m256d,
}

/// Eight packed `f64` values in one AVX-512 register.
#[derive(Clone, Copy)]
pub(super) struct F64x8(__m512d);

/// Eight packed complex values in split real/imaginary AVX-512 registers.
#[derive(Clone, Copy)]
pub(super) struct C64x8 {
    re: __m512d,
    im: __m512d,
}

impl F64x4 {
    /// Construct zero packed real values.
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

    /// Construct four packed real values from independent scalar lanes.
    /// # Arguments:
    /// - `v0`: Lane 0 value.
    /// - `v1`: Lane 1 value.
    /// - `v2`: Lane 2 value.
    /// - `v3`: Lane 3 value.
    /// # Returns
    /// - `F64x4`: Packed values `[v0,v1,v2,v3]`.
    #[inline(always)]
    pub(super) fn from_values(
        v0: f64,
        v1: f64,
        v2: f64,
        v3: f64,
    ) -> Self {
        unsafe { Self(_mm256_set_pd(v3, v2, v1, v0)) }
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

    /// Multiply packed real values and subtract a third packed value.
    /// # Arguments:
    /// - `a`: Left packed product operand.
    /// - `b`: Right packed product operand.
    /// - `c`: Packed value to subtract.
    /// # Returns
    /// - `F64x4`: `a * b - c`.
    #[inline(always)]
    pub(super) fn mul_sub(
        a: Self,
        b: Self,
        c: Self,
    ) -> Self {
        unsafe { Self(_mm256_fmsub_pd(a.0, b.0, c.0)) }
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

    /// Evaluate packed `2 x 2` minor, `a * b - c * d`.
    /// # Arguments:
    /// - `a`: First positive product operand.
    /// - `b`: Second positive product operand.
    /// - `c`: First negative product operand.
    /// - `d`: Second negative product operand.
    /// # Returns
    /// - `F64x4`: `a * b - c * d`.
    #[inline(always)]
    pub(super) fn minor(
        a: Self,
        b: Self,
        c: Self,
        d: Self,
    ) -> Self {
        unsafe { Self(_mm256_fmsub_pd(a.0, b.0, _mm256_mul_pd(c.0, d.0))) }
    }

    /// Evaluate positive packed `3 x 3` cofactor, `a * m0 - b * m1 + c * m2`.
    /// # Arguments:
    /// - `a`: First scalar entry.
    /// - `m0`: First `2 x 2` minor.
    /// - `b`: Second scalar entry.
    /// - `m1`: Second `2 x 2` minor.
    /// - `c`: Third scalar entry.
    /// - `m2`: Third `2 x 2` minor.
    /// # Returns
    /// - `F64x4`: Positive cofactor.
    #[inline(always)]
    pub(super) fn cof_pos(
        a: Self,
        m0: Self,
        b: Self,
        m1: Self,
        c: Self,
        m2: Self,
    ) -> Self {
        unsafe {
            let value = _mm256_fmsub_pd(a.0, m0.0, _mm256_mul_pd(b.0, m1.0));
            Self(_mm256_fmadd_pd(c.0, m2.0, value))
        }
    }

    /// Evaluate negative packed `3 x 3` cofactor, `b * m1 - a * m0 - c * m2`.
    /// # Arguments:
    /// - `a`: First scalar entry.
    /// - `m0`: First `2 x 2` minor.
    /// - `b`: Second scalar entry.
    /// - `m1`: Second `2 x 2` minor.
    /// - `c`: Third scalar entry.
    /// - `m2`: Third `2 x 2` minor.
    /// # Returns
    /// - `F64x4`: Negative cofactor.
    #[inline(always)]
    pub(super) fn cof_neg(
        a: Self,
        m0: Self,
        b: Self,
        m1: Self,
        c: Self,
        m2: Self,
    ) -> Self {
        unsafe {
            let value = _mm256_fmsub_pd(b.0, m1.0, _mm256_mul_pd(a.0, m0.0));
            Self(_mm256_fnmadd_pd(c.0, m2.0, value))
        }
    }
}

impl C64x4 {
    /// Construct zero packed complex values.
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

    /// Multiply packed complex values and subtract a third packed value.
    /// # Arguments:
    /// - `a`: Left packed product operand.
    /// - `b`: Right packed product operand.
    /// - `c`: Packed value to subtract.
    /// # Returns
    /// - `C64x4`: `a * b - c`.
    #[inline(always)]
    pub(super) fn mul_sub(
        a: Self,
        b: Self,
        c: Self,
    ) -> Self {
        unsafe {
            let re = _mm256_fmsub_pd(a.re, b.re, c.re);
            let re = _mm256_fnmadd_pd(a.im, b.im, re);

            let im = _mm256_fmsub_pd(a.re, b.im, c.im);
            let im = _mm256_fmadd_pd(a.im, b.re, im);

            Self { re, im }
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

    /// Evaluate packed complex `2 x 2` minor, `a * b - c * d`.
    /// # Arguments:
    /// - `a`: First positive product operand.
    /// - `b`: Second positive product operand.
    /// - `c`: First negative product operand.
    /// - `d`: Second negative product operand.
    /// # Returns
    /// - `C64x4`: `a * b - c * d`.
    #[inline(always)]
    pub(super) fn minor(
        a: Self,
        b: Self,
        c: Self,
        d: Self,
    ) -> Self {
        Self::msub(Self::mul(a, b), c, d)
    }

    /// Evaluate positive packed complex `3 x 3` cofactor, `a * m0 - b * m1 + c * m2`.
    /// # Arguments:
    /// - `a`: First scalar entry.
    /// - `m0`: First `2 x 2` minor.
    /// - `b`: Second scalar entry.
    /// - `m1`: Second `2 x 2` minor.
    /// - `c`: Third scalar entry.
    /// - `m2`: Third `2 x 2` minor.
    /// # Returns
    /// - `C64x4`: Positive cofactor.
    #[inline(always)]
    pub(super) fn cof_pos(
        a: Self,
        m0: Self,
        b: Self,
        m1: Self,
        c: Self,
        m2: Self,
    ) -> Self {
        let value = Self::msub(Self::mul(a, m0), b, m1);
        Self::madd(value, c, m2)
    }

    /// Evaluate negative packed complex `3 x 3` cofactor, `b * m1 - a * m0 - c * m2`.
    /// # Arguments:
    /// - `a`: First scalar entry.
    /// - `m0`: First `2 x 2` minor.
    /// - `b`: Second scalar entry.
    /// - `m1`: Second `2 x 2` minor.
    /// - `c`: Third scalar entry.
    /// - `m2`: Third `2 x 2` minor.
    /// # Returns
    /// - `C64x4`: Negative cofactor.
    #[inline(always)]
    pub(super) fn cof_neg(
        a: Self,
        m0: Self,
        b: Self,
        m1: Self,
        c: Self,
        m2: Self,
    ) -> Self {
        let value = Self::msub(Self::mul(b, m1), a, m0);
        Self::msub(value, c, m2)
    }
}

impl F64x8 {
    /// Construct zero packed real values.
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

    /// Construct eight packed real values from independent scalar lanes.
    /// # Arguments:
    /// - `values`: Independent scalar lane values.
    /// # Returns
    /// - `F64x8`: Packed values in array order.
    #[inline(always)]
    pub(super) fn from_values(values: [f64; 8]) -> Self {
        unsafe { Self(_mm512_loadu_pd(values.as_ptr())) }
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

    /// Multiply packed real values and subtract a third packed value.
    /// # Arguments:
    /// - `a`: Left packed product operand.
    /// - `b`: Right packed product operand.
    /// - `c`: Packed value to subtract.
    /// # Returns
    /// - `F64x8`: `a * b - c`.
    #[inline(always)]
    pub(super) fn mul_sub(
        a: Self,
        b: Self,
        c: Self,
    ) -> Self {
        unsafe { Self(_mm512_fmsub_pd(a.0, b.0, c.0)) }
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

    /// Evaluate packed `2 x 2` minor, `a * b - c * d`.
    /// # Arguments:
    /// - `a`: First positive product operand.
    /// - `b`: Second positive product operand.
    /// - `c`: First negative product operand.
    /// - `d`: Second negative product operand.
    /// # Returns
    /// - `F64x8`: `a * b - c * d`.
    #[inline(always)]
    pub(super) fn minor(
        a: Self,
        b: Self,
        c: Self,
        d: Self,
    ) -> Self {
        unsafe { Self(_mm512_fmsub_pd(a.0, b.0, _mm512_mul_pd(c.0, d.0))) }
    }

    /// Evaluate positive packed `3 x 3` cofactor, `a * m0 - b * m1 + c * m2`.
    /// # Arguments:
    /// - `a`: First scalar entry.
    /// - `m0`: First `2 x 2` minor.
    /// - `b`: Second scalar entry.
    /// - `m1`: Second `2 x 2` minor.
    /// - `c`: Third scalar entry.
    /// - `m2`: Third `2 x 2` minor.
    /// # Returns
    /// - `F64x8`: Positive cofactor.
    #[inline(always)]
    pub(super) fn cof_pos(
        a: Self,
        m0: Self,
        b: Self,
        m1: Self,
        c: Self,
        m2: Self,
    ) -> Self {
        unsafe {
            let value = _mm512_fmsub_pd(a.0, m0.0, _mm512_mul_pd(b.0, m1.0));
            Self(_mm512_fmadd_pd(c.0, m2.0, value))
        }
    }

    /// Evaluate negative packed `3 x 3` cofactor, `b * m1 - a * m0 - c * m2`.
    /// # Arguments:
    /// - `a`: First scalar entry.
    /// - `m0`: First `2 x 2` minor.
    /// - `b`: Second scalar entry.
    /// - `m1`: Second `2 x 2` minor.
    /// - `c`: Third scalar entry.
    /// - `m2`: Third `2 x 2` minor.
    /// # Returns
    /// - `F64x8`: Negative cofactor.
    #[inline(always)]
    pub(super) fn cof_neg(
        a: Self,
        m0: Self,
        b: Self,
        m1: Self,
        c: Self,
        m2: Self,
    ) -> Self {
        unsafe {
            let value = _mm512_fmsub_pd(b.0, m1.0, _mm512_mul_pd(a.0, m0.0));
            Self(_mm512_fnmadd_pd(c.0, m2.0, value))
        }
    }
}

impl C64x8 {
    /// Construct zero packed complex values.
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

    /// Multiply packed complex values and subtract a third packed value.
    /// # Arguments:
    /// - `a`: Left packed product operand.
    /// - `b`: Right packed product operand.
    /// - `c`: Packed value to subtract.
    /// # Returns
    /// - `C64x8`: `a * b - c`.
    #[inline(always)]
    pub(super) fn mul_sub(
        a: Self,
        b: Self,
        c: Self,
    ) -> Self {
        unsafe {
            let re = _mm512_fmsub_pd(a.re, b.re, c.re);
            let re = _mm512_fnmadd_pd(a.im, b.im, re);

            let im = _mm512_fmsub_pd(a.re, b.im, c.im);
            let im = _mm512_fmadd_pd(a.im, b.re, im);

            Self { re, im }
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

    /// Evaluate packed complex `2 x 2` minor, `a * b - c * d`.
    /// # Arguments:
    /// - `a`: First positive product operand.
    /// - `b`: Second positive product operand.
    /// - `c`: First negative product operand.
    /// - `d`: Second negative product operand.
    /// # Returns
    /// - `C64x8`: `a * b - c * d`.
    #[inline(always)]
    pub(super) fn minor(
        a: Self,
        b: Self,
        c: Self,
        d: Self,
    ) -> Self {
        Self::msub(Self::mul(a, b), c, d)
    }

    /// Evaluate positive packed complex `3 x 3` cofactor, `a * m0 - b * m1 + c * m2`.
    /// # Arguments:
    /// - `a`: First scalar entry.
    /// - `m0`: First `2 x 2` minor.
    /// - `b`: Second scalar entry.
    /// - `m1`: Second `2 x 2` minor.
    /// - `c`: Third scalar entry.
    /// - `m2`: Third `2 x 2` minor.
    /// # Returns
    /// - `C64x8`: Positive cofactor.
    #[inline(always)]
    pub(super) fn cof_pos(
        a: Self,
        m0: Self,
        b: Self,
        m1: Self,
        c: Self,
        m2: Self,
    ) -> Self {
        let value = Self::msub(Self::mul(a, m0), b, m1);
        Self::madd(value, c, m2)
    }

    /// Evaluate negative packed complex `3 x 3` cofactor, `b * m1 - a * m0 - c * m2`.
    /// # Arguments:
    /// - `a`: First scalar entry.
    /// - `m0`: First `2 x 2` minor.
    /// - `b`: Second scalar entry.
    /// - `m1`: Second `2 x 2` minor.
    /// - `c`: Third scalar entry.
    /// - `m2`: Third `2 x 2` minor.
    /// # Returns
    /// - `C64x8`: Negative cofactor.
    #[inline(always)]
    pub(super) fn cof_neg(
        a: Self,
        m0: Self,
        b: Self,
        m1: Self,
        c: Self,
        m2: Self,
    ) -> Self {
        let value = Self::msub(Self::mul(b, m1), a, m0);
        Self::msub(value, c, m2)
    }
}
