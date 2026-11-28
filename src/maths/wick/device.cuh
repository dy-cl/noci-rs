// maths/wick/device.cuh
// Device counterparts of the fixed-rank determinant routines of `maths/wick.rs`: determinants and
// cofactor matrices by the pivot-free Faddeev-LeVerrier recurrence of `det_const` and
// `adjugate_transpose_const`, second minors as `second_minor_const` forms them, and the row-major
// rank-four tensor index. Matrices are row-major with runtime order up to `MAXL`.

#pragma once

/// Largest contraction-determinant order of one spin: a bra and a ket excitation of full rank.
#define MAXL (2 * MAXEXCIT)

/// @brief Determinant of a row-major `n x n` matrix by the Faddeev-LeVerrier recurrence, the
/// counterpart of `det_const`.
/// @param matrix Row-major matrix.
/// @param n Order, at most `MAXL`.
/// @return The determinant, one for `n = 0`.
__device__ double determinant(const double *matrix, int n) {
    if (n == 0) {
        return 1.0;
    }

    double b[MAXL * MAXL];
    double product[MAXL * MAXL];

    // Start the recurrence with `B_0 = I`.
    for (int i = 0; i < n * n; ++i) {
        b[i] = 0.0;
    }
    for (int i = 0; i < n; ++i) {
        b[i * n + i] = 1.0;
    }

    for (int k = 1; k <= n; ++k) {
        // `A B_{k-1}` and its characteristic coefficient `c_k = -tr(A B_{k-1}) / k`.
        double trace = 0.0;
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
                double value = 0.0;
                for (int inner = 0; inner < n; ++inner) {
                    value = fma(matrix[row * n + inner], b[inner * n + col], value);
                }
                product[row * n + col] = value;
            }
            trace += product[row * n + row];
        }
        double coefficient = trace * (-1.0 / k);

        if (k == n) {
            return (n & 1) == 0 ? coefficient : -coefficient;
        }

        // `B_k = A B_{k-1} + c_k I`.
        for (int i = 0; i < n * n; ++i) {
            b[i] = product[i];
        }
        for (int i = 0; i < n; ++i) {
            b[i * n + i] += coefficient;
        }
    }

    return 0.0;
}

/// @brief Determinant and cofactor matrix `C_{rc} = (-1)^{r+c} \det A[r|c]` of a row-major
/// `n x n` matrix by the Faddeev-LeVerrier recurrence, the counterpart of
/// `adjugate_transpose_const`.
/// @param cof Row-major cofactor matrix to write; untouched for `n = 0`.
/// @param matrix Row-major matrix.
/// @param n Order, at most `MAXL`.
/// @return The determinant, one for `n = 0`.
__device__ double adjugate_transpose(double *cof, const double *matrix, int n) {
    if (n == 0) {
        return 1.0;
    }

    double b[MAXL * MAXL];
    double product[MAXL * MAXL];

    for (int i = 0; i < n * n; ++i) {
        b[i] = 0.0;
    }
    for (int i = 0; i < n; ++i) {
        b[i * n + i] = 1.0;
    }

    // The cofactor of a `1 x 1` matrix is the empty determinant.
    if (n == 1) {
        cof[0] = 1.0;
    }

    for (int k = 1; k <= n; ++k) {
        double trace = 0.0;
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
                double value = 0.0;
                for (int inner = 0; inner < n; ++inner) {
                    value = fma(matrix[row * n + inner], b[inner * n + col], value);
                }
                product[row * n + col] = value;
            }
            trace += product[row * n + row];
        }
        double coefficient = trace * (-1.0 / k);

        if (k == n) {
            return (n & 1) == 0 ? coefficient : -coefficient;
        }

        for (int i = 0; i < n * n; ++i) {
            b[i] = product[i];
        }
        for (int i = 0; i < n; ++i) {
            b[i * n + i] += coefficient;
        }

        // `C = (-1)^{n-1} B_{n-1}^T`.
        if (k + 1 == n) {
            double sign = ((n - 1) & 1) == 0 ? 1.0 : -1.0;
            for (int row = 0; row < n; ++row) {
                for (int col = 0; col < n; ++col) {
                    cof[row * n + col] = sign * b[col * n + row];
                }
            }
        }
    }

    return 0.0;
}

/// @brief Second minor `A[\eta,\xi|z,y]` of a row-major `n x n` matrix, the counterpart of
/// `second_minor_const`.
/// @param minor Row-major `(n - 2) x (n - 2)` minor to write.
/// @param matrix Row-major matrix.
/// @param n Order.
/// @param eta First removed row.
/// @param xi Second removed row.
/// @param z First removed column.
/// @param y Second removed column.
__device__ void second_minor(
    double *minor, const double *matrix, int n, int eta, int xi, int z, int y
) {
    int k = 0;
    for (int row = 0; row < n; ++row) {
        if (row == eta || row == xi) {
            continue;
        }
        for (int col = 0; col < n; ++col) {
            if (col == z || col == y) {
                continue;
            }
            minor[k++] = matrix[row * n + col];
        }
    }
}

/// @brief Position of `(p, q, r, s)` in a row-major rank-four tensor.
/// @param n Dimension of every axis.
/// @param p First index.
/// @param q Second index.
/// @param r Third index.
/// @param s Fourth index.
/// @return The flat position `((p n + q) n + r) n + s`.
__device__ unsigned long long tensor(unsigned long long n, int p, int q, int r, int s) {
    return ((p * n + q) * n + r) * n + s;
}
