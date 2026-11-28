// elements/nonorthogonalwicks/eval/device.cuh
// Device counterparts of the Wick evaluators, one determinant pair per thread: the overlap and
// generalised-Fock factors of `OneBodyFactor` in `onebody.rs`, and the fused Hamiltonian and overlap
// of `hamiltonian.rs`, `xw_hamiltonian_overlap_m0_const` when both spins have `m = 0` and the
// constrained distribution sums otherwise. They read the intermediates that `DeviceWicks` in
// `view/device.rs` copies and the determinants of `determinant/state/device.cuh`, with the
// determinants and cofactors of `maths/wick/device.cuh`.
//
// For bra and ket excitations of one spin with ranks `R_x` and `R_w`, `L = R_x + R_w` contraction
// labels are `(r_k, c_k) = (a_k - N_occ, i_k)` for the bra and `(N_virt + j_k, b_k)` for the ket.
// The contraction matrix of a distribution `\{m_z\}` has `D_{\eta z} = X^{(m_z)}_{r_\eta c_z}` for
// `\eta \ge z` and `Y^{(m_z)}_{r_\eta c_z}` otherwise; its distributions put the `m` zero-overlap
// orbital pairs on the operator contractions and columns, `m_i \in \{0, 1\}`.

#pragma once

#include "determinant/state/device.cuh"
#include "maths/wick/device.cuh"

/// @brief Same-spin intermediates of one combination, laid out as `DeviceWicksSpin`.
struct DeviceWicksSpin {
    /// Number `m` of zero-overlap orbital pairs.
    unsigned long long m;
    /// Number of molecular orbitals.
    unsigned long long nmo;
    /// Number of occupied orbitals.
    unsigned long long nocc;
    /// Reduced reference overlap with its orbital-pairing phase, `{}^{xw}\tilde S`.
    double pref;
    /// `X^{(m_i)}`, row-major `nmo x nmo`.
    unsigned long long x[2];
    /// `Y^{(m_i)}`, row-major `nmo x nmo`.
    unsigned long long y[2];
    /// Generalised-Fock columns `\mathcal F^{(m_i,m_j)}`, stored transposed.
    unsigned long long ff[2][2];
    /// One-electron columns `\mathcal F^{h(m_i,m_j)}`, stored transposed.
    unsigned long long fh[2][2];
    /// Same-spin one-column `\mathcal V^{(m_i,m_j,m_k)}`, stored transposed.
    unsigned long long v[2][2][2];
    /// Same-spin two-column `\mathcal J` slots, `nmo^4` entries each.
    unsigned long long j[10];
    /// Combined one-column `\mathcal H`, stored transposed, only when both spins have `m = 0`.
    unsigned long long hcol0;
    /// Generalised-Fock reference elements `F_0^{(m_1)}`.
    double f0f[2];
    /// One-electron reference elements `F_0^{h,(m_1)}`.
    double f0h[2];
    /// Same-spin two-electron reference elements `V_0^{(m_1+m_2)}`.
    double v0[3];
};

/// @brief Mixed-spin intermediates of one combination, laid out as `DeviceWicksMixed`.
struct DeviceWicksMixed {
    /// Orbital dimension of the mixed-spin intermediates.
    unsigned long long nab;
    /// Alpha one-column `\mathcal V^{\alpha(m_{\alpha 0},m_{\beta 0},m_{\alpha z})}`, transposed.
    unsigned long long vab[2][2][2];
    /// Beta one-column `\mathcal V^{\beta(m_{\beta 0},m_{\alpha 0},m_{\beta y})}`, transposed.
    unsigned long long vba[2][2][2];
    /// Two-column `\mathcal{II}^{(m_{\alpha 0},m_{\alpha z},m_{\beta 0},m_{\beta y})}`.
    unsigned long long ii[2][2][2][2];
    /// Reference elements `V_{\alpha\beta,0}^{(m_{\alpha 0},m_{\beta 0})}`.
    double vab0[2][2];
    /// Reference elements `V_{\beta\alpha,0}^{(m_{\beta 0},m_{\alpha 0})}`.
    double vba0[2][2];
};

/// @brief Intermediates of one near-zero branch combination, laid out as `DeviceWicksCombination`.
struct DeviceWicksCombination {
    /// Alpha and beta same-spin intermediates.
    DeviceWicksSpin spin[2];
    /// Mixed-spin intermediates.
    DeviceWicksMixed mixed;
};

/// @brief Location of one ordered reference pair's combinations, laid out as `DeviceWicksPair`.
struct DeviceWicksPair {
    /// One when the GPU evaluates this pair, zero otherwise.
    unsigned long long evaluated;
    /// Position of combination `(0, 0)`; `(i_\alpha, i_\beta)` is at `start + i_\alpha n_\beta + i_\beta`.
    unsigned long long start;
    /// Number of alpha branches.
    unsigned long long na;
    /// Number of beta branches.
    unsigned long long nb;
};

/// @brief Advance a distribution mask to the next with the same number of set bits, Gosper's
/// successor, the counterpart of `MCombinations::next`. Start from `(1 << m) - 1`.
/// @param mask Current mask, updated in place.
/// @param bits Number of contractions the mask spans.
/// @return Whether a next mask exists.
__device__ bool next_distribution(unsigned long long *mask, int bits) {
    unsigned long long x = *mask;
    if (x == 0) {
        return false;
    }
    unsigned long long c = x & (~x + 1);
    unsigned long long r = x + c;
    unsigned long long successor = (((r ^ x) >> 2) / c) | r;
    if (successor >= (1ULL << bits)) {
        return false;
    }
    *mask = successor;
    return true;
}

/// @brief Stored same-spin `\mathcal J` slot of the assignments `(m_1, m_2, m_{other}, m_{fixed})`
/// and whether the two label pairs are read exchanged, the counterpart of `jslot`.
/// @param m1 First operator assignment.
/// @param m2 Second operator assignment.
/// @param other Assignment of the other column.
/// @param fixed Assignment of the fixed column.
/// @param swap Whether the label pairs are exchanged, to write.
/// @return The slot.
__device__ int jslot(int m1, int m2, int other, int fixed, bool *swap) {
    const int slots[16] = {0, 1, 2, 3, 1, 4, 5, 6, 2, 5, 7, 8, 3, 6, 8, 9};
    const bool swaps[16] = {
        false, false, false, false, true, false, false, false,
        true, true, false, false, true, true, true, false,
    };
    int k = (m1 << 3) | (m2 << 2) | (other << 1) | fixed;
    *swap = swaps[k];
    return slots[k];
}

/// @brief Contraction labels of one spin, the counterpart of `construct_determinant_indices`.
/// @param spin Same-spin intermediates of the combination.
/// @param bra Bra determinant, excited from reference `x`.
/// @param ket Ket determinant, excited from reference `w`.
/// @param s `0` for alpha, `1` for beta.
/// @param rows Row labels `r_\eta` to write.
/// @param cols Column labels `c_z` to write.
/// @return The order `L = R_x + R_w`.
__device__ int labels(
    const DeviceWicksSpin &spin,
    const DeviceDeterminant &bra,
    const DeviceDeterminant &ket,
    int s,
    int *rows,
    int *cols
) {
    int nocc = (int)spin.nocc;
    int nvirt = (int)spin.nmo - nocc;
    int rx = bra.rank[s];
    int rw = ket.rank[s];
    for (int k = 0; k < rx; ++k) {
        rows[k] = bra.particles[s][k] - nocc;
        cols[k] = bra.holes[s][k];
    }
    for (int k = 0; k < rw; ++k) {
        rows[rx + k] = nvirt + ket.holes[s][k];
        cols[rx + k] = ket.particles[s][k];
    }
    return rx + rw;
}

/// @brief Contraction matrix with every column assigned `m_i`, `X^{(m_i)}` on and below the
/// diagonal and `Y^{(m_i)}` above it.
/// @param spin Same-spin intermediates of the combination.
/// @param slab Device slab of `DeviceWicks`.
/// @param mi Assignment of every column.
/// @param rows Row labels.
/// @param cols Column labels.
/// @param l Order.
/// @param d Row-major `l x l` matrix to write.
__device__ void endpoint(
    const DeviceWicksSpin &spin,
    const double *slab,
    int mi,
    const int *rows,
    const int *cols,
    int l,
    double *d
) {
    unsigned long long n = spin.nmo;
    const double *x = slab + spin.x[mi];
    const double *y = slab + spin.y[mi];
    for (int eta = 0; eta < l; ++eta) {
        for (int z = 0; z < l; ++z) {
            d[eta * l + z] = (eta >= z ? x : y)[rows[eta] * n + cols[z]];
        }
    }
}

/// @brief Contraction matrix of one distribution, each column copied from the endpoint of its
/// assignment, the counterpart of `mix_dets_same_const`.
/// @param mixed Row-major `l x l` matrix to write.
/// @param d0 All-`m_i = 0` endpoint.
/// @param d1 All-`m_i = 1` endpoint.
/// @param columns Column assignments, bit `z` for column `z`.
/// @param l Order.
__device__ void mix(
    double *mixed, const double *d0, const double *d1, unsigned long long columns, int l
) {
    for (int z = 0; z < l; ++z) {
        const double *source = ((columns >> z) & 1) ? d1 : d0;
        for (int row = 0; row < l; ++row) {
            mixed[row * l + z] = source[row * l + z];
        }
    }
}

/// @brief Overlap and generalised-Fock factors of one spin of one combination, excluding the
/// excitation phases, the counterpart of `OneBodyFactor`:
/// `S = \tilde S \sum_{\sum_z m_z = m} \det D` and
/// `F = \tilde S \sum_{m_1 + \sum_z m_z = m} [F_0^{(m_1)} \det D
///      - \sum_{\eta z} C_{\eta z} \mathcal F^{(m_1, m_z)}_{\eta z}]`. With `m = 0` the sums hold the
/// one empty distribution.
/// @param spin Same-spin intermediates of the combination.
/// @param slab Device slab of `DeviceWicks`.
/// @param bra Bra determinant, excited from reference `x`.
/// @param ket Ket determinant, excited from reference `w`.
/// @param s `0` for alpha, `1` for beta.
/// @param overlap Overlap factor to write.
/// @param fock Generalised-Fock factor to write.
__device__ void calculate_fs_spin_wicks(
    const DeviceWicksSpin &spin,
    const double *slab,
    const DeviceDeterminant &bra,
    const DeviceDeterminant &ket,
    int s,
    double *overlap,
    double *fock
) {
    int rows[MAXL], cols[MAXL];
    int l = labels(spin, bra, ket, s, rows, cols);
    int m = (int)spin.m;
    *overlap = 0.0;
    *fock = 0.0;

    // No distribution satisfies `m_1 + \sum_z m_z = m` when `m > L + 1`.
    if (m > l + 1) {
        return;
    }

    double d0[MAXL * MAXL], d1[MAXL * MAXL], mixed[MAXL * MAXL], cof[MAXL * MAXL];
    endpoint(spin, slab, 0, rows, cols, l, d0);
    endpoint(spin, slab, 1, rows, cols, l, d1);
    unsigned long long n = spin.nmo;

    // Bit 0 is the operator assignment `m_1`; bits `1..=L` select the determinant columns.
    unsigned long long bits = m == 0 ? 0 : (1ULL << m) - 1;
    double sum_s = 0.0, sum_f = 0.0;
    do {
        mix(mixed, d0, d1, bits >> 1, l);
        int m1 = bits & 1;
        double det = adjugate_transpose(cof, mixed, l);
        if (m1 == 0) {
            sum_s += det;
        }

        // `R_F = \sum_{\eta z} C_{\eta z} \mathcal F^{(m_1, m_z)}_{\eta z}`, each column reading the
        // transposed block of its own assignment.
        double replacement = 0.0;
        for (int z = 0; z < l; ++z) {
            const double *ff = slab + spin.ff[m1][(bits >> (z + 1)) & 1];
            for (int eta = 0; eta < l; ++eta) {
                replacement = fma(cof[eta * l + z], ff[cols[z] * n + rows[eta]], replacement);
            }
        }
        sum_f += det * spin.f0f[m1] - replacement;
    } while (next_distribution(&bits, l + 1));

    *overlap = spin.pref * sum_s;
    *fock = spin.pref * sum_f;
}

/// @brief Generalised-Fock and overlap elements of one determinant pair, the counterpart of the
/// Wick path of `Fock`: each spin's factors sum over its own near-zero branches, then
/// `S = p S_\alpha S_\beta` and `F = p (F_\alpha S_\beta + S_\alpha F_\beta)` with the excitation
/// phases `p`.
/// @param pair Location of the reference pair's combinations.
/// @param combinations Intermediates of every combination.
/// @param slab Device slab of `DeviceWicks`.
/// @param bra Bra determinant, excited from reference `x`.
/// @param ket Ket determinant, excited from reference `w`.
/// @param f Generalised-Fock element to write.
/// @param s Overlap element to write.
__device__ void calculate_fs_pair_wicks(
    const DeviceWicksPair &pair,
    const DeviceWicksCombination *combinations,
    const double *slab,
    const DeviceDeterminant &bra,
    const DeviceDeterminant &ket,
    double *f,
    double *s
) {
    double sa = 0.0, fa = 0.0, sb = 0.0, fb = 0.0;
    for (unsigned long long k = 0; k < pair.na; ++k) {
        double so, fo;
        const DeviceWicksCombination &c = combinations[pair.start + k * pair.nb];
        calculate_fs_spin_wicks(c.spin[0], slab, bra, ket, 0, &so, &fo);
        sa += so;
        fa += fo;
    }
    for (unsigned long long k = 0; k < pair.nb; ++k) {
        double so, fo;
        const DeviceWicksCombination &c = combinations[pair.start + k];
        calculate_fs_spin_wicks(c.spin[1], slab, bra, ket, 1, &so, &fo);
        sb += so;
        fb += fo;
    }

    double phase = bra.phase * ket.phase;
    *s = phase * (sa * sb);
    *f = phase * (fa * sb + sa * fb);
}

/// @brief Determinant, cofactors, one-column term `R` and two-column term `Q` of one spin with
/// `m = 0`, as `xw_hamiltonian_overlap_m0_const` forms them:
/// `R = \sum_{\eta z} C_{\eta z} \mathcal H_{\eta z}` and
/// `Q = \sum_{\eta<\xi, z<y} (-1)^{\eta+\xi+z+y} \det D[\eta,\xi|z,y]
///      (\mathcal J_{\eta z,\xi y} - \mathcal J_{\eta y,\xi z})`.
/// @param spin Same-spin intermediates of the combination.
/// @param slab Device slab of `DeviceWicks`.
/// @param rows Row labels.
/// @param cols Column labels.
/// @param l Order.
/// @param cof Cofactor matrix to write.
/// @param det Determinant to write.
/// @param r One-column term to write.
/// @param q Two-column term to write.
__device__ void calculate_h_spin_wicks_m0(
    const DeviceWicksSpin &spin,
    const double *slab,
    const int *rows,
    const int *cols,
    int l,
    double *cof,
    double *det,
    double *r,
    double *q
) {
    double d[MAXL * MAXL], minor[MAXL * MAXL];
    endpoint(spin, slab, 0, rows, cols, l, d);
    *det = adjugate_transpose(cof, d, l);

    unsigned long long n = spin.nmo;
    const double *j = slab + spin.j[0];
    double two = 0.0;
    for (int eta = 0; eta < l; ++eta) {
        for (int xi = eta + 1; xi < l; ++xi) {
            for (int z = 0; z < l; ++z) {
                for (int y = z + 1; y < l; ++y) {
                    second_minor(minor, d, l, eta, xi, z, y);
                    double second = determinant(minor, l - 2);
                    double direct = j[tensor(n, rows[eta], cols[z], rows[xi], cols[y])];
                    double exchange = j[tensor(n, rows[eta], cols[y], rows[xi], cols[z])];
                    double sign = ((eta + xi + z + y) & 1) == 0 ? 1.0 : -1.0;
                    two = fma(sign * second, direct - exchange, two);
                }
            }
        }
    }
    *q = two;

    const double *hcol0 = slab + spin.hcol0;
    double one = 0.0;
    for (int z = 0; z < l; ++z) {
        for (int eta = 0; eta < l; ++eta) {
            one = fma(cof[eta * l + z], hcol0[cols[z] * n + rows[eta]], one);
        }
    }
    *r = one;
}

/// @brief Hamiltonian and overlap elements of one combination with `m_\alpha = m_\beta = 0`, the
/// counterpart of `xw_hamiltonian_overlap_m0_const`:
/// `H = p (g_0 d_\alpha d_\beta - d_\beta R_\alpha - d_\alpha R_\beta + Q_\alpha d_\beta
///         + Q_\beta d_\alpha + Q_{\alpha\beta})` and `S = p d_\alpha d_\beta`, with
/// `p = p_x p_w {}^{xw}\tilde S_\alpha {}^{xw}\tilde S_\beta`.
/// @param c Intermediates of the combination.
/// @param slab Device slab of `DeviceWicks`.
/// @param bra Bra determinant, excited from reference `x`.
/// @param ket Ket determinant, excited from reference `w`.
/// @param enuc Nuclear repulsion energy.
/// @param h Hamiltonian element to write.
/// @param s Overlap element to write.
__device__ void calculate_hs_combination_wicks_m0(
    const DeviceWicksCombination &c,
    const double *slab,
    const DeviceDeterminant &bra,
    const DeviceDeterminant &ket,
    double enuc,
    double *h,
    double *s
) {
    int rows_a[MAXL], cols_a[MAXL], rows_b[MAXL], cols_b[MAXL];
    double cof_a[MAXL * MAXL], cof_b[MAXL * MAXL];
    double det_a, det_b, r_a, r_b, q_a, q_b;
    int la = labels(c.spin[0], bra, ket, 0, rows_a, cols_a);
    int lb = labels(c.spin[1], bra, ket, 1, rows_b, cols_b);
    calculate_h_spin_wicks_m0(c.spin[0], slab, rows_a, cols_a, la, cof_a, &det_a, &r_a, &q_a);
    calculate_h_spin_wicks_m0(c.spin[1], slab, rows_b, cols_b, lb, cof_b, &det_b, &r_b, &q_b);

    // `Q_{\alpha\beta} = \sum_{\eta z, \xi y} C^\alpha_{\eta z} \mathcal{II}_{\eta z,\xi y} C^\beta_{\xi y}`.
    unsigned long long n = c.mixed.nab;
    const double *ii = slab + c.mixed.ii[0][0][0][0];
    double mixed = 0.0;
    for (int z = 0; z < la; ++z) {
        for (int eta = 0; eta < la; ++eta) {
            double inner = 0.0;
            for (int y = 0; y < lb; ++y) {
                for (int xi = 0; xi < lb; ++xi) {
                    double value = ii[tensor(n, rows_a[eta], cols_a[z], rows_b[xi], cols_b[y])];
                    inner = fma(cof_b[xi * lb + y], value, inner);
                }
            }
            mixed = fma(cof_a[eta * la + z], inner, mixed);
        }
    }

    // `g_0 = E_nuc + F^h_{0,\alpha} + F^h_{0,\beta} + (V_{0,\alpha\alpha} + V_{0,\beta\beta}) / 2
    //        + V_{\alpha\beta,0}`.
    double g0 = enuc + c.spin[0].f0h[0] + 0.5 * c.spin[0].v0[0] + c.spin[1].f0h[0]
        + 0.5 * c.spin[1].v0[0] + c.mixed.vab0[0][0];

    double det_ab = det_a * det_b;
    double core = g0 * det_ab;
    core = fma(-det_b, r_a, core);
    core = fma(-det_a, r_b, core);
    core = fma(q_a, det_b, core);
    core = fma(q_b, det_a, core);
    core += mixed;

    double p = bra.phase * ket.phase * c.spin[0].pref * c.spin[1].pref;
    *h = p * core;
    *s = p * det_ab;
}

/// @brief Same-spin overlap `s`, one-body `H_1` and two-body `H_2` sums of one spin, the
/// counterpart of `xw_hamiltonian_same_m_const`, over the distributions of `m` onto the two
/// operator contractions and the `L` columns. `H_2` carries the factor two of the one-column
/// class; the caller applies the Hamiltonian's one half.
/// @param spin Same-spin intermediates of the combination.
/// @param slab Device slab of `DeviceWicks`.
/// @param rows Row labels.
/// @param cols Column labels.
/// @param l Order.
/// @param d0 All-`m_i = 0` endpoint.
/// @param d1 All-`m_i = 1` endpoint.
/// @param overlap Overlap sum to write.
/// @param one One-body sum to write.
/// @param two Two-body sum to write.
__device__ void calculate_h_spin_wicks(
    const DeviceWicksSpin &spin,
    const double *slab,
    const int *rows,
    const int *cols,
    int l,
    const double *d0,
    const double *d1,
    double *overlap,
    double *one,
    double *two
) {
    *overlap = 0.0;
    *one = 0.0;
    *two = 0.0;
    int m = (int)spin.m;
    if (m > l + 2) {
        return;
    }

    unsigned long long n = spin.nmo;
    double mixed[MAXL * MAXL], cof[MAXL * MAXL], minor[MAXL * MAXL];

    // Bits 0 and 1 are the operator assignments `m_1` and `m_2`; bits `2..L+2` select the columns.
    unsigned long long bits = m == 0 ? 0 : (1ULL << m) - 1;
    do {
        mix(mixed, d0, d1, bits >> 2, l);
        int m1 = bits & 1;
        int m2 = (bits >> 1) & 1;
        double det = adjugate_transpose(cof, mixed, l);

        // Overlap class: both operator assignments are zero.
        if (m1 == 0 && m2 == 0) {
            *overlap += det;
        }

        // One-body class for `m_2 = 0`:
        // `F^h_0{}^{(m_1)} d - \sum_{\eta z} C_{\eta z} \mathcal F^{h(m_1, m_z)}_{\eta z}`.
        if (m2 == 0) {
            double replacement = 0.0;
            for (int z = 0; z < l; ++z) {
                const double *fh = slab + spin.fh[m1][(bits >> (z + 2)) & 1];
                for (int eta = 0; eta < l; ++eta) {
                    replacement = fma(cof[eta * l + z], fh[cols[z] * n + rows[eta]], replacement);
                }
            }
            *one += det * spin.f0h[m1] - replacement;
        }

        // Scalar and one-column classes:
        // `V_0^{(m_1+m_2)} d - 2 \sum_{\eta z} C_{\eta z} \mathcal V^{(m_1, m_2, m_z)}_{\eta z}`.
        double replacement = 0.0;
        for (int z = 0; z < l; ++z) {
            const double *v = slab + spin.v[m1][m2][(bits >> (z + 2)) & 1];
            for (int eta = 0; eta < l; ++eta) {
                replacement = fma(cof[eta * l + z], v[cols[z] * n + rows[eta]], replacement);
            }
        }
        double term = det * spin.v0[m1 + m2] - 2.0 * replacement;

        // Two-column class: the ordered pairs fixing `(\eta, z)` or `(\xi, y)` enter with
        // `(-1)^{\eta+\xi+z+y}`, those fixing `(\eta, y)` or `(\xi, z)` with the opposite sign.
        for (int eta = 0; eta < l; ++eta) {
            for (int xi = eta + 1; xi < l; ++xi) {
                for (int z = 0; z < l; ++z) {
                    for (int y = z + 1; y < l; ++y) {
                        second_minor(minor, mixed, l, eta, xi, z, y);
                        double second = determinant(minor, l - 2);

                        // Fixed and other label pairs `(row, column)` of the four ordered terms.
                        const int ordered[4][4] = {
                            {eta, z, xi, y},
                            {xi, y, eta, z},
                            {eta, y, xi, z},
                            {xi, z, eta, y},
                        };
                        double entries[4];
                        for (int k = 0; k < 4; ++k) {
                            const int *o = ordered[k];
                            int mf = (bits >> (o[1] + 2)) & 1;
                            int mo = (bits >> (o[3] + 2)) & 1;
                            bool swap;
                            const double *j = slab + spin.j[jslot(m1, m2, mo, mf, &swap)];
                            entries[k] = swap
                                ? j[tensor(n, rows[o[2]], cols[o[3]], rows[o[0]], cols[o[1]])]
                                : j[tensor(n, rows[o[0]], cols[o[1]], rows[o[2]], cols[o[3]])];
                        }

                        double difference = (entries[0] + entries[1]) - (entries[2] + entries[3]);
                        double sign = ((eta + xi + z + y) & 1) == 0 ? 1.0 : -1.0;
                        term = fma(sign * second, difference, term);
                    }
                }
            }
        }

        *two += term;
    } while (next_distribution(&bits, l + 2));
}

/// @brief Mixed-spin two-body sum of one combination over the independent alpha and beta
/// distributions, the counterpart of the different-spin loop of the general Hamiltonian kernel:
/// `V_{\alpha\beta,0} d_\alpha d_\beta - d_\beta \sum C^\alpha \mathcal V^\alpha
///  - d_\alpha \sum C^\beta \mathcal V^\beta + \sum C^\alpha \mathcal{II} C^\beta`.
/// @param c Intermediates of the combination.
/// @param slab Device slab of `DeviceWicks`.
/// @param rows_a Alpha row labels.
/// @param cols_a Alpha column labels.
/// @param la Alpha order.
/// @param d0_a Alpha all-`m_i = 0` endpoint.
/// @param d1_a Alpha all-`m_i = 1` endpoint.
/// @param rows_b Beta row labels.
/// @param cols_b Beta column labels.
/// @param lb Beta order.
/// @param d0_b Beta all-`m_i = 0` endpoint.
/// @param d1_b Beta all-`m_i = 1` endpoint.
/// @return The mixed-spin sum.
__device__ double calculate_h_mixed_wicks(
    const DeviceWicksCombination &c,
    const double *slab,
    const int *rows_a,
    const int *cols_a,
    int la,
    const double *d0_a,
    const double *d1_a,
    const int *rows_b,
    const int *cols_b,
    int lb,
    const double *d0_b,
    const double *d1_b
) {
    int ma = (int)c.spin[0].m;
    int mb = (int)c.spin[1].m;
    if (ma > la + 1 || mb > lb + 1) {
        return 0.0;
    }

    unsigned long long n = c.mixed.nab;
    double mixed_a[MAXL * MAXL], mixed_b[MAXL * MAXL];
    double cof_a[MAXL * MAXL], cof_b[MAXL * MAXL];
    double sum = 0.0;

    // Bit 0 of each spin's assignment is the operator contraction; the rest select the columns.
    unsigned long long bits_a = ma == 0 ? 0 : (1ULL << ma) - 1;
    do {
        mix(mixed_a, d0_a, d1_a, bits_a >> 1, la);
        int ma0 = bits_a & 1;
        double det_a = adjugate_transpose(cof_a, mixed_a, la);

        unsigned long long bits_b = mb == 0 ? 0 : (1ULL << mb) - 1;
        do {
            mix(mixed_b, d0_b, d1_b, bits_b >> 1, lb);
            int mb0 = bits_b & 1;
            double det_b = adjugate_transpose(cof_b, mixed_b, lb);

            double term = c.mixed.vab0[ma0][mb0] * (det_a * det_b);

            double replacement_a = 0.0;
            for (int z = 0; z < la; ++z) {
                const double *v = slab + c.mixed.vab[ma0][mb0][(bits_a >> (z + 1)) & 1];
                for (int eta = 0; eta < la; ++eta) {
                    replacement_a = fma(cof_a[eta * la + z], v[cols_a[z] * n + rows_a[eta]],
                        replacement_a);
                }
            }
            term = fma(-det_b, replacement_a, term);

            double replacement_b = 0.0;
            for (int y = 0; y < lb; ++y) {
                const double *v = slab + c.mixed.vba[mb0][ma0][(bits_b >> (y + 1)) & 1];
                for (int xi = 0; xi < lb; ++xi) {
                    replacement_b = fma(cof_b[xi * lb + y], v[cols_b[y] * n + rows_b[xi]],
                        replacement_b);
                }
            }
            term = fma(-det_a, replacement_b, term);

            for (int z = 0; z < la; ++z) {
                int maz = (bits_a >> (z + 1)) & 1;
                for (int eta = 0; eta < la; ++eta) {
                    double inner = 0.0;
                    for (int y = 0; y < lb; ++y) {
                        const double *ii =
                            slab + c.mixed.ii[ma0][maz][mb0][(bits_b >> (y + 1)) & 1];
                        for (int xi = 0; xi < lb; ++xi) {
                            double value =
                                ii[tensor(n, rows_a[eta], cols_a[z], rows_b[xi], cols_b[y])];
                            inner = fma(cof_b[xi * lb + y], value, inner);
                        }
                    }
                    term = fma(cof_a[eta * la + z], inner, term);
                }
            }

            sum += term;
        } while (next_distribution(&bits_b, lb + 1));
    } while (next_distribution(&bits_a, la + 1));

    return sum;
}

/// @brief Hamiltonian and overlap elements of one combination with `m_\alpha > 0` or
/// `m_\beta > 0`: `S = p s_\alpha s_\beta` and
/// `H = E_nuc S + p [H_{2,\alpha\beta} + H_{1,\alpha} s_\beta + H_{1,\beta} s_\alpha
///                   + H_{2,\alpha\alpha} s_\beta / 2 + H_{2,\beta\beta} s_\alpha / 2]`.
/// @param c Intermediates of the combination.
/// @param slab Device slab of `DeviceWicks`.
/// @param bra Bra determinant, excited from reference `x`.
/// @param ket Ket determinant, excited from reference `w`.
/// @param enuc Nuclear repulsion energy.
/// @param h Hamiltonian element to write.
/// @param s Overlap element to write.
__device__ void calculate_hs_combination_wicks(
    const DeviceWicksCombination &c,
    const double *slab,
    const DeviceDeterminant &bra,
    const DeviceDeterminant &ket,
    double enuc,
    double *h,
    double *s
) {
    int rows_a[MAXL], cols_a[MAXL], rows_b[MAXL], cols_b[MAXL];
    double d0_a[MAXL * MAXL], d1_a[MAXL * MAXL], d0_b[MAXL * MAXL], d1_b[MAXL * MAXL];
    int la = labels(c.spin[0], bra, ket, 0, rows_a, cols_a);
    int lb = labels(c.spin[1], bra, ket, 1, rows_b, cols_b);
    endpoint(c.spin[0], slab, 0, rows_a, cols_a, la, d0_a);
    endpoint(c.spin[0], slab, 1, rows_a, cols_a, la, d1_a);
    endpoint(c.spin[1], slab, 0, rows_b, cols_b, lb, d0_b);
    endpoint(c.spin[1], slab, 1, rows_b, cols_b, lb, d1_b);

    double s_a, h1_a, h2_a, s_b, h1_b, h2_b;
    calculate_h_spin_wicks(c.spin[0], slab, rows_a, cols_a, la, d0_a, d1_a, &s_a, &h1_a, &h2_a);
    calculate_h_spin_wicks(c.spin[1], slab, rows_b, cols_b, lb, d0_b, d1_b, &s_b, &h1_b, &h2_b);
    double h2_ab = calculate_h_mixed_wicks(
        c, slab, rows_a, cols_a, la, d0_a, d1_a, rows_b, cols_b, lb, d0_b, d1_b);

    double core = h2_ab;
    core = fma(h1_a, s_b, core);
    core = fma(h1_b, s_a, core);
    core = fma(0.5 * h2_a, s_b, core);
    core = fma(0.5 * h2_b, s_a, core);

    double p = bra.phase * ket.phase * c.spin[0].pref * c.spin[1].pref;
    double overlap = p * (s_a * s_b);
    *s = overlap;
    *h = fma(enuc, overlap, p * core);
}

/// @brief Hamiltonian and overlap elements of one determinant pair, summed over every near-zero
/// branch combination of the reference pair, the counterpart of the Wick path of
/// `HamiltonianOverlap`.
/// @param pair Location of the reference pair's combinations.
/// @param combinations Intermediates of every combination.
/// @param slab Device slab of `DeviceWicks`.
/// @param bra Bra determinant, excited from reference `x`.
/// @param ket Ket determinant, excited from reference `w`.
/// @param enuc Nuclear repulsion energy.
/// @param h Hamiltonian element to write.
/// @param s Overlap element to write.
__device__ void calculate_hs_pair_wicks(
    const DeviceWicksPair &pair,
    const DeviceWicksCombination *combinations,
    const double *slab,
    const DeviceDeterminant &bra,
    const DeviceDeterminant &ket,
    double enuc,
    double *h,
    double *s
) {
    *h = 0.0;
    *s = 0.0;
    for (unsigned long long k = 0; k < pair.na * pair.nb; ++k) {
        const DeviceWicksCombination &c = combinations[pair.start + k];
        double hk, sk;
        if (c.spin[0].m == 0 && c.spin[1].m == 0) {
            calculate_hs_combination_wicks_m0(c, slab, bra, ket, enuc, &hk, &sk);
        } else {
            calculate_hs_combination_wicks(c, slab, bra, ket, enuc, &hk, &sk);
        }
        *h += hk;
        *s += sk;
    }
}
