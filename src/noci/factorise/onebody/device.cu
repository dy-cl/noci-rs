// noci/factorise/onebody/device.cu
// CUDA kernels of `DeviceOneBody` in `device.rs`, the GPU counterpart of `OneBodyFactorisation` in
// `onebody.rs`. `build/build.rs` compiles this file to PTX, and every kernel has C linkage so the
// host looks it up by name. Parameters carry the names of the `device.rs` fields passed to them.
//
// The spin-factorised one-body operator acts on a determinant vector parent pair by parent pair.
// For a factorised pair, with the first spin's tables `F_1`, `S_1` and the second spin's `F_2`,
// `S_2` (alpha then beta for alpha-first pairs, the reverse for beta-first pairs),
//     T^F_{t_1 s_2} = \sum_{s_1} F_{1, t_1 s_1} D_{s_1 s_2},  T^S likewise with S_1,
//     y_{(t_1, t_2)} += \sum_{s_2} T^F_{t_1 s_2} S_{2, t_2 s_2}
//                       + T^S_{t_1 s_2} (F_{2, t_2 s_2} + \lambda S_{2, t_2 s_2}),
// where `D` holds the source amplitudes at their spin strings. Same-parent pairs of an orthonormal
// parent apply tabulated Slater-Condon couplings instead.

/// Number of threads in a warp.
#define WARP 32

/// @brief First contraction of a factorised parent pair, `T^F = F_1 D` and `T^S = S_1 D`.
///
/// One thread forms one element `(t_1, s_2)`. The source determinants are grouped by their
/// second-spin string, so column `s_2` of `D` is the range `offsets[s_2]..offsets[s_2 + 1]` of
/// `first` (first-spin strings) and `sources` (positions in `x`).
///
/// @param f1 First-spin Fock factors, row-major `n_t1 x n_s1`.
/// @param s1 First-spin overlap factors, row-major `n_t1 x n_s1`.
/// @param n_s1 Number of source first-spin strings.
/// @param offsets Start of each second-spin string's sources, `n_s2 + 1` entries.
/// @param first First-spin string of each source.
/// @param sources Position of each source in `x`.
/// @param x Source determinant vector.
/// @param n_t1 Number of target first-spin strings.
/// @param n_s2 Number of source second-spin strings.
/// @param tf Output `T^F`, row-major `n_t1 x n_s2`.
/// @param ts Output `T^S`, row-major `n_t1 x n_s2`.
extern "C" __global__ void apply_one_body_factorised_first(
    const double *f1,
    const double *s1,
    unsigned long long n_s1,
    const unsigned long long *offsets,
    const unsigned long long *first,
    const unsigned long long *sources,
    const double *x,
    unsigned long long n_t1,
    unsigned long long n_s2,
    double *tf,
    double *ts
) {
    unsigned long long idx = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (idx >= n_t1 * n_s2) {
        return;
    }

    unsigned long long t1 = idx / n_s2;
    unsigned long long s2 = idx % n_s2;
    const double *f1row = f1 + t1 * n_s1;
    const double *s1row = s1 + t1 * n_s1;

    // Sum the first-spin factors over the sources of second-spin string `s_2`.
    double af = 0.0;
    double as = 0.0;
    for (unsigned long long k = offsets[s2]; k < offsets[s2 + 1]; ++k) {
        double xe = x[sources[k]];
        af += f1row[first[k]] * xe;
        as += s1row[first[k]] * xe;
    }

    tf[idx] = af;
    ts[idx] = as;
}

/// @brief Second contraction of a factorised parent pair, accumulated into the target vector.
///
/// One warp forms one target determinant `(t_1, t_2)`:
/// `y += \sum_{s_2} T^F_{t_1 s_2} S_{2, t_2 s_2} + T^S_{t_1 s_2} (F_{2, t_2 s_2} + \lambda S_{2, t_2 s_2})`.
/// Each target appears once per pair, so no two warps write the same element.
///
/// @param tf First contraction `T^F`, row-major `n_t1 x n_s2`.
/// @param ts First contraction `T^S`, row-major `n_t1 x n_s2`.
/// @param f2 Second-spin Fock factors, row-major `n_t2 x n_s2`.
/// @param s2 Second-spin overlap factors, row-major `n_t2 x n_s2`.
/// @param n_s2 Number of source second-spin strings.
/// @param targets Position of each target in `y`.
/// @param t1 First-spin string of each target.
/// @param t2 Second-spin string of each target.
/// @param ntarget Number of targets.
/// @param lambda Overlap shift `\lambda`.
/// @param y Target determinant vector to accumulate into.
extern "C" __global__ void apply_one_body_factorised_second(
    const double *tf,
    const double *ts,
    const double *f2,
    const double *s2,
    unsigned long long n_s2,
    const unsigned long long *targets,
    const unsigned long long *t1,
    const unsigned long long *t2,
    unsigned long long ntarget,
    double lambda,
    double *y
) {
    unsigned long long warp = (blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x) / WARP;
    unsigned int lane = threadIdx.x % WARP;
    if (warp >= ntarget) {
        return;
    }

    const double *tfrow = tf + t1[warp] * n_s2;
    const double *tsrow = ts + t1[warp] * n_s2;
    const double *f2row = f2 + t2[warp] * n_s2;
    const double *s2row = s2 + t2[warp] * n_s2;

    // Each lane sums a strided share of the second-spin strings.
    double value = 0.0;
    for (unsigned long long j = lane; j < n_s2; j += WARP) {
        value += tfrow[j] * s2row[j] + tsrow[j] * (f2row[j] + lambda * s2row[j]);
    }

    // Reduce the lanes' partial sums into lane zero.
    for (int offset = WARP / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }

    if (lane == 0) {
        y[targets[warp]] += value;
    }
}

/// @brief Same-parent orthogonal one-body action, one thread per target determinant.
///
/// Row `k` adds its single-excitation couplings `\sum_s F_{ts} x_s` and its identical-occupation
/// terms `\sum_s (F_{ts} + \lambda S_{ts}) x_s`, each held in compressed rows.
///
/// @param single_offsets Start of each row's single couplings, `ntarget + 1` entries.
/// @param single_source Source position of each single coupling.
/// @param single_f Fock element of each single coupling.
/// @param same_offsets Start of each row's identical-occupation terms, `ntarget + 1` entries.
/// @param same_source Source position of each identical-occupation term.
/// @param same_f Fock element of each identical-occupation term.
/// @param same_s Overlap element of each identical-occupation term.
/// @param targets Position of each target in `y`.
/// @param ntarget Number of targets.
/// @param x Source determinant vector.
/// @param lambda Overlap shift `\lambda`.
/// @param y Target determinant vector to accumulate into.
extern "C" __global__ void apply_one_body_orthogonal(
    const unsigned long long *single_offsets,
    const unsigned long long *single_source,
    const double *single_f,
    const unsigned long long *same_offsets,
    const unsigned long long *same_source,
    const double *same_f,
    const double *same_s,
    const unsigned long long *targets,
    unsigned long long ntarget,
    const double *x,
    double lambda,
    double *y
) {
    unsigned long long k = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (k >= ntarget) {
        return;
    }

    // Single-excitation couplings, `F_{ts} x_s`.
    double value = 0.0;
    for (unsigned long long j = single_offsets[k]; j < single_offsets[k + 1]; ++j) {
        value += single_f[j] * x[single_source[j]];
    }

    // Identical occupations, `(F_{ts} + \lambda S_{ts}) x_s`.
    for (unsigned long long j = same_offsets[k]; j < same_offsets[k + 1]; ++j) {
        value += (same_f[j] + lambda * same_s[j]) * x[same_source[j]];
    }

    y[targets[k]] += value;
}
