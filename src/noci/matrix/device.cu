// noci/matrix/device.cu
// CUDA kernels of `DeviceNOCIRows` in `device.rs`, the GPU counterpart of `contract_noci_rows` in
// `matrix.rs`: `y_l = \sum_r O_{lr} c_r` over the determinant pairs of the reference pairs the GPU
// evaluates, without storing the block. Pairs of distinct references go through the Wick
// intermediates of `DeviceWicks`, and pairs of one orthonormal parent through the Slater-Condon
// integrals of `DeviceOrthogonal`. `build/build.rs` compiles this file to PTX, and
// every kernel has C linkage so the host looks it up by name. Parameters carry the names of the
// `device.rs` values passed to them.
//
// One block contracts one bra row. Its threads take the kets in a fixed stride and sum their
// shares in a fixed tree, so every row is summed in the same order on every run.

#include "elements/nonorthogonalwicks/eval/device.cuh"
#include "elements/orthogonal/eval/device.cuh"

/// Threads per block, matching `THREADS` in `device.rs`.
#define THREADS 128

/// @brief Sum one value per thread of the block in a fixed order.
/// @param value This thread's value.
/// @param shared Block-shared buffer of `THREADS` entries.
/// @return The block sum, in thread zero.
__device__ double reduce(double value, double *shared) {
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

/// @brief Contract generalised-Fock rows, `y_l = \sum_r (F_{lr} - shift S_{lr}) c_r`, over the
/// GPU-evaluated reference pairs.
/// @param pairs Location of each ordered reference pair's combinations, `nref x nref`.
/// @param combinations Wick intermediates of every combination.
/// @param nref Number of references.
/// @param slab Device slab of `DeviceWicks`.
/// @param parents Orthogonal integrals record of each parent.
/// @param integrals Device array of `DeviceOrthogonal`.
/// @param left Bra determinants, one row each.
/// @param right Ket determinants, the order of `c`.
/// @param c Coefficients of the ket determinants.
/// @param nright Number of ket determinants.
/// @param shift Overlap shift of the Fock operator.
/// @param y Contracted rows to write, one per bra.
extern "C" __global__ void contract_fock_rows(
    const DeviceWicksPair *pairs,
    const DeviceWicksCombination *combinations,
    unsigned long long nref,
    const double *slab,
    const DeviceOrthogonalParent *parents,
    const double *integrals,
    const DeviceDeterminant *left,
    const DeviceDeterminant *right,
    const double *c,
    unsigned long long nright,
    double shift,
    double *y
) {
    __shared__ double shared[THREADS];
    const DeviceDeterminant bra = left[blockIdx.x];

    double sum = 0.0;
    for (unsigned long long r = threadIdx.x; r < nright; r += THREADS) {
        const DeviceDeterminant ket = right[r];
        double f, s;
        if (bra.parent == ket.parent) {
            const DeviceOrthogonalParent &parent = parents[bra.parent];
            if (!parent.fock) {
                continue;
            }
            calculate_fs_pair_orthogonal(parent, integrals, bra, ket, &f, &s);
        } else {
            const DeviceWicksPair &pair = pairs[bra.parent * nref + ket.parent];
            if (!pair.evaluated) {
                continue;
            }
            calculate_fs_pair_wicks(pair, combinations, slab, bra, ket, &f, &s);
        }
        sum = fma(f - shift * s, c[r], sum);
    }

    double total = reduce(sum, shared);
    if (threadIdx.x == 0) {
        y[blockIdx.x] = total;
    }
}

/// @brief Contract Hamiltonian and overlap rows, `v_l = \sum_r H_{lr} c_r` and
/// `s_l = \sum_r S_{lr} c_r`, over the GPU-evaluated reference pairs.
/// @param pairs Location of each ordered reference pair's combinations, `nref x nref`.
/// @param combinations Wick intermediates of every combination.
/// @param nref Number of references.
/// @param slab Device slab of `DeviceWicks`.
/// @param parents Orthogonal integrals record of each parent.
/// @param integrals Device array of `DeviceOrthogonal`.
/// @param left Bra determinants, one row each.
/// @param right Ket determinants, the order of `c`.
/// @param c Coefficients of the ket determinants.
/// @param nright Number of ket determinants.
/// @param enuc Nuclear repulsion energy.
/// @param v Contracted Hamiltonian rows to write, one per bra.
/// @param s Contracted overlap rows to write, one per bra.
extern "C" __global__ void contract_hamiltonian_overlap_rows(
    const DeviceWicksPair *pairs,
    const DeviceWicksCombination *combinations,
    unsigned long long nref,
    const double *slab,
    const DeviceOrthogonalParent *parents,
    const double *integrals,
    const DeviceDeterminant *left,
    const DeviceDeterminant *right,
    const double *c,
    unsigned long long nright,
    double enuc,
    double *v,
    double *s
) {
    __shared__ double shared[THREADS];
    const DeviceDeterminant bra = left[blockIdx.x];

    double hsum = 0.0;
    double ssum = 0.0;
    for (unsigned long long r = threadIdx.x; r < nright; r += THREADS) {
        const DeviceDeterminant ket = right[r];
        double h, o;
        if (bra.parent == ket.parent) {
            const DeviceOrthogonalParent &parent = parents[bra.parent];
            if (!parent.hamiltonian) {
                continue;
            }
            calculate_hs_pair_orthogonal(parent, integrals, bra, ket, enuc, &h, &o);
        } else {
            const DeviceWicksPair &pair = pairs[bra.parent * nref + ket.parent];
            if (!pair.evaluated) {
                continue;
            }
            calculate_hs_pair_wicks(pair, combinations, slab, bra, ket, enuc, &h, &o);
        }
        hsum = fma(h, c[r], hsum);
        ssum = fma(o, c[r], ssum);
    }

    double htotal = reduce(hsum, shared);
    __syncthreads();
    double stotal = reduce(ssum, shared);
    if (threadIdx.x == 0) {
        v[blockIdx.x] = htotal;
        s[blockIdx.x] = stotal;
    }
}
