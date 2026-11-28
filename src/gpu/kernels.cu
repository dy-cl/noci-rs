// gpu/kernels.cu
// CUDA kernels of the `gpu` feature, compiled by `build/build.rs` into one fatbin and loaded by
// `gpu::Device`. Every kernel has C linkage so the host looks it up by name.

/// Accumulate `y[index[i]] += values[i]` for `n` gathered values. Each index appears once per call,
/// as each target determinant is written by one gathered row, so no atomics are needed.
/// # Arguments:
/// - `y`: Output vector to accumulate into.
/// - `index`: Output position of each value.
/// - `values`: Values to add.
/// - `n`: Number of values.
extern "C" __global__ void scatter_add(
    double *y,
    const unsigned long long *index,
    const double *values,
    unsigned long long n
) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) {
        y[index[i]] += values[i];
    }
}
