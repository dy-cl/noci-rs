// noci/selected/operators.rs

// Standard library imports.
use std::fs::OpenOptions;
use std::path::Path;

// External crate imports.
use memmap2::{MmapMut, MmapOptions};
use mpi::topology::Communicator;
use ndarray::Array1;
use num_complex::Complex64;
use rayon::prelude::*;

// Crate-root imports.
use crate::NOCIScalar;
use crate::determinant::NOCIIndex;
use crate::elements::calculate_pairs;
use crate::elements::nonorthogonalwicks::WickScratchSpin;
use crate::elements::{
    DetPair, Fock, FockData, HamiltonianOverlap, NOCIData, Overlap, PairOperator,
};
use crate::mpiutils::all_reduce_array1;
use crate::noci::contract_noci_rows;
use crate::noci::{OneBodyFactorisation, OneBodyScratch};
use crate::time_call;

// Parent/sibling imports.
use super::{PT2ProjectedOperator, PT2Projection, Preconditioner};

/// Number of diagonal candidate pairs evaluated per batched dispatch.
const DIAGCHUNK: usize = 256;

/// Relative score difference below which candidates are tied and selected together.
const TIETOL: f64 = 1e-4;

pub(in crate::noci::selected) enum CandidateM<T: NOCIScalar> {
    /// Packed candidate-candidate matrix stored in process memory.
    Ram(Vec<T>),
    /// Packed candidate-candidate matrix stored in a writable memory map.
    Disk(MmapMut),
}

impl<T: NOCIScalar> CandidateM<T> {
    /// Borrow the packed upper-triangular candidate matrix as scalar values.
    /// Disk storage is reinterpreted using the scalar type and byte length fixed at creation.
    /// # Arguments:
    /// - `self`: Candidate matrix stored in RAM or a memory map.
    /// # Returns
    /// - `&[T]`: Immutable packed upper triangle in row-major packed order.
    pub(in crate::noci::selected) fn as_slice(&self) -> &[T] {
        match self {
            Self::Ram(m) => m.as_slice(),
            Self::Disk(m) => unsafe {
                // Mmap length is set from `len * size_of::<T>()` when the packed matrix is built.
                std::slice::from_raw_parts(
                    m.as_ptr() as *const T,
                    m.len() / std::mem::size_of::<T>(),
                )
            },
        }
    }

    /// Mutably borrow the packed upper-triangular candidate matrix as scalar values.
    /// Disk storage is reinterpreted using the scalar type and byte length fixed at creation.
    /// # Arguments:
    /// - `self`: Candidate matrix stored in RAM or a writable memory map.
    /// # Returns
    /// - `&mut [T]`: Mutable packed upper triangle in row-major packed order.
    pub(in crate::noci::selected) fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            Self::Ram(m) => m.as_mut_slice(),
            Self::Disk(m) => unsafe {
                // Mmap length is set from `len * size_of::<T>()` when the packed matrix is built.
                std::slice::from_raw_parts_mut(
                    m.as_mut_ptr() as *mut T,
                    m.len() / std::mem::size_of::<T>(),
                )
            },
        }
    }
}

/// Build the projection contractions and the projected coupling vector of the NOCI-PT2 problem
/// from the current NOCI state `|\Psi_0\rangle = \sum_i c_i |i\rangle`, contracting every
/// candidate-current block with `c` row by row rather than storing it:
/// `E_0 = \langle \Psi_0|\hat F|\Psi_0\rangle`, `V_a = \sum_i H_{ai} c_i`,
/// `S_{a0} = \sum_i S_{ai} c_i` and `F_{a0} = \sum_i F_{ai} c_i`. A projected candidate
/// `|\Omega_a\rangle = |\Phi_a\rangle - |\Psi_0\rangle\langle\Psi_0|\Phi_a\rangle` then has
/// `V_a^\Omega = V_a - E_{\text{NOCI}} S_{a0}`, and since `S` and `F` are Hermitian,
/// `S_{0a} = S_{a0}^*` and `F_{0a} = F_{a0}^*`.
/// # Arguments:
/// - `data`: Shared NOCI matrix-element data over the selected and candidate determinants.
/// - `fock`: Fock-specific matrix-element data.
/// - `selected_space`: Current selected nonorthogonal determinant space.
/// - `candidates`: Candidate determinant space.
/// - `coeffs`: Current-space NOCI eigenvector.
/// - `e_noci`: Current selected-space NOCI energy.
/// # Returns:
/// - `(PT2Projection<T>, Array1<T>)`: Projection contractions used to form `M^\Omega`, and the
///   projected coupling vector `V^\Omega`.
pub(in crate::noci::selected) fn build_snoci_couplings<T: NOCIScalar + Into<Complex64>>(
    data: &NOCIData<'_, T>,
    fock: &FockData<'_, T>,
    selected_space: &[NOCIIndex],
    candidates: &[NOCIIndex],
    coeffs: &Array1<T>,
    e_noci: f64,
) -> (PT2Projection<T>, Array1<T>) {
    time_call!(crate::timers::selected::add_build_snoci_couplings, {
        let op = Fock { fock, shift: 0.0 };
        let fock_row = |y: &mut T, f: T, c: T| *y += f * c;

        // `E_0 = \sum_{ij} c_i^* F_{ij} c_j` and `F_{a0}` from the Fock rows.
        let f_i0 = contract_noci_rows(data, &op, selected_space, selected_space, coeffs, fock_row);
        let e0 = coeffs
            .iter()
            .zip(&f_i0)
            .fold(T::from_real(0.0), |acc, (&c, &f)| acc + c.conj() * f);
        let e0 = e0.into().re;

        let f_a0 = contract_noci_rows(data, &op, candidates, selected_space, coeffs, fock_row);
        let f_a0 = Array1::from(f_a0);

        // `V_a` and `S_{a0}` from the fused Hamiltonian and overlap rows.
        let hs = contract_noci_rows(
            data,
            &HamiltonianOverlap,
            candidates,
            selected_space,
            coeffs,
            |y: &mut (T, T), (h, s), c| {
                y.0 += h * c;
                y.1 += s * c;
            },
        );
        let (v_a, s_a0): (Vec<T>, Vec<T>) = hs.into_iter().unzip();
        let (v_a, s_a0) = (Array1::from(v_a), Array1::from(s_a0));

        // `V^\Omega_a = V_a - E_{\text{NOCI}} S_{a0}`, and the projection with
        // `S_{0a} = S_{a0}^*` and `F_{0a} = F_{a0}^*`.
        let v_omega = &v_a - &s_a0.mapv(|x| T::from_real(e_noci) * x);
        let projection = PT2Projection {
            e0,
            s_0a: s_a0.mapv(|x| x.conj()),
            s_a0,
            f_0a: f_a0.mapv(|x| x.conj()),
            f_a0,
        };

        (projection, v_omega)
    })
}

/// Build the upper triangle of the unprojected candidate-candidate shifted Fock matrix `M` in RAM.
/// # Arguments:
/// - `op`: Matrix-free projected NOCI-PT2 operator data.
/// # Returns:
/// - `CandidateM<T>`: Packed upper triangle of `M`.
pub(in crate::noci::selected) fn build_candidate_m<T: NOCIScalar>(
    op: &PT2ProjectedOperator<'_, '_, '_, T>
) -> CandidateM<T> {
    time_call!(crate::timers::selected::add_build_candidate_m, {
        let n = op.candidates.len();
        let len = n
            .checked_mul(n + 1)
            .and_then(|x| x.checked_div(2))
            .expect("packed candidate matrix length overflow");
        let mut m = Vec::new();
        m.try_reserve_exact(len)
            .expect("failed to allocate packed candidate matrix in RAM");
        m.resize(len, T::from_real(0.0));

        let mut m = CandidateM::Ram(m);
        fill_candidate_m(op, m.as_mut_slice());
        m
    })
}

/// Build the upper triangle of the unprojected candidate-candidate shifted Fock matrix `M` in a
/// writable memory map.
/// # Arguments:
/// - `op`: Matrix-free projected NOCI-PT2 operator data.
/// - `path`: File path for the packed matrix.
/// # Returns:
/// - `CandidateM<T>`: File-backed packed upper triangle of `M`.
pub(in crate::noci::selected) fn build_candidate_m_disk<T: NOCIScalar>(
    op: &PT2ProjectedOperator<'_, '_, '_, T>,
    path: &Path,
) -> CandidateM<T> {
    time_call!(crate::timers::selected::add_build_candidate_m, {
        let n = op.candidates.len();
        // Packed upper-triangle storage has `n(n+1)/2` elements; check the
        // element and byte counts before sizing the memory-mapped file.
        let len = n
            .checked_mul(n + 1)
            .and_then(|x| x.checked_div(2))
            .expect("packed candidate matrix length overflow");
        let nbytes = len
            .checked_mul(std::mem::size_of::<T>())
            .expect("packed candidate matrix byte size overflow");
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .unwrap();
        file.set_len(nbytes as u64).unwrap();

        let mmap = unsafe { MmapOptions::new().len(nbytes).map_mut(&file).unwrap() };
        let mut m = CandidateM::Disk(mmap);
        fill_candidate_m(op, m.as_mut_slice());
        m
    })
}

/// Evaluate every upper-triangular unprojected candidate matrix element in parallel.
/// Stores `M_ab = <Phi_a|(F - E_0)|Phi_b>` in packed row-major order for `a <= b`.
/// # Arguments:
/// - `op`: Matrix-element data, candidates, Fock operator, and zeroth-order energy.
/// - `m`: Mutable packed upper-triangular destination of length `n(n+1)/2`.
/// # Returns
/// - `()`: Fills every packed candidate matrix entry.
fn fill_candidate_m<T: NOCIScalar>(
    op: &PT2ProjectedOperator<'_, '_, '_, T>,
    m: &mut [T],
) {
    // Share the destination address because each parallel task owns one disjoint packed row.
    let n = op.candidates.len();
    let m_addr = m.as_mut_ptr() as usize;

    (0..n).into_par_iter().for_each_init(
        || (WickScratchSpin::new(), Vec::new()),
        |(scratch, pairs), a| {
            let ldet = op.candidates[a];
            let row = a * (2 * n - a + 1) / 2;
            let row_len = n - a;
            // Rows in packed upper-triangular storage are disjoint, and each `a` is visited once.
            let row =
                unsafe { std::slice::from_raw_parts_mut((m_addr as *mut T).add(row), row_len) };

            // Evaluate the packed row as one batch of `(a, b)` pairs.
            pairs.clear();
            pairs.extend(
                op.candidates[a..]
                    .iter()
                    .map(|&gdet| DetPair::from_space(op.data.space, ldet, gdet)),
            );
            calculate_pairs(
                op.data,
                &Fock {
                    fock: op.fock,
                    shift: op.projection.e0,
                },
                pairs,
                Some(scratch),
                row,
            );
        },
    );
}

/// Build the diagonal of the unprojected candidate-candidate shifted Fock matrix `M`.
/// Uses a cached packed matrix if provided, otherwise evaluates diagonal matrix elements on the fly.
/// # Arguments:
/// - `op`: Matrix-free projected NOCI-PT2 operator data.
/// - `m`: Optional packed candidate-candidate shifted Fock matrix.
/// # Returns:
/// - `Array1<T>`: Diagonal entries of `M`.
pub(in crate::noci::selected) fn build_candidate_m_diag<T: NOCIScalar>(
    op: &PT2ProjectedOperator<'_, '_, '_, T>,
    m: Option<&[T]>,
) -> Array1<T> {
    time_call!(crate::timers::selected::add_build_candidate_m_diag, {
        let n = op.candidates.len();

        // In packed upper-triangle storage, diagonal `i` begins at
        // `i (2n - i + 1) / 2`; otherwise evaluate each pair `(i,i)`.
        if let Some(m) = m {
            let mut diag = Array1::from_elem(n, T::from_real(0.0));

            for i in 0..n {
                let k = i * (2 * n - i + 1) / 2;
                diag[i] = m[k];
            }

            return diag;
        }

        // Evaluate diagonal pairs `(a, a)` in batched chunks.
        let mut diag = vec![T::from_real(0.0); n];
        diag.par_chunks_mut(DIAGCHUNK)
            .zip(op.candidates.par_chunks(DIAGCHUNK))
            .for_each_init(
                || (WickScratchSpin::new(), Vec::new()),
                |(scratch, pairs), (diag, dets)| {
                    pairs.clear();
                    pairs.extend(
                        dets.iter()
                            .map(|&det| DetPair::from_space(op.data.space, det, det)),
                    );
                    calculate_pairs(
                        op.data,
                        &Fock {
                            fock: op.fock,
                            shift: op.projection.e0,
                        },
                        pairs,
                        Some(scratch),
                        diag,
                    );
                },
            );

        Array1::from_vec(diag)
    })
}

/// Build the diagonal of the unprojected candidate-candidate overlap matrix `S`.
/// # Arguments:
/// - `op`: Matrix-free projected NOCI-PT2 operator data.
/// # Returns:
/// - `Array1<T>`: Diagonal entries of `S`.
pub(in crate::noci::selected) fn build_candidate_s_diag<T: NOCIScalar>(
    op: &PT2ProjectedOperator<'_, '_, '_, T>
) -> Array1<T> {
    // Evaluate diagonal pairs `(a, a)` in batched chunks.
    let mut diag = vec![T::from_real(0.0); op.candidates.len()];
    diag.par_chunks_mut(DIAGCHUNK)
        .zip(op.candidates.par_chunks(DIAGCHUNK))
        .for_each_init(
            || (WickScratchSpin::new(), Vec::new()),
            |(scratch, pairs), (diag, dets)| {
                pairs.clear();
                pairs.extend(
                    dets.iter()
                        .map(|&det| DetPair::from_space(op.data.space, det, det)),
                );
                calculate_pairs(op.data, &Overlap, pairs, Some(scratch), diag);
            },
        );

    Array1::from_vec(diag)
}

/// Build unprojected candidate-candidate diagonals from factorised `F + \lambda S` and `S`.
/// # Arguments:
/// - `op`: Candidate operator whose diagonals are requested.
/// - `one_body`: Cached spin-factorised one-body operator.
/// - `lambda`: Scalar overlap shift in `F + \lambda S`.
/// # Returns:
/// - `(Array1<T>, Array1<T>)`: Diagonal of `F + \lambda S` and diagonal of `S`.
pub(in crate::noci::selected) fn build_factorised_candidate_diags<T: NOCIScalar>(
    op: &PT2ProjectedOperator<'_, '_, '_, T>,
    one_body: &OneBodyFactorisation<T>,
    lambda: T,
) -> (Array1<T>, Array1<T>) {
    one_body.one_body_diagonals(op.data, op.fock, lambda)
}

/// Apply the rows of a Hermitian candidate-candidate matrix `A_{ab} = \langle a|\hat O|b\rangle`
/// owned by this process to `x`, where row `a` is owned when `a = a_0 + k s` for offset `a_0`
/// and stride `s`.
/// Each owned row adds its upper-triangle pairs `b \ge a` as `y_a \mathrel{+}= A_{ab} x_b` and,
/// for `b \ne a`, the Hermitian partner `y_b \mathrel{+}= A_{ab}^* x_a`, so the partial products
/// of the offsets `0, \ldots, s - 1` sum to `y = A x`. Elements are read from the packed upper
/// triangle when it is given and are otherwise evaluated on demand, skipping pairs with
/// `x_a = x_b = 0`. `A_{ab}` uses chemistry scalar `T` and `x` uses Krylov scalar `R`.
/// # Arguments:
/// - `op`: Matrix-free projected NOCI-PT2 operator data.
/// - `pair_op`: Pair operator `\hat O` evaluating `A_{ab}` on demand.
/// - `packed`: Optional packed upper triangle of `A`, used in place of `pair_op`.
/// - `x`: Krylov vector to apply `A` to.
/// - `offset`: First owned row `a_0`.
/// - `stride`: Row stride `s`.
/// # Returns:
/// - `Vec<R>`: Partial product of the owned rows.
fn apply_owned_rows<T, R, Op>(
    op: &PT2ProjectedOperator<'_, '_, '_, T>,
    pair_op: &Op,
    packed: Option<&[T]>,
    x: &Array1<R>,
    offset: usize,
    stride: usize,
) -> Vec<R>
where
    T: NOCIScalar,
    R: NOCIScalar + From<T>,
    Op: PairOperator<T, Out = T> + Sync,
{
    let n = op.candidates.len();
    let zero = R::from_real(0.0);

    if offset >= n {
        return vec![zero; n];
    }

    let xs = x.as_slice_memory_order().unwrap();
    let nrows = (n - offset).div_ceil(stride);
    let min_len = (nrows / (rayon::current_num_threads() * 8)).max(1);

    // Rayon splits the owned rows; each worker accumulates into its own output vector.
    (0..nrows)
        .into_par_iter()
        .with_min_len(min_len)
        .fold(
            || {
                (
                    WickScratchSpin::new(),
                    Vec::new(),
                    Vec::new(),
                    vec![zero; n],
                )
            },
            |(mut scratch, mut pairs, mut values, mut y), k| {
                let a = offset + k * stride;
                let xa = xs[a];

                // Read the row `A_{ab}, b \ge a` from the packed upper triangle, or evaluate
                // its contributing pairs as one batch.
                let row: &[T] = match packed {
                    Some(packed) => {
                        let start = a * (2 * n - a + 1) / 2;
                        &packed[start..start + n - a]
                    }
                    None => {
                        let ldet = op.candidates[a];
                        pairs.clear();
                        for (&xb, &gdet) in xs[a..].iter().zip(&op.candidates[a..]) {
                            if xa == zero && xb == zero {
                                continue;
                            }
                            pairs.push(DetPair::from_space(op.data.space, ldet, gdet));
                        }
                        values.clear();
                        values.resize(pairs.len(), T::from_real(0.0));
                        calculate_pairs(op.data, pair_op, &pairs, Some(&mut scratch), &mut values);
                        &values
                    }
                };

                // Scatter `A_{ab} x_b` to row `a` and `A_{ab}^* x_a` to row `b`.
                let cols = (a..n).filter(|&b| packed.is_some() || xa != zero || xs[b] != zero);
                for (b, &a_ab) in cols.zip(row) {
                    let xb = xs[b];
                    let a_ab = <R as From<T>>::from(a_ab);

                    if xb != zero {
                        y[a] += a_ab * xb;
                    }

                    if b != a && xa != zero {
                        y[b] += a_ab.conj() * xa;
                    }
                }

                (scratch, pairs, values, y)
            },
        )
        .map(|(_, _, _, y)| y)
        .reduce(
            || vec![zero; n],
            |mut lhs, rhs| {
                for i in 0..n {
                    lhs[i] += rhs[i];
                }
                lhs
            },
        )
}

/// Apply the unprojected candidate-candidate shifted Fock matrix `M = F - E_0 S`.
/// Uses a cached packed matrix if provided, otherwise evaluates matrix elements on the fly.
/// # Arguments:
/// - `op`: Matrix-free projected NOCI-PT2 operator data.
/// - `x`: Krylov vector to apply `M` to.
/// - `m`: Optional packed candidate-candidate shifted Fock matrix.
/// # Returns:
/// - `Array1<R>`: Matrix-vector product `M x`.
pub(in crate::noci::selected) fn apply_candidate_m<T, R>(
    op: &PT2ProjectedOperator<'_, '_, '_, T>,
    x: &Array1<R>,
    m: Option<&[T]>,
) -> Array1<R>
where
    T: NOCIScalar,
    R: NOCIScalar + From<T>,
{
    time_call!(crate::timers::selected::add_apply_candidate_m, {
        let fock = Fock {
            fock: op.fock,
            shift: op.projection.e0,
        };

        Array1::from_vec(apply_owned_rows(op, &fock, m, x, 0, 1))
    })
}

/// Apply the unprojected candidate-candidate shifted Fock matrix `M = F - E_0 S` across MPI
/// ranks. Each rank applies every `n_{\text{ranks}}`-th row with Rayon, and the partial
/// products are summed into the replicated result on every rank.
/// # Arguments:
/// - `op`: Matrix-free projected NOCI-PT2 operator data.
/// - `x`: Replicated Krylov vector to apply `M` to.
/// - `m`: Optional packed candidate-candidate shifted Fock matrix.
/// - `world`: MPI communicator.
/// # Returns:
/// - `Array1<R>`: Globally reduced matrix-vector product `M x`.
pub(in crate::noci::selected) fn apply_candidate_m_mpi<T, R>(
    op: &PT2ProjectedOperator<'_, '_, '_, T>,
    x: &Array1<R>,
    m: Option<&[T]>,
    world: &impl Communicator,
) -> Array1<R>
where
    T: NOCIScalar,
    R: NOCIScalar + From<T> + Into<Complex64>,
{
    time_call!(crate::timers::selected::add_apply_candidate_m, {
        if op.candidates.is_empty() {
            return Array1::from_vec(Vec::new());
        }

        let fock = Fock {
            fock: op.fock,
            shift: op.projection.e0,
        };
        let irank = world.rank() as usize;
        let nranks = world.size() as usize;

        // Hermitian-partner contributions overlap across ranks, so sum the partial products.
        let y = apply_owned_rows(op, &fock, m, x, irank, nranks);
        all_reduce_array1(world, Array1::from_vec(y))
    })
}

/// Apply the unprojected candidate-candidate overlap matrix `S`.
/// # Arguments:
/// - `op`: Matrix-free projected NOCI-PT2 operator data.
/// - `x`: Krylov vector to apply `S` to.
/// # Returns:
/// - `Array1<R>`: Matrix-vector product `S x`.
pub(in crate::noci::selected) fn apply_candidate_s<T, R>(
    op: &PT2ProjectedOperator<'_, '_, '_, T>,
    x: &Array1<R>,
) -> Array1<R>
where
    T: NOCIScalar,
    R: NOCIScalar + From<T>,
{
    Array1::from_vec(apply_owned_rows(op, &Overlap, None, x, 0, 1))
}

/// Apply the unprojected candidate-candidate overlap matrix `S` across MPI ranks.
/// Each rank applies every `n_{\text{ranks}}`-th row with Rayon, and the partial products are
/// summed into the replicated result on every rank.
/// # Arguments:
/// - `op`: Matrix-free projected NOCI-PT2 operator data.
/// - `x`: Replicated Krylov vector to apply `S` to.
/// - `world`: MPI communicator.
/// # Returns:
/// - `Array1<R>`: Globally reduced matrix-vector product `S x`.
pub(in crate::noci::selected) fn apply_candidate_s_mpi<T, R>(
    op: &PT2ProjectedOperator<'_, '_, '_, T>,
    x: &Array1<R>,
    world: &impl Communicator,
) -> Array1<R>
where
    T: NOCIScalar,
    R: NOCIScalar + From<T> + Into<Complex64>,
{
    if op.candidates.is_empty() {
        return Array1::from_vec(Vec::new());
    }

    let irank = world.rank() as usize;
    let nranks = world.size() as usize;

    // Hermitian-partner contributions overlap across ranks, so sum the partial products.
    let y = apply_owned_rows(op, &Overlap, None, x, irank, nranks);
    all_reduce_array1(world, Array1::from_vec(y))
}

/// Apply the projected NOCI-PT2 shifted Fock matrix `M^Omega`.
/// Uses a cached packed candidate-candidate shifted Fock matrix if provided,
/// otherwise applies `M` without materialising it.
/// # Arguments:
/// - `op`: Matrix-free projected NOCI-PT2 operator data.
/// - `p`: Projection contractions represented in Krylov scalar `R`.
/// - `x`: Krylov vector to apply `M^Omega` to.
/// - `m`: Optional packed unprojected candidate-candidate shifted Fock matrix `M`.
/// # Returns:
/// - `Array1<R>`: Matrix-vector product `M^Omega x`.
pub(in crate::noci::selected) fn apply_omega_m<T, R>(
    op: &PT2ProjectedOperator<'_, '_, '_, T>,
    p: &PT2Projection<R>,
    x: &Array1<R>,
    m: Option<&[T]>,
) -> Array1<R>
where
    T: NOCIScalar,
    R: NOCIScalar + From<T>,
{
    time_call!(crate::timers::selected::add_apply_omega_m, {
        // `M^Omega_{ab} = M_{ab} - F_{a0} S_{0b} - S_{a0} F_{0b}
        //                  + 2 E_0 S_{a0} S_{0b}`.
        let sx = p.s_0a.dot(x);
        let fx = p.f_0a.dot(x);
        let mut y = apply_candidate_m(op, x, m);
        apply_projection_correction(&mut y, p, sx, fx, R::from_real(0.0));

        y
    })
}

/// Apply the projected NOCI-PT2 shifted Fock matrix `M^Omega` across MPI ranks.
/// # Arguments:
/// - `op`: Matrix-free projected NOCI-PT2 operator data.
/// - `p`: Projection contractions represented in Krylov scalar `R`.
/// - `x`: Replicated Krylov vector to apply `M^Omega` to.
/// - `m`: Optional packed unprojected candidate-candidate shifted Fock matrix `M`.
/// - `world`: MPI communicator.
/// # Returns:
/// - `Array1<R>`: Globally reduced matrix-vector product `M^Omega x`.
pub(in crate::noci::selected) fn apply_omega_m_mpi<T, R>(
    op: &PT2ProjectedOperator<'_, '_, '_, T>,
    p: &PT2Projection<R>,
    x: &Array1<R>,
    m: Option<&[T]>,
    world: &impl Communicator,
) -> Array1<R>
where
    T: NOCIScalar,
    R: NOCIScalar + From<T> + Into<Complex64>,
{
    time_call!(crate::timers::selected::add_apply_omega_m, {
        // `M^Omega_{ab} = M_{ab} - F_{a0} S_{0b} - S_{a0} F_{0b}
        //                  + 2 E_0 S_{a0} S_{0b}`.
        let sx = p.s_0a.dot(x);
        let fx = p.f_0a.dot(x);

        let mut y = apply_candidate_m_mpi(op, x, m, world);
        apply_projection_correction(&mut y, p, sx, fx, R::from_real(0.0));

        y
    })
}

/// Apply factorised projected `(F - E0 S + i epsilon Q)x`.
/// # Arguments:
/// - `op`: Matrix-free projected NOCI-PT2 operator data.
/// - `p`: Projection contractions represented in Krylov scalar `R`.
/// - `one_body`: Cached spin-factorised one-body operator.
/// - `scratch`: Reusable one-body contraction buffers in scalar `R`.
/// - `x`: Krylov vector to apply shifted operator to.
/// - `imag_shift`: Imaginary shift strength `epsilon`.
/// # Returns:
/// - `Array1<R>`: Matrix-vector product `(M^Omega + i epsilon Q)x`.
pub(in crate::noci::selected) fn apply_factorised_shifted_omega_m<T, R>(
    op: &PT2ProjectedOperator<'_, '_, '_, T>,
    p: &PT2Projection<R>,
    one_body: &OneBodyFactorisation<T>,
    scratch: &mut OneBodyScratch<R>,
    x: &Array1<R>,
    imag_shift: f64,
) -> Array1<R>
where
    T: NOCIScalar,
    R: NOCIScalar + From<T>,
{
    if x.iter().all(|&xi| xi == R::from_real(0.0)) {
        return Array1::from_elem(x.len(), R::from_real(0.0));
    }
    // `M^Omega(epsilon) = F - E_0 S + i epsilon S
    //                     - F_{a0} S_{0b} - S_{a0} F_{0b}
    //                     + (2 E_0 - i epsilon) S_{a0} S_{0b}`.
    let sx = p.s_0a.dot(x);
    let fx = p.f_0a.dot(x);
    let lambda = R::from_real(-p.e0) + R::from_imag(imag_shift);
    let mut y = one_body.apply_one_body(x, op.data, op.fock, lambda, scratch, (0, 1));
    apply_projection_correction(&mut y, p, sx, fx, R::from_imag(imag_shift));
    y
}

/// Apply factorised projected `(F - E0 S + i epsilon Q)x` across MPI ranks.
/// # Arguments:
/// - `op`: Matrix-free projected NOCI-PT2 operator data.
/// - `p`: Projection contractions represented in Krylov scalar `R`.
/// - `one_body`: Cached spin-factorised one-body operator.
/// - `scratch`: Reusable one-body contraction buffers in scalar `R`.
/// - `x`: Replicated Krylov vector to apply shifted operator to.
/// - `world`: MPI communicator.
/// - `imag_shift`: Imaginary shift strength `epsilon`.
/// # Returns:
/// - `Array1<R>`: Globally reduced matrix-vector product `(M^Omega + i epsilon Q)x`.
pub(in crate::noci::selected) fn apply_factorised_shifted_omega_m_mpi<T, R>(
    op: &PT2ProjectedOperator<'_, '_, '_, T>,
    p: &PT2Projection<R>,
    one_body: &OneBodyFactorisation<T>,
    scratch: &mut OneBodyScratch<R>,
    x: &Array1<R>,
    world: &impl Communicator,
    imag_shift: f64,
) -> Array1<R>
where
    T: NOCIScalar,
    R: NOCIScalar + From<T> + Into<Complex64>,
{
    if x.iter().all(|&xi| xi == R::from_real(0.0)) {
        return Array1::from_elem(x.len(), R::from_real(0.0));
    }
    // `M^Omega(epsilon) = F - E_0 S + i epsilon S
    //                     - F_{a0} S_{0b} - S_{a0} F_{0b}
    //                     + (2 E_0 - i epsilon) S_{a0} S_{0b}`.
    let sx = p.s_0a.dot(x);
    let fx = p.f_0a.dot(x);
    let lambda = R::from_real(-p.e0) + R::from_imag(imag_shift);
    let partial = one_body.apply_one_body(
        x,
        op.data,
        op.fock,
        lambda,
        scratch,
        (world.rank() as usize, world.size() as usize),
    );
    let mut y = all_reduce_array1(world, partial);
    apply_projection_correction(&mut y, p, sx, fx, R::from_imag(imag_shift));
    y
}

/// Apply the projected first-order-space overlap matrix `Q`.
/// # Arguments:
/// - `op`: Matrix-free projected NOCI-PT2 operator data.
/// - `p`: Projection contractions represented in Krylov scalar `R`.
/// - `x`: Krylov vector to apply `Q` to.
/// # Returns:
/// - `Array1<R>`: Matrix-vector product `Q x`.
pub(in crate::noci::selected) fn apply_omega_s<T, R>(
    op: &PT2ProjectedOperator<'_, '_, '_, T>,
    p: &PT2Projection<R>,
    x: &Array1<R>,
) -> Array1<R>
where
    T: NOCIScalar,
    R: NOCIScalar + From<T>,
{
    // `Q_{ab} = S_{ab} - S_{a0} S_{0b}`.
    let sx = p.s_0a.dot(x);
    let mut y = apply_candidate_s(op, x);

    for a in 0..y.len() {
        y[a] -= p.s_a0[a] * sx;
    }

    y
}

/// Apply the projected first-order-space overlap matrix `Q` across MPI ranks.
/// # Arguments:
/// - `op`: Matrix-free projected NOCI-PT2 operator data.
/// - `p`: Projection contractions represented in Krylov scalar `R`.
/// - `x`: Replicated Krylov vector to apply `Q` to.
/// - `world`: MPI communicator.
/// # Returns:
/// - `Array1<R>`: Globally reduced matrix-vector product `Q x`.
pub(in crate::noci::selected) fn apply_omega_s_mpi<T, R>(
    op: &PT2ProjectedOperator<'_, '_, '_, T>,
    p: &PT2Projection<R>,
    x: &Array1<R>,
    world: &impl Communicator,
) -> Array1<R>
where
    T: NOCIScalar,
    R: NOCIScalar + From<T> + Into<Complex64>,
{
    // `Q_{ab} = S_{ab} - S_{a0} S_{0b}`.
    let sx = p.s_0a.dot(x);

    let mut y = apply_candidate_s_mpi(op, x, world);
    for a in 0..y.len() {
        y[a] -= p.s_a0[a] * sx;
    }

    y
}

/// Apply the projected first-order-space overlap matrix `Q` with the factorised one-body tables.
/// The tables apply `(F + \lambda S)x` for any scalar `\lambda`, so
/// `S x = (F + S)x - F x`, and then `Q_{ab} = S_{ab} - S_{a0} S_{0b}`. Each rank applies its
/// share of the target rows and the partial products are summed over ranks.
/// # Arguments:
/// - `op`: Matrix-free projected NOCI-PT2 operator data.
/// - `p`: Projection contractions represented in Krylov scalar `R`.
/// - `one_body`: Cached spin-factorised one-body operator.
/// - `scratch`: Reusable one-body contraction buffers in scalar `R`.
/// - `x`: Replicated Krylov vector to apply `Q` to.
/// - `world`: MPI communicator.
/// # Returns:
/// - `Array1<R>`: Globally reduced matrix-vector product `Q x`.
pub(in crate::noci::selected) fn apply_factorised_omega_s<T, R>(
    op: &PT2ProjectedOperator<'_, '_, '_, T>,
    p: &PT2Projection<R>,
    one_body: &OneBodyFactorisation<T>,
    scratch: &mut OneBodyScratch<R>,
    x: &Array1<R>,
    world: &impl Communicator,
) -> Array1<R>
where
    T: NOCIScalar,
    R: NOCIScalar + From<T> + Into<Complex64>,
{
    let partition = (world.rank() as usize, world.size() as usize);

    // `S x` as the difference of the one-body products at `\lambda = 1` and `\lambda = 0`.
    let fsx = one_body.apply_one_body(x, op.data, op.fock, R::from_real(1.0), scratch, partition);
    let fx = one_body.apply_one_body(x, op.data, op.fock, R::from_real(0.0), scratch, partition);
    let mut y = all_reduce_array1(world, fsx - fx);

    // Remove the component along the zeroth-order state.
    let sx = p.s_0a.dot(x);
    for a in 0..y.len() {
        y[a] -= p.s_a0[a] * sx;
    }

    y
}

/// Apply the explicit imaginary-shifted NOCI-PT2 matrix `M + i epsilon Q`.
/// # Arguments:
/// - `op`: Matrix-free projected NOCI-PT2 operator data.
/// - `p`: Projection contractions represented in Krylov scalar `R`.
/// - `x`: Krylov vector to apply the shifted matrix to.
/// - `m`: Optional packed unprojected candidate-candidate shifted Fock matrix `M`.
/// - `imag_shift`: Imaginary shift strength `epsilon`.
/// # Returns:
/// - `Array1<R>`: Matrix-vector product `(M + i epsilon Q) x`.
pub(in crate::noci::selected) fn apply_shifted_omega_m<T, R>(
    op: &PT2ProjectedOperator<'_, '_, '_, T>,
    p: &PT2Projection<R>,
    x: &Array1<R>,
    m: Option<&[T]>,
    imag_shift: f64,
) -> Array1<R>
where
    T: NOCIScalar,
    R: NOCIScalar + From<T>,
{
    let mut y = apply_omega_m(op, p, x, m);

    if imag_shift != 0.0 {
        let iq = R::from_imag(imag_shift);
        let qx = apply_omega_s(op, p, x);
        for a in 0..y.len() {
            y[a] += iq * qx[a];
        }
    }

    y
}

/// Apply the explicit imaginary-shifted NOCI-PT2 matrix `M + i epsilon Q` across MPI ranks.
/// # Arguments:
/// - `op`: Matrix-free projected NOCI-PT2 operator data.
/// - `p`: Projection contractions represented in Krylov scalar `R`.
/// - `x`: Replicated Krylov vector to apply the shifted matrix to.
/// - `m`: Optional packed unprojected candidate-candidate shifted Fock matrix `M`.
/// - `world`: MPI communicator.
/// - `imag_shift`: Imaginary shift strength `epsilon`.
/// # Returns:
/// - `Array1<R>`: Globally reduced matrix-vector product `(M + i epsilon Q) x`.
pub(in crate::noci::selected) fn apply_shifted_omega_m_mpi<T, R>(
    op: &PT2ProjectedOperator<'_, '_, '_, T>,
    p: &PT2Projection<R>,
    x: &Array1<R>,
    m: Option<&[T]>,
    world: &impl Communicator,
    imag_shift: f64,
) -> Array1<R>
where
    T: NOCIScalar,
    R: NOCIScalar + From<T> + Into<Complex64>,
{
    let mut y = apply_omega_m_mpi(op, p, x, m, world);

    if imag_shift != 0.0 {
        let iq = R::from_imag(imag_shift);
        let qx = apply_omega_s_mpi(op, p, x, world);
        for a in 0..y.len() {
            y[a] += iq * qx[a];
        }
    }

    y
}

/// Apply shared NOCI-PT2 projection correction to an unprojected one-body action.
/// # Arguments:
/// - `y`: Unprojected candidate-space action to correct in place.
/// - `p`: Projection contractions used to form `M^Omega`.
/// - `sx`: Contraction `s_x = \sum_b S_{0b}x_b`.
/// - `fx`: Contraction `f_x = \sum_b F_{0b}x_b`.
/// - `imag_shift`: Imaginary scalar `i epsilon`.
/// # Returns:
/// - `()`: Applies projection terms in place.
fn apply_projection_correction<R: NOCIScalar>(
    y: &mut Array1<R>,
    p: &PT2Projection<R>,
    sx: R,
    fx: R,
    imag_shift: R,
) {
    let scalar = R::from_real(2.0 * p.e0) - imag_shift;
    for a in 0..y.len() {
        y[a] += -p.f_a0[a] * sx - p.s_a0[a] * fx + scalar * p.s_a0[a] * sx;
    }
}

/// Build a preconditioner for the projected NOCI-PT2 shifted Fock matrix.
/// # Arguments:
/// - `m_diag`: Diagonal of the unprojected candidate-candidate matrix `M`.
/// - `s_diag`: Optional diagonal of the unprojected candidate-candidate overlap matrix `S`.
/// - `p`: Projection contractions used to form `M^Omega`.
/// - `kind`: Requested SNOCI preconditioner type.
/// - `imag_shift`: Imaginary shift strength `epsilon`.
/// # Returns:
/// - `Preconditioner<R>`: Preconditioner for applying an approximate inverse of `M^Omega`.
pub(in crate::noci::selected) fn build_preconditioner<T, R>(
    m_diag: &Array1<T>,
    s_diag: Option<&Array1<T>>,
    p: &PT2Projection<R>,
    kind: crate::input::SNOCIPreconditioner,
    imag_shift: f64,
) -> Preconditioner<R>
where
    T: NOCIScalar,
    R: NOCIScalar + From<T>,
{
    if imag_shift != 0.0 {
        // `diag[M^Omega(epsilon)] = diag[M^Omega] + i epsilon diag[Q]`.
        let s_diag = s_diag.expect("shifted preconditioner requires S diagonal");
        let iq = R::from_imag(imag_shift);
        let diag = if matches!(kind, crate::input::SNOCIPreconditioner::Diag) {
            let q_diag = build_omega_s_diag(s_diag, p);
            build_omega_m_diag(m_diag, p) + q_diag.mapv(|q| iq * q)
        } else {
            Array1::from_iter(
                m_diag
                    .iter()
                    .zip(s_diag.iter())
                    .map(|(&m, &s)| <R as From<T>>::from(m) + iq * <R as From<T>>::from(s)),
            )
        };
        return Preconditioner::new(&diag, p, kind, imag_shift);
    }

    let diag = if matches!(kind, crate::input::SNOCIPreconditioner::Diag) {
        build_omega_m_diag(m_diag, p)
    } else {
        m_diag.mapv(<R as From<T>>::from)
    };

    Preconditioner::new(&diag, p, kind, 0.0)
}

/// Build the diagonal of the projected candidate-candidate shifted Fock matrix `M^Omega`.
///
/// # Arguments:
/// - `m_diag`: Diagonal of the unprojected candidate-candidate matrix `M`.
/// - `p`: Projection contractions used to form `M^Omega`.
///
/// # Returns:
/// - `Array1<R>`: Diagonal entries of `M^Omega`.
pub(in crate::noci::selected) fn build_omega_m_diag<T, R>(
    m_diag: &Array1<T>,
    p: &PT2Projection<R>,
) -> Array1<R>
where
    T: NOCIScalar,
    R: NOCIScalar + From<T>,
{
    let two_e0 = R::from_real(2.0 * p.e0);

    Array1::from_iter((0..m_diag.len()).map(|a| {
        <R as From<T>>::from(m_diag[a]) - p.f_a0[a] * p.s_0a[a] - p.s_a0[a] * p.f_0a[a]
            + two_e0 * p.s_a0[a] * p.s_0a[a]
    }))
}

/// Build the diagonal of the projected first-order-space overlap matrix `Q`.
/// # Arguments:
/// - `s_diag`: Diagonal of the unprojected candidate-candidate overlap matrix `S`.
/// - `p`: Projection contractions used to form `Q`.
/// # Returns:
/// - `Array1<R>`: Diagonal entries of `Q`.
pub(in crate::noci::selected) fn build_omega_s_diag<T, R>(
    s_diag: &Array1<T>,
    p: &PT2Projection<R>,
) -> Array1<R>
where
    T: NOCIScalar,
    R: NOCIScalar + From<T>,
{
    Array1::from_iter(
        (0..s_diag.len()).map(|a| <R as From<T>>::from(s_diag[a]) - p.s_a0[a] * p.s_0a[a]),
    )
}

/// Select the highest-scoring candidates above the selection threshold.
/// Symmetry-equivalent determinants have equal scores up to the GMRES tolerance, so a cut at
/// `max_add` that falls within such a group would choose between them by solver noise. The cut is
/// instead extended through every candidate whose score lies within a relative `TIETOL` of the last
/// one taken, so tied candidates are selected together and `max_add` may be exceeded.
/// # Arguments:
/// - `candidates`: Candidate determinants currently present in the pool.
/// - `candidate_scores`: Candidate importance scores.
/// - `sigma`: Selection threshold.
/// - `max_add`: Number of candidates to add before completing the last tied group.
/// # Returns:
/// - `Vec<NOCIIndex>`: Selected candidates sorted by decreasing score.
pub(in crate::noci::selected) fn select_candidates(
    candidates: &[NOCIIndex],
    candidate_scores: &[f64],
    sigma: f64,
    max_add: usize,
) -> Vec<NOCIIndex> {
    let mut ranked: Vec<(NOCIIndex, f64)> = candidates
        .iter()
        .copied()
        .zip(candidate_scores.iter().copied())
        .filter(|(_, score)| *score > sigma)
        .collect();

    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

    // Extend the cut through candidates tied with the last one taken.
    let mut n = max_add.min(ranked.len());
    if n > 0 {
        let last = ranked[n - 1].1;
        while n < ranked.len() && ranked[n].1 >= last * (1.0 - TIETOL) {
            n += 1;
        }
    }

    ranked.into_iter().take(n).map(|(state, _)| state).collect()
}
