// noci/factorise/onebody/gpu/orthogonal.rs
//! GPU same-parent orthogonal one-body NOCI operator contractions.

// Standard library imports.
use std::any::TypeId;
use std::collections::HashMap;

// External crate imports.
use cubecl::prelude::*;

// Crate-root imports.
use crate::gpu::{GpuBuffer, GpuContext, GpuRuntime};
use crate::noci::fock::calculate_f_pair_orthogonal;
use crate::noci::overlap::calculate_s_pair_orthogonal;
use crate::noci::types::{FockData, NOCIData, NOCIScalar};

// Parent/sibling imports.
use super::super::super::SpinFactorisation;

/// Number of GPU threads per orthogonal kernel cube.
const ORTHOGONAL_CUBE_DIM: u32 = 128;

/// One sparse same-parent orthogonal operator block.
pub(crate) struct GpuOrthogonalBlock {
    /// CSR row offsets, one row per actual target determinant.
    row_offsets: GpuBuffer<u32>,
    /// Global source determinant IDs for CSR entries.
    source_det: GpuBuffer<u32>,
    /// Fock matrix elements for CSR entries.
    f: GpuBuffer<f64>,
    /// Overlap matrix elements for CSR entries.
    s: GpuBuffer<f64>,
    /// Global target determinant ID for each CSR row.
    row_det: GpuBuffer<u32>,
    /// Target alpha component used for MPI row ownership.
    row_a: GpuBuffer<u32>,
    /// Diagonal Fock matrix element for each row.
    diag_f: GpuBuffer<f64>,
    /// Diagonal overlap matrix element for each row.
    diag_s: GpuBuffer<f64>,
    /// Number of target rows.
    nrow: usize,
}

/// GPU same-parent orthogonal blocks indexed by parent.
pub(crate) struct GpuOrthogonalBlocks {
    blocks: Vec<Option<GpuOrthogonalBlock>>,
}

#[derive(Clone, Copy)]
struct OrthogonalEdge {
    source_det: usize,
    f: f64,
    s: f64,
}

impl GpuOrthogonalBlocks {
    /// Build sparse same-parent Slater-Condon blocks and upload them.
    pub(crate) fn new<T: NOCIScalar + 'static>(
        context: &GpuContext,
        spin: &SpinFactorisation,
        data: &NOCIData<'_, T>,
        fock: &FockData<'_, T>,
    ) -> Self {
        let mut blocks = Vec::with_capacity(spin.parents.len());

        for parent in 0..spin.parents.len() {
            if fock.fock_mocache[parent].orthogonal_slater_condon {
                blocks.push(Some(build_orthogonal_block(
                    context, spin, data, fock, parent,
                )));
            } else {
                blocks.push(None);
            }
        }

        Self { blocks }
    }

    /// Return the orthogonal block for one parent.
    pub(crate) fn block(
        &self,
        parent: usize,
    ) -> &GpuOrthogonalBlock {
        self.blocks[parent]
            .as_ref()
            .expect("requested GPU orthogonal block is not orthogonal")
    }
}

/// Build one sparse same-parent Slater-Condon matrix.
///
/// This mirrors the CPU orthogonal path, but performs the expensive
/// repeated GMRES applications later on the GPU.
fn build_orthogonal_block<T: NOCIScalar + 'static>(
    context: &GpuContext,
    spin: &SpinFactorisation,
    data: &NOCIData<'_, T>,
    fock: &FockData<'_, T>,
    parent_id: usize,
) -> GpuOrthogonalBlock {
    let parent = &spin.parents[parent_id];
    let cache = &fock.fock_mocache[parent_id];

    // Occupation-pair -> target row IDs.
    let mut groups: HashMap<(u128, u128), Vec<usize>> = HashMap::new();

    for (row, entry) in parent.entries.iter().enumerate() {
        let det = &data.basis[entry.det];
        groups.entry((det.oa, det.ob)).or_default().push(row);
    }

    // Gather sparse matrix elements by target row.
    let mut rows = vec![Vec::<OrthogonalEdge>::new(); parent.entries.len()];

    for source_entry in &parent.entries {
        let source_det = source_entry.det;
        let source = &data.basis[source_det];

        // Same-occupation F + lambda S contribution.
        if let Some(target_rows) = groups.get(&(source.oa, source.ob)) {
            for &row in target_rows {
                let target_det = parent.entries[row].det;
                let target = &data.basis[target_det];

                rows[row].push(OrthogonalEdge {
                    source_det,
                    f: real_scalar(calculate_f_pair_orthogonal(cache, target, source)),
                    s: real_scalar(calculate_s_pair_orthogonal(target, source)),
                });
            }
        }

        // Alpha single substitutions.
        let mut holes = source.oa;
        while holes != 0 {
            let hole = holes.trailing_zeros() as usize;
            holes &= holes - 1;

            for part in 0..cache.fa.nrows() {
                if ((source.oa >> part) & 1) != 0 {
                    continue;
                }

                let target_oa = (source.oa & !(1u128 << hole)) | (1u128 << part);

                let Some(target_rows) = groups.get(&(target_oa, source.ob)) else {
                    continue;
                };

                for &row in target_rows {
                    let target_det = parent.entries[row].det;
                    let target = &data.basis[target_det];

                    rows[row].push(OrthogonalEdge {
                        source_det,
                        f: real_scalar(calculate_f_pair_orthogonal(cache, target, source)),
                        s: 0.0,
                    });
                }
            }
        }

        // Beta single substitutions.
        let mut holes = source.ob;
        while holes != 0 {
            let hole = holes.trailing_zeros() as usize;
            holes &= holes - 1;

            for part in 0..cache.fb.nrows() {
                if ((source.ob >> part) & 1) != 0 {
                    continue;
                }

                let target_ob = (source.ob & !(1u128 << hole)) | (1u128 << part);

                let Some(target_rows) = groups.get(&(source.oa, target_ob)) else {
                    continue;
                };

                for &row in target_rows {
                    let target_det = parent.entries[row].det;
                    let target = &data.basis[target_det];

                    rows[row].push(OrthogonalEdge {
                        source_det,
                        f: real_scalar(calculate_f_pair_orthogonal(cache, target, source)),
                        s: 0.0,
                    });
                }
            }
        }
    }

    let mut row_offsets = Vec::with_capacity(rows.len() + 1);
    let mut source_det = Vec::new();
    let mut f_values = Vec::new();
    let mut s_values = Vec::new();

    row_offsets.push(0u32);

    for row in &rows {
        for edge in row {
            source_det.push(checked_u32(edge.source_det));
            f_values.push(edge.f);
            s_values.push(edge.s);
        }
        row_offsets.push(checked_u32(source_det.len()));
    }

    let mut row_det = Vec::with_capacity(parent.entries.len());
    let mut row_a = Vec::with_capacity(parent.entries.len());
    let mut diag_f = Vec::with_capacity(parent.entries.len());
    let mut diag_s = Vec::with_capacity(parent.entries.len());

    for entry in &parent.entries {
        let det = &data.basis[entry.det];

        row_det.push(checked_u32(entry.det));
        row_a.push(checked_u32(entry.a));

        diag_f.push(real_scalar(calculate_f_pair_orthogonal(cache, det, det)));
        diag_s.push(real_scalar(calculate_s_pair_orthogonal(det, det)));
    }

    GpuOrthogonalBlock {
        row_offsets: GpuBuffer::from_slice(context, &row_offsets),
        source_det: GpuBuffer::from_slice(context, &source_det),
        f: GpuBuffer::from_slice(context, &f_values),
        s: GpuBuffer::from_slice(context, &s_values),
        row_det: GpuBuffer::from_slice(context, &row_det),
        row_a: GpuBuffer::from_slice(context, &row_a),
        diag_f: GpuBuffer::from_slice(context, &diag_f),
        diag_s: GpuBuffer::from_slice(context, &diag_s),
        nrow: parent.entries.len(),
    }
}

/// Apply one sparse same-parent orthogonal block.
pub(crate) fn launch_apply_orthogonal_block(
    context: &GpuContext,
    block: &GpuOrthogonalBlock,
    x: &GpuBuffer<f64>,
    y: &GpuBuffer<f64>,
    lambda: f64,
    partition: (usize, usize),
) {
    if block.nrow == 0 {
        return;
    }

    let (worker, nworker) = partition;

    unsafe {
        apply_orthogonal_block_kernel::launch_unchecked::<GpuRuntime>(
            context.client(),
            CubeCount::Static(launch_cubes(block.nrow), 1, 1),
            CubeDim::new_1d(ORTHOGONAL_CUBE_DIM),
            block.row_offsets.array_arg(),
            block.source_det.array_arg(),
            block.f.array_arg(),
            block.s.array_arg(),
            block.row_det.array_arg(),
            block.row_a.array_arg(),
            x.array_arg(),
            y.array_arg(),
            block.nrow,
            lambda,
            worker,
            nworker,
        );
    }
}

/// Fill diagonals for one sparse same-parent orthogonal block.
pub(crate) fn launch_fill_orthogonal_diagonal(
    context: &GpuContext,
    block: &GpuOrthogonalBlock,
    m_diag: &GpuBuffer<f64>,
    s_diag: &GpuBuffer<f64>,
    lambda: f64,
) {
    if block.nrow == 0 {
        return;
    }

    unsafe {
        fill_orthogonal_diagonal_kernel::launch_unchecked::<GpuRuntime>(
            context.client(),
            CubeCount::Static(launch_cubes(block.nrow), 1, 1),
            CubeDim::new_1d(ORTHOGONAL_CUBE_DIM),
            block.row_det.array_arg(),
            block.diag_f.array_arg(),
            block.diag_s.array_arg(),
            m_diag.array_arg(),
            s_diag.array_arg(),
            block.nrow,
            lambda,
        );
    }
}

/// Sparse same-parent GPU matrix-vector action.
#[cube(launch_unchecked)]
fn apply_orthogonal_block_kernel(
    row_offsets: &Array<u32>,
    source_det: &Array<u32>,
    f: &Array<f64>,
    s: &Array<f64>,
    row_det: &Array<u32>,
    row_a: &Array<u32>,
    x: &Array<f64>,
    y: &mut Array<f64>,
    nrow: usize,
    lambda: f64,
    worker: usize,
    nworker: usize,
) {
    if ABSOLUTE_POS < nrow {
        let row = ABSOLUTE_POS;
        let owner = usize::cast_from(row_a[row]);

        if owner % nworker == worker {
            let start = usize::cast_from(row_offsets[row]);
            let end = usize::cast_from(row_offsets[row + 1usize]);

            let mut value = 0.0;

            for p in start..end {
                let source = usize::cast_from(source_det[p]);
                value += (f[p] + lambda * s[p]) * x[source];
            }

            let target = usize::cast_from(row_det[row]);
            y[target] = y[target] + value;
        }
    }
}

/// Same-parent orthogonal diagonal kernel.
#[cube(launch_unchecked)]
fn fill_orthogonal_diagonal_kernel(
    row_det: &Array<u32>,
    diag_f: &Array<f64>,
    diag_s: &Array<f64>,
    m_diag: &mut Array<f64>,
    s_diag: &mut Array<f64>,
    nrow: usize,
    lambda: f64,
) {
    if ABSOLUTE_POS < nrow {
        let row = ABSOLUTE_POS;
        let det = usize::cast_from(row_det[row]);
        let s = diag_s[row];

        s_diag[det] = s;
        m_diag[det] = diag_f[row] + lambda * s;
    }
}

fn launch_cubes(len: usize) -> u32 {
    checked_u32(len.div_ceil(ORTHOGONAL_CUBE_DIM as usize))
}

fn checked_u32(value: usize) -> u32 {
    u32::try_from(value).expect("GPU orthogonal index exceeds u32")
}

fn real_scalar<T: NOCIScalar + 'static>(value: T) -> f64 {
    if TypeId::of::<T>() != TypeId::of::<f64>() {
        eprintln!("GPU orthogonal one-body path currently requires f64");
        std::process::exit(1);
    }

    let ptr = &value as *const T as *const f64;

    // SAFETY: the type check above proves T = f64.
    unsafe { *ptr }
}
