// maths/contract.rs
//! Pairwise contraction of dense, strided, labelled tensors.
//!
//! A tensor is addressed by one stride per distinct index label, so permuted and diagonal
//! views need no copies. Two tensors are contracted over every label that is not kept,
//! `C_{\text{keep}} = \sum_{\text{summed}} A B`: large contractions whose summed labels occur in
//! both operands run as batched matrix products, and the rest as register-accumulated dot
//! products. Shapes are fixed-size and label sets are 64-bit masks, so contracting many small
//! tensors does not touch the heap beyond the result buffer.

// External crate imports.
use ndarray::linalg::general_mat_mul;
use ndarray::{ArrayView2, ArrayViewMut2};

/// Smallest `m n k` for which a pairwise contraction runs as matrix products.
const MATMULMIN: usize = 1 << 12;

/// Largest number of distinct labels on one operand.
pub const MAXLABELS: usize = 16;

/// Index labels, extents and strides of one operand.
#[derive(Clone, Copy)]
pub struct TensorShape {
    /// Number of distinct labels.
    pub n: usize,
    /// Distinct labels.
    pub labels: [u16; MAXLABELS],
    /// Extent of every label.
    pub dims: [usize; MAXLABELS],
    /// Stride of every label; a label repeated within one factor sums its slot strides.
    pub strides: [usize; MAXLABELS],
    /// Bit mask of the labels.
    pub mask: u64,
}

/// Build the shape of one dense block, merging labels repeated within the factor.
/// # Arguments:
/// - `slots`: Label of every block slot.
/// - `extent`: Extent of every label, indexed by label.
/// # Returns:
/// - `TensorShape`: Distinct labels with extents and strides.
pub fn strided_tensor_shape(
    slots: &[u16],
    extent: &[usize],
) -> TensorShape {
    let mut shape = TensorShape {
        n: 0,
        labels: [0; MAXLABELS],
        dims: [0; MAXLABELS],
        strides: [0; MAXLABELS],
        mask: 0,
    };

    // Row-major slot strides, walking from the last slot.
    let mut stride = 1;
    for &l in slots.iter().rev() {
        let d = extent[l as usize];
        match shape.labels[..shape.n].iter().position(|&x| x == l) {
            Some(k) => shape.strides[k] += stride,
            None => {
                shape.labels[shape.n] = l;
                shape.dims[shape.n] = d;
                shape.strides[shape.n] = stride;
                shape.mask |= 1 << l;
                shape.n += 1;
            }
        }
        stride *= d;
    }

    shape
}

/// Choose the operand pair whose joint index space is smallest.
/// # Arguments:
/// - `operands`: Remaining operands, at least two.
/// - `extent`: Extent of every label, indexed by label.
/// # Returns:
/// - `(usize, usize)`: Positions of the chosen pair.
pub fn cheapest_contraction_pair<T>(
    operands: &[(T, TensorShape)],
    extent: &[usize],
) -> (usize, usize) {
    let size = |mask: u64| {
        let mut m = mask;
        let mut p = 1usize;
        while m != 0 {
            p *= extent[m.trailing_zeros() as usize];
            m &= m - 1;
        }
        p
    };

    let mut best = (0, 1);
    let mut cost = usize::MAX;
    for i in 0..operands.len() {
        for j in i + 1..operands.len() {
            let joint = size(operands[i].1.mask | operands[j].1.mask);
            if joint < cost {
                cost = joint;
                best = (i, j);
            }
        }
    }

    best
}

/// Contract two operands, summing every label not kept for the output or later operands:
/// `C_{\text{keep}} = \sum_{\text{summed}} A B`.
/// # Arguments:
/// - `a`: First operand data and shape.
/// - `b`: Second operand data and shape.
/// - `keep`: Bit mask of labels that survive the contraction.
/// - `buffer`: Reusable storage for the result.
/// # Returns:
/// - `(Vec<f64>, TensorShape)`: Row-major result and its shape.
pub fn contract_tensor_pair(
    a: (&[f64], &TensorShape),
    b: (&[f64], &TensorShape),
    keep: u64,
    mut buffer: Vec<f64>,
) -> (Vec<f64>, TensorShape) {
    let ((da, a), (db, b)) = (a, b);

    // Joint labels with their strides in both operands.
    let mut n = 0;
    let mut labels = [0u16; 2 * MAXLABELS];
    let mut dims = [0usize; 2 * MAXLABELS];
    let mut sa = [0usize; 2 * MAXLABELS];
    let mut sb = [0usize; 2 * MAXLABELS];
    for k in 0..a.n {
        labels[n] = a.labels[k];
        dims[n] = a.dims[k];
        sa[n] = a.strides[k];
        n += 1;
    }
    for k in 0..b.n {
        match labels[..n].iter().position(|&x| x == b.labels[k]) {
            Some(p) => sb[p] = b.strides[k],
            None => {
                labels[n] = b.labels[k];
                dims[n] = b.dims[k];
                sb[n] = b.strides[k];
                n += 1;
            }
        }
    }

    // Large contractions whose summed labels occur in both operands run as matrix products.
    let joint = (&labels[..n], &dims[..n], &sa[..n], &sb[..n]);
    if let Some(shape) = contract_by_matrix_product((da, a), (db, b), joint, keep, &mut buffer) {
        return (buffer, shape);
    }

    // Kept labels form the row-major result.
    let mut shape = TensorShape {
        n: 0,
        labels: [0; MAXLABELS],
        dims: [0; MAXLABELS],
        strides: [0; MAXLABELS],
        mask: 0,
    };
    let mut kept = [(0usize, 0usize, 0usize); 2 * MAXLABELS];
    let mut summed = [(0usize, 0usize, 0usize); 2 * MAXLABELS];
    let mut ns = 0;
    for k in 0..n {
        if keep & (1 << labels[k]) != 0 {
            shape.labels[shape.n] = labels[k];
            shape.dims[shape.n] = dims[k];
            shape.mask |= 1 << labels[k];
            kept[shape.n] = (dims[k], sa[k], sb[k]);
            shape.n += 1;
        } else {
            summed[ns] = (dims[k], sa[k], sb[k]);
            ns += 1;
        }
    }
    let mut stride = 1;
    for k in (0..shape.n).rev() {
        shape.strides[k] = stride;
        stride *= shape.dims[k];
    }

    // The summed label with the smallest strides runs innermost.
    let summed = &mut summed[..ns];
    summed.sort_unstable_by_key(|&(_, a, b)| std::cmp::Reverse(a + b));

    // Every result element is one dot product over the summed labels, written once in
    // row-major order of the kept labels.
    buffer.clear();
    buffer.resize(stride, 0.0);
    let nk = shape.n;
    let mut idx = [0usize; 2 * MAXLABELS];
    let (mut oa, mut ob) = (0usize, 0usize);
    for value in buffer.iter_mut() {
        *value = summed_dot_product(da, db, oa, ob, summed);

        let mut k = nk;
        while k > 0 {
            k -= 1;
            idx[k] += 1;
            oa += kept[k].1;
            ob += kept[k].2;
            if idx[k] < kept[k].0 {
                break;
            }
            oa -= kept[k].1 * kept[k].0;
            ob -= kept[k].2 * kept[k].0;
            idx[k] = 0;
        }
    }

    (buffer, shape)
}

/// Contract two operands as a batched matrix product,
/// `C_{b,ij} = \sum_k A_{b,ik} B_{b,kj}`, where `b` runs over labels kept from both operands,
/// `i` and `j` over labels kept from only `A` or only `B`, and `k` over the summed labels.
/// Both operands are packed contiguously in that order and each batch is one GEMM call.
/// # Arguments:
/// - `a`: First operand data and shape.
/// - `b`: Second operand data and shape.
/// - `joint`: Joint labels with their extents and strides in `A` and `B`.
/// - `keep`: Bit mask of labels that survive the contraction.
/// - `buffer`: Reusable storage, filled with the row-major result over batch, `i` then `j`.
/// # Returns:
/// - `Option<TensorShape>`: Result shape, or `None` without touching `buffer` when a summed label
///   occurs in only one operand or the product is too small to benefit.
fn contract_by_matrix_product(
    a: (&[f64], &TensorShape),
    b: (&[f64], &TensorShape),
    joint: (&[u16], &[usize], &[usize], &[usize]),
    keep: u64,
    buffer: &mut Vec<f64>,
) -> Option<TensorShape> {
    let ((da, a), (db, b)) = (a, b);
    let (labels, dims, sa, sb) = joint;

    // Group the joint labels as batch, `A`-only kept, `B`-only kept and summed.
    let mut batch = Vec::new();
    let mut left = Vec::new();
    let mut right = Vec::new();
    let mut summed = Vec::new();
    for k in 0..labels.len() {
        let bit = 1u64 << labels[k];
        let (in_a, in_b) = (a.mask & bit != 0, b.mask & bit != 0);
        let entry = (k, dims[k]);
        match (keep & bit != 0, in_a, in_b) {
            (true, true, true) => batch.push(entry),
            (true, true, false) => left.push(entry),
            (true, false, true) => right.push(entry),
            (false, true, true) => summed.push(entry),
            _ => return None,
        }
    }
    let size = |g: &[(usize, usize)]| g.iter().map(|&(_, d)| d).product::<usize>();
    let (nb, m, nn, kk) = (size(&batch), size(&left), size(&right), size(&summed));
    if m * nn * kk < MATMULMIN || summed.is_empty() {
        return None;
    }

    // Pack `A` as `[b][i][k]` and `B` as `[b][k][j]`.
    let layout = |groups: &[&[(usize, usize)]], strides: &[usize]| {
        groups
            .iter()
            .flat_map(|g| g.iter().map(|&(k, d)| (d, strides[k])))
            .collect::<Vec<_>>()
    };
    let pa = pack_strided(da, &layout(&[&batch, &left, &summed], sa));
    let pb = pack_strided(db, &layout(&[&batch, &summed, &right], sb));

    // One GEMM per batch element into the row-major result.
    buffer.clear();
    buffer.resize(nb * m * nn, 0.0);
    for (i, c) in buffer.chunks_mut(m * nn).enumerate() {
        let av = ArrayView2::from_shape((m, kk), &pa[i * m * kk..(i + 1) * m * kk]).ok()?;
        let bv = ArrayView2::from_shape((kk, nn), &pb[i * kk * nn..(i + 1) * kk * nn]).ok()?;
        let mut cv = ArrayViewMut2::from_shape((m, nn), c).ok()?;
        general_mat_mul(1.0, &av, &bv, 0.0, &mut cv);
    }

    // Result labels in batch, `i`, `j` order with row-major strides.
    let mut shape = TensorShape {
        n: 0,
        labels: [0; MAXLABELS],
        dims: [0; MAXLABELS],
        strides: [0; MAXLABELS],
        mask: 0,
    };
    for &(k, d) in batch.iter().chain(&left).chain(&right) {
        shape.labels[shape.n] = labels[k];
        shape.dims[shape.n] = d;
        shape.mask |= 1 << labels[k];
        shape.n += 1;
    }
    let mut stride = 1;
    for k in (0..shape.n).rev() {
        shape.strides[k] = stride;
        stride *= shape.dims[k];
    }

    Some(shape)
}

/// Gather a strided tensor into a contiguous row-major array.
/// # Arguments:
/// - `data`: Source data.
/// - `layout`: Extent and source stride of every output axis, outermost first.
/// # Returns:
/// - `Vec<f64>`: Row-major gathered elements.
fn pack_strided(
    data: &[f64],
    layout: &[(usize, usize)],
) -> Vec<f64> {
    let total = layout.iter().map(|&(d, _)| d).product::<usize>();
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; layout.len()];
    let mut offset = 0;

    for _ in 0..total {
        out.push(data[offset]);

        let mut k = layout.len();
        while k > 0 {
            k -= 1;
            idx[k] += 1;
            offset += layout[k].1;
            if idx[k] < layout[k].0 {
                break;
            }
            offset -= layout[k].1 * layout[k].0;
            idx[k] = 0;
        }
    }

    out
}

/// Sum `A B` over the summed labels for one result element,
/// `\sum_{\text{summed}} A_{a_0 + \cdots} B_{b_0 + \cdots}`.
/// The last summed label runs as a plain strided loop; the others advance by odometer.
/// # Arguments:
/// - `da`: First operand data.
/// - `db`: Second operand data.
/// - `a0`: Offset of the result element in the first operand.
/// - `b0`: Offset of the result element in the second operand.
/// - `summed`: Extent and strides in both operands of every summed label.
/// # Returns:
/// - `f64`: Summed product.
fn summed_dot_product(
    da: &[f64],
    db: &[f64],
    a0: usize,
    b0: usize,
    summed: &[(usize, usize, usize)],
) -> f64 {
    let Some((&(d, sa, sb), outer)) = summed.split_last() else {
        return da[a0] * db[b0];
    };

    let count = outer.iter().map(|x| x.0).product::<usize>();
    let mut idx = [0usize; 2 * MAXLABELS];
    let (mut oa, mut ob) = (a0, b0);
    let mut total = 0.0;
    for _ in 0..count {
        let mut acc = 0.0;
        for i in 0..d {
            acc += da[oa + i * sa] * db[ob + i * sb];
        }
        total += acc;

        let mut k = outer.len();
        while k > 0 {
            k -= 1;
            idx[k] += 1;
            oa += outer[k].1;
            ob += outer[k].2;
            if idx[k] < outer[k].0 {
                break;
            }
            oa -= outer[k].1 * outer[k].0;
            ob -= outer[k].2 * outer[k].0;
            idx[k] = 0;
        }
    }

    total
}
