// maths/contract.rs
//! Pairwise contraction of dense, strided, labelled tensors.
//!
//! A tensor is addressed by one stride per distinct index label, so permuted and diagonal
//! views need no copies. Two tensors are contracted over every label that is not kept,
//! `C_{\text{keep}} = \sum_{\text{summed}} A B`: large contractions whose summed labels occur in
//! both operands run as batched matrix products, and the rest as register-accumulated dot
//! products. Shapes are fixed-size and label sets are 64-bit masks, so contracting many small
//! tensors does not touch the heap beyond the result buffer.

// Standard library imports.
use std::borrow::Cow;

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

/// Largest number of operands whose contraction order is optimised exhaustively.
const MAXOPTIMAL: usize = 14;

/// Joint index-space sizes of label masks, `\prod_{l \in m} n_l`, from one product table per
/// byte of the mask.
pub struct LabelSizes {
    /// Product of the extents of the labels set in every byte value, for each byte position.
    bytes: Vec<[f64; 256]>,
}

impl LabelSizes {
    /// Build the byte product tables of every label extent.
    /// # Arguments:
    /// - `extent`: Extent of every label, indexed by label; at most 64 labels.
    /// # Returns:
    /// - `Self`: Size tables.
    pub fn new(extent: &[usize]) -> Self {
        let bytes = (0..extent.len().div_ceil(8))
            .map(|b| {
                let mut table = [1.0; 256];
                for (value, product) in table.iter_mut().enumerate() {
                    for bit in 0..8 {
                        let label = 8 * b + bit;
                        if value & (1 << bit) != 0 && label < extent.len() {
                            *product *= extent[label] as f64;
                        }
                    }
                }
                table
            })
            .collect();
        Self { bytes }
    }

    /// Return the joint index-space size of a label mask.
    /// # Arguments:
    /// - `mask`: Bit mask of labels.
    /// # Returns:
    /// - `f64`: Product of the label extents.
    pub fn size(
        &self,
        mask: u64,
    ) -> f64 {
        self.bytes
            .iter()
            .enumerate()
            .map(|(b, table)| table[((mask >> (8 * b)) & 0xff) as usize])
            .product()
    }
}

/// Find the pairwise contraction order of a product of operands that minimises the total
/// number of multiply-adds, `\sum_{\text{steps}} \prod_{l \in A \cup B} n_l`, by dynamic
/// programming over operand subsets. The intermediate of a subset keeps the labels it shares
/// with its complement or with the final result. Products of more than `MAXOPTIMAL` operands
/// fall back to repeatedly contracting the pair with the smallest joint index space.
/// # Arguments:
/// - `masks`: Label mask of every operand.
/// - `kept`: Bit mask of labels kept in the final result.
/// - `sizes`: Joint index-space sizes of label masks.
/// - `steps`: Receives the steps in execution order as operand numbers; operands are numbered
///   by position and each step's result takes the next number.
/// # Returns:
/// - `()`: Appends to `steps`.
pub fn optimal_contraction_steps(
    masks: &[u64],
    kept: u64,
    sizes: &LabelSizes,
    steps: &mut Vec<(u8, u8)>,
) {
    let n = masks.len();
    if n < 2 {
        return;
    }
    if n > MAXOPTIMAL {
        greedy_contraction_steps(masks, kept, sizes, steps);
        return;
    }

    // Labels of every subset of operands.
    let full = (1usize << n) - 1;
    let mut labels = vec![0u64; full + 1];
    for set in 1..=full {
        labels[set] = labels[set & (set - 1)] | masks[set.trailing_zeros() as usize];
    }
    let external = |set: usize| labels[set] & (labels[full ^ set] | kept);

    // Cheapest cost and split of every subset; each split puts the lowest operand in `part`
    // so every unordered split is tried once.
    let mut cost = vec![0.0f64; full + 1];
    let mut split = vec![0usize; full + 1];
    for set in 1..=full {
        if set & (set - 1) == 0 {
            continue;
        }
        let low = set & set.wrapping_neg();
        let mut best = f64::INFINITY;
        let mut part = (set - 1) & set;
        while part != 0 {
            if part & low != 0 {
                let rest = set ^ part;
                let c = cost[part] + cost[rest] + sizes.size(external(part) | external(rest));
                if c < best {
                    best = c;
                    split[set] = part;
                }
            }
            part = (part - 1) & set;
        }
        cost[set] = best;
    }

    let mut next = n as u8;
    emit_contraction_steps(full, &split, &mut next, steps);
}

/// Emit the steps of an optimal contraction tree by post-order traversal.
/// # Arguments:
/// - `set`: Operand subset of the current subtree.
/// - `split`: Optimal split of every subset.
/// - `next`: Number of the next intermediate.
/// - `steps`: Receives the steps.
/// # Returns:
/// - `u8`: Number of the operand holding the subtree.
fn emit_contraction_steps(
    set: usize,
    split: &[usize],
    next: &mut u8,
    steps: &mut Vec<(u8, u8)>,
) -> u8 {
    if set & (set - 1) == 0 {
        return set.trailing_zeros() as u8;
    }
    let a = emit_contraction_steps(split[set], split, next, steps);
    let b = emit_contraction_steps(set ^ split[set], split, next, steps);
    steps.push((a, b));
    *next += 1;
    *next - 1
}

/// Order a pairwise contraction by repeatedly contracting the pair with the smallest joint index
/// space.
/// # Arguments:
/// - `masks`: Label mask of every operand.
/// - `kept`: Bit mask of labels kept in the final result.
/// - `sizes`: Joint index-space sizes of label masks.
/// - `steps`: Receives the steps, numbered as in `optimal_contraction_steps`.
/// # Returns:
/// - `()`: Appends to `steps`.
fn greedy_contraction_steps(
    masks: &[u64],
    kept: u64,
    sizes: &LabelSizes,
    steps: &mut Vec<(u8, u8)>,
) {
    let mut live = masks
        .iter()
        .enumerate()
        .map(|(k, &m)| (k as u8, m))
        .collect::<Vec<_>>();
    let mut next = masks.len() as u8;
    while live.len() > 1 {
        let mut best = (0, 1);
        let mut cost = f64::INFINITY;
        for i in 0..live.len() {
            for j in i + 1..live.len() {
                let c = sizes.size(live[i].1 | live[j].1);
                if c < cost {
                    cost = c;
                    best = (i, j);
                }
            }
        }
        let (b, mb) = live.swap_remove(best.1);
        let (a, ma) = live.swap_remove(best.0);
        let rest = live.iter().fold(kept, |m, &(_, x)| m | x);
        steps.push((a, b));
        live.push((next, (ma | mb) & rest));
        next += 1;
    }
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

    // Order every group by decreasing stride in its operand, so operands already laid out as
    // `[b][i][k]` or `[b][k][i]` (and `[b][k][j]` or `[b][j][k]`) are used in place.
    batch.sort_by_key(|&(k, _)| std::cmp::Reverse(sa[k]));
    left.sort_by_key(|&(k, _)| std::cmp::Reverse(sa[k]));
    summed.sort_by_key(|&(k, _)| std::cmp::Reverse(sa[k]));
    right.sort_by_key(|&(k, _)| std::cmp::Reverse(sb[k]));

    let layout = |groups: &[&[(usize, usize)]], strides: &[usize]| {
        groups
            .iter()
            .flat_map(|g| g.iter().map(|&(k, d)| (d, strides[k])))
            .collect::<Vec<_>>()
    };

    // Use the transposed matrix of an operand when only that order is contiguous.
    let a_direct = layout(&[&batch, &left, &summed], sa);
    let a_swapped = layout(&[&batch, &summed, &left], sa);
    let a_transposed = !is_contiguous(&a_direct) && is_contiguous(&a_swapped);
    let pa = pack_strided(da, if a_transposed { &a_swapped } else { &a_direct });
    let b_direct = layout(&[&batch, &summed, &right], sb);
    let b_swapped = layout(&[&batch, &right, &summed], sb);
    let b_transposed = !is_contiguous(&b_direct) && is_contiguous(&b_swapped);
    let pb = pack_strided(db, if b_transposed { &b_swapped } else { &b_direct });

    // One GEMM per batch element into the row-major result.
    buffer.clear();
    buffer.resize(nb * m * nn, 0.0);
    for (i, c) in buffer.chunks_mut(m * nn).enumerate() {
        let sa_i = &pa[i * m * kk..(i + 1) * m * kk];
        let sb_i = &pb[i * kk * nn..(i + 1) * kk * nn];
        let av = if a_transposed {
            ArrayView2::from_shape((kk, m), sa_i).ok()?.reversed_axes()
        } else {
            ArrayView2::from_shape((m, kk), sa_i).ok()?
        };
        let bv = if b_transposed {
            ArrayView2::from_shape((nn, kk), sb_i).ok()?.reversed_axes()
        } else {
            ArrayView2::from_shape((kk, nn), sb_i).ok()?
        };
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

/// Return whether a strided layout is one contiguous row-major run from the start of its data.
/// # Arguments:
/// - `layout`: Extent and source stride of every axis, outermost first.
/// # Returns:
/// - `bool`: Whether the layout addresses `0..\prod_k d_k` in order.
fn is_contiguous(layout: &[(usize, usize)]) -> bool {
    let mut expected = 1;
    for &(d, st) in layout.iter().rev() {
        if d != 1 && st != expected {
            return false;
        }
        expected *= d;
    }
    true
}

/// Gather a strided tensor into a contiguous row-major array, borrowing the source when it is
/// already contiguous in the requested order. Axes that are contiguous with their inner
/// neighbour are merged first, and the innermost axis is copied as one run.
/// # Arguments:
/// - `data`: Source data.
/// - `layout`: Extent and source stride of every output axis, outermost first.
/// # Returns:
/// - `Cow<[f64]>`: Row-major elements, borrowed from `data` when no copy is needed.
fn pack_strided<'a>(
    data: &'a [f64],
    layout: &[(usize, usize)],
) -> Cow<'a, [f64]> {
    // Merge each axis into its inner neighbour when `s_{\text{outer}} = d_{\text{inner}}
    // s_{\text{inner}}`, dropping unit axes.
    let mut axes = [(1usize, 0usize); 2 * MAXLABELS];
    let mut n = 0;
    for &(d, st) in layout.iter().rev() {
        if d == 1 {
            continue;
        }
        if n > 0 && axes[n - 1].0 * axes[n - 1].1 == st {
            axes[n - 1].0 *= d;
        } else {
            axes[n] = (d, st);
            n += 1;
        }
    }
    let axes = &mut axes[..n];
    axes.reverse();
    let total = axes.iter().map(|&(d, _)| d).product::<usize>();

    // A single unit-stride run is the source itself.
    if n == 0 {
        return Cow::Borrowed(&data[..1]);
    }
    if n == 1 && axes[0].1 == 1 {
        return Cow::Borrowed(&data[..total]);
    }

    let (inner, outer) = axes.split_last().unwrap();
    let (d, st) = *inner;
    let count = total / d;
    let mut out = Vec::with_capacity(total);
    let mut idx = [0usize; 2 * MAXLABELS];
    let mut offset = 0;
    for _ in 0..count {
        if st == 1 {
            out.extend_from_slice(&data[offset..offset + d]);
        } else {
            out.extend((0..d).map(|i| data[offset + i * st]));
        }

        let mut k = outer.len();
        while k > 0 {
            k -= 1;
            idx[k] += 1;
            offset += outer[k].1;
            if idx[k] < outer[k].0 {
                break;
            }
            offset -= outer[k].1 * outer[k].0;
            idx[k] = 0;
        }
    }

    Cow::Owned(out)
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
