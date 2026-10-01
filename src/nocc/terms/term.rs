// nocc/terms/term.rs
//! Contraction of one planned term and its accumulation into an output block.

// Standard library imports.
use std::ops::{AddAssign, Mul};

// External crate imports.
use num_complex::Complex64;

// Crate-root imports.
use crate::maths::contract::{MAXLABELS, TensorShape, contract_tensor_pair, strided_tensor_shape};
use crate::maths::gemm::{
    strided_gemm, strided_gemm_complex, strided_gemm_complex_real, strided_gemm_real_complex,
};

// Parent/sibling imports.
use super::plan::Anchor;
use super::schema::TensorFactor;
use super::workspace::{Out, Source, Values, View, Workspace, pooled_buffer};

/// Largest intermediate, in elements, of a term contracted whole; larger ones are contracted in
/// slices over free indices so every intermediate stays in the cache.
const SLICEDTERM: usize = 1 << 16;

/// Data of every table-local block, and the group's shared product with its shape and use.
type TermSources<'a> = (
    &'a [View<'a>],
    Option<(View<'a>, &'a TensorShape, &'a Anchor)>,
);

/// One kept term with its resolved plan.
pub(super) struct PlannedTerm<'a> {
    /// Tensor factors of the term.
    pub(super) factors: &'a [TensorFactor],
    /// Table-local block id of every tensor factor.
    pub(super) blocks: &'a [u32],
    /// Pairwise contraction steps, as operand numbers.
    pub(super) steps: &'a [(u8, u8)],
    /// Representative of every label after resolving the deltas.
    pub(super) map: [u16; 64],
    /// Coefficient of the term.
    pub(super) coefficient: f64,
}

/// Contract one term, with some free indices fixed, and add it to an output block over the
/// remaining free indices. A term whose largest intermediate exceeds `SLICEDTERM` is contracted
/// once per value of further free indices, with the steps independent of them contracted once.
/// # Arguments:
/// - `term`: Kept term with its plan.
/// - `tensors`: Data of every table-local block and extent of every class-local label.
/// - `indices`: Class-local ids of the free indices of `out` in output order, and the free
///   indices fixed to one value with their values.
/// - `shared`: Data and canonical shape of the term's shared product with its use, which
///   replaces the anchored step, or `None`.
/// - `out`: Row-major output block over the unfixed free indices, updated in place.
/// - `ws`: Worker storage.
/// # Returns:
/// - `()`: Mutates `out` and `ws`.
pub(super) fn accumulate_term(
    term: &PlannedTerm<'_>,
    tensors: (&[View<'_>], &[usize]),
    indices: (&[u16], &[(u16, usize)]),
    shared: Option<(View<'_>, &TensorShape, &Anchor)>,
    out: &mut Out<'_>,
    ws: &mut Workspace,
) {
    let (data, extent) = tensors;
    let (free, fixed) = indices;
    let map = &term.map;

    // Fixed indices by representative; a representative fixed to two values vanishes here.
    let mut reps = [(0u16, 0usize); MAXLABELS];
    let mut nf = 0;
    for &(l, v) in fixed {
        let r = map[l as usize];
        match reps[..nf].iter().find(|x| x.0 == r) {
            Some(&(_, w)) if w != v => return,
            Some(_) => {}
            None => {
                reps[nf] = (r, v);
                nf += 1;
            }
        }
    }
    let fixed = &reps[..nf];
    let fixed_mask = fixed.iter().fold(0u64, |m, x| m | (1 << x.0));
    let free_mask = free.iter().fold(0u64, |m, &x| m | (1 << map[x as usize])) & !fixed_mask;

    // Every tensor factor becomes an operand over its block, with its labels substituted and
    // its fixed labels folded into an offset; a label repeated within one factor becomes a
    // diagonal view.
    ws.operands.clear();
    for (f, &id) in term.factors.iter().zip(term.blocks) {
        let mut labels = [0u16; 2 * MAXLABELS];
        let n = f.1.len() + f.2.len();
        for (slot, &l) in labels.iter_mut().zip(f.1.iter().chain(&f.2)) {
            *slot = map[l as usize];
        }
        let mut shape = strided_tensor_shape(&labels[..n], extent);
        let offset = fixed
            .iter()
            .map(|&(r, v)| fix_label(&mut shape, r, v))
            .sum();
        ws.operands
            .push((Source::Block(id as usize, offset), shape));
    }

    // Labels of every operand and step result, and the labels each step keeps: those of the
    // operands still to be contracted and the free labels.
    let inputs = ws.operands.len();
    ws.masks.clear();
    ws.masks
        .extend(ws.operands.iter().map(|(_, shape)| shape.mask));
    ws.keeps.clear();
    let mut live = (1u64 << inputs) - 1;
    for &(i, j) in term.steps {
        live &= !((1 << i) | (1 << j));
        let keep = (0..ws.masks.len())
            .filter(|&k| live & (1 << k) != 0)
            .fold(free_mask, |m, k| m | ws.masks[k]);
        ws.keeps.push(keep);
        ws.masks
            .push((ws.masks[i as usize] | ws.masks[j as usize]) & keep);
        live |= 1 << (ws.masks.len() - 1);
    }

    // Free labels are never summed, so fixing one slices every intermediate that carries it at
    // no extra arithmetic; fix those that shrink the largest intermediate most until it fits in
    // the cache.
    let size = |mask: u64| {
        (0..64)
            .filter(|&l| mask & (1 << l) != 0)
            .map(|l| extent[l])
            .product::<usize>()
    };
    let peak = |sliced: u64| {
        ws.masks[inputs..]
            .iter()
            .map(|&m| size(m & !sliced))
            .max()
            .unwrap_or(0)
    };
    let mut sliced = 0u64;
    let mut largest = peak(0);
    while largest > SLICEDTERM {
        let best = (0..64)
            .filter(|&l| free_mask & !sliced & (1 << l) != 0)
            .map(|l| (peak(sliced | (1 << l)), l))
            .min();
        match best {
            Some((p, l)) if p < largest => {
                sliced |= 1 << l;
                largest = p;
            }
            _ => break,
        }
    }

    // Operands and steps that depend on a sliced label; the others are contracted once.
    let mut depends = 0u64;
    for (k, &m) in ws.masks[..inputs].iter().enumerate() {
        if m & sliced != 0 {
            depends |= 1 << k;
        }
    }
    for (s, &(i, j)) in term.steps.iter().enumerate() {
        if depends & ((1 << i) | (1 << j)) != 0 {
            depends |= 1 << (inputs + s);
        }
    }
    let placeholder = (Source::Buffer(0), strided_tensor_shape(&[], extent));
    ws.operands.resize(inputs + term.steps.len(), placeholder);
    contract_steps(term, (data, shared), (inputs, &[], !depends), ws);
    if sliced == 0 {
        scatter_last(term, (data, shared), (free, fixed), extent, out, ws);
        ws.release(0);
        return;
    }

    // Sliced labels with their extents, and every sliced operand's view with those labels
    // removed and its offset per unit of each.
    let mut labels = [(0u16, 0usize); MAXLABELS];
    let mut ns = 0;
    for l in (0..64).filter(|&l| sliced & (1 << l) != 0) {
        labels[ns] = (l as u16, extent[l]);
        ns += 1;
    }
    let labels = &labels[..ns];
    ws.views.clear();
    for k in 0..inputs {
        let (source, mut shape) = ws.operands[k];
        let mut steps = [0usize; MAXLABELS];
        if ws.masks[k] & sliced != 0 {
            for (step, &(l, _)) in steps.iter_mut().zip(labels) {
                *step = fix_label(&mut shape, l, 1);
            }
        }
        ws.views.push((source, shape, steps));
    }

    // Every tuple of sliced values contracts the dependent steps and adds its slice.
    let hoisted = ws.buffers.len();
    let mut values = [0usize; MAXLABELS];
    let mut all = [(0u16, 0usize); MAXLABELS];
    all[..fixed.len()].copy_from_slice(fixed);
    loop {
        for k in 0..inputs {
            let (source, shape, steps) = ws.views[k];
            if ws.masks[k] & sliced == 0 {
                continue;
            }
            let shift = values[..ns]
                .iter()
                .zip(&steps)
                .map(|(v, st)| v * st)
                .sum::<usize>();
            let source = match source {
                Source::Block(id, off) => Source::Block(id, off + shift),
                other => other,
            };
            ws.operands[k] = (source, shape);
        }
        for (slot, (&(l, _), &v)) in all[fixed.len()..]
            .iter_mut()
            .zip(labels.iter().zip(&values))
        {
            *slot = (l, v);
        }
        let current = &all[fixed.len()..fixed.len() + ns];
        contract_steps(term, (data, shared), (inputs, current, depends), ws);
        scatter_last(
            term,
            (data, shared),
            (free, &all[..fixed.len() + ns]),
            extent,
            out,
            ws,
        );
        ws.release(hoisted);

        // Advance the sliced values as an odometer.
        let mut k = ns;
        loop {
            if k == 0 {
                ws.release(0);
                return;
            }
            k -= 1;
            values[k] += 1;
            if values[k] < labels[k].1 {
                break;
            }
            values[k] = 0;
        }
    }
}

/// Contract the selected steps of a term in its planned order, each into a new buffer that
/// becomes the step's result operand.
/// # Arguments:
/// - `term`: Kept term with its plan.
/// - `sources`: Data of every table-local block, and the group's shared product with its use.
/// - `selection`: Number of input operands, the sliced labels with their current values, and
///   the operands whose steps run, as a bit mask over operand numbers.
/// - `ws`: Worker storage, with `operands` sized for every step result.
/// # Returns:
/// - `()`: Mutates `ws`.
fn contract_steps(
    term: &PlannedTerm<'_>,
    sources: TermSources<'_>,
    selection: (usize, &[(u16, usize)], u64),
    ws: &mut Workspace,
) {
    let (data, shared) = sources;
    let (inputs, slice, run) = selection;
    let sliced = slice.iter().fold(0u64, |m, &(l, _)| m | (1 << l));
    for (s, &(i, j)) in term.steps.iter().enumerate() {
        let target = inputs + s;
        if run & (1 << target) == 0 {
            continue;
        }

        // The anchored step takes the group's shared product, relabelled to this term, with
        // its sliced labels fixed to their current values.
        if let Some((_, canonical, anchor)) = shared
            && s == anchor.step as usize
        {
            let mut shape = *canonical;
            shape.mask = 0;
            for l in shape.labels[..shape.n].iter_mut() {
                *l = anchor.labels[*l as usize];
                shape.mask |= 1 << *l;
            }
            let offset = slice
                .iter()
                .map(|&(l, v)| fix_label(&mut shape, l, v))
                .sum();
            ws.operands[target] = (Source::Shared(offset), shape);
            continue;
        }
        let (sa, a) = ws.operands[i as usize];
        let (sb, b) = ws.operands[j as usize];
        let keep = ws.keeps[s] & !sliced;
        let len = result_size(&a, &b, keep);
        let (result, shape) = {
            let source = |s: Source| match s {
                Source::Block(id, off) => offset_view(data[id], off),
                Source::Buffer(id) => ws.buffers[id].view(0),
                Source::Shared(off) => shared.map_or(View::Real(&[]), |x| offset_view(x.0, off)),
            };
            let pools = (&mut ws.pool, &mut ws.complex_pool);
            contract_views((source(sa), &a), (source(sb), &b), keep, len, pools)
        };
        ws.buffers.push(result);
        ws.operands[target] = (Source::Buffer(ws.buffers.len() - 1), shape);
    }
}

/// Number of elements of a pairwise contraction's result.
/// # Arguments:
/// - `a`: First operand shape.
/// - `b`: Second operand shape.
/// - `keep`: Labels kept by the contraction.
/// # Returns:
/// - `usize`: Product of the extents of the kept labels of either operand.
fn result_size(
    a: &TensorShape,
    b: &TensorShape,
    keep: u64,
) -> usize {
    let from_a = (0..a.n)
        .filter(|&k| keep & (1 << a.labels[k]) != 0)
        .map(|k| a.dims[k])
        .product::<usize>();
    let from_b = (0..b.n)
        .filter(|&k| keep & !a.mask & (1 << b.labels[k]) != 0)
        .map(|k| b.dims[k])
        .product::<usize>();
    from_a * from_b
}

/// Add the term's final operand, times its coefficient, to the output block.
/// # Arguments:
/// - `term`: Kept term with its plan.
/// - `sources`: Data of every table-local block, and the group's shared product with its use.
/// - `indices`: Class-local ids of the free indices of `out` in output order, and the fixed
///   representatives with their values.
/// - `extent`: Extent of every class-local label.
/// - `out`: Row-major output block, updated in place.
/// - `ws`: Worker storage holding the term's operands.
/// # Returns:
/// - `()`: Mutates `out`.
fn scatter_last(
    term: &PlannedTerm<'_>,
    sources: TermSources<'_>,
    indices: (&[u16], &[(u16, usize)]),
    extent: &[usize],
    out: &mut Out<'_>,
    ws: &Workspace,
) {
    let (data, shared) = sources;
    let unit = [1.0];
    let (values, shape) = ws.operands.last().map_or_else(
        || (View::Real(&unit), strided_tensor_shape(&[], extent)),
        |&(s, shape)| {
            let values = match s {
                Source::Block(id, off) => offset_view(data[id], off),
                Source::Buffer(id) => ws.buffers[id].view(0),
                Source::Shared(off) => shared.map_or(View::Real(&[]), |x| offset_view(x.0, off)),
            };
            (values, shape)
        },
    );
    let (map, coeff) = (&term.map, term.coefficient);
    match (values, out) {
        (View::Real(x), Out::Real(o)) => {
            scatter_into_output((x, shape), indices, map, extent, coeff, o)
        }
        (View::Real(x), Out::Complex(o)) => {
            scatter_into_output((x, shape), indices, map, extent, coeff, o)
        }
        (View::Complex(x), Out::Complex(o)) => {
            scatter_into_output((x, shape), indices, map, extent, coeff, o)
        }
        (View::Complex(_), Out::Real(_)) => panic!("complex term in a real output block"),
    }
}

/// View a block from an element offset.
/// # Arguments:
/// - `view`: Block data.
/// - `offset`: First element of the view.
/// # Returns:
/// - `View<'a>`: Elements from `offset` on.
fn offset_view(
    view: View<'_>,
    offset: usize,
) -> View<'_> {
    match view {
        View::Real(x) => View::Real(&x[offset..]),
        View::Complex(x) => View::Complex(&x[offset..]),
    }
}

/// Contract two real or complex operands, with the matrix products of their element types; the
/// result is complex when either operand is.
/// # Arguments:
/// - `a`: First operand data and shape.
/// - `b`: Second operand data and shape.
/// - `keep`: Bit mask of labels that survive the contraction.
/// - `len`: Number of result elements, used to pick a reused buffer.
/// - `pools`: Released real and complex buffers.
/// # Returns:
/// - `(Values, TensorShape)`: Row-major result and its shape.
pub(super) fn contract_views(
    a: (View<'_>, &TensorShape),
    b: (View<'_>, &TensorShape),
    keep: u64,
    len: usize,
    pools: (&mut Vec<Vec<f64>>, &mut Vec<Vec<Complex64>>),
) -> (Values, TensorShape) {
    let (real, complex) = pools;
    match (a.0, b.0) {
        (View::Real(x), View::Real(y)) => {
            let buffer = pooled_buffer(real, len);
            let (r, s) =
                contract_tensor_pair((x, a.1), (y, b.1), keep, buffer, strided_gemm, strided_gemm);
            (Values::Real(r), s)
        }
        (View::Real(x), View::Complex(y)) => {
            let buffer = pooled_buffer(complex, len);
            let (r, s) = contract_tensor_pair(
                (x, a.1),
                (y, b.1),
                keep,
                buffer,
                strided_gemm_real_complex,
                strided_gemm_complex_real,
            );
            (Values::Complex(r), s)
        }
        (View::Complex(x), View::Real(y)) => {
            let buffer = pooled_buffer(complex, len);
            let (r, s) = contract_tensor_pair(
                (x, a.1),
                (y, b.1),
                keep,
                buffer,
                strided_gemm_complex_real,
                strided_gemm_real_complex,
            );
            (Values::Complex(r), s)
        }
        (View::Complex(x), View::Complex(y)) => {
            let buffer = pooled_buffer(complex, len);
            let (r, s) = contract_tensor_pair(
                (x, a.1),
                (y, b.1),
                keep,
                buffer,
                strided_gemm_complex,
                strided_gemm_complex,
            );
            (Values::Complex(r), s)
        }
    }
}

/// Fix one label of an operand shape to one value, removing it from the shape.
/// # Arguments:
/// - `shape`: Operand shape, updated in place.
/// - `label`: Label to fix.
/// - `value`: Value of the label.
/// # Returns:
/// - `usize`: Element offset of the fixed slice, zero when the operand lacks the label.
fn fix_label(
    shape: &mut TensorShape,
    label: u16,
    value: usize,
) -> usize {
    let Some(k) = shape.labels[..shape.n].iter().position(|&x| x == label) else {
        return 0;
    };
    let offset = value * shape.strides[k];
    for j in k..shape.n - 1 {
        shape.labels[j] = shape.labels[j + 1];
        shape.dims[j] = shape.dims[j + 1];
        shape.strides[j] = shape.strides[j + 1];
    }
    shape.n -= 1;
    shape.mask &= !(1 << label);
    offset
}

/// Add `c` times the final operand to the output block over the free indices.
/// Free indices sharing a representative are written only on their diagonal, representatives
/// absent from the operand are broadcast, labels not free are summed, and a term with no
/// factors is a constant.
/// # Arguments:
/// - `operand`: Final operand data and shape; a term without factors passes the unit scalar.
/// - `indices`: Class-local ids of the free indices of `out` in output order, and the fixed
///   representatives with their values; an output index whose representative is fixed is
///   written only at that value.
/// - `map`: Representative of every label.
/// - `extent`: Extent of every class-local label.
/// - `coeff`: Term coefficient.
/// - `out`: Row-major output block, updated in place.
/// # Returns:
/// - `()`: Mutates `out`.
fn scatter_into_output<S, T>(
    operand: (&[S], TensorShape),
    indices: (&[u16], &[(u16, usize)]),
    map: &[u16; 64],
    extent: &[usize],
    coeff: f64,
    out: &mut [T],
) where
    S: Copy + AddAssign + From<f64>,
    f64: Mul<S, Output = S>,
    T: AddAssign<S>,
{
    if out.is_empty() {
        return;
    }
    let (free, fixed) = indices;
    let (data, shape) = operand;

    // Extent, output stride and operand stride of every representative of the free indices;
    // free indices sharing a representative add their output strides, and an index whose
    // representative is fixed moves the base offset to its value.
    let mut reps = [(0u16, 0usize, 0usize, 0usize); 2 * MAXLABELS];
    let mut nr = 0;
    let mut base = 0;
    let mut stride = out.len();
    for &l in free {
        stride /= extent[l as usize];
        let r = map[l as usize];
        if let Some(&(_, v)) = fixed.iter().find(|x| x.0 == r) {
            base += v * stride;
            continue;
        }
        match reps[..nr].iter().position(|x| x.0 == r) {
            Some(k) => reps[k].2 += stride,
            None => {
                let from = shape.labels[..shape.n]
                    .iter()
                    .position(|&x| x == r)
                    .map_or(0, |p| shape.strides[p]);
                reps[nr] = (r, extent[r as usize], stride, from);
                nr += 1;
            }
        }
    }
    let reps = &reps[..nr];

    // Remaining operand labels are summed.
    let rep_mask = reps.iter().fold(0u64, |m, x| m | (1 << x.0));
    let mut summed = [(0usize, 0usize); MAXLABELS];
    let mut ns = 0;
    for k in 0..shape.n {
        if rep_mask & (1 << shape.labels[k]) == 0 {
            summed[ns] = (shape.dims[k], shape.strides[k]);
            ns += 1;
        }
    }
    let summed = &summed[..ns];
    let total = |offset: usize| {
        if summed.is_empty() {
            return data[offset];
        }
        let mut idx = [0usize; MAXLABELS];
        let mut extra = 0;
        let mut total = S::from(0.0);
        loop {
            total += data[offset + extra];
            let mut k = summed.len();
            loop {
                if k == 0 {
                    return total;
                }
                k -= 1;
                idx[k] += 1;
                extra += summed[k].1;
                if idx[k] < summed[k].0 {
                    break;
                }
                extra -= summed[k].1 * summed[k].0;
                idx[k] = 0;
            }
        }
    };

    // Walk the representatives with the last one innermost, advancing both offsets by odometer.
    let (last, inner) = match reps.split_last() {
        Some((&(_, d, so, sd), outer)) => ((d, so, sd), outer),
        None => ((1, 0, 0), reps),
    };
    let count = inner.iter().map(|x| x.1).product::<usize>();
    let mut idx = [0usize; 2 * MAXLABELS];
    let (mut o, mut p) = (base, 0usize);
    for _ in 0..count {
        let (d, so, sd) = last;
        if summed.is_empty() {
            for i in 0..d {
                out[o + i * so] += coeff * data[p + i * sd];
            }
        } else {
            for i in 0..d {
                out[o + i * so] += coeff * total(p + i * sd);
            }
        }

        let mut k = inner.len();
        while k > 0 {
            k -= 1;
            idx[k] += 1;
            o += inner[k].2;
            p += inner[k].3;
            if idx[k] < inner[k].1 {
                break;
            }
            o -= inner[k].2 * inner[k].1;
            p -= inner[k].3 * inner[k].1;
            idx[k] = 0;
        }
    }
}
