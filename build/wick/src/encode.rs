// encode.rs 

use std::collections::BTreeMap;
use crate::ir::{Expr, Idx, Space, TensorKind, Term};
use crate::schema::{GeneratedTerm, OverlapBlockTerms, OverlapTermSet, TensorFactor};
use crate::specs::{idx, BlockSpec, BLOCKS};

/// Encode orbital space.
/// # Arguments:
/// - `x`: Orbital space.
/// # Returns:
/// - `u8`: Runtime space id.
fn sp(x: Space) -> u8 {
    match x {
        Space::Core => 0,
        Space::Active => 1,
        Space::Virtual => 2,
    }
}

/// Encode tensor kind.
/// # Arguments:
/// - `x`: Tensor kind.
/// # Returns:
/// - `u8`: Runtime tensor id.
fn tk(x: TensorKind) -> u8 {
    match x {
        TensorKind::Gamma1 => 0,
        TensorKind::Theta => 1,
        TensorKind::Lambda2 => 4,
        TensorKind::Lambda3 => 5,
        TensorKind::Lambda4 => 6,
    }
}

/// Return all index occurrences in one term.
/// # Arguments:
/// - `t`: Term.
/// # Returns:
/// - `Vec<Idx>`: Index occurrences.
fn inds(t: &Term) -> Vec<Idx> {
    let mut out = Vec::new();

    for d in &t.deltas {
        out.push(d.left);
        out.push(d.right);
    }

    for x in &t.tensors {
        out.extend(x.upper.iter().copied());
        out.extend(x.lower.iter().copied());
    }

    out
}

/// Build index table for a block.
/// # Arguments:
/// - `b`: Block metadata.
/// - `e`: Wick expression.
/// # Returns:
/// - `(Vec<Idx>, BTreeMap<Idx, u16>)`: Index table and ids.
fn table(b: BlockSpec, e: &Expr) -> (Vec<Idx>, BTreeMap<Idx, u16>) {
    let mut xs = Vec::new();
    let mut ids = BTreeMap::new();

    for name in b.lf.iter().chain(b.rf.iter()) {
        push(&mut xs, &mut ids, idx(name));
    }

    for t in e {
        for x in inds(t) {
            push(&mut xs, &mut ids, x);
        }
    }

    (xs, ids)
}

/// Push one index if new.
/// # Arguments:
/// - `xs`: Index table.
/// - `ids`: Index ids.
/// - `x`: Index.
/// # Returns:
/// - `()`: Mutates `xs` and `ids`.
fn push(xs: &mut Vec<Idx>, ids: &mut BTreeMap<Idx, u16>, x: Idx) {
    if ids.contains_key(&x) {
        return;
    }

    let id = xs.len() as u16;
    ids.insert(x, id);
    xs.push(x);
}

/// Encode one term.
/// # Arguments:
/// - `t`: Term.
/// - `ids`: Index ids.
/// - `free`: Free index ids.
/// # Returns:
/// - `GeneratedTerm`: Runtime term.
fn term(t: &Term, ids: &BTreeMap<Idx, u16>, free: &[u16]) -> GeneratedTerm {
    let mut loops = Vec::new();

    for x in inds(t) {
        let id = ids[&x];

        if !free.contains(&id) && !loops.contains(&id) {
            loops.push(id);
        }
    }

    GeneratedTerm(
        [t.coeff.num, t.coeff.den],
        loops,
        t.deltas.iter().map(|d| [ids[&d.left], ids[&d.right]]).collect(),
        t.tensors.iter().map(|x| {
            TensorFactor(
                tk(x.kind),
                x.upper.iter().map(|i| ids[i]).collect(),
                x.lower.iter().map(|i| ids[i]).collect(),
            )
        }).collect(),
    )
}

/// Generate one compact metric block.
/// # Arguments:
/// - `b`: Block metadata.
/// # Returns:
/// - `OverlapBlockTerms`: Runtime block terms.
fn block_terms(b: BlockSpec) -> OverlapBlockTerms {
    let e = crate::wick::eval(&crate::overlap::block(b.name));
    let (xs, ids) = table(b, &e);

    let left_free = b.lf.iter().map(|name| ids[&idx(name)]).collect::<Vec<_>>();
    let right_free = b.rf.iter().map(|name| ids[&idx(name)]).collect::<Vec<_>>();

    let mut free = left_free.clone();
    free.extend(right_free.iter().copied());

    OverlapBlockTerms {
        left: b.left.to_string(),
        right: b.right.to_string(),
        indices: xs.iter().map(|x| (x.name.to_string(), sp(x.space))).collect(),
        left_free,
        right_free,
        terms: e.iter().map(|t| term(t, &ids, &free)).collect(),
    }
}

/// Generate compact metric terms.
/// # Arguments:
/// - None.
/// # Returns:
/// - `OverlapTermSet`: Complete metric term table.
pub fn overlap_terms() -> OverlapTermSet {
    let mut space_kinds = BTreeMap::new();
    space_kinds.insert("core".to_string(), 0);
    space_kinds.insert("active".to_string(), 1);
    space_kinds.insert("virtual".to_string(), 2);

    let mut tensor_kinds = BTreeMap::new();
    tensor_kinds.insert("Gamma1".to_string(), 0);
    tensor_kinds.insert("Theta".to_string(), 1);
    tensor_kinds.insert("f".to_string(), 2);
    tensor_kinds.insert("g".to_string(), 3);
    tensor_kinds.insert("Lambda2".to_string(), 4);
    tensor_kinds.insert("Lambda3".to_string(), 5);
    tensor_kinds.insert("Lambda4".to_string(), 6);
    tensor_kinds.insert("t1".to_string(), 8);
    tensor_kinds.insert("t2".to_string(), 9);

    OverlapTermSet {
        version: 1,
        space_kinds,
        tensor_kinds,
        blocks: BLOCKS.iter().map(|&b| (b.name.to_string(), block_terms(b))).collect(),
    }
}
