// overlap.rs

use crate::ir::{a, c, v, Group, Idx, Op, OpKind, Product, Spin};

/// Build spin components of `E^p_q`.
/// # Arguments:
/// - `p`: Creation index.
/// - `q`: Annihilation index.
/// - `g`: Group id.
/// # Returns:
/// - `Group`: Spin-expanded one-body group.
pub fn e1(p: Idx, q: Idx, g: usize) -> Group {
    Group {
        strings: vec![
            vec![
                Op { kind: OpKind::Create, idx: p, spin: Spin::Alpha, group: g },
                Op { kind: OpKind::Annihilate, idx: q, spin: Spin::Alpha, group: g },
            ],
            vec![
                Op { kind: OpKind::Create, idx: p, spin: Spin::Beta, group: g },
                Op { kind: OpKind::Annihilate, idx: q, spin: Spin::Beta, group: g },
            ],
        ],
    }
}

/// Build spin components of `E^{pq}_{rs}`.
/// # Arguments:
/// - `p`: First creation index.
/// - `q`: Second creation index.
/// - `r`: First annihilation index.
/// - `s`: Second annihilation index.
/// - `g`: Group id.
/// # Returns:
/// - `Group`: Spin-expanded two-body group.
pub fn e2(p: Idx, q: Idx, r: Idx, s: Idx, g: usize) -> Group {
    let mut strings = Vec::new();

    for sp in [Spin::Alpha, Spin::Beta] {
        for sq in [Spin::Alpha, Spin::Beta] {
            strings.push(vec![
                Op { kind: OpKind::Create, idx: p, spin: sp, group: g },
                Op { kind: OpKind::Create, idx: q, spin: sq, group: g },
                Op { kind: OpKind::Annihilate, idx: s, spin: sq, group: g },
                Op { kind: OpKind::Annihilate, idx: r, spin: sp, group: g },
            ]);
        }
    }

    Group { strings }
}

/// Build one metric block operator product.
/// # Arguments:
/// - `name`: Block name.
/// # Returns:
/// - `Product`: GNO product for this metric block.
pub fn block(name: &str) -> Product {
    match name {
        "C1" => Product { groups: vec![e1(c("i"), a("u"), 0), e1(a("v"), c("j"), 1)] },
        "C2" => Product { groups: vec![e1(a("t"), v("a"), 0), e1(v("b"), a("u"), 1)] },
        "C3" => Product { groups: vec![e1(a("u"), a("v"), 0), e1(a("x"), a("w"), 1)] },
        "C4" => Product { groups: vec![e2(c("i"), a("u"), a("v"), v("a"), 0), e2(a("x"), v("b"), c("j"), a("w"), 1)] },
        "C5" => Product { groups: vec![e2(c("i"), a("u"), v("a"), a("v"), 0), e2(v("b"), a("x"), c("j"), a("w"), 1)] },
        "C6" => Product { groups: vec![e2(c("i"), a("u"), v("a"), v("b"), 0), e2(v("c"), v("d"), c("j"), a("v"), 1)] },
        "C7" => Product { groups: vec![e2(c("i"), c("j"), a("u"), v("a"), 0), e2(a("v"), v("b"), c("k"), c("l"), 1)] },
        "C8" => Product { groups: vec![e2(c("i"), c("j"), a("u"), a("v"), 0), e2(a("w"), a("x"), c("k"), c("l"), 1)] },
        "C9" => Product { groups: vec![e2(c("i"), a("u"), a("v"), a("w"), 0), e2(a("y"), a("z"), c("j"), a("x"), 1)] },
        "C10" => Product { groups: vec![e2(a("t"), a("u"), a("v"), v("a"), 0), e2(a("z"), v("b"), a("x"), a("y"), 1)] },
        "C11" => Product { groups: vec![e2(a("t"), a("u"), v("a"), v("b"), 0), e2(v("c"), v("d"), a("v"), a("w"), 1)] },
        "C12" => Product { groups: vec![e2(a("p"), a("r"), a("q"), a("s"), 0), e2(a("t"), a("v"), a("u"), a("w"), 1)] },
        "C13" => Product { groups: vec![e1(a("u"), v("a"), 0), e2(a("x"), v("b"), a("v"), a("w"), 1)] },
        "C14" => Product { groups: vec![e1(c("i"), a("u"), 0), e2(a("w"), a("x"), c("j"), a("v"), 1)] },
        "C15" => Product { groups: vec![e1(a("t"), a("u"), 0), e2(a("y"), a("z"), a("w"), a("x"), 1)] },
        "C16" => Product { groups: vec![e2(c("i"), a("u"), a("w"), v("a"), 0), e2(v("b"), a("y"), c("j"), a("x"), 1)] },
        "C17" => Product { groups: vec![e1(c("i"), v("a"), 0), e1(v("b"), c("j"), 1)] },
        "C18" => Product { groups: vec![e1(c("i"), v("a"), 0), e2(a("x"), v("b"), c("j"), a("w"), 1)] },
        "C19" => Product { groups: vec![e1(c("i"), v("a"), 0), e2(v("b"), a("x"), c("j"), a("w"), 1)] },
        _ => panic!("unknown metric block {name}"),
    }
}
