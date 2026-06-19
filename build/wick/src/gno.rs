// gno.rs

use crate::ir::{Group, Idx, Op, OpKind, Spin};

/// Build spin components of `E^p_q`.
/// # Arguments:
/// - `p`: Creation index.
/// - `q`: Annihilation index.
/// - `g`: Group id.
/// # Returns:
/// - `Group`: Spin-expanded one-body group.
pub fn e1(
    p: Idx,
    q: Idx,
    g: usize,
) -> Group {
    Group {
        strings: vec![
            vec![
                Op {
                    kind: OpKind::Create,
                    idx: p,
                    spin: Spin::Alpha,
                    group: g,
                },
                Op {
                    kind: OpKind::Annihilate,
                    idx: q,
                    spin: Spin::Alpha,
                    group: g,
                },
            ],
            vec![
                Op {
                    kind: OpKind::Create,
                    idx: p,
                    spin: Spin::Beta,
                    group: g,
                },
                Op {
                    kind: OpKind::Annihilate,
                    idx: q,
                    spin: Spin::Beta,
                    group: g,
                },
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
pub fn e2(
    p: Idx,
    q: Idx,
    r: Idx,
    s: Idx,
    g: usize,
) -> Group {
    let mut strings = Vec::new();

    for sp in [Spin::Alpha, Spin::Beta] {
        for sq in [Spin::Alpha, Spin::Beta] {
            strings.push(vec![
                Op {
                    kind: OpKind::Create,
                    idx: p,
                    spin: sp,
                    group: g,
                },
                Op {
                    kind: OpKind::Create,
                    idx: q,
                    spin: sq,
                    group: g,
                },
                Op {
                    kind: OpKind::Annihilate,
                    idx: s,
                    spin: sq,
                    group: g,
                },
                Op {
                    kind: OpKind::Annihilate,
                    idx: r,
                    spin: sp,
                    group: g,
                },
            ]);
        }
    }

    Group { strings }
}
