// so/wick.rs

// Standard library imports.
use std::sync::OnceLock;

// External crate imports.
use num_rational::Ratio;
use rayon::prelude::*;
use smallvec::SmallVec;

// Parent/sibling imports.
use super::ops::{self, Component};
use super::{Expr, Kind, Space, Tensor, Term};

/// Largest cumulant rank kept in contractions.
const MAXCUMULANT: usize = 4;

/// One elementary operator of an instantiated product.
#[derive(Clone, Copy, Debug)]
struct Op {
    /// Creation (`true`) or annihilation operator.
    cre: bool,
    /// Index id.
    idx: u16,
    /// Orbital space.
    space: Space,
    /// Normal-ordered string the operator belongs to; operators of one string never contract.
    group: u8,
    /// Operator component the operator belongs to; every component must be connected.
    part: u8,
}

/// One contraction block: operator positions in the order that defines its value, the tensor it
/// produces (if any) and whether it identifies two indices through a Kronecker delta.
#[derive(Clone, Debug)]
struct Block {
    /// Operator positions in value order.
    order: SmallVec<[u8; 8]>,
    /// Contraction tensor, or `None` for a core or virtual delta.
    tensor: Option<Tensor>,
}

/// Instantiated operator product ready for contraction.
struct Product {
    /// Elementary operators in product order.
    ops: Vec<Op>,
    /// Operator tensors of every component.
    tensors: Vec<Tensor>,
    /// Orbital space of every index id.
    spaces: Vec<Space>,
    /// Product of component prefactors.
    coeff: Ratio<i64>,
}

/// Instantiate a product of operator components, each in its own normal-ordered string.
/// A component `X^{u}_{l}` contributes `a^\dagger_{l_1}\cdots a^\dagger_{l_m} a_{u_n}\cdots a_{u_1}`.
/// # Arguments:
/// - `comps`: Components with their normal-ordered string ids.
/// # Returns:
/// - `Product`: Operators, tensors, index spaces and prefactor.
fn instantiate_product(comps: &[(&Component, u8)]) -> Product {
    let mut out = Product {
        ops: Vec::new(),
        tensors: Vec::new(),
        spaces: Vec::new(),
        coeff: Ratio::from_integer(1),
    };

    for (part, &(c, group)) in comps.iter().enumerate() {
        let part = part as u8;
        let fresh = |space: Space, spaces: &mut Vec<Space>| {
            spaces.push(space);
            (spaces.len() - 1) as u16
        };
        let ann = c
            .ann
            .iter()
            .map(|&s| fresh(s, &mut out.spaces))
            .collect::<SmallVec<[u16; 4]>>();
        let cre = c
            .cre
            .iter()
            .map(|&s| fresh(s, &mut out.spaces))
            .collect::<SmallVec<[u16; 4]>>();

        // Creators in order, then annihilators in reverse order.
        for &i in &cre {
            out.ops.push(Op {
                cre: true,
                idx: i,
                space: out.spaces[i as usize],
                group,
                part,
            });
        }
        for &i in ann.iter().rev() {
            out.ops.push(Op {
                cre: false,
                idx: i,
                space: out.spaces[i as usize],
                group,
                part,
            });
        }

        out.tensors.push(Tensor {
            kind: c.kind,
            upper: ann,
            lower: cre,
        });
        out.coeff *= c.coeff;
    }

    out
}

/// Contract one product of normal-ordered operator components fully and connectedly.
/// Contractions are core pairs `a^\dagger_i a_j \to \delta_{ij}`, virtual pairs
/// `a_a a^\dagger_b \to \delta_{ab}`, active pairs `a^\dagger_p a_q \to \gamma^p_q` and
/// `a_q a^\dagger_p \to \eta^p_q`, and active blocks of `k` creators and `k` annihilators
/// (`2 \le k \le 4`) giving `\lambda_k^{c_1..c_k}_{d_1..d_k}` with the block reordered to
/// `a^\dagger_{c_1}\cdots a^\dagger_{c_k} a_{d_k}\cdots a_{d_1}`. No block may lie within a single
/// normal-ordered string, and every string must be connected to every other.
/// # Arguments:
/// - `comps`: Components with their normal-ordered string ids.
/// # Returns:
/// - `Vec<Term>`: One term per complete connected contraction.
pub(crate) fn contract_components(comps: &[(&Component, u8)]) -> Vec<Term> {
    let p = instantiate_product(comps);

    // Every contraction consumes equal numbers of creators and annihilators per space.
    for s in [Space::Core, Space::Active, Space::Virtual] {
        let cre = p.ops.iter().filter(|o| o.space == s && o.cre).count();
        let ann = p.ops.iter().filter(|o| o.space == s && !o.cre).count();
        if cre != ann {
            return Vec::new();
        }
    }

    let mut out = Vec::new();
    let mut blocks = Vec::new();
    let full = if p.ops.is_empty() {
        0
    } else {
        (1u32 << p.ops.len()) - 1
    };

    cover_operators(&p, full, &mut blocks, &mut out);
    out
}

/// Enumerate exact covers of the remaining operators by contraction blocks.
/// # Arguments:
/// - `p`: Instantiated product.
/// - `left`: Mask of uncontracted operator positions.
/// - `blocks`: Blocks chosen so far.
/// - `out`: Completed terms.
/// # Returns:
/// - `()`: Appends one term per connected complete contraction.
fn cover_operators(
    p: &Product,
    left: u32,
    blocks: &mut Vec<Block>,
    out: &mut Vec<Term>,
) {
    if left == 0 {
        if let Some(t) = finish_contraction(p, blocks) {
            out.push(t);
        }
        return;
    }

    // The lowest remaining operator must open a block with later partners.
    let i = left.trailing_zeros() as usize;
    let a = p.ops[i];
    let later = (i + 1..p.ops.len()).filter(|&j| left & (1 << j) != 0);

    match a.space {
        Space::Core | Space::Virtual => {
            // Core pairs need the creator first; virtual pairs need the annihilator first.
            if a.cre != (a.space == Space::Core) {
                return;
            }
            for j in later {
                let b = p.ops[j];
                if b.space == a.space && b.cre != a.cre && b.group != a.group {
                    blocks.push(Block {
                        order: SmallVec::from_slice(&[i as u8, j as u8]),
                        tensor: None,
                    });
                    cover_operators(p, left & !(1 << i) & !(1 << j), blocks, out);
                    blocks.pop();
                }
            }
        }
        Space::Active => {
            let partners = later
                .filter(|&j| p.ops[j].space == Space::Active)
                .collect::<SmallVec<[usize; 16]>>();

            for k in 1..=MAXCUMULANT {
                choose_combinations(p, i, k, &partners, left, blocks, out);
            }
        }
    }
}

/// Enumerate active blocks of rank `k` that contain operator `i` and recurse on each.
/// # Arguments:
/// - `p`: Instantiated product.
/// - `i`: Lowest remaining operator position.
/// - `k`: Block rank.
/// - `partners`: Remaining later active operator positions.
/// - `left`: Mask of uncontracted operator positions.
/// - `blocks`: Blocks chosen so far.
/// - `out`: Completed terms.
/// # Returns:
/// - `()`: Recurses once per admissible block.
fn choose_combinations(
    p: &Product,
    i: usize,
    k: usize,
    partners: &[usize],
    left: u32,
    blocks: &mut Vec<Block>,
    out: &mut Vec<Term>,
) {
    let need = 2 * k - 1;
    if partners.len() < need {
        return;
    }

    let mut pick = (0..need).collect::<SmallVec<[usize; 8]>>();

    loop {
        // Positions of this candidate block in product order.
        let mut pos = SmallVec::<[usize; 8]>::new();
        pos.push(i);
        pos.extend(pick.iter().map(|&n| partners[n]));

        let cre = pos.iter().filter(|&&q| p.ops[q].cre).count();
        let groups = pos
            .iter()
            .map(|&q| p.ops[q].group)
            .collect::<SmallVec<[u8; 8]>>();
        let mixed = groups.iter().any(|&g| g != groups[0]);

        if cre == k && mixed {
            let block = active_blocks(p, &pos, k);
            let mask = pos.iter().fold(left, |m, &q| m & !(1 << q));
            blocks.push(block);
            cover_operators(p, mask, blocks, out);
            blocks.pop();
        }

        // Advance to the next combination of `need` partners.
        let mut n = need;
        loop {
            if n == 0 {
                return;
            }
            n -= 1;
            if pick[n] < partners.len() - need + n {
                pick[n] += 1;
                for m in n + 1..need {
                    pick[m] = pick[m - 1] + 1;
                }
                break;
            }
        }
    }
}

/// Build one active contraction block from its operator positions.
/// # Arguments:
/// - `p`: Instantiated product.
/// - `pos`: Operator positions in product order.
/// - `k`: Block rank.
/// # Returns:
/// - `Block`: Block with its value order and contraction tensor.
fn active_blocks(
    p: &Product,
    pos: &[usize],
    k: usize,
) -> Block {
    let cre = pos
        .iter()
        .copied()
        .filter(|&q| p.ops[q].cre)
        .collect::<SmallVec<[usize; 4]>>();
    let ann = pos
        .iter()
        .copied()
        .filter(|&q| !p.ops[q].cre)
        .collect::<SmallVec<[usize; 4]>>();
    let idx = |xs: &[usize]| {
        xs.iter()
            .map(|&q| p.ops[q].idx)
            .collect::<SmallVec<[u16; 4]>>()
    };

    // One-body blocks keep product order: `\gamma` for `a^\dagger a` and `\eta` for `a a^\dagger`.
    if k == 1 {
        let kind = if p.ops[pos[0]].cre {
            Kind::Gamma
        } else {
            Kind::Eta
        };
        return Block {
            order: pos.iter().map(|&q| q as u8).collect(),
            tensor: Some(Tensor {
                kind,
                upper: idx(&cre),
                lower: idx(&ann),
            }),
        };
    }

    // Cumulant blocks are reordered to `a^\dagger_{c_1}..a^\dagger_{c_k} a_{d_k}..a_{d_1}`.
    let mut order = cre.iter().map(|&q| q as u8).collect::<SmallVec<[u8; 8]>>();
    order.extend(ann.iter().rev().map(|&q| q as u8));
    let kind = match k {
        2 => Kind::Lambda2,
        3 => Kind::Lambda3,
        _ => Kind::Lambda4,
    };

    Block {
        order,
        tensor: Some(Tensor {
            kind,
            upper: idx(&cre),
            lower: idx(&ann),
        }),
    }
}

/// Turn one complete contraction into a term if it is connected.
/// # Arguments:
/// - `p`: Instantiated product.
/// - `blocks`: Complete set of contraction blocks.
/// # Returns:
/// - `Option<Term>`: Signed term with deltas eliminated, or `None` if disconnected.
fn finish_contraction(
    p: &Product,
    blocks: &[Block],
) -> Option<Term> {
    // Connectivity over operator components, with blocks as hyperedges. Components sharing a
    // normal-ordered string, such as the two amplitudes of `\{T T\}`, are still separate
    // vertices: a cluster operator linked only to the projector is a disconnected term.
    let nparts = p
        .ops
        .iter()
        .map(|o| o.part)
        .max()
        .map(|x| x as usize + 1)
        .unwrap_or(0);
    let mut parent = (0..nparts).collect::<SmallVec<[usize; 8]>>();
    let find = |parent: &mut SmallVec<[usize; 8]>, mut x: usize| {
        while parent[x] != x {
            x = parent[x];
        }
        x
    };
    for b in blocks {
        let g0 = p.ops[b.order[0] as usize].part as usize;
        for &q in &b.order[1..] {
            let (x, y) = (
                find(&mut parent, g0),
                find(&mut parent, p.ops[q as usize].part as usize),
            );
            parent[x] = y;
        }
    }
    let present = p
        .ops
        .iter()
        .map(|o| o.part as usize)
        .collect::<SmallVec<[usize; 8]>>();
    let root = find(&mut parent, present[0]);
    if present.iter().any(|&g| find(&mut parent, g) != root) {
        return None;
    }

    // Sign of reordering the product into the concatenated block orders.
    let seq = blocks
        .iter()
        .flat_map(|b| b.order.iter().copied())
        .collect::<SmallVec<[u8; 16]>>();
    let mut odd = false;
    for x in 0..seq.len() {
        for y in x + 1..seq.len() {
            if seq[x] > seq[y] {
                odd = !odd;
            }
        }
    }

    // Deltas identify index pairs; keep the lower id of each pair.
    let mut rename = (0..p.spaces.len() as u16).collect::<Vec<_>>();
    for b in blocks.iter().filter(|b| b.tensor.is_none()) {
        // Resolve both ends through earlier identifications so chains merge fully.
        let x = rename[p.ops[b.order[0] as usize].idx as usize];
        let y = rename[p.ops[b.order[1] as usize].idx as usize];
        let (lo, hi) = (x.min(y), x.max(y));
        for r in rename.iter_mut() {
            if *r == hi {
                *r = lo;
            }
        }
    }
    let relabel = |t: &Tensor| Tensor {
        kind: t.kind,
        upper: t.upper.iter().map(|&x| rename[x as usize]).collect(),
        lower: t.lower.iter().map(|&x| rename[x as usize]).collect(),
    };

    let mut tensors = p.tensors.iter().map(relabel).collect::<Vec<_>>();
    tensors.extend(blocks.iter().filter_map(|b| b.tensor.as_ref()).map(relabel));

    // Renumber the surviving indices densely so eliminated ids leave no trace.
    let mut dense = vec![u16::MAX; p.spaces.len()];
    let mut spaces = Vec::new();
    for t in &mut tensors {
        for x in t.upper.iter_mut().chain(t.lower.iter_mut()) {
            if dense[*x as usize] == u16::MAX {
                dense[*x as usize] = spaces.len() as u16;
                spaces.push(p.spaces[*x as usize]);
            }
            *x = dense[*x as usize];
        }
    }

    let coeff = if odd { -p.coeff } else { p.coeff };

    Some(Term {
        coeff,
        spaces,
        tensors,
    })
}

/// Generate the spin-orbital residual of one excitation class at one order in `T`.
/// `R_0 = \langle\tau^\dagger H\rangle_c`, `R_1 = \langle\tau^\dagger H T\rangle_c` and
/// `R_2 = \tfrac12\langle\tau^\dagger H\{T T\}\rangle_c`, where both cluster operators of the
/// last share one normal-ordered string.
/// # Arguments:
/// - `bra`: Residual projector component.
/// - `order`: Order in `T`.
/// # Returns:
/// - `Expr`: Canonically combined spin-orbital residual.
pub(crate) fn residual_expression(
    bra: &Component,
    order: usize,
) -> Expr {
    let h = ops::normal_ordered_hamiltonian();
    let products = h
        .iter()
        .map(|x| vec![(bra, 0u8), (x, 1u8)])
        .collect::<Vec<_>>();

    sum_products(
        &append_cluster_operators(products, order),
        taylor_factor(order),
    )
}

/// Generate the spin-orbital correlation energy at one order in `T`,
/// `E_1 = \langle\Phi|H T|\Phi\rangle_c` and `E_2 = \tfrac12\langle\Phi|H\{T T\}|\Phi\rangle_c`.
/// The reference energy `E_0 = \langle\Phi|H|\Phi\rangle` is evaluated from the RDMs instead.
/// # Arguments:
/// - `order`: Order in `T`, `1` or `2`.
/// # Returns:
/// - `Expr`: Canonically combined spin-orbital energy contribution.
pub(crate) fn energy_expression(order: usize) -> Expr {
    let h = ops::normal_ordered_hamiltonian();
    let products = h.iter().map(|x| vec![(x, 1u8)]).collect::<Vec<_>>();

    sum_products(
        &append_cluster_operators(products, order),
        taylor_factor(order),
    )
}

/// Derive the spin-orbital metric `\langle\Phi|\hat\tau_\mu^\dagger\hat\tau_\nu|\Phi\rangle` of two
/// excitation classes. Neither operator contracts with itself, so every term joins both.
/// # Arguments:
/// - `bra`: Left projector component.
/// - `ket`: Right excitation component.
/// # Returns:
/// - `Expr`: Canonically combined spin-orbital metric block.
pub(crate) fn metric_expression(
    bra: &Component,
    ket: &Component,
) -> Expr {
    sum_products(&[vec![(bra, 0), (ket, 1)]], Ratio::from_integer(1))
}

/// Derive the connected zeroth-order coupling
/// `\langle\Phi|\hat\tau_\mu^\dagger\hat H_0\hat\tau_\nu|\Phi\rangle_c` of two excitation classes
/// for the normal-ordered Dyall Hamiltonian `\hat H_0`.
/// # Arguments:
/// - `bra`: Left projector component.
/// - `ket`: Right excitation component.
/// # Returns:
/// - `Expr`: Canonically combined spin-orbital coupling block.
pub(crate) fn dyall_coupling_expression(
    bra: &Component,
    ket: &Component,
) -> Expr {
    let h0 = ops::dyall_hamiltonian();
    let products = h0
        .iter()
        .map(|x| vec![(bra, 0u8), (x, 1u8), (ket, 2u8)])
        .collect::<Vec<_>>();

    sum_products(&products, Ratio::from_integer(1))
}

/// Append `order` cluster operators, sharing one normal-ordered string, to every product.
/// # Arguments:
/// - `products`: Component products with their normal-ordered string ids.
/// - `order`: Number of cluster operators to append.
/// # Returns:
/// - `Vec<Vec<(&Component, u8)>>`: Products with every combination of cluster types.
fn append_cluster_operators(
    mut products: Vec<Vec<(&Component, u8)>>,
    order: usize,
) -> Vec<Vec<(&Component, u8)>> {
    static CLUSTER: OnceLock<Vec<Component>> = OnceLock::new();
    let t = CLUSTER.get_or_init(ops::cluster_operator);

    for _ in 0..order {
        products = products
            .into_iter()
            .flat_map(|p| t.iter().map(move |x| [p.clone(), vec![(x, 2u8)]].concat()))
            .collect();
    }
    products
}

/// Return the Taylor prefactor `1 / n!` of the normal-ordered exponential at order `n`.
/// # Arguments:
/// - `order`: Order in `T`.
/// # Returns:
/// - `Ratio<i64>`: Prefactor.
fn taylor_factor(order: usize) -> Ratio<i64> {
    Ratio::new(1, (1..=order as i64).product::<i64>())
}

/// Contract every component product and combine the terms canonically.
/// # Arguments:
/// - `products`: Component products with their normal-ordered string ids.
/// - `scale`: Prefactor applied to every term.
/// # Returns:
/// - `Expr`: Canonically combined spin-orbital expression.
fn sum_products(
    products: &[Vec<(&Component, u8)>],
    scale: Ratio<i64>,
) -> Expr {
    products
        .par_iter()
        .fold(Expr::default, |mut acc, comps| {
            for mut term in contract_components(comps) {
                term.coeff *= scale;
                super::add_term(&mut acc, &term);
            }
            acc
        })
        .reduce(Expr::default, |mut a, b| {
            super::merge_expressions(&mut a, b);
            a
        })
}
