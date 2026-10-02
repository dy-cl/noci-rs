// nocc/terms/graph.rs
//! Shared contraction graph of one planned table.
//!
//! Every intermediate of every kept term is a node of one graph, identified by its canonical
//! form up to a relabelling of its labels, so an intermediate occurring in many terms is
//! contracted once. Terms whose final products `A B_t` share the operand `A`, attached in the
//! same way, form a group evaluated by distributivity as one product,
//! `\sum_t c_t A B_t = A \sum_t c_t B_t`. Large groups are split into pieces. Workers take the
//! pieces in order from one queue and share one cache of nodes: each node is contracted once,
//! by the first worker to need it, and freed after its last use, so the nodes held stay close
//! to those of an evaluation in order.

// Standard library imports.
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// External crate imports.
use rayon::prelude::*;

// Crate-root imports.
use crate::maths::contract::{LabelSizes, TensorShape, strided_tensor_shape};

// Parent/sibling imports.
use super::plan::{TablePlan, TermTable, label_map};
use super::term::{add_view, contract_views, result_size, scatter_view};
use super::workspace::{Values, View, Workspace};

/// Number of kept terms analysed together before their nodes are merged into the graph.
const BATCH: usize = 1 << 16;

/// Pieces per worker thread into which the largest groups are split.
const PIECES: usize = 16;

/// Largest label id of a table, above which labels are given to summed operand labels.
const LABELS: usize = 64;

/// Operand of a node or a group over that node's or group's labels.
#[derive(Clone)]
pub(super) enum Child {
    /// A dense factor block, by table-local block id, with the label of every slot.
    Block(u32, Vec<u16>),
    /// A node, by id, with the label of every canonical label of the node.
    Node(u32, Vec<u16>),
}

/// One shared intermediate, the contraction of two operands over its canonical labels.
pub(super) struct Node {
    /// The two operands over the node's canonical labels.
    operands: [Child; 2],
    /// Canonical labels kept by the contraction.
    keep: u64,
    /// Extent of every canonical label.
    extent: Vec<usize>,
}

/// One term's contribution to a group: its other final operand and coefficient.
pub(super) struct Member {
    /// Other final operand over the group's labels.
    operand: Child,
    /// Labels summed within the operand alone, with their extents.
    summed: Vec<(u16, usize)>,
    /// Coefficient of the term.
    coefficient: f64,
}

/// Terms whose final products share one operand attached in the same way.
pub(super) struct Group {
    /// Shared operand over the group's labels, those of its first term.
    shared: Child,
    /// Labels of the summed other operands, in accumulator order.
    labels: Vec<u16>,
    /// Labels kept by the final product: the free labels of either operand.
    keep: u64,
    /// Representative of every label of the first term, for the final scatter.
    map: [u16; 64],
    /// Contribution of every term.
    members: Vec<Member>,
}

/// Shared contraction graph of one table.
#[derive(Default)]
pub(super) struct TableGraph {
    /// Shared intermediates, every node after the nodes it contracts.
    nodes: Vec<Node>,
    /// Groups of final products, in evaluation order.
    groups: Vec<Group>,
    /// Number of consumers of every node: the nodes that contract it and the group operands
    /// that are it.
    uses: Vec<u32>,
    /// Kept-term positions evaluated term by term: terms with fewer than two factors, or whose
    /// summed labels do not fit beside the table's.
    pub(super) direct: Vec<u32>,
}

/// Identity of a node or a block operand pattern.
type Key = u128;

/// Operand of a term during canonicalisation: its identity, its canonical labels as term
/// labels, and its serialisation.
struct Canon {
    /// Block id for a block, or node key.
    child: CanonChild,
    /// Term label of every canonical label, in canonical order.
    names: Vec<u16>,
    /// Canonical serialisation, comparable between operands.
    serial: Vec<u64>,
}

/// Identity of a term operand.
#[derive(Clone)]
enum CanonChild {
    /// Block with the term label of every slot.
    Block(u32, Vec<u16>),
    /// Node by key.
    Node(Key),
}

/// Node of one term, before merging into the graph.
struct TermNode {
    /// Node key.
    key: Key,
    /// The two operands, with their labels over the node's canonical labels.
    operands: [(CanonChild, Vec<u16>); 2],
    /// Canonical labels kept.
    keep: u64,
    /// Extent of every canonical label.
    extent: Vec<usize>,
}

/// One way of grouping a term's final product, sharing one of its two operands.
struct Side {
    /// Group key.
    key: Key,
    /// Shared operand.
    shared: CanonChild,
    /// Term label of every canonical label of the shared operand.
    names: Vec<u16>,
    /// Other operand and its labels.
    other: (CanonChild, Vec<u16>),
    /// Accumulator labels as term labels, in group order.
    labels: Vec<u16>,
    /// Labels of the other operand summed within it alone.
    summed: Vec<u16>,
    /// Labels kept by the final product.
    keep: u64,
}

/// Analysis of one kept term.
struct TermGraph {
    /// Nodes in contraction order, every node after its operands.
    nodes: Vec<TermNode>,
    /// The two ways of grouping the final product.
    sides: [Side; 2],
    /// Representative of every label.
    map: [u16; 64],
    /// Coefficient.
    coefficient: f64,
    /// Estimated multiply-adds of the term.
    cost: f64,
}

/// Hash a serialisation to a 128-bit key.
/// # Arguments:
/// - `serial`: Serialisation.
/// # Returns:
/// - `Key`: Two independent 64-bit hashes.
fn hash_key(serial: &[u64]) -> Key {
    let mut a = std::collections::hash_map::DefaultHasher::new();
    let mut b = std::collections::hash_map::DefaultHasher::new();
    0x9e37_79b9_7f4a_7c15u64.hash(&mut b);
    serial.hash(&mut a);
    serial.hash(&mut b);
    ((a.finish() as u128) << 64) | b.finish() as u128
}

/// Canonical form of one factor block: labels numbered by first appearance over its slots.
/// # Arguments:
/// - `block`: Table-local block id.
/// - `slots`: Term label of every slot.
/// # Returns:
/// - `Canon`: Block operand.
fn block_canon(
    block: u32,
    slots: Vec<u16>,
) -> Canon {
    let mut names = Vec::<u16>::new();
    let mut serial = vec![1, block as u64, slots.len() as u64];
    for &l in &slots {
        let p = names.iter().position(|&x| x == l).unwrap_or_else(|| {
            names.push(l);
            names.len() - 1
        });
        serial.push(p as u64);
    }
    Canon {
        child: CanonChild::Block(block, slots),
        names,
        serial,
    }
}

/// Labels of one operand over a node's canonical labels.
/// # Arguments:
/// - `x`: Operand.
/// - `names`: Term label of every canonical label of the node.
/// # Returns:
/// - `Vec<u16>`: Slot labels of a block, or the node label of every canonical label of a node.
fn operand_labels(
    x: &Canon,
    names: &[u16],
) -> Vec<u16> {
    let at = |l: &u16| names.iter().position(|n| n == l).unwrap() as u16;
    match &x.child {
        CanonChild::Block(_, slots) => slots.iter().map(at).collect(),
        CanonChild::Node(_) => x.names.iter().map(at).collect(),
    }
}

/// Canonical form of the contraction of two operands keeping some term labels: the operand
/// order giving the smaller serialisation, with labels numbered by first appearance.
/// # Arguments:
/// - `x`: First operand.
/// - `y`: Second operand.
/// - `keep`: Term labels kept.
/// - `extent`: Extent of every term label.
/// # Returns:
/// - `(Canon, TermNode)`: Node operand and node.
fn product_canon(
    x: &Canon,
    y: &Canon,
    keep: u64,
    extent: &[usize],
) -> (Canon, TermNode) {
    let ordered = |a: &Canon, b: &Canon| {
        let mut names = a.names.clone();
        for &l in &b.names {
            if !names.contains(&l) {
                names.push(l);
            }
        }
        let mut serial = vec![2];
        for x in [a, b] {
            serial.extend(&x.serial);
            serial.push(u64::MAX);
            serial.extend(
                x.names
                    .iter()
                    .map(|l| names.iter().position(|n| n == l).unwrap() as u64),
            );
            serial.push(u64::MAX);
        }
        let kept = names
            .iter()
            .enumerate()
            .filter(|&(_, &l)| keep & (1 << l) != 0)
            .fold(0u64, |m, (p, _)| m | (1 << p));
        serial.push(kept);
        (serial, names, kept)
    };
    let (sx, nx, kx) = ordered(x, y);
    let (sy, ny, ky) = ordered(y, x);
    let (serial, names, kept, first, second) = if sx <= sy {
        (sx, nx, kx, x, y)
    } else {
        (sy, ny, ky, y, x)
    };
    let key = hash_key(&serial);
    let node = TermNode {
        key,
        operands: [
            (first.child.clone(), operand_labels(first, &names)),
            (second.child.clone(), operand_labels(second, &names)),
        ],
        keep: kept,
        extent: names.iter().map(|&l| extent[l as usize]).collect(),
    };
    let canon = Canon {
        child: CanonChild::Node(key),
        names,
        serial: vec![3, (key >> 64) as u64, key as u64],
    };
    (canon, node)
}

/// Grouping of a final product that shares operand `a`: the key fixes `a` with the attachment
/// of its labels to the free labels, the layout of the accumulated labels of `b`, and the
/// representatives of the free labels.
/// # Arguments:
/// - `a`: Shared operand.
/// - `b`: Other operand.
/// - `free`: Free term labels.
/// - `fixed`: Representative of every non-summed label, as a serialisation.
/// # Returns:
/// - `Side`: Grouping by `a`.
fn final_side(
    a: &Canon,
    b: &Canon,
    free: u64,
    fixed: &[u64],
) -> Side {
    let is_free = |l: u16| free & (1 << l) != 0;

    // Accumulated labels of `b`: free labels, and labels shared with `a` by their canonical
    // position in `a`.
    let mut layout = Vec::<(u64, u64, u16)>::new();
    let mut summed = Vec::new();
    for &l in &b.names {
        if is_free(l) {
            layout.push((0, l as u64, l));
        } else if let Some(p) = a.names.iter().position(|&x| x == l) {
            layout.push((1, p as u64, l));
        } else {
            summed.push(l);
        }
    }
    layout.sort_unstable();

    let mut serial = a.serial.clone();
    serial.push(u64::MAX);
    serial.extend(
        a.names
            .iter()
            .map(|&l| if is_free(l) { l as u64 } else { u64::MAX - 1 }),
    );
    serial.push(u64::MAX);
    serial.extend(layout.iter().flat_map(|&(k, v, _)| [k, v]));
    serial.push(u64::MAX);
    serial.extend(fixed);

    let labels = layout.iter().map(|&(_, _, l)| l).collect::<Vec<_>>();
    let keep = a
        .names
        .iter()
        .chain(&labels)
        .filter(|&&l| is_free(l))
        .fold(0u64, |m, &l| m | (1 << l));
    let other_labels = match &b.child {
        CanonChild::Block(_, slots) => slots.clone(),
        CanonChild::Node(_) => b.names.clone(),
    };
    Side {
        key: hash_key(&serial),
        shared: a.child.clone(),
        names: a.names.clone(),
        other: (b.child.clone(), other_labels),
        labels,
        summed,
        keep,
    }
}

/// Analyse one kept term: its nodes in contraction order and the two ways of grouping its
/// final product, or `None` when it is evaluated term by term or vanishes.
/// # Arguments:
/// - `table`: Terms and index spaces of the table.
/// - `plan`: Table plan.
/// - `k`: Kept-term position.
/// - `extent`: Extent of every label.
/// - `sizes`: Joint index-space sizes of label masks.
/// # Returns:
/// - `Option<Option<TermGraph>>`: `None` when the term vanishes, `Some(None)` when it is
///   evaluated term by term, and otherwise its analysis.
fn term_graph(
    table: TermTable<'_>,
    plan: &TablePlan,
    k: usize,
    extent: &[usize],
    sizes: &LabelSizes,
) -> Option<Option<TermGraph>> {
    let (terms, indices) = table;
    let term = &terms[plan.terms[k] as usize];
    let range = |starts: &[u32]| starts[k] as usize..starts[k + 1] as usize;
    let map = label_map(&plan.substitutions[range(&plan.substitution_starts)]);
    let blocks = &plan.factor_blocks[range(&plan.term_starts)];
    let steps = &plan.steps[range(&plan.step_starts)];

    // Leaf operands over their substituted labels; a term over an empty space vanishes.
    let mut ops = term
        .3
        .iter()
        .zip(blocks)
        .map(|(f, &b)| {
            block_canon(
                b,
                f.1.iter().chain(&f.2).map(|&x| map[x as usize]).collect(),
            )
        })
        .collect::<Vec<_>>();
    let mut masks = ops
        .iter()
        .map(|c| c.names.iter().fold(0u64, |m, &l| m | (1 << l)))
        .collect::<Vec<_>>();
    let all = masks.iter().fold(0u64, |m, &x| m | x);
    if sizes.size(all) == 0.0 {
        return None;
    }
    if steps.is_empty() {
        return Some(None);
    }
    let summed = term.1.iter().fold(0u64, |m, &x| m | (1 << x));
    let free = all & !summed;

    // Every step but the last is a node; each keeps the labels of the operands still to be
    // contracted and the free labels.
    let mut nodes = Vec::with_capacity(steps.len());
    let mut live = (1u64 << ops.len()) - 1;
    let mut cost = 0.0;
    for &(i, j) in &steps[..steps.len() - 1] {
        live &= !((1 << i) | (1 << j));
        let keep = (0..masks.len())
            .filter(|&q| live & (1 << q) != 0)
            .fold(free, |m, q| m | masks[q]);
        let joint = masks[i as usize] | masks[j as usize];
        cost += sizes.size(joint);
        let (canon, node) = product_canon(&ops[i as usize], &ops[j as usize], keep, extent);
        nodes.push(node);
        live |= 1 << ops.len();
        ops.push(canon);
        masks.push(joint & keep);
    }

    // The final product groups by either operand.
    let (i, j) = *steps.last().unwrap();
    let (a, b) = (&ops[i as usize], &ops[j as usize]);
    cost += sizes.size(masks[i as usize] | masks[j as usize]);
    let fixed = (0..indices.len())
        .filter(|&l| summed & (1 << l) == 0)
        .flat_map(|l| [l as u64, map[l] as u64])
        .collect::<Vec<_>>();
    let sides = [
        final_side(a, b, free, &fixed),
        final_side(b, a, free, &fixed),
    ];

    // Summed labels of the other operand take ids above the table's labels.
    if sides
        .iter()
        .any(|s| indices.len() + s.summed.len() > LABELS)
    {
        return Some(None);
    }

    Some(Some(TermGraph {
        nodes,
        sides,
        map,
        coefficient: plan.coefficients[k],
        cost,
    }))
}

/// Map term operand children to graph children.
/// # Arguments:
/// - `child`: Term child.
/// - `labels`: Labels of the child in the target label set.
/// - `ids`: Graph id of every node key.
/// # Returns:
/// - `Child`: Graph child.
fn graph_child(
    child: &CanonChild,
    labels: Vec<u16>,
    ids: &HashMap<Key, u32>,
) -> Child {
    match child {
        CanonChild::Block(b, _) => Child::Block(*b, labels),
        CanonChild::Node(key) => Child::Node(ids[key], labels),
    }
}

/// Build the shared contraction graph of one table: merge the nodes of every kept term, group
/// the final products by the operand shared by more terms, and split the largest groups into
/// pieces of similar estimated cost.
/// # Arguments:
/// - `table`: Terms and index spaces of the table.
/// - `plan`: Table plan without its graph.
/// - `extent`: Extent of every label.
/// # Returns:
/// - `TableGraph`: Graph of the table.
pub(super) fn build_table_graph(
    table: TermTable<'_>,
    plan: &TablePlan,
    extent: &[usize],
) -> TableGraph {
    let sizes = LabelSizes::new(extent);
    let mut graph = TableGraph::default();
    let mut ids = HashMap::<Key, u32>::new();
    let mut analysed = Vec::new();

    // Analyse the terms in parallel batches and merge their nodes in term order, so every node
    // follows the nodes it contracts.
    let positions = (0..plan.terms.len()).collect::<Vec<_>>();
    for batch in positions.chunks(BATCH) {
        let part = batch
            .par_iter()
            .map(|&k| (k, term_graph(table, plan, k, extent, &sizes)))
            .collect::<Vec<_>>();
        for (k, result) in part {
            match result {
                None => {}
                Some(None) => graph.direct.push(k as u32),
                Some(Some(mut t)) => {
                    for node in t.nodes.drain(..) {
                        if ids.contains_key(&node.key) {
                            continue;
                        }
                        let [(a, la), (b, lb)] = node.operands;
                        let operands = [graph_child(&a, la, &ids), graph_child(&b, lb, &ids)];
                        ids.insert(node.key, graph.nodes.len() as u32);
                        graph.nodes.push(Node {
                            operands,
                            keep: node.keep,
                            extent: node.extent,
                        });
                    }
                    analysed.push(t);
                }
            }
        }
    }

    // Each term joins the larger of its two candidate groups.
    let mut count = HashMap::<Key, usize>::new();
    for t in &analysed {
        for s in &t.sides {
            *count.entry(s.key).or_default() += 1;
        }
    }
    let mut slot = HashMap::<Key, usize>::new();
    let mut costs = Vec::new();
    for t in &analysed {
        let side = &t.sides[(count[&t.sides[1].key] > count[&t.sides[0].key]) as usize];
        let g = *slot.entry(side.key).or_insert_with(|| {
            let labels = side.labels.clone();
            graph.groups.push(Group {
                shared: graph_child(
                    &side.shared,
                    match &side.shared {
                        CanonChild::Block(_, slots) => slots.clone(),
                        CanonChild::Node(_) => side.names.clone(),
                    },
                    &ids,
                ),
                labels,
                keep: side.keep,
                map: t.map,
                members: Vec::new(),
            });
            costs.push(Vec::new());
            graph.groups.len() - 1
        });

        // Relabel the other operand to the group's labels: accumulated labels by position,
        // labels summed within it to ids above the table's.
        let group = &mut graph.groups[g];
        let base = table.1.len() as u16;
        let rename = |l: u16| {
            side.labels
                .iter()
                .position(|&x| x == l)
                .map(|p| group.labels[p])
                .unwrap_or_else(|| base + side.summed.iter().position(|&x| x == l).unwrap() as u16)
        };
        let labels = side.other.1.iter().map(|&l| rename(l)).collect();
        let summed = side
            .summed
            .iter()
            .map(|&l| (rename(l), extent[l as usize]))
            .collect();
        group.members.push(Member {
            operand: graph_child(&side.other.0, labels, &ids),
            summed,
            coefficient: t.coefficient,
        });
        costs[g].push(t.cost);
    }

    // Split the largest groups into pieces of similar estimated cost, each with its own final
    // product, so no group holds up the workers.
    let limit =
        costs.iter().flatten().sum::<f64>() / (rayon::current_num_threads().max(1) * PIECES) as f64;
    let mut groups = Vec::with_capacity(graph.groups.len());
    for (g, member_costs) in graph.groups.drain(..).zip(costs) {
        let mut members = g.members.into_iter().zip(member_costs).peekable();
        while members.peek().is_some() {
            let mut piece = Vec::new();
            let mut weight = 0.0;
            while let Some((m, c)) = members.next_if(|_| piece.is_empty() || weight < limit) {
                weight += c;
                piece.push(m);
            }
            groups.push(Group {
                shared: g.shared.clone(),
                labels: g.labels.clone(),
                keep: g.keep,
                map: g.map,
                members: piece,
            });
        }
    }
    graph.groups = groups;

    // Consumers of every node over the whole evaluation.
    graph.uses = vec![0; graph.nodes.len()];
    let node = |c: &Child| match c {
        Child::Node(id, _) => Some(*id as usize),
        Child::Block(..) => None,
    };
    for n in &graph.nodes {
        for id in n.operands.iter().filter_map(node) {
            graph.uses[id] += 1;
        }
    }
    for g in &graph.groups {
        for id in std::iter::once(&g.shared)
            .chain(g.members.iter().map(|m| &m.operand))
            .filter_map(node)
        {
            graph.uses[id] += 1;
        }
    }
    graph
}

/// Shared state of one node during an evaluation: its data once contracted, and its remaining
/// consumers.
struct Slot {
    /// Data and shape over the node's canonical labels, while held.
    held: Mutex<Option<Arc<(Values, TensorShape)>>>,
    /// Remaining consumers.
    left: AtomicU32,
}

/// Data and shape of one operand over its parent's labels.
/// # Arguments:
/// - `child`: Operand.
/// - `data`: Data of every table-local block.
/// - `extent`: Extent of every parent label.
/// - `node`: Data and shape of the operand's node, for a node operand.
/// # Returns:
/// - `(View<'a>, TensorShape)`: Operand data and shape.
fn child_view<'a>(
    child: &Child,
    data: &[View<'a>],
    extent: &[usize],
    node: Option<&'a (Values, TensorShape)>,
) -> (View<'a>, TensorShape) {
    match (child, node) {
        (Child::Node(_, labels), Some((values, shape))) => {
            let mut shape = *shape;
            shape.mask = 0;
            for l in shape.labels[..shape.n].iter_mut() {
                *l = labels[*l as usize];
                shape.mask |= 1 << *l;
            }
            (values.view(0), shape)
        }
        (Child::Block(id, slots), _) => (data[*id as usize], strided_tensor_shape(slots, extent)),
        (Child::Node(..), None) => panic!("node operand without its data"),
    }
}

/// Return an operand's node, contracting it and the nodes it needs unless held already. A node
/// is contracted once, under its lock, by the first worker to need it.
/// # Arguments:
/// - `child`: Operand.
/// - `graph`: Table graph.
/// - `data`: Data of every table-local block.
/// - `slots`: Shared state of every node.
/// - `ws`: Worker storage.
/// # Returns:
/// - `Option<Arc<(Values, TensorShape)>>`: Node data and shape, or `None` for a block.
fn ensure(
    child: &Child,
    graph: &TableGraph,
    data: &[View<'_>],
    slots: &[Slot],
    ws: &mut Workspace,
) -> Option<Arc<(Values, TensorShape)>> {
    let Child::Node(id, _) = child else {
        return None;
    };
    let mut held = slots[*id as usize].held.lock().unwrap();
    if let Some(x) = held.as_ref() {
        return Some(x.clone());
    }
    let node = &graph.nodes[*id as usize];
    let operands = node
        .operands
        .iter()
        .map(|c| ensure(c, graph, data, slots, ws))
        .collect::<Vec<_>>();
    let result = {
        let a = child_view(
            &node.operands[0],
            data,
            &node.extent,
            operands[0].as_deref(),
        );
        let b = child_view(
            &node.operands[1],
            data,
            &node.extent,
            operands[1].as_deref(),
        );
        let len = result_size(&a.1, &b.1, node.keep);
        contract_views(
            (a.0, &a.1),
            (b.0, &b.1),
            node.keep,
            len,
            (&mut ws.pool, &mut ws.complex_pool),
        )
    };
    let result = Arc::new(result);
    *held = Some(result.clone());
    drop(held);
    for c in &node.operands {
        release(c, slots);
    }
    Some(result)
}

/// Mark one use of an operand's node, dropping the shared reference after its last use.
/// # Arguments:
/// - `child`: Operand.
/// - `slots`: Shared state of every node.
/// # Returns:
/// - `()`: Mutates `slots`.
fn release(
    child: &Child,
    slots: &[Slot],
) {
    let Child::Node(id, _) = child else {
        return;
    };
    let slot = &slots[*id as usize];
    if slot.left.fetch_sub(1, Ordering::AcqRel) == 1 {
        slot.held.lock().unwrap().take();
    }
}

/// Evaluate one group into an output block: sum every term's other operand, times its
/// coefficient, over the accumulator labels, then contract the sum with the shared operand
/// and scatter it as the group's first term.
/// # Arguments:
/// - `g`: Group.
/// - `graph`: Table graph.
/// - `tensors`: Data of every table-local block, extent of every label, and the extents widened
///   to every label id.
/// - `free`: Class-local ids of the free indices, in output order.
/// - `slots`: Shared state of every node.
/// - `out`: Output block, updated in place.
/// - `ws`: Worker storage.
/// # Returns:
/// - `()`: Mutates `out`, `slots` and `ws`.
fn evaluate_group(
    g: &Group,
    graph: &TableGraph,
    tensors: (&[View<'_>], &[usize], &[usize; LABELS]),
    free: &[u16],
    slots: &[Slot],
    out: &mut Values,
    ws: &mut Workspace,
) {
    let (data, extent, wide) = tensors;
    let complex = matches!(out, Values::Complex(_));

    // Labels summed within one operand alone are summed as it is added.
    let len = g
        .labels
        .iter()
        .map(|&l| extent[l as usize])
        .product::<usize>();
    let mut acc = Values::zeros(len, complex);
    for m in &g.members {
        let node = ensure(&m.operand, graph, data, slots, ws);
        let mut ext = *wide;
        for &(l, d) in &m.summed {
            ext[l as usize] = d;
        }
        let operand = child_view(&m.operand, data, &ext, node.as_deref());
        add_view(operand, &g.labels, &ext, m.coefficient, &mut acc.out());
        drop(node);
        release(&m.operand, slots);
    }

    // One product with the shared operand.
    let node = ensure(&g.shared, graph, data, slots, ws);
    let product = {
        let a = child_view(&g.shared, data, extent, node.as_deref());
        let b = strided_tensor_shape(&g.labels, extent);
        let len = result_size(&a.1, &b, g.keep);
        contract_views(
            (a.0, &a.1),
            (acc.view(0), &b),
            g.keep,
            len,
            (&mut ws.pool, &mut ws.complex_pool),
        )
    };
    drop(node);
    release(&g.shared, slots);
    scatter_view(
        (product.0.view(0), product.1),
        (free, &[]),
        (&g.map, extent),
        1.0,
        &mut out.out(),
    );
    for values in [product.0, acc] {
        match values {
            Values::Real(x) => ws.pool.push(x),
            Values::Complex(x) => ws.complex_pool.push(x),
        }
    }
}

/// Evaluate the groups of a table graph into a dense output block over the free indices.
/// Every worker takes the next group from one queue into its own block, with nodes shared
/// through one cache, and the blocks are summed.
/// # Arguments:
/// - `graph`: Table graph.
/// - `data`: Data of every table-local block.
/// - `extent`: Extent of every label.
/// - `free`: Class-local ids of the free indices, in output order.
/// - `output`: Number of output elements and whether they are complex.
/// # Returns:
/// - `Values`: Sum of every group over the free indices.
pub(super) fn evaluate_graph(
    graph: &TableGraph,
    data: &[View<'_>],
    extent: &[usize],
    free: &[u16],
    output: (usize, bool),
) -> Values {
    let (size, complex) = output;
    let mut wide = [0usize; LABELS];
    wide[..extent.len()].copy_from_slice(extent);
    let slots = graph
        .uses
        .iter()
        .map(|&n| Slot {
            held: Mutex::new(None),
            left: AtomicU32::new(n),
        })
        .collect::<Vec<_>>();
    let next = AtomicUsize::new(0);

    (0..rayon::current_num_threads())
        .into_par_iter()
        .map(|_| {
            let mut ws = Workspace::new();
            let mut out = Values::zeros(size, complex);
            loop {
                let g = next.fetch_add(1, Ordering::Relaxed);
                let Some(group) = graph.groups.get(g) else {
                    break;
                };
                evaluate_group(
                    group,
                    graph,
                    (data, extent, &wide),
                    free,
                    &slots,
                    &mut out,
                    &mut ws,
                );
            }
            out
        })
        .reduce(
            || Values::zeros(size, complex),
            |mut a, b| {
                a.add_assign(b);
                a
            },
        )
}
