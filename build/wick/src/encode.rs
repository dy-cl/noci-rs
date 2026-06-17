// encode.rs

use std::collections::{BTreeMap, BTreeSet};

use num_rational::Ratio;
use num_traits::Zero;

use crate::ir::{Expr, Idx, Space, Tensor, TensorKind, Term};
use crate::schema::{GeneratedTerm, OverlapBlockTerms, OverlapTermSet, ResidualClassTerms, ResidualTermSet, TensorFactor};
use crate::specs::{idx, BlockSpec, ExcSpec, BLOCKS, EXCS};

type Rat = Ratio<i64>;

/// Canonical encoded symbolic index.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
enum IKey {
    /// Free external index.
    Free(u16),
    /// Dummy summed index, labelled by space and local dummy number.
    Dummy(u8, u16),
}

/// Canonical encoded tensor factor.
#[derive(Clone, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
struct FKey {
    /// Tensor kind.
    kind: u8,
    /// Upper index keys.
    upper: Vec<IKey>,
    /// Lower index keys.
    lower: Vec<IKey>,
}

/// Canonical encoded term without its coefficient.
#[derive(Clone, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
struct TKey {
    /// Delta factors.
    deltas: Vec<[IKey; 2]>,
    /// Tensor factors.
    tensors: Vec<FKey>,
}

/// Global canonical accumulator for one generated class or block.
#[derive(Clone, Debug)]
struct Acc {
    /// Fixed free indices.
    free: Vec<Idx>,
    /// Free-index map.
    free_ids: BTreeMap<Idx, u16>,
    /// Globally combined terms.
    terms: BTreeMap<TKey, Rat>,
}

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
        TensorKind::Fock => 2,
        TensorKind::ERI => 3,
        TensorKind::Lambda2 => 4,
        TensorKind::Lambda3 => 5,
        TensorKind::Lambda4 => 6,
        TensorKind::T1 => 8,
        TensorKind::T2 => 9,
    }
}

/// Return the default generated space table.
/// # Arguments:
/// - None.
/// # Returns:
/// - `BTreeMap<String, u8>`: Space-name map.
fn space_kinds() -> BTreeMap<String, u8> {
    let mut out = BTreeMap::new();

    out.insert("core".to_string(), 0);
    out.insert("active".to_string(), 1);
    out.insert("virtual".to_string(), 2);

    out
}

/// Return the default generated tensor table.
/// # Arguments:
/// - None.
/// # Returns:
/// - `BTreeMap<String, u8>`: Tensor-name map.
fn tensor_kinds() -> BTreeMap<String, u8> {
    let mut out = BTreeMap::new();

    out.insert("Gamma1".to_string(), 0);
    out.insert("Theta".to_string(), 1);
    out.insert("f".to_string(), 2);
    out.insert("g".to_string(), 3);
    out.insert("Lambda2".to_string(), 4);
    out.insert("Lambda3".to_string(), 5);
    out.insert("Lambda4".to_string(), 6);
    out.insert("t1".to_string(), 8);
    out.insert("t2".to_string(), 9);

    out
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

/// Return one encoded index from an index map.
/// # Arguments:
/// - `ids`: Index map.
/// - `x`: Symbolic index.
/// # Returns:
/// - `IKey`: Encoded index.
fn ikey(ids: &BTreeMap<Idx, IKey>, x: Idx) -> IKey {
    *ids.get(&x).unwrap_or_else(|| panic!("missing canonical id for index {}", x.name))
}

/// Build one encoded tensor factor from an index map.
/// # Arguments:
/// - `x`: Tensor factor.
/// - `ids`: Index map.
/// # Returns:
/// - `FKey`: Encoded tensor factor.
fn fkey(x: &Tensor, ids: &BTreeMap<Idx, IKey>) -> FKey {
    FKey {
        kind: tk(x.kind),
        upper: x.upper.iter().map(|&i| ikey(ids, i)).collect(),
        lower: x.lower.iter().map(|&i| ikey(ids, i)).collect(),
    }
}

/// Insert all fixed free-index mappings.
/// # Arguments:
/// - `out`: Index map.
/// - `free`: Free-index map.
/// # Returns:
/// - `()`: Mutates `out`.
fn insert_free(out: &mut BTreeMap<Idx, IKey>, free: &BTreeMap<Idx, u16>) {
    for (&x, &id) in free {
        out.insert(x, IKey::Free(id));
    }
}

/// Assign one dummy index if required.
/// # Arguments:
/// - `out`: Index map.
/// - `free`: Free-index map.
/// - `next`: Next dummy id per space.
/// - `x`: Symbolic index.
/// # Returns:
/// - `()`: Mutates `out`.
fn assign(out: &mut BTreeMap<Idx, IKey>, free: &BTreeMap<Idx, u16>, next: &mut [u16; 3], x: Idx) {
    if free.contains_key(&x) || out.contains_key(&x) {
        return;
    }

    let s = sp(x.space);
    let n = next[s as usize];

    next[s as usize] += 1;
    out.insert(x, IKey::Dummy(s, n));
}

/// Build an initial dummy-index map.
/// # Arguments:
/// - `t`: Term.
/// - `free`: Free-index map.
/// # Returns:
/// - `BTreeMap<Idx, IKey>`: Initial index map.
fn initial(t: &Term, free: &BTreeMap<Idx, u16>) -> BTreeMap<Idx, IKey> {
    let mut out = BTreeMap::new();
    let mut next = [0u16; 3];

    insert_free(&mut out, free);

    for x in inds(t) {
        assign(&mut out, free, &mut next, x);
    }

    out
}

/// Return raw deltas in canonical order under the current index map.
/// # Arguments:
/// - `t`: Term.
/// - `ids`: Current index map.
/// # Returns:
/// - `Vec<(Idx, Idx)>`: Raw delta pairs, oriented and sorted.
fn ordered_deltas(t: &Term, ids: &BTreeMap<Idx, IKey>) -> Vec<(Idx, Idx)> {
    let mut out = Vec::new();

    for d in &t.deltas {
        let mut l = d.left;
        let mut r = d.right;

        if ikey(ids, r) < ikey(ids, l) {
            std::mem::swap(&mut l, &mut r);
        }

        out.push((l, r));
    }

    out.sort_by_key(|&(l, r)| [ikey(ids, l), ikey(ids, r)]);
    out
}

/// Return raw tensors in canonical order under the current index map.
/// # Arguments:
/// - `t`: Term.
/// - `ids`: Current index map.
/// # Returns:
/// - `Vec<&Tensor>`: Tensor factors sorted by encoded representation.
fn ordered_tensors<'a>(t: &'a Term, ids: &BTreeMap<Idx, IKey>) -> Vec<&'a Tensor> {
    let mut out = t.tensors.iter().collect::<Vec<_>>();

    out.sort_by_key(|x| fkey(x, ids));
    out
}

/// Rebuild dummy labels from the current canonical factor order.
/// # Arguments:
/// - `t`: Term.
/// - `free`: Free-index map.
/// - `ids`: Current index map.
/// # Returns:
/// - `BTreeMap<Idx, IKey>`: Refined index map.
fn refine(t: &Term, free: &BTreeMap<Idx, u16>, ids: &BTreeMap<Idx, IKey>) -> BTreeMap<Idx, IKey> {
    let mut out = BTreeMap::new();
    let mut next = [0u16; 3];

    insert_free(&mut out, free);

    for (l, r) in ordered_deltas(t, ids) {
        assign(&mut out, free, &mut next, l);
        assign(&mut out, free, &mut next, r);
    }

    for x in ordered_tensors(t, ids) {
        for &i in &x.upper {
            assign(&mut out, free, &mut next, i);
        }

        for &i in &x.lower {
            assign(&mut out, free, &mut next, i);
        }
    }

    out
}

/// Canonicalise one term to an alpha-renamed key.
/// # Arguments:
/// - `t`: Term.
/// - `free`: Free-index map.
/// # Returns:
/// - `TKey`: Canonical term key without coefficient.
fn key(t: &Term, free: &BTreeMap<Idx, u16>) -> TKey {
    let mut ids = initial(t, free);

    for _ in 0..8 {
        let next = refine(t, free, &ids);

        if next == ids {
            break;
        }

        ids = next;
    }

    let deltas = ordered_deltas(t, &ids)
        .into_iter()
        .map(|(l, r)| [ikey(&ids, l), ikey(&ids, r)])
        .collect();

    let tensors = ordered_tensors(t, &ids)
        .into_iter()
        .map(|x| fkey(x, &ids))
        .collect();

    TKey { deltas, tensors }
}

/// Convert a symbolic coefficient to a rational.
/// # Arguments:
/// - `t`: Term.
/// # Returns:
/// - `Rat`: Rational coefficient.
fn coeff(t: &Term) -> Rat {
    Ratio::new(t.coeff.num, t.coeff.den)
}

/// Return all dummy labels used by one key.
/// # Arguments:
/// - `k`: Term key.
/// # Returns:
/// - `BTreeSet<(u8, u16)>`: Dummy labels as `(space, local_id)`.
fn dummies(k: &TKey) -> BTreeSet<(u8, u16)> {
    let mut out = BTreeSet::new();

    for d in &k.deltas {
        for &x in d {
            if let IKey::Dummy(s, n) = x {
                out.insert((s, n));
            }
        }
    }

    for f in &k.tensors {
        for &x in f.upper.iter().chain(f.lower.iter()) {
            if let IKey::Dummy(s, n) = x {
                out.insert((s, n));
            }
        }
    }

    out
}

/// Add one dummy loop id in first-occurrence order.
/// # Arguments:
/// - `out`: Loop id list.
/// - `ids`: Runtime dummy-id map.
/// - `x`: Encoded index.
/// # Returns:
/// - `()`: Mutates `out`.
fn push_loop(out: &mut Vec<u16>, ids: &BTreeMap<(u8, u16), u16>, x: IKey) {
    if let IKey::Dummy(s, n) = x {
        let id = ids[&(s, n)];

        if !out.contains(&id) {
            out.push(id);
        }
    }
}

/// Convert one encoded index to a runtime index id.
/// # Arguments:
/// - `x`: Encoded index.
/// - `ids`: Runtime dummy-id map.
/// # Returns:
/// - `u16`: Runtime class-local index id.
fn rid(x: IKey, ids: &BTreeMap<(u8, u16), u16>) -> u16 {
    match x {
        IKey::Free(i) => i,
        IKey::Dummy(s, n) => ids[&(s, n)],
    }
}

/// Build a generated dummy-index name.
/// # Arguments:
/// - `space`: Runtime space id.
/// - `n`: Dummy number.
/// # Returns:
/// - `String`: Generated dummy label.
fn dummy_name(space: u8, n: u16) -> String {
    match space {
        0 => format!("dc{n}"),
        1 => format!("da{n}"),
        2 => format!("dv{n}"),
        _ => panic!("unknown space id {space}"),
    }
}

impl Acc {
    /// Construct an empty global accumulator.
    /// # Arguments:
    /// - `free`: Fixed free indices.
    /// # Returns:
    /// - `Acc`: Empty accumulator.
    fn new(free: Vec<Idx>) -> Self {
        let free_ids = free.iter()
            .enumerate()
            .map(|(i, &x)| (x, i as u16))
            .collect();

        Self { free, free_ids, terms: BTreeMap::new() }
    }

    /// Add one symbolic expression to the global canonical accumulator.
    /// # Arguments:
    /// - `e`: Symbolic expression chunk.
    /// # Returns:
    /// - `()`: Mutates `self`.
    fn addexpr(&mut self, e: Expr) {
        for t in e {
            self.addterm(t);
        }
    }

    /// Add one symbolic term to the global canonical accumulator.
    /// # Arguments:
    /// - `t`: Symbolic term.
    /// # Returns:
    /// - `()`: Mutates `self`.
    fn addterm(&mut self, t: Term) {
        let k = key(&t, &self.free_ids);
        let c = coeff(&t);
        let entry = self.terms.entry(k.clone()).or_insert_with(Rat::zero);

        *entry += c;

        if entry.is_zero() {
            self.terms.remove(&k);
        }
    }

    /// Build the common runtime index table.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `(Vec<(String, u8)>, BTreeMap<(u8, u16), u16>)`: Runtime indices and dummy ids.
    fn table(&self) -> (Vec<(String, u8)>, BTreeMap<(u8, u16), u16>) {
        let mut indices = self.free.iter()
            .map(|x| (x.name.to_string(), sp(x.space)))
            .collect::<Vec<_>>();

        let mut seen = BTreeSet::new();

        for k in self.terms.keys() {
            seen.extend(dummies(k));
        }

        let mut ids = BTreeMap::new();

        for (s, n) in seen {
            let id = indices.len() as u16;

            ids.insert((s, n), id);
            indices.push((dummy_name(s, n), s));
        }

        (indices, ids)
    }

    /// Convert the accumulator to generated terms.
    /// # Arguments:
    /// - `ids`: Runtime dummy-id map.
    /// # Returns:
    /// - `Vec<GeneratedTerm>`: Runtime generated terms.
    fn terms(&self, ids: &BTreeMap<(u8, u16), u16>) -> Vec<GeneratedTerm> {
        let mut out = Vec::new();

        for (k, c) in &self.terms {
            if c.is_zero() {
                continue;
            }

            let mut loops = Vec::new();

            for d in &k.deltas {
                push_loop(&mut loops, ids, d[0]);
                push_loop(&mut loops, ids, d[1]);
            }

            for f in &k.tensors {
                for &x in f.upper.iter().chain(f.lower.iter()) {
                    push_loop(&mut loops, ids, x);
                }
            }

            out.push(GeneratedTerm(
                [*c.numer(), *c.denom()],
                loops,
                k.deltas.iter().map(|d| [rid(d[0], ids), rid(d[1], ids)]).collect(),
                k.tensors.iter().map(|f| {
                    TensorFactor(
                        f.kind,
                        f.upper.iter().map(|&x| rid(x, ids)).collect(),
                        f.lower.iter().map(|&x| rid(x, ids)).collect(),
                    )
                }).collect(),
            ));
        }

        out
    }

    /// Convert to one residual class table.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `ResidualClassTerms`: Runtime residual class terms.
    fn residual(self) -> ResidualClassTerms {
        let free = (0..self.free.len()).map(|i| i as u16).collect();
        let (indices, ids) = self.table();
        let terms = self.terms(&ids);

        ResidualClassTerms { indices, free, terms }
    }

    /// Convert to one overlap block table.
    /// # Arguments:
    /// - `b`: Block specification.
    /// # Returns:
    /// - `OverlapBlockTerms`: Runtime overlap block terms.
    fn overlap(self, b: BlockSpec) -> OverlapBlockTerms {
        let left_free = (0..b.lf.len()).map(|i| i as u16).collect::<Vec<_>>();
        let right_free = (b.lf.len()..b.lf.len() + b.rf.len()).map(|i| i as u16).collect::<Vec<_>>();
        let (indices, ids) = self.table();
        let terms = self.terms(&ids);

        OverlapBlockTerms {
            left: b.left.to_string(),
            right: b.right.to_string(),
            indices,
            left_free,
            right_free,
            terms,
        }
    }
}

/// Return residual free indices.
/// # Arguments:
/// - `x`: Excitation-class specification.
/// # Returns:
/// - `Vec<Idx>`: Free indices.
fn rfree(x: ExcSpec) -> Vec<Idx> {
    x.f.iter().map(|&name| idx(name)).collect()
}

/// Return overlap free indices.
/// # Arguments:
/// - `b`: Block specification.
/// # Returns:
/// - `Vec<Idx>`: Free indices.
fn bfree(b: BlockSpec) -> Vec<Idx> {
    b.lf.iter().chain(b.rf.iter()).map(|&name| idx(name)).collect()
}

/// Generate one compact metric block.
/// # Arguments:
/// - `b`: Block metadata.
/// # Returns:
/// - `OverlapBlockTerms`: Runtime block terms.
fn block_terms(b: BlockSpec) -> OverlapBlockTerms {
    let mut acc = Acc::new(bfree(b));

    acc.addexpr(crate::wick::eval(&crate::overlap::block(b.name)));
    acc.overlap(b)
}

/// Generate one compact residual class.
/// # Arguments:
/// - `order`: Residual order.
/// - `x`: Excitation-class specification.
/// # Returns:
/// - `ResidualClassTerms`: Runtime residual class terms.
fn residual_class(order: u8, x: ExcSpec) -> ResidualClassTerms {
    let mut acc = Acc::new(rfree(x));

    match order {
        0 => crate::residual::r0(x.name, |_, e| acc.addexpr(e)),
        1 => crate::residual::r1(x.name, |_, e| acc.addexpr(e)),
        2 => crate::residual::r2(x.name, |_, e| acc.addexpr(e)),
        _ => panic!("unsupported residual order {order}"),
    }

    acc.residual()
}

/// Generate compact metric terms.
/// # Arguments:
/// - None.
/// # Returns:
/// - `OverlapTermSet`: Complete metric term table.
pub fn overlap_terms() -> OverlapTermSet {
    OverlapTermSet {
        version: 1,
        space_kinds: space_kinds(),
        tensor_kinds: tensor_kinds(),
        blocks: BLOCKS.iter().map(|&b| (b.name.to_string(), block_terms(b))).collect(),
    }
}

/// Generate compact residual terms.
/// # Arguments:
/// - `order`: Residual order.
/// # Returns:
/// - `ResidualTermSet`: Complete residual term table.
pub fn residual_terms(order: u8) -> ResidualTermSet {
    ResidualTermSet {
        version: 1,
        order,
        space_kinds: space_kinds(),
        tensor_kinds: tensor_kinds(),
        classes: EXCS.iter().map(|&x| (x.name.to_string(), residual_class(order, x))).collect(),
    }
}
