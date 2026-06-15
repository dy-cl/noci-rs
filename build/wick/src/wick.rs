// wick.rs

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use itertools::Itertools;
use num_rational::Ratio;
use num_traits::{One, Zero};
use rayon::prelude::*;
use smallvec::SmallVec;

pub use crate::canonical::canon;
use crate::spinsum;
use crate::ir::{Delta, Expr, Op, OpKind, Product, Rational, Space, Spin, Tensor, TensorKind, Term};

type Rat = Ratio<i64>;
type Id = u32;
type ProjKey = (usize, u8, u8);

static PROJ: OnceLock<BTreeMap<ProjKey, Vec<(Vec<usize>, Rat)>>> = OnceLock::new();

/// One numeric Wick product row.
#[derive(Clone, Debug)]
struct Row {
    /// Exact rational coefficient.
    c: Rat,
    /// Delta factor ids.
    d: SmallVec<[Id; 4]>,
    /// Tensor factor ids.
    t: SmallVec<[Id; 4]>,
    /// Wick-block GNO-group masks.
    e: SmallVec<[u64; 4]>,
}

/// One possible Wick block in an operator string.
#[derive(Clone, Debug)]
struct Block {
    /// Operator-position bit mask covered by this block.
    m: u64,
    /// GNO-group bit mask touched by this block.
    g: u64,
    /// Numeric values of this block.
    v: Vec<Row>,
}

/// Interned scalar factors.
#[derive(Clone, Debug, Default)]
struct Store {
    /// Delta ids.
    ds: BTreeMap<Delta, Id>,
    /// Tensor ids.
    ts: BTreeMap<Tensor, Id>,
    /// Delta factors by id.
    d: Vec<Delta>,
    /// Tensor factors by id.
    t: Vec<Tensor>,
}

impl Store {
    /// Intern one delta factor.
    /// # Arguments:
    /// - `x`: Delta factor.
    /// # Returns:
    /// - `Id`: One-based factor id.
    fn delta(&mut self, x: Delta) -> Id {
        if let Some(&id) = self.ds.get(&x) {
            return id;
        }

        let id = self.d.len() as Id + 1;
        self.ds.insert(x, id);
        self.d.push(x);
        id
    }

    /// Intern one tensor factor.
    /// # Arguments:
    /// - `x`: Tensor factor.
    /// # Returns:
    /// - `Id`: One-based factor id.
    fn tensor(&mut self, x: Tensor) -> Id {
        if let Some(&id) = self.ts.get(&x) {
            return id;
        }

        let id = self.t.len() as Id + 1;
        self.ts.insert(x.clone(), id);
        self.t.push(x);
        id
    }
}

/// Final numeric accumulator.
type Acc = BTreeMap<(Vec<Id>, Vec<Id>), Rat>;

/// Expand spin-free GNO groups into spin-orbital strings.
/// # Arguments:
/// - `p`: Product of spin-free GNO groups.
/// # Returns:
/// - `Vec<Vec<Op>>`: Spin-orbital operator strings.
fn spin(p: &Product) -> Vec<Vec<Op>> {
    let mut out = vec![Vec::new()];

    for g in &p.groups {
        let mut next = Vec::new();

        for a in &out {
            for b in &g.strings {
                let mut x = a.clone();
                x.extend_from_slice(b);
                next.push(x);
            }
        }

        out = next;
    }

    out
}

/// Build all nonzero non-internal Wick blocks for one spin string.
/// # Arguments:
/// - `ops`: Spin-orbital operator string.
/// - `s`: Factor store.
/// # Returns:
/// - `Vec<Block>`: Allowed Wick blocks.
fn blocks(ops: &[Op], s: &mut Store) -> Vec<Block> {
    let mut out = Vec::new();

    for b in frozen(ops).into_iter().chain(active(ops)) {
        let xs = b.iter().map(|&i| ops[i]).collect::<Vec<_>>();
        let v = val(&xs, s);

        if !v.is_empty() {
            out.push(Block { m: mask(&b), g: gmask(&b, ops), v });
        }
    }

    out
}

/// Enumerate core and virtual pair contractions.
/// # Arguments:
/// - `ops`: Spin-orbital operator string.
/// # Returns:
/// - `Vec<Vec<usize>>`: Operator-position blocks.
fn frozen(ops: &[Op]) -> Vec<Vec<usize>> {
    let mut out = Vec::new();

    for (i, j) in (0..ops.len()).tuple_combinations() {
        let a = ops[i];
        let b = ops[j];

        if a.group == b.group || a.spin != b.spin || a.idx.space != b.idx.space {
            continue;
        }

        let core = a.idx.space == Space::Core && a.kind == OpKind::Create && b.kind == OpKind::Annihilate;
        let virt = a.idx.space == Space::Virtual && a.kind == OpKind::Annihilate && b.kind == OpKind::Create;

        if core || virt {
            out.push(vec![i, j]);
        }
    }

    out
}

/// Enumerate active cumulant blocks up to rank four.
/// # Arguments:
/// - `ops`: Spin-orbital operator string.
/// # Returns:
/// - `Vec<Vec<usize>>`: Operator-position blocks.
fn active(ops: &[Op]) -> Vec<Vec<usize>> {
    let cs = ops.iter().enumerate()
        .filter(|(_, o)| o.idx.space == Space::Active && o.kind == OpKind::Create)
        .map(|(i, _)| i)
        .collect::<Vec<_>>();

    let as_ = ops.iter().enumerate()
        .filter(|(_, o)| o.idx.space == Space::Active && o.kind == OpKind::Annihilate)
        .map(|(i, _)| i)
        .collect::<Vec<_>>();

    let mut out = Vec::new();

    for r in 1..=cs.len().min(as_.len()).min(4) {
        for c in cs.iter().copied().combinations(r) {
            for a in as_.iter().copied().combinations(r) {
                let mut b = c.clone();
                b.extend(a.iter().copied());
                b.sort_unstable();

                let groups = b.iter().map(|&i| ops[i].group).collect::<BTreeSet<_>>();
                if groups.len() == 1 {
                    continue;
                }

                let ca = c.iter().filter(|&&i| ops[i].spin == Spin::Alpha).count();
                let aa = a.iter().filter(|&&i| ops[i].spin == Spin::Alpha).count();

                if ca == aa {
                    out.push(b);
                }
            }
        }
    }

    out
}

/// Evaluate one Wick block.
/// # Arguments:
/// - `ops`: Operators in the block.
/// - `s`: Factor store.
/// # Returns:
/// - `Vec<Row>`: Numeric value of the block.
fn val(ops: &[Op], s: &mut Store) -> Vec<Row> {
    if ops.len() == 2 {
        let a = ops[0];
        let b = ops[1];

        if a.spin != b.spin || a.idx.space != b.idx.space {
            return Vec::new();
        }

        if a.idx.space == Space::Core && a.kind == OpKind::Create && b.kind == OpKind::Annihilate {
            return vec![row(Rat::one(), &[s.delta(Delta { left: a.idx, right: b.idx })], &[])];
        }
        
        if a.idx.space == Space::Virtual && a.kind == OpKind::Annihilate && b.kind == OpKind::Create {
            return vec![row(Rat::one(), &[s.delta(Delta { left: b.idx, right: a.idx })], &[])];
        }

        if a.idx.space == Space::Active && a.kind == OpKind::Create && b.kind == OpKind::Annihilate {
            return vec![row(Rat::new(1, 2), &[], &[s.tensor(Tensor { kind: TensorKind::Gamma1, upper: vec![a.idx], lower: vec![b.idx] })])];
        }

        if a.idx.space == Space::Active && a.kind == OpKind::Annihilate && b.kind == OpKind::Create {
            return vec![row(Rat::new(1, 2), &[], &[s.tensor(Tensor { kind: TensorKind::Theta, upper: vec![b.idx], lower: vec![a.idx] })])];
        }

        return Vec::new();
    }

    if !ops.iter().all(|o| o.idx.space == Space::Active) {
        return Vec::new();
    }

    let Some((sign, cs, as_)) = norm(ops) else {
        return Vec::new();
    };

    let r = cs.len();

    if r == 0 || r != as_.len() || r > 4 {
        return Vec::new();
    }

    let upper = cs.iter().map(|o| o.idx).collect::<Vec<_>>();
    let lower = as_.iter().rev().map(|o| o.idx).collect::<Vec<_>>();
    let us = cs.iter().map(|o| o.spin).collect::<Vec<_>>();
    let ls = as_.iter().rev().map(|o| o.spin).collect::<Vec<_>>();
    let kind = match r {
        2 => TensorKind::Lambda2,
        3 => TensorKind::Lambda3,
        4 => TensorKind::Lambda4,
        _ => return Vec::new(),
    };

    let mut out = Vec::new();

    for (p, mut c) in proj(r, &us, &ls) {
        if sign < 0 {
            c = -c;
        }

        let lower_p = p.iter().map(|&i| lower[i]).collect::<Vec<_>>();
        let id = s.tensor(Tensor { kind, upper: upper.clone(), lower: lower_p });
        out.push(row(c, &[], &[id]));
    }

    out
}

/// Project one spin-orbital active cumulant into spin-free cumulants.
/// # Arguments:
/// - `r`: Cumulant rank.
/// - `us`: Upper spin labels.
/// - `ls`: Lower spin labels.
/// # Returns:
/// - `Vec<(Vec<usize>, Rat)>`: Lower-index permutations and coefficients.
fn proj(r: usize, us: &[Spin], ls: &[Spin]) -> Vec<(Vec<usize>, Rat)> {
    let key = (r, sbits(us), sbits(ls));
    PROJ.get_or_init(ptab)[&key].clone()
}

/// Build all small-rank spin projections.
/// # Arguments:
/// - None.
/// # Returns:
/// - `BTreeMap<ProjKey, Vec<(Vec<usize>, Rat)>>`: Projection table.
fn ptab() -> BTreeMap<ProjKey, Vec<(Vec<usize>, Rat)>> {
    let mut out = BTreeMap::new();

    for r in 1..=4 {
        for us in 0..(1u8 << r) {
            for ls in 0..(1u8 << r) {
                out.insert((r, us, ls), pval(r, us, ls));
            }
        }
    }

    out
}

/// Build one small-rank spin projection.
/// # Arguments:
/// - `r`: Cumulant rank.
/// - `us`: Upper spin bits.
/// - `ls`: Lower spin bits.
/// # Returns:
/// - `Vec<(Vec<usize>, Rat)>`: Lower-index permutations and coefficients.
fn pval(r: usize, us: u8, ls: u8) -> Vec<(Vec<usize>, Rat)> {
    let (ps, g) = spinsum::data(r).expect("Cached spin projection rank missing.");

    let b = ps.iter()
        .map(|p| {
            if (0..r).all(|i| bit(us, i) == bit(ls, p[i])) {
                Rat::from_integer(spinsum::sgn(p))
            } else {
                Rat::zero()
            }
        })
        .collect::<Vec<_>>();

    let x = spinsum::solve(g.to_vec(), b).expect("Inconsistent spin projection.");

    ps.iter().cloned().zip(x).filter(|(_, c)| !c.is_zero()).collect()
}

/// Return one spin bit.
/// # Arguments:
/// - `x`: Spin bit field.
/// - `i`: Bit index.
/// # Returns:
/// - `u8`: Spin bit.
fn bit(x: u8, i: usize) -> u8 {
    (x >> i) & 1
}

/// Encode spin labels into a compact cache key.
/// # Arguments:
/// - `xs`: Spin labels.
/// # Returns:
/// - `u8`: One bit per spin label.
fn sbits(xs: &[Spin]) -> u8 {
    xs.iter().enumerate().fold(0, |bits, (i, s)| {
        let bit = match s {
            Spin::Alpha => 0,
            Spin::Beta => 1,
        };

        bits | (bit << i)
    })
}

/// Recursively enumerate exact-cover suffixes of one spin string.
/// # Arguments:
/// - `left`: Remaining operator-position mask.
/// - `bs`: Wick blocks.
/// - `by_pos`: Block ids containing each position.
/// - `connected`: Whether to discard disconnected contractions.
/// - `memo`: Exact-cover suffix cache.
/// # Returns:
/// - `Vec<Row>`: Numeric suffix rows.
fn walk(left: u64, bs: &[Block], by_pos: &[Vec<usize>], connected: bool, memo: &mut BTreeMap<u64, Vec<Row>>) -> Vec<Row> {
    if left == 0 {
        return vec![Row { c: Rat::one(), d: SmallVec::new(), t: SmallVec::new(), e: SmallVec::new() }];
    }

    if let Some(x) = memo.get(&left) {
        return x.clone();
    }

    let Some(i) = pick(left, bs, by_pos) else {
        return Vec::new();
    };
    let mut out = Vec::new();

    for &bi in &by_pos[i] {
        let b = &bs[bi];

        if b.m & left != b.m {
            continue;
        }

        let rest = left & !b.m;
        let sgn = cross(b.m, rest);
        let tail = walk(rest, bs, by_pos, connected, memo);

        for v in &b.v {
            for w in &tail {
                let mut row = Row {
                    c: v.c.clone() * w.c.clone(),
                    d: v.d.clone(),
                    t: v.t.clone(),
                    e: SmallVec::new(),
                };

                if sgn < 0 {
                    row.c = -row.c;
                }

                row.d.extend_from_slice(&w.d);
                row.t.extend_from_slice(&w.t);

                if connected {
                    row.e.push(b.g);
                    row.e.extend_from_slice(&w.e);
                }

                out.push(row);
            }
        }
    }

    memo.insert(left, out.clone());
    out
}

/// Pick remaining position with fewest viable Wick blocks.
/// # Arguments:
/// - `left`: Remaining operator-position mask.
/// - `bs`: Wick blocks.
/// - `by_pos`: Block ids containing each position.
/// # Returns:
/// - `Option<usize>`: Best operator-position index.
fn pick(mut left: u64, bs: &[Block], by_pos: &[Vec<usize>]) -> Option<usize> {
    let rem = left;
    let mut best = None;
    let mut min = usize::MAX;

    while left != 0 {
        let b = left & left.wrapping_neg();
        let i = b.trailing_zeros() as usize;
        let n = by_pos[i].iter().filter(|&&bi| bs[bi].m & rem == bs[bi].m).count();

        if n == 0 {
            return None;
        }

        if n < min {
            best = Some(i);
            min = n;

            if n == 1 {
                break;
            }
        }

        left ^= b;
    }

    best
}

/// Accumulate one numeric row.
/// # Arguments:
/// - `acc`: Final accumulator.
/// - `r`: Numeric row.
/// # Returns:
/// - `()`: Mutates `acc`.
fn add(acc: &mut Acc, mut r: Row) {
    if r.c.is_zero() {
        return;
    }

    r.d.sort_unstable();
    r.t.sort_unstable();

    let key = (r.d.to_vec(), r.t.to_vec());
    let x = acc.entry(key.clone()).or_insert_with(Rat::zero);
    *x += r.c;

    if x.is_zero() {
        acc.remove(&key);
    }
}

/// Decode numeric rows to symbolic terms.
/// # Arguments:
/// - `s`: Factor store.
/// - `acc`: Final numeric accumulator.
/// # Returns:
/// - `Expr`: Symbolic expression.
fn out(s: &Store, acc: Acc) -> Expr {
    acc.into_iter()
        .filter(|(_, c)| !c.is_zero())
        .map(|((ds, ts), c)| {
            let deltas = ds.iter().map(|&i| s.d[(i - 1) as usize]).collect();
            let tensors = ts.iter().map(|&i| s.t[(i - 1) as usize].clone()).collect();

            Term {
                coeff: Rational { num: *c.numer(), den: *c.denom() },
                deltas,
                tensors,
            }
        })
        .collect()
}

/// Build one numeric row.
/// # Arguments:
/// - `c`: Coefficient.
/// - `d`: Delta ids.
/// - `t`: Tensor ids.
/// # Returns:
/// - `Row`: Numeric row.
fn row(c: Rat, d: &[Id], t: &[Id]) -> Row {
    Row { c, d: d.iter().copied().collect(), t: t.iter().copied().collect(), e: SmallVec::new() }
}

/// Return bit positions in a mask.
/// # Arguments:
/// - `m`: Bit mask.
/// # Returns:
/// - `Vec<usize>`: Set-bit positions.
fn bits(mut m: u64) -> Vec<usize> {
    let mut out = Vec::new();

    while m != 0 {
        let b = m & m.wrapping_neg();
        out.push(b.trailing_zeros() as usize);
        m ^= b;
    }

    out
}

/// Return the mask for selected positions.
/// # Arguments:
/// - `xs`: Positions.
/// # Returns:
/// - `u64`: Bit mask.
fn mask(xs: &[usize]) -> u64 {
    xs.iter().fold(0, |m, &i| m | (1u64 << i))
}

/// Build the GNO-group mask touched by a Wick block.
/// # Arguments:
/// - `xs`: Operator positions in the block.
/// - `ops`: Spin-orbital operator string.
/// # Returns:
/// - `u64`: Bit mask of GNO group ids.
fn gmask(xs: &[usize], ops: &[Op]) -> u64 {
    xs.iter().fold(0, |m, &i| m | (1u64 << ops[i].group))
}

/// Check whether a set of Wick-block group masks forms a connected graph.
/// # Arguments:
/// - `want`: Required group mask.
/// - `edges`: Wick-block group masks.
/// # Returns:
/// - `bool`: Whether all required groups are connected.
fn conn(want: u64, edges: &[u64]) -> bool {
    if want.count_ones() <= 1 {
        return true;
    }

    let mut seen = 1u64;

    loop {
        let old = seen;

        for &e in edges {
            if e & seen != 0 {
                seen |= e;
            }
        }

        if seen == old {
            break;
        }
    }

    seen == want
}

/// Return the Wick sign from moving a block before the remaining operators.
/// # Arguments:
/// - `m`: Selected block mask.
/// - `rest`: Remaining operator mask.
/// # Returns:
/// - `i64`: Fermionic sign.
fn cross(mut m: u64, rest: u64) -> i64 {
    let mut p = 0;

    while m != 0 {
        let b = m & m.wrapping_neg();
        let i = b.trailing_zeros();
        p ^= (rest & ((1u64 << i) - 1)).count_ones() & 1;
        m ^= b;
    }

    if p == 0 { 1 } else { -1 }
}

/// Normal-order active operators.
/// # Arguments:
/// - `ops`: Active operators.
/// # Returns:
/// - `Option<(i64, Vec<Op>, Vec<Op>)>`: Sign, creators, annihilators.
fn norm(ops: &[Op]) -> Option<(i64, Vec<Op>, Vec<Op>)> {
    let cs = ops.iter().copied().filter(|o| o.kind == OpKind::Create).collect::<Vec<_>>();
    let as_ = ops.iter().copied().filter(|o| o.kind == OpKind::Annihilate).collect::<Vec<_>>();

    if cs.len() + as_.len() != ops.len() {
        return None;
    }

    let mut inv = 0;

    for i in 0..ops.len() {
        if ops[i].kind != OpKind::Annihilate {
            continue;
        }

        inv += ops[i + 1..].iter().filter(|o| o.kind == OpKind::Create).count();
    }

    Some((if inv % 2 == 0 { 1 } else { -1 }, cs, as_))
}

/// Evaluate a spin-free product with all Wick contractions retained.
/// # Arguments:
/// - `p`: Spin-free product.
/// # Returns:
/// - `Expr`: Canonical contracted expression.
pub fn eval(p: &Product) -> Expr {
    eval0(p, false)
}

/// Evaluate a spin-free product keeping only connected contractions.
/// # Arguments:
/// - `p`: Spin-free product.
/// # Returns:
/// - `Expr`: Canonical connected contracted expression.
pub fn evalc(p: &Product) -> Expr {
    eval0(p, true)
}

/// Evaluate a spin-free product.
/// # Arguments:
/// - `p`: Spin-free product.
/// - `connected`: Whether to discard disconnected contractions.
/// # Returns:
/// - `Expr`: Canonical contracted expression.
fn eval0(p: &Product, connected: bool) -> Expr {
    let want = (1u64 << p.groups.len()) - 1;

    let e = spin(p)
        .into_par_iter()
        .flat_map(|ops| eval1(&ops, connected, want))
        .collect();

    canon(e)
}

/// Evaluate one spin-orbital string.
/// # Arguments:
/// - `ops`: Spin-orbital operator string.
/// - `connected`: Whether to discard disconnected contractions.
/// - `want`: Required GNO-group mask.
/// # Returns:
/// - `Expr`: Contracted expression.
fn eval1(ops: &[Op], connected: bool, want: u64) -> Expr {
    let mut s = Store::default();
    let mut acc = Acc::new();
    let bs = blocks(ops, &mut s);
    let mut by_pos = vec![Vec::<usize>::new(); ops.len()];

    for (i, b) in bs.iter().enumerate() {
        for pos in bits(b.m) {
            by_pos[pos].push(i);
        }
    }

    let full = (1u64 << ops.len()) - 1;
    let mut memo = BTreeMap::new();

    for row in walk(full, &bs, &by_pos, connected, &mut memo) {
        if !connected || conn(want, &row.e) {
            add(&mut acc, row);
        }
    }

    out(&s, acc)
}
