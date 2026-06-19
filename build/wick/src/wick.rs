// wick.rs

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use itertools::Itertools;
use num_rational::Ratio;
use num_traits::{One, Zero};
use rayon::prelude::*;
use smallvec::SmallVec;

pub use crate::canonical::canon;
use crate::ir::{Delta, Expr, Op, OpKind, Product, Rational, Space, Spin, Tensor, TensorKind, Term};
use crate::spinsum;

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
        crate::time_call!(crate::timers::wick::add_store_delta, {
            if let Some(&id) = self.ds.get(&x) {
                return id;
            }

            let id = self.d.len() as Id + 1;
            self.ds.insert(x, id);
            self.d.push(x);
            id
        })
    }

    /// Intern one tensor factor.
    /// # Arguments:
    /// - `x`: Tensor factor.
    /// # Returns:
    /// - `Id`: One-based factor id.
    fn tensor(&mut self, x: Tensor) -> Id {
        crate::time_call!(crate::timers::wick::add_store_tensor, {
            if let Some(&id) = self.ts.get(&x) {
                return id;
            }

            let id = self.t.len() as Id + 1;
            self.ts.insert(x.clone(), id);
            self.t.push(x);
            id
        })
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
    crate::time_call!(crate::timers::wick::add_spin, {
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
    })
}

/// Build all nonzero non-internal Wick blocks for one spin string.
/// # Arguments:
/// - `ops`: Spin-orbital operator string.
/// - `s`: Factor store.
/// # Returns:
/// - `Vec<Block>`: Allowed Wick blocks.
fn blocks(ops: &[Op], s: &mut Store) -> Vec<Block> {
    crate::time_call!(crate::timers::wick::add_blocks, {
        let mut out = Vec::new();

        for b in frozen(ops).into_iter().chain(active(ops)) {
            let xs = b.iter().map(|&i| ops[i]).collect::<Vec<_>>();
            let v = val(&xs, s);

            if !v.is_empty() {
                out.push(Block {
                    m: mask(&b),
                    g: gmask(&b, ops),
                    v,
                });
            }
        }

        out
    })
}

/// Enumerate core and virtual pair contractions.
/// # Arguments:
/// - `ops`: Spin-orbital operator string.
/// # Returns:
/// - `Vec<Vec<usize>>`: Operator-position blocks.
fn frozen(ops: &[Op]) -> Vec<Vec<usize>> {
    crate::time_call!(crate::timers::wick::add_frozen, {
        let mut out = Vec::new();

        for (i, j) in (0..ops.len()).tuple_combinations() {
            let a = ops[i];
            let b = ops[j];

            if a.group == b.group || a.spin != b.spin || a.idx.space != b.idx.space {
                continue;
            }

            let core = a.idx.space == Space::Core
                && a.kind == OpKind::Create
                && b.kind == OpKind::Annihilate;
            let virt = a.idx.space == Space::Virtual
                && a.kind == OpKind::Annihilate
                && b.kind == OpKind::Create;

            if core || virt {
                out.push(vec![i, j]);
            }
        }

        out
    })
}

/// Enumerate active cumulant blocks up to rank four.
/// # Arguments:
/// - `ops`: Spin-orbital operator string.
/// # Returns:
/// - `Vec<Vec<usize>>`: Operator-position blocks.
fn active(ops: &[Op]) -> Vec<Vec<usize>> {
    crate::time_call!(crate::timers::wick::add_active, {
        let cs = ops
            .iter()
            .enumerate()
            .filter(|(_, o)| o.idx.space == Space::Active && o.kind == OpKind::Create)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();

        let as_ = ops
            .iter()
            .enumerate()
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
    })
}

/// Evaluate one Wick block.
/// # Arguments:
/// - `ops`: Operators in the block.
/// - `s`: Factor store.
/// # Returns:
/// - `Vec<Row>`: Numeric value of the block.
fn val(ops: &[Op], s: &mut Store) -> Vec<Row> {
    crate::time_call!(crate::timers::wick::add_val, {
        if ops.len() == 2 {
            let a = ops[0];
            let b = ops[1];

            if a.spin != b.spin || a.idx.space != b.idx.space {
                return Vec::new();
            }

            if a.idx.space == Space::Core
                && a.kind == OpKind::Create
                && b.kind == OpKind::Annihilate
            {
                return vec![row(
                    Rat::one(),
                    &[s.delta(Delta {
                        left: a.idx,
                        right: b.idx,
                    })],
                    &[],
                )];
            }

            if a.idx.space == Space::Virtual
                && a.kind == OpKind::Annihilate
                && b.kind == OpKind::Create
            {
                return vec![row(
                    Rat::one(),
                    &[s.delta(Delta {
                        left: b.idx,
                        right: a.idx,
                    })],
                    &[],
                )];
            }

            if a.idx.space == Space::Active
                && a.kind == OpKind::Create
                && b.kind == OpKind::Annihilate
            {
                return vec![row(
                    Rat::new(1, 2),
                    &[],
                    &[s.tensor(Tensor {
                        kind: TensorKind::Gamma1,
                        upper: vec![a.idx],
                        lower: vec![b.idx],
                    })],
                )];
            }

            if a.idx.space == Space::Active
                && a.kind == OpKind::Annihilate
                && b.kind == OpKind::Create
            {
                return vec![row(
                    Rat::new(1, 2),
                    &[],
                    &[s.tensor(Tensor {
                        kind: TensorKind::Theta,
                        upper: vec![b.idx],
                        lower: vec![a.idx],
                    })],
                )];
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
            let id = s.tensor(Tensor {
                kind,
                upper: upper.clone(),
                lower: lower_p,
            });

            out.push(row(c, &[], &[id]));
        }

        out
    })
}

/// Project one spin-orbital active cumulant into spin-free cumulants.
/// # Arguments:
/// - `r`: Cumulant rank.
/// - `us`: Upper spin labels.
/// - `ls`: Lower spin labels.
/// # Returns:
/// - `Vec<(Vec<usize>, Rat)>`: Lower-index permutations and coefficients.
fn proj(r: usize, us: &[Spin], ls: &[Spin]) -> Vec<(Vec<usize>, Rat)> {
    crate::time_call!(crate::timers::wick::add_proj, {
        let key = (r, sbits(us), sbits(ls));
        PROJ.get_or_init(ptab)[&key].clone()
    })
}

/// Build all small-rank spin projections.
/// # Arguments:
/// - None.
/// # Returns:
/// - `BTreeMap<ProjKey, Vec<(Vec<usize>, Rat)>>`: Projection table.
fn ptab() -> BTreeMap<ProjKey, Vec<(Vec<usize>, Rat)>> {
    crate::time_call!(crate::timers::wick::add_ptab, {
        let mut out = BTreeMap::new();

        for r in 1..=4 {
            for us in 0..(1u8 << r) {
                for ls in 0..(1u8 << r) {
                    out.insert((r, us, ls), pval(r, us, ls));
                }
            }
        }

        out
    })
}

/// Build one small-rank spin projection.
/// # Arguments:
/// - `r`: Cumulant rank.
/// - `us`: Upper spin bits.
/// - `ls`: Lower spin bits.
/// # Returns:
/// - `Vec<(Vec<usize>, Rat)>`: Lower-index permutations and coefficients.
fn pval(r: usize, us: u8, ls: u8) -> Vec<(Vec<usize>, Rat)> {
    crate::time_call!(crate::timers::wick::add_pval, {
        let (ps, g) = spinsum::data(r).expect("Cached spin projection rank missing.");

        let b = ps
            .iter()
            .map(|p| {
                if (0..r).all(|i| bit(us, i) == bit(ls, p[i])) {
                    Rat::from_integer(spinsum::sgn(p))
                } else {
                    Rat::zero()
                }
            })
            .collect::<Vec<_>>();

        let x = spinsum::solve(g.to_vec(), b).expect("Inconsistent spin projection.");

        ps.iter()
            .cloned()
            .zip(x)
            .filter(|(_, c)| !c.is_zero())
            .collect()
    })
}

/// Return one spin bit.
/// # Arguments:
/// - `x`: Spin bit field.
/// - `i`: Bit index.
/// # Returns:
/// - `u8`: Spin bit.
fn bit(x: u8, i: usize) -> u8 {
    crate::time_call!(crate::timers::wick::add_bit, { (x >> i) & 1 })
}

/// Encode spin labels into a compact cache key.
/// # Arguments:
/// - `xs`: Spin labels.
/// # Returns:
/// - `u8`: One bit per spin label.
fn sbits(xs: &[Spin]) -> u8 {
    crate::time_call!(crate::timers::wick::add_sbits, {
        xs.iter().enumerate().fold(0, |bits, (i, s)| {
            let bit = match s {
                Spin::Alpha => 0,
                Spin::Beta => 1,
            };

            bits | (bit << i)
        })
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
fn walk(
    left: u64,
    bs: &[Block],
    by_pos: &[Vec<usize>],
    connected: bool,
    memo: &mut BTreeMap<u64, Vec<Row>>,
) -> Vec<Row> {
    crate::time_call!(crate::timers::wick::add_walk, {
        walk_inner(left, bs, by_pos, connected, memo)
    })
}

/// Recursively enumerate exact-cover suffixes without starting nested timers.
fn walk_inner(
    left: u64,
    bs: &[Block],
    by_pos: &[Vec<usize>],
    connected: bool,
    memo: &mut BTreeMap<u64, Vec<Row>>,
) -> Vec<Row> {
    if left == 0 {
        return vec![Row {
            c: Rat::one(),
            d: SmallVec::new(),
            t: SmallVec::new(),
            e: SmallVec::new(),
        }];
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
        let tail = walk_inner(rest, bs, by_pos, connected, memo);

        for v in &b.v {
            for w in &tail {
                let mut row = Row {
                    c: v.c * w.c,
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

/// Pick the remaining position with the fewest viable Wick blocks.
/// # Arguments:
/// - `left`: Remaining operator-position mask.
/// - `bs`: Wick blocks.
/// - `by_pos`: Block ids containing each position.
/// # Returns:
/// - `Option<usize>`: Best operator-position index.
fn pick(mut left: u64, bs: &[Block], by_pos: &[Vec<usize>]) -> Option<usize> {
    crate::time_call!(crate::timers::wick::add_pick, {
        let rem = left;
        let mut best = None;
        let mut min = usize::MAX;

        while left != 0 {
            let b = left & left.wrapping_neg();
            let i = b.trailing_zeros() as usize;
            let n = by_pos[i]
                .iter()
                .filter(|&&bi| bs[bi].m & rem == bs[bi].m)
                .count();

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
    })
}

/// Accumulate one numeric row.
/// # Arguments:
/// - `acc`: Final accumulator.
/// - `r`: Numeric row.
/// # Returns:
/// - `()`: Mutates `acc`.
fn add(acc: &mut Acc, mut r: Row) {
    crate::time_call!(crate::timers::wick::add_add, {
        if r.c.is_zero() {
            return;
        }

        sortids(&mut r.d);
        sortids(&mut r.t);

        let key = (r.d.to_vec(), r.t.to_vec());

        match acc.entry(key) {
            Entry::Vacant(e) => {
                e.insert(r.c);
            }
            Entry::Occupied(mut e) => {
                *e.get_mut() += r.c;

                if e.get().is_zero() {
                    e.remove();
                }
            }
        }
    });
}

/// Sort a tiny factor-id list.
/// # Arguments:
/// - `xs`: Factor ids.
/// # Returns:
/// - `()`: Sorts `xs` in place.
fn sortids(xs: &mut SmallVec<[Id; 4]>) {
    crate::time_call!(crate::timers::wick::add_sortids, {
        for i in 1..xs.len() {
            let x = xs[i];
            let mut j = i;

            while j > 0 && xs[j - 1] > x {
                xs[j] = xs[j - 1];
                j -= 1;
            }

            xs[j] = x;
        }
    });
}

/// Decode numeric rows to symbolic terms.
/// # Arguments:
/// - `s`: Factor store.
/// - `acc`: Final numeric accumulator.
/// # Returns:
/// - `Expr`: Symbolic expression.
fn out(s: &Store, acc: Acc) -> Expr {
    crate::time_call!(crate::timers::wick::add_out, {
        acc.into_iter()
            .filter(|(_, c)| !c.is_zero())
            .map(|((ds, ts), c)| {
                let deltas = ds.iter().map(|&i| s.d[(i - 1) as usize]).collect();
                let tensors = ts
                    .iter()
                    .map(|&i| s.t[(i - 1) as usize].clone())
                    .collect();

                Term {
                    coeff: Rational {
                        num: *c.numer(),
                        den: *c.denom(),
                    },
                    deltas,
                    tensors,
                }
            })
            .collect()
    })
}

/// Build one numeric row.
/// # Arguments:
/// - `c`: Coefficient.
/// - `d`: Delta ids.
/// - `t`: Tensor ids.
/// # Returns:
/// - `Row`: Numeric row.
fn row(c: Rat, d: &[Id], t: &[Id]) -> Row {
    crate::time_call!(crate::timers::wick::add_row, {
        Row {
            c,
            d: d.iter().copied().collect(),
            t: t.iter().copied().collect(),
            e: SmallVec::new(),
        }
    })
}

/// Return bit positions in a mask.
/// # Arguments:
/// - `m`: Bit mask.
/// # Returns:
/// - `Vec<usize>`: Set-bit positions.
fn bits(mut m: u64) -> Vec<usize> {
    crate::time_call!(crate::timers::wick::add_bits, {
        let mut out = Vec::new();

        while m != 0 {
            let b = m & m.wrapping_neg();
            out.push(b.trailing_zeros() as usize);
            m ^= b;
        }

        out
    })
}

/// Return the mask for selected positions.
/// # Arguments:
/// - `xs`: Positions.
/// # Returns:
/// - `u64`: Bit mask.
fn mask(xs: &[usize]) -> u64 {
    crate::time_call!(crate::timers::wick::add_mask, {
        xs.iter().fold(0, |m, &i| m | (1u64 << i))
    })
}

/// Build the GNO-group mask touched by a Wick block.
/// # Arguments:
/// - `xs`: Operator positions in the block.
/// - `ops`: Spin-orbital operator string.
/// # Returns:
/// - `u64`: Bit mask of GNO group ids.
fn gmask(xs: &[usize], ops: &[Op]) -> u64 {
    crate::time_call!(crate::timers::wick::add_gmask, {
        xs.iter().fold(0, |m, &i| m | (1u64 << ops[i].group))
    })
}

/// Check whether a set of Wick-block group masks forms a connected graph.
/// # Arguments:
/// - `want`: Required group mask.
/// - `edges`: Wick-block group masks.
/// # Returns:
/// - `bool`: Whether all required groups are connected.
fn conn(want: u64, edges: &[u64]) -> bool {
    crate::time_call!(crate::timers::wick::add_conn, {
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
    })
}

/// Return the Wick sign from moving a block before the remaining operators.
/// # Arguments:
/// - `m`: Selected block mask.
/// - `rest`: Remaining operator mask.
/// # Returns:
/// - `i64`: Fermionic sign.
fn cross(mut m: u64, rest: u64) -> i64 {
    crate::time_call!(crate::timers::wick::add_cross, {
        let mut p = 0;

        while m != 0 {
            let b = m & m.wrapping_neg();
            let i = b.trailing_zeros();
            p ^= (rest & ((1u64 << i) - 1)).count_ones() & 1;
            m ^= b;
        }

        if p == 0 { 1 } else { -1 }
    })
}

/// Normal-order active operators.
/// # Arguments:
/// - `ops`: Active operators.
/// # Returns:
/// - `Option<(i64, Vec<Op>, Vec<Op>)>`: Sign, creators, and annihilators.
fn norm(ops: &[Op]) -> Option<(i64, Vec<Op>, Vec<Op>)> {
    crate::time_call!(crate::timers::wick::add_norm, {
        let cs = ops
            .iter()
            .copied()
            .filter(|o| o.kind == OpKind::Create)
            .collect::<Vec<_>>();

        let as_ = ops
            .iter()
            .copied()
            .filter(|o| o.kind == OpKind::Annihilate)
            .collect::<Vec<_>>();

        if cs.len() + as_.len() != ops.len() {
            return None;
        }

        let mut inv = 0;

        for i in 0..ops.len() {
            if ops[i].kind != OpKind::Annihilate {
                continue;
            }

            inv += ops[i + 1..]
                .iter()
                .filter(|o| o.kind == OpKind::Create)
                .count();
        }

        Some((if inv % 2 == 0 { 1 } else { -1 }, cs, as_))
    })
}

/// Evaluate a spin-free product with all Wick contractions retained.
/// # Arguments:
/// - `p`: Spin-free product.
/// # Returns:
/// - `Expr`: Canonical contracted expression.
pub fn eval(p: &Product) -> Expr {
    crate::time_call!(crate::timers::wick::add_eval, { eval0(p, false) })
}

/// Evaluate a spin-free product keeping only connected contractions.
/// # Arguments:
/// - `p`: Spin-free product.
/// # Returns:
/// - `Expr`: Canonical connected contracted expression.
pub fn evalc(p: &Product) -> Expr {
    crate::time_call!(crate::timers::wick::add_evalc, { evalc0(p) })
}

/// Number of spin strings to evaluate in one connected Wick batch.
/// # Arguments:
/// - None.
/// # Returns:
/// - `usize`: Spin-string batch size.
fn spinbatch() -> usize {
    crate::time_call!(crate::timers::wick::add_spinbatch, {
        std::env::var("WICK_SPIN_BATCH")
            .ok()
            .and_then(|x| x.parse::<usize>().ok())
            .filter(|&x| x > 0)
            .unwrap_or(64)
    })
}

/// Number of spin strings to stream concurrently in split mode.
/// # Arguments:
/// - None.
/// # Returns:
/// - `usize`: Number of active spin-string workers.
fn spinpar() -> usize {
    crate::time_call!(crate::timers::wick::add_spinpar, {
        std::env::var("WICK_SPIN_PAR")
            .ok()
            .and_then(|x| x.parse::<usize>().ok())
            .filter(|&x| x > 0)
            .unwrap_or(2)
    })
}

/// Number of streamed chunks allowed to queue before workers block.
/// # Arguments:
/// - None.
/// # Returns:
/// - `usize`: Bounded output queue length.
fn streamqueue() -> usize {
    crate::time_call!(crate::timers::wick::add_streamqueue, {
        std::env::var("WICK_STREAM_QUEUE")
            .ok()
            .and_then(|x| x.parse::<usize>().ok())
            .filter(|&x| x > 0)
            .unwrap_or(2)
    })
}

/// Connected accumulator flush threshold for streaming split mode.
/// # Arguments:
/// - None.
/// # Returns:
/// - `usize`: Approximate numeric terms per streamed subchunk.
fn accflush() -> usize {
    crate::time_call!(crate::timers::wick::add_accflush, {
        std::env::var("WICK_ACC_FLUSH")
            .ok()
            .and_then(|x| x.parse::<usize>().ok())
            .filter(|&x| x > 0)
            .unwrap_or(100_000)
    })
}

/// Evaluate one connected spin-orbital string and stream accumulator chunks.
/// # Arguments:
/// - `ops`: Spin-orbital operator string.
/// - `want`: Required GNO-group mask.
/// - `id`: Spin-string id for tracing.
/// - `emit`: Callback receiving one expression subchunk.
/// # Returns:
/// - `()`: Calls `emit` for each non-empty subchunk.
fn eval1cstream(ops: &[Op], want: u64, id: usize, mut emit: impl FnMut(Expr)) {
    crate::time_call!(crate::timers::wick::add_eval1cstream, {
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
        let mut cur = Row {
            c: Rat::one(),
            d: SmallVec::new(),
            t: SmallVec::new(),
            e: SmallVec::new(),
        };
        let mut tail_memo = BTreeMap::new();
        let lim = accflush();
        let mut n = 0usize;

        {
            let mut flush = |acc: &mut Acc| {
                if acc.len() < lim {
                    return;
                }

                let e = out(&s, std::mem::take(acc));

                if !e.is_empty() {
                    crate::progress::mem(format!(
                        "wick::eval1cstream emit spin {id} chunk {n} terms: {}",
                        e.len()
                    ));
                    n += 1;
                    emit(e);
                }
            };

            walkc(
                full,
                &bs,
                &by_pos,
                want,
                &mut cur,
                &mut tail_memo,
                &mut acc,
                &mut flush,
            );
        }

        let e = out(&s, acc);

        if !e.is_empty() {
            crate::progress::mem(format!(
                "wick::eval1cstream emit spin {id} chunk {n} terms: {}",
                e.len()
            ));
            emit(e);
        }
    });
}

/// Evaluate a connected spin-free product either as one fast chunk or in spin batches.
/// # Arguments:
/// - `p`: Spin-free product.
/// - `split`: Whether to split this product by spin-string batches.
/// - `emit`: Callback receiving `(spin_chunk_index, was_split, expression)`.
/// # Returns:
/// - `()`: Calls `emit` for each non-empty expression.
pub fn evalcstream(
    p: &Product,
    split: bool,
    mut emit: impl FnMut(usize, bool, Expr) + Send,
) {
    crate::time_call!(crate::timers::wick::add_evalcstream, {
        let want = (1u64 << p.groups.len()) - 1;
        let ss = spin(p);
        let nspin = ss.len();
        let nops = ss.iter().map(|x| x.len()).max().unwrap_or(0);
        let batch = spinbatch();

        crate::progress::mem(format!(
            "wick::evalcstream spin strings: {nspin}, ops: {nops}, split: {split}, batch: {batch}"
        ));

        if !split {
            let e = ss
                .into_par_iter()
                .fold(crate::canonical::Acc::new, |mut acc, ops| {
                    let e = eval1c(&ops, want);
                    crate::time_call!(crate::timers::canonical::add_accumulate, {
                        acc.addexpr(e);
                    });
                    acc
                })
                .reduce(crate::canonical::Acc::new, |mut a, b| {
                    a.merge(b);
                    a
                })
                .finish();

            if !e.is_empty() {
                emit(0, false, e);
            }

            return;
        }

        if batch == 1 {
            let par = spinpar();
            let queue = streamqueue();
            let mut ci = 0usize;

            for (base, group) in ss.chunks(par).enumerate() {
                let first = base * par;
                let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, usize, Expr)>(queue);

                rayon::scope(|scope| {
                    for (j, ops) in group.iter().enumerate() {
                        let id = first + j;
                        let tx = tx.clone();

                        scope.spawn(move |_| {
                            crate::progress::mem(format!("wick::evalcstream start spin {id}"));
                            eval1cstream(ops, want, id, |e| {
                                let _ = tx.send((id, 0usize, e));
                            });
                            crate::progress::mem(format!("wick::evalcstream end spin {id}"));
                        });
                    }

                    drop(tx);

                    for (_id, _local, e) in rx {
                        emit(ci, true, e);
                        ci += 1;
                    }
                });
            }

            return;
        }

        for (si, xs) in ss.chunks(batch).enumerate() {
            let e = xs
                .par_iter()
                .fold(crate::canonical::Acc::new, |mut acc, ops| {
                    let e = eval1c(ops, want);
                    crate::time_call!(crate::timers::canonical::add_accumulate, {
                        acc.addexpr(e);
                    });
                    acc
                })
                .reduce(crate::canonical::Acc::new, |mut a, b| {
                    a.merge(b);
                    a
                })
                .finish();

            if !e.is_empty() {
                emit(si, true, e);
            }
        }
    });
}

/// Evaluate a spin-free product.
/// # Arguments:
/// - `p`: Spin-free product.
/// - `connected`: Whether to discard disconnected contractions.
/// # Returns:
/// - `Expr`: Canonical contracted expression.
fn eval0(p: &Product, connected: bool) -> Expr {
    crate::time_call!(crate::timers::wick::add_eval0, {
        let want = (1u64 << p.groups.len()) - 1;
        let e = spin(p)
            .into_par_iter()
            .flat_map(|ops| eval1(&ops, connected, want))
            .collect();

        canon(e)
    })
}

/// Evaluate one spin-orbital string.
/// # Arguments:
/// - `ops`: Spin-orbital operator string.
/// - `connected`: Whether to discard disconnected contractions.
/// - `want`: Required GNO-group mask.
/// # Returns:
/// - `Expr`: Contracted expression.
fn eval1(ops: &[Op], connected: bool, want: u64) -> Expr {
    crate::time_call!(crate::timers::wick::add_eval1, {
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
    })
}

/// Evaluate a connected spin-free product.
/// # Arguments:
/// - `p`: Spin-free product.
/// # Returns:
/// - `Expr`: Canonical connected contracted expression.
fn evalc0(p: &Product) -> Expr {
    crate::time_call!(crate::timers::wick::add_evalc0, {
        let want = (1u64 << p.groups.len()) - 1;

        spin(p)
            .into_par_iter()
            .fold(crate::canonical::Acc::new, |mut acc, ops| {
                let e = eval1c(&ops, want);
                crate::time_call!(crate::timers::canonical::add_accumulate, {
                    acc.addexpr(e);
                });
                acc
            })
            .reduce(crate::canonical::Acc::new, |mut a, b| {
                a.merge(b);
                a
            })
            .finish()
    })
}

/// Evaluate one spin-orbital string using hybrid connected enumeration.
/// # Arguments:
/// - `ops`: Spin-orbital operator string.
/// - `want`: Required GNO-group mask.
/// # Returns:
/// - `Expr`: Connected contracted expression.
fn eval1c(ops: &[Op], want: u64) -> Expr {
    crate::time_call!(crate::timers::wick::add_eval1c, {
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
        let mut cur = Row {
            c: Rat::one(),
            d: SmallVec::new(),
            t: SmallVec::new(),
            e: SmallVec::new(),
        };
        let mut tail_memo = BTreeMap::new();
        let mut flush = |_: &mut Acc| {};

        walkc(
            full,
            &bs,
            &by_pos,
            want,
            &mut cur,
            &mut tail_memo,
            &mut acc,
            &mut flush,
        );

        out(&s, acc)
    })
}

/// Recursively enumerate until selected contractions are connected, then memoise the suffix.
/// # Arguments:
/// - `left`: Remaining operator-position mask.
/// - `bs`: Wick blocks.
/// - `by_pos`: Block ids containing each position.
/// - `want`: Required GNO-group mask.
/// - `cur`: Current partial row.
/// - `tail_memo`: Memo table for connected suffix enumeration.
/// - `acc`: Final numeric accumulator.
/// - `flush`: Accumulator flush callback.
/// # Returns:
/// - `()`: Mutates `acc`.
fn walkc(
    left: u64,
    bs: &[Block],
    by_pos: &[Vec<usize>],
    want: u64,
    cur: &mut Row,
    tail_memo: &mut BTreeMap<u64, Vec<Row>>,
    acc: &mut Acc,
    flush: &mut impl FnMut(&mut Acc),
) {
    crate::time_call!(crate::timers::wick::add_walkc, {
        walkc_inner(left, bs, by_pos, want, cur, tail_memo, acc, flush)
    });
}

/// Recursively enumerate connected contractions without starting nested timers.
fn walkc_inner(
    left: u64,
    bs: &[Block],
    by_pos: &[Vec<usize>],
    want: u64,
    cur: &mut Row,
    tail_memo: &mut BTreeMap<u64, Vec<Row>>,
    acc: &mut Acc,
    flush: &mut impl FnMut(&mut Acc),
) {
    if rootseen(want, &cur.e) == want {
        for tail in walk(left, bs, by_pos, false, tail_memo) {
            add(acc, joinrow(cur, &tail));
            flush(acc);
        }

        return;
    }

    if left == 0 {
        return;
    }

    if !canconnect(left, bs, want, &cur.e) {
        return;
    }

    let Some(i) = pick(left, bs, by_pos) else {
        return;
    };

    for &bi in &by_pos[i] {
        let b = &bs[bi];

        if b.m & left != b.m {
            continue;
        }

        let rest = left & !b.m;

        if !canconnect1(rest, bs, want, &cur.e, b.g) {
            continue;
        }

        let sgn = cross(b.m, rest);
        let c0 = cur.c;
        let nd = cur.d.len();
        let nt = cur.t.len();
        let ne = cur.e.len();

        for v in &b.v {
            let mut c = c0 * v.c;

            if sgn < 0 {
                c = -c;
            }

            cur.c = c;
            cur.d.extend_from_slice(&v.d);
            cur.t.extend_from_slice(&v.t);
            cur.e.push(b.g);

            walkc_inner(rest, bs, by_pos, want, cur, tail_memo, acc, flush);

            cur.d.truncate(nd);
            cur.t.truncate(nt);
            cur.e.truncate(ne);
        }

        cur.c = c0;
    }
}

/// Join a connected prefix row with a memoised suffix row.
/// # Arguments:
/// - `a`: Prefix row.
/// - `b`: Suffix row.
/// # Returns:
/// - `Row`: Combined row.
fn joinrow(a: &Row, b: &Row) -> Row {
    crate::time_call!(crate::timers::wick::add_joinrow, {
        let mut out = Row {
            c: a.c * b.c,
            d: a.d.clone(),
            t: a.t.clone(),
            e: SmallVec::new(),
        };

        out.d.extend_from_slice(&b.d);
        out.t.extend_from_slice(&b.t);
        out
    })
}

/// Return the root-connected GNO-group component implied by selected Wick blocks.
/// # Arguments:
/// - `want`: Required group mask.
/// - `edges`: Selected Wick-block group masks.
/// # Returns:
/// - `u64`: Root-connected group mask.
fn rootseen(want: u64, edges: &[u64]) -> u64 {
    crate::time_call!(crate::timers::wick::add_rootseen, {
        if want.count_ones() <= 1 {
            return want;
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

        seen & want
    })
}

/// Check whether the partial contraction can still become connected.
/// # Arguments:
/// - `left`: Remaining operator-position mask.
/// - `bs`: Wick blocks.
/// - `want`: Required GNO-group mask.
/// - `edges`: Already selected Wick-block group masks.
/// # Returns:
/// - `bool`: Whether connected completion remains possible.
fn canconnect(left: u64, bs: &[Block], want: u64, edges: &[u64]) -> bool {
    crate::time_call!(crate::timers::wick::add_canconnect, {
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

            for b in bs {
                if b.m & left == b.m && b.g & seen != 0 {
                    seen |= b.g;
                }
            }

            if seen == old {
                break;
            }
        }

        seen & want == want
    })
}

/// Check whether adding one candidate block can still lead to connected completion.
/// # Arguments:
/// - `left`: Remaining operator-position mask after the candidate block.
/// - `bs`: Wick blocks.
/// - `want`: Required GNO-group mask.
/// - `edges`: Already selected Wick-block group masks.
/// - `extra`: Candidate Wick-block group mask.
/// # Returns:
/// - `bool`: Whether connected completion remains possible.
fn canconnect1(left: u64, bs: &[Block], want: u64, edges: &[u64], extra: u64) -> bool {
    crate::time_call!(crate::timers::wick::add_canconnect1, {
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

            if extra & seen != 0 {
                seen |= extra;
            }

            for b in bs {
                if b.m & left == b.m && b.g & seen != 0 {
                    seen |= b.g;
                }
            }

            if seen == old {
                break;
            }
        }

        seen & want == want
    })
}
