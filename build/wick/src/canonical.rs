// canonical.rs

use std::collections::{BTreeMap, BTreeSet};
use std::collections::btree_map::Entry;

use itertools::Itertools;
use num_traits::Zero;
use smallvec::SmallVec;

use crate::ir::{Delta, Expr, Idx, Rational, Tensor, TensorKind, Term};
use crate::spinsum::{self, Rat};

/// High-rank cumulant orbit key.
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
struct OrbitKey {
    /// Cumulant rank.
    n: usize,
    /// Delta factors.
    d: Vec<Delta>,
    /// Non-orbit tensor factors.
    t: Vec<Tensor>,
    /// Cumulant kind.
    k: TensorKind,
    /// Upper indices.
    up: Vec<Idx>,
    /// Sorted lower-index base.
    lo: Vec<Idx>,
}

/// Interned factor id.
type Id = u32;

/// Canonical term key.
type TermKey = (SmallVec<[Id; 4]>, SmallVec<[Id; 8]>);

/// Interned canonical factors.
#[derive(Clone, Debug, Default)]
struct Store {
    /// Delta-to-id lookup.
    ds: BTreeMap<Delta, Id>,
    /// Tensor-to-id lookup.
    ts: BTreeMap<Tensor, Id>,
    /// Delta table.
    d: Vec<Delta>,
    /// Tensor table.
    t: Vec<Tensor>,
}

impl Store {
    /// Intern one delta.
    /// # Arguments:
    /// - `x`: Delta factor.
    /// # Returns:
    /// - `Id`: One-based delta id.
    fn delta(&mut self, x: Delta) -> Id {
        if let Some(&id) = self.ds.get(&x) {
            return id;
        }

        let id = self.d.len() as Id + 1;

        self.ds.insert(x, id);
        self.d.push(x);

        id
    }

    /// Intern one tensor.
    /// # Arguments:
    /// - `x`: Tensor factor.
    /// # Returns:
    /// - `Id`: One-based tensor id.
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

/// Sort a tiny factor-id list.
/// # Arguments:
/// - `xs`: Factor ids.
/// # Returns:
/// - `()`: Sorts `xs` in place.
fn sortids(xs: &mut [Id]) {
    for i in 1..xs.len() {
        let x = xs[i];
        let mut j = i;

        while j > 0 && xs[j - 1] > x {
            xs[j] = xs[j - 1];
            j -= 1;
        }

        xs[j] = x;
    }
}

/// Canonical expression accumulator.
pub struct Acc {
    /// Interned canonical factor store.
    store: Store,
    /// Combined coefficients keyed by interned canonical factors.
    acc: BTreeMap<TermKey, Rat>,
}

impl Acc {
    /// Create an empty canonical accumulator.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `Acc`: Empty accumulator.
    pub fn new() -> Self {
        Self {
            store: Store::default(),
            acc: BTreeMap::default(),
        }
    }
    
    /// Add one coefficient by canonical integer key.
    /// # Arguments:
    /// - `key`: Canonical integer factor key.
    /// - `c`: Coefficient.
    /// # Returns:
    /// - `()`: Mutates the accumulator.
    fn addkey(&mut self, key: TermKey, c: Rat) {
        if c.is_zero() {
            return;
        }

        match self.acc.entry(key) {
            Entry::Vacant(e) => {
                e.insert(c);
            }
            Entry::Occupied(mut e) => {
                *e.get_mut() += c;

                if e.get().is_zero() {
                    e.remove();
                }
            }
        }
    }

    /// Add one term to the accumulator.
    /// # Arguments:
    /// - `x`: Term to add.
    /// # Returns:
    /// - `()`: Mutates the accumulator.
    pub fn addterm(&mut self, x: Term) {
        let mut ds = x.deltas.into_iter()
            .map(|d| self.store.delta(d))
            .collect::<SmallVec<[Id; 4]>>();

        let mut ts = x.tensors.into_iter()
            .map(ten)
            .map(|t| self.store.tensor(t))
            .collect::<SmallVec<[Id; 8]>>();

        sortids(ds.as_mut_slice());
        sortids(ts.as_mut_slice());

        self.addkey((ds, ts), Rat::new(x.coeff.num, x.coeff.den));
    }

    /// Merge another accumulator into this accumulator.
    /// # Arguments:
    /// - `other`: Accumulator to merge.
    /// # Returns:
    /// - `()`: Mutates the accumulator.
    pub fn merge(&mut self, other: Acc) {
        let Acc { store, acc } = other;

        let mut dmap = Vec::with_capacity(store.d.len() + 1);
        dmap.push(0);

        for d in store.d {
            dmap.push(self.store.delta(d));
        }

        let mut tmap = Vec::with_capacity(store.t.len() + 1);
        tmap.push(0);

        for t in store.t {
            tmap.push(self.store.tensor(t));
        }

        for ((ds, ts), c) in acc {
            let mut ds = ds.into_iter()
                .map(|i| dmap[i as usize])
                .collect::<SmallVec<[Id; 4]>>();

            let mut ts = ts.into_iter()
                .map(|i| tmap[i as usize])
                .collect::<SmallVec<[Id; 8]>>();

            sortids(ds.as_mut_slice());
            sortids(ts.as_mut_slice());

            self.addkey((ds, ts), c);
        }
    }

    /// Add one expression to the accumulator.
    /// # Arguments:
    /// - `e`: Expression to add.
    /// # Returns:
    /// - `()`: Mutates the accumulator.
    pub fn addexpr(&mut self, e: Expr) {
        for x in e {
            self.addterm(x);
        }
    }
    
    /// Convert the accumulator into an expression without sparsification.
    /// # Arguments:
    /// - `self`: Accumulator.
    /// # Returns:
    /// - `Expr`: Combined expression.
    fn intoexpr(self) -> Expr {
        let Acc { store, acc } = self;
        let Store { d, t, .. } = store;

        acc.into_iter()
            .filter(|(_, c)| !c.is_zero())
            .map(|((ds, ts), c)| Term {
                coeff: Rational { num: *c.numer(), den: *c.denom() },
                deltas: ds.into_iter().map(|i| d[(i - 1) as usize]).collect(),
                tensors: ts.into_iter().map(|i| t[(i - 1) as usize].clone()).collect(),
            })
            .collect()
    }

    /// Finish canonicalisation.
    /// # Arguments:
    /// - `self`: Accumulator.
    /// # Returns:
    /// - `Expr`: Canonical expression.
    pub fn finish(self) -> Expr {
        sum(spar(self.intoexpr()))
    }
}

/// Canonicalise a symbolic expression.
/// # Arguments:
/// - `e`: Symbolic expression.
/// # Returns:
/// - `Expr`: Canonical symbolic expression.
pub fn canon(e: Expr) -> Expr {
    let mut acc = Acc::new();

    acc.addexpr(e);
    acc.finish()
}

/// Combine equal terms and apply simple tensor symmetries.
/// # Arguments:
/// - `e`: Symbolic expression.
/// # Returns:
/// - `Expr`: Combined expression.
fn sum(e: Expr) -> Expr {
    let mut acc = Acc::new();

    acc.addexpr(e);
    acc.intoexpr()
}

/// Canonicalise one tensor.
/// # Arguments:
/// - `x`: Tensor factor.
/// # Returns:
/// - `Tensor`: Canonical tensor factor.
fn ten(x: Tensor) -> Tensor {
    if x.kind == TensorKind::Lambda2 && x.upper.len() == 2 && x.lower.len() == 2 {
        let y = Tensor {
            kind: TensorKind::Lambda2,
            upper: vec![x.upper[1], x.upper[0]],
            lower: vec![x.lower[1], x.lower[0]],
        };

        if y < x { y } else { x }
    } else {
        x
    }
}

/// Return high cumulant rank.
/// # Arguments:
/// - `k`: Tensor kind.
/// # Returns:
/// - `Option<usize>`: Rank for Lambda3/Lambda4.
fn rank(k: TensorKind) -> Option<usize> {
    match k {
        TensorKind::Lambda3 => Some(3),
        TensorKind::Lambda4 => Some(4),
        _ => None,
    }
}

/// Sparsify high-rank cumulant orbits.
/// # Arguments:
/// - `e`: Symbolic expression.
/// # Returns:
/// - `Expr`: Sparsified expression.
fn spar(e: Expr) -> Expr {
    let mut out = Vec::new();
    let mut gs = BTreeMap::<OrbitKey, BTreeMap<Vec<usize>, Rat>>::new();

    for term in e {
        let hs = term.tensors.iter()
            .enumerate()
            .filter_map(|(i, t)| rank(t.kind).map(|n| (i, n, t)))
            .collect::<Vec<_>>();

        if hs.len() != 1 {
            out.push(term);
            continue;
        }

        let (i, n, lam) = hs[0];

        if lam.upper.len() != n || lam.lower.len() != n {
            out.push(term);
            continue;
        }

        if lam.lower.iter().copied().collect::<BTreeSet<_>>().len() != n {
            out.push(term);
            continue;
        }

        let mut lo = lam.lower.clone();
        lo.sort_unstable();

        let Some(p) = perm(&lo, &lam.lower) else {
            out.push(term);
            continue;
        };

        let mut ts = term.tensors.clone();
        ts.remove(i);
        
        let key = OrbitKey {
            n,
            d: term.deltas.clone(),
            t: ts,
            k: lam.kind,
            up: lam.upper.clone(),
            lo,
        };

        let c = Rat::new(term.coeff.num, term.coeff.den);
        let v = gs.entry(key).or_default().entry(p).or_insert_with(Rat::zero);
        *v += c;
    }

    for (key, cs) in gs {
        for (p, c) in best(key.n, cs) {
            if c.is_zero() {
                continue;
            }

            let mut ts = key.t.clone();
            ts.push(Tensor {
                kind: key.k,
                upper: key.up.clone(),
                lower: p.iter().map(|&i| key.lo[i]).collect(),
            });

            out.push(Term {
                coeff: Rational { num: *c.numer(), den: *c.denom() },
                deltas: key.d.clone(),
                tensors: ts,
            });
        }
    }

    out
}

/// Express `x` as a permutation of `base`.
/// # Arguments:
/// - `base`: Sorted base.
/// - `x`: Actual ordering.
/// # Returns:
/// - `Option<Vec<usize>>`: Permutation.
fn perm(base: &[Idx], x: &[Idx]) -> Option<Vec<usize>> {
    let mut out = Vec::with_capacity(x.len());

    for y in x {
        out.push(base.iter().position(|z| z == y)?);
    }

    Some(out)
}

/// Find the sparsest equivalent orbit coefficient vector.
/// # Arguments:
/// - `n`: Cumulant rank.
/// - `cs`: Coefficients by lower permutation.
/// # Returns:
/// - `BTreeMap<Vec<usize>, Rat>`: Sparse coefficients.
fn best(n: usize, cs: BTreeMap<Vec<usize>, Rat>) -> BTreeMap<Vec<usize>, Rat> {
    let cs = cs.into_iter()
        .filter(|(_, c)| !c.is_zero())
        .collect::<BTreeMap<_, _>>();

    if cs.is_empty() {
        return cs;
    }

    let (ps, g) = spinsum::data(n).expect("Cached spin sparsifier rank missing.");
    let y = img(ps, g, &cs);
    let m = cs.len();

    for size in 1..=m {
        for sup in (0..ps.len()).combinations(size) {
            let a = (0..ps.len())
                .map(|r| sup.iter().map(|&c| g[r][c].clone()).collect::<Vec<_>>())
                .collect::<Vec<_>>();

            let Some(x) = spinsum::solve(a, y.clone()) else {
                continue;
            };

            let out = sup.iter()
                .copied()
                .zip(x)
                .filter(|(_, c)| !c.is_zero())
                .map(|(i, c)| (ps[i].clone(), c))
                .collect::<BTreeMap<_, _>>();

            if out.len() <= size {
                return out;
            }
        }
    }

    cs
}

/// Project coefficients through the spin Gram matrix.
/// # Arguments:
/// - `ps`: Permutations.
/// - `g`: Gram matrix.
/// - `cs`: Coefficients.
/// # Returns:
/// - `Vec<Rat>`: Projected vector.
fn img(ps: &[Vec<usize>], g: &[Vec<Rat>], cs: &BTreeMap<Vec<usize>, Rat>) -> Vec<Rat> {
    (0..ps.len())
        .map(|i| {
            let mut x = Rat::zero();

            for (j, p) in ps.iter().enumerate() {
                if let Some(c) = cs.get(p) {
                    x += g[i][j].clone() * c.clone();
                }
            }

            x
        })
        .collect()
}
