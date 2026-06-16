// canonical.rs

use std::collections::{BTreeMap, BTreeSet};

use itertools::Itertools;
use num_traits::Zero;

use crate::ir::{Delta, Expr, Idx, Rational, Tensor, TensorKind, Term};
use crate::spinsum::{self, Rat};

/// High-rank cumulant orbit key.
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
struct Key {
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

/// Canonical expression accumulator.
pub struct Acc {
    /// Combined coefficients keyed by canonical deltas and tensors.
    acc: BTreeMap<(Vec<Delta>, Vec<Tensor>), Rat>,
}

impl Acc {
    /// Create an empty canonical accumulator.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `Acc`: Empty accumulator.
    pub fn new() -> Self {
        Self { acc: BTreeMap::new() }
    }
    
    /// Add one term to the accumulator.
    /// # Arguments:
    /// - `x`: Term to add.
    /// # Returns:
    /// - `()`: Mutates the accumulator.
    pub fn addterm(&mut self, mut x: Term) {
        x.deltas.sort();
        x.tensors = x.tensors.into_iter().map(ten).collect();
        x.tensors.sort();

        let key = (x.deltas, x.tensors);
        let c = Rat::new(x.coeff.num, x.coeff.den);

        let remove = {
            let v = self.acc.entry(key.clone()).or_insert_with(Rat::zero);
            *v += c;
            v.is_zero()
        };

        if remove {
            self.acc.remove(&key);
        }
    }

    /// Merge another accumulator into this accumulator.
    /// # Arguments:
    /// - `other`: Accumulator to merge.
    /// # Returns:
    /// - `()`: Mutates the accumulator.
    pub fn merge(&mut self, other: Acc) {
        for (key, c) in other.acc {
            let remove = {
                let v = self.acc.entry(key.clone()).or_insert_with(Rat::zero);
                *v += c;
                v.is_zero()
            };

            if remove {
                self.acc.remove(&key);
            }
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
        self.acc.into_iter()
            .filter(|(_, c)| !c.is_zero())
            .map(|((deltas, tensors), c)| Term {
                coeff: Rational { num: *c.numer(), den: *c.denom() },
                deltas,
                tensors,
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
    let mut gs = BTreeMap::<Key, BTreeMap<Vec<usize>, Rat>>::new();

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

        let key = Key {
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
