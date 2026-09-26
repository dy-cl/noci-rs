// reduce.rs
//! Global reduction of spin-free residuals modulo exact spin relations.
//!
//! For an `SU(2)`-invariant spin-\tfrac12 ensemble the spin-free cumulants of rank three and
//! four satisfy linear relations: symmetrising any three lower indices projects the spin part
//! onto the totally antisymmetric representation of three spins, which is empty for two spin
//! states,
//!
//! `\sum_{\rho \in S_3} \Lambda^{\cdots}_{\cdots q_{\rho(1)} q_{\rho(2)} q_{\rho(3)} \cdots} = 0.`
//!
//! Every term containing such a cumulant therefore lies in a family of terms whose sum
//! vanishes. Adding multiples of these relations leaves the residual unchanged, and the
//! reduction chooses multiples that minimise the number of distinct terms.

// Standard library imports.
use std::collections::{BTreeMap, VecDeque};

// External crate imports.
use itertools::Itertools;
use num_rational::Ratio;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

// Crate-root imports.
use crate::canon::{self, Form, Key};
use crate::spin::{LAMBDA3, LAMBDA4, Table};

/// One vanishing linear combination of canonical terms, normalised for deduplication.
type Relation = Vec<(Key, i64)>;

/// Minimise the number of terms of one spin-free residual modulo cumulant relations.
/// Relations are generated from the terms present, and each relation is applied with the
/// multiple that removes the most terms; relations touching changed terms are revisited until
/// no relation lowers the count. New terms introduced by a move seed a further round.
/// # Arguments:
/// - `res`: Spin-free residual, reduced in place.
/// # Returns:
/// - `()`: Mutates `res.terms`.
pub(crate) fn reduce_by_cumulant_relations(res: &mut Table) {
    let spaces = res.free.iter().map(|&s| s as u8).collect::<Vec<_>>();
    let mut seen = FxHashSet::<Key>::default();

    loop {
        // Generate relations from terms not yet expanded.
        let fresh = res
            .terms
            .keys()
            .filter(|k| !seen.contains(*k))
            .cloned()
            .collect::<Vec<_>>();
        if fresh.is_empty() {
            break;
        }
        seen.extend(fresh.iter().cloned());

        let mut rels = fresh
            .par_iter()
            .flat_map_iter(|k| cumulant_relations(&spaces, k))
            .collect::<Vec<_>>();
        rels.par_sort_unstable();
        rels.dedup();

        if !greedy_descent(&rels, &mut res.terms) {
            break;
        }
    }
}

/// Test whether a combination of terms vanishes identically modulo cumulant relations.
/// Relations containing the terms are collected, then those containing the terms they reach,
/// for a bounded number of rounds; the combination vanishes when it lies in their span.
/// # Arguments:
/// - `spaces`: Orbital space of every free index.
/// - `terms`: Coefficient of every canonical term.
/// # Returns:
/// - `bool`: Whether the combination is a sum of relations.
pub(crate) fn vanishes_modulo_relations(
    spaces: &[u8],
    terms: &FxHashMap<Key, Ratio<i64>>,
) -> bool {
    const ROUNDS: usize = 3;
    let mut seen = FxHashSet::<Key>::default();
    let mut frontier = terms.keys().cloned().collect::<Vec<_>>();
    let mut rels = Vec::new();

    for _ in 0..ROUNDS {
        if lies_in_span(&rels, terms) {
            return true;
        }

        // Expand the relations to every term reached so far.
        let fresh = frontier
            .drain(..)
            .filter(|k| seen.insert(k.clone()))
            .collect::<Vec<_>>();
        if fresh.is_empty() {
            return false;
        }
        let more = fresh
            .par_iter()
            .flat_map_iter(|k| cumulant_relations(spaces, k))
            .collect::<Vec<_>>();
        frontier.extend(
            more.iter()
                .flat_map(|rel| rel.iter().map(|(k, _)| k))
                .filter(|k| !seen.contains(*k))
                .cloned(),
        );
        rels.extend(more);
    }

    lies_in_span(&rels, terms)
}

/// Test whether one vector lies in the span of a set of relations by exact elimination.
/// # Arguments:
/// - `rels`: Relations.
/// - `x`: Coefficient of every canonical term.
/// # Returns:
/// - `bool`: Whether `x` is a rational combination of `rels`.
fn lies_in_span(
    rels: &[Relation],
    x: &FxHashMap<Key, Ratio<i64>>,
) -> bool {
    if x.is_empty() {
        return true;
    }

    // Number every term that appears.
    let mut ids = FxHashMap::<&Key, usize>::default();

    // Echelon basis keyed by pivot column, each row normalised to a unit pivot.
    let mut basis = BTreeMap::<usize, BTreeMap<usize, Ratio<i64>>>::new();
    for rel in rels {
        let mut row = BTreeMap::new();
        for (k, c) in rel {
            let n = ids.len();
            let j = *ids.entry(k).or_insert(n);
            row.insert(j, Ratio::from_integer(*c));
        }
        eliminate_pivots(&basis, &mut row);
        if let Some((&p, &v)) = row.iter().next() {
            for w in row.values_mut() {
                *w /= v;
            }
            basis.insert(p, row);
        }
    }

    let mut row = BTreeMap::new();
    for (k, c) in x {
        let n = ids.len();
        let j = *ids.entry(k).or_insert(n);
        row.insert(j, *c);
    }
    eliminate_pivots(&basis, &mut row);

    row.is_empty()
}

/// Eliminate every pivot column of an echelon basis from one row.
/// Basis rows only contain columns after their pivot, so pivots are eliminated in increasing
/// column order.
/// # Arguments:
/// - `basis`: Echelon rows keyed by pivot column, with unit pivots.
/// - `row`: Row to reduce, updated in place.
/// # Returns:
/// - `()`: Mutates `row`.
fn eliminate_pivots(
    basis: &BTreeMap<usize, BTreeMap<usize, Ratio<i64>>>,
    row: &mut BTreeMap<usize, Ratio<i64>>,
) {
    let zero = Ratio::from_integer(0);
    let mut from = 0;

    while let Some((c, v)) = row
        .range(from..)
        .find(|(c, _)| basis.contains_key(c))
        .map(|(&c, &v)| (c, v))
    {
        for (&j, &w) in &basis[&c] {
            let e = row.entry(j).or_insert(zero);
            *e -= v * w;
            if *e == zero {
                row.remove(&j);
            }
        }
        from = c + 1;
    }
}

/// Apply relations greedily while each application lowers the number of terms.
/// For relation `r` and a term `k` of it with coefficient `x_k`, the move
/// `x \leftarrow x - (x_k / r_k) r` removes `k`; the best such move is taken when it removes
/// more terms than it creates.
/// # Arguments:
/// - `rels`: Relations in deterministic order.
/// - `x`: Term coefficients, updated in place.
/// # Returns:
/// - `bool`: Whether any move was applied.
fn greedy_descent(
    rels: &[Relation],
    x: &mut FxHashMap<Key, Ratio<i64>>,
) -> bool {
    // Relations touching each term.
    let mut touch = FxHashMap::<&Key, Vec<u32>>::default();
    for (r, rel) in rels.iter().enumerate() {
        for (k, _) in rel {
            touch.entry(k).or_default().push(r as u32);
        }
    }

    let mut queue = (0..rels.len() as u32).collect::<VecDeque<_>>();
    let mut queued = vec![true; rels.len()];
    let mut moved = false;
    let zero = Ratio::from_integer(0);

    while let Some(r) = queue.pop_front() {
        queued[r as usize] = false;
        let rel = &rels[r as usize];
        let value = |k: &Key| x.get(k).copied().unwrap_or(zero);
        let before = rel.iter().filter(|(k, _)| value(k) != zero).count();

        // Choose the multiple that leaves the fewest nonzero terms in the relation.
        let mut best = None;
        let mut fewest = before;
        for (k, c) in rel {
            let v = value(k);
            if v == zero {
                continue;
            }
            let alpha = v / Ratio::from_integer(*c);
            let after = rel
                .iter()
                .filter(|(j, d)| value(j) - alpha * Ratio::from_integer(*d) != zero)
                .count();
            if after < fewest {
                fewest = after;
                best = Some(alpha);
            }
        }
        let Some(alpha) = best else {
            continue;
        };

        // Apply the move and revisit every relation sharing a changed term.
        let next = rel
            .iter()
            .map(|(k, c)| value(k) - alpha * Ratio::from_integer(*c))
            .collect::<Vec<_>>();
        for ((k, _), v) in rel.iter().zip(next) {
            if v == zero {
                x.remove(k);
            } else {
                x.insert(k.clone(), v);
            }
            for &s in &touch[k] {
                if !queued[s as usize] {
                    queued[s as usize] = true;
                    queue.push_back(s);
                }
            }
        }
        moved = true;
    }

    moved
}

/// Return the cumulant relations containing one term.
/// A rank-three cumulant gives one relation, the symmetrisation of all its lower indices. A
/// rank-four cumulant gives one relation for each lower slot `p` and index `q` placed there,
/// symmetrising the remaining three lower indices over the remaining slots.
/// # Arguments:
/// - `spaces`: Orbital space of every free index.
/// - `key`: Canonical term.
/// # Returns:
/// - `Vec<Relation>`: Nontrivial normalised relations.
fn cumulant_relations(
    spaces: &[u8],
    key: &Key,
) -> Vec<Relation> {
    let mut out = Vec::new();
    let all = spaces
        .iter()
        .chain(&key.dummies)
        .copied()
        .collect::<Vec<_>>();

    for (m, f) in key.factors.iter().enumerate() {
        let k = f.lower.len();
        if f.kind != LAMBDA3 && f.kind != LAMBDA4 {
            continue;
        }

        // Fixed slot and index choices; rank three symmetrises every slot.
        let fixed = if k == 3 {
            vec![None]
        } else {
            (0..k).cartesian_product(0..k).map(Some).collect()
        };

        for choice in fixed {
            let slots = (0..k)
                .filter(|&s| choice.is_none_or(|(p, _)| s != p))
                .collect::<Vec<_>>();
            let rest = (0..k)
                .filter(|&q| choice.is_none_or(|(_, v)| q != v))
                .map(|q| f.lower[q])
                .collect::<Vec<_>>();

            // Sum every placement of the remaining indices into the free slots.
            let mut rel = FxHashMap::<Key, i64>::default();
            for order in rest.iter().permutations(rest.len()) {
                let mut form = Form {
                    spaces: all.clone(),
                    nfree: spaces.len(),
                    factors: key.factors.clone(),
                };
                let g = &mut form.factors[m];
                if let Some((p, v)) = choice {
                    g.lower[p] = f.lower[v];
                }
                for (&s, &&q) in slots.iter().zip(&order) {
                    g.lower[s] = q;
                }
                let (variant, sign) = canon::canonical_key(&form);
                if sign != 0 {
                    *rel.entry(variant).or_insert(0) += sign as i64;
                }
            }

            if let Some(rel) = normalise_relation(rel) {
                out.push(rel);
            }
        }
    }

    out
}

/// Normalise one relation: drop zero coefficients, sort terms, divide by the coefficient gcd
/// and make the leading coefficient positive.
/// # Arguments:
/// - `rel`: Coefficient of every term.
/// # Returns:
/// - `Option<Relation>`: Normalised relation, or `None` when it has fewer than two terms.
fn normalise_relation(rel: FxHashMap<Key, i64>) -> Option<Relation> {
    let mut rel = rel.into_iter().filter(|(_, c)| *c != 0).collect::<Vec<_>>();
    if rel.len() < 2 {
        return None;
    }
    rel.sort_unstable();

    let g = rel
        .iter()
        .fold(0i64, |g, (_, c)| greatest_common_divisor(g, c.abs()));
    let s = if rel[0].1 < 0 { -g } else { g };
    for (_, c) in &mut rel {
        *c /= s;
    }

    Some(rel)
}

/// Greatest common divisor of two non-negative integers.
/// # Arguments:
/// - `a`: First integer.
/// - `b`: Second integer.
/// # Returns:
/// - `i64`: `\greatest_common_divisor(a, b)`.
fn greatest_common_divisor(
    a: i64,
    b: i64,
) -> i64 {
    if b == 0 {
        a
    } else {
        greatest_common_divisor(b, a % b)
    }
}
