// spin/mod.rs
//! Spin adaptation of spin-orbital GNOCC residuals.
//!
//! Spin-orbital residuals are converted to spin-free form for an `SU(2)`-invariant spin
//! ensemble by summing over spin labels and replacing every spin block by spin-free tensors:
//!
//! - one-body densities `\gamma^{p\sigma}_{q\tau} = \tfrac12\Gamma^p_q\delta_{\sigma\tau}` and
//!   `\eta^{p\sigma}_{q\tau} = \tfrac12\Theta^p_q\delta_{\sigma\tau}` with `\Theta = 2\delta -
//!   \Gamma`;
//! - Fock, singles-amplitude and one-body projector blocks equal their spin-free tensors when
//!   the spins agree;
//! - antisymmetrised two-body blocks of the integrals, doubles amplitudes and projector become
//!   direct minus exchange spin-free tensors with pair symmetry;
//! - cumulants become the exact `SU(2)` combinations of spin-free lower-index permutations.
//!
//! The projector is written as `\tfrac12\sum R^{pq}_{rs}\hat E^{rs\dagger}_{pq}` (doubles) or
//! `\sum R^p_q\hat E^{q\dagger}_p` (singles) with a symmetric placeholder `R`. Differentiating
//! with respect to one placeholder element gives the residual of one spin-free excitation: each
//! placeholder orientation compatible with a class layout binds the free indices of that class.
//! Metric blocks carry a second placeholder for the right excitation, bound in the same way;
//! an index shared by both placeholders is bound twice and joined by a Kronecker delta.
//!
//! # References
//!
//! - Lee and Tew, *Spin-free Generalised Normal Ordered Coupled Cluster*, arXiv:2507.13472
//!   (2025), Eqs. (21)-(28) and Appendix A.

// Private submodules.
mod gram;
mod rules;

// External crate imports.
use itertools::Itertools;
use num_rational::Ratio;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

// Crate-root imports.
use crate::canon::{self, Factor, Form, Key, Sym};
use crate::so;
use crate::specs::{self, Space};

/// Spin-free one-particle density `\Gamma`, matching the runtime tensor id.
pub(crate) const GAMMA: u8 = 0;
/// Spin-free one-hole density `\Theta`.
pub(crate) const THETA: u8 = 1;
/// Spin-free Fock operator.
pub(crate) const FOCK: u8 = 2;
/// Spin-free two-electron integral.
pub(crate) const ERI: u8 = 3;
/// Spin-free two-body cumulant.
pub(crate) const LAMBDA2: u8 = 4;
/// Spin-free three-body cumulant.
pub(crate) const LAMBDA3: u8 = 5;
/// Spin-free four-body cumulant.
pub(crate) const LAMBDA4: u8 = 6;
/// Kronecker delta between free indices.
pub(crate) const DELTA: u8 = 7;
/// Spin-free singles amplitude.
pub(crate) const T1: u8 = 8;
/// Spin-free pair-symmetric doubles amplitude.
pub(crate) const T2: u8 = 9;
/// Metric excitation placeholder, removed when free indices are bound.
const KET: u8 = 14;
/// Residual projector placeholder, removed when free indices are bound.
const BRA: u8 = 15;

/// Spin-free terms of one residual class or metric block over its free indices.
#[derive(Clone, Debug)]
pub(crate) struct Table {
    /// Spin-free excitation class or metric block name.
    pub(crate) name: &'static str,
    /// Orbital space of every free index, in layout order.
    pub(crate) free: Vec<Space>,
    /// Canonically combined spin-free terms.
    pub(crate) terms: FxHashMap<Key, Ratio<i64>>,
}

/// One spin-free factor before binding, over spin-orbital index ids.
type Raw = (u8, SmallVec<[u16; 4]>, SmallVec<[u16; 4]>);

/// Return the slot symmetry of one spin-free tensor kind.
/// Two-electron integrals `g^{pq}_{rs} = (pr|qs)` over real or holomorphic orbitals are
/// unchanged by exchanging the indices of either electron, `(pr|qs) = (rp|qs)`, as well as by
/// exchanging the electrons.
/// # Arguments:
/// - `k`: Spin-free kind id.
/// # Returns:
/// - `Sym`: Slot symmetry used by the canonical form.
pub(crate) fn slot_symmetry(k: u8) -> Sym {
    match k {
        ERI | DELTA => Sym::Pairs,
        T2 | LAMBDA2 | LAMBDA3 | LAMBDA4 => Sym::Columns,
        _ => Sym::Ordered,
    }
}

/// Return the spin-free classes produced by one spin-orbital class.
/// Spin-orbital `CAToAV` covers spin-free `CAToAV` and `CAToVA`.
/// # Arguments:
/// - `class`: Spin-orbital class name.
/// # Returns:
/// - `Vec<(&'static str, Vec<Space>)>`: Spin-free class names with free-index spaces.
fn class_layouts(class: &str) -> Vec<(&'static str, Vec<Space>)> {
    specs::EXCS
        .iter()
        .filter(|x| x.name == class || (class == "CAToAV" && x.name == "CAToVA"))
        .map(|x| (x.name, x.f.iter().map(|&n| specs::index_space(n)).collect()))
        .collect()
}

/// Expand one spin-orbital factor with fixed spins into spin-free factors.
/// # Arguments:
/// - `f`: Spin-orbital factor.
/// - `spin`: Spin bit of every index.
/// # Returns:
/// - `SmallVec<[(Ratio<i64>, Raw); 4]>`: Coefficients and spin-free factors.
fn expand_spin_block(
    f: &Factor,
    spin: &[u8],
) -> SmallVec<[(Ratio<i64>, Raw); 4]> {
    let bits = |xs: &[u16]| {
        xs.iter()
            .enumerate()
            .fold(0u8, |b, (i, &x)| b | (spin[x as usize] << i))
    };
    let (upper, lower) = (bits(&f.upper), bits(&f.lower));
    let permute = |rho: &[u8]| rho.iter().map(|&i| f.lower[i as usize]).collect();

    // One-body blocks are diagonal in spin.
    let k = f.upper.len();
    let so = |x: so::Kind| x as u8 == f.kind;
    if k == 1 && !so(so::Kind::Gamma) && !so(so::Kind::Eta) {
        if upper != lower {
            return SmallVec::new();
        }
        let name = match f.kind {
            x if x == so::Kind::Bra as u8 => BRA,
            x if x == so::Kind::Ket as u8 => KET,
            x if x == so::Kind::Fock as u8 => FOCK,
            _ => T1,
        };
        return SmallVec::from_elem(
            (
                Ratio::from_integer(1),
                (name, f.upper.clone(), f.lower.clone()),
            ),
            1,
        );
    }

    // Densities and cumulants follow the exact spin-ensemble replacements.
    if so(so::Kind::Gamma) || so(so::Kind::Eta) || k >= 2 && is_cumulant(f.kind) {
        let name = match f.kind {
            x if x == so::Kind::Gamma as u8 => GAMMA,
            x if x == so::Kind::Eta as u8 => THETA,
            x if x == so::Kind::Lambda2 as u8 => LAMBDA2,
            x if x == so::Kind::Lambda3 as u8 => LAMBDA3,
            _ => LAMBDA4,
        };
        return rules::cumulant_replacement(k, upper, lower)
            .iter()
            .map(|(rho, c)| (*c, (name, f.upper.clone(), permute(rho))))
            .collect();
    }

    // Antisymmetrised two-body blocks become direct minus exchange.
    let name = match f.kind {
        x if x == so::Kind::Bra as u8 => BRA,
        x if x == so::Kind::Ket as u8 => KET,
        x if x == so::Kind::Eri as u8 => ERI,
        _ => T2,
    };
    rules::pair_replacement(upper, lower)
        .into_iter()
        .map(|(rho, c)| (c, (name, f.upper.clone(), permute(&rho))))
        .collect()
}

/// Test whether a spin-orbital kind is a cumulant of rank two or more.
/// # Arguments:
/// - `k`: Spin-orbital kind id.
/// # Returns:
/// - `bool`: Whether the kind is `\lambda_2`, `\lambda_3` or `\lambda_4`.
fn is_cumulant(k: u8) -> bool {
    k == so::Kind::Lambda2 as u8 || k == so::Kind::Lambda3 as u8 || k == so::Kind::Lambda4 as u8
}

/// Enumerate every spin assignment of a term with the first index fixed to alpha.
/// Spin-orbital tensors conserve spin, so each tensor needs equal numbers of beta indices in
/// its upper and lower slots. The global spin flip leaves every replacement unchanged, so the
/// omitted half of the assignments contributes the same as the enumerated half.
/// # Arguments:
/// - `n`: Number of indices.
/// - `factors`: Spin-orbital factors.
/// # Returns:
/// - `Vec<Vec<u8>>`: Spin bits of every index for each assignment.
fn spin_assignments(
    n: usize,
    factors: &[Factor],
) -> Vec<Vec<u8>> {
    // Check each factor once its last index is assigned.
    let mut last = vec![Vec::new(); n];
    for (m, f) in factors.iter().enumerate() {
        if let Some(&x) = f.upper.iter().chain(&f.lower).max() {
            last[x as usize].push(m);
        }
    }

    let mut out = Vec::new();
    let mut spin = vec![0u8; n];
    let conserved = |f: &Factor, spin: &[u8]| {
        let count = |xs: &[u16]| xs.iter().filter(|&&x| spin[x as usize] == 1).count();
        count(&f.upper) == count(&f.lower)
    };

    let mut stack = vec![(0usize, 0u8)];
    while let Some((i, s)) = stack.pop() {
        if i == n {
            continue;
        }
        spin[i] = s;
        if !last[i].iter().all(|&m| conserved(&factors[m], &spin)) {
            continue;
        }
        if i + 1 == n {
            out.push(spin.clone());
            continue;
        }
        stack.push((i + 1, 1));
        stack.push((i + 1, 0));
    }

    out
}

/// Spin-adapt one spin-orbital residual into the spin-free residuals it produces.
/// # Arguments:
/// - `class`: Spin-orbital excitation class name.
/// - `expr`: Canonical spin-orbital residual.
/// # Returns:
/// - `Vec<Table>`: Spin-free residual of every class covered by `class`.
pub(crate) fn adapt_residual(
    class: &str,
    expr: &so::Expr,
) -> Vec<Table> {
    let layouts = class_layouts(class);
    let outputs = layouts
        .iter()
        .map(|(_, free)| vec![free.clone()])
        .collect::<Vec<_>>();

    layouts
        .into_iter()
        .zip(accumulate_outputs(&outputs, expr))
        .map(|((name, free), terms)| Table { name, free, terms })
        .collect()
}

/// Spin-adapt one spin-orbital metric block for one pair of spin-free classes.
/// Free indices are those of the left class followed by those of the right class.
/// # Arguments:
/// - `name`: Metric block name.
/// - `left`: Left spin-free excitation class name.
/// - `right`: Right spin-free excitation class name.
/// - `expr`: Canonical spin-orbital metric of the covering spin-orbital classes.
/// # Returns:
/// - `Table`: Spin-free metric block.
/// # Panics
/// - Panics if either class is unknown.
pub(crate) fn adapt_metric_block(
    name: &'static str,
    left: &str,
    right: &str,
    expr: &so::Expr,
) -> Table {
    let layout = |class: &str| {
        specs::EXCS
            .iter()
            .find(|x| x.name == class)
            .map(|x| {
                x.f.iter()
                    .map(|&n| specs::index_space(n))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| panic!("unknown excitation class {class}"))
    };
    let outputs = vec![vec![layout(left), layout(right)]];
    let terms = accumulate_outputs(&outputs, expr).pop().unwrap_or_default();

    Table {
        name,
        free: outputs[0].concat(),
        terms,
    }
}

/// Spin-adapt one spin-orbital scalar with no placeholders, such as an energy contribution.
/// # Arguments:
/// - `name`: Table name.
/// - `expr`: Canonical spin-orbital scalar expression.
/// # Returns:
/// - `Table`: Spin-free terms with no free indices.
pub(crate) fn adapt_scalar(
    name: &'static str,
    expr: &so::Expr,
) -> Table {
    let terms = accumulate_outputs(&[Vec::new()], expr)
        .pop()
        .unwrap_or_default();

    Table {
        name,
        free: Vec::new(),
        terms,
    }
}

/// Spin-adapt every term of one spin-orbital expression into the requested outputs.
/// # Arguments:
/// - `outputs`: Free-index layout of every placeholder, bra first, for each output.
/// - `expr`: Canonical spin-orbital expression.
/// # Returns:
/// - `Vec<FxHashMap<Key, Ratio<i64>>>`: Nonzero canonical terms of every output.
fn accumulate_outputs(
    outputs: &[Vec<Vec<Space>>],
    expr: &so::Expr,
) -> Vec<FxHashMap<Key, Ratio<i64>>> {
    // Spin-adapt terms in parallel, keeping one accumulator per output.
    let mut maps = expr
        .par_iter()
        .fold(
            || vec![FxHashMap::<Key, Ratio<i64>>::default(); outputs.len()],
            |mut acc, (key, &coeff)| {
                let spaces = key
                    .dummies
                    .iter()
                    .map(|&s| space_from_id(s))
                    .collect::<Vec<_>>();
                for (raw, c) in sum_over_spins(key, coeff) {
                    bind_placeholders(outputs, &spaces, &raw, c, &mut acc);
                }
                acc
            },
        )
        .reduce(
            || vec![FxHashMap::default(); outputs.len()],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    for (k, c) in y {
                        *x.entry(k).or_insert_with(|| Ratio::from_integer(0)) += c;
                    }
                }
                a
            },
        );

    for terms in &mut maps {
        terms.retain(|_, c| *c != Ratio::from_integer(0));
    }
    maps
}

/// Sum one spin-orbital term over spins, merging identical spin-free products.
/// # Arguments:
/// - `key`: Canonical spin-orbital term.
/// - `coeff`: Term coefficient.
/// # Returns:
/// - `FxHashMap<Vec<Raw>, Ratio<i64>>`: Spin-free products over the term's index ids.
fn sum_over_spins(
    key: &Key,
    coeff: Ratio<i64>,
) -> FxHashMap<Vec<Raw>, Ratio<i64>> {
    let n = key.dummies.len();
    let mut out = FxHashMap::<Vec<Raw>, Ratio<i64>>::default();

    // Both spins of the first index contribute equally.
    let scale = coeff * Ratio::from_integer(if n > 0 { 2 } else { 1 });

    for spin in spin_assignments(n, &key.factors) {
        let mut prods = vec![(scale, Vec::<Raw>::with_capacity(key.factors.len()))];

        for f in &key.factors {
            let ex = expand_spin_block(f, &spin);
            prods = prods
                .into_iter()
                .flat_map(|(c, raw)| {
                    ex.iter().map(move |(d, x)| {
                        let mut next = raw.clone();
                        next.push(x.clone());
                        (c * *d, next)
                    })
                })
                .collect();
            if prods.is_empty() {
                break;
            }
        }

        for (c, mut raw) in prods {
            raw.sort_unstable();
            *out.entry(raw).or_insert_with(|| Ratio::from_integer(0)) += c;
        }
    }

    out.retain(|_, c| *c != Ratio::from_integer(0));
    out
}

/// Bind the placeholders of one spin-free product to every compatible output layout.
/// Each placeholder orientation, `X^{pq}_{rs}` and for doubles its pair swap `X^{qp}_{sr}`,
/// whose index spaces match the output layout of that placeholder contributes the product
/// with its indices bound as free indices, bra first. An index shared by two placeholders is
/// bound at both positions, joined by a Kronecker delta.
/// # Arguments:
/// - `outputs`: Free-index layout of every placeholder, bra first, for each output.
/// - `all`: Orbital space of every spin-orbital index id.
/// - `raw`: Spin-free product over spin-orbital index ids.
/// - `c`: Product coefficient.
/// - `acc`: Per-output canonical accumulators.
/// # Returns:
/// - `()`: Mutates `acc`.
fn bind_placeholders(
    outputs: &[Vec<Vec<Space>>],
    all: &[Space],
    raw: &[Raw],
    c: Ratio<i64>,
    acc: &mut [FxHashMap<Key, Ratio<i64>>],
) {
    let holders = [BRA, KET]
        .iter()
        .filter_map(|&k| raw.iter().position(|f| f.0 == k))
        .collect::<SmallVec<[usize; 2]>>();
    let spaces = |xs: &[u16]| xs.iter().map(|&x| all[x as usize]).collect::<Vec<_>>();

    // Candidate free-index orders of each placeholder, created indices of the excitation first:
    // the projector `\tau^\dagger` lists them upper and the excitation `\tau` lower. Doubles
    // also bind their pair swap.
    let orders = holders
        .iter()
        .map(|&m| {
            let (kind, upper, lower) = &raw[m];
            let (x, y) = if *kind == BRA {
                (upper, lower)
            } else {
                (lower, upper)
            };
            let mut out = vec![[x.as_slice(), y.as_slice()].concat()];
            if x.len() == 2 {
                out.push(vec![x[1], x[0], y[1], y[0]]);
            }
            out
        })
        .collect::<Vec<_>>();

    for (n, layouts) in outputs.iter().enumerate() {
        if layouts.len() != holders.len() {
            continue;
        }

        // A scalar has no placeholders and binds once, with no free indices.
        let choices = if orders.is_empty() {
            vec![Vec::new()]
        } else {
            orders.iter().multi_cartesian_product().collect()
        };
        for choice in choices {
            if !choice
                .iter()
                .zip(layouts)
                .all(|(free, l)| spaces(free) == *l)
            {
                continue;
            }

            // Free indices take ids `0..n` in layout order; a repeated index gets a new id
            // joined to its first occurrence by a delta. The rest are dummies.
            let free = choice
                .iter()
                .flat_map(|x| x.iter().copied())
                .collect::<Vec<_>>();
            let mut rename = vec![u16::MAX; all.len()];
            let mut deltas = Vec::new();
            for (i, &x) in free.iter().enumerate() {
                if rename[x as usize] == u16::MAX {
                    rename[x as usize] = i as u16;
                } else {
                    deltas.push((rename[x as usize], i as u16));
                }
            }
            let mut next = free.len() as u16;
            let mut form_spaces = layouts
                .concat()
                .iter()
                .map(|&s| s as u8)
                .collect::<Vec<_>>();
            for (x, &s) in all.iter().enumerate() {
                if rename[x] == u16::MAX {
                    rename[x] = next;
                    next += 1;
                    form_spaces.push(s as u8);
                }
            }

            let factors = raw
                .iter()
                .enumerate()
                .filter(|&(m, _)| !holders.contains(&m))
                .map(|(_, (k, u, l))| Factor {
                    kind: *k,
                    sym: slot_symmetry(*k),
                    upper: u.iter().map(|&x| rename[x as usize]).collect(),
                    lower: l.iter().map(|&x| rename[x as usize]).collect(),
                })
                .chain(deltas.iter().map(|&(a, b)| Factor {
                    kind: DELTA,
                    sym: slot_symmetry(DELTA),
                    upper: SmallVec::from_elem(a, 1),
                    lower: SmallVec::from_elem(b, 1),
                }))
                .collect();
            let form = Form {
                spaces: form_spaces,
                nfree: free.len(),
                factors,
            };
            let (key, sign) = canon::canonical_key(&form);

            if sign != 0 {
                *acc[n].entry(key).or_insert_with(|| Ratio::from_integer(0)) +=
                    c * Ratio::from_integer(sign as i64);
            }
        }
    }
}

/// Return the orbital space of one canonical space id.
/// # Arguments:
/// - `s`: Space id as stored in canonical keys.
/// # Returns:
/// - `Space`: Orbital space.
fn space_from_id(s: u8) -> Space {
    match s {
        0 => Space::Core,
        1 => Space::Active,
        _ => Space::Virtual,
    }
}
