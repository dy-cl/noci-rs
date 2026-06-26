// residual.rs

use num_rational::Ratio;
use rayon::prelude::*;

use crate::canonical;
use crate::cluster;
use crate::hamiltonian;
use crate::ir::{Expr, Product, Rational, Tensor, TensorKind, Term};
use crate::specs;
use crate::wick;

/// Build a rational coefficient.
/// # Arguments:
/// - `n`: Numerator.
/// - `d`: Denominator.
/// # Returns:
/// - `Rational`: Rational coefficient.
fn q(
    n: i64,
    d: i64,
) -> Rational {
    Rational { num: n, den: d }
}

/// Join two products.
/// # Arguments:
/// - `a`: Left product.
/// - `b`: Right product.
/// # Returns:
/// - `Product`: Concatenated product.
fn join(
    a: &Product,
    b: &Product,
) -> Product {
    let mut groups = a.groups.clone();
    groups.extend(b.groups.clone());
    Product { groups }
}

/// Multiply one Wick term by a Hamiltonian coefficient tensor.
/// # Arguments:
/// - `x`: Wick term.
/// - `c`: Scalar coefficient.
/// - `fac`: Hamiltonian coefficient tensor.
/// # Returns:
/// - `Term`: Residual term.
fn mulh(
    mut x: Term,
    c: Rational,
    fac: Tensor,
) -> Term {
    let a = Ratio::<i64>::new(x.coeff.num, x.coeff.den);
    let b = Ratio::<i64>::new(c.num, c.den);
    let q = a * b;

    x.coeff = Rational {
        num: *q.numer(),
        den: *q.denom(),
    };
    x.tensors.push(fac);
    x
}

/// Multiply two rational coefficients.
/// # Arguments:
/// - `a`: First coefficient.
/// - `b`: Second coefficient.
/// # Returns:
/// - `Rational`: Product coefficient.
fn mulr(
    a: Rational,
    b: Rational,
) -> Rational {
    let x = Ratio::<i64>::new(a.num, a.den);
    let y = Ratio::<i64>::new(b.num, b.den);
    let q = x * y;

    Rational {
        num: *q.numer(),
        den: *q.denom(),
    }
}

/// Canonicalise one chunk from parallel Hamiltonian-term contributions.
/// # Arguments:
/// - `items`: Hamiltonian terms.
/// - `make`: Function generating terms for one Hamiltonian term.
/// # Returns:
/// - `Expr`: Canonical chunk expression.
fn hchunk<T: Sync>(
    items: &[T],
    make: impl Fn(&T) -> Expr + Sync,
) -> Expr {
    items
        .par_iter()
        .fold(canonical::Acc::new, |mut acc, h| {
            for x in make(h) {
                acc.addterm(x);
            }

            acc
        })
        .reduce(canonical::Acc::new, |mut a, b| {
            a.merge(b);
            a
        })
        .finish()
}

/// Number of Hamiltonian terms to evaluate in parallel per batch.
/// # Arguments:
/// - None.
/// # Returns:
/// - `usize`: H-term batch size.
fn hbatch() -> usize {
    std::env::var("WICK_H_BATCH")
        .ok()
        .and_then(|x| x.parse::<usize>().ok())
        .filter(|&x| x > 0)
        .unwrap_or(4)
}

/// Build a stable Hamiltonian-term chunk key.
/// # Arguments:
/// - `h`: Hamiltonian term.
/// # Returns:
/// - `String`: Hamiltonian chunk key.
fn hkey(h: &hamiltonian::HTerm) -> String {
    let kind = match h.fac.kind {
        TensorKind::Fock => "f",
        TensorKind::ERI => "g",
        _ => "x",
    };
    let up = h
        .fac
        .upper
        .iter()
        .map(|x| x.name)
        .collect::<Vec<_>>()
        .join("_");
    let lo = h
        .fac
        .lower
        .iter()
        .map(|x| x.name)
        .collect::<Vec<_>>()
        .join("_");

    format!("{kind}_{up}_{lo}")
}

/// Generate zeroth-order residual chunks for one excitation class.
/// # Arguments:
/// - `name`: Excitation class name.
/// - `emit`: Callback receiving `(chunk_key, expression)`.
/// # Returns:
/// - `()`: Calls `emit` once per non-empty chunk.
pub fn r0(
    name: &str,
    mut emit: impl FnMut(String, Expr),
) {
    let spec = specs::exc(name);
    let blocks: Vec<_> = specs::BLOCKS
        .iter()
        .filter(|b| b.left == spec.class)
        .collect();
    let prog = crate::progress::Prog::new(format!("residual::r0({name}) blocks"), blocks.len());

    let chunks: Vec<_> = blocks
        .par_iter()
        .filter_map(|b| {
            let lspec = specs::Exc {
                class: b.left,
                f: b.lf,
            };
            let bra = specs::bra(&lspec, 0);
            let h = hamiltonian::term(b.rf, 1);
            let p = join(&bra, &h.op);
            let mut out = Vec::new();

            for x in wick::eval(&p) {
                out.push(mulh(x, h.coeff, h.fac.clone()));
            }

            let e = canonical::canon(out);
            prog.tick();

            if e.is_empty() {
                None
            } else {
                Some((b.name.to_string(), e))
            }
        })
        .collect();

    for (k, e) in chunks {
        emit(k, e);
    }
}

/// Generate first-order residual chunks for one excitation class.
/// # Arguments:
/// - `name`: Excitation class name.
/// - `emit`: Callback receiving `(chunk_key, expression)`.
/// # Returns:
/// - `()`: Calls `emit` once per non-empty chunk.
pub fn r1(
    name: &str,
    mut emit: impl FnMut(String, Expr),
) {
    let spec = specs::exc(name);
    let bra = specs::bra(&spec, 0);
    let brab = specs::bal(spec.f, true);
    let terms = cluster::terms(2, 't');
    let prog = crate::progress::Prog::new(format!("residual::r1({name}) T terms"), terms.len());

    for (ti, t) in terms.iter().enumerate() {
        let req = specs::neg(specs::add(brab, t.balance));
        let hs = hamiltonian::terms_with_balance(1, req);

        let e = hchunk(&hs, |h| {
            let p = join(&join(&bra, &h.op), &t.op);
            let mut out = Vec::new();

            for x in wick::evalc(&p) {
                let x = mulh(x, h.coeff, h.fac.clone());
                out.push(mulh(x, t.coeff, t.fac.clone()));
            }

            out
        });

        if !e.is_empty() {
            emit(format!("t{ti}"), e);
        }

        prog.tick();
    }
}

/// Check one split-pattern field.
/// # Arguments:
/// - `pat`: Pattern field.
/// - `x`: Value field.
/// # Returns:
/// - `bool`: True if the pattern matches.
fn splitfield(
    pat: &str,
    x: &str,
) -> bool {
    pat == "*" || pat == x
}

/// Check whether one split pattern matches an R2 H chunk.
/// # Arguments:
/// - `pat`: Pattern of the form `Class:lN:rN:hkey`.
/// - `name`: Excitation class name.
/// - `li`: Left cluster-term index.
/// - `ri`: Right cluster-term index.
/// - `h`: Hamiltonian key.
/// # Returns:
/// - `bool`: True if the pattern matches.
fn splitpat(
    pat: &str,
    name: &str,
    li: usize,
    ri: usize,
    h: &str,
) -> bool {
    let pat = pat.trim();

    if pat.is_empty() {
        return false;
    }

    let xs = pat.split(':').collect::<Vec<_>>();

    if xs.len() != 4 {
        return false;
    }

    let l = format!("l{li}");
    let r = format!("r{ri}");

    splitfield(xs[0], name)
        && splitfield(xs[1], &l)
        && splitfield(xs[2], &r)
        && splitfield(xs[3], h)
}

/// Decide whether one R2 H chunk should be split by spin batches.
/// # Arguments:
/// - `name`: Excitation class name.
/// - `li`: Left cluster-term index.
/// - `ri`: Right cluster-term index.
/// - `h`: Hamiltonian key.
/// # Returns:
/// - `bool`: True if this chunk should be spin-batched.
fn splitr2hterm(
    name: &str,
    li: usize,
    ri: usize,
    h: &str,
) -> bool {
    if std::env::var_os("WICK_SPLIT_ALL_SPINS").is_some() {
        return true;
    }

    if std::env::var_os("WICK_NO_SPIN_SPLIT").is_some() {
        return false;
    }

    let pats = std::env::var("WICK_SPIN_SPLIT_CHUNKS")
        .unwrap_or_else(|_| "AAToAA:l12:r12:g_ha2_ha3_ha0_ha1".to_string());

    pats.split(',').any(|pat| splitpat(pat, name, li, ri, h))
}

/// Generate one second-order residual Hamiltonian subchunk.
/// # Arguments:
/// - `base`: Base chunk key.
/// - `split`: Whether to split this product by spin-string batches.
/// - `bra`: Residual bra product.
/// - `l`: Left cluster term.
/// - `r`: Right cluster term.
/// - `h`: Hamiltonian term.
/// - `emit`: Callback receiving `(chunk_key, expression)`.
/// # Returns:
/// - `()`: Calls `emit` once per non-empty generated subchunk.
fn r2hterm(
    base: String,
    split: bool,
    bra: &Product,
    l: &cluster::TTerm,
    r: &cluster::TTerm,
    h: &hamiltonian::HTerm,
    mut emit: impl FnMut(String, Expr) + Send,
) {
    let p = join(&join(&join(bra, &h.op), &l.op), &r.op);

    wick::evalcstream(&p, split, |si, split, e| {
        let key = if split {
            format!("{base}_s{si}")
        } else {
            base.clone()
        };

        if split && std::env::var_os("WICK_CANON_SPLIT").is_none() {
            let mut out = Vec::with_capacity(e.len());

            for x in e {
                let x = mulh(x, h.coeff, h.fac.clone());
                let x = mulh(x, l.coeff, l.fac.clone());
                let c = mulr(q(1, 2), r.coeff);

                out.push(mulh(x, c, r.fac.clone()));
            }

            if !out.is_empty() {
                emit(key, out);
            }

            return;
        }

        let mut acc = canonical::Acc::new();

        for x in e {
            let x = mulh(x, h.coeff, h.fac.clone());
            let x = mulh(x, l.coeff, l.fac.clone());
            let c = mulr(q(1, 2), r.coeff);

            acc.addterm(mulh(x, c, r.fac.clone()));
        }

        let e = acc.finish();

        if e.is_empty() {
            return;
        }

        emit(key, e);
    });
}

/// Generate second-order residual chunks for one excitation class.
/// # Arguments:
/// - `name`: Excitation class name.
/// - `emit`: Callback receiving `(chunk_key, expression)`.
/// # Returns:
/// - `()`: Calls `emit` once per non-empty chunk.
pub fn r2(
    name: &str,
    mut emit: impl FnMut(String, Expr) + Send,
) {
    let spec = specs::exc(name);
    let bra = specs::bra(&spec, 0);
    let brab = specs::bal(spec.f, true);
    let left = cluster::terms(2, 'l');
    let right = cluster::terms(3, 'r');
    let total = left.len() * right.len();
    let prog = crate::progress::Prog::new(format!("residual::r2({name}) T-pairs"), total);
    let batch = hbatch();

    for (li, l) in left.iter().enumerate() {
        for (ri, r) in right.iter().enumerate() {
            let tb = specs::add(l.balance, r.balance);
            let req = specs::neg(specs::add(brab, tb));
            let hs = hamiltonian::terms_with_balance(1, req);

            let mut normal = Vec::new();
            let mut split = Vec::new();

            for h in &hs {
                let hk = hkey(h);

                if splitr2hterm(name, li, ri, &hk) {
                    split.push((hk, h));
                } else {
                    normal.push((hk, h));
                }
            }

            for hs in normal.chunks(batch) {
                let chunks: Vec<_> = hs
                    .par_iter()
                    .flat_map(|(hk, h)| {
                        let base = format!("l{li}_r{ri}_h{hk}");
                        let mut out = Vec::new();

                        crate::progress::mem(format!("residual::r2({name}) start {base}"));

                        r2hterm(base.clone(), false, &bra, l, r, h, |k, e| {
                            crate::progress::mem(format!(
                                "residual::r2({name}) emit {k} terms: {}",
                                e.len()
                            ));
                            out.push((k, e));
                        });

                        crate::progress::mem(format!(
                            "residual::r2({name}) end {base} subchunks: {}",
                            out.len()
                        ));

                        out
                    })
                    .collect();

                for (k, e) in chunks {
                    emit(k, e);
                }
            }

            for (hk, h) in split {
                let base = format!("l{li}_r{ri}_h{hk}");
                let mut subchunks = 0usize;
                let start = std::time::Instant::now();

                crate::progress::mem(format!("residual::r2({name}) start {base}"));

                r2hterm(base.clone(), true, &bra, l, r, h, |k, e| {
                    subchunks += 1;
                    if std::env::var_os("WICK_PROGRESS").is_some() && subchunks % 100 == 0 {
                        eprintln!(
                            "[wick-time] residual::r2({name}) {base}: subchunks {subchunks}, elapsed {:?}.",
                            start.elapsed()
                        );
                    }
                    crate::progress::mem(format!(
                        "residual::r2({name}) emit {k} terms: {}",
                        e.len()
                    ));
                    emit(k, e);
                });

                crate::progress::mem(format!(
                    "residual::r2({name}) end {base} subchunks: {subchunks}"
                ));
            }

            prog.tick();
        }
    }
}
