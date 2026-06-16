// residual.rs

use num_rational::Ratio;
use rayon::prelude::*;

use crate::canonical;
use crate::hamiltonian;
use crate::specs;
use crate::cluster;
use crate::ir::{Expr, Product, Rational, Tensor, Term};
use crate::wick;

type Rat = Ratio<i64>;

/// Build a rational coefficient.
/// # Arguments:
/// - `n`: Numerator.
/// - `d`: Denominator.
/// # Returns:
/// - `Rational`: Rational coefficient.
fn q(n: i64, d: i64) -> Rational {
    Rational { num: n, den: d }
}

/// Join two products.
/// # Arguments:
/// - `a`: Left product.
/// - `b`: Right product.
/// # Returns:
/// - `Product`: Concatenated product.
fn join(a: &Product, b: &Product) -> Product {
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
fn mulh(mut x: Term, c: Rational, fac: Tensor) -> Term {
    let a = Rat::new(x.coeff.num, x.coeff.den);
    let b = Rat::new(c.num, c.den);
    let q = a * b;

    x.coeff = Rational { num: *q.numer(), den: *q.denom() };
    x.tensors.push(fac);
    x
}

/// Multiply two rational coefficients.
/// # Arguments:
/// - `a`: First coefficient.
/// - `b`: Second coefficient.
/// # Returns:
/// - `Rational`: Product coefficient.
fn mulr(a: Rational, b: Rational) -> Rational {
    let x = Rat::new(a.num, a.den);
    let y = Rat::new(b.num, b.den);
    let q = x * y;

    Rational { num: *q.numer(), den: *q.denom() }
}

/// Canonicalise one chunk from parallel Hamiltonian-term contributions.
/// # Arguments:
/// - `items`: Hamiltonian terms.
/// - `make`: Function generating terms for one Hamiltonian term.
/// # Returns:
/// - `Expr`: Canonical chunk expression.
fn hchunk<T: Sync>(items: &[T], make: impl Fn(&T) -> Expr + Sync) -> Expr {
    items.par_iter()
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

/// Generate zeroth-order residual chunks for one excitation class.
/// # Arguments:
/// - `name`: Excitation class name.
/// - `emit`: Callback receiving `(chunk_key, expression)`.
/// # Returns:
/// - `()`: Calls `emit` once per non-empty chunk.
pub fn r0(name: &str, mut emit: impl FnMut(String, Expr)) {
    let spec = specs::exc(name);
    let blocks: Vec<_> = specs::BLOCKS.iter()
        .filter(|b| b.left == spec.class)
        .collect();
    let prog = crate::progress::Prog::new(format!("residual::r0({name}) blocks"), blocks.len());

    let chunks: Vec<_> = blocks.par_iter()
        .filter_map(|b| {
            let lspec = specs::Exc { class: b.left, f: b.lf };
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
pub fn r1(name: &str, mut emit: impl FnMut(String, Expr)) {
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

/// Generate second-order residual chunks for one excitation class.
/// # Arguments:
/// - `name`: Excitation class name.
/// - `emit`: Callback receiving `(chunk_key, expression)`.
/// # Returns:
/// - `()`: Calls `emit` once per non-empty chunk.
pub fn r2(name: &str, mut emit: impl FnMut(String, Expr)) {
    let spec = specs::exc(name);
    let bra = specs::bra(&spec, 0);
    let brab = specs::bal(spec.f, true);
    let left = cluster::terms(2, 'l');
    let right = cluster::terms(3, 'r');
    let total = left.len() * right.len();
    let prog = crate::progress::Prog::new(format!("residual::r2({name}) T-pairs"), total);

    for (li, l) in left.iter().enumerate() {
        for (ri, r) in right.iter().enumerate() {
            let tb = specs::add(l.balance, r.balance);
            let req = specs::neg(specs::add(brab, tb));
            let hs = hamiltonian::terms_with_balance(1, req);

            let e = hchunk(&hs, |h| {
                let p = join(&join(&join(&bra, &h.op), &l.op), &r.op);
                let mut out = Vec::new();

                for x in wick::evalc(&p) {
                    let x = mulh(x, h.coeff, h.fac.clone());
                    let x = mulh(x, l.coeff, l.fac.clone());
                    let c = mulr(q(1, 2), r.coeff);
                    out.push(mulh(x, c, r.fac.clone()));
                }

                out
            });

            if !e.is_empty() {
                emit(format!("l{li}_r{ri}"), e);
            }

            prog.tick();
        }
    }
}
