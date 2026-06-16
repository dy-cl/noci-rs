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

/// Generate the zeroth-order residual for one excitation class.
/// # Arguments:
/// - `name`: Excitation class name.
/// # Returns:
/// - `Expr`: Canonical zeroth-order residual.
pub fn r0(name: &str) -> Expr {
    let spec = specs::exc(name);
    let blocks: Vec<_> = specs::BLOCKS.iter()
        .filter(|b| b.left == spec.class)
        .collect();
    let prog = crate::progress::Prog::new(format!("residual::r0({name}) blocks"), blocks.len());

    blocks.par_iter()
        .fold(canonical::Acc::new, |mut acc, b| {
            let lspec = specs::Exc { class: b.left, f: b.lf };
            let bra = specs::bra(&lspec, 0);
            let h = hamiltonian::term(b.rf, 1);
            let p = join(&bra, &h.op);

            for x in wick::eval(&p) {
                acc.addterm(mulh(x, h.coeff, h.fac.clone()));
            }

            prog.tick();
            acc
        })
        .reduce(canonical::Acc::new, |mut a, b| {
            a.merge(b);
            a
        })
        .finish()
}

/// Generate the first-order residual for one excitation class.
/// # Arguments:
/// - `name`: Excitation class name.
/// # Returns:
/// - `Expr`: Canonical first-order residual.
pub fn r1(name: &str) -> Expr {
    let spec = specs::exc(name);
    let bra = specs::bra(&spec, 0);
    let brab = specs::bal(spec.f, true);
    let terms = cluster::terms(2, 't');
    let prog = crate::progress::Prog::new(format!("residual::r1({name}) T terms"), terms.len());

    terms.par_iter()
        .fold(canonical::Acc::new, |mut acc, t| {
            let req = specs::neg(specs::add(brab, t.balance));

            for h in hamiltonian::terms_with_balance(1, req) {
                let p = join(&join(&bra, &h.op), &t.op);

                for x in wick::evalc(&p) {
                    let x = mulh(x, h.coeff, h.fac.clone());
                    acc.addterm(mulh(x, t.coeff, t.fac.clone()));
                }
            }

            prog.tick();
            acc
        })
        .reduce(canonical::Acc::new, |mut a, b| {
            a.merge(b);
            a
        })
        .finish()
}

/// Generate the second-order residual for one excitation class.
/// # Arguments:
/// - `name`: Excitation class name.
/// # Returns:
/// - `Expr`: Canonical second-order residual.
pub fn r2(name: &str) -> Expr {
    let spec = specs::exc(name);
    let bra = specs::bra(&spec, 0);
    let brab = specs::bal(spec.f, true);
    let left = cluster::terms(2, 'l');
    let right = cluster::terms(3, 'r');

    let pairs: Vec<_> = left.iter()
        .flat_map(|l| right.iter().map(move |r| (l, r)))
        .collect();

    let prog = crate::progress::Prog::new(format!("residual::r2({name}) T-pairs"), pairs.len());

    pairs.par_iter()
        .fold(canonical::Acc::new, |mut acc, (l, r)| {
            let tb = specs::add(l.balance, r.balance);
            let req = specs::neg(specs::add(brab, tb));

            for h in hamiltonian::terms_with_balance(1, req) {
                let p = join(&join(&join(&bra, &h.op), &l.op), &r.op);

                for x in wick::evalc(&p) {
                    let x = mulh(x, h.coeff, h.fac.clone());
                    let x = mulh(x, l.coeff, l.fac.clone());
                    let c = mulr(q(1, 2), r.coeff);

                    acc.addterm(mulh(x, c, r.fac.clone()));
                }
            }

            prog.tick();
            acc
        })
        .reduce(canonical::Acc::new, |mut a, b| {
            a.merge(b);
            a
        })
        .finish()
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
