// residual.rs

use num_rational::Ratio;

use crate::canonical;
use crate::hamiltonian;
use crate::specs;
use crate::ir::{Expr, Product, Rational, Tensor, Term};
use crate::wick;

type Rat = Ratio<i64>;

/// Generate the zeroth-order residual for one excitation class.
/// # Arguments:
/// - `name`: Excitation class name.
/// # Returns:
/// - `Expr`: Canonical zeroth-order residual.
pub fn r0(name: &str) -> Expr {
    let spec = specs::exc(name);
    let mut out = Vec::new();

    for b in specs::BLOCKS.iter().filter(|b| b.left == spec.class) {
        let lspec = specs::Exc { class: b.left, f: b.lf };
        let bra = specs::bra(&lspec, 0);
        let h = hamiltonian::term(b.rf, 1);
        let p = join(&bra, &h.op);
        let e = wick::eval(&p);

        for x in e {
            out.push(mul_h(x, h.coeff, h.fac.clone()));
        }
    }

    canonical::canon(out)
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
fn mul_h(mut x: Term, c: Rational, fac: Tensor) -> Term {
    let a = Rat::new(x.coeff.num, x.coeff.den);
    let b = Rat::new(c.num, c.den);
    let q = a * b;

    x.coeff = Rational { num: *q.numer(), den: *q.denom() };
    x.tensors.push(fac);
    x
}
