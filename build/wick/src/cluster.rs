// cluster.rs

use crate::gno::{e1, e2};
use crate::ir::{Product, Rational, Tensor, TensorKind};
use crate::specs;

/// One cluster-operator term.
#[derive(Clone, Debug)]
pub struct TTerm {
    /// Scalar coefficient.
    pub coeff: Rational,
    /// Cluster-amplitude tensor.
    pub fac: Tensor,
    /// Spin-free operator product.
    pub op: Product,
    /// Orbital-space balance.
    pub balance: specs::Balance,
}

/// Build an integer rational.
/// # Arguments:
/// - `n`: Numerator.
/// # Returns:
/// - `Rational`: Integer coefficient.
fn r(n: i64) -> Rational {
    Rational { num: n, den: 1 }
}

/// Build a rational coefficient.
/// # Arguments:
/// - `n`: Numerator.
/// - `d`: Denominator.
/// # Returns:
/// - `Rational`: Rational coefficient.
fn q(n: i64, d: i64) -> Rational {
    Rational { num: n, den: d }
}

/// Build the cluster-amplitude coefficient for one excitation pattern.
/// # Arguments:
/// - `xs`: Free-index names.
/// # Returns:
/// - `(Rational, Tensor)`: Scalar prefactor and amplitude tensor.
pub fn fac(xs: &[&'static str]) -> (Rational, Tensor) {
    match xs.len() {
        2 => {
            let p = specs::idx(xs[0]);
            let q_ = specs::idx(xs[1]);

            (
                r(1),
                Tensor { kind: TensorKind::T1, upper: vec![q_], lower: vec![p] },
            )
        }
        4 => {
            let p = specs::idx(xs[0]);
            let q_ = specs::idx(xs[1]);
            let r_ = specs::idx(xs[2]);
            let s = specs::idx(xs[3]);

            (
                q(1, 2),
                Tensor { kind: TensorKind::T2, upper: vec![r_, s], lower: vec![p, q_] },
            )
        }
        _ => panic!("unsupported cluster rank"),
    }
}

/// Build the cluster operator product for one excitation pattern.
/// # Arguments:
/// - `xs`: Free-index names.
/// - `g`: Group id.
/// # Returns:
/// - `Product`: Spin-free cluster operator product.
pub fn op(xs: &[&'static str], g: usize) -> Product {
    match xs.len() {
        2 => {
            let p = specs::idx(xs[0]);
            let q_ = specs::idx(xs[1]);

            Product { groups: vec![e1(p, q_, g)] }
        }
        4 => {
            let p = specs::idx(xs[0]);
            let q_ = specs::idx(xs[1]);
            let r_ = specs::idx(xs[2]);
            let s = specs::idx(xs[3]);

            Product { groups: vec![e2(p, q_, r_, s, g)] }
        }
        _ => panic!("unsupported cluster rank"),
    }
}

/// Build one cluster term from free-index names.
/// # Arguments:
/// - `xs`: Free-index names.
/// - `g`: Group id.
/// # Returns:
/// - `TTerm`: Cluster term.
pub fn term(xs: &[&'static str], g: usize) -> TTerm {
    let (coeff, fac) = fac(xs);

    TTerm {
        coeff,
        fac,
        op: op(xs, g),
        balance: specs::bal(xs, false),
    }
}

/// Build all singles and doubles cluster terms.
/// # Arguments:
/// - `g`: Group id.
/// # Returns:
/// - `Vec<TTerm>`: Cluster terms.
pub fn terms(g: usize) -> Vec<TTerm> {
    specs::EXCS.iter()
        .map(|x| {
            let xs = specs::tlabels(x.f);
            term(&xs, g)
        })
        .collect()
}
