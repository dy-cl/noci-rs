// hamiltonian.rs

use crate::gno::{e1, e2};
use crate::ir::{Product, Rational, Tensor, TensorKind, Space};
use crate::specs;

/// One Hamiltonian operator term.
#[derive(Clone, Debug)]
pub struct HTerm {
    /// Scalar coefficient.
    pub coeff: Rational,
    /// Hamiltonian coefficient tensor.
    pub fac: Tensor,
    /// Spin-free operator product.
    pub op: Product,
}

const SPACES: [crate::ir::Space; 3] = [
    crate::ir::Space::Core,
    crate::ir::Space::Active,
    crate::ir::Space::Virtual,
];

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

/// Build a Fock tensor factor.
/// # Arguments:
/// - `upper`: Upper index.
/// - `lower`: Lower index.
/// # Returns:
/// - `Tensor`: Fock tensor.
fn f(upper: crate::ir::Idx, lower: crate::ir::Idx) -> Tensor {
    Tensor { kind: TensorKind::Fock, upper: vec![upper], lower: vec![lower] }
}

/// Build an ERI tensor factor.
/// # Arguments:
/// - `u0`: First upper index.
/// - `u1`: Second upper index.
/// - `l0`: First lower index.
/// - `l1`: Second lower index.
/// # Returns:
/// - `Tensor`: ERI tensor.
fn eri(u0: crate::ir::Idx, u1: crate::ir::Idx, l0: crate::ir::Idx, l1: crate::ir::Idx) -> Tensor {
    Tensor { kind: TensorKind::ERI, upper: vec![u0, u1], lower: vec![l0, l1] }
}

/// Build the Hamiltonian coefficient for one excitation pattern.
/// # Arguments:
/// - `xs`: Free-index names.
/// # Returns:
/// - `(Rational, Tensor)`: Scalar prefactor and Hamiltonian tensor.
pub fn fac(xs: &[&'static str]) -> (Rational, Tensor) {
    match xs.len() {
        2 => {
            let p = specs::idx(xs[0]);
            let q_ = specs::idx(xs[1]);

            (
                r(1),
                f(q_, p),
            )
        }
        4 => {
            let p = specs::idx(xs[0]);
            let q_ = specs::idx(xs[1]);
            let r_ = specs::idx(xs[2]);
            let s = specs::idx(xs[3]);

            (
                q(1, 2),
                eri(r_, s, p, q_),
            )
        }
        _ => panic!("unsupported Hamiltonian rank"),
    }
}

/// Build the Hamiltonian operator product for one excitation pattern.
/// # Arguments:
/// - `xs`: Free-index names.
/// - `g`: Group id.
/// # Returns:
/// - `Product`: Spin-free Hamiltonian operator product.
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
        _ => panic!("unsupported Hamiltonian rank"),
    }
}

/// Build one Hamiltonian term from free-index names.
/// # Arguments:
/// - `xs`: Free-index names.
/// - `g`: Group id.
/// # Returns:
/// - `HTerm`: Hamiltonian term.
pub fn term(xs: &[&'static str], g: usize) -> HTerm {
    let (coeff, fac) = fac(xs);

    HTerm {
        coeff,
        fac,
        op: op(xs, g),
    }
}

/// Build Hamiltonian dummy labels from orbital spaces.
/// # Arguments:
/// - `xs`: Orbital spaces.
/// # Returns:
/// - `Vec<&'static str>`: Hamiltonian dummy labels.
fn hlabels(xs: &[Space]) -> Vec<&'static str> {
    xs.iter()
        .enumerate()
        .map(|(i, &x)| specs::hname(x, i))
        .collect()
}

/// Build general Hamiltonian terms with a specified orbital-space balance.
/// # Arguments:
/// - `g`: Group id.
/// - `required`: Required orbital-space balance.
/// # Returns:
/// - `Vec<HTerm>`: Matching one- and two-body Hamiltonian terms.
pub fn terms_with_balance(g: usize, required: specs::Balance) -> Vec<HTerm> {
    let mut out = Vec::new();

    for p in SPACES {
        for q_ in SPACES {
            let xs = hlabels(&[p, q_]);

            if specs::bal(&xs, false) == required {
                out.push(term(&xs, g));
            }
        }
    }

    for p in SPACES {
        for q_ in SPACES {
            for r_ in SPACES {
                for s in SPACES {
                    let xs = hlabels(&[p, q_, r_, s]);

                    if specs::bal(&xs, false) == required {
                        out.push(term(&xs, g));
                    }
                }
            }
        }
    }

    out
}

/// Build restricted one-body Hamiltonian terms.
/// # Arguments:
/// - `g`: Group id.
/// # Returns:
/// - `Vec<HTerm>`: One-body Hamiltonian terms.
pub fn h1terms(g: usize) -> Vec<HTerm> {
    specs::classes(1)
        .into_iter()
        .map(|x| term(x.f, g))
        .collect()
}

/// Build restricted two-body Hamiltonian terms.
/// # Arguments:
/// - `g`: Group id.
/// # Returns:
/// - `Vec<HTerm>`: Two-body Hamiltonian terms.
pub fn h2terms(g: usize) -> Vec<HTerm> {
    specs::classes(2)
        .into_iter()
        .map(|x| term(x.f, g))
        .collect()
}

/// Build all restricted Hamiltonian terms.
/// # Arguments:
/// - `g`: Group id.
/// # Returns:
/// - `Vec<HTerm>`: One- and two-body Hamiltonian terms.
pub fn terms(g: usize) -> Vec<HTerm> {
    let mut out = h1terms(g);
    out.extend(h2terms(g));
    out
}
