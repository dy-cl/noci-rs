// specs.rs

use crate::ir::{Group, Idx, Product, Space};
use crate::gno::{e1, e2};

/// One overlap block metadata row.
#[derive(Clone, Copy, Debug)]
pub struct BlockSpec {
    /// Block name.
    pub name: &'static str,
    /// Left excitation class.
    pub left: &'static str,
    /// Right excitation class.
    pub right: &'static str,
    /// Left free-index names.
    pub lf: &'static [&'static str],
    /// Right free-index names.
    pub rf: &'static [&'static str],
}

/// One excitation-class metadata row.
#[derive(Clone, Copy, Debug)]
pub struct ExcSpec {
    /// Excitation class name.
    pub name: &'static str,
    /// Free-index names in annihilator-first order.
    pub f: &'static [&'static str],
}

/// One concrete residual projector.
#[derive(Clone, Copy, Debug)]
pub struct Exc {
    /// Excitation class name.
    pub class: &'static str,
    /// Free-index names in annihilator-first order.
    pub f: &'static [&'static str],
}

pub const BLOCKS: &[BlockSpec] = &[
    BlockSpec { name: "C1", left: "CToA", right: "CToA", lf: &["u", "i"], rf: &["v", "j"] },
    BlockSpec { name: "C2", left: "AToV", right: "AToV", lf: &["a", "t"], rf: &["b", "u"] },
    BlockSpec { name: "C3", left: "AToA", right: "AToA", lf: &["v", "u"], rf: &["x", "w"] },
    BlockSpec { name: "C4", left: "CAToAV", right: "CAToAV", lf: &["v", "a", "i", "u"], rf: &["x", "b", "j", "w"] },
    BlockSpec { name: "C5", left: "CAToVA", right: "CAToVA", lf: &["a", "v", "i", "u"], rf: &["b", "x", "j", "w"] },
    BlockSpec { name: "C6", left: "CAToVV", right: "CAToVV", lf: &["a", "b", "i", "u"], rf: &["c", "d", "j", "v"] },
    BlockSpec { name: "C7", left: "CCToAV", right: "CCToAV", lf: &["u", "a", "i", "j"], rf: &["v", "b", "k", "l"] },
    BlockSpec { name: "C8", left: "CCToAA", right: "CCToAA", lf: &["u", "v", "i", "j"], rf: &["w", "x", "k", "l"] },
    BlockSpec { name: "C9", left: "CAToAA", right: "CAToAA", lf: &["v", "w", "i", "u"], rf: &["y", "z", "j", "x"] },
    BlockSpec { name: "C10", left: "AAToAV", right: "AAToAV", lf: &["v", "a", "t", "u"], rf: &["z", "b", "x", "y"] },
    BlockSpec { name: "C11", left: "AAToVV", right: "AAToVV", lf: &["a", "b", "t", "u"], rf: &["c", "d", "v", "w"] },
    BlockSpec { name: "C12", left: "AAToAA", right: "AAToAA", lf: &["q", "s", "p", "r"], rf: &["t", "v", "u", "w"] },
    BlockSpec { name: "C13", left: "AToV", right: "AAToAV", lf: &["a", "u"], rf: &["x", "b", "v", "w"] },
    BlockSpec { name: "C14", left: "CToA", right: "CAToAA", lf: &["u", "i"], rf: &["w", "x", "j", "v"] },
    BlockSpec { name: "C15", left: "AToA", right: "AAToAA", lf: &["u", "t"], rf: &["y", "z", "w", "x"] },
    BlockSpec { name: "C16", left: "CAToAV", right: "CAToVA", lf: &["w", "a", "i", "u"], rf: &["b", "y", "j", "x"] },
    BlockSpec { name: "C17", left: "CToV", right: "CToV", lf: &["a", "i"], rf: &["b", "j"] },
    BlockSpec { name: "C18", left: "CToV", right: "CAToAV", lf: &["a", "i"], rf: &["x", "b", "j", "w"] },
    BlockSpec { name: "C19", left: "CToV", right: "CAToVA", lf: &["a", "i"], rf: &["b", "x", "j", "w"] },
];

pub const EXCS: &[ExcSpec] = &[
    ExcSpec { name: "CToA", f: &["u", "i"] },
    ExcSpec { name: "AToV", f: &["a", "t"] },
    ExcSpec { name: "AToA", f: &["v", "u"] },
    ExcSpec { name: "CToV", f: &["a", "i"] },
    ExcSpec { name: "CAToAV", f: &["v", "a", "i", "u"] },
    ExcSpec { name: "CAToVA", f: &["a", "v", "i", "u"] },
    ExcSpec { name: "CAToVV", f: &["a", "b", "i", "u"] },
    ExcSpec { name: "CCToAV", f: &["u", "a", "i", "j"] },
    ExcSpec { name: "CCToAA", f: &["u", "v", "i", "j"] },
    ExcSpec { name: "CAToAA", f: &["v", "w", "i", "u"] },
    ExcSpec { name: "AAToAV", f: &["v", "a", "t", "u"] },
    ExcSpec { name: "AAToVV", f: &["a", "b", "t", "u"] },
    ExcSpec { name: "AAToAA", f: &["q", "s", "p", "r"] },
];

/// Find one overlap block specification.
/// # Arguments:
/// - `name`: Block name.
/// # Returns:
/// - `BlockSpec`: Matching block specification.
pub fn block(name: &str) -> BlockSpec {
    *BLOCKS.iter()
        .find(|x| x.name == name)
        .unwrap_or_else(|| panic!("unknown block {name}"))
}

/// Return one residual projector.
/// # Arguments:
/// - `name`: Excitation class name or metric block name.
/// # Returns:
/// - `Exc`: Excitation projector.
pub fn exc(name: &str) -> Exc {
    if let Some(x) = EXCS.iter().find(|x| x.name == name) {
        return Exc { class: x.name, f: x.f };
    }

    let x = block(name);
    Exc { class: x.left, f: x.lf }
}

/// Return all excitation classes with a given rank.
/// # Arguments:
/// - `rank`: Excitation rank.
/// # Returns:
/// - `Vec<ExcSpec>`: Matching classes.
pub fn classes(rank: usize) -> Vec<ExcSpec> {
    EXCS.iter()
        .copied()
        .filter(|x| x.f.len() == 2 * rank)
        .collect()
}

/// Build the daggered residual projector.
/// # Arguments:
/// - `x`: Excitation projector.
/// - `g`: GNO group id.
/// # Returns:
/// - `Product`: One-group projector product.
pub fn bra(x: &Exc, g: usize) -> Product {
    Product { groups: vec![left(x.class, x.f, g)] }
}

/// Build a left metric group.
/// # Arguments:
/// - `class`: Excitation class name.
/// - `f`: Free-index names.
/// - `g`: GNO group id.
/// # Returns:
/// - `Group`: Spin-expanded GNO group.
pub fn left(class: &str, f: &[&'static str], g: usize) -> Group {
    match f.len() {
        2 => e1(idx(f[1]), idx(f[0]), g),
        4 => e2(idx(f[2]), idx(f[3]), idx(f[0]), idx(f[1]), g),
        _ => panic!("unsupported left excitation {class}"),
    }
}

/// Build a right metric group.
/// # Arguments:
/// - `class`: Excitation class name.
/// - `f`: Free-index names.
/// - `g`: GNO group id.
/// # Returns:
/// - `Group`: Spin-expanded GNO group.
pub fn right(class: &str, f: &[&'static str], g: usize) -> Group {
    match f.len() {
        2 => e1(idx(f[0]), idx(f[1]), g),
        4 => e2(idx(f[0]), idx(f[1]), idx(f[2]), idx(f[3]), g),
        _ => panic!("unsupported right excitation {class}"),
    }
}

/// Build one metric block product from metadata.
/// # Arguments:
/// - `name`: Block name.
/// # Returns:
/// - `Product`: GNO product for this metric block.
pub fn product(name: &str) -> Product {
    let x = block(name);

    Product {
        groups: vec![
            left(x.left, x.lf, 0),
            right(x.right, x.rf, 1),
        ],
    }
}

/// Infer index space from symbolic name.
/// # Arguments:
/// - `name`: Index name.
/// # Returns:
/// - `Space`: Orbital space.
pub fn space(name: &str) -> Space {
    match name {
        "i" | "j" | "k" | "l" => Space::Core,
        "a" | "b" | "c" | "d" => Space::Virtual,
        _ => Space::Active,
    }
}

/// Construct an index from a name.
/// # Arguments:
/// - `name`: Index name.
/// # Returns:
/// - `Idx`: Index with inferred space.
pub fn idx(name: &'static str) -> Idx {
    Idx { name, space: space(name) }
}
