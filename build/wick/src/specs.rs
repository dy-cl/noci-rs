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
    if name.starts_with("hc") || name.starts_with("tc") {
        return Space::Core;
    }

    if name.starts_with("ha") || name.starts_with("ta") {
        return Space::Active;
    }

    if name.starts_with("hv") || name.starts_with("tv") {
        return Space::Virtual;
    }

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

/// Orbital-space balance vector in `(core, active, virtual)` order.
pub type Balance = [i8; 3];

/// Return the slot of an orbital space in a balance vector.
/// # Arguments:
/// - `x`: Orbital space.
/// # Returns:
/// - `usize`: Balance-vector slot.
fn sid(x: Space) -> usize {
    match x {
        Space::Core => 0,
        Space::Active => 1,
        Space::Virtual => 2,
    }
}

/// Add two orbital-space balances.
/// # Arguments:
/// - `a`: First balance.
/// - `b`: Second balance.
/// # Returns:
/// - `Balance`: Elementwise sum.
pub fn add(mut a: Balance, b: Balance) -> Balance {
    for i in 0..3 {
        a[i] += b[i];
    }

    a
}

/// Negate an orbital-space balance.
/// # Arguments:
/// - `a`: Input balance.
/// # Returns:
/// - `Balance`: Elementwise negation.
pub fn neg(a: Balance) -> Balance {
    [-a[0], -a[1], -a[2]]
}

/// Compute the orbital-space balance of an excitation pattern.
/// # Arguments:
/// - `xs`: Free-index names, with creation labels first and annihilation labels second.
/// - `daggered`: Whether to compute the balance of the daggered operator.
/// # Returns:
/// - `Balance`: Net `(core, active, virtual)` balance.
pub fn bal(xs: &[&'static str], daggered: bool) -> Balance {
    let mut out = [0, 0, 0];
    let r = xs.len() / 2;

    for &x in &xs[..r] {
        let s = sid(space(x));
        out[s] += if daggered { -1 } else { 1 };
    }

    for &x in &xs[r..] {
        let s = sid(space(x));
        out[s] += if daggered { 1 } else { -1 };
    }

    out
}

/// Construct a Hamiltonian dummy-index name.
/// # Arguments:
/// - `space`: Orbital space.
/// - `slot`: Dummy slot.
/// # Returns:
/// - `&'static str`: Hamiltonian dummy-index name.
pub fn hname(space: Space, slot: usize) -> &'static str {
    const HC: [&str; 4] = ["hc0", "hc1", "hc2", "hc3"];
    const HA: [&str; 4] = ["ha0", "ha1", "ha2", "ha3"];
    const HV: [&str; 4] = ["hv0", "hv1", "hv2", "hv3"];

    match space {
        Space::Core => HC[slot],
        Space::Active => HA[slot],
        Space::Virtual => HV[slot],
    }
}

/// Construct a cluster-amplitude dummy-index name.
/// # Arguments:
/// - `space`: Orbital space.
/// - `slot`: Dummy slot.
/// # Returns:
/// - `&'static str`: Cluster dummy-index name.
pub fn tname(space: Space, slot: usize) -> &'static str {
    const TC: [&str; 4] = ["tc0", "tc1", "tc2", "tc3"];
    const TA: [&str; 4] = ["ta0", "ta1", "ta2", "ta3"];
    const TV: [&str; 4] = ["tv0", "tv1", "tv2", "tv3"];

    match space {
        Space::Core => TC[slot],
        Space::Active => TA[slot],
        Space::Virtual => TV[slot],
    }
}

/// Convert excitation-class labels into cluster dummy labels.
/// # Arguments:
/// - `xs`: Excitation-class free-index names.
/// # Returns:
/// - `Vec<&'static str>`: Cluster dummy-index labels with matching spaces.
pub fn tlabels(xs: &[&'static str]) -> Vec<&'static str> {
    xs.iter()
        .enumerate()
        .map(|(i, &x)| tname(space(x), i))
        .collect()
}
