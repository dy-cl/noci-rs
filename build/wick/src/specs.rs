// specs.rs

use crate::ir::{Idx, Space};

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
