// specs.rs
//! Orbital spaces, spin-free excitation classes and FOIS metric blocks.

/// Orbital reference space.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub enum Space {
    /// Core orbital.
    Core,
    /// Active orbital.
    Active,
    /// Virtual orbital.
    Virtual,
}

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
    /// Free-index names, created indices first.
    pub f: &'static [&'static str],
}

/// FOIS metric blocks of Lee and Tew Appendix C, extended by the `C \to V` and `CC \to VV`
/// blocks, which need no redundancy handling and are not listed there.
pub const BLOCKS: &[BlockSpec] = &[
    BlockSpec {
        name: "C1",
        left: "CToA",
        right: "CToA",
        lf: &["u", "i"],
        rf: &["v", "j"],
    },
    BlockSpec {
        name: "C2",
        left: "AToV",
        right: "AToV",
        lf: &["a", "t"],
        rf: &["b", "u"],
    },
    BlockSpec {
        name: "C3",
        left: "AToA",
        right: "AToA",
        lf: &["v", "u"],
        rf: &["x", "w"],
    },
    BlockSpec {
        name: "C4",
        left: "CAToAV",
        right: "CAToAV",
        lf: &["v", "a", "i", "u"],
        rf: &["x", "b", "j", "w"],
    },
    BlockSpec {
        name: "C5",
        left: "CAToVA",
        right: "CAToVA",
        lf: &["a", "v", "i", "u"],
        rf: &["b", "x", "j", "w"],
    },
    BlockSpec {
        name: "C6",
        left: "CAToVV",
        right: "CAToVV",
        lf: &["a", "b", "i", "u"],
        rf: &["c", "d", "j", "v"],
    },
    BlockSpec {
        name: "C7",
        left: "CCToAV",
        right: "CCToAV",
        lf: &["u", "a", "i", "j"],
        rf: &["v", "b", "k", "l"],
    },
    BlockSpec {
        name: "C8",
        left: "CCToAA",
        right: "CCToAA",
        lf: &["u", "v", "i", "j"],
        rf: &["w", "x", "k", "l"],
    },
    BlockSpec {
        name: "C9",
        left: "CAToAA",
        right: "CAToAA",
        lf: &["v", "w", "i", "u"],
        rf: &["y", "z", "j", "x"],
    },
    BlockSpec {
        name: "C10",
        left: "AAToAV",
        right: "AAToAV",
        lf: &["v", "a", "t", "u"],
        rf: &["z", "b", "x", "y"],
    },
    BlockSpec {
        name: "C11",
        left: "AAToVV",
        right: "AAToVV",
        lf: &["a", "b", "t", "u"],
        rf: &["c", "d", "v", "w"],
    },
    BlockSpec {
        name: "C12",
        left: "AAToAA",
        right: "AAToAA",
        lf: &["q", "s", "p", "r"],
        rf: &["t", "v", "u", "w"],
    },
    BlockSpec {
        name: "C13",
        left: "AToV",
        right: "AAToAV",
        lf: &["a", "u"],
        rf: &["x", "b", "v", "w"],
    },
    BlockSpec {
        name: "C14",
        left: "CToA",
        right: "CAToAA",
        lf: &["u", "i"],
        rf: &["w", "x", "j", "v"],
    },
    BlockSpec {
        name: "C15",
        left: "AToA",
        right: "AAToAA",
        lf: &["u", "t"],
        rf: &["y", "z", "w", "x"],
    },
    BlockSpec {
        name: "C16",
        left: "CAToAV",
        right: "CAToVA",
        lf: &["w", "a", "i", "u"],
        rf: &["b", "y", "j", "x"],
    },
    BlockSpec {
        name: "C17",
        left: "CToV",
        right: "CToV",
        lf: &["a", "i"],
        rf: &["b", "j"],
    },
    BlockSpec {
        name: "C18",
        left: "CToV",
        right: "CAToAV",
        lf: &["a", "i"],
        rf: &["x", "b", "j", "w"],
    },
    BlockSpec {
        name: "C19",
        left: "CToV",
        right: "CAToVA",
        lf: &["a", "i"],
        rf: &["b", "x", "j", "w"],
    },
    BlockSpec {
        name: "C20",
        left: "CCToVV",
        right: "CCToVV",
        lf: &["a", "b", "i", "j"],
        rf: &["c", "d", "k", "l"],
    },
];

/// Spin-free excitation classes with free-index names in created-then-annihilated order.
pub const EXCS: &[ExcSpec] = &[
    ExcSpec {
        name: "CToA",
        f: &["u", "i"],
    },
    ExcSpec {
        name: "AToV",
        f: &["a", "t"],
    },
    ExcSpec {
        name: "AToA",
        f: &["v", "u"],
    },
    ExcSpec {
        name: "CToV",
        f: &["a", "i"],
    },
    ExcSpec {
        name: "CAToAV",
        f: &["v", "a", "i", "u"],
    },
    ExcSpec {
        name: "CAToVA",
        f: &["a", "v", "i", "u"],
    },
    ExcSpec {
        name: "CAToVV",
        f: &["a", "b", "i", "u"],
    },
    ExcSpec {
        name: "CCToAV",
        f: &["u", "a", "i", "j"],
    },
    ExcSpec {
        name: "CCToAA",
        f: &["u", "v", "i", "j"],
    },
    ExcSpec {
        name: "CAToAA",
        f: &["v", "w", "i", "u"],
    },
    ExcSpec {
        name: "AAToAV",
        f: &["v", "a", "t", "u"],
    },
    ExcSpec {
        name: "AAToVV",
        f: &["a", "b", "t", "u"],
    },
    ExcSpec {
        name: "AAToAA",
        f: &["q", "s", "p", "r"],
    },
    ExcSpec {
        name: "CCToVV",
        f: &["a", "b", "i", "j"],
    },
];

/// Find one metric block specification.
/// # Arguments:
/// - `name`: Block name.
/// # Returns:
/// - `BlockSpec`: Matching block specification.
/// # Panics
/// - Panics if `name` is not a known block.
pub fn metric_block_spec(name: &str) -> BlockSpec {
    *BLOCKS
        .iter()
        .find(|x| x.name == name)
        .unwrap_or_else(|| panic!("unknown block {name}"))
}

/// Return the orbital space of one free-index name.
/// Core indices are `i, j, k, l`, virtual indices `a, b, c, d`, and all others are active.
/// # Arguments:
/// - `name`: Index name.
/// # Returns:
/// - `Space`: Orbital space.
pub fn index_space(name: &str) -> Space {
    match name {
        "i" | "j" | "k" | "l" => Space::Core,
        "a" | "b" | "c" | "d" => Space::Virtual,
        _ => Space::Active,
    }
}
