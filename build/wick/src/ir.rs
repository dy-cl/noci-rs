// ir.rs

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

/// Spin-free orbital index.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct Idx {
    /// Symbolic name.
    pub name: &'static str,
    /// Orbital space.
    pub space: Space,
}

/// Fermion operator kind.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub enum OpKind {
    /// Creation operator.
    Create,
    /// Annihilation operator.
    Annihilate,
}

/// Spin label.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub enum Spin {
    /// Alpha spin.
    Alpha,
    /// Beta spin.
    Beta,
}

/// Spin-orbital operator.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct Op {
    /// Operator kind.
    pub kind: OpKind,
    /// Orbital index.
    pub idx: Idx,
    /// Spin label.
    pub spin: Spin,
    /// GNO group id.
    pub group: usize,
}

/// One spin-free GNO group expanded into spin-orbital strings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Group {
    /// Spin-orbital strings.
    pub strings: Vec<Vec<Op>>,
}

/// Product of GNO groups.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Product {
    /// Groups in product order.
    pub groups: Vec<Group>,
}

/// Kronecker delta.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct Delta {
    /// Left index.
    pub left: Idx,
    /// Right index.
    pub right: Idx,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub enum TensorKind {
    /// One-particle RDM.
    Gamma1,
    /// One-hole RDM.
    Theta,
    /// Two-body cumulant.
    Lambda2,
    /// Three-body cumulant.
    Lambda3,
    /// Four-body cumulant.
    Lambda4,
    /// One-body Hamiltonian coefficient.
    Fock,
    /// Two-body Hamiltonian coefficient.
    ERI,
}

/// Tensor factor.
#[derive(Clone, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct Tensor {
    /// Tensor kind.
    pub kind: TensorKind,
    /// Upper indices.
    pub upper: Vec<Idx>,
    /// Lower indices.
    pub lower: Vec<Idx>,
}

/// Rational coefficient.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct Rational {
    /// Numerator.
    pub num: i64,
    /// Denominator.
    pub den: i64,
}

/// One symbolic term after numerical Wick evaluation.
#[derive(Clone, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct Term {
    /// Rational coefficient.
    pub coeff: Rational,
    /// Delta factors.
    pub deltas: Vec<Delta>,
    /// Tensor factors.
    pub tensors: Vec<Tensor>,
}

/// Symbolic expression.
pub type Expr = Vec<Term>;

/// Construct a core index.
/// # Arguments:
/// - `name`: Index name.
/// # Returns:
/// - `Idx`: Core index.
pub const fn c(name: &'static str) -> Idx {
    Idx { name, space: Space::Core }
}

/// Construct an active index.
/// # Arguments:
/// - `name`: Index name.
/// # Returns:
/// - `Idx`: Active index.
pub const fn a(name: &'static str) -> Idx {
    Idx { name, space: Space::Active }
}

/// Construct a virtual index.
/// # Arguments:
/// - `name`: Index name.
/// # Returns:
/// - `Idx`: Virtual index.
pub const fn v(name: &'static str) -> Idx {
    Idx { name, space: Space::Virtual }
}
