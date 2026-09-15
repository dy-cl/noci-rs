// determinant/reduced.rs

// Crate-root imports.
use crate::{ReducedOneSpinState, ReducedTwoSpinState};

/// Numerical one-spin payload with its parent-local component identity.
#[derive(Clone, Copy)]
pub(crate) struct ReducedOneSpinDeterminantState<I> {
    /// Parent orbital-frame identity for this one-spin component.
    pub(crate) parent: usize,
    /// Typed component identity within the parent.
    pub(crate) component: I,
    /// Identity-free reduced numerical payload.
    pub(crate) state: ReducedOneSpinState,
}

/// Numerical two-spin payload with its determinant identity.
#[derive(Clone, Copy)]
pub(crate) struct ReducedTwoSpinDeterminantState<I> {
    /// Typed determinant identity represented by this payload.
    pub(crate) det: I,
    /// Identity-free reduced numerical payload.
    pub(crate) state: ReducedTwoSpinState,
}
