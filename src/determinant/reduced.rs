// determinant/reduced.rs

// Crate-root imports.
use crate::ReducedOneSpinState;

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
