// scf/h/types.rs

// External crate imports.
use ndarray::Array2;
use num_complex::Complex64;

/// Stored quasi-Newton secant pair in the current local tangent basis.
#[derive(Clone, Debug)]
pub(crate) struct SecantPair {
    /// Previous alpha-spin accepted step in unweighted occupied-virtual rotation coordinates.
    pub(crate) sa: Array2<Complex64>,
    /// Previous beta-spin accepted step in unweighted occupied-virtual rotation coordinates.
    pub(crate) sb: Array2<Complex64>,
    /// Previous alpha-spin gradient change in unweighted occupied-virtual coordinates.
    pub(crate) ya: Array2<Complex64>,
    /// Previous beta-spin gradient change in unweighted occupied-virtual coordinates.
    pub(crate) yb: Array2<Complex64>,
}

/// Spin block being pseudo-canonicalised.
#[derive(Clone, Copy, Debug)]
pub(crate) enum SpinBlock {
    /// Alpha-spin orbital block.
    Alpha,
    /// Beta-spin orbital block.
    Beta,
}

/// Immutable data required to run one h-SCF optimisation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HSCFRunData<'a> {
    /// Label assigned to the resulting h-SCF state.
    pub(crate) label: &'a str,
    /// Whether the state should enter the NOCI basis.
    pub(crate) noci_basis: bool,
    /// Parent recipe index.
    pub(crate) parent: usize,
    /// Two-electron integral complex scaling parameter.
    pub(crate) lambda: Complex64,
}
