// scf/h/types.rs

use std::collections::HashMap;

use ndarray::Array2;
use num_complex::Complex64;

use crate::input::StateRecipe;
use crate::{HSCFState, SCFState};

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

/// Current- and previous-geometry determinant states keyed by label.
pub struct StateLookups<'a, T> {
    /// Current-geometry states keyed by label.
    pub current: &'a HashMap<&'a str, &'a T>,
    /// Previous-geometry states keyed by label.
    pub previous: &'a HashMap<&'a str, &'a T>,
}

/// Lookup tables used while constructing h-SCF states at one geometry.
pub struct HSCFGenerationLookups<'a> {
    /// State recipes keyed by label.
    pub recipes: &'a HashMap<&'a str, &'a StateRecipe>,
    /// Real SCF states used as h-SCF seeds.
    pub real: StateLookups<'a, SCFState>,
    /// h-SCF states used for continuation and spin-flip reuse.
    pub h: StateLookups<'a, HSCFState>,
}

/// Spin block being pseudo-canonicalised.
#[derive(Clone, Copy, Debug)]
pub(crate) enum SpinBlock {
    /// Alpha-spin orbital block.
    Alpha,
    /// Beta-spin orbital block.
    Beta,
}

/// Outcome of checking whether a holomorphic recipe should use its real partner seed.
pub(crate) enum PartnerSeed<'a> {
    /// Recipe has no partner gate.
    NoPartner,
    /// Partner exists and collapsed out of the NOCI basis, so h-SCF should be attempted.
    Use(&'a SCFState),
    /// Partner gate was present, but h-SCF should not be attempted at this geometry.
    Skip,
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
}
