// elements/rdm/common.rs

// External crate imports.
use ndarray::Array2;

// Crate-root imports.
use crate::NOCIScalar;

/// Determinant data required by the NOCC RDM evaluators, resolved from the authoritative
/// NOCI space.
pub(in crate::elements) struct RDMDeterminantView<'a, T: NOCIScalar> {
    /// Parent-reference index used to select ordered Wick intermediates.
    pub(in crate::elements) parent: usize,
    /// Alpha-spin occupation of the retained determinant.
    pub(in crate::elements) oa: u128,
    /// Beta-spin occupation of the retained determinant.
    pub(in crate::elements) ob: u128,
    /// Alpha-spin parent orbital coefficients.
    pub(in crate::elements) ca: &'a std::sync::Arc<Array2<T>>,
    /// Beta-spin parent orbital coefficients.
    pub(in crate::elements) cb: &'a std::sync::Arc<Array2<T>>,
    /// Alpha- and beta-spin excitations relative to the parent determinant.
    pub(in crate::elements) excitation: crate::Excitation,
    /// Alpha-spin fermionic excitation phase.
    pub(in crate::elements) pha: f64,
    /// Beta-spin fermionic excitation phase.
    pub(in crate::elements) phb: f64,
}

/// Resolve a retained determinant into the orbital and excitation data required by the RDM code.
/// # Arguments:
/// - `space`: Authoritative retained NOCI determinant space.
/// - `index`: Retained determinant index to resolve.
/// # Returns
/// - `RDMDeterminantView<'a, T>`: Borrowed orbital data and copied determinant metadata.
pub(in crate::elements) fn resolve_rdm_determinant<'a, T: NOCIScalar>(
    space: &'a crate::determinant::NOCISpace<T>,
    index: crate::determinant::NOCIIndex,
) -> RDMDeterminantView<'a, T> {
    let state = space.state(index);
    let parent = space.parent(index);
    let alpha = space.alpha(index);
    let beta = space.beta(index);

    RDMDeterminantView {
        parent: state.parent,
        oa: alpha.occupation,
        ob: beta.occupation,
        ca: &parent.ca,
        cb: &parent.cb,
        excitation: crate::Excitation {
            alpha: alpha.excitation,
            beta: beta.excitation,
        },
        pha: alpha.reduced.phase,
        phb: beta.reduced.phase,
    }
}
