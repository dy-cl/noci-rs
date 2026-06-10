// nocc/overlap.rs

use crate::nocc::common::{eval, Tensors};
use crate::nocc::loader::overlap_terms;
use crate::nocc::space::{Excitation, ExcitationClass, Spaces, excitation_class};
use crate::nocc::{Cumulants, RDM1};

/// Return the generated overlap block for two excitation classes.
/// # Arguments:
/// - `lclass`: Left excitation class.
/// - `rclass`: Right excitation class.
/// # Returns:
/// - `Option<(&'static str, bool)>`: Block name and whether to swap excitations before evaluation.
fn block(
    lclass: ExcitationClass,
    rclass: ExcitationClass,
) -> Option<(&'static str, bool)> {
    use ExcitationClass::*;

    match (lclass, rclass) {
        (CToA, CToA) => Some(("C1", false)),
        (AToV, AToV) => Some(("C2", false)),
        (AToA, AToA) => Some(("C3", false)),
        (CAToAV, CAToAV) => Some(("C4", false)),
        (CAToVA, CAToVA) => Some(("C5", false)),
        (CAToVV, CAToVV) => Some(("C6", false)),
        (CCToAV, CCToAV) => Some(("C7", false)),
        (CCToAA, CCToAA) => Some(("C8", false)),
        (CAToAA, CAToAA) => Some(("C9", false)),
        (AAToAV, AAToAV) => Some(("C10", false)),
        (AAToVV, AAToVV) => Some(("C11", false)),
        (AAToAA, AAToAA) => Some(("C12", false)),
        (AToV, AAToAV) => Some(("C13", false)),
        (AAToAV, AToV) => Some(("C13", true)),
        (CToA, CAToAA) => Some(("C14", false)),
        (CAToAA, CToA) => Some(("C14", true)),
        (AToA, AAToAA) => Some(("C15", false)),
        (AAToAA, AToA) => Some(("C15", true)),
        (CAToAV, CAToVA) => Some(("C16", false)),
        (CAToVA, CAToAV) => Some(("C16", true)),
        (CToV, CToV) => Some(("C17", false)),
        (CToV, CAToAV) => Some(("C18", false)),
        (CAToAV, CToV) => Some(("C18", true)),
        (CToV, CAToVA) => Some(("C19", false)),
        (CAToVA, CToV) => Some(("C19", true)),
        _ => None,
    }
}

/// Evaluate one generated FOIS overlap metric element 
/// S_{\mu\nu} = \langle \Phi | \hat t_\mu^\dagger \hat \tau_\nu | \Phi \rangle.
/// # Arguments:
/// - `left`: Left excitation operator.
/// - `right`: Right excitation operator.
/// - `spaces`: Core, active, and virtual orbital-space maps.
/// - `gamma1`: Spin-free one-particle RDM.
/// - `lambdas`: Spin-free cumulants.
/// # Returns:
/// - `f64`: Raw FOIS overlap metric element, or `0.0` for orthogonal class pairs.
pub(crate) fn overlap_element(
    left: Excitation,
    right: Excitation,
    spaces: &Spaces,
    gamma1: &RDM1<f64>,
    lambdas: &Cumulants<f64>,
) -> f64 {
    let lclass = excitation_class(spaces, left);
    let rclass = excitation_class(spaces, right);

    let Some((name, swap)) = block(lclass, rclass) else {
        return 0.0;
    };

    let block = overlap_terms()
        .blocks
        .get(name)
        .expect("missing overlap block terms");

    let tensors = Tensors {
        ao: None,
        f: None,
        spaces,
        gamma1,
        lambdas,
        t1: None,
        t2: None,
    };

    let (left, right) = if swap {
        (right, left)
    } else {
        (left, right)
    };

    eval(
        block.indices.len(),
        &block.indices,
        &block.terms,
        &[
            (block.left_free.as_slice(), left),
            (block.right_free.as_slice(), right),
        ],
        &tensors,
    )
}
