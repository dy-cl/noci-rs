// nocc/overlap.rs

use crate::nocc::common::{delta, ids, orbs, theta, unpack};
use crate::nocc::loader::overlap_terms;
use crate::nocc::space::{Excitation, ExcitationClass, Spaces, excitation_class};
use crate::nocc::terms::{GeneratedTerm, OverlapBlockTerms, TensorFactor};
use crate::nocc::{Cumulants, RDM1};

/// Reference tensors needed to evaluate overlap term tables.
#[derive(Clone, Copy)]
struct Tensors<'a> {
    /// Core, active, and virtual orbital-space maps.
    spaces: &'a Spaces,
    /// Spin-free one-particle RDM.
    gamma1: &'a RDM1<f64>,
    /// Spin-free active-space cumulants.
    lambdas: &'a Cumulants<f64>,
}

/// Fill free indices for one overlap element.
/// # Arguments:
/// - `idx`: Block-local orbital index values.
/// - `block`: Generated terms for the overlap block.
/// - `left`: Left raw spin-free excitation.
/// - `right`: Right raw spin-free excitation.
/// # Returns:
/// - `()`: Mutates `idx` in place.
fn fill(
    idx: &mut [usize],
    block: &OverlapBlockTerms,
    left: Excitation,
    right: Excitation,
) {
    let lval = unpack(left);
    let rval = unpack(right);

    assert_eq!(
        block.left_free.len(),
        lval.len(),
        "left free-index count does not match excitation rank",
    );
    assert_eq!(
        block.right_free.len(),
        rval.len(),
        "right free-index count does not match excitation rank",
    );

    for (&id, value) in block.left_free.iter().zip(lval) {
        idx[id as usize] = value;
    }

    for (&id, value) in block.right_free.iter().zip(rval) {
        idx[id as usize] = value;
    }
}

/// Evaluate one generated tensor factor.
/// # Arguments:
/// - `tensor`: Generated tensor factor.
/// - `idx`: Block-local orbital index values.
/// - `tensors`: Reference tensors needed by the overlap evaluator.
/// # Returns:
/// - `f64`: Tensor element.
fn factor(
    tensor: &TensorFactor,
    idx: &[usize],
    tensors: Tensors,
) -> f64 {
    let upper = &tensor.1;
    let lower = &tensor.2;

    match tensor.0 {
        0 => {
            tensors.gamma1.data[idx[upper[0] as usize] * tensors.gamma1.n + idx[lower[0] as usize]]
        }
        1 => theta(
            tensors.gamma1,
            idx[upper[0] as usize],
            idx[lower[0] as usize],
        ),
        4 => {
            let u = ids(tensors.spaces, upper, idx);
            let l = ids(tensors.spaces, lower, idx);

            tensors.lambdas.lambda2.get(&u, &l)
        }
        5 => {
            let u = ids(tensors.spaces, upper, idx);
            let l = ids(tensors.spaces, lower, idx);

            tensors.lambdas.lambda3.get(&u, &l)
        }
        6 => {
            let u = ids(tensors.spaces, upper, idx);
            let l = ids(tensors.spaces, lower, idx);

            tensors.lambdas.lambda4.get(&u, &l)
        }
        2 | 3 | 8 | 9 => panic!("unexpected tensor kind {} in overlap term table", tensor.0),
        _ => panic!("unknown tensor kind {}", tensor.0),
    }
}

/// Evaluate one generated term at fixed index values.
/// # Arguments:
/// - `item`: Generated overlap term.
/// - `idx`: Block-local orbital index values.
/// - `tensors`: Reference tensors needed by the overlap evaluator.
/// # Returns:
/// - `f64`: Term value.
fn term(
    item: &GeneratedTerm,
    idx: &[usize],
    tensors: Tensors,
) -> f64 {
    let mut out = item.0[0] as f64 / item.0[1] as f64;

    for pair in item.2.iter() {
        out *= delta(idx[pair[0] as usize], idx[pair[1] as usize]);

        if out == 0.0 {
            return 0.0;
        }
    }

    for tensor in item.3.iter() {
        out *= factor(tensor, idx, tensors);

        if out == 0.0 {
            return 0.0;
        }
    }

    out
}

/// Sum one generated term over its dummy indices.
/// # Arguments:
/// - `item`: Generated overlap term.
/// - `block`: Generated terms for the overlap block.
/// - `depth`: Current dummy-loop depth.
/// - `idx`: Block-local orbital index values.
/// - `tensors`: Reference tensors needed by the overlap evaluator.
/// # Returns:
/// - `f64`: Dummy-summed term contribution.
fn sum(
    item: &GeneratedTerm,
    block: &OverlapBlockTerms,
    depth: usize,
    idx: &mut [usize],
    tensors: Tensors,
) -> f64 {
    if depth == item.1.len() {
        return term(item, idx, tensors);
    }

    let id = item.1[depth] as usize;
    let kind = block.indices[id].1;
    let mut out = 0.0;

    for &p in orbs(tensors.spaces, kind).iter() {
        idx[id] = p;
        out += sum(item, block, depth + 1, idx, tensors);
    }

    out
}

/// Evaluate one overlap element from a term table.
/// # Arguments:
/// - `block`: Generated terms for the overlap block.
/// - `left`: Left raw spin-free excitation.
/// - `right`: Right raw spin-free excitation.
/// - `tensors`: Reference tensors needed by the overlap evaluator.
/// # Returns:
/// - `f64`: Overlap element.
fn eval(
    block: &OverlapBlockTerms,
    left: Excitation,
    right: Excitation,
    tensors: Tensors,
) -> f64 {
    let mut idx = vec![0; block.indices.len()];

    fill(&mut idx, block, left, right);

    block
        .terms
        .iter()
        .map(|item| sum(item, block, 0, &mut idx, tensors))
        .sum()
}

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

/// Evaluate one generated FOIS overlap metric element.
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
        spaces,
        gamma1,
        lambdas,
    };

    if swap {
        eval(block, right, left, tensors)
    } else {
        eval(block, left, right, tensors)
    }
}
