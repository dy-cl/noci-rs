// nocc/common.rs

use ndarray::{Array2, Array4};

use crate::AoData;
use crate::nocc::space::{Excitation, ExcitationClass, Spaces};
use crate::nocc::terms::{GeneratedTerm, TensorFactor};
use crate::nocc::{Cumulants, RDM1};

/// Reference tensors needed to evaluate generated term tables.
pub(super) struct Tensors<'a> {
    /// Ao integrals.
    pub(super) ao: Option<&'a AoData>,
    /// Spin-free Fock matrix.
    pub(super) f: Option<&'a Array2<f64>>,
    /// Core, active, and virtual orbital-space maps.
    pub(super) spaces: &'a Spaces,
    /// Spin-free one-particle RDM.
    pub(super) gamma1: &'a RDM1<f64>,
    /// Spin-free active-space cumulants.
    pub(super) lambdas: &'a Cumulants<f64>,
    /// Spin-free single-excitation amplitude tensor.
    pub(super) t1: Option<&'a Array2<f64>>,
    /// Spin-free double-excitation amplitude tensor.
    pub(super) t2: Option<&'a Array4<f64>>,
}

/// Evaluate a Kronecker delta.
/// # Arguments:
/// - `p`: Left orbital index.
/// - `q`: Right orbital index.
/// # Returns:
/// - `f64`: `1.0` if the indices are equal, otherwise `0.0`.
pub(super) fn delta(
    p: usize,
    q: usize,
) -> f64 {
    if p == q { 1.0 } else { 0.0 }
}

/// Evaluate an active-space hole density.
/// # Arguments:
/// - `gamma1`: Spin-free one-particle RDM.
/// - `p`: Upper orbital index.
/// - `q`: Lower orbital index.
/// # Returns:
/// - `f64`: `Theta^p_q = 2 delta^p_q - Gamma^p_q`.
pub(super) fn theta(
    gamma1: &RDM1<f64>,
    p: usize,
    q: usize,
) -> f64 {
    2.0 * delta(p, q) - gamma1.data[p * gamma1.n + q]
}

/// Convert a global orbital index to an active-space index.
/// # Arguments:
/// - `spaces`: Orbital-space partitioning and index maps.
/// - `p`: Global orbital index.
/// # Returns:
/// - `usize`: Active-space index corresponding to `p`.
pub(super) fn active(
    spaces: &Spaces,
    p: usize,
) -> usize {
    spaces.active_map[p].expect("expected active orbital index")
}

/// Return orbitals belonging to a generated orbital-space id.
/// # Arguments:
/// - `spaces`: Orbital-space partitioning and index maps.
/// - `kind`: Generated orbital-space id.
/// # Returns:
/// - `&[usize]`: Orbital indices in the requested space.
pub(super) fn orbs(
    spaces: &Spaces,
    kind: u8,
) -> &[usize] {
    match kind {
        0 => &spaces.core,
        1 => &spaces.active,
        2 => &spaces.virtuals,
        _ => panic!("unknown orbital space kind {kind}"),
    }
}

/// Convert class-local active indices to active-space tensor indices.
/// # Arguments:
/// - `spaces`: Orbital-space partitioning and index maps.
/// - `raw`: Class-local index ids.
/// - `idx`: Class-local orbital index values.
/// # Returns:
/// - `([usize; 4], usize)`: Active-space tensor indices and active length.
fn ids(
    spaces: &Spaces,
    raw: &[u16],
    idx: &[usize],
) -> ([usize; 4], usize) {
    let mut out = [0; 4];
    for (i, &id) in raw.iter().enumerate() {
        out[i] = active(spaces, idx[id as usize]);
    }
    (out, raw.len())
}

/// Return excitation indices in generated free-index order.
/// # Arguments:
/// - `ex`: Raw spin-free excitation.
/// # Returns:
/// - `([usize; 4], usize)`: Creation indices followed by annihilation indices and active length.
fn unpack(ex: Excitation) -> ([usize; 4], usize) {
    match ex {
        Excitation::Single { p, q } => ([p, q, 0, 0], 2),
        Excitation::Double { p, q, r, s } => ([p, q, r, s], 4),
    }
}

/// Return the generated term-table name for an excitation class.
/// # Arguments:
/// - `class`: Excitation class.
/// # Returns:
/// - `&'static str`: Generated excitation class name.
pub(super) fn class_name(class: ExcitationClass) -> &'static str {
    match class {
        ExcitationClass::CToA => "CToA",
        ExcitationClass::CToV => "CToV",
        ExcitationClass::AToA => "AToA",
        ExcitationClass::AToV => "AToV",
        ExcitationClass::CCToAA => "CCToAA",
        ExcitationClass::CCToAV => "CCToAV",
        ExcitationClass::CAToAA => "CAToAA",
        ExcitationClass::CAToAV => "CAToAV",
        ExcitationClass::CAToVA => "CAToVA",
        ExcitationClass::CAToVV => "CAToVV",
        ExcitationClass::AAToAA => "AAToAA",
        ExcitationClass::AAToAV => "AAToAV",
        ExcitationClass::AAToVV => "AAToVV",
    }
}

/// Fill generated free indices from one or more raw spin-free excitations.
/// # Arguments:
/// - `idx`: Local orbital index values.
/// - `sources`: Pairs of generated free-index IDs and the excitation that supplies them.
/// # Returns:
/// - `()`: Mutates `idx` in place.
fn fill(
    idx: &mut [usize],
    sources: &[(&[u16], Excitation)],
) {
    for &(free, ex) in sources.iter() {
        let (values, nvalue) = unpack(ex);

        for (&id, &value) in free.iter().zip(values[..nvalue].iter()) {
            idx[id as usize] = value;
        }
    }
}

/// Evaluate one generated term at fixed index values. Each term is essentially
/// a combination of a coefficient, sum over dummy indices, product of deltas,
/// product of tensors. So here we calculate symbolically:
/// T_a(i) = c_a \prod_{r, s \in D_a} \delta_{i_r, i_s} \prod_{F \in \mathcal{F}_a} F(i).
/// Where a labels the current Wick term, i is index assignment vector, c_a is a coefficient,
/// D_a is the list of Kronecker delta constraints and \mathcal{F}_a is the list of tensors.
/// # Arguments:
/// - `item`: Generated overlap term.
/// - `idx`: Block-local orbital index values.
/// - `tensors`: Reference tensors needed by the overlap evaluator.
/// # Returns:
/// - `f64`: Term value.
fn term(
    item: &GeneratedTerm,
    idx: &[usize],
    tensors: &Tensors<'_>,
) -> f64 {
    // Begin with coefficient c_a.
    let mut out = item.0[0] as f64 / item.0[1] as f64;

    // Multiply by all deltas \delta_{i_r, i_s}.
    for pair in item.2.iter() {
        out *= delta(idx[pair[0] as usize], idx[pair[1] as usize]);

        if out == 0.0 {
            return 0.0;
        }
    }

    // Multiply by all tensors F(i) \in \mathcal{F}_a.
    for tensor in item.3.iter() {
        out *= factor(tensor, idx, tensors);

        if out == 0.0 {
            return 0.0;
        }
    }

    out
}

/// Sum one generated term over its dummy indices. This is a sum over all the
/// terms T_a(i) generated in the above function:
/// \sum_{i_d_0 \in \Omega_d_0} \sum_{i_d_1 \in \Omega_d_1} \cdots \sum_{i_d_k \in \Omega_d_k} T_a(i)
/// where each \Omega_d_k is an allowed orbital space for the given dummy index.
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
    indices: &[(String, u8)],
    depth: usize,
    idx: &mut [usize],
    tensors: &Tensors<'_>,
) -> f64 {
    // All dummy indices are assigned already, so we can just evaluate directly.
    if depth == item.1.len() {
        return term(item, idx, tensors);
    }

    // Pick next dummy index to assign.
    let id = item.1[depth] as usize;
    let kind = indices[id].1;

    let mut out = 0.0;

    // Single summation loop \sum_{i_d_k}, call recursively for deeper loops.
    for &p in orbs(tensors.spaces, kind).iter() {
        idx[id] = p;
        out += sum(item, indices, depth + 1, idx, tensors);
    }

    out
}

/// Evaluate one generated tensor factor.
/// # Arguments:
/// - `tensor`: Generated tensor factor.
/// - `idx`: Local orbital index values.
/// - `tensors`: Runtime tensors needed by the generated-term evaluator.
/// # Returns:
/// - `f64`: Tensor element.
pub(super) fn factor(
    tensor: &TensorFactor,
    idx: &[usize],
    tensors: &Tensors<'_>,
) -> f64 {
    let upper = &tensor.1;
    let lower = &tensor.2;

    // First element of tensor dictates what type it is.
    match tensor.0 {
        // Gamma_{i_l}^{i_u}.
        0 => {
            tensors.gamma1.data[idx[upper[0] as usize] * tensors.gamma1.n + idx[lower[0] as usize]]
        }
        // Theta_{i_l}^{i_u}.
        1 => theta(
            tensors.gamma1,
            idx[upper[0] as usize],
            idx[lower[0] as usize],
        ),
        // f_{i_l}^{i_u}.
        2 => tensors.f.unwrap()[(idx[upper[0] as usize], idx[lower[0] as usize])],
        // g_{i_l_1, i_l_2}^{i_u_1, i_u_2}.
        3 => {
            tensors.ao.unwrap().eri_coul[(
                idx[upper[0] as usize],
                idx[upper[1] as usize],
                idx[lower[0] as usize],
                idx[lower[1] as usize],
            )]
        }
        // Lambda_{i_l_1, i_l_2}^{i_u_1, i_u_2}.
        4 => {
            let (u, nu) = ids(tensors.spaces, upper, idx);
            let (l, nl) = ids(tensors.spaces, lower, idx);

            tensors.lambdas.lambda2.get(&u[..nu], &l[..nl])
        }
        // Lambda_{i_l_1, i_l_2, i_l_3}^{i_u_1, i_u_2, i_u_3}.
        5 => {
            let (u, nu) = ids(tensors.spaces, upper, idx);
            let (l, nl) = ids(tensors.spaces, lower, idx);

            tensors.lambdas.lambda3.get(&u[..nu], &l[..nl])
        }
        // Lambda_{i_l_1, i_l_2, i_l_3, i_l_4}^{i_u_1, i_u_2, i_u_3, i_u_4}.
        6 => {
            let (u, nu) = ids(tensors.spaces, upper, idx);
            let (l, nl) = ids(tensors.spaces, lower, idx);

            tensors.lambdas.lambda4.get(&u[..nu], &l[..nl])
        }
        // t_{i_l}^{i_u}.
        8 => tensors.t1.unwrap()[(idx[upper[0] as usize], idx[lower[0] as usize])],
        // t_{i_l_1, i_l_2}^{i_u_1, i_u_2}.
        9 => tensors.t2.unwrap()[(
            idx[upper[0] as usize],
            idx[upper[1] as usize],
            idx[lower[0] as usize],
            idx[lower[1] as usize],
        )],
        _ => panic!("unknown tensor kind {}", tensor.0),
    }
}

/// Evaluate one generated element from a term table:
/// X^B = \sum_{\alpha \in B} \sum_{\mathbf{d}_\alpha} T_\alpha(\mathbf{i}, \mathbf{d}_\alpha)
/// where B is the generated block/class, \alpha runs over terms,
/// \mathbf{d}_\alpha is a vector of dummy indices, and \mathbf{i} is a vector of free indices.
/// # Arguments:
/// - `nidx`: Number of block/class-local symbolic indices.
/// - `terms`: Generated terms for the block/class.
/// - `sources`: Pairs of generated free-index IDs and the excitation that supplies them.
/// - `kind`: Function returning the orbital-space kind for a symbolic index.
/// - `tensors`: Runtime tensors needed by the generated-term evaluator.
/// # Returns:
/// - `f64`: Generated element.
pub(super) fn eval(
    nidx: usize,
    indices: &[(String, u8)],
    terms: &[GeneratedTerm],
    sources: &[(&[u16], Excitation)],
    tensors: &Tensors<'_>,
) -> f64 {
    let mut idx = vec![0; nidx];

    // Set the free indices specified by the excitations.
    fill(&mut idx, sources);

    // Sum all the generated terms.
    terms
        .iter()
        .map(|item| sum(item, indices, 0, &mut idx, tensors))
        .sum()
}
