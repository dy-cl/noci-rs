// nocc/residual.rs

use ndarray::{Array1, Array2, Array4};

use crate::AoData;
use crate::nocc::loader::{r0_terms, r1_terms};
use crate::nocc::space::{Excitation, ExcitationClass, Spaces, excitation_class};
use crate::nocc::terms::{ResidualClassTerms, ResidualTerm, ResidualTermSet, TensorFactor};
use crate::nocc::{Cumulants, RDM1};
use crate::scf::fock;

/// Reference tensors needed to evaluate residual term tables.
#[derive(Clone, Copy)]
struct Tensors<'a> {
    /// Ao integrals.
    ao: &'a AoData,
    /// Spin-free Fock matrix.
    f: &'a Array2<f64>,
    /// Core, active, and virtual orbital-space maps.
    spaces: &'a Spaces,
    /// Spin-free one-particle RDM.
    gamma1: &'a RDM1<f64>,
    /// Spin-free active-space cumulants.
    lambdas: &'a Cumulants<f64>,
}

/// Cluster amplitude tensors used by amplitude-dependent residual terms.
#[derive(Clone, Copy)]
struct Amplitudes<'a> {
    /// Spin-free single-excitation amplitude tensor.
    t1: Option<&'a Array2<f64>>,
    /// Spin-free double-excitation amplitude tensor.
    t2: Option<&'a Array4<f64>>,
}

/// Evaluate a Kronecker delta.
/// # Arguments:
/// - `p`: Left orbital index.
/// - `q`: Right orbital index.
/// # Returns:
/// - `f64`: `1.0` if the indices are equal, otherwise `0.0`.
fn delta(
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
fn theta(
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
fn active(
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
fn orbs(
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

/// Return the generated term-table name for an excitation class.
/// # Arguments:
/// - `class`: Excitation class.
/// # Returns:
/// - `&'static str`: Generated excitation class name.
fn name(class: ExcitationClass) -> &'static str {
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

/// Return residual terms for one excitation class.
/// # Arguments:
/// - `set`: Residual term table.
/// - `class`: Excitation class.
/// # Returns:
/// - `&ResidualClassTerms`: Generated terms for the excitation class.
fn data(
    set: &ResidualTermSet,
    class: ExcitationClass,
) -> &ResidualClassTerms {
    set.classes
        .get(name(class))
        .expect("missing residual class terms")
}

/// Return excitation indices in generated free-index order.
/// # Arguments:
/// - `ex`: Raw spin-free excitation.
/// # Returns:
/// - `Vec<usize>`: Creation indices followed by annihilation indices.
fn unpack(ex: Excitation) -> Vec<usize> {
    match ex {
        Excitation::Single { p, q } => vec![p, q],
        Excitation::Double { p, q, r, s } => vec![p, q, r, s],
    }
}

/// Build dense spin-free amplitude tensors.
/// # Arguments:
/// - `n`: Number of molecular orbitals.
/// - `excitations`: Raw spin-free excitation list defining the amplitude ordering.
/// - `amplitudes`: Cluster amplitude vector in the same order as `excitations`.
/// # Returns:
/// - `(Array2<f64>, Array4<f64>)`: Dense `t1` and `t2` amplitude tensors.
fn amps(
    n: usize,
    excitations: &[Excitation],
    amplitudes: &Array1<f64>,
) -> (Array2<f64>, Array4<f64>) {
    assert_eq!(
        amplitudes.len(),
        excitations.len(),
        "amplitude vector length must match excitation list length",
    );

    let mut t1 = Array2::<f64>::zeros((n, n));
    let mut t2 = Array4::<f64>::zeros((n, n, n, n));

    for (nu, &ex) in excitations.iter().enumerate() {
        match ex {
            Excitation::Single { p, q } => {
                t1[(q, p)] = amplitudes[nu];
            }
            Excitation::Double { p, q, r, s } => {
                t2[(r, s, p, q)] = amplitudes[nu];
            }
        }
    }

    (t1, t2)
}

/// Build a spin-free Fock matrix from the reference one-particle density.
/// # Arguments:
/// - `ao`: Integrals in the NOCI natural-orbital basis.
/// - `gamma1`: Spin-free one-particle RDM.
/// # Returns:
/// - `Array2<f64>`: Spin-free Fock matrix.
fn fockm(
    ao: &AoData,
    gamma1: &RDM1<f64>,
) -> Array2<f64> {
    let n = gamma1.n;
    let mut da = Array2::<f64>::zeros((n, n));
    let mut db = Array2::<f64>::zeros((n, n));

    for p in 0..n {
        for q in 0..n {
            let value = 0.5 * gamma1.data[p * n + q];
            da[(p, q)] = value;
            db[(p, q)] = value;
        }
    }

    fock(&ao.h, &ao.eri_coul, &da, &db).0
}

/// Fill free indices for one residual element.
/// # Arguments:
/// - `idx`: Class-local orbital index values.
/// - `class`: Generated terms for the excitation class.
/// - `ex`: Raw spin-free excitation.
/// # Returns:
/// - `()`: Mutates `idx` in place.
fn fill(
    idx: &mut [usize],
    class: &ResidualClassTerms,
    ex: Excitation,
) {
    let values = unpack(ex);

    assert_eq!(
        class.free.len(),
        values.len(),
        "free-index count does not match excitation rank",
    );

    for (&id, value) in class.free.iter().zip(values) {
        idx[id as usize] = value;
    }
}

/// Evaluate one residual element from a term table.
/// # Arguments:
/// - `class`: Generated terms for the excitation class.
/// - `ex`: Raw spin-free excitation.
/// - `tensors`: Reference tensors needed by the residual evaluator.
/// - `amplitudes`: Cluster amplitude tensors, if required by the residual order.
/// # Returns:
/// - `f64`: Residual element.
fn eval(
    class: &ResidualClassTerms,
    ex: Excitation,
    tensors: Tensors,
    amplitudes: Amplitudes,
) -> f64 {
    let mut idx = vec![0; class.indices.len()];

    fill(&mut idx, class, ex);

    class
        .terms
        .iter()
        .map(|item| sum(item, class, 0, &mut idx, tensors, amplitudes))
        .sum()
}

/// Sum one generated term over its dummy indices.
/// # Arguments:
/// - `item`: Generated residual term.
/// - `class`: Generated terms for the excitation class.
/// - `depth`: Current dummy-loop depth.
/// - `idx`: Class-local orbital index values.
/// - `tensors`: Reference tensors needed by the residual evaluator.
/// - `amplitudes`: Cluster amplitude tensors, if required by the residual order.
/// # Returns:
/// - `f64`: Dummy-summed term contribution.
fn sum(
    item: &ResidualTerm,
    class: &ResidualClassTerms,
    depth: usize,
    idx: &mut [usize],
    tensors: Tensors,
    amplitudes: Amplitudes,
) -> f64 {
    if depth == item.1.len() {
        return term(item, idx, tensors, amplitudes);
    }

    let id = item.1[depth] as usize;
    let kind = class.indices[id].1;
    let mut out = 0.0;

    for &p in orbs(tensors.spaces, kind).iter() {
        idx[id] = p;
        out += sum(item, class, depth + 1, idx, tensors, amplitudes);
    }

    out
}

/// Evaluate one generated term at fixed index values.
/// # Arguments:
/// - `item`: Generated residual term.
/// - `idx`: Class-local orbital index values.
/// - `tensors`: Reference tensors needed by the residual evaluator.
/// - `amplitudes`: Cluster amplitude tensors, if required by the residual order.
/// # Returns:
/// - `f64`: Term value.
fn term(
    item: &ResidualTerm,
    idx: &[usize],
    tensors: Tensors,
    amplitudes: Amplitudes,
) -> f64 {
    let mut out = item.0[0] as f64 / item.0[1] as f64;

    for pair in item.2.iter() {
        out *= delta(idx[pair[0] as usize], idx[pair[1] as usize]);

        if out == 0.0 {
            return 0.0;
        }
    }

    for tensor in item.3.iter() {
        out *= factor(tensor, idx, tensors, amplitudes);

        if out == 0.0 {
            return 0.0;
        }
    }

    out
}

/// Convert class-local active indices to active-space tensor indices.
/// # Arguments:
/// - `spaces`: Orbital-space partitioning and index maps.
/// - `raw`: Class-local index ids.
/// - `idx`: Class-local orbital index values.
/// # Returns:
/// - `Vec<usize>`: Active-space tensor indices.
fn ids(
    spaces: &Spaces,
    raw: &[u16],
    idx: &[usize],
) -> Vec<usize> {
    raw.iter()
        .map(|&id| active(spaces, idx[id as usize]))
        .collect()
}

/// Evaluate one generated tensor factor.
/// # Arguments:
/// - `tensor`: Generated tensor factor.
/// - `idx`: Class-local orbital index values.
/// - `tensors`: Reference tensors needed by the residual evaluator.
/// - `amplitudes`: Cluster amplitude tensors, if required by the residual order.
/// # Returns:
/// - `f64`: Tensor element.
fn factor(
    tensor: &TensorFactor,
    idx: &[usize],
    tensors: Tensors,
    amplitudes: Amplitudes,
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
        2 => tensors.f[(idx[upper[0] as usize], idx[lower[0] as usize])],
        3 => {
            tensors.ao.eri_coul[(
                idx[upper[0] as usize],
                idx[upper[1] as usize],
                idx[lower[0] as usize],
                idx[lower[1] as usize],
            )]
        }
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
        8 => amplitudes.t1.expect("missing t1 amplitude tensor")
            [(idx[upper[0] as usize], idx[lower[0] as usize])],
        9 => amplitudes.t2.expect("missing t2 amplitude tensor")[(
            idx[upper[0] as usize],
            idx[upper[1] as usize],
            idx[lower[0] as usize],
            idx[lower[1] as usize],
        )],
        _ => panic!("unknown tensor kind {}", tensor.0),
    }
}

/// Evaluate one zeroth-order residual element.
/// # Arguments:
/// - `ex`: Raw spin-free excitation.
/// - `tensors`: Reference tensors needed by the residual evaluator.
/// # Returns:
/// - `f64`: Zeroth-order residual element.
fn r0e(
    ex: Excitation,
    tensors: Tensors,
) -> f64 {
    let class = data(r0_terms(), excitation_class(tensors.spaces, ex));
    let amplitudes = Amplitudes { t1: None, t2: None };

    eval(class, ex, tensors, amplitudes)
}

/// Evaluate one first-order residual element.
/// # Arguments:
/// - `ex`: Raw spin-free excitation.
/// - `tensors`: Reference tensors needed by the residual evaluator.
/// - `amplitudes`: Cluster amplitude tensors.
/// # Returns:
/// - `f64`: First-order residual element.
fn r1e(
    ex: Excitation,
    tensors: Tensors,
    amplitudes: Amplitudes,
) -> f64 {
    let class = data(r1_terms(), excitation_class(tensors.spaces, ex));

    eval(class, ex, tensors, amplitudes)
}

/// Build the direct zeroth-order residual vector.
/// # Arguments:
/// - `ao`: Integrals in the NOCI natural-orbital basis.
/// - `gamma1`: Spin-free one-particle RDM.
/// - `lambdas`: Spin-free active-space cumulants.
/// - `spaces`: Core, active, and virtual orbital-space maps.
/// - `excitations`: Raw spin-free excitation list.
/// # Returns:
/// - `Array1<f64>`: Direct zeroth-order residual vector.
pub(crate) fn r0(
    ao: &AoData,
    gamma1: &RDM1<f64>,
    lambdas: &Cumulants<f64>,
    spaces: &Spaces,
    excitations: &[Excitation],
) -> Array1<f64> {
    let f = fockm(ao, gamma1);
    let tensors = Tensors {
        ao,
        f: &f,
        spaces,
        gamma1,
        lambdas,
    };
    let mut out = Array1::<f64>::zeros(excitations.len());

    for (mu, &ex) in excitations.iter().enumerate() {
        out[mu] = r0e(ex, tensors);
    }

    out
}

/// Build the first-order residual vector, linear in the supplied amplitudes.
/// # Arguments:
/// - `ao`: Integrals in the NOCI natural-orbital basis.
/// - `gamma1`: Spin-free one-particle RDM.
/// - `lambdas`: Spin-free active-space cumulants.
/// - `spaces`: Core, active, and virtual orbital-space maps.
/// - `excitations`: Raw spin-free excitation list.
/// - `amplitudes`: Cluster amplitude vector in the same order as `excitations`.
/// # Returns:
/// - `Array1<f64>`: First-order residual contribution.
pub(crate) fn r1(
    ao: &AoData,
    gamma1: &RDM1<f64>,
    lambdas: &Cumulants<f64>,
    spaces: &Spaces,
    excitations: &[Excitation],
    amplitudes: &Array1<f64>,
) -> Array1<f64> {
    let n = gamma1.n;
    let f = fockm(ao, gamma1);
    let (t1, t2) = amps(n, excitations, amplitudes);
    let tensors = Tensors {
        ao,
        f: &f,
        spaces,
        gamma1,
        lambdas,
    };
    let amplitudes = Amplitudes {
        t1: Some(&t1),
        t2: Some(&t2),
    };
    let mut out = Array1::<f64>::zeros(excitations.len());

    for (mu, &ex) in excitations.iter().enumerate() {
        out[mu] = r1e(ex, tensors, amplitudes);
    }

    out
}
