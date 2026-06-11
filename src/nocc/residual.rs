// nocc/residual.rs
//
use ndarray::{Array1, Array2, Array4};
use rayon::prelude::*;

use crate::AoData;
use crate::nocc::common::{Tensors, class_name, eval};
use crate::nocc::loader::{r0_terms, r1_terms, r2_terms};
use crate::nocc::space::{Excitation, ExcitationClass, Spaces, excitation_class};
use crate::nocc::terms::{ResidualClassTerms, ResidualTermSet};
use crate::nocc::{Cumulants, RDM1};
use crate::scf::fock;

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
        .get(class_name(class))
        .expect("missing residual class terms")
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

/// Evaluate one zeroth-order residual element.
/// # Arguments:
/// - `ex`: Raw spin-free excitation.
/// - `tensors`: Reference tensors needed by the residual evaluator.
/// # Returns:
/// - `f64`: Zeroth-order residual element.
fn r0e(
    ex: Excitation,
    tensors: &Tensors<'_>,
) -> f64 {
    let class = data(r0_terms(), excitation_class(tensors.spaces, ex));

    eval(
        class.indices.len(),
        &class.indices,
        &class.terms,
        &[(class.free.as_slice(), ex)],
        tensors,
    )
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
    tensors: &Tensors<'_>,
) -> f64 {
    let class = data(r1_terms(), excitation_class(tensors.spaces, ex));

    eval(
        class.indices.len(),
        &class.indices,
        &class.terms,
        &[(class.free.as_slice(), ex)],
        tensors,
    )
}

/// Evaluate one second-order residual element.
/// # Arguments:
/// - `ex`: Raw spin-free excitation.
/// - `tensors`: Reference tensors needed by the residual evaluator.
/// - `amplitudes`: Cluster amplitude tensors.
/// # Returns:
/// - `f64`: First-order residual element.
fn r2e(
    ex: Excitation,
    tensors: &Tensors<'_>,
) -> f64 {
    let class = data(r2_terms(), excitation_class(tensors.spaces, ex));

    eval(
        class.indices.len(),
        &class.indices,
        &class.terms,
        &[(class.free.as_slice(), ex)],
        tensors,
    )
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
        ao: Some(ao),
        f: Some(&f),
        spaces,
        gamma1,
        lambdas,
        t1: None,
        t2: None,
    };

    let out: Vec<f64> = excitations
        .par_iter()
        .map(|&ex| r0e(ex, &tensors))
        .collect();

    Array1::from_vec(out)
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
        ao: Some(ao),
        f: Some(&f),
        spaces,
        gamma1,
        lambdas,
        t1: Some(&t1),
        t2: Some(&t2),
    };

    let out: Vec<f64> = excitations
        .par_iter()
        .map(|&ex| r1e(ex, &tensors))
        .collect();

    Array1::from_vec(out)
}

/// Build the first-order residual vector, quadratic in the supplied amplitudes.
/// # Arguments:
/// - `ao`: Integrals in the NOCI natural-orbital basis.
/// - `gamma1`: Spin-free one-particle RDM.
/// - `lambdas`: Spin-free active-space cumulants.
/// - `spaces`: Core, active, and virtual orbital-space maps.
/// - `excitations`: Raw spin-free excitation list.
/// - `amplitudes`: Cluster amplitude vector in the same order as `excitations`.
/// # Returns:
/// - `Array1<f64>`: First-order residual contribution.
pub(crate) fn r2(
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
        ao: Some(ao),
        f: Some(&f),
        spaces,
        gamma1,
        lambdas,
        t1: Some(&t1),
        t2: Some(&t2),
    };

    let out: Vec<f64> = excitations
        .par_iter()
        .map(|&ex| r2e(ex, &tensors))
        .collect();

    Array1::from_vec(out)
}
