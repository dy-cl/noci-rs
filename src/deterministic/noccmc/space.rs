// deterministic/noccmc/space.rs

use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, UPLO};
use std::collections::BTreeMap;

use crate::AoData;
use crate::maths::linalg::loewdin_x;
use crate::noci::{Cumulants, RDM1};
use crate::scf::fock;
use crate::utils::print_array2;

/// NOCC orbital class in the NOCI natural-orbital basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum OrbitalClass {
    /// Doubly occupied in every determinant of the reference.
    Core,
    /// Partially occupied reference orbital.
    Active,
    /// Unoccupied in every determinant of the reference.
    Virtual,
}

/// Spin-free GNOCC excitation class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum ExcitationClass {
    /// C -> A single excitation.
    CToA,
    /// C -> V single excitation.
    CToV,
    /// A -> A single excitation.
    AToA,
    /// A -> V single excitation.
    AToV,
    /// CC -> AA double excitation.
    CCToAA,
    /// CC -> AV double excitation.
    CCToAV,
    /// CA -> AA double excitation.
    CAToAA,
    /// CA -> AV double excitation.
    CAToAV,
    /// CA -> VA double excitation.
    CAToVA,
    /// CA -> VV double excitation.
    CAToVV,
    /// AA -> AA double excitation.
    AAToAA,
    /// AA -> AV double excitation.
    AAToAV,
    /// AA -> VV double excitation.
    AAToVV,
}

/// Spin-free GNOCC excitation operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Excitation {
    /// Spin-free single excitation E^p_q.
    Single { p: usize, q: usize },
    /// Spin-free double excitation E^{pq}_{rs}.
    Double {
        p: usize,
        q: usize,
        r: usize,
        s: usize,
    },
}

/// NOCC orbital spaces in the NOCI natural-orbital basis.
#[derive(Clone, Debug)]
pub(crate) struct Spaces {
    /// Number of molecular orbitals.
    pub nmo: usize,
    /// Core orbital indices.
    pub core: Vec<usize>,
    /// Active orbital indices.
    pub active: Vec<usize>,
    /// Virtual orbital indices.
    pub virtuals: Vec<usize>,
    /// Creator-side orbital indices, A ∪ V.
    pub creators: Vec<usize>,
    /// Annihilator-side orbital indices, C ∪ A.
    pub annihilators: Vec<usize>,
    /// Orbital class lookup by full MO index.
    pub class_of: Vec<OrbitalClass>,
    /// Active local index lookup by full MO index.
    pub active_map: Vec<Option<usize>>,
}

/// Build NOCC orbital spaces.
/// # Arguments:
/// - `nmo`: Number of molecular orbitals.
/// - `active`: Active orbitals from the existing NOCI natural-orbital machinery.
/// - `gamma1`: Full spin-free one-body RDM in the NOCI natural-orbital basis.
/// - `core_tol`: Tolerance for identifying inactive occupation two orbitals.
/// - `virtual_tol`: Tolerance for identifying inactive occupation zero orbitals.
/// # Returns:
/// - `Spaces`: Core, active, virtual, creator, and annihilator spaces.
pub(crate) fn build_spaces(
    nmo: usize,
    active: &[usize],
    gamma1: &RDM1<f64>,
    core_tol: f64,
    virtual_tol: f64,
) -> Spaces {
    let mut active_sorted = active.to_vec();
    active_sorted.sort_unstable();
    active_sorted.dedup();

    let mut core = Vec::new();
    let mut virtuals = Vec::new();
    let mut class_of = vec![OrbitalClass::Virtual; nmo];
    let mut active_map = vec![None; nmo];

    for (i, &p) in active_sorted.iter().enumerate() {
        class_of[p] = OrbitalClass::Active;
        active_map[p] = Some(i);
    }

    for p in 0..nmo {
        if active_map[p].is_some() {
            continue;
        }

        let occ = gamma1.data[p * gamma1.n + p];

        if (2.0 - occ).abs() <= core_tol {
            core.push(p);
            class_of[p] = OrbitalClass::Core;
        } else if occ.abs() <= virtual_tol {
            virtuals.push(p);
            class_of[p] = OrbitalClass::Virtual;
            );
        }
    }

    let mut creators = active_sorted.clone();
    creators.extend(virtuals.iter().copied());

    let mut annihilators = core.clone();
    annihilators.extend(active_sorted.iter().copied());

    Spaces {
        nmo,
        core,
        active: active_sorted,
        virtuals,
        creators,
        annihilators,
        class_of,
        active_map,
    }
}

/// Build the raw spin-free singles and doubles excitation list.
/// # Arguments:
/// - `spaces`: NOCC orbital spaces.
/// # Returns:
/// - `Vec<Excitation>`: Spin-free GNOCCSD excitations.
pub(crate) fn build_excitations(spaces: &Spaces) -> Vec<Excitation> {
    let mut out = Vec::new();

    for &p in spaces.creators.iter() {
        for &q in spaces.annihilators.iter() {
            if p != q {
                out.push(Excitation::Single { p, q });
            }
        }
    }

    for &p in spaces.creators.iter() {
        for &q in spaces.creators.iter() {
            for &r in spaces.annihilators.iter() {
                for &s in spaces.annihilators.iter() {
                    if p == r && q == s {
                        continue;
                    }

                    out.push(Excitation::Double { p, q, r, s });
                }
            }
        }
    }

    out
}

/// Print NOCC excitation-space diagnostics.
/// # Arguments:
/// - `spaces`: NOCC orbital spaces.
/// - `excitations`: Raw excitation list.
/// # Returns:
/// - `()`: Prints counts by class.
pub(crate) fn print_space_diagnostics(
    spaces: &Spaces,
    excitations: &[Excitation],
) {
    let mut counts = BTreeMap::new();
    let mut nrepeated = 0;

    for &ex in excitations.iter() {
        *counts.entry(excitation_class(spaces, ex)).or_insert(0usize) += 1;

        if matches!(ex, Excitation::Double { p, q, r, s } if p == q || r == s) {
            nrepeated += 1;
        }
    }

    println!("{}", "=".repeat(100));
    println!("GNOCC FOIS excitation-space diagnostics");
    println!("Number of MOs: {}", spaces.nmo);
    println!("Core orbitals: {}", spaces.core.len());
    println!("Active orbitals: {}", spaces.active.len());
    println!("Virtual orbitals: {}", spaces.virtuals.len());
    println!("Creator-side orbitals A ∪ V: {}", spaces.creators.len());
    println!(
        "Annihilator-side orbitals C ∪ A: {}",
        spaces.annihilators.len()
    );
    println!("Total raw spin-free excitations: {}", excitations.len());
    println!("Repeated-side spin-free doubles retained: {}", nrepeated);

    for (class, count) in counts.iter() {
        println!("{:?}: {}", class, count);
    }
}

/// Print the weighted FOIS metric diagnostics.
/// # Arguments:
/// - `ao`: Integrals transformed to the NOCI natural-orbital basis.
/// - `gamma1`: Full-space spin-free one-body RDM.
/// - `lambdas`: Active-space spin-free cumulants.
/// - `spaces`: NOCC orbital spaces.
/// - `excitations`: Raw spin-free excitation list.
/// - `tol`: Weighted overlap eigenvalue threshold.
/// # Returns:
/// - `()`: Prints raw and weighted FOIS metric diagnostics.
pub(crate) fn print_fois_metric_diagnostics(
    ao: &AoData,
    gamma1: &RDM1<f64>,
    lambdas: &Cumulants<f64>,
    spaces: &Spaces,
    excitations: &[Excitation],
    tol: f64,
) {
    let mut s: Array2<f64> = Array2::zeros((excitations.len(), excitations.len()));

    for (i, &left) in excitations.iter().enumerate() {
        for (j, &right) in excitations.iter().enumerate() {
            s[(i, j)] = overlap_element(left, right, spaces, gamma1, lambdas);
        }
    }

    let mut asym: f64 = 0.0;
    for i in 0..s.nrows() {
        for j in 0..s.ncols() {
            asym = asym.max((s[(i, j)] - s[(j, i)]).abs());
        }
    }

    let (s_evals, _) = s
        .clone()
        .eigh(UPLO::Upper)
        .expect("raw FOIS overlap diagonalisation failed");

    let h = hamiltonian_weights(ao, gamma1, excitations);
    let mut stilde: Array2<f64> = Array2::zeros(s.raw_dim());

    for i in 0..s.nrows() {
        for j in 0..s.ncols() {
            stilde[(i, j)] = h[i] * s[(i, j)] * h[j];
        }
    }

    let (stilde_evals, _) = stilde
        .clone()
        .eigh(UPLO::Upper)
        .expect("weighted FOIS overlap diagonalisation failed");

    let xtilde = loewdin_x(&stilde, true, tol);
    let mut y = xtilde.clone();

    for mu in 0..h.len() {
        for col in 0..y.ncols() {
            y[(mu, col)] *= h[mu];
        }
    }

    let nkeep = y.ncols();
    let nnull = h.len() - nkeep;
    let ytsy = y.t().dot(&s).dot(&y);
    let mut orth_err: f64 = 0.0;

    for i in 0..ytsy.nrows() {
        for j in 0..ytsy.ncols() {
            let target = if i == j { 1.0 } else { 0.0 };
            orth_err = orth_err.max((ytsy[(i, j)] - target).abs());
        }
    }

    println!("{}", "=".repeat(100));
    println!("GNOCC weighted FOIS metric diagnostics");
    println!("Raw metric dimension: {}", s.nrows());
    println!("Raw metric max asymmetry: {:.6e}", asym);
    println!(
        "Raw metric min eigenvalue: {:.6e}",
        s_evals.iter().copied().fold(f64::INFINITY, f64::min)
    );
    println!(
        "Raw metric max eigenvalue: {:.6e}",
        s_evals.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    );
    println!(
        "Weighted metric min eigenvalue: {:.6e}",
        stilde_evals.iter().copied().fold(f64::INFINITY, f64::min)
    );
    println!(
        "Weighted metric max eigenvalue: {:.6e}",
        stilde_evals
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    );
    println!("Weighted FOIS kept directions: {}", nkeep);
    println!("Weighted FOIS near-null directions: {}", nnull);
    println!("Max Y^T S Y - I error: {:.6e}", orth_err);

    print_block_diagnostics(&s, spaces, excitations);
    print_diagonal_diagnostics(&s, spaces, excitations);

    if excitations.len() <= 16 {
        println!("{}", "-".repeat(100));
        println!("Raw FOIS metric:");
        print_array2(&s);
    }

    println!("{}", "-".repeat(100));
    println!("Hamiltonian weights:");
    for (i, &hi) in h.iter().enumerate() {
        println!("{:4}: {:.12e} {}", i, hi, excitation_label(excitations[i]));
    }
}

/// Print per-excitation-class raw metric block diagnostics.
/// # Arguments:
/// - `s`: Raw FOIS metric.
/// - `spaces`: NOCC orbital spaces.
/// - `excitations`: Raw spin-free excitation list.
/// # Returns:
/// - `()`: Prints block dimensions, asymmetries, and eigenvalue ranges.
fn print_block_diagnostics(
    s: &Array2<f64>,
    spaces: &Spaces,
    excitations: &[Excitation],
) {
    let mut blocks = BTreeMap::new();

    for (i, &left) in excitations.iter().enumerate() {
        blocks
            .entry(excitation_class(spaces, left))
            .or_insert_with(Vec::new)
            .push(i);
    }

    println!("{}", "-".repeat(100));
    println!("Raw metric block diagnostics");

    for (class, indices) in blocks.iter() {
        let mut block: Array2<f64> = Array2::zeros((indices.len(), indices.len()));

        for (ii, &i) in indices.iter().enumerate() {
            for (jj, &j) in indices.iter().enumerate() {
                block[(ii, jj)] = s[(i, j)];
            }
        }

        let mut asym: f64 = 0.0;
        for i in 0..block.nrows() {
            for j in 0..block.ncols() {
                asym = asym.max((block[(i, j)] - block[(j, i)]).abs());
            }
        }

        let (evals, _) = block
            .clone()
            .eigh(UPLO::Upper)
            .expect("raw FOIS block diagonalisation failed");

        println!(
            "{:?}: dim: {}, asym: {:.6e}, raw evals: [{:.6e}, {:.6e}]",
            class,
            indices.len(),
            asym,
            evals.iter().copied().fold(f64::INFINITY, f64::min),
            evals.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        );
    }
}

/// Print negative raw metric diagonal diagnostics.
/// # Arguments:
/// - `s`: Raw FOIS metric.
/// - `spaces`: NOCC orbital spaces.
/// - `excitations`: Raw spin-free excitation list.
/// # Returns:
/// - `()`: Prints all diagonal elements below tolerance.
fn print_diagonal_diagnostics(
    s: &Array2<f64>,
    spaces: &Spaces,
    excitations: &[Excitation],
) {
    println!("{}", "-".repeat(100));
    println!("Negative raw metric diagonal diagnostics");

    let mut nbad = 0;

    for (i, &ex) in excitations.iter().enumerate() {
        if s[(i, i)] < -1.0e-10 {
            nbad += 1;

            println!(
                "{:4}: diag: {:.12e}, class: {:?}, {}",
                i,
                s[(i, i)],
                excitation_class(spaces, ex),
                excitation_label(ex)
            );
        }
    }

    println!("Negative raw metric diagonals: {}", nbad);
}

/// Classify a spin-free excitation by orbital spaces.
/// # Arguments:
/// - `spaces`: NOCC orbital spaces.
/// - `ex`: Spin-free excitation.
/// # Returns:
/// - `ExcitationClass`: Excitation class used for Appendix-C dispatch.
fn excitation_class(
    spaces: &Spaces,
    ex: Excitation,
) -> ExcitationClass {
    match ex {
        Excitation::Single { p, q } => match (spaces.class_of[q], spaces.class_of[p]) {
            (OrbitalClass::Core, OrbitalClass::Active) => ExcitationClass::CToA,
            (OrbitalClass::Core, OrbitalClass::Virtual) => ExcitationClass::CToV,
            (OrbitalClass::Active, OrbitalClass::Active) => ExcitationClass::AToA,
            (OrbitalClass::Active, OrbitalClass::Virtual) => ExcitationClass::AToV,
        },
        Excitation::Double { p, q, r, s } => match (
            spaces.class_of[r],
            spaces.class_of[s],
            spaces.class_of[p],
            spaces.class_of[q],
        ) {
            (
                OrbitalClass::Core,
                OrbitalClass::Core,
                OrbitalClass::Active,
                OrbitalClass::Active,
            ) => ExcitationClass::CCToAA,
            (
                OrbitalClass::Core,
                OrbitalClass::Core,
                OrbitalClass::Active,
                OrbitalClass::Virtual,
            ) => ExcitationClass::CCToAV,
            (
                OrbitalClass::Core,
                OrbitalClass::Active,
                OrbitalClass::Active,
                OrbitalClass::Active,
            ) => ExcitationClass::CAToAA,
            (
                OrbitalClass::Core,
                OrbitalClass::Active,
                OrbitalClass::Active,
                OrbitalClass::Virtual,
            ) => ExcitationClass::CAToAV,
            (
                OrbitalClass::Core,
                OrbitalClass::Active,
                OrbitalClass::Virtual,
                OrbitalClass::Active,
            ) => ExcitationClass::CAToVA,
            (
                OrbitalClass::Core,
                OrbitalClass::Active,
                OrbitalClass::Virtual,
                OrbitalClass::Virtual,
            ) => ExcitationClass::CAToVV,
            (
                OrbitalClass::Active,
                OrbitalClass::Active,
                OrbitalClass::Active,
                OrbitalClass::Active,
            ) => ExcitationClass::AAToAA,
            (
                OrbitalClass::Active,
                OrbitalClass::Active,
                OrbitalClass::Active,
                OrbitalClass::Virtual,
            ) => ExcitationClass::AAToAV,
            (
                OrbitalClass::Active,
                OrbitalClass::Active,
                OrbitalClass::Virtual,
                OrbitalClass::Virtual,
            ) => ExcitationClass::AAToVV,
        },
    }
}

/// Return one raw FOIS overlap element from the Appendix-C block dispatcher.
/// # Arguments:
/// - `left`: Left stored excitation, daggered in the metric.
/// - `right`: Right stored excitation.
/// - `spaces`: NOCC orbital spaces.
/// - `gamma1`: Full-space spin-free one-body RDM.
/// - `lambdas`: Active-space spin-free cumulants.
/// # Returns:
/// - `f64`: Raw overlap element.
fn overlap_element(
    left: Excitation,
    right: Excitation,
    spaces: &Spaces,
    gamma1: &RDM1<f64>,
    lambdas: &Cumulants<f64>,
) -> f64 {
    let lclass = excitation_class(spaces, left);
    let rclass = excitation_class(spaces, right);

    match (lclass, rclass) {
        (ExcitationClass::CToA, ExcitationClass::CToA) => overlap_c_to_a(left, right, gamma1),
        (ExcitationClass::AToV, ExcitationClass::AToV) => overlap_a_to_v(left, right, gamma1),
        (ExcitationClass::AToA, ExcitationClass::AToA) => {
            overlap_a_to_a(left, right, spaces, gamma1, lambdas)
        }
        (ExcitationClass::CAToAV, ExcitationClass::CAToAV) => {
            overlap_ca_to_av(left, right, spaces, gamma1, lambdas)
        }
        (ExcitationClass::CAToVA, ExcitationClass::CAToVA) => {
            overlap_ca_to_va(left, right, spaces, gamma1, lambdas)
        }
        (ExcitationClass::CAToVV, ExcitationClass::CAToVV) => overlap_ca_to_vv(left, right, gamma1),
        (ExcitationClass::CCToAV, ExcitationClass::CCToAV) => overlap_cc_to_av(left, right, gamma1),
        (ExcitationClass::CCToAA, ExcitationClass::CCToAA) => {
            overlap_cc_to_aa(left, right, spaces, gamma1, lambdas)
        }
        (ExcitationClass::CAToAA, ExcitationClass::CAToAA) => {
            overlap_ca_to_aa(left, right, spaces, gamma1, lambdas)
        }
        (ExcitationClass::AAToAV, ExcitationClass::AAToAV) => {
            overlap_aa_to_av(left, right, spaces, gamma1, lambdas)
        }
        (ExcitationClass::AAToVV, ExcitationClass::AAToVV) => {
            overlap_aa_to_vv(left, right, spaces, gamma1, lambdas)
        }
        (ExcitationClass::AAToAA, ExcitationClass::AAToAA) => {
            overlap_aa_to_aa(left, right, spaces, gamma1, lambdas)
        }
        (ExcitationClass::AToV, ExcitationClass::AAToAV) => {
            overlap_a_to_v_aa_to_av(left, right, spaces, lambdas)
        }
        (ExcitationClass::AAToAV, ExcitationClass::AToV) => {
            overlap_a_to_v_aa_to_av(right, left, spaces, lambdas)
        }
        (ExcitationClass::CToA, ExcitationClass::CAToAA) => {
            overlap_c_to_a_ca_to_aa(left, right, spaces, lambdas)
        }
        (ExcitationClass::CAToAA, ExcitationClass::CToA) => {
            overlap_c_to_a_ca_to_aa(right, left, spaces, lambdas)
        }
        (ExcitationClass::AToA, ExcitationClass::AAToAA) => {
            overlap_a_to_a_aa_to_aa(left, right, spaces, gamma1, lambdas)
        }
        (ExcitationClass::AAToAA, ExcitationClass::AToA) => {
            overlap_a_to_a_aa_to_aa(right, left, spaces, gamma1, lambdas)
        }
        (ExcitationClass::CAToAV, ExcitationClass::CAToVA) => {
            overlap_ca_to_av_ca_to_va(left, right, spaces, gamma1, lambdas)
        }
        (ExcitationClass::CAToVA, ExcitationClass::CAToAV) => {
            overlap_ca_to_av_ca_to_va(right, left, spaces, gamma1, lambdas)
        }
    }
}

/// Evaluate the C1 C -> A / C -> A overlap block.
/// # Arguments:
/// - `left`: Left C -> A excitation.
/// - `right`: Right C -> A excitation.
/// - `gamma1`: Full-space spin-free one-body RDM.
/// # Returns:
/// - `f64`: Raw overlap element.
fn overlap_c_to_a(
    left: Excitation,
    right: Excitation,
    gamma1: &RDM1<f64>,
) -> f64 {
    let (u, i) = single(left);
    let (v, j) = single(right);
    delta(i, j) * theta(gamma1, v, u)
}

/// Evaluate the C2 A -> V / A -> V overlap block.
/// # Arguments:
/// - `left`: Left A -> V excitation.
/// - `right`: Right A -> V excitation.
/// - `gamma1`: Full-space spin-free one-body RDM.
/// # Returns:
/// - `f64`: Raw overlap element.
fn overlap_a_to_v(
    left: Excitation,
    right: Excitation,
    gamma1: &RDM1<f64>,
) -> f64 {
    let (a, t) = single(left);
    let (b, u) = single(right);
    delta(b, a) * gamma1.data[t * gamma1.n + u]
}

/// Evaluate the C3 A -> A / A -> A overlap block.
/// # Arguments:
/// - `left`: Left A -> A excitation.
/// - `right`: Right A -> A excitation.
/// - `spaces`: NOCC orbital spaces.
/// - `gamma1`: Full-space spin-free one-body RDM.
/// - `lambdas`: Active-space spin-free cumulants.
/// # Returns:
/// - `f64`: Raw overlap element.
fn overlap_a_to_a(
    left: Excitation,
    right: Excitation,
    spaces: &Spaces,
    gamma1: &RDM1<f64>,
    lambdas: &Cumulants<f64>,
) -> f64 {
    let (v, u) = single(left);
    let (x, w) = single(right);

    0.5 * gamma1.data[u * gamma1.n + w] * theta(gamma1, x, v)
        + lambdas.lambda2.get(
            &[spaces.active_map[u].unwrap(), spaces.active_map[x].unwrap()],
            &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
        )
}

/// Evaluate the C4 CA -> AV / CA -> AV overlap block.
/// # Arguments:
/// - `left`: Left CA -> AV excitation.
/// - `right`: Right CA -> AV excitation.
/// - `spaces`: NOCC orbital spaces.
/// - `gamma1`: Full-space spin-free one-body RDM.
/// - `lambdas`: Active-space spin-free cumulants.
/// # Returns:
/// - `f64`: Raw overlap element.
fn overlap_ca_to_av(
    left: Excitation,
    right: Excitation,
    spaces: &Spaces,
    gamma1: &RDM1<f64>,
    lambdas: &Cumulants<f64>,
) -> f64 {
    let (v, a, i, u) = double(left);
    let (x, b, j, w) = double(right);
    delta(i, j)
        * delta(b, a)
        * (gamma1.data[u * gamma1.n + w] * theta(gamma1, x, v)
            - lambdas.lambda2.get(
                &[spaces.active_map[u].unwrap(), spaces.active_map[x].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            ))
}

/// Evaluate the C5 CA -> VA / CA -> VA overlap block.
/// # Arguments:
/// - `left`: Left CA -> VA excitation.
/// - `right`: Right CA -> VA excitation.
/// - `spaces`: NOCC orbital spaces.
/// - `gamma1`: Full-space spin-free one-body RDM.
/// - `lambdas`: Active-space spin-free cumulants.
/// # Returns:
/// - `f64`: Raw overlap element.
fn overlap_ca_to_va(
    left: Excitation,
    right: Excitation,
    spaces: &Spaces,
    gamma1: &RDM1<f64>,
    lambdas: &Cumulants<f64>,
) -> f64 {
    let (a, v, i, u) = double(left);
    let (b, x, j, w) = double(right);
    delta(i, j)
        * delta(b, a)
        * (gamma1.data[u * gamma1.n + w] * theta(gamma1, x, v)
            + 2.0
                * lambdas.lambda2.get(
                    &[spaces.active_map[u].unwrap(), spaces.active_map[x].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                ))
}

/// Evaluate the C6 CA -> VV / CA -> VV overlap block.
/// # Arguments:
/// - `left`: Left CA -> VV excitation.
/// - `right`: Right CA -> VV excitation.
/// - `gamma1`: Full-space spin-free one-body RDM.
/// # Returns:
/// - `f64`: Raw overlap element.
fn overlap_ca_to_vv(
    left: Excitation,
    right: Excitation,
    gamma1: &RDM1<f64>,
) -> f64 {
    let (a, b, i, u) = double(left);
    let (c, d, j, v) = double(right);
    delta(i, j)
        * gamma1.data[u * gamma1.n + v]
        * (2.0 * delta(d, b) * delta(c, a) - delta(d, a) * delta(c, b))
}

/// Evaluate the C7 CC -> AV / CC -> AV overlap block.
/// # Arguments:
/// - `left`: Left CC -> AV excitation.
/// - `right`: Right CC -> AV excitation.
/// - `gamma1`: Full-space spin-free one-body RDM.
/// # Returns:
/// - `f64`: Raw overlap element.
fn overlap_cc_to_av(
    left: Excitation,
    right: Excitation,
    gamma1: &RDM1<f64>,
) -> f64 {
    let (u, a, i, j) = double(left);
    let (v, b, k, l) = double(right);
    delta(b, a)
        * theta(gamma1, v, u)
        * (2.0 * delta(i, k) * delta(j, l) - delta(i, l) * delta(j, k))
}

/// Evaluate the C8 CC -> AA / CC -> AA overlap block.
/// # Arguments:
/// - `left`: Left CC -> AA excitation.
/// - `right`: Right CC -> AA excitation.
/// - `spaces`: NOCC orbital spaces.
/// - `gamma1`: Full-space spin-free one-body RDM.
/// - `lambdas`: Active-space spin-free cumulants.
/// # Returns:
/// - `f64`: Raw overlap element.
fn overlap_cc_to_aa(
    left: Excitation,
    right: Excitation,
    spaces: &Spaces,
    gamma1: &RDM1<f64>,
    lambdas: &Cumulants<f64>,
) -> f64 {
    let (u, v, i, j) = double(left);
    let (w, x, k, l) = double(right);

    (theta(gamma1, w, u) * theta(gamma1, x, v) - 0.5 * theta(gamma1, w, v) * theta(gamma1, x, u)
        + lambdas.lambda2.get(
            &[spaces.active_map[w].unwrap(), spaces.active_map[x].unwrap()],
            &[spaces.active_map[u].unwrap(), spaces.active_map[v].unwrap()],
        ))
        * delta(i, k)
        * delta(j, l)
        + (theta(gamma1, w, v) * theta(gamma1, x, u)
            - 0.5 * theta(gamma1, w, u) * theta(gamma1, x, v)
            + lambdas.lambda2.get(
                &[spaces.active_map[w].unwrap(), spaces.active_map[x].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[u].unwrap()],
            ))
            * delta(i, l)
            * delta(j, k)
}

/// Evaluate the C9 CA -> AA / CA -> AA overlap block.
/// # Arguments:
/// - `left`: Left CA -> AA excitation.
/// - `right`: Right CA -> AA excitation.
/// - `spaces`: NOCC orbital spaces.
/// - `gamma1`: Full-space spin-free one-body RDM.
/// - `lambdas`: Active-space spin-free cumulants.
/// # Returns:
/// - `f64`: Raw overlap element.
fn overlap_ca_to_aa(
    left: Excitation,
    right: Excitation,
    spaces: &Spaces,
    gamma1: &RDM1<f64>,
    lambdas: &Cumulants<f64>,
) -> f64 {
    let (v, w, i, u) = double(left);
    let (y, z, j, x) = double(right);

    delta(i, j)
        * (-lambdas.lambda3.get(
            &[
                spaces.active_map[u].unwrap(),
                spaces.active_map[y].unwrap(),
                spaces.active_map[z].unwrap(),
            ],
            &[
                spaces.active_map[w].unwrap(),
                spaces.active_map[v].unwrap(),
                spaces.active_map[x].unwrap(),
            ],
        ) - 0.5
            * theta(gamma1, y, w)
            * lambdas.lambda2.get(
                &[spaces.active_map[u].unwrap(), spaces.active_map[z].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[x].unwrap()],
            )
            - 0.5
                * theta(gamma1, z, w)
                * lambdas.lambda2.get(
                    &[spaces.active_map[u].unwrap(), spaces.active_map[y].unwrap()],
                    &[spaces.active_map[x].unwrap(), spaces.active_map[v].unwrap()],
                )
            - 0.5
                * theta(gamma1, z, v)
                * lambdas.lambda2.get(
                    &[spaces.active_map[u].unwrap(), spaces.active_map[y].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[x].unwrap()],
                )
            + theta(gamma1, y, v)
                * lambdas.lambda2.get(
                    &[spaces.active_map[u].unwrap(), spaces.active_map[z].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[x].unwrap()],
                )
            + 0.5
                * gamma1.data[u * gamma1.n + x]
                * (theta(gamma1, z, w) * theta(gamma1, y, v)
                    - 0.5 * theta(gamma1, z, v) * theta(gamma1, y, w)
                    + lambdas.lambda2.get(
                        &[spaces.active_map[y].unwrap(), spaces.active_map[z].unwrap()],
                        &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                    )))
}

/// Evaluate the C10 AA -> AV / AA -> AV overlap block.
/// # Arguments:
/// - `left`: Left AA -> AV excitation.
/// - `right`: Right AA -> AV excitation.
/// - `spaces`: NOCC orbital spaces.
/// - `gamma1`: Full-space spin-free one-body RDM.
/// - `lambdas`: Active-space spin-free cumulants.
/// # Returns:
/// - `f64`: Raw overlap element.
fn overlap_aa_to_av(
    left: Excitation,
    right: Excitation,
    spaces: &Spaces,
    gamma1: &RDM1<f64>,
    lambdas: &Cumulants<f64>,
) -> f64 {
    let (v, a, t, u) = double(left);
    let (z, b, x, y) = double(right);

    delta(b, a)
        * (0.5
            * theta(gamma1, z, v)
            * (gamma1.data[t * gamma1.n + x] * gamma1.data[u * gamma1.n + y]
                - 0.5 * gamma1.data[t * gamma1.n + y] * gamma1.data[u * gamma1.n + x]
                + lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[x].unwrap(), spaces.active_map[y].unwrap()],
                ))
            - 0.5
                * gamma1.data[t * gamma1.n + x]
                * lambdas.lambda2.get(
                    &[spaces.active_map[u].unwrap(), spaces.active_map[z].unwrap()],
                    &[spaces.active_map[y].unwrap(), spaces.active_map[v].unwrap()],
                )
            - 0.5
                * gamma1.data[t * gamma1.n + y]
                * lambdas.lambda2.get(
                    &[spaces.active_map[u].unwrap(), spaces.active_map[z].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[x].unwrap()],
                )
            + gamma1.data[u * gamma1.n + y]
                * lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[z].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[x].unwrap()],
                )
            - 0.5
                * gamma1.data[u * gamma1.n + x]
                * lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[z].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[y].unwrap()],
                )
            + lambdas.lambda3.get(
                &[
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                    spaces.active_map[z].unwrap(),
                ],
                &[
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[y].unwrap(),
                    spaces.active_map[x].unwrap(),
                ],
            ))
}

/// Evaluate the C11 AA -> VV / AA -> VV overlap block.
/// # Arguments:
/// - `left`: Left AA -> VV excitation.
/// - `right`: Right AA -> VV excitation.
/// - `spaces`: NOCC orbital spaces.
/// - `gamma1`: Full-space spin-free one-body RDM.
/// - `lambdas`: Active-space spin-free cumulants.
/// # Returns:
/// - `f64`: Raw overlap element.
fn overlap_aa_to_vv(
    left: Excitation,
    right: Excitation,
    spaces: &Spaces,
    gamma1: &RDM1<f64>,
    lambdas: &Cumulants<f64>,
) -> f64 {
    let (a, b, t, u) = double(left);
    let (c, d, v, w) = double(right);

    (gamma1.data[t * gamma1.n + v] * gamma1.data[u * gamma1.n + w]
        - 0.5 * gamma1.data[t * gamma1.n + w] * gamma1.data[u * gamma1.n + v]
        + lambdas.lambda2.get(
            &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
            &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
        ))
        * delta(d, b)
        * delta(c, a)
        + (gamma1.data[u * gamma1.n + v] * gamma1.data[t * gamma1.n + w]
            - 0.5 * gamma1.data[u * gamma1.n + w] * gamma1.data[t * gamma1.n + v]
            + lambdas.lambda2.get(
                &[spaces.active_map[u].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            ))
            * delta(c, b)
            * delta(d, a)
}

/// Evaluate the C12 AA -> AA / AA -> AA overlap block.
/// # Arguments:
/// - `left`: Left AA -> AA excitation.
/// - `right`: Right AA -> AA excitation.
/// - `spaces`: NOCC orbital spaces.
/// - `gamma1`: Full-space spin-free one-body RDM.
/// - `lambdas`: Active-space spin-free cumulants.
/// # Returns:
/// - `f64`: Raw overlap element.
fn overlap_aa_to_aa(
    left: Excitation,
    right: Excitation,
    spaces: &Spaces,
    gamma1: &RDM1<f64>,
    lambdas: &Cumulants<f64>,
) -> f64 {
    let (p, q, r, s) = double(left);
    let (t, u, v, w) = double(right);

    // The left excitation is daggered in S_mu_nu.
    // This evaluates <Phi | tau^{rs}_{pq} tau^{tu}_{vw} | Phi>.
    let mut out = 0.0;

    out += lambdas.lambda4.get(
        &[
            spaces.active_map[r].unwrap(),
            spaces.active_map[s].unwrap(),
            spaces.active_map[t].unwrap(),
            spaces.active_map[u].unwrap(),
        ],
        &[
            spaces.active_map[p].unwrap(),
            spaces.active_map[q].unwrap(),
            spaces.active_map[v].unwrap(),
            spaces.active_map[w].unwrap(),
        ],
    ) + (lambdas.lambda3.get(
        &[
            spaces.active_map[r].unwrap(),
            spaces.active_map[s].unwrap(),
            spaces.active_map[t].unwrap(),
        ],
        &[
            spaces.active_map[p].unwrap(),
            spaces.active_map[q].unwrap(),
            spaces.active_map[v].unwrap(),
        ],
    )) * (gamma1.data[u * gamma1.n + w])
        + ((-1.0 / 2.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                ],
                &[
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[w].unwrap(),
                ],
            )))
            * (gamma1.data[u * gamma1.n + v])
        + ((-1.0 / 2.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                ],
                &[
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[q].unwrap(),
                ],
            )))
            * (gamma1.data[u * gamma1.n + w])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                ],
                &[
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[w].unwrap(),
                ],
            )))
            * (gamma1.data[u * gamma1.n + q])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                ],
                &[
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[q].unwrap(),
                ],
            )))
            * (gamma1.data[u * gamma1.n + v])
        + ((-1.0 / 2.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                ],
                &[
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[v].unwrap(),
                ],
            )))
            * (gamma1.data[u * gamma1.n + q])
        + ((-1.0 / 2.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                ],
                &[
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[v].unwrap(),
                ],
            )))
            * (gamma1.data[u * gamma1.n + w])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                ],
                &[
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[w].unwrap(),
                ],
            )))
            * (gamma1.data[u * gamma1.n + v])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                ],
                &[
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[p].unwrap(),
                ],
            )))
            * (gamma1.data[u * gamma1.n + w])
        + ((-1.0 / 8.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                ],
                &[
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[w].unwrap(),
                ],
            )))
            * (gamma1.data[u * gamma1.n + p])
        + ((-1.0 / 8.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                ],
                &[
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[p].unwrap(),
                ],
            )))
            * (gamma1.data[u * gamma1.n + v])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                ],
                &[
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[v].unwrap(),
                ],
            )))
            * (gamma1.data[u * gamma1.n + p])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                ],
                &[
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[q].unwrap(),
                ],
            )))
            * (gamma1.data[u * gamma1.n + w])
        + ((-1.0 / 8.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                ],
                &[
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[w].unwrap(),
                ],
            )))
            * (gamma1.data[u * gamma1.n + q])
        + ((-1.0 / 2.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                ],
                &[
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[p].unwrap(),
                ],
            )))
            * (gamma1.data[u * gamma1.n + w])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                ],
                &[
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[w].unwrap(),
                ],
            )))
            * (gamma1.data[u * gamma1.n + p])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                ],
                &[
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[p].unwrap(),
                ],
            )))
            * (gamma1.data[u * gamma1.n + q])
        + ((-1.0 / 8.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                ],
                &[
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[q].unwrap(),
                ],
            )))
            * (gamma1.data[u * gamma1.n + p])
        + ((-1.0 / 8.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                ],
                &[
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[q].unwrap(),
                ],
            )))
            * (gamma1.data[u * gamma1.n + v])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                ],
                &[
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[v].unwrap(),
                ],
            )))
            * (gamma1.data[u * gamma1.n + q])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                ],
                &[
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[p].unwrap(),
                ],
            )))
            * (gamma1.data[u * gamma1.n + v])
        + ((-1.0 / 2.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                ],
                &[
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[v].unwrap(),
                ],
            )))
            * (gamma1.data[u * gamma1.n + p])
        + ((-1.0 / 8.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                ],
                &[
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[p].unwrap(),
                ],
            )))
            * (gamma1.data[u * gamma1.n + q])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                ],
                &[
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[q].unwrap(),
                ],
            )))
            * (gamma1.data[u * gamma1.n + p])
        + (lambdas.lambda3.get(
            &[
                spaces.active_map[r].unwrap(),
                spaces.active_map[s].unwrap(),
                spaces.active_map[u].unwrap(),
            ],
            &[
                spaces.active_map[p].unwrap(),
                spaces.active_map[q].unwrap(),
                spaces.active_map[w].unwrap(),
            ],
        )) * (gamma1.data[t * gamma1.n + v])
        + ((-1.0 / 2.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[v].unwrap(),
                ],
            )))
            * (gamma1.data[t * gamma1.n + w])
        + ((-1.0 / 2.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[w].unwrap(),
                ],
            )))
            * (gamma1.data[t * gamma1.n + q])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[q].unwrap(),
                ],
            )))
            * (gamma1.data[t * gamma1.n + w])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[v].unwrap(),
                ],
            )))
            * (gamma1.data[t * gamma1.n + q])
        + ((-1.0 / 2.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[q].unwrap(),
                ],
            )))
            * (gamma1.data[t * gamma1.n + v])
        + ((-1.0 / 2.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[w].unwrap(),
                ],
            )))
            * (gamma1.data[t * gamma1.n + v])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[v].unwrap(),
                ],
            )))
            * (gamma1.data[t * gamma1.n + w])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[w].unwrap(),
                ],
            )))
            * (gamma1.data[t * gamma1.n + p])
        + ((-1.0 / 8.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[p].unwrap(),
                ],
            )))
            * (gamma1.data[t * gamma1.n + w])
        + ((-1.0 / 8.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[v].unwrap(),
                ],
            )))
            * (gamma1.data[t * gamma1.n + p])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[p].unwrap(),
                ],
            )))
            * (gamma1.data[t * gamma1.n + v])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[w].unwrap(),
                ],
            )))
            * (gamma1.data[t * gamma1.n + q])
        + ((-1.0 / 8.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[q].unwrap(),
                ],
            )))
            * (gamma1.data[t * gamma1.n + w])
        + ((-1.0 / 2.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[w].unwrap(),
                ],
            )))
            * (gamma1.data[t * gamma1.n + p])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[p].unwrap(),
                ],
            )))
            * (gamma1.data[t * gamma1.n + w])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[q].unwrap(),
                ],
            )))
            * (gamma1.data[t * gamma1.n + p])
        + ((-1.0 / 8.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[p].unwrap(),
                ],
            )))
            * (gamma1.data[t * gamma1.n + q])
        + ((-1.0 / 8.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[v].unwrap(),
                ],
            )))
            * (gamma1.data[t * gamma1.n + q])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[q].unwrap(),
                ],
            )))
            * (gamma1.data[t * gamma1.n + v])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[v].unwrap(),
                ],
            )))
            * (gamma1.data[t * gamma1.n + p])
        + ((-1.0 / 2.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[p].unwrap(),
                ],
            )))
            * (gamma1.data[t * gamma1.n + v])
        + ((-1.0 / 8.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[q].unwrap(),
                ],
            )))
            * (gamma1.data[t * gamma1.n + p])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[p].unwrap(),
                ],
            )))
            * (gamma1.data[t * gamma1.n + q])
        + (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
        )) * (lambdas.lambda2.get(
            &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
            &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
        ))
        + ((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            ))
        + ((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
            ))
        + ((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
            ))
        + ((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            ))
        + ((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
            ))
        + ((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            ))
        + ((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
            ))
        + ((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
            ))
        + ((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
            ))
        + ((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            ))
        + ((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
            ))
        + ((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
            ))
        + ((lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
        )) * (gamma1.data[t * gamma1.n + v]))
            * (gamma1.data[u * gamma1.n + w])
        + (((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + w]))
            * (gamma1.data[u * gamma1.n + v])
        + (((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + q]))
            * (gamma1.data[u * gamma1.n + w])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + w]))
            * (gamma1.data[u * gamma1.n + q])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + q]))
            * (gamma1.data[u * gamma1.n + v])
        + (((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + v]))
            * (gamma1.data[u * gamma1.n + q])
        + (((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + v]))
            * (gamma1.data[u * gamma1.n + w])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + w]))
            * (gamma1.data[u * gamma1.n + v])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + p]))
            * (gamma1.data[u * gamma1.n + w])
        + (((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + w]))
            * (gamma1.data[u * gamma1.n + p])
        + (((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + p]))
            * (gamma1.data[u * gamma1.n + v])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + v]))
            * (gamma1.data[u * gamma1.n + p])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + q]))
            * (gamma1.data[u * gamma1.n + w])
        + (((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + w]))
            * (gamma1.data[u * gamma1.n + q])
        + (((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + p]))
            * (gamma1.data[u * gamma1.n + w])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + w]))
            * (gamma1.data[u * gamma1.n + p])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + p]))
            * (gamma1.data[u * gamma1.n + q])
        + (((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + q]))
            * (gamma1.data[u * gamma1.n + p])
        + (((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + q]))
            * (gamma1.data[u * gamma1.n + v])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + v]))
            * (gamma1.data[u * gamma1.n + q])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + p]))
            * (gamma1.data[u * gamma1.n + v])
        + (((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + v]))
            * (gamma1.data[u * gamma1.n + p])
        + (((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + p]))
            * (gamma1.data[u * gamma1.n + q])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + q]))
            * (gamma1.data[u * gamma1.n + p])
        + (lambdas.lambda3.get(
            &[
                spaces.active_map[r].unwrap(),
                spaces.active_map[t].unwrap(),
                spaces.active_map[u].unwrap(),
            ],
            &[
                spaces.active_map[p].unwrap(),
                spaces.active_map[v].unwrap(),
                spaces.active_map[w].unwrap(),
            ],
        )) * (gamma1.data[s * gamma1.n + q])
        + ((-1.0 / 2.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[v].unwrap(),
                ],
            )))
            * (gamma1.data[s * gamma1.n + q])
        + ((-1.0 / 2.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[w].unwrap(),
                ],
            )))
            * (gamma1.data[s * gamma1.n + v])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[q].unwrap(),
                ],
            )))
            * (gamma1.data[s * gamma1.n + v])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[v].unwrap(),
                ],
            )))
            * (gamma1.data[s * gamma1.n + w])
        + ((-1.0 / 2.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[q].unwrap(),
                ],
            )))
            * (gamma1.data[s * gamma1.n + w])
        + ((-1.0 / 2.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[w].unwrap(),
                ],
            )))
            * (gamma1.data[s * gamma1.n + p])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[v].unwrap(),
                ],
            )))
            * (gamma1.data[s * gamma1.n + p])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[w].unwrap(),
                ],
            )))
            * (gamma1.data[s * gamma1.n + v])
        + ((-1.0 / 8.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[p].unwrap(),
                ],
            )))
            * (gamma1.data[s * gamma1.n + v])
        + ((-1.0 / 8.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[v].unwrap(),
                ],
            )))
            * (gamma1.data[s * gamma1.n + w])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[p].unwrap(),
                ],
            )))
            * (gamma1.data[s * gamma1.n + w])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[w].unwrap(),
                ],
            )))
            * (gamma1.data[s * gamma1.n + p])
        + ((-1.0 / 8.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[q].unwrap(),
                ],
            )))
            * (gamma1.data[s * gamma1.n + p])
        + ((-1.0 / 2.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[w].unwrap(),
                ],
            )))
            * (gamma1.data[s * gamma1.n + q])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[p].unwrap(),
                ],
            )))
            * (gamma1.data[s * gamma1.n + q])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[q].unwrap(),
                ],
            )))
            * (gamma1.data[s * gamma1.n + w])
        + ((-1.0 / 8.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[p].unwrap(),
                ],
            )))
            * (gamma1.data[s * gamma1.n + w])
        + ((-1.0 / 8.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[v].unwrap(),
                ],
            )))
            * (gamma1.data[s * gamma1.n + p])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[q].unwrap(),
                ],
            )))
            * (gamma1.data[s * gamma1.n + p])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[v].unwrap(),
                ],
            )))
            * (gamma1.data[s * gamma1.n + q])
        + ((-1.0 / 2.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[p].unwrap(),
                ],
            )))
            * (gamma1.data[s * gamma1.n + q])
        + ((-1.0 / 8.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[q].unwrap(),
                ],
            )))
            * (gamma1.data[s * gamma1.n + v])
        + ((1.0 / 4.0)
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[r].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[p].unwrap(),
                ],
            )))
            * (gamma1.data[s * gamma1.n + v])
        + (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
        )) * (lambdas.lambda2.get(
            &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
            &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
        ))
        + ((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
            ))
        + ((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            ))
        + ((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
            ))
        + ((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            ))
        + ((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
            ))
        + ((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            ))
        + ((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
            ))
        + ((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
            ))
        + ((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
            ))
        + ((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
            ))
        + ((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
            ))
        + ((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
            ))
        + ((lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
        )) * (gamma1.data[s * gamma1.n + q]))
            * (gamma1.data[u * gamma1.n + w])
        + (((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + q]))
            * (gamma1.data[u * gamma1.n + v])
        + (((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + v]))
            * (gamma1.data[u * gamma1.n + w])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + v]))
            * (gamma1.data[u * gamma1.n + q])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + w]))
            * (gamma1.data[u * gamma1.n + v])
        + (((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + w]))
            * (gamma1.data[u * gamma1.n + q])
        + (((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + p]))
            * (gamma1.data[u * gamma1.n + w])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + p]))
            * (gamma1.data[u * gamma1.n + v])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + v]))
            * (gamma1.data[u * gamma1.n + w])
        + (((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + v]))
            * (gamma1.data[u * gamma1.n + p])
        + (((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + w]))
            * (gamma1.data[u * gamma1.n + v])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + w]))
            * (gamma1.data[u * gamma1.n + p])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + p]))
            * (gamma1.data[u * gamma1.n + w])
        + (((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + p]))
            * (gamma1.data[u * gamma1.n + q])
        + (((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + q]))
            * (gamma1.data[u * gamma1.n + w])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + q]))
            * (gamma1.data[u * gamma1.n + p])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + w]))
            * (gamma1.data[u * gamma1.n + q])
        + (((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + w]))
            * (gamma1.data[u * gamma1.n + p])
        + (((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + p]))
            * (gamma1.data[u * gamma1.n + v])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + p]))
            * (gamma1.data[u * gamma1.n + q])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + q]))
            * (gamma1.data[u * gamma1.n + v])
        + (((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + q]))
            * (gamma1.data[u * gamma1.n + p])
        + (((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + v]))
            * (gamma1.data[u * gamma1.n + q])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + v]))
            * (gamma1.data[u * gamma1.n + p])
        + (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
        )) * (lambdas.lambda2.get(
            &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
        ))
        + ((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            ))
        + ((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
            ))
        + ((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            ))
        + ((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
            ))
        + ((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            ))
        + ((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
            ))
        + ((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            ))
        + ((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
            ))
        + ((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
            ))
        + ((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
            ))
        + ((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
            ))
        + ((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
            ))
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
            ))
        + (gamma1.data[r * gamma1.n + p])
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[w].unwrap(),
                ],
            ))
        + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p]))
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[v].unwrap(),
                ],
            ))
        + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p]))
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[w].unwrap(),
                ],
            ))
        + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + p]))
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[q].unwrap(),
                ],
            ))
        + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + p]))
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[v].unwrap(),
                ],
            ))
        + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p]))
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[q].unwrap(),
                ],
            ))
        + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q]))
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[w].unwrap(),
                ],
            ))
        + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q]))
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[v].unwrap(),
                ],
            ))
        + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q]))
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[w].unwrap(),
                ],
            ))
        + ((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + q]))
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[p].unwrap(),
                ],
            ))
        + ((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + q]))
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[v].unwrap(),
                ],
            ))
        + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q]))
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[p].unwrap(),
                ],
            ))
        + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v]))
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[w].unwrap(),
                ],
            ))
        + ((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + v]))
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[q].unwrap(),
                ],
            ))
        + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v]))
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[w].unwrap(),
                ],
            ))
        + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v]))
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[p].unwrap(),
                ],
            ))
        + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v]))
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[q].unwrap(),
                ],
            ))
        + ((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + v]))
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[w].unwrap(),
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[p].unwrap(),
                ],
            ))
        + ((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + w]))
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[v].unwrap(),
                ],
            ))
        + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w]))
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[q].unwrap(),
                ],
            ))
        + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w]))
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[v].unwrap(),
                ],
            ))
        + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w]))
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[p].unwrap(),
                ],
            ))
        + ((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + w]))
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[p].unwrap(),
                    spaces.active_map[q].unwrap(),
                ],
            ))
        + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w]))
            * (lambdas.lambda3.get(
                &[
                    spaces.active_map[s].unwrap(),
                    spaces.active_map[t].unwrap(),
                    spaces.active_map[u].unwrap(),
                ],
                &[
                    spaces.active_map[v].unwrap(),
                    spaces.active_map[q].unwrap(),
                    spaces.active_map[p].unwrap(),
                ],
            ))
        + ((gamma1.data[r * gamma1.n + p])
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + w])
        + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + v])
        + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + w])
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + p]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + q])
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + p]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + v])
        + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + q])
        + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + w])
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + v])
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + w])
        + (((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + q]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + p])
        + (((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + q]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + v])
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + p])
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + w])
        + (((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + v]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + q])
        + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + w])
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + p])
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + q])
        + (((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + v]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + p])
        + (((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + w]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + v])
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + q])
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + v])
        + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + p])
        + (((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + w]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + q])
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + p])
        + ((lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
        )) * (gamma1.data[s * gamma1.n + q]))
            * (gamma1.data[t * gamma1.n + v])
        + (((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + q]))
            * (gamma1.data[t * gamma1.n + w])
        + (((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + v]))
            * (gamma1.data[t * gamma1.n + q])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + v]))
            * (gamma1.data[t * gamma1.n + w])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + w]))
            * (gamma1.data[t * gamma1.n + q])
        + (((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + w]))
            * (gamma1.data[t * gamma1.n + v])
        + (((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + p]))
            * (gamma1.data[t * gamma1.n + v])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + p]))
            * (gamma1.data[t * gamma1.n + w])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + v]))
            * (gamma1.data[t * gamma1.n + p])
        + (((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + v]))
            * (gamma1.data[t * gamma1.n + w])
        + (((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + w]))
            * (gamma1.data[t * gamma1.n + p])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + w]))
            * (gamma1.data[t * gamma1.n + v])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + p]))
            * (gamma1.data[t * gamma1.n + q])
        + (((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + p]))
            * (gamma1.data[t * gamma1.n + w])
        + (((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + q]))
            * (gamma1.data[t * gamma1.n + p])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + q]))
            * (gamma1.data[t * gamma1.n + w])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + w]))
            * (gamma1.data[t * gamma1.n + p])
        + (((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + w]))
            * (gamma1.data[t * gamma1.n + q])
        + (((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + p]))
            * (gamma1.data[t * gamma1.n + q])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + p]))
            * (gamma1.data[t * gamma1.n + v])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + q]))
            * (gamma1.data[t * gamma1.n + p])
        + (((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + q]))
            * (gamma1.data[t * gamma1.n + v])
        + (((-1.0 / 8.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + v]))
            * (gamma1.data[t * gamma1.n + p])
        + (((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[s * gamma1.n + v]))
            * (gamma1.data[t * gamma1.n + q])
        + ((gamma1.data[r * gamma1.n + p])
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + v])
        + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + w])
        + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + q])
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + p]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + w])
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + p]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + q])
        + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + v])
        + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + v])
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + w])
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + p])
        + (((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + q]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + w])
        + (((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + q]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + p])
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + v])
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + q])
        + (((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + v]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + w])
        + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + p])
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + w])
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + p])
        + (((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + v]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + q])
        + (((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + w]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + q])
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + v])
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + p])
        + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + v])
        + (((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + w]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + p])
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + q])
        + ((gamma1.data[r * gamma1.n + p]) * (gamma1.data[s * gamma1.n + q]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            ))
        + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p])) * (gamma1.data[s * gamma1.n + q]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            ))
        + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p])) * (gamma1.data[s * gamma1.n + v]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            ))
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + p])) * (gamma1.data[s * gamma1.n + v]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
            ))
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + p])) * (gamma1.data[s * gamma1.n + w]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
            ))
        + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p])) * (gamma1.data[s * gamma1.n + w]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
            ))
        + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + p]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            ))
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + p]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            ))
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + v]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            ))
        + (((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + v]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
            ))
        + (((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + w]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            ))
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + w]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
            ))
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[s * gamma1.n + p]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            ))
        + (((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[s * gamma1.n + p]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
            ))
        + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[s * gamma1.n + q]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            ))
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[s * gamma1.n + q]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
            ))
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[s * gamma1.n + w]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
            ))
        + (((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[s * gamma1.n + w]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
            ))
        + (((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + p]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
            ))
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + p]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
            ))
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + q]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            ))
        + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + q]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
            ))
        + (((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + v]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
            ))
        + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + v]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
            ))
        + (((gamma1.data[r * gamma1.n + p]) * (gamma1.data[s * gamma1.n + q]))
            * (gamma1.data[t * gamma1.n + v]))
            * (gamma1.data[u * gamma1.n + w])
        + ((((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p])) * (gamma1.data[s * gamma1.n + q]))
            * (gamma1.data[t * gamma1.n + w]))
            * (gamma1.data[u * gamma1.n + v])
        + ((((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p])) * (gamma1.data[s * gamma1.n + v]))
            * (gamma1.data[t * gamma1.n + q]))
            * (gamma1.data[u * gamma1.n + w])
        + ((((1.0 / 4.0) * (gamma1.data[r * gamma1.n + p])) * (gamma1.data[s * gamma1.n + v]))
            * (gamma1.data[t * gamma1.n + w]))
            * (gamma1.data[u * gamma1.n + q])
        + ((((1.0 / 4.0) * (gamma1.data[r * gamma1.n + p])) * (gamma1.data[s * gamma1.n + w]))
            * (gamma1.data[t * gamma1.n + q]))
            * (gamma1.data[u * gamma1.n + v])
        + ((((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p])) * (gamma1.data[s * gamma1.n + w]))
            * (gamma1.data[t * gamma1.n + v]))
            * (gamma1.data[u * gamma1.n + q])
        + ((((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + p]))
            * (gamma1.data[t * gamma1.n + v]))
            * (gamma1.data[u * gamma1.n + w])
        + ((((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + p]))
            * (gamma1.data[t * gamma1.n + w]))
            * (gamma1.data[u * gamma1.n + v])
        + ((((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + v]))
            * (gamma1.data[t * gamma1.n + p]))
            * (gamma1.data[u * gamma1.n + w])
        + ((((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + v]))
            * (gamma1.data[t * gamma1.n + w]))
            * (gamma1.data[u * gamma1.n + p])
        + ((((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + w]))
            * (gamma1.data[t * gamma1.n + p]))
            * (gamma1.data[u * gamma1.n + v])
        + ((((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + w]))
            * (gamma1.data[t * gamma1.n + v]))
            * (gamma1.data[u * gamma1.n + p])
        + ((((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[s * gamma1.n + p]))
            * (gamma1.data[t * gamma1.n + q]))
            * (gamma1.data[u * gamma1.n + w])
        + ((((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[s * gamma1.n + p]))
            * (gamma1.data[t * gamma1.n + w]))
            * (gamma1.data[u * gamma1.n + q])
        + ((((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[s * gamma1.n + q]))
            * (gamma1.data[t * gamma1.n + p]))
            * (gamma1.data[u * gamma1.n + w])
        + ((((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[s * gamma1.n + q]))
            * (gamma1.data[t * gamma1.n + w]))
            * (gamma1.data[u * gamma1.n + p])
        + ((((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[s * gamma1.n + w]))
            * (gamma1.data[t * gamma1.n + p]))
            * (gamma1.data[u * gamma1.n + q])
        + ((((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[s * gamma1.n + w]))
            * (gamma1.data[t * gamma1.n + q]))
            * (gamma1.data[u * gamma1.n + p])
        + ((((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + p]))
            * (gamma1.data[t * gamma1.n + q]))
            * (gamma1.data[u * gamma1.n + v])
        + ((((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + p]))
            * (gamma1.data[t * gamma1.n + v]))
            * (gamma1.data[u * gamma1.n + q])
        + ((((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + q]))
            * (gamma1.data[t * gamma1.n + p]))
            * (gamma1.data[u * gamma1.n + v])
        + ((((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + q]))
            * (gamma1.data[t * gamma1.n + v]))
            * (gamma1.data[u * gamma1.n + p])
        + ((((-1.0 / 8.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + v]))
            * (gamma1.data[t * gamma1.n + p]))
            * (gamma1.data[u * gamma1.n + q])
        + ((((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + v]))
            * (gamma1.data[t * gamma1.n + q]))
            * (gamma1.data[u * gamma1.n + p]);
    out += (delta(p, s))
        * (lambdas.lambda3.get(
            &[
                spaces.active_map[r].unwrap(),
                spaces.active_map[t].unwrap(),
                spaces.active_map[u].unwrap(),
            ],
            &[
                spaces.active_map[q].unwrap(),
                spaces.active_map[v].unwrap(),
                spaces.active_map[w].unwrap(),
            ],
        ) + (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
        )) * (gamma1.data[u * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + w])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + q])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + q])
            + (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            )) * (gamma1.data[t * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + q])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + w])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + v])
            + (gamma1.data[r * gamma1.n + q])
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
                ))
            + ((gamma1.data[r * gamma1.n + q]) * (gamma1.data[t * gamma1.n + v]))
                * (gamma1.data[u * gamma1.n + w])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[t * gamma1.n + w]))
                * (gamma1.data[u * gamma1.n + v])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[t * gamma1.n + q]))
                * (gamma1.data[u * gamma1.n + w])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[t * gamma1.n + w]))
                * (gamma1.data[u * gamma1.n + q])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[t * gamma1.n + q]))
                * (gamma1.data[u * gamma1.n + v])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[t * gamma1.n + v]))
                * (gamma1.data[u * gamma1.n + q]));
    out += (delta(p, t))
        * (lambdas.lambda3.get(
            &[
                spaces.active_map[r].unwrap(),
                spaces.active_map[s].unwrap(),
                spaces.active_map[u].unwrap(),
            ],
            &[
                spaces.active_map[v].unwrap(),
                spaces.active_map[q].unwrap(),
                spaces.active_map[w].unwrap(),
            ],
        ) + (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
            &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
        )) * (gamma1.data[u * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + w])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + v])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + v])
            + (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            )) * (gamma1.data[s * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + v])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + w])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + q])
            + (gamma1.data[r * gamma1.n + v])
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
                ))
            + ((gamma1.data[r * gamma1.n + v]) * (gamma1.data[s * gamma1.n + q]))
                * (gamma1.data[u * gamma1.n + w])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[s * gamma1.n + w]))
                * (gamma1.data[u * gamma1.n + q])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + v]))
                * (gamma1.data[u * gamma1.n + w])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + w]))
                * (gamma1.data[u * gamma1.n + v])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + v]))
                * (gamma1.data[u * gamma1.n + q])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + q]))
                * (gamma1.data[u * gamma1.n + v]));
    out += (delta(p, u))
        * (lambdas.lambda3.get(
            &[
                spaces.active_map[r].unwrap(),
                spaces.active_map[s].unwrap(),
                spaces.active_map[t].unwrap(),
            ],
            &[
                spaces.active_map[w].unwrap(),
                spaces.active_map[q].unwrap(),
                spaces.active_map[v].unwrap(),
            ],
        ) + (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
            &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
        )) * (gamma1.data[t * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + v])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + w])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + w])
            + (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            )) * (gamma1.data[s * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + w])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + v])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + q])
            + (gamma1.data[r * gamma1.n + w])
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((gamma1.data[r * gamma1.n + w]) * (gamma1.data[s * gamma1.n + q]))
                * (gamma1.data[t * gamma1.n + v])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + v]))
                * (gamma1.data[t * gamma1.n + q])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + w]))
                * (gamma1.data[t * gamma1.n + v])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + v]))
                * (gamma1.data[t * gamma1.n + w])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[s * gamma1.n + w]))
                * (gamma1.data[t * gamma1.n + q])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[s * gamma1.n + q]))
                * (gamma1.data[t * gamma1.n + w]));
    out += (delta(q, t))
        * (lambdas.lambda3.get(
            &[
                spaces.active_map[r].unwrap(),
                spaces.active_map[s].unwrap(),
                spaces.active_map[u].unwrap(),
            ],
            &[
                spaces.active_map[p].unwrap(),
                spaces.active_map[v].unwrap(),
                spaces.active_map[w].unwrap(),
            ],
        ) + (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
        )) * (gamma1.data[u * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + w])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + p])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + p])
            + (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            )) * (gamma1.data[s * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + p])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + w])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + p])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + v])
            + (gamma1.data[r * gamma1.n + p])
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
                ))
            + ((gamma1.data[r * gamma1.n + p]) * (gamma1.data[s * gamma1.n + v]))
                * (gamma1.data[u * gamma1.n + w])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p])) * (gamma1.data[s * gamma1.n + w]))
                * (gamma1.data[u * gamma1.n + v])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[s * gamma1.n + p]))
                * (gamma1.data[u * gamma1.n + w])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[s * gamma1.n + w]))
                * (gamma1.data[u * gamma1.n + p])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + p]))
                * (gamma1.data[u * gamma1.n + v])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + v]))
                * (gamma1.data[u * gamma1.n + p]));
    out += ((delta(q, t)) * (delta(p, s)))
        * (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
            &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
        ) + (gamma1.data[r * gamma1.n + v]) * (gamma1.data[u * gamma1.n + w])
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[u * gamma1.n + v]));
    out += ((delta(q, t)) * (delta(p, u)))
        * (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
            &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
        ) + (gamma1.data[r * gamma1.n + w]) * (gamma1.data[s * gamma1.n + v])
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[s * gamma1.n + w]));
    out += (delta(q, u))
        * (lambdas.lambda3.get(
            &[
                spaces.active_map[r].unwrap(),
                spaces.active_map[s].unwrap(),
                spaces.active_map[t].unwrap(),
            ],
            &[
                spaces.active_map[p].unwrap(),
                spaces.active_map[w].unwrap(),
                spaces.active_map[v].unwrap(),
            ],
        ) + (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
        )) * (gamma1.data[t * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + v])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + p])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + p])
            + (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            )) * (gamma1.data[s * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + p])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + v])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + p])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + w])
            + (gamma1.data[r * gamma1.n + p])
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
                ))
            + ((gamma1.data[r * gamma1.n + p]) * (gamma1.data[s * gamma1.n + w]))
                * (gamma1.data[t * gamma1.n + v])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p])) * (gamma1.data[s * gamma1.n + v]))
                * (gamma1.data[t * gamma1.n + w])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + p]))
                * (gamma1.data[t * gamma1.n + v])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + v]))
                * (gamma1.data[t * gamma1.n + p])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[s * gamma1.n + p]))
                * (gamma1.data[t * gamma1.n + w])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[s * gamma1.n + w]))
                * (gamma1.data[t * gamma1.n + p]));
    out += ((delta(q, u)) * (delta(p, s)))
        * (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
        ) + (gamma1.data[r * gamma1.n + w]) * (gamma1.data[t * gamma1.n + v])
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[t * gamma1.n + w]));
    out += ((delta(q, u)) * (delta(p, t)))
        * (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
            &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
        ) + (gamma1.data[r * gamma1.n + v]) * (gamma1.data[s * gamma1.n + w])
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + v]));
    out += (delta(v, u))
        * (lambdas.lambda3.get(
            &[
                spaces.active_map[r].unwrap(),
                spaces.active_map[s].unwrap(),
                spaces.active_map[t].unwrap(),
            ],
            &[
                spaces.active_map[p].unwrap(),
                spaces.active_map[q].unwrap(),
                spaces.active_map[w].unwrap(),
            ],
        ) + (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
        )) * (gamma1.data[t * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + w])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + p])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + p])
            + (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            )) * (gamma1.data[s * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + p])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + w])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + p])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + q])
            + (gamma1.data[r * gamma1.n + p])
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
                ))
            + ((gamma1.data[r * gamma1.n + p]) * (gamma1.data[s * gamma1.n + q]))
                * (gamma1.data[t * gamma1.n + w])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p])) * (gamma1.data[s * gamma1.n + w]))
                * (gamma1.data[t * gamma1.n + q])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + p]))
                * (gamma1.data[t * gamma1.n + w])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + w]))
                * (gamma1.data[t * gamma1.n + p])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + p]))
                * (gamma1.data[t * gamma1.n + q])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + q]))
                * (gamma1.data[t * gamma1.n + p]));
    out += ((delta(v, u)) * (delta(p, s)))
        * (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
        ) + (gamma1.data[r * gamma1.n + q]) * (gamma1.data[t * gamma1.n + w])
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[t * gamma1.n + q]));
    out += ((delta(v, u)) * (delta(p, t)))
        * (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
            &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
        ) + (gamma1.data[r * gamma1.n + w]) * (gamma1.data[s * gamma1.n + q])
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + w]));
    out += ((delta(v, u)) * (delta(q, t)))
        * (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
        ) + (gamma1.data[r * gamma1.n + p]) * (gamma1.data[s * gamma1.n + w])
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + p]));
    out += (((delta(v, u)) * (delta(q, t))) * (delta(p, s))) * (gamma1.data[r * gamma1.n + w]);
    out += (-delta(v, u))
        * (lambdas.lambda3.get(
            &[
                spaces.active_map[r].unwrap(),
                spaces.active_map[s].unwrap(),
                spaces.active_map[t].unwrap(),
            ],
            &[
                spaces.active_map[p].unwrap(),
                spaces.active_map[q].unwrap(),
                spaces.active_map[w].unwrap(),
            ],
        ) + (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
        )) * (gamma1.data[t * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + w])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + p])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + p])
            + (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            )) * (gamma1.data[s * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + p])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + w])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + p])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + q])
            + (gamma1.data[r * gamma1.n + p])
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
                ))
            + ((gamma1.data[r * gamma1.n + p]) * (gamma1.data[s * gamma1.n + q]))
                * (gamma1.data[t * gamma1.n + w])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p])) * (gamma1.data[s * gamma1.n + w]))
                * (gamma1.data[t * gamma1.n + q])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + p]))
                * (gamma1.data[t * gamma1.n + w])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + w]))
                * (gamma1.data[t * gamma1.n + p])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + p]))
                * (gamma1.data[t * gamma1.n + q])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + q]))
                * (gamma1.data[t * gamma1.n + p]));
    out += (-delta(v, u))
        * ((delta(p, s))
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            ) + (gamma1.data[r * gamma1.n + q]) * (gamma1.data[t * gamma1.n + w])
                + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w]))
                    * (gamma1.data[t * gamma1.n + q])));
    out += (-delta(v, u))
        * ((delta(p, t))
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
            ) + (gamma1.data[r * gamma1.n + w]) * (gamma1.data[s * gamma1.n + q])
                + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q]))
                    * (gamma1.data[s * gamma1.n + w])));
    out += (-delta(v, u))
        * ((delta(q, t))
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            ) + (gamma1.data[r * gamma1.n + p]) * (gamma1.data[s * gamma1.n + w])
                + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w]))
                    * (gamma1.data[s * gamma1.n + p])));
    out += (-delta(v, u)) * (((delta(q, t)) * (delta(p, s))) * (gamma1.data[r * gamma1.n + w]));
    out += (-0.5 * gamma1.data[t * gamma1.n + v])
        * (lambdas.lambda3.get(
            &[
                spaces.active_map[r].unwrap(),
                spaces.active_map[s].unwrap(),
                spaces.active_map[u].unwrap(),
            ],
            &[
                spaces.active_map[p].unwrap(),
                spaces.active_map[q].unwrap(),
                spaces.active_map[w].unwrap(),
            ],
        ) + (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
        )) * (gamma1.data[u * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + w])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + p])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + p])
            + (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            )) * (gamma1.data[s * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + p])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + w])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + p])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + q])
            + (gamma1.data[r * gamma1.n + p])
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
                ))
            + ((gamma1.data[r * gamma1.n + p]) * (gamma1.data[s * gamma1.n + q]))
                * (gamma1.data[u * gamma1.n + w])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p])) * (gamma1.data[s * gamma1.n + w]))
                * (gamma1.data[u * gamma1.n + q])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + p]))
                * (gamma1.data[u * gamma1.n + w])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + w]))
                * (gamma1.data[u * gamma1.n + p])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + p]))
                * (gamma1.data[u * gamma1.n + q])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + q]))
                * (gamma1.data[u * gamma1.n + p]));
    out += (-0.5 * gamma1.data[t * gamma1.n + v])
        * ((delta(p, s))
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            ) + (gamma1.data[r * gamma1.n + q]) * (gamma1.data[u * gamma1.n + w])
                + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w]))
                    * (gamma1.data[u * gamma1.n + q])));
    out += (-0.5 * gamma1.data[t * gamma1.n + v])
        * ((delta(p, u))
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
            ) + (gamma1.data[r * gamma1.n + w]) * (gamma1.data[s * gamma1.n + q])
                + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q]))
                    * (gamma1.data[s * gamma1.n + w])));
    out += (-0.5 * gamma1.data[t * gamma1.n + v])
        * ((delta(q, u))
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            ) + (gamma1.data[r * gamma1.n + p]) * (gamma1.data[s * gamma1.n + w])
                + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w]))
                    * (gamma1.data[s * gamma1.n + p])));
    out += (-0.5 * gamma1.data[t * gamma1.n + v])
        * (((delta(q, u)) * (delta(p, s))) * (gamma1.data[r * gamma1.n + w]));
    out += (-0.5 * gamma1.data[u * gamma1.n + w])
        * (lambdas.lambda3.get(
            &[
                spaces.active_map[r].unwrap(),
                spaces.active_map[s].unwrap(),
                spaces.active_map[t].unwrap(),
            ],
            &[
                spaces.active_map[p].unwrap(),
                spaces.active_map[q].unwrap(),
                spaces.active_map[v].unwrap(),
            ],
        ) + (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
        )) * (gamma1.data[t * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + v])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + p])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + p])
            + (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            )) * (gamma1.data[s * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + p])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + v])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + p])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + q])
            + (gamma1.data[r * gamma1.n + p])
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
                ))
            + ((gamma1.data[r * gamma1.n + p]) * (gamma1.data[s * gamma1.n + q]))
                * (gamma1.data[t * gamma1.n + v])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p])) * (gamma1.data[s * gamma1.n + v]))
                * (gamma1.data[t * gamma1.n + q])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + p]))
                * (gamma1.data[t * gamma1.n + v])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + v]))
                * (gamma1.data[t * gamma1.n + p])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[s * gamma1.n + p]))
                * (gamma1.data[t * gamma1.n + q])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[s * gamma1.n + q]))
                * (gamma1.data[t * gamma1.n + p]));
    out += (-0.5 * gamma1.data[u * gamma1.n + w])
        * ((delta(p, s))
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
            ) + (gamma1.data[r * gamma1.n + q]) * (gamma1.data[t * gamma1.n + v])
                + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v]))
                    * (gamma1.data[t * gamma1.n + q])));
    out += (-0.5 * gamma1.data[u * gamma1.n + w])
        * ((delta(p, t))
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
            ) + (gamma1.data[r * gamma1.n + v]) * (gamma1.data[s * gamma1.n + q])
                + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q]))
                    * (gamma1.data[s * gamma1.n + v])));
    out += (-0.5 * gamma1.data[u * gamma1.n + w])
        * ((delta(q, t))
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            ) + (gamma1.data[r * gamma1.n + p]) * (gamma1.data[s * gamma1.n + v])
                + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v]))
                    * (gamma1.data[s * gamma1.n + p])));
    out += (-0.5 * gamma1.data[u * gamma1.n + w])
        * (((delta(q, t)) * (delta(p, s))) * (gamma1.data[r * gamma1.n + v]));
    out += (0.5 * gamma1.data[u * gamma1.n + v])
        * (lambdas.lambda3.get(
            &[
                spaces.active_map[r].unwrap(),
                spaces.active_map[s].unwrap(),
                spaces.active_map[t].unwrap(),
            ],
            &[
                spaces.active_map[p].unwrap(),
                spaces.active_map[q].unwrap(),
                spaces.active_map[w].unwrap(),
            ],
        ) + (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
        )) * (gamma1.data[t * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + w])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + p])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + p])
            + (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            )) * (gamma1.data[s * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + p])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + w])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + p])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + q])
            + (gamma1.data[r * gamma1.n + p])
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
                ))
            + ((gamma1.data[r * gamma1.n + p]) * (gamma1.data[s * gamma1.n + q]))
                * (gamma1.data[t * gamma1.n + w])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p])) * (gamma1.data[s * gamma1.n + w]))
                * (gamma1.data[t * gamma1.n + q])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + p]))
                * (gamma1.data[t * gamma1.n + w])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + w]))
                * (gamma1.data[t * gamma1.n + p])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + p]))
                * (gamma1.data[t * gamma1.n + q])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[s * gamma1.n + q]))
                * (gamma1.data[t * gamma1.n + p]));
    out += (0.5 * gamma1.data[u * gamma1.n + v])
        * ((delta(p, s))
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            ) + (gamma1.data[r * gamma1.n + q]) * (gamma1.data[t * gamma1.n + w])
                + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w]))
                    * (gamma1.data[t * gamma1.n + q])));
    out += (0.5 * gamma1.data[u * gamma1.n + v])
        * ((delta(p, t))
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
            ) + (gamma1.data[r * gamma1.n + w]) * (gamma1.data[s * gamma1.n + q])
                + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q]))
                    * (gamma1.data[s * gamma1.n + w])));
    out += (0.5 * gamma1.data[u * gamma1.n + v])
        * ((delta(q, t))
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            ) + (gamma1.data[r * gamma1.n + p]) * (gamma1.data[s * gamma1.n + w])
                + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w]))
                    * (gamma1.data[s * gamma1.n + p])));
    out += (0.5 * gamma1.data[u * gamma1.n + v])
        * (((delta(q, t)) * (delta(p, s))) * (gamma1.data[r * gamma1.n + w]));
    out += (0.5 * gamma1.data[t * gamma1.n + w])
        * (lambdas.lambda3.get(
            &[
                spaces.active_map[r].unwrap(),
                spaces.active_map[s].unwrap(),
                spaces.active_map[u].unwrap(),
            ],
            &[
                spaces.active_map[p].unwrap(),
                spaces.active_map[q].unwrap(),
                spaces.active_map[v].unwrap(),
            ],
        ) + (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
        )) * (gamma1.data[u * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + v])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + p])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + p])
            + (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            )) * (gamma1.data[s * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + p])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + v])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + p])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[s * gamma1.n + q])
            + (gamma1.data[r * gamma1.n + p])
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[p].unwrap()],
                ))
            + ((gamma1.data[r * gamma1.n + p]) * (gamma1.data[s * gamma1.n + q]))
                * (gamma1.data[u * gamma1.n + v])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p])) * (gamma1.data[s * gamma1.n + v]))
                * (gamma1.data[u * gamma1.n + q])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + p]))
                * (gamma1.data[u * gamma1.n + v])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + v]))
                * (gamma1.data[u * gamma1.n + p])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[s * gamma1.n + p]))
                * (gamma1.data[u * gamma1.n + q])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[s * gamma1.n + q]))
                * (gamma1.data[u * gamma1.n + p]));
    out += (0.5 * gamma1.data[t * gamma1.n + w])
        * ((delta(p, s))
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
            ) + (gamma1.data[r * gamma1.n + q]) * (gamma1.data[u * gamma1.n + v])
                + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v]))
                    * (gamma1.data[u * gamma1.n + q])));
    out += (0.5 * gamma1.data[t * gamma1.n + w])
        * ((delta(p, u))
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
            ) + (gamma1.data[r * gamma1.n + v]) * (gamma1.data[s * gamma1.n + q])
                + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q]))
                    * (gamma1.data[s * gamma1.n + v])));
    out += (0.5 * gamma1.data[t * gamma1.n + w])
        * ((delta(q, u))
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            ) + (gamma1.data[r * gamma1.n + p]) * (gamma1.data[s * gamma1.n + v])
                + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v]))
                    * (gamma1.data[s * gamma1.n + p])));
    out += (0.5 * gamma1.data[t * gamma1.n + w])
        * (((delta(q, u)) * (delta(p, s))) * (gamma1.data[r * gamma1.n + v]));
    out += (2.0 * gamma1.data[t * gamma1.n + v] * gamma1.data[u * gamma1.n + w]
        - 1.5 * gamma1.data[t * gamma1.n + w] * gamma1.data[u * gamma1.n + v]
        + lambdas.lambda2.get(
            &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
            &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
        ))
        * (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
        ) + (gamma1.data[r * gamma1.n + p]) * (gamma1.data[s * gamma1.n + q])
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[s * gamma1.n + p]));
    out += (2.0 * gamma1.data[t * gamma1.n + v] * gamma1.data[u * gamma1.n + w]
        - 1.5 * gamma1.data[t * gamma1.n + w] * gamma1.data[u * gamma1.n + v]
        + lambdas.lambda2.get(
            &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
            &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
        ))
        * ((delta(p, s)) * (gamma1.data[r * gamma1.n + q]));
    out += (-delta(p, s))
        * (lambdas.lambda3.get(
            &[
                spaces.active_map[r].unwrap(),
                spaces.active_map[t].unwrap(),
                spaces.active_map[u].unwrap(),
            ],
            &[
                spaces.active_map[q].unwrap(),
                spaces.active_map[v].unwrap(),
                spaces.active_map[w].unwrap(),
            ],
        ) + (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
        )) * (gamma1.data[u * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + w])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + q])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + q])
            + (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            )) * (gamma1.data[t * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + q])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + w])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + v])
            + (gamma1.data[r * gamma1.n + q])
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
                ))
            + ((gamma1.data[r * gamma1.n + q]) * (gamma1.data[t * gamma1.n + v]))
                * (gamma1.data[u * gamma1.n + w])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[t * gamma1.n + w]))
                * (gamma1.data[u * gamma1.n + v])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[t * gamma1.n + q]))
                * (gamma1.data[u * gamma1.n + w])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[t * gamma1.n + w]))
                * (gamma1.data[u * gamma1.n + q])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[t * gamma1.n + q]))
                * (gamma1.data[u * gamma1.n + v])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[t * gamma1.n + v]))
                * (gamma1.data[u * gamma1.n + q]));
    out += (-delta(p, s))
        * ((delta(q, t))
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            ) + (gamma1.data[r * gamma1.n + v]) * (gamma1.data[u * gamma1.n + w])
                + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w]))
                    * (gamma1.data[u * gamma1.n + v])));
    out += (-delta(p, s))
        * ((delta(q, u))
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            ) + (gamma1.data[r * gamma1.n + w]) * (gamma1.data[t * gamma1.n + v])
                + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v]))
                    * (gamma1.data[t * gamma1.n + w])));
    out += (-delta(p, s))
        * ((delta(v, u))
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            ) + (gamma1.data[r * gamma1.n + q]) * (gamma1.data[t * gamma1.n + w])
                + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w]))
                    * (gamma1.data[t * gamma1.n + q])));
    out += (-delta(p, s)) * (((delta(v, u)) * (delta(q, t))) * (gamma1.data[r * gamma1.n + w]));
    out += ((-delta(p, s)) * (-delta(v, u)))
        * (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
        ) + (gamma1.data[r * gamma1.n + q]) * (gamma1.data[t * gamma1.n + w])
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[t * gamma1.n + q]));
    out += ((-delta(p, s)) * (-delta(v, u))) * ((delta(q, t)) * (gamma1.data[r * gamma1.n + w]));
    out += ((-delta(p, s)) * (-0.5 * gamma1.data[t * gamma1.n + v]))
        * (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
            &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
        ) + (gamma1.data[r * gamma1.n + q]) * (gamma1.data[u * gamma1.n + w])
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[u * gamma1.n + q]));
    out += ((-delta(p, s)) * (-0.5 * gamma1.data[t * gamma1.n + v]))
        * ((delta(q, u)) * (gamma1.data[r * gamma1.n + w]));
    out += ((-delta(p, s)) * (-0.5 * gamma1.data[u * gamma1.n + w]))
        * (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
        ) + (gamma1.data[r * gamma1.n + q]) * (gamma1.data[t * gamma1.n + v])
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[t * gamma1.n + q]));
    out += ((-delta(p, s)) * (-0.5 * gamma1.data[u * gamma1.n + w]))
        * ((delta(q, t)) * (gamma1.data[r * gamma1.n + v]));
    out += ((-delta(p, s)) * (0.5 * gamma1.data[u * gamma1.n + v]))
        * (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
        ) + (gamma1.data[r * gamma1.n + q]) * (gamma1.data[t * gamma1.n + w])
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[t * gamma1.n + q]));
    out += ((-delta(p, s)) * (0.5 * gamma1.data[u * gamma1.n + v]))
        * ((delta(q, t)) * (gamma1.data[r * gamma1.n + w]));
    out += ((-delta(p, s)) * (0.5 * gamma1.data[t * gamma1.n + w]))
        * (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
            &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
        ) + (gamma1.data[r * gamma1.n + q]) * (gamma1.data[u * gamma1.n + v])
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[u * gamma1.n + q]));
    out += ((-delta(p, s)) * (0.5 * gamma1.data[t * gamma1.n + w]))
        * ((delta(q, u)) * (gamma1.data[r * gamma1.n + v]));
    out += ((-delta(p, s))
        * (2.0 * gamma1.data[t * gamma1.n + v] * gamma1.data[u * gamma1.n + w]
            - 1.5 * gamma1.data[t * gamma1.n + w] * gamma1.data[u * gamma1.n + v]
            + lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            )))
        * (gamma1.data[r * gamma1.n + q]);
    out += (-0.5 * gamma1.data[r * gamma1.n + p])
        * (lambdas.lambda3.get(
            &[
                spaces.active_map[s].unwrap(),
                spaces.active_map[t].unwrap(),
                spaces.active_map[u].unwrap(),
            ],
            &[
                spaces.active_map[q].unwrap(),
                spaces.active_map[v].unwrap(),
                spaces.active_map[w].unwrap(),
            ],
        ) + (lambdas.lambda2.get(
            &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
        )) * (gamma1.data[u * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + w])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + q])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + q])
            + (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            )) * (gamma1.data[t * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + q])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + w])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + v])
            + (gamma1.data[s * gamma1.n + q])
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + q]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + v]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[s * gamma1.n + v]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[s * gamma1.n + w]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + w]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
                ))
            + ((gamma1.data[s * gamma1.n + q]) * (gamma1.data[t * gamma1.n + v]))
                * (gamma1.data[u * gamma1.n + w])
            + (((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + q])) * (gamma1.data[t * gamma1.n + w]))
                * (gamma1.data[u * gamma1.n + v])
            + (((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + v])) * (gamma1.data[t * gamma1.n + q]))
                * (gamma1.data[u * gamma1.n + w])
            + (((1.0 / 4.0) * (gamma1.data[s * gamma1.n + v])) * (gamma1.data[t * gamma1.n + w]))
                * (gamma1.data[u * gamma1.n + q])
            + (((1.0 / 4.0) * (gamma1.data[s * gamma1.n + w])) * (gamma1.data[t * gamma1.n + q]))
                * (gamma1.data[u * gamma1.n + v])
            + (((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + w])) * (gamma1.data[t * gamma1.n + v]))
                * (gamma1.data[u * gamma1.n + q]));
    out += (-0.5 * gamma1.data[r * gamma1.n + p])
        * ((delta(q, t))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            ) + (gamma1.data[s * gamma1.n + v]) * (gamma1.data[u * gamma1.n + w])
                + ((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + w]))
                    * (gamma1.data[u * gamma1.n + v])));
    out += (-0.5 * gamma1.data[r * gamma1.n + p])
        * ((delta(q, u))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            ) + (gamma1.data[s * gamma1.n + w]) * (gamma1.data[t * gamma1.n + v])
                + ((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + v]))
                    * (gamma1.data[t * gamma1.n + w])));
    out += (-0.5 * gamma1.data[r * gamma1.n + p])
        * ((delta(v, u))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            ) + (gamma1.data[s * gamma1.n + q]) * (gamma1.data[t * gamma1.n + w])
                + ((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + w]))
                    * (gamma1.data[t * gamma1.n + q])));
    out += (-0.5 * gamma1.data[r * gamma1.n + p])
        * (((delta(v, u)) * (delta(q, t))) * (gamma1.data[s * gamma1.n + w]));
    out += ((-0.5 * gamma1.data[r * gamma1.n + p]) * (-delta(v, u)))
        * (lambdas.lambda2.get(
            &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
        ) + (gamma1.data[s * gamma1.n + q]) * (gamma1.data[t * gamma1.n + w])
            + ((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + w])) * (gamma1.data[t * gamma1.n + q]));
    out += ((-0.5 * gamma1.data[r * gamma1.n + p]) * (-delta(v, u)))
        * ((delta(q, t)) * (gamma1.data[s * gamma1.n + w]));
    out += ((-0.5 * gamma1.data[r * gamma1.n + p]) * (-0.5 * gamma1.data[t * gamma1.n + v]))
        * (lambdas.lambda2.get(
            &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
            &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
        ) + (gamma1.data[s * gamma1.n + q]) * (gamma1.data[u * gamma1.n + w])
            + ((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + w])) * (gamma1.data[u * gamma1.n + q]));
    out += ((-0.5 * gamma1.data[r * gamma1.n + p]) * (-0.5 * gamma1.data[t * gamma1.n + v]))
        * ((delta(q, u)) * (gamma1.data[s * gamma1.n + w]));
    out += ((-0.5 * gamma1.data[r * gamma1.n + p]) * (-0.5 * gamma1.data[u * gamma1.n + w]))
        * (lambdas.lambda2.get(
            &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
        ) + (gamma1.data[s * gamma1.n + q]) * (gamma1.data[t * gamma1.n + v])
            + ((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + v])) * (gamma1.data[t * gamma1.n + q]));
    out += ((-0.5 * gamma1.data[r * gamma1.n + p]) * (-0.5 * gamma1.data[u * gamma1.n + w]))
        * ((delta(q, t)) * (gamma1.data[s * gamma1.n + v]));
    out += ((-0.5 * gamma1.data[r * gamma1.n + p]) * (0.5 * gamma1.data[u * gamma1.n + v]))
        * (lambdas.lambda2.get(
            &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
        ) + (gamma1.data[s * gamma1.n + q]) * (gamma1.data[t * gamma1.n + w])
            + ((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + w])) * (gamma1.data[t * gamma1.n + q]));
    out += ((-0.5 * gamma1.data[r * gamma1.n + p]) * (0.5 * gamma1.data[u * gamma1.n + v]))
        * ((delta(q, t)) * (gamma1.data[s * gamma1.n + w]));
    out += ((-0.5 * gamma1.data[r * gamma1.n + p]) * (0.5 * gamma1.data[t * gamma1.n + w]))
        * (lambdas.lambda2.get(
            &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
            &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
        ) + (gamma1.data[s * gamma1.n + q]) * (gamma1.data[u * gamma1.n + v])
            + ((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + v])) * (gamma1.data[u * gamma1.n + q]));
    out += ((-0.5 * gamma1.data[r * gamma1.n + p]) * (0.5 * gamma1.data[t * gamma1.n + w]))
        * ((delta(q, u)) * (gamma1.data[s * gamma1.n + v]));
    out += ((-0.5 * gamma1.data[r * gamma1.n + p])
        * (2.0 * gamma1.data[t * gamma1.n + v] * gamma1.data[u * gamma1.n + w]
            - 1.5 * gamma1.data[t * gamma1.n + w] * gamma1.data[u * gamma1.n + v]
            + lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            )))
        * (gamma1.data[s * gamma1.n + q]);
    out += (-0.5 * gamma1.data[s * gamma1.n + q])
        * (lambdas.lambda3.get(
            &[
                spaces.active_map[r].unwrap(),
                spaces.active_map[t].unwrap(),
                spaces.active_map[u].unwrap(),
            ],
            &[
                spaces.active_map[p].unwrap(),
                spaces.active_map[v].unwrap(),
                spaces.active_map[w].unwrap(),
            ],
        ) + (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
        )) * (gamma1.data[u * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + w])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + p])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + p])
            + (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            )) * (gamma1.data[t * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + p])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + w])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + p])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + v])
            + (gamma1.data[r * gamma1.n + p])
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
                ))
            + ((gamma1.data[r * gamma1.n + p]) * (gamma1.data[t * gamma1.n + v]))
                * (gamma1.data[u * gamma1.n + w])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + p])) * (gamma1.data[t * gamma1.n + w]))
                * (gamma1.data[u * gamma1.n + v])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[t * gamma1.n + p]))
                * (gamma1.data[u * gamma1.n + w])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[t * gamma1.n + w]))
                * (gamma1.data[u * gamma1.n + p])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[t * gamma1.n + p]))
                * (gamma1.data[u * gamma1.n + v])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[t * gamma1.n + v]))
                * (gamma1.data[u * gamma1.n + p]));
    out += (-0.5 * gamma1.data[s * gamma1.n + q])
        * ((delta(p, t))
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            ) + (gamma1.data[r * gamma1.n + v]) * (gamma1.data[u * gamma1.n + w])
                + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w]))
                    * (gamma1.data[u * gamma1.n + v])));
    out += (-0.5 * gamma1.data[s * gamma1.n + q])
        * ((delta(p, u))
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            ) + (gamma1.data[r * gamma1.n + w]) * (gamma1.data[t * gamma1.n + v])
                + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v]))
                    * (gamma1.data[t * gamma1.n + w])));
    out += (-0.5 * gamma1.data[s * gamma1.n + q])
        * ((delta(v, u))
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            ) + (gamma1.data[r * gamma1.n + p]) * (gamma1.data[t * gamma1.n + w])
                + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w]))
                    * (gamma1.data[t * gamma1.n + p])));
    out += (-0.5 * gamma1.data[s * gamma1.n + q])
        * (((delta(v, u)) * (delta(p, t))) * (gamma1.data[r * gamma1.n + w]));
    out += ((-0.5 * gamma1.data[s * gamma1.n + q]) * (-delta(v, u)))
        * (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
        ) + (gamma1.data[r * gamma1.n + p]) * (gamma1.data[t * gamma1.n + w])
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[t * gamma1.n + p]));
    out += ((-0.5 * gamma1.data[s * gamma1.n + q]) * (-delta(v, u)))
        * ((delta(p, t)) * (gamma1.data[r * gamma1.n + w]));
    out += ((-0.5 * gamma1.data[s * gamma1.n + q]) * (-0.5 * gamma1.data[t * gamma1.n + v]))
        * (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
        ) + (gamma1.data[r * gamma1.n + p]) * (gamma1.data[u * gamma1.n + w])
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[u * gamma1.n + p]));
    out += ((-0.5 * gamma1.data[s * gamma1.n + q]) * (-0.5 * gamma1.data[t * gamma1.n + v]))
        * ((delta(p, u)) * (gamma1.data[r * gamma1.n + w]));
    out += ((-0.5 * gamma1.data[s * gamma1.n + q]) * (-0.5 * gamma1.data[u * gamma1.n + w]))
        * (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
        ) + (gamma1.data[r * gamma1.n + p]) * (gamma1.data[t * gamma1.n + v])
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[t * gamma1.n + p]));
    out += ((-0.5 * gamma1.data[s * gamma1.n + q]) * (-0.5 * gamma1.data[u * gamma1.n + w]))
        * ((delta(p, t)) * (gamma1.data[r * gamma1.n + v]));
    out += ((-0.5 * gamma1.data[s * gamma1.n + q]) * (0.5 * gamma1.data[u * gamma1.n + v]))
        * (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
        ) + (gamma1.data[r * gamma1.n + p]) * (gamma1.data[t * gamma1.n + w])
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[t * gamma1.n + p]));
    out += ((-0.5 * gamma1.data[s * gamma1.n + q]) * (0.5 * gamma1.data[u * gamma1.n + v]))
        * ((delta(p, t)) * (gamma1.data[r * gamma1.n + w]));
    out += ((-0.5 * gamma1.data[s * gamma1.n + q]) * (0.5 * gamma1.data[t * gamma1.n + w]))
        * (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
        ) + (gamma1.data[r * gamma1.n + p]) * (gamma1.data[u * gamma1.n + v])
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[u * gamma1.n + p]));
    out += ((-0.5 * gamma1.data[s * gamma1.n + q]) * (0.5 * gamma1.data[t * gamma1.n + w]))
        * ((delta(p, u)) * (gamma1.data[r * gamma1.n + v]));
    out += ((-0.5 * gamma1.data[s * gamma1.n + q])
        * (2.0 * gamma1.data[t * gamma1.n + v] * gamma1.data[u * gamma1.n + w]
            - 1.5 * gamma1.data[t * gamma1.n + w] * gamma1.data[u * gamma1.n + v]
            + lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            )))
        * (gamma1.data[r * gamma1.n + p]);
    out += (0.5 * gamma1.data[s * gamma1.n + p])
        * (lambdas.lambda3.get(
            &[
                spaces.active_map[r].unwrap(),
                spaces.active_map[t].unwrap(),
                spaces.active_map[u].unwrap(),
            ],
            &[
                spaces.active_map[q].unwrap(),
                spaces.active_map[v].unwrap(),
                spaces.active_map[w].unwrap(),
            ],
        ) + (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
        )) * (gamma1.data[u * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + w])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + q])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + q])
            + (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            )) * (gamma1.data[t * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + q])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + w])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + q])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + v])
            + (gamma1.data[r * gamma1.n + q])
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[q].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[q].unwrap()],
                ))
            + ((gamma1.data[r * gamma1.n + q]) * (gamma1.data[t * gamma1.n + v]))
                * (gamma1.data[u * gamma1.n + w])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + q])) * (gamma1.data[t * gamma1.n + w]))
                * (gamma1.data[u * gamma1.n + v])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[t * gamma1.n + q]))
                * (gamma1.data[u * gamma1.n + w])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[t * gamma1.n + w]))
                * (gamma1.data[u * gamma1.n + q])
            + (((1.0 / 4.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[t * gamma1.n + q]))
                * (gamma1.data[u * gamma1.n + v])
            + (((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[t * gamma1.n + v]))
                * (gamma1.data[u * gamma1.n + q]));
    out += (0.5 * gamma1.data[s * gamma1.n + p])
        * ((delta(q, t))
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            ) + (gamma1.data[r * gamma1.n + v]) * (gamma1.data[u * gamma1.n + w])
                + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w]))
                    * (gamma1.data[u * gamma1.n + v])));
    out += (0.5 * gamma1.data[s * gamma1.n + p])
        * ((delta(q, u))
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            ) + (gamma1.data[r * gamma1.n + w]) * (gamma1.data[t * gamma1.n + v])
                + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v]))
                    * (gamma1.data[t * gamma1.n + w])));
    out += (0.5 * gamma1.data[s * gamma1.n + p])
        * ((delta(v, u))
            * (lambdas.lambda2.get(
                &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
            ) + (gamma1.data[r * gamma1.n + q]) * (gamma1.data[t * gamma1.n + w])
                + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w]))
                    * (gamma1.data[t * gamma1.n + q])));
    out += (0.5 * gamma1.data[s * gamma1.n + p])
        * (((delta(v, u)) * (delta(q, t))) * (gamma1.data[r * gamma1.n + w]));
    out += ((0.5 * gamma1.data[s * gamma1.n + p]) * (-delta(v, u)))
        * (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
        ) + (gamma1.data[r * gamma1.n + q]) * (gamma1.data[t * gamma1.n + w])
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[t * gamma1.n + q]));
    out += ((0.5 * gamma1.data[s * gamma1.n + p]) * (-delta(v, u)))
        * ((delta(q, t)) * (gamma1.data[r * gamma1.n + w]));
    out += ((0.5 * gamma1.data[s * gamma1.n + p]) * (-0.5 * gamma1.data[t * gamma1.n + v]))
        * (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
            &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
        ) + (gamma1.data[r * gamma1.n + q]) * (gamma1.data[u * gamma1.n + w])
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[u * gamma1.n + q]));
    out += ((0.5 * gamma1.data[s * gamma1.n + p]) * (-0.5 * gamma1.data[t * gamma1.n + v]))
        * ((delta(q, u)) * (gamma1.data[r * gamma1.n + w]));
    out += ((0.5 * gamma1.data[s * gamma1.n + p]) * (-0.5 * gamma1.data[u * gamma1.n + w]))
        * (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
        ) + (gamma1.data[r * gamma1.n + q]) * (gamma1.data[t * gamma1.n + v])
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[t * gamma1.n + q]));
    out += ((0.5 * gamma1.data[s * gamma1.n + p]) * (-0.5 * gamma1.data[u * gamma1.n + w]))
        * ((delta(q, t)) * (gamma1.data[r * gamma1.n + v]));
    out += ((0.5 * gamma1.data[s * gamma1.n + p]) * (0.5 * gamma1.data[u * gamma1.n + v]))
        * (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[q].unwrap(), spaces.active_map[w].unwrap()],
        ) + (gamma1.data[r * gamma1.n + q]) * (gamma1.data[t * gamma1.n + w])
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + w])) * (gamma1.data[t * gamma1.n + q]));
    out += ((0.5 * gamma1.data[s * gamma1.n + p]) * (0.5 * gamma1.data[u * gamma1.n + v]))
        * ((delta(q, t)) * (gamma1.data[r * gamma1.n + w]));
    out += ((0.5 * gamma1.data[s * gamma1.n + p]) * (0.5 * gamma1.data[t * gamma1.n + w]))
        * (lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[u].unwrap()],
            &[spaces.active_map[q].unwrap(), spaces.active_map[v].unwrap()],
        ) + (gamma1.data[r * gamma1.n + q]) * (gamma1.data[u * gamma1.n + v])
            + ((-1.0 / 2.0) * (gamma1.data[r * gamma1.n + v])) * (gamma1.data[u * gamma1.n + q]));
    out += ((0.5 * gamma1.data[s * gamma1.n + p]) * (0.5 * gamma1.data[t * gamma1.n + w]))
        * ((delta(q, u)) * (gamma1.data[r * gamma1.n + v]));
    out += ((0.5 * gamma1.data[s * gamma1.n + p])
        * (2.0 * gamma1.data[t * gamma1.n + v] * gamma1.data[u * gamma1.n + w]
            - 1.5 * gamma1.data[t * gamma1.n + w] * gamma1.data[u * gamma1.n + v]
            + lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            )))
        * (gamma1.data[r * gamma1.n + q]);
    out += (0.5 * gamma1.data[r * gamma1.n + q])
        * (lambdas.lambda3.get(
            &[
                spaces.active_map[s].unwrap(),
                spaces.active_map[t].unwrap(),
                spaces.active_map[u].unwrap(),
            ],
            &[
                spaces.active_map[p].unwrap(),
                spaces.active_map[v].unwrap(),
                spaces.active_map[w].unwrap(),
            ],
        ) + (lambdas.lambda2.get(
            &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
        )) * (gamma1.data[u * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + w])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + p])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[u * gamma1.n + p])
            + (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            )) * (gamma1.data[t * gamma1.n + v])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + w])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + p])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + w])
            + ((1.0 / 4.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + p])
            + ((-1.0 / 2.0)
                * (lambdas.lambda2.get(
                    &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
                )))
                * (gamma1.data[t * gamma1.n + v])
            + (gamma1.data[s * gamma1.n + p])
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + p]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + v]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[s * gamma1.n + v]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
                ))
            + ((1.0 / 4.0) * (gamma1.data[s * gamma1.n + w]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
                ))
            + ((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + w]))
                * (lambdas.lambda2.get(
                    &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                    &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
                ))
            + ((gamma1.data[s * gamma1.n + p]) * (gamma1.data[t * gamma1.n + v]))
                * (gamma1.data[u * gamma1.n + w])
            + (((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + p])) * (gamma1.data[t * gamma1.n + w]))
                * (gamma1.data[u * gamma1.n + v])
            + (((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + v])) * (gamma1.data[t * gamma1.n + p]))
                * (gamma1.data[u * gamma1.n + w])
            + (((1.0 / 4.0) * (gamma1.data[s * gamma1.n + v])) * (gamma1.data[t * gamma1.n + w]))
                * (gamma1.data[u * gamma1.n + p])
            + (((1.0 / 4.0) * (gamma1.data[s * gamma1.n + w])) * (gamma1.data[t * gamma1.n + p]))
                * (gamma1.data[u * gamma1.n + v])
            + (((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + w])) * (gamma1.data[t * gamma1.n + v]))
                * (gamma1.data[u * gamma1.n + p]));
    out += (0.5 * gamma1.data[r * gamma1.n + q])
        * ((delta(p, t))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            ) + (gamma1.data[s * gamma1.n + v]) * (gamma1.data[u * gamma1.n + w])
                + ((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + w]))
                    * (gamma1.data[u * gamma1.n + v])));
    out += (0.5 * gamma1.data[r * gamma1.n + q])
        * ((delta(p, u))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            ) + (gamma1.data[s * gamma1.n + w]) * (gamma1.data[t * gamma1.n + v])
                + ((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + v]))
                    * (gamma1.data[t * gamma1.n + w])));
    out += (0.5 * gamma1.data[r * gamma1.n + q])
        * ((delta(v, u))
            * (lambdas.lambda2.get(
                &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            ) + (gamma1.data[s * gamma1.n + p]) * (gamma1.data[t * gamma1.n + w])
                + ((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + w]))
                    * (gamma1.data[t * gamma1.n + p])));
    out += (0.5 * gamma1.data[r * gamma1.n + q])
        * (((delta(v, u)) * (delta(p, t))) * (gamma1.data[s * gamma1.n + w]));
    out += ((0.5 * gamma1.data[r * gamma1.n + q]) * (-delta(v, u)))
        * (lambdas.lambda2.get(
            &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
        ) + (gamma1.data[s * gamma1.n + p]) * (gamma1.data[t * gamma1.n + w])
            + ((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + w])) * (gamma1.data[t * gamma1.n + p]));
    out += ((0.5 * gamma1.data[r * gamma1.n + q]) * (-delta(v, u)))
        * ((delta(p, t)) * (gamma1.data[s * gamma1.n + w]));
    out += ((0.5 * gamma1.data[r * gamma1.n + q]) * (-0.5 * gamma1.data[t * gamma1.n + v]))
        * (lambdas.lambda2.get(
            &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
        ) + (gamma1.data[s * gamma1.n + p]) * (gamma1.data[u * gamma1.n + w])
            + ((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + w])) * (gamma1.data[u * gamma1.n + p]));
    out += ((0.5 * gamma1.data[r * gamma1.n + q]) * (-0.5 * gamma1.data[t * gamma1.n + v]))
        * ((delta(p, u)) * (gamma1.data[s * gamma1.n + w]));
    out += ((0.5 * gamma1.data[r * gamma1.n + q]) * (-0.5 * gamma1.data[u * gamma1.n + w]))
        * (lambdas.lambda2.get(
            &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
        ) + (gamma1.data[s * gamma1.n + p]) * (gamma1.data[t * gamma1.n + v])
            + ((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + v])) * (gamma1.data[t * gamma1.n + p]));
    out += ((0.5 * gamma1.data[r * gamma1.n + q]) * (-0.5 * gamma1.data[u * gamma1.n + w]))
        * ((delta(p, t)) * (gamma1.data[s * gamma1.n + v]));
    out += ((0.5 * gamma1.data[r * gamma1.n + q]) * (0.5 * gamma1.data[u * gamma1.n + v]))
        * (lambdas.lambda2.get(
            &[spaces.active_map[s].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
        ) + (gamma1.data[s * gamma1.n + p]) * (gamma1.data[t * gamma1.n + w])
            + ((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + w])) * (gamma1.data[t * gamma1.n + p]));
    out += ((0.5 * gamma1.data[r * gamma1.n + q]) * (0.5 * gamma1.data[u * gamma1.n + v]))
        * ((delta(p, t)) * (gamma1.data[s * gamma1.n + w]));
    out += ((0.5 * gamma1.data[r * gamma1.n + q]) * (0.5 * gamma1.data[t * gamma1.n + w]))
        * (lambdas.lambda2.get(
            &[spaces.active_map[s].unwrap(), spaces.active_map[u].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
        ) + (gamma1.data[s * gamma1.n + p]) * (gamma1.data[u * gamma1.n + v])
            + ((-1.0 / 2.0) * (gamma1.data[s * gamma1.n + v])) * (gamma1.data[u * gamma1.n + p]));
    out += ((0.5 * gamma1.data[r * gamma1.n + q]) * (0.5 * gamma1.data[t * gamma1.n + w]))
        * ((delta(p, u)) * (gamma1.data[s * gamma1.n + v]));
    out += ((0.5 * gamma1.data[r * gamma1.n + q])
        * (2.0 * gamma1.data[t * gamma1.n + v] * gamma1.data[u * gamma1.n + w]
            - 1.5 * gamma1.data[t * gamma1.n + w] * gamma1.data[u * gamma1.n + v]
            + lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            )))
        * (gamma1.data[s * gamma1.n + p]);
    out += (2.0 * gamma1.data[r * gamma1.n + p] * gamma1.data[s * gamma1.n + q]
        - 1.5 * gamma1.data[r * gamma1.n + q] * gamma1.data[s * gamma1.n + p]
        + lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
        ))
        * (lambdas.lambda2.get(
            &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
            &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
        ) + (gamma1.data[t * gamma1.n + v]) * (gamma1.data[u * gamma1.n + w])
            + ((-1.0 / 2.0) * (gamma1.data[t * gamma1.n + w])) * (gamma1.data[u * gamma1.n + v]));
    out += (2.0 * gamma1.data[r * gamma1.n + p] * gamma1.data[s * gamma1.n + q]
        - 1.5 * gamma1.data[r * gamma1.n + q] * gamma1.data[s * gamma1.n + p]
        + lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
        ))
        * ((delta(v, u)) * (gamma1.data[t * gamma1.n + w]));
    out += ((2.0 * gamma1.data[r * gamma1.n + p] * gamma1.data[s * gamma1.n + q]
        - 1.5 * gamma1.data[r * gamma1.n + q] * gamma1.data[s * gamma1.n + p]
        + lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
        ))
        * (-delta(v, u)))
        * (gamma1.data[t * gamma1.n + w]);
    out += ((2.0 * gamma1.data[r * gamma1.n + p] * gamma1.data[s * gamma1.n + q]
        - 1.5 * gamma1.data[r * gamma1.n + q] * gamma1.data[s * gamma1.n + p]
        + lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
        ))
        * (-0.5 * gamma1.data[t * gamma1.n + v]))
        * (gamma1.data[u * gamma1.n + w]);
    out += ((2.0 * gamma1.data[r * gamma1.n + p] * gamma1.data[s * gamma1.n + q]
        - 1.5 * gamma1.data[r * gamma1.n + q] * gamma1.data[s * gamma1.n + p]
        + lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
        ))
        * (-0.5 * gamma1.data[u * gamma1.n + w]))
        * (gamma1.data[t * gamma1.n + v]);
    out += ((2.0 * gamma1.data[r * gamma1.n + p] * gamma1.data[s * gamma1.n + q]
        - 1.5 * gamma1.data[r * gamma1.n + q] * gamma1.data[s * gamma1.n + p]
        + lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
        ))
        * (0.5 * gamma1.data[u * gamma1.n + v]))
        * (gamma1.data[t * gamma1.n + w]);
    out += ((2.0 * gamma1.data[r * gamma1.n + p] * gamma1.data[s * gamma1.n + q]
        - 1.5 * gamma1.data[r * gamma1.n + q] * gamma1.data[s * gamma1.n + p]
        + lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
        ))
        * (0.5 * gamma1.data[t * gamma1.n + w]))
        * (gamma1.data[u * gamma1.n + v]);
    out += (2.0 * gamma1.data[r * gamma1.n + p] * gamma1.data[s * gamma1.n + q]
        - 1.5 * gamma1.data[r * gamma1.n + q] * gamma1.data[s * gamma1.n + p]
        + lambdas.lambda2.get(
            &[spaces.active_map[r].unwrap(), spaces.active_map[s].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[q].unwrap()],
        ))
        * (2.0 * gamma1.data[t * gamma1.n + v] * gamma1.data[u * gamma1.n + w]
            - 1.5 * gamma1.data[t * gamma1.n + w] * gamma1.data[u * gamma1.n + v]
            + lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            ));

    out
}

/// Evaluate the C13 A -> V / AA -> AV mixed overlap block.
/// # Arguments:
/// - `left`: Left A -> V excitation.
/// - `right`: Right AA -> AV excitation.
/// - `spaces`: NOCC orbital spaces.
/// - `lambdas`: Active-space spin-free cumulants.
/// # Returns:
/// - `f64`: Raw overlap element.
fn overlap_a_to_v_aa_to_av(
    left: Excitation,
    right: Excitation,
    spaces: &Spaces,
    lambdas: &Cumulants<f64>,
) -> f64 {
    let (a, u) = single(left);
    let (x, b, v, w) = double(right);
    delta(b, a)
        * lambdas.lambda2.get(
            &[spaces.active_map[u].unwrap(), spaces.active_map[x].unwrap()],
            &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
        )
}

/// Evaluate the C14 C -> A / CA -> AA mixed overlap block.
/// # Arguments:
/// - `left`: Left C -> A excitation.
/// - `right`: Right CA -> AA excitation.
/// - `spaces`: NOCC orbital spaces.
/// - `lambdas`: Active-space spin-free cumulants.
/// # Returns:
/// - `f64`: Raw overlap element.
fn overlap_c_to_a_ca_to_aa(
    left: Excitation,
    right: Excitation,
    spaces: &Spaces,
    lambdas: &Cumulants<f64>,
) -> f64 {
    let (u, i) = single(left);
    let (w, x, j, v) = double(right);
    -delta(i, j)
        * lambdas.lambda2.get(
            &[spaces.active_map[w].unwrap(), spaces.active_map[x].unwrap()],
            &[spaces.active_map[u].unwrap(), spaces.active_map[v].unwrap()],
        )
}

/// Evaluate the C15 A -> A / AA -> AA mixed overlap block.
/// # Arguments:
/// - `left`: Left A -> A excitation.
/// - `right`: Right AA -> AA excitation.
/// - `spaces`: NOCC orbital spaces.
/// - `gamma1`: Full-space spin-free one-body RDM.
/// - `lambdas`: Active-space spin-free cumulants.
/// # Returns:
/// - `f64`: Raw overlap element.
fn overlap_a_to_a_aa_to_aa(
    left: Excitation,
    right: Excitation,
    spaces: &Spaces,
    gamma1: &RDM1<f64>,
    lambdas: &Cumulants<f64>,
) -> f64 {
    let (p, q) = single(left);
    let (t, u, v, w) = double(right);

    // The left excitation is daggered in S_mu_nu.
    // This evaluates <Phi | tau^q_p tau^{tu}_{vw} | Phi>.
    let mut out = 0.0;

    out += lambdas.lambda3.get(
        &[
            spaces.active_map[q].unwrap(),
            spaces.active_map[t].unwrap(),
            spaces.active_map[u].unwrap(),
        ],
        &[
            spaces.active_map[p].unwrap(),
            spaces.active_map[v].unwrap(),
            spaces.active_map[w].unwrap(),
        ],
    ) + (lambdas.lambda2.get(
        &[spaces.active_map[q].unwrap(), spaces.active_map[t].unwrap()],
        &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
    )) * (gamma1.data[u * gamma1.n + w])
        + ((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[q].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + v])
        + ((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[q].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + w])
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[q].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + p])
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[q].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + v])
        + ((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[q].unwrap(), spaces.active_map[t].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[u * gamma1.n + p])
        + (lambdas.lambda2.get(
            &[spaces.active_map[q].unwrap(), spaces.active_map[u].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
        )) * (gamma1.data[t * gamma1.n + v])
        + ((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[q].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + w])
        + ((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[q].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + p])
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[q].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + w])
        + ((1.0 / 4.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[q].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + p])
        + ((-1.0 / 2.0)
            * (lambdas.lambda2.get(
                &[spaces.active_map[q].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
            )))
            * (gamma1.data[t * gamma1.n + v])
        + (gamma1.data[q * gamma1.n + p])
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            ))
        + ((-1.0 / 2.0) * (gamma1.data[q * gamma1.n + p]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
            ))
        + ((-1.0 / 2.0) * (gamma1.data[q * gamma1.n + v]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
            ))
        + ((1.0 / 4.0) * (gamma1.data[q * gamma1.n + v]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[p].unwrap()],
            ))
        + ((1.0 / 4.0) * (gamma1.data[q * gamma1.n + w]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
            ))
        + ((-1.0 / 2.0) * (gamma1.data[q * gamma1.n + w]))
            * (lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[p].unwrap()],
            ))
        + ((gamma1.data[q * gamma1.n + p]) * (gamma1.data[t * gamma1.n + v]))
            * (gamma1.data[u * gamma1.n + w])
        + (((-1.0 / 2.0) * (gamma1.data[q * gamma1.n + p])) * (gamma1.data[t * gamma1.n + w]))
            * (gamma1.data[u * gamma1.n + v])
        + (((-1.0 / 2.0) * (gamma1.data[q * gamma1.n + v])) * (gamma1.data[t * gamma1.n + p]))
            * (gamma1.data[u * gamma1.n + w])
        + (((1.0 / 4.0) * (gamma1.data[q * gamma1.n + v])) * (gamma1.data[t * gamma1.n + w]))
            * (gamma1.data[u * gamma1.n + p])
        + (((1.0 / 4.0) * (gamma1.data[q * gamma1.n + w])) * (gamma1.data[t * gamma1.n + p]))
            * (gamma1.data[u * gamma1.n + v])
        + (((-1.0 / 2.0) * (gamma1.data[q * gamma1.n + w])) * (gamma1.data[t * gamma1.n + v]))
            * (gamma1.data[u * gamma1.n + p]);
    out += (delta(p, t))
        * (lambdas.lambda2.get(
            &[spaces.active_map[q].unwrap(), spaces.active_map[u].unwrap()],
            &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
        ) + (gamma1.data[q * gamma1.n + v]) * (gamma1.data[u * gamma1.n + w])
            + ((-1.0 / 2.0) * (gamma1.data[q * gamma1.n + w])) * (gamma1.data[u * gamma1.n + v]));
    out += (delta(p, u))
        * (lambdas.lambda2.get(
            &[spaces.active_map[q].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[w].unwrap(), spaces.active_map[v].unwrap()],
        ) + (gamma1.data[q * gamma1.n + w]) * (gamma1.data[t * gamma1.n + v])
            + ((-1.0 / 2.0) * (gamma1.data[q * gamma1.n + v])) * (gamma1.data[t * gamma1.n + w]));
    out += (delta(v, u))
        * (lambdas.lambda2.get(
            &[spaces.active_map[q].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
        ) + (gamma1.data[q * gamma1.n + p]) * (gamma1.data[t * gamma1.n + w])
            + ((-1.0 / 2.0) * (gamma1.data[q * gamma1.n + w])) * (gamma1.data[t * gamma1.n + p]));
    out += ((delta(v, u)) * (delta(p, t))) * (gamma1.data[q * gamma1.n + w]);
    out += (-delta(v, u))
        * (lambdas.lambda2.get(
            &[spaces.active_map[q].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
        ) + (gamma1.data[q * gamma1.n + p]) * (gamma1.data[t * gamma1.n + w])
            + ((-1.0 / 2.0) * (gamma1.data[q * gamma1.n + w])) * (gamma1.data[t * gamma1.n + p]));
    out += (-delta(v, u)) * ((delta(p, t)) * (gamma1.data[q * gamma1.n + w]));
    out += (-0.5 * gamma1.data[t * gamma1.n + v])
        * (lambdas.lambda2.get(
            &[spaces.active_map[q].unwrap(), spaces.active_map[u].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
        ) + (gamma1.data[q * gamma1.n + p]) * (gamma1.data[u * gamma1.n + w])
            + ((-1.0 / 2.0) * (gamma1.data[q * gamma1.n + w])) * (gamma1.data[u * gamma1.n + p]));
    out +=
        (-0.5 * gamma1.data[t * gamma1.n + v]) * ((delta(p, u)) * (gamma1.data[q * gamma1.n + w]));
    out += (-0.5 * gamma1.data[u * gamma1.n + w])
        * (lambdas.lambda2.get(
            &[spaces.active_map[q].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
        ) + (gamma1.data[q * gamma1.n + p]) * (gamma1.data[t * gamma1.n + v])
            + ((-1.0 / 2.0) * (gamma1.data[q * gamma1.n + v])) * (gamma1.data[t * gamma1.n + p]));
    out +=
        (-0.5 * gamma1.data[u * gamma1.n + w]) * ((delta(p, t)) * (gamma1.data[q * gamma1.n + v]));
    out += (0.5 * gamma1.data[u * gamma1.n + v])
        * (lambdas.lambda2.get(
            &[spaces.active_map[q].unwrap(), spaces.active_map[t].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[w].unwrap()],
        ) + (gamma1.data[q * gamma1.n + p]) * (gamma1.data[t * gamma1.n + w])
            + ((-1.0 / 2.0) * (gamma1.data[q * gamma1.n + w])) * (gamma1.data[t * gamma1.n + p]));
    out +=
        (0.5 * gamma1.data[u * gamma1.n + v]) * ((delta(p, t)) * (gamma1.data[q * gamma1.n + w]));
    out += (0.5 * gamma1.data[t * gamma1.n + w])
        * (lambdas.lambda2.get(
            &[spaces.active_map[q].unwrap(), spaces.active_map[u].unwrap()],
            &[spaces.active_map[p].unwrap(), spaces.active_map[v].unwrap()],
        ) + (gamma1.data[q * gamma1.n + p]) * (gamma1.data[u * gamma1.n + v])
            + ((-1.0 / 2.0) * (gamma1.data[q * gamma1.n + v])) * (gamma1.data[u * gamma1.n + p]));
    out +=
        (0.5 * gamma1.data[t * gamma1.n + w]) * ((delta(p, u)) * (gamma1.data[q * gamma1.n + v]));
    out += (2.0 * gamma1.data[t * gamma1.n + v] * gamma1.data[u * gamma1.n + w]
        - 1.5 * gamma1.data[t * gamma1.n + w] * gamma1.data[u * gamma1.n + v]
        + lambdas.lambda2.get(
            &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
            &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
        ))
        * (gamma1.data[q * gamma1.n + p]);
    out += (-gamma1.data[q * gamma1.n + p])
        * (lambdas.lambda2.get(
            &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
            &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
        ) + (gamma1.data[t * gamma1.n + v]) * (gamma1.data[u * gamma1.n + w])
            + ((-1.0 / 2.0) * (gamma1.data[t * gamma1.n + w])) * (gamma1.data[u * gamma1.n + v]));
    out += (-gamma1.data[q * gamma1.n + p]) * ((delta(v, u)) * (gamma1.data[t * gamma1.n + w]));
    out += ((-gamma1.data[q * gamma1.n + p]) * (-delta(v, u))) * (gamma1.data[t * gamma1.n + w]);
    out += ((-gamma1.data[q * gamma1.n + p]) * (-0.5 * gamma1.data[t * gamma1.n + v]))
        * (gamma1.data[u * gamma1.n + w]);
    out += ((-gamma1.data[q * gamma1.n + p]) * (-0.5 * gamma1.data[u * gamma1.n + w]))
        * (gamma1.data[t * gamma1.n + v]);
    out += ((-gamma1.data[q * gamma1.n + p]) * (0.5 * gamma1.data[u * gamma1.n + v]))
        * (gamma1.data[t * gamma1.n + w]);
    out += ((-gamma1.data[q * gamma1.n + p]) * (0.5 * gamma1.data[t * gamma1.n + w]))
        * (gamma1.data[u * gamma1.n + v]);
    out += (-gamma1.data[q * gamma1.n + p])
        * (2.0 * gamma1.data[t * gamma1.n + v] * gamma1.data[u * gamma1.n + w]
            - 1.5 * gamma1.data[t * gamma1.n + w] * gamma1.data[u * gamma1.n + v]
            + lambdas.lambda2.get(
                &[spaces.active_map[t].unwrap(), spaces.active_map[u].unwrap()],
                &[spaces.active_map[v].unwrap(), spaces.active_map[w].unwrap()],
            ));

    out
}

/// Evaluate the C16 CA -> AV / CA -> VA mixed overlap block.
/// # Arguments:
/// - `left`: Left CA -> AV excitation.
/// - `right`: Right CA -> VA excitation.
/// - `spaces`: NOCC orbital spaces.
/// - `gamma1`: Full-space spin-free one-body RDM.
/// - `lambdas`: Active-space spin-free cumulants.
/// # Returns:
/// - `f64`: Raw overlap element.
fn overlap_ca_to_av_ca_to_va(
    left: Excitation,
    right: Excitation,
    spaces: &Spaces,
    gamma1: &RDM1<f64>,
    lambdas: &Cumulants<f64>,
) -> f64 {
    let (w, a, i, u) = double(left);
    let (b, y, j, x) = double(right);
    -delta(i, j)
        * delta(b, a)
        * (0.5 * gamma1.data[u * gamma1.n + x] * theta(gamma1, y, w)
            + lambdas.lambda2.get(
                &[spaces.active_map[u].unwrap(), spaces.active_map[y].unwrap()],
                &[spaces.active_map[w].unwrap(), spaces.active_map[x].unwrap()],
            ))
}

/// Build Hamiltonian coupling weights used for the weighted FOIS metric.
/// # Arguments:
/// - `ao`: Integrals transformed to the NOCI natural-orbital basis.
/// - `gamma1`: Full-space spin-free one-body RDM.
/// - `excitations`: Raw spin-free excitation list.
/// # Returns:
/// - `Array1<f64>`: One Hamiltonian weight per excitation.
fn hamiltonian_weights(
    ao: &AoData,
    gamma1: &RDM1<f64>,
    excitations: &[Excitation],
) -> Array1<f64> {
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

    let (fa, _fb) = fock(&ao.h, &ao.eri_coul, &da, &db);
    let mut h = Array1::zeros(excitations.len());

    for (i, &ex) in excitations.iter().enumerate() {
        h[i] = match ex {
            Excitation::Single { p, q } => fa[(q, p)],
            Excitation::Double { p, q, r, s } => 0.5 * ao.eri_coul[(r, s, p, q)],
        };
    }

    h
}

/// Extract indices from a single excitation.
/// # Arguments:
/// - `ex`: Spin-free excitation.
/// # Returns:
/// - `(usize, usize)`: Creator and annihilator spatial orbital indices.
fn single(ex: Excitation) -> (usize, usize) {
    match ex {
        Excitation::Single { p, q } => (p, q),
        _ => panic!("expected single excitation"),
    }
}

/// Extract indices from a double excitation.
/// # Arguments:
/// - `ex`: Spin-free excitation.
/// # Returns:
/// - `(usize, usize, usize, usize)`: Two creator and two annihilator spatial orbital indices.
fn double(ex: Excitation) -> (usize, usize, usize, usize) {
    match ex {
        Excitation::Double { p, q, r, s } => (p, q, r, s),
        _ => panic!("expected double excitation"),
    }
}

/// Return one one-hole RDM element.
/// # Arguments:
/// - `gamma1`: Full-space spin-free one-body RDM.
/// - `p`: Creator spatial orbital index.
/// - `q`: Annihilator spatial orbital index.
/// # Returns:
/// - `f64`: `2 delta[p, q] - Gamma[p, q]`.
fn theta(
    gamma1: &RDM1<f64>,
    p: usize,
    q: usize,
) -> f64 {
    (if p == q { 2.0 } else { 0.0 }) - gamma1.data[p * gamma1.n + q]
}

/// Return a Kronecker delta.
/// # Arguments:
/// - `p`: First index.
/// - `q`: Second index.
/// # Returns:
/// - `f64`: `1.0` if the indices are equal, otherwise `0.0`.
fn delta(
    p: usize,
    q: usize,
) -> f64 {
    if p == q { 1.0 } else { 0.0 }
}

/// Format an excitation for diagnostics.
/// # Arguments:
/// - `ex`: Spin-free excitation.
/// # Returns:
/// - `String`: Human-readable excitation label.
fn excitation_label(ex: Excitation) -> String {
    match ex {
        Excitation::Single { p, q } => format!("single p: {}, q: {}", p, q),
        Excitation::Double { p, q, r, s } => {
            format!("double p: {}, q: {}, r: {}, s: {}", p, q, r, s)
        }
    }
}
