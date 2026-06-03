// deterministic/noccmc/space.rs

use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, UPLO};
use std::collections::BTreeMap;

use super::overlap;
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
    /// C -> V single excitation.
    CToV,
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

/// Classify a supported spin-free single excitation.
/// # Arguments:
/// - `spaces`: NOCC orbital spaces.
/// - `p`: Creator-side orbital index.
/// - `q`: Annihilator-side orbital index.
/// # Returns:
/// - `Option<ExcitationClass>`: Appendix-C excitation class, or `None` if unsupported.
fn single_excitation_class(
    spaces: &Spaces,
    p: usize,
    q: usize,
) -> Option<ExcitationClass> {
    match (spaces.class_of[q], spaces.class_of[p]) {
        (OrbitalClass::Core, OrbitalClass::Active) => Some(ExcitationClass::CToA),
        (OrbitalClass::Active, OrbitalClass::Active) => Some(ExcitationClass::AToA),
        (OrbitalClass::Active, OrbitalClass::Virtual) => Some(ExcitationClass::AToV),
        (OrbitalClass::Core, OrbitalClass::Virtual) => Some(ExcitationClass::CToV),
        _ => None,
    }
}

/// Classify a supported spin-free double excitation.
/// # Arguments:
/// - `spaces`: NOCC orbital spaces.
/// - `p`: First creator-side orbital index.
/// - `q`: Second creator-side orbital index.
/// - `r`: First annihilator-side orbital index.
/// - `s`: Second annihilator-side orbital index.
/// # Returns:
/// - `Option<ExcitationClass>`: Appendix-C excitation class, or `None` if unsupported.
fn double_excitation_class(
    spaces: &Spaces,
    p: usize,
    q: usize,
    r: usize,
    s: usize,
) -> Option<ExcitationClass> {
    match (
        spaces.class_of[r],
        spaces.class_of[s],
        spaces.class_of[p],
        spaces.class_of[q],
    ) {
        (OrbitalClass::Core, OrbitalClass::Core, OrbitalClass::Active, OrbitalClass::Active) => {
            Some(ExcitationClass::CCToAA)
        }
        (OrbitalClass::Core, OrbitalClass::Core, OrbitalClass::Active, OrbitalClass::Virtual) => {
            Some(ExcitationClass::CCToAV)
        }
        (OrbitalClass::Core, OrbitalClass::Active, OrbitalClass::Active, OrbitalClass::Active) => {
            Some(ExcitationClass::CAToAA)
        }
        (OrbitalClass::Core, OrbitalClass::Active, OrbitalClass::Active, OrbitalClass::Virtual) => {
            Some(ExcitationClass::CAToAV)
        }
        (OrbitalClass::Core, OrbitalClass::Active, OrbitalClass::Virtual, OrbitalClass::Active) => {
            Some(ExcitationClass::CAToVA)
        }
        (
            OrbitalClass::Core,
            OrbitalClass::Active,
            OrbitalClass::Virtual,
            OrbitalClass::Virtual,
        ) => Some(ExcitationClass::CAToVV),
        (
            OrbitalClass::Active,
            OrbitalClass::Active,
            OrbitalClass::Active,
            OrbitalClass::Active,
        ) => Some(ExcitationClass::AAToAA),
        (
            OrbitalClass::Active,
            OrbitalClass::Active,
            OrbitalClass::Active,
            OrbitalClass::Virtual,
        ) => Some(ExcitationClass::AAToAV),
        (
            OrbitalClass::Active,
            OrbitalClass::Active,
            OrbitalClass::Virtual,
            OrbitalClass::Virtual,
        ) => Some(ExcitationClass::AAToVV),
        _ => None,
    }
}

/// Check whether an orbital index belongs to the active space.
/// # Arguments:
/// - `spaces`: NOCC orbital spaces.
/// - `p`: Orbital index.
/// # Returns:
/// - `bool`: `true` if `p` is active, otherwise `false`.
fn is_active(
    spaces: &Spaces,
    p: usize,
) -> bool {
    spaces.class_of[p] == OrbitalClass::Active
}

/// Check whether a double excitation contains an active spectator index.
/// # Arguments:
/// - `spaces`: NOCC orbital spaces.
/// - `p`: First creator-side orbital index.
/// - `q`: Second creator-side orbital index.
/// - `r`: First annihilator-side orbital index.
/// - `s`: Second annihilator-side orbital index.
/// # Returns:
/// - `bool`: `true` if an active creator also appears on the annihilator side.
fn has_active_spectator(
    spaces: &Spaces,
    p: usize,
    q: usize,
    r: usize,
    s: usize,
) -> bool {
    (is_active(spaces, p) && (p == r || p == s)) || (is_active(spaces, q) && (q == r || q == s))
}

/// Check whether a double excitation has a repeated same-side active pair.
/// # Arguments:
/// - `spaces`: NOCC orbital spaces.
/// - `p`: First creator-side orbital index.
/// - `q`: Second creator-side orbital index.
/// - `r`: First annihilator-side orbital index.
/// - `s`: Second annihilator-side orbital index.
/// # Returns:
/// - `bool`: `true` if either active creator pair or active annihilator pair is repeated.
fn has_repeated_same_side_active_pair(
    spaces: &Spaces,
    p: usize,
    q: usize,
    r: usize,
    s: usize,
) -> bool {
    (is_active(spaces, p) && is_active(spaces, q) && p == q)
        || (is_active(spaces, r) && is_active(spaces, s) && r == s)
}

/// Check whether a supported double excitation is retained as a representative direction.
/// # Arguments:
/// - `spaces`: NOCC orbital spaces.
/// - `p`: First creator-side orbital index.
/// - `q`: Second creator-side orbital index.
/// - `r`: First annihilator-side orbital index.
/// - `s`: Second annihilator-side orbital index.
/// # Returns:
/// - `bool`: `true` if the excitation should be retained in the FOIS basis.
fn is_representative_double(
    spaces: &Spaces,
    p: usize,
    q: usize,
    r: usize,
    s: usize,
) -> bool {
    if has_repeated_same_side_active_pair(spaces, p, q, r, s) {
        return false;
    }

    if has_active_spectator(spaces, p, q, r, s) {
        return false;
    }

    true
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
            if p == q {
                continue;
            }

            if single_excitation_class(spaces, p, q).is_none() {
                continue;
            }

            out.push(Excitation::Single { p, q });
        }
    }

    for &p in spaces.creators.iter() {
        for &q in spaces.creators.iter() {
            for &r in spaces.annihilators.iter() {
                for &s in spaces.annihilators.iter() {
                    if p == r && q == s {
                        continue;
                    }

                    if double_excitation_class(spaces, p, q, r, s).is_none() {
                        continue;
                    }

                    if !is_representative_double(spaces, p, q, r, s) {
                        continue;
                    }

                    out.push(Excitation::Double { p, q, r, s });
                }
            }
        }
    }

    out
}

/// Classify a spin-free excitation by orbital spaces.
/// # Arguments:
/// - `spaces`: NOCC orbital spaces.
/// - `ex`: Spin-free excitation.
/// # Returns:
/// - `ExcitationClass`: Excitation class used for overlap dispatch.
pub(in crate::deterministic::noccmc) fn excitation_class(
    spaces: &Spaces,
    ex: Excitation,
) -> ExcitationClass {
    match ex {
        Excitation::Single { p, q } => single_excitation_class(spaces, p, q).unwrap_or_else(|| {
            panic!(
                "unsupported single excitation class: lower {:?}, upper {:?}, excitation {:?}",
                spaces.class_of[q],
                spaces.class_of[p],
                ex,
            )
        }),
        Excitation::Double { p, q, r, s } => {
            double_excitation_class(spaces, p, q, r, s).unwrap_or_else(|| {
                panic!(
                    "unsupported double excitation class: lower {:?} {:?}, upper {:?} {:?}, excitation {:?}",
                    spaces.class_of[r],
                    spaces.class_of[s],
                    spaces.class_of[p],
                    spaces.class_of[q],
                    ex,
                )
            })
        }
    }
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
            s[(i, j)] = overlap::overlap_element(left, right, spaces, gamma1, lambdas);
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
