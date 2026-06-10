// nocc/space.rs

use ndarray::{Array1, Array2};
use rayon::prelude::*;

use super::overlap;
use crate::AoData;
use crate::maths::linalg::loewdin_x;
use crate::nocc::{Cumulants, RDM1};
use crate::scf::fock;

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

/// Reusable raw and orthogonalized FOIS basis data.
pub(crate) struct FoisBasis {
    /// Raw spin-free FOIS metric S.
    pub metric: Array2<f64>,
    /// Hamiltonian coupling weights h.
    pub h: Array1<f64>,
    /// Weighted metric h S h.
    pub weighted_metric: Array2<f64>,
    /// Canonical FOIS transformation Y = h X.
    pub y: Array2<f64>,
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

/// Build the raw spin-free singles and doubles excitation list.
/// # Arguments:
/// - `spaces`: NOCC orbital spaces.
/// # Returns:
/// - `Vec<Excitation>`: Spin-free GNOCCSD excitations.
pub(crate) fn build_excitations(spaces: &Spaces) -> Vec<Excitation> {
    let mut out = Vec::new();

    for &p in spaces.creators.iter() {
        for &q in spaces.annihilators.iter() {
            if single_excitation_class(spaces, p, q).is_some() {
                out.push(Excitation::Single { p, q });
            }
        }
    }

    for &p in spaces.creators.iter() {
        for &q in spaces.creators.iter() {
            for &r in spaces.annihilators.iter() {
                for &s in spaces.annihilators.iter() {
                    if double_excitation_class(spaces, p, q, r, s).is_some() {
                        out.push(Excitation::Double { p, q, r, s });
                    }
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
pub(in crate::nocc) fn excitation_class(
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

/// Build the weighted FOIS basis from the full raw excitation list.
/// # Arguments:
/// - `ao`: Integrals transformed to the NOCI natural-orbital basis.
/// - `gamma1`: Full-space spin-free one-body RDM.
/// - `lambdas`: Active-space spin-free cumulants.
/// - `spaces`: NOCC orbital spaces.
/// - `excitations`: Raw spin-free excitation list.
/// - `tol`: Weighted overlap eigenvalue threshold.
/// # Returns:
/// - `FoisBasis`: Raw metric, Hamiltonian weights, weighted metric, and Y.
pub(crate) fn build_fois_basis(
    ao: &AoData,
    gamma1: &RDM1<f64>,
    lambdas: &Cumulants<f64>,
    spaces: &Spaces,
    excitations: &[Excitation],
    tol: f64,
) -> FoisBasis {
    let nexc = excitations.len();
    let upper_rows: Vec<Vec<f64>> = (0..nexc)
        .into_par_iter()
        .map(|i| {
            let left = excitations[i];
            (i..nexc)
                .map(|j| overlap::overlap_element(left, excitations[j], spaces, gamma1, lambdas))
                .collect()
        })
        .collect();
    let mut s = Array2::zeros((nexc, nexc));

    for (i, row) in upper_rows.iter().enumerate() {
        for (offset, &value) in row.iter().enumerate() {
            let j = i + offset;
            s[(i, j)] = value;
            s[(j, i)] = value;
        }
    }

    let h = hamiltonian_weights(ao, gamma1, excitations);
    let mut stilde: Array2<f64> = Array2::zeros(s.raw_dim());

    for i in 0..s.nrows() {
        for j in 0..s.ncols() {
            stilde[(i, j)] = h[i] * s[(i, j)] * h[j];
        }
    }

    let xtilde = loewdin_x(&stilde, true, tol);
    let mut y = xtilde.clone();

    for mu in 0..h.len() {
        for col in 0..y.ncols() {
            y[(mu, col)] *= h[mu];
        }
    }

    FoisBasis {
        metric: s,
        h,
        weighted_metric: stilde,
        y,
    }
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
