// nocc/space/excitations.rs
//! Spin-free GNOCC excitations, their classes and dense amplitude tensors.

// External crate imports.
use ndarray::{Array1, Array2, Array4};

// Parent/sibling imports.
use super::orbitals::{OrbitalClass, Spaces};

/// Spin-free GNOCC excitation class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(in crate::nocc) enum ExcitationClass {
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
    /// CC -> VV double excitation.
    CCToVV,
}

impl ExcitationClass {
    /// Every excitation class.
    pub(in crate::nocc) const ALL: [Self; 14] = [
        Self::CToA,
        Self::AToA,
        Self::AToV,
        Self::CCToAA,
        Self::CCToAV,
        Self::CAToAA,
        Self::CAToAV,
        Self::CAToVA,
        Self::CAToVV,
        Self::AAToAA,
        Self::AAToAV,
        Self::AAToVV,
        Self::CToV,
        Self::CCToVV,
    ];

    /// Return the generated term-table name of the excitation class.
    /// # Arguments:
    /// - `self`: Excitation class.
    /// # Returns:
    /// - `&'static str`: Generated excitation class name.
    pub(in crate::nocc) fn name(self) -> &'static str {
        match self {
            Self::CToA => "CToA",
            Self::CToV => "CToV",
            Self::AToA => "AToA",
            Self::AToV => "AToV",
            Self::CCToAA => "CCToAA",
            Self::CCToAV => "CCToAV",
            Self::CAToAA => "CAToAA",
            Self::CAToAV => "CAToAV",
            Self::CAToVA => "CAToVA",
            Self::CAToVV => "CAToVV",
            Self::AAToAA => "AAToAA",
            Self::AAToAV => "AAToAV",
            Self::AAToVV => "AAToVV",
            Self::CCToVV => "CCToVV",
        }
    }

    /// Convert a generated term-table class name.
    /// # Arguments:
    /// - `name`: Generated excitation class name.
    /// # Returns:
    /// - `Self`: Excitation class.
    /// # Panics
    /// - Panics if `name` does not identify a known excitation class.
    pub(in crate::nocc) fn from_name(name: &str) -> Self {
        Self::ALL
            .into_iter()
            .find(|class| class.name() == name)
            .unwrap_or_else(|| panic!("unknown excitation class {name}"))
    }
}

/// Spin-free GNOCC excitation operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Excitation {
    /// `Spin-free single excitation E^p_q.`
    Single { p: usize, q: usize },
    /// `Spin-free double excitation E^{pq}_{rs}.`
    Double {
        p: usize,
        q: usize,
        r: usize,
        s: usize,
    },
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
        (OrbitalClass::Core, OrbitalClass::Core, OrbitalClass::Virtual, OrbitalClass::Virtual) => {
            Some(ExcitationClass::CCToVV)
        }
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

    // Enumerate supported creator-annihilator class pairs for singles.
    for &p in spaces.creators.iter() {
        for &q in spaces.annihilators.iter() {
            if single_excitation_class(spaces, p, q).is_some() {
                out.push(Excitation::Single { p, q });
            }
        }
    }

    // Doubles retain the creator and annihilator order required by the
    // spin-free excitation operator `E^{pq}_{rs}`.
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
/// # Panics
/// - Panics if the excitation has an unsupported orbital-space class.
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

/// Dense spin-free amplitude tensors of one cluster operator.
pub(in crate::nocc) struct DenseAmplitudes {
    /// Singles amplitudes `t^q_p`, stored as `[q, p]`.
    pub(in crate::nocc) t1: Array2<f64>,
    /// Pair-symmetric doubles amplitudes `\bar t^{rs}_{pq}`, stored as `[r, s, p, q]`.
    pub(in crate::nocc) t2: Array4<f64>,
}

/// Build the dense amplitude tensors of one amplitude vector.
/// The cluster operator is `\hat T = \sum_\mu t_\mu \hat\tau_\mu` over the excitation list.
/// The generated tables use a pair-symmetric `\bar t` with
/// `\hat T_2 = \tfrac12 \sum \bar t^{rs}_{pq} \hat E^{pq}_{rs}` over all orbitals, and
/// `\hat E^{qp}_{sr} = \hat E^{pq}_{rs}`, so each amplitude enters both `X` and its pair swap
/// `PX`: `\bar t_X = \bar t_{PX} = t_X + t_{PX}`, where `t_{PX}` is zero when the swapped
/// excitation is not in the list.
/// # Arguments:
/// - `spaces`: Core, active, and virtual orbital-space maps.
/// - `excitations`: Raw spin-free excitation list.
/// - `amplitudes`: Cluster amplitude vector in the same order as the excitation list.
/// # Returns:
/// - `DenseAmplitudes`: Dense `t_1` and `\bar t_2` tensors.
pub(in crate::nocc) fn dense_amplitudes(
    spaces: &Spaces,
    excitations: &[Excitation],
    amplitudes: &Array1<f64>,
) -> DenseAmplitudes {
    let n = spaces.nmo;
    let mut t1 = Array2::<f64>::zeros((n, n));
    let mut t2 = Array4::<f64>::zeros((n, n, n, n));

    for (nu, &ex) in excitations.iter().enumerate() {
        match ex {
            Excitation::Single { p, q } => {
                t1[(q, p)] = amplitudes[nu];
            }
            Excitation::Double { p, q, r, s } => {
                t2[(r, s, p, q)] += amplitudes[nu];
                t2[(s, r, q, p)] += amplitudes[nu];
            }
        }
    }

    DenseAmplitudes { t1, t2 }
}
