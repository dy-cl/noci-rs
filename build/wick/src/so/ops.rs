// so/ops.rs

// External crate imports.
use num_rational::Ratio;
use smallvec::SmallVec;

// Parent/sibling imports.
use super::{Kind, Space};

/// One spin-orbital operator component `c X^{u_1..u_n}_{l_1..l_m} a^\dagger_{l_1}..a^\dagger_{l_m}
/// a_{u_n}..a_{u_1}` summed over its indices, with annihilated spaces `u` and created spaces
/// `l`. Spaces are sorted within each side, and `c = 1 / \prod_s n_s!` counts identical spaces on
/// each side, following the Wick&D operator convention.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Component {
    /// Tensor kind.
    pub(crate) kind: Kind,
    /// Annihilated spaces, sorted.
    pub(crate) ann: SmallVec<[Space; 4]>,
    /// Created spaces, sorted.
    pub(crate) cre: SmallVec<[Space; 4]>,
    /// Prefactor.
    pub(crate) coeff: Ratio<i64>,
}

/// Spin-orbital excitation types of the cluster operator as `(created, annihilated)` spaces.
/// Spin-free `CA -> AV` and `CA -> VA` coincide in the antisymmetric spin-orbital basis.
const CLUSTER: &[(&[Space], &[Space])] = &[
    (&[Space::Active], &[Space::Core]),
    (&[Space::Virtual], &[Space::Active]),
    (&[Space::Active], &[Space::Active]),
    (&[Space::Virtual], &[Space::Core]),
    (
        &[Space::Active, Space::Virtual],
        &[Space::Core, Space::Active],
    ),
    (
        &[Space::Virtual, Space::Virtual],
        &[Space::Core, Space::Active],
    ),
    (
        &[Space::Active, Space::Virtual],
        &[Space::Core, Space::Core],
    ),
    (&[Space::Active, Space::Active], &[Space::Core, Space::Core]),
    (
        &[Space::Active, Space::Active],
        &[Space::Core, Space::Active],
    ),
    (
        &[Space::Active, Space::Virtual],
        &[Space::Active, Space::Active],
    ),
    (
        &[Space::Virtual, Space::Virtual],
        &[Space::Active, Space::Active],
    ),
    (
        &[Space::Active, Space::Active],
        &[Space::Active, Space::Active],
    ),
];

/// Every orbital space.
const SPACES: [Space; 3] = [Space::Core, Space::Active, Space::Virtual];

/// Build one component with the Wick&D prefactor.
/// # Arguments:
/// - `kind`: Tensor kind.
/// - `cre`: Created spaces.
/// - `ann`: Annihilated spaces.
/// # Returns:
/// - `Component`: Operator component.
pub(crate) fn component(
    kind: Kind,
    cre: &[Space],
    ann: &[Space],
) -> Component {
    let mut cre = SmallVec::<[Space; 4]>::from_slice(cre);
    let mut ann = SmallVec::<[Space; 4]>::from_slice(ann);
    cre.sort_unstable();
    ann.sort_unstable();

    // `1 / \prod_s n_s!` over repeated spaces on each side.
    let mut den = 1i64;
    for side in [&cre, &ann] {
        for s in SPACES {
            let n = side.iter().filter(|&&x| x == s).count() as i64;
            den *= (1..=n).product::<i64>();
        }
    }

    Component {
        kind,
        ann,
        cre,
        coeff: Ratio::new(1, den),
    }
}

/// Return the normal-ordered spin-orbital Hamiltonian `f + v` over every space block.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Vec<Component>`: One-body Fock and antisymmetrised two-body components.
pub(crate) fn hamiltonian() -> Vec<Component> {
    let mut out = Vec::new();

    for c in SPACES {
        for a in SPACES {
            out.push(component(Kind::Fock, &[c], &[a]));
        }
    }

    // Unordered space pairs on each side of the two-body operator.
    let pairs = SPACES
        .iter()
        .enumerate()
        .flat_map(|(i, &x)| SPACES[i..].iter().map(move |&y| [x, y]))
        .collect::<Vec<_>>();
    for c in &pairs {
        for a in &pairs {
            out.push(component(Kind::Eri, c, a));
        }
    }

    out
}

/// Return the spin-orbital cluster operator over every supported excitation type.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Vec<Component>`: Singles and doubles amplitude components.
pub(crate) fn cluster() -> Vec<Component> {
    CLUSTER
        .iter()
        .map(|&(cre, ann)| {
            let kind = if cre.len() == 1 { Kind::T1 } else { Kind::T2 };
            component(kind, cre, ann)
        })
        .collect()
}

/// Return the residual projector `\tau^\dagger` of one excitation class.
/// The excitation `\tau` creates into `targets` and annihilates from `sources`, so its adjoint
/// creates into the sources and annihilates from the targets.
/// # Arguments:
/// - `sources`: Spaces the excitation annihilates from.
/// - `targets`: Spaces the excitation creates into.
/// # Returns:
/// - `Component`: Projector component with kind `Kind::Bra`.
pub(crate) fn bra(
    sources: &[Space],
    targets: &[Space],
) -> Component {
    component(Kind::Bra, sources, targets)
}

/// Return the metric excitation `\tau` of one excitation class.
/// # Arguments:
/// - `sources`: Spaces the excitation annihilates from.
/// - `targets`: Spaces the excitation creates into.
/// # Returns:
/// - `Component`: Excitation component with kind `Kind::Ket`.
pub(crate) fn ket(
    sources: &[Space],
    targets: &[Space],
) -> Component {
    component(Kind::Ket, targets, sources)
}

/// Return the residual projector of one spin-orbital excitation class by name.
/// Spin-free `CAToAV` and `CAToVA` share the spin-orbital class `CAToAV`.
/// # Arguments:
/// - `name`: Excitation class name.
/// # Returns:
/// - `Option<Component>`: Projector component, or `None` for an unknown class.
pub(crate) fn class(name: &str) -> Option<Component> {
    spaces(name).map(|(sources, targets)| bra(sources, targets))
}

/// Return the metric excitation of one spin-orbital excitation class by name.
/// # Arguments:
/// - `name`: Excitation class name.
/// # Returns:
/// - `Option<Component>`: Excitation component, or `None` for an unknown class.
pub(crate) fn excitation(name: &str) -> Option<Component> {
    spaces(name).map(|(sources, targets)| ket(sources, targets))
}

/// Return the source and target spaces of one excitation class by name.
/// # Arguments:
/// - `name`: Excitation class name.
/// # Returns:
/// - `Option<(&'static [Space], &'static [Space])>`: Annihilated and created spaces.
fn spaces(name: &str) -> Option<(&'static [Space], &'static [Space])> {
    const C: Space = Space::Core;
    const A: Space = Space::Active;
    const V: Space = Space::Virtual;
    let (sources, targets): (&'static [Space], &'static [Space]) = match name {
        "CToA" => (&[C], &[A]),
        "AToV" => (&[A], &[V]),
        "AToA" => (&[A], &[A]),
        "CToV" => (&[C], &[V]),
        "CAToAV" | "CAToVA" => (&[C, A], &[A, V]),
        "CAToVV" => (&[C, A], &[V, V]),
        "CCToAV" => (&[C, C], &[A, V]),
        "CCToAA" => (&[C, C], &[A, A]),
        "CAToAA" => (&[C, A], &[A, A]),
        "AAToAV" => (&[A, A], &[A, V]),
        "AAToVV" => (&[A, A], &[V, V]),
        "AAToAA" => (&[A, A], &[A, A]),
        _ => return None,
    };

    Some((sources, targets))
}
