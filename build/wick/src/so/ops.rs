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
    (
        &[Space::Virtual, Space::Virtual],
        &[Space::Core, Space::Core],
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
pub(crate) fn operator_component(
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
pub(crate) fn normal_ordered_hamiltonian() -> Vec<Component> {
    let mut out = Vec::new();

    for c in SPACES {
        for a in SPACES {
            out.push(operator_component(Kind::Fock, &[c], &[a]));
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
            out.push(operator_component(Kind::Eri, c, a));
        }
    }

    out
}

/// Return the normal-ordered spin-orbital Dyall Hamiltonian.
/// Normal ordering `\hat H_0 = \sum_{ij} f^j_i\hat E^i_j + \sum_{ab} f^b_a\hat E^a_b +
/// \sum_{tu} f^u_t\hat E^t_u + \tfrac12\sum_{tuvw} g^{vw}_{tu}\hat E^{tu}_{vw}` with respect to the
/// reference turns the active one-body part into the generalised Fock operator, so `\hat H_0`
/// keeps the core-core, active-active and virtual-virtual Fock blocks and the all-active
/// two-body block of the normal-ordered Hamiltonian.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Vec<Component>`: Diagonal-block Fock and active two-body components.
/// # References
/// - Dyall, *J. Chem. Phys.* **102**, 4909 (1995); Lee and Tew, arXiv:2507.13472 (2025),
///   Eq. (59).
pub(crate) fn dyall_hamiltonian() -> Vec<Component> {
    let mut out = SPACES
        .iter()
        .map(|&s| operator_component(Kind::Fock, &[s], &[s]))
        .collect::<Vec<_>>();
    out.push(operator_component(
        Kind::Eri,
        &[Space::Active, Space::Active],
        &[Space::Active, Space::Active],
    ));

    out
}

/// Return the spin-orbital cluster operator over every supported excitation type.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Vec<Component>`: Singles and doubles amplitude components.
pub(crate) fn cluster_operator() -> Vec<Component> {
    CLUSTER
        .iter()
        .map(|&(cre, ann)| {
            let kind = if cre.len() == 1 { Kind::T1 } else { Kind::T2 };
            operator_component(kind, cre, ann)
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
pub(crate) fn projector_component(
    sources: &[Space],
    targets: &[Space],
) -> Component {
    operator_component(Kind::Bra, sources, targets)
}

/// Return the metric excitation `\tau` of one excitation class.
/// # Arguments:
/// - `sources`: Spaces the excitation annihilates from.
/// - `targets`: Spaces the excitation creates into.
/// # Returns:
/// - `Component`: Excitation component with kind `Kind::Ket`.
pub(crate) fn excitation_component(
    sources: &[Space],
    targets: &[Space],
) -> Component {
    operator_component(Kind::Ket, targets, sources)
}

/// Return the residual projector of one spin-orbital excitation class by name.
/// Spin-free `CAToAV` and `CAToVA` share the spin-orbital class `CAToAV`.
/// # Arguments:
/// - `name`: Excitation class name.
/// # Returns:
/// - `Option<Component>`: Projector component, or `None` for an unknown class.
pub(crate) fn projector_for_class(name: &str) -> Option<Component> {
    class_spaces(name).map(|(sources, targets)| projector_component(sources, targets))
}

/// Return the metric excitation of one spin-orbital excitation class by name.
/// # Arguments:
/// - `name`: Excitation class name.
/// # Returns:
/// - `Option<Component>`: Excitation component, or `None` for an unknown class.
pub(crate) fn excitation_for_class(name: &str) -> Option<Component> {
    class_spaces(name).map(|(sources, targets)| excitation_component(sources, targets))
}

/// Return the source and target spaces of one excitation class by name.
/// # Arguments:
/// - `name`: Excitation class name.
/// # Returns:
/// - `Option<(&'static [Space], &'static [Space])>`: Annihilated and created spaces.
fn class_spaces(name: &str) -> Option<(&'static [Space], &'static [Space])> {
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
        "CCToVV" => (&[C, C], &[V, V]),
        "AAToAA" => (&[A, A], &[A, A]),
        _ => return None,
    };

    Some((sources, targets))
}
