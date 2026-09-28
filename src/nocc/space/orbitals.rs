// nocc/space/orbitals.rs
//! Core, active and virtual orbital spaces in the NOCI natural-orbital basis.

// Crate-root imports.
use crate::nocc::rdm::RDM1;

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

    // Outside the explicit active set, classify orbitals by diagonal natural
    // occupation: core near two electrons and virtual near zero.
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

    // The excitation manifold creates in `A \cup V` and annihilates from
    // `C \cup A`.
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
