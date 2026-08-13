// noci/factorise/mod.rs
//! Spin-factorised determinant topology and factorised NOCI operators.

mod onebody;
mod overlap;
mod storage;

// Crate-visible type re-exports.
pub(crate) use onebody::{OneBodyBackend, OneBodyBlockPlan, OneBodyContraction, OneBodyPlan};
pub(crate) use overlap::OverlapScratch;

// Standard library imports.
use std::collections::{BTreeMap, HashMap};

// Crate-root imports.
use crate::DetState;

// Parent/sibling imports.
use super::types::{NOCIData, NOCIScalar};

/// Actual determinant entry in a parent-local spin factorisation.
#[derive(Clone, Copy)]
pub(super) struct FactorEntry {
    /// Global determinant index `I`.
    pub(super) det: usize,
    /// Parent-local alpha component `a_I`.
    pub(super) a: usize,
    /// Parent-local beta component `b_I`.
    pub(super) b: usize,
}

/// Populated joint alpha/beta excitation-rank sector for one parent.
#[derive(Clone)]
pub(super) struct ParentRankBlock {
    /// Alpha excitation rank `r`.
    pub(super) alpha_rank: usize,
    /// Beta excitation rank `s`.
    pub(super) beta_rank: usize,
    /// Parent-local alpha components in rank-local order.
    pub(super) alpha_components: Vec<usize>,
    /// Parent-local beta components in rank-local order.
    pub(super) beta_components: Vec<usize>,
    /// Global determinant IDs in alpha-major row-major order when dense.
    pub(super) dets: Vec<usize>,
    /// Whether this sector is the complete Cartesian product without duplicates.
    pub(super) dense: bool,
}

/// Parent-local determinant space in the shared spin factorisation.
#[derive(Default)]
pub(super) struct ParentSpinSpace {
    /// Representative determinant for each parent-local alpha component.
    pub(super) areps: Vec<usize>,
    /// Representative determinant for each parent-local beta component.
    pub(super) breps: Vec<usize>,
    /// Actual determinants belonging to this parent as `(I,a_I,b_I)`.
    pub(super) entries: Vec<FactorEntry>,
    /// Representative determinant for each parent-local occupation pair.
    pub(super) oreps: Vec<usize>,
    /// Occupation-pair ID keyed by determinant offset from `first_det`.
    pub(super) oids: Vec<usize>,
    /// First determinant index belonging to this parent.
    pub(super) first_det: usize,
    /// One-past-last determinant index belonging to this parent when parent blocks are contiguous.
    pub(super) last_det: usize,
    /// Populated joint alpha/beta excitation-rank sectors.
    pub(super) rank_blocks: Vec<ParentRankBlock>,
}

/// Shared determinant-space spin factorisation `I <-> (P,a_I,b_I)`.
pub(crate) struct SpinFactorisation {
    /// Alpha compact IDs keyed by determinant index and local to the determinant parent.
    pub(super) aids: Vec<usize>,
    /// Beta compact IDs keyed by determinant index and local to the determinant parent.
    pub(super) bids: Vec<usize>,
    /// Largest number of unique alpha spin components in one parent reference.
    pub(super) ma: usize,
    /// Largest number of unique beta spin components in one parent reference.
    pub(super) mb: usize,
    /// Parent-local determinant ranges, representatives and actual determinant entries.
    pub(super) parents: Vec<ParentSpinSpace>,
}

impl SpinFactorisation {
    /// Construct `I <-> (P,a_I,b_I)` for a NOCI determinant basis.
    /// # Arguments:
    /// - `data`: Shared NOCI data defining the determinant basis and parent references.
    /// # Returns
    /// - `SpinFactorisation`: Shared parent-local spin topology.
    pub(crate) fn new<T: NOCIScalar>(data: &NOCIData<'_, T>) -> Self {
        let mut aids = vec![0usize; data.basis.len()];
        let mut bids = vec![0usize; data.basis.len()];
        let ma = assign_aids(data.basis, &mut aids);
        let mb = assign_bids(data.basis, &mut bids);
        let mut parents = build_parent_spin_spaces(data.basis, &aids, &bids);
        build_parent_rank_blocks(data.basis, &mut parents);

        Self {
            aids,
            bids,
            ma,
            mb,
            parents,
        }
    }
}

/// Ordered Wick parent pair `(x,w)` and whether target parent is left.
/// Existing factorised overlap evaluates Wick factors with the earlier parent block on the left.
/// This preserves that ordering convention for every operator using the shared topology.
/// # Arguments:
/// - `factorisation`: Shared determinant-space spin topology.
/// - `target_parent`: Target parent `Q`.
/// - `source_parent`: Source parent `P`.
/// # Returns
/// - `(usize, usize, bool)`: Ordered pair `(lp,gp,target_left)`.
pub(super) fn ordered_parent_pair(
    factorisation: &SpinFactorisation,
    target_parent: usize,
    source_parent: usize,
) -> (usize, usize, bool) {
    if factorisation.parents[target_parent].first_det
        <= factorisation.parents[source_parent].first_det
    {
        (target_parent, source_parent, true)
    } else {
        (source_parent, target_parent, false)
    }
}

/// Assign compact alpha IDs by sorting determinant indices and deduplicating consecutive identities.
/// # Arguments:
/// - `basis`: NOCI determinant basis.
/// - `aids`: Output alpha compact IDs keyed by determinant index.
/// # Returns
/// - `usize`: Largest number of unique alpha components in any parent.
fn assign_aids<T: NOCIScalar>(
    basis: &[DetState<T>],
    aids: &mut [usize],
) -> usize {
    let mut indices = (0..basis.len()).collect::<Vec<_>>();

    indices.sort_unstable_by(|&i, &j| {
        let id = &basis[i];
        let jd = &basis[j];
        id.parent
            .cmp(&jd.parent)
            .then_with(|| id.oa.cmp(&jd.oa))
            .then_with(|| id.excitation.alpha.holes.cmp(&jd.excitation.alpha.holes))
            .then_with(|| id.excitation.alpha.parts.cmp(&jd.excitation.alpha.parts))
            .then_with(|| id.pha.to_bits().cmp(&jd.pha.to_bits()))
    });

    assign_spin_ids(&indices, basis, aids, same_alpha_key)
}

/// Assign compact beta IDs by sorting determinant indices and deduplicating consecutive identities.
/// # Arguments:
/// - `basis`: NOCI determinant basis.
/// - `bids`: Output beta compact IDs keyed by determinant index.
/// # Returns
/// - `usize`: Largest number of unique beta components in any parent.
fn assign_bids<T: NOCIScalar>(
    basis: &[DetState<T>],
    bids: &mut [usize],
) -> usize {
    let mut indices = (0..basis.len()).collect::<Vec<_>>();

    indices.sort_unstable_by(|&i, &j| {
        let id = &basis[i];
        let jd = &basis[j];
        id.parent
            .cmp(&jd.parent)
            .then_with(|| id.ob.cmp(&jd.ob))
            .then_with(|| id.excitation.beta.holes.cmp(&jd.excitation.beta.holes))
            .then_with(|| id.excitation.beta.parts.cmp(&jd.excitation.beta.parts))
            .then_with(|| id.phb.to_bits().cmp(&jd.phb.to_bits()))
    });

    assign_spin_ids(&indices, basis, bids, same_beta_key)
}

/// Assign compact same-spin IDs after sorting by parent-local spin key.
/// # Arguments:
/// - `indices`: Determinant indices sorted by parent and same-spin key.
/// - `basis`: NOCI determinant basis.
/// - `ids`: Output compact IDs keyed by determinant index.
/// - `same_key`: Same-spin key equality predicate.
/// # Returns
/// - `usize`: Largest number of unique components in any parent.
fn assign_spin_ids<T, F>(
    indices: &[usize],
    basis: &[DetState<T>],
    ids: &mut [usize],
    same_key: F,
) -> usize
where
    T: NOCIScalar,
    F: Fn(&DetState<T>, &DetState<T>) -> bool,
{
    let mut last = usize::MAX;
    let mut next = 0usize;
    let mut maxu = 0usize;

    for (pos, &det) in indices.iter().enumerate() {
        let parent = basis[det].parent;
        if pos == 0 || parent != basis[last].parent {
            if pos != 0 {
                maxu = maxu.max(next);
            }
            next = 0;
            ids[det] = next;
            next += 1;
        } else if same_key(&basis[last], &basis[det]) {
            ids[det] = ids[last];
        } else {
            ids[det] = next;
            next += 1;
        }
        last = det;
    }

    maxu.max(next)
}

/// Test equality of parent-local alpha determinant components.
/// # Arguments:
/// - `lhs`: Previous determinant in sorted alpha key order.
/// - `rhs`: Current determinant in sorted alpha key order.
/// # Returns
/// - `bool`: Whether both determinants share one alpha component ID.
fn same_alpha_key<T: NOCIScalar>(
    lhs: &DetState<T>,
    rhs: &DetState<T>,
) -> bool {
    lhs.oa == rhs.oa
        && lhs.excitation.alpha.holes == rhs.excitation.alpha.holes
        && lhs.excitation.alpha.parts == rhs.excitation.alpha.parts
        && lhs.pha.to_bits() == rhs.pha.to_bits()
}

/// Test equality of parent-local beta determinant components.
/// # Arguments:
/// - `lhs`: Previous determinant in sorted beta key order.
/// - `rhs`: Current determinant in sorted beta key order.
/// # Returns
/// - `bool`: Whether both determinants share one beta component ID.
fn same_beta_key<T: NOCIScalar>(
    lhs: &DetState<T>,
    rhs: &DetState<T>,
) -> bool {
    lhs.ob == rhs.ob
        && lhs.excitation.beta.holes == rhs.excitation.beta.holes
        && lhs.excitation.beta.parts == rhs.excitation.beta.parts
        && lhs.phb.to_bits() == rhs.phb.to_bits()
}

/// Build parent-local spin and occupation representative tables.
/// `D_P` is stored as actual determinant entries `(I,a_I,b_I)` and is not assumed to span
/// the complete Cartesian product `A_P \times B_P`.
/// # Arguments:
/// - `basis`: NOCI determinant basis.
/// - `aids`: Parent-local alpha component IDs keyed by determinant.
/// - `bids`: Parent-local beta component IDs keyed by determinant.
/// # Returns
/// - `Vec<ParentSpinSpace>`: Per-parent determinant ranges, representatives, occupation IDs and actual entries.
fn build_parent_spin_spaces<T: NOCIScalar>(
    basis: &[DetState<T>],
    aids: &[usize],
    bids: &[usize],
) -> Vec<ParentSpinSpace> {
    let nparents = basis
        .iter()
        .map(|det| det.parent)
        .max()
        .map(|parent| parent + 1)
        .unwrap_or(0);
    let mut parents = (0..nparents)
        .map(|_| ParentSpinSpace {
            areps: Vec::new(),
            breps: Vec::new(),
            entries: Vec::new(),
            oreps: Vec::new(),
            oids: Vec::new(),
            first_det: usize::MAX,
            last_det: 0,
            rank_blocks: Vec::new(),
        })
        .collect::<Vec<_>>();

    for (det, state) in basis.iter().enumerate() {
        let parent = &mut parents[state.parent];
        parent.first_det = parent.first_det.min(det);
        parent.last_det = parent.last_det.max(det + 1);
        parent.entries.push(FactorEntry {
            det,
            a: aids[det],
            b: bids[det],
        });

        if parent.areps.len() <= aids[det] {
            parent.areps.resize(aids[det] + 1, usize::MAX);
        }
        if parent.areps[aids[det]] == usize::MAX {
            parent.areps[aids[det]] = det;
        }

        if parent.breps.len() <= bids[det] {
            parent.breps.resize(bids[det] + 1, usize::MAX);
        }
        if parent.breps[bids[det]] == usize::MAX {
            parent.breps[bids[det]] = det;
        }
    }

    for parent in &mut parents {
        if parent.first_det != usize::MAX {
            parent
                .oids
                .resize(parent.last_det - parent.first_det, usize::MAX);
        }
    }

    let mut occupation_ids = (0..nparents)
        .map(|_| HashMap::new())
        .collect::<Vec<HashMap<(u128, u128), usize>>>();

    for (det, state) in basis.iter().enumerate() {
        let parent = &mut parents[state.parent];
        let oid = *occupation_ids[state.parent]
            .entry((state.oa, state.ob))
            .or_insert_with(|| {
                parent.oreps.push(det);
                parent.oreps.len() - 1
            });
        parent.oids[det - parent.first_det] = oid;
    }

    parents
}

/// Build populated joint excitation-rank sectors from actual determinant entries.
/// A sector is dense only when every alpha/beta rank-local pair occurs exactly once.
/// # Arguments:
/// - `basis`: NOCI determinant basis defining excitation ranks.
/// - `parents`: Parent-local spin spaces to extend with rank sectors.
/// # Returns
/// - `()`: Stores deterministic rank-block descriptors on each parent.
fn build_parent_rank_blocks<T: NOCIScalar>(
    basis: &[DetState<T>],
    parents: &mut [ParentSpinSpace],
) {
    for parent in parents {
        let alpha_rank = parent
            .areps
            .iter()
            .map(|&det| basis[det].excitation.alpha.holes.count_ones() as usize)
            .collect::<Vec<_>>();
        let beta_rank = parent
            .breps
            .iter()
            .map(|&det| basis[det].excitation.beta.holes.count_ones() as usize)
            .collect::<Vec<_>>();
        let mut sectors = BTreeMap::<(usize, usize), Vec<FactorEntry>>::new();

        for &entry in &parent.entries {
            sectors
                .entry((alpha_rank[entry.a], beta_rank[entry.b]))
                .or_default()
                .push(entry);
        }

        for ((arank, brank), entries) in sectors {
            let alpha_components = alpha_rank
                .iter()
                .enumerate()
                .filter_map(|(component, &rank)| (rank == arank).then_some(component))
                .collect::<Vec<_>>();
            let beta_components = beta_rank
                .iter()
                .enumerate()
                .filter_map(|(component, &rank)| (rank == brank).then_some(component))
                .collect::<Vec<_>>();
            let alpha_local = alpha_components
                .iter()
                .enumerate()
                .map(|(local, &component)| (component, local))
                .collect::<HashMap<_, _>>();
            let beta_local = beta_components
                .iter()
                .enumerate()
                .map(|(local, &component)| (component, local))
                .collect::<HashMap<_, _>>();
            let size = alpha_components
                .len()
                .checked_mul(beta_components.len())
                .expect("joint excitation-rank block size overflow");
            let mut slots = vec![None; size];
            let mut dense = entries.len() == size;

            for entry in entries {
                let ia = alpha_local[&entry.a];
                let ib = beta_local[&entry.b];
                let slot = ia * beta_components.len() + ib;
                if slots[slot].replace(entry.det).is_some() {
                    dense = false;
                }
            }
            dense &= slots.iter().all(Option::is_some);
            let dets = if dense {
                slots.into_iter().map(Option::unwrap).collect()
            } else {
                Vec::new()
            };

            parent.rank_blocks.push(ParentRankBlock {
                alpha_rank: arank,
                beta_rank: brank,
                alpha_components,
                beta_components,
                dets,
                dense,
            });
        }
    }
}
