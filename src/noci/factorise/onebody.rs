// noci/factorise/onebody.rs
//! Spin-factorised one-body NOCI operator contractions.

// Standard library imports.
use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;

// External crate imports.
use ndarray::Array1;
use rayon::prelude::*;

// Crate-root imports.
use crate::input::SNOCIStorage;
use crate::noci::fock::calculate_f_pair_orthogonal;
use crate::noci::overlap::calculate_s_pair_orthogonal;
use crate::noci::types::{FockData, FockMOCache, NOCIData, NOCIScalar};
use crate::nonorthogonalwicks::{WickScratchSpin, WicksPairView};
use crate::nonorthogonalwicks::{prepare_same, xw_f, xw_overlap};

// Parent/sibling imports.
use super::storage::{OneBodyFactorStorage, OneBodyStoragePlan};
use super::{ParentSpinSpace, SpinFactorisation, ordered_parent_pair};

/// Persistent one-body block for one ordered source-target parent pair `QP`.
enum OneBodyBlock<T: NOCIScalar> {
    /// Same-parent standard Slater-Condon sparse one-body action.
    Orthogonal(OrthogonalOneBodyBlock),
    /// Spin-factorised nonorthogonal one-body action.
    Factorised(FactorisedOneBodyBlock<T>),
    /// Spin-factorised nonorthogonal one-body action with regenerated factor tables.
    Transient(TransientOneBodyBlock),
}

/// Cached spin-factorised tables and dimensions for one ordered source-target parent pair `QP`.
struct FactorisedOneBodyBlock<T: NOCIScalar> {
    /// Target parent `Q`.
    target_parent: usize,
    /// Source parent `P`.
    source_parent: usize,
    /// Number of target alpha rows.
    nta: usize,
    /// Number of target beta rows.
    ntb: usize,
    /// Number of source alpha columns.
    nsa: usize,
    /// Number of source beta columns.
    nsb: usize,
    /// Selected dense contraction order for this parent pair.
    contraction: OneBodyContraction,
    /// Raw `S/F` alpha and beta factor backing.
    factors: OneBodyFactorStorage<T>,
}

/// Nonpersistent spin-factorised parent pair `QP`.
struct TransientOneBodyBlock {
    /// Target parent `Q`.
    target_parent: usize,
    /// Source parent `P`.
    source_parent: usize,
}

/// Actual determinant target in one orthogonal occupation group.
#[derive(Clone, Copy)]
struct OrthogonalTarget {
    /// Global determinant index `I`.
    det: usize,
    /// Parent-local alpha component `a_I`.
    a: usize,
}

/// Parent-local occupation group for the orthogonal one-body shortcut.
struct OrthogonalOccupationGroup {
    /// Determinants sharing one occupation pair.
    targets: Vec<OrthogonalTarget>,
}

/// Parent-local occupation lookup for same-parent orthogonal one-body application.
struct OrthogonalOneBodyBlock {
    /// Parent `P`.
    parent: usize,
    /// Occupation-pair ID keyed by `(o_alpha,o_beta)`.
    opos: HashMap<(u128, u128), usize>,
    /// Determinants grouped by occupation pair.
    groups: Vec<OrthogonalOccupationGroup>,
}

/// Shared data for same-parent orthogonal one-body application.
struct OrthogonalApplyContext<'a, T: NOCIScalar> {
    /// Shared NOCI data used to read determinant states.
    data: &'a NOCIData<'a, T>,
    /// MO-basis Fock cache for the parent.
    cache: &'a FockMOCache<T>,
    /// Scalar overlap shift in `F + \lambda S`.
    lambda: T,
    /// Worker index and worker count for target rows.
    partition: (usize, usize),
}

/// Reusable dense one-body contraction buffers.
pub(crate) struct OneBodyScratch<T: NOCIScalar> {
    /// Temporary `T^F_{\bar a b}` or `U^F_{a\bar b}` table.
    first_f: Vec<T>,
    /// Temporary `T^S_{\bar a b}` or `U^S_{a\bar b}` table.
    first_s: Vec<T>,
}

/// Dense one-body contraction order for one parent pair.
#[derive(Clone, Copy)]
enum OneBodyContraction {
    /// Form alpha-first intermediates `T^F_{\bar a b}` and `T^S_{\bar a b}`.
    AFirst,
    /// Form beta-first intermediates `U^F_{a\bar b}` and `U^S_{a\bar b}`.
    BFirst,
}

/// Cached spin-factorised one-body operator for the current generalised Fock.
pub(crate) struct OneBodyFactorisation<T: NOCIScalar> {
    /// Shared determinant-space factorisation `I <-> (P,a_I,b_I)`.
    spin: SpinFactorisation,
    /// Cached parent-pair factor blocks indexed as `Q * nparent + P`.
    blocks: Vec<OneBodyBlock<T>>,
    /// Number of parent references.
    nparent: usize,
}

impl<T: NOCIScalar> OneBodyFactorisation<T> {
    /// Build `F^{QP}_{\bar a\bar b,ab}` spin factors for the current generalised Fock operator.
    /// # Arguments:
    /// - `data`: Shared NOCI data with Wick intermediates for the candidate determinant basis.
    /// - `fock`: Current generalised-Fock data, already reflected in Wick intermediates.
    /// - `cache`: Directory for persistent file-backed factor blocks.
    /// - `rank`: MPI rank used in factor-cache filenames.
    /// - `iteration`: SNOCI iteration used in factor-cache filenames.
    /// - `storage`: Requested persistent factor-table storage backend.
    /// # Returns
    /// - `OneBodyFactorisation<T>`: Cached spin-factorised one-body operator.
    pub(crate) fn new(
        data: &NOCIData<'_, T>,
        fock: &FockData<'_, T>,
        cache: &Path,
        rank: i32,
        iteration: usize,
        storage: SNOCIStorage,
    ) -> Self {
        let spin = SpinFactorisation::new(data);
        let nparent = spin.parents.len();
        let mut blocks = Vec::with_capacity(nparent * nparent);
        let mut storage_plan = OneBodyStoragePlan::new(cache, rank, iteration, storage);

        for target_parent in 0..nparent {
            for source_parent in 0..nparent {
                if target_parent == source_parent
                    && fock.fock_mocache[target_parent].orthogonal_slater_condon
                {
                    blocks.push(OneBodyBlock::Orthogonal(build_orthogonal_one_body_block(
                        &spin,
                        data,
                        target_parent,
                    )));
                } else if matches!(storage, SNOCIStorage::None) {
                    blocks.push(OneBodyBlock::Transient(build_transient_one_body_block(
                        target_parent,
                        source_parent,
                    )));
                } else {
                    blocks.push(OneBodyBlock::Factorised(build_one_body_factor_tables(
                        &spin,
                        data,
                        &mut storage_plan,
                        target_parent,
                        source_parent,
                    )));
                }
            }
        }

        Self {
            spin,
            blocks,
            nparent,
        }
    }

    /// Construct reusable storage for dense one-body applications.
    /// # Arguments:
    /// - `self`: Cached one-body factorisation.
    /// # Returns
    /// - `OneBodyScratch<T>`: Empty reusable contraction buffers.
    pub(crate) fn scratch(&self) -> OneBodyScratch<T> {
        OneBodyScratch {
            first_f: Vec::new(),
            first_s: Vec::new(),
        }
    }

    /// Count raw factor storage bytes for `S^{alpha}`, `F^{alpha}`, `S^{beta}` and `F^{beta}`.
    /// Same-parent orthogonal Slater-Condon blocks require no dense factor storage.
    /// # Arguments:
    /// - `data`: Shared NOCI data defining the candidate determinant basis.
    /// - `fock`: Current generalised-Fock data used to identify orthogonal same-parent blocks.
    /// # Returns
    /// - `usize`: Number of bytes required to store all nonorthogonal raw factor tables.
    pub(crate) fn storage_bytes(
        data: &NOCIData<'_, T>,
        fock: &FockData<'_, T>,
    ) -> usize {
        let spin = SpinFactorisation::new(data);
        let nparent = spin.parents.len();
        let mut nentries = 0usize;
        for target_parent in 0..nparent {
            let target = &spin.parents[target_parent];
            for source_parent in 0..nparent {
                if target_parent == source_parent
                    && fock.fock_mocache[target_parent].orthogonal_slater_condon
                {
                    continue;
                }
                let source = &spin.parents[source_parent];
                let alpha = target
                    .areps
                    .len()
                    .checked_mul(source.areps.len())
                    .expect("alpha one-body factor length overflow");
                let beta = target
                    .breps
                    .len()
                    .checked_mul(source.breps.len())
                    .expect("beta one-body factor length overflow");
                let block_entries = 2usize
                    .checked_mul(
                        alpha
                            .checked_add(beta)
                            .expect("one-body factor length overflow"),
                    )
                    .expect("one-body factor length overflow");
                nentries = nentries
                    .checked_add(block_entries)
                    .expect("one-body factor total length overflow");
            }
        }
        nentries
            .checked_mul(std::mem::size_of::<T>())
            .expect("one-body factor byte size overflow")
    }

    /// Apply `Y = (F + \lambda S)x` using cached spin factors.
    /// # Arguments:
    /// - `x`: Source vector over actual candidate determinants.
    /// - `data`: Shared NOCI data used by same-parent orthogonal blocks.
    /// - `fock`: Current generalised-Fock data used by same-parent orthogonal blocks.
    /// - `lambda`: Scalar shift multiplying the overlap operator.
    /// - `scratch`: Reusable dense contraction buffers.
    /// - `partition`: Worker index and worker count for first-stage target rows.
    /// # Returns
    /// - `Array1<T>`: Partial or complete determinant-space result vector.
    pub(crate) fn apply_one_body(
        &self,
        x: &Array1<T>,
        data: &NOCIData<'_, T>,
        fock: &FockData<'_, T>,
        lambda: T,
        scratch: &mut OneBodyScratch<T>,
        partition: (usize, usize),
    ) -> Array1<T> {
        let zero = T::from_real(0.0);
        let mut y = vec![zero; x.len()];
        let xs = x
            .as_slice_memory_order()
            .expect("NOCI-PT2 vector must be contiguous.");
        let (worker, nworker) = partition;

        for block in &self.blocks {
            match block {
                OneBodyBlock::Orthogonal(block) => {
                    let context = OrthogonalApplyContext {
                        data,
                        cache: &fock.fock_mocache[block.parent],
                        lambda,
                        partition,
                    };
                    self.apply_one_body_orthogonal(block, xs, &mut y, &context)
                }
                OneBodyBlock::Factorised(block) => self.apply_one_body_factorised(
                    block,
                    xs,
                    &mut y,
                    lambda,
                    scratch,
                    (worker, nworker),
                ),
                OneBodyBlock::Transient(block) => {
                    let mut storage_plan =
                        OneBodyStoragePlan::new(Path::new("."), 0, 0, SNOCIStorage::RAM);
                    let block = build_one_body_factor_tables(
                        &self.spin,
                        data,
                        &mut storage_plan,
                        block.target_parent,
                        block.source_parent,
                    );
                    self.apply_one_body_factorised(
                        &block,
                        xs,
                        &mut y,
                        lambda,
                        scratch,
                        (worker, nworker),
                    )
                }
            }
        }

        Array1::from_vec(y)
    }

    /// Build diagonal entries of `F + \lambda S` and `S` from cached same-spin factors.
    /// # Arguments:
    /// - `data`: Shared NOCI data used by same-parent orthogonal blocks.
    /// - `fock`: Current generalised-Fock data used by same-parent orthogonal blocks.
    /// - `lambda`: Scalar overlap shift in `F + \lambda S`.
    /// # Returns
    /// - `(Array1<T>, Array1<T>)`: Diagonal of `F + \lambda S` and diagonal of `S`.
    pub(crate) fn one_body_diagonals(
        &self,
        data: &NOCIData<'_, T>,
        fock: &FockData<'_, T>,
        lambda: T,
    ) -> (Array1<T>, Array1<T>) {
        let zero = T::from_real(0.0);
        let ndet = self.spin.aids.len();
        let mut m_diag = vec![zero; ndet];
        let mut s_diag = vec![zero; ndet];

        for (parent_id, parent) in self.spin.parents.iter().enumerate() {
            if parent.entries.is_empty() {
                continue;
            }
            match self.block(parent_id, parent_id) {
                OneBodyBlock::Orthogonal(_) => {
                    fill_orthogonal_one_body_diagonal_block(
                        parent,
                        data,
                        &fock.fock_mocache[parent_id],
                        lambda,
                        &mut m_diag,
                        &mut s_diag,
                    );
                }
                OneBodyBlock::Factorised(block) => {
                    fill_one_body_diagonal_block(parent, block, lambda, &mut m_diag, &mut s_diag);
                }
                OneBodyBlock::Transient(block) => {
                    let mut storage_plan =
                        OneBodyStoragePlan::new(Path::new("."), 0, 0, SNOCIStorage::RAM);
                    let block = build_one_body_factor_tables(
                        &self.spin,
                        data,
                        &mut storage_plan,
                        block.target_parent,
                        block.source_parent,
                    );
                    fill_one_body_diagonal_block(parent, &block, lambda, &mut m_diag, &mut s_diag);
                }
            }
        }

        (Array1::from_vec(m_diag), Array1::from_vec(s_diag))
    }

    /// Return cached factor block for target parent `Q` and source parent `P`.
    /// # Arguments:
    /// - `target_parent`: Target parent `Q`.
    /// - `source_parent`: Source parent `P`.
    /// # Returns
    /// - `&OneBodyBlock<T>`: Cached parent-pair block.
    fn block(
        &self,
        target_parent: usize,
        source_parent: usize,
    ) -> &OneBodyBlock<T> {
        &self.blocks[target_parent * self.nparent + source_parent]
    }

    /// Apply one cached spin-factorised parent-pair block of `F + \lambda S`.
    /// # Arguments:
    /// - `block`: Cached spin-factorised one-body factors.
    /// - `x`: Source determinant vector.
    /// - `y`: Output determinant vector to accumulate.
    /// - `lambda`: Scalar overlap shift.
    /// - `scratch`: Reusable dense contraction buffers.
    /// - `partition`: Worker index and worker count for target rows.
    /// # Returns
    /// - `()`: Adds this parent-pair contribution into `y`.
    fn apply_one_body_factorised(
        &self,
        block: &FactorisedOneBodyBlock<T>,
        x: &[T],
        y: &mut [T],
        lambda: T,
        scratch: &mut OneBodyScratch<T>,
        partition: (usize, usize),
    ) {
        match block.contraction {
            OneBodyContraction::AFirst => {
                self.apply_one_body_a_first(block, x, y, lambda, scratch, partition)
            }
            OneBodyContraction::BFirst => {
                self.apply_one_body_b_first(block, x, y, lambda, scratch, partition)
            }
        }
    }

    /// Apply same-parent orthogonal `Y^P += (F^{PP}+\lambda S^{PP})D^P`.
    /// # Arguments:
    /// - `block`: Same-parent orthogonal action data.
    /// - `x`: Source determinant vector.
    /// - `y`: Output determinant vector to accumulate.
    /// - `context`: Shared determinant, Fock, shift and worker data.
    /// # Returns
    /// - `()`: Adds this same-parent contribution into `y`.
    fn apply_one_body_orthogonal(
        &self,
        block: &OrthogonalOneBodyBlock,
        x: &[T],
        y: &mut [T],
        context: &OrthogonalApplyContext<'_, T>,
    ) {
        let zero = T::from_real(0.0);
        let source = &self.spin.parents[block.parent];

        for entry in &source.entries {
            let xe = x[entry.det];
            if xe == zero {
                continue;
            }
            let sdet = &context.data.basis[entry.det];
            let oid = source.oids[entry.det - source.first_det];
            scatter_orthogonal_group(&block.groups[oid], entry.det, xe, y, context);

            apply_orthogonal_alpha_singles(block, entry.det, sdet.oa, sdet.ob, xe, y, context);
            apply_orthogonal_beta_singles(block, entry.det, sdet.oa, sdet.ob, xe, y, context);
        }
    }

    /// Apply alpha-first contraction for `Y^Q += F^alpha D (S^beta)^T + S^alpha D (F^beta+\lambda S^beta)^T`.
    /// # Arguments:
    /// - `block`: Cached parent-pair one-body factors.
    /// - `x`: Source determinant vector.
    /// - `y`: Output determinant vector to accumulate.
    /// - `lambda`: Scalar overlap shift.
    /// - `scratch`: Reusable dense contraction buffers.
    /// - `partition`: Worker index and worker count for target alpha rows.
    /// # Returns
    /// - `()`: Adds this parent-pair contribution into `y`.
    fn apply_one_body_a_first(
        &self,
        block: &FactorisedOneBodyBlock<T>,
        x: &[T],
        y: &mut [T],
        lambda: T,
        scratch: &mut OneBodyScratch<T>,
        partition: (usize, usize),
    ) {
        let zero = T::from_real(0.0);
        let source = &self.spin.parents[block.source_parent];
        let target = &self.spin.parents[block.target_parent];
        let (worker, nworker) = partition;
        let (sa, fa, sb, fb) = block.factors.factors();

        for a0 in (0..block.nta).step_by(512) {
            let a1 = (a0 + 512).min(block.nta);
            let nrow = a1 - a0;

            scratch.first_f.clear();
            scratch.first_s.clear();
            scratch.first_f.resize(nrow * block.nsb, zero);
            scratch.first_s.resize(nrow * block.nsb, zero);

            for ta in (a0..a1).filter(|ta| ta % nworker == worker) {
                let row = ta - a0;
                let frow = &fa[ta * block.nsa..(ta + 1) * block.nsa];
                let srow = &sa[ta * block.nsa..(ta + 1) * block.nsa];
                let tf = &mut scratch.first_f[row * block.nsb..(row + 1) * block.nsb];
                let ts = &mut scratch.first_s[row * block.nsb..(row + 1) * block.nsb];
                for entry in &source.entries {
                    let xe = x[entry.det];
                    if xe != zero {
                        tf[entry.b] += frow[entry.a] * xe;
                        ts[entry.b] += srow[entry.a] * xe;
                    }
                }
            }

            for entry in &target.entries {
                if entry.a < a0 || entry.a >= a1 || entry.a % nworker != worker {
                    continue;
                }
                let row = entry.a - a0;
                let tf = &scratch.first_f[row * block.nsb..(row + 1) * block.nsb];
                let ts = &scratch.first_s[row * block.nsb..(row + 1) * block.nsb];
                let sbrow = &sb[entry.b * block.nsb..(entry.b + 1) * block.nsb];
                let fbrow = &fb[entry.b * block.nsb..(entry.b + 1) * block.nsb];
                let mut value = zero;
                for b in 0..block.nsb {
                    value += tf[b] * sbrow[b] + ts[b] * (fbrow[b] + lambda * sbrow[b]);
                }
                y[entry.det] += value;
            }
        }
    }

    /// Apply beta-first contraction for `Y^Q += S^alpha D (F^beta)^T + (F^alpha+\lambda S^alpha)D(S^beta)^T`.
    /// # Arguments:
    /// - `block`: Cached parent-pair one-body factors.
    /// - `x`: Source determinant vector.
    /// - `y`: Output determinant vector to accumulate.
    /// - `lambda`: Scalar overlap shift.
    /// - `scratch`: Reusable dense contraction buffers.
    /// - `partition`: Worker index and worker count for target beta rows.
    /// # Returns
    /// - `()`: Adds this parent-pair contribution into `y`.
    fn apply_one_body_b_first(
        &self,
        block: &FactorisedOneBodyBlock<T>,
        x: &[T],
        y: &mut [T],
        lambda: T,
        scratch: &mut OneBodyScratch<T>,
        partition: (usize, usize),
    ) {
        let zero = T::from_real(0.0);
        let source = &self.spin.parents[block.source_parent];
        let target = &self.spin.parents[block.target_parent];
        let (worker, nworker) = partition;
        let (sa, fa, sb, fb) = block.factors.factors();

        for b0 in (0..block.ntb).step_by(512) {
            let b1 = (b0 + 512).min(block.ntb);
            let nrow = b1 - b0;

            scratch.first_f.clear();
            scratch.first_s.clear();
            scratch.first_f.resize(nrow * block.nsa, zero);
            scratch.first_s.resize(nrow * block.nsa, zero);

            for tb in (b0..b1).filter(|tb| tb % nworker == worker) {
                let row = tb - b0;
                let frow = &fb[tb * block.nsb..(tb + 1) * block.nsb];
                let srow = &sb[tb * block.nsb..(tb + 1) * block.nsb];
                let uf = &mut scratch.first_f[row * block.nsa..(row + 1) * block.nsa];
                let us = &mut scratch.first_s[row * block.nsa..(row + 1) * block.nsa];
                for entry in &source.entries {
                    let xe = x[entry.det];
                    if xe != zero {
                        uf[entry.a] += xe * frow[entry.b];
                        us[entry.a] += xe * srow[entry.b];
                    }
                }
            }

            for entry in &target.entries {
                if entry.b < b0 || entry.b >= b1 || entry.b % nworker != worker {
                    continue;
                }
                let row = entry.b - b0;
                let uf = &scratch.first_f[row * block.nsa..(row + 1) * block.nsa];
                let us = &scratch.first_s[row * block.nsa..(row + 1) * block.nsa];
                let sarow = &sa[entry.a * block.nsa..(entry.a + 1) * block.nsa];
                let farow = &fa[entry.a * block.nsa..(entry.a + 1) * block.nsa];
                let mut value = zero;
                for a in 0..block.nsa {
                    value += sarow[a] * uf[a] + (farow[a] + lambda * sarow[a]) * us[a];
                }
                y[entry.det] += value;
            }
        }
    }
}

/// Build `S^alpha`, `F^alpha`, `S^beta` and `F^beta` tables for one parent pair `QP`.
/// # Arguments:
/// - `spin`: Shared determinant-space factorisation.
/// - `data`: Shared NOCI data containing Wick intermediates.
/// - `target_parent`: Target parent `Q`.
/// - `source_parent`: Source parent `P`.
/// # Returns
/// - `FactorisedOneBodyBlock<T>`: Cached row-major factor tables for this parent pair.
fn build_one_body_factor_tables<T: NOCIScalar>(
    spin: &SpinFactorisation,
    data: &NOCIData<'_, T>,
    storage_plan: &mut OneBodyStoragePlan,
    target_parent: usize,
    source_parent: usize,
) -> FactorisedOneBodyBlock<T> {
    let target = &spin.parents[target_parent];
    let source = &spin.parents[source_parent];
    let nta = target.areps.len();
    let ntb = target.breps.len();
    let nsa = source.areps.len();
    let nsb = source.breps.len();
    let (lp, gp, target_left) = ordered_parent_pair(spin, target_parent, source_parent);
    let pair = data
        .wicks
        .expect("factorised one-body requires Wick intermediates")
        .pair(lp, gp);

    let na = nta
        .checked_mul(nsa)
        .expect("alpha one-body factor length overflow");
    let nb = ntb
        .checked_mul(nsb)
        .expect("beta one-body factor length overflow");
    let mut factors = storage_plan.allocate::<T>(target_parent, source_parent, na, nb);

    for row0 in (0..nta).step_by(512) {
        let row1 = (row0 + 512).min(nta);
        let out = factors.alpha_mut();
        build_spin_one_body_factors(
            &pair,
            data,
            (target.areps.as_slice(), source.areps.as_slice()),
            row0..row1,
            target_left,
            true,
            out,
        );
    }
    factors.flush();

    for row0 in (0..ntb).step_by(512) {
        let row1 = (row0 + 512).min(ntb);
        let out = factors.beta_mut();
        build_spin_one_body_factors(
            &pair,
            data,
            (target.breps.as_slice(), source.breps.as_slice()),
            row0..row1,
            target_left,
            false,
            out,
        );
    }
    factors.flush();

    let contraction = select_one_body_contraction(
        nta,
        ntb,
        nsa,
        nsb,
        target.entries.len(),
        source.entries.len(),
    );

    FactorisedOneBodyBlock {
        target_parent,
        source_parent,
        nta,
        ntb,
        nsa,
        nsb,
        contraction,
        factors,
    }
}

/// Build nonpersistent one-body factor metadata for parent pair `QP`.
/// # Arguments:
/// - `target_parent`: Target parent `Q`.
/// - `source_parent`: Source parent `P`.
/// # Returns
/// - `TransientOneBodyBlock`: Parent-pair metadata for regenerated factor tables.
fn build_transient_one_body_block(
    target_parent: usize,
    source_parent: usize,
) -> TransientOneBodyBlock {
    TransientOneBodyBlock {
        target_parent,
        source_parent,
    }
}

/// Build same-parent orthogonal one-body action metadata for parent `P`.
/// # Arguments:
/// - `spin`: Shared determinant-space factorisation.
/// - `data`: Shared NOCI determinant data.
/// - `parent_id`: Parent `P`.
/// # Returns
/// - `OrthogonalOneBodyBlock`: Orthogonal same-parent block with occupation lookup data.
fn build_orthogonal_one_body_block<T: NOCIScalar>(
    spin: &SpinFactorisation,
    data: &NOCIData<'_, T>,
    parent_id: usize,
) -> OrthogonalOneBodyBlock {
    let parent = &spin.parents[parent_id];
    let mut opos = HashMap::new();
    let mut groups = Vec::with_capacity(parent.oreps.len());

    for &det in &parent.oreps {
        let state = &data.basis[det];
        opos.insert((state.oa, state.ob), groups.len());
        groups.push(OrthogonalOccupationGroup {
            targets: Vec::new(),
        });
    }

    for entry in &parent.entries {
        let oid = parent.oids[entry.det - parent.first_det];
        groups[oid].targets.push(OrthogonalTarget {
            det: entry.det,
            a: entry.a,
        });
    }

    OrthogonalOneBodyBlock {
        parent: parent_id,
        opos,
        groups,
    }
}

/// Fill determinant diagonals from one same-parent factor block.
/// # Arguments:
/// - `parent`: Parent-local actual determinant entries.
/// - `block`: Same-parent factor block.
/// - `lambda`: Scalar overlap shift in `F + \lambda S`.
/// - `m_diag`: Output diagonal of `F + \lambda S`.
/// - `s_diag`: Output diagonal of `S`.
/// # Returns
/// - `()`: Writes diagonal values for actual determinants.
fn fill_one_body_diagonal_block<T: NOCIScalar>(
    parent: &ParentSpinSpace,
    block: &FactorisedOneBodyBlock<T>,
    lambda: T,
    m_diag: &mut [T],
    s_diag: &mut [T],
) {
    let (sa, fa, sb, fb) = block.factors.factors();
    for entry in &parent.entries {
        let saa = sa[entry.a * block.nsa + entry.a];
        let faa = fa[entry.a * block.nsa + entry.a];
        let sbb = sb[entry.b * block.nsb + entry.b];
        let fbb = fb[entry.b * block.nsb + entry.b];
        let s = saa * sbb;
        s_diag[entry.det] = s;
        m_diag[entry.det] = faa * sbb + saa * fbb + lambda * s;
    }
}

/// Fill same-parent orthogonal diagonals from parent-local Slater-Condon rules.
/// # Arguments:
/// - `parent`: Parent-local actual determinant entries.
/// - `data`: Shared NOCI determinant data.
/// - `cache`: MO-basis Fock cache for the parent.
/// - `lambda`: Scalar overlap shift in `F + \lambda S`.
/// - `m_diag`: Output diagonal of `F + \lambda S`.
/// - `s_diag`: Output diagonal of `S`.
/// # Returns
/// - `()`: Writes diagonal values for actual determinants.
fn fill_orthogonal_one_body_diagonal_block<T: NOCIScalar>(
    parent: &ParentSpinSpace,
    data: &NOCIData<'_, T>,
    cache: &FockMOCache<T>,
    lambda: T,
    m_diag: &mut [T],
    s_diag: &mut [T],
) {
    for entry in &parent.entries {
        let det = &data.basis[entry.det];
        let s = calculate_s_pair_orthogonal(det, det);
        s_diag[entry.det] = s;
        m_diag[entry.det] = calculate_f_pair_orthogonal(cache, det, det) + lambda * s;
    }
}

/// Scatter one same-occupation orthogonal contribution to assigned target rows.
/// # Arguments:
/// - `group`: Target determinants with the same occupation pair.
/// - `source_det`: Source determinant index.
/// - `xe`: Source vector coefficient.
/// - `y`: Output determinant vector to accumulate.
/// - `context`: Shared determinant, Fock, shift and worker data.
/// # Returns
/// - `()`: Adds same-occupation contributions into `y`.
fn scatter_orthogonal_group<T: NOCIScalar>(
    group: &OrthogonalOccupationGroup,
    source_det: usize,
    xe: T,
    y: &mut [T],
    context: &OrthogonalApplyContext<'_, T>,
) {
    let source = &context.data.basis[source_det];
    let (worker, nworker) = context.partition;
    for target in &group.targets {
        if target.a % nworker == worker {
            let target_det = &context.data.basis[target.det];
            let f = calculate_f_pair_orthogonal(context.cache, target_det, source);
            let s = calculate_s_pair_orthogonal(target_det, source);
            y[target.det] += (f + context.lambda * s) * xe;
        }
    }
}

/// Apply all alpha single-excitation orthogonal Fock couplings from one source determinant.
/// # Arguments:
/// - `orthogonal`: Parent-local occupation lookup.
/// - `source_det`: Source determinant index.
/// - `oa`: Source alpha occupation bitstring.
/// - `ob`: Source beta occupation bitstring.
/// - `xe`: Source vector coefficient.
/// - `y`: Output determinant vector to accumulate.
/// - `context`: Shared determinant, Fock and worker data.
/// # Returns
/// - `()`: Adds alpha single-excitation Fock contributions into `y`.
fn apply_orthogonal_alpha_singles<T: NOCIScalar>(
    orthogonal: &OrthogonalOneBodyBlock,
    source_det: usize,
    oa: u128,
    ob: u128,
    xe: T,
    y: &mut [T],
    context: &OrthogonalApplyContext<'_, T>,
) {
    let (worker, nworker) = context.partition;
    let source = &context.data.basis[source_det];
    let nmo = context.cache.fa.nrows();
    let mut holes = oa;
    while holes != 0 {
        let hole = holes.trailing_zeros() as usize;
        holes &= holes - 1;
        for part in 0..nmo {
            if ((oa >> part) & 1) == 1 {
                continue;
            }
            let target_oa = (oa & !(1u128 << hole)) | (1u128 << part);
            let Some(&opos) = orthogonal.opos.get(&(target_oa, ob)) else {
                continue;
            };
            for target in &orthogonal.groups[opos].targets {
                if target.a % nworker == worker {
                    let target_det = &context.data.basis[target.det];
                    y[target.det] +=
                        calculate_f_pair_orthogonal(context.cache, target_det, source) * xe;
                }
            }
        }
    }
}

/// Apply all beta single-excitation orthogonal Fock couplings from one source determinant.
/// # Arguments:
/// - `orthogonal`: Parent-local occupation lookup.
/// - `source_det`: Source determinant index.
/// - `oa`: Source alpha occupation bitstring.
/// - `ob`: Source beta occupation bitstring.
/// - `xe`: Source vector coefficient.
/// - `y`: Output determinant vector to accumulate.
/// - `context`: Shared determinant, Fock and worker data.
/// # Returns
/// - `()`: Adds beta single-excitation Fock contributions into `y`.
fn apply_orthogonal_beta_singles<T: NOCIScalar>(
    orthogonal: &OrthogonalOneBodyBlock,
    source_det: usize,
    oa: u128,
    ob: u128,
    xe: T,
    y: &mut [T],
    context: &OrthogonalApplyContext<'_, T>,
) {
    let (worker, nworker) = context.partition;
    let source = &context.data.basis[source_det];
    let nmo = context.cache.fb.nrows();
    let mut holes = ob;
    while holes != 0 {
        let hole = holes.trailing_zeros() as usize;
        holes &= holes - 1;
        for part in 0..nmo {
            if ((ob >> part) & 1) == 1 {
                continue;
            }
            let target_ob = (ob & !(1u128 << hole)) | (1u128 << part);
            let Some(&opos) = orthogonal.opos.get(&(oa, target_ob)) else {
                continue;
            };
            for target in &orthogonal.groups[opos].targets {
                if target.a % nworker == worker {
                    let target_det = &context.data.basis[target.det];
                    y[target.det] +=
                        calculate_f_pair_orthogonal(context.cache, target_det, source) * xe;
                }
            }
        }
    }
}

/// Build same-spin `S` and `F` factor rows from one prepared Wick scratch per component pair.
/// # Arguments:
/// - `pair`: Wick intermediates for the ordered parent pair.
/// - `data`: Shared NOCI determinant data.
/// - `reps`: Representative determinants for target and source spin components.
/// - `rows`: Target-row range to fill.
/// - `target_left`: Whether target determinants are left determinants in `pair`.
/// - `alpha`: Whether to build alpha or beta factors.
/// - `out`: Mutable row-major overlap and Fock factor tables.
/// # Returns
/// - `()`: Fills `out` factor tables.
fn build_spin_one_body_factors<T: NOCIScalar>(
    pair: &WicksPairView<'_, T>,
    data: &NOCIData<'_, T>,
    reps: (&[usize], &[usize]),
    rows: Range<usize>,
    target_left: bool,
    alpha: bool,
    out: (&mut [T], &mut [T]),
) {
    let (target_reps, source_reps) = reps;
    let nsource = source_reps.len();
    let tol = data.tol;
    let row0 = rows.start;
    let row1 = rows.end;
    out.0[row0 * nsource..row1 * nsource]
        .par_chunks_mut(nsource)
        .zip(out.1[row0 * nsource..row1 * nsource].par_chunks_mut(nsource))
        .zip(target_reps[row0..row1].par_iter())
        .for_each_init(WickScratchSpin::new, |scratch, ((srow, frow), &tdet)| {
            for (col, &sdet) in source_reps.iter().enumerate() {
                let (ldet, gdet) = if target_left {
                    (&data.basis[tdet], &data.basis[sdet])
                } else {
                    (&data.basis[sdet], &data.basis[tdet])
                };
                if alpha {
                    let lex = &ldet.excitation.alpha;
                    let gex = &gdet.excitation.alpha;
                    let phase = T::from_real(ldet.pha * gdet.pha);
                    prepare_same(&pair.aa, lex, gex, &mut scratch.aa);
                    srow[col] = phase * xw_overlap(&pair.aa, lex, gex, &mut scratch.aa);
                    frow[col] = phase * xw_f(&pair.aa, lex, gex, &mut scratch.aa, tol);
                } else {
                    let lex = &ldet.excitation.beta;
                    let gex = &gdet.excitation.beta;
                    let phase = T::from_real(ldet.phb * gdet.phb);
                    prepare_same(&pair.bb, lex, gex, &mut scratch.bb);
                    srow[col] = phase * xw_overlap(&pair.bb, lex, gex, &mut scratch.bb);
                    frow[col] = phase * xw_f(&pair.bb, lex, gex, &mut scratch.bb, tol);
                }
            }
        });
}

/// Select alpha-first or beta-first contraction from dense structural costs.
/// # Arguments:
/// - `nta`: Number of target alpha components.
/// - `ntb`: Number of target beta components.
/// - `nsa`: Number of source alpha components.
/// - `nsb`: Number of source beta components.
/// - `nt`: Number of actual target determinants.
/// - `ns`: Number of actual source determinants.
/// # Returns
/// - `OneBodyContraction`: Lower estimated-cost contraction.
fn select_one_body_contraction(
    nta: usize,
    ntb: usize,
    nsa: usize,
    nsb: usize,
    nt: usize,
    ns: usize,
) -> OneBodyContraction {
    let ca = 2usize
        .saturating_mul(nta)
        .saturating_mul(ns)
        .saturating_add(2usize.saturating_mul(nt).saturating_mul(nsb));
    let cb = 2usize
        .saturating_mul(ntb)
        .saturating_mul(ns)
        .saturating_add(2usize.saturating_mul(nt).saturating_mul(nsa));
    if ca <= cb {
        OneBodyContraction::AFirst
    } else {
        OneBodyContraction::BFirst
    }
}
