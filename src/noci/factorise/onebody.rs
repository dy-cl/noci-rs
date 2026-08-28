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
use crate::nonorthogonalwicks::{
    SameSpinOneBodyBatch, WickScratchSpin, WicksPairView, xw_f_overlap_prepared_batched,
};
use crate::{DetState, ReducedOneSpinDetState};

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
    /// Selected dense contraction order for this parent pair.
    contraction: OneBodyContraction,
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
                    let target = &spin.parents[target_parent];
                    let source = &spin.parents[source_parent];
                    let contraction = select_one_body_contraction(
                        target.areps.len(),
                        target.breps.len(),
                        source.areps.len(),
                        source.breps.len(),
                        target.entries.len(),
                        source.entries.len(),
                    );

                    blocks.push(OneBodyBlock::Transient(TransientOneBodyBlock {
                        target_parent,
                        source_parent,
                        contraction,
                    }));
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

    /// Construct reusable storage for a one-body application in scalar `R`.
    /// # Arguments:
    /// - `self`: Cached one-body factorisation.
    /// # Returns:
    /// - `OneBodyScratch<R>`: Empty reusable contraction buffers.
    pub(crate) fn scratch<R: NOCIScalar>(&self) -> OneBodyScratch<R> {
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
    /// - `scratch`: Reusable dense contraction buffers in scalar `R`.
    /// - `partition`: Worker index and worker count for first-stage target rows.
    /// # Returns:
    /// - `Array1<R>`: Partial or complete determinant-space result vector.
    pub(crate) fn apply_one_body<R>(
        &self,
        x: &Array1<R>,
        data: &NOCIData<'_, T>,
        fock: &FockData<'_, T>,
        lambda: R,
        scratch: &mut OneBodyScratch<R>,
        partition: (usize, usize),
    ) -> Array1<R>
    where
        R: NOCIScalar + From<T>,
    {
        let zero = R::from_real(0.0);
        let mut y = vec![zero; x.len()];

        let xs = x
            .as_slice_memory_order()
            .expect("NOCI-PT2 vector must be contiguous.");

        let (worker, nworker) = partition;

        // `Y = (F + lambda S)x`: `F` and `S` use chemistry scalar `T`,
        // while `x`, `lambda` and `Y` use Krylov scalar `R`.
        for block in &self.blocks {
            match block {
                OneBodyBlock::Orthogonal(block) => self.apply_one_body_orthogonal(
                    block,
                    (xs, &mut y),
                    data.basis,
                    &fock.fock_mocache[block.parent],
                    lambda,
                    partition,
                ),
                OneBodyBlock::Factorised(block) => self.apply_one_body_factorised(
                    block,
                    xs,
                    &mut y,
                    lambda,
                    scratch,
                    (worker, nworker),
                ),
                OneBodyBlock::Transient(block) => match block.contraction {
                    OneBodyContraction::AFirst => self.apply_one_body_transient_a_first(
                        block,
                        (xs, &mut y),
                        data,
                        lambda,
                        scratch,
                        (worker, nworker),
                    ),
                    OneBodyContraction::BFirst => self.apply_one_body_transient_b_first(
                        block,
                        (xs, &mut y),
                        data,
                        lambda,
                        scratch,
                        (worker, nworker),
                    ),
                },
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
    /// - `scratch`: Reusable dense contraction buffers in scalar `R`.
    /// - `partition`: Worker index and worker count for target rows.
    /// # Returns
    /// - `()`: Adds this parent-pair contribution into `y`.
    fn apply_one_body_factorised<R>(
        &self,
        block: &FactorisedOneBodyBlock<T>,
        x: &[R],
        y: &mut [R],
        lambda: R,
        scratch: &mut OneBodyScratch<R>,
        partition: (usize, usize),
    ) where
        R: NOCIScalar + From<T>,
    {
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
    /// - `vectors`: Source determinant vector and output determinant vector to accumulate.
    /// - `basis`: Candidate determinant basis.
    /// - `cache`: MO-basis Fock cache for the parent.
    /// - `lambda`: Scalar overlap shift.
    /// - `partition`: Worker index and worker count for target rows.
    /// # Returns
    /// - `()`: Adds this same-parent contribution into `y`.
    fn apply_one_body_orthogonal<R>(
        &self,
        block: &OrthogonalOneBodyBlock,
        vectors: (&[R], &mut [R]),
        basis: &[DetState<T>],
        cache: &FockMOCache<T>,
        lambda: R,
        partition: (usize, usize),
    ) where
        R: NOCIScalar + From<T>,
    {
        let (x, y) = vectors;
        let zero = R::from_real(0.0);
        let source_parent = &self.spin.parents[block.parent];
        let (worker, nworker) = partition;

        for entry in &source_parent.entries {
            let xe = x[entry.det];
            if xe == zero {
                continue;
            }
            let source = &basis[entry.det];
            let oid = source_parent.oids[entry.det - source_parent.first_det];
            let group = &block.groups[oid];
            for target in &group.targets {
                if target.a % nworker == worker {
                    let target_det = &basis[target.det];
                    let f = <R as From<T>>::from(calculate_f_pair_orthogonal(
                        cache, target_det, source,
                    ));
                    let s = <R as From<T>>::from(calculate_s_pair_orthogonal(target_det, source));
                    y[target.det] += (f + lambda * s) * xe;
                }
            }

            apply_orthogonal_alpha_singles(block, source, xe, y, basis, cache, partition);
            apply_orthogonal_beta_singles(block, source, xe, y, basis, cache, partition);
        }
    }

    /// Apply alpha-first contraction for
    /// `Y^Q += F^alpha D (S^beta)^T + S^alpha D (F^beta+\lambda S^beta)^T`.
    /// # Arguments:
    /// - `block`: Cached parent-pair one-body factors.
    /// - `x`: Source determinant vector.
    /// - `y`: Output determinant vector to accumulate.
    /// - `lambda`: Scalar overlap shift.
    /// - `scratch`: Reusable dense contraction buffers in scalar `R`.
    /// - `partition`: Worker index and worker count for target alpha rows.
    /// # Returns
    /// - `()`: Adds this parent-pair contribution into `y`.
    fn apply_one_body_a_first<R>(
        &self,
        block: &FactorisedOneBodyBlock<T>,
        x: &[R],
        y: &mut [R],
        lambda: R,
        scratch: &mut OneBodyScratch<R>,
        partition: (usize, usize),
    ) where
        R: NOCIScalar + From<T>,
    {
        let zero = R::from_real(0.0);
        let source = &self.spin.parents[block.source_parent];
        let target = &self.spin.parents[block.target_parent];
        let (worker, nworker) = partition;
        let (sa, fa, sb, fb) = block.factors.factors();

        scratch.first_f.clear();
        scratch.first_s.clear();
        scratch.first_f.resize(block.nta * block.nsb, zero);
        scratch.first_s.resize(block.nta * block.nsb, zero);

        scratch
            .first_f
            .par_chunks_mut(block.nsb)
            .zip(scratch.first_s.par_chunks_mut(block.nsb))
            .enumerate()
            .for_each(|(ta, (tf, ts))| {
                if ta % nworker != worker {
                    return;
                }

                let frow = &fa[ta * block.nsa..(ta + 1) * block.nsa];
                let srow = &sa[ta * block.nsa..(ta + 1) * block.nsa];

                // `T^F_{\bar a b} = \sum_a F^\alpha_{\bar a a} D_{ab}`,
                // `T^S_{\bar a b} = \sum_a S^\alpha_{\bar a a} D_{ab}`.
                for entry in &source.entries {
                    let xe = x[entry.det];

                    if xe != zero {
                        tf[entry.b] += <R as From<T>>::from(frow[entry.a]) * xe;
                        ts[entry.b] += <R as From<T>>::from(srow[entry.a]) * xe;
                    }
                }
            });

        let updates: Vec<(usize, R)> = target
            .entries
            .par_iter()
            .filter(|entry| entry.a % nworker == worker)
            .map(|entry| {
                let tf = &scratch.first_f[entry.a * block.nsb..(entry.a + 1) * block.nsb];
                let ts = &scratch.first_s[entry.a * block.nsb..(entry.a + 1) * block.nsb];
                let sbrow = &sb[entry.b * block.nsb..(entry.b + 1) * block.nsb];
                let fbrow = &fb[entry.b * block.nsb..(entry.b + 1) * block.nsb];

                let mut value = zero;

                // `Y^Q = T^F (S^\beta)^T + T^S (F^\beta + lambda S^\beta)^T`.
                for b in 0..block.nsb {
                    let s = <R as From<T>>::from(sbrow[b]);
                    let f = <R as From<T>>::from(fbrow[b]);
                    value += tf[b] * s + ts[b] * (f + lambda * s);
                }

                (entry.det, value)
            })
            .collect();

        for (det, value) in updates {
            y[det] += value;
        }
    }

    /// Apply beta-first contraction for
    /// `Y^Q += S^alpha D (F^beta)^T + (F^alpha+\lambda S^alpha)D(S^beta)^T`.
    /// # Arguments:
    /// - `block`: Cached parent-pair one-body factors.
    /// - `x`: Source determinant vector.
    /// - `y`: Output determinant vector to accumulate.
    /// - `lambda`: Scalar overlap shift.
    /// - `scratch`: Reusable dense contraction buffers in scalar `R`.
    /// - `partition`: Worker index and worker count for target beta rows.
    /// # Returns
    /// - `()`: Adds this parent-pair contribution into `y`.
    fn apply_one_body_b_first<R>(
        &self,
        block: &FactorisedOneBodyBlock<T>,
        x: &[R],
        y: &mut [R],
        lambda: R,
        scratch: &mut OneBodyScratch<R>,
        partition: (usize, usize),
    ) where
        R: NOCIScalar + From<T>,
    {
        let zero = R::from_real(0.0);
        let source = &self.spin.parents[block.source_parent];
        let target = &self.spin.parents[block.target_parent];
        let (worker, nworker) = partition;
        let (sa, fa, sb, fb) = block.factors.factors();

        scratch.first_f.clear();
        scratch.first_s.clear();
        scratch.first_f.resize(block.ntb * block.nsa, zero);
        scratch.first_s.resize(block.ntb * block.nsa, zero);

        scratch
            .first_f
            .par_chunks_mut(block.nsa)
            .zip(scratch.first_s.par_chunks_mut(block.nsa))
            .enumerate()
            .for_each(|(tb, (uf, us))| {
                if tb % nworker != worker {
                    return;
                }

                let frow = &fb[tb * block.nsb..(tb + 1) * block.nsb];
                let srow = &sb[tb * block.nsb..(tb + 1) * block.nsb];

                // `U^F_{a\bar b} = \sum_b D_{ab} F^\beta_{\bar b b}`,
                // `U^S_{a\bar b} = \sum_b D_{ab} S^\beta_{\bar b b}`.
                for entry in &source.entries {
                    let xe = x[entry.det];

                    if xe != zero {
                        uf[entry.a] += xe * <R as From<T>>::from(frow[entry.b]);
                        us[entry.a] += xe * <R as From<T>>::from(srow[entry.b]);
                    }
                }
            });

        let updates: Vec<(usize, R)> = target
            .entries
            .par_iter()
            .filter(|entry| entry.b % nworker == worker)
            .map(|entry| {
                let uf = &scratch.first_f[entry.b * block.nsa..(entry.b + 1) * block.nsa];
                let us = &scratch.first_s[entry.b * block.nsa..(entry.b + 1) * block.nsa];
                let sarow = &sa[entry.a * block.nsa..(entry.a + 1) * block.nsa];
                let farow = &fa[entry.a * block.nsa..(entry.a + 1) * block.nsa];

                let mut value = zero;

                // `Y^Q = S^\alpha U^F + (F^\alpha + lambda S^\alpha) U^S`.
                for a in 0..block.nsa {
                    let s = <R as From<T>>::from(sarow[a]);
                    let f = <R as From<T>>::from(farow[a]);
                    value += s * uf[a] + (f + lambda * s) * us[a];
                }

                (entry.det, value)
            })
            .collect();

        for (det, value) in updates {
            y[det] += value;
        }
    }

    /// Apply transient alpha-first contraction without materialising same-spin factor tables.
    /// Forms `T^F_{\bar a b} = \sum_a F^\alpha_{\bar a a} D_{ab}` and
    /// `T^S_{\bar a b} = \sum_a S^\alpha_{\bar a a} D_{ab}` while alpha factors are generated,
    /// then generates each beta factor row once and immediately contracts it into `Y^Q`.
    /// # Arguments:
    /// - `block`: Nonpersistent parent-pair metadata.
    /// - `vectors`: Source determinant vector and output determinant vector to accumulate.
    /// - `data`: Shared NOCI data containing Wick intermediates.
    /// - `lambda`: Scalar overlap shift.
    /// - `scratch`: Reusable dense first-stage contraction buffers in scalar `R`.
    /// - `partition`: Worker index and worker count for target alpha rows.
    /// # Returns
    /// - `()`: Adds this parent-pair contribution into `y`.
    fn apply_one_body_transient_a_first<R>(
        &self,
        block: &TransientOneBodyBlock,
        vectors: (&[R], &mut [R]),
        data: &NOCIData<'_, T>,
        lambda: R,
        scratch: &mut OneBodyScratch<R>,
        partition: (usize, usize),
    ) where
        R: NOCIScalar + From<T>,
    {
        let (x, y) = vectors;
        let zero = R::from_real(0.0);
        let factor_zero = T::from_real(0.0);

        let source = &self.spin.parents[block.source_parent];
        let target = &self.spin.parents[block.target_parent];

        let nta = target.areps.len();
        let ntb = target.breps.len();

        let nsa = source.areps.len();
        let nsb = source.breps.len();

        let (worker, nworker) = partition;

        let (lp, gp, target_left) =
            ordered_parent_pair(&self.spin, block.target_parent, block.source_parent);

        let pair = data
            .wicks
            .expect("factorised one-body requires Wick intermediates")
            .pair(lp, gp);

        let first_len = nta
            .checked_mul(nsb)
            .expect("alpha-first transient intermediate length overflow");

        // Retain the allocation between parent blocks. Unlike clear followed by
        // resize, equal-sized blocks do not serially rewrite the complete buffer.
        if scratch.first_f.len() != first_len {
            scratch.first_f.resize(first_len, zero);
        }

        if scratch.first_s.len() != first_len {
            scratch.first_s.resize(first_len, zero);
        }

        // Generate one alpha factor row per target alpha component and consume it
        // immediately into the first-stage F and S intermediates.
        scratch
            .first_f
            .par_chunks_mut(nsb)
            .zip(scratch.first_s.par_chunks_mut(nsb))
            .enumerate()
            .for_each_init(
                || {
                    (
                        WickScratchSpin::new(),
                        vec![factor_zero; nsa],
                        vec![factor_zero; nsa],
                    )
                },
                |state, (ta, (tf, ts))| {
                    if ta % nworker != worker {
                        return;
                    }

                    tf.fill(zero);
                    ts.fill(zero);

                    let (wick, srow, frow) = state;
                    build_spin_one_body_factor_row(
                        &pair,
                        data,
                        (target.areps[ta], source.areps.as_slice()),
                        source.a_eval_order.as_slice(),
                        source.a_eval_groups.as_slice(),
                        target_left,
                        true,
                        wick,
                        (srow.as_mut_slice(), frow.as_mut_slice()),
                    );

                    // `T^F_{\bar a b} = \sum_a F^\alpha_{\bar a a} D_{ab}`,
                    // `T^S_{\bar a b} = \sum_a S^\alpha_{\bar a a} D_{ab}`.
                    for entry in &source.entries {
                        let xe = x[entry.det];

                        if xe != zero {
                            tf[entry.b] += <R as From<T>>::from(frow[entry.a]) * xe;
                            ts[entry.b] += <R as From<T>>::from(srow[entry.a]) * xe;
                        }
                    }
                },
            );

        // A beta row is shared by every target determinant with the same beta
        // component. Generate it once, consume it completely, then reuse the row
        // buffers for the next component.
        let update_groups = (0..ntb)
            .into_par_iter()
            .map_init(
                || {
                    (
                        WickScratchSpin::new(),
                        vec![factor_zero; nsb],
                        vec![factor_zero; nsb],
                    )
                },
                |state, tb| {
                    let (wick, srow, frow) = state;
                    build_spin_one_body_factor_row(
                        &pair,
                        data,
                        (target.breps[tb], source.breps.as_slice()),
                        source.b_eval_order.as_slice(),
                        source.b_eval_groups.as_slice(),
                        target_left,
                        false,
                        wick,
                        (srow.as_mut_slice(), frow.as_mut_slice()),
                    );

                    let mut updates = Vec::with_capacity(target.entries_by_b[tb].len());

                    for &det in &target.entries_by_b[tb] {
                        let ta = self.spin.aids[det];
                        if ta % nworker != worker {
                            continue;
                        }

                        let tf = &scratch.first_f[ta * nsb..(ta + 1) * nsb];
                        let ts = &scratch.first_s[ta * nsb..(ta + 1) * nsb];
                        let mut value = zero;

                        // `Y^Q = T^F (S^\beta)^T + T^S (F^\beta + lambda S^\beta)^T`.
                        for b in 0..nsb {
                            let s = <R as From<T>>::from(srow[b]);
                            let f = <R as From<T>>::from(frow[b]);
                            value += tf[b] * s + ts[b] * (f + lambda * s);
                        }

                        updates.push((det, value));
                    }

                    updates
                },
            )
            .collect::<Vec<_>>();

        for updates in update_groups {
            for (det, value) in updates {
                y[det] += value;
            }
        }
    }

    /// Apply transient beta-first contraction without materialising same-spin factor tables.
    /// Forms `U^F_{a\bar b} = \sum_b D_{ab} F^\beta_{\bar b b}` and
    /// `U^S_{a\bar b} = \sum_b D_{ab} S^\beta_{\bar b b}` while beta factors are generated,
    /// then generates each alpha factor row once and immediately contracts it into `Y^Q`.
    /// # Arguments:
    /// - `block`: Nonpersistent parent-pair metadata.
    /// - `vectors`: Source determinant vector and output determinant vector to accumulate.
    /// - `data`: Shared NOCI data containing Wick intermediates.
    /// - `lambda`: Scalar overlap shift.
    /// - `scratch`: Reusable dense first-stage contraction buffers in scalar `R`.
    /// - `partition`: Worker index and worker count for target beta rows.
    /// # Returns
    /// - `()`: Adds this parent-pair contribution into `y`.
    fn apply_one_body_transient_b_first<R>(
        &self,
        block: &TransientOneBodyBlock,
        vectors: (&[R], &mut [R]),
        data: &NOCIData<'_, T>,
        lambda: R,
        scratch: &mut OneBodyScratch<R>,
        partition: (usize, usize),
    ) where
        R: NOCIScalar + From<T>,
    {
        let (x, y) = vectors;
        let zero = R::from_real(0.0);
        let factor_zero = T::from_real(0.0);

        let source = &self.spin.parents[block.source_parent];
        let target = &self.spin.parents[block.target_parent];

        let nta = target.areps.len();
        let ntb = target.breps.len();

        let nsa = source.areps.len();
        let nsb = source.breps.len();

        let (worker, nworker) = partition;

        let (lp, gp, target_left) =
            ordered_parent_pair(&self.spin, block.target_parent, block.source_parent);

        let pair = data
            .wicks
            .expect("factorised one-body requires Wick intermediates")
            .pair(lp, gp);

        let first_len = ntb
            .checked_mul(nsa)
            .expect("beta-first transient intermediate length overflow");

        if scratch.first_f.len() != first_len {
            scratch.first_f.resize(first_len, zero);
        }

        if scratch.first_s.len() != first_len {
            scratch.first_s.resize(first_len, zero);
        }

        // Generate one beta factor row per target beta component and consume it
        // immediately into the first-stage F and S intermediates.
        scratch
            .first_f
            .par_chunks_mut(nsa)
            .zip(scratch.first_s.par_chunks_mut(nsa))
            .enumerate()
            .for_each_init(
                || {
                    (
                        WickScratchSpin::new(),
                        vec![factor_zero; nsb],
                        vec![factor_zero; nsb],
                    )
                },
                |state, (tb, (uf, us))| {
                    if tb % nworker != worker {
                        return;
                    }

                    uf.fill(zero);
                    us.fill(zero);

                    let (wick, srow, frow) = state;
                    build_spin_one_body_factor_row(
                        &pair,
                        data,
                        (target.breps[tb], source.breps.as_slice()),
                        source.b_eval_order.as_slice(),
                        source.b_eval_groups.as_slice(),
                        target_left,
                        false,
                        wick,
                        (srow.as_mut_slice(), frow.as_mut_slice()),
                    );

                    // `U^F_{a\bar b} = \sum_b D_{ab} F^\beta_{\bar b b}`,
                    // `U^S_{a\bar b} = \sum_b D_{ab} S^\beta_{\bar b b}`.
                    for entry in &source.entries {
                        let xe = x[entry.det];

                        if xe != zero {
                            uf[entry.a] += xe * <R as From<T>>::from(frow[entry.b]);
                            us[entry.a] += xe * <R as From<T>>::from(srow[entry.b]);
                        }
                    }
                },
            );

        // Generate each alpha factor row once and immediately consume it for all
        // target determinants sharing that alpha component.
        let update_groups = (0..nta)
            .into_par_iter()
            .map_init(
                || {
                    (
                        WickScratchSpin::new(),
                        vec![factor_zero; nsa],
                        vec![factor_zero; nsa],
                    )
                },
                |state, ta| {
                    let (wick, srow, frow) = state;
                    build_spin_one_body_factor_row(
                        &pair,
                        data,
                        (target.areps[ta], source.areps.as_slice()),
                        source.a_eval_order.as_slice(),
                        source.a_eval_groups.as_slice(),
                        target_left,
                        true,
                        wick,
                        (srow.as_mut_slice(), frow.as_mut_slice()),
                    );

                    let mut updates = Vec::with_capacity(target.entries_by_a[ta].len());

                    for &det in &target.entries_by_a[ta] {
                        let tb = self.spin.bids[det];
                        if tb % nworker != worker {
                            continue;
                        }

                        let uf = &scratch.first_f[tb * nsa..(tb + 1) * nsa];
                        let us = &scratch.first_s[tb * nsa..(tb + 1) * nsa];
                        let mut value = zero;

                        // `Y^Q = S^\alpha U^F + (F^\alpha + lambda S^\alpha) U^S`.
                        for a in 0..nsa {
                            let s = <R as From<T>>::from(srow[a]);
                            let f = <R as From<T>>::from(frow[a]);
                            value += s * uf[a] + (f + lambda * s) * us[a];
                        }

                        updates.push((det, value));
                    }

                    updates
                },
            )
            .collect::<Vec<_>>();

        for updates in update_groups {
            for (det, value) in updates {
                y[det] += value;
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

    {
        let out = factors.alpha_mut();
        build_spin_one_body_factors(
            &pair,
            data,
            (target.areps.as_slice(), source.areps.as_slice()),
            source.a_eval_order.as_slice(),
            source.a_eval_groups.as_slice(),
            0..nta,
            target_left,
            true,
            out,
        );
    }
    factors.flush();

    {
        let out = factors.beta_mut();
        build_spin_one_body_factors(
            &pair,
            data,
            (target.breps.as_slice(), source.breps.as_slice()),
            source.b_eval_order.as_slice(),
            source.b_eval_groups.as_slice(),
            0..ntb,
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

/// Apply all alpha single-excitation orthogonal Fock couplings from one source determinant.
/// # Arguments:
/// - `orthogonal`: Parent-local occupation lookup.
/// - `source`: Source determinant.
/// - `xe`: Source vector coefficient.
/// - `y`: Output determinant vector to accumulate.
/// - `basis`: Candidate determinant basis.
/// - `cache`: MO-basis Fock cache for the parent.
/// - `partition`: Worker index and worker count for target rows.
/// # Returns
/// - `()`: Adds alpha single-excitation Fock contributions into `y`.
fn apply_orthogonal_alpha_singles<T, R>(
    orthogonal: &OrthogonalOneBodyBlock,
    source: &DetState<T>,
    xe: R,
    y: &mut [R],
    basis: &[DetState<T>],
    cache: &FockMOCache<T>,
    partition: (usize, usize),
) where
    T: NOCIScalar,
    R: NOCIScalar + From<T>,
{
    let (worker, nworker) = partition;
    let nmo = cache.fa.nrows();
    let mut holes = source.oa;
    while holes != 0 {
        let hole = holes.trailing_zeros() as usize;
        holes &= holes - 1;

        for part in 0..nmo {
            if ((source.oa >> part) & 1) == 1 {
                continue;
            }
            let target_oa = (source.oa & !(1u128 << hole)) | (1u128 << part);

            let Some(&opos) = orthogonal.opos.get(&(target_oa, source.ob)) else {
                continue;
            };

            for target in &orthogonal.groups[opos].targets {
                if target.a % nworker == worker {
                    let target_det = &basis[target.det];
                    y[target.det] += <R as From<T>>::from(calculate_f_pair_orthogonal(
                        cache, target_det, source,
                    )) * xe;
                }
            }
        }
    }
}

/// Apply all beta single-excitation orthogonal Fock couplings from one source determinant.
/// # Arguments:
/// - `orthogonal`: Parent-local occupation lookup.
/// - `source`: Source determinant.
/// - `xe`: Source vector coefficient.
/// - `y`: Output determinant vector to accumulate.
/// - `basis`: Candidate determinant basis.
/// - `cache`: MO-basis Fock cache for the parent.
/// - `partition`: Worker index and worker count for target rows.
/// # Returns
/// - `()`: Adds beta single-excitation Fock contributions into `y`.
fn apply_orthogonal_beta_singles<T, R>(
    orthogonal: &OrthogonalOneBodyBlock,
    source: &DetState<T>,
    xe: R,
    y: &mut [R],
    basis: &[DetState<T>],
    cache: &FockMOCache<T>,
    partition: (usize, usize),
) where
    T: NOCIScalar,
    R: NOCIScalar + From<T>,
{
    let (worker, nworker) = partition;
    let nmo = cache.fb.nrows();
    let mut holes = source.ob;
    while holes != 0 {
        let hole = holes.trailing_zeros() as usize;
        holes &= holes - 1;

        for part in 0..nmo {
            if ((source.ob >> part) & 1) == 1 {
                continue;
            }
            let target_ob = (source.ob & !(1u128 << hole)) | (1u128 << part);

            let Some(&opos) = orthogonal.opos.get(&(source.oa, target_ob)) else {
                continue;
            };

            for target in &orthogonal.groups[opos].targets {
                if target.a % nworker == worker {
                    let target_det = &basis[target.det];
                    y[target.det] += <R as From<T>>::from(calculate_f_pair_orthogonal(
                        cache, target_det, source,
                    )) * xe;
                }
            }
        }
    }
}

/// Build same-spin `S` and `F` factor rows from prepared Wick pair batches.
/// Independent source components are evaluated together so the widest fixed-rank SIMD kernel can be used.
/// # Arguments:
/// - `pair`: Wick intermediates for the ordered parent pair.
/// - `data`: Shared NOCI determinant data.
/// - `reps`: Reduced target and source spin representatives.
/// - `source_order`: Source component IDs in fixed-rank Wick evaluation order.
/// - `source_groups`: Boundaries of equal-rank, common-hole source groups in `source_order`.
/// - `rows`: Target-row range to fill.
/// - `target_left`: Whether target determinants are left determinants in `pair`.
/// - `alpha`: Whether to build alpha or beta factors.
/// - `out`: Mutable row-major overlap and Fock factor tables.
/// # Returns
/// - `()`: Fills `out` factor tables.
fn build_spin_one_body_factors<T: NOCIScalar>(
    pair: &WicksPairView<'_, T>,
    data: &NOCIData<'_, T>,
    reps: (&[ReducedOneSpinDetState], &[ReducedOneSpinDetState]),
    source_order: &[usize],
    source_groups: &[usize],
    rows: Range<usize>,
    target_left: bool,
    alpha: bool,
    out: (&mut [T], &mut [T]),
) {
    let (target_reps, source_reps) = reps;
    let nsource = source_reps.len();
    let row0 = rows.start;
    let row1 = rows.end;

    out.0[row0 * nsource..row1 * nsource]
        .par_chunks_mut(nsource)
        .zip(out.1[row0 * nsource..row1 * nsource].par_chunks_mut(nsource))
        .zip(target_reps[row0..row1].par_iter())
        .for_each_init(
            WickScratchSpin::new,
            |scratch, ((srow, frow), &target_rep)| {
                build_spin_one_body_factor_row(
                    pair,
                    data,
                    (target_rep, source_reps),
                    source_order,
                    source_groups,
                    target_left,
                    alpha,
                    scratch,
                    (&mut *srow, &mut *frow),
                );
            },
        );
}

/// Build one same-spin `S` and `F` factor row from prepared Wick pair batches.
/// Fixed-rank paths read phase and excitation metadata directly from `ReducedOneSpinDetState`.
/// # Arguments:
/// - `pair`: Wick intermediates for the ordered parent pair.
/// - `data`: Shared NOCI determinant data used by the generic fallback.
/// - `reps`: Reduced target representative and source representatives.
/// - `source_order`: Source component IDs in fixed-rank Wick evaluation order.
/// - `source_groups`: Boundaries of equal-rank, common-hole source groups in `source_order`.
/// - `target_left`: Whether the target determinant belongs to the left Wick reference.
/// - `alpha`: Whether to evaluate alpha-alpha or beta-beta factors.
/// - `scratch`: Reusable spin-resolved Wick evaluator workspace.
/// - `out`: Output same-spin overlap and generalised-Fock factor rows.
/// # Returns
/// - `()`: Fills `srow` and `frow` for the target spin component.
fn build_spin_one_body_factor_row<T: NOCIScalar>(
    pair: &WicksPairView<'_, T>,
    data: &NOCIData<'_, T>,
    reps: (ReducedOneSpinDetState, &[ReducedOneSpinDetState]),
    source_order: &[usize],
    source_groups: &[usize],
    target_left: bool,
    alpha: bool,
    scratch: &mut WickScratchSpin<T>,
    out: (&mut [T], &mut [T]),
) {
    let (target, sources) = reps;
    let (overlap, fock) = out;

    let (w, scratch) = if alpha {
        (&pair.aa, &mut scratch.aa)
    } else {
        (&pair.bb, &mut scratch.bb)
    };

    xw_f_overlap_prepared_batched(
        w,
        SameSpinOneBodyBatch {
            basis: data.basis,
            target,
            sources,
            source_order,
            source_groups,
            target_left,
            alpha,
            overlap,
            fock,
        },
        scratch,
        data.tol,
    );
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
