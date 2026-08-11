// noci/factorise/onebody/cpu/backend.rs
//! CPU backend for spin-factorised one-body NOCI operator contractions.

// Standard library imports.
use std::path::Path;

// External crate imports.
use ndarray::Array1;

// Crate-root imports.
use crate::input::SNOCIStorage;
use crate::noci::types::{FockData, NOCIData, NOCIScalar};

// Parent/sibling imports.
use super::super::super::SpinFactorisation;
use super::super::super::storage::OneBodyStoragePlan;
use super::super::plan::{OneBodyBlockPlan, OneBodyPlan};
use super::contract::{OneBodyScratch, apply_one_body_factorised};
use super::diagonals::one_body_diagonals;
use super::factors::{
    FactorisedOneBodyBlock, TransientOneBodyBlock, build_one_body_factor_tables,
    build_transient_one_body_block,
};
use super::orthogonal::{
    OrthogonalApplyContext, OrthogonalOneBodyBlock, apply_one_body_orthogonal,
    build_orthogonal_one_body_block,
};

/// Persistent one-body block for one ordered source-target parent pair `QP`.
pub(super) enum OneBodyBlock<T: NOCIScalar> {
    /// Same-parent standard Slater-Condon sparse one-body action.
    Orthogonal(OrthogonalOneBodyBlock),
    /// Spin-factorised nonorthogonal one-body action.
    Factorised(FactorisedOneBodyBlock<T>),
    /// Spin-factorised nonorthogonal one-body action with regenerated factor tables.
    Transient(TransientOneBodyBlock),
}

/// Cached spin-factorised one-body operator for the current generalised Fock.
pub(crate) struct CpuOneBodyBackend<T: NOCIScalar> {
    /// Shared determinant-space factorisation `I <-> (P,a_I,b_I)`.
    spin: SpinFactorisation,
    /// Cached parent-pair factor blocks indexed as `Q * nparent + P`.
    blocks: Vec<OneBodyBlock<T>>,
    /// Number of parent references.
    nparent: usize,
    /// Reusable dense contraction scratch buffers.
    scratch: OneBodyScratch<T>,
}

impl<T: NOCIScalar> CpuOneBodyBackend<T> {
    /// Build `F^{QP}_{\bar a\bar b,ab}` spin factors for the current generalised Fock operator.
    /// # Arguments:
    /// - `data`: Shared NOCI data with Wick intermediates for the candidate determinant basis.
    /// - `fock`: Current generalised-Fock data, already reflected in Wick intermediates.
    /// - `cache`: Directory for persistent file-backed factor blocks.
    /// - `rank`: MPI rank used in factor-cache filenames.
    /// - `iteration`: SNOCI iteration used in factor-cache filenames.
    /// - `storage`: Requested persistent factor-table storage backend.
    /// # Returns
    /// - `CpuOneBodyBackend<T>`: Cached CPU spin-factorised one-body operator.
    pub(crate) fn new(
        data: &NOCIData<'_, T>,
        fock: &FockData<'_, T>,
        cache: &Path,
        rank: i32,
        iteration: usize,
        storage: SNOCIStorage,
    ) -> Self {
        let spin = SpinFactorisation::new(data);
        let plan = OneBodyPlan::new(&spin, fock);
        let nparent = plan.nparent;
        let mut blocks = Vec::with_capacity(nparent * nparent);
        let mut storage_plan = OneBodyStoragePlan::new(cache, rank, iteration, storage);

        for block_plan in &plan.blocks {
            match *block_plan {
                OneBodyBlockPlan::Orthogonal { parent } => {
                    blocks.push(OneBodyBlock::Orthogonal(build_orthogonal_one_body_block(
                        &spin, data, parent,
                    )));
                }
                OneBodyBlockPlan::NonOrthogonal {
                    target_parent,
                    source_parent,
                    lp,
                    gp,
                    target_left,
                    nta: _,
                    ntb: _,
                    nsa: _,
                    nsb: _,
                    contraction,
                } if matches!(storage, SNOCIStorage::None) => {
                    blocks.push(OneBodyBlock::Transient(build_transient_one_body_block(
                        target_parent,
                        source_parent,
                        lp,
                        gp,
                        target_left,
                        contraction,
                    )))
                }
                OneBodyBlockPlan::NonOrthogonal {
                    target_parent,
                    source_parent,
                    lp,
                    gp,
                    target_left,
                    nta,
                    ntb,
                    nsa,
                    nsb,
                    contraction,
                } => {
                    blocks.push(OneBodyBlock::Factorised(build_one_body_factor_tables(
                        &spin,
                        data,
                        &mut storage_plan,
                        target_parent,
                        source_parent,
                        lp,
                        gp,
                        target_left,
                        nta,
                        ntb,
                        nsa,
                        nsb,
                        contraction,
                    )));
                }
            }
        }

        Self {
            spin,
            blocks,
            nparent,
            scratch: OneBodyScratch::default(),
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
        let plan = OneBodyPlan::new(&spin, fock);
        let mut nentries = 0usize;
        for block in plan.blocks {
            let OneBodyBlockPlan::NonOrthogonal {
                nta, ntb, nsa, nsb, ..
            } = block
            else {
                continue;
            };
            let alpha = nta
                .checked_mul(nsa)
                .expect("alpha one-body factor length overflow");
            let beta = ntb
                .checked_mul(nsb)
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
    /// - `partition`: Worker index and worker count for first-stage target rows.
    /// # Returns
    /// - `Array1<T>`: Partial or complete determinant-space result vector.
    pub(crate) fn apply_one_body(
        &mut self,
        x: &Array1<T>,
        data: &NOCIData<'_, T>,
        fock: &FockData<'_, T>,
        lambda: T,
        partition: (usize, usize),
    ) -> Array1<T> {
        let mut scratch = std::mem::take(&mut self.scratch);
        let y = self.apply_one_body_with_scratch(x, data, fock, lambda, &mut scratch, partition);
        self.scratch = scratch;
        y
    }

    /// Apply `Y = (F + \lambda S)x` using cached spin factors and explicit scratch.
    /// # Arguments:
    /// - `x`: Source vector over actual candidate determinants.
    /// - `data`: Shared NOCI data used by same-parent orthogonal blocks.
    /// - `fock`: Current generalised-Fock data used by same-parent orthogonal blocks.
    /// - `lambda`: Scalar shift multiplying the overlap operator.
    /// - `scratch`: Reusable dense contraction buffers.
    /// - `partition`: Worker index and worker count for first-stage target rows.
    /// # Returns
    /// - `Array1<T>`: Partial or complete determinant-space result vector.
    fn apply_one_body_with_scratch(
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
                    apply_one_body_orthogonal(&self.spin, block, xs, &mut y, &context)
                }
                OneBodyBlock::Factorised(block) => apply_one_body_factorised(
                    &self.spin,
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
                        block.lp,
                        block.gp,
                        block.target_left,
                        self.spin.parents[block.target_parent].areps.len(),
                        self.spin.parents[block.target_parent].breps.len(),
                        self.spin.parents[block.source_parent].areps.len(),
                        self.spin.parents[block.source_parent].breps.len(),
                        block.contraction,
                    );
                    apply_one_body_factorised(
                        &self.spin,
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
        one_body_diagonals(&self.spin, &self.blocks, self.nparent, data, fock, lambda)
    }
}
