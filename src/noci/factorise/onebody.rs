// noci/factorise/onebody.rs
//! Spin-factorised one-body NOCI operator contractions.

// Standard library imports.
use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;
use std::sync::Mutex;

// External crate imports.
use ndarray::{Array1, Array2, Array4, ArrayView2, Axis, s};
use ndarray_linalg::{Determinant, Inverse};
use rayon::prelude::*;

// Crate-root imports.
use crate::NOCIScalar;
use crate::determinant::{NOCIIndex, NOCISpace, NOCISpinStringIndex, SpinStringData};
use crate::elements::calculate_f_pair_orthogonal;
use crate::elements::calculate_s_pair_orthogonal;
use crate::elements::nonorthogonalwicks::{
    OneBodyFactor, SINGULARTOL, SameSpinBuild, SpinFactorRow, SpinStringSources, WickScratchSpin,
    WicksPairView, xw_spin_factor_row,
};
use crate::elements::{FockData, FockMOCache, NOCIData};
use crate::input::SNOCIStorage;
use crate::maths::adjoint;

// Parent/sibling imports.
use super::storage::{OneBodyFactorStorage, OneBodyStoragePlan};
use super::{FactorEntry, ParentSpinSpace, SpinFactorisation, ordered_parent_pair};

/// Largest same-spin replacement rank applied in contraction form.
const CONTRACTRANK: usize = 2;
/// Extrapolation points `\epsilon` for deformed near-zero occupied overlaps.
const EPSNODES: [f64; 6] = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0];
/// Number of output entries summed per Rayon task when combining per-thread products.
const SUMCHUNK: usize = 4096;
/// Minimum number of target determinants per Rayon task when one spin string's products are split.
const DOTCHUNK: usize = 64;

/// Persistent one-body block for one ordered source-target parent pair `QP`.
enum OneBodyBlock<T: NOCIScalar> {
    /// Same-parent standard Slater-Condon sparse one-body action.
    Orthogonal(OrthogonalOneBodyBlock),
    /// Spin-factorised nonorthogonal one-body action.
    Factorised(FactorisedOneBodyBlock<T>),
    /// Spin-factorised nonorthogonal one-body action with regenerated factor tables.
    Transient(TransientOneBodyBlock),
    /// Nonorthogonal one-body action in contraction form.
    Contracted(ContractedOneBodyBlock<T>),
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

/// Nonorthogonal parent pair `QP` applied in contraction form: every same-spin factor is applied
/// to whole amplitude vectors through orbital-indexed matrices, never element by element.
struct ContractedOneBodyBlock<T: NOCIScalar> {
    /// Target parent `Q`, the bra.
    target_parent: usize,
    /// Source parent `P`, the ket.
    source_parent: usize,
    /// Alpha extrapolation points of `F^\alpha` and `S^\alpha`.
    alpha: Vec<ContractionPoint<T>>,
    /// Beta extrapolation points of `F^\beta` and `S^\beta`.
    beta: Vec<ContractionPoint<T>>,
}

/// Same-spin intermediates of one parent pair at one extrapolation point `\epsilon`. Near-zero
/// occupied overlaps are deformed to `\hat S_k + \epsilon`, which makes every factor a polynomial
/// in `\epsilon`, so the factors at `\epsilon = 0` are a weighted sum over a few points. The bra
/// occupied orbitals are `\phi_i`; the ket occupied orbitals are made biorthogonal to them,
/// `\tilde\phi = C^x_o G`; and both virtual spaces are reduced to their remainders outside the
/// occupied spaces, `\tilde v_a` and `\hat q_c`, biorthogonal to each other.
struct ContractionPoint<T: NOCIScalar> {
    /// Lagrange weight of the point times `\det M_0`, with `M_0 = C^{w\dagger}_o S C^x_o`.
    scale: T,
    /// `G = M_0^{-1}`.
    g: Array2<T>,
    /// `\alpha_{kb} = \phi_k^\dagger S u_b`, the ket virtual `u_b` along `\tilde\phi_k`.
    alpha: Array2<T>,
    /// `\beta_{ai} = v_a^\dagger S \tilde\phi_i`, the bra virtual `v_a` along `\phi_i`.
    beta: Array2<T>,
    /// `Q_{ab} = \tilde v_a^\dagger S q_b`, the overlap of the virtual remainders.
    q: Array2<T>,
    /// One-body operator in the biorthogonal frame, with occupied-occupied block
    /// `f_{ik} = \phi_i^\dagger F \tilde\phi_k`, occupied-virtual block
    /// `f_{ic} = \phi_i^\dagger F \hat q_c`, virtual-occupied block
    /// `f_{ak} = \tilde v_a^\dagger F \tilde\phi_k` and virtual-virtual block
    /// `f_{ac} = \tilde v_a^\dagger F \hat q_c`.
    f: Array2<T>,
}

/// Same-spin amplitudes over the strings of one parent with at most two replacements, as
/// in-place replacements of occupied positions by virtual positions: `c_0`, `c_{ia}` and the
/// antisymmetric `c_{ijab}`, absent when every double is zero.
struct StringAmplitudes<R: NOCIScalar> {
    /// Reference amplitude `c_0`.
    c0: R,
    /// Single replacement amplitudes `c_{ia}`.
    c1: Array2<R>,
    /// Double replacement amplitudes `c_{ijab}`.
    c2: Option<Array4<R>>,
}

/// Actual determinant target in one orthogonal occupation group.
#[derive(Clone, Copy)]
struct OrthogonalTarget {
    /// Global determinant index `I`.
    det: usize,
    /// Parent-local alpha spin string `a_I`.
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
    /// Numerical group keyed by canonical alpha/beta spin-string IDs.
    opos: HashMap<(NOCISpinStringIndex, NOCISpinStringIndex), usize>,
    /// Determinants grouped by the same canonical spin-string pair.
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

        // Choose a representation per ordered parent pair: exact orthogonal Slater-Condon
        // blocks, the contraction form, transient evaluation, or persistent spin factors.
        for target_parent in 0..nparent {
            for source_parent in 0..nparent {
                let contracted =
                    ContractedOneBodyBlock::new((target_parent, source_parent), &spin, data, fock);

                if target_parent == source_parent
                    && fock.fock_mocache[target_parent].orthogonal_slater_condon
                {
                    blocks.push(OneBodyBlock::Orthogonal(build_orthogonal_one_body_block(
                        &spin,
                        target_parent,
                    )));
                } else if let Some(block) = contracted {
                    blocks.push(OneBodyBlock::Contracted(block));
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

        // Each nonorthogonal parent pair stores `\alpha` and `\beta` `S` and `F` factors;
        // same-parent orthogonal blocks and contracted parent pairs need no factor table.
        let mut nentries = 0usize;
        for target_parent in 0..nparent {
            let target = &spin.parents[target_parent];
            for source_parent in 0..nparent {
                if (target_parent == source_parent
                    && fock.fock_mocache[target_parent].orthogonal_slater_condon)
                    || contractible(&spin, (target_parent, source_parent))
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
                    data.space,
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
                // Contracted parent pairs are applied together below.
                OneBodyBlock::Contracted(_) => {}
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

        // Contracted parent pairs run concurrently, each assigned whole to one worker.
        let contracted = self
            .blocks
            .par_iter()
            .filter_map(|block| match block {
                OneBodyBlock::Contracted(block)
                    if (block.target_parent * self.nparent + block.source_parent) % nworker
                        == worker =>
                {
                    Some(block.updates(&self.spin, data, xs, lambda))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        for updates in contracted {
            for (det, value) in updates {
                y[det] += value;
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
        let ndet = data.space.len();
        let mut m_diag = vec![zero; ndet];
        let mut s_diag = vec![zero; ndet];

        // Only same-parent blocks contribute to determinant diagonals.
        // Dispatch through the representation chosen for each parent.
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
                // Contracted blocks join two different parents and carry no diagonal.
                OneBodyBlock::Contracted(_) => {}
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
    /// Sources are applied in parallel, each Rayon thread accumulating into its own output vector.
    /// # Arguments:
    /// - `block`: Same-parent orthogonal action data.
    /// - `vectors`: Source determinant vector and output determinant vector to accumulate.
    /// - `basis`: Candidate determinant basis.
    /// - `cache`: MO-basis Fock cache for the parent.
    /// - `lambda`: Scalar overlap shift.
    /// - `partition`: Worker index and worker count for target rows.
    /// - `space`: Candidate determinant space used for occupations and indices.
    /// # Returns
    /// - `()`: Adds this same-parent contribution into `y`.
    fn apply_one_body_orthogonal<R>(
        &self,
        block: &OrthogonalOneBodyBlock,
        vectors: (&[R], &mut [R]),
        space: &NOCISpace<T>,
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

        // Sources scatter into overlapping targets, so every Rayon thread accumulates into its
        // own output vector, added into `y` once all sources are applied.
        let outputs: Vec<_> = (0..rayon::current_num_threads())
            .map(|_| Mutex::new(vec![zero; y.len()]))
            .collect();

        // Identical occupations contribute `(F + \lambda S)_{II} x_I` to
        // every retained target sharing the source spin strings.
        source_parent.entries.par_iter().for_each(|entry| {
            let xe = x[entry.det];
            if xe == zero {
                return;
            }
            let mut output = outputs[rayon::current_thread_index().unwrap()]
                .lock()
                .unwrap();
            let y = output.as_mut_slice();
            let source = NOCIIndex(entry.det);
            let source_state = space.spin_string_pair(source);
            let group = &block.groups[block.opos[&(source_state.aid, source_state.bid)]];
            for target in &group.targets {
                if target.a % nworker == worker {
                    let target_det = NOCIIndex(target.det);
                    let f = <R as From<T>>::from(calculate_f_pair_orthogonal(
                        cache, space, target_det, source,
                    ));
                    let s = <R as From<T>>::from(calculate_s_pair_orthogonal(
                        space, target_det, source,
                    ));
                    y[target.det] += (f + lambda * s) * xe;
                }
            }

            // A one-body operator also connects `\alpha` or `\beta` single
            // excitations; the helpers enumerate only retained targets.
            apply_orthogonal_alpha_singles(block, source, xe, y, space, cache, partition);
            apply_orthogonal_beta_singles(block, source, xe, y, space, cache, partition);
        });

        // Add the per-thread outputs, splitting the entries of `y` across threads.
        let outputs: Vec<Vec<R>> = outputs
            .into_iter()
            .map(|output| output.into_inner().unwrap())
            .collect();

        y.par_chunks_mut(SUMCHUNK)
            .enumerate()
            .for_each(|(chunk, entries)| {
                let first = chunk * SUMCHUNK;
                for output in &outputs {
                    for (entry, &value) in entries.iter_mut().zip(&output[first..]) {
                        *entry += value;
                    }
                }
            });
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

        // Generate one alpha factor row per target alpha spin string and consume it
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
                        target.areps[ta],
                        &source.a_sources,
                        (target_left, true),
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
        // spin string. Generate it once, consume it completely, then reuse the row
        // buffers for the next spin string.
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
                        target.breps[tb],
                        &source.b_sources,
                        (target_left, false),
                        wick,
                        (srow.as_mut_slice(), frow.as_mut_slice()),
                    );

                    // One beta string can be shared by most target determinants, such as the
                    // reference string with every alpha excitation, so split its products
                    // across threads as well.
                    let (srow, frow) = (&*srow, &*frow);
                    target.entries_by_b[tb]
                        .par_iter()
                        .with_min_len(DOTCHUNK)
                        .filter_map(|&det| {
                            let ta = data.space.spin_string_pair(NOCIIndex(det)).aid.0;
                            if ta % nworker != worker {
                                return None;
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

                            Some((det, value))
                        })
                        .collect::<Vec<_>>()
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

        // Generate one beta factor row per target beta spin string and consume it
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
                        target.breps[tb],
                        &source.b_sources,
                        (target_left, false),
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
        // target determinants sharing that alpha spin string.
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
                        target.areps[ta],
                        &source.a_sources,
                        (target_left, true),
                        wick,
                        (srow.as_mut_slice(), frow.as_mut_slice()),
                    );

                    // One alpha string can be shared by most target determinants, such as the
                    // reference string with every beta excitation, so split its products
                    // across threads as well.
                    let (srow, frow) = (&*srow, &*frow);
                    target.entries_by_a[ta]
                        .par_iter()
                        .with_min_len(DOTCHUNK)
                        .filter_map(|&det| {
                            let tb = data.space.spin_string_pair(NOCIIndex(det)).bid.0;
                            if tb % nworker != worker {
                                return None;
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

                            Some((det, value))
                        })
                        .collect::<Vec<_>>()
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

impl<T: NOCIScalar> ContractedOneBodyBlock<T> {
    /// Prepare the contraction form of one ordered parent pair `QP`, when `contractible` holds and
    /// every near-zero occupied overlap can be extrapolated.
    /// # Arguments:
    /// - `parents`: Target parent `Q` and source parent `P`.
    /// - `spin`: Shared determinant-space factorisation.
    /// - `data`: Shared NOCI data holding the parent orbitals, the AO overlap and the zero-overlap
    ///   tolerance.
    /// - `fock`: Current generalised Fock matrices in the AO basis.
    /// # Returns
    /// - `Option<Self>`: The contracted block, or `None` when the pair keeps the factor rows.
    fn new(
        parents: (usize, usize),
        spin: &SpinFactorisation,
        data: &NOCIData<'_, T>,
        fock: &FockData<'_, T>,
    ) -> Option<Self> {
        if !contractible(spin, parents) {
            return None;
        }
        let (target_parent, source_parent) = parents;
        let (bra, ket) = (
            &data.space.parents()[target_parent],
            &data.space.parents()[source_parent],
        );

        Some(Self {
            target_parent,
            source_parent,
            alpha: ContractionPoint::build(
                (&bra.ca, bra.oa),
                (&ket.ca, ket.oa),
                (&data.ao.s, fock.fa),
                data.tol,
            )?,
            beta: ContractionPoint::build(
                (&bra.cb, bra.ob),
                (&ket.cb, ket.ob),
                (&data.ao.s, fock.fb),
                data.tol,
            )?,
        })
    }

    /// Contributions to `Y^Q = (F + \lambda S)^{QP} x^P` of this parent pair, with
    /// `F^{QP} = F^\alpha D (S^\beta)^T + S^\alpha D (F^\beta)^T` and
    /// `S^{QP} = S^\alpha D (S^\beta)^T` for the source amplitudes `D_{ab}` over alpha and beta
    /// strings, whose determinants have at most two replacements in total. `D` is split by
    /// excitation class and, by linearity, no string-by-string factor block is formed:
    /// 1. the beta-reference column `D_{a0}`: one alpha apply, times the beta factors of the
    ///    beta reference;
    /// 2. the alpha-reference row `D_{0b}` over excited beta strings: one beta apply, times the
    ///    alpha factors of the alpha reference;
    /// 3. the alpha-single by beta-single block `D^{11}`, with columns `c` over beta singles.
    ///    Targets with the beta reference receive the alpha applies of `D^{11} w` with
    ///    `w_c = S^\beta_{0c}` or `F^\beta_{0c}`; targets with the alpha reference the beta
    ///    applies of the reference rows `(F^\alpha D^{11})_{0c}` and `(S^\alpha D^{11})_{0c}`;
    ///    single-single targets the product of the single rows of `F^\alpha D^{11}` and
    ///    `S^\alpha D^{11}` with those of `S^\beta` and `F^\beta`.
    /// # Arguments:
    /// - `spin`: Shared determinant-space factorisation.
    /// - `data`: Shared NOCI data holding the parent occupations.
    /// - `x`: Source determinant vector.
    /// - `lambda`: Scalar overlap shift.
    /// # Returns
    /// - `Vec<(usize, R)>`: Target determinant and the value to add to it.
    fn updates<R>(
        &self,
        spin: &SpinFactorisation,
        data: &NOCIData<'_, T>,
        x: &[R],
        lambda: R,
    ) -> Vec<(usize, R)>
    where
        R: NOCIScalar + From<T>,
    {
        let (source, target) = (
            &spin.parents[self.source_parent],
            &spin.parents[self.target_parent],
        );
        let (bra, ket) = (
            &data.space.parents()[self.target_parent],
            &data.space.parents()[self.source_parent],
        );
        let rank = |e: &FactorEntry| (source.areps[e.a].cache.rank, source.breps[e.b].cache.rank);
        let n = ket.ca.ncols();
        let amplitudes = |occupation: u128,
                          strings: &[SpinStringData],
                          values: Vec<(usize, R)>,
                          doubles: bool| {
            StringAmplitudes::gather((n, occupation, strings), values, doubles)
        };
        let reference = |occupation: u128| {
            let nocc = occupation.count_ones() as usize;
            let mut r = StringAmplitudes::zeros((nocc, n - nocc), false);
            r.c0 = R::one();
            r
        };

        // Class 1: the beta-reference column, and the beta factors of the beta reference.
        let column = source
            .entries
            .iter()
            .filter(|e| rank(e).1 == 0)
            .map(|e| (e.a, x[e.det]))
            .collect();
        let (fa1, sa1) = ContractionPoint::apply(
            &self.alpha,
            &amplitudes(ket.oa, &source.areps, column, true),
            true,
        );
        let (fb0, sb0) = ContractionPoint::apply(&self.beta, &reference(ket.ob), true);

        // Class 2: the alpha-reference row over excited beta strings, and the alpha factors of
        // the alpha reference.
        let row = source
            .entries
            .iter()
            .filter(|e| rank(e).0 == 0 && rank(e).1 > 0)
            .map(|e| (e.b, x[e.det]))
            .collect();
        let (fb2, sb2) = ContractionPoint::apply(
            &self.beta,
            &amplitudes(ket.ob, &source.breps, row, true),
            true,
        );
        let (fa0, sa0) = ContractionPoint::apply(&self.alpha, &reference(ket.oa), true);

        // Class 3: one column of `D^{11}` per beta single. Each column needs only the reference and
        // single-replacement outputs of its alpha apply and of the beta factors of its single.
        let mut grouped: HashMap<usize, Vec<(usize, R)>> = HashMap::new();
        for e in source.entries.iter().filter(|e| rank(e) == (1, 1)) {
            grouped.entry(e.b).or_default().push((e.a, x[e.det]));
        }
        let grouped = grouped.into_iter().collect::<Vec<_>>();
        let columns = grouped
            .par_iter()
            .map(|(b, column)| {
                let alpha = amplitudes(ket.oa, &source.areps, column.clone(), false);
                let beta = amplitudes(ket.ob, &source.breps, vec![(*b, R::one())], false);
                (
                    ContractionPoint::apply(&self.alpha, &alpha, false),
                    ContractionPoint::apply(&self.beta, &beta, false),
                )
            })
            .collect::<Vec<_>>();

        // Beta-reference targets: alpha applies of `D^{11} w^S` and `D^{11} w^F`.
        let weighted = |w: &dyn Fn(usize) -> R| {
            let values = grouped
                .iter()
                .enumerate()
                .flat_map(|(c, (_, column))| {
                    let wc = w(c);
                    column.iter().map(move |&(a, value)| (a, value * wc))
                })
                .collect();
            amplitudes(ket.oa, &source.areps, values, false)
        };
        let (fws, sws) =
            ContractionPoint::apply(&self.alpha, &weighted(&|c| columns[c].1.1.c0), true);
        let (_, swf) =
            ContractionPoint::apply(&self.alpha, &weighted(&|c| columns[c].1.0.c0), true);

        // Alpha-reference targets: beta applies of the reference rows of `F^\alpha D^{11}` and
        // `S^\alpha D^{11}`.
        let reference_row = |pick: &dyn Fn(usize) -> R| {
            let values = grouped
                .iter()
                .enumerate()
                .map(|(c, (b, _))| (*b, pick(c)))
                .collect();
            amplitudes(ket.ob, &source.breps, values, false)
        };
        let (_, suf) =
            ContractionPoint::apply(&self.beta, &reference_row(&|c| columns[c].0.0.c0), true);
        let (fus, sus) =
            ContractionPoint::apply(&self.beta, &reference_row(&|c| columns[c].0.1.c0), true);

        // Single-single targets: `Y^F = X^F (S^\beta)^T + X^S (F^\beta)^T` and
        // `Y^S = X^S (S^\beta)^T`, with one row per target single and one column per beta single.
        // `rows` maps every target string to its row, `usize::MAX` for strings that are not singles.
        let singles =
            |occupation: u128, strings: &[SpinStringData], pick: &[&StringAmplitudes<R>]| {
                let ids = (0..strings.len())
                    .filter(|&k| strings[k].cache.rank == 1)
                    .collect::<Vec<_>>();
                let mut rows = vec![usize::MAX; strings.len()];
                for (r, &k) in ids.iter().enumerate() {
                    rows[k] = r;
                }
                let m = Array2::from_shape_fn((ids.len(), pick.len()), |(r, c)| {
                    pick[c].read(occupation, &strings[ids[r]])
                });
                (m, rows)
            };
        let (fa, sa): (Vec<_>, Vec<_>) = columns.iter().map(|c| (&c.0.0, &c.0.1)).unzip();
        let (fb, sb): (Vec<_>, Vec<_>) = columns.iter().map(|c| (&c.1.0, &c.1.1)).unzip();
        let (xf, arows) = singles(bra.oa, &target.areps, &fa);
        let (xs, _) = singles(bra.oa, &target.areps, &sa);
        let (bf, brows) = singles(bra.ob, &target.breps, &fb);
        let (bs, _) = singles(bra.ob, &target.breps, &sb);
        let yf = xf.dot(&bs.t()) + xs.dot(&bf.t());
        let ys = xs.dot(&bs.t());

        // Every class at the target strings.
        let at = |amp: &StringAmplitudes<R>, occupation: u128, strings: &[SpinStringData]| {
            strings
                .iter()
                .map(|st| amp.read(occupation, st))
                .collect::<Vec<_>>()
        };
        let alpha = |amp: &StringAmplitudes<R>| at(amp, bra.oa, &target.areps);
        let beta = |amp: &StringAmplitudes<R>| at(amp, bra.ob, &target.breps);
        let (fa1, sa1, fa0, sa0) = (alpha(&fa1), alpha(&sa1), alpha(&fa0), alpha(&sa0));
        let (fws, sws, swf) = (alpha(&fws), alpha(&sws), alpha(&swf));
        let (fb0, sb0, fb2, sb2) = (beta(&fb0), beta(&sb0), beta(&fb2), beta(&sb2));
        let (suf, fus, sus) = (beta(&suf), beta(&fus), beta(&sus));

        // `Y^Q_{\bar a\bar b} = F_{\bar a\bar b} + \lambda S_{\bar a\bar b}` at every target
        // determinant: classes 1 and 2 everywhere, class 3 by target class.
        target
            .entries
            .par_iter()
            .with_min_len(DOTCHUNK)
            .map(|e| {
                let (i, j) = (e.a, e.b);
                let mut f = fa1[i] * sb0[j] + sa1[i] * fb0[j] + fa0[i] * sb2[j] + sa0[i] * fb2[j];
                let mut s = sa1[i] * sb0[j] + sa0[i] * sb2[j];

                match (target.areps[i].cache.rank, target.breps[j].cache.rank) {
                    (_, 0) => {
                        f += fws[i] + swf[i];
                        s += sws[i];
                    }
                    (0, _) => {
                        f += suf[j] + fus[j];
                        s += sus[j];
                    }
                    (1, 1) => {
                        f += yf[(arows[i], brows[j])];
                        s += ys[(arows[i], brows[j])];
                    }
                    _ => {}
                }

                (e.det, f + lambda * s)
            })
            .collect()
    }
}

impl<T: NOCIScalar> ContractionPoint<T> {
    /// Prepare the extrapolation points of the same-spin factors between the strings of a bra
    /// reference `w` and a ket reference `x`. With the Löwdin pairing
    /// `M_0 = U \tilde S V^\dagger`, every pair with `\tilde S_k < \mathtt{SINGULARTOL}` is
    /// deformed along `(C^w_o u_k) v_k^\dagger` to `\hat S_k + \epsilon`, where `\hat S_k = 0` when
    /// `\tilde S_k \le \mathtt{tol}`, as in the Wick kernels, and `\hat S_k = \tilde S_k`
    /// otherwise. Every factor is then a polynomial of degree at most the number of deformed pairs
    /// in `\epsilon`, and the factors at `\epsilon = 0` follow exactly from Lagrange weights at
    /// well-conditioned points.
    /// # Arguments:
    /// - `bra`: Bra orbitals and occupation of this spin.
    /// - `ket`: Ket orbitals and occupation of this spin.
    /// - `operators`: AO overlap matrix and one-body operator of this spin.
    /// - `tol`: Singular values at or below this are exact zeros.
    /// # Returns
    /// - `Option<Vec<Self>>`: The points, or `None` when the pair cannot be extrapolated.
    fn build(
        bra: (&Array2<T>, u128),
        ket: (&Array2<T>, u128),
        operators: (&Array2<f64>, &Array2<T>),
        tol: f64,
    ) -> Option<Vec<Self>> {
        let (s, f) = operators;

        // Occupied and virtual orbitals of both references, each in ascending orbital order.
        let split = |c: &Array2<T>, occupation: u128| {
            let n = c.ncols();
            let pick = |occupied: bool| {
                let columns = (0..n)
                    .filter(|&k| (occupation >> k & 1 == 1) == occupied)
                    .collect::<Vec<_>>();
                c.select(Axis(1), &columns)
            };
            (pick(true), pick(false))
        };
        let (wo, wv) = split(bra.0, bra.1);
        let (xo, xv) = split(ket.0, ket.1);
        let o = wo.ncols();
        if xo.ncols() != o || o == 0 || wv.ncols() == 0 {
            return None;
        }

        // Löwdin pairing of the occupied overlap, `C^w_o U` and `C^x_o V`, and
        // `V^\dagger = (C^x_o V)^\dagger S C^x_o` for orthonormal ket orbitals.
        let (sv, x_paired, w_paired, _) =
            SameSpinBuild::perform_ortho_and_svd_and_rotate(s, &wo, &xo);
        let st = s.mapv(T::from_real);
        let vh = adjoint(&x_paired).dot(&st).dot(&xo);
        let small = (0..o).filter(|&k| sv[k] < SINGULARTOL).collect::<Vec<_>>();
        if small.len() >= EPSNODES.len() {
            return None;
        }

        // Occupied ket orbitals at `\epsilon`: `C^x_o + B + \epsilon E`, where `B` moves each
        // deformed pair to `\hat S_k` and `E` lifts it by `\epsilon`.
        let mut base = Array2::<T>::zeros(xo.raw_dim());
        let mut lift = Array2::<T>::zeros(xo.raw_dim());
        for &k in &small {
            let (a, b) = (w_paired.column(k), vh.row(k));
            let outer = Array2::from_shape_fn(xo.raw_dim(), |(mu, j)| a[mu] * b[j]);
            let target = if sv[k] <= tol { 0.0 } else { sv[k] };

            base.scaled_add(T::from_real(target - sv[k]), &outer);
            lift += &outer;
        }

        // One point per extrapolation node, weighted by its Lagrange weight at `\epsilon = 0`.
        let points: &[f64] = if small.is_empty() {
            &[0.0]
        } else {
            &EPSNODES[..=small.len()]
        };
        points
            .iter()
            .map(|&e| {
                let weight = points
                    .iter()
                    .filter(|&&p| p != e)
                    .map(|&p| p / (p - e))
                    .product::<f64>();
                let xo = &xo + &base + &lift.mapv(|z| z * T::from_real(e));
                Self::new((&wo, &wv), (&xo, &xv), (&st, f), weight)
            })
            .collect()
    }

    /// Form the intermediates of one point, for invertible `M_0` and `Q`. The ket occupied
    /// orbitals are made biorthogonal to the bra ones, `\tilde\phi = C^x_o G`; the virtual
    /// orbitals keep their remainders outside the occupied spaces, `q_b = (1 - P) u_b` with
    /// `P = \tilde\phi C^{w\dagger}_o S`, and `\tilde v_a = v_a - C^w_o \beta_a^*`; and the ket
    /// remainders are made biorthogonal to the bra ones, `\hat q = q Q^{-1}`.
    /// # Arguments:
    /// - `bra`: Bra occupied and virtual orbitals.
    /// - `ket`: Ket occupied and virtual orbitals.
    /// - `operators`: AO overlap matrix and one-body operator.
    /// - `weight`: Lagrange weight of this point.
    /// # Returns
    /// - `Option<Self>`: The intermediates, or `None` for a singular `M_0` or `Q`.
    fn new(
        bra: (&Array2<T>, &Array2<T>),
        ket: (&Array2<T>, &Array2<T>),
        operators: (&Array2<T>, &Array2<T>),
        weight: f64,
    ) -> Option<Self> {
        let ((wo, wv), (xo, xv), (s, f)) = (bra, ket, operators);
        let (o, v) = (wo.ncols(), wv.ncols());
        let woh = adjoint(wo);

        // Biorthogonal ket occupied orbitals, `\tilde\phi = C^x_o M_0^{-1}`.
        let m0 = woh.dot(s).dot(xo);
        let d = m0.det().ok()?;
        let g = m0.inv().ok()?;
        let bt = xo.dot(&g);

        // Occupied components and virtual remainders on both sides.
        let alpha = woh.dot(s).dot(xv);
        let beta = adjoint(wv).dot(s).dot(&bt);
        let q_orb = xv - &bt.dot(&alpha);
        let vth = adjoint(&(wv - &wo.dot(&adjoint(&beta))));

        // Biorthogonal ket virtual remainders, `\hat q = q Q^{-1}`.
        let q = vth.dot(s).dot(&q_orb);
        let qh = q_orb.dot(&q.inv().ok()?);

        // One-body operator in the biorthogonal frame, blocks `[[f_oo, f_ov], [f_vo, f_vv]]`.
        let mut fb = Array2::<T>::zeros((o + v, o + v));
        fb.slice_mut(s![..o, ..o]).assign(&woh.dot(f).dot(&bt));
        fb.slice_mut(s![..o, o..]).assign(&woh.dot(f).dot(&qh));
        fb.slice_mut(s![o.., ..o]).assign(&vth.dot(f).dot(&bt));
        fb.slice_mut(s![o.., o..]).assign(&vth.dot(f).dot(&qh));

        Some(Self {
            scale: d * T::from_real(weight),
            g,
            alpha,
            beta,
            q,
            f: fb,
        })
    }

    /// Apply the same-spin factors of every point to amplitudes over the ket strings, returning
    /// `F c` and `S c` over the bra strings with at most two replacements, or one when `doubles`
    /// is false. At each point the ket amplitudes are carried into the biorthogonal frame,
    /// `t_{kb} = \sum_j G_{jk} c_{jb}` and `t_{klab} = \sum_{ij} G_{ik} G_{jl} c_{ijab}`; their
    /// virtual orbitals split as `u_b = \tilde\phi\alpha_b + \sum_c \hat q_c Q_{cb}`; the operator
    /// acts by the orthogonal one-body rules with `f`; and the bra virtual orbitals split as
    /// `v_a = \tilde v_a + \phi\beta_a^*`.
    /// # Arguments:
    /// - `points`: Extrapolation points of one spin and parent pair.
    /// - `c`: Ket string amplitudes.
    /// - `doubles`: Whether double-replacement outputs are needed.
    /// # Returns
    /// - `(StringAmplitudes<R>, StringAmplitudes<R>)`: `F c` and `S c`.
    fn apply<R>(
        points: &[Self],
        c: &StringAmplitudes<R>,
        doubles: bool,
    ) -> (StringAmplitudes<R>, StringAmplitudes<R>)
    where
        R: NOCIScalar + From<T>,
    {
        let (o, v) = c.c1.dim();
        let mut sf = StringAmplitudes::zeros((o, v), doubles);
        let mut ss = StringAmplitudes::zeros((o, v), doubles);
        let half = R::from_real(0.5);

        for point in points {
            let cast = |a: ArrayView2<T>| a.mapv(<R as From<T>>::from);
            let (g, alpha, beta, q) = (
                cast(point.g.view()),
                cast(point.alpha.view()),
                cast(point.beta.view()),
                cast(point.q.view()),
            );
            let f_oo = cast(point.f.slice(s![..o, ..o]));
            let f_ov = cast(point.f.slice(s![..o, o..]));
            let f_vo = cast(point.f.slice(s![o.., ..o]));
            let f_vv = cast(point.f.slice(s![o.., o..]));
            let e0 = f_oo.diag().sum();

            // Ket: `G` on each occupied index, `t_{klab} = \sum_{ij} G_{ik} G_{jl} c_{ijab}` as
            // one product per index.
            let t1 = g.t().dot(&c.c1);
            let t2 = c.c2.as_ref().map(|c2| {
                let first = g.t().dot(&c2.view().into_shape((o, o * v * v)).unwrap());
                let first = first.into_shape((o, o, v * v)).unwrap();
                let mut t2 = Array4::from_elem((o, o, v, v), R::from_real(0.0));
                for k in 0..o {
                    let tk = g.t().dot(&first.index_axis(Axis(0), k));
                    t2.index_axis_mut(Axis(0), k)
                        .assign(&tk.into_shape((o, v, v)).unwrap());
                }
                t2
            });

            // Ket: lower the rank through `\alpha`, then carry each virtual index through `Q`,
            // `r_{klcd} = \sum_{ab} Q_{ca} Q_{db} t_{klab}`.
            let z = t2.as_ref().map(|t2| contract_pair(t2, &alpha));
            let mut r0 = c.c0 + (&t1 * &alpha).sum();
            if let Some(z) = &z {
                r0 += half * (z * &alpha).sum();
            }
            let r1 = match &z {
                Some(z) => (&t1 + z).dot(&q.t()),
                None => t1.dot(&q.t()),
            };
            let r2 = t2.map(|t2| {
                let swap = |x: Array4<R>| {
                    x.permuted_axes([0, 1, 3, 2])
                        .as_standard_layout()
                        .into_owned()
                };
                let once = t2.into_shape((o * o * v, v)).unwrap().dot(&q.t());
                let once = swap(once.into_shape((o, o, v, v)).unwrap());
                let twice = once.into_shape((o * o * v, v)).unwrap().dot(&q.t());
                swap(twice.into_shape((o, o, v, v)).unwrap())
            });

            // One-body sigma by the orthogonal rules:
            // `\sigma_0 = e_0 r_0 + \sum_{kc} f_{kc} r_{kc}`,
            // `\sigma_{ia} = f_{ai} r_0 + e_0 r_{ia} + \sum_c f_{ac} r_{ic} - \sum_k f_{ki} r_{ka}
            //  + \sum_{kc} f_{kc} r_{ikac}`.
            let s0 = e0 * r0 + (&f_ov * &r1).sum();
            let mut s1 = f_vo.t().mapv(|z| z * r0) + r1.mapv(|z| z * e0) + r1.dot(&f_vv.t())
                - f_oo.t().dot(&r1);
            if let Some(r2) = &r2 {
                s1 += &contract_pair(r2, &f_ov);
            }

            // `\sigma_{ijab} = e_0 r_{ijab} + P(ab)\sum_c f_{bc} r_{ijac}
            //  - P(ij)\sum_k f_{kj} r_{ikab} + P(ij)P(ab) f_{bj} r_{ia}`.
            let s2 = doubles.then(|| {
                let mut s2 = antisym_outer(&r1, &f_vo);
                if let Some(r2) = &r2 {
                    let vv = r2.view().into_shape((o * o * v, v)).unwrap().dot(&f_vv.t());
                    let vv = vv.into_shape((o, o, v, v)).unwrap();
                    let mut oo = Array4::from_elem((o, o, v, v), R::from_real(0.0));
                    for i in 0..o {
                        let ri = r2.index_axis(Axis(0), i).into_shape((o, v * v)).unwrap();
                        oo.index_axis_mut(Axis(0), i)
                            .assign(&f_oo.t().dot(&ri).into_shape((o, v, v)).unwrap());
                    }
                    s2 += &r2.mapv(|z| z * e0);
                    s2 += &(&vv - &vv.view().permuted_axes([0, 1, 3, 2]));
                    s2 -= &(&oo - &oo.view().permuted_axes([1, 0, 2, 3]));
                }
                s2
            });

            // Bra: raise the rank through `\beta`, for `F c` from `\sigma` and for `S c` from `r`,
            // `w_{ijab} = s_{ijab} + P(ij)P(ab) \beta_{bj} s_{ia}
            //  + s_0 (\beta_{ai}\beta_{bj} - \beta_{aj}\beta_{bi})`.
            let scale = <R as From<T>>::from(point.scale);
            let pairs = doubles.then(|| {
                Array4::from_shape_fn((o, o, v, v), |(i, j, a, b)| {
                    beta[(a, i)] * beta[(b, j)] - beta[(a, j)] * beta[(b, i)]
                })
            });
            for (out, (x0, x1, x2)) in [
                (&mut sf, (s0, &s1, s2.as_ref())),
                (&mut ss, (r0, &r1, r2.as_ref())),
            ] {
                out.c0 += scale * x0;
                out.c1.scaled_add(scale, &(x1 + &beta.t().mapv(|z| z * x0)));
                if let (Some(out2), Some(pairs)) = (out.c2.as_mut(), pairs.as_ref()) {
                    let mut w2 = antisym_outer(x1, &beta) + pairs.mapv(|z| z * x0);
                    if let Some(x2) = x2 {
                        w2 += x2;
                    }
                    out2.scaled_add(scale, &w2);
                }
            }
        }

        (sf, ss)
    }
}

impl<R: NOCIScalar> StringAmplitudes<R> {
    /// Zero amplitudes, with room for doubles when requested.
    /// # Arguments:
    /// - `dims`: Numbers of occupied and virtual orbitals.
    /// - `doubles`: Whether double replacements are held.
    /// # Returns
    /// - `Self`: Zero amplitudes.
    fn zeros(
        dims: (usize, usize),
        doubles: bool,
    ) -> Self {
        let (o, v) = dims;
        let zero = R::from_real(0.0);
        Self {
            c0: zero,
            c1: Array2::from_elem((o, v), zero),
            c2: doubles.then(|| Array4::from_elem((o, o, v, v), zero)),
        }
    }

    /// Collect amplitudes given per parent-local string ID. The determinant of a string differs
    /// from the in-place replacements of its parent reference by the string's phase.
    /// # Arguments:
    /// - `parent`: Number of orbitals, parent occupation and its spin strings.
    /// - `values`: String IDs and amplitudes.
    /// - `doubles`: Whether double replacements may be present.
    /// # Returns
    /// - `Self`: Amplitudes in tensor form.
    fn gather(
        parent: (usize, u128, &[SpinStringData]),
        values: Vec<(usize, R)>,
        doubles: bool,
    ) -> Self {
        let (n, occupation, strings) = parent;
        let nocc = occupation.count_ones() as usize;
        let mut out = Self::zeros((nocc, n - nocc), doubles);

        for (id, value) in values {
            let st = &strings[id];
            let value = value * R::from_real(st.phase);
            let ([i, j], [a, b]) = string_positions(occupation, st);
            match st.cache.rank {
                0 => out.c0 += value,
                1 => out.c1[(i, a)] += value,
                _ => {
                    let c2 = out
                        .c2
                        .as_mut()
                        .expect("doubles requested for double replacements");
                    c2[(i, j, a, b)] += value;
                    c2[(j, i, a, b)] -= value;
                    c2[(i, j, b, a)] -= value;
                    c2[(j, i, b, a)] += value;
                }
            }
        }
        out
    }

    /// Read the amplitude of one string in the determinant convention of the spin strings.
    /// Double replacements read zero when no doubles are held.
    /// # Arguments:
    /// - `occupation`: Parent occupation.
    /// - `st`: String to read.
    /// # Returns
    /// - `R`: Its amplitude.
    fn read(
        &self,
        occupation: u128,
        st: &SpinStringData,
    ) -> R {
        let ([i, j], [a, b]) = string_positions(occupation, st);
        let value = match st.cache.rank {
            0 => self.c0,
            1 => self.c1[(i, a)],
            _ => self
                .c2
                .as_ref()
                .map_or(R::from_real(0.0), |c2| c2[(i, j, a, b)]),
        };
        value * R::from_real(st.phase)
    }
}

/// Build `S^alpha`, `F^alpha`, `S^beta` and `F^beta` tables for one parent pair `QP`.
/// # Arguments:
/// - `spin`: Shared determinant-space factorisation.
/// - `data`: Shared NOCI data containing Wick intermediates.
/// - `storage_plan`: Selected RAM or disk allocation plan for factor tables.
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
    // Resolve ordered target/source parent topologies and spin-string dimensions.
    let target = &spin.parents[target_parent];
    let source = &spin.parents[source_parent];

    let nta = target.areps.len();
    let ntb = target.breps.len();

    let nsa = source.areps.len();
    let nsb = source.breps.len();

    // Map `Q <- P` to the canonical Wick pair and remember whether Q is the left reference.
    let (lp, gp, target_left) = ordered_parent_pair(spin, target_parent, source_parent);

    let pair = data
        .wicks
        .expect("factorised one-body requires Wick intermediates")
        .pair(lp, gp);

    // Allocate checked row-major alpha and beta `(S,F)` factor tables.
    let na = nta
        .checked_mul(nsa)
        .expect("alpha one-body factor length overflow");

    let nb = ntb
        .checked_mul(nsb)
        .expect("beta one-body factor length overflow");

    let mut factors = storage_plan.allocate::<T>(target_parent, source_parent, na, nb);

    // Build and persist all alpha target rows in fixed-rank source evaluation order.
    {
        let out = factors.alpha_mut();
        build_spin_one_body_factors(
            &pair,
            data,
            target.areps.as_slice(),
            &source.a_sources,
            0..nta,
            (target_left, true),
            out,
        );
    }
    factors.flush();

    // Build the symmetric beta tables with beta representatives and evaluation metadata.
    {
        let out = factors.beta_mut();
        build_spin_one_body_factors(
            &pair,
            data,
            target.breps.as_slice(),
            &source.b_sources,
            0..ntb,
            (target_left, false),
            out,
        );
    }
    factors.flush();

    // Select the cheaper alpha-first or beta-first contraction for this block shape.
    let contraction = select_one_body_contraction(
        nta,
        ntb,
        nsa,
        nsb,
        target.entries.len(),
        source.entries.len(),
    );

    // Publish the factor dimensions, contraction plan, and owned storage together.
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
/// - `parent_id`: Parent `P`.
/// # Returns
/// - `OrthogonalOneBodyBlock`: Orthogonal same-parent block with occupation lookup data.
fn build_orthogonal_one_body_block(
    spin: &SpinFactorisation,
    parent_id: usize,
) -> OrthogonalOneBodyBlock {
    // Group retained determinants by their exact parent-local alpha/beta spin-string pair.
    let parent = &spin.parents[parent_id];
    let mut opos = HashMap::new();
    let mut groups = Vec::new();

    for entry in &parent.entries {
        let key = (NOCISpinStringIndex(entry.a), NOCISpinStringIndex(entry.b));
        let group = *opos.entry(key).or_insert_with(|| {
            let index = groups.len();
            groups.push(OrthogonalOccupationGroup {
                targets: Vec::new(),
            });
            index
        });
        groups[group].targets.push(OrthogonalTarget {
            det: entry.det,
            a: entry.a,
        });
    }

    // Store a direct occupation-pair lookup for same-parent Slater-Condon application.
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
        let det = NOCIIndex(entry.det);
        let s = calculate_s_pair_orthogonal(data.space, det, det);
        s_diag[entry.det] = s;
        m_diag[entry.det] = calculate_f_pair_orthogonal(cache, data.space, det, det) + lambda * s;
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
/// - `space`: Candidate determinant space used for occupations and indices.
/// # Returns
/// - `()`: Adds alpha single-excitation Fock contributions into `y`.
fn apply_orthogonal_alpha_singles<T, R>(
    orthogonal: &OrthogonalOneBodyBlock,
    source: NOCIIndex,
    xe: R,
    y: &mut [R],
    space: &NOCISpace<T>,
    cache: &FockMOCache<T>,
    partition: (usize, usize),
) where
    T: NOCIScalar,
    R: NOCIScalar + From<T>,
{
    let (worker, nworker) = partition;
    let nmo = cache.fa.nrows();
    let state = space.spin_string_pair(source);
    let (oa, _) = space.occupations(source);
    // Enumerate `\alpha` excitations `i \to a` with `i` occupied and `a`
    // virtual, leaving the `\beta` occupation unchanged.
    let mut holes = oa;
    while holes != 0 {
        let hole = holes.trailing_zeros() as usize;
        holes &= holes - 1;

        for part in 0..nmo {
            if ((oa >> part) & 1) == 1 {
                continue;
            }
            let target_oa = (oa & !(1u128 << hole)) | (1u128 << part);

            let strings = space.parent_spin_strings(state.parent);
            let Some(&aid) = strings.aids.get(&target_oa) else {
                continue;
            };
            let Some(&opos) = orthogonal.opos.get(&(aid, state.bid)) else {
                continue;
            };

            // Preserve every retained target with this occupation, but let
            // only its assigned worker update the output element.
            for target in &orthogonal.groups[opos].targets {
                if target.a % nworker == worker {
                    let target_det = NOCIIndex(target.det);
                    y[target.det] += <R as From<T>>::from(calculate_f_pair_orthogonal(
                        cache, space, target_det, source,
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
/// - `space`: Candidate determinant space used for occupations and indices.
/// # Returns
/// - `()`: Adds beta single-excitation Fock contributions into `y`.
fn apply_orthogonal_beta_singles<T, R>(
    orthogonal: &OrthogonalOneBodyBlock,
    source: NOCIIndex,
    xe: R,
    y: &mut [R],
    space: &NOCISpace<T>,
    cache: &FockMOCache<T>,
    partition: (usize, usize),
) where
    T: NOCIScalar,
    R: NOCIScalar + From<T>,
{
    let (worker, nworker) = partition;
    let nmo = cache.fb.nrows();
    let state = space.spin_string_pair(source);
    let (_, ob) = space.occupations(source);
    // Enumerate `\beta` excitations `i \to a` while `\alpha` occupation stays fixed.
    let mut holes = ob;
    while holes != 0 {
        let hole = holes.trailing_zeros() as usize;
        holes &= holes - 1;

        for part in 0..nmo {
            if ((ob >> part) & 1) == 1 {
                continue;
            }
            let target_ob = (ob & !(1u128 << hole)) | (1u128 << part);

            let strings = space.parent_spin_strings(state.parent);
            let Some(&bid) = strings.bids.get(&target_ob) else {
                continue;
            };
            let Some(&opos) = orthogonal.opos.get(&(state.aid, bid)) else {
                continue;
            };

            // Preserve duplicate retained targets under worker ownership.
            for target in &orthogonal.groups[opos].targets {
                if target.a % nworker == worker {
                    let target_det = NOCIIndex(target.det);
                    y[target.det] += <R as From<T>>::from(calculate_f_pair_orthogonal(
                        cache, space, target_det, source,
                    )) * xe;
                }
            }
        }
    }
}

/// Build same-spin `S` and `F` factor rows from prepared Wick pair batches.
/// Independent source spin strings are evaluated together so the widest fixed-rank SIMD kernel can be used.
/// # Arguments:
/// - `pair`: Wick intermediates for the ordered parent pair.
/// - `data`: Shared NOCI determinant data.
/// - `target_reps`: Target spin-string payloads.
/// - `sources`: Source representatives in packed evaluation order.
/// - `rows`: Target-row range to fill.
/// - `flags`: Whether targets are left determinants and whether to build alpha factors.
/// - `out`: Mutable row-major overlap and Fock factor tables.
/// # Returns
/// - `()`: Fills `out` factor tables.
fn build_spin_one_body_factors<T: NOCIScalar>(
    pair: &WicksPairView<'_, T>,
    data: &NOCIData<'_, T>,
    target_reps: &[SpinStringData],
    sources: &SpinStringSources,
    rows: Range<usize>,
    flags: (bool, bool),
    out: (&mut [T], &mut [T]),
) {
    let nsource = sources.len();
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
                    target_rep,
                    sources,
                    flags,
                    scratch,
                    (&mut *srow, &mut *frow),
                );
            },
        );
}

/// Build one same-spin `S` and `F` factor row from prepared Wick pair batches.
/// # Arguments:
/// - `pair`: Wick intermediates for the ordered parent pair.
/// - `data`: Shared NOCI determinant data used by the generic fallback.
/// - `target`: Target spin-string payload.
/// - `sources`: Source representatives in packed evaluation order.
/// - `flags`: Whether target is left determinant and whether to evaluate alpha factors.
/// - `scratch`: Reusable spin-resolved Wick evaluator workspace.
/// - `out`: Output same-spin overlap and generalised-Fock factor rows.
/// # Returns
/// - `()`: Fills `srow` and `frow` for the target spin string.
fn build_spin_one_body_factor_row<T: NOCIScalar>(
    pair: &WicksPairView<'_, T>,
    data: &NOCIData<'_, T>,
    target: SpinStringData,
    sources: &SpinStringSources,
    flags: (bool, bool),
    scratch: &mut WickScratchSpin<T>,
    out: (&mut [T], &mut [T]),
) {
    let (target_left, alpha) = flags;
    let (overlap, fock) = out;

    let (w, scratch) = if alpha {
        (&pair.aa, &mut scratch.aa)
    } else {
        (&pair.bb, &mut scratch.bb)
    };

    let row = SpinFactorRow {
        target,
        sources,
        target_left,
        out: [overlap, fock],
    };
    xw_spin_factor_row(&OneBodyFactor { tol: data.tol }, w, row, scratch);
}

/// Select alpha-first or beta-first contraction from dense structural costs.
/// # Arguments:
/// - `nta`: Number of target alpha spin strings.
/// - `ntb`: Number of target beta spin strings.
/// - `nsa`: Number of source alpha spin strings.
/// - `nsb`: Number of source beta spin strings.
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

/// Whether an ordered parent pair `QP` can be applied in contraction form: two different parents
/// whose determinants have at most `CONTRACTRANK` replacements in total.
/// # Arguments:
/// - `spin`: Shared determinant-space factorisation.
/// - `parents`: Target parent `Q` and source parent `P`.
/// # Returns
/// - `bool`: Whether the class decomposition of the contraction form covers the pair.
fn contractible(
    spin: &SpinFactorisation,
    parents: (usize, usize),
) -> bool {
    let (target_parent, source_parent) = parents;
    let within = |p: usize| {
        let parent = &spin.parents[p];
        parent.entries.iter().all(|e| {
            (parent.areps[e.a].cache.rank + parent.breps[e.b].cache.rank) as usize <= CONTRACTRANK
        })
    };
    target_parent != source_parent && within(target_parent) && within(source_parent)
}

/// Positions of the replaced occupied orbitals among the occupied orbitals, and of the replacing
/// virtual orbitals among the virtual orbitals, of a string with at most two replacements.
/// # Arguments:
/// - `occupation`: Parent occupation.
/// - `st`: Spin string.
/// # Returns
/// - `([usize; 2], [usize; 2])`: Hole and particle positions in ascending order.
fn string_positions(
    occupation: u128,
    st: &SpinStringData,
) -> ([usize; 2], [usize; 2]) {
    let below = |k: usize| (occupation & ((1u128 << k) - 1)).count_ones() as usize;
    let mut holes = [0; 2];
    let mut parts = [0; 2];
    for r in 0..st.cache.rank as usize {
        holes[r] = below(st.cache.holes[r] as usize);
        let p = st.cache.particles[r] as usize;
        parts[r] = p - below(p);
    }
    (holes, parts)
}

/// Contract the second occupied and virtual indices of a double, `z_{ka} = \sum_{lb} t_{klab} m_{lb}`.
/// # Arguments:
/// - `t2`: Amplitudes `t_{klab}`.
/// - `m`: Occupied-virtual matrix `m_{lb}`.
/// # Returns
/// - `Array2<R>`: `z_{ka}`.
fn contract_pair<R: NOCIScalar>(
    t2: &Array4<R>,
    m: &Array2<R>,
) -> Array2<R> {
    let (o, _, v, _) = t2.dim();
    let t = t2.view().permuted_axes([0, 2, 1, 3]);
    let t = t.as_standard_layout();
    let z = t
        .view()
        .into_shape((o * v, o * v))
        .unwrap()
        .dot(&m.view().into_shape(o * v).unwrap());
    z.into_shape((o, v)).unwrap()
}

/// Antisymmetrised outer product
/// `P(ij)P(ab)\, x_{ia} y_{bj} = x_{ia}y_{bj} - x_{ja}y_{bi} - x_{ib}y_{aj} + x_{jb}y_{ai}`.
/// # Arguments:
/// - `x`: Occupied-virtual matrix `x_{ia}`.
/// - `y`: Virtual-occupied matrix `y_{bj}`.
/// # Returns
/// - `Array4<R>`: The antisymmetric double.
fn antisym_outer<R: NOCIScalar>(
    x: &Array2<R>,
    y: &Array2<R>,
) -> Array4<R> {
    let (o, v) = x.dim();
    Array4::from_shape_fn((o, o, v, v), |(i, j, a, b)| {
        x[(i, a)] * y[(b, j)] - x[(j, a)] * y[(b, i)] - x[(i, b)] * y[(a, j)]
            + x[(j, b)] * y[(a, i)]
    })
}
