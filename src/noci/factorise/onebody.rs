// noci/factorise/onebody.rs
//! Spin-factorised one-body NOCI operator contractions.

// Standard library imports.
use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;
use std::sync::Mutex;

// External crate imports.
use ndarray::{Array1, Array2, Array4, Axis, s};
use ndarray_linalg::{Determinant, Inverse, SVD};
use rayon::prelude::*;

// Crate-root imports.
use crate::NOCIScalar;
use crate::determinant::{
    NOCIIndex, NOCISpace, NOCISpinStringIndex, ParentDeterminant, SpinStringData,
};
use crate::elements::calculate_f_pair_orthogonal;
use crate::elements::calculate_s_pair_orthogonal;
use crate::elements::nonorthogonalwicks::{
    OneBodyFactor, SINGULARTOL, SpinFactorRow, SpinStringSources, WickScratchSpin, WicksPairView,
    xw_spin_factor_row,
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

/// Nonorthogonal parent pair `QP` applied in contraction form: every same-spin factor block is
/// applied to whole amplitude vectors through orbital-indexed matrices, never element by element.
struct ContractedOneBodyBlock<T: NOCIScalar> {
    /// Target parent `Q`, the bra.
    target_parent: usize,
    /// Source parent `P`, the ket.
    source_parent: usize,
    /// Alpha factors `F^\alpha` and `S^\alpha`.
    alpha: SpinContraction<T>,
    /// Beta factors `F^\beta` and `S^\beta`.
    beta: SpinContraction<T>,
}

/// Same-spin one-body and overlap factors of one ordered parent pair in contraction form.
/// Near-zero occupied overlaps are deformed to `\tilde S_k + \epsilon`, every factor is then a
/// polynomial of degree at most the number of deformed pairs in `\epsilon`, and the factors at
/// `\epsilon = 0` follow exactly by Lagrange extrapolation from well-conditioned points.
struct SpinContraction<T: NOCIScalar> {
    /// Extrapolation points, a single undeformed point when no overlap is near zero.
    nodes: Vec<ContractionNode<T>>,
}

/// Intermediates of one same-spin parent pair at one extrapolation point, with bra occupied
/// orbitals `\phi_i`, biorthogonal ket occupied orbitals `\tilde\phi = C^x_o G`, bra virtual
/// remainders `\tilde v_a` and biorthogonal ket virtual remainders `\hat q_c`.
struct ContractionNode<T: NOCIScalar> {
    /// Lagrange weight of this point in the extrapolation to `\epsilon = 0`.
    weight: f64,
    /// `d = \det M_0` with `M_0 = C^{w\dagger}_o S C^x_o`.
    d: T,
    /// `G = M_0^{-1}`.
    g: Array2<T>,
    /// `\alpha_{kb} = \phi_k^\dagger S u_b`, the ket virtual `u_b` along `\tilde\phi_k`.
    alpha: Array2<T>,
    /// `\beta_{ai} = v_a^\dagger S \tilde\phi_i`, the bra virtual `v_a` along `\phi_i`.
    beta: Array2<T>,
    /// `Q_{ab} = \tilde v_a^\dagger S q_b`, the overlap of the virtual remainders.
    q: Array2<T>,
    /// `f_{ik} = \phi_i^\dagger F \tilde\phi_k`.
    f_oo: Array2<T>,
    /// `f_{ac} = \tilde v_a^\dagger F \hat q_c`.
    f_vv: Array2<T>,
    /// `f_{ic} = \phi_i^\dagger F \hat q_c`.
    f_ov: Array2<T>,
    /// `f_{ak} = \tilde v_a^\dagger F \tilde\phi_k`.
    f_vo: Array2<T>,
    /// `e_0 = \sum_i f_{ii}`.
    e0: T,
}

/// One spin of a parent in contraction form: its occupied-first orbitals and spin strings.
struct ContractedFrame<T: NOCIScalar> {
    /// Occupied-first molecular orbital coefficients.
    c: Array2<T>,
    /// Number of occupied orbitals.
    nocc: usize,
    /// Spin strings in parent-local ID order.
    strings: Vec<ContractedString>,
}

/// One parent-local spin string as in-place replacements of its parent reference: occupied
/// positions `i_1 < i_2` replaced by virtual positions `a_1 < a_2`.
#[derive(Clone, Copy)]
struct ContractedString {
    /// Number of replacements.
    rank: usize,
    /// Replaced occupied positions.
    holes: [usize; 2],
    /// Replacing virtual positions.
    parts: [usize; 2],
    /// Sign of the in-place determinant relative to the ascending-order determinant.
    sign: f64,
}

/// Same-spin amplitudes over the strings of one parent with at most two replacements:
/// `c_0`, `c_{ia}` and the antisymmetric `c_{ijab}`, absent when every double is zero.
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
    /// Alpha and beta frames of every parent whose spin strings the contraction form covers.
    frames: Vec<Option<[ContractedFrame<T>; 2]>>,
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

        // Spin frames of the parents whose strings have at most `CONTRACTRANK` replacements.
        let frames = data
            .space
            .parents()
            .iter()
            .zip(&spin.parents)
            .map(|(parent, strings)| contracted_frames(parent, strings))
            .collect::<Vec<_>>();

        // Choose a representation per ordered parent pair: exact orthogonal Slater-Condon
        // blocks, the contraction form, transient evaluation, or persistent spin factors.
        for target_parent in 0..nparent {
            for source_parent in 0..nparent {
                let contracted = ContractedOneBodyBlock::new(
                    (target_parent, source_parent),
                    &spin,
                    &frames,
                    data,
                    fock,
                );

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
            frames,
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
        let frames = data
            .space
            .parents()
            .iter()
            .zip(&spin.parents)
            .map(|(parent, strings)| contracted_frames(parent, strings))
            .collect::<Vec<_>>();

        // Each nonorthogonal parent pair stores `\alpha` and `\beta` `S` and `F` factors;
        // same-parent orthogonal blocks and contracted parent pairs need no factor table.
        let mut nentries = 0usize;
        for target_parent in 0..nparent {
            let target = &spin.parents[target_parent];
            for source_parent in 0..nparent {
                if (target_parent == source_parent
                    && fock.fock_mocache[target_parent].orthogonal_slater_condon)
                    || contractible(&spin, &frames, (target_parent, source_parent))
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
                    Some(block.updates(&self.spin, &self.frames, xs, lambda))
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
    /// Prepare the contraction form of one ordered parent pair `QP`, when both parents have
    /// frames, every determinant has at most `CONTRACTRANK` replacements in total, and
    /// every near-zero occupied overlap can be extrapolated.
    /// # Arguments:
    /// - `parents`: Target parent `Q` and source parent `P`.
    /// - `spin`: Shared determinant-space factorisation.
    /// - `frames`: Contraction frames of every parent.
    /// - `data`: Shared NOCI data holding the AO overlap and the zero-overlap tolerance.
    /// - `fock`: Current generalised Fock matrices in the AO basis.
    /// # Returns
    /// - `Option<Self>`: The contracted block, or `None` when the pair keeps the factor rows.
    fn new(
        parents: (usize, usize),
        spin: &SpinFactorisation,
        frames: &[Option<[ContractedFrame<T>; 2]>],
        data: &NOCIData<'_, T>,
        fock: &FockData<'_, T>,
    ) -> Option<Self> {
        let (target_parent, source_parent) = parents;
        if !contractible(spin, frames, parents) {
            return None;
        }
        let [ta, tb] = frames[target_parent].as_ref()?;
        let [sa, sb] = frames[source_parent].as_ref()?;

        Some(Self {
            target_parent,
            source_parent,
            alpha: SpinContraction::new((ta, sa), &data.ao.s, fock.fa, data.tol)?,
            beta: SpinContraction::new((tb, sb), &data.ao.s, fock.fb, data.tol)?,
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
    /// - `frames`: Contraction frames of every parent.
    /// - `x`: Source determinant vector.
    /// - `lambda`: Scalar overlap shift.
    /// # Returns
    /// - `Vec<(usize, R)>`: Target determinant and the value to add to it.
    fn updates<R>(
        &self,
        spin: &SpinFactorisation,
        frames: &[Option<[ContractedFrame<T>; 2]>],
        x: &[R],
        lambda: R,
    ) -> Vec<(usize, R)>
    where
        R: NOCIScalar + From<T>,
    {
        let source = &spin.parents[self.source_parent];
        let target = &spin.parents[self.target_parent];
        let [sa, sb] = frames[self.source_parent]
            .as_ref()
            .expect("contracted source frame");
        let [ta, tb] = frames[self.target_parent]
            .as_ref()
            .expect("contracted target frame");
        let rank = |e: &FactorEntry| (sa.strings[e.a].rank, sb.strings[e.b].rank);

        // Class 1: the beta-reference column.
        let column = StringAmplitudes::gather(
            sa,
            source
                .entries
                .iter()
                .filter(|e| rank(e).1 == 0)
                .map(|e| (e.a, x[e.det])),
            true,
        );
        let (fa1, sa1) = self.alpha.apply(&column, true);
        let (fb0, sb0) = self.beta.apply(&StringAmplitudes::<R>::reference(sb), true);

        // Class 2: the alpha-reference row over excited beta strings.
        let row = StringAmplitudes::gather(
            sb,
            source
                .entries
                .iter()
                .filter(|e| rank(e).0 == 0 && rank(e).1 > 0)
                .map(|e| (e.b, x[e.det])),
            true,
        );
        let (fb2, sb2) = self.beta.apply(&row, true);
        let (fa0, sa0) = self
            .alpha
            .apply(&StringAmplitudes::<R>::reference(sa), true);

        // Class 3: one column of `D^{11}` per beta single, with the single-replacement factors
        // of its alpha apply and of the beta single.
        let mut grouped: HashMap<usize, Vec<(usize, R)>> = HashMap::new();
        for e in source.entries.iter().filter(|e| rank(e) == (1, 1)) {
            grouped.entry(e.b).or_default().push((e.a, x[e.det]));
        }
        let grouped = grouped.into_iter().collect::<Vec<_>>();
        let columns = grouped
            .par_iter()
            .map(|(b, column)| {
                let alpha = StringAmplitudes::gather(sa, column.iter().copied(), false);
                let beta = StringAmplitudes::gather(sb, [(*b, R::one())], false);
                (
                    self.alpha.apply(&alpha, false),
                    self.beta.apply(&beta, false),
                )
            })
            .collect::<Vec<_>>();

        // Beta-reference targets: alpha applies of `D^{11} w^S` and `D^{11} w^F`.
        let weighted = |w: &dyn Fn(usize) -> R| {
            StringAmplitudes::gather(
                sa,
                grouped.iter().enumerate().flat_map(|(c, (_, column))| {
                    let wc = w(c);
                    column.iter().map(move |&(a, value)| (a, value * wc))
                }),
                false,
            )
        };
        let (fws, sws) = self.alpha.apply(&weighted(&|c| columns[c].1.1.c0), true);
        let (_, swf) = self.alpha.apply(&weighted(&|c| columns[c].1.0.c0), true);

        // Alpha-reference targets: beta applies of the reference rows of `F^\alpha D^{11}` and
        // `S^\alpha D^{11}`.
        let reference_row = |pick: &dyn Fn(usize) -> R| {
            StringAmplitudes::gather(
                sb,
                grouped.iter().enumerate().map(|(c, (b, _))| (*b, pick(c))),
                false,
            )
        };
        let (_, suf) = self
            .beta
            .apply(&reference_row(&|c| columns[c].0.0.c0), true);
        let (fus, sus) = self
            .beta
            .apply(&reference_row(&|c| columns[c].0.1.c0), true);

        // Single-single targets: `Y^F = X^F (S^\beta)^T + X^S (F^\beta)^T` and
        // `Y^S = X^S (S^\beta)^T` over the target singles.
        let asingles = target_singles(ta);
        let bsingles = target_singles(tb);
        let gather_rows =
            |rows: &[usize], frame: &ContractedFrame<T>, amps: Vec<&StringAmplitudes<R>>| {
                Array2::from_shape_fn((rows.len(), amps.len()), |(r, c)| {
                    amps[c].read(&frame.strings[rows[r]])
                })
            };
        let xf = gather_rows(&asingles, ta, columns.iter().map(|c| &c.0.0).collect());
        let xs = gather_rows(&asingles, ta, columns.iter().map(|c| &c.0.1).collect());
        let bf = gather_rows(&bsingles, tb, columns.iter().map(|c| &c.1.0).collect());
        let bs = gather_rows(&bsingles, tb, columns.iter().map(|c| &c.1.1).collect());
        let yf = xf.dot(&bs.t()) + xs.dot(&bf.t());
        let ys = xs.dot(&bs.t());
        let arow = single_rows(ta, &asingles);
        let brow = single_rows(tb, &bsingles);

        // Every class at the target strings.
        let (fa1, sa1, fa0, sa0) = (
            fa1.read_all(ta),
            sa1.read_all(ta),
            fa0.read_all(ta),
            sa0.read_all(ta),
        );
        let (fws, sws, swf) = (fws.read_all(ta), sws.read_all(ta), swf.read_all(ta));
        let (fb0, sb0, fb2, sb2) = (
            fb0.read_all(tb),
            sb0.read_all(tb),
            fb2.read_all(tb),
            sb2.read_all(tb),
        );
        let (suf, fus, sus) = (suf.read_all(tb), fus.read_all(tb), sus.read_all(tb));

        // `Y^Q_{\bar a\bar b} = F_{\bar a\bar b} + \lambda S_{\bar a\bar b}` at every target determinant.
        target
            .entries
            .par_iter()
            .with_min_len(DOTCHUNK)
            .map(|e| {
                let (i, j) = (e.a, e.b);
                let mut f = fa1[i] * sb0[j] + sa1[i] * fb0[j] + fa0[i] * sb2[j] + sa0[i] * fb2[j];
                let mut s = sa1[i] * sb0[j] + sa0[i] * sb2[j];
                match (ta.strings[i].rank, tb.strings[j].rank) {
                    (_, 0) => {
                        f += fws[i] + swf[i];
                        s += sws[i];
                    }
                    (0, _) => {
                        f += suf[j] + fus[j];
                        s += sus[j];
                    }
                    (1, 1) => {
                        f += yf[(arow[i], brow[j])];
                        s += ys[(arow[i], brow[j])];
                    }
                    _ => {}
                }
                (e.det, f + lambda * s)
            })
            .collect()
    }
}

impl<T: NOCIScalar> SpinContraction<T> {
    /// Prepare the same-spin factors between the strings of a bra parent `w` and a ket parent
    /// `x`. With the singular value decomposition `M_0 = U \tilde S V^\dagger`, every pair with
    /// `\tilde S_k < \mathtt{SINGULARTOL}` is deformed along `(C^w_o u_k) v_k^\dagger` to
    /// `\hat S_k + \epsilon`, where `\hat S_k = 0` when `\tilde S_k \le \mathtt{tol}`, as in the
    /// Wick kernels, and `\hat S_k = \tilde S_k` otherwise.
    /// # Arguments:
    /// - `frames`: Bra and ket frames of this spin.
    /// - `s`: AO overlap matrix.
    /// - `f`: One-body operator of this spin in the AO basis.
    /// - `tol`: Singular values at or below this are exact zeros.
    /// # Returns
    /// - `Option<Self>`: The factors, or `None` when the pair cannot be extrapolated.
    fn new(
        frames: (&ContractedFrame<T>, &ContractedFrame<T>),
        s: &Array2<f64>,
        f: &Array2<T>,
        tol: f64,
    ) -> Option<Self> {
        let (w, x) = frames;
        let o = w.nocc;
        if x.nocc != o {
            return None;
        }
        let s = s.mapv(T::from_real);
        let (wo, wv) = (w.c.slice(s![.., ..o]), w.c.slice(s![.., o..]));
        let (xo, xv) = (x.c.slice(s![.., ..o]), x.c.slice(s![.., o..]));

        // Löwdin pairing of the occupied overlap `M_0 = C^{w\dagger}_o S C^x_o`.
        let m0 = adjoint(&wo.to_owned()).dot(&s).dot(&xo);
        let (u, sv, vh) = m0.svd(true, true).ok()?;
        let (u, vh) = (u?, vh?);
        let small = (0..o).filter(|&k| sv[k] < SINGULARTOL).collect::<Vec<_>>();
        if small.len() >= EPSNODES.len() {
            return None;
        }

        // Occupied ket columns at `\epsilon`: `C^x_o + B + \epsilon E`, where `B` moves each
        // deformed pair to `\hat S_k` and `E` lifts it by `\epsilon`.
        let mut base = Array2::<T>::zeros(xo.raw_dim());
        let mut lift = Array2::<T>::zeros(xo.raw_dim());
        for &k in &small {
            let a = wo.dot(&u.column(k));
            let b = vh.row(k);
            let outer = Array2::from_shape_fn(xo.raw_dim(), |(mu, j)| a[mu] * b[j]);
            let target = if sv[k] <= tol { 0.0 } else { sv[k] };
            base.scaled_add(T::from_real(target - sv[k]), &outer);
            lift += &outer;
        }

        // Lagrange weights of the points for extrapolation to `\epsilon = 0`.
        let points: &[f64] = if small.is_empty() {
            &[0.0]
        } else {
            &EPSNODES[..=small.len()]
        };
        let nodes = points
            .iter()
            .map(|&e| {
                let weight = points
                    .iter()
                    .filter(|&&p| p != e)
                    .map(|&p| p / (p - e))
                    .product::<f64>();
                let xo = &xo + &base + &lift.mapv(|z| z * T::from_real(e));
                ContractionNode::new(
                    (&wo.to_owned(), &wv.to_owned()),
                    (&xo, &xv.to_owned()),
                    &s,
                    f,
                    weight,
                )
            })
            .collect::<Option<Vec<_>>>()?;

        Some(Self { nodes })
    }

    /// Apply the same-spin factors to amplitudes over the ket strings, returning
    /// `\sigma^F = F c` and `\sigma^S = S c` over the bra strings with at most two replacements,
    /// or at most one when `doubles` is false.
    /// # Arguments:
    /// - `c`: Ket string amplitudes.
    /// - `doubles`: Whether double-replacement outputs are needed.
    /// # Returns
    /// - `(StringAmplitudes<R>, StringAmplitudes<R>)`: `F c` and `S c`.
    fn apply<R>(
        &self,
        c: &StringAmplitudes<R>,
        doubles: bool,
    ) -> (StringAmplitudes<R>, StringAmplitudes<R>)
    where
        R: NOCIScalar + From<T>,
    {
        let mut total: Option<(StringAmplitudes<R>, StringAmplitudes<R>)> = None;
        for node in &self.nodes {
            let scale = <R as From<T>>::from(node.d) * R::from_real(node.weight);
            let (f, s) = node.apply(c, doubles);
            total = Some(match total {
                None => (f.scaled(scale), s.scaled(scale)),
                Some((tf, ts)) => (tf.add_scaled(&f, scale), ts.add_scaled(&s, scale)),
            });
        }
        total.expect("a contraction has at least one point")
    }
}

impl<T: NOCIScalar> ContractionNode<T> {
    /// Form the intermediates of one point from the bra and ket orbitals, for invertible
    /// `M_0` and `Q`. The ket occupied orbitals are made biorthogonal to the bra ones,
    /// `\tilde\phi = C^x_o G`, the virtual orbitals keep only their remainders outside the
    /// occupied spaces, `q_b = (1 - P) u_b` with `P = \tilde\phi C^{w\dagger}_o S` and
    /// `\tilde v_a = v_a - C^w_o \beta_a^*`, and the ket remainders are made biorthogonal to the
    /// bra ones, `\hat q = q Q^{-1}`.
    /// # Arguments:
    /// - `bra`: Bra occupied and virtual orbitals.
    /// - `ket`: Ket occupied and virtual orbitals.
    /// - `s`: AO overlap matrix.
    /// - `f`: One-body operator in the AO basis.
    /// - `weight`: Lagrange weight of this point.
    /// # Returns
    /// - `Option<Self>`: The intermediates, or `None` for a singular `M_0` or `Q`.
    fn new(
        bra: (&Array2<T>, &Array2<T>),
        ket: (&Array2<T>, &Array2<T>),
        s: &Array2<T>,
        f: &Array2<T>,
        weight: f64,
    ) -> Option<Self> {
        let ((wo, wv), (xo, xv)) = (bra, ket);
        let woh_s = adjoint(wo).dot(s);

        // Biorthogonal ket occupied orbitals.
        let m0 = woh_s.dot(xo);
        let d = m0.det().ok()?;
        let g = m0.inv().ok()?;
        let bt = xo.dot(&g);

        // Occupied components and virtual remainders on both sides.
        let alpha = woh_s.dot(xv);
        let beta = adjoint(wv).dot(s).dot(&bt);
        let q_orb = xv - &bt.dot(&alpha);
        let vt = wv - &wo.dot(&adjoint(&beta));
        let vth = adjoint(&vt);
        let q = vth.dot(s).dot(&q_orb);
        let qh = q_orb.dot(&q.inv().ok()?);

        // One-body operator in the biorthogonal frame.
        let f_oo = adjoint(wo).dot(f).dot(&bt);
        let e0 = f_oo.diag().sum();
        Some(Self {
            weight,
            d,
            g,
            alpha,
            beta,
            q,
            f_vv: vth.dot(f).dot(&qh),
            f_ov: adjoint(wo).dot(f).dot(&qh),
            f_vo: vth.dot(f).dot(&bt),
            f_oo,
            e0,
        })
    }

    /// Apply this point's factors without `d`. Ket amplitudes are carried into the biorthogonal
    /// frame, `t_{kb} = \sum_j G_{jk} c_{jb}` and `t_{klab} = \sum_{ij} G_{ik} G_{jl} c_{ijab}`,
    /// their virtual orbitals are split as `u_b = \tilde\phi\alpha_b + \sum_c \hat q_c Q_{cb}`,
    /// the one-body operator acts by the orthogonal rules with `f`, and the bra virtual orbitals
    /// are split as `v_a = \tilde v_a + \phi\beta_a^*`.
    /// # Arguments:
    /// - `c`: Ket string amplitudes.
    /// - `doubles`: Whether double-replacement outputs are needed.
    /// # Returns
    /// - `(StringAmplitudes<R>, StringAmplitudes<R>)`: `F c / d` and `S c / d`.
    fn apply<R>(
        &self,
        c: &StringAmplitudes<R>,
        doubles: bool,
    ) -> (StringAmplitudes<R>, StringAmplitudes<R>)
    where
        R: NOCIScalar + From<T>,
    {
        let cast = |a: &Array2<T>| a.mapv(<R as From<T>>::from);
        let (g, alpha, beta, q) = (
            cast(&self.g),
            cast(&self.alpha),
            cast(&self.beta),
            cast(&self.q),
        );
        let (f_oo, f_vv, f_ov, f_vo) = (
            cast(&self.f_oo),
            cast(&self.f_vv),
            cast(&self.f_ov),
            cast(&self.f_vo),
        );
        let e0 = <R as From<T>>::from(self.e0);
        let half = R::from_real(0.5);

        // Ket: x frame to the biorthogonal frame, lowering the rank through `\alpha`.
        let t1 = g.t().dot(&c.c1);
        let t2 = c.c2.as_ref().map(|c2| occupied_transform(c2, &g));
        let z = t2.as_ref().map(|t2| contract_pair(t2, &alpha));
        let mut r0 = c.c0 + (&t1 * &alpha).sum();
        if let Some(z) = &z {
            r0 += half * (z * &alpha).sum();
        }
        let r1 = match &z {
            Some(z) => (&t1 + z).dot(&q.t()),
            None => t1.dot(&q.t()),
        };
        let r2 = t2.map(|t2| virtual_transform(&t2, &q));

        // One-body sigma by the orthogonal rules:
        // `\sigma_0 = e_0 r_0 + \sum_{kc} f_{kc} r_{kc}`,
        // `\sigma_{ia} = f_{ai} r_0 + e_0 r_{ia} + \sum_c f_{ac} r_{ic} - \sum_k f_{ki} r_{ka}
        //  + \sum_{kc} f_{kc} r_{ikac}`,
        // `\sigma_{ijab} = e_0 r_{ijab} + P(ab)\sum_c f_{bc} r_{ijac} - P(ij)\sum_k f_{kj} r_{ikab}
        //  + P(ij)P(ab) f_{bj} r_{ia}`.
        let s0 = e0 * r0 + (&f_ov * &r1).sum();
        let mut s1 =
            f_vo.t().mapv(|z| z * r0) + r1.mapv(|z| z * e0) + r1.dot(&f_vv.t()) - f_oo.t().dot(&r1);
        if let Some(r2) = &r2 {
            s1 += &contract_pair(r2, &f_ov);
        }
        let s2 = doubles.then(|| {
            let mut s2 = antisym_outer(&r1, &f_vo);
            if let Some(r2) = &r2 {
                let vv = virtual_one(r2, &f_vv);
                let oo = occupied_one(r2, &f_oo);
                s2 =
                    s2 + r2.mapv(|z| z * e0) + &vv - &swap_virtual(&vv) - &oo + &swap_occupied(&oo);
            }
            s2
        });

        // Bra: biorthogonal frame to the w frame, raising the rank through `\beta`.
        let bra = |s0: R, s1: &Array2<R>, s2: Option<&Array4<R>>| {
            let w2 = doubles.then(|| {
                let mut w2 = antisym_outer(s1, &beta) + pair_det(&beta).mapv(|z| z * s0);
                if let Some(s2) = s2 {
                    w2 += s2;
                }
                w2
            });
            StringAmplitudes {
                c0: s0,
                c1: s1 + &beta.t().mapv(|z| z * s0),
                c2: w2,
            }
        };
        (bra(s0, &s1, s2.as_ref()), bra(r0, &r1, r2.as_ref()))
    }
}

impl<R: NOCIScalar> StringAmplitudes<R> {
    /// Unit amplitude on the reference string of a frame.
    /// # Arguments:
    /// - `frame`: Frame of the parent and spin.
    /// # Returns
    /// - `Self`: `c_0 = 1` and no replacements.
    fn reference(frame: &ContractedFrame<impl NOCIScalar>) -> Self {
        let (o, v) = (frame.nocc, frame.c.ncols() - frame.nocc);
        Self {
            c0: R::one(),
            c1: Array2::from_elem((o, v), R::from_real(0.0)),
            c2: None,
        }
    }

    /// Collect amplitudes given per parent-local string ID, converting the ascending-order
    /// determinant convention to in-place replacements.
    /// # Arguments:
    /// - `frame`: Frame of the parent and spin the strings belong to.
    /// - `values`: String IDs and amplitudes.
    /// - `doubles`: Whether double replacements may be present.
    /// # Returns
    /// - `Self`: Amplitudes in tensor form.
    fn gather(
        frame: &ContractedFrame<impl NOCIScalar>,
        values: impl IntoIterator<Item = (usize, R)>,
        doubles: bool,
    ) -> Self {
        let (o, v) = (frame.nocc, frame.c.ncols() - frame.nocc);
        let zero = R::from_real(0.0);
        let mut out = Self {
            c0: zero,
            c1: Array2::from_elem((o, v), zero),
            c2: doubles.then(|| Array4::from_elem((o, o, v, v), zero)),
        };
        for (id, value) in values {
            let st = frame.strings[id];
            let value = value * R::from_real(st.sign);
            let ([i, j], [a, b]) = (st.holes, st.parts);
            match st.rank {
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

    /// Read the amplitude of every string of a frame, in parent-local ID order and in the
    /// ascending-order determinant convention.
    /// # Arguments:
    /// - `frame`: Frame of the parent and spin to read.
    /// # Returns
    /// - `Vec<R>`: One amplitude per string.
    fn read_all(
        &self,
        frame: &ContractedFrame<impl NOCIScalar>,
    ) -> Vec<R> {
        frame.strings.iter().map(|st| self.read(st)).collect()
    }

    /// Read the amplitude of one string in the ascending-order determinant convention.
    /// Double replacements read zero when no doubles are held.
    /// # Arguments:
    /// - `st`: String to read.
    /// # Returns
    /// - `R`: Its amplitude.
    fn read(
        &self,
        st: &ContractedString,
    ) -> R {
        let ([i, j], [a, b]) = (st.holes, st.parts);
        let value = match st.rank {
            0 => self.c0,
            1 => self.c1[(i, a)],
            _ => self
                .c2
                .as_ref()
                .map_or(R::from_real(0.0), |c2| c2[(i, j, a, b)]),
        };
        value * R::from_real(st.sign)
    }

    /// Scale every amplitude.
    /// # Arguments:
    /// - `scale`: Scalar factor.
    /// # Returns
    /// - `Self`: Scaled amplitudes.
    fn scaled(
        self,
        scale: R,
    ) -> Self {
        Self {
            c0: self.c0 * scale,
            c1: self.c1.mapv(|z| z * scale),
            c2: self.c2.map(|c2| c2.mapv(|z| z * scale)),
        }
    }

    /// Add scaled amplitudes, `self + scale\,other`.
    /// # Arguments:
    /// - `other`: Amplitudes to add.
    /// - `scale`: Scalar factor of `other`.
    /// # Returns
    /// - `Self`: Sum.
    fn add_scaled(
        mut self,
        other: &Self,
        scale: R,
    ) -> Self {
        self.c0 += other.c0 * scale;
        self.c1.scaled_add(scale, &other.c1);
        match (&mut self.c2, &other.c2) {
            (Some(a), Some(b)) => a.scaled_add(scale, b),
            (None, Some(b)) => self.c2 = Some(b.mapv(|z| z * scale)),
            _ => {}
        }
        self
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

/// Build the alpha and beta contraction frames of one parent, when every spin string has at most
/// `CONTRACTRANK` replacements and both spins have occupied and virtual orbitals.
/// # Arguments:
/// - `parent`: Parent orbitals and occupations.
/// - `strings`: Parent-local spin strings.
/// # Returns
/// - `Option<[ContractedFrame<T>; 2]>`: Alpha and beta frames, or `None`.
fn contracted_frames<T: NOCIScalar>(
    parent: &ParentDeterminant<T>,
    strings: &ParentSpinSpace,
) -> Option<[ContractedFrame<T>; 2]> {
    Some([
        contracted_frame(&parent.ca, parent.oa, &strings.areps)?,
        contracted_frame(&parent.cb, parent.ob, &strings.breps)?,
    ])
}

/// Build the contraction frame of one spin: occupied-first orbitals, and each spin string as
/// in-place replacements with the sign that converts from ascending orbital order.
/// # Arguments:
/// - `c`: Parent molecular orbital coefficients.
/// - `occupation`: Parent occupation bitstring of this spin.
/// - `strings`: Parent-local spin strings of this spin.
/// # Returns
/// - `Option<ContractedFrame<T>>`: The frame, or `None`.
fn contracted_frame<T: NOCIScalar>(
    c: &Array2<T>,
    occupation: u128,
    strings: &[SpinStringData],
) -> Option<ContractedFrame<T>> {
    let n = c.ncols();
    let occ = (0..n)
        .filter(|&k| occupation >> k & 1 == 1)
        .collect::<Vec<_>>();
    let vir = (0..n)
        .filter(|&k| occupation >> k & 1 == 0)
        .collect::<Vec<_>>();
    if occ.is_empty() || vir.is_empty() {
        return None;
    }
    let order = occ.iter().chain(&vir).copied().collect::<Vec<_>>();
    let c = Array2::from_shape_fn(c.raw_dim(), |(mu, k)| c[(mu, order[k])]);

    let bits = |mask: u128| (0..n).filter(|&k| mask >> k & 1 == 1).collect::<Vec<_>>();
    let mut out = Vec::with_capacity(strings.len());
    for string in strings {
        let holes = bits(string.excitation.holes);
        let parts = bits(string.excitation.parts);
        let rank = holes.len();
        if rank > CONTRACTRANK {
            return None;
        }

        // Occupied positions of the holes and virtual positions of the particles.
        let mut h = [0; 2];
        let mut p = [0; 2];
        for r in 0..rank {
            h[r] = occ.binary_search(&holes[r]).ok()?;
            p[r] = vir.binary_search(&parts[r]).ok()?;
        }

        // Sign of the in-place column order relative to ascending orbital order.
        let mut columns = occ.clone();
        for r in 0..rank {
            columns[h[r]] = parts[r];
        }
        let inversions = (0..columns.len())
            .map(|i| {
                (i + 1..columns.len())
                    .filter(|&j| columns[i] > columns[j])
                    .count()
            })
            .sum::<usize>();
        let sign = if inversions % 2 == 0 { 1.0 } else { -1.0 };

        out.push(ContractedString {
            rank,
            holes: h,
            parts: p,
            sign,
        });
    }

    Some(ContractedFrame {
        c,
        nocc: occ.len(),
        strings: out,
    })
}

/// Transform both occupied indices, `t_{klab} = \sum_{ij} G_{ik} G_{jl} c_{ijab}`.
/// # Arguments:
/// - `c2`: Amplitudes `c_{ijab}`.
/// - `g`: Occupied transformation `G`.
/// # Returns
/// - `Array4<R>`: `t_{klab}`.
fn occupied_transform<R: NOCIScalar>(
    c2: &Array4<R>,
    g: &Array2<R>,
) -> Array4<R> {
    let (o, _, v, _) = c2.dim();
    // First index, `A_{kjab} = \sum_i G_{ik} c_{ijab}`.
    let a = g.t().dot(&c2.view().into_shape((o, o * v * v)).unwrap());
    let a = a.into_shape((o, o, v * v)).unwrap();
    // Second index, `t_{klab} = \sum_j G_{jl} A_{kjab}`.
    let mut t = Array4::from_elem((o, o, v, v), R::from_real(0.0));
    for k in 0..o {
        let tk = g.t().dot(&a.index_axis(Axis(0), k));
        t.index_axis_mut(Axis(0), k)
            .assign(&tk.into_shape((o, v, v)).unwrap());
    }
    t
}

/// Transform both virtual indices, `r_{klcd} = \sum_{ab} Q_{ca} Q_{db} t_{klab}`.
/// # Arguments:
/// - `t2`: Amplitudes `t_{klab}`.
/// - `q`: Virtual transformation `Q`.
/// # Returns
/// - `Array4<R>`: `r_{klcd}`.
fn virtual_transform<R: NOCIScalar>(
    t2: &Array4<R>,
    q: &Array2<R>,
) -> Array4<R> {
    let (o, _, v, _) = t2.dim();
    // Last index, then the third after swapping the two virtual indices.
    let b = t2.view().into_shape((o * o * v, v)).unwrap().dot(&q.t());
    let b = b
        .into_shape((o, o, v, v))
        .unwrap()
        .permuted_axes([0, 1, 3, 2]);
    let b = b.as_standard_layout().into_owned();
    let r = b.into_shape((o * o * v, v)).unwrap().dot(&q.t());
    r.into_shape((o, o, v, v))
        .unwrap()
        .permuted_axes([0, 1, 3, 2])
        .as_standard_layout()
        .into_owned()
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

/// Virtual one-body term of a double, `X_{ijab} = \sum_c f_{bc} r_{ijac}`.
/// # Arguments:
/// - `r2`: Amplitudes `r_{ijab}`.
/// - `f`: Virtual-virtual block `f_{bc}`.
/// # Returns
/// - `Array4<R>`: `X_{ijab}`.
fn virtual_one<R: NOCIScalar>(
    r2: &Array4<R>,
    f: &Array2<R>,
) -> Array4<R> {
    let (o, _, v, _) = r2.dim();
    let x = r2.view().into_shape((o * o * v, v)).unwrap().dot(&f.t());
    x.into_shape((o, o, v, v)).unwrap()
}

/// Occupied one-body term of a double, `Y_{ijab} = \sum_k f_{kj} r_{ikab}`.
/// # Arguments:
/// - `r2`: Amplitudes `r_{ijab}`.
/// - `f`: Occupied-occupied block `f_{kj}`.
/// # Returns
/// - `Array4<R>`: `Y_{ijab}`.
fn occupied_one<R: NOCIScalar>(
    r2: &Array4<R>,
    f: &Array2<R>,
) -> Array4<R> {
    let (o, _, v, _) = r2.dim();
    let mut y = Array4::from_elem((o, o, v, v), R::from_real(0.0));
    for i in 0..o {
        let ri = r2.index_axis(Axis(0), i).into_shape((o, v * v)).unwrap();
        let yi = f.t().dot(&ri);
        y.index_axis_mut(Axis(0), i)
            .assign(&yi.into_shape((o, v, v)).unwrap());
    }
    y
}

/// Exchange the two virtual indices, `X_{ijba}`.
/// # Arguments:
/// - `x`: Four-index tensor.
/// # Returns
/// - `Array4<R>`: Tensor with the virtual indices exchanged.
fn swap_virtual<R: NOCIScalar>(x: &Array4<R>) -> Array4<R> {
    x.view()
        .permuted_axes([0, 1, 3, 2])
        .as_standard_layout()
        .into_owned()
}

/// Exchange the two occupied indices, `X_{jiab}`.
/// # Arguments:
/// - `x`: Four-index tensor.
/// # Returns
/// - `Array4<R>`: Tensor with the occupied indices exchanged.
fn swap_occupied<R: NOCIScalar>(x: &Array4<R>) -> Array4<R> {
    x.view()
        .permuted_axes([1, 0, 2, 3])
        .as_standard_layout()
        .into_owned()
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

/// Pair determinant of a virtual-occupied matrix, `\beta_{ai}\beta_{bj} - \beta_{aj}\beta_{bi}`.
/// # Arguments:
/// - `beta`: Virtual-occupied matrix `\beta_{ai}`.
/// # Returns
/// - `Array4<R>`: The antisymmetric double indexed `[i, j, a, b]`.
fn pair_det<R: NOCIScalar>(beta: &Array2<R>) -> Array4<R> {
    let (v, o) = beta.dim();
    Array4::from_shape_fn((o, o, v, v), |(i, j, a, b)| {
        beta[(a, i)] * beta[(b, j)] - beta[(a, j)] * beta[(b, i)]
    })
}

/// Parent-local IDs of the single-replacement strings of a frame.
/// # Arguments:
/// - `frame`: Frame of one parent and spin.
/// # Returns
/// - `Vec<usize>`: String IDs with one replacement.
fn target_singles<T: NOCIScalar>(frame: &ContractedFrame<T>) -> Vec<usize> {
    (0..frame.strings.len())
        .filter(|&k| frame.strings[k].rank == 1)
        .collect()
}

/// Row of every single-replacement string among `singles`, `usize::MAX` for other strings.
/// # Arguments:
/// - `frame`: Frame of one parent and spin.
/// - `singles`: String IDs with one replacement.
/// # Returns
/// - `Vec<usize>`: Row index per string ID.
fn single_rows<T: NOCIScalar>(
    frame: &ContractedFrame<T>,
    singles: &[usize],
) -> Vec<usize> {
    let mut rows = vec![usize::MAX; frame.strings.len()];
    for (r, &k) in singles.iter().enumerate() {
        rows[k] = r;
    }
    rows
}

/// Whether an ordered parent pair `QP` can be applied in contraction form: two different parents
/// with frames, whose determinants have at most `CONTRACTRANK` replacements in total.
/// # Arguments:
/// - `spin`: Shared determinant-space factorisation.
/// - `frames`: Contraction frames of every parent.
/// - `parents`: Target parent `Q` and source parent `P`.
/// # Returns
/// - `bool`: Whether the class decomposition of the contraction form covers the pair.
fn contractible<T: NOCIScalar>(
    spin: &SpinFactorisation,
    frames: &[Option<[ContractedFrame<T>; 2]>],
    parents: (usize, usize),
) -> bool {
    let (target_parent, source_parent) = parents;
    let within = |parent: usize| {
        frames[parent].as_ref().is_some_and(|[a, b]| {
            spin.parents[parent]
                .entries
                .iter()
                .all(|e| a.strings[e.a].rank + b.strings[e.b].rank <= CONTRACTRANK)
        })
    };
    target_parent != source_parent && within(target_parent) && within(source_parent)
}
