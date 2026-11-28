// noci/factorise/onebody.rs
//! Spin-factorised one-body NOCI operator contractions.

#[cfg(feature = "gpu")]
mod device;

// Crate-visible type re-exports.
#[cfg(feature = "gpu")]
pub(crate) use device::DeviceOneBody;

// Standard library imports.
use std::ops::Range;
use std::path::Path;

// External crate imports.
use ndarray::Array1;
use rayon::prelude::*;

// Crate-root imports.
use crate::NOCIScalar;
use crate::basis::excitation_phase_bits;
use crate::determinant::Bits;
use crate::determinant::{NOCIIndex, NOCISpace, NOCISpinStringIndex, SpinStringData};
use crate::elements::calculate_f_pair_orthogonal;
use crate::elements::calculate_s_pair_orthogonal;
use crate::elements::nonorthogonalwicks::{
    OneBodyFactor, SpinFactorRow, SpinStringSources, WickScratchSpin, WicksPairView,
    xw_spin_factor_row,
};
use crate::elements::{DetPair, Fock, FockData, FockMOCache, NOCIData, Overlap, calculate_pairs};
use crate::hash::FastMap;
use crate::input::SNOCIStorage;

// Parent/sibling imports.
use super::contracted::{ContractedPair, DOTCHUNK, contractible};
use super::storage::{OneBodyFactorStorage, OneBodyStoragePlan};
use super::{SpinFactorisation, ordered_parent_pair};

/// Persistent one-body block for one ordered source-target parent pair `QP`.
enum OneBodyBlock<T: NOCIScalar> {
    /// Same-parent standard Slater-Condon sparse one-body action, for pairs outside the
    /// contraction form.
    Orthogonal(OrthogonalOneBodyBlock<T>),
    /// Spin-factorised nonorthogonal one-body action.
    Factorised(FactorisedOneBodyBlock<T>),
    /// Spin-factorised nonorthogonal one-body action with regenerated factor tables.
    Transient(TransientOneBodyBlock),
    /// One-body action in contraction form, same-parent or not.
    Contracted(ContractedPair<T>),
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
}

/// Parent-local occupation group for the orthogonal one-body shortcut.
struct OrthogonalOccupationGroup {
    /// Determinants sharing one occupation pair.
    targets: Vec<OrthogonalTarget>,
}

/// Parent-local occupation lookup and single-excitation couplings for same-parent orthogonal
/// one-body application.
struct OrthogonalOneBodyBlock<T: NOCIScalar> {
    /// Parent `P`.
    parent: usize,
    /// Numerical group keyed by canonical alpha/beta spin-string IDs.
    opos: FastMap<(NOCISpinStringIndex, NOCISpinStringIndex), usize>,
    /// Determinants grouped by the same canonical spin-string pair.
    groups: Vec<OrthogonalOccupationGroup>,
    /// Offset of each parent entry's couplings in `singles`, followed by their total count.
    starts: Vec<usize>,
    /// Single-excitation Fock couplings `(s, F_{ts})` of every target `t`, in parent-entry order.
    singles: Vec<(usize, T)>,
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
    /// Pairs in contraction form, same-parent or not, and the remaining same-parent orthonormal
    /// pairs need no Wick intermediates; every other pair uses Wick factor rows.
    /// # Arguments:
    /// - `data`: Shared NOCI data, with Wick intermediates when a pair needs factor rows.
    /// - `fock`: Current generalised-Fock data, already reflected in Wick intermediates.
    /// - `cache`: Directory for persistent file-backed factor blocks.
    /// - `rank`: MPI rank used in factor-cache filenames.
    /// - `iteration`: SNOCI iteration used in factor-cache filenames.
    /// - `storage`: Requested persistent factor-table storage backend.
    /// # Returns
    /// - `Option<OneBodyFactorisation<T>>`: Cached spin-factorised one-body operator, or `None`
    ///   when a pair needs factor rows and no Wick intermediates are available.
    pub(crate) fn new(
        data: &NOCIData<'_, T>,
        fock: &FockData<'_, T>,
        cache: &Path,
        rank: i32,
        iteration: usize,
        storage: SNOCIStorage,
    ) -> Option<Self> {
        let spin = SpinFactorisation::new(data);
        let nparent = spin.parents.len();
        let wicks = data.input.wicks.enabled && data.wicks.is_some();

        let mut blocks = Vec::with_capacity(nparent * nparent);
        let mut storage_plan = OneBodyStoragePlan::new(cache, rank, iteration, storage);

        // Choose a representation per ordered parent pair: the contraction form for every pair it
        // covers, same-parent or not; otherwise exact orthogonal Slater-Condon blocks for an
        // orthonormal parent, transient evaluation, or persistent spin factors.
        for target_parent in 0..nparent {
            for source_parent in 0..nparent {
                let contracted =
                    ContractedPair::new((target_parent, source_parent), &spin, data, Some(fock));

                if let Some(block) = contracted {
                    blocks.push(OneBodyBlock::Contracted(block));
                } else if target_parent == source_parent
                    && fock.fock_mocache[target_parent].orthogonal_slater_condon
                {
                    blocks.push(OneBodyBlock::Orthogonal(build_orthogonal_one_body_block(
                        &spin,
                        target_parent,
                        data.space,
                        &fock.fock_mocache[target_parent],
                    )));
                } else if !wicks {
                    return None;
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

        Some(Self {
            spin,
            blocks,
            nparent,
        })
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

        // Each pair with factor rows stores `\alpha` and `\beta` `S` and `F` factors; contracted
        // pairs, same-parent or not, and same-parent orthogonal blocks need no factor table.
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

        // Source amplitudes of each parent, built once for all its pairs.
        let sources = self
            .spin
            .parents
            .par_iter()
            .map(|parent| {
                parent
                    .entries
                    .iter()
                    .map(|e| (e.a, e.b, xs[e.det]))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // Target rows of each parent, built once for all its pairs.
        let targets = self
            .spin
            .parents
            .par_iter()
            .map(|parent| {
                parent
                    .entries
                    .iter()
                    .map(|e| (e.det, e.a, e.b))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // Contracted parent pairs run concurrently, each assigned whole to one worker. Every pair
        // returns its contributions in the row order of its target parent.
        let contracted = self
            .blocks
            .par_iter()
            .filter_map(|block| match block {
                OneBodyBlock::Contracted(block)
                    if (block.target_parent * self.nparent + block.source_parent) % nworker
                        == worker =>
                {
                    let out = block.apply_pair(
                        &self.spin,
                        data,
                        &sources[block.source_parent],
                        &targets[block.target_parent],
                        lambda,
                    );

                    Some((block.target_parent, out))
                }
                _ => None,
            })
            .collect::<Vec<_>>();

        // Sum the pairs of each target parent row by row, in parallel over parents, then add each
        // sum into `y` once.
        let sums = (0..self.nparent)
            .into_par_iter()
            .map(|parent| {
                let mut sum = vec![zero; targets[parent].len()];
                for (_, out) in contracted.iter().filter(|(p, _)| *p == parent) {
                    for (s, &(_, value)) in sum.iter_mut().zip(out) {
                        *s += value;
                    }
                }

                sum
            })
            .collect::<Vec<_>>();

        for (rows, sum) in targets.iter().zip(sums) {
            for (&(det, _, _), value) in rows.iter().zip(sum) {
                y[det] += value;
            }
        }

        Array1::from_vec(y)
    }

    /// Build diagonal entries of `F + \lambda S` and `S`, `F_{II} + \lambda S_{II}` and `S_{II}`.
    /// Only same-parent pairs reach the diagonal, and every one is evaluated by the general pair
    /// dispatch: orthogonal Slater-Condon rules for an orthonormal parent, Wick or generalised
    /// Slater-Condon rules otherwise, whatever representation applies the pair.
    /// # Arguments:
    /// - `data`: Shared NOCI data, with Wick intermediates when available.
    /// - `fock`: Current generalised-Fock data.
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

        // Every determinant of every parent, in parent order.
        let dets = self
            .spin
            .parents
            .iter()
            .flat_map(|parent| parent.entries.iter().map(|e| e.det))
            .collect::<Vec<_>>();

        // `F_{II}` and `S_{II}` in chunks of `DOTCHUNK` determinants, one Wick scratch per Rayon
        // worker, each chunk returning its values in determinant order.
        let operator = Fock { fock, shift: 0.0 };
        let values = dets
            .par_chunks(DOTCHUNK)
            .map_init(WickScratchSpin::new, |scratch, chunk| {
                // The diagonal pair `(I, I)` of every determinant in the chunk.
                let pairs = chunk
                    .iter()
                    .map(|&det| DetPair::from_space(data.space, NOCIIndex(det), NOCIIndex(det)))
                    .collect::<Vec<_>>();

                // Both operators over the same pairs, sharing the scratch.
                let mut f = vec![zero; pairs.len()];
                let mut s = vec![zero; pairs.len()];
                calculate_pairs(data, &operator, &pairs, Some(&mut *scratch), &mut f);
                calculate_pairs(data, &Overlap, &pairs, Some(&mut *scratch), &mut s);

                (f, s)
            })
            .collect::<Vec<_>>();

        // `M_{II} = F_{II} + \lambda S_{II}`.
        for (chunk, (f, s)) in dets.chunks(DOTCHUNK).zip(values) {
            for (k, &det) in chunk.iter().enumerate() {
                s_diag[det] = s[k];
                m_diag[det] = f[k] + lambda * s[k];
            }
        }

        (Array1::from_vec(m_diag), Array1::from_vec(s_diag))
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
    /// Target rows are gathered in parallel, each from the sources it connects to.
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
        block: &OrthogonalOneBodyBlock<T>,
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
        let parent = &self.spin.parents[block.parent];
        let (worker, nworker) = partition;

        // `F` and `S` are Hermitian, so each target row `t` gathers
        // `\sum_s (F + \lambda S)_{ts} x_s` over the sources it connects to: those sharing its
        // spin strings and its `\alpha` or `\beta` single excitations. Each row is owned by one
        // worker and written once, so threads never share an output.
        let rows: Vec<(usize, R)> = parent
            .entries
            .par_iter()
            .enumerate()
            .filter(|(_, entry)| entry.a % nworker == worker)
            .map(|(k, entry)| {
                let target = NOCIIndex(entry.det);
                let mut value = zero;

                // Identical occupations contribute `(F + \lambda S)_{ts} x_s`.
                let state = space.spin_string_pair(target);
                let group = &block.groups[block.opos[&(state.aid, state.bid)]];
                for source in &group.targets {
                    let xs = x[source.det];
                    if xs != zero {
                        let source = NOCIIndex(source.det);
                        let f = <R as From<T>>::from(calculate_f_pair_orthogonal(
                            cache, space, target, source,
                        ));
                        let s = <R as From<T>>::from(calculate_s_pair_orthogonal(
                            space, target, source,
                        ));
                        value += (f + lambda * s) * xs;
                    }
                }

                // A one-body operator also connects `\alpha` or `\beta` single excitations.
                for &(source, f) in &block.singles[block.starts[k]..block.starts[k + 1]] {
                    value += <R as From<T>>::from(f) * x[source];
                }

                (entry.det, value)
            })
            .collect();

        for (det, value) in rows {
            y[det] += value;
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

/// Build same-parent orthogonal one-body action data for parent `P`: the occupation lookup and
/// every target's single-excitation Fock couplings, which stay fixed while the operator is
/// applied.
/// # Arguments:
/// - `spin`: Shared determinant-space factorisation.
/// - `parent_id`: Parent `P`.
/// - `space`: Candidate determinant space used for occupations and indices.
/// - `cache`: MO-basis Fock cache for the parent.
/// # Returns
/// - `OrthogonalOneBodyBlock<T>`: Orthogonal same-parent block.
fn build_orthogonal_one_body_block<T: NOCIScalar>(
    spin: &SpinFactorisation,
    parent_id: usize,
    space: &NOCISpace<T>,
    cache: &FockMOCache<T>,
) -> OrthogonalOneBodyBlock<T> {
    // Group retained determinants by their exact parent-local alpha/beta spin-string pair.
    let parent = &spin.parents[parent_id];
    let mut opos = FastMap::default();
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
        groups[group]
            .targets
            .push(OrthogonalTarget { det: entry.det });
    }

    // Store a direct occupation-pair lookup for same-parent Slater-Condon application.
    let mut block = OrthogonalOneBodyBlock {
        parent: parent_id,
        opos,
        groups,
        starts: Vec::new(),
        singles: Vec::new(),
    };

    // Tabulate each target's `\alpha` and `\beta` single-excitation couplings in compressed rows.
    let rows: Vec<Vec<(usize, T)>> = parent
        .entries
        .par_iter()
        .map(|entry| {
            let mut row = Vec::new();
            orthogonal_singles(&block, NOCIIndex(entry.det), space, cache, true, &mut row);
            orthogonal_singles(&block, NOCIIndex(entry.det), space, cache, false, &mut row);

            row
        })
        .collect();

    // Row `k` occupies `singles[starts[k]..starts[k + 1]]`.
    block.starts = std::iter::once(0)
        .chain(rows.iter().scan(0, |end, row| {
            *end += row.len();
            Some(*end)
        }))
        .collect();
    block.singles = rows.into_iter().flatten().collect();

    block
}

/// List the same-parent single-excitation Fock couplings `(s, F_{ts})` of one target row, over the
/// retained sources `s` differing from `t` by one `\alpha` or one `\beta` replacement. Overlaps
/// vanish between distinct determinants of one orthonormal parent.
/// # Arguments:
/// - `orthogonal`: Parent-local occupation lookup.
/// - `target`: Target determinant `t`.
/// - `space`: Candidate determinant space used for occupations and indices.
/// - `cache`: MO-basis Fock cache for the parent.
/// - `alpha`: Whether to replace an `\alpha` orbital rather than a `\beta` orbital.
/// - `out`: Couplings to extend.
/// # Returns
/// - `()`: Appends the couplings of row `t` to `out`.
fn orthogonal_singles<T: NOCIScalar>(
    orthogonal: &OrthogonalOneBodyBlock<T>,
    target: NOCIIndex,
    space: &NOCISpace<T>,
    cache: &FockMOCache<T>,
    alpha: bool,
    out: &mut Vec<(usize, T)>,
) {
    let nmo = cache.fa.nrows();

    // The target's spin strings within its parent.
    let state = space.spin_string_pair(target);
    let strings = space.parent_spin_strings(state.parent);

    // The occupation and MO Fock matrix of the replaced spin.
    let (oa, ob) = space.occupations(target);
    let (occ, f_sigma) = if alpha {
        (oa, &cache.fa)
    } else {
        (ob, &cache.fb)
    };

    // Enumerate replacements `i \to a` of the chosen spin, with `i` occupied and `a` virtual,
    // leaving the other spin unchanged.
    for hole in occ.ones() {
        for part in 0..nmo {
            if occ.contains(part) {
                continue;
            }

            let source_occ = (occ & !Bits::bit(hole)) | Bits::bit(part);

            // Retained sources with this occupation, if any.
            let key = if alpha {
                strings.aids.get(&source_occ).map(|&aid| (aid, state.bid))
            } else {
                strings.bids.get(&source_occ).map(|&bid| (state.aid, bid))
            };

            let Some(&opos) = key.and_then(|key| orthogonal.opos.get(&key)) else {
                continue;
            };

            // Measured from a source, the target replaces its `part` by `hole`, so by the
            // Slater-Condon rules `F_{ts} = p F^\sigma_{\text{hole},\text{part}}` with the phase `p`
            // of that replacement; it is shared by every source with this occupation.
            let phase = excitation_phase_bits(source_occ, Bits::bit(part), Bits::bit(hole));
            let f = T::from_real(phase) * f_sigma[(hole, part)];

            out.extend(
                orthogonal.groups[opos]
                    .targets
                    .iter()
                    .map(|source| (source.det, f)),
            );
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
