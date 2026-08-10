// noci/factorise/onebody.rs
//! Spin-factorised one-body NOCI operator contractions.

// External crate imports.
use ndarray::Array1;
use rayon::prelude::*;

// Crate-root imports.
use crate::nonorthogonalwicks::{WickScratchSpin, WicksPairView};
use crate::nonorthogonalwicks::{prepare_same, xw_f, xw_overlap};

// Parent/sibling imports.
use super::{ParentSpinSpace, SpinFactorisation, ordered_parent_pair};
use crate::noci::types::{FockData, NOCIData, NOCIScalar};

const MAX_CACHED_FACTOR_BYTES: usize = 1usize << 30;

/// Cached spin-factor tables for one ordered source-target parent pair `QP`.
struct OneBodyFactorBlock<T: NOCIScalar> {
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
    /// Row-major `S^{alpha,QP}_{\bar a a}` factor table.
    sa: Vec<T>,
    /// Row-major `F^{alpha,QP}_{\bar a a}` factor table.
    fa: Vec<T>,
    /// Row-major `S^{beta,QP}_{\bar b b}` factor table.
    sb: Vec<T>,
    /// Row-major `F^{beta,QP}_{\bar b b}` factor table.
    fb: Vec<T>,
    /// Selected dense contraction order for this parent pair.
    contraction: OneBodyContraction,
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
    blocks: Vec<OneBodyFactorBlock<T>>,
    /// Whether all parent-pair factor blocks are cached in memory.
    cache_blocks: bool,
    /// Number of parent references.
    nparent: usize,
}

impl<T: NOCIScalar> OneBodyFactorisation<T> {
    /// Build `F^{QP}_{\bar a\bar b,ab}` spin factors for the current generalised Fock operator.
    /// # Arguments:
    /// - `data`: Shared NOCI data with Wick intermediates for the candidate determinant basis.
    /// - `fock`: Current generalised-Fock data, already reflected in Wick intermediates.
    /// # Returns
    /// - `OneBodyFactorisation<T>`: Cached spin-factorised one-body operator.
    pub(crate) fn new(
        data: &NOCIData<'_, T>,
        fock: &FockData<'_, T>,
    ) -> Self {
        let spin = SpinFactorisation::new(data);
        let nparent = spin.parents.len();
        let cache_blocks = estimate_factor_bytes::<T>(&spin) <= MAX_CACHED_FACTOR_BYTES;
        let mut blocks = Vec::with_capacity(if cache_blocks { nparent * nparent } else { 0 });

        if cache_blocks {
            for target_parent in 0..nparent {
                for source_parent in 0..nparent {
                    blocks.push(build_one_body_factor_tables(
                        &spin,
                        data,
                        fock,
                        target_parent,
                        source_parent,
                    ));
                }
            }
        }

        Self {
            spin,
            blocks,
            cache_blocks,
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

    /// Apply `Y = (F + \lambda S)x` using cached spin factors.
    /// # Arguments:
    /// - `x`: Source vector over actual candidate determinants.
    /// - `data`: Shared NOCI data used when parent-pair factor blocks are streamed.
    /// - `fock`: Current generalised-Fock data used when parent-pair factor blocks are streamed.
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

        if self.cache_blocks {
            for block in &self.blocks {
                self.apply_one_body_block(block, xs, &mut y, lambda, scratch, (worker, nworker));
            }
        } else {
            for target_parent in 0..self.nparent {
                for source_parent in 0..self.nparent {
                    let block = build_one_body_factor_tables(
                        &self.spin,
                        data,
                        fock,
                        target_parent,
                        source_parent,
                    );
                    self.apply_one_body_block(
                        &block,
                        xs,
                        &mut y,
                        lambda,
                        scratch,
                        (worker, nworker),
                    );
                }
            }
        }

        Array1::from_vec(y)
    }

    /// Build diagonal entries of `F + \lambda S` and `S` from cached same-spin factors.
    /// # Arguments:
    /// - `data`: Shared NOCI data used when parent-pair factor blocks are streamed.
    /// - `fock`: Current generalised-Fock data used when parent-pair factor blocks are streamed.
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
            let streamed;
            let block = if self.cache_blocks {
                self.block(parent_id, parent_id)
            } else {
                streamed =
                    build_one_body_factor_tables(&self.spin, data, fock, parent_id, parent_id);
                &streamed
            };
            fill_one_body_diagonal_block(parent, block, lambda, &mut m_diag, &mut s_diag);
        }

        (Array1::from_vec(m_diag), Array1::from_vec(s_diag))
    }

    /// Return cached factor block for target parent `Q` and source parent `P`.
    /// # Arguments:
    /// - `target_parent`: Target parent `Q`.
    /// - `source_parent`: Source parent `P`.
    /// # Returns
    /// - `&OneBodyFactorBlock<T>`: Cached parent-pair factor block.
    fn block(
        &self,
        target_parent: usize,
        source_parent: usize,
    ) -> &OneBodyFactorBlock<T> {
        &self.blocks[target_parent * self.nparent + source_parent]
    }

    /// Apply one cached or streamed parent-pair block of `F + \lambda S`.
    /// # Arguments:
    /// - `block`: Cached parent-pair one-body factors.
    /// - `x`: Source determinant vector.
    /// - `y`: Output determinant vector to accumulate.
    /// - `lambda`: Scalar overlap shift.
    /// - `scratch`: Reusable dense contraction buffers.
    /// - `partition`: Worker index and worker count for target rows.
    /// # Returns
    /// - `()`: Adds this parent-pair contribution into `y`.
    fn apply_one_body_block(
        &self,
        block: &OneBodyFactorBlock<T>,
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
        block: &OneBodyFactorBlock<T>,
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

        scratch.first_f.clear();
        scratch.first_s.clear();
        scratch.first_f.resize(block.nta * block.nsb, zero);
        scratch.first_s.resize(block.nta * block.nsb, zero);

        for ta in (worker..block.nta).step_by(nworker) {
            let frow = &block.fa[ta * block.nsa..(ta + 1) * block.nsa];
            let srow = &block.sa[ta * block.nsa..(ta + 1) * block.nsa];
            let tf = &mut scratch.first_f[ta * block.nsb..(ta + 1) * block.nsb];
            let ts = &mut scratch.first_s[ta * block.nsb..(ta + 1) * block.nsb];
            for entry in &source.entries {
                let xe = x[entry.det];
                if xe != zero {
                    tf[entry.b] += frow[entry.a] * xe;
                    ts[entry.b] += srow[entry.a] * xe;
                }
            }
        }

        for entry in &target.entries {
            if entry.a % nworker != worker {
                continue;
            }
            let tf = &scratch.first_f[entry.a * block.nsb..(entry.a + 1) * block.nsb];
            let ts = &scratch.first_s[entry.a * block.nsb..(entry.a + 1) * block.nsb];
            let sbrow = &block.sb[entry.b * block.nsb..(entry.b + 1) * block.nsb];
            let fbrow = &block.fb[entry.b * block.nsb..(entry.b + 1) * block.nsb];
            let mut value = zero;
            for b in 0..block.nsb {
                value += tf[b] * sbrow[b] + ts[b] * (fbrow[b] + lambda * sbrow[b]);
            }
            y[entry.det] += value;
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
        block: &OneBodyFactorBlock<T>,
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

        scratch.first_f.clear();
        scratch.first_s.clear();
        scratch.first_f.resize(block.ntb * block.nsa, zero);
        scratch.first_s.resize(block.ntb * block.nsa, zero);

        for tb in (worker..block.ntb).step_by(nworker) {
            let frow = &block.fb[tb * block.nsb..(tb + 1) * block.nsb];
            let srow = &block.sb[tb * block.nsb..(tb + 1) * block.nsb];
            let uf = &mut scratch.first_f[tb * block.nsa..(tb + 1) * block.nsa];
            let us = &mut scratch.first_s[tb * block.nsa..(tb + 1) * block.nsa];
            for entry in &source.entries {
                let xe = x[entry.det];
                if xe != zero {
                    uf[entry.a] += xe * frow[entry.b];
                    us[entry.a] += xe * srow[entry.b];
                }
            }
        }

        for entry in &target.entries {
            if entry.b % nworker != worker {
                continue;
            }
            let uf = &scratch.first_f[entry.b * block.nsa..(entry.b + 1) * block.nsa];
            let us = &scratch.first_s[entry.b * block.nsa..(entry.b + 1) * block.nsa];
            let sarow = &block.sa[entry.a * block.nsa..(entry.a + 1) * block.nsa];
            let farow = &block.fa[entry.a * block.nsa..(entry.a + 1) * block.nsa];
            let mut value = zero;
            for a in 0..block.nsa {
                value += sarow[a] * uf[a] + (farow[a] + lambda * sarow[a]) * us[a];
            }
            y[entry.det] += value;
        }
    }
}

/// Build `S^alpha`, `F^alpha`, `S^beta` and `F^beta` tables for one parent pair `QP`.
/// # Arguments:
/// - `spin`: Shared determinant-space factorisation.
/// - `data`: Shared NOCI data containing Wick intermediates.
/// - `fock`: Current generalised-Fock data.
/// - `target_parent`: Target parent `Q`.
/// - `source_parent`: Source parent `P`.
/// # Returns
/// - `OneBodyFactorBlock<T>`: Cached row-major factor tables for this parent pair.
fn build_one_body_factor_tables<T: NOCIScalar>(
    spin: &SpinFactorisation,
    data: &NOCIData<'_, T>,
    fock: &FockData<'_, T>,
    target_parent: usize,
    source_parent: usize,
) -> OneBodyFactorBlock<T> {
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

    let mut sa = vec![T::from_real(0.0); checked_len(nta, nsa, "alpha one-body factors")];
    let mut fa = vec![T::from_real(0.0); checked_len(nta, nsa, "alpha one-body factors")];
    let mut sb = vec![T::from_real(0.0); checked_len(ntb, nsb, "beta one-body factors")];
    let mut fb = vec![T::from_real(0.0); checked_len(ntb, nsb, "beta one-body factors")];

    build_spin_one_body_factors(
        &pair,
        data,
        (target.areps.as_slice(), source.areps.as_slice()),
        target_left,
        true,
        fock,
        (&mut sa, &mut fa),
    );
    build_spin_one_body_factors(
        &pair,
        data,
        (target.breps.as_slice(), source.breps.as_slice()),
        target_left,
        false,
        fock,
        (&mut sb, &mut fb),
    );

    let contraction = select_one_body_contraction(
        nta,
        ntb,
        nsa,
        nsb,
        target.entries.len(),
        source.entries.len(),
    );

    OneBodyFactorBlock {
        target_parent,
        source_parent,
        nta,
        ntb,
        nsa,
        nsb,
        sa,
        fa,
        sb,
        fb,
        contraction,
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
    block: &OneBodyFactorBlock<T>,
    lambda: T,
    m_diag: &mut [T],
    s_diag: &mut [T],
) {
    for entry in &parent.entries {
        let saa = block.sa[entry.a * block.nsa + entry.a];
        let faa = block.fa[entry.a * block.nsa + entry.a];
        let sbb = block.sb[entry.b * block.nsb + entry.b];
        let fbb = block.fb[entry.b * block.nsb + entry.b];
        let s = saa * sbb;
        s_diag[entry.det] = s;
        m_diag[entry.det] = faa * sbb + saa * fbb + lambda * s;
    }
}

/// Build same-spin `S` and `F` factor rows from one prepared Wick scratch per component pair.
/// # Arguments:
/// - `pair`: Wick intermediates for the ordered parent pair.
/// - `data`: Shared NOCI determinant data.
/// - `reps`: Representative determinants for target and source spin components.
/// - `target_left`: Whether target determinants are left determinants in `pair`.
/// - `alpha`: Whether to build alpha or beta factors.
/// - `fock`: Current Fock data providing tolerance.
/// - `out`: Mutable row-major overlap and Fock factor tables.
/// # Returns
/// - `()`: Fills `out` factor tables.
fn build_spin_one_body_factors<T: NOCIScalar>(
    pair: &WicksPairView<'_, T>,
    data: &NOCIData<'_, T>,
    reps: (&[usize], &[usize]),
    target_left: bool,
    alpha: bool,
    _fock: &FockData<'_, T>,
    out: (&mut [T], &mut [T]),
) {
    let (target_reps, source_reps) = reps;
    let nsource = source_reps.len();
    let tol = data.tol;
    out.0
        .par_chunks_mut(nsource)
        .zip(out.1.par_chunks_mut(nsource))
        .zip(target_reps.par_iter())
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

/// Compute checked row-major factor-table length.
/// # Arguments:
/// - `nrow`: Number of rows.
/// - `ncol`: Number of columns.
/// - `name`: Allocation name used in panic text.
/// # Returns
/// - `usize`: Product `nrow * ncol`.
fn checked_len(
    nrow: usize,
    ncol: usize,
    name: &str,
) -> usize {
    nrow.checked_mul(ncol)
        .unwrap_or_else(|| panic!("{name} length overflow"))
}

/// Estimate cached one-body factor-table byte count.
/// # Arguments:
/// - `spin`: Shared determinant-space spin factorisation.
/// # Returns
/// - `usize`: Saturating byte estimate for `S/F` alpha and beta tables over all parent pairs.
fn estimate_factor_bytes<T: NOCIScalar>(spin: &SpinFactorisation) -> usize {
    let scalar = std::mem::size_of::<T>();
    let mut entries = 0usize;
    for target in &spin.parents {
        for source in &spin.parents {
            let a = target.areps.len().saturating_mul(source.areps.len());
            let b = target.breps.len().saturating_mul(source.breps.len());
            entries = entries.saturating_add(2usize.saturating_mul(a.saturating_add(b)));
        }
    }
    entries.saturating_mul(scalar)
}
