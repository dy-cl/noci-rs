// nocc/stochastic/population.rs

// Crate-root imports.
use crate::basis::{apply_excitation, excitation_phase_bits, undo_excitation};
use crate::determinant::{DeterminantState, NOCIIndex};
use crate::elements::NOCIData;
use crate::hash::FastMap;
use crate::input::JMNOCCMCOptions;
use crate::nocc::reference_determinants;
use crate::qmc::AliasTable;
use crate::qmc::excitgen::OrthogonalUniformGenerator;

/// Flat index of an excitor `(z, D)` of reference block `z`, with
/// `|D\rangle = \hat\tau|{}^z\Psi\rangle` in the parent frame of `z`: the offset of block `z`,
/// then `0` for its reference or `1 + r` for its projection row `r`. Every excitor lies in its
/// block's rows, since spawns never leave the cluster rank, and the block is part of the index
/// because several blocks may share one parent frame.
#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(super) struct ExcitorIndex(pub(super) usize);

/// One excitation `\hat\tau_{z\mu}` of a reference block, with the sign convention
/// `\hat\tau_{z\mu}|{}^z\Psi\rangle = +|D_\mu\rangle` so that its population is the coefficient
/// of `|D_\mu\rangle` in `\hat T_z|{}^z\Psi\rangle`.
#[derive(Clone, Copy)]
pub(super) struct Excitor {
    /// Excited determinant `|D_\mu\rangle` in the parent frame of the block.
    pub(super) det: DeterminantState,
    /// Alpha and beta holes relative to the block reference.
    pub(super) holes: (u128, u128),
    /// Alpha and beta particles relative to the block reference.
    pub(super) parts: (u128, u128),
    /// Sign `s_\mu` making `s_\mu a^\dagger_{\mathrm{parts}}a_{\mathrm{holes}}` map the reference
    /// to `+|D_\mu\rangle`.
    pub(super) sign: f64,
}

/// Projection and expansion spaces of every reference block: the block references, their
/// projection rows up to the cluster rank, the expansion rank of each `e^{\hat T_x}|{}^x\Psi\rangle`
/// and the Hamiltonian-connection generator of each reference frame.
pub(super) struct ReferenceSpaces {
    /// Block reference determinants.
    pub(super) references: Vec<DeterminantState>,
    /// Projection rows of every block with ranks `1..=cluster_rank`.
    pub(super) rows: Vec<Vec<DeterminantState>>,
    /// Flat index of every block's reference, then the total number of excitors.
    offsets: Vec<usize>,
    /// Position within its block, `0` for the reference and `1 + r` for row `r`, of every
    /// excitor's occupations.
    positions: Vec<FastMap<(u128, u128), usize>>,
    /// Number of spatial orbitals of every block.
    pub(super) norb: Vec<usize>,
    /// Uniform connection generator of every block.
    pub(super) generators: Vec<OrthogonalUniformGenerator>,
    /// Highest excitation rank of the cluster operators.
    pub(super) cluster_rank: usize,
    /// Highest rank and cluster size of the expansion of each `e^{\hat T_x}|{}^x\Psi\rangle`.
    pub(super) expansion_rank: usize,
}

/// Population of one reference block on this rank for one propagation step: the global
/// reference population `N_z`, every locally owned excitor with its population `n_{z\mu}`, the
/// alias table selecting excitors with probability `|n_{z\mu}|/N_{\text{ex}}` for linked
/// clusters, and the per-level and combination tables of the even selection of coupling
/// clusters. Its size is the number of occupied excitors, never that of the excitor space.
pub(super) struct Block {
    /// Block reference determinant.
    pub(super) reference: DeterminantState,
    /// Reference population `N_z`.
    pub(super) nref: f64,
    /// Excitors with their populations.
    pub(super) excitors: Vec<(Excitor, f64)>,
    /// Alias table over `|n_{z\mu}|` aligned with `excitors`.
    pub(super) selection: AliasTable,
    /// Indices into `excitors` of each excitation level `1,\ldots,l`, with an alias table over
    /// their `|n_{z\mu}|`.
    pub(super) levels: Vec<(Vec<usize>, AliasTable)>,
    /// Combinations `\eta` of excitors per level sampled by the coupling.
    pub(super) combinations: Vec<Vec<usize>>,
    /// Alias table over the combination weights `a_\eta`, whose total is `A`.
    pub(super) coupling: AliasTable,
}

impl Excitor {
    /// Construct the excitor mapping a block reference to `det`.
    /// # Arguments:
    /// - `reference`: Block reference determinant.
    /// - `det`: Excited determinant in the same parent frame.
    /// # Returns:
    /// - `Excitor`: Excitor with `\hat\tau|{}^z\Psi\rangle = +|D\rangle`.
    pub(super) fn new(
        reference: &DeterminantState,
        det: DeterminantState,
    ) -> Self {
        let holes = (reference.alpha & !det.alpha, reference.beta & !det.beta);
        let parts = (det.alpha & !reference.alpha, det.beta & !reference.beta);
        let sign = excitation_phase_bits(reference.alpha, holes.0, parts.0)
            * excitation_phase_bits(reference.beta, holes.1, parts.1);
        Self {
            det,
            holes,
            parts,
            sign,
        }
    }

    /// Apply the excitor to a determinant of the same frame, `\hat\tau|D\rangle = \pm|D'\rangle`.
    /// # Arguments:
    /// - `self`: Excitor.
    /// - `det`: Determinant acted on.
    /// # Returns:
    /// - `Option<(DeterminantState, f64)>`: Image and sign, or `None` if a hole is empty or a
    ///   particle is already occupied.
    #[inline(always)]
    pub(super) fn apply(
        &self,
        det: &DeterminantState,
    ) -> Option<(DeterminantState, f64)> {
        let (alpha, sa) = apply_excitation(det.alpha, self.holes.0, self.parts.0)?;
        let (beta, sb) = apply_excitation(det.beta, self.holes.1, self.parts.1)?;
        Some((
            DeterminantState {
                parent: det.parent,
                alpha,
                beta,
            },
            self.sign * sa * sb,
        ))
    }

    /// Invert the excitor on a determinant, `|D\rangle` with `\hat\tau|D\rangle = \pm|D'\rangle`.
    /// # Arguments:
    /// - `self`: Excitor.
    /// - `det`: Image determinant `|D'\rangle`.
    /// # Returns:
    /// - `Option<(DeterminantState, f64)>`: Preimage `|D\rangle` and the sign
    ///   `\langle D'|\hat\tau|D\rangle`, or `None` if `|D'\rangle` has no preimage.
    #[inline(always)]
    pub(super) fn invert(
        &self,
        det: &DeterminantState,
    ) -> Option<(DeterminantState, f64)> {
        // A preimage needs every particle occupied and every hole empty in `|D'\rangle`.
        let valid = |occ: u128, holes: u128, parts: u128| occ & parts == parts && occ & holes == 0;
        if !valid(det.alpha, self.holes.0, self.parts.0)
            || !valid(det.beta, self.holes.1, self.parts.1)
        {
            return None;
        }
        let source = DeterminantState {
            parent: det.parent,
            alpha: undo_excitation(det.alpha, self.holes.0, self.parts.0),
            beta: undo_excitation(det.beta, self.holes.1, self.parts.1),
        };
        let (_, sign) = self.apply(&source)?;
        Some((source, sign))
    }
}

impl ReferenceSpaces {
    /// Build the spaces of every reference determinant of a NOCI space.
    /// # Arguments:
    /// - `data`: NOCI data over the reference determinants.
    /// - `options`: JM-NOCCMC options.
    /// # Returns:
    /// - `ReferenceSpaces`: Spaces of every reference block.
    /// # Panics
    /// - Panics if the MO caches are absent.
    pub(super) fn new(
        data: &NOCIData<'_, f64>,
        options: &JMNOCCMCOptions,
    ) -> Self {
        let mocache = data.mocache.expect("JM-NOCCMC requires MO caches");
        let parents = data.space.parents();
        let m = data.space.len();
        let mut spaces = Self {
            references: Vec::with_capacity(m),
            rows: Vec::with_capacity(m),
            offsets: vec![0],
            positions: Vec::with_capacity(m),
            norb: Vec::with_capacity(m),
            generators: Vec::with_capacity(m),
            cluster_rank: options.cluster_rank,
            expansion_rank: 0,
        };

        let mut nelec = 0;
        for z in 0..m {
            let parent = data.space.parent(NOCIIndex(z));
            let (oa, ob) = data.space.occupations(NOCIIndex(z));
            let norb = parents[parent].ca.ncols();
            nelec = (oa.count_ones() + ob.count_ones()) as usize;
            let reference = DeterminantState {
                parent,
                alpha: oa,
                beta: ob,
            };
            spaces.references.push(reference);
            spaces.rows.push(
                reference_determinants(oa, ob, norb, options.cluster_rank)
                    .into_iter()
                    .skip(1)
                    .map(|(alpha, beta, _)| DeterminantState {
                        parent,
                        alpha,
                        beta,
                    })
                    .collect(),
            );
            spaces.norb.push(norb);
            spaces
                .generators
                .push(OrthogonalUniformGenerator::new((oa, ob), &mocache[parent]));

            // Flat indices: the reference first, then the rows in order.
            let rows = &spaces.rows[z];
            let positions = std::iter::once((oa, ob))
                .chain(rows.iter().map(|row| (row.alpha, row.beta)))
                .enumerate()
                .map(|(position, occupations)| (occupations, position))
                .collect();
            spaces.positions.push(positions);
            spaces.offsets.push(spaces.offsets[z] + 1 + rows.len());
        }
        spaces.expansion_rank = options.expansion_rank.unwrap_or(nelec).min(nelec);
        spaces
    }

    /// Return the flat index of the excitor of a block with the given occupations.
    /// # Arguments:
    /// - `self`: Spaces of every reference block.
    /// - `block`: Reference block `z`.
    /// - `occupations`: Alpha and beta occupations of the excitor's determinant.
    /// # Returns:
    /// - `ExcitorIndex`: Flat index of the excitor.
    /// # Panics
    /// - Panics if the occupations are neither the block's reference nor one of its rows.
    pub(super) fn index(
        &self,
        block: usize,
        occupations: (u128, u128),
    ) -> ExcitorIndex {
        ExcitorIndex(self.offsets[block] + self.positions[block][&occupations])
    }

    /// Return the block of an excitor.
    /// # Arguments:
    /// - `self`: Spaces of every reference block.
    /// - `index`: Flat excitor index.
    /// # Returns:
    /// - `usize`: Reference block `z` of the excitor.
    pub(super) fn block(
        &self,
        index: ExcitorIndex,
    ) -> usize {
        self.offsets.partition_point(|&offset| offset <= index.0) - 1
    }

    /// Return the determinant of an excitor in the parent frame of its block.
    /// # Arguments:
    /// - `self`: Spaces of every reference block.
    /// - `index`: Flat excitor index.
    /// # Returns:
    /// - `DeterminantState`: The block's reference or one of its rows.
    pub(super) fn determinant(
        &self,
        index: ExcitorIndex,
    ) -> DeterminantState {
        let block = self.block(index);
        match index.0 - self.offsets[block] {
            0 => self.references[block],
            position => self.rows[block][position - 1],
        }
    }

    /// Return whether an excitor is its block's reference.
    /// # Arguments:
    /// - `self`: Spaces of every reference block.
    /// - `index`: Flat excitor index.
    /// # Returns:
    /// - `bool`: Whether the index is the first of its block.
    pub(super) fn is_reference(
        &self,
        index: ExcitorIndex,
    ) -> bool {
        self.offsets.binary_search(&index.0).is_ok()
    }
}

impl Block {
    /// Total excitor population `N_{\text{ex}} = \sum_\mu |n_{z\mu}|`.
    /// # Arguments:
    /// - `self`: Block population.
    /// # Returns:
    /// - `f64`: Excitor population 1-norm.
    pub(super) fn nex(&self) -> f64 {
        self.selection.total()
    }
}

/// Excitation rank of a determinant relative to a reference of the same frame.
/// # Arguments:
/// - `reference`: Reference determinant.
/// - `det`: Determinant.
/// # Returns:
/// - `usize`: Number of holes in both spins.
#[inline(always)]
pub(super) fn rank(
    reference: &DeterminantState,
    det: &DeterminantState,
) -> usize {
    ((reference.alpha & !det.alpha).count_ones() + (reference.beta & !det.beta).count_ones())
        as usize
}
