// nocc/stochastic/population.rs

// External crate imports.
use mpi::traits::*;

// Crate-root imports.
use crate::basis::{apply_excitation, excitation_phase_bits, undo_excitation};
use crate::determinant::{ExcitedDeterminant, ExcitedState, NOCIIndex};
use crate::elements::NOCIData;
use crate::input::JMNOCCMCOptions;
use crate::nocc::reference_determinants;
use crate::qmc::excitgen::OrthogonalUniformGenerator;
use crate::qmc::{AliasTable, FriAmplitude, KeyedUpdate};

/// Excitor `(z, D)` of reference block `z`, identified by the occupations of its determinant
/// `|D\rangle = \hat\tau|{}^z\Psi\rangle` in the parent frame of `z`. The block's reference has
/// no holes.
pub(super) type ExcitorKey = (usize, u128, u128);

/// One excitation `\hat\tau_{z\mu}` of a reference block, with the sign convention
/// `\hat\tau_{z\mu}|{}^z\Psi\rangle = +|D_\mu\rangle` so that its population is the coefficient
/// of `|D_\mu\rangle` in `\hat T_z|{}^z\Psi\rangle`.
#[derive(Clone, Copy)]
pub(super) struct Excitor {
    /// Excited determinant `|D_\mu\rangle` in the parent frame of the block.
    pub(super) det: ExcitedDeterminant,
    /// Alpha and beta holes relative to the block reference.
    pub(super) holes: (u128, u128),
    /// Alpha and beta particles relative to the block reference.
    pub(super) parts: (u128, u128),
    /// Sign `s_\mu` making `s_\mu a^\dagger_{\mathrm{parts}}a_{\mathrm{holes}}` map the reference
    /// to `+|D_\mu\rangle`.
    pub(super) sign: f64,
}

impl Excitor {
    /// Construct the excitor mapping a block reference to `det`.
    /// # Arguments:
    /// - `reference`: Block reference determinant.
    /// - `det`: Excited determinant in the same parent frame.
    /// # Returns:
    /// - `Excitor`: Excitor with `\hat\tau|{}^z\Psi\rangle = +|D\rangle`.
    pub(super) fn new(
        reference: &ExcitedDeterminant,
        det: ExcitedDeterminant,
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
    /// - `Option<(ExcitedDeterminant, f64)>`: Image and sign, or `None` if a hole is empty or a
    ///   particle is already occupied.
    #[inline(always)]
    pub(super) fn apply(
        &self,
        det: &ExcitedDeterminant,
    ) -> Option<(ExcitedDeterminant, f64)> {
        let (alpha, sa) = apply_excitation(det.alpha, self.holes.0, self.parts.0)?;
        let (beta, sb) = apply_excitation(det.beta, self.holes.1, self.parts.1)?;
        Some((
            ExcitedDeterminant {
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
    /// - `Option<(ExcitedDeterminant, f64)>`: Preimage `|D\rangle` and the sign
    ///   `\langle D'|\hat\tau|D\rangle`, or `None` if `|D'\rangle` has no preimage.
    #[inline(always)]
    pub(super) fn invert(
        &self,
        det: &ExcitedDeterminant,
    ) -> Option<(ExcitedDeterminant, f64)> {
        // A preimage needs every particle occupied and every hole empty in `|D'\rangle`.
        let valid = |occ: u128, holes: u128, parts: u128| occ & parts == parts && occ & holes == 0;
        if !valid(det.alpha, self.holes.0, self.parts.0)
            || !valid(det.beta, self.holes.1, self.parts.1)
        {
            return None;
        }
        let source = ExcitedDeterminant {
            parent: det.parent,
            alpha: undo_excitation(det.alpha, self.holes.0, self.parts.0),
            beta: undo_excitation(det.beta, self.holes.1, self.parts.1),
        };
        let (_, sign) = self.apply(&source)?;
        Some((source, sign))
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
    reference: &ExcitedDeterminant,
    det: &ExcitedDeterminant,
) -> usize {
    ((reference.alpha & !det.alpha).count_ones() + (reference.beta & !det.beta).count_ones())
        as usize
}

/// Projection and expansion spaces of every reference block: the block references, their
/// projection rows up to the cluster rank, the expansion rank of each `e^{\hat T_x}|{}^x\Psi\rangle`
/// and the Hamiltonian-connection generator of each reference frame.
pub(super) struct ReferenceSpaces {
    /// Block reference determinants.
    pub(super) references: Vec<ExcitedDeterminant>,
    /// Prepared block references.
    pub(super) states: Vec<ExcitedState>,
    /// Prepared projection rows of every block with ranks `1..=cluster_rank`.
    pub(super) rows: Vec<Vec<ExcitedState>>,
    /// Number of spatial orbitals of every block.
    pub(super) norb: Vec<usize>,
    /// Uniform connection generator of every block.
    pub(super) generators: Vec<OrthogonalUniformGenerator>,
    /// Highest excitation rank of the cluster operators.
    pub(super) cluster_rank: usize,
    /// Highest rank and cluster size of the expansion of each `e^{\hat T_x}|{}^x\Psi\rangle`.
    pub(super) expansion_rank: usize,
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
            states: Vec::with_capacity(m),
            rows: Vec::with_capacity(m),
            norb: Vec::with_capacity(m),
            generators: Vec::with_capacity(m),
            cluster_rank: options.cluster_rank,
            expansion_rank: 0,
        };

        let mut nelec = 0;
        for z in 0..m {
            let parent = data.space.state(NOCIIndex(z)).parent;
            let (oa, ob) = data.space.occupations(NOCIIndex(z));
            let norb = parents[parent].ca.ncols();
            nelec = (oa.count_ones() + ob.count_ones()) as usize;
            let reference = ExcitedDeterminant {
                parent,
                alpha: oa,
                beta: ob,
            };
            spaces.references.push(reference);
            spaces.states.push(reference.state(parents));
            spaces.rows.push(
                reference_determinants(oa, ob, norb, options.cluster_rank)
                    .into_iter()
                    .skip(1)
                    .map(|(alpha, beta, _)| {
                        ExcitedDeterminant {
                            parent,
                            alpha,
                            beta,
                        }
                        .state(parents)
                    })
                    .collect(),
            );
            spaces.norb.push(norb);
            spaces
                .generators
                .push(OrthogonalUniformGenerator::new((oa, ob), &mocache[parent]));
        }
        spaces.expansion_rank = options.expansion_rank.unwrap_or(nelec).min(nelec);
        spaces
    }
}

/// Population of one reference block on this rank for one propagation step: the global
/// reference population `N_z`, every locally owned excitor with its population `n_{z\mu}`, and
/// the alias table selecting excitors with probability `|n_{z\mu}|/N_{\text{ex}}`. Its size is
/// the number of occupied excitors, never that of the excitor space.
pub(super) struct Block {
    /// Block reference determinant.
    pub(super) reference: ExcitedDeterminant,
    /// Reference population `N_z`.
    pub(super) nref: f64,
    /// Excitors with their populations.
    pub(super) excitors: Vec<(Excitor, f64)>,
    /// Alias table over `|n_{z\mu}|` aligned with `excitors`.
    pub(super) selection: AliasTable,
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

/// Sparse population change of one excitor communicated across MPI ranks. The occupations are
/// split into 64-bit words for MPI transport.
#[repr(C)]
#[derive(Copy, Clone, Default, Equivalence)]
pub(super) struct ExcitorUpdate {
    /// Reference block.
    block: u64,
    /// Low and high words of the alpha occupation.
    alpha: [u64; 2],
    /// Low and high words of the beta occupation.
    beta: [u64; 2],
    /// Signed population change.
    pub(super) dn: f64,
}

impl ExcitorUpdate {
    /// Encode one population change.
    /// # Arguments:
    /// - `key`: Excitor receiving the change.
    /// - `dn`: Signed population change.
    /// # Returns:
    /// - `ExcitorUpdate`: MPI-safe update.
    pub(super) fn new(
        key: ExcitorKey,
        dn: f64,
    ) -> Self {
        let words = |x: u128| [x as u64, (x >> 64) as u64];
        Self {
            block: key.0 as u64,
            alpha: words(key.1),
            beta: words(key.2),
            dn,
        }
    }

    /// Decode the excitor receiving the change.
    /// # Arguments:
    /// - `self`: Update.
    /// # Returns:
    /// - `ExcitorKey`: Reference block and occupations.
    pub(super) fn excitor(&self) -> ExcitorKey {
        let join = |w: [u64; 2]| w[0] as u128 | ((w[1] as u128) << 64);
        (self.block as usize, join(self.alpha), join(self.beta))
    }
}

impl FriAmplitude for ExcitorUpdate {
    /// Return the population change.
    /// # Arguments:
    /// - `self`: Update.
    /// # Returns:
    /// - `f64`: Signed population change.
    fn amplitude(&self) -> f64 {
        self.dn
    }

    /// Replace the population change.
    /// # Arguments:
    /// - `self`: Update.
    /// - `amplitude`: Replacement population change.
    /// # Returns:
    /// - `()`: Updates `dn` in place.
    fn set_amplitude(
        &mut self,
        amplitude: f64,
    ) {
        self.dn = amplitude;
    }
}

impl KeyedUpdate for ExcitorUpdate {
    type Key = (u64, [u64; 2], [u64; 2]);

    /// Return the excitor key of the update.
    /// # Arguments:
    /// - `self`: Update.
    /// # Returns:
    /// - `Self::Key`: Block and occupation words.
    fn key(&self) -> Self::Key {
        (self.block, self.alpha, self.beta)
    }
}
