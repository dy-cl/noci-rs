// nocc/terms/factors.rs
//! Dense blocks of every tensor factor used by a set of planned tables.

// Standard library imports.
use std::collections::HashMap;

// External crate imports.
use rayon::prelude::*;

// Parent/sibling imports.
use super::plan::{BlockKey, TablePlan};
use super::tensors::{
    Amplitudes, SpaceKind, TensorKind, Tensors, amplitude_element, evaluate_factor, space_orbitals,
};
use super::workspace::Values;

/// Dense tensor blocks of every tensor kind and slot-space pattern used by a set of tables.
pub(super) struct FactorBlocks {
    /// Row-major block data over the orbitals of each slot space.
    pub(super) blocks: HashMap<BlockKey, Values>,
    /// Number of orbitals in the core, active and virtual spaces.
    pub(super) dims: [usize; 3],
}

impl FactorBlocks {
    /// Build every dense factor block used by the given table plans.
    /// Block elements are the runtime tensor elements over the orbitals of each slot space, so
    /// every tensor convention is shared with the element-wise evaluator.
    /// # Arguments:
    /// - `plans`: Plans of the tables to evaluate.
    /// - `tensors`: Runtime tensors, including the current amplitudes when needed.
    /// # Returns:
    /// - `Self`: Dense blocks keyed by tensor kind and slot spaces.
    /// # Panics
    /// - Panics if a plan needs an amplitude block and `tensors` holds no amplitudes.
    pub(super) fn build_factor_blocks(
        plans: &[&TablePlan],
        tensors: &Tensors<'_>,
    ) -> Self {
        let dims = SpaceKind::ALL.map(|s| space_orbitals(tensors.spaces, s).len());

        let mut keys = plans
            .iter()
            .flat_map(|p| p.keys.iter().cloned())
            .collect::<Vec<_>>();
        keys.sort_unstable();
        keys.dedup();

        let blocks = keys
            .into_par_iter()
            .map(|key| {
                let data = dense_factor_block(&key, tensors);
                (key, data)
            })
            .collect();

        Self { blocks, dims }
    }
}

/// Build one dense factor block by evaluating the runtime tensor element at every orbital tuple.
/// Amplitude blocks take the scalar type of the amplitudes; every other block is real.
/// # Arguments:
/// - `key`: Tensor kind and slot spaces.
/// - `tensors`: Runtime tensors.
/// # Returns:
/// - `Values`: Row-major block elements.
/// # Panics
/// - Panics if the block is an amplitude block and `tensors` holds no amplitudes.
fn dense_factor_block(
    key: &BlockKey,
    tensors: &Tensors<'_>,
) -> Values {
    let kind = key.0;
    match tensors.amplitudes {
        Some(Amplitudes::Complex(t)) if matches!(kind, TensorKind::T1 | TensorKind::T2) => {
            Values::Complex(dense_block(key, tensors, |slots, idx| {
                amplitude_element(kind, slots, idx, t)
            }))
        }
        _ => Values::Real(dense_block(key, tensors, |slots, idx| {
            evaluate_factor(kind, slots, idx, tensors)
        })),
    }
}

/// Evaluate one element function at every orbital tuple of a block, in row-major order.
/// # Arguments:
/// - `key`: Tensor kind and slot spaces.
/// - `tensors`: Runtime tensors, for the orbitals of every space.
/// - `element`: Element at the upper and lower slot ids and the orbital tuple.
/// # Returns:
/// - `Vec<T>`: Row-major block elements.
fn dense_block<T>(
    key: &BlockKey,
    tensors: &Tensors<'_>,
    element: impl Fn((&[u16], &[u16]), &[usize]) -> T,
) -> Vec<T> {
    let (_, spaces) = key;
    let orbitals = spaces
        .iter()
        .map(|&s| space_orbitals(tensors.spaces, s))
        .collect::<Vec<_>>();
    let dims = orbitals.iter().map(|o| o.len()).collect::<Vec<_>>();
    let size = dims.iter().product::<usize>();

    // Slot ids `0..k` index the orbital tuple; the first half are upper slots.
    let k = spaces.len();
    let upper = (0..k as u16 / 2).collect::<Vec<_>>();
    let lower = (k as u16 / 2..k as u16).collect::<Vec<_>>();

    // A slot over an empty orbital space gives an empty block.
    if size == 0 {
        return Vec::new();
    }

    // Odometer over the orbital tuples in row-major order.
    let mut idx = orbitals.iter().map(|o| o[0]).collect::<Vec<_>>();
    let mut pos = vec![0usize; k];
    let mut data = Vec::with_capacity(size);
    for _ in 0..size {
        data.push(element((&upper, &lower), &idx));

        let mut slot = k;
        while slot > 0 {
            slot -= 1;
            pos[slot] += 1;
            if pos[slot] < dims[slot] {
                idx[slot] = orbitals[slot][pos[slot]];
                break;
            }
            pos[slot] = 0;
            idx[slot] = orbitals[slot][0];
        }
    }

    data
}
