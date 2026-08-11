// noci/factorise/onebody/cpu/contract.rs
//! CPU dense contractions for factorised one-body NOCI operator contractions.

// External crate imports.
use rayon::prelude::*;

// Crate-root imports.
use crate::noci::types::NOCIScalar;

// Parent/sibling imports.
use super::super::super::SpinFactorisation;
use super::super::plan::OneBodyContraction;
use super::factors::FactorisedOneBodyBlock;

/// Reusable dense one-body contraction buffers.
pub(super) struct OneBodyScratch<T: NOCIScalar> {
    /// Temporary `T^F_{\bar a b}` or `U^F_{a\bar b}` table.
    first_f: Vec<T>,
    /// Temporary `T^S_{\bar a b}` or `U^S_{a\bar b}` table.
    first_s: Vec<T>,
}

impl<T: NOCIScalar> Default for OneBodyScratch<T> {
    /// Construct empty reusable storage for dense one-body applications.
    /// # Returns
    /// - `OneBodyScratch<T>`: Empty reusable contraction buffers.
    fn default() -> Self {
        Self {
            first_f: Vec::new(),
            first_s: Vec::new(),
        }
    }
}

/// Apply one cached spin-factorised parent-pair block of `F + \lambda S`.
/// # Arguments:
/// - `spin`: Shared determinant-space factorisation.
/// - `block`: Cached spin-factorised one-body factors.
/// - `x`: Source determinant vector.
/// - `y`: Output determinant vector to accumulate.
/// - `lambda`: Scalar overlap shift.
/// - `scratch`: Reusable dense contraction buffers.
/// - `partition`: Worker index and worker count for target rows.
/// # Returns
/// - `()`: Adds this parent-pair contribution into `y`.
pub(super) fn apply_one_body_factorised<T: NOCIScalar>(
    spin: &SpinFactorisation,
    block: &FactorisedOneBodyBlock<T>,
    x: &[T],
    y: &mut [T],
    lambda: T,
    scratch: &mut OneBodyScratch<T>,
    partition: (usize, usize),
) {
    match block.contraction {
        OneBodyContraction::AFirst => {
            apply_one_body_a_first(spin, block, x, y, lambda, scratch, partition)
        }
        OneBodyContraction::BFirst => {
            apply_one_body_b_first(spin, block, x, y, lambda, scratch, partition)
        }
    }
}

/// Apply alpha-first contraction for `Y^Q += F^alpha D (S^beta)^T + S^alpha D (F^beta+\lambda S^beta)^T`.
/// # Arguments:
/// - `spin`: Shared determinant-space factorisation.
/// - `block`: Cached parent-pair one-body factors.
/// - `x`: Source determinant vector.
/// - `y`: Output determinant vector to accumulate.
/// - `lambda`: Scalar overlap shift.
/// - `scratch`: Reusable dense contraction buffers.
/// - `partition`: Worker index and worker count for target alpha rows.
/// # Returns
/// - `()`: Adds this parent-pair contribution into `y`.
pub(super) fn apply_one_body_a_first<T: NOCIScalar>(
    spin: &SpinFactorisation,
    block: &FactorisedOneBodyBlock<T>,
    x: &[T],
    y: &mut [T],
    lambda: T,
    scratch: &mut OneBodyScratch<T>,
    partition: (usize, usize),
) {
    let zero = T::from_real(0.0);
    let source = &spin.parents[block.source_parent];
    let target = &spin.parents[block.target_parent];
    let (worker, nworker) = partition;
    let (sa, fa, sb, fb) = block.factors.factors();

    for a0 in (0..block.nta).step_by(512) {
        let a1 = (a0 + 512).min(block.nta);
        let nrow = a1 - a0;

        scratch.first_f.clear();
        scratch.first_s.clear();
        scratch.first_f.resize(nrow * block.nsb, zero);
        scratch.first_s.resize(nrow * block.nsb, zero);

        scratch
            .first_f
            .par_chunks_mut(block.nsb)
            .zip(scratch.first_s.par_chunks_mut(block.nsb))
            .enumerate()
            .for_each(|(row, (tf, ts))| {
                let ta = a0 + row;
                if ta % nworker != worker {
                    return;
                }
                let frow = &fa[ta * block.nsa..(ta + 1) * block.nsa];
                let srow = &sa[ta * block.nsa..(ta + 1) * block.nsa];
                for entry in &source.entries {
                    let xe = x[entry.det];
                    if xe != zero {
                        tf[entry.b] += frow[entry.a] * xe;
                        ts[entry.b] += srow[entry.a] * xe;
                    }
                }
            });

        let updates: Vec<(usize, T)> = target
            .entries
            .par_iter()
            .filter(|entry| entry.a >= a0 && entry.a < a1 && entry.a % nworker == worker)
            .map(|entry| {
                let row = entry.a - a0;
                let tf = &scratch.first_f[row * block.nsb..(row + 1) * block.nsb];
                let ts = &scratch.first_s[row * block.nsb..(row + 1) * block.nsb];
                let sbrow = &sb[entry.b * block.nsb..(entry.b + 1) * block.nsb];
                let fbrow = &fb[entry.b * block.nsb..(entry.b + 1) * block.nsb];
                let mut value = zero;
                for b in 0..block.nsb {
                    value += tf[b] * sbrow[b] + ts[b] * (fbrow[b] + lambda * sbrow[b]);
                }
                (entry.det, value)
            })
            .collect();

        for (det, value) in updates {
            y[det] += value;
        }
    }
}

/// Apply beta-first contraction for `Y^Q += S^alpha D (F^beta)^T + (F^alpha+\lambda S^alpha)D(S^beta)^T`.
/// # Arguments:
/// - `spin`: Shared determinant-space factorisation.
/// - `block`: Cached parent-pair one-body factors.
/// - `x`: Source determinant vector.
/// - `y`: Output determinant vector to accumulate.
/// - `lambda`: Scalar overlap shift.
/// - `scratch`: Reusable dense contraction buffers.
/// - `partition`: Worker index and worker count for target beta rows.
/// # Returns
/// - `()`: Adds this parent-pair contribution into `y`.
pub(super) fn apply_one_body_b_first<T: NOCIScalar>(
    spin: &SpinFactorisation,
    block: &FactorisedOneBodyBlock<T>,
    x: &[T],
    y: &mut [T],
    lambda: T,
    scratch: &mut OneBodyScratch<T>,
    partition: (usize, usize),
) {
    let zero = T::from_real(0.0);
    let source = &spin.parents[block.source_parent];
    let target = &spin.parents[block.target_parent];
    let (worker, nworker) = partition;
    let (sa, fa, sb, fb) = block.factors.factors();

    for b0 in (0..block.ntb).step_by(512) {
        let b1 = (b0 + 512).min(block.ntb);
        let nrow = b1 - b0;

        scratch.first_f.clear();
        scratch.first_s.clear();
        scratch.first_f.resize(nrow * block.nsa, zero);
        scratch.first_s.resize(nrow * block.nsa, zero);

        scratch
            .first_f
            .par_chunks_mut(block.nsa)
            .zip(scratch.first_s.par_chunks_mut(block.nsa))
            .enumerate()
            .for_each(|(row, (uf, us))| {
                let tb = b0 + row;
                if tb % nworker != worker {
                    return;
                }
                let frow = &fb[tb * block.nsb..(tb + 1) * block.nsb];
                let srow = &sb[tb * block.nsb..(tb + 1) * block.nsb];
                for entry in &source.entries {
                    let xe = x[entry.det];
                    if xe != zero {
                        uf[entry.a] += xe * frow[entry.b];
                        us[entry.a] += xe * srow[entry.b];
                    }
                }
            });

        let updates: Vec<(usize, T)> = target
            .entries
            .par_iter()
            .filter(|entry| entry.b >= b0 && entry.b < b1 && entry.b % nworker == worker)
            .map(|entry| {
                let row = entry.b - b0;
                let uf = &scratch.first_f[row * block.nsa..(row + 1) * block.nsa];
                let us = &scratch.first_s[row * block.nsa..(row + 1) * block.nsa];
                let sarow = &sa[entry.a * block.nsa..(entry.a + 1) * block.nsa];
                let farow = &fa[entry.a * block.nsa..(entry.a + 1) * block.nsa];
                let mut value = zero;
                for a in 0..block.nsa {
                    value += sarow[a] * uf[a] + (farow[a] + lambda * sarow[a]) * us[a];
                }
                (entry.det, value)
            })
            .collect();

        for (det, value) in updates {
            y[det] += value;
        }
    }
}
