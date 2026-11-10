// elements/orthogonal/eval/overlap.rs

// Crate-root imports.
use crate::NOCIScalar;
use crate::ReducedTwoSpinState;
#[cfg(target_arch = "x86_64")]
use crate::maths::Simd;

// Parent/sibling imports.
use super::batch::OrthogonalElement;
use super::dispatch::dispatch_orthogonal_ranks;

/// Evaluate `\langle D|\Phi_x\rangle` from a reduced orthogonal excitation.
/// Orthonormality of the parent orbitals restricts the fixed rank `(R_\alpha,R_\beta)` to the
/// diagonal sector `(0,0)`; every other supported rank returns zero.
/// # Arguments:
/// - `state`: Reduced phase and fixed-rank excitation labels connecting source to child.
/// # Returns
/// - `T`: Orthogonal Slater-Condon overlap matrix element.
#[inline(always)]
pub(crate) fn xw_overlap_orthogonal<T: NOCIScalar>(state: &ReducedTwoSpinState) -> T {
    let ranks = (
        usize::from(state.excitation_cache.alpha.rank),
        usize::from(state.excitation_cache.beta.rank),
    );

    // Dispatch by fixed spin excitation rank. This keeps the Slater-Condon formula compile-time
    // specialised while returning zero for every excited sector.
    dispatch_orthogonal_ranks!(
        ranks,
        |RA, RB| xw_overlap_orthogonal_const::<T, RA, RB>(state),
        T::from_real(0.0),
    )
}

/// Evaluate a fixed-rank orthogonal Slater-Condon overlap matrix element.
/// Only the diagonal sector survives, giving `S_{xx} = p` for the reduced phase `p`.
/// # Arguments:
/// - `state`: Reduced phase and fixed-rank orbital labels.
/// # Returns
/// - `T`: Fixed-rank overlap matrix element.
#[inline(always)]
fn xw_overlap_orthogonal_const<T: NOCIScalar, const RA: usize, const RB: usize>(
    state: &ReducedTwoSpinState
) -> T {
    if RA == 0 && RB == 0 {
        // Diagonal element: identical occupations overlap with the reduced phase.
        // `S_{xx} = p`.
        return T::from_real(state.phase);
    }

    T::from_real(0.0)
}

/// The orthogonal overlap, nonzero only on the diagonal sector, `S_{xx} = p`.
pub(crate) struct OrthogonalOverlap;

impl<T: NOCIScalar> OrthogonalElement<T> for OrthogonalOverlap {
    /// Return whether a sector is evaluated in packets: only the diagonal sector is nonzero.
    /// # Arguments:
    /// - `ranks`: Alpha and beta excitation ranks of the sector.
    /// # Returns
    /// - `bool`: Whether the sector is `(0,0)`.
    fn packed_sector(ranks: (usize, usize)) -> bool {
        ranks == (0, 0)
    }

    /// Evaluate one orthogonal overlap through the scalar fixed-rank path.
    /// # Arguments:
    /// - `self`: Overlap operator.
    /// - `occupation`: Source alpha and beta occupation bitstrings, unused by the overlap.
    /// - `state`: Reduced phase and fixed-rank excitation labels connecting source to child.
    /// # Returns
    /// - `T`: Orthogonal overlap matrix element.
    fn value(
        &self,
        _occupation: (u128, u128),
        state: &ReducedTwoSpinState,
    ) -> T {
        xw_overlap_orthogonal(state)
    }

    /// Phase one packet of orthogonal diagonal overlaps, `S_{xx} = p`, as one lane-wise real phase
    /// multiplication of packed one.
    /// # Arguments:
    /// - `self`: Overlap operator.
    /// - `states`: Reduced excitation labels and phases of every lane.
    /// # Returns
    /// - `[T; N]`: Overlap of every lane.
    /// # Safety
    /// - The caller must establish CPU support for `V`.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn packed<V: Simd<N, Scalar = T>, const N: usize, const RA: usize, const RB: usize>(
        &self,
        states: &[ReducedTwoSpinState; N],
    ) -> [T; N] {
        let mut out = [T::from_real(0.0); N];
        if RA == 0 && RB == 0 {
            let mut phases = [1.0f64; N];
            for lane in 0..N {
                phases[lane] = states[lane].phase;
            }
            V::multiply_real_lanes(V::one(), &phases).store(&mut out);
        }
        out
    }
}
