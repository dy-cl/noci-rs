// elements/orthogonal/eval/fock.rs

// Crate-root imports.
use crate::NOCIScalar;
use crate::determinant::DeterminantData;
use crate::elements::FockMOCache;
#[cfg(target_arch = "x86_64")]
use crate::maths::Simd;

// Parent/sibling imports.
use super::batch::OrthogonalElement;
use super::dispatch::dispatch_orthogonal_ranks;

/// The orthogonal Fock operator of one parent, nonzero on the diagonal and single sectors.
pub(crate) struct OrthogonalFock<'a, T: NOCIScalar> {
    /// Parent-specific orthogonal MO Fock matrices.
    pub(crate) cache: &'a FockMOCache<T>,
}

impl<T: NOCIScalar> OrthogonalElement<T> for OrthogonalFock<'_, T> {
    /// Return whether a sector is evaluated in packets: the single sectors are one Fock gather,
    /// while the diagonal sector sums over the occupied orbitals.
    /// # Arguments:
    /// - `ranks`: Alpha and beta excitation ranks of the sector.
    /// # Returns
    /// - `bool`: Whether the sector is `(1,0)` or `(0,1)`.
    fn packed_sector(ranks: (usize, usize)) -> bool {
        matches!(ranks, (1, 0) | (0, 1))
    }

    /// Evaluate one orthogonal Fock matrix element through the scalar fixed-rank path.
    /// # Arguments:
    /// - `self`: Fock operator of the parent.
    /// - `occupation`: Source alpha and beta occupation bitstrings.
    /// - `state`: Child data measured from the source: phases and fixed-rank excitation labels.
    /// # Returns
    /// - `T`: Orthogonal Slater-Condon Fock matrix element.
    fn value(
        &self,
        occupation: (u128, u128),
        state: &DeterminantData,
    ) -> T {
        xw_fock_orthogonal(self.cache, occupation, state)
    }

    /// Gather and phase one packet of orthogonal single-excitation Fock elements, `pF_{ai}`, with
    /// one indexed gather and one lane-wise real phase multiplication.
    /// # Arguments:
    /// - `self`: Fock operator of the parent.
    /// - `states`: Data of every lane: fixed-rank excitation labels and phases.
    /// # Returns
    /// - `[T; N]`: Fock matrix element of every lane.
    /// # Safety
    /// - The caller must establish CPU support for `V`, and every cached orbital label must index
    ///   the Fock matrices.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn packed<V: Simd<N, Scalar = T>, const N: usize, const RA: usize, const RB: usize>(
        &self,
        states: &[DeterminantData; N],
    ) -> [T; N] {
        let cache = self.cache;
        let mut out = [T::from_real(0.0); N];
        let mut indices = [0usize; N];
        let mut phases = [1.0f64; N];

        if RA == 1 && RB == 0 {
            // Alpha single: gather `pF_{ai}` from the alpha Fock matrix.
            let shape = cache.fa.shape();

            for lane in 0..N {
                let ex = &states[lane].alpha.cache;
                let i = usize::from(ex.holes[0]);
                let a = usize::from(ex.particles[0]);
                indices[lane] = a * shape[1] + i;
                phases[lane] = states[lane].phase();
            }

            // Gather the selected `F_{ai}` entries, then restore each determinant phase.
            let fock = cache
                .fa
                .as_slice()
                .expect("orthogonal alpha Fock matrix must be contiguous");
            let values = unsafe { V::gather(fock, &indices) };
            V::multiply_real_lanes(values, &phases).store(&mut out);

            return out;
        }

        if RA == 0 && RB == 1 {
            // Beta single: gather the spin-swapped `pF_{ai}` contribution.
            let shape = cache.fb.shape();

            for lane in 0..N {
                let ex = &states[lane].beta.cache;
                let i = usize::from(ex.holes[0]);
                let a = usize::from(ex.particles[0]);
                indices[lane] = a * shape[1] + i;
                phases[lane] = states[lane].phase();
            }

            // Gather the selected `F_{ai}` entries, then restore each determinant phase.
            let fock = cache
                .fb
                .as_slice()
                .expect("orthogonal beta Fock matrix must be contiguous");
            let values = unsafe { V::gather(fock, &indices) };
            V::multiply_real_lanes(values, &phases).store(&mut out);
        }

        out
    }
}

/// Evaluate `\langle D|\hat F|\Phi_x\rangle` from a reduced orthogonal excitation.
/// The fixed rank `(R_\alpha,R_\beta)` is restricted by the one-body Fock operator to `(0,0)`,
/// `(1,0)`, or `(0,1)`; every other supported rank returns zero.
/// # Arguments:
/// - `cache`: Parent-specific orthogonal MO Fock matrices.
/// - `source`: Source alpha and beta occupation bitstrings.
/// - `state`: Child data measured from the source: phases and fixed-rank excitation labels.
/// # Returns
/// - `T`: Orthogonal Slater-Condon Fock matrix element.
#[inline(always)]
pub(crate) fn xw_fock_orthogonal<T: NOCIScalar>(
    cache: &FockMOCache<T>,
    source: (u128, u128),
    state: &DeterminantData,
) -> T {
    let ranks = (
        usize::from(state.alpha.cache.rank),
        usize::from(state.beta.cache.rank),
    );

    // Dispatch by fixed spin excitation rank. This keeps Slater-Condon formulas compile-time
    // specialised while returning zero for ranks unsupported by a one-body operator.
    dispatch_orthogonal_ranks!(
        ranks,
        |RA, RB| xw_fock_orthogonal_const::<T, RA, RB>(cache, source, state),
        T::from_real(0.0),
    )
}

/// Evaluate a fixed-rank orthogonal Slater-Condon Fock matrix element.
/// The compile-time branch implements the diagonal and alpha/beta single sectors without
/// rediscovering excitation masks or fermionic phases.
/// For excitation phase `p`, singles use `pF_{ai}` and ranks above one vanish.
/// # Arguments:
/// - `cache`: Parent-specific alpha and beta MO Fock matrices.
/// - `source`: Source alpha and beta occupation bitstrings.
/// - `state`: Child data measured from the source: phases and fixed-rank orbital labels.
/// # Returns
/// - `T`: Fixed-rank Fock matrix element.
#[inline(always)]
fn xw_fock_orthogonal_const<T: NOCIScalar, const RA: usize, const RB: usize>(
    cache: &FockMOCache<T>,
    source: (u128, u128),
    state: &DeterminantData,
) -> T {
    let alpha = &state.alpha.cache;
    let beta = &state.beta.cache;
    let phase = T::from_real(state.phase());

    if RA == 0 && RB == 0 {
        // Diagonal element: accumulate occupied alpha and beta orbital Fock energies.
        // `F_xx = \sum_{i\in O_\alpha}F^\alpha_{ii} + \sum_{i\in O_\beta}F^\beta_{ii}`.
        let mut f = T::from_real(0.0);
        let mut bits = source.0;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            f += cache.fa[(i, i)];
        }

        let mut bits = source.1;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            f += cache.fb[(i, i)];
        }

        return f;
    }

    if RA == 1 && RB == 0 {
        // Alpha single: only the signed alpha Fock coupling remains.
        let i = usize::from(alpha.holes[0]);
        let a = usize::from(alpha.particles[0]);

        // `F_{a\leftarrow i,x} = pF^\alpha_{ai}`.
        return phase * cache.fa[(a, i)];
    }

    if RA == 0 && RB == 1 {
        // Beta single is the spin-swapped alpha-single expression.
        let i = usize::from(beta.holes[0]);
        let a = usize::from(beta.particles[0]);

        // `F_{a\leftarrow i,x} = pF^\beta_{ai}`.
        return phase * cache.fb[(a, i)];
    }

    T::from_real(0.0)
}
