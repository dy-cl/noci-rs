// elements/orthogonal/eval/hamiltonian.rs

// Crate-root imports.
use crate::AoData;
use crate::NOCIScalar;
use crate::determinant::DeterminantData;
use crate::elements::MOCache;
#[cfg(target_arch = "x86_64")]
use crate::maths::Simd;

// Parent/sibling imports.
use super::batch::OrthogonalElement;
use super::dispatch::dispatch_orthogonal_ranks;

/// The orthogonal Hamiltonian of one parent, nonzero on the diagonal, single and double sectors.
pub(crate) struct OrthogonalHamiltonian<'a, T: NOCIScalar> {
    /// AO data containing the nuclear-repulsion energy.
    pub(crate) ao: &'a AoData,
    /// Parent-specific orthogonal MO Hamiltonian integrals.
    pub(crate) cache: &'a MOCache<T>,
}

impl<T: NOCIScalar> OrthogonalElement<T> for OrthogonalHamiltonian<'_, T> {
    /// Return whether a sector is evaluated in packets: the double sectors are one ERI gather,
    /// while the diagonal and single sectors sum over the occupied orbitals.
    /// # Arguments:
    /// - `ranks`: Alpha and beta excitation ranks of the sector.
    /// # Returns
    /// - `bool`: Whether the sector is `(2,0)`, `(1,1)` or `(0,2)`.
    fn packed_sector(ranks: (usize, usize)) -> bool {
        matches!(ranks, (2, 0) | (1, 1) | (0, 2))
    }

    /// Evaluate one orthogonal Hamiltonian matrix element through the scalar fixed-rank path.
    /// # Arguments:
    /// - `self`: Hamiltonian of the parent.
    /// - `occupation`: Source alpha and beta occupation bitstrings.
    /// - `state`: Child data measured from the source: phases and fixed-rank excitation labels.
    /// # Returns
    /// - `T`: Orthogonal Slater-Condon Hamiltonian matrix element.
    fn value(
        &self,
        occupation: (u128, u128),
        state: &DeterminantData,
    ) -> T {
        xw_hamiltonian_orthogonal(self.ao, self.cache, occupation, state)
    }

    /// Gather and phase one packet of orthogonal double-excitation ERIs, `p(ai||jb)` or `p(ai|jb)`,
    /// with one indexed gather and one lane-wise real phase multiplication.
    /// # Arguments:
    /// - `self`: Hamiltonian of the parent.
    /// - `states`: Data of every lane: fixed-rank excitation labels and phases.
    /// # Returns
    /// - `[T; N]`: Hamiltonian matrix element of every lane.
    /// # Safety
    /// - The caller must establish CPU support for `V`, and every cached orbital label must index
    ///   the integrals.
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

        if RA == 2 && RB == 0 {
            // Alpha double: gather `p(ai||jb)` from antisymmetrised alpha ERIs.
            let shape = cache.eri_aa_asym.shape();

            for lane in 0..N {
                let ex = &states[lane].alpha.cache;
                let i = usize::from(ex.holes[0]);
                let j = usize::from(ex.holes[1]);
                let a = usize::from(ex.particles[0]);
                let b = usize::from(ex.particles[1]);
                indices[lane] = ((a * shape[1] + i) * shape[2] + j) * shape[3] + b;
                phases[lane] = states[lane].phase();
            }

            // Gather the selected two-electron integrals before applying the lane phases.
            let eri = cache
                .eri_aa_asym
                .as_slice()
                .expect("orthogonal alpha-alpha ERIs must be contiguous");
            let values = unsafe { V::gather(eri, &indices) };
            V::multiply_real_lanes(values, &phases).store(&mut out);

            return out;
        }

        if RA == 0 && RB == 2 {
            // Beta double: gather the spin-swapped `p(ai||jb)` contribution.
            let shape = cache.eri_bb_asym.shape();

            for lane in 0..N {
                let ex = &states[lane].beta.cache;
                let i = usize::from(ex.holes[0]);
                let j = usize::from(ex.holes[1]);
                let a = usize::from(ex.particles[0]);
                let b = usize::from(ex.particles[1]);
                indices[lane] = ((a * shape[1] + i) * shape[2] + j) * shape[3] + b;
                phases[lane] = states[lane].phase();
            }

            // Gather the selected two-electron integrals before applying the lane phases.
            let eri = cache
                .eri_bb_asym
                .as_slice()
                .expect("orthogonal beta-beta ERIs must be contiguous");
            let values = unsafe { V::gather(eri, &indices) };
            V::multiply_real_lanes(values, &phases).store(&mut out);

            return out;
        }

        if RA == 1 && RB == 1 {
            // Mixed-spin double: exchange is absent, leaving the Coulomb term `p(ai|jb)`.
            let shape = cache.eri_ab_coul.shape();

            for lane in 0..N {
                let alpha = &states[lane].alpha.cache;
                let beta = &states[lane].beta.cache;
                let i = usize::from(alpha.holes[0]);
                let j = usize::from(beta.holes[0]);
                let a = usize::from(alpha.particles[0]);
                let b = usize::from(beta.particles[0]);
                indices[lane] = ((a * shape[1] + i) * shape[2] + j) * shape[3] + b;
                phases[lane] = states[lane].phase();
            }

            // Gather the selected two-electron integrals before applying the lane phases.
            let eri = cache
                .eri_ab_coul
                .as_slice()
                .expect("orthogonal alpha-beta ERIs must be contiguous");
            let values = unsafe { V::gather(eri, &indices) };
            V::multiply_real_lanes(values, &phases).store(&mut out);
        }

        out
    }
}

/// Evaluate `\langle D|\hat H|\Phi_x\rangle` from a reduced orthogonal excitation.
/// The fixed rank `(R_\alpha,R_\beta)` is restricted by the one- plus two-body Hamiltonian to
/// `(0,0)`, `(1,0)`, `(0,1)`, `(2,0)`, `(1,1)`, or `(0,2)`.
/// # Arguments:
/// - `ao`: AO data containing the nuclear-repulsion energy.
/// - `cache`: Parent-specific orthogonal MO Hamiltonian integrals.
/// - `source`: Source alpha and beta occupation bitstrings.
/// - `state`: Child data measured from the source: phases and fixed-rank excitation labels.
/// # Returns
/// - `T`: Orthogonal Slater-Condon Hamiltonian matrix element.
#[inline(always)]
pub(crate) fn xw_hamiltonian_orthogonal<T: NOCIScalar>(
    ao: &AoData,
    cache: &MOCache<T>,
    source: (u128, u128),
    state: &DeterminantData,
) -> T {
    let ranks = (
        usize::from(state.alpha.cache.rank),
        usize::from(state.beta.cache.rank),
    );

    // Dispatch by fixed spin excitation rank. This keeps Slater-Condon formulas compile-time
    // specialised while returning zero for ranks unsupported by a two-body Hamiltonian.
    dispatch_orthogonal_ranks!(
        ranks,
        |RA, RB| xw_hamiltonian_orthogonal_const::<T, RA, RB>(ao, cache, source, state,),
        T::from_real(0.0),
    )
}

/// Evaluate a fixed-rank orthogonal Slater-Condon Hamiltonian matrix element.
/// The compile-time branch implements the diagonal, alpha/beta singles, and three double sectors
/// without rediscovering excitation masks or fermionic phases.
/// For excitation phase `p`, singles use `p(h_ai + sum_j (ai||jj))`; same-spin doubles use
/// `p(ai||jb)`, mixed-spin doubles use `p(ai|jb)`, and ranks above two vanish.
/// # Arguments:
/// - `ao`: AO data containing `E_\mathrm{nuc}`.
/// - `cache`: Parent-specific one- and two-electron MO integrals.
/// - `source`: Source alpha and beta occupation bitstrings.
/// - `state`: Child data measured from the source: phases and fixed-rank orbital labels.
/// # Returns
/// - `T`: Fixed-rank Hamiltonian matrix element.
#[inline(always)]
fn xw_hamiltonian_orthogonal_const<T: NOCIScalar, const RA: usize, const RB: usize>(
    ao: &AoData,
    cache: &MOCache<T>,
    source: (u128, u128),
    state: &DeterminantData,
) -> T {
    let alpha = &state.alpha.cache;
    let beta = &state.beta.cache;
    let phase = T::from_real(state.phase());

    if RA == 0 && RB == 0 {
        // Diagonal element: accumulate nuclear, one-electron, same-spin, and opposite-spin terms
        // from occupied source orbitals.
        // `H_xx = E_\mathrm{nuc} + \sum_{i\sigma}h^\sigma_{ii}`
        // `+ \frac12\sum_{ij\sigma}\langle ii||jj\rangle`
        // `+ \sum_{i\in O_\alpha,j\in O_\beta}(ii|jj)`.
        let mut h = T::from_real(ao.enuc);
        let mut bits = source.0;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            h += cache.ha[(i, i)];
        }

        let mut bits = source.1;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            h += cache.hb[(i, i)];
        }

        let mut bits_i = source.0;
        while bits_i != 0 {
            let i = bits_i.trailing_zeros() as usize;
            bits_i &= bits_i - 1;
            let mut bits_j = source.0;
            while bits_j != 0 {
                let j = bits_j.trailing_zeros() as usize;
                bits_j &= bits_j - 1;
                h += T::from_real(0.5) * cache.eri_aa_asym[(i, i, j, j)];
            }
        }

        let mut bits_i = source.1;
        while bits_i != 0 {
            let i = bits_i.trailing_zeros() as usize;
            bits_i &= bits_i - 1;
            let mut bits_j = source.1;
            while bits_j != 0 {
                let j = bits_j.trailing_zeros() as usize;
                bits_j &= bits_j - 1;
                h += T::from_real(0.5) * cache.eri_bb_asym[(i, i, j, j)];
            }
        }

        let mut bits_i = source.0;
        while bits_i != 0 {
            let i = bits_i.trailing_zeros() as usize;
            bits_i &= bits_i - 1;
            let mut bits_j = source.1;
            while bits_j != 0 {
                let j = bits_j.trailing_zeros() as usize;
                bits_j &= bits_j - 1;
                h += cache.eri_ab_coul[(i, i, j, j)];
            }
        }

        return h;
    }

    if RA == 1 && RB == 0 {
        // Alpha single: remove the hole from alpha Coulomb/exchange sums while retaining every
        // beta occupied contribution.
        let i = usize::from(alpha.holes[0]);
        let a = usize::from(alpha.particles[0]);

        // `H_{a\leftarrow i,x} = p[h^\alpha_{ai}`
        // `+ \sum_{j\in O_\alpha\setminus i}(ai||jj)`
        // `+ \sum_{j\in O_\beta}(ai|jj)]`.
        let mut h = cache.ha[(a, i)];
        let mut bits = source.0 & !(1u128 << i);
        while bits != 0 {
            let j = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            h += cache.eri_aa_asym[(a, i, j, j)];
        }

        let mut bits = source.1;
        while bits != 0 {
            let j = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            h += cache.eri_ab_coul[(a, i, j, j)];
        }

        return phase * h;
    }

    if RA == 0 && RB == 1 {
        // Beta single is the spin-swapped alpha-single expression.
        let i = usize::from(beta.holes[0]);
        let a = usize::from(beta.particles[0]);

        // `H_{a\leftarrow i,x} = p[h^\beta_{ai}`
        // `+ \sum_{j\in O_\beta\setminus i}(ai||jj)`
        // `+ \sum_{j\in O_\alpha}(jj|ia)]`.
        let mut h = cache.hb[(a, i)];
        let mut bits = source.1 & !(1u128 << i);
        while bits != 0 {
            let j = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            h += cache.eri_bb_asym[(a, i, j, j)];
        }

        let mut bits = source.0;
        while bits != 0 {
            let j = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            h += cache.eri_ab_coul[(j, j, i, a)];
        }

        return phase * h;
    }

    if RA == 2 && RB == 0 {
        // Same-spin alpha double: only the antisymmetrised two-electron contribution remains.
        let i = usize::from(alpha.holes[0]);
        let j = usize::from(alpha.holes[1]);
        let a = usize::from(alpha.particles[0]);
        let b = usize::from(alpha.particles[1]);

        // `H_{ab\leftarrow ij,x} = p\langle ai||jb\rangle` for alpha-alpha doubles.
        return phase * cache.eri_aa_asym[(a, i, j, b)];
    }

    if RA == 0 && RB == 2 {
        // Same-spin beta double is the spin-swapped alpha-double expression.
        let i = usize::from(beta.holes[0]);
        let j = usize::from(beta.holes[1]);
        let a = usize::from(beta.particles[0]);
        let b = usize::from(beta.particles[1]);

        // `H_{ab\leftarrow ij,x} = p\langle ai||jb\rangle` for beta-beta doubles.
        return phase * cache.eri_bb_asym[(a, i, j, b)];
    }

    if RA == 1 && RB == 1 {
        // Mixed-spin double: only the Coulomb coupling between alpha and beta replacements remains.
        let i = usize::from(alpha.holes[0]);
        let j = usize::from(beta.holes[0]);
        let a = usize::from(alpha.particles[0]);
        let b = usize::from(beta.particles[0]);

        // `H_{ab\leftarrow ij,x} = p(ai|jb)` for alpha-beta doubles.
        return phase * cache.eri_ab_coul[(a, i, j, b)];
    }

    T::from_real(0.0)
}
