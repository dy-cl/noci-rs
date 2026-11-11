// elements/operator.rs
//! Operators whose determinant-pair matrix elements the dispatch in `pairs` evaluates.
//!
//! Each operator supplies its evaluation in the three engines: a parent-local batched kernel for
//! the orthogonal Slater-Condon rules, a reference-pair batch for the extended nonorthogonal Wick
//! theorem, and a single-pair evaluation for the generalised Slater-Condon rules. The dispatch
//! itself, which routes every pair to one engine, is shared by all operators. Each operator times
//! its engine evaluations with its own counters.

// Crate-root imports.
use crate::NOCIScalar;
use crate::determinant::DeterminantData;
use crate::determinant::DeterminantState;
use crate::elements::{DetPair, FockData, NOCIData};
use crate::time_call;

// Parent/sibling imports.
use super::naive::{calculate_fs_pair_naive, calculate_hs_pair_naive, calculate_s_pair_naive};
use super::nonorthogonalwicks::{
    WickScratchSpin, WicksPairView, calculate_fs_pairs_wicks, calculate_s_pairs_wicks,
    xw_hamiltonian_overlap_prepared_batched,
};
use super::orthogonal::{
    OrthogonalFock, OrthogonalHamiltonian, OrthogonalOverlap, xw_orthogonal_batched,
};

/// An operator whose matrix elements between pairs of determinants can be evaluated by every
/// engine.
pub(crate) trait PairOperator<T: NOCIScalar> {
    /// Matrix element of one determinant pair.
    type Out: Copy;

    /// Largest excitation rank connecting two determinants of one orthonormal frame, so that
    /// same-parent pairs differing by more than `2 RANK` occupation bits are zero.
    const RANK: usize;

    /// Timing counter of the dispatch of this operator.
    const TIMER: fn(u64);

    /// Return the vanishing matrix element.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `Self::Out`: Zero matrix element.
    fn zero() -> Self::Out;

    /// Return whether same-parent pairs of a parent may use the orthogonal Slater-Condon rules.
    /// # Arguments:
    /// - `self`: Operator.
    /// - `data`: Shared data required for NOCI matrix-element evaluation.
    /// - `parent`: Parent orbital frame.
    /// # Returns:
    /// - `bool`: Whether the parent's MO cache is Hermitian orthonormal.
    fn orthonormal(
        &self,
        data: &NOCIData<'_, T>,
        parent: usize,
    ) -> bool;

    /// Evaluate a parent-local batch of same-parent pairs with the orthogonal kernels.
    /// # Arguments:
    /// - `self`: Operator.
    /// - `data`: Shared data required for NOCI matrix-element evaluation.
    /// - `parent`: Shared orthonormal parent of every pair in the batch.
    /// - `occupations`: Ket alpha and beta occupations of each pair.
    /// - `connections`: Bra data measured from the ket of each pair.
    /// - `values`: Matrix elements in batch order.
    /// # Returns:
    /// - `()`: Writes every matrix element of the batch into `values`.
    fn orthogonal(
        &self,
        data: &NOCIData<'_, T>,
        parent: usize,
        occupations: &[(u128, u128)],
        connections: &[DeterminantData],
        values: &mut [Self::Out],
    );

    /// Evaluate the pairs of one ordered reference pair with the extended nonorthogonal Wick
    /// theorem.
    /// # Arguments:
    /// - `self`: Operator.
    /// - `data`: Shared data required for NOCI matrix-element evaluation.
    /// - `w`: Wick intermediates of the ordered reference pair.
    /// - `requests`: Output positions with their determinant pairs.
    /// - `scratch`: Scratch space for Wick's calculations.
    /// - `out`: Matrix elements indexed by output position.
    /// # Returns:
    /// - `()`: Writes every requested matrix element into `out`.
    fn wicks(
        &self,
        data: &NOCIData<'_, T>,
        w: &WicksPairView<'_, T>,
        requests: &[(usize, DetPair)],
        scratch: &mut WickScratchSpin<T>,
        out: &mut [Self::Out],
    );

    /// Evaluate one pair with the generalised Slater-Condon rules.
    /// # Arguments:
    /// - `self`: Operator.
    /// - `data`: Shared data required for NOCI matrix-element evaluation.
    /// - `bra`: Bra determinant.
    /// - `ket`: Ket determinant.
    /// # Returns:
    /// - `Self::Out`: Matrix element of the pair.
    fn naive(
        &self,
        data: &NOCIData<'_, T>,
        bra: &DeterminantState,
        ket: &DeterminantState,
    ) -> Self::Out;

    /// Measure the discrepancy between two evaluations of one matrix element.
    /// # Arguments:
    /// - `a`: First evaluation.
    /// - `b`: Second evaluation.
    /// # Returns:
    /// - `(f64, f64)`: Total and largest elementwise absolute discrepancy.
    fn discrepancy(
        a: Self::Out,
        b: Self::Out,
    ) -> (f64, f64);
}

/// Overlap operator, `S_{xw} = \langle x|w\rangle`.
pub(crate) struct Overlap;

/// Shifted generalised Fock operator, `M_{xw} = F_{xw} - E_0 S_{xw}`; `E_0 = 0` gives the Fock
/// matrix itself.
pub(crate) struct Fock<'a, T: NOCIScalar> {
    /// Spin-resolved Fock matrices in the AO and parent MO bases.
    pub(crate) fock: &'a FockData<'a, T>,
    /// Shift `E_0` multiplying the overlap.
    pub(crate) shift: f64,
}

/// Hamiltonian and overlap operators together, `(H_{xw}, S_{xw})`.
pub(crate) struct HamiltonianOverlap;

impl<T: NOCIScalar> PairOperator<T> for Overlap {
    type Out = T;
    const RANK: usize = 0;
    const TIMER: fn(u64) = crate::timers::noci::add_calculate_s_pair;

    /// Return the vanishing overlap.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `T`: Zero.
    fn zero() -> T {
        T::from_real(0.0)
    }

    /// Return whether the parent's Hamiltonian MO cache is Hermitian orthonormal.
    /// # Arguments:
    /// - `self`: Overlap operator.
    /// - `data`: Shared data holding the parent MO caches.
    /// - `parent`: Parent orbital frame.
    /// # Returns:
    /// - `bool`: Whether same-parent overlaps may use the orthogonal rules.
    fn orthonormal(
        &self,
        data: &NOCIData<'_, T>,
        parent: usize,
    ) -> bool {
        data.mocache
            .is_some_and(|mocache| mocache[parent].orthogonal_slater_condon)
    }

    /// Evaluate a parent-local batch of orthogonal overlaps.
    /// # Arguments:
    /// - `self`: Overlap operator.
    /// - `data`: Unused, as orthogonal overlaps need no integrals.
    /// - `parent`: Unused, as orthogonal overlaps need no integrals.
    /// - `occupations`: Unused, as orthogonal overlaps depend only on the connections.
    /// - `connections`: Bra data measured from the ket of each pair.
    /// - `values`: Overlaps in batch order.
    /// # Returns:
    /// - `()`: Writes every overlap of the batch into `values`.
    fn orthogonal(
        &self,
        _data: &NOCIData<'_, T>,
        _parent: usize,
        occupations: &[(u128, u128)],
        connections: &[DeterminantData],
        values: &mut [T],
    ) {
        time_call!(crate::timers::noci::add_calculate_s_pair_orthogonal, {
            xw_orthogonal_batched(&OrthogonalOverlap, occupations, connections, values);
        });
    }

    /// Evaluate the overlaps of one ordered reference pair with Wick's theorem.
    /// # Arguments:
    /// - `self`: Overlap operator.
    /// - `data`: Shared data holding the determinant space.
    /// - `w`: Wick intermediates of the ordered reference pair.
    /// - `requests`: Output positions with their determinant pairs.
    /// - `scratch`: Scratch space for Wick's calculations.
    /// - `out`: Overlaps indexed by output position.
    /// # Returns:
    /// - `()`: Writes every requested overlap into `out`.
    fn wicks(
        &self,
        data: &NOCIData<'_, T>,
        w: &WicksPairView<'_, T>,
        requests: &[(usize, DetPair)],
        scratch: &mut WickScratchSpin<T>,
        out: &mut [T],
    ) {
        time_call!(crate::timers::noci::add_calculate_s_pair_wicks, {
            let overlaps = calculate_s_pairs_wicks(w, data.space, requests, scratch);
            for ((output, _), s) in requests.iter().zip(overlaps) {
                out[*output] = s;
            }
        });
    }

    /// Evaluate one overlap with the generalised Slater-Condon rules.
    /// # Arguments:
    /// - `self`: Overlap operator.
    /// - `data`: Shared data required for NOCI matrix-element evaluation.
    /// - `bra`: Bra determinant.
    /// - `ket`: Ket determinant.
    /// # Returns:
    /// - `T`: Overlap of the pair.
    fn naive(
        &self,
        data: &NOCIData<'_, T>,
        bra: &DeterminantState,
        ket: &DeterminantState,
    ) -> T {
        time_call!(crate::timers::noci::add_calculate_s_pair_naive, {
            calculate_s_pair_naive(data, bra, ket)
        })
    }

    /// Measure the discrepancy between two overlaps.
    /// # Arguments:
    /// - `a`: First evaluation.
    /// - `b`: Second evaluation.
    /// # Returns:
    /// - `(f64, f64)`: Absolute discrepancy, as both total and largest.
    fn discrepancy(
        a: T,
        b: T,
    ) -> (f64, f64) {
        let d = (a - b).abs();
        (d, d)
    }
}

impl<T: NOCIScalar> PairOperator<T> for Fock<'_, T> {
    type Out = T;
    const RANK: usize = 1;
    const TIMER: fn(u64) = crate::timers::noci::add_calculate_f_pair;

    /// Return the vanishing shifted Fock element.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `T`: Zero.
    fn zero() -> T {
        T::from_real(0.0)
    }

    /// Return whether the parent's Fock MO cache is Hermitian orthonormal.
    /// # Arguments:
    /// - `self`: Shifted Fock operator.
    /// - `data`: Unused, as the Fock MO caches carry their own orthonormality.
    /// - `parent`: Parent orbital frame.
    /// # Returns:
    /// - `bool`: Whether same-parent elements may use the orthogonal rules.
    fn orthonormal(
        &self,
        _data: &NOCIData<'_, T>,
        parent: usize,
    ) -> bool {
        self.fock.fock_mocache[parent].orthogonal_slater_condon
    }

    /// Evaluate a parent-local batch of orthogonal shifted Fock elements.
    /// # Arguments:
    /// - `self`: Shifted Fock operator.
    /// - `data`: Unused, as the Fock MO caches hold the integrals.
    /// - `parent`: Shared orthonormal parent of the batch.
    /// - `occupations`: Ket alpha and beta occupations of each pair.
    /// - `connections`: Bra data measured from the ket of each pair.
    /// - `values`: Shifted Fock elements in batch order.
    /// # Returns:
    /// - `()`: Writes every element of the batch into `values`.
    fn orthogonal(
        &self,
        _data: &NOCIData<'_, T>,
        parent: usize,
        occupations: &[(u128, u128)],
        connections: &[DeterminantData],
        values: &mut [T],
    ) {
        time_call!(crate::timers::noci::add_calculate_f_pair_orthogonal, {
            // Evaluate `F` and `S` with their own batched kernels before shifting.
            let fock = OrthogonalFock {
                cache: &self.fock.fock_mocache[parent],
            };
            xw_orthogonal_batched(&fock, occupations, connections, values);
            let mut s = vec![T::from_real(0.0); connections.len()];
            xw_orthogonal_batched(&OrthogonalOverlap, occupations, connections, &mut s);
            for (value, &s) in values.iter_mut().zip(&s) {
                *value -= T::from_real(self.shift) * s;
            }
        });
    }

    /// Evaluate the shifted Fock elements of one ordered reference pair with Wick's theorem.
    /// # Arguments:
    /// - `self`: Shifted Fock operator.
    /// - `data`: Shared data holding the numerical tolerance.
    /// - `w`: Wick intermediates of the ordered reference pair.
    /// - `requests`: Output positions with their determinant pairs.
    /// - `scratch`: Scratch space for Wick's calculations.
    /// - `out`: Shifted Fock elements indexed by output position.
    /// # Returns:
    /// - `()`: Writes every requested element into `out`.
    fn wicks(
        &self,
        data: &NOCIData<'_, T>,
        w: &WicksPairView<'_, T>,
        requests: &[(usize, DetPair)],
        scratch: &mut WickScratchSpin<T>,
        out: &mut [T],
    ) {
        time_call!(crate::timers::noci::add_calculate_f_pair_wicks, {
            let elements = calculate_fs_pairs_wicks(w, data.space, requests, data.tol, scratch);
            for ((output, _), (f, s)) in requests.iter().zip(elements) {
                out[*output] = f - T::from_real(self.shift) * s;
            }
        });
    }

    /// Evaluate one shifted Fock element with the generalised Slater-Condon rules.
    /// # Arguments:
    /// - `self`: Shifted Fock operator.
    /// - `data`: Shared data required for NOCI matrix-element evaluation.
    /// - `bra`: Bra determinant.
    /// - `ket`: Ket determinant.
    /// # Returns:
    /// - `T`: Shifted Fock element of the pair.
    fn naive(
        &self,
        data: &NOCIData<'_, T>,
        bra: &DeterminantState,
        ket: &DeterminantState,
    ) -> T {
        time_call!(crate::timers::noci::add_calculate_f_pair_naive, {
            let (f, s) = calculate_fs_pair_naive(self.fock, data, bra, ket);
            f - T::from_real(self.shift) * s
        })
    }

    /// Measure the discrepancy between two shifted Fock elements.
    /// # Arguments:
    /// - `a`: First evaluation.
    /// - `b`: Second evaluation.
    /// # Returns:
    /// - `(f64, f64)`: Absolute discrepancy, as both total and largest.
    fn discrepancy(
        a: T,
        b: T,
    ) -> (f64, f64) {
        let d = (a - b).abs();
        (d, d)
    }
}

impl<T: NOCIScalar> PairOperator<T> for HamiltonianOverlap {
    type Out = (T, T);
    const RANK: usize = 2;
    const TIMER: fn(u64) = crate::timers::noci::add_calculate_hs_pair;

    /// Return the vanishing Hamiltonian and overlap pair.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `(T, T)`: Zero Hamiltonian and overlap.
    fn zero() -> (T, T) {
        (T::from_real(0.0), T::from_real(0.0))
    }

    /// Return whether the parent's Hamiltonian MO cache is Hermitian orthonormal.
    /// # Arguments:
    /// - `self`: Hamiltonian and overlap operator.
    /// - `data`: Shared data holding the parent MO caches.
    /// - `parent`: Parent orbital frame.
    /// # Returns:
    /// - `bool`: Whether same-parent elements may use the orthogonal rules.
    fn orthonormal(
        &self,
        data: &NOCIData<'_, T>,
        parent: usize,
    ) -> bool {
        data.mocache
            .is_some_and(|mocache| mocache[parent].orthogonal_slater_condon)
    }

    /// Evaluate a parent-local batch of orthogonal Hamiltonian and overlap elements.
    /// # Arguments:
    /// - `self`: Hamiltonian and overlap operator.
    /// - `data`: Shared data holding the AO data and parent MO caches.
    /// - `parent`: Shared orthonormal parent of the batch.
    /// - `occupations`: Ket alpha and beta occupations of each pair.
    /// - `connections`: Bra data measured from the ket of each pair.
    /// - `values`: Hamiltonian and overlap elements in batch order.
    /// # Returns:
    /// - `()`: Writes every `(H, S)` pair of the batch into `values`.
    fn orthogonal(
        &self,
        data: &NOCIData<'_, T>,
        parent: usize,
        occupations: &[(u128, u128)],
        connections: &[DeterminantData],
        values: &mut [(T, T)],
    ) {
        time_call!(crate::timers::noci::add_calculate_hs_pair_orthogonal, {
            // Evaluate `H` and `S` with their own batched kernels before pairing them.
            let mocache = data
                .mocache
                .expect("orthogonal Hamiltonian elements require parent MO caches");
            let hamiltonian = OrthogonalHamiltonian {
                ao: data.ao,
                cache: &mocache[parent],
            };
            let mut h = vec![T::from_real(0.0); connections.len()];
            let mut s = vec![T::from_real(0.0); connections.len()];
            xw_orthogonal_batched(&hamiltonian, occupations, connections, &mut h);
            xw_orthogonal_batched(&OrthogonalOverlap, occupations, connections, &mut s);
            for ((value, &h), &s) in values.iter_mut().zip(&h).zip(&s) {
                *value = (h, s);
            }
        });
    }

    /// Evaluate the Hamiltonian and overlap elements of one ordered reference pair with Wick's
    /// theorem.
    /// # Arguments:
    /// - `self`: Hamiltonian and overlap operator.
    /// - `data`: Shared data holding the nuclear repulsion and numerical tolerance.
    /// - `w`: Wick intermediates of the ordered reference pair.
    /// - `requests`: Output positions with their determinant pairs.
    /// - `scratch`: Scratch space for Wick's calculations.
    /// - `out`: Hamiltonian and overlap elements indexed by output position.
    /// # Returns:
    /// - `()`: Writes every requested `(H, S)` pair into `out`.
    fn wicks(
        &self,
        data: &NOCIData<'_, T>,
        w: &WicksPairView<'_, T>,
        requests: &[(usize, DetPair)],
        scratch: &mut WickScratchSpin<T>,
        out: &mut [(T, T)],
    ) {
        time_call!(crate::timers::noci::add_calculate_hs_pair_wicks, {
            // The two-spin kernel reads every request's determinants relative to their parents.
            let parents = data.space.parents();
            let states: Vec<_> = requests
                .iter()
                .map(|&(output, pair)| (output, pair.bra.data(parents), pair.ket.data(parents)))
                .collect();
            xw_hamiltonian_overlap_prepared_batched(
                w,
                &states,
                data.ao.enuc,
                scratch,
                data.tol,
                out,
            );
        });
    }

    /// Evaluate one Hamiltonian and overlap pair with the generalised Slater-Condon rules.
    /// # Arguments:
    /// - `self`: Hamiltonian and overlap operator.
    /// - `data`: Shared data required for NOCI matrix-element evaluation.
    /// - `bra`: Bra determinant.
    /// - `ket`: Ket determinant.
    /// # Returns:
    /// - `(T, T)`: Hamiltonian and overlap elements of the pair.
    fn naive(
        &self,
        data: &NOCIData<'_, T>,
        bra: &DeterminantState,
        ket: &DeterminantState,
    ) -> (T, T) {
        time_call!(crate::timers::noci::add_calculate_hs_pair_naive, {
            calculate_hs_pair_naive(data, bra, ket)
        })
    }

    /// Measure the discrepancy between two Hamiltonian and overlap pairs.
    /// # Arguments:
    /// - `a`: First evaluation.
    /// - `b`: Second evaluation.
    /// # Returns:
    /// - `(f64, f64)`: Summed and largest absolute discrepancy of `H` and `S`.
    fn discrepancy(
        a: (T, T),
        b: (T, T),
    ) -> (f64, f64) {
        let dh = (a.0 - b.0).abs();
        let ds = (a.1 - b.1).abs();
        (dh + ds, f64::max(dh, ds))
    }
}
