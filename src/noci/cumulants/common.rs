// noci/cumulants/common.rs

use crate::noci::types::NOCIScalar;

/// Active-space spin-free cumulant tensor stored in upper-then-lower index order.
pub(crate) struct CumulantTensor<T: NOCIScalar> {
    /// Cumulant rank.
    pub _rank: usize,
    /// Number of active orbitals.
    pub n: usize,
    /// Flat tensor storage in upper-then-lower index order.
    pub data: Vec<T>,
}

impl<T: NOCIScalar> CumulantTensor<T> {
    /// Allocate a zero-filled active-space cumulant tensor.
    /// # Arguments:
    /// - `n`: Number of active orbitals.
    /// # Returns:
    /// - `CumulantTensor<T>`: Zero-filled cumulant tensor.
    pub(super) fn zeros(
        rank: usize,
        n: usize,
    ) -> Self {
        Self {
            _rank: rank,
            n,
            data: vec![<T as From<f64>>::from(0.0); n.pow((2 * rank) as u32)],
        }
    }

    /// Return a cumulant element.
    /// # Arguments:
    /// - `upper`: Upper active-space indices.
    /// - `lower`: Lower active-space indices.
    /// # Returns:
    /// - `T`: Tensor element.
    pub(crate) fn get(
        &self,
        upper: &[usize],
        lower: &[usize],
    ) -> T {
        self.data[self.index(upper, lower)]
    }

    /// Set a cumulant element.
    /// # Arguments:
    /// - `upper`: Upper active-space indices.
    /// - `lower`: Lower active-space indices.
    /// - `value`: Tensor element.
    pub(super) fn set(
        &mut self,
        upper: &[usize],
        lower: &[usize],
        value: T,
    ) {
        let i = self.index(upper, lower);
        self.data[i] = value;
    }

    /// Return the flat index for an upper-then-lower tensor element.
    /// # Arguments:
    /// - `upper`: Upper active-space indices.
    /// - `lower`: Lower active-space indices.
    /// # Returns:
    /// - `usize`: Flat tensor index.
    fn index(
        &self,
        upper: &[usize],
        lower: &[usize],
    ) -> usize {
        let mut i = 0;
        for &p in upper.iter().chain(lower.iter()) {
            i = i * self.n + p;
        }

        i
    }
}
