// noci/cumulants/common.rs

use itertools::Itertools;

use super::helpers::{decode_index, permutation_cycles, permutation_sign, set_partitions};
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

/// Disconnected product used in the cumulant recursion.
pub(super) struct Product {
    /// Spin-free permutation coefficient.
    coeff: f64,
    /// Cumulant factors in the product.
    blocks: Vec<Block>,
}

/// One cumulant factor in a disconnected product.
struct Block {
    /// Upper slot positions included in this factor.
    upper: Vec<usize>,
    /// Lower slot positions included in this factor.
    lower: Vec<usize>,
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

/// Build one active-space cumulant rank from the corresponding RDM.
/// # Arguments:
/// - `rank`: Cumulant rank.
/// - `n`: Number of active orbitals.
/// - `moment`: Function returning the matching RDM element.
/// - `lower`: Lower-rank cumulants already generated.
/// - `products`: Disconnected products for this rank.
/// # Returns:
/// - `CumulantTensor<T>`: Active-space cumulant tensor.
pub(super) fn build_cumulant<T, F>(
    rank: usize,
    n: usize,
    moment: F,
    lower: &[&CumulantTensor<T>],
    products: &[Product],
) -> CumulantTensor<T>
where
    T: NOCIScalar,
    F: Fn(&[usize], &[usize]) -> T,
{
    let mut lambda = CumulantTensor::zeros(rank, n);
    let total = n.pow((2 * rank) as u32);

    for i in 0..total {
        let indices = decode_index(i, 2 * rank, n);
        let upper = &indices[..rank];
        let lower_indices = &indices[rank..];

        let mut value = moment(upper, lower_indices);

        for product in products {
            let mut term = <T as From<f64>>::from(product.coeff);

            for block in product.blocks.iter() {
                let block_rank = block.upper.len();
                let block_upper = block.upper.iter().map(|&i| upper[i]).collect::<Vec<_>>();
                let block_lower = block
                    .lower
                    .iter()
                    .map(|&i| lower_indices[i])
                    .collect::<Vec<_>>();

                term *= lower[block_rank - 1].get(&block_upper, &block_lower);
            }

            value -= term;
        }

        lambda.set(upper, lower_indices, value);
    }

    lambda
}

/// Generate all disconnected spin-free products for one cumulant rank.
/// # Arguments:
/// - `rank`: Cumulant rank.
/// # Returns:
/// - `Vec<Product>`: Disconnected products.
pub(super) fn disconnected(rank: usize) -> Vec<Product> {
    let partitions = set_partitions(rank);
    let permutations = (0..rank).permutations(rank).collect::<Vec<_>>();
    let mut products = Vec::new();

    for partition in partitions {
        if partition.len() == 1 {
            continue;
        }

        for permutation in permutations.iter() {
            let coeff = permutation_sign(permutation) as f64
                * 2.0_f64.powi(permutation_cycles(permutation) as i32 - rank as i32);

            let blocks = partition
                .iter()
                .map(|block| Block {
                    upper: block.clone(),
                    lower: block.iter().map(|&i| permutation[i]).collect(),
                })
                .collect();

            products.push(Product { coeff, blocks });
        }
    }

    products
}
