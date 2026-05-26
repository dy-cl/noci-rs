// noci/cumulants/cumulants.rs

use itertools::Itertools;

use super::helpers::{decode_index, permutation_cycles, permutation_sign, set_partitions};
use crate::noci::rdm::{RDM1, RDM2, RDM3, RDM4};
use crate::noci::types::NOCIScalar;

/// Spin-free cumulants through rank four.
pub(crate) struct Cumulants<T: NOCIScalar> {
    /// Active-space one-cumulant Λ[p, q].
    pub lambda1: CumulantTensor<T>,
    /// Active-space two-cumulant Λ[p, q, r, s].
    pub lambda2: CumulantTensor<T>,
    /// Active-space three-cumulant Λ[p, q, r, s, t, u].
    pub lambda3: CumulantTensor<T>,
    /// Active-space four-cumulant Λ[p, q, r, s, t, u, v, w].
    pub lambda4: CumulantTensor<T>,
}

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
struct Product {
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
    fn zeros(
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
    fn set(
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

/// Build spin-free cumulants from one- to four-body spin-free RDMs.
/// # Arguments:
/// - `gamma1`: Full-space spin-free one-body RDM.
/// - `gamma2`: Full-space spin-free two-body RDM.
/// - `gamma3`: Active-space spin-free three-body RDM.
/// - `gamma4`: Active-space spin-free four-body RDM.
/// - `active`: Active orbital indices used to build `gamma3` and `gamma4`.
/// # Returns:
/// - `Cumulants<T>`: Active-space spin-free cumulants through rank four.
pub(crate) fn cumulants<T: NOCIScalar>(
    gamma1: &RDM1<T>,
    gamma2: &RDM2<T>,
    gamma3: &RDM3<T>,
    gamma4: &RDM4<T>,
    active: &[usize],
) -> Cumulants<T> {
    let n = active.len();

    let lambda1 = build_cumulant(
        1,
        n,
        |upper, lower| {
            let p = active[upper[0]];
            let q = active[lower[0]];

            gamma1.data[p * gamma1.n + q]
        },
        &[],
        &[],
    );

    let products2 = disconnected(2);
    let lambda2 = build_cumulant(
        2,
        n,
        |upper, lower| {
            let p = active[upper[0]];
            let q = active[upper[1]];
            let r = active[lower[0]];
            let s = active[lower[1]];

            gamma2.data[(((p * gamma2.n + q) * gamma2.n + r) * gamma2.n) + s]
        },
        &[&lambda1],
        &products2,
    );

    let products3 = disconnected(3);
    let lambda3 = build_cumulant(
        3,
        n,
        |upper, lower| {
            let i = upper
                .iter()
                .chain(lower.iter())
                .fold(0, |acc, &p| acc * gamma3.n + p);

            gamma3.data[i]
        },
        &[&lambda1, &lambda2],
        &products3,
    );

    let products4 = disconnected(4);
    let lambda4 = build_cumulant(
        4,
        n,
        |upper, lower| {
            let i = upper
                .iter()
                .chain(lower.iter())
                .fold(0, |acc, &p| acc * gamma4.n + p);

            gamma4.data[i]
        },
        &[&lambda1, &lambda2, &lambda3],
        &products4,
    );

    Cumulants {
        lambda1,
        lambda2,
        lambda3,
        lambda4,
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
fn build_cumulant<T, F>(
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
fn disconnected(rank: usize) -> Vec<Product> {
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
