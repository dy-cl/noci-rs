// overlap.rs

use crate::ir::Product;

/// Build one metric block operator product.
/// # Arguments:
/// - `name`: Block name.
/// # Returns:
/// - `Product`: GNO product for this metric block.
pub fn block(name: &str) -> Product {
    crate::specs::product(name)
}
