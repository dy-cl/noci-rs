// overlap.rs

use crate::ir::{Expr, Product};

/// Build one metric block operator product.
/// # Arguments:
/// - `name`: Block name.
/// # Returns:
/// - `Product`: GNO product for this metric block.
pub fn block(name: &str) -> Product {
    crate::specs::product(name)
}

/// Generate one canonical metric block expression.
/// # Arguments:
/// - `name`: Metric block name.
/// # Returns:
/// - `Expr`: Canonical metric expression.
pub fn expr(name: &str) -> Expr {
    crate::canonical::canon(crate::wick::eval(&block(name)))
}

/// Visit all metric chunks.
/// # Arguments:
/// - `emit`: Callback receiving `(chunk_key, expression)`.
/// # Returns:
/// - `()`: Calls `emit` once per metric block.
pub fn chunks(mut emit: impl FnMut(String, Expr)) {
    let prog =
        crate::progress::Prog::new("overlap::chunks metric blocks", crate::specs::BLOCKS.len());

    for b in crate::specs::BLOCKS {
        emit(b.name.to_string(), expr(b.name));
        prog.tick();
    }
}
