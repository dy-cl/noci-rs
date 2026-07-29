# Style Guide

## Rust Module Headers

Use this order, omitting empty sections:

1. File banner, for example `// noci/mod.rs`.
2. Module documentation, using `//!`.
3. Public submodules, `pub mod`.
4. Restricted submodules, `pub(crate) mod` or `pub(in path) mod`.
5. Private submodules, `mod`.
6. Public type re-exports: structs, enums, traits and type aliases.
7. Public function re-exports.
8. Restricted type re-exports: `pub(crate) use`, `pub(super) use` or `pub(in path) use`.
9. Restricted function re-exports.
10. Private imports needed by the `mod.rs` implementation itself.

Sort entries alphabetically within each section.

Split mixed re-exports when practical:

```rust
// Public type re-exports.
pub use nociqmc::{Coefficients, Projectors};

// Public function re-exports.
pub use nociqmc::{projected_energy, propagate};
```
