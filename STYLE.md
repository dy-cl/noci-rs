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

## Rust Imports

For implementation files, keep imports at the top of the file in this order, omitting empty
sections. Add a short comment above each non-empty import section, matching the labelled section
style used by `mod.rs` re-exports.

1. File-level attributes, for example `#![allow(...)]`.
2. File banner, for example `// noci/cache.rs`.
3. Standard library imports: `std`, `core` and `alloc`.
4. External crate imports.
5. Crate-root imports: `crate::...`.
6. Parent or sibling imports: `self::...` then `super::...`.

Separate each section with one blank line. Sort imports alphabetically within each section.

Keep `#[cfg(...)]` attributes directly attached to the import they guard. Prefer grouped imports
from the same path when this stays readable:

```rust
// noci/cache.rs

// Standard library imports.
use std::sync::Arc;

// External crate imports.
use ndarray::{Array2, Array4};
use num_complex::Complex64;

// Crate-root imports.
use crate::maths::{adjoint, real2_as};
use crate::{AoData, DetState};

// Parent/sibling imports.
use super::types::{FockMOCache, MOCache, NOCIScalar};
```

In `mod.rs` files, keep the module-header order above. Private imports used only by the
`mod.rs` implementation come after submodules and re-exports.

## Rust Function Documentation

Every function should have a `///` documentation block. Function docs should look like the
following example from `src/maths/linalg.rs`:

```rust
/// Loewdin symmetric orthogonalizer, computes `X = S^{-1/2}`.
/// If `project` is true, returns the rectangular orthogonalizer `X = U_+ Lambda_+^{-1/2}`.
/// If `project` is false, returns the square orthogonalizer `X = U Lambda^{-1/2} U^\dagger`.
/// # Arguments:
/// - `s`: Hermitian matrix, uses only the lower triangle.
/// - `project`: Whether or not to project to non-zero positive subspace of `s`.
/// - `tol`: Tolerance for whether a number is considered zero.
/// # Returns
/// - `Array2<T>`: Orthogonalizer.
pub fn loewdin_x<T: StateScalar>(
    s: &Array2<T>,
    project: bool,
    tol: f64,
) -> Array2<T> {
    // ...
}
```

This is a good function doc because it:

1. Starts with the mathematical operation being performed.
2. Gives the governing equation, `X = S^{-1/2}`, before implementation details.
3. Uses LaTeX-style notation for branch-specific formulas, such as
   `X = U_+ Lambda_+^{-1/2}` and `X = U Lambda^{-1/2} U^\dagger`.
4. Documents numerical conventions, such as using only the lower triangle of a Hermitian
   matrix and applying a tolerance to identify the positive subspace.
5. Lists each argument and the return value using the same `# Arguments` and `# Returns`
   structure used throughout the crate.

Use equations where they make the scientific meaning clearer. In function docs, surround
LaTeX-style notation with backticks so Rustdoc treats it as code and preserves underscores,
backslashes and spacing. Prefer notation such as `H C = S C E`, `C^\dagger S C = I`,
`\sum_i x_i y_i`, or tensor index order like `[p, q, r, s]` when it is more precise than prose.

Use optional sections only when applicable:

1. `# Panics` for intentional panics or indexing assumptions that must hold.
2. `# Errors` for functions returning `Result`.
3. `# Safety` for unsafe functions.
