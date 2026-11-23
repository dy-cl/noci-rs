// input/replace.rs

// Parent/sibling imports.
use super::state::SCFExcitation;

/// Options for replacing duplicate or unconverged SCF states by excited SCF states.
#[derive(Default)]
pub struct ReplaceOptions {
    /// Label of the state whose excitations replace missing states. When absent, the first
    /// state in the NOCI basis is used.
    pub parent: Option<String>,
    /// Excitations of the parent to try first, in order, before the generated ones.
    pub excits: Vec<SCFExcitation>,
}
