// input/excit.rs

pub struct ExcitationOptions {
    /// Excitation orders to generate.
    pub orders: Vec<usize>,
    /// Whether to generate all possible excitations.
    pub all: bool,
}

impl Default for ExcitationOptions {
    /// Return default NOCI excitation options.
    /// # Returns:
    /// - `Self`: Excitation options with singles and doubles enabled.
    fn default() -> Self {
        Self {
            orders: [1, 2].to_vec(),
            all: false,
        }
    }
}
