// input/mol.rs

pub struct MolOptions {
    /// Atomic orbital basis name.
    pub basis: String,
    /// Geometry length unit.
    pub unit: String,
    /// Bond distances or scan coordinates.
    pub r_list: Vec<f64>,
    /// Atomic geometries for each scan point.
    pub geoms: Vec<Vec<String>>,
}

impl Default for MolOptions {
    /// Return default molecular options.
    /// # Returns:
    /// - `Self`: Molecular options with placeholder empty geometry data.
    fn default() -> Self {
        Self {
            basis: String::new(),
            unit: "Ang".to_string(),
            r_list: Vec::new(),
            geoms: Vec::new(),
        }
    }
}
