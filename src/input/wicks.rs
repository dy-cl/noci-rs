// input/wicks.rs

pub enum WicksStorage {
    RAM,
    Disk,
}

pub struct WicksOptions {
    /// Whether to compare Wick and Slater-Condon matrix elements.
    pub compare: bool,
    /// Whether to use Wick intermediates.
    pub enabled: bool,
    /// Wick intermediate storage backend.
    pub storage: WicksStorage,
    /// Optional disk cache directory.
    pub cachedir: Option<String>,
}

impl Default for WicksOptions {
    /// Return default Wick's theorem options.
    /// # Returns:
    /// - `Self`: Wick's options with comparison disabled.
    fn default() -> Self {
        Self {
            compare: false,
            enabled: true,
            storage: WicksStorage::RAM,
            cachedir: Some(".".to_string()),
        }
    }
}
