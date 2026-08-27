// paths.rs

// Standard library imports.
use std::path::PathBuf;

// Crate-root imports.
use crate::input::Input;

/// Effective filesystem paths for one calculation run.
#[derive(Clone, Debug)]
pub struct RunPaths {
    /// Directory used for calculation outputs.
    pub output_dir: PathBuf,
    /// Directory used by disk-backed Wick/factor caches unless explicitly overridden.
    pub wicks_cache_dir: PathBuf,
}

impl RunPaths {
    /// Build run paths from input options.
    /// # Arguments:
    /// - `input`: Parsed user input.
    /// # Returns:
    /// - `Self`: Effective output and cache paths.
    pub fn from_input(input: &Input) -> Self {
        let output_dir = PathBuf::from(&input.write.write_dir);
        let wicks_cache_dir = input
            .wicks
            .cachedir
            .as_deref()
            .map(PathBuf::from)
            .unwrap_or_else(|| output_dir.join("wicks"));

        Self {
            output_dir,
            wicks_cache_dir,
        }
    }
}
