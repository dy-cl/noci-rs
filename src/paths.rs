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
    /// HDF5 file containing generated AO integrals.
    pub integral_file: PathBuf,
    /// Directory used by disk-backed Wick/factor caches unless explicitly overridden.
    pub wicks_cache_dir: PathBuf,
}

impl RunPaths {
    /// Build run paths from input options.
    /// # Arguments:
    /// - `input`: Parsed user input.
    /// # Returns:
    /// - `Self`: Effective output, integral, and cache paths.
    pub fn from_input(input: &Input) -> Self {
        let output_dir = PathBuf::from(&input.write.write_dir);
        let integral_file = output_dir.join("data.h5");
        let wicks_cache_dir = input
            .wicks
            .cachedir
            .as_deref()
            .map(PathBuf::from)
            .unwrap_or_else(|| output_dir.join("wicks"));

        Self {
            output_dir,
            integral_file,
            wicks_cache_dir,
        }
    }
}
