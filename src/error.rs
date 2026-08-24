// error.rs

// Standard library imports.
use std::path::PathBuf;
use std::process::ExitStatus;

/// Crate-wide result type.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors produced while parsing inputs, preparing data, or running calculations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Filesystem operation failed.
    #[error("{context} at {}: {source}", path.display())]
    Io {
        /// Operation that failed.
        context: &'static str,
        /// Path being accessed.
        path: PathBuf,
        /// Underlying IO error.
        #[source]
        source: std::io::Error,
    },

    /// Lua input parsing or execution failed.
    #[error("Lua input error: {0}")]
    Lua(#[from] rlua::Error),

    /// HDF5 read/write failed.
    #[error("HDF5 error: {0}")]
    Hdf5(#[from] hdf5::Error),

    /// PySCF process exited unsuccessfully.
    #[error("PySCF failed with status {0}")]
    PyscfFailed(ExitStatus),
}

impl Error {
    /// Add filesystem context to an IO error.
    /// # Arguments:
    /// - `context`: Operation that failed.
    /// - `path`: Path being accessed.
    /// - `source`: Underlying IO error.
    /// # Returns:
    /// - `Self`: Contextual IO error.
    pub fn io(
        context: &'static str,
        path: impl Into<PathBuf>,
        source: std::io::Error,
    ) -> Self {
        Self::Io {
            context,
            path: path.into(),
            source,
        }
    }
}
