// External crate imports.
use noci_rs::Result;
use noci_rs::driver::{load_config, run};

/// Parse the input, initialise the runtime, and execute the requested calculation.
/// # Returns
/// - `Result<()>`: Success after all geometries finish, or the propagated driver error.
fn main() -> Result<()> {
    let config = load_config()?;
    run(config)
}
