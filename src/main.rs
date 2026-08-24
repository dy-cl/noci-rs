// External crate imports.
use noci_rs::Result;
use noci_rs::driver::{load_config, run};

fn main() -> Result<()> {
    let config = load_config()?;
    run(config)
}
