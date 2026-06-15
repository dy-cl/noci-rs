// build.rs

use std::env;
use std::fs;
use std::path::PathBuf;
use bincode::Options;

/// Generate build-time NOCC data.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Writes bincode files under `OUT_DIR`.
fn main() {
    println!("cargo:rerun-if-changed=build/wick");

    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is not set"));

    let terms = wick_build::overlap_terms();
    let bytes = bincode::DefaultOptions::new()
        .with_varint_encoding()
        .serialize(&terms)
        .expect("failed to serialize metric terms");

    fs::write(out.join("overlapterms.bin"), bytes).expect("failed to write metric terms");
}
