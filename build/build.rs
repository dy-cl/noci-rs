// build.rs

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use bincode::Options;
use serde::Serialize;

/// Return whether cached generated terms should be forcibly regenerated.
/// # Arguments:
/// - None.
/// # Returns:
/// - `bool`: True when `WICK_FORCE_REGENERATE=1`.
fn force() -> bool {
    env::var("WICK_FORCE_REGENERATE").map(|x| x == "1").unwrap_or(false)
}

/// Serialize one generated table.
/// # Arguments:
/// - `x`: Generated table.
/// # Returns:
/// - `Vec<u8>`: Bincode payload.
fn bytes<T: Serialize>(x: &T) -> Vec<u8> {
    bincode::DefaultOptions::new()
        .with_varint_encoding()
        .serialize(x)
        .expect("failed to serialize generated terms")
}

/// Copy one cached file into `OUT_DIR`.
/// # Arguments:
/// - `src`: Cached source path.
/// - `dst`: Output path.
/// # Returns:
/// - `()`: Copies the file.
fn copy(src: &Path, dst: &Path) {
    fs::copy(src, dst).unwrap_or_else(|e| panic!("failed to copy {} to {}: {e}", src.display(), dst.display()));
}

/// Ensure one cached generated term table exists and is copied to `OUT_DIR`.
/// # Arguments:
/// - `cache`: Cached artifact path.
/// - `out`: Output artifact path.
/// - `make`: Generator callback.
/// # Returns:
/// - `()`: Writes or copies generated terms.
fn ensure<T: Serialize>(cache: &Path, out: &Path, make: impl FnOnce() -> T) {
    if cache.exists() && !force() {
        copy(cache, out);
        return;
    }

    let data = make();
    let data = bytes(&data);

    fs::write(cache, &data).unwrap_or_else(|e| panic!("failed to write {}: {e}", cache.display()));
    fs::write(out, data).unwrap_or_else(|e| panic!("failed to write {}: {e}", out.display()));
}

/// Generate build-time NOCC data.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Writes bincode files under `OUT_DIR`.
fn main() {
    println!("cargo:rerun-if-changed=build/wick");
    println!("cargo:rerun-if-env-changed=WICK_FORCE_REGENERATE");
    println!("cargo:rerun-if-env-changed=WICK_PROGRESS");
    println!("cargo:rerun-if-env-changed=WICK_PROGRESS_STEP");
    println!("cargo:rerun-if-env-changed=WICK_H_BATCH");
    println!("cargo:rerun-if-env-changed=WICK_SPIN_BATCH");
    println!("cargo:rerun-if-env-changed=WICK_SPIN_PAR");
    println!("cargo:rerun-if-env-changed=WICK_STREAM_QUEUE");
    println!("cargo:rerun-if-env-changed=WICK_ACC_FLUSH");
    println!("cargo:rerun-if-env-changed=WICK_SPIN_SPLIT_CHUNKS");

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is not set"));
    let cache = manifest.join("build/generated");
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is not set"));

    fs::create_dir_all(&cache).unwrap_or_else(|e| panic!("failed to create {}: {e}", cache.display()));

    ensure(&cache.join("overlapterms.bin"), &out.join("overlapterms.bin"), wick_build::overlap_terms);
    ensure(&cache.join("r0terms.bin"), &out.join("r0terms.bin"), wick_build::r0_terms);
    ensure(&cache.join("r1terms.bin"), &out.join("r1terms.bin"), wick_build::r1_terms);
    ensure(&cache.join("r2terms.bin"), &out.join("r2terms.bin"), wick_build::r2_terms);
}
