// build.rs

use std::env;
use std::fs;
use std::path::PathBuf;

use bincode::Options;
use serde::de::DeserializeOwned;

#[path = "src/nocc/terms.rs"]
mod terms;

use terms::{OverlapTermSet, ResidualTermSet};

/// Convert one generated term JSON file to bincode.
/// # Arguments:
/// - `input`: Path to the generated term JSON file.
/// - `output`: Name of the bincode file to write under Cargo's output directory.
/// # Returns:
/// - `()`: Writes the bincode file and registers the JSON file as a build input.
fn convert_terms<T>(
    input: &str,
    output: &str,
) where
    T: serde::Serialize + DeserializeOwned,
{
    println!("cargo:rerun-if-changed={input}");

    let text =
        fs::read_to_string(input).unwrap_or_else(|err| panic!("failed to read {input}: {err}"));

    let terms: T =
        serde_json::from_str(&text).unwrap_or_else(|err| panic!("failed to parse {input}: {err}"));

    let bytes = bincode::DefaultOptions::new()
        .with_varint_encoding()
        .serialize(&terms)
        .unwrap_or_else(|err| panic!("failed to bincode-serialize {input}: {err}"));

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is not set"));
    let output = out_dir.join(output);

    fs::write(&output, bytes)
        .unwrap_or_else(|err| panic!("failed to write {}: {err}", output.display()));
}

/// Build generated term bincode files.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Writes generated bincode term tables under Cargo's output directory.
fn main() {
    println!("cargo:rerun-if-changed=src/nocc/terms.rs");

    convert_terms::<ResidualTermSet>("src/nocc/terms/r0terms.json", "r0terms.bin");
    convert_terms::<ResidualTermSet>("src/nocc/terms/r1terms.json", "r1terms.bin");
    convert_terms::<OverlapTermSet>("src/nocc/terms/overlapterms.json", "overlapterms.bin");
}
