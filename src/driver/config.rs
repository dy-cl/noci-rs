// driver/config.rs

// Crate-root imports.
use crate::Result;
use crate::input::{Input, load_input};

/// Load user configuration from the command line.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Result<Input>`: Parsed user input specifications.
/// # Errors
/// - Returns an error if the input file cannot be read or its Lua code fails to execute.
pub fn load_config() -> Result<Input> {
    let input_path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("Usage: cargo run <input.lua>");
            std::process::exit(1);
        }
    };
    load_input(&input_path)
}
