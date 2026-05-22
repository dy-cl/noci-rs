// driver/config.rs

use crate::input::{Input, load_input};

/// Load user configuration from the command line.
/// # Arguments:
/// - None.
/// # Returns:
/// - `Input`: Parsed user input specifications.
pub fn load_config() -> Input {
    let input_path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("Usage: cargo run <input.lua>");
            std::process::exit(1);
        }
    };
    load_input(&input_path)
}
