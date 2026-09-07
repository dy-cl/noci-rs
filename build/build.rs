// build/build.rs

// Standard library imports.
use std::env;
use std::fmt::Write;
use std::fs;
use std::path::PathBuf;

#[cfg(feature = "nocc")]
mod nocc {
    use std::env;
    use std::fmt::Write;
    use std::fs;
    use std::io::BufWriter;
    use std::path::{Path, PathBuf};

    use bincode::Options;
    use serde::Serialize;

    const CLASSES: &[&str] = &[
        "CToA", "AToA", "AToV", "CAToAV", "CAToVA", "CAToVV", "CCToAV", "CCToAA", "CAToAA",
        "AAToAV", "AAToVV", "AAToAA",
    ];

    /// Return whether cached generated terms should be forcibly regenerated.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `bool`: True when `WICK_FORCE_REGENERATE=1`.
    fn force() -> bool {
        env::var("WICK_FORCE_REGENERATE")
            .map(|x| x == "1")
            .unwrap_or(false)
    }

    /// Print a visible Cargo build-script status line.
    /// # Arguments:
    /// - `msg`: Message to print.
    /// # Returns:
    /// - `()`: Emits a Cargo warning line.
    fn status(msg: impl std::fmt::Display) {
        println!("cargo:warning={msg}");
    }

    /// Write one generated table.
    /// # Arguments:
    /// - `x`: Generated table.
    /// - `path`: Output path.
    /// # Returns:
    /// - `()`: Writes bincode payload.
    fn write_table<T: Serialize>(
        x: &T,
        path: &Path,
    ) {
        let file = fs::File::create(path)
            .unwrap_or_else(|e| panic!("failed to create {}: {e}", path.display()));
        let mut out = BufWriter::new(file);

        bincode::DefaultOptions::new()
            .with_varint_encoding()
            .serialize_into(&mut out, x)
            .unwrap_or_else(|e| {
                panic!(
                    "failed to serialize generated terms to {}: {e}",
                    path.display()
                )
            });
    }

    /// Copy one cached file into `OUT_DIR`.
    /// # Arguments:
    /// - `src`: Cached source path.
    /// - `dst`: Output path.
    /// # Returns:
    /// - `()`: Copies the file.
    fn copy(
        src: &Path,
        dst: &Path,
    ) {
        fs::copy(src, dst).unwrap_or_else(|e| {
            panic!("failed to copy {} to {}: {e}", src.display(), dst.display())
        });
    }

    /// Ensure one cached generated term table exists and is copied to `OUT_DIR`.
    /// # Arguments:
    /// - `name`: Human-readable table name.
    /// - `cache`: Cached artifact path.
    /// - `out`: Output artifact path.
    /// - `make`: Generator callback.
    /// # Returns:
    /// - `()`: Writes or copies generated terms.
    fn ensure<T: Serialize>(
        name: &str,
        cache: &Path,
        out: &Path,
        make: impl FnOnce() -> T,
    ) {
        if cache.exists() && !force() {
            status(format!("using cached generated terms: {name}"));
            copy(cache, out);
            return;
        }

        if cache.exists() {
            status(format!(
                "regenerating terms because WICK_FORCE_REGENERATE=1: {name}"
            ));
        } else {
            status(format!("generated terms missing, generating now: {name}"));
        }

        let data = make();

        status(format!("serializing generated terms: {name}"));

        write_table(&data, cache);
        copy(cache, out);

        status(format!(
            "ready generated terms: {name}, bytes: {}",
            fs::metadata(cache).map(|x| x.len()).unwrap_or(0)
        ));
    }

    /// Ensure one cached residual class exists and is copied to `OUT_DIR`.
    /// # Arguments:
    /// - `order`: Residual order.
    /// - `class`: Excitation-class name.
    /// - `cache`: Generated cache root.
    /// - `out`: Output root.
    /// # Returns:
    /// - `()`: Writes or copies one class table.
    fn ensure_class(
        order: u8,
        class: &str,
        cache: &Path,
        out: &Path,
    ) {
        let dir = format!("r{order}");
        let cache_dir = cache.join(&dir);
        let out_dir = out.join(&dir);
        let cache_file = cache_dir.join(format!("{class}.bin"));
        let out_file = out_dir.join(format!("{class}.bin"));

        fs::create_dir_all(&cache_dir)
            .unwrap_or_else(|e| panic!("failed to create {}: {e}", cache_dir.display()));
        fs::create_dir_all(&out_dir)
            .unwrap_or_else(|e| panic!("failed to create {}: {e}", out_dir.display()));

        ensure(
            &format!("R{order}({class})"),
            &cache_file,
            &out_file,
            || wick_build::encode::residual_class(order, class),
        );
    }

    /// Append one residual loader function to generated source.
    /// # Arguments:
    /// - `src`: Source buffer.
    /// - `fn_name`: Loader function name.
    /// - `static_name`: Static `OnceLock` name.
    /// - `order`: Residual order.
    /// # Returns:
    /// - `()`: Appends Rust source.
    fn write_residual_loader(
        src: &mut String,
        fn_name: &str,
        static_name: &str,
        order: u8,
    ) {
        let _ = writeln!(
            src,
            "pub(crate) fn {fn_name}() -> &'static ResidualTermSet {{"
        );
        let _ = writeln!(
            src,
            "    {static_name}.get_or_init(|| residual_terms({order}, &["
        );

        for class in CLASSES {
            let _ = writeln!(
                src,
                "        (\"{class}\", include_bytes!(concat!(env!(\"OUT_DIR\"), \"/r{order}/{class}.bin\")) as &[u8]),"
            );
        }

        let _ = writeln!(src, "    ]))");
        let _ = writeln!(src, "}}");
        let _ = writeln!(src);
    }

    /// Build generated NOCC term-loader source.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `String`: Rust source.
    fn generated_loader() -> String {
        let mut src = String::new();

        src.push_str("// generated by build.rs\n\n");

        write_residual_loader(&mut src, "r0_terms", "R0_TERMS", 0);
        write_residual_loader(&mut src, "r1_terms", "R1_TERMS", 1);
        write_residual_loader(&mut src, "r2_terms", "R2_TERMS", 2);

        src.push_str("pub(crate) fn overlap_terms() -> &'static OverlapTermSet {\n");
        src.push_str("    OVERLAP_TERMS.get_or_init(|| decode_overlap(include_bytes!(concat!(env!(\"OUT_DIR\"), \"/overlapterms.bin\"))))\n");
        src.push_str("}\n");

        src
    }

    /// Generate build-time NOCC data.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `()`: Writes bincode files under `OUT_DIR`.
    pub(super) fn run() {
        println!("cargo:rerun-if-changed=build/wick");
        println!("cargo:rerun-if-changed=build/generated");
        println!("cargo:rerun-if-env-changed=WICK_FORCE_REGENERATE");
        println!("cargo:rerun-if-env-changed=WICK_PROGRESS");
        println!("cargo:rerun-if-env-changed=WICK_PROGRESS_STEP");
        println!("cargo:rerun-if-env-changed=WICK_H_BATCH");
        println!("cargo:rerun-if-env-changed=WICK_SPIN_BATCH");
        println!("cargo:rerun-if-env-changed=WICK_SPIN_PAR");
        println!("cargo:rerun-if-env-changed=WICK_STREAM_QUEUE");
        println!("cargo:rerun-if-env-changed=WICK_ACC_FLUSH");
        println!("cargo:rerun-if-env-changed=WICK_SPIN_SPLIT_CHUNKS");

        let manifest =
            PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is not set"));
        let cache = manifest.join("build/generated");
        let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is not set"));

        fs::create_dir_all(&cache)
            .unwrap_or_else(|e| panic!("failed to create {}: {e}", cache.display()));

        ensure(
            "overlap",
            &cache.join("overlapterms.bin"),
            &out.join("overlapterms.bin"),
            wick_build::encode::overlap_terms,
        );

        for order in 0..=2 {
            for class in CLASSES {
                ensure_class(order, class, &cache, &out);
            }
        }

        fs::write(out.join("nocc_terms.rs"), generated_loader())
            .unwrap_or_else(|e| panic!("failed to write generated NOCC loader source: {e}"));
    }
}

/// Read the maximum determinant excitation rank compiled into fixed-rank Wick kernels.
/// # Arguments:
/// - None.
/// # Returns:
/// - `usize`: Build-time `MAXEXCIT`, defaulting to four when the environment variable is absent.
/// # Panics
/// - Panics when `MAXEXCIT` is present but is not valid UTF-8 or cannot be parsed as `usize`.
fn maxexcit() -> usize {
    match env::var("MAXEXCIT") {
        Ok(value) => value
            .parse::<usize>()
            .unwrap_or_else(|e| panic!("invalid MAXEXCIT={value}: {e}")),
        Err(env::VarError::NotPresent) => 4,
        Err(env::VarError::NotUnicode(value)) => {
            panic!("MAXEXCIT is not valid UTF-8: {value:?}")
        }
    }
}

/// Generate every nonzero same-spin fixed-rank excitation pair compiled by `MAXEXCIT`.
/// # Arguments:
/// - `maxexcit`: Maximum excitation rank of either determinant in one spin sector.
/// # Returns:
/// - `Vec<(usize, usize)>`: Ordered `(RX,RW)` pairs.
fn pair_ranks(maxexcit: usize) -> Vec<(usize, usize)> {
    let mut ranks = Vec::new();

    for l in 1..=2 * maxexcit {
        for rx in 0..=maxexcit {
            if rx > l {
                continue;
            }

            let rw = l - rx;

            if rw <= maxexcit {
                ranks.push((rx, rw));
            }
        }
    }

    ranks
}

/// Generate every two-spin Hamiltonian excitation-rank tuple compiled by `MAXEXCIT`.
/// Each bra and ket determinant independently has total alpha-plus-beta excitation rank no larger
/// than `maxexcit`.
/// # Arguments:
/// - `maxexcit`: Maximum total determinant excitation rank.
/// # Returns:
/// - `Vec<(usize, usize, usize, usize)>`: Ordered `(RXA,RWA,RXB,RWB)` tuples.
fn hamiltonian_ranks(maxexcit: usize) -> Vec<(usize, usize, usize, usize)> {
    let mut ranks = Vec::new();

    for rxa in 0..=maxexcit {
        for rwa in 0..=maxexcit {
            for rxb in 0..=maxexcit {
                for rwb in 0..=maxexcit {
                    if rxa + rxb <= maxexcit && rwa + rwb <= maxexcit {
                        ranks.push((rxa, rwa, rxb, rwb));
                    }
                }
            }
        }
    }

    ranks
}

/// Generate crate-wide build-time Wick configuration constants.
/// # Arguments:
/// - `maxexcit`: Maximum compiled determinant excitation rank.
/// # Returns:
/// - `String`: Rust source written to generated `config.rs`.
fn config_source(maxexcit: usize) -> String {
    format!(
        concat!(
            "// generated by build/build.rs\n",
            "pub(crate) const MAXEXCIT: usize = {};\n",
        ),
        maxexcit,
    )
}

/// Append one generated same-spin rank-dispatch macro.
/// # Arguments:
/// - `src`: Generated Rust source buffer.
/// - `name`: Dispatch macro name.
/// - `ranks`: Supported `(RX,RW)` rank pairs.
/// # Returns:
/// - `()`: Appends one macro definition to `src`.
fn write_pair_macro(
    src: &mut String,
    name: &str,
    ranks: &[(usize, usize)],
) {
    let _ = writeln!(src, "macro_rules! {name} {{");
    src.push_str(
        r#"    (
        $ranks:expr,
        |$rx:ident, $rw:ident, $l:ident, $d:ident| $kernel:expr,
        $fallback:expr $(,)?
    ) => {{
        dispatch_pair_ranks!(
            @match $ranks,
            |$rx, $rw, $l, $d| $kernel,
            $fallback;
"#,
    );

    for &(rx, rw) in ranks {
        let _ = writeln!(src, "            ({rx}, {rw}),");
    }

    src.push_str(
        r#"        )
    }};
}

"#,
    );
}

/// Append one generated two-spin Hamiltonian rank-dispatch macro.
/// # Arguments:
/// - `src`: Generated Rust source buffer.
/// - `name`: Dispatch macro name.
/// - `ranks`: Supported `(RXA,RWA,RXB,RWB)` tuples.
/// # Returns:
/// - `()`: Appends one macro definition to `src`.
fn write_hamiltonian_macro(
    src: &mut String,
    name: &str,
    ranks: &[(usize, usize, usize, usize)],
) {
    let _ = writeln!(src, "macro_rules! {name} {{");
    src.push_str(
        r#"    (
        $ranks:expr,
        |
            $rxa:ident, $rwa:ident, $la:ident, $da:ident, $ma:ident, $mda:ident,
            $rxb:ident, $rwb:ident, $lb:ident, $db:ident, $mb:ident, $mdb:ident
        | $kernel:expr,
        $fallback:expr $(,)?
    ) => {{
        dispatch_hamiltonian_ranks_inner!(
            @match $ranks,
            |
                $rxa, $rwa, $la, $da, $ma, $mda,
                $rxb, $rwb, $lb, $db, $mb, $mdb
            | $kernel,
            $fallback;
"#,
    );

    for &(rxa, rwa, rxb, rwb) in ranks {
        let _ = writeln!(src, "            ({rxa}, {rwa}, {rxb}, {rwb}),");
    }

    src.push_str(
        r#"        )
    }};
}

"#,
    );
}

/// Generate fixed-rank scalar and SIMD Wick dispatch source.
/// Both scalar and SIMD dispatch span the complete `MAXEXCIT` fixed-rank region.
/// # Arguments:
/// - `maxexcit`: Maximum compiled determinant excitation rank.
/// # Returns:
/// - `String`: Rust source written to generated `dispatch.rs`.
fn dispatch_source(maxexcit: usize) -> String {
    let pairs = pair_ranks(maxexcit);
    let hamiltonian = hamiltonian_ranks(maxexcit);
    let hamradix = maxexcit + 1;
    let hamspace = hamradix * hamradix * hamradix * hamradix;

    let mut src = String::new();
    src.push_str("// generated by build/build.rs\n");

    write_pair_macro(&mut src, "dispatch_onebody_scalar_ranks", &pairs);
    write_pair_macro(&mut src, "dispatch_onebody_ranks", &pairs);
    write_pair_macro(&mut src, "dispatch_overlap_scalar_ranks", &pairs);
    write_pair_macro(&mut src, "dispatch_overlap_ranks", &pairs);
    write_hamiltonian_macro(&mut src, "dispatch_hamiltonian_scalar_ranks", &hamiltonian);
    write_hamiltonian_macro(&mut src, "dispatch_hamiltonian_ranks", &hamiltonian);

    let _ = writeln!(
        src,
        "pub(super) const HAMNRANKS: usize = {};",
        hamiltonian.len()
    );
    let _ = writeln!(src, "pub(super) const HAMRADIX: usize = {hamradix};");
    let _ = writeln!(src, "pub(super) const HAMSPACE: usize = {hamspace};");
    src.push_str("pub(super) const HAMRANKS: [(usize, usize, usize, usize); HAMNRANKS] = [\n");

    for &(rxa, rwa, rxb, rwb) in &hamiltonian {
        let _ = writeln!(src, "    ({rxa}, {rwa}, {rxb}, {rwb}),");
    }

    src.push_str("];\n");

    src
}

/// Generate build-time Wick configuration and rank-dispatch source.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Writes generated `config.rs` and `dispatch.rs` under `OUT_DIR`.
fn generate_eval() {
    println!("cargo:rerun-if-env-changed=MAXEXCIT");

    let maxexcit = maxexcit();
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is not set"));

    fs::write(out.join("config.rs"), config_source(maxexcit))
        .unwrap_or_else(|e| panic!("failed to write generated config.rs: {e}"));

    fs::write(out.join("dispatch.rs"), dispatch_source(maxexcit))
        .unwrap_or_else(|e| panic!("failed to write generated dispatch.rs: {e}"));
}

/// Generate fixed-rank Wick source and optional NOCC term data.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Writes all build-time artifacts under `OUT_DIR`.
fn main() {
    println!("cargo:rerun-if-changed=build/build.rs");

    generate_eval();

    #[cfg(feature = "nocc")]
    nocc::run();
}
