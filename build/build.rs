// Standard library imports.
use std::env;
use std::fmt::Write;
use std::fs;
use std::path::PathBuf;

const SIMDHAMMAXL: usize = 6;
const SIMDMAXRANK: usize = 4;
const SIMDONEMAXL: usize = 4;
const SIMDOVERLAPMAXL: usize = 6;

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

/// Generate one same-spin excitation-rank table.
/// The table contains every nonzero pair `(RX,RW)` with individual ranks no larger than
/// `maxexcit` and total contraction rank no larger than `maxl`.
/// # Arguments:
/// - `maxexcit`: Maximum individual same-spin excitation rank.
/// - `maxl`: Maximum total contraction rank `L = RX + RW`.
/// # Returns:
/// - `Vec<(usize, usize)>`: Ordered `(RX,RW)` rank pairs.
fn pair_ranks(
    maxexcit: usize,
    maxl: usize,
) -> Vec<(usize, usize)> {
    let mut ranks = Vec::new();

    for l in 1..=2 * maxexcit {
        if l > maxl {
            break;
        }

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

/// Generate two-spin Hamiltonian excitation-rank tuples.
/// Scalar tuples contain every bra and ket determinant whose total alpha-plus-beta excitation rank
/// is at most `maxexcit`. SIMD tuples additionally obey the currently implemented SIMD limits.
/// # Arguments:
/// - `maxexcit`: Maximum total excitation rank of each determinant.
/// - `simd`: Whether to restrict the table to the currently implemented SIMD region.
/// # Returns:
/// - `Vec<(usize, usize, usize, usize)>`: Ordered `(RXA,RWA,RXB,RWB)` tuples.
fn hamiltonian_ranks(
    maxexcit: usize,
    simd: bool,
) -> Vec<(usize, usize, usize, usize)> {
    let maxrank = if simd {
        maxexcit.min(SIMDMAXRANK)
    } else {
        maxexcit
    };
    let mut ranks = Vec::new();

    for rxa in 0..=maxrank {
        for rwa in 0..=maxrank {
            for rxb in 0..=maxrank {
                for rwb in 0..=maxrank {
                    if rxa + rxb > maxexcit || rwa + rwb > maxexcit {
                        continue;
                    }

                    if simd && rxa + rwa + rxb + rwb > SIMDHAMMAXL {
                        continue;
                    }

                    ranks.push((rxa, rwa, rxb, rwb));
                }
            }
        }
    }

    ranks
}

/// Generate crate-wide compile-time Wick configuration constants.
/// # Arguments:
/// - `maxexcit`: Maximum compiled determinant excitation rank.
/// # Returns:
/// - `String`: Rust source written to generated `config.rs`.
fn config_source(maxexcit: usize) -> String {
    let maxl = 2 * maxexcit;
    let maxdet = maxl * maxl;
    let minorrank = maxl.saturating_sub(2);
    let maxminor = minorrank * minorrank;

    format!(
        concat!(
            "// generated by build/build.rs\n",
            "pub(crate) const MAXDET: usize = {};\n",
            "pub(crate) const MAXEXCIT: usize = {};\n",
            "pub(crate) const MAXL: usize = {};\n",
            "pub(crate) const MAXMINOR: usize = {};\n",
            "pub(crate) const SIMDHAMMAXL: usize = {};\n",
            "pub(crate) const SIMDMAXRANK: usize = {};\n",
            "pub(crate) const SIMDONEMAXL: usize = {};\n",
            "pub(crate) const SIMDOVERLAPMAXL: usize = {};\n",
        ),
        maxdet, maxexcit, maxl, maxminor, SIMDHAMMAXL, SIMDMAXRANK, SIMDONEMAXL, SIMDOVERLAPMAXL,
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
        |$rx:ident, $rw:ident, $l:ident| $kernel:expr,
        $fallback:expr $(,)?
    ) => {{
        dispatch_pair_ranks!(
            @match $ranks,
            |$rx, $rw, $l| $kernel,
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
            $rxa:ident, $rwa:ident, $la:ident,
            $rxb:ident, $rwb:ident, $lb:ident,
            $da:ident, $db:ident, $sa:ident, $sb:ident
        | $kernel:expr,
        $fallback:expr $(,)?
    ) => {{
        dispatch_hamiltonian_ranks_inner!(
            @match $ranks,
            |
                $rxa, $rwa, $la, $rxb, $rwb, $lb,
                $da, $db, $sa, $sb
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
/// Scalar dispatch follows `MAXEXCIT`, while SIMD dispatch is its intersection with the currently
/// implemented SIMD rank regions.
/// # Arguments:
/// - `maxexcit`: Maximum compiled determinant excitation rank.
/// # Returns:
/// - `String`: Rust source written to generated `dispatch.rs`.
fn dispatch_source(maxexcit: usize) -> String {
    let scalar_pairs = pair_ranks(maxexcit, 2 * maxexcit);
    let onebody_simd = pair_ranks(maxexcit.min(SIMDMAXRANK), SIMDONEMAXL);
    let overlap_simd = pair_ranks(maxexcit.min(SIMDMAXRANK), SIMDOVERLAPMAXL);
    let hamiltonian_scalar = hamiltonian_ranks(maxexcit, false);
    let hamiltonian_simd = hamiltonian_ranks(maxexcit, true);
    let hamradix = maxexcit.min(SIMDMAXRANK) + 1;
    let hamspace = hamradix * hamradix * hamradix * hamradix;

    let mut src = String::new();
    src.push_str("// generated by build/build.rs\n");

    write_pair_macro(&mut src, "dispatch_onebody_scalar_ranks", &scalar_pairs);
    write_pair_macro(&mut src, "dispatch_onebody_ranks", &onebody_simd);
    write_pair_macro(&mut src, "dispatch_overlap_scalar_ranks", &scalar_pairs);
    write_pair_macro(&mut src, "dispatch_overlap_ranks", &overlap_simd);
    write_hamiltonian_macro(
        &mut src,
        "dispatch_hamiltonian_scalar_ranks",
        &hamiltonian_scalar,
    );
    write_hamiltonian_macro(&mut src, "dispatch_hamiltonian_ranks", &hamiltonian_simd);

    let _ = writeln!(
        src,
        "pub(super) const HAMNRANKS: usize = {};",
        hamiltonian_simd.len()
    );
    let _ = writeln!(src, "pub(super) const HAMRADIX: usize = {hamradix};");
    let _ = writeln!(src, "pub(super) const HAMSPACE: usize = {hamspace};");
    src.push_str("pub(super) const HAMRANKS: [(usize, usize, usize, usize); HAMNRANKS] = [\n");

    for &(rxa, rwa, rxb, rwb) in &hamiltonian_simd {
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
