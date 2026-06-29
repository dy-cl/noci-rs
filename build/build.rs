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

#[cfg(feature = "nocc")]
fn main() {
    nocc::run();
}

#[cfg(not(feature = "nocc"))]
fn main() {
    println!("cargo:rerun-if-changed=build/build.rs");
}
