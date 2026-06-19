// timers/print.rs

use super::{Counter, Totals};

/// Add several non-overlapping timing counters.
/// # Arguments:
/// - `counters`: Timing counters to combine.
/// # Returns:
/// - `Counter`: Sum of elapsed nanoseconds and calls.
fn sum(counters: &[Counter]) -> Counter {
    let mut out = Counter::default();

    for counter in counters {
        out.merge_from(counter);
    }

    out
}

/// Print one timing counter.
/// # Arguments:
/// - `label`: Counter label.
/// - `counter`: Timing counter.
/// - `indent`: Number of leading spaces.
/// # Returns:
/// - `()`: Prints one timing row.
fn print_counter(label: &str, counter: Counter, indent: usize) {
    let avg = counter
        .ns
        .checked_div(counter.calls)
        .map(std::time::Duration::from_nanos)
        .unwrap_or_default();

    println!(
        "{}{}: {:?} [{} calls, avg {:?}/call]",
        " ".repeat(indent),
        label,
        counter.duration(),
        counter.calls,
        avg,
    );
}

/// Print one timing counter relative to a parent counter.
/// # Arguments:
/// - `label`: Counter label.
/// - `counter`: Child timing counter.
/// - `parent`: Parent timing counter.
/// - `indent`: Number of leading spaces.
/// # Returns:
/// - `()`: Prints one relative timing row.
fn print_relative_counter(label: &str, counter: Counter, parent: Counter, indent: usize) {
    let avg = std::time::Duration::from_nanos(
        counter.ns.checked_div(counter.calls).unwrap_or(0),
    );

    let pct = if parent.ns > 0 {
        100.0 * counter.ns as f64 / parent.ns as f64
    } else {
        0.0
    };

    println!(
        "{}{}: {:?} [{} calls, avg {:?}/call, {:.2}%]",
        " ".repeat(indent),
        label,
        counter.duration(),
        counter.calls,
        avg,
        pct,
    );
}

/// Print Wick-tool timing totals.
/// # Arguments:
/// - `timings`: Timing totals summed across all Rayon threads.
/// # Returns:
/// - `()`: Prints the timing report.
pub fn print(timings: Totals) {
    let nthreads = rayon::current_num_threads();

    let canonical_total = sum(&[
        timings.canonical.accumulate,
        timings.canonical.merge,
        timings.canonical.finish,
    ]);

    let wick_total = sum(&[
        timings.wick.spin,
        timings.wick.eval1,
        timings.wick.eval1c,
        timings.wick.eval1cstream,
    ]);

    println!();
    println!("{}", "=".repeat(100));
    println!("Number of Rayon threads: {}", nthreads);
    println!("Warning: Timing functions will impact performance and thread efficiency.");
    println!("Please interpret these timings as a distribution but not absolute.");
    println!("Timing overhead will be quite large relative to some of the smaller kernels.");

    println!("{}", "-".repeat(100));

    println!("Top-level Wick and canonicalisation comparison");
    print_counter("Wick generation work", wick_total, 0);
    print_counter("Canonicalisation work", canonical_total, 0);

    println!("{}", "-".repeat(100));

    println!("Canonicalisation timings");
    print_relative_counter(
        "Canonical term accumulation",
        timings.canonical.accumulate,
        canonical_total,
        2,
    );
    print_relative_counter(
        "Canonical accumulator merging",
        timings.canonical.merge,
        canonical_total,
        2,
    );
    print_relative_counter(
        "Canonical accumulator finish",
        timings.canonical.finish,
        canonical_total,
        2,
    );
    print_relative_counter(
        "Initial expression reconstruction",
        timings.canonical.intoexpr,
        timings.canonical.finish,
        5,
    );
    print_relative_counter(
        "High-rank cumulant sparsification",
        timings.canonical.spar,
        timings.canonical.finish,
        5,
    );
    print_relative_counter(
        "Final canonical sum",
        timings.canonical.final_sum,
        timings.canonical.finish,
        5,
    );

    println!("{}", "-".repeat(100));

    println!("Non-overlapping Wick generation timings");
    print_relative_counter(
        "Spin-string expansion",
        timings.wick.spin,
        wick_total,
        2,
    );
    print_relative_counter(
        "Non-connected spin-string evaluation",
        timings.wick.eval1,
        wick_total,
        2,
    );
    print_relative_counter(
        "Connected spin-string evaluation",
        timings.wick.eval1c,
        wick_total,
        2,
    );
    print_relative_counter(
        "Streamed connected spin-string evaluation",
        timings.wick.eval1cstream,
        wick_total,
        2,
    );

    println!("{}", "-".repeat(100));

    println!("Wick block construction breakdown");
    print_counter("Construct Wick blocks", timings.wick.blocks, 2);
    print_relative_counter(
        "Enumerate frozen contractions",
        timings.wick.frozen,
        timings.wick.blocks,
        5,
    );
    print_relative_counter(
        "Enumerate active cumulant blocks",
        timings.wick.active,
        timings.wick.blocks,
        5,
    );
    print_relative_counter(
        "Evaluate block values",
        timings.wick.val,
        timings.wick.blocks,
        5,
    );
    print_counter(
        "Intern Wick delta factors",
        timings.wick.store_delta,
        5,
    );
    print_counter(
        "Intern Wick tensor factors",
        timings.wick.store_tensor,
        5,
    );

    println!("{}", "-".repeat(100));

    println!("Wick spin-projection timings");
    print_counter(
        "Spin-projection lookup",
        timings.wick.proj,
        2,
    );
    print_counter(
        "Projection-table construction",
        timings.wick.ptab,
        5,
    );
    print_counter(
        "Individual projection construction",
        timings.wick.pval,
        5,
    );
    print_counter(
        "Pack spin labels",
        timings.wick.sbits,
        5,
    );
    print_counter(
        "Extract spin bits",
        timings.wick.bit,
        5,
    );

    println!("{}", "-".repeat(100));

    println!("Wick enumeration timings");
    print_counter(
        "Exact-cover suffix enumeration",
        timings.wick.walk,
        2,
    );
    print_counter(
        "Connected recursive enumeration",
        timings.wick.walkc,
        2,
    );
    print_relative_counter(
        "Choose exact-cover pivot",
        timings.wick.pick,
        timings.wick.walkc,
        5,
    );
    print_relative_counter(
        "Check partial connectivity",
        timings.wick.canconnect,
        timings.wick.walkc,
        5,
    );
    print_relative_counter(
        "Check candidate connectivity",
        timings.wick.canconnect1,
        timings.wick.walkc,
        5,
    );
    print_relative_counter(
        "Find root-connected component",
        timings.wick.rootseen,
        timings.wick.walkc,
        5,
    );
    print_relative_counter(
        "Check completed connectivity",
        timings.wick.conn,
        timings.wick.walkc,
        5,
    );
    print_relative_counter(
        "Compute crossing sign",
        timings.wick.cross,
        timings.wick.walkc,
        5,
    );
    print_relative_counter(
        "Normal order active operators",
        timings.wick.norm,
        timings.wick.walkc,
        5,
    );
    print_relative_counter(
        "Join connected prefix and suffix",
        timings.wick.joinrow,
        timings.wick.walkc,
        5,
    );

    println!("{}", "-".repeat(100));

    println!("Wick numeric accumulation and decoding timings");
    print_counter(
        "Accumulate numeric rows",
        timings.wick.add,
        2,
    );
    print_relative_counter(
        "Sort numeric factor ids",
        timings.wick.sortids,
        timings.wick.add,
        5,
    );
    print_counter(
        "Decode numeric rows",
        timings.wick.out,
        2,
    );
    print_relative_counter(
        "Construct numeric rows",
        timings.wick.row,
        timings.wick.out,
        5,
    );
    print_relative_counter(
        "Extract mask bits",
        timings.wick.bits,
        timings.wick.out,
        5,
    );
    print_relative_counter(
        "Construct position masks",
        timings.wick.mask,
        timings.wick.out,
        5,
    );
    print_relative_counter(
        "Construct group masks",
        timings.wick.gmask,
        timings.wick.out,
        5,
    );

    println!("{}", "-".repeat(100));

    println!("Overlapping Wick wrapper timings");
    print_counter(
        "Public non-connected evaluator",
        timings.wick.eval,
        2,
    );
    print_counter(
        "Public connected evaluator",
        timings.wick.evalc,
        2,
    );
    print_counter(
        "Non-connected evaluator implementation",
        timings.wick.eval0,
        2,
    );
    print_counter(
        "Connected evaluator implementation",
        timings.wick.evalc0,
        2,
    );
    print_counter(
        "Connected streaming wrapper",
        timings.wick.evalcstream,
        2,
    );

    println!("{}", "-".repeat(100));

    println!("Wick streaming configuration timings");
    print_counter(
        "Read spin-batch setting",
        timings.wick.spinbatch,
        2,
    );
    print_counter(
        "Read spin-parallelism setting",
        timings.wick.spinpar,
        2,
    );
    print_counter(
        "Read stream-queue setting",
        timings.wick.streamqueue,
        2,
    );
    print_counter(
        "Read accumulator-flush setting",
        timings.wick.accflush,
        2,
    );

    println!("{}", "=".repeat(100));
}
