// progress.rs

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Environment-gated progress printer.
pub struct Prog {
    /// Whether progress printing is enabled.
    enabled: bool,
    /// Progress label.
    label: String,
    /// Total number of work items.
    total: usize,
    /// Print interval.
    step: usize,
    /// Completed work-item count.
    done: AtomicUsize,
    /// Start time.
    start: Instant,
}

impl Prog {
    /// Create a progress printer.
    /// # Arguments:
    /// - `label`: Progress label.
    /// - `total`: Total number of work items.
    /// # Returns:
    /// - `Prog`: Progress printer.
    pub fn new(
        label: impl Into<String>,
        total: usize,
    ) -> Self {
        let enabled = std::env::var_os("WICK_PROGRESS").is_some();
        let label = label.into();
        let step = std::env::var("WICK_PROGRESS_STEP")
            .ok()
            .and_then(|x| x.parse::<usize>().ok())
            .filter(|&x| x > 0)
            .unwrap_or_else(|| if total <= 20 { 1 } else { (total / 20).max(1) });

        if enabled {
            eprintln!("[wick] {label}: start 0/{total}");
        }

        Self {
            enabled,
            label,
            total,
            step,
            done: AtomicUsize::new(0),
            start: Instant::now(),
        }
    }

    /// Mark one work item as complete.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `()`: Prints progress when enabled.
    pub fn tick(&self) {
        if !self.enabled {
            return;
        }

        let n = self.done.fetch_add(1, Ordering::Relaxed) + 1;

        if n == self.total || n.is_multiple_of(self.step) {
            let pct = if self.total == 0 {
                100.0
            } else {
                100.0 * n as f64 / self.total as f64
            };
            eprintln!(
                "[wick-time] {}: {}/{} ({pct:.1}%), elapsed {:?}.",
                self.label,
                n,
                self.total,
                self.start.elapsed()
            );
        }
    }
}

/// Read resident memory from `/proc/self/status`.
/// # Arguments:
/// - None.
/// # Returns:
/// - `usize`: Resident memory in MiB, or zero if unavailable.
pub fn rssmb() -> usize {
    let Ok(s) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };

    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb = rest
                .split_whitespace()
                .next()
                .and_then(|x| x.parse::<usize>().ok())
                .unwrap_or(0);
            return kb / 1024;
        }
    }

    0
}

/// Print a memory trace note when enabled.
/// # Arguments:
/// - `msg`: Message to print.
/// # Returns:
/// - `()`: Prints to stderr when `WICK_MEM` is set.
pub fn mem(msg: impl std::fmt::Display) {
    if std::env::var_os("WICK_MEM").is_some() {
        eprintln!("[wick-mem] {msg}, rss: {} MiB", rssmb());
    }
}
