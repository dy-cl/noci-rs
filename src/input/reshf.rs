// input/reshf.rs

// External crate imports.
use rlua::Table;

/// Largest allowed deviation of the summed state weights from one.
const SUMTOL: f64 = 1e-12;

/// Options of resonating Hartree-Fock (ResHF) reference optimisation.
#[derive(Clone, Debug)]
pub struct ResHFOptions {
    /// State weights `w_I` of `E_{\text{SA}} = \sum_I w_I E_I`, lowest root first.
    pub weights: Vec<f64>,
    /// Relative floor on the singular values of each occupied orbital overlap matrix
    /// (ResHF-cutoff).
    pub cutoff: f64,
}

impl Default for ResHFOptions {
    /// Return default ResHF options: the ground state alone and a floor of `10^{-8}`.
    /// # Returns:
    /// - `Self`: Default ResHF options.
    fn default() -> Self {
        Self {
            weights: vec![1.0],
            cutoff: 1e-8,
        }
    }
}

/// Print an input error and exit, as the other input readers do.
/// # Arguments:
/// - `msg`: Error message.
/// # Returns
/// - `!`: Never returns.
fn invalid(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(1);
}

/// Read ResHF options from an optional Lua table, using defaults for absent keys.
/// The weights must be non-empty, non-negative and sum to one, and the cutoff must be positive.
/// # Arguments:
/// - `tbl`: Optional Lua reshf table.
/// # Returns:
/// - `Option<ResHFOptions>`: Parsed ResHF options, or `None` if the table is absent.
pub(super) fn read_reshf(tbl: Option<Table>) -> Option<ResHFOptions> {
    tbl.map(|tbl| {
        let defaults = ResHFOptions::default();
        let weights: Vec<f64> = tbl
            .get::<_, Option<Table>>("weights")
            .unwrap_or(None)
            .map(|t| {
                t.sequence_values::<f64>()
                    .map(|x| x.unwrap())
                    .collect::<Vec<_>>()
            })
            .unwrap_or(defaults.weights);
        let cutoff: f64 = tbl.get("cutoff").unwrap_or(defaults.cutoff);

        // Validate the state average and the floor.
        if weights.is_empty() {
            invalid("reshf.weights must contain at least one weight");
        }
        if weights.iter().any(|&w| w < 0.0) {
            invalid("reshf.weights must be non-negative");
        }
        let total: f64 = weights.iter().sum();
        if (total - 1.0).abs() > SUMTOL {
            invalid(&format!("reshf.weights must sum to 1, not {total}"));
        }
        if cutoff.is_nan() || cutoff <= 0.0 {
            invalid("reshf.cutoff must be positive");
        }

        ResHFOptions { weights, cutoff }
    })
}
