// nocc/terms/workspace.rs
//! Reusable per-worker storage for contracting terms.

// Crate-root imports.
use crate::maths::contract::{MAXLABELS, TensorShape};

/// Location of an operand's data.
#[derive(Clone, Copy)]
pub(super) enum Source {
    /// A dense factor block, by table-local block id, viewed from an element offset.
    Block(usize, usize),
    /// An intermediate buffer of the current term.
    Buffer(usize),
    /// The product shared by the current term's group, viewed from an element offset.
    Shared(usize),
}

/// Reusable per-worker storage for contracting terms.
pub(super) struct Workspace {
    /// Operands of the current term.
    pub(super) operands: Vec<(Source, TensorShape)>,
    /// Intermediate buffers of the current term.
    pub(super) buffers: Vec<Vec<f64>>,
    /// Released buffers available for reuse.
    pub(super) pool: Vec<Vec<f64>>,
    /// Labels of every operand and step result of the current term.
    pub(super) masks: Vec<u64>,
    /// Labels kept by every step of the current term.
    pub(super) keeps: Vec<u64>,
    /// Every input operand of a sliced term with its sliced labels removed, and its offset per
    /// unit of each sliced label.
    pub(super) views: Vec<(Source, TensorShape, [usize; MAXLABELS])>,
}

impl Workspace {
    /// Build empty worker storage.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `Self`: Empty workspace.
    pub(super) fn new() -> Self {
        Self {
            operands: Vec::new(),
            buffers: Vec::new(),
            pool: Vec::new(),
            masks: Vec::new(),
            keeps: Vec::new(),
            views: Vec::new(),
        }
    }
}

/// Take the released buffer that best fits a result: the smallest that holds it, or else the
/// largest, so buffers are rarely grown.
/// # Arguments:
/// - `pool`: Released buffers.
/// - `len`: Number of result elements.
/// # Returns:
/// - `Vec<f64>`: Buffer removed from the pool, or a new empty one.
pub(super) fn pooled_buffer(
    pool: &mut Vec<Vec<f64>>,
    len: usize,
) -> Vec<f64> {
    let fits = (0..pool.len())
        .filter(|&k| pool[k].capacity() >= len)
        .min_by_key(|&k| pool[k].capacity());
    let chosen = fits.or_else(|| (0..pool.len()).max_by_key(|&k| pool[k].capacity()));
    chosen.map_or_else(Vec::new, |k| pool.swap_remove(k))
}
