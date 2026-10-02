// nocc/terms/workspace.rs
//! Reusable per-worker storage for contracting terms.

// Standard library imports.
use std::any::TypeId;

// External crate imports.
use num_complex::Complex64;

// Crate-root imports.
use crate::NOCIScalar;
use crate::maths::contract::{MAXLABELS, TensorShape};

/// Data of one dense block or intermediate: real, or complex once it depends on complex
/// amplitudes.
pub(in crate::nocc) enum Values {
    /// Real elements.
    Real(Vec<f64>),
    /// Complex elements.
    Complex(Vec<Complex64>),
}

/// Borrowed data of one operand.
#[derive(Clone, Copy)]
pub(super) enum View<'a> {
    /// Real elements.
    Real(&'a [f64]),
    /// Complex elements.
    Complex(&'a [Complex64]),
}

/// Mutable output block of one table.
pub(super) enum Out<'a> {
    /// Real elements.
    Real(&'a mut [f64]),
    /// Complex elements.
    Complex(&'a mut [Complex64]),
}

impl Values {
    /// Build a zero block, complex or real.
    /// # Arguments:
    /// - `len`: Number of elements.
    /// - `complex`: Whether the elements are complex.
    /// # Returns:
    /// - `Self`: Zero block.
    pub(super) fn zeros(
        len: usize,
        complex: bool,
    ) -> Self {
        if complex {
            Self::Complex(vec![Complex64::new(0.0, 0.0); len])
        } else {
            Self::Real(vec![0.0; len])
        }
    }

    /// Borrow the data from an element offset.
    /// # Arguments:
    /// - `offset`: First element of the view.
    /// # Returns:
    /// - `View<'_>`: Borrowed elements.
    pub(super) fn view(
        &self,
        offset: usize,
    ) -> View<'_> {
        match self {
            Self::Real(x) => View::Real(&x[offset..]),
            Self::Complex(x) => View::Complex(&x[offset..]),
        }
    }

    /// Borrow the data mutably as an output block.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `Out<'_>`: Mutable elements.
    pub(super) fn out(&mut self) -> Out<'_> {
        match self {
            Self::Real(x) => Out::Real(x),
            Self::Complex(x) => Out::Complex(x),
        }
    }

    /// Add another block of the same kind element by element.
    /// # Arguments:
    /// - `other`: Block to add.
    /// # Returns:
    /// - `()`: Mutates `self`.
    /// # Panics
    /// - Panics if one block is real and the other complex.
    pub(super) fn add_assign(
        &mut self,
        other: Self,
    ) {
        match (self, other) {
            (Self::Real(a), Self::Real(b)) => a.iter_mut().zip(b).for_each(|(x, y)| *x += y),
            (Self::Complex(a), Self::Complex(b)) => a.iter_mut().zip(b).for_each(|(x, y)| *x += y),
            _ => panic!("cannot add a real block to a complex block"),
        }
    }

    /// Convert the elements to the amplitude scalar type, promoting real elements when it is
    /// complex.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `Vec<T>`: Elements as `T`.
    /// # Panics
    /// - Panics if complex elements are requested as real, or `T` is neither `f64` nor
    ///   `Complex64`.
    pub(in crate::nocc) fn into_elements<T: NOCIScalar>(self) -> Vec<T> {
        match self {
            Self::Real(x) => x.into_iter().map(<T as From<f64>>::from).collect(),
            Self::Complex(x) => {
                if TypeId::of::<T>() != TypeId::of::<Complex64>() {
                    panic!("complex block requested as real elements");
                }
                x.into_iter()
                    .map(|z| <T as From<f64>>::from(z.re) + T::from_imag(z.im))
                    .collect()
            }
        }
    }
}

/// Location of an operand's data.
#[derive(Clone, Copy)]
pub(super) enum Source {
    /// A dense factor block, by table-local block id, viewed from an element offset.
    Block(usize, usize),
    /// An intermediate buffer of the current term.
    Buffer(usize),
}

/// Reusable per-worker storage for contracting terms.
pub(super) struct Workspace {
    /// Operands of the current term.
    pub(super) operands: Vec<(Source, TensorShape)>,
    /// Intermediate buffers of the current term.
    pub(super) buffers: Vec<Values>,
    /// Released real buffers available for reuse.
    pub(super) pool: Vec<Vec<f64>>,
    /// Released complex buffers available for reuse.
    pub(super) complex_pool: Vec<Vec<Complex64>>,
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
            complex_pool: Vec::new(),
            masks: Vec::new(),
            keeps: Vec::new(),
            views: Vec::new(),
        }
    }

    /// Return the intermediate buffers from `start` on to the pool of their kind.
    /// # Arguments:
    /// - `start`: First buffer to release.
    /// # Returns:
    /// - `()`: Mutates the buffers and pools.
    pub(super) fn release(
        &mut self,
        start: usize,
    ) {
        for buffer in self.buffers.drain(start..) {
            match buffer {
                Values::Real(x) => self.pool.push(x),
                Values::Complex(x) => self.complex_pool.push(x),
            }
        }
    }
}

/// Take the released buffer that best fits a result: the smallest that holds it, or else the
/// largest, so buffers are rarely grown.
/// # Arguments:
/// - `pool`: Released buffers.
/// - `len`: Number of result elements.
/// # Returns:
/// - `Vec<T>`: Buffer removed from the pool, or a new empty one.
pub(super) fn pooled_buffer<T>(
    pool: &mut Vec<Vec<T>>,
    len: usize,
) -> Vec<T> {
    let fits = (0..pool.len())
        .filter(|&k| pool[k].capacity() >= len)
        .min_by_key(|&k| pool[k].capacity());
    let chosen = fits.or_else(|| (0..pool.len()).max_by_key(|&k| pool[k].capacity()));
    chosen.map_or_else(Vec::new, |k| pool.swap_remove(k))
}
