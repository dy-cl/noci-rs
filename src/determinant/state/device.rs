// determinant/state/device.rs
//! `DeviceDeterminant`, the GPU counterpart of `DeterminantState` and `DeterminantData` in
//! `state.rs`: a determinant's occupations and parent, with its alpha and beta excitations and
//! their combined phase measured from that parent. Its layout is fixed so the CUDA struct of the
//! same name in `device.cuh` beside it reads it field by field.

// External crate imports.
use cudarc::driver::DeviceRepr;

// Crate-root imports.
use crate::StateScalar;
use crate::config::MAXEXCIT;

// Parent/sibling imports.
use super::super::{Bits, NOCIIndex, NOCISpace};

/// Number of 64-bit words in a bitstring, `WORDS` in `device.cuh`.
const WORDS: usize = Bits::CAPACITY / 64;

/// One determinant relative to its parent, as the GPU Wick kernels read it.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct DeviceDeterminant {
    /// Alpha and beta occupation bitstrings in the parent frame, orbital `p` at bit `p % 64` of
    /// word `p / 64`.
    occupation: [[u64; WORDS]; 2],
    /// Parent orbital frame.
    parent: u32,
    /// Excitation rank of the alpha and beta spin strings.
    rank: [u8; 2],
    /// Hole orbitals of the alpha and beta spin strings, ascending.
    holes: [[u8; MAXEXCIT]; 2],
    /// Particle orbitals of the alpha and beta spin strings, ascending.
    particles: [[u8; MAXEXCIT]; 2],
    /// Product of the alpha and beta excitation phases.
    phase: f64,
}

// Plain data with a C layout, so the driver may copy it byte for byte.
unsafe impl DeviceRepr for DeviceDeterminant {}

impl DeviceDeterminant {
    /// Collect a determinant's parent-relative excitations.
    /// # Arguments:
    /// - `space`: Determinant space holding the determinant.
    /// - `index`: Determinant in `space`.
    /// # Returns:
    /// - `DeviceDeterminant`: Its record for the GPU.
    pub(crate) fn new<T: StateScalar>(
        space: &NOCISpace<T>,
        index: NOCIIndex,
    ) -> Self {
        let state = space.determinant(index);
        let alpha = &space.alpha(index).data;
        let beta = &space.beta(index).data;
        let mut occupation = [[0; WORDS]; 2];
        occupation[0].copy_from_slice(state.alpha.words());
        occupation[1].copy_from_slice(state.beta.words());

        Self {
            occupation,
            parent: space.parent(index) as u32,
            rank: [alpha.cache.rank, beta.cache.rank],
            holes: [alpha.cache.holes, beta.cache.holes],
            particles: [alpha.cache.particles, beta.cache.particles],
            phase: alpha.phase * beta.phase,
        }
    }
}
