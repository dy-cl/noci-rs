// qmc/excitgen/mod.rs
//! Excitation generators within one orthonormal parent orbital frame.
//!
//! A source determinant is described by its parent frame and the occupied and virtual orbital
//! labels of each spin, so the generators serve any method whose determinants carry those lists,
//! whether or not the determinant belongs to a precomputed space. Connections are returned in the
//! occupied- and virtual-rank form of `OrthogonalConnection` with their exact generation
//! probability.

mod pchb;
mod uniform;

// Crate-visible type re-exports.
pub(crate) use pchb::{PchbGenerator, PchbScratch};
pub(crate) use uniform::OrthogonalUniformGenerator;

/// Occupied and virtual orbital labels of one spin of a source determinant.
#[derive(Clone, Copy)]
pub(crate) struct FrameSpin<'a> {
    /// Physical orbital labels indexed by occupied rank.
    pub(crate) occupied: &'a [u8],
    /// Physical orbital labels indexed by virtual rank.
    pub(crate) virtuals: &'a [u8],
}

/// Source determinant of an excitation, described in its parent orbital frame.
#[derive(Clone, Copy)]
pub(crate) struct FrameSource<'a> {
    /// Parent orbital frame of the source.
    pub(crate) parent: usize,
    /// Whether the source is the parent's reference determinant.
    pub(crate) reference: bool,
    /// Alpha orbital labels of the source.
    pub(crate) alpha: FrameSpin<'a>,
    /// Beta orbital labels of the source.
    pub(crate) beta: FrameSpin<'a>,
}

/// Storage for histogrammed data.
#[derive(Clone)]
pub struct ExcitationHist {
    /// Lower logarithmic bound of the histogram range.
    pub logmin: f64,
    /// Upper logarithmic bound of the histogram range.
    pub logmax: f64,
    /// Number of samples that fell below the histogram range.
    pub noverflow_low: u64,
    /// Number of samples that fell above the histogram range.
    pub noverflow_high: u64,
    /// Bin counts over the configured logarithmic range.
    pub counts: Vec<u64>,
    /// Total number of bins in the histogram.
    pub nbins: usize,
    /// Total number of samples processed by the histogram.
    pub ntotal: u64,
}

impl ExcitationHist {
    /// Constructor for ExcitationHist object. Creates ExcitationHist with chosen parameters.
    /// # Arguments:
    /// - `logmin`: Minimum histogram value on a logarithmic scale.
    /// - `logmax`: Maximum histogram value on a logarithmic scale.
    /// - `nbins`: Number of histogram bins.
    /// # Returns
    /// - `ExcitationHist`: Empty histogram with the requested parameters.
    pub(crate) fn new(
        logmin: f64,
        logmax: f64,
        nbins: usize,
    ) -> Self {
        Self {
            logmin,
            logmax,
            noverflow_low: 0,
            noverflow_high: 0,
            counts: vec![0u64; nbins],
            nbins,
            ntotal: 0,
        }
    }

    /// Add the absolute magnitude of an unrounded spawned population change.
    /// # Arguments:
    /// `self`: ExcitationHist.
    /// - `pspawn`: Spawning probability as defined in population dynamics routines.
    /// # Returns
    /// - `()`: Updates the histogram in place.
    pub(crate) fn add(
        &mut self,
        pspawn: f64,
    ) {
        self.ntotal += 1;

        if !pspawn.is_finite() || pspawn <= 0.0 {
            self.noverflow_low += 1;
            return;
        }

        let logpspawn = pspawn.ln();

        if logpspawn < self.logmin {
            self.noverflow_low += 1;
            return;
        }
        if logpspawn >= self.logmax {
            self.noverflow_high += 1;
            return;
        }

        // Fractional position of logpspawn in histogram range.
        let t = (logpspawn - self.logmin) / (self.logmax - self.logmin);
        // Convert to bin units.
        let b = (t * self.nbins as f64) as usize;
        self.counts[b] += 1;
    }
}
