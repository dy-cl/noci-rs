// nonorthogonalwicks/requirements.rs
//! Backend capability requirements for nonorthogonal Wick data.

/// Required nonorthogonal Wick data for a backend consumer.
#[derive(Clone, Copy)]
pub(crate) struct WicksRequirements {
    /// Whether overlap intermediates are required.
    pub(crate) overlap: bool,
    /// Whether one-body intermediates are required.
    pub(crate) one_body: bool,
    /// Whether two-body intermediates are required.
    pub(crate) two_body: bool,
    /// Whether transition-density intermediates are required.
    pub(crate) rdm: bool,
}

impl WicksRequirements {
    /// Request overlap-only Wick data.
    /// # Returns
    /// - `WicksRequirements`: Requirements for overlap matrix elements.
    pub(crate) fn overlap() -> Self {
        Self {
            overlap: true,
            one_body: false,
            two_body: false,
            rdm: false,
        }
    }

    /// Request overlap and one-body Wick data.
    /// # Returns
    /// - `WicksRequirements`: Requirements for factorised one-body NOCI-PT2 evaluation.
    pub(crate) fn one_body() -> Self {
        Self {
            overlap: true,
            one_body: true,
            two_body: false,
            rdm: false,
        }
    }

    /// Request overlap, one-body and two-body Wick data.
    /// # Returns
    /// - `WicksRequirements`: Requirements for Hamiltonian matrix elements.
    pub(crate) fn hamiltonian() -> Self {
        Self {
            overlap: true,
            one_body: true,
            two_body: true,
            rdm: false,
        }
    }

    /// Request transition-density Wick data.
    /// # Returns
    /// - `WicksRequirements`: Requirements for RDM matrix elements.
    pub(crate) fn rdm() -> Self {
        Self {
            overlap: true,
            one_body: true,
            two_body: true,
            rdm: true,
        }
    }
}
