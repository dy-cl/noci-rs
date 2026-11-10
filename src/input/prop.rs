// input/prop.rs

// Standard library imports.
use std::str::FromStr;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Propagator {
    /// Unshifted propagator on the sparse coefficient vector `c`:
    /// `U_{wx} = \delta_{wx} - \Delta\tau[H_{wx} - E_s S_{wx}]`, valid only when `S` has full rank.
    Unshifted,
    /// S-apply propagator with persistent range population `N = Sc`:
    /// `\Delta \simeq -dt(H-E_sS)N`, followed by `N' = N + S\Delta`.
    SApply,
    /// B-apply propagator with persistent range population `N = Sc`:
    /// `\chi \simeq -dt(\hat H-E_s)BN`, followed by `N' = N + B^\dagger\chi`.
    BApply,
}

impl Propagator {
    /// Return propagator as input string.
    /// # Arguments:
    /// - `self`: Propagator selection.
    /// # Returns:
    /// - `&'static str`: String representation used in input parsing.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Unshifted => "unshifted",
            Self::SApply => "s-apply",
            Self::BApply => "b-apply",
        }
    }
}

impl FromStr for Propagator {
    type Err = String;

    /// Parse propagator type from input string.
    /// # Arguments:
    /// - `s`: String specifying the propagator type.
    /// # Returns:
    /// - `Result<Self, Self::Err>`: Parsed propagator if valid string, otherwise error message.
    /// # Errors
    /// - Returns an error if `s` does not name a supported propagator.
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "unshifted" => Ok(Self::Unshifted),
            "s-apply" => Ok(Self::SApply),
            "b-apply" => Ok(Self::BApply),
            _ => Err(format!("invalid propagator: {s}")),
        }
    }
}

impl Default for Propagator {
    /// Return default propagator.
    /// # Returns:
    /// - `Self`: Default propagator choice.
    fn default() -> Self {
        Self::Unshifted
    }
}

pub struct PropagationOptions {
    /// Imaginary-time propagation timestep.
    pub dt: f64,
    /// Propagator approximation.
    pub propagator: Propagator,
}

impl Default for PropagationOptions {
    /// Return default propagation options.
    /// # Returns:
    /// - `Self`: Propagation options with default timestep, step count, and propagator.
    fn default() -> Self {
        Self {
            dt: 1e-4,
            propagator: Propagator::default(),
        }
    }
}
