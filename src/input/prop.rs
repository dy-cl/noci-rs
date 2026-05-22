// input/prop.rs

use std::str::FromStr;

pub enum Propagator {
    Unshifted,
    Shifted,
    DoublyShifted,
    DifferenceDoublyShiftedU1,
    DifferenceDoublyShiftedU2,
}

impl Propagator {
    /// Return propagator as input string.
    /// # Returns:
    /// - `&'static str`: String representation used in input parsing.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Unshifted => "unshifted",
            Self::Shifted => "shifted",
            Self::DoublyShifted => "doubly-shifted",
            Self::DifferenceDoublyShiftedU1 => "difference-doubly-shifted-u1",
            Self::DifferenceDoublyShiftedU2 => "difference-doubly-shifted-u2",
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
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "unshifted" => Ok(Self::Unshifted),
            "shifted" => Ok(Self::Shifted),
            "doubly-shifted" => Ok(Self::DoublyShifted),
            "difference-doubly-shifted-u1" => Ok(Self::DifferenceDoublyShiftedU1),
            "difference-doubly-shifted-u2" => Ok(Self::DifferenceDoublyShiftedU2),
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
