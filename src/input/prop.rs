// input/prop.rs

use std::str::FromStr;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Propagator {
    /// Unshifted propagator
    /// U^{\mathrm{U}}_{wx} = \delta_{wx} - \Delta\tau[H_{wx} - E_s^S S_{wx}].
    Unshifted,
    /// Shifted propagator
    /// U^{\mathrm{S}}_{wx} = [1 + \Delta\tau E_s^S]\delta_{wx}
    /// - \Delta\tau[H_{wx} - E_s^S S_{wx}].
    Shifted,
    /// Doubly-shifted propagator
    /// U^{\mathrm{DS}}_{wx} = [1 + \Delta\tau E_s]\delta_{wx}
    /// - \Delta\tau[H_{wx} - E_s^S S_{wx}].
    DoublyShifted,
    /// Difference doubly-shifted propagator using
    /// \bar{E}_s = \frac{1}{2}[E_s + E_s^S]:
    /// U^{\mathrm{DDS},1}_{wx} = [1 + \Delta\tau(E_s - E_s^S)]\delta_{wx}
    /// - \Delta\tau[H_{wx} - \bar{E}_s S_{wx}].
    DifferenceDoublyShiftedU1,
    /// Difference doubly-shifted propagator
    /// U^{\mathrm{DDS},2}_{wx} = [1 + \Delta\tau(E_s - E_s^S)]\delta_{wx}
    /// - \Delta\tau[H_{wx} - E_s^S S_{wx}].
    DifferenceDoublyShiftedU2,
    /// Direct-overlap propagator with persistent population N_w = S_{wx}c_x:
    /// U^{\mathrm{DO}}_{wx} = \delta_{wx}
    /// - \Delta\tau S_{wy}[H_{yx} - E_s^S S_{yx}].
    DirectOverlap,
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
            Self::DirectOverlap => "direct-overlap",
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
            "direct-overlap" => Ok(Self::DirectOverlap),
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
