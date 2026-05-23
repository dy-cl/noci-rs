// input/mod.rs

mod det;
mod excit;
mod mol;
mod parse;
mod prop;
mod qmc;
mod scf;
mod snoci;
mod state;
mod wicks;
mod write;

pub use det::DeterministicOptions;
pub use excit::ExcitationOptions;
pub use mol::MolOptions;
pub use parse::load_input;
pub use prop::{PropagationOptions, Propagator};
pub use qmc::{ExcitationGen, NOCCMCOptions, QMCOptions};
pub use scf::{DiisOptions, HSCFOptions, SCFInfo};
pub use snoci::{GMRESOptions, SNOCIOptions, SNOCIPreconditioner};
pub use state::{Metadynamics, SCFExcitation, SpatialBias, Spin, SpinBias, StateRecipe, StateType};
pub use wicks::{WicksOptions, WicksStorage};
pub use write::WriteOptions;

pub struct Input {
    /// Molecular geometry and basis options.
    pub mol: MolOptions,
    /// SCF convergence and h-SCF options.
    pub scf: SCFInfo,
    /// Output and restart options.
    pub write: WriteOptions,
    /// Reference state search options.
    pub states: StateType,
    /// Deterministic propagation options.
    pub det: Option<DeterministicOptions>,
    /// Stochastic QMC propagation options.
    pub qmc: Option<QMCOptions>,
    /// SNOCI adaptive-space options.
    pub snoci: Option<SNOCIOptions>,
    /// NOCCMC options.
    pub noccmc: Option<NOCCMCOptions>,
    /// Excitation generation options.
    pub excit: ExcitationOptions,
    /// Shared propagation options.
    pub prop: Option<PropagationOptions>,
    /// Non-orthogonal Wick's theorem options.
    pub wicks: WicksOptions,
}

impl Input {
    /// Return immutable reference to propagation options. Will panic if propagation options are
    /// missing when doing QMC or deterministic propagation.
    /// # Returns:
    /// - `&PropagationOptions`: Immutable reference to propagation options.
    pub fn prop_ref(&self) -> &PropagationOptions {
        self.prop.as_ref().unwrap_or_else(|| {
            panic!("Propagation options are required when running deterministic or QMC propagation")
        })
    }

    /// Return mutable reference to propagation options. Will panic if propagation options are
    /// missing when doing QMC or deterministic propagation.
    /// # Returns:
    /// - `&mut PropagationOptions`: Mutable reference to propagation options.
    pub fn prop_mut(&mut self) -> &mut PropagationOptions {
        self.prop.as_mut().unwrap_or_else(|| {
            panic!("Propagation options are required when running deterministic or QMC propagation")
        })
    }
}

impl Default for Input {
    /// Return default input options.
    /// # Returns:
    /// - `Self`: Input options with placeholder mol and states data and default settings elsewhere.
    fn default() -> Self {
        Self {
            mol: MolOptions::default(),
            scf: SCFInfo::default(),
            write: WriteOptions::default(),
            states: StateType::default(),
            det: None,
            qmc: None,
            snoci: None,
            noccmc: None,
            excit: ExcitationOptions::default(),
            prop: None,
            wicks: WicksOptions::default(),
        }
    }
}
