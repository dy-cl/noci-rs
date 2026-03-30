use std::fs;
use std::str::FromStr;

use rlua::{Lua, Value, Table};

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
    /// - `Result<Self,  Self::Err>`: Parsed propagator  if  valid string,  otherwise error
    ///   message.
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

pub enum ExcitationGen {
    Uniform,
    HeatBath,
    ApproximateHeatBath,
}

impl FromStr for ExcitationGen {
    type Err = String;

    /// Parse excitation generator from input string.
    /// # Arguments:
    /// - `s`: String specifying the excitation generator.
    /// # Returns:
    /// - `Result<Self, Self::Err>`: Parsed excitation generator if valid string, otherwise error message.
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "uniform" => Ok(Self::Uniform),
            "heat-bath" => Ok(Self::HeatBath),
            "approximate-heat-bath" => Ok(Self::ApproximateHeatBath),
            _ => Err(format!("invalid excitation generator: {s}")),
        }
    }
}

impl Default for ExcitationGen {
    /// Return default excitation generator.
    /// # Returns:
    /// - `Self`: Default excitation generator choice.
    fn default() -> Self {
        Self::Uniform
    }
}

pub enum Spin {
    Alpha, 
    Beta,
    Both,
}

impl Spin {
    /// Return excitation spin as input string.
    /// # Returns:
    /// - `&'static str`: String representation used in input parsing.
    fn as_str(&self) -> &'static str {
        match self {
            Self::Alpha => "alpha",
            Self::Beta => "beta",
            Self::Both => "both",
        }
    }
}

impl FromStr for Spin {
    type Err = String;

    /// Parse excitation spin from input string.
    /// # Arguments:
    /// - `s`: String specifying the excitation spin.
    /// # Returns:
    /// - `Result<Self, Self::Err>`: Parsed spin if valid string, otherwise error message.
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "alpha" => Ok(Self::Alpha),
            "beta" => Ok(Self::Beta),
            "both" => Ok(Self::Both),
            _ => Err(format!("invalid excitation spin: {s}")),
        }
    }
}

impl Default for Spin {
    /// Return default excitation spin.
    /// # Returns:
    /// - `Self`: Default excitation spin choice.
    fn default() -> Self {
        Self::Both
    }
}

pub struct MolOptions {
    pub basis: String,
    pub unit: String,
    pub r_list: Vec<f64>,
    pub geoms: Vec<Vec<String>>,
}

impl Default for MolOptions {
    /// Return default molecular options.
    /// # Returns:
    /// - `Self`: Molecular options with placeholder empty geometry data.
    fn default() -> Self {
        Self {
            basis: String::new(),
            unit: "Ang".to_string(),
            r_list: Vec::new(),
            geoms: Vec::new(),
        }
    }
}

pub struct DiisOptions {
    pub space: usize,
}

impl Default for DiisOptions {
    /// Return default DIIS options.
    /// # Returns:
    /// - `Self`: DIIS options with default subspace size.
    fn default() -> Self {
        Self {
            space: 8,
        }
    }
}

pub struct SCFInfo {
    pub max_cycle: i32,
    pub e_tol: f64,
    pub diis: DiisOptions,
    pub do_fci: bool,
}

impl Default for SCFInfo {
    /// Return default SCF options.
    /// # Returns:
    /// - `Self`: SCF options with standard convergence and DIIS settings.
    fn default() -> Self {
        Self {
            max_cycle: 10_000,
            e_tol: 1e-12,
            diis: DiisOptions::default(),
            do_fci: false,
        }
    }
}

pub struct SCFExcitation {
    pub spin: Spin, 
    pub occ: i32, 
    pub vir: i32,
}

impl Default for SCFExcitation {
    /// Return default SCF excitation.
    /// # Returns:
    /// - `Self`: Excitation with no excitation.
    fn default() -> Self {
        Self {
            spin: Spin::default(),
            occ: 0,
            vir: 0,
        }
    }
}

pub struct SpinBias {
    pub pattern: Vec<i8>,
    pub pol: f64
}

impl Default for SpinBias {
    /// Return default spin bias options.
    /// # Returns:
    /// - `Self`: Spin bias with empty pattern and zero polarization.
    fn default() -> Self {
        Self {
            pattern: Vec::new(),
            pol: 0.0,
        }
    }
}

pub struct SpatialBias {
    pub pattern: Vec<i8>,
    pub pol: f64,
}

impl Default for SpatialBias {
    /// Return default spatial bias options.
    /// # Returns:
    /// - `Self`: Spatial bias with empty pattern and zero polarization.
    fn default() -> Self {
        Self {
            pattern: Vec::new(),
            pol: 0.0,
        }
    }
}

pub struct StateRecipe {
    pub label: String, 
    pub spin_bias: Option<SpinBias>,
    pub spatial_bias: Option<SpatialBias>,
    pub scfexcitation: Option<SCFExcitation>,
    pub noci: bool,
}

impl Default for StateRecipe {
    /// Return default state recipe.
    /// # Returns:
    /// - `Self`: State recipe with empty label and no bias or excitation. We assume that a state
    ///   is desired to be used within NOCI by default.
    fn default() -> Self {
        Self {
            label: String::new(),
            spin_bias: None,
            spatial_bias: None,
            scfexcitation: None,
            noci: true,
        }
    }
}

pub struct ExcitationOptions {
    pub orders: Vec<usize>, 
}

impl ExcitationOptions {
    /// Return default NOCI excitation options.
    /// # Returns:
    /// - `Self`: Excitation options with singles and doubles enabled.
    fn default() -> Self {
        Self {
            orders: [1, 2].to_vec(),
        }
    }
}

pub struct PropagationOptions {
    pub dt: f64,
    pub max_steps: usize,
    pub propagator: Propagator,
}

impl Default for PropagationOptions {
    /// Return default propagation options.
    /// # Returns:
    /// - `Self`: Propagation options with default timestep, step count, and propagator.
    fn default() -> Self {
        Self {
            dt: 1e-4,
            max_steps: 5000000,
            propagator: Propagator::default(),
        }
    }
}

pub struct DeterministicOptions {
    pub dynamic_shift: bool,
    pub dynamic_shift_alpha: f64,
    pub e_tol: f64,
}

impl Default for DeterministicOptions {
    /// Return default deterministic propagation options.
    /// # Returns:
    /// - `Self`: Deterministic propagation options with dynamic shift disabled.
    fn default() -> Self {
        Self {
            dynamic_shift: true,
            dynamic_shift_alpha: 0.1,
            e_tol: 1e-10,
        }
    }
}

pub struct QMCOptions {
    pub initial_population: i64,
    pub target_population: i64,
    pub shift_damping: f64, 
    pub shift_update_freq: usize,
    pub excitation_gen: ExcitationGen,
    pub seed: Option<u64>,
}

impl Default for QMCOptions {
    /// Return default QMC propagation options.
    /// # Returns:
    /// - `Self`: QMC propagation options with default population and shift settings.
    fn default() -> Self {
        Self {
            initial_population: 100,
            target_population: 100000,
            shift_damping: 5e-4,
            shift_update_freq: 1,
            excitation_gen: ExcitationGen::default(),
            seed: None,
        }
    }
}

pub struct WriteOptions {
    pub verbose: bool,
    pub write_deterministic_coeffs: bool,
    pub write_orbitals: bool,
    pub write_excitation_hist: bool,
    pub write_matrices: bool,
    pub write_dir: String,
}

impl Default for WriteOptions {
    /// Return default output options.
    /// # Returns:
    /// - `Self`: Output options with all optional writes disabled.
    fn default() -> Self {
        Self {
            verbose: true,
            write_deterministic_coeffs: false,
            write_orbitals: false,
            write_excitation_hist: false,
            write_matrices: false,
            write_dir: "outputs/".to_string(),
        }
    }
}

pub enum WicksStorage {
    RAM,
    Disk,
}

pub struct WicksOptions {
    pub compare: bool,
    pub enabled: bool,
    pub storage: WicksStorage,
    pub cachedir: Option<String>,
}

impl Default for WicksOptions {
    /// Return default Wick's theorem options.
    /// # Returns:
    /// - `Self`: Wick's options with comparison disabled.
    fn default() -> Self {
        Self {
            compare: false,
            enabled: true,
            storage: WicksStorage::RAM,
            cachedir: Some(".".to_string()),
        }
    }
}

pub struct Metadynamics {
    pub nstates_rhf: usize,
    pub nstates_uhf: usize,
    pub spinpol: f64,
    pub spatialpol: f64,
    pub lambda: f64,
    pub labels_rhf: Vec<String>,
    pub labels_uhf: Vec<String>,
    pub spatial_patterns_rhf: Vec<Option<Vec<i8>>>, 
    pub spin_patterns_uhf: Vec<Option<Vec<i8>>>,   
    pub max_attempts: usize,
}

impl Default for Metadynamics {
    /// Return default metadynamics options.
    /// # Returns:
    /// - `Self`: Metadynamics options with no requested states and empty labels and patterns.
    fn default() -> Self {
        Self {
            nstates_rhf: 0,
            nstates_uhf: 0,
            spinpol: 0.0,
            spatialpol: 0.0,
            lambda: 0.0,
            labels_rhf: Vec::new(),
            labels_uhf: Vec::new(),
            spatial_patterns_rhf: Vec::new(),
            spin_patterns_uhf: Vec::new(),
            max_attempts: 100,
        }
    }
}

pub enum StateType {
    Mom(Vec<StateRecipe>),
    Metadynamics(Metadynamics),
}

impl Default for StateType {
    /// Return default state specification.
    /// # Returns:
    /// - `Self`: Default empty MOM state list.
    fn default() -> Self {
        Self::Mom(Vec::new())
    }
}

pub struct GMRESOptions {
    pub max_iter: usize,      
    pub res_tol: f64, 
    pub metric_tol: f64,
}

impl Default for GMRESOptions {
    /// Return default GMRES options.
    /// # Returns:
    /// - `Self`: GMRES options with default iteration limit and residual tolerance.
    fn default() -> Self {
        Self {
            max_iter: 100,
            res_tol: 1e-8,
            metric_tol: 1e-8,
        }
    }
}

pub struct SNOCIOptions {
    pub sigma: f64,
    pub tol: f64,
    pub max_iter: usize,
    pub max_add: usize,
    pub max_dim: usize,
    pub gmres: GMRESOptions,
}

impl Default for SNOCIOptions {
    /// Return default SNOCI options.
    /// # Returns:
    /// - `Self`: SNOCI options with default selection and GMRES parameters.
    fn default() -> Self {
        Self {
            sigma: 1e-6,
            tol: 1e-8,
            max_iter: 100,
            max_add: 1,
            max_dim: 100,
            gmres: GMRESOptions::default(),
        }
    }
}

pub struct Input {
    pub mol: MolOptions,
    pub scf: SCFInfo,
    pub write: WriteOptions,
    pub states: StateType,
    pub det: Option<DeterministicOptions>,
    pub qmc: Option<QMCOptions>,
    pub snoci: Option<SNOCIOptions>,
    pub excit: ExcitationOptions,
    pub prop: Option<PropagationOptions>,
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
            excit: ExcitationOptions::default(),
            prop: None,
            wicks: WicksOptions::default(),
        }
    }
}

/// Read integer pattern entries taking values in {-1, 0, 1}.
/// # Arguments:
/// - `pat_tbl`: Lua table containing the pattern entries.
/// # Returns:
/// - `Vec<i8>`: Parsed pattern entries.
fn read_pattern(pat_tbl: Table) -> Vec<i8> {
    pat_tbl.sequence_values::<i64>().map(|x| x.unwrap()).map(|x| match x {
        -1 => -1,
        0 => 0,
        1 => 1,
        _ => {
            eprintln!("pattern entries must be -1, 0, or 1");
            std::process::exit(1);
        }
    }).collect()
}

/// Read basis state recipe from Lua table.
/// # Arguments:
/// - `t`: Lua table containing the state recipe specification.
/// # Returns:
/// - `StateRecipe`: Parsed state recipe with optional spin bias, spatial bias, and SCF excitation
///   data.
fn read_state_recipe(t: Table) -> StateRecipe {
    let defaults = StateRecipe::default();
    let label: String = t.get("label").unwrap_or(defaults.label);
    let noci: bool = t.get("noci").unwrap_or(defaults.noci);

    let spin_bias = t.get::<_, Option<Table>>("spin_bias").unwrap_or(None).map(|sb| {
        let defaults = SpinBias::default();
        let pol: f64 = sb.get("pol").unwrap_or(defaults.pol);
        let pat_tbl: Table = sb.get("pattern").unwrap();
        let pattern = read_pattern(pat_tbl);
        SpinBias { pattern, pol }
    });

    let spatial_bias = t.get::<_, Option<Table>>("spatial_bias").unwrap_or(None).map(|sb| {
        let defaults = SpatialBias::default();
        let pol: f64 = sb.get("pol").unwrap_or(defaults.pol);
        let pat_tbl: Table = sb.get("pattern").unwrap();
        let pattern = read_pattern(pat_tbl);
        SpatialBias { pattern, pol }
    });

    let scfexcitation = t.get::<_, Option<Table>>("excit").unwrap_or(None).map(|ex| {
        let defaults = SCFExcitation::default();
        let s: String = ex.get("spin").unwrap_or_else(|_| defaults.spin.as_str().to_string());
        let spin: Spin = s.parse().unwrap_or_else(|msg| {
            eprintln!("{msg}");
            std::process::exit(1);
        });
        SCFExcitation {
            spin,
            occ: ex.get("occ").unwrap_or(defaults.occ),
            vir: ex.get("vir").unwrap_or(defaults.vir),
        }
    });

    StateRecipe { label, spin_bias, spatial_bias, scfexcitation, noci }
}

/// Read input parameters from lua file and assign to Input object.
/// # Arguments
/// - `path`: File path to input file.
pub fn load_input(path: &str) -> Input {

    let src = fs::read_to_string(path).unwrap();
    let lua = Lua::new();

    let ctx = lua; 
    ctx.load(&src).exec().unwrap();
    let globals = ctx.globals();

    // Non-optional table headers.
    let mol_tbl: Table = globals.get("mol").unwrap_or_else(|_| {
        println!("Missing required table 'mol'");
        std::process::exit(1);
    });
    let state_tbl: Table = globals.get("states").unwrap_or_else(|_| {
        println!("Missing required table 'states'");
        std::process::exit(1);
    });

    // Optional table headers.
    let scf_tbl: Option<Table> = globals.get::<_, Option<Table>>("scf").unwrap_or(None);
    let write_tbl: Option<Table> = globals.get::<_, Option<Table>>("write").unwrap_or(None);
    let excit_tbl: Option<Table> = globals.get::<_, Option<Table>>("excit").unwrap_or(None);
    let prop_tbl: Option<Table> = globals.get::<_, Option<Table>>("prop").unwrap_or(None);
    let wicks_tbl: Option<Table> = globals.get::<_, Option<Table>>("wicks").unwrap_or(None);

    // Optional subtables within states.
    let mom_tbl: Option<Table> = state_tbl.get::<_, Option<Table>>("mom").unwrap_or(None);
    let meta_tbl: Option<Table> = state_tbl.get::<_, Option<Table>>("metadynamics").unwrap_or(None);

    // Mol table.
    let basis: String = mol_tbl.get("basis").unwrap();
    let unit: String = mol_tbl.get("unit").unwrap();
    // Allow mol.r to be a number or table. 
    let r_val: Value = mol_tbl.get("r").unwrap();
    let mut r_list: Vec<f64> = Vec::new();
    match r_val {
        // For a number simply add to r_list.
        Value::Number(x) => {r_list.push(x)},
        // For a table of r iterate over all values and add to r_list.
        Value::Table(t) => {
            for item in t.sequence_values::<f64>() {
                let r = item.unwrap();
                r_list.push(r);
            }
        },
        _ => {eprintln!("Number or table required by mol.r"); std::process::exit(1);},
    }
    // Allow mol.atoms to be either a lua table or function.
    let atoms_val: Value = mol_tbl.get("atoms").unwrap();
    let geoms: Vec<Vec<String>> = match atoms_val {
        // If atoms is a lua table we have a static geometry and can duplicate 
        // this geometry across all r (which for static geometry should be 1 value).
        Value::Table(t) => {
            let static_atoms = t.sequence_values::<String>().map(|x| x.unwrap()).collect::<Vec<_>>();
            vec![static_atoms; r_list.len()]
        }
        // If atoms is a lua function which returns a table we have a dynamic geometry.
        Value::Function(f) => {
            let mut out = Vec::with_capacity(r_list.len());
            for &r in &r_list {
                let tbl: Table = f.call(r).unwrap();
                let atoms = tbl.sequence_values::<String>().map(|x| x.unwrap()).collect::<Vec<_>>();
                out.push(atoms);
            }
            out
        }
        _ => {eprintln!("Table or function required by mol.atoms"); std::process::exit(1);}
    };
    let mol = MolOptions {basis, unit, r_list, geoms};

    // SCF table.
    let scf = if let Some(scf_tbl) = scf_tbl {
        let defaults = SCFInfo::default();
        let diis_defaults = DiisOptions::default();
        let diis_tbl: Option<Table> = scf_tbl.get::<_, Option<Table>>("diis").unwrap_or(None);
        let diis = if let Some(diis_tbl) = diis_tbl {
            DiisOptions {
                space: diis_tbl.get("space").unwrap_or(diis_defaults.space),
            }
        } else {
            diis_defaults
        };
        SCFInfo {
            max_cycle: scf_tbl.get("max_cycle").unwrap_or(defaults.max_cycle),
            e_tol: scf_tbl.get("e_tol").unwrap_or(defaults.e_tol),
            diis,
            do_fci: scf_tbl.get("do_fci").unwrap_or(defaults.do_fci),
        }
    } else {
        SCFInfo::default()
    };

    // Write table.
    let write = if let Some(write_tbl) = write_tbl {
        let defaults = WriteOptions::default();
        WriteOptions {
            verbose: write_tbl.get("verbose").unwrap_or(defaults.verbose),
            write_deterministic_coeffs: write_tbl.get("write_deterministic_coeffs").unwrap_or(defaults.write_deterministic_coeffs),
            write_orbitals: write_tbl.get("write_orbitals").unwrap_or(defaults.write_orbitals),
            write_excitation_hist: write_tbl.get("write_excitation_hist").unwrap_or(defaults.write_excitation_hist),
            write_matrices: write_tbl.get("write_matrices").unwrap_or(defaults.write_matrices),
            write_dir: write_tbl.get("write_dir").unwrap_or(defaults.write_dir),
        }
    } else {
        WriteOptions::default()
    };

    // States tables (MOM or metadynamics).
    let states: StateType = match (mom_tbl, meta_tbl) {
        (Some(_), Some(_)) => {eprintln!("Cannot use MOM and SCF metadynamics simultaneously."); std::process::exit(1);}
        (Some(mom_tbl), None) => {
            let mut recipes: Vec<StateRecipe> = Vec::new();
            for st in mom_tbl.sequence_values::<rlua::Table>() {
                let t = st.unwrap();
                recipes.push(read_state_recipe(t));
            }
            StateType::Mom(recipes)
        }
        (None, Some(meta_tbl)) => {
            let defaults = Metadynamics::default();
            let nstates_rhf: usize = meta_tbl.get("nstates_rhf").unwrap_or(defaults.nstates_rhf);
            let nstates_uhf: usize = meta_tbl.get("nstates_uhf").unwrap_or(defaults.nstates_uhf);
            let spinpol: f64 = meta_tbl.get("spinpol").unwrap_or(defaults.spinpol);
            let spatialpol: f64 = meta_tbl.get("spatialpol").unwrap_or(defaults.spatialpol);
            let lambda: f64 = meta_tbl.get("lambda").unwrap_or(defaults.lambda);
            let max_attempts: usize = meta_tbl.get("max_attempts").unwrap_or(defaults.max_attempts);

            let labels_rhf = (1..=nstates_rhf).map(|k| format!("RHF M {}", k)).collect::<Vec<_>>();
            let labels_uhf = (1..=nstates_uhf).map(|k| {
                let pair = k.div_ceil(2);
                let ab = if (k % 2) == 1 { "A" } else { "B" };
                format!("UHF M {} {}", pair, ab)
            }).collect::<Vec<_>>();
            let spatial_patterns_rhf = vec![None; nstates_rhf];
            let spin_patterns_uhf = vec![None; nstates_uhf];

            StateType::Metadynamics(Metadynamics {nstates_rhf,  nstates_uhf, spinpol, spatialpol,  lambda, labels_rhf, labels_uhf,
                                                  spatial_patterns_rhf, spin_patterns_uhf, max_attempts})
        }
        (None, None) => {eprintln!("Must use either MOM or SCF metadynamics to locate SCF solutions"); std::process::exit(1);}
    };

    // Deterministic table.
    let det: Option<DeterministicOptions> = globals.get::<_, Option<Table>>("det").unwrap().map(|det_tbl| {
    let defaults = DeterministicOptions::default();
        DeterministicOptions {
            dynamic_shift: det_tbl.get("dynamic_shift").unwrap_or(defaults.dynamic_shift),
            dynamic_shift_alpha: det_tbl.get("dynamic_shift_alpha").unwrap_or(defaults.dynamic_shift_alpha),
            e_tol: det_tbl.get("e_tol").unwrap_or(defaults.e_tol),
        }
    });

    // QMC table.
    let qmc: Option<QMCOptions> = globals.get::<_, Option<Table>>("qmc").unwrap().map(|qmc_tbl| {
        let defaults = QMCOptions::default();
        let excitation_gen_str: String = qmc_tbl.get("excitation_gen").unwrap_or_else(|_| match defaults.excitation_gen {
            ExcitationGen::Uniform => "uniform".to_string(),
            ExcitationGen::HeatBath => "heat-bath".to_string(),
            ExcitationGen::ApproximateHeatBath => "approximate-heat-bath".to_string(),
        });
        let excitation_gen: ExcitationGen = excitation_gen_str.parse().unwrap_or_else(|msg| {
            eprintln!("{msg}");
            std::process::exit(1);
        });
        QMCOptions {
            initial_population: qmc_tbl.get("initial_population").unwrap_or(defaults.initial_population),
            target_population: qmc_tbl.get("target_population").unwrap_or(defaults.target_population),
            shift_damping: qmc_tbl.get("shift_damping").unwrap_or(defaults.shift_damping),
            shift_update_freq: qmc_tbl.get("shift_update_freq").unwrap_or(defaults.shift_update_freq),
            excitation_gen,
            seed: qmc_tbl.get("seed").unwrap_or(defaults.seed),
        }
    });

    // SNOCI table.
    let snoci: Option<SNOCIOptions> = globals.get::<_, Option<Table>>("snoci").unwrap().map(|snoci_tbl| {
        let defaults = SNOCIOptions::default();
        let gmres_defaults = GMRESOptions::default();
        let gmres_tbl: Option<Table> = snoci_tbl.get::<_, Option<Table>>("gmres").unwrap_or(None);
        let gmres = if let Some(gmres_tbl) = gmres_tbl {
            GMRESOptions {
                max_iter: gmres_tbl.get("max_iter").unwrap_or(gmres_defaults.max_iter),
                res_tol: gmres_tbl.get("res_tol").unwrap_or(gmres_defaults.res_tol),
                metric_tol: gmres_tbl.get("metric_tol").unwrap_or(gmres_defaults.metric_tol),
            }
        } else {
            gmres_defaults
        };
        SNOCIOptions {
            sigma: snoci_tbl.get("sigma").unwrap_or(defaults.sigma),
            tol: snoci_tbl.get("tol").unwrap_or(defaults.tol),
            max_iter: snoci_tbl.get("max_iter").unwrap_or(defaults.max_iter),
            max_add: snoci_tbl.get("max_add").unwrap_or(defaults.max_add),
            max_dim: snoci_tbl.get("max_dim").unwrap_or(defaults.max_dim),
            gmres,
        }
    });

    // Excitation table.
    let excit = if let Some(excit_tbl) = excit_tbl {
        let defaults = ExcitationOptions::default();
        ExcitationOptions {
            orders: excit_tbl.get("orders").unwrap_or(defaults.orders),
        }
    } else {
        ExcitationOptions::default()
    };

    // Propagation table.
    let prop: Option<PropagationOptions> = prop_tbl.map(|prop_tbl| {
        let defaults = PropagationOptions::default();

        let propagator_str: String = prop_tbl.get("propagator").unwrap_or_else(|_| defaults.propagator.as_str().to_string());
        let propagator: Propagator = propagator_str.parse().unwrap_or_else(|msg| {
            eprintln!("{msg}");
            std::process::exit(1);
        });
        PropagationOptions {
            dt: prop_tbl.get("dt").unwrap_or(defaults.dt),
            max_steps: prop_tbl.get("max_steps").unwrap_or(defaults.max_steps),
            propagator,
        }
    });

    // Wick's table.
    let wicks = if let Some(wicks_tbl) = wicks_tbl {
        let defaults = WicksOptions::default();
        let storage = match wicks_tbl.get::<_, Option<String>>("storage").unwrap() {
            Some(s) => match s.to_lowercase().as_str() {
                "ram" => WicksStorage::RAM,
                "disk" => WicksStorage::Disk,
                other => panic!("Unknown wicks.storage value: {other}. Use 'ram' or 'disk'."),
            },
            None => defaults.storage,
        };
        WicksOptions {
            compare: wicks_tbl.get("compare").unwrap_or(defaults.compare),
            enabled: wicks_tbl.get("enabled").unwrap_or(defaults.enabled),
            storage,
            cachedir: wicks_tbl.get("cachedir").unwrap_or(defaults.cachedir),
        }
    } else {
        WicksOptions::default()
    };
    Input {mol, scf, write, states, det, qmc, snoci, excit, prop, wicks}
}

