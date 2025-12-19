use rlua::{Lua, Value, Table};
use std::fs;

// Choice of propagator.
pub enum Propagator {
    Unshifted,
    Shifted,
}

// Electron spin for excitation input.
pub enum Spin {
    Alpha, 
    Beta,
    Both,
}

// Storage for mol info.
pub struct MolOptions {
    pub basis: String,
    pub unit: String,
    pub r_list: Vec<f64>,
    pub geoms: Vec<Vec<String>>,
}

// Storage for SCF info.
pub struct SCFInfo {
    pub max_cycle: i32,
    pub e_tol: f64,
    pub diis: DiisOptions,
    pub do_fci: bool,
}

// Storage for excitation data.
pub struct Excitation {
    pub spin: Spin, 
    pub occ: i32, 
    pub vir: i32,
}

// Storage for spin biasing specifications.
pub struct SpinBias {
    pub pattern: Vec<i8>,
    pub pol: f64
}

// Storage for spatial biasing specifications.
pub struct SpatialBias {
    pub pattern: Vec<i8>,
    pub pol: f64,
}

// Storage for basis state recipes.
pub struct StateRecipe {
    pub label: String, 
    pub spin_bias: Option<SpinBias>,
    pub spatial_bias: Option<SpatialBias>,
    pub excitation: Option<Excitation>,
    pub noci: bool,
}

// Storage for DIIS options 
pub struct DiisOptions {
    pub space: usize,
}

// Storage for QMC options 
pub struct QMCOptions {
    pub singles: bool, 
    pub doubles: bool,
    pub dt: f64, 
    pub e_tol: f64,
    pub max_steps: usize,
    pub propagator: Propagator,
    pub dynamic_shift: bool,
    pub dynamic_shift_alpha: f64,
}

// Storage for output options
pub struct WriteOptions {
    pub verbose: bool,
    pub write_coeffs: bool,
    pub coeffs_dir: String,
    pub coeffs_filename: String,
}

/// Storage for Input file parameters.
pub struct Input {
    pub mol: MolOptions,
    pub scf: SCFInfo,
    pub write: WriteOptions,
    pub states: Vec<StateRecipe>,
    pub qmc: QMCOptions,
}

/// Read input parameters from lua file and assign to Input object.
/// # Arguments
///     path: str, file path to input file.
pub fn load_input(path: &str) -> Input {

    let src = fs::read_to_string(path).unwrap();
    let lua = Lua::new();

    let ctx = lua; 
    ctx.load(&src).exec().unwrap();
    let globals = ctx.globals();

    // Table headers.
    let mol_tbl: rlua::Table = globals.get("mol").unwrap();
    let scf_tbl: rlua::Table = globals.get("scf").unwrap();
    let write_tbl: rlua::Table = globals.get("write").unwrap();
    let state_tbl: rlua::Table = globals.get("states").unwrap();
    let qmc_tbl: Table = globals.get("qmc").unwrap();
    
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
    let max_cycle: i32 = scf_tbl.get("max_cycle").unwrap();
    let e_tol: f64 = scf_tbl.get("e_tol").unwrap();
    let diis: rlua::Table = scf_tbl.get("diis").unwrap();
    let space: usize = diis.get("space").unwrap();
    let diis = DiisOptions {space};
    let do_fci: bool = scf_tbl.get("do_fci").unwrap();
    let scf = SCFInfo {max_cycle, e_tol, diis, do_fci};

    // Write table.
    let verbose: bool = write_tbl.get("verbose").unwrap();
    let write_coeffs: bool = write_tbl.get("write_coeffs").unwrap();
    let coeffs_dir: String = write_tbl.get("coeffs_dir").unwrap();
    let coeffs_filename: String = write_tbl.get("coeffs_filename").unwrap();
    let write = WriteOptions {verbose, write_coeffs, coeffs_dir, coeffs_filename};

    // States table.
    let mut states: Vec<StateRecipe> = Vec::new();
    for st in state_tbl.sequence_values::<rlua::Table>() {

        let t = st.unwrap();
        let label: String = t.get("label").unwrap();
        let noci: bool = t.get("noci").unwrap_or(true);
        
        let spin_bias = t.get::<_, Option<rlua::Table>>("spin_bias").unwrap_or(None)
                        .map(|sb| {
                            let pol: f64 = sb.get("pol").unwrap();
                            let pat_tbl: rlua::Table = sb.get("pattern").unwrap();
                            let pattern: Vec<i8> = pat_tbl.sequence_values::<i64>().map(|x| x.unwrap()).map(|x| match x {1 => 1, 0 => 0, -1 => -1, 
                                _ => {println!("spin_bias.pattern entries must be -1, 0, or 1");
                                      std::process::exit(1);}
                            }).collect();
        SpinBias {pattern, pol}});
        let spatial_bias = t.get::<_, Option<rlua::Table>>("spatial_bias").unwrap_or(None)
                        .map(|sb| {
                            let pol: f64 = sb.get("pol").unwrap();
                            let pat_tbl: rlua::Table = sb.get("pattern").unwrap();
                            let pattern: Vec<i8> = pat_tbl.sequence_values::<i64>().map(|x| x.unwrap()).map(|x| match x {1 => 1, 0 => 0, -1 => -1, 
                                _ => {println!("spin_bias.pattern entries must be -1, 0, or 1");
                                      std::process::exit(1);}
                            }).collect();
        SpatialBias {pattern, pol}}); 
        let excitation = t.get::<_, Option<rlua::Table>>("excit").unwrap_or(None)
                         .map(|ex| {
                            let s: String = ex.get("spin").unwrap();
                            let spin = match s.as_str() {
                                "alpha" => Spin::Alpha,
                                "beta" => Spin::Beta,
                                "both" => Spin::Both,
                                _ => { eprintln!("Excitation spin must be 'alpha', 'beta', or 'both'"); std::process::exit(1);}
                            };
        Excitation {spin, occ: ex.get("occ").unwrap(), vir: ex.get("vir").unwrap()}});
        states.push(StateRecipe {label, spin_bias, spatial_bias, excitation, noci});
    }
    
    // QMC table
    let singles: bool = qmc_tbl.get("singles").unwrap();
    let doubles: bool = qmc_tbl.get("doubles").unwrap();
    let dt: f64 = qmc_tbl.get("dt").unwrap();
    let qmc_e_tol: f64 = qmc_tbl.get("e_tol").unwrap();
    let max_steps: usize = qmc_tbl.get("max_steps").unwrap();
    let propagator_str: String = qmc_tbl.get("propagator").unwrap();
    let propagator = match propagator_str.as_str() {
        "unshifted" => Propagator::Unshifted,
        "shifted" => Propagator::Shifted,
        _ => {eprintln!("Propagator must be 'unshifted', 'shifted'."); std::process::exit(1);} 
    };
    let dynamic_shift = qmc_tbl.get("dynamic_shift").unwrap();
    let dynamic_shift_alpha = qmc_tbl.get("dynamic_shift_alpha").unwrap();

    let qmc = QMCOptions {singles, doubles, dt, e_tol: qmc_e_tol, max_steps, propagator, dynamic_shift, dynamic_shift_alpha};

    Input {mol, scf, write, states, qmc}
}

