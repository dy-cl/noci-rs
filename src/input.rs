use rlua::{Lua, Value, Table};
use std::fs;

// Electron spin.
pub enum Spin {Alpha, Beta}

// Storage for excitation data.
pub struct Excitation {
    pub spin: Spin, 
    pub occ: i32, 
    pub vir: i32,
}

// Storage for spin biasing specifications.
pub struct SpinBias {
    pub pattern: String, 
    pub pol: f64
}

// Storage for basis state recipes.
pub struct StateRecipe {
    pub label: String, 
    pub spin_bias: Option<SpinBias>,
    pub excitation: Option<Excitation>,
    pub noci: bool,
}

// Storage for DIIS options 
pub struct DiisOptions {
    pub space: usize,
}

// Storage for QMC options 
pub struct QMCOptions {
    pub qmc_singles: bool, 
    pub qmc_doubles: bool,
    pub dt: f64, 
    pub qmc_e_tol: f64,
    pub max_steps: usize,
}

/// Storage for Input file parameters.
pub struct Input {
    // SCF table.
    pub max_cycle: i32,
    pub e_tol: f64,
    pub diis: DiisOptions,
    pub do_fci: bool,
    
    // Mol table.
    pub basis: String,
    pub unit: String,
    pub r_list: Vec<f64>,
    pub geoms: Vec<Vec<String>>,

    // Write table. 
    pub verbose: bool,

    // States table.
    pub states: Vec<StateRecipe>,

    // QMC table
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
    let scf: rlua::Table = globals.get("scf").unwrap();
    let mol: rlua::Table = globals.get("mol").unwrap();
    let write: rlua::Table = globals.get("write").unwrap();
    
    // SCF table.
    let max_cycle: i32 = scf.get("max_cycle").unwrap();
    let e_tol: f64 = scf.get("e_tol").unwrap();
    let diis: rlua::Table = scf.get("diis").unwrap();
    let space: usize = diis.get("space").unwrap();
    let diis = DiisOptions {space};
    let do_fci: bool = scf.get("do_fci").unwrap();
    
    // Mol table.
    let basis: String = mol.get("basis").unwrap();
    let unit: String = mol.get("unit").unwrap();
    // Allow mol.r to be a number or table. 
    let r_val: Value = mol.get("r").unwrap();
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
    let atoms_val: Value = mol.get("atoms").unwrap();
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
        _ => { eprintln!("Table or function required by mol.atoms"); std::process::exit(1); }
    };

    // Write table. 
    let verbose: bool = write.get("verbose").unwrap();

    // States table.
    let state_tbl: rlua::Table = globals.get("states").unwrap();
    let mut states: Vec<StateRecipe> = Vec::new();
    for st in state_tbl.sequence_values::<rlua::Table>() {

        let t = st.unwrap();
        let label: String = t.get("label").unwrap();
        let noci: bool = t.get("noci").unwrap_or(true);

        let spin_bias = t.get::<_, Option<rlua::Table>>("spin_bias").unwrap_or(None)
                        .map(|sb| SpinBias 
                        {pattern: sb.get("pattern").unwrap(),pol: sb.get("pol").unwrap()});

        let excitation = t.get::<_, Option<rlua::Table>>("excit").unwrap_or(None)
                         .map(|ex| {
                            let s: String = ex.get("spin").unwrap();
                            let spin = if s.to_ascii_lowercase()
                                          .starts_with('a') {Spin::Alpha} else {Spin::Beta};
        Excitation {spin, occ: ex.get("occ").unwrap(), vir: ex.get("vir").unwrap()}});
        states.push(StateRecipe {label, spin_bias, excitation, noci});
    }
    
    // QMC table
    let qmc_tbl: Table = globals.get("qmc").unwrap();
    let qmc_singles: bool = qmc_tbl.get("singles").unwrap();
    let qmc_doubles: bool = qmc_tbl.get("doubles").unwrap();
    let dt: f64 = qmc_tbl.get("dt").unwrap();
    let qmc_e_tol: f64 = qmc_tbl.get("e_tol").unwrap();
    let max_steps: usize = qmc_tbl.get("max_steps").unwrap();
    let qmc = QMCOptions {qmc_singles, qmc_doubles, dt, qmc_e_tol, max_steps};

    Input {max_cycle, e_tol, diis, do_fci, basis, r_list, geoms, unit, verbose, states, qmc,}
}

