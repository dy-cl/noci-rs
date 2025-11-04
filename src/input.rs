use rlua::{Lua, Value, Table};
use std::fs;

/// Storage for Input file parameters.
pub struct Input {
    // SCF table.
    pub max_cycle: i32,
    pub e_tol: f64,
    pub err_tol: f64,
    pub pol: f64,
    
    // Mol table.
    pub basis: String,
    pub unit: String,
    pub r_list: Vec<f64>,
    pub geoms: Vec<Vec<String>>,

    // Write table. 
    pub verbose: bool,
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
    let err_tol: f64 = scf.get("err_tol").unwrap();
    let pol: f64 = scf.get("pol").unwrap();
    
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

    Input {max_cycle, e_tol, err_tol, pol,
           basis, r_list, geoms, unit, verbose}
}

