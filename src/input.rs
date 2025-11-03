use rlua::{Lua, Value};
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
    pub atom1: String,
    pub atom2: String,
    pub unit: String,
    pub r_list: Vec<f64>,

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
    let atom1: String = mol.get("atom1").unwrap();
    let atom2: String = mol.get("atom2").unwrap();
    let unit: String = mol.get("unit").unwrap();
    let r_val: Value = mol.get("r").unwrap();
    let mut r_list: Vec<f64> = Vec::new();
    match r_val {
        Value::Number(x) => {r_list.push(x)},
        Value::Table(t) => {
            for item in t.sequence_values::<f64>() {
                let r = item.unwrap();
                r_list.push(r);
            }
        },
        _ => {eprintln!("Input: r, must be number or list."); std::process::exit(1);},

    }

    // Write table. 
    let verbose: bool = write.get("verbose").unwrap();

    Input {max_cycle, e_tol, err_tol, pol,
           basis, atom1, atom2, r_list, unit, verbose}
}

