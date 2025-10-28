use ndarray::{Array2};
use crate::{AoData, SCFState};
use crate::scf::scf_cycle;

pub fn generate_scf_state(ao: &AoData, max_cycle: i32, tol: f64) -> SCFState {

    let nocc_a = ao.nelec[0];
    let nocc_b = ao.nelec[1];
    
    // Density matrix ansatz provided by PySCF is RHF and therefore we assign
    // both spin density matrices da and db to 1/2 dm
    let da0: Array2<f64> = ao.dm.clone() * 0.5;
    let db0: Array2<f64> = ao.dm.clone() * 0.5;

    let (e, ca, cb) = scf_cycle(&da0, &db0, &ao.h, &ao.eri, &ao.s, ao.enuc,
                                max_cycle, tol, nocc_a, nocc_b,);
    
    return SCFState{e, ca, cb};
}
