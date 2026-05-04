// scf/bias.rs

use std::sync::Arc;
use ndarray::Array2;
use crate::{AoData, Excitation, ExcitationSpin, SCFState};
use crate::basis::electron_distance;

/// Construct the SCF metadynamics bias term.
/// # Arguments
/// - `da`: Alpha-spin density matrix.
/// - `db`: Beta-spin density matrix.
/// - `ao`: AO integrals and metadata.
/// - `biases`: Previously found SCF states.
/// - `lambda`: Bias strength.
/// # Returns
/// - `(Array2<f64>, Array2<f64>)`: Alpha- and beta-spin bias matrices.
pub(crate) fn metadynamics_bias(da: &Array2<f64>, db: &Array2<f64>, ao: &AoData, biases: &[SCFState], lambda: f64) -> (Array2<f64>, Array2<f64>) {
    let nbf = da.nrows();
    let mut ba = Array2::<f64>::zeros((nbf, nbf));
    let mut bb = Array2::<f64>::zeros((nbf, nbf));

    let tmpscf = SCFState {
        e: 0.0,
        oa: 0u128,
        ob: 0u128,
        pha: 1.0,
        phb: 1.0,
        ca: Arc::new(Array2::zeros((nbf, nbf))),
        cb: Arc::new(Array2::zeros((nbf, nbf))),
        da: Arc::new(da.clone()),
        db: Arc::new(db.clone()),
        label: String::new(),
        noci_basis: false,
        parent: 0,
        excitation: Excitation {alpha: ExcitationSpin {holes: vec![], parts: vec![]}, beta: ExcitationSpin {holes: vec![], parts: vec![]}},
    };

    for bias in biases {
        let d2 = electron_distance(&tmpscf, bias, &ao.s);
        let nlambda = (bias.da.dot(&ao.s)).diag().sum() + (bias.db.dot(&ao.s)).diag().sum();
        let c = nlambda * lambda * (-lambda * d2).exp();
        ba = ba + &*bias.da * c;
        bb = bb + &*bias.db * c;
    }

    (ba, bb)
}
