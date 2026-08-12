// noci/factorise/onebody/gpu/contract.rs
//! GPU dense contractions for factorised one-body NOCI operator contractions.

// External crate imports.
use cubecl::prelude::*;
use ndarray::Array1;

// Crate-root imports.
use crate::noci::types::NOCIScalar;

/// Apply alpha-first contraction on the device.
/// Computes `Y^Q += F^alpha D (S^beta)^T
/// + S^alpha D (F^beta+\lambda S^beta)^T`.
/// # Arguments:
/// - `x`: Source determinant vector.
/// - `lambda`: Scalar overlap shift.
/// # Returns
/// - `Array1<T>`: Device-computed contribution after final download.
pub(crate) fn apply_one_body_a_first<T: NOCIScalar>(
    _x: &Array1<T>,
    _lambda: T,
) -> Array1<T> {
    eprintln!("GPU alpha-first one-body contraction is not implemented yet");
    std::process::exit(1);
}

/// Apply beta-first contraction on the device.
/// Computes `Y^Q += S^alpha D (F^beta)^T
/// + (F^alpha+\lambda S^alpha)D(S^beta)^T`.
/// # Arguments:
/// - `x`: Source determinant vector.
/// - `lambda`: Scalar overlap shift.
/// # Returns
/// - `Array1<T>`: Device-computed contribution after final download.
pub(crate) fn apply_one_body_b_first<T: NOCIScalar>(
    _x: &Array1<T>,
    _lambda: T,
) -> Array1<T> {
    eprintln!("GPU beta-first one-body contraction is not implemented yet");
    std::process::exit(1);
}

/// Zero a real device buffer before accumulation.
/// # Arguments:
/// - `out`: Device buffer to zero.
/// - `len`: Number of entries.
/// # Returns
/// - `()`: Writes zero to every entry.
#[cube(launch_unchecked)]
pub(crate) fn zero_f64_kernel(
    out: &mut Array<f64>,
    len: usize,
) {
    if ABSOLUTE_POS < len {
        out[ABSOLUTE_POS] = 0.0;
    }
}

/// Alpha-first stage kernel:
/// `T^F_{\bar a,b} = \sum_a F^\alpha_{\bar a,a}D_{a,b}` and
/// `T^S_{\bar a,b} = \sum_a S^\alpha_{\bar a,a}D_{a,b}`.
/// # Arguments:
/// - `sa`: Row-major alpha overlap factors.
/// - `fa`: Row-major alpha Fock factors.
/// - `x`: Source determinant vector.
/// - `by_beta_offsets`: CSR offsets keyed by source beta component.
/// - `by_beta_det`: CSR determinant IDs.
/// - `by_beta_alpha`: CSR source alpha component IDs.
/// - `tf`: Output Fock first-stage table.
/// - `ts`: Output overlap first-stage table.
/// - `nta`: Target alpha component count.
/// - `nsb`: Source beta component count.
/// - `nsa`: Source alpha component count.
/// - `worker`: MPI worker id.
/// - `nworker`: MPI worker count.
/// # Returns
/// - `()`: Writes `TF` and `TS`.
#[cube(launch_unchecked)]
pub(crate) fn a_first_stage_kernel(
    sa: &Array<f64>,
    fa: &Array<f64>,
    x: &Array<f64>,
    by_beta_offsets: &Array<u32>,
    by_beta_det: &Array<u32>,
    by_beta_alpha: &Array<u32>,
    tf: &mut Array<f64>,
    ts: &mut Array<f64>,
    nta: usize,
    nsb: usize,
    nsa: usize,
    worker: usize,
    nworker: usize,
) {
    if ABSOLUTE_POS >= nta * nsb {
        terminate!();
    }
    let abar = ABSOLUTE_POS / nsb;
    let b = ABSOLUTE_POS - abar * nsb;
    if abar % nworker != worker {
        terminate!();
    }
    let mut vf = 0.0;
    let mut vs = 0.0;
    let start = usize::cast_from(by_beta_offsets[b]);
    let end = usize::cast_from(by_beta_offsets[b + 1]);
    for p in start..end {
        let det = usize::cast_from(by_beta_det[p]);
        let a = usize::cast_from(by_beta_alpha[p]);
        let xe = x[det];
        vf += fa[abar * nsa + a] * xe;
        vs += sa[abar * nsa + a] * xe;
    }
    tf[ABSOLUTE_POS] = vf;
    ts[ABSOLUTE_POS] = vs;
}

/// Alpha-first final kernel:
/// `Y_{\bar a,\bar b}=\sum_b[T^F_{\bar a,b}S^\beta_{\bar b,b}
/// +T^S_{\bar a,b}(F^\beta_{\bar b,b}+\lambda S^\beta_{\bar b,b})]`.
/// # Arguments:
/// - `sb`: Row-major beta overlap factors.
/// - `fb`: Row-major beta Fock factors.
/// - `target_entry_det`: Target determinant IDs.
/// - `target_entry_a`: Target alpha components.
/// - `target_entry_b`: Target beta components.
/// - `tf`: First-stage Fock table.
/// - `ts`: First-stage overlap table.
/// - `y`: Output determinant vector.
/// - `lambda`: Overlap shift.
/// - `nentry`: Number of target entries.
/// - `nsb`: Source beta component count.
/// - `worker`: MPI worker id.
/// - `nworker`: MPI worker count.
/// # Returns
/// - `()`: Accumulates final target determinant values.
#[cube(launch_unchecked)]
pub(crate) fn a_first_final_kernel(
    sb: &Array<f64>,
    fb: &Array<f64>,
    target_entry_det: &Array<u32>,
    target_entry_a: &Array<u32>,
    target_entry_b: &Array<u32>,
    tf: &Array<f64>,
    ts: &Array<f64>,
    y: &mut Array<f64>,
    lambda: f64,
    nentry: usize,
    nsb: usize,
    worker: usize,
    nworker: usize,
) {
    if ABSOLUTE_POS >= nentry {
        terminate!();
    }
    let abar = usize::cast_from(target_entry_a[ABSOLUTE_POS]);
    if abar % nworker != worker {
        terminate!();
    }
    let bbar = usize::cast_from(target_entry_b[ABSOLUTE_POS]);
    let mut value = 0.0;
    for b in 0usize..nsb {
        value += tf[abar * nsb + b] * sb[bbar * nsb + b]
            + ts[abar * nsb + b] * (fb[bbar * nsb + b] + lambda * sb[bbar * nsb + b]);
    }
    let det = usize::cast_from(target_entry_det[ABSOLUTE_POS]);
    y[det] += value;
}

/// Beta-first stage kernel:
/// `U^F_{a,\bar b} = \sum_b D_{a,b}F^\beta_{\bar b,b}` and
/// `U^S_{a,\bar b} = \sum_b D_{a,b}S^\beta_{\bar b,b}`.
/// # Arguments:
/// - `sb`: Row-major beta overlap factors.
/// - `fb`: Row-major beta Fock factors.
/// - `x`: Source determinant vector.
/// - `by_alpha_offsets`: CSR offsets keyed by source alpha component.
/// - `by_alpha_det`: CSR determinant IDs.
/// - `by_alpha_beta`: CSR source beta component IDs.
/// - `uf`: Output Fock first-stage table.
/// - `us`: Output overlap first-stage table.
/// - `ntb`: Target beta component count.
/// - `nsa`: Source alpha component count.
/// - `nsb`: Source beta component count.
/// - `worker`: MPI worker id.
/// - `nworker`: MPI worker count.
/// # Returns
/// - `()`: Writes `UF` and `US`.
#[cube(launch_unchecked)]
pub(crate) fn b_first_stage_kernel(
    sb: &Array<f64>,
    fb: &Array<f64>,
    x: &Array<f64>,
    by_alpha_offsets: &Array<u32>,
    by_alpha_det: &Array<u32>,
    by_alpha_beta: &Array<u32>,
    uf: &mut Array<f64>,
    us: &mut Array<f64>,
    ntb: usize,
    nsa: usize,
    nsb: usize,
    worker: usize,
    nworker: usize,
) {
    if ABSOLUTE_POS >= nsa * ntb {
        terminate!();
    }
    let a = ABSOLUTE_POS / ntb;
    let bbar = ABSOLUTE_POS - a * ntb;
    if bbar % nworker != worker {
        terminate!();
    }
    let mut vf = 0.0;
    let mut vs = 0.0;
    let start = usize::cast_from(by_alpha_offsets[a]);
    let end = usize::cast_from(by_alpha_offsets[a + 1]);
    for p in start..end {
        let det = usize::cast_from(by_alpha_det[p]);
        let b = usize::cast_from(by_alpha_beta[p]);
        let xe = x[det];
        vf += xe * fb[bbar * nsb + b];
        vs += xe * sb[bbar * nsb + b];
    }
    uf[ABSOLUTE_POS] = vf;
    us[ABSOLUTE_POS] = vs;
}

/// Beta-first final kernel:
/// `Y_{\bar a,\bar b}=\sum_a[S^\alpha_{\bar a,a}U^F_{a,\bar b}
/// +(F^\alpha_{\bar a,a}+\lambda S^\alpha_{\bar a,a})U^S_{a,\bar b}]`.
/// # Arguments:
/// - `sa`: Row-major alpha overlap factors.
/// - `fa`: Row-major alpha Fock factors.
/// - `target_entry_det`: Target determinant IDs.
/// - `target_entry_a`: Target alpha components.
/// - `target_entry_b`: Target beta components.
/// - `uf`: First-stage Fock table.
/// - `us`: First-stage overlap table.
/// - `y`: Output determinant vector.
/// - `lambda`: Overlap shift.
/// - `nentry`: Number of target entries.
/// - `nsa`: Source alpha component count.
/// - `ntb`: Target beta component count.
/// - `worker`: MPI worker id.
/// - `nworker`: MPI worker count.
/// # Returns
/// - `()`: Accumulates final target determinant values.
#[cube(launch_unchecked)]
pub(crate) fn b_first_final_kernel(
    sa: &Array<f64>,
    fa: &Array<f64>,
    target_entry_det: &Array<u32>,
    target_entry_a: &Array<u32>,
    target_entry_b: &Array<u32>,
    uf: &Array<f64>,
    us: &Array<f64>,
    y: &mut Array<f64>,
    lambda: f64,
    nentry: usize,
    nsa: usize,
    ntb: usize,
    worker: usize,
    nworker: usize,
) {
    if ABSOLUTE_POS >= nentry {
        terminate!();
    }
    let bbar = usize::cast_from(target_entry_b[ABSOLUTE_POS]);
    if bbar % nworker != worker {
        terminate!();
    }
    let abar = usize::cast_from(target_entry_a[ABSOLUTE_POS]);
    let mut value = 0.0;
    for a in 0usize..nsa {
        value += sa[abar * nsa + a] * uf[a * ntb + bbar]
            + (fa[abar * nsa + a] + lambda * sa[abar * nsa + a]) * us[a * ntb + bbar];
    }
    let det = usize::cast_from(target_entry_det[ABSOLUTE_POS]);
    y[det] += value;
}
