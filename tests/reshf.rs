#![cfg(feature = "reference-optimisation")]
// tests/reshf.rs

// The ResHF tests never initialise MPI, so `common::mpi_universe` is unused here.
#[allow(dead_code)]
mod common;

// Standard library imports.
use std::sync::Arc;

// External crate imports.
use ndarray::Array2;
use noci_rs::basis::generate_reference_noci_basis;
use noci_rs::determinant::{NOCIIndex, NOCISpace};
use noci_rs::elements::NOCIData;
use noci_rs::input::{Input, SCFExcitation, Spin};
use noci_rs::noci::build_noci_hs;
use noci_rs::noci::optimised::reshf_expansion;
use noci_rs::scf::{DensityMode, density, energy, fock, occ_first};
use noci_rs::{AoData, SCFState};
use serde::Deserialize;
use serial_test::serial;

// Parent/sibling imports.
use self::common::{assert_close, load_test};

/// Tolerance up to which a number is considered zero, as in the reference NOCI tests.
const TOL: f64 = 1e-12;

/// Overlap `\langle \Phi_{\text{RHF}} | \Phi_{+-} \rangle` printed by yucca for a 40 degree rotation.
const YUCCARHFROT: f64 = 0.58682409;

/// Overlap `\langle \Phi_{+-} | \Phi_{-+} \rangle` printed by yucca for a 40 degree rotation.
const YUCCAROTROT: f64 = 0.03015369;

/// Expected exact energies for a reference NOCI fixture.
#[derive(Deserialize)]
struct ExpectedReferenceNoci {
    /// Expected reference-space NOCI energy.
    reference_noci_energy: f64,
}

/// Run the MOM recipes of a reference NOCI fixture and keep the states marked for NOCI.
/// # Arguments:
/// - `fixture`: Name of the test fixture to load.
/// # Returns
/// - `(Input, AoData, Vec<SCFState>, f64)`: Input, AO data, NOCI states and the expected
///   reference NOCI energy.
fn reference_states(fixture: &str) -> (Input, AoData, Vec<SCFState>, f64) {
    let (mut input, ao, expected): (_, _, ExpectedReferenceNoci) = load_test(fixture);
    let states = generate_reference_noci_basis(&ao, &mut input, None, None).states;
    let states: Vec<SCFState> = states.into_iter().filter(|s| s.noci_basis).collect();
    (input, ao, states, expected.reference_noci_energy)
}

/// Rotate occupied orbital `i` into virtual orbital `a` of one spin by angle `\theta`:
/// `\phi_i' = \cos\theta\,\phi_i + \sin\theta\,\phi_a` and
/// `\phi_a' = \cos\theta\,\phi_a - \sin\theta\,\phi_i`, then rebuild the density and energy.
/// `i` and `a` are offsets from the HOMO and LUMO, as in MOM excitations: `occ = -1` is the
/// HOMO and `vir = 0` the LUMO. This makes yucca's `rotate` starting determinants.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `st`: Determinant to rotate.
/// - `ex`: Spin and orbital offsets; `Spin::Both` rotates both spins by the same angle.
/// - `theta_deg`: Rotation angle in degrees.
/// - `label`: Label of the new determinant.
/// # Returns
/// - `SCFState`: The rotated determinant with density and energy rebuilt.
/// # Panics
/// - Panics if `occ` is not an occupied offset or `vir` is not a virtual offset.
fn rotate(
    ao: &AoData,
    st: &SCFState,
    ex: &SCFExcitation,
    theta_deg: f64,
    label: &str,
) -> SCFState {
    // Occupied orbitals first, so offsets from the HOMO and LUMO become column indices.
    let base = occ_first(st);
    let na = base.oa.count_ones() as usize;
    let nb = base.ob.count_ones() as usize;
    let nmo = base.ca.ncols();
    let (sin, cos) = theta_deg.to_radians().sin_cos();

    // Mix columns `i` and `a` of one spin's coefficients after checking both indices.
    let mix = |c: &Array2<f64>, nocc: usize| {
        assert!(
            ex.occ < 0 && -ex.occ <= nocc as i32,
            "occ = {} is not an occupied offset",
            ex.occ
        );
        assert!(
            ex.vir >= 0 && nocc + (ex.vir as usize) < nmo,
            "vir = {} is not a virtual offset",
            ex.vir
        );
        let i = (nocc as i32 + ex.occ) as usize;
        let a = nocc + ex.vir as usize;
        let ci = c.column(i).to_owned();
        let ca = c.column(a).to_owned();
        let mut out = c.clone();
        out.column_mut(i).assign(&(&ci * cos + &ca * sin));
        out.column_mut(a).assign(&(&ca * cos - &ci * sin));
        out
    };
    let ca = match ex.spin {
        Spin::Alpha | Spin::Both => mix(&base.ca, na),
        Spin::Beta => (*base.ca).clone(),
    };
    let cb = match ex.spin {
        Spin::Beta | Spin::Both => mix(&base.cb, nb),
        Spin::Alpha => (*base.cb).clone(),
    };

    // Rebuild `P^\sigma`, `F^\sigma` and `E` with the existing SCF kernels.
    let da = density(&ca, na, DensityMode::Hermitian);
    let db = density(&cb, nb, DensityMode::Hermitian);
    let (fa, fb) = fock(&ao.h, &ao.eri_coul, &da, &db);
    let e = energy(&ao.h, ao.enuc, &da, &db, &fa, &fb);

    SCFState {
        e,
        oa: base.oa,
        ob: base.ob,
        ca: Arc::new(ca),
        cb: Arc::new(cb),
        da: Arc::new(da),
        db: Arc::new(db),
        label: label.to_string(),
        noci_basis: true,
    }
}

/// Make yucca's pair of spin-broken guesses from one determinant: alpha rotated by
/// `+\theta` and beta by `-\theta`, then its mirror image, both for the HOMO-LUMO pair.
/// # Arguments:
/// - `ao`: Contains AO integrals and other system data.
/// - `st`: Determinant to rotate, usually RHF.
/// - `theta_deg`: Rotation angle in degrees.
/// # Returns
/// - `(SCFState, SCFState)`: The `(+\theta, -\theta)` and `(-\theta, +\theta)` determinants.
fn spin_broken_pair(
    ao: &AoData,
    st: &SCFState,
    theta_deg: f64,
) -> (SCFState, SCFState) {
    let alpha = SCFExcitation {
        spin: Spin::Alpha,
        occ: -1,
        vir: 0,
    };
    let beta = SCFExcitation {
        spin: Spin::Beta,
        occ: -1,
        vir: 0,
    };

    let plus = rotate(ao, st, &alpha, theta_deg, "rotate (+, -)");
    let plus = rotate(ao, &plus, &beta, -theta_deg, "rotate (+, -)");
    let minus = rotate(ao, st, &alpha, -theta_deg, "rotate (-, +)");
    let minus = rotate(ao, &minus, &beta, theta_deg, "rotate (-, +)");
    (plus, minus)
}

/// Find the RHF determinant among a fixture's states by its label.
/// # Arguments:
/// - `states`: SCF states of the fixture.
/// # Returns
/// - `SCFState`: A copy of the state whose label starts with `RHF`.
/// # Panics
/// - Panics if no state is labelled RHF.
fn rhf_state(states: &[SCFState]) -> SCFState {
    states
        .iter()
        .find(|st| st.label.starts_with("RHF"))
        .expect("fixture has an RHF state")
        .clone()
}

/// Largest absolute difference between two matrices.
/// # Arguments:
/// - `x`: First matrix.
/// - `y`: Second matrix.
/// # Returns
/// - `f64`: `\max_{pq} |x_{pq} - y_{pq}|`.
fn max_diff(
    x: &Array2<f64>,
    y: &Array2<f64>,
) -> f64 {
    (x - y).iter().fold(0.0, |m, v| m.max(v.abs()))
}

/// The lowest ResHF expansion energy of the MOM states equals the reference NOCI energy, each
/// state overlaps itself perfectly, and `H_{AA}` is that state's own SCF energy.
#[test]
#[serial]
fn reshf_expansion_matches_reference_noci_h2_sto_3g_1_5_ang() {
    let (_input, ao, states, want) = reference_states("REF_NOCI_H2_STO-3G_1_5");
    let got = reshf_expansion(&ao, &states, &[1.0], TOL);

    for (a, st) in states.iter().enumerate() {
        assert_close(got.s[(a, a)], 1.0, 1e-10, &format!("S_AA for {}", st.label));
        assert_close(got.h[(a, a)], st.e, 1e-8, &format!("H_AA for {}", st.label));
    }
    assert_close(got.energies[0], want, 1e-8, "H2 ResHF expansion energy");
}

/// The new CI step builds exactly the same `H` and `S` as the existing `build_noci_hs`
/// (this fixture has Wick's intermediates switched off).
#[test]
#[serial]
fn reshf_expansion_agrees_with_build_noci_hs() {
    let (input, ao, states, _want) = reference_states("REF_NOCI_H2_STO-3G_1_5");
    let got = reshf_expansion(&ao, &states, &[1.0], TOL);

    let dets: Vec<SCFState> = states.iter().map(occ_first).collect();
    let space = NOCISpace::from_scf(&dets);
    let data = NOCIData::new(&ao, &space, &input, TOL, None);
    let idx: Vec<NOCIIndex> = (0..space.len()).map(NOCIIndex).collect();
    let (h, s, _dt) = build_noci_hs(&data, &idx, &idx, true);

    assert_close(max_diff(&got.h, &h), 0.0, 1e-12, "H against build_noci_hs");
    assert_close(max_diff(&got.s, &s), 0.0, 1e-12, "S against build_noci_hs");
}

/// A rotation keeps each spin's orbitals orthonormal, `C^\dagger S C = 1`.
#[test]
#[serial]
fn rotation_keeps_orbitals_orthonormal() {
    let (_input, ao, states, _want) = reference_states("REF_NOCI_H2_STO-3G_1_5");
    let (plus, _minus) = spin_broken_pair(&ao, &rhf_state(&states), 40.0);

    for c in [&plus.ca, &plus.cb] {
        let ctsc = c.t().dot(&ao.s).dot(&**c);
        let eye = Array2::<f64>::eye(ctsc.nrows());
        assert_close(
            max_diff(&ctsc, &eye),
            0.0,
            1e-10,
            "C^T S C against identity",
        );
    }
}

/// Rotated guesses reproduce yucca's overlaps at 40 degrees: `\cos^2 40^\circ` between RHF and
/// either rotated determinant and `\cos^2 80^\circ` between the two rotated determinants.
#[test]
#[serial]
fn rotated_guess_overlaps_match_yucca() {
    let (_input, ao, states, _want) = reference_states("REF_NOCI_H2_STO-3G_1_5");
    let rhf = rhf_state(&states);
    let (plus, minus) = spin_broken_pair(&ao, &rhf, 40.0);
    let got = reshf_expansion(&ao, &[rhf, plus, minus], &[1.0], TOL);

    let c40 = 40.0_f64.to_radians().cos();
    let c80 = 80.0_f64.to_radians().cos();
    assert_close(
        got.s[(0, 1)],
        c40 * c40,
        1e-10,
        "S(RHF, +-) against cos^2 40",
    );
    assert_close(
        got.s[(1, 2)],
        c80 * c80,
        1e-10,
        "S(+-, -+) against cos^2 80",
    );
    assert_close(got.s[(0, 1)], YUCCARHFROT, 1e-8, "S(RHF, +-) against yucca");
    assert_close(got.s[(1, 2)], YUCCAROTROT, 1e-8, "S(+-, -+) against yucca");
}

/// In minimal-basis H2, RHF and the two rotated determinants span the exact ground state for
/// any non-zero angle, so the lowest energy must equal the reference NOCI energy at every angle.
#[test]
#[serial]
fn rotated_guess_energy_is_independent_of_angle() {
    let (_input, ao, states, want) = reference_states("REF_NOCI_H2_STO-3G_1_5");
    let rhf = rhf_state(&states);

    for theta in [10.0, 40.0, 70.0] {
        let (plus, minus) = spin_broken_pair(&ao, &rhf, theta);
        let got = reshf_expansion(&ao, &[rhf.clone(), plus, minus], &[1.0], TOL);
        assert_close(
            got.energies[0],
            want,
            1e-8,
            &format!("energy at {theta} degrees"),
        );
    }
}
