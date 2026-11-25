mod common;

// External crate imports.
use noci_rs::basis::generate_reference_noci_basis;
use noci_rs::determinant::NOCISpace;
use noci_rs::elements::build_mo_cache;
use noci_rs::noci::calculate_noci_energy;
use serde::Deserialize;
use serial_test::serial;

// Parent/sibling imports.
use self::common::{assert_close, load_test};

/// Expected exact energies for a replacement fixture.
#[derive(Deserialize)]
struct ExpectedReplace {
    /// Expected SCF energies of the NOCI-basis states, in basis order.
    scf_energies: Vec<f64>,
    /// Expected reference-space NOCI energy.
    reference_noci_energy: f64,
}

/// Compare the NOCI-basis state and reference NOCI energies of a replacement fixture with its
/// expected energies.
/// # Arguments:
/// - `fixture`: Name of the test fixture to load.
/// - `system`: System label used in failure messages.
/// # Returns
/// - `()`: Panics if any energy differs from its known good value outside tolerance.
fn check_replace_fixture(
    fixture: &str,
    system: &str,
) {
    let (mut input, ao, expected): (_, _, ExpectedReplace) = load_test(fixture);

    let states = generate_reference_noci_basis(&ao, &mut input, None, None).states;
    let basis: Vec<_> = states.into_iter().filter(|s| s.noci_basis).collect();
    let got_scf: Vec<f64> = basis.iter().map(|s| s.e).collect();

    let space = NOCISpace::from_scf(&basis);
    let mocache = build_mo_cache(&ao, space.parents(), input.scf.d_tol);
    let (got_ref, _coeffs, _dt_hs) =
        calculate_noci_energy(&ao, &input, &space, 1e-12, &mocache, None);

    assert_eq!(got_scf.len(), expected.scf_energies.len());
    for (i, (&x, &y)) in got_scf.iter().zip(expected.scf_energies.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("{system} NOCI-basis state {i}"));
    }
    assert_close(
        got_ref,
        expected.reference_noci_energy,
        1e-8,
        &format!("{system} replacement reference NOCI energy"),
    );
}

/// Test that the H2 3-21G 0.75 Angstrom fixture, where both UHF states collapse to RHF,
/// reproduces the expected energies of RHF and its replacement double excitation.
/// # Panics
/// - Panics if any energy differs from its known good value outside tolerance.
#[test]
#[serial]
fn replace_h2_3_21g_0_75_ang_energies() {
    check_replace_fixture("REPLACE_H2_3-21G_0_75", "H2");
}
