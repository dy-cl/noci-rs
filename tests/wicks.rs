mod common;

// Standard library imports.
use std::sync::Arc;

// External crate imports.
use ndarray::Array2;
use noci_rs::basis::generate_reference_noci_basis;
use noci_rs::determinant::{NOCIIndex, NOCISpace};
use noci_rs::elements::{NOCIData, build_mo_cache, build_wicks_shared};
use noci_rs::noci::build_noci_hs;
use noci_rs::scf::occ_first;
use serde::Deserialize;
use serial_test::serial;

// Parent/sibling imports.
use self::common::{assert_close, assert_wicks_naive_agree, load_test, mpi_universe};

/// Expected exact energies for a Wick matrix-element fixture.
#[derive(Deserialize)]
struct ExpectedWicks {
    /// Expected SCF state energies.
    scf_energies: Vec<f64>,
}

/// Evaluate the Hamiltonian and overlap cross block between excitations of the RHF reference of
/// a fixture and of a second reference, with Wick's theorem and with the naive generalised
/// Slater-Condon rules. The second reference rotates the alpha pairs
/// `(\mathrm{HOMO}-j, \mathrm{LUMO}+j)`, `j < nrot`, by `\theta = \arccos\sigma`, so the
/// occupied-orbital overlap has `nrot` singular values `\sigma`. `\sigma = 0` swaps the orbitals
/// exactly, giving zero-overlap pairs.
/// # Arguments:
/// - `fixture`: Name of the test fixture to load.
/// - `nrot`: Number of rotated alpha orbital pairs.
/// - `sigma`: Occupied-orbital singular value of each rotated pair.
/// - `max_rank`: Largest excitation rank, relative to its parent, of the selected determinants.
/// - `alpha_only`: Whether to select only alpha-spin excitations.
/// # Returns
/// - `(f64, [Array2<f64>; 4])`: RHF energy, then `H` and `S` from Wick's theorem and `H` and
///   `S` from the naive rules.
fn run_wicks_naive_fixture(
    fixture: &str,
    nrot: usize,
    sigma: f64,
    max_rank: u32,
    alpha_only: bool,
) -> (f64, [Array2<f64>; 4]) {
    let (mut input, ao, _expected): (_, _, ExpectedWicks) = load_test(fixture);
    let basis = generate_reference_noci_basis(&ao, &mut input, None, None);
    let rhf = occ_first(&basis.states[0]);
    let nocc = rhf.oa.count_ones() as usize;

    // Rotate the selected alpha HOMO/LUMO pairs of the RHF orbitals.
    let mut ca = (*rhf.ca).clone();
    for j in 0..nrot {
        let (occ, vir) = (nocc - 1 - j, nocc + j);
        let phi_occ = ca.column(occ).to_owned();
        let phi_vir = ca.column(vir).to_owned();
        if sigma == 0.0 {
            ca.column_mut(occ).assign(&phi_vir);
            ca.column_mut(vir).assign(&(-&phi_occ));
        } else {
            let theta = sigma.acos();
            ca.column_mut(occ)
                .assign(&(&phi_occ * theta.cos() + &phi_vir * theta.sin()));
            ca.column_mut(vir)
                .assign(&(&phi_vir * theta.cos() - &phi_occ * theta.sin()));
        }
    }
    let mut rotated = rhf.clone();
    rotated.ca = Arc::new(ca);
    rotated.label = "rotated".to_string();

    // Build the excited space and the intermediates for every ordered reference pair.
    let reference_space = NOCISpace::from_scf(&[rhf.clone(), rotated]);
    let sources = (0..reference_space.len())
        .map(NOCIIndex)
        .collect::<Vec<_>>();
    let space = reference_space.excited_from(&sources, &input, true);
    let mocache = build_mo_cache(&ao, space.parents(), input.scf.d_tol);
    let (_mpi_lock, universe) = mpi_universe();
    let world = universe.world();
    let wicks = build_wicks_shared::<f64>(&world, &ao, space.parents(), 1e-12, &input);

    // Both references share the occupied-first occupation, so the excitation rank of a
    // determinant is half the number of occupation bits that differ from it.
    let select = |parent: usize| -> Vec<NOCIIndex> {
        (0..space.len())
            .map(NOCIIndex)
            .filter(|&i| {
                let (oa, ob) = space.occupations(i);
                let rank = ((oa ^ rhf.oa).count_ones() + (ob ^ rhf.ob).count_ones()) / 2;
                space.state(i).parent == parent && rank <= max_rank && (!alpha_only || ob == rhf.ob)
            })
            .collect()
    };
    let left = select(0);
    let right = select(1);

    // Evaluate the cross block with Wick's theorem and with the naive rules.
    input.wicks.enabled = true;
    let data = NOCIData::new(&ao, &space, &input, 1e-12, Some(wicks.view())).withmocache(&mocache);
    let (h_wicks, s_wicks, _) = build_noci_hs(&data, &left, &right, false);
    input.wicks.enabled = false;
    let data = NOCIData::new(&ao, &space, &input, 1e-12, None).withmocache(&mocache);
    let (h_naive, s_naive, _) = build_noci_hs(&data, &left, &right, false);

    (rhf.e, [h_wicks, s_wicks, h_naive, s_naive])
}

/// Test that the BeH2 STO-3G 1.3 Angstrom fixture gives Wick matrix elements that agree with
/// the naive elements for a reference pair with one zero-overlap alpha orbital pair
/// (`m_\alpha = 1`). The singles and doubles cross block contains elements with `S = 0` but
/// `H \neq 0`.
/// # Panics
/// - If the SCF energy differs from the known good value outside tolerance.
/// - If the cross block contains no singular overlap elements.
/// - If any Wick H or S element differs from the naive element outside tolerance.
#[test]
#[serial]
fn wicks_beh2_sto_3g_1_3_ang_one_zero_overlap_pair_agree() {
    let (_input, _ao, expected): (_, _, ExpectedWicks) = load_test("WICKS_BeH2_STO-3G_1_3_C1");
    let (e_scf, blocks) = run_wicks_naive_fixture("WICKS_BeH2_STO-3G_1_3_C1", 1, 0.0, 2, false);

    assert_close(e_scf, expected.scf_energies[0], 1e-8, "BeH2 SCF energy");
    let [h_wicks, s_wicks, h_naive, s_naive] = &blocks;
    assert!(
        s_naive.iter().any(|x| x.abs() < 1e-12),
        "cross block contains no singular overlaps"
    );
    assert_wicks_naive_agree(
        (h_wicks, s_wicks),
        (h_naive, s_naive),
        1e-10,
        "BeH2 m_alpha = 1 Wicks agreement",
    );
}

/// Test that the BeH2 STO-3G 1.3 Angstrom fixture gives Wick matrix elements that agree with
/// the naive elements for a reference pair with two zero-overlap alpha orbital pairs
/// (`m_\alpha = 2`).
/// # Panics
/// - If the SCF energy differs from the known good value outside tolerance.
/// - If the cross block contains no singular overlap elements.
/// - If any Wick H or S element differs from the naive element outside tolerance.
#[test]
#[serial]
fn wicks_beh2_sto_3g_1_3_ang_two_zero_overlap_pairs_agree() {
    let (_input, _ao, expected): (_, _, ExpectedWicks) = load_test("WICKS_BeH2_STO-3G_1_3_C1");
    let (e_scf, blocks) = run_wicks_naive_fixture("WICKS_BeH2_STO-3G_1_3_C1", 2, 0.0, 2, false);

    assert_close(e_scf, expected.scf_energies[0], 1e-8, "BeH2 SCF energy");
    let [h_wicks, s_wicks, h_naive, s_naive] = &blocks;
    assert!(
        s_naive.iter().any(|x| x.abs() < 1e-12),
        "cross block contains no singular overlaps"
    );
    assert_wicks_naive_agree(
        (h_wicks, s_wicks),
        (h_naive, s_naive),
        1e-10,
        "BeH2 m_alpha = 2 Wicks agreement",
    );
}
