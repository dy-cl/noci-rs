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

/// Rotate the occupied-virtual orbital pairs `(\mathrm{HOMO}-j, \mathrm{LUMO}+j)` of one spin by
/// `\theta_j = \arccos\sigma_j`, so the occupied-orbital overlap with the unrotated orbitals has
/// singular values `\sigma_j`. `\sigma_j = 0` swaps the orbitals exactly.
/// # Arguments:
/// - `c`: MO coefficients with occupied orbitals first.
/// - `nocc`: Number of occupied orbitals.
/// - `sigmas`: Singular value `\sigma_j` of each rotated pair.
/// # Returns
/// - `Array2<f64>`: Rotated MO coefficients.
fn rotate_pairs(
    c: &Array2<f64>,
    nocc: usize,
    sigmas: &[f64],
) -> Array2<f64> {
    let mut c = c.clone();
    for (j, &sigma) in sigmas.iter().enumerate() {
        let (occ, vir) = (nocc - 1 - j, nocc + j);
        let phi_occ = c.column(occ).to_owned();
        let phi_vir = c.column(vir).to_owned();
        if sigma == 0.0 {
            c.column_mut(occ).assign(&phi_vir);
            c.column_mut(vir).assign(&(-&phi_occ));
        } else {
            let theta = sigma.acos();
            c.column_mut(occ)
                .assign(&(&phi_occ * theta.cos() + &phi_vir * theta.sin()));
            c.column_mut(vir)
                .assign(&(&phi_vir * theta.cos() - &phi_occ * theta.sin()));
        }
    }
    c
}

/// Evaluate the Hamiltonian and overlap cross block between excitations of the RHF reference of
/// a fixture and of a second reference, with Wick's theorem and with the naive generalised
/// Slater-Condon rules. The second reference rotates the RHF orbitals with `rotate_pairs`, so
/// its occupied-orbital overlap with the RHF reference has the singular values `sigmas_a` and
/// `sigmas_b`: zero-overlap pairs for `\sigma = 0` and near-zero pairs for small `\sigma`.
/// # Arguments:
/// - `fixture`: Name of the test fixture to load.
/// - `sigmas_a`: Singular values of the rotated alpha-spin orbital pairs.
/// - `sigmas_b`: Singular values of the rotated beta-spin orbital pairs.
/// - `max_rank`: Largest excitation rank, relative to its parent, of the selected determinants.
/// - `alpha_only`: Whether to select only alpha-spin excitations.
/// # Returns
/// - `(f64, [Array2<f64>; 4])`: RHF energy, then `H` and `S` from Wick's theorem and `H` and
///   `S` from the naive rules.
fn run_wicks_naive_fixture(
    fixture: &str,
    sigmas_a: &[f64],
    sigmas_b: &[f64],
    max_rank: u32,
    alpha_only: bool,
) -> (f64, [Array2<f64>; 4]) {
    let (mut input, ao, _expected): (_, _, ExpectedWicks) = load_test(fixture);
    let basis = generate_reference_noci_basis(&ao, &mut input, None, None);
    let rhf = occ_first(&basis.states[0]);

    // Rotate the selected alpha- and beta-spin HOMO/LUMO pairs of the RHF orbitals.
    let mut rotated = rhf.clone();
    rotated.ca = Arc::new(rotate_pairs(
        &rhf.ca,
        rhf.oa.count_ones() as usize,
        sigmas_a,
    ));
    rotated.cb = Arc::new(rotate_pairs(
        &rhf.cb,
        rhf.ob.count_ones() as usize,
        sigmas_b,
    ));
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
    let (e_scf, blocks) =
        run_wicks_naive_fixture("WICKS_BeH2_STO-3G_1_3_C1", &[0.0], &[], 2, false);

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
    let (e_scf, blocks) =
        run_wicks_naive_fixture("WICKS_BeH2_STO-3G_1_3_C1", &[0.0, 0.0], &[], 2, false);

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

/// Test that the BeH2 STO-3G 1.3 Angstrom fixture gives Wick matrix elements that agree with
/// the naive elements for a reference pair with one near-zero alpha orbital pair,
/// `\tilde S_k = 10^{-3}`. Alpha single to triple excitations give contraction determinants up
/// to `L = 6`, where the `\tilde S_k^{-1}` terms of the unsplit intermediates lose precision.
/// # Panics
/// - If the SCF energy differs from the known good value outside tolerance.
/// - If any Wick H or S element differs from the naive element outside tolerance.
#[test]
#[serial]
fn wicks_beh2_sto_3g_1_3_ang_one_near_zero_pair_agree() {
    let (_input, _ao, expected): (_, _, ExpectedWicks) = load_test("WICKS_BeH2_STO-3G_1_3_C1");
    let (e_scf, [h_wicks, s_wicks, h_naive, s_naive]) =
        run_wicks_naive_fixture("WICKS_BeH2_STO-3G_1_3_C1", &[1e-3], &[], 3, true);

    assert_close(e_scf, expected.scf_energies[0], 1e-8, "BeH2 SCF energy");
    assert_wicks_naive_agree(
        (&h_wicks, &s_wicks),
        (&h_naive, &s_naive),
        1e-10,
        "BeH2 one near-zero alpha pair Wicks agreement",
    );
}

/// Test that the BeH2 STO-3G 1.3 Angstrom fixture gives Wick matrix elements that agree with
/// the naive elements for a reference pair with two near-zero alpha orbital pairs, which give
/// four alpha-spin branches.
/// # Panics
/// - If the SCF energy differs from the known good value outside tolerance.
/// - If any Wick H or S element differs from the naive element outside tolerance.
#[test]
#[serial]
fn wicks_beh2_sto_3g_1_3_ang_two_near_zero_pairs_agree() {
    let (_input, _ao, expected): (_, _, ExpectedWicks) = load_test("WICKS_BeH2_STO-3G_1_3_C1");
    let (e_scf, [h_wicks, s_wicks, h_naive, s_naive]) =
        run_wicks_naive_fixture("WICKS_BeH2_STO-3G_1_3_C1", &[1e-3, 2e-3], &[], 2, false);

    assert_close(e_scf, expected.scf_energies[0], 1e-8, "BeH2 SCF energy");
    assert_wicks_naive_agree(
        (&h_wicks, &s_wicks),
        (&h_naive, &s_naive),
        1e-10,
        "BeH2 two near-zero alpha pairs Wicks agreement",
    );
}

/// Test that the BeH2 STO-3G 1.3 Angstrom fixture gives Wick matrix elements that agree with
/// the naive elements for a reference pair with one zero-overlap and one near-zero alpha orbital
/// pair.
/// # Panics
/// - If the SCF energy differs from the known good value outside tolerance.
/// - If any Wick H or S element differs from the naive element outside tolerance.
#[test]
#[serial]
fn wicks_beh2_sto_3g_1_3_ang_zero_and_near_zero_pairs_agree() {
    let (_input, _ao, expected): (_, _, ExpectedWicks) = load_test("WICKS_BeH2_STO-3G_1_3_C1");
    let (e_scf, [h_wicks, s_wicks, h_naive, s_naive]) =
        run_wicks_naive_fixture("WICKS_BeH2_STO-3G_1_3_C1", &[0.0, 1e-3], &[], 2, false);

    assert_close(e_scf, expected.scf_energies[0], 1e-8, "BeH2 SCF energy");
    assert_wicks_naive_agree(
        (&h_wicks, &s_wicks),
        (&h_naive, &s_naive),
        1e-10,
        "BeH2 zero-overlap and near-zero alpha pairs Wicks agreement",
    );
}

/// Test that the BeH2 STO-3G 1.3 Angstrom fixture gives Wick matrix elements that agree with
/// the naive elements for a reference pair with one near-zero orbital pair in each spin, whose
/// different-spin terms sum over all combinations of alpha- and beta-spin branches.
/// # Panics
/// - If the SCF energy differs from the known good value outside tolerance.
/// - If any Wick H or S element differs from the naive element outside tolerance.
#[test]
#[serial]
fn wicks_beh2_sto_3g_1_3_ang_near_zero_pairs_both_spins_agree() {
    let (_input, _ao, expected): (_, _, ExpectedWicks) = load_test("WICKS_BeH2_STO-3G_1_3_C1");
    let (e_scf, [h_wicks, s_wicks, h_naive, s_naive]) =
        run_wicks_naive_fixture("WICKS_BeH2_STO-3G_1_3_C1", &[1e-3], &[2e-3], 2, false);

    assert_close(e_scf, expected.scf_energies[0], 1e-8, "BeH2 SCF energy");
    assert_wicks_naive_agree(
        (&h_wicks, &s_wicks),
        (&h_naive, &s_naive),
        1e-10,
        "BeH2 near-zero alpha and beta pairs Wicks agreement",
    );
}
