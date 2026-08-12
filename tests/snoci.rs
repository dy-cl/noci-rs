mod common;

// External crate imports.
use noci_rs::PostSCFData;
use noci_rs::basis::{generate_excited_basis, generate_reference_noci_basis};
use noci_rs::input::SNOCIBackend;
use noci_rs::noci::{
    NOCIData, build_mo_cache, build_noci_hs, build_wicks_shared, calculate_noci_energy,
};
use noci_rs::snoci::snoci_step;
use serde::Deserialize;
use serial_test::serial;

// Parent/sibling imports.
use self::common::{assert_close, load_test, mpi_universe};

/// Expected exact energies for a selected NOCI fixture.
#[derive(Deserialize)]
struct ExpectedSNOCI {
    /// Expected SCF state energies.
    scf_energies: Vec<f64>,
    /// Expected reference-space NOCI energy.
    reference_noci_energy: f64,
    /// Expected selected NOCI energy.
    snoci_energy: f64,
}

/// Energies and GMRES diagnostics from one backend-specific SNOCI run.
struct SNOCIGpuComparisonRun {
    /// Sorted SCF state energies.
    scf_energies: Vec<f64>,
    /// Reference-space NOCI energy.
    reference_noci_energy: f64,
    /// Selected NOCI energy after the SNOCI step.
    snoci_energy: f64,
    /// Last NOCI-PT2 correction energy used for selection.
    ept2: f64,
    /// Final GMRES residual.
    gmres_residual: f64,
    /// Final GMRES iteration count.
    gmres_iterations: usize,
}

/// Run SCF, reference NOCI and selected NOCI and compare energies with known good energies.
/// # Arguments:
/// - `fixture`: Name of the test fixture to load.
/// # Returns
/// - `(Vec<f64>, f64, f64)`: Sorted SCF state energies, the reference NOCI energy and the
///   selected NOCI energy.
fn run_snoci_fixture(fixture: &str) -> (Vec<f64>, f64, f64) {
    let (mut input, ao, _expected): (_, _, ExpectedSNOCI) = load_test(fixture);

    let basis = generate_reference_noci_basis(&ao, &mut input, None, None);
    let states = basis.states;

    let mut scf_energies: Vec<f64> = states.iter().map(|s| s.e).collect();
    scf_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut noci_reference_basis: Vec<_> =
        states.iter().filter(|s| s.noci_basis).cloned().collect();
    for (i, st) in noci_reference_basis.iter_mut().enumerate() {
        st.parent = i;
    }

    let mocache = build_mo_cache(&ao, &noci_reference_basis, input.scf.d_tol);

    let (e_ref, _c0, _dt_hs_ref) =
        calculate_noci_energy(&ao, &input, &noci_reference_basis, 1e-12, &mocache, None);

    let post = PostSCFData {
        ao: &ao,
        states: &states,
        noci_reference_basis: &noci_reference_basis,
        mocache: &mocache,
        tol: 1e-12,
    };

    let (_mpi_lock, universe) = mpi_universe();
    let world = universe.world();

    let result = snoci_step(&post, &noci_reference_basis, &input, None, &world);

    assert!(
        !result.pt2.is_empty(),
        "SNOCI did not produce a NOCI-PT2 result"
    );
    for pt2 in &result.pt2 {
        assert!(
            pt2.gmres_converged,
            "SNOCI GMRES failed to converge: residual {}",
            pt2.gmres_residual,
        );
    }

    (scf_energies, e_ref, result.ecurrent)
}

/// Run SCF, reference NOCI and selected NOCI with Wick's intermediates and compare energies with known good energies.
/// # Arguments:
/// - `fixture`: Name of the test fixture to load.
/// # Returns
/// - `(Vec<f64>, f64, f64)`: Sorted SCF state energies, the reference NOCI energy and the
///   selected NOCI energy.
fn run_snoci_fixture_wicks(fixture: &str) -> (Vec<f64>, f64, f64) {
    let (mut input, ao, _expected): (_, _, ExpectedSNOCI) = load_test(fixture);

    let basis = generate_reference_noci_basis(&ao, &mut input, None, None);
    let states = basis.states;

    let mut scf_energies: Vec<f64> = states.iter().map(|s| s.e).collect();
    scf_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut noci_reference_basis: Vec<_> =
        states.iter().filter(|s| s.noci_basis).cloned().collect();
    for (i, st) in noci_reference_basis.iter_mut().enumerate() {
        st.parent = i;
    }

    let mocache = build_mo_cache(&ao, &noci_reference_basis, input.scf.d_tol);

    let (_mpi_lock, universe) = mpi_universe();
    let world = universe.world();

    let mut wicks = build_wicks_shared::<f64>(&world, &ao, &noci_reference_basis, 1e-12, &input);
    let (e_ref, _c0, _dt_hs_ref) = {
        let wicks_view = wicks.view();
        calculate_noci_energy(
            &ao,
            &input,
            &noci_reference_basis,
            1e-12,
            &mocache,
            Some(wicks_view),
        )
    };

    let post = PostSCFData {
        ao: &ao,
        states: &states,
        noci_reference_basis: &noci_reference_basis,
        mocache: &mocache,
        tol: 1e-12,
    };

    let result = snoci_step(
        &post,
        &noci_reference_basis,
        &input,
        Some(&mut wicks),
        &world,
    );

    assert!(
        !result.pt2.is_empty(),
        "SNOCI did not produce a NOCI-PT2 result"
    );
    for pt2 in &result.pt2 {
        assert!(
            pt2.gmres_converged,
            "SNOCI GMRES failed to converge: residual {}",
            pt2.gmres_residual,
        );
    }

    (scf_energies, e_ref, result.ecurrent)
}

/// Run a Wicks SNOCI fixture with an explicit one-body backend.
/// The input fixture supplies all numerical settings, while this helper overwrites only
/// `snoci.backend` so CPU and GPU runs compare the same matrix-free factorised path.
/// # Arguments:
/// - `fixture`: Name of the Wicks fixture to load.
/// - `backend`: One-body backend to use for the NOCI-PT2 solves inside SNOCI.
/// # Returns
/// - `SNOCIBackendRun`: Energies and GMRES diagnostics from the final completed iteration.
fn run_snoci_fixture_wicks_with(
    fixture: &str,
    backend: SNOCIBackend,
) -> SNOCIGpuComparisonRun {
    let (mut input, ao, _expected): (_, _, ExpectedSNOCI) = load_test(fixture);
    input
        .snoci
        .as_mut()
        .expect("SNOCI fixture must define snoci")
        .backend = backend;

    let basis = generate_reference_noci_basis(&ao, &mut input, None, None);
    let states = basis.states;

    let mut scf_energies: Vec<f64> = states.iter().map(|s| s.e).collect();
    scf_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut noci_reference_basis: Vec<_> =
        states.iter().filter(|s| s.noci_basis).cloned().collect();
    for (i, st) in noci_reference_basis.iter_mut().enumerate() {
        st.parent = i;
    }

    let mocache = build_mo_cache(&ao, &noci_reference_basis, input.scf.d_tol);

    let (_mpi_lock, universe) = mpi_universe();
    let world = universe.world();

    let mut wicks = build_wicks_shared::<f64>(&world, &ao, &noci_reference_basis, 1e-12, &input);
    let (e_ref, _c0, _dt_hs_ref) = {
        let wicks_view = wicks.view();
        calculate_noci_energy(
            &ao,
            &input,
            &noci_reference_basis,
            1e-12,
            &mocache,
            Some(wicks_view),
        )
    };

    let post = PostSCFData {
        ao: &ao,
        states: &states,
        noci_reference_basis: &noci_reference_basis,
        mocache: &mocache,
        tol: 1e-12,
    };

    let result = snoci_step(
        &post,
        &noci_reference_basis,
        &input,
        Some(&mut wicks),
        &world,
    );
    let pt2 = result
        .pt2
        .last()
        .expect("SNOCI did not produce a NOCI-PT2 result");
    assert!(
        pt2.gmres_converged,
        "SNOCI GMRES failed to converge: residual {}",
        pt2.gmres_residual,
    );

    SNOCIGpuComparisonRun {
        scf_energies,
        reference_noci_energy: e_ref,
        snoci_energy: result.ecurrent,
        ept2: pt2.ept2,
        gmres_residual: pt2.gmres_residual,
        gmres_iterations: pt2.gmres_iterations,
    }
}

/// Test that the H2 STO-3G 1.5 Angstrom fixture reproduces the expected SCF state energies,
/// reference NOCI energy and selected NOCI energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF, reference NOCI or selected NOCI energy differs from the known good value
///   outside tolerance.
/// - If the NOCI-PT2 GMRES solve does not converge.
#[test]
#[serial]
fn snoci_h2_sto_3g_1_5_ang_energies() {
    let (_input, _ao, expected): (_, _, ExpectedSNOCI) = load_test("SNOCI_H2_STO-3G_1_5");
    let (got_scf, got_ref, got_snoci) = run_snoci_fixture("SNOCI_H2_STO-3G_1_5");

    let mut want_scf = expected.scf_energies;
    want_scf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got_scf.len(), want_scf.len());
    for (i, (&x, &y)) in got_scf.iter().zip(want_scf.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("H2 SCF state {i}"));
    }

    assert_close(
        got_ref,
        expected.reference_noci_energy,
        1e-8,
        "H2 reference NOCI energy",
    );
    assert_close(
        got_snoci,
        expected.snoci_energy,
        1e-8,
        "H2 selected NOCI energy",
    );
}

/// Test that the H2 STO-3G 1.5 Angstrom fixture reproduces the expected SCF state energies,
/// reference NOCI energy and selected NOCI energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF, reference NOCI or selected NOCI energy differs from the known good value
///   outside tolerance.
/// - If the NOCI-PT2 GMRES solve does not converge.
#[test]
#[serial]
fn snoci_h2_sto_3g_1_5_ang_energies_wicks() {
    let (_input, _ao, expected): (_, _, ExpectedSNOCI) = load_test("SNOCI_H2_STO-3G_1_5_WICKS");
    let (got_scf, got_ref, got_snoci) = run_snoci_fixture_wicks("SNOCI_H2_STO-3G_1_5_WICKS");

    let mut want_scf = expected.scf_energies;
    want_scf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got_scf.len(), want_scf.len());
    for (i, (&x, &y)) in got_scf.iter().zip(want_scf.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("H2 SCF state {i}"));
    }

    assert_close(
        got_ref,
        expected.reference_noci_energy,
        1e-8,
        "H2 reference NOCI energy",
    );
    assert_close(
        got_snoci,
        expected.snoci_energy,
        1e-8,
        "H2 selected NOCI energy",
    );
}

/// Test that the H2 STO-3G 1.5 Angstrom fixture agrees with and without Wick's intermediates.
/// # Panics
/// - If the number of SCF states differs between implementations.
/// - If SCF, reference NOCI or selected NOCI energy differs between implementations
///   outside tolerance.
/// - If the NOCI-PT2 GMRES solve does not converge.
#[test]
#[serial]
fn snoci_h2_sto_3g_1_5_ang_energies_agree() {
    let (got_scf, got_ref, got_snoci) = run_snoci_fixture("SNOCI_H2_STO-3G_1_5");
    let (got_scf_wicks, got_ref_wicks, got_snoci_wicks) =
        run_snoci_fixture_wicks("SNOCI_H2_STO-3G_1_5_WICKS");

    assert_eq!(got_scf.len(), got_scf_wicks.len());
    for (i, (&x, &y)) in got_scf.iter().zip(got_scf_wicks.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("H2 SCF state {i} Wicks agreement"));
    }

    assert_close(
        got_ref,
        got_ref_wicks,
        1e-8,
        "H2 reference NOCI Wicks agreement",
    );
    assert_close(
        got_snoci,
        got_snoci_wicks,
        1e-8,
        "H2 selected NOCI Wicks agreement",
    );
}

/// Test that the H4 STO-3G 1.5 Angstrom fixture reproduces the expected SCF state energies,
/// reference NOCI energy and selected NOCI energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF, reference NOCI or selected NOCI energy differs from the known good value
///   outside tolerance.
/// - If the NOCI-PT2 GMRES solve does not converge.
#[test]
#[serial]
fn snoci_h4_sto_3g_1_5_ang_energies() {
    let (_input, _ao, expected): (_, _, ExpectedSNOCI) = load_test("SNOCI_H4_STO-3G_1_5");
    let (got_scf, got_ref, got_snoci) = run_snoci_fixture("SNOCI_H4_STO-3G_1_5");

    let mut want_scf = expected.scf_energies;
    want_scf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got_scf.len(), want_scf.len());
    for (i, (&x, &y)) in got_scf.iter().zip(want_scf.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("H4 SCF state {i}"));
    }

    assert_close(
        got_ref,
        expected.reference_noci_energy,
        1e-8,
        "H4 reference NOCI energy",
    );
    assert_close(
        got_snoci,
        expected.snoci_energy,
        1e-8,
        "H4 selected NOCI energy",
    );
}

/// Test that the H4 STO-3G 1.5 Angstrom fixture reproduces the expected SCF state energies,
/// reference NOCI energy and selected NOCI energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF, reference NOCI or selected NOCI energy differs from the known good value
///   outside tolerance.
/// - If the NOCI-PT2 GMRES solve does not converge.
#[test]
#[serial]
fn snoci_h4_sto_3g_1_5_ang_energies_wicks() {
    let (_input, _ao, expected): (_, _, ExpectedSNOCI) = load_test("SNOCI_H4_STO-3G_1_5_WICKS");
    let (got_scf, got_ref, got_snoci) = run_snoci_fixture_wicks("SNOCI_H4_STO-3G_1_5_WICKS");

    let mut want_scf = expected.scf_energies;
    want_scf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got_scf.len(), want_scf.len());
    for (i, (&x, &y)) in got_scf.iter().zip(want_scf.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("H4 SCF state {i}"));
    }

    assert_close(
        got_ref,
        expected.reference_noci_energy,
        1e-8,
        "H4 reference NOCI energy",
    );
    assert_close(
        got_snoci,
        expected.snoci_energy,
        1e-8,
        "H4 selected NOCI energy",
    );
}

/// Test that the H4 STO-3G 1.5 Angstrom reference space agrees with and without Wick's intermediates.
/// # Panics
/// - If the number of SCF states differs between implementations.
/// - If SCF or reference NOCI energy differs between implementations outside tolerance.
/// - If the NOCI-PT2 GMRES solve does not converge.
#[test]
#[serial]
fn snoci_h4_sto_3g_1_5_ang_energies_agree() {
    let (got_scf, got_ref, _got_snoci) = run_snoci_fixture("SNOCI_H4_STO-3G_1_5");
    let (got_scf_wicks, got_ref_wicks, _got_snoci_wicks) =
        run_snoci_fixture_wicks("SNOCI_H4_STO-3G_1_5_WICKS");

    assert_eq!(got_scf.len(), got_scf_wicks.len());
    for (i, (&x, &y)) in got_scf.iter().zip(got_scf_wicks.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("H4 SCF state {i} Wicks agreement"));
    }

    assert_close(
        got_ref,
        got_ref_wicks,
        1e-8,
        "H4 reference NOCI Wicks agreement",
    );
}

/// Test an H4 double-excitation pair through ordinary and Wick's matrix builders.
/// # Panics
/// - If the selected candidates are not alpha-alpha doubles relative to their parents.
/// - If ordinary and Wick's Hamiltonian or overlap matrices disagree outside tolerance.
#[test]
#[serial]
fn snoci_h4_sto_3g_1_5_ang_double_excitation_pair_wicks() {
    let (mut input, ao, _expected): (_, _, ExpectedSNOCI) = load_test("SNOCI_H4_STO-3G_1_5");

    let basis = generate_reference_noci_basis(&ao, &mut input, None, None);
    let mut refs: Vec<_> = basis
        .states
        .iter()
        .filter(|s| s.noci_basis)
        .cloned()
        .collect();
    for (i, st) in refs.iter_mut().enumerate() {
        st.parent = i;
    }

    let candidates = generate_excited_basis(&refs, &input, false);
    let rank = |parent: usize, child_oa: u128, child_ob: u128| {
        let parent = &refs[parent];
        (
            (parent.oa & !child_oa).count_ones() as usize,
            (child_oa & !parent.oa).count_ones() as usize,
            (parent.ob & !child_ob).count_ones() as usize,
            (child_ob & !parent.ob).count_ones() as usize,
        )
    };

    let left = candidates
        .iter()
        .find(|st| st.parent == 0 && rank(0, st.oa, st.ob) == (2, 2, 0, 0))
        .expect("missing RHF alpha-alpha double excitation")
        .clone();
    let right = candidates
        .iter()
        .find(|st| st.parent == 1 && rank(1, st.oa, st.ob) == (2, 2, 0, 0))
        .expect("missing alpha-parent alpha-alpha double excitation")
        .clone();
    assert_eq!(rank(0, left.oa, left.ob), (2, 2, 0, 0));
    assert_eq!(rank(1, right.oa, right.ob), (2, 2, 0, 0));

    let probe = vec![left, right];
    let mocache = build_mo_cache(&ao, &refs, input.scf.d_tol);
    let data = NOCIData::new(&ao, &probe, &input, 1e-12, None).withmocache(&mocache);
    let (h, s, _dt) = build_noci_hs(&data, &probe, &probe, true);

    let (mut input_wicks, ao_wicks, _expected): (_, _, ExpectedSNOCI) =
        load_test("SNOCI_H4_STO-3G_1_5_WICKS");
    input_wicks.wicks.compare = false;
    let basis_wicks = generate_reference_noci_basis(&ao_wicks, &mut input_wicks, None, None);
    let mut refs_wicks: Vec<_> = basis_wicks
        .states
        .iter()
        .filter(|s| s.noci_basis)
        .cloned()
        .collect();
    for (i, st) in refs_wicks.iter_mut().enumerate() {
        st.parent = i;
    }
    let candidates_wicks = generate_excited_basis(&refs_wicks, &input_wicks, false);
    let rank_wicks = |parent: usize, child_oa: u128, child_ob: u128| {
        let parent = &refs_wicks[parent];
        (
            (parent.oa & !child_oa).count_ones() as usize,
            (child_oa & !parent.oa).count_ones() as usize,
            (parent.ob & !child_ob).count_ones() as usize,
            (child_ob & !parent.ob).count_ones() as usize,
        )
    };
    let left_wicks = candidates_wicks
        .iter()
        .find(|st| st.parent == 0 && rank_wicks(0, st.oa, st.ob) == (2, 2, 0, 0))
        .expect("missing Wicks RHF alpha-alpha double excitation")
        .clone();
    let right_wicks = candidates_wicks
        .iter()
        .find(|st| st.parent == 1 && rank_wicks(1, st.oa, st.ob) == (2, 2, 0, 0))
        .expect("missing Wicks alpha-parent alpha-alpha double excitation")
        .clone();
    let probe_wicks = vec![left_wicks, right_wicks];
    let mocache_wicks = build_mo_cache(&ao_wicks, &refs_wicks, input_wicks.scf.d_tol);
    let (_mpi_lock, universe) = mpi_universe();
    let world = universe.world();
    let wicks = build_wicks_shared::<f64>(&world, &ao_wicks, &refs_wicks, 1e-12, &input_wicks);
    let data_wicks = NOCIData::new(
        &ao_wicks,
        &probe_wicks,
        &input_wicks,
        1e-12,
        Some(wicks.view()),
    )
    .withmocache(&mocache_wicks);
    let (h_wicks, s_wicks, _dt) = build_noci_hs(&data_wicks, &probe_wicks, &probe_wicks, true);

    assert_eq!(h.shape(), &[2, 2]);
    assert_eq!(s.shape(), &[2, 2]);
    for i in 0..2 {
        for j in 0..2 {
            assert!(h[(i, j)].is_finite());
            assert!(s[(i, j)].is_finite());
            assert_close(h[(i, j)], h[(j, i)], 1e-10, &format!("H4 H {i},{j}"));
            assert_close(s[(i, j)], s[(j, i)], 1e-10, &format!("H4 S {i},{j}"));
            assert_close(
                h[(i, j)],
                h_wicks[(i, j)],
                1e-9,
                &format!("H4 rank-two H Wicks {i},{j}"),
            );
            assert_close(
                s[(i, j)],
                s_wicks[(i, j)],
                1e-9,
                &format!("H4 rank-two S Wicks {i},{j}"),
            );
        }
        assert_close(s[(i, i)], 1.0, 1e-10, &format!("H4 S diagonal {i}"));
    }
    assert!(h[(0, 1)].abs() > 1e-12 || s[(0, 1)].abs() > 1e-12);
}

/// Test that the H2 3-21G 1.5 Angstrom fixture reproduces the expected SCF state energies,
/// reference NOCI energy and selected NOCI energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF, reference NOCI or selected NOCI energy differs from the known good value
///   outside tolerance.
/// - If the NOCI-PT2 GMRES solve does not converge.
#[test]
#[serial]
fn snoci_h2_3_21g_1_5_ang_energies() {
    let (_input, _ao, expected): (_, _, ExpectedSNOCI) = load_test("SNOCI_H2_3-21G_1_5");
    let (got_scf, got_ref, got_snoci) = run_snoci_fixture("SNOCI_H2_3-21G_1_5");

    let mut want_scf = expected.scf_energies;
    want_scf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got_scf.len(), want_scf.len());
    for (i, (&x, &y)) in got_scf.iter().zip(want_scf.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("H2 SCF state {i}"));
    }

    assert_close(
        got_ref,
        expected.reference_noci_energy,
        1e-8,
        "H2 reference NOCI energy",
    );
    assert_close(
        got_snoci,
        expected.snoci_energy,
        1e-8,
        "H2 selected NOCI energy",
    );
}

/// Test that the H2 3-21G 1.5 Angstrom fixture reproduces the expected SCF state energies,
/// reference NOCI energy and selected NOCI energy.
/// # Panics
/// - If the number of SCF states differs from the stored reference.
/// - If SCF, reference NOCI or selected NOCI energy differs from the known good value
///   outside tolerance.
/// - If the NOCI-PT2 GMRES solve does not converge.
#[test]
#[serial]
fn snoci_h2_3_21g_1_5_ang_energies_wicks() {
    let (_input, _ao, expected): (_, _, ExpectedSNOCI) = load_test("SNOCI_H2_3-21G_1_5_WICKS");
    let (got_scf, got_ref, got_snoci) = run_snoci_fixture_wicks("SNOCI_H2_3-21G_1_5_WICKS");

    let mut want_scf = expected.scf_energies;
    want_scf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(got_scf.len(), want_scf.len());
    for (i, (&x, &y)) in got_scf.iter().zip(want_scf.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("H2 SCF state {i}"));
    }

    assert_close(
        got_ref,
        expected.reference_noci_energy,
        1e-8,
        "H2 reference NOCI energy",
    );
    assert_close(
        got_snoci,
        expected.snoci_energy,
        1e-8,
        "H2 selected NOCI energy",
    );
}

/// Test that the H2 3-21G 1.5 Angstrom fixture agrees with and without Wick's intermediates.
/// # Panics
/// - If the number of SCF states differs between implementations.
/// - If SCF, reference NOCI or selected NOCI energy differs between implementations
///   outside tolerance.
/// - If the NOCI-PT2 GMRES solve does not converge.
#[test]
#[serial]
fn snoci_h2_3_21g_1_5_ang_energies_agree() {
    let (got_scf, got_ref, got_snoci) = run_snoci_fixture("SNOCI_H2_3-21G_1_5");
    let (got_scf_wicks, got_ref_wicks, got_snoci_wicks) =
        run_snoci_fixture_wicks("SNOCI_H2_3-21G_1_5_WICKS");

    assert_eq!(got_scf.len(), got_scf_wicks.len());
    for (i, (&x, &y)) in got_scf.iter().zip(got_scf_wicks.iter()).enumerate() {
        assert_close(x, y, 1e-8, &format!("H2 SCF state {i} Wicks agreement"));
    }

    assert_close(
        got_ref,
        got_ref_wicks,
        1e-8,
        "H2 reference NOCI Wicks agreement",
    );
    assert_close(
        got_snoci,
        got_snoci_wicks,
        1e-8,
        "H2 selected NOCI Wicks agreement",
    );
}

/// Test that the H2 3-21G GPU SNOCI path agrees with the CPU factorised path.
/// # Panics
/// - If CPU or GPU GMRES does not converge.
/// - If SCF, reference NOCI, selected NOCI, PT2 correction or GMRES diagnostics differ outside
///   tolerance.
#[cfg(feature = "gpu")]
#[test]
#[ignore = "gpu"]
#[serial]
fn snoci_h2_3_21g_1_5_ang_gpu_matches_cpu() {
    let cpu = run_snoci_fixture_wicks_with("SNOCI_H2_3-21G_1_5_GPU", SNOCIBackend::CPU);
    let gpu = run_snoci_fixture_wicks_with("SNOCI_H2_3-21G_1_5_GPU", SNOCIBackend::GPU);

    assert_eq!(cpu.scf_energies.len(), gpu.scf_energies.len());
    for (i, (&x, &y)) in cpu
        .scf_energies
        .iter()
        .zip(gpu.scf_energies.iter())
        .enumerate()
    {
        assert_close(x, y, 1e-8, &format!("H2 SCF state {i} GPU agreement"));
    }

    assert_close(
        cpu.reference_noci_energy,
        gpu.reference_noci_energy,
        1e-8,
        "H2 reference NOCI GPU agreement",
    );
    assert_close(
        cpu.snoci_energy,
        gpu.snoci_energy,
        1e-8,
        "H2 selected NOCI GPU agreement",
    );
    assert_close(cpu.ept2, gpu.ept2, 1e-8, "H2 EPT2 GPU agreement");
    assert_close(
        cpu.gmres_residual,
        gpu.gmres_residual,
        1e-8,
        "H2 GMRES residual GPU agreement",
    );
    assert_eq!(cpu.gmres_iterations, gpu.gmres_iterations);
}
