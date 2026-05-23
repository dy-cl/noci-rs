// driver/report.rs

use crate::HSCFState;
use crate::driver::types::GeometryResults;
use crate::input::{Input, StateType};

/// Print important information for current geometry.
/// # Arguments:
/// - `res`: Contains the aforementioned important information.
/// - `input`: User input specifications.
/// # Returns:
/// - `()`: Prints the report to stdout.
pub fn print_report(
    res: &GeometryResults,
    input: &Input,
) {
    fn print_counter(
        lbl: &str,
        c: crate::timers::Counter,
        indent: usize,
    ) {
        let avg = c
            .ns
            .checked_div(c.calls)
            .map(std::time::Duration::from_nanos)
            .unwrap_or_default();

        println!(
            "{}{}: {:?} [{} calls, avg {:?}/call]",
            " ".repeat(indent),
            lbl,
            c.duration(),
            c.calls,
            avg
        );
    }

    fn print_relative_counter(
        label: &str,
        counter: crate::timers::Counter,
        parent: crate::timers::Counter,
        indent: usize,
    ) {
        let avg = if counter.calls > 0 {
            std::time::Duration::from_nanos(counter.ns / counter.calls)
        } else {
            std::time::Duration::from_nanos(0)
        };
        let pct = if parent.ns > 0 {
            100.0 * counter.ns as f64 / parent.ns as f64
        } else {
            0.0
        };

        println!(
            "{}{}: {:?} [{} calls, avg {:?}/call, {:.2}%]",
            " ".repeat(indent),
            label,
            counter.duration(),
            counter.calls,
            avg,
            pct
        );
    }

    let nthreads = rayon::current_num_threads();
    let hlabels: Vec<&str> = match &input.states {
        StateType::Mom(recipes) => recipes
            .iter()
            .filter(|recipe| recipe.holomorphic)
            .map(|recipe| recipe.label.as_str())
            .collect(),
        StateType::Metadynamics(_) => Vec::new(),
    };
    let requested_noci = |label: &str| -> bool {
        match &input.states {
            StateType::Mom(recipes) => recipes
                .iter()
                .find(|recipe| recipe.label == label)
                .map(|recipe| recipe.noci)
                .unwrap_or(false),
            StateType::Metadynamics(_) => false,
        }
    };
    let report_label = |label: &str, noci_basis: bool| -> String {
        if requested_noci(label) && !noci_basis {
            format!("(Removed) {}", label)
        } else {
            label.to_string()
        }
    };

    let s_pair_total = res.timings.noci.calculate_s_pair;
    let f_pair_total = res.timings.noci.calculate_f_pair;
    let hs_pair_total = res.timings.noci.calculate_hs_pair;
    let one_body_operator_total = crate::timers::Counter {
        ns: res.timings.nonorthogonalwicks.lg_h1.ns + res.timings.nonorthogonalwicks.lg_f.ns,
        calls: res.timings.nonorthogonalwicks.lg_h1.calls
            + res.timings.nonorthogonalwicks.lg_f.calls,
    };

    let one_body_alg_total = crate::timers::Counter {
        ns: res.timings.nonorthogonalwicks.lg_one_body_gen.ns
            + res.timings.nonorthogonalwicks.lg_one_body_m0.ns,
        calls: res.timings.nonorthogonalwicks.lg_one_body_gen.calls
            + res.timings.nonorthogonalwicks.lg_one_body_m0.calls,
    };

    let wick_total = crate::timers::Counter {
        ns: res.timings.noci.calculate_s_pair_wicks.ns
            + res.timings.noci.calculate_f_pair_wicks.ns
            + res.timings.noci.calculate_hs_pair_wicks.ns,
        calls: res.timings.noci.calculate_s_pair_wicks.calls
            + res.timings.noci.calculate_f_pair_wicks.calls
            + res.timings.noci.calculate_hs_pair_wicks.calls,
    };

    println!("{}", "=".repeat(100));
    println!("Number of MPI ranks: {}", res.nranks);
    println!("Number of Rayon threads per rank: {}", nthreads);
    println!("Warning: Timing functions will impact performance and thread efficiency.");
    println!("Please interpret these timings as a distribution but not absolute.");
    println!("Timing overhead will be quite large relative to some of the smaller kernels.");

    println!("{}", "-".repeat(100));

    print_counter("Total PySCF time", res.timings.general.run_pyscf, 0);
    print_counter("Total SCF time", res.timings.general.run_scf, 0);
    print_counter(
        "Total Reference NOCI time",
        res.timings.general.run_reference_noci,
        0,
    );
    print_counter(
        r"S & H_1 & H_2",
        res.timings.general.calculate_noci_energy,
        1,
    );
    print_counter(
        "Wick's Intermediates Construction time",
        res.timings.general.build_wicks_shared,
        0,
    );

    println!("{}", "-".repeat(100));

    if input.snoci.is_some() {
        print_counter("Total SNOCI time", res.timings.snoci.run_snoci, 0);
        print_counter("Full SNOCI step", res.timings.snoci.snoci_step, 2);
        print_counter(
            "Current space H, S and GEVP",
            res.timings.snoci.solve_current_space,
            2,
        );
        print_counter(
            "Initial candidate pool generation",
            res.timings.snoci.candidate_pool_new,
            2,
        );
        print_counter(
            "Update candidate pool overlaps",
            res.timings.snoci.candidate_pool_update,
            2,
        );

        println!("{}", "-".repeat(100));

        print_counter(
            "Candidate-current overlaps",
            res.timings.snoci.build_snoci_overlaps,
            2,
        );
        print_counter(
            "Candidate-current space H",
            res.timings.snoci.build_candidate_h_ai,
            2,
        );
        print_counter(
            "Generalised Fock build",
            res.timings.snoci.build_generalised_fock,
            2,
        );
        print_counter(
            "Current-current and candidate-current Fock blocks",
            res.timings.snoci.build_snoci_focks,
            2,
        );
        print_counter(
            "Build cached candidate shifted Fock",
            res.timings.snoci.build_candidate_m,
            2,
        );
        print_counter(
            "Build candidate shifted Fock diagonal",
            res.timings.snoci.build_candidate_m_diag,
            2,
        );
        print_counter(
            "PT2 projection contractions",
            res.timings.snoci.build_snoci_projection,
            2,
        );
        print_counter(
            "Candidate coupling vector",
            res.timings.snoci.build_candidate_v,
            2,
        );
        print_counter(
            "Projected coupling vector",
            res.timings.snoci.build_omega_v,
            2,
        );
        print_counter(
            "Build projected PT2 diagonal",
            res.timings.snoci.build_omega_m_diag,
            2,
        );

        println!("{}", "-".repeat(100));

        print_counter("GMRES Solve", res.timings.snoci.gmres, 2);
        print_counter(
            "Apply projected PT2 operator",
            res.timings.snoci.apply_omega_m,
            4,
        );
        print_counter(
            "Apply unprojected candidate M",
            res.timings.snoci.apply_candidate_m,
            6,
        );

        println!("{}", "-".repeat(100));
    }

    println!("Shared full matrix build timings");
    print_counter(
        "Full Fock matrix build",
        res.timings.noci.build_full_fock,
        2,
    );
    print_counter(
        "Full overlap matrix build",
        res.timings.noci.build_full_overlap,
        2,
    );
    print_counter(
        "Full Hamiltonian and overlap matrix build",
        res.timings.noci.build_full_hs,
        2,
    );

    println!("{}", "-".repeat(100));

    println!("Shared overlap matrix-element timings");
    print_counter("Overlap dispatch", res.timings.noci.calculate_s_pair, 2);
    print_relative_counter(
        "Overlap via Wick's theorem",
        res.timings.noci.calculate_s_pair_wicks,
        s_pair_total,
        5,
    );
    print_relative_counter(
        "Overlap via generalised Slater-Condon",
        res.timings.noci.calculate_s_pair_naive,
        s_pair_total,
        5,
    );
    print_relative_counter(
        "Overlap via orthogonal Slater-Condon",
        res.timings.noci.calculate_s_pair_orthogonal,
        s_pair_total,
        5,
    );

    println!("{}", "-".repeat(100));

    println!("Shared Fock matrix-element timings");
    print_counter("Fock dispatch", res.timings.noci.calculate_f_pair, 2);
    print_relative_counter(
        "Fock via Wick's theorem",
        res.timings.noci.calculate_f_pair_wicks,
        f_pair_total,
        5,
    );
    print_relative_counter(
        "Fock via generalised Slater-Condon",
        res.timings.noci.calculate_f_pair_naive,
        f_pair_total,
        5,
    );
    print_relative_counter(
        "Fock via orthogonal Slater-Condon",
        res.timings.noci.calculate_f_pair_orthogonal,
        f_pair_total,
        5,
    );

    println!("{}", "-".repeat(100));

    println!("Shared Hamiltonian and overlap matrix-element timings");
    print_counter(
        "Hamiltonian and overlap dispatch",
        res.timings.noci.calculate_hs_pair,
        2,
    );
    print_relative_counter(
        "Hamiltonian and overlap via Wick's theorem",
        res.timings.noci.calculate_hs_pair_wicks,
        hs_pair_total,
        5,
    );
    print_relative_counter(
        "Hamiltonian and overlap via generalised Slater-Condon",
        res.timings.noci.calculate_hs_pair_naive,
        hs_pair_total,
        5,
    );
    print_relative_counter(
        "Hamiltonian and overlap via orthogonal Slater-Condon",
        res.timings.noci.calculate_hs_pair_orthogonal,
        hs_pair_total,
        5,
    );

    println!("{}", "-".repeat(100));

    println!("Shared NOCI MO cache timings");
    print_counter(
        "MO integral cache build",
        res.timings.noci.build_mo_cache,
        2,
    );
    print_counter(
        "Fock MO cache build",
        res.timings.noci.build_fock_mo_cache,
        2,
    );

    println!("{}", "-".repeat(100));

    println!("Shared nonorthogonal Wick timings");
    print_relative_counter(
        "Same-spin mix determinants and adjugates",
        res.timings.nonorthogonalwicks.get_det_adjt_same,
        wick_total,
        2,
    );
    print_relative_counter(
        "Different-spin mix determinants and adjugates",
        res.timings.nonorthogonalwicks.get_det_adjt_diff,
        wick_total,
        2,
    );

    println!("{}", "-".repeat(100));

    print_relative_counter(
        "Prepare same-spin information",
        res.timings.nonorthogonalwicks.prepare_same,
        wick_total,
        2,
    );
    print_relative_counter(
        "Prepare same-spin information (generic)",
        res.timings.nonorthogonalwicks.prepare_same_gen,
        res.timings.nonorthogonalwicks.prepare_same,
        5,
    );
    print_relative_counter(
        "Prepare same-spin information (m = 0)",
        res.timings.nonorthogonalwicks.prepare_same_m0,
        res.timings.nonorthogonalwicks.prepare_same,
        5,
    );
    print_relative_counter(
        "Prepare same-spin determinant fill (m = 0, l = 1)",
        res.timings.nonorthogonalwicks.prepare_same_m0_l1,
        res.timings.nonorthogonalwicks.prepare_same_m0,
        8,
    );
    print_relative_counter(
        "Prepare same-spin determinant fill (m = 0, l = 2)",
        res.timings.nonorthogonalwicks.prepare_same_m0_l2,
        res.timings.nonorthogonalwicks.prepare_same_m0,
        8,
    );
    print_relative_counter(
        "Prepare same-spin determinant fill (m = 0, l = 3)",
        res.timings.nonorthogonalwicks.prepare_same_m0_l3,
        res.timings.nonorthogonalwicks.prepare_same_m0,
        8,
    );
    print_relative_counter(
        "Prepare same-spin determinant fill (m = 0, l = 4)",
        res.timings.nonorthogonalwicks.prepare_same_m0_l4,
        res.timings.nonorthogonalwicks.prepare_same_m0,
        8,
    );

    println!("{}", "-".repeat(100));

    print_relative_counter(
        "Construct determinant indices",
        res.timings.nonorthogonalwicks.construct_determinant_indices,
        wick_total,
        2,
    );
    print_relative_counter(
        "Construct determinant indices (generic)",
        res.timings
            .nonorthogonalwicks
            .construct_determinant_indices_gen,
        res.timings.nonorthogonalwicks.construct_determinant_indices,
        5,
    );
    print_relative_counter(
        "Construct determinant indices (l = 1)",
        res.timings
            .nonorthogonalwicks
            .construct_determinant_indices_l1,
        res.timings.nonorthogonalwicks.construct_determinant_indices,
        5,
    );
    print_relative_counter(
        "Construct determinant indices (l = 2)",
        res.timings
            .nonorthogonalwicks
            .construct_determinant_indices_l2,
        res.timings.nonorthogonalwicks.construct_determinant_indices,
        5,
    );
    print_relative_counter(
        "Construct determinant indices (l = 3)",
        res.timings
            .nonorthogonalwicks
            .construct_determinant_indices_l3,
        res.timings.nonorthogonalwicks.construct_determinant_indices,
        5,
    );
    print_relative_counter(
        "Construct determinant indices (l = 4)",
        res.timings
            .nonorthogonalwicks
            .construct_determinant_indices_l4,
        res.timings.nonorthogonalwicks.construct_determinant_indices,
        5,
    );

    println!("{}", "-".repeat(100));

    print_relative_counter(
        "Overlap matrix elements",
        res.timings.nonorthogonalwicks.lg_overlap,
        wick_total,
        2,
    );
    print_relative_counter(
        "Overlap matrix elements (m = 0)",
        res.timings.nonorthogonalwicks.lg_overlap_m0,
        res.timings.nonorthogonalwicks.lg_overlap,
        5,
    );
    print_relative_counter(
        "Overlap matrix elements (m = 0, l = 1)",
        res.timings.nonorthogonalwicks.lg_overlap_m0_l1,
        res.timings.nonorthogonalwicks.lg_overlap_m0,
        8,
    );
    print_relative_counter(
        "Overlap matrix elements (m = 0, l = 2)",
        res.timings.nonorthogonalwicks.lg_overlap_m0_l2,
        res.timings.nonorthogonalwicks.lg_overlap_m0,
        8,
    );
    print_relative_counter(
        "Overlap matrix elements (m = 0, l = 3)",
        res.timings.nonorthogonalwicks.lg_overlap_m0_l3,
        res.timings.nonorthogonalwicks.lg_overlap_m0,
        8,
    );
    print_relative_counter(
        "Overlap matrix elements (m = l)",
        res.timings.nonorthogonalwicks.lg_overlap_ml,
        res.timings.nonorthogonalwicks.lg_overlap,
        5,
    );
    print_relative_counter(
        "Overlap matrix elements (m = l, l = 1)",
        res.timings.nonorthogonalwicks.lg_overlap_ml_l1,
        res.timings.nonorthogonalwicks.lg_overlap_ml,
        8,
    );
    print_relative_counter(
        "Overlap matrix elements (m = l, l = 2)",
        res.timings.nonorthogonalwicks.lg_overlap_ml_l2,
        res.timings.nonorthogonalwicks.lg_overlap_ml,
        8,
    );
    print_relative_counter(
        "Overlap matrix elements (m = l, l = 3)",
        res.timings.nonorthogonalwicks.lg_overlap_ml_l3,
        res.timings.nonorthogonalwicks.lg_overlap_ml,
        8,
    );
    print_relative_counter(
        "Overlap matrix elements (generic)",
        res.timings.nonorthogonalwicks.lg_overlap_gen,
        res.timings.nonorthogonalwicks.lg_overlap,
        5,
    );

    println!("{}", "-".repeat(100));

    println!("One-body operator timings");
    print_relative_counter(
        "One-body operator wrappers",
        one_body_operator_total,
        wick_total,
        2,
    );
    print_relative_counter(
        "Hamiltonian one-body matrix elements",
        res.timings.nonorthogonalwicks.lg_h1,
        one_body_operator_total,
        5,
    );
    print_relative_counter(
        "Fock one-body matrix elements",
        res.timings.nonorthogonalwicks.lg_f,
        one_body_operator_total,
        5,
    );

    println!("One-body algorithmic timings");
    print_relative_counter(
        "One-body algorithmic paths",
        one_body_alg_total,
        wick_total,
        2,
    );
    print_relative_counter(
        "One-body matrix elements (generic)",
        res.timings.nonorthogonalwicks.lg_one_body_gen,
        one_body_alg_total,
        5,
    );
    print_relative_counter(
        "One-body matrix elements (m = 0)",
        res.timings.nonorthogonalwicks.lg_one_body_m0,
        one_body_alg_total,
        5,
    );
    print_relative_counter(
        "One-body matrix elements (m = 0, generic)",
        res.timings.nonorthogonalwicks.lg_one_body_m0_gen,
        res.timings.nonorthogonalwicks.lg_one_body_m0,
        8,
    );
    print_relative_counter(
        "One-body matrix elements (m = 0, l = 1)",
        res.timings.nonorthogonalwicks.lg_one_body_m0_l1,
        res.timings.nonorthogonalwicks.lg_one_body_m0,
        8,
    );
    print_relative_counter(
        "One-body matrix elements (m = 0, l = 2)",
        res.timings.nonorthogonalwicks.lg_one_body_m0_l2,
        res.timings.nonorthogonalwicks.lg_one_body_m0,
        8,
    );

    println!("{}", "-".repeat(100));

    print_relative_counter(
        "Same-spin two-electron matrix elements",
        res.timings.nonorthogonalwicks.lg_h2_same,
        wick_total,
        2,
    );
    print_relative_counter(
        "Same-spin two-electron matrix elements (generic)",
        res.timings.nonorthogonalwicks.lg_h2_same_gen,
        res.timings.nonorthogonalwicks.lg_h2_same,
        5,
    );
    print_relative_counter(
        "Same-spin two-electron matrix elements (m = 0)",
        res.timings.nonorthogonalwicks.lg_h2_same_m0,
        res.timings.nonorthogonalwicks.lg_h2_same,
        5,
    );
    print_relative_counter(
        "Same-spin two-electron matrix elements (m = 0, generic)",
        res.timings.nonorthogonalwicks.lg_h2_same_m0_gen,
        res.timings.nonorthogonalwicks.lg_h2_same_m0,
        8,
    );
    print_relative_counter(
        "Same-spin two-electron matrix elements (m = 0, l = 1)",
        res.timings.nonorthogonalwicks.lg_h2_same_m0_l1,
        res.timings.nonorthogonalwicks.lg_h2_same_m0,
        8,
    );
    print_relative_counter(
        "Same-spin two-electron matrix elements (m = 0, l = 2)",
        res.timings.nonorthogonalwicks.lg_h2_same_m0_l2,
        res.timings.nonorthogonalwicks.lg_h2_same_m0,
        8,
    );
    print_relative_counter(
        "Same-spin two-electron matrix elements (m = 0, l = 3)",
        res.timings.nonorthogonalwicks.lg_h2_same_m0_l3,
        res.timings.nonorthogonalwicks.lg_h2_same_m0,
        8,
    );

    println!("{}", "-".repeat(100));

    print_relative_counter(
        "Different-spin two-electron matrix elements",
        res.timings.nonorthogonalwicks.lg_h2_diff,
        wick_total,
        2,
    );
    print_relative_counter(
        "Different-spin two-electron matrix elements (generic)",
        res.timings.nonorthogonalwicks.lg_h2_diff_gen,
        res.timings.nonorthogonalwicks.lg_h2_diff,
        5,
    );
    print_relative_counter(
        "Different-spin two-electron matrix elements (m = 0)",
        res.timings.nonorthogonalwicks.lg_h2_diff_m0,
        res.timings.nonorthogonalwicks.lg_h2_diff,
        5,
    );
    print_relative_counter(
        "Different-spin two-electron matrix elements (m = 0, generic)",
        res.timings.nonorthogonalwicks.lg_h2_diff_m0_gen,
        res.timings.nonorthogonalwicks.lg_h2_diff_m0,
        8,
    );
    print_relative_counter(
        "Different-spin two-electron matrix elements (m = 0, la = 1, lb = 1)",
        res.timings.nonorthogonalwicks.lg_h2_diff_m0_11,
        res.timings.nonorthogonalwicks.lg_h2_diff_m0,
        8,
    );
    print_relative_counter(
        "Different-spin two-electron matrix elements (m = 0, la = 1, lb = 3)",
        res.timings.nonorthogonalwicks.lg_h2_diff_m0_13,
        res.timings.nonorthogonalwicks.lg_h2_diff_m0,
        8,
    );
    print_relative_counter(
        "Different-spin two-electron matrix elements (m = 0, la = 2, lb = 2)",
        res.timings.nonorthogonalwicks.lg_h2_diff_m0_22,
        res.timings.nonorthogonalwicks.lg_h2_diff_m0,
        8,
    );
    print_relative_counter(
        "Different-spin two-electron matrix elements (m = 0, la = 3, lb = 1)",
        res.timings.nonorthogonalwicks.lg_h2_diff_m0_31,
        res.timings.nonorthogonalwicks.lg_h2_diff_m0,
        8,
    );

    println!("{}", "-".repeat(100));

    println!("R: {}", res.r);

    let ref_energy = if res.states.is_empty() {
        let hprint: Vec<&HSCFState> = res
            .hstates
            .iter()
            .filter(|st| hlabels.is_empty() || hlabels.contains(&st.label.as_str()))
            .collect();

        for (i, st) in hprint.iter().enumerate() {
            println!(
                "State({}): {},  E: {} + {}i",
                i + 1,
                report_label(&st.label, st.noci_basis),
                st.e.re,
                st.e.im
            );
        }

        hprint
            .first()
            .map(|st| (format!("Re(E({}))", st.label), st.e.re))
    } else {
        for (i, st) in res.states.iter().enumerate() {
            println!(
                "State({}): {},  E: {}",
                i + 1,
                report_label(&st.label, st.noci_basis),
                st.e
            );
        }

        let hprint: Vec<&HSCFState> = res
            .hstates
            .iter()
            .filter(|st| hlabels.contains(&st.label.as_str()))
            .collect();

        for (i, st) in hprint.iter().enumerate() {
            println!(
                "State({}): {},  E: {} + {}i",
                res.states.len() + i + 1,
                report_label(&st.label, st.noci_basis),
                st.e.re,
                st.e.im
            );
        }

        Some(("E(RHF)".to_string(), res.e_rhf))
    };

    let (label, e0) = ref_energy
        .as_ref()
        .expect("Reference energy should exist when printing report.");

    println!(
        "State(NOCI-reference): E: {}, [E - {}]: {}",
        res.e_noci_ref,
        label,
        res.e_noci_ref - e0
    );

    if !res.hstates.is_empty() {
        if let Some(e_snoci) = res.e_snoci {
            println!(
                "State(SNOCI): E: {}, [E - {}]: {}",
                e_snoci,
                label,
                e_snoci - e0
            );
        }

        if let (Some(e_snoci), Some(ept2)) = (res.e_snoci, res.e_pt2.as_ref()) {
            for &(imag_shift, ept2) in ept2 {
                let e_noci_pt2 = e_snoci + ept2;
                println!(
                    "State(NOCI-PT2 i: {}): E: {}, [E - {}]: {}",
                    imag_shift,
                    e_noci_pt2,
                    label,
                    e_noci_pt2 - e0
                );
            }
        }

        if let Some(e_fci) = res.e_fci {
            println!("State(FCI): E: {}, [E - {}]: {}", e_fci, label, e_fci - e0);
        }

        println!("{}", "=".repeat(100));
        return;
    }

    if let Some(e_det) = res.e_noci_qmc_det {
        println!(
            "State(NOCI-qmc-deterministic): E: {}, [E - {}]: {}",
            e_det,
            label,
            e_det - e0
        );
    }

    if res.e_noci_qmc_stoch.is_some() {
        println!("State(NOCI-qmc-qmc): Blocking analysis must be performed");
    }

    if let Some(e_snoci) = res.e_snoci {
        println!(
            "State(SNOCI): E: {}, [E - {}]: {}",
            e_snoci,
            label,
            e_snoci - e0
        );
    }

    if let (Some(e_snoci), Some(ept2)) = (res.e_snoci, res.e_pt2.as_ref()) {
        for &(imag_shift, ept2) in ept2 {
            let e_noci_pt2 = e_snoci + ept2;
            println!(
                "State(NOCI-PT2: i: {}): E: {}, [E - {}]: {}",
                imag_shift,
                e_noci_pt2,
                label,
                e_noci_pt2 - e0
            );
        }
    }

    if let Some(e_fci) = res.e_fci {
        println!("State(FCI): E: {},  [E - {}]: {}", e_fci, label, e_fci - e0);
    }

    println!("{}", "=".repeat(100));
}
