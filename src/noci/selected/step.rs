// noci/selected/step.rs

// Standard library imports.
use std::collections::HashMap;

// External crate imports.
use mpi::topology::Communicator;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

// Crate-root imports.
use crate::NOCIScalar;
use crate::determinant::Bits;
use crate::determinant::{NOCIIndex, NOCISpace};
use crate::elements::nonorthogonalwicks::WicksShared;
use crate::elements::{FockData, NOCIData};
use crate::elements::{build_fock_mo_cache, noci_density, update_wicks_fock};
use crate::input::{SNOCIPreconditioner, SNOCIStorage, StateType};
use crate::maths::general_evp;
use crate::maths::gmres::{GMRESPrint, gmres};
use crate::noci::{OneBodyFactorisation, OneBodyScratch};
use crate::scf::fock;
use crate::time_call;
use crate::{PostSCFData, input::Input};

// Parent/sibling imports.
use super::guess::FollowedStates;
use super::{
    CandidatePool, PT2Guess, PT2ProjectedOperator, SNOCIBlocks, SNOCIPT2Result, SNOCIState,
};
use super::{
    apply_factorised_omega_s, apply_factorised_shifted_omega_m,
    apply_factorised_shifted_omega_m_mpi, apply_omega_s, apply_omega_s_mpi, apply_shifted_omega_m,
    apply_shifted_omega_m_mpi, build_candidate_m, build_candidate_m_diag, build_candidate_m_disk,
    build_candidate_s_diag, build_factorised_candidate_diags, build_preconditioner,
    build_snoci_couplings, select_candidates,
};

/// Return the real spin string of a scalar used for printed and stored energies.
/// # Arguments:
/// - `z`: Scalar value.
/// # Returns:
/// - `f64`: Real spin string.
fn scalar_real<T: NOCIScalar + Into<Complex64>>(z: T) -> f64 {
    z.into().re
}

/// Return a SNOCI state with empty selected, candidate score, and EPT2 fields.
/// # Arguments:
/// - `ecurrent`: Current NOCI energy.
/// - `coeffs`: Current-space ground-state eigenvector.
/// - `hcurrent`: Hamiltonian matrix in the current space.
/// - `scurrent`: Overlap matrix in the current space.
/// # Returns:
/// - `SNOCIState`: SNOCI state with empty selected determinants, empty candidate scores, and
///   zero EPT2 correction.
fn empty_state<T: NOCIScalar>(
    ecurrent: f64,
    coeffs: Array1<T>,
    hcurrent: Array2<T>,
    scurrent: Array2<T>,
) -> SNOCIState<T> {
    SNOCIState {
        ecurrent,
        coeffs,
        hcurrent,
        scurrent,
        selected: Vec::new(),
        pt2: Vec::new(),
        guess: PT2Guess::default(),
    }
}

/// Print the start of a SNOCI iteration block.
/// # Arguments:
/// - `it`: SNOCI iteration index.
/// - `n_current`: Number of determinants in the current selected space.
/// - `npoolpre`: Candidate-pool size before projected-norm filter.
/// - `npoolpost`: Candidate-pool size after projected-norm filter.
/// # Returns:
/// - `()`: Prints the SNOCI iteration block header to standard output.
fn print_snoci_iteration_start(
    it: usize,
    n_current: usize,
    npoolpre: usize,
    npoolpost: usize,
) {
    println!("SNOCI iteration: {}", it);
    println!("  NCurr:     {}", n_current);
    println!("  NCand (R): {}", npoolpre);
    println!("  NCand:     {}", npoolpost);
}

/// Print the status message for building the cached shifted Fock matrix.
/// # Arguments:
/// - `n`: Number of candidates.
/// # Returns:
/// - `()`: Prints the shifted Fock build message to standard output.
fn print_build_candidate_m<T: NOCIScalar>(n: usize) {
    let nelem = n * (n + 1) / 2;
    let mib = nelem as f64 * std::mem::size_of::<T>() as f64 / 1024.0 / 1024.0;
    println!(
        "  Building upper triangle shifted Fock matrix ({} elements, {:.3} MiB)...",
        nelem, mib
    );
}

/// Print candidate full-M storage required for a packed upper-triangular matrix.
/// # Arguments:
/// - `n`: Number of candidates.
/// # Returns:
/// - `()`: Prints the packed full-M storage estimate to standard output.
fn print_candidate_m_storage<T: NOCIScalar>(n: usize) {
    let nelem = n
        .checked_mul(n + 1)
        .and_then(|x| x.checked_div(2))
        .expect("packed candidate matrix length overflow");
    let mib = nelem as f64 * std::mem::size_of::<T>() as f64 / 1024.0 / 1024.0;
    println!("  Estimated storage required for full_m (MiB): {:.3}", mib);
}

/// Print spin-factorised one-body factor-table storage required for this candidate basis.
/// # Arguments:
/// - `data`: Shared candidate NOCI data defining spin-string topology.
/// - `fock`: Current generalised-Fock data identifying orthogonal same-parent shortcuts.
/// # Returns:
/// - `()`: Prints the raw factor-table storage estimate to standard output.
fn print_factor_table_storage<T: NOCIScalar>(
    data: &NOCIData<'_, T>,
    fock: &FockData<'_, T>,
) {
    let nbytes = OneBodyFactorisation::storage_bytes(data, fock);
    let mib = nbytes as f64 / 1024.0 / 1024.0;
    println!(
        "  Estimated storage required for factor_tables (MiB): {:.3}",
        mib
    );
}

/// Print the SNOCI result for a completed iteration.
/// # Arguments:
/// - `it`: SNOCI iteration index.
/// - `n_current`: Number of determinants in the current selected space.
/// - `e0`: RHF reference energy used to define the correlation energy.
/// - `state`: SNOCI state for the current iteration.
/// # Returns:
/// - `()`: Prints the SNOCI iteration result to standard output.
fn print_snoci_iteration_result<T: NOCIScalar>(
    it: usize,
    n_current: usize,
    e0: f64,
    state: &SNOCIState<T>,
) {
    println!();
    println!("  SNOCI result");
    println!("  {}", "-".repeat(98));
    println!("  Iteration:          {}", it);
    println!("  NCurr:              {}", n_current);
    println!("  NSelect:            {}", state.selected.len());
    println!("  E:                  {:.12}", state.ecurrent);
    println!("  Ecorr:              {:.12}", state.ecurrent - e0);

    for r in &state.pt2 {
        println!();
        println!("  NOCI-PT2 result");
        println!("  NOCI state:         {}", r.root);
        println!("  <S^2>:              {:.12}", r.s2);
        println!("  Imag shift:         {:.12}", r.imag_shift);
        println!("  EPT2:               {:.12}", r.ept2);
        println!("  E + EPT2:           {:.12}", r.e_noci + r.ept2);
        println!("  <Psi1|Psi1>:        {:.12}", r.norm_first_order);
        println!("  ErPT2:              {:.12}", r.erpt2);
        println!("  E + ErPT2:          {:.12}", r.e_noci + r.erpt2);
        println!("  Max |a_i|:          {:.12}", r.max_abs_a);
        println!("  Max |v_i|:          {:.12}", r.max_abs_v);
        println!("  Max |a_i v_i|:      {:.12}", r.max_abs_av);
        println!("  GMRES residual:     {:.12}", r.gmres_residual);
        println!("  GMRES iterations:   {}", r.gmres_iterations);
        println!("  GMRES converged:    {}", r.gmres_converged);
        println!("  {}", "-".repeat(98));
    }
}

/// Solve the state-specific NOCI-PT2 problem `M^Omega(epsilon) a(epsilon) = -V^Omega` for one
/// NOCI state of the current space at every imaginary shift.
/// The generalised Fock operator, `E_0`, the projector `Q = 1 - |\Psi\rangle\langle\Psi|` and
/// `V^Omega` are all built from the chosen state, so each state needs its own Fock-dependent
/// intermediates and GMRES solve. The candidate-selected `H_{ai}` and `S_{ai}` do not depend on the
/// state and are passed in.
/// # Arguments:
/// - `post`: Data shared by post-SCF methods.
/// - `input`: User-defined input options.
/// - `spaces`: Determinant space, current selected space, candidate determinants and the candidate
///   subset of the determinant space.
/// - `blocks`: State-independent Hamiltonian and overlap blocks of the selected space.
/// - `target`: Position in `snoci.roots`, energy-order index, NOCI energy and current-space
///   eigenvector of the corrected NOCI state.
/// - `wicks`: Mutable Wick's intermediates, whose Fock intermediates are rebuilt for this state.
/// - `run`: MPI communicator, SNOCI iteration index and the converged amplitudes of the previous
///   SNOCI iteration, or of the previous geometry on the first iteration, the GMRES initial
///   guesses.
/// # Returns:
/// - `Vec<SNOCIPT2Result>`: NOCI-PT2 results for this state, one per imaginary shift.
fn solve_pt2_state<T, R, C>(
    post: &PostSCFData<'_, T>,
    input: &Input,
    spaces: (&NOCISpace<T>, &[NOCIIndex], &[NOCIIndex], &NOCISpace<T>),
    blocks: &SNOCIBlocks<T>,
    target: (usize, usize, f64, &Array1<T>),
    mut wicks: Option<&mut WicksShared<T>>,
    run: (&C, usize, &PT2Guess),
) -> Vec<SNOCIPT2Result>
where
    T: NOCIScalar + Into<Complex64>,
    R: NOCIScalar + From<T> + Into<Complex64>,
    C: Communicator,
{
    let opts = input.snoci.as_ref().unwrap();
    let (space, selected_space, candidates, candidate_basis) = spaces;
    let (slot, root, e_noci, coeffs) = target;
    let (world, it, previous) = run;

    // `\langle S^2\rangle = c^\dagger \langle i|S^2|j\rangle c / c^\dagger S c` identifies the spin of the
    // corrected state in the printed results.
    let coeffs_conj = coeffs.mapv(|x| x.conj());
    let ss_c = coeffs_conj.dot(&blocks.ss_ii.dot(coeffs));
    let s_c = coeffs_conj.dot(&blocks.s_ii.dot(coeffs));
    let s2 = scalar_real(ss_c) / scalar_real(s_c);
    if world.rank() == 0 {
        println!();
        println!("  NOCI state:         {}", root);
        println!("  E:                  {:.12}", e_noci);
        println!("  <S^2>:              {:.12}", s2);
    }

    // Form multireference NOCI density and generalised AO Focks.
    let (da, db) = {
        let wview = wicks.as_ref().map(|ws| ws.view());
        let data = NOCIData::new(post.ao, space, input, post.tol, wview);
        noci_density(&data, selected_space, coeffs)
    };
    let (fa, fb) = time_call!(crate::timers::selected::add_build_generalised_fock, {
        fock(&post.ao.h, &post.ao.eri_coul, &da, &db)
    });

    // Transform Focks into MO basis for each reference.
    let fock_mocache = build_fock_mo_cache(&fa, &fb, &post.space.parents, &post.ao.s, post.tol);

    // Update the Wick's intermediates if using them.
    if input.wicks.enabled
        && let Some(ws) = wicks.as_deref_mut()
    {
        update_wicks_fock(&fa, &fb, &post.space.parents, &post.ao.s, post.tol, ws);
    }

    // Contract the candidate-current Hamiltonian, overlap and Fock blocks with the current state.
    let wview = wicks.as_ref().map(|ws| ws.view());
    let data = NOCIData::new(post.ao, space, input, post.tol, wview).withmocache(post.mocache);
    let fock = FockData::new(&fock_mocache, &fa, &fb);
    let (projection, v_omega) =
        build_snoci_couplings(&data, &fock, selected_space, candidates, coeffs, e_noci);

    // `M^Omega(epsilon) a(epsilon) = -V^Omega` is solved in scalar `R`.
    // Chemistry-derived projection vectors and `V^Omega` cross the scalar boundary only here.
    let krylov_projection = projection.cast::<R>();
    let v_omega_krylov = v_omega.mapv(<R as From<T>>::from);

    // PT2's numerical plan uses dense candidate-local indices into the candidate subset.
    let candidate_indices = (0..candidate_basis.len())
        .map(NOCIIndex)
        .collect::<Vec<_>>();
    let operator_data =
        NOCIData::new(post.ao, candidate_basis, input, post.tol, wview).withmocache(post.mocache);
    let op = PT2ProjectedOperator {
        data: &operator_data,
        fock: &fock,
        candidates: &candidate_indices,
        projection: &projection,
    };

    if world.rank() == 0 {
        print_candidate_m_storage::<T>(op.candidates.len());
        print_factor_table_storage(&operator_data, &fock);
    }

    // Without a stored operator, products go through the spin factorisation whenever every
    // parent pair can be served, with or without Wick intermediates.
    let one_body = if matches!(opts.full_m, SNOCIStorage::None) {
        let cache = input.wicks.cachedir.as_deref().unwrap_or(".");
        OneBodyFactorisation::new(
            &operator_data,
            &fock,
            std::path::Path::new(cache),
            world.rank(),
            it,
            opts.factor_tables,
        )
    } else {
        None
    };

    let m = match opts.full_m {
        SNOCIStorage::None => None,
        SNOCIStorage::RAM => {
            if world.rank() == 0 {
                print_build_candidate_m::<T>(op.candidates.len());
            }
            Some(build_candidate_m(&op))
        }
        SNOCIStorage::Disk => {
            if world.rank() == 0 {
                print_build_candidate_m::<T>(op.candidates.len());
            }
            let cache = input.wicks.cachedir.as_deref().unwrap_or(".");
            let path = std::path::Path::new(cache).join(format!(
                "snoci_m_rank{}_iter{}.bin",
                world.rank(),
                it
            ));
            Some(build_candidate_m_disk(&op, &path))
        }
    };

    let m_slice = m.as_ref().map(|m| m.as_slice());

    // Candidate diagonals are needed only for actual preconditioners.
    let (m_diag, factorised_s_diag) = match opts.preconditioner {
        SNOCIPreconditioner::None => (None, None),
        SNOCIPreconditioner::Diag | SNOCIPreconditioner::Woodbury => {
            if let Some(one_body) = one_body.as_ref() {
                let (m_diag, s_diag) = build_factorised_candidate_diags(
                    &op,
                    one_body,
                    T::from_real(-op.projection.e0),
                );
                (Some(m_diag), Some(s_diag))
            } else {
                (Some(build_candidate_m_diag(&op, m_slice)), None)
            }
        }
    };

    let shifts = if opts.imag_shifts.is_empty() {
        vec![0.0]
    } else {
        opts.imag_shifts.clone()
    };

    // Shifted diagonal data are also unnecessary when GMRES is unpreconditioned.
    let s_diag = if m_diag.is_some() && shifts.iter().any(|&imag_shift| imag_shift != 0.0) {
        factorised_s_diag.or_else(|| Some(build_candidate_s_diag(&op)))
    } else {
        None
    };

    let rhs = v_omega_krylov.mapv(|x| -x);
    let mut one_body_scratch: Option<OneBodyScratch<R>> = one_body
        .as_ref()
        .map(|factorisation| factorisation.scratch());

    // Evaluate NOCI-PT2 energies, scores and diagnostics for each imaginary shift.
    let mut pt2 = Vec::new();
    for &imag_shift in &shifts {
        let prec = m_diag.as_ref().map(|m_diag| {
            build_preconditioner(
                m_diag,
                s_diag.as_ref(),
                &krylov_projection,
                opts.preconditioner,
                imag_shift,
            )
        });

        let a = {
            let mut apply = |x: &Array1<R>| {
                if let (Some(one_body), Some(scratch)) =
                    (one_body.as_ref(), one_body_scratch.as_mut())
                {
                    if world.size() > 1 {
                        apply_factorised_shifted_omega_m_mpi(
                            &op,
                            &krylov_projection,
                            one_body,
                            scratch,
                            x,
                            world,
                            imag_shift,
                        )
                    } else {
                        apply_factorised_shifted_omega_m(
                            &op,
                            &krylov_projection,
                            one_body,
                            scratch,
                            x,
                            imag_shift,
                        )
                    }
                } else if world.size() > 1 {
                    apply_shifted_omega_m_mpi(
                        &op,
                        &krylov_projection,
                        x,
                        m_slice,
                        world,
                        imag_shift,
                    )
                } else {
                    apply_shifted_omega_m(&op, &krylov_projection, x, m_slice, imag_shift)
                }
            };

            // Start from the previous iteration's or geometry's amplitudes of this state and shift.
            let x0 = previous.continue_to(candidate_basis, slot, imag_shift, &rhs);

            let g = &opts.gmres;
            let print = if world.rank() == 0 {
                GMRESPrint::Hylleraas
            } else {
                GMRESPrint::Silent
            };
            time_call!(crate::timers::selected::add_gmres, {
                if let Some(prec) = prec.as_ref() {
                    gmres(
                        &mut apply,
                        |x| prec.apply(x),
                        (&rhs, x0.as_ref()),
                        g.restart,
                        g.max_iter,
                        g.res_tol,
                        print,
                    )
                } else {
                    gmres(
                        &mut apply,
                        |x| x.clone(),
                        (&rhs, x0.as_ref()),
                        g.restart,
                        g.max_iter,
                        g.res_tol,
                        print,
                    )
                }
            })
        };

        let ma = if let (Some(one_body), Some(scratch)) =
            (one_body.as_ref(), one_body_scratch.as_mut())
        {
            if world.size() > 1 {
                apply_factorised_shifted_omega_m_mpi(
                    &op,
                    &krylov_projection,
                    one_body,
                    scratch,
                    &a.x,
                    world,
                    imag_shift,
                )
            } else {
                apply_factorised_shifted_omega_m(
                    &op,
                    &krylov_projection,
                    one_body,
                    scratch,
                    &a.x,
                    imag_shift,
                )
            }
        } else if world.size() > 1 {
            apply_shifted_omega_m_mpi(&op, &krylov_projection, &a.x, m_slice, world, imag_shift)
        } else {
            apply_shifted_omega_m(&op, &krylov_projection, &a.x, m_slice, imag_shift)
        };

        let ama =
            a.x.iter()
                .zip(ma.iter())
                .fold(R::from_real(0.0), |acc, (&aa, &maa)| acc + aa.conj() * maa);

        let av =
            a.x.iter()
                .zip(v_omega_krylov.iter())
                .fold(R::from_real(0.0), |acc, (&aa, &v)| acc + aa.conj() * v);

        let va = v_omega_krylov
            .iter()
            .zip(a.x.iter())
            .fold(R::from_real(0.0), |acc, (&v, &aa)| acc + v.conj() * aa);
        let ept2 = scalar_real(ama + av + va);

        // `\langle\Psi^{(1)}|\Psi^{(1)}\rangle = a^\dagger Q a` renormalises the correction,
        // `E^{(2)}_{\text{r}} = E^{(2)} / (1 + \langle\Psi^{(1)}|\Psi^{(1)}\rangle)`.
        let qa = if let (Some(one_body), Some(scratch)) =
            (one_body.as_ref(), one_body_scratch.as_mut())
        {
            apply_factorised_omega_s(&op, &krylov_projection, one_body, scratch, &a.x, world)
        } else if world.size() > 1 {
            apply_omega_s_mpi(&op, &krylov_projection, &a.x, world)
        } else {
            apply_omega_s(&op, &krylov_projection, &a.x)
        };
        let aqa =
            a.x.iter()
                .zip(qa.iter())
                .fold(R::from_real(0.0), |acc, (&aa, &qaa)| acc + aa.conj() * qaa);
        let norm_first_order = scalar_real(aqa);
        let erpt2 = ept2 / (1.0 + norm_first_order);

        let candidate_scores: Vec<f64> =
            a.x.iter()
                .zip(v_omega_krylov.iter())
                .map(|(&a, &v)| (a * v).abs())
                .collect();

        let max_abs_a = a.x.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let max_abs_v = v_omega_krylov.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let max_abs_av = candidate_scores.iter().copied().fold(0.0, f64::max);

        pt2.push(SNOCIPT2Result {
            root,
            e_noci,
            s2,
            imag_shift,
            ept2,
            norm_first_order,
            erpt2,
            candidate_scores,
            amplitudes: a.x.mapv(Into::into),
            rhs: rhs.mapv(Into::into),
            max_abs_a,
            max_abs_v,
            max_abs_av,
            gmres_residual: a.residual_rms,
            gmres_iterations: a.iterations,
            gmres_converged: a.converged,
        });
    }

    pt2
}

/// Perform selected NOCI and solve
/// `M^Omega(epsilon) a(epsilon) = -V^Omega`.
/// Determinants, matrix elements, Fock data and Wick intermediates use chemistry scalar `T`.
/// The shifted linear system and Krylov vectors use scalar `R`.
/// # Arguments:
/// - `post`: Data shared by post-SCF methods.
/// - `current_space`: Current selected nonorthogonal determinant space.
/// - `input`: User-defined input options.
/// - `wicks`: Mutable Wick's intermediates as we need to update Fock intermediates.
/// - `previous`: Converged NOCI-PT2 amplitudes of the previous geometry, the GMRES initial
///   guesses.
/// - `world`: MPI communicator used to distribute NOCI-PT2 matrix-vector products.
/// # Returns:
/// - `SNOCIState`: Final SNOCI state from the last completed iteration.
pub fn snoci_step<T, R>(
    post: &PostSCFData<'_, T>,
    current_space: &NOCISpace<T>,
    input: &Input,
    mut wicks: Option<&mut WicksShared<T>>,
    previous: &PT2Guess,
    world: &impl Communicator,
) -> SNOCIState<T>
where
    T: NOCIScalar + Into<Complex64>,
    R: NOCIScalar + From<T> + Into<Complex64>,
{
    time_call!(crate::timers::selected::add_snoci_step, {
        let opts = input.snoci.as_ref().unwrap();

        let initial_indices = (0..current_space.len()).map(NOCIIndex).collect::<Vec<_>>();
        let mut space = current_space.subset(&initial_indices);
        let mut selected_space = (0..space.len()).map(NOCIIndex).collect::<Vec<_>>();

        let mut final_state: Option<SNOCIState<T>> = None;

        // The candidate pool and the state-independent selected-space blocks are built once and
        // extended as determinants are promoted.
        let mut pool = CandidatePool::new(&mut space, &selected_space, input);
        let mut blocks = {
            let wview = wicks.as_ref().map(|ws| ws.view());
            let data =
                NOCIData::new(post.ao, &space, input, post.tol, wview).withmocache(post.mocache);
            SNOCIBlocks::new(&data, &selected_space)
        };

        // States corrected at the previous geometry, followed through this one when enabled. A
        // holomorphic recipe continues its real partner, which it replaces in the NOCI basis.
        let mut followed = if opts.follow {
            previous.followed.clone()
        } else {
            FollowedStates::default()
        };
        let partners: HashMap<&str, &str> = match &input.states {
            StateType::Mom(recipes) => recipes
                .iter()
                .filter(|recipe| recipe.holomorphic)
                .filter_map(|recipe| Some((recipe.label.as_str(), recipe.partner.as_deref()?)))
                .collect(),
            StateType::Metadynamics(_) => HashMap::new(),
        };

        for it in 0..opts.max_iter {
            // Solve the current-space GEVP `H c = S c E` for every NOCI state.
            let (evals, evecs) = general_evp(&blocks.h_ii, &blocks.s_ii, true, post.tol);

            // Every requested state must exist after the overlap projection of the GEVP.
            if let Some(&root) = opts.roots.iter().find(|&&root| root >= evals.len()) {
                eprintln!(
                    "snoci.roots requests NOCI state {} but the current space has only {} states",
                    root,
                    evals.len()
                );
                std::process::exit(1);
            }

            // Correct the listed roots, or give each state corrected before the remaining
            // eigenvector of largest overlap `|c^{\text{prev}\dagger} S c_j|`.
            let mut roots = opts.roots.clone();
            let mut similarity = Vec::new();
            if opts.follow {
                // Identify every determinant by its parent, or the parent's partner, and its
                // replacements, so the overlap does not depend on the order of the NOCI basis.
                let keys: HashMap<(String, [Bits; 4]), usize> = selected_space
                    .iter()
                    .enumerate()
                    .map(|(pos, &index)| {
                        let label = space.parent_determinant(index).label.as_str();
                        let alpha = &space.alpha(index).data.excitation;
                        let beta = &space.beta(index).data.excitation;
                        let label = partners.get(label).copied().unwrap_or(label).to_string();
                        (
                            (label, [alpha.holes, alpha.parts, beta.holes, beta.parts]),
                            pos,
                        )
                    })
                    .collect();

                if !followed.coeffs.is_empty() {
                    let s = blocks.s_ii.mapv(Into::<Complex64>::into);
                    let sc = s.dot(&evecs.mapv(Into::<Complex64>::into));
                    roots.clear();
                    for prev in &followed.coeffs {
                        // The earlier state in the current determinants, zero where absent.
                        let mut p = Array1::<Complex64>::zeros(keys.len());
                        for (key, &pos) in &keys {
                            if let Some(&i) = followed.keys.get(key) {
                                p[pos] = prev[i];
                            }
                        }

                        // `|p^\dagger S c_j| / \sqrt{p^\dagger S p}`, at most one for normalised `c_j`.
                        let pdag = p.mapv(|x| x.conj());
                        let norm = pdag.dot(&s.dot(&p)).norm().sqrt();
                        let overlaps = pdag.dot(&sc).mapv(|x| x.norm() / norm);

                        let root = (0..evecs.ncols())
                            .filter(|j| !roots.contains(j))
                            .max_by(|&a, &b| overlaps[a].total_cmp(&overlaps[b]))
                            .unwrap();
                        roots.push(root);
                        similarity.push(overlaps[root]);
                    }
                }

                followed = FollowedStates {
                    keys,
                    coeffs: roots
                        .iter()
                        .map(|&root| evecs.column(root).mapv(Into::into))
                        .collect(),
                };
            }

            // The first listed state drives candidate selection and convergence.
            let ecurrent = evals[roots[0]];
            let coeffs = evecs.column(roots[0]).to_owned();

            let hcurrent = blocks.h_ii.clone();
            let scurrent = blocks.s_ii.clone();

            let npoolpre = pool.candidates.len();
            let npoolpost = pool.candidates.len();
            if pool.candidates.is_empty() {
                return empty_state(ecurrent, coeffs, hcurrent, scurrent);
            }

            if it > 0 && world.rank() == 0 {
                println!("{}", "=".repeat(100));
            }

            if world.rank() == 0 {
                print_snoci_iteration_start(it, selected_space.len(), npoolpre, npoolpost);
                for (slot, (root, overlap)) in roots.iter().zip(&similarity).enumerate() {
                    println!(
                        "  Followed state {}:   NOCI state {}, overlap {:.6}",
                        slot, root, overlap
                    );
                }
            }

            // PT2's numerical plan uses dense candidate-local indices. This temporary
            // NOCISpace subset shares parent frames with the selection topology and does not
            // depend on the corrected state.
            let candidate_basis = space.subset(&pool.candidates);

            // Each listed NOCI state receives its own state-specific NOCI-PT2 correction. Only a few
            // determinants move between iterations, so GMRES starts from the previous iteration's
            // amplitudes, or from the previous geometry's on the first iteration.
            let initial = final_state.as_ref().map_or(previous, |state| &state.guess);
            let mut pt2 = Vec::new();
            for (slot, &root) in roots.iter().enumerate() {
                let state_coeffs = evecs.column(root).to_owned();
                pt2.extend(solve_pt2_state::<T, R, _>(
                    post,
                    input,
                    (&space, &selected_space, &pool.candidates, &candidate_basis),
                    &blocks,
                    (slot, root, evals[root], &state_coeffs),
                    wicks.as_deref_mut(),
                    (world, it, initial),
                ));
            }
            let guess = PT2Guess::new(&candidate_basis, &pt2, &roots, followed.clone());

            let remaining = opts.max_dim.saturating_sub(selected_space.len());
            // Every rank stops once the selected space is full; rank zero reports the final result.
            if remaining == 0 {
                let state = SNOCIState {
                    ecurrent,
                    coeffs,
                    hcurrent,
                    scurrent,
                    selected: Vec::new(),
                    pt2,
                    guess,
                };
                if world.rank() == 0 {
                    print_snoci_iteration_result(
                        it,
                        selected_space.len(),
                        scalar_real(post.space.parents[0].e),
                        &state,
                    );
                    println!(
                        "SNOCI stopped at iteration {}: selected space reached max_dim ({}).",
                        it, opts.max_dim
                    );
                }
                return state;
            }

            // Use the final imaginary shift of the first listed state as the main result for selection.
            let main_pt2 = pt2
                .iter()
                .rfind(|r| r.root == roots[0])
                .expect("At least one NOCI-PT2 shift must be evaluated.");

            // Add `\max(\text{max\_add}, \lceil \text{growth}\, N_{\text{sel}} \rceil)` determinants,
            // so `growth = 1` doubles the selected space every iteration.
            let grown = (opts.growth * selected_space.len() as f64).ceil() as usize;
            let selected = select_candidates(
                &pool.candidates,
                &main_pt2.candidate_scores,
                opts.sigma,
                opts.max_add.max(grown).min(remaining),
            );
            let state = SNOCIState {
                ecurrent,
                coeffs,
                hcurrent,
                scurrent,
                selected,
                pt2,
                guess,
            };

            if world.rank() == 0 {
                print_snoci_iteration_result(
                    it,
                    selected_space.len(),
                    scalar_real(post.space.parents[0].e),
                    &state,
                );
            }

            if state.selected.is_empty() {
                if world.rank() == 0 {
                    println!(
                        "SNOCI stopped at iteration {}: no candidates satisfied the selection threshold ({}).",
                        it, opts.sigma
                    );
                }
                return state;
            }

            let main_pt2 = state
                .pt2
                .iter()
                .rfind(|r| r.root == roots[0])
                .expect("At least one NOCI-PT2 shift must be evaluated.");

            if main_pt2.ept2.abs() < opts.tol {
                if world.rank() == 0 {
                    println!(
                        "SNOCI stopped at iteration {}: |EPT2|: {:.12} fell below tolerance {:.12}.",
                        it,
                        main_pt2.ept2.abs(),
                        opts.tol
                    );
                }
                return state;
            }

            // Promote the selected candidates, regenerate the pool and extend the blocks.
            selected_space.extend(state.selected.iter().copied());
            pool.update(&mut space, &state.selected, input);

            let wview = wicks.as_ref().map(|ws| ws.view());
            let data =
                NOCIData::new(post.ao, &space, input, post.tol, wview).withmocache(post.mocache);
            blocks.extend(&data, &selected_space, &state.selected);

            final_state = Some(state);
        }

        if world.rank() == 0 {
            println!(
                "SNOCI stopped: Maximum iteration was reached ({}).",
                opts.max_iter
            );
        }

        final_state.unwrap_or_else(|| {
            let (evals, evecs) = general_evp(&blocks.h_ii, &blocks.s_ii, true, post.tol);
            let coeffs = evecs.column(opts.roots[0]).to_owned();
            empty_state(evals[opts.roots[0]], coeffs, blocks.h_ii, blocks.s_ii)
        })
    })
}
