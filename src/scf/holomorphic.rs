// scf/holomorphic.rs

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{Array1, Array2, Axis, s};
use ndarray_linalg::Solve;
use num_complex::Complex64;

use crate::input::{HSCFOptions, Input, StateRecipe};
use crate::scf::DensityMode;
use crate::{AoData, Excitation, HSCFState, SCFState};

use super::occupation::spin_occupation;
use super::print::print_header_h;
use super::{density, energy, fock, orbital_energies, orbital_gradient};
use crate::maths::{
    complex_metric_orthonormalize, matrix_exp_complex, real2_as, symmetric_evp_complex,
};
use crate::utils::print_array2_indexed;

/// Stored quasi-Newton secant pair in the current local tangent basis.
#[derive(Clone, Debug)]
struct SecantPair {
    /// Previous alpha-spin accepted step in unweighted occupied-virtual rotation coordinates.
    sa: Array2<Complex64>,
    /// Previous beta-spin accepted step in unweighted occupied-virtual rotation coordinates.
    sb: Array2<Complex64>,
    /// Previous alpha-spin gradient change in unweighted occupied-virtual coordinates.
    ya: Array2<Complex64>,
    /// Previous beta-spin gradient change in unweighted occupied-virtual coordinates.
    yb: Array2<Complex64>,
}

/// Current- and previous-geometry determinant states keyed by label.
pub struct StateLookups<'a, T> {
    /// Current-geometry states keyed by label.
    pub current: &'a HashMap<&'a str, &'a T>,
    /// Previous-geometry states keyed by label.
    pub previous: &'a HashMap<&'a str, &'a T>,
}

/// Lookup tables used while constructing h-SCF states at one geometry.
pub struct HSCFGenerationLookups<'a> {
    /// State recipes keyed by label.
    pub recipes: &'a HashMap<&'a str, &'a StateRecipe>,
    /// Real SCF states used as h-SCF seeds.
    pub real: StateLookups<'a, SCFState>,
    /// h-SCF states used for continuation and spin-flip reuse.
    pub h: StateLookups<'a, HSCFState>,
}

/// Spin block being pseudo-canonicalised.
#[derive(Clone, Copy, Debug)]
enum SpinBlock {
    /// Alpha-spin orbital block.
    Alpha,
    /// Beta-spin orbital block.
    Beta,
}

/// Outcome of checking whether a holomorphic recipe should use its real partner seed.
enum PartnerSeed<'a> {
    /// Recipe has no partner gate.
    NoPartner,
    /// Partner exists and collapsed out of the NOCI basis, so h-SCF should be attempted.
    Use(&'a SCFState),
    /// Partner gate was present, but h-SCF should not be attempted at this geometry.
    Skip,
}

/// Build one h-SCF state, including partner gating and spin-flip reuse.
/// # Arguments
/// - `ao`: AO data containing integrals and electron counts.
/// - `input`: User input controlling SCF and h-SCF.
/// - `recipes`: All state recipes.
/// - `recipe`: Holomorphic recipe being generated.
/// - `lookups`: Recipe, real-state, and h-state lookup tables.
/// - `i`: Current recipe index.
/// - `fallback`: Callback that generates a real MOM seed when no generated real seed exists.
/// # Returns
/// - `Option<HSCFState>`: Generated h-SCF state, or `None` when a partner gate says to skip it.
pub fn build_hscf_state<F>(
    ao: &AoData,
    input: &Input,
    recipes: &[StateRecipe],
    recipe: &StateRecipe,
    lookups: HSCFGenerationLookups<'_>,
    i: usize,
    fallback: F,
) -> Option<HSCFState>
where
    F: FnOnce() -> SCFState,
{
    let partner_seed = hscf_partner_seed(recipe, input, lookups.real.current, lookups.recipes);
    if matches!(partner_seed, PartnerSeed::Skip) {
        return None;
    }

    if let Some(source) = hscf_spin_flip_source(ao, recipes, recipe, i, lookups.h.current) {
        if input.write.verbose {
            println!(
                "Constructing h-SCF state '{}' by spin flip of '{}'.",
                recipe.label, source.label
            );
        }

        // The opposite spin-bias partner is obtained by swapping alpha and beta orbitals,
        // densities, occupations, and phases.
        return Some(HSCFState {
            e: source.e,
            oa: source.ob,
            ob: source.oa,
            pha: source.phb,
            phb: source.pha,
            ca: source.cb.clone(),
            cb: source.ca.clone(),
            da: source.db.clone(),
            db: source.da.clone(),
            label: recipe.label.to_string(),
            noci_basis: recipe.noci,
            parent: i,
            excitation: source.excitation.clone(),
        });
    }

    let fallback_seed;
    let seed_label = recipe.partner.as_deref().unwrap_or(recipe.label.as_str());

    let seed = match partner_seed {
        PartnerSeed::Use(current_partner) => {
            if let Some(previous_partner) = recipe
                .partner
                .as_deref()
                .and_then(|label| lookups.real.previous.get(label).copied())
            {
                if input.write.verbose {
                    println!(
                        "Seeding h-SCF state '{}' from previous real partner '{}'.",
                        recipe.label, seed_label
                    );
                }
                previous_partner
            } else {
                if input.write.verbose {
                    println!(
                        "Seeding h-SCF state '{}' from current collapsed partner '{}'.",
                        recipe.label, seed_label
                    );
                }
                current_partner
            }
        }
        PartnerSeed::NoPartner | PartnerSeed::Skip => {
            if let Some(st) = lookups.real.current.get(seed_label).copied() {
                st
            } else {
                fallback_seed = fallback();
                &fallback_seed
            }
        }
    };

    Some(run_hscf_state(
        ao,
        input,
        recipe,
        seed,
        lookups.h.previous,
        i,
    ))
}

/// Run h-SCF continuation and fresh-seed attempts for one recipe.
/// # Arguments
/// - `ao`: AO data containing integrals and overlap.
/// - `input`: User input controlling h-SCF options and printing.
/// - `recipe`: Holomorphic recipe being optimized.
/// - `seed`: Real SCF state used for fresh h-SCF seeding.
/// - `prev_h`: Previous h-SCF states keyed by label.
/// - `i`: Recipe index used as the parent/state index.
/// # Returns
/// - `HSCFState`: Selected converged h-SCF state.
fn run_hscf_state(
    ao: &AoData,
    input: &Input,
    recipe: &StateRecipe,
    seed: &SCFState,
    prev_h: &HashMap<&str, &HSCFState>,
    i: usize,
) -> HSCFState {
    let mut candidates: Vec<HSCFState> = Vec::new();

    // Try analytic continuation from the previous geometry.
    if let Some(st) = prev_h.get(recipe.label.as_str()).copied() {
        if input.write.verbose {
            println!(
                "Seeding h-SCF state '{}' from previous complex geometry.",
                recipe.label
            );
        }

        let ca = complex_metric_orthonormalize(&st.ca, &ao.s);
        let cb = complex_metric_orthonormalize(&st.cb, &ao.s);

        if let Some(hst) = hscf_cycle(&ca, &cb, ao, input, &recipe.label, recipe.noci, i) {
            candidates.push(hst);
        } else if input.write.verbose {
            println!(
                "Previous-geometry h-SCF seed for '{}' did not converge.",
                recipe.label
            );
        }
    }

    // Also try the fresh h-SCF seed. This is important because the continuation seed can
    // collapse onto the real branch near a coalescence, while the imaginary kick can still
    // recover the holomorphic branch.
    if recipe.spin_bias.is_some() || recipe.spatial_bias.is_some() {
        let (ca, cb) = h_seed_orbitals(seed, recipe, ao);

        if let Some(hst) = hscf_cycle(&ca, &cb, ao, input, &recipe.label, recipe.noci, i) {
            candidates.push(hst);
        } else if input.write.verbose {
            println!("Fresh h-SCF seed for '{}' did not converge.", recipe.label);
        }
    } else {
        let (ca, cb) = complex_orbitals_from_real(seed, ao);

        if let Some(hst) = hscf_cycle(&ca, &cb, ao, input, &recipe.label, recipe.noci, i) {
            candidates.push(hst);
        } else if input.write.verbose {
            println!(
                "Real-state h-SCF seed for '{}' did not converge.",
                recipe.label
            );
        }
    }

    select_hscf_candidate(recipe, candidates)
}

/// Run a holomorphic unrestricted SCF quasi-Newton optimisation.
/// # Arguments:
/// - `ca0`: Initial alpha-spin MO coefficients ordered as occupied then virtual.
/// - `cb0`: Initial beta-spin MO coefficients ordered as occupied then virtual.
/// - `ao`: Contains AO integrals and metadata.
/// - `input`: User input specifications.
/// - `label`: Label for the h-SCF state.
/// - `noci_basis`: Whether this state is intended for a later NOCI basis.
/// - `i`: State index.
/// # Returns:
/// - `Option<HSCFState>`: Converged h-SCF state if optimisation succeeds.
pub fn hscf_cycle(
    ca0: &Array2<Complex64>,
    cb0: &Array2<Complex64>,
    ao: &AoData,
    input: &Input,
    label: &str,
    noci_basis: bool,
    i: usize,
) -> Option<HSCFState> {
    let na = usize::try_from(ao.nelec[0]).unwrap();
    let nb = usize::try_from(ao.nelec[1]).unwrap();
    let opts = &input.scf.h;
    let mut ca = ca0.clone();
    let mut cb = cb0.clone();
    let mut hist: Vec<SecantPair> = Vec::new();

    print_header_h(input, label);

    let mut g_prev: Option<(Array2<Complex64>, Array2<Complex64>)> = None;
    let mut step_prev: Option<(Array2<Complex64>, Array2<Complex64>)> = None;
    let mut best_gnorm = f64::INFINITY;
    let mut stagnant = 0usize;

    for iter in 0..opts.max_cycle {
        let da = density(&ca, na, DensityMode::Holomorphic);
        let db = density(&cb, nb, DensityMode::Holomorphic);
        let (fa, fb) = fock(&ao.h, &ao.eri_coul, &da, &db);

        let mut extra_a: Vec<&mut Array2<Complex64>> = Vec::new();
        if let Some((sa, _)) = step_prev.as_mut() {
            extra_a.push(sa);
        }
        if let Some((ga, _)) = g_prev.as_mut() {
            extra_a.push(ga);
        }
        pseudo_canonicalise(&mut ca, &fa, na, &mut hist, SpinBlock::Alpha, &mut extra_a);

        let mut extra_b: Vec<&mut Array2<Complex64>> = Vec::new();
        if let Some((_, sb)) = step_prev.as_mut() {
            extra_b.push(sb);
        }
        if let Some((_, gb)) = g_prev.as_mut() {
            extra_b.push(gb);
        }
        pseudo_canonicalise(&mut cb, &fb, nb, &mut hist, SpinBlock::Beta, &mut extra_b);

        let da = density(&ca, na, DensityMode::Holomorphic);
        let db = density(&cb, nb, DensityMode::Holomorphic);
        let (fa, fb) = fock(&ao.h, &ao.eri_coul, &da, &db);
        let e = energy(&ao.h, ao.enuc, &da, &db, &fa, &fb);
        let epsa = orbital_energies(&ca, &fa, DensityMode::Holomorphic);
        let epsb = orbital_energies(&cb, &fb, DensityMode::Holomorphic);

        // Calculate g_{ai} = 2 \sum_{\mu\nu} C_a^\mu F_{\mu\nu} C_i^\nu.
        let ga = orbital_gradient(&ca, &fa, na, DensityMode::Holomorphic);
        let gb = orbital_gradient(&cb, &fb, nb, DensityMode::Holomorphic);

        // Use the Euclidean Frobenius norm only as a real convergence diagnostic.
        let gnorm = (ga.iter().map(|z| z.norm_sqr()).sum::<f64>()
            + gb.iter().map(|z| z.norm_sqr()).sum::<f64>())
        .sqrt();

        if gnorm < opts.g_tol {
            if input.write.verbose {
                println!(
                    "{:4} {:16.10} {:+16.10}i {:12.4e} {:>12} {:>12}",
                    iter, e.re, e.im, gnorm, "-", "-"
                );
            }
            return Some(finalise(ca, cb, ao, input, label, noci_basis, i));
        }

        if let (Some((sa, sb)), Some((gpa, gpb))) = (step_prev.take(), g_prev.take()) {
            let ya = &ga - &gpa;
            let yb = &gb - &gpb;

            hist.push(SecantPair { sa, sb, ya, yb });
            if hist.len() > opts.history {
                hist.remove(0);
            }
        }

        if gnorm < best_gnorm * 0.95 {
            best_gnorm = gnorm;
            stagnant = 0;
        } else {
            stagnant += 1;
        }

        let use_fd_newton = stagnant >= 8;
        if use_fd_newton && !hist.is_empty() {
            hist.clear();
            stagnant = 0;
            if input.write.verbose {
                println!("h-SCF progress stalled; using finite-difference Newton rescue step.");
            }
        }

        let (mut pa, mut pb) = if use_fd_newton {
            finite_difference_newton_step(&ca, &cb, ao, na, nb, &ga, &gb)
                .unwrap_or_else(|| sr1_step(&hist, (&ga, &gb), (&epsa, &epsb), (na, nb), opts))
        } else {
            sr1_step(&hist, (&ga, &gb), (&epsa, &epsb), (na, nb), opts)
        };

        limit_step(&mut pa, &mut pb, opts.max_step);
        let pnorm = step_norm(&pa, &pb);

        let (alpha, ca_new, cb_new) =
            line_search((&ca, &cb), ao, (na, nb), (&pa, &pb), gnorm, opts);

        if input.write.verbose {
            println!(
                "{:4} {:16.10} {:+16.10}i {:12.4e} {:12.4e} {:12.4e}",
                iter, e.re, e.im, gnorm, alpha, pnorm
            );
        }

        if alpha == 0.0 {
            if input.write.verbose {
                println!("h-SCF line search found no improving step.");
            }
            finalise(ca_new, cb_new, ao, input, label, noci_basis, i);
            return None;
        }

        let pa_acc = pa.mapv(|z| z * alpha);
        let pb_acc = pb.mapv(|z| z * alpha);

        // Store the accepted unweighted displacement and gradient for the next SR1 secant pair.
        step_prev = Some((pa_acc, pb_acc));
        g_prev = Some((ga, gb));
        ca = ca_new;
        cb = cb_new;
    }

    finalise(ca, cb, ao, input, label, noci_basis, i);
    None
}

/// Build initial h-SCF orbitals from a real SCF seed and state recipe.
/// # Arguments:
/// - `seed`: Real SCF seed state.
/// - `recipe`: State construction recipe.
/// - `ao`: Contains electron counts and AO metadata.
/// # Returns:
/// - `(Array2<Complex64>, Array2<Complex64>)`: Alpha and beta h-SCF initial orbitals.
pub fn h_seed_orbitals(
    seed: &SCFState,
    recipe: &StateRecipe,
    ao: &AoData,
) -> (Array2<Complex64>, Array2<Complex64>) {
    let na = usize::try_from(ao.nelec[0]).unwrap();
    let nb = usize::try_from(ao.nelec[1]).unwrap();

    let (mut ca, mut cb) = complex_orbitals_from_real(seed, ao);

    if let Some(sb) = &recipe.spin_bias {
        let sgn = sb.pattern.iter().copied().find(|&x| x != 0).unwrap_or(1) as f64;
        let theta = Complex64::new(0.0, sgn * sb.pol.abs().max(0.05));
        ca = perturb_ov(&ca, na, theta);
        cb = perturb_ov(&cb, nb, -theta);
    }

    if let Some(spb) = &recipe.spatial_bias {
        let sgn = spb.pattern.iter().copied().find(|&x| x != 0).unwrap_or(1) as f64;
        let theta = Complex64::new(0.0, sgn * spb.pol.abs().max(0.05));
        ca = perturb_ov(&ca, na, theta);
        cb = perturb_ov(&cb, nb, theta);
    }

    (ca, cb)
}

/// Apply the optional h-SCF partner gate for one holomorphic recipe.
/// # Arguments
/// - `recipe`: Holomorphic state recipe being considered.
/// - `input`: User input controlling verbose logging.
/// - `real_map`: Generated real states keyed by label.
/// - `recipe_map`: All state recipes keyed by label.
/// # Returns
/// - `PartnerSeed`: Whether to skip the h-SCF attempt, proceed without a partner, or use a collapsed partner seed.
fn hscf_partner_seed<'a>(
    recipe: &StateRecipe,
    input: &Input,
    real_map: &HashMap<&str, &'a SCFState>,
    recipe_map: &HashMap<&str, &StateRecipe>,
) -> PartnerSeed<'a> {
    let Some(label) = recipe.partner.as_deref() else {
        return PartnerSeed::NoPartner;
    };

    let partner_state = real_map.get(label).copied();
    let partner_recipe = recipe_map.get(label).copied();

    match (partner_state, partner_recipe) {
        (Some(st), Some(partner_recipe)) if partner_recipe.noci && !st.noci_basis => {
            PartnerSeed::Use(st)
        }
        (Some(_), Some(partner_recipe)) => {
            if input.write.verbose {
                if partner_recipe.noci {
                    println!(
                        "Skipping h-SCF state '{}' because partner '{}' remains in the NOCI basis.",
                        recipe.label, label
                    );
                } else {
                    println!(
                        "Skipping h-SCF state '{}' because partner '{}' was not requested for the NOCI basis.",
                        recipe.label, label
                    );
                }
            }
            PartnerSeed::Skip
        }
        (Some(_), None) => {
            if input.write.verbose {
                println!(
                    "Skipping h-SCF state '{}' because partner recipe '{}' was not found.",
                    recipe.label, label
                );
            }
            PartnerSeed::Skip
        }
        (None, _) => {
            if input.write.verbose {
                println!(
                    "Skipping h-SCF state '{}' because partner state '{}' was not generated.",
                    recipe.label, label
                );
            }
            PartnerSeed::Skip
        }
    }
}

/// Find a previously generated spin-flipped h-SCF partner for an opposite spin-bias recipe.
/// # Arguments
/// - `ao`: AO data containing alpha/beta electron counts.
/// - `recipes`: All state recipes.
/// - `recipe`: Current holomorphic recipe.
/// - `i`: Current recipe index.
/// - `current_h`: Current-geometry h-SCF states keyed by recipe label.
/// # Returns
/// - `Option<&HSCFState>`: Previously generated h-SCF state that can be spin-flipped.
fn hscf_spin_flip_source<'a>(
    ao: &AoData,
    recipes: &'a [StateRecipe],
    recipe: &StateRecipe,
    i: usize,
    current_h: &HashMap<&str, &'a HSCFState>,
) -> Option<&'a HSCFState> {
    if ao.nelec[0] != ao.nelec[1] || recipe.spin_bias.is_none() {
        return None;
    }

    recipes
        .iter()
        .take(i)
        .find(|other| other.holomorphic && opposite_spin_bias(other, recipe))
        .and_then(|other| current_h.get(other.label.as_str()).copied())
}

/// Check whether two state recipes request opposite spin-bias patterns.
/// # Arguments
/// - `a`: First state recipe.
/// - `b`: Second state recipe.
/// # Returns
/// - `bool`: Whether both recipes have spin biases and the patterns are sign opposites.
fn opposite_spin_bias(
    a: &StateRecipe,
    b: &StateRecipe,
) -> bool {
    let (Some(sa), Some(sb)) = (&a.spin_bias, &b.spin_bias) else {
        return false;
    };

    sa.pattern.len() == sb.pattern.len()
        && sa
            .pattern
            .iter()
            .zip(sb.pattern.iter())
            .all(|(&x, &y)| x == -y)
}

/// Select the h-SCF candidate consistent with the recipe's branch preference.
/// # Arguments
/// - `recipe`: Holomorphic recipe used to generate the candidates.
/// - `candidates`: Converged h-SCF candidates.
/// # Returns
/// - `HSCFState`: Selected h-SCF state.
fn select_hscf_candidate(
    recipe: &StateRecipe,
    candidates: Vec<HSCFState>,
) -> HSCFState {
    if candidates.is_empty() {
        panic!("No converged h-SCF candidate for '{}'", recipe.label);
    }

    if recipe.spin_bias.is_some() {
        candidates
            .into_iter()
            .min_by(|a, b| a.e.re.partial_cmp(&b.e.re).unwrap())
            .unwrap()
    } else if recipe.spatial_bias.is_some() {
        candidates
            .into_iter()
            .max_by(|a, b| a.e.re.partial_cmp(&b.e.re).unwrap())
            .unwrap()
    } else {
        candidates.into_iter().next().unwrap()
    }
}

/// Build occupied-first complex orbitals from a real SCF seed.
/// # Arguments:
/// - `seed`: Real SCF state used as the orbital source.
/// - `ao`: AO data containing the overlap matrix.
/// # Returns:
/// - `(Array2<Complex64>, Array2<Complex64>)`: Alpha and beta complex orbitals ordered occupied first.
fn complex_orbitals_from_real(
    seed: &SCFState,
    ao: &AoData,
) -> (Array2<Complex64>, Array2<Complex64>) {
    let occ = spin_occupation(seed);
    let idx_a = occ.alpha_occupied_first();
    let idx_b = occ.beta_occupied_first();

    let ca = real2_as::<Complex64>(&seed.ca).select(Axis(1), &idx_a);
    let cb = real2_as::<Complex64>(&seed.cb).select(Axis(1), &idx_b);

    (
        complex_metric_orthonormalize(&ca, &ao.s),
        complex_metric_orthonormalize(&cb, &ao.s),
    )
}

/// Construct final h-SCF state from optimised complex orbitals.
/// # Arguments:
/// - `ca`: Final alpha-spin MO coefficients.
/// - `cb`: Final beta-spin MO coefficients.
/// - `ao`: Contains AO integrals and metadata.
/// - `input`: User input specifications.
/// - `label`: Label for the h-SCF state.
/// - `noci_basis`: Whether this state is intended for a later NOCI basis.
/// - `i`: State index.
/// # Returns:
/// - `HSCFState`: Final h-SCF determinant state.
fn finalise(
    ca: Array2<Complex64>,
    cb: Array2<Complex64>,
    ao: &AoData,
    input: &Input,
    label: &str,
    noci_basis: bool,
    i: usize,
) -> HSCFState {
    let na = usize::try_from(ao.nelec[0]).unwrap();
    let nb = usize::try_from(ao.nelec[1]).unwrap();

    let da = density(&ca, na, DensityMode::Holomorphic);
    let db = density(&cb, nb, DensityMode::Holomorphic);

    let (fa, fb) = fock(&ao.h, &ao.eri_coul, &da, &db);
    let e = energy(&ao.h, ao.enuc, &da, &db, &fa, &fb);

    if input.write.verbose {
        println!("{}", "-".repeat(100));
        println!("Complex coefficients ca:");
        print_array2_indexed(&ca);
        println!("Complex coefficients cb:");
        print_array2_indexed(&cb);
    }

    if input.write.write_orbitals {
        println!("Complex h-SCF orbital HDF5 writing is not implemented yet.");
    }

    // Occupy the first `na` and `nb` orbitals because h-SCF keeps occupied orbitals first throughout.
    let oa = (0..na).fold(0u128, |bits, j| bits | (1u128 << j));
    let ob = (0..nb).fold(0u128, |bits, j| bits | (1u128 << j));

    HSCFState {
        e,
        oa,
        ob,
        pha: 1.0,
        phb: 1.0,
        ca: Arc::new(ca),
        cb: Arc::new(cb),
        da: Arc::new(da),
        db: Arc::new(db),
        label: label.to_string(),
        noci_basis,
        parent: i,
        excitation: Excitation::empty(),
    }
}

/// Pseudo-canonicalise occupied and virtual spaces for one spin block.
/// # Arguments:
/// - `c`: MO coefficient matrix ordered as occupied then virtual.
/// - `f`: Spin Fock matrix.
/// - `nocc`: Number of occupied orbitals.
/// - `hist`: Stored SR1 secant pairs transformed into the new tangent basis.
/// - `spin`: Spin block being transformed.
/// - `extra`: Additional tangent matrices transformed into the new tangent basis.
/// # Returns:
/// - `Array1<Complex64>`: Occupied followed by virtual pseudo-canonical orbital energies.
fn pseudo_canonicalise(
    c: &mut Array2<Complex64>,
    f: &Array2<Complex64>,
    nocc: usize,
    hist: &mut [SecantPair],
    spin: SpinBlock,
    extra: &mut [&mut Array2<Complex64>],
) -> Array1<Complex64> {
    let n = c.ncols();

    // Transform Fock matrix into MO basis such that we have o-o, o-v, v-o, v-v blocks.
    let fmo = c.t().dot(f).dot(c);

    // Diagonalise o-o and v-v blocks.
    let (eo, uo) = symmetric_evp_complex(&fmo.slice(s![0..nocc, 0..nocc]).to_owned());
    let (ev, uv) = symmetric_evp_complex(&fmo.slice(s![nocc..n, nocc..n]).to_owned());

    // Rotate orbitals within occupied and virtual spaces as C_o = C_o U_o, C_v = C_v U_v.
    let cocc = c.slice(s![.., 0..nocc]).to_owned().dot(&uo);
    let cvir = c.slice(s![.., nocc..n]).to_owned().dot(&uv);
    c.slice_mut(s![.., 0..nocc]).assign(&cocc);
    c.slice_mut(s![.., nocc..n]).assign(&cvir);

    // Transform the stored SR1 secant-pair history into the new pseudo-canonical tangent basis.
    for pair in hist.iter_mut() {
        match spin {
            SpinBlock::Alpha => {
                // Transform the previous alpha-spin step as s_ai -> (U_v^T s U_o)_ai.
                pair.sa = uv.t().dot(&pair.sa).dot(&uo);

                // Transform the previous alpha-spin gradient change as y_ai -> (U_v^T y U_o)_ai.
                pair.ya = uv.t().dot(&pair.ya).dot(&uo);
            }
            SpinBlock::Beta => {
                // Transform the previous beta-spin step as s_ai -> (U_v^T s U_o)_ai.
                pair.sb = uv.t().dot(&pair.sb).dot(&uo);

                // Transform the previous beta-spin gradient change as y_ai -> (U_v^T y U_o)_ai.
                pair.yb = uv.t().dot(&pair.yb).dot(&uo);
            }
        }
    }

    // Transform any additional tangent-space matrices into the new pseudo-canonical basis.
    for x in extra.iter_mut() {
        // Transform the extra occupied-virtual block as X_ai -> (U_v^T X U_o)_ai.
        **x = uv.t().dot(&**x).dot(&uo);
    }

    let mut eps = Array1::<Complex64>::zeros(n);
    eps.slice_mut(s![0..nocc]).assign(&eo);
    eps.slice_mut(s![nocc..n]).assign(&ev);
    eps
}

/// Solve the complex-symmetric SR1 quasi-Newton equation in energy-weighted coordinates.
/// # Arguments:
/// - `hist`: Stored unweighted secant pairs.
/// - `g`: Alpha- and beta-spin occupied-virtual gradients.
/// - `eps`: Alpha- and beta-spin pseudo-canonical orbital energies.
/// - `nocc`: Number of occupied alpha- and beta-spin orbitals.
/// - `opts`: h-SCF quasi-Newton options.
/// # Returns:
/// - `(Array2<Complex64>, Array2<Complex64>)`: Unweighted alpha- and beta-spin orbital steps.
fn sr1_step(
    hist: &[SecantPair],
    g: (&Array2<Complex64>, &Array2<Complex64>),
    eps: (&Array1<Complex64>, &Array1<Complex64>),
    nocc: (usize, usize),
    opts: &HSCFOptions,
) -> (Array2<Complex64>, Array2<Complex64>) {
    let (ga, gb) = g;
    let (epsa, epsb) = eps;
    let (na, nb) = nocc;

    // Convert current gradient into energy-weighted coordinates as
    // \bar g_{ai} = g_{ai} / \sqrt{\Delta{ai}} such that the true Hessian is
    // closer to the identity.
    let gpa = weight_by_gap(ga, epsa, na, opts.denom_tol, false);
    let gpb = weight_by_gap(gb, epsb, nb, opts.denom_tol, false);

    let n = gpa.len() + gpb.len();

    // Begin initial guess of the approximate Hessian as identity.
    let mut b = Array2::<Complex64>::zeros((n, n));
    for i in 0..n {
        b[(i, i)] = Complex64::new(1.0, 0.0);
    }

    // For every secant pair stored update the approximated Hessian.
    for pair in hist {
        // Convert previous step into energy-weighted coordinates as \bar s_{ai} = s_{ai} \sqrt{\Delta_{ai}}.
        let sa = weight_by_gap(&pair.sa, epsa, na, opts.denom_tol, true);
        let sb = weight_by_gap(&pair.sb, epsb, nb, opts.denom_tol, true);

        // Convert previous gradient change into energy-weighted coordinates as \bar y_{ai} = y_{ai} / \sqrt{\Delta_{ai}}.
        let ya = weight_by_gap(&pair.ya, epsa, na, opts.denom_tol, false);
        let yb = weight_by_gap(&pair.yb, epsb, nb, opts.denom_tol, false);

        let s = pack(&sa, &sb);
        let y = pack(&ya, &yb);

        // Calculate residual error in current prediction.
        let r = &y - &b.dot(&s);

        // Update Hessian approximation as B_{k + 1} = B_k + (r_k r_k^T) / r_k^T s_k.
        let denom = r.dot(&s);
        if denom.norm() > opts.sr1_tol {
            let col = r.view().insert_axis(Axis(1));
            let row = r.view().insert_axis(Axis(0));
            b = b + col.dot(&row).mapv(|z| z / denom);
        }
    }

    // Solve B \bar p = - \bar g for energy-weighted occupied-virtual rotation amplitudes.
    let rhs = pack(&gpa, &gpb).mapv(|z| -z);
    let p = b.solve_into(rhs.clone()).unwrap_or(rhs);
    let (pa_bar, pb_bar) = unpack(&p, (ga.nrows(), ga.ncols()), (gb.nrows(), gb.ncols()));

    // Convert the solution back to unweighted occupied-virtual rotation coordinates.
    (
        weight_by_gap(&pa_bar, epsa, na, opts.denom_tol, false),
        weight_by_gap(&pb_bar, epsb, nb, opts.denom_tol, false),
    )
}

/// Backtrack along the complex-orthogonal geodesic and minimise gradient norm.
/// # Arguments:
/// - `c`: Current alpha- and beta-spin MO coefficient matrices.
/// - `ao`: Contains AO integrals and metadata.
/// - `nocc`: Number of occupied alpha- and beta-spin orbitals.
/// - `p`: Alpha- and beta-spin occupied-virtual steps.
/// - `g0`: Current occupied-virtual gradient norm.
/// - `opts`: h-SCF quasi-Newton options.
/// # Returns:
/// - `(f64, Array2<Complex64>, Array2<Complex64>)`: Step length and updated alpha/beta orbitals.
fn line_search(
    c: (&Array2<Complex64>, &Array2<Complex64>),
    ao: &AoData,
    nocc: (usize, usize),
    p: (&Array2<Complex64>, &Array2<Complex64>),
    g0: f64,
    opts: &HSCFOptions,
) -> (f64, Array2<Complex64>, Array2<Complex64>) {
    let (ca, cb) = c;
    let (na, nb) = nocc;
    let (pa, pb) = p;

    // Start with the full step length.
    let mut alpha = 1.0;
    let mut best = (0.0, ca.clone(), cb.clone(), g0);

    // Try a sequence of increasingly smaller step lengths.
    for _ in 0..opts.line_steps {
        // Try both directions along the geodesic.
        for sign in [1.0, -1.0] {
            let cat = geodesic_step(ca, pa, na, alpha * sign);
            let cbt = geodesic_step(cb, pb, nb, alpha * sign);

            let da = density(&cat, na, DensityMode::Holomorphic);
            let db = density(&cbt, nb, DensityMode::Holomorphic);

            let (fa, fb) = fock(&ao.h, &ao.eri_coul, &da, &db);

            // Calculate orbital gradient g_{ai} = 2 \sum_{\mu\nu} C_a^\mu F_{\mu\nu} C_i^\nu.
            let ga = orbital_gradient(&cat, &fa, na, DensityMode::Holomorphic);
            let gb = orbital_gradient(&cbt, &fb, nb, DensityMode::Holomorphic);

            // Compare candidate steps using the real diagnostic norm of the h-SCF gradient.
            let g = (ga.iter().map(|z| z.norm_sqr()).sum::<f64>()
                + gb.iter().map(|z| z.norm_sqr()).sum::<f64>())
            .sqrt();

            if g < best.3 {
                best = (alpha * sign, cat.clone(), cbt.clone(), g);
            }
            if g < g0 {
                return (alpha * sign, cat, cbt);
            }
        }
        alpha *= opts.line_shrink;
    }

    (best.0, best.1, best.2)
}

/// Build and solve a finite-difference local Newton equation for stalled h-SCF iterations.
/// # Arguments:
/// - `ca`: Current alpha-spin MO coefficients.
/// - `cb`: Current beta-spin MO coefficients.
/// - `ao`: AO data.
/// - `na`: Number of occupied alpha-spin orbitals.
/// - `nb`: Number of occupied beta-spin orbitals.
/// - `ga`: Current alpha-spin occupied-virtual gradient.
/// - `gb`: Current beta-spin occupied-virtual gradient.
/// # Returns:
/// - `Option<(Array2<Complex64>, Array2<Complex64>)>`: Newton step if the linear solve succeeds.
fn finite_difference_newton_step(
    ca: &Array2<Complex64>,
    cb: &Array2<Complex64>,
    ao: &AoData,
    na: usize,
    nb: usize,
    ga: &Array2<Complex64>,
    gb: &Array2<Complex64>,
) -> Option<(Array2<Complex64>, Array2<Complex64>)> {
    let g0 = pack(ga, gb);
    let n = g0.len();
    let eps = 1.0e-4;
    let mut h = Array2::<Complex64>::zeros((n, n));

    for j in 0..n {
        let mut va = Array2::<Complex64>::zeros(ga.raw_dim());
        let mut vb = Array2::<Complex64>::zeros(gb.raw_dim());

        if j < ga.len() {
            for (k, x) in va.iter_mut().enumerate() {
                if k == j {
                    *x = Complex64::new(eps, 0.0);
                    break;
                }
            }
        } else {
            let jb = j - ga.len();
            for (k, x) in vb.iter_mut().enumerate() {
                if k == jb {
                    *x = Complex64::new(eps, 0.0);
                    break;
                }
            }
        }

        let cat = geodesic_step(ca, &va, na, 1.0);
        let cbt = geodesic_step(cb, &vb, nb, 1.0);

        let da = density(&cat, na, DensityMode::Holomorphic);
        let db = density(&cbt, nb, DensityMode::Holomorphic);

        let (fa, fb) = fock(&ao.h, &ao.eri_coul, &da, &db);
        let gt_a = orbital_gradient(&cat, &fa, na, DensityMode::Holomorphic);
        let gt_b = orbital_gradient(&cbt, &fb, nb, DensityMode::Holomorphic);

        let dg = (pack(&gt_a, &gt_b) - &g0).mapv(|z| z / eps);
        h.column_mut(j).assign(&dg);
    }

    let rhs = g0.mapv(|z| -z);
    let p = h.solve_into(rhs).ok()?;
    Some(unpack(
        &p,
        (ga.nrows(), ga.ncols()),
        (gb.nrows(), gb.ncols()),
    ))
}

/// Apply or remove pseudo-canonical orbital-gap weighting:
///     \sqrt{\Delta_{ai}} = \sqrt{\tilde{F}_{aa} - \tilde{F}_{ii}}.
/// # Arguments:
/// - `x`: Occupied-virtual block.
/// - `eps`: Pseudo-canonical orbital energies.
/// - `nocc`: Number of occupied orbitals.
/// - `tol`: Minimum allowed gap magnitude.
/// - `multiply`: If true multiply by the gap square root, otherwise divide.
/// # Returns:
/// - `Array2<Complex64>`: Weighted or unweighted occupied-virtual block.
fn weight_by_gap(
    x: &Array2<Complex64>,
    eps: &Array1<Complex64>,
    nocc: usize,
    tol: f64,
    multiply: bool,
) -> Array2<Complex64> {
    let mut y = x.clone();

    for a in 0..x.nrows() {
        for i in 0..x.ncols() {
            // \Delta_{ai} = \tilde{F}_{aa} - \tilde{F}_{ii}.
            let mut gap = eps[nocc + a] - eps[i];

            if gap.norm() < tol {
                gap = Complex64::new(tol, 0.0);
            }

            let w = gap.sqrt();
            y[(a, i)] = if multiply {
                x[(a, i)] * w
            } else {
                x[(a, i)] / w
            };
        }
    }

    y
}

/// Limit combined alpha/beta occupied-virtual step norm.
/// # Arguments:
/// - `pa`: Alpha-spin occupied-virtual step.
/// - `pb`: Beta-spin occupied-virtual step.
/// - `max_step`: Maximum allowed combined step norm.
/// # Returns:
/// - `()`: Updates `pa` and `pb` in place.
fn limit_step(
    pa: &mut Array2<Complex64>,
    pb: &mut Array2<Complex64>,
    max_step: f64,
) {
    // ||p|| = \sqrt{\sum_{ai} |p_{ai}^\alpha|^2 + \sum_{ai} |p_{ai}^\beta|^2}.
    let n = step_norm(pa, pb);

    // If the proposed step size found by the SR1 solve is too big scale it down.
    if n > max_step && n > 0.0 {
        let scale = max_step / n;
        pa.mapv_inplace(|z| z * scale);
        pb.mapv_inplace(|z| z * scale);
    }
}

/// Combined alpha/beta occupied-virtual step norm.
/// # Arguments:
/// - `pa`: Alpha-spin occupied-virtual step.
/// - `pb`: Beta-spin occupied-virtual step.
/// # Returns:
/// - `f64`: Euclidean norm of the complex alpha/beta step blocks.
fn step_norm(
    pa: &Array2<Complex64>,
    pb: &Array2<Complex64>,
) -> f64 {
    (pa.iter().map(|z| z.norm_sqr()).sum::<f64>() + pb.iter().map(|z| z.norm_sqr()).sum::<f64>())
        .sqrt()
}

/// Apply a complex-orthogonal occupied-virtual geodesic step.
/// # Arguments:
/// - `c`: Current MO coefficient matrix ordered as occupied then virtual.
/// - `p`: Occupied-virtual step block with shape `(nvir, nocc)`.
/// - `nocc`: Number of occupied orbitals.
/// - `alpha`: Step length.
/// # Returns:
/// - `Array2<Complex64>`: Updated MO coefficient matrix.
fn geodesic_step(
    c: &Array2<Complex64>,
    p: &Array2<Complex64>,
    nocc: usize,
    alpha: f64,
) -> Array2<Complex64> {
    let n = c.ncols();
    let nvir = n - nocc;

    let mut k = Array2::<Complex64>::zeros((n, n));

    // Construct the \exp(0 -\alpha_k p_k^T \\ \alpha_k p_k) matrix exponential.
    for a in 0..nvir {
        for i in 0..nocc {
            let z = p[(a, i)] * alpha;
            k[(nocc + a, i)] = z;
            k[(i, nocc + a)] = -z;
        }
    }

    c.dot(&matrix_exp_complex(&k))
}

/// Apply an imaginary occupied-virtual perturbation to initialise a complex h-SCF branch.
/// # Arguments:
/// - `c`: MO coefficient matrix ordered as occupied then virtual.
/// - `nocc`: Number of occupied orbitals.
/// - `theta`: Complex rotation amplitude applied to corresponding occupied-virtual pairs.
/// # Returns:
/// - `Array2<Complex64>`: Kicked MO coefficient matrix.
fn perturb_ov(
    c: &Array2<Complex64>,
    nocc: usize,
    theta: Complex64,
) -> Array2<Complex64> {
    let n = c.ncols();
    let nvir = n - nocc;

    if nocc == 0 || nvir == 0 || theta.norm() == 0.0 {
        return c.clone();
    }

    let mut p = Array2::<Complex64>::zeros((nvir, nocc));
    p[(0, nocc - 1)] = theta;

    geodesic_step(c, &p, nocc, 1.0)
}

/// Pack alpha and beta tangent blocks into one vector.
/// # Arguments:
/// - `a`: Alpha-spin tangent block.
/// - `b`: Beta-spin tangent block.
/// # Returns:
/// - `Array1<Complex64>`: Concatenated vector.
fn pack(
    a: &Array2<Complex64>,
    b: &Array2<Complex64>,
) -> Array1<Complex64> {
    Array1::from_iter(a.iter().chain(b.iter()).copied())
}

/// Unpack one vector into alpha and beta tangent blocks.
/// # Arguments:
/// - `x`: Packed tangent vector.
/// - `adim`: Alpha-spin block dimensions.
/// - `bdim`: Beta-spin block dimensions.
/// # Returns:
/// - `(Array2<Complex64>, Array2<Complex64>)`: Alpha- and beta-spin tangent blocks.
fn unpack(
    x: &Array1<Complex64>,
    adim: (usize, usize),
    bdim: (usize, usize),
) -> (Array2<Complex64>, Array2<Complex64>) {
    let na = adim.0 * adim.1;
    let mut a = Array2::<Complex64>::zeros(adim);
    let mut b = Array2::<Complex64>::zeros(bdim);

    for (dst, src) in a.iter_mut().zip(x.slice(s![0..na]).iter()) {
        *dst = *src;
    }
    for (dst, src) in b.iter_mut().zip(x.slice(s![na..]).iter()) {
        *dst = *src;
    }

    (a, b)
}
