// nocc/stochastic/propagate.rs

// External crate imports.
use mpi::topology::Communicator;
use mpi::traits::*;
use ndarray::Array1;
use rayon::prelude::*;

// Crate-root imports.
use crate::determinant::ExcitedDeterminant;
use crate::elements::nonorthogonalwicks::WickScratchSpin;
use crate::elements::{NOCIData, calculate_hs_excited};
use crate::input::JMNOCCMCOptions;
use crate::mpiutils::all_reduce_array1;
use crate::nocc::jm_energies;
use crate::qmc::{
    ExchangeScratch, FastMap, QmcRng, coalesce, compress_sparse, exchange_updates, gather_updates,
    owner_words, pack_updates, print_table_header, print_table_row, rank_seed, stop_requested,
    stream, update_shift,
};

// Parent/sibling imports.
use super::cluster::{collapse, select_cluster};
use super::coupling::{Projections, left_action, project_cluster};
use super::effective::{Effective, ground_energy, reblocked_energy, scaled};
use super::linked::linked_attempt;
use super::population::{Block, Excitor, ExcitorKey, ExcitorUpdate, ReferenceSpaces, rank};

/// Smallest number of cluster attempts in one parallel task.
const CHUNK: usize = 256;

/// Number of parallel tasks per block and kind on each Rayon thread, bounding the per-task
/// spawn maps held at once.
const TASKSPERTHREAD: usize = 4;

/// Salt of the rank-local population-rounding stream.
const ROUNDSALT: u64 = 0x8EBC6AF09C88C6E3;

/// Multiplier giving each iteration its own excitor-to-rank assignment.
const ASSIGNSTRIDE: u64 = 0xD1B54A32D192ED03;

/// Salt of the rank-local spawn-compression stream.
const SPAWNSALT: u64 = 0xA0761D6478BD642F;

/// Kind of cluster attempts in one parallel task.
#[derive(Clone, Copy)]
enum Task {
    /// Linked same-reference attempts on a block.
    Linked(usize, usize),
    /// Coupling attempts projecting a block onto the other references.
    Coupling(usize, usize),
}

/// Output of one parallel task.
struct TaskOutput {
    /// Population changes per unit time, summed per excitor.
    spawns: FastMap<ExcitorKey, f64>,
    /// Contribution to `N_z H^e_{zz}` of each block.
    energy: Vec<f64>,
    /// Coupling projections.
    projections: Projections,
}

/// Rank owning an excitor during one iteration. The assignment is redrawn every iteration from
/// a salt shared by all ranks, so that over the assignment any `s` excitors share a rank with
/// probability `P^{-(s-1)}`.
/// # Arguments:
/// - `key`: Excitor.
/// - `assignment`: Salt of the iteration's assignment.
/// - `nranks`: Number of ranks `P`.
/// # Returns:
/// - `usize`: Owning rank.
fn owner(
    key: &ExcitorKey,
    assignment: u64,
    nranks: usize,
) -> usize {
    if nranks == 1 {
        return 0;
    }
    owner_words(
        &[
            assignment,
            key.0 as u64,
            key.1 as u64,
            (key.1 >> 64) as u64,
            key.2 as u64,
            (key.2 >> 64) as u64,
        ],
        nranks,
    )
}

/// Build the per-step view of every block from this rank's populations, with excitors in key
/// order, and the global reference populations.
/// # Arguments:
/// - `spaces`: Projection and expansion spaces of every reference block.
/// - `populations`: Excitor populations owned by this rank, references included.
/// - `nref`: Global reference populations `N_z`.
/// # Returns:
/// - `Vec<Block>`: Block populations of this rank.
fn blocks(
    spaces: &ReferenceSpaces,
    populations: &FastMap<ExcitorKey, f64>,
    nref: &[f64],
) -> Vec<Block> {
    let mut keys = populations.keys().copied().collect::<Vec<_>>();
    keys.par_sort_unstable();
    let mut out = spaces
        .references
        .iter()
        .zip(nref)
        .map(|(reference, &nref)| Block {
            reference: *reference,
            nref,
            excitors: Vec::new(),
            cumulative: Vec::new(),
        })
        .collect::<Vec<_>>();

    for key in keys {
        let block = &mut out[key.0];
        let det = ExcitedDeterminant {
            parent: block.reference.parent,
            alpha: key.1,
            beta: key.2,
        };
        if det != block.reference {
            let n = populations[&key];
            let total = block.cumulative.last().copied().unwrap_or(0.0) + n.abs();
            block
                .excitors
                .push((Excitor::new(&block.reference, det), n));
            block.cumulative.push(total);
        }
    }
    out
}

/// Run one parallel task of cluster attempts.
/// # Arguments:
/// - `data`: NOCI data holding the parent frames, MO caches and Wick intermediates.
/// - `spaces`: Projection and expansion spaces of every reference block.
/// - `blocks`: Block populations of this rank.
/// - `task`: Task kind with its block and number of attempts.
/// - `attempts`: Attempts of the task's block on this rank and the number of ranks.
/// - `rng`: Task random-number generator.
/// # Returns:
/// - `TaskOutput`: Population changes per unit time and accumulated projections.
fn run_task(
    data: &NOCIData<'_, f64>,
    spaces: &ReferenceSpaces,
    blocks: &[Block],
    task: Task,
    attempts: (f64, f64),
    mut rng: QmcRng,
) -> TaskOutput {
    let (attempts, ranks) = attempts;
    let m = blocks.len();
    let mut out = TaskOutput {
        spawns: FastMap::default(),
        energy: vec![0.0; m],
        projections: Projections::new(m),
    };
    let mut scratch = WickScratchSpin::new();
    let mut excitors = Vec::new();

    match task {
        Task::Linked(z, count) => {
            let smax = 4.min(spaces.cluster_rank + 2);
            for _ in 0..count {
                let weight =
                    select_cluster(&blocks[z], smax, attempts, ranks, &mut rng, &mut excitors);
                out.energy[z] += linked_attempt(
                    data,
                    spaces,
                    z,
                    (&excitors, weight),
                    &mut rng,
                    &mut scratch,
                    &mut out.spawns,
                );
            }
        }
        Task::Coupling(x, count) => {
            for _ in 0..count {
                let weight = select_cluster(
                    &blocks[x],
                    spaces.expansion_rank,
                    attempts,
                    ranks,
                    &mut rng,
                    &mut excitors,
                );
                let Some((det, sign)) = collapse(&spaces.references[x], &excitors) else {
                    continue;
                };
                if rank(&spaces.references[x], &det) > spaces.expansion_rank {
                    continue;
                }
                project_cluster(
                    data,
                    (&spaces.states, &spaces.rows),
                    x,
                    (&det.state(data.space.parents()), weight * sign),
                    &mut rng,
                    &mut scratch,
                    &mut out.projections,
                );
            }
        }
    }
    out
}

/// Stochastically propagate the JM-NOCCMC amplitudes by linked coupled cluster Monte Carlo
/// with the plain update `t \leftarrow t - \Delta\tau R` and one population-control shift.
///
/// Populations `n_{x\mu}` live on the excitors of each reference block and the reference
/// populations `N_x` carry the normalisation, `t_{x\mu}=n_{x\mu}/N_x`. With
/// `\tilde G_{xy}=G_{xy}N_y/N_x` and `g_x=\sum_y\tilde G_{xy}`, every population is updated as
///
/// `n_{x\mu} \leftarrow n_{x\mu} - \Delta\tau N_xR_{x\mu} - \Delta\tau(g_x-S)n_{x\mu},`
///
/// with `R_{x0}=0` for the reference. The reference populations then follow the power iteration
/// `\Delta N=-\Delta\tau(G-S)N` of the effective Hamiltonian, converging to its ground-state
/// coefficients, while the amplitudes follow `t \leftarrow t - \Delta\tau R` exactly. The
/// same-reference part of `N_xR_{x\mu}` is sampled by linked CCMC, the coupling between
/// references through sampled projections, and `G` is taken from the previous report block.
/// The populations are distributed over the MPI ranks by a hash of the excitor redrawn every
/// iteration: each rank forms clusters from its own excitors with the weight `P^{s-1}`, its
/// spawns and the excitors changing owner are exchanged in one step, and annihilation happens on
/// the owner. Within a rank the attempts, the left action and the sorts run on Rayon threads.
/// # Arguments:
/// - `data`: NOCI data over the reference determinants.
/// - `options`: JM-NOCCMC options with stochastic settings.
/// - `c0`: Reference NOCI coefficients setting the initial reference populations.
/// - `shift`: Initial population-control shift.
/// - `world`: MPI communicator.
/// # Returns:
/// - `f64`: Average ground-state energy after the population target is reached, or the last
///   report-block energy if it is never reached.
/// # Panics
/// - Panics if the stochastic options or MO caches are absent.
/// # References
/// - R. S. T. Franklin, J. S. Spencer, A. Zoccante and A. J. W. Thom, *J. Chem. Phys.* **144**,
///   044111 (2016), [doi:10.1063/1.4940317](https://doi.org/10.1063/1.4940317), for the
///   fixed-point-preserving death term.
/// - J. S. Spencer, V. A. Neufeld, W. A. Vigor, R. S. T. Franklin and A. J. W. Thom,
///   *J. Chem. Phys.* **149**, 204103 (2018), [doi:10.1063/1.5047420](https://doi.org/10.1063/1.5047420),
///   for cluster selection from rank-local excitors under a redrawn assignment.
pub(crate) fn jm_qmc_step(
    data: &NOCIData<'_, f64>,
    options: &JMNOCCMCOptions,
    c0: &[f64],
    mut shift: f64,
    world: &impl Communicator,
) -> f64 {
    let qmc = options
        .qmc
        .as_ref()
        .expect("jmnoccmc.qmc options are required");
    let irank = world.rank() as usize;
    let nranks = world.size() as usize;
    let m = data.space.len();

    // One base seed for every rank, from which the shared assignments and the rank streams follow.
    let mut base = qmc.seed.unwrap_or_else(rand::random);
    world.process_at_rank(0).broadcast_into(&mut base);
    let local = rank_seed(base, irank);

    let spaces = ReferenceSpaces::new(data, options);
    let dt = options.dt;

    // Initial reference populations from the NOCI coefficients and the exact effective
    // Hamiltonian of uncorrelated references, `G = S^{-1}H`.
    let norm = c0.iter().map(|c| c.abs()).sum::<f64>();
    let assignment = |iteration: usize| base ^ (iteration as u64).wrapping_mul(ASSIGNSTRIDE);
    let trial = (0..m)
        .map(|z| {
            if c0.len() == m && norm > 0.0 {
                c0[z] / norm
            } else {
                1.0 / m as f64
            }
        })
        .collect::<Vec<_>>();
    let mut populations = FastMap::default();
    for (z, reference) in spaces.references.iter().enumerate() {
        let c = trial[z];
        let key = (z, reference.alpha, reference.beta);
        if owner(&key, assignment(0), nranks) == irank {
            populations.insert(key, qmc.initial_population * c);
        }
    }
    let (mut lagged, e0) = {
        let mut scratch = WickScratchSpin::new();
        let mut effective = Effective::new(m);
        for (z, bra) in spaces.references.iter().enumerate() {
            for (x, ket) in spaces.references.iter().enumerate() {
                let (h, s) = calculate_hs_excited(data, (bra, ket), &mut scratch);
                effective.se[(z, x)] = s;
                effective.he[(z, x)] = h;
            }
        }
        (
            effective.g().expect("singular reference overlap"),
            effective.he[(0, 0)],
        )
    };

    print_table_header(irank, ["NWalk", "NRef", "-", "-"]);
    let mut reached = false;
    let local_norm = populations.values().map(|n: &f64| n.abs()).sum::<f64>();
    let mut previous = all_reduce_array1(world, Array1::from_vec(vec![local_norm]))[0];
    let mut exchange = ExchangeScratch::new(nranks);
    let mut cumulative = Effective::new(m);
    let mut history = Vec::new();
    let mut energy = f64::NAN;
    let mut average = f64::NAN;

    for report in 0..qmc.nreports {
        let mut block = Effective::new(m);
        let mut projected = (0.0, 0.0);

        for cycle in 0..qmc.ncycles {
            let iteration = report * qmc.ncycles + cycle;
            let ranks = nranks as f64;
            let is_reference = |key: &ExcitorKey| {
                let reference = &spaces.references[key.0];
                (key.1, key.2) == (reference.alpha, reference.beta)
            };

            // Global reference populations and excitor norms, then this rank's block view.
            let mut totals = vec![0.0; 2 * m];
            for (key, &n) in &populations {
                if is_reference(key) {
                    totals[key.0] += n;
                } else {
                    totals[m + key.0] += n.abs();
                }
            }
            let totals = all_reduce_array1(world, Array1::from_vec(totals)).to_vec();
            let nref = totals[..m].to_vec();
            let current = blocks(&spaces, &populations, &nref);
            let gt = scaled(&lagged, &nref);

            // Share every block's attempts evenly between ranks and divide them into tasks.
            let attempts = (0..m)
                .map(|z| ((nref[z].abs() + totals[m + z]) / ranks).ceil().max(1.0))
                .collect::<Vec<_>>();
            let threads = rayon::current_num_threads();
            let mut tasks = Vec::new();
            for (z, &a) in attempts.iter().enumerate() {
                let a = a as usize;
                let chunk = CHUNK.max(a.div_ceil(TASKSPERTHREAD * threads));
                for start in (0..a).step_by(chunk) {
                    let count = chunk.min(a - start);
                    tasks.push(Task::Linked(z, count));
                    if m > 1 {
                        tasks.push(Task::Coupling(z, count));
                    }
                }
            }
            let outputs = tasks
                .into_par_iter()
                .enumerate()
                .map(|(k, task)| {
                    let z = match task {
                        Task::Linked(z, _) | Task::Coupling(z, _) => z,
                    };
                    let rng = stream(local, (iteration as u64) << 32, k as u64);
                    run_task(data, &spaces, &current, task, (attempts[z], ranks), rng)
                })
                .collect::<Vec<_>>();

            // Combine the tasks in order.
            let mut spawns = FastMap::default();
            let mut energies = vec![0.0; m];
            let mut projections = Projections::new(m);
            for output in &outputs {
                for (&key, &dn) in &output.spawns {
                    *spawns.entry(key).or_insert(0.0) += dn;
                }
                for (e, v) in energies.iter_mut().zip(&output.energy) {
                    *e += v;
                }
                projections.add(&output.projections);
            }

            // Complete the projections `\sum_{x \ne y}\tilde G_{xy}q^x_y` over ranks, then add the
            // coupling through the left action of `e^{-T_y}` on this rank's excitors.
            if m > 1 {
                let mut q = Vec::new();
                for y in 0..m {
                    for x in (0..m).filter(|&x| x != y) {
                        for (&(alpha, beta), &v) in &projections.q[y][x] {
                            q.push(ExcitorUpdate::new((y, alpha, beta), gt[(x, y)] * v));
                        }
                    }
                }
                let mut q = if nranks > 1 {
                    gather_updates(world, &q, &mut exchange).to_vec()
                } else {
                    q
                };
                q.par_sort_unstable_by_key(|u| u.excitor());
                coalesce(&mut q);
                for (y, b) in current.iter().enumerate() {
                    let entries = q
                        .iter()
                        .map(|u| u.excitor())
                        .zip(q.iter().map(|u| u.dn))
                        .filter(|((block, _, _), _)| *block == y)
                        .map(|((_, alpha, beta), v)| ((alpha, beta), v))
                        .collect::<Vec<_>>();
                    for (key, dn) in
                        left_action(b, y, &entries, spaces.cluster_rank, (ranks, irank == 0))
                    {
                        *spawns.entry(key).or_insert(0.0) += dn;
                    }
                }
            }

            // Annihilate this rank's spawns and round small ones stochastically.
            let mut updates = spawns
                .into_iter()
                .map(|(key, dn)| ExcitorUpdate::new(key, dt * dn))
                .collect::<Vec<_>>();
            updates.par_sort_unstable_by_key(|u| u.excitor());
            coalesce(&mut updates);
            let mut rng = stream(local, SPAWNSALT, iteration as u64);
            compress_sparse(&mut updates, qmc.fri.spawn_cutoff, &mut rng);

            // Sum the step's effective overlap and Hamiltonian over ranks, normalised by `N_x`.
            let mut sums = vec![0.0; m + 2 * m * m];
            sums[..m].copy_from_slice(&energies);
            for y in 0..m {
                for x in 0..m {
                    sums[m + y * m + x] = projections.se[y][x];
                    sums[m + m * m + y * m + x] = projections.he[y][x];
                }
            }
            let sums = all_reduce_array1(world, Array1::from_vec(sums)).to_vec();

            // Projected energy onto the fixed trial `\langle\Psi_T| = \sum_z c_z\langle{}^z\Psi|`:
            // `\sum_z c_z\sum_x\langle{}^z\Psi|\hat H|\tilde\Phi_x\rangle` over
            // `\sum_z c_z\sum_x\langle{}^z\Psi|\tilde\Phi_x\rangle`.
            for z in 0..m {
                projected.0 += trial[z] * sums[z];
                projected.1 += trial[z] * nref[z];
                for x in (0..m).filter(|&x| x != z) {
                    projected.0 += trial[z] * sums[m + m * m + z * m + x];
                    projected.1 += trial[z] * sums[m + z * m + x];
                }
            }

            for x in 0..m {
                block.se[(x, x)] += 1.0;
                block.he[(x, x)] += sums[x] / nref[x];
                for y in (0..m).filter(|&y| y != x) {
                    block.se[(y, x)] += sums[m + y * m + x] / nref[x];
                    block.he[(y, x)] += sums[m + m * m + y * m + x] / nref[x];
                }
            }

            // Fixed-point-preserving death `-\Delta\tau(g_x-S)n` on every owned population.
            let growth = (0..m).map(|x| gt.row(x).sum()).collect::<Vec<_>>();
            for (key, n) in populations.iter_mut() {
                *n *= 1.0 - dt * (growth[key.0] - shift);
            }

            // Send the spawns, and the populations whose owner changes under the next
            // iteration's assignment, to their next owners in one exchange.
            let next = assignment(iteration + 1);
            let received = if nranks > 1 {
                exchange.send_ranked.clear();
                for update in updates {
                    let peer = owner(&update.excitor(), next, nranks);
                    exchange.send_ranked.push((peer, update));
                }
                let moving = populations
                    .keys()
                    .filter(|key| owner(key, next, nranks) != irank)
                    .copied()
                    .collect::<Vec<_>>();
                for key in moving {
                    let n = populations.remove(&key).unwrap();
                    exchange
                        .send_ranked
                        .push((owner(&key, next, nranks), ExcitorUpdate::new(key, n)));
                }
                let (local_updates, remote) = exchange
                    .send_ranked
                    .drain(..)
                    .partition::<Vec<_>, _>(|&(peer, _)| peer == irank);
                exchange.send_ranked = remote;
                pack_updates(nranks, &mut exchange, true);
                let mut received = exchange_updates(world, &mut exchange).to_vec();
                received.extend(local_updates.into_iter().map(|(_, u)| u));
                received.par_sort_unstable_by_key(|u| u.excitor());
                coalesce(&mut received);
                received
            } else {
                updates
            };
            for update in &received {
                *populations.entry(update.excitor()).or_insert(0.0) += update.dn;
            }

            // Round owned excitor populations below the cutoff; references are never removed.
            let mut small = populations
                .iter()
                .filter(|(key, n)| !is_reference(key) && n.abs() < qmc.fri.population_cutoff)
                .map(|(&key, &n)| ExcitorUpdate::new(key, n))
                .collect::<Vec<_>>();
            small.sort_unstable_by_key(|u| u.excitor());
            for update in &small {
                populations.remove(&update.excitor());
            }
            let mut rng = stream(local, ROUNDSALT, iteration as u64);
            compress_sparse(&mut small, qmc.fri.population_cutoff, &mut rng);
            populations.extend(small.iter().map(|u| (u.excitor(), u.dn)));
        }

        // Report-block energy, lagged effective Hamiltonian and running average.
        if let Some(g) = block.g() {
            energy = jm_energies(&g)[0].re;
            lagged = g;
        }
        if reached {
            cumulative.add(&block);
            if let Some(e) = ground_energy(&cumulative) {
                average = e;
            }
            history.push(block);
        }

        // Population statistics over every rank and the shift update.
        let mut stats = vec![0.0; 2];
        for (key, n) in &populations {
            let reference = &spaces.references[key.0];
            stats[0] += n.abs();
            if (key.1, key.2) == (reference.alpha, reference.beta) {
                stats[1] += n.abs();
            }
        }
        let stats = all_reduce_array1(world, Array1::from_vec(stats));
        let (total, nref) = (stats[0], stats[1]);
        update_shift(
            &mut shift,
            &mut reached,
            (total, previous),
            (
                qmc.target_population,
                qmc.shift_damping,
                dt * qmc.ncycles as f64,
            ),
        );
        previous = total;
        let cycles = qmc.ncycles as f64;
        print_table_row(
            irank,
            (report + 1) * qmc.ncycles,
            (projected.0 / cycles, projected.1 / cycles),
            (e0, if reached { shift } else { 0.0 }),
            (total, nref, None),
        );

        if stop_requested(irank, nranks, world) {
            break;
        }
    }

    // Ratio estimate over every report block after the shift starts, with its reblocked error.
    if let Some((e, error)) = reblocked_energy(&history) {
        if irank == 0 {
            println!(
                "Stochastic JM-NOCCMC energy over {} report blocks: {:.10} +/- {:.10}",
                history.len(),
                e,
                error
            );
        }
        return e;
    }
    if average.is_finite() { average } else { energy }
}
