// nocc/stochastic/propagate.rs

// External crate imports.
use mpi::topology::Communicator;
use mpi::traits::*;
use ndarray::{Array1, Array2};
use rayon::prelude::*;

// Crate-root imports.
use crate::determinant::ExcitedDeterminant;
use crate::elements::nonorthogonalwicks::WickScratchSpin;
use crate::elements::{NOCIData, calculate_hs_excited};
use crate::input::JMNOCCMCOptions;
use crate::mpiutils::all_reduce_array1;
use crate::nocc::{Effective, jm_energies};
use crate::qmc::{
    AliasTable, ExchangeScratch, FastMap, QmcRng, coalesce, compress_sparse, exchange_updates,
    gather_updates, owner_words, pack_updates, print_table_header, print_table_row, rank_seed,
    stop_requested, stream, update_shift,
};

// Parent/sibling imports.
use super::cluster::{collapse, coupling_combinations, select_cluster, select_coupling_cluster};
use super::coupling::{left_action, project_cluster};
use super::effective::{ground_energy, reblocked_energy, scaled};
use super::linked::linked_attempt;
use super::population::{Block, Excitor, ExcitorKey, ExcitorUpdate, ReferenceSpaces, rank};
use super::state::JMTaskPropagation;

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
/// order, the global reference populations and the selection tables of linked and coupling
/// clusters.
/// # Arguments:
/// - `spaces`: Projection and expansion spaces of every reference block.
/// - `populations`: Excitor populations owned by this rank, references included.
/// - `nref`: Global reference populations `N_z`.
/// - `ranks`: Number of ranks `P` holding the distributed excitors.
/// # Returns:
/// - `Vec<Block>`: Block populations of this rank.
fn blocks(
    spaces: &ReferenceSpaces,
    populations: &FastMap<ExcitorKey, f64>,
    nref: &[f64],
    ranks: f64,
) -> Vec<Block> {
    let mut keys = populations.keys().copied().collect::<Vec<_>>();
    keys.par_sort_unstable();
    let mut out = spaces
        .references
        .iter()
        .zip(nref)
        .map(|(reference, &nref)| (*reference, nref, Vec::new()))
        .collect::<Vec<_>>();

    for key in keys {
        let (reference, _, excitors) = &mut out[key.0];
        let det = ExcitedDeterminant {
            parent: reference.parent,
            alpha: key.1,
            beta: key.2,
        };
        if det != *reference {
            excitors.push((Excitor::new(reference, det), populations[&key]));
        }
    }
    out.into_iter()
        .map(|(reference, nref, excitors)| {
            let weights = excitors.iter().map(|(_, n)| n.abs()).collect::<Vec<_>>();

            // Excitors of each level and the weights of the coupling combinations.
            let levels = (1..=spaces.cluster_rank)
                .map(|j| {
                    let indices = (0..excitors.len())
                        .filter(|&i| rank(&reference, &excitors[i].0.det) == j)
                        .collect::<Vec<_>>();
                    let level = indices.iter().map(|&i| weights[i]).collect::<Vec<_>>();
                    (indices, AliasTable::new(&level))
                })
                .collect::<Vec<_>>();
            let totals = levels.iter().map(|(_, t)| t.total()).collect::<Vec<_>>();
            let (combinations, coupling) =
                coupling_combinations(&totals, nref, spaces.expansion_rank, ranks);

            Block {
                reference,
                nref,
                selection: AliasTable::new(&weights),
                levels,
                combinations,
                coupling: AliasTable::new(&coupling),
                excitors,
            }
        })
        .collect()
}

/// Run one parallel task of cluster attempts, queueing their matrix elements and resolving them
/// in one batch at the end.
/// # Arguments:
/// - `data`: NOCI data holding the parent frames, MO caches and Wick intermediates.
/// - `spaces`: Projection and expansion spaces of every reference block.
/// - `blocks`: Block populations of this rank.
/// - `task`: Task kind with its block and number of attempts.
/// - `attempts`: Attempts of the task's block on this rank and the number of ranks.
/// - `rng`: Task random-number generator.
/// - `out`: Reusable task storage receiving the population changes and projections.
/// # Returns:
/// - `()`: Overwrites `out` with the task's results.
fn run_task(
    data: &NOCIData<'_, f64>,
    spaces: &ReferenceSpaces,
    blocks: &[Block],
    task: Task,
    attempts: (f64, f64),
    mut rng: QmcRng,
    out: &mut JMTaskPropagation,
) {
    let (attempts, ranks) = attempts;
    out.clear();

    match task {
        Task::Linked(z, count) => {
            let smax = 4.min(spaces.cluster_rank + 2);
            for _ in 0..count {
                let weight = select_cluster(
                    &blocks[z],
                    smax,
                    attempts,
                    ranks,
                    &mut rng,
                    &mut out.excitors,
                );
                linked_attempt(
                    spaces,
                    z,
                    (&out.excitors, weight),
                    &mut rng,
                    &mut out.linked_requests,
                );
            }
            out.resolve_linked(data);
        }
        Task::Coupling(x, count) => {
            for _ in 0..count {
                let weight =
                    select_coupling_cluster(&blocks[x], attempts, &mut rng, &mut out.excitors);
                if weight == 0.0 {
                    continue;
                }
                let Some((det, sign)) = collapse(&spaces.references[x], &out.excitors) else {
                    continue;
                };
                if rank(&spaces.references[x], &det) > spaces.expansion_rank {
                    continue;
                }
                project_cluster(
                    &spaces.rows,
                    x,
                    (det.state(data.space.parents()), weight * sign),
                    &mut rng,
                    (&mut out.coupling_clusters, &mut out.coupling_requests),
                );
            }
            out.resolve_coupling(data, spaces);
        }
    }
}

/// Trial and initial populations: the reference NOCI coefficients normalised to
/// `\sum_z|c_z| = 1`, or uniform coefficients without them, set the trial and, scaled by the
/// initial population, the reference populations on their owning ranks.
/// # Arguments:
/// - `spaces`: Projection and expansion spaces of every reference block.
/// - `c0`: Reference NOCI coefficients.
/// - `initial`: Initial population.
/// - `ownership`: Salt of the first assignment, this rank and the number of ranks.
/// # Returns:
/// - `(Vec<f64>, FastMap<ExcitorKey, f64>)`: Trial coefficients and this rank's populations.
fn initial_populations(
    spaces: &ReferenceSpaces,
    c0: &[f64],
    initial: f64,
    ownership: (u64, usize, usize),
) -> (Vec<f64>, FastMap<ExcitorKey, f64>) {
    let (assignment, irank, nranks) = ownership;
    let m = spaces.references.len();
    let norm = c0.iter().map(|c| c.abs()).sum::<f64>();
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
        let key = (z, reference.alpha, reference.beta);
        if owner(&key, assignment, nranks) == irank {
            populations.insert(key, initial * trial[z]);
        }
    }
    (trial, populations)
}

/// Exact effective overlap and Hamiltonian of the uncorrelated references, the overlap and
/// Hamiltonian between the reference determinants, from which the first `G = S^{-1}H` is taken.
/// # Arguments:
/// - `data`: NOCI data over the reference determinants.
/// - `spaces`: Projection and expansion spaces of every reference block.
/// # Returns:
/// - `Effective`: Effective overlap and Hamiltonian at zero amplitudes.
fn uncorrelated_effective(
    data: &NOCIData<'_, f64>,
    spaces: &ReferenceSpaces,
) -> Effective {
    let m = spaces.references.len();
    let mut scratch = WickScratchSpin::new();
    let mut effective = Effective::new(m);
    let pairs = spaces
        .references
        .iter()
        .flat_map(|bra| spaces.references.iter().map(|ket| (*bra, *ket)))
        .collect::<Vec<_>>();
    let mut hs = vec![(0.0, 0.0); pairs.len()];
    calculate_hs_excited(data, &pairs, &mut scratch, &mut hs);
    for (k, &(h, s)) in hs.iter().enumerate() {
        effective.se[(k / m, k % m)] = s;
        effective.he[(k / m, k % m)] = h;
    }
    effective
}

/// Run one step's cluster attempts. Each block's attempts are divided into linked and, with
/// several references, coupling tasks of at least `CHUNK` attempts, at most `TASKSPERTHREAD`
/// per Rayon thread, and each task runs on its own random stream into its reusable storage.
/// # Arguments:
/// - `data`: NOCI data holding the parent frames, MO caches and Wick intermediates.
/// - `spaces`: Projection and expansion spaces of every reference block.
/// - `current`: Block populations of this rank.
/// - `attempts`: Attempts of every block on this rank and the number of ranks.
/// - `seed`: Rank-local seed and iteration selecting the task streams.
/// - `workers`: Reusable task storage, grown to the number of tasks.
/// # Returns:
/// - `usize`: Number of tasks, whose results are in the leading `workers`.
fn run_tasks(
    data: &NOCIData<'_, f64>,
    spaces: &ReferenceSpaces,
    current: &[Block],
    attempts: (&[f64], f64),
    seed: (u64, usize),
    workers: &mut Vec<JMTaskPropagation>,
) -> usize {
    let (attempts, ranks) = attempts;
    let (local, iteration) = seed;
    let m = current.len();

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
    if workers.len() < tasks.len() {
        workers.resize_with(tasks.len(), || JMTaskPropagation::new(m));
    }

    tasks
        .par_iter()
        .zip(workers.par_iter_mut())
        .enumerate()
        .for_each(|(k, (&task, worker))| {
            let z = match task {
                Task::Linked(z, _) | Task::Coupling(z, _) => z,
            };
            let rng = stream(local, (iteration as u64) << 32, k as u64);
            run_task(
                data,
                spaces,
                current,
                task,
                (attempts[z], ranks),
                rng,
                worker,
            );
        });
    tasks.len()
}

/// Combine the tasks in order: spawns per unit time are summed per excitor into `merged`, and
/// the block energies `N_zH^e_{zz}` and effective elements `N_xS^e_{yx}` and `N_xH^e_{yx}` into
/// one vector `[N_zH^e_{zz}, N_xS^e_{yx}, N_xH^e_{yx}]` to be summed over ranks.
/// # Arguments:
/// - `workers`: Storage of the step's tasks.
/// - `m`: Number of references.
/// - `merged`: Reusable map receiving the summed spawns.
/// # Returns:
/// - `Vec<f64>`: This rank's energies and effective elements, `m + 2m^2` values.
fn sum_tasks(
    workers: &[JMTaskPropagation],
    m: usize,
    merged: &mut FastMap<ExcitorKey, f64>,
) -> Vec<f64> {
    merged.clear();
    for worker in workers {
        for (&key, &dn) in &worker.spawns {
            *merged.entry(key).or_insert(0.0) += dn;
        }
    }
    let mut sums = vec![0.0; m + 2 * m * m];
    for worker in workers {
        for (e, v) in sums[..m].iter_mut().zip(&worker.energy) {
            *e += v;
        }
        for y in 0..m {
            for x in 0..m {
                sums[m + y * m + x] += worker.projections.se[y][x];
                sums[m + m * m + y * m + x] += worker.projections.he[y][x];
            }
        }
    }
    sums
}

/// Add the coupling between references to the spawns: the projections
/// `\sum_{x \ne y}\tilde G_{xy}q^x_y` are completed over ranks, then the left action of
/// `e^{-\hat T_y}` on this rank's excitors of every block `y` gives its population changes.
/// # Arguments:
/// - `workers`: Storage of the step's tasks.
/// - `gt`: Scaled effective Hamiltonian `\tilde G`.
/// - `current`: Block populations of this rank.
/// - `spaces`: Projection and expansion spaces of every reference block.
/// - `comm`: MPI communicator and exchange scratch.
/// - `coupled`: Reusable map of this rank's projections.
/// - `merged`: Spawns per unit time, receiving the coupling.
/// # Returns:
/// - `()`: Adds the coupling to `merged`.
fn couple<C: Communicator>(
    workers: &[JMTaskPropagation],
    gt: &Array2<f64>,
    current: &[Block],
    spaces: &ReferenceSpaces,
    comm: (&C, &mut ExchangeScratch<ExcitorUpdate>),
    coupled: &mut FastMap<ExcitorKey, f64>,
    merged: &mut FastMap<ExcitorKey, f64>,
) {
    let (world, exchange) = comm;
    let irank = world.rank() as usize;
    let nranks = world.size() as usize;
    let m = current.len();

    coupled.clear();
    for worker in workers {
        for y in 0..m {
            for x in (0..m).filter(|&x| x != y) {
                for (&(alpha, beta), &v) in &worker.projections.q[y][x] {
                    *coupled.entry((y, alpha, beta)).or_insert(0.0) += gt[(x, y)] * v;
                }
            }
        }
    }
    let q = coupled
        .iter()
        .map(|(&key, &v)| ExcitorUpdate::new(key, v))
        .collect::<Vec<_>>();
    let mut q = if nranks > 1 {
        gather_updates(world, &q, exchange).to_vec()
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
        for (key, dn) in left_action(
            b,
            y,
            &entries,
            spaces.cluster_rank,
            (nranks as f64, irank == 0),
        ) {
            *merged.entry(key).or_insert(0.0) += dn;
        }
    }
}

/// Send the compressed spawns, and the populations whose owner changes under the next
/// assignment, to their next owners in one exchange, and annihilate what arrives into the
/// populations.
/// # Arguments:
/// - `populations`: Excitor populations owned by this rank, references included.
/// - `updates`: Compressed spawns of this rank.
/// - `next`: Salt of the next iteration's assignment.
/// - `comm`: MPI communicator and exchange scratch.
/// # Returns:
/// - `()`: Updates `populations` in place.
fn exchange_populations<C: Communicator>(
    populations: &mut FastMap<ExcitorKey, f64>,
    updates: Vec<ExcitorUpdate>,
    next: u64,
    comm: (&C, &mut ExchangeScratch<ExcitorUpdate>),
) {
    let (world, exchange) = comm;
    let irank = world.rank() as usize;
    let nranks = world.size() as usize;

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
        pack_updates(nranks, exchange, true);
        let mut received = exchange_updates(world, exchange).to_vec();
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
}

/// Round the owned excitor populations below the cutoff by pivotal FRI; references are never
/// removed.
/// # Arguments:
/// - `populations`: Excitor populations owned by this rank, references included.
/// - `is_reference`: Whether an excitor key is a block reference.
/// - `cutoff`: Population cutoff.
/// - `rng`: Rank-local rounding stream of the iteration.
/// # Returns:
/// - `()`: Updates `populations` in place.
fn round_populations(
    populations: &mut FastMap<ExcitorKey, f64>,
    is_reference: &impl Fn(&ExcitorKey) -> bool,
    cutoff: f64,
    mut rng: QmcRng,
) {
    let mut small = populations
        .iter()
        .filter(|(key, n)| !is_reference(key) && n.abs() < cutoff)
        .map(|(&key, &n)| ExcitorUpdate::new(key, n))
        .collect::<Vec<_>>();
    small.sort_unstable_by_key(|u| u.excitor());
    for update in &small {
        populations.remove(&update.excitor());
    }
    compress_sparse(&mut small, cutoff, &mut rng);
    populations.extend(small.iter().map(|u| (u.excitor(), u.dn)));
}

/// Population statistics over every rank: the total population `\sum|n|`, the reference
/// population `\sum_x|N_x|` and the number of occupied excitors.
/// # Arguments:
/// - `populations`: Excitor populations owned by this rank, references included.
/// - `is_reference`: Whether an excitor key is a block reference.
/// - `world`: MPI communicator.
/// # Returns:
/// - `(f64, f64, i64)`: Total population, reference population and occupied excitors.
fn population_stats(
    populations: &FastMap<ExcitorKey, f64>,
    is_reference: &impl Fn(&ExcitorKey) -> bool,
    world: &impl Communicator,
) -> (f64, f64, i64) {
    let mut stats = vec![0.0; 3];
    for (key, n) in populations {
        stats[0] += n.abs();
        if is_reference(key) {
            stats[1] += n.abs();
        } else {
            stats[2] += 1.0;
        }
    }
    let stats = all_reduce_array1(world, Array1::from_vec(stats));
    (stats[0], stats[1], stats[2] as i64)
}

/// Stochastically propagate the JM-NOCCMC amplitudes by linked coupled cluster Monte Carlo
/// with the plain update `t \leftarrow t - \Delta\tau R` and one population-control shift.
/// Populations `n_{x\mu}` live on the excitors of each reference block and the reference
/// populations `N_x` carry the normalisation, `t_{x\mu}=n_{x\mu}/N_x`. With
/// `\tilde G_{xy}=G_{xy}N_y/N_x` and `g_x=\sum_y\tilde G_{xy}`, every population is updated as
/// `n_{x\mu} \leftarrow n_{x\mu} - \Delta\tau N_xR_{x\mu} - \Delta\tau(g_x-E_s)n_{x\mu}`, with
/// `R_{x0}=0` for the reference and `E_s` the shift. The reference populations then follow the
/// power iteration `\Delta N=-\Delta\tau(G-E_s)N` of the effective Hamiltonian, converging to its
/// ground-state coefficients `c_x^{(0)}`, while the amplitudes follow
/// `t_{x\mu} \leftarrow t_{x\mu} - \Delta\tau R_{x\mu}/[1-\Delta\tau(g_x-E_s)]`, the plain update
/// once the shift balances the growth. The same-reference part of `N_xR_{x\mu}` is sampled by
/// linked CCMC, the coupling between references through sampled projections.
/// `G=\bar S^{-1}\bar H` is taken from running sums `\bar S \leftarrow \lambda\bar S+S^e` and
/// `\bar H \leftarrow \lambda\bar H+H^e` of the report blocks so far, with
/// `\lambda = 1 - 1/\text{memory\_effective}`, so that one noisy
/// block with a nearly singular `S^e` cannot make `G` diverge.
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
    let dt = options.dt;

    // One base seed for every rank, from which the shared assignments and the rank streams follow.
    let mut base = qmc.seed.unwrap_or_else(rand::random);
    world.process_at_rank(0).broadcast_into(&mut base);
    let local = rank_seed(base, irank);
    let assignment = |iteration: usize| base ^ (iteration as u64).wrapping_mul(ASSIGNSTRIDE);

    let spaces = ReferenceSpaces::new(data, options);
    let is_reference = |key: &ExcitorKey| {
        let reference = &spaces.references[key.0];
        (key.1, key.2) == (reference.alpha, reference.beta)
    };

    // Initial reference populations from the NOCI coefficients and the exact effective
    // Hamiltonian of uncorrelated references, `G = S^{-1}H`.
    let (trial, mut populations) = initial_populations(
        &spaces,
        c0,
        qmc.initial_population,
        (assignment(0), irank, nranks),
    );
    let uncorrelated = uncorrelated_effective(data, &spaces);
    let mut lagged = uncorrelated.g().expect("singular reference overlap");
    let e0 = uncorrelated.he[(0, 0)];

    print_table_header(irank, ["NWalk", "NRef", "-", "NExcitors"]);
    let mut reached = false;
    let local_norm = populations.values().map(|n: &f64| n.abs()).sum::<f64>();
    let mut previous = all_reduce_array1(world, Array1::from_vec(vec![local_norm]))[0];
    let mut exchange = ExchangeScratch::new(nranks);
    let mut workers = Vec::<JMTaskPropagation>::new();
    let mut merged = FastMap::<ExcitorKey, f64>::default();
    let mut coupled = FastMap::<ExcitorKey, f64>::default();
    let mut running = Effective::new(m);
    let mut history = Vec::new();
    let mut energy = f64::NAN;

    for report in 0..qmc.nreports {
        let mut block = Effective::new(m);
        let mut projected = (0.0, 0.0);

        for cycle in 0..qmc.ncycles {
            let iteration = report * qmc.ncycles + cycle;
            let ranks = nranks as f64;

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
            let current = blocks(&spaces, &populations, &nref, ranks);
            let gt = scaled(&lagged, &nref);

            // Every block's attempts, shared evenly between ranks, and the coupling.
            let attempts = (0..m)
                .map(|z| ((nref[z].abs() + totals[m + z]) / ranks).ceil().max(1.0))
                .collect::<Vec<_>>();
            let ntasks = run_tasks(
                data,
                &spaces,
                &current,
                (&attempts, ranks),
                (local, iteration),
                &mut workers,
            );
            let sums = sum_tasks(&workers[..ntasks], m, &mut merged);
            if m > 1 {
                couple(
                    &workers[..ntasks],
                    &gt,
                    &current,
                    &spaces,
                    (world, &mut exchange),
                    &mut coupled,
                    &mut merged,
                );
            }

            // Annihilate this rank's spawns and round small ones stochastically.
            let mut updates = merged
                .iter()
                .map(|(&key, &dn)| ExcitorUpdate::new(key, dt * dn))
                .collect::<Vec<_>>();
            updates.par_sort_unstable_by_key(|u| u.excitor());
            coalesce(&mut updates);
            let mut rng = stream(local, SPAWNSALT, iteration as u64);
            compress_sparse(&mut updates, qmc.fri.spawn_cutoff, &mut rng);

            // Sum the step's effective overlap and Hamiltonian over ranks, normalised by `N_x`.
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

            // Fixed-point-preserving death `-\Delta\tau(g_x-E_s)n` on every owned population.
            let growth = (0..m).map(|x| gt.row(x).sum()).collect::<Vec<_>>();
            for (key, n) in populations.iter_mut() {
                *n *= 1.0 - dt * (growth[key.0] - shift);
            }

            exchange_populations(
                &mut populations,
                updates,
                assignment(iteration + 1),
                (world, &mut exchange),
            );
            round_populations(
                &mut populations,
                &is_reference,
                qmc.fri.population_cutoff,
                stream(local, ROUNDSALT, iteration as u64),
            );
        }

        // Report-block energy and the lagged effective Hamiltonian from the running sums with
        // forgetting.
        if let Some(g) = block.g() {
            energy = jm_energies(&g)[0].re;
        }
        running.scale(1.0 - 1.0 / options.memory_effective);
        running.add(&block);
        if let Some(g) = running.g() {
            lagged = g;
        }
        if reached {
            history.push(block);
        }

        // Population statistics over every rank and the shift update.
        let (total, nref, nexcitors) = population_stats(&populations, &is_reference, world);
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
            (total, nref, None, Some(nexcitors)),
        );

        if stop_requested(irank, nranks, world) {
            break;
        }
    }

    // Ratio estimate over every report block after the shift starts, with its reblocked error,
    // or without an error if there are too few blocks.
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
    let mut total = Effective::new(m);
    for block in &history {
        total.add(block);
    }
    ground_energy(&total).unwrap_or(energy)
}
