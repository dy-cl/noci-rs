// nocc/stochastic/propagate.rs

// External crate imports.
use mpi::topology::Communicator;
use mpi::traits::*;
use ndarray::{Array1, Array2};
use rayon::prelude::*;

// Crate-root imports.
use crate::elements::nonorthogonalwicks::WickScratchSpin;
use crate::elements::{DetPair, HamiltonianOverlap, NOCIData, calculate_pairs};
use crate::hash::FastMap;
use crate::input::{JMNOCCMCOptions, QMCOptions};
use crate::mpiutils::all_reduce_array1;
use crate::nocc::Effective;
use crate::qmc::{
    AliasTable, ExchangeScratch, PopulationUpdate, Propagator, QmcRng, coalesce, compress_sparse,
    exchange_to_owners, gather_updates, owner_words, print_table_header, print_table_row,
    rank_seed, run, stream, update_shift,
};

// Parent/sibling imports.
use super::cluster::{collapse, coupling_combinations, select_cluster, select_coupling_cluster};
use super::coupling::{left_action, project_cluster};
use super::linked::linked_attempt;
use super::population::{Block, Excitor, ExcitorIndex, ReferenceSpaces, rank};
use super::state::JMTaskPropagation;

/// Smallest number of cluster attempts in one parallel task.
const CHUNK: usize = 256;

/// Number of parallel tasks per block and kind on each Rayon thread, bounding the per-task
/// spawn maps held at once.
const TASKSPERTHREAD: usize = 4;

/// Salt of the rank-local population-rounding stream.
const ROUNDSALT: u64 = 0x8EBC6AF09C88C6E3;

/// Multiplier giving each report its own excitor-to-rank assignment.
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

/// Rank owning an excitor during one report. The assignment is redrawn every report from a salt
/// shared by all ranks, so that over the assignment any `s` excitors share a rank with
/// probability `P^{-(s-1)}`.
/// # Arguments:
/// - `key`: Excitor.
/// - `assignment`: Salt of the report's assignment.
/// - `nranks`: Number of ranks `P`.
/// # Returns:
/// - `usize`: Owning rank.
fn owner(
    key: ExcitorIndex,
    assignment: u64,
    nranks: usize,
) -> usize {
    if nranks == 1 {
        return 0;
    }
    owner_words(&[assignment, key.0 as u64], nranks)
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
    populations: &FastMap<ExcitorIndex, f64>,
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

    for key in keys.into_iter().filter(|&key| !spaces.is_reference(key)) {
        let (reference, _, excitors) = &mut out[spaces.block(key)];
        let excitor = Excitor::new(reference, spaces.determinant(key));
        excitors.push((excitor, populations[&key]));
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
                    (det, weight * sign),
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
/// - `(Vec<f64>, FastMap<ExcitorIndex, f64>)`: Trial coefficients and this rank's populations.
fn initial_populations(
    spaces: &ReferenceSpaces,
    c0: &[f64],
    initial: f64,
    ownership: (u64, usize, usize),
) -> (Vec<f64>, FastMap<ExcitorIndex, f64>) {
    let (assignment, irank, nranks) = ownership;
    let m = spaces.references.len();

    // Trial `c_z/\sum_z|c_z|`, or uniform without usable NOCI coefficients.
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

    // Reference populations `N_z = N_0 c_z` on the ranks owning them, with no excitors.
    let mut populations = FastMap::default();
    for (z, reference) in spaces.references.iter().enumerate() {
        let key = spaces.index(z, (reference.alpha, reference.beta));
        if owner(key, assignment, nranks) == irank {
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

    // `\langle{}^z\Psi|\hat H|{}^x\Psi\rangle` and `\langle{}^z\Psi|{}^x\Psi\rangle` for every
    // ordered pair of references, row by row.
    let pairs = spaces
        .references
        .iter()
        .flat_map(|&bra| {
            spaces
                .references
                .iter()
                .map(move |&ket| DetPair::new(bra, ket))
        })
        .collect::<Vec<_>>();
    let mut hs = vec![(0.0, 0.0); pairs.len()];
    calculate_pairs(
        data,
        &HamiltonianOverlap,
        &pairs,
        Some(&mut scratch),
        &mut hs,
    );

    // At zero amplitudes `|\Phi_x\rangle = |{}^x\Psi\rangle`, so these are `S^e` and `H^e`.
    let mut effective = Effective::new(m);
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

    // Chunks of each block's attempts, one linked and, between references, one coupling task
    // per chunk.
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

    // Storage for every task, kept between steps.
    if workers.len() < tasks.len() {
        workers.resize_with(tasks.len(), || JMTaskPropagation::new(m));
    }

    // Each task draws from the stream of its iteration and index, so the results do not
    // depend on the thread that runs it.
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
    merged: &mut FastMap<ExcitorIndex, f64>,
) -> Vec<f64> {
    // Spawns of every task, summed per excitor.
    merged.clear();
    for worker in workers {
        for (&key, &dn) in &worker.spawns {
            *merged.entry(key).or_insert(0.0) += dn;
        }
    }

    // Energies in the first `m` entries, then `S^e` and `H^e` row by row.
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

/// Spawn the coupling between references,
/// `-\sum_{x \ne y}G_{xy}\langle{}^y\Psi|\hat\tau_{y\mu}^\dagger e^{-\hat T_y}|\Phi_x\rangle`
/// for every excitor `\mu` of every block `y`, scaled by `N_y`. The report's projections
/// `\sum_{x \ne y}\tilde G_{xy}q^x_y(\nu)`, summed over its cycles per row `\nu`, are completed
/// over ranks and acted on from the left by `e^{-\hat T_y}` with this rank's excitors of `y`.
/// # Arguments:
/// - `current`: Block populations of this rank.
/// - `spaces`: Projection and expansion spaces of every reference block.
/// - `comm`: MPI communicator and exchange scratch.
/// - `coupled`: This rank's weighted projections of the report.
/// - `merged`: Spawns per unit time, receiving the coupling.
/// # Returns:
/// - `()`: Adds the coupling to `merged`.
fn spawn_coupling<C: Communicator>(
    current: &[Block],
    spaces: &ReferenceSpaces,
    comm: (&C, &mut ExchangeScratch<PopulationUpdate>),
    coupled: &FastMap<ExcitorIndex, f64>,
    merged: &mut FastMap<ExcitorIndex, f64>,
) {
    let (world, exchange) = comm;
    let irank = world.rank() as usize;
    let nranks = world.size() as usize;

    // Every rank needs the complete projection, since its excitors of `y` act on all rows.
    let q = coupled
        .iter()
        .map(|(&key, &dn)| PopulationUpdate {
            index: key.0 as u64,
            dn,
        })
        .collect::<Vec<_>>();
    let mut q = if nranks > 1 {
        gather_updates(world, &q, exchange).to_vec()
    } else {
        q
    };
    q.par_sort_unstable_by_key(|u| u.index);
    coalesce(&mut q);

    // Left action of `e^{-\hat T_y}` on the rows of each block, added to the spawns.
    for (y, b) in current.iter().enumerate() {
        let entries = q
            .iter()
            .map(|u| (ExcitorIndex(u.index as usize), u.dn))
            .filter(|&(key, _)| spaces.block(key) == y)
            .map(|(key, dn)| {
                let det = spaces.determinant(key);
                ((det.alpha, det.beta), dn)
            })
            .collect::<Vec<_>>();
        for (key, dn) in left_action(b, y, &entries, spaces, (nranks as f64, irank == 0)) {
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
    populations: &mut FastMap<ExcitorIndex, f64>,
    updates: Vec<PopulationUpdate>,
    next: u64,
    comm: (&C, &mut ExchangeScratch<PopulationUpdate>),
) {
    let (world, exchange) = comm;
    let irank = world.rank() as usize;
    let nranks = world.size() as usize;

    // With one rank every spawn and population stays where it is.
    let received = if nranks > 1 {
        // Route every spawn to the next owner of its excitor.
        exchange.send_ranked.clear();
        for update in updates {
            let peer = owner(ExcitorIndex(update.index as usize), next, nranks);
            exchange.send_ranked.push((peer, update));
        }

        // Move the populations whose excitor changes owner under the next assignment.
        let moving = populations
            .keys()
            .filter(|&&key| owner(key, next, nranks) != irank)
            .copied()
            .collect::<Vec<_>>();
        for key in moving {
            let n = populations.remove(&key).unwrap();
            let update = PopulationUpdate {
                index: key.0 as u64,
                dn: n,
            };
            exchange
                .send_ranked
                .push((owner(key, next, nranks), update));
        }

        exchange_to_owners(world, irank, exchange)
    } else {
        updates
    };

    // Annihilate the arrivals with the owned populations.
    for update in &received {
        *populations
            .entry(ExcitorIndex(update.index as usize))
            .or_insert(0.0) += update.dn;
    }
}

/// Round the owned excitor populations below the cutoff by pivotal FRI; references are never
/// removed.
/// # Arguments:
/// - `populations`: Excitor populations owned by this rank, references included.
/// - `spaces`: Spaces of every reference block, which tell its references apart.
/// - `cutoff`: Population cutoff.
/// - `rng`: Rank-local rounding stream of the iteration.
/// # Returns:
/// - `()`: Updates `populations` in place.
fn round_populations(
    populations: &mut FastMap<ExcitorIndex, f64>,
    spaces: &ReferenceSpaces,
    cutoff: f64,
    mut rng: QmcRng,
) {
    // Take out the small excitor populations in key order, so the rounding is reproducible.
    let mut small = populations
        .iter()
        .filter(|&(&key, n)| !spaces.is_reference(key) && n.abs() < cutoff)
        .map(|(&key, &n)| PopulationUpdate {
            index: key.0 as u64,
            dn: n,
        })
        .collect::<Vec<_>>();
    small.sort_unstable_by_key(|u| u.index);
    for update in &small {
        populations.remove(&ExcitorIndex(update.index as usize));
    }

    // Each survives as `\pm c` with probability `|n|/c` and the survivors return.
    compress_sparse(&mut small, cutoff, &mut rng);
    populations.extend(small.iter().map(|u| (ExcitorIndex(u.index as usize), u.dn)));
}

/// Population statistics over every rank: the total population `\sum|n|`, the reference
/// population `\sum_x|N_x|` and the number of occupied excitors.
/// # Arguments:
/// - `populations`: Excitor populations owned by this rank, references included.
/// - `spaces`: Spaces of every reference block, which tell its references apart.
/// - `world`: MPI communicator.
/// # Returns:
/// - `(f64, f64, i64)`: Total population, reference population and occupied excitors.
fn population_stats(
    populations: &FastMap<ExcitorIndex, f64>,
    spaces: &ReferenceSpaces,
    world: &impl Communicator,
) -> (f64, f64, i64) {
    // This rank's contributions.
    let mut stats = vec![0.0; 3];
    for (&key, n) in populations {
        stats[0] += n.abs();
        if spaces.is_reference(key) {
            stats[1] += n.abs();
        } else {
            stats[2] += 1.0;
        }
    }

    // Summed over every rank.
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
/// As in NOCIQMC, the `n_\text{cyc}` cycles of a report sample the same frozen populations and
/// accumulate their spawns and coupling projections, and the report applies the accumulated
/// change with the death term as one step of `n_\text{cyc}\Delta\tau`.
/// The populations are distributed over the MPI ranks by a hash of the excitor redrawn every
/// report: each rank forms clusters from its own excitors with the weight `P^{s-1}`, its spawns
/// and the excitors changing owner are exchanged once per report, and annihilation happens on the
/// owner. Within a rank the attempts, the left action and the sorts run on Rayon threads.
/// # Arguments:
/// - `data`: NOCI data over the reference determinants.
/// - `options`: JM-NOCCMC options with stochastic settings.
/// - `c0`: Reference NOCI coefficients setting the initial reference populations.
/// - `shift`: Initial population-control shift.
/// - `world`: MPI communicator.
/// # Returns:
/// - `f64`: Projected energy of the last report, as stochastic NOCIQMC returns; error analysis
///   of the printed table is left to `scripts/blocking.py`.
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
    shift: f64,
    world: &impl Communicator,
) -> f64 {
    let mut jm = JMRun::new(data, options, c0, shift, world);
    let qmc = jm.qmc;
    run(
        &mut jm,
        (0..qmc.nreports, qmc.ncycles),
        (None, "RESTART.H5"),
        world,
    );
    jm.eproj
}

/// The populations of one report, frozen through its cycles: the global reference populations,
/// this rank's blocks, the scaled effective Hamiltonian and the attempts of every block.
struct ReportView {
    /// Global reference populations `N_z`.
    nref: Vec<f64>,
    /// This rank's view of every block.
    blocks: Vec<Block>,
    /// Effective Hamiltonian of the unnormalised correlated references, `\tilde G_{xy}`.
    gt: Array2<f64>,
    /// Cluster attempts of every block on this rank.
    attempts: Vec<f64>,
}

/// Build the frozen view of the populations for the next report.
/// # Arguments:
/// - `spaces`: Projection and expansion spaces of every reference block.
/// - `populations`: Excitor populations owned by this rank, references included.
/// - `lagged`: Lagged effective Hamiltonian `G`.
/// - `world`: MPI communicator.
/// # Returns:
/// - `ReportView`: Reference populations, blocks, `\tilde G` and attempts of the report.
fn report_view(
    spaces: &ReferenceSpaces,
    populations: &FastMap<ExcitorIndex, f64>,
    lagged: &Array2<f64>,
    world: &impl Communicator,
) -> ReportView {
    let m = spaces.references.len();
    let ranks = world.size() as f64;

    // Global reference populations and excitor norms, then this rank's block view.
    let mut totals = vec![0.0; 2 * m];
    for (&key, &n) in populations {
        if spaces.is_reference(key) {
            totals[spaces.block(key)] += n;
        } else {
            totals[m + spaces.block(key)] += n.abs();
        }
    }
    let totals = all_reduce_array1(world, Array1::from_vec(totals)).to_vec();
    let nref = totals[..m].to_vec();
    let blocks = blocks(spaces, populations, &nref, ranks);

    // Effective Hamiltonian of the unnormalised correlated references
    // `|\tilde\Phi_x\rangle = N_x|\Phi_x\rangle`, `\tilde G_{xy} = G_{xy}N_y/N_x`, whose row sums
    // `g_x = (GN)_x/N_x` are the growth rates of the reference populations.
    let gt = Array2::from_shape_fn(lagged.dim(), |(x, y)| lagged[(x, y)] * nref[y] / nref[x]);

    // Every block's attempts, one per unit of its population, shared evenly between ranks.
    let attempts = (0..m)
        .map(|z| ((nref[z].abs() + totals[m + z]) / ranks).ceil().max(1.0))
        .collect();

    ReportView {
        nref,
        blocks,
        gt,
        attempts,
    }
}

/// One JM-NOCCMC run on this rank: the spaces and populations, the effective Hamiltonian and the
/// shift carried between reports, and what the current report accumulates over its cycles.
struct JMRun<'a, C> {
    /// NOCI data over the reference determinants.
    data: &'a NOCIData<'a, f64>,
    /// MPI communicator.
    world: &'a C,
    /// JM-NOCCMC options.
    options: &'a JMNOCCMCOptions,
    /// Stochastic settings of the run.
    qmc: &'a QMCOptions,
    /// Projection and expansion spaces of every reference block.
    spaces: ReferenceSpaces,
    /// Trial coefficients `c_z` of the projected energy.
    trial: Vec<f64>,
    /// Excitor populations owned by this rank, references included.
    populations: FastMap<ExcitorIndex, f64>,
    /// Base seed shared by every rank and this rank's seed.
    seeds: (u64, u64),
    /// Energy of the first reference, the zero of the reported shift.
    e0: f64,
    /// Lagged effective Hamiltonian `G`.
    lagged: Array2<f64>,
    /// Running effective overlap and Hamiltonian sums with forgetting.
    running: Effective,
    /// Population-control shift `E_s`.
    shift: f64,
    /// Whether the population has reached its target.
    reached: bool,
    /// Population norm after the previous report.
    previous: f64,
    /// Frozen populations of the current report.
    view: ReportView,
    /// Effective overlap and Hamiltonian of the current report.
    block: Effective,
    /// Projected-energy numerator and denominator summed over the current report's cycles.
    projected: (f64, f64),
    /// Spawns per unit time accumulated over the current report.
    spawns: FastMap<ExcitorIndex, f64>,
    /// Weighted coupling projections accumulated over the current report.
    coupled: FastMap<ExcitorIndex, f64>,
    /// Reusable spawns of one cycle.
    merged: FastMap<ExcitorIndex, f64>,
    /// Per-task propagation storage.
    workers: Vec<JMTaskPropagation>,
    /// Population exchange scratch.
    exchange: ExchangeScratch<PopulationUpdate>,
    /// Projected-energy terms per cycle and population statistics of the last report.
    row: ((f64, f64), (f64, f64, i64)),
    /// Projected energy of the last report.
    eproj: f64,
}

impl<'a, C: Communicator> JMRun<'a, C> {
    /// Start a run from the reference NOCI coefficients and print the report header.
    /// # Arguments:
    /// - `data`: NOCI data over the reference determinants.
    /// - `options`: JM-NOCCMC options with stochastic settings.
    /// - `c0`: Reference NOCI coefficients setting the initial reference populations.
    /// - `shift`: Initial population-control shift.
    /// - `world`: MPI communicator.
    /// # Returns:
    /// - `Self`: Run at its first report.
    /// # Panics
    /// - Panics if the stochastic options are absent or the reference overlap is singular.
    fn new(
        data: &'a NOCIData<'a, f64>,
        options: &'a JMNOCCMCOptions,
        c0: &[f64],
        shift: f64,
        world: &'a C,
    ) -> Self {
        let qmc = options
            .qmc
            .as_ref()
            .expect("jmnoccmc.qmc options are required");
        let irank = world.rank() as usize;
        let nranks = world.size() as usize;
        let m = data.space.len();

        // One base seed for every rank, from which the shared assignments and the rank streams
        // follow.
        let mut base = qmc.seed.unwrap_or_else(rand::random);
        world.process_at_rank(0).broadcast_into(&mut base);

        // Fixed spaces of every block, which also index their excitors.
        let spaces = ReferenceSpaces::new(data, options);

        // Initial reference populations from the NOCI coefficients and the exact effective
        // Hamiltonian of uncorrelated references, `G = S^{-1}H`.
        let (trial, populations) =
            initial_populations(&spaces, c0, qmc.initial_population, (base, irank, nranks));
        let uncorrelated = uncorrelated_effective(data, &spaces);
        let lagged = uncorrelated.g().expect("singular reference overlap");

        print_table_header(irank, ["NWalk", "NRef", "-", "NExcitors"]);
        let local_norm = populations.values().map(|n: &f64| n.abs()).sum::<f64>();
        let previous = all_reduce_array1(world, Array1::from_vec(vec![local_norm]))[0];

        Self {
            data,
            world,
            options,
            qmc,
            view: report_view(&spaces, &populations, &lagged, world),
            spaces,
            trial,
            populations,
            seeds: (base, rank_seed(base, irank)),
            e0: uncorrelated.he[(0, 0)],
            lagged,
            running: Effective::new(m),
            shift,
            reached: false,
            previous,
            block: Effective::new(m),
            projected: (0.0, 0.0),
            spawns: FastMap::default(),
            coupled: FastMap::default(),
            merged: FastMap::default(),
            workers: Vec::new(),
            exchange: ExchangeScratch::new(nranks),
            row: ((0.0, 0.0), (0.0, 0.0, 0)),
            eproj: f64::NAN,
        }
    }
}

impl<C: Communicator> Propagator for JMRun<'_, C> {
    /// Run one cycle of linked and coupling attempts on the frozen populations, accumulating its
    /// spawns, coupling projections, energies and effective elements over the report.
    /// # Arguments:
    /// - `self`: JM-NOCCMC run.
    /// - `iteration`: Global iteration of the cycle.
    /// # Returns
    /// - `()`: Accumulates the cycle into the report.
    fn cycle(
        &mut self,
        iteration: usize,
    ) {
        let m = self.spaces.references.len();
        let view = &self.view;

        // Linked and coupling attempts per unit time, with the cycle's energies and effective
        // elements.
        let ntasks = run_tasks(
            self.data,
            &self.spaces,
            &view.blocks,
            (&view.attempts, self.world.size() as f64),
            (self.seeds.1, iteration),
            &mut self.workers,
        );
        let sums = sum_tasks(&self.workers[..ntasks], m, &mut self.merged);
        for (&key, &dn) in &self.merged {
            *self.spawns.entry(key).or_insert(0.0) += dn;
        }

        // `\sum_{x \ne y}\tilde G_{xy}q^x_y(\nu)` keyed by block `y` and row `\nu`.
        for worker in &self.workers[..ntasks] {
            for y in 0..m {
                for x in (0..m).filter(|&x| x != y) {
                    for (&(alpha, beta), &v) in &worker.projections.q[y][x] {
                        let key = self.spaces.index(y, (alpha, beta));
                        *self.coupled.entry(key).or_insert(0.0) += view.gt[(x, y)] * v;
                    }
                }
            }
        }

        // Sum the cycle's effective overlap and Hamiltonian over ranks, normalised by `N_x`.
        let sums = all_reduce_array1(self.world, Array1::from_vec(sums)).to_vec();
        let nref = &view.nref;

        // Projected energy onto the fixed trial `\langle\Psi_T| = \sum_z c_z\langle{}^z\Psi|`:
        // `\sum_z c_z\sum_x\langle{}^z\Psi|\hat H|\tilde\Phi_x\rangle` over
        // `\sum_z c_z\sum_x\langle{}^z\Psi|\tilde\Phi_x\rangle`.
        for z in 0..m {
            self.projected.0 += self.trial[z] * sums[z];
            self.projected.1 += self.trial[z] * nref[z];
            for x in (0..m).filter(|&x| x != z) {
                self.projected.0 += self.trial[z] * sums[m + m * m + z * m + x];
                self.projected.1 += self.trial[z] * sums[m + z * m + x];
            }
        }

        // This cycle's `S^e_{yx}` and `H^e_{yx}` per unit `N_x`, added to the report block.
        for x in 0..m {
            self.block.se[(x, x)] += 1.0;
            self.block.he[(x, x)] += sums[x] / nref[x];
            for y in (0..m).filter(|&y| y != x) {
                self.block.se[(y, x)] += sums[m + y * m + x] / nref[x];
                self.block.he[(y, x)] += sums[m + m * m + y * m + x] / nref[x];
            }
        }
    }

    /// Apply the report's accumulated spawns and coupling with the death term as one step of
    /// `n_\text{cyc}\Delta\tau`, move the populations to the next assignment, and update the
    /// lagged `G` and the shift.
    /// # Arguments:
    /// - `self`: JM-NOCCMC run.
    /// - `report`: Report ending with this step.
    /// # Returns
    /// - `()`: Advances the populations and prepares the next report's view.
    fn step(
        &mut self,
        report: usize,
    ) {
        let qmc = self.qmc;
        let dt = self.options.dt;
        let m = self.spaces.references.len();

        // The coupling of the report's summed projections joins the spawns.
        if m > 1 {
            spawn_coupling(
                &self.view.blocks,
                &self.spaces,
                (self.world, &mut self.exchange),
                &self.coupled,
                &mut self.spawns,
            );
        }

        // Annihilate this rank's spawns of the report and round small ones stochastically.
        let mut updates = self
            .spawns
            .iter()
            .map(|(&key, &dn)| PopulationUpdate {
                index: key.0 as u64,
                dn: dt * dn,
            })
            .collect::<Vec<_>>();
        updates.par_sort_unstable_by_key(|u| u.index);
        coalesce(&mut updates);
        let mut rng = stream(self.seeds.1, SPAWNSALT, report as u64);
        compress_sparse(&mut updates, qmc.fri.spawn_cutoff, &mut rng);

        // Fixed-point-preserving death `-n_\text{cyc}\Delta\tau(g_x-E_s)n` on every owned
        // population, the report's step of the frozen populations.
        let growth = (0..m)
            .map(|x| self.view.gt.row(x).sum())
            .collect::<Vec<_>>();
        let step = qmc.ncycles as f64 * dt;
        for (&key, n) in self.populations.iter_mut() {
            *n *= 1.0 - step * (growth[self.spaces.block(key)] - self.shift);
        }

        // Spawns and populations to their owners under the next report's assignment, annihilation,
        // then rounding of small excitors.
        exchange_populations(
            &mut self.populations,
            updates,
            self.seeds.0 ^ ((report + 1) as u64).wrapping_mul(ASSIGNSTRIDE),
            (self.world, &mut self.exchange),
        );
        round_populations(
            &mut self.populations,
            &self.spaces,
            qmc.fri.population_cutoff,
            stream(self.seeds.1, ROUNDSALT, report as u64),
        );

        // Lagged effective Hamiltonian from the running sums with forgetting.
        self.running
            .scale(1.0 - 1.0 / self.options.memory_effective);
        self.running.add(&self.block);
        if let Some(g) = self.running.g() {
            self.lagged = g;
        }

        // Population statistics over every rank and the shift update.
        let stats = population_stats(&self.populations, &self.spaces, self.world);
        update_shift(
            &mut self.shift,
            &mut self.reached,
            (stats.0, self.previous),
            (qmc.target_population, qmc.population_restoring),
            Some(qmc.shift_damping / (dt * qmc.ncycles as f64)),
        );
        self.previous = stats.0;

        // Keep the report's row, averaged over its cycles, then start the next report.
        let cycles = qmc.ncycles as f64;
        self.eproj = self.projected.0 / self.projected.1;
        self.row = (
            (self.projected.0 / cycles, self.projected.1 / cycles),
            stats,
        );
        self.block = Effective::new(m);
        self.projected = (0.0, 0.0);
        self.spawns.clear();
        self.coupled.clear();
        self.view = report_view(&self.spaces, &self.populations, &self.lagged, self.world);
    }

    /// JM-NOCCMC does not write restarts yet.
    /// # Arguments:
    /// - `self`: JM-NOCCMC run.
    /// - `_`: Last completed report.
    /// # Returns
    /// - `()`: Writes nothing.
    fn write_restart(
        &mut self,
        _: usize,
    ) {
    }

    /// Print the report's populations, projected-energy terms and shift.
    /// # Arguments:
    /// - `self`: JM-NOCCMC run.
    /// - `end`: Iteration count at the end of the report.
    /// # Returns
    /// - `()`: Prints one row on the root rank.
    fn print_row(
        &self,
        end: usize,
    ) {
        let (energy, (total, nref, nexcitors)) = self.row;
        print_table_row(
            self.world.rank() as usize,
            end,
            energy,
            (self.e0, if self.reached { self.shift } else { 0.0 }),
            (total, nref, None, Some(nexcitors)),
        );
    }
}
