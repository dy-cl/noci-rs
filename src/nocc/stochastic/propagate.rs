// nocc/stochastic/propagate.rs

// External crate imports.
use mpi::collective::SystemOperation;
use mpi::topology::Communicator;
use mpi::traits::*;
use rayon::prelude::*;

// Crate-root imports.
use crate::determinant::{ExcitedDeterminant, NOCIIndex};
use crate::elements::nonorthogonalwicks::WickScratchSpin;
use crate::elements::{NOCIData, calculate_hs_excited};
use crate::input::JMNOCCMCOptions;
use crate::nocc::{jm_energies, reference_determinants};
use crate::qmc::excitgen::OrthogonalUniformGenerator;
use crate::qmc::{
    QmcRng, coalesce, compress_sparse, cutoff_candidate, gather_updates, pivotal_push, rank_seed,
    resolve_pivotal, stop_requested, stream, update_shift,
};

// Parent/sibling imports.
use super::cluster::{collapse, select_cluster};
use super::coupling::{Projections, left_action, project_cluster};
use super::effective::{Effective, scaled};
use super::linked::{LinkedBlock, linked_attempt};
use super::population::{Block, Excitor, ExcitorKey, ExcitorUpdate, FixedMap, rank};
use super::report::{print_header, print_row};

/// Number of cluster attempts in one parallel task.
const CHUNK: usize = 256;

/// Salt of the population-rounding stream shared by every rank.
const ROUNDSALT: u64 = 0x8EBC6AF09C88C6E3;

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
    /// Spawned population changes.
    spawns: Vec<(ExcitorKey, f64)>,
    /// Contribution to `N_z H^e_{zz}` of each block.
    energy: Vec<f64>,
    /// Coupling projections.
    projections: Projections,
}

/// Fixed data of the propagation.
struct Setup<'a> {
    /// NOCI data holding the parent frames, MO caches and Wick intermediates.
    data: &'a NOCIData<'a, f64>,
    /// Block reference determinants.
    references: Vec<ExcitedDeterminant>,
    /// Projection rows of every block with ranks `1..=cluster_rank`.
    rows: Vec<Vec<ExcitedDeterminant>>,
    /// Number of spatial orbitals of every block.
    norb: Vec<usize>,
    /// Uniform connection generator of every block.
    generators: Vec<OrthogonalUniformGenerator>,
    /// Highest excitation rank of the cluster operators.
    cluster_rank: usize,
    /// Highest rank and cluster size of the expansion of each `e^{\hat T_x}|{}^x\Psi\rangle`.
    expansion_rank: usize,
    /// Time step `\Delta\tau`.
    dt: f64,
}

/// Build the per-step view of every block from the replicated populations, with excitors in
/// key order so that every rank selects clusters identically.
/// # Arguments:
/// - `setup`: Fixed propagation data.
/// - `populations`: Replicated excitor populations, references included.
/// # Returns:
/// - `Vec<Block>`: Block populations.
fn blocks(
    setup: &Setup<'_>,
    populations: &FixedMap<ExcitorKey, f64>,
) -> Vec<Block> {
    let mut keys = populations.keys().copied().collect::<Vec<_>>();
    keys.sort_unstable();
    let mut out = setup
        .references
        .iter()
        .zip(&setup.rows)
        .map(|(reference, rows)| Block {
            reference: *reference,
            nref: 0.0,
            excitors: Vec::new(),
            cumulative: Vec::new(),
            rows: rows.clone(),
        })
        .collect::<Vec<_>>();

    for key in keys {
        let n = populations[&key];
        let block = &mut out[key.0];
        let det = ExcitedDeterminant {
            parent: block.reference.parent,
            alpha: key.1,
            beta: key.2,
        };
        if det == block.reference {
            block.nref = n;
        } else {
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
/// - `setup`: Fixed propagation data.
/// - `blocks`: Block populations.
/// - `task`: Task kind with its block and number of attempts.
/// - `attempts`: Total attempts of the task's block.
/// - `rng`: Task random-number generator.
/// # Returns:
/// - `TaskOutput`: Spawns and accumulated projections.
fn run_task(
    setup: &Setup<'_>,
    blocks: &[Block],
    task: Task,
    attempts: f64,
    mut rng: QmcRng,
) -> TaskOutput {
    let m = blocks.len();
    let mut out = TaskOutput {
        spawns: Vec::new(),
        energy: vec![0.0; m],
        projections: Projections::new(m),
    };
    let mut scratch = WickScratchSpin::new();
    let mut excitors = Vec::new();

    match task {
        Task::Linked(z, count) => {
            let block = LinkedBlock {
                index: z,
                reference: setup.references[z],
                norb: setup.norb[z],
                cluster_rank: setup.cluster_rank,
                generator: &setup.generators[z],
            };
            let smax = 4.min(setup.cluster_rank + 2);
            for _ in 0..count {
                let weight = select_cluster(&blocks[z], smax, attempts, &mut rng, &mut excitors);
                out.energy[z] += linked_attempt(
                    setup.data,
                    &block,
                    (&excitors, weight),
                    setup.dt,
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
                    setup.expansion_rank,
                    attempts,
                    &mut rng,
                    &mut excitors,
                );
                let Some((det, sign)) = collapse(&setup.references[x], &excitors) else {
                    continue;
                };
                if rank(&setup.references[x], &det) > setup.expansion_rank {
                    continue;
                }
                project_cluster(
                    setup.data,
                    blocks,
                    x,
                    (&det, weight * sign),
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
/// Every rank holds the full sparse population; cluster attempts are divided between ranks and
/// their spawns gathered, annihilated and applied identically on every rank.
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
    let mocache = data.mocache.expect("JM-NOCCMC requires MO caches");
    let irank = world.rank() as usize;
    let nranks = world.size() as usize;
    let m = data.space.len();

    // One base seed for every rank, since tasks draw streams independent of their rank.
    let mut base = qmc.seed.unwrap_or_else(rand::random);
    world.process_at_rank(0).broadcast_into(&mut base);
    let local = rank_seed(base, irank);

    // Block references, projection rows and connection generators.
    let parents = data.space.parents();
    let mut setup = Setup {
        data,
        references: Vec::with_capacity(m),
        rows: Vec::with_capacity(m),
        norb: Vec::with_capacity(m),
        generators: Vec::with_capacity(m),
        cluster_rank: options.cluster_rank,
        expansion_rank: 0,
        dt: options.dt,
    };
    let mut nelec = 0;
    for z in 0..m {
        let parent = data.space.state(NOCIIndex(z)).parent;
        let (oa, ob) = data.space.occupations(NOCIIndex(z));
        let norb = parents[parent].ca.ncols();
        nelec = (oa.count_ones() + ob.count_ones()) as usize;
        setup.references.push(ExcitedDeterminant {
            parent,
            alpha: oa,
            beta: ob,
        });
        setup.rows.push(
            reference_determinants(oa, ob, norb, options.cluster_rank)
                .into_iter()
                .skip(1)
                .map(|(alpha, beta, _)| ExcitedDeterminant {
                    parent,
                    alpha,
                    beta,
                })
                .collect(),
        );
        setup.norb.push(norb);
        setup
            .generators
            .push(OrthogonalUniformGenerator::new((oa, ob), &mocache[parent]));
    }
    setup.expansion_rank = options.expansion_rank.unwrap_or(nelec).min(nelec);

    // Initial reference populations from the NOCI coefficients and the exact effective
    // Hamiltonian of uncorrelated references, `G = S^{-1}H`.
    let norm = c0.iter().map(|c| c.abs()).sum::<f64>();
    let mut populations = FixedMap::default();
    for (z, reference) in setup.references.iter().enumerate() {
        let c = if c0.len() == m && norm > 0.0 {
            c0[z] / norm
        } else {
            1.0 / m as f64
        };
        populations.insert(
            (z, reference.alpha, reference.beta),
            qmc.initial_population * c,
        );
    }
    let mut lagged = {
        let mut scratch = WickScratchSpin::new();
        let mut effective = Effective::new(m);
        for (z, bra) in setup.references.iter().enumerate() {
            for (x, ket) in setup.references.iter().enumerate() {
                let (h, s) = calculate_hs_excited(data, (bra, ket), &mut scratch);
                effective.se[(z, x)] = s;
                effective.he[(z, x)] = h;
            }
        }
        effective.g().expect("singular reference overlap")
    };

    print_header(irank);
    let initial = shift;
    let mut reached = false;
    let mut previous = populations.values().map(|n: &f64| n.abs()).sum::<f64>();
    let mut cumulative = Effective::new(m);
    let mut energy = f64::NAN;
    let mut average = f64::NAN;

    for report in 0..qmc.nreports {
        let mut block = Effective::new(m);

        for cycle in 0..qmc.ncycles {
            let iteration = report * qmc.ncycles + cycle;
            let current = blocks(&setup, &populations);
            let nref = current.iter().map(|b| b.nref).collect::<Vec<_>>();
            let gt = scaled(&lagged, &nref);

            // Divide every block's attempts into tasks, alternating linked and coupling work,
            // and keep the tasks of this rank.
            let attempts = current
                .iter()
                .map(|b| (b.nref.abs() + b.nex()).ceil().max(1.0))
                .collect::<Vec<_>>();
            let mut tasks = Vec::new();
            for (z, &a) in attempts.iter().enumerate() {
                let a = a as usize;
                for start in (0..a).step_by(CHUNK) {
                    let count = CHUNK.min(a - start);
                    tasks.push(Task::Linked(z, count));
                    if m > 1 {
                        tasks.push(Task::Coupling(z, count));
                    }
                }
            }
            let outputs = tasks
                .into_par_iter()
                .enumerate()
                .filter(|(k, _)| k % nranks == irank)
                .map(|(k, task)| {
                    let z = match task {
                        Task::Linked(z, _) | Task::Coupling(z, _) => z,
                    };
                    let rng = stream(base, (iteration as u64) << 32, k as u64);
                    run_task(&setup, &current, task, attempts[z], rng)
                })
                .collect::<Vec<_>>();

            // Combine the tasks in order, then add the coupling through the exact left action of
            // `e^{-T_y}` on `\sum_{x \ne y}\tilde G_{xy}q^x_y`.
            let mut spawns = Vec::new();
            let mut energies = vec![0.0; m];
            let mut projections = Projections::new(m);
            for output in &outputs {
                spawns.extend_from_slice(&output.spawns);
                for (e, v) in energies.iter_mut().zip(&output.energy) {
                    *e += v;
                }
                projections.add(&output.projections);
            }
            for (y, b) in current.iter().enumerate() {
                let mut q = FixedMap::default();
                for x in (0..m).filter(|&x| x != y) {
                    for (&nu, &v) in &projections.q[y][x] {
                        *q.entry(nu).or_insert(0.0) += gt[(x, y)] * v;
                    }
                }
                left_action(b, y, &q, setup.cluster_rank, setup.dt, &mut spawns);
            }

            // Annihilate this rank's spawns, round small ones stochastically, then gather.
            let mut updates = spawns
                .into_iter()
                .map(|(key, dn)| ExcitorUpdate::new(key, dn))
                .collect::<Vec<_>>();
            updates.sort_unstable_by_key(|u| u.excitor());
            coalesce(&mut updates);
            let mut rng = stream(local, SPAWNSALT, iteration as u64);
            compress_sparse(&mut updates, qmc.fri.spawn_cutoff, &mut rng);
            let mut gathered = if nranks > 1 {
                let mut scratch = crate::qmc::ExchangeScratch::new(nranks);
                gather_updates(world, &updates, &mut scratch).to_vec()
            } else {
                updates
            };
            gathered.sort_unstable_by_key(|u| u.excitor());
            coalesce(&mut gathered);

            // Sum the step's effective overlap and Hamiltonian over ranks, normalised by `N_x`.
            let mut sums = vec![0.0; m + 2 * m * m];
            sums[..m].copy_from_slice(&energies);
            for y in 0..m {
                for x in 0..m {
                    sums[m + y * m + x] = projections.se[y][x];
                    sums[m + m * m + y * m + x] = projections.he[y][x];
                }
            }
            if nranks > 1 {
                let local_sums = sums.clone();
                world.all_reduce_into(&local_sums[..], &mut sums[..], SystemOperation::sum());
            }
            for x in 0..m {
                block.se[(x, x)] += 1.0;
                block.he[(x, x)] += sums[x] / nref[x];
                for y in (0..m).filter(|&y| y != x) {
                    block.se[(y, x)] += sums[m + y * m + x] / nref[x];
                    block.he[(y, x)] += sums[m + m * m + y * m + x] / nref[x];
                }
            }

            // Fixed-point-preserving death `-\Delta\tau(g_x-S)n` on every population, then the
            // gathered spawns.
            let growth = (0..m).map(|x| gt.row(x).sum()).collect::<Vec<_>>();
            for (key, n) in populations.iter_mut() {
                *n *= 1.0 - setup.dt * (growth[key.0] - shift);
            }
            for update in &gathered {
                *populations.entry(update.excitor()).or_insert(0.0) += update.dn;
            }

            // Round excitor populations below the cutoff with a stream shared by every rank, so
            // the replicas stay identical. References are never removed.
            let mut small = populations
                .iter()
                .filter(|(key, n)| {
                    let reference = &setup.references[key.0];
                    (key.1, key.2) != (reference.alpha, reference.beta)
                        && n.abs() < qmc.fri.population_cutoff
                })
                .map(|(&key, &n)| (key, n))
                .collect::<Vec<_>>();
            small.sort_unstable_by_key(|&(key, _)| key);
            let mut rng = stream(base, ROUNDSALT, iteration as u64);
            let mut pending = None;
            let mut kept = Vec::new();
            for &(key, n) in &small {
                populations.remove(&key);
                if n != 0.0
                    && let Some(selected) = pivotal_push(
                        &mut pending,
                        cutoff_candidate(key, n, qmc.fri.population_cutoff),
                        &mut rng,
                    )
                {
                    kept.push(selected);
                }
            }
            kept.extend(resolve_pivotal(pending, &mut rng));
            populations.extend(kept);
        }

        // Report-block energy, lagged effective Hamiltonian and running average.
        if let Some(g) = block.g() {
            energy = jm_energies(&g)[0].re;
            lagged = g;
        }
        if reached {
            cumulative.add(&block);
            if let Some(g) = cumulative.g() {
                average = jm_energies(&g)[0].re;
            }
        }

        // Population statistics and the shift update.
        let total = populations.values().map(|n: &f64| n.abs()).sum::<f64>();
        let nref = setup
            .references
            .iter()
            .enumerate()
            .map(|(z, r)| {
                populations
                    .get(&(z, r.alpha, r.beta))
                    .map_or(0.0, |n| n.abs())
            })
            .sum::<f64>();
        update_shift(
            &mut shift,
            &mut reached,
            (total, previous),
            (
                qmc.target_population,
                qmc.shift_damping,
                setup.dt * qmc.ncycles as f64,
            ),
        );
        previous = total;
        print_row(
            irank,
            (report + 1) * qmc.ncycles,
            (energy, average, initial, if reached { shift } else { 0.0 }),
            (total, nref, populations.len() - m),
        );

        if stop_requested(irank, nranks, world) {
            break;
        }
    }

    if average.is_finite() { average } else { energy }
}
