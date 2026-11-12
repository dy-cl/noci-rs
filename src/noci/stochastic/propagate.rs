// noci/stochastic/propagate.rs
//! The NOCIQMC side of the shared report loop. `NOCIRun` holds what every NOCIQMC propagator shares
//! during one run: the data and topology, the populations and their controller history, the shift,
//! and the sampling and communication scratch. `NOCIPropagator` is implemented by the SApply,
//! BApply and unshifted propagators for what differs: their set-up, the spawning of one cycle
//! and the update applied at the end of a report. `Propagation` joins the two into one
//! `qmc::Propagator`.

// Standard library imports.
use std::sync::Mutex;

// External crate imports.
use mpi::topology::Communicator;

// Crate-root imports.
use crate::elements::NOCIData;
use crate::elements::nonorthogonalwicks::WickScratchSpin;
use crate::input::Propagator as PropagatorKind;
use crate::noci::OverlapFactors;
use crate::qmc::excitgen::ExcitationHist;
use crate::qmc::{Propagator, run, stream, update_shift};

// Parent/sibling imports.
use super::bapply::BApply;
use super::common::{
    accumulate_generated_updates, construct_qmc_run, population_stats_projected_energy,
    propagate_iteration,
};
use super::excit::update_overlap_weight;
use super::fri::sample_populations;
use super::init::initialise_qmc_state;
use super::overlapweighted::OverlapWeightedGenerator;
use super::report::{print_header, print_initial_row, print_row, write_restart};
use super::sapply::SApply;
use super::shift::tangent_scale;
use super::state::{
    NOCIMPIScratch, NOCIPropagationResult, NOCIThreadPropagation, OverlapDerivativeSums,
    PopulationStats, PropagationState, QMCRunInfo, ScratchSize,
};
use super::unshifted::Unshifted;

/// One NOCIQMC run: the shared run state and the propagator advancing it.
struct Propagation<'a, C, P> {
    /// Shared run state.
    run: NOCIRun<'a, C>,
    /// Propagator advancing the populations.
    propagator: P,
}

/// The state every NOCIQMC propagator shares during one run.
pub(in crate::noci::stochastic) struct NOCIRun<'a, C> {
    /// Immutable stochastic propagation data.
    pub(in crate::noci::stochastic) data: &'a NOCIData<'a, f64>,
    /// MPI communicator.
    pub(in crate::noci::stochastic) world: &'a C,
    /// Rank-local topology and determinant ownership.
    pub(in crate::noci::stochastic) run: QMCRunInfo,
    /// Whether each determinant is a reference.
    isref: Vec<bool>,
    /// Populations, projected energy and controller history.
    pub(in crate::noci::stochastic) state: PropagationState,
    /// Population statistics of the last report.
    stats: PopulationStats,
    /// Population-control shift `E_s`.
    pub(in crate::noci::stochastic) shift: f64,
    /// Overlap-weight derivatives accumulated over the cycles of a report.
    derivatives: OverlapDerivativeSums,
    /// Retained-space MPI exchange scratch.
    pub(in crate::noci::stochastic) mpi: NOCIMPIScratch,
    /// Reusable chunk storage of the population sampling.
    chunks: Vec<Vec<(usize, f64)>>,
}

/// A NOCIQMC propagator: its set-up, the spawning of one cycle and the update at the end of a
/// report. The run state it shares with the others is passed as `NOCIRun`.
pub(in crate::noci::stochastic) trait NOCIPropagator<C: Communicator>:
    Sized
{
    /// Build the propagator's generator, factor tables and workers.
    /// # Arguments:
    /// - `data`: Immutable stochastic propagation data.
    /// - `run`: Rank-local topology of the run.
    /// - `sizes`: Wick scratch dimensions of the worker threads.
    /// - `world`: MPI communicator.
    /// # Returns
    /// - `Self`: Propagator ready for its first cycle.
    fn new(
        data: &NOCIData<'_, f64>,
        run: &QMCRunInfo,
        sizes: ScratchSize,
        world: &C,
    ) -> Self;

    /// Spawn one cycle from the sampled populations and accumulate its change.
    /// # Arguments:
    /// - `self`: Propagator.
    /// - `run`: Shared run state, whose sampled populations are the cycle's.
    /// - `iteration`: Global iteration of the cycle.
    /// # Returns
    /// - `()`: Accumulates the cycle's spawns.
    fn cycle(
        &mut self,
        run: &mut NOCIRun<'_, C>,
        iteration: usize,
    );

    /// Exchange, compress and apply the report's accumulated change to the populations.
    /// # Arguments:
    /// - `self`: Propagator.
    /// - `run`: Shared run state.
    /// - `report`: Report ending with this update.
    /// # Returns
    /// - `Option<&[f64]>`: Propagated shift tangent `\partial N/\partial E_s` for a tangent
    ///   controller, or `None` for the logarithmic one.
    fn apply(
        &mut self,
        run: &mut NOCIRun<'_, C>,
        report: usize,
    ) -> Option<&[f64]>;

    /// Print the propagator's final diagnostics, by default none.
    /// # Arguments:
    /// - `self`: Propagator.
    /// - `_`: Shared run state.
    /// # Returns
    /// - `()`: Prints on the root rank.
    fn finish(
        &mut self,
        _: &NOCIRun<'_, C>,
    ) {
    }
}

impl<'a, C: Communicator> NOCIRun<'a, C> {
    /// Restore or initialise the populations of a run and print its header and initial row.
    /// # Arguments:
    /// - `data`: Immutable stochastic propagation data.
    /// - `world`: MPI communicator.
    /// - `start`: Initial coefficient vector and shift.
    /// - `topology`: Reference mask and rank-local topology of the run.
    /// # Returns
    /// - `Self`: Run state at its first report.
    fn new(
        data: &'a NOCIData<'a, f64>,
        world: &'a C,
        start: (&[f64], f64),
        topology: (Vec<bool>, QMCRunInfo),
    ) -> Self {
        let (c0, mut shift) = start;
        let (isref, run) = topology;
        let mut mpi = NOCIMPIScratch::new(run.nranks);
        let state = initialise_qmc_state(
            c0,
            &mut shift,
            data,
            &run,
            &isref,
            &mut WickScratchSpin::new(),
            (world, &mut mpi),
        );

        let qmc = data.input.qmc.as_ref().unwrap();
        let propagator = data.input.prop_ref().propagator;
        print_header(run.irank, propagator);
        print_initial_row(
            run.irank,
            state.start_report * qmc.ncycles,
            &state,
            data.space.parents[0].e,
            shift,
            propagator,
        );

        Self {
            data,
            world,
            run,
            isref,
            stats: state.prev_pop,
            state,
            shift,
            derivatives: OverlapDerivativeSums::default(),
            mpi,
            chunks: Vec::new(),
        }
    }

    /// Sample the populations of one cycle by FRI on the frozen persistent populations.
    /// # Arguments:
    /// - `self`: Run state.
    /// - `iteration`: Global iteration of the cycle.
    /// # Returns
    /// - `()`: Replaces the sampled populations.
    fn sample(
        &mut self,
        iteration: usize,
    ) {
        let qmc = self.data.input.qmc.as_ref().unwrap();
        let mut rng = stream(self.run.rank_seed, 0xD1B54A32D192ED03, iteration as u64);
        sample_populations(
            &self.state.mc.populations,
            &mut self.state.mc.sampled,
            qmc.fri.population_cutoff,
            &self.run,
            &mut rng,
            &mut self.chunks,
        );
    }

    /// Spawn one cycle within the retained space and accumulate its changes and overlap-weight
    /// derivatives, for the propagators whose spawns land on retained determinants.
    /// # Arguments:
    /// - `self`: Run state.
    /// - `iteration`: Global iteration of the cycle.
    /// - `overlap`: Overlap factors and overlap-weighted generator, when the generator uses them.
    /// - `workers`: Per-thread propagation workers.
    /// - `result`: Reusable gathered spawns of the cycle.
    /// # Returns
    /// - `()`: Accumulates the cycle's spawns into the dense report change.
    pub(in crate::noci::stochastic) fn spawn(
        &mut self,
        iteration: usize,
        overlap: (Option<&OverlapFactors>, Option<&OverlapWeightedGenerator>),
        workers: &mut [Mutex<NOCIThreadPropagation>],
        result: &mut NOCIPropagationResult,
    ) {
        let qmc = self.data.input.qmc.as_ref().unwrap();
        let (factors, generator) = overlap;

        // Batch the cycle's matrix elements on the workers before draining their results.
        propagate_iteration(
            (iteration, &self.state.mc.sampled),
            self.data,
            &self.run,
            self.shift,
            (
                factors,
                generator,
                self.state.overlap_weight,
                qmc.optimise_overlap_weight,
            ),
            workers,
            result,
        );

        // Keep the report's overlap-weight estimator and population change in separate sums.
        self.derivatives.add(&result.overlap_derivatives);
        accumulate_generated_updates(&mut self.state.mc, result, self.data.input, &mut self.mpi);
    }

    /// Update the observables after a step, then the shift and the overlap weight. A propagated
    /// tangent `\partial N/\partial E_s` selects the tangent Newton controller; without one the
    /// shift follows the logarithmic population growth.
    /// # Arguments:
    /// - `self`: Run state.
    /// - `tangent`: Propagated shift tangent of the range propagators.
    /// # Returns
    /// - `()`: Updates the projected energy, statistics, shift and overlap weight.
    fn observe(
        &mut self,
        tangent: Option<&[f64]>,
    ) {
        let (stats, pe) =
            population_stats_projected_energy(&self.state.mc, &self.isref, &self.run, self.world);

        self.state.pe = pe;
        self.state.eprojcur = self.state.pe.num / self.state.pe.den;

        // The range propagators take `\partial N/\partial E_s` from their propagated tangent. The
        // unshifted shift enters only through death, so
        // `\partial N/\partial E_s = n_\text{cyc}\Delta\tau N`, and it ignores the restoring term.
        let qmc = self.data.input.qmc.as_ref().unwrap();
        let norms = (stats.nw, self.state.prev_pop.nw);
        let (scale, restoring) = match tangent {
            Some(tangent) => (
                tangent_scale(
                    &self.state.mc.populations,
                    tangent,
                    norms,
                    qmc.shift_damping,
                    &self.run,
                    self.world,
                ),
                qmc.population_restoring,
            ),
            None => {
                let dteff = self.data.input.prop_ref().dt * qmc.ncycles as f64;
                (Some(qmc.shift_damping / dteff), 0.0)
            }
        };
        update_shift(
            &mut self.shift,
            &mut self.state.reached,
            norms,
            (qmc.target_population, restoring),
            scale,
        );
        self.state.prev_pop = stats;

        // Reweight the proposal only after using the realised report for shift control.
        update_overlap_weight(
            &mut self.state,
            &mut self.derivatives,
            self.data.input,
            &self.run,
            self.world,
        );
        self.stats = stats;
    }
}

impl<C: Communicator, P: NOCIPropagator<C>> Propagator for Propagation<'_, C, P> {
    /// Sample the frozen populations and spawn one cycle from them.
    /// # Arguments:
    /// - `self`: NOCIQMC run.
    /// - `iteration`: Global iteration of the cycle.
    /// # Returns
    /// - `()`: Accumulates the cycle's spawns.
    fn cycle(
        &mut self,
        iteration: usize,
    ) {
        self.run.sample(iteration);
        self.propagator.cycle(&mut self.run, iteration);
    }

    /// Apply the report's change, then update the observables, the shift and the overlap weight.
    /// # Arguments:
    /// - `self`: NOCIQMC run.
    /// - `report`: Report ending with this step.
    /// # Returns
    /// - `()`: Advances the populations by one step.
    fn step(
        &mut self,
        report: usize,
    ) {
        let tangent = self.propagator.apply(&mut self.run, report);
        self.run.observe(tangent);
    }

    /// Write the persistent populations and controller history.
    /// # Arguments:
    /// - `self`: NOCIQMC run.
    /// - `report`: Last completed report.
    /// # Returns
    /// - `()`: Writes the restart file.
    fn write_restart(
        &mut self,
        report: usize,
    ) {
        let run = &self.run;
        write_restart(
            report,
            &run.state,
            run.shift,
            &run.run,
            run.world,
            run.data.input.write.write_restart.as_ref(),
        );
    }

    /// Print the populations, energies and shift of the last report.
    /// # Arguments:
    /// - `self`: NOCIQMC run.
    /// - `end`: Iteration count at the end of the report.
    /// # Returns
    /// - `()`: Prints one row on the root rank.
    fn print_row(
        &self,
        end: usize,
    ) {
        let run = &self.run;
        print_row(
            run.run.irank,
            end,
            &run.state,
            &run.stats,
            run.data.space.parents[0].e,
            run.shift,
            run.data.input.prop_ref().propagator,
        );
    }

    /// Print the propagator's final diagnostics.
    /// # Arguments:
    /// - `self`: NOCIQMC run.
    /// # Returns
    /// - `()`: Prints on the root rank.
    fn finish(&mut self) {
        self.propagator.finish(&self.run);
    }
}

/// Build one NOCIQMC run of a propagator and drive it through the shared report loop.
/// # Arguments:
/// - `data`: Immutable stochastic propagation data.
/// - `c0`: Initial determinant coefficient vector.
/// - `es`: Population-control shift energy, updated to its final value.
/// - `ref_indices`: Indices of the reference determinants in the stochastic basis.
/// - `world`: MPI communicator.
/// # Returns:
/// - `(f64, Option<ExcitationHist>)`: Final projected energy estimate and optional
///   spawning-magnitude histogram.
fn propagate<C: Communicator, P: NOCIPropagator<C>>(
    data: &NOCIData<'_, f64>,
    c0: &[f64],
    es: &mut f64,
    ref_indices: &[usize],
    world: &C,
) -> (f64, Option<ExcitationHist>) {
    // The propagator is built before the populations so its set-up prints precede the header.
    let (isref, sizes, topology) = construct_qmc_run(data, c0, ref_indices, world);
    let propagator = P::new(data, &topology, sizes, world);
    let mut propagation = Propagation {
        run: NOCIRun::new(data, world, (c0, *es), (isref, topology)),
        propagator,
    };

    let qmc = data.input.qmc.as_ref().unwrap();
    let write = &data.input.write;
    let path = write.write_restart.as_deref().unwrap_or("RESTART.H5");
    let reports = propagation.run.state.start_report..qmc.nreports;
    run(
        &mut propagation,
        (reports, qmc.ncycles),
        (write.write_restart_interval, path),
        world,
    );

    let finished = propagation.run;
    *es = finished.shift;
    (finished.state.eprojcur, finished.state.mc.excitation_hist)
}

/// Propagate according to the stochastic update equations selected by `prop.propagator`.
/// # Arguments:
/// - `data`: Immutable stochastic propagation data.
/// - `c0`: Initial determinant coefficient vector.
/// - `es`: Population-control shift energy.
/// - `ref_indices`: Indices of the reference determinants in the stochastic basis.
/// - `world`: MPI communicator.
/// # Returns:
/// - `(f64, Option<ExcitationHist>)`: Final projected energy estimate and optional
///   spawning-magnitude histogram.
pub fn qmc_step<C: Communicator>(
    data: &NOCIData<'_, f64>,
    c0: &[f64],
    es: &mut f64,
    ref_indices: &[usize],
    world: &C,
) -> (f64, Option<ExcitationHist>) {
    match data.input.prop_ref().propagator {
        PropagatorKind::Unshifted => propagate::<C, Unshifted>(data, c0, es, ref_indices, world),
        PropagatorKind::SApply => propagate::<C, SApply>(data, c0, es, ref_indices, world),
        PropagatorKind::BApply => propagate::<C, BApply>(data, c0, es, ref_indices, world),
    }
}
