// stochastic.rs
use std::collections::HashMap;
use rand::Rng;
use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::AoData;
use crate::SCFState;
use crate::input::{Input, Propagator};

use crate::noci::calculate_hs_pair;

// Storage for walker information.
pub struct Walkers {
    // Signed population vector length n determinants.
    pop: Vec<i64>,
    // List of indices with non-zero population.
    occ: Vec<usize>,
    // Position of index in `occ` or usize::MAX if not present, length n determinants.
    pos: Vec<usize>,
}

impl Walkers {
    /// Construct empty walker object for n determinants.
    /// # Arguments: 
    ///    `n`: usize, number of determinants. 
    pub fn new(n: usize) -> Self {
        Self {
            pop: vec![0; n],
            occ: Vec::new(),
            pos: vec![usize::MAX; n],
        }
    }
    
    /// Return the current population of determinant i.
    /// # Arguments: 
    ///     `self`: Walkers, object containing information about current walkers.
    ///     `i`: usize, determinant index of choice.
    pub fn get(&self, i: usize) -> i64 {
        self.pop[i]
    }
    
    /// Return a list of occupied determinant indices.
    /// # Arguments:
    ///     `self`: Walkers, object containing information about current walkers.
    pub fn occ(&self) -> &[usize] {
        &self.occ
    }

    /// Add dn (change in population) to determinant i, modifying pop, occ and pos as required
    /// # Arguments:
    ///     `self`: Walkers, object containing information about current walkers.
    ///     `i`: usize, determinant index of choice. 
    ///     `dn`: i64, change in population of determinant i.
    pub fn add(&mut self, i: usize, dn: i64) {
        // No changes to be made.
        if dn == 0 {return;}
        
        // Update population vector.
        let old = self.pop[i];
        let new = old + dn;
        self.pop[i] = new;
        
        // If old population of determinant i was 0 and we are introducing population we must
        // add determinant i to the occupied list and store its position in pos.
        if old == 0 && new != 0 {
            // Position of determinant i in occupied list is the current end.
            self.pos[i] = self.occ.len();
            self.occ.push(i);
        // If old population of determinant i was not 0 and we have removed population we must
        // remove i from the occupied list and remove its position from pos.
        } else if old != 0 && new == 0 {
            // Find where i is in occ. Should not return usize::MAX.
            let p = self.pos[i];
            // Pop the last occupied determinant index from occ.
            let last = self.occ.pop().unwrap();

            // If the popped element is not i, then i was somewhere in the middle of occ and we
            // must move the popped element (last) to position p where i used to be. The position of  
            // last is then updated in the position vector. If the popped element is i then we do
            // nothing as we have directly removed it by popping.
            if last != i {
                self.occ[p] = last;
                self.pos[last] = p;
            }
            // Position i is no longer occupied.
            self.pos[i] = usize::MAX;
        }
    }

    /// Compute the total walker population 1-norm.
    /// # Arguments:
    ///     `self`: Walkers, object containing information about current walkers.
    pub fn norm(&self) -> i64 {
        self.occ.iter().map(|&i| self.pop[i].abs()).sum()
    }
}

// Storage for Monte Carlo state. 
pub struct MCState {
    pub walkers: Walkers,
    // Changes to walker population of length n determinants.
    pub delta: Vec<i64>,
    // Indices for which delta[i] != 0, i.e., determinants with changed population.
    pub changed: Vec<usize>,
    pub cache: ElemCache,
    pub rng: SmallRng,
}

// Cache storage for already computed H, S matrix elements, avoids needless recomputation.
pub struct ElemCache {
    // (i, j) element indices and H_{ij}, S_{ij} element values.
    map: HashMap<(usize, usize), (f64, f64)>,
}

impl ElemCache {
    /// Constructor for elemCache object. Creates elemCache object with field map being an empty
    /// hash map. 
    /// # Arguments:
    ///     None.
    fn new() -> Self {
        Self {
            map: HashMap::new()
        }
    }
    
    /// Return a canonical ordering of the pair of indices i, j. Assigns smallest element to be in
    /// first position. 
    /// # Arguments:
    ///     `i`: usize, index of state i. 
    ///     `j`: usize, index of state k.
    fn key(i: usize, j: usize) -> (usize, usize) {
        if i <= j {(i, j)} else {(j, i)}
    }
    
    /// Find matrix elements H_{ij} and S_{ij} either from the cache or by computing them for the
    /// first time.
    /// # Arguments:
    ///     `self`: ElemCache, contains only HashMap of state index to H_{ij}, S_{ij} hash map.
    ///     `basis`: Vec<SCFState>, vector of all SCF states in the basis.
    ///     `i`: usize, index of state i. 
    ///     `j`: usize, index of state k.
    fn find_elem(&mut self, ao: &AoData, basis: &[SCFState], i: usize, j: usize) -> (f64, f64) {
        // Get the sorted pair of indices 
        let (a, b) = Self::key(i, j);
        
        // If we have already computed these matrix elements just return it.
        if let Some(&(h, s)) = self.map.get(&(a, b)) {
            return (h, s)
        }

        // Otherwise compute them for the first time
        let (h, s) = calculate_hs_pair(ao, basis, a, b);
        self.map.insert((a, b), (h, s));
        (h, s)
    }
}

impl Default for ElemCache {
    // Default cache is empty cache.
    fn default() -> Self {
        Self::new()
    }
}

fn add_delta(mc: &mut MCState, i: usize, dn: i64) {
    if dn == 0 {return;}
    // If current delta for this determinant is zero this function being called is its first
    // modification for this iteration and so we record this fact in changed.
    if mc.delta[i] == 0 {
        mc.changed.push(i);
    }
    // Add the population change to delta.
    mc.delta[i] += dn;
}

fn apply_delta(mc: &mut MCState) {
    // Consider only changed determinants.
    for &i in &mc.changed {
        // Total population change for this determinant for this iteration.
        let dn = mc.delta[i];
        // As we are applying the change we reset the delta.
        mc.delta[i] = 0;

        mc.walkers.add(i, dn);
    }
    // Reset list of changed determinants.
    mc.changed.clear()
}

/// For each entry in initial coefficient vector c0 calculate (c0_i / ||c||) * N_0 (initial population) 
/// and assign this value  as the initial walker population on this determinant. Sign of population is given 
/// by sign of c0_i.
/// # Arguments:
///     `c0`: Vec<f64>: Initial determinant coefficient vector.
///     `init_pop`: Number of walkers to start with.
///     `n`: Number of determinants.
pub fn initialse_walkers(c0: &[f64], init_pop: i64, n: usize, cache: &mut ElemCache, ao: &AoData, basis: &[SCFState], iref: usize) -> Walkers {
    let mut w = Walkers::new(n);

    // Ill-conditioning threshold.
    let threshold = 1e-6; 

    // Calculate 1-norm of initial coefficient vector.
    let norm1: f64 = c0.iter().map(|x| x.abs()).sum::<f64>();

    // Assign initial populations based on c0.
    for (i, &ci) in c0.iter().enumerate() {
        let ni = ((ci.abs() / norm1) * (init_pop as f64)).round() as i64;
        // Decide sign when not zero.
        if ni != 0 {
            let sgn = if ci >= 0.0 {1} else {-1};
            w.add(i, sgn * ni);
        }
    }
    
    // Calculate projected energy denominator \Sum_{\Gamma} N_{\Gamma} S_{\Gamma, iref}. If this
    // quantity is very small we have large amounts of overcompleteness and therefore
    // ill-conditioning in the NOCI-QMC overlap matrix.
    let mut den = 0.0; 
    for &gamma in w.occ() {
        let ngamma = w.get(gamma);
        // Calculate or retrieve matrix element H_{\Gamma, \text{Reference}}, S_{\Gamma,
        // \text{Reference}} from cache.
        let (_, sgr) = cache.find_elem(ao, basis, gamma, iref);
        den += (ngamma as f64) * sgr;
    }

    // If the NOCI-QMC overlap matrix is very ill-conditioned starting with all walkers on the RHF
    // reference rather than distributed according to c0 can prevent initial blow-ups in the
    // projected energy.
    if den.abs() < threshold {
        println!("NOCI-QMC overlap very ill-conditioned. Starting from reference index 0 (Denominator: {})", den);
        let mut w0 = Walkers::new(n);
        w0.add(0, init_pop);
        return w0;
    }
    w
}

/// Perform off-diagonal spawning (amplitude transfer) step by calculating:
///     $P_{\text{Spawn}}(\Lambda|\Gamma) = \frac{\Delta\tau|H_{\Lambda\Gamma} - 
///     E_sS_{\Lambda\Gamma}|}{P_{\text{gen}}(\Lambda|\Gamma)}$
/// where if P_{\text{spawn}} > random float in [0, 1] we spawn a child walker onto determinant \Lambda 
/// with the same sign of its parent if H_{\Lambda\Gamma} > 0 and -sign if H_{\Lambda\Gamma} < 0. Furthermore, 
/// if P_{\text{spawn}} > 1 then we spawn floor(P_{\text{spawn}}) extra children and a final child with 
/// probability P_{\text{spawn}} - floor(P_{\text{spawn}}).
/// # Arguments
///     `ao`: AoData, contains AO integrals and other system data. 
///     `basis`: Vec<SCFState>, list of the full NOCI-QMC basis.
///     `gamma`: usize, index of determinant \Gamma. 
///     `ngamma`: i64, walker population on determinant \Gamma.
///     `dt`: f64, time-step.
///     `es`: f64, non-overlap transformed shift energy.
///     `es_s`: f64, overlap-transformed shift energy.
///     `mc`: MCState, contains information about the current Monte Carlo state.
fn spawning(ao: &AoData, basis: &[SCFState], gamma: usize, ngamma: i64, es: f64,  es_s: f64, input: &Input, mc: &mut MCState) {

    let parent_sign: i64 = if ngamma > 0 {1} else {-1};
    let nwalkers = ngamma.unsigned_abs();

    let ndets = basis.len();
    // Probability of choosing a Lambda when using uniform excitation.
    let pgen = 1.0 / ((ndets - 1) as f64);
    
    // Iterate over all walkers on state Gamma.
    for _ in 0..nwalkers {
        // Simplistic uniform excitation generation for now. This should of course be developed.
        // Select Lambda from all (n - 1) avaliable determinants (i.e., those that are not Gamma).
        let mut lambda = mc.rng.gen_range(0..(basis.len() - 1));
        // If Lambda index is the same or more than Gamma index we must map it back into the full
        // index set.
        if lambda >= gamma {lambda += 1;}
     
        // Calculate or retrieve matrix elements H_{\Lambda\Gamma}, S_{\Lambda\Gamma} from cache.
        let (hlg, slg) = mc.cache.find_elem(ao, basis, lambda, gamma);

        // Spawning probability.
        let k = match input.prop.propagator {
            Propagator::Unshifted => hlg - es_s * slg,
            Propagator::Shifted => hlg - es_s * slg,
            Propagator::DoublyShifted => hlg - es_s * slg,
            Propagator::DifferenceDoublyShifted => hlg - (0.5 * (es_s + es)) * slg,
        };
        let pspawn = input.prop.dt * k.abs() / pgen;
        
        // Evaluate spawning outcomes
        let m = pspawn.floor() as i64;
        let frac = pspawn - (m as f64);
        let extra = if mc.rng.gen_range(0.0..1.0) < frac {1} else {0};
        let nchildren = m + extra;

        let sign: i64 = if k > 0.0 {1} else {-1};
        let child_sign: i64 = -sign * parent_sign;
        
        // Accumulate population change in delta.
        add_delta(mc, lambda, nchildren * child_sign);
    }
}

/// Perform diagonal death and cloning step by calculating:
///     $P_{\text{Death}}(\Lambda) = \Delta\tau(H_{\Lambda\Lambda}- 2E_s)$,
/// where which if P_{\text{death}} > 0 a walker dies with probability P_{\text{death}}, 
/// if P_{\text{death}} < 0 a walker is cloned with probability |P_{\text{death}}|. 
/// If P_{\text{death}} = 0, nothing will happen.  
/// # Arguments
///     `ao`: AoData, contains AO integrals and other system data. 
///     `basis`: Vec<SCFState>, list of the full NOCI-QMC basis.
///     `gamma`: usize, index of determinant \Gamma. 
///     `ngamma`: i64, walker population on determinant \Gamma.
///     `dt`: f64, time-step.
///     `es`: f64, non-overlap transformed shift energy.
///     `es_s`: f64, overlap-transformed shift energy.
///     `mc`: MCState, contains information about the current Monte Carlo state.
fn death_cloning(ao: &AoData, basis: &[SCFState], gamma: usize, ngamma: i64, es: f64,  es_s: f64, input: &Input, mc: &mut MCState) {

    // Calculate or retrieve matrix elements H_{\Gamma\Gamma}, S_{\Gamma\Gamma} from cache.
    let (hgg, sgg) = mc.cache.find_elem(ao, basis, gamma, gamma);

    // Death probability.
    let pdeath = match input.prop.propagator {
        Propagator::Unshifted => input.prop.dt * (hgg - sgg * es_s),
        Propagator::Shifted => input.prop.dt * (hgg - sgg * es_s - es_s),
        Propagator::DoublyShifted => input.prop.dt * (hgg - sgg * es_s - es),
        Propagator::DifferenceDoublyShifted => input.prop.dt * (hgg - sgg * 0.5 * (es_s + es) - (es - es_s)),
    };
    let p = pdeath.abs();

    // Sign of parent walkers on state Gamma determines which way round death and cloning occur.
    let parent_sign = if ngamma > 0 {1} else {-1};

    if pdeath == 0.0 {return;}
    // Iterate over all walkers on state Gamma.
    for _ in 0..ngamma.abs() {
        // Compare probability against randomly generated number in [0, 1].
        if mc.rng.gen_range(0.0..1.0) < p {
            // If walker dies we remove 1 walker of the parent's sign. If walker is cloned we add
            // one walker of the parent's sign.
            let dn = if pdeath > 0.0 {-parent_sign} else {parent_sign};
            // Accumulate population change in delta.
            add_delta(mc, gamma, dn);
        }
    }
}

/// Calculate projected energy as \frac{\sum_{H_{\Gamma, \text{Reference}}} N_{\Gamma}}{\sum_{S_{\Gamma, \text{Reference}}} N_{\Gamma}}.
/// # Arguments
///     `ao`: AoData, contains AO integrals and other system data. 
///     `basis`: Vec<SCFState>, list of the full NOCI-QMC basis.
///     `walkers`: Walkers, hash map between determinant index and walker population at that
///     determinant.
///     `cache`: ElemCache, hash map between two determinant indices i, j and matrix elements
///     H_{ij}, S_{ij}.
///     `iref`: usize, index of determinant we are projecting onto.
fn projected_energy(ao: &AoData, basis: &[SCFState], walkers: &Walkers, iref: usize, cache: &mut ElemCache) -> f64 {
    let mut num = 0.0; 
    let mut den = 0.0; 

    for &gamma in walkers.occ() {
        let ngamma = walkers.get(gamma);
        // Calculate or retrieve matrix elements H_{\Gamma, \text{Reference}}, S_{\Gamma,
        // \text{Reference}} from cache.
        let (hgr, sgr) = cache.find_elem(ao, basis, gamma, iref);
        num += (ngamma as f64) * hgr;
        den += (ngamma as f64) * sgr;
    }
    num / den
}

/// Compute the number of walkers $\tilde{N}_w(\tau) = ||S_{\Gamma\Omega}C^\Omega(\tau)||$ that are
/// not belonging to the null space as opposed to doing $N_w(\tau) = ||C^\Gamma(\tau)||_1$.
/// # Arguments
///     `ao`: AoData, contains AO integrals and other system data. 
///     `basis`: Vec<SCFState>, list of the full NOCI-QMC basis.
///     `walkers`: Walkers, hash map between determinant index and walker population at that
///     determinant.
///     `cache`: ElemCache, hash map between two determinant indices i, j and matrix elements
///     H_{ij}, S_{ij}.
fn nw_sc(ao: &AoData, basis: &[SCFState], walkers: &Walkers, cache: &mut ElemCache) -> f64 {
    let ndets = basis.len();
    let mut acc = 0.0;

    for gamma in 0..ndets {
        //p_{\Gamma} = \sum_\Omega S_{\Gamma, \Omega} N_{\Omega}.
        let mut pgamma = 0.0;
        for &omega in walkers.occ() {
            let nomega = walkers.get(omega);
            let (_, sgo) = cache.find_elem(ao, basis, gamma, omega);
            pgamma += (nomega as f64) * sgo;
        }
        // ||S_{\Gamma\Omega}C^\Omega(\tau)|| = \sum_{\Gamma} |p_{\Gamma}|.
        acc += pgamma.abs();
    }
    acc
}

/// Propagate according to the stochastic update equations for max_steps iterations.
/// # Arguments: 
///     `c0`: Vec<f64>, initial determinant coefficient vector to be translated into walker
///     populations.
///     `ao`: AoData, contains AO integrals and other system data. 
///     `basis`: Vec<SCFState>, list of the full NOCI-QMC basis.
///     `es`: f64, initial shift energy.
///     `input`: Input, user specified input options.
pub fn step(c0: &[f64], ao: &AoData, basis: &[SCFState], es: &mut f64, input: &Input) -> f64 {
    
    // Unwrap QMC propagation specific options
    let qmc = input.qmc.as_ref().unwrap();
    let ndets = basis.len();

    let iref = 0;
    let mut cache = ElemCache::new();

    // Initialise overlap-transformed shift E_s^S. This is distinct from es (E_s) in that E_s is
    // the shift updated using N_w, whilst E_s^S is updated using \tilde{N}_w.
    let mut es_s = *es;

    // Initialise walker populations based on total initial population and c0.
    println!("Initialising walkers.....");
    let w = initialse_walkers(c0, qmc.initial_population, ndets, &mut cache, ao, basis, iref);

    // Flags activated once total walker population exceeds target population 
    let mut reached_sc = false;
    let mut reached_c = false;

    // Initialise Monte Carlo state. All population updates for a given iteration are accumulated within delta.
    let rng = SmallRng::from_entropy();
    let mut mc = MCState {walkers: w, delta: vec![0; ndets], changed: Vec::new(), cache, rng};

    // Project onto determinant with index zero (this is usually the first RHF reference).
    let eproj = projected_energy(ao, basis, &mc.walkers, iref, &mut mc.cache);

    let mut nwscprev: f64 = nw_sc(ao, basis, &mc.walkers, &mut mc.cache);
    let mut nwcprev = qmc.initial_population;

    // Print table header.
    println!("{}", "=".repeat(100));
    println!("{:<6} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16}", "iter", "E", "Ecorr", "Shift (Es)", "Shift (EsS)", "Nw (||C||)", "Nw (||SC||)");
    // Print initial.
    println!("{:<6} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12}", 
              0, eproj, eproj - basis[0].e, *es, es_s, nwcprev, nwscprev);

    for it in 0..input.prop.max_steps {
        // Copy occupied list.
        let occ = mc.walkers.occ().to_vec();
        for gamma in occ {
            let ngamma = mc.walkers.get(gamma);
            // If the walker population on state Gamma is zero we ignore it.
            if ngamma == 0 {continue;}
            // Diagonal death and cloning step.
            death_cloning(ao, basis, gamma, ngamma, *es, es_s, input, &mut mc);
            // Off-diagonal amplitude transfer.
            spawning(ao, basis, gamma, ngamma, *es, es_s, input, &mut mc);
        }
        // Update walker populations according to accumulated delta. Annhilation happens
        // automatically here.
        apply_delta(&mut mc);
        
        // Non overlap-transformed walker population.
        let nwc = mc.walkers.norm();
        // Overlap-transformed walker population.
        let nwsc = nw_sc(ao, basis, &mc.walkers, &mut mc.cache);
        
        // Activate overlap-transformed and non-overlap transformed shifts.
        if !reached_c && (nwc > qmc.target_population) {
            reached_c = true;
            nwcprev = nwc;
        }
        if !reached_sc && (nwsc > qmc.target_population as f64) {
            reached_sc = true;
            nwscprev = nwsc;
        }
        // Update shift once total populations have exceeded target population.
        if reached_c && (it + 1) % qmc.shift_update_freq == 0 {
            *es -= (qmc.shift_damping / (input.prop.dt * (qmc.shift_update_freq as f64))) * (nwc as f64 / nwcprev as f64).ln();
            nwcprev = nwc;
        }
        if reached_sc && (it + 1) % qmc.shift_update_freq == 0 {
            es_s -= (qmc.shift_damping / (input.prop.dt * (qmc.shift_update_freq as f64))) * (nwsc / nwscprev).ln();
            nwscprev = nwsc;
        }

        // Update energy. 
        let iref = 0;
        let eproj = projected_energy(ao, basis, &mc.walkers, iref, &mut mc.cache);

        // Print table rows.
        println!("{:<6} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12}", it + 1, eproj, eproj - basis[0].e, *es, es_s, nwc, nwsc);
    }
    eproj
}


