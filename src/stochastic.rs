// stochastic.rs
use std::collections::HashMap;
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::AoData;
use crate::SCFState;
use crate::input::Input;

use crate::noci::calculate_hs_pair;

// Storage for walkers, determinant index and signed population.
pub type Walkers = HashMap<usize, i64>;

// Storage for Monte Carlo state. 
pub struct MCState {
    pub walkers: Walkers,
    pub delta: HashMap<usize, i64>,
    pub cache: ElemCache,
    pub rng: StdRng,
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


/// For each entry in initial coefficient vector c0 calculate (c0_i / ||c||) * N_0 (initial population) 
/// and assign this value  as the initial walker population on this determinant. Sign of population is given 
/// by sign of c0_i.
/// # Arguments:
///     `c0`: Vec<f64>: Initial determinant coefficient vector.
///     `init_pop`: Number of walkers to start with.
pub fn initialse_walkers(c0: &[f64], init_pop: i64) -> Walkers {
    let mut w = Walkers::new();
    // Calculate 1-norm of initial coefficient vector.
    let norm1: f64 = c0.iter().map(|x| x.abs()).sum::<f64>();

    // Assign initial populations based on c0.
    for (i, &ci) in c0.iter().enumerate() {
        let ni = ((ci.abs() / norm1) * (init_pop as f64)).round() as i64;
        // Decide sign when not zero.
        if ni != 0 {
            let sgn = if ci >= 0.0 {1} else {-1};
            w.insert(i, sgn * ni);
        }
    }
    // Alternatively place all weight on first reference (index 0)
    //w.insert(0, init_pop);

    w
}

/// Apply the accumulated delta (change in walker populations) from one iteration to the full
/// walker population. 
/// # Arguments:
///     delta: HashMap, contains changes in walker population for each determinant.
///     Walkers: Hashmap, contains full walker population for each determinant.
fn apply_delta(walkers: &mut Walkers, delta: &mut HashMap<usize, i64>) {
    // Loop over indices i of determinants and their population change dn.
    for (i, dn) in delta.drain() {
        // If population change is 0 we leave the determinant alone.
        if dn == 0 {continue;}
        // Walker population on determinant i. If i is not in the HashMap of walkers we insert it
        // with zero population.
        let ni = walkers.entry(i).or_insert(0);
        // Apply change in population and perform annhilation.
        *ni += dn;
        if *ni == 0 {
            walkers.remove(&i);
        }
    }
}

/// Perform off-diagonal spawning (amplitude transfer) step by calculating:
///     $P_{\text{Spawn}}(\Lambda|\Gamma) = \frac{\Delta\tau|H_{\Lambda\Gamma} - 
///     E_sS_{\Lambda\Gamma}|}{P_{\text{gen}}(\Lambda|\Gamma)}$
/// where if P_{\text{spawn}} > random float in [0, 1] we spawn a child walker onto determinant \Lambda 
/// with the same sign of its parent if H_{\Lambda\Gamma} > 0 and -sign if H_{\Lambda\Gamma} < 0. Furthermore, 
/// if P_{\text{spawn}} > 1 then we spawn floor(P_{\text{spawn}}) extra children and a final child with 
/// probability P_{\text{spawn}} - floor(P_{\text{spawn}}).
/// # Arguments:
///     `ao`: AoData, contains AO integrals and other system data. 
///     `basis`: Vec<SCFState>, list of the full NOCI-QMC basis.
///     `gamma`: usize, index of determinant \Gamma. 
///     `ngamma`: i64, walker population on determinant \Gamma.
///     `dt`: f64, time-step.
///     `es`: f64, shift energy.
///     `mc`: MCState, contains information about the current Monte Carlo state.
fn spawning(ao: &AoData, basis: &[SCFState], gamma: usize, ngamma: i64, dt: f64, es: f64, mc: &mut MCState) {

    let parent_sign: i64 = if ngamma > 0 { 1 } else { -1 };
    
    // Iterate over all walkers on state Gamma.
    for _ in 0..ngamma.abs() {
        // Simplistic uniform excitation generation for now. This should of course be developed.
        // Select Lambda from all (n - 1) avaliable determinants (i.e., those that are not Gamma).
        let mut lambda = mc.rng.gen_range(0..(basis.len() - 1));
        // If Lambda index is the same or more than Gamma index we must map it back into the full
        // index set.
        if lambda >= gamma {lambda += 1;}
        // Probability of choosing this Lambda when using uniform excitation.
        let pgen = 1.0 / ((basis.len() - 1) as f64);

        // Calculate or retrieve matrix elements H_{\Lambda\Gamma}, S_{\Lambda\Gamma} from cache.
        let (hlg, slg) = mc.cache.find_elem(ao, basis, lambda, gamma);

        // Spawning probability. 
        let k = hlg - es * slg;
        let pspawn = dt * k.abs() / pgen;
        
        // Evaluate spawning outcomes
        let m = pspawn.floor() as i64;
        let frac = pspawn - (m as f64);
        let extra = if mc.rng.gen_range(0.0..1.0) < frac {1} else {0};
        let nchildren = m + extra;

        let ksign: i64 = if k > 0.0 {1} else {-1};
        let child_sign: i64 = -ksign * parent_sign;

        *mc.delta.entry(lambda).or_insert(0) += nchildren * child_sign;
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
///     `es`: f64, shift energy.
///     `mc`: MCState, contains information about the current Monte Carlo state.
fn death_cloning(ao: &AoData, basis: &[SCFState], gamma: usize, ngamma: i64, dt: f64, es: f64, mc: &mut MCState) {

    // Calculate or retrieve matrix elements H_{\Gamma\Gamma}, S_{\Gamma\Gamma} from cache.
    let (hgg, sgg) = mc.cache.find_elem(ao, basis, gamma, gamma);

    // Death probability.
    let pdeath = dt * (hgg - sgg * es - es);
    let p = pdeath.abs();

    // Sign of parent walkers on state Gamma determines which way round death and cloning occur.
    let parent_sign = if ngamma > 0 {1} else {-1};

    // If 
    if pdeath == 0.0 {return;}
    // Iterate over all walkers on state Gamma.
    for _ in 0..ngamma.abs() {
        // Compare probability against randomly generated number in [0, 1].
        if mc.rng.gen_range(0.0..1.0) < p {
            let dn = if pdeath > 0.0 {
                // Death of walker. Remove one walker of the parent's sign.
                -parent_sign
            } else {
                // Cloning of walker. Add one walker of the parent's sign.
                parent_sign
            };
            // Store this population change at index Gamma in delta HashMap.
            *mc.delta.entry(gamma).or_insert(0) += dn;
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

    for (&gamma, &ngamma) in walkers.iter() {
        if ngamma == 0 {continue;}
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
/// Warning: this is currently very very very slow and should be improved with haste.
/// # Arguments
///     `ao`: AoData, contains AO integrals and other system data. 
///     `basis`: Vec<SCFState>, list of the full NOCI-QMC basis.
///     `walkers`: Walkers, hash map between determinant index and walker population at that
///     determinant.
///     `cache`: ElemCache, hash map between two determinant indices i, j and matrix elements
///     H_{ij}, S_{ij}.
fn nw_sc(ao: &AoData, basis: &[SCFState], walkers: &Walkers, cache: &mut ElemCache) -> f64 {
    //p_{\Pi} = \sum_\Omega S_{\Pi, \Omega} N_{\Omega}
    let mut p: HashMap<usize, f64> = HashMap::new();

    let occ: Vec<(usize, i64)> = walkers.iter().map(|(&i, &ni)| (i, ni)).collect();
    // N_w = \sum_{\Pi} |p_{\Pi}|
    for (omega, nomega) in &occ {
        for (pi, _) in &occ {
            let (_, spo) = cache.find_elem(ao, basis, *pi, *omega);
            *p.entry(*pi).or_insert(0.0) += (*nomega as f64) * spo;
        }
    }
    p.values().map(|x| x.abs()).sum()
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

    // Initialise walker populations based on total initial population and c0.
    println!("Initialising walkers.....");
    let w = initialse_walkers(c0, qmc.initial_population);

    // Shift update parameters, these should become an input option.
    let zeta = 0.05;
    let a = 50usize;
    // Flag activated once total walker population exceeds target population 
    let mut reached = false;

    // Initialise Monte Carlo state. All population updates for a given iteration are accumulated within delta.
    let rng = StdRng::from_entropy();
    let mut mc = MCState {walkers: w, delta: HashMap::new(), cache: ElemCache::new(), rng};

    // Project onto determinant with index zero (this is usually the first RHF reference).
    let iref = 0;
    let eproj = projected_energy(ao, basis, &mc.walkers, iref, &mut mc.cache);

    let mut nsprev: f64 = nw_sc(ao, basis, &mc.walkers, &mut mc.cache);

    // Print table header.
    println!("{}", "=".repeat(100));
    println!("{:<6} {:>16} {:>16} {:>16} {:>16} {:>16}", "iter", "E", "Ecorr", "Shift", "Nw (||C||)", "Nw (||SC||)");
    // Print initial.
    println!("{:<6} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12}", 
              0, eproj, eproj - basis[0].e, es, qmc.initial_population, nsprev);

    for it in 0..input.prop.max_steps {
        // Copy walkers into occ.
        let occ: Vec<(usize, i64)> = mc.walkers.iter().map(|(&i, &ni)| (i, ni)).collect(); 

        for (gamma, ngamma) in occ {

            // If the walker population on state Gamma is zero we ignore it.
            if ngamma == 0 {continue;}
            
            // Diagonal death and cloning step.
            death_cloning(ao, basis, gamma, ngamma, input.prop.dt, *es, &mut mc);
            
            // Off-diagonal amplitude transfer.
            spawning(ao, basis, gamma, ngamma, input.prop.dt, *es, &mut mc);
        }
        // Update walker populations according to accumulated delta. Annhilation happens
        // automatically here.
        apply_delta(&mut mc.walkers, &mut mc.delta);
        let n: i64 = mc.walkers.values().map(|n| n.abs()).sum();
        let ns: f64 = nw_sc(ao, basis, &mc.walkers, &mut mc.cache); 
        if ns > (qmc.target_population as f64) && !reached {
            reached = true;
            nsprev = ns;
        }

        // Update shift once total population has exceeded target population.
        if reached && (it + 1) % a == 0 {
            *es -= (zeta / (input.prop.dt * (a as f64))) * (ns / nsprev).ln();
            nsprev = ns;
        }

        // Update energy. 
        let iref = 0;
        let eproj = projected_energy(ao, basis, &mc.walkers, iref, &mut mc.cache);

        // Print table rows.
        println!("{:<6} {:>16.12} {:>16.12} {:>16.12} {:>16.12} {:>16.12}", it + 1, eproj, eproj - basis[0].e, es, n, ns);
    }
    eproj
}


