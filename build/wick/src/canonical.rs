// canonical.rs

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{Mutex, OnceLock};
#[cfg(feature = "timings")]
use std::time::Instant;

use num_rational::Ratio;
use num_traits::Zero;
use smallvec::SmallVec;

use crate::gram::{self, Filter, State};
use crate::ir::{Delta, Expr, Idx, Rational, Tensor, TensorKind, Term};

/// High-rank cumulant orbit key.
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
struct OrbitKey {
    /// Cumulant rank.
    n: usize,
    /// Delta factors.
    d: Vec<Delta>,
    /// Non-orbit tensor factors.
    t: Vec<Tensor>,
    /// Cumulant kind.
    k: TensorKind,
    /// Upper indices.
    up: Vec<Idx>,
    /// Sorted lower-index base.
    lo: Vec<Idx>,
}

/// Interned factor id.
type Id = u32;

/// Canonical term key.
type TermKey = (SmallVec<[Id; 4]>, SmallVec<[Id; 8]>);

/// Maximum number of uncombined canonical terms.
const PENDING_LIMIT: usize = 262_144;

/// Maximum cached sparsifier results.
const BEST_CACHE: usize = 4096;

/// Canonical term key and coefficient.
type RunTerm = (TermKey, Ratio<i64>);

/// Sparsifier cache key.
type BestKey = (usize, BTreeMap<Vec<usize>, Ratio<i64>>);

/// Sparsifier cache value.
type BestVal = (BestKey, BTreeMap<Vec<usize>, Ratio<i64>>);

/// Interned canonical factors.
#[derive(Clone, Debug, Default)]
struct Store {
    /// Delta-to-id lookup.
    ds: BTreeMap<Delta, Id>,
    /// Tensor-to-id lookup.
    ts: BTreeMap<Tensor, Id>,
    /// Delta table.
    d: Vec<Delta>,
    /// Tensor table.
    t: Vec<Tensor>,
}

impl Store {
    /// Intern one delta.
    /// # Arguments:
    /// - `x`: Delta factor.
    /// # Returns:
    /// - `Id`: One-based delta id.
    fn delta(
        &mut self,
        x: Delta,
    ) -> Id {
        if let Some(&id) = self.ds.get(&x) {
            return id;
        }

        let id = self.d.len() as Id + 1;

        self.ds.insert(x, id);
        self.d.push(x);

        id
    }

    /// Intern one tensor.
    /// # Arguments:
    /// - `x`: Tensor factor.
    /// # Returns:
    /// - `Id`: One-based tensor id.
    fn tensor(
        &mut self,
        x: Tensor,
    ) -> Id {
        if let Some(&id) = self.ts.get(&x) {
            return id;
        }

        let id = self.t.len() as Id + 1;

        self.ts.insert(x.clone(), id);
        self.t.push(x);

        id
    }
}

/// Sort one canonical run and combine equal adjacent keys.
/// # Arguments:
/// - `run`: Canonical terms to reduce.
/// # Returns:
/// - `()`: Sorts and reduces `run` in place.
fn reducerun(run: &mut Vec<RunTerm>) {
    run.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    let mut write = 0usize;
    for read in 0..run.len() {
        if write > 0 && run[write - 1].0 == run[read].0 {
            let c = run[read].1;
            run[write - 1].1 += c;
        } else {
            if write != read {
                run.swap(write, read);
            }
            write += 1;
        }
    }

    run.truncate(write);
    run.retain(|(_, c)| !c.is_zero());
}

/// Merge two sorted unique canonical runs.
/// # Arguments:
/// - `left`: First sorted run.
/// - `right`: Second sorted run.
/// - `out`: Reusable output run.
/// # Returns:
/// - `()`: Drains both inputs into `out`.
fn mergeruns(
    left: &mut Vec<RunTerm>,
    right: &mut Vec<RunTerm>,
    out: &mut Vec<RunTerm>,
) {
    out.clear();
    out.reserve(left.len() + right.len());

    let mut a = left.drain(..).peekable();
    let mut b = right.drain(..).peekable();

    while let (Some(x), Some(y)) = (a.peek(), b.peek()) {
        match x.0.cmp(&y.0) {
            Ordering::Less => out.push(a.next().unwrap()),
            Ordering::Greater => out.push(b.next().unwrap()),
            Ordering::Equal => {
                let (key, mut c) = a.next().unwrap();
                c += b.next().unwrap().1;
                if !c.is_zero() {
                    out.push((key, c));
                }
            }
        }
    }

    out.extend(a);
    out.extend(b);
}

/// Canonical expression accumulator.
pub struct Acc {
    /// Interned canonical factor store.
    store: Store,
    /// Sorted unique canonical terms.
    terms: Vec<RunTerm>,
    /// Unsorted canonical terms awaiting reduction.
    pending: Vec<RunTerm>,
    /// Reusable output buffer for run merging.
    scratch: Vec<RunTerm>,
}

impl Acc {
    /// Create an empty canonical accumulator.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `Acc`: Empty accumulator.
    pub fn new() -> Self {
        Self {
            store: Store::default(),
            terms: Vec::new(),
            pending: Vec::new(),
            scratch: Vec::new(),
        }
    }

    /// Merge the pending terms into the sorted main run.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `()`: Leaves `terms` sorted and unique.
    fn flush(&mut self) {
        if self.pending.is_empty() {
            return;
        }

        reducerun(&mut self.pending);
        if self.pending.is_empty() {
            return;
        }

        if self.terms.is_empty() {
            std::mem::swap(&mut self.terms, &mut self.pending);
            return;
        }

        let Acc {
            terms,
            pending,
            scratch,
            ..
        } = self;

        mergeruns(terms, pending, scratch);
        std::mem::swap(terms, scratch);
    }

    /// Add one coefficient by canonical integer key.
    /// # Arguments:
    /// - `key`: Canonical integer factor key.
    /// - `c`: Coefficient.
    /// # Returns:
    /// - `()`: Mutates the accumulator.
    fn addkey(
        &mut self,
        key: TermKey,
        c: Ratio<i64>,
    ) {
        if c.is_zero() {
            return;
        }

        self.pending.push((key, c));
        if self.pending.len() >= PENDING_LIMIT {
            self.flush();
        }
    }

    /// Add one term to the accumulator.
    /// # Arguments:
    /// - `x`: Term to add.
    /// # Returns:
    /// - `()`: Mutates the accumulator.
    pub fn addterm(
        &mut self,
        mut x: Term,
    ) {
        x.deltas.sort_unstable();

        x.tensors = x.tensors.into_iter().map(ten).collect();
        x.tensors.sort_unstable();

        let ds = x
            .deltas
            .into_iter()
            .map(|d| self.store.delta(d))
            .collect::<SmallVec<[Id; 4]>>();

        let ts = x
            .tensors
            .into_iter()
            .map(|t| self.store.tensor(t))
            .collect::<SmallVec<[Id; 8]>>();

        self.addkey((ds, ts), Ratio::<i64>::new(x.coeff.num, x.coeff.den));
    }

    /// Merge another accumulator into this accumulator.
    /// # Arguments:
    /// - `other`: Accumulator to merge.
    /// # Returns:
    /// - `()`: Mutates the accumulator.
    pub fn merge(
        &mut self,
        mut other: Acc,
    ) {
        crate::time_call!(crate::timers::canonical::add_merge, {
            self.flush();
            other.flush();

            if other.terms.is_empty() {
                return;
            }

            if self.terms.is_empty() {
                *self = other;
                return;
            }

            let Acc {
                store, mut terms, ..
            } = other;

            let mut dmap = Vec::with_capacity(store.d.len() + 1);
            dmap.push(0);
            for d in store.d {
                dmap.push(self.store.delta(d));
            }

            let mut tmap = Vec::with_capacity(store.t.len() + 1);
            tmap.push(0);
            for t in store.t {
                tmap.push(self.store.tensor(t));
            }

            for ((ds, ts), _) in &mut terms {
                for i in ds.iter_mut() {
                    *i = dmap[*i as usize];
                }

                for i in ts.iter_mut() {
                    *i = tmap[*i as usize];
                }
            }

            reducerun(&mut terms);

            if terms.is_empty() {
                return;
            }

            let Acc {
                terms: left,
                scratch,
                ..
            } = self;

            mergeruns(left, &mut terms, scratch);
            std::mem::swap(left, scratch);
        });
    }

    /// Add one expression to the accumulator.
    /// # Arguments:
    /// - `e`: Expression to add.
    /// # Returns:
    /// - `()`: Mutates the accumulator.
    pub fn addexpr(
        &mut self,
        e: Expr,
    ) {
        for x in e {
            self.addterm(x);
        }
    }

    /// Convert the accumulator into an expression without sparsification.
    /// # Arguments:
    /// - `self`: Accumulator.
    /// # Returns:
    /// - `Expr`: Combined expression.
    fn intoexpr(mut self) -> Expr {
        self.flush();

        let Acc { store, terms, .. } = self;
        let Store { d, t, .. } = store;

        let mut out = terms
            .into_iter()
            .filter(|(_, c)| !c.is_zero())
            .map(|((ds, ts), c)| {
                let mut deltas = ds
                    .into_iter()
                    .map(|i| d[(i - 1) as usize])
                    .collect::<Vec<_>>();

                let mut tensors = ts
                    .into_iter()
                    .map(|i| t[(i - 1) as usize].clone())
                    .collect::<Vec<_>>();

                deltas.sort_unstable();
                tensors.sort_unstable();

                Term {
                    coeff: Rational {
                        num: *c.numer(),
                        den: *c.denom(),
                    },
                    deltas,
                    tensors,
                }
            })
            .collect::<Expr>();

        out.sort_unstable_by(|a, b| {
            a.deltas
                .cmp(&b.deltas)
                .then_with(|| a.tensors.cmp(&b.tensors))
        });

        out
    }

    /// Finish canonicalisation.
    /// # Arguments:
    /// - `self`: Accumulator.
    /// # Returns:
    /// - `Expr`: Canonical expression.
    pub fn finish(self) -> Expr {
        crate::time_call!(crate::timers::canonical::add_finish, {
            let e = crate::time_call!(crate::timers::canonical::add_intoexpr, { self.intoexpr() });
            let e = crate::time_call!(crate::timers::canonical::add_spar, { spar(e) });
            crate::time_call!(crate::timers::canonical::add_final_sum, { sum(e) })
        })
    }
}

/// Canonicalise a symbolic expression.
/// # Arguments:
/// - `e`: Symbolic expression.
/// # Returns:
/// - `Expr`: Canonical symbolic expression.
pub fn canon(e: Expr) -> Expr {
    let mut acc = Acc::new();

    crate::time_call!(crate::timers::canonical::add_accumulate, {
        acc.addexpr(e);
    });

    acc.finish()
}

/// Combine equal terms and apply simple tensor symmetries.
/// # Arguments:
/// - `e`: Symbolic expression.
/// # Returns:
/// - `Expr`: Combined expression.
fn sum(e: Expr) -> Expr {
    let mut acc = Acc::new();

    acc.addexpr(e);
    acc.intoexpr()
}

/// Canonicalise one tensor.
/// # Arguments:
/// - `x`: Tensor factor.
/// # Returns:
/// - `Tensor`: Canonical tensor factor.
fn ten(x: Tensor) -> Tensor {
    if x.kind == TensorKind::Lambda2 && x.upper.len() == 2 && x.lower.len() == 2 {
        let y = Tensor {
            kind: TensorKind::Lambda2,
            upper: vec![x.upper[1], x.upper[0]],
            lower: vec![x.lower[1], x.lower[0]],
        };

        if y < x { y } else { x }
    } else {
        x
    }
}

/// Return high cumulant rank.
/// # Arguments:
/// - `k`: Tensor kind.
/// # Returns:
/// - `Option<usize>`: Rank for Lambda3/Lambda4.
fn rank(k: TensorKind) -> Option<usize> {
    match k {
        TensorKind::Lambda3 => Some(3),
        TensorKind::Lambda4 => Some(4),
        _ => None,
    }
}

/// Sparsify high-rank cumulant orbits.
/// # Arguments:
/// - `e`: Symbolic expression.
/// # Returns:
/// - `Expr`: Sparsified expression.
fn spar(e: Expr) -> Expr {
    let mut out = Vec::new();
    let mut gs = BTreeMap::<OrbitKey, BTreeMap<Vec<usize>, Ratio<i64>>>::new();

    for term in e {
        let hs = term
            .tensors
            .iter()
            .enumerate()
            .filter_map(|(i, t)| rank(t.kind).map(|n| (i, n, t)))
            .collect::<Vec<_>>();

        if hs.len() != 1 {
            out.push(term);
            continue;
        }

        let (i, n, lam) = hs[0];

        if lam.upper.len() != n || lam.lower.len() != n {
            out.push(term);
            continue;
        }

        if lam.lower.iter().copied().collect::<BTreeSet<_>>().len() != n {
            out.push(term);
            continue;
        }

        let mut lo = lam.lower.clone();
        lo.sort_unstable();

        let Some(p) = perm(&lo, &lam.lower) else {
            out.push(term);
            continue;
        };

        let mut ts = term.tensors.clone();
        ts.remove(i);

        let key = OrbitKey {
            n,
            d: term.deltas.clone(),
            t: ts,
            k: lam.kind,
            up: lam.upper.clone(),
            lo,
        };

        let c = Ratio::<i64>::new(term.coeff.num, term.coeff.den);
        let v = gs
            .entry(key)
            .or_default()
            .entry(p)
            .or_insert_with(Ratio::<i64>::zero);

        *v += c;
    }

    for (key, cs) in gs {
        for (p, c) in best(key.n, cs) {
            if c.is_zero() {
                continue;
            }

            let mut ts = key.t.clone();
            ts.push(Tensor {
                kind: key.k,
                upper: key.up.clone(),
                lower: p.iter().map(|&i| key.lo[i]).collect(),
            });

            out.push(Term {
                coeff: Rational {
                    num: *c.numer(),
                    den: *c.denom(),
                },
                deltas: key.d.clone(),
                tensors: ts,
            });
        }
    }

    out
}

/// Express `x` as a permutation of `base`.
/// # Arguments:
/// - `base`: Sorted base.
/// - `x`: Actual ordering.
/// # Returns:
/// - `Option<Vec<usize>>`: Permutation.
fn perm(
    base: &[Idx],
    x: &[Idx],
) -> Option<Vec<usize>> {
    let mut out = Vec::with_capacity(x.len());

    for y in x {
        out.push(base.iter().position(|z| z == y)?);
    }

    Some(out)
}

/// Find the sparsest spin-equivalent coefficient vector below a basis fallback.
/// # Arguments:
/// - `n`: Cumulant rank.
/// - `cs`: Coefficients indexed by lower-index permutation.
/// # Returns:
/// - `BTreeMap<Vec<usize>, Ratio<i64>>`: Exact representation no larger than `cs`.
fn best(
    n: usize,
    cs: BTreeMap<Vec<usize>, Ratio<i64>>,
) -> BTreeMap<Vec<usize>, Ratio<i64>> {
    bestn(n, cs, None).0
}

/// One sparsifier profiling record.
/// # Arguments:
/// - None.
/// # Returns:
/// - `BestStats`: Aggregate search statistics.
#[derive(Clone, Debug, Default)]
pub(crate) struct BestStats {
    /// Cumulant rank.
    pub(crate) rank: usize,
    /// Input support size.
    pub(crate) input: usize,
    /// Gram rank.
    pub(crate) gram_rank: usize,
    /// Basis fallback support size.
    pub(crate) basis_support: usize,
    /// Effective searched support limit.
    pub(crate) limit: usize,
    /// Candidate supports visited by candidate support size.
    pub(crate) visited_by_size: BTreeMap<usize, u64>,
    /// Candidate supports rejected before exact solve.
    pub(crate) rejected: u64,
    /// Exact integer span checks performed.
    pub(crate) checks: u64,
    /// Exact rational solves performed.
    pub(crate) solves: u64,
    /// Successful returned support size.
    pub(crate) success_size: Option<usize>,
    /// Whether the known fallback was returned.
    pub(crate) fallback_return: bool,
    /// Time spent in support-size enumeration.
    pub(crate) enumerate_ns: u64,
    /// Time spent in exact cheap rejection.
    pub(crate) reject_ns: u64,
    /// Time spent in exact rational solve.
    pub(crate) solve_ns: u64,
    /// Total single-call elapsed time.
    pub(crate) elapsed_ns: u64,
}

impl BestStats {
    /// Count one visited candidate support.
    /// # Arguments:
    /// - `size`: Candidate support size.
    /// # Returns:
    /// - `()`: Updates this stats record.
    fn visited(
        &mut self,
        size: usize,
    ) {
        #[cfg(feature = "timings")]
        {
            *self.visited_by_size.entry(size).or_default() += 1;
        }
        #[cfg(not(feature = "timings"))]
        let _ = size;
    }

    /// Count one pre-solve rejection.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `()`: Updates this stats record when enabled.
    fn reject(&mut self) {
        #[cfg(feature = "timings")]
        {
            self.rejected += 1;
        }
    }

    /// Count one exact integer span check.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `()`: Updates this stats record when enabled.
    fn check(&mut self) {
        #[cfg(feature = "timings")]
        {
            self.checks += 1;
        }
    }

    /// Count one exact solve.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `()`: Updates this stats record when enabled.
    fn solve(&mut self) {
        #[cfg(feature = "timings")]
        {
            self.solves += 1;
        }
    }
}

/// Find the sparsest equivalent vector below a deterministic fallback.
/// # Arguments:
/// - `n`: Cumulant rank.
/// - `cs`: Coefficients indexed by lower-index permutation.
/// - `stats`: Optional stats sink.
/// # Returns:
/// - `(BTreeMap<Vec<usize>, Ratio<i64>>, BestStats)`: Representation and stats.
fn bestn(
    n: usize,
    cs: BTreeMap<Vec<usize>, Ratio<i64>>,
    mut stats: Option<&mut BestStats>,
) -> (BTreeMap<Vec<usize>, Ratio<i64>>, BestStats) {
    #[cfg(feature = "timings")]
    let start = Instant::now();
    let mut local = BestStats::default();

    let cs = cs
        .into_iter()
        .filter(|(_, c)| !c.is_zero())
        .collect::<BTreeMap<_, _>>();
    local.rank = n;

    if cs.is_empty() {
        done(&mut local, stats.as_deref_mut());
        return (cs, local);
    }

    let original_support = cs.len();
    local.input = original_support;

    let cache = n >= 4 && original_support >= 8;
    let key = cache.then(|| bkey(n, &cs));
    if let Some((key, scale)) = &key
        && let Some(out) = bget(key)
    {
        let out = bscale(out, scale);
        local.basis_support = out.len();
        local.limit = out.len().saturating_sub(1);
        local.success_size = Some(out.len());
        #[cfg(feature = "timings")]
        {
            local.elapsed_ns = start.elapsed().as_nanos() as u64;
        }
        done(&mut local, stats.as_deref_mut());
        return (out, local);
    }

    let data = gram::GramBasis::cached(n).expect("Cached spin sparsifier rank missing.");
    let ps = data.ps.as_slice();
    let g = data.g.as_slice();
    let y = gram::gram_image(ps, g, &cs);
    let (fallback, fallback_support) = fallback(n, ps, g, &y, &cs);
    local.basis_support = fallback_support;
    let limit = fallback_support.saturating_sub(1);
    local.limit = limit;

    if limit == 0 {
        local.fallback_return = true;
        #[cfg(feature = "timings")]
        {
            local.elapsed_ns = start.elapsed().as_nanos() as u64;
        }
        done(&mut local, stats.as_deref_mut());
        return (fallback, local);
    }

    let filter = Filter::new(n, g, &y);
    local.gram_rank = filter.rows().len();
    let yr = filter.rows().iter().map(|&row| y[row]).collect::<Vec<_>>();
    let columns = cols(ps, &fallback);

    let found = {
        let mut search = SupportSearch {
            columns: &columns,
            filter: &filter,
            g,
            yr: &yr,
            y: &y,
            ps,
            support: Vec::with_capacity(limit),
            state: filter.state(),
            stats: &mut local,
        };

        search.run(limit)
    };

    if let Some(out) = found {
        local.success_size = Some(out.len());
        #[cfg(feature = "timings")]
        {
            local.elapsed_ns = start.elapsed().as_nanos() as u64;
        }
        if let Some((key, scale)) = key {
            bput(key, bnorm(&out, &scale));
        }
        done(&mut local, stats.as_deref_mut());
        return (out, local);
    }

    local.fallback_return = true;
    #[cfg(feature = "timings")]
    {
        local.elapsed_ns = start.elapsed().as_nanos() as u64;
    }
    if let Some((key, scale)) = key {
        bput(key, bnorm(&fallback, &scale));
    }
    done(&mut local, stats);
    (fallback, local)
}

/// Build the deterministic search fallback.
/// # Arguments:
/// - `n`: Cumulant rank.
/// - `ps`: Permutations.
/// - `g`: Spin Gram matrix.
/// - `y`: Full target image.
/// - `cs`: Original coefficients.
/// # Returns:
/// - `(BTreeMap<Vec<usize>, Ratio<i64>>, usize)`: Fallback map and support.
fn fallback(
    n: usize,
    ps: &[Vec<usize>],
    g: &[Vec<Ratio<i64>>],
    y: &[Ratio<i64>],
    cs: &BTreeMap<Vec<usize>, Ratio<i64>>,
) -> (BTreeMap<Vec<usize>, Ratio<i64>>, usize) {
    let basis = gram::basis_solution(n, g, y).map(|(support, x)| {
        support
            .into_iter()
            .zip(x)
            .filter(|(_, coefficient)| !coefficient.is_zero())
            .map(|(index, coefficient)| (ps[index].clone(), coefficient))
            .collect::<BTreeMap<_, _>>()
    });

    if let Some(out) = basis
        && out.len() < cs.len()
    {
        let support = out.len();
        return (out, support);
    }

    (cs.clone(), cs.len())
}

/// Build one normalized sparsifier cache key.
/// # Arguments:
/// - `n`: Cumulant rank.
/// - `cs`: Exact coefficient map.
/// # Returns:
/// - `(BestKey, Ratio<i64>)`: Normalized key and scale factor.
fn bkey(
    n: usize,
    cs: &BTreeMap<Vec<usize>, Ratio<i64>>,
) -> (BestKey, Ratio<i64>) {
    let scale = cs.values().next().cloned().unwrap();
    let norm = bnorm(cs, &scale);

    ((n, norm), scale)
}

/// Divide one coefficient map by one scale.
/// # Arguments:
/// - `cs`: Exact coefficient map.
/// - `scale`: Nonzero scale factor.
/// # Returns:
/// - `BTreeMap<Vec<usize>, Ratio<i64>>`: Normalized coefficient map.
fn bnorm(
    cs: &BTreeMap<Vec<usize>, Ratio<i64>>,
    scale: &Ratio<i64>,
) -> BTreeMap<Vec<usize>, Ratio<i64>> {
    cs.iter().map(|(p, c)| (p.clone(), *c / *scale)).collect()
}

/// Multiply one coefficient map by one scale.
/// # Arguments:
/// - `cs`: Exact coefficient map.
/// - `scale`: Scale factor.
/// # Returns:
/// - `BTreeMap<Vec<usize>, Ratio<i64>>`: Scaled coefficient map.
fn bscale(
    cs: BTreeMap<Vec<usize>, Ratio<i64>>,
    scale: &Ratio<i64>,
) -> BTreeMap<Vec<usize>, Ratio<i64>> {
    cs.into_iter().map(|(p, c)| (p, c * *scale)).collect()
}

/// Get one sparsifier cache entry.
/// # Arguments:
/// - `key`: Exact sparsifier input key.
/// # Returns:
/// - `Option<BTreeMap<Vec<usize>, Ratio<i64>>>`: Cached result.
fn bget(key: &BestKey) -> Option<BTreeMap<Vec<usize>, Ratio<i64>>> {
    let mut cache = bcache().lock().unwrap();
    let index = cache.iter().position(|(k, _)| k == key)?;
    let item = cache.remove(index).unwrap();
    let out = item.1.clone();
    cache.push_back(item);
    Some(out)
}

/// Put one sparsifier cache entry.
/// # Arguments:
/// - `key`: Exact sparsifier input key.
/// - `out`: Exact sparsifier output.
/// # Returns:
/// - `()`: Updates cache.
fn bput(
    key: BestKey,
    out: BTreeMap<Vec<usize>, Ratio<i64>>,
) {
    let mut cache = bcache().lock().unwrap();

    if let Some(index) = cache.iter().position(|(k, _)| k == &key) {
        cache.remove(index);
    }

    cache.push_back((key, out));

    if cache.len() > BEST_CACHE {
        cache.pop_front();
    }
}

/// Return sparsifier result cache.
/// # Arguments:
/// - None.
/// # Returns:
/// - `&'static Mutex<VecDeque<BestVal>>`: Shared bounded cache.
fn bcache() -> &'static Mutex<VecDeque<BestVal>> {
    static CACHE: OnceLock<Mutex<VecDeque<BestVal>>> = OnceLock::new();

    CACHE.get_or_init(|| Mutex::new(VecDeque::new()))
}

/// Finish one sparsifier stats record.
/// # Arguments:
/// - `local`: Current call stats.
/// - `stats`: Optional aggregate stats sink.
/// # Returns:
/// - `()`: Updates global and requested stats.
fn done(
    local: &mut BestStats,
    stats: Option<&mut BestStats>,
) {
    #[cfg(feature = "timings")]
    {
        crate::timers::canonical::add_best_rank(local.rank);
        crate::timers::canonical::add_best_input(local.input);
        crate::timers::canonical::add_best_gram_rank(local.gram_rank);
        crate::timers::canonical::add_best_basis_support(local.basis_support);
        crate::timers::canonical::add_best_limit(local.limit);
        for (&size, &count) in &local.visited_by_size {
            crate::timers::canonical::add_best_visited(size, count);
        }
        crate::timers::canonical::add_best_rejected(local.rejected);
        crate::timers::canonical::add_best_checks(local.checks);
        crate::timers::canonical::add_best_solves(local.solves);
        if let Some(size) = local.success_size {
            crate::timers::canonical::add_best_success(size);
        }
        if local.fallback_return {
            crate::timers::canonical::add_best_fallback_return();
        }
        crate::timers::canonical::add_best_enumerate(local.enumerate_ns);
        crate::timers::canonical::add_best_reject_time(local.reject_ns);
        crate::timers::canonical::add_best_solve_time(local.solve_ns);
        crate::timers::canonical::add_best_max(local.elapsed_ns);
    }

    if let Some(stats) = stats {
        for (&size, &count) in &local.visited_by_size {
            *stats.visited_by_size.entry(size).or_default() += count;
        }

        stats.rejected += local.rejected;
        stats.checks += local.checks;
        stats.solves += local.solves;
        stats.success_size = local.success_size;
        stats.fallback_return |= local.fallback_return;
        stats.enumerate_ns += local.enumerate_ns;
        stats.reject_ns += local.reject_ns;
        stats.solve_ns += local.solve_ns;
        stats.elapsed_ns += local.elapsed_ns;
    }
}

/// Return all candidate columns in deterministic order.
/// # Arguments:
/// - `ps`: Permutations.
/// - `cs`: Input coefficient map.
/// # Returns:
/// - `Vec<usize>`: Candidate column order.
fn cols(
    ps: &[Vec<usize>],
    cs: &BTreeMap<Vec<usize>, Ratio<i64>>,
) -> Vec<usize> {
    let mut out = Vec::with_capacity(ps.len());

    for (i, p) in ps.iter().enumerate() {
        if cs.contains_key(p) {
            out.push(i);
        }
    }

    for (i, p) in ps.iter().enumerate() {
        if !cs.contains_key(p) {
            out.push(i);
        }
    }

    out
}

/// Depth-first candidate-support search.
/// # Arguments:
/// - None.
/// # Returns:
/// - `SupportSearch`: Search state.
struct SupportSearch<'a> {
    /// Candidate column order.
    columns: &'a [usize],
    /// Exact rejection filter.
    filter: &'a Filter,
    /// Spin Gram matrix.
    g: &'a [Vec<Ratio<i64>>],
    /// Reduced target image.
    yr: &'a [Ratio<i64>],
    /// Full target image.
    y: &'a [Ratio<i64>],
    /// Lower-index permutations.
    ps: &'a [Vec<usize>],
    /// Current candidate support stack.
    support: Vec<usize>,
    /// Incremental modular span state.
    state: Option<State>,
    /// Search stats.
    stats: &'a mut BestStats,
}

impl<'a> SupportSearch<'a> {
    /// Search all support sizes up to a limit.
    /// # Arguments:
    /// - `limit`: Maximum searched support size.
    /// # Returns:
    /// - `Option<BTreeMap<Vec<usize>, Ratio<i64>>>`: First exact smaller representation.
    fn run(
        &mut self,
        limit: usize,
    ) -> Option<BTreeMap<Vec<usize>, Ratio<i64>>> {
        for size in 1..=limit {
            #[cfg(feature = "timings")]
            let enum_start = Instant::now();
            let found = self.walk(0, size);
            #[cfg(feature = "timings")]
            {
                let elapsed = enum_start.elapsed();
                self.stats.enumerate_ns += elapsed.as_nanos() as u64;
            }

            if found.is_some() {
                return found;
            }
        }

        None
    }

    /// Search one support size in deterministic order.
    /// # Arguments:
    /// - `start`: First column-order index allowed at this depth.
    /// - `want`: Desired candidate support size.
    /// # Returns:
    /// - `Option<BTreeMap<Vec<usize>, Ratio<i64>>>`: First exact smaller representation.
    fn walk(
        &mut self,
        start: usize,
        want: usize,
    ) -> Option<BTreeMap<Vec<usize>, Ratio<i64>>> {
        if self.support.len() == want {
            return self.check(want);
        }

        let need = want - self.support.len();
        let last = self.columns.len() + 1 - need;

        for i in start..last {
            let col = self.columns[i];
            let rank = self
                .state
                .as_mut()
                .map(|state| state.push(self.filter, col));
            self.support.push(col);
            if let Some(out) = self.walk(i + 1, want) {
                return Some(out);
            }
            self.support.pop();
            if let Some(rank) = rank
                && let Some(state) = self.state.as_mut()
            {
                state.pop(rank);
            }
        }

        None
    }

    /// Check the current support.
    /// # Arguments:
    /// - `want`: Desired candidate support size.
    /// # Returns:
    /// - `Option<BTreeMap<Vec<usize>, Ratio<i64>>>`: Exact representation if valid.
    fn check(
        &mut self,
        want: usize,
    ) -> Option<BTreeMap<Vec<usize>, Ratio<i64>>> {
        self.stats.visited(want);

        #[cfg(feature = "timings")]
        let reject_start = Instant::now();
        let plausible = self
            .state
            .as_ref()
            .map(|state| state.span())
            .unwrap_or(true);
        let (plausible, checked) = if plausible {
            self.filter.span_stats(&self.support)
        } else {
            (false, false)
        };
        if checked {
            self.stats.check();
        }
        #[cfg(feature = "timings")]
        {
            self.stats.reject_ns += reject_start.elapsed().as_nanos() as u64;
        }

        if !plausible {
            self.stats.reject();
            return None;
        }

        #[cfg(feature = "timings")]
        let solve_start = Instant::now();
        self.stats.solve();
        let x = gram::solve_rows(self.g, self.yr, &self.support, self.filter.rows())?;
        #[cfg(feature = "timings")]
        {
            self.stats.solve_ns += solve_start.elapsed().as_nanos() as u64;
        }

        let out = self
            .support
            .iter()
            .copied()
            .zip(x)
            .filter(|(_, coefficient)| !coefficient.is_zero())
            .map(|(index, coefficient)| (self.ps[index].clone(), coefficient))
            .collect::<BTreeMap<_, _>>();

        if gram::gram_image(self.ps, self.g, &out) == self.y {
            return Some(out);
        }

        None
    }
}
