// encode.rs

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use bincode::Options;
use num_rational::Ratio;
use num_traits::Zero;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::ir::{Expr, Idx, Space, Tensor, TensorKind, Term};
use crate::schema::{
    GeneratedTerm, OverlapBlockTerms, OverlapTermSet, ResidualClassTerms, ResidualTermSet,
    TensorFactor,
};
use crate::specs::{BLOCKS, BlockSpec, EXCS, ExcSpec, idx};

/// Canonical encoded symbolic index.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, PartialOrd, Ord, Serialize)]
enum IKey {
    /// Free external index.
    Free(u16),
    /// Dummy summed index, labelled by space and local dummy number.
    Dummy(u8, u16),
}

/// Canonical encoded tensor factor.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, PartialOrd, Ord, Serialize)]
struct FKey {
    /// Tensor kind.
    kind: u8,
    /// Upper index keys.
    upper: Vec<IKey>,
    /// Lower index keys.
    lower: Vec<IKey>,
}

/// Canonical encoded term without its coefficient.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, PartialOrd, Ord, Serialize)]
struct TKey {
    /// Delta factors.
    deltas: Vec<[IKey; 2]>,
    /// Tensor factors.
    tensors: Vec<FKey>,
}

#[derive(Clone, Debug)]
struct Acc {
    /// Fixed free indices.
    free: Vec<Idx>,
    /// Free-index map.
    free_ids: BTreeMap<Idx, u16>,
    /// Globally combined terms.
    terms: HashMap<TKey, Ratio<i64>>,
    /// Sorted flushed run files.
    runs: Vec<RunFile>,
    /// Number of raw terms seen.
    seen: usize,
    /// Encoding start time.
    start: Instant,
}

/// Maximum in-memory encoded terms before flushing to disk.
const FLUSH_TERMS: usize = 250_000;

/// Unique encoded-run file counter.
static RUN_ID: AtomicUsize = AtomicUsize::new(0);

/// One serialized run entry.
type RunEntry = (TKey, [i64; 2]);

/// Runtime index names and spaces.
type RuntimeIndexTable = Vec<(String, u8)>;

/// Runtime dummy-index id map.
type DummyIds = BTreeMap<(u8, u16), u16>;

/// Runtime index table and dummy ids.
type RuntimeTable = (RuntimeIndexTable, DummyIds);

/// One sorted encoded run file.
#[derive(Clone, Debug)]
struct RunFile {
    /// Run path.
    path: PathBuf,
    /// Merge level.
    level: usize,
}

/// Encode orbital space.
/// # Arguments:
/// - `x`: Orbital space.
/// # Returns:
/// - `u8`: Runtime space id.
fn sp(x: Space) -> u8 {
    match x {
        Space::Core => 0,
        Space::Active => 1,
        Space::Virtual => 2,
    }
}

/// Encode tensor kind.
/// # Arguments:
/// - `x`: Tensor kind.
/// # Returns:
/// - `u8`: Runtime tensor id.
fn tk(x: TensorKind) -> u8 {
    match x {
        TensorKind::Gamma1 => 0,
        TensorKind::Theta => 1,
        TensorKind::Fock => 2,
        TensorKind::ERI => 3,
        TensorKind::Lambda2 => 4,
        TensorKind::Lambda3 => 5,
        TensorKind::Lambda4 => 6,
        TensorKind::T1 => 8,
        TensorKind::T2 => 9,
    }
}

/// Return the default generated space table.
/// # Arguments:
/// - None.
/// # Returns:
/// - `BTreeMap<String, u8>`: Space-name map.
fn space_kinds() -> BTreeMap<String, u8> {
    let mut out = BTreeMap::new();

    out.insert("core".to_string(), 0);
    out.insert("active".to_string(), 1);
    out.insert("virtual".to_string(), 2);

    out
}

/// Return the default generated tensor table.
/// # Arguments:
/// - None.
/// # Returns:
/// - `BTreeMap<String, u8>`: Tensor-name map.
fn tensor_kinds() -> BTreeMap<String, u8> {
    let mut out = BTreeMap::new();

    out.insert("Gamma1".to_string(), 0);
    out.insert("Theta".to_string(), 1);
    out.insert("f".to_string(), 2);
    out.insert("g".to_string(), 3);
    out.insert("Lambda2".to_string(), 4);
    out.insert("Lambda3".to_string(), 5);
    out.insert("Lambda4".to_string(), 6);
    out.insert("t1".to_string(), 8);
    out.insert("t2".to_string(), 9);

    out
}

/// Return all index occurrences in one term.
/// # Arguments:
/// - `t`: Term.
/// # Returns:
/// - `Vec<Idx>`: Index occurrences.
fn inds(t: &Term) -> Vec<Idx> {
    let mut out = Vec::new();

    for d in &t.deltas {
        out.push(d.left);
        out.push(d.right);
    }

    for x in &t.tensors {
        out.extend(x.upper.iter().copied());
        out.extend(x.lower.iter().copied());
    }

    out
}

/// Return one encoded index from an index map.
/// # Arguments:
/// - `ids`: Index map.
/// - `x`: Symbolic index.
/// # Returns:
/// - `IKey`: Encoded index.
fn ikey(
    ids: &BTreeMap<Idx, IKey>,
    x: Idx,
) -> IKey {
    *ids.get(&x)
        .unwrap_or_else(|| panic!("missing canonical id for index {}", x.name))
}

/// Build one encoded tensor factor from an index map.
/// # Arguments:
/// - `x`: Tensor factor.
/// - `ids`: Index map.
/// # Returns:
/// - `FKey`: Encoded tensor factor.
fn fkey(
    x: &Tensor,
    ids: &BTreeMap<Idx, IKey>,
) -> FKey {
    FKey {
        kind: tk(x.kind),
        upper: x.upper.iter().map(|&i| ikey(ids, i)).collect(),
        lower: x.lower.iter().map(|&i| ikey(ids, i)).collect(),
    }
}

/// Insert all fixed free-index mappings.
/// # Arguments:
/// - `out`: Index map.
/// - `free`: Free-index map.
/// # Returns:
/// - `()`: Mutates `out`.
fn insert_free(
    out: &mut BTreeMap<Idx, IKey>,
    free: &BTreeMap<Idx, u16>,
) {
    for (&x, &id) in free {
        out.insert(x, IKey::Free(id));
    }
}

/// Assign one dummy index if required.
/// # Arguments:
/// - `out`: Index map.
/// - `free`: Free-index map.
/// - `next`: Next dummy id per space.
/// - `x`: Symbolic index.
/// # Returns:
/// - `()`: Mutates `out`.
fn assign(
    out: &mut BTreeMap<Idx, IKey>,
    free: &BTreeMap<Idx, u16>,
    next: &mut [u16; 3],
    x: Idx,
) {
    if free.contains_key(&x) || out.contains_key(&x) {
        return;
    }

    let s = sp(x.space);
    let n = next[s as usize];

    next[s as usize] += 1;
    out.insert(x, IKey::Dummy(s, n));
}

/// Build an initial dummy-index map.
/// # Arguments:
/// - `t`: Term.
/// - `free`: Free-index map.
/// # Returns:
/// - `BTreeMap<Idx, IKey>`: Initial index map.
fn initial(
    t: &Term,
    free: &BTreeMap<Idx, u16>,
) -> BTreeMap<Idx, IKey> {
    let mut out = BTreeMap::new();
    let mut next = [0u16; 3];

    insert_free(&mut out, free);

    for x in inds(t) {
        assign(&mut out, free, &mut next, x);
    }

    out
}

/// Return raw deltas in canonical order under the current index map.
/// # Arguments:
/// - `t`: Term.
/// - `ids`: Current index map.
/// # Returns:
/// - `Vec<(Idx, Idx)>`: Raw delta pairs, oriented and sorted.
fn ordered_deltas(
    t: &Term,
    ids: &BTreeMap<Idx, IKey>,
) -> Vec<(Idx, Idx)> {
    let mut out = Vec::new();

    for d in &t.deltas {
        let mut l = d.left;
        let mut r = d.right;

        if ikey(ids, r) < ikey(ids, l) {
            std::mem::swap(&mut l, &mut r);
        }

        out.push((l, r));
    }

    out.sort_by_key(|&(l, r)| [ikey(ids, l), ikey(ids, r)]);
    out
}

/// Return raw tensors in canonical order under the current index map.
/// # Arguments:
/// - `t`: Term.
/// - `ids`: Current index map.
/// # Returns:
/// - `Vec<&Tensor>`: Tensor factors sorted by encoded representation.
fn ordered_tensors<'a>(
    t: &'a Term,
    ids: &BTreeMap<Idx, IKey>,
) -> Vec<&'a Tensor> {
    let mut out = t.tensors.iter().collect::<Vec<_>>();

    out.sort_by_key(|x| fkey(x, ids));
    out
}

/// Rebuild dummy labels from the current canonical factor order.
/// # Arguments:
/// - `t`: Term.
/// - `free`: Free-index map.
/// - `ids`: Current index map.
/// # Returns:
/// - `BTreeMap<Idx, IKey>`: Refined index map.
fn refine(
    t: &Term,
    free: &BTreeMap<Idx, u16>,
    ids: &BTreeMap<Idx, IKey>,
) -> BTreeMap<Idx, IKey> {
    let mut out = BTreeMap::new();
    let mut next = [0u16; 3];

    insert_free(&mut out, free);

    for (l, r) in ordered_deltas(t, ids) {
        assign(&mut out, free, &mut next, l);
        assign(&mut out, free, &mut next, r);
    }

    for x in ordered_tensors(t, ids) {
        for &i in &x.upper {
            assign(&mut out, free, &mut next, i);
        }

        for &i in &x.lower {
            assign(&mut out, free, &mut next, i);
        }
    }

    out
}

/// Canonicalise one term to an alpha-renamed key.
/// # Arguments:
/// - `t`: Term.
/// - `free`: Free-index map.
/// # Returns:
/// - `TKey`: Canonical term key without coefficient.
fn key(
    t: &Term,
    free: &BTreeMap<Idx, u16>,
) -> TKey {
    let mut ids = initial(t, free);

    for _ in 0..8 {
        let next = refine(t, free, &ids);

        if next == ids {
            break;
        }

        ids = next;
    }

    let deltas = ordered_deltas(t, &ids)
        .into_iter()
        .map(|(l, r)| [ikey(&ids, l), ikey(&ids, r)])
        .collect();

    let tensors = ordered_tensors(t, &ids)
        .into_iter()
        .map(|x| fkey(x, &ids))
        .collect();

    TKey { deltas, tensors }
}

/// Convert a symbolic coefficient to a rational.
/// # Arguments:
/// - `t`: Term.
/// # Returns:
/// - `Ratio<i64>`: Rational coefficient.
fn coeff(t: &Term) -> Ratio<i64> {
    Ratio::new(t.coeff.num, t.coeff.den)
}

/// Return all dummy labels used by one key.
/// # Arguments:
/// - `k`: Term key.
/// # Returns:
/// - `BTreeSet<(u8, u16)>`: Dummy labels as `(space, local_id)`.
fn dummies(k: &TKey) -> BTreeSet<(u8, u16)> {
    let mut out = BTreeSet::new();

    for d in &k.deltas {
        for &x in d {
            if let IKey::Dummy(s, n) = x {
                out.insert((s, n));
            }
        }
    }

    for f in &k.tensors {
        for &x in f.upper.iter().chain(f.lower.iter()) {
            if let IKey::Dummy(s, n) = x {
                out.insert((s, n));
            }
        }
    }

    out
}

/// Add one dummy loop id in first-occurrence order.
/// # Arguments:
/// - `out`: Loop id list.
/// - `ids`: Runtime dummy-id map.
/// - `x`: Encoded index.
/// # Returns:
/// - `()`: Mutates `out`.
fn push_loop(
    out: &mut Vec<u16>,
    ids: &BTreeMap<(u8, u16), u16>,
    x: IKey,
) {
    if let IKey::Dummy(s, n) = x {
        let id = ids[&(s, n)];

        if !out.contains(&id) {
            out.push(id);
        }
    }
}

/// Convert one encoded index to a runtime index id.
/// # Arguments:
/// - `x`: Encoded index.
/// - `ids`: Runtime dummy-id map.
/// # Returns:
/// - `u16`: Runtime class-local index id.
fn rid(
    x: IKey,
    ids: &BTreeMap<(u8, u16), u16>,
) -> u16 {
    match x {
        IKey::Free(i) => i,
        IKey::Dummy(s, n) => ids[&(s, n)],
    }
}

/// Build a generated dummy-index name.
/// # Arguments:
/// - `space`: Runtime space id.
/// - `n`: Dummy number.
/// # Returns:
/// - `String`: Generated dummy label.
fn dummy_name(
    space: u8,
    n: u16,
) -> String {
    match space {
        0 => format!("dc{n}"),
        1 => format!("da{n}"),
        2 => format!("dv{n}"),
        _ => panic!("unknown space id {space}"),
    }
}

/// Add one coefficient to a canonical term map.
/// # Arguments:
/// - `map`: Canonical term map.
/// - `k`: Canonical term key.
/// - `c`: Coefficient to add.
/// # Returns:
/// - `()`: Mutates `map`.
fn addkey(
    map: &mut HashMap<TKey, Ratio<i64>>,
    k: TKey,
    c: Ratio<i64>,
) {
    if c.is_zero() {
        return;
    }

    match map.entry(k) {
        Entry::Vacant(e) => {
            e.insert(c);
        }
        Entry::Occupied(mut e) => {
            *e.get_mut() += c;

            if e.get().is_zero() {
                e.remove();
            }
        }
    }
}

/// Merge one canonical term map into another.
/// # Arguments:
/// - `dst`: Destination term map.
/// - `src`: Source term map.
/// # Returns:
/// - `()`: Mutates `dst`.
fn mergemap(
    dst: &mut HashMap<TKey, Ratio<i64>>,
    src: HashMap<TKey, Ratio<i64>>,
) {
    for (k, c) in src {
        addkey(dst, k, c);
    }
}

/// Return sorted canonical term entries.
/// # Arguments:
/// - `terms`: Canonical term map.
/// # Returns:
/// - `Vec<(&TKey, &Ratio<i64>)>`: Deterministically sorted term entries.
fn sorted(terms: &HashMap<TKey, Ratio<i64>>) -> Vec<(&TKey, &Ratio<i64>)> {
    let mut out = terms.iter().collect::<Vec<_>>();

    out.sort_by_key(|(a, _)| *a);

    out
}

/// Build one generated term.
/// # Arguments:
/// - `k`: Encoded term key.
/// - `c`: Term coefficient.
/// - `ids`: Runtime dummy-id map.
/// # Returns:
/// - `GeneratedTerm`: Runtime generated term.
fn term(
    k: &TKey,
    c: &Ratio<i64>,
    ids: &BTreeMap<(u8, u16), u16>,
) -> GeneratedTerm {
    let mut loops = Vec::new();

    for d in &k.deltas {
        push_loop(&mut loops, ids, d[0]);
        push_loop(&mut loops, ids, d[1]);
    }

    for f in &k.tensors {
        for &x in f.upper.iter().chain(f.lower.iter()) {
            push_loop(&mut loops, ids, x);
        }
    }

    GeneratedTerm(
        [*c.numer(), *c.denom()],
        loops,
        k.deltas
            .iter()
            .map(|d| [rid(d[0], ids), rid(d[1], ids)])
            .collect(),
        k.tensors
            .iter()
            .map(|f| {
                TensorFactor(
                    f.kind,
                    f.upper.iter().map(|&x| rid(x, ids)).collect(),
                    f.lower.iter().map(|&x| rid(x, ids)).collect(),
                )
            })
            .collect(),
    )
}

/// Return a unique temporary run path.
/// # Arguments:
/// - None.
/// # Returns:
/// - `PathBuf`: Temporary run path.
fn runpath() -> PathBuf {
    let id = RUN_ID.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let root = std::env::var_os("OUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let dir = root.join(format!("noci-wick-encode-{pid}"));

    fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("failed to create {}: {e}", dir.display()));
    dir.join(format!("run-{id}.bin"))
}

/// Remove one temporary run file.
/// # Arguments:
/// - `path`: Run path.
/// # Returns:
/// - `()`: Removes the file and empty parent directory.
fn removerun(path: &PathBuf) {
    let _ = fs::remove_file(path);

    if let Some(dir) = path.parent() {
        let _ = fs::remove_dir(dir);
    }
}

/// Convert one rational coefficient to a serialized pair.
/// # Arguments:
/// - `x`: Rational coefficient.
/// # Returns:
/// - `[i64; 2]`: Numerator and denominator.
fn rpair(x: &Ratio<i64>) -> [i64; 2] {
    [*x.numer(), *x.denom()]
}

/// Convert one serialized coefficient pair to a rational.
/// # Arguments:
/// - `x`: Numerator and denominator.
/// # Returns:
/// - `Ratio<i64>`: Rational coefficient.
fn ratio(x: [i64; 2]) -> Ratio<i64> {
    Ratio::new(x[0], x[1])
}

/// Write one sorted encoded run.
/// # Arguments:
/// - `path`: Output path.
/// - `xs`: Sorted encoded entries.
/// # Returns:
/// - `()`: Writes one temporary run.
fn writerun(
    path: &PathBuf,
    xs: &[(TKey, Ratio<i64>)],
) {
    let file =
        File::create(path).unwrap_or_else(|e| panic!("failed to create {}: {e}", path.display()));
    let mut out = BufWriter::new(file);
    let options = bincode::DefaultOptions::new().with_varint_encoding();

    for (k, c) in xs {
        let entry = (k, rpair(c));
        options
            .serialize_into(&mut out, &entry)
            .unwrap_or_else(|e| panic!("failed to write {}: {e}", path.display()));
    }
}

/// Read one sorted encoded run.
struct RunReader {
    /// Input path.
    path: PathBuf,
    /// Bincode input.
    input: BufReader<File>,
}

impl RunReader {
    /// Open one sorted encoded run.
    /// # Arguments:
    /// - `path`: Input path.
    /// # Returns:
    /// - `RunReader`: Run reader.
    fn open(path: PathBuf) -> Self {
        let file =
            File::open(&path).unwrap_or_else(|e| panic!("failed to open {}: {e}", path.display()));

        Self {
            path,
            input: BufReader::new(file),
        }
    }

    /// Read one encoded entry.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `Option<(TKey, Ratio<i64>)>`: Next entry if present.
    fn next(&mut self) -> Option<(TKey, Ratio<i64>)> {
        let options = bincode::DefaultOptions::new().with_varint_encoding();

        match options.deserialize_from::<_, RunEntry>(&mut self.input) {
            Ok((k, c)) => Some((k, ratio(c))),
            Err(e) => {
                if let bincode::ErrorKind::Io(ref err) = *e
                    && err.kind() == ErrorKind::UnexpectedEof
                {
                    return None;
                }

                panic!("failed to read {}: {e}", self.path.display());
            }
        }
    }
}

/// One run merge heap entry.
#[derive(Clone, Debug)]
struct Head {
    /// Encoded key.
    key: TKey,
    /// Coefficient.
    coeff: Ratio<i64>,
    /// Run index.
    run: usize,
}

impl Eq for Head {}

impl PartialEq for Head {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        self.key == other.key && self.run == other.run
    }
}

impl Ord for Head {
    fn cmp(
        &self,
        other: &Self,
    ) -> std::cmp::Ordering {
        other
            .key
            .cmp(&self.key)
            .then_with(|| other.run.cmp(&self.run))
    }
}

impl PartialOrd for Head {
    fn partial_cmp(
        &self,
        other: &Self,
    ) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Return whether encoder progress should be printed.
/// # Arguments:
/// - None.
/// # Returns:
/// - `bool`: True if `WICK_ENCODE_PROGRESS` is set.
fn encprog() -> bool {
    std::env::var_os("WICK_ENCODE_PROGRESS").is_some()
}

impl Acc {
    /// Construct an empty global accumulator.
    /// # Arguments:
    /// - `free`: Fixed free indices.
    /// # Returns:
    /// - `Acc`: Empty accumulator.
    fn new(free: Vec<Idx>) -> Self {
        let free_ids = free
            .iter()
            .enumerate()
            .map(|(i, &x)| (x, i as u16))
            .collect();

        Self {
            free,
            free_ids,
            terms: HashMap::new(),
            runs: Vec::new(),
            seen: 0,
            start: Instant::now(),
        }
    }

    /// Flush the in-memory encoded map to a sorted run.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `()`: Writes one temporary run and clears memory.
    fn flush(&mut self) {
        if self.terms.is_empty() {
            return;
        }

        let mut out = self.terms.drain().collect::<Vec<_>>();

        out.sort_by(|(a, _), (b, _)| a.cmp(b));
        out.retain(|(_, c)| !c.is_zero());

        if out.is_empty() {
            return;
        }

        let path = runpath();
        if encprog() {
            eprintln!(
                "[wick-time] encode: flush run {}, raw terms: {}, unique terms: {}, elapsed {:?}.",
                self.runs.len() + 1,
                self.seen,
                out.len(),
                self.start.elapsed()
            );
        }
        writerun(&path, &out);
        self.runs.push(RunFile { path, level: 0 });
        self.compact();
    }

    /// Add one symbolic expression to the global canonical accumulator.
    /// # Arguments:
    /// - `e`: Symbolic expression chunk.
    /// # Returns:
    /// - `()`: Mutates `self`.
    fn addexpr(
        &mut self,
        e: Expr,
    ) {
        let n = e.len();
        let free = &self.free_ids;

        let local = e
            .into_par_iter()
            .fold(HashMap::new, |mut acc, t| {
                let k = key(&t, free);
                let c = coeff(&t);

                addkey(&mut acc, k, c);

                acc
            })
            .reduce(HashMap::new, |mut a, b| {
                mergemap(&mut a, b);
                a
            });

        self.seen += n;
        mergemap(&mut self.terms, local);

        if self.terms.len() >= FLUSH_TERMS {
            self.flush();
        }

        if encprog() {
            crate::progress::mem(format!(
                "encode chunk terms: {n}, raw terms: {}, unique terms: {}",
                self.seen,
                self.terms.len()
            ));
        }
    }

    /// Merge all sorted runs into one sorted run.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `Option<PathBuf>`: Final sorted run path, if flushed.
    fn run(&mut self) -> Option<PathBuf> {
        if !self.runs.is_empty() {
            self.flush();
        }

        match self.runs.len() {
            0 => None,
            1 => {
                if encprog() {
                    eprintln!(
                        "[wick-time] encode: using one flushed run, elapsed {:?}.",
                        self.start.elapsed()
                    );
                }
                Some(self.runs[0].path.clone())
            }
            _ => {
                while self.runs.len() > 1 {
                    self.compact1();
                }

                Some(self.runs[0].path.clone())
            }
        }
    }

    /// Compact runs with matching levels.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `()`: Merges runs until no equal levels remain.
    fn compact(&mut self) {
        loop {
            let mut found = None;

            'outer: for i in 0..self.runs.len() {
                for j in i + 1..self.runs.len() {
                    if self.runs[i].level == self.runs[j].level {
                        found = Some((i, j));
                        break 'outer;
                    }
                }
            }

            let Some((i, j)) = found else {
                break;
            };

            self.merge2(i, j);
        }
    }

    /// Compact one pair of runs.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `()`: Merges one pair of runs.
    fn compact1(&mut self) {
        if self.runs.len() <= 1 {
            return;
        }

        let mut pair = (0, 1);
        let mut level = self.runs[0].level.min(self.runs[1].level);

        for i in 0..self.runs.len() {
            for j in i + 1..self.runs.len() {
                let next = self.runs[i].level.min(self.runs[j].level);
                if next < level {
                    level = next;
                    pair = (i, j);
                }
            }
        }

        self.merge2(pair.0, pair.1);
    }

    /// Merge two sorted runs into one sorted run.
    /// # Arguments:
    /// - `i`: First run index.
    /// - `j`: Second run index.
    /// # Returns:
    /// - `()`: Replaces both inputs with one higher-level run.
    fn merge2(
        &mut self,
        i: usize,
        j: usize,
    ) {
        let right = self.runs.remove(j);
        let left = self.runs.remove(i);
        let level = left.level.max(right.level) + 1;
        let inputs = [left.path, right.path];

        if encprog() {
            eprintln!(
                "[wick-time] encode: merge start, level: {level}, runs: 2, elapsed {:?}.",
                self.start.elapsed()
            );
        }

        let path = self.merge_paths(&inputs);

        for path in &inputs {
            removerun(path);
        }

        self.runs.push(RunFile { path, level });
    }

    /// Merge sorted run paths into one sorted run.
    /// # Arguments:
    /// - `paths`: Input run paths.
    /// # Returns:
    /// - `PathBuf`: Merged run path.
    fn merge_paths(
        &self,
        paths: &[PathBuf],
    ) -> PathBuf {
        let mut readers = paths
            .iter()
            .cloned()
            .map(RunReader::open)
            .collect::<Vec<_>>();
        let mut heap = BinaryHeap::new();

        for (run, reader) in readers.iter_mut().enumerate() {
            if let Some((key, coeff)) = reader.next() {
                heap.push(Head { key, coeff, run });
            }
        }

        let path = runpath();
        let file = File::create(&path)
            .unwrap_or_else(|e| panic!("failed to create {}: {e}", path.display()));
        let mut out = BufWriter::new(file);
        let options = bincode::DefaultOptions::new().with_varint_encoding();
        let mut merged = 0usize;

        while let Some(head) = heap.pop() {
            let key = head.key;
            let mut coeff = head.coeff;
            let run = head.run;

            if let Some((key, coeff)) = readers[run].next() {
                heap.push(Head { key, coeff, run });
            }

            while heap.peek().map(|x| &x.key) == Some(&key) {
                let head = heap.pop().unwrap();
                coeff += head.coeff;
                let run = head.run;

                if let Some((key, coeff)) = readers[run].next() {
                    heap.push(Head { key, coeff, run });
                }
            }

            if !coeff.is_zero() {
                let entry = (key, rpair(&coeff));
                options
                    .serialize_into(&mut out, &entry)
                    .unwrap_or_else(|e| panic!("failed to write {}: {e}", path.display()));
                merged += 1;

                if encprog() && merged.is_multiple_of(1_000_000) {
                    eprintln!(
                        "[wick-time] encode: merge terms: {merged}, elapsed {:?}.",
                        self.start.elapsed()
                    );
                }
            }
        }

        if encprog() {
            eprintln!(
                "[wick-time] encode: merge end, terms: {merged}, elapsed {:?}.",
                self.start.elapsed()
            );
        }

        path
    }

    /// Build the common runtime index table.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `RuntimeTable`: Runtime indices and dummy ids.
    fn table(&self) -> RuntimeTable {
        let mut indices = self
            .free
            .iter()
            .map(|x| (x.name.to_string(), sp(x.space)))
            .collect::<Vec<_>>();

        let mut seen = BTreeSet::new();

        for (k, _) in sorted(&self.terms) {
            seen.extend(dummies(k));
        }

        let mut ids = BTreeMap::new();

        for (s, n) in seen {
            let id = indices.len() as u16;

            ids.insert((s, n), id);
            indices.push((dummy_name(s, n), s));
        }

        (indices, ids)
    }

    /// Convert the accumulator to generated terms.
    /// # Arguments:
    /// - `ids`: Runtime dummy-id map.
    /// # Returns:
    /// - `Vec<GeneratedTerm>`: Runtime generated terms.
    fn terms(
        &self,
        ids: &DummyIds,
    ) -> Vec<GeneratedTerm> {
        let mut out = Vec::new();

        for (k, c) in sorted(&self.terms) {
            if c.is_zero() {
                continue;
            }

            out.push(term(k, c, ids));
        }

        out
    }

    /// Build the common runtime index table from a run.
    /// # Arguments:
    /// - `path`: Final sorted run path.
    /// # Returns:
    /// - `RuntimeTable`: Runtime indices and dummy ids.
    fn table_run(
        &self,
        path: &Path,
    ) -> RuntimeTable {
        if encprog() {
            eprintln!(
                "[wick-time] encode: build index table from encoded run, elapsed {:?}.",
                self.start.elapsed()
            );
        }

        let mut indices = self
            .free
            .iter()
            .map(|x| (x.name.to_string(), sp(x.space)))
            .collect::<Vec<_>>();
        let mut seen = BTreeSet::new();
        let mut input = RunReader::open(path.to_path_buf());

        while let Some((k, _)) = input.next() {
            seen.extend(dummies(&k));
        }

        let mut ids = BTreeMap::new();

        for (s, n) in seen {
            let id = indices.len() as u16;

            ids.insert((s, n), id);
            indices.push((dummy_name(s, n), s));
        }

        (indices, ids)
    }

    /// Convert a run to generated terms.
    /// # Arguments:
    /// - `path`: Final sorted run path.
    /// - `ids`: Runtime dummy-id map.
    /// # Returns:
    /// - `Vec<GeneratedTerm>`: Runtime generated terms.
    fn terms_run(
        &self,
        path: &Path,
        ids: &DummyIds,
    ) -> Vec<GeneratedTerm> {
        if encprog() {
            eprintln!(
                "[wick-time] encode: build generated terms from encoded run, elapsed {:?}.",
                self.start.elapsed()
            );
        }

        let mut out = Vec::new();
        let mut input = RunReader::open(path.to_path_buf());
        let mut n = 0usize;

        while let Some((k, c)) = input.next() {
            if c.is_zero() {
                continue;
            }

            out.push(term(&k, &c, ids));
            n += 1;

            if encprog() && n.is_multiple_of(1_000_000) {
                eprintln!(
                    "[wick-time] encode: generated terms: {n}, elapsed {:?}.",
                    self.start.elapsed()
                );
            }
        }

        out
    }

    /// Convert to one residual class table.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `ResidualClassTerms`: Runtime residual class terms.
    fn residual(mut self) -> ResidualClassTerms {
        let free = (0..self.free.len()).map(|i| i as u16).collect();
        let run = self.run();
        let (indices, ids) = if let Some(path) = &run {
            self.table_run(path)
        } else {
            self.table()
        };
        let terms = if let Some(path) = &run {
            self.terms_run(path, &ids)
        } else {
            self.terms(&ids)
        };

        if let Some(path) = run {
            removerun(&path);
        }

        ResidualClassTerms {
            indices,
            free,
            terms,
        }
    }

    /// Convert to one overlap block table.
    /// # Arguments:
    /// - `b`: Block specification.
    /// # Returns:
    /// - `OverlapBlockTerms`: Runtime overlap block terms.
    fn overlap(
        self,
        b: BlockSpec,
    ) -> OverlapBlockTerms {
        let left_free = (0..b.lf.len()).map(|i| i as u16).collect::<Vec<_>>();
        let right_free = (b.lf.len()..b.lf.len() + b.rf.len())
            .map(|i| i as u16)
            .collect::<Vec<_>>();
        let (indices, ids) = self.table();
        let terms = self.terms(&ids);

        OverlapBlockTerms {
            left: b.left.to_string(),
            right: b.right.to_string(),
            indices,
            left_free,
            right_free,
            terms,
        }
    }
}

/// Return residual free indices.
/// # Arguments:
/// - `x`: Excitation-class specification.
/// # Returns:
/// - `Vec<Idx>`: Free indices.
fn rfree(x: ExcSpec) -> Vec<Idx> {
    x.f.iter().map(|&name| idx(name)).collect()
}

/// Return overlap free indices.
/// # Arguments:
/// - `b`: Block specification.
/// # Returns:
/// - `Vec<Idx>`: Free indices.
fn bfree(b: BlockSpec) -> Vec<Idx> {
    b.lf.iter()
        .chain(b.rf.iter())
        .map(|&name| idx(name))
        .collect()
}

/// Generate one compact metric block.
/// # Arguments:
/// - `b`: Block metadata.
/// # Returns:
/// - `OverlapBlockTerms`: Runtime block terms.
fn block_terms(b: BlockSpec) -> OverlapBlockTerms {
    let mut acc = Acc::new(bfree(b));

    acc.addexpr(crate::wick::eval(&crate::overlap::block(b.name)));
    acc.overlap(b)
}

/// Generate one compact residual class.
/// # Arguments:
/// - `order`: Residual order.
/// - `name`: Excitation-class name.
/// # Returns:
/// - `ResidualClassTerms`: Runtime residual class terms.
pub fn residual_class(
    order: u8,
    name: &str,
) -> ResidualClassTerms {
    let x = EXCS
        .iter()
        .copied()
        .find(|x| x.name == name)
        .unwrap_or_else(|| panic!("unknown excitation class {name}"));

    let mut acc = Acc::new(rfree(x));

    match order {
        0 => crate::residual::r0(x.name, |_, e| acc.addexpr(e)),
        1 => crate::residual::r1(x.name, |_, e| acc.addexpr(e)),
        2 => crate::residual::r2(x.name, |_, e| acc.addexpr(e)),
        _ => panic!("unsupported residual order {order}"),
    }

    acc.residual()
}

/// Generate compact metric terms.
/// # Arguments:
/// - None.
/// # Returns:
/// - `OverlapTermSet`: Complete metric term table.
pub fn overlap_terms() -> OverlapTermSet {
    OverlapTermSet {
        version: 1,
        space_kinds: space_kinds(),
        tensor_kinds: tensor_kinds(),
        blocks: BLOCKS
            .iter()
            .map(|&b| (b.name.to_string(), block_terms(b)))
            .collect(),
    }
}

/// Generate compact residual terms.
/// # Arguments:
/// - `order`: Residual order.
/// # Returns:
/// - `ResidualTermSet`: Complete residual term table.
pub fn residual_terms(order: u8) -> ResidualTermSet {
    ResidualTermSet {
        version: 1,
        order,
        space_kinds: space_kinds(),
        tensor_kinds: tensor_kinds(),
        classes: EXCS
            .iter()
            .map(|&x| (x.name.to_string(), residual_class(order, x.name)))
            .collect(),
    }
}
