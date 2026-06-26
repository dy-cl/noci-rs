// gram.rs

use crate::{field, int};
use itertools::Itertools;
use num_rational::Ratio;
use num_traits::Zero;
use std::collections::BTreeMap;
use std::sync::OnceLock;

// For a spin-orbital n-body cumulant if we fix the upper-index order
// the lower indices have n! possible permutations. One set of indices is
// kept fixed as it is only their order relative to one another that defines
// a unique permutation. These permutations label columns of the spin Gram
// matrix which is used to determine whether different coefficient vectors
// may represent the same cumulant and to find a sparser representation. Since
// the largest cumulants retained are rank four the largest permutation space
// is is 4!
pub(crate) const MAX: usize = 24;

// In practice however the spin Gram matrix will have a null space dimension
// of 10 for a 4-cumulant (see Young diagrams and Catalan numbers) and as such
// once independent vectors are obtained we may operate in a reduced space of
// dimension 14.
pub(crate) const RMAX: usize = 14;

/// Spin algebra data.
#[derive(Clone, Debug)]
pub(crate) struct GramBasis {
    /// All permutations in S_n for cumulant rank n.
    pub(crate) ps: Vec<Vec<usize>>,
    /// n! by n! spin Gram matrix.
    pub(crate) g: Vec<Vec<Ratio<i64>>>,
    /// Indices of Gram matrix rows which are not linearly dependent.
    rows: Vec<usize>,
    /// Indices of Gram matrix columns which form a non-singular Gram matrix.
    basis: Vec<usize>,
}

impl GramBasis {
    /// Build one spin algebra basis.
    /// # Arguments:
    /// - `n`: Number of elements.
    /// # Returns:
    /// - `Self`: Permutations and Gram matrix.
    fn new(n: usize) -> Self {
        let ps = (0..n).permutations(n).collect::<Vec<_>>();
        let g = build_gram(&ps);
        let rows = independent_rows(&g);
        let basis = independent_cols(&g, &rows);

        Self { ps, g, rows, basis }
    }

    /// Return cached spin algebra basis.
    /// # Arguments:
    /// - `n`: Number of elements.
    /// # Returns:
    /// - `Option<&'static Self>`: Cached basis.
    pub(crate) fn cached(n: usize) -> Option<&'static Self> {
        static BASES: OnceLock<[GramBasis; 5]> = OnceLock::new();

        BASES
            .get_or_init(|| std::array::from_fn(GramBasis::new))
            .get(n)
    }
}

/// One saved incremental state marker.
pub struct Mark {
    /// Previous rank.
    rank: usize,
    /// Previous reduced target.
    y: [u16; MAX],
}

/// Reduced Gram representations for one cumulant rank.
struct GramMats {
    /// Number of independent Gram rows or rank of the Gram matrix.
    n: usize,
    /// Indices of the independent rows selected from the full Gram matrix.
    rows: Vec<usize>,
    /// Selected-row Gram columns stored as exact integers.
    a: Vec<[i128; MAX]>,
    /// Selected-row Gram columns modulo the fast incremental-state prime.
    s: Vec<[u16; MAX]>,
    /// Selected-row Gram columns modulo the certificate primes.
    f: [Vec<[u64; MAX]>; 3],
}

impl GramMats {
    /// Construct reduced and convenient representations of the Gram matrix.
    /// # Arguments:
    /// - `n`: Cumulant rank.
    /// - `g`: Spin Gram matrix.
    /// # Returns:
    /// - `Self`: Matrix projections.
    fn new(
        n: usize,
        g: &[Vec<Ratio<i64>>],
    ) -> Self {
        let data = GramBasis::cached(n).unwrap();

        Self {
            n: data.rows.len(),
            rows: data.rows.clone(),
            a: integer_columns(g, &data.rows),
            s: field::columns_fast_u16(g, &data.rows),
            f: field::columns_u64(g, &data.rows),
        }
    }

    /// Return cached fixed Gram projections.
    /// # Arguments:
    /// - `n`: Cumulant rank.
    /// - `g`: Spin Gram matrix.
    /// # Returns:
    /// - `&'static Self`: Cached matrix projections.
    fn cached(
        n: usize,
        g: &[Vec<Ratio<i64>>],
    ) -> &'static Self {
        static M0: OnceLock<GramMats> = OnceLock::new();
        static M1: OnceLock<GramMats> = OnceLock::new();
        static M2: OnceLock<GramMats> = OnceLock::new();
        static M3: OnceLock<GramMats> = OnceLock::new();
        static M4: OnceLock<GramMats> = OnceLock::new();

        match n {
            0 => M0.get_or_init(|| Self::new(n, g)),
            1 => M1.get_or_init(|| Self::new(n, g)),
            2 => M2.get_or_init(|| Self::new(n, g)),
            3 => M3.get_or_init(|| Self::new(n, g)),
            4 => M4.get_or_init(|| Self::new(n, g)),
            _ => panic!(),
        }
    }
}

/// Exact integer span filter.
pub struct Filter {
    /// Scaled integer target.
    b: Option<Vec<i128>>,
    /// Cached fixed Gram projections.
    mats: Option<&'static GramMats>,
    /// Fast target vector.
    sb: [u16; MAX],
    /// Prime-field target vectors.
    fb: [[u64; MAX]; 3],
}

impl Filter {
    /// Build exact integer filter for one Gram image.
    /// # Arguments:
    /// - `g`: Spin Gram matrix.
    /// - `y`: Target image.
    /// # Returns:
    /// - `Self`: Exact span filter.
    pub fn new(
        n: usize,
        g: &[Vec<Ratio<i64>>],
        y: &[Ratio<i64>],
    ) -> Self {
        let mats = GramMats::cached(n, g);
        let yr = mats.rows.iter().map(|&row| y[row]).collect::<Vec<_>>();
        let b = int::scale_vector(&yr);
        let sb = b
            .as_ref()
            .map(|b| field::vector_fast_u16(b))
            .unwrap_or([0; MAX]);
        let fb = b
            .as_ref()
            .map(|b| field::vectors_u64(b))
            .unwrap_or([[0; MAX]; 3]);

        Self {
            b,
            mats: Some(mats),
            sb,
            fb,
        }
    }

    /// Return selected independent rows.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `&[usize]`: Independent row indices.
    pub fn rows(&self) -> &[usize] {
        self.mats.map(|mats| mats.rows.as_slice()).unwrap_or(&[])
    }

    /// Start one incremental modular span state.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `Option<State>`: Initial state if modular data exists.
    pub fn state(&self) -> Option<State> {
        self.mats.map(|mats| State::new(mats, self.sb))
    }

    /// Test candidate support and report exact-check use.
    /// # Arguments:
    /// - `s`: Candidate support columns.
    /// # Returns:
    /// - `(bool, bool)`: Plausible result and whether exact integer rank was used.
    pub fn span_stats(
        &self,
        s: &[usize],
    ) -> (bool, bool) {
        let Some(b) = &self.b else {
            return (true, false);
        };
        let Some(mats) = self.mats else {
            return (true, false);
        };

        if !field::spans_u64(mats.n, &mats.f, &self.fb, s) {
            return (false, false);
        }

        (
            int::selected_columns_span_target(&mats.a, b, s).unwrap_or(true),
            true,
        )
    }
}

/// One incremental prime-field span state.
pub struct State {
    /// Matrix dimension.
    n: usize,
    /// Selected support length.
    len: usize,
    /// Current rank.
    rank: usize,
    /// Pivot rows.
    piv: [usize; MAX],
    /// Normalized echelon rows.
    rows: [[u16; MAX]; MAX],
    /// Reduced target vector modulo `p`.
    y: [u16; MAX],
}

impl State {
    /// Build one empty state.
    /// # Arguments:
    /// - `f`: Prime-field projection.
    /// # Returns:
    /// - `Self`: Empty span state.
    fn new(
        mats: &'static GramMats,
        y: [u16; MAX],
    ) -> Self {
        Self {
            n: mats.n,
            len: 0,
            rank: 0,
            piv: [0; MAX],
            rows: [[0; MAX]; MAX],
            y,
        }
    }

    /// Push one selected column.
    /// # Arguments:
    /// - `f`: Exact filter owning modular columns.
    /// - `col`: Selected column index.
    /// # Returns:
    /// - `Mark`: Previous state marker.
    pub fn push(
        &mut self,
        f: &Filter,
        col: usize,
    ) -> Mark {
        let mark = Mark {
            rank: self.rank,
            y: self.y,
        };
        self.len += 1;

        let Some(mats) = f.mats else {
            return mark;
        };

        let mut x = mats.s[col];
        self.red(&mut x);

        let Some(piv) = (0..self.n).find(|&i| x[i] != 0) else {
            return mark;
        };

        let q = field::invert_fast_u16(x[piv]);
        for item in x.iter_mut().take(self.n) {
            *item = field::multiply_fast_u16(*item, q);
        }

        let q = self.y[piv];
        if q != 0 {
            for (j, &xj) in x.iter().enumerate().take(self.n).skip(piv) {
                let z = field::multiply_fast_u16(q, xj);
                self.y[j] = field::subtract_fast_u16(self.y[j], z);
            }
        }

        self.piv[self.rank] = piv;
        self.rows[self.rank] = x;
        self.rank += 1;
        mark
    }

    /// Pop one selected column.
    /// # Arguments:
    /// - `mark`: Previous state marker returned by `push`.
    /// # Returns:
    /// - `()`: Restores this state to the previous depth.
    pub fn pop(
        &mut self,
        mark: Mark,
    ) {
        self.len -= 1;
        self.rank = mark.rank;
        self.y = mark.y;
    }

    /// Test whether target can still be spanned.
    /// # Arguments:
    /// - None.
    /// # Returns:
    /// - `bool`: False only when exact span is impossible.
    pub fn span(&self) -> bool {
        if self.rank != self.len {
            return true;
        }

        self.y[..self.n].iter().all(|&x| x == 0)
    }

    /// Reduce one vector by this state.
    /// # Arguments:
    /// - `x`: Vector scratch.
    /// # Returns:
    /// - `()`: Updates `x`.
    fn red(
        &self,
        x: &mut [u16; MAX],
    ) {
        for i in 0..self.rank {
            let piv = self.piv[i];
            let q = x[piv];
            if q == 0 {
                continue;
            }

            for (j, xj) in x.iter_mut().enumerate().take(self.n).skip(piv) {
                let y = field::multiply_fast_u16(q, self.rows[i][j]);
                *xj = field::subtract_fast_u16(*xj, y);
            }
        }
    }
}

/// Build the spin Gram matrix from permutations.
/// # Arguments:
/// - `ps`: Permutations.
/// # Returns:
/// - `Vec<Vec<Ratio<i64>>>`: Gram matrix.
fn build_gram(ps: &[Vec<usize>]) -> Vec<Vec<Ratio<i64>>> {
    ps.iter()
        .map(|p| {
            ps.iter()
                .map(|q| {
                    let r = permutation_compose(&permutation_inverse(p), q);
                    Ratio::<i64>::from_integer(
                        permutation_sign(p)
                            * permutation_sign(q)
                            * (1_i64 << permutation_cycles(&r)),
                    )
                })
                .collect()
        })
        .collect()
}

/// Find deterministic independent rows.
/// # Arguments:
/// - `g`: Spin Gram matrix.
/// # Returns:
/// - `Vec<usize>`: Independent row indices.
fn independent_rows(g: &[Vec<Ratio<i64>>]) -> Vec<usize> {
    let mut rows = Vec::<usize>::new();
    let mut rank = 0;

    for (i, row) in g.iter().enumerate() {
        let mut a = rows.iter().map(|&j| g[j].clone()).collect::<Vec<_>>();
        a.push(row.clone());
        let next = rational_rank(a);

        if next > rank {
            rows.push(i);
            rank = next;
        }
    }

    rows
}

/// Find deterministic independent reduced columns.
/// # Arguments:
/// - `g`: Spin Gram matrix.
/// - `rows`: Independent row indices.
/// # Returns:
/// - `Vec<usize>`: Independent column indices.
fn independent_cols(
    g: &[Vec<Ratio<i64>>],
    rows: &[usize],
) -> Vec<usize> {
    let mut cols = Vec::<usize>::new();
    let mut rank = 0;

    for col in 0..g.len() {
        let mut a = vec![vec![Ratio::<i64>::zero(); cols.len() + 1]; rows.len()];

        for (i, &row) in rows.iter().enumerate() {
            for (j, &old) in cols.iter().enumerate() {
                a[i][j] = g[row][old];
            }

            a[i][cols.len()] = g[row][col];
        }

        let next = rational_rank(a);
        if next > rank {
            cols.push(col);
            rank = next;
        }
    }

    cols
}

/// Compute exact rational rank.
/// # Arguments:
/// - `a`: Matrix.
/// # Returns:
/// - `usize`: Rank.
fn rational_rank(mut a: Vec<Vec<Ratio<i64>>>) -> usize {
    let rows = a.len();
    if rows == 0 {
        return 0;
    }

    let cols = a[0].len();
    let mut rank = 0;

    for col in 0..cols {
        let Some(piv) = (rank..rows).find(|&row| !a[row][col].is_zero()) else {
            continue;
        };

        a.swap(rank, piv);
        let q = a[rank][col];
        for j in col..cols {
            a[rank][j] /= q;
        }

        for row in 0..rows {
            if row == rank {
                continue;
            }

            let q = a[row][col];
            if q.is_zero() {
                continue;
            }

            for j in col..cols {
                let x = q * a[rank][j];
                a[row][j] -= x;
            }
        }

        rank += 1;
        if rank == rows {
            break;
        }
    }

    rank
}

/// Solve a rational linear system if consistent.
/// # Arguments:
/// - `a`: Matrix.
/// - `b`: Right-hand side.
/// # Returns:
/// - `Option<Vec<Ratio<i64>>>`: Solution with free variables set to zero.
pub fn solve_system(
    mut a: Vec<Vec<Ratio<i64>>>,
    b: Vec<Ratio<i64>>,
) -> Option<Vec<Ratio<i64>>> {
    let rows = a.len();

    if rows == 0 {
        return Some(Vec::new());
    }

    let cols = a[0].len();

    for (r, x) in b.into_iter().enumerate() {
        a[r].push(x);
    }

    let mut piv = Vec::new();
    let mut row = 0;

    for col in 0..cols {
        let Some(p) = (row..rows).find(|&r| !a[r][col].is_zero()) else {
            continue;
        };

        a.swap(row, p);

        let q = a[row][col];
        for j in col..=cols {
            a[row][j] /= q;
        }

        for r in 0..rows {
            if r == row {
                continue;
            }

            let q = a[r][col];

            if q.is_zero() {
                continue;
            }

            for j in col..=cols {
                let x = q * a[row][j];
                a[r][j] -= x;
            }
        }

        piv.push(col);
        row += 1;

        if row == rows {
            break;
        }
    }

    for item in a.iter().take(rows).skip(row) {
        if item[..cols].iter().all(|x| x.is_zero()) && !item[cols].is_zero() {
            return None;
        }
    }

    let mut x = vec![Ratio::<i64>::zero(); cols];

    for (r, col) in piv.into_iter().enumerate() {
        x[col] = a[r][cols];
    }

    Some(x)
}

/// Solve a selected-column Gram system.
/// # Arguments:
/// - `g`: Spin Gram matrix.
/// - `b`: Row-reduced right-hand side.
/// - `s`: Selected Gram columns.
/// - `rows`: Selected Gram rows.
/// # Returns:
/// - `Option<Vec<Ratio<i64>>>`: Solution with free variables set to zero.
pub fn solve_rows(
    g: &[Vec<Ratio<i64>>],
    b: &[Ratio<i64>],
    s: &[usize],
    rows: &[usize],
) -> Option<Vec<Ratio<i64>>> {
    match s.len() {
        0 => Some(Vec::new()),
        1 => solven::<1>(g, b, s, rows),
        2 => solven::<2>(g, b, s, rows),
        3 => solven::<3>(g, b, s, rows),
        4 => solven::<4>(g, b, s, rows),
        5 => solven::<5>(g, b, s, rows),
        6 => solven::<6>(g, b, s, rows),
        7 => solven::<7>(g, b, s, rows),
        8 => solven::<8>(g, b, s, rows),
        9 => solven::<9>(g, b, s, rows),
        10 => solven::<10>(g, b, s, rows),
        11 => solven::<11>(g, b, s, rows),
        12 => solven::<12>(g, b, s, rows),
        13 => solven::<13>(g, b, s, rows),
        14 => solven::<14>(g, b, s, rows),
        _ => solve_selected(g, b, s, rows),
    }
}

/// Solve one fixed-width selected-column Gram system.
/// # Arguments:
/// - `g`: Spin Gram matrix.
/// - `b`: Right-hand side.
/// - `s`: Selected Gram columns.
/// # Returns:
/// - `Option<Vec<Ratio<i64>>>`: Exact solution if consistent.
fn solven<const C: usize>(
    g: &[Vec<Ratio<i64>>],
    b: &[Ratio<i64>],
    s: &[usize],
    rows: &[usize],
) -> Option<Vec<Ratio<i64>>> {
    let nrows = rows.len();
    let mut a = rows
        .iter()
        .enumerate()
        .map(|(i, &r)| (std::array::from_fn(|c| g[r][s[c]]), b[i]))
        .collect::<Vec<([Ratio<i64>; C], Ratio<i64>)>>();

    let mut piv = Vec::new();
    let mut row = 0;

    for col in 0..C {
        let Some(p) = (row..nrows).find(|&r| !a[r].0[col].is_zero()) else {
            continue;
        };

        a.swap(row, p);

        let q = a[row].0[col];
        for j in col..C {
            a[row].0[j] /= q;
        }
        a[row].1 /= q;

        for r in 0..nrows {
            if r == row {
                continue;
            }

            let q = a[r].0[col];

            if q.is_zero() {
                continue;
            }

            for j in col..C {
                let x = q * a[row].0[j];
                a[r].0[j] -= x;
            }

            let x = q * a[row].1;
            a[r].1 -= x;
        }

        piv.push(col);
        row += 1;

        if row == nrows {
            break;
        }
    }

    for item in a.iter().take(nrows).skip(row) {
        if item.0.iter().all(|x| x.is_zero()) && !item.1.is_zero() {
            return None;
        }
    }

    let mut x = std::array::from_fn::<_, C, _>(|_| Ratio::<i64>::zero());

    for (r, col) in piv.into_iter().enumerate() {
        x[col] = a[r].1;
    }

    Some(x.into_iter().collect())
}

/// Solve one dynamically sized selected-column Gram system.
/// # Arguments:
/// - `g`: Spin Gram matrix.
/// - `b`: Right-hand side.
/// - `s`: Selected Gram columns.
/// # Returns:
/// - `Option<Vec<Ratio<i64>>>`: Exact solution if consistent.
fn solve_selected(
    g: &[Vec<Ratio<i64>>],
    b: &[Ratio<i64>],
    s: &[usize],
    rows: &[usize],
) -> Option<Vec<Ratio<i64>>> {
    let a = rows
        .iter()
        .map(|&row| s.iter().map(|&column| g[row][column]).collect::<Vec<_>>())
        .collect::<Vec<_>>();

    solve_system(a, b.to_vec())
}

/// Project coefficients through the spin Gram matrix.
/// # Arguments:
/// - `ps`: Permutations.
/// - `g`: Gram matrix.
/// - `cs`: Coefficients.
/// # Returns:
/// - `Vec<Ratio<i64>>`: Projected vector.
pub fn gram_image(
    ps: &[Vec<usize>],
    g: &[Vec<Ratio<i64>>],
    cs: &BTreeMap<Vec<usize>, Ratio<i64>>,
) -> Vec<Ratio<i64>> {
    (0..ps.len())
        .map(|i| {
            let mut x = Ratio::<i64>::zero();

            for (j, p) in ps.iter().enumerate() {
                if let Some(c) = cs.get(p) {
                    x += g[i][j] * *c;
                }
            }

            x
        })
        .collect()
}

/// Build deterministic basis-column solution.
/// # Arguments:
/// - `n`: Cumulant rank.
/// - `g`: Spin Gram matrix.
/// - `y`: Full target image.
/// # Returns:
/// - `Option<(Vec<usize>, Vec<Ratio<i64>>)>`: Basis columns and coefficients.
pub fn basis_solution(
    n: usize,
    g: &[Vec<Ratio<i64>>],
    y: &[Ratio<i64>],
) -> Option<(Vec<usize>, Vec<Ratio<i64>>)> {
    let data = GramBasis::cached(n)?;
    let basis_cols = data.basis.as_slice();
    let yr = data.rows.iter().map(|&row| y[row]).collect::<Vec<_>>();
    let z = solve_selected(g, &yr, basis_cols, &data.rows)?;

    Some((basis_cols.to_vec(), z))
}

/// Store all Gram columns restricted to the selected independent rows
/// as exact integers.
/// # Arguments:
/// - `g`: Spin Gram matrix.
/// - `rows`: Independent row indices.
/// # Returns:
/// - `Vec<[i128; MAX]>`: Integer Gram columns.
fn integer_columns(
    g: &[Vec<Ratio<i64>>],
    rows: &[usize],
) -> Vec<[i128; MAX]> {
    let mut a = vec![[0; MAX]; g.len()];

    for (i, &row) in rows.iter().enumerate() {
        for (j, x) in g[row].iter().enumerate() {
            a[j][i] = *x.numer() as i128;
        }
    }

    a
}

/// Return permutation sign. This is determined by the inversion
/// count of the permutation and returns:
///     \sign(p) = (-1)^{N_{\text{inv}}}.
/// # Arguments:
/// - `p`: Permutation.
/// # Returns:
/// - `i64`: Sign.
pub fn permutation_sign(p: &[usize]) -> i64 {
    let n = (0..p.len())
        .cartesian_product(0..p.len())
        .filter(|&(i, j)| i < j && p[i] > p[j])
        .count();

    if n % 2 == 0 { 1 } else { -1 }
}

/// Return inverse permutation. For a permutation p and an index i this
/// returns the permutation defined by:
///     p^{-1}(p(i)) = i.
/// # Arguments:
/// - `p`: Permutation.
/// # Returns:
/// - `Vec<usize>`: Inverse permutation.
pub fn permutation_inverse(p: &[usize]) -> Vec<usize> {
    let mut out = vec![0; p.len()];

    for (i, &x) in p.iter().enumerate() {
        out[x] = i;
    }

    out
}

/// Compose two permutations. For permutations p, q and an index i this
/// returns the composite permutation which performs p(q(i)).
/// # Arguments:
/// - `p`: Outer permutation.
/// - `q`: Inner permutation.
/// # Returns:
/// - `Vec<usize>`: Composition.
pub fn permutation_compose(
    p: &[usize],
    q: &[usize],
) -> Vec<usize> {
    (0..p.len()).map(|i| p[q[i]]).collect()
}

/// Count number of cycles in a permutation. Starting from each unseen
/// element i we follow the cycle:
///     i ---> p(i) ---> p(p(i)) ---> ...
/// until a previously visited element is arrived at. Each traversal
/// is a new cycle which is used in the Gram matrix expression:
///     G_{\pi \rho} = \sign(\pi) \sign(\rho) 2^{c(\pi^{-1} \rho)}.
/// # Arguments:
/// - `p`: Permutation.
/// # Returns:
/// - `usize`: Number of cycles.
pub fn permutation_cycles(p: &[usize]) -> usize {
    let mut seen = vec![false; p.len()];
    let mut out = 0;

    for i in 0..p.len() {
        if seen[i] {
            continue;
        }

        out += 1;
        let mut j = i;

        while !seen[j] {
            seen[j] = true;
            j = p[j];
        }
    }

    out
}
