// canon/graph.rs

// External crate imports.
use smallvec::SmallVec;

/// Maximum number of search-tree leaves explored before the search stops branching.
/// Terms are small, so this only guards against pathological symmetric inputs.
const LEAFMAX: usize = 1 << 16;

/// Vertex-coloured undirected graph.
#[derive(Clone, Debug, Default)]
pub(crate) struct Graph {
    /// Vertex colours; only their relative order is significant.
    pub(crate) colors: Vec<u32>,
    /// Sorted adjacency lists.
    pub(crate) adj: Vec<SmallVec<[u32; 4]>>,
}

/// Canonical labelling of one graph.
#[derive(Clone, Debug)]
pub(crate) struct Canon {
    /// Canonical position of every vertex.
    pub(crate) label: Vec<u32>,
    /// Every labelling that attains the canonical certificate, including `label`.
    /// Two such labellings differ by a graph automorphism.
    pub(crate) automorphs: Vec<Vec<u32>>,
}

impl Graph {
    /// Add one vertex with a colour.
    /// # Arguments:
    /// - `color`: Vertex colour.
    /// # Returns:
    /// - `u32`: New vertex id.
    pub(crate) fn vertex(
        &mut self,
        color: u32,
    ) -> u32 {
        self.colors.push(color);
        self.adj.push(SmallVec::new());
        (self.colors.len() - 1) as u32
    }

    /// Add one undirected edge.
    /// # Arguments:
    /// - `a`: First vertex.
    /// - `b`: Second vertex.
    /// # Returns:
    /// - `()`: Mutates the adjacency lists.
    pub(crate) fn edge(
        &mut self,
        a: u32,
        b: u32,
    ) {
        self.adj[a as usize].push(b);
        self.adj[b as usize].push(a);
    }
}

/// Return the canonical labelling of a vertex-coloured graph.
/// Individualisation–refinement: the colouring is refined to an equitable partition; while
/// cells remain non-singleton, every vertex of the first smallest cell is individualised in
/// turn and refined again. Each discrete leaf yields a certificate (colours and edges under the
/// labelling), and the lexicographically smallest certificate defines the canonical form.
/// # Arguments:
/// - `g`: Graph with sorted colour classes of arbitrary values.
/// # Returns:
/// - `Canon`: Canonical labelling and all labellings attaining it.
pub(crate) fn canonical(g: &Graph) -> Canon {
    // Rank the input colours densely so cells are ordered by colour value.
    let mut values = g.colors.clone();
    values.sort_unstable();
    values.dedup();
    let initial = g
        .colors
        .iter()
        .map(|c| values.binary_search(c).unwrap_or(0) as u32)
        .collect::<Vec<_>>();

    let mut search = Search {
        g,
        best: None,
        automorphs: Vec::new(),
        leaves: 0,
    };

    search.descend(refine(g, initial));

    let (_, label) = search.best.unwrap_or_default();

    Canon {
        automorphs: if search.automorphs.is_empty() {
            vec![label.clone()]
        } else {
            search.automorphs
        },
        label,
    }
}

/// Individualisation–refinement search state.
struct Search<'a> {
    /// Graph being labelled.
    g: &'a Graph,
    /// Best certificate and labelling found so far.
    best: Option<(Vec<u32>, Vec<u32>)>,
    /// All labellings attaining the best certificate.
    automorphs: Vec<Vec<u32>>,
    /// Number of leaves visited.
    leaves: usize,
}

impl Search<'_> {
    /// Descend from one equitable colouring.
    /// # Arguments:
    /// - `colors`: Equitable colouring with cells numbered in canonical order.
    /// # Returns:
    /// - `()`: Updates the best certificate and its automorphic labellings.
    fn descend(
        &mut self,
        colors: Vec<u32>,
    ) {
        let n = colors.len();
        let ncell = colors.iter().max().map(|&c| c as usize + 1).unwrap_or(0);

        // A discrete colouring is a labelling; compare its certificate with the best.
        if ncell == n {
            self.leaves += 1;
            let cert = certificate(self.g, &colors);

            match &self.best {
                Some((best, _)) if cert > *best => {}
                Some((best, _)) if cert == *best => self.automorphs.push(colors),
                _ => {
                    self.automorphs = vec![colors.clone()];
                    self.best = Some((cert, colors));
                }
            }
            return;
        }

        // Individualise each vertex of the first smallest non-singleton cell in turn.
        let mut sizes = vec![0usize; ncell];
        for &c in &colors {
            sizes[c as usize] += 1;
        }
        let target = (0..ncell)
            .filter(|&c| sizes[c] > 1)
            .min_by_key(|&c| (sizes[c], c))
            .unwrap_or(0) as u32;

        for v in 0..n {
            if colors[v] != target {
                continue;
            }
            if self.leaves >= LEAFMAX {
                return;
            }

            // The individualised vertex takes the cell's first colour; the rest shift up.
            let split = colors
                .iter()
                .enumerate()
                .map(|(u, &c)| {
                    if c > target || (c == target && u != v) {
                        c + 1
                    } else {
                        c
                    }
                })
                .collect::<Vec<_>>();

            self.descend(refine(self.g, split));
        }
    }
}

/// Refine a colouring to the coarsest equitable partition finer than it.
/// Each pass colours vertices by `(current colour, sorted neighbour colours)` and renumbers
/// cells in that sort order, so the result depends only on graph invariants.
/// # Arguments:
/// - `g`: Graph.
/// - `colors`: Initial colouring with dense cell numbers.
/// # Returns:
/// - `Vec<u32>`: Equitable colouring with dense, canonically ordered cell numbers.
fn refine(
    g: &Graph,
    mut colors: Vec<u32>,
) -> Vec<u32> {
    let n = colors.len();

    loop {
        // Build each vertex's refinement signature.
        let mut keys = (0..n)
            .map(|v| {
                let mut nb = g.adj[v]
                    .iter()
                    .map(|&u| colors[u as usize])
                    .collect::<SmallVec<[u32; 8]>>();
                nb.sort_unstable();
                (colors[v], nb, v)
            })
            .collect::<Vec<_>>();

        keys.sort_unstable_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));

        // Renumber cells densely in signature order.
        let mut next = vec![0u32; n];
        let mut cell = 0u32;
        for (k, key) in keys.iter().enumerate() {
            if k > 0 && (key.0, &key.1) != (keys[k - 1].0, &keys[k - 1].1) {
                cell += 1;
            }
            next[key.2] = cell;
        }

        let before = colors.iter().max().copied().unwrap_or(0);
        colors = next;

        if cell == before {
            return colors;
        }
    }
}

/// Return the certificate of a discrete colouring: vertex colours followed by sorted edges,
/// both expressed in canonical positions.
/// # Arguments:
/// - `g`: Graph.
/// - `label`: Canonical position of every vertex.
/// # Returns:
/// - `Vec<u32>`: Certificate comparable lexicographically.
fn certificate(
    g: &Graph,
    label: &[u32],
) -> Vec<u32> {
    let n = label.len();
    let mut order = vec![0u32; n];
    for (v, &l) in label.iter().enumerate() {
        order[l as usize] = v as u32;
    }

    // Colours in canonical order.
    let mut out = order
        .iter()
        .map(|&v| g.colors[v as usize])
        .collect::<Vec<_>>();

    // Edges in canonical positions, each stored once as `(low, high)`.
    let mut edges = Vec::new();
    for (v, adj) in g.adj.iter().enumerate() {
        for &u in adj {
            let (a, b) = (label[v], label[u as usize]);
            if a < b {
                edges.push((a, b));
            }
        }
    }
    edges.sort_unstable();

    out.push(u32::MAX);
    for (a, b) in edges {
        out.push(a);
        out.push(b);
    }
    out
}
