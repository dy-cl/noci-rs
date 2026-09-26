// canon/form.rs

// External crate imports.
use smallvec::SmallVec;

// Parent/sibling imports.
use super::graph::{self, Graph};

/// Index slot symmetry of one tensor factor.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub(crate) enum Sym {
    /// Slots are ordered; no permutation is allowed.
    Ordered,
    /// Upper and lower index sets are each antisymmetric: permuting either set by `\pi`
    /// multiplies the factor by `\mathrm{sgn}(\pi)`.
    Antisymmetric,
    /// Columns `(upper_i, lower_i)` may be permuted simultaneously without a sign.
    Columns,
    /// Columns may be permuted, and the upper and lower index of each column exchanged,
    /// without a sign.
    Pairs,
}

/// One tensor factor over integer index ids.
#[derive(Clone, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub(crate) struct Factor {
    /// Tensor kind id; factors of different kinds never match.
    pub(crate) kind: u8,
    /// Slot symmetry.
    pub(crate) sym: Sym,
    /// Upper index ids.
    pub(crate) upper: SmallVec<[u16; 4]>,
    /// Lower index ids.
    pub(crate) lower: SmallVec<[u16; 4]>,
}

/// One term's tensor structure for canonicalisation.
#[derive(Clone, Debug)]
pub(crate) struct Form {
    /// Orbital space of every index id.
    pub(crate) spaces: Vec<u8>,
    /// Number of free (external) indices; ids `0..nfree` are free and keep their labels.
    pub(crate) nfree: usize,
    /// Tensor factors.
    pub(crate) factors: Vec<Factor>,
}

/// Canonical form of one term.
#[derive(Clone, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub(crate) struct Key {
    /// Orbital space of every dummy index, in canonical id order after the free indices.
    pub(crate) dummies: SmallVec<[u8; 16]>,
    /// Factors in canonical order with canonically ordered slots and relabelled dummies.
    pub(crate) factors: Vec<Factor>,
}

/// Graph vertex categories, packed into the top bits of a colour.
const CATINDEX: u32 = 0;
/// Tensor vertex category.
const CATTENSOR: u32 = 1;
/// Unordered slot-set vertex category.
const CATSET: u32 = 2;
/// Column vertex category.
const CATCOLUMN: u32 = 3;
/// Slot port vertex category.
const CATPORT: u32 = 4;

/// Pack one vertex colour.
/// # Arguments:
/// - `cat`: Vertex category.
/// - `kind`: Tensor kind or orbital space.
/// - `side`: Upper (`0`) or lower (`1`) side, or `2` when not applicable.
/// - `pos`: Slot position or free-index label, `0` when not applicable.
/// # Returns:
/// - `u32`: Colour value.
fn vertex_colour(
    cat: u32,
    kind: u32,
    side: u32,
    pos: u32,
) -> u32 {
    (cat << 28) | (kind << 18) | (side << 16) | pos
}

/// Return the canonical key and sign of one term.
/// The term is encoded as a coloured graph in which unordered structure is represented by
/// unordered edges: antisymmetric slot sets attach their indices to one set vertex,
/// column-symmetric factors attach interchangeable column vertices, through upper and lower
/// ports or, for pair-symmetric factors, directly, and ordered slots attach through
/// positional port vertices. Graph isomorphism is then exactly term equality
/// under relabelling of dummies and the declared slot symmetries.
/// # Arguments:
/// - `form`: Term structure.
/// # Returns:
/// - `(Key, i8)`: Canonical key and the sign relating the term to it; the sign is `0` when an
///   odd automorphism makes the term vanish identically.
pub(crate) fn canonical_key(form: &Form) -> (Key, i8) {
    // An antisymmetric set with a repeated index vanishes identically.
    for f in &form.factors {
        if f.sym == Sym::Antisymmetric
            && (has_repeated_index(&f.upper) || has_repeated_index(&f.lower))
        {
            return (
                Key {
                    dummies: SmallVec::new(),
                    factors: Vec::new(),
                },
                0,
            );
        }
    }

    // Index vertices come first so their ids equal their graph vertices.
    let mut g = Graph::default();
    for (id, &space) in form.spaces.iter().enumerate() {
        let label = if id < form.nfree { id as u32 + 1 } else { 0 };
        g.add_vertex(vertex_colour(CATINDEX, space as u32, 2, label));
    }

    // Attach every factor through structure vertices matching its symmetry.
    let mut tensors = Vec::with_capacity(form.factors.len());
    let mut columns = Vec::with_capacity(form.factors.len());
    for f in &form.factors {
        let kind = f.kind as u32;
        let t = g.add_vertex(vertex_colour(CATTENSOR, kind, 2, f.sym as u32));
        let mut cols = SmallVec::<[u32; 4]>::new();

        match f.sym {
            Sym::Antisymmetric => {
                for (side, xs) in [(0, &f.upper), (1, &f.lower)] {
                    if xs.is_empty() {
                        continue;
                    }
                    let s = g.add_vertex(vertex_colour(CATSET, kind, side, 0));
                    g.add_edge(t, s);
                    for &x in xs.iter() {
                        g.add_edge(s, x as u32);
                    }
                }
            }
            Sym::Columns => {
                for (&u, &l) in f.upper.iter().zip(&f.lower) {
                    let c = g.add_vertex(vertex_colour(CATCOLUMN, kind, 2, 0));
                    let pu = g.add_vertex(vertex_colour(CATPORT, kind, 0, 0));
                    let pl = g.add_vertex(vertex_colour(CATPORT, kind, 1, 0));
                    g.add_edge(t, c);
                    g.add_edge(c, pu);
                    g.add_edge(c, pl);
                    g.add_edge(pu, u as u32);
                    g.add_edge(pl, l as u32);
                    cols.push(c);
                }
            }
            Sym::Pairs => {
                for (&u, &l) in f.upper.iter().zip(&f.lower) {
                    let c = g.add_vertex(vertex_colour(CATCOLUMN, kind, 2, 0));
                    g.add_edge(t, c);
                    g.add_edge(c, u as u32);
                    g.add_edge(c, l as u32);
                    cols.push(c);
                }
            }
            Sym::Ordered => {
                for (side, xs) in [(0, &f.upper), (1, &f.lower)] {
                    for (pos, &x) in xs.iter().enumerate() {
                        let p = g.add_vertex(vertex_colour(CATPORT, kind, side, pos as u32 + 1));
                        g.add_edge(t, p);
                        g.add_edge(p, x as u32);
                    }
                }
            }
        }

        tensors.push(t);
        columns.push(cols);
    }

    for adj in &mut g.adj {
        adj.sort_unstable();
    }

    // Build the key from the canonical labelling; every automorphic labelling must agree on
    // the sign, otherwise the term is its own negative and vanishes.
    let canon = graph::canonical_labelling(&g);
    let (key, sign) = key_from_labelling(form, &canon.label, &tensors, &columns);

    for label in &canon.automorphs {
        if key_from_labelling(form, label, &tensors, &columns).1 != sign {
            return (key, 0);
        }
    }

    (key, sign)
}

/// Build the canonical key and sign implied by one labelling.
/// # Arguments:
/// - `form`: Term structure.
/// - `label`: Canonical position of every graph vertex.
/// - `tensors`: Tensor vertex of every factor.
/// - `columns`: Column vertices of every column-symmetric factor.
/// # Returns:
/// - `(Key, i8)`: Canonical key and sign.
fn key_from_labelling(
    form: &Form,
    label: &[u32],
    tensors: &[u32],
    columns: &[SmallVec<[u32; 4]>],
) -> (Key, i8) {
    // Dummies are renumbered by canonical position; free indices keep their ids.
    let mut dummies = (form.nfree..form.spaces.len()).collect::<Vec<_>>();
    dummies.sort_unstable_by_key(|&id| label[id]);
    let mut rename = (0..form.spaces.len() as u16).collect::<Vec<_>>();
    for (rank, &id) in dummies.iter().enumerate() {
        rename[id] = (form.nfree + rank) as u16;
    }

    // Order factors by canonical tensor position.
    let mut order = (0..form.factors.len()).collect::<Vec<_>>();
    order.sort_unstable_by_key(|&n| label[tensors[n] as usize]);

    let mut sign = 1i8;
    let mut factors = Vec::with_capacity(form.factors.len());

    for n in order {
        let f = &form.factors[n];
        let (upper, lower) = match f.sym {
            // Sort each set by canonical index position, tracking permutation parity.
            Sym::Antisymmetric => {
                let (u, pu) = sort_by_label(&f.upper, label);
                let (l, pl) = sort_by_label(&f.lower, label);
                if pu ^ pl {
                    sign = -sign;
                }
                (u, l)
            }
            // Reorder columns by canonical column position.
            Sym::Columns => {
                let mut cols = (0..f.upper.len()).collect::<SmallVec<[usize; 4]>>();
                cols.sort_unstable_by_key(|&c| label[columns[n][c] as usize]);
                (
                    cols.iter().map(|&c| f.upper[c]).collect(),
                    cols.iter().map(|&c| f.lower[c]).collect(),
                )
            }
            // Reorder columns, then place the lower-labelled index of each column upper.
            Sym::Pairs => {
                let mut cols = (0..f.upper.len()).collect::<SmallVec<[usize; 4]>>();
                cols.sort_unstable_by_key(|&c| label[columns[n][c] as usize]);
                cols.iter()
                    .map(|&c| {
                        let (u, l) = (f.upper[c], f.lower[c]);
                        if label[l as usize] < label[u as usize] {
                            (l, u)
                        } else {
                            (u, l)
                        }
                    })
                    .unzip()
            }
            Sym::Ordered => (f.upper.clone(), f.lower.clone()),
        };

        factors.push(Factor {
            kind: f.kind,
            sym: f.sym,
            upper: upper.iter().map(|&x| rename[x as usize]).collect(),
            lower: lower.iter().map(|&x| rename[x as usize]).collect(),
        });
    }

    let key = Key {
        dummies: dummies.iter().map(|&id| form.spaces[id]).collect(),
        factors,
    };

    (key, sign)
}

/// Sort indices by canonical position and report the permutation parity.
/// # Arguments:
/// - `xs`: Index ids.
/// - `label`: Canonical position of every vertex.
/// # Returns:
/// - `(SmallVec<[u16; 4]>, bool)`: Sorted ids and whether the sorting permutation is odd.
fn sort_by_label(
    xs: &[u16],
    label: &[u32],
) -> (SmallVec<[u16; 4]>, bool) {
    let mut out = SmallVec::<[u16; 4]>::from_slice(xs);
    let mut odd = false;

    // Insertion sort counts transpositions exactly.
    for i in 1..out.len() {
        let mut j = i;
        while j > 0 && label[out[j - 1] as usize] > label[out[j] as usize] {
            out.swap(j - 1, j);
            odd = !odd;
            j -= 1;
        }
    }

    (out, odd)
}

/// Test whether a slot list contains a repeated index.
/// # Arguments:
/// - `xs`: Index ids.
/// # Returns:
/// - `bool`: Whether any index appears twice.
fn has_repeated_index(xs: &[u16]) -> bool {
    xs.iter().enumerate().any(|(i, x)| xs[..i].contains(x))
}
