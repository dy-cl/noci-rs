// noci/factorise/contracted.rs
//! Contraction form of the same-spin one-body and overlap factors between a pair of references,
//! the same one or two different ones, for determinants with at most two replacements. The ket occupied orbitals are made biorthogonal
//! to the bra ones, both virtual spaces are reduced to their remainders outside the occupied
//! spaces, and the operator acts by the orthogonal one-body rules, so a whole amplitude vector
//! is applied by matrix products over orbital indices rather than element by element. Near-zero
//! occupied overlaps are handled exactly by deformation and extrapolation. The NOCI-PT2 one-body
//! operator and the QMC overlap products share it.

// Standard library imports.
use std::collections::HashMap;

// External crate imports.
use ndarray::linalg::general_mat_mul;
use ndarray::{Array1, Array2, Array3, ArrayView2, Axis, Zip, s};
use ndarray_linalg::{Determinant, Inverse};
use rayon::prelude::*;

// Crate-root imports.
use crate::NOCIScalar;
use crate::determinant::Bits;
use crate::elements::nonorthogonalwicks::{SINGULARTOL, SameSpinBuild};
use crate::elements::{FockData, NOCIData};
use crate::maths::adjoint;

// Parent/sibling imports.
use super::{SpinFactorisation, StringSlot};

/// Largest same-spin replacement rank applied in contraction form.
pub(super) const CONTRACTRANK: usize = 2;
/// Extrapolation points `\epsilon` for deformed near-zero occupied overlaps.
const EPSNODES: [f64; 6] = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0];
/// Minimum number of target determinants per Rayon task when one spin string's products are split.
pub(super) const DOTCHUNK: usize = 64;
/// Class-3 columns applied together as one batch per Rayon task.
const COLCHUNK: usize = 64;

/// Parent pair `QP`, same-parent or not, applied in contraction form: every same-spin factor is
/// applied to whole amplitude vectors through orbital-indexed matrices, never element by element.
pub(super) struct ContractedPair<T: NOCIScalar> {
    /// Target parent `Q`, the bra.
    pub(super) target_parent: usize,
    /// Source parent `P`, the ket.
    pub(super) source_parent: usize,
    /// Alpha extrapolation points of `F^\alpha` and `S^\alpha`.
    alpha: Vec<ContractionPoint<T>>,
    /// Beta extrapolation points of `F^\beta` and `S^\beta`.
    beta: Vec<ContractionPoint<T>>,
    /// Whether the pair is a same-parent pair of an orthonormal parent, whose biorthogonal frame
    /// is the identity, `G = I`, `\alpha = \beta = 0` and `Q = I`, and whose overlap is
    /// `S^{PP} = I` on the parent's strings.
    identity: bool,
    /// `F^\alpha` and `S^\alpha` applied to the ket alpha reference, at every target alpha string.
    reference_alpha: (Vec<T>, Vec<T>),
    /// `F^\beta` and `S^\beta` applied to the ket beta reference, at every target beta string.
    reference_beta: (Vec<T>, Vec<T>),
    /// Target alpha double replacements grouped by packed hole pair `p = j(j-1)/2 + i` of
    /// `i < j`, each with its string index and particle positions `a < b`.
    doubles_alpha: Vec<Vec<(usize, usize, usize)>>,
    /// Target beta double replacements grouped as `doubles_alpha`.
    doubles_beta: Vec<Vec<(usize, usize, usize)>>,
}

/// Same-spin intermediates of one parent pair at one extrapolation point `\epsilon`. Near-zero
/// occupied overlaps are deformed to `\hat S_k + \epsilon`, which makes every factor a polynomial
/// in `\epsilon`, so the factors at `\epsilon = 0` are a weighted sum over a few points. The bra
/// occupied orbitals are `\phi_i`; the ket occupied orbitals are made biorthogonal to them,
/// `\tilde\phi = C^x_o G`; and both virtual spaces are reduced to their remainders outside the
/// occupied spaces, `\tilde v_a` and `\hat q_c`, biorthogonal to each other.
struct ContractionPoint<T: NOCIScalar> {
    /// Lagrange weight of the point times `\det M_0`, with `M_0 = C^{w\dagger}_o S C^x_o`.
    scale: T,
    /// `G = M_0^{-1}`.
    g: Array2<T>,
    /// `\alpha_{kb} = \phi_k^\dagger S u_b`, the ket virtual `u_b` along `\tilde\phi_k`.
    alpha: Array2<T>,
    /// `\beta_{ai} = v_a^\dagger S \tilde\phi_i`, the bra virtual `v_a` along `\phi_i`.
    beta: Array2<T>,
    /// `Q_{ab} = \tilde v_a^\dagger S q_b`, the overlap of the virtual remainders.
    q: Array2<T>,
    /// One-body operator in the biorthogonal frame, absent for overlap-only products, with
    /// occupied-occupied block
    /// `f_{ik} = \phi_i^\dagger F \tilde\phi_k`, occupied-virtual block
    /// `f_{ic} = \phi_i^\dagger F \hat q_c`, virtual-occupied block
    /// `f_{ak} = \tilde v_a^\dagger F \tilde\phi_k` and virtual-virtual block
    /// `f_{ac} = \tilde v_a^\dagger F \hat q_c`.
    f: Option<Array2<T>>,
}

/// Same-spin amplitudes over the strings of one parent with at most two replacements, as
/// in-place replacements of occupied positions by virtual positions, for a batch of `N` vectors
/// applied together: `c_0`, `c_{ia}` and the antisymmetric `c_{ijab}`, held only for a single
/// vector and absent when every double is zero. Applied amplitudes hold their doubles only at
/// the target strings.
struct StringAmplitudes<R: NOCIScalar> {
    /// Reference amplitudes `c^n_0`, one per vector.
    c0: Array1<R>,
    /// Single replacement amplitudes `c^n_{ia}`, indexed `[i, n, a]` so that products over the
    /// occupied or the virtual index act on the whole batch at once.
    c1: Array3<R>,
    /// Double replacement amplitudes `c_{ijab}` of a single vector, indexed `[a, b, p]` by the
    /// virtual pair and the packed occupied pair `p = j(j-1)/2 + i` of `i < j`.
    c2: Option<Array3<R>>,
    /// Applied double replacement amplitudes of a single vector at every target string, by string
    /// index, empty when not requested.
    d2: Vec<R>,
}

impl<T: NOCIScalar> ContractedPair<T> {
    /// Prepare the contraction form of one ordered parent pair `QP`, when `contractible` holds and
    /// every near-zero occupied overlap can be extrapolated.
    /// # Arguments:
    /// - `parents`: Target parent `Q` and source parent `P`.
    /// - `spin`: Shared determinant-space factorisation.
    /// - `data`: Shared NOCI data holding the parent orbitals, the AO overlap and the zero-overlap
    ///   tolerance.
    /// - `fock`: Current generalised Fock matrices in the AO basis, `None` for overlap products.
    /// # Returns
    /// - `Option<Self>`: The contracted block, or `None` when the pair keeps the factor rows.
    pub(super) fn new(
        parents: (usize, usize),
        spin: &SpinFactorisation,
        data: &NOCIData<'_, T>,
        fock: Option<&FockData<'_, T>>,
    ) -> Option<Self> {
        if !contractible(spin, parents) {
            return None;
        }
        let (target_parent, source_parent) = parents;
        let (bra, ket) = (
            &data.space.parents()[target_parent],
            &data.space.parents()[source_parent],
        );

        let alpha = ContractionPoint::build(
            (&bra.ca, bra.oa),
            (&ket.ca, ket.oa),
            (&data.ao.s, fock.map(|f| f.fa)),
            data.tol,
        )?;
        let beta = ContractionPoint::build(
            (&bra.cb, bra.ob),
            (&ket.cb, ket.ob),
            (&data.ao.s, fock.map(|f| f.fb)),
            data.tol,
        )?;

        // A same-parent pair of an orthonormal parent has the identity frame.
        let identity = target_parent == source_parent
            && data
                .mocache
                .is_some_and(|mocache| mocache[target_parent].orthogonal_slater_condon);

        let n = ket.ca.ncols();
        let target = &spin.parents[target_parent];

        // Target double replacements of each spin grouped by packed hole pair, so each occupied
        // pair of an apply reads its own strings.
        let doubles = |occupation: Bits, slots: &[StringSlot]| {
            let nocc = occupation.count_ones() as usize;
            let mut grouped = vec![Vec::new(); nocc * nocc.saturating_sub(1) / 2];
            for (k, slot) in slots.iter().enumerate().filter(|(_, slot)| slot.rank == 2) {
                let ([i, j], [a, b]) = (slot.holes, slot.particles);
                grouped[j * (j - 1) / 2 + i].push((k, a, b));
            }

            grouped
        };
        let doubles_alpha = doubles(bra.oa, &target.aslots);
        let doubles_beta = doubles(bra.ob, &target.bslots);

        // The factors of each reference string do not depend on the amplitudes, so they are
        // applied once here and kept at every target string.
        let reference = |points: &[ContractionPoint<T>],
                         slots: &[StringSlot],
                         grouped: &[Vec<(usize, usize, usize)>],
                         source_occ: Bits| {
            // The ket reference string alone, `c_0 = 1`, through every point.
            let nocc = source_occ.count_ones() as usize;
            let mut r = StringAmplitudes::zeros((nocc, n - nocc), 1, false);
            r.c0[0] = T::one();
            let (f, s) = ContractionPoint::apply(points, &r, true, grouped);

            // `F c_0` and `S c_0` at every target string.
            let at = |amp: &StringAmplitudes<T>| {
                slots
                    .iter()
                    .enumerate()
                    .map(|(k, slot)| amp.read(slot, 0, k))
                    .collect::<Vec<_>>()
            };
            (at(&f), at(&s))
        };

        let reference_alpha = reference(&alpha, &target.aslots, &doubles_alpha, ket.oa);
        let reference_beta = reference(&beta, &target.bslots, &doubles_beta, ket.ob);

        Some(Self {
            target_parent,
            source_parent,
            alpha,
            beta,
            identity,
            reference_alpha,
            reference_beta,
            doubles_alpha,
            doubles_beta,
        })
    }

    /// Contributions to `Y^Q = (F + \lambda S)^{QP} x^P` of this parent pair, with
    /// `F^{QP} = F^\alpha D (S^\beta)^T + S^\alpha D (F^\beta)^T` and
    /// `S^{QP} = S^\alpha D (S^\beta)^T` for the source amplitudes `D_{ab}` over alpha and beta
    /// strings, whose determinants have at most two replacements in total. In the identity frame
    /// an overlap-only product is `S^{PP} = I` on the parent's strings and reduces to occupation
    /// matching. Otherwise `D` is split by excitation class and, by linearity, no string-by-string
    /// factor block is formed:
    /// 1. the beta-reference column `D_{a0}`: one alpha apply, times the beta factors of the
    ///    beta reference;
    /// 2. the alpha-reference row `D_{0b}` over excited beta strings: one beta apply, times the
    ///    alpha factors of the alpha reference;
    /// 3. the alpha-single by beta-single block `D^{11}`, with columns `c` over beta singles.
    ///    Targets with the beta reference receive the alpha applies of `D^{11} w` with
    ///    `w_c = S^\beta_{0c}` or `F^\beta_{0c}`; targets with the alpha reference the beta
    ///    applies of the reference rows `(F^\alpha D^{11})_{0c}` and `(S^\alpha D^{11})_{0c}`;
    ///    single-single targets the product of the single rows of `F^\alpha D^{11}` and
    ///    `S^\alpha D^{11}` with those of `S^\beta` and `F^\beta`.
    /// # Arguments:
    /// - `spin`: Shared determinant-space factorisation.
    /// - `data`: Shared NOCI data holding the parent occupations.
    /// - `sources`: Source alpha and beta string IDs with their amplitude.
    /// - `targets`: Target output index with its alpha and beta string IDs.
    /// - `lambda`: Scalar overlap shift.
    /// # Returns
    /// - `Vec<(usize, R)>`: Target output index and the value to add to it.
    pub(super) fn apply_pair<R>(
        &self,
        spin: &SpinFactorisation,
        data: &NOCIData<'_, T>,
        sources: &[(usize, usize, R)],
        targets: &[(usize, usize, usize)],
        lambda: R,
    ) -> Vec<(usize, R)>
    where
        R: NOCIScalar + From<T>,
    {
        // In the identity frame the overlap is `S^{PP} = I` on the parent's strings, so an
        // overlap-only product reduces to occupation matching: every target receives the source
        // amplitudes with the same alpha and beta strings. The string phases cancel in pairs.
        if self.identity && self.alpha[0].f.is_none() {
            let mut matched: HashMap<(usize, usize), R> = HashMap::new();
            for &(a, b, value) in sources {
                *matched.entry((a, b)).or_insert(R::from_real(0.0)) += value;
            }
            return targets
                .par_iter()
                .with_min_len(DOTCHUNK)
                .filter_map(|&(out, a, b)| matched.get(&(a, b)).map(|&value| (out, lambda * value)))
                .collect();
        }

        let (source, target) = (
            &spin.parents[self.source_parent],
            &spin.parents[self.target_parent],
        );
        let ket = &data.space.parents()[self.source_parent];

        // Occupied and virtual orbital counts of each spin.
        let n = ket.ca.ncols();
        let (na, nb) = (ket.oa.count_ones() as usize, ket.ob.count_ones() as usize);
        let (adims, bdims) = ((na, n - na), (nb, n - nb));

        // Doubles are applied only at the target strings of each spin.
        let (alpha_targets, beta_targets) = (&self.doubles_alpha, &self.doubles_beta);

        // Amplitudes of a single vector over the source strings of one spin.
        let amplitudes =
            |dims: (usize, usize), slots: &[StringSlot], values: Vec<(usize, R)>, doubles: bool| {
                let values = values
                    .into_iter()
                    .map(|(id, value)| (0, id, value))
                    .collect();
                StringAmplitudes::gather(dims, slots, 1, values, doubles)
            };

        // One pass over the sources splits them by class: the beta-reference column, the
        // alpha-reference row over excited beta strings, and the single-single block grouped by
        // beta single in string order, so every sum runs in the same order on every call.
        let mut column = Vec::new();
        let mut row = Vec::new();
        let mut grouped: Vec<Vec<(usize, R)>> = vec![Vec::new(); source.bslots.len()];
        for &(a, b, value) in sources {
            match (source.aslots[a].rank, source.bslots[b].rank) {
                (_, 0) => column.push((a, value)),
                (0, _) => row.push((b, value)),
                (1, 1) => grouped[b].push((a, value)),
                _ => {}
            }
        }
        let grouped = grouped
            .into_iter()
            .enumerate()
            .filter(|(_, entries)| !entries.is_empty())
            .collect::<Vec<_>>();
        let ncol = grouped.len();

        // Class 1: the beta-reference column, with the beta factors of the beta reference, which
        // are kept from construction.
        let class1 = || {
            let c = amplitudes(adims, &source.aslots, column, true);
            ContractionPoint::apply(&self.alpha, &c, true, alpha_targets)
        };

        // Class 2: the alpha-reference row over excited beta strings, with the alpha factors of
        // the alpha reference.
        let class2 = || {
            let c = amplitudes(bdims, &source.bslots, row, true);
            ContractionPoint::apply(&self.beta, &c, true, beta_targets)
        };

        // Class 3: one column of `D^{11}` per beta single. Each column needs only the reference and
        // single-replacement outputs of its alpha apply and of the beta factors of its single, so
        // every column is applied in one batch per spin. Chunks of `COLCHUNK` columns run
        // concurrently, column `c` of a chunk as its vector `c`.
        let chunks = if grouped.is_empty() {
            vec![&grouped[..]]
        } else {
            grouped.chunks(COLCHUNK).collect::<Vec<_>>()
        };
        let class3 = || {
            chunks
                .par_iter()
                .map(|chunk| {
                    // Alpha amplitudes: the alpha singles of every column.
                    let values = chunk
                        .iter()
                        .enumerate()
                        .flat_map(|(c, (_, column))| {
                            column.iter().map(move |&(a, value)| (c, a, value))
                        })
                        .collect();
                    let alpha =
                        StringAmplitudes::gather(adims, &source.aslots, chunk.len(), values, false);

                    // Beta amplitudes: the beta single of every column, with unit amplitude.
                    let values = chunk
                        .iter()
                        .enumerate()
                        .map(|(c, (b, _))| (c, *b, R::one()))
                        .collect();
                    let beta =
                        StringAmplitudes::gather(bdims, &source.bslots, chunk.len(), values, false);

                    // One batched apply per spin, reference and single outputs only.
                    let alpha = ContractionPoint::apply(&self.alpha, &alpha, true, &[]);
                    let beta = ContractionPoint::apply(&self.beta, &beta, true, &[]);

                    (alpha, beta)
                })
                .collect::<Vec<_>>()
        };

        // The three classes are independent, so they run concurrently, letting idle threads take
        // part of a long pair.
        let (((fa1, sa1), (fb2, sb2)), applied) =
            rayon::join(|| rayon::join(class1, class2), class3);

        // The class-3 outputs of every chunk, by spin and operator.
        let mut fa3 = Vec::with_capacity(applied.len());
        let mut sa3 = Vec::with_capacity(applied.len());
        let mut fb3 = Vec::with_capacity(applied.len());
        let mut sb3 = Vec::with_capacity(applied.len());
        for ((fa, sa), (fb, sb)) in applied {
            fa3.push(fa);
            sa3.push(sa);
            fb3.push(fb);
            sb3.push(sb);
        }

        // Reference output of column `c`, vector `c % COLCHUNK` of chunk `c / COLCHUNK`, read in
        // place.
        let reference =
            |chunks: &[StringAmplitudes<R>], c: usize| chunks[c / COLCHUNK].c0[c % COLCHUNK];

        // Beta-reference targets: alpha applies of `D^{11} w^S` and `D^{11} w^F`.
        let weighted = |w: &(dyn Fn(usize) -> R + Sync)| {
            let values = grouped
                .iter()
                .enumerate()
                .flat_map(|(c, (_, column))| {
                    let wc = w(c);
                    column.iter().map(move |&(a, value)| (a, value * wc))
                })
                .collect();
            amplitudes(adims, &source.aslots, values, false)
        };

        // Alpha-reference targets: beta applies of the reference rows of `F^\alpha D^{11}` and
        // `S^\alpha D^{11}`.
        let reference_row = |pick: &(dyn Fn(usize) -> R + Sync)| {
            let values = grouped
                .iter()
                .enumerate()
                .map(|(c, (b, _))| (*b, pick(c)))
                .collect();
            amplitudes(bdims, &source.bslots, values, false)
        };

        // The four applies are independent and run concurrently.
        let (((fws, sws), (_, swf)), ((_, suf), (fus, sus))) = rayon::join(
            || {
                rayon::join(
                    || {
                        let w = weighted(&|c| reference(&sb3, c));
                        ContractionPoint::apply(&self.alpha, &w, true, alpha_targets)
                    },
                    || {
                        let w = weighted(&|c| reference(&fb3, c));
                        ContractionPoint::apply(&self.alpha, &w, false, alpha_targets)
                    },
                )
            },
            || {
                rayon::join(
                    || {
                        let row = reference_row(&|c| reference(&fa3, c));
                        ContractionPoint::apply(&self.beta, &row, false, beta_targets)
                    },
                    || {
                        let row = reference_row(&|c| reference(&sa3, c));
                        ContractionPoint::apply(&self.beta, &row, true, beta_targets)
                    },
                )
            },
        );

        // Single-single targets: `Y^F = X^F (S^\beta)^T + X^S (F^\beta)^T` and
        // `Y^S = X^S (S^\beta)^T`, with one row per target single and one column per beta single.
        // `rows` maps every target string to its row, `usize::MAX` for strings that are not
        // singles.
        let singles = |slots: &[StringSlot], chunks: &[StringAmplitudes<R>]| {
            let ids = (0..slots.len())
                .filter(|&k| slots[k].rank == 1)
                .collect::<Vec<_>>();
            let mut rows = vec![usize::MAX; slots.len()];
            for (r, &k) in ids.iter().enumerate() {
                rows[k] = r;
            }

            // Each matrix is filled as its transpose, one column at a time, so the reads run along
            // the stored singles of one column.
            let mt = Array2::from_shape_fn((ncol, ids.len()), |(c, r)| {
                chunks[c / COLCHUNK].read(&slots[ids[r]], c % COLCHUNK, ids[r])
            });
            (mt, rows)
        };

        // The single rows of `F^\alpha D^{11}`, `S^\alpha D^{11}`, `F^\beta` and `S^\beta`.
        let (xf, arows) = singles(&target.aslots, &fa3);
        let (xs, _) = singles(&target.aslots, &sa3);
        let (bf, brows) = singles(&target.bslots, &fb3);
        let (bs, _) = singles(&target.bslots, &sb3);

        // Both products, side by side for each target pair so one read serves both.
        let yf = xf.t().dot(&bs) + xs.t().dot(&bf);
        let ys = xs.t().dot(&bs);
        let y = Array2::from_shape_fn(yf.dim(), |(r, c)| [yf[(r, c)], ys[(r, c)]]);

        // Every class at the target strings, packed per string so a target reads one record per
        // spin: alpha `[F_1, S_1, F_0, S_0, F_{ws}, S_{ws}, S_{wf}]` and beta
        // `[F_0, S_0, F_2, S_2, S_{uf}, F_{us}, S_{us}]`.
        let cast = |x: T| <R as From<T>>::from(x);

        let alpha = (0..target.aslots.len())
            .map(|k| {
                let read = |amp: &StringAmplitudes<R>| amp.read(&target.aslots[k], 0, k);
                [
                    read(&fa1),
                    read(&sa1),
                    cast(self.reference_alpha.0[k]),
                    cast(self.reference_alpha.1[k]),
                    read(&fws),
                    read(&sws),
                    read(&swf),
                ]
            })
            .collect::<Vec<_>>();

        let beta = (0..target.bslots.len())
            .map(|k| {
                let read = |amp: &StringAmplitudes<R>| amp.read(&target.bslots[k], 0, k);
                [
                    cast(self.reference_beta.0[k]),
                    cast(self.reference_beta.1[k]),
                    read(&fb2),
                    read(&sb2),
                    read(&suf),
                    read(&fus),
                    read(&sus),
                ]
            })
            .collect::<Vec<_>>();

        // `Y^Q_{\bar a\bar b} = F_{\bar a\bar b} + \lambda S_{\bar a\bar b}` at every target
        // determinant: classes 1 and 2 everywhere, class 3 by target class.
        targets
            .par_iter()
            .with_min_len(DOTCHUNK)
            .map(|&(out, i, j)| {
                let [fa1, sa1, fa0, sa0, fws, sws, swf] = alpha[i];
                let [fb0, sb0, fb2, sb2, suf, fus, sus] = beta[j];

                // Classes 1 and 2.
                let mut f = fa1 * sb0 + sa1 * fb0 + fa0 * sb2 + sa0 * fb2;
                let mut s = sa1 * sb0 + sa0 * sb2;

                // Class 3 by target class.
                match (target.aslots[i].rank, target.bslots[j].rank) {
                    (_, 0) => {
                        f += fws + swf;
                        s += sws;
                    }
                    (0, _) => {
                        f += suf + fus;
                        s += sus;
                    }
                    (1, 1) => {
                        let [yf, ys] = y[(arows[i], brows[j])];
                        f += yf;
                        s += ys;
                    }
                    _ => {}
                }

                (out, f + lambda * s)
            })
            .collect()
    }
}

impl<T: NOCIScalar> ContractionPoint<T> {
    /// Prepare the extrapolation points of the same-spin factors between the strings of a bra
    /// reference `w` and a ket reference `x`. With the Löwdin pairing
    /// `M_0 = U \tilde S V^\dagger`, every pair with `\tilde S_k < \mathtt{SINGULARTOL}` is
    /// deformed along `(C^w_o u_k) v_k^\dagger` to `\hat S_k + \epsilon`, where `\hat S_k = 0` when
    /// `\tilde S_k \le \mathtt{tol}`, as in the Wick kernels, and `\hat S_k = \tilde S_k`
    /// otherwise. Every factor is then a polynomial of degree at most the number of deformed pairs
    /// in `\epsilon`, and the factors at `\epsilon = 0` follow exactly from Lagrange weights at
    /// well-conditioned points.
    /// # Arguments:
    /// - `bra`: Bra orbitals and occupation of this spin.
    /// - `ket`: Ket orbitals and occupation of this spin.
    /// - `operators`: AO overlap matrix and, unless only overlaps are needed, the one-body
    ///   operator of this spin.
    /// - `tol`: Singular values at or below this are exact zeros.
    /// # Returns
    /// - `Option<Vec<Self>>`: The points, or `None` when the pair cannot be extrapolated.
    fn build(
        bra: (&Array2<T>, Bits),
        ket: (&Array2<T>, Bits),
        operators: (&Array2<f64>, Option<&Array2<T>>),
        tol: f64,
    ) -> Option<Vec<Self>> {
        let (s, f) = operators;

        // Occupied and virtual orbitals of both references, each in ascending orbital order.
        let split = |c: &Array2<T>, occupation: Bits| {
            let n = c.ncols();
            let pick = |occupied: bool| {
                let columns = (0..n)
                    .filter(|&k| occupation.contains(k) == occupied)
                    .collect::<Vec<_>>();
                c.select(Axis(1), &columns)
            };
            (pick(true), pick(false))
        };
        let (wo, wv) = split(bra.0, bra.1);
        let (xo, xv) = split(ket.0, ket.1);
        let o = wo.ncols();
        if xo.ncols() != o || o == 0 || wv.ncols() == 0 {
            return None;
        }

        // Löwdin pairing of the occupied overlap, `C^w_o U` and `C^x_o V`, and
        // `V^\dagger = (C^x_o V)^\dagger S C^x_o` for orthonormal ket orbitals.
        let (sv, x_paired, w_paired, _) =
            SameSpinBuild::perform_ortho_and_svd_and_rotate(s, &wo, &xo);
        let st = s.mapv(T::from_real);
        let vh = adjoint(&x_paired).dot(&st).dot(&xo);
        let small = (0..o).filter(|&k| sv[k] < SINGULARTOL).collect::<Vec<_>>();
        if small.len() >= EPSNODES.len() {
            return None;
        }

        // Occupied ket orbitals at `\epsilon`: `C^x_o + B + \epsilon E`, where `B` moves each
        // deformed pair to `\hat S_k` and `E` lifts it by `\epsilon`.
        let mut base = Array2::<T>::zeros(xo.raw_dim());
        let mut lift = Array2::<T>::zeros(xo.raw_dim());
        for &k in &small {
            let (a, b) = (w_paired.column(k), vh.row(k));
            let outer = Array2::from_shape_fn(xo.raw_dim(), |(mu, j)| a[mu] * b[j]);
            let target = if sv[k] <= tol { 0.0 } else { sv[k] };

            base.scaled_add(T::from_real(target - sv[k]), &outer);
            lift += &outer;
        }

        // One point per extrapolation node, weighted by its Lagrange weight at `\epsilon = 0`.
        let points: &[f64] = if small.is_empty() {
            &[0.0]
        } else {
            &EPSNODES[..=small.len()]
        };
        points
            .iter()
            .map(|&e| {
                let weight = points
                    .iter()
                    .filter(|&&p| p != e)
                    .map(|&p| p / (p - e))
                    .product::<f64>();
                let xo = &xo + &base + &lift.mapv(|z| z * T::from_real(e));
                Self::new((&wo, &wv), (&xo, &xv), (&st, f), weight)
            })
            .collect()
    }

    /// Form the intermediates of one point, for invertible `M_0` and `Q`. The ket occupied
    /// orbitals are made biorthogonal to the bra ones, `\tilde\phi = C^x_o G`; the virtual
    /// orbitals keep their remainders outside the occupied spaces, `q_b = (1 - P) u_b` with
    /// `P = \tilde\phi C^{w\dagger}_o S`, and `\tilde v_a = v_a - C^w_o \beta_a^*`; and the ket
    /// remainders are made biorthogonal to the bra ones, `\hat q = q Q^{-1}`.
    /// # Arguments:
    /// - `bra`: Bra occupied and virtual orbitals.
    /// - `ket`: Ket occupied and virtual orbitals.
    /// - `operators`: AO overlap matrix and, when present, the one-body operator.
    /// - `weight`: Lagrange weight of this point.
    /// # Returns
    /// - `Option<Self>`: The intermediates, or `None` for a singular `M_0` or `Q`.
    fn new(
        bra: (&Array2<T>, &Array2<T>),
        ket: (&Array2<T>, &Array2<T>),
        operators: (&Array2<T>, Option<&Array2<T>>),
        weight: f64,
    ) -> Option<Self> {
        let ((wo, wv), (xo, xv), (s, f)) = (bra, ket, operators);
        let (o, v) = (wo.ncols(), wv.ncols());
        let woh = adjoint(wo);

        // Biorthogonal ket occupied orbitals, `\tilde\phi = C^x_o M_0^{-1}`.
        let m0 = woh.dot(s).dot(xo);
        let d = m0.det().ok()?;
        let g = m0.inv().ok()?;
        let bt = xo.dot(&g);

        // Occupied components and virtual remainders on both sides.
        let alpha = woh.dot(s).dot(xv);
        let beta = adjoint(wv).dot(s).dot(&bt);
        let q_orb = xv - &bt.dot(&alpha);
        let vth = adjoint(&(wv - &wo.dot(&adjoint(&beta))));

        // Biorthogonal ket virtual remainders, `\hat q = q Q^{-1}`.
        let q = vth.dot(s).dot(&q_orb);
        let qh = q_orb.dot(&q.inv().ok()?);

        // One-body operator in the biorthogonal frame, blocks `[[f_oo, f_ov], [f_vo, f_vv]]`.
        let f = f.map(|f| {
            let mut fb = Array2::<T>::zeros((o + v, o + v));
            fb.slice_mut(s![..o, ..o]).assign(&woh.dot(f).dot(&bt));
            fb.slice_mut(s![..o, o..]).assign(&woh.dot(f).dot(&qh));
            fb.slice_mut(s![o.., ..o]).assign(&vth.dot(f).dot(&bt));
            fb.slice_mut(s![o.., o..]).assign(&vth.dot(f).dot(&qh));
            fb
        });

        Some(Self {
            scale: d * T::from_real(weight),
            g,
            alpha,
            beta,
            q,
            f,
        })
    }

    /// Apply the same-spin factors of every point to a batch of amplitude vectors over the ket
    /// strings, returning `F c` and `S c` with their reference and single replacements, and their
    /// double replacements at the requested target strings. `F c` stays zero for overlap-only
    /// points. At each point the ket amplitudes are carried into the biorthogonal frame,
    /// `t_{kb} = \sum_j G_{jk} c_{jb}` and `t_{klab} = \sum_{ij} G_{ik} G_{jl} c_{ijab}`; their
    /// virtual orbitals split as `u_b = \tilde\phi\alpha_b + \sum_c \hat q_c Q_{cb}`; the operator
    /// acts by the orthogonal one-body rules with `f`; and the bra virtual orbitals split as
    /// `v_a = \tilde v_a + \phi\beta_a^*`. Every product over an occupied or a virtual index acts on
    /// the whole batch at once; doubles are carried only for a single vector. The doubles take two
    /// passes, because every product over occupied indices commutes with every product over
    /// virtual indices. Pass A acts on the occupied pair of every virtual pair as one product over
    /// the packed pairs `p = (k < l)`: `t = G \otimes G` on `c` and, for `F c`, the occupied part
    /// of `\sigma`. Pass B takes one occupied pair at a time and applies `Q \otimes Q`, the
    /// virtual part of `\sigma` and the rank raise in one `v \times v` tile, read at once at the
    /// target strings with those holes, so no double-replacement output is stored whole.
    /// # Arguments:
    /// - `points`: Extrapolation points of one spin and parent pair.
    /// - `c`: Ket string amplitudes of every vector.
    /// - `fock`: Whether `F c` is needed; when false it stays zero.
    /// - `doubles`: Target double replacements grouped by packed hole pair, each with its string
    ///   index and particle positions; empty when no double-replacement output is needed.
    /// # Returns
    /// - `(StringAmplitudes<R>, StringAmplitudes<R>)`: `F c` and `S c`.
    fn apply<R>(
        points: &[Self],
        c: &StringAmplitudes<R>,
        fock: bool,
        doubles: &[Vec<(usize, usize, usize)>],
    ) -> (StringAmplitudes<R>, StringAmplitudes<R>)
    where
        R: NOCIScalar + From<T>,
    {
        let (o, nb, v) = c.c1.dim();
        let zero = R::from_real(0.0);
        let one = R::one();
        let half = R::from_real(0.5);

        // Occupied pairs `k < l` in packed order `p = l(l-1)/2 + k`.
        let pairs = (0..o)
            .flat_map(|l| (0..l).map(move |k| (k, l)))
            .collect::<Vec<_>>();
        let npair = pairs.len();

        // Outputs, with their doubles indexed by target string up to the last double. Each group
        // lists its strings in ascending order, so the last double ends one of the groups.
        let ntarget = doubles
            .iter()
            .filter_map(|group| group.last())
            .map(|&(k, _, _)| k + 1)
            .max()
            .unwrap_or(0);

        let mut sf = StringAmplitudes::zeros((o, v), nb, false);
        let mut ss = StringAmplitudes::zeros((o, v), nb, false);
        sf.d2 = vec![zero; ntarget];
        ss.d2 = vec![zero; ntarget];

        // `\sum_{kb} x_{knb} m_{kb}` for every vector `n`.
        let traced = |x: &Array3<R>, m: &Array2<R>| {
            Array1::from_shape_fn(nb, |n| {
                x.index_axis(Axis(1), n)
                    .iter()
                    .zip(m.iter())
                    .fold(zero, |acc, (&xi, &mi)| acc + xi * mi)
            })
        };

        // `x_{knb} \leftarrow \sum_j m_{jk} x_{jnb}` as one product over the batch.
        let left = |m: &Array2<R>, x: &Array3<R>| {
            m.t()
                .dot(&x.view().into_shape((o, nb * v)).unwrap())
                .into_shape((o, nb, v))
                .unwrap()
        };

        // `x_{knb} \leftarrow \sum_c x_{knc} m_{bc}` as one product over the batch.
        let right = |x: &Array3<R>, m: &Array2<R>| {
            x.view()
                .into_shape((o * nb, v))
                .unwrap()
                .dot(&m.t())
                .into_shape((o, nb, v))
                .unwrap()
        };

        // The unit antisymmetric occupied matrix `E_p` of pair `p = (k, l)`, `+1` at `(k, l)` and
        // `-1` at `(l, k)`, so an antisymmetric `x = \sum_p x_p E_p`.
        let unit = |p: usize| {
            let (k, l) = pairs[p];
            let mut e = Array2::from_elem((o, o), zero);
            e[(k, l)] = one;
            e[(l, k)] = -one;

            e
        };

        // The packed coefficients `x_p = x_{kl}` of an antisymmetric occupied matrix.
        let pack = |x: &Array2<R>| Array1::from_iter(pairs.iter().map(|&(k, l)| x[(k, l)]));

        // `P(ij)P(ab)\, x_{ia} y_{bj} = x_{ia}y_{bj} - x_{ja}y_{bi} - x_{ib}y_{aj} + x_{jb}y_{ai}`.
        let outer =
            |x: ArrayView2<R>, y: &Array2<R>, (i, j): (usize, usize), (a, b): (usize, usize)| {
                x[(i, a)] * y[(b, j)] - x[(j, a)] * y[(b, i)] - x[(i, b)] * y[(a, j)]
                    + x[(j, b)] * y[(a, i)]
            };

        for point in points {
            let cast = |a: ArrayView2<T>| a.mapv(<R as From<T>>::from);
            let (g, alpha, beta, q) = (
                cast(point.g.view()),
                cast(point.alpha.view()),
                cast(point.beta.view()),
                cast(point.q.view()),
            );

            // Operator blocks, for points with a one-body operator when `F c` is needed.
            let blocks = point.f.as_ref().filter(|_| fock).map(|f| {
                let f_oo = cast(f.slice(s![..o, ..o]));
                let f_ov = cast(f.slice(s![..o, o..]));
                let f_vo = cast(f.slice(s![o.., ..o]));
                let f_vv = cast(f.slice(s![o.., o..]));
                let e0 = f_oo.diag().sum();

                (f_oo, f_ov, f_vo, f_vv, e0)
            });

            // Ket singles: `G` on the occupied index, `t_{knb} = \sum_j G_{jk} c_{jnb}`.
            let mut t1 = left(&g, &c.c1);

            // Pass A, as products over the packed occupied pair of every virtual pair `ab`. First
            // `t_{ab} = c_{ab} D` with `D_{qp} = (G^T E_q G)_p`, which is
            // `t_{klab} = \sum_{ij} G_{ik} G_{jl} c_{ijab}`.
            let doubles_in = c.c2.as_ref().filter(|_| npair > 0);
            let t2 = doubles_in.map(|c2| {
                let mut d = Array2::from_elem((npair, npair), zero);
                for p in 0..npair {
                    d.row_mut(p).assign(&pack(&g.t().dot(&unit(p)).dot(&g)));
                }

                c2.view().into_shape((v * v, npair)).unwrap().dot(&d)
            });

            // Then, for `F c`, `m_{ab} = t_{ab} M` with
            // `M_{qp} = (e_0 E_q - E_q f_{oo} + (E_q f_{oo})^T)_p`, the
            // `e_0 r_{ijab} - P(ij) \sum_k f_{kj} r_{ikab}` part of `\sigma`, which commutes with
            // `Q`.
            let m2 = t2
                .as_ref()
                .zip(blocks.as_ref())
                .map(|(t2, (f_oo, _, _, _, e0))| {
                    let mut m = Array2::from_elem((npair, npair), zero);
                    for p in 0..npair {
                        let ef = unit(p).dot(f_oo);
                        m.row_mut(p)
                            .assign(&pack(&(unit(p).mapv(|x| x * *e0) - &ef + ef.t())));
                    }

                    t2.dot(&m).into_shape((v, v, npair)).unwrap()
                });

            // Both indexed `[a, b, p]`, so pass B reads one `v \times v` tile per pair.
            let t2 = t2.map(|t2| t2.into_shape((v, v, npair)).unwrap());

            // `z_{ka} = \sum_{lb} t_{klab}\alpha_{lb}` and, for `F c`, the contraction of
            // `r_{klcd}` with `f_{ov}` through `\tilde f = f_{ov} Q`, as one product
            // `\sum_{bp} t_{abp} (E_p x_b)_k` with `x = \alpha` and `x = \tilde f`.
            let contracted = t2.as_ref().map(|t2| {
                let f_tilde = blocks.as_ref().map(|(_, f_ov, _, _, _)| f_ov.dot(&q));
                let width = if f_tilde.is_some() { 2 * o } else { o };

                // Row `(b, p)` holds `E_p x_b`: `x_{lb}` at `k` and `-x_{kb}` at `l`.
                let mut e = Array2::from_elem((v * npair, width), zero);
                for b in 0..v {
                    for (p, &(k, l)) in pairs.iter().enumerate() {
                        let mut row = e.row_mut(b * npair + p);
                        row[k] = alpha[(l, b)];
                        row[l] = -alpha[(k, b)];
                        if let Some(ft) = &f_tilde {
                            row[o + k] = ft[(l, b)];
                            row[o + l] = -ft[(k, b)];
                        }
                    }
                }

                t2.view().into_shape((v, v * npair)).unwrap().dot(&e)
            });

            // Ket: lower the rank through `\alpha`, then carry the virtual index through `Q`,
            // `r_{knd} = \sum_b (t_{knb} + z_{kb}) Q_{db}`, with `z` for the single vector.
            let mut r0 = &c.c0 + &traced(&t1, &alpha);
            if let Some(zy) = &contracted {
                let z = zy.slice(s![.., ..o]).t().to_owned();
                r0[0] += half * (&z * &alpha).sum();
                t1.index_axis_mut(Axis(1), 0).scaled_add(one, &z);
            }

            let r1 = right(&t1, &q);

            // One-body sigma by the orthogonal rules: the one-body Slater-Condon rules of
            // `elements::orthogonal`, contracted over whole amplitude arrays rather than evaluated
            // per determinant pair, which would visit the `O(ov)` connections of every string.
            // `\sigma^n_0 = e_0 r^n_0 + \sum_{kc} f_{kc} r_{knc}`,
            // `\sigma_{ina} = f_{ai} r^n_0 + e_0 r_{ina} + \sum_c f_{ac} r_{inc}
            //  - \sum_k f_{ki} r_{kna} + \sum_{kc} f_{kc} r_{ikac}`, the last term from pass A as
            // `\sum_{a'} Q_{aa'} \sum_{kb} t_{ika'b} \tilde f_{kb}`.
            let sigma = blocks.as_ref().map(|(f_oo, f_ov, f_vo, f_vv, e0)| {
                let s0 = r0.mapv(|z| *e0 * z) + traced(&r1, f_ov);
                let mut s1 = Array3::from_shape_fn((o, nb, v), |(i, n, a)| f_vo[(a, i)] * r0[n])
                    + r1.mapv(|z| z * *e0)
                    + right(&r1, f_vv)
                    - left(f_oo, &r1);

                if let Some(zy) = &contracted {
                    let y = zy.slice(s![.., o..]).t().dot(&q.t());
                    s1.index_axis_mut(Axis(1), 0).scaled_add(one, &y);
                }

                (s0, s1)
            });

            // Bra: raise the rank through `\beta`, for `F c` from `\sigma` and for `S c` from `r`,
            // `w^n_0 = s^n_0` and `w_{ina} = s_{ina} + \beta_{ai} s^n_0`, weighted by the point's
            // Lagrange weight.
            let scale = <R as From<T>>::from(point.scale);
            let raise = |out: &mut StringAmplitudes<R>, x0: &Array1<R>, x1: &Array3<R>| {
                out.c0.scaled_add(scale, x0);
                Zip::indexed(&mut out.c1)
                    .and(x1)
                    .for_each(|(i, n, a), c, &x| *c += scale * (x + beta[(a, i)] * x0[n]));
            };

            if let Some((s0, s1)) = &sigma {
                raise(&mut sf, s0, s1);
            }
            raise(&mut ss, &r0, &r1);

            if doubles.is_empty() {
                continue;
            }

            // Pass B, one hole pair `ij` at a time: `r_{ij} = Q t_{ij} Q^T`, and for `F c`
            // `\sigma_{ij} = Q m_{ij} Q^T + V - V^T + P(ij)P(ab) r_{ia} f_{bj}` with
            // `V = r_{ij} f_{vv}^T`. Each tile is raised,
            // `w_{ijab} = x_{ijab} + P(ij)P(ab) x_{ia}\beta_{bj}
            //  + x_0 (\beta_{ai}\beta_{bj} - \beta_{aj}\beta_{bi})`,
            // and read at the target strings with holes `ij`.
            let r1 = r1.index_axis(Axis(1), 0);
            let s01 = sigma
                .as_ref()
                .map(|(s0, s1)| (s0[0], s1.index_axis(Axis(1), 0)));

            // Tiles reused for every pair.
            let mut half_tile = Array2::from_elem((v, v), zero);
            let mut r = Array2::from_elem((v, v), zero);
            let mut qm = Array2::from_elem((v, v), zero);
            let mut vv = Array2::from_elem((v, v), zero);
            for (p, group) in doubles.iter().enumerate() {
                if group.is_empty() {
                    continue;
                }

                let (i, j) = pairs[p];

                // The tiles, zero when the ket has no doubles.
                if let Some(t2) = &t2 {
                    let tp = t2.slice(s![.., .., p]);
                    general_mat_mul(one, &q, &tp, zero, &mut half_tile);
                    general_mat_mul(one, &half_tile, &q.t(), zero, &mut r);
                }
                if let (Some(m2), Some((_, _, _, f_vv, _))) = (&m2, &blocks) {
                    let mp = m2.slice(s![.., .., p]);
                    general_mat_mul(one, &q, &mp, zero, &mut half_tile);
                    general_mat_mul(one, &half_tile, &q.t(), zero, &mut qm);
                    general_mat_mul(one, &r, &f_vv.t(), zero, &mut vv);
                }

                // Raised tiles at the target strings.
                for &(k, a, b) in group {
                    let pair = beta[(a, i)] * beta[(b, j)] - beta[(a, j)] * beta[(b, i)];
                    let ws = r[(a, b)] + outer(r1, &beta, (i, j), (a, b)) + r0[0] * pair;
                    ss.d2[k] += scale * ws;

                    if let (Some((s0, s1)), Some((_, _, f_vo, _, _))) = (&s01, &blocks) {
                        let wf = qm[(a, b)] + vv[(a, b)] - vv[(b, a)]
                            + outer(r1, f_vo, (i, j), (a, b))
                            + outer(s1.view(), &beta, (i, j), (a, b))
                            + *s0 * pair;
                        sf.d2[k] += scale * wf;
                    }
                }
            }
        }

        (sf, ss)
    }
}

impl<R: NOCIScalar> StringAmplitudes<R> {
    /// Zero amplitudes for a batch of vectors, with room for doubles when requested.
    /// # Arguments:
    /// - `dims`: Numbers of occupied and virtual orbitals.
    /// - `batch`: Number of vectors `N`.
    /// - `doubles`: Whether double replacements are held, for a single vector only.
    /// # Returns
    /// - `Self`: Zero amplitudes.
    fn zeros(
        dims: (usize, usize),
        batch: usize,
        doubles: bool,
    ) -> Self {
        let (o, v) = dims;
        let zero = R::from_real(0.0);
        let npair = o * o.saturating_sub(1) / 2;

        Self {
            c0: Array1::from_elem(batch, zero),
            c1: Array3::from_elem((o, batch, v), zero),
            c2: doubles.then(|| Array3::from_elem((v, v, npair), zero)),
            d2: Vec::new(),
        }
    }

    /// Collect amplitudes given per vector and parent-local string ID. The determinant of a string
    /// differs from the in-place replacements of its parent reference by the string's phase.
    /// # Arguments:
    /// - `dims`: Numbers of occupied and virtual orbitals.
    /// - `slots`: Slots of the parent's spin strings.
    /// - `batch`: Number of vectors `N`.
    /// - `values`: Vector index, string ID and amplitude.
    /// - `doubles`: Whether double replacements may be present, for a single vector only.
    /// # Returns
    /// - `Self`: Amplitudes in tensor form.
    fn gather(
        dims: (usize, usize),
        slots: &[StringSlot],
        batch: usize,
        values: Vec<(usize, usize, R)>,
        doubles: bool,
    ) -> Self {
        let mut out = Self::zeros(dims, batch, doubles);

        for (k, id, value) in values {
            let slot = &slots[id];
            let value = value * R::from_real(slot.phase);
            let ([i, j], [a, b]) = (slot.holes, slot.particles);
            match slot.rank {
                0 => out.c0[k] += value,
                1 => out.c1[(i, k, a)] += value,
                _ => {
                    let c2 = out
                        .c2
                        .as_mut()
                        .expect("doubles requested for double replacements");
                    let p = j * (j - 1) / 2 + i;
                    c2[(a, b, p)] += value;
                    c2[(b, a, p)] -= value;
                }
            }
        }

        out
    }

    /// Read the applied amplitude of one target string of one vector in the determinant convention
    /// of the spin strings. Double replacements are read from the target-string doubles.
    /// # Arguments:
    /// - `slot`: Slot of the string to read.
    /// - `k`: Vector in the batch.
    /// - `id`: Its index among the target strings.
    /// # Returns
    /// - `R`: Its amplitude.
    fn read(
        &self,
        slot: &StringSlot,
        k: usize,
        id: usize,
    ) -> R {
        let (i, a) = (slot.holes[0], slot.particles[0]);
        let value = match slot.rank {
            0 => self.c0[k],
            1 => self.c1[(i, k, a)],
            _ => self.d2[id],
        };

        value * R::from_real(slot.phase)
    }
}

/// Whether an ordered parent pair `QP` can be applied in contraction form: both parents' determinants
/// have at most `CONTRACTRANK` replacements in total. A same-parent pair is the special case whose
/// frame is the parent's own: for orthonormal orbitals `M_0 = I`, so `G = I`, `\alpha = \beta = 0`,
/// `Q = I` and a single point, and the orthogonal one-body rules apply directly.
/// # Arguments:
/// - `spin`: Shared determinant-space factorisation.
/// - `parents`: Target parent `Q` and source parent `P`.
/// # Returns
/// - `bool`: Whether the class decomposition of the contraction form covers the pair.
pub(super) fn contractible(
    spin: &SpinFactorisation,
    parents: (usize, usize),
) -> bool {
    let (target_parent, source_parent) = parents;
    let within = |p: usize| {
        let parent = &spin.parents[p];
        parent
            .entries
            .iter()
            .all(|e| parent.aslots[e.a].rank + parent.bslots[e.b].rank <= CONTRACTRANK)
    };
    within(target_parent) && within(source_parent)
}
