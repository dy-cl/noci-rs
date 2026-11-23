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
use ndarray::{Array1, Array2, Array3, Array4, ArrayView2, Axis, Zip, s};
use ndarray_linalg::{Determinant, Inverse};
use rayon::prelude::*;

// Crate-root imports.
use crate::NOCIScalar;
use crate::determinant::SpinStringData;
use crate::elements::nonorthogonalwicks::{SINGULARTOL, SameSpinBuild};
use crate::elements::{FockData, NOCIData};
use crate::maths::adjoint;

// Parent/sibling imports.
use super::SpinFactorisation;

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
/// vector and absent when every double is zero.
struct StringAmplitudes<R: NOCIScalar> {
    /// Reference amplitudes `c^n_0`, one per vector.
    c0: Array1<R>,
    /// Single replacement amplitudes `c^n_{ia}`, indexed `[i, n, a]` so that products over the
    /// occupied or the virtual index act on the whole batch at once.
    c1: Array3<R>,
    /// Double replacement amplitudes `c_{ijab}` of a single vector.
    c2: Option<Array4<R>>,
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

        // The factors of each reference string do not depend on the amplitudes, so they are
        // applied once here and kept at every target string.
        let reference = |points: &[ContractionPoint<T>],
                         strings: &[SpinStringData],
                         occupations: (u128, u128)| {
            let (source_occ, target_occ) = occupations;

            // The ket reference string alone, `c_0 = 1`, through every point.
            let nocc = source_occ.count_ones() as usize;
            let mut r = StringAmplitudes::zeros((nocc, n - nocc), 1, false);
            r.c0[0] = T::one();
            let (f, s) = ContractionPoint::apply(points, &r, true, true);

            // `F c_0` and `S c_0` at every target string.
            let at = |amp: &StringAmplitudes<T>| {
                strings
                    .iter()
                    .map(|st| amp.read(st, string_positions(target_occ, st), 0))
                    .collect::<Vec<_>>()
            };
            (at(&f), at(&s))
        };

        let reference_alpha = reference(&alpha, &target.areps, (ket.oa, bra.oa));
        let reference_beta = reference(&beta, &target.breps, (ket.ob, bra.ob));

        Some(Self {
            target_parent,
            source_parent,
            alpha,
            beta,
            identity,
            reference_alpha,
            reference_beta,
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
        let (bra, ket) = (
            &data.space.parents()[self.target_parent],
            &data.space.parents()[self.source_parent],
        );
        let n = ket.ca.ncols();

        // String ranks in small arrays, so each pass over the sources reads no string records.
        let ranks =
            |strings: &[SpinStringData]| strings.iter().map(|st| st.cache.rank).collect::<Vec<_>>();
        let (sarank, sbrank) = (ranks(&source.areps), ranks(&source.breps));
        let rank = |&(a, b, _): &(usize, usize, R)| (sarank[a], sbrank[b]);

        // Amplitudes of a single vector over the source strings of one spin.
        let amplitudes = |occupation: u128,
                          strings: &[SpinStringData],
                          values: Vec<(usize, R)>,
                          doubles: bool| {
            let values = values
                .into_iter()
                .map(|(id, value)| (0, id, value))
                .collect();
            StringAmplitudes::gather((n, occupation, strings), 1, values, doubles)
        };

        // Class 1: the beta-reference column, with the beta factors of the beta reference, which
        // are kept from construction.
        let column = sources
            .iter()
            .filter(|s| rank(s).1 == 0)
            .map(|&(a, _, value)| (a, value))
            .collect();
        let class1 = || {
            let c = amplitudes(ket.oa, &source.areps, column, true);
            ContractionPoint::apply(&self.alpha, &c, true, true)
        };

        // Class 2: the alpha-reference row over excited beta strings, with the alpha factors of
        // the alpha reference.
        let row = sources
            .iter()
            .filter(|s| rank(s).0 == 0 && rank(s).1 > 0)
            .map(|&(_, b, value)| (b, value))
            .collect();
        let class2 = || {
            let c = amplitudes(ket.ob, &source.breps, row, true);
            ContractionPoint::apply(&self.beta, &c, true, true)
        };

        // Class 3: one column of `D^{11}` per beta single. Each column needs only the reference and
        // single-replacement outputs of its alpha apply and of the beta factors of its single, so
        // every column is applied in one batch per spin.
        let mut grouped: HashMap<usize, Vec<(usize, R)>> = HashMap::new();
        for &(a, b, value) in sources.iter().filter(|s| rank(s) == (1, 1)) {
            grouped.entry(b).or_default().push((a, value));
        }
        let grouped = grouped.into_iter().collect::<Vec<_>>();
        let ncol = grouped.len();

        // Chunks of `COLCHUNK` columns run concurrently, column `c` of a chunk as its vector `c`.
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
                    let parent = (n, ket.oa, &source.areps[..]);
                    let alpha = StringAmplitudes::gather(parent, chunk.len(), values, false);

                    // Beta amplitudes: the beta single of every column, with unit amplitude.
                    let values = chunk
                        .iter()
                        .enumerate()
                        .map(|(c, (b, _))| (c, *b, R::one()))
                        .collect();
                    let parent = (n, ket.ob, &source.breps[..]);
                    let beta = StringAmplitudes::gather(parent, chunk.len(), values, false);

                    // One batched apply per spin, reference and single outputs only.
                    let alpha = ContractionPoint::apply(&self.alpha, &alpha, true, false);
                    let beta = ContractionPoint::apply(&self.beta, &beta, true, false);

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
            amplitudes(ket.oa, &source.areps, values, false)
        };

        // Alpha-reference targets: beta applies of the reference rows of `F^\alpha D^{11}` and
        // `S^\alpha D^{11}`.
        let reference_row = |pick: &(dyn Fn(usize) -> R + Sync)| {
            let values = grouped
                .iter()
                .enumerate()
                .map(|(c, (b, _))| (*b, pick(c)))
                .collect();
            amplitudes(ket.ob, &source.breps, values, false)
        };

        // The four applies are independent and run concurrently.
        let (((fws, sws), (_, swf)), ((_, suf), (fus, sus))) = rayon::join(
            || {
                rayon::join(
                    || {
                        let w = weighted(&|c| reference(&sb3, c));
                        ContractionPoint::apply(&self.alpha, &w, true, true)
                    },
                    || {
                        let w = weighted(&|c| reference(&fb3, c));
                        ContractionPoint::apply(&self.alpha, &w, false, true)
                    },
                )
            },
            || {
                rayon::join(
                    || {
                        let row = reference_row(&|c| reference(&fa3, c));
                        ContractionPoint::apply(&self.beta, &row, false, true)
                    },
                    || {
                        let row = reference_row(&|c| reference(&sa3, c));
                        ContractionPoint::apply(&self.beta, &row, true, true)
                    },
                )
            },
        );

        // Hole and particle positions of every target string, found once for every read.
        let positions = |occupation: u128, strings: &[SpinStringData]| {
            strings
                .iter()
                .map(|st| string_positions(occupation, st))
                .collect::<Vec<_>>()
        };
        let apos = positions(bra.oa, &target.areps);
        let bpos = positions(bra.ob, &target.breps);

        // Single-single targets: `Y^F = X^F (S^\beta)^T + X^S (F^\beta)^T` and
        // `Y^S = X^S (S^\beta)^T`, with one row per target single and one column per beta single.
        // `rows` maps every target string to its row, `usize::MAX` for strings that are not singles.
        let singles = |strings: &[SpinStringData],
                       at: &[([usize; 2], [usize; 2])],
                       chunks: &[StringAmplitudes<R>]| {
            let ids = (0..strings.len())
                .filter(|&k| strings[k].cache.rank == 1)
                .collect::<Vec<_>>();
            let mut rows = vec![usize::MAX; strings.len()];
            for (r, &k) in ids.iter().enumerate() {
                rows[k] = r;
            }

            // Each matrix is filled as its transpose, one column at a time, so the reads run along
            // the stored singles of one column.
            let mt = Array2::from_shape_fn((ncol, ids.len()), |(c, r)| {
                chunks[c / COLCHUNK].read(&strings[ids[r]], at[ids[r]], c % COLCHUNK)
            });
            (mt, rows)
        };

        // The single rows of `F^\alpha D^{11}`, `S^\alpha D^{11}`, `F^\beta` and `S^\beta`.
        let (xf, arows) = singles(&target.areps, &apos, &fa3);
        let (xs, _) = singles(&target.areps, &apos, &sa3);
        let (bf, brows) = singles(&target.breps, &bpos, &fb3);
        let (bs, _) = singles(&target.breps, &bpos, &sb3);

        // Both products, side by side for each target pair so one read serves both.
        let yf = xf.t().dot(&bs) + xs.t().dot(&bf);
        let ys = xs.t().dot(&bs);
        let y = Array2::from_shape_fn(yf.dim(), |(r, c)| [yf[(r, c)], ys[(r, c)]]);

        // Every class at the target strings, packed per string so a target reads one record per
        // spin: alpha `[F_1, S_1, F_0, S_0, F_{ws}, S_{ws}, S_{wf}]` and beta
        // `[F_0, S_0, F_2, S_2, S_{uf}, F_{us}, S_{us}]`.
        let at = |amp: &StringAmplitudes<R>,
                  strings: &[SpinStringData],
                  positions: &[([usize; 2], [usize; 2])],
                  k: usize| { amp.read(&strings[k], positions[k], 0) };
        let cast = |x: T| <R as From<T>>::from(x);

        let alpha = (0..target.areps.len())
            .map(|k| {
                let read = |amp| at(amp, &target.areps, &apos, k);
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

        let beta = (0..target.breps.len())
            .map(|k| {
                let read = |amp| at(amp, &target.breps, &bpos, k);
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

        // Target string ranks kept apart from the packed records.
        let (arank, brank) = (ranks(&target.areps), ranks(&target.breps));

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
                match (arank[i], brank[j]) {
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
        bra: (&Array2<T>, u128),
        ket: (&Array2<T>, u128),
        operators: (&Array2<f64>, Option<&Array2<T>>),
        tol: f64,
    ) -> Option<Vec<Self>> {
        let (s, f) = operators;

        // Occupied and virtual orbitals of both references, each in ascending orbital order.
        let split = |c: &Array2<T>, occupation: u128| {
            let n = c.ncols();
            let pick = |occupied: bool| {
                let columns = (0..n)
                    .filter(|&k| (occupation >> k & 1 == 1) == occupied)
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
    /// strings, returning `F c` and `S c` over the bra strings with at most two replacements, or
    /// one when `doubles` is false. `F c` stays zero for overlap-only points. At each point the ket
    /// amplitudes are carried into the biorthogonal frame, `t_{kb} = \sum_j G_{jk} c_{jb}` and
    /// `t_{klab} = \sum_{ij} G_{ik} G_{jl} c_{ijab}`; their virtual orbitals split as
    /// `u_b = \tilde\phi\alpha_b + \sum_c \hat q_c Q_{cb}`; the operator acts by the orthogonal
    /// one-body rules with `f`; and the bra virtual orbitals split as
    /// `v_a = \tilde v_a + \phi\beta_a^*`. Every product over an occupied or a virtual index acts on
    /// the whole batch at once; doubles are carried only for a single vector.
    /// # Arguments:
    /// - `points`: Extrapolation points of one spin and parent pair.
    /// - `c`: Ket string amplitudes of every vector.
    /// - `fock`: Whether `F c` is needed; when false it stays zero.
    /// - `doubles`: Whether double-replacement outputs are needed.
    /// # Returns
    /// - `(StringAmplitudes<R>, StringAmplitudes<R>)`: `F c` and `S c`.
    fn apply<R>(
        points: &[Self],
        c: &StringAmplitudes<R>,
        fock: bool,
        doubles: bool,
    ) -> (StringAmplitudes<R>, StringAmplitudes<R>)
    where
        R: NOCIScalar + From<T>,
    {
        let (o, nb, v) = c.c1.dim();
        let half = R::from_real(0.5);
        let zero = R::from_real(0.0);

        // The doubles outputs are created by the first point, so they need no zeroing.
        let mut sf = StringAmplitudes::zeros((o, v), nb, false);
        let mut ss = StringAmplitudes::zeros((o, v), nb, false);

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

        for point in points {
            let cast = |a: ArrayView2<T>| a.mapv(<R as From<T>>::from);
            let (g, alpha, beta, q) = (
                cast(point.g.view()),
                cast(point.alpha.view()),
                cast(point.beta.view()),
                cast(point.q.view()),
            );

            // Ket: `G` on each occupied index, `t_{knb} = \sum_j G_{jk} c_{jnb}` and
            // `t_{klab} = \sum_{ij} G_{ik} G_{jl} c_{ijab}` as one product per index.
            let mut t1 = left(&g, &c.c1);

            let t2 = c.c2.as_ref().map(|c2| {
                let first = g.t().dot(&c2.view().into_shape((o, o * v * v)).unwrap());
                let first = first.into_shape((o, o, v * v)).unwrap();
                let mut t2 = Array4::from_elem((o, o, v, v), zero);
                for k in 0..o {
                    let tk = g.t().dot(&first.index_axis(Axis(0), k));
                    t2.index_axis_mut(Axis(0), k)
                        .assign(&tk.into_shape((o, v, v)).unwrap());
                }
                t2
            });

            // Ket: lower the rank through `\alpha`, then carry each virtual index through `Q`,
            // `r_{knd} = \sum_b (t_{knb} + z_{kb}) Q_{db}` and `r_{klcd} = \sum_{ab} Q_{ca} Q_{db}
            // t_{klab}`, with `z_{ka} = \sum_{lb} t_{klab}\alpha_{lb}` for the single vector.
            let mut r0 = &c.c0 + &traced(&t1, &alpha);
            if let Some(t2) = &t2 {
                let z = contract_pair(t2, &alpha);
                r0[0] += half * (&z * &alpha).sum();
                t1.index_axis_mut(Axis(1), 0).scaled_add(R::one(), &z);
            }
            let r1 = right(&t1, &q);

            // `Q` on the second virtual index as one product, then on the first as one product per
            // occupied pair written in place, so no permuted copy of the doubles is made.
            let r2 = t2.map(|t2| {
                let once = t2.into_shape((o * o * v, v)).unwrap().dot(&q.t());
                let once = once.into_shape((o * o, v, v)).unwrap();
                let mut r2 = Array3::from_elem((o * o, v, v), zero);
                for (mut r, x) in r2.outer_iter_mut().zip(once.outer_iter()) {
                    general_mat_mul(R::one(), &q, &x, zero, &mut r);
                }
                r2.into_shape((o, o, v, v)).unwrap()
            });

            // One-body sigma by the orthogonal rules, for points with a one-body operator when
            // `F c` is needed.
            let sigma = point.f.as_ref().filter(|_| fock).map(|f| {
                let f_oo = cast(f.slice(s![..o, ..o]));
                let f_ov = cast(f.slice(s![..o, o..]));
                let f_vo = cast(f.slice(s![o.., ..o]));
                let f_vv = cast(f.slice(s![o.., o..]));
                let e0 = f_oo.diag().sum();

                // One-body sigma by the orthogonal rules: the one-body Slater-Condon rules of
                // `elements::orthogonal`, contracted over whole amplitude arrays rather than
                // evaluated per determinant pair, which would visit the `O(ov)` connections of
                // every string.
                // `\sigma^n_0 = e_0 r^n_0 + \sum_{kc} f_{kc} r_{knc}`,
                // `\sigma_{ina} = f_{ai} r^n_0 + e_0 r_{ina} + \sum_c f_{ac} r_{inc}
                //  - \sum_k f_{ki} r_{kna} + \sum_{kc} f_{kc} r_{ikac}`.
                let s0 = r0.mapv(|z| e0 * z) + traced(&r1, &f_ov);
                let mut s1 = Array3::from_shape_fn((o, nb, v), |(i, n, a)| f_vo[(a, i)] * r0[n])
                    + r1.mapv(|z| z * e0)
                    + right(&r1, &f_vv)
                    - left(&f_oo, &r1);

                if let Some(r2) = &r2 {
                    s1.index_axis_mut(Axis(1), 0)
                        .scaled_add(R::one(), &contract_pair(r2, &f_ov));
                }

                // `\sigma_{ijab} = e_0 r_{ijab} + P(ab)\sum_c f_{bc} r_{ijac}
                //  - P(ij)\sum_k f_{kj} r_{ikab} + P(ij)P(ab) f_{bj} r_{ia}`, single vector only.
                let s2 = doubles.then(|| {
                    let r1 = r1.index_axis(Axis(1), 0);
                    let mut s2 = Array4::from_shape_fn((o, o, v, v), |(i, j, a, b)| {
                        r1[(i, a)] * f_vo[(b, j)]
                            - r1[(j, a)] * f_vo[(b, i)]
                            - r1[(i, b)] * f_vo[(a, j)]
                            + r1[(j, b)] * f_vo[(a, i)]
                    });

                    if let Some(r2) = &r2 {
                        // `e_0 r_{ijab} + (vv)_{ijab} - (vv)_{ijba}` with
                        // `(vv)_{ijab} = \sum_c r_{ijac} f_{bc}`, one `v \times v` tile per
                        // occupied pair, kept in cache.
                        let mut tile = Array2::from_elem((v, v), zero);
                        for i in 0..o {
                            for j in 0..o {
                                let rij = r2.slice(s![i, j, .., ..]);
                                general_mat_mul(R::one(), &rij, &f_vv.t(), zero, &mut tile);
                                Zip::from(s2.slice_mut(s![i, j, .., ..]))
                                    .and(&rij)
                                    .and(&tile)
                                    .and(tile.t())
                                    .for_each(|x, &r, &y, &yt| *x += e0 * r + y - yt);
                            }
                        }

                        // `-(oo)_{ijab} + (oo)_{jiab}` with `(oo)_{ijab} = \sum_k f_{kj} r_{ikab}`,
                        // one slab per occupied `i`, kept in cache and added to both `s_{ij}` and
                        // `s_{ji}`.
                        let mut slab = Array2::from_elem((o, v * v), zero);
                        for i in 0..o {
                            let ri = r2.index_axis(Axis(0), i).into_shape((o, v * v)).unwrap();
                            general_mat_mul(R::one(), &f_oo.t(), &ri, zero, &mut slab);
                            for (j, row) in slab.outer_iter().enumerate() {
                                let x = row.into_shape((v, v)).unwrap();
                                s2.slice_mut(s![i, j, .., ..]).scaled_add(-R::one(), &x);
                                s2.slice_mut(s![j, i, .., ..]).scaled_add(R::one(), &x);
                            }
                        }
                    }

                    s2
                });

                (s0, s1, s2)
            });

            // Bra: raise the rank through `\beta`, for `F c` from `\sigma` and for `S c` from `r`,
            // `w_{ina} = s_{ina} + \beta_{ai} s^n_0` and, for the single vector,
            // `w_{ijab} = s_{ijab} + P(ij)P(ab) \beta_{bj} s_{ia}
            //  + s_0 (\beta_{ai}\beta_{bj} - \beta_{aj}\beta_{bi})`, weighted by the point's
            // Lagrange weight.
            let scale = <R as From<T>>::from(point.scale);

            let raise = |out: &mut StringAmplitudes<R>,
                         x0: &Array1<R>,
                         x1: &Array3<R>,
                         x2: Option<Array4<R>>| {
                out.c0.scaled_add(scale, x0);
                Zip::indexed(&mut out.c1)
                    .and(x1)
                    .for_each(|(i, n, a), c, &x| *c += scale * (x + beta[(a, i)] * x0[n]));

                if !doubles {
                    return;
                }

                // The doubles are raised in one pass, without forming the antisymmetric products:
                // in place in `x_{ijab}` at the first point, which then becomes the output, and
                // added to the output at the others.
                let (y0, x1) = (x0[0], x1.index_axis(Axis(1), 0));
                let raised = |i: usize, j: usize, a: usize, b: usize| {
                    x1[(i, a)] * beta[(b, j)]
                        - x1[(j, a)] * beta[(b, i)]
                        - x1[(i, b)] * beta[(a, j)]
                        + x1[(j, b)] * beta[(a, i)]
                        + y0 * (beta[(a, i)] * beta[(b, j)] - beta[(a, j)] * beta[(b, i)])
                };

                if let Some(out2) = out.c2.as_mut() {
                    if let Some(x2) = x2 {
                        out2.scaled_add(scale, &x2);
                    }
                    Zip::indexed(out2).for_each(|(i, j, a, b), c| *c += scale * raised(i, j, a, b));
                } else {
                    let w = match x2 {
                        Some(mut x2) => {
                            Zip::indexed(&mut x2)
                                .for_each(|(i, j, a, b), c| *c = scale * (*c + raised(i, j, a, b)));
                            x2
                        }
                        None => Array4::from_shape_fn((o, o, v, v), |(i, j, a, b)| {
                            scale * raised(i, j, a, b)
                        }),
                    };
                    out.c2 = Some(w);
                }
            };

            if let Some((s0, s1, s2)) = sigma {
                raise(&mut sf, &s0, &s1, s2);
            }
            raise(&mut ss, &r0, &r1, r2);
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
        Self {
            c0: Array1::from_elem(batch, zero),
            c1: Array3::from_elem((o, batch, v), zero),
            c2: doubles.then(|| Array4::from_elem((o, o, v, v), zero)),
        }
    }

    /// Collect amplitudes given per vector and parent-local string ID. The determinant of a string
    /// differs from the in-place replacements of its parent reference by the string's phase.
    /// # Arguments:
    /// - `parent`: Number of orbitals, parent occupation and its spin strings.
    /// - `batch`: Number of vectors `N`.
    /// - `values`: Vector index, string ID and amplitude.
    /// - `doubles`: Whether double replacements may be present, for a single vector only.
    /// # Returns
    /// - `Self`: Amplitudes in tensor form.
    fn gather(
        parent: (usize, u128, &[SpinStringData]),
        batch: usize,
        values: Vec<(usize, usize, R)>,
        doubles: bool,
    ) -> Self {
        let (n, occupation, strings) = parent;
        let nocc = occupation.count_ones() as usize;
        let mut out = Self::zeros((nocc, n - nocc), batch, doubles);

        for (k, id, value) in values {
            let st = &strings[id];
            let value = value * R::from_real(st.phase);
            let ([i, j], [a, b]) = string_positions(occupation, st);
            match st.cache.rank {
                0 => out.c0[k] += value,
                1 => out.c1[(i, k, a)] += value,
                _ => {
                    let c2 = out
                        .c2
                        .as_mut()
                        .expect("doubles requested for double replacements");
                    c2[(i, j, a, b)] += value;
                    c2[(j, i, a, b)] -= value;
                    c2[(i, j, b, a)] -= value;
                    c2[(j, i, b, a)] += value;
                }
            }
        }

        out
    }

    /// Read the amplitude of one string of one vector in the determinant convention of the spin
    /// strings. Double replacements read zero when no doubles are held.
    /// # Arguments:
    /// - `st`: String to read.
    /// - `positions`: Its hole and particle positions, from `string_positions`.
    /// - `k`: Vector in the batch.
    /// # Returns
    /// - `R`: Its amplitude.
    fn read(
        &self,
        st: &SpinStringData,
        positions: ([usize; 2], [usize; 2]),
        k: usize,
    ) -> R {
        let ([i, j], [a, b]) = positions;
        let value = match st.cache.rank {
            0 => self.c0[k],
            1 => self.c1[(i, k, a)],
            _ => self
                .c2
                .as_ref()
                .map_or(R::from_real(0.0), |c2| c2[(i, j, a, b)]),
        };
        value * R::from_real(st.phase)
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
        parent.entries.iter().all(|e| {
            (parent.areps[e.a].cache.rank + parent.breps[e.b].cache.rank) as usize <= CONTRACTRANK
        })
    };
    within(target_parent) && within(source_parent)
}

/// Positions of the replaced occupied orbitals among the occupied orbitals, and of the replacing
/// virtual orbitals among the virtual orbitals, of a string with at most two replacements.
/// # Arguments:
/// - `occupation`: Parent occupation.
/// - `st`: Spin string.
/// # Returns
/// - `([usize; 2], [usize; 2])`: Hole and particle positions in ascending order.
fn string_positions(
    occupation: u128,
    st: &SpinStringData,
) -> ([usize; 2], [usize; 2]) {
    let below = |k: usize| (occupation & ((1u128 << k) - 1)).count_ones() as usize;
    let mut holes = [0; 2];
    let mut parts = [0; 2];
    for r in 0..st.cache.rank as usize {
        holes[r] = below(st.cache.holes[r] as usize);
        let p = st.cache.particles[r] as usize;
        parts[r] = p - below(p);
    }
    (holes, parts)
}

/// Contract the second occupied and virtual indices of a double, `z_{ka} = \sum_{lb} t_{klab} m_{lb}`,
/// in one pass over `t` in its storage order, so no permuted copy is made.
/// # Arguments:
/// - `t2`: Amplitudes `t_{klab}`.
/// - `m`: Occupied-virtual matrix `m_{lb}`.
/// # Returns
/// - `Array2<R>`: `z_{ka}`.
fn contract_pair<R: NOCIScalar>(
    t2: &Array4<R>,
    m: &Array2<R>,
) -> Array2<R> {
    let (o, _, v, _) = t2.dim();
    let zero = R::from_real(0.0);
    let mut z = Array2::from_elem((o, v), zero);

    // `z_{ka} \mathrel{+}= \sum_b t_{klab} m_{lb}` for every `(k, l, a)`, each row of `t` read once.
    for (k, tk) in t2.outer_iter().enumerate() {
        for (l, tkl) in tk.outer_iter().enumerate() {
            let ml = m.row(l);
            for (a, row) in tkl.outer_iter().enumerate() {
                z[(k, a)] += row
                    .iter()
                    .zip(ml.iter())
                    .fold(zero, |acc, (&t, &x)| acc + t * x);
            }
        }
    }

    z
}
