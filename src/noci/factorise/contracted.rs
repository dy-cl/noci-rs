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
use ndarray::{Array2, Array4, ArrayView2, Axis, s};
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
/// in-place replacements of occupied positions by virtual positions: `c_0`, `c_{ia}` and the
/// antisymmetric `c_{ijab}`, absent when every double is zero.
struct StringAmplitudes<R: NOCIScalar> {
    /// Reference amplitude `c_0`.
    c0: R,
    /// Single replacement amplitudes `c_{ia}`.
    c1: Array2<R>,
    /// Double replacement amplitudes `c_{ijab}`.
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
        let identity = target_parent == source_parent
            && data
                .mocache
                .is_some_and(|mocache| mocache[target_parent].orthogonal_slater_condon);

        Some(Self {
            target_parent,
            source_parent,
            alpha,
            beta,
            identity,
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
        let rank = |&(a, b, _): &(usize, usize, R)| {
            (source.areps[a].cache.rank, source.breps[b].cache.rank)
        };
        let n = ket.ca.ncols();
        let amplitudes = |occupation: u128,
                          strings: &[SpinStringData],
                          values: Vec<(usize, R)>,
                          doubles: bool| {
            StringAmplitudes::gather((n, occupation, strings), values, doubles)
        };
        let reference = |occupation: u128| {
            let nocc = occupation.count_ones() as usize;
            let mut r = StringAmplitudes::zeros((nocc, n - nocc), false);
            r.c0 = R::one();
            r
        };

        // Class 1: the beta-reference column, and the beta factors of the beta reference.
        let column = sources
            .iter()
            .filter(|s| rank(s).1 == 0)
            .map(|&(a, _, value)| (a, value))
            .collect();
        let (fa1, sa1) = ContractionPoint::apply(
            &self.alpha,
            &amplitudes(ket.oa, &source.areps, column, true),
            true,
        );
        let (fb0, sb0) = ContractionPoint::apply(&self.beta, &reference(ket.ob), true);

        // Class 2: the alpha-reference row over excited beta strings, and the alpha factors of
        // the alpha reference.
        let row = sources
            .iter()
            .filter(|s| rank(s).0 == 0 && rank(s).1 > 0)
            .map(|&(_, b, value)| (b, value))
            .collect();
        let (fb2, sb2) = ContractionPoint::apply(
            &self.beta,
            &amplitudes(ket.ob, &source.breps, row, true),
            true,
        );
        let (fa0, sa0) = ContractionPoint::apply(&self.alpha, &reference(ket.oa), true);

        // Class 3: one column of `D^{11}` per beta single. Each column needs only the reference and
        // single-replacement outputs of its alpha apply and of the beta factors of its single.
        let mut grouped: HashMap<usize, Vec<(usize, R)>> = HashMap::new();
        for &(a, b, value) in sources.iter().filter(|s| rank(s) == (1, 1)) {
            grouped.entry(b).or_default().push((a, value));
        }
        let grouped = grouped.into_iter().collect::<Vec<_>>();
        let columns = grouped
            .par_iter()
            .map(|(b, column)| {
                let alpha = amplitudes(ket.oa, &source.areps, column.clone(), false);
                let beta = amplitudes(ket.ob, &source.breps, vec![(*b, R::one())], false);
                (
                    ContractionPoint::apply(&self.alpha, &alpha, false),
                    ContractionPoint::apply(&self.beta, &beta, false),
                )
            })
            .collect::<Vec<_>>();

        // Beta-reference targets: alpha applies of `D^{11} w^S` and `D^{11} w^F`.
        let weighted = |w: &dyn Fn(usize) -> R| {
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
        let (fws, sws) =
            ContractionPoint::apply(&self.alpha, &weighted(&|c| columns[c].1.1.c0), true);
        let (_, swf) =
            ContractionPoint::apply(&self.alpha, &weighted(&|c| columns[c].1.0.c0), true);

        // Alpha-reference targets: beta applies of the reference rows of `F^\alpha D^{11}` and
        // `S^\alpha D^{11}`.
        let reference_row = |pick: &dyn Fn(usize) -> R| {
            let values = grouped
                .iter()
                .enumerate()
                .map(|(c, (b, _))| (*b, pick(c)))
                .collect();
            amplitudes(ket.ob, &source.breps, values, false)
        };
        let (_, suf) =
            ContractionPoint::apply(&self.beta, &reference_row(&|c| columns[c].0.0.c0), true);
        let (fus, sus) =
            ContractionPoint::apply(&self.beta, &reference_row(&|c| columns[c].0.1.c0), true);

        // Single-single targets: `Y^F = X^F (S^\beta)^T + X^S (F^\beta)^T` and
        // `Y^S = X^S (S^\beta)^T`, with one row per target single and one column per beta single.
        // `rows` maps every target string to its row, `usize::MAX` for strings that are not singles.
        let singles =
            |occupation: u128, strings: &[SpinStringData], pick: &[&StringAmplitudes<R>]| {
                let ids = (0..strings.len())
                    .filter(|&k| strings[k].cache.rank == 1)
                    .collect::<Vec<_>>();
                let mut rows = vec![usize::MAX; strings.len()];
                for (r, &k) in ids.iter().enumerate() {
                    rows[k] = r;
                }
                let m = Array2::from_shape_fn((ids.len(), pick.len()), |(r, c)| {
                    pick[c].read(occupation, &strings[ids[r]])
                });
                (m, rows)
            };
        let (fa, sa): (Vec<_>, Vec<_>) = columns.iter().map(|c| (&c.0.0, &c.0.1)).unzip();
        let (fb, sb): (Vec<_>, Vec<_>) = columns.iter().map(|c| (&c.1.0, &c.1.1)).unzip();
        let (xf, arows) = singles(bra.oa, &target.areps, &fa);
        let (xs, _) = singles(bra.oa, &target.areps, &sa);
        let (bf, brows) = singles(bra.ob, &target.breps, &fb);
        let (bs, _) = singles(bra.ob, &target.breps, &sb);
        let yf = xf.dot(&bs.t()) + xs.dot(&bf.t());
        let ys = xs.dot(&bs.t());

        // Every class at the target strings.
        let at = |amp: &StringAmplitudes<R>, occupation: u128, strings: &[SpinStringData]| {
            strings
                .iter()
                .map(|st| amp.read(occupation, st))
                .collect::<Vec<_>>()
        };
        let alpha = |amp: &StringAmplitudes<R>| at(amp, bra.oa, &target.areps);
        let beta = |amp: &StringAmplitudes<R>| at(amp, bra.ob, &target.breps);
        let (fa1, sa1, fa0, sa0) = (alpha(&fa1), alpha(&sa1), alpha(&fa0), alpha(&sa0));
        let (fws, sws, swf) = (alpha(&fws), alpha(&sws), alpha(&swf));
        let (fb0, sb0, fb2, sb2) = (beta(&fb0), beta(&sb0), beta(&fb2), beta(&sb2));
        let (suf, fus, sus) = (beta(&suf), beta(&fus), beta(&sus));

        // `Y^Q_{\bar a\bar b} = F_{\bar a\bar b} + \lambda S_{\bar a\bar b}` at every target
        // determinant: classes 1 and 2 everywhere, class 3 by target class.
        targets
            .par_iter()
            .with_min_len(DOTCHUNK)
            .map(|&(out, i, j)| {
                let mut f = fa1[i] * sb0[j] + sa1[i] * fb0[j] + fa0[i] * sb2[j] + sa0[i] * fb2[j];
                let mut s = sa1[i] * sb0[j] + sa0[i] * sb2[j];

                match (target.areps[i].cache.rank, target.breps[j].cache.rank) {
                    (_, 0) => {
                        f += fws[i] + swf[i];
                        s += sws[i];
                    }
                    (0, _) => {
                        f += suf[j] + fus[j];
                        s += sus[j];
                    }
                    (1, 1) => {
                        f += yf[(arows[i], brows[j])];
                        s += ys[(arows[i], brows[j])];
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

    /// Apply the same-spin factors of every point to amplitudes over the ket strings, returning
    /// `F c` and `S c` over the bra strings with at most two replacements, or one when `doubles`
    /// is false. `F c` stays zero for overlap-only points. At each point the ket amplitudes are carried into the biorthogonal frame,
    /// `t_{kb} = \sum_j G_{jk} c_{jb}` and `t_{klab} = \sum_{ij} G_{ik} G_{jl} c_{ijab}`; their
    /// virtual orbitals split as `u_b = \tilde\phi\alpha_b + \sum_c \hat q_c Q_{cb}`; the operator
    /// acts by the orthogonal one-body rules with `f`; and the bra virtual orbitals split as
    /// `v_a = \tilde v_a + \phi\beta_a^*`.
    /// # Arguments:
    /// - `points`: Extrapolation points of one spin and parent pair.
    /// - `c`: Ket string amplitudes.
    /// - `doubles`: Whether double-replacement outputs are needed.
    /// # Returns
    /// - `(StringAmplitudes<R>, StringAmplitudes<R>)`: `F c` and `S c`.
    fn apply<R>(
        points: &[Self],
        c: &StringAmplitudes<R>,
        doubles: bool,
    ) -> (StringAmplitudes<R>, StringAmplitudes<R>)
    where
        R: NOCIScalar + From<T>,
    {
        let (o, v) = c.c1.dim();
        let mut sf = StringAmplitudes::zeros((o, v), doubles);
        let mut ss = StringAmplitudes::zeros((o, v), doubles);
        let half = R::from_real(0.5);

        for point in points {
            let cast = |a: ArrayView2<T>| a.mapv(<R as From<T>>::from);
            let (g, alpha, beta, q) = (
                cast(point.g.view()),
                cast(point.alpha.view()),
                cast(point.beta.view()),
                cast(point.q.view()),
            );

            // Ket: `G` on each occupied index, `t_{klab} = \sum_{ij} G_{ik} G_{jl} c_{ijab}` as
            // one product per index.
            let t1 = g.t().dot(&c.c1);
            let t2 = c.c2.as_ref().map(|c2| {
                let first = g.t().dot(&c2.view().into_shape((o, o * v * v)).unwrap());
                let first = first.into_shape((o, o, v * v)).unwrap();
                let mut t2 = Array4::from_elem((o, o, v, v), R::from_real(0.0));
                for k in 0..o {
                    let tk = g.t().dot(&first.index_axis(Axis(0), k));
                    t2.index_axis_mut(Axis(0), k)
                        .assign(&tk.into_shape((o, v, v)).unwrap());
                }
                t2
            });

            // Ket: lower the rank through `\alpha`, then carry each virtual index through `Q`,
            // `r_{klcd} = \sum_{ab} Q_{ca} Q_{db} t_{klab}`.
            let z = t2.as_ref().map(|t2| contract_pair(t2, &alpha));
            let mut r0 = c.c0 + (&t1 * &alpha).sum();
            if let Some(z) = &z {
                r0 += half * (z * &alpha).sum();
            }
            let r1 = match &z {
                Some(z) => (&t1 + z).dot(&q.t()),
                None => t1.dot(&q.t()),
            };
            let r2 = t2.map(|t2| {
                let swap = |x: Array4<R>| {
                    x.permuted_axes([0, 1, 3, 2])
                        .as_standard_layout()
                        .into_owned()
                };
                let once = t2.into_shape((o * o * v, v)).unwrap().dot(&q.t());
                let once = swap(once.into_shape((o, o, v, v)).unwrap());
                let twice = once.into_shape((o * o * v, v)).unwrap().dot(&q.t());
                swap(twice.into_shape((o, o, v, v)).unwrap())
            });

            // One-body sigma by the orthogonal rules, for points with a one-body operator.
            let sigma = point.f.as_ref().map(|f| {
                let f_oo = cast(f.slice(s![..o, ..o]));
                let f_ov = cast(f.slice(s![..o, o..]));
                let f_vo = cast(f.slice(s![o.., ..o]));
                let f_vv = cast(f.slice(s![o.., o..]));
                let e0 = f_oo.diag().sum();

                // One-body sigma by the orthogonal rules: the one-body Slater-Condon rules of
                // `elements::orthogonal`, contracted over whole amplitude arrays rather than evaluated
                // per determinant pair, which would visit the `O(ov)` connections of every string.
                // `\sigma_0 = e_0 r_0 + \sum_{kc} f_{kc} r_{kc}`,
                // `\sigma_{ia} = f_{ai} r_0 + e_0 r_{ia} + \sum_c f_{ac} r_{ic} - \sum_k f_{ki} r_{ka}
                //  + \sum_{kc} f_{kc} r_{ikac}`.
                let s0 = e0 * r0 + (&f_ov * &r1).sum();
                let mut s1 = f_vo.t().mapv(|z| z * r0) + r1.mapv(|z| z * e0) + r1.dot(&f_vv.t())
                    - f_oo.t().dot(&r1);
                if let Some(r2) = &r2 {
                    s1 += &contract_pair(r2, &f_ov);
                }

                // `\sigma_{ijab} = e_0 r_{ijab} + P(ab)\sum_c f_{bc} r_{ijac}
                //  - P(ij)\sum_k f_{kj} r_{ikab} + P(ij)P(ab) f_{bj} r_{ia}`.
                let s2 = doubles.then(|| {
                    let mut s2 = antisym_outer(&r1, &f_vo);
                    if let Some(r2) = &r2 {
                        let vv = r2.view().into_shape((o * o * v, v)).unwrap().dot(&f_vv.t());
                        let vv = vv.into_shape((o, o, v, v)).unwrap();
                        let mut oo = Array4::from_elem((o, o, v, v), R::from_real(0.0));
                        for i in 0..o {
                            let ri = r2.index_axis(Axis(0), i).into_shape((o, v * v)).unwrap();
                            oo.index_axis_mut(Axis(0), i)
                                .assign(&f_oo.t().dot(&ri).into_shape((o, v, v)).unwrap());
                        }
                        s2 += &r2.mapv(|z| z * e0);
                        s2 += &(&vv - &vv.view().permuted_axes([0, 1, 3, 2]));
                        s2 -= &(&oo - &oo.view().permuted_axes([1, 0, 2, 3]));
                    }
                    s2
                });
                (s0, s1, s2)
            });

            // Bra: raise the rank through `\beta`, for `F c` from `\sigma` and for `S c` from `r`,
            // `w_{ijab} = s_{ijab} + P(ij)P(ab) \beta_{bj} s_{ia}
            //  + s_0 (\beta_{ai}\beta_{bj} - \beta_{aj}\beta_{bi})`.
            let scale = <R as From<T>>::from(point.scale);
            let pairs = doubles.then(|| {
                Array4::from_shape_fn((o, o, v, v), |(i, j, a, b)| {
                    beta[(a, i)] * beta[(b, j)] - beta[(a, j)] * beta[(b, i)]
                })
            });
            let outputs = [
                sigma
                    .as_ref()
                    .map(|(s0, s1, s2)| (&mut sf, (*s0, s1, s2.as_ref()))),
                Some((&mut ss, (r0, &r1, r2.as_ref()))),
            ];
            for (out, (x0, x1, x2)) in outputs.into_iter().flatten() {
                out.c0 += scale * x0;
                out.c1.scaled_add(scale, &(x1 + &beta.t().mapv(|z| z * x0)));
                if let (Some(out2), Some(pairs)) = (out.c2.as_mut(), pairs.as_ref()) {
                    let mut w2 = antisym_outer(x1, &beta) + pairs.mapv(|z| z * x0);
                    if let Some(x2) = x2 {
                        w2 += x2;
                    }
                    out2.scaled_add(scale, &w2);
                }
            }
        }

        (sf, ss)
    }
}

impl<R: NOCIScalar> StringAmplitudes<R> {
    /// Zero amplitudes, with room for doubles when requested.
    /// # Arguments:
    /// - `dims`: Numbers of occupied and virtual orbitals.
    /// - `doubles`: Whether double replacements are held.
    /// # Returns
    /// - `Self`: Zero amplitudes.
    fn zeros(
        dims: (usize, usize),
        doubles: bool,
    ) -> Self {
        let (o, v) = dims;
        let zero = R::from_real(0.0);
        Self {
            c0: zero,
            c1: Array2::from_elem((o, v), zero),
            c2: doubles.then(|| Array4::from_elem((o, o, v, v), zero)),
        }
    }

    /// Collect amplitudes given per parent-local string ID. The determinant of a string differs
    /// from the in-place replacements of its parent reference by the string's phase.
    /// # Arguments:
    /// - `parent`: Number of orbitals, parent occupation and its spin strings.
    /// - `values`: String IDs and amplitudes.
    /// - `doubles`: Whether double replacements may be present.
    /// # Returns
    /// - `Self`: Amplitudes in tensor form.
    fn gather(
        parent: (usize, u128, &[SpinStringData]),
        values: Vec<(usize, R)>,
        doubles: bool,
    ) -> Self {
        let (n, occupation, strings) = parent;
        let nocc = occupation.count_ones() as usize;
        let mut out = Self::zeros((nocc, n - nocc), doubles);

        for (id, value) in values {
            let st = &strings[id];
            let value = value * R::from_real(st.phase);
            let ([i, j], [a, b]) = string_positions(occupation, st);
            match st.cache.rank {
                0 => out.c0 += value,
                1 => out.c1[(i, a)] += value,
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

    /// Read the amplitude of one string in the determinant convention of the spin strings.
    /// Double replacements read zero when no doubles are held.
    /// # Arguments:
    /// - `occupation`: Parent occupation.
    /// - `st`: String to read.
    /// # Returns
    /// - `R`: Its amplitude.
    fn read(
        &self,
        occupation: u128,
        st: &SpinStringData,
    ) -> R {
        let ([i, j], [a, b]) = string_positions(occupation, st);
        let value = match st.cache.rank {
            0 => self.c0,
            1 => self.c1[(i, a)],
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

/// Contract the second occupied and virtual indices of a double, `z_{ka} = \sum_{lb} t_{klab} m_{lb}`.
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
    let t = t2.view().permuted_axes([0, 2, 1, 3]);
    let t = t.as_standard_layout();
    let z = t
        .view()
        .into_shape((o * v, o * v))
        .unwrap()
        .dot(&m.view().into_shape(o * v).unwrap());
    z.into_shape((o, v)).unwrap()
}

/// Antisymmetrised outer product
/// `P(ij)P(ab)\, x_{ia} y_{bj} = x_{ia}y_{bj} - x_{ja}y_{bi} - x_{ib}y_{aj} + x_{jb}y_{ai}`.
/// # Arguments:
/// - `x`: Occupied-virtual matrix `x_{ia}`.
/// - `y`: Virtual-occupied matrix `y_{bj}`.
/// # Returns
/// - `Array4<R>`: The antisymmetric double.
fn antisym_outer<R: NOCIScalar>(
    x: &Array2<R>,
    y: &Array2<R>,
) -> Array4<R> {
    let (o, v) = x.dim();
    Array4::from_shape_fn((o, o, v, v), |(i, j, a, b)| {
        x[(i, a)] * y[(b, j)] - x[(j, a)] * y[(b, i)] - x[(i, b)] * y[(a, j)]
            + x[(j, b)] * y[(a, i)]
    })
}
