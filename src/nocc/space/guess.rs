// nocc/space/guess.rs
//! Continuation of converged GNOCC amplitudes to the next geometry of a scan.

// External crate imports.
use ndarray::{Array1, Array2, Array4, Axis};
use num_complex::Complex64;

// Crate-root imports.
use crate::maths::loewdin_x;

// Parent/sibling imports.
use super::excitations::{Excitation, dense_amplitudes};
use super::fois::FoisBasis;
use super::orbitals::Spaces;

/// Eigenvalue tolerance of the orthonormalisation of the orbital overlap between geometries.
const OVERLAPTOL: f64 = 1e-10;

/// Converged amplitudes of one geometry, kept as the starting guess of the next.
#[derive(Clone)]
pub(crate) struct AmplitudeGuess {
    /// AO coefficients of the natural orbitals the amplitudes are expressed in.
    c: Array2<f64>,
    /// Singles amplitudes `t^q_p`, stored as `[q, p]`.
    t1: Array2<Complex64>,
    /// Pair-symmetric doubles amplitudes `\bar t^{rs}_{pq}`, stored as `[r, s, p, q]`.
    t2: Array4<Complex64>,
}

impl AmplitudeGuess {
    /// Keep the converged amplitudes of one geometry in its natural-orbital basis.
    /// # Arguments:
    /// - `c`: AO coefficients of the natural orbitals.
    /// - `spaces`: Core, active, and virtual orbital-space maps.
    /// - `excitations`: Raw spin-free excitation list.
    /// - `amplitudes`: Converged amplitude vector, complex in a holomorphic run.
    /// # Returns:
    /// - `Self`: Dense amplitudes and their orbitals.
    pub(crate) fn new(
        c: &Array2<f64>,
        spaces: &Spaces,
        excitations: &[Excitation],
        amplitudes: &Array1<Complex64>,
    ) -> Self {
        let dense = dense_amplitudes(spaces, excitations, amplitudes);
        Self {
            c: c.clone(),
            t1: dense.t1,
            t2: dense.t2,
        }
    }

    /// Continue the amplitudes to the natural orbitals of a new geometry. The orbitals are
    /// matched by their overlap `U = C_{\text{old}}^T S C_{\text{new}}`, orthonormalised as
    /// `U (U^T U)^{-1/2}`, every orbital index of the dense amplitudes is rotated by `U`, and
    /// the amplitudes of the new excitation list are projected onto the new FOIS,
    /// `t = Y Y^T S t`.
    /// # Arguments:
    /// - `c`: AO coefficients of the new natural orbitals.
    /// - `s`: AO overlap matrix of the new geometry.
    /// - `spaces`: Core, active, and virtual orbital-space maps of the new geometry.
    /// - `excitations`: Raw spin-free excitation list of the new geometry.
    /// - `fois`: FOIS basis of the new geometry.
    /// # Returns:
    /// - `Array1<Complex64>`: Starting amplitude vector in the new excitation list.
    pub(crate) fn continue_to(
        &self,
        c: &Array2<f64>,
        s: &Array2<f64>,
        spaces: &Spaces,
        excitations: &[Excitation],
        fois: &FoisBasis,
    ) -> Array1<Complex64> {
        // Orthonormal orbital rotation from the old to the new natural orbitals.
        let overlap = self.c.t().dot(s).dot(c);
        let u = overlap.dot(&loewdin_x(&overlap.t().dot(&overlap), false, OVERLAPTOL));

        // `t'_{j\ldots} = \sum_{p\ldots} U_{pj} \cdots t_{p\ldots}` over every orbital index.
        let u = u.mapv(Complex64::from);
        let t1 = u.t().dot(&self.t1).dot(&u);
        let mut t2 = self.t2.clone();
        for axis in 0..4 {
            t2 = rotate_axis(&t2, &u, axis);
        }

        // Each listed excitation takes its share of the dense element it enters, so the dense
        // amplitudes of the read vector reproduce the representable part of `\bar t`.
        let ones = Array1::<f64>::ones(excitations.len());
        let shares = dense_amplitudes(spaces, excitations, &ones);
        let raw = Array1::from_iter(excitations.iter().map(|&ex| match ex {
            Excitation::Single { p, q } => t1[(q, p)],
            Excitation::Double { p, q, r, s } => t2[(r, s, p, q)] / shares.t2[(r, s, p, q)],
        }));

        // Project onto the FOIS, `t = Y Y^T S t`, separately for the real and imaginary parts.
        let project = |x: Array1<f64>| fois.y.dot(&fois.y.t().dot(&fois.metric.dot(&x)));
        let re = project(raw.mapv(|z| z.re));
        let im = project(raw.mapv(|z| z.im));
        re.iter()
            .zip(&im)
            .map(|(&a, &b)| Complex64::new(a, b))
            .collect()
    }
}

/// Rotate one orbital index of a four-index tensor, `t'_{\ldots j \ldots} = \sum_p U_{pj}
/// t_{\ldots p \ldots}`.
/// # Arguments:
/// - `t`: Four-index tensor.
/// - `u`: Orbital rotation.
/// - `axis`: Index to rotate.
/// # Returns:
/// - `Array4<Complex64>`: Rotated tensor.
fn rotate_axis(
    t: &Array4<Complex64>,
    u: &Array2<Complex64>,
    axis: usize,
) -> Array4<Complex64> {
    let mut out = Array4::<Complex64>::zeros(t.raw_dim());
    for (j, mut lane) in out.axis_iter_mut(Axis(axis)).enumerate() {
        for (p, slice) in t.axis_iter(Axis(axis)).enumerate() {
            let w = u[(p, j)];
            if w != Complex64::new(0.0, 0.0) {
                lane.scaled_add(w, &slice);
            }
        }
    }
    out
}
