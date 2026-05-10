// scf/kernels.rs

use ndarray::{s, Array1, Array2, Array4};

use crate::StateScalar;

/// Density and bra-contraction mode for SCF kernels.
#[derive(Clone, Copy, Debug)]
pub enum DensityMode {
    /// Use transpose products with no complex conjugation.
    Holomorphic,
    /// Use adjoint products with complex conjugation.
    Hermitian,
}

/// Build a spin density matrix from occupied orbitals.
/// # Arguments:
/// - `c`: Full MO coefficient matrix ordered as occupied then virtual.
/// - `nocc`: Number of occupied orbitals.
/// - `mode`: Whether to use transpose or adjoint in the bra.
/// # Returns:
/// - `Array2<T>`: Spin density matrix.
pub fn density<T: StateScalar>(c: &Array2<T>, nocc: usize, mode: DensityMode) -> Array2<T> {
    let cocc = c.slice(s![.., 0..nocc]).to_owned();
    match mode {
        DensityMode::Holomorphic => cocc.dot(&cocc.t()),
        DensityMode::Hermitian => cocc.dot(&cocc.t().mapv(|z| z.conj())),
    }
}

/// Build spin-resolved unrestricted Fock matrices from spin densities.
/// # Arguments:
/// - `h`: One-electron Hamiltonian.
/// - `eri`: Coulomb ERIs `(pq|rs)`.
/// - `da`: Alpha-spin density matrix.
/// - `db`: Beta-spin density matrix.
/// # Returns:
/// - `(Array2<T>, Array2<T>)`: Alpha- and beta-spin Fock matrices.
pub fn fock<T: StateScalar>(h: &Array2<f64>, eri: &Array4<f64>, da: &Array2<T>, db: &Array2<T>) -> (Array2<T>, Array2<T>) {
    let n = h.nrows();
    let d = da + db;
    let mut fa = h.mapv(T::from_real);
    let mut fb = h.mapv(T::from_real);
    for p in 0..n {
        for q in 0..n {
            let mut j = T::from_real(0.0);
            let mut ka = T::from_real(0.0);
            let mut kb = T::from_real(0.0);
            for r in 0..n {
                for s in 0..n {
                    j += d[(r, s)] * T::from_real(eri[(p, q, r, s)]);
                    ka += da[(r, s)] * T::from_real(eri[(p, r, q, s)]);
                    kb += db[(r, s)] * T::from_real(eri[(p, r, q, s)]);
                }
            }
            fa[(p, q)] += j - ka;
            fb[(p, q)] += j - kb;
        }
    }
    (fa, fb)
}

/// Calculate unrestricted SCF energy from spin densities and Fock matrices.
/// # Arguments:
/// - `h`: One-electron Hamiltonian.
/// - `enuc`: Nuclear repulsion energy.
/// - `da`: Alpha-spin density matrix.
/// - `db`: Beta-spin density matrix.
/// - `fa`: Alpha-spin Fock matrix.
/// - `fb`: Beta-spin Fock matrix.
/// # Returns:
/// - `T`: SCF energy.
pub fn energy<T: StateScalar>(h: &Array2<f64>, enuc: f64, da: &Array2<T>, db: &Array2<T>, fa: &Array2<T>, fb: &Array2<T>) -> T {
    let p = da + db;
    let mut e = T::from_real(enuc);
    for mu in 0..h.nrows() {
        for nu in 0..h.ncols() {
            let hmn = T::from_real(h[(mu, nu)]);
            e += hmn * p[(mu, nu)] + T::from_real(0.5) * ((fa[(mu, nu)] - hmn) * da[(mu, nu)] + (fb[(mu, nu)] - hmn) * db[(mu, nu)]);
        }
    }
    e
}

/// Compute occupied-virtual orbital gradient g_{ai} = 2 \sum_{\mu\nu} C_a^\mu F_{\mu\nu} C_i^\nu.
/// # Arguments:
/// - `c`: Full MO coefficient matrix ordered as occupied then virtual.
/// - `f`: Spin Fock matrix.
/// - `nocc`: Number of occupied orbitals.
/// - `mode`: Whether to use transpose or adjoint in the virtual bra.
/// # Returns:
/// - `Array2<T>`: Gradient block with shape `(nvir, nocc)`.
pub fn orbital_gradient<T: StateScalar>(c: &Array2<T>, f: &Array2<T>, nocc: usize, mode: DensityMode) -> Array2<T> {
    let cocc = c.slice(s![.., 0..nocc]).to_owned();
    let cvir = c.slice(s![.., nocc..]).to_owned();
    let bra = match mode {
        DensityMode::Holomorphic => cvir.t().to_owned(),
        DensityMode::Hermitian => cvir.t().mapv(|z| z.conj()),
    };
    bra.dot(f).dot(&cocc).mapv(|z| z * T::from_real(2.0))
}

/// Compute diagonal MO Fock elements.
/// # Arguments:
/// - `c`: MO coefficient matrix.
/// - `f`: Fock matrix.
/// - `mode`: Whether to use transpose or adjoint in the bra.
/// # Returns:
/// - `Array1<T>`: Diagonal MO Fock elements.
pub fn orbital_energies<T: StateScalar>(c: &Array2<T>, f: &Array2<T>, mode: DensityMode) -> Array1<T> {
    let bra = match mode {
        DensityMode::Holomorphic => c.t().to_owned(),
        DensityMode::Hermitian => c.t().mapv(|z| z.conj()),
    };
    bra.dot(f).dot(c).diag().to_owned()
}
