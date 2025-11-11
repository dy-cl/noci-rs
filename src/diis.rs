// diis.rs
use ndarray::{s, Array1, Array2};
use ndarray_linalg::{EighInto, UPLO};

use crate::maths::loewdin_x_real;

/// DIIS storage for SCF states. Stores Fock and error matrices for each spin.
pub struct Diis {
    m: usize, // Size of the DIIS subspace, number of past SCF iterations with stored history.
    f_hist_a: Vec<Array2<f64>>, // History of Fock matrices spin a. 
    f_hist_b: Vec<Array2<f64>>, // History of Fock matrices spin b
    e_hist_a: Vec<Array2<f64>>, // History of error matrices spin a.
    e_hist_b: Vec<Array2<f64>>, // History of error matrices spin b.
}

impl Diis {
    /// Constructor for DIIS object, initialises object with DIIS space m. 
    /// History of Fock and error matrices are initialised as vectors with length m.
    /// # Arguments:
    ///     `m`: usize, size of the DIIS space.
    pub fn new(m: usize) -> Self {
        Self {
            m, 
            f_hist_a: Vec::with_capacity(m),
            f_hist_b: Vec::with_capacity(m),
            e_hist_a: Vec::with_capacity(m),
            e_hist_b: Vec::with_capacity(m),
        }
    }
    
    /// Calculate the orthonormalised DIIS error. R = F^s D^s S - S D^s F^s, then 
    /// R' = X^T R X where X = S^{-1/2}.
    /// # Arguments:
    ///     `f`: Array2, spin specific Fock matrix. 
    ///     `d`: Array2, spin specific density matrix. 
    ///     `s`: Array2, AO overlap matrix.
    fn build_error(f: &Array2<f64>, d: &Array2<f64>, s: &Array2<f64>) -> Array2<f64> {
        let r = f.dot(d).dot(s) - s.dot(d).dot(f);
        let x = loewdin_x_real(s, false, 1e-12);
        x.t().dot(&r).dot(&x)
    }
    
    /// Add Fock matrices and DIIS error matrices from current SCF cycle to history.
    /// # Arguments:
    ///     `fa`: Array2, spin a Fock matrix.
    ///     `fb`: Array2, spin b Fock matrix.
    ///     `da`: Array2, spin a density matrix.
    ///     `db`: Array2, spin b density matrix.
    ///     `s`: Array2, AO overlap matrix.
    pub fn push(&mut self, fa: &Array2<f64>, fb: &Array2<f64>, da: &Array2<f64>, 
                db: &Array2<f64>, s: &Array2<f64>) {
        let r_prime_a = Self::build_error(fa, da, s); // R'^a = X^T R^a X. 
        let r_prime_b = Self::build_error(fb, db, s); // R'^b = X^T R^b X.

        self.f_hist_a.push(fa.clone());
        self.e_hist_a.push(r_prime_a);
        self.f_hist_b.push(fb.clone());
        self.e_hist_b.push(r_prime_b);

        // Cap DIIS subspace at size m.
        if self.f_hist_a.len() > self.m {
            self.f_hist_a.remove(0);
            self.e_hist_a.remove(0);
            self.f_hist_b.remove(0);
            self.e_hist_b.remove(0);
        }
    }
    
    /// Calculate squared norm of most recent error pair. ||E^a|| + ||E^b||. 
    /// # Arguments:
    ///     `self`: Diis struct, contains DIIS data. 
    pub fn last_error_norm2(&self) -> Option<f64> {
        let m = self.e_hist_a.len();
        // If our subspace size is currently zero we cannot calculate an error.
        if m == 0 {
            return None;
        }
        let ea = &self.e_hist_a[m - 1];
        let eb = &self.e_hist_b[m - 1];
        Some((ea * ea).sum() + (eb * eb).sum())
    }
    
    /// Use past Fock matrices and solutions to augmented linear system to extrapolate. 
    /// Calculates the extrapolated DIIS Fock matrix F_{DIIS}^s = \sum_i^m c_i F_i^s.
    /// # Arguments
    ///     `self`: Diis struct, contains DIIS data.
    pub fn extrapolate_fock(&self) -> Option<(Array2<f64>, Array2<f64>)> {
        let m = self.e_hist_a.len();

        // With less than 2 cycles of history we cannot extrapolate to DIIS Fock.
        if m < 2 {
            return None;
        }
        
        // Construct augmented matrix of DIIS residuals.
        //  B_{ij}^a = \sum_{pq}(E^\alpha_i)_{pq}(E^\alpha_j)_{pq}.
        //  B_{ij}^b = \sum_{pq}(E^\beta_i)_{pq}(E^\beta_j)_{pq}.
        //  Main matrix elements are B_{ij}^a + B_{ij}^b, augmentation 
        //  applies H[0, 0] = 0, H[0, i] = 1 for all i..m, and 
        //  H[i, 0] = 1 for all i..m.
        let mut h = Array2::<f64>::zeros((m + 1, m + 1));
        h.slice_mut(s![0, 1..]).fill(1.0);
        h.slice_mut(s![1.., 0]).fill(1.0);
        for i in 0..m {
            for j in 0..=i {
                let ae = (&self.e_hist_a[i] * &self.e_hist_a[j]).sum();
                let be = (&self.e_hist_b[i] * &self.e_hist_b[j]).sum(); 
                let bij = ae + be;
                h[(i + 1, j + 1)] = bij;
                h[(j + 1, i + 1)] = bij;
            }
        }
        
        // Form RHS of the Pulay system g = [1, \mathbf{0}]^T. We are solving 
        // for [\lambda, c_1, .., c_m] with constraint 1^T c = 1.
        let mut g = Array1::<f64>::zeros(m + 1);
        g[0] = 1.0;

        // Diagonalise augmeneted matrix.
        let (w, v) = h.clone().eigh_into(UPLO::Lower).unwrap();
        
        // Drop egienvectors with small eigenvalues to keep well conditioned.
        let tol = 1e-13; 
        let keep: Vec<usize> = (0..w.len()).filter(|&k| w[k].abs() > tol).collect();
        if keep.is_empty() {return None;}
        
        // Calculate coefficients V \Lambda^{-1} V^T g which are eigenvectors of 
        // the pseudo-inverse H^{-1} (i.e., not truly inverse due to the dropping 
        // of tiny eigenvalues). Doing so reduces linear dependence in B.
        let mut c_aug = Array1::<f64>::zeros(m + 1);
        for &k in &keep {
            let vk = v.slice(s![.., k]);            
            let alpha = vk.dot(&g) / w[k];           
            for i in 0..c_aug.len() {c_aug[i] += vk[i] * alpha;}
        }
        
        // Ignore Lagrange multiplier \lambda.
        let mut c = c_aug.slice(s![1..]).to_owned();

        // Normalise coefficients such that \sum_i c_i = 1.
        let n: f64 = c.sum();
        if n.abs() > 0.0 {c.mapv_inplace(|x| x / n);}

        // Extrapolate to get DIIS Fock matrix.
        // F_{DIIS}^s = \sum_i^m c_i F_i^s.
        let mut fa_diis = self.f_hist_a[0].clone() * c[0];
        let mut fb_diis = self.f_hist_b[0].clone() * c[0];
        for i in 1..m {
            fa_diis = fa_diis + self.f_hist_a[i].clone() * c[i];
            fb_diis = fb_diis + self.f_hist_b[i].clone() * c[i];
        }
        Some((fa_diis, fb_diis))
    }
}

