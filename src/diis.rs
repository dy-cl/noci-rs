// diis.rs
use ndarray::{s, Array1, Array2};
use ndarray_linalg::{Solve};

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
        
        // Construct matrix of DIIS residuals
        //  B_{ij}^a = \sum_{pq}(E^\alpha_i)_{pq}(E^\alpha_j)_{pq}.
        //  B_{ij}^b = \sum_{pq}(E^\beta_i)_{pq}(E^\beta_j)_{pq}.
        //  Elements are B_{ij}^a + B_{ij}^b.
        //  Note we may construct the augmented matrix common in DIIS and solve 
        //  the augmented problem, but this may have issues with bad conditioning. Alternative is 
        //  to Instead only B and solve Bc' = I as in arXiv:2112.08890v1 (Eqn. 14). This should be 
        //  turned into an input option at some point.
        let mut aug = Array2::<f64>::zeros((m + 1, m + 1));
        //let mut b = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                let ae = (&self.e_hist_a[i] * &self.e_hist_a[j]).sum();
                let be = (&self.e_hist_b[i] * &self.e_hist_b[j]).sum(); 
                //b[(i, j)] = ae + be;
                aug[(i, j)] = ae + be;
            }
            aug[(i, m)] = 1.0;    
            aug[(m, i)] = 1.0;    
        }

        // Form RHS vector [1,1,..,1]^T and solve for [\mathbf{c}]^T.
        //let rhs = Array1::<f64>::from_elem(m, 1.0);
        //let c_prime = match b.clone().solve_into(rhs) {
        //    Ok(x) => x,
        //    Err(_) => return None, // Should probably do something if this fails. 
        //};
        
        // Normalise coefficient vector. 
        //let factor = c_prime.sum();
        //let c = &c_prime / factor;
        
        let mut rhs = Array1::<f64>::zeros(m + 1);
        rhs[m] = 1.0;
        let sol = match aug.solve_into(rhs) {
            Ok(x) => x,
            Err(_) => return None, // May fail if singular or ill conditioned.
        };
        let c = sol.slice(s![0..m]).to_owned();
       
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

