// diis.rs
use ndarray::{Array1, Array2, s};
use ndarray_linalg::{Solve};

/// DIIS storage for SCF states. Stores Fock and error matrices for each spin.
pub struct Diis {
    m: usize, // Size of the DIIS subspace, number of past SCF iterations with stored history.
    f_hist: Vec<Array2<f64>>, // History of Fock matrices spin a. 
    e_hist: Vec<Array2<f64>>, // History of error matrices spin b.
}

impl Diis {
    /// Constructor for DIIS object, initialises object with DIIS space m. 
    /// History of Fock and error matrices are initialised as vectors with length m.
    /// # Arguments:
    ///     `m`: usize, size of the DIIS space.
    pub fn new(m: usize) -> Self {
        Self {
            m, 
            f_hist: Vec::with_capacity(m),
            e_hist: Vec::with_capacity(m), 
        }
    }
    
    /// Calculate the Pulay DIIS error. E^s = F^s D^s S - S D^s F^s. 
    /// # Arguments:
    ///     `f`: Array2, spin specific Fock matrix. 
    ///     `d`: Array2, spin specific density matrix. 
    ///     `s`: Array2, AO overlap matrix.
    fn build_error(f: &Array2<f64>, d: &Array2<f64>, s: &Array2<f64>) -> Array2<f64> {
        let fds = f.dot(d).dot(s);
        let sdf = s.dot(d).dot(f);

        &fds - &sdf
    }
    
    /// Add Fock matrices and DIIS error matrices from current SCF cycle to history.
    /// # Arguments:
    ///     `f`: Array2, spin Fock matrix.
    ///     `d`: Array2, spin density matrix.
    ///     `s`: Array2, AO overlap matrix.
    pub fn push(&mut self, f: &Array2<f64>, d: &Array2<f64>, s: &Array2<f64>) {
        let e = Self::build_error(f, d, s);

        self.f_hist.push(f.clone());
        self.e_hist.push(e);

        // Cap DIIS subspace at size m.
        if self.f_hist.len() > self.m {
            self.f_hist.remove(0);
            self.e_hist.remove(0);
        }
    }
    
    /// Calculate squared norm of most recent error pair. ||E^a|| + ||E^b||. 
    /// # Arguments:
    ///     `self`: Diis struct, contains DIIS data. 
    pub fn last_error_norm2(&self) -> Option<f64> {
        let m = self.e_hist.len();
        // If our subspace size is currently zero we cannot calculate an error.
        if m == 0 {
            return None;
        }
        let e = &self.e_hist[m - 1];
        Some((e * e).sum())
    }
    
    /// Use past Fock matrices and solutions to augmented linear system to extrapolate. 
    /// Calculates the extrapolated DIIS Fock matrix F_{DIIS}^s = \sum_i^m c_i F_i^s.
    /// # Arguments
    ///     `self`: Diis struct, contains DIIS data.
    pub fn extrapolate_fock(&self) -> Option<Array2<f64>> {
        let m = self.e_hist.len();
        // With less than 2 cycles of history we cannot extrapolate to DIIS Fock.
        if m < 2 {
            return None;
        }

        // Construct augmented matrix.
        //  B_{ij}^a = \sum_{pq}(E^\alpha_i)_{pq}(E^\alpha_j)_{pq}
        //  B_{ij}^b = \sum_{pq}(E^\beta_i)_{pq}(E^\beta_j)_{pq}
        //  Elements are B_{ij}^a + B_{ij}^b with additional row/column of 1s. 
        //  aug[(m + 1, m + 1)] = 0.
        let mut aug = Array2::<f64>::zeros((m + 1, m + 1));
        for i in 0..m {
            for j in 0..m {
                let bij = (&self.e_hist[i] * &self.e_hist[j]).sum();
                aug[(i, j)] = bij;
            }
            aug[(i, m)] = 1.0;
            aug[(m, i)] = 1.0;
        }
        
        // Form RHS vector [\mathbf{0} 1]^T and solve for [\mathbf{c} \lambda]^T.
        let mut rhs = Array1::<f64>::zeros(m + 1);
        rhs[m] = 1.0;
        let sol = match aug.solve_into(rhs) {
            Ok(x) => x,
            Err(_) => return None, // May fail if singular or ill conditioned.
        };
        let c = sol.slice(s![0..m]).to_owned();
        
        // Extrapolate to get DIIS Fock matrix.
        // F_{DIIS}^s = \sum_i^m c_i F_i^s 
        let mut f_diis = self.f_hist[0].clone() * c[0];
        for i in 1..m {
            f_diis = f_diis + self.f_hist[i].clone() * c[i];
        }

        Some(f_diis)

    }

}
