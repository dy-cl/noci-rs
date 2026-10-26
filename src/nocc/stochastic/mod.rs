// nocc/stochastic/mod.rs
//! Stochastic Jeziorski-Monkhorst NOCC: coupled cluster Monte Carlo of the state-universal
//! residuals
//!
//! `R_{y\mu} = \langle{}^y\Psi|\hat\tau_{y\mu}^\dagger\bar H_y|{}^y\Psi\rangle - \sum_{x\neq y}G_{xy}\langle{}^y\Psi|\hat\tau_{y\mu}^\dagger e^{-\hat T_y}|\Phi_x\rangle`
//!
//! with the plain update `t \leftarrow t - \Delta\tau R`. Populations live on the excitors of
//! each reference block in a sparse store, rather than on a dense determinant vector, because
//! the JM equations have no overlap null space in which stochastic noise could accumulate.
//!
//! The same-reference term is sampled by linked CCMC: within one reference the determinants
//! are orthonormal and `\bar H_y=e^{-\hat T_y}\hat He^{\hat T_y}` terminates after four nested
//! commutators. The coupling between references does not terminate, since the exponentials act
//! in different nonorthogonal orbital frames; it is sampled through the projections
//! `q^x_y(\nu)=\langle D_\nu|\tilde\Phi_x\rangle` of clusters of the unlinked expansion
//! `e^{\hat T_x}` up to the expansion rank, evaluated with the nonorthogonal matrix elements of
//! [`crate::elements`], followed by the exact left action of `e^{-\hat T_y}`. The same
//! projections give the effective overlap and Hamiltonian, so that
//! `G=(S^e)^{-1}H^e` is estimated from the sampled populations and applied one report block
//! late. One population-control shift acts on all references, whose populations follow the
//! power iteration of `G` and so carry the ground-state mixing coefficients.
//!
//! # References
//! - B. Jeziorski and H. J. Monkhorst, *Phys. Rev. A* **24**, 1668 (1981),
//!   [doi:10.1103/PhysRevA.24.1668](https://doi.org/10.1103/PhysRevA.24.1668).
//! - A. J. W. Thom, *Phys. Rev. Lett.* **105**, 263004 (2010),
//!   [doi:10.1103/PhysRevLett.105.263004](https://doi.org/10.1103/PhysRevLett.105.263004).
//! - R. S. T. Franklin, J. S. Spencer, A. Zoccante and A. J. W. Thom, *J. Chem. Phys.* **144**,
//!   044111 (2016), [doi:10.1063/1.4940317](https://doi.org/10.1063/1.4940317).

mod cluster;
mod coupling;
mod effective;
mod linked;
mod population;
mod propagate;

// Crate-visible function re-exports.
pub(crate) use propagate::jm_qmc_step;
