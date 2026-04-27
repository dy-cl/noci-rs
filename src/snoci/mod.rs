mod candidate;
mod solve;
mod step;
mod types;

pub use step::snoci_step;
pub use types::SNOCIState;

pub(in crate::snoci) use candidate::CandidatePool;
pub(in crate::snoci) use solve::{build_candidate_current_h, build_snoci_overlaps, build_snoci_focks, build_omega_m_diag, build_snoci_projection, 
                                 apply_omega_m, build_candidate_v, build_omega_v, gmres, select_candidates, solve_current_space};
pub(in crate::snoci) use types::{GMRES, SNOCIOverlaps, SNOCIFocks, PT2Projection, PT2ProjectedOperator};
