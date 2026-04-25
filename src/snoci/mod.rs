mod candidate;
mod solve;
mod step;
mod types;

pub use step::snoci_step;
pub use types::SNOCIState;

pub(in crate::snoci) use candidate::CandidatePool;
pub(in crate::snoci) use solve::{build_candidate_current_h, build_snoci_overlaps, build_snoci_focks, build_candidate_m, 
                                 build_omega_candidate_m, build_omega_coupling_v, gmres, select_candidates, solve_current_space};
pub(in crate::snoci) use types::{GMRES, SNOCIOverlaps, SNOCIFocks};
