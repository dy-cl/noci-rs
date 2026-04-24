mod candidate;
mod solve;
mod step;
mod types;

pub use step::snoci_step;
pub use types::SNOCIState;

pub(in crate::snoci) use candidate::CandidatePool;
pub(in crate::snoci) use solve::{build_focks, build_omega_fock, gmres, project_candidate_space, select_candidates, solve_current_space};
pub(in crate::snoci) use types::{FockMatrixElems, GmresResult, ProjectedCandidateSpaceElems};
