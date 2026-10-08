// elements/orthogonal/eval/dispatch.rs

/// Dispatch one orthogonal alpha/beta excitation-rank pair to compile-time constants.
/// The supported set is exactly the Slater-Condon support of a two-body Hamiltonian:
/// `(0,0)`, `(1,0)`, `(0,1)`, `(2,0)`, `(1,1)`, and `(0,2)`. Overlap and one-body kernels share
/// this dispatch and return zero from the sectors outside their own support.
macro_rules! dispatch_orthogonal_ranks {
    (
        $ranks:expr,
        |$ra:ident, $rb:ident| $kernel:expr,
        $fallback:expr $(,)?
    ) => {{
        match $ranks {
            (0, 0) => {
                const $ra: usize = 0;
                const $rb: usize = 0;
                $kernel
            }
            (1, 0) => {
                const $ra: usize = 1;
                const $rb: usize = 0;
                $kernel
            }
            (0, 1) => {
                const $ra: usize = 0;
                const $rb: usize = 1;
                $kernel
            }
            (2, 0) => {
                const $ra: usize = 2;
                const $rb: usize = 0;
                $kernel
            }
            (1, 1) => {
                const $ra: usize = 1;
                const $rb: usize = 1;
                $kernel
            }
            (0, 2) => {
                const $ra: usize = 0;
                const $rb: usize = 2;
                $kernel
            }
            _ => $fallback,
        }
    }};
}

pub(super) use dispatch_orthogonal_ranks;
