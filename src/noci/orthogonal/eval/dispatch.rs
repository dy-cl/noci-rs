// noci/orthogonal/eval/dispatch.rs

/// Dispatch one orthogonal alpha/beta excitation-rank pair to compile-time constants.
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
