// nonorthogonalwicks/eval/dispatch.rs

/// Dispatch a supported same-spin one-body rank pair.
/// The supplied identifiers become arm-local constants, allowing the caller expression to
/// instantiate any scalar or SIMD kernel with the same `(RX, RW, L)` table.
macro_rules! dispatch_onebody_ranks {
    (
        @rank ($rx_value:literal, $rw_value:literal),
        |$rx:ident, $rw:ident, $l:ident| $kernel:expr
    ) => {{
        const $rx: usize = $rx_value;
        const $rw: usize = $rw_value;
        const $l: usize = $rx + $rw;
        $kernel
    }};
    (
        @match $ranks:expr,
        |$rx:ident, $rw:ident, $l:ident| $kernel:expr,
        $fallback:expr;
        $(($rx_value:literal, $rw_value:literal)),+ $(,)?
    ) => {{
        match $ranks {
            $(
                ($rx_value, $rw_value) => dispatch_onebody_ranks!(
                    @rank ($rx_value, $rw_value),
                    |$rx, $rw, $l| $kernel
                ),
            )+
            _ => $fallback,
        }
    }};
    (
        $ranks:expr,
        |$rx:ident, $rw:ident, $l:ident| $kernel:expr,
        $fallback:expr $(,)?
    ) => {{
        dispatch_onebody_ranks!(
            @match $ranks,
            |$rx, $rw, $l| $kernel,
            $fallback;
            (0, 1),
            (1, 0),
            (0, 2),
            (1, 1),
            (2, 0),
            (0, 3),
            (1, 2),
            (2, 1),
            (3, 0),
            (0, 4),
            (1, 3),
            (2, 2),
            (3, 1),
            (4, 0),
        )
    }};
}

/// Dispatch a supported same-spin overlap rank pair.
/// The supplied identifiers become arm-local constants, allowing scalar and SIMD overlap kernels
/// to share the same `(RX, RW, L)` table for individual ranks up to four and total ranks up to six.
macro_rules! dispatch_overlap_ranks {
    (
        @rank ($rx_value:literal, $rw_value:literal),
        |$rx:ident, $rw:ident, $l:ident| $kernel:expr
    ) => {{
        const $rx: usize = $rx_value;
        const $rw: usize = $rw_value;
        const $l: usize = $rx + $rw;
        $kernel
    }};
    (
        @match $ranks:expr,
        |$rx:ident, $rw:ident, $l:ident| $kernel:expr,
        $fallback:expr;
        $(($rx_value:literal, $rw_value:literal)),+ $(,)?
    ) => {{
        match $ranks {
            $(
                ($rx_value, $rw_value) => dispatch_overlap_ranks!(
                    @rank ($rx_value, $rw_value),
                    |$rx, $rw, $l| $kernel
                ),
            )+
            _ => $fallback,
        }
    }};
    (
        $ranks:expr,
        |$rx:ident, $rw:ident, $l:ident| $kernel:expr,
        $fallback:expr $(,)?
    ) => {{
        dispatch_overlap_ranks!(
            @match $ranks,
            |$rx, $rw, $l| $kernel,
            $fallback;
            (0, 1),
            (1, 0),
            (0, 2),
            (1, 1),
            (2, 0),
            (0, 3),
            (1, 2),
            (2, 1),
            (3, 0),
            (0, 4),
            (1, 3),
            (2, 2),
            (3, 1),
            (4, 0),
            (1, 4),
            (2, 3),
            (3, 2),
            (4, 1),
            (2, 4),
            (3, 3),
            (4, 2),
        )
    }};
}

#[cfg(feature = "nocc")]
/// Dispatch a supported same-spin rank-`K` RDM and excitation-rank tuple.
/// The RDM operator contributes `K` external contraction pairs, so the augmented determinant has
/// compile-time dimension `D = K + L`, where `L = RX + RW` is supplied by the overlap rank table.
macro_rules! dispatch_rdm_ranks {
    (
        @operator $k_value:literal,
        $ranks:expr,
        |$k:ident, $rx:ident, $rw:ident, $l:ident, $d:ident| $kernel:expr,
        $fallback:expr
    ) => {{
        match $ranks {
            (0, 0) => {
                const $k: usize = $k_value;
                const $rx: usize = 0;
                const $rw: usize = 0;
                const $l: usize = 0;
                const $d: usize = $k;
                $kernel
            }
            ranks => dispatch_overlap_ranks!(
                ranks,
                |$rx, $rw, $l| {
                    const $k: usize = $k_value;
                    const $d: usize = $k + $l;
                    $kernel
                },
                $fallback,
            ),
        }
    }};
    (
        $k_value:expr,
        $ranks:expr,
        |$k:ident, $rx:ident, $rw:ident, $l:ident, $d:ident| $kernel:expr,
        $fallback:expr $(,)?
    ) => {{
        match $k_value {
            0 => dispatch_rdm_ranks!(
                @operator 0,
                $ranks,
                |$k, $rx, $rw, $l, $d| $kernel,
                $fallback
            ),
            1 => dispatch_rdm_ranks!(
                @operator 1,
                $ranks,
                |$k, $rx, $rw, $l, $d| $kernel,
                $fallback
            ),
            2 => dispatch_rdm_ranks!(
                @operator 2,
                $ranks,
                |$k, $rx, $rw, $l, $d| $kernel,
                $fallback
            ),
            3 => dispatch_rdm_ranks!(
                @operator 3,
                $ranks,
                |$k, $rx, $rw, $l, $d| $kernel,
                $fallback
            ),
            4 => dispatch_rdm_ranks!(
                @operator 4,
                $ranks,
                |$k, $rx, $rw, $l, $d| $kernel,
                $fallback
            ),
            _ => $fallback,
        }
    }};
}

/// Dispatch a supported two-spin Hamiltonian rank tuple.
/// `LA`, `LB`, determinant storage, and second-minor storage are derived as arm-local
/// constants from the four independent excitation ranks. The caller expression can therefore
/// instantiate scalar or SIMD kernels from this single rank table.
macro_rules! dispatch_hamiltonian_ranks {
    (
        @rank (
            $rxa_value:literal, $rwa_value:literal,
            $rxb_value:literal, $rwb_value:literal
        ),
        |
            $rxa:ident, $rwa:ident, $la:ident,
            $rxb:ident, $rwb:ident, $lb:ident,
            $da:ident, $db:ident, $sa:ident, $sb:ident
        | $kernel:expr
    ) => {{
        const $rxa: usize = $rxa_value;
        const $rwa: usize = $rwa_value;
        const $la: usize = $rxa + $rwa;
        const $rxb: usize = $rxb_value;
        const $rwb: usize = $rwb_value;
        const $lb: usize = $rxb + $rwb;
        const $da: usize = $la * $la;
        const $db: usize = $lb * $lb;
        const $sa: usize = if $la < 2 {
            0
        } else {
            let pairs = $la * ($la - 1) / 2;
            pairs * pairs
        };
        const $sb: usize = if $lb < 2 {
            0
        } else {
            let pairs = $lb * ($lb - 1) / 2;
            pairs * pairs
        };
        $kernel
    }};
    (
        @match $ranks:expr,
        |
            $rxa:ident, $rwa:ident, $la:ident,
            $rxb:ident, $rwb:ident, $lb:ident,
            $da:ident, $db:ident, $sa:ident, $sb:ident
        | $kernel:expr,
        $fallback:expr;
        $(
            (
                $rxa_value:literal, $rwa_value:literal,
                $rxb_value:literal, $rwb_value:literal
            )
        ),+ $(,)?
    ) => {{
        match $ranks {
            $(
                ($rxa_value, $rwa_value, $rxb_value, $rwb_value) => {
                    dispatch_hamiltonian_ranks!(
                        @rank (
                            $rxa_value, $rwa_value,
                            $rxb_value, $rwb_value
                        ),
                        |
                            $rxa, $rwa, $la, $rxb, $rwb, $lb,
                            $da, $db, $sa, $sb
                        | $kernel
                    )
                }
            )+
            _ => $fallback,
        }
    }};
    (
        $ranks:expr,
        |
            $rxa:ident, $rwa:ident, $la:ident,
            $rxb:ident, $rwb:ident, $lb:ident,
            $da:ident, $db:ident, $sa:ident, $sb:ident
        | $kernel:expr,
        $fallback:expr $(,)?
    ) => {{
        dispatch_hamiltonian_ranks!(
            @match $ranks,
            |
                $rxa, $rwa, $la, $rxb, $rwb, $lb,
                $da, $db, $sa, $sb
            | $kernel,
            $fallback;
            (0, 0, 0, 0),
            (0, 0, 0, 1),
            (0, 0, 0, 2),
            (0, 0, 0, 3),
            (0, 0, 0, 4),
            (0, 0, 1, 0),
            (0, 0, 1, 1),
            (0, 0, 1, 2),
            (0, 0, 1, 3),
            (0, 0, 1, 4),
            (0, 0, 2, 0),
            (0, 0, 2, 1),
            (0, 0, 2, 2),
            (0, 0, 2, 3),
            (0, 0, 2, 4),
            (0, 0, 3, 0),
            (0, 0, 3, 1),
            (0, 0, 3, 2),
            (0, 0, 3, 3),
            (0, 0, 4, 0),
            (0, 0, 4, 1),
            (0, 0, 4, 2),
            (0, 1, 0, 0),
            (0, 1, 0, 1),
            (0, 1, 0, 2),
            (0, 1, 0, 3),
            (0, 1, 0, 4),
            (0, 1, 1, 0),
            (0, 1, 1, 1),
            (0, 1, 1, 2),
            (0, 1, 1, 3),
            (0, 1, 1, 4),
            (0, 1, 2, 0),
            (0, 1, 2, 1),
            (0, 1, 2, 2),
            (0, 1, 2, 3),
            (0, 1, 3, 0),
            (0, 1, 3, 1),
            (0, 1, 3, 2),
            (0, 1, 4, 0),
            (0, 1, 4, 1),
            (0, 2, 0, 0),
            (0, 2, 0, 1),
            (0, 2, 0, 2),
            (0, 2, 0, 3),
            (0, 2, 0, 4),
            (0, 2, 1, 0),
            (0, 2, 1, 1),
            (0, 2, 1, 2),
            (0, 2, 1, 3),
            (0, 2, 2, 0),
            (0, 2, 2, 1),
            (0, 2, 2, 2),
            (0, 2, 3, 0),
            (0, 2, 3, 1),
            (0, 2, 4, 0),
            (0, 3, 0, 0),
            (0, 3, 0, 1),
            (0, 3, 0, 2),
            (0, 3, 0, 3),
            (0, 3, 1, 0),
            (0, 3, 1, 1),
            (0, 3, 1, 2),
            (0, 3, 2, 0),
            (0, 3, 2, 1),
            (0, 3, 3, 0),
            (0, 4, 0, 0),
            (0, 4, 0, 1),
            (0, 4, 0, 2),
            (0, 4, 1, 0),
            (0, 4, 1, 1),
            (0, 4, 2, 0),
            (1, 0, 0, 0),
            (1, 0, 0, 1),
            (1, 0, 0, 2),
            (1, 0, 0, 3),
            (1, 0, 0, 4),
            (1, 0, 1, 0),
            (1, 0, 1, 1),
            (1, 0, 1, 2),
            (1, 0, 1, 3),
            (1, 0, 1, 4),
            (1, 0, 2, 0),
            (1, 0, 2, 1),
            (1, 0, 2, 2),
            (1, 0, 2, 3),
            (1, 0, 3, 0),
            (1, 0, 3, 1),
            (1, 0, 3, 2),
            (1, 0, 4, 0),
            (1, 0, 4, 1),
            (1, 1, 0, 0),
            (1, 1, 0, 1),
            (1, 1, 0, 2),
            (1, 1, 0, 3),
            (1, 1, 0, 4),
            (1, 1, 1, 0),
            (1, 1, 1, 1),
            (1, 1, 1, 2),
            (1, 1, 1, 3),
            (1, 1, 2, 0),
            (1, 1, 2, 1),
            (1, 1, 2, 2),
            (1, 1, 3, 0),
            (1, 1, 3, 1),
            (1, 1, 4, 0),
            (1, 2, 0, 0),
            (1, 2, 0, 1),
            (1, 2, 0, 2),
            (1, 2, 0, 3),
            (1, 2, 1, 0),
            (1, 2, 1, 1),
            (1, 2, 1, 2),
            (1, 2, 2, 0),
            (1, 2, 2, 1),
            (1, 2, 3, 0),
            (1, 3, 0, 0),
            (1, 3, 0, 1),
            (1, 3, 0, 2),
            (1, 3, 1, 0),
            (1, 3, 1, 1),
            (1, 3, 2, 0),
            (1, 4, 0, 0),
            (1, 4, 0, 1),
            (1, 4, 1, 0),
            (2, 0, 0, 0),
            (2, 0, 0, 1),
            (2, 0, 0, 2),
            (2, 0, 0, 3),
            (2, 0, 0, 4),
            (2, 0, 1, 0),
            (2, 0, 1, 1),
            (2, 0, 1, 2),
            (2, 0, 1, 3),
            (2, 0, 2, 0),
            (2, 0, 2, 1),
            (2, 0, 2, 2),
            (2, 0, 3, 0),
            (2, 0, 3, 1),
            (2, 0, 4, 0),
            (2, 1, 0, 0),
            (2, 1, 0, 1),
            (2, 1, 0, 2),
            (2, 1, 0, 3),
            (2, 1, 1, 0),
            (2, 1, 1, 1),
            (2, 1, 1, 2),
            (2, 1, 2, 0),
            (2, 1, 2, 1),
            (2, 1, 3, 0),
            (2, 2, 0, 0),
            (2, 2, 0, 1),
            (2, 2, 0, 2),
            (2, 2, 1, 0),
            (2, 2, 1, 1),
            (2, 2, 2, 0),
            (2, 3, 0, 0),
            (2, 3, 0, 1),
            (2, 3, 1, 0),
            (2, 4, 0, 0),
            (3, 0, 0, 0),
            (3, 0, 0, 1),
            (3, 0, 0, 2),
            (3, 0, 0, 3),
            (3, 0, 1, 0),
            (3, 0, 1, 1),
            (3, 0, 1, 2),
            (3, 0, 2, 0),
            (3, 0, 2, 1),
            (3, 0, 3, 0),
            (3, 1, 0, 0),
            (3, 1, 0, 1),
            (3, 1, 0, 2),
            (3, 1, 1, 0),
            (3, 1, 1, 1),
            (3, 1, 2, 0),
            (3, 2, 0, 0),
            (3, 2, 0, 1),
            (3, 2, 1, 0),
            (3, 3, 0, 0),
            (4, 0, 0, 0),
            (4, 0, 0, 1),
            (4, 0, 0, 2),
            (4, 0, 1, 0),
            (4, 0, 1, 1),
            (4, 0, 2, 0),
            (4, 1, 0, 0),
            (4, 1, 0, 1),
            (4, 1, 1, 0),
            (4, 2, 0, 0),
        )
    }};
}

pub(super) use dispatch_hamiltonian_ranks;
pub(super) use dispatch_onebody_ranks;
pub(super) use dispatch_overlap_ranks;
#[cfg(feature = "nocc")]
pub(super) use dispatch_rdm_ranks;
