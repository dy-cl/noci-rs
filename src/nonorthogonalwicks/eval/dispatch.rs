// nonorthogonalwicks/eval/dispatch.rs

/// Dispatch one same-spin rank pair to arm-local compile-time constants.
macro_rules! dispatch_pair_ranks {
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
        $(($rx_value:literal, $rw_value:literal)),* $(,)?
    ) => {{
        match $ranks {
            $(
                ($rx_value, $rw_value) => dispatch_pair_ranks!(
                    @rank ($rx_value, $rw_value),
                    |$rx, $rw, $l| $kernel
                ),
            )*
            _ => $fallback,
        }
    }};
}

/// Dispatch one two-spin Hamiltonian rank tuple to arm-local compile-time constants.
macro_rules! dispatch_hamiltonian_ranks_inner {
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
        ),* $(,)?
    ) => {{
        match $ranks {
            $(
                ($rxa_value, $rwa_value, $rxb_value, $rwb_value) => {
                    dispatch_hamiltonian_ranks_inner!(
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
            )*
            _ => $fallback,
        }
    }};
}

include!(concat!(env!("OUT_DIR"), "/dispatch.rs"));

#[cfg(feature = "nocc")]
/// Dispatch a scalar same-spin rank-`K` RDM and excitation-rank tuple.
macro_rules! dispatch_rdm_scalar_ranks {
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
            ranks => dispatch_overlap_scalar_ranks!(
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
            0 => dispatch_rdm_scalar_ranks!(
                @operator 0, $ranks, |$k, $rx, $rw, $l, $d| $kernel, $fallback
            ),
            1 => dispatch_rdm_scalar_ranks!(
                @operator 1, $ranks, |$k, $rx, $rw, $l, $d| $kernel, $fallback
            ),
            2 => dispatch_rdm_scalar_ranks!(
                @operator 2, $ranks, |$k, $rx, $rw, $l, $d| $kernel, $fallback
            ),
            3 => dispatch_rdm_scalar_ranks!(
                @operator 3, $ranks, |$k, $rx, $rw, $l, $d| $kernel, $fallback
            ),
            4 => dispatch_rdm_scalar_ranks!(
                @operator 4, $ranks, |$k, $rx, $rw, $l, $d| $kernel, $fallback
            ),
            _ => $fallback,
        }
    }};
}

#[cfg(feature = "nocc")]
/// Dispatch a SIMD same-spin rank-`K` RDM and excitation-rank tuple.
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
                @operator 0, $ranks, |$k, $rx, $rw, $l, $d| $kernel, $fallback
            ),
            1 => dispatch_rdm_ranks!(
                @operator 1, $ranks, |$k, $rx, $rw, $l, $d| $kernel, $fallback
            ),
            2 => dispatch_rdm_ranks!(
                @operator 2, $ranks, |$k, $rx, $rw, $l, $d| $kernel, $fallback
            ),
            3 => dispatch_rdm_ranks!(
                @operator 3, $ranks, |$k, $rx, $rw, $l, $d| $kernel, $fallback
            ),
            4 => dispatch_rdm_ranks!(
                @operator 4, $ranks, |$k, $rx, $rw, $l, $d| $kernel, $fallback
            ),
            _ => $fallback,
        }
    }};
}

pub(super) use dispatch_hamiltonian_ranks;
pub(super) use dispatch_hamiltonian_ranks_inner;
pub(super) use dispatch_hamiltonian_scalar_ranks;
pub(super) use dispatch_onebody_ranks;
pub(super) use dispatch_onebody_scalar_ranks;
pub(super) use dispatch_overlap_ranks;
pub(super) use dispatch_overlap_scalar_ranks;
pub(super) use dispatch_pair_ranks;
#[cfg(feature = "nocc")]
pub(super) use dispatch_rdm_ranks;
#[cfg(feature = "nocc")]
pub(super) use dispatch_rdm_scalar_ranks;
