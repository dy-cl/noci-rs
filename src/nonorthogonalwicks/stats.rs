// nonorthogonalwicks/stats.rs

#![allow(dead_code)]

#[cfg(feature = "wick-stats")]
use std::sync::{
    LazyLock,
    atomic::{AtomicU64, Ordering},
};

#[cfg(feature = "wick-stats")]
pub const MAX_L: usize = 8;
#[cfg(feature = "wick-stats")]
pub const MAX_M: usize = 8;

#[cfg(feature = "wick-stats")]
const NL: usize = MAX_L + 2; // last bin = overflow
#[cfg(feature = "wick-stats")]
const NM: usize = MAX_M + 2;

#[cfg(feature = "wick-stats")]
#[inline(always)]
fn bin(x: usize, max: usize) -> usize {
    x.min(max + 1)
}

#[cfg(feature = "wick-stats")]
#[inline(always)]
fn show(x: usize, max: usize) -> String {
    if x == max + 1 { format!(">{max}") } else { x.to_string() }
}

#[cfg(feature = "wick-stats")]
#[inline(always)]
fn idx2(l: usize, m: usize) -> usize {
    l * NM + m
}

#[cfg(feature = "wick-stats")]
#[inline(always)]
fn idx4(la: usize, lb: usize, ma: usize, mb: usize) -> usize {
    (((la * NL + lb) * NM + ma) * NM + mb)
}

#[cfg(feature = "wick-stats")]
pub static H2_SAME: LazyLock<Vec<AtomicU64>> =
    LazyLock::new(|| (0..(NL * NM)).map(|_| AtomicU64::new(0)).collect());

#[cfg(feature = "wick-stats")]
pub static H2_DIFF: LazyLock<Vec<AtomicU64>> =
    LazyLock::new(|| (0..(NL * NL * NM * NM)).map(|_| AtomicU64::new(0)).collect());

#[cfg(feature = "wick-stats")]
#[inline(always)]
pub fn bump_h2_same(l: usize, m: usize) {
    let l = bin(l, MAX_L);
    let m = bin(m, MAX_M);
    H2_SAME[idx2(l, m)].fetch_add(1, Ordering::Relaxed);
}

#[cfg(feature = "wick-stats")]
#[inline(always)]
pub fn bump_h2_diff(la: usize, lb: usize, ma: usize, mb: usize) {
    let la = bin(la, MAX_L);
    let lb = bin(lb, MAX_L);
    let ma = bin(ma, MAX_M);
    let mb = bin(mb, MAX_M);
    H2_DIFF[idx4(la, lb, ma, mb)].fetch_add(1, Ordering::Relaxed);
}

#[cfg(feature = "wick-stats")]
pub fn dump() {
    eprintln!("=== Wick h2 same counts ===");
    for l in 0..NL {
        for m in 0..NM {
            let c = H2_SAME[idx2(l, m)].load(Ordering::Relaxed);
            if c != 0 {
                eprintln!("  (l={}, m={}) -> {}", show(l, MAX_L), show(m, MAX_M), c);
            }
        }
    }

    eprintln!("=== Wick h2 diff counts ===");
    for la in 0..NL {
        for lb in 0..NL {
            for ma in 0..NM {
                for mb in 0..NM {
                    let c = H2_DIFF[idx4(la, lb, ma, mb)].load(Ordering::Relaxed);
                    if c != 0 {
                        eprintln!(
                            "  (la={}, lb={}, ma={}, mb={}) -> {}",
                            show(la, MAX_L),
                            show(lb, MAX_L),
                            show(ma, MAX_M),
                            show(mb, MAX_M),
                            c
                        );
                    }
                }
            }
        }
    }
}
