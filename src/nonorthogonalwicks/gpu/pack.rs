// nonorthogonalwicks/gpu/pack.rs
//! Pack CPU nonorthogonal Wick intermediates into compact GPU storage.

// Crate-root imports.
use crate::noci::NOCIScalar;
use crate::nonorthogonalwicks::WicksRequirements;
use crate::nonorthogonalwicks::cpu;

// Parent/sibling imports.
use super::storage::WicksShared;
use super::types::{PairMeta, PairOffset, SameSpinMeta, SameSpinOffset};

/// Pack the CPU Wick data required by factorised NOCI-PT2 into compact GPU storage.
/// For one-body factorisation this uploads only `X`, `Y`, current-Fock `ff`, scalar `f0f`,
/// `phase`, `tilde_s_prod`, `m`, `nmo` and `nocc` for same-spin alpha and beta sectors.
/// # Arguments:
/// - `cpu`: CPU Wick view containing the current same-spin intermediates.
/// - `requirements`: Wick capability set requested by the GPU consumer.
/// # Returns
/// - `WicksShared<T>`: Compact GPU Wick storage.
pub(crate) fn pack_wicks<T: NOCIScalar>(
    cpu: &cpu::WicksView<T>,
    requirements: WicksRequirements,
) -> WicksShared<T> {
    if !requirements.overlap || !requirements.one_body || requirements.two_body || requirements.rdm
    {
        eprintln!(
            "GPU Wick packing currently supports only overlap + one_body requirements for NOCI-PT2"
        );
        std::process::exit(1);
    }

    let nref = cpu.nref();
    let mut slab = Vec::new();
    let mut off = Vec::with_capacity(nref * nref);
    let mut meta = Vec::with_capacity(nref * nref);

    for lp in 0..nref {
        for gp in 0..nref {
            let pair = cpu.pair(lp, gp);
            let (aa_off, aa_meta) = pack_same_spin(&pair.aa, &mut slab);
            let (bb_off, bb_meta) = pack_same_spin(&pair.bb, &mut slab);
            off.push(PairOffset {
                aa: aa_off,
                bb: bb_off,
            });
            meta.push(PairMeta {
                aa: aa_meta,
                bb: bb_meta,
            });
        }
    }

    WicksShared::new(requirements, slab, nref, off, meta)
}

/// Pack one same-spin CPU Wick view into a compact same-spin GPU slab segment.
/// # Arguments:
/// - `source`: CPU same-spin Wick view.
/// - `slab`: Compact GPU slab being appended to.
/// # Returns
/// - `(SameSpinOffset, SameSpinMeta<T>)`: Offsets and scalar metadata for the packed sector.
fn pack_same_spin<T: NOCIScalar>(
    source: &cpu::view::SameSpinView<'_, T>,
    slab: &mut Vec<T>,
) -> (SameSpinOffset, SameSpinMeta<T>) {
    let mut off = SameSpinOffset::default();
    for mi in 0..2 {
        off.x[mi] = push_slice(slab, source.x_slice(mi));
        off.y[mi] = push_slice(slab, source.y_slice(mi));
    }
    for mi in 0..2 {
        for mj in 0..2 {
            off.ff[mi][mj] = push_slice(slab, source.ff_t_slice(mi, mj));
        }
    }
    let meta = SameSpinMeta {
        tilde_s_prod: source.tilde_s_prod,
        phase: source.phase,
        m: source.m,
        nmo: source.nmo,
        nocc: source.nocc,
        f0f: source.f0f,
    };
    (off, meta)
}

/// Append a contiguous CPU slice to the compact GPU slab.
/// # Arguments:
/// - `slab`: Compact GPU slab being appended to.
/// - `values`: Source values to append.
/// # Returns
/// - `usize`: Offset of `values` in `slab`.
fn push_slice<T: NOCIScalar>(
    slab: &mut Vec<T>,
    values: &[T],
) -> usize {
    let off = slab.len();
    slab.extend_from_slice(values);
    off
}
