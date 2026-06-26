// timers/wick.rs

use super::{Counter, with_totals};

/// Wick contraction timing totals.
#[derive(Clone, Copy, Debug, Default)]
pub struct Totals {
    /// Time spent interning Wick delta factors.
    pub store_delta: Counter,
    /// Time spent interning Wick tensor factors.
    pub store_tensor: Counter,
    /// Time spent expanding spin-free groups into spin strings.
    pub spin: Counter,
    /// Time spent constructing Wick blocks.
    pub blocks: Counter,
    /// Time spent enumerating frozen-space contractions.
    pub frozen: Counter,
    /// Time spent enumerating active-space cumulant blocks.
    pub active: Counter,
    /// Time spent evaluating Wick block values.
    pub val: Counter,
    /// Time spent looking up spin projections.
    pub proj: Counter,
    /// Time spent constructing the spin-projection table.
    pub ptab: Counter,
    /// Time spent constructing individual spin projections.
    pub pval: Counter,
    /// Time spent extracting spin bits.
    pub bit: Counter,
    /// Time spent packing spin labels.
    pub sbits: Counter,
    /// Time spent enumerating exact-cover suffixes.
    pub walk: Counter,
    /// Time spent choosing exact-cover pivots.
    pub pick: Counter,
    /// Time spent accumulating numeric Wick rows.
    pub add: Counter,
    /// Time spent sorting numeric factor ids.
    pub sortids: Counter,
    /// Time spent decoding numeric Wick rows.
    pub out: Counter,
    /// Time spent constructing numeric Wick rows.
    pub row: Counter,
    /// Time spent extracting mask bit positions.
    pub bits: Counter,
    /// Time spent constructing position masks.
    pub mask: Counter,
    /// Time spent constructing group masks.
    pub gmask: Counter,
    /// Time spent checking completed connectivity.
    pub conn: Counter,
    /// Time spent computing Wick crossing signs.
    pub cross: Counter,
    /// Time spent normal ordering active operators.
    pub norm: Counter,
    /// Time spent in the public non-connected evaluator.
    pub eval: Counter,
    /// Time spent in the public connected evaluator.
    pub evalc: Counter,
    /// Time spent reading the spin-batch setting.
    pub spinbatch: Counter,
    /// Time spent reading the spin-parallelism setting.
    pub spinpar: Counter,
    /// Time spent reading the stream-queue setting.
    pub streamqueue: Counter,
    /// Time spent reading the accumulator-flush setting.
    pub accflush: Counter,
    /// Time spent evaluating one streamed connected spin string.
    pub eval1cstream: Counter,
    /// Time spent in the connected streaming wrapper.
    pub evalcstream: Counter,
    /// Time spent in the non-connected evaluator implementation.
    pub eval0: Counter,
    /// Time spent evaluating one non-connected spin string.
    pub eval1: Counter,
    /// Time spent in the connected evaluator implementation.
    pub evalc0: Counter,
    /// Time spent evaluating one connected spin string.
    pub eval1c: Counter,
    /// Time spent in connected recursive enumeration.
    pub walkc: Counter,
    /// Time spent joining connected prefixes and suffixes.
    pub joinrow: Counter,
    /// Time spent finding the root-connected component.
    pub rootseen: Counter,
    /// Time spent checking whether a partial contraction remains connectable.
    pub canconnect: Counter,
    /// Time spent checking one candidate connected block.
    pub canconnect1: Counter,
}

impl Totals {
    /// Add another set of Wick timings into this one.
    /// # Arguments:
    /// - `other`: Wick timings to accumulate.
    /// # Returns:
    /// - `()`: Updates this timing collection in place.
    #[inline(always)]
    pub fn merge_from(
        &mut self,
        other: &Totals,
    ) {
        self.store_delta.merge_from(&other.store_delta);
        self.store_tensor.merge_from(&other.store_tensor);
        self.spin.merge_from(&other.spin);
        self.blocks.merge_from(&other.blocks);
        self.frozen.merge_from(&other.frozen);
        self.active.merge_from(&other.active);
        self.val.merge_from(&other.val);
        self.proj.merge_from(&other.proj);
        self.ptab.merge_from(&other.ptab);
        self.pval.merge_from(&other.pval);
        self.bit.merge_from(&other.bit);
        self.sbits.merge_from(&other.sbits);
        self.walk.merge_from(&other.walk);
        self.pick.merge_from(&other.pick);
        self.add.merge_from(&other.add);
        self.sortids.merge_from(&other.sortids);
        self.out.merge_from(&other.out);
        self.row.merge_from(&other.row);
        self.bits.merge_from(&other.bits);
        self.mask.merge_from(&other.mask);
        self.gmask.merge_from(&other.gmask);
        self.conn.merge_from(&other.conn);
        self.cross.merge_from(&other.cross);
        self.norm.merge_from(&other.norm);
        self.eval.merge_from(&other.eval);
        self.evalc.merge_from(&other.evalc);
        self.spinbatch.merge_from(&other.spinbatch);
        self.spinpar.merge_from(&other.spinpar);
        self.streamqueue.merge_from(&other.streamqueue);
        self.accflush.merge_from(&other.accflush);
        self.eval1cstream.merge_from(&other.eval1cstream);
        self.evalcstream.merge_from(&other.evalcstream);
        self.eval0.merge_from(&other.eval0);
        self.eval1.merge_from(&other.eval1);
        self.evalc0.merge_from(&other.evalc0);
        self.eval1c.merge_from(&other.eval1c);
        self.walkc.merge_from(&other.walkc);
        self.joinrow.merge_from(&other.joinrow);
        self.rootseen.merge_from(&other.rootseen);
        self.canconnect.merge_from(&other.canconnect);
        self.canconnect1.merge_from(&other.canconnect1);
    }
}

macro_rules! add_counter {
    ($name:ident, $field:ident, $label:literal) => {
        #[doc = concat!("Add one ", $label, " timing.")]
        #[doc = "# Arguments:"]
        #[doc = "- `ns`: Elapsed nanoseconds."]
        #[doc = "# Returns:"]
        #[doc = "- `()`: Updates the current thread-local counter."]
        #[inline(always)]
        pub fn $name(ns: u64) {
            with_totals(|totals| totals.wick.$field.add_ns(ns));
        }
    };
}

add_counter!(add_store_delta, store_delta, "Wick delta interning");
add_counter!(add_store_tensor, store_tensor, "Wick tensor interning");
add_counter!(add_spin, spin, "spin-string expansion");
add_counter!(add_blocks, blocks, "Wick-block construction");
add_counter!(add_frozen, frozen, "frozen-contraction enumeration");
add_counter!(add_active, active, "active-cumulant enumeration");
add_counter!(add_val, val, "Wick-block evaluation");
add_counter!(add_proj, proj, "spin-projection lookup");
add_counter!(add_ptab, ptab, "spin-projection table construction");
add_counter!(add_pval, pval, "individual spin-projection construction");
add_counter!(add_bit, bit, "spin-bit extraction");
add_counter!(add_sbits, sbits, "spin-label packing");
add_counter!(add_walk, walk, "exact-cover suffix enumeration");
add_counter!(add_pick, pick, "exact-cover pivot selection");
add_counter!(add_add, add, "numeric Wick-row accumulation");
add_counter!(add_sortids, sortids, "numeric factor-id sorting");
add_counter!(add_out, out, "numeric Wick-row decoding");
add_counter!(add_row, row, "numeric Wick-row construction");
add_counter!(add_bits, bits, "mask-bit extraction");
add_counter!(add_mask, mask, "position-mask construction");
add_counter!(add_gmask, gmask, "group-mask construction");
add_counter!(add_conn, conn, "completed-connectivity checking");
add_counter!(add_cross, cross, "Wick crossing-sign evaluation");
add_counter!(add_norm, norm, "active-operator normal ordering");
add_counter!(add_eval, eval, "public non-connected evaluation");
add_counter!(add_evalc, evalc, "public connected evaluation");
add_counter!(add_spinbatch, spinbatch, "spin-batch setting lookup");
add_counter!(add_spinpar, spinpar, "spin-parallelism setting lookup");
add_counter!(add_streamqueue, streamqueue, "stream-queue setting lookup");
add_counter!(add_accflush, accflush, "accumulator-flush setting lookup");
add_counter!(
    add_eval1cstream,
    eval1cstream,
    "streamed connected spin-string evaluation"
);
add_counter!(
    add_evalcstream,
    evalcstream,
    "connected streaming-wrapper evaluation"
);
add_counter!(add_eval0, eval0, "non-connected evaluator implementation");
add_counter!(add_eval1, eval1, "non-connected spin-string evaluation");
add_counter!(add_evalc0, evalc0, "connected evaluator implementation");
add_counter!(add_eval1c, eval1c, "connected spin-string evaluation");
add_counter!(add_walkc, walkc, "connected recursive enumeration");
add_counter!(add_joinrow, joinrow, "connected prefix-suffix joining");
add_counter!(add_rootseen, rootseen, "root-component construction");
add_counter!(add_canconnect, canconnect, "partial-connectivity checking");
add_counter!(
    add_canconnect1,
    canconnect1,
    "candidate-connectivity checking"
);
