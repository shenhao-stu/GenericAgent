//! render/viewport.rs — the scroll [`ScrollAnchor`] + visible-window derivation
//! (checklist §3, P1 — "Resize never corrupts scroll"). This is the half of P1
//! that keeps the user's position fixed across width changes; [`crate::render::measure`]
//! is the half that makes the reflow cheap.
//!
//! THE INVARIANT (zero drift): scroll position is stored as a *logical*
//! coordinate that is independent of width:
//!   * [`ScrollAnchor::Bottom`] — follow the tail. Survives content growth AND
//!     resize because it is not "anchored to the last visual line" (which a
//!     reflow would move) but a distinct mode meaning "always show the end".
//!   * [`ScrollAnchor::Anchored`] `{block_id, intra}` — pin the TOP row of the
//!     viewport to a specific wrapped line of a specific block. A width change
//!     re-derives the visual top from the SAME `(block_id, intra)` via the wrap
//!     cache, so the same logical line stays at the top. No visual row number is
//!     ever carried across a resize → no drift, no smear, no jump.
//!
//! Every operation works in two steps: resolve the anchor → a visual top via the
//! cache, mutate in visual space, then re-derive the anchor from the new top.
//! "Resize re-derives, it does not re-track." The whole module is pure over a
//! [`WrapCache`] + a viewport height, so the headline `resize_then_scroll_no_drift`
//! test runs headlessly with no terminal.

use crate::render::block::BlockId;
use crate::render::measure::{VisualLine, WrapCache};

/// Where the viewport is scrolled to, stored LOGICALLY (width-independent).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScrollAnchor {
    /// Follow mode: always show the tail. The common live-streaming case.
    #[default]
    Bottom,
    /// Pinned mode: the top visual row of the viewport is `(block_id, intra)`.
    Anchored { block_id: BlockId, intra: usize },
}

/// The scrolling viewport over a transcript. Owns the logical [`ScrollAnchor`]
/// and the viewport height (rows of the transcript region). The wrap cache is
/// owned by the caller and passed in, so the same cache feeds rendering and the
/// other regions.
#[derive(Debug, Clone)]
pub struct Viewport {
    anchor: ScrollAnchor,
    /// Visible rows of the transcript region (chrome already subtracted).
    height: usize,
}

impl Default for Viewport {
    fn default() -> Self {
        Viewport {
            anchor: ScrollAnchor::Bottom,
            height: 1,
        }
    }
}

impl Viewport {
    /// A new viewport of `height` rows, in follow (`Bottom`) mode.
    pub fn new(height: usize) -> Self {
        Viewport {
            anchor: ScrollAnchor::Bottom,
            height: height.max(1),
        }
    }

    /// The current logical anchor (used by the drift tests + a future scroll-bar
    /// indicator).
    #[allow(dead_code)]
    pub fn anchor(&self) -> ScrollAnchor {
        self.anchor
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn is_following(&self) -> bool {
        matches!(self.anchor, ScrollAnchor::Bottom)
    }

    /// The maximum valid top row: total visual lines minus a screenful, floored
    /// at 0 (a transcript shorter than the viewport pins to top).
    fn max_top(&self, cache: &WrapCache) -> usize {
        cache.total_visual_lines().saturating_sub(self.height)
    }

    fn clamp_top(&self, top: usize, cache: &WrapCache) -> usize {
        top.min(self.max_top(cache))
    }

    /// Resolve the current logical anchor to a visual top row under `cache`.
    /// `Bottom` maps to `max_top` (the tail screenful). `Anchored` maps to the
    /// cached global index of `(block_id, intra)`, clamped; if the anchored block
    /// no longer exists, fall back to the bottom (robust to a cleared transcript).
    pub fn visual_top(&self, cache: &WrapCache) -> usize {
        match self.anchor {
            ScrollAnchor::Bottom => self.max_top(cache),
            ScrollAnchor::Anchored { block_id, intra } => match cache.locate(block_id, intra) {
                Some(top) => self.clamp_top(top, cache),
                None => self.max_top(cache),
            },
        }
    }

    /// Re-derive the logical anchor from a (clamped) visual top. If the top is at
    /// or past `max_top`, we re-enter `Bottom` so follow resumes naturally;
    /// otherwise we anchor to the `(block_id, intra)` owning that row.
    fn anchor_from_top(&mut self, top: usize, cache: &WrapCache) {
        let max_top = self.max_top(cache);
        if top >= max_top {
            self.anchor = ScrollAnchor::Bottom;
            return;
        }
        match cache.anchor_at(top) {
            Some((block_id, intra)) => {
                self.anchor = ScrollAnchor::Anchored { block_id, intra };
            }
            None => self.anchor = ScrollAnchor::Bottom,
        }
    }

    /// Scroll by `delta` visual rows (positive = down / toward the tail, negative
    /// = up / toward history). Clamps at both ends; reaching the bottom re-enters
    /// follow mode.
    pub fn scroll_by(&mut self, delta: isize, cache: &WrapCache) {
        let top = self.visual_top(cache) as isize;
        let new_top = (top + delta).max(0) as usize;
        let new_top = self.clamp_top(new_top, cache);
        self.anchor_from_top(new_top, cache);
    }

    /// Scroll up one row (mouse wheel up is typically 3 of these). The app wires
    /// the wheel via `scroll_by(±WHEEL_STEP)`; this single-row helper is used by
    /// the resize/scroll regression tests and any future arrow-key binding.
    #[allow(dead_code)]
    pub fn scroll_up(&mut self, cache: &WrapCache) {
        self.scroll_by(-1, cache);
    }

    /// Scroll down one row (see [`Viewport::scroll_up`]).
    #[allow(dead_code)]
    pub fn scroll_down(&mut self, cache: &WrapCache) {
        self.scroll_by(1, cache);
    }

    /// Page up — keep one line of context (less/lazygit convention).
    pub fn page_up(&mut self, cache: &WrapCache) {
        let step = self.height.saturating_sub(1).max(1) as isize;
        self.scroll_by(-step, cache);
    }

    /// Page down — keep one line of context.
    pub fn page_down(&mut self, cache: &WrapCache) {
        let step = self.height.saturating_sub(1).max(1) as isize;
        self.scroll_by(step, cache);
    }

    /// Home — jump to the very top (global row 0). If the whole transcript fits
    /// the viewport (max_top == 0) the top and the tail coincide, so we stay in
    /// follow mode; otherwise we explicitly anchor at the first visual row.
    pub fn home(&mut self, cache: &WrapCache) {
        if self.max_top(cache) == 0 {
            self.anchor = ScrollAnchor::Bottom;
            return;
        }
        match cache.anchor_at(0) {
            Some((block_id, intra)) => {
                self.anchor = ScrollAnchor::Anchored { block_id, intra };
            }
            None => self.anchor = ScrollAnchor::Bottom,
        }
    }

    /// End — jump to the tail and re-enter follow mode.
    pub fn end(&mut self) {
        self.anchor = ScrollAnchor::Bottom;
    }

    /// RESIZE — the heart of P1. The new viewport height is applied, the caller
    /// has already rewidthed `cache` to the new width, and we re-derive the
    /// visual top from the SAME logical anchor. Because the anchor is logical
    /// (`Bottom` or `(block_id, intra)`), the same content stays in view with
    /// ZERO drift; we only clamp so a now-shorter transcript can't scroll past
    /// the end (in which case we re-enter follow mode).
    ///
    /// Returns `true` (a "force a clean repaint" signal for the immediate-mode
    /// caller) — resize always warrants a full clear so no stale-geometry row can
    /// survive. (ratatui already repaints from state each frame; this mirrors the
    /// recon's "one clear + one repaint per settled resize".)
    pub fn resize(&mut self, new_height: usize, cache: &WrapCache) -> bool {
        self.height = new_height.max(1);
        // The logical anchor is unchanged and width-independent. We only resolve
        // it to a visual top under the NEW geometry and re-clamp so a transcript
        // that is now shorter than the viewport (fewer wraps at a wider width)
        // cannot scroll past the end — `anchor_from_top` re-enters Bottom in that
        // case, and otherwise preserves the exact `(block_id, intra)` pin.
        if matches!(self.anchor, ScrollAnchor::Bottom) {
            // Follow stays follow; nothing to re-derive.
            return true;
        }
        let clamped = self.clamp_top(self.visual_top(cache), cache);
        self.anchor_from_top(clamped, cache);
        true
    }

    /// Re-anchor so the row `(block_id, intra)` sits `screen_offset` rows below the
    /// viewport top, then re-derive the logical anchor from that top (Fix E / Q8). A
    /// fold toggle changes a block's row count; calling this with the CLICKED node's
    /// `(block_id, intra)` and the screen offset the user clicked at keeps that node
    /// visually fixed — so expanding a node above the viewport does NOT jump the view
    /// (the failure the spec calls out), and the anchor lands on the clicked node, not
    /// `Bottom`. Clamped at both ends; if the node is gone the anchor is left as-is.
    /// Resolve against the cache the toggle was applied to (the next frame re-syncs).
    pub fn anchor_node_at_offset(
        &mut self,
        block_id: BlockId,
        intra: usize,
        screen_offset: usize,
        cache: &WrapCache,
    ) {
        let Some(node_row) = cache.locate(block_id, intra) else {
            return;
        };
        let top = node_row.saturating_sub(screen_offset);
        let top = self.clamp_top(top, cache);
        self.anchor_from_top(top, cache);
    }

    /// The visible window: exactly the `[top, top + height)` slice of visual
    /// lines at the current width. Virtualized — cost is O(height), not
    /// O(transcript). The caller pads any short tail with blank rows so the grid
    /// is fully painted (no ghosts).
    pub fn visible(&self, cache: &WrapCache) -> Vec<VisualLine> {
        let top = self.visual_top(cache);
        cache.window(top, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::block::{Block, BlockId, BlockRole};

    /// A tall, wide transcript engineered to stress anchor drift:
    ///   * block 1 is multi-line, so block 2 does NOT start at global row 0 (i.e.
    ///     anchoring at block 2 is a genuine "scrolled into the middle" state),
    ///   * block 2 is one long line that wraps to a DIFFERENT number of rows at
    ///     each test width (the thing that would drift a visual-row anchor),
    ///   * blocks 3..=8 give enough trailing height that block 2's start stays
    ///     comfortably above `max_top` at every tested width — so it remains
    ///     `Anchored`, never collapsing into `Bottom`, no matter the geometry.
    /// Every assistant body begins with a unique sentinel so we can assert WHICH
    /// logical line sits at the viewport top.
    fn wide_blocks() -> Vec<Block> {
        let mut v = vec![
            Block::finalized(1, BlockRole::User, "USER first\nUSER second\nUSER third\nUSER fourth"),
            Block::finalized(
                2,
                BlockRole::Assistant,
                "ASSISTANT alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho",
            ),
        ];
        for id in 3..=8u64 {
            v.push(Block::finalized(
                id,
                BlockRole::Assistant,
                &format!("TAIL{id} keeps the transcript tall so block 2 stays mid-scroll and above max_top across widths"),
            ));
        }
        v
    }

    fn cache_at(width: u16, blocks: &[Block]) -> WrapCache {
        let mut c = WrapCache::new(width);
        c.sync(blocks);
        c
    }

    // ---- THE HEADLINE TEST: resize never drifts the scroll position ---------

    #[test]
    fn resize_then_scroll_no_drift() {
        let blocks = wide_blocks();
        let target_block: BlockId = 2; // the long, re-wrapping assistant line.

        // Start wide and scroll so the viewport TOP is block 2's first row — a
        // genuine mid-transcript anchored position (content above and below it).
        let mut cache = cache_at(120, &blocks);
        let mut vp = Viewport::new(4);
        let want_top = cache.locate(target_block, 0).unwrap();
        let max_top = cache.total_visual_lines().saturating_sub(vp.height());
        assert!(want_top > 0, "block 2 must not start at row 0 (real scroll)");
        assert!(want_top < max_top, "block 2 must sit above max_top so it stays anchored");
        // From Bottom (top == max_top), scroll UP to put block 2's first row at top.
        vp.scroll_by(want_top as isize - max_top as isize, &cache);

        // We are anchored exactly at block 2 / intra 0.
        assert_eq!(
            vp.anchor(),
            ScrollAnchor::Anchored { block_id: target_block, intra: 0 },
            "precondition: anchored at the start of block 2"
        );
        let top_sentinel = vp.visible(&cache)[0].text.clone();
        assert!(top_sentinel.starts_with("ASSISTANT"), "top is block 2, got {top_sentinel:?}");

        // The TRUE no-drift assertion, exercised across a sweep of widths: after
        // each resize the SAME logical line `(block 2, intra 0)` is still pinned
        // to the viewport top — even though block 2 wraps to a different number
        // of visual rows at each width. A visual-row anchor would drift here.
        for &w in &[40u16, 30, 80, 200, 52, 120] {
            cache.rewidth(w, &blocks);
            vp.resize(4, &cache);
            assert_eq!(
                vp.anchor(),
                ScrollAnchor::Anchored { block_id: target_block, intra: 0 },
                "anchor drifted at width {w}"
            );
            let top = vp.visible(&cache)[0].text.clone();
            assert!(
                top.starts_with("ASSISTANT"),
                "at width {w} the top row must still be the start of block 2, got {top:?}"
            );
            // The resolved visual top maps back to the same block.
            assert_eq!(cache.anchor_at(vp.visual_top(&cache)).unwrap().0, target_block);
        }

        // After the width churn, SCROLL up and down many times and assert the
        // emitted window is ALWAYS a contiguous, strictly-increasing slice — the
        // single invariant that catches duplicated/overlapping/jumped rows (C1).
        for _ in 0..15 {
            vp.scroll_up(&cache);
            assert_contiguous_in_order(&vp, &cache);
        }
        for _ in 0..30 {
            vp.scroll_down(&cache);
            assert_contiguous_in_order(&vp, &cache);
        }
        // Scrolled all the way down → back in follow mode, tail visible.
        assert!(vp.is_following());
    }

    /// Assert the visible window is exactly a contiguous, strictly-increasing
    /// (in global visual index) slice — the single invariant that catches
    /// duplicated/overlapping/jumped rows after a resize.
    fn assert_contiguous_in_order(vp: &Viewport, cache: &WrapCache) {
        let win = vp.visible(cache);
        let mut prev: Option<usize> = None;
        for vl in &win {
            let gi = cache.locate(vl.block_id, vl.intra).unwrap();
            if let Some(p) = prev {
                assert_eq!(gi, p + 1, "rows must be contiguous (no gap/dup): {p} → {gi}");
            }
            prev = Some(gi);
        }
        // Never exceeds a screenful.
        assert!(win.len() <= vp.height());
    }

    #[test]
    fn bottom_mode_is_sticky_across_resize_and_append() {
        let mut blocks = wide_blocks();
        let mut cache = cache_at(120, &blocks);
        let mut vp = Viewport::new(3);
        // Default is Bottom; the last visible row is the true last line.
        assert!(vp.is_following());
        let last_text = "TAIL8"; // sentinel of the final block in wide_blocks()
        assert!(vp.visible(&cache).iter().any(|v| v.text.contains(last_text)));

        // Resize to any size → still pinned to the tail.
        cache.rewidth(30, &blocks);
        assert!(vp.resize(5, &cache));
        assert!(vp.is_following());
        let win = vp.visible(&cache);
        // The very last reflowed visual line is visible at the bottom of the window.
        let total = cache.total_visual_lines();
        let last_gi = cache.locate(win.last().unwrap().block_id, win.last().unwrap().intra).unwrap();
        assert_eq!(last_gi, total - 1, "bottom mode must show the true last row");

        // Append more content (a new block, id 99 to avoid colliding with the
        // TAIL3..=8 ids) → still follows the new tail.
        blocks.push(Block::finalized(99, BlockRole::Assistant, "brand new tail line"));
        cache.sync(&blocks);
        assert!(vp.is_following());
        assert!(vp.visible(&cache).iter().any(|v| v.text.contains("brand new tail")));
    }

    #[test]
    fn anchored_mode_frozen_across_append() {
        let mut blocks = wide_blocks();
        let mut cache = cache_at(80, &blocks);
        let mut vp = Viewport::new(3);
        // Anchor to the top of block 1 (the very first row).
        vp.home(&cache);
        let top_before = vp.visual_top(&cache);
        let row_before = vp.visible(&cache)[0].text.clone();

        // Append 3 new blocks BELOW the anchor.
        for id in 10..13u64 {
            blocks.push(Block::finalized(id, BlockRole::Assistant, "appended below"));
        }
        cache.sync(&blocks);

        // The top row is unchanged (no auto-scroll while anchored).
        assert_eq!(vp.visual_top(&cache), top_before);
        assert_eq!(vp.visible(&cache)[0].text, row_before);
    }

    #[test]
    fn page_up_then_page_down_returns_to_bottom_no_drift() {
        // Build a tall transcript so paging has room.
        let mut blocks = Vec::new();
        for id in 0..40u64 {
            blocks.push(Block::finalized(id, BlockRole::Assistant, &format!("line {id}")));
        }
        let cache = cache_at(80, &blocks);
        let mut vp = Viewport::new(10);

        assert!(vp.is_following());
        vp.page_up(&cache);
        assert!(!vp.is_following(), "page up leaves follow mode");
        // Page down enough times to return to the bottom → follow mode again.
        vp.page_down(&cache);
        vp.page_down(&cache);
        assert!(vp.is_following(), "paging back to the tail re-enters follow mode");
    }

    #[test]
    fn scroll_up_one_leaves_bottom_then_back_re_enters() {
        let mut blocks = Vec::new();
        for id in 0..20u64 {
            blocks.push(Block::finalized(id, BlockRole::Assistant, &format!("row {id}")));
        }
        let cache = cache_at(80, &blocks);
        let mut vp = Viewport::new(5);
        assert!(vp.is_following());
        vp.scroll_up(&cache);
        assert!(!vp.is_following());
        vp.scroll_down(&cache);
        assert!(vp.is_following(), "scrolling back to the last line re-enters follow");
    }

    #[test]
    fn clamp_on_shrink_to_tiny_transcript() {
        // A transcript shorter than the viewport: top clamps to 0, no negatives.
        let blocks = vec![Block::finalized(1, BlockRole::Assistant, "only line")];
        let cache = cache_at(80, &blocks);
        let mut vp = Viewport::new(10);
        assert_eq!(vp.visual_top(&cache), 0);
        vp.scroll_up(&cache); // can't go above 0
        assert_eq!(vp.visual_top(&cache), 0);
        vp.page_up(&cache);
        assert_eq!(vp.visual_top(&cache), 0);
        let win = vp.visible(&cache);
        assert_eq!(win.len(), 1); // exactly the one row (caller pads the rest)
    }

    #[test]
    fn empty_transcript_is_safe() {
        let blocks: Vec<Block> = Vec::new();
        let cache = cache_at(80, &blocks);
        let mut vp = Viewport::new(5);
        assert_eq!(vp.visual_top(&cache), 0);
        assert!(vp.visible(&cache).is_empty());
        vp.scroll_up(&cache);
        vp.page_down(&cache);
        vp.home(&cache);
        vp.end();
        assert!(vp.is_following());
    }
}
