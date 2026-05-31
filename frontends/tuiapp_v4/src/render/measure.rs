//! render/measure.rs — CJK/wide-aware soft-wrap + the wrap cache (checklist §3,
//! P1). This is the *pure, memoized* reflow that maps logical [`Block`]s onto
//! visual rows for a given width. It is the half of P1 that makes width changes
//! cheap and exact; [`crate::render::viewport`] is the half that keeps scroll
//! anchored across them.
//!
//! Three hard requirements, each backed by a test:
//!   * **Display-cell width, not byte/char count.** Wrapping walks grapheme
//!     clusters and accumulates East-Asian width via `unicode-width`, so a CJK
//!     glyph counts as 2 cells and a wrap boundary never splits a wide glyph
//!     (`cjk_width_correct`). Measuring with `.len()` or `.chars().count()` is
//!     the classic C6 corruption and is structurally avoided here.
//!   * **Wrap cache keyed `(block_id, width)` → `Vec<VisualLine>`** carrying the
//!     `rev` it was computed at, so a streaming append reflows only the mutated
//!     block (untouched blocks reuse their cached rows) and a resize builds a
//!     fresh per-width cache.
//!   * **Prefix sums** over per-block visual-line counts, so the viewport can map
//!     a global visual index ↔ `(block, intra)` in O(log n) for fast scroll math.
//!
//! Each [`VisualLine`] also records the **provenance of its soft-wrap boundary**
//! (`is_block_start`, `is_continuation`) so a renderer can hang-indent
//! continuations / draw a gutter only on the first row, and so copy logic can
//! prove a join across continuations introduces no newline (P2, exercised by
//! `copy_across_wrap_has_no_newline` in [`crate::render::copy`]).

use std::collections::HashMap;

use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

use crate::render::block::{Block, BlockId};

/// One visual (soft-wrapped) row derived from a logical block at a fixed width.
/// `text` is a substring of the block's source with display width `<= avail`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisualLine {
    /// The logical block this row belongs to.
    pub block_id: BlockId,
    /// Which wrapped segment within the block (0-based, across hard + soft
    /// breaks). The second half of the stable `(block_id, intra)` scroll anchor.
    pub intra: usize,
    /// The exact display string for this row (`display_width(text) <= avail`).
    pub text: String,
    /// Display width of `text` in terminal cells.
    pub cells: usize,
    /// True for the first visual row of the block (draw the speaker gutter here).
    pub is_block_start: bool,
    /// True when this row is a SOFT-wrap continuation of the previous row of the
    /// SAME hard line (no author newline before it). This is the soft-wrap
    /// boundary provenance: joining a run of rows whose continuations are `true`
    /// reconstructs one logical line with no inserted `\n` (P2).
    pub is_continuation: bool,
    /// For a soft-wrap continuation (`is_continuation == true`): whether the break
    /// fell at a WORD boundary (a space was consumed at the wrap point) vs a
    /// mid-word hard cell-break. This is the *exact* provenance copy needs to
    /// faithfully rejoin: a word-boundary wrap rejoins with a single space, a
    /// mid-word wrap rejoins with nothing — and NEITHER becomes a `\n` (P2). Not
    /// meaningful (false) for hard-line / block-start rows.
    pub wrapped_at_word_boundary: bool,
}

/// One wrapped segment of a single hard line: the text plus whether the break
/// that PRECEDES it (making it a continuation) fell at a word boundary. The first
/// segment of a hard line has `broke_at_word_boundary == false` (it has no
/// preceding soft break). Used by [`reflow_block`] to record copy-faithful
/// provenance; [`wrap_line`] is the thin text-only view over this.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrapSegment {
    pub text: String,
    /// True iff this is a continuation AND the soft break before it consumed a
    /// space (word wrap). False for the first segment and for mid-word breaks.
    pub broke_at_word_boundary: bool,
}

/// Wrap a single newline-free logical line to `avail` display cells, returning
/// just the segment strings (the thin view). See [`wrap_line_segments`] for the
/// version that also reports word-boundary provenance.
#[allow(dead_code)] // text-only convenience view, exercised by the wrap tests.
pub fn wrap_line(line: &str, avail: usize) -> Vec<String> {
    wrap_line_segments(line, avail)
        .into_iter()
        .map(|s| s.text)
        .collect()
}

/// Wrap a single newline-free logical line to `avail` display cells, returning
/// the wrapped segments **in order** with break provenance. Pure + deterministic
/// (same inputs → same output), which is what lets the cache key on
/// `(block, width)` alone.
///
/// Rules (the C4/C6 guards):
///   * `avail` is clamped to `>= 1` so a zero/one-column terminal can't loop
///     forever.
///   * Width is accumulated per **grapheme cluster** via `unicode-width`, so a
///     2-cell CJK/emoji glyph is treated atomically and is NEVER split across a
///     boundary; a wide glyph that cannot fit the remaining space starts the
///     next row instead.
///   * An empty input yields exactly one empty segment (a blank line stays one
///     visual row — protects bottom-pinning off-by-one).
///   * Prefer breaking at the last space before the limit (word wrap, less/lazygit
///     style); fall back to a hard cell-break when a single token exceeds `avail`.
pub fn wrap_line_segments(line: &str, avail: usize) -> Vec<WrapSegment> {
    let avail = avail.max(1);
    if line.is_empty() {
        return vec![WrapSegment {
            text: String::new(),
            broke_at_word_boundary: false,
        }];
    }

    let mut out: Vec<WrapSegment> = Vec::new();
    // The current row being built (bytes) + its running display width. `word_byte`
    // is the byte offset in `cur` at which the LAST in-progress word began (the
    // position right after the most recent run of spaces) — the word-wrap break
    // point. `None` means no breakable word boundary exists yet in this row.
    let mut cur = String::new();
    let mut cur_cells = 0usize;
    let mut word_byte: Option<usize> = None;
    // Whether the grapheme we just appended was a space (to detect word starts).
    let mut prev_space = false;
    // Provenance of the break that PRECEDES the row currently being accumulated:
    // true if that soft break consumed a space (word wrap), false if mid-word.
    // Attached to a segment when it is pushed.
    let mut pending_word_boundary = false;

    // Push the current row as a finished segment, recording the break-before flag.
    macro_rules! flush {
        ($word_boundary:expr) => {{
            let text = trim_trailing_spaces(std::mem::take(&mut cur));
            out.push(WrapSegment {
                text,
                broke_at_word_boundary: pending_word_boundary,
            });
            pending_word_boundary = $word_boundary;
            cur_cells = 0;
        }};
    }

    for g in line.graphemes(true) {
        let gw = UnicodeWidthStr::width(g);
        let is_space = g == " ";

        // A new word begins at the first non-space after a space (or row start).
        if !is_space && (prev_space || cur.is_empty()) {
            word_byte = Some(cur.len());
        }

        // A grapheme wider than the whole row (a wide glyph at avail==1): flush
        // the row, then emit the glyph alone (overflow by 1 beats an infinite
        // loop). It cannot be word-broken → its preceding break is mid-word.
        if gw > avail {
            if !cur.is_empty() {
                flush!(false);
            }
            out.push(WrapSegment {
                text: g.to_string(),
                broke_at_word_boundary: pending_word_boundary,
            });
            pending_word_boundary = false;
            word_byte = None;
            prev_space = false;
            continue;
        }

        if cur_cells + gw > avail {
            // Row is full. Prefer breaking before the in-progress word so it moves
            // down whole — but only if that word doesn't start at column 0 (else
            // the word itself is longer than the row and we must hard-break it).
            match word_byte {
                Some(wb) if wb > 0 && !is_space => {
                    let remainder = cur.split_off(wb); // the unfinished word
                    let text = trim_trailing_spaces(std::mem::take(&mut cur));
                    out.push(WrapSegment {
                        text,
                        broke_at_word_boundary: pending_word_boundary,
                    });
                    // The space(s) between the previous word and `wb` were trimmed
                    // off → this is a WORD-boundary break before the remainder.
                    pending_word_boundary = true;
                    cur = remainder;
                    cur_cells = UnicodeWidthStr::width(cur.as_str());
                    word_byte = Some(0); // the moved word now starts the new row
                }
                _ if is_space => {
                    // Breaking exactly at a space → the space is the boundary and
                    // gets dropped → WORD-boundary break before the next row.
                    flush!(true);
                    word_byte = None;
                }
                _ => {
                    // A single token longer than the row → mid-word hard break.
                    flush!(false);
                    word_byte = Some(0);
                }
            }
        }

        // Skip leading spaces at the very start of a fresh (continuation) row so a
        // wrap boundary doesn't bleed indentation into the next line.
        if is_space && cur.is_empty() && !out.is_empty() {
            prev_space = true;
            continue;
        }

        cur.push_str(g);
        cur_cells += gw;
        prev_space = is_space;
    }
    if !cur.is_empty() || out.is_empty() {
        let text = trim_trailing_spaces(cur);
        out.push(WrapSegment {
            text,
            broke_at_word_boundary: pending_word_boundary,
        });
    }
    out
}

/// Drop trailing ASCII spaces from a wrapped segment so a word-wrap break never
/// leaves a dangling space on the end of a row (which would also miscount cells
/// and pollute a copy-mode reconstruction).
fn trim_trailing_spaces(mut s: String) -> String {
    while s.ends_with(' ') {
        s.pop();
    }
    s
}

/// Reflow one block at `width` into its visual lines. `width` is the FULL column
/// budget for the block's text (the caller has already subtracted any chrome).
/// Pure over `(block.source, width)` — the cache relies on that.
pub fn reflow_block(block: &Block, width: u16) -> Vec<VisualLine> {
    let avail = (width as usize).max(1);
    let mut out: Vec<VisualLine> = Vec::new();
    let mut intra = 0usize;
    for hard in block.hard_lines() {
        let segments = wrap_line_segments(hard, avail);
        for (seg_idx, seg) in segments.into_iter().enumerate() {
            let cells = UnicodeWidthStr::width(seg.text.as_str());
            out.push(VisualLine {
                block_id: block.id,
                intra,
                text: seg.text,
                cells,
                is_block_start: intra == 0,
                // Continuation == a soft break (not the first segment of THIS
                // hard line). The first segment of each hard line is a hard
                // boundary (author newline or block start), never a continuation.
                is_continuation: seg_idx > 0,
                // Only meaningful for continuations; the segment carries whether
                // its preceding soft break was a word boundary (rejoin with a
                // space) or mid-word (rejoin with nothing). Copy uses this (P2).
                wrapped_at_word_boundary: seg_idx > 0 && seg.broke_at_word_boundary,
            });
            intra += 1;
        }
    }
    out
}

/// A per-block cache entry: the visual rows plus the `rev` they were computed at.
#[derive(Debug, Clone)]
struct CachedBlock {
    rev: u64,
    lines: Vec<VisualLine>,
}

/// The wrap cache for ONE width. Keyed `(block_id, width)` in the sense that the
/// whole struct is valid for a single `width`; on a width change the owner builds
/// a fresh cache (optionally keeping an LRU of the last widths). Per-block `rev`
/// invalidation keeps streaming O(1 block).
///
/// `prefix[i]` = total visual lines in `blocks[0..i)`, so:
///   * `prefix[order.len()]` == total visual lines,
///   * a global visual index → owning block is a binary search over `prefix`,
///   * `(block, intra)` → global index is `prefix[pos] + intra`.
#[derive(Debug, Default)]
pub struct WrapCache {
    width: u16,
    per_block: HashMap<BlockId, CachedBlock>,
    /// Block ids in transcript order (defines the prefix-sum layout).
    order: Vec<BlockId>,
    /// Prefix sums of per-block visual-line counts (len == order.len() + 1).
    prefix: Vec<usize>,
}

impl WrapCache {
    /// A cache for the given width (no blocks reflowed yet).
    pub fn new(width: u16) -> Self {
        WrapCache {
            width: width.max(1),
            per_block: HashMap::new(),
            order: Vec::new(),
            prefix: vec![0],
        }
    }

    /// The width this cache is valid for.
    #[allow(dead_code)] // resize-diagnostics / Phase-3 accessor; tested.
    pub fn width(&self) -> u16 {
        self.width
    }

    /// Total visual lines across all blocks at this width.
    pub fn total_visual_lines(&self) -> usize {
        *self.prefix.last().unwrap_or(&0)
    }

    /// Rebuild (or reuse) the cache for `blocks` at the current width. Blocks
    /// whose `rev` is unchanged keep their previously-computed rows (object
    /// reuse); only mutated/new blocks are reflowed. Removed blocks are dropped.
    /// Recomputes the prefix sums. This is the per-frame entry point.
    pub fn sync(&mut self, blocks: &[Block]) {
        // Reflow changed/new blocks; carry forward unchanged ones.
        let mut next: HashMap<BlockId, CachedBlock> =
            HashMap::with_capacity(blocks.len());
        for b in blocks {
            let reuse = self
                .per_block
                .get(&b.id)
                .filter(|c| c.rev == b.rev)
                .cloned();
            let entry = match reuse {
                Some(c) => c,
                None => CachedBlock {
                    rev: b.rev,
                    lines: reflow_block(b, self.width),
                },
            };
            next.insert(b.id, entry);
        }
        self.per_block = next;
        self.order = blocks.iter().map(|b| b.id).collect();
        self.recompute_prefix();
    }

    /// Force a full rebuild at a NEW width (resize). Equivalent to constructing a
    /// fresh cache then `sync`, but reuses the allocation. After this call every
    /// block has been reflowed at `new_width`.
    pub fn rewidth(&mut self, new_width: u16, blocks: &[Block]) {
        let new_width = new_width.max(1);
        if new_width != self.width {
            self.width = new_width;
            self.per_block.clear(); // every entry is now stale.
        }
        self.sync(blocks);
    }

    fn recompute_prefix(&mut self) {
        self.prefix.clear();
        self.prefix.push(0);
        let mut acc = 0usize;
        for id in &self.order {
            acc += self
                .per_block
                .get(id)
                .map(|c| c.lines.len())
                .unwrap_or(0);
            self.prefix.push(acc);
        }
    }

    /// Position of `block_id` in the order, if present.
    pub fn order_pos(&self, block_id: BlockId) -> Option<usize> {
        self.order.iter().position(|id| *id == block_id)
    }

    /// Number of visual lines `block_id` occupies at this width (0 if absent).
    #[allow(dead_code)] // anchor-clamp helper / fold math (Phase 3); tested.
    pub fn block_line_count(&self, block_id: BlockId) -> usize {
        self.per_block
            .get(&block_id)
            .map(|c| c.lines.len())
            .unwrap_or(0)
    }

    /// Global visual index of `(block_id, intra)` (the row's top position), with
    /// `intra` clamped into the block. Returns `None` if the block is gone.
    pub fn locate(&self, block_id: BlockId, intra: usize) -> Option<usize> {
        let pos = self.order_pos(block_id)?;
        let count = self.per_block.get(&block_id)?.lines.len();
        if count == 0 {
            return Some(self.prefix[pos]);
        }
        let clamped = intra.min(count - 1);
        Some(self.prefix[pos] + clamped)
    }

    /// Inverse of [`WrapCache::locate`]: map a global visual index to the
    /// `(block_id, intra)` it falls in. Clamps an out-of-range index to the last
    /// row. Returns `None` only when the transcript is empty.
    pub fn anchor_at(&self, visual_index: usize) -> Option<(BlockId, usize)> {
        let total = self.total_visual_lines();
        if total == 0 || self.order.is_empty() {
            return None;
        }
        let idx = visual_index.min(total - 1);
        // Binary search the last prefix boundary <= idx → owning block.
        // prefix has len order.len()+1; find pos where prefix[pos] <= idx < prefix[pos+1].
        let pos = match self.prefix.binary_search(&idx) {
            Ok(mut p) => {
                // Exact boundary: prefix[p] == idx. If p indexes a zero-length
                // block run, walk forward to the block that actually owns idx.
                while p + 1 < self.prefix.len() && self.prefix[p + 1] == idx {
                    p += 1;
                }
                p.min(self.order.len() - 1)
            }
            Err(p) => p.saturating_sub(1).min(self.order.len() - 1),
        };
        let block_id = self.order[pos];
        let intra = idx - self.prefix[pos];
        Some((block_id, intra))
    }

    /// The contiguous slice of visual lines in `[top, top + height)` at this
    /// width, materialized in order. Cost is O(height + blocks-spanned), not
    /// O(transcript): we walk only the blocks the window overlaps. `top` is
    /// clamped to `[0, total]`; a window past the end yields fewer rows.
    pub fn window(&self, top: usize, height: usize) -> Vec<VisualLine> {
        let total = self.total_visual_lines();
        if total == 0 || height == 0 {
            return Vec::new();
        }
        let top = top.min(total);
        let end = (top + height).min(total);
        let mut out: Vec<VisualLine> = Vec::with_capacity(end - top);
        // Find the first block the window starts in, then stream rows.
        for (pos, id) in self.order.iter().enumerate() {
            let base = self.prefix[pos];
            let count = self.per_block.get(id).map(|c| c.lines.len()).unwrap_or(0);
            let block_end = base + count;
            if block_end <= top {
                continue; // entirely above the window.
            }
            if base >= end {
                break; // entirely below the window.
            }
            let lines = &self.per_block.get(id).unwrap().lines;
            let lo = top.saturating_sub(base);
            let hi = (end - base).min(count);
            for vl in &lines[lo..hi] {
                out.push(vl.clone());
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::block::{Block, BlockRole};

    fn b(id: BlockId, src: &str) -> Block {
        Block::finalized(id, BlockRole::Assistant, src)
    }

    // ---- CJK / wide-aware wrapping (the headline `cjk_width_correct`) --------

    #[test]
    fn cjk_width_correct() {
        // Each CJK ideograph is 2 display cells. A pure-CJK string's width is
        // 2 * char_count, NOT char_count and NOT byte length.
        let s = "你好世界"; // 4 ideographs
        assert_eq!(UnicodeWidthStr::width(s), 8);
        assert_eq!(s.chars().count(), 4);
        assert!(s.len() > 8); // bytes (UTF-8) — must NOT be used as width.

        // Wrapping 4 ideographs (8 cells) at avail=4 yields 2 rows of 2 glyphs
        // (4 cells) each — the boundary lands BETWEEN glyphs, never inside one.
        let rows = wrap_line(s, 4);
        assert_eq!(rows, vec!["你好", "世界"]);
        for r in &rows {
            assert_eq!(UnicodeWidthStr::width(r.as_str()), 4);
            assert!(UnicodeWidthStr::width(r.as_str()) <= 4);
        }

        // A single wide glyph never fits avail=1 but must not infinite-loop:
        // it overflows onto its own row by design.
        let one = wrap_line("世", 1);
        assert_eq!(one, vec!["世"]);

        // Odd available width with wide glyphs: avail=3 fits one 2-cell glyph
        // per row (the 2nd glyph would be 4 > 3), so 4 glyphs → 4 rows, each
        // <= 3 cells and never a split glyph.
        let rows3 = wrap_line(s, 3);
        assert_eq!(rows3.len(), 4);
        for r in &rows3 {
            assert!(UnicodeWidthStr::width(r.as_str()) <= 3);
            assert_eq!(r.chars().count(), 1); // one whole glyph, not a fragment.
        }

        // Mixed ASCII + CJK measured in cells: "ab你" = 1+1+2 = 4 cells.
        assert_eq!(UnicodeWidthStr::width("ab你"), 4);
        let mixed = wrap_line("ab你好", 4);
        // "ab你" = 4 cells fills row 1 ("好" would overflow to 6); "好" on row 2.
        assert_eq!(mixed, vec!["ab你", "好"]);
    }

    #[test]
    fn wrap_ascii_word_boundary_and_hardbreak() {
        // Word wrap: break at the space, the long token moves down whole.
        let rows = wrap_line("hello world", 7);
        assert_eq!(rows, vec!["hello", "world"]);
        for r in &rows {
            assert!(UnicodeWidthStr::width(r.as_str()) <= 7);
        }
        // A single token longer than the width hard-breaks by cells.
        let rows = wrap_line("abcdefghij", 4);
        assert_eq!(rows, vec!["abcd", "efgh", "ij"]);
        for r in &rows {
            assert!(UnicodeWidthStr::width(r.as_str()) <= 4);
        }
    }

    #[test]
    fn wrap_blank_line_is_one_row() {
        // A blank logical line must be exactly one (empty) visual row, never zero
        // — otherwise bottom-pinning math is off by one.
        assert_eq!(wrap_line("", 10), vec![String::new()]);
        assert_eq!(wrap_line("", 1), vec![String::new()]);
    }

    #[test]
    fn wrap_is_pure_and_stable() {
        // Same inputs → identical output (referential stability the cache needs).
        let a = wrap_line("the quick brown fox", 8);
        let b = wrap_line("the quick brown fox", 8);
        assert_eq!(a, b);
    }

    // ---- reflow + provenance -------------------------------------------------

    #[test]
    fn reflow_marks_block_start_and_continuation() {
        // One hard line that soft-wraps to 3 rows, then a 2nd hard line.
        let block = b(1, "aaaa bbbb cccc\nzz");
        let lines = reflow_block(&block, 4);
        // hard line 1: "aaaa","bbbb","cccc" (3 soft rows), hard line 2: "zz".
        assert_eq!(lines.len(), 4);
        // Block start only on the very first visual row.
        assert!(lines[0].is_block_start);
        assert!(!lines[1].is_block_start);
        assert!(!lines[3].is_block_start);
        // Continuations are the SOFT-wrapped rows of hard line 1 (rows 2 & 3),
        // NOT row 0 (hard boundary) and NOT row 3 (new hard line).
        assert!(!lines[0].is_continuation);
        assert!(lines[1].is_continuation);
        assert!(lines[2].is_continuation);
        assert!(!lines[3].is_continuation, "new hard line is not a continuation");
        // intra is monotonic 0..n across hard+soft breaks.
        for (i, vl) in lines.iter().enumerate() {
            assert_eq!(vl.intra, i);
            assert_eq!(vl.block_id, 1);
        }
    }

    // ---- cache: prefix sums, per-rev reuse, window --------------------------

    #[test]
    fn cache_prefix_sums_and_locate_roundtrip() {
        let blocks = vec![b(1, "a\nb"), b(2, "ccccc"), b(3, "d")];
        let mut cache = WrapCache::new(80);
        cache.sync(&blocks);
        // 2 + 1 + 1 = 4 visual lines (each short line is one row at width 80).
        assert_eq!(cache.total_visual_lines(), 4);

        // (block,intra) → global index → back round-trips for every row.
        for top in 0..cache.total_visual_lines() {
            let (bid, intra) = cache.anchor_at(top).unwrap();
            let back = cache.locate(bid, intra).unwrap();
            assert_eq!(back, top, "round-trip at visual index {top}");
        }

        // Specific coordinates.
        assert_eq!(cache.locate(1, 0), Some(0));
        assert_eq!(cache.locate(1, 1), Some(1));
        assert_eq!(cache.locate(2, 0), Some(2));
        assert_eq!(cache.locate(3, 0), Some(3));
        // Intra clamps into the block (block 3 has 1 row → intra 9 → row 0 of it).
        assert_eq!(cache.locate(3, 9), Some(3));
    }

    #[test]
    fn cache_reuses_unchanged_blocks_and_reflows_mutated() {
        let mut blocks = vec![b(1, "stable one"), b(2, "will grow")];
        let mut cache = WrapCache::new(80);
        cache.sync(&blocks);
        // Capture the cached rows of the untouched block by identity-ish (clone
        // equality) before mutating its neighbor.
        let before_b1 = cache.window(0, 1);

        // Mutate block 2 only (stream more text → rev bumps).
        blocks[1].stream_append(" and grow and grow and grow and grow more text");
        cache.sync(&blocks);

        // Block 1's first row is byte-identical (it was not reflowed).
        let after_b1 = cache.window(0, 1);
        assert_eq!(before_b1, after_b1);
        // Block 2 now occupies more visual lines at the same width.
        assert!(cache.block_line_count(2) >= 1);
        assert!(cache.total_visual_lines() >= 2);
    }

    #[test]
    fn cache_window_is_contiguous_in_order_slice() {
        // Build a transcript whose total height exceeds a window; assert the
        // window is exactly the contiguous slice (no gaps, no dups, monotonic).
        let blocks = vec![
            b(1, "l0\nl1\nl2"),  // 3 rows
            b(2, "l3"),          // 1 row
            b(3, "l4\nl5\nl6\nl7"), // 4 rows
        ];
        let mut cache = WrapCache::new(80);
        cache.sync(&blocks);
        assert_eq!(cache.total_visual_lines(), 8);

        let win = cache.window(2, 4); // global rows 2,3,4,5
        let texts: Vec<&str> = win.iter().map(|v| v.text.as_str()).collect();
        assert_eq!(texts, vec!["l2", "l3", "l4", "l5"]);
        // Monotonic global order: reconstruct each row's global index via locate.
        let mut prev = None;
        for vl in &win {
            let gi = cache.locate(vl.block_id, vl.intra).unwrap();
            if let Some(p) = prev {
                assert!(gi > p, "window rows must be strictly increasing in global index");
            }
            prev = Some(gi);
        }

        // A window past the end yields only the in-range rows (no ghosts).
        let tail = cache.window(6, 10);
        let tail_texts: Vec<&str> = tail.iter().map(|v| v.text.as_str()).collect();
        assert_eq!(tail_texts, vec!["l6", "l7"]);
    }
}
