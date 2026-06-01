//! render/block.rs — the logical transcript [`Block`]: the ONLY authoritative,
//! width-independent unit of truth in the render plane (checklist §3, P1/P2).
//!
//! A `Block` stores its **SOURCE** text verbatim (`{id, role, source, rev}`).
//! Everything visual — soft-wrapped rows, the viewport window — is *derived*
//! from `(block, width)` by [`crate::render::measure`] and can be thrown away and
//! rebuilt at any width. Two consequences fall out by construction:
//!
//!   * **P1 (resize never corrupts scroll):** scroll is anchored to a logical
//!     `(block_id, intra)` coordinate, never a visual row, so a width change
//!     re-derives the window from the same logical anchor with ZERO drift.
//!   * **P2 (copy yields clean text):** copy reads `block.source` (the logical
//!     string) — it never reassembles rendered rows, so a soft-wrap can never
//!     become an embedded `\n`.
//!
//! `rev` is a monotonic content version bumped on every mutation; the wrap cache
//! keyed `(block_id, width)` stores the `rev` it was computed at and recomputes a
//! block only when its `rev` moves (streaming reflows O(1 block), not the whole
//! transcript). This module is pure (no I/O, no ratatui) and unit-tested.

/// The author of a transcript block. Mirrors `app::Role` at the protocol edge
/// but lives here so the render plane has no upward dependency on `app`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockRole {
    User,
    Assistant,
    System,
    Tool,
    /// Out-of-band notice (bridge errors, child exit, stderr) — never silent.
    Notice,
}

/// A stable, monotonic block id. NEVER reused, so a [`crate::render::viewport::ScrollAnchor`]
/// that points at one survives appends/splices to the transcript (fixes the
/// "scroll jumps to a random message after resize" failure mode).
pub type BlockId = u64;

/// One logical transcript block. The source of truth; visual rows are derived.
///
/// `finalized` is part of the model (a renderer drops the streaming caret once a
/// block finalizes — wired in the streaming-caret polish); kept here so the
/// logical model is complete. `streaming`/`stream_append`/`finalize` are the
/// explicit deliverable API (checklist §3 "finalize/stream append"); the app
/// currently mutates its own `app::Block` and snapshots into this, so these
/// constructors are exercised by the unit tests and available to a future direct
/// owner.
#[allow(dead_code)] // `finalized` field + streaming API: model completeness / spec.
#[derive(Debug, Clone)]
pub struct Block {
    /// Stable id (monotonic, never reused).
    pub id: BlockId,
    /// Who authored it (drives gutter glyph + theme token at render time).
    pub role: BlockRole,
    /// The verbatim logical text. Copy (P2) reads THIS, never rendered rows.
    pub source: String,
    /// Monotonic content version, bumped on every mutation. The wrap cache uses
    /// it to invalidate exactly the blocks that changed (streaming → O(1)).
    pub rev: u64,
    /// True once the block is finalized (`MessageEnd`, or a synchronous block).
    /// Streaming blocks are `false` until then; a renderer may show a caret.
    pub finalized: bool,
    /// Per-HARD-LINE atomic (never-soft-wrap-split) byte ranges, indexed by
    /// hard-line (same order as [`Block::hard_lines`]); each range is in that hard
    /// line's LOCAL byte coordinates. Empty (the common case) ⇒ ordinary wrapping.
    ///
    /// Set by the assistant projection ([`crate::app::Block::to_render_block`]) so
    /// the wrap cache keeps a rendered inline-math glyph run intact, matching the
    /// styled draw row-for-row (P1, the parity invariant). For non-assistant /
    /// math-free blocks it stays empty and wrapping is unchanged.
    pub atomic_ranges: Vec<Vec<std::ops::Range<usize>>>,
}

#[allow(dead_code)] // streaming-construction API is the spec deliverable (tested).
impl Block {
    /// A finalized block with the given source (a synchronous user/notice line).
    pub fn finalized(id: BlockId, role: BlockRole, source: impl Into<String>) -> Self {
        Block {
            id,
            role,
            source: source.into(),
            rev: 1,
            finalized: true,
            atomic_ranges: Vec::new(),
        }
    }

    /// An empty, streaming block (a freshly-begun assistant message). Deltas are
    /// appended via [`Block::stream_append`]; [`Block::finalize`] closes it.
    pub fn streaming(id: BlockId, role: BlockRole) -> Self {
        Block {
            id,
            role,
            source: String::new(),
            rev: 1,
            finalized: false,
            atomic_ranges: Vec::new(),
        }
    }

    /// Attach per-hard-line atomic (no-soft-wrap-split) byte ranges, then return
    /// self (builder style). See [`Block::atomic_ranges`].
    pub fn with_atomic_ranges(mut self, ranges: Vec<Vec<std::ops::Range<usize>>>) -> Self {
        self.atomic_ranges = ranges;
        self
    }

    /// Append a streamed delta to the source and bump `rev` so the wrap cache
    /// reflows only this block. No-op for an empty chunk (keeps `rev` stable so
    /// untouched-block cache reuse is not needlessly invalidated).
    pub fn stream_append(&mut self, chunk: &str) {
        if chunk.is_empty() {
            return;
        }
        self.source.push_str(chunk);
        self.rev = self.rev.wrapping_add(1);
    }

    /// Mark the block finalized. Bumps `rev` (the trailing-state changed: a
    /// renderer may drop the streaming caret), so the final frame reflows it.
    pub fn finalize(&mut self) {
        if !self.finalized {
            self.finalized = true;
            self.rev = self.rev.wrapping_add(1);
        }
    }

    /// The source split into **hard** logical lines (on `\n`), each with NO
    /// embedded newline. Soft-wrapping in [`crate::render::measure`] then only
    /// worries about fitting one newline-free logical line to a width — cleanly
    /// separating "the author put a line break here" from "the terminal is
    /// narrow". An empty source yields one empty hard line (so a blank block
    /// still occupies exactly one visual row — protects bottom-pinning math).
    ///
    /// A trailing `\n` does NOT create a phantom extra line beyond the natural
    /// empty segment `split` already yields, matching authored intent.
    pub fn hard_lines(&self) -> Vec<&str> {
        if self.source.is_empty() {
            return vec![""];
        }
        self.source.split('\n').collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finalized_block_is_closed_with_source() {
        let b = Block::finalized(7, BlockRole::User, "hello");
        assert_eq!(b.id, 7);
        assert!(b.finalized);
        assert_eq!(b.source, "hello");
        assert_eq!(b.role, BlockRole::User);
    }

    #[test]
    fn streaming_appends_assemble_source_and_bump_rev() {
        let mut b = Block::streaming(1, BlockRole::Assistant);
        assert!(!b.finalized);
        let r0 = b.rev;
        b.stream_append("Hello ");
        b.stream_append("世界");
        assert_eq!(b.source, "Hello 世界");
        assert!(b.rev > r0, "rev must advance on append");
        // An empty append is a no-op that does NOT churn rev.
        let r1 = b.rev;
        b.stream_append("");
        assert_eq!(b.rev, r1);
    }

    #[test]
    fn finalize_is_idempotent_and_bumps_rev_once() {
        let mut b = Block::streaming(2, BlockRole::Assistant);
        b.stream_append("done");
        let before = b.rev;
        b.finalize();
        assert!(b.finalized);
        let after = b.rev;
        assert!(after > before);
        // Second finalize is a no-op (no further rev churn).
        b.finalize();
        assert_eq!(b.rev, after);
    }

    #[test]
    fn hard_lines_split_on_newline_without_phantom() {
        // Empty source → exactly one (empty) hard line, not zero.
        let empty = Block::streaming(3, BlockRole::Assistant);
        assert_eq!(empty.hard_lines(), vec![""]);

        // Two authored lines.
        let two = Block::finalized(4, BlockRole::Assistant, "a\nb");
        assert_eq!(two.hard_lines(), vec!["a", "b"]);

        // A trailing newline yields the natural trailing empty segment (one),
        // not two phantom blanks.
        let trail = Block::finalized(5, BlockRole::Assistant, "x\n");
        assert_eq!(trail.hard_lines(), vec!["x", ""]);

        // A leading/interior blank line is preserved as an empty segment.
        let blank = Block::finalized(6, BlockRole::Assistant, "a\n\nb");
        assert_eq!(blank.hard_lines(), vec!["a", "", "b"]);
    }
}
