//! app/fold_hit.rs — the per-node fold STATE + clickable-node hit-testing surface
//! (Fix E / Q8), split out of `app/mod.rs` for cohesion. Rebuilds the global
//! row→node hit table each sync, resolves a transcript left-click on the
//! triangle/bullet gutter to a fold node, and toggles per-node / global folds.

use crate::render::fold::{BlockFolds, NodeId};
use crate::theme::Theme;

use super::AppState;

/// How many leading cells of a transcript row are the fold "triangle/bullet column"
/// (Fix E / Q8): a left-click within the first this-many cells of a clickable node's
/// row toggles its fold; a click past it flows through to native selection. The `▸`
/// fold header + the `⏺`/`○` tool bullet both sit in cell 0, so 2 cells covers the
/// glyph + its trailing space.
const FOLD_HIT_COLS: u16 = 2;

impl AppState {
    /// Rebuild the clickable-node hit table (Fix E / Q8) from each assistant block's
    /// per-node INTRA ranges, lifted to GLOBAL visual indices via the just-synced
    /// wrap cache (so a row range here indexes the same visual rows the transcript
    /// draws this frame). PURE-ish (reads the memo, writes `node_hit`).
    pub(in crate::app) fn rebuild_node_hit(&mut self, theme: &Theme, fold_all: bool, width: u16) {
        let mut out: Vec<(std::ops::RangeInclusive<usize>, NodeId)> = Vec::new();
        for b in &self.transcript {
            if b.render_role() != crate::render::BlockRole::Assistant {
                continue;
            }
            let folds = BlockFolds { block_id: b.id, fold_all, overrides: Some(&self.folds) };
            for (range, node) in b.cockpit_node_hits(theme, &folds, width) {
                // Lift the block-local intra range to global visual rows. `locate`
                // clamps intra into the block, so a transient skew can't panic; skip
                // a range whose block is (briefly) absent from the cache.
                let (Some(start), Some(end)) = (
                    self.wrap_cache.locate(b.id, *range.start()),
                    self.wrap_cache.locate(b.id, *range.end()),
                ) else {
                    continue;
                };
                out.push((start..=end, node));
            }
        }
        self.node_hit = out;
    }

    /// Resolve a transcript left-click at full-frame `(col, row)` to the fold node
    /// whose triangle/bullet column it hit (Fix E / Q8), or `None`. The clickable
    /// zone is the FIRST [`FOLD_HIT_COLS`] cells of a node's visual rows (the `▸`/`⏺`
    /// gutter) — a click in the body flows through to native selection. `row` is
    /// mapped to a global visual index via the transcript region top + the viewport
    /// top, mirroring [`crate::components::dashboard::click_to_row_index`]. PURE-ish
    /// (reads `node_hit` + the viewport).
    pub fn transcript_node_at(&self, col: u16, row: u16, transcript_top: u16) -> Option<NodeId> {
        if col >= FOLD_HIT_COLS || row < transcript_top {
            return None;
        }
        let offset = (row - transcript_top) as usize;
        // A click below the transcript region (in the composer / footer rows) is not
        // a transcript click — the visible window is exactly `viewport.height()` rows.
        if offset >= self.viewport.height() {
            return None;
        }
        let visual = self.viewport.visual_top(&self.wrap_cache) + offset;
        self.node_hit
            .iter()
            .find(|(range, _)| range.contains(&visual))
            .map(|(_, node)| *node)
    }

    /// Handle a left-click at full-frame `(col, row)` (transcript region top
    /// `transcript_top`): if it lands on a fold node's triangle/bullet column, toggle
    /// that node and RE-DERIVE the scroll anchor on the clicked node so the toggled
    /// content stays visually fixed (an expand above the viewport doesn't jump the
    /// view, Fix E / Q8). Returns `true` if a node was toggled (the caller skips its
    /// other click handling), `false` to fall through. The wrap cache is re-synced on
    /// the next `prepare_frame`; the logical `(block, intra)` anchor we set here is
    /// resolved against the NEW geometry then, so it stays correct after the reflow.
    pub fn click_fold_at(&mut self, col: u16, row: u16, transcript_top: u16) -> bool {
        let Some(node) = self.transcript_node_at(col, row, transcript_top) else {
            return false;
        };
        // Capture the clicked node's CURRENT anchor (its first visual row → a stable
        // `(block, intra)`) and the screen offset it sits at, BEFORE the toggle moves
        // rows around — so we can re-pin it to the same screen position after.
        let visual_top = self.viewport.visual_top(&self.wrap_cache);
        let clicked_visual = visual_top + (row.saturating_sub(transcript_top)) as usize;
        let node_first = self
            .node_hit
            .iter()
            .find(|(r, n)| *n == node && r.contains(&clicked_visual))
            .map(|(r, _)| *r.start());
        let pin = node_first.and_then(|first| {
            self.wrap_cache
                .anchor_at(first)
                .map(|(block, intra)| (block, intra, first.saturating_sub(visual_top)))
        });

        self.toggle_fold(node);

        // Re-anchor on the clicked node at the SAME screen offset (not Bottom). The
        // cache still reflects the pre-toggle geometry here; the next frame's sync
        // re-resolves this logical anchor under the new row counts (zero jump, P1).
        if let Some((block, intra, screen_offset)) = pin {
            self.viewport
                .anchor_node_at_offset(block, intra, screen_offset, &self.wrap_cache);
        }
        true
    }

    /// Toggle one node's fold (Fix E / Q8): flip its override in [`AppState::folds`]
    /// (a turn folds/unfolds; a tool result expands/collapses), defaulting from the
    /// node's CURRENT effective fold so the first click always visibly flips it. The
    /// click handler then re-derives the scroll anchor on the clicked node so an
    /// expand above the viewport doesn't jump the view (done by the caller). PURE-ish.
    pub fn toggle_fold(&mut self, node: NodeId) {
        let now = self.node_is_folded(node);
        self.folds.insert(node, !now);
        // A fold toggle changes a block's PROJECTED row count without bumping its
        // `rev` (rev is the streaming-delta key), so the `(block_id, rev)`-keyed wrap
        // cache would otherwise reuse the stale row count. Bumping the fold epoch makes
        // the next `sync_transcript` reflow every block from the new projection so the
        // wrap cache + the cockpit memo agree on the new geometry (P1).
        self.fold_epoch = self.fold_epoch.wrapping_add(1);
    }

    /// Flip the GLOBAL fold-all (the `/fold` command / `Ctrl+Shift+O` chord, Q8):
    /// fold every completed turn, or unfold them all. A clean reset — it CLEARS the
    /// per-node overrides so "fold all" / "unfold all" wins over any prior per-node
    /// toggles — and bumps the fold epoch so the wrap cache reflows. Keeps the
    /// existing global-fold behavior intact while making it compose with Fix E.
    pub fn toggle_fold_all(&mut self) {
        self.fold_all = !self.fold_all;
        self.folds.clear();
        self.fold_epoch = self.fold_epoch.wrapping_add(1);
    }

    /// The CURRENT effective fold of a node (the override if present, else the
    /// default: a tool result is collapsed; a turn folds unless it's the last AND
    /// `fold_all` is off). Used to seed [`AppState::toggle_fold`] so the first click
    /// flips the VISIBLE state. PURE.
    pub(in crate::app) fn node_is_folded(&self, node: NodeId) -> bool {
        if let Some(&v) = self.folds.get(&node) {
            return v;
        }
        match node {
            // A tool result defaults to collapsed (`folds` stores `true` = expanded),
            // so its effective "folded/collapsed" default is `false`-expanded == not
            // expanded → return `false` so the first toggle sets it expanded (`true`).
            NodeId::Tool { .. } => false,
            // A turn folds by default UNLESS it is the last turn of its block and
            // `fold_all` is off. Determine "is last" from the block's turn markers.
            NodeId::Turn { block, turn } => {
                let is_last = self.turn_is_last(block, turn);
                crate::render::fold::default_turn_folded(is_last, self.fold_all)
            }
        }
    }

    /// Whether `turn` is the LAST turn marker of block `block` (so the default fold
    /// keeps it open). PURE-ish (scans the block's source once).
    pub(in crate::app) fn turn_is_last(&self, block: u64, turn: u32) -> bool {
        self.transcript
            .iter()
            .find(|b| b.id == block)
            .map(|b| crate::render::fold::last_turn_number(&b.source) == Some(turn))
            .unwrap_or(false)
    }
}
