//! app/types.rs — the value types the app state is built from: the transcript
//! [`Block`], the author [`Role`], the connection lifecycle [`ConnStatus`], the
//! full-screen [`View`], the modal [`Overlay`] stack, the [`PendingAsk`] /
//! [`RenameState`] records, and the [`CostBreakdown`] for `/cost`. Pure data +
//! their pure derivations; the reducer/overlay/multi modules mutate them.

use std::cell::RefCell;
use std::ops::{Range, RangeInclusive};

use ratatui::text::Line;

use crate::app::effort::EffortSlider;
use crate::bridge::protocol::AskUserOption;
use crate::components::picker::{AskUserPicker, Picker};
use crate::render::fold::NodeId;
use crate::render::{Block as RenderBlock, BlockFolds, BlockRole};
use crate::theme::Theme;

/// The active modal OVERLAY stacked over the current view (§3 "overlay stack").
/// At most one is up at a time (the cockpit + dashboard are VIEWS; these are
/// transient modals dismissed with `Esc`). The slash dispatcher opens them; the
/// key handler routes keys into the active one before the composer sees them.
#[derive(Debug, Clone)]
pub enum Overlay {
    /// A reusable list picker (`/llm` `/theme` `/emoji` …). Carries the picker
    /// model + a snapshot of the pre-preview theme so a live-preview picker can
    /// REVERT on `Esc`.
    Picker {
        picker: Picker,
        theme_backup: Option<Theme>,
    },
    /// The unified ask_user card (single / multi / numeric).
    AskUser(AskUserPicker),
    Help,
    /// The `/keybindings` cheat-sheet (§7 / Q7): the keyboard-shortcut pairs table
    /// + the magic-prefix line. Opened by `/keybindings` or the `Ctrl+/` chord.
    Keybindings,
    Status,
    Cost,
    Verbose,
    /// A transient text card (`/btw`'s answer box) above the composer. The
    /// `ask_id` ties an incoming `BtwAnswer` frame back to THIS card so a stale
    /// card (the user fired a second `/btw`) can't show the wrong answer.
    Btw { ask_id: String, question: String, answer: Option<String> },
    /// The `/scheduler` 3-step flow card (multi-pick → confirm diff → apply).
    Scheduler(crate::components::scheduler::Scheduler),
    /// The `/continue` searchable session picker (content-grep + lazy load).
    Continue(crate::components::continue_picker::ContinuePicker),
    /// The `/effects demo` splash overlay (§9): auto-reverts after a few seconds.
    Effects,
    /// The `/effort` slider overlay (redesign_cc.md §3). ←/→ moves the marker;
    /// Enter forwards `/session.reasoning_effort=<level>` (max→xhigh); Esc cancels.
    EffortSlider(EffortSlider),
}

impl Overlay {
    /// True if this overlay is the FULL-SCREEN kind (help / status / cost /
    /// verbose) vs a compact centered card. The renderer uses it to size the
    /// overlay region.
    pub fn is_fullscreen(&self) -> bool {
        matches!(
            self,
            Overlay::Help | Overlay::Keybindings | Overlay::Status | Overlay::Cost | Overlay::Verbose
        )
    }

    /// True if this overlay is MODAL — it captures keyboard input. A NON-modal
    /// overlay (the `/btw` card) is a transient toast that floats above the
    /// composer WITHOUT stealing input (§7 `/btw` "non-blocking"); only `Esc`
    /// dismisses it, every other key flows to the cockpit.
    pub fn is_modal(&self) -> bool {
        !matches!(self, Overlay::Btw { .. })
    }
}

/// The connection lifecycle, reflecting the REAL handshake (N1). The status word
/// the UI shows is derived straight from this — never a guessed "disconnected".
#[derive(Debug, Clone, PartialEq)]
pub enum ConnStatus {
    Connecting,
    Connected { model: Option<String> },
    /// Carries the real reason so the UI shows "disconnected: <reason>" (N1).
    Disconnected { reason: String },
}

impl ConnStatus {
    /// The short human status word for the footer/header.
    pub fn label(&self) -> String {
        match self {
            ConnStatus::Connecting => "connecting…".to_string(),
            ConnStatus::Connected { model } => match model {
                Some(m) => format!("connected {m}"),
                None => "connected".to_string(),
            },
            ConnStatus::Disconnected { reason } => format!("disconnected: {reason}"),
        }
    }
}

/// Which full-screen VIEW the app is showing. The dashboard (§6 / N2) is a
/// SEPARATE full-screen view (Ctrl+S / left-click; `Esc` exits), NOT a sidebar.
/// Overlays stack on top of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum View {
    #[default]
    Cockpit,
    Dashboard,
    /// The full-screen `/workflows` panel (§7), backed by the singleton watcher
    /// (its own threads, never blocks chat); the panel only tracks focus + style.
    Workflows,
}

/// A pending rename in the dashboard: the session id + the current edit buffer.
/// `r` opens it; typing edits; Enter commits; Esc cancels.
#[derive(Debug, Clone, Default)]
pub struct RenameState {
    pub id: u64,
    pub buffer: String,
}

/// The author of a transcript block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    System,
    Tool,
    /// An out-of-band bridge/system notice (errors, child exit, stderr).
    Notice,
}

impl Role {
    /// Map a protocol role string onto a [`Role`].
    pub fn from_proto(s: &str) -> Role {
        match s {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "system" => Role::System,
            "tool" => Role::Tool,
            _ => Role::Assistant,
        }
    }
}

/// The block-OWNED memo of an assistant block's cockpit render (C1-F4): the styled
/// lines the transcript draws AND the plain/atomic projection the wrap cache counts
/// — produced TOGETHER by ONE [`render_assistant_cockpit_streaming`](crate::markdown)
/// call, so they are 1:1 by construction (the load-bearing parity). Keyed by the
/// full render input `(rev, width, fold_all, theme)`; any change misses and
/// re-renders. It
/// replaces the per-frame `md_cache` HashMap the transcript rebuilt every draw, and
/// collapses the two separate render passes ([`Block::to_render_block`] for the
/// projection, the transcript for the styled lines) into one cached result.
#[derive(Debug, Clone)]
struct CockpitCache {
    rev: u64,
    width: u16,
    fold_all: bool,
    /// A stable digest of the per-node fold overrides affecting THIS block (Fix E /
    /// Q8): a per-node toggle changes the projected rows, so it is part of the key —
    /// toggling one node's fold misses the cache and re-renders, an unrelated block's
    /// toggle does not. `0` when this block has no overrides.
    fold_digest: u64,
    /// The active theme's identity. A `/theme` live-preview swaps the WHOLE theme
    /// (a distinct named palette), so the styled colors are stale if it changes —
    /// keyed here so a preview re-renders. Cheap: `&'static str` (interned).
    theme_name: &'static str,
    /// The styled visual rows (already soft-wrapped to `width`); the transcript
    /// indexes these per visible row.
    lines: Vec<Line<'static>>,
    /// The plain-text projection of `lines` (`\n`-joined span text) — what the
    /// wrap cache wraps to COUNT rows. Stored alongside `lines` so the projection
    /// can never drift from the draw (same render).
    plain: String,
    /// The per-hard-line atomic (inline-math) byte ranges of `plain`, so the wrap
    /// cache keeps rendered math glyph runs intact 1:1 with `lines`.
    atomic_ranges: Vec<Vec<Range<usize>>>,
    /// The clickable-node hit map (Fix E): per-node INTRA-row ranges (inclusive, in
    /// this block's visual-row coordinates), produced by the same render as `lines`.
    /// `sync_transcript` lifts these to global visual indices for the mouse hit test.
    node_hits: Vec<(RangeInclusive<usize>, NodeId)>,
}

/// One logical transcript block. Stores SOURCE text — the only width-independent
/// truth. The render plane derives soft-wrapped rows + the viewport window from
/// `(id, source, rev)` (P1) and copies `source` verbatim (P2). `mid` ties
/// streaming deltas to their block; `id` is the STABLE, monotonic anchor key
/// (never reused, so a scroll anchor survives appends); `rev` bumps on every
/// mutation to drive cache invalidation.
#[derive(Debug, Clone)]
pub struct Block {
    pub id: u64,
    pub mid: Option<String>,
    pub role: Role,
    pub source: String,
    pub rev: u64,
    /// True once `MessageEnd` arrived (or for a synchronous block).
    pub finalized: bool,
    /// Block-owned memo of the cockpit render (assistant blocks only). Interior
    /// mutability because both readers borrow `&self`: [`Block::to_render_block`]
    /// (from `sync_transcript`) and the transcript draw (render is `&AppState`,
    /// P11). A clone carries a still-valid memo (the key fully determines it).
    cockpit_cache: RefCell<Option<CockpitCache>>,
}

impl Block {
    /// Shared block constructor for the `app` module tree (the active reducer +
    /// the `push_*` helpers). `pub(in crate::app)` so it never leaks past `app`.
    pub(in crate::app) fn new(id: u64, mid: Option<String>, role: Role, source: String, finalized: bool) -> Self {
        Block { id, mid, role, source, rev: 1, finalized, cockpit_cache: RefCell::new(None) }
    }

    /// Public block constructor for the multi-session plane (`app::session`),
    /// which owns its own per-session transcripts + id space. Identical to the
    /// `app`-private [`Block::new`].
    pub fn new_external(
        id: u64,
        mid: Option<String>,
        role: Role,
        source: String,
        finalized: bool,
    ) -> Self {
        Block::new(id, mid, role, source, finalized)
    }

    /// A snapshot of this block as a render-plane [`RenderBlock`] (the logical
    /// unit the wrap cache + viewport consume). Cheap clone of the source; the
    /// `(id, rev)` pair lets the cache reuse untouched blocks across frames.
    ///
    /// For an **assistant** block the wrap cache must count the rows the markdown
    /// render actually draws (tables → aligned rows, `$$…$$` → a stacked
    /// fraction, a code fence → label + gutters), NOT the raw source lines —
    /// else the styled draw and the scroll math would disagree (P1). So we hand
    /// the cache the markdown-PLAIN projection for assistants. Copy (P2) still
    /// reads the original `source`, so clean-logical-text copy is unaffected.
    pub fn to_render_block(&self, theme: &Theme, folds: &BlockFolds, width: u16) -> RenderBlock {
        // For an assistant block the cache must count the rows the STYLED draw
        // produces, so we project the SAME render `transcript` draws: the cockpit
        // pass with the same `streaming` flag (`!finalized`) — finalized has no
        // volatile tail, in-flight holds its tail back. The block-owned memo renders
        // that pass ONCE per `(rev, width, fold_all, fold overrides, theme)` and the
        // transcript reads the very same styled lines back, so the projection here and
        // the draw there are 1:1 by construction (P1). Copy (P2) still reads `source`.
        let (source, atomic_ranges) = if self.role == Role::Assistant {
            self.ensure_cockpit_cache(theme, folds, width);
            let cache = self.cockpit_cache.borrow();
            let c = cache.as_ref().expect("ensured above");
            (c.plain.clone(), c.atomic_ranges.clone())
        } else {
            (self.source.clone(), Vec::new())
        };
        RenderBlock {
            id: self.id,
            role: render_role(self.role),
            source,
            rev: self.rev,
            finalized: self.finalized,
            atomic_ranges,
        }
    }

    /// The styled cockpit row at visual index `intra` for this ASSISTANT block,
    /// served from the block-owned memo (rendered once per the full render key).
    /// The transcript calls this for each visible assistant row instead of rebuilding
    /// a per-frame `md_cache` and re-running the markdown pass (C1-F4). Returns
    /// `None` when `intra` is past the block's rows (the caller falls back to the
    /// wrap-cache plain row), so a transient row-count skew can never panic. The
    /// returned line still carries the [`ATOMIC`](crate::markdown) wrap sentinel; the
    /// caller strips it at the draw boundary.
    pub(crate) fn cockpit_line(
        &self,
        theme: &Theme,
        folds: &BlockFolds,
        width: u16,
        intra: usize,
    ) -> Option<Line<'static>> {
        self.ensure_cockpit_cache(theme, folds, width);
        self.cockpit_cache.borrow().as_ref().and_then(|c| c.lines.get(intra).cloned())
    }

    /// The clickable-node hit ranges (Fix E) for this assistant block at the given
    /// render key — INTRA-row ranges (inclusive) the caller lifts to global visual
    /// indices. Empty for a non-foldable block. Served from the same memo as the
    /// styled lines, so the ranges always index real drawn rows.
    pub(crate) fn cockpit_node_hits(
        &self,
        theme: &Theme,
        folds: &BlockFolds,
        width: u16,
    ) -> Vec<(RangeInclusive<usize>, NodeId)> {
        if self.role != Role::Assistant {
            return Vec::new();
        }
        self.ensure_cockpit_cache(theme, folds, width);
        self.cockpit_cache
            .borrow()
            .as_ref()
            .map(|c| c.node_hits.clone())
            .unwrap_or_default()
    }

    /// Populate [`Block::cockpit_cache`] for `(rev, width, fold_all, fold overrides,
    /// theme)` on a key miss — the SINGLE place the cockpit markdown pass runs for an
    /// assistant block. The styled lines, the plain/atomic projection, AND the node
    /// hit map are produced from the one render so they stay 1:1 (P1). A hit (same
    /// key) is a no-op.
    fn ensure_cockpit_cache(&self, theme: &Theme, folds: &BlockFolds, width: u16) {
        let fold_digest = folds.digest();
        if let Some(c) = self.cockpit_cache.borrow().as_ref() {
            if c.rev == self.rev
                && c.width == width
                && c.fold_all == folds.fold_all
                && c.fold_digest == fold_digest
                && c.theme_name == theme.name
            {
                return;
            }
        }
        let render = crate::markdown::render_assistant_cockpit_full(
            &self.source,
            theme,
            folds,
            width,
            !self.finalized,
        );
        let (plain, atomic_ranges) = crate::markdown::lines_to_plain_atomic(&render.lines);
        *self.cockpit_cache.borrow_mut() = Some(CockpitCache {
            rev: self.rev,
            width,
            fold_all: folds.fold_all,
            fold_digest,
            theme_name: theme.name,
            lines: render.lines,
            plain,
            atomic_ranges,
            node_hits: render.node_hits,
        });
    }

    /// This block's render-plane [`BlockRole`] (without a full snapshot). The
    /// transcript widget uses it to choose the gutter + markdown routing.
    pub fn render_role(&self) -> BlockRole {
        render_role(self.role)
    }
}

/// Map an app [`Role`] onto the render plane's [`BlockRole`].
pub(in crate::app) fn render_role(role: Role) -> BlockRole {
    match role {
        Role::User => BlockRole::User,
        Role::Assistant => BlockRole::Assistant,
        Role::System => BlockRole::System,
        Role::Tool => BlockRole::Tool,
        Role::Notice => BlockRole::Notice,
    }
}

/// A pending `AskUser` the UI must surface. The ask-user picker reads these
/// fields to build the candidate list + free-text escape.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PendingAsk {
    pub ask_id: String,
    pub question: String,
    pub options: Vec<AskUserOption>,
    pub free_text: bool,
}

/// The token-cost breakdown surfaced by `/cost` (§4). Live estimates from
/// `Status` frames + a final aggregation; a pure formatter renders the card.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CostBreakdown {
    pub input: u64,
    pub output: u64,
    /// Cumulative cache-read tokens (when the backend reports them).
    pub cache: u64,
    /// The last context-window utilization percent reported (0..=100).
    pub context_percent: Option<f64>,
    /// Accumulated cost in USD (mirrors `AppState::cost_usd`).
    pub cost_usd: f64,
}

impl CostBreakdown {
    /// The total tokens accounted (input + output). PURE.
    pub fn total(&self) -> u64 {
        self.input.saturating_add(self.output)
    }

    /// Render the `/cost` report card as plain lines (the load-bearing formatter,
    /// unit-tested). PURE. `model` is the active model name for the header.
    pub fn report_lines(&self, model: &str) -> Vec<String> {
        let ctx = match self.context_percent {
            Some(p) => format!("{p:.0}%"),
            None => "—".to_string(),
        };
        // GA has no per-token price table, so a 0 cost is "unknown", not "free":
        // show a dash rather than a misleading $0.0000 (a future price map can
        // fill this in). A real accumulated cost still prints with the `$` suffix.
        let cost = if self.cost_usd > 0.0 {
            format!("{:.4}$", self.cost_usd)
        } else {
            "—".to_string()
        };
        vec![
            format!("Token usage · {model}"),
            String::new(),
            format!("  input    {:>10}", self.input),
            format!("  output   {:>10}", self.output),
            format!("  cache    {:>10}", self.cache),
            format!("  total    {:>10}", self.total()),
            String::new(),
            format!("  context  {ctx:>10}"),
            format!("  cost     {cost:>10}"),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::markdown::{lines_to_plain, lines_to_plain_atomic, render_assistant_cockpit_streaming};

    /// A no-override fold view for block `id` at `fold_all` — the default policy the
    /// memo tests exercise (per-node overrides are tested in `app::tests`).
    fn plain_folds(id: u64, fold_all: bool) -> BlockFolds<'static> {
        BlockFolds::plain(id, fold_all)
    }

    // A finalized assistant block with two turns + a chip + inline math — exercises
    // the full cockpit pass (fold, chip, atomic-math) the memo caches.
    const SRC: &str = "\
**Turn 1 ...**
<summary>read the config</summary>
🛠️ Tool: `file_read` path: config.toml
port = 8080
**Turn 2 ...**
the closed form $\\sum_{i=0}^{n} x_i$ holds here
done.";

    /// The block-owned memo (C1-F4) must produce the EXACT lines + plain/atomic
    /// projection a fresh `render_assistant_cockpit_streaming` produces — the memo
    /// is a cache, never a geometry change (the load-bearing P1 parity). We assert
    /// the cached styled lines and the cached projection are byte-identical to the
    /// uncached render at several widths and fold states.
    #[test]
    fn cockpit_cache_matches_fresh_render() {
        let theme = Theme::default_theme();
        for width in [24u16, 40, 80] {
            for fold_all in [false, true] {
                let block = Block::new(1, None, Role::Assistant, SRC.to_string(), true);
                // Fresh, uncached reference.
                let fresh = render_assistant_cockpit_streaming(SRC, &theme, fold_all, width, false);
                let (fresh_plain, fresh_ranges) = lines_to_plain_atomic(&fresh);

                // The memo's styled lines (served row-by-row to the transcript).
                let folds = plain_folds(1, fold_all);
                let cached: Vec<Line<'static>> = (0..fresh.len())
                    .map(|i| block.cockpit_line(&theme, &folds, width, i).expect("row present"))
                    .collect();
                assert_eq!(cached.len(), fresh.len(), "row count w={width} fold={fold_all}");
                assert_eq!(
                    lines_to_plain(&cached),
                    fresh_plain,
                    "cached styled lines diverged from fresh render w={width} fold={fold_all}"
                );
                // Past-the-end index yields None (caller falls back; never panics).
                assert!(block.cockpit_line(&theme, &folds, width, fresh.len()).is_none());

                // The projection `to_render_block` hands the wrap cache is the SAME
                // plain + atomic ranges — produced from the very lines drawn above.
                let rb = block.to_render_block(&theme, &folds, width);
                assert_eq!(rb.source, fresh_plain, "projection plain mismatch w={width} fold={fold_all}");
                assert_eq!(rb.atomic_ranges, fresh_ranges, "projection ranges mismatch w={width} fold={fold_all}");
            }
        }
    }

    /// The memo is keyed by the FULL render input `(rev, width, fold_all)`: a change
    /// to ANY of the three misses and re-renders, while a repeat call with the same
    /// key reuses the stored result (no second markdown pass). A streaming append
    /// (which bumps `rev`) therefore reflows exactly this block.
    #[test]
    fn cockpit_cache_invalidates_on_key_change() {
        let theme = Theme::default_theme();
        let mut block = Block::new(2, None, Role::Assistant, "alpha".to_string(), false);

        // First populate at (rev=1, width=40, fold_all=false).
        let _ = block.cockpit_line(&theme, &plain_folds(2, false), 40, 0);
        assert_eq!(block.cockpit_cache.borrow().as_ref().map(|c| (c.rev, c.width, c.fold_all)), Some((1, 40, false)));

        // Width change → re-keyed.
        let _ = block.cockpit_line(&theme, &plain_folds(2, false), 80, 0);
        assert_eq!(block.cockpit_cache.borrow().as_ref().map(|c| c.width), Some(80));
        // Fold change → re-keyed.
        let _ = block.cockpit_line(&theme, &plain_folds(2, true), 80, 0);
        assert_eq!(block.cockpit_cache.borrow().as_ref().map(|c| c.fold_all), Some(true));
        // Theme change (a `/theme` live-preview swap) → re-keyed, so the cached
        // colors can't go stale under the previewed palette.
        let light = crate::theme::by_name("light").expect("the `light` theme exists");
        let _ = block.cockpit_line(&light, &plain_folds(2, true), 80, 0);
        assert_eq!(block.cockpit_cache.borrow().as_ref().map(|c| c.theme_name), Some(light.name));

        // A streaming append bumps rev; the next consult re-renders to the new text
        // (the stale cache, keyed by the old rev, is replaced).
        block.source.push_str(" beta");
        block.rev = block.rev.wrapping_add(1);
        let _ = block.to_render_block(&theme, &plain_folds(2, true), 80);
        let c = block.cockpit_cache.borrow();
        let c = c.as_ref().unwrap();
        assert_eq!(c.rev, block.rev, "cache re-keyed to the new rev");
        assert!(c.plain.contains("alpha beta"), "re-render picked up the appended delta: {:?}", c.plain);
    }

    /// A non-assistant block never populates the cockpit memo (its projection is the
    /// verbatim source, no markdown pass) — the cache stays empty.
    #[test]
    fn cockpit_cache_skipped_for_non_assistant() {
        let theme = Theme::default_theme();
        let block = Block::new(3, None, Role::User, "hello".to_string(), true);
        let rb = block.to_render_block(&theme, &plain_folds(3, false), 40);
        assert_eq!(rb.source, "hello", "non-assistant projects verbatim source");
        assert!(rb.atomic_ranges.is_empty());
        assert!(block.cockpit_cache.borrow().is_none(), "no memo for a user block");
    }
}
