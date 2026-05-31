//! input/ — the multi-line composer + input magic (checklist §4 / §8; tui_v3
//! §E). The [`Composer`] owns the LOGICAL buffer (real `\n` newlines), a cursor
//! (byte offset), an optional selection anchor, a 200-deep undo/redo history, the
//! paste side-table, and the input history. Keystrokes fold into it via
//! [`Composer::on_key`], which returns a [`ComposerAction`] the app acts on
//! (submit a message, run a `!shell` line, request a redraw, …).
//!
//! Editing parity (tui_v3 internal-byte model):
//!   * Enter submits · Ctrl+J / Shift+Enter insert a newline
//!   * arrows move the cursor (visual-row aware via the wrap algorithm); Up/Down
//!     at the top/bottom visual edge walk the input history
//!   * Ctrl+A select-all · Ctrl+E end-of-line · Ctrl+U kill-to-line-start ·
//!     Ctrl+X cut · Ctrl+V paste · Ctrl+Z/Y undo/redo (200 deep)
//!   * Shift+arrows extend a selection
//!   * placeholder-aware Backspace/Delete (whole `[Image #N]`/`[File #N]`/
//!     `[Pasted text #N]` block)
//!   * magic prefixes: a leading `!` is shell-mode (border tints hot-pink, runs
//!     the host shell on submit); an `@query` under the caret opens the
//!     gitignore-aware file picker (Tab/Enter completes)
//!
//! All load-bearing logic is PURE over the buffer and unit-tested; the widget
//! just paints `text` + the cursor cell and the (optional) completion dropdown.

pub mod file_expand;
pub mod history;
pub mod keychord;
pub mod paste;
pub mod paths;
pub mod shell;

use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

use crate::render::measure::wrap_line_segments;
use file_expand::AtQuery;
use history::History;
use paste::{PasteStore, Side};

/// Max undo/redo depth (checklist §8: "Ctrl+Z/Y undo/redo (200 deep)").
pub const UNDO_DEPTH: usize = 200;

/// A snapshot of the editable state for undo/redo.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Snapshot {
    text: String,
    cursor: usize,
}

/// The high-level outcome of feeding a key to the composer — the app reacts.
// Escape/Redraw/ToggleFold round out the action vocabulary; the cockpit routes
// those chords at the app level today, so they aren't yet emitted from here.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComposerAction {
    /// Nothing observable changed (e.g. an arrow at a wall) — caller may still
    /// repaint, but no app-level effect.
    None,
    /// The buffer/cursor changed; repaint the composer. (Most edits.)
    Changed,
    /// Submit a NORMAL message. `text` is already expanded (paste placeholders →
    /// payloads; `@path` left for the app to expand against the real root).
    Submit { text: String },
    /// Run a `!cmd` host-shell line. `cmd` is the command body (bang stripped).
    Shell { cmd: String },
    /// The user pressed Esc — the app decides (abort task / clear / close).
    Escape,
    /// Force a full redraw (Ctrl+L).
    Redraw,
    /// Toggle tool-chip folding (Ctrl+O).
    ToggleFold,
}

/// The multi-line composer.
#[derive(Debug)]
pub struct Composer {
    /// The logical buffer (real `\n` newlines).
    text: String,
    /// Cursor as a byte offset into `text` (always on a char boundary).
    cursor: usize,
    /// Selection anchor (byte offset); `Some` while a selection is active. The
    /// selection is `[min(anchor,cursor), max(anchor,cursor))`.
    sel_anchor: Option<usize>,
    /// Undo stack (most recent last) — snapshots BEFORE a mutation.
    undo: Vec<Snapshot>,
    /// Redo stack (cleared on a new edit).
    redo: Vec<Snapshot>,
    /// Folded paste payloads keyed `#N`.
    pub paste: PasteStore,
    /// Persisted input history + nav cursor.
    pub history: History,
    /// A stashed draft for Ctrl+S (stash → restore on next press).
    draft_stash: Option<String>,
    /// The highlighted row in the active `@` file picker (clamped on use).
    pub file_sel: usize,
}

impl Default for Composer {
    fn default() -> Self {
        Composer {
            text: String::new(),
            cursor: 0,
            sel_anchor: None,
            undo: Vec::new(),
            redo: Vec::new(),
            paste: PasteStore::new(),
            history: History::new(),
            draft_stash: None,
            file_sel: 0,
        }
    }
}

/// Which arrow / nav key was pressed (decoupled from crossterm so the core is
/// testable without the event crate).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nav {
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
}

impl Composer {
    pub fn new() -> Self {
        Composer::default()
    }

    /// Construct with a loaded history (the app builds this once the bridge's
    /// repo root is known).
    pub fn with_history(history: History) -> Self {
        Composer {
            history,
            ..Composer::default()
        }
    }

    // ---- read accessors (the widget + app read these) -----------------------

    pub fn text(&self) -> &str {
        &self.text
    }

    #[allow(dead_code)] // cursor offset accessor (tested; used by drag-select in Phase 3).
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// True when the buffer is a shell-mode line (leading `!`) — the widget tints
    /// the border hot-pink (checklist §4: "shell-mode composer border hot-pink").
    pub fn is_shell_mode(&self) -> bool {
        shell::is_shell_mode(&self.text)
    }

    /// The active `@` file query under the caret, if any (drives the picker).
    pub fn at_query(&self) -> Option<AtQuery> {
        file_expand::at_query(&self.text, self.cursor)
    }

    /// The current selection span `[lo, hi)` if one is active.
    pub fn selection(&self) -> Option<(usize, usize)> {
        self.sel_anchor.map(|a| (a.min(self.cursor), a.max(self.cursor)))
    }

    /// The selected text, if any.
    #[allow(dead_code)] // selection-text accessor (tested; used by copy in Phase 3).
    pub fn selected_text(&self) -> Option<&str> {
        self.selection().map(|(lo, hi)| &self.text[lo..hi])
    }

    // ---- mutation primitives (snapshot-aware) -------------------------------

    /// Push the current state onto the undo stack (capped) and clear redo. Call
    /// BEFORE a mutation. Coalescing single-char inserts is intentionally NOT
    /// done (simpler + matches tui_v3's per-edit snapshots well enough at depth
    /// 200).
    fn snapshot(&mut self) {
        if self.undo.len() >= UNDO_DEPTH {
            self.undo.remove(0);
        }
        self.undo.push(Snapshot {
            text: self.text.clone(),
            cursor: self.cursor,
        });
        self.redo.clear();
    }

    /// Set the cursor to a byte offset, clamped to a char boundary within text.
    fn set_cursor(&mut self, off: usize) {
        let off = off.min(self.text.len());
        // Walk back to a char boundary if needed.
        let mut o = off;
        while o > 0 && !self.text.is_char_boundary(o) {
            o -= 1;
        }
        self.cursor = o;
    }

    /// Replace the selection (or insert at the cursor) with `chunk`. Snapshots.
    fn insert(&mut self, chunk: &str) {
        self.snapshot();
        if let Some((lo, hi)) = self.selection() {
            self.text.replace_range(lo..hi, chunk);
            self.cursor = lo + chunk.len();
            self.sel_anchor = None;
        } else {
            self.text.insert_str(self.cursor, chunk);
            self.cursor += chunk.len();
        }
        self.history.reset_nav();
    }

    /// Clear the buffer + cursor + selection (NOT undo — a submit clears via
    /// this, and an explicit undo after a submit is not expected). Does not
    /// snapshot (the caller decides).
    fn clear_buffer(&mut self) {
        self.text.clear();
        self.cursor = 0;
        self.sel_anchor = None;
    }

    /// Programmatically set the whole buffer (history nav / picker completion /
    /// stash restore). Snapshots so it is undoable.
    fn set_text(&mut self, text: String, cursor: usize) {
        self.snapshot();
        self.text = text;
        self.set_cursor(cursor);
        self.sel_anchor = None;
    }

    /// Public buffer-set (slash-palette completion). Undoable; resets history nav.
    pub fn set_buffer(&mut self, text: String, cursor: usize) {
        self.set_text(text, cursor);
        self.history.reset_nav();
    }

    // ---- public editing entry points ----------------------------------------

    /// Insert a printable chunk (a typed char or a small inline paste). Sanitizes
    /// CRLF→LF and drops tabs (a literal tab breaks cursor-cell math + lets a
    /// host overlay reclaim Tab). Returns `Changed`.
    pub fn type_str(&mut self, chunk: &str) -> ComposerAction {
        let clean = sanitize_input(chunk);
        if clean.is_empty() {
            return ComposerAction::None;
        }
        // Big / multi-line chunk → fold to a paste placeholder (tui_v3 §D-50).
        if clean.len() > 1 && (clean.contains('\n') || clean.chars().count() > PASTE_FOLD_THRESHOLD) {
            let pathish = clean.trim();
            if !pathish.contains('\n') && pathish.len() <= 1024 && paste::looks_like_path(pathish) {
                let r = if paste::is_image_path(pathish) {
                    self.paste.fold_image(pathish)
                } else {
                    self.paste.fold_file(pathish)
                };
                let token = r.insert;
                self.insert(&token);
                return ComposerAction::Changed;
            }
            let r = self.paste.fold_text(&clean);
            let token = r.insert;
            self.insert(&token);
            return ComposerAction::Changed;
        }
        self.insert(&clean);
        ComposerAction::Changed
    }

    /// Insert a literal newline (Ctrl+J / Shift+Enter).
    pub fn newline(&mut self) -> ComposerAction {
        self.insert("\n");
        ComposerAction::Changed
    }

    /// Submit the buffer on Enter. Empty → `None`. A `!cmd` line → `Shell`;
    /// otherwise expand paste placeholders and return `Submit`. Clears the buffer
    /// and pushes history on a real submit.
    pub fn submit(&mut self) -> ComposerAction {
        let raw = self.text.clone();
        if raw.trim().is_empty() {
            return ComposerAction::None;
        }
        // `!cmd` shell line → run the host shell (the app owns it).
        if shell::is_shell_line(&raw) {
            let cmd = shell::strip_bang(&raw);
            self.history.push(&raw);
            self.clear_buffer();
            self.paste = PasteStore::new();
            return ComposerAction::Shell { cmd };
        }
        // Normal: expand paste placeholders to payloads (the app then expands
        // `@path` against the real root + collects images from the store first).
        let expanded = self.paste.expand(&raw);
        self.history.push(&raw);
        self.clear_buffer();
        // NOTE: the app reads `self.paste.collect_images(&raw)` BEFORE we reset;
        // `submit_images` exposes that, so reset the store here is safe because
        // the app calls images() then submit(); we keep the store until reset.
        self.paste = PasteStore::new();
        ComposerAction::Submit { text: expanded }
    }

    /// The image paths the CURRENT buffer references (Submit.images). The app
    /// calls this just before `submit()` (which resets the store).
    #[allow(dead_code)] // image attachment wires through here in Phase 3.
    pub fn submit_images(&self) -> Vec<String> {
        self.paste.collect_images(&self.text)
    }

    /// Backspace: delete the selection, else a whole adjacent placeholder, else
    /// the previous grapheme. Placeholder-aware (checklist §8).
    pub fn backspace(&mut self) -> ComposerAction {
        if self.selection().is_some() {
            self.insert("");
            return ComposerAction::Changed;
        }
        if let Some(ph) = paste::placeholder_adjacent(&self.text, self.cursor, Side::Left) {
            self.snapshot();
            let (out, caret) = paste::delete_placeholder(&self.text, ph, &mut self.paste);
            self.text = out;
            self.set_cursor(caret);
            self.history.reset_nav();
            return ComposerAction::Changed;
        }
        if self.cursor == 0 {
            return ComposerAction::None;
        }
        self.snapshot();
        let prev = prev_grapheme_boundary(&self.text, self.cursor);
        self.text.replace_range(prev..self.cursor, "");
        self.cursor = prev;
        self.history.reset_nav();
        ComposerAction::Changed
    }

    /// Delete (forward): selection, else a whole adjacent placeholder, else the
    /// next grapheme. Falls back to backspace semantics at end-of-text.
    pub fn delete(&mut self) -> ComposerAction {
        if self.selection().is_some() {
            self.insert("");
            return ComposerAction::Changed;
        }
        let side = if self.cursor < self.text.len() {
            Side::Right
        } else {
            Side::Left
        };
        if let Some(ph) = paste::placeholder_adjacent(&self.text, self.cursor, side) {
            self.snapshot();
            let (out, caret) = paste::delete_placeholder(&self.text, ph, &mut self.paste);
            self.text = out;
            self.set_cursor(caret);
            self.history.reset_nav();
            return ComposerAction::Changed;
        }
        if self.cursor < self.text.len() {
            self.snapshot();
            let next = next_grapheme_boundary(&self.text, self.cursor);
            self.text.replace_range(self.cursor..next, "");
            self.history.reset_nav();
            ComposerAction::Changed
        } else {
            self.backspace()
        }
    }

    /// Cursor / selection movement. `extend` keeps/starts a selection (Shift+arrow).
    /// `width` is the composer's inner text width for visual-row Up/Down. Up/Down
    /// at the top/bottom visual edge walk the input history instead.
    pub fn nav(&mut self, dir: Nav, extend: bool, width: u16) -> ComposerAction {
        // Manage the selection anchor.
        if extend {
            if self.sel_anchor.is_none() {
                self.sel_anchor = Some(self.cursor);
            }
        } else {
            self.sel_anchor = None;
        }

        let rows = self.visual_rows(width);
        let (row, _col) = self.cursor_visual_pos(width, &rows);

        match dir {
            Nav::Left => {
                let p = prev_grapheme_boundary(&self.text, self.cursor);
                if p == self.cursor {
                    return ComposerAction::None;
                }
                self.cursor = p;
                ComposerAction::Changed
            }
            Nav::Right => {
                let n = next_grapheme_boundary(&self.text, self.cursor);
                if n == self.cursor {
                    return ComposerAction::None;
                }
                self.cursor = n;
                ComposerAction::Changed
            }
            Nav::Home => {
                self.cursor = rows.get(row).map(|r| r.start).unwrap_or(0);
                ComposerAction::Changed
            }
            Nav::End => {
                self.cursor = rows.get(row).map(|r| r.end).unwrap_or(self.text.len());
                ComposerAction::Changed
            }
            Nav::Up => {
                if row == 0 {
                    // Top visual edge → input history (unless extending a sel).
                    if !extend {
                        if let Some(v) = self.history.nav_up(&self.text) {
                            self.set_text(v.clone(), v.len());
                            return ComposerAction::Changed;
                        }
                    }
                    return ComposerAction::None;
                }
                let col = self.cursor_visual_pos(width, &rows).1;
                self.cursor = offset_in_row(&self.text, &rows, row - 1, col);
                ComposerAction::Changed
            }
            Nav::Down => {
                if row + 1 >= rows.len() {
                    if !extend {
                        if let Some(v) = self.history.nav_down() {
                            self.set_text(v.clone(), v.len());
                            return ComposerAction::Changed;
                        }
                    }
                    return ComposerAction::None;
                }
                let col = self.cursor_visual_pos(width, &rows).1;
                self.cursor = offset_in_row(&self.text, &rows, row + 1, col);
                ComposerAction::Changed
            }
        }
    }

    /// Ctrl+A — select the WHOLE buffer (tui_v3 maps Ctrl+A to select-all).
    pub fn select_all(&mut self) -> ComposerAction {
        if self.text.is_empty() {
            return ComposerAction::None;
        }
        self.sel_anchor = Some(0);
        self.cursor = self.text.len();
        ComposerAction::Changed
    }

    /// Ctrl+E — move to end-of-(visual)-line.
    pub fn end_of_line(&mut self, width: u16) -> ComposerAction {
        self.sel_anchor = None;
        let rows = self.visual_rows(width);
        let (row, _) = self.cursor_visual_pos(width, &rows);
        self.cursor = rows.get(row).map(|r| r.end).unwrap_or(self.text.len());
        ComposerAction::Changed
    }

    /// Ctrl+U — kill from the cursor back to the start of the logical line.
    pub fn kill_to_line_start(&mut self) -> ComposerAction {
        let line_start = self.text[..self.cursor]
            .rfind('\n')
            .map(|i| i + 1)
            .unwrap_or(0);
        if line_start == self.cursor {
            return ComposerAction::None;
        }
        self.snapshot();
        self.text.replace_range(line_start..self.cursor, "");
        self.cursor = line_start;
        self.sel_anchor = None;
        self.history.reset_nav();
        ComposerAction::Changed
    }

    /// Ctrl+X — cut the selection, returning it for the clipboard. No selection →
    /// `None` action and no clipboard text.
    pub fn cut(&mut self) -> (ComposerAction, Option<String>) {
        match self.selection() {
            Some((lo, hi)) => {
                let cut = self.text[lo..hi].to_string();
                self.insert(""); // replaces the selection with nothing (snapshots).
                (ComposerAction::Changed, Some(cut))
            }
            None => (ComposerAction::None, None),
        }
    }

    /// Ctrl+V — paste `text` (folds large / multi-line via [`Composer::type_str`]).
    pub fn paste(&mut self, text: &str) -> ComposerAction {
        self.type_str(text)
    }

    /// Ctrl+Z — undo one step.
    pub fn undo(&mut self) -> ComposerAction {
        if let Some(prev) = self.undo.pop() {
            self.redo.push(Snapshot {
                text: self.text.clone(),
                cursor: self.cursor,
            });
            self.text = prev.text;
            self.set_cursor(prev.cursor);
            self.sel_anchor = None;
            ComposerAction::Changed
        } else {
            ComposerAction::None
        }
    }

    /// Ctrl+Y — redo one step.
    pub fn redo(&mut self) -> ComposerAction {
        if let Some(next) = self.redo.pop() {
            self.undo.push(Snapshot {
                text: self.text.clone(),
                cursor: self.cursor,
            });
            self.text = next.text;
            self.set_cursor(next.cursor);
            self.sel_anchor = None;
            ComposerAction::Changed
        } else {
            ComposerAction::None
        }
    }

    /// Ctrl+S — stash the current draft, or restore a previously stashed one
    /// (toggle). Returns `Changed` when something moved.
    pub fn stash_or_restore(&mut self) -> ComposerAction {
        match self.draft_stash.take() {
            Some(stashed) => {
                // Restore (and re-stash whatever is here so a 2nd press swaps back).
                let current = std::mem::replace(&mut self.text, stashed);
                if !current.is_empty() {
                    self.draft_stash = Some(current);
                }
                self.set_cursor(self.text.len());
                self.sel_anchor = None;
                ComposerAction::Changed
            }
            None => {
                if self.text.is_empty() {
                    return ComposerAction::None;
                }
                self.draft_stash = Some(std::mem::take(&mut self.text));
                self.cursor = 0;
                self.sel_anchor = None;
                ComposerAction::Changed
            }
        }
    }

    /// Complete the active `@` file query with `picked` (Tab/Enter in the picker).
    /// Splices `@picked ` over the query span. Returns `Changed`, or `None` when
    /// no query is active.
    pub fn complete_file(&mut self, picked: &str) -> ComposerAction {
        let Some(q) = self.at_query() else {
            return ComposerAction::None;
        };
        let res = file_expand::apply_pick(&self.text, &q, picked);
        self.set_text(res.text, res.caret);
        self.file_sel = 0;
        ComposerAction::Changed
    }

    // ---- visual-row geometry (for nav + the widget) -------------------------

    /// Compute the VISUAL rows of the buffer at `width`: each row's byte span
    /// `[start, end)` in `text`, across hard newlines and soft wraps. This uses
    /// the SAME CJK/word-wrap algorithm as the transcript so the cursor math
    /// matches what the widget draws. Returns at least one (empty) row.
    pub fn visual_rows(&self, width: u16) -> Vec<RowSpan> {
        rows_for(&self.text, width)
    }

    /// The cursor's (row, visual-column) at `width`, given the precomputed rows.
    fn cursor_visual_pos(&self, width: u16, rows: &[RowSpan]) -> (usize, usize) {
        let _ = width;
        for (i, r) in rows.iter().enumerate() {
            // The cursor belongs to the row whose span contains it; a cursor at a
            // row boundary belongs to the EARLIER row's end unless it's the very
            // next row's start after a hard newline.
            if self.cursor >= r.start && self.cursor <= r.end {
                let col = UnicodeWidthStr::width(&self.text[r.start..self.cursor]);
                // If the cursor is exactly at this row's end AND there's a later
                // row that starts here (a soft wrap with no separator char),
                // prefer reporting it on this row (col == row width) — natural.
                return (i, col);
            }
        }
        let last = rows.len().saturating_sub(1);
        let col = rows
            .get(last)
            .map(|r| UnicodeWidthStr::width(&self.text[r.start..self.text.len().min(r.end)]))
            .unwrap_or(0);
        (last, col)
    }

    /// The cursor's (row, col) for the WIDGET to place the inverse-cell caret.
    pub fn cursor_rc(&self, width: u16) -> (usize, usize) {
        let rows = self.visual_rows(width);
        self.cursor_visual_pos(width, &rows)
    }
}

/// One visual row's byte span in the buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowSpan {
    pub start: usize,
    pub end: usize,
    /// True if this row begins a NEW hard line (after a `\n`) vs a soft-wrap.
    pub hard_start: bool,
}

/// Threshold (in characters) above which a single typed/pasted chunk is treated
/// as a paste to fold. A normal keystroke is 1 char; an editor/IDE paste is many.
pub const PASTE_FOLD_THRESHOLD: usize = 80;

/// Sanitize a raw input chunk: CRLF/CR → LF, drop tabs (tui_v3
/// `sanitizeComposerInput`). PURE.
pub fn sanitize_input(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                out.push('\n');
            }
            '\t' => { /* drop tabs */ }
            other => out.push(other),
        }
    }
    out
}

/// The byte offset of the grapheme boundary BEFORE `pos` in `s`.
fn prev_grapheme_boundary(s: &str, pos: usize) -> usize {
    if pos == 0 {
        return 0;
    }
    let mut last = 0;
    for (i, _) in s.grapheme_indices(true) {
        if i >= pos {
            break;
        }
        last = i;
    }
    last
}

/// The byte offset of the grapheme boundary AFTER `pos` in `s`.
fn next_grapheme_boundary(s: &str, pos: usize) -> usize {
    if pos >= s.len() {
        return s.len();
    }
    for (i, g) in s.grapheme_indices(true) {
        if i == pos {
            return i + g.len();
        }
        if i > pos {
            return i;
        }
    }
    s.len()
}

/// Compute the visual rows of `text` at `width` (the pure core of
/// [`Composer::visual_rows`], so it is unit-tested directly). Each hard line is
/// soft-wrapped with the transcript's algorithm; we map each wrapped SEGMENT
/// back to a byte span in the original text.
pub fn rows_for(text: &str, width: u16) -> Vec<RowSpan> {
    let w = width.max(1) as usize;
    let mut rows: Vec<RowSpan> = Vec::new();
    let mut line_start = 0usize;
    // Iterate hard lines (split on '\n'), preserving byte offsets.
    let mut hard_iter_start = 0usize;
    loop {
        // Find the end of this hard line.
        let rest = &text[hard_iter_start..];
        let nl = rest.find('\n');
        let line_end = match nl {
            Some(rel) => hard_iter_start + rel,
            None => text.len(),
        };
        let line = &text[line_start..line_end];

        // Soft-wrap the hard line; map each segment to a byte span by re-finding
        // the (trimmed) segment text from a moving cursor in `line`.
        let segments = wrap_line_segments(line, w);
        let mut seg_cursor = 0usize; // byte offset within `line`
        let n_segs = segments.len();
        for (si, seg) in segments.iter().enumerate() {
            // Locate the segment text in the remaining line (wrap trims boundary
            // spaces, so search from seg_cursor for the first occurrence).
            let hay = &line[seg_cursor..];
            let (seg_lo, seg_hi) = if seg.text.is_empty() {
                (seg_cursor, seg_cursor)
            } else {
                match hay.find(&seg.text) {
                    Some(rel) => {
                        let lo = seg_cursor + rel;
                        (lo, lo + seg.text.len())
                    }
                    None => (seg_cursor, seg_cursor + seg.text.len().min(hay.len())),
                }
            };
            rows.push(RowSpan {
                start: line_start + seg_lo,
                // The LAST segment of a hard line ends at the line end (so the
                // cursor can sit just before the newline); inner segments end at
                // the start of the next segment so the spans tile the line.
                end: if si + 1 == n_segs {
                    line_end
                } else {
                    line_start + seg_hi
                },
                hard_start: si == 0,
            });
            seg_cursor = seg_hi.max(seg_cursor);
        }
        if n_segs == 0 {
            // Empty hard line → one empty row.
            rows.push(RowSpan {
                start: line_start,
                end: line_end,
                hard_start: true,
            });
        }

        match nl {
            Some(_) => {
                hard_iter_start = line_end + 1; // skip the '\n'
                line_start = hard_iter_start;
                if hard_iter_start > text.len() {
                    break;
                }
                // A trailing newline yields a final empty row (caret can land there).
                if hard_iter_start == text.len() {
                    rows.push(RowSpan {
                        start: text.len(),
                        end: text.len(),
                        hard_start: true,
                    });
                    break;
                }
            }
            None => break,
        }
    }
    if rows.is_empty() {
        rows.push(RowSpan {
            start: 0,
            end: 0,
            hard_start: true,
        });
    }
    rows
}

/// The byte offset in `rows[row]` closest to visual column `col` (used for Up/Down
/// to keep the cursor in the same column). PURE.
fn offset_in_row(text: &str, rows: &[RowSpan], row: usize, col: usize) -> usize {
    let Some(r) = rows.get(row) else {
        return text.len();
    };
    let slice = &text[r.start..r.end];
    let mut acc = 0usize;
    let mut off = r.start;
    for (i, g) in slice.grapheme_indices(true) {
        let gw = UnicodeWidthStr::width(g);
        if acc + gw > col {
            return r.start + i;
        }
        acc += gw;
        off = r.start + i + g.len();
    }
    off
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE deliverable test: the composer's editing operations.
    #[test]
    fn composer_edit_ops() {
        let mut c = Composer::new();

        // Typing inserts at the cursor.
        c.type_str("hello");
        assert_eq!(c.text(), "hello");
        assert_eq!(c.cursor(), 5);

        // Left arrow moves the cursor; typing inserts mid-buffer.
        c.nav(Nav::Left, false, 80);
        c.nav(Nav::Left, false, 80);
        assert_eq!(c.cursor(), 3);
        c.type_str("X");
        assert_eq!(c.text(), "helXlo");
        assert_eq!(c.cursor(), 4);

        // Backspace deletes the previous grapheme.
        c.backspace();
        assert_eq!(c.text(), "hello");
        assert_eq!(c.cursor(), 3);

        // Ctrl+E to end-of-line, newline (Ctrl+J / Shift+Enter), type on line 2.
        c.end_of_line(80);
        assert_eq!(c.cursor(), 5);
        c.newline();
        c.type_str("world");
        assert_eq!(c.text(), "hello\nworld");

        // Visual-row Up/Down keeps the column.
        // cursor at end of "world" (col 5). Up → line 0 col 5 = end of "hello".
        c.nav(Nav::Up, false, 80);
        let (row, col) = c.cursor_rc(80);
        assert_eq!((row, col), (0, 5));
        assert_eq!(c.cursor(), 5);

        // Select-all + cut empties the buffer and returns the text.
        c.select_all();
        assert_eq!(c.selected_text(), Some("hello\nworld"));
        let (_, cut) = c.cut();
        assert_eq!(cut.as_deref(), Some("hello\nworld"));
        assert!(c.is_empty());

        // Undo restores the cut text; redo removes it again.
        c.undo();
        assert_eq!(c.text(), "hello\nworld");
        c.redo();
        assert!(c.is_empty());

        // Ctrl+U kill-to-line-start.
        c.type_str("abc def");
        c.nav(Nav::Home, false, 80);
        c.nav(Nav::Right, false, 80);
        c.nav(Nav::Right, false, 80);
        c.nav(Nav::Right, false, 80);
        c.nav(Nav::Right, false, 80); // cursor after "abc "
        c.kill_to_line_start();
        assert_eq!(c.text(), "def");

        // Shift+arrow selection then replace by typing.
        c.nav(Nav::Home, false, 80);
        c.nav(Nav::Right, true, 80);
        c.nav(Nav::Right, true, 80);
        assert_eq!(c.selected_text(), Some("de"));
        c.type_str("XY");
        assert_eq!(c.text(), "XYf");
    }

    #[test]
    fn submit_normal_and_shell_and_empty() {
        let mut c = Composer::new();
        // Empty submit is a no-op.
        assert_eq!(c.submit(), ComposerAction::None);

        // Normal submit returns the (placeholder-expanded) text + clears + history.
        c.type_str("ask the model");
        let act = c.submit();
        assert_eq!(act, ComposerAction::Submit { text: "ask the model".into() });
        assert!(c.is_empty());
        assert_eq!(c.history.entries().last().map(String::as_str), Some("ask the model"));

        // Shell line returns Shell{cmd} with the bang stripped.
        c.type_str("!ls -la");
        assert!(c.is_shell_mode());
        let act = c.submit();
        assert_eq!(act, ComposerAction::Shell { cmd: "ls -la".into() });
        assert!(c.is_empty());
    }

    #[test]
    fn placeholder_aware_backspace_whole_block() {
        let mut c = Composer::new();
        c.type_str("see ");
        // Simulate a folded multi-line paste (a big chunk).
        let big = "line1\nline2\nline3\nline4";
        c.type_str(big); // folds to [Pasted text #1 +3 lines]
        assert!(c.text().contains("[Pasted text #1"));
        // The cursor is just after the placeholder; one Backspace wipes it whole.
        c.backspace();
        assert_eq!(c.text(), "see ");
        assert!(!c.text().contains('['));
    }

    #[test]
    fn history_walk_at_edges() {
        let mut c = Composer::new();
        c.type_str("first");
        c.submit();
        c.type_str("second");
        c.submit();
        // At an empty fresh buffer, Up (top edge) recalls newest.
        c.type_str("wip");
        c.nav(Nav::Up, false, 80);
        assert_eq!(c.text(), "second");
        c.nav(Nav::Up, false, 80);
        assert_eq!(c.text(), "first");
        // Down restores toward the draft.
        c.nav(Nav::Down, false, 80);
        assert_eq!(c.text(), "second");
        c.nav(Nav::Down, false, 80);
        assert_eq!(c.text(), "wip");
    }

    #[test]
    fn at_query_drives_file_completion() {
        let mut c = Composer::new();
        c.type_str("open @ma");
        let q = c.at_query().expect("an @ query is active");
        assert_eq!(q.partial, "ma");
        c.complete_file("src/main.rs");
        assert_eq!(c.text(), "open @src/main.rs ");
        // Token closed → no active query now.
        assert!(c.at_query().is_none());
    }

    #[test]
    fn rows_for_handles_wrap_and_newlines_and_cursor_geometry() {
        // Hard newline → two rows, second is a hard_start.
        let rows = rows_for("ab\ncd", 80);
        assert_eq!(rows.len(), 2);
        assert!(rows[0].hard_start);
        assert!(rows[1].hard_start);
        assert_eq!(&"ab\ncd"[rows[1].start..rows[1].end], "cd");

        // Soft wrap of one long hard line at a narrow width → multiple rows; the
        // 2nd row is NOT a hard_start.
        let text = "alpha beta gamma";
        let rows = rows_for(text, 7);
        assert!(rows.len() >= 2);
        assert!(rows[0].hard_start);
        assert!(!rows[1].hard_start);

        // Trailing newline yields a final empty row so the caret can land there.
        let rows = rows_for("x\n", 80);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1].start, rows[1].end);
    }

    #[test]
    fn undo_redo_depth_is_bounded() {
        let mut c = Composer::new();
        for i in 0..(UNDO_DEPTH + 50) {
            c.type_str(&i.to_string());
        }
        // Undo as far as possible — never panics, bounded by the depth.
        let mut undos = 0;
        while c.undo() == ComposerAction::Changed {
            undos += 1;
            if undos > UNDO_DEPTH + 100 {
                panic!("undo did not terminate");
            }
        }
        assert!(undos <= UNDO_DEPTH);
    }

    #[test]
    fn sanitize_strips_tabs_and_normalizes_newlines() {
        // CRLF→LF, lone CR→LF (recon "CRLF/CR → LF"), tab dropped.
        assert_eq!(sanitize_input("a\r\nb\tc\rd"), "a\nbc\nd");
        // A plain CRLF paste becomes a single LF.
        assert_eq!(sanitize_input("x\r\ny"), "x\ny");
    }

    #[test]
    fn ctrl_s_stash_and_restore() {
        let mut c = Composer::new();
        c.type_str("draft text");
        c.stash_or_restore(); // stash
        assert!(c.is_empty());
        c.stash_or_restore(); // restore
        assert_eq!(c.text(), "draft text");
    }
}
