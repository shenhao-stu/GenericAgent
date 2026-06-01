//! components/picker.rs — the reusable menu/picker PRIMITIVE (checklist §7
//! "Menu/picker primitive: reusable for /llm /continue /rewind /export /scheduler
//! /language /emoji /theme") + the unified `ask_user` card model (§7).
//!
//! Two pure models live here; the overlay widget in `components::overlay` PAINTS
//! them and `main.rs` feeds keys in. Keeping the selection / toggle / window logic
//! pure is the load-bearing discipline (ratatui widgets can't be headlessly
//! tested), so the four pickers (`/llm` index mapping, ask_user single/multi/
//! numeric, theme preview) are unit-tested without a TTY.
//!
//!   * [`Picker`] — a vertical list of [`PickItem`]s with a clamped, scrolling
//!     selection. `/llm` `/continue` `/rewind` `/export` `/scheduler` `/language`
//!     `/emoji` `/theme` all build one of these; the SELECTED item's `id` is what
//!     the dispatcher acts on (e.g. `/llm` → `SwitchLlm(id+1)`, theme → apply).
//!   * [`AskUserPicker`] — the unified ask_user card: a question + candidate rows
//!     with an inline free-text escape row, plus a multi-pick mode (`[多选]`
//!     auto-detected, Space toggles) and a numeric-pick mode. Mirrors tui_v3's
//!     `ask_user` UX exactly.

/// How many rows a picker shows at once before it scrolls (a compact overlay).
pub const PICKER_ROWS: usize = 10;

/// What kind of picker this is — drives the dispatch action on Enter and the
/// title/hint the overlay paints. Each variant maps 1:1 to a §4/§7 command. The
/// `Continue` / `Scheduler` kinds document the reusable-primitive surface (§7
/// "reusable for /llm /continue /rewind /export /scheduler /language /emoji
/// /theme") but those two commands now drive their OWN richer overlays
/// (`continue_picker` / `scheduler`); the variants remain so the primitive is
/// complete and `apply_picker` keeps defensive arms.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PickerKind {
    /// `/llm` — the selected item's `id` is the 0-based LLM index → `SwitchLlm`.
    Llm,
    /// `/theme` — LIVE PREVIEW: moving the selection previews; Enter commits, Esc
    /// reverts (the theme picker's commit/revert contract).
    Theme,
    /// `/pets` (alias `/emoji`) — pet style; preview on move, commit on Enter.
    Emoji,
    /// `/language` — interface language; full repaint on commit.
    Language,
    /// `/export` — clip / all / file destination for the last reply.
    Export,
    /// `/rewind` — pick one of the last ~20 real turns to truncate to.
    Rewind,
    /// `/continue` — searchable picker over session logs (content-grep).
    Continue,
    /// `/scheduler` — multi-pick reflect tasks (Space toggles, then confirm).
    Scheduler,
}

impl PickerKind {
    /// The i18n KEY for this picker's title (the renderer resolves it through
    /// `crate::i18n::t(lang, key)` for the active language — §9 "route user-facing
    /// strings through i18n"). PURE.
    pub fn title_key(self) -> &'static str {
        match self {
            PickerKind::Llm => "picker.title.llm",
            PickerKind::Theme => "picker.title.theme",
            PickerKind::Emoji => "picker.title.emoji",
            PickerKind::Language => "picker.title.language",
            PickerKind::Export => "picker.title.export",
            PickerKind::Rewind => "picker.title.rewind",
            PickerKind::Continue => "picker.title.continue",
            PickerKind::Scheduler => "picker.title.scheduler",
        }
    }

    /// The localized overlay title for this picker.
    pub fn title(self, lang: crate::i18n::Lang) -> &'static str {
        crate::i18n::t(lang, self.title_key())
    }

    /// Whether this picker LIVE-PREVIEWS on selection move (theme / emoji /
    /// language re-skin the cockpit as you arrow; Esc reverts). PURE.
    pub fn previews(self) -> bool {
        matches!(self, PickerKind::Theme | PickerKind::Emoji | PickerKind::Language)
    }

    /// Whether this picker is MULTI-SELECT (Space toggles a row; Enter applies the
    /// set). Only `/scheduler` (multi-pick reflect tasks) today. PURE.
    pub fn multi(self) -> bool {
        matches!(self, PickerKind::Scheduler)
    }

    /// The i18n KEY for this picker's bottom hint (multi / preview / single). PURE.
    pub fn hint_key(self) -> &'static str {
        if self.multi() {
            "picker.hint.multi"
        } else if self.previews() {
            "picker.hint.preview"
        } else {
            "picker.hint.single"
        }
    }

    /// The localized bottom hint line for the overlay.
    pub fn hint(self, lang: crate::i18n::Lang) -> &'static str {
        crate::i18n::t(lang, self.hint_key())
    }
}

/// One selectable row in a [`Picker`]. `id` is the STABLE payload the dispatcher
/// acts on (an LLM index, a theme name index, a turn number, …) — NOT the visual
/// row, so collapsing/filtering never desyncs the action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PickItem {
    /// The action payload (e.g. the 0-based LLM index for `/llm`).
    pub id: usize,
    /// The primary label (e.g. the model name, the theme name).
    pub label: String,
    /// A trailing dim detail (e.g. a turn preview, an export hint). May be empty.
    pub detail: String,
    /// True if this row is the CURRENT one (drawn with a `●` marker — the active
    /// model / theme / language). PURE data; the widget paints the marker.
    pub current: bool,
    /// For a multi-select picker: whether this row is toggled ON. Ignored by
    /// single-select pickers.
    pub checked: bool,
}

impl PickItem {
    /// A single-select row with `id` + `label` (no detail, not current).
    pub fn new(id: usize, label: impl Into<String>) -> Self {
        PickItem {
            id,
            label: label.into(),
            detail: String::new(),
            current: false,
            checked: false,
        }
    }

    /// Builder: set the trailing dim detail.
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = detail.into();
        self
    }

    /// Builder: mark this row as the current/active one (`●`).
    pub fn current(mut self, current: bool) -> Self {
        self.current = current;
        self
    }
}

/// A reusable list picker overlay model. Owns the items + a clamped selection; the
/// dispatcher reads [`Picker::selected`] (single) or [`Picker::checked_ids`]
/// (multi) on Enter. PURE — all nav/toggle/window logic is unit-tested.
#[derive(Debug, Clone)]
pub struct Picker {
    /// Which command opened it (drives the title/hint + dispatch action).
    pub kind: PickerKind,
    /// The rows.
    pub items: Vec<PickItem>,
    /// The highlighted row index (clamped to `items`).
    pub sel: usize,
}

impl Picker {
    /// Build a picker, seeding the selection on the `current` item if there is one
    /// (so `/llm` opens highlighting the active model, `/theme` the active theme).
    pub fn new(kind: PickerKind, items: Vec<PickItem>) -> Self {
        let sel = items.iter().position(|i| i.current).unwrap_or(0);
        Picker { kind, items, sel }
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Move the selection by `delta` (+down/-up), clamped (no wrap). PURE.
    pub fn move_sel(&mut self, delta: isize) {
        self.sel = crate::commands::registry::move_sel(self.sel, delta, self.items.len());
    }

    /// The currently-selected item, if any.
    pub fn selected(&self) -> Option<&PickItem> {
        self.items.get(self.sel)
    }

    /// The selected item's `id` (the dispatch payload), if any.
    pub fn selected_id(&self) -> Option<usize> {
        self.selected().map(|i| i.id)
    }

    /// Toggle the checked state of the selected row (Space, multi-select). No-op
    /// for a single-select picker (the dispatcher gates this on `kind.multi()`).
    pub fn toggle_selected(&mut self) {
        if let Some(item) = self.items.get_mut(self.sel) {
            item.checked = !item.checked;
        }
    }

    /// The `id`s of all CHECKED rows (multi-select apply). PURE. (The generic
    /// multi-select surface; the live `/scheduler` flow uses its own overlay, so
    /// this is exercised by the picker tests + kept for any future multi-picker.)
    #[allow(dead_code)]
    pub fn checked_ids(&self) -> Vec<usize> {
        self.items.iter().filter(|i| i.checked).map(|i| i.id).collect()
    }

    /// The visible window `(start, slice)` of [`PICKER_ROWS`] rows around the
    /// selection (a scrolling viewport). PURE.
    pub fn window(&self) -> (usize, &[PickItem]) {
        window_slice(&self.items, self.sel, PICKER_ROWS)
    }
}

/// The scrolling-window slice for a list of length `len`, selection `sel`, showing
/// at most `rows` items, kept centered-ish and clamped at the ends. PURE — shared
/// by [`Picker`] + the ask_user card. Returns `(start_index, slice)`.
pub fn window_slice<T>(items: &[T], sel: usize, rows: usize) -> (usize, &[T]) {
    if items.len() <= rows || rows == 0 {
        return (0, items);
    }
    let half = rows / 2;
    let start = sel.saturating_sub(half).min(items.len() - rows);
    (start, &items[start..start + rows])
}

// ---------------------------------------------------------------------------
// ask_user — the unified card (single / multi / numeric).
// ---------------------------------------------------------------------------

/// The ask_user pick MODE, auto-detected from the question + candidates (tui_v3
/// `ask_user` UX, §7). PURE detection in [`detect_ask_mode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AskMode {
    /// Single pick: ↑↓ cycle candidates ↔ free-text input; Enter submits the
    /// highlighted candidate (or the typed text).
    Single,
    /// Multi pick (`[多选]`/`[multi-select]` detected): Space toggles each
    /// candidate; Enter submits the joined set.
    Multi,
    /// Numeric pick: the answer is a number (1..=N) — typing digits selects a
    /// candidate by ordinal; Enter submits.
    Numeric,
}

/// Detect the ask_user mode from the `question` + `candidates` (the tui_v3 rule):
///   * a `[多选]` / `[multi-select]` / `multi-select` / `多选` marker → Multi.
///   * else a `[数字]` / `numeric` / "enter a number" marker (or all-numeric
///     candidates) → Numeric.
///   * else → Single.
///
/// PURE + unit-tested (the `askuser_single_multi_numeric` deliverable).
pub fn detect_ask_mode(question: &str, candidates: &[String]) -> AskMode {
    let q = question.to_lowercase();
    let multi_markers = ["[多选]", "多选", "[multi-select]", "multi-select", "multiple"];
    if multi_markers.iter().any(|m| q.contains(m)) {
        return AskMode::Multi;
    }
    let numeric_markers = ["[数字]", "numeric", "enter a number", "输入数字", "pick a number"];
    if numeric_markers.iter().any(|m| q.contains(m)) {
        return AskMode::Numeric;
    }
    // All candidates parse as integers AND there are some → numeric ordinal pick.
    if !candidates.is_empty() && candidates.iter().all(|c| c.trim().parse::<i64>().is_ok()) {
        return AskMode::Numeric;
    }
    AskMode::Single
}

/// The unified ask_user card model (§7): the question, its candidates, the
/// detected [`AskMode`], the highlighted row, the inline free-text buffer, and (for
/// Multi) the per-candidate toggles. Mirrors tui_v3's ask_user picker.
///
/// SELECTION MODEL (single): rows are `candidates` followed by a synthetic
/// FREE-TEXT row at index `candidates.len()`; ↑↓ cycle through all of them, so the
/// user can either pick a candidate or drop to the free-text input and type.
#[derive(Debug, Clone)]
pub struct AskUserPicker {
    /// The opaque ask id echoed back in the `Answer` frame.
    pub ask_id: String,
    /// The question text.
    pub question: String,
    /// The candidate labels (the option list).
    pub candidates: Vec<String>,
    /// The detected interaction mode.
    pub mode: AskMode,
    /// Whether a free-text escape is allowed (from the frame's `free_text`).
    pub free_text: bool,
    /// The highlighted row (0..=candidates.len(); the last is the free-text row
    /// when `free_text`).
    pub sel: usize,
    /// The inline free-text buffer (typed on the free-text row, or in numeric mode).
    pub input: String,
    /// Multi-select toggles, parallel to `candidates`.
    pub checked: Vec<bool>,
}

impl AskUserPicker {
    /// Build the card from an `AskUser` frame's fields. `candidates` is the option
    /// labels; `free_text` whether a free-text escape is offered. The selection
    /// starts on the first candidate (or the free-text row if there are none).
    pub fn new(
        ask_id: impl Into<String>,
        question: impl Into<String>,
        candidates: Vec<String>,
        free_text: bool,
    ) -> Self {
        let question = question.into();
        let mode = detect_ask_mode(&question, &candidates);
        let n = candidates.len();
        AskUserPicker {
            ask_id: ask_id.into(),
            question,
            checked: vec![false; n],
            candidates,
            mode,
            free_text,
            // Start on the first candidate, or the free-text row if no candidates.
            sel: 0,
            input: String::new(),
        }
    }

    /// The number of navigable rows: candidates + (1 free-text row if allowed).
    pub fn row_count(&self) -> usize {
        self.candidates.len() + if self.free_text { 1 } else { 0 }
    }

    /// True when the selection is on the synthetic FREE-TEXT row (single mode), so
    /// keystrokes edit `input` and Enter submits the typed text. PURE.
    pub fn on_free_text_row(&self) -> bool {
        self.free_text && self.sel == self.candidates.len()
    }

    /// Move the highlight by `delta` (↑↓), clamped over [`Self::row_count`]. PURE.
    pub fn move_sel(&mut self, delta: isize) {
        let n = self.row_count().max(1);
        self.sel = crate::commands::registry::move_sel(self.sel, delta, n);
    }

    /// Toggle the highlighted candidate (Space, Multi mode). No-op on the free-text
    /// row or for a non-multi card.
    pub fn toggle_selected(&mut self) {
        if self.mode != AskMode::Multi {
            return;
        }
        if self.sel < self.checked.len() {
            self.checked[self.sel] = !self.checked[self.sel];
        }
    }

    /// In Numeric mode, re-home the highlight onto the candidate whose 1-based
    /// ordinal matches the current `input` (so `2` highlights the 2nd). No-op for
    /// an out-of-range / non-numeric buffer. PURE-ish (only touches `sel`).
    fn rehome_numeric(&mut self) {
        if let Ok(n) = self.input.parse::<usize>()
            && n >= 1
            && n <= self.candidates.len()
        {
            self.sel = n - 1;
        }
    }

    /// Type a char into the inline input. In Numeric mode digits also re-home the
    /// selection onto the matching ordinal candidate (so `2` highlights the 2nd).
    pub fn type_char(&mut self, c: char) {
        // Numeric mode: digits drive an ordinal selection + the input buffer.
        if self.mode == AskMode::Numeric && c.is_ascii_digit() {
            self.input.push(c);
            self.rehome_numeric();
            return;
        }
        // Single/free-text: only accept typing when on the free-text row (or no
        // candidates at all), so arrowing through candidates doesn't eat chars.
        if self.on_free_text_row() || self.candidates.is_empty() {
            self.input.push(c);
        }
    }

    /// Backspace the inline input.
    pub fn backspace(&mut self) {
        self.input.pop();
        if self.mode == AskMode::Numeric {
            self.rehome_numeric();
        }
    }

    /// Resolve the card to the answer text to send in the `Answer` frame on Enter.
    /// PURE — the `askuser_single_multi_numeric` deliverable pins each mode:
    ///   * Multi → the joined CHECKED candidate labels (newline-joined; falls back
    ///     to the highlighted one if none toggled).
    ///   * Numeric → the typed number if present, else the 1-based ordinal of the
    ///     highlighted candidate.
    ///   * Single → the non-empty free-text input (if on the free-text row), else
    ///     the highlighted candidate label.
    ///
    /// Returns `None` if there is nothing to submit (empty free-text, no candidate).
    pub fn resolve_answer(&self) -> Option<String> {
        match self.mode {
            AskMode::Multi => {
                let picked: Vec<String> = self
                    .candidates
                    .iter()
                    .zip(self.checked.iter())
                    .filter(|(_, on)| **on)
                    .map(|(c, _)| c.clone())
                    .collect();
                if !picked.is_empty() {
                    Some(picked.join("\n"))
                } else {
                    // Nothing toggled → fall back to the highlighted candidate.
                    self.candidates.get(self.sel).cloned()
                }
            }
            AskMode::Numeric => {
                let t = self.input.trim();
                if !t.is_empty() {
                    Some(t.to_string())
                } else {
                    // The 1-based ordinal of the highlighted candidate.
                    if self.sel < self.candidates.len() {
                        Some((self.sel + 1).to_string())
                    } else {
                        None
                    }
                }
            }
            AskMode::Single => {
                if self.on_free_text_row() {
                    let t = self.input.trim();
                    if t.is_empty() {
                        None
                    } else {
                        Some(t.to_string())
                    }
                } else {
                    self.candidates.get(self.sel).cloned()
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE deliverable test: /llm picker maps the SELECTED row back to its LLM
    /// index. The picker is built from `LlmList` items `(idx, name, current)`; the
    /// selected item's `id` is the 0-based LLM index the dispatcher turns into
    /// `SwitchLlm(id+1)` (1-based protocol). The current item is pre-highlighted.
    #[test]
    fn llm_picker_maps_index() {
        let items = vec![
            PickItem::new(0, "OpenAI/gpt").current(false),
            PickItem::new(1, "GLM/glm-4").current(true), // the active model.
            PickItem::new(2, "Kimi/k2").current(false),
        ];
        let mut p = Picker::new(PickerKind::Llm, items);
        // Opens highlighting the CURRENT model (index 1).
        assert_eq!(p.sel, 1);
        assert_eq!(p.selected_id(), Some(1));

        // Move down to the 3rd model; selected id maps to its LLM index (2).
        p.move_sel(1);
        assert_eq!(p.selected_id(), Some(2));
        assert_eq!(p.selected().unwrap().label, "Kimi/k2");

        // Move up twice → clamps at the first model (index 0). The protocol switch
        // is 1-based, so the dispatcher sends SwitchLlm(0+1) = SwitchLlm{n:1}.
        p.move_sel(-1);
        p.move_sel(-1);
        p.move_sel(-1); // saturates.
        assert_eq!(p.sel, 0);
        assert_eq!(p.selected_id(), Some(0));
        let one_based = p.selected_id().unwrap() as u32 + 1;
        assert_eq!(one_based, 1);
    }

    /// THE deliverable test: ask_user single / multi / numeric.
    #[test]
    fn askuser_single_multi_numeric() {
        // --- SINGLE: ↑↓ cycle candidates ↔ free-text; Enter submits the choice. --
        let mut single = AskUserPicker::new(
            "a1",
            "Which file should I edit?",
            vec!["src/main.rs".into(), "src/lib.rs".into()],
            true,
        );
        assert_eq!(single.mode, AskMode::Single);
        // Starts on the first candidate.
        assert_eq!(single.resolve_answer().as_deref(), Some("src/main.rs"));
        single.move_sel(1);
        assert_eq!(single.resolve_answer().as_deref(), Some("src/lib.rs"));
        // Arrow onto the synthetic free-text row (index == candidates.len()).
        single.move_sel(1);
        assert!(single.on_free_text_row());
        // Typing on candidates was ignored; now on the free-text row it accepts.
        single.type_char('h');
        single.type_char('i');
        assert_eq!(single.resolve_answer().as_deref(), Some("hi"));
        // An empty free-text row resolves to None (nothing to submit).
        single.backspace();
        single.backspace();
        assert_eq!(single.resolve_answer(), None);

        // --- MULTI: [多选] auto-detected; Space toggles; Enter joins the set. ----
        let mut multi = AskUserPicker::new(
            "a2",
            "选择要执行的任务 [多选]",
            vec!["build".into(), "test".into(), "deploy".into()],
            true,
        );
        assert_eq!(multi.mode, AskMode::Multi);
        // Toggle rows 0 and 2.
        multi.sel = 0;
        multi.toggle_selected();
        multi.sel = 2;
        multi.toggle_selected();
        assert_eq!(multi.resolve_answer().as_deref(), Some("build\ndeploy"));
        // None toggled → falls back to the highlighted candidate.
        let mut multi2 = AskUserPicker::new("a3", "pick [multi-select]", vec!["x".into(), "y".into()], true);
        multi2.sel = 1;
        assert_eq!(multi2.resolve_answer().as_deref(), Some("y"));

        // --- NUMERIC: all-numeric candidates → ordinal pick; typing selects. -----
        let mut numeric = AskUserPicker::new(
            "a4",
            "How many retries?",
            vec!["1".into(), "2".into(), "3".into()],
            true,
        );
        assert_eq!(numeric.mode, AskMode::Numeric);
        // Highlighted candidate's 1-based ordinal by default (row 0 → "1").
        assert_eq!(numeric.resolve_answer().as_deref(), Some("1"));
        // Typing a digit re-homes the selection AND becomes the answer.
        numeric.type_char('3');
        assert_eq!(numeric.sel, 2);
        assert_eq!(numeric.resolve_answer().as_deref(), Some("3"));

        // An explicit "[数字]" marker also forces numeric even with text candidates.
        let forced = AskUserPicker::new("a5", "请输入数字 [数字]", vec!["low".into(), "high".into()], true);
        assert_eq!(forced.mode, AskMode::Numeric);
    }

    #[test]
    fn picker_window_scrolls_and_clamps() {
        let items: Vec<PickItem> = (0..30).map(|i| PickItem::new(i, format!("item {i}"))).collect();
        let mut p = Picker::new(PickerKind::Continue, items);
        // Top: window starts at 0.
        let (start, slice) = p.window();
        assert_eq!(start, 0);
        assert_eq!(slice.len(), PICKER_ROWS);
        // Selecting near the end clamps the window to the tail.
        p.sel = 29;
        let (start_end, slice_end) = p.window();
        assert_eq!(start_end, 30 - PICKER_ROWS);
        assert_eq!(slice_end.len(), PICKER_ROWS);
    }

    #[test]
    fn multi_picker_collects_checked_ids() {
        let items = vec![
            PickItem::new(10, "a"),
            PickItem::new(20, "b"),
            PickItem::new(30, "c"),
        ];
        let mut p = Picker::new(PickerKind::Scheduler, items);
        assert!(p.kind.multi());
        p.sel = 0;
        p.toggle_selected();
        p.sel = 2;
        p.toggle_selected();
        assert_eq!(p.checked_ids(), vec![10, 30]);
    }

    #[test]
    fn picker_kind_hints_and_preview_flags() {
        use crate::i18n::Lang;
        assert!(PickerKind::Theme.previews());
        assert!(PickerKind::Emoji.previews());
        assert!(!PickerKind::Llm.previews());
        assert!(PickerKind::Scheduler.multi());
        assert!(!PickerKind::Llm.multi());
        // The hint is now i18n-keyed; the English resolution carries the gesture.
        assert!(PickerKind::Theme.hint(Lang::En).contains("revert"));
        assert!(PickerKind::Scheduler.hint(Lang::En).contains("toggle"));
        assert!(PickerKind::Llm.hint(Lang::En).contains("select"));
        // The key selectors are stable regardless of language.
        assert_eq!(PickerKind::Theme.hint_key(), "picker.hint.preview");
        assert_eq!(PickerKind::Scheduler.hint_key(), "picker.hint.multi");
        assert_eq!(PickerKind::Llm.title_key(), "picker.title.llm");
    }
}
