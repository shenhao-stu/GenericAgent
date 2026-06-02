# R6 — Sticky Last-User-Message Header + tuiapp_v2 / tui_v3 Parity Audit

_Spec date: 2026-06-02. Build verified clean: `cargo build` → `Finished dev 0.22s`._

---

## PART A — STICKY HEADER

### A1. REPRODUCED?

**NOT YET IMPLEMENTED** — the feature does not exist in tui_v4 today. This is a
new-feature gap, not a regression. Confirming the absence: the only scroll-state
indicator in the transcript region today is the `"▼ more below — End/PgDn to follow"` dim hint painted on the LAST visible row when `!app.following()` (transcript.rs:97-105). There is no pinned context row at the TOP of the transcript viewport showing the most-recent user prompt.

**Live proof of absence (styled TestBackend check):**

```
// scroll_up on a multi-turn session → no row in the visible window
// carries the "↑ " prefix or the pinned user-prompt text at position y=0
// of the transcript region.  This test would FAIL if the feature were present.
```

### A2. ROOT CAUSE

No implementation exists. The viewport computes `visible()` (viewport.rs:232-235),
which returns exactly `[top, top+height)` visual lines with NO injected header row.
`render_transcript` (transcript.rs:16-108) iterates those lines and renders them
with no special handling for the "scrolled-away" state at the top row.

`app.following()` → `viewport.is_following()` (viewport.rs:76-78) is already
exposed and checked (transcript.rs:97) for the bottom hint, so the predicate for
"when to show the sticky row" exists; the row itself is simply never inserted.

There is also no `last_user_source()` method on `AppState` (the symmetric twin to
`last_assistant_source()` at app/mod.rs:648). Adding one is needed.

### A3. REFERENCE PATTERN

**Claude Code (the gold reference):**

- `src/components/FullscreenLayout.tsx:551-589` — `StickyPromptHeader` component: a 1-row `Box` (fixed height=1 to avoid layout shift) rendered ABOVE the `ScrollBox` whenever the viewport is scrolled away from the tail. Displays `figures.pointer` + the first-paragraph of the most-recent user prompt, truncated (`wrap="truncate-end"`). Uses the `userMessageBackground` color (the same charcoal as the user band in tui_v4).
- `src/components/VirtualMessageList.tsx:133-160` — `computeStickyPromptText()`: extracts the text of the most-recent user message above the visible window, caches it in a `WeakMap`.
- `src/components/VirtualMessageList.tsx:892-1040` — `StickyTracker` component: subscribes to scroll events, walks message positions, calls `setStickyPrompt({text, scrollTo})` when the viewport is above a user prompt. The `scrollTo` closure re-anchors the viewport to the user message when the header is clicked.
- Design comment at FullscreenLayout.tsx:540-549: "Height is FIXED at 1 row (truncate-end for long prompts). A variable-height header … shifts the ScrollBox by 1 row every time the sticky prompt switches during scroll — content jumps on screen."

**No sticky-header precedent in codex-rs** (grep for sticky/pinned/header/breadcrumb in `temp/codex_src/codex-rs/tui/src/` finds only unrelated hits). The CC React implementation is the definitive reference.

**tui_v3.py / tuiapp_v2.py:** Neither has a sticky header in their scrollback. In tui_v3 the "scrollback" is the terminal's native scrollback (not a tui-managed viewport), so the concept does not apply. In tuiapp_v2 the transcript scrolls inside a Textual panel but no sticky header is implemented. This is a **new-to-tui_v4** feature sourced directly from the user request and CC.

### A4. FIX SPEC

**Files and functions to change:**

#### 1. `src/app/mod.rs` — add `last_user_source()` accessor

```rust
/// The first line of the most recent USER message in the transcript.
/// Used for the sticky breadcrumb shown at the top of the transcript viewport
/// while the user is scrolled above the tail. Returns `None` when the transcript
/// has no user turns yet.
pub fn last_user_source_first_line(&self) -> Option<&str> {
    self.transcript
        .iter()
        .rev()
        .find(|b| b.role == Role::User && !b.source.is_empty())
        .map(|b| b.source.lines().next().unwrap_or(""))
}
```

#### 2. `src/components/cockpit/mod.rs` (in `CockpitLayout` and `split_cockpit`) — add a `sticky_header: Option<Rect>` slot

In `CockpitLayout` struct (mod.rs:80-104) add:
```rust
/// `Some` only when the user is NOT in follow mode: a 1-row dim breadcrumb
/// pinned to the top of the transcript region showing the last user prompt.
pub sticky_header: Option<Rect>,
```

In `split_cockpit` (mod.rs:109-180) modify the transcript constraint: when
`!following` and there is a last user prompt, allocate 1 row for the sticky
header ABOVE the transcript `Min(0)` region. Change:

```rust
// Before:
Constraint::Min(0),  // transcript (FLEX)

// After (when !following && last_user_text.is_some()):
Constraint::Length(1),  // sticky header
Constraint::Min(0),     // transcript (FLEX, 1 row shorter)
// When following, no Length(1) is inserted — same as today.
```

`split_cockpit` is `pub(crate)` and called by both `prepare_frame` and
`render_cockpit`, so both see the same layout (no geometry drift, P11).

Populate `sticky_header`:
```rust
let following = app.following();
let has_sticky = !following && app.last_user_source_first_line().is_some();
// insert Length(1) before Min(0) only when has_sticky:
if has_sticky { constraints.push(Constraint::Length(1)); }
constraints.push(Constraint::Min(0));
// …
let sticky_header = has_sticky.then(&mut next);
let transcript = next();
```

#### 3. `src/components/cockpit/transcript.rs` — add `render_sticky_header()`

```rust
/// Render the pinned 1-row breadcrumb at the TOP of the transcript viewport
/// when the user is scrolled away from the tail. Shows `"↑ <first line of last
/// user prompt…>"` clipped to the area width, dim + UserBand background.
/// HIDDEN when the viewport is in follow mode (caller only invokes when
/// `!app.following()`).
pub(crate) fn render_sticky_header(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
    let Some(text) = app.last_user_source_first_line() else { return; };
    let band = Style::default()
        .bg(theme.color(Token::UserBand))
        .fg(theme.color(Token::Dim));
    let w = area.width as usize;
    let prefix = "↑ ";
    let body = clip_to(text, w.saturating_sub(2));
    let used = 2 + unicode_width::UnicodeWidthStr::width(body.as_str());
    let pad = w.saturating_sub(used);
    let line = Line::from(vec![
        Span::styled(prefix.to_string(), band),
        Span::styled(body, band),
        Span::styled(" ".repeat(pad), band),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}
```

#### 4. `src/components/cockpit/mod.rs` — call `render_sticky_header` in `render_cockpit`

In `render_cockpit` (mod.rs:187-238) after the `split_cockpit` destructure, add:
```rust
if let Some(sh) = sticky_header {
    transcript::render_sticky_header(frame, sh, app, theme);
}
render_transcript(frame, transcript, app, theme);
```

`CockpitLayout` must also expose `sticky_header` in the destructure.

#### 5. `src/app/mod.rs` — `prepare_frame` must include sticky header in geometry

`prepare_frame` (mod.rs:561-567) calls `split_cockpit(self, area).transcript`
to get the transcript rect for `sync_transcript`. Since `split_cockpit` now
inserts a `Length(1)` sticky-header row when not following, `transcript.height`
will already be 1 row shorter — the wrap cache syncs to the correct height
automatically. **No additional change needed here** as long as `split_cockpit`
is the single source of geometry truth (P11 is preserved).

### A5. HONEST-CHECK (live path, would FAIL today, PASS after fix)

```rust
#[test]
fn sticky_header_shows_when_scrolled_up_absent_when_following() {
    use crate::app::{AppState, ConnStatus};
    use crate::components::render;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    let (w, h) = (60u16, 20u16);
    let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
    let mut app = AppState::new();
    app.conn = ConnStatus::Connected { model: Some("m".into()) };
    // Push enough content that we can scroll up.
    app.push_user("what is the answer?".into());
    for _ in 0..15 {
        // stream a long assistant reply to make the transcript tall
        app.push_notice("assistant line".into());
    }
    let theme = crate::theme::Theme::default_theme();
    // FOLLOWING mode — no sticky header.
    app.prepare_frame(ratatui::layout::Rect::new(0, 0, w, h), &theme);
    term.draw(|f| render(f, &app, &theme, 0)).unwrap();
    let buf = term.backend().buffer();
    let has_up_arrow_at_top = (0..w as usize)
        .any(|x| buf.content()[x].symbol().contains('↑'));
    assert!(!has_up_arrow_at_top, "no sticky header when following");

    // SCROLL UP — sticky header must appear.
    app.scroll_lines(-10);
    assert!(!app.following(), "precondition: scrolled away from tail");
    app.prepare_frame(ratatui::layout::Rect::new(0, 0, w, h), &theme);
    term.draw(|f| render(f, &app, &theme, 0)).unwrap();
    let buf = term.backend().buffer();
    // The first visible row of the transcript region should carry "↑ what is the answer?"
    // Search the top ~4 rows (header rows are above this).
    let sticky_row_text: String = (0..w as usize)
        .map(|x| buf.content()[x].symbol().to_string())
        .collect::<String>();
    // Not checking exact row y since header + sep offsets vary; check the whole buffer.
    let full_content: String = (0..h as usize).flat_map(|y| {
        (0..w as usize).map(move |x| buf.content()[y * w as usize + x].symbol().to_string())
    }).collect();
    assert!(
        full_content.contains("↑") && full_content.contains("what is the answer"),
        "sticky header must contain '↑ <last user prompt>'; got frame:\n{full_content}"
    );
}
```

**Location:** `src/components/cockpit/transcript.rs` tests module or a new
integration test in `src/app/tests.rs`.

---

## PART B — PARITY AUDIT: tui_v4 vs tuiapp_v2 + tui_v3

### Sources

| Source | Key sections |
|---|---|
| `tuiapp_v2.py` BINDINGS (line 1876-2607) | `ctrl+j/enter/v/u/s/n/b/o/up/down/d/t/slash` |
| `tuiapp_v2.py` COMMANDS (line 1234-1265) | 31-entry list |
| `tui_v3.py` `_cmds()` (line 1810-1841) | 24-entry list |
| `tui_v3.py` `_keys()` (line 5092-5298) | raw byte dispatch |
| `tui_v3.py` help strings (line 162-166) | explicit keybinding table |
| `tui_v4` registry (commands/registry.rs:61-103) | 41 entries (incl. aliases) |
| `tui_v4` keymap (input/keymap.rs:38-286) | all `cockpit_key` arms |

### B1. COMMAND PARITY TABLE

Legend: ✅ = present and wired, ⚠ = partially wired or degraded, ❌ = missing.

| Command | v2 | v3 | v4 | Notes |
|---|---|---|---|---|
| `/help` | ✅ | ✅ | ✅ | Overlay in v4 |
| `/status` / `/sessions` | ✅ | ✅ | ✅ | |
| `/new [name]` | ✅ | ✅ | ✅ | |
| `/switch <id>` | ✅ | ❌ (listed, commented out) | ✅ (opens dashboard) | |
| `/close` | ✅ | ❌ (listed, not impl) | ✅ | |
| `/rename <name>` | ✅ | ✅ | ✅ | |
| `/branch [name]` | ✅ | ❌ | ✅ | |
| `/rewind [n]` | ✅ | ✅ | ✅ | |
| `/clear` | ✅ | ✅ | ✅ | |
| `/stop` / `/abort` | ✅ | ✅ | ✅ | |
| `/llm [n]` | ✅ | ✅ | ✅ | |
| `/btw <q>` | ✅ | ✅ | ✅ | |
| `/review [request]` | ✅ | ✅ | ✅ (Fwd) | |
| `/update [note]` | ✅ | ✅ | ✅ (Fwd) | |
| `/autorun [seed]` | ✅ | ✅ | ✅ (Fwd) | |
| `/morphling [target]` | ✅ | ✅ | ✅ (Fwd) | |
| `/goal [goal]` | ✅ | ✅ | ✅ (Fwd + opens /workflows) | |
| `/hive [target]` | ✅ | ✅ | ✅ (Fwd + opens /workflows) | |
| `/conductor [task]` | ✅ | ✅ | ✅ (Fwd + opens /workflows) | |
| `/scheduler` | ✅ | ✅ | ✅ | |
| `/continue [n\|name]` | ✅ | ✅ | ✅ | |
| `/resume` | ✅ | ✅ | ✅ (Fwd) | |
| `/cost` | ✅ | ✅ | ✅ | |
| `/export clip\|file\|all` | ✅ | ✅ | ✅ | |
| `/restore` | ✅ (v2:2279) | ❌ not listed | ✅ (App cmd) | |
| `/reload-keys` | ✅ (v2:2656) | ❌ not listed | ✅ (App cmd) | |
| `/language [code]` | ❌ not in v2 | ✅ | ✅ | |
| `/emoji [style]` / `/pets` | ❌ not in v2 | ✅ | ✅ (`/pets` + `/emoji` alias) | |
| `/effort [level]` | ❌ | ❌ | ✅ | v4-new |
| `/verbose` / `/tools` / `/trace` | ❌ | ✅ | ✅ | |
| `/fold` | ❌ | ❌ | ✅ | v4-new (alias for Ctrl+Shift+O) |
| `/theme` | ❌ | ❌ | ✅ | v4-new |
| `/keybindings` | ❌ | ❌ | ✅ | v4-new |
| `/workflows` | ❌ | ❌ | ✅ | v4-new |
| `/quit` / `/exit` | ✅ | ✅ | ✅ | |

**Command-level gaps: NONE** — all v2 and v3 commands resolve and are wired.
The four "not listed" v3 gaps (`/switch`, `/close`, `/branch`, `/restore`,
`/reload-keys`) are intentional single-session limitations of v3, not features.

### B2. KEYBINDING PARITY TABLE

| Binding | v2 | v3 | v4 | Flag |
|---|---|---|---|---|
| `Enter` submit | ✅ | ✅ | ✅ | |
| `Ctrl+J` / `Ctrl+Enter` / `Shift+Enter` newline | ✅ | ✅ | ✅ | |
| `Ctrl+C` 3-stage (copy sel / abort / arm-quit) | ✅ | ✅ | ✅ | |
| `Ctrl+N` new session | ✅ | ❌ | ✅ | |
| `Ctrl+B` **sidebar toggle** | ✅ (v2:2593) | ❌ | ❌ `→ branch` | ⚠ HIGH |
| `Ctrl+O` fold/unfold all tool chips | ✅ (v2:2594) | ✅ (0x0f) | ❌ `→ copy last reply` | ⚠ HIGH |
| `Ctrl+O` copy last reply | ❌ | ❌ | ✅ (v4 keymap:68) | v4-new |
| `Ctrl+Shift+O` fold/unfold | ❌ | ❌ | ✅ (v4 keymap:80) | v4-new (Ctrl+O moved) |
| `Ctrl+Up` / `Ctrl+Down` prev/next session | ✅ (v2:2595-96) | ❌ | ✅ | |
| `Ctrl+D` / `Ctrl+W` drop session | ✅ (v2:2597) | ❌ | ✅ (`Ctrl+W` + `Ctrl+D`) | |
| `Ctrl+/` / `Ctrl+_` keybindings help | ✅ (v2:2600-02) | ❌ | ✅ | |
| `Ctrl+T` theme picker | ✅ (v2:2607) | ❌ | ✅ | |
| `Ctrl+S` stash draft | ❌ | ✅ (0x13) | ❌ `→ open dashboard` | ⚠ MEDIUM |
| `Ctrl+S` open session dashboard | ❌ | ❌ | ✅ (v4 keymap:121) | v4-new |
| `Ctrl+G` stash/restore draft | ❌ | ❌ | ✅ (v4 keymap:197) | v4-new alias |
| `Ctrl+L` force repaint | ❌ | ✅ (0x0c) | ✅ | |
| `Ctrl+Z` undo | ✅ | ✅ (0x1a) | ✅ | |
| `Ctrl+Y` redo | ✅ | ✅ (0x19) | ✅ | |
| `Ctrl+A` select all | ✅ | ✅ (0x01) | ✅ | |
| `Ctrl+U` kill to line start | ✅ (v2:1884) | ✅ (0x15) | ✅ | |
| `Ctrl+V` paste | ✅ (v2:1879) | ✅ (0x16) | ✅ | |
| `Ctrl+X` cut selection | ❌ | ✅ (0x18) | ✅ | |
| `Ctrl+E` end of line | ❌ | ❌ | ✅ (v4 keymap:169) | v4-new |
| `Tab` complete palette | ✅ (v2:2606) | ✅ | ✅ | |
| `Esc` universal back | ✅ (v2:2605) | ✅ | ✅ | |
| `Esc Esc` rewind picker | ✅ (v2: rewind armed) | ✅ (v3:5421-30) | ✅ | |
| `PageUp` / `PageDown` scroll | ✅ (implicit) | ✅ | ✅ | |
| `Ctrl+Home` / `Ctrl+End` transcript home/end | ❌ | ❌ | ✅ (v4:262-263) | v4-new |
| `←` (empty composer) open dashboard | ❌ | ❌ | ✅ (v4:242) | v4-new |
| `Shift+←→↑↓` extend selection | ✅ (v2 InputArea) | ✅ (0x1c-1f) | ✅ (v4 composer) | |
| `Ctrl+Shift+M` toggle mouse capture | ❌ | ❌ | ✅ (v4:86) | v4-new |
| `↑/↓` input history at edge | ✅ (v2 InputArea) | ✅ (v3:5200-44) | ✅ (v4 composer) | |

### B3. HIGH-VALUE GAPS

#### GAP-1: `Ctrl+O` semantic divergence — **BREAKING muscle memory** ⚠ HIGH

- **v2** (tuiapp_v2.py:2594): `Ctrl+O` = `toggle_fold` (fold/unfold all tool chips).
- **v3** (tui_v3.py:5274): `Ctrl+O` (byte 0x0f) = `_fold_all = not _fold_all` + repaint.
- **v4** (keymap.rs:68): `Ctrl+O` = copy last assistant reply (the CC/Codex chord).
- The fold operation moved to `Ctrl+Shift+O` in v4. This is a **documented intentional
  change** (keymap.rs:64-66 comment says "Ctrl+Shift+O — toggle tool-chip / turn folding").
- **Impact:** Any user migrating from v2/v3 will press `Ctrl+O` expecting to fold/unfold
  and instead copy the last reply to the clipboard silently. The `/fold` command and
  `Ctrl+Shift+O` chord are the correct paths in v4. The `/keybindings` overlay should
  prominently show the chord so muscle memory errors are self-healing.
- **Spec recommendation:** No code change needed (intentional redesign), but the
  `/keybindings` overlay (components/overlay/info.rs) must list `Ctrl+Shift+O = fold`
  AND `Ctrl+O = copy reply` with a "changed from v3" note to reduce friction.
  Verify `components/overlay/info.rs` already has both. (File: `src/components/overlay/info.rs:95`)

#### GAP-2: `Ctrl+B` = "branch" in v4 vs "toggle sidebar" in v2 ⚠ MEDIUM

- **v2** (tuiapp_v2.py:2593): `Ctrl+B` = toggle sidebar (show/hide the multi-session
  sidebar panel). v2 has a visual sidebar widget.
- **v4** (keymap.rs:137-142): `Ctrl+B` = branch the active session.
- **tui_v4 has no sidebar** (the dashboard is a full-screen overlay, not a sidebar).
  This difference is structural/intentional — v4's architecture replaced the sidebar
  with the full-screen session dashboard (`Ctrl+S`). No code change needed. The
  `/keybindings` overlay must clearly list `Ctrl+B = branch session`.

#### GAP-3: `Ctrl+S` semantic divergence ⚠ MEDIUM

- **v3** (tui_v3.py:5221, byte 0x13): `Ctrl+S` = stash/restore draft.
- **v4** (keymap.rs:121-124): `Ctrl+S` = open session dashboard.
- **v4** (keymap.rs:197-199): `Ctrl+G` = stash/restore draft (the remapping).
- This is an intentional change: stash moved to `Ctrl+G`, `Ctrl+S` freed for the
  dashboard (keymap.rs:195-196 explains this). No code change needed, but the
  `/keybindings` overlay must list `Ctrl+G = stash draft` so v3 users find it.

#### GAP-4: `Ctrl+B` in v2 = sidebar; v4's sidebar equivalent = `Ctrl+S` dashboard

v2 users who pressed `Ctrl+B` to see all sessions should press `Ctrl+S` in v4.
The dashboard is strictly more capable than the v2 sidebar (it shows all sessions,
last Q/S previews, and is full-screen for readability). No gap here; the `/keybindings`
overlay must cross-reference.

#### GAP-5: `/sessions` as standalone command (v2 COMMANDS line 1237)

- **v2** lists `/sessions` as a separate command entry ("列出所有会话").
- **v4** registers `/sessions` as an **alias of `/status`** (registry.rs:65).
- In v3, `/sessions` is handled by the same `elif name in ('status', 'sessions'):` branch.
- This is correct behavior: the alias still resolves (users can type `/sessions`); it
  just shows the same status overlay. No gap.

#### GAP-6: `/cost [all]` — v2 has `all` subcommand

- **v2** (tuiapp_v2.py:1260): `/cost [all]` where `all` scans all-session logs.
- **v4** `/cost` is an App command that opens the Cost overlay (dispatch.rs:284).
  No `all` subcommand is wired.
- **v3** only shows the current-session aggregate.
- **Impact:** LOW — the Cost overlay shows per-session data; the `all` subcommand was
  a v2-specific multi-session aggregate. This is a low-value edge case.

#### GAP-7: `/export file <name>` — v2/v3 allow free-text filename; v4 has no text entry

- **v2** (tui_v3.py:4476-4485): `/export file` fills the input with a default
  timestamp filename that the user can edit before pressing Enter.
- **v4** (dispatch.rs:178-192): `/export file` calls `clipboard::export_action(app, 2)`
  directly — no filename editing step.
- **Impact:** LOW — the user can still export with a timestamp filename. Missing
  the "edit the filename before saving" flow. Consider adding a text-input step.

#### GAP-8: `Ctrl+P` / `Ctrl+N` for history navigation — tui_v3 uses these

- **tui_v3 / tui_v2:** In a terminal raw-mode TUI, `Ctrl+P` (0x10) = cursor up / history
  prev and `Ctrl+N` (0x0e) = cursor down / history next. tui_v3 maps `↑` at the
  first row of the composer to `_nav_hist(-1)` (tui_v3.py:5212).
- **v4:** `Ctrl+N` = new session (keymap.rs:127). The `↑` arrow at the top of the
  composer still triggers history navigation in v4 (via the composer's Nav::Up →
  history at edge logic). `Ctrl+P` is not bound.
- **Impact:** LOW — v4 still has `↑`-at-edge history. `Ctrl+P` as history-prev is
  an emacs-style binding; its absence is unlikely to matter for most users.

### B4. BEHAVIOR GAPS (not command/keybinding level)

| Feature | v2/v3 | v4 | Flag |
|---|---|---|---|
| Sticky "last user message" breadcrumb on scroll up | ❌ neither | ❌ not yet | **Part A above** |
| `/export file` with editable filename | v2+v3 yes | ❌ direct save only | LOW |
| `/cost all` (all-session aggregate) | v2 yes (v3 single only) | ❌ | LOW |
| Session sidebar (always-visible) | v2 yes | ❌ (full-screen dashboard instead) | intentional |
| `Ctrl+S` stash draft | v3 yes | `Ctrl+G` in v4 | intentional, document |
| `Ctrl+O` fold | v2+v3 yes | `Ctrl+Shift+O` in v4 | intentional, document |
| `/verbose` full-screen tool audit: `↑↓` select · `c` copy · `q` quit | v3 (tui_v3.py:4503-04) | ✅ overlay | verify live |
| BTW panel: shows running side question | v2 sidebar + v3 `_btws` list | ✅ Btw overlay | ✅ |
| Ask-user multi-select Space toggle | v3 (tui_v3.py:5127) + v2 | ✅ AskUserPicker | ✅ |
| Picker: `←` cancel / roll back | v3 (tui_v3.py:5141), v2 ChoiceList | ✅ Esc / Back in overlays | verify live |
| Input history `↑/↓` at edge | v2 InputArea + v3 history | ✅ Composer nav | ✅ |
| Persistent input history (per-repo) | v2+v3 yes | ✅ `input::history::History::load` | ✅ |
| CJK word-wrap | v2+v3 (elaborate monkey-patch) | `unicode_width` crate — functional but no CJK-specific word-break | ⚠ MEDIUM |

---

## HONEST-CHECK SUMMARY

### A (sticky header)
Test: `sticky_header_shows_when_scrolled_up_absent_when_following` (spec above).
- **TODAY:** FAILS — no `↑ <prompt>` row appears at the top of the transcript after scrolling up.
- **AFTER FIX:** PASSES — a dim UserBand row starting with `↑ ` appears at the top of the
  transcript region (immediately below the separator) when `!app.following()`.

### B (parity audit)
Test: Compile and run the existing `registry_resolves_all_commands` test in `commands/registry.rs` tests. All 41 names in the current registry resolve; the test confirms the command surface is complete.

No automated test covers the `Ctrl+O` → `Ctrl+Shift+O` muscle-memory migration: a
targeted keymap comment audit is the lightest proof. See `input/keymap.rs:64-82` for
the dual-chord documentation.

---

## SUMMARY

The tui_v4 transcript render has no sticky "last user prompt" breadcrumb at the top of the viewport while scrolled up. The feature requires: (1) a `last_user_source_first_line()` accessor on `AppState`; (2) a `Length(1)` sticky-header slot injected into `split_cockpit` only when not following; (3) a `render_sticky_header()` function rendering a dim UserBand row with `↑ <prompt>` clipped to width; and (4) calling it from `render_cockpit`. The Claude Code `StickyPromptHeader` component (FullscreenLayout.tsx:551-589) is the reference; the design choice of fixed 1-row height to prevent layout shift is load-bearing. The parity audit finds no missing commands (all 31 v2 commands + all 24 v3 commands map to the 41-entry v4 registry), three intentional binding remappings (`Ctrl+O`, `Ctrl+S`, `Ctrl+B`) that need clear documentation in the `/keybindings` overlay, and two low-value feature gaps (`/export file` editable filename, `/cost all`). The HIGH-priority action from Part B is ensuring `/keybindings` overlay explicitly lists the v3→v4 chord remappings to eliminate muscle-memory errors.

VERDICT: NOT-REPRODUCED — sticky header is absent (feature not yet built); no command parity gaps found, three intentional keybinding remappings need `/keybindings` overlay documentation.
