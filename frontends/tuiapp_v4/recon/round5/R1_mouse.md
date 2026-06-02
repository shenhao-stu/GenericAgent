# R1 — Mouse: Select+Copy AND Click Expand/Collapse with ▸/▾

Recon date: 2026-06-02. Build verified: `cargo build` exits `Finished dev` (no warnings).

---

## (1) REPRODUCED?

### Bug A — Mouse cannot select+copy text

**REPRODUCED.** Confirmed by reading `src/term.rs:29` and running `cargo test mouse_capture_defaults_on`:

```
// src/term.rs:29
execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
```

`EnableMouseCapture` is emitted unconditionally at startup and stays on. While ON, the terminal's native drag-select is suppressed; the user cannot select text with the mouse. The round-4 workaround was `Shift+drag` — but on Windows (the user's OS) most terminals (Windows Terminal, mintty/git-bash) do NOT pass `Shift+drag` events through `EnableMouseCapture` differently; they still suppress native selection. The user reports it still does not work.

Rendered evidence (what the user sees in src/term.rs):
```
EnterAlternateScreen + EnableMouseCapture  ← capture ON, native drag killed
```

`Ctrl+Shift+M` (`src/input/keymap.rs:86`) toggles it off, but this is an invisible, undiscoverable opt-out. The user has never been told this.

### Bug B — Click-to-collapse: expanded turn has no clickable target

**REPRODUCED.** The current design has a fundamental asymmetry:

- **Folded turn** (`FoldSegment::Fold`): renders `" ▸ summary"` at `src/markdown/mod.rs:168`. This line is tagged with `NodeId::Turn { block, turn }` (line 178). A click at col 0–1 on that row hits the node and calls `toggle_fold` → turn expands.
- **Expanded turn** (`FoldSegment::Text`): calls `render_turn_body` at `src/markdown/mod.rs:182`. `render_turn_body` pushes prose rows with `None` tag and tool-box rows with `NodeId::Tool` tag (line 354). **No row is ever tagged `NodeId::Turn` for an expanded turn.** There is also no visible `▾` header row for the expanded turn body.

Result: once a turn is expanded by clicking `▸`, there is no clickable row to collapse it again. The `toggle_fold` logic itself is correct (`src/app/fold_hit.rs:115`), but it can never be reached for an expanded turn via mouse because `transcript_node_at` returns `None` for every row of the expanded body (the node-hit table has no `NodeId::Turn` entry covering those rows).

Concrete rendered frame (via `render_to_rows` / TestBackend, 80 cols, turn expanded):

```
Row 3: [empty blank line between turns]
Row 4:  ↳ first summary                       ← breadcrumb, col 0 = space, no ▾
Row 5: ╭─ web_scan  ✓ ok  ·t1 ──────────────╮ ← tool box, col 0 = ╭ (clickable if expandable)
Row 6: │  tabs_only: true                    │
Row 7: │  3 tabs scanned                     │
Row 8: ╰───────────────────────────────────╯
```

No row in the expanded turn body carries a `▾` glyph to indicate "click to collapse."

### Bug C — ▸ never rotates to ▾

**REPRODUCED.** The `▸` in `src/markdown/mod.rs:168` is a constant literal. There is no code path that renders `▾` for a turn that was *previously folded and is now toggling back*. The `▾` glyph appears only inside a tool box's interior as a collapse-affordance row (`src/render/chip.rs:299`) when a tool result is expanded — never on a turn header. The turn header is always `" ▸ summary"` regardless of fold state, because by the time it could render `▾` it would be a `FoldSegment::Text` (and render nothing at all for the turn header).

---

## (2) ROOT CAUSE

### Bug A root cause

File: `src/term.rs:29`  
```rust
execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
```
`EnableMouseCapture` (xterm `?1000h` + `?1002h` or `?1003h`) is set at startup and held on. While active, the terminal intercepts all mouse events and stops forwarding native selection to the OS clipboard. `Shift+drag` is terminal-dependent; on Windows Terminal / Windows ConPTY it does NOT bypass capture.

The toggle at `src/input/keymap.rs:86` (`Ctrl+Shift+M`) works but is invisible and non-sticky — user has to press it every session.

### Bug B root cause

File: `src/markdown/mod.rs:159–184` — `render_assistant_cockpit_full` match on `FoldSegment`.

When a turn is *folded*, `FoldSegment::Fold` renders one `" ▸ summary"` line tagged `NodeId::Turn`. When a turn is *expanded* (after a click or default), it becomes `FoldSegment::Text` and is routed to `render_turn_body`. **`render_turn_body` never emits a row tagged `NodeId::Turn`** — it has no concept of an "expanded turn header." The node-hit table (`app.node_hit` rebuilt in `src/app/fold_hit.rs:23`) therefore contains no `NodeId::Turn` entry for any row of an expanded turn's body. `transcript_node_at` at `src/app/fold_hit.rs:53` returns `None` for every click on expanded turn rows, so `click_fold_at` at line 78 returns `false` (not handled) and the fold is never collapsed via mouse.

Secondary: `src/app/fold_hit.rs:54` limits the clickable zone to `col < FOLD_HIT_COLS` (= 2). Even if a `▾` were added, it would need to live in col 0–1 to be reachable. For the tool box, `╭` is at col 0 and IS within range — but the turn body has no equivalent fixed gutter.

### Bug C root cause

The `▸` is a constant in `src/markdown/mod.rs:168`. `FoldSegment::Fold` renders `▸` and `FoldSegment::Text` renders nothing for the turn header. There is no intermediate `▾`-expanded state for turns. The existing `▾` for tool boxes (`src/render/chip.rs:299`) is interior to the box, not a header glyph.

---

## (3) REFERENCE: How tuiapp_v2, tui_v3, and Codex do it

### tuiapp_v2.py — fold ▸/▾ with click toggle

`tuiapp_v2.py:5398–5401`:
```python
expanded = (not self.fold_mode) ^ (i in m._toggled_folds)
arrow = "▾" if expanded else "▸"
title = seg.get("title") or "completed turn"
header = Text(); header.append(f"{arrow} ", style=C_DIM); header.append(title, style=C_MUTED)
out.append(("fold-header", header, i))
if expanded:
    out.append(("fold-body", cached_render(seg.get("content", "")), i))
```

The **header is ALWAYS rendered** as a `FoldHeader` widget (`tuiapp_v2.py:1797`), whether the turn is collapsed OR expanded. When collapsed it shows `▸ title`, when expanded it shows `▾ title` and the body follows. `App.on_click` at line 3222 checks `isinstance(w, FoldHeader)` and calls `msg._toggled_folds.discard/add(idx)` then `_remount_assistant_message`. The fold header is clickable **in both states**.

Key insight: **the header is always rendered regardless of fold state**. Clicking it in either state toggles.

### tui_v3.py — fold rendering

`tui_v3.py` uses the same pattern (grep `▸`/`▾` shows same approach): a turn header line is always rendered, with `▸` collapsed and `▾` expanded. The body is conditionally shown below.

### Codex (temp/codex_src/codex-rs) — mouse capture

`tui/src/tui.rs:156–169` (`set_modes`):
```rust
pub fn set_modes() -> Result<()> {
    execute!(stdout(), EnableBracketedPaste)?;
    enable_raw_mode()?;
    keyboard_modes::enable_keyboard_enhancement();
    let _ = execute!(stdout(), EnableFocusChange);
    Ok(())
}
```

**Codex does NOT call `EnableMouseCapture` at all.** There is no `EnableMouseCapture` / `DisableMouseCapture` anywhere in `tui/src/`.

For wheel scroll, Codex uses `EnableAlternateScroll` (`tui/src/tui.rs:177`, `\x1b[?1007h`), only when entering alt-screen (`tui/src/tui.rs:621`):
```rust
let _ = execute!(self.terminal.backend_mut(), EnableAlternateScroll);
```
`?1007h` is the "alternate scroll" mode: terminals translate wheel events into cursor-up/cursor-down key events WITHOUT stealing native mouse selection. This is a fundamentally different mechanism: the terminal still does OS-level drag-select normally, but wheel motions arrive as arrow keys. Codex's scroll handler in `chatwidget` (not mouse.rs) receives `KeyEvent` arrow presses from wheel, not `MouseEvent::ScrollUp`.

Codex achieves: **native terminal drag-select works; wheel scrolls; no click-to-fold** (Codex has no fold feature at all — its chat history is in normal terminal scrollback, not managed in a ratatui widget). This means Codex's design is NOT directly applicable to tui_v4's click-to-fold requirement, but the `?1007h` mechanism IS directly applicable to the select+copy problem.

---

## (4) FIX SPEC

### Fix A — Replace EnableMouseCapture with AlternateScroll + app-internal fold clicks

**Design decision**: Keep `EnableMouseCapture` OFF by default (or replace with `?1007h`). Implement click-to-fold using a different event source. This satisfies select+copy AND wheel AND click-fold simultaneously.

**Mechanism**: `?1007h` (alternate scroll) translates wheel → arrow key events. The fold click problem is solved by making the fold header always a visible, full-width clickable band — any terminal with `EnableMouseCapture` ON can click it, OR the user uses keyboard (already works: `Ctrl+O`, `Enter` on fold header in future).

**Alternative design (recommended for tui_v4)**: Keep `EnableMouseCapture` ON (for wheel and fold clicks), but emit `?1007h` in ADDITION — `?1007h` does not conflict with `?1000h`. The key fix is: add a `DisableMouseCapture` toggle on startup (default OFF — no capture), use `?1007h` for wheel, and teach the fold-click to work with keyboard shortcuts. OR: the cleaner path is an in-app drag-select overlay using `MouseEvent::Drag` (available only when capture is on).

**Cleanest unified design** satisfying all three requirements simultaneously:

1. **Default**: `EnableMouseCapture` ON (for wheel + click-fold). Emit `?1007h` as well (belt-and-suspenders for terminals that need it for wheel).
2. **App-internal text selection**: Handle `MouseEventKind::Down(Left)`, `MouseEventKind::Drag(Left)`, `MouseEventKind::Up(Left)` in the BODY area (col >= FOLD_HIT_COLS). Maintain `selection_anchor: Option<(u16, u16)>` and `selection_end: Option<(u16, u16)>` in `AppState`. On `Up`, extract text from the viewport's visual rows for the row range, call `copy_to_clipboard` (already fully implemented in `src/render/copy.rs`), flash a "Copied N chars" notice.
3. The body click (col >= 2) never reaches fold-hit code (already gated in `transcript_node_at:54`), so fold clicks and body clicks are non-conflicting.

#### Files to change for Fix A:

**`src/term.rs`**: Emit `?1007h` alongside `EnableMouseCapture`. Add a custom `EnableAlternateScroll` command (copy from codex `tui/src/tui.rs:173–191`):
```rust
// In setup():
execute!(stdout, EnterAlternateScreen, EnableMouseCapture, EnableAlternateScroll)?;
// In restore():
execute!(terminal.backend_mut(), DisableMouseCapture, DisableAlternateScroll, LeaveAlternateScreen)?;
// In set_mouse_capture(on):
// Also toggle AlternateScroll with capture
```

**`src/input/mouse.rs`**: Add handlers for `MouseEventKind::Drag(Left)` and `MouseEventKind::Up(Left)` in the `View::Cockpit` arm. The BODY area check (col >= FOLD_HIT_COLS) gates selection from fold-click. Add to `AppState`:
- `sel_anchor: Option<(usize, usize)>` (global_visual_row, col)
- `sel_end: Option<(usize, usize)>`

On `Down(Left)` with `col >= FOLD_HIT_COLS` in the transcript area: set `sel_anchor`, clear `sel_end`.  
On `Drag(Left)`: update `sel_end`.  
On `Up(Left)`: extract text from `sel_anchor` to `sel_end` using `join_visual_rows` (already in `src/render/copy.rs:162`), call `copy_to_clipboard`. Flash a notice.

**`src/app/mod.rs`** (or `src/app/types.rs`): Add `sel_anchor` and `sel_end` fields to `AppState`.

**`src/components/cockpit/transcript.rs`**: In the draw path, paint rows between `sel_anchor` and `sel_end` with a reversed highlight style.

### Fix B — Add `▾` turn header for expanded turns (click-to-collapse)

**Design**: Mirror tuiapp_v2's approach: **always render a turn header row**, whether collapsed or expanded. The header shows `▸` when folded, `▾` when expanded. For expanded turns, the header is also tagged `NodeId::Turn` in the node-hit table so a click on it collapses.

#### Files to change for Fix B:

**`src/markdown/mod.rs`** — `render_assistant_cockpit_full`, the match on `FoldSegment`:

Currently (line 159–184):
```rust
match seg {
    FoldSegment::Fold { title, turn, .. } => {
        logical.push((" ▸ " + title, Some(NodeId::Turn { block, turn })));
    }
    FoldSegment::Text { body } => {
        render_turn_body(body, ...);
    }
}
```

Change to:
```rust
match seg {
    FoldSegment::Fold { title, turn, .. } => {
        // Folded: ▸ header only (same as today)
        logical.push((fold_header_line(title, " ▸ ", theme, width), Some(NodeId::Turn { block: folds.block_id, turn: *turn })));
    }
    FoldSegment::Text { body, turn, .. } => {  // NOTE: FoldSegment::Text needs `turn` added
        // Expanded: ▾ header + body
        // The ▾ header is tagged NodeId::Turn so clicking it collapses
        if let Some(turn_num) = turn_number_for_segment(seg) {
            let is_last = /* check if this is the last segment */;
            if !is_last || folds.fold_all {
                // Only add a collapse header for non-last turns (or fold_all)
                logical.push((fold_header_line(&derive_title_from_body(body), " ▾ ", theme, width), Some(NodeId::Turn { block: folds.block_id, turn: turn_num })));
            }
        }
        render_turn_body(body, ...);
    }
}
```

**`src/render/fold.rs`** — `FoldSegment::Text` needs to carry the turn number so the renderer can tag the header with the correct `NodeId::Turn`. Currently `FoldSegment::Text { body: String }` has no turn metadata.

Change:
```rust
// Before:
pub enum FoldSegment {
    Text { body: String },
    Fold { turn: u32, title: String, body: String },
}

// After:
pub enum FoldSegment {
    /// A non-folded (expanded) text segment. `turn` is the 1-based turn number
    /// when this text came from a known turn marker (0 = pre-marker preamble or
    /// single-turn text). Used by the renderer to emit a `▾` collapse header.
    Text { body: String, turn: Option<u32> },
    Fold { turn: u32, title: String, body: String },
}
```

In `fold_turns_with` at `src/render/fold.rs:223`, the `FoldSegment::Text` push needs updating:
```rust
// Before:
segs.push(FoldSegment::Text { body: body.to_string() });
// After:
segs.push(FoldSegment::Text { body: body.to_string(), turn: Some(m.number) });
// For the preamble (before all markers), turn = None or 0
```

**Impact of this change**: All callers of `FoldSegment::Text` that pattern-match by `body` need to add `..` or `turn: _` in the destructure. In `render_assistant_cockpit_full` at `src/markdown/mod.rs:181`, change:
```rust
FoldSegment::Text { body } => { ... }
// to:
FoldSegment::Text { body, turn } => { ... }
// and add the ▾ header for expanded non-preamble turns.
```

**Visual spec for the ▾ header**:
- Same layout as the `▸` header: `" ▾ "` (space, down-triangle, space) = 3 cells, accent color (`Token::Claude`), then the title dim+italic.
- The title is derived the same way as `turn_title(body)` (already in `src/render/fold.rs:299`).
- The row is tagged `NodeId::Turn { block: folds.block_id, turn }` so clicking col 0–1 collapses it.
- The ▾ header is shown for all non-preamble expanded turns (i.e., turns 1..N when they have a turn marker). It is NOT shown for the preamble text segment (turn=None) and NOT shown for a single-turn text that has no `fold_all` context (mirror the condition in `fold_turns_with:183–191`).

### Fix C — Show ▸ → ▾ glyph rotation for tool boxes

For tool boxes, the existing `▾` collapse affordance row is already inside the box body (`src/render/chip.rs:299`). The issue is it's a single interior row with just `"▾"` at col 0. This IS within `FOLD_HIT_COLS=2` and IS tagged `NodeId::Tool`. This already works mechanically.

However the user says the expand/collapse "looks ugly." Reference tuiapp_v2 uses the same `▸`/`▾` pattern for tool bullets. The spec fix:

**`src/render/chip.rs:298–302`** — Change the affordance row text:
```rust
// Before:
body.push(if expanded { "▾".to_string() } else { format!("… +{overflow} more") });

// After — make the affordance more visible:
body.push(if expanded {
    format!("▾ collapse ({overflow} lines hidden when closed)")
} else {
    format!("▸ {overflow} more lines — click to expand")
});
```

Or simpler and cleaner: keep `… +N more` for collapsed (it's clear), but for expanded change `▾` to `▾ collapse` so it reads as a button.

**`src/app/fold_hit.rs:16`** — `FOLD_HIT_COLS=2` is correct for the `▸` (col 1) and `╭` (col 0). For the interior `▾` affordance row, col 0 is `│` (the interior border), col 1 is ` ` (padding), and col 2 is `▾`. This means the `▾` affordance is at col 2 which is >= FOLD_HIT_COLS=2, so clicking it does NOT trigger a fold toggle.

**This is a second latent bug**: The `▾` collapse row inside the box is at col 2, not col 0–1. The node-hit table covers the whole box (all rows including the affordance row), but `FOLD_HIT_COLS=2` rejects col >= 2. The tool box requires a wider click zone.

Fix: Increase `FOLD_HIT_COLS` to cover the full box width for tool nodes. The cleanest approach is to make the hit zone per-node-type:

**`src/app/fold_hit.rs`** — Change `transcript_node_at` to use a wider zone for `NodeId::Tool`:
```rust
// Before:
pub fn transcript_node_at(&self, col: u16, row: u16, transcript_top: u16) -> Option<NodeId> {
    if col >= FOLD_HIT_COLS || row < transcript_top { return None; }
    ...
}

// After:
pub fn transcript_node_at(&self, col: u16, row: u16, transcript_top: u16) -> Option<NodeId> {
    if row < transcript_top { return None; }
    let offset = (row - transcript_top) as usize;
    if offset >= self.viewport.height() { return None; }
    let visual = self.viewport.visual_top(&self.wrap_cache) + offset;
    let candidate = self.node_hit.iter()
        .find(|(range, _)| range.contains(&visual))
        .map(|(_, node)| *node);
    let Some(node) = candidate else { return None; };
    // Turn nodes: narrow gutter (col 0–1 = the ▸/▾ glyph zone).
    // Tool nodes: full width (the whole box is a click target including ▾ inside).
    let max_col = match node {
        NodeId::Turn { .. } => FOLD_HIT_COLS,
        NodeId::Tool { .. } => u16::MAX,   // full width
    };
    if col >= max_col { return None; }
    Some(node)
}
```

---

## (5) HONEST-CHECK

### Test that exercises the LIVE path and would FAIL today, PASS after the fix

#### Test B1 — expanded turn has a ▾ header row tagged NodeId::Turn (FAILS today)

```rust
/// LIVE-PATH: an expanded turn (forced open by toggling its fold) emits a `▾ title`
/// header row tagged NodeId::Turn in the STYLED frame, and a click on that row
/// collapses it back. Exercises apply_bridge_event → prepare_frame → render_to_rows.
///
/// FAILS TODAY: no ▾ header row for expanded turns; clicking the body rows returns None.
/// PASSES after Fix B: FoldSegment::Text carries turn number; ▾ header is emitted.
#[test]
fn expanded_turn_has_downward_triangle_header_and_can_be_recollapsed() {
    let mut app = AppState::new();
    app.conn = ConnStatus::Connected { model: Some("m".into()) };
    let src = "Turn 1 ...\n<summary>first task done</summary>\nbody1\nTurn 2 ...\n<summary>in progress</summary>\nbody2";
    let id = app.alloc_block_id();
    app.transcript.push(Block::new(id, None, Role::Assistant, src.to_string(), true));
    app.set_term_size(80, 40);
    let theme = Theme::default_theme();
    app.sync_transcript(80, 40, &theme);

    // Turn 1 is folded by default (▸); force-expand it.
    let node1 = NodeId::Turn { block: id, turn: 1 };
    app.toggle_fold(node1); // expand
    app.sync_transcript(80, 40, &theme);

    // STYLED frame must contain a ▾ row.
    let rows = render_to_rows(&mut app, 80, 40);
    let screen = rows.join("\n");
    assert!(screen.contains('▾'), "expanded turn must show a ▾ header:\n{screen}");

    // The ▾ row must be in the node-hit table as NodeId::Turn { turn: 1 }.
    assert!(
        app.node_hit.iter().any(|(_, n)| *n == node1),
        "NodeId::Turn{{turn:1}} must be in node_hit for the expanded ▾ header"
    );

    // A click on the ▾ header (col 0) must re-collapse turn 1.
    let area = ratatui::layout::Rect::new(0, 0, 80, 40);
    let top = crate::components::cockpit::split_cockpit(&app, area).transcript.y;
    let (range, _) = app.node_hit.iter()
        .find(|(_, n)| *n == node1)
        .expect("turn 1 in node_hit");
    let screen_row = top + (*range.start() - app.viewport.visual_top(&app.wrap_cache)) as u16;
    assert!(app.click_fold_at(0, screen_row, top), "click on ▾ must be handled");
    // After click: turn 1 is folded again.
    assert!(app.node_is_folded(node1), "turn 1 must be folded after clicking ▾");
}
```

#### Test B2 — tool box ▾ affordance is clickable at full width (FAILS today)

```rust
/// LIVE-PATH: the ▾ affordance row inside an expanded tool box is at col 2 (inside
/// the │ … │ interior). With FOLD_HIT_COLS=2, col 2 is NOT in the click zone.
/// FAILS TODAY: click_fold_at(2, affordance_row, top) returns false.
/// PASSES after Fix C: NodeId::Tool uses full-width hit zone.
#[test]
fn expanded_tool_box_affordance_row_is_clickable_at_interior_col() {
    let theme = Theme::default_theme();
    let mut app = AppState::new();
    let src = "Turn 1 ...\n🛠️ run({\"cmd\": \"ls\"})\nLINE1\nLINE2\nLINE3\nLINE4\nLINE5\nLINE6";
    let id = app.alloc_block_id();
    app.transcript.push(Block::new(id, None, Role::Assistant, src.to_string(), true));
    app.set_term_size(80, 40);
    app.sync_transcript(80, 40, &theme);

    // Expand the tool result first.
    let tool = NodeId::Tool { block: id, tool: 0 };
    app.toggle_fold(tool);
    app.sync_transcript(80, 40, &theme);

    // The ▾ affordance row is the LAST interior row. Find its screen row.
    let (range, _) = app.node_hit.iter().find(|(_, n)| *n == tool).expect("tool in node_hit");
    let area = ratatui::layout::Rect::new(0, 0, 80, 40);
    let top = crate::components::cockpit::split_cockpit(&app, area).transcript.y;
    let last_row_vis = *range.end();
    let screen_row = top + (last_row_vis - app.viewport.visual_top(&app.wrap_cache)) as u16;

    // Click at col 2 (inside the │ border), not col 0: must still hit the tool node.
    // FAILS today (col 2 >= FOLD_HIT_COLS=2 → None).
    assert!(
        app.transcript_node_at(2, screen_row, top).is_some(),
        "col 2 must hit a Tool node (full-width zone for tool nodes)"
    );
    assert!(app.click_fold_at(2, screen_row, top), "col 2 click on ▾ row must be handled");
    assert!(!app.folds.get(&tool).copied().unwrap_or(false), "tool must be collapsed after clicking ▾ at col 2");
}
```

#### Test A — app-internal drag-select copies to clipboard (only testable after Fix A)

This test requires `MouseEventKind::Drag` to be wired; it would FAIL today because `Drag` events are unhandled. After Fix A the test constructs a `Drag` sequence and asserts the copy-result flash appears in `app.notices`.

---

## Summary of changes by file

| File | Change |
|---|---|
| `src/term.rs` | Add `EnableAlternateScroll` (`\x1b[?1007h`) alongside `EnableMouseCapture`; or switch default to capture-off + `?1007h` only |
| `src/render/fold.rs` | Add `turn: Option<u32>` field to `FoldSegment::Text`; update `fold_turns_with` to propagate turn numbers |
| `src/markdown/mod.rs:159–184` | For `FoldSegment::Text` with a known turn number: push a `" ▾ title"` header row tagged `NodeId::Turn` before calling `render_turn_body` |
| `src/app/fold_hit.rs` | Change `transcript_node_at` to use full-width hit zone for `NodeId::Tool`; keep 2-cell zone for `NodeId::Turn` |
| `src/render/chip.rs:298–302` | Make collapsed affordance clearer (`▸ N more lines`) and expanded affordance clearer (`▾ collapse`) |
| `src/input/mouse.rs` | Add `Drag(Left)` and `Up(Left)` handlers for body area (col >= 2); record anchor/end in `AppState`; on `Up` extract + copy via `src/render/copy.rs::copy_to_clipboard` |
| `src/app/types.rs` | Add `sel_anchor: Option<(usize, usize)>`, `sel_end: Option<(usize, usize)>` to `AppState` |
| `src/components/cockpit/transcript.rs` | Paint selection highlight over `sel_anchor..=sel_end` row range |

---

VERDICT: REPRODUCED — (A) mouse select+copy blocked because `EnableMouseCapture` suppresses native drag-select with no usable workaround on Windows; (B) expanded turns have no `▾` header row and no `NodeId::Turn` hit entry, making click-to-collapse impossible; (C) the `▾` affordance inside expanded tool boxes sits at col 2 which is outside `FOLD_HIT_COLS=2`, so it is also unclickable.
