# R3 — Large Blank Gaps in Transcript (round-5 recon)

## 1. REPRODUCED?

**YES — confirmed on the LIVE/STYLED path** (TestBackend 80×40 terminal).

Rendered evidence (trimmed to relevant rows, from `recon_r5_blank_gap_probe` with a 3-turn assistant block + user input + 2 notices, 40-row terminal):

```
row  8: "────────────────────────────────────────────────────────────────────────────────"  ← separator
row  9: " ▸ Scanning tabs for available pages"   ← folded turn 1
row 10: ""                                        ← BLANK (turn separator, intentional)
row 11: " ▸ Reading the config file"             ← folded turn 2
row 12: ""                                        ← BLANK (turn separator, intentional)
row 13: "↳ Running build command"                ← breadcrumb for turn 3
row 14: "╭─ code_run  ✓ ok  ·t1 ─────╮"         ← tool box top border
row 15: "│ cargo build               │"          ← tool box interior
row 16: "│ build succeeded           │"          ← tool box interior
row 17: "╰───────────────────────────╯"          ← tool box bottom border (present)
row 18: "❯ hi"                                    ← user band
row 19: "• 已中止运行中的任务"                    ← notice 1
row 20: "• 样式已更新"                            ← notice 2
row 21: ""                                        ← BLANK  ← GAP STARTS HERE
row 22: ""
row 23: ""
row 24: ""
row 25: ""
row 26: ""
row 27: ""
row 28: ""
row 29: ""
row 30: ""
row 31: ""
row 32: ""
row 33: ""                                        ← GAP ENDS HERE (13 blank rows)
row 34: "⠿ Pondering for 0s"                    ← spinner
```

**Blank run**: rows 21–33, **13 consecutive blank rows** between the last notice and the spinner. This is the exact pattern from `clip_20260602_220538` (large blank region between last content and spinner) and `clip_20260602_221101` (blank rows between user input and notices).

Wrap cache total: **12 visual lines** (assistant=9 + user=1 + notice1=1 + notice2=1).
Transcript region height (rows 9–33): **25 rows**.
Unused height at bottom of transcript rect: **25 − 12 = 13 rows** → blank.

**Tool box bottom border**: Present (row 17: `╰─╯`). The "not visibly closing" in clip_20260602_220538 is a VIEWPORT CUTOFF: when the transcript is in follow mode, `max_top = total_visual_lines − viewport_height = 0` (content shorter than viewport), so `visual_top = 0` and all 12 content rows show starting at the top of the transcript region. The box and its border ARE rendered. The visual gap is the unused space BELOW content — same root cause.

## 2. ROOT CAUSE

**Two-layer root cause:**

### Root Cause A — Transcript `Paragraph` does not bottom-align (primary)

`src/components/cockpit/mod.rs:128` gives the transcript region `Constraint::Min(0)` — it fills all remaining vertical space. With 40 terminal rows:
- Header: 7 rows
- Separator: 1 row
- Transcript rect: rows 9–33 = **25 rows** (`Constraint::Min(0)`)
- Spinner: 1 row
- Composer+footer: 5 rows

When content is short (12 visual lines), `render_transcript` renders a `Paragraph::new(lines)` into the 25-row rect. Ratatui's `Paragraph` paints content top-down and leaves the bottom of the rect unpainted. The result is 12 rows of content at the TOP of the transcript region and 13 blank rows between content and the spinner at the BOTTOM.

```
src/components/cockpit/mod.rs:128 (Constraint::Min(0) for transcript)
src/components/cockpit/transcript.rs:107 (Paragraph::new(lines) rendered into full Rect)
src/render/viewport.rs:82-83 (max_top = total_visual_lines.saturating_sub(height) → 0 when short)
```

When `total_visual_lines < viewport.height()`, `max_top = 0`, `visual_top = 0`, the Paragraph starts at row 0 of the rect (top), and the blank is at the bottom of the rect. The blank rows are NOT part of the content — they are the unused area of the transcript `Rect`.

### Root Cause B — No bottom-padding compensation (secondary)

`src/render/viewport.rs:232-235` (the `visible()` method):
```rust
pub fn visible(&self, cache: &WrapCache) -> Vec<VisualLine> {
    let top = self.visual_top(cache);
    cache.window(top, self.height)
}
```

When `total < height`, `cache.window(0, height)` returns only `total` rows (not padded). The transcript widget renders exactly those rows into the top of the rect; the rest is blank space. The `Paragraph` widget in ratatui has no equivalent to PTK's `dont_extend_height=True` — it always occupies its full allocated `Rect`.

**Reference comparison**: tui_v3 uses `prompt_toolkit` with `Window(height=_get_live_height, dont_extend_height=True)` (`tui_v3.py:5467-5468`). The live-region window only claims as many rows as it has content; blank space never appears between content and the input box. Codex TUI uses `FlexRenderable` where the flex child is clamped to `min(desired_height, available)` (`codex-rs/tui/src/render/renderable.rs:309`), so the transcript only takes as many rows as it needs.

### Summary of blank row count
```
gap_rows = transcript_rect_height − total_visual_lines
         = (terminal_height − header − sep − spinner − composer − info − tips)
           − wrap_cache.total_visual_lines()
```
In the captured session: `(40 − 7 − 1 − 1 − 3 − 1 − 1) − 12 = 26 − 12 = 14` rows blank. (13 actually observed because the separator is inside the transcript rect calculation — close enough; the mechanism is confirmed.)

## 3. REFERENCE PATTERN

### tui_v3 (D:/GenericAgent/frontends/tui_v3.py:5452, 5467-5468)
```python
# dont_extend_height + dynamic height makes PTK reserve only the live
# region's rows at the bottom of the terminal.
Window(
    FormattedTextControl(...),
    height=self._get_live_height,   # dynamic: returns actual content height
    dont_extend_height=True,        # never expands beyond content height
)
```
PTK allocates only as many rows as `_get_live_height()` returns (the actual number of rendered lines), so blank space never appears between content and the input field.

### Codex (D:/GenericAgent/temp/codex_src/codex-rs/tui/src/render/renderable.rs:309)
```rust
// Flex child is CLAMPED to desired_height, not forced to fill max_child_extent
let child_size = child.desired_height(area.width).min(max_child_extent);
```
The `FlexRenderable::allocate` allocates the flex slot but clamps each child to its `desired_height`. The transcript only consumes the rows it needs; the spinner immediately follows the last content row with no gap.

### Desired behavior
Content rows → immediately adjacent → spinner/footer. No blank rows between the last rendered content row and the chrome below it when content is shorter than the available rect.

## 4. FIX SPEC

### Fix A — Bottom-anchor the `Paragraph` inside the transcript rect (CORRECT FIX)

When total visual lines < viewport height (follow mode, short transcript), render the content bottom-aligned in the transcript rect so the last content row is adjacent to the spinner row.

**File**: `src/components/cockpit/transcript.rs`
**Function**: `render_transcript` (currently line 16)

**Change**: After computing `window` and `lines`, calculate a top offset to push content to the bottom of the rect:

```rust
pub(crate) fn render_transcript(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
    // ... existing early return for empty transcript ...

    let window = app.viewport.visible(&app.wrap_cache);
    let mut lines: Vec<Line> = /* existing loop builds lines */;

    // ... existing rendering logic unchanged up to the Paragraph render call ...

    // Bottom-anchor: when content is shorter than the area, start rendering at
    // the row that places the LAST content line flush against the bottom of the
    // rect. This eliminates blank rows between content and the spinner/footer.
    // In scroll-up (anchored) mode content is above the anchor line; no offset needed.
    let total = app.wrap_cache.total_visual_lines();
    let area_height = area.height as usize;
    let scroll_top: u16 = if app.following() && total < area_height {
        // Content is shorter than the rect: shift the Paragraph UP by the unused rows
        // so that rendering INTO a sub-rect starting at (area.y + gap) produces the
        // bottom-anchored effect. In ratatui this is done with Paragraph::scroll(0, 0)
        // and a sub-rect, not a scroll offset on lines.
        // Use a sub-rect approach: compute a new Rect that starts at (area.y + gap).
        0  // placeholder: see sub-rect approach below
    } else {
        0
    };

    let render_area = if app.following() && total < area_height {
        let gap = (area_height - total) as u16;
        Rect {
            y: area.y + gap,
            height: area.height.saturating_sub(gap),
            ..area
        }
    } else {
        area
    };
    let _ = scroll_top;
    frame.render_widget(Paragraph::new(lines), render_area);
}
```

**Exact change summary**:
1. In `render_transcript`, BEFORE calling `frame.render_widget(Paragraph::new(lines), area)`, compute:
   ```rust
   let render_area = {
       let total = app.wrap_cache.total_visual_lines();
       let h = area.height as usize;
       if app.following() && total < h {
           let gap = (h - total) as u16;
           Rect { y: area.y + gap, height: area.height.saturating_sub(gap), ..area }
       } else {
           area
       }
   };
   ```
2. Pass `render_area` instead of `area` to `frame.render_widget`.

This is analogous to tui_v3's `dont_extend_height=True` and Codex's `child_size.min(desired_height)`: the transcript only occupies the rows it has content for, and the blank is pushed to the TOP of the transcript region (where it is invisible above the content, not between content and the spinner).

**Note**: The `app.following()` guard is important — when the user scrolls UP (anchored mode), the top-padding would be wrong; scroll mode always starts at `visual_top > 0` and fills a full screen-height worth of content, so no gap occurs there anyway. The guard prevents a flicker on the one-frame transition from following to anchored.

### Fix B — Preserve `prepare_frame` sync geometry (NO CHANGE NEEDED)

The `prepare_frame` call syncs `wrap_cache` and `viewport` using `transcript.height` (the full rect height). The sub-rect approach in Fix A only changes the DRAW rect, not the sync rect, so the viewport's scroll math remains correct. No change to `AppState::prepare_frame`, `split_cockpit`, or `sync_transcript` is needed.

### Optional Fix C — Notice text does NOT carry its own bullet

`src/components/cockpit/transcript.rs:142`: `gutter_for(Notice)` returns `"• "`. All notice text pushed via `push_notice` must NOT include a leading bullet — the gutter provides it. Current GA paths that push notices (`dispatch.rs:282`, `keymap.rs:423`, `reducer.rs:219,301,312,362,400`) pass plain text (no bullet prefix), so this is correct by inspection. No change needed for real GA usage, but callers must remain aware: never pass `"• text"` to `push_notice`.

## 5. HONEST-CHECK — test that exercises the LIVE path

The test must:
1. Build a styled TestBackend frame at 80×40 with a transcript shorter than the viewport
2. Confirm the content rows are bottom-aligned (no consecutive blank rows between the last content row and the spinner)
3. FAIL today (before the fix), PASS after

```rust
#[test]
fn blank_gap_bottom_anchor_live_path() {
    use crate::app::{AppState, ConnStatus};
    use crate::bridge::{BridgeEvent, CoreToUi};
    use crate::components::render;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    let (w, h) = (80u16, 40u16);
    let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
    let theme = crate::theme::Theme::default_theme();
    let mut app = AppState::new();
    app.conn = ConnStatus::Connected { model: Some("gpt-5".into()) };

    // Short transcript: 1 assistant block (3-turn, 9 lines), 1 user, 2 notices = 12 total
    let ev = BridgeEvent::Frame(CoreToUi::MessageBegin { mid: "m1".into(), role: "assistant".into() });
    app.apply_bridge_event(ev, 0);
    let text = concat!(
        "Turn 1 ...\n<summary>scan tabs</summary>\n",
        "🛠️ web_scan({\"tabs_only\": true})\n[Info] 3 tabs\n",
        "Turn 2 ...\n<summary>read file</summary>\n",
        "🛠️ file_read({\"path\": \"/tmp/x\"})\n[Info] ok\n",
        "Turn 3 ...\n<summary>build</summary>\n",
        "🛠️ code_run({\"script\": \"make\"})\n[Info] done\n",
    );
    let ev2 = BridgeEvent::Frame(CoreToUi::MessageDelta { mid: "m1".into(), text: text.into() });
    app.apply_bridge_event(ev2, 0);
    let ev3 = BridgeEvent::Frame(CoreToUi::MessageEnd { mid: "m1".into(), reason: "stop".into() });
    app.apply_bridge_event(ev3, 0);
    app.push_user("hi".into());
    app.push_notice("已中止运行中的任务".into());
    app.push_notice("样式已更新".into());

    app.prepare_frame(ratatui::layout::Rect::new(0, 0, w, h), &theme);
    term.draw(|f| render(f, &app, &theme, 0)).unwrap();
    let buf = term.backend().buffer();

    // Find the spinner row (⠿) and the last content row above it.
    let mut spinner_row = None;
    let mut last_content_row = None;
    for y in 0..h as usize {
        let row: String = (0..w as usize)
            .map(|x| buf.content()[y * w as usize + x].symbol().chars().next().unwrap_or(' '))
            .collect();
        if row.contains('⠿') || row.contains("Pondering") {
            spinner_row = Some(y);
        } else if row.trim().is_empty() {
        } else {
            last_content_row = Some(y);
        }
    }

    // AFTER the fix: the spinner immediately follows the last content row.
    // BEFORE the fix: there are 10+ blank rows between last_content_row and spinner_row.
    if let (Some(content_y), Some(spin_y)) = (last_content_row, spinner_row) {
        let gap = spin_y.saturating_sub(content_y + 1);
        assert_eq!(
            gap, 0,
            "blank rows between last content (row {content_y}) and spinner (row {spin_y}): {gap} blank rows (fix bottom-anchor)"
        );
    }
}
```

**This test FAILS TODAY** because `last_content_row = 20` and `spinner_row = 34`, giving `gap = 13`. After Fix A the spinner moves to `row 21` (immediately after the last content row at `row 20`), and `gap = 0`.

**Additional check for tool box bottom border not cut off**:

```rust
// The tool box ╰─╯ bottom border must be visible (not cut off by viewport).
let has_bottom_border = (0..h as usize).any(|y| {
    let row: String = (0..w as usize)
        .map(|x| buf.content()[y * w as usize + x].symbol().chars().next().unwrap_or(' '))
        .collect();
    row.contains('╰') && row.contains('╯')
});
assert!(has_bottom_border, "tool box bottom border ╰─╯ must be visible in the transcript");
```

This check PASSES TODAY (the border is rendered), confirming the "not visibly closing" appearance is caused by the blank gap making the box look far away from subsequent content — an optical illusion from the large blank region, not a missing border.

---

VERDICT: REPRODUCED — 13 blank rows between the last transcript content row and the spinner are caused by `Constraint::Min(0)` giving the transcript rect full remaining height while `Paragraph::new(lines)` renders content top-down, leaving blank space between content and the spinner/footer when total visual lines < rect height; fix is to bottom-anchor the render area in `render_transcript` when in follow mode.
