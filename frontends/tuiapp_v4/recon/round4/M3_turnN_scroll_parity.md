# M3 Verification Report — TURN-N / SCROLL / PARITY / BUDGETS / WARNINGS

**Date:** 2026-06-01  
**Verifier:** M3 (adversarial render-based)  
**Method:** `cargo run -- --dump-frame <scenario>` + source trace + unit tests

---

## 1. TURN-N — Three Shapes  CONFIRMED

### Shape (a): Preamble prose → `Turn 1 ...` → body (single marker, preamble before it)

**Code path:**  
`fold_turns_with` returns one `Text{body=whole_source}` (<2 markers).  
`render_turn_body(whole_source)` calls `strip_leading_turn_line`: the first line is prose, not a turn marker → no-op.  
Then `render_prose_with_inline_markers(body)` iterates line-by-line; for the `Turn 1 ...` line it calls `crate::render::chip::find_turn_line(line) == Some(0)` → TRUE → the marker is dropped and converted to a blank-line gap.

**Code evidence:**  
`src/markdown/mod.rs:511-516` (`render_prose_with_inline_markers` per-line turn-marker drop):
```rust
if crate::render::chip::find_turn_line(line) == Some(0) {
    flush(&mut prose_buf, &mut out);
    continue;
}
```

**Rendered proof:**  
`live_active_turn_marker_leaks_into_styled_frame` test (app/tests.rs:1028) seeds exactly this shape (`"Let me look into this.\nTurn 1 ...\nthe answer is 42"`) through the real streaming pipeline and asserts the styled frame never contains `"Turn 1"`. → PASSES.

`turn_marker_with_preamble_is_stripped` (markdown/tests) also covers `"Some intro prose.\nTurn 1 ...\nthe answer is 42"` at widths 40/80/100, streaming and finalized. → PASSES.

**Dump-frame grep:**  
All 10 scenarios rendered; `grep -i "Turn"` on every scenario output → zero matches.

### Shape (b): Bare single-turn message (`Turn 1 ...\nbody`, no preamble)

**Code path:**  
`fold_turns_with` → single marker, `fold_all=false`, `is_folded(1, true)=false` → returns `Text{body=whole}`.  
`render_turn_body` calls `strip_leading_turn_line("Turn 1 ...\nbody")`:  
`find_turn_line` at offset 0 matches → strips the marker line → body becomes `"body"`.

**Code evidence:**  
`src/markdown/mod.rs:555-565` (`strip_leading_turn_line`):
```rust
fn strip_leading_turn_line(body: &str) -> &str {
    if crate::render::chip::find_turn_line(body) == Some(0) {
        match body.find('\n') { Some(nl) => &body[nl + 1..], None => "" }
    } else { body }
}
```

**Test proof:**  
`turn_marker_not_rendered` (markdown/tests:1347) covers bare two-turn form with no summary. `no_turn_n_anywhere` (markdown/render/tests:587) covers a single-turn message at widths 40/80/120. Both PASS.

### Shape (c): Marker mid-tail behind open summary/fence (volatile streaming tail)

**Code path:**  
`render_assistant_cockpit_full` splits into `(head, tail)` at `safe_commit_pos`. The tail begins with a turn marker.  
`strip_leading_turn_line(tail)` at `src/markdown/mod.rs:191-196` is called before `render_turn_body`:
```rust
let tail_body = strip_leading_turn_line(tail);
```
This drops the leading `Turn N ...` line from the volatile tail.

**Test proof:**  
`tail_strips_turn_marker` (markdown/tests:1414) — seeds `"committed stable prose\n\nTurn 7 ...\n<summary>scanning the page"` with streaming=true, and a bold-marker variant. Both assert `!plain.contains("Turn 7")`. PASS.

### All Three Shapes: CONFIRMED

The fix is NOT a dump-only seed artifact. All three shapes are covered by:
1. `render_prose_with_inline_markers` line-loop (mid-body markers, shape a)
2. `strip_leading_turn_line` in `render_turn_body` (leading marker, shape b)
3. `strip_leading_turn_line` on the volatile tail in `render_assistant_cockpit_full` (streaming tail, shape c)

None of these paths are exclusive to the dump-frame fixture. The three unit tests (`live_active_turn_marker_leaks_into_styled_frame`, `turn_marker_with_preamble_is_stripped`, `tail_strips_turn_marker`) drive real streaming state, not the dump-frame fixture.

---

## 2. SCROLL WIRING  CONFIRMED (code; live confirmation requires real terminal)

### Mouse capture defaults ON

`src/term.rs:29`:
```rust
execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
```
`src/app/mod.rs:260`:
```rust
mouse_capture: true,
```
These are synchronized: `term::setup` calls `EnableMouseCapture` on launch; `AppState::default` initializes `mouse_capture: true`. The comment at app/mod.rs:255-259 explicitly states "Mouse capture starts ON so the WHEEL scrolls (crossterm only delivers ScrollUp/Down under capture — Slice 0 root cause)."

### Mouse events route to scroll

`src/input/mouse.rs:40-41`:
```rust
(View::Cockpit, MouseEventKind::ScrollUp) => app.scroll_lines(-WHEEL_STEP),
(View::Cockpit, MouseEventKind::ScrollDown) => app.scroll_lines(WHEEL_STEP),
```

`src/app/mod.rs:571-573`:
```rust
pub fn scroll_lines(&mut self, delta: isize) {
    self.viewport.scroll_by(delta, &self.wrap_cache);
}
```

### PageUp/PageDown/Home/End route to viewport

`src/input/keymap.rs:262-272`:
```rust
KeyCode::Home if ctrl => app.scroll_home(),
KeyCode::End if ctrl => app.scroll_end(),
KeyCode::Home => { /* content depends on composer context */ }
KeyCode::End => { /* content depends on composer context */ }
KeyCode::PageUp => app.page_up(),
KeyCode::PageDown => app.page_down(),
```
`app.page_up/down/scroll_home/scroll_end` delegate to `self.viewport.*` (app/mod.rs:576-593).

**Test proof (live-order):**  
`page_up_and_wheel_move_the_live_viewport` (app/tests.rs:1056) seeds 60 system lines, renders a frame to sync geometry, calls `app.page_up()`, re-renders, and asserts the tail line scrolled out of view. Wheel-up via `scroll_lines(-3)` is also asserted to move the top. PASSES.

**Note:** Full confirmation that crossterm delivers wheel events requires a real PTY. The code path is wired correctly; in a POSIX TTY or Windows Console with mouse enabled, crossterm delivers `ScrollUp/ScrollDown`; on some terminal emulators (no-mouse mode, SSH without forwarding) wheel events may not arrive. This is a platform-level caveat, not a code gap.

---

## 3. PARITY INVARIANTS  CONFIRMED — 354 passed, 0 failed

The four named parity invariants all PASS:

| Test | File | Status |
|------|------|--------|
| `styled_wrap_rowcount_matches_wrap_cache` | markdown/tests | PASS |
| `cockpit_render_rowcount_matches_plain_projection` | markdown/tests | PASS |
| `cockpit_full_rowcount_matches_projection_under_overrides` | markdown/tests | PASS |
| `embedded_newline_in_span_keeps_rowcount_parity` | markdown/tests | PASS |

Full test suite: **354 passed; 0 failed; 1 ignored** (the integration bridge test, requires live Python).

### 0 warnings

`cargo build` and `cargo test` emit **zero compiler warnings**.

---

## 4. GOD-FILE BUDGETS  CONFIRMED (all within limits)

Non-test line counts (up to `#[cfg(test)]` or total when no test block):

| File | Non-test lines | Budget | Status |
|------|----------------|--------|--------|
| `src/main.rs` | 609 | ≤700 | PASS |
| `src/app/mod.rs` | 743 | ≤800 | PASS |
| `src/markdown/render.rs` | 540 | <600 | PASS |
| `src/components/cockpit/composer.rs` | 251 | ≤400 | PASS |
| `src/components/cockpit/dropdown.rs` | 154 | ≤400 | PASS |
| `src/components/cockpit/footer.rs` | 200 | ≤400 | PASS |
| `src/components/cockpit/header.rs` | 139 | ≤400 | PASS |
| `src/components/cockpit/mod.rs` | 238 | ≤400 | PASS |
| `src/components/cockpit/transcript.rs` | 145 | ≤400 | PASS |
| `src/components/overlay/mod.rs` | 74 | ≤200 | PASS |

---

## Summary

| Item | Verdict |
|------|---------|
| TURN-N shape (a): preamble+single-marker | CONFIRMED |
| TURN-N shape (b): bare single-turn | CONFIRMED |
| TURN-N shape (c): marker mid-tail/streaming | CONFIRMED |
| SCROLL: mouse capture default ON | CONFIRMED |
| SCROLL: wheel events → scroll_lines | CONFIRMED |
| SCROLL: PageUp/Down/Home/End → viewport | CONFIRMED |
| SCROLL: live terminal confirmation | NOTE (requires real PTY) |
| PARITY: 4 invariants named | CONFIRMED (354 pass, 0 fail) |
| WARNINGS: 0 compiler warnings | CONFIRMED |
| BUDGETS: all god-files within limits | CONFIRMED |

**No GAPs found.**
