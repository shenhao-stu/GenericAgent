# R3 — Honest Root-Cause Diagnosis of Three "Fixed-but-Failing" Runtime Bugs

**Scope:** `frontends/tuiapp_v4` (Rust + ratatui, v0.3.0). The prior round (commit
`d120173`, the round-3 polish) claimed FAILURES A/B/C fixed via headless
`--dump-frame` + unit tests on CLEAN fixtures. This report traces the REAL code
paths, drives the REAL render pipeline (`apply_bridge_event` → `prepare_frame` →
`terminal.draw` → scan the STYLED frame buffer), and reports what is genuinely
broken vs. genuinely fixed — with file:line.

**Method.** I built the binary, ran `--dump-frame busy|normal`, and added
end-to-end tests that render the WHOLE cockpit to a `ratatui::backend::TestBackend`
and scan the styled cell grid (`src/app/tests.rs`, the new `render_to_rows`
helper). This is the path the prior round's "unit-on-clean-fixtures" tests
bypassed.

## TL;DR verdict

| Failure | Claimed | Reality in the running binary | Genuine bug? |
|---|---|---|---|
| **A — cannot scroll** | broken | **WORKS** — PageUp/PageDown/wheel/Home/End all move the live viewport offset and the styled frame re-windows | **NO** (fixed in round 3) |
| **B — markdown not live** | broken | **WORKS** — headings (bold + `## ` glyph), bold/italic/code/lists/inline-math all render formatted while streaming | **NO** (the `## ` glyph is intentional, not a leak) |
| **C — `Turn N` leaks** | broken | **GENUINELY BROKEN** — a single `Turn N ...` marker preceded by preamble prose (the canonical GA reply) leaks the literal marker into the STYLED transcript | **YES** |

The lead's structural hypothesis for C is **correct**; A and B were actually
repaired in round 3 and the "false fix" claim does not hold for them under the
real pipeline.

---

## FAILURE A — CANNOT SCROLL → **not reproducible; scroll works**

### The live wiring (verified end-to-end)

- **Event loop** (`src/main.rs:443` `event_loop`): per iteration it (1)
  `prepare_frame` + `terminal.draw`, then (2) `rx.recv_timeout` →
  `handle_term_event` (`src/main.rs:503`). Keys in the cockpit route
  `route_view_key` → `keymap::cockpit_key` (`src/input/views.rs:36`).
- **Scroll keys** (`src/input/keymap.rs`): `PageUp → app.page_up()` (line 271),
  `PageDown → app.page_down()` (272), `Ctrl+Home → app.scroll_home()` (262),
  `Ctrl+End → app.scroll_end()` (263). `PageUp/PageDown` are NOT intercepted by
  `try_complete_dropdown` (only Up/Down/Tab/Enter are, lines 304–331), so they
  always reach the viewport.
- **Wheel** (`src/input/mouse.rs:40-41`): `ScrollUp → app.scroll_lines(-3)`,
  `ScrollDown → app.scroll_lines(+3)` (`WHEEL_STEP = 3`).
- **App → viewport** (`src/app/mod.rs:533-555`): `scroll_lines`/`page_up`/
  `page_down`/`scroll_home`/`scroll_end` delegate to
  `Viewport::scroll_by/page_up/page_down/home/end` (`src/render/viewport.rs:124-176`).
- **Offset applied to the rendered frame**: `render_transcript`
  (`src/components/cockpit/transcript.rs:37`) draws
  `app.viewport.visible(&app.wrap_cache)`, which is the `[visual_top, +height)`
  slice (`viewport.rs:232-235`). The anchor set by a keypress survives the next
  frame because `prepare_frame`→`sync_transcript` only calls `viewport.resize`
  on a **width or height change** (`src/app/mod.rs:500` and `508`), not every
  frame, and `resize` preserves a non-`Bottom` anchor (`viewport.rs:189-203`).

### Why it actually works (and why the lead thought it didn't)

The round-3 polish (commit `d120173`) wired all of the above. My E2E test
`page_up_and_wheel_move_the_live_viewport` (`src/app/tests.rs`) seeds 60 lines,
renders, presses PageUp, re-renders, and asserts the tail line scrolls OUT of the
styled frame; then wheel-up moves `visual_top` further; then `scroll_end`
restores follow. **It passes.** A unit test on a hand-built `WrapCache` would also
pass — but so does the real pipeline, so there is no gap here. **No fix needed.**

---

## FAILURE B — MARKDOWN NOT RENDERING LIVE → **not reproducible; markdown runs live**

### The live vs. finalized render path (they are the SAME)

- Both the streaming draw (`Block::cockpit_line`, `src/app/types.rs:270`) and the
  plain wrap-cache projection (`Block::to_render_block`, `types.rs:236`) call the
  ONE renderer `render_assistant_cockpit_full` (`src/markdown/mod.rs:112`) via the
  block-owned memo `ensure_cockpit_cache` (`types.rs:307-325`), passing
  `streaming = !self.finalized`.
- The volatile streaming tail is NOT dumped raw: `render_assistant_cockpit_full`
  routes the tail (after `safe_commit_pos`) through the SAME `render_turn_body`
  path as the committed head (`src/markdown/mod.rs:191-197`, "Fix D"). I verified
  with `--dump-frame busy` (an in-flight turn) and a probe: `**bold**`→`bold`,
  `- item`→`• item`, `# Title`→ styled heading, inline `$…$`→ glyphs — all while
  streaming. An UNCLOSED `**bol` correctly stays raw until its closer arrives
  (correct streaming behavior, held by `safe_commit_pos`).

### The one thing that LOOKS like a leak but is intentional

`## Heading One` renders on screen WITH a leading `## ` — because
`code::heading_style` (`src/markdown/code.rs:53-62`) deliberately emits `"## "` as
a restrained, CC-style **heading glyph** (the row is also BOLD + colored). My E2E
test `streaming_markdown_is_formatted_in_styled_frame` (`src/app/tests.rs`) asserts
the heading row carries `Modifier::BOLD` and that raw `**` never appears — **it
passes**. So the markdown walker ran; the `## ` is style, not unformatted text.

**Honest caveat:** if the *product intent* is that `## ` should be hidden (clean
bold "Heading One" with no `#`), that is a DESIGN change in `heading_style`, not a
streaming bug — and it would be identical live and finalized. There is no
live-vs-finalized markdown divergence to fix.

---

## FAILURE C — `Turn N` LEAKS → **GENUINELY BROKEN. Pinned.**

### The exact leaking function:line

**`render_prose_with_inline_markers` at `src/markdown/mod.rs:449` (the per-line
loop, lines 449–468)** is where the literal `Turn N ...` reaches the markdown
walker and is drawn as prose. The chain that delivers a marker to it:

1. **`fold.rs:183-189`** — `fold_turns_with`, the `markers.len() < 2`
   short-circuit: with a SINGLE turn marker that the closure does not fold, it
   returns the **WHOLE text as one `FoldSegment::Text { body }`** — preamble +
   marker line + body together. (For ≥2 markers, lines 208-228 split each turn so
   each segment body STARTS with its marker — that is why the multi-turn fixtures
   never leak.)
2. **`mod.rs:175`** — `render_assistant_cockpit_full` hands that whole-text body to
   **`render_turn_body` (`mod.rs:288`)**.
3. **`mod.rs:305`** — `render_turn_body` calls
   **`strip_leading_turn_line` (`mod.rs:498`)**, which only strips when the marker
   is **line 0** (`find_turn_line(body) == Some(0)`, `mod.rs:500`). When PREAMBLE
   prose precedes the marker, line 0 is the prose, so the marker is NOT stripped.
4. **`mod.rs:329`** — the un-stripped body flows to
   `render_prose_with_inline_markers`, whose loop (`mod.rs:449`) had no
   turn-marker case, so the `Turn N ...` line fell into `prose_buf` (`mod.rs:466`)
   and was rendered verbatim by `render_assistant` (`mod.rs:444`).

This leaks on **BOTH** the styled draw and the plain projection (both call
`render_turn_body`), and on **both** `streaming=true` and `streaming=false`.

A second, independent path: the streaming TAIL strip (`mod.rs:192`
`strip_leading_turn_line(tail)`) has the same line-0-only blind spot — a marker
sitting mid-tail (stable prose before it, whole tail held back by an open
`<summary>`/fence/`$$`) leaks the same way.

A third, lower-severity raw-source leak (out of the main transcript, but real):
the **dashboard preview** `preview_line` (`src/app/session.rs:733-742`) takes
`last_non_blank_line(&b.source)` straight from the RAW source, so if the newest
assistant line is a bare `Turn N ...` it shows in the session card.

### Proof in the REAL STYLED frame

E2E test `live_active_turn_marker_leaks_into_styled_frame` (`src/app/tests.rs`)
streams `"Let me look into this.\nTurn 1 ...\nthe answer is 42"` through
`apply_bridge_event(MessageBegin) + MessageDelta`, renders the cockpit to a
`TestBackend`, and scans the cells. **Before the fix it FAILED**, with the frame
showing the literal row:

```
Let me look into this. Turn 1 ... the answer is 42
```

### Why `no_turn_n_anywhere` did NOT catch it

`no_turn_n_anywhere` (`src/markdown/render.rs:591`) renders ONLY via
`render_assistant_cockpit_plain` (→ `streaming=false`) on the fixture
`"Turn 1 ...\n🛠️…\n[Info] ok\nTurn 2 ...\nTurn 3 ...\nthe final answer is 42"`.
That fixture (a) starts at `Turn 1 ...` (marker IS line 0) and (b) has **3
markers**, so `fold_turns_with` takes the per-turn split (`fold.rs:208`) and EVERY
segment body begins with its own marker → every marker is line-0 of its segment →
all stripped. The test therefore never exercises a marker preceded by non-marker
text in the same `Text` segment — exactly the single-marker / preamble / mid-tail
shape that leaks. (The `app/tests.rs` `THREE_TURNS` fixture and the
`tail_strips_turn_marker` test share the same blind spot: marker always at a
segment/tail line-0.)

### Minimal fix (applied + validated)

Drop a `Turn N ...` marker line **wherever** it appears in a rendered turn body —
it is spacing, never content — instead of only when it is line 0. Implemented in
the per-line loop of `render_prose_with_inline_markers`
(`src/markdown/mod.rs:449`):

```rust
for line in text.split('\n') {
    let l = line.trim_start();
    // A `Turn N ...` marker line is spacing, NEVER content — drop it wherever it
    // appears in a turn body, not just the leading line. (C1-F6 / Q8 leak.)
    if crate::render::chip::find_turn_line(line) == Some(0) {
        flush(&mut prose_buf, &mut out); // keep the §2.4 turn-separation spacing
        continue;
    }
    if l.starts_with("!!!Error") { /* … unchanged … */ }
    /* … */
}
```

This sits on the path BOTH the styled draw and the plain projection share, so the
P1 row-count parity (`render_assistant_cockpit_full` vs. the wrap cache) is
preserved by construction. (A belt-and-braces variant would also make
`strip_leading_turn_line` at `mod.rs:192/305` strip any leading-or-interior marker,
but the single fix above already closes every reproduced case.)

**Optional companion fix (dashboard only):** in `preview_line`
(`src/app/session.rs:733`) skip a `Turn N ...` line when choosing the preview
(reuse `crate::render::chip::find_turn_line`).

### Verification that exercises the REAL path

- `live_active_turn_marker_leaks_into_styled_frame` (`src/app/tests.rs`): feeds a
  live, un-ended active turn carrying a `Turn N ...` marker via
  `apply_bridge_event` `MessageBegin`+`MessageDelta`, renders the whole cockpit to
  a `TestBackend`, and scans the **STYLED** cell grid for `"Turn 1"`. **Now passes.**
- `turn_marker_with_preamble_is_stripped` (`src/markdown/mod.rs`): renders the
  preamble / mid-tail shapes through `render_assistant_cockpit_full` for BOTH
  `streaming ∈ {false,true}` × widths `{40,80,100}`; asserts no `"Turn"` and that
  the surrounding prose survives.
- Full suite: **335 passed, 0 failed** (incl. `no_turn_n_anywhere`, `fold_summary`,
  and the P1 row-count-parity tests) — no regression.

---

## Files touched (all additive, all green)

- `src/markdown/mod.rs` — the C fix (per-line marker drop at line 449) + the
  `turn_marker_with_preamble_is_stripped` regression test.
- `src/app/tests.rs` — three end-to-end STYLED-frame verifications
  (`live_active_turn_marker_leaks_into_styled_frame`,
  `page_up_and_wheel_move_the_live_viewport`,
  `streaming_markdown_is_formatted_in_styled_frame`) + the `render_to_rows` helper.

## Reality check on the round-3 "false fix" claim

Scroll (A) and live markdown (B) were genuinely repaired in round 3 and hold up
under the real render pipeline — the "false fix" framing does not apply to them.
Only the `Turn N` strip (C) was incompletely fixed: round 3 added the leading-line
strip + a clean-fixture test (`tail_strips_turn_marker`, marker-at-line-0), which
structurally cannot observe the preamble/mid-body marker. That is the single real
defect, and it is now closed with the test gap covered.
