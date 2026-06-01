# tui_v4 — Round-4 Implementation Plan

**Goal:** Make tui_v4 faithfully reproduce tui_v3's look (multi-line header box, bordered
tool-call boxes, `▸` narration) and the CC spinner-with-tip, fix the real-terminal
failures (wheel scroll, Turn-N leak, blank effort/ctx, dead per-command effects), and
add `@` completeness + `/continue` search — all verified against the **live/styled
render path**, not clean fixtures.

**Method:** sequenced, gated slices (halt-on-break). Each slice: implement → `cargo build`
→ `cargo test` → an HONEST acceptance check that exercises the live path → `VERDICT: PASS|FAIL`.
Source specs: `recon/round4/R1..R6`. **Lesson encoded from round 3:** a passing unit test on
a clean fixture ≠ correct in the running binary. Every render check below must feed text
through `apply_bridge_event` (the bridge path) and/or scan the **styled** `TestBackend`
cell grid, or drive real key/mouse events — never assert only on a hand-built fixture.

**Architecture invariants (must stay green):** the 4 parity tests
(`styled_wrap_rowcount_matches_wrap_cache`, `plain_projection_matches_rendered_text`,
`cockpit_render_rowcount_matches_plain_projection`, `embedded_newline_in_span_keeps_rowcount_parity`);
god-file budgets (main.rs ≤700, app/mod.rs ≤800, render.rs <600, cockpit/* ≤400,
overlay/mod.rs <200). Reduce redundant comments per `memory/code_review_principles.md`.

---

## Slice 0 — Wheel scroll (the real "can't scroll") + Turn-N confirm  [FOUNDATION]

**Root cause (R3 + lead):** `term.rs:67` puts `EnableMouseCapture` only in the
`set_mouse_capture(on)` ON-branch; capture defaults OFF (round-3 Q2, for native copy);
`mouse.rs:40-41` wheel scroll needs `ScrollUp/ScrollDown` events crossterm only delivers
**when capture is on**. So in the real terminal the wheel is dead — R3's TestBackend tested
`scroll_lines` directly, bypassing the capture gate (the same headless blind spot as round 3).

**Change:**
- `src/term.rs` `setup()`: enable mouse capture by default so the wheel scrolls; keep the
  `set_mouse_capture` toggle + `/mouse`/Ctrl+Shift+M to turn it OFF for pure native drag-select.
- `src/app/mod.rs` `Default`: `mouse_capture: true`.
- Add a one-line composer/tip hint: "Shift+drag to select" (terminal-standard under capture).
- **Turn-N:** R3 already fixed the source leak (`markdown/mod.rs:449` per-line marker drop) +
  added `live_active_turn_marker_leaks_into_styled_frame`. Confirm it builds + passes here;
  also apply the dashboard companion (`app/session.rs:733` `preview_line` skip a `Turn N` line).

**Honest check:** a test that a `ScrollDown` mouse event with `mouse_capture=true` reaches
`app.scroll_lines(+3)` and moves `viewport.visual_top`; assert `mouse_capture` default is true;
`live_active_turn_marker_leaks_into_styled_frame` green. **Flag for user:** wheel scroll +
Shift+drag-copy must be confirmed in the real terminal.

---

## Slice 1 — Model identity on the wire (codex-pro / gpt-5.5)  [FOUNDATION, R2]

**Change (bridge — `frontends/tuiapp_v4/scripts/ga_bridge.py`; it is a bridge, NOT GA core):**
- Add `llm_identity(self) -> (name, model_real)` reading the ACTIVE member
  `b._sessions[b._cur_idx]` (`.name`=`codex-pro`, `.model`=`gpt-5.5`, strip a trailing `[...]`
  tag); scalar backend degrades to itself. Wrap in try/except → `("?","?")`.
- Emit additive fields in BOTH report sites: `Ready` (`:384`) and `Status` (`_status_payload`
  `:417`): `"llm": name, "model_real": real`; keep existing `"model"` for back-compat.

**Change (Rust):**
- `src/bridge/protocol.rs`: add `#[serde(default)] llm: Option<String>` + `model_real: Option<String>`
  to `Ready` and `Status` (additive — old bridges still parse).
- `src/app/reducer.rs`: store `app.llm_name` + `app.model_real` on Ready AND Status (so a
  mid-turn failover live-updates). Widen `FrameSink::on_ready` signature.
- `src/app/mod.rs`: add `pub llm_name: Option<String>`, `pub model_real: Option<String>`.

**Honest check:** feed `Ready{llm:"codex-pro", model_real:"gpt-5.5"}` then a failover
`Status{llm:"getoken_20x", model_real:"claude-opus-4-8[1m]"}` via `apply_bridge_event`; assert
`app.llm_name`/`app.model_real` update and the `[1m]` tag is stripped to `claude-opus-4-8`.

---

## Slice 2 — Multi-line rounded HEADER box  [R1 + R2]

**Change — `src/components/cockpit/header.rs` `render_header`:** replace the one-line cram
with a `BorderType::Rounded` block (lavender border) holding 5 interior rows:
`>_ GenericAgent` (only `>_` accent) / blank / `model:   <llm>·<model>   /llm switch` /
`directory:   <cwd>` / `session:   <name> · scrollback`. **Use the round-4 identity:**
`llm = app.llm_name (codex-pro)`, `model = app.model_real (gpt-5.5)` — NOT the full pipe-chain
(this overrides tui_v3's full-chain banner per the user). Bump header height to ~8 rows in
`cockpit/mod.rs` + the `main.rs` layout split.

**Honest check:** `--dump-frame normal` → frame contains `╭` and `╰`, a line starting `>_ GenericAgent`,
`codex-pro` and `gpt-5.5` on SEPARATE rows, `/llm switch`, `directory:`, `session:`.

---

## Slice 3 — Tool-call BORDERED BOX (tui_v3 style)  [R1, RISKIEST]

**Change — `src/render/chip.rs` + `src/markdown/mod.rs`:** replace the flat
`render_chip_bullet`/`ChipBullet` (the `○ web_scan {json}` look) with a `render_chip_box`
that emits box ROWS: top border `╭─ <name> <badge> ·tN ─…─╮` (name bold, badge
`✓ ok`/`✕ error`/`· …` colored, `·tN` dim, accent corners), interior `│ <arg-hint> │` then
`│ <result-preview ≤4> │` (dim), `│ … +N more │` fold affordance, bottom `╰─…─╯`. Reuse
`tool_status`, `parse_tool_calls`, `clip_cells`. Keep `NodeId::Tool` fold tagging so
click-to-expand still works. Update/delete the chip.rs "no box glyphs" tests (chip.rs:473-477)
to expect the box. Border = accent (`Token::Claude`).

**Honest check:** stream a real-GA-shaped call `🛠️ web_scan({"tabs_only":true})\n[Info] ok`
through `apply_bridge_event` MessageDelta; scan the **styled** frame for `╭─`, `web_scan`,
`·t`, the `[Info]`-derived result row INSIDE, and `╰`. The 4 parity invariants stay green
(box rows go through both styled-draw and plain-projection).

---

## Slice 4 — Spinner animates + footer effort/ctx  [R4 items 1,4]

**Change:**
- `src/components/cockpit/footer.rs` `render_spinner`: replace `let glyph = '⠿'` with
  `app.spinner_style.glyph(tick)` (cycles braille while busy). `render_done_line` keeps the
  static `⠿` for idle — they are mutually exclusive (`cockpit/mod.rs:106-109`). Flip
  `SpinnerStyle::default()` (flavor/mod.rs:26) to `Braille`.
- `render_session_info` effort: when `app.effort_label()` is `None`, show i18n `effort.none`
  (EN "non-thinking" / ZH "非思考模式"), not `—`.
- **ctx blank root cause:** the BACKGROUND `on_status` reducer (`reducer.rs:344-361`) ignores
  `context_percent` (`_context_percent`); the active one (`:216-265`) stores it. Make the
  background reducer store it too. Confirm `ga_bridge.py` Status carries `context_percent`.

**Honest check:** drive `cockpit_rows` at `now_ms=0` vs `300` → busy spinner glyph DIFFERS;
`render_session_info` row contains `非思考模式` (no effort) and `ctx 48%`; reducer unit test
that background `on_status` stores `context_percent`.

---

## Slice 5 — Spinner status line + hanging `⎿ Tip:` (CC layout)  [R4 item 5 + R5]

**Change:**
- `footer.rs render_spinner`: compose the status line as `<braille> <gerund> (elapsed · ↓ tokens · thinking <effort>)`
  (prefix the effort token with the i18n "thinking" word; "thinking max effort"); progressive
  width-gating per R5 (drop tokens-part on narrow). NO emoji pet here — pet lives in the tab title
  (per "spinner 就只用 ⠿ / emoji 只出现 pet").
- `cockpit/mod.rs split_cockpit`: when busy, render the Tip (`⎿ `) as the row DIRECTLY UNDER the
  spinner status line (above the composer) — add a `spinner_tip: Option<Rect>`, draw `render_tips`
  there; remove the detached below-composer Tip for the busy case (row1 session-info stays).

**Honest check:** busy frame → find the spinner glyph row; assert the very NEXT row starts `⎿ `
and the spinner row matches `… (… · ↓ … · thinking …)`. Invert `below_composer_has_two_rows`.

---

## Slice 6 — `/pets` rename + default Bear + tab title  [R4 items 2,3]

**Change:**
- `src/flavor/mod.rs`: move `#[default]` from `PetStyle::Off` to `PetStyle::Bear`.
- `src/commands/dispatch.rs`: delete the spinner rows in `emoji_picker_items` (`:567-577`) +
  the `(100..103)` arm in `apply_emoji_choice` (`:545-547`) — pets only, no spinner config.
- `src/commands/registry.rs`: rename `cmd("emoji",…)` → `cmd("pets","pet style",Ui)` + `alias(…, "emoji")`;
  update the resolve test name-list/count.
- `src/app/mod.rs terminal_title`: `<dynamic-pet-face> <session_name> · GenericAgent` (heat-aware
  face for `pet_style`, fall back to bear when Off; session name via
  `sessions.session(active).name`, fall back to just `GenericAgent`). Keep it coalesced; do NOT
  introduce "NativeClaude" anywhere.

**Honest check:** `PetStyle::default()==Bear`; `terminal_title()` contains the active session
name + `GenericAgent` + a bear face, and NOT `NativeClaude`; emoji/pets picker has zero spinner rows.

---

## Slice 7 — Per-command effects parity + DELETE `/effects`  [R4 item 6]

**Change:**
- `src/components/cockpit/composer.rs render_composer`: the FX engine exists
  (`FxCommand`, `fx_command()`, `draw_composer_border_fx`, `command_word_spans`) but the border
  overlay is gated on `effects.caps.enabled()` (truecolor) — so in a plain terminal `/goal` shows
  NOTHING, unlike `!`. Extend the base `border_tok`/`mark_tok` selection so when
  `fx_command(text).is_some() && !shell`, the base border token = the command accent
  (Goal/Morphling→Claude, Hive→Success, Conductor→Suggestion) — ALWAYS visible, like `!`. The
  animated truecolor overlay layers on top when enabled.
- DELETE the `/effects` command: `registry.rs:98` row, `dispatch.rs` `"effects"` arm (`:309-335`),
  the `cmd.effects` i18n keys. **Keep the effects ENGINE** (separator shimmer + border FX run
  automatically); pick a sane default `EffectMode` in `AppState::new`.

**Honest check:** `render_composer` with buffer `/goal` → base border token ≠ `Token::Border`
(visible even at NO_COLOR/mono); `/effects` no longer resolves in the registry.

---

## Slice 8 — `@` completeness vs performance  [R6]

**Change:**
- `src/input/file_expand.rs rank_files`: stop hard-truncating to 8 (`:100,128,133`); return the
  full ranked list (cap ~500) — truncation becomes a VIEW concern.
- `src/components/cockpit/dropdown.rs`: scrolling window via the existing `window_slice` + a
  `… +N more` hint row (`:103`).
- `src/input/paths.rs`: switch the walk from non-deterministic DFS (`queue.pop()` `:218`, sorts
  AFTER the cap) to deterministic BFS so the cap (`MAX_INDEXED_FILES`, raise ~5000→20000) drops
  only DEEP files; lengthen the 5s TTL (`mod.rs:44`) or background non-blocking re-walk.

**Honest check:** index a dir with >8 matches → the `@` picker shows a scrolling window + `+N more`;
ordering is deterministic across two index builds; the walk reaches a known-deep file under the cap.

---

## Slice 9 — `/continue` search parity with tuiapp_v2  [R6]

**Change — `src/components/continue_picker.rs`** (the searchable picker already exists end-to-end):
- Add the v2 debounced content-grep (`tuiapp_v2.py:1517-1592`): split a cheap meta-only filter
  (name + path basename, whitespace-AND terms) that runs immediately from the lazy first-~1MB
  content grep that runs on a tick after a ~0.2s pause (cancel the prior keystroke's pending grep).
- Add a relative-age prefix per row (port `_rel_time`).
- Optional: restore the `/continue N` non-interactive form. No `ga_bridge.py` change.

**Honest check:** typing a query filters meta immediately; the content grep fires only after the
debounce; rows show a relative-age prefix; restore still routes `Command{restore}` → `handle_restore`.

---

## Slice 10 — Markdown heading polish  [R3 caveat]

`code::heading_style` (`markdown/code.rs:53`) emits a literal `## ` glyph before headings; the user
reads visible `##` as "markdown not rendering". Make headings render clean like tui_v3/CC: drop the
raw `## ` prefix, keep BOLD + heading color (or use tui_v3's restrained marker). Identical live +
finalized (one renderer). **Honest check:** `--dump-frame` of a `## Heading` → the row is BOLD/colored
and contains `Heading` but NOT `##`.

---

## Monitor (final gate)

5 adversarial verifiers, each demanded to reason about the RUNNING binary, not a passing test:
scroll-via-wheel, Turn-N-in-styled-live-frame, header box + codex-pro/gpt-5.5, tool box, spinner
animate + Tip-under-spinner, effort/ctx, /pets+tab, command-FX-visible-at-mono, @ window, /continue
debounce. Each must cite rendered-output evidence. Close any gap, then update `checklist.md` + `query.md`.
