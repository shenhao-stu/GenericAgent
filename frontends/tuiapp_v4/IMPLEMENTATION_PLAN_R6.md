# tui_v4 — Round-6 Implementation Plan

> Verification rule (unchanged R4/R5): a box ticks ONLY on evidence from the
> **live/styled** path (text fed through `apply_bridge_event`, styled `TestBackend`
> cells scanned, real key/mouse events) — never a clean-fixture unit test alone.
> Safety (from the R5 freeze): every cargo gate is `timeout`-wrapped; only ONE cargo
> runs at a time; impl on **Opus**; every hand-written `while`/`loop` must prove it
> advances/breaks at EOF.

Baseline: round-5 committed (`2c0b4d1`, v0.5.0, builds green). Upstream `lsdefine/main`
merged (brings `continue_cmd.py` preview fix to port). Specs: `recon/round6/R1_audit.md`
(parity), `recon/round6/R2_reference.md` (codex/cc rendering).

---

## Slices (each: edit → `timeout cargo build` → live honest-check → final full test → Monitor)

### S1 — Content + composer HUG THE TOP (task 1c). RISKIEST (geometry).
**Problem:** round-5 bottom-anchored the transcript → content sits at the BOTTOM with a
huge blank gap ABOVE (clip_20260603_020949). User wants model output at the TOP.
Neither pure top- nor bottom-align fixes the gap — it just moves. Fix = let content +
spinner + composer + footer flow from the top; blank absorbed at the very bottom.
**Files:** `components/cockpit/mod.rs` (`split_cockpit`), `components/cockpit/transcript.rs`
(`render_transcript`), `app/mod.rs` (`prepare_frame`).
- In `split_cockpit`: when `app.wrap_cache.total_visual_lines() <= available_transcript_h`,
  set the transcript constraint to `Length(total_visual_lines)` and append a trailing
  `Constraint::Min(0)` SPACER (absorbs blank at the bottom). When content overflows, keep
  `Constraint::Min(0)` transcript + NO spacer (composer pinned at bottom, scrolls as today).
- CRITICAL (geometry contract): `split_cockpit` is called in BOTH `prepare_frame` (to size
  the viewport) and `render_cockpit` (to draw). Both must read the SAME
  `total_visual_lines()`. Restructure `prepare_frame` so the wrap cache is synced at
  `area.width` (transcript is a full-width vertical split → width == area.width) BEFORE the
  split that decides the hug-top height; then resize the viewport to the final transcript
  height. Add the trailing spacer field to `CockpitLayout` or absorb via an extra chunk.
- `render_transcript`: REMOVE the round-5 bottom-anchor sub-rect (lines 107–127) — render
  `Paragraph::new(lines)` top-aligned into the given `area`.
- All 4 parity invariants stay green (`styled_wrap_rowcount_matches_wrap_cache`,
  `plain_projection_matches_rendered_text`, `cockpit_render_rowcount_matches_plain_projection`,
  `embedded_newline_in_span_keeps_rowcount_parity`).
- **Test rewrite** `blank_gap_bottom_anchor_live_path` → `content_hugs_top_blank_at_bottom`:
  content starts at transcript row 0 (NOT blank); spinner immediately follows last content
  row (gap 0); blank rows are at the screen BOTTOM (below the tips row). LIVE styled path.

### S2 — Remove the ugly duplicate `↳` breadcrumb (task 1b), keep fold.
**Problem:** `markdown/mod.rs:191` emits ` ▾ <title>` for an expanded turn AND
`render_turn_body` (line 343) re-emits `↳ <summary>` — the SAME text twice
(clip_20260603_020949). R2: "One caret in the header, one `└` block of genuine detail."
**Files:** `markdown/mod.rs` (`render_turn_body`).
- Keep the `hoist_summary` call (still STRIP `<summary>` tags from the flowing body so they
  never render raw), but DELETE the `↳ <crumb>` line push (lines 340–353). The ` ▾ `/` ▸ `
  header is the canonical summary display.
- Update tests `mod.rs:1369` (`assert plain.contains('↳')`) + `mod.rs:1429`
  (`↳ 准备执行echo hi`) → assert the ` ▾ ` header carries the summary, NO `↳` anywhere.
- Caret convention (R2): `▸` collapsed, `▾` expanded, ALWAYS in the header line.

### S3 — `/goal /hive /conductor` command-char FX like `/morphling` (task 3).
**Problem:** `command_word_spans` (composer.rs:228) already wires all 4, but only Morphling
is visible (full `flow_color` rainbow per char); Goal/Conductor use a near-invisible
`intensity_at` sheen, Hive a faint mint blend. Plus the cursor-in-token guard
(composer.rs:128 `cur_col <= tok_w → None`) hides the effect while typing the bare command.
**Files:** `components/cockpit/composer.rs` (`command_word_spans`, the cursor guard).
- Make each of Goal/Hive/Conductor a STRONG, col-based per-char effect (visible statically,
  like Morphling), each distinct: e.g. Goal = a 2-stop gradient sweep across its accent
  (Claude→lightened) with high amplitude; Hive = alternating mint swarm shades (strong
  contrast); Conductor = a clear directional baton (Suggestion→lightened). Keep Morphling's
  rainbow. All BOLD, all explicit RGB per char.
- Relax the cursor-in-token guard so the command word STYLES while being typed: style every
  char of the `/word` token, and REVERSE just the single cursor-cell grapheme within it (so
  the effect is visible immediately on `/goal`, not only after a trailing space).
- Update `command_word_spans_goal_returns_styled_spans` to assert Goal's first/last char
  colors differ (a real gradient, not a flat run), same as the Morphling assertion.

### S4 — Footer: labeled pipe fields + ctx progress bar (task 4).
**Target:** `Channel: codex-pro | Model: gpt-5.5 | Effort: none | ctx: [░░░░…] Nk/Mk (P%) | branch: feat/tui-v4 | mouse: select`
**Files:** `components/cockpit/footer.rs` (`render_session_info`), `bridge/protocol.rs`
(Status frame — additive fields), `app/mod.rs` (fields + reducer), `scripts/ga_bridge.py`
(`_status_payload`), `i18n/mod.rs`.
- ga_bridge `_status_payload` already computes `cap = context_window_chars(be)` +
  `used = current_input_chars(be)`. Send them as ADDITIVE fields `context_used` +
  `context_limit` (chars; serde-default None — like `llm`/`model_real` were added). NEVER
  break the wire: both optional, old frames still parse.
- protocol `Status` + `app` gain `context_used: Option<u64>` + `context_limit: Option<u64>`;
  the Status reducer fills them.
- `render_session_info`: switch to LABELED fields with ` | ` separators (was ` · `):
  `Channel:`, `Model:`, `Effort:`, `ctx:`, `branch:`, then `mouse:` + conn chip on the tail.
  The ctx field renders a `[░░░░…]` bar (width ~16, filled by `context_percent`) + a
  `{used_k}/{limit_k} ({P}%)` readout (k = chars/1000, rounded; honest — the real GA trim
  metric). Fall back to `[…] —` when context data is absent. Width-gate gracefully on narrow
  terminals (drop the bar first, keep the labels).
- i18n: add `footer.channel`/`footer.model`/`footer.effort`/`footer.branch` labels (EN+ZH).
- **Test:** styled frame at 120-wide shows `Channel:`/`Model:`/`ctx:` + a `[` bar `]` +
  `(P%)`; the pipe `|` separators present; `context_win*3` truth preserved.

### S5 — Paste IMAGES + FILES into the composer (task 2).
**Problem:** the `[Image #N]`/`[File #N]` infra is half-built (`collect_images` is
`#[allow(dead_code)]` "Phase 3"; every `Submit{images: None}`). `read_clipboard()` only does
`get_text()`, so a copied screenshot / file produces nothing. ga_bridge `handle_submit`
already forwards `images` → `put_task(images=...)`; `_coerce_images` wants `{data: <base64>}`
dicts; `SubmitImage{data, mime}` matches.
**Files:** `clipboard.rs` (clipboard image read), `input/keymap.rs` (Ctrl+V), `input/mod.rs`
(submit → collect_images), `commands/dispatch.rs` (Submit images wiring), `input/paste.rs`
(collect_images → base64 payloads), `Cargo.toml` (arboard `image-data` feature + a PNG
encoder), `bridge/mod.rs` (Submit images), `i18n/mod.rs`.
- `Cargo.toml`: `arboard = { version = "3.6.1", features = ["image-data"] }` (keeps existing
  cfg); add a minimal PNG encoder (`png = "0.17"`) to turn arboard's raw RGBA `ImageData`
  into PNG bytes.
- `clipboard.rs`: add `read_clipboard_image() -> Option<(Vec<u8> /*png*/, w, h)>` via
  `arboard::Clipboard::get_image()` → encode RGBA→PNG.
- Ctrl+V (keymap.rs): try text first (existing). If no text but an image is present, encode →
  write a temp PNG under `repo_root/temp/tui_v4_paste_<pid>_<n>.png` AND fold to `[Image #N]`
  with the temp path (so the existing `is_image_path`/`fold_image` path + on-submit
  `collect_images` reuse works). If clipboard text IS a file path (existing `looks_like_path`),
  it already folds to `[File #N]`.
- On submit: wire `paste.collect_images(text)` → read each image file → base64 →
  `Vec<SubmitImage{data, mime}>` → `Submit.images` (replace the `None` at dispatch.rs:34 /
  input/mod.rs submit). For a `[File #N]` path, the existing `expand` inlines the path so the
  model/`@` can read it.
- i18n notice: "已粘贴图片 / Image pasted (#N)".
- **Test (PURE):** `collect_images` returns the right paths; a fake clipboard image →
  fold_image placeholder; submit builds `Submit.images` with base64. (Clipboard read itself
  is effectful → behind a thin fn, not unit-tested; the fold/collect/encode pieces are.)

### S6 — Port the `/continue` preview fix to Rust (task 5).
**Problem:** `continue_picker.rs:369 preview_from_windows` + `last_summary` is the OLD
`_preview_from_file` port — same two bugs upstream `6f71212` fixed: (1) an unclosed
`<summary>` makes the match swallow cross-segment `=== ` headers / JSON; (2) the fallback
returns the first non-`===` head line = JSON debris.
**Files:** `components/continue_picker.rs`.
- `preview_from_windows`: use only the LATEST `<summary>`, and REJECT it if dirty (contains
  `=== ` or `"role"` or > 200 chars) — fall through instead of digging older ones.
- Replace the "first non-`===` head line" fallback with a `last_user(text)` that scans
  `=== Prompt ===` blocks newest-first and returns the first that survives a `user_text`
  filter (skips tool_result continuations + inject markers like `[WORKING MEMORY]`,
  `### `, `<history>`). Cap to 120 chars.
- Mirror `continue_cmd.py` `_last_user` + the dirty-summary guard exactly.
- **Test:** a malformed log (unclosed `<summary>` swallowing `=== Response ===` + JSON) →
  preview is the last real user prompt, NOT JSON debris; a clean summary still wins.

### S7 — Restore `/verbose` interactive tool inspector (task 1a, R1 GAP B).
**Problem (R1 §3.2):** v4's `/verbose` (`/tools` `/trace`) opens `Overlay::Verbose` over
`app.tool_audit: Vec<String>` (`app/mod.rs:165`) — each entry only `format!("{badge}
{name}{args}")` (`reducer.rs:209`); result/raw never stored. The overlay is read-only
(`input/views.rs:465-470` closes on any key). v3 (`tui_v3.py:4684-4747`) is a full inspector
backed by `ToolRecord{id,name,args,result,status,raw}`: list+detail panes, ↑/↓ (k/j) select,
PgUp/PgDn scroll detail, Enter cycles result/args/raw, `c` copy, `e` export. v4 lost args
view, raw view, selection, scroll, copy, export. Red line: "功能必须全有".
**Files:** `app/mod.rs` (data model: `tool_audit: Vec<ToolRecord>`), `app/reducer.rs:209`
(populate name/args/result/status — parse from the assistant block's `🛠️` chips via
`render::chip::parse_tool_calls`, which already yields name/args/result/status — read
`render/chip.rs` to confirm the fields), `components/overlay/info.rs:215-241` (two-pane
list+detail render with a selection marker + per-tool status color), `input/views.rs:465-470`
(interactive keys: ↑/↓·k/j select, PgUp/PgDn scroll detail, Enter cycle field
result→args→raw, `c` copy current field via `clipboard::copy_text`, `e` export to a temp
file). Keep a small `VerboseState{ selected, field, scroll }` (in the overlay enum or app).
- "raw": if GA exposes no distinct raw payload, use the full untruncated result text as the
  raw field (args=the parsed args, result=the truncated preview, raw=full result). Document
  this. Do NOT invent data.
- **Test:** populate `tool_audit` with 2 ToolRecords; open Verbose; assert the styled overlay
  shows BOTH a list (2 names + selection marker) and a detail pane; ↓ moves selection; Enter
  cycles the visible field label (result→args→raw); LIVE styled path.

---

## Gate / Monitor / Release
- Per-slice: `timeout 220 cargo build -q` (compile gate, Opus agent self-checks).
- Final (inline, serial): `timeout 500 cargo test` (full suite, 0 failed / 0 warnings) →
  `cargo build --release` → `--version` smoke → kill any stray cargo/tui processes.
- Monitor (workflow, parallel Opus, render-based over dumps I produce inline — agents run NO
  cargo): M1 hug-top layout · M2 `↳` gone + model-output · M3 command FX 4-distinct ·
  M4 footer pipe+ctx-bar · M5 paste img/file end-to-end + continue-preview · all with
  rendered-frame / wire evidence.
- Release (AUTHORIZED this round): bump `0.5.0`→`0.6.0` (Cargo.toml + main.rs), commit
  (stage ONLY `frontends/tuiapp_v4/`), `git push origin feat/tui-v4`, tag `tui-v4-v0.6.0`
  (push tag → CI release). The upstream merge commit pushes too.
