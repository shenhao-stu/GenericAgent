# tui_v4 — Claude-Code-grade redesign (real-GA-format edition)

> Goal: a TUI that **surpasses Claude Code**, adapted to GenericAgent. The transcript currently shows RAW GA markers (ugly) because the renderer was tested against synthetic data, not real GA output. This spec pins the **exact GA output format** + **CC's actual visual design** so the redesign renders real sessions beautifully.

---

## 1. GA core output format — the PARSE+RENDER contract (load-bearing, verified)

The assistant stream (`MessageDelta.text`) contains these literal markers. The renderer MUST parse + transform them — never show them raw.

| Marker (literal) | Source | RENDER AS |
|---|---|---|
| `<summary>…</summary>` | ga.py:552 (LLM per-turn ≤80-char summary) | **Hide the tags.** Show inner text as a subtle dim breadcrumb above the turn (or fold it). Never show `<summary>`/`</summary>`. |
| `Turn N ...` (line) | agent_loop.py:62 (compact; tui gets this) | **Do NOT render as text.** Use it only as a turn boundary → vertical spacing (CC has no "Turn N" clutter). |
| `🛠️ NAME(ARGS)` then `\n\n\n` | agent_loop.py:89 **COMPACT** (tui_v4's mode) | A **tool-call chip/bullet** (see §2.3). NOTE: the marker is `🛠️ name(args)`, **NOT** `🛠️ Tool:` — fix the parser. Result = everything after, up to the next `🛠️`/`Turn N`/`<summary>`/EOT. |
| `[Action] …` / `[Status] …` / `[Info] …` | ga.py (tool output prefixes) | The tool **result body** (indented under the chip). Keep the prefixes but style dim. |
| `!!!Error: …` (in stream) | llmcore.py:178/378 (real model error) | A distinct **dim/red error line** in the transcript (a real error the model hit). Compact, not a wall. |
| `[MixinSession] …, retry N/M (s→s)` | llmcore.py:988 → **STDERR** → tui `BridgeEvent::Stderr` → `[bridge] …` notice | **HIDE from the transcript.** These are failover retry diagnostics on stderr. Drop them (or route to a debug-only log + a tiny "retrying…" status), never as `[bridge]` rows. |

**Net:** one assistant turn = `Turn N ...` (→ spacing) + `<summary>…</summary>` (→ breadcrumb) + prose (markdown) + `🛠️ name(args)` (→ chip) + `[Action]/[Status]/[Info]` result (→ indented) … repeat.

**Update the `--dump-frame` seed** to use THESE real markers (a turn line, a `<summary>`, a `🛠️ web_scan({"tabs_only": true})` + `[Info] …` result, a `!!!Error:` line, and a simulated `[MixinSession] … retry` stderr) so the dump verifies they render clean.

---

## 2. Claude Code visual design (exact, from temp/claude-code) — adapt to GA

### 2.0 Color palette (theme tokens — RGB; map into theme/tokens.rs)
- `claude` **rgb(215,119,87)** (brand orange, spinner/accent) · `claudeShimmer` rgb(235,159,127)
- `success` rgb(78,186,101) · `error` rgb(255,107,128) · `warning` rgb(255,193,7)
- `suggestion` rgb(177,185,249) (selection/focus) · `subtle` rgb(80,80,80) (dim) · `text` rgb(255,255,255) · `inverseText` rgb(0,0,0)
- `ide` rgb(71,130,200) (links) · `bashBorder` rgb(253,93,177) (shell `!`) · **`userMessageBackground` rgb(58,58,58)** (user input band — user-specified)
- planMode rgb(72,150,140) · autoAccept rgb(175,135,255)

### 2.1 USER message — full-width inverse band (the headline)
- A **full-terminal-width band** with bg **rgb(58,58,58)**, white text, `marginTop:1`. The user's prompt text sits in the band (wrapped). Right pad 1. (CC: `UserPromptMessage.tsx:76` `backgroundColor=userMessageBackground`.) This replaces the bare `> hello`.

### 2.2 ASSISTANT message
- No leading glyph; markdown-rendered prose. Blank line between turns (no rule, no "Turn N").

### 2.3 TOOL call — CC bullet style (replace the ugly box/raw text)
- `⏺` (done, `BLACK_CIRCLE`) / `○` (running) bullet + **tool name** (e.g. `⏺ web_scan`), colored by status (success/dim/error). Args as a dim one-liner after the name; **result indented 2 cols** below, dim, truncated to a few lines with `… +N more`. No heavy box. (CC: `AssistantToolUseMessage.tsx`.)

### 2.4 Turn separation
- Implicit: a blank line + the user band. **No `Turn N` text, no horizontal rules.**

### 2.5 Bottom area (declutter — current footer is too noisy)
- Composer box (rounded border; pink in shell). Below it ONE status row: `❯ <mode>` (left) · right-aligned `<model> · ctx <pct> · $<cost> · <git>`.
- **TRUNCATE the model name**: MixinSession shows a long `a|b|c|…|kiro`. Show only the **primary segment** (e.g. `codex-pro`) or `MixinSession·codex-pro` — cap ~22 cells. Never the full pipe-list.
- Hint row (very bottom, dim): `⏎ send · ⇧⏎ newline · ⌃S sessions · / cmds · ! shell · @ file · ⌃C quit` + right-aligned dim `Tip: …` (NO emoji).
- Keep it tight: status + hint = 2 rows, clean, CC-spaced.

### 2.6 Spinner (busy)
- Custom pulse `·✢✳✶✻✽` (or our arc/braille) in `claude` orange + a gerund + `· <elapsed> · <tokens>`. NOT CC's `✻` alone, NOT emoji pet by default (kaomoji pet OK as an opt-in /emoji style).

---

## 3. `/effort` command (NEW) — adapt to native_claude + native_oai, hot-reload

- Backend contract (llmcore.py:540-559): set `reasoning_effort` ∈ {none,minimal,low,medium,high,xhigh} on `agent.llmclient.backend` — read by BOTH NativeOAISession (payload.reasoning_effort / reasoning.effort) AND NativeClaudeSession (output_config.effort, where xhigh→max). Claude also has `thinking_type`/`thinking_budget_tokens`.
- **Hot-reload path:** GA core intercepts `/session.<k>=<v>` and does `setattr(self.llmclient.backend, k, v)` live (agentmain.py:122). So `/effort <level>` in tui_v4 → forward `/session.reasoning_effort=<level>` to the bridge (the existing Command/slash-forward path). Takes effect next turn. No restart.
- **UI (slider, mirror CC's clip_20260531_040300):** a `Faster ←——▲——→ Smarter` horizontal slider, stops `low  medium  high  xhigh  max` (map max→xhigh for the backend) with a `▲` marker on the current level; current level read from `app` (track it). Footer `←/→ to adjust · Enter to confirm · Esc to cancel`. On Enter → send `/session.reasoning_effort=<level>` + show a confirm line + update the status/spinner "thinking <level>" suffix.
- Show the active effort in the spinner ("thinking · high") and/or status.

---

## 4. Session view via mouse
- **Left-click** on the cockpit (header area at minimum; broaden to the transcript/sidebar zone) → open the session **Dashboard** (View::Dashboard) showing the multiple sessions (already mostly wired at main.rs:401). 
- **Right-click** anywhere → go BACK (close dashboard → cockpit). ADD the `MouseButton::Right` handler (currently missing) in main.rs for both views (right-click in Dashboard → close_dashboard; right-click in Cockpit → no-op or also toggle).
- Keep Ctrl+S toggle + Esc back.

---

## 5. Work items (the build)
- [x] **Transcript renderer rewrite** to the §1 contract: hide `<summary>` tags (→ dim breadcrumb), drop `Turn N` text (→ spacing), **fix chip parser to `🛠️ name(args)` compact** + parse result to next marker, render tool calls as §2.3 `⏺` bullets, keep `[Action]/[Status]/[Info]` dim-indented, render `!!!Error:` as a compact dim/red line, **suppress `[bridge]` stderr/retry notices**. ✅ `markdown/mod.rs::render_turn_body` (strip_leading_turn_line / hoist_summary → `↳` breadcrumb / `!!!Error`→`Token::Error`); chip parser `render/chip.rs` uses compact `🛠️ name(args)` marker (`compact_chip_parse` test); stderr→debug-ring only (`bridge_stderr_suppressed`). Verified clean in all `--dump-frame` outputs (no raw `🛠️`/`<summary>`/`Turn N`/`[bridge]`).
- [x] **User-input full-width band** bg rgb(58,58,58) (§2.1). ✅ `components/mod.rs::user_band_line` + `user_row_has_band_bg` test (every cell on the user row carries bg rgb(58,58,58) edge-to-edge).
- [x] **CC color palette** into theme tokens (§2.0). ✅ `theme/tokens.rs`: Claude rgb(215,119,87), UserBand rgb(58,58,58), ShellAccent (bashBorder) rgb(253,93,177), success/error/warning/suggestion/subtle/ide/planMode/autoAccept all present.
- [x] **Bottom declutter** + **model-name truncation** (§2.5). ✅ footer = 1 status row + 1 hint row; `truncate_model` shows only the primary segment `MixinSession·codex-pro` (cap 22), never the pipe-list (`truncate_model_primary_segment` test, verified against the real live `MixinSession/codex-pro|getoken_20x|…` model).
- [x] **`/effort` slider command** (§3). ✅ `app/effort.rs` (5 stops, `max→xhigh`, `effort_levels_and_mapping`+`effort_slider_nav` tests) + `components/overlay.rs` slider paint + `main.rs::apply_effort` forwards `Command{name:"session.reasoning_effort=<v>"}` (`effort_forwards_session_command` test); `--dump-frame effort` shows `Faster ←—▲—→ Smarter`, `●medium` applied stop, `thinking · medium` spinner suffix.
- [x] **Right-click → back** + left-click → dashboard (§4). ✅ `main.rs::handle_mouse_event` — `Down(Right)` in Dashboard → `close_dashboard()`; `Down(Left)` in Cockpit header/footer → dashboard; `Down(Left)` in Dashboard row → switch. `mouse_*` tests exercise the Right arm.
- [x] **`--dump-frame` seed → real GA markers** (§1) so it verifies the new rendering. ✅ `main.rs::run_dump_frame` seeds `Turn 1 ...`, `<summary>…</summary>`, compact `🛠️ web_scan({"tabs_only": true})` + `[Info]` result, inline `!!!Error:`, and a simulated `[MixinSession] …retry` stderr; scenarios `normal|shell|busy|effort`.
- [x] Keep ≥262 tests green; add tests for: summary-tag hiding, turn-marker suppression, compact-chip parse, bridge-stderr suppression, model-name truncation, effort-level mapping (max→xhigh). ✅ **278 tests green** (0 failed; 2 ignored real-GA), all listed tests present: `summary_tags_hidden`, `turn_marker_not_rendered`, `compact_chip_parse`, `bridge_stderr_suppressed`, `truncate_model_primary_segment`, `effort_levels_and_mapping`.

## 6. Acceptance (verify via `--dump-frame` reading + real connect) — ✅ ALL VERIFIED 2026-05-31
- [x] Dump (seeded with real GA markers) shows: NO raw `<summary>`/`Turn N`/`🛠️ name(`/`[bridge]` text; user prompt in a full-width band; tool calls as `⏺ name` + indented result; `!!!Error:` compact. ✅ Read all of `--dump-frame normal|shell|busy|effort`; a strict codepoint scan of the combined output confirms `🛠️`/`<summary>`/`</summary>`/`Turn 1-3`/`[MixinSession]`/`[bridge]` are ALL absent from rendered text; user row carries bg rgb(58,58,58) edge-to-edge; `⏺ web_scan` + `  [Info] …` indented; `!!!Error:` is one compact red line.
- [x] `/effort` slider renders + maps levels (max→xhigh) + forwards `/session.reasoning_effort=`. ✅ `--dump-frame effort`: `Faster ←—▲—→ Smarter`, stops `low medium high xhigh max`, `●` on the applied stop, `thinking · medium` suffix; `effort_forwards_session_command` pins the forwarded `Command{name:"session.reasoning_effort=xhigh"}` for `max`.
- [x] Right-click returns from dashboard; left-click opens it. ✅ `handle_mouse_event` arms + mouse tests.
- [x] No emoji in chrome (symbols ok). Build + tests green; real GA connect = Ready. ✅ strict pictographic-emoji/VS16/ZWJ scan of all dumps = NONE (only monochrome symbols `◆ ⏺ ❯ ↳ ▲ ● ✦ ✧ ▰▱`); the kaomoji spinner pet is now OFF by default per §2.6 (Bear/Cat/Dot/Unicode/Fox still selectable via `/emoji`). `cargo build --release` green (4.92 MB exe); **278 tests pass (0 fail, 2 ignored real-GA)**; real GA core handshakes a `Ready` frame through the release-dir `ga_bridge.py` (live model `MixinSession/codex-pro|getoken_20x|…` → footer shows `MixinSession·codex-pro`).
