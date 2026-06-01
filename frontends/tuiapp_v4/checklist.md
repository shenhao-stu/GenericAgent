# tui_v4 — Round-4 checklist (LIVE tracker)

> Round 3's all-green claim was a false positive: it verified the finalized/plain path
> (`--dump-frame` + clean-fixture unit tests), never the live/styled path the user sees.
> **This round every box ticks only on evidence from the live path** — text fed through
> `apply_bridge_event`, the **styled** `TestBackend` cell grid scanned, or real key/mouse
> events driven. A passing unit test on a hand-built fixture is NOT acceptance evidence.
> Plan: `IMPLEMENTATION_PLAN_R4.md`. Specs: `recon/round4/R1..R6`.
> Gate per slice = `cargo build` + `cargo test` (0 warn) + the honest check + `VERDICT: PASS`.
> Baseline (R3 Turn-N source fix already in): **335 passed, 0 failed, 0 warnings**.

---

## Slices (DO IN ORDER; halt-on-break)

- [x] **S0 — wheel scroll + Turn-N confirm.** Capture default ON; `mouse_capture_defaults_on`; Shift+drag tip; `wheel_scroll_event_moves_viewport_under_default_capture`; `dashboard_preview_skips_turn_marker_line`; `live_active_turn_marker_leaks_into_styled_frame` green. **+ user real-terminal confirm still pending.**
- [x] **S1 — model identity on the wire.** `ga_bridge.py llm_identity()` emits `llm`/`model_real` in Ready+Status; protocol/app/reducer parse+store; `ready_then_failover_status_updates_llm_identity` (failover live-updates + `[1m]` stripped). **VERIFIED in `--dump-frame normal`: header shows `codex-pro · gpt-5.5`.**
- [x] **S2 — multi-line rounded HEADER box.** `render_header` → `BorderType::Rounded` box, `>_ GenericAgent` + model/directory/session rows + `/llm switch`; `HEADER_ROWS=8`. **VERIFIED rendered:** box `╭…╮`/`╰…╯`, `>_ GenericAgent`, `model: codex-pro · gpt-5.5 /llm switch`, `directory:`, `session:` each own row. *Nit:* interior lacks leading-space pad (`│>_` vs `│ >_`) → polish pass.
- [x] **S3 — tool-call BORDERED BOX (tui_v3).** `render_chip_box`+`push_tool_box`. **VERIFIED rendered:** `╭─ web_scan  ✕ error  ·t1 ──╮` / `│ {"tabs_only": true} │` / `│ 3 tabs scanned · ok │` / `│ !!!Error: SSE overloaded │` / `╰──╯`; `live_tool_call_renders_bordered_box_in_styled_frame`; 4 parity invariants green; click-expand kept.
- [x] **S4 — spinner animates + effort/ctx.** `render_spinner` glyph = `spinner_style.glyph(tick)` (Braille default); done-line keeps static `⠿`; effort→`非思考模式`/`non-thinking`; ctx via background `on_status` + per-session `context_percent` round-trip; footer llm/model → codex-pro/gpt-5.5. **VERIFIED rendered:** `⠙ Pondering… (…)`; footer `codex-pro · gpt-5.5 · non-thinking · ctx 48% · —`.
- [x] **S5 — spinner status line + hanging `⎿ Tip:` (CC).** status = `<braille> <gerund>… (elapsed · ↓tokens · thinking <effort>)`, pet removed, width-gated. **VERIFIED rendered:** `⠙ Pondering… (0.1s · ↓1.6k · non-thinking)` then `⎿ Tip: …` directly under, above composer. *Nit:* spinner could read `↓ 1.6k tokens` per user example → polish.
- [x] **S6 — `/pets` + default Bear + tab title.** `PetStyle::default()==Bear`; picker drops spinner rows; `cmd("pets")` + `alias("emoji"→"pets")`; `terminal_title` = dynamic-pet + session_name + GenericAgent, no NativeClaude. *Evidence:* registry + title tests.
- [x] **S7 — per-command FX parity + delete `/effects`.** `render_composer` base border = command accent always (not truecolor-gated), like `!`; `live_command_border_restyles_at_mono_like_shell_bang`. **VERIFIED:** `resolve("effects").is_none()`, not in COMMANDS/palette; engine kept.
- [x] **S8 — `@` completeness vs perf.** `rank_files` returns full ranked list (cap ~500, no 8-cap); dropdown scroll window (`window_slice`) + `… +N more`; deterministic FIFO BFS walk (`VecDeque::pop_front`, per-dir sort), `MAX_INDEXED_FILES` 5000→20000, TTL 5s→30s; `walk_with_cap` testable. *Evidence:* determinism + shallow-vs-deep + scroll-window tests.
- [x] **S9 — `/continue` search parity (v2).** immediate meta filter split from a debounced (~0.2s, `spinner_tick`) lazy content grep run off the keystroke path via `tick()`; `rel_age` prefix per row; `searching…` hint; `/continue N` form. *Evidence:* two-stage + restore-routing live test; no `ga_bridge.py` change.
- [x] **S10 — markdown heading clean.** `heading_style` returns Token only (no glyph); `Tag::Heading` drops the prefix span, keeps bold + per-level color. **VERIFIED:** `--dump-frame normal/busy` hash-count = 0; `headings_render_bold_and_colored_without_hashes_in_styled_frame`.
- [x] **S11 — visual polish nits.** Header interior left-inset 1 col (`│ >_ GenericAgent`); spinner token readout `↓ <count> tokens` (i18n `tokens.unit`), width-gating intact. **VERIFIED rendered:** `│ >_ GenericAgent`, `⠙ Pondering… (0.1s · ↓ 1.6k tokens · non-thinking)`, effort-high → `thinking with medium effort`.

---

## Monitor (final adversarial gate)
- [x] 4 verifiers (render-based, Sonnet — Opus weekly-limit fallback), citing rendered-frame evidence. **M2/M3/M4: ALL CONFIRMED, zero gaps** (spinner animate+settle, status+hanging tip, footer identity+ctx, tab title, Turn-N absent across 3 shapes × 10 dump scenarios, scroll wiring, 4 parity invariants, budgets, /pets, /effects gone, FX-at-mono, @ window, /continue debounce). **M1: 2 gaps found + CLOSED** — narration `▸` now accent + leading space (` ▸ title`), `cockpit_folds_completed_turn_to_one_line` updated. Final: **354 passed / 0 failed / 0 warnings**.

## Iteration log
- **2026-06-01 R4 kickoff** — Recon workflow (R1-R5) + focused agent (R6) wrote 6 specs (1712 lines); the lead found the Turn-N false-positive (expanded bodies keep the marker; tests only checked folded/plain) and the wheel-scroll blind spot (capture-off kills wheel). R3 agent applied the Turn-N source fix + 3 styled-frame E2E tests. Baseline re-confirmed **335 passed / 0 failed / 0 warnings**. `IMPLEMENTATION_PLAN_R4.md` written (11 slices). Implementation workflow dispatched.
