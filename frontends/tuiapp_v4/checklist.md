# tui_v4 — Round-3 polish checklist (v0.3, LIVE tracker)

> Supersedes the stale all-green v0.2 contract (preserved in git history; architecture/module-map now live in `IMPLEMENTATION_PLAN.md` + `recon/round3/ARCH_refactor.md`).
> Every box starts UNCHECKED — this round is NOT done. A box ticks only when its **acceptance evidence** is produced (dump-frame substring / test name / codepoint scan), never on self-report.
> Source: `recon/round3/*` → `IMPLEMENTATION_PLAN.md`. Closes `query.md` Q1..Q12 + the ARCH god-file split.
> Gate = `cargo build` + `cargo test` + `cargo run -- --dump-frame <scenario>`. Baseline (pre-round-3): **281 passed, 0 failed, 1 ignored**; 6 dumps in `/tmp/tui_v4_baseline/`.

---

## Slice 0 — ARCH refactor (split god-files; behavior identical) — DO FIRST

- [x] **0a `AppEvent` bus** — `src/app_event.rs` (37 LoC); `AppState.actions` + `emit`/`drain_actions`; loop drains. ✅ 281 tests, 6 dumps byte-identical, 0 warnings.
- [x] **0b transport params dropped** — 9 dispatch/apply fns emit intents; `tx_bridge` gone. ✅ `grep tx_bridge src/input src/commands` = clean; build green; dumps identical.
- [x] **0c main.rs split** — `input/{keymap,views,mouse}.rs`, `commands/dispatch.rs`, `term.rs`, `clipboard.rs`; quit-chord 4→1 (`is_quit_chord`); `dashboard_new_session` uses `split_command`. ✅ `main.rs` = **516** LoC; 281 tests; dumps identical.
- [x] **0d components/mod.rs split** — `cockpit/{mod,header,transcript,composer,footer,dropdown}.rs` + `text.rs`. ✅ each cockpit file ≤ 255; dumps identical.
- [x] **0e render is PURE** — `prepare_frame` hoists `set_term_size`+`sync_transcript`; `cockpit::render` takes `&AppState`. ✅ parity tests green; 6 dumps identical.
- [x] **0f app/markdown/overlay split** — `app/{reducer,overlay_ops,types,multi,tests}.rs` (shared `FrameSink` kills dup reducer); `markdown/{inline_math,table,code}.rs`; `components/overlay/{mod,effort,picker,info,effects}.rs`. ✅ full test green; 0 warnings.
- [x] **ARCH DONE gate** — all god-files < target; `tx_bridge` severed from handlers; full `cargo test` **281** green; 6 baseline dumps byte-identical. **Independently verified 2026-06-01.**

### Refactor targets (verified actuals)
- [x] `main.rs` 2505 → **516** (entry+loop+harness only).
- [x] `app/mod.rs` 2078 → **621**.
- [x] `components/mod.rs` 1123 → `cockpit/` dir, each ≤ **255** (<500).
- [x] `markdown/render.rs` 1070 → **595** (<600).
- [x] `components/overlay.rs` 720 → `components/overlay/` dir, `mod.rs` = **143** (<200).

---

## Query items Q1..Q12

- [x] **Q1 — history input = full-row `rgb(58,58,58)` band WITH `❯ ` prefix** (Slice 3). *Evidence:* `user_band_line_spans_width_with_band_bg` asserts `starts_with("❯ hello")`; `user_row_has_band_bg`; old `!starts_with('❯')` assertion DELETED; `--dump-frame normal` user row begins `❯ `.
- [x] **Q2 — native terminal selection-copy; no `ctrl+shift+c`-as-all-copy; no-newline table/multiline copy** (Slice 5). *Evidence:* `! grep 'EnableMouseCapture' src/term.rs`; `mouse_capture_defaults_off`; `! grep "Char('c' | 'C') if ctrl && shift" src/input/keymap.rs`; `ctrl_o_copies_last_reply` + `copy_across_wrap_has_no_newline`.
- [x] **Q3 — realtime markdown + latex (edge-to-edge, stable; inline math not shredded)** (Slice 8). *Evidence:* `inline_math_not_split_by_narrow_wrap`; `styled_wrap_rowcount_matches_wrap_cache` + `plain_projection_matches_rendered_text` STILL green; `safe_commit_pos_holds_streaming_table`/`_holds_unclosed_math`/`_holds_open_fence`; `block_math_renders_inside_prose_paragraph`; dump `busy` half-typed `$$` → `! grep -F '$$'`.
- [x] **Q4 — token ↑/↓ render LIVE during a turn** (Slice 4). *Evidence:* `spinner_token_readout_reflects_live_counts`; `--dump-frame busy` spinner row has `↑ ` + `↓ ` + seeded counts.
- [x] **Q5 — empty composer: Left → session view, Right → conversation view** (Slice 5). *Evidence:* `left_on_empty_composer_opens_dashboard`; doc comment matches behavior.
- [x] **Q6 — full v2 keybinding parity; shift+enter OR ctrl+enter newline; `/` dedup** (Slices 5+6). *Evidence:* `ctrl_enter_inserts_newline`; `registry_resolves_all_commands` (updated count); `aliases_marked_not_duplicated`; greps for `Ctrl+T`/`Ctrl+/`/`Ctrl+Enter` arms.
- [x] **Q7 — slogan + header(llm·model·dir·session) + above-composer done-line + 2 rows below + no `❯ chat` + `/keybindings`, left-aligned** (Slice 3, overlay Slice 6). *Evidence:* `header_has_slogan_llm_model_dir_session`; `below_composer_has_two_rows` (row2 `⎿ `); `done_line_shows_elapsed_and_tokens_when_idle`; `! grep -F '❯ chat'`; `connection_chip_survives_footer_removal`; dump `keybindings`.
- [x] **Q8 — expand-to-tool / fold-to-summary; click ANY ▸ node; resize/scroll-stable; ZERO "Turn N"** (Slice 8). *Evidence:* `no_turn_n_anywhere`; `turn_title_falls_back_to_tool_then_generic`; `tail_strips_turn_marker`; `toggle_fold_flips_single_node`; `expanded_tool_result_skips_truncation`; dump folded no-summary turn → `! grep -F 'Turn '`.
- [x] **Q9 — spinner = `⠿` only; emoji/face confined to pet; tab defaults to bear** (Slice 4). *Evidence:* `spinner_emits_braille_all_dots` + dump `busy` row has `⠿`, `! grep -F '◜'`; `arc_is_the_default_and_not_the_cc_asterisk` green; `tab_status_and_title_track_state` title contains `ʕ•ᴥ•ʔ`.
- [x] **Q10 — no dup cmds; all work; zh/en parity; `/scheduler` real modes; `/mouse` default-on & removed; `/continue` replays ALL; no `✅`** (Slices 6+7). *Evidence:* `registry_resolves_all_commands` + `dictionaries_cover_the_same_keys`; `scheduler_discovers_reflect_and_cron` + dump `scheduler`, `! grep -F '09:00'`; `mouse_command_removed`; bridge `extract_ui_messages` invoked; dump `continue` `! grep -F '✅'`.
- [x] **Q11 — `/theme` rename `ga-default`→`default` + redesigned dark/light; `/effects` per-command border + char FX** (Slices 1+2). *Evidence:* `default_is_index_0_and_ga_default_alias_resolves`; `theme_count_at_least_6` (≥8); `! grep -rn '"ga-default"' src/`; `each_fxborder_yields_distinct_color_at_0`; `fx_command_maps_four_words_rejects_hivemind`; `command_word_spans_goal_returns_styled_spans`; dump `/goal` vs `/hive` different corner glyph.
- [x] **Q12 — eggs/tips bilingual; `@` fast; Ctrl+S no freeze** (Slices 4+9). *Evidence:* `gerunds_parity` + `gerund_rotation_deterministic` (zh pool); `dictionaries_cover_the_same_keys`; `file_index_walks_once_within_ttl` + `default_skip_excludes_temp`; Ctrl+S re-verified no-stall.

---

## New i18n keys (both EN + ZH — `dictionaries_cover_the_same_keys` gate)
- [x] scheduler (~12): `scheduler.kind.{reflect,cron}`, `scheduler.cadence.reflect`, `scheduler.repeat.{once,daily,weekday,weekly,monthly}`, `scheduler.empty`.
- [x] continue: `continue.restored` (no ✅), `continue.replaying`.
- [x] mouse/help: `mouse.hint.native`, `help.alias_of`.
- [x] gerunds: `GERUNDS_ZH` parallel pool, equal length.
- [x] tips: 3 EN+ZH pairs (scheduler / mouse-off / continue-replay), equal length.

## New dump-frame scenarios to add
- [x] `done` (idle + `last_turn_ms`) — Q7. · [x] `keybindings` — Q7. · [x] `scheduler` — Q10. · [x] `continue` — Q10.
- [x] existing kept byte-identical through Slice 0: `normal`, `shell`, `busy`, `effort`, `effort-high`, `cost`.

---

## Iteration log
- **2026-06-01 R3 kickoff** — temp cleaned (724 scratch dirs); recon workflow (7 agents, 1.6M tok) → `IMPLEMENTATION_PLAN.md` (10 slices) + this checklist; baseline 281 tests + 6 dumps captured. Slice 0 dispatched.
- **2026-06-01 Slice 0 ✅ VERIFIED** — god-file refactor (6 gated sub-steps, 1.22M tok): main.rs 2505→516, app/mod.rs 2078→621, render.rs 1070→595, components/mod.rs→`cockpit/` (≤255), overlay.rs→`overlay/` (143); `AppEvent` bus severs `tx_bridge` from handlers; 281 tests + 6 dumps byte-identical; 0 warnings; 76 src files. Slices 1-9 dispatched.
- **2026-06-01 Slices 1-9 ✅** — feature slices (Q1-Q12) landed across 3 gated workflows (1-7, then 8a-8e+9 after an Opus weekly-limit pause/recovery); 281→331 tests, 0 warnings, the 4 parity invariants held through the riskiest md/latex/fold work.
- **2026-06-01 Monitor ✅ + ARCH gap closed** — 5 adversarial verifiers CONFIRMED Q1-Q12 with *rendered-output* evidence (genuine dual-path parity, `no_turn_n_anywhere` full scan, the `/continue` bridge actually replaying `extract_ui_messages`, real scheduler modes). One GAP (app/mod.rs 861 > 800) closed by extracting `app/fold_hit.rs` → mod.rs **705**. Final: **331 tests / 0 fail / 0 warn**; all 5 god-file budgets met. **ROUND 3 COMPLETE — proceeding to release (v0.3.0); live aesthetic review pending.**
