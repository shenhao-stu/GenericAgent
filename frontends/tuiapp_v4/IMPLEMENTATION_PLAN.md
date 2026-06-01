# tui_v4 round-3 polish — SEQUENCED implementation plan (single Opus session)

> **Synthesis of** `recon/round3/{ARCH_refactor,C1_render_stream,C2_layout_chrome,C3_keys_copy,C4_theme_effects,C5_commands_i18n}.md`.
> **Closes** `query.md` Q1..Q12 + the ARCH god-file split.
> **Mode:** ONE coherent Opus session, slice by slice. NOT parallel agents. `cargo build` stays green between every slice; `cargo test` green at the end of every slice.
> **Discipline:** `memory/code_review_principles.md` (P1 module boundary, P2 local reasoning, P4 small change-radius, P6 constraints-in-types, P9/P10 minimal comments, P11 no render side-effects, P12 features-up/lines-flat). When code moves files, its `#[cfg(test)] mod tests` MOVES IN THE SAME EDIT and keeps `use super::*` resolving.

## Verified ground truth (do not re-discover)
- God-files: `main.rs` **2505**, `app/mod.rs` **2078**, `components/mod.rs` **1123**, `markdown/render.rs` **1070**, `components/overlay.rs` **720**.
- `components::render(frame, &mut AppState, theme, now_ms)` (`components/mod.rs:48`) MUTATES state mid-draw → P11 violation, blocks render purity (Slice 0e / F7).
- Coupling counts: `ga-default`/`ga_default` in **59** sites; `UiToCore::` constructed in **22** sites inside `main.rs`; `tx_bridge`/`send_active` in **69** sites; `fold_all` in **36** sites.
- Mouse capture FORCED on at `main.rs:2273` (`EnterAlternateScreen, EnableMouseCapture`); `app.mouse_capture` never reconciled (F7-C3).
- `--dump-frame` scenarios today: `normal | shell | busy | effort | effort-high | cost` (`main.rs:155-302`). Slices that need a new view ADD a scenario (Slice 7 `continue`, Slice 8 `dashboard`).
- Real test names that exist and gate slices: `user_band_line_spans_width_with_band_bg`, `user_row_has_band_bg`, `ga_default_uses_cc_rgb_palette`, `theme_count_at_least_6`, `theme_preview_revert`, `registry_resolves_all_commands`, `fx_command_active_only_for_orchestration`, `arc_is_the_default_and_not_the_cc_asterisk`, `safe_commit_pos_holds_back_in_flight_structures`, `styled_wrap_rowcount_matches_wrap_cache`, `plain_projection_matches_rendered_text`, `cockpit_render_rowcount_matches_plain_projection`, `turn_title_falls_back_to_tool_then_generic`, `turn_marker_not_rendered`, `dictionaries_cover_the_same_keys`, `tab_status_and_title_track_state`.

---

## Build-green ordering rationale (why this sequence)
1. **Slice 0 (ARCH) FIRST** so every feature slice edits small focused files, not 2.5k-line monsters. The spine is the `AppEvent` bus (ARCH Fix A): it severs the `tx_bridge` threading that otherwise blocks moving any key/command handler. Sub-steps are ordered so `cargo build` compiles after EACH (bus → drop transport params → cut files → render purity → md/overlay/theme splits).
2. **Feature slices grouped by file** so each is a coherent build-green unit and no file is re-touched across slices:
   - Slice 1 = `theme/mod.rs` only (Q11a).
   - Slice 2 = `components/effects_paint.rs` + `components/cockpit/composer.rs` + `theme/` FxCommand (Q11b/c) — depends on Slice 1's `lighten`/rainbow.
   - Slice 3 = `components/cockpit/{header,footer,hints→session_info+tips,composer}` relayout (Q1, Q7) — the big chrome slice, all in the cockpit dir Slice 0d created.
   - Slice 4 = `flavor/mod.rs` + the spinner/tab call-sites + `app/reducer.rs` last_turn stamp (Q4, Q9, Q12-eggs).
   - Slice 5 = `input/keymap.rs` + `input/views.rs` + `term.rs` + `clipboard.rs` (Q2, Q5, Q6 keys/copy/mouse) — all in the input/term/clipboard files Slice 0b created.
   - Slice 6 = `commands/{registry,dispatch}.rs` + `i18n/mod.rs` (Q6-dedup, Q10-aliases, Q7 `/keybindings`).
   - Slice 7 = `scripts/ga_bridge.py` + `components/scheduler.rs` + `i18n` (Q10 `/continue` replay + de-icon, `/scheduler` real modes).
   - Slice 8 = `markdown/{render,inline_math,mod}.rs` + `render/fold.rs` + `app/Block` cache (Q3 realtime md+latex, Q8 fold/expand) — the largest render slice, done LAST because it depends on the md split (Slice 0f) and touches the most invariant-laden code.
   - Slice 9 = `app/mod.rs` `list_project_files` cache (Q12 @ speed, Q12 Ctrl+S no-freeze) — tiny, isolated, last.

> **Why Q3/Q8 last:** they carry the load-bearing parity invariant `styled_wrap_rowcount_matches_wrap_cache` / `plain_projection_matches_rendered_text`. Doing them after the refactor means they edit small `markdown/` files, and doing them last means a parity regression can't mask an earlier slice's failure.

---

# SLICE 0 — ARCH refactor (split the god-files). FIRST. Behavior identical.

Pure refactor, no Q items close here, but EVERY later slice depends on it. Land as its own commit(s). Execute sub-steps **in order**; build after each.

## 0a — Introduce the `AppEvent` bus (ARCH Fix A; severs F1+F6)
- NEW `src/app_event.rs`:
  ```rust
  pub enum AppEvent {
      ToActive(crate::bridge::protocol::UiToCore),
      ToSession(u64, crate::bridge::protocol::UiToCore),
      OpenWorkflows, OpenDashboard, CloseView,
      Copy { text: String, label: &'static str },
      SetMouseCapture(bool),
      Quit,
  }
  ```
- Add `actions: Vec<AppEvent>` field on `AppState` + `pub fn emit(&mut self, AppEvent)` + `pub fn drain_actions(&mut self) -> Vec<AppEvent>`.
- In `main.rs` event loop, AFTER `handle_term_event`, drain and perform — the ONE place `tx_bridge` lives:
  ```rust
  for ev in app.drain_actions() { match ev {
      AppEvent::ToActive(f)      => app.sessions.send_to(app.sessions.active, f, tx_bridge),
      AppEvent::ToSession(id,f)  => app.sessions.send_to(id, f, tx_bridge),
      AppEvent::Copy{text,label} => clipboard::perform_copy(app, &text, label),
      AppEvent::Quit             => app.should_quit = true,
      AppEvent::OpenDashboard    => app.open_dashboard(),
      AppEvent::CloseView        => app.close_dashboard(),
      AppEvent::OpenWorkflows    => app.open_workflows(),
      AppEvent::SetMouseCapture(on) => { app.mouse_capture=on; set_mouse_capture(on); }
  }}
  ```
  Keep the OLD inline `send_active(...)` calls working at this step → compiles, behavior identical.
- **VERIFY:** `cargo build` + `cargo test` (full suite still green; no behavior change yet).

## 0b — Convert dispatch/cockpit/apply fns to emit intents; drop `tx_bridge` params
- One fn at a time, replace `send_active(app, tx_bridge, f)` → `app.emit(AppEvent::ToActive(f))`; delete the `tx_bridge: &Sender<(u64,BridgeEvent)>` parameter. Each fn still lives in `main.rs` → compiles after each conversion. Targets: `dispatch_slash` (1591), `open_ui_command` (1627), `app_command` (1844), `dispatch_action` (1553), `apply_picker`/`apply_ask_user`/`apply_scheduler`/`apply_effort` (1249-1817), `dashboard_new_session` (1427).
- **VERIFY:** `cargo build` after each fn; `cargo test` at the end of 0b.

## 0c — Cut the now-transport-free fns into new files (ARCH Fix B + G)
Move map (each `pub(crate)`, fix imports, MOVE its tests):

| New file | Moves OUT of `main.rs` (line) | pub surface |
|---|---|---|
| `src/input/keymap.rs` | `handle_key_event` (521), `try_complete_dropdown` (1481), `composer_width` (514), global chords | `pub(crate) fn cockpit_key(&mut AppState, KeyEvent, now_ms) -> Vec<AppEvent>`, `pub(crate) fn is_quit_chord(KeyEvent)->bool` |
| `src/input/views.rs` | `handle_dashboard_key` (853), `handle_workflows_key` (765), `handle_overlay_key` (1007), `route_view_key` (751), `fire_workflow_action` (818) | `pub(crate) fn route_view_key(...) -> Vec<AppEvent>` |
| `src/input/mouse.rs` | `handle_mouse_event` (450) + its integration test `right_click_returns_from_dashboard` (2467) | `pub(crate) fn mouse(&mut AppState, MouseEvent) -> Vec<AppEvent>` |
| `src/commands/dispatch.rs` | `dispatch_slash` (1591), `open_ui_command` (1627), `app_command` (1844), `dispatch_action` (1553), `apply_*` (1249-1817), `*_picker_items` (1942/1974), `new_ask_id` (1832), `effort_command_frame` (1825) | `pub(crate) fn dispatch_slash(&mut AppState, &str) -> Vec<AppEvent>` |
| `src/term.rs` | `setup_terminal`/`restore_terminal`/`install_panic_hook`/`set_mouse_capture` (2203/2269-2304) | `pub fn setup()/restore()/install_panic_hook()/set_mouse_capture(bool)` |
| `src/clipboard.rs` | `copy_text` (2166), `notice_copy` (2178), `read_clipboard` (2213), `export_action` (2001), NEW `perform_copy` | `pub(crate) fn perform_copy(&mut AppState,&str,&'static str)`, `export_action` |

- **Fix G (dedup) in this slice:** add `input/keymap.rs::is_quit_chord`, call once at the top of `route_view_key` (returns `vec![AppEvent::Quit]`), DELETE the four copies at `main.rs:566,772,898,1015`. In `dashboard_new_session`, replace the inline `/name args` parse with `commands::registry::split_command(&expanded)`.
- `main.rs` shrinks to: `main()` + arg parse + `run_app` + `event_loop` + 3 thread spawners + `--smoke`/`--dump-frame` harness (~450 LoC).
- **VERIFY:** `cargo build`; `cargo test` (moved tests resolve `use super::*`); `wc -l src/main.rs` < 700.

## 0d — Split `components/mod.rs` → `components/cockpit/` (ARCH Fix D; mirror Codex `bottom_pane/`)
| New file | Moves (line) |
|---|---|
| `components/cockpit/mod.rs` | `render`/`render_cockpit` layout (48-162) |
| `components/cockpit/header.rs` | `render_header` (167) |
| `components/cockpit/transcript.rs` | `render_transcript` (266), `user_band_line` (357), `gutter_for` (376) |
| `components/cockpit/composer.rs` | `render_composer` (442), `composer_lines` (494), `split_at_col` (562) |
| `components/cockpit/footer.rs` | `render_footer` (591), `mode_indicator` (676), `ctx_bar` (687), `render_hints` (196), `render_separator` (252), `render_spinner` (387) |
| `components/cockpit/dropdown.rs` | `render_dropdown` (716), `render_palette` (731), `render_file_picker` (771), `dropdown_height`/`composer_height` (696/807) |
| `components/text.rs` | `compact_cwd`, `truncate_model`+`MODEL_LABEL_CAP`, `human_count`, `clip_to`, `fx_command_active`, `ctrl_key_label` (pure helpers; tests move too) |

- **VERIFY:** `cargo build`; `cargo test` (esp. `compact_cwd_keeps_tail`, `human_count_compacts_thousands_and_millions`, `user_band_line_spans_width_with_band_bg`, `truncate_model_primary_segment`); `cargo run -- --dump-frame normal` byte-identical to pre-slice.

## 0e — Make render PURE (ARCH Fix D / C1-F7; P11)
- Hoist the two state writes out of `render`. Add `AppState::prepare_frame(&mut self, area: Rect, theme: &Theme)` that does `set_term_size` + `sync_transcript`. In the event loop, call it BEFORE `terminal.draw`, then `components::cockpit::render(f, &app, theme, now_ms)` takes `&AppState` (immutable).
- The `--smoke`/`--dump-frame` harness updates to `app.prepare_frame(area, &theme); terminal.draw(|f| components::cockpit::render(f, &app, &theme, 100))`.
- **VERIFY:** `cargo build`; `cargo test` (`cockpit_render_rowcount_matches_plain_projection`, `styled_wrap_rowcount_matches_wrap_cache` must stay green — proves the hoist didn't change geometry); dump-frame all 6 scenarios byte-identical.

## 0f — Split `app/mod.rs` (ARCH Fix C) + `markdown/render.rs` (Fix E) + `components/overlay.rs` (Fix F)
- `app/mod.rs` → `app/reducer.rs` (`apply_bridge_event`/`apply_frame`/`block_for_mid_mut`/`rewind_real_turns_from` + the `FrameSink` trait shared by `AppState` and `Session`, killing F8a dup), `app/overlay_ops.rs`, `app/types.rs` (`Overlay`/`Role`/`Block`/`ConnStatus`/`CostBreakdown`/`PendingAsk`/`RenameState`/`View`), `app/multi.rs`. `mod.rs` keeps `struct AppState` + `Default`/`new` + tick + `sync_transcript`/`scroll_*` + the `actions` queue + `prepare_frame`.
- `markdown/render.rs` → `markdown/inline_math.rs` (`Seg`, `split_inline_math`, `find_math_close`, `is_probably_math`, `extract_block_math`, `render_block_math` + their tests), `markdown/table.rs` (`TableBuf`/`emit_table`/`table_row`/`pad_cell`), `markdown/code.rs` (`CodeBuf`/`emit_code_block`/`heading_style`/`bullet`). `render.rs` keeps `render_markdown` + the `Walker` core.
- `components/overlay.rs` → `overlay/mod.rs` (dispatch + `centered` + `titled_block`), `overlay/effort.rs` (`render_effort_slider`), `overlay/picker.rs` (`render_picker`/`render_ask_user`), `overlay/info.rs` (`render_help`/`render_status`/`render_cost`/`render_verbose`/`render_btw`), `overlay/effects.rs` (`render_effects_demo`/`render_demo_legend`).
- **Fix H (P9):** in the MOVED files only, strip comments that restate code; keep `//!` module header + non-obvious invariants. Collapse the `--dump-frame` seed prose to one line per scenario.
- **VERIFY:** `cargo build`; `cargo test` (full suite — `md_table`, `block_math_extraction_and_render`, `routes_block_math_to_stacked_layout`, `every_token_resolves_for_every_theme`); `cargo build 2>&1 | grep -c warning` not increased (catches over-`pub` leaks — enforce `pub(crate)`).

**Slice 0 DONE gate:** all 5 god-files < 800 LoC excl. tests; full `cargo test` green; `--dump-frame {normal,shell,busy,effort,effort-high,cost}` byte-identical to the pre-refactor baseline (capture baselines BEFORE 0a).

---

# SLICE 1 — `/theme` rename + redesigned palettes (Q11a)
**Closes:** Q11 (theme half). **Files:** `src/theme/mod.rs` (+ `theme/rainbow.rs` if rainbow const lands there).
**Changes (C4 Fix a):**
- Rename `"ga-default"` → `"default"` across all **59** sites; constructor `ga_default()` → `default_theme()`; `impl Default` calls it; registry keeps it FIRST (picker index 0). `main.rs:1657/1674` need NO change (read `app.theme.name` dynamically) — that is WHY the literal must be consistent.
- **Migration insurance:** `by_name("ga-default")` resolves to `default` for one release (C4 Open-Q1).
- Add shared `const CC_RAINBOW: [Color;7]` (muted ROYGBIV) reused by `default`/`light`/new light themes (collapses 6 ad-hoc rainbows → 1; P12).
- Add `fn lighten(c, k)` helper; replace hand-tuned `*Shimmer` hex with `lighten(base, k)` (P9/P12).
- Add 2 light themes: `catppuccin_latte()`, `solarized_light()` (token order = `Token::ALL`). Keep CC stark `light` → 3 light total.
- Adopt CC `darkTheme` values verbatim for `default` (already close per `ga_default_uses_cc_rgb_palette`).
**VERIFY:** `cargo build` + `cargo test theme` — `ga_default_uses_cc_rgb_palette` (rename its assertion to `"default"`), `every_token_resolves_for_every_theme`, `theme_count_at_least_6` (now ≥8), `theme_preview_revert`. New test `default_is_index_0_and_ga_default_alias_resolves`. **Dump:** `--dump-frame normal` header separator reads soft CC rainbow (no neon).

# SLICE 2 — `/effects` per-command identity + char FX (Q11b/c)
**Closes:** Q11 (effects half). **Files:** `src/theme/mod.rs` (FxCommand+FxBorder), `src/components/effects_paint.rs`, `src/components/cockpit/composer.rs`, `src/components/text.rs` (fx_command). **Depends on Slice 1** (`lighten`, `CC_RAINBOW`).
**Changes (C4 Fix b/c):**
- Replace `fx_command_active(&str)->bool` with `fx_command(&str)->Option<FxCommand>` (P6: identity into the type). Keep `#[inline] fn fx_command_active = fx_command(_).is_some()` so existing tests stay green.
- `enum FxCommand{Goal,Hive,Conductor,Morphling}`; `enum FxBorder{Pulse,Orbit,Sweep,Rainbow}` with `corner: char`. `FxCommand::border(&Theme)->FxBorder` derives colors from EXISTING tokens (no inline RGB; colors stay in `theme/`).
- `draw_composer_border_fx(.., cmd: FxCommand)`: ONE perimeter `color_at` closure switched by `FxBorder` (reuse the edge loops unchanged — 4 identities = one `match`, not four painters; P5/P12). Goal=Pulse◆ Claude; Hive=Orbit⬡ Success; Conductor=Sweep▸ Suggestion; Morphling=Rainbow◆ CC_RAINBOW. Avoid per-frame `Box<dyn Fn>` alloc — inline the 4 cases (effects tenet "no per-frame alloc").
- `command_word_spans(word, cmd, theme, phase)`: per-grapheme fg per command (Morphling=hue march, Hive=mint shades, Goal/Conductor=sheen sweep). Wire at the composer row-0 leading `/word`, mirroring the existing `bang_pink` peel; skip if cursor sits in the word. Pass `app.effects.clock` for phase.
- Call site: `if let Some(cmd)=fx_command(app.composer.text()) { draw_composer_border_fx(.., cmd) }`.
**VERIFY:** `cargo build` + `cargo test` — keep `fx_command_active_only_for_orchestration`; ADD `fx_command_maps_four_words_rejects_hivemind`, `each_fxborder_yields_distinct_color_at_0`, `command_word_spans_goal_returns_styled_spans`. **Dump:** extend `/effects demo` OR add a transient seed; manual `--dump-frame` with composer seeded `/goal` vs `/hive` shows different corner glyph (◆ vs ⬡).

# SLICE 3 — chrome relayout: history-input ❯ band + header + above/below-composer rows (Q1, Q7)
**Closes:** Q1, Q7. **Files:** `src/components/cockpit/{transcript,header,footer,composer,mod}.rs`, `src/components/cockpit/` NEW `session_info.rs`+`tips.rs` (or in footer.rs), `src/components/text.rs` (`llm_channel`, `fmt_dur`), `src/app/{mod,reducer}.rs` (`last_turn_ms`), `src/app/session.rs` (`active_name()`).
**Changes (C2 Fix A–F):**
- **Q1 (Fix A):** `user_band_line` — lead `"❯ "` styled with band bg, body budget `-2`; whole row stays `Token::UserBand`. Update `user_band_line_spans_width_with_band_bg` (`starts_with("❯ hello")`); DELETE the `:1121` negative assertion `!starts_with('❯')` (it now contradicts the spec); keep every-cell-bg assertion in `user_row_has_band_bg`.
- **Q7 header (Fix B):** `render_header` → slogan + `llm · model · dir · session`, left-aligned. Add pure `llm_channel(Option<&str>)->&str` and `SessionMap::active_name()`. Pick slogan const (recommend `❯❯ GenericAgent` — rhymes with the input prefix).
- **Q7 above-composer done-line (Fix C):** add `pub last_turn_ms: Option<u64>` on `AppState`; stamp it in `apply_frame` `MessageEnd` (now in `app/reducer.rs`) before `busy=false`; CLEAR on next `MessageBegin`. New `render_done_line` (1 row, shown when `!busy && last_turn_ms.is_some()`): `⠿ <gerund> for <fmt_dur> · ↑ <in> · ↓ <out>`, mint `⠿`, frozen (no animation). Add pure `fmt_dur(secs)->String` (`1m 46s`).
- **Q7 two below-composer rows (Fix D):** replace the single hint constraint with TWO `Length(1)`. Row1 `render_session_info` = `llm · model · effort · ctx · branch` (moved OUT of footer). Row2 `render_tips` = `⎿ ` + rotating tip (leader glyph restored). Armed-Ctrl+C hint becomes a transient override on row2 in `Token::Warning`.
- **Q7 footer (Fix E):** delete `❯ chat`. Prefer removing the footer row entirely (row1 now owns session info) BUT fold the connection-status chip into row1's tail (Token::Error when disconnected) — N1 "never a silent disconnect" must survive.
- **Q7 `/keybindings` (Fix F):** see Slice 6 (command + overlay land there); here just leave the keybinding pairs OUT of the chrome.
**VERIFY:** `cargo build` + `cargo test` (`user_band_line_spans_width_with_band_bg`, `user_row_has_band_bg`, ADD `done_line_shows_elapsed_and_tokens_when_idle`, `header_has_slogan_llm_model_dir_session`, `below_composer_has_two_rows`, `connection_chip_survives_footer_removal`). **Dump:** `--dump-frame normal` shows `❯ hello` band + new header + 2 rows + NO `❯ chat`; need a NEW done-line seed → add `--dump-frame done` scenario (idle with `last_turn_ms` set) OR assert on `normal` after `MessageEnd` sets it.

# SLICE 4 — spinner=⠿, pet/tab, live token render, bilingual eggs (Q4, Q9, Q12-eggs)
**Closes:** Q4, Q9, Q12 (eggs+tips half). **Files:** `src/flavor/mod.rs`, `src/components/cockpit/footer.rs` (`render_spinner`), `src/app/mod.rs` (`terminal_title`), `src/i18n/mod.rs` (tips), `src/util/osc.rs` (verify title passthrough).
**Changes (C2 Fix G/H + C5 Fix F9/tips):**
- **Q9 spinner (Fix G):** `render_spinner` glyph = static `'⠿'` (drop the per-tick frame index). Animation comes from pet blink + gerund/heat ramp. Pet stays kaomoji (`PetStyle::Off` default unchanged). `arc_is_the_default_and_not_the_cc_asterisk` unaffected; ADD `spinner_emits_braille_all_dots`.
- **Q9 tab=bear (Fix H):** `terminal_title()` leads with `flavor::PETS_BEAR[0][0]` (`ʕ•ᴥ•ʔ`) in BOTH states. Update `tab_status_and_title_track_state` (`:1632` `.contains("GenericAgent")`; busy-glyph assertion still holds — bear leads).
- **Q4 live tokens:** verify `render_spinner`'s `↑in ↓out` reads `app.tok_in/tok_out` live (they update on each `Status` frame). If the spinner caches them, bind to the live fields so the readout animates per delta. (The done-line in Slice 3 freezes them; the spinner shows live.) Add `spinner_token_readout_reflects_live_counts`.
- **Q12 bilingual eggs (C5 F9):** add `GERUNDS_ZH: &[&str]` parallel to `GERUNDS`, SAME length; `gerund(lang, tick)` picks pool by `Lang` (signature change). Translate the existing 34 + append the new EN/ZH pairs from C5 appendix. ADD `gerunds_parity` test (`assert_eq!(GERUNDS.len(), GERUNDS_ZH.len())`).
- **Q12 tips:** append the 3 EN+ZH tip pairs (scheduler/mouse/continue) from C5 appendix to `TIPS_EN`+`TIPS_ZH`, keep lengths equal.
**VERIFY:** `cargo build` + `cargo test flavor` + `cargo test i18n` (`gerund_rotation_deterministic` updated for `lang`, `gerunds_parity`, `tips_rotate_deterministically_per_language`, `dictionaries_cover_the_same_keys`). **Dump:** `--dump-frame busy` spinner row shows `⠿` + live `↑ ↓`; title assert via test (no TTY).

# SLICE 5 — keys/copy/mouse: Left/Right swap, native copy, Ctrl+Enter, Ctrl+O (Q2, Q5, Q6-keys)
**Closes:** Q2, Q5, Q6 (keybinding half). **Files:** `src/input/keymap.rs`, `src/input/views.rs`, `src/input/mouse.rs`, `src/term.rs`, `src/app/mod.rs` (`mouse_capture` default), `src/clipboard.rs`.
**Changes (C3 Fix B/C/C-newline/D + C5 D4):**
- **Q5 Left/Right swap (Fix B):** empty composer `Left → open_dashboard()`, `Right → close_dashboard()` (currently reversed at `main.rs:702-707`). Mirror the dashboard side. Update the wrong doc comment. ADD `left_on_empty_composer_opens_dashboard`.
- **Q2 native copy (Fix C + C5 D4):** `term.rs::setup` drops `EnableMouseCapture` (mouse capture OFF default → terminal owns drag-select). `AppState::new()` default `mouse_capture=false` (F7 reconcile). Keep `Ctrl+Shift+M` + `/mouse`→ wait, `/mouse` is REMOVED in Slice 6; keep ONLY `Ctrl+Shift+M` as the wheel-scroll opt-in. Keep PgUp/PgDn/Ctrl+Home/End keyboard scroll.
- **Q6 Ctrl+Enter newline (Fix C-newline):** `KeyCode::Enter if shift || ctrl => composer.newline()`. Guard: must sit AFTER `try_complete_dropdown`; also add `&& !ctrl` to the dropdown Enter arms (`:1512,1538`) so Ctrl+Enter isn't swallowed as completion (C3 Open-Q2).
- **Q2 explicit clean-copy = Ctrl+O (Fix D):** REMOVE the `ctrl+shift+c`=copy-reply arm (`main.rs:549-552`). Bind `Ctrl+O` → copy last reply via `clipboard::perform_copy` (emits `AppEvent::Copy`, P2-clean `block.source`). MOVE fold off Ctrl+O → `/fold` app-command (Slice 6) OR `Ctrl+Shift+O`. (C3 Open-Q1 decision: Ctrl+O=copy, align Codex.)
**VERIFY:** `cargo build` + `cargo test input` + `cargo test` (copy: `copy_across_wrap_has_no_newline` still green; ADD `ctrl_enter_inserts_newline`, `ctrl_o_copies_last_reply`, `mouse_capture_defaults_off`, `left_on_empty_composer_opens_dashboard`). **Dump:** N/A (input) — assert via key tests on `AppState`.

# SLICE 6 — command dedup, aliases, `/keybindings`, fold→/fold, drop /mouse (Q6-dedup, Q7, Q10-aliases)
**Closes:** Q6 (dedup half), Q7 (`/keybindings`), Q10 (alias presentation). **Files:** `src/commands/registry.rs`, `src/commands/dispatch.rs`, `src/i18n/mod.rs`, `src/app/types.rs` (Overlay::Keybindings or reuse Help), `src/components/overlay/info.rs`.
**Changes (C3 dedup + C5 D7 + C2 Fix F + C5 D4):**
- **Q6 drop alias commands** (C3): remove `sessions`, `abort`, `tools`, `trace`, `exit` from `COMMANDS` + their dispatch arms (40→~34). BUT C5 D7 prefers KEEPING them as marked aliases — RESOLUTION: keep them resolvable (add `alias_of: Option<&'static str>` to `SlashCommand`, set on those 5), but `/help` lists them dimmed as "alias of X" not as peer rows, and `palette_matches` does not surface alias+primary as two hits. (This satisfies both specs: no duplicate primary names; aliases still work.)
- **Q10 drop `/mouse`** (C5 D4): remove `/mouse` row from `registry.rs:83` + the `all[]` test list + decrement `assert_eq!(COMMANDS.len(), all.len())`; `>=33` floor still holds.
- **Q7 `/keybindings`** (C2 Fix F): add `/keybindings` command + `Ctrl+/` chord (in Slice 5's keymap — add the chord there if not yet; here add the command+overlay) → `Overlay::Keybindings` (or Help) rendering the keybinding pairs table + magic-prefix line. Also add `Ctrl+T`→`/theme` and `Ctrl+/`(+legacy `Ctrl+_`)→help GAPS from C3 parity table.
- **fold→/fold:** add `/fold` app-command (Ctrl+O was reassigned to copy in Slice 5).
- **i18n:** add `help.alias_of`, `mouse.hint.native` to BOTH dicts.
**VERIFY:** `cargo build` + `cargo test commands` + `cargo test i18n` (`registry_resolves_all_commands` updated `all[]`+count, `palette_fuzzy_ranks_prefix_then_subsequence`, `did_you_mean_suggests_closest`, `dictionaries_cover_the_same_keys`; ADD `aliases_marked_not_duplicated`, `mouse_command_removed`). **Dump:** add `--dump-frame keybindings` (open the overlay) showing the pairs table.

# SLICE 7 — `/continue` replay + de-icon, `/scheduler` real reflect modes (Q10)
**Closes:** Q10 (`/continue`, `/scheduler`, icon). **Files:** `frontends/tuiapp_v4/scripts/ga_bridge.py` (NOT GA-core `continue_cmd.py`), `src/components/scheduler.rs`, `src/commands/dispatch.rs` (scheduler diff payload), `src/i18n/mod.rs`.
**Changes (C5 D1/D2/D3):**
- **Q10 `/continue` replay (D1):** in `ga_bridge.py::handle_restore`, after `continue_cmd.restore`, loop `continue_cmd.extract_ui_messages(path)` emitting one `MessageBegin/Delta/End` triple per bubble (v3 parity), THEN the de-iconified banner as a final system line. NO Rust change for the replay (frames render through the existing path).
- **Q10 de-icon (D2):** add `_deicon_restore(msg,n)` in `ga_bridge.py` stripping leading `✅`/`⚠️`/`❌`. Do NOT edit `continue_cmd.py` (shared by v2/v3/st/tg/dc/qt).
- **Q10 `/scheduler` real modes (D3):** `components/scheduler.rs::discover_tasks(repo_root)` reads `reflect/*.py` (non-`_`, skip `scheduler.py`) → reflect-mode rows + `sche_tasks/*.json` → cron rows with `read_cron_meta` (repeat label + enabled). Replace `default_tasks()` at the open path with `discover_tasks(&app.repo_root)`; keep `default_tasks()` ONLY as empty-dir fallback. Forward the computed `to_start`/`to_stop` diff to the bridge as `Command{name:"scheduler",args:"start <n>;stop <n>"}` (currently dropped — P7). Render the raw legal cadence labels (`once/daily/weekday/weekly/monthly/every_*`), never invented `09:00`.
- **i18n:** add the ~12 scheduler/reflect keys + `continue.restored`/`continue.replaying` (icon-free) to BOTH dicts.
**VERIFY:** `cargo build` + `cargo test` (`dictionaries_cover_the_same_keys`; ADD `scheduler_discovers_reflect_and_cron`, `scheduler_falls_back_to_default_on_empty`). Python: `python -c "import ast; ast.parse(open('scripts/ga_bridge.py').read())"` + a bridge smoke that `extract_ui_messages` is called. **Dump:** add `--dump-frame scheduler` (open overlay seeded with discovered tasks) showing reflect-mode + cron rows, no fake `09:00`; add `--dump-frame continue` proving the banner has NO `✅`.

# SLICE 8 — realtime markdown+latex + per-node fold/expand + kill residual "Turn N" (Q3, Q8)
**Closes:** Q3, Q8. **Files:** `src/markdown/{render,inline_math,mod}.rs`, `src/markdown/measure.rs`(if atomic-wrap touches cache), `src/render/fold.rs`, `src/render/chip.rs`, `src/app/{mod,types}.rs` (`Block.cockpit_cache`, `folds`, `node_hit`), `src/input/mouse.rs` (hit-test), `src/input/keymap.rs` (Ctrl+O is copy now; fold-all stays its own chord/`/fold`).
**Changes (C1 Fix A–F):**
- **Q3 atomic math wrap (Fix A):** tag rendered inline-math spans with `const ATOMIC: Modifier = Modifier::RAPID_BLINK` sentinel; teach BOTH `wrap_styled_hard_line` AND the cache's `reflow_block` to consume an ATOMIC run as one unit via a shared `wrap_line_segments_atomic(line,width,atomic_ranges)`; strip ATOMIC in `flush_line`. **Riskiest change** — run `styled_wrap_rowcount_matches_wrap_cache` + `plain_projection_matches_rendered_text` across math fixtures FIRST.
- **Q3 per-paragraph block math (Fix B):** route `$$…$$` at the paragraph boundary in `Walker` (not whole-source); keep whole-source `extract_block_math` short-circuit ONLY when `!streaming`.
- **Q3 holdback (Fix C):** extend `safe_commit_pos` to hold back unclosed `$`/`$$`, open fences, and streaming GFM tables (`streaming_table_start` ≈ Codex `TableHoldbackScanner`). Pure helpers + tests `safe_commit_pos_holds_streaming_table`/`_holds_unclosed_math`/`_holds_open_fence` (companions to the existing `safe_commit_pos_holds_back_in_flight_structures`).
- **Q3 tail-render + memo (Fix D):** render the volatile tail through `render_turn_body` (not raw dim) WITH `strip_leading_turn_line` (fixes F6 tail leak); add `Block.cockpit_cache: Option<(rev,width,fold_all,Vec<Line>,String)>`; drop the per-frame `md_cache` HashMap (block owns its cache, kills the C1-F4 dup).
- **Q8 per-node fold + click ▸ (Fix E):** `folds: HashMap<(u64,u32),bool>` on `AppState`; `fold_turns` gains `is_folded: &dyn Fn(u32)->bool`; build `node_hit: Vec<(BlockId,RangeInclusive<usize>,NodeId)>` in `sync_transcript`; mouse `Down(Left)` on the `▸`/`⏺` column → `toggle_fold(node)`; re-derive scroll anchor on the clicked node (not Bottom). Expandable tool result in `render_chip_bullet` (drop `max_preview` truncation when expanded; `▸`/`▾` affordance replacing dead `… +N more`). Ctrl+O-as-fold is GONE (it's copy); keep fold-all on `/fold` (Slice 6) — adjust `fold_all` plumbing.
- **Q8 kill "Turn N" (Fix F):** `turn_title` never bakes `format!("Turn {n}")`; fall to tool name → first prose line → `"…"`. Update `turn_title_falls_back_to_tool_then_generic`.
**VERIFY:** `cargo build` + `cargo test` (FULL — the parity invariants `styled_wrap_rowcount_matches_wrap_cache`, `plain_projection_matches_rendered_text`, `cockpit_render_rowcount_matches_plain_projection`, `embedded_newline_in_span_keeps_rowcount_parity`; `turn_marker_not_rendered`, `turn_title_falls_back_to_tool_then_generic`; `md_inline_math_is_rendered`, `block_math_extraction_and_render`; ADD `inline_math_not_split_by_narrow_wrap`, `safe_commit_pos_holds_streaming_table`, `safe_commit_pos_holds_unclosed_math`, `block_math_renders_inside_prose_paragraph`, `tail_strips_turn_marker`, `toggle_fold_flips_single_node`, `expanded_tool_result_skips_truncation`, `no_turn_n_anywhere`). **Dump:** `--dump-frame busy` (a streaming `$$` in the seed must NOT show raw `$$`); seed a no-summary turn → fold header must NOT contain "Turn".

# SLICE 9 — @ file-index cache (Q12 @ speed + Ctrl+S no-freeze)
**Closes:** Q12 (@ speed, Ctrl+S). **Files:** `src/app/mod.rs` (`list_project_files`), `src/input/paths.rs` (`DEFAULT_SKIP`).
**Changes (C5 D5/D6):**
- Memoize the project-file walk: `file_index: RefCell<Option<(Instant, Arc<Vec<String>>)>>` on `AppState`; `list_project_files(&self)->Arc<Vec<String>>` walks at most once per 5s TTL, returns a cheap `Arc` clone otherwise. The 3 call sites (`components/cockpit/dropdown.rs` ×2, `input/keymap.rs` ×1) keep calling it unchanged. If `&self`/render-borrow is awkward, stash `Arc<Vec<String>>` on the composer per `@`-session (D5 fallback).
- Add `temp` to `paths::DEFAULT_SKIP` (this repo's giant log tree) alongside the existing `target`.
- **Ctrl+S (D6):** no separate handler bug — re-verify after the cache that Ctrl+S→dashboard→describe-task no longer stalls. If residual, instrument `snapshot_active_into_map`'s `Vec<Line>` clone (measure first; candidate `Arc` transcript).
**VERIFY:** `cargo build` + `cargo test` (ADD `file_index_walks_once_within_ttl`, `default_skip_excludes_temp`). **Manual:** `cargo run` → open `@` → typing is fluid; Ctrl+S → dashboard opens without stall (or a micro-bench asserting <1 walk/frame).

---

# MONITOR GATE — per-query-item evidence a verifier MUST demand

> For each item: the EXACT dump-frame substring / test name / codepoint scan. A slice is not "done" until its gate evidence is produced (not self-reported). Codepoint scans use `cargo run -- --dump-frame <s> | grep -F` (or absence-grep `! grep`).

**Q1 (history ❯ band):** dump `normal` line for the user row STARTS WITH `❯ ` AND every cell carries `Token::UserBand` bg → test `user_band_line_spans_width_with_band_bg` asserts `starts_with("❯ hello")` + `user_row_has_band_bg` every-cell-bg. Negative: the old `:1121` assertion is DELETED (grep that it's gone).

**Q2 (native copy, no ctrl+shift+c, no-newline):** (a) `! grep -n 'EnableMouseCapture' src/term.rs` (capture off); (b) test `mouse_capture_defaults_off`; (c) `! grep -n "Char('c' | 'C') if ctrl && shift" src/input/keymap.rs` (the arm removed); (d) test `ctrl_o_copies_last_reply` + `copy_across_wrap_has_no_newline` (P2-clean source). Table/multiline: copying `block.source` is markdown-clean by construction.

**Q3 (realtime md+latex):** (a) `inline_math_not_split_by_narrow_wrap` (math fixture at width 20 keeps the glyph run intact); (b) parity invariants `styled_wrap_rowcount_matches_wrap_cache` + `plain_projection_matches_rendered_text` STILL green (the atomic-wrap didn't drift the cache); (c) `safe_commit_pos_holds_streaming_table` + `_holds_unclosed_math` + `_holds_open_fence`; (d) `block_math_renders_inside_prose_paragraph`; (e) dump `busy` with a half-typed `$$\frac{a}{` in seed → `! grep -F '$$' ` on the active region (no raw `$$` flash).

**Q4 (live tokens):** test `spinner_token_readout_reflects_live_counts` (changing `Status` frames change the readout); dump `busy` spinner row contains `↑ ` and `↓ ` with the seeded numbers (`1.2k`/`340`).

**Q5 (Left/Right):** test `left_on_empty_composer_opens_dashboard` (Left → `View::Dashboard`, Right → `View::Cockpit`); grep the doc comment matches the behavior (no inverted prose).

**Q6 (v2 parity + dedup + shift/ctrl+enter):** (a) tests `ctrl_enter_inserts_newline`; (b) `registry_resolves_all_commands` with updated `all[]` + `COMMANDS.len()` matching (5 aliases marked, `/mouse` removed); (c) `aliases_marked_not_duplicated`; (d) parity-table gaps present: greps for `Ctrl+T`→theme, `Ctrl+/`→help, `Ctrl+Enter`→newline arms in `input/keymap.rs`.

**Q7 (splash/header/2-rows/no ❯ chat/keybindings):** dump `normal` (or new `done`): (a) header line contains the slogan + `llm ` + `model ` + `dir ` + `session `; (b) EXACTLY two rows below composer — row2 starts `⎿ `; (c) above-composer done-line `⠿ ... for ...s · ↑ ... · ↓ ...` when idle-after-turn; (d) `! grep -F '❯ chat'` (absent); (e) connection chip present when disconnected; (f) dump `keybindings` overlay shows the pairs table. Tests: `header_has_slogan_llm_model_dir_session`, `below_composer_has_two_rows`, `done_line_shows_elapsed_and_tokens_when_idle`, `connection_chip_survives_footer_removal`.

**Q8 (fold/expand any node, click ▸, no Turn N):** (a) `no_turn_n_anywhere` (scan rendered output, `! contains "Turn "`); (b) `turn_title_falls_back_to_tool_then_generic` (no `Turn N` title); (c) `tail_strips_turn_marker`; (d) `toggle_fold_flips_single_node` (per-node, not global); (e) `expanded_tool_result_skips_truncation` (+ `▸`/`▾` affordance, no dead `… +N more`); (f) dump with a no-summary folded turn → `! grep -F 'Turn '` on the fold header.

**Q9 (spinner=⠿, pet-only, tab=bear):** (a) `spinner_emits_braille_all_dots` + dump `busy` spinner row contains `⠿` (U+283F) and the frame is NOT an arc char (`! grep -F '◜'`); (b) `arc_is_the_default_and_not_the_cc_asterisk` still green; (c) `tab_status_and_title_track_state` asserts `terminal_title()` contains `ʕ•ᴥ•ʔ` in both states.

**Q10 (no dup cmds, all work, zh/en, /scheduler real, /mouse default-on, /continue replay, no ✅):** (a) `registry_resolves_all_commands` (no dup) + `dictionaries_cover_the_same_keys` (zh/en parity); (b) `scheduler_discovers_reflect_and_cron` + dump `scheduler` shows real mode names (`autonomous`, `goal_mode`, `trader`, …) and cron names (`crypto_morning_brief`, …) with legal cadence labels, `! grep -F '09:00'`; (c) `mouse_command_removed` + capture-off evidence from Q2; (d) `/continue`: bridge test that `extract_ui_messages` is invoked + emits N `MessageBegin` triples, dump `continue` banner `! grep -F '✅'` (absent). UI-render sanity for `/status`,`/workflows`,`/rewind`: dump those overlays (existing `cost`/`effort` scenarios prove the overlay path; add scenarios if a reviewer demands `status`/`rewind`).

**Q11 (theme rename + new themes; per-command effects + char FX):** (a) `default_is_index_0_and_ga_default_alias_resolves` + `theme_count_at_least_6` (≥8) + `! grep -rn '"ga-default"' src/` except the alias resolver; (b) `each_fxborder_yields_distinct_color_at_0` + `fx_command_maps_four_words_rejects_hivemind` + `command_word_spans_goal_returns_styled_spans`; (c) `ga_default_uses_cc_rgb_palette` (renamed) + `every_token_resolves_for_every_theme`; (d) manual dump with `/goal` vs `/hive` in composer → different corner glyph (◆ vs ⬡).

**Q12 (eggs+tips bilingual, @ fast, Ctrl+S no freeze):** (a) `gerunds_parity` (`GERUNDS.len()==GERUNDS_ZH.len()`) + `gerund_rotation_deterministic` picks zh pool for `Lang::Zh`; (b) `dictionaries_cover_the_same_keys` (new tips); (c) `file_index_walks_once_within_ttl` + `default_skip_excludes_temp`; (d) Ctrl+S: re-verify no stall after the cache (manual `cargo run` or a walk-count assertion of ≤1/frame).

**ARCH (god-file split):** per refactor target — `wc -l` excl. tests < 800 for `main.rs`, `app/mod.rs`, `components/mod.rs`(now `cockpit/mod.rs`), `markdown/render.rs`, `components/overlay.rs`(now `overlay/mod.rs`); `cargo build` green after EACH sub-step (0a..0f); full `cargo test` green; `--dump-frame {normal,shell,busy,effort,effort-high,cost}` BYTE-IDENTICAL to the baseline captured before 0a; `cargo build 2>&1 | grep -c warning` not increased (no `pub` leaks). Evidence the bus landed: `! grep -n 'UiToCore::' src/input/ src/commands/dispatch.rs` returns ONLY `AppEvent::ToActive(UiToCore::…)` wrappers, never a raw `send_active(... tx_bridge ...)` in a key handler.
