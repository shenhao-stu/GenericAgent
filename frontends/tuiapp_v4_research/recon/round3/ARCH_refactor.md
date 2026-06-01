# tui_v4 — High-cohesion / low-coupling refactor (round 3)

Pure refactor. Behavior identical, tests green at every step. The 12 feature fixes land AFTER, into the smaller files created here. Audited against `memory/code_review_principles.md` (15 principles) + `qianxuesen_sop` system framing (FE/BE = two coupled subsystems joined by one narrow interface; (x,u,y,J): x = AppState, u = key/bridge events, y = rendered frame, J = "change radius small").

Competitor anchor: **Codex `codex-rs/tui`** is the direct ratatui peer. Its decomposition is our blueprint, and its `AGENTS.md` states the exact rule we're violating (see Competitor patterns §C0).

Current line counts (verified `wc -l`): main.rs **2505**, app/mod.rs **2078**, components/mod.rs **1123**, markdown/render.rs **1070**, components/overlay.rs **720**. Total crate 28 417 LoC across 56 files.

---

## Findings (file:line bugs, root cause not symptom)

### F1 — `main.rs` is 5 modules fused into one binary root (2505 LoC) — ROOT: no `app_event` indirection
`main.rs` is simultaneously: (a) the binary entry + arg parse + smoke/dump harness (`main.rs:52-302`), (b) terminal lifecycle (`setup_terminal`/`restore_terminal`/`install_panic_hook` `main.rs:2269-2304`), (c) the 3-source event loop (`main.rs:304-411`), (d) **all** key handling (`handle_key_event` `main.rs:521-746`, `handle_dashboard_key` `853-1002`, `handle_workflows_key` `765-810`, `handle_overlay_key` `1007-1199`), (e) **all** command dispatch (`dispatch_slash`/`open_ui_command`/`app_command` `main.rs:1591-1938`), (f) mouse (`handle_mouse_event` `450-511`), (g) copy/clipboard (`copy_text`/`notice_copy`/`read_clipboard` `2162-2215`), and (h) pure helpers that belong in `app` (`rewind_real_turns_from` `1378`, `new_ask_id` `1832`, `emoji_picker_items` `1942`, `rewind_picker_items` `1974`).

Root cause: there is **no command/action indirection layer**. Every `dispatch_*`/`cockpit_*` fn takes `&mut AppState, &Sender<(u64,BridgeEvent)>` and performs the effect inline (`main.rs:1582-1585` `send_active`). Codex severs exactly this with an `AppEvent` bus (`app_event.rs`): widgets/handlers emit an enum, one place performs the effect. Without it, key→effect is a 1400-line `if/else` forest that can't be split because every arm closes over the bridge sender.

### F2 — `app/mod.rs`: state + reducers + overlay-construction + multi-session glue tangled (2078 LoC)
One `impl AppState` mixes four unrelated concerns: the bridge **reducer** (`apply_bridge_event`/`apply_frame` `app/mod.rs:684-900`), the **render-plane sync** (`sync_transcript`/`scroll_*` `1049-1106`), the **overlay stack** constructors (`open_picker`/`apply_llm_list`/`open_ask_user`/`surface_next_ask`/`open_overlay` `916-997`), and the **multi-session swap** glue (`snapshot_active_into_map`/`load_active_fields_from`/`switch_session`/`apply_tagged_event` `1209-1430`). The `Overlay` enum + `Block`/`Role`/`CostBreakdown`/`ConnStatus` types (`app/mod.rs:37-350`) also live here. The reducer is duplicated structurally in `app/session.rs:217` (`Session::apply_frame`) — two folds of the same protocol (principle 5/8 violation, see F8).

### F3 — `components/mod.rs`: 8 distinct widgets in one file (1123 LoC)
`render` + `render_cockpit` dispatch (`components/mod.rs:48-162`) sit alongside `render_header` (167), `render_hints` (196), `render_separator` (252), `render_transcript` + `user_band_line` + `gutter_for` (266-384), `render_spinner` (387), `render_composer` + `composer_lines` + `split_at_col` (442-587), `render_footer` + `mode_indicator` (591-684), and the `render_dropdown`/`render_palette`/`render_file_picker` trio (696-801), plus pure label helpers (`compact_cwd`/`truncate_model`/`human_count`/`clip_to`/`ctx_bar` 686-931). Each is an independent widget with its own geometry; they share only `theme` + `clip_to`. This is the file Codex split into a whole `bottom_pane/` dir + a `status_surfaces.rs` (§C2).

### F4 — `markdown/render.rs`: the `Walker` block-emitter and the `$…$` math splitter are two modules (1070 LoC)
`Walker` (`render.rs:61-663`, the pulldown-cmark event→Line emitter) and the inline-math machinery (`split_inline_math` `721`, `find_math_close` `775`, `is_probably_math` `802`, `Seg` `707`, `extract_block_math` `818`, `render_block_math` `832`) are independent: the math splitter is a pure `&str → Vec<Seg>` with zero `Walker` coupling, and the table emitter (`emit_table`/`table_row`/`pad_cell` `562-683`) is another self-contained unit. 230 LoC of the file is tests.

### F5 — `components/overlay.rs`: a paint dispatcher + 9 inlined card painters (720 LoC)
`render` dispatch (`overlay.rs:31-54`) fans out to `render_effort_slider` (63, **104 LoC** — the largest, with its own column-grid math), `render_effects_demo`+`render_demo_legend` (174-219), `render_picker` (246), `render_ask_user` (328), `render_help` (435), `render_status` (487), `render_cost` (540), `render_verbose` (568), `render_btw` (598). Shared chrome (`centered` `223`, `titled_block` `232`) is the only common surface. The slider painter especially has nothing to do with the info-overlay painters.

### F6 — FE→BE coupling: the UI plane constructs **wire frames** and threads the bridge `Sender` through key handling
`main.rs` builds `UiToCore::*` protocol structs in ≥36 places (verified grep) — e.g. `dispatch_slash` (`main.rs:1604`), `apply_picker` (`1260` `SwitchLlm{n}`), `apply_effort`/`effort_command_frame` (`1804`/`1825-1827`), `dashboard_new_session` (`1429-1435` re-parses a `/`-line into `Command{name,args}` — **duplicating** `commands::split_command`). The whole `cockpit_*`/`dashboard_*`/`apply_*` family takes `&Sender<(u64,BridgeEvent)>` as a parameter, so the **transport** leaks into every key handler. The render plane also reaches into protocol types: `app/mod.rs:16` imports `bridge::protocol::{AskUserOption,CoreToUi,LlmItem}` directly into the state module. Root cause: no "intent" type between UI and bridge — the UI speaks raw protocol.

### F7 — State mutated from the render pass (`&mut AppState` in `render`)
`components::render(frame, &mut app, …)` (`components/mod.rs:48`) mutates state mid-draw: `app.set_term_size` (`51`) and `app.sync_transcript` (`136`, which rewidths the wrap cache + resizes the viewport, `app/mod.rs:1062-1064`). Render is supposed to be `y = f(x)`; here the frame pass writes back into `x`. This blocks splitting render into a pure FE module and is a principle 11 (reduce side-effects) + principle 2 (local reasoning) violation — you cannot reason about `render` without knowing it reflows the cache.

### F8 — Duplicated reducer + duplicated command parse (principle 5/12)
(a) `AppState::apply_frame` (`app/mod.rs:728`) and `Session::apply_frame` (`app/session.rs:217`) are two hand-maintained folds of the same `CoreToUi` protocol; a new frame variant must be added in both. (b) `dashboard_new_session` (`main.rs:1427-1435`) re-implements the `/name args` split that `commands::registry::split_command` (`registry.rs:110`) already does. (c) The Ctrl+C/Ctrl+Q quit chord is re-listed in **four** key handlers: `handle_key_event` (`main.rs:566`), `handle_workflows_key` (`772`), `handle_dashboard_key` (`898`), `handle_overlay_key` (`1015`).

### F9 — Redundant-comment hotspots (principle 9: "如果一段代码需要大段注释才能读懂，说明代码本身该重写")
The codebase is heavily over-commented; comments restate the obvious or carry spec-prose that belongs in a design doc, not inline. Worst offenders: `main.rs` chord block (`539-668`) — every `KeyCode` arm has a 2-6 line comment; `app/mod.rs:177-192` `Block` doc (16 lines for a 6-field struct); `main.rs:1166-1198` `handle_overlay_key` tail (a 9-line comment per overlay arm); the `--dump-frame` seed (`main.rs:160-285`) carries ~80 lines of prose comments. Net: comments are ~25-30% of `main.rs`. Principle 10 (视觉均匀/极简) and 9 both flag this.

---

## Competitor patterns (CC / Codex / v2 / v3, with file cites)

### C0 — Codex `AGENTS.md` states the EXACT rule tui_v4 breaks (verbatim from source)
> "Prefer adding new modules instead of growing existing ones. **Target Rust modules under 500 LoC, excluding tests. If a file exceeds roughly 800 LoC, add new functionality in a new module** instead of extending the existing file unless there is a strong documented reason not to. This rule applies especially to high-touch files such as `codex-rs/tui/src/app.rs`, `codex-rs/tui/src/bottom_pane/chat_composer.rs`, `codex-rs/tui/src/bottom_pane/footer.rs`, `codex-rs/tui/src/chatwidget.rs`, and `codex-rs/tui/src/bottom_pane/mod.rs`."

The five files Codex explicitly polices map 1:1 onto tui_v4's five god-files. Codex actively splits them (PR #21866 "Split ChatWidget state into focused modules"). Source: [openai/codex AGENTS.md](https://github.com/openai/codex/blob/main/codex-rs/AGENTS.md), [PR #21866](https://github.com/openai/codex/pull/21866).

### C1 — Codex layering = the target topology
- **`app.rs`** — top-level coordinator only: TUI render dispatch, input routing, the event loop, multiplexing the two streams via `tokio::select!`. *No* widget bodies, *no* command bodies.
- **`app_event.rs`** — the `AppEvent` enum: "an internal message bus to decouple widgets from the top-level App. Widgets emit events to request actions that must be handled at the app layer (opening pickers, persisting configuration, shutting down the agent)." This is the missing layer that severs F1/F6.
- **`chatwidget/`** — a **directory** of focused submodules: `slash_dispatch.rs` (command→`AppEvent`), `protocol.rs` ("app-server notification dispatch moved into chatwidget/protocol.rs"), `status_surfaces.rs` (status-line + terminal-title + git-branch lookup), plus split state for "input queues, turn lifecycle, transcript bookkeeping, status state."
- **`bottom_pane/mod.rs`** — routes keys to **either** the `ChatComposer` **or** the active popup via a `view_stack`: "When overlays need to be displayed, they are pushed onto view_stack. The composer remains in memory but is not rendered. When the overlay is dismissed, it's popped." This is exactly tui_v4's `Overlay`/modal fork (`main.rs:426-437`), but Codex owns it in the input layer, not the binary root.
- **`slash_command.rs`** — the `SlashCommand` enum + `supports_inline_args()`; dispatch arms live in `chatwidget/slash_dispatch.rs` and "call a new `AppEvent::…`". tui_v4 already has the registry half (`commands/registry.rs`) — it is missing the `slash_dispatch` half (it lives inline in `main.rs:1591-1938`).
- **`history_cell.rs`** — a **trait** per transcript-entry kind (user / agent / tool). tui_v4 hard-codes role→gutter in one `match` (`components/mod.rs:376` `gutter_for`); a trait is the principle-3 (composable) form.

Sources: [DeepWiki TUI](https://deepwiki.com/openai/codex/4-user-interfaces), [DeepWiki slash commands](https://deepwiki.com/openai/codex/4.1.3-slash-commands-and-features), [slash_dispatch.rs](https://github.com/openai/codex/blob/rust-v0.128.0/codex-rs/tui/src/chatwidget/slash_dispatch.rs).

### C2 — Codex `bottom_pane/` = how to split tui_v4's `components/mod.rs`
Codex's bottom input area is a whole directory (`chat_composer.rs`, `chat_composer_history.rs`, `textarea.rs`, `footer.rs`, `slash_commands.rs`, `feedback_view.rs`). tui_v4 crams the composer + footer + hints + dropdown into `components/mod.rs`. The `footer.rs` split is named in C0's rule directly.

### C3 — v2/v3 are the cautionary tale (god-file at 5–6k LoC)
`frontends/tui_v3.py` is **5573 LoC** and `frontends/tuiapp_v2.py` is **5812 LoC** — single Python files holding i18n, clipboard, CJK-wrap monkeypatch, `Block`/`FoldSegment`, markdown render, the `AgentBridge`, and every event class (`tui_v3.py:632-1351`+). v4's Rust split is already far better, but main.rs/app/mod.rs at 2.5k/2.1k are trending back toward the v2/v3 god-file. The lesson: a TUI's natural gravity is one mega-file; only an enforced split (C0) resists it.

### C4 — CC (Claude Code) / Codex separate the protocol reducer from view state
Codex moved "app-server notification dispatch into `chatwidget/protocol.rs`" — the EventMsg→UI fold is its own module, distinct from the widget. tui_v4's reducer (`apply_frame`) is buried in the same `impl AppState` as scrolling + overlay construction (F2), and is duplicated (F8). The fix is a `app/reducer.rs` that both `AppState` and `Session` call (one fold).

---

## Fix design (Rust sketches: actual signatures / changed lines)

The spine of the refactor is **C1's `AppEvent` bus** — it is what makes every other split mechanical. Introduce it FIRST; it is behavior-preserving (the loop performs the same effects, just one indirection later).

### Fix A — Introduce `app_event.rs` (the missing indirection; severs F1 + F6)
New file `src/app_event.rs`. The UI plane stops speaking raw `UiToCore` and stops holding the bridge `Sender`:
```rust
// src/app_event.rs
pub enum AppEvent {
    /// Send a wire frame to the ACTIVE session (the only bridge verb the UI emits).
    ToActive(crate::bridge::protocol::UiToCore),
    /// Send to a specific session (dashboard quick-reply, background work).
    ToSession(u64, crate::bridge::protocol::UiToCore),
    OpenWorkflows, OpenDashboard, CloseView,
    Copy { text: String, label: &'static str },
    SetMouseCapture(bool),
    Quit,
}
```
The event loop drains a per-frame `Vec<AppEvent>` (or an `mpsc`) AFTER `handle_term_event`, and is the ONE place `tx_bridge` lives:
```rust
// main.rs event_loop, replacing inline send_active(... tx_bridge ...)
for ev in app.drain_actions() {            // app accumulates intents
    match ev {
        AppEvent::ToActive(f)      => { app.sessions.send_to(app.sessions.active, f, tx_bridge); }
        AppEvent::ToSession(id, f) => { app.sessions.send_to(id, f, tx_bridge); }
        AppEvent::Copy { text, label } => perform_copy(app, &text, label),
        AppEvent::Quit => app.should_quit = true,
        // …
    }
}
```
Then every `dispatch_*`/`cockpit_*`/`apply_*` fn drops its `tx_bridge` param and its signature becomes `fn(&mut AppState, …)`, pushing intents via `app.emit(AppEvent::ToActive(UiToCore::Submit{…}))`. This is exactly Codex's "widgets emit `AppEvent`s; the app layer performs them." Net: key handlers no longer close over the transport, so they can move to other files freely.

### Fix B — Split `main.rs` → `app_event.rs` + `input/` (keys) + `commands/dispatch.rs`
After Fix A, the move is mechanical. Target files + public surface:

| New file | Moves OUT of main.rs | pub surface |
|---|---|---|
| `src/app_event.rs` | (new) | `enum AppEvent` |
| `src/input/keymap.rs` | `handle_key_event` (521-746), `try_complete_dropdown` (1481), `composer_width` (514), the global-chords block | `pub fn cockpit_key(&mut AppState, KeyEvent, now_ms) -> Vec<AppEvent>` |
| `src/input/views.rs` | `handle_dashboard_key` (853), `handle_workflows_key` (765), `handle_overlay_key` (1007), `route_view_key` (751), `fire_workflow_action` (818) | `pub fn route_view_key(...) -> Vec<AppEvent>` |
| `src/input/mouse.rs` | `handle_mouse_event` (450) | `pub fn mouse(&mut AppState, MouseEvent)` |
| `src/commands/dispatch.rs` | `dispatch_slash` (1591), `open_ui_command` (1627), `app_command` (1844), `dispatch_action` (1553), `apply_picker`/`apply_ask_user`/`apply_scheduler`/`apply_effort` (1249-1817), `*_picker_items` (1942/1974), `new_ask_id` (1832), `effort_command_frame` (1825) | `pub fn dispatch_slash(&mut AppState, line) -> Vec<AppEvent>` |
| `src/term.rs` | `setup_terminal`/`restore_terminal`/`install_panic_hook`/`set_mouse_capture` (2203/2269-2304) | `pub fn setup()/restore()/install_panic_hook()` |
| `src/clipboard.rs` | `copy_text` (2166), `notice_copy` (2178), `read_clipboard` (2213), `export_action` (2001) | `pub fn copy(&mut AppState, text, label)` |

`main.rs` shrinks to: `main()` + arg parse + `run_app` + `event_loop` + the 3 thread spawners + the `--smoke`/`--dump-frame` harness (~450 LoC). `rewind_real_turns_from` (1378) moves to `app/reducer.rs` (it's pure state logic).

Execution order (keeps `cargo build` green): **(1)** add `app_event.rs` + an `actions: Vec<AppEvent>` field on `AppState` + `emit`/`drain_actions`, wire the loop to drain it while the old inline sends still work (no fn signatures change yet → compiles). **(2)** convert `dispatch_*` to push intents instead of `send_active`, delete the `tx_bridge` params one fn at a time (each still in main.rs → compiles after each). **(3)** `git mv`-style cut the now-transport-free fns into the new files, `pub(crate)` them, fix imports. Build after each file.

### Fix C — Split `app/mod.rs` → reducer / overlay / session-glue / types
| New file | Moves | pub surface |
|---|---|---|
| `src/app/reducer.rs` | `apply_bridge_event` (684), `apply_frame` (728), `block_for_mid_mut` (903), `rewind_real_turns_from` | `pub fn apply_frame(state: &mut impl FrameSink, CoreToUi, now_ms)` shared by AppState **and** Session (kills F8a) |
| `src/app/overlay_ops.rs` | `apply_llm_list` (916), `open_picker` (936), `open_ask_user` (963), `surface_next_ask` (979), `open_overlay`/`close_overlay` (995-1003), `apply_btw_answer` (947) | `impl AppState` methods (same names) |
| `src/app/types.rs` | `Overlay` (37-97), `Role` (153), `Block` (182-275), `ConnStatus` (101), `CostBreakdown` (303), `PendingAsk` (293), `RenameState` (146), `View` (131) | re-exported from `app::` |
| `src/app/multi.rs` | `snapshot_active_into_map` (1209), `load_active_fields_*` (1238-1319), `switch_session` (1268), `apply_tagged_event` (1417), dashboard open/close (1324-1342) | `impl AppState` methods |
| `app/mod.rs` keeps | `struct AppState` + `Default` + `new` + tick + render-plane sync (`sync_transcript`/`scroll_*`) + the `actions` queue | the struct + sync |

The reducer-sharing trait removes the duplicate fold:
```rust
// app/reducer.rs — one fold, two callers (kills F8a)
pub trait FrameSink {
    fn push_block(&mut self, role: Role, mid: Option<String>, src: String, final_: bool);
    fn block_for_mid_mut(&mut self, mid: &str) -> Option<&mut Block>;
    fn set_busy(&mut self, b: bool, now_ms: u64);
    // … the ~6 mutations apply_frame needs
}
pub fn apply_frame<S: FrameSink>(s: &mut S, frame: CoreToUi, now_ms: u64) { /* the body once */ }
```

### Fix D — Split `components/mod.rs` → `components/cockpit/` (mirror Codex `bottom_pane/`)
| New file | Moves | pub surface |
|---|---|---|
| `components/cockpit/mod.rs` | `render`/`render_cockpit` layout split (48-162) | `pub fn render(frame, &mut AppState, theme, now_ms)` |
| `components/cockpit/header.rs` | `render_header` (167) | `pub(crate) fn render_header(...)` |
| `components/cockpit/transcript.rs` | `render_transcript` (266), `user_band_line` (357), `gutter_for` (376) | `pub(crate) fn render_transcript(...)` |
| `components/cockpit/composer.rs` | `render_composer` (442), `composer_lines` (494), `split_at_col` (562) | `pub(crate) fn render_composer(...)` |
| `components/cockpit/footer.rs` | `render_footer` (591), `mode_indicator` (676), `ctx_bar` (687), `render_hints` (196), `render_separator` (252), `render_spinner` (387) | per-fn `pub(crate)` |
| `components/cockpit/dropdown.rs` | `render_dropdown` (716), `render_palette` (731), `render_file_picker` (771), `dropdown_height`/`composer_height` (696/807) | `pub(crate) fn render_dropdown(...)` |
| `components/text.rs` | `compact_cwd`, `truncate_model`+`MODEL_LABEL_CAP`, `human_count`, `clip_to`, `fx_command_active`, `ctrl_key_label` | pure helpers (already `pub`) |

Address **F7** in the same slice: make render pure by hoisting the two mutations to the loop. Before `terminal.draw`, the loop calls `app.prepare_frame(area)` (does `set_term_size` + `sync_transcript`); `render` then takes `&AppState` (immutable). Sketch:
```rust
// event_loop, before draw:
let area = terminal.size()? ; // or last known
app.prepare_frame(area, &theme);                 // the ONLY state write for the frame
terminal.draw(|f| components::cockpit::render(f, &app, &theme, now_ms))?;  // &app, not &mut
```

### Fix E — Split `markdown/render.rs` → `walker.rs` + `inline_math.rs` + `table.rs`
| New file | Moves | pub surface |
|---|---|---|
| `markdown/render.rs` keeps | `render_markdown` (45) entry + the `Walker` event loop core (`event`/`start`/`end`/`flush_line`) | `pub fn render_markdown` |
| `markdown/inline_math.rs` | `Seg` (707), `split_inline_math` (721), `find_math_close` (775), `is_probably_math` (802), `extract_block_math` (818), `render_block_math` (832) | `pub fn split_inline_math(&str)->Vec<Seg>`, `pub fn extract_block_math` |
| `markdown/table.rs` | `TableBuf` (98), `emit_table` (562), `table_row` (623), `pad_cell` (667) | `pub(crate) fn emit_table(theme, &TableBuf)->Vec<Line>` |
| `markdown/code.rs` | `CodeBuf` (93), `emit_code_block` (535), `heading_style` (687), `bullet` (699) | `pub(crate) fn emit_code_block(...)` |

The math splitter is already a pure free fn with zero `Walker` state — moving it is a cut + `use`. Tests for math (`render.rs:1006-1056`) move with it.

### Fix F — Split `components/overlay.rs` → `overlay/mod.rs` (dispatch + chrome) + per-card files
`overlay/mod.rs` keeps the `render` match (31-54) + `centered` (223) + `titled_block` (232). Move `render_effort_slider` (63) → `overlay/effort.rs`; `render_picker`/`render_ask_user` → `overlay/picker.rs`; `render_help`/`render_status`/`render_cost`/`render_verbose`/`render_btw` → `overlay/info.rs`; `render_effects_demo`/`render_demo_legend` → `overlay/effects.rs`. Each `pub(crate) fn render_<x>(frame, area, …)`.

### Fix G — Dedup the quit chord (F8c) + the command-line parse (F8b)
Add `input/keymap.rs::is_quit_chord(KeyEvent) -> bool` and call it once at the top of `route_view_key` (returns `vec![AppEvent::Quit]`), deleting the four copies (`main.rs:566,772,898,1015`). In `dashboard_new_session`, replace the inline parse (`main.rs:1427-1435`) with `commands::registry::split_command(&expanded)` → one `AppEvent::ToSession`.

### Fix H — Strip redundant comments (F9, principle 9/10)
In the moved files, delete comments that restate the code. Keep only: the module-header `//!` (one short paragraph), non-obvious invariants (e.g. the `block_for_mid_mut` "newest first" rationale, the `is_probably_math` currency heuristic). Target: comment density in the new `input/`/`commands/` files ≤ 10% (from ~28%). The `--dump-frame` seed prose (`main.rs:160-285`) collapses to one line per scenario.

---

## Review-principle violations (cite principle # + file:line)

- **P1 模块边界清晰** — `main.rs` is 8 modules in one (F1); `components/mod.rs:48` does layout+8 widgets (F3); `app/mod.rs` mixes reducer+overlay+multi-session (F2). Circular-ish: `app/mod.rs:18` imports `components::picker`, `components/mod.rs:34` imports `app::AppState` — the FE/BE ring isn't broken by an interface.
- **P2 局部可推理** — `components::render` (`components/mod.rs:48`) silently reflows the wrap cache (F7); you can't reason about a draw without reading `sync_transcript` (`app/mod.rs:1049`). `handle_key_event` (`main.rs:521`) can't be understood without the 4 dispatch fns 1000 lines below.
- **P3 可组合** — role→gutter is a closed `match` (`components/mod.rs:376`) not a `HistoryCell` trait (Codex C1); adding a block kind edits the match (vs. add an impl).
- **P4 变化半径小** — a new `CoreToUi` variant must be edited in TWO folds (`app/mod.rs:728` + `app/session.rs:217`, F8a); a new slash command touches `registry.rs` + the inline dispatch in `main.rs:1627` far away.
- **P5 复杂度线性 / P12 功能越多代码越短** — the dispatch is an `if/else` forest (`main.rs:1591-1938`) that grows per-command; the quit chord is copy-pasted ×4 (F8c). No shared abstraction → each feature adds lines instead of reusing structure.
- **P6 约束写进类型** — the "UI must not speak wire protocol" invariant is unenforced: `UiToCore` is constructed in the render-root (`main.rs`, 36 sites, F6). An `AppEvent` type (Fix A) makes it a compile constraint.
- **P9 注释极简** — F9: `main.rs:539-668`, `:1166-1198`, `:160-285`; `app/mod.rs:177-192`. Dense spec-prose where naming should carry intent.
- **P10 视觉均匀/极简** — the 2505/2078-line files create jagged scroll; the `--dump-frame` seed (`main.rs:155-302`) is 80% comment.
- **P11 减少副作用** — render mutates state (F7, `components/mod.rs:51,136`).
- **P13 为未来接入性设计** — there is no clean "send an intent" entry point; everything is wired through `send_active(app, tx_bridge, …)` (`main.rs:1582`). Fix A gives the clean入口 Codex's `AppEvent` provides.

---

## Open questions / risks

1. **`AppEvent` delivery: per-frame `Vec` vs `mpsc`?** Codex uses an `mpsc` (`app_event.rs` + `AppEventSender`). A per-frame `Vec<AppEvent>` drained in the loop is simpler and keeps everything single-threaded (no new channel), but a handler can't emit an event that must outlive the frame. Recommend the `Vec` (simpler, matches the immediate-mode loop); revisit if a future fix needs deferred/async intents.
2. **`FrameSink` trait vs. macro for the shared reducer (Fix C).** A trait is clean but `Session` and `AppState` have different field layouts; the trait needs ~6-8 methods. If that's heavier than the duplication, a `macro_rules!` fold is the fallback. Measure both at the slice.
3. **Test stability.** ~130 tests live INSIDE the god-files (`app/mod.rs:1466-2078` alone is ~600 LoC of tests; `main.rs:2306-2505`). When code moves files, its `#[cfg(test)] mod tests` must move with it AND keep `use super::*` resolving. Move tests in the SAME commit as their code; run `cargo test` per slice. Pure-fn tests (markdown/registry) move trivially; the integration tests in `main.rs` (`right_click_returns_from_dashboard` 2467) must follow `mouse.rs`.
4. **`pub` surface creep.** Cutting fns into sibling files tends to over-`pub` them. Enforce `pub(crate)` for everything except the genuine entry points (`render_markdown`, `components::cockpit::render`, `AppEvent`, the registry fns). A `cargo build` warning pass (`unused`/`unreachable_pub` if enabled) after each slice catches leaks.
5. **Effort/ordering interaction with the 12 feature fixes.** This refactor must land as its own PR(s) BEFORE the feature slices, else the feature diffs are unreviewable. Sequence: A→B→C (severs main.rs), then D (render purity, unblocks FE work), then E/F/G/H in any order (independent). G/H can be folded into B/C/D/E/F's slices to avoid touching files twice.
6. **Codex source not directly fetchable here** (github.com/deepwiki.com blocked for WebFetch). Patterns/quotes above are from WebSearch over the indexed source + DeepWiki + AGENTS.md; file paths (`app_event.rs`, `chatwidget/slash_dispatch.rs`, `bottom_pane/mod.rs`, `slash_command.rs`) are corroborated across multiple results but the implementer should open them directly to copy exact signatures before mirroring.
