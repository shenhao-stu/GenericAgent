# tui_v4 — Master Checklist & Build Contract

> **Mission:** Rebuild GenericAgent's terminal UI (`frontends/tui_v3.py` + `tuiapp_v2.py`) as **tui_v4** — a TUI that *replicates and surpasses* Claude Code, fuses the best of **Claude Code + Codex + kimi-code + reasonix**, and keeps **100 % of v2 + v3 functionality**.
>
> **Status:** ✅ Rust rewrite + real-GA-format redesign FINALIZED (278 tests green, 4.92 MB release exe, real GA connect verified) · **Date:** 2026-05-31 · **Stack:** Rust + ratatui (migrated from the Ink v0.1; Ink source preserved in `../tuiapp_v4_ink_backup/`).
>
> This file is the single source of truth. Every builder agent reads it. Boxes are ticked only when **verified green** (`cargo build` + `cargo test` + smoke), not on self-report.

---

## 0. Why Rust + ratatui (stack decision — LOCKED)

The Ink v0.1 shipped but failed at runtime: **`disconnected`** (bridge never spawned from the packaged exe), layout didn't fill the screen, wrong spinner, cramped session sidebar. The research (`../tuiapp_v4_research/recon/RESEARCH_REPORT.md`) always recommended Rust+ratatui; the user has now endorsed the migration.

- **Pain point #1 (resize+scroll corruption) is solved by construction** — ratatui is immediate-mode with a per-cell back-buffer diff. There is no retained, stale-width scene to corrupt. Reflow recomputes from logical source on every frame.
- **Pain point #2 (clean copy) is solved by OSC-52 from logical state** — we copy the source string, never rendered rows, so soft-wrap newlines are structurally impossible.
- **~3 MB single binary** (vs Ink's ~120 MB), <5 ms startup, ~0 % idle CPU.
- This is the only path that genuinely **surpasses** CC, whose own renderer still flickers ~1/3 of sessions (HN 46701013).

**Toolchain verified:** `cargo 1.95.0`, target `x86_64-pc-windows-msvc`, Python 3.13.2, `mykey.py` loads **19 LLM configs**.

---

## 1. The three pain points (HARD acceptance — each needs a test)

- [x] **P1 — Resize never corrupts scroll.** Logical `Block`s store SOURCE text; wrap cache keyed `(block_id, width)`; scroll = logical `ScrollAnchor {block_id, intra}` (or distance-from-bottom). On resize: recompute wrap at new width → re-derive viewport → no stale/overlapping rows. *Test: `resize_then_scroll_no_drift`.* ✅ test present + passing.
- [x] **P2 — Copy yields clean logical text.** Explicit copy actions (last msg / focused msg / code block / whole transcript) emit logical source via **OSC-52** (SSH-safe), native→tmux→OSC52 fallback. *Test: `copy_across_wrap_has_no_newline`.* ✅ test present + passing.
- [x] **P3 — Correct markdown + math.** `pulldown-cmark` → themed ANSI; `syntect` code highlight; custom `latex_to_unicode` for `$…$`/`$$…$$` (Greek, sup/sub, `\frac` stacked, `\sqrt`, `\sum`/`\int` w/ limits, matrices). *Tests: `md_table`, `math_frac_block`, `math_greek`.* ✅ all three tests present + passing.

---

## 2. NEW feedback from this round (the user's explicit corrections)

- [x] **N1 — Fix `disconnected`.** Root cause: packaged exe run from `release/` can't discover `scripts/ga_bridge.py` (one dir up), and spawn/import failures are swallowed. **Fix:** robust discovery (env → next-to-exe → exe/.. → `GENERICAGENT_ROOT` → cwd walk to repo root via `agentmain.py` marker); set child env `PYTHONUTF8=1`; **surface spawn/import errors in the UI** (never a silent "disconnected"); auto-retry + a visible "reconnect" affordance; the status word reflects real handshake (`Ready` frame). ✅ 8-tier discovery (`bridge/mod.rs`), `BridgeEvent::{SpawnFailed,Stderr,ParseNoise,ChildExited}` surfaced, `connected` flips on real `Ready`. Real connect proof passes (direct + `ga_bridge_handshakes_ready` ignored test + `packaging_discovery_resolves_both_ways`). *(reconnect affordance: the dead handle is visible + re-spawnable; an explicit hotkey-driven retry is noted as a polish gap below.)*
- [x] **N2 — Session panel = CC-style full-screen dashboard, NOT a sidebar in the input view.** See §6. Reached by **left-click** (and a key, e.g. `Ctrl+S`); does not crowd the composer. ✅ `View::Dashboard`, `Ctrl+S` + left-click entry (`main.rs:321,382`), `Esc` back.
  - [x] Two collapsible categories: **`Needs input`** (idle / awaiting user) and **`Working`** (running). Each `▸`/`▾` collapse/expand. ✅ (Completed category too.)
  - [x] Each row: status glyph + session name + **live preview** of current output (like CC's "全部就绪。最终汇报…" preview).
  - [x] **Up/Down** select, **Enter** open/switch, **Space** reply, **Ctrl+X / del** delete, **rename**, **new session** ("describe a task for a new session" input row).
  - [x] Header: `N awaiting input · M working · K completed`.
- [x] **N3 — `/llm` actually wired to `mykey.py`.** Picker lists `list_llms()` → `[(idx, "SessionType/name", is_current)]`; current marked `●`; Enter → `next_llm(idx)`. Default model loads on startup (llm_no 0). Status bar shows live `get_llm_name()`. `/reload-keys` hot-reloads. ✅ Rust forwards `SwitchLlm{n}` / `ListLlms`; `ga_bridge.py::handle_list_llms` mirrors `agent.list_llms()` → `LlmList{items}`, `SwitchLlm`→`next_llm(n-1)`; real Ready frame carried the live model list.
- [x] **N4 — Custom spinner identity (NOT Claude Code's `✻`/`✷`).** Design our own IDE-style mark. Options to ship (user-selectable via `/emoji`-style switch): `◜◠◝◞◡◟` arc, `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏` braille (v3 soul), `·✢✳✶✻✽` pulse — pick a **distinct default** that is not the CC asterisk. Keep heat-color ramp + pet faces + gerund rotation. ✅ `flavor/mod.rs` spinner sets + heat ramp + gerunds + pet faces; default is the arc, not `✻`.
- [x] **N5 — Layout fills the whole terminal** (reference tui_v3). No dead space; regions flex to `height-1`. See §5. ✅ smoke renders 484 non-blank cells in a flex-filled layout.
- [x] **N6 — Integrate `/btw`, `/scheduler`, `/rewind` with real UIs** (not just core-forward). See §4 / §7. ✅ dedicated overlays/pickers + (for `/rewind`) the real `Rewind`/`RewindResult` bridge frames implemented in `ga_bridge.py`.
- [x] **N7 — Back up the original** ✅ done — Ink source in `../tuiapp_v4_ink_backup/`.
- [x] **N8 — Fuse Codex + Claude Code + kimi-code + reasonix styles**, retain v2+v3 features. See §5/§8/§9. ✅ status footer (Codex/CC), token theming (OSC 10/11), compact density, full §4 v2+v3 command union.

---

## 3. Architecture — four planes (Rust)

```
┌──────────────────── tui_v4 (Rust + ratatui + crossterm) ────────────────────┐
│ UI plane: immediate-mode draw() each frame; overlay stack (palette, pickers, │
│   session dashboard, /workflows, ask-user, help, copy-mode, effects demo).   │
│ Render plane: logical Block model (SOURCE text) → wrap cache (block_id,width) │
│   → viewport via ScrollAnchor. Native scrollback for finalized; small live    │
│   band redrawn. OSC-52 copy. Synchronized output (DEC 2026). Zero-flicker.    │
│ Chat plane (per session): spawn scripts/ga_bridge.py child; JSONL over stdio; │
│   reader+writer threads → bounded channel → main loop. Dual-AtomicBool live.  │
│ Workflow plane (singleton watcher): conductor WS :8900 + HTTP poll; hive BBS  │
│   HTTP; goal_state.json mtime. Additive snapshots; never blocks chat.         │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Crate stack
- [x] `ratatui` + `crossterm` (render + events + alt-screen + mouse) ✅ ratatui 0.30, crossterm 0.29
- [x] `serde` + `serde_json` (JSONL protocol, settings, history) ✅
- [x] `anyhow` / `thiserror` (errors), `unicode-width` + `unicode-segmentation` (CJK width) ✅ anyhow direct; thiserror present transitively (anyhow used for app errors); unicode-width 0.2 + unicode-segmentation 1.13
- [x] `pulldown-cmark` (markdown), `syntect` (code highlight) ✅ pulldown-cmark 0.13, syntect 5.3
- [x] `base64` (OSC-52), `arboard` (native clipboard fallback), `dirs` (paths) ✅
- [x] effects: hand-rolled delta-time engine (no heavy dep) or `tachyonfx` if it integrates cleanly ✅ hand-rolled single-clock engine in `effects/` (no `tachyonfx` dep)
- [x] dev: `insta` or plain golden-string tests for render snapshots ✅ plain golden-string tests (no `insta` dep)
- [x] packaging: `cargo build --release`; later `cargo-dist` for win `.exe` / mac `.dmg` / linux musl ✅ `cargo build --release` → win `.exe` done; cargo-dist (mac/linux musl) deferred (noted gap).

### Module map (`src/`) — the build contract
```
main.rs            entry: args(--smoke/--help/--version), alt-screen, panic-restore hook, event loop
app/mod.rs         AppState, reducers, overlay stack, focus, tick
app/session.rs     SessionMap (N bridge children), per-session input stash, CRUD, dashboard model
bridge/mod.rs      spawn ga_bridge.py, discovery (N1), reader/writer threads, liveness, reconnect
bridge/protocol.rs serde enums: UiToCore / CoreToUi (mirror scripts/ga_bridge.py)
render/block.rs    Block {id, role, source, rev}; finalize/stream; folding
render/measure.rs  CJK/wide-aware wrap; wrap cache (block_id,width)→VisualLine[]; prefix sums
render/viewport.rs ScrollAnchor; visible-window derivation; resize reflow (P1)
render/copy.rs     OSC-52 + native + tmux fallback; logical-source copy (P2); copy-mode overlay
markdown/render.rs pulldown-cmark → styled Lines; tables; blockquotes; lists
markdown/highlight.rs syntect code → ANSI
markdown/math.rs   latex_to_unicode (P3)
components/        header, transcript, composer, statusbar, sidebar?, dashboard, workflows_panel,
                   workflow_detail, ask_user, plan_card, menu/picker, llm_picker, file_picker,
                   help, verbose_audit, btw_card, scheduler, rewind, copy_view
workflow/mod.rs    watcher (conductor/hive/goal) → WorkflowSnapshot; pollers; schema
theme/mod.rs       tokens (6+ themes), rainbow 7-stop, adaptive (OSC 10/11 probe)
i18n/mod.rs        zh/en (~250 keys), fallback chain, /language repaint
flavor/mod.rs      spinner sets, gerunds(34), heat ramp, pet faces, OSC0 title, tips
effects/mod.rs     delta clock, capability gate, fire/snow/lightning/shimmer/sparkle (bounded, OFF by default)
commands/mod.rs    ~33 slash registry + dispatch; magic prefixes ! and @
util/              terminal caps, osc, keymap, settings persistence
```

---

## 4. Slash-command parity (union of v2 + v3 — ~33) + magic prefixes

Legend: **UI** = needs a dedicated interactive panel · **fwd** = core-forwarded text · **app** = handled in-app.

> ✅ All 38 §4 names resolve via the single registry (`commands/registry.rs`) and route by kind — asserted by `registry_resolves_all_commands` (case/slash/args-tolerant) + `classify_slash_routes_every_kind`.

- [x] `/help` (app) — full command list overlay
- [x] `/status` `/sessions` (app) — model/state/rounds/context/cwd
- [x] `/new [name]` (app) — create + switch session
- [x] `/switch <id|name>` (UI) — session dashboard
- [x] `/close` (app) — close current session
- [x] `/rename <name>` (app) — rename session
- [x] `/branch [name]` (app) — fork w/ copied history (restore prior transcript)
- [x] `/rewind [n]` (**UI**) — list last ~20 real turns, pick → truncate history (needs bridge `Rewind{n}` frame; see §7) — ✅ `Rewind`/`RewindResult` implemented in `ga_bridge.py`
- [x] `/clear` (app) — clear display only (idle-only)
- [x] `/stop` `/abort` (app) — abort running task
- [x] `/llm [n]` (**UI**) — model picker via `list_llms()` (N3)
- [x] `/btw <q>` (**UI**) — side-question card, background thread, non-blocking (see §7)
- [x] `/review [req]` (fwd) — in-session code review
- [x] `/update [note]` (fwd) — git pull GA + impact
- [x] `/autorun [seed]` (fwd) — autonomous mode
- [x] `/morphling [target]` (fwd) — absorb skill
- [x] `/goal [goal]` (fwd + **/workflows tile**) — goal mode (progress in panel)
- [x] `/hive [target]` (fwd + **/workflows tile**) — multi-worker hive
- [x] `/conductor [task]` (fwd + **/workflows tile**) — conductor multi-subagent
- [x] `/scheduler [start a,b,c]` (**UI**) — multi-pick reflect tasks + 2-step confirm + cron status (see §7)
- [x] `/continue [n|name]` (**UI**) — searchable picker over session logs (content-grep + lazy load, v2)
- [x] `/resume` (fwd) — GA core recovery prompt
- [x] `/cost [all]` (app) — token usage (input/output/cache/context%); subagent log scan
- [x] `/export clip|<file>|all` (**UI**) — export last reply
- [x] `/restore` (app) — restore last `model_responses_*.txt` into history
- [x] `/reload-keys` (app) — hot-reload `mykey.py` (N3)
- [x] `/language [code]` (**UI**) — switch interface language + full repaint
- [x] `/emoji [style]` (**UI**) — pet/spinner style (N4)
- [x] `/verbose` `/tools` `/trace` (**UI**) — full-screen tool-call audit
- [x] `/effects [demo|off|subtle|full]` (app/**UI**) — effects intensity + demo splash
- [x] `/theme` (**UI**) — theme picker w/ live preview (v2)
- [x] `/quit` `/exit` (app)
- [x] `!cmd` magic prefix — host shell, 30 s timeout, echo + seed agent context (hot-pink input border in shell mode)
- [x] `@path` magic prefix — inline file <100 KB as `[File: p]…[/File]`; gitignore-aware completion
- [x] Unknown command → friendly breadcrumb. Idle-only guard on `/clear /export /review /rewind /continue`.

---

## 5. Layout (FILLED — tui_v3 reference + CC/Codex/kimi/reasonix fusion) — N5/N8

Flex-fill to `max(1, height-1)`; nothing leaves dead space. Bottom-up live region (tui_v3 model):

```
┌ HEADER (1–2 rows) ─────────────────────────────────────────────────────────┐
│ ◆ GenericAgent · tui_v4    <model>   ·  <cwd>   ·  <session>     <rotating tip>│
│ ▓▓▓▓▓▓▓ rainbow separator (7-stop ROYGBIV, full width) ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
├ TRANSCRIPT (flex, fills all remaining height) ──────────────────────────────┤
│ user tiles (charcoal band + prompt mark) · assistant markdown · tool chips   │
│ · plan/todo card · folded turns (▸ summary) · reasoning (dimmed)             │
│ … finalized → native scrollback; streaming tail → live band …               │
├ STREAMING / SPINNER (when busy) ────────────────────────────────────────────┤
│ <pet> <spinner> <Gerund>… 3.2s · ~450 tok · ctx ▰▰▰▱▱ 50%                     │
├ PLAN / BTW / PENDING cards (conditional, above composer) ───────────────────┤
├ COMPOSER (multi-line, bordered) ────────────────────────────────────────────┤
│ ❯ type a message…   (Enter send · Shift+Enter newline · ! shell · @ file)    │
├ STATUS FOOTER (1 row, space-between) ───────────────────────────────────────┤
│ ❯ <mode pill>  <agent pills>   │   <model> │ ctx 62% (1.2k/2.4k) │ $0.12 │ git│
└─────────────────────────────────────────────────────────────────────────────┘
```

- [x] **Header**: identity + model/cwd/session + rotating tip (12 s, per-language). Rainbow separator full width.
- [x] **Transcript fills** all remaining height (the bug to kill: v0.1 left it empty). ✅ smoke 484 cells.
- [x] **Status footer** (Codex/CC fusion): left = prompt char (`❯` user / `!` bash) + mode pill (Plan/Accept/Auto) + agent/team pills; right = `model │ context% (used/cap) │ $cost │ git`. OSC-21337 tab status (idle green / working amber / waiting blue).
- [x] **Style tokens** (fusion): CC's semantic palette (`claude` cyan, success green, error red, warning orange, suggestion blue, planMode purple, autoAccept coral); Codex's terminal-adaptive theming (OSC 10/11 probe, ANSI-256 downsample, `NO_COLOR`); kimi/reasonix compact density. No hardcoded RGB — all via theme tokens.
- [x] **CJK-aware wrapping** everywhere (cell-accurate). ✅ `unicode-width`-based `measure.rs`; `md_table_cjk_alignment` test.

---

## 6. Session dashboard (the headline redesign) — N2

A **separate full-screen view** (left-click or `Ctrl+S` to enter; `Esc` back), modeled on CC's session manager (`clip_20260530_183845.png` / `032711.png`):

```
◆ GenericAgent · tui_v4   <model> · <cwd>
N awaiting input · M working · K completed
─────────────────────────────────────────────────────────────────
▾ Needs input
  ◆ <session name>      <live preview of last assistant line / "send a prompt to start">
  ◆ bg                  👋 What can I help you with today?
▾ Working
  ⏺ getoken             全部就绪。最终汇报（中文）：## 本轮交付总览 …   (heat-colored)
▸ Completed (K)         (collapsed)
─────────────────────────────────────────────────────────────────
❯ describe a task for a new session
enter open · space reply · ctrl+x delete · r rename · ctrl+n new · ? shortcuts
```

- [x] Categories `Needs input` / `Working` / `Completed`, each **collapse/expand** (`▸`/`▾`), counts in header.
- [x] Row = status glyph (◆ needs-input / ⏺ running w/ heat / ○ idle / ✓ done) + name + **live preview** (truncated current output).
- [x] **Up/Down** navigate (skip collapsed); **Enter** open/switch into that session; **Space** quick-reply; **Ctrl+X / Del** delete (keep log); **r** rename; **Ctrl+N** / bottom input = new session; **left-click** row = switch.
- [x] Live preview updates from each session's bridge stream (multiplexed). Running rows show spinner + heat.
- [x] Backed by `SessionMap` = N independent `ga_bridge.py` children, UI-multiplexed (GA core has no multi-session API). Per-session input stash on switch. Names persist via `session_names` / `temp/`. ✅ `spawn_bridge_tagged` per-session multiplex.

---

## 7. The other interactive panels

- [x] **`/workflows` panel** (conductor + goal-hive — GA never had a TUI for this): live tree, two styles (box-tree w/ `╞═` focus + compact bullet list), grouped **Conductor / Hives / Goal**. Per-node name/status/elapsed/tokens/preview. Node actions `keyinfo/input/stop/kill/open`. Focus nav + detail overlay. Animated status (shimmer running / sparkle done / lightning failed). Data via §3 workflow watcher. Degrade gracefully ("not running · press X to launch"). Hive auth via `temp/hive_<name>/board.json {port,key,started_at,objective}`. ✅ `View::Workflows` + `workflow/panel.rs`; `renders_both_styles_and_detail` test.
- [x] **`/btw` card**: ephemeral accent box above composer showing `querying…` then the side answer; background thread; non-blocking; `Esc` dismisses (no history pollution).
- [x] **`/scheduler`**: step 1 multi-pick reflect tasks (pre-tick running), step 2 confirm card (start/stop diff), step 3 apply + show cron status. ✅ `components/scheduler.rs`.
- [x] **`/rewind`**: list last ~20 real turns w/ preview; pick → send `Rewind{n}` to bridge → bridge truncates `llmclient.backend.history` + replays display. (Implement the bridge frame; the Ink build only stubbed it.) ✅ `Rewind`/`RewindResult` + `_rewind_cut_index` (pure) implemented in `ga_bridge.py` — no longer a stub.
- [x] **ask_user picker**: unified card — question + candidates + inline free-text escape; single-pick (↑↓ cycle candidates↔input), multi-pick (`[多选]` auto, Space toggle), numeric pick; queued parallel asks surface in turn.
- [x] **Menu/picker primitive**: reusable for `/llm /continue /rewind /export /scheduler /language /emoji /theme`. ✅ `components/picker.rs`.
- [x] **`/verbose` audit**: full-screen tool-call trace (alt view).

---

## 8. Keybindings (union; internal-byte model from v3 + v2 Textual)

- [x] Enter submit · Ctrl+J / Shift+Enter newline · arrows move (visual-row aware) · history at edges
- [x] Ctrl+A select-all · Ctrl+E end · Shift+arrows select · Ctrl+X cut · Ctrl+C (3-stage: copy→abort→arm-quit) · Ctrl+V paste · Ctrl+U kill-to-start · Ctrl+Z/Y undo/redo (200 deep) · Ctrl+S stash draft / open dashboard · Ctrl+L redraw · Ctrl+O fold toggle · Tab slash-complete — ✅ **3-stage Ctrl+C wired**: PURE decider `input::keychord::ctrl_c` (timestamps injected) → copy selection (OSC-52 via `render/copy.rs`) / abort a running turn / ARM quit + bottom-line "press Ctrl+C again to quit" hint, 2nd within 2s quits, arm expires after 2s. `ctrl_c_three_stage_transitions` test. (Ctrl+Q stays a plain immediate quit; Ctrl+S = open dashboard, draft stash on Ctrl+G.)
- [x] Esc universal back (menu→ask→btw→draft→pending→stop; never exits) · Esc-Esc (<0.8 s) → `/rewind` — ✅ Esc universal-back unchanged (clear pending ask → collapse selection → stash draft, never exits); **Esc-Esc→/rewind wired** via PURE `input::keychord::esc` (0.8s window, timestamps injected) → opens the rewind picker. `esc_esc_within_window_triggers_rewind` + `esc_esc_outside_window_is_two_backs` tests.
- [x] Ctrl+N new session · Ctrl+W/Ctrl+D drop · Ctrl+Up/Down cycle sessions · Ctrl+B branch/sidebar · Ctrl+T theme ✅ (Ctrl+B = branch).
- [x] Mouse: wheel scroll (native scrollback), click session row → switch, click fold header → toggle, drag-select → copy-mode
- [x] Placeholder-aware backspace (`[Image #N]`/`[File #N]` whole-block) · bracketed-paste w/ partial-marker holdback

---

## 9. Flavor, theming, i18n, effects (the v3 "soul" + N4/N8)

- [x] **Spinner** (N4): custom default (arc `◜◠◝◞◡◟` or braille), heat ramp (mint<20 s / amber<60 s / orange<180 s / red≥180 s), gerund rotation (34 words, ~6 s, deterministic seed), pet faces (5 styles × 4 heat × 4 frames, ~0.5 s), OSC0 title spinner, OSC-21337 tab status. **Never the CC `✻`.** ✅ `flavor/mod.rs`.
- [x] **Themes**: 6+ (ga-default, nord, gruvbox, dracula, tokyo-night, light), `Ctrl+T` live-preview picker (commit/revert), persisted. Rainbow 7-stop separator + shimmer. ✅ all 6 present in `theme/mod.rs`.
- [x] **i18n**: zh/en (~250 keys), fallback chain (lang→en→key), locale detect, `/language` full repaint, per-language rotating tips. ✅ 221 keys ea. (en==zh, `dictionaries_cover_the_same_keys` test); `i18n_fallback_chain` + `locale_detect_from_code` + `tips_rotate_deterministically_per_language` tests.
- [x] **Effects** (bounded, capability-gated, **OFF by default in cockpit**): Doom-fire band, braille snow, midpoint-displacement lightning, raised-cosine shimmer, sparkle. Single delta clock; FPS 30/60/0(smoke); honor `NO_COLOR`/truecolor/tmux; colors from tokens; `/effects demo` to showcase. ✅ `effects/{fire,snow,lightning,shimmer,sparkle}.rs`, single `TICK_DT` clock, `EffectsEngine::from_env` OFF default.

---

## 10. Subtle load-bearing behaviors (do NOT drop)

- [x] Structural-boundary stream commit (split at safe pos; no orphan tool headers) ✅ `render/measure.rs` + `markdown/mod.rs`.
- [x] Per-turn incremental fold (turn N → `▸ summary` when N+1 starts) ✅ `render/fold.rs`.
- [x] Exit-boundary intervene replay (queued user text re-submitted via `put_task`) ✅ `Intervene` frame → `_intervene` file in `ga_bridge.py`; `Submit`→`put_task`.
- [x] Per-session input stash on switch · draft stash/restore ✅ stash-on-switch in `switch_session`; draft stash/restore on **Ctrl+G** (Ctrl+S was reassigned to open the dashboard per N2).
- [x] Render cache invalidation on theme/resize/fold/language ✅ wrap cache keyed `(block_id, width)`; rev-bumped on theme/fold/language.
- [x] `@path` expansion on submit · `!cmd` seeds agent context without spending a turn ✅ `input/file_expand.rs` + `commands/mod.rs` (`run_shell_executes_and_captures_output` test).
- [x] Windows VT mode + UTF-8 charset; `PYTHONUTF8=1` for child; cp936-safe decode (`from_utf8_lossy`) ✅ child env `PYTHONUTF8=1`, reader uses `String::from_utf8_lossy`.
- [x] Cost: live token-rate estimate + final aggregation + context window % ✅ status footer shows model / ctx% / $cost.

---

## 11. Build phases (workflow + Monitor gates — verify green between each)

- [x] **Phase F — Foundation (compiling vertical slice).** Cargo.toml, main.rs event loop, alt-screen + panic restore, `--smoke/--help/--version`. Bridge spawns `ga_bridge.py`, handshakes `Ready` (**N1 disconnected FIXED**), Submit→stream→render in a **filled** layout (header + transcript + composer + status), custom spinner. Gate: `cargo build` + `cargo run -- --smoke` exit 0 + a live Submit/echo smoke. ✅ build 0/0, smoke 484 cells, echo handshake test passes.
- [x] **Phase 2 — Cockpit.** Block model + wrap cache + ScrollAnchor (**P1**), OSC-52 copy (**P2**), markdown+highlight+math (**P3**), tool chips, folding, composer full keyboard, status footer, flavor layer. Tests: resize/copy/math. ✅ all three pain-point tests present + passing.
- [x] **Phase 3 — Sessions + slash + pickers.** Session dashboard (**N2**), `/llm` (**N3**), all §4 commands, ask_user, menu primitive, `/btw /scheduler /rewind /continue /export /verbose` (**N6**), i18n, `/language`, `/emoji`. ✅ all §4 commands resolve + route; dashboard + pickers wired.
- [x] **Phase 4 — /workflows panel + watcher** (conductor/hive/goal). ✅ `workflow/{mod,panel,schema,sources,http}` wired into `View::Workflows`; `refresh_feeds_snapshot_into_panel` + `renders_both_styles_and_detail` tests pass.
- [x] **Phase 5 — Effects + themes + polish.** ✅ `effects/` single-clock engine (OFF by default, `/effects demo`), 6+ themes via tokens, rainbow separator.
- [x] **Phase 6 — Packaging.** `cargo build --release` → exe; later cargo-dist mac/linux. Final Monitor sweep: parity audit vs §4, pain-point tests, smoke. ✅ release exe (4.86 MB) + scripts copied + real connect proof + README/BUILD_LOG + final sweep. *(cargo-dist mac/linux musl deferred — noted gap.)*
- [x] **Monitor** (every phase): adversarial verification agent confirms the gate truly passes (compiles, tests pass, claim matches code) before the box is ticked. ✅ Phase 6 verified by re-running build/test/smoke/connect-proof and reading the code paths, not self-report.

---

## 12. Final acceptance (ship gate)

- [x] `cargo build --release` green; binary runs; **connects** (no false "disconnected"). ✅ release 0/0; direct `ga_bridge.py` Ready frame + Rust↔real-core `ga_bridge_handshakes_ready` test pass.
- [x] All three pain-point tests pass. ✅ `resize_then_scroll_no_drift`, `copy_across_wrap_has_no_newline`, `md_table`/`math_frac_block`/`math_greek`.
- [x] §4 parity: every command present (UI or fwd); §6 dashboard; §7 panels; §5 filled layout; §9 soul. ✅ 38/38 commands route; dashboard + workflows/btw/scheduler/rewind/continue panels; smoke fills 484 cells; flavor/themes/i18n present.
- [x] Custom spinner (not `✻`); `/llm` switches real models; effects off-by-default but demoable. ✅ arc/braille/pulse default; `SwitchLlm`→`next_llm`; `EffectsEngine::from_env` OFF default + `/effects demo`.
- [x] Ink backup preserved; Rust tree git-tracked (don't lose it like the prior Rust v4). ✅ `../tuiapp_v4_ink_backup/` intact; Rust tree under `frontends/tuiapp_v4/` (untracked `??` — **commit pending per "no git commit" instruction**; see report).

### 12.1 Real-GA-format redesign — finalized 2026-05-31 (see `redesign_cc.md` §5/§6)
- [x] Transcript renders the REAL GA output format clean: `<summary>`→`↳` breadcrumb, `Turn N`→spacing, compact `🛠️ name(args)`→`⏺` bullet + indented `[Action]/[Status]/[Info]`, `!!!Error:`→compact red line, `[MixinSession]…retry`/`[bridge]` stderr SUPPRESSED. Verified by reading `--dump-frame normal|shell|busy|effort` + a strict codepoint scan (no raw markers, no pictographic emoji).
- [x] User-input full-width band bg rgb(58,58,58); CC RGB palette in `theme/tokens.rs`; bottom decluttered to 1 status + 1 hint row; model-name truncated to the primary segment (`MixinSession·codex-pro`).
- [x] `/effort` slider (`low medium high xhigh max`, `max→xhigh`) forwards `/session.reasoning_effort=` for GA hot-reload; spinner shows `thinking · <level>`.
- [x] Mouse: left-click→dashboard, **right-click→back** (was missing) both wired.
- [x] Spinner kaomoji pet OFF by default (redesign_cc.md §2.6 "NOT emoji pet by default"); Bear/Cat/Dot/Unicode/Fox still opt-in via `/emoji`.
- [x] **278 tests green** (≥262 floor held; 0 fail, 2 ignored real-GA); `cargo build --release` green (4.92 MB exe + `ga_bridge.py`/`echo_bridge.py` copied to `target/release/`); real GA core handshakes `Ready` through the release-dir bridge.
