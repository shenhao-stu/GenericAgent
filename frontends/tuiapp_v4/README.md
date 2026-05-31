# tui_v4 — GenericAgent terminal UI (Rust + ratatui)

A from-scratch rebuild of GenericAgent's terminal UI (`frontends/tui_v3.py` +
`tuiapp_v2.py`) in **Rust + ratatui + crossterm**. It replicates and surpasses
Claude Code, fuses the best of Claude Code / Codex / kimi-code / reasonix, and
keeps 100% of the v2 + v3 feature set. Single ~4.6 MB binary, <5 ms startup,
~0% idle CPU.

The authoritative spec is [`checklist.md`](checklist.md). This README is the
operator's guide.

---

## Run

```bash
# from the package dir (frontends/tuiapp_v4) — dev build, auto-discovers the bridge
cargo run

# non-interactive self-checks
cargo run -- --smoke      # render one frame headless, print non-blank cell count, exit 0
cargo run -- --help       # usage
cargo run -- --version    # version string
```

`cargo run` enters the alt-screen, spawns a GA-core child via
`scripts/ga_bridge.py`, and shows the filled cockpit (header / transcript /
streaming band / composer / status footer). `Esc` is universal-back and never
exits; quit with `Ctrl+C`, `Ctrl+Q`, `/quit`, or `/exit`.

---

## Build

```bash
cargo build              # debug
cargo build --release    # optimized → target/release/tui_v4.exe  (~4.6 MB)
cargo test               # full hermetic suite (258 unit tests, no network/python)
```

**Packaging a release drop:** copy the two bridge scripts next to the exe so a
relocated binary still connects:

```bash
cp scripts/ga_bridge.py   target/release/ga_bridge.py
cp scripts/echo_bridge.py target/release/echo_bridge.py
```

The exe also finds the bridge without the copy via a repo-root walk (see below),
so the copy is belt-and-suspenders for a fully standalone `release/` folder.

> Windows + Chinese locale: the child is always spawned with `PYTHONUTF8=1` and
> all stdout is decoded with `from_utf8_lossy`, so a stray GBK byte can never
> kill the reader. No `chcp` dance required.

---

## Bridge discovery (the `disconnected` fix — N1)

The old Ink build showed a silent **`disconnected`** because a packaged exe in
`release/` could not find `scripts/ga_bridge.py` one dir up, and spawn/import
failures were swallowed. tui_v4 fixes this by construction:

1. **Robust discovery** (`src/bridge/mod.rs::ga_bridge_candidates`), highest
   priority first:
   1. env override — `GA_TUI_BRIDGE` / `GA_BRIDGE_PATH` / `TUI_V4_BRIDGE`
   2. `ga_bridge.py` **next to the exe** (the drop-in deploy)
   3. `scripts/ga_bridge.py` next to the exe
   4. `../scripts/ga_bridge.py` (exe in `release/` beside `scripts/`)
   5. `<GENERICAGENT_ROOT>/frontends/tuiapp_v4/scripts/ga_bridge.py`
   6. **walk UP from the exe dir** to the first ancestor containing
      `agentmain.py` (the GA repo-root marker) → `…/frontends/tuiapp_v4/scripts/ga_bridge.py`
   7. walk UP from the cwd the same way
   8. cwd-relative (`./scripts/ga_bridge.py`, `./ga_bridge.py`, …)

   For `target/release/tui_v4.exe`, candidate **#2** (the copied script) and
   candidate **#6** (the repo-root walk) both resolve — verified by the
   `packaging_discovery_resolves_both_ways` test.

2. **Every failure is surfaced, never silent.** Spawn / import / child-exit /
   stdout-parse-noise / stderr all become visible `BridgeEvent`s the UI shows as
   `disconnected: <real reason>` (e.g. "is python on PATH?", or the actual
   Python traceback) — there is no bare "disconnected" word.

3. **The status word reflects the real handshake.** `connected` flips true only
   when an actual `{"type":"Ready"}` frame arrives from the GA core; a watcher
   thread reports child exit so a dead core never looks like a stall.

**Real connect proof** (run from the repo root):

```bash
PYTHONUTF8=1 python frontends/tuiapp_v4/scripts/ga_bridge.py < /dev/null
# → V=1 caps=submit,abort,intervene,switchllm,ping,askuser
# → {"type":"Ready","version":"1","model":"MixinSession/codex-pro|…"}

cargo test --test bridge_liveness -- --ignored ga_bridge_handshakes_ready
# → the Rust bridge spawns + handshakes the REAL python GA core (passes)
```

---

## Keybindings

### Cockpit (chat) — composer & navigation
| Key | Action |
|---|---|
| `Enter` | submit |
| `Shift+Enter` / `Ctrl+J` | newline |
| `←/→/↑/↓` | move cursor (history at vertical edges); `Shift+arrow` selects |
| `Home` / `End` | start / end of composer line |
| `Ctrl+Home` / `Ctrl+End` | jump transcript to top / tail |
| `PageUp` / `PageDown` | scroll transcript |
| `Ctrl+A` | select all · `Ctrl+E` end-of-line · `Ctrl+U` kill-to-start |
| `Ctrl+X` cut · `Ctrl+V` paste · `Ctrl+Z` undo · `Ctrl+Y` redo |
| `Ctrl+G` | stash / restore draft |
| `Ctrl+L` | force redraw (sleep/wake recovery) |
| `Tab` | complete the `/`-palette or `@`-file dropdown |
| `Esc` | universal back (clear pending ask → collapse selection → stash draft); **never exits** |
| `Ctrl+C` / `Ctrl+Q` | quit |

### Sessions
| Key | Action |
|---|---|
| **`Ctrl+S`** | **open the full-screen session dashboard** (§6 / N2) — also reachable by left-clicking the sessions area |
| `Ctrl+N` | new session (+ switch) |
| `Ctrl+W` / `Ctrl+D` | drop (close) the active session, keeping its log |
| `Ctrl+B` | branch the active session (fork w/ copied transcript) |
| `Ctrl+Up` / `Ctrl+Down` | cycle active session |
| `Ctrl+O` | toggle tool-chip / turn folding |

### Session dashboard (Ctrl+S view)
`↑/↓` navigate (skips collapsed) · `Enter` open/switch · `Space` quick-reply ·
`Ctrl+X`/`Del` delete (keeps log) · `r` rename · `Ctrl+N` / bottom input = new
session · left-click row = switch · `Esc` / `Ctrl+S` back to chat.

### Magic prefixes & overlays
- `!cmd` — run a host shell command (30 s timeout), echo the output, and seed it
  into the agent's context without spending a turn (hot-pink input border in
  shell mode).
- `@path` — inline a file (<100 KB) as `[File: path]…[/File]`; gitignore-aware
  path completion.
- `/workflows` — full-screen live conductor / hive / goal panel (§7). Inside it:
  focus nav, toggle render style, open a per-node detail overlay; `Esc` back.
- `/effects [demo|off|subtle|full]` — effects intensity; `/effects demo` plays the
  showcase splash. Effects are **OFF by default** in the cockpit.
- `Esc` then `Esc` (within 0.8 s) → `/rewind`.

Run `/help` for the full, in-app command list.

---

## Slash commands (§4 — 38 names, union of v2 + v3 + aliases)

All 38 resolve through one registry (`src/commands/registry.rs`) and route by
kind — **app** (in-app), **UI** (dedicated panel/picker), or **fwd**
(core-forwarded to `ga_bridge.py`). Verified by `registry_resolves_all_commands`
+ `classify_slash_routes_every_kind`.

`/help` `/status` `/sessions` `/new` `/switch` `/close` `/rename` `/branch`
`/rewind` `/clear` `/stop` `/abort` `/llm` `/btw` `/review` `/update` `/autorun`
`/morphling` `/goal` `/hive` `/conductor` `/workflows` `/scheduler` `/continue`
`/resume` `/cost` `/export` `/restore` `/reload-keys` `/language` `/emoji`
`/verbose` `/tools` `/trace` `/effects` `/theme` `/quit` `/exit`

Dedicated interactive panels exist for `/llm` (model picker via `mykey.py`'s
`list_llms()`), `/btw`, `/scheduler`, `/rewind`, `/continue`, `/export`,
`/verbose`, `/theme`, `/language`, `/emoji`, the session dashboard (`/switch`),
and `/workflows`.

---

## The three pain-point fixes (HARD acceptance)

- **P1 — Resize never corrupts scroll.** ratatui is immediate-mode: logical
  `Block`s store SOURCE text; the wrap cache is keyed `(block_id, width)`; the
  scroll position is a logical `ScrollAnchor`, not a row count. On resize the
  wrap recomputes at the new width and the viewport re-derives — there is no
  retained, stale-width scene to corrupt. *Test: `resize_then_scroll_no_drift`.*
- **P2 — Copy yields clean logical text.** Explicit copy actions emit the
  logical SOURCE string via **OSC-52** (SSH-safe), with a native→tmux→OSC52
  fallback. We copy the source, never rendered rows, so soft-wrap newlines are
  structurally impossible. *Test: `copy_across_wrap_has_no_newline`.*
- **P3 — Correct markdown + math.** `pulldown-cmark` → themed ANSI, `syntect`
  code highlight, and a custom `latex_to_unicode` for `$…$` / `$$…$$` (Greek,
  sup/sub, `\frac` stacked, `\sqrt`, `\sum`/`\int` with limits, matrices).
  *Tests: `md_table`, `math_frac_block`, `math_greek`.*

---

## Module map

```
src/
  main.rs              entry: --smoke/--help/--version, alt-screen, panic restore, event loop
  app/{mod,session}.rs AppState + reducers + overlay stack; SessionMap (N bridge children)
  bridge/{mod,protocol}.rs  spawn ga_bridge.py, robust discovery (N1), reader/writer/watcher threads
  render/{block,measure,viewport,copy,chip,fold}.rs  logical Block model, wrap cache, ScrollAnchor (P1), OSC-52 (P2)
  markdown/{render,highlight,math}.rs  pulldown-cmark + syntect + latex_to_unicode (P3)
  components/          dashboard, overlay, picker, scheduler, continue_picker, effects_paint
  workflow/{mod,panel,schema,sources,http}.rs  conductor/hive/goal watcher + live panel (§7)
  effects/{mod,fire,snow,lightning,shimmer,sparkle}.rs  bounded, capability-gated, OFF by default (§9)
  theme/{mod,rainbow}.rs  6+ theme tokens (no hardcoded RGB), 7-stop rainbow
  i18n/mod.rs          zh/en
  flavor/mod.rs        custom spinner sets (NOT the CC ✻), gerunds, heat ramp, pet faces
  commands/{mod,registry}.rs  38-command registry + dispatch + magic prefixes
  input/               composer, history, paste, file/path expansion, shell
  util/{mod,osc}.rs    terminal caps, OSC sequences
tests/bridge_liveness.rs  ignored real-GA handshake test
```

---

## See also

- [`checklist.md`](checklist.md) — the full build contract / spec.
- [`BUILD_LOG.md`](BUILD_LOG.md) — phase log + final stats.
- `../tuiapp_v4_research/recon/` — recon notes (GA core bridge, hive, effects engine).
- `../tuiapp_v4_ink_backup/` — the preserved Ink v0.1 source.
