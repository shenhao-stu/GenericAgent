# R1 — tui_v4 Feature-Parity GAP Audit (Round 6)

Scope: legacy `frontends/tuiapp_v2.py` + `frontends/tui_v3.py` vs the Rust rewrite
`frontends/tuiapp_v4`. Emphasis: (a) slash commands, (b) keybindings, (c) how the
MODEL'S INTERMEDIATE OUTPUT (tool calls, reasoning/thinking, step folding) is
rendered. Round-5 claimed "zero gaps"; this re-verifies by reading the actual code.

Method: read every command dispatch table + key handler + the fold/chip/markdown
render path on both sides, and traced the real wire format the v4 bridge emits
(`agent_loop.py` + `ga_bridge.py`) so the comparison is against what v4 ACTUALLY
renders, not a clean fixture.

---

## 1. Slash commands — COMPLETE SUPERSET (no command gaps)

### Legacy command sets (authoritative source lines)
- **v2** `COMMANDS` table, `tuiapp_v2.py:1234-1265`: help, status, sessions, new,
  switch, close, rename, branch, rewind, clear, stop, llm, btw, review, update,
  autorun, morphling, goal, hive, conductor, scheduler, continue, resume, cost,
  export, restore, reload-keys, quit.
- **v3** `_cmds()` palette, `tui_v3.py:1810-1841` + the `_cmd` dispatcher
  `tui_v3.py:4073-4540`: help, status, sessions(=status alias, `:4081`), llm, btw,
  review, update, autorun, morphling, goal, hive, conductor, scheduler, rewind,
  continue, new, rename, clear, cost, verbose(+`tools`+`trace`, `:4503`), export,
  stop(+`abort`, `:4077`), language, emoji, resume(`:4257`), quit(+`exit`,`:4075`).

### v4 registry — `commands/registry.rs:61-103` (42 entries incl. 5 aliases)
Every legacy name resolves AND routes to a real handler (`commands/dispatch.rs`):
- UI overlays/pickers (`open_ui_command`, `dispatch.rs:81-264`): help, keybindings,
  status/sessions, switch, workflows, llm, theme, effort, language, export,
  verbose/tools/trace, rewind, continue, scheduler, btw.
- Forwarded to core (`SlashOutcome::Forward`, `dispatch.rs:49-64`): review, update,
  autorun, morphling, goal, hive, conductor, resume.
- In-app (`app_command`, `dispatch.rs:268-337`): quit/exit, clear, stop/abort, cost,
  fold, mouse, new, close, rename, branch, restore, reload-keys.

**Net-new in v4 (parity-plus, not gaps):** `/keybindings`, `/workflows`, `/effort`,
`/fold`, `/mouse`, `/theme` (v2 had theme only as the Ctrl+T binding, never a
command). **VERDICT: command surface is a strict superset of v2 ∪ v3 — 0 gaps.**

---

## 2. Keybindings — COMPLETE (no functional gaps; one model difference)

### Legacy sources
- **v2** App `BINDINGS` `tuiapp_v2.py:2586-2608`; InputArea `BINDINGS` `:1875-1893`.
- **v3** raw-byte dispatcher `_keys()` `tui_v3.py:5196-5298` + the PTK Esc/Esc-Esc
  handler `:5396-5438`; advertised in `/help` `tui_v3.py:162-166`.

### v4 cockpit handler — `input/keymap.rs:38-286`
| Action | v2/v3 | v4 |
|---|---|---|
| Submit / newline | Enter / Ctrl+J·Shift+Enter·Ctrl+Enter (v2 `:1876-1878`) | `keymap.rs:219-225` ✓ |
| 3-stage Ctrl+C (copy→abort→arm→quit) | v2 `:2587`, v3 `:5319-5344` | `keymap.rs:93-96,401-433` ✓ |
| Esc / Esc-Esc → /rewind | v3 `:5425-5436` | `keymap.rs:273-278,438-450` ✓ |
| Fold all tool chips | Ctrl+O (v2 `:2594`, v3 `:5274`) | moved to **Ctrl+Shift+O** + `/fold`; Ctrl+O is now copy-last-reply (`keymap.rs:68-83`) ✓ intentional |
| Undo/redo, select-all, cut | Ctrl+Z/Y/A/X (v3 `:5257-5267`) | `keymap.rs:154-187` ✓ |
| Help cheat-sheet | Ctrl+/ / Ctrl+_ (v2 `:2600-2602`) | `keymap.rs:115-118` ✓ |
| Theme picker | Ctrl+T (v2 `:2607`) | `keymap.rs:108-111` ✓ |
| New / drop / cycle session | Ctrl+N/D/W, Ctrl+↑↓ (v2 `:2591-2598`) | `keymap.rs:127-152` ✓ |
| Stash draft | Ctrl+S (v2/v3) → v4 uses **Ctrl+G** (`keymap.rs:197-200`); Ctrl+S now opens dashboard (`keymap.rs:121-124`) — intentional remap |
| Transcript scroll | (n/a in v3 — see below) | PageUp/Down + Ctrl+Home/End `keymap.rs:262-272` ✓ (extra) |

**Model difference, not a strict gap:** v2 `Ctrl+B` = `toggle_sidebar` (the session
sidebar, `tuiapp_v2.py:2593,2999`). v4 has **no sidebar** (it uses a full-screen
dashboard, Ctrl+S / ← ), and reassigns Ctrl+B to `branch` (`keymap.rs:137`). Session
visibility/switching is preserved via Ctrl+S, ←, Ctrl+↑↓, so no capability is lost —
only the always-visible sidebar widget is gone. Worth a note, not a blocker.

**Scrolling note:** v3 is scrollback-first (PTK owns only the live region; finalized
history is pushed to native terminal scrollback and the mouse wheel scrolls it —
`tui_v3.py:5358-5369`). v3 therefore has NO in-app PageUp/Home keys; v4 keeps an
in-app scrolling transcript and ADDS them. Not a gap.

**VERDICT: keybindings complete; Ctrl+O / Ctrl+S / Ctrl+B are deliberate remaps, the
sidebar widget is the only dropped surface (capability retained via dashboard).**

---

## 3. Model intermediate-output rendering — REAL GAPS HERE

### 3.0 Wire-format sanity check (avoids a false positive)
`agent_loop.py:88-89` emits tool calls in TWO forms gated on `verbose`:
- `verbose=True` → `🛠️ Tool: \`name\`  📥 args:` + fenced args + fenced result (the
  form v3 parses; v3 forces `agent.verbose=True`, `tui_v3.py:1324`).
- `verbose=False` → `🛠️ name(args)` compact (`agent_loop.py:89`).

The v4 bridge sets **`agent.verbose = False`** and **`agent.task_dir = …`**
(`ga_bridge.py:333,336-340`). With `task_dir` set, `agent_loop.py:62` emits the BARE
`Turn N ...` marker; with `verbose=False`, line 89 emits the COMPACT chip. v4's
`chip::parse_tool_calls` (`render/chip.rs:136`) + `fold::find_turn_markers`
(`render/fold.rs:273`) are written for exactly those forms. **So v4's parser matches
its own bridge — NOT a gap.** (Latent: `chip::find_turn_line` matches `Turn ` by
PREFIX only, so it would miss the `LLM Running (Turn N)` form; that form only appears
when `task_dir` is falsy, which the bridge never does. Latent robustness nit, not a
live bug.)

### 3.1 GAP A — DUPLICATED summary line on every expanded turn  ★ the prompt's "duplicated breadcrumb"
File: `markdown/mod.rs:181-205` (the `▾` collapse header) **and** `markdown/mod.rs:339-353`
(the `↳` breadcrumb), both fed the SAME turn body.

For an EXPANDED, non-preamble turn that carries a `<summary>`, the cockpit renderer
emits the summary text TWICE, back to back:
1. `mod.rs:186-202` pushes a header row ` ▾ <title>` where
   `title = render::fold::turn_title_pub(body)` → `turn_title()` →
   `extract_summary(body)` returns the summary string (`render/fold.rs:320-348`).
2. `mod.rs:339-352` then calls `hoist_summary(body)` which extracts the **same**
   `<summary>` and pushes ` ↳ <summary>` as a dim breadcrumb (`mod.rs:677-700`).

There is NO de-dup between them. Result on screen:

```
 ▾ Read the config and found the port      <- collapse header (title = summary)
 ↳ Read the config and found the port      <- breadcrumb (same summary)
 …body…
```

Trigger frequency is HIGH, not edge: with `<2` turn markers and not-folded,
`fold_turns_with` returns `Text { turn: Some(1) }` (`render/fold.rs:199-209`), so even
a SINGLE-turn in-progress reply (the normal live-streaming case) takes the `▾`-header
path and duplicates its summary. The `↳` marker at `markdown/mod.rs:345` is itself
fine — the bug is that the `▾` header at `mod.rs:188` derives the same summary.

Why v3 is clean: v3's `_compress` deletes `<summary>…</summary>` wholesale via
`_META_TAG_RE.sub('', text)` (`tui_v3.py:1099,1166`), and only ever surfaces the
summary as the `▸ {summary}` title on FOLDED turns (`tui_v3.py:3648`). v3 never draws
a `▾` header for an expanded turn at all (the final/expanded turn is plain body,
`tui_v3.py:3606-3607`). So v3 shows the summary AT MOST once.

Fix options: (i) when the `▾` header is emitted, suppress the `↳` breadcrumb for the
same summary; or (ii) drop the `▾`-header title when a summary breadcrumb will follow;
or (iii) only emit the `▾` header for turns that have NO summary (use the breadcrumb
as the visible label and make the collapse target the breadcrumb row).

### 3.2 GAP B — `/verbose` tool-call audit is a static flat list (was an interactive inspector)
Files: data model `app/mod.rs:165` (`tool_audit: Vec<String>`), population
`app/reducer.rs:209`, render `components/overlay/info.rs:215-241`, key handling
`input/views.rs:465-470`.

v4's `/verbose` (`/tools` `/trace`) opens `Overlay::Verbose`, which renders
`app.tool_audit` — a flat `Vec<String>` where each entry is just
`format!("{badge} {name}{args}")` (`reducer.rs:209`; the result body and raw payload
are NEVER stored). The overlay is non-interactive: `input/views.rs:465-470` is a
catch-all where ANY of Esc/Enter/q closes and every other key is ignored — no select,
no scroll, no field switch, no copy, no export.

v3's `/verbose` (`tui_v3.py:4684-4747`) is a full tool-trace inspector backed by
structured `ToolRecord{id,name,args,result,status,raw}` (`tui_v3.py:1619-1626`):
- two-pane LIST + DETAIL layout with a selection marker + per-tool status color
  (`:4707-4716`);
- `↑/↓` (or `k/j`) select a tool; `PgUp/PgDn` scroll the detail (`:4721-4728`);
- `Enter` cycles the detail field through **result / args / raw** (`:4729-4730`);
- `c` copies the current field; `e` exports it to a temp file (`:4731-4736`).

So v4 lost: the args view, the raw view, per-tool selection, detail scrolling, copy,
and export. Even v4's own `/help` text still implies the richer interaction. This is
the single biggest intermediate-output parity regression. (Note v3 itself flags the
viewer as broken under PTK at `tui_v3.py:4739-4743`, but the FEATURE and data model
exist; v4 dropped the data, not just the input wiring.)

### 3.3 What v4 does WELL (intermediate output) — for the record
- Compact `🛠️ name(args)` chips render as tui_v3-style bordered boxes with name +
  status badge + `·tN` on the top border, arg-hint, result preview, and an in-box
  `▸ +N more` / `▾ collapse` fold affordance (`render/chip.rs:274-319`,
  `markdown/mod.rs:407-438`) — a faithful `_chip_box` port (`tui_v3.py:1732-1779`).
- Per-turn incremental fold (`▸ summary` for completed turns, last turn open) and
  structural-boundary stream commit are ported well, with extra holdbacks for
  unclosed math / fenced code / streaming GFM tables (`render/fold.rs:391-467`)
  beyond v3's `_safe_pos` (`tui_v3.py:4934-4958`).
- `Turn N ...` markers are never rendered as text; `<summary>` tags are hidden — both
  better than raw. The ONLY problem is the §3.1 duplication.
- "Reasoning/thinking": GA's `verbose=False` path does not stream a separate thinking
  channel to either TUI, and both treat `<thinking>` as a hidden/held tag
  (`render/fold.rs:410`, `tui_v3.py:1099`). No gap.

---

## 4. Verdict

- Slash commands: **complete superset** (0 gaps).
- Keybindings: **complete** (Ctrl+O/Ctrl+S/Ctrl+B are intentional remaps; the
  always-on session sidebar is the only dropped widget, capability retained via the
  dashboard).
- Model intermediate-output rendering: **2 concrete gaps** —
  A) duplicated summary line (`▾` header + `↳` breadcrumb) on every expanded turn,
  B) `/verbose` reduced from v3's interactive result/args/raw inspector to a flat,
  read-only, name+args-only list.

Round-5's "zero gaps" claim does NOT hold for the rendering of intermediate output.

**PARITY: 2 gaps**
