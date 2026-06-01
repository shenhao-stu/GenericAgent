# C3 — v2 keybinding parity · clean no-newline copy · Left/Right swap · shift+enter

Audit target: `D:/GenericAgent/frontends/tuiapp_v4`. Competitors: `tuiapp_v2.py`, CC (`temp/claude-code/src/keybindings`), Codex `codex-rs/tui`. Judged vs `memory/code_review_principles.md`.

---

## Findings (file:line bugs, root cause not symptom)

### F1 — Left/Right view-switch is REVERSED (Q5) — `main.rs:702-707`
```rust
702  KeyCode::Left if app.composer.is_empty() && !shift => {
703      app.close_dashboard(); // already-in-cockpit → no-op
704  }
705  KeyCode::Right if app.composer.is_empty() && !shift => {
706      app.open_dashboard();
707  }
```
Spec: **empty composer + Left → session view; Right → conversation view.** Code does the opposite — Left closes (stays in chat), Right opens the dashboard. The dashboard side mirrors the same wrong polarity at `main.rs:919-934`: in `handle_dashboard_key`, `Left` collapses/exits (`close_dashboard`) and `Right` expands. So the user in the cockpit must press **Right** to "enter" sessions — exactly the inversion reported. Root cause: the open/close verbs are bound to the wrong arrow on BOTH sides; this is a 2-line literal swap in the cockpit (the dashboard tree-collapse gesture is a separate concern — see Fix B note).

### F2 — `ctrl+shift+c` is mis-purposed as "copy last reply" (Q2) — `main.rs:549-552`
```rust
549  KeyCode::Char('c' | 'C') if ctrl && shift => {
550      export_action(app, 0);   // copies the LAST ASSISTANT REPLY
551      return;
552  }
```
The comment calls it "inline-copy shortcut, same as /export clip". That is the "把 ctrl+shift+c 当成所有复制" anti-pattern the user rejects. In every peer, `ctrl+shift+c` means **copy the current terminal SELECTION**, not "copy the last reply":
- CC: `defaultBindings.ts:210` → `'ctrl+shift+c': 'selection:copy'`.
- xterm/VTE convention: ctrl+shift+c = copy selection.

tui_v4 has no selection model, so it overloaded the chord to dump the last reply — wrong mental model. Fix: **remove this binding** (F2), let the terminal own ctrl+shift+c for native selection copy, and move "copy last reply" to an explicit, correctly-named action (Fix D → `Ctrl+O`, Codex's choice).

### F3 — mouse capture is FORCED ON at startup → native drag-select is dead by default (Q2 root cause) — `main.rs:2273`
```rust
2273  execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
```
With `EnableMouseCapture`, the app eats every mouse-down/drag, so the terminal never sees a selection gesture — **"仍无法在终端选中文本复制"**. `app.mouse_capture` is never initialized to match this forced-ON state either (see F7). This is the exact regression Codex documents: *"Because the TUI captures mouse events to handle its own scrolling, native terminal text selection is disrupted"* (codex issue #1247, #16149). Codex's answer is `/toggle-mouse-mode` + a default that lets users reclaim selection. Fix C: **start with mouse capture OFF** (terminal owns drag-select), make wheel-scroll an opt-in toggle.

### F4 — there is NO clean-copy of arbitrary selected text; copy paths only ever copy `block.source` wholesale
`render/copy.rs` is solid for the *logical-source* guarantee (P2): `copy_to_clipboard` base64s `block.source` so soft-wraps never become `\n` (copy.rs:130-152, test copy.rs:369-405). BUT every caller copies a *whole block / whole transcript*, never a user-chosen span:
- `export_action` (main.rs:2001-2033): id 0 = last reply, 1 = whole transcript, 2 = file.
- ctrl+c CopySelection (main.rs:2080-2090) copies only the **composer** selection.
- `join_visual_rows` (copy.rs:162-181) — the row→logical reconstruction that WOULD power a drag-select range copy — is `#[allow(dead_code)]`, never wired. `CopyOverlay` (copy.rs:329-359) is a stub.

So Q2's "select arbitrary text in the terminal and copy a markdown table / multiline cleanly" is **unsolved**: the only clean span-copy is via the terminal's own selection, which F3 disables. The redesign must lean on native selection (F3 off) for arbitrary spans, and keep ONE explicit clean-copy action for "the last reply" convenience.

### F5 — Shift+Enter newline works but Ctrl+Enter is NOT wired (Q6) — `main.rs:679-685`
```rust
679  KeyCode::Enter if shift => { app.composer.newline(); }
682  KeyCode::Enter => { let action = app.composer.submit(); … }
```
Shift+Enter → newline is correct (matches v2 `tuiapp_v2.py:1804`). Ctrl+J is the fallback (main.rs:663-666). But **Ctrl+Enter is missing** — v2 binds it explicitly (`tuiapp_v2.py:1803 Binding("ctrl+enter","newline")`). On terminals that deliver ctrl+enter (kitty proto / Windows VT), a user pressing it today **submits** instead of inserting a newline. The query explicitly asks for "shift+enter 或 ctrl+enter". One-line add (Fix C-newline).

### F6 — `Ctrl+O` collides: bound to FOLD, but both peers use it for transcript/copy — `main.rs:611-614`
```rust
611  KeyCode::Char('o') if ctrl => { app.fold_all = !app.fold_all; return; }
```
CC: `ctrl+o` = `app:toggleTranscript` (defaultBindings.ts:44). Codex: `Ctrl+O` = **copy latest output** (PR #16966, chosen over Alt+C *because Ctrl+O is a stable control chord across Terminal.app/iTerm2/SSH*). v2 uses `ctrl+o` for fold (tuiapp_v2.py:2520) — so tui_v4 inherits v2 here, which is fine for parity, but it means Ctrl+O is NOT free for the Codex-style copy. Decision needed (Open Q1): keep Ctrl+O=fold (v2 parity) and make the explicit clean-copy `Ctrl+Y`-free? No — Ctrl+Y is redo (main.rs:616). Recommend: **Ctrl+O = copy-last-reply** (align Codex, the stronger peer for a Rust TUI) and move fold to a `/fold` command + keep it off the hot path, OR keep fold on Ctrl+O and bind copy to `Alt+C`. The spec picks Ctrl+O=copy (see Fix D rationale).

### F7 — `app.mouse_capture` initial value never reconciled with the forced-ON capture (F3)
`set_mouse_capture` (main.rs:2203-2210) and the `/mouse` toggle (main.rs:1861-1874) flip `app.mouse_capture`, but startup (main.rs:2273) calls `EnableMouseCapture` directly without setting `app.mouse_capture=true`. After Fix C (start OFF), `AppState::new()` must default `mouse_capture=false` so the `/mouse` toggle and the footer hint agree with reality. Local-reasoning violation (principle #2): the true terminal state lives in an `execute!` call, not in the field that the UI reads.

### F8 — Esc-as-rewind chord present but the v2 "clear pending queue" + "free-text return" Esc layers are absent
v2 `action_escape` (tuiapp_v2.py:2952-3006) has a 5-level Esc cascade: (1) return from free-text picker, (2) cancel active choice, (3) hide palette, (4) clear pending-message queue, (5) arm-rewind. tui_v4's `cockpit_universal_back` (main.rs:2127-2137) only does pending-ask → collapse-selection → stash-draft. The Esc-Esc→rewind chord (keychord.rs:133-142) is good, but **palette-open Esc does not dismiss the palette** in the cockpit (the palette is implicit, not an overlay) — a parity gap, minor. Flag, not blocker.

---

## Competitor patterns (CC / Codex / v2, with file cites)

### v2 — the clean no-newline copy mechanism (THE answer to Q2 for tables/multiline)
v2 never copies the soft-wrapped visible grid. It renders each assistant message **twice** and maps screen coords back to the un-wrapped source:
- `_align_md_renders(narrow_raw, wide_raw)` (`tuiapp_v2.py:589-720`): walks the **narrow** (visible-width) render and the **wide** (~10000-col, never-wraps) render line-by-line; builds `wrap_groups` so N narrow visual rows map to 1 wide logical line. Copy then emits the WIDE line → soft wraps vanish.
- Tables: `_md_line_has_box_drawing` (`tuiapp_v2.py:540-547`, matches U+2500–U+257F) detects a Rich table run; that run becomes **passthrough** (`tuiapp_v2.py:614-621`, `_build_passthrough_source:554-586`) — copied verbatim as seen, so `│`/`─` survive but cell text stays on its row and DOESN'T get mangled by the wide/narrow aligner. Comment at `tuiapp_v2.py:559-563` is explicit about the tradeoff.
- Selection→clipboard: `action_handle_ctrl_c` (`tuiapp_v2.py:2851-2873`) copies `inp.selected_text` (focused input) FIRST, then `self.screen.get_selected_text()` (Textual's screen drag-selection over the aligned source). So in v2 you DRAG to select, then Ctrl+C copies the mapped logical text. **Native drag-select is the primary path; the wide/narrow map is what makes it newline-free.**
- `copy_to_clipboard` override (`tuiapp_v2.py:2837-2849`): pbcopy on macOS, OSC52 elsewhere — because Textual's OSC52 is broken on Terminal.app. (tui_v4's copy.rs chain is already richer: OSC52→arboard→copy-mode.)
- `wrap_for_clipboard` (`export_cmd.py:15-24`): fences exported text with `max(3, longest_backtick_run+1)` backticks so nested code fences survive. tui_v4's `/export` does NOT do this — a reply containing ``` will break when pasted into markdown (minor follow-up).

**Lesson for tui_v4:** tui_v4's `block.source` IS the wide/logical text already (it stores the un-wrapped source), so it does NOT need v2's dual-render hack for *whole-block* copy — copy.rs already wins P2 by construction. The gap is purely **arbitrary-span selection**: that must come from the terminal (mouse capture OFF), and `join_visual_rows` (copy.rs:162) is the row→logical fallback if tui_v4 ever adds an in-app selection later.

### v2 — complete BINDINGS table (`tuiapp_v2.py`)
App-level (`2512-2534`): `ctrl+c`/`cmd+c`→handle_ctrl_c (copy-sel/stop/quit, priority); `ctrl+n`/`cmd+n`→new; `ctrl+b`→toggle_sidebar; `ctrl+o`→toggle_fold; `ctrl+up`→prev_session; `ctrl+down`→next_session; `ctrl+d`/`cmd+d`→drop_session; `ctrl+slash`/`ctrl+/`/`ctrl+underscore`/`cmd+slash`/`cmd+/`→show_help; `escape`→escape (queue-clear / arm-rewind); `tab`→complete_command (priority); `ctrl+t`→pick_theme.
InputArea (`1801-1818`): `ctrl+j`+`ctrl+enter`+`shift+enter`→newline; `ctrl+v`/`cmd+v`→paste; `ctrl+u`→clear_input; `ctrl+s`/`cmd+s`→stash.
Pickers: SelectionList `enter`→submit, `escape`→cancel (`1638-1640`); OptionList `right`→select, `left`/`escape`→cancel (`1256-1264`); Help overlay `escape`/`ctrl+slash`→dismiss (`2433-2439`).

### CC — keymap model (`temp/claude-code/src/keybindings`)
- Bindings are **declarative data**, context-scoped: `DEFAULT_BINDINGS: KeybindingBlock[]` keyed by context (Global/Chat/Scroll/Transcript/Select/…) (`defaultBindings.ts:32-340`). User overrides layer on top. This is the constraints-in-types model tui_v4's giant match block lacks.
- Copy: `Scroll` ctx `ctrl+shift+c`/`cmd+c` → `selection:copy` (`defaultBindings.ts:204-211`). `ctrl+o`→toggleTranscript (`:44`). `ctrl+c`/`ctrl+d` are time-based double-press, defined but **reserved/unrebindable** (`:36-41`).
- Windows nuance: shift+tab unreliable without VT mode → falls back to `meta+m` (`defaultBindings.ts:21-30`) — relevant to tui_v4's Windows-first target (the env is Windows 10).
- Match: `matchesKeystroke` normalizes Ink Key→name + modifier compare (`match.ts:60-105`); the **escape-meta quirk** (Ink sets meta on Esc) is handled at `match.ts:96-104`.

### Codex — the authoritative Rust+ratatui peer (`codex-rs/tui`)
- **Mouse capture is the selection-vs-scroll tradeoff.** Native selection requires NOT capturing the mouse. Codex ships `/toggle-mouse-mode` and `/raw` (Alt+R) so the user reclaims terminal selection; alt-screen control via `--no-alt-screen` / `tui.alternate_screen`. (issues #1247, #2836, #16149; PR #8555.)
- **Copy:** ONE `/copy` command + ONE hotkey **`Ctrl+O`** (PR #16966). Ctrl+O chosen over Alt+C *deliberately*: "Option+C may be consumed/transformed before the TUI sees Alt+C; Ctrl+O is a stable control chord in Terminal.app, iTerm2, SSH." Copies the **latest completed** assistant turn (bounded ordinal-indexed history `agent_turn_markdowns`), markdown-hardened.
- **Clipboard backend:** `clipboard_copy.rs`, strategy-injection `copy_to_clipboard_with(osc52_fn, arboard_fn, wsl_powershell_fn)` — fully unit-testable; arboard primary on Windows, PowerShell fallback, OSC52 last; Linux keeps a `ClipboardLease` alive so the X11/Wayland owner doesn't drop. **tui_v4's copy.rs already mirrors this design** (OSC52→arboard, `plan_copy` is the pure injectable core, copy.rs:188-248) — it just lacks the WSL/PowerShell arm and the live-owner lease.
- Known regression #16149: capturing the mouse makes "selection disappears on mouse release" in VS Code's terminal — **direct evidence** that tui_v4's forced `EnableMouseCapture` (F3) breaks copy.

---

## Fix design (Rust sketches: the actual changed lines / new fn signatures)

### Fix B — the Left/Right swap (Q5) — `main.rs:702-707`
Swap the verbs so empty-composer **Left = session view**, **Right = conversation view**:
```rust
// EMPTY composer: ← enters the session dashboard, → returns to chat.
KeyCode::Left  if app.composer.is_empty() && !shift => { app.open_dashboard(); }
KeyCode::Right if app.composer.is_empty() && !shift => { app.close_dashboard(); }
```
Dashboard side (`main.rs:919-934`) — keep tree collapse/expand but flip which arrow EXITS so it mirrors the cockpit. Left should *return to chat* when not on an expanded category; Right is the no-op-at-root direction:
```rust
KeyCode::Left => {                     // collapse an expanded category, else BACK… no:
    // after the swap, ← is "enter sessions" from chat, so inside the dashboard ←
    // should DRILL/collapse and → should EXIT back to chat:
    match app.sessions.selected_category() {
        Some(cat) if !app.sessions.is_collapsed(cat) => app.sessions.toggle_category(cat),
        _ => { /* stay; Right is the exit now */ }
    }
}
KeyCode::Right => {                     // → returns to the conversation view (mirror of cockpit)
    match app.sessions.selected_category() {
        Some(cat) if app.sessions.is_collapsed(cat) => app.sessions.toggle_category(cat),
        _ => app.close_dashboard(),
    }
}
```
Update the doc comment at `main.rs:696-701` (it currently states the wrong polarity: "→ enters the session dashboard, ← returns to chat"). Update the test `right_click_returns_from_dashboard` is mouse-only so unaffected; ADD a key test asserting `Left` on empty composer → `View::Dashboard`.

### Fix C — copy redesign: native drag-select ON by default (mouse capture OFF)
1. **Startup (main.rs:2273):** drop `EnableMouseCapture` so the terminal owns selection:
```rust
execute!(stdout, EnterAlternateScreen)?;   // mouse capture OFF by default → native drag-select works
```
2. **`AppState::new()`** (app/mod.rs): default `mouse_capture: false` (F7) so the field is the single source of truth.
3. **Footer hint:** surface the discoverable toggle — "select+copy: drag · scroll: /mouse or Ctrl+Shift+M". The `/mouse` command (registry.rs:83) + Ctrl+Shift+M (main.rs:555-561) already toggle capture ON for wheel-scroll — keep BOTH (they're the Codex `/toggle-mouse-mode` analogue). Re-label: capture ON = "scroll mode", OFF = "select mode".
4. **Keep PageUp/PageDown + Ctrl+Home/End scroll** (main.rs:722-732) so scrolling survives with capture OFF (Codex's "use the keyboard shortcuts we provide to scroll").

### Fix C-newline — wire Ctrl+Enter (Q6) — `main.rs:679`
```rust
// Shift+Enter OR Ctrl+Enter inserts a newline; plain Enter submits. (Ctrl+J is the
// universal fallback for terminals that deliver neither, main.rs:663.)
KeyCode::Enter if shift || ctrl => { app.composer.newline(); }
KeyCode::Enter => { let action = app.composer.submit(); dispatch_action(app, tx_bridge, action); }
```
Guard: this must sit AFTER the `try_complete_dropdown` call (main.rs:672) so `Enter` still completes the palette — it does (the dropdown intercept returns early for plain Enter; ctrl/shift Enter won't hit it because `try_complete_dropdown`'s Enter arm is `!shift` only — verify it also ignores ctrl, see Open Q2).

### Fix D — ONE explicit clean-copy action, correctly named (Q2)
Remove F2 (the ctrl+shift+c=copy-reply binding, main.rs:549-552). Bind the explicit clean-copy to **Ctrl+O** (Codex's stable-chord choice), copying the last assistant reply via the existing P2-clean path:
```rust
// Ctrl+O — copy the LAST completed assistant reply to the clipboard (clean logical
// source, no soft-wrap \n; Codex's Ctrl+O). Arbitrary-span copy is the terminal's
// job (drag-select, capture OFF). This is the one convenience copy action.
KeyCode::Char('o' | 'O') if ctrl => { export_action(app, 0); return; }
```
…and move fold off Ctrl+O to a `/fold` app-command (or Ctrl+Shift+O, free today). `export_action(app,0)` already routes through `copy_text`→`copy_to_clipboard` (main.rs:2003-2008, 2166-2172) which copies `last_assistant_source()` verbatim — newline-clean by construction (copy.rs P2). The `notice_copy` feedback (main.rs:2178-2197) already reports byte count, satisfying "visible feedback". For markdown TABLES specifically: `block.source` holds the un-rendered markdown table (pipes + dashes), so copying it is *already* clean and re-pasteable — better than v2's passthrough.
- Optional hardening: apply `wrap_for_clipboard`-style fence-escaping to `/export file` only (not clipboard), matching `export_cmd.py:15-24`.

### Fix E — the keymap is a 200-line match block → extract a declarative table (principle #6/#5)
Not required for the four bugs, but the de-dup work (Q6) is unsafe while bindings are scattered across `handle_key_event` (main.rs:539-668), `try_complete_dropdown`, `handle_dashboard_key`, `handle_workflows_key`, `handle_overlay_key`. Minimal viable step: a `const CHORDS: &[(KeyCode, KeyModifiers, ChordAction)]` table + a single resolver, mirroring CC's `DEFAULT_BINDINGS`. Defer the full refactor; cite as the structural root of why Ctrl+O/Ctrl+Y/Ctrl+Shift+C collisions went unnoticed.

---

## v2→v4 keybinding parity table + GAPS

| Action | v2 binding | tui_v4 today | Status / Fix |
|---|---|---|---|
| Stop / quit / copy-sel (3-stage) | `ctrl+c` (priority) | `ctrl+c` (3-stage, main.rs:562-565,2073) | ✓ parity (richer) |
| Quit (unambiguous) | — | `ctrl+q` (main.rs:566-569) | ✓ v4 extra |
| New session | `ctrl+n` (main.rs:584) | `ctrl+n` | ✓ |
| Toggle sidebar | `ctrl+b` | `ctrl+b` = **branch** (main.rs:594) | **GAP/CONFLICT** — v2 ctrl+b=sidebar; v4 ctrl+b=branch. v4 has no sidebar (dashboard replaces it). Keep v4 (branch), document divergence. |
| Fold | `ctrl+o` (2520) | `ctrl+o` (main.rs:611) | Moving to copy (Fix D); fold→`/fold` |
| Prev / next session | `ctrl+up`/`ctrl+down` | `ctrl+up`/`ctrl+down` (main.rs:602-609) | ✓ |
| Drop session | `ctrl+d` | `ctrl+w`/`ctrl+d` (main.rs:589) | ✓ (v4 adds ctrl+w) |
| Help | `ctrl+/`,`ctrl+_`,`cmd+/` | — (only `/help` command) | **GAP** — add `ctrl+/` (+ legacy `ctrl+_`) → open Help overlay |
| Escape (back / queue-clear) | `escape` (cascade) | `escape` (back+esc-esc, main.rs:733-738) | partial (F8) — add palette-dismiss + pending-queue clear |
| Complete command | `tab` (priority) | `tab` (try_complete_dropdown, main.rs:672) | ✓ |
| Theme picker | `ctrl+t` (2533) | — (only `/theme`) | **GAP** — add `ctrl+t` → `/theme` |
| Newline | `ctrl+j`+`ctrl+enter`+`shift+enter` | `ctrl+j`+`shift+enter` (main.rs:663,679) | **GAP** — add `ctrl+enter` (Fix C-newline) |
| Paste | `ctrl+v`/`cmd+v` | `ctrl+v` (main.rs:651) | ✓ |
| Clear input | `ctrl+u` | `ctrl+u` kill-to-line-start (main.rs:636) | ~ semantic drift (v2 clears WHOLE input; v4 kills to line start). Acceptable (v4 richer editing) — note it. |
| Stash draft | `ctrl+s`/`cmd+s` | `ctrl+g` (main.rs:659); `ctrl+s`=dashboard | **DIVERGENCE** — v4 repurposed ctrl+s for the session dashboard, moved stash to ctrl+g. Intentional (§6). Document. |
| Select-all / EOL / cut / undo / redo | — | `ctrl+a`/`ctrl+e`/`ctrl+x`/`ctrl+z`/`ctrl+y` (main.rs:626-649,616-624) | ✓ v4 extra (readline) |
| Picker submit / cancel | `enter`/`escape` | `enter`/`esc` (main.rs:1049,1080,1195) | ✓ |
| Picker nav | `up`/`down`; OptionList `right`=select/`left`=cancel | `up`/`down` (overlay), arrows | ✓ (v4 uses up/down uniformly) |
| Copy selection (terminal) | drag + `ctrl+c`/`ctrl+shift+c` | **broken** (capture ON, F3) | **BLOCKER** — Fix C (capture OFF) |
| Copy last reply (explicit) | `/export` | `ctrl+shift+c` (mis-named, F2) | re-bind → `ctrl+o` (Fix D) |

**GAPS to ADD (net):** `ctrl+/` (+`ctrl+_`) Help · `ctrl+t` theme · `ctrl+enter` newline. **RE-BIND:** `ctrl+shift+c` removed; `ctrl+o` = clean-copy; fold → `/fold`. **DIVERGENCES to document (intentional, not gaps):** ctrl+b=branch, ctrl+s=dashboard (stash→ctrl+g), ctrl+u=kill-to-line-start.

## Command + keybinding de-dup list (Q6, principles #5/#10/#12)
`COMMANDS` (registry.rs:46-87) is 38 entries; v2's handler map (`tuiapp_v2.py:2574-2592`) is ~24. Redundant **aliases to drop** (registry plus their dispatch arms):
- `sessions` (registry.rs:49) — pure alias of `status` (both → `Overlay::Status`, main.rs:1631). **Drop.**
- `abort` (registry.rs:58) — alias of `stop` (identical arm, main.rs:1855). **Drop.**
- `tools` + `trace` (registry.rs:80-81) — both alias `verbose` → `Overlay::Verbose` (main.rs:1738). **Drop both**, keep `verbose`.
- `exit` (registry.rs:86) — alias of `quit` (main.rs:1846). **Drop** (keep `quit`; v2 keeps both but they're free strings — in v4 each costs a palette row + a test assertion).

Net: 38 → **33** commands, matching the "~33" the registry's own header (registry.rs:44) and test (registry.rs:290-292) claim — the aliases are what inflate it to 38. Update `registry_resolves_all_commands` (registry.rs:272-292) `all[]` + the `assert_eq!(COMMANDS.len(), all.len())`.
Keyboard de-dup: after Fix D there is exactly ONE clean-copy chord (Ctrl+O) and ONE selection path (terminal). Remove the ctrl+shift+c arm (main.rs:549-552). Keep Ctrl+J as the documented universal newline fallback (not a dup — it's the portable floor for terminals lacking shift/ctrl+enter).

---

## Review-principle violations (cite principle # + file:line)

- **#2 local reasoning / #6 constraints-in-types** — `main.rs:2273` (`EnableMouseCapture`) sets terminal state that `app.mouse_capture` (the field the UI reads) doesn't know about (F7). True state lives in an `execute!` call, not the type. The whole keymap being a 200-line `match` (main.rs:539-744) means a chord's meaning can't be reasoned about locally — Ctrl+O appears once for fold, the copy intent lives 60 lines up in the ctrl+shift+c arm. CC proves the fix: a declarative `KeybindingBlock[]` (defaultBindings.ts:32).
- **#8 consistent & unsurprising** — `main.rs:702-707`: ← and → do the *opposite* of the documented and intuitive direction (F1). The doc comment (main.rs:696-701) asserts the correct behavior while the code does the inverse — actively misleading.
- **#5 linear complexity / #12 more-features-fewer-lines** — registry.rs:46-87: 5 alias commands (`sessions`/`abort`/`tools`/`trace`/`exit`) add rows + dispatch arms + test entries for zero new capability. Each alias is a branch in `open_ui_command`/`app_command` (main.rs:1631,1738,1846,1855).
- **#15 length tracks function** — `render/copy.rs` is 530 lines, ~half of it (`join_visual_rows` :162, `CopyOverlay` :329, much of `plan_copy`'s remote branches) is `#[allow(dead_code)]` — defensive machinery for a copy-mode/drag-select feature that was never wired. The actually-used path is `copy_to_clipboard`→`build_osc52`→`native_copy`. The dead surface dwarfs the live one (principle #15 inverted). Either wire `join_visual_rows` to a real in-app selection (Fix C makes the terminal do it instead → then DELETE the dead code) or delete it now.
- **#9 minimal comments** — main.rs:546-561 carries a 16-line comment block justifying the ctrl+shift+c arm that Fix D deletes; the justification is itself the smell (it's arguing for a design the user rejected).

---

## Open questions / risks

1. **Ctrl+O reassignment (F6/Fix D):** taking Ctrl+O for copy moves fold off its v2-parity key. Two options: (a) Ctrl+O=copy (align Codex, the closer Rust+ratatui peer), fold→`/fold` cmd; (b) keep Ctrl+O=fold (v2 parity), copy→Alt+C (risks the very terminal-eats-Alt problem Codex avoided). Spec recommends (a). **Needs a one-line decision from the owner.**
2. **`try_complete_dropdown` + Ctrl+Enter:** the dropdown Enter arms are `KeyCode::Enter if !shift` (main.rs:1512,1538) — they do NOT exclude `ctrl`, so a Ctrl+Enter while the palette is open could be swallowed as a completion instead of inserting a newline. Verify and, if so, also guard those arms with `&& !ctrl`.
3. **Mouse capture OFF loses click-to-open-dashboard (main.rs:475-483) and click-a-session-row (main.rs:492-508).** With capture OFF the app won't get those clicks. Mitigation: the keyboard path (empty-composer ← after Fix B, Ctrl+S) fully covers dashboard entry; document that clicks require `/mouse` ON. This is the same tradeoff Codex accepts.
4. **Windows terminal modifier delivery:** shift+enter / ctrl+enter / ctrl+shift+c reach the app only on VT-mode terminals (CC's `SUPPORTS_TERMINAL_VT_MODE`, defaultBindings.ts:21-30). On legacy conhost they silently no-op. Ctrl+J (newline) and drag-select (copy) are the always-works floors — keep both. Env here is Windows 10, so this is live, not theoretical.
5. **OSC52 vs native after capture-off:** unchanged — copy.rs already prefers OSC52 (SSH-safe) then arboard. No WSL/PowerShell arm (Codex has one); low priority unless WSL users hit it.
