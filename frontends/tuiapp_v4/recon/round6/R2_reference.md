# R2 — Design Reference for tui_v4 (3 problems), cited from Codex Rust TUI + Claude Code

Sources:
- Codex Rust TUI: `D:/GenericAgent/temp/codex_src/codex-rs/tui/src/*.rs`
- Claude Code (decoded): `D:/GenericAgent/temp/claude-code/src/**` and `D:/GenericAgent/frontends/tuiapp_v4_research/recon/_*.tsx`
- tui_v4 (the thing being fixed): `D:/GenericAgent/frontends/tuiapp_v4/src/components/cockpit/*.rs`

---

## PROBLEM 1 — CONTENT-AT-TOP (kill the blank gap above the transcript)

### How Codex does it (and why it never has this bug)
Codex does NOT run a full-screen alt-screen for the main loop. It uses an **inline viewport** sized to the content tail, and pushes finished history into the terminal's own scrollback (which naturally flows top→down).

- `tui.rs:329` — `init()` comment: "inline viewport; history stays in normal scrollback".
- `tui.rs:651-672` — `insert_history_lines*()` writes finished lines ABOVE the viewport into scrollback (`flush_pending_history_lines` → `insert_history::insert_history_lines_*`, `tui.rs:785-804`).
- `tui.rs:682-713` — `update_inline_viewport()` sizes the viewport to `height` and keeps it **bottom-aligned** (`area.y = size.height - area.height`); only the small active tail (active_cell + composer) lives in the viewport.
- `app.rs:1127-1139` — the draw height is just the tail: `let desired_height = self.chat_widget.desired_height(width); tui.draw(desired_height, |frame| …)`. Codex asks for *only as many rows as the active content needs*, so there is structurally no blank gap — the OS terminal supplies the scrollback above.

**Do NOT port the inline-viewport model to tui_v4.** tui_v4 is a full-screen alt-screen app by design (mouse capture, sticky header, rainbow separator). The lesson to port is: *the transcript region should grow from the top and let any slack fall at the bottom*, the same visual result Codex gets for free.

### The tui_v4 bug is self-inflicted (bottom-anchor offset)
The layout slot is already correct — transcript is `Constraint::Min(0)` so it flexes to fill the space between the rainbow separator and the spinner/composer:
- `cockpit/mod.rs:139` — `constraints.push(Constraint::Min(0)); // transcript (FLEX)`.

The gap is created *inside* the transcript draw by an explicit DOWN-shift:
- `cockpit/transcript.rs:107-125` — "SLICE S2 — bottom-anchor": when `following() && total < h`, it computes `gap = h - total` and renders into `Rect { y: area.y + gap, height: area.height - gap, .. }`. That deliberately pushes content to the bottom of the slot, leaving `gap` blank rows ABOVE — exactly the reported symptom.

### Recommendations (P1)
- **Replace bottom-anchor with top-hug**: in `transcript.rs:112-125`, when `total < h` render into `area` unchanged (content starts at `area.y`); the slack rows fall at the BOTTOM of the transcript slot, immediately above the spinner/composer. i.e. delete the `y: area.y + gap` branch and just use `area`. `Paragraph::new(lines)` already draws top-down, so no scroll math is needed for the short case.
- **Keep the slot as `Constraint::Min(0)`** (mod.rs:139) — the vertical SPLIT is right; only the intra-slot placement was wrong. Do not move the transcript above the header/separator.
- **For the overflow (scrolled) case, keep current windowing** — when `total >= h` the viewport window already fills the slot (`app.viewport.visible`), so no offset is needed there either (the existing `else { area }` branch is correct and stays).
- **Pin the latest user prompt at the top when scrolled up** — Codex/tui_v4 both already do a sticky top breadcrumb (`mod.rs:124-125,136-138` `show_sticky`); keep it. This is the right "content anchored to top" affordance.

---

## PROBLEM 2 — MODEL INTERMEDIATE OUTPUT (tool calls, reasoning, results) — no duplicated breadcrumb

### The canonical Codex pattern: ONE header line, command/title INLINE, result under a single `└`
`exec_cell/render.rs` `command_display_lines()` (`render.rs:365-508`) is the reference. There is no separate title row + repeated breadcrumb. Structure:

1. **Header line** = `bullet + " " + TITLE + " " + <command inline>`:
   - `render.rs:387-391` builds `• Ran ` / `• Running ` / `• You ran `, then `render.rs:407-417` appends the FIRST wrapped command segment **onto the same header line** (`header_line.extend(first_segment)`).
   - Bullet state: `render.rs:371-375` — green `•` (success), red `•` (failure), animated activity marker while running.
   - Confirmed by test `render.rs:986`: `"• Running echo done"` is a **single line**.
2. **Command continuation** (only if the command wraps) uses a `│ ` gutter, capped at 2 lines:
   - `render.rs:706-711` `EXEC_DISPLAY_LAYOUT`: `command_continuation = PrefixedBlock::new("  │ ", "  │ ")`, `command_continuation_max_lines = 2`.
3. **Output/result block** uses `└ ` ONCE on the first row, then 4-space indent:
   - `render.rs:709` `output_block = PrefixedBlock::new("  └ ", "    ")` — `└ ` is the initial prefix, `    ` (spaces) is every subsequent row. So `└` marks the START of the result, it is NOT repeated and it does NOT echo the title.
   - Output is truncated by **viewport rows** (not logical lines) with a middle ellipsis: `render.rs:539-630` `truncate_lines_middle`, default `output_max_lines = 5` (`render.rs:710`), ellipsis text `… +N lines (ctrl + t to view transcript)` (`render.rs:255`).

### MCP tool calls — same shape, inline-or-fold header
`history_cell.rs` `McpToolCallCell::display_lines` (`history_cell.rs:1868-1962`):
- Header: `bullet + "Calling"/"Called" + <invocation>` inline when it fits (`history_cell.rs:1893-1898`).
- If the invocation is too wide, it drops to its own line and the invocation wraps under `  └ ` / `    ` (`history_cell.rs:1900-1909`).
- The RESULT (tool output / error) is the `└`-prefixed block (`history_cell.rs:1952-1959`): `└ ` only when the header was inline, else plain indent — again, `└` introduces the *result*, never a copy of the header.

### Reasoning / thinking
- Streaming reasoning: `ReasoningSummaryCell` (`history_cell.rs:466-536`) renders the reasoning markdown **dim + italic**, bulleted `• ` initial / `  ` subsequent (`history_cell.rs:494-512`). It can be `transcript_only` (hidden from main view, shown only in the Ctrl+T overlay) — `history_cell.rs:518-527`.
- Finalized reasoning summary: `new_reasoning_summary_block` (`history_cell.rs:3208+`) extracts the bold `**…**` header from the buffer and injects a compact summary; the long body lives in the transcript overlay, not inline.
- Plan updates (a structured "thinking" surface): `PlanUpdateCell` (`history_cell.rs:3059-3110`) — header `• Updated Plan`, steps under a SINGLE `  └ ` block with checkbox glyphs `✔` (done, crossed-out dim) / `□` (in-progress cyan bold / pending dim) (`history_cell.rs:3070-3085`).

### Claude Code parallels
- Bash tool: header is the command itself (truncated to 2 lines / 160 chars, `BashTool/UI.tsx:26-27,104-129`); progress and result render as child `MessageResponse` rows (`UI.tsx:131-173`) — never a re-print of the command as a breadcrumb.
- Agent/teammate rows: bullet + name + truncated summary + ` · ` separated suffix metadata (`_CoordinatorAgentStatus.tsx:228-245`).

### The tui_v4 anti-pattern and its fix
tui_v4 today renders `▾ <title>` and then a separate `↳ <text>` line containing the **same** text. Codex never re-states the title: the header carries the title (+ inline command), and the only indented child block (`└`) carries the RESULT/OUTPUT, not the title.

### Recommendations (P2)
- **Collapse header + first detail into ONE line**: render `<bullet> <Title> <inline summary/command>` on a single line (Codex `render.rs:387-417`). Do not emit a second line that repeats the title text.
- **Make the indented child be the RESULT, not a breadcrumb**: the `↳`/`└` block should contain tool output / error / file path — content that is NOT in the header (Codex `render.rs:486-503`, `history_cell.rs:1952-1959`). If there is no distinct detail, render no child line at all.
- **Use a single tree glyph, once**: `└ ` (U+2514 + space) as the INITIAL prefix only; subsequent wrapped rows use plain 4-space indent — never repeat `└` per row (Codex `render.rs:709`, status_indicator `DETAILS_PREFIX = "  └ "` at `status_indicator_widget.rs:35,205-206`).
- **Cap detail rows + ellipsis**: limit inline output to ~5 rows with a middle/`… +N lines` ellipsis, full text behind the expand/overlay (Codex `render.rs:710`, `truncate_lines_middle` `render.rs:539-630`).

---

## PROBLEM 3 — EXPAND / COLLAPSE affordance (caret + summary/detail)

### Codex: there is NO inline ▾/▸ toggle on individual cells
Codex's model is **summary-inline, full-detail-in-overlay**, not per-cell expand carets:
- Each `HistoryCell` exposes `display_lines()` (compact, capped) vs `transcript_lines()` (full) — `history_cell.rs:158,194`. The compact view is what shows in the main viewport; the full view is shown in the **Ctrl+T transcript overlay** (`history_cell.rs:189-196`, comment `tui.rs`/`history_cell.rs:7`).
- The collapse affordance is the truncation ellipsis itself, which doubles as the "where's the rest" hint: `… +N lines (ctrl + t to view transcript)` (`exec_cell/render.rs:35,255`). The "expand" gesture is `Ctrl+T`, not a caret on the row.
- Status bullets encode state, not expansion: animated activity marker while running → solid `•` green/red on completion (`render.rs:371-375`, `history_cell.rs:1872-1881`). Exec verbs flip `Running`→`Ran`, `Exploring`→`Explored` (`render.rs:271-275,377-385`).

### Claude Code: explicit caret/pointer + "ctrl+o to expand"
- Selection / focus pointer uses `figures.pointer` (`›`/`▶`) vs 2-space when not selected, and a filled vs hollow bullet (`BLACK_CIRCLE` ● vs `figures.circle` ○) to mark the "viewed/active" item — `_CoordinatorAgentStatus.tsx:143-144,204-206`.
- CC has a dedicated "expand" hint component `CtrlOToExpand.tsx` (path: `temp/claude-code/src/components/CtrlOToExpand.tsx`) — i.e. CC also uses a **keyboard verb to expand**, surfaced as a dim hint, rather than a clickable triangle per row.

### tui_v4 reality (already has folds + ▾/▸ + click)
tui_v4 already implements a per-block fold model with carets and mouse toggling (`render::BlockFolds`, `app.folds`, `fold_all` — `transcript.rs:35,49-54`; round-5 added `▸⇄▾` click-to-expand). So the convention to KEEP and align is:

### Recommendations (P3)
- **Caret semantics (keep tui_v4's, matches conventions): `▸` = collapsed (has hidden detail), `▾` = expanded.** Put the caret in the HEADER line, immediately before the title — never on a separate row. (CC uses `figures.pointer ›/▶` for selection and a keyboard verb for expand; the triangle pair `▸/▾` is the standard disclosure convention and is fine to keep.)
- **Collapsed state shows a one-line summary; expanded shows the `└` detail block** — when collapsed, render only the header (`<caret> <bullet> <Title> <inline summary>`); when expanded, append the capped `└`-prefixed result block (Problem 2). This mirrors Codex compact-vs-transcript split (`history_cell.rs:158/194`) but inlined behind the caret instead of a separate overlay.
- **Always advertise the hidden content**: when collapsed (or truncated), end the visible region with a dim `… +N lines` / `▸ N more` hint so the affordance is discoverable, exactly like Codex's ellipsis hint (`render.rs:255`) and CC's `CtrlOToExpand`. Provide BOTH a click target (round-5 mouse toggle) and a key (e.g. Ctrl+T/Ctrl+O style) so it works without mouse capture.
- **Do not put the caret AND a duplicate `↳` title** — the round-5/round-6 bug. One caret in the header, one `└` block of genuine detail. (Anti-pattern source: the duplicated `▾ title` + `↳ <same text>` in current tui_v4.)

---

## Quick glyph / prefix cheat-sheet (from Codex)
- Status bullet: `•` (dim idle / green ok / red fail), animated marker while active — `render.rs:371-375`.
- Header→detail tree: detail block starts with `  └ ` then `    ` (4 spaces) continuation — `render.rs:709`, `status_indicator_widget.rs:35`.
- Command wrap gutter: `  │ ` — `render.rs:707`.
- Plan checkboxes: `✔` done (crossed-out dim), `□` in-progress (cyan bold) / pending (dim) — `history_cell.rs:3072-3074`.
- Truncation/expand hint: `… +N lines (ctrl + t to view transcript)` — `render.rs:255`.
- Spinner status line: `<spinner> Working (<elapsed> • esc to interrupt)` + optional ` · <inline ctx>`, with `└ ` detail rows below (max 3) — `status_indicator_widget.rs:247-289,205`.
- Spinner token-direction arrow (`↑ in` / `↓ out` analogue): derive from last-activity, `figures.arrowDown` when receiving / `figures.arrowUp` otherwise; suffix ` · <arrow> N tokens` — `_CoordinatorAgentStatus.tsx:189-195`.
