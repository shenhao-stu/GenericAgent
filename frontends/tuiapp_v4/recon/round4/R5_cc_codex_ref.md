# R5 — CC + Codex reference patterns for tui_v4 (tool-call rendering + spinner-with-tip)

Source dirs both present and decoded:
- CC: `D:/GenericAgent/temp/claude-code/src/` (TypeScript/Ink, React-compiler output — `_c(n)` memo caches; the original `.tsx` is recoverable from the inlined sourcemap `sourcesContent`)
- Codex: `D:/GenericAgent/temp/codex_src/codex-rs/tui/src/` (Rust / ratatui)

Both implement the SAME two visual idioms we want for tui_v4. Neither draws a full box border around tool calls. Both use a **bullet header + hanging `└` gutter** for results/details. This is the dominant pattern to copy.

---

## 1. TOOL-CALL RENDERING

### 1a. Claude Code — `AssistantToolUseMessage.tsx`

File: `D:/GenericAgent/temp/claude-code/src/components/messages/AssistantToolUseMessage.tsx`

CC renders a tool invocation as a **column** with two regions: a one-line header row, then 0+ hanging result/progress lines. There is **no border** — indentation/affiliation is conveyed by a status dot and the `⎿` corner glyph.

Outer wrapper (line 285):
```
<Box flexDirection="row" justifyContent="space-between" marginTop={addMargin?1:0} width="100%" backgroundColor={bg}>
  <Box flexDirection="column">   // {t15}
```

Header row (`t12`, line 228):
```
<Box flexDirection="row" flexWrap="nowrap" minWidth={width(name)+ (dot?2:0)}>
  {t7}   // status dot: <ToolUseLoader> (animated ●) OR ● BLACK_CIRCLE when queued
  {t9}   // <Text bold wrap="truncate-end">{userFacingToolName}</Text>     (line 200)
  {t10}  // renderedToolUseMessage !== "" && <Text>({renderedToolUseMessage})</Text>  (line 210)
  {t11}  // tool-specific tag (timeout/model/resume id) via tool.renderToolUseTag()
</Box>
```
So the header reads:  `●  ToolName(arg summary) [tag]` — dot, bold name, parenthesized arg summary, optional tags. Name + args come from the tool object: `tool.userFacingName(data)` and `tool.renderToolUseMessage(data,…)` (lines 77, 318). The dot color/animation is `ToolUseLoader` (`D:/GenericAgent/temp/claude-code/src/components/ToolUseLoader.tsx`): unresolved = dim `●`, error = `error` color, success = `success` color; blink while animating. Queued = static dim `●` (BLACK_CIRCLE).

Progress / result region (line 240): below the header, when not resolved and not queued, CC renders `renderToolUseProgressMessage(...)`, or `Waiting for permission…`, etc. — each wrapped in `<MessageResponse>`. The finished tool RESULT is rendered by the sibling `UserToolResultMessage` family (`messages/UserToolResultMessage/*.tsx`), also wrapped in `MessageResponse`.

The hanging-continuation glyph (the load-bearing detail) is in `MessageResponse.tsx:22`:
```
<NoSelect fromLeftEdge flexShrink={0}><Text dimColor>{"  "}⎿  </Text></NoSelect>
```
i.e. `MessageResponse` prefixes its child with **two spaces + `⎿` + two spaces**, dim. A React context (`MessageResponseContext`, line 62) suppresses nested `⎿` so sub-steps under an already-`⎿`'d line don't double the corner — they just indent. This is exactly the "collapsible result hangs under the call" behavior.

Nesting / sub-steps: indentation is purely the `⎿` gutter (2-space indent + corner on first line, plain indent after). There is no recursive box; depth is flat (one `⎿` level), and `CollapsedReadSearchContent.tsx` / `GroupedToolUseContent.tsx` collapse repeated tool calls (e.g. many Reads) into one summarized block rather than nesting.

### 1b. Codex — `exec_cell/render.rs` (Bash/exec) and `history_cell.rs` (generic/MCP tools)

Codex uses an identical bullet-header + `└`-gutter idiom, expressed as a flat `Vec<Line>` with **string prefixes** rather than nested widgets. The prefixes are the design system.

Exec/Bash tool call — `D:/GenericAgent/temp/codex_src/codex-rs/tui/src/exec_cell/render.rs`, `command_display_lines()` (lines 365-508):
- bullet (line 371-375): `•` — `activity_marker()` (animated) while running, `•.green().bold()` on success, `•.red().bold()` on failure.
- header line (lines 387-391): `bullet + " " + title + " " + <command>` where `title` ∈ {`Running` (active), `Ran` (agent done), `You ran` (user shell), `""` (interaction)}. First wrapped segment of the command is appended onto the header line; remainder spills to continuation.
- The layout constants are the canonical spec — `EXEC_DISPLAY_LAYOUT` (lines 706-711):
  ```
  command_continuation: PrefixedBlock("  │ ", "  │ ")   // wrapped command tail, max 2 lines
  output_block:         PrefixedBlock("  └ ", "    ")    // result; first line "  └ ", rest "    "
  output_max_lines:     5
  ```
- output rendering (`output_lines`, lines 103-184; applied 442-504): dim, prefixed with `  └ ` on the first row and `    ` on following rows, truncated head+tail with a middle `… +N lines (ctrl + t to view transcript)` ellipsis (`TOOL_CALL_MAX_LINES = 5`, line 32). `(no output)` shown when empty (line 466).

Generic / MCP tool call — `D:/GenericAgent/temp/codex_src/codex-rs/tui/src/history_cell.rs`, `McpToolCallCell::display_lines()` (lines 1868-1962):
- bullet (1872-1881): same `•` active→green/red scheme via `activity_indicator`.
- header (1889): `bullet + " " + ("Calling" active | "Called" done) + " " + invocation`.
- invocation format — `format_mcp_invocation()` (lines 3410-3429): `server.cyan() + "." + tool.cyan() + "(" + args.dim() + ")"`, args = compact JSON. So: `● Called server.tool({"k":"v"})`.
- inline-vs-wrapped (1893-1909): if the invocation fits on the header line it inlines; otherwise the header stands alone and the invocation wraps under a `"  └ "` / `"    "` prefix.
- result/detail (1911-1959): each content block is dim, wrapped, then prefixed with `prefix_lines(detail_lines, "  └ "/"    ")`. Errors render `Error: …` dim (1934-1947). `TOOL_CALL_MAX_LINES` truncation applies (line 1852).

The shared prefix helper — `D:/GenericAgent/temp/codex_src/codex-rs/tui/src/render/line_utils.rs:41` `prefix_lines(lines, initial_prefix, subsequent_prefix)` — pushes `initial_prefix` onto line 0 and `subsequent_prefix` onto the rest. This single function + the `PrefixedBlock` struct (render.rs:661-680, with `wrap_width()` that subtracts the prefix width) is the entire "indented block" mechanism. Web-search cells, exploring/Read-grouping cells (render.rs:262-363, `Read`/`Search`/`List`/`Run` verbs in cyan) all reuse `prefix_lines` with `"  └ "`.

### Tool-call takeaways for tui_v4
- Do NOT box tool calls. Use: `<dot> <Verb> <name/target>(<arg-summary>)` header + a hanging `└ ` first line / 2-space subsequent indent for the collapsible result. (CC `⎿` U+23BF; codex `└` U+2514 — pick one and standardize; codex's `└` reads cleaner in most monospace fonts.)
- Dot/bullet is the status channel: animated while running, green `•`/`✓` on success, red `•`/`✗` on error, dim when queued. Keep it 2 cells wide so the header text never reflows when state flips.
- Verb is state-dependent: Running→Ran, Calling→Called, Searching→Searched, Exploring→Explored. Past tense on completion is a cheap, strong "done" signal.
- Truncate results to a small head/tail (codex: 5 lines) with a `… +N lines (… to view full)` middle ellipsis; compute the budget in **rendered rows after wrap**, not logical lines (codex `truncate_lines_middle`, render.rs:539). Keep a per-source override (codex gives user-run shell 50 lines vs 5 for agent tools).
- Collapse repeated homogeneous calls (CC `CollapsedReadSearchContent`; codex Read-run merging in `exploring_display_lines`) instead of stacking N near-identical blocks.

---

## 2. SPINNER STATUS LINE + HANGING `└ Tip:`

### 2a. Claude Code — the exact composition

Two files compose it: the outer column (status line + hanging tip) is `Spinner.tsx`; the status line itself is `SpinnerAnimationRow.tsx`.

Outer column — `D:/GenericAgent/temp/claude-code/src/components/Spinner.tsx`, `SpinnerWithVerbInner` return (lines 280-300):
```
<Box flexDirection="column" width="100%" alignItems="flex-start">
  <SpinnerAnimationRow … />                         // the status line (row 1)
  … (nextTask || effectiveTip || budgetText) ?      // row 2+ : the hanging continuation
    <Box width="100%" flexDirection="column">
      {budgetText && <MessageResponse><Text dimColor>{budgetText}</Text></MessageResponse>}
      {(nextTask || effectiveTip) &&
        <MessageResponse>
          <Text dimColor>{nextTask ? `Next: ${nextTask.subject}` : `Tip: ${effectiveTip}`}</Text>
        </MessageResponse>}
    </Box> : null}
</Box>
```
So the hanging tip is literally `Tip: ${effectiveTip}` (Spinner.tsx:296) wrapped in the SAME `MessageResponse` component as tool results → it gets the dim `  ⎿  ` corner from `MessageResponse.tsx:22`. That is the `└ Tip: …` hanging continuation directly below the status line. The tip text is chosen at lines 256-259: default `spinnerTip` prop, or time-triggered tips — after 30s with no `/btw` use → the `/btw` tip; after 30min → the `/clear` tip. A pending todo overrides it as `Next: …`. (Toggleable via `settings.spinnerTipsEnabled`.)

The status line itself — `D:/GenericAgent/temp/claude-code/src/components/Spinner/SpinnerAnimationRow.tsx`, `SpinnerAnimationRow` (lines 81-231):
- container (line 226): `<Box flexDirection="row" flexWrap="wrap" marginTop={1} width="100%">`
- children in order: `<SpinnerGlyph>` (the sparkle/asterisk frame), `<GlimmerMessage>` (the gerund verb with the shimmer sweep, e.g. `Forging…`), then `{status}`.
- `{status}` (lines 215-225) is the parenthesized metadata: `<Text dimColor>(</Text><Byline>{parts}</Byline><Text dimColor>)</Text>`.
- `parts` (lines 202-214) is assembled progressively and width-gated (lines 175-193) — each piece only shows if it fits:
  - `spinnerSuffix` (optional),
  - `timerText` = `formatDuration(elapsed)` (the elapsed dot),
  - tokens = `<SpinnerModeGlyph>` (a `↓` arrowDown / `↑` arrowUp glyph by mode, lines 232-263) + `"{N} tokens"` — only after 30s (`SHOW_TOKENS_AFTER_MS`, line 19) unless verbose,
  - thinking = `thinking{effortSuffix}` while thinking, or `thought for Ns` after (lines 171-172), with a sine-wave shimmer color (lines 198-200).
- `<Byline>` (`design-system/Byline.tsx:75`) joins the parts with a dim `" · "` middot separator.

Net rendered status line (matches the prompt's description exactly):
```
✷ Gerund… (0s · ↓ 1.2k tokens · thinking with max effort)
  ⎿  Tip: Use /btw to ask a quick side question without interrupting Claude's current work
```
- `✷` = SpinnerGlyph frame (animated sparkle/teardrop-asterisk).
- gerund = random spinner verb (`getSpinnerVerbs()`), or current-todo `activeForm`, with shimmer.
- `0s` = elapsed; `↓ N tokens` = down-arrow + token count; `thinking with max effort` = thinking + effort suffix (here `getEffortSuffix(model, effortValue)` produced `with max effort`).
- The `⎿ Tip:` line is the `MessageResponse`-wrapped continuation hanging under it.

Performance note worth copying: the 50 ms animation clock (`useAnimationFrame(50)`) lives ONLY in `SpinnerAnimationRow` (line 103). The parent `SpinnerWithVerbInner` is off the hot loop and only re-renders on prop/state change. Token counter is smoothed (lines 142-159). Elapsed/tip thresholds read stale refs (Spinner.tsx:205) — fine for coarse 30s/30min triggers.

### 2b. Codex — the direct analog

File: `D:/GenericAgent/temp/codex_src/codex-rs/tui/src/status_indicator_widget.rs`, `StatusIndicatorWidget::render` (lines 232-289).

Status line spans (lines 247-274):
```
[activity_indicator] " " shimmer_text(header) " " "({elapsed} • " <Esc hint> " to interrupt)" [" · " inline_message]
```
- `activity_indicator(...)` = animated spinner glyph (motion.rs), hidden under reduced-motion.
- `shimmer_text(&self.header, ...)` = the gerund/header with a shimmer sweep (default `"Working"`).
- `fmt_elapsed_compact` (lines 63-76) = `0s / 1m 00s / 1h 02m 03s`.
- interrupt hint uses `key_hint::plain(Esc)`.
- optional `inline_message` appended after the interrupt hint with a dim `" · "` (lines 269-273) — same middot idiom as CC's Byline.

Hanging details below — IDENTICAL to CC's `└ Tip:`:
- `DETAILS_PREFIX = "  └ "` (line 35).
- `desired_height` = `1 + wrapped_details_lines` (line 229) — the widget owns row1 (status) + N detail rows.
- `wrapped_details_lines` (195-224): `initial_indent("  └ ".dim())`, `subsequent_indent(spaces.dim())`, capped at `details_max_lines` (default 3, line 34) with a trailing `…` ellipsis.
So codex draws the same `<spinner> <header> (<elapsed> • esc to interrupt)` then a `  └ <details…>` hanging continuation — the structural twin of CC's status+tip. Verified by test `renders_without_spinner_when_animations_disabled` (line 405): `"Working (0s • esc to interrupt)"`.

### Spinner takeaways for tui_v4
- Compose as a 2-region vertical stack you OWN: row 1 = `<glyph> <gerund-with-shimmer> (<byline metadata>)`; rows 2+ = a dim `└ ` hanging continuation for the tip / details. Reuse the SAME hanging-`└` primitive as tool results so spacing is consistent everywhere (both CC and codex deliberately share it).
- Byline metadata = parts joined by dim `" · "`, each part width-gated so it drops gracefully on narrow terminals (order: suffix, elapsed, `↓ N tokens`, thinking/effort). Implement progressive gating (CC `availableSpace` math, SpinnerAnimationRow.tsx:175-193) rather than a fixed format string.
- Put the animation clock on the innermost status-row component only; keep the tip/parent off the per-frame loop; smooth the token counter; read elapsed/tip thresholds lazily.
- Tip content should be contextual & time-triggered (CC: 30s→`/btw`, 30min→`/clear`; replace with our own slash hints), overridable by a "Next: <todo>" when a plan/todo is active, and globally disable-able via settings.
- Glyph: CC uses an animated sparkle/asterisk (`SpinnerGlyph`, frames from `getDefaultCharacters`), codex an `activity_indicator`. Provide a reduced-motion fallback to a static `●`/`•` (both do: CC `Spinner.tsx:512`, codex `ReducedMotionIndicator::StaticBullet`).
- Elapsed format: copy `fmt_elapsed_compact` (`0s`/`1m 00s`/`1h 02m 03s`) — clean and width-stable.

### Splash (from screenshot `D:/Screenshots/clip_20260601_011003.png`)
CC v2.1.158 splash for cross-check: a single rounded-border box (Claude-orange `#~salmon` accent), title `Claude Code v2.1.158` inset into the top border, two-column interior (left: `Welcome back!` + pixel-crab mascot + `Opus 4.8 (1M context) · API Usage Billing` + cwd; right: bold `What's new` + dim release bullets + italic `/release-notes for more`), then a dim `Using <model> (from …settings.json) · /model to change` line below the box, then the `›` prompt. For tui_v4 this confirms: rounded border + inset title + 2-col + single accent color + dim secondary text. (Tool calls/spinner, by contrast, are NOT boxed — only the splash and dialogs are.)
