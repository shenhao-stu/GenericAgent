# C1 — Transcript / tool-call rendering, folding, realtime markdown+latex

Scope: Q3 (real-time markdown+latex, render edge-to-edge during streaming) and Q8 (expand-to-tool-call / fold-to-summary, **click the ▸ to expand ANY node**, resize/scroll-stable, kill every "Turn N").

Verdict up front: the v4 render plane is architecturally **correct** and is a true peer of Codex `codex-rs/tui` (pure `source → Vec<Line>`, per-block `rev`-keyed wrap cache, logical scroll anchor, newline/structure-gated stream commit, "Turn N" already stripped at finalized-render time). The deliverables fail on five concrete points, none of which require a rewrite:

1. **Inline LaTeX is destroyed by soft-wrap** (a multi-cell math glyph string is one `Span`, re-wrapped char-wise → split mid-formula). Root cause, not symptom.
2. **No streaming math/code/table holdback** — `safe_commit_pos` commits at `\n\n` and never holds back a half-built `$$…$$` / table / fenced block, so they flash wrong then snap. Codex has an explicit `TableHoldbackScanner`; v4 has nothing equivalent.
3. **Streaming renders are NOT incremental** — `render_assistant_cockpit_streaming` re-folds + re-parses the WHOLE source every keystroke with no memo; CC (`cachedLexer`) and Codex (`StreamCore.rendered_lines` rebuilt only on committed delta) both cache.
4. **Q8 "expand any node" does not exist** — fold is a single global `app.fold_all: bool` (Ctrl+O folds/unfolds *all* turns); there is no per-node state, no clickable ▸, no mouse hit-test in the transcript, and tool-call results are hard-truncated (`… +N more`) with no way to expand.
5. **"Turn N" leaks in two residual paths** the strip doesn't cover: the streaming **volatile tail** (rendered as raw dim text) and `turn_title`'s `format!("Turn {number} · {name}")` / `format!("Turn {number}")` fallbacks.

---

## Findings (file:line bugs, root cause not symptom)

### F1 — Inline math is a single Span; `wrap_styled_line` splits it mid-formula (Q3, the #1 latex bug)
`markdown/render.rs:208-214` pushes inline math as **one span**:
```rust
Seg::Math(latex) => {
    let rendered = math::render_math(&latex, Display::Inline);   // a String, may be many cells
    self.cur.push(Span::styled(rendered, self.col(Token::Claude)));
}
```
That span then flows through `markdown/mod.rs:460 wrap_styled_line` → `wrap_styled_hard_line` (mod.rs:512), which flattens **every span to graphemes** (`span.content.graphemes(true)`) and re-wraps by display cell via `wrap_line_segments`. So `∑ᵢ₌₀ⁿ` or a `\frac` rendered inline `(a+b)/c` is treated as ordinary breakable text: at a narrow width the wrapper will break **inside** the rendered formula (e.g. `∑ᵢ₌` / `₀ⁿ`), and a word-wrap boundary can even drop a space that was part of the math. There is no "atomic span" concept. This is *the* reason inline latex "still renders wrong": it renders correctly in `math.rs`, then gets shredded by the generic wrapper. `math.rs` is solid (1228 lines, `catch_unwind`, full symbol map, `unicode-width` throughout); the bug is one layer up.

Severity: this is also a **silent P1-parity risk** — the plain projection (`lines_to_plain`) sees the same many-cell string and wraps it identically, so the row count still matches; the corruption is purely visual (broken glyph runs), which is why no test caught it (`md_inline_math_is_rendered` only asserts `α` present + no `$`).

### F2 — Block math (`$$…$$`) is recognized ONLY as a whole isolated source; never mid-message, and it short-circuits the cockpit (Q3)
`markdown/mod.rs:79-81` and `mod.rs:39-41`: `render_assistant_cockpit_streaming` and `render_assistant` both do `if let Some(latex) = render::extract_block_math(source) { return render::render_block_math(...) }`. `extract_block_math` (render.rs:818) requires the **entire trimmed source** to be `$$…$$`. Consequences:
- A normal assistant turn that contains a display equation *among prose* (`text\n\n$$\\int...$$\n\nmore`) never hits block math — it falls into the markdown walker, where `split_inline_math` collapses `$$…$$` to **inline** (render.rs:739 `double` path returns a single inline `Seg::Math`), losing the multi-line stacked layout the whole `Display::Block` machinery exists for.
- When the source *is* pure `$$…$$`, the early return **bypasses the entire fold/chip cockpit** and (worse) is reachable even while `streaming==true`, so a half-typed `$$\frac{a}{` is fed to `extract_block_math`→ either no match (raw `$$` shows) or a transient wrong stack.

### F3 — `safe_commit_pos` has no math/code/table holdback; commits structurally-unstable blocks (Q3 "edge-to-edge but stable")
`render/fold.rs:227-284`. The "unsafe regions" it knows are: an in-flight `🛠️ ` header line, unclosed `<summary>`/`<thinking>`, an odd `**`. It then commits up to the **last `\n\n`** before that. It does **not** treat as volatile:
- an unclosed inline `$…$` or block `$$…$$` (a half-typed formula commits and renders as literal `$\frac{a}{` then reshapes — exactly Q3's "renders only after the whole thing is done / flashes wrong");
- an unclosed fenced code block (```` ``` ```` with no closer);
- a **GFM table being streamed** — Codex's central insight (`streaming/table_holdback.rs`): a table is non-incremental, every new row reflows all column widths, so committing row-by-row at `\n\n` paints stale-width rows into scrollback. v4 will do exactly that.

Codex peer: `streaming/controller.rs` keeps a `TableHoldbackScanner`; once a header+delimiter pair is seen, **everything from the table start stays mutable tail** until finalize. v4 has no analogue.

### F4 — Streaming re-render is O(whole source) every delta, uncached (Q3 perf / "real-time")
`components/mod.rs:291-297` builds `md_cache` *per frame* (a fresh `HashMap` each call to `render_transcript`), and `app/mod.rs:242-256` `to_render_block` re-runs `render_assistant_cockpit_plain/_streaming` for the streaming block on every `sync_transcript`. Each call re-runs `fold_turns` (scans all markers), `pulldown_cmark::Parser` over the full source, the math splitter, and `safe_commit_pos`, then **re-wraps every logical line**. There is no content-keyed memo. CC explicitly solved this (`utils/markdown.ts` via `Markdown.tsx cachedLexer`: `marked.lexer` memoized by `hashContent`, LRU 500, plain-text fast path skipping the ~3ms parse). Codex caches the rendered stable prefix (`StreamCore.stable_prefix_len_cache`) and only re-renders the tail. v4's per-block `rev` is the right key but is **only** used by the wrap cache (`measure.rs:317`), never by the markdown/fold layer.

### F5 — Q8 "expand any node by clicking ▸" is unimplemented; fold is one global bool
- `app/mod.rs:377 pub fold_all: bool` is the **only** fold state. `main.rs:610-612` Ctrl+O flips it. So every completed turn is folded or unfolded **together** — there is no per-turn, per-node toggle. Q8 requires "点击 summary 前的小箭头 ▸ 可展开任意节点".
- The fold header is drawn at `markdown/mod.rs:107-112` as a plain dim `▸ <title>` span. The `▸` is **decorative** — no row carries back a `(block_id, turn)` identity, and the transcript click handler (`main.rs:475-483`) only reacts to `me.row <= 1` (opens dashboard); a click anywhere in the transcript body is dropped. There is no `click → fold node` path.
- Tool-call **results** can't be expanded either: `render/chip.rs:264-299 render_chip_bullet` hard-truncates to `max_preview` rows + `… +N more` (mod.rs:259 passes `4`). The "+N more" is dead text, not a toggle.

CC peer: `CtrlOToExpand.tsx` (the `(ctrl+o to expand)` hint) + `CollapsedReadSearchContent.tsx` (collapsed group with a `⎿` detail gutter, `isActiveGroup` live hint via `useMinDisplayTime` anti-flicker). Codex peer: every cell is a `HistoryCell` with `display_lines(width)` and a `HistoryRenderMode` it can re-emit expanded.

### F6 — Residual "Turn N" leaks the strip misses (Q8 "绝不能再出现 Turn 1 …")
The finalized strip is good: `render_turn_body` (mod.rs:185-248) drops the leading marker via `strip_leading_turn_line` (mod.rs:356), and `fold_turns` (fold.rs:55) replaces folded turns with `▸ <summary>`. But:
- **Volatile streaming tail**: `markdown/mod.rs:119-127` renders the held-back tail as raw `Span::styled(raw, Dim)` per line — **no turn-line stripping**. If a new `Turn N ...` marker arrives in the live tail before `safe_commit_pos` advances past it, the literal `Turn 12 ...` shows dim in the active region. (The tests only exercise finalized input.)
- **Title fallback**: `fold.rs:174 format!("Turn {number} · {name}")` and `fold.rs:187 format!("Turn {number}")` re-introduce the exact "Turn N" text into a fold header when a turn has no `<summary>` and no tool. So a folded no-summary turn renders `▸ Turn 4 · web_search` or `▸ Turn 4` — the ugly artifact, reborn as a title. (`fold.rs:347` test asserts this on purpose.)

### F7 — `render_math` return-type smell + dead block-math API (review principle, minor)
`math.rs:109 render_math` returns a `\n`-joined `String`; for `Display::Block` that embeds `\n` into whatever the caller pushes. Only `render.rs:209` uses it, always `Inline`, so it's latent — but it's a footgun: any future block caller of `render_math` would push a `\n`-bearing span (F1's exact class of bug). `render_block_math` (render.rs:832) + `extract_block_math` are `#[allow(dead_code)]`-adjacent (only used by `render_assistant*`), and the routing (F2) means block math is never reached from a real multi-part message.

---

## Competitor patterns (CC / Codex / v2 / v3, with file cites)

**Codex `markdown_stream.rs` — newline-gated source commit (the holdback primitive).** `MarkdownStreamCollector::commit_complete_source()`: `let commit_end = self.buffer.rfind('\n').map(|i| i+1)?; if commit_end <= self.committed_source_len { return None } …` — commit only completed *lines*, never a partial line "that may change meaning when the rest arrives". `finalize_and_drain_source()` flushes the remainder (newline-terminating it). This is exactly v4's `safe_commit_pos` but finer-grained and **the right shape to bolt holdback onto**.

**Codex `streaming/controller.rs` — two-region (stable/tail) model.** `StreamCore { raw_source (append-only), rendered_lines (full re-render at width), enqueued_stable_len, emitted_stable_len, holdback_scanner }`. Invariants: `emitted_stable_len <= enqueued_stable_len <= rendered_lines.len()`; "tail starts exactly at `enqueued_stable_len`"; on resize `set_width` "re-renders at the new width and rebuilds the queued stable region". This is v4's `(head, tail)` split (mod.rs:84-89) but with the tail rendered through the **same** markdown path (not dumped as raw dim text) and with a holdback boundary.

**Codex `streaming/table_holdback.rs` — `TableHoldbackScanner`.** States `None | PendingHeader{header_start} | Confirmed{table_start}`; one-line lookbehind; once a header line is followed by a delimiter line it's `Confirmed` and "content from the table header onward stays mutable" until finalize. Fence-aware (`FenceTracker`). Direct model for the v4 fix in F3.

**Codex `markdown.rs` — `append_markdown(source, width, cwd, &mut lines)` is pure `source → Vec<Line>`**; `unwrap_markdown_fences` zero-copy fast path (`if !source.contains("```") { Cow::Borrowed }`). Mirrors v4 `render_markdown` exactly — confirms the architecture.

**Codex `live_wrap.rs` — `RowBuilder` proves incremental wrap with fragmentation-invariance.** `push_fragment` accumulates; `set_width` rewraps all; `take_prefix_by_width` is CJK/emoji-correct (`UnicodeWidthChar`); test `fragmentation_invariance_long_token` asserts chunked pushes == one push. v4's `wrap_line_segments` (measure.rs:102) already has this property; the gap is that **math should be wrapped as an atomic unit**, which `RowBuilder` would also need a "no-break span" for. The lesson: width is the only wrap input, and the same source re-wraps deterministically — so the fix for F1 is to keep math out of the grapheme stream, not to change the wrapper.

**Codex `history_cell/exec.rs` — the tool/exec cell layout v4's chip should match.** Header `vec!["↳ ".dim(), "Interacted…".bold(), " · ".dim(), command.dim()]`; body via `adaptive_wrap_lines(..., RtOptions::new(w).initial_indent("  └ ".dim()).subsequent_indent("    ".dim()))`. Note: **proper hanging indent** (`initial_indent`/`subsequent_indent`) so wrapped result lines align under `└`. v4's `push_tool_bullet` (mod.rs:252) emits a flat 2-space indent with no continuation indent — wrapped result rows lose alignment.

**CC `AssistantToolUseMessage.tsx` — the bullet.** `BLACK_CIRCLE` (`⏺`) when queued, `ToolUseLoader` (animated) while unresolved, dim/error color by `erroredToolUseIDs`; name `bold` `wrap="truncate-end"`; args in `(…)`. v4's `chip.rs` `ToolStatus::bullet()` (`⏺`/`○`) + `token()` already match this 1:1 — good.

**CC `CollapsedReadSearchContent.tsx` — collapsed→expand.** Non-verbose: a single line "Read 3 files, searched 2 patterns … `<CtrlOToExpand/>`" with a `⎿` gutter for the live hint; verbose: each tool with its 1-line result. `useMinDisplayTime(hint, 700ms)` holds a hint ≥700ms so fast tools don't flicker. **`CtrlOToExpand.tsx`**: `chalk.dim(`(${shortcut} to expand)`)`. This is the affordance Q8 wants — but CC's is also global Ctrl+O (transcript mode), so the *click a specific ▸* requirement is a v4 super-set CC doesn't have; Codex's per-cell `HistoryRenderMode` is the closer model.

**CC `Markdown.tsx` / `utils/markdown.ts` — pure walker + memo.** `applyMarkdown = marked.lexer(stripPromptXMLTags(content)).map(formatToken).join('').trim()`. `cachedLexer`: content-hash key, LRU 500, `hasMarkdownSyntax` fast path skips lexer for plain text. `formatToken` is a recursive token→ANSI switch (`blockquote`→`│` bar per line, `codespan`→permission color, `strong`→`chalk.bold`, …). v4 `Walker` (render.rs:61) is the same idea; the missing piece is the **memo keyed on (rev,width,fold)**.

**tui_v3 (`frontends/tui_v3.py`) — the direct ancestor.** `_fold_turns` / `_safe_pos` / `_tool_status` are explicitly ported (fold.rs cites "tui_v3 `_fold_turns` port", chip.rs cites "tui_v3 `_tool_status`"). v3 also lacks per-node fold (same global limitation), so the Q8 click-to-expand is genuinely new work, not a regression.

---

## Fix design (Rust sketches: the actual changed lines / new fn signatures)

### Fix A (F1) — make rendered math an **atomic, non-wrappable** span
Span has no "atomic" flag in ratatui. Add a side-channel the wrapper honors. Minimal, local:

1. `markdown/render.rs` — tag math spans with a sentinel style modifier (free bit) so the wrapper can recognize them without a parallel structure. Reuse a rarely-used `Modifier`:
```rust
// render.rs, near top
/// Marks a Span whose content must never be split by soft-wrap (rendered math).
/// We piggyback on RAPID_BLINK (never used elsewhere in the theme) as an out-of-band tag.
pub(crate) const ATOMIC: Modifier = Modifier::RAPID_BLINK;

// in push_inline_run, Seg::Math arm:
Seg::Math(latex) => {
    let r = math::render_math(&latex, Display::Inline);      // single line (Inline guarantees it)
    self.cur.push(Span::styled(r, self.col(Token::Claude).add_modifier(ATOMIC)));
}
```
2. `markdown/mod.rs wrap_styled_hard_line` (mod.rs:512) — when consuming graphemes, if the current grapheme's style carries `ATOMIC`, **consume the whole atomic run as one unit** and, if it doesn't fit the remaining cells, push it to the next row whole (never split). Sketch inside the segment loop:
```rust
// before the per-grapheme run loop, peek for an atomic run:
if gph[pos].1.add_modifier.contains(ATOMIC) {
    let (atom, w) = take_atomic_run(gph, pos);          // collects until style loses ATOMIC
    if cur_w + w > width && cur_w > 0 { flush_row(...); }// move whole atom to next row
    push_span(atom, gph[pos].1);
    pos += atom_len; continue;
}
```
Because the **plain projection** must match, `lines_to_plain` already emits the same string; but the wrap-cache plain path (`measure.rs reflow_block`) would still break the formula by width. So the cache must agree: simplest correct option is to **clamp narrow widths from breaking math by widening the logical line in the plain projection too** — i.e. teach `wrap_line_segments` the same atomic rule via a wrapper that the cache uses. Cleaner: introduce a private `wrap_line_segments_atomic(line, width, atomic_ranges)` and have BOTH `wrap_styled_hard_line` and the cache's `reflow_block` call it with the math byte-ranges. Strip `ATOMIC` from the final emitted spans (it's not a real visual style) in `flush_line`.

(If touching the cache is too invasive for this slice: the pragmatic 80%-fix is to **render inline math so it never exceeds a sane cell budget** and accept that an ultra-narrow terminal wraps it — but the atomic-range approach is the correct one and is what Codex's wrapper would need too.)

### Fix B (F2/F7) — route block math per-paragraph, not per-whole-source; drop the inline-collapse of `$$`
1. `markdown/render.rs split_inline_math` (render.rs:721) — when `double` (a `$$…$$`) is found **and** the body spans the whole text run (i.e. the run trimmed is exactly the `$$…$$`), emit a **block** marker the walker turns into stacked lines, instead of collapsing to inline. Better: handle block math at the **paragraph** boundary inside the `Walker` (on `TagEnd::Paragraph`, if the paragraph's accumulated text is a lone `$$…$$`, replace its single line with `render_block_math` lines). This makes display math work *inside* a normal multi-part assistant message (F2).
2. `markdown/mod.rs:79-81` + `:39-41` — keep the whole-source `extract_block_math` short-circuit **only for finalized, non-streaming** input; never run it when `streaming==true` (a partial `$$` must stay in the volatile tail, see Fix C). Guard: `if !streaming { if let Some(latex)=extract_block_math(head) {...} }`.
3. `math.rs render_math` — change the block caller contract: callers that need block math call `latex_to_unicode(.., Block).lines` directly (already what `render_block_math` does at render.rs:833); mark `render_math` `#[inline]` and document "Inline only; Block callers must use `latex_to_unicode`".

### Fix C (F3) — add math/code/table holdback to `safe_commit_pos`
Extend `render/fold.rs safe_commit_pos` (fold.rs:227). New unsafe-region scans (each `note(pos, &mut unsafe_from)`):
```rust
// unclosed inline/block math: odd count of `$` (ignoring `\$`) after the last commit boundary
if unmatched_dollar(stream) { note(last_dollar_pos(stream), &mut unsafe_from); }
// unclosed fenced code: odd count of ``` (or ~~~) line-starts
if open_fence_pos(stream).is_some() { note(open_fence_pos(stream).unwrap(), &mut unsafe_from); }
// streaming table: header+delimiter seen, no blank line terminating it yet
if let Some(tbl_start) = streaming_table_start(stream) { note(tbl_start, &mut unsafe_from); }
```
`streaming_table_start` is a ~30-line port of Codex `TableHoldbackScanner::push_line`: one-line lookbehind, `is_table_header_line(prev) && is_table_delimiter_line(cur)` ⇒ `Confirmed{prev_start}`; hold from there until a blank line (table end) appears in committed source. Keep the existing `\n\n` fallback for the committed cut. New helpers are pure + unit-testable (add `safe_commit_pos_holds_streaming_table` / `_holds_unclosed_math` / `_holds_open_fence`). This makes the **head** never contain a half-built block; the **tail** (Fix D) renders it live through the real markdown path so it still streams edge-to-edge.

### Fix D (F4 + F3-tail) — render the volatile tail through markdown + memoize per (rev,width,fold)
1. `markdown/mod.rs:119-127` — replace the raw-dim tail loop with the **same** `render_turn_body` path (so the tail streams *rendered*, just from a not-yet-stable region), and run `strip_leading_turn_line` on it (fixes F6 tail leak):
```rust
if !tail.is_empty() {
    let tail_body = strip_leading_turn_line(tail);
    logical.extend(render_turn_body(tail_body, theme, width));   // not raw spans
}
```
2. Add a content-addressed memo so streaming isn't O(whole source) per frame. Key on the block's `rev` + `width` + `fold_all` (all already tracked). In `app/mod.rs` give `Block` a small cache field:
```rust
// app::Block
cockpit_cache: Option<(u64 /*rev*/, u16 /*width*/, bool /*fold_all*/, Vec<Line<'static>>, String /*plain*/)>,
```
`to_render_block` and `render_transcript` consult it; on miss, render once and store both the styled lines and the plain projection (they must be produced together to stay 1:1). This also lets `components/mod.rs:281 md_cache` (per-frame HashMap) be dropped — the block owns its cache. Mirrors Codex `stable_prefix_len_cache` + CC `cachedLexer`.

### Fix E (F5) — per-node fold + clickable ▸ + expandable tool result (the Q8 deliverable)
This is the largest piece; it is **additive** (the global Ctrl+O can stay as "fold/unfold all").

1. **Per-node fold state** on `AppState`:
```rust
// app/mod.rs
/// Explicit per-turn fold overrides, keyed (block_id, turn). Absent ⇒ default
/// (completed turns folded, last turn open). Ctrl+O still flips `fold_all`.
folds: HashMap<(u64, u32), bool>,
```
`fold_turns` (fold.rs:55) gains a `is_folded: &dyn Fn(u32)->bool` param so the renderer decides per turn instead of "all but last". Default closure = current behavior; an override map entry wins.

2. **Provenance: map a visual row back to a fold node.** Today a `VisualLine` only knows `(block_id, intra)`. Add to the fold-header logical line a tag so the click handler can resolve it. Cheapest: when the cockpit emits a fold header (mod.rs:107) or a tool bullet (mod.rs:273), record an interval `[first_intra, last_intra] -> NodeId::{Turn(block,turn)|Tool(block,call_id)}` in a per-block side-table the transcript exposes (parallel to the wrap cache). `AppState` keeps `node_hit: Vec<(BlockId, RangeInclusive<usize>, NodeId)>` rebuilt in `sync_transcript`.

3. **Mouse hit-test** — extend `main.rs:475` cockpit `Down(Left)` (currently only `me.row<=1`):
```rust
(View::Cockpit, MouseEventKind::Down(MouseButton::Left)) => {
    if me.row <= 1 { app.open_dashboard(); }
    else if let Some(node) = app.transcript_node_at(me.column, me.row) { // col≈arrow cell
        app.toggle_fold(node);   // flips folds[(block,turn)] or expands a tool result
    }
}
```
`transcript_node_at` mirrors `dashboard::click_to_row_index` (main.rs:501): translate `me.row` − transcript-region-top → visual index via the viewport top, then look up `node_hit`. Restrict the clickable cell to the `▸`/`⏺` column (first 2 cells) so body clicks still pass through to selection.

4. **Expandable tool result** — `render/chip.rs render_chip_bullet` (chip.rs:264) takes the node's expanded flag; when expanded, skip the `max_preview` truncation (emit all `result.lines()` with the `  ` indent) and render `▾` instead of the `… +N more`. Add `▸`/`▾` glyph in front of the bullet for any chip whose result exceeds `max_preview`, so the affordance is visible (CC's `(ctrl+o to expand)` analogue, but per-node).

5. **Keyboard parity** — keep Ctrl+O = fold/unfold all (clears `folds`, flips `fold_all`); optionally add Enter/Space on a "focused" node later. The deliverable is the click; keyboard-all already works.

### Fix F (F6) — kill the "Turn N" title fallback
`render/fold.rs turn_title` (fold.rs:167): never bake the literal "Turn N". When no `<summary>` and no tool name, fall to the first prose line (already the `fold.rs:180-186` branch) or a **neutral** glyph-only title (e.g. the first ~6 words of the body), and replace the final `format!("Turn {number}")` (fold.rs:187) and `format!("Turn {number} · {name}")` (fold.rs:174) with just `name` / `"…"`:
```rust
if let Some(name) = first_tool_name(body) { return name; }      // was "Turn N · {name}"
// …prose-line branch unchanged…
String::from("…")                                                // was "Turn {number}"
```
Update the `turn_title_falls_back_to_tool_then_generic` test (fold.rs:344) accordingly. Combined with Fix D's tail strip, "Turn N" can appear **nowhere**.

---

## Review-principle violations (cite principle # + file:line)

- **#2 local reasoning / #6 constraints-in-types** — `markdown/render.rs:209` math span is visually atomic but typed as an ordinary `Span`; nothing in the type stops the wrapper from splitting it (F1). The invariant "math is unbreakable" lives only in a human's head. Fix A puts it in the (ab)used `Modifier`/range channel.
- **#4 variation radius / #1 module boundary** — fold state is split across `app.fold_all` (app/mod.rs:377), `last_fold_all` (mod.rs:476), and the `fold_turns(text, fold_all)` signature (fold.rs:55). Adding per-node fold (Q8) currently means threading a 4th place; the `is_folded` closure (Fix E) collapses the variation point to one.
- **#9 self-documenting / #15 length follows function** — `components/mod.rs render_transcript` (mod.rs:266-350) mixes wrap-cache sync, per-frame md memo, user-band, gutter, and "more below" hint in one 84-line fn; the streaming-vs-finalized `md_cache` decision (mod.rs:291-297) is duplicated against `app/mod.rs to_render_block` (mod.rs:232-260) — two copies of the same fold+chip+stream policy that MUST stay 1:1 (a classic #4 trap; the file even comments "mirror it 1:1"). Fix D's block-owned cache removes the duplication.
- **#14 let-it-crash by blast radius** — `math.rs:96 catch_unwind` around a **pure, total, valid-UTF-8** transform is belt-and-braces over zero blast radius; the doc admits "the transforms themselves never panic". Acceptable as a stream-protection backstop, but it's exactly the "try-catch everything" the principle warns against; the real corruption risk (F1) is *outside* this guard, in the wrapper.
- **#12 more features fewer lines** — `chip.rs render_chip_bullet` hard-truncates and emits dead `… +N more` text (chip.rs:288-291); turning that same code path into the expand affordance (Fix E.4) adds the Q8 feature with ~5 lines, not a new component.
- **#10 visual uniformity** — `mod.rs:119-127` renders the volatile tail as flat dim spans while the committed head gets full markdown/chip treatment; the seam is visible mid-stream (a line "downgrades" from styled to dim as it crosses the commit boundary, then "upgrades" back). Fix D unifies the render path.

---

## Open questions / risks

1. **Atomic-span wrap parity (Fix A) is the riskiest change** — it must touch BOTH the styled wrap (`wrap_styled_hard_line`) and the plain wrap-cache (`reflow_block`) with the *same* math byte-ranges, or the load-bearing `styled_wrap_rowcount_matches_wrap_cache` invariant (mod.rs:631) breaks and scroll drifts (P1). Recommend implementing the shared `wrap_line_segments_atomic` and running that test across the math fixtures before anything else. If schedule-constrained, ship F2/F3/F4/F6 first (no parity risk) and gate F1 behind the shared-wrapper work.
2. **Per-node fold (Fix E) interacts with the wrap cache + scroll anchor.** Toggling a fold changes a block's row count; the `Anchored{block_id,intra}` anchor (viewport.rs:33) must be re-derived after a toggle (call the same path as a resize). A click that expands a node *above* the viewport top must not jump the view — anchor on the clicked node, not Bottom.
3. **Holdback vs. follow-mode latency (Fix C/D)** — holding a streaming table/code/math block as mutable tail means it isn't in scrollback until finalize; that's correct (Codex does it) but the tail region must be tall enough or the user sees only the table's last rows while it builds. Codex's `commit_tick`/`AdaptiveChunkingPolicy` (streaming/commit_tick.rs) exists precisely to pace this; v4 has no commit-pacing — a very long held table could grow the tail unboundedly. Cap the tail height and scroll within it (Codex `live_wrap.rs drain_commit_ready(max_keep)`).
4. **`Modifier::RAPID_BLINK` as the ATOMIC sentinel** assumes the theme/terminal never emits real rapid-blink. Safe today (grep: no `RAPID_BLINK` use), but document it; a cleaner long-term option is a parallel `Vec<Range<usize>>` of no-break byte ranges carried alongside the `Vec<Line>`.
5. **`$$…$$` paragraph detection (Fix B)** must not misfire on a paragraph that merely *contains* `$$` inside prose; gate on "trimmed paragraph text starts and ends with `$$` and the interior is non-empty", reusing the `extract_block_math` predicate at the paragraph level.
6. **CJK + math interaction** — inline math glyphs include combining marks (accents) and full-width fallbacks; the atomic-run width must use the same `unicode-width` path (`math.rs` already does). Verify `\bar{你}` etc. measure correctly in the wrapper's atomic branch.
