# R2 — Markdown + LaTeX Rendering Coverage Spec

## (1) REPRODUCED?

**PARTIAL — confirmed by live/styled TestBackend run.**

The recon test (`cargo test rich_fixture_recon_all_elements -- --nocapture`) fed a rich fixture through `render_markdown` (the exact function `render_assistant` calls) to a real styled render and captured all 50 output rows. Output:

```
[000] "H1 Heading"       [BOLD]
[001] ""
[002] "H2 Heading"       [BOLD]
...
[010] "H6 Heading"       [BOLD]
[011] ""
[012] "│ Left      │ Center │ Right │  │"  [BOLD,BOLD,BOLD,BOLD]  ← phantom 4th col
[013] "├───────────┼────────┼───────┼──┤"
[014] "│ a         │   b    │     c │  │"
[015] "│ long cell │   x    │    99 │  │"
[016] ""
[017] "bold text and italic text and inline code and strikethrough text"  [BOLD,ITAL,STRIKE]
...
[019] "1. First item"
[020] "2. Second item"
[021] "  1. Nested ordered"
[022] ""                           ← spurious blank between nested list and item 3
[023] "3. Third"
...
[025] "• Bullet one"
[026] "• Bullet two"
[027] "  ◦ Nested bullet"
...
[032] "│ This is a blockquote with two lines"  ← two >-lines collapsed to one
...
[041] "────────"                   ← HR is only 8 dashes, not full width
...
[043] "Inline math: E=mc² and ∑ᵢ₌₀ⁿ xᵢ"   ← CORRECT
...
[047] "1     "
[048] "∫ x,dx"                    ← BUG: comma instead of space (pulldown eats \,)
[049] "0     "
```

## (2) ROOT CAUSE (per element)

### GAP A — Phantom 4th column in tables (RENDERS-WRONG)

**File:** `src/markdown/render.rs` lines 486–509.

Event sequence from pulldown-cmark:
```
Start(TableHead)
  Start(TableCell) Text("A") End(TableCell)   ← TableCell flush pushes "A"
  Start(TableCell) Text("B") End(TableCell)   ← TableCell flush pushes "B"
  Start(TableCell) Text("C") End(TableCell)   ← TableCell flush pushes "C"
End(TableHead)                                ← cur_cell is now ""
                                               TagEnd::TableHead also pushes cur_cell → empty phantom column
```

`TagEnd::TableCell` at line 504–508 pushes `cur_cell` for EVERY cell. Then `TagEnd::TableHead` at line 488–490 also does `take cur_cell; cur_row.push(cell)` — at that point `cur_cell` is already empty (cleared by the final `TagEnd::TableCell`), so an empty string is pushed as a 4th column. Same double-push in `TagEnd::TableRow` (lines 494–502).

Confirmed: with a 3-column table, every row shows 5 `│` glyphs (= 4 column separators = phantom 4th column).

### GAP B — Headings: no level differentiation beyond color (RENDERS-WRONG)

**File:** `src/markdown/render.rs` `start()` `Tag::Heading` handler (line 311–321) and `src/markdown/code.rs` `heading_style()` (line 54–63).

All H1–H6 receive only `BOLD` modifier. No `ITALIC`, no `UNDERLINED`, no `#`-prefix glyph. The `heading_style()` function assigns different color tokens (H1=Claude, H2=Suggestion, H3=Success, H4=Warning, H5=PlanMode, H6=Dim) but all share the same `Modifier::BOLD` only. A dark theme with close colors makes H2–H5 visually indistinguishable. No `#` prefix glyph, no underline on H1/H2.

### GAP C — Nested list produces spurious blank line (RENDERS-WRONG)

**File:** `src/markdown/render.rs` `end()` `TagEnd::List` (line 458–460).

When a nested list ends inside a parent item, `TagEnd::List` calls `self.lists.pop()` and sets `self.started = true`. The NEXT event is `TagEnd::Item` for the parent item which calls `flush_line()`. But before that, the next item in the parent list fires `Tag::Item` → `flush_pending_line_if_content()` and then `block_gap()` gets called somewhere upstream. Specifically, `Tag::List` at line 339–344 calls `block_gap()` for the FIRST nested list open (not inner), but on `TagEnd::List` for the nested list, `self.started = true` means the NEXT `block_gap()` call will insert a blank row. The blank at row [022] comes from the outer list trying to insert spacing between items.

### GAP D — Blockquote soft-break: multi-line `>` collapses to one row (RENDERS-WRONG)

**File:** `src/markdown/render.rs` `event()` `Event::SoftBreak` (line 259–266).

```rust
Event::SoftBreak => {
    if self.code.is_none() && self.table.is_none() {
        self.cur.push(Span::raw(" "));  // ← space, not new line
    }
}
```

`> line1\n> line2` fires: Text("line1"), SoftBreak, Text("line2"). The SoftBreak becomes a single space span, so both lines merge into `│ This is a blockquote with two lines`. The user sees a single line where there should be two visually distinct rows.

### GAP E — HR is fixed 8 dashes, not terminal width (RENDERS-WRONG)

**File:** `src/markdown/render.rs` `event()` `Event::Rule` (line 274–279):
```rust
Event::Rule => {
    self.out.push(Line::from(Span::styled(
        "────────".to_string(),   // ← hardcoded 8 dashes
        ...
    )));
}
```
The Walker struct has no width field, so it cannot render a full-width rule. The rule is 8 dashes regardless of terminal width.

### GAP F — LaTeX `\,` (thin space) corrupted by pulldown-cmark (RENDERS-WRONG)

**File:** `src/markdown/render.rs` / pulldown-cmark interaction.

pulldown-cmark treats `\,` as a markdown escape sequence (backslash before a non-alphanumeric char) and fires TWO Text events: `"$$\\int_0^1 x"` + `",dx$$"`. The backslash is consumed and the `,` becomes literal text. The paragraph buffer `para` becomes `"$$\\int_0^1 x,dx$$"` so `extract_block_math` parses latex body `"\\int_0^1 x,dx"` (with a literal comma instead of thin space). The `math.rs` tokenizer then sees `,` as `Tok::Char(',')` → rendered as literal comma instead of space.

Affected LaTeX sequences: ALL `\<non-alpha>` in math bodies. Examples: `\,` (thin space → `,`), `\!` (neg space → `!`), `\;` (med space → `;`), `\:` (colon → `:`), `\.` → `.`.

The `math.rs` `symbol()` function correctly maps `","` to `" "` — but the body reaching it already has the literal `,` at byte level, so the tokenizer sees `Tok::Char(',')` not `Tok::Cmd(",")`.

### NOT A REAL GAP — Headings H1–H6 all present (RENDERS-OK)

All six heading levels render with their text content — confirmed at rows [000]–[010].

### NOT A REAL GAP — Table renders (RENDERS-OK except phantom column)

GFM tables do render with `│` borders, alignment, and `┼` separator. The only bug is the phantom extra column (Gap A).

### NOT A REAL GAP — Bold/italic/inline code/strikethrough (RENDERS-OK)

`**bold**`, `*italic*`, `` `code` ``, `~~strike~~` all render with correct modifiers.

### NOT A REAL GAP — Inline math (RENDERS-OK)

`$E=mc^2$` → `E=mc²`, `$\sum_{i=0}^{n} x_i$` → `∑ᵢ₌₀ⁿ xᵢ`. No raw `$` leaks. The `pending` coalescing (which reassembles inline math split across pulldown Text events at `_`/`*`) works correctly.

### NOT A REAL GAP — Block math `$$…$$` (RENDERS-OK structurally, WRONG for `\,`)

Block math routes through the stacked 3-line layout correctly. The integral produces `1  ` / `∫ x dx` / `0  `. The only corruption is `\,` becoming `,` (Gap F above).

### NOT A REAL GAP — pulldown Options flags (tables + strikethrough enabled)

`render.rs` line 39–43: `Options::ENABLE_TABLES` and `Options::ENABLE_STRIKETHROUGH` and `Options::ENABLE_TASKLISTS` are all set. No missing option flags.

## (3) REFERENCE PATTERN

### codex `tui/src/markdown_render.rs`

**Headings**: emits a `"#".repeat(level) + " "` prefix span in the heading style (line 348), THEN pushes the heading inline style for the text. H1=bold+underline, H2=bold, H3=bold+italic, H4–H6=italic. This gives immediate visual level cues.

```rust
// codex markdown_render.rs lines 340–352
let content = format!("{} ", "#".repeat(level as usize));
self.push_line(Line::from(vec![Span::styled(content, heading_style)]));
self.push_inline_style(heading_style);
```

**Tables**: NOT implemented in codex. `Tag::Table / TableHead / TableRow / TableCell` are all no-ops (lines 281–287). tui_v4 is already ahead of codex here.

**Soft-break**: codex treats `SoftBreak` as a hard line break — `push_line(Line::default())` at line 480. This preserves `>` blockquote multi-line structure.

**HR**: codex uses `"———"` (three em-dashes, fixed) at line 231. No full-width HR either.

**LaTeX / math**: codex has NO math rendering. tui_v4 surpasses codex on all math.

### tui_v3.py / tuiapp_v2.py

Both use Rich's `Markdown` class (via `HardBreakMarkdown` wrapper) which handles all CommonMark + GFM natively at the Python level, including:
- Tables via Rich `Table` with column alignment and border styles
- Headings with level-specific styles (`markdown.h1` = bold+underline, `markdown.h3` = bold, etc.)
- LaTeX: **NOT rendered** — Rich has no math support; `$…$` passes through as literal text

The `_patch_markdown_table_overflow` in `tuiapp_v2.py` (lines 313–343) patches Rich's table element to use `overflow="fold"` instead of `overflow="ellipsis"` so long cells wrap rather than truncate. This is the key table-rendering sophistication in the Python references.

**tui_v3 heading styles** (lines 1480–1483):
```python
'markdown.h1': 'bold underline', 'markdown.h2': 'bold underline',
'markdown.h3': 'bold', 'markdown.h4': 'bold',
'markdown.h5': 'bold', 'markdown.h6': 'bold',
```

**Table border styles** use Rich's SIMPLE box with `show_edge=True`, `pad_edge=False`, `collapse_padding=True` (tuiapp_v2.py line 324–330).

## (4) FIX SPEC

### FIX-A: Table phantom-column (Priority: HIGH — explicit user demand)

**File:** `src/markdown/render.rs`

**Change 1:** Remove the redundant `cur_cell` push from `TagEnd::TableHead`:

```rust
// BEFORE (lines 486–492):
TagEnd::TableHead => {
    if let Some(tb) = self.table.as_mut() {
        let cell = std::mem::take(&mut tb.cur_cell);  // ← remove these two lines
        tb.cur_row.push(cell);                        // ← remove these two lines
        tb.header = std::mem::take(&mut tb.cur_row);
        tb.in_header = false;
    }
}

// AFTER:
TagEnd::TableHead => {
    if let Some(tb) = self.table.as_mut() {
        tb.header = std::mem::take(&mut tb.cur_row);
        tb.in_header = false;
    }
}
```

**Change 2:** Same fix for `TagEnd::TableRow` (lines 494–502) — remove the redundant push:

```rust
// BEFORE:
TagEnd::TableRow => {
    if let Some(tb) = self.table.as_mut() {
        if !tb.in_header {
            let cell = std::mem::take(&mut tb.cur_cell);  // ← remove these two lines
            tb.cur_row.push(cell);                        // ← remove these two lines
            let row = std::mem::take(&mut tb.cur_row);
            tb.rows.push(row);
        }
    }
}

// AFTER:
TagEnd::TableRow => {
    if let Some(tb) = self.table.as_mut() {
        if !tb.in_header {
            let row = std::mem::take(&mut tb.cur_row);
            tb.rows.push(row);
        }
    }
}
```

Rationale: `TagEnd::TableCell` already fires for EVERY cell including the last one before `TagEnd::TableHead`/`TagEnd::TableRow`, so `cur_cell` is always empty at those points. The double-push creates the phantom column.

### FIX-B: Heading level differentiation (Priority: HIGH — explicit user demand)

**File:** `src/markdown/code.rs` `heading_style()` function, and `src/markdown/render.rs` `start()` `Tag::Heading` handler.

**Step 1** — Add `#` prefix with level-appropriate modifier (mirroring codex):

In `render.rs` `start()`, before pushing the heading style to `style_stack`, emit a dim `#`-prefix span:

```rust
Tag::Heading { level, .. } => {
    self.block_gap();
    let tok = code::heading_style(level);
    let prefix_style = Style::default().fg(self.theme.color(Token::Dim));
    let hashes = "#".repeat(level as usize);
    self.cur.push(Span::styled(format!("{hashes} "), prefix_style));
    self.style_stack.push(
        Style::default()
            .fg(self.theme.color(tok))
            .add_modifier(Modifier::BOLD),
    );
}
```

**Step 2** — Add level-appropriate modifiers in `heading_style()` (code.rs lines 54–63). Currently all levels return only a color token and the walker adds only BOLD. Fix the walker to also apply level-specific modifiers:

Replace the single `Style::default().fg(...).add_modifier(Modifier::BOLD)` with a per-level style:

```rust
Tag::Heading { level, .. } => {
    self.block_gap();
    let tok = code::heading_style(level);
    // Level-specific modifier (codex reference: H1=bold+underline, H2=bold,
    // H3=bold+italic, H4-H6=italic)
    let base_style = Style::default().fg(self.theme.color(tok));
    let heading_style = match level {
        HeadingLevel::H1 => base_style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
        HeadingLevel::H2 => base_style.add_modifier(Modifier::BOLD),
        HeadingLevel::H3 => base_style.add_modifier(Modifier::BOLD | Modifier::ITALIC),
        HeadingLevel::H4 => base_style.add_modifier(Modifier::ITALIC),
        HeadingLevel::H5 => base_style.add_modifier(Modifier::ITALIC),
        HeadingLevel::H6 => base_style.add_modifier(Modifier::ITALIC),
        _ => base_style.add_modifier(Modifier::BOLD),
    };
    // Dim prefix glyph
    let prefix_style = Style::default().fg(self.theme.color(Token::Dim));
    let hashes = "#".repeat(level as usize);
    self.cur.push(Span::styled(format!("{hashes} "), prefix_style));
    self.style_stack.push(heading_style);
}
```

Note: `HeadingLevel` implements `as usize` giving 1–6. The `_ =>` arm is unreachable but required for exhaustiveness.

### FIX-C: SoftBreak → hard line (Priority: MEDIUM)

**File:** `src/markdown/render.rs` `event()` `Event::SoftBreak` handler (lines 259–265).

Current:
```rust
Event::SoftBreak => {
    if self.code.is_none() && self.table.is_none() {
        self.cur.push(Span::raw(" "));
        if let Some(p) = self.para.as_mut() { p.push(' '); }
    }
}
```

Change to flush and start a new line (matching codex line 480):
```rust
Event::SoftBreak => {
    if self.code.is_none() && self.table.is_none() {
        self.flush_pending();
        self.flush_line();
        // Still mirror the space into para so extract_block_math still works
        if let Some(p) = self.para.as_mut() { p.push(' '); }
    }
}
```

This makes `> line1\n> line2` render as two separate rows both prefixed with `│ `. It also means a wrapped prose paragraph renders one visual row per source line instead of the whole paragraph on one long logical row — this is closer to the Python reference (HardBreakMarkdown forces hardbreaks). The wrap cache already handles soft-wrapping per source line, so this is consistent with the row-count parity invariant.

**Warning:** This is a behaviour change for ordinary paragraphs too. A paragraph like `Hello\nworld` (soft break in the middle) currently renders as `Hello world` (one line). After the fix it renders as two lines. This matches the `HardBreakMarkdown` behaviour in tui_v3/tuiapp_v2. If it is deemed too aggressive, scope the change to ONLY the blockquote path (`if self.quote_depth > 0`).

### FIX-D: HR full-width (Priority: LOW)

**Files:** `src/markdown/render.rs` (Walker struct + `Event::Rule` handler).

The Walker needs a `width: u16` field (it currently has none). The render entry point `render_markdown` must accept a width. Since `render_markdown` is a public API called from `mod.rs render_assistant` which itself has no width (width is handled at the wrap layer), the cleanest fix is to accept a width parameter in `render_markdown` and thread it:

```rust
// render.rs: add width field to Walker
struct Walker<'t> {
    ...
    width: u16,
}

// Event::Rule:
Event::Rule => {
    self.block_gap();
    let w = (self.width as usize).saturating_sub(2).max(4);
    self.out.push(Line::from(Span::styled(
        "─".repeat(w),
        self.col(Token::Border),
    )));
    self.started = true;
}
```

Alternatively (simpler): pass a fixed-but-large value like `120u16` as default, or use `u16::MAX` and let the soft-wrap layer handle truncation. The safest minimal fix is just to increase the hardcoded count to match a typical terminal: `"─".repeat(80)`.

### FIX-E: LaTeX `\,` and other `\<non-alpha>` escapes (Priority: HIGH — explicit demand)

**Root cause:** pulldown-cmark consumes `\,` as a markdown escape before the text reaches the math pipeline.

**Fix:** Pre-process the source BEFORE passing it to the pulldown Parser. Specifically, inside `$$…$$` and `$…$` regions, protect `\<char>` sequences from pulldown's escape processing by replacing them with a placeholder or by using a custom pre-pass.

**Approach 1 (recommended):** Apply a pre-pass in `render_markdown` that temporarily replaces `\<non-alpha>` inside math delimiters with `\x{private-use}` placeholders that pulldown won't consume, then restore them after parsing. This is complex.

**Approach 2 (simpler and sufficient):** The real issue only affects `$$…$$` (block math) passed through the paragraph buffer. Inline `$…$` is already handled by `split_inline_math` which operates on the POST-pulldown text — but pulldown ALSO eats `\,` in inline math, so both paths are affected.

**Approach 3 (practical):** Add a special-case in `push_text` / `event()` that, when the accumulated `para` buffer shows we are in a `$$…$$` span, rebuilds the raw source by stitching back the Text events with `\` prefixes. This is fragile.

**Approach 4 (most practical):** Detect `$$…$$` paragraphs at the RAW SOURCE level BEFORE passing to pulldown, extract the LaTeX body directly from the raw source, and short-circuit the pulldown walk. This is already the structure of `render_markdown`:

```rust
pub fn render_markdown(source: &str, theme: &Theme) -> Vec<Line<'static>> {
    // FIX: detect whole-paragraph $$…$$ BEFORE pulldown can escape it
    // (split on double blank lines to find paragraphs, check each)
    // ... handle block math paragraphs directly ...
    let parser = Parser::new_ext(source, opts);
    ...
}
```

More precisely: split `source` on `\n\n` (paragraph boundaries), check each paragraph with `inline_math::extract_block_math`, and if it matches, emit it via `render_block_math` directly without running it through pulldown. Non-math paragraphs go to pulldown as before. This preserves inline `$…$` in prose (those are handled at the assembled-text level by `split_inline_math`, AFTER pulldown has already run — so `\,` inside inline math is still broken for the `$...\,...$` case).

For inline math with `\,`, the real fix is: in `split_inline_math`, after extracting the math body from the post-pulldown text, do NOT attempt to reverse the pulldown escape — instead document that `\,` in inline math renders as `,`. Alternatively, document that users should use `\ ` (backslash-space) instead of `\,` for thin-space in this renderer. The most commonly affected case is block `$$…$$` paragraphs (Approach 4 above).

**Concrete minimal fix for block math** (file: `src/markdown/render.rs`, `render_markdown` function):

```rust
pub fn render_markdown(source: &str, theme: &Theme) -> Vec<Line<'static>> {
    // Split into paragraphs and handle $$…$$ before pulldown eats backslash escapes.
    // A block-math paragraph is exactly "$$latex$$" (trimmed), possibly surrounded by
    // blank lines. We detect it by scanning raw source paragraphs FIRST.
    let mut out_lines: Vec<Line<'static>> = Vec::new();
    let mut remainder = source;
    // Simple paragraph splitter: double-newline boundaries.
    // We interleave pre-parsed block-math paragraphs with pulldown's output for the rest.
    // (Simplified: use the existing render_assistant → extract_block_math path which
    //  already handles the whole-source case; what's needed is per-paragraph detection.)
    // ...
}
```

The simplest concrete implementation: in `Walker::end()` `TagEnd::Paragraph`, instead of using `self.para` (which already has pulldown-escaped text), re-scan the `source` at the paragraph's byte position to extract the original raw bytes. This requires threading the source and paragraph offsets into Walker — significant refactor.

**Practical recommendation**: implement Approach 4 by pre-scanning `source` for `\n*$$...(no\n\n)...$$\n*` paragraphs, extract them directly, replace them with a placeholder line, let pulldown parse the rest, then splice the block-math lines back in.

### FIX-F: Nested-list spurious blank (Priority: MEDIUM)

**File:** `src/markdown/render.rs` `end()` `TagEnd::List` (lines 458–460).

The issue is that `block_gap()` being called after a nested list ends inserts a blank row before the next sibling item. The fix is to NOT call `block_gap()` when ending a nested list (depth > 1). Change:

```rust
TagEnd::List(_) => {
    self.lists.pop();
    self.started = true;  // ← this allows block_gap for the NEXT block
}
```

To not set `self.started = true` when inside a parent list:

```rust
TagEnd::List(_) => {
    self.lists.pop();
    // Only set started (enabling the next block_gap) for the TOP-LEVEL list end.
    if self.lists.is_empty() {
        self.started = true;
    }
    // For nested list end, don't advance started — the parent list's item
    // gap management handles spacing.
}
```

## (5) HONEST-CHECK — tests that exercise the LIVE path and FAIL today, PASS after each fix

### Test for FIX-A (table phantom column):
```rust
#[test]
fn table_no_phantom_column() {
    // FAILS today: row has 5 pipes (4 separators = phantom 4th col)
    // PASSES after FIX-A: row has 4 pipes (3 separators = correct)
    let theme = Theme::default_theme();
    let src = "| A | B | C |\n|---|---|---|\n| x | y | z |\n";
    let lines = render_markdown(src, &theme);
    // With 3 columns: each data row has exactly 4 │ glyphs (border + 2 separators + border)
    let header: String = lines[0].spans.iter().map(|s| s.content.as_ref()).collect();
    let pipe_count = header.chars().filter(|&c| c == '│').count();
    assert_eq!(pipe_count, 4, "3-column table must have exactly 4 │ glyphs, got {} in {:?}", pipe_count, header);
}
```
Today this FAILS (pipe_count=5). After FIX-A it passes.

### Test for FIX-B (heading level differentiation):
```rust
#[test]
fn heading_levels_are_visually_distinct() {
    // FAILS today: H1 and H6 both have only BOLD (no UNDERLINED, no ITALIC, no # prefix)
    // PASSES after FIX-B: H1 has UNDERLINED, H3 has ITALIC, all have # prefix
    let theme = Theme::default_theme();
    let src = "# H1\n\n## H2\n\n### H3\n\n###### H6\n";
    let lines = render_markdown(src, &theme);
    // H1 line must contain a "#" glyph AND at least one UNDERLINED span
    let h1_row: String = lines[0].spans.iter().map(|s| s.content.as_ref()).collect();
    assert!(h1_row.contains('#'), "H1 must have # prefix glyph: {:?}", h1_row);
    let h1_underline = lines[0].spans.iter().any(|s|
        s.style.add_modifier.contains(ratatui::style::Modifier::UNDERLINED));
    assert!(h1_underline, "H1 must have UNDERLINED modifier");
    // H3 must have ITALIC
    let h3_row_idx = lines.iter().position(|l| {
        let t: String = l.spans.iter().map(|s| s.content.as_ref()).collect();
        t.contains("H3")
    }).unwrap();
    let h3_italic = lines[h3_row_idx].spans.iter().any(|s|
        s.style.add_modifier.contains(ratatui::style::Modifier::ITALIC));
    assert!(h3_italic, "H3 must have ITALIC modifier");
}
```

### Test for FIX-C (blockquote multiline):
```rust
#[test]
fn blockquote_two_lines_render_as_two_rows() {
    // FAILS today: both lines collapse into one row "│ line1 line2"
    // PASSES after FIX-C: two separate rows each starting with "│ "
    let theme = Theme::default_theme();
    let src = "> line one\n> line two\n";
    let lines = render_markdown(src, &theme);
    let content_rows: Vec<String> = lines.iter()
        .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect::<String>())
        .filter(|t| t.contains("│ "))
        .collect();
    assert_eq!(content_rows.len(), 2,
        "two >-lines must produce two │-prefixed rows, got: {:?}", content_rows);
    assert!(content_rows[0].contains("line one"));
    assert!(content_rows[1].contains("line two"));
}
```

### Test for FIX-E (block math `\,` not corrupted):
```rust
#[test]
fn block_math_thin_space_not_corrupted() {
    // FAILS today: ∫ x,dx (comma from pulldown consuming \,)
    // PASSES after FIX-E: ∫ x dx (space from \, mapping)
    let theme = Theme::default_theme();
    let src = "$$\\int_0^1 x\\,dx$$\n";
    let lines = render_markdown(src, &theme);
    let joined: String = lines.iter()
        .flat_map(|l| l.spans.iter())
        .map(|s| s.content.as_ref())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!joined.contains(','), "\\, must not produce a literal comma: {:?}", joined);
    assert!(joined.contains('∫'), "integral glyph must be present");
}
```

### Test for FIX-F (nested list no spurious blank):
```rust
#[test]
fn nested_list_no_spurious_blank() {
    // FAILS today: blank row between nested list and item 3
    // PASSES after FIX-F: no blank row between "nested" and "Third"
    let theme = Theme::default_theme();
    let src = "1. First\n2. Second\n   1. Nested\n3. Third\n";
    let rows: Vec<String> = render_markdown(src, &theme).iter()
        .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
        .collect();
    // Find "Nested" row index, next row must be "3. Third" not blank
    let nested_idx = rows.iter().position(|r| r.contains("Nested")).unwrap();
    assert!(!rows[nested_idx + 1].is_empty(),
        "row after nested list item must not be blank, got: {:?}", &rows[nested_idx..]);
    assert!(rows[nested_idx + 1].contains("3."),
        "row after nested list must be item 3, got: {:?}", &rows[nested_idx + 1]);
}
```

---

## Priority Summary

| Gap | Element | Status | Priority | Fix |
|-----|---------|--------|----------|-----|
| A | GFM Table (phantom column) | RENDERS-WRONG | HIGH | FIX-A: remove double-push in TagEnd::TableHead + TableRow |
| B | Headings H1–H6 (no level cues) | RENDERS-WRONG | HIGH | FIX-B: add # prefix + per-level modifiers (bold+underline / bold / bold+italic / italic) |
| C | Blockquote multi-line collapse | RENDERS-WRONG | MEDIUM | FIX-C: SoftBreak → flush_line instead of space |
| D | HR narrow (8 dashes) | RENDERS-WRONG | LOW | FIX-D: thread width into Walker or use 80-char default |
| E | LaTeX `\,` corrupted by pulldown | RENDERS-WRONG | HIGH | FIX-E: pre-scan block-math paragraphs in raw source before pulldown |
| F | Nested-list spurious blank | RENDERS-WRONG | MEDIUM | FIX-F: don't set started=true on nested TagEnd::List |
| — | Bold/italic/strike | RENDERS-OK | — | no action needed |
| — | Inline math $…$ | RENDERS-OK | — | no action needed |
| — | Block math $$…$$ | RENDERS-OK (structurally) | — | Fix only for \, (Gap E) |
| — | Tables (overall) | RENDERS-OK (minus phantom col) | — | Fix only Gap A |
| — | Ordered + unordered lists | RENDERS-OK | — | no action needed (except Gap F for nesting) |
| — | Fenced code blocks | RENDERS-OK | — | no action needed |
| — | Links | RENDERS-OK | — | no action needed |
| — | Strikethrough | RENDERS-OK | — | no action needed |
| — | Task lists (☑ ☐) | RENDERS-OK | — | ENABLE_TASKLISTS set, glyphs present |

VERDICT: PARTIAL — five concrete bugs found: phantom table column (TagEnd::TableHead/Row double-push in render.rs), heading levels all identical (only BOLD, missing # prefix/underline/italic), blockquote multiline collapse (SoftBreak→space not newline), narrow HR (8 hardcoded dashes), and LaTeX `\,` corruption (pulldown backslash-escape consumed before math pipeline).
