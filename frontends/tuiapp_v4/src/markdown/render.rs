//! markdown/render.rs — the `pulldown-cmark` `Walker`: events → themed `ratatui`
//! Lines/Spans (checklist §1 P3, §3 markdown/render). Parse markdown to events,
//! then WALK them emitting styled spans — never a live re-parsed widget. Handles
//! headings / inline styles / lists / blockquotes inline; delegates tables to
//! [`super::table`], fenced code to [`super::code`], and `$…$`/`$$…$$` math to
//! [`super::inline_math`] (split before walking, since pulldown can't tokenize `$`).
//!
//! All colors come from [`Theme`] tokens (no hardcoded RGB). Output is a list of
//! LOGICAL pre-styled rows; soft-wrapping to a width is the caller's wrap cache.
//! PURE over `(source, theme)`.

use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use crate::theme::{Theme, Token};

use super::code::{self, CodeBuf};
use super::inline_math::{split_inline_math, Seg};
use super::math::{self, Display};
use super::table::{self, TableBuf};

/// Out-of-band tag on a [`Span`] whose rendered content must NEVER be split by
/// soft-wrap (a rendered inline-math glyph run — `∑ᵢ₌₀ⁿ`, `(a+b)/c`, …). ratatui
/// `Span`s carry no "atomic" flag, so we piggyback a `Modifier` bit unused by the
/// theme (`RAPID_BLINK`; grep confirms nothing else emits it) as a side channel
/// the wrapper reads. It is STRIPPED before the span is drawn ([`super::flush_atomic_style`])
/// so it never becomes a real visual style. The wrapper honoring it lives in
/// [`super::wrap_styled_line`] + [`crate::render::measure::wrap_line_segments_atomic`].
pub(crate) const ATOMIC: Modifier = Modifier::RAPID_BLINK;

/// Render markdown `source` into themed lines. PUBLIC entry point (P3).
///
/// The result is a flat list of styled rows in document order, with one blank
/// line between block elements (CC's `gap={1}` spacing). It never panics: the
/// math sub-renderer degrades to literal LaTeX, the code highlighter degrades to
/// plain text, and unknown constructs fall back to their text content.
///
/// FIX-E: Before handing the source to pulldown-cmark (which consumes `\<non-alpha>`
/// as markdown escape sequences), we pre-scan for whole-paragraph `$$…$$` blocks
/// and render them directly via the block-math path, splicing them back into the
/// output after pulldown processes the rest. This prevents `\,` (thin space) and
/// similar LaTeX commands from being corrupted.
///
/// FIX-D: accepts an optional `width` for the HR rule. Pass `None` to use the
/// default wide value (80).
pub fn render_markdown(source: &str, theme: &Theme) -> Vec<Line<'static>> {
    render_markdown_width(source, theme, 80)
}

/// Like [`render_markdown`] but thread a terminal `width` for width-aware HR
/// rendering (FIX-D).
pub fn render_markdown_width(source: &str, theme: &Theme, width: u16) -> Vec<Line<'static>> {
    // FIX-E: Pre-scan for `$$…$$` block-math paragraphs BEFORE pulldown can eat
    // the backslash escapes inside them. We split the source on paragraph
    // boundaries (blank lines), check each paragraph, and short-circuit the math
    // ones to the block-math renderer; the rest go through pulldown normally.
    //
    // Implementation strategy: replace each block-math paragraph with a unique
    // ASCII-only placeholder line, let pulldown parse the placeholder-substituted
    // source, then splice the pre-rendered block-math lines back at the
    // placeholder positions in the output.
    let (modified_source, math_blocks) = extract_block_math_paragraphs(source);

    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_TABLES);
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_TASKLISTS);
    let parser = Parser::new_ext(&modified_source, opts);

    let mut w = Walker::new(theme, width);
    for ev in parser {
        w.event(ev);
    }
    let mut out = w.finish();

    // Splice the pre-rendered block-math lines back at placeholder positions.
    if !math_blocks.is_empty() {
        splice_math_blocks(&mut out, &math_blocks, theme);
    }
    out
}

/// Pre-scan `source` for whole-paragraph `$$…$$` blocks (FIX-E). Returns:
///   - a modified source where each block-math paragraph is replaced with a
///     placeholder line `\x01BLOCKMATH{idx}\x01` (pulldown treats it as a
///     paragraph of plain text, which we then replace in the output).
///   - a vec of `(idx, latex_body)` pairs for the detected blocks.
fn extract_block_math_paragraphs(source: &str) -> (String, Vec<(usize, String)>) {
    // Split on paragraph boundaries (one or more blank lines between blocks).
    // We keep the exact original offsets so we can reconstruct the source with
    // placeholders.
    let mut math_blocks: Vec<(usize, String)> = Vec::new();
    let mut result = String::with_capacity(source.len());
    let mut idx = 0usize;

    // Walk paragraph chunks: split on double-newline runs.
    let mut start = 0usize;
    let bytes = source.as_bytes();
    let len = bytes.len();
    let mut i = 0usize;

    while i <= len {
        // Detect the end of a paragraph chunk: two or more consecutive newlines,
        // OR the end of the string.
        let is_end = i == len;
        let is_blank_sep = !is_end
            && bytes[i] == b'\n'
            && (i + 1 < len && (bytes[i + 1] == b'\n' || bytes[i + 1] == b'\r'));

        if is_end || is_blank_sep {
            let chunk = &source[start..i];
            let trimmed = chunk.trim();

            if let Some(latex) = extract_single_block_math(trimmed) {
                result.push_str(&format!("\x01BLOCKMATH{idx}\x01"));
                math_blocks.push((idx, latex));
                idx += 1;
            } else {
                result.push_str(chunk);
            }

            // End of source: done. Break BEFORE the separator-advance so `i`
            // (== len) can never spin — the bug that hung every markdown render.
            if is_end {
                break;
            }

            // Advance past the blank-line separator, preserving it verbatim so the
            // reconstructed source keeps paragraph boundaries.
            let sep_start = i;
            let mut j = i;
            while j < len && (bytes[j] == b'\n' || bytes[j] == b'\r') {
                j += 1;
            }
            result.push_str(&source[sep_start..j]);
            i = j;
            start = i;
            continue;
        }
        i += 1;
    }

    (result, math_blocks)
}

/// Detect a SINGLE `$$…$$` block in a trimmed paragraph string. The interior must
/// be non-empty and contain no blank lines (a block-math paragraph is a single $$
/// run). Returns the inner LaTeX body if detected.
fn extract_single_block_math(trimmed: &str) -> Option<String> {
    if trimmed.len() < 4 {
        return None;
    }
    if !trimmed.starts_with("$$") || !trimmed.ends_with("$$") {
        return None;
    }
    // Must not start AND end with `$$` where the whole string IS `$$` (len=2
    // per-side = 4 chars minimum, inner = trimmed[2..len-2]).
    // Edge: `$$` itself (len=2) is handled by the len<4 guard above.
    let inner = &trimmed[2..trimmed.len() - 2];
    // The interior must not be empty and must not contain a blank line (those
    // would indicate a new paragraph boundary, not a single block).
    if inner.trim().is_empty() {
        return None;
    }
    // Reject if inner contains `\n\n` (multi-paragraph — not a block-math para).
    if inner.contains("\n\n") {
        return None;
    }
    // Also reject if the opening `$$` is followed immediately by `$$` (the `$$$$`
    // edge case where inner="" — already caught by trim().is_empty() above).
    Some(inner.trim().to_string())
}

/// Splice pre-rendered block-math lines back into `out` at placeholder positions
/// (FIX-E). A placeholder paragraph renders as a single text line whose content
/// is the placeholder string `\x01BLOCKMATH{n}\x01`. We find those lines in `out`
/// and replace them (and their surrounding blank-gap lines) with the math rows.
fn splice_math_blocks(
    out: &mut Vec<Line<'static>>,
    math_blocks: &[(usize, String)],
    theme: &Theme,
) {
    for (idx, latex) in math_blocks {
        let placeholder = format!("\x01BLOCKMATH{idx}\x01");
        // Find the line whose concatenated span text equals the placeholder.
        let pos = out.iter().position(|l| {
            let t: String = l.spans.iter().map(|s| s.content.as_ref()).collect();
            t.trim() == placeholder.as_str()
        });
        if let Some(p) = pos {
            let math_lines = super::inline_math::render_block_math(latex, theme);
            // Replace just the placeholder row with the math rows.
            out.splice(p..p + 1, math_lines);
        }
    }
}

/// The walker state: a list of completed lines plus the in-progress line's spans,
/// an indent/list/quote context stack, and buffers for tables and code blocks.
struct Walker<'t> {
    theme: &'t Theme,
    /// Terminal width for width-aware rendering (FIX-D: HR rule).
    width: u16,
    /// Completed output rows.
    out: Vec<Line<'static>>,
    /// Spans accumulated for the current (not-yet-flushed) row.
    cur: Vec<Span<'static>>,
    /// Active inline style modifiers (bold/italic/strike/code), nestable.
    style_stack: Vec<Style>,
    /// Per-level list state: `Some(n)` = ordered (next number), `None` = bullet.
    lists: Vec<ListState>,
    /// Blockquote nesting depth (drives the `│ ` gutter count).
    quote_depth: usize,
    /// True while inside a fenced code block; collects its raw text + language.
    code: Option<CodeBuf>,
    /// True while inside a table; buffers cells for width-aligned emission.
    table: Option<TableBuf>,
    /// Whether the current paragraph is the FIRST block (suppress a leading blank).
    started: bool,
    /// Pending link URL (rendered as ` (url)` on the link's close).
    link_url: Option<String>,
    /// Buffered run of consecutive inline text at the current style, coalesced so
    /// inline `$…$` math that pulldown fragmented at `_`/`*` boundaries is whole
    /// before the math split. Flushed at every non-text boundary + each newline.
    pending: String,
    /// While inside a `Tag::Paragraph`, the paragraph's raw text content (SoftBreaks
    /// as spaces) so the close can detect a lone `$$…$$` display-math paragraph and
    /// route it to stacked block math instead of the inline collapse (Fix B). `None`
    /// outside a paragraph.
    para: Option<String>,
}

#[derive(Clone)]
struct ListState {
    /// `Some(next_number)` for ordered lists; `None` for bullets.
    ordered: Option<u64>,
}

impl<'t> Walker<'t> {
    fn new(theme: &'t Theme, width: u16) -> Self {
        Walker {
            theme,
            width,
            out: Vec::new(),
            cur: Vec::new(),
            style_stack: Vec::new(),
            lists: Vec::new(),
            quote_depth: 0,
            code: None,
            table: None,
            started: false,
            link_url: None,
            pending: String::new(),
            para: None,
        }
    }

    fn col(&self, tok: Token) -> Style {
        Style::default().fg(self.theme.color(tok))
    }

    /// The current effective inline style (top of the stack, or default text).
    fn cur_style(&self) -> Style {
        self.style_stack
            .last()
            .copied()
            .unwrap_or_else(|| self.col(Token::Text))
    }

    /// Push raw text as a span in the current inline style — UNLESS we're inside
    /// a code block or table cell (those buffer raw text instead).
    ///
    /// A text run may carry an embedded `\n` (notably from raw/inline HTML events
    /// like `<summary>…</summary>\n`, which pulldown-cmark hands us verbatim). A
    /// logical [`Line`] must represent exactly ONE hard line with NO embedded
    /// newline — that invariant is what keeps the styled soft-wrap
    /// ([`super::wrap_styled_line`]) row-count-identical to the wrap cache, which
    /// splits the plain projection on `\n` (see [`super::lines_to_plain`] +
    /// [`crate::render::measure::reflow_block`]). So we split the run on `\n` here
    /// and flush a line per newline; the remainder continues the current line.
    fn push_text(&mut self, text: &str) {
        if let Some(code) = self.code.as_mut() {
            code.text.push_str(text);
            return;
        }
        if let Some(tb) = self.table.as_mut() {
            tb.cur_cell.push_str(text);
            return;
        }
        // Mirror the run into the paragraph buffer so the close can spot a lone
        // `$$…$$` display-math paragraph (Fix B). Inline HTML/code flow here too,
        // but a block-math paragraph is pure text, so the predicate still gates it.
        if let Some(p) = self.para.as_mut() {
            p.push_str(text);
        }
        // Split on `\n` (no span may carry an embedded newline), accumulating into
        // `pending` so a paragraph pulldown fragmented at `_`/`*` reassembles whole
        // before the inline-math split (e.g. `$\sum_{i}$` → 3 Text events).
        let mut first = true;
        for chunk in text.split('\n') {
            if !first {
                self.flush_pending();
                self.flush_line();
            }
            first = false;
            self.pending.push_str(chunk);
        }
    }

    /// Flush the buffered inline-text run (`pending`) through the inline-math
    /// splitter in the CURRENT style. This is where inline `$…$` is actually
    /// converted — on the REASSEMBLED run, so math split across pulldown Text
    /// events (at `_`/`*`) is recognized. Called at every non-text boundary + each
    /// newline; a no-op when the buffer is empty.
    fn flush_pending(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let run = std::mem::take(&mut self.pending);
        self.push_inline_run(&run);
    }

    /// Push one newline-free text run (math-split into spans) onto the current
    /// logical line. Callers guarantee `run` contains no `\n`.
    fn push_inline_run(&mut self, run: &str) {
        if run.is_empty() {
            return;
        }
        // Inline math extraction: split the text run on `$…$` / `$$…$$`.
        let style = self.cur_style();
        for seg in split_inline_math(run) {
            match seg {
                Seg::Text(t) => {
                    if !t.is_empty() {
                        self.cur.push(Span::styled(t, style));
                    }
                }
                Seg::Math(latex) => {
                    // Inline math is ONE line (Display::Inline never stacks), so the
                    // rendered glyph run is a single span. Tag it ATOMIC so soft-wrap
                    // (styled draw + plain cache) never splits mid-formula (F1).
                    let rendered = math::render_math(&latex, Display::Inline);
                    self.cur.push(Span::styled(
                        rendered,
                        self.col(Token::Claude).add_modifier(ATOMIC),
                    ));
                }
            }
        }
    }

    /// Finish the current line (even if empty when forced) and start a new one.
    fn flush_line(&mut self) {
        let spans = std::mem::take(&mut self.cur);
        // Prepend the blockquote gutter if we're inside one.
        let line = if self.quote_depth > 0 {
            let mut prefixed: Vec<Span<'static>> = Vec::with_capacity(spans.len() + 1);
            let gutter = "│ ".repeat(self.quote_depth);
            prefixed.push(Span::styled(gutter, self.col(Token::Dim)));
            prefixed.extend(spans);
            Line::from(prefixed)
        } else {
            Line::from(spans)
        };
        self.out.push(line);
    }

    /// Emit a blank separator line between blocks (CC's `gap={1}`), but never as
    /// the very first row and never two in a row.
    fn block_gap(&mut self) {
        if !self.started {
            return;
        }
        if self
            .out
            .last()
            .map(|l| l.spans.is_empty())
            .unwrap_or(false)
        {
            return; // already blank
        }
        self.out.push(Line::default());
    }

    /// The hanging-indent prefix for the current list nesting (two spaces/level).
    fn list_indent(&self) -> String {
        "  ".repeat(self.lists.len().saturating_sub(1))
    }

    fn event(&mut self, ev: Event<'_>) {
        // Any non-inline-text event first flushes the pending run, so a coalesced
        // inline-math run never crosses a real style/structural boundary.
        if !matches!(ev, Event::Text(_) | Event::Html(_) | Event::InlineHtml(_)) {
            self.flush_pending();
        }
        match ev {
            Event::Start(tag) => self.start(tag),
            Event::End(tag) => self.end(tag),
            Event::Text(t) => self.push_text(&t),
            Event::Code(c) => {
                if let Some(tb) = self.table.as_mut() {
                    tb.cur_cell.push_str(&c);
                } else {
                    self.cur
                        .push(Span::styled(c.to_string(), self.col(Token::Claude)));
                }
            }
            Event::SoftBreak => {
                // FIX-C: Inside a blockquote, a soft break must produce a NEW ROW
                // so that `> line1\n> line2` renders as two separate `│ `-prefixed
                // rows (matching tui_v3's HardBreakMarkdown behaviour). Outside a
                // blockquote, keep the original space so ordinary prose doesn't
                // reflow (one visual line per source line would break parity for
                // wrapped paragraphs since the wrap cache uses word-wrap, not hard
                // line-per-source-line).
                if self.code.is_none() && self.table.is_none() {
                    if self.quote_depth > 0 {
                        // Blockquote: treat as a hard line break → two rows.
                        self.flush_pending();
                        self.flush_line();
                    } else {
                        self.cur.push(Span::raw(" "));
                    }
                    if let Some(p) = self.para.as_mut() {
                        p.push(' ');
                    }
                }
            }
            Event::HardBreak => {
                if self.code.is_none() && self.table.is_none() {
                    self.flush_line();
                    self.indent_continuation();
                }
            }
            Event::Rule => {
                // FIX-D: use the threaded terminal width for a full-width rule.
                // Subtract 2 to leave a margin; clamp to at least 4.
                self.block_gap();
                let w = (self.width as usize).saturating_sub(2).max(4);
                self.out.push(Line::from(Span::styled(
                    "─".repeat(w),
                    self.col(Token::Border),
                )));
                self.started = true;
            }
            Event::TaskListMarker(done) => {
                let glyph = if done { "☑ " } else { "☐ " };
                self.cur
                    .push(Span::styled(glyph.to_string(), self.col(Token::Success)));
            }
            Event::Html(h) | Event::InlineHtml(h) => {
                self.push_text(&h);
            }
            _ => {}
        }
    }

    /// Re-emit the list/indent prefix at the start of a continuation row (after a
    /// hard break inside a list item) so wrapped content stays aligned.
    fn indent_continuation(&mut self) {
        if !self.lists.is_empty() {
            let pad = self.list_indent();
            // +2 to clear the bullet/number gutter width.
            let pad = format!("{pad}  ");
            self.cur.push(Span::raw(pad));
        }
    }

    fn start(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Paragraph => {
                self.block_gap();
                self.para = Some(String::new());
            }
            Tag::Heading { level, .. } => {
                // FIX-B: differentiate the six levels by MODIFIER + per-level color
                // (H1=BOLD|UNDERLINED, H2=BOLD, H3=BOLD|ITALIC, H4-H6=ITALIC). The
                // round-4 rule stands: NO bare `#`/`##` glyph — the level is conveyed
                // by style alone (the clean tui_v3/CC look), never a hash prefix.
                self.block_gap();
                let tok = code::heading_style(level);
                let base_style = Style::default().fg(self.theme.color(tok));
                let heading_style = match level {
                    pulldown_cmark::HeadingLevel::H1 => {
                        base_style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
                    }
                    pulldown_cmark::HeadingLevel::H2 => {
                        base_style.add_modifier(Modifier::BOLD)
                    }
                    pulldown_cmark::HeadingLevel::H3 => {
                        base_style.add_modifier(Modifier::BOLD | Modifier::ITALIC)
                    }
                    _ => {
                        // H4, H5, H6
                        base_style.add_modifier(Modifier::ITALIC)
                    }
                };
                // No hash prefix (round-4 rule) — just push the level style; the
                // heading text inherits BOLD/UNDERLINED/ITALIC + color from it.
                self.style_stack.push(heading_style);
            }
            Tag::BlockQuote(_) => {
                self.block_gap();
                self.quote_depth += 1;
            }
            Tag::CodeBlock(kind) => {
                self.block_gap();
                let lang = match kind {
                    pulldown_cmark::CodeBlockKind::Fenced(info) => {
                        info.split_whitespace().next().unwrap_or("").to_string()
                    }
                    pulldown_cmark::CodeBlockKind::Indented => String::new(),
                };
                self.code = Some(CodeBuf {
                    lang,
                    text: String::new(),
                });
            }
            Tag::List(start) => {
                self.lists.push(ListState { ordered: start });
                // A nested list shouldn't add a blank gap before its first item.
                if self.lists.len() == 1 {
                    self.block_gap();
                }
            }
            Tag::Item => {
                self.flush_pending_line_if_content();
                let indent = self.list_indent();
                let marker = match self.lists.last_mut() {
                    Some(ListState {
                        ordered: Some(n), ..
                    }) => {
                        let m = format!("{n}. ");
                        *self.lists.last_mut().unwrap() = ListState {
                            ordered: Some(*n + 1),
                        };
                        m
                    }
                    _ => {
                        let depth = self.lists.len().saturating_sub(1);
                        format!("{} ", code::bullet(depth))
                    }
                };
                self.cur.push(Span::raw(indent));
                self.cur
                    .push(Span::styled(marker, self.col(Token::Suggestion)));
            }
            Tag::Emphasis => {
                let base = self.cur_style();
                self.style_stack.push(base.add_modifier(Modifier::ITALIC));
            }
            Tag::Strong => {
                let base = self.cur_style();
                self.style_stack.push(base.add_modifier(Modifier::BOLD));
            }
            Tag::Strikethrough => {
                let base = self.cur_style();
                self.style_stack
                    .push(base.add_modifier(Modifier::CROSSED_OUT));
            }
            Tag::Link { dest_url, .. } => {
                self.link_url = Some(dest_url.to_string());
                self.style_stack
                    .push(self.col(Token::Suggestion).add_modifier(Modifier::UNDERLINED));
            }
            Tag::Image { dest_url, .. } => {
                // No inline images in a TUI; show a text marker.
                self.cur.push(Span::styled(
                    format!("[image: {dest_url}]"),
                    self.col(Token::Dim),
                ));
            }
            Tag::Table(aligns) => {
                self.block_gap();
                self.table = Some(TableBuf {
                    alignments: aligns,
                    header: Vec::new(),
                    rows: Vec::new(),
                    cur_cell: String::new(),
                    cur_row: Vec::new(),
                    in_header: false,
                });
            }
            Tag::TableHead => {
                if let Some(tb) = self.table.as_mut() {
                    tb.in_header = true;
                }
            }
            Tag::TableRow => {
                if let Some(tb) = self.table.as_mut() {
                    tb.cur_row = Vec::new();
                }
            }
            Tag::TableCell => {
                if let Some(tb) = self.table.as_mut() {
                    tb.cur_cell = String::new();
                }
            }
            _ => {}
        }
    }

    fn end(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph => {
                // Fix B: a paragraph whose trimmed text is exactly a `$$…$$` block
                // with a non-empty interior is DISPLAY math — emit it as the stacked
                // multi-line layout instead of the single inline-collapsed line, so
                // display math renders inside a normal multi-part assistant message.
                let para = self.para.take();
                if let Some(latex) = para
                    .as_deref()
                    .and_then(super::inline_math::extract_block_math)
                {
                    self.cur.clear(); // drop the inline-collapsed `$$…$$` span
                    self.out
                        .extend(super::inline_math::render_block_math(&latex, self.theme));
                } else {
                    self.flush_line();
                }
                self.started = true;
            }
            TagEnd::Heading(_) => {
                self.style_stack.pop();
                self.flush_line();
                self.started = true;
            }
            TagEnd::BlockQuote(_) => {
                self.quote_depth = self.quote_depth.saturating_sub(1);
                self.started = true;
            }
            TagEnd::CodeBlock => {
                if let Some(code) = self.code.take() {
                    self.out.extend(code::emit_code_block(self.theme, &code));
                }
                self.started = true;
            }
            TagEnd::List(_) => {
                self.lists.pop();
                // FIX-F: only set started=true when we just popped the TOP-LEVEL
                // list. For a nested list end, the parent list's item-gap management
                // owns spacing — setting started=true here causes block_gap() to
                // insert a spurious blank before the next sibling item.
                if self.lists.is_empty() {
                    self.started = true;
                }
            }
            TagEnd::Item => {
                // FIX-F: flush only when the item line still holds content. An item
                // that CONTAINS a nested list already had its text flushed when the
                // nested list's first item started, so `cur` is empty here — an
                // unconditional flush_line() would emit a spurious blank row before
                // the next sibling item.
                self.flush_pending_line_if_content();
            }
            TagEnd::Emphasis
            | TagEnd::Strong
            | TagEnd::Strikethrough => {
                self.style_stack.pop();
            }
            TagEnd::Link => {
                self.style_stack.pop();
                if let Some(url) = self.link_url.take() {
                    // `text (url)` — the destination after the link text.
                    self.cur.push(Span::styled(
                        format!(" ({url})"),
                        self.col(Token::Dim),
                    ));
                }
            }
            TagEnd::Table => {
                if let Some(tb) = self.table.take() {
                    self.out.extend(table::emit_table(self.theme, &tb));
                }
                self.started = true;
            }
            TagEnd::TableHead => {
                // FIX-A: TagEnd::TableCell already flushed the last cell into
                // cur_row before this fires, so cur_cell is empty here. The old
                // code did an extra take+push of the (empty) cur_cell, producing a
                // phantom empty column. Remove the redundant push.
                if let Some(tb) = self.table.as_mut() {
                    tb.header = std::mem::take(&mut tb.cur_row);
                    tb.in_header = false;
                }
            }
            TagEnd::TableRow => {
                // FIX-A: same phantom-column fix as TableHead above.
                if let Some(tb) = self.table.as_mut() {
                    if !tb.in_header {
                        let row = std::mem::take(&mut tb.cur_row);
                        tb.rows.push(row);
                    }
                }
            }
            TagEnd::TableCell => {
                if let Some(tb) = self.table.as_mut() {
                    let cell = std::mem::take(&mut tb.cur_cell);
                    tb.cur_row.push(cell);
                }
            }
            _ => {}
        }
    }

    /// If the current line already has content (e.g. a list item's text), flush it
    /// before starting a new item so items don't run together.
    fn flush_pending_line_if_content(&mut self) {
        if !self.cur.is_empty() {
            self.flush_line();
        }
    }

    /// Flush any dangling content and return the accumulated lines.
    fn finish(mut self) -> Vec<Line<'static>> {
        self.flush_pending();
        if !self.cur.is_empty() {
            self.flush_line();
        }
        // Trim a single trailing blank line.
        if self
            .out
            .last()
            .map(|l| l.spans.is_empty())
            .unwrap_or(false)
        {
            self.out.pop();
        }
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(theme: &Theme, src: &str) -> Vec<String> {
        render_markdown(src, theme)
            .into_iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect()
    }

    /// Fix B acceptance (Q3): a `$$…$$` display equation sitting AMONG prose in a
    /// normal multi-part message renders as the multi-line STACKED layout (the whole
    /// `Display::Block` machinery), not the single inline-collapsed `a/b` line — and
    /// the surrounding prose still renders, with no raw `$$` leaking.
    #[test]
    fn block_math_renders_inside_prose_paragraph() {
        let theme = Theme::default_theme();
        let src = "Here is the closed form:\n\n$$\\frac{a}{b}$$\n\nwhich holds for all b.";
        let rows = plain(&theme, src);
        let joined = rows.join("\n");
        // Surrounding prose is preserved.
        assert!(joined.contains("Here is the closed form:"), "intro prose kept: {joined:?}");
        assert!(joined.contains("which holds for all b."), "outro prose kept: {joined:?}");
        // No raw delimiter leaks.
        assert!(!joined.contains("$$"), "raw `$$` must not leak: {joined:?}");
        assert!(!joined.contains('$'), "no literal `$` leaks: {joined:?}");
        // STACKED (not inline): the fraction bar row exists as a row of `─` only —
        // an inline collapse (`a/b`) would have no such standalone bar line.
        assert!(
            rows.iter().any(|r| !r.is_empty() && r.chars().all(|c| c == '─')),
            "display math must stack (a `─` fraction-bar row present): {rows:?}"
        );
        // And the numerator/denominator atoms render (the math actually ran).
        assert!(joined.contains('a') && joined.contains('b'));
    }

    /// Fix F acceptance (Q8, "绝不能再出现 Turn 1 …"): scan the FULL rendered output
    /// of a multi-turn folded transcript and assert the literal substring "Turn"
    /// appears NOWHERE — not in a fold title fallback, not in a stripped marker.
    #[test]
    fn no_turn_n_anywhere() {
        let theme = Theme::default_theme();
        // Three turns. Turn 1: a tool but NO summary (so its fold title can only come
        // from the tool name, never "Turn 1 · …"). Turn 2: NO summary, NO tool, NO
        // prose (so it falls to the neutral ellipsis, never "Turn 2"). Turn 3 (last)
        // stays expanded; its marker must be stripped, not rendered.
        let src = "\
Turn 1 ...
🛠️ web_search(query: rust)
[Info] ok
Turn 2 ...
Turn 3 ...
the final answer is 42";
        for fold_all in [false, true] {
            for width in [40u16, 80, 120] {
                let plain = super::super::render_assistant_cockpit_plain(src, &theme, fold_all, width);
                assert!(
                    !plain.contains("Turn"),
                    "rendered output leaked a 'Turn' substring (fold_all={fold_all} width={width}): {plain:?}"
                );
                // The real content still renders.
                assert!(plain.contains("web_search"), "tool name title present: {plain:?}");
            }
        }
    }

    #[test]
    fn md_headings_and_inline_styles() {
        let theme = Theme::default_theme();
        let lines = render_markdown("# Title\n\nSome **bold** and *italic* and `code`.", &theme);
        // First content line is the heading text.
        let h = &lines[0];
        let htext: String = h.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(htext.contains("Title"));
        // Heading is bold.
        assert!(h.spans.iter().any(|s| s.style.add_modifier.contains(Modifier::BOLD)));

        // Find the paragraph line and check bold/italic spans exist with the
        // right modifiers.
        let para: String = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .map(|s| s.content.as_ref())
            .collect();
        assert!(para.contains("bold"));
        assert!(para.contains("italic"));
        assert!(para.contains("code"));
        let has_bold = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .any(|s| s.content.as_ref() == "bold" && s.style.add_modifier.contains(Modifier::BOLD));
        assert!(has_bold, "**bold** span must carry BOLD");
        let has_italic = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .any(|s| {
                s.content.as_ref() == "italic" && s.style.add_modifier.contains(Modifier::ITALIC)
            });
        assert!(has_italic, "*italic* span must carry ITALIC");
    }

    #[test]
    fn md_lists_with_bullets_and_numbers() {
        let theme = Theme::default_theme();
        let lines = plain(&theme, "- first\n- second\n\n1. one\n2. two");
        let joined = lines.join("\n");
        assert!(joined.contains("• first"));
        assert!(joined.contains("• second"));
        assert!(joined.contains("1. one"));
        assert!(joined.contains("2. two"));
    }

    #[test]
    fn md_blockquote_gutter() {
        let theme = Theme::default_theme();
        let lines = plain(&theme, "> quoted text");
        let joined = lines.join("\n");
        assert!(joined.contains("│ "), "blockquote shows a gutter: {joined:?}");
        assert!(joined.contains("quoted text"));
    }

    #[test]
    fn md_link_renders_text_and_url() {
        let theme = Theme::default_theme();
        let lines = plain(&theme, "see [the docs](https://example.com) here");
        let joined = lines.join("\n");
        assert!(joined.contains("the docs"));
        assert!(joined.contains("(https://example.com)"));
    }

    #[test]
    fn md_never_panics_on_weird_input() {
        let theme = Theme::default_theme();
        // Unclosed fence, half a table, dangling emphasis, stray math — all must
        // render *something* without panicking.
        let _ = render_markdown("```\nunclosed", &theme);
        let _ = render_markdown("| a | b\n| - |", &theme);
        let _ = render_markdown("**bold without close", &theme);
        let _ = render_markdown("math $\\frac{1}{ unclosed", &theme);
        let _ = render_markdown("", &theme);
        let _ = render_markdown("你好 **世界** `代码`", &theme);
    }

    #[test]
    fn rich_fixture_recon_all_elements() {
        // RECON test: feed the full fixture through the live path and print every row.
        let theme = Theme::default_theme();
        let fixture = "# H1 Heading\n\n## H2 Heading\n\n### H3 Heading\n\n#### H4 Heading\n\n##### H5 Heading\n\n###### H6 Heading\n\n| Left | Center | Right |\n|:-----|:------:|------:|\n| a    | b      | c     |\n| long cell | x | 99 |\n\n**bold text** and *italic text* and `inline code` and ~~strikethrough~~ text\n\n1. First item\n2. Second item\n   1. Nested ordered\n3. Third\n\n- Bullet one\n- Bullet two\n  - Nested bullet\n  - Another nested\n- Bullet three\n\n> This is a blockquote\n> with two lines\n\n[a link](https://example.com)\n\n```rust\nfn hello() {\n    println!(\"world\");\n}\n```\n\n---\n\nInline math: $E=mc^2$ and $\\sum_{i=0}^{n} x_i$\n\nBlock math paragraph:\n\n$$\\int_0^1 x\\,dx$$\n";

        let lines = render_markdown(fixture, &theme);
        eprintln!("\n=== RICH FIXTURE RENDERED ({} rows) ===", lines.len());
        for (i, line) in lines.iter().enumerate() {
            let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
            let mods: Vec<String> = line.spans.iter().flat_map(|s| {
                let mut v = vec![];
                if s.style.add_modifier.contains(Modifier::BOLD) { v.push("BOLD".to_string()); }
                if s.style.add_modifier.contains(Modifier::ITALIC) { v.push("ITAL".to_string()); }
                if s.style.add_modifier.contains(Modifier::CROSSED_OUT) { v.push("STRIKE".to_string()); }
                v
            }).collect();
            let mods_str = if mods.is_empty() { String::new() } else { format!(" [{:?}]", mods) };
            eprintln!("  [{:03}] {:?}{}", i, text, mods_str);
        }
        eprintln!("=== END ===");

        let all: String = lines.iter().flat_map(|l| l.spans.iter()).map(|s| s.content.as_ref()).collect::<Vec<_>>().join("");

        // Presence assertions (all must pass for RENDERS-OK):
        assert!(all.contains("H1 Heading"), "H1 missing");
        assert!(all.contains("H2 Heading"), "H2 missing");
        assert!(all.contains("H3 Heading"), "H3 missing");
        assert!(all.contains("H4 Heading"), "H4 missing");
        assert!(all.contains("H5 Heading"), "H5 missing");
        assert!(all.contains("H6 Heading"), "H6 missing");
        assert!(all.contains("Left") && all.contains("Center") && all.contains("Right"), "table header missing");
        assert!(all.contains("long cell"), "table body missing");
        assert!(all.contains("bold text"), "bold text missing");
        assert!(all.contains("italic text"), "italic text missing");
        assert!(all.contains("inline code"), "inline code missing");
        assert!(all.contains("strikethrough"), "strikethrough text missing");
        assert!(all.contains("First item"), "ordered list missing");
        assert!(all.contains("•"), "bullet missing");
        assert!(all.contains("│ ") && all.contains("blockquote"), "blockquote missing");
        assert!(all.contains("a link") && all.contains("https://example.com"), "link missing");
        assert!(all.contains("fn hello"), "fenced code missing");
        // HR: a row of all ─
        assert!(lines.iter().any(|l| {
            let t: String = l.spans.iter().map(|s| s.content.as_ref()).collect();
            !t.is_empty() && t.chars().all(|c| c == '─')
        }), "HR missing");
        // Math
        assert!(!all.contains('$'), "raw $ leaked");
        assert!(all.contains('∑'), "inline sum math missing");
        // Block integral
        assert!(all.contains('∫') || lines.iter().any(|l| {
            let t: String = l.spans.iter().map(|s| s.content.as_ref()).collect();
            t.chars().all(|c| c == '─') && !t.is_empty()
        }), "block math (integral/bar) missing");
        // Strikethrough modifier
        assert!(lines.iter().flat_map(|l| l.spans.iter()).any(|s| {
            s.style.add_modifier.contains(Modifier::CROSSED_OUT)
        }), "CROSSED_OUT modifier missing on strikethrough");
    }

    // ── HONEST-CHECK TESTS (R2 §5) ─────────────────────────────────────────────
    // Each test exercises the LIVE path (render_markdown → Vec<Line>) and is
    // designed to FAIL on the old code and PASS after the corresponding FIX.

    /// FIX-A: a 3-column GFM table must have exactly 4 `│` glyphs per row (border
    /// + 2 separators + border = 4), not 5 (phantom 4th column from the old
    /// double-push in TagEnd::TableHead / TagEnd::TableRow).
    #[test]
    fn table_no_phantom_column() {
        let theme = Theme::default_theme();
        let src = "| A | B | C |\n|---|---|---|\n| x | y | z |\n";
        let lines = render_markdown(src, &theme);
        // The first non-blank line is the table header.
        let header: String = lines
            .iter()
            .find(|l| {
                let t: String = l.spans.iter().map(|s| s.content.as_ref()).collect();
                t.contains('│')
            })
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .unwrap_or_default();
        let pipe_count = header.chars().filter(|&c| c == '│').count();
        assert_eq!(
            pipe_count, 4,
            "3-column table must have exactly 4 │ glyphs (no phantom col), got {} in {:?}",
            pipe_count, header
        );
    }

    /// FIX-B: headings must be visually distinct by STYLE (no bare hash — round-4
    /// rule): H1 has `UNDERLINED`, H3 has `ITALIC`; the level is conveyed by
    /// modifier + per-level color, never a `#` glyph.
    #[test]
    fn heading_levels_are_visually_distinct() {
        let theme = Theme::default_theme();
        let src = "# H1\n\n## H2\n\n### H3\n\n###### H6\n";
        let lines = render_markdown(src, &theme);

        // H1: NO bare hash glyph (round-4 clean look) AND an UNDERLINED span.
        let h1_row: String = lines[0].spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(!h1_row.contains('#'), "H1 must NOT show a bare hash (round-4 rule): {:?}", h1_row);
        let h1_underline = lines[0]
            .spans
            .iter()
            .any(|s| s.style.add_modifier.contains(Modifier::UNDERLINED));
        assert!(h1_underline, "H1 must have UNDERLINED modifier; spans: {:?}", lines[0].spans);

        // H3: must have ITALIC modifier on the text span.
        let h3_row_idx = lines
            .iter()
            .position(|l| {
                let t: String = l.spans.iter().map(|s| s.content.as_ref()).collect();
                t.contains("H3")
            })
            .expect("H3 row not found");
        let h3_italic = lines[h3_row_idx]
            .spans
            .iter()
            .any(|s| s.style.add_modifier.contains(Modifier::ITALIC));
        assert!(h3_italic, "H3 must have ITALIC modifier");
    }

    /// FIX-C: two blockquote lines separated by a soft break must render as TWO
    /// separate `│ `-prefixed rows (not collapsed into one).
    #[test]
    fn blockquote_two_lines_render_as_two_rows() {
        let theme = Theme::default_theme();
        let src = "> line one\n> line two\n";
        let lines = render_markdown(src, &theme);
        let content_rows: Vec<String> = lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect::<String>())
            .filter(|t| t.contains("│ "))
            .collect();
        assert_eq!(
            content_rows.len(),
            2,
            "two >-lines must produce two │-prefixed rows, got: {:?}",
            content_rows
        );
        assert!(content_rows[0].contains("line one"), "first row: {:?}", content_rows[0]);
        assert!(content_rows[1].contains("line two"), "second row: {:?}", content_rows[1]);
    }

    /// FIX-E: block math `\,` (thin space) must NOT be corrupted by pulldown-cmark.
    /// The rendered output must contain `∫` (integral) and NO literal comma from `\,`.
    #[test]
    fn block_math_thin_space_not_corrupted() {
        let theme = Theme::default_theme();
        let src = "$$\\int_0^1 x\\,dx$$\n";
        let lines = render_markdown(src, &theme);
        let joined: String = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .map(|s| s.content.as_ref())
            .collect::<Vec<_>>()
            .join("\n");
        // The integral glyph must be present (math rendered at all).
        assert!(joined.contains('∫'), "integral glyph must be present: {:?}", joined);
        // `\,` must NOT produce a literal comma in the math body.
        // The rendered form is `∫ x dx` (space) or `∫ xdx` — not `∫ x,dx`.
        assert!(
            !joined.contains(",dx"),
            "\\, must not produce a literal comma before 'dx': {:?}",
            joined
        );
    }

    /// FIX-F: a nested ordered list must NOT insert a spurious blank row between
    /// the end of the nested list and the next sibling item in the parent list.
    #[test]
    fn nested_list_no_spurious_blank() {
        let theme = Theme::default_theme();
        let src = "1. First\n2. Second\n   1. Nested\n3. Third\n";
        let rows: Vec<String> = render_markdown(src, &theme)
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        // Find the "Nested" row.
        let nested_idx = rows
            .iter()
            .position(|r| r.contains("Nested"))
            .expect("Nested item not found in output");
        // The row immediately after "Nested" must not be blank.
        assert!(
            !rows[nested_idx + 1].is_empty(),
            "row after nested list item must not be blank, got: {:?}",
            &rows[nested_idx..]
        );
        // It must be item 3.
        assert!(
            rows[nested_idx + 1].contains("3."),
            "row after nested list must be item 3, got: {:?}",
            &rows[nested_idx + 1]
        );
    }
}
