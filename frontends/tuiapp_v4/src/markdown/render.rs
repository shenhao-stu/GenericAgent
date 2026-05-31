//! markdown/render.rs — `pulldown-cmark` events → themed `ratatui` Lines/Spans
//! (checklist §1 P3, §3 markdown/render). The terminal analogue of Claude Code's
//! `marked.lexer` + custom `formatToken` ANSI emitter (recon
//! `painpoints/markdown_math.md` §3): parse markdown to events, then WALK the
//! events emitting styled spans — never a live re-parsed widget (recon
//! RESEARCH_REPORT "Markdown → concrete styled lines").
//!
//! Coverage (the deliverable surface):
//!   * **Headings** — bold, a restrained per-level color ramp (theme tokens).
//!   * **Inline** — `**bold**`, `*italic*`, `` `code` `` (accent), `~~strike~~`,
//!     `[text](url)` links rendered `text (url)` in the suggestion color.
//!   * **Lists** — bullets `•`/`◦`/`▪` by depth + ordered numbers, hanging indent.
//!   * **Tables** — GFM, with per-column **display-width** alignment honoring the
//!     `:---`/`:--:`/`---:` column alignment spec.
//!   * **Blockquotes** — a dim `│ ` left gutter, nestable.
//!   * **Fenced code** — routed through [`super::highlight`] (syntect) with a
//!     language label + gutter; unknown langs degrade to dim plain text.
//!   * **Math** — `$…$` inline and `$$…$$` block routed through [`super::math`]
//!     (`latex_to_unicode`). pulldown-cmark does not tokenize `$`, so we split
//!     text runs on balanced `$`/`$$` ourselves (guarding `\$` and `$5`-style
//!     non-math), and detect a paragraph that is wholly `$$…$$` as block math.
//!
//! All colors come from [`Theme`] tokens (no hardcoded RGB at the markdown
//! layer). Output is `Vec<Line<'static>>`, already a list of styled rows; the
//! transcript widget renders them directly. The function is PURE over
//! `(source, theme)` and unit-tested; soft-wrapping to a width is the caller's
//! wrap cache (this layer emits logical, pre-styled lines).

use pulldown_cmark::{Alignment, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

use crate::theme::{Theme, Token};

use super::highlight;
use super::math::{self, Display};

/// Render markdown `source` into themed lines. PUBLIC entry point (P3).
///
/// The result is a flat list of styled rows in document order, with one blank
/// line between block elements (CC's `gap={1}` spacing). It never panics: the
/// math sub-renderer degrades to literal LaTeX, the code highlighter degrades to
/// plain text, and unknown constructs fall back to their text content.
pub fn render_markdown(source: &str, theme: &Theme) -> Vec<Line<'static>> {
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_TABLES);
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_TASKLISTS);
    let parser = Parser::new_ext(source, opts);

    let mut w = Walker::new(theme);
    for ev in parser {
        w.event(ev);
    }
    w.finish()
}

/// The walker state: a list of completed lines plus the in-progress line's spans,
/// an indent/list/quote context stack, and buffers for tables and code blocks.
struct Walker<'t> {
    theme: &'t Theme,
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
}

#[derive(Clone)]
struct ListState {
    /// `Some(next_number)` for ordered lists; `None` for bullets.
    ordered: Option<u64>,
}

struct CodeBuf {
    lang: String,
    text: String,
}

struct TableBuf {
    alignments: Vec<Alignment>,
    /// Header cells, then body rows; each cell is its plain display string.
    header: Vec<String>,
    rows: Vec<Vec<String>>,
    /// Accumulator for the cell currently being built.
    cur_cell: String,
    /// Accumulator for the row currently being built (body).
    cur_row: Vec<String>,
    /// True while reading the header row.
    in_header: bool,
}

impl<'t> Walker<'t> {
    fn new(theme: &'t Theme) -> Self {
        Walker {
            theme,
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
        // Split on hard newlines first so no emitted span can carry an embedded
        // `\n` (the two row-count projections would otherwise disagree). Each
        // newline ends the current logical line; the gutter re-prefixes via
        // `flush_line` and continuations stay inside a blockquote/list context.
        // Accumulate into `pending` so adjacent text fragments reassemble before
        // the inline-math split — pulldown splits a paragraph at `_`/`*` even when
        // they are NOT emphasis, which is THE bug: `$\sum_{i}$` arrived as three
        // Text events (`$\sum`, `_`, `{i}$`), so the `$` never paired. A hard `\n`
        // ends a logical line: flush the run through the math splitter, then the line.
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
                    let rendered = math::render_math(&latex, Display::Inline);
                    self.cur.push(Span::styled(
                        rendered,
                        self.col(Token::Claude),
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
        // Coalesce consecutive inline TEXT/HTML so a fragmented inline-math run
        // reassembles (see push_text). ANY other event first flushes the pending
        // run in the current style, so math never crosses a real style/structural
        // boundary (emphasis, inline code, a break, a block edge).
        if !matches!(ev, Event::Text(_) | Event::Html(_) | Event::InlineHtml(_)) {
            self.flush_pending();
        }
        match ev {
            Event::Start(tag) => self.start(tag),
            Event::End(tag) => self.end(tag),
            Event::Text(t) => self.push_text(&t),
            Event::Code(c) => {
                // Inline code span: accent color, faux-padded with the content.
                if let Some(tb) = self.table.as_mut() {
                    tb.cur_cell.push_str(&c);
                } else {
                    self.cur
                        .push(Span::styled(c.to_string(), self.col(Token::Claude)));
                }
            }
            Event::SoftBreak => {
                // Treat as a space within a paragraph (markdown soft-wrap); the
                // caller's wrap cache handles visual wrapping.
                if self.code.is_none() && self.table.is_none() {
                    self.cur.push(Span::raw(" "));
                }
            }
            Event::HardBreak => {
                if self.code.is_none() && self.table.is_none() {
                    self.flush_line();
                    self.indent_continuation();
                }
            }
            Event::Rule => {
                self.block_gap();
                // A horizontal rule → a dim full-ish line marker (the caller may
                // extend it; we emit a short rule that reads as a divider).
                self.out.push(Line::from(Span::styled(
                    "────────".to_string(),
                    self.col(Token::Border),
                )));
                self.started = true;
            }
            Event::TaskListMarker(done) => {
                let glyph = if done { "☑ " } else { "☐ " };
                self.cur
                    .push(Span::styled(glyph.to_string(), self.col(Token::Success)));
            }
            // HTML / footnotes / inline-math events: render their text literally.
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
            }
            Tag::Heading { level, .. } => {
                self.block_gap();
                let (tok, prefix) = heading_style(level);
                self.cur.push(Span::styled(
                    prefix.to_string(),
                    Style::default()
                        .fg(self.theme.color(tok))
                        .add_modifier(Modifier::BOLD),
                ));
                self.style_stack.push(
                    Style::default()
                        .fg(self.theme.color(tok))
                        .add_modifier(Modifier::BOLD),
                );
            }
            Tag::BlockQuote(_) => {
                self.block_gap();
                self.quote_depth += 1;
            }
            Tag::CodeBlock(kind) => {
                self.block_gap();
                let lang = match kind {
                    pulldown_cmark::CodeBlockKind::Fenced(info) => {
                        // The info string's first token is the language.
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
                        format!("{} ", bullet(depth))
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
                // No inline images in a TUI; show the alt text + a marker.
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
                self.flush_line();
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
                    self.emit_code_block(&code);
                }
                self.started = true;
            }
            TagEnd::List(_) => {
                self.lists.pop();
                self.started = true;
            }
            TagEnd::Item => {
                self.flush_line();
            }
            TagEnd::Emphasis
            | TagEnd::Strong
            | TagEnd::Strikethrough => {
                self.style_stack.pop();
            }
            TagEnd::Link => {
                self.style_stack.pop();
                if let Some(url) = self.link_url.take() {
                    // Render the destination after the link text: `text (url)`.
                    self.cur.push(Span::styled(
                        format!(" ({url})"),
                        self.col(Token::Dim),
                    ));
                }
            }
            TagEnd::Table => {
                if let Some(tb) = self.table.take() {
                    self.emit_table(&tb);
                }
                self.started = true;
            }
            TagEnd::TableHead => {
                if let Some(tb) = self.table.as_mut() {
                    // Move the final header cell + finish the header row.
                    let cell = std::mem::take(&mut tb.cur_cell);
                    tb.cur_row.push(cell);
                    tb.header = std::mem::take(&mut tb.cur_row);
                    tb.in_header = false;
                }
            }
            TagEnd::TableRow => {
                if let Some(tb) = self.table.as_mut() {
                    if !tb.in_header {
                        let cell = std::mem::take(&mut tb.cur_cell);
                        tb.cur_row.push(cell);
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

    /// Emit a fenced code block: a dim language label, then each highlighted line
    /// behind a quiet `│ ` gutter (theme Border color).
    fn emit_code_block(&mut self, code: &CodeBuf) {
        let fallback = self.col(Token::Dim);
        let lines = highlight::highlight(&code.text, &code.lang, fallback);
        // Language label row (if a language was declared).
        if !code.lang.is_empty() {
            self.out.push(Line::from(Span::styled(
                format!("  {} ", code.lang),
                self.col(Token::Dim),
            )));
        }
        let gutter_style = self.col(Token::Border);
        for hl_line in lines {
            let mut spans: Vec<Span<'static>> = Vec::with_capacity(hl_line.len() + 1);
            spans.push(Span::styled("│ ".to_string(), gutter_style));
            if hl_line.is_empty() {
                spans.push(Span::raw(""));
            } else {
                for seg in hl_line {
                    spans.push(Span::styled(seg.text, seg.style));
                }
            }
            self.out.push(Line::from(spans));
        }
    }

    /// Emit a GFM table with per-column display-width alignment honoring the
    /// `:---`/`:--:`/`---:` alignment spec. Header row, a `─┼─` rule, body rows.
    fn emit_table(&mut self, tb: &TableBuf) {
        let ncols = tb
            .header
            .len()
            .max(tb.rows.iter().map(|r| r.len()).max().unwrap_or(0));
        if ncols == 0 {
            return;
        }
        // Per-column display width = max over header + all body cells.
        let mut widths = vec![0usize; ncols];
        let consider = |row: &[String], widths: &mut Vec<usize>| {
            for (i, c) in row.iter().enumerate() {
                if i < ncols {
                    widths[i] = widths[i].max(UnicodeWidthStr::width(c.as_str()));
                }
            }
        };
        consider(&tb.header, &mut widths);
        for r in &tb.rows {
            consider(r, &mut widths);
        }

        let align = |i: usize| -> Alignment {
            tb.alignments
                .get(i)
                .copied()
                .unwrap_or(Alignment::None)
        };

        // Header (bold).
        let header_style = self.col(Token::Text).add_modifier(Modifier::BOLD);
        self.out.push(self.table_row(
            &tb.header,
            &widths,
            ncols,
            &align,
            header_style,
        ));

        // Separator rule: `─` per column joined by `┼`, alignment colons echoed.
        let mut sep_spans: Vec<Span<'static>> = Vec::new();
        sep_spans.push(Span::styled("├─".to_string(), self.col(Token::Border)));
        for i in 0..ncols {
            if i > 0 {
                sep_spans.push(Span::styled("─┼─".to_string(), self.col(Token::Border)));
            }
            sep_spans.push(Span::styled("─".repeat(widths[i]), self.col(Token::Border)));
        }
        sep_spans.push(Span::styled("─┤".to_string(), self.col(Token::Border)));
        self.out.push(Line::from(sep_spans));

        // Body rows.
        let body_style = self.col(Token::Text);
        for r in &tb.rows {
            self.out
                .push(self.table_row(r, &widths, ncols, &align, body_style));
        }
    }

    /// Build one rendered table row: `│ ` borders, each cell aligned in its
    /// column's display width per the column alignment.
    fn table_row(
        &self,
        cells: &[String],
        widths: &[usize],
        ncols: usize,
        align: &dyn Fn(usize) -> Alignment,
        cell_style: Style,
    ) -> Line<'static> {
        let mut spans: Vec<Span<'static>> = Vec::new();
        spans.push(Span::styled("│ ".to_string(), self.col(Token::Border)));
        for i in 0..ncols {
            if i > 0 {
                spans.push(Span::styled(" │ ".to_string(), self.col(Token::Border)));
            }
            let empty = String::new();
            let cell = cells.get(i).unwrap_or(&empty);
            let padded = pad_cell(cell, widths[i], align(i));
            spans.push(Span::styled(padded, cell_style));
        }
        spans.push(Span::styled(" │".to_string(), self.col(Token::Border)));
        Line::from(spans)
    }

    /// Flush any dangling content and return the accumulated lines.
    fn finish(mut self) -> Vec<Line<'static>> {
        self.flush_pending(); // emit any trailing inline run (+ its math).
        if !self.cur.is_empty() {
            self.flush_line();
        }
        // Trim a single trailing blank line for tidiness.
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

/// Pad `cell` to `width` display cells per `align` (default = left). Uses display
/// width, never `.len()`, so CJK/emoji columns line up.
fn pad_cell(cell: &str, width: usize, align: Alignment) -> String {
    let cw = UnicodeWidthStr::width(cell);
    if cw >= width {
        return cell.to_string();
    }
    let pad = width - cw;
    match align {
        Alignment::Right => format!("{}{}", " ".repeat(pad), cell),
        Alignment::Center => {
            let l = pad / 2;
            let r = pad - l;
            format!("{}{}{}", " ".repeat(l), cell, " ".repeat(r))
        }
        // None / Left
        _ => format!("{}{}", cell, " ".repeat(pad)),
    }
}

/// The (color token, leading glyph) for a heading level. A restrained ramp — no
/// ASCII banners, matching CC's understated style.
fn heading_style(level: HeadingLevel) -> (Token, &'static str) {
    match level {
        HeadingLevel::H1 => (Token::Claude, "# "),
        HeadingLevel::H2 => (Token::Suggestion, "## "),
        HeadingLevel::H3 => (Token::Success, "### "),
        HeadingLevel::H4 => (Token::Warning, "#### "),
        HeadingLevel::H5 => (Token::PlanMode, "##### "),
        HeadingLevel::H6 => (Token::Dim, "###### "),
    }
}

/// The bullet glyph for a list nesting depth (cycles `•`/`◦`/`▪`).
fn bullet(depth: usize) -> &'static str {
    match depth % 3 {
        0 => "•",
        1 => "◦",
        _ => "▪",
    }
}

/// One segment of a text run split on math delimiters.
enum Seg {
    Text(String),
    Math(String),
}

/// Split a text run into literal text and inline-math (`$…$`) segments. Guards:
///   * `\$` is an escaped literal dollar (not a delimiter).
///   * a `$` immediately followed/preceded by a digit with no closing `$` (i.e.
///     `$5`) is NOT treated as math — we only emit a Math segment for a balanced
///     pair whose interior is non-empty and does not start/end with a space
///     (the KaTeX/markdown-it-texmath currency heuristic).
///   * `$$…$$` inside a text run collapses to inline (block `$$` is handled at the
///     paragraph level before this is reached).
fn split_inline_math(text: &str) -> Vec<Seg> {
    if !text.contains('$') {
        return vec![Seg::Text(text.to_string())];
    }
    let chars: Vec<char> = text.chars().collect();
    let mut out: Vec<Seg> = Vec::new();
    let mut buf = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '\\' && i + 1 < chars.len() && chars[i + 1] == '$' {
            // Escaped dollar → literal `$`.
            buf.push('$');
            i += 2;
            continue;
        }
        if c == '$' {
            // Determine delimiter length (`$` or `$$`).
            let double = i + 1 < chars.len() && chars[i + 1] == '$';
            let delim_len = if double { 2 } else { 1 };
            let body_start = i + delim_len;
            // Find the matching closing delimiter.
            if let Some(close) = find_math_close(&chars, body_start, double) {
                let body: String = chars[body_start..close].iter().collect();
                // Currency / non-math guard: require a non-space-bounded, non-empty
                // body (so `$5 and $10` with a space-bounded interior is skipped).
                if is_probably_math(&body) {
                    if !buf.is_empty() {
                        out.push(Seg::Text(std::mem::take(&mut buf)));
                    }
                    out.push(Seg::Math(body));
                    i = close + delim_len;
                    continue;
                }
            }
            // Not math → keep the `$` literal.
            buf.push('$');
            i += 1;
            continue;
        }
        buf.push(c);
        i += 1;
    }
    if !buf.is_empty() {
        out.push(Seg::Text(buf));
    }
    if out.is_empty() {
        out.push(Seg::Text(String::new()));
    }
    out
}

/// Find the closing `$`/`$$` for a math span starting at `from`. Honors `\$`
/// escapes inside. Returns the index of the FIRST char of the closing delimiter.
fn find_math_close(chars: &[char], from: usize, double: bool) -> Option<usize> {
    let mut i = from;
    while i < chars.len() {
        if chars[i] == '\\' && i + 1 < chars.len() {
            i += 2;
            continue;
        }
        if chars[i] == '$' {
            if double {
                if i + 1 < chars.len() && chars[i + 1] == '$' {
                    return Some(i);
                }
                // a single `$` inside a `$$…$$` is allowed; keep scanning.
                i += 1;
                continue;
            }
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Heuristic: is this `$…$` interior actually math (not `$5` currency / prose)?
/// Require a non-empty body that does NOT begin or end with a space (the standard
/// texmath rule), and contains at least one math-ish char (a letter, backslash,
/// digit, or operator) so a lone `$ $` isn't captured.
fn is_probably_math(body: &str) -> bool {
    if body.is_empty() {
        return false;
    }
    if body.starts_with(' ') || body.ends_with(' ') {
        return false;
    }
    body.chars()
        .any(|c| c.is_alphanumeric() || c == '\\' || "+-*/=^_(){}".contains(c))
}

/// Whether an entire paragraph's text is a single `$$…$$` block (so it should be
/// rendered as block math rather than inline). Returns the inner LaTeX if so.
/// Used by the transcript routing layer before walking, so block math gets the
/// multi-line stacked treatment.
#[allow(dead_code)] // consumed by the routing helper in `super`.
pub fn extract_block_math(source: &str) -> Option<String> {
    let t = source.trim();
    if t.len() >= 4 && t.starts_with("$$") && t.ends_with("$$") {
        let inner = &t[2..t.len() - 2];
        if !inner.trim().is_empty() {
            return Some(inner.trim().to_string());
        }
    }
    None
}

/// Render a `$$…$$` block as multi-line stacked math lines (P3). Used by the
/// transcript routing layer for a standalone display-math paragraph.
#[allow(dead_code)] // consumed by the routing helper in `super`.
pub fn render_block_math(latex: &str, theme: &Theme) -> Vec<Line<'static>> {
    let r = math::latex_to_unicode(latex, Display::Block);
    r.lines
        .into_iter()
        .map(|l| Line::from(Span::styled(l, Style::default().fg(theme.color(Token::Claude)))))
        .collect()
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

    // ---- md_table (the named test) ----------------------------------------
    #[test]
    fn md_table() {
        let theme = Theme::ga_default();
        let src = "\
| Name | Score |
|:-----|------:|
| Alice | 90 |
| Bob | 100 |
";
        let lines = plain(&theme, src);
        // Header + separator + 2 body rows = 4 content rows (no extra noise).
        assert_eq!(lines.len(), 4, "table = header + rule + 2 rows; got {lines:?}");

        // Header carries both column titles.
        assert!(lines[0].contains("Name"));
        assert!(lines[0].contains("Score"));
        // The rule row is box-drawing.
        assert!(lines[1].contains('┼') || lines[1].contains('─'));

        // Right-aligned "Score" column: "90" is padded on the LEFT to width 5
        // (max of "Score"=5, "100"=3), so the body cell shows "  90".
        let alice = &lines[2];
        assert!(alice.contains("Alice"));
        assert!(alice.contains("  90"), "right-aligned score: {alice:?}");
        let bob = &lines[3];
        assert!(bob.contains("100"));

        // Every rendered row has the SAME display width (columns line up). The
        // table is the headline CJK/alignment correctness check.
        let w0 = UnicodeWidthStr::width(lines[0].as_str());
        for l in &lines {
            assert_eq!(
                UnicodeWidthStr::width(l.as_str()),
                w0,
                "all table rows must share a width: {l:?}"
            );
        }
    }

    #[test]
    fn md_table_cjk_alignment() {
        let theme = Theme::ga_default();
        // CJK cells are 2 cells/char — alignment must use display width.
        let src = "\
| 名字 | 分数 |
|------|------|
| 张三 | 90 |
";
        let lines = plain(&theme, src);
        let w0 = UnicodeWidthStr::width(lines[0].as_str());
        for l in &lines {
            assert_eq!(UnicodeWidthStr::width(l.as_str()), w0);
        }
    }

    #[test]
    fn md_headings_and_inline_styles() {
        let theme = Theme::ga_default();
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
        let theme = Theme::ga_default();
        let lines = plain(&theme, "- first\n- second\n\n1. one\n2. two");
        let joined = lines.join("\n");
        assert!(joined.contains("• first"));
        assert!(joined.contains("• second"));
        assert!(joined.contains("1. one"));
        assert!(joined.contains("2. two"));
    }

    #[test]
    fn md_blockquote_gutter() {
        let theme = Theme::ga_default();
        let lines = plain(&theme, "> quoted text");
        let joined = lines.join("\n");
        assert!(joined.contains("│ "), "blockquote shows a gutter: {joined:?}");
        assert!(joined.contains("quoted text"));
    }

    #[test]
    fn md_link_renders_text_and_url() {
        let theme = Theme::ga_default();
        let lines = plain(&theme, "see [the docs](https://example.com) here");
        let joined = lines.join("\n");
        assert!(joined.contains("the docs"));
        assert!(joined.contains("(https://example.com)"));
    }

    #[test]
    fn md_fenced_code_is_highlighted_and_framed() {
        let theme = Theme::ga_default();
        let lines = render_markdown("```rust\nfn main() {}\n```", &theme);
        let joined: String = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .map(|s| s.content.as_ref())
            .collect();
        // Language label + gutter + the source text are all present.
        assert!(joined.contains("rust"));
        assert!(joined.contains("│ "));
        assert!(joined.contains("fn main"));
    }

    #[test]
    fn md_inline_math_is_rendered() {
        let theme = Theme::ga_default();
        let lines = plain(&theme, "the angle $\\alpha$ is small");
        let joined = lines.join("\n");
        assert!(joined.contains('α'), "inline math \\alpha → α: {joined:?}");
        // The dollar delimiters are consumed.
        assert!(!joined.contains('$'));
    }

    /// REGRESSION (the BUG the latex recon found): pulldown-cmark fragments a
    /// paragraph's text at `_`/`*` (even non-emphasis ones), so inline math whose
    /// body contains a subscript / sum / integral / limit / `*` used to arrive as
    /// several Text events and the `$` leaked literally. The `pending`-coalescing
    /// flush reassembles the run before the math split, so these now render.
    #[test]
    fn md_inline_math_with_underscore_or_star_survives_pulldown_split() {
        let theme = Theme::ga_default();
        for src in [
            "Sum $\\sum_{i=1}^{n} i$ done",
            "A limit $\\lim_{x \\to 0} f$ here",
            "subscript $x_1 + x_2$ ok",
            "product $a*b$ times",
            "Closed form $\\frac{n(n+1)}{2}$ here",
        ] {
            let joined = plain(&theme, src).join("\n");
            assert!(
                !joined.contains('$'),
                "literal $ leaked (math not reassembled) for {src:?}: {joined:?}"
            );
            // The `\sum`/`\lim`/`\frac` macros are converted, not left raw.
            assert!(
                !joined.contains("\\sum") && !joined.contains("\\lim") && !joined.contains("\\frac"),
                "raw LaTeX macro leaked for {src:?}: {joined:?}"
            );
        }
        // The sum glyph + subscript actually render.
        let sum = plain(&theme, "Sum $\\sum_{i=1}^{n} i$ done").join("\n");
        assert!(sum.contains('∑'), "the sum operator renders: {sum:?}");
        // A real emphasis run is still independent (math doesn't swallow it): the
        // emphasized word renders as its text, no stray `$`.
        let emph = plain(&theme, "say *hello* and $x_1$").join("\n");
        assert!(emph.contains("hello") && !emph.contains('$'), "emphasis + math coexist: {emph:?}");
    }

    #[test]
    fn md_currency_dollar_is_not_math() {
        let theme = Theme::ga_default();
        // "$5 and $10" has space-bounded interiors → NOT captured as math.
        let lines = plain(&theme, "it costs $5 and $10 total");
        let joined = lines.join("\n");
        assert!(joined.contains("$5"));
        assert!(joined.contains("$10"));
    }

    #[test]
    fn block_math_extraction_and_render() {
        let theme = Theme::ga_default();
        // A standalone $$…$$ paragraph is detected + stacked (3-line frac).
        let inner = extract_block_math("$$\\frac{a}{b}$$").unwrap();
        assert_eq!(inner, "\\frac{a}{b}");
        let lines = render_block_math(&inner, &theme);
        assert_eq!(lines.len(), 3, "block frac stacks to 3 lines");
        // Not block math: inline `$x$` returns None.
        assert!(extract_block_math("text $x$ more").is_none());
    }

    #[test]
    fn md_never_panics_on_weird_input() {
        let theme = Theme::ga_default();
        // Unclosed fence, half a table, dangling emphasis, stray math — all must
        // render *something* without panicking.
        let _ = render_markdown("```\nunclosed", &theme);
        let _ = render_markdown("| a | b\n| - |", &theme);
        let _ = render_markdown("**bold without close", &theme);
        let _ = render_markdown("math $\\frac{1}{ unclosed", &theme);
        let _ = render_markdown("", &theme);
        let _ = render_markdown("你好 **世界** `代码`", &theme);
    }
}
