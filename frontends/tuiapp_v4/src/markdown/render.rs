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
                // A soft break is a space; the wrap cache handles visual wrapping.
                if self.code.is_none() && self.table.is_none() {
                    self.cur.push(Span::raw(" "));
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
                self.block_gap();
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
                self.block_gap();
                // Render the heading as clean BOLD + colored text (no literal `#`
                // glyph) — the per-level color is the restrained level cue.
                let tok = code::heading_style(level);
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
                if let Some(tb) = self.table.as_mut() {
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
}
