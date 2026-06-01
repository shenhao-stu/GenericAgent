//! markdown/code.rs — fenced-code buffering + emission, plus the heading-ramp and
//! list-bullet glyph tables. [`emit_code_block`] routes the collected source
//! through [`super::highlight`] (syntect) behind a dim language label + a quiet
//! `│ ` gutter; unknown languages degrade to dim plain text.

use pulldown_cmark::HeadingLevel;
use ratatui::style::Style;
use ratatui::text::{Line, Span};

use crate::theme::{Theme, Token};

use super::highlight;

pub(crate) struct CodeBuf {
    pub(crate) lang: String,
    pub(crate) text: String,
}

fn col(theme: &Theme, tok: Token) -> Style {
    Style::default().fg(theme.color(tok))
}

/// Emit a fenced code block: a dim language label, then each highlighted line
/// behind a quiet `│ ` gutter (theme Border color).
pub(crate) fn emit_code_block(theme: &Theme, code: &CodeBuf) -> Vec<Line<'static>> {
    let fallback = col(theme, Token::Dim);
    let lines = highlight::highlight(&code.text, &code.lang, fallback);
    let mut out: Vec<Line<'static>> = Vec::new();
    if !code.lang.is_empty() {
        out.push(Line::from(Span::styled(
            format!("  {} ", code.lang),
            col(theme, Token::Dim),
        )));
    }
    let gutter_style = col(theme, Token::Border);
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
        out.push(Line::from(spans));
    }
    out
}

/// The color token for a heading level — a restrained level cue (accent for H1,
/// dimming toward H6), like tui_v3/CC. No literal `#` glyph: the heading renders
/// as clean BOLD + colored text.
pub(crate) fn heading_style(level: HeadingLevel) -> Token {
    match level {
        HeadingLevel::H1 => Token::Claude,
        HeadingLevel::H2 => Token::Suggestion,
        HeadingLevel::H3 => Token::Success,
        HeadingLevel::H4 => Token::Warning,
        HeadingLevel::H5 => Token::PlanMode,
        HeadingLevel::H6 => Token::Dim,
    }
}

/// The bullet glyph for a list nesting depth (cycles `•`/`◦`/`▪`).
pub(crate) fn bullet(depth: usize) -> &'static str {
    match depth % 3 {
        0 => "•",
        1 => "◦",
        _ => "▪",
    }
}

#[cfg(test)]
mod tests {
    use super::super::render::render_markdown;
    use crate::theme::Theme;

    #[test]
    fn md_fenced_code_is_highlighted_and_framed() {
        let theme = Theme::default_theme();
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
}
