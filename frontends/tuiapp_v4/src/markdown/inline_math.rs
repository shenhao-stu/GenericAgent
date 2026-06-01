//! markdown/inline_math.rs — split a text run on balanced `$…$` / `$$…$$` and
//! detect a whole-paragraph `$$…$$` block. pulldown-cmark does not tokenize `$`,
//! so the [`Walker`](super::render) hands its reassembled inline runs here.
//!
//! Math segments route through [`super::math`] (`latex_to_unicode`); the splitter
//! itself is pure `&str → Vec<Seg>` with zero render coupling.

use ratatui::style::Style;
use ratatui::text::{Line, Span};

use crate::theme::{Theme, Token};

use super::math::{self, Display};

/// One segment of a text run split on math delimiters.
pub(crate) enum Seg {
    Text(String),
    Math(String),
}

/// Split a text run into literal text and inline-math (`$…$`) segments. Guards:
///   * `\$` is an escaped literal dollar (not a delimiter).
///   * a `$5`-style run with no closing `$` is NOT math — only a balanced pair
///     whose interior is non-empty and not space-bounded becomes a Math segment
///     (the KaTeX/markdown-it-texmath currency heuristic).
///   * `$$…$$` inside a text run collapses to inline (block `$$` is handled at the
///     paragraph level before this is reached).
pub(crate) fn split_inline_math(text: &str) -> Vec<Seg> {
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
            buf.push('$');
            i += 2;
            continue;
        }
        if c == '$' {
            let double = i + 1 < chars.len() && chars[i + 1] == '$';
            let delim_len = if double { 2 } else { 1 };
            let body_start = i + delim_len;
            if let Some(close) = find_math_close(&chars, body_start, double) {
                let body: String = chars[body_start..close].iter().collect();
                if is_probably_math(&body) {
                    if !buf.is_empty() {
                        out.push(Seg::Text(std::mem::take(&mut buf)));
                    }
                    out.push(Seg::Math(body));
                    i = close + delim_len;
                    continue;
                }
            }
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
                // A single `$` inside `$$…$$` is allowed; keep scanning.
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
/// Require a non-empty, non-space-bounded body (the texmath rule) with at least
/// one math-ish char so a lone `$ $` isn't captured.
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

/// Whether an entire paragraph is a single `$$…$$` block (→ stacked block math).
/// Returns the inner LaTeX if so. Used by the transcript routing layer before
/// walking, so block math gets the multi-line stacked treatment.
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

/// Render a `$$…$$` block as multi-line stacked math lines (P3).
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
    use super::super::render::render_markdown;

    fn plain(theme: &Theme, src: &str) -> Vec<String> {
        render_markdown(src, theme)
            .into_iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect::<String>())
            .collect()
    }

    #[test]
    fn block_math_extraction_and_render() {
        let theme = Theme::default_theme();
        // A standalone $$…$$ paragraph is detected + stacked (3-line frac).
        let inner = extract_block_math("$$\\frac{a}{b}$$").unwrap();
        assert_eq!(inner, "\\frac{a}{b}");
        let lines = render_block_math(&inner, &theme);
        assert_eq!(lines.len(), 3, "block frac stacks to 3 lines");
        // Not block math: inline `$x$` returns None.
        assert!(extract_block_math("text $x$ more").is_none());
    }

    #[test]
    fn md_inline_math_is_rendered() {
        let theme = Theme::default_theme();
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
        let theme = Theme::default_theme();
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
        // A real emphasis run is still independent (math doesn't swallow it).
        let emph = plain(&theme, "say *hello* and $x_1$").join("\n");
        assert!(emph.contains("hello") && !emph.contains('$'), "emphasis + math coexist: {emph:?}");
    }

    #[test]
    fn md_currency_dollar_is_not_math() {
        let theme = Theme::default_theme();
        // "$5 and $10" has space-bounded interiors → NOT captured as math.
        let lines = plain(&theme, "it costs $5 and $10 total");
        let joined = lines.join("\n");
        assert!(joined.contains("$5"));
        assert!(joined.contains("$10"));
    }
}
