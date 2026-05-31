//! markdown/highlight.rs — fenced code-block syntax highlighting via `syntect`
//! (checklist §1 P3, §3 markdown/highlight). The terminal analogue of Claude
//! Code's `cli-highlight` (recon `painpoints/markdown_math.md` §4): map a code
//! block + its declared language to styled ratatui spans.
//!
//! Design (the recon's two idioms, ported to Rust):
//!   * **Lazy-load the syntax + theme sets ONCE** behind a `OnceLock`. Loading
//!     syntect's default `SyntaxSet`/`ThemeSet` is the ~tens-of-ms cost; doing it
//!     once and sharing it keeps per-block highlighting cheap (the first code
//!     block pays, the rest are free). syntect is built with the **fancy-regex**
//!     engine (pure Rust, no oniguruma C dep — which breaks on Windows).
//!   * **Always honor the fence's declared language** (`highlight(code, "rust")`)
//!     rather than auto-detecting, which misfires. Unknown / absent language →
//!     fall back to plain (dim) text. NEVER panics, never crashes the stream.
//!
//! Output is `Vec<Vec<(Style, String)>>` — one inner `Vec` per source line, each
//! a run of `(style, text)` segments — so the markdown emitter can frame the
//! block (language label, gutter) and the caller stays free of syntect types.

use std::sync::OnceLock;

use ratatui::style::{Color, Style};
use syntect::easy::HighlightLines;
use syntect::highlighting::{FontStyle, Style as SynStyle, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::LinesWithEndings;

/// One highlighted segment: the style to paint it with and the (newline-free)
/// text. A line is a `Vec<HlSpan>`; a block is a `Vec<Vec<HlSpan>>`.
#[derive(Debug, Clone, PartialEq)]
pub struct HlSpan {
    pub style: Style,
    pub text: String,
}

/// The shared, lazily-initialized syntect state. Loading the default sets is the
/// expensive step; we do it exactly once for the whole process.
struct HlState {
    syntaxes: SyntaxSet,
    themes: ThemeSet,
}

/// Global lazy handle. `OnceLock` (std) means the heavy `load_defaults` runs on
/// the first code block only; subsequent blocks reuse it (the "lazy-load
/// syntaxes" requirement).
static HL: OnceLock<HlState> = OnceLock::new();

fn state() -> &'static HlState {
    HL.get_or_init(|| HlState {
        syntaxes: SyntaxSet::load_defaults_newlines(),
        themes: ThemeSet::load_defaults(),
    })
}

/// The theme name we highlight against. `base16-ocean.dark` ships with syntect's
/// defaults and reads well on a dark terminal. (Color tokens for the *frame*
/// come from our theme; the code body uses syntect's per-scope colors, which is
/// the point of real highlighting.)
const THEME: &str = "base16-ocean.dark";

/// Highlight `code` as `lang`, returning per-line styled spans. PUBLIC entry.
///
/// `lang` is the fence info string (e.g. `"rust"`, `"py"`, `"```json"`'s
/// `"json"`). It is matched by token/extension first, then by name; an unknown or
/// empty language falls back to plain dim text. Highlighting failures degrade to
/// the same plain fallback — this function is total and never panics.
///
/// A trailing newline in `code` does not produce a phantom empty final line.
pub fn highlight(code: &str, lang: &str, fallback: Style) -> Vec<Vec<HlSpan>> {
    let st = state();
    let lang = lang.trim();

    // Resolve the syntax for the declared language. Try the token (extension /
    // short name like "rs"/"py"), then the full name ("Rust"), case-insensitively.
    let syntax = if lang.is_empty() {
        None
    } else {
        st.syntaxes
            .find_syntax_by_token(lang)
            .or_else(|| st.syntaxes.find_syntax_by_name(lang))
            .or_else(|| {
                // Try a few common aliases syntect's token match misses.
                let alias = match lang.to_ascii_lowercase().as_str() {
                    "ts" | "typescript" => "TypeScript",
                    "js" | "javascript" | "node" => "JavaScript",
                    "py" | "python3" => "Python",
                    "sh" | "shell" | "zsh" | "console" => "Bourne Again Shell (bash)",
                    "rs" => "Rust",
                    "yml" => "YAML",
                    "md" | "markdown" => "Markdown",
                    "c++" | "cpp" => "C++",
                    "rb" => "Ruby",
                    "" => "",
                    _ => "",
                };
                if alias.is_empty() {
                    None
                } else {
                    st.syntaxes.find_syntax_by_name(alias)
                }
            })
    };

    let Some(syntax) = syntax else {
        // No language / unknown language → plain dim text (faithful fallback).
        return plain(code, fallback);
    };

    let theme = match st.themes.themes.get(THEME) {
        Some(t) => t,
        None => return plain(code, fallback),
    };
    let mut hl = HighlightLines::new(syntax, theme);

    let mut out: Vec<Vec<HlSpan>> = Vec::new();
    for line in LinesWithEndings::from(code) {
        match hl.highlight_line(line, &st.syntaxes) {
            Ok(ranges) => {
                let mut spans: Vec<HlSpan> = Vec::new();
                for (syn_style, text) in ranges {
                    // Drop the trailing newline; ratatui rows carry no `\n`.
                    let text = text.trim_end_matches('\n');
                    if text.is_empty() {
                        continue;
                    }
                    spans.push(HlSpan {
                        style: convert_style(syn_style),
                        text: text.to_string(),
                    });
                }
                out.push(spans);
            }
            // A highlighting error on one line → emit it plain rather than abort.
            Err(_) => {
                let t = line.trim_end_matches('\n');
                out.push(vec![HlSpan {
                    style: fallback,
                    text: t.to_string(),
                }]);
            }
        }
    }
    if out.is_empty() {
        out.push(Vec::new());
    }
    out
}

/// Plain (un-highlighted) fallback: each source line becomes a single span in the
/// `fallback` style. Used for unknown languages and on any highlighter error.
fn plain(code: &str, fallback: Style) -> Vec<Vec<HlSpan>> {
    let mut out: Vec<Vec<HlSpan>> = code
        .split('\n')
        .map(|l| {
            let l = l.trim_end_matches('\r');
            if l.is_empty() {
                Vec::new()
            } else {
                vec![HlSpan {
                    style: fallback,
                    text: l.to_string(),
                }]
            }
        })
        .collect();
    // A trailing newline yields a final empty segment from `split`; drop it so a
    // block doesn't gain a phantom blank line (mirrors Block::hard_lines).
    if code.ends_with('\n') {
        out.pop();
    }
    if out.is_empty() {
        out.push(Vec::new());
    }
    out
}

/// Convert a syntect [`SynStyle`] (24-bit fg + font flags) to a ratatui [`Style`].
/// We take only the foreground + bold/italic/underline; the background stays the
/// terminal default (we frame code with our own theme gutter instead of a block
/// background, which reads cleaner in a TUI and avoids fighting the user's bg).
fn convert_style(s: SynStyle) -> Style {
    let mut style = Style::default().fg(Color::Rgb(
        s.foreground.r,
        s.foreground.g,
        s.foreground.b,
    ));
    if s.font_style.contains(FontStyle::BOLD) {
        style = style.add_modifier(ratatui::style::Modifier::BOLD);
    }
    if s.font_style.contains(FontStyle::ITALIC) {
        style = style.add_modifier(ratatui::style::Modifier::ITALIC);
    }
    if s.font_style.contains(FontStyle::UNDERLINE) {
        style = style.add_modifier(ratatui::style::Modifier::UNDERLINED);
    }
    style
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- md_code_highlight_smoke (the named test) -------------------------
    #[test]
    fn md_code_highlight_smoke() {
        let fb = Style::default();
        // A real language resolves and produces styled spans across lines.
        let code = "fn main() {\n    let x = 1;\n}\n";
        let lines = highlight(code, "rust", fb);
        assert_eq!(lines.len(), 3, "three source lines, no phantom trailing line");
        // Every non-empty line reconstructs to its source text (no chars lost).
        let joined: String = lines
            .iter()
            .map(|line| line.iter().map(|s| s.text.as_str()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(joined, "fn main() {\n    let x = 1;\n}");
        // At least one segment got a non-default (highlighted) foreground —
        // i.e. highlighting actually ran, it's not all-plain.
        let any_colored = lines
            .iter()
            .flatten()
            .any(|s| s.style.fg.is_some() && s.style.fg != fb.fg);
        assert!(any_colored, "rust code should produce colored spans");
    }

    #[test]
    fn unknown_language_falls_back_to_plain() {
        let fb = Style::default().fg(Color::Rgb(0x80, 0x80, 0x80));
        let code = "some plain text\nsecond line";
        let lines = highlight(code, "definitely-not-a-language", fb);
        assert_eq!(lines.len(), 2);
        // Each line is one span in the fallback style — text preserved verbatim.
        assert_eq!(lines[0].len(), 1);
        assert_eq!(lines[0][0].text, "some plain text");
        assert_eq!(lines[0][0].style.fg, fb.fg);
        assert_eq!(lines[1][0].text, "second line");
    }

    #[test]
    fn empty_language_is_plain_not_panic() {
        let fb = Style::default();
        let lines = highlight("x = 1\ny = 2", "", fb);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0][0].text, "x = 1");
    }

    #[test]
    fn empty_code_yields_one_empty_line() {
        let fb = Style::default();
        let lines = highlight("", "rust", fb);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].is_empty());
    }

    #[test]
    fn lazy_state_initializes_once() {
        // Two calls share the same lazily-initialized state pointer.
        let a = state() as *const HlState;
        let b = state() as *const HlState;
        assert_eq!(a, b, "syntect state must be loaded once and shared");
    }
}
