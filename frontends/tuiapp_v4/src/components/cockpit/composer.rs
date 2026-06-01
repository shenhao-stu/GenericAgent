//! cockpit/composer.rs — the bordered multi-line input widget: its row builder
//! (inverse-cell cursor, shell `!` peel) and the visual-column splitter.

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph};
use ratatui::Frame;
use unicode_width::UnicodeWidthStr;

use crate::app::AppState;
use crate::effects::shimmer::{blend, intensity_at};
use crate::theme::{lighten, FxCommand, Theme, Token};

/// COMPOSER: the bordered multi-line input with an inverse-cell cursor. The
/// border tints HOT-PINK (AutoAccept coral token) in shell mode (`!cmd`).
pub(crate) fn render_composer(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
    let shell = app.composer.is_shell_mode();
    // The orchestration command (if any) the buffer leads with — it tints the base
    // border + mark to the command's accent, exactly like shell `!` does its hot-pink.
    let fx_cmd = if shell { None } else { crate::components::text::fx_command(app.composer.text()) };
    // Border/mark token: shell `!` → hot-pink; a `/goal …/hive …/conductor …/morphling`
    // command → its accent (the SAME source `FxCommand::border` uses), so the restyle is
    // VISIBLE at every capability level (mono/NO_COLOR included), with the truecolor
    // `draw_composer_border_fx` overlay layering on top when enabled; else normal/dim.
    let border_tok = if shell {
        Token::ShellAccent
    } else if let Some(cmd) = fx_cmd {
        command_accent(cmd)
    } else if app.busy {
        Token::Dim
    } else {
        Token::Border
    };
    let mark_tok = if shell {
        Token::ShellAccent
    } else if let Some(cmd) = fx_cmd {
        command_accent(cmd)
    } else {
        Token::Claude
    };
    // Rounded corners to match the modern CC/Codex composer aesthetic
    // (clip_20260531_160036). When the flowing-rainbow border is active (the
    // /hive…/morphling gate) the effects painter overwrites these corner cells
    // with its own glyphs, so a single border type here is fine for both states.
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme.color(border_tok)));

    let inner_w = area.width.saturating_sub(4).max(1); // borders + "❯ "
    // Shell mode: the buffer ALREADY starts with the typed `!`, which serves as
    // the prompt — so we must NOT also prepend a `! ` mark, or it renders `! !ls`
    // (the double-`!` bug). The hot-pink border + bottom-hint signal shell mode.
    let mark = if shell { "" } else { "❯ " };

    let lines: Vec<Line> = if app.composer.is_empty() {
        let placeholder = if shell {
            crate::i18n::t(app.lang, "composer.placeholder.shell")
        } else {
            crate::i18n::t(app.lang, "composer.placeholder")
        };
        vec![Line::from(vec![
            Span::styled(mark, Style::default().fg(theme.color(mark_tok))),
            // The cursor cell on an empty buffer.
            Span::styled(" ", Style::default().add_modifier(Modifier::REVERSED)),
            Span::styled(
                placeholder,
                Style::default().fg(theme.color(Token::Dim)),
            ),
        ])]
    } else {
        composer_lines(app, theme, inner_w, mark, mark_tok, shell)
    };

    frame.render_widget(Paragraph::new(lines).block(block), area);
}

/// Build the composer's styled rows with the inverse-cell cursor placed at its
/// visual (row, col). The buffer is rendered as its logical lines (the visual
/// rows from the composer's wrap geometry); the cursor cell on the active row is
/// inverted (robust across terminals while we own the screen).
fn composer_lines<'a>(
    app: &'a AppState,
    theme: &Theme,
    inner_w: u16,
    mark: &'a str,
    mark_tok: Token,
    shell: bool,
) -> Vec<Line<'a>> {
    let rows = app.composer.visual_rows(inner_w);
    let (cur_row, cur_col) = app.composer.cursor_rc(inner_w);
    let text = app.composer.text();
    let text_style = Style::default().fg(theme.color(Token::Text));
    let shell_style = Style::default().fg(theme.color(Token::ShellAccent));

    // Q11b/c: when the buffer leads with an orchestration command (`/goal …`), the
    // command WORD itself gets a per-command char effect (peeled like the shell `!`).
    let fx_cmd = if shell { None } else { crate::components::text::fx_command(text) };

    let mut out: Vec<Line> = Vec::with_capacity(rows.len());
    for (ri, r) in rows.iter().enumerate() {
        let row_text = &text[r.start..r.end.min(text.len())];
        let mut spans: Vec<Span> = Vec::new();
        // The prompt mark only on the first row; a 2-space hanging indent after.
        if ri == 0 {
            spans.push(Span::styled(mark, Style::default().fg(theme.color(mark_tok))));
        } else {
            spans.push(Span::raw("  "));
        }
        // Shell mode: the buffer's OWN leading `!` is the prompt sigil — paint it in
        // hot-pink ShellAccent (matching the pink border + footer `!`), exactly like
        // Claude Code's bash `!` (bashBorder). Only on row 0, and only when the
        // cursor isn't sitting on that `!` cell (col 0) — in that case fall through
        // so the `!` still gets the inverse cursor cell.
        let bang_pink = shell
            && ri == 0
            && row_text.starts_with('!')
            && !(ri == cur_row && cur_col == 0);
        // Command-word effect: peel the leading `/word` token on row 0 and render
        // each char per the command's identity — but ONLY when the cursor is NOT in
        // the token, so the inverse cursor stays visible while it's being typed (the
        // same "row-0, skip if cursor sits on it" guard the `!` peel uses).
        let cmd_tok: Option<(&str, FxCommand)> = match fx_cmd {
            Some(cmd) if ri == 0 && row_text.starts_with('/') => {
                let tok = command_token(row_text);
                let tok_w = UnicodeWidthStr::width(tok);
                if cur_row == 0 && cur_col <= tok_w {
                    None
                } else {
                    Some((tok, cmd))
                }
            }
            _ => None,
        };
        let (lead_pink, lead_cmd_w, row_text) = if bang_pink {
            (true, 0, &row_text[1..])
        } else if let Some((tok, cmd)) = cmd_tok {
            spans.extend(command_word_spans(tok, cmd, theme, app.effects.clock));
            (false, UnicodeWidthStr::width(tok), &row_text[tok.len()..])
        } else {
            (false, 0, row_text)
        };
        if lead_pink {
            // `! ` (with a trailing space) so the command reads `! ls -la`, not
            // `!ls -la` — a clear gap after the bash sigil (user feedback).
            spans.push(Span::styled("! ", shell_style));
        }
        if ri == cur_row {
            // The cursor column is measured from the row start; a peeled `!` (1 cell)
            // or a peeled command word (`lead_cmd_w` cells) shifts the split left.
            let peeled = if lead_pink { 1 } else { lead_cmd_w };
            let split_col = cur_col.saturating_sub(peeled);
            // Split the row around the cursor column and invert one cell.
            let (before, at, after) = split_at_col(row_text, split_col);
            spans.push(Span::styled(before.to_string(), text_style));
            spans.push(Span::styled(
                at.to_string(),
                Style::default().add_modifier(Modifier::REVERSED),
            ));
            spans.push(Span::styled(after.to_string(), text_style));
        } else {
            spans.push(Span::styled(row_text.to_string(), text_style));
        }
        out.push(Line::from(spans));
    }
    if out.is_empty() {
        out.push(Line::from(Span::styled(mark, Style::default().fg(theme.color(mark_tok)))));
    }
    out
}

/// Split a row's text at visual column `col` into (before, cursor-cell, after).
/// The cursor cell is the grapheme at `col` (a space if past end-of-line).
fn split_at_col(row: &str, col: usize) -> (String, String, String) {
    use unicode_segmentation::UnicodeSegmentation;
    use unicode_width::UnicodeWidthStr;
    let mut acc = 0usize;
    let mut before = String::new();
    let mut at = String::new();
    let mut after = String::new();
    let mut placed = false;
    for g in row.graphemes(true) {
        let gw = UnicodeWidthStr::width(g);
        if !placed && acc >= col {
            at.push_str(g);
            placed = true;
        } else if !placed {
            before.push_str(g);
        } else {
            after.push_str(g);
        }
        acc += gw;
    }
    if !placed {
        // Cursor at/after end-of-line → a blank cursor cell.
        at.push(' ');
    }
    (before, at, after)
}

/// The leading `/word` token of a row (slash + the command word, up to the first
/// whitespace) — the unit the command char-effect styles. PURE.
fn command_token(row: &str) -> &str {
    let end = row[1..]
        .find(char::is_whitespace)
        .map(|i| i + 1)
        .unwrap_or(row.len());
    &row[..end]
}

/// The semantic accent [`Token`] for an orchestration command — the SINGLE source
/// the base composer border, the mark, and the per-char word effect all share (the
/// same tokens `FxCommand::border` paints): Hive→Success, Conductor→Suggestion,
/// Goal/Morphling→Claude. PURE.
fn command_accent(cmd: FxCommand) -> Token {
    match cmd {
        FxCommand::Hive => Token::Success,
        FxCommand::Conductor => Token::Suggestion,
        FxCommand::Goal | FxCommand::Morphling => Token::Claude,
    }
}

/// Style each char of the leading command word per its [`FxCommand`] identity
/// (Q11c): Morphling marches the hue through the theme rainbow, Hive paints a swarm
/// of mint shades, Goal/Conductor sweep a sheen across the word. `phase` is the pure
/// effects clock (no `Instant::now`) so the sweep advances frame-to-frame. PURE.
fn command_word_spans(word: &str, cmd: FxCommand, theme: &Theme, phase: f32) -> Vec<Span<'static>> {
    let base = command_accent(cmd);
    let n = word.chars().count().max(1);
    word.char_indices()
        .map(|(i, ch)| {
            let col = i as f32 / n as f32;
            let fg = match cmd {
                FxCommand::Morphling => crate::theme::rainbow::flow_color(theme.rainbow(), col + phase),
                FxCommand::Hive => blend(
                    theme.color(base),
                    lighten(theme.color(base), 0.4),
                    (col * 6.0).fract(),
                ),
                // Goal / Conductor: a sheen sweep across the word.
                _ => {
                    let t = intensity_at(phase.rem_euclid(1.0), 0.25, i, n);
                    blend(theme.color(base), lighten(theme.color(base), 0.35), t)
                }
            };
            Span::styled(ch.to_string(), Style::default().fg(fg).add_modifier(Modifier::BOLD))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_at_col_inverts_one_cell() {
        // Cursor at col 2 of "abcd" → before "ab", at "c", after "d".
        let (b, at, a) = split_at_col("abcd", 2);
        assert_eq!((b.as_str(), at.as_str(), a.as_str()), ("ab", "c", "d"));
        // Cursor past end → a blank cursor cell.
        let (b, at, a) = split_at_col("ab", 5);
        assert_eq!((b.as_str(), at.as_str(), a.as_str()), ("ab", " ", ""));
        // CJK: col counts cells; cursor at col 2 lands after one wide glyph.
        let (b, at, _a) = split_at_col("你好", 2);
        assert_eq!(b, "你");
        assert_eq!(at, "好");
    }

    #[test]
    fn command_token_peels_leading_word() {
        assert_eq!(command_token("/goal build it"), "/goal");
        assert_eq!(command_token("/hive"), "/hive");
        assert_eq!(command_token("/conductor\tgo"), "/conductor");
    }

    /// The command char-effect (Q11c) styles every char of the `/word` token: one
    /// span per char, each with a concrete fg color and BOLD — never a flat run.
    #[test]
    fn command_word_spans_goal_returns_styled_spans() {
        use ratatui::style::{Color, Modifier};
        let theme = Theme::default_theme();
        let spans = command_word_spans("/goal", FxCommand::Goal, &theme, 0.0);
        // One span per char of `/goal` (5 chars).
        assert_eq!(spans.len(), "/goal".chars().count());
        for sp in &spans {
            // Every char carries an explicit RGB fg (not the default/None) + BOLD.
            assert!(matches!(sp.style.fg, Some(Color::Rgb(..))), "char fg must be set: {:?}", sp);
            assert!(sp.style.add_modifier.contains(Modifier::BOLD), "char must be bold");
        }
        // Morphling marches the hue: at a non-zero phase the first/last chars differ.
        let m = command_word_spans("/morphling", FxCommand::Morphling, &theme, 0.0);
        assert_eq!(m.len(), "/morphling".chars().count());
        assert_ne!(m.first().unwrap().style.fg, m.last().unwrap().style.fg, "hue marches");
    }
}
