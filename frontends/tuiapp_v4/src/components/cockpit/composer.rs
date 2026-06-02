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
        // each grapheme per the command's identity. The word STYLES WHILE BEING TYPED
        // — when the cursor sits inside the token we ride the inverse caret on that one
        // grapheme WITHIN the styled word (so `/goal` lights up immediately, not only
        // after a trailing space). When the cursor is at/after the token's trailing
        // edge it falls to the remainder split below (the peeled-column logic).
        let cmd_tok: Option<(&str, FxCommand)> = match fx_cmd {
            Some(cmd) if ri == 0 && row_text.starts_with('/') => Some((command_token(row_text), cmd)),
            _ => None,
        };
        // True once the styled command word has already consumed the cursor cell, so
        // the remainder split below must NOT also invert a cell (one caret per row).
        let mut cursor_in_word = false;
        let (lead_pink, lead_cmd_w, row_text) = if bang_pink {
            (true, 0, &row_text[1..])
        } else if let Some((tok, cmd)) = cmd_tok {
            let tok_w = UnicodeWidthStr::width(tok);
            // The caret is inside the word only when this is the cursor row AND the
            // cursor column lands on one of the word's own cells (`< tok_w`); at exactly
            // `tok_w` it sits on the trailing/space cell handled by the remainder split.
            let word_cursor = if cur_row == 0 && cur_col < tok_w { Some(cur_col) } else { None };
            cursor_in_word = word_cursor.is_some();
            spans.extend(command_word_spans(tok, cmd, theme, app.effects.clock, word_cursor));
            (false, tok_w, &row_text[tok.len()..])
        } else {
            (false, 0, row_text)
        };
        if lead_pink {
            // `! ` (with a trailing space) so the command reads `! ls -la`, not
            // `!ls -la` — a clear gap after the bash sigil (user feedback).
            spans.push(Span::styled("! ", shell_style));
        }
        if ri == cur_row && !cursor_in_word {
            // The cursor column is measured from the row start; a peeled `!` (1 cell)
            // or a peeled command word (`lead_cmd_w` cells) shifts the split left.
            // (Skipped when the caret already rides inside the styled command word —
            // it owns the inverse cell there, so the remainder is plain.)
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

/// The per-char fg for a command word at character index `i` of `n` chars and the
/// pure effects `phase` — each [`FxCommand`] gets a STRONG, col-based, visually
/// DISTINCT effect that reads even at `phase == 0` (the endpoints already differ, so
/// it's a real gradient/swarm, never a flat run). All four are high-amplitude:
/// - Morphling: the full ROYGBIV rainbow marching with the hue (`flow_color`).
/// - Goal: a high-amplitude 2-stop gradient sweep (Claude ↔ `lighten(Claude,0.5)`),
///   a raised cosine over the column so `/g…` and `…l` land on opposite stops.
/// - Hive: an alternating mint SWARM (Success ↔ `lighten(Success,0.5)`) flipping
///   parity by char index — maximal cell-to-cell contrast — with `phase` shifting the
///   parity so the swarm crawls.
/// - Conductor: a clear directional BATON (Suggestion ↔ `lighten(Suggestion,0.5)`), a
///   bright full-amplitude band whose center travels L→R with `phase`.
/// PURE (a function of `i`, `n`, `phase`, theme — no clock).
fn command_char_color(cmd: FxCommand, theme: &Theme, i: usize, n: usize, phase: f32) -> ratatui::style::Color {
    use std::f32::consts::PI;
    let base = command_accent(cmd);
    // Column fraction in [0,1) across the word — the per-char position both the
    // gradient and the baton key on (so the effect is visible statically, by index).
    let col = i as f32 / n as f32;
    match cmd {
        FxCommand::Morphling => crate::theme::rainbow::flow_color(theme.rainbow(), col + phase),
        // Goal: raised-cosine gradient sweep, FULL 0→1 amplitude, travels with phase.
        // At phase 0: i=0 → cos(0)=1 → t=1 (lightened); the last char lands elsewhere
        // on the curve, so first/last colors differ — a real gradient, not flat.
        FxCommand::Goal => {
            let t = 0.5 * (1.0 + (2.0 * PI * (col - phase)).cos());
            blend(theme.color(base), lighten(theme.color(base), 0.5), t)
        }
        // Hive: alternating swarm — even/odd chars snap to opposite mint shades
        // (strong contrast). `phase` shifts the parity so the swarm crawls.
        FxCommand::Hive => {
            let step = (phase * 4.0) as usize;
            let t = if (i + step) % 2 == 0 { 0.0 } else { 1.0 };
            blend(theme.color(base), lighten(theme.color(base), 0.5), t)
        }
        // Conductor: a directional baton — a full-amplitude bright band whose center
        // travels with `phase`; at phase 0 it sits on the first char, so endpoints differ.
        FxCommand::Conductor => {
            let t = intensity_at(phase.rem_euclid(1.0), 0.3, i, n);
            blend(theme.color(base), lighten(theme.color(base), 0.5), t)
        }
    }
}

/// Style each grapheme of the leading command word per its [`FxCommand`] identity
/// (Q11c) via [`command_char_color`]. `phase` is the pure effects clock (no
/// `Instant::now`) so the effect advances frame-to-frame. When `cursor_col` is
/// `Some(c)`, the single grapheme whose visual column spans `c` is rendered with the
/// inverse-cell cursor (BOLD + REVERSED) INSTEAD of its color — so a bare `/goal`
/// lights up the instant it's typed, with the caret riding inside the styled word
/// (CJK-correct: columns are measured by display width, not char count). PURE.
fn command_word_spans(
    word: &str,
    cmd: FxCommand,
    theme: &Theme,
    phase: f32,
    cursor_col: Option<usize>,
) -> Vec<Span<'static>> {
    use unicode_segmentation::UnicodeSegmentation;
    let n = word.chars().count().max(1);
    let mut spans: Vec<Span<'static>> = Vec::with_capacity(n);
    let mut acc = 0usize; // visual columns consumed so far (CJK-correct cursor split)
    for (i, g) in word.graphemes(true).enumerate() {
        let gw = UnicodeWidthStr::width(g);
        // The cursor cell is the grapheme whose column range [acc, acc+gw) covers it.
        let is_cursor = cursor_col.is_some_and(|c| acc <= c && c < acc + gw.max(1));
        let style = if is_cursor {
            Style::default().add_modifier(Modifier::REVERSED | Modifier::BOLD)
        } else {
            let fg = command_char_color(cmd, theme, i, n, phase);
            Style::default().fg(fg).add_modifier(Modifier::BOLD)
        };
        spans.push(Span::styled(g.to_string(), style));
        acc += gw;
    }
    spans
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
    /// span per char, each with a concrete fg color and BOLD — never a flat run. Goal
    /// (like Morphling) is a REAL gradient: its first and last chars differ even at
    /// phase 0; Hive and Conductor are likewise high-amplitude (not flat).
    #[test]
    fn command_word_spans_goal_returns_styled_spans() {
        use ratatui::style::{Color, Modifier};
        let theme = Theme::default_theme();
        let spans = command_word_spans("/goal", FxCommand::Goal, &theme, 0.0, None);
        // One span per char of `/goal` (5 chars).
        assert_eq!(spans.len(), "/goal".chars().count());
        for sp in &spans {
            // Every char carries an explicit RGB fg (not the default/None) + BOLD.
            assert!(matches!(sp.style.fg, Some(Color::Rgb(..))), "char fg must be set: {:?}", sp);
            assert!(sp.style.add_modifier.contains(Modifier::BOLD), "char must be bold");
        }
        // Goal is a gradient sweep: first/last char colors differ at phase 0 (a real
        // gradient, not a flat run) — mirroring the Morphling rainbow assertion.
        assert_ne!(
            spans.first().unwrap().style.fg,
            spans.last().unwrap().style.fg,
            "Goal is a gradient: first/last chars differ at phase 0"
        );
        // Morphling marches the hue: the first/last chars differ.
        let m = command_word_spans("/morphling", FxCommand::Morphling, &theme, 0.0, None);
        assert_eq!(m.len(), "/morphling".chars().count());
        assert_ne!(m.first().unwrap().style.fg, m.last().unwrap().style.fg, "hue marches");
        // Hive (alternating swarm) and Conductor (traveling baton) are likewise
        // high-amplitude — neighbouring/end chars differ, never a flat fill.
        let h = command_word_spans("/hive", FxCommand::Hive, &theme, 0.0, None);
        assert_ne!(h[1].style.fg, h[2].style.fg, "Hive swarm alternates char-to-char");
        let c = command_word_spans("/conductor", FxCommand::Conductor, &theme, 0.0, None);
        assert_ne!(c.first().unwrap().style.fg, c.last().unwrap().style.fg, "Conductor baton is not flat");
    }

    /// Part (b): the cursor-in-token relaxation — when the caret sits INSIDE the
    /// `/word` token, the word is still fully styled and exactly ONE grapheme (the one
    /// under the caret) carries the inverse cell; the others keep their effect color.
    #[test]
    fn command_word_spans_rides_inverse_cursor_inside_word() {
        use ratatui::style::Modifier;
        let theme = Theme::default_theme();
        // Caret at visual col 2 of "/goal" (the `o`): that one grapheme is REVERSED,
        // the rest carry a colored fg. `/goal` is ASCII so col == char index.
        let spans = command_word_spans("/goal", FxCommand::Goal, &theme, 0.0, Some(2));
        let reversed: Vec<usize> = spans
            .iter()
            .enumerate()
            .filter(|(_, s)| s.style.add_modifier.contains(Modifier::REVERSED))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(reversed, vec![2], "exactly the caret grapheme is inverted: {reversed:?}");
        // Every non-caret grapheme keeps a concrete fg color (the effect stays lit).
        for (i, s) in spans.iter().enumerate() {
            if i != 2 {
                assert!(s.style.fg.is_some(), "non-caret char {i} keeps its effect fg");
            }
        }
    }
}
