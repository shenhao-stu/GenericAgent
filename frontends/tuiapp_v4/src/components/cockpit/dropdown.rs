//! cockpit/dropdown.rs — the completion dropdown (slash palette / `@` file picker)
//! and the height math for the composer + dropdown rows.

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::app::AppState;
use crate::commands::registry::{self, SlashCommand};
use crate::input::file_expand;
use crate::theme::{Theme, Token};

use super::COMPOSER_MAX_ROWS;

/// How many rows the active completion dropdown needs (0 = none).
pub(crate) fn dropdown_height(app: &AppState, _width: u16) -> u16 {
    if app.composer.is_empty() {
        return 0;
    }
    let text = app.composer.text();
    let matches = registry::palette_matches(text);
    if registry::palette_visible(text, &matches) {
        return (matches.len().min(registry::PALETTE_ROWS) as u16) + 1; // +1 hint row.
    }
    if let Some(q) = app.composer.at_query() {
        let files = app.list_project_files();
        let ranked = file_expand::rank_files(&q.partial, &files);
        if !ranked.is_empty() {
            return (ranked.len().min(file_expand::MAX_PICKER_ROWS) as u16) + 1;
        }
    }
    0
}

/// Render the active completion dropdown (palette OR file picker).
pub(crate) fn render_dropdown(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
    let text = app.composer.text();
    let matches = registry::palette_matches(text);
    if registry::palette_visible(text, &matches) {
        render_palette(frame, area, &matches, app, theme);
        return;
    }
    if let Some(q) = app.composer.at_query() {
        let files = app.list_project_files();
        let ranked = file_expand::rank_files(&q.partial, &files);
        render_file_picker(frame, area, &ranked, app.composer.file_sel, theme, app.lang);
    }
}

/// The slash-command palette dropdown.
fn render_palette(
    frame: &mut Frame,
    area: Rect,
    matches: &[SlashCommand],
    app: &AppState,
    theme: &Theme,
) {
    // The highlighted row (↑/↓ move `app.palette_sel`; Tab/Enter complete it),
    // clamped to the live match count. Previously hardcoded to 0 so the highlight
    // was stuck on the top row regardless of arrow keys.
    let sel = app.palette_sel.min(matches.len().saturating_sub(1));
    let (start, slice) = registry::palette_window(matches, sel);
    let mut lines: Vec<Line> = Vec::new();
    for (i, cmd) in slice.iter().enumerate() {
        let active = start + i == sel;
        let (cursor, name_tok) = if active {
            ("❯ ", Token::Suggestion)
        } else {
            ("  ", Token::Text)
        };
        lines.push(Line::from(vec![
            Span::styled(cursor, Style::default().fg(theme.color(Token::Suggestion))),
            Span::styled(
                format!("/{}", cmd.name),
                Style::default()
                    .fg(theme.color(name_tok))
                    .add_modifier(if active { Modifier::BOLD } else { Modifier::empty() }),
            ),
            Span::styled("   ", Style::default()),
            Span::styled(cmd.desc.to_string(), Style::default().fg(theme.color(Token::Dim))),
        ]));
    }
    lines.push(Line::from(Span::styled(
        crate::i18n::t(app.lang, "palette.hint"),
        Style::default().fg(theme.color(Token::Dim)),
    )));
    frame.render_widget(Paragraph::new(lines), area);
}

/// The `@`-file-picker dropdown.
fn render_file_picker(
    frame: &mut Frame,
    area: Rect,
    files: &[String],
    sel: usize,
    theme: &Theme,
    lang: crate::i18n::Lang,
) {
    let sel = if files.is_empty() { 0 } else { sel % files.len() };
    let mut lines: Vec<Line> = Vec::new();
    for (i, f) in files.iter().take(file_expand::MAX_PICKER_ROWS).enumerate() {
        let active = i == sel;
        lines.push(Line::from(vec![
            Span::styled(
                if active { "❯ " } else { "  " },
                Style::default().fg(theme.color(Token::Suggestion)),
            ),
            Span::styled(
                f.clone(),
                Style::default()
                    .fg(theme.color(if active { Token::Suggestion } else { Token::Text }))
                    .add_modifier(if active { Modifier::BOLD } else { Modifier::empty() }),
            ),
        ]));
    }
    lines.push(Line::from(Span::styled(
        crate::i18n::t(lang, "filepicker.hint"),
        Style::default().fg(theme.color(Token::Dim)),
    )));
    frame.render_widget(Paragraph::new(lines), area);
}

/// The composer's rendered height (rows + 2 for the border), clamped to
/// `[3, COMPOSER_MAX_ROWS+2]`. PURE over the buffer geometry.
pub(crate) fn composer_height(app: &AppState, inner_w: u16) -> u16 {
    let rows = app.composer.visual_rows(inner_w).len() as u16;
    (rows.clamp(1, COMPOSER_MAX_ROWS)) + 2
}
