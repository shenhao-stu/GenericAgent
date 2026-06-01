//! cockpit/dropdown.rs — the completion dropdown (slash palette / `@` file picker)
//! and the height math for the composer + dropdown rows.

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::app::AppState;
use crate::commands::registry::{self, SlashCommand};
use crate::components::picker::window_slice;
use crate::input::file_expand;
use crate::theme::{Theme, Token};

use super::COMPOSER_MAX_ROWS;

/// How many rows the `@`-file picker occupies for `ranked` matches: the visible
/// window (≤ [`file_expand::MAX_PICKER_ROWS`]) + the hint row + one "+N more" tail
/// row when there are more matches than fit. Single source of truth shared by
/// `dropdown_height` and `render_file_picker` so the layout and paint never disagree.
fn file_picker_rows(n: usize) -> u16 {
    let window = n.min(file_expand::MAX_PICKER_ROWS);
    let more = usize::from(n > file_expand::MAX_PICKER_ROWS);
    (window + 1 + more) as u16
}

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
            return file_picker_rows(ranked.len());
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

/// The `@`-file-picker dropdown. `files` is the FULL ranked match set; we paint a
/// scrolling [`file_expand::MAX_PICKER_ROWS`] window that follows `sel` (via the
/// shared [`window_slice`] helper, same as the `/continue` picker) and a dim
/// "… +N more" tail row when more matches exist than fit — so every match is
/// reachable by arrowing, not silently dropped at row 8.
fn render_file_picker(
    frame: &mut Frame,
    area: Rect,
    files: &[String],
    sel: usize,
    theme: &Theme,
    lang: crate::i18n::Lang,
) {
    let sel = if files.is_empty() { 0 } else { sel % files.len() };
    let (start, slice) = window_slice(files, sel, file_expand::MAX_PICKER_ROWS);
    let mut lines: Vec<Line> = Vec::new();
    for (i, f) in slice.iter().enumerate() {
        let active = start + i == sel;
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
    let hidden = files.len().saturating_sub(file_expand::MAX_PICKER_ROWS);
    if hidden > 0 {
        lines.push(Line::from(Span::styled(
            format!("  … +{hidden} {}", crate::i18n::t(lang, "filepicker.more")),
            Style::default().fg(theme.color(Token::Dim)),
        )));
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

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    /// Render the `@`-file-picker dropdown for `app` into a TestBackend and return
    /// its rows as trimmed strings. Drives the REAL path: `render_dropdown` reads
    /// `app.list_project_files()` (an actual walk over `app.repo_root`) → `rank_files`
    /// → `render_file_picker`.
    fn picker_rows(app: &AppState) -> Vec<String> {
        let theme = Theme::default_theme();
        let (w, h) = (60u16, 14u16);
        let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
        term.draw(|f| render_dropdown(f, Rect::new(0, 0, w, h), app, &theme)).unwrap();
        let buf = term.backend().buffer();
        (0..h as usize)
            .map(|y| {
                let mut row = String::new();
                for x in 0..w as usize {
                    row.push_str(buf.content()[y * w as usize + x].symbol());
                }
                row.trim_end().to_string()
            })
            .filter(|r| !r.is_empty())
            .collect()
    }

    /// HONEST CHECK (live render): with FAR more `@` matches than `MAX_PICKER_ROWS`,
    /// the dropdown paints a scrolling window of exactly `MAX_PICKER_ROWS` file rows
    /// PLUS a dim "… +N more" tail, and the window FOLLOWS `file_sel` (a late match
    /// scrolls into view while the first one scrolls off) — so every match is
    /// reachable, not silently capped at 8 unreachable rows.
    #[test]
    fn file_picker_windows_all_matches_with_more_hint() {
        let total = file_expand::MAX_PICKER_ROWS + 8; // 16 matches > 8 rows.
        let dir = std::env::temp_dir().join(format!("tui_v4_dropdown_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for i in 0..total {
            std::fs::write(dir.join(format!("component{i:02}.rs")), "x").unwrap();
        }

        let mut app = AppState::new();
        app.repo_root = dir.clone();
        // Type `@component` so all `component*.rs` files match.
        app.composer.set_buffer("@component".to_string(), "@component".len());

        // Sanity: the data layer surfaces ALL matches (not 8).
        let files = app.list_project_files();
        let ranked = file_expand::rank_files("component", &files);
        assert_eq!(ranked.len(), total, "rank_files returns the full match set");

        // Selection at the TOP: window shows component00..07, "+8 more" tail.
        app.composer.file_sel = 0;
        let rows = picker_rows(&app);
        let file_rows: Vec<&String> = rows.iter().filter(|r| r.contains("component")).collect();
        assert_eq!(file_rows.len(), file_expand::MAX_PICKER_ROWS, "exactly one window of rows");
        assert!(rows.iter().any(|r| r.contains("component00.rs")), "top-of-list visible");
        let hidden = total - file_expand::MAX_PICKER_ROWS;
        assert!(
            rows.iter().any(|r| r.contains(&format!("+{hidden}")) && r.contains("more")),
            "the '+N more' tail row is shown when matches exceed the window: {rows:?}"
        );
        // The LAST match is NOT yet on screen at the top.
        assert!(
            !rows.iter().any(|r| r.contains(&format!("component{:02}.rs", total - 1))),
            "the last match is off-screen until the selection scrolls to it"
        );

        // Move the selection to the LAST match: the window follows it into view and
        // the first row scrolls off — proving the late match is reachable.
        app.composer.file_sel = total - 1;
        let rows2 = picker_rows(&app);
        assert!(
            rows2.iter().any(|r| r.contains(&format!("component{:02}.rs", total - 1))),
            "selecting the last match scrolls it into the window: {rows2:?}"
        );
        assert!(
            !rows2.iter().any(|r| r.contains("component00.rs")),
            "the window scrolled — the first match is no longer visible"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
