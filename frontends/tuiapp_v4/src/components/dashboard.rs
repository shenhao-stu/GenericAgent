//! components/dashboard.rs — the full-screen, Claude-Code-style session
//! DASHBOARD (checklist §6 / N2).
//!
//! A SEPARATE full-screen view (entered via left-click on the sessions area OR
//! `Ctrl+S`; `Esc` returns to chat) — NOT a sidebar crowding the composer (the
//! rejected v0.1 design). The layout (§6 mockup):
//!
//!   ◆ GenericAgent · tui_v4   <model> · <cwd>
//!   N awaiting input · M working · K completed
//!   ─────────────────────────────────────────────────────────────
//!   ▾ Needs input
//!     ◆ <name>     <live preview of last assistant line / hint>
//!   ▾ Working
//!     ⏺ getoken    全部就绪。最终汇报…              (heat-colored)
//!   ▸ Completed (K)                                 (collapsed)
//!   ─────────────────────────────────────────────────────────────
//!   ❯ describe a task for a new session
//!   enter open · space reply · ctrl+x delete · r rename · ctrl+n new · ? help
//!
//! Each session row shows the status glyph (`◆` needs-input / `⏺` running with
//! heat color / `○` idle / `✓` done) + name + a LIVE PREVIEW of that session's
//! latest output line (multiplexed from its bridge stream into its transcript).
//! The selected row gets a `❯` cursor + highlight. Categories collapse via `▸`/
//! `▾`. ALL the layout-driving logic (rows, counts, glyphs, preview) is in PURE
//! functions in `app::session`; this module only PAINTS them — and maps the
//! click `(row, col)` back to a row index for left-click switching.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::app::session::{status_glyph, Category, DashRow, SessionMap};
use crate::app::AppState;
use crate::flavor;
use crate::theme::{Theme, Token};

use super::{clip_to, compact_cwd};

/// The dashboard's fixed chrome rows: 1 header + 1 counts + 1 separator at the
/// top, and 1 separator + 1 new-session input + 1 key-hint at the bottom. The
/// session-row list flexes between them.
const TOP_CHROME: u16 = 3;
const BOTTOM_CHROME: u16 = 3;

/// The y of the first session-list row in the dashboard (= TOP_CHROME). Used to
/// map a left-click's row back to a [`DashRow`] index (with the list scroll
/// offset). Public so the app's mouse handler can reuse the exact same geometry.
pub const LIST_TOP_Y: u16 = TOP_CHROME;

/// Draw the full-screen dashboard. `now_ms` drives the running-row spinner + heat.
pub fn render(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme, now_ms: u64) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),            // header: identity + model + cwd
            Constraint::Length(1),            // counts line
            Constraint::Length(1),            // separator (rainbow)
            Constraint::Min(0),               // session-row list (FLEX)
            Constraint::Length(1),            // separator (rainbow)
            Constraint::Length(1),            // new-session input row
            Constraint::Length(1),            // key-hint footer
        ])
        .split(area);

    render_header(frame, chunks[0], app, theme);
    render_counts(frame, chunks[1], app, theme);
    render_rainbow(frame, chunks[2], theme);
    render_list(frame, chunks[3], app, theme, now_ms);
    render_rainbow(frame, chunks[4], theme);
    render_new_input(frame, chunks[5], app, theme);
    render_hints(frame, chunks[6], app, theme);
}

/// HEADER: identity + active model + cwd (the §6 top line).
fn render_header(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
    let model = app.model.as_deref().unwrap_or("—");
    let cwd = compact_cwd(&app.cwd, 32);
    let spans = vec![
        Span::styled(
            "◆ ",
            Style::default().fg(theme.color(Token::Claude)).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            "GenericAgent · tui_v4",
            Style::default().fg(theme.color(Token::Text)).add_modifier(Modifier::BOLD),
        ),
        Span::styled("   ", Style::default()),
        Span::styled(model.to_string(), Style::default().fg(theme.color(Token::Claude))),
        Span::styled("  ·  ", Style::default().fg(theme.color(Token::Dim))),
        Span::styled(cwd, Style::default().fg(theme.color(Token::Dim))),
    ];
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// COUNTS: `N awaiting input · M working · K completed` (§6 header line).
fn render_counts(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
    let c = app.sessions.counts();
    let spans = vec![
        Span::styled(
            format!("{}", c.needs_input),
            Style::default().fg(theme.color(Token::Warning)).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" awaiting input", Style::default().fg(theme.color(Token::Dim))),
        Span::styled("  ·  ", Style::default().fg(theme.color(Token::Dim))),
        Span::styled(
            format!("{}", c.working),
            Style::default().fg(theme.color(Token::Claude)).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" working", Style::default().fg(theme.color(Token::Dim))),
        Span::styled("  ·  ", Style::default().fg(theme.color(Token::Dim))),
        Span::styled(
            format!("{}", c.completed),
            Style::default().fg(theme.color(Token::Success)).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" completed", Style::default().fg(theme.color(Token::Dim))),
    ];
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// A full-width rainbow separator (reuses the cockpit's 7-stop ramp).
fn render_rainbow(frame: &mut Frame, area: Rect, theme: &Theme) {
    let width = area.width;
    let mut spans: Vec<Span> = Vec::with_capacity(width as usize);
    for x in 0..width {
        spans.push(Span::styled("─", Style::default().fg(theme.rainbow_at(x, width))));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// The session-row list: category headers (collapsible) + their session rows.
/// Renders the visible window of [`SessionMap::dashboard_rows`] scrolled so the
/// selection is in view.
fn render_list(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme, now_ms: u64) {
    let rows = app.sessions.dashboard_rows();
    let height = area.height as usize;
    let inner_w = area.width as usize;
    let offset = list_scroll_offset(app.sessions.dash_sel, rows.len(), height);

    let mut lines: Vec<Line> = Vec::with_capacity(height);
    for (i, row) in rows.iter().enumerate().skip(offset).take(height) {
        let selected = i == app.sessions.dash_sel;
        lines.push(render_row(row, &app.sessions, selected, theme, now_ms, inner_w, app));
    }
    if rows.is_empty() {
        lines.push(Line::from(Span::styled(
            "  no sessions — type below to start one",
            Style::default().fg(theme.color(Token::Dim)),
        )));
    }
    frame.render_widget(Paragraph::new(lines), area);
}

/// Render ONE dashboard row (header or session) to a styled line.
fn render_row<'a>(
    row: &DashRow,
    map: &'a SessionMap,
    selected: bool,
    theme: &Theme,
    now_ms: u64,
    inner_w: usize,
    app: &'a AppState,
) -> Line<'a> {
    match row {
        DashRow::Header { category, count, collapsed } => {
            let chevron = if *collapsed { "▸" } else { "▾" };
            let title = category.title(app.lang);
            let mut spans = vec![
                Span::styled(
                    format!("{chevron} "),
                    Style::default().fg(theme.color(Token::Suggestion)),
                ),
                Span::styled(
                    title.to_string(),
                    Style::default()
                        .fg(theme.color(category_token(*category)))
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(" ({count})"),
                    Style::default().fg(theme.color(Token::Dim)),
                ),
            ];
            if selected {
                // A subtle leading cursor so a header can be the Tab/▸▾ target.
                spans.insert(0, Span::styled("❯", Style::default().fg(theme.color(Token::Suggestion))));
                // (the chevron span already has a trailing space)
            } else {
                spans.insert(0, Span::raw(" "));
            }
            Line::from(spans)
        }
        DashRow::Session { id, .. } => {
            let Some(s) = map.session(*id) else {
                return Line::from(Span::raw(""));
            };
            let is_active = *id == map.active;
            let status = s.status();
            let elapsed = s.busy_since_ms;
            let elapsed_ms = if status == crate::app::session::SessionStatus::Working {
                now_ms.saturating_sub(elapsed)
            } else {
                0
            };
            // Running rows animate the spinner glyph in the heat color; others use
            // their static status glyph.
            let (glyph, glyph_tok) = status_glyph(status, elapsed_ms);
            let glyph_str = if status == crate::app::session::SessionStatus::Working {
                let tick = (now_ms / 100) as u64;
                app.companion.spinner_style().glyph(tick).to_string()
            } else {
                glyph.to_string()
            };

            // Rename-in-progress on this row → show the edit buffer with a caret.
            let renaming = app.rename.as_ref().filter(|r| r.id == *id);
            let name = match renaming {
                Some(r) => r.buffer.clone(),
                None => s.name.clone(),
            };

            // Layout budget: "  ❯ ⏺ name   preview".
            let cursor = if selected { "❯ " } else { "  " };
            let name_budget = name_budget(inner_w);
            let name_clip = clip_to(&name, name_budget);
            let preview_indent = 2 /*cursor*/ + 2 /*glyph + space*/ + name_budget + 3;
            let preview_budget = inner_w.saturating_sub(preview_indent).max(4);
            let preview = clip_to(&s.preview(), preview_budget);

            let name_color = if selected {
                Token::Suggestion
            } else if is_active {
                Token::Claude
            } else {
                Token::Text
            };
            let name_mod = if selected || is_active {
                Modifier::BOLD
            } else {
                Modifier::empty()
            };

            let mut spans = vec![
                Span::styled(cursor, Style::default().fg(theme.color(Token::Suggestion))),
                Span::styled(
                    format!("{glyph_str} "),
                    Style::default().fg(theme.color(glyph_tok)).add_modifier(Modifier::BOLD),
                ),
            ];
            // The currently-VIEWED session gets a filled-bullet marker before its
            // name (Ink "currently-viewed = filled bullet").
            if is_active {
                spans.push(Span::styled("● ", Style::default().fg(theme.color(Token::Success))));
            }
            spans.push(Span::styled(
                name_clip,
                Style::default().fg(theme.color(name_color)).add_modifier(name_mod),
            ));
            if renaming.is_some() {
                // A caret cell while renaming.
                spans.push(Span::styled(" ", Style::default().add_modifier(Modifier::REVERSED)));
                spans.push(Span::styled(
                    "  (renaming · Enter ✓ · Esc ✗)",
                    Style::default().fg(theme.color(Token::Dim)),
                ));
            } else {
                spans.push(Span::styled("   ", Style::default()));
                // Running preview is heat-colored; others dim.
                let preview_tok = if status == crate::app::session::SessionStatus::Working {
                    glyph_tok
                } else {
                    Token::Dim
                };
                spans.push(Span::styled(preview, Style::default().fg(theme.color(preview_tok))));
            }
            Line::from(spans)
        }
    }
}

/// NEW-SESSION input row: `❯ <typed text or placeholder>` (§6 bottom field).
fn render_new_input(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
    let buf = &app.sessions.new_session_input;
    let mut spans = vec![Span::styled(
        "❯ ",
        Style::default().fg(theme.color(Token::Claude)).add_modifier(Modifier::BOLD),
    )];
    if buf.is_empty() {
        spans.push(Span::styled(" ", Style::default().add_modifier(Modifier::REVERSED)));
        spans.push(Span::styled(
            crate::i18n::t(app.lang, "dash.new_session"),
            Style::default().fg(theme.color(Token::Dim)),
        ));
    } else {
        spans.push(Span::styled(buf.clone(), Style::default().fg(theme.color(Token::Text))));
        spans.push(Span::styled(" ", Style::default().add_modifier(Modifier::REVERSED)));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// KEY-HINT footer (§6). Driven by the i18n language for the small words later;
/// literal for now.
fn render_hints(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
    let _ = app;
    let hint = |k: &'static str, a: &'static str| -> Vec<Span<'static>> {
        vec![
            Span::styled(k, Style::default().fg(theme.color(Token::Text)).add_modifier(Modifier::BOLD)),
            Span::styled(format!(" {a}"), Style::default().fg(theme.color(Token::Dim))),
        ]
    };
    let dot = || Span::styled("  ·  ", Style::default().fg(theme.color(Token::Dim)));
    let mut spans: Vec<Span> = Vec::new();
    spans.extend(hint("↑↓", "nav"));
    spans.push(dot());
    spans.extend(hint("enter", "open"));
    spans.push(dot());
    spans.extend(hint("space", "reply"));
    spans.push(dot());
    spans.extend(hint("r", "rename"));
    spans.push(dot());
    spans.extend(hint("ctrl+x/del", "delete"));
    spans.push(dot());
    spans.extend(hint("ctrl+n", "new"));
    spans.push(dot());
    spans.extend(hint("tab", "fold"));
    spans.push(dot());
    spans.extend(hint("esc", "back"));
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

// ---- pure layout helpers (unit-tested) -------------------------------------

/// The theme token a category header is tinted with.
fn category_token(cat: Category) -> Token {
    match cat {
        Category::NeedsInput => Token::Warning,
        Category::Working => Token::Claude,
        Category::Completed => Token::Success,
    }
}

/// The name column budget for a dashboard row given the list inner width. PURE.
fn name_budget(inner_w: usize) -> usize {
    (inner_w / 3).clamp(8, 28)
}

/// The first visible row index so `sel` is on-screen in a `height`-row window
/// (keeps the selection centered-ish; clamps at the ends). PURE + unit-tested.
pub fn list_scroll_offset(sel: usize, len: usize, height: usize) -> usize {
    if len <= height || height == 0 {
        return 0;
    }
    let half = height / 2;
    sel.saturating_sub(half).min(len - height)
}

/// Map a left-click at terminal `(col, row)` inside the dashboard `area` back to a
/// [`DashRow`] index, accounting for the list's top y + scroll offset. Returns
/// `None` if the click is outside the session-row list (header/counts/separators/
/// input/hints). PURE — the app's mouse handler uses it to switch on a row click.
pub fn click_to_row_index(
    col: u16,
    row: u16,
    area: Rect,
    rows_len: usize,
    sel: usize,
) -> Option<usize> {
    // Inside the area horizontally?
    if col < area.x || col >= area.x + area.width {
        return None;
    }
    let list_top = area.y + LIST_TOP_Y;
    let list_height = area.height.saturating_sub(TOP_CHROME + BOTTOM_CHROME) as usize;
    if list_height == 0 {
        return None;
    }
    if row < list_top || row >= list_top + list_height as u16 {
        return None;
    }
    let offset = list_scroll_offset(sel, rows_len, list_height);
    let idx = offset + (row - list_top) as usize;
    if idx < rows_len {
        Some(idx)
    } else {
        None
    }
}

/// A small `flavor`-backed spinner glyph for a running dashboard row (kept as a
/// thin pass-through so the dashboard's running animation matches the cockpit's
/// spinner identity — never the CC asterisk). PURE-ish (deterministic by tick).
#[allow(dead_code)]
pub fn running_glyph(style: flavor::SpinnerStyle, now_ms: u64) -> char {
    style.glyph((now_ms / 100) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_scroll_offset_keeps_selection_visible() {
        // Fits → no scroll.
        assert_eq!(list_scroll_offset(0, 5, 10), 0);
        assert_eq!(list_scroll_offset(4, 5, 10), 0);
        // Overflow → selection centered, clamped at both ends.
        assert_eq!(list_scroll_offset(0, 100, 10), 0);
        assert_eq!(list_scroll_offset(50, 100, 10), 45);
        assert_eq!(list_scroll_offset(99, 100, 10), 90); // clamped to len-height.
        // Degenerate height.
        assert_eq!(list_scroll_offset(5, 100, 0), 0);
    }

    #[test]
    fn name_budget_is_bounded() {
        assert_eq!(name_budget(0), 8); // floor.
        assert_eq!(name_budget(120), 28); // ceiling (120/3=40 → clamp 28).
        assert_eq!(name_budget(45), 15); // 45/3.
    }

    #[test]
    fn click_maps_into_the_list_region_only() {
        // A 80x24 dashboard at origin: list spans y=3..(24-3)=21 (18 rows).
        let area = Rect { x: 0, y: 0, width: 80, height: 24 };
        let rows_len = 5;
        // A click on the first list row (y=3) with no scroll → row 0.
        assert_eq!(click_to_row_index(10, 3, area, rows_len, 0), Some(0));
        // y=5 → row 2.
        assert_eq!(click_to_row_index(10, 5, area, rows_len, 0), Some(2));
        // Clicking the header/counts/separator region (y<3) → None.
        assert_eq!(click_to_row_index(10, 0, area, rows_len, 0), None);
        assert_eq!(click_to_row_index(10, 2, area, rows_len, 0), None);
        // Clicking the bottom chrome (y>=21) → None.
        assert_eq!(click_to_row_index(10, 22, area, rows_len, 0), None);
        // A row beyond the session count → None (clicked empty list space).
        assert_eq!(click_to_row_index(10, 10, area, rows_len, 0), None);
        // Outside the area horizontally → None.
        assert_eq!(click_to_row_index(200, 3, area, rows_len, 0), None);
    }

    #[test]
    fn category_tokens_are_distinct() {
        assert_eq!(category_token(Category::NeedsInput), Token::Warning);
        assert_eq!(category_token(Category::Working), Token::Claude);
        assert_eq!(category_token(Category::Completed), Token::Success);
    }

    /// Render the dashboard to an in-memory backend and confirm the headline
    /// chrome appears: the three category titles, the counts line, and the
    /// new-session prompt. A render-level guard that the full-screen view paints.
    #[test]
    fn dashboard_renders_categories_and_chrome() {
        use crate::app::View;
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let mut app = AppState::new();
        // A running session + a fresh idle one so two categories are non-empty.
        app.view = View::Dashboard;
        let _s2 = app.sessions.new_session(Some("worker".into()));
        // Mark the active (s2) busy on the LIVE field so the snapshot mirrors it
        // into the record → it lands in the Working category.
        app.busy = true;
        app.snapshot_active_into_map();

        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let theme = Theme::default_theme();
        terminal
            .draw(|f| {
                let area = f.area();
                render(f, area, &app, &theme, 1000);
            })
            .unwrap();

        let buf = terminal.backend().buffer();
        let text: String = buf.content().iter().map(|c| c.symbol()).collect();
        assert!(text.contains("GenericAgent"), "header identity present");
        assert!(text.contains("awaiting input"), "counts line present");
        assert!(text.contains("Needs input"), "NeedsInput category header present");
        assert!(text.contains("Working"), "Working category header present");
        assert!(text.contains("Completed"), "Completed category header present");
        assert!(
            text.contains("describe a task for a new session"),
            "new-session prompt present"
        );
    }
}
