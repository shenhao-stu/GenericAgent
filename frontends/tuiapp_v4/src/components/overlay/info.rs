//! overlay/info.rs — the read-only info cards: `/help` (command list), `/status`
//! (model/connection/session snapshot), `/cost` (token-usage report), `/verbose`
//! (tool-call audit), and the `/btw` side-answer card. No hardcoded colors.

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::app::AppState;
use crate::commands::registry::{aliases_of, primaries_of_kind, CommandKind, COMMANDS};
use crate::components::{clip_to, truncate_model, MODEL_LABEL_CAP};
use crate::i18n::{self, Lang};
use crate::theme::{Theme, Token};

use super::titled_block;

/// The full command-list overlay (`/help`), grouped by [`CommandKind`].
pub(crate) fn render_help(frame: &mut Frame, area: Rect, theme: &Theme, lang: Lang) {
    frame.render_widget(Clear, area);
    let block = titled_block(i18n::t(lang, "help.title"), theme);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let mut lines: Vec<Line> = Vec::new();
    let alias_of_label = i18n::t(lang, "help.alias_of");
    let group = |lines: &mut Vec<Line>, title: &str, kind: CommandKind| {
        lines.push(Line::from(Span::styled(
            title.to_string(),
            Style::default().fg(theme.color(Token::Claude)).add_modifier(Modifier::BOLD),
        )));
        // Only PRIMARY commands are peer rows; their aliases hang under them as a
        // dim "alias of /primary" line (Q6 — no duplicate peer rows).
        for c in primaries_of_kind(kind) {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  /{:<12}", c.name),
                    Style::default().fg(theme.color(Token::Suggestion)),
                ),
                Span::styled(c.desc.to_string(), Style::default().fg(theme.color(Token::Dim))),
            ]));
            for a in aliases_of(c.name) {
                lines.push(Line::from(Span::styled(
                    format!("    /{} — {} /{}", a.name, alias_of_label, c.name),
                    Style::default().fg(theme.color(Token::Dim)),
                )));
            }
        }
        lines.push(Line::from(""));
    };
    group(&mut lines, i18n::t(lang, "help.group.ui"), CommandKind::Ui);
    group(&mut lines, i18n::t(lang, "help.group.app"), CommandKind::App);
    group(&mut lines, i18n::t(lang, "help.group.fwd"), CommandKind::Fwd);
    lines.push(Line::from(Span::styled(
        i18n::t(lang, "help.magic"),
        Style::default().fg(theme.color(Token::Text)),
    )));
    lines.push(Line::from(Span::styled(
        i18n::t(lang, "help.close"),
        Style::default().fg(theme.color(Token::Dim)),
    )));
    frame.render_widget(Paragraph::new(lines).scroll((help_scroll(lines_len_hint(), inner.height), 0)), inner);
}

/// A seam for a future help-scroll cursor; the list fits a normal terminal, so it
/// returns 0 (top). PURE.
fn help_scroll(_total: usize, _height: u16) -> u16 {
    0
}

fn lines_len_hint() -> usize {
    COMMANDS.len() + 8
}

/// The keyboard-shortcut pairs `(chord, description-key)` shown by `/keybindings`
/// (Q7) — the cockpit's real chord map incl. the C3 parity gaps it filled (Ctrl+/
/// help, Ctrl+T theme, Ctrl+Enter newline). The chord literals are ASCII (stable
/// across terminals); the descriptions resolve through i18n. PURE.
pub(crate) fn keybinding_pairs() -> &'static [(&'static str, &'static str)] {
    &[
        ("Enter", "kb.submit"),
        ("Shift+Enter / Ctrl+Enter / Ctrl+J", "kb.newline"),
        ("/", "kb.palette"),
        ("Tab", "kb.complete"),
        ("Ctrl+O", "kb.copy_reply"),
        ("Ctrl+Shift+O / /fold", "kb.fold"),
        ("Ctrl+Shift+M", "kb.mouse"),
        ("PgUp/PgDn · Ctrl+Home/End", "kb.scroll"),
        ("←/→ (empty input)", "kb.views"),
        ("Ctrl+S", "kb.dashboard"),
        ("Ctrl+G", "kb.stash"),
        ("Ctrl+N", "kb.new_session"),
        ("Ctrl+↑/↓", "kb.cycle_session"),
        ("Ctrl+W / Ctrl+D", "kb.drop_session"),
        ("Ctrl+B", "kb.branch"),
        ("Ctrl+T", "kb.theme"),
        ("Ctrl+/ · Ctrl+_", "kb.help"),
        ("Esc · Esc Esc", "kb.escape"),
        ("Ctrl+C · Ctrl+Q", "kb.quit"),
    ]
}

/// The `/keybindings` cheat-sheet overlay (Q7): the chord→action pairs table + the
/// magic-prefix line. Full-screen, no hardcoded colors.
pub(crate) fn render_keybindings(frame: &mut Frame, area: Rect, theme: &Theme, lang: Lang) {
    frame.render_widget(Clear, area);
    let block = titled_block(i18n::t(lang, "keybindings.title"), theme);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let mut lines: Vec<Line> = Vec::new();
    for (chord, desc_key) in keybinding_pairs() {
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {chord:<34}"),
                Style::default().fg(theme.color(Token::Suggestion)),
            ),
            Span::styled(i18n::t(lang, desc_key).to_string(), Style::default().fg(theme.color(Token::Dim))),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        i18n::t(lang, "mouse.hint.native"),
        Style::default().fg(theme.color(Token::Text)),
    )));
    lines.push(Line::from(Span::styled(
        i18n::t(lang, "help.magic"),
        Style::default().fg(theme.color(Token::Text)),
    )));
    lines.push(Line::from(Span::styled(
        i18n::t(lang, "help.close"),
        Style::default().fg(theme.color(Token::Dim)),
    )));
    frame.render_widget(Paragraph::new(lines), inner);
}

/// The status snapshot overlay (`/status` `/sessions`): model / connection /
/// session counts / context / cwd / git.
pub(crate) fn render_status(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme, now_ms: u64) {
    let w = (area.width.saturating_sub(8)).clamp(40, 80);
    let h = 14u16.min(area.height.saturating_sub(2)).max(8);
    let card = super::centered(area, w, h);
    frame.render_widget(Clear, card);
    let block = titled_block("Status · /status", theme);
    let inner = block.inner(card);
    frame.render_widget(block, card);

    let counts = app.sessions.counts();
    let kv = |k: &str, v: String, tok: Token, theme: &Theme| -> Line<'static> {
        Line::from(vec![
            Span::styled(format!("  {k:<12}"), Style::default().fg(theme.color(Token::Dim))),
            Span::styled(v, Style::default().fg(theme.color(tok))),
        ])
    };
    let model = app.model.clone().unwrap_or_else(|| "—".to_string());
    let state = if app.busy { "working" } else { "idle" };
    let elapsed = app.turn_elapsed_ms(now_ms) as f64 / 1000.0;
    let effort = app.effort_label().unwrap_or("default");
    let lines = vec![
        kv("model", model, Token::Claude, theme),
        kv("connection", app.conn.label(), Token::Success, theme),
        kv("state", format!("{state} ({elapsed:.1}s)"), Token::Warning, theme),
        kv("effort", effort.to_string(), Token::Claude, theme),
        kv("sessions", format!("{} total", app.sessions.len()), Token::Text, theme),
        kv(
            "  by status",
            format!(
                "{} needs-input · {} working · {} completed",
                counts.needs_input, counts.working, counts.completed
            ),
            Token::Dim,
            theme,
        ),
        kv(
            "context",
            app.context_percent.map(|p| format!("{p:.0}%")).unwrap_or_else(|| "—".into()),
            Token::Suggestion,
            theme,
        ),
        kv("cost", format!("${:.4}", app.cost_usd), Token::Success, theme),
        kv("git", app.git_branch.clone().unwrap_or_else(|| "—".into()), Token::Suggestion, theme),
        kv("cwd", clip_to(&app.cwd, inner.width.saturating_sub(14) as usize), Token::Text, theme),
        Line::from(Span::styled("  esc close", Style::default().fg(theme.color(Token::Dim)))),
    ];
    frame.render_widget(Paragraph::new(lines), inner);
}

/// The token-usage report (`/cost`) — rendered from the pure
/// [`crate::app::CostBreakdown::report_lines`].
pub(crate) fn render_cost(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
    let w = 48u16.min(area.width.saturating_sub(2)).max(30);
    // Truncate the model to its primary segment (never the MixinSession pipe-list).
    let model = truncate_model(app.model.as_deref().unwrap_or("—"), MODEL_LABEL_CAP);
    let report = app.cost.report_lines(&model);
    let h = (report.len() as u16 + 3).min(area.height.saturating_sub(2)).max(8);
    let card = super::centered(area, w, h);
    frame.render_widget(Clear, card);
    let block = titled_block("Cost · /cost", theme);
    let inner = block.inner(card);
    frame.render_widget(block, card);

    let mut lines: Vec<Line> = report
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let tok = if i == 0 { Token::Claude } else { Token::Text };
            Line::from(Span::styled(l.clone(), Style::default().fg(theme.color(tok))))
        })
        .collect();
    lines.push(Line::from(Span::styled("  esc close", Style::default().fg(theme.color(Token::Dim)))));
    frame.render_widget(Paragraph::new(lines), inner);
}

/// The INTERACTIVE tool-call inspector (`/verbose` `/tools` `/trace`; S7, tui_v3
/// `_verbose_view`). A TWO-PANE layout over [`AppState::tool_audit`]: a top LIST
/// of every captured tool call (`{marker} t{id} {name}  {status}`, the selected
/// row marked with `▌` + its name bold, each colored by the chip status) and a
/// bottom DETAIL pane showing the SELECTED record's current field — Result / Args
/// / Raw — under a heading, scrolled by `state.detail_scroll`. The key handler
/// (`input::views`) drives `state` (↑/↓ select, PgUp/PgDn scroll, Enter cycle
/// field, `c` copy, `e` export). No hardcoded colors.
pub(crate) fn render_verbose(
    frame: &mut Frame,
    area: Rect,
    app: &AppState,
    state: &crate::app::VerboseState,
    theme: &Theme,
    lang: Lang,
) {
    use ratatui::layout::{Constraint, Direction, Layout};

    frame.render_widget(Clear, area);
    let block = titled_block(i18n::t(lang, "verbose.title"), theme);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if app.tool_audit.is_empty() {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                i18n::t(lang, "verbose.empty"),
                Style::default().fg(theme.color(Token::Dim)),
            ))),
            inner,
        );
        return;
    }

    // The selection + field are clamped here too (defensive) so a stale state can
    // never index past the data even if a record was just drained.
    let sel = state.selected.min(app.tool_audit.len() - 1);
    let field_key = state.field.label_key();

    // Hint line: the active field + the key legend (tui_v3 `verbose.hint`).
    let hint = Line::from(Span::styled(
        format!("{} {} · {}", i18n::t(lang, "verbose.field"), i18n::t(lang, field_key), i18n::t(lang, "verbose.hint")),
        Style::default().fg(theme.color(Token::Dim)),
    ));

    // Split: hint row, then a LIST pane (≈⅓, ≥3 rows) over a DETAIL pane (the rest).
    let avail = inner.height.saturating_sub(1);
    let list_h = (app.tool_audit.len() as u16)
        .min((avail / 3).max(3))
        .min(avail.saturating_sub(1).max(1));
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(list_h),
            Constraint::Min(1),
        ])
        .split(inner);
    frame.render_widget(Paragraph::new(hint), chunks[0]);

    // -- LIST pane: window the rows around the selection (tui_v3 `lo`). --------
    let list_rows = chunks[1].height as usize;
    let lo = sel
        .saturating_sub(list_rows / 2)
        .min(app.tool_audit.len().saturating_sub(list_rows));
    let mut list_lines: Vec<Line> = Vec::new();
    for (off, rec) in app.tool_audit[lo..].iter().take(list_rows).enumerate() {
        let idx = lo + off;
        let selected = idx == sel;
        let marker = if selected { "▌" } else { " " };
        let status_tok = rec.status.token();
        let (badge, _) = rec.status.badge();
        let name_style = {
            let s = Style::default().fg(theme.color(Token::Text));
            if selected { s.add_modifier(Modifier::BOLD) } else { s }
        };
        list_lines.push(Line::from(vec![
            Span::styled(format!("{marker} "), Style::default().fg(theme.color(Token::Claude))),
            Span::styled(format!("t{:<3} ", rec.id), Style::default().fg(theme.color(Token::Dim))),
            Span::styled(format!("{:<22}", clip_to(&rec.name, 22)), name_style),
            Span::styled(format!(" {badge}"), Style::default().fg(theme.color(status_tok))),
        ]));
    }
    frame.render_widget(Paragraph::new(list_lines), chunks[1]);

    // -- DETAIL pane: the selected record's current field, under a heading. ----
    let detail = chunks[2];
    let detail_w = detail.width.max(1) as usize;
    let rec = &app.tool_audit[sel];
    let field_text = match state.field {
        crate::app::VerboseField::Result => &rec.result,
        crate::app::VerboseField::Args => &rec.args,
        crate::app::VerboseField::Raw => &rec.raw,
    };
    let mut detail_lines: Vec<Line> = vec![Line::from(Span::styled(
        format!("{} t{}", i18n::t(lang, field_key), rec.id),
        Style::default().fg(theme.color(Token::Claude)).add_modifier(Modifier::BOLD),
    ))];
    if field_text.trim().is_empty() {
        detail_lines.push(Line::from(Span::styled(
            i18n::t(lang, "verbose.field.empty"),
            Style::default().fg(theme.color(Token::Dim)),
        )));
    } else {
        for raw_line in field_text.split('\n') {
            detail_lines.push(Line::from(Span::styled(
                clip_to(raw_line, detail_w),
                Style::default().fg(theme.color(Token::Text)),
            )));
        }
    }
    // Clamp the scroll to the body height so PgDn can't run off the end (the body
    // is `detail_lines.len() - 1` after the heading row).
    let body = detail_lines.len().saturating_sub(1);
    let max_scroll = (body as u16).saturating_sub(detail.height.saturating_sub(1).max(1));
    let scroll = state.detail_scroll.min(max_scroll);
    frame.render_widget(Paragraph::new(detail_lines).scroll((scroll, 0)), detail);
}

/// The `/btw` side-answer card: `querying…` then the answer, above the composer.
/// `Esc` dismisses (no history pollution — the dispatcher discards it).
pub(crate) fn render_btw(
    frame: &mut Frame,
    area: Rect,
    question: &str,
    answer: Option<&str>,
    theme: &Theme,
    lang: Lang,
) {
    let w = (area.width.saturating_sub(8)).clamp(30, 90);
    let h = 8u16.min(area.height.saturating_sub(2)).max(5);
    let card = super::centered(area, w, h);
    frame.render_widget(Clear, card);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.color(Token::Suggestion)))
        .title(Span::styled(
            " /btw ",
            Style::default().fg(theme.color(Token::Suggestion)).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(card);
    frame.render_widget(block, card);

    let inner_w = inner.width as usize;
    let mut lines: Vec<Line> = vec![Line::from(Span::styled(
        clip_to(question, inner_w.saturating_sub(2)),
        Style::default().fg(theme.color(Token::Text)).add_modifier(Modifier::BOLD),
    ))];
    lines.push(Line::from(""));
    match answer {
        None => lines.push(Line::from(Span::styled(
            i18n::t(lang, "btw.querying"),
            Style::default().fg(theme.color(Token::Dim)),
        ))),
        Some(a) => {
            for l in a.lines().take(inner.height.saturating_sub(3) as usize) {
                lines.push(Line::from(Span::styled(
                    clip_to(l, inner_w.saturating_sub(1)),
                    Style::default().fg(theme.color(Token::Text)),
                )));
            }
        }
    }
    lines.push(Line::from(Span::styled(
        i18n::t(lang, "btw.dismiss"),
        Style::default().fg(theme.color(Token::Dim)),
    )));
    frame.render_widget(Paragraph::new(lines), inner);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    /// SLICE S7: the `/keybindings` overlay documents the intentional v3→v4 chord
    /// remaps so migrating users aren't surprised — the fold chord (moved off
    /// `Ctrl+O`), the copy-reply key, the stash chord (moved to `Ctrl+G`), the
    /// dashboard + branch keys, and the mouse-mode toggle + native-select hint are
    /// all present. LIVE styled render of the overlay.
    #[test]
    fn keybindings_overlay_lists_v3_to_v4_remaps() {
        let theme = Theme::default_theme();
        let (w, h) = (96u16, 32u16);
        let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
        term.draw(|f| render_keybindings(f, Rect::new(0, 0, w, h), &theme, Lang::En))
            .unwrap();
        let buf = term.backend().buffer();
        let screen: String = (0..h as usize)
            .map(|y| (0..w as usize).map(|x| buf.content()[y * w as usize + x].symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n");

        for chord in ["Ctrl+O", "Ctrl+Shift+O", "Ctrl+G", "Ctrl+S", "Ctrl+B", "Ctrl+Shift+M"] {
            assert!(screen.contains(chord), "keybindings overlay must list `{chord}`:\n{screen}");
        }
        // The native-select hint (the round-5 mouse model) is shown.
        assert!(
            screen.contains("select") || screen.contains("native"),
            "the native-select mouse hint must appear:\n{screen}"
        );
    }
}
