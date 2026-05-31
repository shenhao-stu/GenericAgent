//! components/overlay.rs — the MODAL overlay renderer (§3 overlay stack / §7).
//!
//! Draws whichever [`Overlay`] is active ON TOP of the cockpit/dashboard: the
//! reusable list picker (`/llm` `/theme` `/emoji` `/language` `/export` `/rewind`
//! `/continue` `/scheduler`), the unified ask_user card (single/multi/numeric),
//! `/help`, `/status`, `/cost`, `/verbose`, and the `/btw` answer card.
//!
//! Layout discipline: full-screen overlays (help/status/cost/verbose) take the
//! whole area with a bordered block; compact overlays (picker / ask / btw) draw a
//! CENTERED bordered card sized to the content. All the load-bearing list/selection
//! logic lives in `components::picker` (pure + tested); this module only PAINTS.
//! No hardcoded colors — every style goes through a theme [`Token`].

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::app::effort::EffortSlider;
use crate::app::{AppState, Overlay};
use crate::commands::registry::{CommandKind, COMMANDS};
use crate::components::picker::{AskMode, AskUserPicker, Picker};
use crate::i18n::{self, Lang};
use crate::theme::{Theme, Token};

use super::clip_to;

/// Draw the active overlay. `area` is the full frame; the overlay sizes its own
/// region within it.
pub fn render(frame: &mut Frame, area: Rect, ov: &Overlay, app: &AppState, theme: &Theme, now_ms: u64) {
    // A full-screen overlay (help / status / cost / verbose) clears the whole
    // area first so the cockpit/dashboard underneath is fully covered; compact
    // cards clear only their own centered rect (done per-renderer).
    if ov.is_fullscreen() {
        frame.render_widget(Clear, area);
    }
    let lang = app.lang;
    match ov {
        Overlay::Picker { picker, .. } => render_picker(frame, area, picker, theme, lang),
        Overlay::AskUser(ask) => render_ask_user(frame, area, ask, theme, lang),
        Overlay::Help => render_help(frame, area, theme, lang),
        Overlay::Status => render_status(frame, area, app, theme, now_ms),
        Overlay::Cost => render_cost(frame, area, app, theme),
        Overlay::Verbose => render_verbose(frame, area, app, theme),
        Overlay::Btw { question, answer, .. } => {
            render_btw(frame, area, question, answer.as_deref(), theme, lang)
        }
        Overlay::Scheduler(sched) => super::scheduler::render(frame, area, sched, theme, lang),
        Overlay::Continue(picker) => super::continue_picker::render(frame, area, picker, theme, lang),
        Overlay::Effects => render_effects_demo(frame, area, app, theme),
        Overlay::EffortSlider(slider) => render_effort_slider(frame, area, slider, theme),
    }
}

// ---- /effort slider --------------------------------------------------------

/// The `/effort` slider overlay (redesign_cc.md §3): a `Faster ←——▲——→ Smarter`
/// horizontal track over the `low medium high xhigh max` stops with a `▲` marker on
/// the chosen level. The marked stop is highlighted; the currently-applied stop
/// carries a `●`. Footer: `←/→ to adjust · Enter to confirm · Esc to cancel`. PURE
/// paint over the [`EffortSlider`] model.
fn render_effort_slider(frame: &mut Frame, area: Rect, slider: &EffortSlider, theme: &Theme) {
    use crate::app::effort::ReasoningEffort;

    let levels = ReasoningEffort::LEVELS;
    // Each stop gets a fixed-width CELL (widest label + 2 gap) so the label row, the
    // `●` applied-marker, and the `▲` track marker share ONE column grid. The track
    // is prefixed with `Faster ←` and suffixed with `→ Smarter`; the LABEL row is
    // left-padded by the same prefix width so every cell lines up under its label.
    let cell = levels.iter().map(|l| l.label().len()).max().unwrap_or(6) + 2;
    let track_w = cell * levels.len();
    const PREFIX: &str = "Faster ←"; // 8 cells; the label grid is offset by this.
    const SUFFIX: &str = "→ Smarter";
    let prefix_w = unicode_width::UnicodeWidthStr::width(PREFIX);
    // Card width fits the prefix + track + suffix; bounded to the area.
    let inner_w = prefix_w + track_w + unicode_width::UnicodeWidthStr::width(SUFFIX) + 2;
    let w = (inner_w as u16 + 4).min(area.width.saturating_sub(2)).max(40);
    let h = 9u16.min(area.height.saturating_sub(2)).max(7);
    let card = centered(area, w, h);
    frame.render_widget(Clear, card);
    let block = titled_block("Reasoning effort · /effort", theme);
    let inner = block.inner(card);
    frame.render_widget(block, card);

    let marker = slider.marker.min(levels.len() - 1);
    let claude = theme.color(Token::Claude);
    let suggestion = theme.color(Token::Suggestion);
    let dim = theme.color(Token::Dim);
    let text = theme.color(Token::Text);
    let success = theme.color(Token::Success);

    // The center column (within a cell) each label/marker aligns to.
    let center_in_cell = |lab: &str| -> usize {
        let pad = cell.saturating_sub(lab.len());
        let left = pad / 2;
        left + lab.len().saturating_sub(1) / 2
    };

    // Row 1: the labels, left-padded by the prefix width so each cell sits over its
    // track cell. The marked stop is accent+bold; the applied stop is green with a
    // `●` (placed at the cell's first column without changing the cell width).
    let mut label_spans: Vec<Span> = vec![Span::raw(" ".repeat(prefix_w))];
    for (i, lvl) in levels.iter().enumerate() {
        let is_marked = i == marker;
        let is_current = *lvl == slider.current;
        let lab = lvl.label();
        let pad = cell.saturating_sub(lab.len());
        let left = pad / 2;
        let right = pad - left;
        // The `●` applied-marker takes the leftmost pad cell (keeps the cell width).
        let lead = if is_current && left > 0 {
            format!("●{}", " ".repeat(left - 1))
        } else {
            " ".repeat(left)
        };
        let style = if is_marked {
            Style::default().fg(claude).add_modifier(Modifier::BOLD)
        } else if is_current {
            Style::default().fg(success)
        } else {
            Style::default().fg(dim)
        };
        label_spans.push(Span::styled(format!("{lead}{lab}{}", " ".repeat(right)), style));
    }

    // Row 2: the `Faster ←——▲——→ Smarter` track. The `▲` sits at the marked stop's
    // label-center column so it lines up with the label above it.
    let mut track = String::new();
    let marked_center = center_in_cell(levels[marker].label());
    for (i, _lvl) in levels.iter().enumerate() {
        for c in 0..cell {
            if i == marker && c == marked_center {
                track.push('▲');
            } else {
                track.push('—');
            }
        }
    }

    // Compose the lines.
    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(label_spans));
    lines.push(Line::from(vec![
        Span::styled(PREFIX, Style::default().fg(dim)),
        Span::styled(track, Style::default().fg(suggestion)),
        Span::styled(SUFFIX, Style::default().fg(dim)),
    ]));
    lines.push(Line::from(""));
    // The chosen value (and the backend value if it differs — max→xhigh).
    let chosen = ReasoningEffort::from_index(marker);
    let val_line = if chosen.label() == chosen.backend_value() {
        format!("→ {}", chosen.label())
    } else {
        format!("→ {} (backend: {})", chosen.label(), chosen.backend_value())
    };
    lines.push(Line::from(Span::styled(
        val_line,
        Style::default().fg(text).add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "←/→ to adjust · Enter to confirm · Esc to cancel",
        Style::default().fg(dim),
    )));
    frame.render_widget(Paragraph::new(lines), inner);
}

// ---- /effects demo ---------------------------------------------------------

/// The `/effects demo` splash: a centered panel showing every effect at once (§9). The
/// engine (already ticking) drives the animation; this just paints the current frame.
/// Honors the capability gate (a mono / NO_COLOR terminal gets a plain legend, no fills).
fn render_effects_demo(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
    let w = (area.width.saturating_sub(6)).clamp(24, 100);
    let h = (area.height.saturating_sub(4)).clamp(8, 30);
    let card = centered(area, w, h);
    frame.render_widget(Clear, card);
    let secs = app.effects.demo_timer.ceil() as u32;
    let block = titled_block(&format!("/effects demo — reverts in {secs}s (any key closes)"), theme);
    let inner = block.inner(card);
    frame.render_widget(block, card);
    if inner.width < 6 || inner.height < 6 {
        return;
    }

    // Split: ambient field on top, an indicator legend at the bottom.
    let parts = ratatui::layout::Layout::default()
        .direction(ratatui::layout::Direction::Vertical)
        .constraints([
            ratatui::layout::Constraint::Min(3),
            ratatui::layout::Constraint::Length(2),
        ])
        .split(inner);
    super::effects_paint::draw_ambient(frame, app, parts[0]);
    render_demo_legend(frame, parts[1], app, theme);
}

/// A small legend under the demo: the transient indicators (lightning / sparkle), via
/// theme tokens.
fn render_demo_legend(frame: &mut Frame, area: Rect, app: &AppState, _theme: &Theme) {
    let pal = crate::effects::EffectPalette::from_theme(&app.theme);
    let dim = app.theme.color(Token::Dim);
    let mut spans: Vec<Span> = vec![Span::styled("lightning ", Style::default().fg(dim))];
    if app.effects.lightning.active() {
        spans.push(Span::styled("╲╱│ ", Style::default().fg(pal.lightning_bolt)));
    } else {
        spans.push(Span::styled("·   ", Style::default().fg(dim)));
    }
    spans.push(Span::styled("  sparkle ", Style::default().fg(dim)));
    if app.effects.sparkle.active() {
        for s in app.effects.sparkle.sparks().iter().take(10) {
            spans.push(Span::styled(s.glyph().to_string(), Style::default().fg(pal.sparkle)));
        }
    } else {
        spans.push(Span::styled("·", Style::default().fg(dim)));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// A centered card `Rect` of `w`×`h` (clamped to `area`), with `Clear` so it
/// covers the view underneath. PURE-ish geometry.
fn centered(area: Rect, w: u16, h: u16) -> Rect {
    let w = w.min(area.width);
    let h = h.min(area.height);
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(h)) / 2;
    Rect { x, y, width: w, height: h }
}

/// Bordered block with a title in the Claude accent.
fn titled_block(title: &str, theme: &Theme) -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.color(Token::Claude)))
        .title(Span::styled(
            format!(" {title} "),
            Style::default().fg(theme.color(Token::Claude)).add_modifier(Modifier::BOLD),
        ))
}

// ---- list picker -----------------------------------------------------------

/// The reusable list picker card (`/llm` etc). Rows show `●` for the current item,
/// `[x]`/`[ ]` checkboxes in multi-select mode, and a `❯` cursor on the selection.
fn render_picker(frame: &mut Frame, area: Rect, picker: &Picker, theme: &Theme, lang: Lang) {
    let multi = picker.kind.multi();
    // Card width: fit the widest row, bounded; height = visible rows + chrome.
    let max_label = picker
        .items
        .iter()
        .map(|i| {
            unicode_width::UnicodeWidthStr::width(i.label.as_str())
                + unicode_width::UnicodeWidthStr::width(i.detail.as_str())
                + 4
        })
        .max()
        .unwrap_or(20);
    let title = picker.kind.title(lang);
    let title_w = unicode_width::UnicodeWidthStr::width(title) + 4;
    let inner_w = max_label.max(title_w).max(28) as u16;
    let (start, slice) = picker.window();
    let w = (inner_w + 6).min(area.width.saturating_sub(2)).max(20);
    let h = (slice.len() as u16 + 4).min(area.height.saturating_sub(2)).max(5);
    let card = centered(area, w, h);
    frame.render_widget(Clear, card);
    let block = titled_block(title, theme);
    let inner = block.inner(card);
    frame.render_widget(block, card);

    let mut lines: Vec<Line> = Vec::with_capacity(slice.len() + 1);
    if picker.is_empty() {
        lines.push(Line::from(Span::styled(
            i18n::t(lang, "picker.empty"),
            Style::default().fg(theme.color(Token::Dim)),
        )));
    }
    for (i, item) in slice.iter().enumerate() {
        let idx = start + i;
        let selected = idx == picker.sel;
        let mut spans: Vec<Span> = Vec::new();
        // Cursor.
        spans.push(Span::styled(
            if selected { "❯ " } else { "  " },
            Style::default().fg(theme.color(Token::Suggestion)),
        ));
        // Multi-select checkbox.
        if multi {
            let box_ = if item.checked { "[x] " } else { "[ ] " };
            let tok = if item.checked { Token::Success } else { Token::Dim };
            spans.push(Span::styled(box_, Style::default().fg(theme.color(tok))));
        } else if item.current {
            // Current marker (●) for single-select pickers (`/llm` `/theme`).
            spans.push(Span::styled("● ", Style::default().fg(theme.color(Token::Success))));
        } else {
            spans.push(Span::raw("  "));
        }
        let label_tok = if selected { Token::Suggestion } else { Token::Text };
        let label_mod = if selected || item.current {
            Modifier::BOLD
        } else {
            Modifier::empty()
        };
        spans.push(Span::styled(
            item.label.clone(),
            Style::default().fg(theme.color(label_tok)).add_modifier(label_mod),
        ));
        if !item.detail.is_empty() {
            spans.push(Span::styled(
                format!("   {}", item.detail),
                Style::default().fg(theme.color(Token::Dim)),
            ));
        }
        lines.push(Line::from(spans));
    }
    // Hint row.
    lines.push(Line::from(Span::styled(
        format!("  {}", picker.kind.hint(lang)),
        Style::default().fg(theme.color(Token::Dim)),
    )));
    frame.render_widget(Paragraph::new(lines), inner);
}

// ---- ask_user card ---------------------------------------------------------

/// The unified ask_user card (§7): question + candidate rows + an inline free-text
/// row, with multi-select checkboxes / numeric ordinals as the mode dictates.
fn render_ask_user(frame: &mut Frame, area: Rect, ask: &AskUserPicker, theme: &Theme, lang: Lang) {
    let title = match ask.mode {
        AskMode::Single => i18n::t(lang, "ask.title.single"),
        AskMode::Multi => i18n::t(lang, "ask.title.multi"),
        AskMode::Numeric => i18n::t(lang, "ask.title.numeric"),
    };
    let w = (area.width.saturating_sub(8)).clamp(30, 96);
    // Height: question (wrapped, ~3) + candidates + free-text + hint + chrome.
    let body_rows = ask.candidates.len() as u16 + if ask.free_text { 1 } else { 0 } + 5;
    let h = body_rows.min(area.height.saturating_sub(2)).max(7);
    let card = centered(area, w, h);
    frame.render_widget(Clear, card);
    let block = titled_block(title, theme);
    let inner = block.inner(card);
    frame.render_widget(block, card);

    let inner_w = inner.width as usize;
    let mut lines: Vec<Line> = Vec::new();
    // The question (clipped to fit the card width).
    lines.push(Line::from(Span::styled(
        clip_to(&ask.question, inner_w.saturating_sub(2)),
        Style::default().fg(theme.color(Token::Text)).add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(""));

    for (i, cand) in ask.candidates.iter().enumerate() {
        let selected = ask.sel == i;
        let mut spans: Vec<Span> = Vec::new();
        spans.push(Span::styled(
            if selected { "❯ " } else { "  " },
            Style::default().fg(theme.color(Token::Suggestion)),
        ));
        match ask.mode {
            AskMode::Multi => {
                let on = ask.checked.get(i).copied().unwrap_or(false);
                let box_ = if on { "[x] " } else { "[ ] " };
                let tok = if on { Token::Success } else { Token::Dim };
                spans.push(Span::styled(box_, Style::default().fg(theme.color(tok))));
            }
            AskMode::Numeric => {
                spans.push(Span::styled(
                    format!("{}. ", i + 1),
                    Style::default().fg(theme.color(Token::Suggestion)),
                ));
            }
            AskMode::Single => {}
        }
        let tok = if selected { Token::Suggestion } else { Token::Text };
        spans.push(Span::styled(
            clip_to(cand, inner_w.saturating_sub(6)),
            Style::default().fg(theme.color(tok)).add_modifier(if selected {
                Modifier::BOLD
            } else {
                Modifier::empty()
            }),
        ));
        lines.push(Line::from(spans));
    }

    // The inline free-text / numeric input row.
    if ask.free_text || ask.mode == AskMode::Numeric {
        let on_input = ask.on_free_text_row() || ask.mode == AskMode::Numeric;
        let label = if ask.mode == AskMode::Numeric {
            i18n::t(lang, "ask.input.number")
        } else {
            "› "
        };
        let mut spans: Vec<Span> = vec![
            Span::styled(
                if on_input { "❯ " } else { "  " },
                Style::default().fg(theme.color(Token::Suggestion)),
            ),
            Span::styled(label, Style::default().fg(theme.color(Token::Dim))),
        ];
        if ask.input.is_empty() {
            spans.push(Span::styled(
                i18n::t(lang, "ask.input.placeholder"),
                Style::default().fg(theme.color(Token::Dim)),
            ));
        } else {
            spans.push(Span::styled(
                ask.input.clone(),
                Style::default().fg(theme.color(Token::Text)),
            ));
        }
        if on_input {
            spans.push(Span::styled(" ", Style::default().add_modifier(Modifier::REVERSED)));
        }
        lines.push(Line::from(spans));
    }

    lines.push(Line::from(""));
    let hint = match ask.mode {
        AskMode::Multi => i18n::t(lang, "ask.hint.multi"),
        AskMode::Numeric => i18n::t(lang, "ask.hint.numeric"),
        AskMode::Single => i18n::t(lang, "ask.hint.single"),
    };
    lines.push(Line::from(Span::styled(
        hint,
        Style::default().fg(theme.color(Token::Dim)),
    )));
    frame.render_widget(Paragraph::new(lines), inner);
}

// ---- /help -----------------------------------------------------------------

/// The full command-list overlay (`/help`), grouped by [`CommandKind`].
fn render_help(frame: &mut Frame, area: Rect, theme: &Theme, lang: Lang) {
    frame.render_widget(Clear, area);
    let block = titled_block(i18n::t(lang, "help.title"), theme);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let mut lines: Vec<Line> = Vec::new();
    let group = |lines: &mut Vec<Line>, title: &str, kind: CommandKind| {
        lines.push(Line::from(Span::styled(
            title.to_string(),
            Style::default().fg(theme.color(Token::Claude)).add_modifier(Modifier::BOLD),
        )));
        for c in COMMANDS.iter().filter(|c| c.kind == kind) {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  /{:<12}", c.name),
                    Style::default().fg(theme.color(Token::Suggestion)),
                ),
                Span::styled(c.desc.to_string(), Style::default().fg(theme.color(Token::Dim))),
            ]));
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

/// A tiny helper so a tall help list scrolls to show the top (we don't track a
/// help scroll cursor; the list is short enough to fit a normal terminal). Returns
/// 0 (top). PURE — kept as a seam for a future scroll cursor.
fn help_scroll(_total: usize, _height: u16) -> u16 {
    0
}

fn lines_len_hint() -> usize {
    COMMANDS.len() + 8
}

// ---- /status ---------------------------------------------------------------

/// The status snapshot overlay (`/status` `/sessions`): model / connection /
/// session counts / context / cwd / git.
fn render_status(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme, now_ms: u64) {
    let w = (area.width.saturating_sub(8)).clamp(40, 80);
    let h = 14u16.min(area.height.saturating_sub(2)).max(8);
    let card = centered(area, w, h);
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

// ---- /cost -----------------------------------------------------------------

/// The token-usage report (`/cost`): input / output / cache / total / context% /
/// cost — rendered from the pure [`crate::app::CostBreakdown::report_lines`].
fn render_cost(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
    let w = 48u16.min(area.width.saturating_sub(2)).max(30);
    // Truncate the model to its primary segment (never the MixinSession pipe-list).
    let model = super::truncate_model(app.model.as_deref().unwrap_or("—"), super::MODEL_LABEL_CAP);
    let report = app.cost.report_lines(&model);
    let h = (report.len() as u16 + 3).min(area.height.saturating_sub(2)).max(8);
    let card = centered(area, w, h);
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

// ---- /verbose --------------------------------------------------------------

/// The full-screen tool-call audit (`/verbose` `/tools` `/trace`): the tail of the
/// session's tool-call lines.
fn render_verbose(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
    frame.render_widget(Clear, area);
    let block = titled_block("Tool-call audit · /verbose", theme);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let cap = inner.height.saturating_sub(1) as usize;
    let mut lines: Vec<Line> = Vec::new();
    if app.tool_audit.is_empty() {
        lines.push(Line::from(Span::styled(
            "  no tool calls yet this session.",
            Style::default().fg(theme.color(Token::Dim)),
        )));
    } else {
        let start = app.tool_audit.len().saturating_sub(cap);
        for (i, line) in app.tool_audit[start..].iter().enumerate() {
            let n = start + i + 1;
            lines.push(Line::from(vec![
                Span::styled(format!("{n:>4} "), Style::default().fg(theme.color(Token::Dim))),
                Span::styled(line.clone(), Style::default().fg(theme.color(Token::Text))),
            ]));
        }
    }
    frame.render_widget(Paragraph::new(lines), inner);
}

// ---- /btw ------------------------------------------------------------------

/// The `/btw` side-answer card: `querying…` then the answer, above the composer.
/// `Esc` dismisses (no history pollution — the dispatcher discards it).
fn render_btw(
    frame: &mut Frame,
    area: Rect,
    question: &str,
    answer: Option<&str>,
    theme: &Theme,
    lang: Lang,
) {
    let w = (area.width.saturating_sub(8)).clamp(30, 90);
    let h = 8u16.min(area.height.saturating_sub(2)).max(5);
    let card = centered(area, w, h);
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
    use crate::components::picker::{PickItem, PickerKind};
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    /// Render each overlay to an in-memory backend and confirm its headline chrome
    /// paints (a render-level guard that the modal surface draws; the load-bearing
    /// selection logic is tested in `components::picker`).
    #[test]
    fn overlays_render_their_chrome() {
        let theme = Theme::ga_default();
        let render_with = |ov: Overlay, needle: &str| {
            let mut app = AppState::new();
            app.cost.input = 100;
            app.cost.output = 250;
            app.push_tool_audit("Read(src/main.rs)".into());
            app.overlay = Some(ov);
            let backend = TestBackend::new(80, 24);
            let mut terminal = Terminal::new(backend).unwrap();
            terminal
                .draw(|f| {
                    let area = f.area();
                    render(f, area, app.overlay.as_ref().unwrap(), &app, &theme, 1000);
                })
                .unwrap();
            let buf = terminal.backend().buffer();
            let text: String = buf.content().iter().map(|c| c.symbol()).collect();
            assert!(text.contains(needle), "overlay must paint {needle:?}; got chrome only");
        };

        render_with(Overlay::Help, "Commands");
        render_with(Overlay::Status, "model");
        render_with(Overlay::Cost, "Token usage");
        render_with(Overlay::Verbose, "audit");
        render_with(
            Overlay::Picker {
                picker: Picker::new(
                    PickerKind::Llm,
                    vec![PickItem::new(0, "OpenAI/gpt").current(true)],
                ),
                theme_backup: None,
            },
            "Switch model",
        );
        render_with(
            Overlay::AskUser(AskUserPicker::new("a1", "pick one?", vec!["yes".into(), "no".into()], true)),
            "pick one?",
        );
        render_with(
            Overlay::Btw { ask_id: "b1".into(), question: "what is 2+2?".into(), answer: None },
            "querying",
        );
        // The /effort slider paints its title, the stops, the Faster/Smarter track,
        // the `▲` marker, and the footer (redesign_cc.md §3).
        render_with(
            Overlay::EffortSlider(EffortSlider::new(crate::app::effort::ReasoningEffort::High)),
            "Reasoning effort",
        );
        render_with(
            Overlay::EffortSlider(EffortSlider::new(crate::app::effort::ReasoningEffort::High)),
            "Smarter",
        );
        render_with(
            Overlay::EffortSlider(EffortSlider::new(crate::app::effort::ReasoningEffort::High)),
            "▲",
        );
        render_with(
            Overlay::EffortSlider(EffortSlider::new(crate::app::effort::ReasoningEffort::High)),
            "Enter to confirm",
        );
    }
}
