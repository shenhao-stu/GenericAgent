//! overlay/effort.rs — the `/effort` slider card (redesign_cc.md §3): a
//! `Faster ←——▲——→ Smarter` track over the `low medium high xhigh max` stops with
//! a `▲` marker on the chosen level. PURE paint over the [`EffortSlider`] model.

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Paragraph};
use ratatui::Frame;

use crate::app::effort::EffortSlider;
use crate::theme::{Theme, Token};

use super::{centered, titled_block};

/// Draw the `/effort` slider. The marked stop is highlighted; the currently-applied
/// stop carries a `●`. Footer: `←/→ to adjust · Enter to confirm · Esc to cancel`.
pub(crate) fn render_effort_slider(frame: &mut Frame, area: Rect, slider: &EffortSlider, theme: &Theme) {
    use crate::app::effort::ReasoningEffort;

    let levels = ReasoningEffort::LEVELS;
    // Each stop gets a fixed-width CELL (widest label + 2 gap) so the label row, the
    // `●` applied-marker, and the `▲` track marker share ONE column grid. The label
    // row is left-padded by the prefix width so every cell lines up under its label.
    let cell = levels.iter().map(|l| l.label().len()).max().unwrap_or(6) + 2;
    let track_w = cell * levels.len();
    const PREFIX: &str = "Faster ←"; // 8 cells; the label grid is offset by this.
    const SUFFIX: &str = "→ Smarter";
    let prefix_w = unicode_width::UnicodeWidthStr::width(PREFIX);
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

    // Row 1: the labels, left-padded by the prefix width. The marked stop is
    // accent+bold; the applied stop is green with a `●` at the cell's first column.
    let mut label_spans: Vec<Span> = vec![Span::raw(" ".repeat(prefix_w))];
    for (i, lvl) in levels.iter().enumerate() {
        let is_marked = i == marker;
        let is_current = *lvl == slider.current;
        let lab = lvl.label();
        let pad = cell.saturating_sub(lab.len());
        let left = pad / 2;
        let right = pad - left;
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
