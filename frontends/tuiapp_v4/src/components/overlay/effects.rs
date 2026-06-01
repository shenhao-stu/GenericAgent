//! overlay/effects.rs — the `/effects demo` splash (§9): a centered panel showing
//! every effect at once (fire band + snow + lightning + sparkle). The engine
//! (already ticking) drives the animation; this paints the current frame and
//! honors the capability gate (a mono / NO_COLOR terminal gets a plain legend).

use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Paragraph};
use ratatui::Frame;

use crate::app::AppState;
use crate::components::effects_paint;
use crate::theme::{Theme, Token};

use super::{centered, titled_block};

/// The `/effects demo` splash: ambient field on top, an indicator legend below.
pub(crate) fn render_effects_demo(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
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

    let parts = ratatui::layout::Layout::default()
        .direction(ratatui::layout::Direction::Vertical)
        .constraints([
            ratatui::layout::Constraint::Min(3),
            ratatui::layout::Constraint::Length(2),
        ])
        .split(inner);
    effects_paint::draw_ambient(frame, app, parts[0]);
    render_demo_legend(frame, parts[1], app, theme);
}

/// A small legend under the demo: the transient indicators (lightning / sparkle).
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
