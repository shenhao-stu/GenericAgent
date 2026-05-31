//! components/effects_paint.rs — the ratatui PAINT side of the effects engine (§9).
//!
//! The engine in `crate::effects` is pure: it advances bounded buffers (fire heat, snow
//! particles, …) by `dt`. This module projects those buffers onto a ratatui [`Frame`],
//! honoring the capability gate (truecolor half-blocks vs fg-only glyphs under tmux, and
//! a no-op at mono / NO_COLOR). It is presentation-only and intentionally separate from
//! the pure steppers so the steppers stay TTY-free and testable.
//!
//! Used by BOTH the `/effects demo` splash overlay and the cockpit's `full`-mode ambient
//! layer, so the look is identical in both.

use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::app::AppState;
use crate::effects::fire::Fire;
use crate::effects::EffectPalette;

/// The blank braille cell (`U+2800`), rendered as a space so snow doesn't fill the field.
const BRAILLE_BLANK: char = '\u{2800}';

/// Draw the full ambient field over `area`, honoring the capability gate (a no-op at
/// mono / NO_COLOR or for a too-small area). Layer order (back → front, the recon's
/// compositing stack): snow, then the fire band, then the transient one-shots
/// (lightning bolt on failure, sparkle burst on success) painted on top.
pub fn draw_ambient(frame: &mut Frame, app: &AppState, area: Rect) {
    if !app.effects.caps.enabled() || area.height < 2 || area.width < 2 {
        return;
    }
    draw_snow(frame, app, area);
    draw_fire(frame, app, area);
    draw_lightning(frame, app, area);
    draw_sparkle(frame, app, area);
}

/// Paint the in-flight lightning bolt (failure flash) as a polyline of box-drawing glyphs
/// over `area`, with a wider dim glow underneath (drawn first) in the glow token. The bolt
/// path + per-segment glyph come from the pure [`crate::effects::lightning::Lightning`].
fn draw_lightning(frame: &mut Frame, app: &AppState, area: Rect) {
    let l = &app.effects.lightning;
    if !l.active() {
        return;
    }
    let pal = EffectPalette::from_theme(&app.theme);
    let path = l.path();
    // Glow: a dim copy one column to each side, drawn first so the core sits on top.
    for (i, &(x, y)) in path.iter().enumerate() {
        if x as usize >= area.width as usize || y as usize >= area.height as usize {
            continue;
        }
        let glyph = l.glyph_at(i);
        for dx in [-1i32, 1] {
            let gx = x as i32 + dx;
            if gx < 0 || gx as usize >= area.width as usize {
                continue;
            }
            let cell = Rect { x: area.x + gx as u16, y: area.y + y, width: 1, height: 1 };
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    glyph.to_string(),
                    Style::default().fg(pal.lightning_glow),
                ))),
                cell,
            );
        }
    }
    // Core: the bright bolt.
    for (i, &(x, y)) in path.iter().enumerate() {
        if x as usize >= area.width as usize || y as usize >= area.height as usize {
            continue;
        }
        let cell = Rect { x: area.x + x, y: area.y + y, width: 1, height: 1 };
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                l.glyph_at(i).to_string(),
                Style::default().fg(pal.lightning_bolt),
            ))),
            cell,
        );
    }
}

/// Paint the in-flight sparkle burst (success) at each spark's `(dx, dy)` offset from the
/// field center, using its TTL-driven glyph. Bounded; a no-op when no burst is active.
fn draw_sparkle(frame: &mut Frame, app: &AppState, area: Rect) {
    let s = &app.effects.sparkle;
    if !s.active() {
        return;
    }
    let pal = EffectPalette::from_theme(&app.theme);
    let cx = (area.width / 2) as i32;
    let cy = (area.height / 2) as i32;
    for spark in s.sparks() {
        let x = cx + spark.dx as i32;
        let y = cy + spark.dy as i32;
        if x < 0 || y < 0 || x as usize >= area.width as usize || y as usize >= area.height as usize {
            continue;
        }
        let cell = Rect { x: area.x + x as u16, y: area.y + y as u16, width: 1, height: 1 };
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                spark.glyph().to_string(),
                Style::default().fg(pal.sparkle),
            ))),
            cell,
        );
    }
}

/// Render the snow braille field across `area` (fg-only — tmux-safe).
fn draw_snow(frame: &mut Frame, app: &AppState, area: Rect) {
    let pal = EffectPalette::from_theme(&app.theme);
    let grid = app.effects.snow.render_grid();
    let gw = app.effects.snow.w;
    let gh = app.effects.snow.h;
    let rows = (area.height as usize).min(gh);
    let cols = (area.width as usize).min(gw);
    for ry in 0..rows {
        let mut spans: Vec<Span> = Vec::with_capacity(cols);
        for rx in 0..cols {
            let ch = grid[ry * gw + rx];
            if ch == BRAILLE_BLANK {
                spans.push(Span::raw(" "));
            } else {
                spans.push(Span::styled(ch.to_string(), Style::default().fg(pal.snow_flake)));
            }
        }
        let line_area = Rect { x: area.x, y: area.y + ry as u16, width: cols as u16, height: 1 };
        frame.render_widget(Paragraph::new(Line::from(spans)), line_area);
    }
}

/// Render the Doom-fire heat band at the BOTTOM of `area`. With truecolor backgrounds we
/// pack two vertical pixels per row via the upper half-block `▀` (fg = upper, bg = lower);
/// under tmux / non-truecolor we fall back to fg-only heat glyphs so we never rely on a bg
/// fill (the recon's tmux lesson).
fn draw_fire(frame: &mut Frame, app: &AppState, area: Rect) {
    let pal = EffectPalette::from_theme(&app.theme);
    let fire = &app.effects.fire;
    let band_rows = (area.height as usize / 2).clamp(1, fire.h / 2);
    let cols = (area.width as usize).min(fire.w);
    let fg_only = app.effects.caps.prefer_fg_only();
    for row in 0..band_rows {
        // Map this character row to two stacked fire pixels (upper, lower).
        let upper_y = fire.h - 1 - row * 2;
        let lower_y = upper_y.saturating_sub(1);
        let screen_y = area.y + area.height - 1 - row as u16;
        let mut spans: Vec<Span> = Vec::with_capacity(cols);
        for x in 0..cols {
            let hu = fire.heat(x, upper_y);
            let hl = fire.heat(x, lower_y);
            if fg_only {
                let h = hu.max(hl);
                let ch = Fire::glyph_for(h);
                if ch == ' ' {
                    spans.push(Span::raw(" "));
                } else {
                    spans.push(Span::styled(
                        ch.to_string(),
                        Style::default().fg(fire.color_for(h, &pal)),
                    ));
                }
            } else if hu == 0 && hl == 0 {
                spans.push(Span::raw(" "));
            } else {
                spans.push(Span::styled(
                    "▀",
                    Style::default()
                        .fg(fire.color_for(hu, &pal))
                        .bg(fire.color_for(hl, &pal)),
                ));
            }
        }
        let line_area = Rect { x: area.x, y: screen_y, width: cols as u16, height: 1 };
        frame.render_widget(Paragraph::new(Line::from(spans)), line_area);
    }
}

/// Paint a FLOWING rainbow over the composer's BORDER, plus a few drifting
/// monochrome particle/snow symbols along the top edge. Bounded to the 1-cell
/// border outline, so the terminal BACKGROUND stays clean (user feedback: bind
/// effects to the input box, don't repaint the whole screen). Truecolor-gated (a
/// no-op at mono / NO_COLOR — the Block's plain border shows) and skipped in shell
/// mode (the hot-pink border is a meaningful indicator we keep). `now_ms` drives
/// the flow + drift so the rainbow advances around the border with the tick clock.
pub fn draw_composer_border_fx(frame: &mut Frame, app: &AppState, area: Rect, now_ms: u64) {
    if !app.effects.caps.enabled() || area.width < 2 || area.height < 2 {
        return;
    }
    if app.composer.is_shell_mode() {
        return;
    }
    let stops = app.theme.rainbow();
    let w = area.width as usize;
    let h = area.height as usize;
    let perim = (2 * (w + h)).saturating_sub(4).max(1) as f32;
    let phase = now_ms as f32 * 0.00007; // gentle flow per millisecond
    let color_at = |k: usize| crate::theme::rainbow::flow_color(stops, k as f32 / perim + phase);

    // Top edge, left→right (perimeter indices 0..w), with corner glyphs.
    let mut top: Vec<Span> = Vec::with_capacity(w);
    for x in 0..w {
        let g = if x == 0 { '┌' } else if x == w - 1 { '┐' } else { '─' };
        top.push(Span::styled(g.to_string(), Style::default().fg(color_at(x))));
    }
    frame.render_widget(
        Paragraph::new(Line::from(top)),
        Rect { x: area.x, y: area.y, width: area.width, height: 1 },
    );
    // Bottom edge (drawn left→right, indexed on the far side of the loop).
    let base_bottom = w + h.saturating_sub(2);
    let mut bot: Vec<Span> = Vec::with_capacity(w);
    for x in 0..w {
        let g = if x == 0 { '└' } else if x == w - 1 { '┘' } else { '─' };
        bot.push(Span::styled(
            g.to_string(),
            Style::default().fg(color_at(base_bottom + (w - 1 - x))),
        ));
    }
    frame.render_widget(
        Paragraph::new(Line::from(bot)),
        Rect { x: area.x, y: area.y + area.height - 1, width: area.width, height: 1 },
    );
    // Side edges (interior rows): right top→bottom, left bottom→top.
    let base_left = 2 * w + h.saturating_sub(2);
    for y in 1..h.saturating_sub(1) {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "│",
                Style::default().fg(color_at(w + (y - 1))),
            ))),
            Rect { x: area.x + area.width - 1, y: area.y + y as u16, width: 1, height: 1 },
        );
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "│",
                Style::default().fg(color_at(base_left + (h - 2 - (y - 1)))),
            ))),
            Rect { x: area.x, y: area.y + y as u16, width: 1, height: 1 },
        );
    }

    // Border-bound particles along the TOP edge — monochrome SYMBOL dingbats (NO
    // emoji), colored by the same flow. A couple by default; more in subtle/full.
    // Still bounded to the top border row → no background change.
    let count = match app.effects.mode {
        crate::effects::EffectMode::Full => 6usize,
        crate::effects::EffectMode::Subtle => 3,
        crate::effects::EffectMode::Off => 2,
    };
    let count = if app.effects.demo_active() { count.max(6) } else { count };
    if w > 6 {
        let glyphs = ['✦', '✧', '·', '✶', '∗'];
        let span_w = w - 2; // interior of the top edge, between the corners
        for p in 0..count {
            let stride = (span_w / count.max(1)).max(2);
            let off = ((now_ms / 90) as usize + p * stride) % span_w;
            let gx = 1 + off;
            let glyph = glyphs[(p + (now_ms / 650) as usize) % glyphs.len()];
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    glyph.to_string(),
                    Style::default().fg(color_at(gx)),
                ))),
                Rect { x: area.x + gx as u16, y: area.y, width: 1, height: 1 },
            );
        }
    }
}
