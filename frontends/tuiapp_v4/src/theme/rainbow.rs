//! Rainbow separator: a 7-stop ROYGBIV gradient mapped across a separator width.
//!
//! Each column gets a color interpolated between adjacent ROYGBIV stops. The stops come from
//! theme tokens ([`crate::theme::Theme::rainbow`]) so themes can restyle. A static version and
//! a shimmer variant (which animates a brightness sweep over the gradient, reusing the shimmer
//! math) are both provided. At mono / NO_COLOR the caller renders a plain line with no color.

use ratatui::style::{Color, Style};
use ratatui::text::Span;

use crate::effects::shimmer::{blend, intensity_at};
use crate::effects::{ColorCaps, EffectPalette};
use crate::theme::Theme;

/// Number of ROYGBIV stops.
pub const STOPS: usize = 7;

/// The glyph used for a colored separator (horizontal box rule).
pub const SEP_GLYPH: char = '─';

/// Interpolate the 7-stop gradient to `width` colors. Each column is a blend between the two
/// adjacent stops. `width == 0` ⇒ empty; `width == 1` ⇒ just the first stop.
pub fn gradient(stops: &[Color; STOPS], width: usize) -> Vec<Color> {
    if width == 0 {
        return Vec::new();
    }
    if width == 1 {
        return vec![stops[0]];
    }
    let mut out = Vec::with_capacity(width);
    let segments = (STOPS - 1) as f32;
    for x in 0..width {
        // Position along the gradient in [0, segments].
        let t = (x as f32 / (width - 1) as f32) * segments;
        let i = (t.floor() as usize).min(STOPS - 2);
        let frac = t - i as f32;
        out.push(blend(stops[i], stops[i + 1], frac));
    }
    out
}

/// A CYCLIC sample of the ROYGBIV gradient for FLOWING effects: `t` (any real,
/// reduced mod 1) maps once UP the 7 stops and back DOWN (a triangle wave), so the
/// value at `t = 0` and `t → 1` match — a seamless loop as `t` advances with time.
/// Used by the flowing composer-border effect (a rainbow bound to the input box).
pub fn flow_color(stops: &[Color; STOPS], t: f32) -> Color {
    let tt = t.rem_euclid(1.0);
    // Triangle 0→1→0 so the loop is seamless (no hard ROYGBIV→red seam).
    let tri = if tt < 0.5 { tt * 2.0 } else { (1.0 - tt) * 2.0 };
    let seg = tri * (STOPS - 1) as f32;
    let i = (seg.floor() as usize).min(STOPS - 2);
    let frac = seg - i as f32;
    blend(stops[i], stops[i + 1], frac)
}

/// Build the header separator as styled [`Span`]s for the given width, honoring capabilities.
///
/// - mono / NO_COLOR ⇒ a single plain dim line (no per-column color).
/// - otherwise ⇒ the ROYGBIV gradient; if `shimmer_phase` is `Some`, a brightness sweep at that
///   phase is blended over the gradient for the "running" animated variant.
pub fn separator_spans(
    theme: &Theme,
    caps: &ColorCaps,
    width: usize,
    shimmer_phase: Option<f32>,
) -> Vec<Span<'static>> {
    let pal = EffectPalette::from_theme(theme);
    if !caps.enabled() {
        // Mono fallback: a plain line, no per-cell color (one raw span).
        return vec![Span::raw(SEP_GLYPH.to_string().repeat(width))];
    }
    // The shimmer highlight blends each gradient cell toward a brighter "glow" theme token
    // (a lighter step of the accent) — the recon's `*_shimmer = lighter(base)` convention —
    // rather than a hardcoded white, so the sweep restyles with the theme.
    let glow = pal.shimmer_glow;
    let base = gradient(&pal.rainbow, width);
    let mut spans = Vec::with_capacity(width);
    for (x, &col) in base.iter().enumerate() {
        let color = match shimmer_phase {
            Some(phase) => {
                let t = intensity_at(phase, 0.18, x, width);
                blend(col, glow, t * 0.8)
            }
            None => col,
        };
        spans.push(Span::styled(SEP_GLYPH.to_string(), Style::default().fg(color)));
    }
    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rainbow_7_stops() {
        // Exactly 7 stops, and the gradient endpoints match the first/last theme tokens.
        let theme = Theme::ga_default();
        let stops = theme.rainbow();
        assert_eq!(stops.len(), 7);
        let g = gradient(stops, 64);
        assert_eq!(g.len(), 64);
        assert_eq!(*g.first().unwrap(), stops[0]);
        assert_eq!(*g.last().unwrap(), stops[STOPS - 1]);
    }

    #[test]
    fn flow_color_is_cyclic_and_moves() {
        let theme = Theme::ga_default();
        let stops = theme.rainbow();
        // Seamless loop: t=0 and t=1 (≡0) match; the midpoint differs (it moves).
        assert_eq!(flow_color(stops, 0.0), flow_color(stops, 1.0));
        assert_ne!(flow_color(stops, 0.0), flow_color(stops, 0.5));
        // Reducible mod 1 (advancing phase wraps cleanly).
        assert_eq!(flow_color(stops, 0.25), flow_color(stops, 1.25));
    }

    #[test]
    fn gradient_edge_cases() {
        let theme = Theme::ga_default();
        let stops = theme.rainbow();
        assert!(gradient(stops, 0).is_empty());
        assert_eq!(gradient(stops, 1), vec![stops[0]]);
    }

    #[test]
    fn separator_mono_is_plain() {
        let theme = Theme::ga_default();
        let caps = ColorCaps::mono();
        let spans = separator_spans(&theme, &caps, 10, None);
        // One plain span, no fg color.
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].style.fg, None);
        assert_eq!(spans[0].content.chars().count(), 10);
    }

    #[test]
    fn separator_color_has_per_column_spans() {
        let theme = Theme::ga_default();
        let caps = ColorCaps::from_env_values(false, "truecolor", "xterm-256color");
        let spans = separator_spans(&theme, &caps, 12, None);
        assert_eq!(spans.len(), 12);
        assert!(spans.iter().all(|s| s.style.fg.is_some()));
    }

    #[test]
    fn separator_shimmer_variant_differs() {
        let theme = Theme::ga_default();
        let caps = ColorCaps::from_env_values(false, "truecolor", "xterm-256color");
        let plain = separator_spans(&theme, &caps, 40, None);
        let shimmered = separator_spans(&theme, &caps, 40, Some(0.5));
        // At least one column near the shimmer center should differ.
        let any_diff = plain
            .iter()
            .zip(shimmered.iter())
            .any(|(a, b)| a.style.fg != b.style.fg);
        assert!(any_diff, "shimmer variant should brighten some columns");
    }
}
