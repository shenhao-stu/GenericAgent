//! Shimmer: a raised-cosine highlight sweep across a label or separator.
//!
//! A subtle "running" indicator. The highlight center advances with `dt` (wrapping). For a
//! column `x` the intensity is `0.5*(1+cos(pi*(x-center)/width))` clamped to `[0,1]`, which is
//! a smooth raised-cosine hump that maps to a brightness blend between two theme tokens
//! (`shimmer.base` → `shimmer.glow`). Pure function of the accumulated phase.

use ratatui::style::Color;

/// The shimmer effect state — just an advancing phase in `[0, 1)`.
#[derive(Debug, Clone)]
pub struct Shimmer {
    /// Sweep center as a fraction in `[0, 1)` across the target width.
    phase: f32,
    /// Sweep speed in fractions/sec.
    speed: f32,
    /// Half-width of the highlight hump as a fraction of the target width. (Consumed by
    /// the `Shimmer::intensity` convenience accessor; the separator calls the free
    /// `intensity_at` with its own half-width, so the field is otherwise introspection.)
    #[allow(dead_code)]
    half_width: f32,
}

impl Default for Shimmer {
    fn default() -> Self {
        Self::new()
    }
}

impl Shimmer {
    pub fn new() -> Self {
        Self { phase: 0.0, speed: 0.55, half_width: 0.18 }
    }

    /// Current sweep center as a fraction in `[0,1)`.
    pub fn phase(&self) -> f32 {
        self.phase
    }

    /// Advance the sweep by `dt`. Wraps in `[0,1)`. Pure (no clock).
    pub fn step(&mut self, dt: f32) {
        self.phase += self.speed * dt;
        // Wrap into [0, 1) with a little overscan so the hump can travel off both edges.
        while self.phase >= 1.2 {
            self.phase -= 1.4;
        }
    }

    /// Intensity in `[0,1]` for column `x` of `width` columns at the current phase.
    /// (Convenience over the free [`intensity_at`]; used by tests + as label-shimmer API.)
    #[allow(dead_code)]
    pub fn intensity(&self, x: usize, width: usize) -> f32 {
        intensity_at(self.phase, self.half_width, x, width)
    }

    /// Blend `base`→`glow` by the shimmer intensity at column `x` (label-shimmer API).
    #[allow(dead_code)]
    pub fn color_at(&self, x: usize, width: usize, base: Color, glow: Color) -> Color {
        let t = self.intensity(x, width);
        blend(base, glow, t)
    }
}

/// Pure raised-cosine intensity: `0.5*(1+cos(pi*d/half_width))` for `|d| < half_width`, else 0,
/// where `d` is the distance (in width-fractions) from the sweep center to column `x`.
pub fn intensity_at(center: f32, half_width: f32, x: usize, width: usize) -> f32 {
    if width == 0 || half_width <= 0.0 {
        return 0.0;
    }
    let pos = (x as f32 + 0.5) / width as f32;
    let d = (pos - center).abs();
    if d >= half_width {
        0.0
    } else {
        let v = 0.5 * (1.0 + (std::f32::consts::PI * d / half_width).cos());
        v.clamp(0.0, 1.0)
    }
}

/// Linear RGB blend `a`→`b` by `t` in `[0,1]`. Non-RGB colors fall back to `a` (t<0.5) or `b`.
pub fn blend(a: Color, b: Color, t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    match (a, b) {
        (Color::Rgb(ar, ag, ab), Color::Rgb(br, bg, bb)) => {
            let lerp = |x: u8, y: u8| -> u8 {
                (x as f32 + (y as f32 - x as f32) * t).round().clamp(0.0, 255.0) as u8
            };
            Color::Rgb(lerp(ar, br), lerp(ag, bg), lerp(ab, bb))
        }
        _ => {
            if t < 0.5 {
                a
            } else {
                b
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shimmer_intensity_peaks_at_center() {
        // Center exactly on a column ⇒ that column near 1.0, far columns 0.0.
        let width = 20;
        let center = 0.5;
        let hw = 0.2;
        let mid = (center * width as f32) as usize;
        let peak = intensity_at(center, hw, mid, width);
        let edge = intensity_at(center, hw, 0, width);
        assert!(peak > 0.8, "peak {}", peak);
        assert_eq!(edge, 0.0);
    }

    #[test]
    fn shimmer_intensity_bounded() {
        for x in 0..50 {
            let v = intensity_at(0.3, 0.25, x, 50);
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn shimmer_phase_wraps() {
        let mut s = Shimmer::new();
        for _ in 0..10_000 {
            s.step(0.05);
            assert!(s.phase() < 1.2 && s.phase() > -0.5);
        }
    }

    #[test]
    fn blend_endpoints_and_midpoint() {
        let a = Color::Rgb(0, 0, 0);
        let b = Color::Rgb(100, 200, 50);
        assert_eq!(blend(a, b, 0.0), a);
        assert_eq!(blend(a, b, 1.0), b);
        assert_eq!(blend(a, b, 0.5), Color::Rgb(50, 100, 25));
    }

    #[test]
    fn blend_non_rgb_fallback() {
        assert_eq!(blend(Color::Red, Color::Blue, 0.2), Color::Red);
        assert_eq!(blend(Color::Red, Color::Blue, 0.9), Color::Blue);
    }
}
