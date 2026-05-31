//! Doom-fire: a heat framebuffer rendered with the upper half-block `▀`.
//!
//! Classic PSX Doom fire. A `w*h` framebuffer of `u8` palette indices (`0..=MAX_HEAT`). The
//! bottom row is pinned at max heat; each step every cell pulls heat from the cell below with
//! a random horizontal drift and a random decay. The stepper is pure (randomness from an
//! embedded [`SplitMix64`]) and bounded (the framebuffer never grows).
//!
//! Rendering produces two vertical pixels per character row via `▀`: fg = upper pixel color,
//! bg = lower pixel color. Under tmux / non-truecolor, the caller renders fg-only using a
//! glyph ramp indexed by heat so we never rely on background fills.

use ratatui::style::Color;

use super::{EffectPalette, SplitMix64};

/// Maximum heat palette index (37-entry ramp: 0..=36).
pub const MAX_HEAT: u8 = 36;

/// Glyph ramp for the fg-only (tmux / no-truecolor-bg) fallback, dark → bright.
pub const HEAT_GLYPHS: &[char] = &[' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];

/// The Doom-fire effect state.
#[derive(Debug, Clone)]
pub struct Fire {
    pub w: usize,
    pub h: usize,
    /// Row-major heat framebuffer, `w*h`, each `0..=MAX_HEAT`.
    buf: Vec<u8>,
    rng: SplitMix64,
    seed: u64,
}

impl Fire {
    pub fn new(w: usize, h: usize, seed: u64) -> Self {
        let w = w.max(1);
        let h = h.max(2);
        let mut f = Self { w, h, buf: vec![0; w * h], rng: SplitMix64::new(seed), seed };
        f.ignite_source();
        f
    }

    #[inline]
    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.w + x
    }

    /// Heat at `(x, y)`; 0 if out of bounds.
    pub fn heat(&self, x: usize, y: usize) -> u8 {
        if x < self.w && y < self.h {
            self.buf[self.idx(x, y)]
        } else {
            0
        }
    }

    /// The raw framebuffer (introspection / the `fire_step_bounded` test).
    #[allow(dead_code)]
    pub fn buffer(&self) -> &[u8] {
        &self.buf
    }

    /// Pin the bottom row to max heat (the fire source).
    fn ignite_source(&mut self) {
        let bottom = self.h - 1;
        for x in 0..self.w {
            let i = self.idx(x, bottom);
            self.buf[i] = MAX_HEAT;
        }
    }

    /// Reset the PRNG and re-ignite (used when (re)starting the effect / demo).
    pub fn reseed(&mut self) {
        self.rng = SplitMix64::new(self.seed);
        for v in self.buf.iter_mut() {
            *v = 0;
        }
        self.ignite_source();
    }

    /// Advance the fire one logical frame. `dt` modulates how many propagation passes run so
    /// the animation speed is roughly frame-rate independent; at least one pass always runs.
    /// Pure: all randomness comes from `self.rng`.
    pub fn step(&mut self, dt: f32) {
        // ~30 logical updates/sec; clamp passes to keep it bounded.
        let passes = ((dt * 30.0).round() as u32).clamp(1, 3);
        for _ in 0..passes {
            self.propagate_once();
        }
        // Keep the source lit.
        self.ignite_source();
    }

    /// One upward heat-propagation pass.
    fn propagate_once(&mut self) {
        // Spread from the bottom upward: each cell draws from the cell directly below.
        for y in 0..self.h - 1 {
            for x in 0..self.w {
                let below = self.buf[self.idx(x, y + 1)];
                // Random decay 0..=3 and horizontal drift -1..=1.
                let decay = self.rng.below(4) as u8;
                let new_heat = below.saturating_sub(decay);
                // Drift the value into a neighbouring column to get the flickery look.
                let drift = self.rng.below(3) as i32 - 1; // -1, 0, +1
                let dst_x = (x as i32 + drift).clamp(0, self.w as i32 - 1) as usize;
                let i = self.idx(dst_x, y);
                self.buf[i] = new_heat.min(MAX_HEAT);
            }
        }
    }

    /// Map a heat value `0..=MAX_HEAT` to a theme color along the hot→smoke ramp.
    pub fn color_for(&self, heat: u8, pal: &EffectPalette) -> Color {
        heat_to_color(heat, pal)
    }

    /// Map a heat value to a fg-only glyph (tmux / no-truecolor-bg fallback).
    pub fn glyph_for(heat: u8) -> char {
        if heat == 0 {
            return ' ';
        }
        let n = HEAT_GLYPHS.len();
        let i = ((heat as usize * (n - 1)) / MAX_HEAT as usize).min(n - 1);
        HEAT_GLYPHS[i]
    }
}

/// Pure heat→color mapping over the 4 fire tokens.
pub fn heat_to_color(heat: u8, pal: &EffectPalette) -> Color {
    let t = heat as f32 / MAX_HEAT as f32;
    if t <= 0.0 {
        pal.fire_smoke
    } else if t < 0.35 {
        pal.fire_smoke
    } else if t < 0.6 {
        pal.fire_cool
    } else if t < 0.85 {
        pal.fire_mid
    } else {
        pal.fire_hot
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::Theme;

    #[test]
    fn fire_step_bounded() {
        let mut fire = Fire::new(40, 12, 0xF1);
        let len0 = fire.buffer().len();
        for _ in 0..500 {
            fire.step(0.033);
            // Buffer length never changes (no growth).
            assert_eq!(fire.buffer().len(), len0);
            // Every cell stays within the palette range.
            for &v in fire.buffer() {
                assert!(v <= MAX_HEAT, "heat {} exceeds MAX_HEAT", v);
            }
        }
        // Bottom row is always max heat (the source).
        for x in 0..fire.w {
            assert_eq!(fire.heat(x, fire.h - 1), MAX_HEAT);
        }
    }

    #[test]
    fn fire_deterministic_for_seed() {
        let mut a = Fire::new(20, 8, 0xABCD);
        let mut b = Fire::new(20, 8, 0xABCD);
        for _ in 0..50 {
            a.step(0.033);
            b.step(0.033);
        }
        assert_eq!(a.buffer(), b.buffer());
    }

    #[test]
    fn fire_glyph_ramp_monotone_endpoints() {
        assert_eq!(Fire::glyph_for(0), ' ');
        assert_eq!(Fire::glyph_for(MAX_HEAT), '@');
    }

    #[test]
    fn fire_color_uses_theme_tokens() {
        let theme = Theme::ga_default();
        let pal = EffectPalette::from_theme(&theme);
        // Hottest maps to fire_hot, coldest to smoke.
        assert_eq!(heat_to_color(MAX_HEAT, &pal), pal.fire_hot);
        assert_eq!(heat_to_color(0, &pal), pal.fire_smoke);
    }
}
