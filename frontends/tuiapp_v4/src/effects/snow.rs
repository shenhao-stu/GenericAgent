//! Snow: a braille particle field.
//!
//! N bounded particles, each with `(x, y, vy, phase)`. A step advances `y` by `vy*dt` and `x`
//! by a gentle sine sway (`phase` advanced by `dt`), wrapping to the top when a flake falls
//! past the bottom. Particles render into a braille grid (2×4 dots per cell, base `U+2800`)
//! using the `snow.flake` token; no background fill is needed, so this is tmux-safe.
//!
//! Determinism: the initial layout comes from an embedded [`SplitMix64`] with a fixed seed,
//! and advancement is a pure function of `dt` and current state.

use super::SplitMix64;

/// Braille cell is 2 dots wide, 4 dots tall.
pub const DOT_W: usize = 2;
pub const DOT_H: usize = 4;
/// Base codepoint of the braille block.
pub const BRAILLE_BASE: u32 = 0x2800;

/// Hard cap on particle count (bounded buffer).
pub const MAX_PARTICLES: usize = 256;

/// A single snow particle. Positions are in *dot* units (subpixel within the braille grid).
#[derive(Debug, Clone, Copy)]
pub struct Flake {
    /// Horizontal position in dots.
    pub x: f32,
    /// Vertical position in dots.
    pub y: f32,
    /// Fall speed in dots/sec.
    pub vy: f32,
    /// Sway phase (radians).
    pub phase: f32,
    /// Sway speed (radians/sec).
    pub sway_speed: f32,
    /// Sway amplitude in dots.
    pub sway_amp: f32,
}

/// The snow effect state.
#[derive(Debug, Clone)]
pub struct Snow {
    /// Grid width in cells.
    pub w: usize,
    /// Grid height in cells.
    pub h: usize,
    flakes: Vec<Flake>,
    rng: SplitMix64,
    seed: u64,
}

impl Snow {
    pub fn new(w: usize, h: usize, count: usize, seed: u64) -> Self {
        let w = w.max(1);
        let h = h.max(1);
        let count = count.min(MAX_PARTICLES);
        let mut s = Self { w, h, flakes: Vec::with_capacity(count), rng: SplitMix64::new(seed), seed };
        s.populate(count);
        s
    }

    fn dot_w(&self) -> f32 {
        (self.w * DOT_W) as f32
    }
    fn dot_h(&self) -> f32 {
        (self.h * DOT_H) as f32
    }

    /// Initialise `count` flakes at random positions (deterministic from the seed).
    fn populate(&mut self, count: usize) {
        self.flakes.clear();
        let dw = self.dot_w();
        let dh = self.dot_h();
        for _ in 0..count {
            let x = self.rng.next_f32() * dw;
            let y = self.rng.next_f32() * dh;
            let vy = 4.0 + self.rng.next_f32() * 10.0; // dots/sec
            let phase = self.rng.next_f32() * std::f32::consts::TAU;
            let sway_speed = 0.5 + self.rng.next_f32() * 1.5;
            let sway_amp = 0.4 + self.rng.next_f32() * 1.2;
            self.flakes.push(Flake { x, y, vy, phase, sway_speed, sway_amp });
        }
    }

    /// Reset PRNG + re-populate with the same count.
    pub fn reseed(&mut self) {
        let count = self.flakes.len();
        self.rng = SplitMix64::new(self.seed);
        self.populate(count);
    }

    /// Number of active flakes. (Introspection / the `snow_*` tests; renderers use
    /// [`Snow::render_grid`].)
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.flakes.len()
    }
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.flakes.is_empty()
    }

    /// Read-only view of the flakes (introspection / determinism tests).
    #[allow(dead_code)]
    pub fn flakes(&self) -> &[Flake] {
        &self.flakes
    }

    /// Advance every flake by `dt`. Pure function of `dt` + current state (no RNG, no clock).
    pub fn step(&mut self, dt: f32) {
        let dh = self.dot_h();
        let dw = self.dot_w();
        for f in self.flakes.iter_mut() {
            f.phase += f.sway_speed * dt;
            f.y += f.vy * dt;
            // Horizontal sway around the flake's drift column.
            f.x += (f.phase.sin()) * f.sway_amp * dt * 4.0;
            // Wrap horizontally.
            if f.x < 0.0 {
                f.x += dw;
            } else if f.x >= dw {
                f.x -= dw;
            }
            // Wrap to the top when it falls past the bottom.
            if f.y >= dh {
                f.y -= dh;
            }
        }
    }

    /// Render the field into a braille grid: returns `w*h` chars, row-major. Empty cells are
    /// the blank braille `U+2800`. Caller styles every glyph with `snow.flake` (fg only).
    pub fn render_grid(&self) -> Vec<char> {
        // Each cell holds an 8-bit braille dot mask.
        let mut masks = vec![0u8; self.w * self.h];
        for f in &self.flakes {
            let dx = f.x as usize;
            let dy = f.y as usize;
            let cx = dx / DOT_W;
            let cy = dy / DOT_H;
            if cx >= self.w || cy >= self.h {
                continue;
            }
            let bit = braille_bit(dx % DOT_W, dy % DOT_H);
            masks[cy * self.w + cx] |= bit;
        }
        masks
            .into_iter()
            .map(|m| char::from_u32(BRAILLE_BASE + m as u32).unwrap_or('⠀'))
            .collect()
    }
}

/// Map a `(col, row)` dot inside a braille cell to its bit in the 8-dot mask.
///
/// Braille dot numbering (Unicode):
/// ```text
///  1 4   -> bits 0 3
///  2 5   -> bits 1 4
///  3 6   -> bits 2 5
///  7 8   -> bits 6 7
/// ```
pub fn braille_bit(col: usize, row: usize) -> u8 {
    // Standard Unicode braille dot bit layout.
    const MAP: [[u8; 2]; 4] = [
        [0x01, 0x08],
        [0x02, 0x10],
        [0x04, 0x20],
        [0x40, 0x80],
    ];
    let c = col.min(1);
    let r = row.min(3);
    MAP[r][c]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snow_advance_deterministic() {
        // Same seed + same dt sequence ⇒ identical particle positions.
        let mut a = Snow::new(40, 20, 64, 0x5E);
        let mut b = Snow::new(40, 20, 64, 0x5E);
        let dts = [0.016f32, 0.033, 0.05, 0.02, 0.04];
        for _ in 0..200 {
            for &dt in &dts {
                a.step(dt);
                b.step(dt);
            }
        }
        assert_eq!(a.len(), b.len());
        for (fa, fb) in a.flakes().iter().zip(b.flakes().iter()) {
            assert_eq!(fa.x.to_bits(), fb.x.to_bits());
            assert_eq!(fa.y.to_bits(), fb.y.to_bits());
        }
    }

    #[test]
    fn snow_particles_bounded() {
        let snow = Snow::new(10, 10, 9999, 0x1);
        assert!(snow.len() <= MAX_PARTICLES);
    }

    #[test]
    fn snow_stays_in_grid_after_many_steps() {
        let mut snow = Snow::new(30, 15, 80, 0xAA);
        for _ in 0..1000 {
            snow.step(0.05);
        }
        let dw = (snow.w * DOT_W) as f32;
        let dh = (snow.h * DOT_H) as f32;
        for f in snow.flakes() {
            assert!(f.x >= 0.0 && f.x < dw, "x {} out of [0,{})", f.x, dw);
            assert!(f.y >= 0.0 && f.y < dh, "y {} out of [0,{})", f.y, dh);
        }
        // Grid renders to exactly w*h cells.
        assert_eq!(snow.render_grid().len(), snow.w * snow.h);
    }

    #[test]
    fn braille_bits_distinct() {
        let mut seen = std::collections::HashSet::new();
        for r in 0..DOT_H {
            for c in 0..DOT_W {
                assert!(seen.insert(braille_bit(c, r)), "duplicate braille bit");
            }
        }
        assert_eq!(seen.len(), 8);
    }
}
