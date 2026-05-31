//! Sparkle: a one-shot success burst.
//!
//! A burst of short-lived sparkles at random offsets near an anchor. Each sparkle has a TTL
//! that decrements by `dt`; its glyph cycles `· ✦ ✧ ✶ ·` by remaining TTL. Offsets come from
//! an embedded [`SplitMix64`]. The engine spawns a burst (success event), then steps until all
//! TTLs expire. Bounded: at most [`MAX_SPARKLES`].

use super::SplitMix64;

/// Hard cap on simultaneous sparkles (bounded buffer).
pub const MAX_SPARKLES: usize = 32;

/// Lifetime of a sparkle in seconds.
pub const SPARKLE_TTL: f32 = 0.9;

/// Glyph cycle from birth → death.
pub const SPARKLE_GLYPHS: &[char] = &['·', '✦', '✧', '✶', '·'];

/// A single sparkle.
#[derive(Debug, Clone, Copy)]
pub struct Spark {
    /// Column offset from the anchor (cells, can be negative).
    pub dx: i16,
    /// Row offset from the anchor (cells, can be negative).
    pub dy: i16,
    /// Remaining time to live (seconds).
    pub ttl: f32,
}

impl Spark {
    /// Glyph for the current TTL (cycles through [`SPARKLE_GLYPHS`]).
    pub fn glyph(&self) -> char {
        let frac = (self.ttl / SPARKLE_TTL).clamp(0.0, 1.0);
        // frac 1.0 (just born) → first glyph; 0.0 (dying) → last.
        let n = SPARKLE_GLYPHS.len();
        let idx = (((1.0 - frac) * (n as f32 - 1.0)).round() as usize).min(n - 1);
        SPARKLE_GLYPHS[idx]
    }
}

/// The sparkle effect state.
#[derive(Debug, Clone)]
pub struct Sparkle {
    sparks: Vec<Spark>,
    rng: SplitMix64,
}

impl Sparkle {
    pub fn new(seed: u64) -> Self {
        Self { sparks: Vec::with_capacity(MAX_SPARKLES), rng: SplitMix64::new(seed) }
    }

    /// Whether any sparkle is still alive.
    pub fn active(&self) -> bool {
        !self.sparks.is_empty()
    }

    /// Live sparkles (for rendering / tests).
    pub fn sparks(&self) -> &[Spark] {
        &self.sparks
    }

    /// Spawn a burst of `count` sparkles (clamped to [`MAX_SPARKLES`]) around the anchor.
    pub fn burst(&mut self, count: usize) {
        let count = count.min(MAX_SPARKLES);
        self.sparks.clear();
        for _ in 0..count {
            // Offsets in a small box around the anchor.
            let dx = (self.rng.below(11) as i16) - 5; // -5..=5
            let dy = (self.rng.below(5) as i16) - 2; // -2..=2
            // Slightly varied TTLs so they don't all blink in lockstep.
            let ttl = SPARKLE_TTL * (0.6 + self.rng.next_f32() * 0.4);
            self.sparks.push(Spark { dx, dy, ttl });
        }
    }

    /// Cancel all sparkles immediately.
    pub fn cancel(&mut self) {
        self.sparks.clear();
    }

    /// Advance all sparkles by `dt`, dropping any whose TTL expired. Pure (no clock, no RNG).
    pub fn step(&mut self, dt: f32) {
        for s in self.sparks.iter_mut() {
            s.ttl -= dt;
        }
        self.sparks.retain(|s| s.ttl > 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparkle_burst_bounded() {
        let mut s = Sparkle::new(0x59);
        s.burst(9999);
        assert!(s.sparks().len() <= MAX_SPARKLES);
        assert!(s.active());
    }

    #[test]
    fn sparkle_expires() {
        let mut s = Sparkle::new(0x59);
        s.burst(12);
        // After well over the max TTL, all should be gone.
        for _ in 0..40 {
            s.step(0.05);
        }
        assert!(!s.active());
        assert_eq!(s.sparks().len(), 0);
    }

    #[test]
    fn sparkle_glyph_cycles_over_ttl() {
        let born = Spark { dx: 0, dy: 0, ttl: SPARKLE_TTL };
        let dying = Spark { dx: 0, dy: 0, ttl: 0.0 };
        assert_eq!(born.glyph(), SPARKLE_GLYPHS[0]);
        assert_eq!(dying.glyph(), SPARKLE_GLYPHS[SPARKLE_GLYPHS.len() - 1]);
    }

    #[test]
    fn sparkle_deterministic() {
        let mut a = Sparkle::new(0x59);
        let mut b = Sparkle::new(0x59);
        a.burst(16);
        b.burst(16);
        for (sa, sb) in a.sparks().iter().zip(b.sparks().iter()) {
            assert_eq!(sa.dx, sb.dx);
            assert_eq!(sa.dy, sb.dy);
            assert_eq!(sa.ttl.to_bits(), sb.ttl.to_bits());
        }
    }
}
