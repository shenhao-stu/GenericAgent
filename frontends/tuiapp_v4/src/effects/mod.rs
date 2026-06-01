//! effects/ — deterministic, bounded, theme-tokened terminal flourishes (§9 / N8).
//!
//! # Design tenets (from `../tuiapp_v4_research/recon/effects_engine.md`)
//!
//! 1. **Pure steppers.** Every effect is a `step(&mut state, dt)` that mutates a small,
//!    pre-allocated, bounded buffer. No allocation in steady state, no `Instant::now()`,
//!    no `rand` inside a stepper — randomness comes from a seeded [`SplitMix64`] embedded
//!    in the state and advanced deterministically. Every effect is thus unit-testable.
//! 2. **One delta-time clock, driven by the tick counter.** The cockpit already has a
//!    single 0.1s tick (`AppState::tick`). [`EffectsEngine::tick`] consumes the elapsed
//!    `dt` (seconds) the app derives from that tick — effects NEVER read a wall clock.
//!    For `--smoke` the app pins FPS to 0 (it simply never calls `tick`), so every effect
//!    is frozen and the smoke frame is fully deterministic.
//! 3. **Capability gate.** Truecolor only when `COLORTERM=truecolor|24bit`; honor
//!    `NO_COLOR`; under tmux/screen prefer fg glyphs (truecolor *backgrounds* are
//!    unreliable there); degrade ANSI256 → 16 → mono.
//! 4. **Theme tokens only.** Colors resolve through [`EffectPalette`], derived from the
//!    active [`Theme`] via its semantic [`Token`]s — no hardcoded RGB at the effect sites.
//! 5. **Off by default.** Ambient effects are OFF in the cockpit unless the user opts in
//!    via `/effects`. The only always-on flourish is the rainbow header separator.

pub mod fire;
pub mod lightning;
pub mod shimmer;
pub mod snow;
pub mod sparkle;

use ratatui::style::Color;

use crate::theme::{Theme, Token};

// ---------------------------------------------------------------------------
// Deterministic PRNG
// ---------------------------------------------------------------------------

/// A tiny SplitMix64 PRNG. Seeded from a fixed constant (never from time) so every
/// effect produces an identical, replayable sequence for a given seed — the whole
/// reason the steppers are testable.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Construct from a fixed seed. Never seed this from wall-clock time.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Standard SplitMix64 step.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A float in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        // Top 24 bits → a uniform float in [0,1).
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }

    /// A `u32` in `[0, n)`; returns 0 if `n == 0`.
    pub fn below(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as u32
    }
}

/// Fixed base seed; per-effect seeds XOR an effect id so each effect has an
/// independent but deterministic stream.
pub const SEED_BASE: u64 = 0x00C0_FFEE_BADC_0DE5;

// ---------------------------------------------------------------------------
// Capability gate
// ---------------------------------------------------------------------------

/// Color depth the terminal is believed to support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorDepth {
    /// No color at all (NO_COLOR, dumb terminal).
    Mono,
    /// Basic 16-color ANSI.
    Ansi16,
    /// 256-color ANSI.
    Ansi256,
    /// 24-bit truecolor.
    True,
}

/// Detected terminal color capabilities. Effects degrade through these, never break.
#[derive(Debug, Clone, Copy)]
pub struct ColorCaps {
    pub depth: ColorDepth,
    /// Running under tmux/screen — truecolor *background* fills are unreliable here,
    /// so prefer fg-driven glyphs.
    pub tmux: bool,
    /// `NO_COLOR` was set — disable all color regardless of depth.
    pub no_color: bool,
}

impl ColorCaps {
    /// Detect capabilities from environment variables.
    pub fn detect() -> Self {
        let no_color = std::env::var_os("NO_COLOR").is_some();
        let colorterm = std::env::var("COLORTERM").unwrap_or_default().to_ascii_lowercase();
        let term = std::env::var("TERM").unwrap_or_default().to_ascii_lowercase();
        Self::from_env_values(no_color, &colorterm, &term)
    }

    /// Pure detection logic, exposed for tests.
    pub fn from_env_values(no_color: bool, colorterm: &str, term: &str) -> Self {
        let tmux = term.contains("tmux") || term.contains("screen");
        let depth = if no_color {
            ColorDepth::Mono
        } else if colorterm.contains("truecolor") || colorterm.contains("24bit") {
            ColorDepth::True
        } else if term.contains("256") {
            ColorDepth::Ansi256
        } else if term.is_empty() || term == "dumb" {
            ColorDepth::Mono
        } else {
            ColorDepth::Ansi16
        };
        Self { depth, tmux, no_color }
    }

    /// All-color-off shorthand (used in tests / smoke).
    pub fn mono() -> Self {
        Self { depth: ColorDepth::Mono, tmux: false, no_color: true }
    }

    /// Whether truecolor fg+bg is available. (Exposed for the capability-gate tests +
    /// callers that want to choose half-block bg fills vs fg glyphs explicitly.)
    #[allow(dead_code)]
    pub fn supports_truecolor(&self) -> bool {
        !self.no_color && self.depth == ColorDepth::True
    }

    /// Whether any color may be emitted at all.
    pub fn enabled(&self) -> bool {
        !self.no_color && self.depth != ColorDepth::Mono
    }

    /// Under tmux (or any non-truecolor depth) prefer fg glyphs over bg fills.
    pub fn prefer_fg_only(&self) -> bool {
        self.tmux || self.depth != ColorDepth::True
    }
}

// ---------------------------------------------------------------------------
// Effect palette (theme tokens → effect colors)
// ---------------------------------------------------------------------------

/// Semantic effect colors derived from the active [`Theme`]'s [`Token`]s. Effects read
/// this palette (never raw theme internals), so the theme→effect mapping lives in one
/// place and every effect re-themes for free when the theme changes.
///
/// `shimmer_base` (the dim end of the shimmer pair) and `plain` (the mono-fallback
/// separator color) are part of this documented contract but not read by the current
/// renderers (the separator blends gradient→`shimmer_glow`, and its mono branch uses an
/// unstyled span); they're kept so the palette stays a complete, stable token set.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct EffectPalette {
    pub fire_hot: Color,
    pub fire_mid: Color,
    pub fire_cool: Color,
    pub fire_smoke: Color,
    pub snow_flake: Color,
    pub lightning_bolt: Color,
    pub lightning_glow: Color,
    pub shimmer_base: Color,
    pub shimmer_glow: Color,
    pub sparkle: Color,
    /// 7 ROYGBIV stops for the rainbow separator (the theme's own rainbow).
    pub rainbow: [Color; 7],
    /// Plain separator color (mono / no-color fallback).
    pub plain: Color,
}

impl EffectPalette {
    /// Derive an effect palette from a theme, reusing its semantic tokens so each theme
    /// auto-restyles its effects. Fire runs warn→error→accent→dim (hot→smoke); the
    /// rainbow is the theme's 7 stops.
    pub fn from_theme(theme: &Theme) -> Self {
        Self {
            fire_hot: theme.color(Token::Warning),
            fire_mid: theme.color(Token::Error),
            fire_cool: theme.color(Token::AutoAccept),
            fire_smoke: theme.color(Token::Dim),
            snow_flake: theme.color(Token::Text),
            lightning_bolt: theme.color(Token::Warning),
            lightning_glow: theme.color(Token::Suggestion),
            shimmer_base: theme.color(Token::Dim),
            shimmer_glow: theme.color(Token::Claude),
            sparkle: theme.color(Token::Success),
            rainbow: *theme.rainbow(),
            plain: theme.color(Token::Border),
        }
    }
}

// ---------------------------------------------------------------------------
// Effect mode & engine
// ---------------------------------------------------------------------------

/// The user-selectable effects mode (`/effects [demo|off|subtle|full]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EffectMode {
    /// Everything off (default). Only the static rainbow separator may show.
    #[default]
    Off,
    /// Shimmer on the running indicator, sparkle on success, animated rainbow — but no
    /// ambient fire/snow.
    Subtle,
    /// Ambient fire/snow + all indicators.
    Full,
}

impl EffectMode {
    /// Parse the persistent `/effects` argument (`demo` is handled separately as a
    /// transient splash, so it is NOT one of these).
    pub fn parse(s: &str) -> Option<EffectMode> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "" => Some(EffectMode::Off),
            "subtle" => Some(EffectMode::Subtle),
            "full" => Some(EffectMode::Full),
            _ => None,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            EffectMode::Off => "off",
            EffectMode::Subtle => "subtle",
            EffectMode::Full => "full",
        }
    }
}

/// Bounded buffer size for the ambient fire band.
const FIRE_W: usize = 120;
const FIRE_H: usize = 16;
/// Snow field dimensions (in cells) + particle count.
const SNOW_W: usize = 120;
const SNOW_H: usize = 40;
const SNOW_N: usize = 110;

/// The effects engine: owns the master mode, per-effect state, a frame counter, and the
/// accumulated clock. `tick(dt)` advances every enabled effect by `dt` seconds. Under
/// `--smoke` the app simply never calls `tick`, freezing every effect.
#[derive(Debug)]
pub struct EffectsEngine {
    pub mode: EffectMode,
    pub caps: ColorCaps,
    /// Monotonic frame counter (advanced once per `tick`).
    pub frame: u64,
    /// Seconds elapsed since the engine started (sum of `dt`). Drives shimmer phase.
    pub clock: f32,
    /// Transient demo-splash countdown (seconds). When > 0 the demo is showing.
    pub demo_timer: f32,
    pub fire: fire::Fire,
    pub snow: snow::Snow,
    pub lightning: lightning::Lightning,
    pub shimmer: shimmer::Shimmer,
    pub sparkle: sparkle::Sparkle,
}

impl EffectsEngine {
    /// How long the demo splash runs before reverting (seconds).
    pub const DEMO_SECS: f32 = 6.0;

    pub fn new(caps: ColorCaps) -> Self {
        Self {
            mode: EffectMode::Off,
            caps,
            frame: 0,
            clock: 0.0,
            demo_timer: 0.0,
            fire: fire::Fire::new(FIRE_W, FIRE_H, SEED_BASE ^ 0xF1),
            snow: snow::Snow::new(SNOW_W, SNOW_H, SNOW_N, SEED_BASE ^ 0x5E),
            lightning: lightning::Lightning::new(SEED_BASE ^ 0x11),
            shimmer: shimmer::Shimmer::new(),
            sparkle: sparkle::Sparkle::new(SEED_BASE ^ 0x59),
        }
    }

    /// Detect caps from env and build the engine.
    pub fn from_env() -> Self {
        Self::new(ColorCaps::detect())
    }

    /// Advance every active effect by `dt` seconds. The single impure clock that
    /// produces `dt` lives in [`crate::app::AppState`]; this function and everything it
    /// calls is pure. `dt` is clamped so a long stall (debugger, suspend) can't fast-
    /// forward the animation.
    pub fn tick(&mut self, dt: f32) {
        let dt = dt.clamp(0.0, 0.1);
        self.frame = self.frame.wrapping_add(1);
        self.clock += dt;

        if self.demo_timer > 0.0 {
            self.demo_timer = (self.demo_timer - dt).max(0.0);
        }

        let ambient = matches!(self.mode, EffectMode::Full) || self.demo_active();
        let indicators =
            matches!(self.mode, EffectMode::Subtle | EffectMode::Full) || self.demo_active();

        if ambient {
            self.fire.step(dt);
            self.snow.step(dt);
        }
        if indicators {
            self.shimmer.step(dt);
        }
        // Transient one-shots (lightning flash / sparkle burst) always decay if active,
        // regardless of mode, so an in-flight effect finishes cleanly.
        self.lightning.step(dt);
        self.sparkle.step(dt);
    }

    /// Whether the transient demo splash is currently showing.
    pub fn demo_active(&self) -> bool {
        self.demo_timer > 0.0
    }

    /// Set the persistent mode (off/subtle/full) and reset transient state as needed.
    pub fn set_mode(&mut self, mode: EffectMode) {
        self.mode = mode;
        if mode == EffectMode::Off {
            self.lightning.cancel();
            self.sparkle.cancel();
        }
    }

    /// Start the demo splash (cycles through every effect for [`Self::DEMO_SECS`]).
    pub fn start_demo(&mut self) {
        self.demo_timer = Self::DEMO_SECS;
        self.fire.reseed();
        self.snow.reseed();
        self.lightning.flash();
        self.sparkle.burst(10);
    }

    /// Failure event → ~0.15s lightning flash (only when effects are active or demoing).
    pub fn flash_lightning(&mut self) {
        if self.mode != EffectMode::Off || self.demo_active() {
            self.lightning.flash();
        }
    }

    /// Success event → sparkle burst.
    pub fn burst_sparkle(&mut self) {
        if self.mode != EffectMode::Off || self.demo_active() {
            self.sparkle.burst(12);
        }
    }

    /// The shimmer/running indicator is visible in subtle+full (and demo).
    pub fn running_indicator_active(&self) -> bool {
        matches!(self.mode, EffectMode::Subtle | EffectMode::Full) || self.demo_active()
    }
}

impl Default for EffectsEngine {
    fn default() -> Self {
        Self::new(ColorCaps::mono())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_gate_no_color() {
        // NO_COLOR set ⇒ enabled() false, depth Mono — regardless of COLORTERM.
        let caps = ColorCaps::from_env_values(true, "truecolor", "xterm-256color");
        assert!(!caps.enabled());
        assert_eq!(caps.depth, ColorDepth::Mono);
        assert!(!caps.supports_truecolor());
    }

    #[test]
    fn caps_truecolor_detected() {
        let caps = ColorCaps::from_env_values(false, "truecolor", "xterm-256color");
        assert_eq!(caps.depth, ColorDepth::True);
        assert!(caps.supports_truecolor());
        assert!(caps.enabled());
        assert!(!caps.prefer_fg_only());
    }

    #[test]
    fn caps_tmux_prefers_fg_even_with_truecolor() {
        let caps = ColorCaps::from_env_values(false, "truecolor", "tmux-256color");
        assert!(caps.tmux);
        assert!(caps.prefer_fg_only());
    }

    #[test]
    fn caps_ansi256_and_16_and_dumb() {
        let c256 = ColorCaps::from_env_values(false, "", "xterm-256color");
        assert_eq!(c256.depth, ColorDepth::Ansi256);
        let c16 = ColorCaps::from_env_values(false, "", "xterm");
        assert_eq!(c16.depth, ColorDepth::Ansi16);
        let dumb = ColorCaps::from_env_values(false, "", "dumb");
        assert_eq!(dumb.depth, ColorDepth::Mono);
    }

    #[test]
    fn splitmix_deterministic_and_bounded() {
        let mut a = SplitMix64::new(SEED_BASE);
        let mut b = SplitMix64::new(SEED_BASE);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut r = SplitMix64::new(7);
        for _ in 0..1000 {
            let v = r.next_f32();
            assert!((0.0..1.0).contains(&v));
            assert!(r.below(5) < 5);
        }
        assert_eq!(SplitMix64::new(1).below(0), 0);
    }

    #[test]
    fn palette_from_theme_uses_tokens() {
        let theme = Theme::default_theme();
        let pal = EffectPalette::from_theme(&theme);
        assert_eq!(pal.fire_hot, theme.color(Token::Warning));
        assert_eq!(pal.sparkle, theme.color(Token::Success));
        assert_eq!(pal.rainbow, *theme.rainbow());
    }

    #[test]
    fn engine_off_by_default_and_smoke_path_freezes() {
        let e = EffectsEngine::new(ColorCaps::mono());
        assert_eq!(e.mode, EffectMode::Off);
        // The smoke path never calls tick ⇒ frame stays 0 (no animation in smoke).
        assert_eq!(e.frame, 0);
        assert_eq!(e.clock, 0.0);
    }

    #[test]
    fn engine_tick_advances_frame_and_clamps_dt() {
        let mut e = EffectsEngine::new(ColorCaps::detect());
        e.set_mode(EffectMode::Full);
        e.tick(999.0); // huge dt clamps to 0.1
        assert_eq!(e.frame, 1);
        assert!(e.clock <= 0.1 + 1e-6);
    }

    #[test]
    fn engine_demo_runs_then_expires() {
        let mut e = EffectsEngine::new(ColorCaps::detect());
        e.start_demo();
        assert!(e.demo_active());
        for _ in 0..200 {
            e.tick(0.05);
        }
        assert!(!e.demo_active());
    }

    #[test]
    fn effect_mode_parse() {
        assert_eq!(EffectMode::parse("off"), Some(EffectMode::Off));
        assert_eq!(EffectMode::parse(""), Some(EffectMode::Off));
        assert_eq!(EffectMode::parse("SUBTLE"), Some(EffectMode::Subtle));
        assert_eq!(EffectMode::parse("full"), Some(EffectMode::Full));
        assert_eq!(EffectMode::parse("demo"), None);
        assert_eq!(EffectMode::parse("bogus"), None);
    }
}
