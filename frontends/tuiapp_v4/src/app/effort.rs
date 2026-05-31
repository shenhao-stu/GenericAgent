//! app/effort.rs — the `/effort` reasoning-effort level + slider model (redesign_cc.md §3).
//!
//! GA's backend (llmcore.py:540-559) reads `reasoning_effort` on
//! `agent.llmclient.backend` for BOTH NativeOAISession (`payload.reasoning_effort`
//! / `reasoning.effort`) AND NativeClaudeSession (`output_config.effort`, where
//! the Claude max tier is spelled `xhigh`). tui_v4 surfaces FIVE slider stops —
//! `low  medium  high  xhigh  max` — and FORWARDS the chosen level to the bridge as
//! `/session.reasoning_effort=<backend>` (the GA core's hot-reload path,
//! agentmain.py:122, does `setattr(backend, "reasoning_effort", v)` live). The only
//! transform is the slider's `max` stop → the backend's `xhigh` value (CC shows a
//! friendly "max"; GA's highest backend tier is `xhigh`); `low/medium/high/xhigh`
//! pass straight through.
//!
//! All the load-bearing logic (the stop list, the slider-label↔backend-value
//! mapping, the marker navigation) is PURE + unit-tested here — the overlay
//! renderer in `components::overlay` only PAINTS the model this produces.

/// One reasoning-effort stop on the slider (redesign_cc.md §3). Ordered
/// `Faster → Smarter`: [`Low`] is the leftmost (fastest), [`Max`] the rightmost
/// (smartest). The slider's `▲` marker sits on the selected level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
    /// The backend's `xhigh` tier (NativeClaude maps this to `output_config.effort`
    /// directly; NativeOAI passes it through as the reasoning effort).
    XHigh,
    /// The slider's friendly top stop (mirrors CC's "max"). Maps to the backend
    /// value `xhigh` — there is no distinct `max` tier in llmcore.py, so `max` and
    /// `xhigh` forward the SAME backend value; the two stops differ only as labels.
    Max,
}

impl ReasoningEffort {
    /// The slider stops, left (Faster) → right (Smarter). The marker index into
    /// this array IS the slider position; nav clamps within `0..LEVELS.len()`.
    pub const LEVELS: [ReasoningEffort; 5] = [
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::High,
        ReasoningEffort::XHigh,
        ReasoningEffort::Max,
    ];

    /// The slider LABEL shown on the track + footer (what CC's clip shows). The
    /// `max` stop reads "max" even though it forwards `xhigh` to the backend.
    pub fn label(self) -> &'static str {
        match self {
            ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::High => "high",
            ReasoningEffort::XHigh => "xhigh",
            ReasoningEffort::Max => "max",
        }
    }

    /// The BACKEND value forwarded as `/session.reasoning_effort=<value>` (the
    /// `setattr(backend, "reasoning_effort", value)` argument GA hot-reloads). The
    /// only transform is `max → xhigh` (redesign_cc.md §3); every other stop maps
    /// to its own name. PURE.
    pub fn backend_value(self) -> &'static str {
        match self {
            ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::High => "high",
            ReasoningEffort::XHigh => "xhigh",
            // The slider `max` stop forwards the backend's highest tier, `xhigh`.
            ReasoningEffort::Max => "xhigh",
        }
    }

    /// The index of this level in [`Self::LEVELS`] (the slider marker position).
    pub fn index(self) -> usize {
        Self::LEVELS.iter().position(|&l| l == self).unwrap_or(0)
    }

    /// The level at slider position `idx`, clamped into range. PURE.
    pub fn from_index(idx: usize) -> ReasoningEffort {
        let i = idx.min(Self::LEVELS.len() - 1);
        Self::LEVELS[i]
    }

    /// Parse a typed `/effort <level>` argument (case-insensitive). Accepts every
    /// slider label (`low`/`medium`/`high`/`xhigh`/`max`); a couple of friendly
    /// aliases are tolerated (`med` → medium, `xh` → xhigh). Returns `None` for an
    /// unknown word so the dispatcher can fall back to opening the slider. PURE.
    pub fn parse(s: &str) -> Option<ReasoningEffort> {
        match s.trim().to_ascii_lowercase().as_str() {
            "low" | "l" => Some(ReasoningEffort::Low),
            "medium" | "med" | "m" => Some(ReasoningEffort::Medium),
            "high" | "h" => Some(ReasoningEffort::High),
            "xhigh" | "xh" | "x" => Some(ReasoningEffort::XHigh),
            "max" => Some(ReasoningEffort::Max),
            _ => None,
        }
    }

    /// The full `/session.<k>=<v>` line this level forwards to the GA core (the
    /// canonical human-readable form; the forwarded frame is built from
    /// [`Self::command_name`], and tests assert the two agree). The core intercepts
    /// `/session.reasoning_effort=<v>` and does `setattr(backend, "reasoning_effort",
    /// v)` live (agentmain.py:122). PURE.
    #[allow(dead_code)] // canonical form asserted by the forwarding tests.
    pub fn session_command(self) -> String {
        format!("/session.reasoning_effort={}", self.backend_value())
    }

    /// The `Command{name}` value to forward over the bridge for THIS level. The
    /// bridge's generic `Command` handler reconstructs `"/" + name + " " + args` and
    /// submits it (ga_bridge.py:822-825), so the name is the slash line WITHOUT the
    /// leading `/`: `session.reasoning_effort=<backend>` (max→xhigh). Forwarding it
    /// with empty args yields exactly `/session.reasoning_effort=<backend>`. PURE.
    pub fn command_name(self) -> String {
        format!("session.reasoning_effort={}", self.backend_value())
    }
}

/// The `/effort` slider overlay model (redesign_cc.md §3): a `▲` marker over the
/// `low medium high xhigh max` stops. `marker` is the index into
/// [`ReasoningEffort::LEVELS`]; ←/→ moves it (clamped, no wrap), Enter applies the
/// `marker` level, Esc cancels (the caller leaves the live level untouched). PURE
/// — the overlay renderer only paints `marker`/`current`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffortSlider {
    /// The slider marker position (index into [`ReasoningEffort::LEVELS`]).
    pub marker: usize,
    /// The level that was LIVE when the slider opened (so the renderer can mark the
    /// applied stop distinctly from the one the marker is hovering, and so a cancel
    /// is a true no-op). PURE state; never mutated by nav.
    pub current: ReasoningEffort,
}

impl EffortSlider {
    /// Open the slider seeded on `current` (the live effort level). The marker
    /// starts on the current level so Enter-without-moving is a no-op.
    pub fn new(current: ReasoningEffort) -> Self {
        EffortSlider { marker: current.index(), current }
    }

    /// Move the `▲` marker by `delta` (+right/Smarter, -left/Faster), clamped to
    /// the valid stop range (no wrap — mirrors the picker's saturating nav). PURE.
    pub fn move_marker(&mut self, delta: isize) {
        let len = ReasoningEffort::LEVELS.len() as isize;
        let next = (self.marker as isize + delta).clamp(0, len - 1);
        self.marker = next as usize;
    }

    /// The level the marker currently rests on (what Enter would APPLY). PURE.
    pub fn selected(&self) -> ReasoningEffort {
        ReasoningEffort::from_index(self.marker)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE deliverable test (redesign_cc.md §3): the slider stops are
    /// `low medium high xhigh max`, the labels match, and the ONLY backend-value
    /// transform is `max → xhigh` (every other stop passes through). The forwarded
    /// `/session.reasoning_effort=` line carries the BACKEND value.
    #[test]
    fn effort_levels_and_mapping() {
        // Five stops, ordered Faster → Smarter.
        assert_eq!(ReasoningEffort::LEVELS.len(), 5);
        let labels: Vec<&str> = ReasoningEffort::LEVELS.iter().map(|l| l.label()).collect();
        assert_eq!(labels, ["low", "medium", "high", "xhigh", "max"]);

        // Backend values: low/medium/high/xhigh pass through; `max` → `xhigh`.
        assert_eq!(ReasoningEffort::Low.backend_value(), "low");
        assert_eq!(ReasoningEffort::Medium.backend_value(), "medium");
        assert_eq!(ReasoningEffort::High.backend_value(), "high");
        assert_eq!(ReasoningEffort::XHigh.backend_value(), "xhigh");
        // The load-bearing mapping: the slider `max` forwards the backend `xhigh`.
        assert_eq!(ReasoningEffort::Max.backend_value(), "xhigh");

        // The forwarded session command carries the backend value (max → xhigh).
        assert_eq!(
            ReasoningEffort::Max.session_command(),
            "/session.reasoning_effort=xhigh"
        );
        assert_eq!(
            ReasoningEffort::High.session_command(),
            "/session.reasoning_effort=high"
        );

        // index ↔ from_index round-trips across every stop.
        for (i, &lvl) in ReasoningEffort::LEVELS.iter().enumerate() {
            assert_eq!(lvl.index(), i);
            assert_eq!(ReasoningEffort::from_index(i), lvl);
        }
        // from_index clamps an out-of-range index to the top stop.
        assert_eq!(ReasoningEffort::from_index(99), ReasoningEffort::Max);

        // Parsing accepts every label (case-insensitive) + a few aliases.
        assert_eq!(ReasoningEffort::parse("HIGH"), Some(ReasoningEffort::High));
        assert_eq!(ReasoningEffort::parse(" max "), Some(ReasoningEffort::Max));
        assert_eq!(ReasoningEffort::parse("xhigh"), Some(ReasoningEffort::XHigh));
        assert_eq!(ReasoningEffort::parse("med"), Some(ReasoningEffort::Medium));
        assert_eq!(ReasoningEffort::parse("nope"), None);
    }

    /// THE slider-nav deliverable test (redesign_cc.md §3): ←/→ moves the `▲`
    /// marker, clamped (no wrap); the marker seeds on the current level; `selected`
    /// reports the hovered stop.
    #[test]
    fn effort_slider_nav() {
        // Seeds on the current level (so Enter-without-moving is a no-op).
        let mut s = EffortSlider::new(ReasoningEffort::High);
        assert_eq!(s.marker, ReasoningEffort::High.index()); // == 2
        assert_eq!(s.selected(), ReasoningEffort::High);

        // → moves toward Smarter; clamps at `max` (no wrap past the right edge).
        s.move_marker(1);
        assert_eq!(s.selected(), ReasoningEffort::XHigh);
        s.move_marker(1);
        assert_eq!(s.selected(), ReasoningEffort::Max);
        s.move_marker(1); // already at the right edge → clamp.
        assert_eq!(s.selected(), ReasoningEffort::Max);

        // ← moves toward Faster; clamps at `low` (no wrap past the left edge).
        let mut s = EffortSlider::new(ReasoningEffort::Medium);
        s.move_marker(-1);
        assert_eq!(s.selected(), ReasoningEffort::Low);
        s.move_marker(-1); // already at the left edge → clamp.
        assert_eq!(s.selected(), ReasoningEffort::Low);

        // The `current` (applied) level is never mutated by nav — only Enter (in the
        // dispatcher) commits a new live level; a cancel leaves `current` intact.
        let mut s = EffortSlider::new(ReasoningEffort::Low);
        s.move_marker(2);
        assert_eq!(s.current, ReasoningEffort::Low, "nav never mutates the applied level");
        assert_eq!(s.selected(), ReasoningEffort::High);
    }
}
