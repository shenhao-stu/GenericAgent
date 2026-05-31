//! theme/mod.rs — semantic color tokens (NO hardcoded RGB at call sites) + the
//! multi-theme registry behind the `/theme` LIVE-PREVIEW picker (§5 / §9 / §11).
//!
//! §5 / §9 constraint: "No hardcoded RGB — all via theme tokens." Every widget
//! asks the theme for a [`Token`] and gets a `ratatui::style::Color`. Swapping the
//! active [`Theme`] re-skins the whole UI; the heat ramp, rainbow separator, and
//! status pills all resolve through here so they re-theme for free.
//!
//! THEMES: this ships 6 named palettes (`ga-default`, `nord`, `gruvbox`,
//! `dracula`, `tokyo-night`, `light`) in [`THEMES`]. The `/theme` picker previews
//! one live (assign it to the app's `theme` field as the selection moves) and
//! commits or reverts it (the `theme_preview_revert` deliverable). A theme is a
//! per-token color TABLE + the 7-stop rainbow, so the whole token surface
//! re-skins atomically.

use ratatui::style::Color;

pub mod rainbow;

/// Semantic color tokens (CC's palette fused with Codex/kimi density). Widgets
/// reference these names, never raw colors — so the palette is swappable and the
/// heat ramp / pills re-theme automatically.
#[allow(dead_code)] // mode-pill / user-band tokens are used by widgets per phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Token {
    /// Primary brand accent (the GA "claude-cyan" analogue / orange brand hot).
    Claude,
    /// Default body text.
    Text,
    /// Dimmed / secondary text (reasoning, hints, placeholders).
    Dim,
    /// Success / "calm" heat tier (mint).
    Success,
    /// Warning / "warming" heat tier (amber).
    Warning,
    /// Error / "critical" heat tier (red).
    Error,
    /// Suggestion / informational blue.
    Suggestion,
    /// Plan-mode purple.
    PlanMode,
    /// Auto-accept coral.
    AutoAccept,
    /// Borders / separators (subtle).
    Border,
    /// The charcoal band behind a user message tile (CC `userMessageBackground`).
    UserBand,
    /// Shell-mode accent (hot pink) — the `!cmd` composer border + mark (§4); CC
    /// `bashBorder` rgb(253,93,177).
    ShellAccent,
    /// Hyperlink / IDE accent (CC `ide` rgb(71,130,200)) — links + file refs.
    Ide,
    /// The brand-accent SHIMMER highlight (CC `claudeShimmer` rgb(235,159,127)) —
    /// the lighter pulse over the `claude` accent (spinner/separator sweep).
    ClaudeShimmer,
}

impl Token {
    /// All tokens, for the `every_token_resolves` guard + table construction.
    #[allow(dead_code)] // enumerated by the theme test + the Phase-5 adaptive probe.
    pub const ALL: [Token; 14] = [
        Token::Claude,
        Token::Text,
        Token::Dim,
        Token::Success,
        Token::Warning,
        Token::Error,
        Token::Suggestion,
        Token::PlanMode,
        Token::AutoAccept,
        Token::Border,
        Token::UserBand,
        Token::ShellAccent,
        Token::Ide,
        Token::ClaudeShimmer,
    ];
}

/// A resolved theme = a per-[`Token`] color table + the 7-stop rainbow used for the
/// header separator. The registry [`THEMES`] holds the named palettes; the
/// `/theme` picker swaps the active one (live preview → commit/revert).
#[derive(Debug, Clone)]
pub struct Theme {
    /// Theme identity (shown in the `/theme` picker).
    pub name: &'static str,
    /// The per-token color table (index by `Token as usize`).
    palette: [Color; 14],
    /// The 7-stop ROYGBIV rainbow for the header separator (§5).
    rainbow: [Color; 7],
}

impl Default for Theme {
    fn default() -> Self {
        Theme::ga_default()
    }
}

impl Theme {
    /// Resolve a semantic [`Token`] to a concrete color for this theme (a table
    /// lookup — total, no catch-all gap).
    pub fn color(&self, token: Token) -> Color {
        self.palette[token as usize]
    }

    /// The 7 rainbow stops for the header separator (used by tests + the shimmer).
    #[allow(dead_code)]
    pub fn rainbow(&self) -> &[Color; 7] {
        &self.rainbow
    }

    /// The rainbow color for column `x` of a `width`-wide separator — a smooth
    /// index into the 7 stops. PURE + unit-tested (the separator is a hot path
    /// recomputed every frame; allocation-free per cell).
    pub fn rainbow_at(&self, x: u16, width: u16) -> Color {
        if width <= 1 {
            return self.rainbow[0];
        }
        let span = self.rainbow.len() as u32 - 1; // 6 segments
        let stop = (x as u32 * span) / (width as u32 - 1);
        self.rainbow[stop.min(span) as usize]
    }

    // -- the named palettes ---------------------------------------------------

    /// The default GA theme — the EXACT Claude Code RGB palette (redesign_cc.md
    /// §2.0): brand orange `claude` rgb(215,119,87), CC's success/error/warning/
    /// suggestion semantics, `subtle` rgb(80,80,80) dim, `userMessageBackground`
    /// rgb(58,58,58), `bashBorder` rgb(253,93,177), `ide` rgb(71,130,200), and
    /// `claudeShimmer` rgb(235,159,127). This is what makes a real GA session look
    /// like Claude Code.
    pub fn ga_default() -> Self {
        Theme {
            name: "ga-default",
            palette: [
                Color::Rgb(215, 119, 87),  // Claude — brand orange (CC `claude`)
                Color::Rgb(255, 255, 255), // Text — white (CC `text`)
                Color::Rgb(80, 80, 80),    // Dim — CC `subtle`
                Color::Rgb(78, 186, 101),  // Success — CC `success`
                Color::Rgb(255, 193, 7),   // Warning — CC `warning`
                Color::Rgb(255, 107, 128), // Error — CC `error`
                Color::Rgb(177, 185, 249), // Suggestion — CC `suggestion`
                Color::Rgb(72, 150, 140),  // PlanMode — CC `planMode`
                Color::Rgb(175, 135, 255), // AutoAccept — CC `autoAccept`
                Color::Rgb(68, 72, 82),    // Border — subtle line
                Color::Rgb(58, 58, 58),    // UserBand — CC `userMessageBackground`
                Color::Rgb(253, 93, 177),  // ShellAccent — CC `bashBorder`
                Color::Rgb(71, 130, 200),  // Ide — CC `ide` (links)
                Color::Rgb(235, 159, 127), // ClaudeShimmer — CC `claudeShimmer`
            ],
            rainbow: [
                Color::Rgb(0xff, 0x5c, 0x57),
                Color::Rgb(0xff, 0x9f, 0x43),
                Color::Rgb(0xff, 0xd5, 0x4d),
                Color::Rgb(0x6b, 0xcb, 0x77),
                Color::Rgb(0x4d, 0xa6, 0xff),
                Color::Rgb(0x6c, 0x5c, 0xe7),
                Color::Rgb(0xb3, 0x7f, 0xeb),
            ],
        }
    }

    /// Nord — the arctic, bluish palette.
    pub fn nord() -> Self {
        Theme {
            name: "nord",
            palette: [
                Color::Rgb(0x88, 0xc0, 0xd0), // Claude — frost cyan
                Color::Rgb(0xec, 0xef, 0xf4), // Text — snow
                Color::Rgb(0x6b, 0x73, 0x85), // Dim — slate
                Color::Rgb(0xa3, 0xbe, 0x8c), // Success — green
                Color::Rgb(0xeb, 0xcb, 0x8b), // Warning — yellow
                Color::Rgb(0xbf, 0x61, 0x6a), // Error — aurora red
                Color::Rgb(0x81, 0xa1, 0xc1), // Suggestion — frost blue
                Color::Rgb(0xb4, 0x8e, 0xad), // PlanMode — aurora purple
                Color::Rgb(0xd0, 0x87, 0x70), // AutoAccept — aurora orange
                Color::Rgb(0x43, 0x4c, 0x5e), // Border
                Color::Rgb(0x3b, 0x42, 0x52), // UserBand
                Color::Rgb(0xb4, 0x8e, 0xad), // ShellAccent — aurora purple-pink
                Color::Rgb(0x5e, 0x81, 0xac), // Ide — frost blue (links)
                Color::Rgb(0xa3, 0xd4, 0xe4), // ClaudeShimmer — lighter frost cyan
            ],
            rainbow: [
                Color::Rgb(0xbf, 0x61, 0x6a),
                Color::Rgb(0xd0, 0x87, 0x70),
                Color::Rgb(0xeb, 0xcb, 0x8b),
                Color::Rgb(0xa3, 0xbe, 0x8c),
                Color::Rgb(0x88, 0xc0, 0xd0),
                Color::Rgb(0x81, 0xa1, 0xc1),
                Color::Rgb(0xb4, 0x8e, 0xad),
            ],
        }
    }

    /// Gruvbox — the warm, retro palette.
    pub fn gruvbox() -> Self {
        Theme {
            name: "gruvbox",
            palette: [
                Color::Rgb(0x83, 0xa5, 0x98), // Claude — aqua
                Color::Rgb(0xeb, 0xdb, 0xb2), // Text — cream
                Color::Rgb(0x92, 0x83, 0x74), // Dim — gray
                Color::Rgb(0xb8, 0xbb, 0x26), // Success — green
                Color::Rgb(0xfa, 0xbd, 0x2f), // Warning — yellow
                Color::Rgb(0xfb, 0x49, 0x34), // Error — red
                Color::Rgb(0x83, 0xa5, 0x98), // Suggestion — blue-aqua
                Color::Rgb(0xd3, 0x86, 0x9b), // PlanMode — purple
                Color::Rgb(0xfe, 0x80, 0x19), // AutoAccept — orange
                Color::Rgb(0x50, 0x49, 0x45), // Border
                Color::Rgb(0x3c, 0x38, 0x36), // UserBand
                Color::Rgb(0xd3, 0x86, 0x9b), // ShellAccent — pink
                Color::Rgb(0x45, 0x85, 0x88), // Ide — blue-aqua (links)
                Color::Rgb(0xa9, 0xc6, 0xbc), // ClaudeShimmer — lighter aqua
            ],
            rainbow: [
                Color::Rgb(0xfb, 0x49, 0x34),
                Color::Rgb(0xfe, 0x80, 0x19),
                Color::Rgb(0xfa, 0xbd, 0x2f),
                Color::Rgb(0xb8, 0xbb, 0x26),
                Color::Rgb(0x83, 0xa5, 0x98),
                Color::Rgb(0x45, 0x85, 0x88),
                Color::Rgb(0xd3, 0x86, 0x9b),
            ],
        }
    }

    /// Dracula — the dark, vivid palette.
    pub fn dracula() -> Self {
        Theme {
            name: "dracula",
            palette: [
                Color::Rgb(0x8b, 0xe9, 0xfd), // Claude — cyan
                Color::Rgb(0xf8, 0xf8, 0xf2), // Text — foreground
                Color::Rgb(0x62, 0x72, 0xa4), // Dim — comment
                Color::Rgb(0x50, 0xfa, 0x7b), // Success — green
                Color::Rgb(0xf1, 0xfa, 0x8c), // Warning — yellow
                Color::Rgb(0xff, 0x55, 0x55), // Error — red
                Color::Rgb(0xbd, 0x93, 0xf9), // Suggestion — purple
                Color::Rgb(0xbd, 0x93, 0xf9), // PlanMode — purple
                Color::Rgb(0xff, 0xb8, 0x6c), // AutoAccept — orange
                Color::Rgb(0x44, 0x47, 0x5a), // Border
                Color::Rgb(0x28, 0x2a, 0x36), // UserBand
                Color::Rgb(0xff, 0x79, 0xc6), // ShellAccent — pink
                Color::Rgb(0x62, 0x72, 0xa4), // Ide — comment-blue (links)
                Color::Rgb(0xb4, 0xf2, 0xff), // ClaudeShimmer — lighter cyan
            ],
            rainbow: [
                Color::Rgb(0xff, 0x55, 0x55),
                Color::Rgb(0xff, 0xb8, 0x6c),
                Color::Rgb(0xf1, 0xfa, 0x8c),
                Color::Rgb(0x50, 0xfa, 0x7b),
                Color::Rgb(0x8b, 0xe9, 0xfd),
                Color::Rgb(0xbd, 0x93, 0xf9),
                Color::Rgb(0xff, 0x79, 0xc6),
            ],
        }
    }

    /// Tokyo Night — the muted, modern palette.
    pub fn tokyo_night() -> Self {
        Theme {
            name: "tokyo-night",
            palette: [
                Color::Rgb(0x7d, 0xcf, 0xff), // Claude — cyan
                Color::Rgb(0xc0, 0xca, 0xf5), // Text — foreground
                Color::Rgb(0x56, 0x5f, 0x89), // Dim — comment
                Color::Rgb(0x9e, 0xce, 0x6a), // Success — green
                Color::Rgb(0xe0, 0xaf, 0x68), // Warning — yellow
                Color::Rgb(0xf7, 0x76, 0x8e), // Error — red
                Color::Rgb(0x7a, 0xa2, 0xf7), // Suggestion — blue
                Color::Rgb(0xbb, 0x9a, 0xf7), // PlanMode — purple
                Color::Rgb(0xff, 0x9e, 0x64), // AutoAccept — orange
                Color::Rgb(0x3b, 0x42, 0x61), // Border
                Color::Rgb(0x24, 0x28, 0x3b), // UserBand
                Color::Rgb(0xbb, 0x9a, 0xf7), // ShellAccent — magenta-purple
                Color::Rgb(0x7a, 0xa2, 0xf7), // Ide — blue (links)
                Color::Rgb(0xb4, 0xe1, 0xff), // ClaudeShimmer — lighter cyan
            ],
            rainbow: [
                Color::Rgb(0xf7, 0x76, 0x8e),
                Color::Rgb(0xff, 0x9e, 0x64),
                Color::Rgb(0xe0, 0xaf, 0x68),
                Color::Rgb(0x9e, 0xce, 0x6a),
                Color::Rgb(0x7d, 0xcf, 0xff),
                Color::Rgb(0x7a, 0xa2, 0xf7),
                Color::Rgb(0xbb, 0x9a, 0xf7),
            ],
        }
    }

    /// Light — a high-contrast light-ground palette (Codex's `NO_COLOR`-adjacent
    /// daytime option; dark text on the terminal's light background).
    pub fn light() -> Self {
        Theme {
            name: "light",
            palette: [
                Color::Rgb(0x00, 0x87, 0x9b), // Claude — teal
                Color::Rgb(0x1a, 0x1a, 0x1a), // Text — near-black
                Color::Rgb(0x70, 0x70, 0x70), // Dim — grey
                Color::Rgb(0x1b, 0x80, 0x3b), // Success — green
                Color::Rgb(0xb0, 0x6a, 0x00), // Warning — amber
                Color::Rgb(0xc6, 0x28, 0x28), // Error — red
                Color::Rgb(0x15, 0x65, 0xc0), // Suggestion — blue
                Color::Rgb(0x6a, 0x1b, 0x9a), // PlanMode — purple
                Color::Rgb(0xd8, 0x4f, 0x2a), // AutoAccept — coral
                Color::Rgb(0xc7, 0xc7, 0xc7), // Border — light line
                Color::Rgb(0xe8, 0xe8, 0xe8), // UserBand — light band
                Color::Rgb(0xc2, 0x18, 0x5b), // ShellAccent — magenta
                Color::Rgb(0x15, 0x65, 0xc0), // Ide — blue (links)
                Color::Rgb(0x00, 0xa5, 0xbb), // ClaudeShimmer — lighter teal
            ],
            rainbow: [
                Color::Rgb(0xc6, 0x28, 0x28),
                Color::Rgb(0xd8, 0x4f, 0x2a),
                Color::Rgb(0xb0, 0x6a, 0x00),
                Color::Rgb(0x1b, 0x80, 0x3b),
                Color::Rgb(0x00, 0x87, 0x9b),
                Color::Rgb(0x15, 0x65, 0xc0),
                Color::Rgb(0x6a, 0x1b, 0x9a),
            ],
        }
    }
}

/// The named theme constructors, in `/theme` picker order. The picker lists these;
/// `by_name` resolves a `/theme <name>` argument; the live preview swaps the
/// active [`Theme`] as the selection moves.
pub const THEME_BUILDERS: &[(&str, fn() -> Theme)] = &[
    ("ga-default", Theme::ga_default),
    ("nord", Theme::nord),
    ("gruvbox", Theme::gruvbox),
    ("dracula", Theme::dracula),
    ("tokyo-night", Theme::tokyo_night),
    ("light", Theme::light),
];

/// All theme names, in picker order. PURE.
pub fn all_names() -> Vec<&'static str> {
    THEME_BUILDERS.iter().map(|(n, _)| *n).collect()
}

/// Build a theme by name (case-insensitive). `None` for an unknown name (the
/// picker / `/theme <name>` then keeps the current theme). PURE-ish.
pub fn by_name(name: &str) -> Option<Theme> {
    let n = name.trim().to_ascii_lowercase();
    THEME_BUILDERS
        .iter()
        .find(|(tn, _)| tn.eq_ignore_ascii_case(&n))
        .map(|(_, build)| build())
}

/// The 0-based index of a theme name in picker order (to seed the `/theme` picker
/// selection on the CURRENT theme). PURE.
pub fn index_of(name: &str) -> Option<usize> {
    THEME_BUILDERS.iter().position(|(tn, _)| tn.eq_ignore_ascii_case(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rainbow_spans_all_stops() {
        let t = Theme::ga_default();
        assert_eq!(t.rainbow_at(0, 80), t.rainbow()[0]);
        assert_eq!(t.rainbow_at(79, 80), t.rainbow()[6]);
        assert_eq!(t.rainbow_at(0, 1), t.rainbow()[0]);
        assert_eq!(t.rainbow_at(0, 0), t.rainbow()[0]);
        let mid = t.rainbow_at(40, 80);
        assert!(t.rainbow().contains(&mid));
    }

    #[test]
    fn every_token_resolves_for_every_theme() {
        for (_, build) in THEME_BUILDERS {
            let t = build();
            for tok in Token::ALL {
                // Resolution is total (a table lookup; no panic / gap).
                let _ = t.color(tok);
            }
        }
    }

    /// The ga-default theme carries the EXACT Claude Code RGB palette (§2.0) — so a
    /// real GA session renders in CC's brand colors, not the old ad-hoc cyan set.
    #[test]
    fn ga_default_uses_cc_rgb_palette() {
        let t = Theme::ga_default();
        assert_eq!(t.color(Token::Claude), Color::Rgb(215, 119, 87), "CC brand orange");
        assert_eq!(t.color(Token::Success), Color::Rgb(78, 186, 101));
        assert_eq!(t.color(Token::Error), Color::Rgb(255, 107, 128));
        assert_eq!(t.color(Token::Warning), Color::Rgb(255, 193, 7));
        assert_eq!(t.color(Token::Suggestion), Color::Rgb(177, 185, 249));
        assert_eq!(t.color(Token::Dim), Color::Rgb(80, 80, 80), "CC subtle");
        assert_eq!(t.color(Token::Text), Color::Rgb(255, 255, 255));
        assert_eq!(t.color(Token::Ide), Color::Rgb(71, 130, 200), "CC ide/link");
        assert_eq!(t.color(Token::ShellAccent), Color::Rgb(253, 93, 177), "CC bashBorder");
        assert_eq!(
            t.color(Token::UserBand),
            Color::Rgb(58, 58, 58),
            "CC userMessageBackground (the §2.1 user band)"
        );
        assert_eq!(t.color(Token::ClaudeShimmer), Color::Rgb(235, 159, 127));
    }

    /// GATE (§9): the registry ships at least the 6 required named themes, each with a
    /// UNIQUE name and a full 7-stop rainbow (the effects separator depends on it).
    #[test]
    fn theme_count_at_least_6() {
        let names = all_names();
        assert!(names.len() >= 6, "expected >=6 themes, got {}", names.len());
        for required in ["ga-default", "nord", "gruvbox", "dracula", "tokyo-night", "light"] {
            assert!(names.contains(&required), "missing required theme {required}");
            let t = by_name(required).unwrap_or_else(|| panic!("{required} resolves"));
            assert_eq!(t.rainbow().len(), 7, "{required} needs 7 rainbow stops");
        }
        // Names are unique (no duplicate registry entries).
        let mut sorted = names.clone();
        sorted.sort_unstable();
        let n = sorted.len();
        sorted.dedup();
        assert_eq!(sorted.len(), n, "theme names must be unique");
    }

    /// THE deliverable test: the `/theme` picker LIVE-PREVIEWS then commits or
    /// reverts. We model exactly what the dispatcher does: stash the original
    /// theme, swap to a preview as the selection moves, and either KEEP the preview
    /// (commit) or restore the original (revert) — proving the swap is lossless and
    /// the names round-trip through the registry.
    #[test]
    fn theme_preview_revert() {
        // The registry exposes all 6 named themes, ga-default first.
        let names = all_names();
        assert!(names.len() >= 6);
        assert_eq!(names[0], "ga-default");
        assert!(names.contains(&"nord"));
        assert!(names.contains(&"dracula"));

        // Start on the default; remember it as the revert target.
        let original = Theme::ga_default();
        let original_name = original.name;
        let original_claude = original.color(Token::Claude);

        // --- PREVIEW: the picker moves the selection onto "nord" → swap live. ---
        let preview = by_name("nord").expect("nord resolves");
        assert_eq!(preview.name, "nord");
        // The live theme is now nord (a different Claude accent than ga-default).
        assert_ne!(preview.color(Token::Claude), original_claude);

        // --- REVERT (Esc): restore the original theme; the preview is discarded. -
        let reverted = by_name(original_name).expect("original resolves");
        assert_eq!(reverted.name, "ga-default");
        assert_eq!(reverted.color(Token::Claude), original_claude);

        // --- COMMIT (Enter): keep the previewed theme. -------------------------
        let committed = by_name("dracula").expect("dracula resolves");
        assert_eq!(committed.name, "dracula");
        // index_of seeds the picker selection on the committed theme.
        assert_eq!(index_of("dracula"), Some(3));
        assert_eq!(index_of("ga-default"), Some(0));

        // Unknown name → no theme (picker keeps current).
        assert!(by_name("not-a-theme").is_none());
        // Case-insensitive resolution.
        assert!(by_name("NORD").is_some());
    }
}
