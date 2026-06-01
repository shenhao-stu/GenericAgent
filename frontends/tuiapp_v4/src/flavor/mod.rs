//! flavor/mod.rs — the cockpit "soul" (N4 / §9): custom spinner, heat ramp,
//! the 34-word gerund pool, pet faces (5 styles × 4 heat × 4 frames), rotating
//! tips, and the OSC0/OSC-21337 terminal-title/tab-status payloads.
//!
//! IDENTITY: this is NOT Claude Code's `✻`/`✷`. The default is the arc cycle
//! `◜◠◝◞◡◟`; a braille set `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏` (the tui_v3 soul) and a pulse set
//! `·✢✳✶✻✽` are the alternates. As a turn runs longer the spinner color
//! escalates through the "patience heat" ramp (mint <20s, amber <60s, orange
//! <180s, red ≥180s) returned as a THEME TOKEN (no hardcoded color).
//!
//! DETERMINISM CONTRACT (the load-bearing rule): every rotation here is a PURE
//! function of an integer index (the 0.1s tick counter, or a turn index) — NO
//! randomness and NO wall-clock reads inside this module. The same index always
//! yields the same glyph / gerund / tip / pet frame, so a redraw never jitters
//! the word and the `gerund_rotation_deterministic` test can pin exact output.

use crate::theme::Token;

/// The selectable spinner aesthetics. Default = [`SpinnerStyle::Arc`] (our
/// distinct mark, never the CC asterisk). Switched via the `/emoji`-style picker.
// Braille/Pulse + the name pickers are selected by the Phase-3 `/emoji` overlay.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpinnerStyle {
    /// `◜◠◝◞◡◟` — the distinct tui_v4 arc (default; NOT the CC asterisk).
    #[default]
    Arc,
    /// `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏` — the tui_v3 braille soul.
    Braille,
    /// `·✢✳✶✻✽` — pulse (in/out), the alternate aesthetic.
    Pulse,
}

/// The arc frames (our default spinner identity).
pub const ARC_FRAMES: &[char] = &['◜', '◠', '◝', '◞', '◡', '◟'];
/// The braille frames (tui_v3 soul).
pub const BRAILLE_FRAMES: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
/// The pulse base glyphs; the live sequence is forward then reversed (in/out).
pub const PULSE_GLYPHS: &[char] = &['·', '✢', '✳', '✶', '✻', '✽'];

#[allow(dead_code)] // from_name/name feed the Phase-3 /emoji style picker.
impl SpinnerStyle {
    /// The frame sequence for this style. For [`SpinnerStyle::Pulse`] this is the
    /// forward-then-reversed in/out cycle.
    pub fn frames(self) -> Vec<char> {
        match self {
            SpinnerStyle::Arc => ARC_FRAMES.to_vec(),
            SpinnerStyle::Braille => BRAILLE_FRAMES.to_vec(),
            SpinnerStyle::Pulse => {
                let mut v = PULSE_GLYPHS.to_vec();
                let mut rev: Vec<char> = PULSE_GLYPHS.iter().rev().copied().collect();
                v.append(&mut rev);
                v
            }
        }
    }

    /// The glyph for an integer frame counter (deterministic, no clock). At a
    /// 0.1s tick, `frame = elapsed_ms / 100`. A non-negative modulo wraps safely.
    pub fn glyph(self, frame: u64) -> char {
        let frames = self.frames();
        let n = frames.len() as u64;
        frames[(frame % n) as usize]
    }

    /// Parse a spinner style from a `/emoji`-style name (case-insensitive). Used
    /// by the style picker; unknown names keep the current style (None).
    pub fn from_name(name: &str) -> Option<SpinnerStyle> {
        match name.trim().to_ascii_lowercase().as_str() {
            "arc" => Some(SpinnerStyle::Arc),
            "braille" => Some(SpinnerStyle::Braille),
            "pulse" => Some(SpinnerStyle::Pulse),
            _ => None,
        }
    }

    /// The style's display name (for the picker / status line).
    pub fn name(self) -> &'static str {
        match self {
            SpinnerStyle::Arc => "arc",
            SpinnerStyle::Braille => "braille",
            SpinnerStyle::Pulse => "pulse",
        }
    }
}

// ---------------------------------------------------------------------------
// Heat ramp (tui_v3 `_heat`).
// ---------------------------------------------------------------------------

/// A heat tier index 0..=3 (calm → warming → hot → critical).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeatTier {
    Calm,
    Warming,
    Hot,
    Critical,
}

impl HeatTier {
    /// The 0-based tier index (used to index the pet pools).
    pub fn index(self) -> usize {
        match self {
            HeatTier::Calm => 0,
            HeatTier::Warming => 1,
            HeatTier::Hot => 2,
            HeatTier::Critical => 3,
        }
    }
}

/// Tier thresholds in **milliseconds** (verbatim from tui_v3: 20s / 60s / 180s).
pub const HEAT_THRESHOLDS_MS: [u64; 3] = [20_000, 60_000, 180_000];

/// The heat tier for an elapsed time (ms). Saturating so a not-yet-started turn
/// never underflows.
pub fn heat_tier(elapsed_ms: u64) -> HeatTier {
    if elapsed_ms < HEAT_THRESHOLDS_MS[0] {
        HeatTier::Calm
    } else if elapsed_ms < HEAT_THRESHOLDS_MS[1] {
        HeatTier::Warming
    } else if elapsed_ms < HEAT_THRESHOLDS_MS[2] {
        HeatTier::Hot
    } else {
        HeatTier::Critical
    }
}

/// The heat **color token** for an elapsed time (ms) — the tui_v3 `_heat`
/// analogue. The four tiers map onto the closest semantic tokens so the ramp
/// re-themes with every palette:
///   calm → Success (mint) · warming → Warning (amber)
///   hot → Claude (orange brand) · critical → Error (red)
pub fn heat_token(elapsed_ms: u64) -> Token {
    match heat_tier(elapsed_ms) {
        HeatTier::Calm => Token::Success,
        HeatTier::Warming => Token::Warning,
        HeatTier::Hot => Token::Claude,
        HeatTier::Critical => Token::Error,
    }
}

/// Whether the critical (red) tier also renders bold (tui_v3 prefixed `\x1b[1m`).
pub fn heat_bold(elapsed_ms: u64) -> bool {
    matches!(heat_tier(elapsed_ms), HeatTier::Critical)
}

// ---------------------------------------------------------------------------
// Gerund pool — bilingual (Q12 / C5 F9). The base 34 are ported verbatim from
// tui_v3 `SPINNER_GERUNDS`; the C5 appendix appends 7 genuinely-NEW eggs (the
// appendix's other "new" words — Sleuthing/Reticulating/Spelunking/Conjuring/
// Marinating/Untangling — already live in the base set, so appending them would
// duplicate; only the 7 not-already-present are added). `GERUNDS_ZH` is the
// PARALLEL Simplified-Chinese pool, SAME length (the `gerunds_parity` guard).
// ---------------------------------------------------------------------------

/// The English gerund pool (ported from tui_v3 `SPINNER_GERUNDS` + 7 C5-appendix
/// eggs). Rotates every ~6s so a long wait feels alive. `GERUNDS_ZH` mirrors it
/// index-for-index; the two MUST stay the same length (`gerunds_parity`).
pub const GERUNDS: &[&str] = &[
    "Pondering",
    "Reticulating",
    "Sleuthing",
    "Hatching",
    "Pouncing",
    "Brewing",
    "Sharpening",
    "Untangling",
    "Compiling",
    "Unraveling",
    "Distilling",
    "Calibrating",
    "Marinating",
    "Conjuring",
    "Foraging",
    "Spelunking",
    "Synthesizing",
    "Refactoring thoughts",
    "Tracing breadcrumbs",
    "Following the rabbit hole",
    "Routing",
    "Threading",
    "Polling",
    "Spinning",
    "Hooking",
    "Patching",
    "Caching",
    "Yielding",
    "Hydrating",
    "Folding",
    "Streaming",
    "Resolving",
    "Reaping",
    "Tuning",
    // -- C5-appendix eggs (the 7 not already present above) -------------------
    "Percolating",
    "Bamboozling",
    "Galaxy-braining",
    "Vibing",
    "Summoning daemons",
    "Bikeshedding",
    "Yak-shaving",
];

/// The Simplified-Chinese gerund pool — PARALLEL to [`GERUNDS`] (same length,
/// translated index-for-index). A zh user sees `沉思中…` where an en user sees
/// `Pondering…`. The `gerunds_parity` test pins `GERUNDS.len() == GERUNDS_ZH.len()`.
pub const GERUNDS_ZH: &[&str] = &[
    "沉思中",       // Pondering
    "织网中",       // Reticulating
    "探案中",       // Sleuthing
    "孵化中",       // Hatching
    "扑击中",       // Pouncing
    "酝酿中",       // Brewing
    "磨刀中",       // Sharpening
    "解结中",       // Untangling
    "编译中",       // Compiling
    "抽丝剥茧中",   // Unraveling
    "提炼中",       // Distilling
    "校准中",       // Calibrating
    "腌制中",       // Marinating
    "施法中",       // Conjuring
    "觅食中",       // Foraging
    "探洞中",       // Spelunking
    "合成中",       // Synthesizing
    "重构思路中",   // Refactoring thoughts
    "循迹面包屑",   // Tracing breadcrumbs
    "钻兔子洞中",   // Following the rabbit hole
    "路由中",       // Routing
    "穿线中",       // Threading
    "轮询中",       // Polling
    "运转中",       // Spinning
    "挂钩中",       // Hooking
    "打补丁中",     // Patching
    "缓存中",       // Caching
    "让步中",       // Yielding
    "注水中",       // Hydrating
    "折叠中",       // Folding
    "流式中",       // Streaming
    "解析中",       // Resolving
    "回收中",       // Reaping
    "调优中",       // Tuning
    // -- C5-appendix eggs (parallel to the EN tail) ---------------------------
    "渗滤中",       // Percolating
    "谋划中",       // Bamboozling
    "烧脑中",       // Galaxy-braining
    "找感觉中",     // Vibing
    "召唤守护进程", // Summoning daemons
    "纠结细节中",   // Bikeshedding
    "剃牦牛中",     // Yak-shaving
];

/// Number of 0.1s ticks per gerund step (~6s rotation: 60 ticks × 0.1s = 6s).
pub const GERUND_TICKS_PER_STEP: u64 = 60;

/// The gerund pool for a language. PURE.
pub fn gerunds_for(lang: Lang) -> &'static [&'static str] {
    match lang {
        Lang::En => GERUNDS,
        Lang::Zh => GERUNDS_ZH,
    }
}

/// Pick a gerund for a slow rotation by an explicit STEP INDEX, in `lang`.
/// Deterministic: `gerund_at(lang, k)` is `pool[k % pool.len()]`. This is the
/// pure core the `gerund_rotation_deterministic` test pins, and what a frozen
/// turn-indexed readout (e.g. the done-line) calls directly.
pub fn gerund_at(lang: Lang, step: u64) -> &'static str {
    let pool = gerunds_for(lang);
    pool[(step % pool.len() as u64) as usize]
}

/// Pick a gerund for the 0.1s tick clock, in `lang`: rotates one word every ~6s.
/// `tick` is the 0.1s counter, so the step is `tick / 60`. Deterministic (no
/// wall-clock, no randomness) — a redraw within the same 6s window shows the
/// SAME word, and the pool is selected by the interface language (Q12).
pub fn gerund(lang: Lang, tick: u64) -> &'static str {
    gerund_at(lang, tick / GERUND_TICKS_PER_STEP)
}

// ---------------------------------------------------------------------------
// Pet faces — 5 styles × 4 heat tiers × 4 blink frames (tui_v3 `_PETS_*`,
// plus a 5th "fox" style to reach the spec's 5-style requirement).
// ---------------------------------------------------------------------------

/// The selectable pet styles (the `/emoji` faces). `Off` hides the pet.
// All 5 styles ship; the non-default ones are chosen via the Phase-3 picker.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PetStyle {
    /// `ʕ•ᴥ•ʔ` bear (the tui_v3 default kaomoji; now an opt-in `/emoji` style).
    Bear,
    /// `=^•.•^=` cat.
    Cat,
    /// `[•.•]` bracketed dot-eyes.
    Dot,
    /// `(•‿•)` kaomoji.
    Unicode,
    /// `•ᴗ•ฅ` fox (the 5th style — paw + ears).
    Fox,
    /// Hide the pet entirely. DEFAULT (redesign_cc.md §2.6: "NOT emoji pet by
    /// default — kaomoji pet OK as an opt-in /emoji style"). So the out-of-the-box
    /// spinner is the clean `<arc> <gerund>… <elapsed>` with NO pet; Bear/Cat/Dot/
    /// Unicode/Fox are still selectable via `/emoji`.
    #[default]
    Off,
}

/// Bear faces: 4 heat tiers × 4 blink frames (mood escalates calm→stressed).
pub const PETS_BEAR: [[&str; 4]; 4] = [
    ["ʕ•ᴥ•ʔ", "ʕ-ᴥ-ʔ", "ʕ•ᴥ•ʔ", "ʕ•ᴥ-ʔ"],
    ["ʕoᴥoʔ", "ʕoᴥ-ʔ", "ʕoᴥoʔ", "ʕ-ᴥoʔ"],
    ["ʕ-ᴥ-ʔ", "ʕ-ᴥ-ʔ", "ʕ~ᴥ~ʔ", "ʕ-ᴥ-ʔ"],
    ["ʕ>ᴥ<ʔ", "ʕ@ᴥ@ʔ", "ʕ>ᴥ<ʔ", "ʕTᴥTʔ"],
];

/// Cat faces.
pub const PETS_CAT: [[&str; 4]; 4] = [
    ["=^•.•^=", "=^•.•^=", "=^-.-^=", "=^•.•^="],
    ["=^o.o^=", "=^o.-^=", "=^o.o^=", "=^-.o^="],
    ["=^-.-^=", "=^-.-^=", "=^v.v^=", "=^-.-^="],
    ["=^>.<^=", "=^@.@^=", "=^>.<^=", "=^T.T^="],
];

/// Bracketed dot-eye faces.
pub const PETS_DOT: [[&str; 4]; 4] = [
    ["[•.•]", "[•.•]", "[-.-]", "[•.•]"],
    ["[o.o]", "[o.-]", "[o.o]", "[-.o]"],
    ["[-.-]", "[-.-]", "[v.v]", "[-.-]"],
    ["[>.<]", "[@.@]", "[>.<]", "[T.T]"],
];

/// Kaomoji faces.
pub const PETS_UNICODE: [[&str; 4]; 4] = [
    ["(•‿•)", "(•‿•)", "(•‿•)", "(-‿-)"],
    ["(•_•)", "(•_-)", "(•_•)", "(-_•)"],
    ["(˘_˘)", "(˘_˘)", "(-_-)", "(˘_˘)"],
    ["(>_<)", "(@_@)", "(>_<)", "(T_T)"],
];

/// Fox faces (the 5th style — ears `ᴥ` + paw `ฅ`, mood escalates the same way).
pub const PETS_FOX: [[&str; 4]; 4] = [
    ["•ᴥ•ฅ", "-ᴥ•ฅ", "•ᴥ•ฅ", "•ᴥ-ฅ"],
    ["oᴥoฅ", "oᴥ-ฅ", "oᴥoฅ", "-ᴥoฅ"],
    ["-ᴥ-ฅ", "-ᴥ-ฅ", "~ᴥ~ฅ", "-ᴥ-ฅ"],
    [">ᴥ<ฅ", "@ᴥ@ฅ", ">ᴥ<ฅ", "TᴥTฅ"],
];

#[allow(dead_code)] // from_name/name/all feed the Phase-3 /emoji style picker.
impl PetStyle {
    /// The 4×4 face pool for this style (None for [`PetStyle::Off`]).
    fn pool(self) -> Option<&'static [[&'static str; 4]; 4]> {
        match self {
            PetStyle::Bear => Some(&PETS_BEAR),
            PetStyle::Cat => Some(&PETS_CAT),
            PetStyle::Dot => Some(&PETS_DOT),
            PetStyle::Unicode => Some(&PETS_UNICODE),
            PetStyle::Fox => Some(&PETS_FOX),
            PetStyle::Off => None,
        }
    }

    /// Parse a pet style by `/emoji`-style name (case-insensitive).
    pub fn from_name(name: &str) -> Option<PetStyle> {
        match name.trim().to_ascii_lowercase().as_str() {
            "bear" => Some(PetStyle::Bear),
            "cat" => Some(PetStyle::Cat),
            "dot" => Some(PetStyle::Dot),
            "unicode" => Some(PetStyle::Unicode),
            "fox" => Some(PetStyle::Fox),
            "off" | "none" | "hidden" => Some(PetStyle::Off),
            _ => None,
        }
    }

    /// The style's display name (for the picker).
    pub fn name(self) -> &'static str {
        match self {
            PetStyle::Bear => "bear",
            PetStyle::Cat => "cat",
            PetStyle::Dot => "dot",
            PetStyle::Unicode => "unicode",
            PetStyle::Fox => "fox",
            PetStyle::Off => "off",
        }
    }

    /// The 5 real (non-off) styles, for the picker to iterate.
    pub fn all() -> [PetStyle; 5] {
        [
            PetStyle::Bear,
            PetStyle::Cat,
            PetStyle::Dot,
            PetStyle::Unicode,
            PetStyle::Fox,
        ]
    }
}

/// The pet face for `(elapsed_ms, frame)`. `frame` ticks at the spinner 0.1s
/// rate; callers pass `tick / 5` to land a ~0.5s pet-frame cadence (tui_v3
/// `_spin//5`). Returns "" for [`PetStyle::Off`]. Deterministic.
pub fn pet_face(style: PetStyle, elapsed_ms: u64, frame: u64) -> &'static str {
    let Some(pool) = style.pool() else {
        return "";
    };
    let tier = heat_tier(elapsed_ms).index();
    let row = &pool[tier];
    row[(frame % row.len() as u64) as usize]
}

/// Number of 0.1s ticks per pet-frame step (~0.5s cadence; tui_v3 `_spin//5`).
pub const PET_TICKS_PER_FRAME: u64 = 5;

/// The pet face for the 0.1s tick clock — convenience over [`pet_face`] that
/// applies the `/5` cadence (a ~0.5s blink).
pub fn pet(style: PetStyle, elapsed_ms: u64, tick: u64) -> &'static str {
    pet_face(style, elapsed_ms, tick / PET_TICKS_PER_FRAME)
}

// ---------------------------------------------------------------------------
// Rotating tips + the interface language now live in the `i18n` plane (§9 "i18n:
// … per-language rotating tips"). `flavor` RE-EXPORTS them so the historical
// `flavor::Lang` / `flavor::tip(...)` call sites keep compiling against one
// source of truth (the dictionaries + tip pools in `crate::i18n`).
// ---------------------------------------------------------------------------

// Some of these are consumed only by `flavor`'s own tests (the cross-check that
// the re-export surface stays deterministic) and by `crate::i18n` directly; the
// re-export keeps `flavor::tip(...)` / `flavor::Lang` working for non-test code.
#[allow(unused_imports)]
pub use crate::i18n::{tip, tip_at, tips_for, Lang, TIPS_EN, TIPS_ZH, TIP_TICKS_PER_STEP};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arc_is_the_default_and_not_the_cc_asterisk() {
        assert_eq!(SpinnerStyle::default(), SpinnerStyle::Arc);
        // The CC marks must NOT appear in our default spinner.
        let frames = SpinnerStyle::Arc.frames();
        assert!(!frames.contains(&'✻'));
        assert!(!frames.contains(&'✷'));
        assert_eq!(frames, ARC_FRAMES.to_vec());
    }

    #[test]
    fn spinner_wraps_cyclically() {
        let s = SpinnerStyle::Arc;
        let n = ARC_FRAMES.len() as u64;
        assert_eq!(s.glyph(0), ARC_FRAMES[0]);
        assert_eq!(s.glyph(n), ARC_FRAMES[0]); // wraps
        assert_eq!(s.glyph(n + 2), ARC_FRAMES[2]);
        // Pulse is forward then reversed (length doubles).
        assert_eq!(SpinnerStyle::Pulse.frames().len(), PULSE_GLYPHS.len() * 2);
        // Round-trips through the name parser.
        assert_eq!(SpinnerStyle::from_name("braille"), Some(SpinnerStyle::Braille));
        assert_eq!(SpinnerStyle::from_name("ARC"), Some(SpinnerStyle::Arc));
        assert_eq!(SpinnerStyle::from_name("nope"), None);
    }

    #[test]
    fn heat_ramp_matches_v3_thresholds() {
        assert_eq!(heat_token(0), Token::Success);
        assert_eq!(heat_token(19_999), Token::Success);
        assert_eq!(heat_token(20_000), Token::Warning);
        assert_eq!(heat_token(59_999), Token::Warning);
        assert_eq!(heat_token(60_000), Token::Claude);
        assert_eq!(heat_token(179_999), Token::Claude);
        assert_eq!(heat_token(180_000), Token::Error);
        assert!(heat_bold(200_000));
        assert!(!heat_bold(10_000));
    }

    /// THE deliverable test: gerund rotation is deterministic — a pure function
    /// of `(lang, tick)` (no randomness, no wall-clock). The same tick always
    /// yields the same word; it advances exactly once per ~6s window, wraps over
    /// the whole pool, and PICKS THE POOL BY LANGUAGE (Q12: zh pool for `Lang::Zh`).
    #[test]
    fn gerund_rotation_deterministic() {
        let n = GERUNDS.len() as u64;

        // Step index → word is pool[step % n], and pinned exactly (en).
        assert_eq!(gerund_at(Lang::En, 0), "Pondering");
        assert_eq!(gerund_at(Lang::En, 1), "Reticulating");
        assert_eq!(gerund_at(Lang::En, 33), "Tuning");
        assert_eq!(gerund_at(Lang::En, n), "Pondering"); // wraps over the whole pool.
        assert_eq!(gerund_at(Lang::En, n + 5), gerund_at(Lang::En, 5)); // periodic.

        // The 0.1s-tick view holds ONE word for a full 6s window (ticks 0..59),
        // then steps at tick 60 — and is identical for any tick in the window
        // (a redraw never jitters the word).
        for t in 0..GERUND_TICKS_PER_STEP {
            assert_eq!(gerund(Lang::En, t), "Pondering", "tick {t} must stay on word 0");
        }
        assert_eq!(gerund(Lang::En, GERUND_TICKS_PER_STEP), "Reticulating");
        assert_eq!(gerund(Lang::En, GERUND_TICKS_PER_STEP * 33), "Tuning");
        assert_eq!(gerund(Lang::En, GERUND_TICKS_PER_STEP * n), "Pondering"); // full wrap.

        // Q12: `Lang::Zh` selects the parallel zh pool — same index, zh word.
        assert_eq!(gerund_at(Lang::Zh, 0), "沉思中");
        assert_eq!(gerund_at(Lang::Zh, 1), "织网中");
        assert_eq!(gerund(Lang::Zh, 0), GERUNDS_ZH[0]);
        assert_ne!(gerund_at(Lang::Zh, 0), gerund_at(Lang::En, 0));

        // Purity: calling twice with the same (lang, tick) returns the SAME word.
        assert_eq!(gerund(Lang::En, 777), gerund(Lang::En, 777));
        // And it is the static pool entry it indexes (value equality).
        assert_eq!(gerund(Lang::En, 0), GERUNDS[0]);
    }

    /// THE Q12 parity guard: the en/zh gerund pools are the SAME length, so every
    /// English egg has a Chinese sibling at the same index (no index can pick a
    /// word in one language but fall out of range in the other).
    #[test]
    fn gerunds_parity() {
        assert_eq!(GERUNDS.len(), GERUNDS_ZH.len(), "en/zh gerund pools must match length");
        // No empty entries on either side (a blank gerund would read as a gap).
        assert!(GERUNDS.iter().all(|g| !g.is_empty()));
        assert!(GERUNDS_ZH.iter().all(|g| !g.is_empty()));
    }

    #[test]
    fn pet_faces_are_five_styles_four_tiers_four_frames() {
        // Five real styles (plus Off).
        assert_eq!(PetStyle::all().len(), 5);
        for style in PetStyle::all() {
            let pool = style.pool().expect("real style has a pool");
            assert_eq!(pool.len(), 4, "{} must have 4 heat tiers", style.name());
            for tier in pool {
                assert_eq!(tier.len(), 4, "each tier has 4 blink frames");
            }
        }
        // Off renders nothing.
        assert_eq!(pet_face(PetStyle::Off, 0, 0), "");
        // Heat selects the tier: calm vs critical differ for the bear.
        let calm = pet_face(PetStyle::Bear, 0, 0);
        let crit = pet_face(PetStyle::Bear, 200_000, 0);
        assert_ne!(calm, crit);
        assert_eq!(calm, "ʕ•ᴥ•ʔ");
        // Frame wraps within the tier; deterministic.
        assert_eq!(pet_face(PetStyle::Bear, 0, 4), pet_face(PetStyle::Bear, 0, 0));
        // The /5 cadence: ticks 0..4 share frame 0.
        assert_eq!(pet(PetStyle::Cat, 0, 0), pet(PetStyle::Cat, 0, 4));
        // Name round-trip.
        assert_eq!(PetStyle::from_name("fox"), Some(PetStyle::Fox));
        assert_eq!(PetStyle::from_name("off"), Some(PetStyle::Off));
    }

    #[test]
    fn tips_rotate_deterministically_per_language() {
        assert!(!TIPS_EN.is_empty());
        assert_eq!(TIPS_EN.len(), TIPS_ZH.len(), "parallel tip lists");
        // Step index → tip is pool[step % n]; wraps; deterministic.
        assert_eq!(tip_at(Lang::En, 0), TIPS_EN[0]);
        assert_eq!(tip_at(Lang::En, TIPS_EN.len() as u64), TIPS_EN[0]);
        assert_eq!(tip_at(Lang::Zh, 1), TIPS_ZH[1]);
        // The 0.1s-tick view holds one tip for a ~12s window.
        for t in 0..TIP_TICKS_PER_STEP {
            assert_eq!(tip(Lang::En, t), TIPS_EN[0]);
        }
        assert_eq!(tip(Lang::En, TIP_TICKS_PER_STEP), TIPS_EN[1]);
    }
}
