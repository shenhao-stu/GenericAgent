//! components/text.rs — pure label/format helpers shared across the cockpit
//! widgets, overlays, and the dashboard. No ratatui frame I/O: each fn is a pure
//! `&str/u64 → String/&str` transform, unit-tested in isolation.

/// Shorten a long cwd so the header never overflows. Keeps the tail (most
/// informative) with a leading ellipsis. PURE + unit-tested.
pub(crate) fn compact_cwd(cwd: &str, max: usize) -> String {
    let max = max.max(8);
    if cwd.chars().count() <= max {
        return cwd.to_string();
    }
    let tail: String = cwd
        .chars()
        .rev()
        .take(max - 1)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("…{tail}")
}

/// Default cap (display cells) for the footer/header model label (redesign_cc.md
/// §2.5: "cap ~22 cells").
pub(crate) const MODEL_LABEL_CAP: usize = 22;

/// Shorten the model label for the footer/header (redesign_cc.md §2.5). GA's
/// `get_llm_name()` for a MixinSession is a long pipe-list of the whole fallback
/// chain — e.g. `codex-pro|gpt-5|claude-opus|…|kiro` or `MixinSession/codex-pro|…`.
/// We show ONLY the PRIMARY segment (the first pipe member, the model that's
/// actually driving the turn): `codex-pro`, or `MixinSession·codex-pro` when a
/// `SessionType/` prefix is present. Never the full pipe-list. Capped to `cap`
/// display cells with a trailing `…`. PURE + unit-tested (`truncate_model_*`).
pub(crate) fn truncate_model(raw: &str, cap: usize) -> String {
    let raw = raw.trim();
    if raw.is_empty() {
        return "—".to_string();
    }
    // Peel an optional `SessionType/rest` prefix (GA emits `SessionType/name`).
    // Use the FIRST `/` so a model name that itself contains `/` keeps its tail in
    // the pipe-list step below.
    let (prefix, rest) = match raw.split_once('/') {
        Some((p, r)) if !p.is_empty() && !r.is_empty() => (Some(p.trim()), r.trim()),
        _ => (None, raw),
    };
    // The PRIMARY segment = the first pipe-separated member of the chain.
    let primary = rest.split('|').next().unwrap_or(rest).trim();
    let primary = if primary.is_empty() { rest } else { primary };
    // `SessionType·primary` (middot join) when a prefix was present; else bare.
    let label = match prefix {
        Some(p) => format!("{p}·{primary}"),
        None => primary.to_string(),
    };
    // Cap to `cap` cells; if it overflows, clip to `cap-1` and append `…`.
    use unicode_width::UnicodeWidthStr;
    if UnicodeWidthStr::width(label.as_str()) <= cap {
        return label;
    }
    // Prefer keeping the bare primary if the `SessionType·` join is what blew the
    // cap (the model name is the load-bearing part) — but only if it then fits.
    if prefix.is_some() && UnicodeWidthStr::width(primary) <= cap {
        return primary.to_string();
    }
    let body = clip_to(&label, cap.saturating_sub(1));
    format!("{body}…")
}

/// Which ORCHESTRATION command (if any) the composer currently holds — the one that
/// earns a DISTINCT input-box effect identity (Q11b/c): `/goal` `/hive` `/conductor`
/// `/morphling`. Matches the command WORD at the start, so `/hive do x` counts but
/// `/hivemind` does not, and a plain message yields `None`. Returning the identity
/// (not a `bool`) lets the painter pick the per-command border + char effect — the
/// constraint lives in the type, not a re-parse (P6). PURE + unit-tested.
pub(crate) fn fx_command(text: &str) -> Option<crate::theme::FxCommand> {
    use crate::theme::FxCommand;
    let rest = text.trim_start().strip_prefix('/')?;
    let word = rest.split(|c: char| c.is_whitespace()).next().unwrap_or("");
    Some(match word {
        "goal" => FxCommand::Goal,
        "hive" => FxCommand::Hive,
        "conductor" => FxCommand::Conductor,
        "morphling" => FxCommand::Morphling,
        _ => return None,
    })
}

/// Whether the composer holds an orchestration command (the input-box effects gate).
/// A thin predicate over [`fx_command`]; the live gate now reads the identity via
/// `fx_command`, so this is the predicate the regression test exercises.
#[inline]
#[allow(dead_code)] // the cockpit gate uses `fx_command`; kept for the bool-shape test.
pub(crate) fn fx_command_active(text: &str) -> bool {
    fx_command(text).is_some()
}

/// The SIMPLIFIED llm-channel label for the header (Q7 "llm 渠道[简洁版]"). GA's
/// model string is a MixinSession pipe-list (`SessionType/primary|b|c|…`); the
/// CHANNEL is the `SessionType/` prefix when present (e.g. `MixinSession`), which
/// is the "how it's routed" hint distinct from the `model` field next to it. With
/// no prefix the model isn't routed through a session type → `direct`; an empty /
/// missing model → `—`. PURE + unit-tested (the bridge doesn't report a provider
/// name today; the session-type prefix is the only channel signal available).
pub(crate) fn llm_channel(raw: Option<&str>) -> &'static str {
    let raw = raw.unwrap_or("").trim();
    if raw.is_empty() {
        return "—";
    }
    match raw.split_once('/') {
        Some((p, r)) if !p.trim().is_empty() && !r.trim().is_empty() => match p.trim() {
            // The few channels GA emits as a SessionType prefix → a stable label
            // (so the header reads `llm MixinSession` not the full chain).
            "MixinSession" => "MixinSession",
            "MultiModelSession" => "MultiModel",
            other if other.len() <= 16 => intern_channel(other),
            _ => "session",
        },
        // No `SessionType/` prefix → a single model called directly.
        _ => "direct",
    }
}

/// Intern a session-type prefix as a `'static` label (bounded: the prefix set is a
/// handful of GA `SessionType` names). Keeps [`llm_channel`] returning `&'static`
/// without a giant match. PURE-ish (a small process-lifetime leak set).
fn intern_channel(name: &str) -> &'static str {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();
    let map = SEEN.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = map.lock().expect("llm_channel intern lock");
    if let Some(s) = guard.get(name) {
        return s;
    }
    let leaked: &'static str = Box::leak(name.to_string().into_boxed_str());
    guard.insert(name.to_string(), leaked);
    leaked
}

/// Format an elapsed DURATION in whole seconds as the compact `1m 46s` / `46s` /
/// `1h 02m 03s` form (the above-composer done-line, Q7 — v2's `_fmt_elapsed` /
/// Codex `fmt_elapsed_compact`). PURE + unit-tested.
pub(crate) fn fmt_dur(secs: u64) -> String {
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if h > 0 {
        format!("{h}h {m:02}m {s:02}s")
    } else if m > 0 {
        format!("{m}m {s:02}s")
    } else {
        format!("{s}s")
    }
}

/// The platform's Ctrl-modifier label for the keybindings help (request #2:
/// "detect the system, show different bindings"). macOS uses the compact `⌃`
/// symbol; other platforms spell `Ctrl-` (the `⌃` glyph reads as noise off-mac).
/// Compile-time `cfg!` equals the running OS for a native binary, so each release
/// artifact (the win .exe, the mac .dmg, the linux build) shows its own
/// convention. PURE. Consumed by the `/keybindings` overlay (Slice 6); the chrome
/// rows no longer inline keybinding pairs (Q7 routes them to that overlay).
#[allow(dead_code)] // wired to the /keybindings overlay landing in Slice 6.
pub(crate) fn ctrl_key_label() -> &'static str {
    if cfg!(target_os = "macos") {
        "⌃"
    } else {
        "Ctrl-"
    }
}

/// Compact token count for the spinner readout: `950 → "950"`, `1234 → "1.2k"`,
/// `2_300_000 → "2.3m"` (CC's `formatNumber` / tui_v3's `_human`). PURE.
pub(crate) fn human_count(n: u64) -> String {
    if n < 1000 {
        n.to_string()
    } else if n < 1_000_000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        format!("{:.1}m", n as f64 / 1_000_000.0)
    }
}

/// Clip a string to at most `max` display cells (no ellipsis). PURE. Shared with
/// the dashboard component (name/preview truncation).
pub(crate) fn clip_to(s: &str, max: usize) -> String {
    use unicode_segmentation::UnicodeSegmentation;
    use unicode_width::UnicodeWidthStr;
    if UnicodeWidthStr::width(s) <= max {
        return s.to_string();
    }
    let mut out = String::new();
    let mut acc = 0usize;
    for g in s.graphemes(true) {
        let gw = UnicodeWidthStr::width(g);
        if acc + gw > max {
            break;
        }
        out.push_str(g);
        acc += gw;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_cwd_keeps_tail() {
        let long = "/very/long/path/to/some/deeply/nested/project/dir";
        let out = compact_cwd(long, 20);
        assert!(out.chars().count() <= 20);
        assert!(out.starts_with('…'));
        assert!(out.ends_with("project/dir"));
        assert_eq!(compact_cwd("/a/b", 40), "/a/b");
    }

    #[test]
    fn clip_to_respects_cells() {
        assert_eq!(clip_to("hello", 3), "hel");
        assert_eq!(clip_to("你好世界", 4), "你好"); // 2 wide glyphs = 4 cells.
        assert_eq!(clip_to("hi", 10), "hi");
    }

    /// The compact token formatter (spinner `↑in ↓out`): thousands → `k`, millions
    /// → `m`; small counts stay literal.
    #[test]
    fn human_count_compacts_thousands_and_millions() {
        assert_eq!(human_count(0), "0");
        assert_eq!(human_count(950), "950");
        assert_eq!(human_count(1234), "1.2k");
        assert_eq!(human_count(340), "340");
        assert_eq!(human_count(2_300_000), "2.3m");
    }

    /// The input-box effects (redesign #4) light up ONLY for the orchestration
    /// commands — matched on the command word at the start, not a substring.
    #[test]
    fn fx_command_active_only_for_orchestration() {
        assert!(fx_command_active("/hive"));
        assert!(fx_command_active("/goal build the thing"));
        assert!(fx_command_active("  /conductor"));
        assert!(fx_command_active("/morphling absorb a skill"));
        // NOT a longer word that merely starts with one, a different command,
        // a mid-line slash, or a plain message.
        assert!(!fx_command_active("/hivemind"));
        assert!(!fx_command_active("/goalkeeper"));
        assert!(!fx_command_active("/help"));
        assert!(!fx_command_active("hello /hive"));
        assert!(!fx_command_active(""));
        assert!(!fx_command_active("just a normal message"));
    }

    /// `fx_command` maps each of the four orchestration words to its identity and
    /// rejects look-alikes / non-commands (so the painter never lights the wrong box).
    #[test]
    fn fx_command_maps_four_words_rejects_hivemind() {
        use crate::theme::FxCommand;
        assert_eq!(fx_command("/goal build the thing"), Some(FxCommand::Goal));
        assert_eq!(fx_command("/hive"), Some(FxCommand::Hive));
        assert_eq!(fx_command("  /conductor go"), Some(FxCommand::Conductor));
        assert_eq!(fx_command("/morphling absorb a skill"), Some(FxCommand::Morphling));
        // Look-alikes, other commands, mid-line slashes, and plain text → None.
        assert_eq!(fx_command("/hivemind"), None);
        assert_eq!(fx_command("/goalkeeper"), None);
        assert_eq!(fx_command("/help"), None);
        assert_eq!(fx_command("hello /hive"), None);
        assert_eq!(fx_command(""), None);
    }

    /// `llm_channel` (Q7 header) extracts the SIMPLIFIED routing channel from the
    /// model string: the `SessionType/` prefix when present, `direct` for a bare
    /// model, `—` for none.
    #[test]
    fn llm_channel_simplifies_session_prefix() {
        // The real MixinSession shape → the channel is the session-type prefix.
        assert_eq!(
            llm_channel(Some("MixinSession/codex-pro|gpt-5.2|kiro")),
            "MixinSession"
        );
        assert_eq!(llm_channel(Some("MultiModelSession/glm-4|x")), "MultiModel");
        // A bare model with no session-type prefix → routed directly.
        assert_eq!(llm_channel(Some("codex-pro|gpt-5|kiro")), "direct");
        assert_eq!(llm_channel(Some("gpt-5.2-mini")), "direct");
        // Empty / missing → the em-dash placeholder.
        assert_eq!(llm_channel(None), "—");
        assert_eq!(llm_channel(Some("")), "—");
        assert_eq!(llm_channel(Some("   ")), "—");
    }

    /// `fmt_dur` (the done-line) renders whole seconds as `46s` / `1m 46s` /
    /// `1h 02m 03s`.
    #[test]
    fn fmt_dur_compact_forms() {
        assert_eq!(fmt_dur(0), "0s");
        assert_eq!(fmt_dur(46), "46s");
        assert_eq!(fmt_dur(60), "1m 00s");
        assert_eq!(fmt_dur(106), "1m 46s");
        assert_eq!(fmt_dur(3600), "1h 00m 00s");
        assert_eq!(fmt_dur(3723), "1h 02m 03s");
    }

    /// `truncate_model` shows ONLY the primary segment of a MixinSession's pipe-list
    /// (redesign_cc.md §2.5) — `MixinSession·codex-pro`, never `…|kiro`, capped to
    /// ~22 cells.
    #[test]
    fn truncate_model_primary_segment() {
        use unicode_width::UnicodeWidthStr;
        let cap = MODEL_LABEL_CAP;

        // The real MixinSession shape: `SessionType/primary|b|c|…|kiro`.
        let raw = "MixinSession/codex-pro|gpt-5.2|claude-opus-4|gemini-2.5-pro|grok-4|kiro";
        let out = truncate_model(raw, cap);
        assert_eq!(out, "MixinSession·codex-pro", "primary segment with the session prefix");
        assert!(UnicodeWidthStr::width(out.as_str()) <= cap, "within the ~22-cell cap");
        // The full pipe-list NEVER survives.
        assert!(!out.contains('|'), "no pipe-list");
        assert!(!out.contains("kiro"), "no trailing chain member");
        assert!(!out.contains("gpt-5.2"), "no secondary segments");

        // A bare pipe-list (no `SessionType/`) → just the primary, no middot prefix.
        assert_eq!(truncate_model("codex-pro|gpt-5|claude|kiro", cap), "codex-pro");

        // A plain single model passes through unchanged.
        assert_eq!(truncate_model("gpt-5.2-mini", cap), "gpt-5.2-mini");
        // The simple `SessionType/name` (no pipes) → `SessionType·name`.
        assert_eq!(truncate_model("MixinSession/codex-pro", cap), "MixinSession·codex-pro");

        // Empty / blank → the em-dash placeholder.
        assert_eq!(truncate_model("", cap), "—");
        assert_eq!(truncate_model("   ", cap), "—");

        // Over-cap: a long bare primary is clipped with a trailing `…` to the cap.
        let long = "supercalifragilistic-model-name|fallback";
        let out = truncate_model(long, cap);
        assert!(UnicodeWidthStr::width(out.as_str()) <= cap);
        assert!(out.ends_with('…'));
        assert!(!out.contains("fallback"));

        // Over-cap where the `SessionType·` join blows the budget but the bare
        // primary fits → drop the prefix, keep the (load-bearing) model name whole.
        let out = truncate_model("VeryLongSessionTypeName/codex-pro|x|y", cap);
        assert_eq!(out, "codex-pro");
        assert!(UnicodeWidthStr::width(out.as_str()) <= cap);
    }
}
