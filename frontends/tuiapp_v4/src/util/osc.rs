//! util/osc.rs — OSC (Operating System Command) escape sequences for the
//! terminal window title (OSC 0) and the iTerm2/Ghostty/WezTerm tab status
//! (OSC 21337), used by the flavor layer (§9: "OSC0 terminal title spinner",
//! "OSC-21337 tab status").
//!
//! These are a clean cross-terminal "session needs you" signal: a terminal that
//! understands the sequence shows it; one that doesn't silently ignores it (the
//! bytes are a no-op). So we ALWAYS emit when the state changes and never gate
//! on capability detection — a wrong guess would suppress the signal on a
//! supporting terminal, which is worse than a harmless no-op on an old one.
//!
//! The payload BUILDERS here are PURE + unit-tested (exact bytes pinned); the
//! `write_*` functions are thin wrappers that push the bytes to stdout. We write
//! straight to the real stdout (not through ratatui's buffer) because these are
//! out-of-band terminal commands, not cell content.

use std::io::Write;

const ESC: char = '\x1b';
const BEL: char = '\x07';

/// The tab-status the OSC 21337 signal reports (recon `claude_code_patterns.md`
/// §"Terminal tab status (OSC 21337)"). Drives a tab color + label so a
/// background session that needs attention is visible without focusing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TabStatus {
    /// Nothing running, nothing waiting — green "Idle".
    Idle,
    /// A turn is streaming — orange "Working…".
    Working,
    /// Blocked on the user (an ask_user / pending input) — blue "Waiting".
    Waiting,
}

impl TabStatus {
    /// The (r, g, b) tab color for this status (recon-pinned values).
    pub fn rgb(self) -> (u8, u8, u8) {
        match self {
            TabStatus::Idle => (0, 215, 95),     // green
            TabStatus::Working => (255, 149, 0), // orange
            TabStatus::Waiting => (95, 135, 255), // blue
        }
    }

    /// The short label for this status.
    pub fn label(self) -> &'static str {
        match self {
            TabStatus::Idle => "Idle",
            TabStatus::Working => "Working…",
            TabStatus::Waiting => "Waiting",
        }
    }
}

/// Build the OSC 0 "set window/icon title" sequence: `ESC ] 0 ; <text> BEL`.
/// PURE. `text` is sanitized of control bytes (a stray `\n`/BEL would terminate
/// the sequence early or corrupt the title), keeping only printable content.
pub fn build_title(text: &str) -> String {
    let safe: String = text
        .chars()
        .filter(|c| !c.is_control())
        .collect();
    format!("{ESC}]0;{safe}{BEL}")
}

/// Build the OSC 21337 tab-status sequence reporting `status` with `label`. The
/// iTerm2-style form sets both a label and a color via two `SetUserVar`-like
/// fields: `ESC ] 21337 ; SetTabColor=r,g,b ; SetTabTitle=<label> BEL`. On a
/// terminal that doesn't grok it, the whole thing is ignored (no-op). PURE.
pub fn build_tab_status(status: TabStatus, label: &str) -> String {
    let (r, g, b) = status.rgb();
    let safe: String = label.chars().filter(|c| !c.is_control()).collect();
    format!("{ESC}]21337;SetTabColor={r},{g},{b};SetTabTitle={safe}{BEL}")
}

/// Write an OSC 0 window title. Best-effort: a failure (e.g. no stdout) is
/// swallowed — a title is cosmetic and must never break the render loop.
pub fn write_title(text: &str) {
    let seq = build_title(text);
    let mut out = std::io::stdout();
    let _ = out.write_all(seq.as_bytes());
    let _ = out.flush();
}

/// Write an OSC 21337 tab status. Best-effort (see [`write_title`]).
pub fn write_tab_status(status: TabStatus, label: &str) {
    let seq = build_tab_status(status, label);
    let mut out = std::io::stdout();
    let _ = out.write_all(seq.as_bytes());
    let _ = out.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn title_golden_bytes_and_control_strip() {
        // Exact OSC 0 form.
        assert_eq!(build_title("hi"), "\x1b]0;hi\x07");
        // Control bytes (newline / BEL) are stripped so they can't terminate the
        // sequence early or smear the title.
        assert_eq!(build_title("a\nb\x07c"), "\x1b]0;abc\x07");
        // CJK survives (it is not a control char).
        assert_eq!(build_title("会话"), "\x1b]0;会话\x07");
    }

    #[test]
    fn tab_status_colors_and_labels_match_recon() {
        assert_eq!(TabStatus::Idle.rgb(), (0, 215, 95));
        assert_eq!(TabStatus::Working.rgb(), (255, 149, 0));
        assert_eq!(TabStatus::Waiting.rgb(), (95, 135, 255));

        let seq = build_tab_status(TabStatus::Working, "Working…");
        assert!(seq.starts_with("\x1b]21337;SetTabColor=255,149,0;"));
        assert!(seq.contains("SetTabTitle=Working…"));
        assert!(seq.ends_with('\x07'));

        // Idle is green.
        assert!(build_tab_status(TabStatus::Idle, "Idle").contains("SetTabColor=0,215,95;"));
        // Waiting is blue.
        assert!(build_tab_status(TabStatus::Waiting, "Waiting").contains("SetTabColor=95,135,255;"));
    }
}
