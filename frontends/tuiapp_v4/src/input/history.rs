//! input/history.rs — persisted input history + Up/Down navigation
//! (checklist §8 "history at edges", "input history (persisted to temp/)";
//! tui_v3 §E-59). The composer asks the history for the previous/next entry when
//! the caret is at the buffer's top/bottom visual edge; a fresh draft is stashed
//! so leaving history restores it.
//!
//! Behavior (verbatim from tui_v3 `_nav_hist`): consecutive duplicates are
//! skipped on push; the store caps at 500 and trims to 250 when exceeded;
//! navigation walks a cursor over the list with a stashed draft at the bottom.
//!
//! Persistence: one entry per line in `temp/tui_v4_input_history.txt` under the
//! GA repo root (the bridge's `repo_root`). The PURE navigation logic is fully
//! unit-tested; load/save are thin effectful wrappers (newline-escaped so a
//! multi-line entry round-trips as one line).

use std::path::{Path, PathBuf};

/// Hard cap before trimming (tui_v3: cap 500 / trim 250).
pub const HISTORY_CAP: usize = 500;
/// Size to trim BACK to once the cap is exceeded.
pub const HISTORY_TRIM: usize = 250;

/// The history filename under `<repo>/temp/`.
pub const HISTORY_FILE: &str = "tui_v4_input_history.txt";

/// Persisted input history + a navigation cursor.
///
/// `cursor == None` means "editing a fresh line" (not browsing). When browsing,
/// `cursor == Some(i)` points at `entries[i]`, and `draft` holds the in-progress
/// line the user had typed before they started pressing Up.
#[derive(Debug, Default, Clone)]
pub struct History {
    entries: Vec<String>,
    cursor: Option<usize>,
    draft: String,
    /// Path we persist to (None = in-memory only, e.g. in tests).
    path: Option<PathBuf>,
}

impl History {
    /// An empty in-memory history (no persistence).
    pub fn new() -> Self {
        History::default()
    }

    /// Resolve the history file path under a GA repo root (`<root>/temp/<file>`).
    pub fn path_for(repo_root: &Path) -> PathBuf {
        repo_root.join("temp").join(HISTORY_FILE)
    }

    /// Load history from `<repo>/temp/` (creating nothing). Missing file → empty.
    /// Effectful; the parse (newline-unescape) is via [`decode_line`].
    pub fn load(repo_root: &Path) -> Self {
        let path = History::path_for(repo_root);
        let mut entries = Vec::new();
        if let Ok(text) = std::fs::read_to_string(&path) {
            for line in text.lines() {
                let entry = decode_line(line);
                if !entry.is_empty() {
                    entries.push(entry);
                }
            }
        }
        // Apply the cap on load too (a hand-edited file could be huge).
        if entries.len() > HISTORY_CAP {
            let cut = entries.len() - HISTORY_TRIM;
            entries.drain(0..cut);
        }
        History {
            entries,
            cursor: None,
            draft: String::new(),
            path: Some(path),
        }
    }

    /// The entries (oldest → newest).
    #[allow(dead_code)] // read by tests + the Phase-3 history viewer.
    pub fn entries(&self) -> &[String] {
        &self.entries
    }

    /// Number of stored entries.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Push a submitted line (skipping a consecutive duplicate + blanks),
    /// trimming when the cap is exceeded, and resetting the nav cursor. Persists
    /// (best-effort) when a path is set. Mirrors tui_v3 `_nav_hist` push rules.
    pub fn push(&mut self, line: &str) {
        let line = line.trim_end_matches('\n');
        if line.trim().is_empty() {
            self.cursor = None;
            return;
        }
        if self.entries.last().map(String::as_str) == Some(line) {
            // Consecutive duplicate → don't store again, just reset nav.
            self.cursor = None;
            self.draft.clear();
            return;
        }
        self.entries.push(line.to_string());
        if self.entries.len() > HISTORY_CAP {
            let cut = self.entries.len() - HISTORY_TRIM;
            self.entries.drain(0..cut);
        }
        self.cursor = None;
        self.draft.clear();
        self.save();
    }

    /// Begin/continue browsing UP (toward older). `current` is the live buffer
    /// (stashed as the draft the first time). Returns the entry to show, or
    /// `None` if there's nothing older (caller leaves the buffer unchanged).
    pub fn nav_up(&mut self, current: &str) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }
        match self.cursor {
            None => {
                // Starting to browse → stash the draft, jump to the newest entry.
                self.draft = current.to_string();
                let idx = self.entries.len() - 1;
                self.cursor = Some(idx);
                Some(self.entries[idx].clone())
            }
            Some(0) => None, // already at the oldest.
            Some(i) => {
                let idx = i - 1;
                self.cursor = Some(idx);
                Some(self.entries[idx].clone())
            }
        }
    }

    /// Browse DOWN (toward newer). Returns the entry to show; past the newest
    /// entry it returns the stashed draft (and ends browsing). `None` when not
    /// currently browsing (caller does nothing).
    pub fn nav_down(&mut self) -> Option<String> {
        match self.cursor {
            None => None,
            Some(i) if i + 1 < self.entries.len() => {
                let idx = i + 1;
                self.cursor = Some(idx);
                Some(self.entries[idx].clone())
            }
            Some(_) => {
                // Past the newest → restore the in-progress draft, stop browsing.
                self.cursor = None;
                Some(std::mem::take(&mut self.draft))
            }
        }
    }

    /// True while the user is browsing history (a cursor is set).
    #[allow(dead_code)] // surfaced by the Phase-3 history-mode indicator.
    pub fn is_browsing(&self) -> bool {
        self.cursor.is_some()
    }

    /// Reset navigation (called when the user edits the buffer mid-browse so the
    /// next Up starts a fresh browse from the newest entry).
    pub fn reset_nav(&mut self) {
        self.cursor = None;
    }

    /// Persist the entries to the configured path (best-effort; a failure is
    /// swallowed — history is a convenience, not load-bearing). Effectful.
    fn save(&self) {
        let Some(path) = &self.path else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let body: String = self
            .entries
            .iter()
            .map(|e| encode_line(e))
            .collect::<Vec<_>>()
            .join("\n");
        let _ = std::fs::write(path, body);
    }
}

/// Encode one entry for one-line storage: real newlines → `\n` (literal
/// backslash-n), and a literal backslash → `\\`, so a multi-line entry survives
/// a round-trip through a line-oriented file. PURE.
pub fn encode_line(entry: &str) -> String {
    entry.replace('\\', "\\\\").replace('\n', "\\n")
}

/// Decode a stored line back to the original entry. PURE inverse of
/// [`encode_line`].
pub fn decode_line(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_skips_consecutive_dupes_and_blanks() {
        let mut h = History::new();
        h.push("first");
        h.push("first"); // consecutive dup → skipped.
        h.push("second");
        h.push("   "); // blank → skipped.
        h.push("first"); // non-consecutive dup → stored.
        assert_eq!(h.entries(), &["first", "second", "first"]);
    }

    /// THE deliverable test: Up/Down history navigation with the draft stash.
    #[test]
    fn history_nav() {
        let mut h = History::new();
        h.push("alpha");
        h.push("beta");
        h.push("gamma"); // entries: [alpha, beta, gamma]

        // The user has typed "wip" and presses Up: draft stashed, show newest.
        assert_eq!(h.nav_up("wip").as_deref(), Some("gamma"));
        assert!(h.is_browsing());
        assert_eq!(h.nav_up("gamma").as_deref(), Some("beta"));
        assert_eq!(h.nav_up("beta").as_deref(), Some("alpha"));
        // At the oldest → further Up does nothing (buffer stays on "alpha").
        assert_eq!(h.nav_up("alpha"), None);

        // Down walks back toward newer…
        assert_eq!(h.nav_down().as_deref(), Some("beta"));
        assert_eq!(h.nav_down().as_deref(), Some("gamma"));
        // …and past the newest restores the stashed draft + ends browsing.
        assert_eq!(h.nav_down().as_deref(), Some("wip"));
        assert!(!h.is_browsing());
        // Down when not browsing is a no-op.
        assert_eq!(h.nav_down(), None);

        // Up with an empty history does nothing.
        let mut empty = History::new();
        assert_eq!(empty.nav_up("x"), None);
    }

    #[test]
    fn cap_trims_to_250_when_exceeded() {
        let mut h = History::new();
        for i in 0..(HISTORY_CAP + 10) {
            h.push(&format!("line {i}"));
        }
        // Crossing the cap trims back to TRIM (then it grows again as more push),
        // so the count stays bounded in [TRIM, CAP] and the most recent is kept
        // (tui_v3 per-push "cap 500 / trim 250" semantics).
        assert!(
            h.len() >= HISTORY_TRIM && h.len() <= HISTORY_CAP,
            "history len {} out of [{HISTORY_TRIM}, {HISTORY_CAP}]",
            h.len()
        );
        // The trim only happened once (501 → 250, then +9 = 259).
        assert_eq!(h.len(), HISTORY_TRIM + (HISTORY_CAP + 10 - (HISTORY_CAP + 1)));
        assert_eq!(h.entries().last().unwrap(), &format!("line {}", HISTORY_CAP + 9));
        // The oldest surviving entries were dropped.
        assert!(!h.entries().iter().any(|e| e == "line 0"));
    }

    #[test]
    fn encode_decode_round_trips_multiline() {
        let entry = "line one\nline two\\with backslash";
        let enc = encode_line(entry);
        assert!(!enc.contains('\n')); // one-line storable.
        assert_eq!(decode_line(&enc), entry);
    }

    #[test]
    fn load_save_round_trip_to_temp() {
        let root = std::env::temp_dir().join(format!("tui_v4_hist_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("temp")).unwrap();

        let mut h = History::load(&root);
        assert!(h.is_empty());
        h.push("remembered\nmultiline");
        h.push("second");

        // Reload from disk → entries survive (incl. the multi-line one).
        let h2 = History::load(&root);
        assert_eq!(h2.entries(), &["remembered\nmultiline", "second"]);

        let _ = std::fs::remove_dir_all(&root);
    }
}
