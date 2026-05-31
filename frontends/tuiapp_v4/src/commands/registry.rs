//! commands/registry.rs — the slash-command registry + resolution used by the
//! composer's `/`-prefix dropdown AND the dispatcher (checklist §4; tui_v3 §B).
//!
//! The registry is the SINGLE source of truth for the ~33-command union (§4). It
//! exposes four pure, unit-tested surfaces:
//!   * [`resolve`] — map a typed `/name` (exact, case-insensitive) to its
//!     [`SlashCommand`]; this is what the dispatcher uses so EVERY §4 name routes.
//!   * [`palette_matches`] — the live fuzzy palette while a `/word` is typed: a
//!     prefix match ranks first, then a SUBSEQUENCE fuzzy match (so `/cont` and
//!     `/cnt` both find `continue`), in a stable, deterministic order.
//!   * [`complete_to`] — Tab/Enter completion of a highlighted match into `/name `.
//!   * [`did_you_mean`] — for an UNKNOWN `/typo`, the closest command name by edit
//!     distance, for the friendly "did you mean /x?" breadcrumb (§4).
//!
//! Each command also carries its [`CommandKind`] (App / Ui / Fwd) so the dispatcher
//! knows whether to handle it in-app, open a dedicated panel, or core-forward it as
//! a `Command{name,args}` frame.

/// How a slash command is handled (checklist §4 legend: app / UI / fwd).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandKind {
    /// Handled entirely in-app (no bridge round-trip, no dedicated overlay).
    App,
    /// Opens a dedicated interactive panel / picker (the §4 **UI** commands).
    Ui,
    /// Core-forwarded: sent to `ga_bridge.py` as a `Command{name,args}` frame
    /// (the §4 **fwd** commands; the GA core intercepts the leading-slash itself).
    Fwd,
}

/// One slash command's metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlashCommand {
    /// The command name (without the leading `/`).
    pub name: &'static str,
    /// A one-line description (shown in the palette / `/help`).
    pub desc: &'static str,
    /// How the dispatcher handles it (app / UI / fwd).
    pub kind: CommandKind,
}

use CommandKind::{App, Fwd, Ui};

/// The slash-command registry (union of v2 + v3, ~33 — checklist §4). The cockpit
/// lists them for discovery + completion; the dispatcher routes by [`CommandKind`].
pub const COMMANDS: &[SlashCommand] = &[
    SlashCommand { name: "help", desc: "show the full command list", kind: Ui },
    SlashCommand { name: "status", desc: "model / state / rounds / context / cwd", kind: Ui },
    SlashCommand { name: "sessions", desc: "session snapshot (alias of /status)", kind: Ui },
    SlashCommand { name: "new", desc: "create + switch to a new session", kind: App },
    SlashCommand { name: "switch", desc: "switch session (dashboard)", kind: Ui },
    SlashCommand { name: "close", desc: "close the current session", kind: App },
    SlashCommand { name: "rename", desc: "rename the current session", kind: App },
    SlashCommand { name: "branch", desc: "fork the session with copied history", kind: App },
    SlashCommand { name: "rewind", desc: "rewind the last N real turns", kind: Ui },
    SlashCommand { name: "clear", desc: "clear the display (idle only)", kind: App },
    SlashCommand { name: "stop", desc: "abort the running task", kind: App },
    SlashCommand { name: "abort", desc: "abort the running task", kind: App },
    SlashCommand { name: "llm", desc: "view / switch the model", kind: Ui },
    SlashCommand { name: "btw", desc: "side-question (background, non-blocking)", kind: Ui },
    SlashCommand { name: "review", desc: "in-session code review", kind: Fwd },
    SlashCommand { name: "update", desc: "git pull GA + impact audit", kind: Fwd },
    SlashCommand { name: "autorun", desc: "autonomous-operation mode", kind: Fwd },
    SlashCommand { name: "morphling", desc: "absorb an external skill", kind: Fwd },
    SlashCommand { name: "goal", desc: "goal mode (progress in /workflows)", kind: Fwd },
    SlashCommand { name: "hive", desc: "multi-worker hive", kind: Fwd },
    SlashCommand { name: "conductor", desc: "conductor multi-subagent", kind: Fwd },
    SlashCommand { name: "workflows", desc: "live conductor / hive / goal panel", kind: Ui },
    SlashCommand { name: "scheduler", desc: "reflect tasks + cron status", kind: Ui },
    SlashCommand { name: "continue", desc: "searchable picker over session logs", kind: Ui },
    SlashCommand { name: "resume", desc: "GA core recovery prompt", kind: Fwd },
    SlashCommand { name: "cost", desc: "token usage (in/out/cache/context%)", kind: App },
    SlashCommand { name: "export", desc: "export the last reply (clip/file/all)", kind: Ui },
    SlashCommand { name: "restore", desc: "restore last model_responses into history", kind: App },
    SlashCommand { name: "reload-keys", desc: "hot-reload mykey.py", kind: App },
    SlashCommand { name: "language", desc: "switch interface language + repaint", kind: Ui },
    SlashCommand { name: "emoji", desc: "pet / spinner style", kind: Ui },
    SlashCommand { name: "effort", desc: "reasoning-effort slider (low…max)", kind: Ui },
    SlashCommand { name: "verbose", desc: "full-screen tool-call audit", kind: Ui },
    SlashCommand { name: "tools", desc: "full-screen tool-call audit (alias)", kind: Ui },
    SlashCommand { name: "trace", desc: "full-screen tool-call audit (alias)", kind: Ui },
    SlashCommand { name: "effects", desc: "effects intensity + demo splash", kind: App },
    SlashCommand { name: "mouse", desc: "toggle mouse capture (off = native drag-select to copy)", kind: App },
    SlashCommand { name: "theme", desc: "theme picker with live preview", kind: Ui },
    SlashCommand { name: "quit", desc: "quit tui_v4", kind: App },
    SlashCommand { name: "exit", desc: "quit tui_v4 (alias)", kind: App },
];

/// Max rows the palette shows at once (tui_v3's 6-row viewport).
pub const PALETTE_ROWS: usize = 6;

/// Resolve a typed command NAME (without the leading `/`, args already stripped)
/// to its [`SlashCommand`], case-insensitively. This is what the dispatcher calls
/// so EVERY §4 command name routes to a handler. PURE. Returns `None` for an
/// unknown name (the caller then offers [`did_you_mean`]).
pub fn resolve(name: &str) -> Option<&'static SlashCommand> {
    let n = name.trim().trim_start_matches('/');
    // Args may be glued on by a caller that passes the whole word; take the head.
    let n = n.split_whitespace().next().unwrap_or("");
    if n.is_empty() {
        return None;
    }
    COMMANDS.iter().find(|c| c.name.eq_ignore_ascii_case(n))
}

/// Split a `/command args…` line into `(name, args)`, both trimmed (name lowercased
/// for resolution by the caller). A bare `/` → empty name. PURE. The leading `/`
/// must already be present; callers pass `expanded.strip_prefix('/')`'s remainder
/// or the whole line.
pub fn split_command(line: &str) -> (String, String) {
    let rest = line.trim_start().strip_prefix('/').unwrap_or(line.trim_start());
    let mut parts = rest.splitn(2, char::is_whitespace);
    let name = parts.next().unwrap_or("").trim().to_string();
    let args = parts.next().unwrap_or("").trim().to_string();
    (name, args)
}

/// The commands matching the partial typed after `/`, fuzzy + ranked (the live
/// palette). Ranking (deterministic, stable): exact name first, then PREFIX
/// matches in registry order, then SUBSEQUENCE-fuzzy matches in registry order (so
/// `/cont`→continue by prefix, `/cnt`→continue by fuzzy). An empty partial (just
/// `/`) lists ALL commands in registry order. A space → args started → no palette.
/// PURE.
pub fn palette_matches(text: &str) -> Vec<SlashCommand> {
    let Some(rest) = text.strip_prefix('/') else {
        return Vec::new();
    };
    if rest.contains(char::is_whitespace) {
        return Vec::new();
    }
    let q = rest.to_ascii_lowercase();
    if q.is_empty() {
        return COMMANDS.to_vec();
    }
    let mut exact: Vec<SlashCommand> = Vec::new();
    let mut prefix: Vec<SlashCommand> = Vec::new();
    let mut fuzzy: Vec<SlashCommand> = Vec::new();
    for c in COMMANDS {
        let name = c.name;
        if name.eq_ignore_ascii_case(&q) {
            exact.push(*c);
        } else if name.starts_with(&q) {
            prefix.push(*c);
        } else if is_subsequence(&q, name) {
            fuzzy.push(*c);
        }
    }
    let mut out = exact;
    out.extend(prefix);
    out.extend(fuzzy);
    out
}

/// Whether `needle`'s chars appear in `haystack` in order (a subsequence / classic
/// fuzzy match). Case-insensitive on ASCII. PURE.
pub fn is_subsequence(needle: &str, haystack: &str) -> bool {
    let mut hay = haystack.bytes().map(|b| b.to_ascii_lowercase());
    'outer: for nb in needle.bytes().map(|b| b.to_ascii_lowercase()) {
        for hb in hay.by_ref() {
            if hb == nb {
                continue 'outer;
            }
        }
        return false; // ran out of haystack before matching this needle char.
    }
    true
}

/// Whether the palette should be shown for the current buffer: a partial `/word`
/// with at least one match and NOT already an exact-and-unique command name (once
/// the buffer is exactly `/help`, the palette hides so the next Enter runs it).
/// PURE.
pub fn palette_visible(text: &str, matches: &[SlashCommand]) -> bool {
    let Some(rest) = text.strip_prefix('/') else {
        return false;
    };
    if rest.contains(char::is_whitespace) || matches.is_empty() {
        return false;
    }
    // Hide once it's an exact command (only that one match and it equals rest).
    !(matches.len() == 1 && matches[0].name.eq_ignore_ascii_case(rest))
}

/// The closest command name to an UNKNOWN `typo` by Levenshtein edit distance, for
/// the "did you mean /x?" breadcrumb (§4 "Unknown command → friendly breadcrumb").
/// Returns `None` if nothing is within a small distance threshold (so we don't
/// suggest nonsense for a wildly different string). PURE.
pub fn did_you_mean(typo: &str) -> Option<&'static str> {
    let t = typo.trim().trim_start_matches('/').to_ascii_lowercase();
    if t.is_empty() {
        return None;
    }
    let mut best: Option<(&'static str, usize)> = None;
    for c in COMMANDS {
        let d = levenshtein(&t, c.name);
        match best {
            Some((_, bd)) if d >= bd => {}
            _ => best = Some((c.name, d)),
        }
    }
    // Threshold: at most a third of the longer length, min 2 — tolerant of a typo
    // but not of an unrelated word.
    best.and_then(|(name, d)| {
        let cap = (t.len().max(name.len()) / 3).max(2);
        if d <= cap {
            Some(name)
        } else {
            None
        }
    })
}

/// Classic Levenshtein edit distance (two-row DP). PURE; O(len(a)·len(b)) over
/// short command names, so cheap. ASCII-oriented (command names are ASCII).
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<u8> = a.bytes().collect();
    let b: Vec<u8> = b.bytes().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, &ai) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, &bj) in b.iter().enumerate() {
            let cost = if ai == bj { 0 } else { 1 };
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Complete a chosen command into the buffer as `/name ` (trailing space hides the
/// palette; a second Enter runs it). PURE.
pub fn complete_to(cmd: &SlashCommand) -> String {
    format!("/{} ", cmd.name)
}

/// The visible window of `matches` around the selected index (a scrolling viewport
/// of [`PALETTE_ROWS`] rows). Returns `(start, slice)`. PURE.
pub fn palette_window(matches: &[SlashCommand], sel: usize) -> (usize, Vec<SlashCommand>) {
    if matches.len() <= PALETTE_ROWS {
        return (0, matches.to_vec());
    }
    let half = PALETTE_ROWS / 2;
    let start = sel.saturating_sub(half).min(matches.len() - PALETTE_ROWS);
    (start, matches[start..start + PALETTE_ROWS].to_vec())
}

/// Move a palette selection by `delta`, clamped (no wrap, tui_v3 saturating). PURE.
pub fn move_sel(sel: usize, delta: isize, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let next = (sel as isize + delta).clamp(0, len as isize - 1);
    next as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE deliverable test: the registry resolves ALL ~33 §4 command names.
    /// Every name in the §4 union (UI / fwd / app) must `resolve()` to a command,
    /// case-insensitively, and report a sensible [`CommandKind`]; the count is the
    /// full union incl. aliases.
    #[test]
    fn registry_resolves_all_commands() {
        // The full §4 union (every name listed in checklist §4, incl. aliases).
        let all: &[&str] = &[
            "help", "status", "sessions", "new", "switch", "close", "rename", "branch",
            "rewind", "clear", "stop", "abort", "llm", "btw", "review", "update", "autorun",
            "morphling", "goal", "hive", "conductor", "workflows", "scheduler", "continue", "resume",
            "cost", "export", "restore", "reload-keys", "language", "emoji", "effort", "verbose",
            "tools", "trace", "effects", "mouse", "theme", "quit", "exit",
        ];
        for name in all {
            let c = resolve(name).unwrap_or_else(|| panic!("/{name} must resolve"));
            assert_eq!(c.name, *name);
            // Case-insensitive + leading-slash tolerant.
            assert!(resolve(&name.to_uppercase()).is_some(), "/{name} resolves uppercase");
            assert!(resolve(&format!("/{name}")).is_some(), "/{name} resolves with slash");
            // Resolution survives glued-on args (the dispatcher may pass the head).
            assert!(resolve(&format!("{name} some args")).is_some(), "/{name} resolves with args");
        }
        // The §4 spec says ~33; the union with aliases (incl. /effort) is 38.
        assert_eq!(COMMANDS.len(), all.len());
        assert!(COMMANDS.len() >= 33, "expected ~33+ commands, got {}", COMMANDS.len());

        // The kind legend is wired: the §4 fwd commands are Fwd, the UI ones Ui.
        assert_eq!(resolve("review").unwrap().kind, CommandKind::Fwd);
        assert_eq!(resolve("conductor").unwrap().kind, CommandKind::Fwd);
        assert_eq!(resolve("llm").unwrap().kind, CommandKind::Ui);
        assert_eq!(resolve("theme").unwrap().kind, CommandKind::Ui);
        assert_eq!(resolve("clear").unwrap().kind, CommandKind::App);

        // An unknown name does NOT resolve.
        assert!(resolve("definitely-not-a-command").is_none());
        assert!(resolve("").is_none());
        assert!(resolve("/").is_none());
    }

    #[test]
    fn palette_fuzzy_ranks_prefix_then_subsequence() {
        // `/re` matches rewind, review, resume, restore, reload-keys (prefix).
        let m = palette_matches("/re");
        assert!(m.iter().any(|c| c.name == "rewind"));
        assert!(m.iter().any(|c| c.name == "review"));
        assert!(m.iter().any(|c| c.name == "resume"));
        assert!(palette_visible("/re", &m));

        // A non-prefix SUBSEQUENCE finds the command (`cnt` ⊂ continue).
        let m = palette_matches("/cnt");
        assert!(m.iter().any(|c| c.name == "continue"), "fuzzy subsequence finds continue");

        // Exact unique match ranks FIRST.
        let m = palette_matches("/cost");
        assert_eq!(m.first().map(|c| c.name), Some("cost"));

        // A space → args started → no palette.
        assert!(palette_matches("/llm 2").is_empty());
        assert!(!palette_visible("/llm 2", &palette_matches("/llm 2")));

        // Exact unique command → palette hides (so Enter runs it).
        assert!(!palette_visible("/help", &palette_matches("/help")));

        // Non-slash text → nothing.
        assert!(palette_matches("hello").is_empty());

        // Just `/` → all commands listed.
        assert_eq!(palette_matches("/").len(), COMMANDS.len());
    }

    #[test]
    fn did_you_mean_suggests_closest() {
        // A one-edit typo → the intended command.
        assert_eq!(did_you_mean("/halp"), Some("help"));
        assert_eq!(did_you_mean("statuss"), Some("status"));
        assert_eq!(did_you_mean("/lln"), Some("llm"));
        // A wildly different word → no suggestion (above the distance cap).
        assert_eq!(did_you_mean("zzzzxyqwerty"), None);
        // Empty → None.
        assert_eq!(did_you_mean(""), None);
    }

    #[test]
    fn split_command_separates_name_and_args() {
        assert_eq!(split_command("/llm 2"), ("llm".into(), "2".into()));
        assert_eq!(split_command("/export clip"), ("export".into(), "clip".into()));
        assert_eq!(split_command("/help"), ("help".into(), "".into()));
        // Tolerant of leading whitespace + multi-space args.
        assert_eq!(split_command("  /review  fix the bug "), ("review".into(), "fix the bug".into()));
        // No leading slash still parses (defensive).
        assert_eq!(split_command("status"), ("status".into(), "".into()));
    }

    #[test]
    fn is_subsequence_matches_in_order() {
        assert!(is_subsequence("cnt", "continue"));
        assert!(is_subsequence("cont", "continue"));
        assert!(is_subsequence("", "anything"));
        assert!(!is_subsequence("xyz", "continue"));
        assert!(!is_subsequence(" continueX", "continue")); // longer than haystack.
    }

    #[test]
    fn complete_to_appends_trailing_space() {
        let cmd = COMMANDS.iter().find(|c| c.name == "rewind").unwrap();
        assert_eq!(complete_to(cmd), "/rewind ");
    }

    #[test]
    fn window_and_sel_movement_are_bounded() {
        let matches = palette_matches("/"); // all commands.
        assert!(matches.len() > PALETTE_ROWS);
        let (start, slice) = palette_window(&matches, 0);
        assert_eq!(start, 0);
        assert_eq!(slice.len(), PALETTE_ROWS);
        // Selecting near the end clamps the window to the tail.
        let (start_end, slice_end) = palette_window(&matches, matches.len() - 1);
        assert_eq!(start_end, matches.len() - PALETTE_ROWS);
        assert_eq!(slice_end.len(), PALETTE_ROWS);
        // move_sel saturates at both ends.
        assert_eq!(move_sel(0, -1, 5), 0);
        assert_eq!(move_sel(4, 1, 5), 4);
        assert_eq!(move_sel(2, 1, 5), 3);
    }
}
