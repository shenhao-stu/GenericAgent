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
    /// `Some(primary)` if this is an ALIAS of another command — it still resolves
    /// + completes (so the typed alias keeps working), but `/help` lists it dimmed
    /// as "alias of /primary" under its primary instead of as a peer row, and the
    /// palette doesn't surface an alias+primary as two competing primary hits.
    pub alias_of: Option<&'static str>,
}

use CommandKind::{App, Fwd, Ui};

/// A primary (non-alias) command shorthand (`alias_of: None`).
const fn cmd(name: &'static str, desc: &'static str, kind: CommandKind) -> SlashCommand {
    SlashCommand { name, desc, kind, alias_of: None }
}

/// An ALIAS command shorthand (`alias_of: Some(primary)`).
const fn alias(name: &'static str, desc: &'static str, kind: CommandKind, of: &'static str) -> SlashCommand {
    SlashCommand { name, desc, kind, alias_of: Some(of) }
}

/// The slash-command registry (union of v2 + v3, ~33 — checklist §4). The cockpit
/// lists them for discovery + completion; the dispatcher routes by [`CommandKind`].
pub const COMMANDS: &[SlashCommand] = &[
    cmd("help", "show the full command list", Ui),
    cmd("keybindings", "show the keyboard shortcuts", Ui),
    cmd("status", "model / state / rounds / context / cwd", Ui),
    alias("sessions", "session snapshot", Ui, "status"),
    cmd("new", "create + switch to a new session", App),
    cmd("switch", "switch session (dashboard)", Ui),
    cmd("close", "close the current session", App),
    cmd("rename", "rename the current session", App),
    cmd("branch", "fork the session with copied history", App),
    cmd("rewind", "rewind the last N real turns", Ui),
    cmd("clear", "clear the display (idle only)", App),
    cmd("stop", "abort the running task", App),
    alias("abort", "abort the running task", App, "stop"),
    cmd("llm", "view / switch the model", Ui),
    cmd("btw", "side-question (background, non-blocking)", Ui),
    cmd("review", "in-session code review", Fwd),
    cmd("update", "git pull GA + impact audit", Fwd),
    cmd("autorun", "autonomous-operation mode", Fwd),
    cmd("morphling", "absorb an external skill", Fwd),
    cmd("goal", "goal mode (progress in /workflows)", Fwd),
    cmd("hive", "multi-worker hive", Fwd),
    cmd("conductor", "conductor multi-subagent", Fwd),
    cmd("workflows", "live conductor / hive / goal panel", Ui),
    cmd("scheduler", "reflect tasks + cron status", Ui),
    cmd("continue", "searchable picker over session logs", Ui),
    cmd("resume", "GA core recovery prompt", Fwd),
    cmd("cost", "token usage (in/out/cache/context%)", App),
    cmd("export", "export the last reply (clip/file/all)", Ui),
    cmd("restore", "restore last model_responses into history", App),
    cmd("reload-keys", "hot-reload mykey.py", App),
    cmd("language", "switch interface language + repaint", Ui),
    cmd("emoji", "spinner or pet companion style", Ui),
    cmd("effort", "reasoning-effort slider (low…max)", Ui),
    cmd("verbose", "full-screen tool-call audit", Ui),
    alias("tools", "full-screen tool-call audit", Ui, "verbose"),
    alias("trace", "full-screen tool-call audit", Ui, "verbose"),
    cmd("fold", "fold / unfold all completed tool chips", App),
    cmd("mouse", "toggle mouse mode (native select ↔ interactive click)", App),
    cmd("theme", "theme picker with live preview", Ui),
    cmd("quit", "quit tui_v4", App),
    alias("exit", "quit tui_v4", App, "quit"),
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
    // De-dup: when an alias AND its primary both match the partial, drop the alias
    // so the palette never offers `/tools` and `/verbose` as two competing rows
    // (the alias still resolves + completes when typed alone). An alias whose
    // primary is NOT in the match set survives (so `/abort` still completes).
    let present: Vec<&'static str> = out.iter().map(|c| c.name).collect();
    out.retain(|c| match c.alias_of {
        Some(primary) => !present.contains(&primary),
        None => true,
    });
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
    let mut best: Option<(&'static SlashCommand, usize)> = None;
    for c in COMMANDS {
        let d = levenshtein(&t, c.name);
        let better = match best {
            None => true,
            Some((bc, bd)) => {
                // Strictly closer wins; on a TIE prefer a PRIMARY over an alias so
                // the breadcrumb never flaps between equidistant aliases (C5 F7).
                d < bd || (d == bd && bc.alias_of.is_some() && c.alias_of.is_none())
            }
        };
        if better {
            best = Some((c, d));
        }
    }
    let best = best.map(|(c, d)| (c.name, d));
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

/// The PRIMARY commands of a [`CommandKind`] in registry order (aliases excluded).
/// `/help` lists these as peer rows; the aliases hang under their primary as dim
/// "alias of /X" lines (see [`aliases_of`]). PURE.
pub fn primaries_of_kind(kind: CommandKind) -> impl Iterator<Item = &'static SlashCommand> {
    COMMANDS.iter().filter(move |c| c.kind == kind && c.alias_of.is_none())
}

/// The aliases that point at `primary` (in registry order), for the `/help` dim
/// "alias of /primary" lines. PURE.
pub fn aliases_of(primary: &str) -> impl Iterator<Item = &'static SlashCommand> + '_ {
    COMMANDS.iter().filter(move |c| c.alias_of == Some(primary))
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
    /// full union incl. the 5 marked aliases. `/mouse` is GONE (Q10 D4) and the
    /// `/keybindings` + `/fold` rows are present (Q7 / Slice 6).
    #[test]
    fn registry_resolves_all_commands() {
        // The full §4 union (every name listed in checklist §4, incl. the 5 marked
        // aliases). `/mouse` was dropped (Q10); `/keybindings` + `/fold` were added.
        let all: &[&str] = &[
            "help", "keybindings", "status", "sessions", "new", "switch", "close", "rename", "branch",
            "rewind", "clear", "stop", "abort", "llm", "btw", "review", "update", "autorun",
            "morphling", "goal", "hive", "conductor", "workflows", "scheduler", "continue", "resume",
            "cost", "export", "restore", "reload-keys", "language", "emoji", "effort", "verbose",
            "tools", "trace", "fold", "mouse", "theme", "quit", "exit",
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
        // The §4 spec says ~33; the union with the 6 marked aliases is 42 (S1 added
        // /mouse back as a discoverable App command).
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

        // Slice 7: `/effects` (the COMMAND) was removed — it no longer resolves and is
        // gone from COMMANDS + the palette (the effects ENGINE keeps running on its own).
        assert!(resolve("effects").is_none(), "/effects no longer resolves");
        assert!(resolve("/effects").is_none(), "/effects (with slash) no longer resolves");
        assert!(!COMMANDS.iter().any(|c| c.name == "effects"), "/effects is gone from COMMANDS");
        assert!(
            !palette_matches("/effects").iter().any(|c| c.name == "effects"),
            "/effects does not surface in the palette"
        );
    }

    /// Q6 — the marked alias commands are MARKED (`alias_of`) at their primary, still
    /// RESOLVE (so the typed alias keeps working), but the palette does not surface
    /// an alias AND its primary as two competing rows (dedup), and each primary they
    /// point at exists. This is the "no duplicate primary name; aliases still work"
    /// resolution of the C3-dedup / C5-D7 specs.
    #[test]
    fn aliases_marked_not_duplicated() {
        // Exactly these aliases, each pointing at its primary. (`/emoji` became a
        // PRIMARY command in round-5 — `/pets` was removed — so it is NOT listed here.)
        let expected = [
            ("sessions", "status"),
            ("abort", "stop"),
            ("tools", "verbose"),
            ("trace", "verbose"),
            ("exit", "quit"),
        ];
        for (a, primary) in expected {
            let cmd = resolve(a).unwrap_or_else(|| panic!("/{a} must still resolve"));
            assert_eq!(cmd.alias_of, Some(primary), "/{a} is marked alias of /{primary}");
            // The primary it points at is a real, non-alias command.
            let p = resolve(primary).unwrap_or_else(|| panic!("primary /{primary} must exist"));
            assert_eq!(p.alias_of, None, "/{primary} is a primary, not itself an alias");
        }
        // Those are the ONLY aliases — nothing else carries `alias_of`.
        let marked: Vec<&str> = COMMANDS.iter().filter(|c| c.alias_of.is_some()).map(|c| c.name).collect();
        assert_eq!(marked.len(), expected.len(), "exactly {} aliases marked, got {marked:?}", expected.len());

        // DEDUP: typing a prefix that hits BOTH an alias and its primary surfaces
        // the PRIMARY only — `/verbose` is present, neither `tools` nor `trace`
        // appears as a competing primary row (they share no prefix with verbose, so
        // the real collision is the EXPLICIT `aliases_of`/`primaries_of_kind` split;
        // assert that split here).
        let ui_primaries: Vec<&str> = primaries_of_kind(CommandKind::Ui).map(|c| c.name).collect();
        assert!(ui_primaries.contains(&"verbose"), "verbose is a UI primary row");
        assert!(!ui_primaries.contains(&"tools"), "tools is NOT a peer primary row");
        assert!(!ui_primaries.contains(&"trace"), "trace is NOT a peer primary row");
        // The aliases hang under their primary instead.
        let verbose_aliases: Vec<&str> = aliases_of("verbose").map(|c| c.name).collect();
        assert_eq!(verbose_aliases, vec!["tools", "trace"], "tools+trace are listed under /verbose");

        // A SHARED-PREFIX collision is the load-bearing palette case: `/st` matches
        // BOTH `status` and (fuzzily) others, and `stop`+`abort` is the alias pair —
        // typing a partial matching both `stop` and its alias `abort` must NOT show
        // both. `abort` shares no prefix with `stop`, but a fuzzy `/aot` hits both;
        // verify the alias is dropped when its primary co-occurs, and survives alone.
        let both = palette_matches("/stop"); // exact `stop`; `abort` is not a match here
        assert!(both.iter().any(|c| c.name == "stop"));
        // When ONLY the alias matches (its primary absent), it survives so it still
        // completes — `/abort` exact.
        let alias_only = palette_matches("/abort");
        assert!(alias_only.iter().any(|c| c.name == "abort"), "/abort still completes alone");
    }

    /// S1 — `/mouse` is back in the registry as an App command so it's discoverable
    /// in the palette. It toggles between native mode (drag-to-copy) and interactive
    /// mode (click ▸/▾ fold). The Ctrl+Shift+M chord still works in parallel.
    #[test]
    fn mouse_command_present() {
        let cmd = resolve("mouse").expect("/mouse must be in the registry (S1)");
        assert_eq!(cmd.name, "mouse");
        assert_eq!(cmd.kind, CommandKind::App, "/mouse is an App command");
        assert!(COMMANDS.iter().any(|c| c.name == "mouse"), "/mouse is in COMMANDS");
        assert!(
            palette_matches("/mouse").iter().any(|c| c.name == "mouse"),
            "/mouse surfaces in the palette"
        );
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
