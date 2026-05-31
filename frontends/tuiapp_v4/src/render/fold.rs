//! render/fold.rs — per-turn incremental folding + structural-boundary stream
//! commit (checklist §10: "Per-turn incremental fold (turn N → ▸ summary when
//! N+1 starts)" and "Structural-boundary stream commit (split at safe pos; no
//! orphan tool headers)"; tui_v3 §F 69-70).
//!
//! GA's assistant text is a sequence of TURNS delimited by `Turn N ...` markers
//! (bare, tui_v4's non-verbose mode — or legacy bold `**Turn N ...**`). Each turn
//! usually carries a `<summary>…</summary>` and one or more compact
//! `🛠️ name(args)` calls. The cockpit:
//!   * folds every COMPLETED turn (turn k where k < the last turn marker seen)
//!     to a one-line `▸ <summary>` header — exactly when turn k+1's marker
//!     arrives — and keeps the final (in-progress) turn expanded;
//!   * commits the streaming tail only up to a STRUCTURAL boundary (`safe_pos`),
//!     never inside a half-written tool block / summary / turn marker, so a chip
//!     header is never orphaned and a later regex reshape can't duplicate it.
//!
//! All PURE over the text + a width; unit-tested (`fold_summary` is the
//! deliverable). The renderer projects the fold segments to lines.

/// One segment of a turn-folded assistant message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FoldSegment {
    /// Prose / the final (expanded) turn body — rendered in full.
    Text { body: String },
    /// A completed, FOLDED turn — rendered as a single `▸ <title>` header. The
    /// full `body` replays when the fold is toggled open (Ctrl+O).
    Fold {
        /// 1-based turn number (0 for the pre-first-marker preamble).
        turn: u32,
        /// The one-line fold title (the turn's `<summary>`, else a derived hint).
        title: String,
        /// The full source of the folded turn (shown when expanded).
        body: String,
    },
}

impl FoldSegment {
    /// The one-line header text a renderer draws for a folded turn.
    #[allow(dead_code)] // header accessor (tested; the transcript inlines the `▸` itself).
    pub fn header(&self) -> Option<String> {
        match self {
            FoldSegment::Fold { title, .. } => Some(format!("▸ {title}")),
            FoldSegment::Text { .. } => None,
        }
    }
}

/// Split assistant `text` into fold/text segments, folding every COMPLETED turn
/// (all but the last) to a `▸ summary` header and keeping the last turn expanded.
/// PURE (tui_v3 `_fold_turns` port). When `fold_all` is true, the LAST turn is
/// folded too (the Ctrl+O "fold everything" toggle).
///
/// With 0 or 1 turn markers the whole text is a single `Text` segment (nothing to
/// fold yet — the incremental fold needs ≥2 turns).
pub fn fold_turns(text: &str, fold_all: bool) -> Vec<FoldSegment> {
    let markers = find_turn_markers(text);
    if markers.len() < 2 && !fold_all {
        return vec![FoldSegment::Text {
            body: text.to_string(),
        }];
    }
    if markers.is_empty() {
        return vec![FoldSegment::Text {
            body: text.to_string(),
        }];
    }

    let mut segs: Vec<FoldSegment> = Vec::new();

    // Preamble before the first marker (turn 0) — prose, kept as text.
    let first = markers[0];
    if first.start > 0 {
        let pre = &text[..first.start];
        if !pre.trim().is_empty() {
            segs.push(FoldSegment::Text {
                body: pre.to_string(),
            });
        }
    }

    // Each turn spans from its marker to the next marker (or end).
    for (i, m) in markers.iter().enumerate() {
        let body_start = m.start;
        let body_end = markers
            .get(i + 1)
            .map(|next| next.start)
            .unwrap_or(text.len());
        let body = &text[body_start..body_end];
        let is_last = i + 1 == markers.len();
        if is_last && !fold_all {
            // The in-progress turn stays expanded.
            segs.push(FoldSegment::Text {
                body: body.to_string(),
            });
        } else {
            let title = turn_title(body, m.number);
            segs.push(FoldSegment::Fold {
                turn: m.number,
                title,
                body: body.to_string(),
            });
        }
    }
    segs
}

/// A located `**Turn N ...**` marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TurnMarker {
    /// Byte offset of the marker start (`**`).
    start: usize,
    /// The turn number N.
    number: u32,
}

/// Find every turn marker in `text`, in BOTH GA forms: the BARE `Turn N ...`
/// (tui_v4's non-verbose mode, agent_loop.py:62) and the legacy bold
/// `**Turn N ...**` / `**LLM Running (Turn N) ...**` (verbose). A marker must sit
/// at a LINE START so a `Turn` inside prose is never a false boundary. PURE; a
/// hand-rolled scan (no regex dep needed for this shape).
fn find_turn_markers(text: &str) -> Vec<TurnMarker> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < text.len() {
        let at_line_start = i == 0 || bytes.get(i - 1) == Some(&b'\n');
        if at_line_start {
            if let Some((num, end)) = parse_turn_marker(&text[i..]) {
                out.push(TurnMarker {
                    start: i,
                    number: num,
                });
                i += end.max(1);
                continue;
            }
        }
        i += 1;
    }
    out
}

/// Parse a turn marker at the START of `s`. Returns `(turn_number,
/// byte_len_consumed)` or `None`. Accepts BOTH the bare form (`Turn N ...`, the
/// tui_v4 non-verbose stream) and the bold form (`**...Turn N...**`, verbose).
/// PURE.
fn parse_turn_marker(s: &str) -> Option<(u32, usize)> {
    // The marker is a single line.
    let line_end = s.find('\n').unwrap_or(s.len());
    let line = &s[..line_end];
    // Strip an optional opening `**` (bold/verbose form).
    let inner = line.strip_prefix("**").unwrap_or(line);
    // Find "Turn " followed by digits.
    let tpos = inner.find("Turn ")?;
    let after = &inner[tpos + 5..];
    let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    let num: u32 = digits.parse().ok()?;
    // Consume the whole marker line (both forms are single-line).
    Some((num, line.len()))
}

/// Derive a one-line fold title for a turn body: prefer its `<summary>…</summary>`,
/// else the first compact tool name (`🛠️ name(args)`), else the first non-marker
/// prose line, else a generic "Turn N". PURE.
fn turn_title(body: &str, number: u32) -> String {
    if let Some(s) = extract_summary(body) {
        let cleaned = collapse_ws(&s);
        if !cleaned.is_empty() {
            return cleaned;
        }
    }
    if let Some(name) = first_tool_name(body) {
        return format!("Turn {number} · {name}");
    }
    // Fall back to the first non-empty, non-marker line. Skip turn-marker lines
    // in BOTH the bare (`Turn N ...`) and bold (`**Turn N ...**`) forms so the
    // marker itself is never the title.
    for line in body.lines() {
        let l = line.trim();
        if l.is_empty() || crate::render::chip::find_turn_line(l) == Some(0) {
            continue;
        }
        return collapse_ws(l);
    }
    format!("Turn {number}")
}

/// Extract the first `<summary>…</summary>` body from `text`. PURE.
pub fn extract_summary(text: &str) -> Option<String> {
    let open = text.find("<summary>")? + "<summary>".len();
    let rest = &text[open..];
    let close = rest.find("</summary>")?;
    Some(rest[..close].trim().to_string())
}

/// The first tool name in `text` from a COMPACT `🛠️ NAME(ARGS)` marker
/// (agent_loop.py:89). PURE.
fn first_tool_name(text: &str) -> Option<String> {
    let calls = crate::render::chip::parse_tool_calls(text);
    calls.first().map(|c| c.name.clone())
}

/// Collapse internal whitespace runs to single spaces and trim. PURE.
fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ---------------------------------------------------------------------------
// Structural-boundary stream commit (tui_v3 `_safe_pos`).
// ---------------------------------------------------------------------------

/// The byte offset up to which `stream` is STRUCTURALLY stable to commit (no
/// in-flight tool block / summary / turn marker past it). Committing only up to
/// this point means a later regex reshape can't duplicate a half-written chip and
/// no tool header is ever orphaned. PURE.
///
/// Unsafe (keep volatile) starting at the LAST of:
///   * a `🛠️ ` compact tool header whose result line is still being written (the
///     header line has no trailing `\n` yet — so the name/args could still grow),
///   * a `<summary>` / `<thinking>` with no closing tag,
///   * a `**` (start of a possibly-incomplete `**Turn N …**` marker) with no
///     closing `**` yet.
/// Falls back to the last paragraph boundary (`\n\n`), else 0 (commit nothing
/// until a boundary exists).
pub fn safe_commit_pos(stream: &str) -> usize {
    use crate::render::chip::TOOL_MARK;
    let mut unsafe_from: Option<usize> = None;
    fn note(pos: usize, current: &mut Option<usize>) {
        *current = Some(current.map_or(pos, |c| c.max(pos)));
    }

    // In-flight compact tool header: a `🛠️ ` whose header line hasn't completed
    // (no `\n` after it yet → the `name(args)` could still be growing). Once the
    // header line is terminated by a newline, the chip is structurally parseable.
    if let Some(p) = stream.rfind(TOOL_MARK) {
        let after = p + TOOL_MARK.len();
        let header_unterminated = !stream[after..].contains('\n');
        if header_unterminated {
            note(p, &mut unsafe_from);
        }
    }

    // Unclosed `<summary>` / `<thinking>`.
    for tag in ["<summary>", "<thinking>"] {
        if let Some(p) = stream.rfind(tag) {
            let close = match tag {
                "<summary>" => "</summary>",
                _ => "</thinking>",
            };
            if !stream[p..].contains(close) {
                note(p, &mut unsafe_from);
            }
        }
    }

    // A trailing `**` that hasn't closed into a complete marker yet.
    if let Some(p) = stream.rfind("**") {
        // If there is an odd number of `**` (an unmatched opener) treat the last
        // one as in-flight.
        let count = stream.matches("**").count();
        if count % 2 == 1 {
            note(p, &mut unsafe_from);
        }
    }

    match unsafe_from {
        Some(u) => {
            // Commit up to the last paragraph boundary BEFORE the unsafe region.
            stream[..u]
                .rfind("\n\n")
                .map(|i| i + 2)
                .unwrap_or(0)
        }
        None => {
            // Fully stable: commit up to the last paragraph boundary, else all.
            stream
                .rfind("\n\n")
                .map(|i| i + 2)
                .unwrap_or(stream.len())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE deliverable test: per-turn folding collapses completed turns to a
    /// `▸ summary` header and keeps the final turn expanded.
    #[test]
    fn fold_summary() {
        // Two turns: turn 1 has a summary + a tool; turn 2 is in progress. BARE
        // GA markers (tui_v4 non-verbose) + the COMPACT tool form.
        let text = "\
Turn 1 ...
<summary>Read the config and found the port</summary>
🛠️ file_read(config.toml)
result here
Turn 2 ...
<summary>Now editing the file</summary>
still working on it";

        let segs = fold_turns(text, false);
        // Turn 1 folds; turn 2 stays as expanded text.
        let folds: Vec<&FoldSegment> = segs
            .iter()
            .filter(|s| matches!(s, FoldSegment::Fold { .. }))
            .collect();
        assert_eq!(folds.len(), 1, "exactly the one COMPLETED turn folds");
        match folds[0] {
            FoldSegment::Fold { turn, title, .. } => {
                assert_eq!(*turn, 1);
                // The fold title is the turn's <summary>.
                assert_eq!(title, "Read the config and found the port");
            }
            _ => unreachable!(),
        }
        // The fold header renders as `▸ <summary>`.
        assert_eq!(
            folds[0].header().as_deref(),
            Some("▸ Read the config and found the port")
        );
        // The final (in-progress) turn is present as expanded text.
        assert!(segs.iter().any(|s| matches!(s, FoldSegment::Text { body } if body.contains("still working on it"))));

        // A single-turn (or no-marker) text does NOT fold (needs ≥2 turns).
        let one = fold_turns("Turn 1 ...\n<summary>only turn</summary>\nbody", false);
        assert_eq!(one.len(), 1);
        assert!(matches!(one[0], FoldSegment::Text { .. }));

        // fold_all collapses the last turn too.
        let all = fold_turns(text, true);
        assert!(all.iter().all(|s| matches!(s, FoldSegment::Fold { .. } | FoldSegment::Text { .. })));
        assert_eq!(
            all.iter().filter(|s| matches!(s, FoldSegment::Fold { .. })).count(),
            2,
            "fold_all folds BOTH turns"
        );
    }

    #[test]
    fn turn_title_falls_back_to_tool_then_generic() {
        // No summary → first (compact) tool name.
        let body = "Turn 3 ...\n🛠️ web_search(query: rust)";
        assert_eq!(turn_title(body, 3), "Turn 3 · web_search");
        // No summary, no tool → first prose line.
        let body2 = "Turn 4 ...\njust some prose here";
        assert_eq!(turn_title(body2, 4), "just some prose here");
    }

    #[test]
    fn find_turn_markers_parses_both_forms_and_numbers() {
        // Mixed: a BARE `Turn 1 ...` and a BOLD `**LLM Running (Turn 2) ...**`.
        let text = "pre\nTurn 1 ...\nbody\n**LLM Running (Turn 2) ...**\nmore";
        let segs = fold_turns(text, false);
        // The preamble "pre" is its own text segment.
        assert!(matches!(&segs[0], FoldSegment::Text { body } if body.contains("pre")));
        // Turn 1 folds (number 1); turn 2 stays expanded.
        assert!(segs.iter().any(|s| matches!(s, FoldSegment::Fold { turn: 1, .. })));
    }

    #[test]
    fn safe_commit_pos_holds_back_in_flight_structures() {
        // A complete paragraph then an in-flight COMPACT tool header (no newline
        // after the marker yet → the name/args could still grow) → commit only the
        // stable paragraph, hold the half-written tool header.
        let s = "done paragraph.\n\nmore stable text.\n\n🛠️ web_sc";
        let pos = safe_commit_pos(s);
        let committed = &s[..pos];
        assert!(committed.contains("more stable text."));
        assert!(!committed.contains("🛠️"), "the in-flight tool header is held back");

        // An unclosed <summary> is held back.
        let s2 = "stable.\n\n<summary>partial sum";
        let pos2 = safe_commit_pos(s2);
        assert!(!s2[..pos2].contains("<summary>"));

        // Stable text with a paragraph break: commit the closed paragraph, keep
        // the final (still-growing) paragraph volatile (it may continue).
        let s3 = "all done.\n\nsecond paragraph still typing";
        let pos3 = safe_commit_pos(s3);
        assert!(s3[..pos3].contains("all done."));
        assert!(!s3[..pos3].contains("still typing"), "final paragraph stays live");

        // No paragraph break at all + no in-flight structure → commit everything
        // (there is nothing being structurally assembled).
        let s3b = "one single closed line.";
        assert_eq!(safe_commit_pos(s3b), s3b.len());

        // A COMPLETED compact tool header (its line is terminated) followed by a
        // result paragraph + a turn marker is safe to commit past the result.
        let s4 = "🛠️ x(arg)\nresult\n\nTurn 2 ...\nnext line growing";
        let pos4 = safe_commit_pos(s4);
        assert!(s4[..pos4].contains("result"));
    }
}
