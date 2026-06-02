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

use std::collections::HashMap;

/// A clickable, toggleable node in a rendered assistant block (Fix E / Q8). A
/// left-click on its triangle/bullet column flips its fold via
/// [`crate::app::AppState::toggle_fold`]; the renderer keys the per-node fold
/// override map on this. Two kinds:
///   * [`NodeId::Turn`] — a foldable TURN (its `▸ summary` header, or the turn's
///     body when expanded), keyed `(block_id, turn)` (the turn number is stable
///     across a stream — a later turn appends a higher number, never renumbering).
///   * [`NodeId::Tool`] — a foldable TOOL-CALL result, keyed `(block_id, tool)`
///     where `tool` is the 0-based index of the call in the WHOLE block (stable +
///     monotonic across the stream: a new call appends a higher index).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeId {
    /// A foldable turn, by `(block_id, turn_number)`.
    Turn { block: u64, turn: u32 },
    /// A foldable tool-call result, by `(block_id, block_global_tool_index)`.
    Tool { block: u64, tool: u32 },
}

/// The fold view for ONE assistant block: the global `fold_all` flag plus the
/// sparse per-node overrides (absent ⇒ the default policy). Passed down the
/// cockpit render so [`fold_turns_with`] (turns) and the chip layer (tool results)
/// resolve each node's fold the SAME way. Cheap (two refs); a stable digest of the
/// block-relevant overrides keys the block's render memo ([`Self::digest`]).
#[derive(Clone, Copy)]
pub struct BlockFolds<'a> {
    /// The block these overrides apply to (so `Self::is_*` only consult its keys).
    pub block_id: u64,
    /// The global "fold/unfold all" flag (Ctrl+Shift+O / `/fold`) — the DEFAULT a
    /// node falls back to when it has no explicit override.
    pub fold_all: bool,
    /// The sparse per-node overrides (`None` ⇒ no overrides at all, the common
    /// case for a freshly-streamed block).
    pub overrides: Option<&'a HashMap<NodeId, bool>>,
}

impl<'a> BlockFolds<'a> {
    /// A no-override view for `block_id` at `fold_all` — the default fold policy
    /// (completed turns folded, last open; tool results truncated). Used by the
    /// convenience renders + the row-count tests.
    pub fn plain(block_id: u64, fold_all: bool) -> Self {
        BlockFolds { block_id, fold_all, overrides: None }
    }

    /// Whether TURN `turn` of this block is folded: an explicit override wins, else
    /// the default ([`default_turn_folded`]). The `is_last` flag distinguishes the
    /// in-progress turn (kept open by default) from a completed one.
    pub(crate) fn turn_folded(&self, turn: u32, is_last: bool) -> bool {
        self.lookup(NodeId::Turn { block: self.block_id, turn })
            .unwrap_or_else(|| default_turn_folded(is_last, self.fold_all))
    }

    /// Whether TOOL `tool` (block-global index) of this block is EXPANDED (its full
    /// result shown). Tool results default to COLLAPSED (truncated preview); an
    /// override of `true` expands. The fold map stores `true` = expanded for a tool.
    pub(crate) fn tool_expanded(&self, tool: u32) -> bool {
        self.lookup(NodeId::Tool { block: self.block_id, tool }).unwrap_or(false)
    }

    fn lookup(&self, node: NodeId) -> Option<bool> {
        self.overrides.and_then(|m| m.get(&node).copied())
    }

    /// A stable 64-bit digest of the overrides that affect THIS block — the cockpit
    /// render memo keys on it so a per-node toggle (which changes the projected
    /// rows) misses the cache and re-renders, while an unrelated block's toggle does
    /// not. Order-independent (XOR of per-entry hashes) so it's deterministic
    /// regardless of `HashMap` iteration order. `0` ⇒ no overrides for this block.
    pub fn digest(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let Some(map) = self.overrides else { return 0 };
        let mut acc = 0u64;
        for (node, &val) in map {
            let owner = match node {
                NodeId::Turn { block, .. } | NodeId::Tool { block, .. } => *block,
            };
            if owner != self.block_id {
                continue;
            }
            let mut h = std::collections::hash_map::DefaultHasher::new();
            node.hash(&mut h);
            val.hash(&mut h);
            acc ^= h.finish();
        }
        acc
    }
}

/// The DEFAULT fold for a turn: a COMPLETED turn (`!is_last`) folds, the in-progress
/// last turn stays open — unless `fold_all` forces everything folded. PURE.
pub fn default_turn_folded(is_last: bool, fold_all: bool) -> bool {
    fold_all || !is_last
}

/// One segment of a turn-folded assistant message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FoldSegment {
    /// Prose / the final (expanded) turn body — rendered in full.
    ///
    /// `turn` is `Some(n)` when this text came from a known turn marker (the
    /// 1-based turn number). `None` for the pre-marker preamble or a whole-
    /// message text where no turn markers were found. The renderer uses this to
    /// emit a `▾` collapse header (tagged `NodeId::Turn`) for expanded non-
    /// preamble turns so click-to-collapse works (S1 Fix B).
    Text {
        body: String,
        /// The 1-based turn number, or `None` for the preamble / single-turn text.
        turn: Option<u32>,
    },
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

/// Make `turn_title` accessible to the renderer for generating ▾ headers.
pub(crate) fn turn_title_pub(body: &str) -> String {
    turn_title(body)
}

/// Split assistant `text` into fold/text segments, folding every COMPLETED turn
/// (all but the last) to a `▸ summary` header and keeping the last turn expanded.
/// PURE (tui_v3 `_fold_turns` port). When `fold_all` is true, the LAST turn is
/// folded too (the Ctrl+O "fold everything" toggle).
///
/// With 0 or 1 turn markers the whole text is a single `Text` segment (nothing to
/// fold yet — the incremental fold needs ≥2 turns).
// The default-policy convenience over `fold_turns_with`; the render path now calls
// `fold_turns_with` with the per-node closure, so this is exercised by the fold
// unit tests (the deliverable `fold_summary` + the no-Turn-label gates).
#[allow(dead_code)]
pub fn fold_turns(text: &str, fold_all: bool) -> Vec<FoldSegment> {
    // The default policy: a completed turn folds, the last stays open (or all fold
    // when `fold_all`). Per-node overrides go through [`fold_turns_with`].
    fold_turns_with(text, fold_all, |_turn, is_last| default_turn_folded(is_last, fold_all))
}

/// Like [`fold_turns`], but the renderer decides EACH turn's fold via the
/// `is_folded(turn_number, is_last) -> bool` closure (Fix E / Q8) instead of the
/// fixed "all-but-last" policy. This is the single variation point per-node fold
/// rides on: the default closure reproduces [`fold_turns`] exactly; an
/// [`AppState`](crate::app::AppState) override map makes one node fold/unfold
/// independently. PURE over `(text, closure)`.
///
/// The `< 2 markers && nothing-forced-folded` short-circuit (a single-turn message
/// has nothing to fold yet) still holds: we only take the per-turn path once there
/// are ≥2 markers OR a turn is forced folded by the closure.
pub fn fold_turns_with(
    text: &str,
    fold_all: bool,
    is_folded: impl Fn(u32, bool) -> bool,
) -> Vec<FoldSegment> {
    let markers = find_turn_markers(text);
    if markers.is_empty() {
        return vec![FoldSegment::Text {
            body: text.to_string(),
            turn: None,
        }];
    }
    // With a single turn marker the incremental fold needs nothing folded UNLESS the
    // closure (an explicit override / fold_all) folds it. Probe the lone turn first.
    if markers.len() < 2 {
        let only = markers[0];
        if !fold_all && !is_folded(only.number, true) {
            // Single expanded turn: carry its turn number so the renderer can emit ▾.
            return vec![FoldSegment::Text {
                body: text.to_string(),
                turn: Some(only.number),
            }];
        }
        // else: fall through and let the loop fold it (preamble + the folded turn).
    }

    let mut segs: Vec<FoldSegment> = Vec::new();

    // Preamble before the first marker (turn 0) — prose, kept as text with turn=None.
    let first = markers[0];
    if first.start > 0 {
        let pre = &text[..first.start];
        if !pre.trim().is_empty() {
            segs.push(FoldSegment::Text {
                body: pre.to_string(),
                turn: None, // preamble: no turn number
            });
        }
    }

    // Each turn spans from its marker to the next marker (or end). The closure
    // decides its fold per turn (completed-folded/last-open default, override wins).
    for (i, m) in markers.iter().enumerate() {
        let body_start = m.start;
        let body_end = markers
            .get(i + 1)
            .map(|next| next.start)
            .unwrap_or(text.len());
        let body = &text[body_start..body_end];
        let is_last = i + 1 == markers.len();
        if is_folded(m.number, is_last) {
            let title = turn_title(body);
            segs.push(FoldSegment::Fold {
                turn: m.number,
                title,
                body: body.to_string(),
            });
        } else {
            // Expanded turn: carry its turn number for the ▾ collapse header.
            segs.push(FoldSegment::Text {
                body: body.to_string(),
                turn: Some(m.number),
            });
        }
    }
    segs
}

/// The number of the LAST turn marker in `text`, or `None` if there are no markers.
/// Used by the per-node fold default (the last turn stays open). PURE.
pub fn last_turn_number(text: &str) -> Option<u32> {
    find_turn_markers(text).last().map(|m| m.number)
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
/// prose line, else a neutral ellipsis. The literal `Turn N` is NEVER baked into a
/// title (Q8: "绝不能再出现 Turn 1 …") — the turn number is spacing, not a label. PURE.
fn turn_title(body: &str) -> String {
    if let Some(s) = extract_summary(body) {
        let cleaned = collapse_ws(&s);
        if !cleaned.is_empty() {
            return cleaned;
        }
    }
    if let Some(name) = first_tool_name(body) {
        return name;
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
    String::from("…")
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
/// half-built block past it). Committing only up to this point means the
/// finalized fold + chip + markdown pass never sees a partially-written
/// structure — a later regex reshape can't duplicate a half-written chip, no
/// tool header is ever orphaned, and a half-typed `$$…$$` / fenced code block /
/// GFM table never flashes wrong then snaps. PURE.
///
/// Unsafe (keep volatile) starting at the EARLIEST of:
///   * a `🛠️ ` compact tool header whose result line is still being written (the
///     header line has no trailing `\n` yet — so the name/args could still grow),
///   * a `<summary>` / `<thinking>` with no closing tag,
///   * a `**` (start of a possibly-incomplete `**Turn N …**` marker) with no
///     closing `**` yet,
///   * an unclosed inline/block `$…$` / `$$…$$` (odd count of unescaped `$` after
///     the last commit boundary — the open formula could still grow),
///   * an open fenced code block (a ```` ``` ````/`~~~` fence line-start with no
///     matching closing fence yet),
///   * a streaming GFM table (a header line followed by a delimiter line, held
///     from the table start until a blank line terminates it — every new row
///     reflows the column widths, so a row committed early paints a stale width).
/// We hold from the EARLIEST unsafe region so the commit cut protects ALL
/// simultaneous in-flight structures (cutting before the latest could still
/// commit an earlier open block). Falls back to the last paragraph boundary
/// (`\n\n`), else 0 (commit nothing until a boundary exists).
pub fn safe_commit_pos(stream: &str) -> usize {
    use crate::render::chip::TOOL_MARK;
    let mut unsafe_from: Option<usize> = None;
    fn note(pos: usize, current: &mut Option<usize>) {
        *current = Some(current.map_or(pos, |c| c.min(pos)));
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

    // Unclosed inline/block math: an odd number of unescaped `$` leaves the last
    // one opening a formula that could still grow. Hold from that opener.
    if let Some(p) = open_math_pos(stream) {
        note(p, &mut unsafe_from);
    }

    // An open fenced code block (```` ``` ````/`~~~`): its closing fence hasn't
    // arrived, so the language/contents are still in flight. Hold from the opener.
    if let Some(p) = open_fence_pos(stream) {
        note(p, &mut unsafe_from);
    }

    // A streaming GFM table: a header line followed by a `---|:--:` delimiter line
    // begins a table whose column widths reflow with every new row. Hold from the
    // table's first line until a blank line terminates it.
    if let Some(p) = streaming_table_start(stream) {
        note(p, &mut unsafe_from);
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

/// Byte offset of an OPEN (unclosed) inline/block math run, or `None` if math is
/// balanced. A `$` escaped by a backslash (`\$`) is a literal dollar, not a math
/// delimiter, and a `$` inside a CLOSED fenced code block is literal too (so it
/// can't toggle math parity). When the count of remaining (unescaped, non-fenced)
/// `$` is odd, the LAST one opened a formula that is still growing — return its
/// position. `$$` block delimiters are two `$`, so they stay parity-neutral when
/// balanced and leave an odd tail when half-open, which is exactly what we want.
/// PURE.
fn open_math_pos(stream: &str) -> Option<usize> {
    let bytes = stream.as_bytes();
    let mut fence: Option<FenceMark> = None;
    let mut last_open: Option<usize> = None;
    let mut open = false;
    let mut i = 0usize;
    while i < bytes.len() {
        let at_line_start = i == 0 || bytes[i - 1] == b'\n';
        if at_line_start {
            // A fence line-start toggles fenced-code state; `$` inside a fence is
            // literal, so skip the whole fenced span for math purposes.
            if let Some(mark) = fence_at(stream, i) {
                fence = match fence {
                    Some(open_mark) if mark.closes(&open_mark) => None,
                    Some(open_mark) => Some(open_mark),
                    None => Some(mark),
                };
                i = line_end(stream, i);
                continue;
            }
        }
        if fence.is_some() {
            i += 1;
            continue;
        }
        if bytes[i] == b'$' {
            let escaped = i > 0 && bytes[i - 1] == b'\\';
            if !escaped {
                if open {
                    open = false;
                } else {
                    open = true;
                    last_open = Some(i);
                }
            }
        }
        i += 1;
    }
    if open {
        last_open
    } else {
        None
    }
}

/// Byte offset of an OPEN fenced code block (a ```` ``` ```` or `~~~` fence with
/// no matching closer yet), or `None` if every fence is closed. A closing fence
/// must use the SAME char and be at least as long as the opener (CommonMark);
/// otherwise it's body content. PURE.
fn open_fence_pos(stream: &str) -> Option<usize> {
    let bytes = stream.as_bytes();
    let mut open: Option<FenceMark> = None;
    let mut i = 0usize;
    while i < bytes.len() {
        let at_line_start = i == 0 || bytes[i - 1] == b'\n';
        if at_line_start {
            if let Some(mark) = fence_at(stream, i) {
                open = match open {
                    Some(o) if mark.closes(&o) => None,
                    other @ Some(_) => other,
                    None => Some(mark),
                };
            }
        }
        i += 1;
    }
    open.map(|m| m.start)
}

/// A located fenced-code fence line (``` ``` ```/`~~~`, 3+ of the char, indented
/// ≤ 3 spaces per CommonMark).
#[derive(Clone, Copy)]
struct FenceMark {
    /// Byte offset of the fence line start (the indentation, if any).
    start: usize,
    /// The fence character (`` ` `` or `~`).
    ch: u8,
    /// The run length of the fence character.
    len: usize,
}

impl FenceMark {
    /// A fence `self` closes an open fence `opener` when it uses the same char and
    /// is at least as long (CommonMark).
    fn closes(&self, opener: &FenceMark) -> bool {
        self.ch == opener.ch && self.len >= opener.len
    }
}

/// Parse a fenced-code fence at line offset `at` (which must be a line start).
/// Returns the fence if the line (after ≤ 3 spaces of indent) opens with 3+ of the
/// same fence char; otherwise `None`. PURE.
fn fence_at(s: &str, at: usize) -> Option<FenceMark> {
    let bytes = s.as_bytes();
    let mut i = at;
    let mut indent = 0usize;
    while i < bytes.len() && bytes[i] == b' ' && indent < 4 {
        indent += 1;
        i += 1;
    }
    if indent >= 4 {
        return None;
    }
    let ch = bytes.get(i).copied()?;
    if ch != b'`' && ch != b'~' {
        return None;
    }
    let mut len = 0usize;
    while i < bytes.len() && bytes[i] == ch {
        len += 1;
        i += 1;
    }
    if len < 3 {
        return None;
    }
    Some(FenceMark { start: at, ch, len })
}

/// Byte offset where the CURRENT (still-open) streaming GFM table begins, or
/// `None` if no table is in flight. A table is "Confirmed" once a header line is
/// immediately followed by a delimiter line (`---`/`:--:`/`---:`, pipe-separated);
/// it then stays held from the header's start UNTIL a blank line terminates it
/// (every new row reflows every column width, so a row committed early paints a
/// stale-width row into scrollback). One-line lookbehind, fence-aware. A direct
/// port of Codex `streaming/table_holdback.rs` `TableHoldbackScanner`. PURE.
fn streaming_table_start(stream: &str) -> Option<usize> {
    let mut fence: Option<FenceMark> = None;
    let mut prev: Option<(usize, &str)> = None; // (line_start, line) one-line lookbehind
    let mut table_start: Option<usize> = None;
    for (start, text) in split_lines_with_offsets(stream) {
        // Fence lines toggle fenced state; a table delimiter inside a fence is
        // literal, so reset the lookbehind and skip while fenced.
        if let Some(mark) = fence_at(stream, start) {
            fence = match fence {
                Some(o) if mark.closes(&o) => None,
                other @ Some(_) => other,
                None => Some(mark),
            };
            prev = None;
            continue;
        }
        if fence.is_some() {
            prev = None;
            continue;
        }
        if text.trim().is_empty() {
            // A blank line terminates any open table; the closed table is now
            // structurally stable and may commit.
            table_start = None;
            prev = None;
            continue;
        }
        if table_start.is_none() {
            if let Some((pstart, ptext)) = prev {
                if is_table_delimiter_line(text) && is_table_header_line(ptext) {
                    table_start = Some(pstart);
                }
            }
        }
        prev = Some((start, text));
    }
    table_start
}

/// True if `line` could be a GFM table header row: non-blank and containing a `|`
/// (the column separator). PURE.
fn is_table_header_line(line: &str) -> bool {
    let t = line.trim();
    !t.is_empty() && t.contains('|')
}

/// True if `line` is a GFM table delimiter row: after stripping optional outer
/// pipes, every `|`-separated cell is a run of `-` with optional leading/trailing
/// `:` (alignment) and surrounding spaces, and there is at least one cell. PURE.
fn is_table_delimiter_line(line: &str) -> bool {
    let t = line.trim();
    if t.is_empty() {
        return false;
    }
    let inner = t.strip_prefix('|').unwrap_or(t);
    let inner = inner.strip_suffix('|').unwrap_or(inner);
    let mut cells = 0usize;
    for cell in inner.split('|') {
        let c = cell.trim();
        let c = c.strip_prefix(':').unwrap_or(c);
        let c = c.strip_suffix(':').unwrap_or(c);
        if c.is_empty() || !c.bytes().all(|b| b == b'-') {
            return false;
        }
        cells += 1;
    }
    cells >= 1
}

/// Byte offset of the end of the line starting at `at` (the position of the next
/// `\n`, or the string length). PURE.
fn line_end(s: &str, at: usize) -> usize {
    s[at..].find('\n').map(|i| at + i).unwrap_or(s.len())
}

/// Iterate `(line_start_byte, line_text)` over `s` split on `\n` (the newline is
/// NOT included in the text; a trailing empty segment after a final `\n` is
/// yielded so a terminating blank line is observed). PURE.
fn split_lines_with_offsets(s: &str) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let bytes = s.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'\n' {
            out.push((start, &s[start..i]));
            start = i + 1;
        }
    }
    out.push((start, &s[start..]));
    out
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
        assert!(segs.iter().any(|s| matches!(s, FoldSegment::Text { body, .. } if body.contains("still working on it"))));

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
        // No summary → the bare tool name (NEVER a "Turn N · …" literal).
        let body = "Turn 3 ...\n🛠️ web_search(query: rust)";
        let t = turn_title(body);
        assert_eq!(t, "web_search");
        assert!(!t.contains("Turn"), "tool fallback must not bake a Turn label: {t:?}");
        // No summary, no tool → first prose line.
        let body2 = "Turn 4 ...\njust some prose here";
        assert_eq!(turn_title(body2), "just some prose here");
        // No summary, no tool, no prose → a neutral ellipsis, NOT "Turn N".
        let body3 = "Turn 5 ...\n";
        let t3 = turn_title(body3);
        assert_eq!(t3, "…");
        assert!(!t3.contains("Turn"), "generic fallback must not be a Turn label: {t3:?}");
    }

    /// GATE: a FOLDED turn that has NO `<summary>` must NOT render a "Turn N" fold
    /// header (Fix F). Build a 2-turn fixture where turn 1 (the completed, folded one)
    /// has no summary — only a tool — and assert its fold header is the tool name with
    /// no "Turn" substring anywhere; and a no-summary/no-tool/no-prose turn folds to a
    /// neutral `▸ …` header, still free of "Turn".
    #[test]
    fn folded_no_summary_turn_header_has_no_turn_word() {
        // Turn 1 folds (tool, no summary); turn 2 is the expanded last turn.
        let text = "Turn 1 ...\n🛠️ web_search(query: rust)\n[Info] ok\nTurn 2 ...\nworking";
        let segs = fold_turns(text, false);
        let header = segs
            .iter()
            .find_map(|s| s.header())
            .expect("a folded turn yields a `▸` header");
        assert_eq!(header, "▸ web_search");
        assert!(!header.contains("Turn"), "fold header must not contain 'Turn': {header:?}");

        // No summary, no tool, no prose → fold_all forces a fold; header is `▸ …`.
        let bare = fold_turns("Turn 7 ...\n", true);
        for s in &bare {
            if let Some(h) = s.header() {
                assert!(!h.contains("Turn"), "fold header must not contain 'Turn': {h:?}");
                assert_eq!(h, "▸ …", "no-content fold header is a neutral ellipsis");
            }
        }
    }

    #[test]
    fn find_turn_markers_parses_both_forms_and_numbers() {
        // Mixed: a BARE `Turn 1 ...` and a BOLD `**LLM Running (Turn 2) ...**`.
        let text = "pre\nTurn 1 ...\nbody\n**LLM Running (Turn 2) ...**\nmore";
        let segs = fold_turns(text, false);
        // The preamble "pre" is its own text segment.
        assert!(matches!(&segs[0], FoldSegment::Text { body, .. } if body.contains("pre")));
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

    /// NEW (8c): a half-typed inline/block formula is held back — the committed
    /// head NEVER contains an unclosed `$…$` / `$$…$$`. A balanced formula commits;
    /// an escaped `\$` is a literal dollar and never opens math.
    #[test]
    fn safe_commit_pos_holds_unclosed_math() {
        // An open block `$$` (half-typed `\frac`) after a stable paragraph: hold the
        // formula, commit only the prose before it.
        let s = "intro prose.\n\n$$\\frac{a}{";
        let pos = safe_commit_pos(s);
        let head = &s[..pos];
        assert!(head.contains("intro prose."));
        assert!(!head.contains("$$"), "the half-built block math is held back: {head:?}");

        // An open INLINE `$…` (odd single `$`) after a paragraph is held too.
        let s_inline = "see this:\n\nthe cost is $5 and $x = ";
        let pos_i = safe_commit_pos(s_inline);
        assert!(!s_inline[..pos_i].contains("$x ="), "open inline math held: head={:?}", &s_inline[..pos_i]);

        // A BALANCED inline formula in a closed paragraph commits (even count of
        // `$` → no open math → the `\n\n` fallback governs).
        let s_balanced = "alpha is $\\alpha$ exactly.\n\nnext paragraph typing";
        let pos_b = safe_commit_pos(s_balanced);
        assert!(s_balanced[..pos_b].contains("$\\alpha$"), "balanced inline math commits");

        // A BALANCED block `$$…$$` standing as its own paragraph commits.
        let s_block = "$$E = mc^2$$\n\nand then more text being typed";
        let pos_bb = safe_commit_pos(s_block);
        assert!(s_block[..pos_bb].contains("$$E = mc^2$$"), "balanced block math commits");

        // An ESCAPED `\$` (literal dollar) does NOT open math — a single `\$5` in a
        // closed paragraph commits whole.
        let s_esc = "it costs \\$5 total.";
        assert_eq!(safe_commit_pos(s_esc), s_esc.len(), "escaped dollar is not math");
    }

    /// NEW (8c): an open fenced code block (```` ``` ````/`~~~`) is held back until
    /// its closing fence arrives; a closed fence commits.
    #[test]
    fn safe_commit_pos_holds_open_fence() {
        // Open ``` fence after a stable paragraph: the fence + its body are held.
        let s = "here is code:\n\n```rust\nfn main() {";
        let pos = safe_commit_pos(s);
        let head = &s[..pos];
        assert!(head.contains("here is code:"));
        assert!(!head.contains("```"), "the open code fence is held back: {head:?}");

        // A CLOSED fence (matching ```), followed by a stable paragraph break,
        // commits the whole fenced block.
        let s_closed = "```rust\nfn main() {}\n```\n\nafter the code now typing";
        let pos_c = safe_commit_pos(s_closed);
        let head_c = &s_closed[..pos_c];
        assert!(head_c.contains("fn main() {}"), "closed fence commits its body");
        assert!(head_c.matches("```").count() == 2, "both fences committed: {head_c:?}");

        // A `$` INSIDE an open fence is literal and must NOT be mistaken for open
        // math (the fence holdback governs, not a phantom math parity).
        let s_dollar_fence = "text.\n\n```\nprice = $5\n";
        let pos_df = safe_commit_pos(s_dollar_fence);
        assert!(!s_dollar_fence[..pos_df].contains("```"), "open fence with $ inside is held");

        // A `~~~` tilde fence is recognized too.
        let s_tilde = "note:\n\n~~~\nstill open";
        let pos_t = safe_commit_pos(s_tilde);
        assert!(!s_tilde[..pos_t].contains("~~~"), "open tilde fence is held back");
    }

    /// NEW (8c): a streaming GFM table (header line + delimiter line, no terminating
    /// blank line yet) is held from the table start; once a blank line ends it the
    /// table commits. Port of Codex `TableHoldbackScanner`.
    #[test]
    fn safe_commit_pos_holds_streaming_table() {
        // Header + delimiter + one row, still streaming (no blank line after): hold
        // from the table start; commit only the prose before it.
        let s = "summary:\n\n| Name | Score |\n|------|------:|\n| Alice | 90 |";
        let pos = safe_commit_pos(s);
        let head = &s[..pos];
        assert!(head.contains("summary:"));
        assert!(!head.contains("| Name |"), "the streaming table header is held back: {head:?}");
        assert!(!head.contains("Alice"), "in-flight table rows are held back");

        // Just the header (no delimiter seen yet) is NOT yet a confirmed table — but
        // the final still-growing line stays volatile via the paragraph fallback.
        let s_hdr = "intro.\n\n| Name | Score |";
        let pos_h = safe_commit_pos(s_hdr);
        assert!(s_hdr[..pos_h].contains("intro."));

        // A COMPLETED table terminated by a blank line commits in full.
        let s_done = "| Name | Score |\n|------|------:|\n| Alice | 90 |\n\nnext line typing";
        let pos_d = safe_commit_pos(s_done);
        let head_d = &s_done[..pos_d];
        assert!(head_d.contains("Alice"), "a blank-line-terminated table commits");
        assert!(head_d.contains("| Name | Score |"), "table header committed: {head_d:?}");
        assert!(!head_d.contains("next line typing"), "post-table tail stays volatile");

        // A `|`-bearing delimiter-looking line inside a fenced block is NOT a table.
        let s_fenced = "x.\n\n```\n| a | b |\n|---|---|\n";
        let pos_f = safe_commit_pos(s_fenced);
        // The OPEN fence governs (held); regardless, no table confirmation leaks past it.
        assert!(!s_fenced[..pos_f].contains("| a | b |"), "fenced pipe rows are not a table");
    }
}
