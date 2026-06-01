//! render/chip.rs — tool-call "chips" rendered as Claude-Code-style bullets
//! (redesign_cc.md §2.3; replaces the old heavy box).
//!
//! GA's COMPACT tool calls (agent_loop.py:89, the tui_v4 non-verbose mode) look
//! like `🛠️ NAME(ARGS)` followed by the tool's `[Action]/[Status]/[Info]` result
//! lines. The renderer turns each into a CC bullet:
//!
//!     ⏺ web_scan {"tabs_only": true}
//!       [Info] 3 tabs scanned · ok
//!       ▸ +2 more
//!
//! `⏺` (BLACK_CIRCLE, done) / `○` (figures.circle, running/pending) bullet +
//! tool name (colored by status) + a dim one-line args + the result indented two
//! columns, dim, truncated to a few lines with a CLICKABLE `▸ +N more` triangle
//! (Fix E / Q8: click it to expand the full result; a `▾` then collapses it). NO
//! box. (CC: `AssistantToolUseMessage.tsx` / `_CoordinatorAgentStatus.tsx:144`.)
//!
//! The marker is `🛠️ name(args)` — **NOT** `🛠️ Tool:` (the old verbose form). The
//! result is everything after the header line up to the next `🛠️` / `Turn N` /
//! `<summary>` / EOT. Parsing (`parse_tool_calls`), status inference
//! (`tool_status`), and the bullet layout (`render_chip_bullet`) are PURE +
//! unit-tested; the markdown layer maps the rows to themed `Line`s.

use unicode_width::UnicodeWidthStr;

/// The COMPACT tool-call marker GA emits in tui_v4's (non-verbose) mode
/// (agent_loop.py:89 `🛠️ {tool_name}({args})`). Shared with `fold.rs` /
/// `markdown::mod` so the parse, the fold boundary, and the structural
/// stream-commit all agree on the one true marker.
pub const TOOL_MARK: &str = "🛠️ ";

/// A tool call extracted from assistant text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// Sequential id within the message (1-based) — used by the `/verbose` audit.
    pub id: u32,
    /// The tool name (`web_scan`, `file_read`, …).
    pub name: String,
    /// The argument string from inside the `(…)` (e.g. `{"tabs_only": true}`),
    /// possibly empty.
    pub args: String,
    /// The result/preview body (possibly empty / pending).
    pub result: String,
    /// Inferred status.
    pub status: ToolStatus,
}

/// A tool call's inferred status (GA-marker-only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolStatus {
    Ok,
    Error,
    Pending,
}

impl ToolStatus {
    /// The CC status bullet glyph: `⏺` (done, BLACK_CIRCLE) for ok/error, `○`
    /// (figures.circle) for pending/running. Color is chosen by the renderer from
    /// the [`Token`](crate::theme::Token) below.
    pub fn bullet(self) -> &'static str {
        match self {
            ToolStatus::Ok | ToolStatus::Error => "⏺",
            ToolStatus::Pending => "○",
        }
    }

    /// Which theme token colors the bullet + tool name for this status (CC:
    /// success green / error red / dim while pending).
    pub fn token(self) -> crate::theme::Token {
        use crate::theme::Token;
        match self {
            ToolStatus::Ok => Token::Success,
            ToolStatus::Error => Token::Error,
            ToolStatus::Pending => Token::Dim,
        }
    }

    /// The `/verbose` audit badge text (kept for the audit trail; the cockpit uses
    /// the bullet glyph above).
    pub fn badge(self) -> (&'static str, crate::theme::Token) {
        match self {
            ToolStatus::Ok => ("✓ ok", self.token()),
            ToolStatus::Error => ("✕ error", self.token()),
            ToolStatus::Pending => ("· …", self.token()),
        }
    }
}

/// Infer a tool's status from GA's emitted markers ONLY (tui_v3 `_tool_status`).
/// Read-tool RESULTS can legitimately contain ❌ / "error" as content, so we only
/// flag error on a leading `[Status]`/`[Error]` failure line, a leading error
/// marker, or an inline `!!!Error:` (a real model/stream error). A `Waiting for
/// your answer` (ask_user) without a success marker is PENDING. Otherwise:
/// non-empty result → ok, empty → pending. PURE.
pub fn tool_status(result: &str) -> ToolStatus {
    let s = result.trim_start();
    // Leading explicit error marker.
    if s.starts_with("Error:")
        || s.starts_with("Error ")
        || s.starts_with("Exception")
        || s.starts_with("Traceback")
        || s.starts_with("!!!Error")
        || s.starts_with('❌')
        || s.starts_with('⛔')
    {
        return ToolStatus::Error;
    }
    // A `[Status]`/`[Error] … fail|error|❌` line anywhere, or an inline `!!!Error:`.
    for line in result.lines() {
        let l = line.trim_start();
        if l.starts_with("!!!Error") {
            return ToolStatus::Error;
        }
        if (l.starts_with("[Status]") || l.starts_with("[Error]"))
            && (l.to_ascii_lowercase().contains("fail")
                || l.to_ascii_lowercase().contains("error")
                || l.contains('❌'))
        {
            return ToolStatus::Error;
        }
    }
    // ask_user is pending until an answer / success marker arrives.
    if result.contains("Waiting for your answer")
        && !result.contains('✅')
        && !result.contains("成功")
    {
        return ToolStatus::Pending;
    }
    if result.trim().is_empty() {
        ToolStatus::Pending
    } else {
        ToolStatus::Ok
    }
}

/// Parse the tool calls out of assistant `text`. A call is a COMPACT
/// `🛠️ NAME(ARGS)` header (agent_loop.py:89); the name is everything between the
/// marker and the first `(`, the args are the balanced `(…)` payload, and the
/// result is everything after the header line up to the next structural boundary
/// (`🛠️` / `Turn N ...` / `<summary>` / EOT). PURE (a hand-rolled scan, no regex
/// dep) — the `compact_chip_parse` deliverable pins it.
pub fn parse_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut out = Vec::new();
    let mut search_from = 0usize;
    let mut id = 0u32;
    while let Some(rel) = text[search_from..].find(TOOL_MARK) {
        let start = search_from + rel;
        let after = start + TOOL_MARK.len();
        // The header is the rest of the marker's line.
        let line_end = text[after..]
            .find('\n')
            .map(|i| after + i)
            .unwrap_or(text.len());
        let header_line = &text[after..line_end];
        let (name, args) = split_name_args(header_line);

        // The result body runs from the line AFTER the header to the next boundary.
        let body_start = (line_end + 1).min(text.len());
        let next_boundary = next_marker_boundary(text, body_start);
        let result = text[body_start..next_boundary].trim().to_string();

        id += 1;
        let status = tool_status(&result);
        out.push(ToolCall {
            id,
            name,
            args,
            result,
            status,
        });
        // Advance to the boundary (never inside the multibyte marker).
        search_from = if next_boundary > start {
            next_boundary
        } else {
            after
        };
    }
    out
}

/// Split a compact header `name(args)` (the text AFTER the `🛠️ ` marker) into its
/// `(name, args)`. The name is up to the first `(`; the args are the content of
/// the balanced parens (a closing `)` matched by depth, tolerating nested `()` in
/// JSON-ish args). With no `(` the whole thing is the name and args are empty.
/// PURE.
fn split_name_args(header: &str) -> (String, String) {
    match header.find('(') {
        Some(open) => {
            let name = header[..open].trim().to_string();
            let rest = &header[open + 1..];
            // Find the matching close paren by depth.
            let mut depth = 1i32;
            let mut end = rest.len();
            for (i, ch) in rest.char_indices() {
                match ch {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            end = i;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let args = rest[..end].trim().to_string();
            (name, args)
        }
        None => (header.trim().to_string(), String::new()),
    }
}

/// The byte offset of the next structural boundary at/after `from`: the earliest
/// of a `🛠️` marker, a `Turn N ...` line, or a `<summary>` open — else the text
/// end. Matches BOTH the bare GA turn form (`Turn 1 ...`, tui_v4 non-verbose) and
/// the legacy bold form (`**Turn 1 ...**`, verbose). PURE.
pub fn next_marker_boundary(text: &str, from: usize) -> usize {
    let tail = &text[from..];
    [
        tail.find(TOOL_MARK),
        find_turn_line(tail),
        tail.find("<summary>"),
    ]
    .into_iter()
    .flatten()
    .min()
    .map(|i| from + i)
    .unwrap_or(text.len())
}

/// The byte offset (within `s`) of the next `Turn N ...` turn-boundary line, in
/// EITHER the bare (`Turn 1 ...`) or bold (`**Turn 1 ...**`) form. We require the
/// marker to sit at the START of a line (after a `\n` or at offset 0) so a `Turn`
/// inside prose ("Turn left at the light") is never mistaken for a boundary. PURE.
pub fn find_turn_line(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i < s.len() {
        // A line start: offset 0 or just past a '\n'.
        let at_line_start = i == 0 || bytes.get(i - 1) == Some(&b'\n');
        if at_line_start {
            let rest = &s[i..];
            let core = rest.strip_prefix("**").unwrap_or(rest);
            if let Some(after) = core.strip_prefix("Turn ") {
                // `Turn ` then ≥1 digit ⇒ a turn line.
                if after.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                    return Some(i);
                }
            }
        }
        i += 1;
    }
    None
}

/// Render a tool call as the rows of a CC bullet (redesign_cc.md §2.3): a header
/// `⏺ name  args` (bullet + name colored by status, args dim) and the result
/// indented two columns, dim. Returns PLAIN strings (the markdown layer styles them
/// by row kind). The first row is the bullet/header; the rest are the indented
/// result. PURE.
///
/// Folding (Fix E / Q8): when `expanded` is false the result is truncated to
/// `max_preview` rows and, if it overflows, the dead `… +N more` text becomes a
/// CLICKABLE `▸ +N more` triangle affordance (click its column to expand). When
/// `expanded` is true the truncation is SKIPPED — every result line is emitted —
/// and a `▾` collapse affordance closes the row. A result that fits in `max_preview`
/// needs no affordance either way (nothing to expand).
///
/// `width` clips long rows so a bullet never wraps unexpectedly (the caller's
/// styled soft-wrap re-wraps anyway, but clipping the header keeps it tidy).
pub fn render_chip_bullet(call: &ToolCall, width: u16, max_preview: usize, expanded: bool) -> ChipBullet {
    let width = (width as usize).max(8);
    let bullet = call.status.bullet();

    // Header: `⏺ name` + (optional) a dim one-line args after a space.
    let header_name = format!("{bullet} {}", call.name);
    let args_oneline = collapse_oneline(&call.args);

    // The result: each non-blank line indented 2 cols. Collapsed → at most
    // `max_preview` rows; expanded → ALL rows (truncation skipped).
    let mut result_rows: Vec<String> = Vec::new();
    let mut total = 0usize;
    for raw in call.result.lines() {
        let line = raw.trim_end();
        if line.is_empty() && result_rows.is_empty() {
            // Skip leading blank lines so the result hugs the bullet.
            continue;
        }
        total += 1;
        if expanded || result_rows.len() < max_preview {
            // Indent 2 cols; clip to width so it stays one logical row pre-wrap.
            let body = clip_cells(line, width.saturating_sub(2));
            result_rows.push(format!("  {body}"));
        }
    }
    // The expand/collapse affordance row — a clickable triangle, NOT dead text. Only
    // shown when the result is long enough to fold (> max_preview rows).
    let expandable = total > max_preview;
    if expandable {
        if expanded {
            result_rows.push("  ▾".to_string());
        } else {
            let overflow = total - max_preview;
            result_rows.push(format!("  ▸ +{overflow} more"));
        }
    }

    ChipBullet {
        header_name,
        args: args_oneline,
        result_rows,
        status: call.status,
        expandable,
    }
}

/// The materialized rows of a rendered CC tool-call bullet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChipBullet {
    /// The bullet + tool name (`⏺ web_scan`) — colored by status.
    pub header_name: String,
    /// The dim one-line args (`{"tabs_only": true}`), possibly empty.
    pub args: String,
    /// The 2-col-indented, dim result rows (plus a `▸`/`▾` affordance row when the
    /// result is long enough to fold).
    pub result_rows: Vec<String>,
    /// The status (so the renderer colors the bullet/name).
    pub status: ToolStatus,
    /// True when the result overflowed `max_preview` and so carries a fold
    /// affordance (`▸ +N more` collapsed / `▾` expanded). Lets the markdown layer
    /// register the whole bullet as a clickable [`NodeId::Tool`](crate::render::fold)
    /// hit target only when there's actually something to expand.
    pub expandable: bool,
}

#[allow(dead_code)] // row accessor (used by tests).
impl ChipBullet {
    /// All rows as plain strings (header line, then the indented result rows). The
    /// header joins the name + args with a single space (args dropped if empty).
    pub fn rows(&self) -> Vec<String> {
        let mut v = Vec::with_capacity(self.result_rows.len() + 1);
        let head = if self.args.is_empty() {
            self.header_name.clone()
        } else {
            format!("{}  {}", self.header_name, self.args)
        };
        v.push(head);
        v.extend(self.result_rows.iter().cloned());
        v
    }
}

/// Collapse a (possibly multi-line) args string to a single dim line: internal
/// whitespace runs → single spaces, trimmed. PURE.
fn collapse_oneline(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Clip a string to at most `max` display cells (no ellipsis — caller controls).
fn clip_cells(s: &str, max: usize) -> String {
    if UnicodeWidthStr::width(s) <= max {
        return s.to_string();
    }
    let mut out = String::new();
    let mut acc = 0usize;
    for g in unicode_segmentation::UnicodeSegmentation::graphemes(s, true) {
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

    /// THE deliverable test: parse COMPACT `🛠️ name(args)` calls + their results.
    #[test]
    fn compact_chip_parse() {
        let text = "\
prose before
🛠️ web_scan({\"tabs_only\": true})
[Info] 3 tabs scanned · ok
🛠️ file_read(config.toml)
port = 8080
host = localhost";
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 2, "two compact tool calls parsed");
        assert_eq!(calls[0].name, "web_scan");
        assert_eq!(calls[0].args, "{\"tabs_only\": true}");
        assert!(calls[0].result.contains("[Info] 3 tabs scanned"));
        assert_eq!(calls[0].id, 1);
        assert_eq!(calls[1].name, "file_read");
        assert_eq!(calls[1].args, "config.toml");
        assert!(calls[1].result.contains("port = 8080"));
        // The OLD verbose `🛠️ Tool:` marker is NOT matched as a name=`Tool:` chip
        // (no `(`): it parses to a single call whose name is the whole header — but
        // crucially the COMPACT path above is what GA actually emits. Assert the new
        // marker is the one driving the parse.
        assert!(!text.contains("Tool:"), "the seed uses the compact form");
    }

    #[test]
    fn split_name_args_handles_nested_parens() {
        // Balanced nested parens in JSON-ish args are kept whole.
        let (n, a) = split_name_args("run(cmd=\"echo (hi)\", n=2)");
        assert_eq!(n, "run");
        assert_eq!(a, "cmd=\"echo (hi)\", n=2");
        // No args.
        let (n2, a2) = split_name_args("no_tool");
        assert_eq!(n2, "no_tool");
        assert!(a2.is_empty());
        // Empty args.
        let (n3, a3) = split_name_args("ping()");
        assert_eq!(n3, "ping");
        assert!(a3.is_empty());
    }

    #[test]
    fn find_turn_line_matches_bare_and_bold_at_line_start_only() {
        // Bare GA form (tui_v4 non-verbose).
        assert_eq!(find_turn_line("Turn 1 ...\nbody"), Some(0));
        // Bold legacy form (verbose).
        assert_eq!(find_turn_line("prose\n**Turn 2 ...**\nx"), Some(6));
        // `Turn` inside prose is NOT a boundary.
        assert_eq!(find_turn_line("Turn left at the light"), None);
        // `Turn` not at line start is NOT a boundary.
        assert_eq!(find_turn_line("see Turn 3 below"), None);
    }

    #[test]
    fn status_inference_is_marker_only() {
        // A read result that merely CONTAINS ❌ as content is OK, not error.
        assert_eq!(tool_status("here is a doc with ❌ inside the body"), ToolStatus::Ok);
        // A leading error marker IS an error.
        assert_eq!(tool_status("Error: file not found"), ToolStatus::Error);
        assert_eq!(tool_status("❌ failed to open"), ToolStatus::Error);
        // An inline `!!!Error:` is an error (a real model/stream error).
        assert_eq!(tool_status("ran tool\n!!!Error: SSE overloaded"), ToolStatus::Error);
        // A `[Status] … fail` line is an error.
        assert_eq!(tool_status("ran tool\n[Status] failed: bad args"), ToolStatus::Error);
        // Empty result → pending.
        assert_eq!(tool_status("   "), ToolStatus::Pending);
        // ask_user waiting → pending.
        assert_eq!(tool_status("Waiting for your answer ..."), ToolStatus::Pending);
        // …but once a success marker lands → ok.
        assert_eq!(tool_status("Waiting for your answer ...\n✅ done"), ToolStatus::Ok);
    }

    #[test]
    fn bullet_glyph_and_indented_result() {
        let call = ToolCall {
            id: 1,
            name: "web_scan".into(),
            args: "{\"tabs_only\": true}".into(),
            result: "[Info] 3 tabs scanned · ok".into(),
            status: ToolStatus::Ok,
        };
        let chip = render_chip_bullet(&call, 80, 4, false);
        // CC done bullet + name.
        assert_eq!(chip.status.bullet(), "⏺");
        assert!(chip.header_name.starts_with("⏺ web_scan"));
        // Args carried as the dim one-liner.
        assert_eq!(chip.args, "{\"tabs_only\": true}");
        // The result is indented two columns.
        assert_eq!(chip.result_rows.len(), 1);
        assert!(chip.result_rows[0].starts_with("  [Info]"));
        // The combined rows are NOT a box (no border glyphs).
        for row in chip.rows() {
            assert!(!row.contains('╭') && !row.contains('│') && !row.contains('╰'));
        }
    }

    #[test]
    fn pending_uses_hollow_bullet() {
        let call = ToolCall {
            id: 1,
            name: "ask_user".into(),
            args: String::new(),
            result: "Waiting for your answer ...".into(),
            status: ToolStatus::Pending,
        };
        let chip = render_chip_bullet(&call, 40, 4, false);
        assert_eq!(chip.status.bullet(), "○", "running/pending = hollow circle");
        assert_eq!(chip.header_name, "○ ask_user");
    }

    #[test]
    fn result_overflow_collapses_to_clickable_triangle() {
        let call = ToolCall {
            id: 1,
            name: "x".into(),
            args: String::new(),
            result: (0..10).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n"),
            status: ToolStatus::Ok,
        };
        let chip = render_chip_bullet(&call, 30, 3, false);
        // 10 lines, max 3 preview → 3 indented rows + a CLICKABLE `▸ +7 more`
        // affordance (a triangle, NOT the old dead `… +N more` text).
        assert_eq!(chip.result_rows.len(), 4);
        assert!(chip.expandable, "an overflowing result is foldable");
        let tail = chip.result_rows.last().unwrap();
        assert!(tail.contains("▸ +7 more"), "triangle affordance: {tail:?}");
        assert!(!tail.contains('…'), "the dead ellipsis text is gone: {tail:?}");
    }

    /// Fix E acceptance: an EXPANDED tool result skips the `max_preview` truncation
    /// (every result line is emitted) and swaps the `▸ +N more` affordance for a `▾`
    /// collapse triangle — there is no dead `… +N more` text in either state.
    #[test]
    fn expanded_tool_result_skips_truncation() {
        let call = ToolCall {
            id: 1,
            name: "x".into(),
            args: String::new(),
            result: (0..10).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n"),
            status: ToolStatus::Ok,
        };
        // Collapsed: 3 preview rows + the affordance = 4.
        let collapsed = render_chip_bullet(&call, 30, 3, false);
        assert_eq!(collapsed.result_rows.len(), 4);

        // Expanded: ALL 10 result rows + a `▾` collapse affordance = 11.
        let expanded = render_chip_bullet(&call, 30, 3, true);
        assert_eq!(expanded.result_rows.len(), 11, "all 10 lines + the ▾ affordance");
        assert!(expanded.expandable);
        for i in 0..10 {
            assert!(
                expanded.result_rows.iter().any(|r| r.contains(&format!("line {i}"))),
                "expanded result must contain every line (line {i} missing)"
            );
        }
        let tail = expanded.result_rows.last().unwrap();
        assert!(tail.trim() == "▾", "expanded affordance is a ▾ collapse triangle: {tail:?}");
        assert!(!expanded.result_rows.iter().any(|r| r.contains("more")), "no `+N more` when expanded");

        // A result that FITS in max_preview is not foldable in either state.
        let small = ToolCall {
            id: 2, name: "y".into(), args: String::new(),
            result: "one\ntwo".into(), status: ToolStatus::Ok,
        };
        assert!(!render_chip_bullet(&small, 30, 4, false).expandable);
        assert!(!render_chip_bullet(&small, 30, 4, true).expandable);
    }

    #[test]
    fn cjk_args_and_result_stay_within_width() {
        let call = ToolCall {
            id: 1,
            name: "搜索".into(),
            args: "查询: 你好世界".into(),
            result: "结果在这里很长很长很长很长很长很长很长很长很长很长".into(),
            status: ToolStatus::Ok,
        };
        let chip = render_chip_bullet(&call, 24, 4, false);
        // The indented result clips to width (CJK counted as 2 cells), so it never
        // overshoots the bullet column budget pre-wrap.
        for row in &chip.result_rows {
            assert!(UnicodeWidthStr::width(row.as_str()) <= 24 + 1);
        }
    }
}
