//! render/chip.rs — tool-call "chips" rendered as a tui_v3-style BORDERED BOX
//! (R1 ITEM 2; replaces the flat CC bullet).
//!
//! GA's COMPACT tool calls (agent_loop.py:89, the tui_v4 non-verbose mode) look
//! like `🛠️ NAME(ARGS)` followed by the tool's `[Action]/[Status]/[Info]` result
//! lines. The renderer turns each into a fully-enclosed box (tui_v3 `_chip_box`):
//!
//!     ╭─ web_scan  ✓ ok  ·t1 ─────────────────────╮
//!     │ {"tabs_only": true}                        │
//!     │ 3 tabs scanned                             │
//!     ╰────────────────────────────────────────────╯
//!
//! The TOP border carries the tool name (BOLD), a colored status badge
//! (`✓ ok` / `✕ error` / `· …`), and the `·tN` turn-id (dim) ON the line; accent
//! corners/dashes frame it. Interior `│ … │` rows (accent border, dim body) hold
//! the arg-hint (what was called) FIRST, then the result preview (≤4 rows). When
//! the result overflows, a `… +N more` fold affordance row sits INSIDE the box
//! (Fix E / Q8: click it to expand; a `▾` then collapses). Border = accent
//! (`Token::Claude`), matching tui_v3's `_ACCENT` corners.
//!
//! The marker is `🛠️ name(args)` — **NOT** `🛠️ Tool:` (the old verbose form). The
//! result is everything after the header line up to the next `🛠️` / `Turn N` /
//! `<summary>` / EOT. Parsing (`parse_tool_calls`), status inference
//! (`tool_status`), and the box layout (`render_chip_box`) are PURE + unit-tested;
//! the markdown layer maps the rows to themed `Line`s.

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
    /// Which theme token colors the status badge for this status (success green /
    /// error red / dim while pending) — tui_v3 `_OK`/`_ERR`/`_DIM`.
    pub fn token(self) -> crate::theme::Token {
        use crate::theme::Token;
        match self {
            ToolStatus::Ok => Token::Success,
            ToolStatus::Error => Token::Error,
            ToolStatus::Pending => Token::Dim,
        }
    }

    /// The status badge text + its color token: `✓ ok` (green) / `✕ error` (red) /
    /// `· …` (dim pending) — tui_v3 `_chip_box`'s `sti`/`stcol`. Used BOTH on the
    /// box's top border and in the `/verbose` audit trail.
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

/// The accent corners/dashes/verticals; matched to tui_v3 `_chip_box` rounded glyphs.
const TL: char = '╭';
const TR: char = '╮';
const BL: char = '╰';
const BR: char = '╯';
const HBAR: char = '─';
const VBAR: char = '│';

/// Render a tool call as the ROWS of a tui_v3-style bordered box (R1 ITEM 2 /
/// `_chip_box`). Returns the materialized [`ChipBox`]: a TOP border line carrying
/// `╭─ name  badge  ·tN ─…─╮` (name BOLD, badge colored, `·tN` dim, accent frame),
/// the interior `│ … │` rows (the arg-hint FIRST, then the result preview ≤
/// `max_preview`, then a `… +N more` fold affordance when truncated), and a BOTTOM
/// `╰─…─╯` border line. Every row is sized to EXACTLY `inner` cells so each maps to
/// ONE visual row after the caller's soft-wrap (the styled draw and the plain
/// projection both flow these rows through the same wrap → row-count parity holds by
/// construction). The markdown layer styles the rows by kind. PURE.
///
/// Folding (Fix E / Q8): when `expanded` is false the result preview is truncated to
/// `max_preview` rows and an overflow yields a `… +N more` affordance row INSIDE the
/// box (click its row to expand). When `expanded` is true the truncation is SKIPPED —
/// every result line shows — and a `▾` collapse affordance closes the box. A result
/// that fits in `max_preview` needs no affordance either way.
pub fn render_chip_box(call: &ToolCall, width: u16, max_preview: usize, expanded: bool) -> ChipBox {
    // The box occupies `inner` cells (full available width, clamped to a usable
    // minimum so a narrow pane still draws a coherent — if cramped — frame).
    let inner = (width as usize).max(MIN_BOX_INNER);
    let (badge, _) = call.status.badge();
    let tag = format!("·t{}", call.id);

    let (top, top_name) = top_border(&call.name, badge, &tag, inner);

    // Interior content (each clipped/padded to `content_w` so `│ … │` is exactly
    // `inner` cells): the arg-hint (what was called) FIRST, then the result preview.
    let content_w = inner.saturating_sub(4).max(1);
    let mut body: Vec<String> = Vec::new();
    let hint = arg_hint(&call.name, &call.args);
    if !hint.is_empty() {
        body.extend(wrap_cells(&hint, content_w));
    }
    let (mut preview, overflow) = result_preview(&call.result, max_preview, content_w, expanded);
    body.append(&mut preview);

    // The expand/collapse affordance row, INSIDE the box (Fix E / S1 Fix C).
    // Shown only when the result overflows `max_preview`. Clearer affordance
    // text (S1): collapsed = "▸ +N more", expanded = "▾ collapse".
    let expandable = overflow > 0;
    if expandable {
        body.push(if expanded {
            "▾ collapse".to_string()
        } else {
            format!("▸ +{overflow} more")
        });
    }

    let interior: Vec<String> = body.iter().map(|c| interior_row(c, content_w)).collect();
    let bottom = border_line(BL, BR, inner);

    ChipBox {
        top,
        top_name,
        top_badge: badge.to_string(),
        top_tag: tag,
        interior,
        bottom,
        status: call.status,
        expandable,
    }
}

/// The smallest inner width the box renders at; below this a pane is barely usable,
/// but a coherent (cramped) frame still beats a wrapped, edge-broken one.
const MIN_BOX_INNER: usize = 8;

/// The materialized rows of a rendered tool-call BOX. The markdown layer paints the
/// `top`/`bottom` border lines (accent frame + bold name + colored badge + dim tag /
/// the `┊` interior pad) and each `interior` row (accent `│` + dim body + accent `│`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChipBox {
    /// The top border line `╭─ name  badge  ·tN ─…─╮` (already laid out, exactly
    /// `inner` cells) — the PLAIN projection. The markdown layer rebuilds it as
    /// styled spans from the pieces below, whose concatenation is byte-identical.
    pub top: String,
    /// The CLIPPED tool name as it appears on the top border (bold span).
    pub top_name: String,
    /// The status badge text on the top border (`✓ ok`/`✕ error`/`· …`; status color).
    pub top_badge: String,
    /// The `·tN` turn-id tag on the top border (dim span).
    pub top_tag: String,
    /// The interior `│ … │` rows (border + body + border), exactly `inner` cells each
    /// — the arg-hint, the result preview, and the fold affordance row when present.
    pub interior: Vec<String>,
    /// The bottom border line `╰─…─╯` (accent), exactly `inner` cells.
    pub bottom: String,
    /// The status (so the renderer colors the badge on the top border).
    pub status: ToolStatus,
    /// True when the result overflowed `max_preview` and so the box carries a fold
    /// affordance (`… +N more` collapsed / `▾` expanded). Lets the markdown layer
    /// register the whole box as a clickable [`NodeId::Tool`](crate::render::fold)
    /// hit target only when there's actually something to expand.
    pub expandable: bool,
}

#[allow(dead_code)] // row accessor (used by tests).
impl ChipBox {
    /// All rows as plain strings (top border, interior rows, bottom border) — the
    /// flat projection a test (or a plain-text consumer) reads.
    pub fn rows(&self) -> Vec<String> {
        let mut v = Vec::with_capacity(self.interior.len() + 2);
        v.push(self.top.clone());
        v.extend(self.interior.iter().cloned());
        v.push(self.bottom.clone());
        v
    }
}

/// Build the box TOP border laying the tool name + status badge + `·tN` tag ON the
/// line (tui_v3 `_chip_box` wide branch): `╭─` + ` name ` + `  ` + `badge` + `  ` +
/// `tag` + ` ` + `─`×fill + `╮`, padded to EXACTLY `inner` cells. The name is clipped
/// when the header would otherwise overflow the inner width. Returns `(line,
/// clipped_name)` so the styled renderer can re-segment without re-parsing. PURE.
fn top_border(name: &str, badge: &str, tag: &str, inner: usize) -> (String, String) {
    // Reserve: `╭─` (2) + a trailing `╮` (1) + the framed header text + ≥1 dash.
    // header text = ` {name}  {badge}  {tag} ` → its non-name width is fixed.
    let fixed = 2 /*spaces around name region*/ + 2 + cell_width(badge) + 2 + cell_width(tag);
    let name_max = inner.saturating_sub(3 + fixed).max(1);
    let name = clip_cells(name, name_max);
    let header = format!(" {name}  {badge}  {tag} ");
    let fill = inner.saturating_sub(3 + cell_width(&header)).max(1);
    let mut s = String::with_capacity(inner * 3);
    s.push(TL);
    s.push(HBAR);
    s.push_str(&header);
    for _ in 0..fill {
        s.push(HBAR);
    }
    s.push(TR);
    (s, name)
}

/// A plain border line `left + ─…─ + right` of EXACTLY `inner` cells (tui_v3
/// `_border`). PURE.
fn border_line(left: char, right: char, inner: usize) -> String {
    let dashes = inner.saturating_sub(2).max(0);
    let mut s = String::with_capacity(inner * 3);
    s.push(left);
    for _ in 0..dashes {
        s.push(HBAR);
    }
    s.push(right);
    s
}

/// Lay one interior body row as `│ {content}{pad} │` of EXACTLY `inner` cells
/// (`content_w = inner - 4`; tui_v3 `_boxln`). The content is clipped to `content_w`
/// first, then right-padded with spaces. PURE.
fn interior_row(content: &str, content_w: usize) -> String {
    let c = clip_cells(content, content_w);
    let pad = content_w.saturating_sub(cell_width(&c));
    let mut s = String::with_capacity((content_w + 4) * 3);
    s.push(VBAR);
    s.push(' ');
    s.push_str(&c);
    for _ in 0..pad {
        s.push(' ');
    }
    s.push(' ');
    s.push(VBAR);
    s
}

/// Pluck a useful one-line HINT from a tool's args (tui_v3 `_arg_hint`): the first
/// priority field (`command/script/path/file_path/url/query/question`) of a JSON
/// args object, else the first string value, else the args verbatim. Empty args yield
/// an EMPTY hint (no row) — the result preview, not a fabricated hint, shows the
/// output, so an empty-args tool never duplicates its first result line. File paths
/// are shortened to `…/basename`; the hint is clipped to 60 cells. Args are matched
/// textually (no JSON dep) — robust to the un-escaped `\n` that breaks strict JSON on
/// multi-line scripts. PURE.
fn arg_hint(name: &str, args: &str) -> String {
    if args.trim().is_empty() {
        return String::new();
    }
    let mut src = String::new();
    for key in ["command", "script", "path", "file_path", "url", "query", "question"] {
        if let Some(v) = json_string_field(args, key) {
            if !v.trim().is_empty() {
                src = v;
                break;
            }
        }
    }
    if src.is_empty() {
        if let Some(v) = first_json_string_value(args) {
            src = v;
        }
    }
    if src.is_empty() {
        // Not JSON-ish (e.g. a bare `config.toml`): use the args verbatim.
        src = args.to_string();
    }
    let first = src.split('\n').next().unwrap_or("").trim().to_string();
    let shortened = if matches!(name, "file_read" | "file_write" | "file_patch")
        && first.contains('/')
    {
        match first.rsplit_once('/') {
            Some((_, base)) => format!("…/{base}"),
            None => first,
        }
    } else {
        first
    };
    clip_cells(&shortened, 60)
}

/// The first few CONTENT lines of a tool result (tui_v3 `_result_preview`, refined):
/// STRIP GA's `[Action]/[Status]/[Info]/…` meta PREFIX and keep the line's content
/// (so a tool whose only output is `[Info] ok` still shows `ok` inside the box,
/// instead of an empty box), drop lines that are empty after stripping, unwrap a JSON
/// envelope (`stdout/output/result/content/text`) so the preview shows real content
/// not serialization noise, trim blank edges, clip each kept line to `row_w` cells
/// with a trailing `…`. Returns `(rows, overflow)`: `rows` is at most `max_rows` (or
/// ALL when `expanded`), `overflow` is how many content lines sit BEYOND the
/// collapsed `max_rows` preview — reported regardless of `expanded` so the caller can
/// always tell whether the result is foldable (and draw `… +N more` collapsed / `▾`
/// expanded). 0 ⇒ the result fits the preview (no affordance). PURE.
fn result_preview(
    result: &str,
    max_rows: usize,
    row_w: usize,
    expanded: bool,
) -> (Vec<String>, usize) {
    if result.trim().is_empty() {
        return (Vec::new(), 0);
    }
    let unwrapped = unwrap_json_envelope(result);
    let mut lines: Vec<String> = unwrapped
        .split('\n')
        .filter_map(strip_meta_prefix)
        .collect();
    while lines.first().is_some_and(|l| l.trim().is_empty()) {
        lines.remove(0);
    }
    while lines.last().is_some_and(|l| l.trim().is_empty()) {
        lines.pop();
    }
    if lines.is_empty() {
        return (Vec::new(), 0);
    }
    // The overflow is measured against the COLLAPSED preview height — it is what
    // makes the result foldable, independent of the current expand state.
    let overflow = lines.len().saturating_sub(max_rows);
    let take = if expanded { lines.len() } else { max_rows.min(lines.len()) };
    let mut out: Vec<String> = Vec::with_capacity(take);
    for ln in &lines[..take] {
        out.push(clip_row(ln.trim_end(), row_w));
    }
    (out, overflow)
}

/// Strip a leading GA tool-result META marker prefix
/// (`[Action]/[Status]/[Info]/[Debug]/[Warn]/[Warning]/[Error]/[Stdout]/[Stderr]`)
/// from a result line and return its remaining content, or `None` to DROP the line
/// when nothing meaningful is left after the prefix (a bare meta marker is noise).
/// A non-meta line is returned unchanged. PURE.
fn strip_meta_prefix(line: &str) -> Option<String> {
    let l = line.trim_start();
    for tag in [
        "[Action]", "[Status]", "[Info]", "[Debug]", "[Warn]", "[Warning]", "[Error]",
        "[Stdout]", "[Stderr]",
    ] {
        if let Some(rest) = l.strip_prefix(tag) {
            let content = rest.trim();
            return if content.is_empty() { None } else { Some(content.to_string()) };
        }
    }
    Some(line.to_string())
}

/// If `result` is a `{…}` JSON envelope, return the first meaningful string field
/// (`stdout/output/result/content/text`); else return `result` unchanged (owned).
/// PURE.
fn unwrap_json_envelope(result: &str) -> String {
    let s = result.trim();
    if s.starts_with('{') && s.ends_with('}') {
        for key in ["stdout", "output", "result", "content", "text"] {
            if let Some(v) = json_string_field(s, key) {
                if !v.trim().is_empty() {
                    return v;
                }
            }
        }
    }
    result.to_string()
}

/// Extract the string value of `"key": "…"` from a JSON-ish object textually
/// (un-escaping `\n`/`\t`/`\"`/`\\`/`\/`), tolerant of the un-escaped newlines that
/// break strict JSON. Returns the value up to the first un-escaped closing quote, or
/// `None` if the key/opening quote isn't found. PURE.
fn json_string_field(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let mut from = 0usize;
    while let Some(rel) = json[from..].find(&needle) {
        let after_key = from + rel + needle.len();
        let rest = &json[after_key..];
        // Expect `:` then (optional spaces) an opening quote for a STRING value.
        if let Some(colon) = rest.find(':') {
            let val = &rest[colon + 1..];
            if let Some((j, c)) = val.char_indices().find(|(_, c)| !c.is_whitespace()) {
                if c == '"' {
                    return Some(unescape_json_string(&val[j + 1..]));
                }
            }
        }
        from = after_key; // a non-string value for this key — keep searching.
    }
    None
}

/// The first string VALUE of any `"k": "v"` pair in a JSON-ish object (tui_v3 falls
/// back to `next(d.values())` of string type). PURE.
fn first_json_string_value(json: &str) -> Option<String> {
    let bytes = json.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b != b':' {
            continue;
        }
        // The first non-whitespace char after `:` must be a quote for a string value.
        let rest = &json[i + 1..];
        if let Some((j, c)) = rest.char_indices().find(|(_, c)| !c.is_whitespace()) {
            if c == '"' {
                return Some(unescape_json_string(&rest[j + 1..]));
            }
        }
    }
    None
}

/// Decode a JSON string body up to its first UN-escaped `"`: handle `\n \t \r \" \\
/// \/`; leave other escapes as their literal second char. PURE.
fn unescape_json_string(body: &str) -> String {
    let mut out = String::new();
    let mut chars = body.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => break,
            '\\' => match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('/') => out.push('/'),
                Some(other) => out.push(other),
                None => break,
            },
            _ => out.push(c),
        }
    }
    out
}

/// Wrap `text` to rows of at most `width` cells (CJK-correct, greedy by grapheme; no
/// word-boundary fanciness — the box body is dense). Always yields ≥1 row for
/// non-empty text. tui_v3 `_wrap_cells`. PURE.
fn wrap_cells(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut rows: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut acc = 0usize;
    for g in unicode_segmentation::UnicodeSegmentation::graphemes(text, true) {
        if g == "\n" {
            rows.push(std::mem::take(&mut cur));
            acc = 0;
            continue;
        }
        let gw = UnicodeWidthStr::width(g);
        if acc + gw > width && acc > 0 {
            rows.push(std::mem::take(&mut cur));
            acc = 0;
        }
        cur.push_str(g);
        acc += gw;
    }
    if !cur.is_empty() || rows.is_empty() {
        rows.push(cur);
    }
    rows
}

/// Clip a line to `row_w` cells, appending a `…` when it overflows (tui_v3
/// `_result_preview` row clip). PURE.
fn clip_row(s: &str, row_w: usize) -> String {
    if cell_width(s) <= row_w {
        return s.to_string();
    }
    let clipped = clip_cells(s, row_w.saturating_sub(1).max(1));
    format!("{clipped}…")
}

/// Display width of a string in terminal cells. PURE.
fn cell_width(s: &str) -> usize {
    UnicodeWidthStr::width(s)
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

    /// The box form (R1 ITEM 2): the tool name + `✓ ok` badge + `·tN` tag live ON
    /// the top border (accent corners), the arg-hint is the FIRST interior row, the
    /// `[Info]`-meta result line is the next interior row (meta prefix skipped, body
    /// kept), and the box closes with a `╰─…─╯` bottom border. Every row is exactly
    /// `inner` cells wide so it never soft-wraps to >1 row.
    #[test]
    fn box_form_name_badge_tag_on_top_border() {
        let call = ToolCall {
            id: 1,
            name: "web_scan".into(),
            args: "{\"tabs_only\": true}".into(),
            result: "[Info] 3 tabs scanned · ok".into(),
            status: ToolStatus::Ok,
        };
        let chip = render_chip_box(&call, 60, 4, false);
        // Top border: accent corners, bold name, badge, dim tag all ON the line.
        assert!(chip.top.starts_with("╭─"), "accent top-left corner + dash: {:?}", chip.top);
        assert!(chip.top.ends_with('╮'), "accent top-right corner: {:?}", chip.top);
        assert!(chip.top.contains("web_scan"), "tool name on the top border");
        assert!(chip.top.contains("✓ ok"), "status badge on the top border");
        assert!(chip.top.contains("·t1"), "turn-id tag on the top border");
        // Interior: the arg-hint row (tabs_only field), then the result line.
        assert!(chip.interior.iter().any(|r| r.contains("tabs_only")), "arg-hint row: {:?}", chip.interior);
        assert!(
            chip.interior.iter().any(|r| r.contains("3 tabs scanned")),
            "the [Info]-derived result row is INSIDE the box (meta prefix skipped): {:?}",
            chip.interior
        );
        assert!(!chip.interior.iter().any(|r| r.contains("[Info]")), "the meta prefix is stripped");
        // Bottom border.
        assert!(chip.bottom.starts_with('╰') && chip.bottom.ends_with('╯'), "accent bottom border: {:?}", chip.bottom);
        // Every row is exactly the same (inner) width → one visual row each.
        let w0 = cell_width(&chip.top);
        for row in chip.rows() {
            assert_eq!(cell_width(&row), w0, "every box row is the same cell width: {row:?}");
        }
    }

    /// A PENDING tool shows the `· …` badge on the top border (tui_v3 `stcol+sti`).
    #[test]
    fn pending_uses_dim_badge() {
        let call = ToolCall {
            id: 2,
            name: "ask_user".into(),
            args: String::new(),
            result: "Waiting for your answer ...".into(),
            status: ToolStatus::Pending,
        };
        let chip = render_chip_box(&call, 40, 4, false);
        assert!(chip.top.contains("· …"), "pending badge on the top border: {:?}", chip.top);
        assert!(chip.top.contains("ask_user"), "name on the top border");
        assert!(chip.top.contains("·t2"), "turn-id tag");
    }

    #[test]
    fn result_overflow_folds_inside_the_box() {
        let call = ToolCall {
            id: 1,
            name: "x".into(),
            args: String::new(),
            result: (0..10).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n"),
            status: ToolStatus::Ok,
        };
        let chip = render_chip_box(&call, 30, 3, false);
        // 10 content lines, max 3 preview → the last interior row is a `▸ +7 more`
        // fold affordance, INSIDE the box (between the borders). S1: affordance
        // text changed from `… +N more` to `▸ +N more` for clearer discoverability.
        assert!(chip.expandable, "an overflowing result is foldable");
        let last = chip.interior.last().unwrap();
        assert!(last.contains("▸ +7 more"), "fold affordance inside the box: {last:?}");
        // line 0..2 are shown; line 9 (past the preview) is hidden when collapsed.
        assert!(chip.interior.iter().any(|r| r.contains("line 0")));
        assert!(!chip.interior.iter().any(|r| r.contains("line 9")), "overflow line hidden: {:?}", chip.interior);
    }

    /// Fix E acceptance: an EXPANDED tool result skips the `max_preview` truncation
    /// (every result line shows) and swaps the `… +N more` affordance for a `▾`
    /// collapse row, still INSIDE the box.
    #[test]
    fn expanded_tool_result_skips_truncation() {
        let call = ToolCall {
            id: 1,
            name: "x".into(),
            args: String::new(),
            result: (0..10).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n"),
            status: ToolStatus::Ok,
        };
        let collapsed = render_chip_box(&call, 30, 3, false);
        // 3 preview rows + the affordance row = 4 interior rows.
        assert_eq!(collapsed.interior.len(), 4);

        let expanded = render_chip_box(&call, 30, 3, true);
        // ALL 10 result rows + a `▾` collapse row = 11 interior rows.
        assert_eq!(expanded.interior.len(), 11, "all 10 lines + the ▾ affordance");
        assert!(expanded.expandable);
        for i in 0..10 {
            assert!(
                expanded.interior.iter().any(|r| r.contains(&format!("line {i}"))),
                "expanded result must contain every line (line {i} missing)"
            );
        }
        let last = expanded.interior.last().unwrap();
        assert!(last.contains('▾'), "expanded affordance is a ▾ collapse row: {last:?}");
        assert!(!expanded.interior.iter().any(|r| r.contains("more")), "no `+N more` when expanded");

        // A result that FITS in max_preview is not foldable in either state.
        let small = ToolCall {
            id: 2, name: "y".into(), args: String::new(),
            result: "one\ntwo".into(), status: ToolStatus::Ok,
        };
        assert!(!render_chip_box(&small, 30, 4, false).expandable);
        assert!(!render_chip_box(&small, 30, 4, true).expandable);
    }

    /// The arg-hint plucks the first priority field of a JSON args object (tui_v3
    /// `_arg_hint`); a file path is shortened to `…/basename`; the result envelope is
    /// unwrapped to real content.
    #[test]
    fn arg_hint_and_result_preview_pluck_real_content() {
        // Priority field `script` is plucked from JSON args (un-escaped \n tolerated).
        assert_eq!(
            arg_hint("code_run", "{\"cwd\": \"/tmp\", \"script\": \"echo hi\"}"),
            "echo hi"
        );
        // A file path arg is shortened to its basename.
        assert_eq!(
            arg_hint("file_read", "{\"path\": \"/a/b/config.toml\"}"),
            "…/config.toml"
        );
        // Bare (non-JSON) args are used verbatim.
        assert_eq!(arg_hint("file_read", "config.toml"), "config.toml");
        // Empty args yield no hint (the result preview shows the output instead).
        assert_eq!(arg_hint("run", ""), "");
        // A JSON envelope result is unwrapped to its `stdout` field, meta skipped.
        let (rows, overflow) =
            result_preview("{\"status\": \"ok\", \"stdout\": \"the real output\"}", 4, 40, false);
        assert_eq!(rows, vec!["the real output".to_string()]);
        assert_eq!(overflow, 0);
    }

    #[test]
    fn cjk_box_rows_stay_exact_width() {
        let call = ToolCall {
            id: 1,
            name: "搜索".into(),
            args: "查询: 你好世界".into(),
            result: "结果在这里很长很长很长很长很长很长很长很长很长很长".into(),
            status: ToolStatus::Ok,
        };
        let chip = render_chip_box(&call, 24, 4, false);
        // Every box row is EXACTLY the inner width (CJK counted as 2 cells) so none
        // soft-wraps to a second visual row — the parity invariant by construction.
        let w0 = cell_width(&chip.top);
        for row in chip.rows() {
            assert_eq!(cell_width(&row), w0, "row not exact width: {row:?}");
        }
    }
}
