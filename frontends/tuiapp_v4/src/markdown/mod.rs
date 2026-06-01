//! markdown/ — the markdown + math render plane (checklist §1 P3, §3 markdown/*,
//! §11 Phase 3). Turns an assistant message's SOURCE text into themed
//! `ratatui::text::Line`s: CommonMark + GFM via [`render`] (pulldown-cmark →
//! styled spans), fenced-code syntax highlighting via [`highlight`] (syntect,
//! fancy-regex / pure Rust), and `$…$`/`$$…$$` LaTeX → Unicode via [`math`]
//! (`latex_to_unicode`) — the surpass-Claude-Code math feature.
//!
//! Assistant transcript blocks are routed through [`render_assistant`]; user /
//! tool / notice blocks keep the plain wrap path (they aren't markdown). The
//! whole plane is pure over `(source, theme)` and degrades gracefully — math
//! parse failures return literal LaTeX, code highlight failures return dim plain
//! text, malformed markdown returns its text content — so it can never panic or
//! corrupt the live stream.

pub mod code;
pub mod highlight;
pub mod inline_math;
pub mod math;
pub mod render;
pub mod table;

use std::ops::RangeInclusive;

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

use crate::render::fold::{BlockFolds, NodeId};
use crate::theme::Theme;

/// The full result of a cockpit assistant render (Fix E / Q8): the styled visual
/// rows the transcript draws PLUS the clickable-node hit map. Both come from ONE
/// render so they can never disagree about which visual row a node spans.
#[derive(Debug, Clone, Default)]
pub struct CockpitRender {
    /// The styled, soft-wrapped visual rows (still carrying the ATOMIC sentinel; the
    /// draw boundary strips it).
    pub lines: Vec<Line<'static>>,
    /// Per-node VISUAL-row ranges (inclusive, in `lines` coordinates): a fold-header
    /// row maps to its [`NodeId::Turn`]; a tool-bullet's rows map to its
    /// [`NodeId::Tool`]. A left-click on the triangle/bullet column of a row in one
    /// of these ranges toggles that node's fold. Built by coalescing the per-logical-
    /// line node tags after the wrap, so it is exact w.r.t. the drawn rows.
    pub node_hits: Vec<(RangeInclusive<usize>, NodeId)>,
}

/// Render an assistant block's `source` into themed, pre-styled lines (P3). This
/// is the routing entry the transcript uses for `BlockRole::Assistant`.
///
/// It first checks whether the whole source is a single `$$…$$` display-math
/// paragraph (rendered as a multi-line stacked layout); otherwise it runs the
/// full markdown walker (which also handles inline `$…$` math, fenced code
/// highlighting, tables, lists, blockquotes, links, and inline styles).
///
/// The returned lines are LOGICAL rows (already styled, not yet soft-wrapped to a
/// terminal width); the caller's wrap cache handles width-aware wrapping, so P1
/// (resize-safe scroll) and the CJK-correct measurement still own line breaking.
pub fn render_assistant(source: &str, theme: &Theme) -> Vec<Line<'static>> {
    if let Some(latex) = inline_math::extract_block_math(source) {
        return inline_math::render_block_math(&latex, theme);
    }
    render::render_markdown(source, theme)
}

/// Render an assistant block's `source` with the COCKPIT layer applied
/// (checklist §5/§10): per-turn folds collapse to one `▸ summary` line and tool
/// calls render as boxed chips. This is the Phase-2 transcript routing; the plain
/// projection ([`render_assistant_cockpit_plain`]) mirrors it line-for-line so
/// the wrap cache's row accounting stays exact (P1).
///
/// `fold_all` folds the FINAL turn too (Ctrl+O). `width` sizes the chip boxes.
/// Convenience over [`render_assistant_cockpit_streaming`] for FINALIZED blocks
/// (no volatile tail to hold back).
#[allow(dead_code)] // finalized-convenience cockpit render (exercised by the row-count tests).
pub fn render_assistant_cockpit(
    source: &str,
    theme: &Theme,
    fold_all: bool,
    width: u16,
) -> Vec<Line<'static>> {
    render_assistant_cockpit_streaming(source, theme, fold_all, width, false)
}

/// Cockpit assistant render with STRUCTURAL-BOUNDARY stream commit (§10): when
/// `streaming` is true, only the part of `source` up to a safe structural
/// boundary ([`crate::render::fold::safe_commit_pos`]) gets the fold + chip
/// transform; the in-flight tail past it is rendered as PLAIN dim text so a
/// half-written `🛠️ Tool:` header / `<summary>` never flashes as a broken chip or
/// an orphaned header. For a finalized block the whole source is committed.
pub fn render_assistant_cockpit_streaming(
    source: &str,
    theme: &Theme,
    fold_all: bool,
    width: u16,
    streaming: bool,
) -> Vec<Line<'static>> {
    // The no-per-node-override path (the default fold policy). The cockpit cache /
    // transcript go through [`render_assistant_cockpit_full`] with the real
    // [`BlockFolds`]; this thin wrapper keeps the convenience signature + tests.
    render_assistant_cockpit_full(source, theme, &BlockFolds::plain(0, fold_all), width, streaming)
        .lines
}

/// Cockpit assistant render that also reports the clickable-node hit map (Fix E /
/// Q8). `folds` carries the global `fold_all` AND the per-node overrides for THIS
/// block, so each turn folds per [`fold_turns_with`](crate::render::fold) (override
/// wins, else completed-folded/last-open) and each tool result expands per its
/// override. Otherwise identical to [`render_assistant_cockpit_streaming`]: the same
/// head/tail commit, the same soft-wrap, the same row-count parity (the node hits
/// are derived from the SAME wrap, so they index real drawn rows).
pub fn render_assistant_cockpit_full(
    source: &str,
    theme: &Theme,
    folds: &BlockFolds,
    width: u16,
    streaming: bool,
) -> CockpitRender {
    use crate::render::fold::{fold_turns_with, safe_commit_pos, FoldSegment};

    // Split into a committable head (structurally stable) and a volatile tail.
    let (head, tail) = if streaming {
        let pos = safe_commit_pos(source);
        (&source[..pos], &source[pos..])
    } else {
        (source, "")
    };

    // Whole-source block math short-circuits the cockpit pass — but ONLY when the
    // block is finalized (not streaming). A half-typed `$$\frac{a}{` must stay in
    // the volatile tail (a partial `$$` would otherwise flash a wrong stack); when
    // streaming, display math is routed at the paragraph boundary in the walker
    // (Fix B) once the closing `$$` has committed into the head.
    if !streaming {
        if let Some(latex) = inline_math::extract_block_math(head) {
            let lines = inline_math::render_block_math(&latex, theme);
            return CockpitRender { lines, node_hits: Vec::new() };
        }
    }

    let segments =
        fold_turns_with(head, folds.fold_all, |turn, is_last| folds.turn_folded(turn, is_last));
    // Accumulate LOGICAL styled lines, each tagged with the clickable node it
    // belongs to (a fold header → its `Turn`, a tool bullet → its `Tool`; `None` for
    // ordinary prose/blank rows), then soft-wrap them all to `width` at the end so
    // the returned rows are 1:1 with the wrap cache (which wraps the plain projection
    // of these same lines at the same width) — the P1 row-count invariant. The node
    // hit map is coalesced from the per-line tags AFTER the wrap so its row ranges
    // index real drawn rows. A running block-global tool index gives each tool a
    // stable [`NodeId::Tool`] across turns.
    let mut logical: Vec<(Line<'static>, Option<NodeId>)> = Vec::new();
    let mut tool_idx: u32 = 0;
    for (si, seg) in segments.iter().enumerate() {
        // §2.4 turn separation: a BLANK line between turn segments (no rule, no
        // "Turn N" text). The preamble (segment 0) gets none.
        if si > 0 {
            logical.push((Line::default(), None));
        }
        match seg {
            FoldSegment::Fold { title, turn, .. } => {
                // A folded turn is EXACTLY one `▸ summary` line (dimmed), tagged with
                // its turn node so a click on the `▸` expands it. The title is
                // collapsed so it never itself wraps to >1 row.
                logical.push((
                    Line::from(Span::styled(
                        fold_header_line(title, width),
                        Style::default()
                            .fg(theme.color(crate::theme::Token::Dim))
                            .add_modifier(Modifier::ITALIC),
                    )),
                    Some(NodeId::Turn { block: folds.block_id, turn: *turn }),
                ));
            }
            FoldSegment::Text { body } => {
                render_turn_body(body, theme, width, folds, &mut tool_idx, &mut logical);
            }
        }
    }
    // The volatile tail: render it through the SAME `render_turn_body` path the
    // committed head uses (Fix D) — so a line streams already markdown/chip-styled
    // instead of flashing as raw dim text and then "upgrading" when it crosses the
    // commit boundary (C1-F10 visual seam). `strip_leading_turn_line` drops a fresh
    // `Turn N ...` marker that arrived in the live tail before `safe_commit_pos`
    // advanced past it, so the literal marker never shows dim in the active region
    // (C1-F6 tail leak). A half-built `$…$`/`$$…$$` at the tail end is trimmed off
    // first: the markdown walker paints an unclosed run as raw `$`/`$$`, so we hold
    // the in-flight formula unpainted until its closer commits (C1-F3 tail half) —
    // the prose BEFORE it still streams. Geometry stays consistent because BOTH the
    // styled draw (`render_transcript`) and the plain projection (`to_render_block`)
    // call THIS fn — the tail renders identically on both sides, so rows agree.
    if !tail.is_empty() {
        let tail_body = strip_leading_turn_line(tail);
        let tail_body = trim_inflight_math(tail_body);
        if !tail_body.is_empty() {
            render_turn_body(tail_body, theme, width, folds, &mut tool_idx, &mut logical);
        }
    }

    // Soft-wrap every logical line to `width` (CJK-correct, same algorithm as the
    // cache), producing the visual rows the transcript draws; coalesce the per-line
    // node tags into visual-row ranges as we go.
    let mut out: Vec<Line<'static>> = Vec::new();
    let mut node_hits: Vec<(RangeInclusive<usize>, NodeId)> = Vec::new();
    for (line, tag) in &logical {
        let start = out.len();
        out.extend(wrap_styled_line(line, width));
        let end = out.len(); // exclusive
        if let Some(node) = tag {
            if end > start {
                merge_node_hit(&mut node_hits, *node, start, end - 1);
            }
        }
    }
    if out.is_empty() {
        out.push(Line::default());
    }
    CockpitRender { lines: out, node_hits }
}

/// Append `[start, end]` (inclusive visual-row range) for `node` to `hits`, MERGING
/// it with the last entry when that entry is the same node and immediately precedes
/// this range (so a multi-row tool bullet — header + result rows — becomes ONE
/// contiguous clickable span). PURE.
fn merge_node_hit(
    hits: &mut Vec<(RangeInclusive<usize>, NodeId)>,
    node: NodeId,
    start: usize,
    end: usize,
) {
    if let Some((range, last)) = hits.last_mut() {
        if *last == node && *range.end() + 1 == start {
            *range = *range.start()..=end;
            return;
        }
    }
    hits.push((start..=end, node));
}

/// Build a fold header line `▸ <title>` clipped so it never wraps past `width`
/// (a fold MUST be exactly one visual row). PURE.
fn fold_header_line(title: &str, width: u16) -> String {
    let prefix = "▸ ";
    let avail = (width as usize).saturating_sub(2).max(1);
    let mut out = String::from(prefix);
    let mut acc = 0usize;
    for g in unicode_segmentation::UnicodeSegmentation::graphemes(title, true) {
        let gw = UnicodeWidthStr::width(g);
        if acc + gw > avail {
            out.push('…');
            break;
        }
        out.push_str(g);
        acc += gw;
    }
    out
}

/// The plain-text projection of [`render_assistant_cockpit`] — the concatenated
/// span text per line. It stays 1:1 with `render_assistant_cockpit` (same fold +
/// chip transform), so it simply renders then flattens. The wrap cache itself
/// consumes the atomic-aware projection ([`lines_to_plain_atomic`] via
/// [`crate::app::Block::to_render_block`]) so inline-math runs stay intact (P1);
/// this text-only view is kept for the row-count parity + content tests.
#[allow(dead_code)] // cockpit-plain projection (exercised by the row-count tests).
pub fn render_assistant_cockpit_plain(
    source: &str,
    theme: &Theme,
    fold_all: bool,
    width: u16,
) -> String {
    let lines = render_assistant_cockpit(source, theme, fold_all, width);
    lines_to_plain(&lines)
}

/// Render ONE turn's body (a `fold_turns` `Text` segment, or a whole message) into
/// `logical` as `(styled line, node tag)` pairs (redesign_cc.md §1/§2):
///   * the leading `Turn N ...` boundary line is DROPPED (it's spacing, never text);
///   * a `<summary>…</summary>` becomes a dim italic breadcrumb above the body
///     (the tags themselves are HIDDEN — never shown raw);
///   * compact `🛠️ name(args)` calls render as CC `⏺`/`○` bullets — each tagged with
///     a stable block-global [`NodeId::Tool`] (so a click expands its result) and
///     expanded per `folds`; `tool_idx` is the running block-global tool counter;
///   * `[Action]/[Status]/[Info]` result prefixes stay dim;
///   * `!!!Error: …` renders as a compact dim/red line;
///   * everything else flows through the markdown walker.
/// Prose / breadcrumb rows are tagged `None`. PURE over `(body, theme, width, folds)`
/// + the `tool_idx` it advances.
fn render_turn_body(
    body: &str,
    theme: &Theme,
    width: u16,
    folds: &BlockFolds,
    tool_idx: &mut u32,
    logical: &mut Vec<(Line<'static>, Option<NodeId>)>,
) {
    use crate::theme::Token;

    let start_len = logical.len();
    let push_prose = |lines: Vec<Line<'static>>, logical: &mut Vec<(Line<'static>, Option<NodeId>)>| {
        logical.extend(lines.into_iter().map(|l| (l, None)));
    };

    // 1) Strip the leading `Turn N ...` boundary line (it is the turn marker, not
    //    content — §1: do NOT render it as text).
    let body = strip_leading_turn_line(body);

    // 2) Hoist the `<summary>…</summary>` to a dim breadcrumb and remove the tags
    //    (+ their inner text) from the flowing body so they never show raw (§1).
    let (breadcrumb, body_no_summary) = hoist_summary(body);
    if let Some(crumb) = breadcrumb {
        let crumb = collapse_ws(&crumb);
        if !crumb.is_empty() {
            logical.push((
                Line::from(Span::styled(
                    format!("↳ {}", clip_to_cells(&crumb, (width as usize).saturating_sub(2).max(1))),
                    Style::default()
                        .fg(theme.color(Token::Dim))
                        .add_modifier(Modifier::ITALIC),
                )),
                None,
            ));
        }
    }

    // 3) Interleave prose (markdown) and compact tool-call bullets, preserving order.
    let body = &body_no_summary;
    let calls = crate::render::chip::parse_tool_calls(body);
    if calls.is_empty() {
        push_prose(render_prose_with_inline_markers(body, theme), logical);
        if logical.len() == start_len {
            logical.push((Line::default(), None));
        }
        return;
    }

    let mut cursor = 0usize;
    for call in &calls {
        if let Some(rel) = body[cursor..].find(crate::render::chip::TOOL_MARK) {
            let header_start = cursor + rel;
            // Prose before the chip.
            let prose = &body[cursor..header_start];
            if !prose.trim().is_empty() {
                push_prose(render_prose_with_inline_markers(prose, theme), logical);
            }
            // The CC tool-call bullet (NOT a box), tagged with its block-global node.
            let node = NodeId::Tool { block: folds.block_id, tool: *tool_idx };
            let expanded = folds.tool_expanded(*tool_idx);
            push_tool_bullet(logical, call, theme, width, node, expanded);
            *tool_idx += 1;
            // Advance the cursor past this call's result (the next structural marker).
            let header_line_end = body[header_start..]
                .find('\n')
                .map(|i| header_start + i + 1)
                .unwrap_or(body.len());
            cursor = crate::render::chip::next_marker_boundary(body, header_line_end);
        }
    }
    // Trailing prose after the last chip.
    let tail = &body[cursor..];
    if !tail.trim().is_empty() {
        push_prose(render_prose_with_inline_markers(tail, theme), logical);
    }
    if logical.len() == start_len {
        logical.push((Line::default(), None));
    }
}

/// Push a tool call as CC bullet rows (§2.3): `⏺ name  args` (bullet + name in the
/// status color, args dim) then the 2-col-indented dim result rows. No box. Every
/// row of the bullet (header + result, including the `▸`/`▾` affordance) is tagged
/// with `node` so the WHOLE bullet is one clickable expand/collapse target (Fix E);
/// `expanded` skips the result truncation.
fn push_tool_bullet(
    logical: &mut Vec<(Line<'static>, Option<NodeId>)>,
    call: &crate::render::chip::ToolCall,
    theme: &Theme,
    width: u16,
    node: NodeId,
    expanded: bool,
) {
    use crate::theme::Token;
    let chip = crate::render::chip::render_chip_bullet(call, width, 4, expanded);
    let status_tok = chip.status.token();
    // The bullet is a clickable node ONLY when it can actually fold (a long result);
    // a short result has nothing to expand, so its rows stay un-tagged (a click there
    // falls through to native selection, like ordinary prose).
    let tag = if chip.expandable { Some(node) } else { None };

    // Header: the bullet + name colored by status, then a dim one-line args.
    let mut head: Vec<Span> = vec![Span::styled(
        chip.header_name.clone(),
        Style::default().fg(theme.color(status_tok)),
    )];
    if !chip.args.is_empty() {
        head.push(Span::styled(
            format!("  {}", chip.args),
            Style::default().fg(theme.color(Token::Dim)),
        ));
    }
    logical.push((Line::from(head), tag));

    // Result rows: dim, already indented two columns (+ the `▸`/`▾` affordance row).
    // A `!!!Error:` line inside the result is colored red (a real model/stream
    // error surfaced in the tool's output) — §1 "compact dim/red error line". The
    // affordance row (`▸`/`▾`) is colored as a SUGGESTION so it reads as actionable.
    for row in &chip.result_rows {
        let trimmed = row.trim_start();
        let tok = if trimmed.starts_with("!!!Error") {
            Token::Error
        } else if trimmed.starts_with('▸') || trimmed.starts_with('▾') {
            Token::Suggestion
        } else {
            Token::Dim
        };
        logical.push((
            Line::from(Span::styled(row.clone(), Style::default().fg(theme.color(tok)))),
            tag,
        ));
    }
}

/// Render prose that may carry inline `[Action]/[Status]/[Info]` result lines and
/// `!!!Error:` lines. A line that IS one of those markers is emitted as a single
/// styled line (dim for the result prefixes, dim/red for `!!!Error:`); contiguous
/// runs of ordinary prose between them flow through the markdown walker so bold /
/// code / links / math still render. PURE over `(text, theme)`.
fn render_prose_with_inline_markers(text: &str, theme: &Theme) -> Vec<Line<'static>> {
    use crate::theme::Token;
    let mut out: Vec<Line<'static>> = Vec::new();
    let mut prose_buf: Vec<&str> = Vec::new();

    // Flush the buffered ordinary-prose lines through the markdown renderer.
    let flush = |buf: &mut Vec<&str>, out: &mut Vec<Line<'static>>| {
        if buf.is_empty() {
            return;
        }
        let joined = buf.join("\n");
        if joined.trim().is_empty() {
            // Preserve blank spacing as blank rows (one per buffered line).
            for _ in 0..buf.len() {
                out.push(Line::default());
            }
        } else {
            out.extend(render_assistant(&joined, theme));
        }
        buf.clear();
    };

    for line in text.split('\n') {
        let l = line.trim_start();
        if l.starts_with("!!!Error") {
            flush(&mut prose_buf, &mut out);
            // A real model/stream error → a compact dim/red line (no wall).
            out.push(Line::from(Span::styled(
                line.trim_end().to_string(),
                Style::default().fg(theme.color(Token::Error)),
            )));
        } else if is_result_prefix(l) {
            flush(&mut prose_buf, &mut out);
            // Tool result prefix lines stay dim (the prefix is kept, styled subtle).
            out.push(Line::from(Span::styled(
                line.trim_end().to_string(),
                Style::default().fg(theme.color(Token::Dim)),
            )));
        } else {
            prose_buf.push(line);
        }
    }
    flush(&mut prose_buf, &mut out);
    out
}

/// True if a (left-trimmed) line begins with a GA tool-result prefix
/// (`[Action]`/`[Status]`/`[Info]`/`[Stdout]`/`[Error]`). PURE.
fn is_result_prefix(l: &str) -> bool {
    l.starts_with("[Action]")
        || l.starts_with("[Status]")
        || l.starts_with("[Info]")
        || l.starts_with("[Stdout]")
        || l.starts_with("[Error]")
}

/// Drop a leading `Turn N ...` (bare or `**Turn N ...**` bold) boundary line from a
/// turn body — it is the turn marker, rendered as spacing, NEVER as text (§1).
/// Returns the body with that first line (and its trailing newline) removed; a
/// body that doesn't start with a turn line is returned unchanged. PURE.
fn strip_leading_turn_line(body: &str) -> &str {
    // Only strip when the FIRST line is the turn marker.
    if crate::render::chip::find_turn_line(body) == Some(0) {
        match body.find('\n') {
            Some(nl) => &body[nl + 1..],
            None => "",
        }
    } else {
        body
    }
}

/// Trim a TRAILING unclosed math run (`$…`/`$$…`) off a volatile-tail body so the
/// in-flight formula isn't painted as raw `$`/`$$` while it streams (C1-F3 tail
/// half): the markdown walker emits literal dollars for an unclosed delimiter, so
/// a half-typed `$$\frac{a}{` would flash raw `$$` in the active region. We walk
/// the body with the SAME delimiter rules as
/// [`inline_math::split_inline_math`](crate::markdown::inline_math) — `\$` is a
/// literal, a BALANCED `$…$`/`$$…$$` is skipped whole — and return the body cut at
/// the first delimiter that has no closer (the in-flight opener), so the stable
/// prose BEFORE it still renders. The committed head already excludes this region
/// (`safe_commit_pos` holds it back), so the cut only affects the volatile tail.
/// A lone `$5`-style currency dollar (no math-ish closer ahead) is NOT trimmed.
/// PURE.
fn trim_inflight_math(body: &str) -> &str {
    let bytes = body.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' if i + 1 < bytes.len() => i += 2, // an escaped char (incl. `\$`)
            b'$' => {
                let double = i + 1 < bytes.len() && bytes[i + 1] == b'$';
                let delim = if double { 2 } else { 1 };
                let body_start = i + delim;
                match find_close_dollar(bytes, body_start, double) {
                    // Balanced: skip the whole `$…$`/`$$…$$` span and keep scanning.
                    Some(close) => i = close + delim,
                    // No closer → this `$`/`$$` opens the in-flight formula. A `$$`
                    // block opener is always math; a single `$` is only trimmed when
                    // a plausible formula follows (so a stray `$5.00` price stays).
                    None => {
                        if double || opens_inline_math(&body[body_start..]) {
                            return &body[..i];
                        }
                        i += 1;
                    }
                }
            }
            _ => i += 1,
        }
    }
    body
}

/// Byte offset of the FIRST char of the closing `$`/`$$` for a math span whose
/// interior starts at `from`, honoring `\$` escapes — the byte-coordinate twin of
/// [`inline_math::find_math_close`](crate::markdown::inline_math). `None` if the
/// run never closes. PURE.
fn find_close_dollar(bytes: &[u8], from: usize, double: bool) -> Option<usize> {
    let mut i = from;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' if i + 1 < bytes.len() => i += 2,
            b'$' if double => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'$' {
                    return Some(i);
                }
                i += 1; // a lone `$` is allowed inside `$$…$$`
            }
            b'$' => return Some(i),
            _ => i += 1,
        }
    }
    None
}

/// Whether an unclosed single-`$` run starting with `rest` plausibly opens inline
/// math (vs. a `$5` currency dollar): a non-empty, non-space-led interior with a
/// math-ish first chunk — the in-flight twin of `inline_math::is_probably_math`'s
/// leading guard. PURE.
fn opens_inline_math(rest: &str) -> bool {
    let mut chars = rest.chars();
    match chars.next() {
        None | Some(' ') => false,
        Some(c) => c.is_alphabetic() || c == '\\' || "+-*/=^_({".contains(c),
    }
}

/// Extract the FIRST `<summary>…</summary>` inner text as a breadcrumb and return
/// `(breadcrumb, body_without_the_summary_block)`. The whole `<summary>…</summary>`
/// span (tags + inner) is removed from the body so it never renders raw (§1). A
/// body with no summary returns `(None, body.to_string())`. PURE.
fn hoist_summary(body: &str) -> (Option<String>, String) {
    const OPEN: &str = "<summary>";
    const CLOSE: &str = "</summary>";
    let Some(open) = body.find(OPEN) else {
        return (None, body.to_string());
    };
    let after_open = open + OPEN.len();
    let Some(rel_close) = body[after_open..].find(CLOSE) else {
        // An unclosed tag (shouldn't reach a finalized render): hide just the
        // opener so a bare `<summary>` never shows, keep the rest as prose.
        let mut s = String::with_capacity(body.len());
        s.push_str(&body[..open]);
        s.push_str(&body[after_open..]);
        return (None, s);
    };
    let inner = body[after_open..after_open + rel_close].trim().to_string();
    let close_end = after_open + rel_close + CLOSE.len();
    // Splice out the whole `<summary>…</summary>` span.
    let mut s = String::with_capacity(body.len());
    s.push_str(&body[..open]);
    s.push_str(&body[close_end..]);
    let breadcrumb = if inner.is_empty() { None } else { Some(inner) };
    (breadcrumb, s)
}

/// Collapse internal whitespace runs to single spaces and trim. PURE.
fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Clip a string to at most `max` display cells (CJK-correct, no ellipsis). PURE.
fn clip_to_cells(s: &str, max: usize) -> String {
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

/// The plain-text projection of an assistant block's rendered markdown — the
/// concatenated span text per line. Used by the wrap cache / scroll math so the
/// VISIBLE rows it counts match what `render_assistant` actually draws (the
/// markdown transform changes the line set vs. the raw source: a table becomes
/// aligned rows, `$$frac$$` becomes 3 rows, etc.). Pure; styling-independent.
// Superseded at the call sites by the COCKPIT variants (fold + chip aware), but
// kept as the un-cockpit projection (exercised by the row-count parity test).
#[allow(dead_code)]
pub fn render_assistant_plain(source: &str, theme: &Theme) -> String {
    let lines = render_assistant(source, theme);
    lines_to_plain(&lines)
}

/// Concatenate the span text of styled lines into one `\n`-joined plain string.
pub fn lines_to_plain(lines: &[Line<'static>]) -> String {
    lines
        .iter()
        .map(|l| {
            l.spans
                .iter()
                .map(|s| s.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Like [`lines_to_plain`], but ALSO return the per-HARD-LINE atomic byte ranges
/// (rendered inline-math runs tagged [`render::ATOMIC`]), indexed by hard line
/// (split on `\n`, exactly as [`crate::render::measure::reflow_block`] splits the
/// source). The string is byte-identical to [`lines_to_plain`]; the ranges let the
/// wrap cache keep math glyph runs intact so it counts the SAME rows the styled
/// draw produces (P1). PURE.
pub fn lines_to_plain_atomic(lines: &[Line<'static>]) -> (String, Vec<Vec<std::ops::Range<usize>>>) {
    let plain = lines_to_plain(lines);
    // Walk the joined plain by hard line; within each hard line, record maximal
    // runs of ATOMIC-styled graphemes in that line's LOCAL byte coordinates. We
    // re-walk the spans against the joined string so an embedded `\n` (which a
    // hand-built line could carry) is split into separate hard lines correctly —
    // matching the cache's `source.split('\n')`. Math never embeds `\n`, so the
    // common case is one hard line per logical line.
    let mut per_line: Vec<Vec<std::ops::Range<usize>>> = Vec::new();
    let mut cur: Vec<std::ops::Range<usize>> = Vec::new();
    let mut local = 0usize; // byte offset within the current hard line
    let mut run_start: Option<usize> = None;
    let close = |cur: &mut Vec<std::ops::Range<usize>>, run_start: &mut Option<usize>, end: usize| {
        if let Some(start) = run_start.take() {
            cur.push(start..end);
        }
    };
    for (li, line) in lines.iter().enumerate() {
        if li > 0 {
            // The `\n` join boundary ends the previous hard line.
            close(&mut cur, &mut run_start, local);
            per_line.push(std::mem::take(&mut cur));
            local = 0;
        }
        for span in &line.spans {
            let atomic = span.style.add_modifier.contains(render::ATOMIC);
            for g in span.content.as_ref().graphemes(true) {
                if g == "\n" {
                    // A span-embedded newline: close the current hard line here.
                    close(&mut cur, &mut run_start, local);
                    per_line.push(std::mem::take(&mut cur));
                    local = 0;
                    continue;
                }
                match (atomic, run_start) {
                    (true, None) => run_start = Some(local),
                    (false, Some(start)) => {
                        cur.push(start..local);
                        run_start = None;
                    }
                    _ => {}
                }
                local += g.len();
            }
        }
    }
    close(&mut cur, &mut run_start, local);
    per_line.push(cur);
    (plain, per_line)
}

/// Soft-wrap one styled logical [`Line`] to `width` display cells, producing the
/// styled visual rows — using the SAME CJK/word-wrap algorithm as the plain wrap
/// cache ([`wrap_line_segments`]). This is what keeps the styled markdown draw
/// row-count-identical to the wrap cache that drives scroll math (P1): both wrap
/// the same plain text at the same width with the same algorithm, so segment
/// boundaries match exactly; here we additionally carry each grapheme's style
/// onto the matching segment.
///
/// Because `wrap_line_segments` trims boundary spaces (which are style-neutral),
/// we two-pointer-walk the styled graphemes against each segment's text: skip
/// leading/trailing spaces that the wrap dropped, then consume the segment's
/// graphemes carrying their original styles.
#[allow(dead_code)] // styled soft-wrap helper (exercised by the row-count test).
pub fn wrap_styled_line(line: &Line<'static>, width: u16) -> Vec<Line<'static>> {
    // Flatten to (grapheme, style) pairs (the styled grapheme stream).
    let mut gph: Vec<(&str, Style)> = Vec::new();
    for span in &line.spans {
        for g in span.content.as_ref().graphemes(true) {
            gph.push((g, span.style));
        }
    }

    // Split the styled grapheme stream into HARD lines on any embedded `\n`,
    // EXACTLY as `reflow_block` splits the plain projection (`source.split('\n')`)
    // before wrapping each. A logical line should never contain a `\n` (the
    // markdown walker now flushes per newline), but a `\n` could be reintroduced
    // by any hand-built `Line`; splitting here keeps the styled draw row-count
    // identical to the wrap cache for ALL inputs (the P1 invariant) instead of
    // silently merging two hard lines into one row. The `\n` grapheme itself is
    // dropped (it is the boundary, not content).
    let hard_lines = split_styled_hard_lines(&gph);

    let mut rows: Vec<Line<'static>> = Vec::new();
    for hard in &hard_lines {
        wrap_styled_hard_line(hard, width, &mut rows);
    }
    if rows.is_empty() {
        rows.push(Line::default());
    }
    rows
}

/// Split a styled grapheme stream into hard-line slices on `\n` (the `\n` is the
/// boundary and is excluded from both sides). Always yields at least one slice
/// (empty input → one empty slice), and a trailing `\n` yields a trailing empty
/// slice — matching `str::split('\n')` exactly so the styled wrap and the plain
/// `reflow_block` agree row-for-row.
fn split_styled_hard_lines<'a>(gph: &'a [(&'a str, Style)]) -> Vec<&'a [(&'a str, Style)]> {
    let mut out: Vec<&[(&str, Style)]> = Vec::new();
    let mut start = 0usize;
    for (i, (g, _)) in gph.iter().enumerate() {
        if *g == "\n" {
            out.push(&gph[start..i]);
            start = i + 1;
        }
    }
    out.push(&gph[start..]);
    out
}

/// Soft-wrap ONE hard line (a newline-free styled grapheme slice) to `width`,
/// pushing the styled visual rows onto `rows`. Uses the SAME
/// [`wrap_line_segments_atomic`](crate::render::measure::wrap_line_segments_atomic)
/// the cache uses — fed the SAME atomic byte-ranges (rendered inline-math runs,
/// tagged [`render::ATOMIC`]) — so the styled draw and the wrap cache agree
/// row-for-row including math (P1). Then carries each grapheme's style onto the
/// matching segment (skipping the boundary spaces the wrap trims).
///
/// The `ATOMIC` sentinel is DELIBERATELY preserved on the emitted spans: the wrap
/// cache's projection ([`lines_to_plain_atomic`]) reads it back off these rows so
/// the cache keeps the same math runs intact (the wrapped output is re-wrapped by
/// the cache only to COUNT rows, and re-wrap is idempotent only if math stays
/// atomic). It is stripped at the draw boundary by [`strip_atomic_line`]. An empty
/// hard line yields one empty row.
fn wrap_styled_hard_line(
    gph: &[(&str, Style)],
    width: u16,
    rows: &mut Vec<Line<'static>>,
) {
    let plain: String = gph.iter().map(|(g, _)| *g).collect();
    let ranges = atomic_byte_ranges(gph);
    let segments = crate::render::measure::wrap_line_segments_atomic(
        &plain,
        width.max(1) as usize,
        &ranges,
    );

    let mut pos = 0usize; // index into gph
    for seg in &segments {
        // The number of graphemes this segment's (trimmed) text contains.
        let seg_graphemes: Vec<&str> = seg.text.graphemes(true).collect();
        // Skip any leading style-neutral spaces the wrap trimmed at this boundary.
        while pos < gph.len()
            && (seg_graphemes.first().map(|f| *f != gph[pos].0).unwrap_or(true))
            && gph[pos].0 == " "
        {
            pos += 1;
        }
        // Consume the segment's graphemes, coalescing consecutive same-style runs
        // into single spans (fewer spans → cheaper render).
        let mut spans: Vec<Span<'static>> = Vec::new();
        let mut consumed = 0usize;
        while consumed < seg_graphemes.len() && pos < gph.len() {
            let (g, style) = gph[pos];
            // Defensive: if the stream desyncs from the segment text, still emit
            // the segment text so nothing is lost (never panic / never corrupt).
            let mut run = String::new();
            let run_style = style;
            while consumed < seg_graphemes.len()
                && pos < gph.len()
                && gph[pos].1 == run_style
            {
                run.push_str(gph[pos].0);
                pos += 1;
                consumed += 1;
                let _ = g;
            }
            spans.push(Span::styled(run, run_style));
        }
        // If we couldn't reconstruct from the stream (desync), fall back to the
        // plain segment text in the default style.
        if spans.is_empty() && !seg.text.is_empty() {
            spans.push(Span::raw(seg.text.clone()));
        }
        rows.push(Line::from(spans));
    }
}

/// Strip the out-of-band [`render::ATOMIC`] sentinel from every span of a rendered
/// row so it never paints as a real `RAPID_BLINK`. Applied at the DRAW boundary
/// (the transcript widget), AFTER the wrap-cache projection has read the sentinel
/// off the wrapped rows. PURE.
pub(crate) fn strip_atomic_line(line: &Line<'static>) -> Line<'static> {
    if !line
        .spans
        .iter()
        .any(|s| s.style.add_modifier.contains(render::ATOMIC))
    {
        return line.clone();
    }
    let spans: Vec<Span<'static>> = line
        .spans
        .iter()
        .map(|s| {
            let mut style = s.style;
            style.add_modifier.remove(render::ATOMIC);
            Span::styled(s.content.clone(), style)
        })
        .collect();
    Line::from(spans)
}

/// Collect the byte ranges (in the concatenated `plain` of `gph`) of maximal runs
/// of consecutive graphemes whose style carries [`render::ATOMIC`] — the rendered
/// inline-math glyph runs the wrapper must keep intact. Ranges are sorted,
/// non-overlapping, and aligned to grapheme boundaries by construction. PURE.
fn atomic_byte_ranges(gph: &[(&str, Style)]) -> Vec<std::ops::Range<usize>> {
    let mut ranges: Vec<std::ops::Range<usize>> = Vec::new();
    let mut byte = 0usize;
    let mut run_start: Option<usize> = None;
    for (g, style) in gph {
        let is_atomic = style.add_modifier.contains(render::ATOMIC);
        match (is_atomic, run_start) {
            (true, None) => run_start = Some(byte),
            (false, Some(start)) => {
                ranges.push(start..byte);
                run_start = None;
            }
            _ => {}
        }
        byte += g.len();
    }
    if let Some(start) = run_start {
        ranges.push(start..byte);
    }
    ranges
}

/// Render an assistant block's source to styled lines AND soft-wrap them to
/// `width`, returning the styled visual rows in order. The row count equals the
/// plain wrap cache's count for the same `(source, width)` (the markdown-plain
/// projection wrapped by the same algorithm), so the transcript can substitute
/// these styled rows for the assistant block's plain rows without breaking scroll
/// math (P1). Pure over `(source, theme, width)`.
#[allow(dead_code)] // un-cockpit wrapped render (exercised by the row-count test).
pub fn render_assistant_wrapped(source: &str, theme: &Theme, width: u16) -> Vec<Line<'static>> {
    let logical = render_assistant(source, theme);
    let mut out: Vec<Line<'static>> = Vec::new();
    for line in &logical {
        out.extend(wrap_styled_line(line, width));
    }
    if out.is_empty() {
        out.push(Line::default());
    }
    out
}

/// The display width of a styled line in terminal cells (sum of span widths).
#[allow(dead_code)] // diagnostics / future right-alignment; kept for completeness.
pub fn line_width(line: &Line<'static>) -> usize {
    line.spans
        .iter()
        .map(|s| UnicodeWidthStr::width(s.content.as_ref()))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_block_math_to_stacked_layout() {
        let theme = Theme::default_theme();
        let lines = render_assistant("$$\\frac{a}{b}$$", &theme);
        assert_eq!(lines.len(), 3, "display math stacks to a 3-line fraction");
    }

    #[test]
    fn routes_prose_through_markdown() {
        let theme = Theme::default_theme();
        let lines = render_assistant("# Hi\n\nsome **bold** text", &theme);
        let joined: String = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .map(|s| s.content.as_ref())
            .collect();
        assert!(joined.contains("Hi"));
        assert!(joined.contains("bold"));
    }

    #[test]
    fn plain_projection_matches_rendered_text() {
        let theme = Theme::default_theme();
        // The plain projection's line count equals the styled render's line count
        // (so the wrap cache's row accounting is consistent with the draw).
        let styled = render_assistant("a\n\nb\n\nc", &theme);
        let plain = render_assistant_plain("a\n\nb\n\nc", &theme);
        assert_eq!(plain.split('\n').count(), styled.len());
    }

    // The load-bearing P1-with-markdown invariant: the styled, soft-wrapped rows
    // the transcript DRAWS for an assistant block must be EXACTLY as many as the
    // wrap cache (fed the markdown-plain projection) reports for that block — at
    // any width. If these ever diverge, a visible row's `intra` would index the
    // wrong styled line and scroll would drift. We assert equality across several
    // widths and content kinds (paragraph that wraps, table, code block, block
    // math, CJK).
    #[test]
    fn styled_wrap_rowcount_matches_wrap_cache() {
        use crate::render::block::{Block as RB, BlockRole};
        use crate::render::measure::WrapCache;

        let theme = Theme::default_theme();
        let samples = [
            "the quick brown fox jumps over the lazy dog several times in a row",
            "# Heading\n\nSome **bold** and `code` and a [link](https://x.io) here.",
            "| A | B |\n|:--|--:|\n| one | 1 |\n| two | 22 |",
            "```rust\nfn main() {\n    println!(\"hi\");\n}\n```",
            "$$\\frac{a+b}{c}$$",
            // Inline math among prose: the rendered glyph run is ATOMIC, so the
            // styled wrap and the atomic-aware cache must STILL agree (Fix A).
            "prose then $x_1 + x_2 + x_3 + x_4 + x_5 + x_6$ and more prose after it",
            "> quoted line one\n> quoted line two that is rather long and wraps",
            "- item one is quite long and will wrap at narrow widths for sure\n- two",
            "中文段落很长很长很长很长很长很长很长很长很长会换行 and some ascii too",
        ];
        for src in samples {
            for width in [20u16, 40, 80, 120] {
                // What the transcript draws.
                let styled = render_assistant_wrapped(src, &theme, width);
                // What the wrap cache counts — fed the SAME (plain, atomic-ranges)
                // projection the app attaches in `to_render_block`, so inline-math
                // runs stay intact identically on both sides.
                let (plain, ranges) = lines_to_plain_atomic(&render_assistant(src, &theme));
                let rb = RB::finalized(1, BlockRole::Assistant, plain).with_atomic_ranges(ranges);
                let mut cache = WrapCache::new(width);
                cache.sync(std::slice::from_ref(&rb));
                let cache_rows = cache.block_line_count(1);
                assert_eq!(
                    styled.len(),
                    cache_rows,
                    "row count mismatch for src={src:?} width={width}: \
                     styled={} cache={cache_rows}",
                    styled.len()
                );
            }
        }
    }

    #[test]
    fn wrap_styled_preserves_text_per_row() {
        // A styled line wrapped to a width reconstructs (per row) to the same text
        // the plain wrapper would, so styling never loses or duplicates content.
        use crate::render::measure::wrap_line_segments;
        let theme = Theme::default_theme();
        let logical = render_assistant("alpha beta gamma delta epsilon zeta", &theme);
        assert_eq!(logical.len(), 1);
        let rows = wrap_styled_line(&logical[0], 12);
        let plain_rows: Vec<String> = rows
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        let expect: Vec<String> = wrap_line_segments("alpha beta gamma delta epsilon zeta", 12)
            .into_iter()
            .map(|s| s.text)
            .collect();
        assert_eq!(plain_rows, expect);
    }

    /// The cockpit assistant render (folds + chips) keeps the wrap cache's row
    /// accounting EXACT (P1): the styled rows the transcript draws equal the rows
    /// the cache counts from the cockpit-plain projection — for content that
    /// folds a turn AND renders a chip.
    #[test]
    fn cockpit_render_rowcount_matches_plain_projection() {
        use crate::render::block::{Block as RB, BlockRole};
        use crate::render::measure::WrapCache;

        let theme = Theme::default_theme();
        // Two turns (turn 1 folds) + a chip in turn 2 + prose + CJK + INLINE MATH
        // (the atomic glyph run must keep the cockpit projection 1:1 too — Fix A).
        let src = "\
**Turn 1 ...**
<summary>read the config file</summary>
🛠️ Tool: `file_read` path: config.toml
port = 8080
**Turn 2 ...**
<summary>now searching</summary>
🛠️ Tool: `web_search` query: rust ratatui cockpit
the closed form $\\sum_{i=0}^{n} x_i + x_{i+1}$ holds for found 3 results
中文结果也在这里很长很长很长很长很长很长很长很长会换行
done.";
        for width in [24u16, 40, 80, 120] {
            for fold_all in [false, true] {
                let styled = render_assistant_cockpit(src, &theme, fold_all, width);
                // The cache is fed the SAME (plain, atomic-ranges) projection the app
                // attaches for a finalized assistant block in `to_render_block`.
                let (plain, ranges) =
                    lines_to_plain_atomic(&render_assistant_cockpit(src, &theme, fold_all, width));
                let rb = RB::finalized(1, BlockRole::Assistant, plain).with_atomic_ranges(ranges);
                let mut cache = WrapCache::new(width);
                cache.sync(std::slice::from_ref(&rb));
                let cache_rows = cache.block_line_count(1);
                assert_eq!(
                    styled.len(),
                    cache_rows,
                    "cockpit row mismatch width={width} fold_all={fold_all}: \
                     styled={} cache={cache_rows}",
                    styled.len()
                );
            }
        }
    }

    /// Fix E parity (gate 5): the cockpit render UNDER per-node fold overrides keeps
    /// the wrap cache's row accounting EXACT — the styled rows
    /// [`render_assistant_cockpit_full`] draws equal the rows the cache counts from
    /// the SAME (plain, atomic) projection, for EVERY combination of {turn folded /
    /// expanded} × {tool collapsed / expanded}. This is what makes a fold toggle
    /// re-derive the scroll anchor cleanly (no drift): the projection the cache
    /// counts is byte-identical to the lines drawn, override by override.
    #[test]
    fn cockpit_full_rowcount_matches_projection_under_overrides() {
        use crate::render::block::{Block as RB, BlockRole};
        use crate::render::fold::{BlockFolds, NodeId};
        use crate::render::measure::WrapCache;
        use std::collections::HashMap;

        let theme = Theme::default_theme();
        // Two turns; turn 2 carries a LONG tool result (foldable) + inline math + CJK.
        let src = "\
Turn 1 ...
<summary>read the config file</summary>
some prose in the first turn that is long enough to wrap at a narrow width here
Turn 2 ...
<summary>now searching</summary>
🛠️ run({\"q\": \"rust\"})
RESULT line one
RESULT line two
RESULT line three
RESULT line four
RESULT line five
RESULT line six
the closed form $\\sum_{i=0}^{n} x_i$ holds 中文也在这里很长很长很长会换行
done.";
        let block_id = 7u64;
        // Sweep every override combination for turn 1 (fold/expand) and the tool
        // (collapse/expand). `None` ⇒ default policy.
        for turn1 in [None, Some(true), Some(false)] {
            for tool0 in [None, Some(true), Some(false)] {
                let mut map: HashMap<NodeId, bool> = HashMap::new();
                if let Some(v) = turn1 {
                    map.insert(NodeId::Turn { block: block_id, turn: 1 }, v);
                }
                if let Some(v) = tool0 {
                    map.insert(NodeId::Tool { block: block_id, tool: 0 }, v);
                }
                let folds = BlockFolds { block_id, fold_all: false, overrides: Some(&map) };
                for width in [24u16, 40, 80] {
                    // What the transcript draws (the full render's lines).
                    let render = render_assistant_cockpit_full(src, &theme, &folds, width, false);
                    // What the wrap cache counts — the SAME projection `to_render_block`
                    // attaches (lines → plain + atomic ranges).
                    let (plain, ranges) = lines_to_plain_atomic(&render.lines);
                    let rb = RB::finalized(block_id, BlockRole::Assistant, plain)
                        .with_atomic_ranges(ranges);
                    let mut cache = WrapCache::new(width);
                    cache.sync(std::slice::from_ref(&rb));
                    assert_eq!(
                        render.lines.len(),
                        cache.block_line_count(block_id),
                        "row parity broke under overrides turn1={turn1:?} tool0={tool0:?} width={width}"
                    );
                    // Every node-hit range stays within the drawn rows (no dangling
                    // range that a click would mis-resolve).
                    for (range, _) in &render.node_hits {
                        assert!(
                            *range.end() < render.lines.len(),
                            "node-hit range {range:?} exceeds {} rows", render.lines.len()
                        );
                    }
                }
            }
        }
    }

    /// Regression for the off-by-one that the fix targets: a span carrying an
    /// embedded `\n` (raw/inline HTML like `<summary>…</summary>\n`) must NOT make
    /// the styled draw and the wrap cache disagree. We assert the cockpit
    /// styled-vs-cache row parity AND the un-cockpit `render_assistant_wrapped`
    /// parity across a battery of `\n`-leaking inputs at every width — the "ALL
    /// inputs" guarantee, not just the one canned sample.
    #[test]
    fn embedded_newline_in_span_keeps_rowcount_parity() {
        use crate::render::block::{Block as RB, BlockRole};
        use crate::render::measure::WrapCache;

        let theme = Theme::default_theme();
        let samples = [
            // Bare HTML tags (pulldown-cmark hands these back with a trailing \n).
            "<summary>collapse me</summary>",
            "prose then <summary>a longer summary that wraps for sure</summary> tail",
            // Multiple HTML lines in one block.
            "<summary>one</summary>\n<details>two</details>\n<thinking>three</thinking>",
            // HTML interleaved with markdown structure + CJK.
            "**Turn 1 ...**\n<summary>读取配置文件并检查端口设置很长很长很长</summary>\nbody",
            // A raw HTML block followed by prose.
            "<div class=\"x\">raw block</div>\n\nafter the html paragraph here",
            // HTML (embedded \n) AND inline math (atomic glyph run) together — the
            // most adversarial mix for the Fix-A wrap parity.
            "<summary>note</summary>\nthe form $\\sum_{i=0}^{n} x_i + x_{i+1}$ is below",
        ];
        for src in samples {
            for width in [12u16, 20, 24, 40, 80] {
                // Un-cockpit path. Feed the cache the SAME (plain, atomic-ranges)
                // projection the app attaches (`to_render_block`) so inline-math
                // runs are kept intact identically on both sides.
                let styled = render_assistant_wrapped(src, &theme, width);
                let (plain, ranges) = lines_to_plain_atomic(&render_assistant(src, &theme));
                let rb = RB::finalized(1, BlockRole::Assistant, plain).with_atomic_ranges(ranges);
                let mut cache = WrapCache::new(width);
                cache.sync(std::slice::from_ref(&rb));
                assert_eq!(
                    styled.len(),
                    cache.block_line_count(1),
                    "un-cockpit row mismatch src={src:?} width={width}"
                );
                // No emitted span may contain an embedded newline (the invariant
                // the fix establishes at the markdown source).
                for line in render_assistant(src, &theme) {
                    for span in &line.spans {
                        assert!(
                            !span.content.contains('\n'),
                            "span carries an embedded \\n (breaks row parity): \
                             src={src:?} span={:?}",
                            span.content
                        );
                    }
                }
                // Cockpit path (fold + chip aware) too.
                for fold_all in [false, true] {
                    let cstyled = render_assistant_cockpit(src, &theme, fold_all, width);
                    let (cplain, cranges) = lines_to_plain_atomic(&render_assistant_cockpit(
                        src, &theme, fold_all, width,
                    ));
                    let crb =
                        RB::finalized(2, BlockRole::Assistant, cplain).with_atomic_ranges(cranges);
                    let mut ccache = WrapCache::new(width);
                    ccache.sync(std::slice::from_ref(&crb));
                    assert_eq!(
                        cstyled.len(),
                        ccache.block_line_count(2),
                        "cockpit row mismatch src={src:?} width={width} fold_all={fold_all}"
                    );
                }
            }
        }
    }

    #[test]
    fn cockpit_folds_completed_turn_to_one_line() {
        let theme = Theme::default_theme();
        // BARE GA turn markers (tui_v4 non-verbose) + a summary per turn.
        let src = "Turn 1 ...\n<summary>first thing</summary>\nbody one\nTurn 2 ...\n<summary>second</summary>\nbody two";
        let lines = render_assistant_cockpit(src, &theme, false, 80);
        // Turn 1 collapses to a single `▸ first thing` line (its summary as title).
        let joined: Vec<String> = lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        assert!(joined.iter().any(|l| l == "▸ first thing"));
        // Turn 2 (the last) stays expanded — its body is present.
        assert!(joined.iter().any(|l| l.contains("body two")));
    }

    /// THE deliverable test (§1): `<summary>…</summary>` tags are HIDDEN — never
    /// rendered raw. The inner text surfaces as a dim breadcrumb (`↳ …`); the
    /// literal `<summary>`/`</summary>` strings appear NOWHERE in the output.
    #[test]
    fn summary_tags_hidden() {
        let theme = Theme::default_theme();
        let src = "Turn 1 ...\n<summary>用户打招呼，扫描标签页</summary>\nHello there.";
        let plain = render_assistant_cockpit_plain(src, &theme, false, 80);
        assert!(!plain.contains("<summary>"), "the opening tag must never render: {plain:?}");
        assert!(!plain.contains("</summary>"), "the closing tag must never render: {plain:?}");
        // The inner text is preserved as a breadcrumb (the `↳` marker is the tell).
        assert!(plain.contains("用户打招呼，扫描标签页"), "summary inner is kept as a breadcrumb");
        assert!(plain.contains('↳'), "the breadcrumb marker is present");
        // The prose after the summary still renders.
        assert!(plain.contains("Hello there."));
    }

    /// THE deliverable test (§1): a `Turn N ...` boundary line is NOT rendered as
    /// text — it becomes blank-line spacing. The literal `Turn 1 ...` / `Turn 2 ...`
    /// strings never appear in the output, in EITHER the bare or bold form.
    #[test]
    fn turn_marker_not_rendered() {
        let theme = Theme::default_theme();
        // Two turns (so a fold + an expanded turn both exercise the stripping) with
        // BARE markers; turn 1 has no summary (so nothing else could echo "Turn 1").
        let src = "Turn 1 ...\nfirst body\nTurn 2 ...\nsecond body";
        let plain = render_assistant_cockpit_plain(src, &theme, false, 80);
        assert!(!plain.contains("Turn 1"), "bare turn marker text must not render: {plain:?}");
        assert!(!plain.contains("Turn 2"), "bare turn marker text must not render: {plain:?}");
        assert!(!plain.contains("..."), "the `Turn N ...` ellipsis must not render: {plain:?}");
        // The turn BODIES still render.
        assert!(plain.contains("first body"));
        assert!(plain.contains("second body"));

        // The bold/verbose form is stripped too.
        let bold = "**Turn 1 ...**\nalpha\n**Turn 2 ...**\nbeta";
        let bold_plain = render_assistant_cockpit_plain(bold, &theme, false, 80);
        assert!(!bold_plain.contains("Turn 1") && !bold_plain.contains("Turn 2"));
        assert!(bold_plain.contains("alpha") && bold_plain.contains("beta"));
    }

    /// REGRESSION against REAL captured GA output (the `code_run` multi-turn trace
    /// recorded from a live `ga_bridge.py` session): bare `Turn N ...` boundaries,
    /// `<summary>…</summary>`, and compact `🛠️ code_run({…json…})` with nested braces
    /// in the args. The renderer must hide every raw marker and surface ⏺ bullets +
    /// breadcrumbs — this is the exact format that motivated the redesign.
    #[test]
    fn real_ga_code_run_trace_renders_clean() {
        let theme = Theme::default_theme();
        // Verbatim from a live capture (see report): two tool turns + a final turn.
        let src = "\n\nTurn 1 ...\n\n<summary>准备执行echo hi</summary>\n🛠️ code_run({\"cwd\": \"D:\\\\GenericAgent\\\\temp\", \"inline_eval\": false, \"script\": \"echo hi\"})\n\n\n\n\nTurn 2 ...\n\n<summary>上次超时，改用PowerShell直跑</summary>\n🛠️ code_run({\"script\": \"echo hi\", \"type\": \"powershell\"})\n\n\n\n\nTurn 3 ...\n\nOutput:\n\n```text\nhi\n```\n";
        let plain = render_assistant_cockpit_plain(src, &theme, false, 100);
        // No raw markers survive.
        assert!(!plain.contains("<summary>") && !plain.contains("</summary>"));
        assert!(!plain.contains("🛠️"), "raw tool marker hidden: {plain:?}");
        for tl in ["Turn 1", "Turn 2", "Turn 3"] {
            assert!(!plain.contains(tl), "turn marker text {tl} hidden: {plain:?}");
        }
        // With 3 turns, the two COMPLETED turns fold to `▸ <summary>` one-liners
        // (their summaries as the fold title — §1 "or fold it"); the final turn
        // stays expanded.
        assert!(plain.contains("▸ 准备执行echo hi"), "completed turn 1 folds to its summary: {plain:?}");
        assert!(plain.contains("▸ 上次超时，改用PowerShell直跑"), "completed turn 2 folds: {plain:?}");
        // The final (expanded) turn's output renders.
        assert!(plain.contains("hi"));

        // EXPANDED-tool check: a SINGLE-turn version (the active turn) shows the
        // summary as a dim `↳` breadcrumb + the compact tool as a ⏺ bullet with its
        // nested-brace JSON args parsed whole.
        let one = "Turn 1 ...\n<summary>准备执行echo hi</summary>\n🛠️ code_run({\"script\": \"echo hi\", \"type\": \"powershell\"})\n[Info] ok";
        let oplain = render_assistant_cockpit_plain(one, &theme, false, 100);
        assert!(oplain.contains("↳ 准备执行echo hi"), "active-turn summary as breadcrumb: {oplain:?}");
        assert!(oplain.contains("⏺ code_run"), "compact tool as ⏺ bullet: {oplain:?}");
        assert!(oplain.contains("powershell"), "nested-brace JSON args parsed whole");
        assert!(oplain.contains("  [Info] ok"), "result indented 2 cols");

        // And the streaming path (an in-flight final turn) never panics / corrupts.
        let _ = render_assistant_cockpit_streaming(src, &theme, false, 100, true);
    }

    /// Fix D acceptance (Q8 / C1-F6 tail leak): a `Turn N ...` marker that lands in
    /// the VOLATILE streaming tail (held back, not yet committed) is STRIPPED — it
    /// never shows as literal "Turn N" dim text in the active region — and the tail
    /// streams RENDERED through `render_turn_body` (the prose after the marker, the
    /// hoisted `<summary>` breadcrumb), not as a raw dim dump. We pin the tail to a
    /// fresh turn by holding it back with an unclosed `<summary>`.
    #[test]
    fn tail_strips_turn_marker() {
        let theme = Theme::default_theme();
        // `safe_commit_pos` commits up to the last `\n\n` BEFORE the unclosed
        // `<summary>`, so `Turn 7 ...\n<summary>scanning the page` is the volatile
        // tail. The OLD raw-dim tail would echo the literal "Turn 7 ..." line.
        let src = "committed stable prose\n\nTurn 7 ...\n<summary>scanning the page";
        let lines = render_assistant_cockpit_streaming(src, &theme, false, 60, true);
        let plain = lines_to_plain(&lines);
        assert!(
            !plain.contains("Turn 7"),
            "a Turn marker in the volatile tail must be stripped: {plain:?}"
        );
        assert!(
            !plain.contains("..."),
            "the `Turn N ...` ellipsis must not leak from the tail: {plain:?}"
        );
        // The committed head still rendered; the in-flight summary inner streams as
        // a breadcrumb (the tail went through `render_turn_body`, not a raw dump).
        assert!(plain.contains("committed stable prose"), "head still renders: {plain:?}");
        assert!(plain.contains("scanning the page"), "tail streams rendered: {plain:?}");

        // A bold `**Turn N ...**` marker that arrives whole in the tail is stripped
        // too (held back by a trailing unclosed inline `$` formula after it).
        let bold = "committed prose line\n\n**Turn 9 ...**\nmid sentence $x_1 +";
        let bplain = lines_to_plain(&render_assistant_cockpit_streaming(bold, &theme, false, 60, true));
        assert!(!bplain.contains("Turn 9"), "bold tail marker stripped: {bplain:?}");
        assert!(bplain.contains("mid sentence"), "tail prose before the open `$` renders: {bplain:?}");
        assert!(!bplain.contains('$'), "the in-flight `$` formula is held, not painted raw: {bplain:?}");
    }

    /// Fix D + C1-F3 (gate 5): a half-built block math `$$…` seeded in the VOLATILE
    /// tail must NOT paint raw `$$` in the active region — the markdown walker emits
    /// literal dollars for an unclosed delimiter, so the tail renderer trims the
    /// in-flight formula (keeping the stable prose before it). We check both an
    /// inline-prose `$$` opener and a lone-line `$$` opener, plus an open single `$`.
    #[test]
    fn tail_holds_back_half_built_math() {
        let theme = Theme::default_theme();
        let cases = [
            // (source with a held-back open formula, the stable text that must show)
            "stable prose line here\n\nthe identity $$\\frac{a}{b",
            "stable prose line here\n\n$$\\sum_{i=0}^{n}",
            "stable prose line here\n\ninline $E = mc^",
        ];
        for src in cases {
            let plain = lines_to_plain(&render_assistant_cockpit_streaming(src, &theme, false, 60, true));
            assert!(
                !plain.contains("$$"),
                "half-built block math leaked raw `$$` into the tail: src={src:?} -> {plain:?}"
            );
            assert!(
                !plain.contains('$'),
                "an in-flight `$` delimiter leaked into the tail: src={src:?} -> {plain:?}"
            );
            assert!(
                plain.contains("stable prose line here"),
                "the stable head must still render: src={src:?} -> {plain:?}"
            );
        }

        // A genuine `$5.00` price in the tail is NOT a math opener — it stays.
        let price = "ledger entry below\n\nthe cost is $5.00 today";
        let pplain = lines_to_plain(&render_assistant_cockpit_streaming(price, &theme, false, 60, true));
        assert!(pplain.contains("$5.00"), "a currency dollar must not be trimmed: {pplain:?}");
    }

    /// The compact `🛠️ name(args)` tool call renders as a CC `⏺` bullet + the tool
    /// name + a dim one-line args + the 2-col-indented `[Info]` result — and NOT as
    /// a heavy box, and the raw `🛠️` marker never survives to the output (§2.3).
    #[test]
    fn compact_tool_call_renders_as_bullet_not_box() {
        let theme = Theme::default_theme();
        let src = "Turn 1 ...\n🛠️ web_scan({\"tabs_only\": true})\n[Info] 3 tabs scanned · ok";
        let plain = render_assistant_cockpit_plain(src, &theme, false, 80);
        assert!(plain.contains("⏺ web_scan"), "CC bullet + tool name: {plain:?}");
        assert!(plain.contains("{\"tabs_only\": true}"), "args shown as a dim one-liner");
        assert!(plain.contains("  [Info] 3 tabs scanned"), "result indented 2 cols, prefix kept");
        // No raw tool marker, no box glyphs.
        assert!(!plain.contains("🛠️"), "the raw tool marker must not render");
        assert!(!plain.contains('╭') && !plain.contains('│') && !plain.contains('╰'), "no box");
    }

    /// Fix A acceptance (Q3, the #1 latex bug): a rendered inline-math glyph run is
    /// ATOMIC — soft-wrap at a NARROW width keeps the whole run on ONE visual row
    /// (never split mid-formula, never an internal space dropped), AND the styled
    /// draw stays row-count-identical to the wrap cache fed the SAME atomic ranges
    /// (the load-bearing P1 parity, now WITH inline math present). We exercise the
    /// exact projection the app uses (`lines_to_plain_atomic` → `with_atomic_ranges`).
    #[test]
    fn inline_math_not_split_by_narrow_wrap() {
        use crate::render::block::{Block as RB, BlockRole};
        use crate::render::measure::WrapCache;

        let theme = Theme::default_theme();
        // (source, the rendered math glyph run that MUST stay intact). The runs are
        // wider than width 20 (spaced sum) or hard-break-prone (the frac) — a naive
        // wrapper would shatter them; atomic keeps each whole on a single row.
        let cases = [
            (
                "the value $x_1 + x_2 + x_3 + x_4 + x_5 + x_6$ scales linearly here",
                "x₁ + x₂ + x₃ + x₄ + x₅ + x₆",
            ),
            (
                "sum $\\sum_{i=0}^{n} x_i$ over the range of indices we care about",
                "∑ᵢ₌₀ⁿ xᵢ",
            ),
            (
                "ratio $\\frac{a+b+c+d}{x+y+z}$ stays a single token even when narrow",
                "(a+b+c+d)/(x+y+z)",
            ),
            (
                "greek $\\alpha + \\beta + \\gamma + \\delta + \\epsilon$ list of letters",
                "α + β + γ + δ + ε",
            ),
        ];

        for (src, glyphs) in cases {
            // The rendered math actually appears (math.rs converted it, not raw `$`).
            let logical_plain = render_assistant_plain(src, &theme);
            assert!(
                logical_plain.contains(glyphs),
                "math fixture must render to {glyphs:?}: {logical_plain:?}"
            );
            assert!(!logical_plain.contains('$'), "no literal $ leaks: {logical_plain:?}");

            for width in [10u16, 14, 18, 20, 24] {
                let styled = render_assistant_wrapped(src, &theme, width);
                // (1) INTACT: the whole glyph run sits inside ONE visual row's text —
                //     never split across rows, never an interior space dropped.
                let row_texts: Vec<String> = styled
                    .iter()
                    .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
                    .collect();
                assert!(
                    row_texts.iter().any(|r| r.contains(glyphs)),
                    "inline math split by wrap at width={width}: run {glyphs:?} not whole on any row: {row_texts:?}"
                );

                // (2) PARITY: feed the cache the SAME atomic ranges the app attaches
                //     in `to_render_block`, then assert equal row counts. Without the
                //     ranges (or with a non-atomic wrapper) these diverge → drift.
                let (plain, ranges) = {
                    let lines = render_assistant(src, &theme);
                    lines_to_plain_atomic(&lines)
                };
                let rb = RB::finalized(1, BlockRole::Assistant, plain).with_atomic_ranges(ranges);
                let mut cache = WrapCache::new(width);
                cache.sync(std::slice::from_ref(&rb));
                assert_eq!(
                    styled.len(),
                    cache.block_line_count(1),
                    "row-count parity broke WITH inline math: src={src:?} width={width} \
                     styled={} cache={}",
                    styled.len(),
                    cache.block_line_count(1),
                );
            }
        }
    }

    /// The ATOMIC sentinel ([`render::ATOMIC`]) is a WRAP HINT only: it tags inline
    /// math during wrapping but is STRIPPED from every emitted span so it never
    /// paints as a real `RAPID_BLINK`. No drawn span may carry it.
    /// The ATOMIC sentinel ([`render::ATOMIC`]) survives ON the wrapped rows (so the
    /// wrap-cache projection [`lines_to_plain_atomic`] can read it back and keep the
    /// same math runs intact), but [`strip_atomic_line`] — applied at the DRAW
    /// boundary — removes it so it never paints as a real `RAPID_BLINK`.
    #[test]
    fn atomic_sentinel_survives_wrap_and_is_stripped_at_draw() {
        let theme = Theme::default_theme();
        let src = "energy $E = mc^2$ and a sum $\\sum_{i=0}^{n} x_i$ here";
        for width in [12u16, 20, 80] {
            let rows = render_assistant_wrapped(src, &theme, width);
            // The wrapped rows DO carry the sentinel on the rendered math spans (the
            // projection depends on it).
            let any_atomic = rows
                .iter()
                .flat_map(|l| l.spans.iter())
                .any(|s| s.style.add_modifier.contains(render::ATOMIC));
            assert!(
                any_atomic,
                "math wrapped at width={width} must keep an ATOMIC span for the projection"
            );
            // After the draw-boundary strip, NO span carries it.
            for line in &rows {
                let drawn = strip_atomic_line(line);
                for span in &drawn.spans {
                    assert!(
                        !span.style.add_modifier.contains(render::ATOMIC),
                        "a drawn span leaked the ATOMIC sentinel at width={width}: {:?}",
                        span.content
                    );
                }
            }
        }
    }
}
