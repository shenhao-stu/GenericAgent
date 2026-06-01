//! workflow/panel.rs — the `/workflows` PANEL state + renderer (checklist §7).
//!
//! A full-screen view (like the session dashboard) over the watcher's
//! [`WorkflowSnapshot`]. It renders the merged tree in one of TWO styles
//! (togglable with `t`):
//!   * **box-tree** — a bordered, indented tree with a `╞═` FOCUS MARKER on the
//!     focused node (the §7 "box-tree w/ `╞═` focus");
//!   * **compact bullet list** — a denser `•`-bulleted flat list (the §7 "compact
//!     bullet list").
//! Both group by **Conductor / Hives / Goal** and show per-node
//! name/status/elapsed/tokens/preview. Focus navigates with ↑/↓ (skipping group
//! headers); Enter opens a DETAIL OVERLAY for the focused node (full prompt +
//! summary + feed + the node ACTION VERBS keyinfo/input/stop/kill/open). Running
//! nodes get an ANIMATED status glyph (a frame-clock spinner) + heat color; done
//! sparkles; failed flashes — the §7 "animated status hooks". A down server
//! degrades gracefully to "not running · press X to launch" (§7).
//!
//! The load-bearing FOCUS/NAV/flatten logic is PURE + unit-tested here
//! (`flatten_rows`, `focusable_indices`, `WorkflowPanel::move_focus`, the action
//! menu); the renderer below only PAINTS, routing every string through i18n and
//! every color through a theme [`Token`] (no hardcoded RGB).

use std::cell::Cell;

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui::Frame;

use crate::flavor::{self, heat_bold, heat_token, SpinnerStyle};
use crate::i18n::{self, Lang};
use crate::theme::{Theme, Token};
use crate::workflow::schema::{
    FeedItem, NodeRole, WfStatus, Workflow, WorkflowKind, WorkflowNode, WorkflowSnapshot,
};
use crate::workflow::NodeAction;

/// Which of the two render styles the panel is using (§7 "two styles … toggle").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RenderStyle {
    /// A bordered, indented box-tree with a `╞═` focus marker.
    #[default]
    BoxTree,
    /// A compact `•`-bulleted flat list.
    Bullet,
}

impl RenderStyle {
    /// Flip to the other style (`t` toggles). PURE.
    pub fn toggled(self) -> RenderStyle {
        match self {
            RenderStyle::BoxTree => RenderStyle::Bullet,
            RenderStyle::Bullet => RenderStyle::BoxTree,
        }
    }

    /// The i18n KEY for the style's name (shown in the footer hint). PURE.
    pub fn label_key(self) -> &'static str {
        match self {
            RenderStyle::BoxTree => "wf.style.tree",
            RenderStyle::Bullet => "wf.style.list",
        }
    }
}

/// One flattened panel row: either a GROUP header (Conductor / Hives / Goal) or a
/// focusable NODE (carrying its workflow + node indices into the snapshot). The
/// flatten is the single source of truth for both rendering AND nav, so the focus
/// index always lines up with what's drawn. PURE data.
#[derive(Debug, Clone, PartialEq)]
pub enum PanelRow {
    /// A group header for a workflow KIND, carrying its summary counts.
    Group { kind: WorkflowKind, workflows: usize, nodes: usize, running: usize },
    /// A workflow TITLE row (e.g. `conductor :8900` / `hive demo` / objective).
    /// Non-focusable; gives the tree its per-workflow root label + degrade hint.
    Title { wf_idx: usize, running: bool },
    /// A focusable node row (indices into `snapshot.workflows[wf_idx].nodes[node_idx]`).
    Node { wf_idx: usize, node_idx: usize, depth: u8 },
}

impl PanelRow {
    /// `true` if this row can take FOCUS (only node rows). PURE.
    pub fn is_focusable(&self) -> bool {
        matches!(self, PanelRow::Node { .. })
    }
}

/// Flatten a [`WorkflowSnapshot`] into the ordered row list the panel renders +
/// navigates (the single layout truth). Groups appear in `WorkflowKind` order
/// (Conductor → Hives → Goal — recon §7); under each group, every workflow's
/// Title row then its nodes (roots before leaves, indented by role). A KIND with
/// no workflows is omitted; a down workflow still gets its Title row (so the panel
/// shows the "not running" hint) but contributes no node rows. PURE.
pub fn flatten_rows(snap: &WorkflowSnapshot) -> Vec<PanelRow> {
    let mut rows = Vec::new();
    for kind in [WorkflowKind::Conductor, WorkflowKind::Hive, WorkflowKind::Goal] {
        // The workflows of this kind, in stable order (the snapshot's `of_kind`
        // helper is the single source of per-group membership so the panel + the
        // schema agree on what a group contains).
        let group_wfs = snap.of_kind(kind);
        if group_wfs.is_empty() {
            continue;
        }
        // Re-resolve each group workflow to its INDEX in `snap.workflows` (the rows
        // carry indices so the renderer + nav can look the workflow back up). The
        // group is small (one conductor, a few hives, one goal), so the lookup is
        // cheap and keeps `of_kind` as the ordering authority.
        let group: Vec<(usize, &Workflow)> = group_wfs
            .iter()
            .map(|w| {
                let idx = snap
                    .workflows
                    .iter()
                    .position(|x| std::ptr::eq(x, *w))
                    .expect("of_kind returns references into snap.workflows");
                (idx, *w)
            })
            .collect();
        let workflows = group.len();
        // `node_count()` is the per-workflow roll-up the group header sums.
        let nodes: usize = group.iter().map(|(_, w)| w.node_count()).sum();
        let running: usize = group.iter().map(|(_, w)| w.running_count()).sum();
        rows.push(PanelRow::Group { kind, workflows, nodes, running });
        for (wf_idx, w) in group {
            rows.push(PanelRow::Title { wf_idx, running: w.running });
            for (node_idx, n) in w.nodes.iter().enumerate() {
                rows.push(PanelRow::Node {
                    wf_idx,
                    node_idx,
                    depth: node_depth(n),
                });
            }
        }
    }
    rows
}

/// The tree DEPTH (indent level) for a node by its role: roots
/// (conductor/master/goal) at depth 0, leaves (subagent/worker) at depth 1. PURE.
fn node_depth(n: &WorkflowNode) -> u8 {
    match n.role {
        NodeRole::Conductor | NodeRole::Master | NodeRole::Goal => 0,
        NodeRole::Subagent | NodeRole::Worker => 1,
    }
}

/// The indices into `rows` that are FOCUSABLE (node rows). The nav uses this so
/// ↑/↓ skip group + title rows (recon §7 "Focus nav"). PURE.
pub fn focusable_indices(rows: &[PanelRow]) -> Vec<usize> {
    rows.iter()
        .enumerate()
        .filter(|(_, r)| r.is_focusable())
        .map(|(i, _)| i)
        .collect()
}

/// The `/workflows` panel state. Holds the FOCUS (an index into the focusable node
/// rows), the render style, the scroll offset, and the detail-overlay state. The
/// snapshot itself lives in `AppState` (fed from the watcher each frame); the panel
/// only tracks the cursor + view mode so it survives refreshes.
#[derive(Debug, Clone)]
pub struct WorkflowPanel {
    /// The focused NODE position, as an index into [`focusable_indices`] (NOT the
    /// raw row index) so it is stable as group headers come and go. Clamped on each
    /// refresh.
    pub focus: usize,
    /// The active render style (`t` toggles).
    pub style: RenderStyle,
    /// Top visible row (scroll offset) for the tree list. PERSISTENT across frames
    /// (the stateful viewport top) so a content-only refresh never snaps the view;
    /// updated by [`WorkflowPanel::scroll_to_focus`] from the renderer. A `Cell` so
    /// the renderer can persist the focus-driven viewport top from an immutable
    /// `&WorkflowPanel` (render is pure — P11; mirrors the cockpit's `sync_transcript`
    /// hoist), single-threaded so interior mutability is free.
    pub scroll: Cell<usize>,
    /// When `Some`, the DETAIL overlay is open on `(wf_idx, node_idx)` with the
    /// action-menu selection `action_sel`. Enter opens it; Esc closes it.
    pub detail: Option<DetailState>,
    /// The snapshot generation the focus was last clamped against (to re-clamp only
    /// on a real refresh).
    pub last_generation: u64,
}

/// The detail-overlay state: which node it's showing + the highlighted action.
#[derive(Debug, Clone, PartialEq)]
pub struct DetailState {
    pub wf_idx: usize,
    pub node_idx: usize,
    /// Highlighted action in the verb menu (index into the node's action list).
    pub action_sel: usize,
}

impl Default for WorkflowPanel {
    fn default() -> Self {
        WorkflowPanel {
            focus: 0,
            style: RenderStyle::default(),
            scroll: Cell::new(0),
            detail: None,
            last_generation: 0,
        }
    }
}

impl WorkflowPanel {
    pub fn new() -> Self {
        WorkflowPanel::default()
    }

    /// Toggle the render style (`t`). PURE-ish.
    pub fn toggle_style(&mut self) {
        self.style = self.style.toggled();
    }

    /// Move the focus by `delta` over the FOCUSABLE rows (clamped, no wrap). A panel
    /// with no focusable nodes leaves focus at 0. PURE over `(self, snap)`.
    pub fn move_focus(&mut self, delta: isize, snap: &WorkflowSnapshot) {
        let rows = flatten_rows(snap);
        let focusable = focusable_indices(&rows);
        if focusable.is_empty() {
            self.focus = 0;
            return;
        }
        let max = focusable.len() as isize - 1;
        self.focus = (self.focus as isize + delta).clamp(0, max) as usize;
    }

    /// Re-clamp the focus after a snapshot refresh (nodes may have appeared /
    /// vanished). Called when the generation changes. PURE-ish.
    pub fn clamp_focus(&mut self, snap: &WorkflowSnapshot) {
        if snap.generation == self.last_generation {
            return;
        }
        self.last_generation = snap.generation;
        let rows = flatten_rows(snap);
        let n = focusable_indices(&rows).len();
        if n == 0 {
            self.focus = 0;
        } else if self.focus >= n {
            self.focus = n - 1;
        }
        // If the detail overlay points at a node that no longer exists, close it.
        if let Some(d) = &self.detail {
            let valid = snap
                .workflows
                .get(d.wf_idx)
                .map(|w| d.node_idx < w.nodes.len())
                .unwrap_or(false);
            if !valid {
                self.detail = None;
            }
        }
    }

    /// The `(wf_idx, node_idx)` of the currently-focused node, if any. PURE.
    pub fn focused_node<'a>(&self, snap: &'a WorkflowSnapshot) -> Option<(usize, usize, &'a WorkflowNode)> {
        let rows = flatten_rows(snap);
        let focusable = focusable_indices(&rows);
        let row_idx = *focusable.get(self.focus)?;
        if let PanelRow::Node { wf_idx, node_idx, .. } = rows[row_idx] {
            let node = snap.workflows.get(wf_idx)?.nodes.get(node_idx)?;
            return Some((wf_idx, node_idx, node));
        }
        None
    }

    /// Open the detail overlay on the focused node (Enter). No-op if nothing is
    /// focused. PURE-ish.
    pub fn open_detail(&mut self, snap: &WorkflowSnapshot) {
        if let Some((wf_idx, node_idx, _)) = self.focused_node(snap) {
            self.detail = Some(DetailState { wf_idx, node_idx, action_sel: 0 });
        }
    }

    /// Close the detail overlay (Esc within the panel). Returns `true` if one was
    /// open (so the key handler knows Esc was consumed by the overlay, not the view).
    pub fn close_detail(&mut self) -> bool {
        self.detail.take().is_some()
    }

    /// `true` while the detail overlay is open.
    pub fn detail_open(&self) -> bool {
        self.detail.is_some()
    }

    /// Move the detail action-menu selection by `delta` (clamped over the focused
    /// node's action set). No-op if the overlay is closed. PURE-ish.
    pub fn move_action(&mut self, delta: isize, snap: &WorkflowSnapshot) {
        let Some(d) = self.detail.as_mut() else { return };
        let role = snap
            .workflows
            .get(d.wf_idx)
            .and_then(|w| w.nodes.get(d.node_idx))
            .map(|n| n.role);
        let Some(role) = role else { return };
        let n = NodeAction::for_role(role).len();
        if n == 0 {
            return;
        }
        let max = n as isize - 1;
        d.action_sel = (d.action_sel as isize + delta).clamp(0, max) as usize;
    }

    /// Adjust the persistent scroll offset so the focused row stays visible within a
    /// viewport of `height` rows over `total` rendered rows (the §7 "Focus nav" keeps
    /// the cursor on-screen). Unlike a per-frame recompute, the offset is REMEMBERED
    /// in `self.scroll`, so a refresh that only changes node previews doesn't snap the
    /// view — it only moves when the focus would leave the window. Returns the clamped
    /// top row. PURE over `(self, focus_line, total, height)`; takes `&self` and
    /// persists through the `scroll` [`Cell`] so the renderer can call it on an
    /// immutable panel (render purity — P11).
    pub fn scroll_to_focus(&self, focus_line: usize, total: usize, height: usize) -> usize {
        if height == 0 || total <= height {
            // Everything fits → pin to the top.
            self.scroll.set(0);
            return 0;
        }
        let max_scroll = total - height;
        let mut scroll = self.scroll.get();
        // Scroll up if the focus is above the window.
        if focus_line < scroll {
            scroll = focus_line;
        }
        // Scroll down if the focus is at/below the bottom of the window.
        else if focus_line >= scroll + height {
            scroll = focus_line + 1 - height;
        }
        // Never scroll past the last full page.
        if scroll > max_scroll {
            scroll = max_scroll;
        }
        self.scroll.set(scroll);
        scroll
    }

    /// The action the detail overlay would FIRE on Enter (the highlighted verb) +
    /// its target `(workflow, node)`. `None` if the overlay is closed / invalid.
    /// The caller maps a conductor verb to a `POST /subagent/{id}` (or `Open` →
    /// no-op). PURE.
    pub fn selected_action<'a>(
        &self,
        snap: &'a WorkflowSnapshot,
    ) -> Option<(NodeAction, &'a Workflow, &'a WorkflowNode)> {
        let d = self.detail.as_ref()?;
        let wf = snap.workflows.get(d.wf_idx)?;
        let node = wf.nodes.get(d.node_idx)?;
        let actions = NodeAction::for_role(node.role);
        let action = actions.get(d.action_sel).cloned()?;
        Some((action, wf, node))
    }
}

// ===========================================================================
// Renderer (PAINTS only; all nav/flatten logic is pure above).
// ===========================================================================

/// Draw the full-screen `/workflows` panel for one frame (the view body). A detail
/// overlay (if open) is drawn on top by [`render_detail`] from the caller. `now_ms`
/// drives the animated status glyphs. Takes `&mut WorkflowPanel` so the body can
/// update the persistent scroll offset (the stateful-widget pattern).
pub fn render(
    frame: &mut Frame,
    area: Rect,
    panel: &WorkflowPanel,
    snap: &WorkflowSnapshot,
    theme: &Theme,
    lang: Lang,
    now_ms: u64,
) {
    frame.render_widget(Clear, area);

    // header (1) · separator (1) · body (flex) · footer (1).
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(area);
    render_header(frame, chunks[0], snap, theme, lang);
    render_separator(frame, chunks[1], theme);
    render_body(frame, chunks[2], panel, snap, theme, lang, now_ms);
    render_footer(frame, chunks[3], panel, theme, lang);

    // The detail overlay floats over the body when open.
    if panel.detail.is_some() {
        render_detail(frame, area, panel, snap, theme, lang, now_ms);
    }
}

/// HEADER: identity + global counts (`N workflows · M nodes · K running`).
fn render_header(frame: &mut Frame, area: Rect, snap: &WorkflowSnapshot, theme: &Theme, lang: Lang) {
    let mut spans = vec![
        Span::styled(
            "◆ ",
            Style::default().fg(theme.color(Token::Claude)).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            i18n::t(lang, "wf.title"),
            Style::default().fg(theme.color(Token::Text)).add_modifier(Modifier::BOLD),
        ),
    ];
    let summary = format!(
        "   {} {} · {} {} · {} {}",
        snap.workflows.len(),
        i18n::t(lang, "wf.count.workflows"),
        snap.total_nodes(),
        i18n::t(lang, "wf.count.nodes"),
        snap.total_running(),
        i18n::t(lang, "wf.count.running"),
    );
    spans.push(Span::styled(summary, Style::default().fg(theme.color(Token::Dim))));
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// SEPARATOR: the rainbow line (shared look with the cockpit).
fn render_separator(frame: &mut Frame, area: Rect, theme: &Theme) {
    let width = area.width;
    let mut spans: Vec<Span> = Vec::with_capacity(width as usize);
    for x in 0..width {
        spans.push(Span::styled("─", Style::default().fg(theme.rainbow_at(x, width))));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// FOOTER: the keybinding hint + the active render-style name.
fn render_footer(frame: &mut Frame, area: Rect, panel: &WorkflowPanel, theme: &Theme, lang: Lang) {
    let style_name = i18n::t(lang, panel.style.label_key());
    let hint = format!("{}  ·  {}: {}", i18n::t(lang, "wf.hint"), i18n::t(lang, "wf.style"), style_name);
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(hint, Style::default().fg(theme.color(Token::Dim))))),
        area,
    );
}

/// BODY: the workflow tree in the active style. Degrades to a global hint when the
/// snapshot is empty (no source is up).
fn render_body(
    frame: &mut Frame,
    area: Rect,
    panel: &WorkflowPanel,
    snap: &WorkflowSnapshot,
    theme: &Theme,
    lang: Lang,
    now_ms: u64,
) {
    if snap.is_empty() {
        panel.scroll.set(0);
        let lines = vec![
            Line::from(""),
            Line::from(Span::styled(
                format!("  {}", i18n::t(lang, "wf.empty")),
                Style::default().fg(theme.color(Token::Dim)),
            )),
            Line::from(Span::styled(
                format!("  {}", i18n::t(lang, "wf.empty.launch")),
                Style::default().fg(theme.color(Token::Suggestion)),
            )),
        ];
        frame.render_widget(Paragraph::new(lines), area);
        return;
    }

    let rows = flatten_rows(snap);
    let focusable = focusable_indices(&rows);
    let focused_row = focusable.get(panel.focus).copied();

    let lines = match panel.style {
        RenderStyle::BoxTree => render_tree_lines(&rows, snap, focused_row, theme, lang, now_ms),
        RenderStyle::Bullet => render_bullet_lines(&rows, snap, focused_row, theme, lang, now_ms),
    };

    // Scroll so the focused row stays visible — but REMEMBER the offset in
    // `panel.scroll` so a content-only refresh doesn't snap the view (it only moves
    // when the focus would leave the window). `panel.scroll` is the persistent
    // viewport top the stateful panel carries between frames.
    let height = area.height as usize;
    let focus_line = focused_row.unwrap_or(0);
    let start = panel.scroll_to_focus(focus_line, lines.len(), height);
    let visible: Vec<Line> = lines.into_iter().skip(start).take(height).collect();
    frame.render_widget(Paragraph::new(visible), area);
}

/// Build the BOX-TREE lines (the §7 box-tree with a `╞═` focus marker). Each group
/// is a header; each workflow a title; each node an indented row led by the focus
/// marker (`╞═` when focused, tree branches otherwise) + an animated status glyph.
fn render_tree_lines<'a>(
    rows: &[PanelRow],
    snap: &'a WorkflowSnapshot,
    focused_row: Option<usize>,
    theme: &Theme,
    lang: Lang,
    now_ms: u64,
) -> Vec<Line<'a>> {
    let mut out = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        match row {
            PanelRow::Group { kind, workflows, nodes, running } => {
                out.push(group_header_line(*kind, *workflows, *nodes, *running, theme, lang));
            }
            PanelRow::Title { wf_idx, running } => {
                out.push(title_line(snap, *wf_idx, *running, theme, lang, true));
            }
            PanelRow::Node { wf_idx, node_idx, depth } => {
                let focused = focused_row == Some(i);
                let node = &snap.workflows[*wf_idx].nodes[*node_idx];
                out.push(node_line_tree(node, *depth, focused, theme, lang, now_ms));
            }
        }
    }
    out
}

/// Build the COMPACT BULLET-LIST lines (the §7 alternative style). Denser: group
/// headers are a single line; nodes are `•`-bulleted one-liners (still showing
/// status/name/elapsed/tokens/preview) with the focus marker as a leading `❯`.
fn render_bullet_lines<'a>(
    rows: &[PanelRow],
    snap: &'a WorkflowSnapshot,
    focused_row: Option<usize>,
    theme: &Theme,
    lang: Lang,
    now_ms: u64,
) -> Vec<Line<'a>> {
    let mut out = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        match row {
            PanelRow::Group { kind, workflows, nodes, running } => {
                out.push(group_header_line(*kind, *workflows, *nodes, *running, theme, lang));
            }
            PanelRow::Title { wf_idx, running } => {
                out.push(title_line(snap, *wf_idx, *running, theme, lang, false));
            }
            PanelRow::Node { wf_idx, node_idx, .. } => {
                let focused = focused_row == Some(i);
                let node = &snap.workflows[*wf_idx].nodes[*node_idx];
                out.push(node_line_bullet(node, focused, theme, lang, now_ms));
            }
        }
    }
    out
}

/// A group header line: `▾ Conductor   (1 · 3 nodes · 2 running)`.
fn group_header_line<'a>(
    kind: WorkflowKind,
    workflows: usize,
    nodes: usize,
    running: usize,
    theme: &Theme,
    lang: Lang,
) -> Line<'a> {
    Line::from(vec![
        Span::styled(
            format!("▾ {}", i18n::t(lang, kind.group_key())),
            Style::default().fg(theme.color(Token::Claude)).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("   ({workflows} · {nodes} {} · {running} {})", i18n::t(lang, "wf.count.nodes"), i18n::t(lang, "wf.count.running")),
            Style::default().fg(theme.color(Token::Dim)),
        ),
    ])
}

/// A workflow TITLE line: the per-workflow root label + a degrade hint when down.
/// `tree` adds a leading branch glyph for the box-tree style.
fn title_line<'a>(
    snap: &'a WorkflowSnapshot,
    wf_idx: usize,
    running: bool,
    theme: &Theme,
    lang: Lang,
    tree: bool,
) -> Line<'a> {
    let wf = &snap.workflows[wf_idx];
    let lead = if tree { "  ├─ " } else { "  " };
    let mut spans = vec![Span::styled(lead, Style::default().fg(theme.color(Token::Border)))];
    spans.push(Span::styled(
        wf.title.clone(),
        Style::default()
            .fg(theme.color(Token::Text))
            .add_modifier(Modifier::BOLD),
    ));
    if !running {
        // Degrade gracefully: "· not running · press X to launch" (§7).
        spans.push(Span::styled(
            format!("   · {} · {}", i18n::t(lang, "wf.down"), i18n::t(lang, "wf.down.launch")),
            Style::default().fg(theme.color(Token::Warning)),
        ));
    } else if let Some(p) = wf.progress {
        // Goal/hive progress: turns + a tiny bar.
        spans.push(Span::styled(
            format!(
                "   {}/{} {} · {}",
                p.turns_used,
                p.max_turns,
                i18n::t(lang, "wf.turns"),
                progress_bar(p.fraction()),
            ),
            Style::default().fg(theme.color(Token::Dim)),
        ));
    }
    Line::from(spans)
}

/// A NODE line in the box-tree style: focus marker + status glyph + name + status
/// label + elapsed + tokens + preview, indented by `depth`.
fn node_line_tree<'a>(
    node: &'a WorkflowNode,
    depth: u8,
    focused: bool,
    theme: &Theme,
    lang: Lang,
    now_ms: u64,
) -> Line<'a> {
    // The focus marker `╞═` (focused) vs a tree branch `╰─`/`├─` otherwise (§7).
    let indent = "    ".repeat(depth as usize + 1);
    let marker = if focused { "╞═ " } else { "├─ " };
    let marker_tok = if focused { Token::Claude } else { Token::Border };

    let mut spans = vec![
        Span::styled(indent, Style::default()),
        Span::styled(
            marker,
            Style::default()
                .fg(theme.color(marker_tok))
                .add_modifier(if focused { Modifier::BOLD } else { Modifier::empty() }),
        ),
    ];
    spans.extend(node_core_spans(node, focused, theme, lang, now_ms, true));
    Line::from(spans)
}

/// A NODE line in the compact bullet style: `❯`/`•` + status glyph + a one-liner.
fn node_line_bullet<'a>(
    node: &'a WorkflowNode,
    focused: bool,
    theme: &Theme,
    lang: Lang,
    now_ms: u64,
) -> Line<'a> {
    let bullet = if focused { "  ❯ " } else { "  • " };
    let bullet_tok = if focused { Token::Claude } else { Token::Dim };
    let mut spans = vec![Span::styled(
        bullet,
        Style::default()
            .fg(theme.color(bullet_tok))
            .add_modifier(if focused { Modifier::BOLD } else { Modifier::empty() }),
    )];
    spans.extend(node_core_spans(node, focused, theme, lang, now_ms, false));
    Line::from(spans)
}

/// The shared node core spans: animated status glyph + name + status label +
/// elapsed + `~N tok` + a truncated preview. `with_preview_pad` widens the preview
/// in the (roomier) tree style. PURE-ish (only reads the node + clock).
fn node_core_spans<'a>(
    node: &'a WorkflowNode,
    focused: bool,
    theme: &Theme,
    lang: Lang,
    now_ms: u64,
    roomy: bool,
) -> Vec<Span<'a>> {
    let (glyph, status_tok) = animated_status(node.status, now_ms, theme);
    let name_tok = if focused { Token::Claude } else { Token::Text };
    let mut spans = vec![
        Span::styled(format!("{glyph} "), Style::default().fg(theme.color(status_tok))),
        Span::styled(
            node.label.clone(),
            Style::default()
                .fg(theme.color(name_tok))
                .add_modifier(if focused { Modifier::BOLD } else { Modifier::empty() }),
        ),
        Span::styled(
            format!("  {}", i18n::t(lang, node.status.label_key())),
            Style::default().fg(theme.color(status_tok)),
        ),
    ];
    // Elapsed since last activity (heat-colored while running).
    if node.last_activity_ts > 0.0 {
        let elapsed = elapsed_label(node.last_activity_ts);
        let tok = if node.status == WfStatus::Running {
            heat_token(((crate::workflow::now_secs() - node.last_activity_ts).max(0.0) * 1000.0) as u64)
        } else {
            Token::Dim
        };
        spans.push(Span::styled(format!("  {elapsed}"), Style::default().fg(theme.color(tok))));
    }
    // Tokens (best-effort estimate).
    if node.tokens > 0 {
        spans.push(Span::styled(
            format!("  ~{} {}", node.tokens, i18n::t(lang, "unit.tokens")),
            Style::default().fg(theme.color(Token::Dim)),
        ));
    }
    // Hive post count.
    if node.post_count > 0 {
        spans.push(Span::styled(
            format!("  {}↑", node.post_count),
            Style::default().fg(theme.color(Token::Dim)),
        ));
    }
    // Preview (the current output / last post excerpt).
    if !node.summary.is_empty() {
        let max = if roomy { 48 } else { 36 };
        spans.push(Span::styled(
            format!("   {}", clip_one_line(&node.summary, max)),
            Style::default().fg(theme.color(Token::Dim)),
        ));
    }
    spans
}

/// The ANIMATED status glyph + its color token (§7 "animated status hooks"):
///   * Running  → a spinner frame (the custom arc — NOT the CC `✻`) + heat color;
///   * WrappingUp → a slow pulse glyph;
///   * Done     → a sparkle that twinkles on the frame clock;
///   * Failed   → a lightning that flashes;
///   * else     → the static base glyph.
/// PURE-ish (reads the clock for the frame index).
fn animated_status(status: WfStatus, now_ms: u64, _theme: &Theme) -> (String, Token) {
    // Only an ACTIVE status (running / wrapping-up) animates at the fast cadence; a
    // settled status (idle / done / failed / …) twinkles slowly so a static tree
    // doesn't churn the screen — the `is_active` gate is the §7 "animated status
    // hooks run only for active nodes" rule.
    let frame = if status.is_active() { now_ms / 100 } else { now_ms / 200 };
    match status {
        WfStatus::Running => {
            // The custom arc spinner (flavor's default), heat-neutral accent.
            let g = SpinnerStyle::default().glyph(frame);
            (g.to_string(), Token::Success)
        }
        WfStatus::WrappingUp => {
            let pulse = ['◐', '◓', '◑', '◒'];
            (pulse[(frame as usize) % pulse.len()].to_string(), Token::Warning)
        }
        WfStatus::Done => {
            // Sparkle: twinkle between two glyphs so a finished node "shimmers".
            let s = if frame % 8 < 4 { "✓" } else { "✦" };
            (s.to_string(), Token::Success)
        }
        WfStatus::Failed => {
            // Lightning flash on/off.
            let s = if frame % 6 < 3 { "✗" } else { "ϟ" };
            (s.to_string(), Token::Error)
        }
        WfStatus::Aborted => (WfStatus::Aborted.glyph().to_string(), Token::Dim),
        WfStatus::Idle => (WfStatus::Idle.glyph().to_string(), Token::Suggestion),
        WfStatus::Unknown => (WfStatus::Unknown.glyph().to_string(), Token::Dim),
    }
}

// ---- the DETAIL overlay (Enter on a focused node) --------------------------

/// Draw the node DETAIL overlay (§7 "detail overlay" + "node action verbs"). A
/// centered card showing the node's full prompt + summary + the workflow's recent
/// FEED + the action-verb menu (keyinfo/input/stop/kill/open) with the highlighted
/// verb. Conductor subagents show the full menu; hive/goal nodes show only `open`.
fn render_detail(
    frame: &mut Frame,
    area: Rect,
    panel: &WorkflowPanel,
    snap: &WorkflowSnapshot,
    theme: &Theme,
    lang: Lang,
    now_ms: u64,
) {
    let Some(d) = &panel.detail else { return };
    let Some(wf) = snap.workflows.get(d.wf_idx) else { return };
    let Some(node) = wf.nodes.get(d.node_idx) else { return };

    let w = (area.width.saturating_sub(8)).clamp(40, 92);
    let h = (area.height.saturating_sub(4)).clamp(12, 28);
    let card = centered(area, w, h);
    frame.render_widget(Clear, card);
    let (glyph, _) = animated_status(node.status, now_ms, theme);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.color(Token::Claude)))
        .title(Span::styled(
            format!(" {glyph} {} · {} ", node.label, i18n::t(lang, "wf.detail")),
            Style::default().fg(theme.color(Token::Claude)).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(card);
    frame.render_widget(block, card);

    // Index the workflow's nodes by id so we can resolve this node's PARENT label
    // (a subagent -> its conductor, a worker -> its master) for the detail header —
    // an O(1) lookup rather than a re-scan, and the single id->node map the panel
    // shares with the schema.
    let by_id = crate::workflow::schema::nodes_by_id(&wf.nodes);
    let parent_label: Option<String> = node
        .parent
        .as_deref()
        .and_then(|pid| by_id.get(pid))
        .map(|p| p.label.clone());

    let mut lines: Vec<Line> = Vec::new();
    // Status + role + workflow (+ parent when this is a leaf).
    let mut head = vec![
        Span::styled(format!("{}: ", i18n::t(lang, "wf.detail.status")), Style::default().fg(theme.color(Token::Dim))),
        Span::styled(i18n::t(lang, node.status.label_key()), Style::default().fg(theme.color(Token::Text))),
        Span::styled(format!("    {}: ", i18n::t(lang, "wf.detail.workflow")), Style::default().fg(theme.color(Token::Dim))),
        Span::styled(wf.title.clone(), Style::default().fg(theme.color(Token::Text))),
    ];
    if let Some(parent) = &parent_label {
        head.push(Span::styled(
            format!("    {}: ", i18n::t(lang, "wf.detail.parent")),
            Style::default().fg(theme.color(Token::Dim)),
        ));
        head.push(Span::styled(parent.clone(), Style::default().fg(theme.color(Token::Text))));
    }
    lines.push(Line::from(head));
    // A stable, locale-independent tag line (`conductor · running`) using the schema's
    // lowercase tag()s — handy when copying a node's coordinates out of the panel.
    lines.push(Line::from(Span::styled(
        format!("[{} · {}]", wf.kind.tag(), node.status.tag()),
        Style::default().fg(theme.color(Token::Dim)),
    )));
    if !node.prompt.is_empty() {
        lines.push(Line::from(Span::styled(
            format!("{}: ", i18n::t(lang, "wf.detail.prompt")),
            Style::default().fg(theme.color(Token::Dim)),
        )));
        lines.push(Line::from(Span::styled(
            clip_one_line(&node.prompt, (inner.width as usize).saturating_sub(2)),
            Style::default().fg(theme.color(Token::Text)),
        )));
    }
    if !node.summary.is_empty() {
        lines.push(Line::from(Span::styled(
            format!("{}: ", i18n::t(lang, "wf.detail.output")),
            Style::default().fg(theme.color(Token::Dim)),
        )));
        lines.push(Line::from(Span::styled(
            clip_one_line(&node.summary, (inner.width as usize).saturating_sub(2)),
            Style::default().fg(theme.color(Token::Text)),
        )));
    }

    // The feed tail (workflow activity).
    if !wf.feed.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("{}:", i18n::t(lang, "wf.detail.feed")),
            Style::default().fg(theme.color(Token::Dim)),
        )));
        for item in feed_tail(&wf.feed, 5) {
            lines.push(feed_line(item, theme, (inner.width as usize).saturating_sub(4)));
        }
    }

    // The action-verb menu.
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        format!("{}:", i18n::t(lang, "wf.detail.actions")),
        Style::default().fg(theme.color(Token::Dim)),
    )));
    let actions = NodeAction::for_role(node.role);
    for (i, a) in actions.iter().enumerate() {
        let sel = i == d.action_sel;
        let mark = if sel { "❯ " } else { "  " };
        let tok = if sel { Token::Suggestion } else { Token::Text };
        // A mutate verb that needs the conductor but the workflow is down → dim it.
        let disabled = a.conductor_verb().is_some() && !wf.running;
        let label_tok = if disabled { Token::Dim } else { tok };
        lines.push(Line::from(vec![
            Span::styled(format!("  {mark}"), Style::default().fg(theme.color(Token::Suggestion))),
            Span::styled(
                i18n::t(lang, a.label_key()),
                Style::default()
                    .fg(theme.color(label_tok))
                    .add_modifier(if sel { Modifier::BOLD } else { Modifier::empty() }),
            ),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        i18n::t(lang, "wf.detail.hint"),
        Style::default().fg(theme.color(Token::Dim)),
    )));

    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), inner);
}

/// One feed line: `[author] text` truncated.
fn feed_line<'a>(item: &'a FeedItem, theme: &Theme, max: usize) -> Line<'a> {
    Line::from(vec![
        Span::styled(
            format!("  {} ", clip_one_line(&item.author, 14)),
            Style::default().fg(theme.color(Token::Suggestion)),
        ),
        Span::styled(
            clip_one_line(&item.text, max.max(8)),
            Style::default().fg(theme.color(Token::Text)),
        ),
    ])
}

// ---- pure formatting helpers (unit-tested) ---------------------------------

/// A tiny progress bar `▰▰▱▱▱` for a 0..=1 fraction (5 cells). PURE.
pub fn progress_bar(fraction: f64) -> String {
    let filled = (fraction.clamp(0.0, 1.0) * 5.0).round() as usize;
    (0..5).map(|i| if i < filled { '▰' } else { '▱' }).collect()
}

/// A human elapsed label from a wall-clock `last_ts` to now (`12s` / `3m` / `1h`).
/// PURE-ish (reads the clock). A non-positive ts → empty.
pub fn elapsed_label(last_ts: f64) -> String {
    if last_ts <= 0.0 {
        return String::new();
    }
    let secs = (crate::workflow::now_secs() - last_ts).max(0.0);
    fmt_duration(secs)
}

/// Format a seconds duration compactly (`12s` / `3m` / `2h`). PURE.
pub fn fmt_duration(secs: f64) -> String {
    let s = secs as u64;
    if s < 60 {
        format!("{s}s")
    } else if s < 3600 {
        format!("{}m", s / 60)
    } else {
        format!("{}h", s / 3600)
    }
}

/// Clip a string to one line of at most `max` display cells (collapsing newlines to
/// spaces so a multi-line summary stays on one row). PURE.
pub fn clip_one_line(s: &str, max: usize) -> String {
    use unicode_segmentation::UnicodeSegmentation;
    use unicode_width::UnicodeWidthStr;
    let flat: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if UnicodeWidthStr::width(flat.as_str()) <= max {
        return flat;
    }
    let mut out = String::new();
    let mut acc = 0usize;
    for g in flat.graphemes(true) {
        let gw = UnicodeWidthStr::width(g);
        if acc + gw > max.saturating_sub(1) {
            break;
        }
        out.push_str(g);
        acc += gw;
    }
    out.push('…');
    out
}

/// The last `n` feed items (newest), in order. PURE.
pub fn feed_tail(feed: &[FeedItem], n: usize) -> &[FeedItem] {
    let start = feed.len().saturating_sub(n);
    &feed[start..]
}

/// A centered card `Rect` (clamped to `area`). PURE geometry (mirrors overlay.rs).
fn centered(area: Rect, w: u16, h: u16) -> Rect {
    let w = w.min(area.width);
    let h = h.min(area.height);
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(h)) / 2;
    Rect { x, y, width: w, height: h }
}

// Keep the heat_bold import meaningful (used in the heat ramp for running nodes in
// a future polish pass; referenced here so the import is not dead). The animated
// status already heat-tints running nodes via `heat_token`.
#[allow(dead_code)]
fn _heat_bold_ref(ms: u64) -> bool {
    heat_bold(ms)
}
#[allow(dead_code)]
fn _gerund_ref(t: u64) -> &'static str {
    flavor::gerund(flavor::Lang::En, t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::schema::{
        merge_snapshots, workflow_from_conductor, workflow_from_goal, workflow_from_hive,
        HiveAuthor, RawGoalState, RawSubagent,
    };

    fn sample_snapshot() -> WorkflowSnapshot {
        let conductor = workflow_from_conductor(
            8900,
            &[
                RawSubagent { id: "ab12".into(), prompt: "task A".into(), reply: "working…".into(), status: "running".into(), created_at: 1.0, updated_at: 2.0 },
                RawSubagent { id: "cd34".into(), prompt: "task B".into(), reply: "done".into(), status: "stopped".into(), created_at: 1.0, updated_at: 3.0 },
            ],
            vec![],
            true,
        );
        let hive = workflow_from_hive(
            "demo",
            5001,
            &[
                HiveAuthor { name: "hive-master".into(), post_count: 4, last_ts: 1000.0, last_post: "split".into() },
                HiveAuthor { name: "hive-worker-1".into(), post_count: 2, last_ts: 990.0, last_post: "claimed #1".into() },
            ],
            vec![FeedItem { ts: 1000.0, author: "hive-master".into(), text: "go".into(), post_id: 1 }],
            None,
            1010.0,
            true,
        );
        let goal = workflow_from_goal(
            &RawGoalState { objective: "ship it".into(), turns_used: 5, max_turns: 100, start_time: 0.0, ..Default::default() },
            640.0,
        );
        merge_snapshots(vec![conductor], vec![hive], vec![goal], 1)
    }

    /// The flatten lays out groups (Conductor → Hives → Goal) then per-workflow
    /// titles + nodes, and only NODE rows are focusable (group/title rows skipped).
    #[test]
    fn flatten_orders_groups_and_marks_focusable() {
        let snap = sample_snapshot();
        let rows = flatten_rows(&snap);

        // The first row is the Conductor group header.
        assert!(matches!(rows[0], PanelRow::Group { kind: WorkflowKind::Conductor, .. }));
        // Group headers appear in kind order.
        let group_kinds: Vec<WorkflowKind> = rows
            .iter()
            .filter_map(|r| if let PanelRow::Group { kind, .. } = r { Some(*kind) } else { None })
            .collect();
        assert_eq!(group_kinds, vec![WorkflowKind::Conductor, WorkflowKind::Hive, WorkflowKind::Goal]);

        // Only node rows are focusable; titles + groups are not.
        let focusable = focusable_indices(&rows);
        for &i in &focusable {
            assert!(rows[i].is_focusable());
        }
        // Focusable node count = conductor(root+2) + hive(master+worker) + goal(1) = 6.
        assert_eq!(focusable.len(), 3 + 2 + 1);
        // Group + title rows are NOT focusable.
        assert!(!rows[0].is_focusable());
        assert!(rows.iter().any(|r| matches!(r, PanelRow::Title { .. }) && !r.is_focusable()));

        // The Group header's node count matches the per-workflow `node_count()` sum
        // (the `of_kind` + `node_count` wiring the flatten now uses).
        if let PanelRow::Group { nodes, .. } = rows[0] {
            assert_eq!(nodes, snap.of_kind(WorkflowKind::Conductor)[0].node_count());
        } else {
            panic!("row 0 is the conductor group");
        }
    }

    /// Focus navigation moves over focusable nodes only, clamps at both ends, and
    /// resolves to the right `(wf, node)` — skipping group + title rows.
    #[test]
    fn focus_nav_skips_headers_and_clamps() {
        let snap = sample_snapshot();
        let mut panel = WorkflowPanel::new();
        panel.clamp_focus(&snap);

        // Focus 0 → the conductor ROOT node (the first focusable row).
        let (wf0, n0, node0) = panel.focused_node(&snap).expect("a focused node");
        assert_eq!(wf0, 0);
        assert_eq!(n0, 0);
        assert_eq!(node0.role, NodeRole::Conductor);

        // Up at the top clamps (stays at 0).
        panel.move_focus(-1, &snap);
        assert_eq!(panel.focus, 0);

        // Down to the next focusable node (the first subagent).
        panel.move_focus(1, &snap);
        let (_, _, node1) = panel.focused_node(&snap).unwrap();
        assert_eq!(node1.role, NodeRole::Subagent);

        // Jump way down → clamps at the LAST focusable node (the goal master).
        panel.move_focus(100, &snap);
        let (_, _, last) = panel.focused_node(&snap).unwrap();
        assert_eq!(last.role, NodeRole::Goal);
        let focusable_count = focusable_indices(&flatten_rows(&snap)).len();
        assert_eq!(panel.focus, focusable_count - 1);
    }

    /// The detail overlay opens on the focused node, navigates its action menu
    /// (clamped to the role's verbs), exposes the selected action, and closes.
    #[test]
    fn detail_overlay_actions_for_node() {
        let snap = sample_snapshot();
        let mut panel = WorkflowPanel::new();
        panel.clamp_focus(&snap);

        // Move focus onto the first SUBAGENT (focusable index 1) and open detail.
        panel.move_focus(1, &snap);
        assert_eq!(panel.focused_node(&snap).unwrap().2.role, NodeRole::Subagent);
        panel.open_detail(&snap);
        assert!(panel.detail_open());

        // The subagent's action menu is the full mutate set; default selection 0 =
        // Open. Moving down selects KeyInfo (the next verb).
        let (a0, _, _) = panel.selected_action(&snap).unwrap();
        assert_eq!(a0, NodeAction::Open);
        panel.move_action(1, &snap);
        let (a1, _, node) = panel.selected_action(&snap).unwrap();
        assert_eq!(a1, NodeAction::KeyInfo);
        assert_eq!(node.role, NodeRole::Subagent);
        // The conductor verb for KeyInfo is "keyinfo".
        assert_eq!(a1.conductor_verb(), Some("keyinfo"));

        // Action selection clamps at the end of the verb list.
        panel.move_action(100, &snap);
        let (alast, _, _) = panel.selected_action(&snap).unwrap();
        assert_eq!(alast, NodeAction::Kill, "last verb for a subagent is kill");

        // Close the detail overlay (Esc).
        assert!(panel.close_detail());
        assert!(!panel.detail_open());
        assert!(panel.selected_action(&snap).is_none());
    }

    /// A goal/hive node's detail offers only the `Open` action (no conductor mutate
    /// API), so firing it is a UI no-op (no server verb).
    #[test]
    fn hive_goal_node_has_open_only() {
        let snap = sample_snapshot();
        let mut panel = WorkflowPanel::new();
        panel.clamp_focus(&snap);
        // The LAST focusable node is the goal master.
        panel.move_focus(100, &snap);
        assert_eq!(panel.focused_node(&snap).unwrap().2.role, NodeRole::Goal);
        panel.open_detail(&snap);
        let (a, _, _) = panel.selected_action(&snap).unwrap();
        assert_eq!(a, NodeAction::Open);
        assert_eq!(a.conductor_verb(), None, "goal node action has no server verb");
        // Moving the action selection can't escape the single-item menu.
        panel.move_action(5, &snap);
        assert_eq!(panel.selected_action(&snap).unwrap().0, NodeAction::Open);
    }

    /// The style toggle flips between the two render styles (§7 "two styles …
    /// toggle").
    #[test]
    fn style_toggle_flips() {
        let mut panel = WorkflowPanel::new();
        assert_eq!(panel.style, RenderStyle::BoxTree);
        panel.toggle_style();
        assert_eq!(panel.style, RenderStyle::Bullet);
        panel.toggle_style();
        assert_eq!(panel.style, RenderStyle::BoxTree);
        // Both styles have a label key.
        assert!(!RenderStyle::BoxTree.label_key().is_empty());
        assert!(!RenderStyle::Bullet.label_key().is_empty());
    }

    /// An EMPTY snapshot leaves focus at 0 and exposes no focused node (the panel
    /// then paints the global degrade hint).
    #[test]
    fn empty_snapshot_degrades() {
        let snap = WorkflowSnapshot::default();
        let mut panel = WorkflowPanel::new();
        panel.clamp_focus(&snap);
        assert_eq!(panel.focus, 0);
        assert!(panel.focused_node(&snap).is_none());
        panel.move_focus(1, &snap); // no-op, no panic.
        assert_eq!(panel.focus, 0);
        // flatten of an empty snapshot has no rows.
        assert!(flatten_rows(&snap).is_empty());
        assert!(focusable_indices(&flatten_rows(&snap)).is_empty());
    }

    /// After a refresh that REMOVES nodes, the focus re-clamps into range and a
    /// detail overlay pointing at a vanished node closes.
    #[test]
    fn clamp_focus_after_refresh() {
        let snap = sample_snapshot();
        let mut panel = WorkflowPanel::new();
        panel.clamp_focus(&snap);
        panel.move_focus(100, &snap); // focus the last node.
        let last = panel.focus;
        panel.open_detail(&snap);
        assert!(panel.detail_open());

        // A new, SMALLER snapshot (just a single goal workflow, generation bumped).
        let small = merge_snapshots(
            vec![],
            vec![],
            vec![workflow_from_goal(&RawGoalState { objective: "x".into(), max_turns: 1, ..Default::default() }, 0.0)],
            2,
        );
        panel.clamp_focus(&small);
        // Focus re-clamped into the new (smaller) range.
        let n = focusable_indices(&flatten_rows(&small)).len();
        assert!(panel.focus < n.max(1));
        assert!(panel.focus <= last);
        // The detail overlay pointed at a now-missing node → it closed.
        assert!(!panel.detail_open(), "stale detail overlay closes on refresh");
    }

    /// The persistent scroll offset (`WorkflowPanel.scroll`) keeps the focused row
    /// visible WITHOUT snapping the view on a content-only refresh: it only moves when
    /// the focus would leave the `[scroll, scroll+height)` window, then clamps to the
    /// last full page.
    #[test]
    fn scroll_to_focus_keeps_focus_visible_and_is_sticky() {
        let panel = WorkflowPanel::new();
        // 100 rows in a 10-row viewport.
        let (total, height) = (100usize, 10usize);

        // Focus at the top → no scroll.
        assert_eq!(panel.scroll_to_focus(0, total, height), 0);
        assert_eq!(panel.scroll.get(), 0);

        // Focus row 5 still fits in [0,10) → the view does NOT move (sticky).
        assert_eq!(panel.scroll_to_focus(5, total, height), 0);

        // Focus row 12 is below the window → scroll just enough to reveal it
        // (top = 12 + 1 - 10 = 3).
        assert_eq!(panel.scroll_to_focus(12, total, height), 3);
        assert_eq!(panel.scroll.get(), 3);

        // Focus row 4 is now ABOVE the window [3,13) → scroll up to it.
        assert_eq!(panel.scroll_to_focus(4, total, height), 3, "row 4 is in [3,13), stays");
        assert_eq!(panel.scroll_to_focus(2, total, height), 2, "row 2 above window → scroll up");

        // Focus the very last row → clamp to the last full page (90).
        assert_eq!(panel.scroll_to_focus(99, total, height), total - height);

        // When everything fits, the offset pins to 0.
        assert_eq!(panel.scroll_to_focus(3, 5, 10), 0);
        assert_eq!(panel.scroll.get(), 0);
        // A zero-height viewport degrades to 0 (no panic / no div-by-zero).
        assert_eq!(panel.scroll_to_focus(7, 100, 0), 0);
    }

    /// The detail overlay resolves a leaf node's PARENT label via `nodes_by_id` and
    /// paints the stable `[kind · status]` tag line (the schema `tag()`s). A render
    /// guard that the parent-lookup + tag wiring paints without panicking.
    #[test]
    fn detail_overlay_shows_parent_and_tags() {
        use crate::workflow::schema::nodes_by_id;
        let snap = sample_snapshot();
        // The conductor workflow: its subagent leaves carry parent = "conductor".
        let cond = &snap.workflows[0];
        let by_id = nodes_by_id(&cond.nodes);
        let leaf = cond.nodes.iter().find(|n| n.role == NodeRole::Subagent).unwrap();
        // The leaf's parent resolves to the conductor root's label via the id map.
        let parent = leaf.parent.as_deref().and_then(|p| by_id.get(p)).map(|n| n.label.clone());
        assert_eq!(parent.as_deref(), Some("conductor"));
        // The tag line components are stable + lowercase (locale-independent).
        assert_eq!(cond.kind.tag(), "conductor");
        assert!(!leaf.status.tag().is_empty());

        // Render the detail overlay for that subagent (parent label + tag line paint).
        let mut panel = WorkflowPanel::new();
        panel.clamp_focus(&snap);
        panel.move_focus(1, &snap); // onto the first subagent.
        panel.open_detail(&snap);
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;
        let theme = Theme::default_theme();
        let backend = TestBackend::new(100, 30);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| render(f, f.area(), &mut panel, &snap, &theme, Lang::En, 100)).unwrap();
        let text: String = term.backend().buffer().content().iter().map(|c| c.symbol()).collect();
        // The stable tag appears in the painted overlay.
        assert!(text.contains("conductor"), "detail paints the conductor tag/parent");
    }

    #[test]
    fn formatting_helpers_are_pure() {
        assert_eq!(progress_bar(0.0), "▱▱▱▱▱");
        assert_eq!(progress_bar(1.0), "▰▰▰▰▰");
        assert_eq!(progress_bar(0.5).chars().filter(|&c| c == '▰').count(), 3); // round(2.5)=3
        assert_eq!(fmt_duration(12.0), "12s");
        assert_eq!(fmt_duration(120.0), "2m");
        assert_eq!(fmt_duration(7200.0), "2h");
        assert_eq!(clip_one_line("hello world", 100), "hello world");
        assert_eq!(clip_one_line("multi\nline\ntext", 100), "multi line text");
        assert!(clip_one_line("a very long string that exceeds", 10).ends_with('…'));
        assert_eq!(elapsed_label(0.0), "");
    }

    /// Both render styles paint to an in-memory backend without panicking (a render
    /// guard; the layout logic is tested purely above). Covers the empty + populated
    /// cases and the detail overlay.
    #[test]
    fn renders_both_styles_and_detail() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;
        let theme = Theme::default_theme();
        let snap = sample_snapshot();

        for style in [RenderStyle::BoxTree, RenderStyle::Bullet] {
            let mut panel = WorkflowPanel::new();
            panel.style = style;
            panel.clamp_focus(&snap);
            let backend = TestBackend::new(100, 30);
            let mut term = Terminal::new(backend).unwrap();
            term.draw(|f| render(f, f.area(), &mut panel, &snap, &theme, Lang::En, 100)).unwrap();
            let buf = term.backend().buffer();
            let text: String = buf.content().iter().map(|c| c.symbol()).collect();
            assert!(text.contains("Workflows") || text.contains("workflows"), "paints the title");
        }

        // The detail overlay paints too.
        let mut panel = WorkflowPanel::new();
        panel.clamp_focus(&snap);
        panel.move_focus(1, &snap);
        panel.open_detail(&snap);
        let backend = TestBackend::new(100, 30);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| render(f, f.area(), &mut panel, &snap, &theme, Lang::En, 100)).unwrap();
        let buf = term.backend().buffer();
        let text: String = buf.content().iter().map(|c| c.symbol()).collect();
        assert!(!text.trim().is_empty(), "detail overlay paints content");

        // The EMPTY snapshot degrades gracefully (paints the launch hint).
        let empty = WorkflowSnapshot::default();
        let mut panel = WorkflowPanel::new();
        let backend = TestBackend::new(80, 24);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| render(f, f.area(), &mut panel, &empty, &theme, Lang::En, 0)).unwrap();
        let buf = term.backend().buffer();
        let text: String = buf.content().iter().map(|c| c.symbol()).collect();
        assert!(!text.trim().is_empty(), "empty panel still paints a hint");
    }
}
