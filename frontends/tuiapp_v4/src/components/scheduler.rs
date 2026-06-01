//! components/scheduler.rs — the `/scheduler` 3-step flow (checklist §7):
//!
//!   step 1 — MULTI-PICK reflect tasks (the currently-RUNNING ones are pre-ticked,
//!            so leaving a task ticked keeps it running and un-ticking it stops it);
//!   step 2 — CONFIRM card showing the start/stop DIFF (what will change vs the
//!            current cron state) so the user reviews before applying;
//!   step 3 — APPLY + show the resulting cron STATUS (which tasks are now running).
//!
//! The whole flow is ONE overlay whose `step` advances on Enter and steps back on
//! Esc (Esc from step 1 closes). All the load-bearing logic — the start/stop diff,
//! the step transitions, the desired-set computation — lives in PURE functions +
//! the [`Scheduler`] state machine, unit-tested without a TTY (the
//! `scheduler_flow_states` deliverable). The renderer only PAINTS, routing every
//! user-facing string through `crate::i18n` (no hardcoded colors — theme tokens).

use std::path::Path;

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::i18n::{self, Lang};
use crate::theme::{Theme, Token};

/// One schedulable reflect task (a cron-able background job). `running` is its
/// CURRENT state (read from the scheduler when the flow opens); `id` is stable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReflectTask {
    /// Stable task id (the index the apply step reports to the core).
    pub id: usize,
    /// Display name (the reflect-script stem or cron-task stem).
    pub name: String,
    /// The cadence label: a reflect mode reads `"reflect"`; a cron task reads its
    /// legal `repeat` grammar (`once/daily/weekday/weekly/monthly/every_*`).
    pub cadence: String,
    /// Whether this task is currently running/scheduled (drives the pre-tick).
    pub running: bool,
}

impl ReflectTask {
    pub fn new(id: usize, name: impl Into<String>, cadence: impl Into<String>, running: bool) -> Self {
        ReflectTask { id, name: name.into(), cadence: cadence.into(), running }
    }
}

/// Which step of the 3-step flow the overlay is on. PURE state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedStep {
    /// Step 1: multi-pick the desired-running set (running tasks pre-ticked).
    Pick,
    /// Step 2: confirm the start/stop diff before applying.
    Confirm,
    /// Step 3: applied — show the resulting cron status.
    Status,
}

/// One line of the start/stop DIFF (step 2): a task transitioning state. PURE data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffEntry {
    pub id: usize,
    pub name: String,
    pub change: DiffChange,
}

/// The direction of a diff entry (a task being started or stopped). PURE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffChange {
    /// The task is currently stopped and will be STARTED.
    Start,
    /// The task is currently running and will be STOPPED.
    Stop,
}

/// The `/scheduler` overlay state machine (§7). Owns the task rows + per-row
/// toggles + the current step. Built from the live reflect-task list (running ones
/// pre-ticked); `apply()` advances to the status step.
#[derive(Debug, Clone)]
pub struct Scheduler {
    /// The task rows (id / name / cadence / current running state).
    pub tasks: Vec<ReflectTask>,
    /// The DESIRED-running toggles, parallel to `tasks`. Seeded from each task's
    /// `running` flag (the "pre-tick currently-running" rule), then edited by Space.
    pub checked: Vec<bool>,
    /// The highlighted row (step 1).
    pub sel: usize,
    /// The current step.
    pub step: SchedStep,
    /// Once applied (step 3), the desired set is frozen here so the status view
    /// reflects the applied state even if the toggles were later mutated.
    pub applied: Option<Vec<bool>>,
}

impl Scheduler {
    /// Build the flow from the live reflect-task list. The desired-running toggles
    /// start equal to each task's CURRENT running state (so the user sees the
    /// running ones pre-ticked, §7), the selection on the first row, step = Pick.
    pub fn new(tasks: Vec<ReflectTask>) -> Self {
        let checked: Vec<bool> = tasks.iter().map(|t| t.running).collect();
        Scheduler { tasks, checked, sel: 0, step: SchedStep::Pick, applied: None }
    }

    /// Number of task rows.
    #[allow(dead_code)] // symmetry with is_empty; used by tests / future callers.
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Move the step-1 selection by `delta` (clamped). No-op off the pick step.
    pub fn move_sel(&mut self, delta: isize) {
        if self.step != SchedStep::Pick {
            return;
        }
        self.sel = crate::commands::registry::move_sel(self.sel, delta, self.tasks.len());
    }

    /// Toggle the desired-running state of the highlighted row (Space, step 1).
    pub fn toggle_selected(&mut self) {
        if self.step != SchedStep::Pick {
            return;
        }
        if let Some(c) = self.checked.get_mut(self.sel) {
            *c = !*c;
        }
    }

    /// The start/stop DIFF between the CURRENT running state and the desired
    /// (checked) set (step 2). A task that is checked-but-not-running is a Start; a
    /// task that is running-but-unchecked is a Stop; unchanged tasks are omitted.
    /// PURE — the headline diff the confirm card shows and the apply step sends.
    pub fn diff(&self) -> Vec<DiffEntry> {
        compute_diff(&self.tasks, &self.checked)
    }

    /// True if the desired set differs from the current state at all (drives the
    /// "no changes" message on the confirm/apply step). PURE.
    #[allow(dead_code)] // read by the flow test + a natural guard for callers.
    pub fn has_changes(&self) -> bool {
        !self.diff().is_empty()
    }

    /// Advance Pick → Confirm (review the diff). No-op off the pick step. Returns
    /// the step we are NOW on (so the key handler can branch). The transition is
    /// allowed even with no changes — the confirm card then shows "no changes".
    pub fn to_confirm(&mut self) -> SchedStep {
        if self.step == SchedStep::Pick {
            self.step = SchedStep::Confirm;
        }
        self.step
    }

    /// Step Confirm → back to Pick (Esc on the confirm card). No-op otherwise.
    pub fn back_to_pick(&mut self) -> SchedStep {
        if self.step == SchedStep::Confirm {
            self.step = SchedStep::Pick;
        }
        self.step
    }

    /// Apply the desired set: freeze it for the status view and advance to Status
    /// (step 3). The caller forwards the actual start/stop set to the core. No-op
    /// off the confirm step. Returns the (now Status) step. PURE-ish.
    pub fn apply(&mut self) -> SchedStep {
        if self.step == SchedStep::Confirm {
            self.applied = Some(self.checked.clone());
            self.step = SchedStep::Status;
        }
        self.step
    }

    /// The ids the apply step should START (checked but not currently running). PURE.
    pub fn to_start(&self) -> Vec<usize> {
        self.diff()
            .into_iter()
            .filter(|d| d.change == DiffChange::Start)
            .map(|d| d.id)
            .collect()
    }

    /// The ids the apply step should STOP (running but unchecked). PURE.
    pub fn to_stop(&self) -> Vec<usize> {
        self.diff()
            .into_iter()
            .filter(|d| d.change == DiffChange::Stop)
            .map(|d| d.id)
            .collect()
    }

    /// The desired-running ids after apply (the status view's "running" set). Uses
    /// the frozen `applied` set if present, else the live toggles. PURE.
    pub fn running_after(&self) -> Vec<usize> {
        let set = self.applied.as_ref().unwrap_or(&self.checked);
        self.tasks
            .iter()
            .zip(set.iter())
            .filter(|(_, on)| **on)
            .map(|(t, _)| t.id)
            .collect()
    }
}

/// Compute the start/stop diff between tasks' current `running` state and a desired
/// `checked` set (parallel slices). PURE — shared by the state machine + the test.
pub fn compute_diff(tasks: &[ReflectTask], checked: &[bool]) -> Vec<DiffEntry> {
    let mut out = Vec::new();
    for (i, t) in tasks.iter().enumerate() {
        let want = checked.get(i).copied().unwrap_or(t.running);
        if want && !t.running {
            out.push(DiffEntry { id: t.id, name: t.name.clone(), change: DiffChange::Start });
        } else if !want && t.running {
            out.push(DiffEntry { id: t.id, name: t.name.clone(), change: DiffChange::Stop });
        }
    }
    out
}

/// Discover the REAL reflect modes + cron tasks (slash_cmds.py parity, §7 Q10):
///   `reflect/*.py` (non-`_`, minus `scheduler.py` itself) → reflect-mode rows,
///       cadence `"reflect"` (the cron ENGINE, not a cron job; running detection is
///       a future cmdline probe so they start un-ticked here).
///   `sche_tasks/*.json` → cron rows, cadence = the legal `repeat` label, pre-ticked
///       iff `enabled` (the cron task's CURRENT scheduled state).
/// `id` is the row index the apply step reports. Reflect modes sort before cron
/// tasks; within each group the directory order is sorted for stability.
pub fn discover_tasks(repo_root: &Path) -> Vec<ReflectTask> {
    let mut out: Vec<ReflectTask> = Vec::new();
    let mut id = 0usize;

    let mut reflect: Vec<String> = read_dir_stems(&repo_root.join("reflect"), ".py")
        .into_iter()
        .filter(|n| n != "scheduler")
        .collect();
    reflect.sort();
    for name in reflect {
        out.push(ReflectTask::new(id, name, "reflect", false));
        id += 1;
    }

    let mut cron: Vec<String> = read_dir_stems(&repo_root.join("sche_tasks"), ".json");
    cron.sort();
    for name in cron {
        let (cadence, enabled) = read_cron_meta(&repo_root.join("sche_tasks").join(format!("{name}.json")));
        out.push(ReflectTask::new(id, name, cadence, enabled));
        id += 1;
    }
    out
}

/// Names (extension-stripped) of every `<root>/*<ext>` entry whose file name does
/// not start with `_`. Empty (not an error) when the dir is missing — the caller
/// then falls back to [`default_tasks`].
fn read_dir_stems(dir: &Path, ext: &str) -> Vec<String> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in rd.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(ext) && !name.starts_with('_') {
            out.push(name[..name.len() - ext.len()].to_string());
        }
    }
    out
}

/// Read a `sche_tasks/*.json` cron file's `(cadence, enabled)`. The cadence is the
/// legal `repeat` grammar (`once/daily/weekday/weekly/monthly/every_*`, mirroring
/// `slash_cmds.list_scheduler_tasks`'s `repeat|cron|every` fallback); `schedule`
/// (HH:MM) is deliberately NOT the cadence. A missing/unparsable file degrades to
/// `("reflect", false)` so a malformed task still renders.
fn read_cron_meta(path: &Path) -> (String, bool) {
    let Ok(text) = std::fs::read_to_string(path) else {
        return ("reflect".into(), false);
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return ("reflect".into(), false);
    };
    let cadence = ["repeat", "cron", "every"]
        .iter()
        .find_map(|k| value.get(*k).and_then(|v| v.as_str()))
        .unwrap_or("reflect")
        .to_string();
    let enabled = value.get("enabled").and_then(|v| v.as_bool()).unwrap_or(true);
    (cadence, enabled)
}

/// The default reflect-task set surfaced ONLY when [`discover_tasks`] finds an
/// empty tree (no `reflect/` or `sche_tasks/` on disk) — a small representative set
/// so the flow is demonstrable and degrades gracefully (§7 "Degrade gracefully").
/// The first task is shown as already-running so the pre-tick + a stop-diff are
/// both exercisable.
pub fn default_tasks() -> Vec<ReflectTask> {
    vec![
        ReflectTask::new(0, "daily standup reflect", "09:00", true),
        ReflectTask::new(1, "inbox triage", "hourly", false),
        ReflectTask::new(2, "weekly review", "Fri 17:00", false),
        ReflectTask::new(3, "memory consolidation", "03:00", false),
    ]
}

// ---------------------------------------------------------------------------
// Renderer (PAINTS only; all logic is in the state machine above).
// ---------------------------------------------------------------------------

/// Draw the scheduler card for the active step. Centered, bordered, theme-tokened.
pub fn render(frame: &mut Frame, area: Rect, sched: &Scheduler, theme: &Theme, lang: Lang) {
    let (title_key, hint_key) = match sched.step {
        SchedStep::Pick => ("scheduler.title.pick", "scheduler.hint.pick"),
        SchedStep::Confirm => ("scheduler.title.confirm", "scheduler.hint.confirm"),
        SchedStep::Status => ("scheduler.title.status", "scheduler.hint.status"),
    };
    let w = (area.width.saturating_sub(8)).clamp(36, 80);
    let body_rows = sched.tasks.len().max(1) as u16 + 6;
    let h = body_rows.min(area.height.saturating_sub(2)).max(7);
    let card = centered(area, w, h);
    frame.render_widget(Clear, card);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.color(Token::Claude)))
        .title(Span::styled(
            format!(" {} ", i18n::t(lang, title_key)),
            Style::default().fg(theme.color(Token::Claude)).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(card);
    frame.render_widget(block, card);

    let lines = match sched.step {
        SchedStep::Pick => pick_lines(sched, theme, lang),
        SchedStep::Confirm => confirm_lines(sched, theme, lang),
        SchedStep::Status => status_lines(sched, theme, lang),
    };
    let mut lines = lines;
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        i18n::t(lang, hint_key),
        Style::default().fg(theme.color(Token::Dim)),
    )));
    frame.render_widget(Paragraph::new(lines), inner);
}

/// The localized row-kind tag: a `"reflect"`-cadence row is a reflect mode, any
/// other cadence is a cron task. PURE (no `Instant`).
fn kind_label(lang: Lang, t: &ReflectTask) -> &'static str {
    if t.cadence == "reflect" {
        i18n::t(lang, "scheduler.kind.reflect")
    } else {
        i18n::t(lang, "scheduler.kind.cron")
    }
}

/// The display label for a cadence. A reflect mode shows the localized "watcher"
/// label; a cron task shows its legal `repeat` grammar — localized for the well-
/// known repeats (`once/daily/weekday/weekly/monthly`), else the RAW label verbatim
/// (`every_30m`, …). Never an HH:MM `schedule`. PURE.
fn cadence_label(lang: Lang, cadence: &str) -> String {
    match cadence {
        "reflect" => i18n::t(lang, "scheduler.cadence.reflect").to_string(),
        "once" | "daily" | "weekday" | "weekly" | "monthly" => {
            i18n::t(lang, &format!("scheduler.repeat.{cadence}")).to_string()
        }
        other => other.to_string(),
    }
}

/// Step 1 rows: a `[x]`/`[ ]` checkbox + name + cadence + a running marker.
fn pick_lines<'a>(sched: &'a Scheduler, theme: &Theme, lang: Lang) -> Vec<Line<'a>> {
    let mut lines: Vec<Line> = Vec::new();
    if sched.is_empty() {
        lines.push(Line::from(Span::styled(
            i18n::t(lang, "scheduler.cron_none"),
            Style::default().fg(theme.color(Token::Dim)),
        )));
        return lines;
    }
    for (i, t) in sched.tasks.iter().enumerate() {
        let selected = i == sched.sel;
        let on = sched.checked.get(i).copied().unwrap_or(false);
        let mut spans: Vec<Span> = vec![Span::styled(
            if selected { "❯ " } else { "  " },
            Style::default().fg(theme.color(Token::Suggestion)),
        )];
        let box_ = if on { "[x] " } else { "[ ] " };
        let tok = if on { Token::Success } else { Token::Dim };
        spans.push(Span::styled(box_, Style::default().fg(theme.color(tok))));
        spans.push(Span::styled(
            t.name.clone(),
            Style::default()
                .fg(theme.color(if selected { Token::Suggestion } else { Token::Text }))
                .add_modifier(if selected { Modifier::BOLD } else { Modifier::empty() }),
        ));
        spans.push(Span::styled(
            format!("   {} · {}", kind_label(lang, t), cadence_label(lang, &t.cadence)),
            Style::default().fg(theme.color(Token::Dim)),
        ));
        if t.running {
            spans.push(Span::styled(
                format!("  · {}", i18n::t(lang, "scheduler.running")),
                Style::default().fg(theme.color(Token::Success)),
            ));
        }
        lines.push(Line::from(spans));
    }
    lines
}

/// Step 2 rows: the start/stop DIFF (or a "no changes" line).
fn confirm_lines<'a>(sched: &Scheduler, theme: &Theme, lang: Lang) -> Vec<Line<'a>> {
    let mut lines: Vec<Line> = Vec::new();
    let diff = sched.diff();
    if diff.is_empty() {
        lines.push(Line::from(Span::styled(
            i18n::t(lang, "scheduler.no_changes"),
            Style::default().fg(theme.color(Token::Dim)),
        )));
        return lines;
    }
    for d in diff {
        let (arrow, label_key, tok) = match d.change {
            DiffChange::Start => ("＋", "scheduler.will_start", Token::Success),
            DiffChange::Stop => ("－", "scheduler.will_stop", Token::Warning),
        };
        lines.push(Line::from(vec![
            Span::styled(format!("  {arrow} "), Style::default().fg(theme.color(tok))),
            Span::styled(
                format!("{}: ", i18n::t(lang, label_key)),
                Style::default().fg(theme.color(Token::Dim)),
            ),
            Span::styled(d.name, Style::default().fg(theme.color(Token::Text)).add_modifier(Modifier::BOLD)),
        ]));
    }
    lines
}

/// Step 3 rows: the resulting cron status (which tasks are running now).
fn status_lines<'a>(sched: &'a Scheduler, theme: &Theme, lang: Lang) -> Vec<Line<'a>> {
    let mut lines: Vec<Line> = vec![Line::from(Span::styled(
        i18n::t(lang, "scheduler.applied"),
        Style::default().fg(theme.color(Token::Success)).add_modifier(Modifier::BOLD),
    ))];
    let running = sched.running_after();
    if running.is_empty() {
        lines.push(Line::from(Span::styled(
            format!("  {}", i18n::t(lang, "scheduler.cron_none")),
            Style::default().fg(theme.color(Token::Dim)),
        )));
        return lines;
    }
    lines.push(Line::from(Span::styled(
        format!("  {}:", i18n::t(lang, "scheduler.cron_active")),
        Style::default().fg(theme.color(Token::Dim)),
    )));
    for t in &sched.tasks {
        if running.contains(&t.id) {
            lines.push(Line::from(vec![
                Span::styled("  ⏺ ", Style::default().fg(theme.color(Token::Success))),
                Span::styled(t.name.clone(), Style::default().fg(theme.color(Token::Text))),
                Span::styled(
                    format!("   {}", cadence_label(lang, &t.cadence)),
                    Style::default().fg(theme.color(Token::Dim)),
                ),
            ]));
        }
    }
    lines
}

/// A centered card `Rect` (clamped to `area`). PURE geometry (mirrors overlay.rs).
fn centered(area: Rect, w: u16, h: u16) -> Rect {
    let w = w.min(area.width);
    let h = h.min(area.height);
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(h)) / 2;
    Rect { x, y, width: w, height: h }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tasks() -> Vec<ReflectTask> {
        vec![
            ReflectTask::new(0, "standup", "09:00", true),   // running
            ReflectTask::new(1, "triage", "hourly", false),  // stopped
            ReflectTask::new(2, "review", "Fri 17:00", false), // stopped
        ]
    }

    /// THE deliverable test: the 3-step scheduler flow state transitions + the
    /// start/stop diff. Pick (running pre-ticked) → toggle → Confirm (diff shows
    /// the start + stop) → apply → Status (resulting running set). Esc steps back.
    #[test]
    fn scheduler_flow_states() {
        let mut s = Scheduler::new(tasks());

        // -- step 1: PICK. The running task (id 0) is PRE-TICKED; the rest are off.
        assert_eq!(s.step, SchedStep::Pick);
        assert_eq!(s.checked, vec![true, false, false], "running task pre-ticked");
        // No changes yet → the diff is empty.
        assert!(!s.has_changes());
        assert!(s.diff().is_empty());

        // The user STOPS the running task (un-tick id 0) and STARTS id 1.
        s.sel = 0;
        s.toggle_selected(); // id 0 off → will Stop.
        s.sel = 1;
        s.toggle_selected(); // id 1 on → will Start.
        assert_eq!(s.checked, vec![false, true, false]);

        // -- the DIFF: id 1 Start, id 0 Stop (id 2 unchanged → omitted).
        let diff = s.diff();
        assert_eq!(diff.len(), 2);
        assert!(diff.iter().any(|d| d.id == 1 && d.change == DiffChange::Start));
        assert!(diff.iter().any(|d| d.id == 0 && d.change == DiffChange::Stop));
        assert!(!diff.iter().any(|d| d.id == 2), "unchanged task is not in the diff");
        assert_eq!(s.to_start(), vec![1]);
        assert_eq!(s.to_stop(), vec![0]);
        assert!(s.has_changes());

        // -- step 2: CONFIRM. Enter advances Pick → Confirm; toggles are frozen for
        // review (move/toggle are no-ops off the pick step).
        assert_eq!(s.to_confirm(), SchedStep::Confirm);
        s.move_sel(1); // no-op
        s.toggle_selected(); // no-op
        assert_eq!(s.checked, vec![false, true, false], "confirm step doesn't mutate the set");

        // Esc steps BACK to Pick (re-editable), then forward again.
        assert_eq!(s.back_to_pick(), SchedStep::Pick);
        assert_eq!(s.to_confirm(), SchedStep::Confirm);

        // -- step 3: APPLY. Enter on Confirm freezes the set + advances to Status.
        assert_eq!(s.apply(), SchedStep::Status);
        assert_eq!(s.applied.as_ref().unwrap(), &vec![false, true, false]);
        // The resulting running set is exactly the desired (id 1).
        assert_eq!(s.running_after(), vec![1]);
        // Apply is idempotent off the confirm step (already applied).
        assert_eq!(s.apply(), SchedStep::Status);
    }

    /// With no changes the flow still advances to Confirm (showing "no changes")
    /// and applies to the same running set. PURE.
    #[test]
    fn scheduler_no_change_path() {
        let mut s = Scheduler::new(tasks());
        // Don't touch the toggles → no diff.
        assert!(!s.has_changes());
        assert_eq!(s.to_confirm(), SchedStep::Confirm);
        assert!(s.diff().is_empty());
        assert_eq!(s.apply(), SchedStep::Status);
        // The running set is unchanged (only id 0 was running).
        assert_eq!(s.running_after(), vec![0]);
    }

    /// The pre-tick rule + compute_diff are robust to a desired set built from the
    /// running flags (round-trip: seeding checked == running yields an empty diff).
    #[test]
    fn pretick_matches_running_and_diff_is_pure() {
        let t = tasks();
        let checked: Vec<bool> = t.iter().map(|x| x.running).collect();
        assert!(compute_diff(&t, &checked).is_empty(), "pre-ticked == running → no diff");
        // Starting everything: two starts (ids 1,2), no stop.
        let all_on = vec![true, true, true];
        let diff = compute_diff(&t, &all_on);
        assert_eq!(diff.iter().filter(|d| d.change == DiffChange::Start).count(), 2);
        assert_eq!(diff.iter().filter(|d| d.change == DiffChange::Stop).count(), 0);
        // Stopping everything: one stop (id 0), no start.
        let all_off = vec![false, false, false];
        let diff = compute_diff(&t, &all_off);
        assert_eq!(diff.iter().filter(|d| d.change == DiffChange::Stop).count(), 1);
        // default_tasks() has the first task running (pre-tick + stop-diff demoable).
        assert!(default_tasks()[0].running);
    }

    /// The renderer paints each step's chrome to an in-memory backend (a render
    /// guard; the logic is tested above).
    #[test]
    fn scheduler_renders_each_step() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;
        let theme = Theme::default_theme();
        for step in [SchedStep::Pick, SchedStep::Confirm, SchedStep::Status] {
            let mut s = Scheduler::new(tasks());
            s.sel = 1;
            s.toggle_selected();
            s.step = step;
            let backend = TestBackend::new(80, 24);
            let mut term = Terminal::new(backend).unwrap();
            term.draw(|f| render(f, f.area(), &s, &theme, Lang::En)).unwrap();
            let buf = term.backend().buffer();
            let text: String = buf.content().iter().map(|c| c.symbol()).collect();
            assert!(text.contains("Scheduler"), "step {step:?} paints a title");
        }
    }

    /// `discover_tasks` reads the REAL sources: every `reflect/*.py` (non-`_`,
    /// minus `scheduler.py`) becomes a `"reflect"`-cadence mode row, and every
    /// `sche_tasks/*.json` becomes a cron row whose cadence is the legal `repeat`
    /// label and whose pre-tick follows `enabled`. The HH:MM `schedule` is NEVER
    /// surfaced as a cadence (the old fake-`09:00` bug).
    #[test]
    fn scheduler_discovers_reflect_and_cron() {
        let root = std::env::temp_dir().join(format!("tui_v4_sched_disc_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("reflect")).unwrap();
        std::fs::create_dir_all(root.join("sche_tasks")).unwrap();
        // reflect modes: two real, one private (`_`-prefixed, skipped), plus the
        // cron engine `scheduler.py` (skipped).
        std::fs::write(root.join("reflect").join("autonomous.py"), "x").unwrap();
        std::fs::write(root.join("reflect").join("goal_mode.py"), "x").unwrap();
        std::fs::write(root.join("reflect").join("_helper.py"), "x").unwrap();
        std::fs::write(root.join("reflect").join("scheduler.py"), "x").unwrap();
        // cron tasks: an enabled daily one, a disabled every_30m one.
        std::fs::write(
            root.join("sche_tasks").join("crypto_morning_brief.json"),
            r#"{"schedule":"08:30","repeat":"daily","enabled":true}"#,
        )
        .unwrap();
        std::fs::write(
            root.join("sche_tasks").join("linuxdo_monitor.json"),
            r#"{"schedule":"00:00","repeat":"every_30m","enabled":false}"#,
        )
        .unwrap();

        let tasks = discover_tasks(&root);
        // 2 reflect modes (autonomous, goal_mode) + 2 cron tasks. No `_helper`, no
        // `scheduler` engine row.
        assert_eq!(tasks.len(), 4, "2 reflect modes + 2 cron tasks");
        assert!(!tasks.iter().any(|t| t.name == "scheduler"), "cron engine is skipped");
        assert!(!tasks.iter().any(|t| t.name == "_helper"), "private scripts skipped");

        let goal = tasks.iter().find(|t| t.name == "goal_mode").expect("goal_mode mode row");
        assert_eq!(goal.cadence, "reflect", "reflect modes carry the 'reflect' cadence");
        assert!(!goal.running, "reflect modes start un-ticked (running probed separately)");

        let brief = tasks.iter().find(|t| t.name == "crypto_morning_brief").expect("cron row");
        assert_eq!(brief.cadence, "daily", "cron cadence is the legal repeat label, not 08:30");
        assert!(brief.running, "enabled cron task is pre-ticked");
        let lx = tasks.iter().find(|t| t.name == "linuxdo_monitor").expect("cron row");
        assert_eq!(lx.cadence, "every_30m");
        assert!(!lx.running, "disabled cron task is not pre-ticked");
        // No fake HH:MM cadence anywhere.
        assert!(!tasks.iter().any(|t| t.cadence.contains(':')), "no HH:MM cadence labels");

        let _ = std::fs::remove_dir_all(&root);
    }

    /// With no `reflect/` or `sche_tasks/` on disk, `discover_tasks` returns empty
    /// (so the open path falls back to [`default_tasks`] — never a blank overlay).
    #[test]
    fn scheduler_falls_back_to_default_on_empty() {
        let root = std::env::temp_dir().join(format!("tui_v4_sched_empty_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        assert!(discover_tasks(&root).is_empty(), "no sources → empty discovery");
        // The caller's fallback (mirrored here) yields the representative set.
        let tasks = if discover_tasks(&root).is_empty() { default_tasks() } else { discover_tasks(&root) };
        assert!(!tasks.is_empty(), "fallback yields the default task set");
        let _ = std::fs::remove_dir_all(&root);
    }
}
