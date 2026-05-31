//! workflow/schema.rs — the unified `WorkflowSnapshot` model + the PURE,
//! load-bearing logic the `/workflows` panel + watcher depend on (checklist §7;
//! recon `ga_core_bridge_hive.md` §5).
//!
//! The watcher (`workflow/mod.rs`) merges THREE independent live sources into ONE
//! `WorkflowSnapshot` so the panel renders a single tree regardless of origin:
//!   * **Conductor** — `GET http://127.0.0.1:8900/subagent` poll (+ optional WS);
//!     a flat tree `conductor → N subagents`; node status from `SubAgentState`.
//!   * **Hive** — an `agent_bbs.py` board on `127.0.0.1:<port>?key=<key>`; nodes =
//!     distinct `/authors`; status/liveness INFERRED from the last-post age (the
//!     BBS has no status column, recon §3.2). Auth read from `temp/hive_<name>/board.json`.
//!   * **Goal** — `temp/goal_state.json` mtime poll → ONE master node with real
//!     progress (`turns_used`/`max_turns`, elapsed/budget, lifecycle `status`).
//!
//! ALL the logic that the four required deliverable tests pin lives here as PURE
//! functions over plain data (no I/O, no sockets, no ratatui) so it is unit-tested
//! headlessly:
//!   * [`merge_snapshots`]            → `workflow_snapshot_merge`
//!   * [`hive_status_from_age`]       → `hive_liveness_from_age`
//!   * [`apply_conductor_tombstones`] → `conductor_tombstone_hides_aborted`
//!   * [`parse_goal_state`]           → `goal_progress_parse`
//!
//! The watcher just wires sockets/files to these functions; the panel just paints
//! their output. This keeps the orchestration thin and the contract testable.

use std::collections::BTreeMap;

use serde::Deserialize;

// ===========================================================================
// The unified snapshot model (recon §5.1, adapted to the Rust UI).
// ===========================================================================

/// Which orchestration KIND a workflow is (drives the panel's group + glyphs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WorkflowKind {
    /// `conductor.py` — one conductor GA + N subagent GAs on `127.0.0.1:8900`.
    Conductor,
    /// A Goal-Hive — a BBS board + master + worker GA subprocesses.
    Hive,
    /// Goal Mode — a single master subprocess driving `temp/goal_state.json`.
    Goal,
}

impl WorkflowKind {
    /// A stable lowercase tag (used in ids + the `serde` source descriptor).
    pub fn tag(self) -> &'static str {
        match self {
            WorkflowKind::Conductor => "conductor",
            WorkflowKind::Hive => "hive",
            WorkflowKind::Goal => "goal",
        }
    }

    /// The i18n KEY for this kind's group header in the panel (resolved by the
    /// renderer through `crate::i18n::t`). The three §7 groups: Conductor / Hives
    /// / Goal.
    pub fn group_key(self) -> &'static str {
        match self {
            WorkflowKind::Conductor => "wf.group.conductor",
            WorkflowKind::Hive => "wf.group.hives",
            WorkflowKind::Goal => "wf.group.goal",
        }
    }

    /// Panel group order (Conductor first, then Hives, then Goal — recon §7).
    pub fn order(self) -> u8 {
        match self {
            WorkflowKind::Conductor => 0,
            WorkflowKind::Hive => 1,
            WorkflowKind::Goal => 2,
        }
    }
}

/// The lifecycle status of a workflow OR a node within it. A single enum so the
/// panel's animated status hooks (shimmer running / sparkle done / lightning
/// failed) and the heat color map have one source of truth. `Unknown` is the
/// best-effort default for a hive worker whose status is only inferable from
/// post-age (recon §3.4 "worker status is prose, not a field").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WfStatus {
    /// Actively working (conductor `running`; goal `running`; a hive node that
    /// posted recently).
    Running,
    /// Idle / awaiting (conductor `stopped`; a hive node quiet a while).
    Idle,
    /// Goal Mode is in its budget wind-down (`wrapping_up`).
    WrappingUp,
    /// Finished (goal `done_budget`; a conductor subagent done its task).
    Done,
    /// Failed (conductor `failed` — reserved; an error marker in output).
    Failed,
    /// Tombstoned: a conductor subagent that VANISHED from the snapshot
    /// (aborted / auto-cleaned). The panel shows a tombstone row rather than
    /// silently dropping it (recon §5.4.5).
    Aborted,
    /// No status signal yet (a freshly-seen hive author with no age data).
    Unknown,
}

impl WfStatus {
    /// A stable lowercase tag (used in serde + tests).
    pub fn tag(self) -> &'static str {
        match self {
            WfStatus::Running => "running",
            WfStatus::Idle => "idle",
            WfStatus::WrappingUp => "wrapping_up",
            WfStatus::Done => "done",
            WfStatus::Failed => "failed",
            WfStatus::Aborted => "aborted",
            WfStatus::Unknown => "unknown",
        }
    }

    /// Map a conductor `SubAgentState.status` string onto a [`WfStatus`]
    /// (`running | stopped | failed | aborted`, recon §2.1). An unknown string
    /// degrades to `Unknown` rather than panicking. PURE.
    pub fn from_conductor(s: &str) -> WfStatus {
        match s.trim().to_ascii_lowercase().as_str() {
            "running" => WfStatus::Running,
            "stopped" => WfStatus::Idle,
            "failed" => WfStatus::Failed,
            "aborted" => WfStatus::Aborted,
            "done" => WfStatus::Done,
            _ => WfStatus::Unknown,
        }
    }

    /// Map a Goal-Mode `goal_state.status` onto a [`WfStatus`]
    /// (`running | wrapping_up | done_budget`, recon §3.3). PURE.
    pub fn from_goal(s: &str) -> WfStatus {
        match s.trim().to_ascii_lowercase().as_str() {
            "running" => WfStatus::Running,
            "wrapping_up" => WfStatus::WrappingUp,
            "done_budget" | "done" => WfStatus::Done,
            _ => WfStatus::Unknown,
        }
    }

    /// The i18n KEY for the short status label the panel paints. PURE.
    pub fn label_key(self) -> &'static str {
        match self {
            WfStatus::Running => "wf.status.running",
            WfStatus::Idle => "wf.status.idle",
            WfStatus::WrappingUp => "wf.status.wrapping_up",
            WfStatus::Done => "wf.status.done",
            WfStatus::Failed => "wf.status.failed",
            WfStatus::Aborted => "wf.status.aborted",
            WfStatus::Unknown => "wf.status.unknown",
        }
    }

    /// The status glyph the panel leads a node row with (animated by the panel's
    /// frame clock; this is the BASE glyph). PURE.
    pub fn glyph(self) -> &'static str {
        match self {
            WfStatus::Running => "⏺",
            WfStatus::Idle => "○",
            WfStatus::WrappingUp => "◐",
            WfStatus::Done => "✓",
            WfStatus::Failed => "✗",
            WfStatus::Aborted => "⊘",
            WfStatus::Unknown => "·",
        }
    }

    /// True for an actively-animating status (the shimmer/heat hook runs only for
    /// these so a static tree doesn't churn the screen). PURE.
    pub fn is_active(self) -> bool {
        matches!(self, WfStatus::Running | WfStatus::WrappingUp)
    }
}

/// The role a node plays in its workflow's tree (drives indent + the action set).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeRole {
    /// The conductor orchestrator GA (the root of a conductor tree).
    Conductor,
    /// A conductor subagent (a leaf under the conductor).
    Subagent,
    /// A Goal-Hive master (the root of a hive tree).
    Master,
    /// A Goal-Hive worker (a leaf under the master).
    Worker,
    /// A Goal-Mode master (the single node of a goal workflow).
    Goal,
}

/// One node in a workflow's tree (a subagent / worker / master). Plain data the
/// watcher fills and the panel paints. `id` is stable within its workflow so the
/// panel's focus survives a refresh; `parent` ties leaves to their root.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkflowNode {
    /// Node id, unique within its workflow (conductor short_id / worker name /
    /// `"master"`).
    pub id: String,
    /// Human label (subagent id / worker name / "master").
    pub label: String,
    /// The node's role in the tree.
    pub role: NodeRole,
    /// Best-effort status (authoritative for conductor/goal; inferred for hive).
    pub status: WfStatus,
    /// Parent node id (`None` for a root), e.g. a subagent → the conductor.
    pub parent: Option<String>,
    /// Current task prompt (truncated by the panel).
    pub prompt: String,
    /// Last activity wall-clock secs (`time.time()` from the source); `0` if none.
    pub last_activity_ts: f64,
    /// Last `<summary>` / last post excerpt / last reply (the row preview).
    pub summary: String,
    /// Best-effort token estimate for this node (conductor reply length / 4),
    /// shown as `~N tok` in the row; `0` when unknown.
    pub tokens: u64,
    /// Hive-only: number of posts this author has made (the activity count).
    pub post_count: u64,
}

impl WorkflowNode {
    /// A convenience constructor for the common node shape (the rest default).
    pub fn new(
        id: impl Into<String>,
        label: impl Into<String>,
        role: NodeRole,
        status: WfStatus,
    ) -> Self {
        WorkflowNode {
            id: id.into(),
            label: label.into(),
            role,
            status,
            parent: None,
            prompt: String::new(),
            last_activity_ts: 0.0,
            summary: String::new(),
            tokens: 0,
            post_count: 0,
        }
    }
}

/// Aggregate progress for a workflow (goal/hive carry this; conductor is
/// event-driven so it is `None`). Mirrors the `goal_state.json` budget fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorkflowProgress {
    /// Reflect turns spent so far.
    pub turns_used: u32,
    /// The turn cap (the `max_turns` the goal will stop at).
    pub max_turns: u32,
    /// Seconds elapsed since `start_time` (computed at parse time vs a clock).
    pub elapsed_sec: u64,
    /// The time budget in seconds (`budget_seconds`).
    pub budget_sec: u64,
}

impl WorkflowProgress {
    /// The completion fraction by TURNS in `0.0..=1.0` (the panel's primary bar).
    /// Falls back to the time fraction if `max_turns` is 0. PURE.
    pub fn fraction(&self) -> f64 {
        if self.max_turns > 0 {
            (self.turns_used as f64 / self.max_turns as f64).clamp(0.0, 1.0)
        } else if self.budget_sec > 0 {
            (self.elapsed_sec as f64 / self.budget_sec as f64).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// One activity-feed line (a conductor chat item or a BBS post). The detail
/// overlay paints the tail of these as the "progress feed".
#[derive(Debug, Clone, PartialEq)]
pub struct FeedItem {
    /// Wall-clock secs of the item.
    pub ts: f64,
    /// The author / role label.
    pub author: String,
    /// The item text (truncated by the overlay).
    pub text: String,
    /// BBS post id / monotonic seq (hive only; `0` otherwise).
    pub post_id: u64,
}

/// One workflow = a kind + its tree + optional progress + a feed. The merge
/// builds these from each source; the panel groups them by `kind`.
#[derive(Debug, Clone, PartialEq)]
pub struct Workflow {
    /// Stable id, e.g. `"conductor@8900"`, `"hive:tuiapp_v4"`, `"goal"`.
    pub id: String,
    /// The orchestration kind.
    pub kind: WorkflowKind,
    /// Human label (the objective's first line / a port label).
    pub title: String,
    /// The roll-up status for the whole workflow (the group/root status).
    pub status: WfStatus,
    /// A short transport/uri descriptor for the detail overlay + degrade message.
    pub source_uri: String,
    /// Aggregate progress (goal/hive); `None` for conductor.
    pub progress: Option<WorkflowProgress>,
    /// The tree nodes (root(s) first, then leaves; the panel indents by role).
    pub nodes: Vec<WorkflowNode>,
    /// The activity feed tail (newest last).
    pub feed: Vec<FeedItem>,
    /// `true` if the workflow's SERVER is reachable; `false` when down (the panel
    /// shows "not running · press X to launch" for a down, node-less workflow).
    pub running: bool,
}

impl Workflow {
    /// The total node count (for the group header `N nodes`).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Count of nodes currently `Running` (the header `M working`). PURE.
    pub fn running_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.status == WfStatus::Running).count()
    }
}

/// The whole merged picture: every workflow the watcher currently knows about.
/// `generation` bumps on every watcher push so the panel can detect a refresh
/// (and re-clamp focus). The panel groups `workflows` by `kind`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WorkflowSnapshot {
    /// Every workflow, in panel order (`kind.order()` then id).
    pub workflows: Vec<Workflow>,
    /// Monotonic refresh counter (the watcher bumps it on each merge).
    pub generation: u64,
}

impl WorkflowSnapshot {
    /// `true` when there is NOTHING to show (no source is up). The panel then
    /// paints the global empty/degraded hint. PURE.
    pub fn is_empty(&self) -> bool {
        self.workflows.is_empty()
    }

    /// Total node count across all workflows (the panel's global header). PURE.
    pub fn total_nodes(&self) -> usize {
        self.workflows.iter().map(|w| w.nodes.len()).sum()
    }

    /// Total running nodes across all workflows (the global `M working`). PURE.
    pub fn total_running(&self) -> usize {
        self.workflows.iter().map(|w| w.running_count()).sum()
    }

    /// The workflows of one kind, in stable order (the panel iterates per group).
    /// PURE.
    pub fn of_kind(&self, kind: WorkflowKind) -> Vec<&Workflow> {
        self.workflows.iter().filter(|w| w.kind == kind).collect()
    }
}

// ===========================================================================
// PURE merge / liveness / parse — the four load-bearing deliverable surfaces.
// ===========================================================================

/// Merge the three per-source workflow lists into ONE ordered [`WorkflowSnapshot`]
/// (the `workflow_snapshot_merge` deliverable). The watcher polls each source on
/// its own cadence and calls this to produce the single picture the panel renders:
///
///   * workflows are concatenated then SORTED by `(kind.order(), id)` so the panel
///     groups Conductor → Hives → Goal deterministically (recon §7);
///   * a source that is down contributes nothing (or a single `running:false`
///     placeholder the caller supplies) — merge never invents nodes;
///   * `generation` is stamped so the panel can detect the refresh.
///
/// PURE over its inputs (no clock, no I/O) — `generation` is injected.
pub fn merge_snapshots(
    conductor: Vec<Workflow>,
    hives: Vec<Workflow>,
    goal: Vec<Workflow>,
    generation: u64,
) -> WorkflowSnapshot {
    let mut workflows: Vec<Workflow> = Vec::with_capacity(conductor.len() + hives.len() + goal.len());
    workflows.extend(conductor);
    workflows.extend(hives);
    workflows.extend(goal);
    // Stable group order: by kind, then by id (so two hives sort by name).
    workflows.sort_by(|a, b| {
        a.kind
            .order()
            .cmp(&b.kind.order())
            .then_with(|| a.id.cmp(&b.id))
    });
    WorkflowSnapshot {
        workflows,
        generation,
    }
}

/// The hive-liveness heuristic (the `hive_liveness_from_age` deliverable). A BBS
/// worker has NO status field — its liveness is inferred from how long ago it last
/// posted (recon §3.4). Given the age in seconds since the node's last post:
///
///   * `< RUNNING_MAX_AGE`  (recently active)         → `Running`
///   * `< IDLE_MAX_AGE`     (quiet but plausibly live) → `Idle`
///   * otherwise            (long silent)              → `Unknown`  (best-effort,
///     NOT `Done` — the BBS can't tell finished from abandoned, recon §5.4.3)
///
/// A node that has NEVER posted (`age` from a `0` timestamp → very large) lands in
/// `Unknown`, which is correct: we have no signal. PURE.
pub fn hive_status_from_age(age_secs: f64) -> WfStatus {
    if age_secs < 0.0 {
        // A clock skew (post "in the future") → treat as just-active.
        return WfStatus::Running;
    }
    if age_secs < HIVE_RUNNING_MAX_AGE_SECS {
        WfStatus::Running
    } else if age_secs < HIVE_IDLE_MAX_AGE_SECS {
        WfStatus::Idle
    } else {
        WfStatus::Unknown
    }
}

/// A node posted within this many seconds is considered actively `Running`.
/// Chosen to comfortably exceed the worker reflect `INTERVAL=60` (recon §3.4) so a
/// worker mid-cycle isn't flapped to idle.
pub const HIVE_RUNNING_MAX_AGE_SECS: f64 = 90.0;
/// Quiet past `Running` but within this window is `Idle` (a few reflect cycles).
pub const HIVE_IDLE_MAX_AGE_SECS: f64 = 300.0;

/// Compute a hive node's status from its last-post timestamp + the current clock.
/// Thin wrapper over [`hive_status_from_age`] so the watcher passes wall-clock
/// values and the test pins the age boundaries directly. A `last_ts <= 0` (never
/// posted) yields `Unknown`. PURE over `(last_ts, now)`.
pub fn hive_status_at(last_ts: f64, now: f64) -> WfStatus {
    if last_ts <= 0.0 {
        return WfStatus::Unknown;
    }
    hive_status_from_age(now - last_ts)
}

/// Apply CLIENT-SIDE TOMBSTONES to a fresh conductor node list (the
/// `conductor_tombstone_hides_aborted` deliverable). The conductor's
/// `subagent_snapshot()` FILTERS OUT `aborted` subagents (recon §2.2 / §5.4.5), so
/// a node that was present last tick and is now ABSENT has been aborted / auto-
/// cleaned. Rather than let rows silently vanish (jarring + loses history), we keep
/// a tombstone for the disappeared node with status [`WfStatus::Aborted`] for a few
/// ticks so the panel can show it greyed-out, then drop it.
///
/// Inputs:
///   * `fresh` — the nodes in the LATEST `/subagent` snapshot (live ones only);
///   * `previous` — the nodes the panel showed LAST tick (may include tombstones);
///   * `now` / `tombstone_ttl_secs` — when a tombstone expires.
///
/// Returns the node list to SHOW: every fresh node, plus a tombstone for any
/// previously-seen LIVE node now missing (created at `now`), minus expired
/// tombstones. The order is fresh-first then tombstones (so live work stays on
/// top). PURE over its inputs (the clock is injected).
pub fn apply_conductor_tombstones(
    fresh: &[WorkflowNode],
    previous: &[WorkflowNode],
    now: f64,
    tombstone_ttl_secs: f64,
) -> Vec<WorkflowNode> {
    use std::collections::BTreeSet;
    let fresh_ids: BTreeSet<&str> = fresh.iter().map(|n| n.id.as_str()).collect();

    // 1) every fresh (live) node, with any stale tombstone status cleared (a node
    //    that REAPPEARED — e.g. a `stopped` subagent re-`running` on input — is live
    //    again, not a tombstone).
    let mut out: Vec<WorkflowNode> = fresh.to_vec();

    // 2) carry forward / create tombstones for nodes that DISAPPEARED.
    for prev in previous {
        if fresh_ids.contains(prev.id.as_str()) {
            continue; // still live → already in `out`.
        }
        if prev.status == WfStatus::Aborted {
            // An existing tombstone: keep it until its TTL elapses. We stamp the
            // tombstone's birth into `last_activity_ts` when we first create it, so
            // "age" here is `now - last_activity_ts`.
            let age = now - prev.last_activity_ts;
            if age <= tombstone_ttl_secs {
                out.push(prev.clone());
            }
            // else: expired → drop it.
        } else {
            // A live node that just vanished → mint a fresh tombstone.
            let mut tomb = prev.clone();
            tomb.status = WfStatus::Aborted;
            tomb.last_activity_ts = now; // birth stamp for TTL aging.
            out.push(tomb);
        }
    }
    out
}

/// The default conductor tombstone TTL (how long a vanished subagent lingers as a
/// greyed-out tombstone before the panel drops it).
pub const CONDUCTOR_TOMBSTONE_TTL_SECS: f64 = 30.0;

/// The raw `goal_state.json` shape (recon §3.3). Only the fields the panel needs;
/// `#[serde(default)]` so a partial / older file still parses.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct RawGoalState {
    #[serde(default)]
    pub objective: String,
    #[serde(default)]
    pub budget_seconds: f64,
    #[serde(default)]
    pub start_time: f64,
    #[serde(default)]
    pub turns_used: u32,
    #[serde(default)]
    pub max_turns: u32,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub done_prompt: String,
    /// Present only after `on_done` stamps it (recon §3.3); used to freeze elapsed.
    #[serde(default)]
    pub end_time: f64,
}

/// Parse a `temp/goal_state.json` body into a Goal [`Workflow`] with real progress
/// (the `goal_progress_parse` deliverable, recon §3.3). The single master node
/// carries `turns_used`/`max_turns`; `elapsed` is `now - start_time` while running,
/// or `end_time - start_time` once finished (so a done goal stops counting). The
/// objective's FIRST LINE becomes the title (the rest — BBS url / duty text — is
/// dropped from the label). Returns `None` on invalid JSON (the watcher then
/// reports the goal source as down). PURE over `(body, now)`.
pub fn parse_goal_state(body: &str, now: f64) -> Option<Workflow> {
    let raw: RawGoalState = serde_json::from_str(body.trim()).ok()?;
    Some(workflow_from_goal(&raw, now))
}

/// Build a Goal [`Workflow`] from an already-parsed [`RawGoalState`] + clock.
/// Split out so the watcher (which may read the struct directly) and the test
/// share one mapping. PURE.
pub fn workflow_from_goal(raw: &RawGoalState, now: f64) -> Workflow {
    let status = WfStatus::from_goal(&raw.status);
    // Elapsed: freeze at end_time once the goal is done, else live from start.
    let elapsed = if raw.end_time > raw.start_time {
        (raw.end_time - raw.start_time).max(0.0) as u64
    } else if raw.start_time > 0.0 && now >= raw.start_time {
        (now - raw.start_time) as u64
    } else {
        0
    };
    let title = first_line(&raw.objective);
    let progress = WorkflowProgress {
        turns_used: raw.turns_used,
        max_turns: raw.max_turns,
        elapsed_sec: elapsed,
        budget_sec: raw.budget_seconds.max(0.0) as u64,
    };
    let mut master = WorkflowNode::new("master", "goal master", NodeRole::Goal, status);
    master.prompt = title.clone();
    master.summary = first_line(&raw.done_prompt);
    master.last_activity_ts = if raw.end_time > 0.0 { raw.end_time } else { raw.start_time };
    Workflow {
        id: "goal".to_string(),
        kind: WorkflowKind::Goal,
        title: if title.is_empty() { "goal mode".to_string() } else { title },
        status,
        source_uri: "temp/goal_state.json".to_string(),
        progress: Some(progress),
        nodes: vec![master],
        feed: Vec::new(),
        running: true,
    }
}

/// The first non-empty line of a string, trimmed + bounded (the objective's title
/// line; recon §3.3 "objective's first line"). PURE.
pub fn first_line(s: &str) -> String {
    s.lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("")
        .chars()
        .take(120)
        .collect()
}

/// A rough token estimate from a text length (≈4 chars/token — the same cheap
/// heuristic the cost layer uses for a live estimate). PURE.
pub fn estimate_tokens(text: &str) -> u64 {
    (text.chars().count() as u64).div_ceil(4)
}

// ---------------------------------------------------------------------------
// Conductor + hive node builders (PURE mappers from the raw HTTP shapes). Kept
// here next to the schema so the watcher's `sources` module stays a thin I/O
// layer and these mappings are unit-tested without sockets.
// ---------------------------------------------------------------------------

/// One subagent in the conductor's `GET /subagent` → `{"items":[…]}` payload
/// (recon §2.2). `#[serde(default)]` everywhere so a partial item still parses.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct RawSubagent {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub prompt: String,
    #[serde(default)]
    pub reply: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub created_at: f64,
    #[serde(default)]
    pub updated_at: f64,
}

/// Build the conductor [`Workflow`] from its raw subagent items + the chat feed.
/// The tree is flat: a synthetic `conductor` root node + one node per subagent
/// (parent = the root). Status rolls up to `Running` if ANY subagent runs.
/// `port` shapes the id/uri. PURE.
pub fn workflow_from_conductor(
    port: u16,
    subs: &[RawSubagent],
    feed: Vec<FeedItem>,
    running: bool,
) -> Workflow {
    let root_id = "conductor".to_string();
    let mut nodes: Vec<WorkflowNode> = Vec::with_capacity(subs.len() + 1);

    // Subagent leaves.
    let mut any_running = false;
    let mut leaves: Vec<WorkflowNode> = Vec::with_capacity(subs.len());
    for s in subs {
        let status = WfStatus::from_conductor(&s.status);
        if status == WfStatus::Running {
            any_running = true;
        }
        let mut n = WorkflowNode::new(s.id.clone(), s.id.clone(), NodeRole::Subagent, status);
        n.parent = Some(root_id.clone());
        n.prompt = s.prompt.clone();
        n.summary = s.reply.clone();
        n.last_activity_ts = s.updated_at.max(s.created_at);
        n.tokens = estimate_tokens(&s.reply);
        leaves.push(n);
    }

    let root_status = if any_running { WfStatus::Running } else { WfStatus::Idle };
    let mut root = WorkflowNode::new(root_id, "conductor", NodeRole::Conductor, root_status);
    root.prompt = "orchestrating subagents".to_string();
    root.summary = format!("{} subagents", leaves.len());
    nodes.push(root);
    nodes.extend(leaves);

    Workflow {
        id: format!("conductor@{port}"),
        kind: WorkflowKind::Conductor,
        title: format!("conductor :{port}"),
        status: root_status,
        source_uri: format!("http://127.0.0.1:{port}/subagent"),
        progress: None,
        nodes,
        feed,
        running,
    }
}

/// Build a hive [`Workflow`] from a board's authors + per-author last-post info +
/// the feed (recon §3). Each distinct author is a node; the `master`-named author
/// (or the first, by convention) is treated as the master root; the rest are
/// workers. Status is INFERRED from each author's last-post age via
/// [`hive_status_at`] (best-effort, recon §3.4). `name` is the hive short name from
/// `temp/hive_<name>`; `port` shapes the uri. PURE over its inputs (clock injected).
pub fn workflow_from_hive(
    name: &str,
    port: u16,
    authors: &[HiveAuthor],
    feed: Vec<FeedItem>,
    progress: Option<WorkflowProgress>,
    now: f64,
    running: bool,
) -> Workflow {
    // Identify the master: an author whose name contains "master", else none →
    // every author is a worker (the panel still renders a flat list).
    let master_id = authors
        .iter()
        .find(|a| a.name.to_ascii_lowercase().contains("master"))
        .map(|a| a.name.clone());

    let mut nodes: Vec<WorkflowNode> = Vec::with_capacity(authors.len());
    let mut any_running = false;
    // Roots (master) first so the tree indents correctly.
    let mut roots: Vec<WorkflowNode> = Vec::new();
    let mut leaves: Vec<WorkflowNode> = Vec::new();
    for a in authors {
        let status = hive_status_at(a.last_ts, now);
        if status == WfStatus::Running {
            any_running = true;
        }
        let is_master = master_id.as_deref() == Some(a.name.as_str());
        let role = if is_master { NodeRole::Master } else { NodeRole::Worker };
        let mut n = WorkflowNode::new(a.name.clone(), a.name.clone(), role, status);
        n.parent = if is_master { None } else { master_id.clone() };
        n.summary = a.last_post.clone();
        n.last_activity_ts = a.last_ts;
        n.post_count = a.post_count;
        n.tokens = estimate_tokens(&a.last_post);
        if is_master {
            roots.push(n);
        } else {
            leaves.push(n);
        }
    }
    nodes.extend(roots);
    nodes.extend(leaves);

    let status = if any_running { WfStatus::Running } else { WfStatus::Idle };
    Workflow {
        id: format!("hive:{name}"),
        kind: WorkflowKind::Hive,
        title: format!("hive {name}"),
        status,
        source_uri: format!("http://127.0.0.1:{port}/?key=…"),
        progress,
        nodes,
        feed,
        running,
    }
}

/// A hive author + the activity info the watcher derives from `/count` + `/posts`
/// (recon §3.2). PURE data the [`workflow_from_hive`] mapper consumes.
#[derive(Debug, Clone, PartialEq)]
pub struct HiveAuthor {
    /// The author name (a `/authors` entry).
    pub name: String,
    /// Number of posts (`/count?author=`).
    pub post_count: u64,
    /// Last post wall-clock secs (newest `/posts?author=&limit=1`); `0` if none.
    pub last_ts: f64,
    /// Last post excerpt (the row preview).
    pub last_post: String,
}

/// A degraded (server-down) workflow placeholder so the panel can still show the
/// group with a "not running · press X to launch" hint instead of hiding it
/// entirely (recon §7 "Degrade gracefully"). `nodes` is empty + `running:false`.
/// PURE.
pub fn down_workflow(id: &str, kind: WorkflowKind, title: &str, uri: &str) -> Workflow {
    Workflow {
        id: id.to_string(),
        kind,
        title: title.to_string(),
        status: WfStatus::Unknown,
        source_uri: uri.to_string(),
        progress: None,
        nodes: Vec::new(),
        feed: Vec::new(),
        running: false,
    }
}

/// Index nodes by id (a helper for the panel's focus → node lookup). PURE.
pub fn nodes_by_id(nodes: &[WorkflowNode]) -> BTreeMap<&str, &WorkflowNode> {
    nodes.iter().map(|n| (n.id.as_str(), n)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cond_node(id: &str, status: WfStatus) -> WorkflowNode {
        WorkflowNode::new(id, id, NodeRole::Subagent, status)
    }

    /// THE deliverable test #1: `workflow_snapshot_merge`. Three per-source lists
    /// merge into ONE ordered snapshot grouped Conductor → Hives → Goal, the
    /// generation is stamped, totals roll up, and a down source contributes a
    /// placeholder (not invented nodes).
    #[test]
    fn workflow_snapshot_merge() {
        // Two hives (out of name order) + one conductor + one goal.
        let conductor = vec![workflow_from_conductor(
            8900,
            &[RawSubagent { id: "ab12".into(), status: "running".into(), reply: "working…".into(), ..Default::default() }],
            vec![],
            true,
        )];
        let hive_z = workflow_from_hive(
            "zeta",
            5001,
            &[HiveAuthor { name: "hive-master".into(), post_count: 3, last_ts: 100.0, last_post: "plan".into() }],
            vec![],
            None,
            120.0,
            true,
        );
        let hive_a = workflow_from_hive(
            "alpha",
            5002,
            &[HiveAuthor { name: "hive-worker-1".into(), post_count: 1, last_ts: 119.0, last_post: "done part".into() }],
            vec![],
            None,
            120.0,
            true,
        );
        let goal = vec![workflow_from_goal(
            &RawGoalState { objective: "ship the panel\nmore".into(), turns_used: 5, max_turns: 100, start_time: 0.0, ..Default::default() },
            640.0,
        )];

        let snap = merge_snapshots(conductor, vec![hive_z, hive_a], goal, 7);
        assert_eq!(snap.generation, 7);
        // Order: conductor (order 0) → hives (order 1, sorted by id alpha<zeta) →
        // goal (order 2).
        let kinds: Vec<WorkflowKind> = snap.workflows.iter().map(|w| w.kind).collect();
        assert_eq!(
            kinds,
            vec![WorkflowKind::Conductor, WorkflowKind::Hive, WorkflowKind::Hive, WorkflowKind::Goal]
        );
        // Within the hive group, "hive:alpha" sorts before "hive:zeta".
        let hive_ids: Vec<&str> = snap.of_kind(WorkflowKind::Hive).iter().map(|w| w.id.as_str()).collect();
        assert_eq!(hive_ids, vec!["hive:alpha", "hive:zeta"]);

        // The conductor workflow has a root + one subagent leaf (2 nodes). Both the
        // running leaf AND the synthetic root (which rolls up to Running when any
        // subagent runs) count as running → 2.
        let cond = &snap.of_kind(WorkflowKind::Conductor)[0];
        assert_eq!(cond.node_count(), 2);
        assert_eq!(cond.running_count(), 2, "running leaf + rolled-up running root");
        assert_eq!(cond.status, WfStatus::Running, "conductor rolls up to running");
        // The goal title is the objective's FIRST line only.
        assert_eq!(snap.of_kind(WorkflowKind::Goal)[0].title, "ship the panel");

        // Totals roll up across groups.
        assert_eq!(snap.total_nodes(), 2 /*conductor*/ + 1 /*master*/ + 1 /*worker*/ + 1 /*goal*/);
        assert!(snap.total_running() >= 1);

        // A down source contributes a placeholder, never invented nodes.
        let down = down_workflow("conductor@8900", WorkflowKind::Conductor, "conductor :8900", "ws://…");
        let snap2 = merge_snapshots(vec![down], vec![], vec![], 8);
        assert_eq!(snap2.workflows.len(), 1);
        assert!(!snap2.workflows[0].running);
        assert!(snap2.workflows[0].nodes.is_empty());
        assert_eq!(snap2.total_nodes(), 0);

        // An all-empty merge → an empty (but valid) snapshot.
        let empty = merge_snapshots(vec![], vec![], vec![], 9);
        assert!(empty.is_empty());
        assert_eq!(empty.generation, 9);
    }

    /// THE deliverable test #2: `hive_liveness_from_age`. With no status column,
    /// liveness comes from the last-post age: recent → Running, quiet → Idle, long
    /// silent → Unknown (never falsely Done). A never-posted node → Unknown.
    #[test]
    fn hive_liveness_from_age() {
        // Recent post → Running.
        assert_eq!(hive_status_from_age(0.0), WfStatus::Running);
        assert_eq!(hive_status_from_age(HIVE_RUNNING_MAX_AGE_SECS - 1.0), WfStatus::Running);
        // Just past the running window → Idle.
        assert_eq!(hive_status_from_age(HIVE_RUNNING_MAX_AGE_SECS + 1.0), WfStatus::Idle);
        assert_eq!(hive_status_from_age(HIVE_IDLE_MAX_AGE_SECS - 1.0), WfStatus::Idle);
        // Long silent → Unknown (best-effort; NOT Done — the BBS can't tell).
        assert_eq!(hive_status_from_age(HIVE_IDLE_MAX_AGE_SECS + 1.0), WfStatus::Unknown);
        assert_eq!(hive_status_from_age(100_000.0), WfStatus::Unknown);
        // Clock skew (future post) → treated as just-active.
        assert_eq!(hive_status_from_age(-5.0), WfStatus::Running);

        // The wall-clock wrapper: a node that posted 10s ago (now=200, last=190) is
        // Running; one that never posted (last<=0) is Unknown regardless of `now`.
        assert_eq!(hive_status_at(190.0, 200.0), WfStatus::Running);
        assert_eq!(hive_status_at(0.0, 200.0), WfStatus::Unknown);
        assert_eq!(hive_status_at(-1.0, 200.0), WfStatus::Unknown);
        // One that posted 200s ago (now=400, last=200) → Idle.
        assert_eq!(hive_status_at(200.0, 400.0), WfStatus::Idle);
        // One that posted 10 minutes ago → Unknown.
        assert_eq!(hive_status_at(0.0 + 1.0, 601.0 + 1.0), WfStatus::Unknown);
    }

    /// THE deliverable test #3: `conductor_tombstone_hides_aborted`. The conductor
    /// snapshot filters out `aborted` subagents, so a node that disappears must be
    /// kept as a TOMBSTONE (status Aborted) for a TTL — not silently dropped — then
    /// expire. A reappearing node becomes live again.
    #[test]
    fn conductor_tombstone_hides_aborted() {
        // Tick 1: two live subagents.
        let prev = vec![cond_node("a", WfStatus::Running), cond_node("b", WfStatus::Idle)];

        // Tick 2: subagent "b" VANISHED from the conductor snapshot (it was aborted
        // / auto-cleaned). "a" is still live.
        let fresh = vec![cond_node("a", WfStatus::Running)];
        let shown = apply_conductor_tombstones(&fresh, &prev, 1000.0, CONDUCTOR_TOMBSTONE_TTL_SECS);
        // "a" is shown live; "b" is shown as a TOMBSTONE (Aborted), not dropped.
        assert_eq!(shown.len(), 2);
        let a = shown.iter().find(|n| n.id == "a").unwrap();
        assert_eq!(a.status, WfStatus::Running, "live node stays live");
        let b = shown.iter().find(|n| n.id == "b").unwrap();
        assert_eq!(b.status, WfStatus::Aborted, "vanished node is tombstoned, not hidden");
        assert_eq!(b.last_activity_ts, 1000.0, "tombstone birth-stamped for TTL aging");

        // Tick 3: "b" is STILL gone but within the TTL → the tombstone persists.
        let shown2 = apply_conductor_tombstones(&fresh, &shown, 1000.0 + 10.0, CONDUCTOR_TOMBSTONE_TTL_SECS);
        assert!(shown2.iter().any(|n| n.id == "b" && n.status == WfStatus::Aborted), "tombstone persists within TTL");

        // Tick 4: past the TTL → the tombstone is finally dropped.
        let shown3 = apply_conductor_tombstones(&fresh, &shown2, 1000.0 + CONDUCTOR_TOMBSTONE_TTL_SECS + 1.0, CONDUCTOR_TOMBSTONE_TTL_SECS);
        assert!(!shown3.iter().any(|n| n.id == "b"), "tombstone expires after TTL");
        assert_eq!(shown3.len(), 1);

        // A node that REAPPEARS (e.g. a stopped subagent re-running on input) is
        // live again, not stuck as a tombstone.
        let tombstoned = vec![{ let mut n = cond_node("c", WfStatus::Aborted); n.last_activity_ts = 1000.0; n }];
        let reappeared = vec![cond_node("c", WfStatus::Running)];
        let shown4 = apply_conductor_tombstones(&reappeared, &tombstoned, 1005.0, CONDUCTOR_TOMBSTONE_TTL_SECS);
        assert_eq!(shown4.len(), 1);
        assert_eq!(shown4[0].status, WfStatus::Running, "a reappearing node is live, not a tombstone");
    }

    /// THE deliverable test #4: `goal_progress_parse`. A `goal_state.json` body
    /// parses into a Goal workflow with real `turns_used/max_turns`, elapsed from
    /// `start_time` (frozen at `end_time` once done), the objective's first line as
    /// the title, and the lifecycle status mapped. Invalid JSON → None.
    #[test]
    fn goal_progress_parse() {
        let body = r#"{
            "objective": "ship the /workflows panel\n(see BBS http://127.0.0.1:5001)\nmaster duty…",
            "budget_seconds": 10800,
            "start_time": 1000.0,
            "turns_used": 12,
            "max_turns": 200,
            "status": "running",
            "done_prompt": ""
        }"#;
        // now = start + 640s.
        let wf = parse_goal_state(body, 1640.0).expect("valid goal_state parses");
        assert_eq!(wf.kind, WorkflowKind::Goal);
        assert_eq!(wf.id, "goal");
        assert_eq!(wf.status, WfStatus::Running);
        // Title = objective's FIRST line only (BBS url + duty text dropped).
        assert_eq!(wf.title, "ship the /workflows panel");
        let p = wf.progress.expect("goal carries progress");
        assert_eq!(p.turns_used, 12);
        assert_eq!(p.max_turns, 200);
        assert_eq!(p.elapsed_sec, 640, "elapsed = now - start_time while running");
        assert_eq!(p.budget_sec, 10800);
        // Progress fraction = 12/200 = 0.06.
        assert!((p.fraction() - 0.06).abs() < 1e-9);
        // One master node carrying the title as its prompt.
        assert_eq!(wf.nodes.len(), 1);
        assert_eq!(wf.nodes[0].role, NodeRole::Goal);
        assert_eq!(wf.nodes[0].status, WfStatus::Running);

        // A DONE goal freezes elapsed at end_time - start_time (stops counting).
        let done_body = r#"{
            "objective": "wrap up",
            "budget_seconds": 100,
            "start_time": 1000.0,
            "end_time": 1050.0,
            "turns_used": 50,
            "max_turns": 50,
            "status": "done_budget"
        }"#;
        let done = parse_goal_state(done_body, 9999.0).expect("done goal parses");
        assert_eq!(done.status, WfStatus::Done);
        let dp = done.progress.unwrap();
        assert_eq!(dp.elapsed_sec, 50, "elapsed frozen at end_time despite a much later now");
        assert!((dp.fraction() - 1.0).abs() < 1e-9, "50/50 turns → full");

        // wrapping_up maps through.
        let wu = parse_goal_state(r#"{"status":"wrapping_up","start_time":1.0,"max_turns":10,"turns_used":9}"#, 2.0).unwrap();
        assert_eq!(wu.status, WfStatus::WrappingUp);

        // Invalid JSON → None (the watcher reports the goal source as down).
        assert!(parse_goal_state("not json at all", 0.0).is_none());
        assert!(parse_goal_state("", 0.0).is_none());
    }

    /// The conductor mapper builds a flat tree (root + leaves), rolls status up to
    /// Running iff any subagent runs, and estimates tokens from reply length.
    #[test]
    fn conductor_mapper_builds_flat_tree() {
        let subs = vec![
            RawSubagent { id: "aa".into(), prompt: "task A".into(), reply: "abcdefgh".into(), status: "running".into(), created_at: 1.0, updated_at: 2.0 },
            RawSubagent { id: "bb".into(), prompt: "task B".into(), reply: "done".into(), status: "stopped".into(), created_at: 1.0, updated_at: 3.0 },
        ];
        let wf = workflow_from_conductor(8900, &subs, vec![], true);
        assert_eq!(wf.id, "conductor@8900");
        assert_eq!(wf.status, WfStatus::Running, "any running subagent → conductor running");
        // Root + 2 leaves.
        assert_eq!(wf.nodes.len(), 3);
        assert_eq!(wf.nodes[0].role, NodeRole::Conductor);
        assert_eq!(wf.nodes[1].parent.as_deref(), Some("conductor"));
        // token estimate ≈ ceil(8/4) = 2 for "abcdefgh".
        assert_eq!(wf.nodes[1].tokens, 2);
        // All stopped → conductor idle.
        let idle = workflow_from_conductor(8900, &[RawSubagent { id: "c".into(), status: "stopped".into(), ..Default::default() }], vec![], true);
        assert_eq!(idle.status, WfStatus::Idle);
    }

    /// The hive mapper picks the *master* author as the root, the rest as workers,
    /// and infers each node's status from its last-post age (best-effort).
    #[test]
    fn hive_mapper_roots_master_and_infers_status() {
        let authors = vec![
            HiveAuthor { name: "hive-master".into(), post_count: 5, last_ts: 1000.0, last_post: "split tasks".into() },
            HiveAuthor { name: "hive-worker-1".into(), post_count: 2, last_ts: 990.0, last_post: "claimed #1".into() },
            HiveAuthor { name: "hive-worker-2".into(), post_count: 0, last_ts: 0.0, last_post: String::new() },
        ];
        let wf = workflow_from_hive("demo", 5001, &authors, vec![], None, 1010.0, true);
        assert_eq!(wf.id, "hive:demo");
        // Master is the root (no parent); workers reference it.
        let master = wf.nodes.iter().find(|n| n.role == NodeRole::Master).unwrap();
        assert!(master.parent.is_none());
        let w1 = wf.nodes.iter().find(|n| n.id == "hive-worker-1").unwrap();
        assert_eq!(w1.parent.as_deref(), Some("hive-master"));
        // worker-1 posted 20s ago → Running; worker-2 never posted → Unknown.
        assert_eq!(w1.status, WfStatus::Running);
        let w2 = wf.nodes.iter().find(|n| n.id == "hive-worker-2").unwrap();
        assert_eq!(w2.status, WfStatus::Unknown);
        // Master is the FIRST node (roots before leaves) so the tree indents right.
        assert_eq!(wf.nodes[0].role, NodeRole::Master);
    }

    /// Status string mappers cover the documented enums + degrade unknowns.
    #[test]
    fn status_mappers_cover_enums() {
        assert_eq!(WfStatus::from_conductor("running"), WfStatus::Running);
        assert_eq!(WfStatus::from_conductor("stopped"), WfStatus::Idle);
        assert_eq!(WfStatus::from_conductor("aborted"), WfStatus::Aborted);
        assert_eq!(WfStatus::from_conductor("failed"), WfStatus::Failed);
        assert_eq!(WfStatus::from_conductor("weird"), WfStatus::Unknown);
        assert_eq!(WfStatus::from_goal("running"), WfStatus::Running);
        assert_eq!(WfStatus::from_goal("wrapping_up"), WfStatus::WrappingUp);
        assert_eq!(WfStatus::from_goal("done_budget"), WfStatus::Done);
        // Every status has a non-empty glyph + label key + tag.
        for s in [WfStatus::Running, WfStatus::Idle, WfStatus::WrappingUp, WfStatus::Done, WfStatus::Failed, WfStatus::Aborted, WfStatus::Unknown] {
            assert!(!s.glyph().is_empty());
            assert!(!s.label_key().is_empty());
            assert!(!s.tag().is_empty());
        }
        assert!(WfStatus::Running.is_active());
        assert!(!WfStatus::Done.is_active());
    }

    #[test]
    fn first_line_and_estimate_are_pure() {
        assert_eq!(first_line("  hello\nworld"), "hello");
        assert_eq!(first_line("\n\n  second is first non-empty\nx"), "second is first non-empty");
        assert_eq!(first_line(""), "");
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
        assert_eq!(estimate_tokens(""), 0);
    }
}
