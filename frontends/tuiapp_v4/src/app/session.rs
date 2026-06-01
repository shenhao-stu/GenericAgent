//! app/session.rs — multi-session + the Claude-Code-style session DASHBOARD
//! (checklist §6 / N2).
//!
//! THE REALITY (recon `ga_core_bridge_hive.md` §4): the GA core has **no
//! multi-session API** — ACP even refuses `session/list`. So tui_v4 multiplexes
//! **N independent `ga_bridge.py` children in the UI**: one [`Bridge`] per
//! session, each its own GA-core process (and its own PID+uuid task dir, so the
//! `_intervene`/`_stop` file signals never collide — `ga_bridge.py:217-223`).
//! The UI routes each child's frames to ITS session id (never global) so two
//! concurrent agents can't clobber each other's transcript / model / status.
//!
//! THE DASHBOARD is a **separate full-screen view** (entered by left-click on the
//! sessions area OR `Ctrl+S`; `Esc` returns to chat) — NOT a sidebar crowding the
//! composer (the rejected v0.1 design). It groups sessions into three
//! **collapsible** categories and shows, per row, a status glyph + name + a LIVE
//! PREVIEW of that session's latest output line (multiplexed from its stream),
//! with a bottom "describe a task for a new session" input.
//!
//! ARCHITECTURE (mirrors the Ink reference `app/sessions.ts` + `app/sessionView
//! .ts`, ported to Rust): the heavy render-plane state (wrap cache, viewport)
//! stays on [`crate::app::AppState`] for the ACTIVE session; a [`Session`] is the
//! light per-session record (name, transcript, conn, status, input stash, live
//! preview). [`SessionMap`] owns the ordered sessions + the active id + the lazy
//! bridge children. ALL selection / category-assignment / collapse / preview /
//! nav logic lives in PURE functions at the bottom (the Ink "no-drift" discipline)
//! so it is unit-tested with no ratatui / no live python.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;

use crate::app::{Block, ConnStatus, PendingAsk, Role};
use crate::bridge::protocol::CoreToUi;
use crate::bridge::{spawn_bridge_tagged, Bridge, BridgeEvent, BridgeOptions};
use crate::flavor::heat_token;
use crate::theme::Token;

/// The lifecycle status of one chat session — drives the dashboard category, the
/// row glyph, and the heat color. Ported from the Ink `SessionStatus` vocabulary
/// (`idle | working | needs_input`) with an explicit `Done` once a session has
/// produced a reply and gone idle (so "Completed" reads as finished work, not a
/// never-used tab).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionStatus {
    /// Fresh / idle, no reply yet (a blank new session).
    Idle,
    /// A turn is streaming (running).
    Working,
    /// Blocked on a human answer (an `AskUser` is pending).
    NeedsInput,
    /// Idle again AFTER having produced at least one assistant reply (finished).
    Done,
}

/// The three dashboard categories (§6). Sessions bucket into exactly one. The
/// ORDER here is the on-screen order the eye reads top→bottom.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    /// `Needs input` — idle/awaiting user (an `AskUser` is pending).
    NeedsInput,
    /// `Working` — a turn is running.
    Working,
    /// `Completed` — idle or done (no work in flight, nothing blocking).
    Completed,
}

impl Category {
    /// The three categories in on-screen (top→bottom) order.
    pub const ALL: [Category; 3] = [Category::NeedsInput, Category::Working, Category::Completed];

    /// The i18n KEY for this category's title. PURE.
    pub fn title_key(self) -> &'static str {
        match self {
            Category::NeedsInput => "dash.needs_input",
            Category::Working => "dash.working",
            Category::Completed => "dash.completed",
        }
    }

    /// The localized category title shown in the dashboard header (§9 i18n).
    pub fn title(self, lang: crate::i18n::Lang) -> &'static str {
        crate::i18n::t(lang, self.title_key())
    }
}

/// THE category-assignment rule (the headline pure function, §6 + the
/// `dashboard_category_assignment` deliverable test):
///   running   → Working
///   awaiting  → Needs input
///   idle/done → Completed
pub fn category_for_status(status: SessionStatus) -> Category {
    match status {
        SessionStatus::Working => Category::Working,
        SessionStatus::NeedsInput => Category::NeedsInput,
        SessionStatus::Idle | SessionStatus::Done => Category::Completed,
    }
}

/// One chat session: its OWN GA-core child (lazy-spawned) + the light state the
/// dashboard + the active-session cockpit read. The render-plane wrap cache /
/// viewport are NOT here — those live on [`AppState`] for the active session and
/// are re-synced on switch (so a `Session` stays a cheap, cloneable-ish record).
#[derive(Debug)]
pub struct Session {
    /// Stable session id (monotonic; never reused, like a block id).
    pub id: u64,
    /// User-facing name (persisted under `temp/`; defaults to `session N`).
    pub name: String,
    /// This session's logical transcript (oldest → newest) — the SAME `Block`
    /// model the cockpit renders; on switch the active one feeds the wrap cache.
    pub transcript: Vec<Block>,
    /// This session's connection status (its own child handshake, N1).
    pub conn: ConnStatus,
    /// This session's model name (per-session; never a global field, so two
    /// children can't clobber each other's footer model — Ink no-cross-talk rule).
    pub model: Option<String>,
    /// This session's last-known context fill percent (per-session, like `model`);
    /// stored by the background `on_status` so it survives promotion to active.
    pub context_percent: Option<f64>,
    /// Whether a turn is in flight for THIS session.
    pub busy: bool,
    /// Monotonic ms when the current turn began (heat clock for the running row).
    pub busy_since_ms: u64,
    /// A pending `AskUser` for this session, if any (→ `NeedsInput`).
    pub pending_ask: Option<PendingAsk>,
    /// True once this session has produced at least one assistant reply (so an
    /// idle session reads as `Done`/Completed rather than a never-used `Idle`).
    pub had_reply: bool,
    /// The composer text stashed when the user switched AWAY from this session
    /// (per-session input stash, §6 / §10). Restored on switch back.
    pub input_stash: String,
    /// Monotonic block-id source for THIS session (never reused → stable anchors).
    next_block_id: u64,
    /// The live bridge child for this session, spawned lazily on first use. `None`
    /// until the first submit / explicit spawn (so opening ten tabs doesn't fork
    /// ten Pythons until they're used — the Ink LAZY-spawn rule).
    pub bridge: Option<Bridge>,
}

impl Session {
    /// A fresh, unspawned session with `id` + `name`.
    pub fn new(id: u64, name: impl Into<String>) -> Self {
        Session {
            id,
            name: name.into(),
            transcript: Vec::new(),
            conn: ConnStatus::Connecting,
            model: None,
            context_percent: None,
            busy: false,
            busy_since_ms: 0,
            pending_ask: None,
            had_reply: false,
            input_stash: String::new(),
            next_block_id: 1,
            bridge: None,
        }
    }

    /// The derived lifecycle status (the input to [`category_for_status`]). A
    /// pending ask wins (NeedsInput), then busy (Working), then Done-if-replied,
    /// else Idle. PURE over the session's own fields.
    pub fn status(&self) -> SessionStatus {
        if self.pending_ask.is_some() {
            SessionStatus::NeedsInput
        } else if self.busy {
            SessionStatus::Working
        } else if self.had_reply {
            SessionStatus::Done
        } else {
            SessionStatus::Idle
        }
    }

    /// This session's dashboard category (status → category).
    pub fn category(&self) -> Category {
        category_for_status(self.status())
    }

    /// The LIVE preview line for the dashboard row: the latest non-empty output
    /// line this session has produced (last assistant/system block's last
    /// non-blank line), falling back to the most recent user line, then to a
    /// status hint. Multiplexed straight from the per-session transcript that the
    /// bridge stream appends to (so a running row updates as deltas arrive). PURE.
    pub fn preview(&self) -> String {
        preview_line(&self.transcript, self.status())
    }

    /// Allocate the next stable block id for this session. `pub(in crate::app)`
    /// so the shared [`reducer`](crate::app::reducer) fold can mint ids.
    pub(in crate::app) fn alloc_block_id(&mut self) -> u64 {
        let id = self.next_block_id;
        self.next_block_id = self.next_block_id.wrapping_add(1);
        id
    }

    /// Append a user message to this session's transcript (on submit).
    pub fn push_user(&mut self, text: String) {
        let id = self.alloc_block_id();
        self.transcript.push(Block::new_external(id, None, Role::User, text, true));
    }

    /// Append a notice line (bridge errors / child exit) — never silent (N1).
    pub fn push_notice(&mut self, text: String) {
        let id = self.alloc_block_id();
        self.transcript
            .push(Block::new_external(id, None, Role::Notice, text, true));
    }

    /// Append a SYSTEM block (e.g. a `!cmd` echo) for this session. (Parity with
    /// `AppState::push_system`; used when a `!cmd` is run from a background tab.)
    #[allow(dead_code)]
    pub fn push_system(&mut self, text: String) {
        let id = self.alloc_block_id();
        self.transcript
            .push(Block::new_external(id, None, Role::System, text, true));
    }

    /// Fold one bridge frame into THIS session's state, through the ONE shared
    /// [`reducer::apply_frame`](crate::app::reducer) fold (the [`FrameSink`]
    /// hooks for `Session` write only to this session's own fields, so concurrent
    /// children never cross — the per-arm behavior is in `app::reducer`).
    pub fn apply_frame(&mut self, frame: CoreToUi, now_ms: u64) {
        crate::app::reducer::apply_frame(self, frame, now_ms);
    }

    /// Fold a non-frame bridge lifecycle event (spawn failure / child exit /
    /// stderr) into this (BACKGROUND) session. Mirrors the active reducer's §c
    /// transcript hygiene: a FATAL `SpawnFailed`/`ChildExited` becomes the
    /// connection STATUS (surfaced in the dashboard via `conn`, never silent — N1),
    /// while `Stderr`/`ParseNoise` (incl. `[MixinSession] …retry N/M` failover
    /// chatter) are SUPPRESSED — never a transcript row (§c).
    pub fn apply_lifecycle(&mut self, ev: &BridgeEvent) {
        match ev {
            BridgeEvent::Frame(_) => { /* handled by apply_frame */ }
            BridgeEvent::SpawnFailed { detail } => {
                self.conn = ConnStatus::Disconnected { reason: detail.clone() };
                self.busy = false;
            }
            BridgeEvent::ChildExited { code } => {
                let reason = match code {
                    Some(c) => format!("bridge exited (code {c})"),
                    None => "bridge exited".to_string(),
                };
                self.conn = ConnStatus::Disconnected { reason };
                self.busy = false;
            }
            // Stderr / parse-noise are suppressed from a background session's
            // transcript too (no `[bridge]` rows, no retry chatter — §c).
            BridgeEvent::ParseNoise { .. } | BridgeEvent::Stderr { .. } => {}
        }
    }

    /// Find the in-flight block for a mid, newest first. `pub(in crate::app)` so
    /// the shared [`reducer`](crate::app::reducer) fold can stream into it.
    pub(in crate::app) fn block_for_mid_mut(&mut self, mid: &str) -> Option<&mut Block> {
        self.transcript
            .iter_mut()
            .rev()
            .find(|b| b.mid.as_deref() == Some(mid))
    }
}

/// One dashboard row in the flattened, group-ordered list the UI navigates. A row
/// is EITHER a category header (collapsible) or a session under it. Pure data; the
/// component paints it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DashRow {
    /// A category header: the category + its member count + whether it's collapsed.
    Header { category: Category, count: usize, collapsed: bool },
    /// A session row: its id, its category (for the glyph color), and an index
    /// into [`SessionMap::sessions`] so the component can read its name/preview.
    Session { id: u64, category: Category },
}

impl DashRow {
    /// True if this row is SELECTABLE by Up/Down navigation. Headers are
    /// selectable (so the cursor can sit on one to toggle); session rows are
    /// selectable. A collapsed header's children are simply not emitted (see
    /// [`dashboard_rows_pure`]), so navigation "skips collapsed" by construction.
    #[allow(dead_code)]
    pub fn is_navigable(&self) -> bool {
        true
    }

    /// True if this row is a session row (Enter/Space/delete/rename act on these).
    #[allow(dead_code)]
    pub fn is_session(&self) -> bool {
        matches!(self, DashRow::Session { .. })
    }
}

/// The UI-multiplexer over N GA-core bridge children + the dashboard model.
///
/// Holds the ordered sessions, the active session id, the per-category collapse
/// flags, and the dashboard's selection + new-session composer text. The bridge
/// children are owned by their [`Session`] (`Session::bridge`), spawned lazily.
#[derive(Debug)]
pub struct SessionMap {
    /// All sessions, in tab/cycle order (oldest → newest).
    pub sessions: Vec<Session>,
    /// The active (focused) session id — what the cockpit renders.
    pub active: u64,
    /// Monotonic session-id source (never reused).
    next_id: u64,
    /// Which categories are collapsed in the dashboard (default: all expanded).
    collapsed: HashMap<Category, bool>,
    /// The dashboard's selected row index into [`SessionMap::dashboard_rows`].
    pub dash_sel: usize,
    /// The "describe a task for a new session" composer buffer (dashboard footer).
    pub new_session_input: String,
    /// The GA repo root (for the persisted session-names sidecar + child cwd).
    repo_root: PathBuf,
    /// Bridge options every child is spawned with (python, llm_no, …).
    bridge_opts: BridgeOptions,
}

impl SessionMap {
    /// Create the map with ONE initial session (id 1, "session 1"), active. The
    /// initial session's bridge is NOT spawned here; the caller (the app) adopts
    /// the already-spawned foundation bridge into it via [`Self::adopt_bridge`],
    /// or it spawns lazily on first submit.
    pub fn new(repo_root: PathBuf, bridge_opts: BridgeOptions) -> Self {
        let names = load_session_names(&repo_root);
        let first_name = names.get(&1).cloned().unwrap_or_else(|| "session 1".to_string());
        let mut sessions = Vec::new();
        sessions.push(Session::new(1, first_name));
        SessionMap {
            sessions,
            active: 1,
            next_id: 2,
            collapsed: HashMap::new(),
            dash_sel: 0,
            new_session_input: String::new(),
            repo_root,
            bridge_opts,
        }
    }

    /// Number of sessions. (Read by tests + the dashboard header.)
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }

    /// The active session (always present — the map is never empty while running).
    pub fn active(&self) -> &Session {
        self.session(self.active).expect("active session present")
    }

    /// The active session's display NAME (the header's `session …` field, Q7).
    /// Falls back to `"main"` if (impossibly) the active id has no record.
    pub fn active_name(&self) -> &str {
        self.session(self.active).map(|s| s.name.as_str()).unwrap_or("main")
    }

    /// The active session, mutably. (Used by tests + the branch path.)
    #[allow(dead_code)]
    pub fn active_mut(&mut self) -> &mut Session {
        let id = self.active;
        self.session_mut(id).expect("active session present")
    }

    /// A session by id.
    pub fn session(&self, id: u64) -> Option<&Session> {
        self.sessions.iter().find(|s| s.id == id)
    }

    /// A session by id, mutably.
    pub fn session_mut(&mut self, id: u64) -> Option<&mut Session> {
        self.sessions.iter_mut().find(|s| s.id == id)
    }

    /// Whether `category` is collapsed in the dashboard.
    pub fn is_collapsed(&self, category: Category) -> bool {
        self.collapsed.get(&category).copied().unwrap_or(false)
    }

    /// Toggle a category's collapsed flag (the `▸`/`▾` / Tab action). Re-clamps
    /// the dashboard selection so it never points at a now-hidden row.
    pub fn toggle_category(&mut self, category: Category) {
        let v = self.is_collapsed(category);
        self.collapsed.insert(category, !v);
        let rows = self.dashboard_rows();
        self.dash_sel = clamp_sel(self.dash_sel, rows.len());
    }

    /// The dashboard header counts: (needs_input, working, completed).
    pub fn counts(&self) -> SessionCounts {
        session_counts(&self.sessions)
    }

    /// The flattened, group-ordered dashboard rows (headers + visible sessions),
    /// honoring the per-category collapse flags. The pure builder is
    /// [`dashboard_rows_pure`]; this binds it to the live state.
    pub fn dashboard_rows(&self) -> Vec<DashRow> {
        dashboard_rows_pure(&self.sessions, &self.collapsed)
    }

    /// Move the dashboard selection by `delta` (+down / -up), SKIPPING nothing
    /// extra (collapsed children are already absent from the row list, so the
    /// cursor naturally steps over collapsed categories). Saturating at the ends.
    pub fn move_dash_sel(&mut self, delta: isize) {
        let rows = self.dashboard_rows();
        self.dash_sel = nav_select(self.dash_sel, delta, rows.len());
    }

    /// The session id under the current dashboard selection, if the selected row
    /// is a session row (Enter / Space / delete / rename target). A header row →
    /// `None`.
    pub fn selected_session_id(&self) -> Option<u64> {
        let rows = self.dashboard_rows();
        match rows.get(self.dash_sel) {
            Some(DashRow::Session { id, .. }) => Some(*id),
            _ => None,
        }
    }

    /// The category under the current dashboard selection (whether the row is a
    /// header or a session) — the Tab-toggle target.
    pub fn selected_category(&self) -> Option<Category> {
        let rows = self.dashboard_rows();
        match rows.get(self.dash_sel) {
            Some(DashRow::Header { category, .. }) => Some(*category),
            Some(DashRow::Session { category, .. }) => Some(*category),
            None => None,
        }
    }

    // -- CRUD: new / rename / delete / switch / branch -------------------------

    /// Create a new (unspawned) session and switch to it. Persists the name. The
    /// child spawns lazily on first submit (or via [`Self::ensure_child`]).
    /// Returns the new session id.
    pub fn new_session(&mut self, name: Option<String>) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        let name = name.unwrap_or_else(|| format!("session {id}"));
        self.sessions.push(Session::new(id, name));
        self.active = id;
        self.persist_names();
        id
    }

    /// Rename a session + persist the sidecar. No-op for an unknown id / blank
    /// name.
    pub fn rename(&mut self, id: u64, name: String) {
        let name = name.trim().to_string();
        if name.is_empty() {
            return;
        }
        if let Some(s) = self.session_mut(id) {
            s.name = name;
            self.persist_names();
        }
    }

    /// Delete (close) a session, KEEPING its on-disk log (the bridge child's task
    /// dir / model_responses log survive — §6 "delete (keep the log)"). Shuts down
    /// the child cleanly. If the active session is deleted, focus the nearest
    /// remaining one. NEVER removes the last session (a running app always has at
    /// least one) — instead it resets that lone session to a blank state.
    pub fn delete(&mut self, id: u64) {
        let Some(idx) = self.sessions.iter().position(|s| s.id == id) else {
            return;
        };
        if self.sessions.len() == 1 {
            // Don't leave the app session-less: shut the child + reset to blank.
            let s = &mut self.sessions[0];
            if let Some(b) = s.bridge.take() {
                b.shutdown();
            }
            let keep_name = s.name.clone();
            self.sessions[0] = Session::new(s.id, keep_name);
            self.active = self.sessions[0].id;
            self.dash_sel = clamp_sel(self.dash_sel, self.dashboard_rows().len());
            return;
        }
        // Shut its child down (the on-disk log is untouched — we only stop the
        // process; we never delete temp/model_responses).
        if let Some(b) = self.sessions[idx].bridge.take() {
            b.shutdown();
        }
        let removed_active = self.sessions[idx].id == self.active;
        self.sessions.remove(idx);
        if removed_active {
            // Focus the previous neighbour (or the new first).
            let new_idx = idx.saturating_sub(1).min(self.sessions.len() - 1);
            self.active = self.sessions[new_idx].id;
        }
        self.dash_sel = clamp_sel(self.dash_sel, self.dashboard_rows().len());
    }

    /// Switch focus to a session id (free — no spawn). `current_input` is the
    /// composer text to stash on the session we are LEAVING; the returned String
    /// is the incoming session's stashed input to load into the composer (the
    /// per-session input stash, §6/§10). No-op-ish if `id` is unknown (still
    /// stashes + returns "").
    pub fn switch(&mut self, id: u64, current_input: String) -> String {
        // Stash the outgoing session's draft.
        let leaving = self.active;
        if let Some(s) = self.session_mut(leaving) {
            s.input_stash = current_input;
        }
        if self.session(id).is_some() {
            self.active = id;
        }
        // Load the incoming session's stash (and clear it so it isn't re-applied).
        self.session_mut(self.active)
            .map(|s| std::mem::take(&mut s.input_stash))
            .unwrap_or_default()
    }

    /// Cycle the active session by `delta` (Ctrl+Up = -1 / Ctrl+Down = +1) in tab
    /// order, wrapping. `current_input` is stashed on the leaving session; returns
    /// the incoming session's stashed input. No-op (returns the same input) for a
    /// single session.
    pub fn cycle(&mut self, delta: isize, current_input: String) -> String {
        if self.sessions.len() <= 1 {
            return current_input;
        }
        let cur = self
            .sessions
            .iter()
            .position(|s| s.id == self.active)
            .unwrap_or(0);
        let n = self.sessions.len() as isize;
        let next = (((cur as isize + delta) % n) + n) % n;
        let target = self.sessions[next as usize].id;
        self.switch(target, current_input)
    }

    /// Branch the active session into a fresh one seeded with a COPY of its
    /// transcript (so the user keeps reading where they were — §6 / Ctrl+B). The
    /// fork's backend history is best-effort (a true restore needs a
    /// `model_responses` log path; the seeded blocks are display continuity).
    /// Switches to the fork. Returns the new id.
    pub fn branch(&mut self, current_input: String) -> u64 {
        let from = self.active;
        let (name, model, transcript) = {
            let s = self.active();
            (
                format!("{}*", s.name),
                s.model.clone(),
                s.transcript.clone(),
            )
        };
        // Stash the leaving draft before we move focus.
        if let Some(s) = self.session_mut(from) {
            s.input_stash = current_input;
        }
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        let mut fork = Session::new(id, name);
        fork.model = model;
        // Re-id the copied blocks onto the fork's own id space (stable anchors).
        for b in &transcript {
            let bid = fork.alloc_block_id();
            fork.transcript.push(Block::new_external(
                bid,
                None,
                b.role,
                b.source.clone(),
                true,
            ));
        }
        fork.had_reply = fork
            .transcript
            .iter()
            .any(|b| b.role == Role::Assistant && !b.source.is_empty());
        self.sessions.push(fork);
        self.active = id;
        self.persist_names();
        id
    }

    // -- bridge children (lazy spawn + routed send) ----------------------------

    /// Adopt an already-spawned [`Bridge`] into a session (the app hands the
    /// foundation bridge to session 1 so it isn't double-spawned).
    pub fn adopt_bridge(&mut self, id: u64, bridge: Bridge) {
        if let Some(s) = self.session_mut(id) {
            s.bridge = Some(bridge);
        }
    }

    /// Ensure session `id` has a live bridge child, spawning it lazily and wiring
    /// its events onto `events` TAGGED with the session id (so frames route to the
    /// right session). Returns `true` if a child now exists.
    pub fn ensure_child(&mut self, id: u64, events: &Sender<(u64, BridgeEvent)>) -> bool {
        let opts = self.bridge_opts.clone();
        let Some(s) = self.session_mut(id) else {
            return false;
        };
        if s.bridge.is_some() {
            return true;
        }
        let bridge = spawn_bridge_tagged(opts, id, events.clone());
        s.bridge = Some(bridge);
        true
    }

    /// Send a frame to a session's child, spawning it lazily first. Returns
    /// `false` if the child is dead / unspawnable (the caller surfaces a notice).
    pub fn send_to(
        &mut self,
        id: u64,
        frame: crate::bridge::protocol::UiToCore,
        events: &Sender<(u64, BridgeEvent)>,
    ) -> bool {
        if !self.ensure_child(id, events) {
            return false;
        }
        match self.session(id).and_then(|s| s.bridge.as_ref()) {
            Some(b) => b.send(frame),
            None => false,
        }
    }

    /// Shut down every session's bridge child (app exit).
    pub fn shutdown_all(&mut self) {
        for s in &mut self.sessions {
            if let Some(b) = s.bridge.take() {
                b.shutdown();
            }
        }
    }

    // -- persistence (names sidecar under temp/) -------------------------------

    /// Persist the id→name map to `temp/tui_v4_sessions.json` (best-effort). The
    /// schema is a flat `{ "<id>": "<name>" }` JSON object, the v4 analogue of the
    /// core's `session_names.json` sidecar (recon §4.2).
    pub fn persist_names(&self) {
        save_session_names(&self.repo_root, &self.sessions);
    }
}

/// The dashboard header counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SessionCounts {
    pub needs_input: usize,
    pub working: usize,
    pub completed: usize,
}

// ---------------------------------------------------------------------------
// PURE dashboard/selection logic (the load-bearing, unit-tested surface).
// ---------------------------------------------------------------------------

/// Count sessions per category. PURE over a session slice.
pub fn session_counts(sessions: &[Session]) -> SessionCounts {
    let mut c = SessionCounts::default();
    for s in sessions {
        match s.category() {
            Category::NeedsInput => c.needs_input += 1,
            Category::Working => c.working += 1,
            Category::Completed => c.completed += 1,
        }
    }
    c
}

/// Build the flattened dashboard rows: for each category in [`Category::ALL`]
/// order, a header row, then (if NOT collapsed) its member session rows in the
/// session slice's order. An EMPTY category is still shown as a header (count 0)
/// so the layout is stable and the user can see "Completed (0)". PURE.
pub fn dashboard_rows_pure(
    sessions: &[Session],
    collapsed: &HashMap<Category, bool>,
) -> Vec<DashRow> {
    let mut rows = Vec::new();
    for &cat in &Category::ALL {
        let members: Vec<&Session> = sessions.iter().filter(|s| s.category() == cat).collect();
        let is_collapsed = collapsed.get(&cat).copied().unwrap_or(false);
        rows.push(DashRow::Header {
            category: cat,
            count: members.len(),
            collapsed: is_collapsed,
        });
        if !is_collapsed {
            for s in members {
                rows.push(DashRow::Session { id: s.id, category: cat });
            }
        }
    }
    rows
}

/// Move a dashboard selection index by `delta` over `len` rows, saturating at the
/// ends (no wrap — matches the composer/palette saturating nav). Because a
/// collapsed category's children are simply absent from the row list, this
/// naturally "skips collapsed" rows. PURE — the `dashboard_nav_skips_collapsed`
/// deliverable test pins it.
pub fn nav_select(sel: usize, delta: isize, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let next = (sel as isize + delta).clamp(0, len as isize - 1);
    next as usize
}

/// Clamp a selection index to a valid row range (used after a collapse/delete
/// changes the row count). PURE.
pub fn clamp_sel(sel: usize, len: usize) -> usize {
    if len == 0 {
        0
    } else {
        sel.min(len - 1)
    }
}

/// The status glyph + theme token for a session row (the dashboard dot). Running
/// rows get a heat-colored `⏺`; needs-input a warning `◆`; idle a `○`; done a
/// success `✓`. `elapsed_ms` heat-colors the running glyph (the running-with-heat
/// requirement, §6). PURE.
pub fn status_glyph(status: SessionStatus, elapsed_ms: u64) -> (&'static str, Token) {
    match status {
        SessionStatus::NeedsInput => ("◆", Token::Warning),
        SessionStatus::Working => ("⏺", heat_token(elapsed_ms)),
        SessionStatus::Idle => ("○", Token::Dim),
        SessionStatus::Done => ("✓", Token::Success),
    }
}

/// The LIVE preview line for a session row (§6): the latest non-blank output line
/// the session has produced. Walk the transcript newest-first: prefer the last
/// non-empty line of the most recent assistant/system block; else the most recent
/// user line; else a status-appropriate hint ("send a prompt to start" when idle,
/// the question when awaiting). PURE — multiplexed from the per-session transcript
/// that the bridge stream appends to, so a running row's preview updates per delta.
pub fn preview_line(transcript: &[Block], status: SessionStatus) -> String {
    // Most recent assistant/system block → its LAST non-blank line (the "current
    // output" tail, like CC's running-row preview). A `Turn N ...` marker is spacing,
    // not content (Slice 0 / R3 companion), so skip it when picking the tail.
    for b in transcript.iter().rev() {
        if matches!(b.role, Role::Assistant | Role::System) {
            if let Some(line) = last_non_blank_content_line(&b.source) {
                return collapse_ws(line);
            }
        }
    }
    // Else the most recent user line.
    for b in transcript.iter().rev() {
        if b.role == Role::User {
            if let Some(line) = first_non_blank_line(&b.source) {
                return collapse_ws(line);
            }
        }
    }
    // Else a status hint.
    match status {
        SessionStatus::NeedsInput => "awaiting your answer".to_string(),
        SessionStatus::Working => "working…".to_string(),
        _ => "send a prompt to start".to_string(),
    }
}

/// The last non-blank, non-turn-marker line of `s` (for a running/finished reply
/// preview). A `Turn N ...` line is turn spacing, never content, so it is skipped —
/// otherwise a session whose newest output is a bare marker shows `Turn N ...` on its
/// dashboard card (the R3 raw-source preview leak). PURE.
fn last_non_blank_content_line(s: &str) -> Option<&str> {
    s.lines()
        .rev()
        .map(str::trim)
        .find(|l| !l.is_empty() && crate::render::chip::find_turn_line(l) != Some(0))
}

/// The first non-blank line of `s` (for a user-message preview). PURE.
fn first_non_blank_line(s: &str) -> Option<&str> {
    s.lines().map(str::trim).find(|l| !l.is_empty())
}

/// Collapse internal runs of whitespace to single spaces (a one-line preview).
fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ---------------------------------------------------------------------------
// Persistence helpers (the names sidecar under temp/).
// ---------------------------------------------------------------------------

/// The session-names sidecar path under a repo root.
pub fn session_names_path(repo_root: &Path) -> PathBuf {
    repo_root.join("temp").join("tui_v4_sessions.json")
}

/// Load the persisted id→name map (missing / unparsable → empty). Effectful read;
/// the parse is tolerant (a hand-edited file never panics).
pub fn load_session_names(repo_root: &Path) -> HashMap<u64, String> {
    let path = session_names_path(repo_root);
    let Ok(text) = std::fs::read_to_string(&path) else {
        return HashMap::new();
    };
    parse_session_names(&text)
}

/// Parse the names sidecar JSON `{ "<id>": "<name>" }` into a map. PURE +
/// tolerant (non-numeric keys / non-string values are skipped). Unit-tested.
pub fn parse_session_names(text: &str) -> HashMap<u64, String> {
    let mut out = HashMap::new();
    let Ok(value) = serde_json::from_str::<serde_json::Value>(text) else {
        return out;
    };
    if let Some(obj) = value.as_object() {
        for (k, v) in obj {
            if let (Ok(id), Some(name)) = (k.parse::<u64>(), v.as_str()) {
                out.insert(id, name.to_string());
            }
        }
    }
    out
}

/// Serialize sessions' id→name map to the sidecar JSON string. PURE + unit-tested.
pub fn serialize_session_names(sessions: &[Session]) -> String {
    let map: serde_json::Map<String, serde_json::Value> = sessions
        .iter()
        .map(|s| (s.id.to_string(), serde_json::Value::String(s.name.clone())))
        .collect();
    serde_json::to_string_pretty(&serde_json::Value::Object(map)).unwrap_or_else(|_| "{}".to_string())
}

/// Persist the names sidecar (best-effort; a failure is swallowed — names are a
/// convenience, not load-bearing). Effectful.
fn save_session_names(repo_root: &Path, sessions: &[Session]) {
    let path = session_names_path(repo_root);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&path, serialize_session_names(sessions));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a session in a chosen state for the pure tests (no bridge / no I/O).
    fn sess(id: u64, name: &str) -> Session {
        Session::new(id, name)
    }

    /// THE deliverable test #1: category assignment.
    /// running → Working, awaiting → Needs input, idle/done → Completed.
    #[test]
    fn dashboard_category_assignment() {
        // The pure status→category rule (the headline §6 mapping).
        assert_eq!(category_for_status(SessionStatus::Working), Category::Working);
        assert_eq!(category_for_status(SessionStatus::NeedsInput), Category::NeedsInput);
        assert_eq!(category_for_status(SessionStatus::Idle), Category::Completed);
        assert_eq!(category_for_status(SessionStatus::Done), Category::Completed);

        // And via a Session's derived status:
        // a running session → Working.
        let mut running = sess(1, "a");
        running.busy = true;
        assert_eq!(running.status(), SessionStatus::Working);
        assert_eq!(running.category(), Category::Working);

        // an awaiting session (pending ask) → Needs input (wins over busy).
        let mut awaiting = sess(2, "b");
        awaiting.busy = true;
        awaiting.pending_ask = Some(PendingAsk {
            ask_id: "x".into(),
            question: "pick?".into(),
            options: vec![],
            free_text: true,
        });
        assert_eq!(awaiting.status(), SessionStatus::NeedsInput);
        assert_eq!(awaiting.category(), Category::NeedsInput);

        // a fresh idle session → Completed bucket (Idle).
        let idle = sess(3, "c");
        assert_eq!(idle.status(), SessionStatus::Idle);
        assert_eq!(idle.category(), Category::Completed);

        // a session that replied and went idle → Done → Completed.
        let mut done = sess(4, "d");
        done.had_reply = true;
        assert_eq!(done.status(), SessionStatus::Done);
        assert_eq!(done.category(), Category::Completed);

        // Counts bucket correctly.
        let sessions = vec![running, awaiting, idle, done];
        let counts = session_counts(&sessions);
        assert_eq!(counts.working, 1);
        assert_eq!(counts.needs_input, 1);
        assert_eq!(counts.completed, 2);
    }

    /// THE deliverable test #2: navigation skips collapsed rows.
    /// When a category is collapsed, its session rows are absent from the row
    /// list, so Up/Down steps over them (lands on the next visible header).
    #[test]
    fn dashboard_nav_skips_collapsed() {
        // Two working sessions, one needs-input, one completed.
        let mut s_need = sess(1, "need");
        s_need.pending_ask = Some(PendingAsk {
            ask_id: "a".into(),
            question: "q".into(),
            options: vec![],
            free_text: true,
        });
        let mut s_work1 = sess(2, "w1");
        s_work1.busy = true;
        let mut s_work2 = sess(3, "w2");
        s_work2.busy = true;
        let s_done = sess(4, "done");
        s_done_into_completed(&s_done);
        let sessions = vec![s_need, s_work1, s_work2, s_done];

        // Expanded: rows = [H(NeedsInput), need, H(Working), w1, w2, H(Completed), done]
        let expanded = dashboard_rows_pure(&sessions, &HashMap::new());
        assert_eq!(expanded.len(), 7);
        assert!(matches!(expanded[0], DashRow::Header { category: Category::NeedsInput, count: 1, collapsed: false }));
        assert!(matches!(expanded[1], DashRow::Session { id: 1, .. }));
        assert!(matches!(expanded[2], DashRow::Header { category: Category::Working, count: 2, collapsed: false }));
        assert!(matches!(expanded[3], DashRow::Session { id: 2, .. }));
        assert!(matches!(expanded[4], DashRow::Session { id: 3, .. }));
        assert!(matches!(expanded[5], DashRow::Header { category: Category::Completed, count: 1, collapsed: false }));
        assert!(matches!(expanded[6], DashRow::Session { id: 4, .. }));

        // Collapse Working → its two session rows vanish; the header remains and
        // is marked collapsed. rows = [H(NeedsInput), need, H(Working,collapsed), H(Completed), done]
        let mut collapsed = HashMap::new();
        collapsed.insert(Category::Working, true);
        let rows = dashboard_rows_pure(&sessions, &collapsed);
        assert_eq!(rows.len(), 5);
        assert!(matches!(rows[2], DashRow::Header { category: Category::Working, count: 2, collapsed: true }));
        // The row AFTER the collapsed Working header is the Completed header — NOT
        // a Working session row (navigation skips the collapsed children).
        assert!(matches!(rows[3], DashRow::Header { category: Category::Completed, .. }));

        // Navigating from the Working header (index 2) DOWN lands on the Completed
        // header (index 3), skipping the (now-collapsed) working sessions.
        let from_working_header = 2usize;
        let next = nav_select(from_working_header, 1, rows.len());
        assert_eq!(next, 3);
        assert!(matches!(rows[next], DashRow::Header { category: Category::Completed, .. }));

        // nav saturates at the ends (no wrap).
        assert_eq!(nav_select(0, -1, rows.len()), 0);
        assert_eq!(nav_select(rows.len() - 1, 1, rows.len()), rows.len() - 1);
        // Empty list → 0, never panics.
        assert_eq!(nav_select(3, 1, 0), 0);
    }

    /// Helper: assert a fresh session lands in Completed (Idle bucket).
    fn s_done_into_completed(s: &Session) {
        assert_eq!(s.category(), Category::Completed);
    }

    /// THE deliverable test #3: per-session input stash round-trips on switch.
    /// Switching away stashes the composer text on the leaving session; switching
    /// back restores it (and clears the stash so it isn't double-applied).
    #[test]
    fn session_stash_roundtrip() {
        let root = std::env::temp_dir().join(format!("tui_v4_smap_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("temp")).unwrap();

        let mut map = SessionMap::new(root.clone(), BridgeOptions::default());
        // Two sessions: 1 (initial) + a new one (2), now active.
        let s2 = map.new_session(Some("second".into()));
        assert_eq!(map.active, s2);

        // On session 2 the user types a draft, then switches to session 1. The
        // draft is stashed on 2; session 1 had no stash → composer comes back "".
        let restored_for_1 = map.switch(1, "draft on two".to_string());
        assert_eq!(restored_for_1, "");
        assert_eq!(map.active, 1);
        // The stash lives on session 2.
        assert_eq!(map.session(s2).unwrap().input_stash, "draft on two");

        // The user types on session 1, then switches BACK to 2 → 2's stash is
        // restored into the composer, and 1's new draft is now stashed.
        let restored_for_2 = map.switch(s2, "draft on one".to_string());
        assert_eq!(restored_for_2, "draft on two");
        assert_eq!(map.active, s2);
        assert_eq!(map.session(1).unwrap().input_stash, "draft on one");
        // The restored stash on 2 was cleared (so a second switch doesn't re-apply).
        assert_eq!(map.session(s2).unwrap().input_stash, "");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn preview_prefers_latest_output_then_user_then_hint() {
        // Empty idle session → the idle hint.
        let s = sess(1, "x");
        assert_eq!(s.preview(), "send a prompt to start");

        // A user message but no reply → the user line.
        let mut s = sess(1, "x");
        s.push_user("please summarize the repo".into());
        assert_eq!(s.preview(), "please summarize the repo");

        // An assistant block → its LAST non-blank line (the running tail).
        let id = s.alloc_block_id();
        s.transcript.push(Block::new_external(
            id,
            Some("m1".into()),
            Role::Assistant,
            "## Summary\n\n全部就绪。最终汇报".into(),
            false,
        ));
        assert_eq!(s.preview(), "全部就绪。最终汇报");

        // Whitespace is collapsed to single spaces.
        assert_eq!(collapse_ws("a   b\t c"), "a b c");
    }

    #[test]
    fn status_glyph_running_is_heat_colored() {
        // Running → ⏺ with a heat token that escalates with elapsed time.
        let (g_calm, t_calm) = status_glyph(SessionStatus::Working, 0);
        let (g_hot, t_hot) = status_glyph(SessionStatus::Working, 200_000);
        assert_eq!(g_calm, "⏺");
        assert_eq!(g_hot, "⏺");
        assert_ne!(t_calm, t_hot, "the running glyph re-colors with heat");
        // The other categories use their fixed semantic tokens.
        assert_eq!(status_glyph(SessionStatus::NeedsInput, 0), ("◆", Token::Warning));
        assert_eq!(status_glyph(SessionStatus::Idle, 0), ("○", Token::Dim));
        assert_eq!(status_glyph(SessionStatus::Done, 0), ("✓", Token::Success));
    }

    #[test]
    fn crud_new_rename_delete_switch_branch() {
        let root = std::env::temp_dir().join(format!("tui_v4_crud_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("temp")).unwrap();
        let mut map = SessionMap::new(root.clone(), BridgeOptions::default());
        assert_eq!(map.len(), 1);

        // new + switch
        let id2 = map.new_session(None);
        assert_eq!(map.len(), 2);
        assert_eq!(map.active, id2);

        // rename + persist round-trips through the sidecar.
        map.rename(id2, "renamed".into());
        assert_eq!(map.session(id2).unwrap().name, "renamed");
        let reloaded = load_session_names(&root);
        assert_eq!(reloaded.get(&id2).map(String::as_str), Some("renamed"));

        // branch the active session: seeds a copy of its transcript, switches.
        map.active_mut().push_user("hello".into());
        let forked = map.branch(String::new());
        assert_eq!(map.active, forked);
        assert!(map.session(forked).unwrap().name.ends_with('*'));
        assert_eq!(map.session(forked).unwrap().transcript.len(), 1);

        // delete the active fork → focus falls back to a neighbour.
        map.delete(forked);
        assert!(map.session(forked).is_none());
        assert_ne!(map.active, forked);

        // Deleting down to the last session never removes it (resets to blank).
        let ids: Vec<u64> = map.sessions.iter().map(|s| s.id).collect();
        for id in ids {
            map.delete(id);
        }
        assert_eq!(map.len(), 1, "the last session is never removed");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn cycle_wraps_and_stashes() {
        let root = std::env::temp_dir().join(format!("tui_v4_cyc_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("temp")).unwrap();
        let mut map = SessionMap::new(root.clone(), BridgeOptions::default());
        let s2 = map.new_session(None); // active = s2
        let s3 = map.new_session(None); // active = s3
        assert_eq!(map.active, s3);

        // Cycle +1 wraps 3→1; stashes the draft on s3, returns s1's stash ("").
        let restored = map.cycle(1, "draft3".into());
        assert_eq!(map.active, 1);
        assert_eq!(restored, "");
        assert_eq!(map.session(s3).unwrap().input_stash, "draft3");

        // Cycle -1 wraps 1→3 (back to s3); restores s3's stash.
        let restored = map.cycle(-1, String::new());
        assert_eq!(map.active, s3);
        assert_eq!(restored, "draft3");

        let _ = s2;
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn toggle_category_collapses_and_clamps_selection() {
        let root = std::env::temp_dir().join(format!("tui_v4_tog_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("temp")).unwrap();
        let mut map = SessionMap::new(root.clone(), BridgeOptions::default());
        // Initial: one idle session → Completed has 1 member.
        assert!(!map.is_collapsed(Category::Completed));
        // Select the last row, then collapse Completed → selection re-clamps.
        let rows = map.dashboard_rows();
        map.dash_sel = rows.len() - 1;
        map.toggle_category(Category::Completed);
        assert!(map.is_collapsed(Category::Completed));
        assert!(map.dash_sel < map.dashboard_rows().len());
        // Toggling back expands.
        map.toggle_category(Category::Completed);
        assert!(!map.is_collapsed(Category::Completed));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn session_names_sidecar_round_trips() {
        let s1 = sess(1, "alpha");
        let s2 = sess(7, "beta gamma");
        let json = serialize_session_names(&[s1, s2]);
        let parsed = parse_session_names(&json);
        assert_eq!(parsed.get(&1).map(String::as_str), Some("alpha"));
        assert_eq!(parsed.get(&7).map(String::as_str), Some("beta gamma"));
        // Tolerant of garbage.
        assert!(parse_session_names("not json").is_empty());
        assert!(parse_session_names(r#"{"x":1}"#).is_empty()); // non-numeric key, non-string val.
    }

    #[test]
    fn per_session_frames_never_cross() {
        // Two sessions; a frame applied to one must not touch the other.
        let mut a = sess(1, "a");
        let mut b = sess(2, "b");
        a.apply_frame(
            CoreToUi::MessageBegin { mid: "m1".into(), role: "assistant".into() },
            100,
        );
        a.apply_frame(CoreToUi::MessageDelta { mid: "m1".into(), text: "hi from a".into() }, 100);
        assert!(a.busy);
        assert_eq!(a.preview(), "hi from a");
        // b is untouched.
        assert!(!b.busy);
        assert_eq!(b.transcript.len(), 0);

        // Per-session model: a Ready on b sets only b's model.
        b.apply_frame(CoreToUi::Ready { version: None, model: Some("glm-b".into()), llm: None, model_real: None }, 0);
        assert_eq!(b.model.as_deref(), Some("glm-b"));
        assert_eq!(a.model, None);
    }
}
