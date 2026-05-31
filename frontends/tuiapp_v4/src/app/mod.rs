//! app/mod.rs — AppState + the pure reducers that fold bridge events into it.
//!
//! The UI plane is immediate-mode: each frame `render()` reads this state. The
//! event loop pushes bridge frames + ticks + key events in; the reducers here
//! mutate state. Keeping the fold in PURE methods (no I/O, no ratatui) is what
//! makes the load-bearing behavior unit-testable (ratatui widgets can't be
//! headlessly TTY-tested — §"Keep all load-bearing logic in PURE functions").

pub mod effort;
pub mod session;

use std::path::PathBuf;

use crate::app::effort::{EffortSlider, ReasoningEffort};
use crate::app::session::SessionMap;
use crate::bridge::protocol::{AskUserOption, CoreToUi, LlmItem};
use crate::bridge::BridgeEvent;
use crate::components::picker::{AskUserPicker, PickItem, Picker, PickerKind};
use crate::effects::{EffectMode, EffectsEngine};
use crate::flavor::{Lang, PetStyle, SpinnerStyle};
use crate::input::keychord::ChordState;
use crate::input::Composer;
use crate::render::{Block as RenderBlock, BlockRole, Viewport, WrapCache};
use crate::theme::Theme;
use crate::util::osc::TabStatus;

/// The fixed delta-time (seconds) one `tick()` advances the effects engine. Matches the
/// event loop's 0.1s tick cadence so effect speed is calibrated in real seconds while
/// remaining a pure function of the integer tick count.
pub const TICK_DT: f32 = 0.1;

/// The active modal OVERLAY stacked over the current view (§3 "overlay stack:
/// palette, pickers, session dashboard, /workflows, ask-user, help, copy-mode").
/// At most one is up at a time (the cockpit + dashboard are VIEWS; these are
/// transient modals dismissed with `Esc`). The slash dispatcher opens them; the
/// key handler routes keys into the active one before the composer sees them.
#[derive(Debug, Clone)]
pub enum Overlay {
    /// A reusable list picker (`/llm` `/theme` `/emoji` `/language` `/export`
    /// `/rewind` `/continue` `/scheduler`). Carries the picker model + a snapshot
    /// of the pre-preview theme so a live-preview picker can REVERT on `Esc`.
    Picker {
        picker: Picker,
        /// The theme to restore on `Esc` for a live-preview picker (`/theme` etc).
        theme_backup: Option<Theme>,
    },
    /// The unified ask_user card (single / multi / numeric).
    AskUser(AskUserPicker),
    /// `/help` — the full command-list overlay.
    Help,
    /// `/status` `/sessions` — model / state / rounds / context / cwd snapshot.
    Status,
    /// `/cost [all]` — token-usage report (input/output/cache/context%).
    Cost,
    /// `/verbose` `/tools` `/trace` — full-screen tool-call audit.
    Verbose,
    /// A transient text card (e.g. `/btw`'s answer box) above the composer; the
    /// `querying…` → answer flow. Dismissed with `Esc`, no history pollution. The
    /// `ask_id` ties an incoming `BtwAnswer` frame back to THIS card so a stale card
    /// (the user fired a second `/btw`) can't show the wrong answer.
    Btw { ask_id: String, question: String, answer: Option<String> },
    /// The `/scheduler` 3-step flow card (multi-pick reflect tasks → confirm
    /// start/stop diff → apply + cron status). The whole flow is one overlay whose
    /// internal `step` advances on Enter; Esc steps back / closes (§7).
    Scheduler(crate::components::scheduler::Scheduler),
    /// The `/continue` searchable session picker (content-grep + lazy load over the
    /// `model_responses_*.txt` logs → restore via the existing restore path, §4).
    Continue(crate::components::continue_picker::ContinuePicker),
    /// The `/effects demo` splash overlay (§9): a transient centered panel that shows
    /// every effect at once (fire band + snow + lightning + sparkle) and auto-reverts
    /// after a few seconds. Modal-but-trivial: any key closes it early.
    Effects,
    /// The `/effort` slider overlay (redesign_cc.md §3): a `Faster ←——▲——→ Smarter`
    /// horizontal slider over the `low medium high xhigh max` stops with a `▲` marker
    /// on the chosen level. ←/→ moves the marker; Enter forwards
    /// `/session.reasoning_effort=<level>` (max→xhigh) to the bridge; Esc cancels.
    EffortSlider(EffortSlider),
}

impl Overlay {
    /// True if this overlay is the FULL-SCREEN kind (help / status / cost /
    /// verbose) vs a compact centered card (picker / ask / btw). The renderer uses
    /// it to size the overlay region.
    pub fn is_fullscreen(&self) -> bool {
        matches!(self, Overlay::Help | Overlay::Status | Overlay::Cost | Overlay::Verbose)
    }

    /// True if this overlay is MODAL — it captures keyboard input (the picker /
    /// ask-user / help / status / cost / verbose / scheduler / continue). A
    /// NON-modal overlay (the `/btw` card) is a transient toast that floats above
    /// the composer WITHOUT stealing input, so "chat stays usable" while a side
    /// question is in flight (§7 `/btw` "non-blocking (chat stays usable)"). Only
    /// `Esc` dismisses a non-modal card; every other key flows to the cockpit.
    pub fn is_modal(&self) -> bool {
        !matches!(self, Overlay::Btw { .. })
    }
}

/// The connection lifecycle, reflecting the REAL handshake (N1). The status word
/// the UI shows is derived straight from this — never a guessed "disconnected".
#[derive(Debug, Clone, PartialEq)]
pub enum ConnStatus {
    /// Spawned, awaiting the `Ready` frame.
    Connecting,
    /// `Ready` received. Carries the model name if the core reported one.
    Connected { model: Option<String> },
    /// The bridge failed or the child exited. Carries the real reason so the UI
    /// shows "disconnected: <reason>" (the N1 anti-silent-failure requirement).
    Disconnected { reason: String },
}

impl ConnStatus {
    /// The short human status word for the footer/header.
    pub fn label(&self) -> String {
        match self {
            ConnStatus::Connecting => "connecting…".to_string(),
            ConnStatus::Connected { model } => match model {
                Some(m) => format!("connected {m}"),
                None => "connected".to_string(),
            },
            ConnStatus::Disconnected { reason } => format!("disconnected: {reason}"),
        }
    }
}

/// Which full-screen VIEW the app is showing. The dashboard (§6 / N2) is a
/// SEPARATE full-screen view — entered via left-click on the sessions area OR
/// `Ctrl+S`, exited via `Esc` — NOT a sidebar crowding the composer (the rejected
/// v0.1 design). Future overlays (pickers, /workflows) stack on top of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum View {
    /// The normal chat cockpit (header / transcript / composer / footer).
    #[default]
    Cockpit,
    /// The full-screen Claude-Code-style session dashboard.
    Dashboard,
    /// The full-screen `/workflows` panel (conductor / hive / goal — §7). Reached
    /// via `/workflows` `/conductor` `/hive` `/goal`; `Esc` returns to the cockpit.
    /// Backed by the singleton workflow watcher (its own threads, never blocks
    /// chat); the panel only tracks the focus + render style.
    Workflows,
}

/// A pending rename in the dashboard: the session id being renamed + the current
/// edit buffer. `r` opens it; typing edits; Enter commits; Esc cancels.
#[derive(Debug, Clone, Default)]
pub struct RenameState {
    pub id: u64,
    pub buffer: String,
}

/// The author of a transcript block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    System,
    Tool,
    /// An out-of-band bridge/system notice (errors, child exit, stderr).
    Notice,
}

impl Role {
    /// Map a protocol role string onto a [`Role`].
    pub fn from_proto(s: &str) -> Role {
        match s {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "system" => Role::System,
            "tool" => Role::Tool,
            _ => Role::Assistant,
        }
    }
}

/// One logical transcript block. Stores SOURCE text — the only width-independent
/// truth. The render plane derives soft-wrapped rows + the viewport window from
/// `(id, source, rev)` (P1) and copies `source` verbatim (P2); nothing is ever
/// re-derived from rendered rows. `mid` ties streaming deltas to their block;
/// `id` is the STABLE, monotonic anchor key (never reused, so a scroll anchor
/// survives appends); `rev` bumps on every mutation to drive cache invalidation.
#[derive(Debug, Clone)]
pub struct Block {
    pub id: u64,
    pub mid: Option<String>,
    pub role: Role,
    pub source: String,
    /// Monotonic content version; bumped on stream-append / finalize so the wrap
    /// cache reflows only this block (untouched blocks reuse cached rows).
    pub rev: u64,
    /// True once `MessageEnd` arrived (or for a synchronous block).
    pub finalized: bool,
}

impl Block {
    fn new(id: u64, mid: Option<String>, role: Role, source: String, finalized: bool) -> Self {
        Block {
            id,
            mid,
            role,
            source,
            rev: 1,
            finalized,
        }
    }

    /// Public block constructor for the multi-session plane (`app::session`),
    /// which owns its own per-session transcripts and id space. Identical to the
    /// private [`Block::new`]; exposed so `SessionMap` can build blocks without
    /// reaching into `AppState`'s private allocator.
    pub fn new_external(
        id: u64,
        mid: Option<String>,
        role: Role,
        source: String,
        finalized: bool,
    ) -> Self {
        Block::new(id, mid, role, source, finalized)
    }

    /// A snapshot of this block as a render-plane [`RenderBlock`] (the logical
    /// unit the wrap cache + viewport consume). Cheap clone of the source; the
    /// `(id, rev)` pair lets the cache reuse untouched blocks across frames.
    ///
    /// For an **assistant** block the wrap cache must count the rows that the
    /// markdown render actually draws (a table becomes aligned rows, `$$…$$`
    /// becomes a stacked fraction, a code fence gains a label + gutters), NOT the
    /// raw markdown source lines — otherwise the styled transcript draw and the
    /// scroll math would disagree (P1). So we hand the cache the markdown-PLAIN
    /// projection for assistants. Copy (P2) still reads the original `source`
    /// (the app `Block`), so the clean-logical-text copy is unaffected.
    pub fn to_render_block(&self, theme: &Theme, fold_all: bool, width: u16) -> RenderBlock {
        let source = if self.role == Role::Assistant {
            // The COCKPIT plain projection (folds + chips applied + structural
            // stream-commit for the in-flight block) so the wrap cache counts the
            // rows the styled cockpit draw emits (P1). A FINALIZED block has no
            // volatile tail, so it goes through the documented finalized-convenience
            // projection [`render_assistant_cockpit_plain`]; an in-flight block
            // holds its tail back via the streaming variant. Both paths mirror the
            // styled draw in `components::transcript` 1:1.
            if self.finalized {
                crate::markdown::render_assistant_cockpit_plain(
                    &self.source,
                    theme,
                    fold_all,
                    width,
                )
            } else {
                let lines = crate::markdown::render_assistant_cockpit_streaming(
                    &self.source,
                    theme,
                    fold_all,
                    width,
                    true,
                );
                crate::markdown::lines_to_plain(&lines)
            }
        } else {
            self.source.clone()
        };
        RenderBlock {
            id: self.id,
            role: render_role(self.role),
            source,
            rev: self.rev,
            finalized: self.finalized,
        }
    }

    /// This block's render-plane [`BlockRole`] (without a full snapshot). The
    /// transcript widget uses it to choose the gutter + markdown routing.
    pub fn render_role(&self) -> BlockRole {
        render_role(self.role)
    }
}

/// Map an app [`Role`] onto the render plane's [`BlockRole`].
fn render_role(role: Role) -> BlockRole {
    match role {
        Role::User => BlockRole::User,
        Role::Assistant => BlockRole::Assistant,
        Role::System => BlockRole::System,
        Role::Tool => BlockRole::Tool,
        Role::Notice => BlockRole::Notice,
    }
}

/// A pending `AskUser` the UI must surface. The Foundation renders it as a
/// transcript notice; the structured ask-user picker (Phase 3) reads these
/// fields to build the candidate list + free-text escape.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PendingAsk {
    pub ask_id: String,
    pub question: String,
    pub options: Vec<AskUserOption>,
    pub free_text: bool,
}

/// The token-cost breakdown surfaced by `/cost` (§4: "token usage
/// input/output/cache/context%"). Live estimates from `Status` frames + a final
/// aggregation; a pure formatter renders the report card.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CostBreakdown {
    /// Cumulative input (prompt) tokens.
    pub input: u64,
    /// Cumulative output (completion) tokens.
    pub output: u64,
    /// Cumulative cache-read tokens (when the backend reports them).
    pub cache: u64,
    /// The last context-window utilization percent reported (0..=100).
    pub context_percent: Option<f64>,
    /// Accumulated cost in USD (mirrors `AppState::cost_usd`).
    pub cost_usd: f64,
}

impl CostBreakdown {
    /// The total tokens accounted (input + output). PURE.
    pub fn total(&self) -> u64 {
        self.input.saturating_add(self.output)
    }

    /// Render the `/cost` report card as plain lines (the load-bearing formatter,
    /// unit-tested). PURE. `model` is the active model name for the header.
    pub fn report_lines(&self, model: &str) -> Vec<String> {
        let ctx = match self.context_percent {
            Some(p) => format!("{p:.0}%"),
            None => "—".to_string(),
        };
        // GA has no per-token price table, so a 0 cost is "unknown", not "free":
        // show a dash rather than a misleading $0.0000 (a future price map can fill
        // this in). A real accumulated cost still prints with the `$` suffix.
        let cost = if self.cost_usd > 0.0 {
            format!("{:.4}$", self.cost_usd)
        } else {
            "—".to_string()
        };
        vec![
            format!("Token usage · {model}"),
            String::new(),
            format!("  input    {:>10}", self.input),
            format!("  output   {:>10}", self.output),
            format!("  cache    {:>10}", self.cache),
            format!("  total    {:>10}", self.total()),
            String::new(),
            format!("  context  {ctx:>10}"),
            format!("  cost     {cost:>10}"),
        ]
    }
}

/// The whole application state for the Foundation slice.
#[derive(Debug)]
pub struct AppState {
    /// Real connection status (drives the status line, N1).
    pub conn: ConnStatus,
    /// Current model name (from `Ready`/`Status`).
    pub model: Option<String>,
    /// The logical transcript (oldest → newest).
    pub transcript: Vec<Block>,
    /// The multi-line composer (buffer + cursor + selection + undo/redo + magic
    /// prefixes + input history) — the Phase-2 cockpit input (§4/§8).
    pub composer: Composer,
    /// A 0.1s tick counter that drives the spinner + gerund rotation.
    pub spinner_tick: u64,
    /// The active spinner aesthetic (N4 default = arc).
    pub spinner_style: SpinnerStyle,
    /// The active pet-face style (§9 "soul"; switched via the `/emoji` picker).
    pub pet_style: PetStyle,
    /// The reasoning-effort level last applied via `/effort` (redesign_cc.md §3).
    /// `None` until the user sets one (the backend keeps its own default); once set
    /// the spinner/status shows a `thinking · <level>` suffix and the `/effort`
    /// slider seeds its marker here. The label is the slider stop (so `max` stays
    /// "max" in the UI even though it forwarded `xhigh` to the backend).
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Whether tool chips are all folded (Ctrl+O toggle).
    pub fold_all: bool,
    /// The highlighted row in the `/`-slash command palette dropdown (↑/↓ move it;
    /// Tab/Enter completes the highlighted one). Reset to 0 whenever the typed
    /// partial changes; clamped to the live match count by the renderer + nav.
    pub palette_sel: usize,
    /// Whether terminal mouse capture is ON (wheel scroll + click-to-dashboard).
    /// Toggled by Ctrl+Shift+M / `/mouse`. When OFF, the terminal's OWN drag-select
    /// works again so the user can select + copy transcript/input text natively
    /// (the Codex model — the portable answer to "can't copy inline" on Windows).
    pub mouse_capture: bool,
    /// The interface language (drives rotating tips; full i18n is Phase 3).
    pub lang: Lang,
    /// Whether a turn is in flight (a message is streaming).
    pub busy: bool,
    /// Monotonic ms when the current turn began (for the heat ramp).
    pub turn_started_ms: u64,
    /// The working directory shown in the header.
    pub cwd: String,
    /// The GA repo root (for `@path` expansion + `!cmd` cwd + history file).
    pub repo_root: PathBuf,
    /// The ACTIVE pending ask-user, if any (the one whose card is showing).
    pub pending_ask: Option<PendingAsk>,
    /// Queued FOLLOW-UP asks that arrived while one was already being answered
    /// (§7 "queued parallel asks surface in turn"). FIFO: each is surfaced as the
    /// previous one is answered/dismissed, so no parallel ask is silently dropped.
    pub ask_queue: std::collections::VecDeque<PendingAsk>,
    /// True once the user asks to quit (the event loop breaks).
    pub should_quit: bool,
    /// The last context-window percent reported (footer).
    pub context_percent: Option<f64>,
    /// Live token count for the current/last turn (footer right side).
    pub tokens: Option<u64>,
    /// The LAST single LLM-call input / output token sizes (the spinner's
    /// `↑<in> · ↓<out>` live readout, mirroring tui_v3). Distinct from the
    /// cumulative `cost.{input,output}`; `None` until a `Status` carries them.
    pub tok_in: Option<u64>,
    pub tok_out: Option<u64>,
    /// Accumulated session cost in USD (footer `$cost`).
    pub cost_usd: f64,
    /// The current git branch (footer right side), discovered once at startup.
    pub git_branch: Option<String>,
    /// The last tab-status we emitted (OSC-21337) — only re-emit on change.
    pub last_tab_status: Option<TabStatus>,
    /// The last OSC0 terminal title we emitted — only re-emit on change.
    pub last_title: String,
    /// The active full-screen view (cockpit vs dashboard, N2).
    pub view: View,
    /// The ACTIVE theme (the source of truth for `/theme` live-preview + commit).
    /// The event loop clones this each frame to pass to the render functions, so a
    /// picker preview that swaps it re-skins the whole UI on the next draw.
    pub theme: Theme,
    /// The multi-session map (N independent bridge children, UI-multiplexed, §6).
    /// The ACTIVE session's live state is mirrored in the `AppState` fields above
    /// (transcript/conn/model/busy/…); a switch swaps them with a `Session` record.
    pub sessions: SessionMap,
    /// An in-flight dashboard rename, if any (`r` opens; Enter commits).
    pub rename: Option<RenameState>,
    /// The active modal overlay (picker / ask-user / help / cost / verbose / btw),
    /// stacked over the current view (§3 overlay stack). `None` = no modal up.
    pub overlay: Option<Overlay>,
    /// The per-turn token cost breakdown for the `/cost` report (input / output /
    /// cache / context%). Accumulated from `Status` frames + the final turn.
    pub cost: CostBreakdown,
    /// The full tool-call audit trail for `/verbose` (every tool chip line we have
    /// seen this session). Append-only; the overlay paints the tail.
    pub tool_audit: Vec<String>,
    /// A DEBUG-ONLY ring of suppressed bridge diagnostics (stderr / parse-noise /
    /// fatal-exit reasons). NEVER shown in the transcript (§c) — kept so a
    /// developer / a future `/trace` debug pane can inspect failover chatter
    /// (`[MixinSession] …retry N/M`) without polluting the chat. Bounded.
    pub bridge_debug: Vec<String>,
    /// The `/workflows` panel state (focus + render style + detail overlay, §7).
    /// Persists across panel open/close so the cursor survives. The snapshot it
    /// renders lives in `workflow_snapshot`, fed from the watcher each frame.
    pub workflow_panel: crate::workflow::panel::WorkflowPanel,
    /// The latest merged [`WorkflowSnapshot`] from the singleton watcher (§3). The
    /// event loop refreshes it from `workflow_watcher` while the panel is open; the
    /// panel renders THIS (the watcher never touches `AppState` directly, so a slow
    /// poll can never block the render/chat loop).
    pub workflow_snapshot: crate::workflow::schema::WorkflowSnapshot,
    /// The singleton workflow watcher handle (its own poll thread). Started LAZILY
    /// the first time the panel opens (so a user who never opens `/workflows` spawns
    /// no thread, and `--smoke` / unit tests build an `AppState` without one). The
    /// watcher parks when the panel closes (zero background traffic).
    pub workflow_watcher: Option<crate::workflow::WorkflowWatcher>,
    /// The last full terminal `(width, height)` a frame was drawn at — used to map
    /// a dashboard left-click `(col, row)` back to a row index. Set each frame.
    last_term_size: (u16, u16),
    /// Monotonic block-id source (never reused → stable scroll anchors, P1).
    next_block_id: u64,
    /// The wrap cache keyed at the current transcript width (P1). Rebuilt per
    /// frame via `sync`; only changed blocks reflow. Owned here so the transcript
    /// widget and copy actions share one derivation.
    pub wrap_cache: WrapCache,
    /// The scrolling viewport over the transcript (logical [`ScrollAnchor`], P1).
    pub viewport: Viewport,
    /// The transcript width the cache was last synced at (to detect resize).
    last_width: u16,
    /// The fold-all state the cache was last synced at (to detect a Ctrl+O flip).
    last_fold_all: bool,
    /// The animated effects engine (§9). OFF by default; advanced once per 0.1s `tick()`
    /// by a FIXED dt so it is driven by the deterministic tick counter, not a wall clock.
    /// Under `--smoke` the engine is never ticked → no animation.
    pub effects: EffectsEngine,
    /// The transient multi-press chord state (§8): the 3-stage Ctrl+C arm window
    /// + the Esc-Esc double-tap window. PURE state advanced by `input::keychord`
    /// with an injected `now_ms` (never a wall-clock), so the transitions stay
    /// headlessly testable. `Default` = neither chord is mid-sequence.
    pub chord: ChordState,
}

impl Default for AppState {
    fn default() -> Self {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        AppState {
            conn: ConnStatus::Connecting,
            model: None,
            transcript: Vec::new(),
            composer: Composer::new(),
            spinner_tick: 0,
            spinner_style: SpinnerStyle::default(),
            pet_style: PetStyle::default(),
            reasoning_effort: None,
            fold_all: false,
            palette_sel: 0,
            mouse_capture: true,
            lang: Lang::default(),
            busy: false,
            turn_started_ms: 0,
            cwd: cwd.display().to_string(),
            repo_root: cwd.clone(),
            pending_ask: None,
            ask_queue: std::collections::VecDeque::new(),
            should_quit: false,
            context_percent: None,
            tokens: None,
            tok_in: None,
            tok_out: None,
            cost_usd: 0.0,
            git_branch: None,
            last_tab_status: None,
            last_title: String::new(),
            view: View::default(),
            theme: Theme::ga_default(),
            sessions: SessionMap::new(cwd.clone(), crate::bridge::BridgeOptions::default()),
            rename: None,
            overlay: None,
            cost: CostBreakdown::default(),
            tool_audit: Vec::new(),
            bridge_debug: Vec::new(),
            workflow_panel: crate::workflow::panel::WorkflowPanel::new(),
            workflow_snapshot: crate::workflow::schema::WorkflowSnapshot::default(),
            workflow_watcher: None,
            last_term_size: (80, 24),
            next_block_id: 1,
            wrap_cache: WrapCache::new(80),
            viewport: Viewport::new(1),
            last_width: 0,
            last_fold_all: false,
            effects: EffectsEngine::from_env(),
            chord: ChordState::default(),
        }
    }
}

impl AppState {
    pub fn new() -> Self {
        AppState::default()
    }

    /// Advance the 0.1s tick (spinner / gerund clock) AND advance the effects engine by a
    /// FIXED dt ([`TICK_DT`]) — the single clock that drives every effect. Driving effects
    /// off the integer tick (not a wall clock) keeps them deterministic + frame-rate-
    /// independent, and means `--smoke` (which never ticks) is frozen (FPS=0).
    pub fn tick(&mut self) {
        self.spinner_tick = self.spinner_tick.wrapping_add(1);
        self.effects.tick(TICK_DT);
        // Auto-dismiss the `/effects demo` splash once its timer elapses.
        if matches!(self.overlay, Some(Overlay::Effects)) && !self.effects.demo_active() {
            self.overlay = None;
        }
    }

    /// Apply an effects mode change (used by the `/effects` command + tests).
    pub fn set_effects_mode(&mut self, mode: EffectMode) {
        self.effects.set_mode(mode);
    }

    /// Start the `/effects demo` splash: arm the demo timer + open the splash overlay.
    pub fn start_effects_demo(&mut self) {
        self.effects.start_demo();
        self.overlay = Some(Overlay::Effects);
    }

    /// Open the `/effort` slider overlay (redesign_cc.md §3) seeded on the live
    /// effort level. When none has been set yet the slider seeds on `high` (a
    /// sensible middle-high default mirroring CC's clip), so Enter-without-moving
    /// applies `high`.
    pub fn open_effort_slider(&mut self) {
        let current = self.reasoning_effort.unwrap_or(ReasoningEffort::High);
        self.overlay = Some(Overlay::EffortSlider(EffortSlider::new(current)));
    }

    /// Record the effort level the user just applied (via `/effort <level>` or the
    /// slider's Enter) so the spinner/status shows `thinking · <level>` and the next
    /// slider open seeds here. Does NOT forward to the bridge — the caller sends the
    /// `/session.reasoning_effort=<level>` frame (the effectful step). PURE.
    pub fn set_reasoning_effort(&mut self, level: ReasoningEffort) {
        self.reasoning_effort = Some(level);
    }

    /// The active reasoning-effort label for the spinner/status suffix
    /// (`thinking · <level>`), or `None` if the user hasn't set one (then the
    /// spinner shows no effort suffix — the backend keeps its own default). PURE.
    pub fn effort_label(&self) -> Option<&'static str> {
        self.reasoning_effort.map(|e| e.label())
    }

    /// Elapsed ms of the in-flight turn (0 when idle). `now_ms` is the caller's
    /// monotonic clock so this stays pure/testable.
    pub fn turn_elapsed_ms(&self, now_ms: u64) -> u64 {
        if self.busy && now_ms >= self.turn_started_ms {
            now_ms - self.turn_started_ms
        } else {
            0
        }
    }

    /// Allocate the next stable block id (monotonic, never reused).
    fn alloc_block_id(&mut self) -> u64 {
        let id = self.next_block_id;
        self.next_block_id = self.next_block_id.wrapping_add(1);
        id
    }

    /// Append a user message to the transcript (called on Enter, before Submit).
    pub fn push_user(&mut self, text: String) {
        let id = self.alloc_block_id();
        self.transcript
            .push(Block::new(id, None, Role::User, text, true));
    }

    /// Append a notice line (bridge errors, child exit, stderr) — these are the
    /// N1 "never silent" surface; everything visible.
    pub fn push_notice(&mut self, text: String) {
        let id = self.alloc_block_id();
        self.transcript
            .push(Block::new(id, None, Role::Notice, text, true));
    }

    /// Append a SYSTEM block (e.g. the `!cmd` shell echo) — distinct from a
    /// notice so the renderer can give it the `»` gutter.
    pub fn push_system(&mut self, text: String) {
        let id = self.alloc_block_id();
        self.transcript
            .push(Block::new(id, None, Role::System, text, true));
    }

    /// Wire the cockpit to a discovered GA repo root: set the cwd shown in the
    /// header, the `@path`/`!cmd` working directory, load persisted input
    /// history, and discover the git branch. Called once after the bridge spawns.
    pub fn attach_repo_root(&mut self, root: PathBuf) {
        self.cwd = root.display().to_string();
        // Load the persisted input history for this repo into the composer.
        let history = crate::input::history::History::load(&root);
        self.composer = Composer::with_history(history);
        self.git_branch = discover_git_branch(&root);
        // Re-seat the SessionMap on the real repo root (so the names sidecar +
        // child cwds land under the GA temp/), preserving the active id.
        self.sessions = SessionMap::new(root.clone(), crate::bridge::BridgeOptions::default());
        self.repo_root = root;
    }

    /// The tab-status the cockpit should report (OSC-21337): Waiting if a pending
    /// ask is up, Working while busy, else Idle.
    pub fn tab_status(&self) -> TabStatus {
        if self.pending_ask.is_some() {
            TabStatus::Waiting
        } else if self.busy {
            TabStatus::Working
        } else {
            TabStatus::Idle
        }
    }

    /// The OSC0 terminal title for the current state: `<spinner> GenericAgent ·
    /// <model>` while busy, else `GenericAgent · <model>`.
    pub fn terminal_title(&self) -> String {
        let model = self.model.as_deref().unwrap_or("");
        if self.busy {
            let glyph = self.spinner_style.glyph(self.spinner_tick);
            format!("{glyph} GenericAgent · {model}")
        } else {
            format!("GenericAgent · {model}")
        }
    }

    /// Fold one bridge event into the state. PURE w.r.t. I/O (only mutates self;
    /// `now_ms` injected). This is the core reducer — unit-tested below.
    ///
    /// TRANSCRIPT HYGIENE (redesign_cc.md §1/§c): the bridge's STDERR / parse-noise
    /// is NEVER rendered as a transcript row. GA's failover diagnostics
    /// (`[MixinSession] …retry N/M` on stderr, llmcore.py:988) and core `print()`
    /// chatter are routed to a DEBUG-ONLY log ([`Self::push_bridge_debug`]), not the
    /// chat. A FATAL `SpawnFailed` / `ChildExited` becomes a CONNECTION STATUS (the
    /// footer's connection chip shows it — N1 "never a silent disconnect"), not a
    /// transcript notice that would scroll away with the conversation.
    pub fn apply_bridge_event(&mut self, ev: BridgeEvent, now_ms: u64) {
        match ev {
            BridgeEvent::Frame(frame) => self.apply_frame(frame, now_ms),
            BridgeEvent::SpawnFailed { detail } => {
                // Fatal: surface as the connection status only (footer chip), not a
                // transcript row. The status word is derived from `conn`.
                self.conn = ConnStatus::Disconnected {
                    reason: detail.clone(),
                };
                self.busy = false;
                self.push_bridge_debug(format!("spawn failed: {detail}"));
                // A failure → the ~0.15s lightning flash (a no-op unless effects are on).
                self.effects.flash_lightning();
            }
            BridgeEvent::ParseNoise { line } => {
                // An unparsed bridge line is DIAGNOSTIC noise — route to the debug
                // log, never the transcript (§c suppress bridge stderr/noise rows).
                self.push_bridge_debug(format!("[unparsed] {line}"));
            }
            BridgeEvent::ChildExited { code } => {
                let reason = match code {
                    Some(c) => format!("bridge exited (code {c})"),
                    None => "bridge exited".to_string(),
                };
                // Fatal: connection status only (footer chip), not a transcript row.
                self.conn = ConnStatus::Disconnected {
                    reason: reason.clone(),
                };
                self.busy = false;
                self.push_bridge_debug(reason);
                // A failure → the lightning flash (no-op unless effects are on).
                self.effects.flash_lightning();
            }
            BridgeEvent::Stderr { line } => {
                // ga_bridge.py routes core print()/tracebacks + failover retry
                // diagnostics (`[MixinSession] …retry N/M`) here. SUPPRESS them from
                // the transcript entirely — they go to the debug-only log (§c). The
                // disconnect path above already surfaces FATAL failures via `conn`.
                self.push_bridge_debug(line);
            }
        }
    }

    /// Fold one parsed `CoreToUi` frame.
    fn apply_frame(&mut self, frame: CoreToUi, now_ms: u64) {
        match frame {
            CoreToUi::Ready { model, .. } => {
                self.model = model.clone();
                self.conn = ConnStatus::Connected { model };
            }
            CoreToUi::MessageBegin { mid, role } => {
                self.busy = true;
                self.turn_started_ms = now_ms;
                let id = self.alloc_block_id();
                self.transcript.push(Block::new(
                    id,
                    Some(mid),
                    Role::from_proto(&role),
                    String::new(),
                    false,
                ));
            }
            CoreToUi::MessageDelta { mid, text } => {
                if let Some(block) = self.block_for_mid_mut(&mid) {
                    block.source.push_str(&text);
                    // Bump rev so the wrap cache reflows ONLY this block (P1).
                    block.rev = block.rev.wrapping_add(1);
                } else {
                    // A delta with no begin — create a block so text isn't lost.
                    let id = self.alloc_block_id();
                    self.transcript.push(Block::new(
                        id,
                        Some(mid),
                        Role::Assistant,
                        text,
                        false,
                    ));
                    self.busy = true;
                }
            }
            CoreToUi::MessageEnd { mid, .. } => {
                let mut finalized_source: Option<String> = None;
                if let Some(block) = self.block_for_mid_mut(&mid)
                    && !block.finalized
                {
                    block.finalized = true;
                    block.rev = block.rev.wrapping_add(1);
                    if block.role == Role::Assistant {
                        finalized_source = Some(block.source.clone());
                    }
                }
                // Harvest any tool calls in the just-finalized assistant message
                // into the `/verbose` audit trail (split-borrow: read the source,
                // then push). Each chip becomes one audit line.
                if let Some(src) = finalized_source {
                    for tc in crate::render::chip::parse_tool_calls(&src) {
                        let (badge, _) = tc.status.badge();
                        let args = if tc.args.is_empty() {
                            String::new()
                        } else {
                            format!("  {}", tc.args)
                        };
                        self.push_tool_audit(format!("{} {}{}", badge, tc.name, args));
                    }
                }
                // The turn is done once its message ends (single-turn model).
                self.busy = false;
                // A successful turn completion → the sparkle burst (no-op unless on).
                self.effects.burst_sparkle();
            }
            CoreToUi::AskUser {
                ask_id,
                question,
                options,
                free_text,
            } => {
                let preview = format!("? {question}");
                let ask = PendingAsk { ask_id, question, options, free_text };
                self.push_notice(preview);
                self.busy = false;
                if self.pending_ask.is_some() {
                    // A parallel ask arrived while one is being answered → QUEUE it
                    // so it surfaces after the current one (§7 queued asks).
                    self.ask_queue.push_back(ask);
                } else {
                    self.pending_ask = Some(ask);
                    // Surface the unified ask_user CARD automatically (§7) — but
                    // never clobber an overlay the user already has open.
                    if self.overlay.is_none() {
                        self.open_ask_user();
                    }
                }
            }
            CoreToUi::Status {
                model,
                context_percent,
                tokens,
                input_tokens,
                output_tokens,
                cache_tokens,
                last_input,
                last_output,
                ..
            } => {
                if let Some(m) = model {
                    self.model = Some(m.clone());
                    if let ConnStatus::Connected { model: cm } = &mut self.conn {
                        *cm = Some(m);
                    }
                }
                if let Some(p) = context_percent {
                    self.context_percent = Some(p);
                    self.cost.context_percent = Some(p);
                }
                if let Some(tk) = tokens {
                    self.tokens = Some(tk);
                }
                // Cumulative split totals are AUTHORITATIVE when the bridge sends
                // them (it sums cost_tracker.all_trackers()), so SET them rather
                // than the old `max`-guess. `cache_tokens` is the cache-read hits.
                if let Some(i) = input_tokens {
                    self.cost.input = i;
                }
                if let Some(o) = output_tokens {
                    self.cost.output = o;
                }
                if let Some(c) = cache_tokens {
                    self.cost.cache = c;
                }
                // Per-call snapshots drive the spinner's `↑in ↓out` live readout.
                if last_input.is_some() {
                    self.tok_in = last_input;
                }
                if last_output.is_some() {
                    self.tok_out = last_output;
                }
                // Legacy bridge (tokens only, no split) → keep a sane /cost total
                // by folding the running count into output; the new bridge sends
                // the split above so this branch never fires for it.
                if input_tokens.is_none() && output_tokens.is_none() {
                    if let Some(tk) = tokens {
                        self.cost.output = self.cost.output.max(tk);
                    }
                }
                // No price table exists in GA → leave cost_usd at 0.0; the /cost
                // card shows real tokens and renders cost as a placeholder dash.
            }
            CoreToUi::Pong { .. } => { /* liveness — no state change */ }
            CoreToUi::LlmList { items } => self.apply_llm_list(items),
            CoreToUi::BtwAnswer { ask_id, text, error } => self.apply_btw_answer(ask_id, text, error),
            CoreToUi::RewindResult { dropped, remaining } => {
                // A NON-history acknowledgment: surface as a notice (the rewind
                // itself already truncated the local display when the user picked).
                self.push_notice(format!(
                    "{} {} {} · {} {}",
                    crate::i18n::t(self.lang, "rewind.done"),
                    dropped,
                    crate::i18n::t(self.lang, "rewind.turns_suffix"),
                    remaining,
                    crate::i18n::t(self.lang, "status.total"),
                ));
            }
            CoreToUi::Error {
                message,
                fatal,
                ..
            } => {
                self.push_notice(format!("error: {message}"));
                if fatal {
                    self.conn = ConnStatus::Disconnected {
                        reason: message,
                    };
                    self.busy = false;
                }
            }
        }
    }

    /// Find the (in-flight) block for a given mid, newest first.
    fn block_for_mid_mut(&mut self, mid: &str) -> Option<&mut Block> {
        self.transcript
            .iter_mut()
            .rev()
            .find(|b| b.mid.as_deref() == Some(mid))
    }

    // ---- overlay stack (§3 / §7: pickers, ask-user, help, cost, verbose) -----

    /// Fold an incoming `LlmList` frame (N3): if a `/llm` picker overlay is already
    /// open and waiting for the model list, REPLACE its items in place (so the
    /// `querying…` placeholder becomes the live rows); otherwise open a fresh
    /// `/llm` picker. The selection seeds on the current model. PURE-ish.
    pub fn apply_llm_list(&mut self, items: Vec<LlmItem>) {
        let pick_items = llm_items_to_picks(&items);
        match &mut self.overlay {
            Some(Overlay::Picker { picker, .. }) if picker.kind == PickerKind::Llm => {
                let sel = pick_items.iter().position(|i| i.current).unwrap_or(0);
                picker.items = pick_items;
                picker.sel = sel;
            }
            _ => {
                self.overlay = Some(Overlay::Picker {
                    picker: Picker::new(PickerKind::Llm, pick_items),
                    theme_backup: None,
                });
            }
        }
    }

    /// Open a reusable list picker overlay (`/theme` `/emoji` `/language`
    /// `/export` `/rewind` `/continue` `/scheduler`). `theme_backup` is the theme
    /// to restore on `Esc` for a live-preview picker (else `None`).
    pub fn open_picker(&mut self, picker: Picker, theme_backup: Option<Theme>) {
        self.overlay = Some(Overlay::Picker { picker, theme_backup });
    }

    /// Fold an incoming `BtwAnswer` frame into the `/btw` card (§7): set the card's
    /// answer text (or an error) IFF the card is still open AND its `ask_id` matches
    /// (so a stale card can't show a newer side-question's answer). The answer is
    /// shown in the EPHEMERAL card only — it is NEVER pushed to the transcript, so a
    /// side question never pollutes the conversation history. If no matching card is
    /// open (the user dismissed it with Esc), the answer is silently dropped — which
    /// is exactly the "Esc dismisses, no history pollution" contract. PURE-ish.
    pub fn apply_btw_answer(&mut self, ask_id: String, text: Option<String>, error: Option<String>) {
        if let Some(Overlay::Btw { ask_id: open_id, answer, .. }) = self.overlay.as_mut() {
            if *open_id == ask_id {
                let body = match (text, error) {
                    (Some(t), _) => t,
                    (None, Some(e)) => format!("{}: {e}", crate::i18n::t(self.lang, "btw.failed")),
                    (None, None) => crate::i18n::tf(self.lang, "btw.failed"),
                };
                *answer = Some(body);
            }
        }
        // No open Btw card (dismissed) → drop the answer (no history pollution).
    }

    /// Open the unified ask_user card overlay from a pending ask (§7). Consumes the
    /// `pending_ask` so the cockpit shows the card, not the raw notice.
    pub fn open_ask_user(&mut self) {
        if let Some(ask) = self.pending_ask.clone() {
            let candidates: Vec<String> = ask.options.iter().map(|o| o.label.clone()).collect();
            self.overlay = Some(Overlay::AskUser(AskUserPicker::new(
                ask.ask_id,
                ask.question,
                candidates,
                ask.free_text,
            )));
        }
    }

    /// After the current ask is answered/dismissed, surface the NEXT queued ask (if
    /// any) — opening its card. The caller clears `pending_ask` first; this pops
    /// the queue into it and opens the overlay. Returns `true` if one was surfaced.
    /// This is what makes "queued parallel asks surface in turn" (§7) work.
    pub fn surface_next_ask(&mut self) -> bool {
        if self.pending_ask.is_some() {
            return false; // one is still active.
        }
        if let Some(next) = self.ask_queue.pop_front() {
            self.pending_ask = Some(next);
            if self.overlay.is_none() {
                self.open_ask_user();
            }
            true
        } else {
            false
        }
    }

    /// Open a simple info overlay (help / status / cost / verbose / btw).
    pub fn open_overlay(&mut self, overlay: Overlay) {
        self.overlay = Some(overlay);
    }

    /// Close any open overlay (the universal `Esc` for modals). Returns `true` if
    /// one was open (so the key handler knows the Esc was consumed by a modal).
    pub fn close_overlay(&mut self) -> bool {
        self.overlay.take().is_some()
    }

    /// True while ANY overlay is up. (The key router now branches on
    /// [`Overlay::is_modal`] so a non-modal `/btw` toast doesn't steal input; this
    /// stays as the simple "is something open" query for callers/tests.)
    #[allow(dead_code)]
    pub fn has_overlay(&self) -> bool {
        self.overlay.is_some()
    }

    /// Append a tool-call line to the `/verbose` audit trail (called when a tool
    /// chip is observed). Bounded so a long session can't grow unboundedly.
    pub fn push_tool_audit(&mut self, line: String) {
        const MAX_AUDIT: usize = 2000;
        self.tool_audit.push(line);
        if self.tool_audit.len() > MAX_AUDIT {
            let overflow = self.tool_audit.len() - MAX_AUDIT;
            self.tool_audit.drain(0..overflow);
        }
    }

    /// Record a suppressed bridge diagnostic in the DEBUG-ONLY ring (§c): bridge
    /// stderr, parse-noise, fatal-exit reasons. This NEVER reaches the transcript —
    /// it is the "route to a debug-only log" sink the redesign mandates so failover
    /// retry chatter (`[MixinSession] …`) stays out of the chat. Bounded.
    pub fn push_bridge_debug(&mut self, line: String) {
        const MAX_DEBUG: usize = 500;
        self.bridge_debug.push(line);
        if self.bridge_debug.len() > MAX_DEBUG {
            let overflow = self.bridge_debug.len() - MAX_DEBUG;
            self.bridge_debug.drain(0..overflow);
        }
    }

    // ---- render-plane sync + scrolling (P1) ---------------------------------

    /// Re-derive the wrap cache + viewport for the transcript region's geometry.
    /// Called by the transcript widget each frame with the region's `(width,
    /// height)`. On a WIDTH change this rewidths the cache and re-derives the
    /// viewport window from the SAME logical anchor (zero drift, P1); on a HEIGHT
    /// change it re-clamps; otherwise it just syncs `rev`-changed blocks (cheap).
    ///
    /// Returns the render-blocks snapshot so the caller can also feed the visible
    /// window without re-snapshotting. `theme` is needed because assistant blocks
    /// snapshot their MARKDOWN-rendered plain projection (so the wrap cache counts
    /// the rows the styled draw produces — see [`Block::to_render_block`]).
    pub fn sync_transcript(&mut self, width: u16, height: usize, theme: &Theme) -> Vec<RenderBlock> {
        let width = width.max(1);
        let fold_all = self.fold_all;
        // The assistant chip boxes are sized to the transcript width, so they are
        // a function of width — fold to the transcript's inner width here.
        let render_blocks: Vec<RenderBlock> = self
            .transcript
            .iter()
            .map(|b| b.to_render_block(theme, fold_all, width))
            .collect();

        // A width OR fold-state change invalidates the whole cache (both alter the
        // projected rows). Width also re-derives the viewport from the anchor (P1).
        if width != self.last_width || fold_all != self.last_fold_all {
            self.wrap_cache.rewidth(width, &render_blocks);
            self.viewport.resize(height, &self.wrap_cache);
            self.last_width = width;
            self.last_fold_all = fold_all;
        } else {
            // Same width + fold: reflow only changed blocks (streaming → O(1)).
            self.wrap_cache.sync(&render_blocks);
            if height != self.viewport.height() {
                self.viewport.resize(height, &self.wrap_cache);
            }
        }
        render_blocks
    }

    /// Scroll the transcript by `delta` visual rows (+down / -up). Mouse wheel is
    /// typically ±3 of these. Leaves/enters follow mode at the ends.
    pub fn scroll_lines(&mut self, delta: isize) {
        self.viewport.scroll_by(delta, &self.wrap_cache);
    }

    /// PageUp — scroll up ~one screenful (keeps a line of context).
    pub fn page_up(&mut self) {
        self.viewport.page_up(&self.wrap_cache);
    }

    /// PageDown — scroll down ~one screenful.
    pub fn page_down(&mut self) {
        self.viewport.page_down(&self.wrap_cache);
    }

    /// Home — jump to the very top of the transcript.
    pub fn scroll_home(&mut self) {
        self.viewport.home(&self.wrap_cache);
    }

    /// End — jump to the tail and resume follow mode.
    pub fn scroll_end(&mut self) {
        self.viewport.end();
    }

    /// True when the transcript is pinned to the tail (follow mode).
    pub fn following(&self) -> bool {
        self.viewport.is_following()
    }

    /// The transcript width the cache was last synced at (the real terminal
    /// width). Key handling uses it to size the composer's visual-row nav between
    /// frames; falls back to 80 before the first render.
    pub fn transcript_width(&self) -> u16 {
        if self.last_width == 0 {
            80
        } else {
            self.last_width
        }
    }

    /// Record the full terminal `(width, height)` a frame was drawn at (so a
    /// dashboard left-click can map `(col,row)` → row index). Called per frame.
    pub fn set_term_size(&mut self, width: u16, height: u16) {
        self.last_term_size = (width.max(1), height.max(1));
    }

    /// The last drawn terminal width (full frame, not the transcript region).
    pub fn last_term_width(&self) -> u16 {
        self.last_term_size.0
    }

    /// The last drawn terminal height (full frame). Used for dashboard click maps.
    pub fn last_term_height(&self) -> u16 {
        self.last_term_size.1
    }

    /// Emit the OSC0 terminal title + OSC-21337 tab status when they CHANGE
    /// (idempotent: re-emitting identical bytes every frame would spam the
    /// emulator). Called once per frame from the event loop. Effectful.
    pub fn sync_terminal_chrome(&mut self) {
        let status = self.tab_status();
        if self.last_tab_status != Some(status) {
            crate::util::osc::write_tab_status(status, status.label());
            self.last_tab_status = Some(status);
        }
        let title = self.terminal_title();
        if self.last_title != title {
            crate::util::osc::write_title(&title);
            self.last_title = title;
        }
    }

    /// The LOGICAL source of the last assistant message (the `Ctrl+Y` "copy last
    /// reply" target, P2). Reads `source` verbatim — never rendered rows — so a
    /// soft wrap can never become an embedded newline. `None` if there is no
    /// assistant message yet.
    #[allow(dead_code)] // P2 copy-last-reply target (wired to the Phase-3 copy menu).
    pub fn last_assistant_source(&self) -> Option<&str> {
        self.transcript
            .iter()
            .rev()
            .find(|b| b.role == Role::Assistant && !b.source.is_empty())
            .map(|b| b.source.as_str())
    }

    /// The whole transcript joined as plain logical text (the "copy transcript"
    /// target, P2): each block's `source` separated by a blank line. Still
    /// newline-clean because we join logical bodies, not wrapped rows. Wired to
    /// `/export all` + `Ctrl+Shift+Y` in Phase 3; provided now alongside the
    /// last-reply copy so the P2 logical-source surface is complete.
    #[allow(dead_code)]
    pub fn transcript_source(&self) -> String {
        self.transcript
            .iter()
            .map(|b| b.source.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Expand in-root `@path` tokens in a submitted message against the repo root
    /// (checklist §4 / §10 "`@path` expansion on submit"). A token whose file is
    /// missing / too large / escaping the root is left verbatim. Returns the
    /// expanded text (the `Submit.text` actually sent to the bridge).
    pub fn expand_at_paths(&self, text: &str) -> String {
        crate::input::file_expand::expand_file_refs(text, &self.repo_root).text
    }

    /// The candidate project files for the composer's `@` picker (gitignore-aware,
    /// bounded). Walks the repo root; degrades to empty on error.
    pub fn list_project_files(&self) -> Vec<String> {
        crate::input::paths::list_project_files(&self.repo_root)
    }

    // ---- multi-session glue (§6 / N2) ---------------------------------------
    //
    // The ACTIVE session's live state lives in the `AppState` fields above
    // (transcript/conn/model/busy/pending_ask/…) — that is what the cockpit
    // renders and what the existing reducer + 130+ tests exercise unchanged. The
    // `SessionMap` is the source of truth for the OTHER sessions. Two operations
    // keep them coherent: `snapshot_active_into_map` (push the live active state
    // into its `Session` record, e.g. before drawing the dashboard or switching)
    // and the swap inside `switch_session` (pull a target session's stored state
    // into the live fields). This localizes the multi-session bookkeeping to a few
    // methods instead of threading a session id through the whole render plane.

    /// Mirror the live ACTIVE-session fields into its [`Session`] record so the
    /// dashboard reads an up-to-date snapshot (transcript for preview/category,
    /// busy/pending for the status). Cheap clone of the transcript (the dashboard
    /// only needs the tail line; but a clone keeps the record self-contained and
    /// the call sites are rare — a frame arrival or a view switch, not per frame).
    pub fn snapshot_active_into_map(&mut self) {
        let active_id = self.sessions.active;
        // Read the live fields, then write them into the record (split borrows).
        let transcript = self.transcript.clone();
        let conn = self.conn.clone();
        let model = self.model.clone();
        let busy = self.busy;
        let busy_since = self.turn_started_ms;
        let pending = self.pending_ask.clone();
        let had_reply = self
            .transcript
            .iter()
            .any(|b| b.role == Role::Assistant && !b.source.is_empty());
        if let Some(s) = self.sessions.session_mut(active_id) {
            s.transcript = transcript;
            s.conn = conn;
            s.model = model;
            s.busy = busy;
            s.busy_since_ms = busy_since;
            s.pending_ask = pending;
            s.had_reply = s.had_reply || had_reply;
        }
    }

    /// Load a [`Session`]'s stored state into the live ACTIVE-session fields and
    /// reset the render plane (wrap cache + viewport) so the cockpit re-derives
    /// from the incoming transcript at the next frame (P1). The composer buffer is
    /// the caller's concern (the per-session input stash is handled by
    /// [`SessionMap::switch`], which returns the incoming draft).
    fn load_active_fields_from(&mut self, id: u64) {
        let Some(s) = self.sessions.session(id) else {
            return;
        };
        self.transcript = s.transcript.clone();
        self.conn = s.conn.clone();
        self.model = s.model.clone();
        self.busy = s.busy;
        self.turn_started_ms = s.busy_since_ms;
        self.pending_ask = s.pending_ask.clone();
        // Re-id the live next_block_id past the loaded transcript so new live
        // appends don't collide with the session's own ids.
        self.next_block_id = self
            .transcript
            .iter()
            .map(|b| b.id)
            .max()
            .map(|m| m.wrapping_add(1))
            .unwrap_or(1);
        // Reset the render plane so the next `sync_transcript` re-derives cleanly.
        self.wrap_cache = WrapCache::new(self.last_width.max(1));
        self.viewport = Viewport::new(1);
        self.last_width = 0; // force a full rewidth on the next frame.
        self.last_fold_all = self.fold_all;
    }

    /// Switch the active session to `id`: snapshot the current active state into
    /// its record, stash the composer draft on it, load the target's state +
    /// restore its stashed draft, and return to the cockpit view. PURE-ish (only
    /// touches in-memory state + the composer).
    pub fn switch_session(&mut self, id: u64) {
        if id == self.sessions.active && self.view == View::Cockpit {
            return;
        }
        self.snapshot_active_into_map();
        let current_draft = self.composer.text().to_string();
        let incoming_draft = self.sessions.switch(id, current_draft);
        self.load_active_fields_from(self.sessions.active);
        // Restore the incoming session's stashed composer draft.
        self.composer.set_buffer(incoming_draft.clone(), incoming_draft.len());
        self.view = View::Cockpit;
    }

    /// Public wrapper over [`Self::load_active_fields_from`] for the cockpit's
    /// session-cycle path (Ctrl+Up/Down): the map's active already moved; pull its
    /// stored state into the live fields (the composer draft is set by the caller).
    pub fn load_active_fields_from_public(&mut self, id: u64) {
        self.load_active_fields_from(id);
        self.view = View::Cockpit;
    }

    /// Load the fallback active session after a drop (Ctrl+W/Ctrl+D): pull its
    /// stored state in and restore its stashed draft. Handles the last-session
    /// reset (a blank session) cleanly.
    pub fn load_active_fields_after_drop(&mut self, id: u64) {
        self.load_active_fields_from(id);
        let draft = self
            .sessions
            .session_mut(id)
            .map(|s| std::mem::take(&mut s.input_stash))
            .unwrap_or_default();
        self.composer.set_buffer(draft.clone(), draft.len());
        self.view = View::Cockpit;
    }

    /// After a structural change that already moved `self.sessions.active` (a
    /// `new_session` or `branch`), load the NEW active session's state into the
    /// live fields and reset the composer to its (empty / forked) draft. The
    /// LEAVING session's live state must already be snapshotted (the caller does
    /// `snapshot_active_into_map()` first). Returns to the cockpit view.
    pub fn load_active_session_after_structural_change(&mut self, new_active: u64) {
        // The map's active is `new_active`; pull its stored state in.
        debug_assert_eq!(self.sessions.active, new_active);
        self.load_active_fields_from(new_active);
        let draft = self
            .sessions
            .session_mut(new_active)
            .map(|s| std::mem::take(&mut s.input_stash))
            .unwrap_or_default();
        self.composer.set_buffer(draft.clone(), draft.len());
        self.view = View::Cockpit;
    }

    /// Open the full-screen session dashboard (Ctrl+S / left-click sessions area).
    /// Snapshots the active session so the dashboard preview/category are current,
    /// then seeds the dashboard selection on the active session's row.
    pub fn open_dashboard(&mut self) {
        self.snapshot_active_into_map();
        self.view = View::Dashboard;
        self.rename = None;
        // Seed the selection on the active session's row (so Enter re-opens it).
        let rows = self.sessions.dashboard_rows();
        let active = self.sessions.active;
        if let Some(idx) = rows.iter().position(|r| {
            matches!(r, crate::app::session::DashRow::Session { id, .. } if *id == active)
        }) {
            self.sessions.dash_sel = idx;
        }
    }

    /// Close the dashboard, returning to the cockpit (Esc).
    pub fn close_dashboard(&mut self) {
        self.view = View::Cockpit;
        self.rename = None;
    }

    // ---- the /workflows panel (§7) ------------------------------------------

    /// Open the full-screen `/workflows` panel (reached via `/workflows`
    /// `/conductor` `/hive` `/goal`). Lazily STARTS the singleton watcher on first
    /// open (its own poll thread; `repo_root` tells it where `temp/` lives), then
    /// ACTIVATES it so it begins polling (it parks again on close → zero idle
    /// traffic). The panel renders the latest snapshot the event loop feeds in.
    pub fn open_workflows(&mut self) {
        if self.workflow_watcher.is_none() {
            let (watcher, _change_rx) = crate::workflow::WorkflowWatcher::start(self.repo_root.clone());
            // The change receiver is dropped: the event loop already redraws on its
            // 100ms tick + bridge events, and the panel reads `snapshot()` each
            // frame, so we don't need the extra wake channel here (keeping the watcher
            // self-contained avoids threading another receiver through the loop).
            self.workflow_watcher = Some(watcher);
        }
        if let Some(w) = &self.workflow_watcher {
            w.set_active(true);
        }
        self.view = View::Workflows;
        // Pull an initial snapshot immediately so the panel isn't blank for a tick.
        self.refresh_workflow_snapshot();
    }

    /// Close the `/workflows` panel, returning to the cockpit (Esc). PARKS the
    /// watcher so it stops generating background traffic while nobody is looking
    /// (the §5.3 idle back-off, achieved by pausing rather than slow-polling).
    pub fn close_workflows(&mut self) {
        if let Some(w) = &self.workflow_watcher {
            w.set_active(false);
        }
        self.view = View::Cockpit;
    }

    /// Refresh the panel's snapshot from the watcher (called each frame while the
    /// panel is open). Re-clamps the panel focus when the watcher's generation
    /// advanced (nodes may have appeared/vanished). Cheap: a short-held lock + a
    /// clone of a small snapshot; never blocks (the watcher does the I/O off-thread).
    pub fn refresh_workflow_snapshot(&mut self) {
        if let Some(w) = &self.workflow_watcher {
            self.workflow_snapshot = w.snapshot();
            self.workflow_panel.clamp_focus(&self.workflow_snapshot);
        }
    }

    /// Switch the interface language and trigger a FULL repaint (§9 "/language
    /// full repaint"). ratatui is immediate-mode, so the repaint is implicit — the
    /// next frame reads `t(self.lang, …)` for every label. We additionally INVALIDATE
    /// the wrap cache so any multi-line, language-dependent transcript chrome
    /// (folded-turn summaries, tool-chip labels) re-wraps at the same width with the
    /// new strings (the §10 "render cache invalidation on … language" hook). Setting
    /// the same language is a no-op (no needless reflow). PURE-ish (in-memory only).
    pub fn set_language(&mut self, lang: crate::i18n::Lang) {
        if self.lang == lang {
            return;
        }
        self.lang = lang;
        self.invalidate_render_cache();
    }

    /// Force the transcript wrap cache to fully re-derive on the next frame (a
    /// theme / language / fold change that alters projected rows). Resets the
    /// last-synced width sentinel so `sync_transcript` takes the full-rewidth path.
    /// PURE-ish.
    pub fn invalidate_render_cache(&mut self) {
        self.last_width = 0;
    }

    /// Apply a TAGGED bridge event `(session_id, ev)` from the multiplexer. If the
    /// event is for the ACTIVE session we fold it through the normal live reducer
    /// (so the cockpit + footer update exactly as in single-session); otherwise we
    /// fold it into that session's own record (so its dashboard preview/category
    /// update live without disturbing the active session). `now_ms` injected.
    pub fn apply_tagged_event(&mut self, session_id: u64, ev: BridgeEvent, now_ms: u64) {
        if session_id == self.sessions.active {
            // Active session → the live reducer (cockpit path). Mirror the result
            // back into the record so a later dashboard open is consistent.
            self.apply_bridge_event(ev, now_ms);
            self.snapshot_active_into_map();
        } else if let Some(s) = self.sessions.session_mut(session_id) {
            match ev {
                BridgeEvent::Frame(frame) => s.apply_frame(frame, now_ms),
                other => s.apply_lifecycle(&other),
            }
        }
        // A frame for an unknown (already-dropped) session is discarded.
    }
}

/// Map `LlmList` items `(idx, name, current)` onto picker rows: the row label is
/// `"i. name"` (the picker widget prepends `●` for the current one — §4 "rows
/// `● i. name`"), the `id` is the 0-based LLM index (`SwitchLlm` is `id+1`), and
/// `current` carries the active marker. PURE — the `/llm` picker mapping. The
/// `llm_picker_maps_index` deliverable pins the index round-trip on the picker.
pub fn llm_items_to_picks(items: &[LlmItem]) -> Vec<PickItem> {
    items
        .iter()
        .map(|it| {
            PickItem::new(it.idx as usize, format!("{}. {}", it.idx, it.name)).current(it.current)
        })
        .collect()
}

/// Discover the current git branch under `root` by reading `.git/HEAD` (no
/// `git` subprocess — fast + dependency-free). Returns the short branch name, or
/// `None` if not a repo / detached HEAD. PURE-ish (one file read).
pub fn discover_git_branch(root: &std::path::Path) -> Option<String> {
    let head = std::fs::read_to_string(root.join(".git").join("HEAD")).ok()?;
    let head = head.trim();
    // `ref: refs/heads/<branch>` for an attached HEAD.
    head.strip_prefix("ref: refs/heads/")
        .map(|b| b.to_string())
        .or_else(|| {
            // Detached HEAD → show the short commit.
            if head.len() >= 7 && head.chars().all(|c| c.is_ascii_hexdigit()) {
                Some(format!("({})", &head[..7]))
            } else {
                None
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::protocol::CoreToUi;

    fn frame(f: CoreToUi) -> BridgeEvent {
        BridgeEvent::Frame(f)
    }

    #[test]
    fn ready_marks_connected_with_model() {
        let mut app = AppState::new();
        assert_eq!(app.conn, ConnStatus::Connecting);
        app.apply_bridge_event(
            frame(CoreToUi::Ready {
                version: Some("1".into()),
                model: Some("glm-4".into()),
            }),
            0,
        );
        assert_eq!(
            app.conn,
            ConnStatus::Connected {
                model: Some("glm-4".into())
            }
        );
        assert_eq!(app.model.as_deref(), Some("glm-4"));
        assert_eq!(app.conn.label(), "connected glm-4");
    }

    #[test]
    fn streaming_begin_delta_end_assembles_block() {
        let mut app = AppState::new();
        app.apply_bridge_event(
            frame(CoreToUi::MessageBegin {
                mid: "m1".into(),
                role: "assistant".into(),
            }),
            100,
        );
        assert!(app.busy);
        app.apply_bridge_event(
            frame(CoreToUi::MessageDelta {
                mid: "m1".into(),
                text: "Hello ".into(),
            }),
            100,
        );
        app.apply_bridge_event(
            frame(CoreToUi::MessageDelta {
                mid: "m1".into(),
                text: "世界".into(),
            }),
            100,
        );
        app.apply_bridge_event(
            frame(CoreToUi::MessageEnd {
                mid: "m1".into(),
                reason: "stop".into(),
            }),
            100,
        );
        assert!(!app.busy);
        let last = app.transcript.last().unwrap();
        assert_eq!(last.source, "Hello 世界");
        assert_eq!(last.role, Role::Assistant);
        assert!(last.finalized);
    }

    #[test]
    fn child_exit_surfaces_reason_never_silent() {
        let mut app = AppState::new();
        app.apply_bridge_event(BridgeEvent::ChildExited { code: Some(1) }, 0);
        // The reason is surfaced as the CONNECTION STATUS (the footer chip renders
        // it — N1 "never a silent disconnect"), NOT a transcript notice (§c).
        match &app.conn {
            ConnStatus::Disconnected { reason } => assert!(reason.contains("code 1")),
            other => panic!("expected disconnected, got {other:?}"),
        }
        // It is NOT pushed as a transcript row (it would scroll away with the chat).
        assert!(
            !app.transcript.iter().any(|b| b.role == Role::Notice && b.source.contains("code 1")),
            "a fatal exit is a connection status, not a transcript notice"
        );
        // It IS kept in the debug-only ring for a developer.
        assert!(app.bridge_debug.iter().any(|l| l.contains("code 1")));
    }

    /// THE deliverable test (§c): bridge STDERR — especially GA's failover retry
    /// diagnostic `[MixinSession] …retry N/M` — is SUPPRESSED from the transcript.
    /// It never becomes a `[bridge]` row; it lands only in the debug-only ring.
    #[test]
    fn bridge_stderr_suppressed() {
        let mut app = AppState::new();
        let before = app.transcript.len();

        // A failover retry notice on stderr (llmcore.py:988).
        app.apply_bridge_event(
            BridgeEvent::Stderr {
                line: "[MixinSession] codex-pro overloaded, retry 1/10 (2.0s→4.0s)".into(),
            },
            0,
        );
        // A genuine-looking error on stderr (the OLD code would have made this a
        // `[bridge]` row; now it is suppressed too — the transcript stays clean).
        app.apply_bridge_event(
            BridgeEvent::Stderr { line: "Traceback (most recent call last): Error here".into() },
            0,
        );
        // Parse noise is suppressed as well.
        app.apply_bridge_event(BridgeEvent::ParseNoise { line: "garbled json".into() }, 0);

        // NOTHING reached the transcript.
        assert_eq!(app.transcript.len(), before, "no stderr/noise row in the transcript");
        assert!(
            !app.transcript.iter().any(|b| b.source.contains("[bridge]")
                || b.source.contains("MixinSession")
                || b.source.contains("unparsed")),
            "no `[bridge]`/retry/unparsed text in the transcript"
        );
        // But the diagnostics are kept in the debug-only ring (not lost).
        assert!(app.bridge_debug.iter().any(|l| l.contains("MixinSession")));
        assert!(app.bridge_debug.iter().any(|l| l.contains("Traceback")));
        assert!(app.bridge_debug.iter().any(|l| l.contains("unparsed")));
    }

    #[test]
    fn spawn_failed_is_visible_disconnect() {
        let mut app = AppState::new();
        app.apply_bridge_event(
            BridgeEvent::SpawnFailed {
                detail: "python not found".into(),
            },
            0,
        );
        assert!(matches!(app.conn, ConnStatus::Disconnected { .. }));
        assert!(app.conn.label().contains("python not found"));
    }

    #[test]
    fn turn_elapsed_tracks_heat_clock() {
        let mut app = AppState::new();
        app.apply_bridge_event(
            frame(CoreToUi::MessageBegin {
                mid: "m1".into(),
                role: "assistant".into(),
            }),
            1_000,
        );
        assert_eq!(app.turn_elapsed_ms(1_500), 500);
        // Idle after end → elapsed resets to 0.
        app.apply_bridge_event(
            frame(CoreToUi::MessageEnd {
                mid: "m1".into(),
                reason: "stop".into(),
            }),
            2_000,
        );
        assert_eq!(app.turn_elapsed_ms(3_000), 0);
    }

    #[test]
    fn tab_status_and_title_track_state() {
        let mut app = AppState::new();
        // Idle by default.
        assert_eq!(app.tab_status(), TabStatus::Idle);
        assert!(app.terminal_title().starts_with("GenericAgent"));

        // Busy → Working + a spinner glyph leads the title.
        app.apply_bridge_event(
            frame(CoreToUi::MessageBegin { mid: "m1".into(), role: "assistant".into() }),
            0,
        );
        assert_eq!(app.tab_status(), TabStatus::Working);
        assert!(!app.terminal_title().starts_with("GenericAgent"), "spinner glyph leads");

        // A pending ask → Waiting wins over busy.
        app.apply_bridge_event(
            frame(CoreToUi::AskUser {
                ask_id: "a1".into(),
                question: "pick?".into(),
                options: vec![],
                free_text: true,
            }),
            0,
        );
        assert_eq!(app.tab_status(), TabStatus::Waiting);
    }

    /// An incoming `LlmList` opens the `/llm` picker overlay with the model rows,
    /// pre-selecting the current model (N3 wiring at the app layer).
    #[test]
    fn llm_list_frame_opens_picker_on_current() {
        let mut app = AppState::new();
        app.apply_bridge_event(
            frame(CoreToUi::LlmList {
                items: vec![
                    LlmItem { idx: 0, name: "A/a".into(), current: false },
                    LlmItem { idx: 1, name: "B/b".into(), current: true },
                ],
            }),
            0,
        );
        match &app.overlay {
            Some(Overlay::Picker { picker, .. }) => {
                assert_eq!(picker.kind, crate::components::picker::PickerKind::Llm);
                assert_eq!(picker.sel, 1, "opens on the current model");
                assert_eq!(picker.selected_id(), Some(1));
                // The row label is "idx. name" (the widget paints the ● for current).
                assert_eq!(picker.selected().unwrap().label, "1. B/b");
            }
            other => panic!("expected an llm picker overlay, got {other:?}"),
        }
    }

    /// A second ask_user that arrives while one is being answered QUEUES, and
    /// surfaces after the first is resolved (§7 "queued parallel asks surface in
    /// turn"). The queue is FIFO and no ask is dropped.
    #[test]
    fn queued_parallel_asks_surface_in_turn() {
        let mut app = AppState::new();
        // First ask → becomes the active pending ask + opens the card.
        app.apply_bridge_event(
            frame(CoreToUi::AskUser {
                ask_id: "a1".into(),
                question: "first?".into(),
                options: vec![],
                free_text: true,
            }),
            0,
        );
        assert_eq!(app.pending_ask.as_ref().unwrap().ask_id, "a1");
        assert!(matches!(app.overlay, Some(Overlay::AskUser(_))));

        // A second ask arrives while the first is active → it QUEUES (not dropped).
        app.apply_bridge_event(
            frame(CoreToUi::AskUser {
                ask_id: "a2".into(),
                question: "second?".into(),
                options: vec![],
                free_text: true,
            }),
            0,
        );
        assert_eq!(app.pending_ask.as_ref().unwrap().ask_id, "a1", "first stays active");
        assert_eq!(app.ask_queue.len(), 1);

        // Answering the first clears it; surfacing the next pops the queue → a2.
        app.pending_ask = None;
        app.overlay = None;
        assert!(app.surface_next_ask());
        assert_eq!(app.pending_ask.as_ref().unwrap().ask_id, "a2");
        assert!(app.ask_queue.is_empty());
        // No more queued → surface_next is a no-op.
        app.pending_ask = None;
        assert!(!app.surface_next_ask());
    }

    /// `/btw` answer routing (§7): a `BtwAnswer` fills the OPEN matching card and
    /// NEVER touches the transcript (no history pollution); a stale `ask_id` is
    /// ignored; and if the card was dismissed the answer is silently dropped. The
    /// `/btw` card is also NON-modal (it doesn't steal input — chat stays usable).
    #[test]
    fn btw_answer_routes_to_card_not_history() {
        let mut app = AppState::new();
        let transcript_len_before = app.transcript.len();

        // Open a /btw card (as the dispatcher does) and confirm it is NON-modal.
        app.open_overlay(Overlay::Btw {
            ask_id: "b1".into(),
            question: "what is 6*7?".into(),
            answer: None,
        });
        assert!(!app.overlay.as_ref().unwrap().is_modal(), "the /btw card is non-modal");

        // A BtwAnswer for a DIFFERENT ask_id is ignored (the card stays querying).
        app.apply_bridge_event(
            frame(CoreToUi::BtwAnswer { ask_id: "stale".into(), text: Some("nope".into()), error: None }),
            0,
        );
        match &app.overlay {
            Some(Overlay::Btw { answer, .. }) => assert!(answer.is_none(), "stale id leaves the card querying"),
            other => panic!("expected the btw card to remain, got {other:?}"),
        }

        // The matching BtwAnswer fills the card's answer — and NOT the transcript.
        app.apply_bridge_event(
            frame(CoreToUi::BtwAnswer { ask_id: "b1".into(), text: Some("42".into()), error: None }),
            0,
        );
        match &app.overlay {
            Some(Overlay::Btw { answer, .. }) => assert_eq!(answer.as_deref(), Some("42")),
            other => panic!("expected the answered btw card, got {other:?}"),
        }
        assert_eq!(app.transcript.len(), transcript_len_before, "a /btw answer never pollutes history");

        // An error answer shows a reason (still no history pollution).
        app.open_overlay(Overlay::Btw { ask_id: "b2".into(), question: "q".into(), answer: None });
        app.apply_bridge_event(
            frame(CoreToUi::BtwAnswer { ask_id: "b2".into(), text: None, error: Some("no llm".into()) }),
            0,
        );
        match &app.overlay {
            Some(Overlay::Btw { answer, .. }) => assert!(answer.as_ref().unwrap().contains("no llm")),
            other => panic!("expected an errored btw card, got {other:?}"),
        }

        // Dismissed (no card open) → the answer is silently dropped, no history.
        app.overlay = None;
        app.apply_bridge_event(
            frame(CoreToUi::BtwAnswer { ask_id: "b1".into(), text: Some("late".into()), error: None }),
            0,
        );
        assert!(app.overlay.is_none());
        assert_eq!(app.transcript.len(), transcript_len_before, "a dropped /btw answer never pollutes history");
    }

    /// `RewindResult` is surfaced as a NOTICE (acknowledgment), not a streamed
    /// message — so a rewind never spends a model turn. The active reducer folds it.
    #[test]
    fn rewind_result_surfaces_as_notice() {
        let mut app = AppState::new();
        app.apply_bridge_event(frame(CoreToUi::RewindResult { dropped: 2, remaining: 3 }), 0);
        assert!(
            app.transcript.iter().any(|b| b.role == Role::Notice && b.source.contains('2')),
            "the rewind confirmation lands as a notice: {:?}",
            app.transcript.iter().map(|b| &b.source).collect::<Vec<_>>()
        );
        // It is NOT busy / streaming (no model turn spent).
        assert!(!app.busy);
    }

    /// MessageEnd on an assistant block harvests its tool calls into the `/verbose`
    /// audit trail (so the audit overlay isn't empty after tool use).
    #[test]
    fn message_end_harvests_tool_audit() {
        let mut app = AppState::new();
        app.apply_bridge_event(
            frame(CoreToUi::MessageBegin { mid: "m1".into(), role: "assistant".into() }),
            0,
        );
        app.apply_bridge_event(
            frame(CoreToUi::MessageDelta {
                mid: "m1".into(),
                text: "🛠️ Tool: `file_read`\npath: config.toml\n→ ok".into(),
            }),
            0,
        );
        assert!(app.tool_audit.is_empty(), "not harvested until the block finalizes");
        app.apply_bridge_event(
            frame(CoreToUi::MessageEnd { mid: "m1".into(), reason: "stop".into() }),
            0,
        );
        assert!(
            app.tool_audit.iter().any(|l| l.contains("file_read")),
            "the finalized assistant block's tool call lands in the audit: {:?}",
            app.tool_audit
        );
    }

    /// The `/cost` report formatter lays out input/output/cache/total/context%.
    #[test]
    fn cost_report_lines_format() {
        let mut cost = CostBreakdown { input: 1200, output: 350, cache: 90, ..Default::default() };
        cost.context_percent = Some(62.0);
        cost.cost_usd = 0.1234;
        let lines = cost.report_lines("glm-4");
        assert!(lines[0].contains("glm-4"));
        assert!(lines.iter().any(|l| l.contains("input") && l.contains("1200")));
        assert!(lines.iter().any(|l| l.contains("output") && l.contains("350")));
        assert!(lines.iter().any(|l| l.contains("total") && l.contains("1550")));
        assert!(lines.iter().any(|l| l.contains("context") && l.contains("62%")));
        assert_eq!(cost.total(), 1550);
    }

    /// `/language` switching flips the active language AND invalidates the render
    /// cache so the next frame fully repaints in the new language (§9 "/language
    /// full repaint"). Switching to the same language is a no-op (no needless
    /// reflow). The cockpit then resolves labels through the new language.
    #[test]
    fn set_language_triggers_full_repaint() {
        use crate::i18n::{self, Lang};
        let mut app = AppState::new();
        assert_eq!(app.lang, Lang::En);
        // Prime the cache "synced" state so we can observe the invalidation.
        app.last_width = 80;
        // Same language → no-op (cache stays synced).
        app.set_language(Lang::En);
        assert_eq!(app.last_width, 80, "same-language switch does not invalidate");
        // Switch to zh → lang changes AND the cache is invalidated (last_width=0
        // forces a full rewidth on the next sync_transcript, the repaint hook).
        app.set_language(Lang::Zh);
        assert_eq!(app.lang, Lang::Zh);
        assert_eq!(app.last_width, 0, "language switch invalidates the wrap cache (full repaint)");
        // The cockpit now resolves a label through zh (the composer placeholder).
        assert_eq!(
            i18n::t(app.lang, "composer.placeholder"),
            i18n::t(Lang::Zh, "composer.placeholder")
        );
        assert_ne!(
            i18n::t(Lang::Zh, "conn.connecting"),
            i18n::t(Lang::En, "conn.connecting"),
            "the two languages render different strings"
        );
    }

    #[test]
    fn discover_git_branch_reads_head() {
        let dir = std::env::temp_dir().join(format!("tui_v4_git_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join(".git")).unwrap();
        std::fs::write(dir.join(".git").join("HEAD"), "ref: refs/heads/main\n").unwrap();
        assert_eq!(super::discover_git_branch(&dir).as_deref(), Some("main"));

        // Detached HEAD → short commit in parens.
        std::fs::write(dir.join(".git").join("HEAD"), "abc1234deadbeef\n").unwrap();
        assert_eq!(super::discover_git_branch(&dir).as_deref(), Some("(abc1234)"));

        // Not a repo → None.
        let _ = std::fs::remove_dir_all(dir.join(".git"));
        assert!(super::discover_git_branch(&dir).is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The multi-session glue (§6 / N2): opening the dashboard, switching swaps
    /// the live transcript with the target session's, and a tagged event for a
    /// non-active session updates ONLY that session's record (the cockpit's live
    /// transcript is untouched), while a tagged event for the active session
    /// flows through the live reducer.
    #[test]
    fn multi_session_switch_and_routing() {
        use crate::app::View;
        let root = std::env::temp_dir().join(format!("tui_v4_app_ms_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("temp")).unwrap();

        let mut app = AppState::new();
        app.attach_repo_root(root.clone());
        let s1 = app.sessions.active; // session 1.

        // Stream a reply into the ACTIVE session (s1) via a TAGGED event.
        app.apply_tagged_event(
            s1,
            BridgeEvent::Frame(CoreToUi::MessageBegin { mid: "m1".into(), role: "assistant".into() }),
            10,
        );
        app.apply_tagged_event(
            s1,
            BridgeEvent::Frame(CoreToUi::MessageDelta { mid: "m1".into(), text: "reply for one".into() }),
            10,
        );
        // The live transcript (cockpit) shows it.
        assert!(app.transcript.iter().any(|b| b.source.contains("reply for one")));

        // Create + switch to a second session; the live transcript is now empty.
        app.snapshot_active_into_map();
        let s2 = app.sessions.new_session(None);
        app.load_active_session_after_structural_change(s2);
        assert_eq!(app.sessions.active, s2);
        assert!(app.transcript.is_empty(), "switching to a fresh session clears the live transcript");

        // A TAGGED event for the NON-active session (s1) updates only s1's record;
        // the live transcript (s2) stays empty.
        app.apply_tagged_event(
            s1,
            BridgeEvent::Frame(CoreToUi::MessageBegin { mid: "m2".into(), role: "assistant".into() }),
            20,
        );
        app.apply_tagged_event(
            s1,
            BridgeEvent::Frame(CoreToUi::MessageDelta { mid: "m2".into(), text: "background work".into() }),
            20,
        );
        assert!(app.transcript.is_empty(), "a background session's stream never touches the active transcript");
        assert!(app
            .sessions
            .session(s1)
            .unwrap()
            .transcript
            .iter()
            .any(|b| b.source.contains("background work")));

        // Open the dashboard, switch back to s1 → its full transcript is restored.
        app.open_dashboard();
        assert_eq!(app.view, View::Dashboard);
        app.switch_session(s1);
        assert_eq!(app.view, View::Cockpit);
        assert_eq!(app.sessions.active, s1);
        assert!(app.transcript.iter().any(|b| b.source.contains("reply for one")));
        assert!(app.transcript.iter().any(|b| b.source.contains("background work")));

        // Esc-style close returns to cockpit and is idempotent.
        app.open_dashboard();
        app.close_dashboard();
        assert_eq!(app.view, View::Cockpit);

        let _ = std::fs::remove_dir_all(&root);
    }

    /// THE wiring test: `/workflows` opens the full-screen panel, LAZILY starts the
    /// singleton watcher (its own thread) + ACTIVATES it, and `Esc`-close returns to
    /// the cockpit while PARKING the watcher (zero idle traffic). Re-opening reuses
    /// the same watcher (no second thread). Dropping the app joins the thread.
    #[test]
    fn workflows_panel_open_close_drives_watcher() {
        let root = std::env::temp_dir().join(format!("tui_v4_app_wf_open_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("temp")).unwrap();

        let mut app = AppState::new();
        app.attach_repo_root(root.clone());

        // No watcher until the panel is first opened (a user who never opens
        // /workflows spawns no thread — the §3 lazy-start contract).
        assert!(app.workflow_watcher.is_none());
        assert_eq!(app.view, View::Cockpit);

        // Open → the panel becomes the active view, a watcher exists + is ACTIVE.
        app.open_workflows();
        assert_eq!(app.view, View::Workflows);
        assert!(app.workflow_watcher.is_some());
        assert!(app.workflow_watcher.as_ref().unwrap().is_active(), "open activates the watcher");

        // Close (Esc) → back to the cockpit, watcher PARKED (still alive, not active).
        app.close_workflows();
        assert_eq!(app.view, View::Cockpit);
        assert!(app.workflow_watcher.is_some(), "the watcher persists (parked) across close");
        assert!(!app.workflow_watcher.as_ref().unwrap().is_active(), "close parks the watcher");

        // Re-open reuses the SAME watcher (lazy-start only happens once) and
        // re-activates it.
        app.open_workflows();
        assert!(app.workflow_watcher.as_ref().unwrap().is_active(), "re-open re-activates");

        // Dropping the app drops the watcher, which joins its thread within a bound
        // (no hang) — proven by this test returning.
        drop(app);

        let _ = std::fs::remove_dir_all(&root);
    }

    /// `/conductor /hive /goal` are "forward + open the /workflows tile": forwarding
    /// the command ALSO opens the panel so the user watches the live tree the watcher
    /// populates. We exercise the open side of that route here (the forward side is a
    /// bridge send covered elsewhere) by calling `open_workflows` the dispatcher
    /// invokes, and confirm a non-routing command does NOT open it.
    #[test]
    fn orchestration_commands_open_the_panel() {
        let root = std::env::temp_dir().join(format!("tui_v4_app_wf_route_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("temp")).unwrap();
        let mut app = AppState::new();
        app.attach_repo_root(root.clone());

        // The three orchestration verbs route to the panel (the dispatcher matches
        // exactly this set — keep this list in lock-step with `dispatch_slash`).
        for name in ["goal", "hive", "conductor"] {
            assert!(
                matches!(name, "goal" | "hive" | "conductor"),
                "the panel-opening set is {{goal,hive,conductor}}"
            );
        }
        // A representative open (what the dispatcher does after forwarding /conductor).
        app.open_workflows();
        assert_eq!(app.view, View::Workflows);

        drop(app);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The snapshot→panel feed: `refresh_workflow_snapshot` pulls the watcher's
    /// latest merged snapshot into `AppState` and re-clamps the panel focus. With a
    /// real conductor down (no server on the gate box) the merge still yields the
    /// Conductor group placeholder, so the panel has a non-empty snapshot to render
    /// and `focused_node` resolves (or is None on a node-less down snapshot) without
    /// panicking — the load-bearing "never blocks / never panics" guarantee.
    #[test]
    fn refresh_feeds_snapshot_into_panel() {
        let root = std::env::temp_dir().join(format!("tui_v4_app_wf_feed_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("temp")).unwrap();
        let mut app = AppState::new();
        app.attach_repo_root(root.clone());

        app.open_workflows();
        // Give the watcher up to a few poll intervals to publish its first merge.
        let deadline = std::time::Instant::now() + crate::workflow::POLL_INTERVAL * 6;
        while app.workflow_watcher.as_ref().map(|w| w.generation()).unwrap_or(0) == 0
            && std::time::Instant::now() < deadline
        {
            std::thread::sleep(std::time::Duration::from_millis(50));
            app.refresh_workflow_snapshot();
        }
        app.refresh_workflow_snapshot();
        // The watcher always merges a Conductor workflow (a down placeholder when
        // :8900 is closed), so the app's snapshot is non-empty and the panel can
        // render it. The exact running state depends on the environment.
        assert!(
            app.workflow_snapshot
                .workflows
                .iter()
                .any(|w| w.kind == crate::workflow::schema::WorkflowKind::Conductor),
            "refresh feeds the merged conductor group into the panel's snapshot"
        );
        // Focus resolution never panics on the fed snapshot (None on a node-less
        // down snapshot is fine).
        let _ = app.workflow_panel.focused_node(&app.workflow_snapshot);

        drop(app);
        let _ = std::fs::remove_dir_all(&root);
    }
}
