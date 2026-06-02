//! app/mod.rs — `AppState`: the whole application state + its construction, the
//! per-tick clock, the render-plane sync/scroll surface, and the UI→app intent
//! queue. The value TYPES live in [`types`]; the bridge fold in [`reducer`]
//! (the ONE [`reducer::apply_frame`] shared by `AppState` + `Session`, killing
//! the dup); the overlay-stack constructors in [`overlay_ops`]; the multi-session
//! + view-switch glue in [`multi`]. Keeping the load-bearing logic in PURE
//! methods (no I/O, no ratatui) is what makes it headlessly unit-testable.

pub mod effort;
mod fold_hit;
mod multi;
mod overlay_ops;
mod reducer;
pub mod session;
mod types;

pub use types::{Block, ConnStatus, CostBreakdown, Overlay, PendingAsk, RenameState, Role, View};

use std::path::PathBuf;

use crate::app::effort::{EffortSlider, ReasoningEffort};
use crate::app::session::SessionMap;
use crate::app_event::AppEvent;
use crate::effects::{EffectMode, EffectsEngine};
use crate::flavor::{CompanionKind, Lang};
use crate::input::keychord::ChordState;
use crate::input::Composer;
use crate::render::fold::{BlockFolds, NodeId};
use crate::render::{Block as RenderBlock, Viewport, WrapCache};
use crate::theme::Theme;
use crate::util::osc::TabStatus;

/// The fixed delta-time (seconds) one `tick()` advances the effects engine. Matches the
/// event loop's 0.1s tick cadence so effect speed is calibrated in real seconds while
/// remaining a pure function of the integer tick count.
pub const TICK_DT: f32 = 0.1;

/// How long a memoized `@`-picker file-index snapshot stays fresh before
/// [`AppState::list_project_files`] re-walks the repo tree (Q12 @ speed). 30s is a
/// coarse debounce (CC-style, not an async actor): fluid enough that typing in the
/// picker never pays a walk, yet short enough that a newly-created file is
/// `@`-completable within a session. Raised from 5s to amortize the now-deeper
/// `MAX_INDEXED_FILES`=20000 BFS so the first keystroke after the window lapses
/// doesn't pay a big synchronous walk on the render thread. The walk is bounded by
/// `MAX_INDEXED_FILES` either way; the TTL just collapses the 3×/frame re-walk into
/// ≤1 walk per window.
const FILE_INDEX_TTL: std::time::Duration = std::time::Duration::from_secs(30);


/// The whole application state for the Foundation slice.
#[derive(Debug)]
pub struct AppState {
    /// Real connection status (drives the status line, N1).
    pub conn: ConnStatus,
    /// Current model name (from `Ready`/`Status`). Legacy full SessionType/chain
    /// string; kept for the `llm_channel`/`truncate_model` fallback path.
    pub model: Option<String>,
    /// Active LLM config name on the wire (`Ready`/`Status` `llm`, e.g. `codex-pro`).
    /// `None` on a stale bridge that only sends `model` — the header falls back.
    pub llm_name: Option<String>,
    /// Real underlying model on the wire (`Ready`/`Status` `model_real`, e.g.
    /// `gpt-5.5`), `[...]` tag already stripped by the bridge. Updates on a mid-turn
    /// failover (Status re-emits both).
    pub model_real: Option<String>,
    /// The logical transcript (oldest → newest).
    pub transcript: Vec<Block>,
    /// The multi-line composer (buffer + cursor + selection + undo/redo + magic
    /// prefixes + input history) — the Phase-2 cockpit input (§4/§8).
    pub composer: Composer,
    /// A 0.1s tick counter that drives the spinner + gerund rotation.
    pub spinner_tick: u64,
    /// The unified /emoji companion selection (spinner glyph or pet face).
    /// Drives both the spinner lead glyph and the tab-title face.
    pub companion: CompanionKind,
    /// The reasoning-effort level last applied via `/effort` (redesign_cc.md §3).
    /// `None` until the user sets one (the backend keeps its own default); once set
    /// the spinner/status shows a `thinking · <level>` suffix and the `/effort`
    /// slider seeds its marker here. The label is the slider stop (so `max` stays
    /// "max" in the UI even though it forwarded `xhigh` to the backend).
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Whether tool chips are all folded (Ctrl+Shift+O / `/fold` toggle).
    pub fold_all: bool,
    /// Per-node fold OVERRIDES (Fix E / Q8), keyed by [`NodeId`]. Absent ⇒ the
    /// default policy (a completed turn folds, the last stays open; a tool result is
    /// truncated). An entry wins: `Turn{..}=>true/false` folds/expands that turn;
    /// `Tool{..}=>true` expands that tool's result. A left-click on a node's
    /// triangle/bullet column toggles its entry ([`AppState::toggle_fold`]); the
    /// global `fold_all` flip clears this map so "fold all" / "unfold all" is a clean
    /// reset. Sparse — only toggled nodes appear.
    pub folds: std::collections::HashMap<NodeId, bool>,
    /// The highlighted row in the `/`-slash command palette dropdown (↑/↓ move it;
    /// Tab/Enter completes the highlighted one). Reset to 0 whenever the typed
    /// partial changes; clamped to the live match count by the renderer + nav.
    pub palette_sel: usize,
    /// Whether terminal mouse capture is ON (INTERACTIVE mode: click fold ▸/▾ +
    /// wheel ScrollUp/Down). Defaults to `false` (NATIVE mode, S1): native drag-select
    /// works and wheel translates to arrow keys via EnableAlternateScroll (?1007h).
    /// Toggled by Ctrl+Shift+M / `/mouse`; the mode is shown in the session-info row.
    pub mouse_capture: bool,
    /// The interface language (drives rotating tips; full i18n is Phase 3).
    pub lang: Lang,
    /// Whether a turn is in flight (a message is streaming).
    pub busy: bool,
    /// Monotonic ms when the current turn began (for the heat ramp).
    pub turn_started_ms: u64,
    /// Wall-time (ms) the LAST finished turn took, stamped in `MessageEnd` before
    /// `busy=false` and CLEARED on the next `MessageBegin`. Drives the frozen
    /// above-composer done-line (`⠿ … for <fmt_dur> · ↑in · ↓out`, Q7) — shown only
    /// while idle right after a turn. `None` before the first turn finishes.
    pub last_turn_ms: Option<u64>,
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
    /// Smoothly-animated DISPLAY values for the spinner's ↑/↓ readout.
    /// Each tick() steps these toward the live tok_in/tok_out targets.
    /// None until the first Status frame arrives (shows nothing until data exists).
    pub display_tok_in: Option<u64>,
    pub display_tok_out: Option<u64>,
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
    pub(in crate::app) next_block_id: u64,
    /// The wrap cache keyed at the current transcript width (P1). Rebuilt per
    /// frame via `sync`; only changed blocks reflow. Owned here so the transcript
    /// widget and copy actions share one derivation.
    pub wrap_cache: WrapCache,
    /// The scrolling viewport over the transcript (logical [`ScrollAnchor`], P1).
    pub viewport: Viewport,
    /// The clickable-node hit table (Fix E / Q8), rebuilt every `sync_transcript`:
    /// each entry maps a contiguous GLOBAL visual-row range to the fold node that
    /// occupies it (a turn's `▸` header, or a tool bullet). A left-mouse-down on the
    /// triangle/bullet column of a row in one of these ranges resolves the node
    /// ([`AppState::transcript_node_at`]) and toggles its fold. Width-keyed implicitly
    /// (derived from the just-synced cache), so it is always consistent with the rows
    /// the transcript draws this frame.
    node_hit: Vec<(std::ops::RangeInclusive<usize>, NodeId)>,
    /// The transcript width the cache was last synced at (to detect resize).
    pub(in crate::app) last_width: u16,
    /// The fold-all state the cache was last synced at (to detect a `/fold` flip).
    pub(in crate::app) last_fold_all: bool,
    /// A monotonic FOLD epoch, bumped whenever the fold STATE changes without a
    /// block `rev` change (a per-node toggle, or a `fold_all` flip that also clears
    /// overrides). `sync_transcript` rebuilds the wrap cache when it advances past
    /// `last_fold_epoch` — the `(block_id, rev)` reuse cannot otherwise see that a
    /// block's projected rows changed (the fold projection is not in `rev`). (Fix E.)
    pub(in crate::app) fold_epoch: u64,
    /// The fold epoch the wrap cache was last synced at.
    pub(in crate::app) last_fold_epoch: u64,
    /// The animated effects engine (§9). OFF by default; advanced once per 0.1s `tick()`
    /// by a FIXED dt so it is driven by the deterministic tick counter, not a wall clock.
    /// Under `--smoke` the engine is never ticked → no animation.
    pub effects: EffectsEngine,
    /// The transient multi-press chord state (§8): the 3-stage Ctrl+C arm window
    /// + the Esc-Esc double-tap window. PURE state advanced by `input::keychord`
    /// with an injected `now_ms` (never a wall-clock), so the transitions stay
    /// headlessly testable. `Default` = neither chord is mid-sequence.
    pub chord: ChordState,
    /// The UI→app INTENT queue (ARCH Fix A). Key/command/apply handlers push
    /// [`AppEvent`]s here via [`AppState::emit`] instead of closing over the bridge
    /// `Sender`; the event loop drains it each iteration ([`AppState::drain_actions`])
    /// and performs each intent in ONE place. Always empty between iterations.
    actions: Vec<AppEvent>,
    /// Memoized snapshot of the `@`-picker project-file walk (Q12 @ speed). The
    /// walk descends every non-ignored dir of the (giant) repo and was previously
    /// run 3× per frame while an `@query` was live, reading as the `@`/Ctrl+S
    /// freeze. We cache `(walked_at, files)` and re-walk at most once per
    /// [`FILE_INDEX_TTL`]; otherwise the call sites get a cheap `Arc` clone. Async
    /// is banned here, so this is the synchronous-equivalent of Codex's
    /// `FileSearchManager` (walk once, hand out a shared snapshot). `RefCell` so the
    /// memoization works behind the `&self` render-plane borrow.
    file_index: std::cell::RefCell<Option<(std::time::Instant, std::sync::Arc<Vec<String>>)>>,
}

impl Default for AppState {
    fn default() -> Self {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        AppState {
            conn: ConnStatus::Connecting,
            model: None,
            llm_name: None,
            model_real: None,
            transcript: Vec::new(),
            composer: Composer::new(),
            spinner_tick: 0,
            companion: CompanionKind::default(),
            reasoning_effort: None,
            fold_all: false,
            folds: std::collections::HashMap::new(),
            palette_sel: 0,
            // Mouse capture starts OFF (NATIVE mode, S1): the terminal does native
            // OS drag-select + copy, and the wheel translates to arrow keys via
            // EnableAlternateScroll (?1007h). `Ctrl+Shift+M` / `/mouse` flips it ON
            // for INTERACTIVE mode (click ▸/▾ fold + wheel ScrollUp/Down). Matches
            // `term::setup` (which does NOT EnableMouseCapture by default) — the
            // field is the single source of truth.
            mouse_capture: false,
            lang: Lang::default(),
            busy: false,
            turn_started_ms: 0,
            last_turn_ms: None,
            cwd: cwd.display().to_string(),
            repo_root: cwd.clone(),
            pending_ask: None,
            ask_queue: std::collections::VecDeque::new(),
            should_quit: false,
            context_percent: None,
            tokens: None,
            tok_in: None,
            tok_out: None,
            display_tok_in: None,
            display_tok_out: None,
            cost_usd: 0.0,
            git_branch: None,
            last_tab_status: None,
            last_title: String::new(),
            view: View::default(),
            theme: Theme::default_theme(),
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
            node_hit: Vec::new(),
            last_width: 0,
            last_fold_all: false,
            fold_epoch: 0,
            last_fold_epoch: 0,
            effects: EffectsEngine::from_env(),
            chord: ChordState::default(),
            actions: Vec::new(),
            file_index: std::cell::RefCell::new(None),
        }
    }
}

impl AppState {
    pub fn new() -> Self {
        AppState::default()
    }

    /// Queue a UI→app intent (ARCH Fix A). Handlers call this instead of
    /// performing the bridge/view effect inline; the event loop drains the queue
    /// each iteration via [`AppState::drain_actions`] and performs each one.
    pub fn emit(&mut self, ev: AppEvent) {
        self.actions.push(ev);
    }

    /// Take the queued intents, leaving the queue empty. The event loop performs
    /// each drained [`AppEvent`] in the ONE place the bridge transport lives.
    pub fn drain_actions(&mut self) -> Vec<AppEvent> {
        std::mem::take(&mut self.actions)
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
        // Drive the `/continue` picker's debounced content grep: typing only applied
        // the cheap META filter, so once the user pauses (GREP_DEBOUNCE_TICKS) the
        // lazy ≤1 MiB head-window grep runs here, off the keystroke path.
        if let Some(Overlay::Continue(picker)) = self.overlay.as_mut() {
            picker.tick(self.spinner_tick, crate::components::continue_picker::read_head_window);
        }
        // Ease display_tok_in/display_tok_out toward their live targets each tick (0.1s).
        // Mirrors CC SpinnerAnimationRow.tsx:142-158: gap-proportional step so small gaps
        // feel snappy and large jumps animate smoothly.
        // THIS IS A SINGLE BOUNDED STEP PER CALL, NOT A LOOP.
        ease_display_tok(&mut self.display_tok_in, self.tok_in);
        ease_display_tok(&mut self.display_tok_out, self.tok_out);
    }

    /// Apply an effects mode change. The `/effects` COMMAND was removed (Slice 7); the
    /// effects ENGINE + this setter are kept so the separator shimmer / border FX keep
    /// running automatically and the mode stays programmatically settable.
    #[allow(dead_code)]
    pub fn set_effects_mode(&mut self, mode: EffectMode) {
        self.effects.set_mode(mode);
    }

    /// Start the effects-demo splash: arm the demo timer + open the splash overlay. The
    /// `/effects demo` command was removed (Slice 7); the splash machinery is retained
    /// (it is the sole constructor of `Overlay::Effects`, whose render/input/tick arms
    /// stay live for any future trigger).
    #[allow(dead_code)]
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
    /// `pub(in crate::app)` so the [`reducer`](crate::app::reducer) fold can mint
    /// ids when it appends streamed blocks.
    pub(in crate::app) fn alloc_block_id(&mut self) -> u64 {
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

    /// The OSC0 terminal title (S5): `<face> <session_name> · GenericAgent`.
    /// For a pet companion the face animates via spinner_tick / PET_TICKS_PER_FRAME;
    /// for a spinner companion the animated spinner glyph is used. Pet(Off) falls back
    /// to the bear constant so the tab always carries the bear identity. busy vs idle
    /// is conveyed by the OSC tab-STATUS channel, not by swapping the title.
    pub fn terminal_title(&self) -> String {
        use crate::flavor::{CompanionKind, PetStyle, PET_TICKS_PER_FRAME, PETS_BEAR};
        let tick = self.spinner_tick;
        let elapsed = self.turn_elapsed_ms(0);
        let face: String = match self.companion {
            CompanionKind::Pet(PetStyle::Off) => PETS_BEAR[0][0].to_string(),
            CompanionKind::Pet(p) => {
                let f = crate::flavor::pet_face(p, elapsed, tick / PET_TICKS_PER_FRAME);
                if f.is_empty() { PETS_BEAR[0][0].to_string() } else { f.to_string() }
            }
            CompanionKind::Spinner(s) => s.glyph(tick).to_string(),
        };
        let name = self.sessions.active_name();
        if name.is_empty() {
            format!("{face} GenericAgent")
        } else {
            format!("{face} {name} · GenericAgent")
        }
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
        // a function of width — fold to the transcript's inner width here. Each block
        // sees the per-node fold overrides through a `BlockFolds` keyed on its id.
        let render_blocks: Vec<RenderBlock> = self
            .transcript
            .iter()
            .map(|b| {
                let folds = BlockFolds { block_id: b.id, fold_all, overrides: Some(&self.folds) };
                b.to_render_block(theme, &folds, width)
            })
            .collect();

        // A width / fold-all / per-node-fold change invalidates the whole cache (all
        // alter the projected rows). Width also re-derives the viewport from the
        // anchor (P1). A per-node fold toggle bumps `fold_epoch` WITHOUT bumping any
        // block `rev`, so the `(id, rev)` reuse can't see the projection changed — we
        // must drop the per-block memo so every block reflows from the new projection.
        let fold_changed = fold_all != self.last_fold_all || self.fold_epoch != self.last_fold_epoch;
        if width != self.last_width || fold_changed {
            if width == self.wrap_cache.width() && fold_changed {
                // Same width, only the fold projection changed: clear the per-block
                // memo so `rewidth` (a no-op clearer at an unchanged width) can't keep
                // stale rows; then the rebuild below reflows everything fresh.
                self.wrap_cache.invalidate_all();
            }
            self.wrap_cache.rewidth(width, &render_blocks);
            self.viewport.resize(height, &self.wrap_cache);
            self.last_width = width;
            self.last_fold_all = fold_all;
            self.last_fold_epoch = self.fold_epoch;
        } else {
            // Same width + fold: reflow only changed blocks (streaming → O(1)).
            self.wrap_cache.sync(&render_blocks);
            if height != self.viewport.height() {
                self.viewport.resize(height, &self.wrap_cache);
            }
        }
        self.rebuild_node_hit(theme, fold_all, width);
        render_blocks
    }

    /// Apply the per-frame STATE WRITES that render used to do mid-draw, hoisted out
    /// so [`crate::components::cockpit::render`] is pure (`y = f(x)`, P11 / F7). The
    /// event loop calls this with the terminal `area` BEFORE `terminal.draw`:
    ///   * record the full terminal size (dashboard click-mapping geometry);
    ///   * in the COCKPIT view ONLY, sync the wrap cache + viewport to the transcript
    ///     region — derived from the SAME layout split render draws into (so geometry
    ///     never drifts), matching the previous in-render behavior exactly (the
    ///     dashboard/workflows views never synced the transcript).
    pub fn prepare_frame(&mut self, area: ratatui::layout::Rect, theme: &Theme) {
        self.set_term_size(area.width, area.height);
        if self.view == View::Cockpit {
            let transcript = crate::components::cockpit::split_cockpit(self, area).transcript;
            self.sync_transcript(transcript.width, transcript.height as usize, theme);
        }
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

    /// The FIRST line of the most recent USER message — the sticky breadcrumb pinned
    /// at the top of the transcript while scrolled above the tail (R6 Part A), so the
    /// user always sees which prompt they are reading under. `None` if there is no
    /// user turn yet (then the sticky row is not allocated).
    pub fn last_user_source_first_line(&self) -> Option<&str> {
        self.transcript
            .iter()
            .rev()
            .find(|b| b.role == Role::User && !b.source.trim().is_empty())
            .map(|b| b.source.lines().next().unwrap_or("").trim())
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
    /// bounded). Walks the repo root once per [`FILE_INDEX_TTL`] and returns a cheap
    /// `Arc` clone of the cached snapshot on every other call — so the three live
    /// call sites (dropdown layout + paint + the `@` keystroke handler) share ONE
    /// walk per frame instead of three (Q12 @ speed / Ctrl+S no-freeze). A walk
    /// degrades to empty on error inside `paths::list_project_files`. PURE w.r.t.
    /// the public state (the `RefCell` cache is interior mutability for the `&self`
    /// render-plane borrow); the returned `Arc<Vec<String>>` derefs to `&[String]`
    /// at the call sites, so they keep calling `app.list_project_files()` unchanged.
    pub fn list_project_files(&self) -> std::sync::Arc<Vec<String>> {
        let mut slot = self.file_index.borrow_mut();
        if let Some((walked_at, files)) = slot.as_ref() {
            if walked_at.elapsed() < FILE_INDEX_TTL {
                return std::sync::Arc::clone(files);
            }
        }
        let files = std::sync::Arc::new(crate::input::paths::list_project_files(&self.repo_root));
        *slot = Some((std::time::Instant::now(), std::sync::Arc::clone(&files)));
        files
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

}


/// Single-step ease helper for display_tok_* fields.
/// Takes ONE bounded step toward `target` per call — NOT a loop.
/// gap<70 → +3; gap<200 → ceil(gap*0.15); else +50.  Clamps to target.
/// If display > target (shouldn't happen normally), reset to target instantly.
fn ease_display_tok(display: &mut Option<u64>, target: Option<u64>) {
    if let Some(t) = target {
        let d = display.get_or_insert(0);
        if *d < t {
            let gap = t - *d;
            let step: u64 = if gap < 70 {
                3
            } else if gap < 200 {
                (gap as f64 * 0.15).ceil() as u64
            } else {
                50
            };
            *d = (*d + step).min(t);
        } else if *d > t {
            // Token counts should only go up during a turn; on a new turn reset instantly.
            *d = t;
        }
    }
    // If target is None, leave display as-is (None means no data yet).
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
mod tests;
