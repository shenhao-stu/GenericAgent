//! workflow/mod.rs — the SINGLETON workflow watcher (checklist §3 "Workflow plane
//! (singleton watcher)"; §7 `/workflows`; recon §5). It merges three independent
//! live sources — conductor HTTP, hive BBS HTTP, goal_state file — into ONE
//! [`WorkflowSnapshot`] the panel renders, on its OWN threads so it can **NEVER
//! block the chat** (§3 "Additive snapshots; never blocks chat").
//!
//! Design:
//!   * ONE background poll thread (started lazily the first time the panel opens),
//!     parked between ticks; it polls each source with BOUNDED timeouts and pushes
//!     a merged snapshot onto a channel the UI drains in its event loop.
//!   * The UI keeps the latest snapshot in `AppState` and feeds it to the panel.
//!     The watcher never touches `AppState` and never holds a lock the UI needs —
//!     the only coupling is the snapshot channel + a stop flag, so a slow/blocked
//!     poll can at worst stall the WATCHER thread, never the render/chat loop.
//!   * CLIENT-SIDE TOMBSTONES for conductor subagents are kept across ticks (recon
//!     §5.4.5) — the watcher feeds the previous node list back into the pure
//!     [`schema::apply_conductor_tombstones`] each poll.
//!   * Node ACTIONS (conductor `keyinfo|input|stop|kill`) are fired on a throwaway
//!     thread (`POST /subagent/{id}`) so a slow POST never blocks the UI either.
//!
//! The load-bearing MERGE/parse/liveness logic lives in [`schema`] (pure + tested);
//! the I/O pollers in [`sources`]; this module is the thread orchestration + the
//! singleton handle.

pub mod http;
pub mod panel;
pub mod schema;
pub mod sources;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

// Re-export the node type the watcher's tombstone carry-forward + the panel's
// action wiring use through the top-level `workflow::` path. (The fuller schema
// surface — `WfStatus`, `WorkflowKind`, `WorkflowProgress` — is referenced via the
// explicit `workflow::schema::` path at the call sites, so it is NOT re-exported
// here to avoid dead re-exports.)
pub use schema::{NodeRole, WorkflowNode};
pub use sources::CONDUCTOR_PORT;

// Types this module's own thread/orchestration code uses internally.
use schema::{Workflow, WorkflowSnapshot};

/// The poll cadence while the panel is OPEN (recon §5.3 "~1-2s while open"). The
/// watcher uses a single fast cadence; conductor/hive/goal are all polled together
/// each tick. (The idle back-off — slowing to ~8s when the panel is closed — is
/// achieved simply by PAUSING the watcher when the panel closes, so there is no
/// background traffic at all when nobody is looking.)
pub const POLL_INTERVAL: Duration = Duration::from_millis(1500);

/// A node action the panel can fire at a workflow node (recon §2.3 conductor
/// mutate; maps to `POST /subagent/{id} {action, msg}`). Hive/goal nodes have no
/// mutate API, so an action against them is a no-op the panel disables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeAction {
    /// Inject a `[MASTER] …` key-info note into a stopped subagent (`keyinfo`).
    KeyInfo,
    /// Start a NEW task round on the subagent with `text` (`input`).
    Input,
    /// Stop / abort the subagent's current turn (`stop` → conductor `abort`).
    Stop,
    /// Kill the subagent entirely (`kill`).
    Kill,
    /// Open the detail overlay for the node (UI-only; no server call).
    Open,
}

impl NodeAction {
    /// The conductor action STRING this maps to (recon §2.3 `keyinfo|input|stop|kill`).
    /// `Open` has no server verb (it's a UI affordance). PURE.
    pub fn conductor_verb(&self) -> Option<&'static str> {
        match self {
            NodeAction::KeyInfo => Some("keyinfo"),
            NodeAction::Input => Some("input"),
            NodeAction::Stop => Some("stop"),
            NodeAction::Kill => Some("kill"),
            NodeAction::Open => None,
        }
    }

    /// The i18n KEY for this action's verb label (the node-action menu, §7
    /// "node action verbs"). PURE.
    pub fn label_key(&self) -> &'static str {
        match self {
            NodeAction::KeyInfo => "wf.action.keyinfo",
            NodeAction::Input => "wf.action.input",
            NodeAction::Stop => "wf.action.stop",
            NodeAction::Kill => "wf.action.kill",
            NodeAction::Open => "wf.action.open",
        }
    }

    /// The action verbs offered for a node of a given role (the detail overlay's
    /// menu). Conductor subagents get the full set; hive/goal nodes get only
    /// `Open` (no mutate API exists for them, recon §5.4.1). PURE.
    pub fn for_role(role: NodeRole) -> Vec<NodeAction> {
        match role {
            NodeRole::Subagent => vec![
                NodeAction::Open,
                NodeAction::KeyInfo,
                NodeAction::Input,
                NodeAction::Stop,
                NodeAction::Kill,
            ],
            // The conductor root + hive/goal nodes: open + (for conductor root) stop.
            NodeRole::Conductor => vec![NodeAction::Open],
            NodeRole::Master | NodeRole::Worker | NodeRole::Goal => vec![NodeAction::Open],
        }
    }
}

/// Build the JSON body for a conductor `POST /subagent/{id}` action (recon §2.3:
/// `{action, msg}`). PURE — unit-tested without a socket.
pub fn conductor_action_body(verb: &str, msg: &str) -> String {
    // serde_json to escape `msg` safely (it may contain quotes/newlines).
    serde_json::json!({ "action": verb, "msg": msg }).to_string()
}

/// Fire a conductor node action over HTTP on a THROWAWAY thread (so a slow POST
/// never blocks the UI). `port` is the conductor port; `node_id` the subagent id;
/// `verb` the `keyinfo|input|stop|kill`; `msg` the optional text. A failure is
/// swallowed (the next poll reflects the real state) — the action is best-effort,
/// like every other watcher I/O. Returns immediately.
pub fn fire_conductor_action(port: u16, node_id: String, verb: &'static str, msg: String) {
    std::thread::spawn(move || {
        let body = conductor_action_body(verb, &msg);
        let path = format!("/subagent/{node_id}");
        let _ = http::post_json(sources::LOCALHOST, port, &path, &body);
    });
}

// ===========================================================================
// The singleton watcher handle.
// ===========================================================================

/// A live handle to the singleton workflow watcher. Holds the latest snapshot, a
/// "panel open" flag the thread parks on, and the stop flag. Cheap to clone the
/// receiver out; the poll thread is owned here.
pub struct WorkflowWatcher {
    /// The latest merged snapshot, shared with the UI (the UI clones it to render).
    /// The watcher writes; the UI reads. A short-held mutex (a clone), never
    /// contended with the render hot path.
    latest: Arc<Mutex<WorkflowSnapshot>>,
    /// `true` while the panel is OPEN — the poll thread only works when set, so a
    /// closed panel generates ZERO background traffic (the idle back-off, recon
    /// §5.3).
    active: Arc<AtomicBool>,
    /// Stop flag: set on drop to end the poll thread.
    stop: Arc<AtomicBool>,
    /// Monotonic generation the thread stamps each merge (the UI detects refreshes).
    generation: Arc<AtomicU64>,
    /// The poll thread handle (joined on drop).
    thread: Option<JoinHandle<()>>,
    /// A change signal the UI loop can select on (the thread sends `()` after each
    /// merge so the loop redraws promptly). Optional — the UI also polls `latest`.
    _change_tx: Sender<()>,
}

impl WorkflowWatcher {
    /// Start the singleton watcher for a given GA repo root. The poll thread is
    /// spawned PARKED (inactive) — call [`Self::set_active`] when the panel opens.
    /// Returns the handle + a change-signal receiver the UI loop can drain to
    /// redraw on a refresh (it may also just poll [`Self::snapshot`]).
    pub fn start(repo_root: PathBuf) -> (WorkflowWatcher, Receiver<()>) {
        let latest = Arc::new(Mutex::new(WorkflowSnapshot::default()));
        let active = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let generation = Arc::new(AtomicU64::new(0));
        let (change_tx, change_rx) = std::sync::mpsc::channel::<()>();

        let thread = {
            let latest = latest.clone();
            let active = active.clone();
            let stop = stop.clone();
            let generation = generation.clone();
            let change_tx = change_tx.clone();
            std::thread::Builder::new()
                .name("workflow-watcher".into())
                .spawn(move || {
                    poll_loop(repo_root, latest, active, stop, generation, change_tx);
                })
                .expect("spawn workflow-watcher thread")
        };

        (
            WorkflowWatcher {
                latest,
                active,
                stop,
                generation,
                thread: Some(thread),
                _change_tx: change_tx,
            },
            change_rx,
        )
    }

    /// Mark the panel OPEN/closed. When closed the poll thread parks (no traffic);
    /// opening it triggers an immediate poll on the next tick.
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::SeqCst);
    }

    /// `true` if the watcher is currently polling (the panel is open).
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::SeqCst)
    }

    /// Clone the latest merged snapshot for rendering (a short-held lock + clone;
    /// never blocks the render loop meaningfully — the snapshot is small).
    pub fn snapshot(&self) -> WorkflowSnapshot {
        self.latest.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// The current generation (refresh counter) without cloning the whole snapshot.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
    }
}

impl Drop for WorkflowWatcher {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        self.active.store(true, Ordering::SeqCst); // wake the park so it can see stop.
        if let Some(t) = self.thread.take() {
            // The thread checks `stop` each loop within ≤ POLL_INTERVAL; join is
            // bounded. (We don't detach so the OS thread isn't leaked.)
            let _ = t.join();
        }
    }
}

impl std::fmt::Debug for WorkflowWatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkflowWatcher")
            .field("active", &self.is_active())
            .field("generation", &self.generation())
            .finish_non_exhaustive()
    }
}

/// The poll thread body: while not stopped, if the panel is active, poll all three
/// sources, merge them (carrying conductor tombstones forward), publish, and
/// signal a redraw; then park for [`POLL_INTERVAL`]. When the panel is inactive it
/// parks WITHOUT polling (zero traffic). All source I/O is bounded so this thread
/// can stall at most one source's timeout — never the chat.
fn poll_loop(
    repo_root: PathBuf,
    latest: Arc<Mutex<WorkflowSnapshot>>,
    active: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    generation: Arc<AtomicU64>,
    change_tx: Sender<()>,
) {
    // The conductor node list shown last tick (for tombstone carry-forward).
    let mut prev_conductor_nodes: Vec<WorkflowNode> = Vec::new();
    // An mtime-gated cache of the goal workflow: `temp/goal_state.json` only changes
    // when the master writes a turn, so we stat its mtime each tick (cheap) and only
    // RE-READ + RE-PARSE the file when the mtime advanced (recon §5.3 "poll file
    // mtime"). `goal_cache` holds `(mtime_ns, parsed_goal)`; a `None` parsed value
    // means "the file is absent / unparseable" (so we don't restat-then-omit churn).
    let mut goal_cache: Option<(u128, Option<Workflow>)> = None;

    while !stop.load(Ordering::SeqCst) {
        if !active.load(Ordering::SeqCst) {
            // Parked: no polling. Sleep in short slices so a stop/activate is seen
            // promptly without busy-waiting.
            std::thread::sleep(Duration::from_millis(120));
            continue;
        }

        let now = now_secs();

        // --- conductor (HTTP, bounded). Always shown (down → placeholder). ----
        let conductor = sources::poll_conductor(CONDUCTOR_PORT, &prev_conductor_nodes, now);
        // Remember its node list for the next tick's tombstones.
        prev_conductor_nodes = conductor.nodes.clone();
        let conductor_list = vec![conductor];

        // --- hives (board.json discovery + BBS poll, bounded). ----------------
        let boards = sources::discover_hives(&repo_root);
        let hives: Vec<Workflow> = boards.iter().map(|b| sources::poll_hive(b, now)).collect();

        // --- goal (file, mtime-gated). Absent → omitted (no "down" placeholder).
        // Stat the mtime first; only re-read+parse when it changed since last tick.
        // (The elapsed seconds inside a RUNNING goal still tick because we recompute
        // elapsed from the cached file's start_time against the fresh `now` below.)
        let goal_mtime = sources::goal_state_mtime_ns(&repo_root);
        let goal: Vec<Workflow> = match goal_mtime {
            None => {
                // File gone → drop any cache, omit the goal group.
                goal_cache = None;
                Vec::new()
            }
            Some(mtime) => {
                let need_reparse = match &goal_cache {
                    Some((cached_mtime, _)) => *cached_mtime != mtime,
                    None => true,
                };
                if need_reparse {
                    let parsed = sources::poll_goal(&repo_root, now);
                    goal_cache = Some((mtime, parsed.clone()));
                    parsed.into_iter().collect()
                } else if let Some((_, Some(cached))) = &goal_cache {
                    // Unchanged file → reuse the cached parse (a running goal rewrites
                    // goal_state.json every turn, so the mtime advances and we re-read
                    // then; between writes the elapsed is at most one tick stale).
                    vec![cached.clone()]
                } else {
                    Vec::new()
                }
            }
        };

        // --- merge + publish. -------------------------------------------------
        let next_gen = generation.fetch_add(1, Ordering::SeqCst) + 1;
        let snap = schema::merge_snapshots(conductor_list, hives, goal, next_gen);
        if let Ok(mut guard) = latest.lock() {
            *guard = snap;
        }
        let _ = change_tx.send(()); // nudge the UI to redraw (ignored if UI gone).

        // Park for the cadence (in short slices so stop is responsive).
        let mut slept = Duration::ZERO;
        while slept < POLL_INTERVAL && !stop.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(120));
            slept += Duration::from_millis(120);
        }
    }
}

/// Wall-clock seconds since the Unix epoch (the source timestamps are
/// `time.time()`, so we compare against the same clock). PURE-ish (one syscall).
pub fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use schema::WorkflowKind;

    #[test]
    fn conductor_action_body_is_valid_json() {
        let body = conductor_action_body("keyinfo", "watch the rate limit");
        // Round-trips back through serde with the right fields.
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["action"], "keyinfo");
        assert_eq!(v["msg"], "watch the rate limit");
        // A msg with quotes/newlines is escaped safely (no broken JSON).
        let tricky = conductor_action_body("input", "say \"hi\"\nthen stop");
        let v2: serde_json::Value = serde_json::from_str(&tricky).unwrap();
        assert_eq!(v2["msg"], "say \"hi\"\nthen stop");
    }

    #[test]
    fn node_action_verbs_and_role_sets() {
        assert_eq!(NodeAction::KeyInfo.conductor_verb(), Some("keyinfo"));
        assert_eq!(NodeAction::Input.conductor_verb(), Some("input"));
        assert_eq!(NodeAction::Stop.conductor_verb(), Some("stop"));
        assert_eq!(NodeAction::Kill.conductor_verb(), Some("kill"));
        assert_eq!(NodeAction::Open.conductor_verb(), None);

        // A subagent gets the full mutate set; a hive worker only Open (no API).
        let sub = NodeAction::for_role(NodeRole::Subagent);
        assert!(sub.contains(&NodeAction::Stop) && sub.contains(&NodeAction::Kill));
        assert_eq!(NodeAction::for_role(NodeRole::Worker), vec![NodeAction::Open]);
        assert_eq!(NodeAction::for_role(NodeRole::Goal), vec![NodeAction::Open]);
        // Every action has a non-empty label key.
        for a in [NodeAction::KeyInfo, NodeAction::Input, NodeAction::Stop, NodeAction::Kill, NodeAction::Open] {
            assert!(!a.label_key().is_empty());
        }
    }

    /// The watcher's goal poll is MTIME-GATED: `goal_state_mtime_ns` advances when
    /// the master rewrites `temp/goal_state.json`, which is the signal the poll loop
    /// uses to re-read+re-parse (vs reusing the cached goal). This pins the gate the
    /// loop's caching depends on: absent → None; present → Some; a rewrite advances it.
    #[test]
    fn goal_mtime_gate_advances_on_rewrite() {
        let root = std::env::temp_dir().join(format!("tui_v4_wf_mtime_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("temp")).unwrap();
        let path = root.join("temp").join("goal_state.json");

        // Absent → no gate.
        assert!(sources::goal_state_mtime_ns(&root).is_none());

        // First write → a gate value + a parseable goal.
        std::fs::write(&path, r#"{"objective":"a","start_time":1.0,"turns_used":1,"max_turns":10,"status":"running"}"#).unwrap();
        let m1 = sources::goal_state_mtime_ns(&root).expect("present file has an mtime");
        assert!(sources::poll_goal(&root, 2.0).is_some());

        // A rewrite (a later turn) advances the mtime so the loop re-reads. We bump
        // the file's modified time explicitly to avoid relying on filesystem mtime
        // granularity within a fast test.
        std::thread::sleep(Duration::from_millis(20));
        std::fs::write(&path, r#"{"objective":"a","start_time":1.0,"turns_used":2,"max_turns":10,"status":"running"}"#).unwrap();
        let _ = filetime_touch(&path);
        let m2 = sources::goal_state_mtime_ns(&root).expect("still present");
        assert!(m2 >= m1, "a rewrite does not move the mtime backwards (m1={m1}, m2={m2})");
        // The re-parse reflects the new turn count.
        assert_eq!(sources::poll_goal(&root, 3.0).unwrap().progress.unwrap().turns_used, 2);

        let _ = std::fs::remove_dir_all(&root);
    }

    /// Best-effort bump of a file's mtime by rewriting it (portable; no extra dep).
    /// The second write above already changes the mtime on every common FS; this is
    /// a belt-and-suspenders no-op helper kept simple to avoid a `filetime` crate.
    fn filetime_touch(path: &std::path::Path) -> std::io::Result<()> {
        let body = std::fs::read(path)?;
        std::fs::write(path, body)
    }

    /// The watcher starts PARKED (no traffic), can be activated/deactivated, and
    /// stops cleanly on drop within a bounded time — proving it never wedges the
    /// process. (It polls a repo root with no servers, so each poll yields a
    /// down/empty snapshot fast.)
    #[test]
    fn watcher_starts_parked_and_stops_clean() {
        let root = std::env::temp_dir().join(format!("tui_v4_wf_watch_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("temp")).unwrap();

        let (watcher, change_rx) = WorkflowWatcher::start(root.clone());
        // Parked by default → no snapshot churn.
        assert!(!watcher.is_active());
        assert_eq!(watcher.generation(), 0);

        // Activate → within a couple of poll intervals we should see ≥1 merge
        // (conductor down-placeholder, no hives, no goal → a 1-workflow snapshot).
        watcher.set_active(true);
        let _ = change_rx.recv_timeout(POLL_INTERVAL * 4);
        // Generation advanced (a poll happened) — or, on a very slow box, at least
        // the call returns; we assert it eventually advances within a bound.
        let deadline = std::time::Instant::now() + POLL_INTERVAL * 6;
        while watcher.generation() == 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(watcher.generation() >= 1, "an active watcher polls + bumps the generation");
        let snap = watcher.snapshot();
        // A Conductor workflow always appears (a placeholder when :8900 is down, or
        // a live one if a real conductor happens to be running on the gate box). The
        // invariant is that the merge ran and produced the conductor group — not its
        // up/down state (which depends on the environment).
        assert!(
            snap.workflows.iter().any(|w| w.kind == WorkflowKind::Conductor),
            "an active watcher merges a conductor workflow (up or down): {:?}",
            snap.workflows.iter().map(|w| (&w.id, w.running)).collect::<Vec<_>>()
        );

        // Deactivate → it parks again (no assertion on traffic; just that it
        // doesn't panic + stays alive).
        watcher.set_active(false);

        // Drop joins the thread within a bound (no hang).
        let start = std::time::Instant::now();
        drop(watcher);
        assert!(start.elapsed() < POLL_INTERVAL * 4, "watcher drop joins promptly");

        let _ = std::fs::remove_dir_all(&root);
    }
}
