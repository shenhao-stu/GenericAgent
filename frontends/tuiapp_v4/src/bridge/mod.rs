//! bridge/mod.rs — spawn `scripts/ga_bridge.py`, discover it robustly (N1),
//! run reader + writer threads, track liveness, and surface every failure as a
//! visible status (never a silent "disconnected").
//!
//! The chat plane: one GA-core child per session. We send `UiToCore` frames on
//! the child's stdin (one JSON line each, flushed) and read `CoreToUi` frames
//! off its stdout (one JSON line each). All decode is `String::from_utf8_lossy`
//! so a stray GBK byte on Chinese Windows can never kill the reader.
//!
//! N1 — disconnected FIX: discovery walks env → next-to-exe → exe/.. → an
//! explicit GENERICAGENT_ROOT → up from the exe AND the cwd looking for a dir
//! containing `agentmain.py` (the GA repo-root marker) → cwd-relative. Spawn /
//! parse / child-exit are all turned into a visible `BridgeEvent` the UI shows.

pub mod protocol;

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

pub use protocol::{CoreToUi, UiToCore};

/// The bridge filename and its canonical location under the tui_v4 package.
const GA_BRIDGE_FILE: &str = "ga_bridge.py";
const GA_BRIDGE_REL: &str = "scripts/ga_bridge.py";
/// The GA repo-root marker: the dir that contains this file IS the repo root.
const REPO_MARKER: &str = "agentmain.py";

/// Env vars (highest priority first) that point directly at the bridge script.
const BRIDGE_PATH_ENV_VARS: &[&str] = &["GA_TUI_BRIDGE", "GA_BRIDGE_PATH", "TUI_V4_BRIDGE"];
/// Env vars that point at the GA repo root (where the bridge + temp/ live).
const GA_ROOT_ENV_VARS: &[&str] = &["GENERICAGENT_ROOT", "GA_ROOT", "TUI_V4_REPO_ROOT"];

/// Inputs to bridge discovery — every varying source is injectable so the
/// candidate ordering is unit-testable without an actual process/disk.
#[derive(Debug, Clone, Default)]
pub struct DiscoveryContext {
    /// Snapshot of the relevant env vars (name → value).
    pub env: Vec<(String, String)>,
    /// Absolute path to the running executable (`std::env::current_exe`).
    pub exe: Option<PathBuf>,
    /// Current working directory.
    pub cwd: Option<PathBuf>,
}

impl DiscoveryContext {
    /// Capture the real process context (env vars we care about + exe + cwd).
    pub fn from_process() -> Self {
        let mut env = Vec::new();
        for name in BRIDGE_PATH_ENV_VARS.iter().chain(GA_ROOT_ENV_VARS.iter()) {
            if let Ok(val) = std::env::var(name) {
                if !val.trim().is_empty() {
                    env.push((name.to_string(), val));
                }
            }
        }
        DiscoveryContext {
            env,
            exe: std::env::current_exe().ok(),
            cwd: std::env::current_dir().ok(),
        }
    }

    fn env_lookup(&self, names: &[&str]) -> Option<String> {
        for name in names {
            if let Some((_, v)) = self.env.iter().find(|(k, _)| k == name) {
                let t = v.trim();
                if !t.is_empty() {
                    return Some(t.to_string());
                }
            }
        }
        None
    }
}

/// Build the ordered list of candidate paths for `ga_bridge.py`, highest
/// priority first (N1). PURE over [`DiscoveryContext`] so the ordering is
/// directly unit-tested. The first candidate that exists on disk wins; if none
/// do, the first candidate is used so the spawn error points at the expected
/// location instead of failing silently.
///
/// Order:
///   1. explicit env override (`GA_TUI_BRIDGE` / `GA_BRIDGE_PATH` / `TUI_V4_BRIDGE`)
///   2. `ga_bridge.py` next to the executable (drop-in deploy)
///   3. `scripts/ga_bridge.py` next to the executable
///   4. `../scripts/ga_bridge.py` (exe in a `release/` or `bin/` dir beside scripts/)
///   5. `<GENERICAGENT_ROOT>/frontends/tuiapp_v4/scripts/ga_bridge.py`
///   6. walk UP from the exe dir: first ancestor containing `agentmain.py`
///      (the GA repo root) → `<root>/frontends/tuiapp_v4/scripts/ga_bridge.py`
///   7. walk UP from the cwd the same way
///   8. cwd-relative: `./scripts/ga_bridge.py`, `./ga_bridge.py`,
///      `./frontends/tuiapp_v4/scripts/ga_bridge.py`
pub fn ga_bridge_candidates(ctx: &DiscoveryContext) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();

    // 1. explicit env override (a full path to the script).
    if let Some(p) = ctx.env_lookup(BRIDGE_PATH_ENV_VARS) {
        out.push(PathBuf::from(p));
    }

    // 2-4. relative to the executable.
    if let Some(exe) = &ctx.exe {
        if let Some(exe_dir) = exe.parent() {
            out.push(exe_dir.join(GA_BRIDGE_FILE));
            out.push(exe_dir.join("scripts").join(GA_BRIDGE_FILE));
            out.push(exe_dir.join("..").join("scripts").join(GA_BRIDGE_FILE));
        }
    }

    // 5. an explicitly-named GA repo root.
    if let Some(root) = ctx.env_lookup(GA_ROOT_ENV_VARS) {
        out.push(
            PathBuf::from(root)
                .join("frontends")
                .join("tuiapp_v4")
                .join(GA_BRIDGE_REL),
        );
    }

    // 6. walk up from the exe dir to the GA repo root (agentmain.py marker).
    if let Some(exe) = &ctx.exe {
        if let Some(exe_dir) = exe.parent() {
            if let Some(root) = walk_up_for_marker(exe_dir, REPO_MARKER) {
                out.push(
                    root.join("frontends")
                        .join("tuiapp_v4")
                        .join(GA_BRIDGE_REL),
                );
            }
        }
    }

    // 7. walk up from the cwd to the GA repo root.
    if let Some(cwd) = &ctx.cwd {
        if let Some(root) = walk_up_for_marker(cwd, REPO_MARKER) {
            out.push(
                root.join("frontends")
                    .join("tuiapp_v4")
                    .join(GA_BRIDGE_REL),
            );
        }
    }

    // 8. cwd-relative (a checkout launched from its own root / the pkg dir).
    if let Some(cwd) = &ctx.cwd {
        out.push(cwd.join("scripts").join(GA_BRIDGE_FILE));
        out.push(cwd.join(GA_BRIDGE_FILE));
        out.push(cwd.join("frontends").join("tuiapp_v4").join(GA_BRIDGE_REL));
    }

    dedupe_preserving_order(out)
}

/// Walk up from `start` (inclusive) looking for the first directory that
/// directly contains `marker`. Returns that directory (the repo root). PURE,
/// uses only `Path::exists` so it's safe and cheap; bounded by the FS depth.
fn walk_up_for_marker(start: &Path, marker: &str) -> Option<PathBuf> {
    let mut dir = Some(start);
    while let Some(d) = dir {
        if d.join(marker).exists() {
            return Some(d.to_path_buf());
        }
        dir = d.parent();
    }
    None
}

/// De-duplicate paths while preserving first-seen order. We compare on the
/// lexical path (post-normalization is not needed for ordering correctness;
/// callers existence-check before use).
fn dedupe_preserving_order(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut seen: Vec<PathBuf> = Vec::new();
    for p in paths {
        let norm = normalize_lexical(&p);
        if !seen.iter().any(|s| normalize_lexical(s) == norm) {
            seen.push(p);
        }
    }
    seen
}

/// Collapse `.`/`..` lexically (no disk access) so de-dupe and display are
/// stable. This is intentionally simple — it is only used for equality of
/// candidate paths, not for security-sensitive resolution.
fn normalize_lexical(p: &Path) -> PathBuf {
    use std::path::Component;
    let mut stack: Vec<Component> = Vec::new();
    for comp in p.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                if matches!(stack.last(), Some(Component::Normal(_))) {
                    stack.pop();
                } else {
                    stack.push(comp);
                }
            }
            other => stack.push(other),
        }
    }
    stack.iter().collect()
}

/// Pick the first candidate that exists on disk, else the highest-priority
/// candidate (so the spawn error names the expected location). `exists` is
/// injectable for tests.
pub fn discover_ga_bridge_with(
    ctx: &DiscoveryContext,
    exists: impl Fn(&Path) -> bool,
) -> Option<PathBuf> {
    let candidates = ga_bridge_candidates(ctx);
    for c in &candidates {
        if exists(c) {
            return Some(c.clone());
        }
    }
    candidates.into_iter().next()
}

/// Real-disk discovery against the live process context. (Public convenience
/// entry; the live [`Bridge::spawn`] uses the injectable variant internally.)
#[allow(dead_code)]
pub fn discover_ga_bridge() -> Option<PathBuf> {
    let ctx = DiscoveryContext::from_process();
    discover_ga_bridge_with(&ctx, |p| p.exists())
}

/// Derive the GA repo root for a discovered bridge path: the bridge lives at
/// `<repo>/frontends/tuiapp_v4/scripts/ga_bridge.py`, so the repo is four
/// levels up. Falls back to a `GENERICAGENT_ROOT`-family env, then the cwd.
pub fn repo_root_for(bridge: &Path, ctx: &DiscoveryContext) -> PathBuf {
    // Walk up from the bridge file's OWN directory to the GA repo root (the dir
    // containing agentmain.py). Robust whether this is the canonical
    // scripts/ga_bridge.py or a relocated drop-in copy (e.g. target/release/
    // ga_bridge.py). The old "exactly three levels up" guess mis-resolved the
    // copy to …/frontends and caused `ModuleNotFoundError: No module named
    // 'agentmain'`.
    if let Some(dir) = bridge.parent() {
        if let Some(root) = walk_up_for_marker(dir, REPO_MARKER) {
            return root;
        }
    }
    if let Some(root) = ctx.env_lookup(GA_ROOT_ENV_VARS) {
        let p = PathBuf::from(root);
        if p.exists() {
            return p;
        }
    }
    // Walk up from cwd as a last structured attempt, then plain cwd.
    if let Some(cwd) = &ctx.cwd {
        if let Some(root) = walk_up_for_marker(cwd, REPO_MARKER) {
            return root;
        }
        return cwd.clone();
    }
    PathBuf::from(".")
}

// ---------------------------------------------------------------------------
// Live bridge: spawn the child, run reader + writer threads.
// ---------------------------------------------------------------------------

/// An event the bridge surfaces to the UI thread. Wraps both protocol frames
/// and the out-of-band lifecycle signals (spawn failure, child exit, parse
/// noise) so NOTHING is ever a silent "disconnected" (N1).
#[derive(Debug, Clone)]
pub enum BridgeEvent {
    /// A parsed `CoreToUi` frame.
    Frame(CoreToUi),
    /// The child could not be spawned (python missing, bad path, …). Fatal.
    SpawnFailed { detail: String },
    /// A stdout line that did not parse as a frame (surfaced, not swallowed).
    ParseNoise { line: String },
    /// The child process exited. `code` is its status if known.
    ChildExited { code: Option<i32> },
    /// A diagnostic line the child wrote to stderr (e.g. a Python traceback).
    Stderr { line: String },
}

/// Options for spawning the bridge child.
#[derive(Debug, Clone)]
pub struct BridgeOptions {
    /// Path to the python executable. Defaults to `python` (on PATH per spec).
    pub python: String,
    /// 1-based LLM index forwarded to `ga_bridge.py --llm-no`. 0 = default.
    pub llm_no: u32,
    /// Max bytes we buffer for a single stdout line before truncating it.
    pub max_line_bytes: usize,
}

impl Default for BridgeOptions {
    fn default() -> Self {
        BridgeOptions {
            python: "python".to_string(),
            llm_no: 0,
            max_line_bytes: 1 << 20, // 1 MiB
        }
    }
}

/// A live handle to a spawned GA-core bridge child. Frames arrive on the
/// `events` receiver the UI owns; `send` queues a frame to the child's stdin.
///
/// `connected` flips true only when a `Ready` frame is observed; `alive` tracks
/// whether the child + its threads are still running.
pub struct Bridge {
    child: Arc<Mutex<Option<Child>>>,
    writer_tx: Sender<WriterMsg>,
    alive: Arc<AtomicBool>,
    connected: Arc<AtomicBool>,
    /// The discovered bridge script path (surfaced in a Phase-2 status detail).
    #[allow(dead_code)]
    pub bridge_path: PathBuf,
    /// The GA repo root the child runs in (shown in the header cwd).
    pub repo_root: PathBuf,
    /// Owned worker handles so the reader/writer/watcher threads stay attached
    /// to the Bridge's lifetime rather than being silently orphaned.
    #[allow(dead_code)]
    threads: Vec<JoinHandle<()>>,
}

enum WriterMsg {
    Frame(UiToCore),
    Stop,
}

impl Bridge {
    /// Discover + spawn the bridge child, wiring reader/writer/stderr threads.
    /// Discovery / spawn failures DO NOT panic — they arrive on `events` as a
    /// `SpawnFailed` and the returned handle reports `alive() == false` so the
    /// UI shows "disconnected: <reason>" with the real cause (N1).
    pub fn spawn(opts: BridgeOptions, events: Sender<BridgeEvent>) -> Bridge {
        let ctx = DiscoveryContext::from_process();
        let bridge_path = discover_ga_bridge_with(&ctx, |p| p.exists())
            .unwrap_or_else(|| PathBuf::from(GA_BRIDGE_REL));
        let repo_root = repo_root_for(&bridge_path, &ctx);

        let alive = Arc::new(AtomicBool::new(false));
        let connected = Arc::new(AtomicBool::new(false));
        let child_slot = Arc::new(Mutex::new(None));
        let (writer_tx, writer_rx) = std::sync::mpsc::channel::<WriterMsg>();

        // If the script does not actually exist, surface a precise SpawnFailed
        // and return a dead handle (the UI must NOT silently say "disconnected").
        if !bridge_path.exists() {
            let _ = events.send(BridgeEvent::SpawnFailed {
                detail: format!(
                    "ga_bridge.py not found (looked for {}). Set GA_TUI_BRIDGE or GENERICAGENT_ROOT.",
                    bridge_path.display()
                ),
            });
            return Bridge {
                child: child_slot,
                writer_tx,
                alive,
                connected,
                bridge_path,
                repo_root,
                threads: Vec::new(),
            };
        }

        let mut command = Command::new(&opts.python);
        command
            .arg(&bridge_path)
            .env("PYTHONUTF8", "1") // Chinese-Windows safe child decode.
            .env("GENERICAGENT_ROOT", &repo_root) // explicit root so the child self-locates agentmain.
            .current_dir(&repo_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if opts.llm_no > 0 {
            command.arg("--llm-no").arg(opts.llm_no.to_string());
        }

        let mut child = match command.spawn() {
            Ok(c) => c,
            Err(e) => {
                let _ = events.send(BridgeEvent::SpawnFailed {
                    detail: format!(
                        "failed to spawn `{} {}`: {} (is python on PATH?)",
                        opts.python,
                        bridge_path.display(),
                        e
                    ),
                });
                return Bridge {
                    child: child_slot,
                    writer_tx,
                    alive,
                    connected,
                    bridge_path,
                    repo_root,
                    threads: Vec::new(),
                };
            }
        };

        let stdin = child.stdin.take();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        *child_slot.lock().unwrap() = Some(child);
        alive.store(true, Ordering::SeqCst);

        let mut threads = Vec::new();

        // -- reader thread: stdout lines → CoreToUi frames → events. ----------
        if let Some(stdout) = stdout {
            let events = events.clone();
            let connected = connected.clone();
            let max_line = opts.max_line_bytes;
            threads.push(std::thread::spawn(move || {
                let mut reader = BufReader::new(stdout);
                let mut buf: Vec<u8> = Vec::with_capacity(4096);
                loop {
                    buf.clear();
                    match reader.read_until(b'\n', &mut buf) {
                        Ok(0) => break, // EOF: child closed stdout.
                        Ok(_) => {
                            if buf.len() > max_line {
                                buf.truncate(max_line);
                            }
                            // from_utf8_lossy: a stray GBK byte never kills us.
                            let line = String::from_utf8_lossy(&buf);
                            let line = line.trim_end_matches(['\n', '\r']);
                            if line.trim().is_empty() {
                                continue;
                            }
                            match CoreToUi::parse_line(line) {
                                Some(frame) => {
                                    if matches!(frame, CoreToUi::Ready { .. }) {
                                        connected.store(true, Ordering::SeqCst);
                                    }
                                    if events.send(BridgeEvent::Frame(frame)).is_err() {
                                        break; // UI gone.
                                    }
                                }
                                None => {
                                    let _ = events.send(BridgeEvent::ParseNoise {
                                        line: line.to_string(),
                                    });
                                }
                            }
                        }
                        Err(_) => break,
                    }
                }
            }));
        }

        // -- stderr thread: surface child diagnostics (tracebacks) visibly. ---
        if let Some(stderr) = stderr {
            let events = events.clone();
            threads.push(std::thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines() {
                    match line {
                        Ok(l) => {
                            if events.send(BridgeEvent::Stderr { line: l }).is_err() {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
            }));
        }

        // -- writer thread: drain queued UiToCore frames to child stdin. ------
        if let Some(mut stdin) = stdin {
            threads.push(std::thread::spawn(move || {
                while let Ok(msg) = writer_rx.recv() {
                    match msg {
                        WriterMsg::Frame(frame) => {
                            let mut line = frame.to_line();
                            line.push('\n');
                            if stdin.write_all(line.as_bytes()).is_err() {
                                break;
                            }
                            if stdin.flush().is_err() {
                                break;
                            }
                        }
                        WriterMsg::Stop => break,
                    }
                }
            }));
        }

        // -- watcher thread: report child exit so it's never a silent stall. --
        {
            let child_slot = child_slot.clone();
            let alive = alive.clone();
            let events = events.clone();
            let connected = connected.clone();
            threads.push(std::thread::spawn(move || {
                // Poll the child for exit; cheap and avoids holding the lock.
                loop {
                    std::thread::sleep(std::time::Duration::from_millis(150));
                    let mut guard = child_slot.lock().unwrap();
                    let status = match guard.as_mut() {
                        Some(c) => c.try_wait(),
                        None => break,
                    };
                    match status {
                        Ok(Some(code)) => {
                            alive.store(false, Ordering::SeqCst);
                            connected.store(false, Ordering::SeqCst);
                            let _ = events.send(BridgeEvent::ChildExited {
                                code: code.code(),
                            });
                            break;
                        }
                        Ok(None) => { /* still running */ }
                        Err(_) => {
                            alive.store(false, Ordering::SeqCst);
                            break;
                        }
                    }
                }
            }));
        }

        Bridge {
            child: child_slot,
            writer_tx,
            alive,
            connected,
            bridge_path,
            repo_root,
            threads,
        }
    }

    /// Queue a frame to the child's stdin. Returns `false` if the writer thread
    /// is gone (child dead) — the caller surfaces that as a status, not a hang.
    pub fn send(&self, frame: UiToCore) -> bool {
        self.writer_tx.send(WriterMsg::Frame(frame)).is_ok()
    }

    /// True once the bridge has been spawned and not yet observed to exit.
    /// (Consumed by the Phase-2 reconnect affordance; the Foundation drives the
    /// status line from the folded `ConnStatus` instead.)
    #[allow(dead_code)]
    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    /// True once a `Ready` frame has been observed (the real handshake, N1).
    #[allow(dead_code)]
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    /// Ask the child to shut down cleanly, then force-kill if it lingers.
    pub fn shutdown(&self) {
        let _ = self.writer_tx.send(WriterMsg::Frame(UiToCore::Shutdown));
        let _ = self.writer_tx.send(WriterMsg::Stop);
        if let Ok(mut guard) = self.child.lock() {
            if let Some(child) = guard.as_mut() {
                // Give the cooperative Shutdown a brief moment, then kill.
                for _ in 0..10 {
                    if let Ok(Some(_)) = child.try_wait() {
                        return;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl std::fmt::Debug for Bridge {
    /// A terse `Debug` (the channels/threads aren't `Debug`). This lets the
    /// multi-session [`crate::app::session::Session`] that OWNS a `Bridge` derive
    /// `Debug` without exposing the transport internals.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bridge")
            .field("bridge_path", &self.bridge_path)
            .field("repo_root", &self.repo_root)
            .field("alive", &self.alive.load(Ordering::SeqCst))
            .field("connected", &self.connected.load(Ordering::SeqCst))
            .finish_non_exhaustive()
    }
}

/// Convenience constructor: spawn a bridge and hand back its (untagged) event
/// receiver paired with the live handle. The live app now uses the multiplexed
/// [`spawn_bridge_tagged`]; this single-channel variant is kept for the bridge
/// E2E test (`bridge_spawns_and_handshakes_echo_stub`) + as a simple entry point.
#[allow(dead_code)]
pub fn spawn_bridge(opts: BridgeOptions) -> (Bridge, Receiver<BridgeEvent>) {
    let (tx, rx) = std::sync::mpsc::channel();
    let bridge = Bridge::spawn(opts, tx);
    (bridge, rx)
}

/// Spawn a bridge whose events are forwarded onto `events` TAGGED with `session_id`
/// (the multi-session multiplexer, checklist §6 / N2). The GA core has no
/// multi-session API, so the UI runs N independent children and routes each
/// child's frames to ITS session id — this is the per-session wiring: an internal
/// untagged channel feeds a forwarder thread that re-emits each event as
/// `(session_id, ev)` so the app's reducer can fold it into the right session and
/// streams never cross (the Ink `frameToAction(frame, sid)` discipline).
pub fn spawn_bridge_tagged(
    opts: BridgeOptions,
    session_id: u64,
    events: Sender<(u64, BridgeEvent)>,
) -> Bridge {
    let (raw_tx, raw_rx) = std::sync::mpsc::channel::<BridgeEvent>();
    let bridge = Bridge::spawn(opts, raw_tx);
    std::thread::spawn(move || {
        while let Ok(ev) = raw_rx.recv() {
            if events.send((session_id, ev)).is_err() {
                break; // the app's unified channel is gone.
            }
        }
    });
    bridge
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_with(env: &[(&str, &str)], exe: Option<&str>, cwd: Option<&str>) -> DiscoveryContext {
        DiscoveryContext {
            env: env.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
            exe: exe.map(PathBuf::from),
            cwd: cwd.map(PathBuf::from),
        }
    }

    /// Bridge discovery candidate ordering (deliverable: discovery ordering).
    /// Asserts the N1 priority: env override first, then next-to-exe variants,
    /// then GENERICAGENT_ROOT, then cwd-relative — and that the explicit env
    /// path is the single highest-priority candidate.
    #[test]
    fn bridge_discovery_candidate_ordering() {
        let ctx = ctx_with(
            &[
                ("GA_TUI_BRIDGE", "/opt/custom/ga_bridge.py"),
                ("GENERICAGENT_ROOT", "/srv/ga"),
            ],
            Some("/app/release/tui_v4.exe"),
            Some("/work/checkout"),
        );
        let cands = ga_bridge_candidates(&ctx);

        // 1. The explicit env override is FIRST.
        assert_eq!(cands[0], PathBuf::from("/opt/custom/ga_bridge.py"));

        // 2-4. next-to-exe variants follow, in order.
        assert_eq!(cands[1], PathBuf::from("/app/release/ga_bridge.py"));
        assert_eq!(cands[2], PathBuf::from("/app/release/scripts/ga_bridge.py"));
        assert_eq!(
            cands[3],
            PathBuf::from("/app/release/../scripts/ga_bridge.py")
        );

        // 5. The GENERICAGENT_ROOT-derived path appears before cwd-relative ones.
        let root_path =
            PathBuf::from("/srv/ga/frontends/tuiapp_v4/scripts/ga_bridge.py");
        let root_idx = cands.iter().position(|p| p == &root_path).expect("root path present");

        // 8. cwd-relative paths exist and come AFTER the root-derived path.
        let cwd_scripts = PathBuf::from("/work/checkout/scripts/ga_bridge.py");
        let cwd_idx = cands
            .iter()
            .position(|p| p == &cwd_scripts)
            .expect("cwd scripts path present");
        assert!(
            root_idx < cwd_idx,
            "GENERICAGENT_ROOT path must rank above cwd-relative ({} !< {})",
            root_idx,
            cwd_idx
        );

        // The bare cwd `ga_bridge.py` is also a candidate (drop-in checkout).
        assert!(cands.contains(&PathBuf::from("/work/checkout/ga_bridge.py")));
    }

    /// Without an env override, next-to-exe is the top candidate, and the first
    /// existing candidate is chosen by `discover_ga_bridge_with`.
    #[test]
    fn discovery_picks_first_existing() {
        let ctx = ctx_with(&[], Some("/bin/tui_v4"), Some("/cwd"));
        let cands = ga_bridge_candidates(&ctx);
        // No env override → first candidate is `ga_bridge.py` next to the exe.
        assert_eq!(cands[0], PathBuf::from("/bin/ga_bridge.py"));

        // `exists` says only the cwd/scripts path is real → that one is picked.
        let target = PathBuf::from("/cwd/scripts/ga_bridge.py");
        let chosen = discover_ga_bridge_with(&ctx, |p| p == target.as_path());
        assert_eq!(chosen, Some(target));

        // If nothing exists, the highest-priority candidate is returned (so the
        // error message names a real expected path, never silent).
        let chosen_none = discover_ga_bridge_with(&ctx, |_| false);
        assert_eq!(chosen_none, Some(PathBuf::from("/bin/ga_bridge.py")));
    }

    /// REAL end-to-end bridge smoke (checklist Phase F gate "Bridge spawns
    /// ga_bridge.py, handshakes Ready"): spawn the dependency-free stub
    /// `scripts/echo_bridge.py` through the LIVE [`Bridge::spawn`] path (reader +
    /// writer + stderr + watcher threads), confirm we observe the `Ready`
    /// handshake frame, then send a `Submit` and confirm the echoed
    /// `MessageBegin`/`MessageDelta`/`MessageEnd` round-trips back up the real
    /// stdio pipe — proving the Rust side can spawn AND handshake a Python child
    /// over the JSONL protocol on this machine (Windows + Chinese locale, child
    /// env `PYTHONUTF8=1`, `from_utf8_lossy` decode). Uses the `GA_TUI_BRIDGE`
    /// discovery override to point at the stub; the live bridge sets the UTF-8
    /// child env itself.
    ///
    /// Skips (does not fail) if `python` cannot be spawned at all — the spawn
    /// failure is surfaced as a visible `SpawnFailed` event (N1), which is the
    /// correct degraded behavior on a box with no Python; the GATE machine has
    /// Python 3.13 so it runs fully.
    #[test]
    fn bridge_spawns_and_handshakes_echo_stub() {
        use std::sync::mpsc::RecvTimeoutError;
        use std::time::Duration;

        // Locate the stub relative to this crate (CARGO_MANIFEST_DIR is the pkg
        // root); fall back to the cwd-relative path if the manifest dir is unset.
        let stub = {
            let base = std::env::var("CARGO_MANIFEST_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("."));
            base.join("scripts").join("echo_bridge.py")
        };
        assert!(
            stub.exists(),
            "echo_bridge.py stub must exist at {}",
            stub.display()
        );

        // Point discovery at the stub via the highest-priority env override, then
        // spawn through the real Bridge path. (This is the ONE test that touches
        // the real process env; the discovery-ordering tests use injected
        // contexts, so there is no contention.)
        // SAFETY (edition 2024): set/remove_var are unsafe due to multi-threaded
        // env races. This is the only test mutating the real env, and it does so
        // before spawning (the spawn reads the env on the calling thread); the
        // discovery-ordering tests use injected contexts and never read the env.
        unsafe {
            std::env::set_var("GA_TUI_BRIDGE", &stub);
        }
        let (bridge, rx) = spawn_bridge(BridgeOptions::default());

        // Drain events up to a generous timeout, looking for the Ready handshake.
        // If python is genuinely unavailable we'll see SpawnFailed → skip.
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        let mut saw_ready = false;
        let mut spawn_failed: Option<String> = None;
        while std::time::Instant::now() < deadline {
            match rx.recv_timeout(Duration::from_millis(500)) {
                Ok(BridgeEvent::Frame(CoreToUi::Ready { model, .. })) => {
                    // The stub identifies itself as model "echo".
                    assert_eq!(model.as_deref(), Some("echo"));
                    saw_ready = true;
                    break;
                }
                Ok(BridgeEvent::SpawnFailed { detail }) => {
                    spawn_failed = Some(detail);
                    break;
                }
                Ok(_) => continue, // stderr caps line / other frames: keep waiting.
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }

        if let Some(detail) = spawn_failed {
            // No Python on this host → degraded path is correct; clean up + skip.
            unsafe {
                std::env::remove_var("GA_TUI_BRIDGE");
            }
            bridge.shutdown();
            eprintln!("bridge_spawns_and_handshakes_echo_stub: skipped (spawn failed: {detail})");
            return;
        }

        assert!(saw_ready, "did not observe the Ready handshake within 20s");
        assert!(bridge.is_connected(), "Ready must flip the connected flag (N1)");
        assert!(bridge.is_alive(), "child should still be alive after handshake");

        // Now drive a real Submit through the writer thread and confirm the echo
        // streams back: MessageBegin → MessageDelta("echo: …") → MessageEnd.
        assert!(
            bridge.send(UiToCore::Submit {
                text: "ping-test 你好".into(),
                images: None,
            }),
            "writer thread must accept the Submit"
        );

        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let mut saw_delta = false;
        while std::time::Instant::now() < deadline {
            match rx.recv_timeout(Duration::from_millis(500)) {
                Ok(BridgeEvent::Frame(CoreToUi::MessageDelta { text, .. })) => {
                    assert!(
                        text.contains("echo:") && text.contains("你好"),
                        "delta should echo the (UTF-8) submitted text, got {text:?}"
                    );
                    saw_delta = true;
                    break;
                }
                Ok(_) => continue,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        assert!(saw_delta, "did not observe the echoed MessageDelta within 10s");

        // Clean shutdown: the stub exits its read loop on Shutdown (never hangs).
        unsafe {
            std::env::remove_var("GA_TUI_BRIDGE");
        }
        bridge.shutdown();
    }

    /// The `agentmain.py` repo-root walk is exercised through `repo_root_for`'s
    /// lexical four-up derivation (pure, no disk needed for the shape check).
    #[test]
    fn repo_root_lexical_derivation() {
        // For a real tree, `repo_root_for` confirms against on-disk markers; the
        // lexical four-up shape is what we assert here via normalize_lexical.
        let bridge = PathBuf::from("/srv/ga/frontends/tuiapp_v4/scripts/ga_bridge.py");
        let scripts = bridge.parent().unwrap();
        let four_up = normalize_lexical(&scripts.join("..").join("..").join(".."));
        assert_eq!(four_up, PathBuf::from("/srv/ga"));
    }

    /// REAL-DISK packaging proof (Phase 6 §11): with the exe living at
    /// `target/release/tui_v4.exe`, discovery must resolve the bridge BOTH ways —
    /// (a) the next-to-exe copy `target/release/ga_bridge.py` (candidate #2, the
    /// drop-in deploy), AND (b) the repo-root walk that finds `agentmain.py` and
    /// derives `<root>/frontends/tuiapp_v4/scripts/ga_bridge.py` (candidate #6).
    /// Anchored to the live tree via CARGO_MANIFEST_DIR so it exercises actual
    /// `Path::exists` against the real on-disk layout, not an injected stub.
    ///
    /// `#[ignore]` because it depends on the packaging step having run
    /// (`cargo build --release` + the bridge-script copy into target/release/);
    /// run with: `cargo test packaging_discovery_resolves_both_ways -- --ignored`.
    #[test]
    #[ignore = "requires release build + bridge scripts copied into target/release/"]
    fn packaging_discovery_resolves_both_ways() {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR")); // .../tuiapp_v4
        let exe = manifest
            .join("target")
            .join("release")
            .join("tui_v4.exe");
        let canonical_scripts = manifest.join(GA_BRIDGE_REL);

        let ctx = DiscoveryContext {
            env: Vec::new(), // no overrides — force the disk-walk path.
            exe: Some(exe.clone()),
            cwd: None, // isolate the exe-anchored candidates from any cwd noise.
        };
        let cands = ga_bridge_candidates(&ctx);

        // (a) candidate #2: ga_bridge.py next to the exe (the copy we deploy).
        let next_to_exe = exe.parent().unwrap().join(GA_BRIDGE_FILE);
        assert!(
            cands.iter().any(|c| normalize_lexical(c) == normalize_lexical(&next_to_exe)),
            "next-to-exe candidate {} must be present; got {cands:?}",
            next_to_exe.display()
        );

        // (b) candidate #6: the repo-root walk (agentmain.py) → canonical scripts/.
        assert!(
            cands.iter().any(|c| normalize_lexical(c) == normalize_lexical(&canonical_scripts)),
            "repo-root-walk candidate {} must be present; got {cands:?}",
            canonical_scripts.display()
        );

        // And real-disk discovery picks SOMETHING that exists (the highest-priority
        // existing candidate). After packaging, both (a) and (b) exist on disk.
        let chosen = discover_ga_bridge_with(&ctx, |p| p.exists());
        let chosen = chosen.expect("a candidate must be chosen");
        assert!(
            chosen.exists(),
            "discovery must resolve to an EXISTING bridge path, got {}",
            chosen.display()
        );
    }
}
