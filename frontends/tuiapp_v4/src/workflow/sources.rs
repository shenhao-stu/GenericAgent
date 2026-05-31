//! workflow/sources.rs — the three per-source POLLERS that turn live conductor /
//! hive / goal state into [`Workflow`] values (checklist §7; recon §2.3 / §3.2 /
//! §3.3 / §5.3). Each is a thin I/O layer over the PURE mappers in
//! [`crate::workflow::schema`]: it does the socket/file read, parses the raw JSON
//! shape, and hands plain data to the mapper. Keeping the parsing of the raw
//! payloads in pure helpers (`parse_conductor_items`, `parse_bbs_authors`, …)
//! lets us unit-test the wire→model mapping without sockets.
//!
//! NONE of these block longer than the bounded [`crate::workflow::http::HTTP_TIMEOUT`]
//! (HTTP) or a single file read (goal/board), so a watcher thread calling them can
//! never stall the chat (§3 "never blocks chat").

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::workflow::http;
use crate::workflow::schema::{
    self, down_workflow, workflow_from_conductor, workflow_from_goal, workflow_from_hive,
    FeedItem, HiveAuthor, RawGoalState, RawSubagent, Workflow, WorkflowKind, WorkflowProgress,
};

// ===========================================================================
// Conductor poller (HTTP GET /subagent + GET /chat).
// ===========================================================================

/// The conductor's well-known localhost port (`conductor.py:25-29`).
pub const CONDUCTOR_PORT: u16 = 8900;
pub const LOCALHOST: &str = "127.0.0.1";

/// Poll the conductor: `GET /subagent` for the node snapshot + `GET /chat` for the
/// feed. Returns a live [`Workflow`] if the server answers, else a `running:false`
/// down placeholder so the panel still shows the Conductor group with a launch
/// hint (recon §7). `tombstones_prev` is the panel's previously-shown node list so
/// vanished (aborted) subagents are kept as tombstones (recon §5.4.5); `now` is the
/// clock for tombstone aging. Effectful (two bounded HTTP calls).
pub fn poll_conductor(
    port: u16,
    tombstones_prev: &[schema::WorkflowNode],
    now: f64,
) -> Workflow {
    let Some(resp) = http::get(LOCALHOST, port, "/subagent") else {
        return down_workflow(
            &format!("conductor@{port}"),
            WorkflowKind::Conductor,
            &format!("conductor :{port}"),
            &format!("http://127.0.0.1:{port}/subagent"),
        );
    };
    if !resp.is_ok() {
        return down_workflow(
            &format!("conductor@{port}"),
            WorkflowKind::Conductor,
            &format!("conductor :{port}"),
            &format!("http://127.0.0.1:{port}/subagent"),
        );
    }
    let subs = parse_conductor_items(&resp.body);
    // Optional chat feed (best-effort; a feed failure must not down the workflow).
    let feed = http::get(LOCALHOST, port, "/chat?last=20")
        .filter(|r| r.is_ok())
        .map(|r| parse_conductor_chat(&r.body))
        .unwrap_or_default();

    let mut wf = workflow_from_conductor(port, &subs, feed, true);
    // Apply client-side tombstones over the JUST-built subagent leaves (the root is
    // synthetic + never tombstoned). Split the root from the leaves, tombstone the
    // leaves against the previous LEAF set, then reassemble.
    if !tombstones_prev.is_empty() {
        let (roots, leaves): (Vec<_>, Vec<_>) = wf
            .nodes
            .into_iter()
            .partition(|n| n.role == schema::NodeRole::Conductor);
        let prev_leaves: Vec<schema::WorkflowNode> = tombstones_prev
            .iter()
            .filter(|n| n.role == schema::NodeRole::Subagent)
            .cloned()
            .collect();
        let merged = schema::apply_conductor_tombstones(
            &leaves,
            &prev_leaves,
            now,
            schema::CONDUCTOR_TOMBSTONE_TTL_SECS,
        );
        let mut nodes = roots;
        nodes.extend(merged);
        wf.nodes = nodes;
    }
    wf
}

/// The conductor `GET /subagent` → `{"items":[…]}` wrapper (recon §2.3).
#[derive(Debug, Clone, Deserialize, Default)]
struct ConductorItems {
    #[serde(default)]
    items: Vec<RawSubagent>,
}

/// Parse the conductor `/subagent` body into raw subagent items (PURE). Tolerant:
/// a body that is a bare array (no `items` wrapper) is also accepted. Returns an
/// empty vec on invalid JSON (the workflow then shows just the conductor root).
pub fn parse_conductor_items(body: &str) -> Vec<RawSubagent> {
    let b = body.trim();
    if let Ok(wrapped) = serde_json::from_str::<ConductorItems>(b) {
        return wrapped.items;
    }
    serde_json::from_str::<Vec<RawSubagent>>(b).unwrap_or_default()
}

/// One conductor chat item (`{id, role, msg, ts, read}`, recon §2.1).
#[derive(Debug, Clone, Deserialize, Default)]
struct ConductorChatItem {
    #[serde(default)]
    role: String,
    #[serde(default)]
    msg: String,
    #[serde(default)]
    ts: f64,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct ConductorChat {
    #[serde(default)]
    items: Vec<ConductorChatItem>,
}

/// Parse the conductor `/chat` body into feed items (PURE). The conductor stamps
/// `ts` in MILLISECONDS (recon §2.1), so divide to seconds for a uniform feed.
pub fn parse_conductor_chat(body: &str) -> Vec<FeedItem> {
    let b = body.trim();
    let items = serde_json::from_str::<ConductorChat>(b)
        .map(|c| c.items)
        .or_else(|_| serde_json::from_str::<Vec<ConductorChatItem>>(b))
        .unwrap_or_default();
    items
        .into_iter()
        .map(|it| FeedItem {
            ts: it.ts / 1000.0, // ms → s
            author: it.role,
            text: it.msg,
            post_id: 0,
        })
        .collect()
}

// ===========================================================================
// Hive poller (board.json discovery + BBS GET /authors /count /posts /poll).
// ===========================================================================

/// A discovered hive board: its short name (from `temp/hive_<name>`), port + key
/// (read from `<dir>/board.json`, written at launch — recon §5.4.4).
#[derive(Debug, Clone, PartialEq)]
pub struct HiveBoard {
    /// The hive short name (the `temp/hive_<name>` suffix).
    pub name: String,
    /// The BBS port.
    pub port: u16,
    /// The board auth key.
    pub key: String,
    /// The board working dir (`temp/hive_<name>`).
    pub dir: PathBuf,
    /// When the board process started (`board.json.started_at`, `time.time()` secs);
    /// `0.0` if the launcher predates the field. Drives the hive's uptime marker.
    pub started_at: f64,
    /// The board's working directory as the launcher recorded it
    /// (`board.json.cwd`); shown in the detail overlay's source uri. Empty if absent.
    pub cwd: String,
}

/// The `board.json` shape `agent_bbs.py` persists at launch (recon §5.4.4 / the
/// additive `_write_board_json`): `{port, key, started_at, cwd}`.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct RawBoardJson {
    #[serde(default)]
    pub port: u16,
    #[serde(default)]
    pub key: String,
    #[serde(default)]
    pub started_at: f64,
    #[serde(default)]
    pub cwd: String,
}

/// Discover running hives by scanning `<repo>/temp/hive_*` for a `board.json`
/// (recon §5.3 "scan `temp/hive_*` dirs"). A dir without a parseable `board.json`
/// is skipped (the watcher can't auth without the key). Effectful (a dir read +
/// a file read per hive); bounded by the small number of hive dirs.
pub fn discover_hives(repo_root: &Path) -> Vec<HiveBoard> {
    let mut out = Vec::new();
    let temp = repo_root.join("temp");
    let Ok(entries) = std::fs::read_dir(&temp) else {
        return out;
    };
    for entry in entries.flatten() {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        let Some(name) = hive_name_from_dir(&dir) else {
            continue;
        };
        let board_path = dir.join("board.json");
        let Ok(body) = std::fs::read_to_string(&board_path) else {
            continue;
        };
        if let Some(board) = parse_board_json(&name, &dir, &body) {
            out.push(board);
        }
    }
    // Stable order by name so the panel + tests are deterministic.
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Extract the hive short name from a `temp/hive_<name>` dir path (PURE). Returns
/// `None` for a dir that isn't a `hive_*` dir.
pub fn hive_name_from_dir(dir: &Path) -> Option<String> {
    let base = dir.file_name()?.to_string_lossy();
    base.strip_prefix("hive_").map(|s| s.to_string())
}

/// Parse a `board.json` body into a [`HiveBoard`] (PURE). Returns `None` if the
/// port/key are missing (can't poll a board without them). The `started_at`/`cwd`
/// provenance is carried through (used by [`decorate_with_board`]).
pub fn parse_board_json(name: &str, dir: &Path, body: &str) -> Option<HiveBoard> {
    let raw: RawBoardJson = serde_json::from_str(body.trim()).ok()?;
    if raw.port == 0 || raw.key.trim().is_empty() {
        return None;
    }
    Some(HiveBoard {
        name: name.to_string(),
        port: raw.port,
        key: raw.key,
        dir: dir.to_path_buf(),
        started_at: raw.started_at,
        cwd: raw.cwd,
    })
}

/// Poll one hive board into a [`Workflow`] (recon §3.2 panel poll plan):
///   1. `GET /authors?key=` → the node list;
///   2. `GET /count?author=&key=` per author → post counts;
///   3. `GET /posts?author=&limit=1&key=` per author → last-post ts + excerpt;
///   4. `GET /poll?since_id=0&limit=20&key=` → the feed tail;
///   5. an optional master `goal_state.json` under the hive dir → progress.
/// Returns a `running:false` placeholder if `/authors` is unreachable. `now` is
/// the clock for the last-post-age liveness heuristic. Effectful (a few bounded
/// HTTP calls); the per-author calls are capped so a huge board can't fan out.
pub fn poll_hive(board: &HiveBoard, now: f64) -> Workflow {
    let auth = |path: &str| format!("{path}{}key={}", if path.contains('?') { "&" } else { "?" }, board.key);

    let Some(authors_resp) = http::get(LOCALHOST, board.port, &auth("/authors")) else {
        return down_workflow(
            &format!("hive:{}", board.name),
            WorkflowKind::Hive,
            &format!("hive {}", board.name),
            &format!("http://127.0.0.1:{}/?key=…", board.port),
        );
    };
    if !authors_resp.is_ok() {
        return down_workflow(
            &format!("hive:{}", board.name),
            WorkflowKind::Hive,
            &format!("hive {}", board.name),
            &format!("http://127.0.0.1:{}/?key=…", board.port),
        );
    }
    let author_names = parse_bbs_authors(&authors_resp.body);

    // Per-author count + last post (capped at a sane fan-out).
    let mut authors: Vec<HiveAuthor> = Vec::with_capacity(author_names.len());
    for name in author_names.iter().take(MAX_HIVE_AUTHORS) {
        let count = http::get(LOCALHOST, board.port, &auth(&format!("/count?author={}", url_encode(name))))
            .filter(|r| r.is_ok())
            .and_then(|r| parse_bbs_count(&r.body))
            .unwrap_or(0);
        let (last_ts, last_post) = http::get(
            LOCALHOST,
            board.port,
            &auth(&format!("/posts?author={}&limit=1", url_encode(name))),
        )
        .filter(|r| r.is_ok())
        .map(|r| parse_bbs_last_post(&r.body))
        .unwrap_or((0.0, String::new()));
        authors.push(HiveAuthor {
            name: name.clone(),
            post_count: count,
            last_ts,
            last_post,
        });
    }

    // The feed tail (newest posts).
    let feed = http::get(LOCALHOST, board.port, &auth("/poll?since_id=0&limit=20"))
        .filter(|r| r.is_ok())
        .map(|r| parse_bbs_feed(&r.body))
        .unwrap_or_default();

    // Optional master progress: a goal_state.json under the hive dir (recon §3.3
    // "a master goal_state.json").
    let progress = read_hive_progress(&board.dir, now);

    let mut wf = workflow_from_hive(&board.name, board.port, &authors, feed, progress, now, true);
    // Enrich the workflow with the board.json provenance the launcher recorded: the
    // working dir (so the detail overlay shows WHERE the hive runs) + a synthetic
    // "started" feed line carrying the board's uptime (so a freshly-launched hive
    // with no posts yet still shows it is alive and for how long).
    decorate_with_board(&mut wf, board, now);
    wf
}

/// Fold a discovered board's `board.json` provenance (`cwd`, `started_at`) into the
/// hive [`Workflow`] (recon §5.4.4): the cwd becomes part of the source uri, and a
/// `started_at` yields a leading "started Ns ago" feed item so even a postless hive
/// shows liveness + uptime. PURE over `(wf, board, now)` (no I/O).
pub fn decorate_with_board(wf: &mut Workflow, board: &HiveBoard, now: f64) {
    if !board.cwd.is_empty() {
        wf.source_uri = format!("http://127.0.0.1:{}/  ·  {}", board.port, board.cwd);
    }
    if board.started_at > 0.0 {
        let uptime = (now - board.started_at).max(0.0);
        // Prepend an oldest-first "started" marker so the detail feed reads
        // chronologically (the launcher event, then the posts).
        let started = FeedItem {
            ts: board.started_at,
            author: "board".to_string(),
            text: format!("started · up {}s", uptime as u64),
            post_id: 0,
        };
        wf.feed.insert(0, started);
    }
}

/// Max authors we fan out per-author count/post calls to (a guard against a board
/// with an unexpected swarm; the cap is far above the SOP's "no more than 5").
pub const MAX_HIVE_AUTHORS: usize = 16;

/// Parse the BBS `GET /authors` body — a JSON array of strings (recon §3.2). PURE.
pub fn parse_bbs_authors(body: &str) -> Vec<String> {
    serde_json::from_str::<Vec<String>>(body.trim()).unwrap_or_default()
}

/// Parse the BBS `GET /count` body `{"total": N}` → the count (recon §3.2). PURE.
pub fn parse_bbs_count(body: &str) -> Option<u64> {
    #[derive(Deserialize)]
    struct Count {
        #[serde(default)]
        total: u64,
    }
    serde_json::from_str::<Count>(body.trim()).ok().map(|c| c.total)
}

/// One BBS post row (`{id, author, content, created_at}`, recon §3.2).
#[derive(Debug, Clone, Deserialize, Default)]
struct RawPost {
    #[serde(default)]
    id: u64,
    #[serde(default)]
    author: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    created_at: f64,
}

/// Parse a BBS `GET /posts?author=&limit=1` body (newest-first array) → the latest
/// post's `(created_at, excerpt)` (recon §3.2). `(0.0, "")` if empty. PURE.
pub fn parse_bbs_last_post(body: &str) -> (f64, String) {
    let posts: Vec<RawPost> = serde_json::from_str(body.trim()).unwrap_or_default();
    match posts.first() {
        Some(p) => (p.created_at, p.content.clone()),
        None => (0.0, String::new()),
    }
}

/// Parse a BBS `GET /poll` body (ascending array) → feed items (recon §3.2). PURE.
pub fn parse_bbs_feed(body: &str) -> Vec<FeedItem> {
    let posts: Vec<RawPost> = serde_json::from_str(body.trim()).unwrap_or_default();
    posts
        .into_iter()
        .map(|p| FeedItem {
            ts: p.created_at,
            author: p.author,
            text: p.content,
            post_id: p.id,
        })
        .collect()
}

/// Read a hive master's `goal_state.json` (if present under the hive dir) into
/// progress (recon §3.2 step 4). `None` if absent/invalid. Effectful (one read).
pub fn read_hive_progress(dir: &Path, now: f64) -> Option<WorkflowProgress> {
    let body = std::fs::read_to_string(dir.join("goal_state.json")).ok()?;
    let raw: RawGoalState = serde_json::from_str(body.trim()).ok()?;
    Some(workflow_from_goal(&raw, now).progress.unwrap_or(WorkflowProgress {
        turns_used: raw.turns_used,
        max_turns: raw.max_turns,
        elapsed_sec: 0,
        budget_sec: raw.budget_seconds.max(0.0) as u64,
    }))
}

/// Minimal URL-encode for a query VALUE: percent-encode the bytes that would break
/// a query (space, `&`, `?`, `#`, `=`, `%`, `+`). Author names are usually plain
/// ASCII (`hive-worker-1`), so this is mostly a no-op; it just keeps an exotic name
/// from corrupting the query. PURE.
pub fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

// ===========================================================================
// Goal poller (temp/goal_state.json mtime + parse).
// ===========================================================================

/// Read + parse `temp/goal_state.json` into a Goal [`Workflow`] (recon §3.3).
/// Returns `None` if the file is absent (no goal running) — the watcher then omits
/// the Goal group entirely (vs a hive/conductor "down" placeholder, since a goal
/// file's ABSENCE means "no goal", not "goal server down"). `now` is the clock.
pub fn poll_goal(repo_root: &Path, now: f64) -> Option<Workflow> {
    let path = repo_root.join("temp").join("goal_state.json");
    let body = std::fs::read_to_string(&path).ok()?;
    schema::parse_goal_state(&body, now)
}

/// The mtime (ns) of `temp/goal_state.json` for the watcher's change-detect poll
/// (recon §5.3 "poll file mtime"). `None` if absent. Effectful (a stat). PURE-ish.
pub fn goal_state_mtime_ns(repo_root: &Path) -> Option<u128> {
    let path = repo_root.join("temp").join("goal_state.json");
    let meta = std::fs::metadata(&path).ok()?;
    let mtime = meta.modified().ok()?;
    mtime
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_nanos())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_conductor_items_handles_wrapper_and_bare() {
        // The documented `{"items":[…]}` wrapper.
        let wrapped = r#"{"items":[{"id":"ab12","prompt":"task","reply":"r","status":"running","created_at":1.0,"updated_at":2.0}]}"#;
        let subs = parse_conductor_items(wrapped);
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].id, "ab12");
        assert_eq!(subs[0].status, "running");

        // A bare array is also tolerated.
        let bare = r#"[{"id":"cd34","status":"stopped"}]"#;
        let subs = parse_conductor_items(bare);
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].id, "cd34");

        // Invalid JSON → empty (the workflow shows just the conductor root).
        assert!(parse_conductor_items("nope").is_empty());
        assert!(parse_conductor_items("{}").is_empty());
    }

    #[test]
    fn parse_conductor_chat_ms_to_seconds() {
        let body = r#"{"items":[{"role":"conductor","msg":"hi","ts":1730000000000.0}]}"#;
        let feed = parse_conductor_chat(body);
        assert_eq!(feed.len(), 1);
        assert_eq!(feed[0].author, "conductor");
        assert_eq!(feed[0].text, "hi");
        assert!((feed[0].ts - 1730000000.0).abs() < 1.0, "ms converted to s");
    }

    #[test]
    fn parse_bbs_shapes() {
        // /authors → array of strings.
        let authors = parse_bbs_authors(r#"["hive-master","hive-worker-1"]"#);
        assert_eq!(authors, vec!["hive-master".to_string(), "hive-worker-1".to_string()]);
        assert!(parse_bbs_authors("nope").is_empty());

        // /count → {"total": N}.
        assert_eq!(parse_bbs_count(r#"{"total":7}"#), Some(7));
        assert_eq!(parse_bbs_count("garbage"), None);

        // /posts?limit=1 → newest-first array; take the first.
        let (ts, excerpt) = parse_bbs_last_post(r#"[{"id":42,"author":"w","content":"latest","created_at":123.5}]"#);
        assert_eq!(ts, 123.5);
        assert_eq!(excerpt, "latest");
        let (ts0, e0) = parse_bbs_last_post("[]");
        assert_eq!((ts0, e0.as_str()), (0.0, ""));

        // /poll → ascending feed.
        let feed = parse_bbs_feed(r#"[{"id":1,"author":"a","content":"first","created_at":1.0},{"id":2,"author":"b","content":"second","created_at":2.0}]"#);
        assert_eq!(feed.len(), 2);
        assert_eq!(feed[0].post_id, 1);
        assert_eq!(feed[1].author, "b");
    }

    #[test]
    fn board_json_parse_and_name_extraction() {
        let dir = PathBuf::from("/repo/temp/hive_alpha");
        assert_eq!(hive_name_from_dir(&dir).as_deref(), Some("alpha"));
        assert_eq!(hive_name_from_dir(&PathBuf::from("/repo/temp/not_a_hive")), None);

        let body = r#"{"port":5001,"key":"sekret","started_at":1730000000.0,"cwd":"/repo/temp/hive_alpha"}"#;
        let board = parse_board_json("alpha", &dir, body).expect("valid board.json");
        assert_eq!(board.port, 5001);
        assert_eq!(board.key, "sekret");
        assert_eq!(board.name, "alpha");
        // The provenance fields (started_at / cwd) are carried through from board.json.
        assert_eq!(board.started_at, 1730000000.0);
        assert_eq!(board.cwd, "/repo/temp/hive_alpha");

        // decorate_with_board folds that provenance into the hive workflow: the cwd
        // joins the source uri and a "started" feed marker carries the uptime.
        let mut wf = workflow_from_hive("alpha", 5001, &[], vec![], None, 1730000100.0, true);
        decorate_with_board(&mut wf, &board, 1730000100.0);
        assert!(wf.source_uri.contains("/repo/temp/hive_alpha"), "cwd in source uri: {}", wf.source_uri);
        assert_eq!(wf.feed.first().map(|f| f.author.as_str()), Some("board"));
        assert!(wf.feed[0].text.contains("up 100s"), "uptime feed: {}", wf.feed[0].text);

        // A board with no provenance leaves the workflow untouched (no panic, no marker).
        let bare = HiveBoard { name: "b".into(), port: 5002, key: "k".into(), dir: dir.clone(), started_at: 0.0, cwd: String::new() };
        let mut wf2 = workflow_from_hive("b", 5002, &[], vec![], None, 1.0, true);
        let uri_before = wf2.source_uri.clone();
        decorate_with_board(&mut wf2, &bare, 1.0);
        assert_eq!(wf2.source_uri, uri_before, "no cwd → uri unchanged");
        assert!(wf2.feed.is_empty(), "no started_at → no marker");

        // Missing port/key → None (can't poll without them).
        assert!(parse_board_json("alpha", &dir, r#"{"started_at":1.0}"#).is_none());
        assert!(parse_board_json("alpha", &dir, r#"{"port":5001}"#).is_none());
        assert!(parse_board_json("alpha", &dir, "garbage").is_none());
    }

    #[test]
    fn url_encode_is_passthrough_for_plain_names() {
        assert_eq!(url_encode("hive-worker-1"), "hive-worker-1");
        assert_eq!(url_encode("a b"), "a%20b");
        assert_eq!(url_encode("x&y"), "x%26y");
    }

    /// `discover_hives` scans `temp/hive_*` for board.json and skips dirs without
    /// one (the on-disk discovery path, using a real temp dir).
    #[test]
    fn discover_hives_scans_board_json() {
        let root = std::env::temp_dir().join(format!("tui_v4_wf_hives_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let temp = root.join("temp");
        // A valid hive dir with a board.json.
        let h1 = temp.join("hive_alpha");
        std::fs::create_dir_all(&h1).unwrap();
        std::fs::write(h1.join("board.json"), r#"{"port":5001,"key":"k1"}"#).unwrap();
        // A hive dir WITHOUT a board.json → skipped.
        std::fs::create_dir_all(temp.join("hive_beta")).unwrap();
        // A non-hive dir → skipped.
        std::fs::create_dir_all(temp.join("model_responses")).unwrap();

        let boards = discover_hives(&root);
        assert_eq!(boards.len(), 1, "only the hive with a board.json is discovered");
        assert_eq!(boards[0].name, "alpha");
        assert_eq!(boards[0].port, 5001);
        assert_eq!(boards[0].key, "k1");

        let _ = std::fs::remove_dir_all(&root);
    }

    /// `poll_goal` reads + parses a real temp/goal_state.json, and returns None when
    /// the file is absent (no goal running).
    #[test]
    fn poll_goal_reads_file_or_none() {
        let root = std::env::temp_dir().join(format!("tui_v4_wf_goal_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        // Absent → None (no goal group shown).
        assert!(poll_goal(&root, 0.0).is_none());
        assert!(goal_state_mtime_ns(&root).is_none());

        // Present → parsed Goal workflow.
        std::fs::create_dir_all(root.join("temp")).unwrap();
        std::fs::write(
            root.join("temp").join("goal_state.json"),
            r#"{"objective":"do the thing","start_time":1000.0,"turns_used":3,"max_turns":50,"status":"running"}"#,
        )
        .unwrap();
        let wf = poll_goal(&root, 1100.0).expect("present goal parses");
        assert_eq!(wf.title, "do the thing");
        assert_eq!(wf.progress.unwrap().turns_used, 3);
        assert!(goal_state_mtime_ns(&root).is_some());

        let _ = std::fs::remove_dir_all(&root);
    }
}
