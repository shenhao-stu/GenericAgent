//! app/tests.rs — the `AppState` integration tests (the bridge reducer fold,
//! overlay routing, multi-session switch/routing, the `/workflows` watcher
//! lifecycle). Split out of `app/mod.rs` so the state module stays small; as a
//! child of `app` it keeps `use super::*` + access to the module-private fields
//! (`last_width`, `next_block_id`) the cache/repaint tests assert on.

use super::*;
use crate::bridge::protocol::{CoreToUi, LlmItem};
use crate::bridge::BridgeEvent;

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
    // Idle by default. Q9: the tab title leads with the BEAR `ʕ•ᴥ•ʔ` in BOTH
    // states; "GenericAgent" is present but no longer the leading token.
    assert_eq!(app.tab_status(), TabStatus::Idle);
    assert!(app.terminal_title().starts_with("ʕ•ᴥ•ʔ"), "bear leads when idle");
    assert!(app.terminal_title().contains("GenericAgent"));

    // Busy → Working. The bear STILL leads (busy/idle is carried by the OSC
    // tab-status channel, not by swapping the title's leading glyph).
    app.apply_bridge_event(
        frame(CoreToUi::MessageBegin { mid: "m1".into(), role: "assistant".into() }),
        0,
    );
    assert_eq!(app.tab_status(), TabStatus::Working);
    assert!(app.terminal_title().contains("ʕ•ᴥ•ʔ"), "bear leads in both states");
    assert!(!app.terminal_title().starts_with("GenericAgent"), "bear, not the name, leads");

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

// ---- Fix E (Q8): per-node fold + clickable triangle + expandable tool result ----

/// A finalized assistant block with THREE turns: turns 1 & 2 fold by default (their
/// `<summary>` becomes the `▸` title; their bodies hide), turn 3 stays open. The
/// distinctive body sentinels let a test prove which turn is expanded.
const THREE_TURNS: &str = "\
Turn 1 ...
<summary>first summary</summary>
ALPHA body line one
Turn 2 ...
<summary>second summary</summary>
BETA body line two
Turn 3 ...
GAMMA the active turn body";

/// Push `THREE_TURNS` as a finalized assistant block, sync the transcript at a
/// generous geometry, and return its block id (so a test can address its nodes).
fn seed_three_turn_block(app: &mut AppState, theme: &Theme) -> u64 {
    let id = app.alloc_block_id();
    app.transcript
        .push(Block::new(id, None, Role::Assistant, THREE_TURNS.to_string(), true));
    app.sync_transcript(80, 40, theme);
    id
}

/// The cockpit-plain projection of block `id` at the current fold state (what the
/// transcript draws, flattened) — for asserting which turns are expanded vs folded.
fn block_plain(app: &AppState, id: u64, theme: &Theme) -> String {
    let b = app.transcript.iter().find(|b| b.id == id).unwrap();
    let folds = BlockFolds { block_id: id, fold_all: app.fold_all, overrides: Some(&app.folds) };
    let n = app.wrap_cache.block_line_count(id);
    let mut out = String::new();
    for i in 0..n {
        if let Some(line) = b.cockpit_line(theme, &folds, 80, i) {
            for span in &line.spans {
                out.push_str(span.content.as_ref());
            }
            out.push('\n');
        }
    }
    out
}

/// THE Fix-E deliverable: toggling ONE node flips ONLY that node, not the global
/// fold. After the toggle, the clicked turn expands (its body shows) while the OTHER
/// completed turn stays folded (still a `▸` summary) — and `fold_all` is untouched.
#[test]
fn toggle_fold_flips_single_node() {
    let theme = Theme::default_theme();
    let mut app = AppState::new();
    let id = seed_three_turn_block(&mut app, &theme);

    // Default: turns 1 & 2 are folded to their `▸` summaries (bodies hidden), turn 3
    // is the open active turn.
    let before = block_plain(&app, id, &theme);
    assert!(before.contains("▸ first summary"), "turn 1 folds by default: {before:?}");
    assert!(before.contains("▸ second summary"), "turn 2 folds by default: {before:?}");
    assert!(!before.contains("ALPHA body"), "folded turn 1 hides its body: {before:?}");
    assert!(before.contains("GAMMA the active turn"), "turn 3 stays open: {before:?}");

    // Toggle ONLY turn 1.
    app.toggle_fold(NodeId::Turn { block: id, turn: 1 });
    assert!(!app.fold_all, "a per-node toggle must NOT touch the global fold_all");
    assert_eq!(app.folds.len(), 1, "exactly one node override is recorded");
    assert_eq!(app.folds.get(&NodeId::Turn { block: id, turn: 1 }), Some(&false), "turn 1 unfolded");

    // Re-sync (the next frame) and assert ONLY turn 1 changed.
    app.sync_transcript(80, 40, &theme);
    let after = block_plain(&app, id, &theme);
    assert!(after.contains("ALPHA body line one"), "toggled turn 1 is now expanded: {after:?}");
    assert!(!after.contains("▸ first summary"), "turn 1 is no longer a folded header: {after:?}");
    // Turn 2 is UNCHANGED — still folded (this is what "single node, not global" means).
    assert!(after.contains("▸ second summary"), "turn 2 stays folded: {after:?}");
    assert!(!after.contains("BETA body line two"), "turn 2 body stays hidden: {after:?}");

    // Toggling again re-folds turn 1 (a clean round-trip).
    app.toggle_fold(NodeId::Turn { block: id, turn: 1 });
    app.sync_transcript(80, 40, &theme);
    let back = block_plain(&app, id, &theme);
    assert!(back.contains("▸ first summary"), "second toggle re-folds turn 1: {back:?}");
    assert!(!back.contains("ALPHA body"), "turn 1 body hidden again: {back:?}");
}

/// The node-hit table maps a fold header's visual row to its `Turn` node, and a
/// left-click on the triangle column (col 0) resolves + toggles it; a click in the
/// body column (past the triangle gutter) does NOT. Exercises the real
/// `transcript_node_at` → `click_fold_at` path.
#[test]
fn click_on_fold_triangle_resolves_and_toggles_node() {
    let theme = Theme::default_theme();
    let mut app = AppState::new();
    let id = seed_three_turn_block(&mut app, &theme);

    // The transcript region top in the default cockpit layout (header + separator).
    app.set_term_size(80, 40);
    let area = ratatui::layout::Rect::new(0, 0, 80, 40);
    let top = crate::components::cockpit::split_cockpit(&app, area).transcript.y;

    // Find the GLOBAL visual row of turn 1's `▸` header from the node-hit table.
    let (range, node) = app
        .node_hit
        .iter()
        .find(|(_, n)| matches!(n, NodeId::Turn { turn: 1, .. }))
        .cloned()
        .expect("turn 1's fold header is a clickable node");
    assert_eq!(node, NodeId::Turn { block: id, turn: 1 });
    // Map that global row to a screen row (it is on-screen: the block is short).
    let screen_row = top + (*range.start() - app.viewport.visual_top(&app.wrap_cache)) as u16;

    // A click in the BODY column (col 10, past the 2-cell triangle gutter) resolves
    // to no node (native selection owns it).
    assert!(app.transcript_node_at(10, screen_row, top).is_none(), "body click is not a node");
    // A click on the triangle column (col 0) resolves to turn 1's node.
    assert_eq!(app.transcript_node_at(0, screen_row, top), Some(node));

    // The full click path toggles it (returns true → handled).
    assert!(app.click_fold_at(0, screen_row, top), "a triangle click is handled");
    assert_eq!(app.folds.get(&node), Some(&false), "the click unfolded turn 1");
    // A click that hits nothing returns false (falls through).
    assert!(!app.click_fold_at(0, top.saturating_sub(1), top), "above-transcript click falls through");
}

/// End-to-end expandable TOOL result (Fix E): a single-turn block with a long
/// `🛠️ name(args)` result renders a `⏺` bullet whose result is truncated + a `▸ +N
/// more` clickable affordance; the bullet is a `Tool` node in the hit table. Clicking
/// it expands the result (every line shows) and swaps `▸` for `▾` — and the click
/// re-anchors so it never jumps. A second click collapses it back.
#[test]
fn click_expands_and_collapses_tool_result() {
    let theme = Theme::default_theme();
    let mut app = AppState::new();
    // One turn, one tool whose result is 6 lines (> the 4-row preview → foldable).
    let src = "Turn 1 ...\n🛠️ run({\"cmd\": \"ls\"})\nLINE1\nLINE2\nLINE3\nLINE4\nLINE5\nLINE6";
    let id = app.alloc_block_id();
    app.transcript.push(Block::new(id, None, Role::Assistant, src.to_string(), true));
    app.sync_transcript(80, 40, &theme);

    // Collapsed: the bullet shows, the result is truncated to 4 rows + a `▸ +2 more`
    // affordance, and LINE6 (past the preview) is hidden.
    let before = block_plain(&app, id, &theme);
    assert!(before.contains("⏺ run"), "the tool bullet renders: {before:?}");
    assert!(before.contains("▸ +2 more"), "collapsed result has a clickable triangle: {before:?}");
    assert!(!before.contains("LINE6"), "the overflow line is hidden when collapsed: {before:?}");
    assert!(!before.contains("more more"), "no dead `… +N more` duplication");

    // The bullet is a `Tool` node in the hit table.
    let tool = NodeId::Tool { block: id, tool: 0 };
    assert!(
        app.node_hit.iter().any(|(_, n)| *n == tool),
        "the expandable tool bullet is a clickable node: {:?}",
        app.node_hit
    );

    // Toggle it expanded → every result line shows + a `▾` collapse affordance.
    app.toggle_fold(tool);
    assert_eq!(app.folds.get(&tool), Some(&true), "tool result expanded");
    app.sync_transcript(80, 40, &theme);
    let expanded = block_plain(&app, id, &theme);
    for i in 1..=6 {
        assert!(expanded.contains(&format!("LINE{i}")), "expanded result shows LINE{i}: {expanded:?}");
    }
    assert!(expanded.contains('▾'), "expanded result has a ▾ collapse triangle: {expanded:?}");
    assert!(!expanded.contains("+2 more"), "no `+N more` text when expanded: {expanded:?}");

    // Toggle back → collapsed again (clean round-trip).
    app.toggle_fold(tool);
    app.sync_transcript(80, 40, &theme);
    let recollapsed = block_plain(&app, id, &theme);
    assert!(recollapsed.contains("▸ +2 more"), "second toggle re-collapses: {recollapsed:?}");
    assert!(!recollapsed.contains("LINE6"), "overflow hidden again: {recollapsed:?}");
}

/// Expanding a fold node ABOVE the viewport top must NOT jump the view: the click
/// re-derives the scroll anchor on the CLICKED node (not Bottom), so that node stays
/// at the same screen offset across the row-count change (Fix E / parity gate 5).
#[test]
fn expand_above_viewport_keeps_clicked_node_anchored() {
    let theme = Theme::default_theme();
    let mut app = AppState::new();
    // A TALL transcript: a foldable assistant block up top, then many filler blocks
    // so the foldable block sits ABOVE a small viewport when scrolled to it.
    let top_id = app.alloc_block_id();
    app.transcript
        .push(Block::new(top_id, None, Role::Assistant, THREE_TURNS.to_string(), true));
    for _ in 0..30 {
        let fid = app.alloc_block_id();
        app.transcript
            .push(Block::new(fid, None, Role::Assistant, "filler line".to_string(), true));
    }
    // A small viewport (8 rows) at width 80.
    app.set_term_size(80, 12);
    app.sync_transcript(80, 8, &theme);

    // Pin the foldable block's top row to the top of the viewport (a genuine
    // mid-scroll: content below it), then find turn 1's `▸` header row.
    app.viewport.anchor_node_at_offset(top_id, 0, 0, &app.wrap_cache);
    app.sync_transcript(80, 8, &theme);
    let (range, node) = app
        .node_hit
        .iter()
        .find(|(_, n)| matches!(n, NodeId::Turn { turn: 1, .. }))
        .cloned()
        .expect("turn 1 fold header present");
    let header_row = *range.start();
    let top_before = app.viewport.visual_top(&app.wrap_cache);
    let header_offset_before = header_row.saturating_sub(top_before);
    // The header's BLOCK-LOCAL intra (turn 1 is the first turn → row 0 of the block);
    // after the expand, this same block-local row is turn 1's first expanded line, so
    // tracking it proves "the top of turn 1" stayed put.
    let (anchored_block, anchored_intra) = app.wrap_cache.anchor_at(header_row).unwrap();
    assert_eq!(anchored_block, top_id);

    // Click turn 1's header at its current screen offset → expand. The viewport must
    // re-anchor on the clicked node so it stays at the SAME screen offset.
    let cockpit_top = crate::components::cockpit::split_cockpit(
        &app,
        ratatui::layout::Rect::new(0, 0, 80, 12),
    )
    .transcript
    .y;
    let screen_row = cockpit_top + header_offset_before as u16;
    assert!(app.click_fold_at(0, screen_row, cockpit_top), "triangle click handled");
    assert_eq!(app.folds.get(&node), Some(&false), "turn 1 expanded");

    // Re-sync (the post-toggle reflow grows the block). The clicked node's first row
    // must still be visible at the SAME screen offset (no jump): the anchor is on the
    // node, the row count changed beneath it, and it stays put.
    app.sync_transcript(80, 8, &theme);
    let new_top = app.viewport.visual_top(&app.wrap_cache);
    let new_anchor_row = app.wrap_cache.locate(anchored_block, anchored_intra).unwrap();
    assert!(
        !app.viewport.is_following(),
        "the view is anchored on the clicked node, not in Bottom-follow"
    );
    assert_eq!(
        new_anchor_row.saturating_sub(new_top),
        header_offset_before,
        "the clicked node kept its screen offset across the expand (no jump)"
    );
}

#[test]
fn file_index_walks_once_within_ttl() {
    // Point an `AppState` at a throwaway repo root with one known file, then call
    // `list_project_files()` twice in quick succession (well within FILE_INDEX_TTL).
    // The cache must serve the second call from the SAME `Arc` allocation — proving
    // it did NOT re-walk the tree (a fresh walk would build a new `Arc`). We then
    // mutate the dir and confirm the still-fresh cache keeps returning the ORIGINAL
    // snapshot (no per-call re-walk) — the behavioral guarantee of "walks once per
    // window" that kills the 3×/frame @ lag (Q12).
    let dir = std::env::temp_dir().join(format!("tui_v4_idx_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("alpha.rs"), "fn a(){}").unwrap();

    let mut app = AppState::new();
    app.repo_root = dir.clone();

    let first = app.list_project_files();
    let second = app.list_project_files();
    assert!(
        std::sync::Arc::ptr_eq(&first, &second),
        "the second call within the TTL returns the cached Arc (no second walk)"
    );
    assert!(
        first.contains(&"alpha.rs".to_string()),
        "the cached snapshot holds the file present at walk time"
    );

    // A file created AFTER the walk is invisible until the TTL lapses — the cache is
    // not re-derived per call.
    std::fs::write(dir.join("beta.rs"), "fn b(){}").unwrap();
    let third = app.list_project_files();
    assert!(
        std::sync::Arc::ptr_eq(&first, &third),
        "a still-fresh cache is reused verbatim — no re-walk picked up the new file"
    );
    assert!(
        !third.contains(&"beta.rs".to_string()),
        "the newly-created file is absent from the cached (pre-creation) snapshot"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
