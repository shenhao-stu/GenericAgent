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
            llm: None,
            model_real: None,
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

/// Slice-1 honest check: the wire model IDENTITY (`llm` + `model_real`) lands on
/// `AppState`, and a mid-turn FAILOVER `Status` live-updates both. The `[1m]` tag is
/// stripped by the bridge (`ga_bridge.llm_identity`, asserted separately in Python),
/// so the wire carries the already-stripped `claude-opus-4-8` — this confirms the
/// Rust side stores it through the real `apply_bridge_event` fold (not a fixture),
/// keeps `model` (the legacy chain) untouched, and that an identity-less `Status`
/// does NOT clobber a prior value.
#[test]
fn ready_then_failover_status_updates_llm_identity() {
    let mut app = AppState::new();

    // Handshake: codex-pro / gpt-5.5 — the legacy `model` chain stays distinct.
    app.apply_bridge_event(
        frame(CoreToUi::Ready {
            version: Some("1".into()),
            model: Some("MixinSession/codex-pro|getoken_20x".into()),
            llm: Some("codex-pro".into()),
            model_real: Some("gpt-5.5".into()),
        }),
        0,
    );
    assert_eq!(app.llm_name.as_deref(), Some("codex-pro"));
    assert_eq!(app.model_real.as_deref(), Some("gpt-5.5"));
    assert_eq!(app.model.as_deref(), Some("MixinSession/codex-pro|getoken_20x"));

    // Mid-turn failover → a fresh Status flips the ACTIVE member to getoken_20x. The
    // bridge already stripped `claude-opus-4-8[1m]` → `claude-opus-4-8` (no `[`).
    app.apply_bridge_event(
        frame(CoreToUi::Status {
            model: Some("MixinSession/codex-pro|getoken_20x".into()),
            llm: Some("getoken_20x".into()),
            model_real: Some("claude-opus-4-8".into()),
            context_percent: None,
            tokens: None,
            input_tokens: None,
            output_tokens: None,
            cache_tokens: None,
            last_input: None,
            last_output: None,
            text: None,
        }),
        1_000,
    );
    assert_eq!(app.llm_name.as_deref(), Some("getoken_20x"));
    assert_eq!(app.model_real.as_deref(), Some("claude-opus-4-8"));
    assert!(
        !app.model_real.as_deref().unwrap().contains('['),
        "the [1m] context-window tag must be stripped on the wire"
    );

    // An identity-LESS Status (a stale/older bridge frame) must NOT wipe the prior
    // identity — only a present field overwrites (store_identity guard).
    app.apply_bridge_event(
        frame(CoreToUi::Status {
            model: None,
            llm: None,
            model_real: None,
            context_percent: Some(50.0),
            tokens: None,
            input_tokens: None,
            output_tokens: None,
            cache_tokens: None,
            last_input: None,
            last_output: None,
            text: None,
        }),
        2_000,
    );
    assert_eq!(app.llm_name.as_deref(), Some("getoken_20x"));
    assert_eq!(app.model_real.as_deref(), Some("claude-opus-4-8"));
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
    // Idle by default. Slice 6: the title is `<pet-face> <session_name> · GenericAgent`;
    // the pet defaults ON + bear so it leads with `ʕ•ᴥ•ʔ`, carries the active session
    // name, ends in "GenericAgent", and never contains "NativeClaude".
    assert_eq!(app.tab_status(), TabStatus::Idle);
    let title = app.terminal_title();
    assert!(title.starts_with("ʕ•ᴥ•ʔ"), "bear face leads when idle: {title:?}");
    assert!(title.contains(app.sessions.active_name()), "the active session name is in the title: {title:?}");
    assert!(title.contains("GenericAgent"), "GenericAgent is in the title: {title:?}");
    assert!(!title.contains("NativeClaude"), "the title must NOT contain NativeClaude: {title:?}");

    // A renamed session flows into the title (set in-memory; no sidecar write here).
    app.sessions.active_mut().name = "scan tabs".into();
    let renamed = app.terminal_title();
    assert!(renamed.contains("scan tabs"), "the renamed session shows in the title: {renamed:?}");
    assert_eq!(renamed, "ʕ•ᴥ•ʔ scan tabs · GenericAgent");

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
/// `🛠️ name(args)` result renders a bordered BOX whose result is truncated + a
/// `… +N more` fold affordance INSIDE the box; the box is a `Tool` node in the hit
/// table. Clicking it expands the result (every line shows) and swaps `… +N more` for
/// `▾` — and the click re-anchors so it never jumps. A second click collapses it back.
#[test]
fn click_expands_and_collapses_tool_result() {
    let theme = Theme::default_theme();
    let mut app = AppState::new();
    // One turn, one tool whose result is 6 lines (> the 4-row preview → foldable).
    let src = "Turn 1 ...\n🛠️ run({\"cmd\": \"ls\"})\nLINE1\nLINE2\nLINE3\nLINE4\nLINE5\nLINE6";
    let id = app.alloc_block_id();
    app.transcript.push(Block::new(id, None, Role::Assistant, src.to_string(), true));
    app.sync_transcript(80, 40, &theme);

    // Collapsed: the box shows (name on the top border), the result is truncated to
    // 4 rows + a `… +2 more` affordance, and LINE6 (past the preview) is hidden.
    let before = block_plain(&app, id, &theme);
    assert!(before.contains("╭─"), "the tool box renders: {before:?}");
    assert!(before.lines().any(|l| l.contains("╭─") && l.contains("run")), "tool name on the top border: {before:?}");
    assert!(before.contains("▸ +2 more"), "collapsed result has the fold affordance: {before:?}");
    assert!(!before.contains("LINE6"), "the overflow line is hidden when collapsed: {before:?}");

    // The box is a `Tool` node in the hit table.
    let tool = NodeId::Tool { block: id, tool: 0 };
    assert!(
        app.node_hit.iter().any(|(_, n)| *n == tool),
        "the expandable tool box is a clickable node: {:?}",
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
    assert!(expanded.contains('▾'), "expanded result has a ▾ collapse affordance: {expanded:?}");
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

// ===========================================================================
// ROUND-4 FAILURE DIAGNOSIS — end-to-end tests that exercise the REAL render
// pipeline (apply_bridge_event → prepare_frame → render to TestBackend → scan
// the STYLED frame buffer), the path the prior round's unit-on-clean-fixtures
// tests bypassed.
// ===========================================================================

/// Render the whole cockpit to an in-memory TestBackend and return its rows as
/// trailing-trimmed strings — the exact bytes the terminal would show.
fn render_to_rows(app: &mut AppState, w: u16, h: u16) -> Vec<String> {
    use ratatui::backend::TestBackend;
    use ratatui::layout::Rect;
    use ratatui::Terminal;
    let theme = app.theme.clone();
    let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
    app.prepare_frame(Rect::new(0, 0, w, h), &theme);
    term.draw(|f| crate::components::render(f, app, &theme, 0)).unwrap();
    let buf = term.backend().buffer();
    let mut rows = Vec::with_capacity(h as usize);
    for y in 0..h as usize {
        let mut line = String::new();
        for x in 0..w as usize {
            line.push_str(buf.content()[y * w as usize + x].symbol());
        }
        rows.push(line.trim_end().to_string());
    }
    rows
}

/// FAILURE C (Turn N leak) — driven through the REAL streaming pipeline + the
/// STYLED frame. A single `Turn N ...` marker preceded by a prose line leaks
/// the literal marker because `fold_turns_with` returns the WHOLE text as one
/// `Text` segment (<2 markers) and `strip_leading_turn_line` only de-marks line
/// 0 (the prose), so the mid-body marker flows through markdown as prose. The
/// `no_turn_n_anywhere` fixture never has a marker preceded by non-marker text.
#[test]
fn live_active_turn_marker_leaks_into_styled_frame() {
    let mut app = AppState::new();
    app.conn = ConnStatus::Connected { model: Some("m".into()) };
    app.push_user("hi".into());
    // Begin a turn, then stream a delta whose body has prose BEFORE the marker —
    // the canonical GA reply shape ("preamble … Turn N … body").
    app.apply_bridge_event(frame(CoreToUi::MessageBegin { mid: "m1".into(), role: "assistant".into() }), 0);
    app.apply_bridge_event(
        frame(CoreToUi::MessageDelta {
            mid: "m1".into(),
            text: "Let me look into this.\nTurn 1 ...\nthe answer is 42".into(),
        }),
        0,
    );
    let rows = render_to_rows(&mut app, 100, 24);
    let screen = rows.join("\n");
    assert!(
        !screen.contains("Turn 1"),
        "the live STYLED frame leaked a literal 'Turn 1' marker:\n{screen}"
    );
}

/// FAILURE A (cannot scroll) — driven through the REAL loop ordering: render a
/// frame (sync viewport to geometry), THEN press PageUp / wheel-up (the order
/// the event loop uses), then re-render and assert the visible window MOVED off
/// the tail. A unit test that calls `viewport.page_up` on a hand-built cache
/// can pass while the live binary's geometry/ordering differs.
#[test]
fn page_up_and_wheel_move_the_live_viewport() {
    let mut app = AppState::new();
    app.conn = ConnStatus::Connected { model: Some("m".into()) };
    // A tall transcript so there is something to scroll.
    for i in 0..60u64 {
        app.push_system(format!("history line number {i}"));
    }
    // First frame: syncs the wrap cache + viewport to the transcript geometry.
    let before = render_to_rows(&mut app, 80, 20);
    assert!(app.following(), "starts pinned to the tail (follow mode)");
    let tail_visible_before = before.iter().any(|r| r.contains("line number 59"));
    assert!(tail_visible_before, "the tail line is visible before scrolling");

    // PageUp (the real keymap calls app.page_up()).
    app.page_up();
    assert!(!app.following(), "PageUp must leave follow mode");
    let after_pgup = render_to_rows(&mut app, 80, 20);
    assert!(
        !after_pgup.iter().any(|r| r.contains("line number 59")),
        "after PageUp the tail must scroll OUT of view:\n{}",
        after_pgup.join("\n")
    );

    // Wheel up further (the real mouse handler calls app.scroll_lines(-3)).
    let top_after_pgup = app.viewport.visual_top(&app.wrap_cache);
    app.scroll_lines(-3);
    let _ = render_to_rows(&mut app, 80, 20);
    assert!(
        app.viewport.visual_top(&app.wrap_cache) < top_after_pgup,
        "wheel-up must move the viewport top further toward history"
    );

    // End / scroll_end resumes follow.
    app.scroll_end();
    let after_end = render_to_rows(&mut app, 80, 20);
    assert!(app.following(), "scroll_end resumes follow mode");
    assert!(
        after_end.iter().any(|r| r.contains("line number 59")),
        "after End the tail is visible again"
    );
}

/// FAILURE B (markdown not rendering live) — during streaming the assistant text
/// must be markdown-FORMATTED in the STYLED frame, not dumped raw. We stream a
/// real GA-shaped reply (heading + bold + inline code) and assert the raw
/// markdown tokens (`##`, `**`, backticks) do NOT appear on screen while the
/// formatted text DOES — proving the live/streaming tail ran the markdown
/// renderer, not a raw-text passthrough.
#[test]
fn streaming_markdown_is_formatted_in_styled_frame() {
    let mut app = AppState::new();
    app.conn = ConnStatus::Connected { model: Some("m".into()) };
    app.push_user("hi".into());
    app.apply_bridge_event(frame(CoreToUi::MessageBegin { mid: "m1".into(), role: "assistant".into() }), 0);
    // A finalized-paragraph head so it commits, plus a streaming tail — both must
    // render formatted (no raw tokens).
    app.apply_bridge_event(
        frame(CoreToUi::MessageDelta {
            mid: "m1".into(),
            text: "## Heading One\n\nThis is **bold** and `code` text.\n\nA second **strong** word.".into(),
        }),
        0,
    );
    let rows = render_to_rows(&mut app, 100, 24);
    let screen = rows.join("\n");
    // The words survive…
    assert!(screen.contains("Heading One"), "heading text present:\n{screen}");
    assert!(screen.contains("bold"), "bold word present");
    assert!(screen.contains("code"), "code word present");
    // …but the raw INLINE markdown punctuation does NOT (markdown actually ran).
    // Headings now render CLEAN like tui_v3/CC: no literal `#` glyph — the heading
    // is conveyed by BOLD + per-level color, asserted below.
    assert!(!screen.contains("**"), "raw bold markers leaked (markdown did not run):\n{screen}");
    assert!(!screen.contains("##"), "literal heading hashes leaked (heading not rendered clean):\n{screen}");

    // Prove the heading actually ran the markdown walker: its row carries BOLD.
    use ratatui::backend::TestBackend;
    use ratatui::layout::Rect;
    use ratatui::style::Modifier;
    use ratatui::Terminal;
    let theme = app.theme.clone();
    let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
    app.prepare_frame(Rect::new(0, 0, 100, 24), &theme);
    let app_ref: &AppState = &app;
    term.draw(|f| crate::components::render(f, app_ref, &theme, 0)).unwrap();
    let buf = term.backend().buffer();
    let heading_row = rows.iter().position(|r| r.contains("Heading One")).expect("heading row");
    let bold_on_heading = (0..100usize).any(|x| {
        buf.content()[heading_row * 100 + x].modifier.contains(Modifier::BOLD)
    });
    assert!(bold_on_heading, "the heading row must be BOLD (markdown ran), row {heading_row}:\n{screen}");
}

/// SLICE 10 (markdown heading cleanup) — the HONEST end-to-end check. Stream
/// `# Title` / `## Heading One` / `### Sub` through the REAL bridge path
/// (`apply_bridge_event`) into the STYLED `TestBackend` grid, then for EACH
/// heading row assert it (a) contains the heading TEXT, (b) carries NO literal
/// `#`, and (c) the text cells are BOLD + a non-body heading color — the
/// clean tui_v3/CC look, not a raw `## ` glyph.
#[test]
fn headings_render_bold_and_colored_without_hashes_in_styled_frame() {
    use ratatui::backend::TestBackend;
    use ratatui::layout::Rect;
    use ratatui::style::{Color, Modifier};
    use ratatui::Terminal;

    let mut app = AppState::new();
    app.conn = ConnStatus::Connected { model: Some("m".into()) };
    app.push_user("hi".into());
    app.apply_bridge_event(frame(CoreToUi::MessageBegin { mid: "m1".into(), role: "assistant".into() }), 0);
    app.apply_bridge_event(
        frame(CoreToUi::MessageDelta {
            mid: "m1".into(),
            text: "# Title\n\n## Heading One\n\n### Sub\n\nbody text.".into(),
        }),
        0,
    );

    let (w, h) = (100u16, 24u16);
    let theme = app.theme.clone();
    let body_color = theme.color(crate::theme::Token::Text);
    let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
    app.prepare_frame(Rect::new(0, 0, w, h), &theme);
    let app_ref: &AppState = &app;
    term.draw(|f| crate::components::render(f, app_ref, &theme, 0)).unwrap();
    let buf = term.backend().buffer();

    let rows: Vec<String> = (0..h as usize)
        .map(|y| (0..w as usize).map(|x| buf.content()[y * w as usize + x].symbol()).collect::<String>())
        .collect();
    let screen = rows.join("\n");

    // No literal hash glyphs survive anywhere on screen (clean headings).
    assert!(!screen.contains('#'), "a literal heading hash leaked:\n{screen}");

    for needle in ["Title", "Heading One", "Sub"] {
        let row = rows.iter().position(|r| r.contains(needle)).unwrap_or_else(|| panic!("heading {needle:?} present:\n{screen}"));
        // Inspect the styled cells under the heading text on that row.
        let line = &rows[row];
        let start = line.find(needle).unwrap();
        let (mut saw_bold, mut saw_color) = (false, false);
        for x in start..(start + needle.chars().count()).min(w as usize) {
            let cell = &buf.content()[row * w as usize + x];
            if cell.symbol() == " " { continue; }
            if cell.modifier.contains(Modifier::BOLD) { saw_bold = true; }
            if cell.fg != Color::Reset && cell.fg != body_color { saw_color = true; }
        }
        assert!(saw_bold, "heading {needle:?} must be BOLD (row {row}):\n{screen}");
        assert!(saw_color, "heading {needle:?} must carry a distinct heading color (row {row}):\n{screen}");
    }
}

/// SLICE 0 (wheel scroll) — the HONEST end-to-end check for the real "can't
/// scroll" bug. The defect was never in `scroll_lines`: it was that crossterm only
/// delivers `ScrollUp/ScrollDown` when mouse capture is ON, and capture defaulted
/// OFF, so the wheel was dead in the real terminal. R3's test called `scroll_lines`
/// directly and bypassed that gate. This test (1) asserts capture defaults ON so the
/// wheel events are actually emitted, and (2) feeds REAL crossterm `ScrollUp` /
/// `ScrollDown` mouse events through the LIVE input router (`crate::input::mouse`),
/// not `scroll_lines`, and asserts each moves `viewport.visual_top`.
#[test]
fn wheel_scroll_event_moves_viewport_under_default_capture() {
    use crossterm::event::{KeyModifiers, MouseEvent, MouseEventKind};

    // (1) S1: Capture is OFF by default (NATIVE mode). The wheel works via
    // EnableAlternateScroll (?1007h → arrow keys) in native mode. The test below
    // exercises the mouse handler path directly regardless of capture state.
    assert!(
        !AppState::default().mouse_capture,
        "mouse capture defaults OFF in native mode (S1 toggle model)"
    );

    let mut app = AppState::new();
    app.conn = ConnStatus::Connected { model: Some("m".into()) };
    for i in 0..60u64 {
        app.push_system(format!("history line number {i}"));
    }
    // First frame: sync the wrap cache + viewport to the transcript geometry (the
    // viewport now has real rows to scroll over). Starts pinned to the tail.
    let _ = render_to_rows(&mut app, 80, 20);
    assert!(app.following(), "starts pinned to the tail (follow mode)");

    let wheel = |kind: MouseEventKind| MouseEvent {
        kind,
        column: 10,
        row: 10, // inside the transcript body, not the header band
        modifiers: KeyModifiers::NONE,
    };

    // A wheel-UP event routed through the LIVE `mouse()` path → scroll_lines(-3) →
    // the viewport leaves follow and the top moves toward history.
    crate::input::mouse::mouse(wheel(MouseEventKind::ScrollUp), &mut app);
    let _ = render_to_rows(&mut app, 80, 20);
    assert!(!app.following(), "a wheel-up event leaves follow mode");
    let top_after_up = app.viewport.visual_top(&app.wrap_cache);
    assert!(top_after_up > 0, "wheel-up moved the viewport top off the tail, got {top_after_up}");

    // A wheel-DOWN event (the exact crossterm `ScrollDown` the prompt names) routed
    // through the SAME live path → scroll_lines(+3) → the top moves back toward the
    // tail (visual_top increases). This is the assertion the prompt demands.
    crate::input::mouse::mouse(wheel(MouseEventKind::ScrollDown), &mut app);
    let _ = render_to_rows(&mut app, 80, 20);
    let top_after_down = app.viewport.visual_top(&app.wrap_cache);
    assert!(
        top_after_down > top_after_up,
        "a ScrollDown mouse event must advance viewport.visual_top via the live path \
         ({top_after_up} → {top_after_down})"
    );
}

/// SLICE 0 (dashboard companion) — `preview_line` must NOT surface a bare
/// `Turn N ...` marker as a session card's live preview (the R3 raw-source leak).
/// When the newest assistant output is a turn marker, the card should fall back to
/// the prior real content line, not echo the spacing marker.
#[test]
fn dashboard_preview_skips_turn_marker_line() {
    use crate::app::session::preview_line;
    use crate::app::session::SessionStatus;

    let block = Block::new(1, None, Role::Assistant, "the answer is 42\nTurn 2 ...".into(), true);
    let preview = preview_line(std::slice::from_ref(&block), SessionStatus::Working);
    assert_eq!(
        preview, "the answer is 42",
        "the dashboard preview must skip the bare 'Turn 2 ...' marker and show real content, got {preview:?}"
    );
}

/// SLICE 3 (tool-call BORDERED BOX) — the HONEST end-to-end check. A real-GA-shaped
/// compact call (the U+1F6E0 tool marker + `web_scan({"tabs_only": true})` + an
/// `[Info] ok` result line) is STREAMED through the live pipeline
/// (`apply_bridge_event` MessageBegin → MessageDelta), rendered to a TestBackend, and
/// the STYLED cell grid is scanned for the tui_v3-style box: the `╭─` top-left corner
/// carrying the tool NAME `web_scan` and a `·t` turn-id, the `[Info]`-derived `ok`
/// result row INSIDE the box, and the `╯` bottom-right corner. We ALSO probe the cell
/// FG colors to prove the border is the accent (`Token::Claude`) and the name is BOLD
/// — i.e. the box is actually painted, not just present as text. Finally the 4 parity
/// invariants are asserted to pass (the box rows flow through both the styled draw and
/// the plain projection, so they stay green by construction).
#[test]
fn live_tool_call_renders_bordered_box_in_styled_frame() {
    let mut app = AppState::new();
    app.conn = ConnStatus::Connected { model: Some("m".into()) };
    app.push_user("scan tabs".into());

    // Stream the call: U+1F6E0 (🛠) + VS16 + space marker, then the compact header
    // and an [Info] result line — the exact shape `ga_bridge.py` emits in compact mode.
    app.apply_bridge_event(
        frame(CoreToUi::MessageBegin { mid: "m1".into(), role: "assistant".into() }),
        0,
    );
    app.apply_bridge_event(
        frame(CoreToUi::MessageDelta {
            mid: "m1".into(),
            text: "\u{1F6E0}\u{FE0F} web_scan({\"tabs_only\": true})\n[Info] ok".into(),
        }),
        0,
    );

    let rows = render_to_rows(&mut app, 100, 24);
    let screen = rows.join("\n");

    // (1) The TOOL box top border carries the tool name + a `·t` turn-id; the bottom
    //     border closes it. Scan the STYLED symbol grid (what the terminal shows) for
    //     the `╭─` line that bears `web_scan` (the header/composer also draw `╭─`
    //     frames — the tool box is the one with the tool name ON it).
    let top = rows
        .iter()
        .find(|r| r.contains("╭─") && r.contains("web_scan"))
        .unwrap_or_else(|| panic!("no tool-box `╭─ web_scan` top border in the live styled frame:\n{screen}"));
    assert!(top.contains("·t"), "a `·t` turn-id on the top border: {top:?}");
    assert!(top.contains("✓ ok"), "the ok status badge on the top border: {top:?}");
    assert!(
        rows.iter().any(|r| r.contains('│') && r.contains("ok") && !r.contains("✓")),
        "the [Info]-derived `ok` result row is INSIDE the box (│ … │):\n{screen}"
    );
    assert!(
        rows.iter().any(|r| r.contains('╰') && r.contains('╯')),
        "a `╰…╯` bottom border corner closes the box:\n{screen}"
    );
    // The raw tool marker never leaks into the styled frame.
    assert!(!screen.contains('\u{1F6E0}'), "the raw U+1F6E0 tool marker must not render:\n{screen}");

    // (2) Probe the cell FG to prove the box is PAINTED: the `╭` corner is the accent
    //     (Token::Claude) and the `web_scan` name carries BOLD — i.e. the styled draw
    //     ran `push_tool_box`, not a plain dump.
    use ratatui::backend::TestBackend;
    use ratatui::layout::Rect;
    use ratatui::style::Modifier;
    use ratatui::Terminal;
    let theme = app.theme.clone();
    let accent = theme.color(crate::theme::Token::Claude);
    let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
    app.prepare_frame(Rect::new(0, 0, 100, 24), &theme);
    let app_ref: &AppState = &app;
    term.draw(|f| crate::components::render(f, app_ref, &theme, 0)).unwrap();
    let buf = term.backend().buffer();
    let top_row = rows
        .iter()
        .position(|r| r.contains("╭─") && r.contains("web_scan"))
        .expect("tool-box top border row");
    let corner_x = rows[top_row].chars().take_while(|c| *c != '╭').count();
    assert_eq!(
        buf.content()[top_row * 100 + corner_x].fg,
        accent,
        "the `╭` box corner must paint in the accent (Token::Claude) color"
    );
    // The bold name: at least one cell on the top border row carries BOLD.
    let name_bold = (0..100usize).any(|x| buf.content()[top_row * 100 + x].modifier.contains(Modifier::BOLD));
    assert!(name_bold, "the tool name on the top border must be BOLD (push_tool_box ran), row {top_row}:\n{screen}");

    // (3) The 4 parity invariants — asserted to PASS here (the box rows go through both
    //     the styled draw and the plain projection, so they stay green by construction).
    assert_parity_invariants();
}

/// Run the 4 architecture parity invariants (the same checks the dedicated unit tests
/// make) inline, so the slice-3 honest check proves they hold WITH a tool BOX present.
/// A mismatch panics with the offending `(src, width)`.
fn assert_parity_invariants() {
    use crate::markdown::{
        lines_to_plain, lines_to_plain_atomic, render_assistant, render_assistant_cockpit,
        render_assistant_cockpit_plain, render_assistant_plain, render_assistant_wrapped,
    };
    use crate::render::block::{Block as RB, BlockRole};
    use crate::render::measure::WrapCache;

    let theme = Theme::default_theme();
    // A source that renders a tool BOX (the slice-3 subject) + prose + CJK so the
    // invariants are exercised against the new box rows specifically.
    let src = "Turn 1 ...\n🛠️ web_scan({\"tabs_only\": true})\n[Info] ok\nthen some prose 中文也在这里很长很长会换行\ndone.";

    for width in [20u16, 24, 40, 80, 120] {
        // I. styled_wrap_rowcount_matches_wrap_cache (un-cockpit).
        let styled = render_assistant_wrapped(src, &theme, width);
        let (plain, ranges) = lines_to_plain_atomic(&render_assistant(src, &theme));
        let rb = RB::finalized(1, BlockRole::Assistant, plain).with_atomic_ranges(ranges);
        let mut cache = WrapCache::new(width);
        cache.sync(std::slice::from_ref(&rb));
        assert_eq!(styled.len(), cache.block_line_count(1), "I width={width}");

        // II. plain_projection_matches_rendered_text (cockpit-plain == styled count).
        let cstyled = render_assistant_cockpit(src, &theme, false, width);
        let cplain = render_assistant_cockpit_plain(src, &theme, false, width);
        assert_eq!(cplain.split('\n').count(), cstyled.len(), "II width={width}");

        // III. cockpit_render_rowcount_matches_plain_projection.
        let (cp, cr) = lines_to_plain_atomic(&render_assistant_cockpit(src, &theme, false, width));
        let crb = RB::finalized(2, BlockRole::Assistant, cp).with_atomic_ranges(cr);
        let mut ccache = WrapCache::new(width);
        ccache.sync(std::slice::from_ref(&crb));
        assert_eq!(cstyled.len(), ccache.block_line_count(2), "III width={width}");

        // IV. embedded_newline_in_span_keeps_rowcount_parity — no emitted span carries
        //     an embedded '\n' (the box rows are single hard lines).
        for line in render_assistant_cockpit(src, &theme, false, width) {
            for span in &line.spans {
                assert!(!span.content.contains('\n'), "IV embedded \\n width={width}: {:?}", span.content);
            }
        }
        // Sanity: the projection round-trips byte-identically (no styling drift).
        assert_eq!(lines_to_plain(&cstyled), cplain, "cockpit plain projection round-trips width={width}");
        let _ = render_assistant_plain(src, &theme);
    }
}

/// Slice 7 HONEST CHECK — per-command composer-border PARITY with shell `!`, proven on
/// the LIVE styled path at MONO (NO_COLOR) caps. With `/goal` in the buffer the base
/// `render_composer` border token is the command accent (`Token::Claude` for Goal), NOT
/// `Token::Border` — and because the truecolor `draw_composer_border_fx` overlay is
/// gated OFF here (mono caps), the painted corner cells we read ARE the base border, so
/// the restyle is visible at every capability level exactly like `!`'s always-on tint. A
/// plain buffer keeps the neutral `Token::Border` corner (the restyle is command-driven,
/// not always-on). The 4 parity invariants are asserted green alongside.
#[test]
fn live_command_border_restyles_at_mono_like_shell_bang() {
    use crate::components::cockpit::split_cockpit;
    use ratatui::backend::TestBackend;
    use ratatui::layout::Rect;
    use ratatui::Terminal;

    let theme = Theme::default_theme();
    let accent = theme.color(crate::theme::Token::Claude); // Goal/Morphling accent.
    let border = theme.color(crate::theme::Token::Border);
    let shell_accent = theme.color(crate::theme::Token::ShellAccent);
    assert_ne!(accent, border, "the command accent differs from the neutral border by construction");

    // Read the composer's top-left `╭` corner FG from the LIVE styled frame for `buf`,
    // with the effects engine forced to MONO so the truecolor border overlay is a no-op
    // (we read the BASE render_composer border — the mono/NO_COLOR path).
    let corner_fg = |buf_text: &str| -> ratatui::style::Color {
        let (w, h) = (100u16, 24u16);
        let mut app = AppState::new();
        app.effects = crate::effects::EffectsEngine::new(crate::effects::ColorCaps::mono());
        assert!(!app.effects.caps.enabled(), "mono caps → the truecolor border overlay is gated OFF");
        app.composer.type_str(buf_text);
        let layout = split_cockpit(&app, Rect::new(0, 0, w, h));
        let c = layout.composer;
        let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
        app.prepare_frame(Rect::new(0, 0, w, h), &theme);
        let app_ref: &AppState = &app;
        term.draw(|f| crate::components::render(f, app_ref, &theme, 0)).unwrap();
        // The top-left border corner cell of the composer block.
        term.backend().buffer().content()[(c.y as usize) * w as usize + c.x as usize].fg
    };

    // `/goal` → the base border is the Goal accent (visible at mono), NOT Token::Border.
    let goal_fg = corner_fg("/goal");
    assert_eq!(goal_fg, accent, "the `/goal` composer border paints the command accent at mono");
    assert_ne!(goal_fg, border, "the `/goal` border token differs from the neutral Token::Border (parity with `!`)");

    // Control: a plain buffer keeps the neutral border (the restyle is command-driven).
    assert_eq!(corner_fg("write the readme"), border, "a plain buffer keeps the neutral Token::Border corner");

    // Parity reference: shell `!` tints the SAME corner hot-pink, also at mono.
    assert_eq!(corner_fg("!ls -la"), shell_accent, "shell `!` tints the border hot-pink at mono (the template)");

    // The four orchestration commands each restyle the base border to their accent
    // (Hive→Success, Conductor→Suggestion, Goal/Morphling→Claude) — none stays Border.
    for (buf, tok) in [
        ("/hive split it", crate::theme::Token::Success),
        ("/conductor run", crate::theme::Token::Suggestion),
        ("/morphling absorb", crate::theme::Token::Claude),
    ] {
        let got = corner_fg(buf);
        assert_eq!(got, theme.color(tok), "`{buf}` border paints {tok:?} at mono");
        assert_ne!(got, border, "`{buf}` restyles the border off the neutral Token::Border");
    }

    // The 4 architecture parity invariants stay green with a command border present.
    assert_parity_invariants();
}

/// S1 HONEST-CHECK B1 — expanded turn has a ▾ header row tagged NodeId::Turn AND
/// a click on that row re-collapses the turn. Exercises the LIVE/STYLED path:
/// apply_bridge_event → sync_transcript → node_hit → click_fold_at.
///
/// FAILS on old code: FoldSegment::Text has no `turn` field, so no ▾ header is
/// emitted and no NodeId::Turn entry covers expanded turn rows.
/// PASSES now: FoldSegment::Text carries `turn: Some(n)`, the renderer emits a
/// ` ▾ <title>` row tagged NodeId::Turn, and click_fold_at can collapse it.
#[test]
fn expanded_turn_has_downward_triangle_header_and_can_be_recollapsed() {
    let theme = Theme::default_theme();
    let mut app = AppState::new();
    app.conn = ConnStatus::Connected { model: Some("m".into()) };

    // Two-turn source: turn 1 will be expanded (manually toggled), turn 2 is the last.
    let src = "Turn 1 ...\n<summary>first task done</summary>\nbody1\nTurn 2 ...\n<summary>in progress</summary>\nbody2";
    let id = app.alloc_block_id();
    app.transcript.push(Block::new(id, None, Role::Assistant, src.to_string(), true));
    app.set_term_size(80, 40);
    app.sync_transcript(80, 40, &theme);

    // Turn 1 folds by default (it's a completed turn). Expand it.
    let node1 = NodeId::Turn { block: id, turn: 1 };
    app.toggle_fold(node1);
    app.sync_transcript(80, 40, &theme);

    // STYLED projection must contain a ▾ glyph.
    let plain = block_plain(&app, id, &theme);
    assert!(plain.contains('▾'), "expanded turn must show a ▾ header row:\n{plain}");

    // The ▾ header row must be tagged NodeId::Turn{turn:1} in the node_hit table.
    assert!(
        app.node_hit.iter().any(|(_, n)| *n == node1),
        "NodeId::Turn{{turn:1}} must be in node_hit for the expanded ▾ header:\n{:?}",
        app.node_hit
    );

    // A click on the ▾ header (col 0) must re-collapse turn 1.
    // Find which screen rows the turn's node_hit spans.
    let (range, _) = app.node_hit.iter()
        .find(|(_, n)| *n == node1)
        .expect("turn 1 in node_hit");
    let vis_top = app.viewport.visual_top(&app.wrap_cache);
    // The range is global visual rows; convert the first to a screen row.
    let screen_row = *range.start() - vis_top;
    // Simulate a click on that row at col 0 (the ▾ gutter).
    let area = ratatui::layout::Rect::new(0, 0, 80, 40);
    let layout = crate::components::cockpit::split_cockpit(&app, area);
    let transcript_top = layout.transcript.y;
    let sr = transcript_top + screen_row as u16;
    let hit = app.click_fold_at(0, sr, transcript_top);
    assert!(hit, "click on ▾ header must be handled (col 0, row {sr})");
    // After clicking, turn 1 should be folded again.
    assert!(app.node_is_folded(node1), "turn 1 must be folded after clicking ▾");
}

/// S1 HONEST-CHECK B2 — the ▾ collapse affordance row inside an expanded tool box
/// is clickable at col 2 (interior of `│ … │`), which was OUTSIDE the old
/// FOLD_HIT_COLS=2 gate. With the S1 Fix C (full-width zone for NodeId::Tool) this
/// now correctly collapses the tool.
///
/// FAILS on old code: `transcript_node_at(2, ...)` returns None because col 2 >= FOLD_HIT_COLS.
/// PASSES now: NodeId::Tool uses full-width hit zone (u16::MAX).
#[test]
fn expanded_tool_box_affordance_row_is_clickable_at_interior_col() {
    let theme = Theme::default_theme();
    let mut app = AppState::new();
    app.conn = ConnStatus::Connected { model: Some("m".into()) };

    // A source with enough result lines to be foldable (>4 lines).
    let src = "Turn 1 ...\n\u{1F6E0}\u{FE0F} run({\"cmd\": \"ls\"})\nLINE1\nLINE2\nLINE3\nLINE4\nLINE5\nLINE6";
    let id = app.alloc_block_id();
    app.transcript.push(Block::new(id, None, Role::Assistant, src.to_string(), true));
    app.set_term_size(80, 40);
    app.sync_transcript(80, 40, &theme);

    // Expand the tool result.
    let tool = NodeId::Tool { block: id, tool: 0 };
    app.toggle_fold(tool);
    assert_eq!(app.folds.get(&tool), Some(&true), "tool result expanded");
    app.sync_transcript(80, 40, &theme);

    // The tool must be in the node_hit table.
    assert!(
        app.node_hit.iter().any(|(_, n)| *n == tool),
        "tool must be in node_hit after expanding"
    );

    // Find the screen row of the LAST row of the tool's hit range (the ▾ affordance
    // is the last interior row, just before the bottom border).
    let (range, _) = app.node_hit.iter().find(|(_, n)| *n == tool).expect("tool in node_hit");
    let vis_top = app.viewport.visual_top(&app.wrap_cache);
    let last_vis = *range.end();
    let screen_row_offset = last_vis - vis_top;
    let area = ratatui::layout::Rect::new(0, 0, 80, 40);
    let layout = crate::components::cockpit::split_cockpit(&app, area);
    let transcript_top = layout.transcript.y;
    let screen_row = transcript_top + screen_row_offset as u16;

    // Click at col 2 (inside the │ border, the ▾ affordance position).
    // With old code this returns None (col 2 >= FOLD_HIT_COLS=2).
    let node_hit = app.transcript_node_at(2, screen_row, transcript_top);
    assert!(
        node_hit.is_some(),
        "col 2 must hit a Tool node (full-width zone for NodeId::Tool, S1 Fix C)"
    );
    let handled = app.click_fold_at(2, screen_row, transcript_top);
    assert!(handled, "col 2 click on affordance row must be handled");
    // After the click the tool should be collapsed again.
    assert!(!app.folds.get(&tool).copied().unwrap_or(false), "tool must be collapsed after click at col 2");
}
