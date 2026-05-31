//! tui_v4 — GenericAgent's terminal UI, rebuilt in Rust + ratatui.
//!
//! This is the FOUNDATION / compiling vertical slice (checklist §11 Phase F):
//!   * arg parse: `--smoke` (render one frame to an in-memory backend, no python
//!     bridge, exit 0), `--version`, `--help`.
//!   * alternate screen + raw mode via crossterm, with a panic hook AND a
//!     normal-exit path that ALWAYS restore the terminal.
//!   * the main event loop: crossterm input events + a 0.1s tick + the bridge
//!     event channel, multiplexed over a bounded std::sync::mpsc.
//!   * Submit on Enter → bridge; render MessageBegin/Delta/End into a FILLED
//!     layout (header + rainbow separator + flex transcript + composer + footer)
//!     with the custom (non-✻) spinner.

mod app;
mod bridge;
mod commands;
mod components;
mod effects;
mod flavor;
mod i18n;
mod input;
mod markdown;
mod render;
mod theme;
mod util;
mod workflow;

use std::io::{self, Stdout};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers, MouseEvent, MouseEventKind,
};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::{execute, ExecutableCommand};
use ratatui::backend::{CrosstermBackend, TestBackend};
use ratatui::layout::Rect;
use ratatui::Terminal;

use app::{AppState, View};
use bridge::protocol::UiToCore;
use bridge::{BridgeEvent, BridgeOptions};
use theme::Theme;

const VERSION: &str = "0.2.0";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    // --version / --help short-circuit before any terminal setup.
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("tui_v4 {VERSION}");
        return Ok(());
    }
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return Ok(());
    }
    if args.iter().any(|a| a == "--smoke") {
        // Render exactly one frame to an in-memory backend and exit 0 WITHOUT
        // spawning the python bridge (so it can never hang). This is the CI gate.
        return run_smoke();
    }
    if let Some(pos) = args.iter().position(|a| a == "--dump-frame") {
        // Render the cockpit with seeded state into an in-memory backend and
        // print it as TEXT ROWS, so the LAYOUT is inspectable headlessly. This
        // closes the gap that let header/footer/shell-mode layout bugs ship
        // (a passing --smoke only counts non-blank cells; it can't SEE layout).
        // Scenario: normal | shell | busy.
        let scenario = args.get(pos + 1).map(|s| s.as_str()).unwrap_or("normal");
        return run_dump_frame(scenario);
    }

    run_app()
}

fn print_help() {
    println!(
        "tui_v4 {VERSION} — GenericAgent terminal UI (Rust + ratatui)\n\
\n\
USAGE:\n\
    tui_v4 [OPTIONS]\n\
\n\
OPTIONS:\n\
    --smoke          Render one frame to an in-memory backend and exit 0\n\
                     (no python bridge; used as the build/CI gate).\n\
    --version, -V    Print the version and exit.\n\
    --help, -h       Print this help and exit.\n\
\n\
ENVIRONMENT:\n\
    GA_TUI_BRIDGE        Explicit path to scripts/ga_bridge.py.\n\
    GENERICAGENT_ROOT    GA repo root (contains agentmain.py + temp/).\n\
\n\
With no options it launches the full TUI: it discovers and spawns\n\
scripts/ga_bridge.py (Python), handshakes, and streams a live session."
    );
}

/// `--smoke`: build state, render ONE frame into an 80x24 in-memory TestBackend,
/// and return 0. Touches no real terminal and spawns no child — guaranteed not
/// to hang.
fn run_smoke() -> Result<()> {
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend)?;

    let mut app = AppState::new();
    // Seed a representative state so the frame exercises every region (header,
    // separator, transcript with a streamed reply, spinner band, composer).
    app.conn = app::ConnStatus::Connected {
        model: Some("smoke-model".into()),
    };
    app.model = Some("smoke-model".into());
    app.push_user("hello from --smoke".into());
    app.apply_bridge_event(
        BridgeEvent::Frame(bridge::protocol::CoreToUi::MessageBegin {
            mid: "m1".into(),
            role: "assistant".into(),
        }),
        0,
    );
    // Stream markdown + inline math so the smoke frame exercises the full
    // markdown render plane (headings, bold, inline code, `$…$` math), not just
    // plain text — proving the assistant-block routing renders without a bridge.
    app.apply_bridge_event(
        BridgeEvent::Frame(bridge::protocol::CoreToUi::MessageDelta {
            mid: "m1".into(),
            text: "## Result\n\nThe angle $\\alpha$ is **small**; see `main.rs`.".into(),
        }),
        0,
    );
    let theme = Theme::ga_default();

    terminal.draw(|f| components::render(f, &mut app, &theme, 100))?;

    // Confirm the frame actually has content (the separator cells, at minimum)
    // so `--smoke` is a real render assertion, not a no-op exit.
    let buffer = terminal.backend().buffer();
    let non_empty = buffer.content().iter().filter(|c| c.symbol() != " ").count();
    if non_empty == 0 {
        anyhow::bail!("smoke render produced an empty frame");
    }

    println!("tui_v4 {VERSION} smoke ok: rendered 1 frame ({non_empty} non-blank cells)");
    Ok(())
}

/// `--dump-frame [normal|shell|busy]`: render the cockpit with representative
/// seeded state into a 100x30 in-memory backend and print every row as text, so
/// the actual LAYOUT can be eyeballed without a TTY (and asserted in review).
fn run_dump_frame(scenario: &str) -> Result<()> {
    let (w, h) = (100u16, 30u16);
    let backend = TestBackend::new(w, h);
    let mut terminal = Terminal::new(backend)?;

    let mut app = AppState::new();
    app.lang = i18n::detect_system_lang();
    // Seed the REAL MixinSession model shape (redesign_cc.md §2.5): `get_llm_name()`
    // returns a long pipe-list of the whole fallback chain. The header/footer MUST
    // truncate this to the primary segment (`MixinSession·codex-pro`), never the
    // full `…|kiro` list — the dump verifies that.
    let model = "MixinSession/codex-pro|gpt-5.2|claude-opus-4|gemini-2.5-pro|grok-4|kiro";
    app.conn = app::ConnStatus::Connected {
        model: Some(model.into()),
    };
    app.model = Some(model.into());
    app.push_user("hello".into());
    app.apply_bridge_event(
        BridgeEvent::Frame(bridge::protocol::CoreToUi::MessageBegin {
            mid: "m1".into(),
            role: "assistant".into(),
        }),
        0,
    );
    // The assistant stream uses the REAL GA markers (redesign_cc.md §1) so the dump
    // verifies they render CLEAN: a bare `Turn N ...` boundary (→ spacing, no text),
    // a `<summary>…</summary>` (→ dim breadcrumb, tags hidden), prose, a COMPACT
    // `🛠️ name(args)` tool call (→ ⏺ bullet) with an `[Info] …` result (→ dim,
    // indented), and an inline `!!!Error:` (→ compact red line). The 🛠️ / <summary>
    // in THIS seed string are the parser's input — they must NOT survive to the
    // rendered output.
    app.apply_bridge_event(
        BridgeEvent::Frame(bridge::protocol::CoreToUi::MessageDelta {
            mid: "m1".into(),
            text: "Turn 1 ...\n\
                   <summary>用户打招呼，扫描浏览器标签页</summary>\n\
                   Sure — let me scan the open tabs first.\n\
                   The cost grows like $\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}$ here.\n\
                   🛠️ web_scan({\"tabs_only\": true})\n\
                   [Info] 3 tabs scanned · ok\n\
                   !!!Error: SSE overloaded"
                .into(),
        }),
        0,
    );
    // Seed a token/context snapshot (what the bridge now emits) so the dump shows
    // the redesigned spinner readout `↑in ↓out · ctx ▰▱ pct%` + a populated /cost.
    app.apply_bridge_event(
        BridgeEvent::Frame(bridge::protocol::CoreToUi::Status {
            model: None,
            context_percent: Some(48.0),
            tokens: Some(1574),
            input_tokens: Some(1234),
            output_tokens: Some(340),
            cache_tokens: Some(96),
            last_input: Some(1234),
            last_output: Some(340),
            text: None,
        }),
        0,
    );
    // Simulate a failover retry diagnostic arriving on the bridge's STDERR
    // (llmcore.py:988 `[MixinSession] …retry N/M` → BridgeEvent::Stderr). The
    // transcript MUST suppress this — it should NOT appear as a `[bridge]` row.
    app.apply_bridge_event(
        BridgeEvent::Stderr {
            line: "[MixinSession] codex-pro overloaded, retry 1/10 (2.0s→4.0s)".into(),
        },
        0,
    );
    match scenario {
        "busy" => {
            // The ONLY spinner frame: leave the turn open (busy=true) so the
            // spinner band + `running` footer pill render.
        }
        "effort" => {
            // Render the `/effort` slider overlay (redesign_cc.md §3) over a busy
            // cockpit so the dump verifies BOTH the slider chrome AND the
            // `thinking · <level>` spinner suffix. Seed a live level so the `●`
            // applied-marker + the suffix show; open the slider seeded there.
            app.set_reasoning_effort(app::effort::ReasoningEffort::Medium);
            app.open_effort_slider();
        }
        "effort-high" => {
            // Same slider, marker nudged right (→ → from medium = xhigh) so the dump
            // shows the marker MOVED off the applied `●` stop.
            app.set_reasoning_effort(app::effort::ReasoningEffort::Medium);
            app.open_effort_slider();
            if let Some(app::Overlay::EffortSlider(s)) = app.overlay.as_mut() {
                s.move_marker(2);
            }
        }
        "shell" => {
            // Finalize the turn (idle) so shell mode is shown WITHOUT the spinner
            // band — the pink composer/footer is the only shell signal.
            app.apply_bridge_event(
                BridgeEvent::Frame(bridge::protocol::CoreToUi::MessageEnd {
                    mid: "m1".into(),
                    reason: "stop".into(),
                }),
                0,
            );
            let _ = app.composer.type_str("!ls -la");
        }
        "cost" => {
            // Finalize the turn, then open the /cost overlay so the dump shows the
            // token-usage card populated from the seeded Status (the /cost fix:
            // input/output/cache/total/context% are now REAL, not all-zero).
            app.apply_bridge_event(
                BridgeEvent::Frame(bridge::protocol::CoreToUi::MessageEnd {
                    mid: "m1".into(),
                    reason: "stop".into(),
                }),
                0,
            );
            app.open_overlay(app::Overlay::Cost);
        }
        _ => {
            // `normal` (and any other arg): a TRUE IDLE cockpit. Finalize the turn
            // so busy=false → no spinner band, the footer pill reads `chat`, and the
            // idle composer/placeholder is rendered (the default-state review target
            // that a perpetually-busy frame could never show).
            app.apply_bridge_event(
                BridgeEvent::Frame(bridge::protocol::CoreToUi::MessageEnd {
                    mid: "m1".into(),
                    reason: "stop".into(),
                }),
                0,
            );
        }
    }

    let theme = Theme::ga_default();
    terminal.draw(|f| components::render(f, &mut app, &theme, 100))?;

    let buffer = terminal.backend().buffer();
    let content = buffer.content();
    let wi = w as usize;
    println!("=== dump-frame [{scenario}] {w}x{h} (trailing spaces trimmed) ===");
    for y in 0..h as usize {
        let mut line = String::new();
        for x in 0..wi {
            line.push_str(content[y * wi + x].symbol());
        }
        println!("{}", line.trim_end());
    }
    Ok(())
}

/// A unified event for the main loop, multiplexing the three sources.
enum AppEvent {
    /// A crossterm input event (key / resize / mouse).
    Term(Event),
    /// The 0.1s tick (spinner / gerund clock).
    Tick,
    /// A frame/lifecycle signal from a bridge child, TAGGED with its session id
    /// (the multi-session multiplexer routes it to the right session, §6 / N2).
    Bridge(u64, BridgeEvent),
}

/// The full interactive app: set up the terminal (alt screen + raw mode), spawn
/// the bridge, run the event loop, and ALWAYS restore the terminal on exit.
fn run_app() -> Result<()> {
    // Install the panic hook FIRST so a panic during setup/teardown still
    // restores the terminal (no garbled prompt left behind).
    install_panic_hook();

    let mut terminal = setup_terminal()?;

    // The single channel all three sources feed (bridge frames, ticks, input).
    // Input + tick run on their own thread so the loop never blocks.
    let (tx, rx): (Sender<AppEvent>, Receiver<AppEvent>) = mpsc::channel();
    spawn_input_thread(tx.clone());

    // The TAGGED bridge-event channel: every session's child forwards
    // `(session_id, BridgeEvent)` here; a forwarder lifts them onto the unified
    // `AppEvent` channel. This is the §6 multiplexer — N children, routed by id.
    let (tx_bridge, rx_bridge): (Sender<(u64, BridgeEvent)>, Receiver<(u64, BridgeEvent)>) =
        mpsc::channel();
    spawn_bridge_forwarder(rx_bridge, tx.clone());

    // Spawn the foundation bridge for session 1, tagged with id 1, and adopt it
    // into the SessionMap so it is owned per-session (not double-spawned). Other
    // sessions spawn their own children lazily on first use.
    let foundation = bridge::spawn_bridge_tagged(BridgeOptions::default(), 1, tx_bridge.clone());
    let repo_root = foundation.repo_root.clone();

    let mut app = AppState::new();
    // Detect the interface language from the environment (system-locale detect,
    // §9) BEFORE the first frame so the cockpit opens already-localized.
    app.lang = i18n::detect_system_lang();
    // Wire the discovered GA repo root: header cwd, @path/!cmd cwd, persisted
    // input history, and the git branch shown in the footer. (This rebuilds the
    // SessionMap on the real root; we adopt the foundation bridge AFTER.)
    app.attach_repo_root(repo_root);
    let session1 = app.sessions.active;
    app.sessions.adopt_bridge(session1, foundation);

    let start = Instant::now();
    let result = event_loop(&mut terminal, &mut app, &tx_bridge, &rx, start);

    // ALWAYS restore — both on the normal path and via the panic hook above.
    restore_terminal(&mut terminal)?;
    app.sessions.shutdown_all();

    result
}

/// The main event loop. Returns when the user quits.
fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    app: &mut AppState,
    tx_bridge: &Sender<(u64, BridgeEvent)>,
    rx: &Receiver<AppEvent>,
    start: Instant,
) -> Result<()> {
    loop {
        let now_ms = start.elapsed().as_millis() as u64;
        // While the /workflows panel is open, pull the watcher's latest merged
        // snapshot into the app before drawing (the watcher polls off-thread; this
        // is a cheap lock+clone that never blocks — §3 "never blocks chat"). When
        // the panel is closed the watcher is parked, so this is skipped entirely.
        if app.view == View::Workflows {
            app.refresh_workflow_snapshot();
        }
        // The theme is owned by the app (so `/theme` live-preview can swap it); the
        // render functions still take it by reference, so clone it for this frame.
        let theme = app.theme.clone();
        terminal.draw(|f| components::render(f, app, &theme, now_ms))?;
        // (`app` is already `&mut AppState` here, so `render` borrows it mutably.)

        // Emit the OSC0 title + OSC-21337 tab status when they change (§9). Done
        // AFTER the draw so the out-of-band escapes never interleave mid-frame.
        app.sync_terminal_chrome();

        // Block for the next event from ANY source (input thread, tick thread,
        // bridge forwarder). A short recv timeout keeps the elapsed clock /
        // spinner ticking even if no event arrives.
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(AppEvent::Term(ev)) => handle_term_event(ev, app, tx_bridge, now_ms),
            Ok(AppEvent::Tick) => app.tick(),
            Ok(AppEvent::Bridge(sid, be)) => {
                let now_ms = start.elapsed().as_millis() as u64;
                // Route the tagged event to its session (active → live reducer;
                // others → their own record so the dashboard preview updates).
                app.apply_tagged_event(sid, be, now_ms);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => app.tick(),
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }

        if app.should_quit {
            break;
        }
    }
    Ok(())
}

/// How many transcript rows one mouse-wheel notch scrolls (recon §6.7).
const WHEEL_STEP: isize = 3;

/// Fold one terminal event into state. Routing forks on the active VIEW: the
/// dashboard (§6) owns its own keys/clicks; the cockpit handles composer + scroll.
fn handle_term_event(ev: Event, app: &mut AppState, tx_bridge: &Sender<(u64, BridgeEvent)>, now_ms: u64) {
    match ev {
        Event::Key(key) => {
            // A MODAL overlay (picker / ask-user / help / cost / scheduler /
            // continue) consumes keys FIRST — the topmost layer of the §3 overlay
            // stack. A NON-modal overlay (the `/btw` toast) only intercepts Esc
            // (dismiss); every other key flows to the cockpit so chat stays usable
            // while the side question is in flight (§7 `/btw` non-blocking).
            let modal = app.overlay.as_ref().map(|o| o.is_modal());
            match modal {
                Some(true) => handle_overlay_key(key, app, tx_bridge),
                Some(false) => {
                    if key.code == KeyCode::Esc && key.kind != KeyEventKind::Release {
                        app.close_overlay(); // dismiss the /btw card (no history pollution).
                    } else {
                        route_view_key(key, app, tx_bridge, now_ms);
                    }
                }
                None => route_view_key(key, app, tx_bridge, now_ms),
            }
        }
        Event::Mouse(me) => handle_mouse_event(me, app),
        // Resize is handled implicitly: the next `render` re-syncs the wrap cache
        // + viewport to the new geometry and re-derives the window from the same
        // logical anchor (P1). ratatui repaints from state, so nothing to do here.
        _ => {}
    }
}

/// Mouse: in the cockpit the wheel scrolls the transcript and a left-click on the
/// header/footer "sessions area" opens the dashboard (§6 "left-click on the
/// sessions area"); in the dashboard a left-click on a row switches into it.
fn handle_mouse_event(me: MouseEvent, app: &mut AppState) {
    // A click anywhere while a MODAL overlay is up dismisses it (a click-outside
    // affordance); the overlay's own keys handle precise selection. A NON-modal
    // overlay (the `/btw` toast) lets the wheel keep scrolling the cockpit beneath
    // (chat stays usable) — only a left-click dismisses it.
    if let Some(modal) = app.overlay.as_ref().map(|o| o.is_modal()) {
        let is_left = matches!(me.kind, MouseEventKind::Down(crossterm::event::MouseButton::Left));
        if modal {
            if is_left {
                app.close_overlay();
            }
            return;
        }
        // Non-modal: a left-click dismisses; otherwise fall through to cockpit
        // scroll handling below.
        if is_left {
            app.close_overlay();
            return;
        }
    }
    use crossterm::event::MouseButton;
    match (app.view, me.kind) {
        // -- cockpit --
        (View::Cockpit, MouseEventKind::ScrollUp) => app.scroll_lines(-WHEEL_STEP),
        (View::Cockpit, MouseEventKind::ScrollDown) => app.scroll_lines(WHEEL_STEP),
        (View::Cockpit, MouseEventKind::Down(MouseButton::Left)) => {
            // The header + rainbow separator (rows 0–1) are the "sessions area" — a
            // left-click there opens the full-screen dashboard (the §4/§6 left-click
            // entry point). Broadened from row 0 to the whole header band so the
            // click target isn't a 1-row sliver (redesign_cc.md §4 "broaden … a bit").
            if me.row <= 1 {
                app.open_dashboard();
            }
        }
        // Right-click in the cockpit → no-op (redesign_cc.md §4: right-click is the
        // universal "go back", and the cockpit is already the root view). Matched
        // explicitly so it's a deliberate no-op, not an accidental fall-through.
        (View::Cockpit, MouseEventKind::Down(MouseButton::Right)) => { /* no-op (already at root) */ }
        // -- dashboard: right-click anywhere → go BACK to the cockpit (§4). This is
        // the missing `MouseButton::Right` handler — the mouse mirror of `Esc`.
        (View::Dashboard, MouseEventKind::Down(MouseButton::Right)) => app.close_dashboard(),
        // -- dashboard: left-click a session row → switch into it --
        (View::Dashboard, MouseEventKind::Down(MouseButton::Left)) => {
            let area = Rect {
                x: 0,
                y: 0,
                width: app.last_term_width(),
                height: app.last_term_height(),
            };
            let rows = app.sessions.dashboard_rows();
            if let Some(idx) =
                components::dashboard::click_to_row_index(me.column, me.row, area, rows.len(), app.sessions.dash_sel)
            {
                app.sessions.dash_sel = idx;
                if let Some(id) = app.sessions.selected_session_id() {
                    app.switch_session(id);
                }
            }
        }
        _ => {}
    }
}

/// The composer's inner text width (borders + prompt) for visual-row nav.
fn composer_width(app: &AppState) -> u16 {
    // Best-effort: the real width is known at draw time, but key handling needs
    // a width for visual Up/Down. Use the last-synced transcript width minus the
    // composer chrome; fall back to 80.
    app.transcript_width().saturating_sub(4).max(1)
}

fn handle_key_event(key: KeyEvent, app: &mut AppState, tx_bridge: &Sender<(u64, BridgeEvent)>, now_ms: u64) {
    use input::{ComposerAction, Nav};

    let KeyEvent {
        code,
        modifiers,
        kind,
        ..
    } = key;
    // crossterm on Windows fires Press AND Release; only act on Press/Repeat.
    if kind == KeyEventKind::Release {
        return;
    }
    let ctrl = modifiers.contains(KeyModifiers::CONTROL);
    let shift = modifiers.contains(KeyModifiers::SHIFT);
    let width = composer_width(app);

    // --- global chords (handled before composer editing) ---------------------
    match code {
        // Ctrl+C — the 3-STAGE chord (§8): 1st press copies a selection if one
        // exists (OSC-52), else aborts a running turn, else ARMS quit + shows the
        // "press Ctrl+C again to quit" hint; a 2nd Ctrl+C within 2s quits. The arm
        // expires after 2s so a stale arm never silently exits. Ctrl+Q stays a
        // plain immediate quit (the unambiguous, never-armed exit key).
        // Ctrl+Shift+C — copy the LAST assistant reply (inline-copy shortcut, same
        // as /export clip) with a visible byte count. MUST precede the plain Ctrl+C
        // arm below (its `if ctrl` guard also matches Ctrl+Shift+C). Shift may make
        // crossterm deliver the char uppercased, so match both 'c'/'C'.
        KeyCode::Char('c' | 'C') if ctrl && shift => {
            export_action(app, 0);
            return;
        }
        // Ctrl+Shift+M — toggle mouse capture so the terminal's native drag-select
        // works for inline copy (capture ON = wheel scroll + click-to-dashboard).
        KeyCode::Char('m' | 'M') if ctrl && shift => {
            app.mouse_capture = !app.mouse_capture;
            set_mouse_capture(app.mouse_capture);
            let k = if app.mouse_capture { "mouse.on" } else { "mouse.off" };
            app.push_notice(i18n::tf(app.lang, k));
            return;
        }
        KeyCode::Char('c') if ctrl => {
            cockpit_ctrl_c(app, tx_bridge, now_ms);
            return;
        }
        KeyCode::Char('q') if ctrl => {
            app.should_quit = true;
            return;
        }
        // Ctrl+L — force a full redraw (sleep/wake recovery).
        KeyCode::Char('l') if ctrl => {
            // ratatui repaints from state each frame; a Changed is enough to
            // trigger the next draw. (A hard clear is the terminal's job.)
            return;
        }
        // Ctrl+S — open the full-screen session dashboard (§6 / N2 entry point).
        // (Draft stashing on switch is handled by the dashboard's switch path.)
        KeyCode::Char('s') if ctrl => {
            app.open_dashboard();
            return;
        }
        // Ctrl+N — create + switch to a new session (the composer's draft is
        // stashed onto the leaving session by switch_session).
        KeyCode::Char('n') if ctrl => {
            cockpit_new_session(app);
            return;
        }
        // Ctrl+W / Ctrl+D — drop (close) the active session, KEEPING its log.
        KeyCode::Char('w') | KeyCode::Char('d') if ctrl => {
            cockpit_drop_active(app);
            return;
        }
        // Ctrl+B — branch the active session (fork w/ copied transcript) + switch.
        KeyCode::Char('b') if ctrl => {
            let draft = app.composer.text().to_string();
            app.snapshot_active_into_map();
            let new_id = app.sessions.branch(draft);
            app.load_active_session_after_structural_change(new_id);
            return;
        }
        // Ctrl+Up / Ctrl+Down — cycle the active session (prev / next, wrapping).
        KeyCode::Up if ctrl => {
            cockpit_cycle(app, -1);
            return;
        }
        KeyCode::Down if ctrl => {
            cockpit_cycle(app, 1);
            return;
        }
        // Ctrl+O — toggle tool-chip / turn folding.
        KeyCode::Char('o') if ctrl => {
            app.fold_all = !app.fold_all;
            return;
        }
        // Ctrl+Y — redo (NOT copy — copy is Ctrl+Shift+Y / a menu in Phase 3).
        KeyCode::Char('y') if ctrl => {
            app.composer.redo();
            return;
        }
        // Ctrl+Z — undo.
        KeyCode::Char('z') if ctrl => {
            app.composer.undo();
            return;
        }
        // Ctrl+A — select all (composer).
        KeyCode::Char('a') if ctrl => {
            app.composer.select_all();
            return;
        }
        // Ctrl+E — end of line.
        KeyCode::Char('e') if ctrl => {
            app.composer.end_of_line(width);
            return;
        }
        // Ctrl+U — kill to line start.
        KeyCode::Char('u') if ctrl => {
            app.composer.kill_to_line_start();
            return;
        }
        // Ctrl+X — cut selection to clipboard.
        KeyCode::Char('x') if ctrl => {
            let (_, cut) = app.composer.cut();
            if let Some(text) = cut {
                let label = i18n::t(app.lang, "copy.label.cut");
                let res = copy_text(&text);
                notice_copy(app, &res, label);
            }
            return;
        }
        // Ctrl+V — paste from the native clipboard.
        KeyCode::Char('v') if ctrl => {
            if let Some(text) = read_clipboard() {
                app.composer.paste(&text);
            }
            return;
        }
        // (Ctrl+S now opens the session dashboard — handled in the chords block
        // above. Draft stashing moves to Ctrl+G so Ctrl+S is the §6 dashboard key.)
        KeyCode::Char('g') if ctrl => {
            app.composer.stash_or_restore();
            return;
        }
        // Ctrl+J — insert a newline (universal fallback for Shift+Enter).
        KeyCode::Char('j') if ctrl => {
            app.composer.newline();
            return;
        }
        _ => {}
    }

    // --- the @ file-picker / slash-palette intercept (Tab/Enter complete) ----
    if try_complete_dropdown(app, code, shift) {
        return;
    }

    // --- composer + transcript navigation ------------------------------------
    match code {
        // Shift+Enter inserts a newline; plain Enter submits.
        KeyCode::Enter if shift => {
            app.composer.newline();
        }
        KeyCode::Enter => {
            let action = app.composer.submit();
            dispatch_action(app, tx_bridge, action);
        }
        KeyCode::Backspace => {
            app.composer.backspace();
            app.palette_sel = 0; // the `/`-palette match set changed → reset highlight.
        }
        KeyCode::Delete => {
            app.composer.delete();
            app.palette_sel = 0;
        }
        // Arrows: composer cursor / selection (history at vertical edges). The
        // transcript scrolls via PageUp/PageDown + the mouse wheel.
        //
        // EXCEPTION (redesign request #2): when the composer is EMPTY there is
        // nothing to navigate, so ←/→ SWITCH VIEWS — the intuitive replacement for
        // the unobvious Ctrl+S. → enters the session dashboard, ← returns to chat.
        // (Ctrl+S still toggles; Esc still backs out.) A non-empty buffer keeps the
        // normal cursor nav so typing/editing is never hijacked.
        KeyCode::Left if app.composer.is_empty() && !shift => {
            app.close_dashboard(); // already-in-cockpit → no-op; harmless + future-proof.
        }
        KeyCode::Right if app.composer.is_empty() && !shift => {
            app.open_dashboard();
        }
        KeyCode::Left => {
            app.composer.nav(Nav::Left, shift, width);
        }
        KeyCode::Right => {
            app.composer.nav(Nav::Right, shift, width);
        }
        KeyCode::Up => {
            app.composer.nav(Nav::Up, shift, width);
        }
        KeyCode::Down => {
            app.composer.nav(Nav::Down, shift, width);
        }
        // Ctrl+Home / Ctrl+End jump the TRANSCRIPT to top / tail; plain Home/End
        // move within the composer line.
        KeyCode::Home if ctrl => app.scroll_home(),
        KeyCode::End if ctrl => app.scroll_end(),
        KeyCode::Home => {
            app.composer.nav(Nav::Home, shift, width);
        }
        KeyCode::End => {
            app.composer.nav(Nav::End, shift, width);
        }
        // Transcript scrolling (P1).
        KeyCode::PageUp => app.page_up(),
        KeyCode::PageDown => app.page_down(),
        KeyCode::Esc => {
            // Esc-Esc chord (§8): a 2nd Esc within 0.8s opens the /rewind picker; a
            // single Esc keeps the universal-back behavior (never exits). The pure
            // decider (timestamps injected) lives in `input::keychord`.
            cockpit_esc(app, tx_bridge, now_ms, width);
        }
        KeyCode::Char(c) => {
            app.composer.type_str(&c.to_string());
            app.palette_sel = 0; // typing changes the `/`-palette matches → reset highlight.
        }
        _ => {}
    }
    let _ = ComposerAction::None; // (action enum referenced via dispatch_action)
}

/// Route a key to the active VIEW's handler (cockpit / dashboard / workflows).
/// Centralizes the per-view fork so the modal-overlay branches above don't each
/// re-list the views.
fn route_view_key(key: KeyEvent, app: &mut AppState, tx_bridge: &Sender<(u64, BridgeEvent)>, now_ms: u64) {
    match app.view {
        View::Dashboard => handle_dashboard_key(key, app, tx_bridge),
        View::Workflows => handle_workflows_key(key, app, tx_bridge),
        View::Cockpit => handle_key_event(key, app, tx_bridge, now_ms),
    }
}

/// The full-screen `/workflows` PANEL key handler (§7). Navigates the workflow
/// tree, toggles the render style, opens the per-node detail overlay, and fires
/// node action verbs. When the detail overlay is open the keys drive its action
/// menu instead. `Esc` closes the detail overlay (if open) else returns to the
/// cockpit (never exits the app). The watcher keeps refreshing the snapshot in the
/// background — these keys only move the cursor / fire one-shot actions.
fn handle_workflows_key(key: KeyEvent, app: &mut AppState, _tx_bridge: &Sender<(u64, BridgeEvent)>) {
    let KeyEvent { code, modifiers, kind, .. } = key;
    if kind == KeyEventKind::Release {
        return;
    }
    let ctrl = modifiers.contains(KeyModifiers::CONTROL);
    // Ctrl+C / Ctrl+Q still quit from the panel.
    if (code == KeyCode::Char('c') || code == KeyCode::Char('q')) && ctrl {
        app.should_quit = true;
        return;
    }

    let snap = app.workflow_snapshot.clone();

    // -- detail-overlay sub-mode: the action menu (↑↓ select, Enter fire, Esc close).
    if app.workflow_panel.detail_open() {
        match code {
            KeyCode::Esc => {
                app.workflow_panel.close_detail();
            }
            KeyCode::Up => app.workflow_panel.move_action(-1, &snap),
            KeyCode::Down => app.workflow_panel.move_action(1, &snap),
            KeyCode::Enter => fire_workflow_action(app, &snap),
            _ => {}
        }
        return;
    }

    match code {
        // Esc / Ctrl+S-style: leave the panel back to the cockpit (never exits).
        KeyCode::Esc => app.close_workflows(),
        // Up/Down move the focus over the focusable nodes (skips group headers).
        KeyCode::Up => app.workflow_panel.move_focus(-1, &snap),
        KeyCode::Down => app.workflow_panel.move_focus(1, &snap),
        // PageUp/PageDown jump a chunk of nodes.
        KeyCode::PageUp => app.workflow_panel.move_focus(-8, &snap),
        KeyCode::PageDown => app.workflow_panel.move_focus(8, &snap),
        // `t` toggles the render style (box-tree ⇄ compact bullet list, §7).
        KeyCode::Char('t') if !ctrl => app.workflow_panel.toggle_style(),
        // Enter opens the per-node DETAIL overlay (full prompt + feed + actions).
        KeyCode::Enter => app.workflow_panel.open_detail(&snap),
        // `r` refreshes the snapshot immediately (otherwise it polls on cadence).
        KeyCode::Char('r') if !ctrl => app.refresh_workflow_snapshot(),
        _ => {}
    }
}

/// Fire the workflow detail overlay's selected node action (§7 "node action
/// verbs"). A conductor mutate verb (`keyinfo|input|stop|kill`) is sent off-thread
/// as a `POST /subagent/{id}` via the watcher (so a slow POST never blocks the UI);
/// `Open` is a UI no-op (the detail is already open). After a mutate the overlay
/// closes so the next poll's fresh state is shown. A verb against a DOWN workflow is
/// ignored (the panel dims it). The conductor port is the watcher's well-known one.
fn fire_workflow_action(app: &mut AppState, snap: &workflow::schema::WorkflowSnapshot) {
    use workflow::NodeAction;
    let Some((action, wf, node)) = app.workflow_panel.selected_action(snap) else {
        return;
    };
    match action {
        NodeAction::Open => { /* already open — no-op */ }
        other => {
            if let Some(verb) = other.conductor_verb() {
                // Only conductor workflows have a mutate API; only when up.
                if wf.kind == workflow::schema::WorkflowKind::Conductor && wf.running {
                    workflow::fire_conductor_action(
                        workflow::CONDUCTOR_PORT,
                        node.id.clone(),
                        verb,
                        String::new(),
                    );
                    app.push_notice(format!(
                        "{} {} → {}",
                        i18n::t(app.lang, "wf.action.sent"),
                        node.label,
                        i18n::t(app.lang, other.label_key()),
                    ));
                } else {
                    app.push_notice(i18n::tf(app.lang, "wf.action.unavailable"));
                }
            }
            app.workflow_panel.close_detail();
        }
    }
}

/// The full-screen session DASHBOARD key handler (§6 / N2). Routes navigation +
/// CRUD over the multiplexed sessions. When a rename is in flight the keys edit
/// the rename buffer instead. `Esc` returns to the cockpit (or cancels a rename).
fn handle_dashboard_key(key: KeyEvent, app: &mut AppState, tx_bridge: &Sender<(u64, BridgeEvent)>) {
    let KeyEvent { code, modifiers, kind, .. } = key;
    if kind == KeyEventKind::Release {
        return;
    }
    let ctrl = modifiers.contains(KeyModifiers::CONTROL);

    // -- rename sub-mode: edit the rename buffer (Enter commits, Esc cancels). --
    if app.rename.is_some() {
        match code {
            KeyCode::Esc => {
                app.rename = None;
            }
            KeyCode::Enter => {
                if let Some(r) = app.rename.take() {
                    app.sessions.rename(r.id, r.buffer);
                }
            }
            KeyCode::Backspace => {
                if let Some(r) = app.rename.as_mut() {
                    r.buffer.pop();
                }
            }
            KeyCode::Char(c) if !ctrl => {
                if let Some(r) = app.rename.as_mut() {
                    r.buffer.push(c);
                }
            }
            _ => {}
        }
        return;
    }

    // -- global dashboard chords --
    match code {
        // Esc / Ctrl+S toggle back to the cockpit (never exits the app).
        KeyCode::Esc => {
            app.close_dashboard();
            return;
        }
        KeyCode::Char('s') if ctrl => {
            app.close_dashboard();
            return;
        }
        // Ctrl+C / Ctrl+Q still quit from the dashboard.
        KeyCode::Char('c') | KeyCode::Char('q') if ctrl => {
            app.should_quit = true;
            return;
        }
        _ => {}
    }

    match code {
        // Up/Down navigate (collapsed children are absent from the row list, so
        // this naturally skips them — the dashboard_nav_skips_collapsed contract).
        KeyCode::Up => app.sessions.move_dash_sel(-1),
        KeyCode::Down => app.sessions.move_dash_sel(1),

        // Tab toggles the selected category's collapse (the keyboard analogue of
        // clicking the ▸/▾ chevron). Left/Right arrows also collapse/expand it, a
        // natural tree gesture that doesn't steal printable chars from the input.
        KeyCode::Tab => {
            if let Some(cat) = app.sessions.selected_category() {
                app.sessions.toggle_category(cat);
            }
        }
        KeyCode::Left => {
            // Tree gesture + the ←/→ view switch (redesign #2): collapse an EXPANDED
            // category; otherwise ← exits the dashboard back to the chat cockpit
            // (the mirror of the cockpit's empty-composer → that opened it).
            match app.sessions.selected_category() {
                Some(cat) if !app.sessions.is_collapsed(cat) => app.sessions.toggle_category(cat),
                _ => app.close_dashboard(),
            }
        }
        KeyCode::Right => {
            if let Some(cat) = app.sessions.selected_category() {
                if app.sessions.is_collapsed(cat) {
                    app.sessions.toggle_category(cat);
                }
            }
        }

        // Enter: if the bottom new-session input has text → create a new session
        // seeded with it; else open/switch into the selected session. (A header
        // row with no input toggles its collapse, for a sensible Enter.)
        KeyCode::Enter => {
            let seed = app.sessions.new_session_input.trim().to_string();
            if !seed.is_empty() {
                dashboard_new_session(app, tx_bridge, seed);
            } else if let Some(id) = app.sessions.selected_session_id() {
                app.switch_session(id);
            } else if let Some(cat) = app.sessions.selected_category() {
                app.sessions.toggle_category(cat);
            }
        }

        // Space: quick-reply. If the bottom input has text, send it to the SELECTED
        // session as a reply (staying in the dashboard); else switch into it.
        KeyCode::Char(' ') => {
            let seed = app.sessions.new_session_input.trim().to_string();
            if let Some(id) = app.sessions.selected_session_id() {
                if seed.is_empty() {
                    app.switch_session(id);
                } else {
                    dashboard_quick_reply(app, tx_bridge, id, seed);
                }
            } else {
                // No session under the cursor: a Space just types into the input.
                app.sessions.new_session_input.push(' ');
            }
        }

        // r — rename the selected session (open the inline rename editor). Only
        // when the bottom new-session input is EMPTY, so a task description that
        // begins with 'r' still types normally (empty input = command mode;
        // non-empty = typing mode — the focus rule that avoids the Ink Tab dance).
        KeyCode::Char('r') if !ctrl && app.sessions.new_session_input.is_empty() => {
            if let Some(id) = app.sessions.selected_session_id() {
                let name = app.sessions.session(id).map(|s| s.name.clone()).unwrap_or_default();
                app.rename = Some(app::RenameState { id, buffer: name });
            }
        }

        // Del — delete the selected session (KEEPING its log).
        KeyCode::Delete => dashboard_delete_selected(app),
        // Ctrl+X / Ctrl+W / Ctrl+D — delete (drop) the selected session.
        KeyCode::Char('x') | KeyCode::Char('w') | KeyCode::Char('d') if ctrl => {
            dashboard_delete_selected(app);
        }

        // Ctrl+N — new session (with no seed it makes a blank session + opens it).
        KeyCode::Char('n') if ctrl => {
            let seed = app.sessions.new_session_input.trim().to_string();
            dashboard_new_session(app, tx_bridge, seed);
        }

        // Backspace edits the bottom new-session input.
        KeyCode::Backspace => {
            app.sessions.new_session_input.pop();
        }

        // Any other printable char types into the bottom new-session input
        // (the "just start typing a task" flow from the screenshot).
        KeyCode::Char(c) if !ctrl => {
            app.sessions.new_session_input.push(c);
        }
        _ => {}
    }
}

/// The MODAL OVERLAY key handler (§3 overlay stack / §7): routes keys into the
/// active picker / ask-user card / info overlay. `Esc` closes (reverting a
/// live-preview theme); Enter applies the selection. Ctrl+C still quits.
fn handle_overlay_key(key: KeyEvent, app: &mut AppState, tx_bridge: &Sender<(u64, BridgeEvent)>) {
    use app::Overlay;
    let KeyEvent { code, modifiers, kind, .. } = key;
    if kind == KeyEventKind::Release {
        return;
    }
    let ctrl = modifiers.contains(KeyModifiers::CONTROL);
    // Ctrl+C / Ctrl+Q quit even from an overlay.
    if (code == KeyCode::Char('c') || code == KeyCode::Char('q')) && ctrl {
        app.should_quit = true;
        return;
    }

    // Take the overlay out so we can match + mutate it, then put it back (unless an
    // action consumed it). This sidesteps borrow conflicts with the dispatch fns.
    let Some(overlay) = app.overlay.take() else { return };
    match overlay {
        Overlay::Picker { mut picker, theme_backup } => {
            match code {
                KeyCode::Esc => {
                    // Revert a live-preview theme/emoji on cancel.
                    if let Some(t) = theme_backup
                        && picker.kind.previews()
                    {
                        app.theme = t;
                    }
                    // overlay already taken → closed.
                }
                KeyCode::Up => {
                    picker.move_sel(-1);
                    preview_picker_selection(app, &picker);
                    app.overlay = Some(Overlay::Picker { picker, theme_backup });
                }
                KeyCode::Down => {
                    picker.move_sel(1);
                    preview_picker_selection(app, &picker);
                    app.overlay = Some(Overlay::Picker { picker, theme_backup });
                }
                KeyCode::Char(' ') if picker.kind.multi() => {
                    picker.toggle_selected();
                    app.overlay = Some(Overlay::Picker { picker, theme_backup });
                }
                KeyCode::Enter => apply_picker(app, tx_bridge, picker),
                _ => {
                    // Unhandled key → keep the overlay open.
                    app.overlay = Some(Overlay::Picker { picker, theme_backup });
                }
            }
        }
        Overlay::AskUser(mut ask) => match code {
            KeyCode::Esc => {
                // Cancel the ask: drop the card AND the pending ask (the turn stays
                // blocked until a later answer; Esc is "not now"). Surface the next
                // queued parallel ask, if any (§7).
                app.pending_ask = None;
                app.surface_next_ask();
            }
            KeyCode::Up => {
                ask.move_sel(-1);
                app.overlay = Some(Overlay::AskUser(ask));
            }
            KeyCode::Down => {
                ask.move_sel(1);
                app.overlay = Some(Overlay::AskUser(ask));
            }
            KeyCode::Char(' ') if ask.mode == components::picker::AskMode::Multi => {
                ask.toggle_selected();
                app.overlay = Some(Overlay::AskUser(ask));
            }
            KeyCode::Backspace => {
                ask.backspace();
                app.overlay = Some(Overlay::AskUser(ask));
            }
            KeyCode::Enter => apply_ask_user(app, tx_bridge, ask),
            KeyCode::Char(c) if !ctrl => {
                ask.type_char(c);
                app.overlay = Some(Overlay::AskUser(ask));
            }
            _ => app.overlay = Some(Overlay::AskUser(ask)),
        },
        // The /scheduler 3-step flow overlay (§7). Step 1 (Pick): ↑↓ move, Space
        // toggle, Enter → Confirm. Step 2 (Confirm): Enter applies + forwards the
        // start/stop set + advances to Status; Esc steps back to Pick. Step 3
        // (Status): any of Esc/Enter/q closes. Esc on Pick closes the flow.
        Overlay::Scheduler(mut sched) => {
            use components::scheduler::SchedStep;
            match (sched.step, code) {
                (SchedStep::Pick, KeyCode::Esc) => { /* closed (taken) */ }
                (SchedStep::Pick, KeyCode::Up) => {
                    sched.move_sel(-1);
                    app.overlay = Some(Overlay::Scheduler(sched));
                }
                (SchedStep::Pick, KeyCode::Down) => {
                    sched.move_sel(1);
                    app.overlay = Some(Overlay::Scheduler(sched));
                }
                (SchedStep::Pick, KeyCode::Char(' ')) => {
                    sched.toggle_selected();
                    app.overlay = Some(Overlay::Scheduler(sched));
                }
                (SchedStep::Pick, KeyCode::Enter) => {
                    sched.to_confirm();
                    app.overlay = Some(Overlay::Scheduler(sched));
                }
                (SchedStep::Confirm, KeyCode::Esc) => {
                    sched.back_to_pick();
                    app.overlay = Some(Overlay::Scheduler(sched));
                }
                (SchedStep::Confirm, KeyCode::Enter) => {
                    apply_scheduler(app, tx_bridge, &mut sched);
                    app.overlay = Some(Overlay::Scheduler(sched));
                }
                (SchedStep::Status, KeyCode::Esc)
                | (SchedStep::Status, KeyCode::Enter)
                | (SchedStep::Status, KeyCode::Char('q')) => { /* closed (taken) */ }
                _ => app.overlay = Some(Overlay::Scheduler(sched)),
            }
        }
        // The /continue searchable picker (§4). Typing edits the search box +
        // lazily re-filters via the on-disk head-window reader; Up/Down move; Enter
        // restores the selected log via the EXISTING restore path; Esc closes.
        Overlay::Continue(mut picker) => {
            use components::continue_picker::read_head_window;
            match code {
                KeyCode::Esc => { /* closed (taken) */ }
                KeyCode::Up => {
                    picker.move_sel(-1);
                    app.overlay = Some(Overlay::Continue(picker));
                }
                KeyCode::Down => {
                    picker.move_sel(1);
                    app.overlay = Some(Overlay::Continue(picker));
                }
                KeyCode::Backspace => {
                    picker.backspace(read_head_window);
                    app.overlay = Some(Overlay::Continue(picker));
                }
                KeyCode::Enter => {
                    if let Some(path) = picker.selected_path() {
                        // Restore via the existing restore path: the bridge's
                        // handle_restore loads the log into backend.history.
                        send_active(
                            app,
                            tx_bridge,
                            UiToCore::Command { name: "restore".into(), args: path },
                        );
                        app.push_notice(i18n::tf(app.lang, "continue.restoring"));
                        // overlay taken → closed.
                    } else {
                        app.overlay = Some(Overlay::Continue(picker));
                    }
                }
                KeyCode::Char(c) if !ctrl => {
                    picker.type_char(c, read_head_window);
                    app.overlay = Some(Overlay::Continue(picker));
                }
                _ => app.overlay = Some(Overlay::Continue(picker)),
            }
        }
        // The `/effort` slider (redesign_cc.md §3): ←/→ moves the `▲` marker; Enter
        // APPLIES the marked level (forward `/session.reasoning_effort=<v>` + update
        // the spinner suffix); Esc cancels (leaving the live level untouched — the
        // overlay was already `take()`n). Other keys keep the slider open.
        Overlay::EffortSlider(mut slider) => match code {
            KeyCode::Esc => { /* cancelled (taken) — live effort unchanged */ }
            KeyCode::Left => {
                slider.move_marker(-1);
                app.overlay = Some(Overlay::EffortSlider(slider));
            }
            KeyCode::Right => {
                slider.move_marker(1);
                app.overlay = Some(Overlay::EffortSlider(slider));
            }
            KeyCode::Enter => {
                // Apply the marked level; the overlay stays taken (closed).
                apply_effort(app, tx_bridge, slider.selected());
            }
            _ => app.overlay = Some(Overlay::EffortSlider(slider)),
        },
        // The `/effects demo` splash: ANY key closes it early (it also auto-reverts when
        // its timer elapses). The overlay was already `take()`n, so leaving it closed is
        // enough; we also zero the demo timer so the ambient engine stops next tick.
        app::Overlay::Effects => {
            app.effects.demo_timer = 0.0;
        }
        // Info overlays (help / status / cost / verbose / btw): any of Esc / q /
        // Enter closes; everything else keeps it open.
        other => match code {
            KeyCode::Esc | KeyCode::Enter | KeyCode::Char('q') => { /* closed (taken) */ }
            _ => app.overlay = Some(other),
        },
    }
}

/// Live-preview the picker's current selection for the preview pickers (theme /
/// emoji / language re-skin as you arrow). No-op for non-preview pickers.
fn preview_picker_selection(app: &mut AppState, picker: &components::picker::Picker) {
    use components::picker::PickerKind;
    if !picker.kind.previews() {
        return;
    }
    let Some(item) = picker.selected() else { return };
    match picker.kind {
        PickerKind::Theme => {
            if let Some(t) = theme::by_name(&strip_theme_label(&item.label)) {
                app.theme = t;
            }
        }
        PickerKind::Emoji => apply_emoji_choice(app, item.id),
        PickerKind::Language => {
            // Live-preview the language: a full repaint happens on the next frame.
            app.set_language(lang_for_picker_id(item.id));
        }
        _ => {}
    }
}

/// Map a `/language` picker row id onto a [`i18n::Lang`] (the picker rows are built
/// in `i18n::Lang::all()` order, so the id is the ordinal). PURE.
fn lang_for_picker_id(id: usize) -> i18n::Lang {
    i18n::Lang::all().get(id).copied().unwrap_or(i18n::Lang::En)
}

/// The theme NAME in a `/theme` picker row label (the label IS the bare name).
fn strip_theme_label(label: &str) -> String {
    label.trim().to_string()
}

/// Apply an `/emoji` picker choice by its row id (pet ids 0..=5, spinner 100..=102).
fn apply_emoji_choice(app: &mut AppState, id: usize) {
    use flavor::{PetStyle, SpinnerStyle};
    if id < 5 {
        app.pet_style = PetStyle::all()[id];
    } else if id == 5 {
        app.pet_style = PetStyle::Off;
    } else if (100..103).contains(&id) {
        app.spinner_style = [SpinnerStyle::Arc, SpinnerStyle::Braille, SpinnerStyle::Pulse][id - 100];
    }
}

/// Apply the picker's selection on Enter (the overlay was already taken; this
/// performs the §4 action for its [`PickerKind`] then closes the overlay).
fn apply_picker(
    app: &mut AppState,
    tx_bridge: &Sender<(u64, BridgeEvent)>,
    picker: components::picker::Picker,
) {
    use components::picker::PickerKind;
    match picker.kind {
        PickerKind::Llm => {
            if let Some(idx) = picker.selected_id() {
                // The picker id is the 0-based LLM index; protocol SwitchLlm is
                // 1-based (the `llm_picker_maps_index` contract).
                send_active(app, tx_bridge, UiToCore::SwitchLlm { n: idx as u32 + 1 });
                app.push_notice(i18n::tf(app.lang, "llm.switching"));
            }
        }
        PickerKind::Theme => {
            // Commit: keep the previewed theme (already applied as we arrowed).
            if let Some(item) = picker.selected() {
                if let Some(t) = theme::by_name(&strip_theme_label(&item.label)) {
                    app.theme = t;
                }
                app.invalidate_render_cache();
                app.push_notice(format!("{} {}", i18n::t(app.lang, "theme.set"), strip_theme_label(&item.label)));
            }
        }
        PickerKind::Emoji => {
            if let Some(id) = picker.selected_id() {
                apply_emoji_choice(app, id);
                app.push_notice(i18n::tf(app.lang, "emoji.updated"));
            }
        }
        PickerKind::Language => {
            if let Some(id) = picker.selected_id() {
                let lang = lang_for_picker_id(id);
                app.set_language(lang);
                let key = match lang {
                    i18n::Lang::Zh => "lang.set.zh",
                    i18n::Lang::En => "lang.set.en",
                };
                app.push_notice(i18n::tf(lang, key));
            }
        }
        PickerKind::Export => {
            if let Some(id) = picker.selected_id() {
                export_action(app, id);
            }
        }
        PickerKind::Rewind => {
            if let Some(id) = picker.selected_id() {
                // `id` is the full-transcript index of the chosen user turn. The
                // number of REAL turns we're dropping = the user messages at/after
                // it (what the core truncates). Compute it BEFORE truncating.
                let back = rewind_real_turns_from(&app.transcript, id);
                // Truncate the local display to the rewind point and replay the
                // remaining transcript (immediate-mode → the next frame shows it).
                app.transcript.truncate(id);
                app.invalidate_render_cache();
                // Send the NEW Rewind{n} frame so the bridge truncates
                // llmclient.backend.history by `n` real turns + replies RewindResult
                // (handle_rewind). A notice is shown now; the RewindResult notice
                // confirms the core-side truncation when it arrives.
                send_active(app, tx_bridge, UiToCore::Rewind { n: back as u32 });
                app.push_notice(format!(
                    "{} {} {}",
                    i18n::t(app.lang, "rewind.done"),
                    back,
                    i18n::t(app.lang, "rewind.turns_suffix"),
                ));
            }
        }
        PickerKind::Continue => {
            if let Some(id) = picker.selected_id() {
                // Switch into that session if it's live; else forward a /continue
                // so the core restores its log into the active session's history.
                if app.sessions.session(id as u64).is_some() {
                    app.switch_session(id as u64);
                } else {
                    send_active(app, tx_bridge, UiToCore::Command { name: "continue".into(), args: id.to_string() });
                    app.push_notice(format!("continuing session {id}…"));
                }
            }
        }
        PickerKind::Scheduler => {
            // The `/scheduler` flow now uses its own multi-step Scheduler overlay
            // (see `open_ui_command` + `apply_scheduler`); this generic-picker arm
            // is unreachable in normal use. Kept as a defensive no-op so a stray
            // Scheduler-kind generic picker can't panic.
            app.push_notice(i18n::tf(app.lang, "scheduler.none_selected"));
        }
    }
}

/// Apply the `/scheduler` confirm step (step 2 → step 3): forward the start/stop
/// diff to the core as a `Command{name:"scheduler"}` and advance the overlay to the
/// status view. The frame `args` is `"start a,b stop c,d"` (ids), so the core can
/// reconcile the cron set; with no changes we still advance (the status view then
/// reflects the unchanged running set). The actual start/stop SETS come from the
/// pure [`components::scheduler::Scheduler`] diff.
fn apply_scheduler(
    app: &mut AppState,
    tx_bridge: &Sender<(u64, BridgeEvent)>,
    sched: &mut components::scheduler::Scheduler,
) {
    let to_start = sched.to_start();
    let to_stop = sched.to_stop();
    // Advance the overlay to the status step (freezes the desired set).
    sched.apply();
    if to_start.is_empty() && to_stop.is_empty() {
        app.push_notice(i18n::tf(app.lang, "scheduler.no_changes"));
        return;
    }
    let join = |v: &[usize]| v.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",");
    let mut parts: Vec<String> = Vec::new();
    if !to_start.is_empty() {
        parts.push(format!("start {}", join(&to_start)));
    }
    if !to_stop.is_empty() {
        parts.push(format!("stop {}", join(&to_stop)));
    }
    let arg = parts.join(" ");
    send_active(app, tx_bridge, UiToCore::Command { name: "scheduler".into(), args: arg });
    app.push_notice(i18n::tf(app.lang, "scheduler.applied"));
}

/// The number of REAL (user) turns dropped by rewinding the transcript to
/// full-index `truncate_at` (the `/rewind` picker's row id). It's the count of
/// USER messages at or after `truncate_at` — i.e. the turns that `transcript
/// .truncate(truncate_at)` removes — clamped to ≥1 (picking a turn always drops at
/// least that turn). PURE — the `rewind_truncation_count` deliverable pins it.
pub fn rewind_real_turns_from(transcript: &[app::Block], truncate_at: usize) -> usize {
    transcript
        .iter()
        .skip(truncate_at)
        .filter(|b| matches!(b.role, app::Role::User))
        .count()
        .max(1)
}

/// Apply the ask_user card on Enter: resolve the answer text and send it back as an
/// `Answer` frame, clearing the pending ask + the overlay.
fn apply_ask_user(
    app: &mut AppState,
    tx_bridge: &Sender<(u64, BridgeEvent)>,
    ask: components::picker::AskUserPicker,
) {
    match ask.resolve_answer() {
        Some(text) => {
            let frame = UiToCore::Answer {
                ask_id: ask.ask_id.clone(),
                option_id: None,
                text: Some(text.clone()),
            };
            send_active(app, tx_bridge, frame);
            app.pending_ask = None;
            app.push_user(text);
            // Surface the next queued parallel ask, if any (§7).
            app.surface_next_ask();
        }
        None => {
            // Nothing to submit (empty free-text) → keep the card open.
            app.overlay = Some(app::Overlay::AskUser(ask));
        }
    }
}

/// Create a new session from the dashboard, optionally seeded with a prompt that
/// is submitted to its (lazy-spawned) child, then switch into it. Clears the
/// bottom input.
fn dashboard_new_session(app: &mut AppState, tx_bridge: &Sender<(u64, BridgeEvent)>, seed: String) {
    app.snapshot_active_into_map();
    app.sessions.new_session_input.clear();
    let new_id = app.sessions.new_session(None);
    // Switch the live fields onto the new (empty) session.
    app.load_active_session_after_structural_change(new_id);
    if !seed.trim().is_empty() {
        // Echo the seed as the user's first message + submit to the child.
        app.push_user(seed.clone());
        let expanded = app.expand_at_paths(&seed);
        let frame = if let Some(rest) = expanded.strip_prefix('/') {
            let mut parts = rest.splitn(2, ' ');
            UiToCore::Command {
                name: parts.next().unwrap_or("").to_string(),
                args: parts.next().unwrap_or("").to_string(),
            }
        } else {
            UiToCore::Submit { text: expanded, images: None }
        };
        send_active(app, tx_bridge, frame);
    }
}

/// Delete the dashboard-selected session (KEEPING its on-disk log). If it was the
/// active session, re-sync the live fields to the fallback so a later Esc shows a
/// coherent cockpit.
fn dashboard_delete_selected(app: &mut AppState) {
    let Some(id) = app.sessions.selected_session_id() else {
        return;
    };
    let was_active = id == app.sessions.active;
    app.sessions.delete(id);
    if was_active {
        let new_active = app.sessions.active;
        app.load_active_fields_after_drop(new_active);
    }
}

/// Quick-reply from the dashboard: submit `text` to session `id`'s child WITHOUT
/// switching the active view away (the reply streams into that session's record;
/// the dashboard row preview updates live). Clears the bottom input.
fn dashboard_quick_reply(
    app: &mut AppState,
    tx_bridge: &Sender<(u64, BridgeEvent)>,
    id: u64,
    text: String,
) {
    app.sessions.new_session_input.clear();
    // Echo the user message into that session's transcript (so its preview shows
    // the prompt), then submit to its child.
    if let Some(s) = app.sessions.session_mut(id) {
        s.push_user(text.clone());
    }
    // If the target is the active session, mirror the echo into the live fields.
    if id == app.sessions.active {
        app.push_user(text.clone());
    }
    let frame = UiToCore::Submit { text, images: None };
    app.sessions.send_to(id, frame, tx_bridge);
}

/// If a completion dropdown is active, handle Tab/Enter (complete) and Up/Down
/// (move highlight) here, returning `true` if the key was consumed. The slash
/// palette wins when both could match (a `/word` has no `@` query).
fn try_complete_dropdown(app: &mut AppState, code: KeyCode, shift: bool) -> bool {
    use commands::registry;
    let text = app.composer.text().to_string();
    let matches = registry::palette_matches(&text);
    if registry::palette_visible(&text, &matches) {
        let n = matches.len();
        // Keep the highlight in range (the match set shrinks as the partial grows).
        let sel = app.palette_sel.min(n.saturating_sub(1));
        match code {
            // ↑/↓ move the highlight (clamped, no wrap) — the §4 "arrow keys to
            // pick" the tui_v3 tip advertises. Previously the palette ignored these
            // (only the @-file picker had Up/Down), so the highlight was stuck on
            // the top row.
            KeyCode::Up => {
                app.palette_sel = registry::move_sel(sel, -1, n);
                return true;
            }
            KeyCode::Down => {
                app.palette_sel = registry::move_sel(sel, 1, n);
                return true;
            }
            // Tab/Enter complete the HIGHLIGHTED match into `/name ` (a trailing
            // space hides the palette; a 2nd Enter runs it). Honors the selection.
            KeyCode::Tab => {
                if let Some(cmd) = matches.get(sel) {
                    let completed = registry::complete_to(cmd);
                    app.composer.set_buffer(completed.clone(), completed.len());
                }
                app.palette_sel = 0;
                return true;
            }
            KeyCode::Enter if !shift => {
                if let Some(cmd) = matches.get(sel) {
                    let completed = registry::complete_to(cmd);
                    app.composer.set_buffer(completed.clone(), completed.len());
                }
                app.palette_sel = 0;
                return true;
            }
            _ => return false,
        }
    }
    // @ file picker.
    if let Some(q) = app.composer.at_query() {
        let files = app.list_project_files();
        let ranked = input::file_expand::rank_files(&q.partial, &files);
        if !ranked.is_empty() {
            match code {
                KeyCode::Down => {
                    app.composer.file_sel = app.composer.file_sel.saturating_add(1) % ranked.len();
                    return true;
                }
                KeyCode::Up => {
                    let n = ranked.len();
                    app.composer.file_sel = (app.composer.file_sel + n - 1) % n;
                    return true;
                }
                KeyCode::Tab | KeyCode::Enter if !shift => {
                    let idx = app.composer.file_sel % ranked.len();
                    let picked = ranked[idx].clone();
                    app.composer.complete_file(&picked);
                    return true;
                }
                _ => return false,
            }
        }
    }
    false
}

/// Act on a [`input::ComposerAction`] produced by Enter: submit to the ACTIVE
/// session's bridge (spawning it lazily, §6), run a `!shell` line, etc.
fn dispatch_action(
    app: &mut AppState,
    tx_bridge: &Sender<(u64, BridgeEvent)>,
    action: input::ComposerAction,
) {
    use input::ComposerAction;
    match action {
        ComposerAction::None | ComposerAction::Changed | ComposerAction::Redraw
        | ComposerAction::ToggleFold | ComposerAction::Escape => {}
        ComposerAction::Submit { text } => {
            let expanded = app.expand_at_paths(&text);
            // A leading `/` is a slash command — classify + route it through the
            // registry (every §4 name resolves; unknowns get a did-you-mean). A
            // plain message echoes into the transcript + submits to the bridge.
            if expanded.trim_start().starts_with('/') {
                dispatch_slash(app, tx_bridge, &expanded);
            } else {
                app.push_user(text.clone());
                if !send_active(app, tx_bridge, UiToCore::Submit { text: expanded, images: None }) {
                    app.push_notice(i18n::tf(app.lang, "notice.bridge.not_connected"));
                }
            }
        }
        ComposerAction::Shell { cmd } => run_shell_line(app, tx_bridge, &cmd),
    }
}

/// Send a frame to the ACTIVE session's bridge child (lazy-spawning it through the
/// tagged multiplexer). Returns `false` if the child is dead/unspawnable.
fn send_active(app: &mut AppState, tx_bridge: &Sender<(u64, BridgeEvent)>, frame: UiToCore) -> bool {
    let active = app.sessions.active;
    app.sessions.send_to(active, frame, tx_bridge)
}

/// Route a submitted `/command args` line (checklist §4). Classifies it via the
/// registry into a [`commands::SlashOutcome`] and performs the effect: open a
/// dedicated UI overlay, core-forward a `Command` frame, run an in-app action, or
/// surface a "did you mean /x?" breadcrumb for an unknown command.
fn dispatch_slash(app: &mut AppState, tx_bridge: &Sender<(u64, BridgeEvent)>, line: &str) {
    use commands::SlashOutcome;
    match commands::classify_slash(line) {
        SlashOutcome::OpenUi { name, args } => open_ui_command(app, tx_bridge, &name, &args),
        SlashOutcome::Forward { name, args } => {
            // Echo the user's command in the transcript, then forward it as a
            // Command frame (the GA core intercepts the leading-slash itself).
            app.push_user(format!("/{name}{}", if args.is_empty() { String::new() } else { format!(" {args}") }));
            // §4: /goal /hive /conductor are "fwd + a /workflows tile" — forwarding
            // kicks off the orchestration AND the panel shows its live progress. So
            // after forwarding one of these, open the /workflows panel so the user
            // watches the tree the watcher will populate (the others just forward).
            let opens_panel = matches!(name.as_str(), "goal" | "hive" | "conductor");
            if !send_active(app, tx_bridge, UiToCore::Command { name, args }) {
                app.push_notice(i18n::tf(app.lang, "notice.bridge.not_connected"));
            }
            if opens_panel {
                app.open_workflows();
            }
        }
        SlashOutcome::App { name, args } => app_command(app, tx_bridge, &name, &args),
        SlashOutcome::Unknown { typed, suggestion } => {
            let unknown = i18n::t(app.lang, "cmd.unknown");
            let msg = match suggestion {
                Some(s) => format!(
                    "{unknown} /{typed} — {} /{s}?",
                    i18n::t(app.lang, "cmd.did_you_mean")
                ),
                None => format!("{unknown} /{typed} {}", i18n::t(app.lang, "cmd.type_help")),
            };
            app.push_notice(msg);
        }
    }
}

/// Open the dedicated UI for a §4 **UI** command (the picker/overlay surface).
fn open_ui_command(app: &mut AppState, tx_bridge: &Sender<(u64, BridgeEvent)>, name: &str, args: &str) {
    use components::picker::{PickItem, Picker, PickerKind};
    match name {
        "help" => app.open_overlay(app::Overlay::Help),
        "status" | "sessions" => app.open_overlay(app::Overlay::Status),
        "switch" => app.open_dashboard(),
        // The /workflows panel (§7): the live conductor / hive / goal tree. Opens
        // the full-screen panel + (lazily) starts the singleton watcher.
        "workflows" => app.open_workflows(),
        "llm" => {
            if !args.is_empty() {
                // `/llm <n>` — switch directly (n is the 0-based index the picker
                // shows; protocol SwitchLlm is 1-based).
                if let Ok(idx) = args.trim().parse::<u32>() {
                    send_active(app, tx_bridge, UiToCore::SwitchLlm { n: idx + 1 });
                    app.push_notice(i18n::tf(app.lang, "llm.switching"));
                    return;
                }
            }
            // Open the picker with a "querying…" placeholder, then ask the bridge
            // for the model list; the LlmList frame fills the rows in place (N3).
            let placeholder = vec![PickItem::new(0, i18n::t(app.lang, "llm.querying"))];
            app.open_picker(Picker::new(PickerKind::Llm, placeholder), None);
            send_active(app, tx_bridge, UiToCore::ListLlms);
        }
        "theme" => {
            // Live-preview theme picker (commit/revert). Seed on the active theme.
            let items: Vec<PickItem> = theme::all_names()
                .iter()
                .enumerate()
                .map(|(i, n)| PickItem::new(i, *n).current(*n == app.theme.name))
                .collect();
            // If `/theme <name>` was given, apply directly.
            if !args.is_empty() {
                if let Some(t) = theme::by_name(args) {
                    app.theme = t;
                    app.invalidate_render_cache();
                    app.push_notice(format!("{} {}", i18n::t(app.lang, "theme.set"), args.trim()));
                } else {
                    app.push_notice(format!("{} ({})", i18n::t(app.lang, "theme.unknown"), args.trim()));
                }
                return;
            }
            let backup = app.theme.clone();
            let mut picker = Picker::new(PickerKind::Theme, items);
            // Seed the highlight on the active theme by name (belt-and-suspenders
            // over the `current` marker, in case the name table drifts).
            if let Some(idx) = theme::index_of(app.theme.name) {
                picker.sel = idx;
            }
            app.open_picker(picker, Some(backup));
        }
        "emoji" => {
            // Pet/spinner style picker (preview on move). Rows = the 5 pet styles +
            // Off, then the 3 spinner styles, all mapped onto a stable id.
            let items = emoji_picker_items(app);
            let backup = app.theme.clone(); // emoji previews don't touch theme; backup harmless.
            app.open_picker(Picker::new(PickerKind::Emoji, items), Some(backup));
        }
        "effort" => {
            // `/effort <level>` (redesign_cc.md §3): a direct set with NO slider when
            // a level word is given (e.g. `/effort high`); a bare `/effort` opens the
            // slider. Either way APPLY = forward `/session.reasoning_effort=<backend>`
            // (max→xhigh) via the existing slash-forward path so GA hot-reloads it
            // (setattr(backend, "reasoning_effort", v)), and remember it for the
            // spinner suffix. An unrecognized word falls back to opening the slider.
            match app::effort::ReasoningEffort::parse(args) {
                Some(level) if !args.trim().is_empty() => apply_effort(app, tx_bridge, level),
                _ => app.open_effort_slider(),
            }
        }
        "language" => {
            // `/language <code>` shortcut: switch directly + full repaint.
            if !args.trim().is_empty() {
                if let Some(lang) = i18n::Lang::from_code(args) {
                    app.set_language(lang);
                    let key = if lang == i18n::Lang::Zh { "lang.set.zh" } else { "lang.set.en" };
                    app.push_notice(i18n::tf(lang, key));
                } else {
                    app.push_notice(format!("unknown language '{}' (try /language)", args.trim()));
                }
                return;
            }
            // The rows are built in `Lang::all()` order so the row id is the ordinal
            // `lang_for_picker_id` decodes; each shows its endonym + a `●` for active.
            let items: Vec<PickItem> = i18n::Lang::all()
                .iter()
                .enumerate()
                .map(|(i, l)| {
                    PickItem::new(i, l.endonym())
                        .with_detail(l.code())
                        .current(*l == app.lang)
                })
                .collect();
            app.open_picker(Picker::new(PickerKind::Language, items), None);
        }
        "export" => {
            let items = vec![
                PickItem::new(0, i18n::t(app.lang, "export.clip")).with_detail(i18n::t(app.lang, "export.clip.detail")),
                PickItem::new(1, i18n::t(app.lang, "export.all")).with_detail(i18n::t(app.lang, "export.all.detail")),
                PickItem::new(2, i18n::t(app.lang, "export.file")).with_detail(i18n::t(app.lang, "export.file.detail")),
            ];
            // `/export clip|all|file` shortcut: act directly.
            match args.trim() {
                "clip" => return export_action(app, 0),
                "all" => return export_action(app, 1),
                "file" => return export_action(app, 2),
                _ => {}
            }
            app.open_picker(Picker::new(PickerKind::Export, items), None);
        }
        "verbose" | "tools" | "trace" => app.open_overlay(app::Overlay::Verbose),
        "rewind" => {
            let items = rewind_picker_items(app);
            if items.is_empty() {
                app.push_notice(i18n::tf(app.lang, "rewind.empty"));
                return;
            }
            app.open_picker(Picker::new(PickerKind::Rewind, items), None);
        }
        "continue" => {
            // Searchable picker over the model_responses_*.txt logs (content-grep +
            // lazy load), restoring the selected log via the existing restore path.
            let sessions = components::continue_picker::list_sessions(&app.repo_root, None);
            if sessions.is_empty() {
                app.push_notice(i18n::tf(app.lang, "continue.empty"));
                return;
            }
            app.open_overlay(app::Overlay::Continue(
                components::continue_picker::ContinuePicker::new(sessions),
            ));
        }
        "scheduler" => {
            // The 3-step flow (§7): step 1 multi-pick reflect tasks with the
            // currently-running ones PRE-TICKED, step 2 confirm the start/stop diff,
            // step 3 apply + show cron status. The whole flow is one Scheduler
            // overlay whose step advances on Enter. Degrades gracefully to a default
            // task set when no live cron data is available.
            let tasks = components::scheduler::default_tasks();
            app.open_overlay(app::Overlay::Scheduler(components::scheduler::Scheduler::new(tasks)));
        }
        "btw" => {
            // Side-question card (§7): show the question with `querying…`, then fire
            // a BACKGROUND side-ask to the bridge (BtwAsk) tied to a fresh ask_id.
            // The bridge answers on a worker thread (the main task keeps running)
            // and replies BtwAnswer, which the reducer routes into THIS card — never
            // into the transcript. Esc dismisses the card (no history pollution).
            let q = args.trim().to_string();
            if q.is_empty() {
                app.push_notice(i18n::tf(app.lang, "btw.usage"));
                return;
            }
            let ask_id = new_ask_id();
            app.open_overlay(app::Overlay::Btw {
                ask_id: ask_id.clone(),
                question: q.clone(),
                answer: None,
            });
            send_active(app, tx_bridge, UiToCore::BtwAsk { ask_id, text: q });
        }
        _ => app.push_notice(format!("/{name} {}", i18n::t(app.lang, "cmd.not_wired"))),
    }
}

/// APPLY a chosen reasoning-effort level (redesign_cc.md §3): remember it for the
/// spinner suffix AND forward `/session.reasoning_effort=<backend>` (max→xhigh) to
/// the active bridge via the existing slash-forward `Command` path so the GA core
/// hot-reloads it live (`setattr(backend, "reasoning_effort", v)`, takes effect next
/// turn — no restart). Shared by `/effort <level>` (direct) and the slider's Enter.
fn apply_effort(
    app: &mut AppState,
    tx_bridge: &Sender<(u64, BridgeEvent)>,
    level: app::effort::ReasoningEffort,
) {
    app.set_reasoning_effort(level);
    // Forward as Command{name:"session.reasoning_effort=<v>", args:""} — the bridge
    // reconstructs "/session.reasoning_effort=<v>" and the core intercepts it.
    let sent = send_active(app, tx_bridge, effort_command_frame(level));
    // Confirm in the transcript (mirrors CC: a "thinking <level>" confirmation line).
    // Show the slider LABEL (so `max` reads "max"); note the backend value if it
    // differs (max→xhigh) so the user sees what the backend actually got.
    let note = if level.label() == level.backend_value() {
        format!("effort · {} (next turn)", level.label())
    } else {
        format!("effort · {} → {} (next turn)", level.label(), level.backend_value())
    };
    app.push_notice(note);
    if !sent {
        app.push_notice(i18n::tf(app.lang, "notice.bridge.not_connected"));
    }
}

/// Build the bridge frame that forwards a `/effort` level (redesign_cc.md §3): a
/// `Command{name:"session.reasoning_effort=<backend>", args:""}` — the bridge
/// reconstructs `"/" + name` → `/session.reasoning_effort=<backend>` and the GA core
/// hot-reloads it via `setattr(backend, "reasoning_effort", v)`. PURE (the routing
/// contract the `effort_forwards_session_command` test pins, without spawning a
/// child).
fn effort_command_frame(level: app::effort::ReasoningEffort) -> UiToCore {
    UiToCore::Command { name: level.command_name(), args: String::new() }
}

/// A fresh, short, unique-enough id for a `/btw` side-question card (ties the
/// `BtwAsk` request to the `BtwAnswer` reply). Derived from a monotonic process
/// clock + a counter so it is collision-free within a session without a uuid dep.
fn new_ask_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    format!("btw-{t:x}-{n:x}")
}

/// Run a §4 **app** command (handled entirely in-app).
fn app_command(app: &mut AppState, tx_bridge: &Sender<(u64, BridgeEvent)>, name: &str, args: &str) {
    match name {
        "quit" | "exit" => app.should_quit = true,
        "clear" => {
            if app.busy {
                app.push_notice(i18n::tf(app.lang, "cmd.clear.busy"));
            } else {
                app.transcript.clear();
                app.invalidate_render_cache();
            }
        }
        "stop" | "abort" => {
            send_active(app, tx_bridge, UiToCore::Abort { mid: None });
            app.busy = false;
            app.push_notice(i18n::tf(app.lang, "cmd.aborted"));
        }
        "cost" => app.open_overlay(app::Overlay::Cost),
        "mouse" => {
            // Toggle (or set `on`/`off`) terminal mouse capture. OFF enables the
            // terminal's native drag-select so the user can copy transcript/input
            // text inline (the discoverable alias for the Ctrl+Shift+M chord).
            let on = match args.trim().to_ascii_lowercase().as_str() {
                "on" => true,
                "off" => false,
                _ => !app.mouse_capture,
            };
            app.mouse_capture = on;
            set_mouse_capture(on);
            let k = if on { "mouse.on" } else { "mouse.off" };
            app.push_notice(i18n::tf(app.lang, k));
        }
        "new" => {
            let seed = args.trim();
            cockpit_new_session(app);
            if !seed.is_empty() {
                app.sessions.rename(app.sessions.active, seed.to_string());
            }
        }
        "close" => cockpit_drop_active(app),
        "rename" => {
            let new = args.trim();
            if new.is_empty() {
                app.push_notice(i18n::tf(app.lang, "cmd.rename.usage"));
            } else {
                let id = app.sessions.active;
                app.sessions.rename(id, new.to_string());
                app.push_notice(format!("{} {new}", i18n::t(app.lang, "cmd.renamed")));
            }
        }
        "branch" => {
            let draft = app.composer.text().to_string();
            app.snapshot_active_into_map();
            let new_id = app.sessions.branch(draft);
            app.load_active_session_after_structural_change(new_id);
        }
        "restore" => {
            // Restore the last model_responses log into the active session's
            // history (the bridge handles the Command{name:"restore"}).
            send_active(app, tx_bridge, UiToCore::Command { name: "restore".into(), args: args.to_string() });
            app.push_notice(i18n::tf(app.lang, "cmd.restoring"));
        }
        "reload-keys" => {
            send_active(app, tx_bridge, UiToCore::Command { name: "reload-keys".into(), args: String::new() });
            app.push_notice(i18n::tf(app.lang, "cmd.reloading_keys"));
        }
        "effects" => {
            // `/effects [demo|off|subtle|full]` (§9). `demo` opens a transient splash
            // overlay showing every effect; off/subtle/full set the persistent mode.
            use effects::EffectMode;
            let arg = args.trim().to_ascii_lowercase();
            let token = if arg.is_empty() { "off" } else { arg.as_str() };
            match token {
                "demo" => {
                    app.start_effects_demo();
                    app.push_notice(format!("{}: demo", i18n::t(app.lang, "cmd.effects")));
                }
                other => match EffectMode::parse(other) {
                    Some(mode) => {
                        app.set_effects_mode(mode);
                        app.push_notice(format!(
                            "{}: {}",
                            i18n::t(app.lang, "cmd.effects"),
                            mode.label()
                        ));
                    }
                    None => app.push_notice(format!(
                        "{} '{other}' (demo|off|subtle|full)",
                        i18n::t(app.lang, "cmd.effects")
                    )),
                },
            }
        }
        _ => app.push_notice(format!("/{name} {}", i18n::t(app.lang, "cmd.not_handled"))),
    }
}

/// Build the `/emoji` picker rows: the 5 pet styles + Off (ids 0..=5) then the 3
/// spinner styles (ids 100..=102), each marked current if active. PURE-ish.
fn emoji_picker_items(app: &AppState) -> Vec<components::picker::PickItem> {
    use components::picker::PickItem;
    use flavor::{PetStyle, SpinnerStyle};
    let pet = i18n::t(app.lang, "emoji.pet");
    let spinner = i18n::t(app.lang, "emoji.spinner");
    let off = i18n::t(app.lang, "emoji.off");
    let mut items: Vec<PickItem> = Vec::new();
    for (i, style) in PetStyle::all().iter().enumerate() {
        items.push(
            PickItem::new(i, format!("{pet} · {}", style.name()))
                .with_detail(flavor::pet_face(*style, 0, 0).to_string())
                .current(*style == app.pet_style),
        );
    }
    items.push(PickItem::new(5, format!("{pet} · {off}")).current(app.pet_style == PetStyle::Off));
    for (i, style) in [SpinnerStyle::Arc, SpinnerStyle::Braille, SpinnerStyle::Pulse]
        .iter()
        .enumerate()
    {
        items.push(
            PickItem::new(100 + i, format!("{spinner} · {}", style.name()))
                .with_detail(style.glyph(0).to_string())
                .current(*style == app.spinner_style),
        );
    }
    items
}

/// Build the `/rewind` picker rows: the last ~20 USER turns, newest first. Each
/// row's `id` is the FULL-transcript index of that user message (so truncating the
/// transcript at `id` removes that turn and everything after — the rewind point);
/// the label shows the turn ordinal + how many turns back. PURE-ish.
fn rewind_picker_items(app: &AppState) -> Vec<components::picker::PickItem> {
    use components::picker::PickItem;
    // (full_transcript_index, user-turn-ordinal, block) for each user message.
    let user_turns: Vec<(usize, usize, &app::Block)> = app
        .transcript
        .iter()
        .enumerate()
        .filter(|(_, b)| matches!(b.role, app::Role::User))
        .enumerate()
        .map(|(ordinal, (full_idx, b))| (full_idx, ordinal, b))
        .collect();
    let total = user_turns.len();
    user_turns
        .iter()
        .rev()
        .take(20)
        .map(|(full_idx, ordinal, b)| {
            let back = total - ordinal; // turns from the end (1 = most recent).
            let preview = b.source.lines().next().unwrap_or("").chars().take(60).collect::<String>();
            // id = the full-transcript index (the truncate point).
            PickItem::new(*full_idx, format!("turn {} (−{back})", ordinal + 1)).with_detail(preview)
        })
        .collect()
}

/// Perform an `/export` action by row id: 0 = clip (last reply via OSC52), 1 = all
/// (whole transcript to clipboard), 2 = file (last reply to a cwd file).
fn export_action(app: &mut AppState, id: usize) {
    match id {
        0 => {
            if let Some(src) = app.last_assistant_source() {
                let src = src.to_string();
                let label = i18n::t(app.lang, "copy.label.reply");
                let res = copy_text(&src);
                notice_copy(app, &res, label);
            } else {
                app.push_notice(i18n::tf(app.lang, "export.none"));
            }
        }
        1 => {
            let all = app.transcript_source();
            let label = i18n::t(app.lang, "copy.label.transcript");
            let res = copy_text(&all);
            notice_copy(app, &res, label);
        }
        2 => {
            let Some(src) = app.last_assistant_source().map(str::to_string) else {
                app.push_notice(i18n::tf(app.lang, "export.none"));
                return;
            };
            let fname = format!("tui_v4_export_{}.md", std::process::id());
            let path = app.repo_root.join(&fname);
            match std::fs::write(&path, &src) {
                Ok(()) => app.push_notice(format!("{} {}", i18n::t(app.lang, "export.wrote"), path.display())),
                Err(e) => app.push_notice(format!("{}: {e}", i18n::t(app.lang, "export.failed"))),
            }
        }
        _ => {}
    }
}

/// Ctrl+N — create a new session + switch to it (the leaving draft is stashed).
fn cockpit_new_session(app: &mut AppState) {
    app.snapshot_active_into_map();
    let draft = app.composer.text().to_string();
    // Stash the leaving draft on the current active session before we move.
    let leaving = app.sessions.active;
    if let Some(s) = app.sessions.session_mut(leaving) {
        s.input_stash = draft;
    }
    let new_id = app.sessions.new_session(None);
    app.load_active_session_after_structural_change(new_id);
}

/// Ctrl+W / Ctrl+D — drop (close) the active session, keeping its log; focus the
/// neighbour the map fell back to.
fn cockpit_drop_active(app: &mut AppState) {
    let active = app.sessions.active;
    app.sessions.delete(active);
    let new_active = app.sessions.active;
    // Loading the fallback session's stored state (a reset blank for the
    // last-session case, else the neighbour's transcript).
    app.load_active_fields_after_drop(new_active);
}

/// Ctrl+Up / Ctrl+Down — cycle the active session, stashing/restoring drafts.
fn cockpit_cycle(app: &mut AppState, delta: isize) {
    app.snapshot_active_into_map();
    let draft = app.composer.text().to_string();
    let incoming = app.sessions.cycle(delta, draft);
    let new_active = app.sessions.active;
    app.load_active_fields_from_public(new_active);
    app.composer.set_buffer(incoming.clone(), incoming.len());
}

/// The 3-STAGE Ctrl+C (§8). Run the PURE decider over the live cockpit state
/// (selection? + turn running?) with the injected `now_ms`, store the new arm
/// state, then perform the chosen effect: copy the selection (OSC-52), abort the
/// running turn, arm quit + surface the "press Ctrl+C again" hint, or quit.
fn cockpit_ctrl_c(app: &mut AppState, tx_bridge: &Sender<(u64, BridgeEvent)>, now_ms: u64) {
    use input::keychord::{ctrl_c, CtrlCAction};
    let has_selection = app.composer.selection().is_some();
    let turn_running = app.busy;
    let (action, next) = ctrl_c(app.chord, now_ms, has_selection, turn_running);
    app.chord = next;
    match action {
        CtrlCAction::CopySelection => {
            // Copy the composer's selected text via the clean logical-source path
            // (OSC-52 → native, P2). The selection is kept (copy doesn't mutate it),
            // matching mainstream "Ctrl+C copies, doesn't clear" behavior.
            if let Some(sel) = app.composer.selected_text() {
                let text = sel.to_string();
                let label = i18n::t(app.lang, "copy.label.selection");
                let res = copy_text(&text);
                notice_copy(app, &res, label);
            }
        }
        CtrlCAction::AbortTurn => {
            // Mirror `/stop`: tell the active child to abort, flip busy off, notice.
            send_active(app, tx_bridge, UiToCore::Abort { mid: None });
            app.busy = false;
            app.push_notice(i18n::tf(app.lang, "cmd.aborted"));
        }
        CtrlCAction::ArmQuit => {
            // The hint ("press Ctrl+C again to quit") is surfaced on the BOTTOM hint
            // line while the arm is live (`chord.ctrl_c_hint_active(now_ms)` in
            // `render_hints`) — NOT as a transcript notice, so it doesn't scroll away
            // or pollute history. A 2nd Ctrl+C within 2s quits; else the arm expires.
        }
        CtrlCAction::Quit => app.should_quit = true,
    }
}

/// The Esc / Esc-Esc chord (§8). Run the PURE decider with the injected `now_ms`,
/// store the new double-tap window, then either open the `/rewind` picker (2nd Esc
/// inside 0.8s) or perform the universal back (single Esc) — which NEVER exits.
fn cockpit_esc(app: &mut AppState, tx_bridge: &Sender<(u64, BridgeEvent)>, now_ms: u64, width: u16) {
    use input::keychord::{esc, EscAction};
    let (action, next) = esc(app.chord, now_ms);
    app.chord = next;
    match action {
        EscAction::Rewind => {
            // Open the rewind picker (same entry the `/rewind` command uses; it
            // shows a "nothing to rewind" notice when there are no turns yet).
            open_ui_command(app, tx_bridge, "rewind", "");
        }
        EscAction::Back => cockpit_universal_back(app, width),
    }
}

/// The cockpit's universal-BACK (a single Esc): clear a pending ask, else collapse
/// a composer selection, else stash the draft so work isn't lost. NEVER exits —
/// the §8 "Esc universal back … never exits" contract. PURE-ish (in-memory only).
fn cockpit_universal_back(app: &mut AppState, width: u16) {
    use input::Nav;
    if app.pending_ask.take().is_none() {
        if app.composer.selection().is_some() {
            app.composer.nav(Nav::Right, false, width); // collapse selection
        } else if !app.composer.is_empty() {
            // Stash the draft so Esc doesn't lose work.
            app.composer.stash_or_restore();
        }
    }
}

/// Run a `!cmd` host-shell line: execute (30s timeout), echo the output into the
/// transcript as a SYSTEM block, and seed the agent's context via a
/// `Command{name:"shell"}` frame (ga_bridge.py stashes it into `_intervene` —
/// WITHOUT spending a turn). N1: a spawn failure still surfaces as a system line.
fn run_shell_line(app: &mut AppState, tx_bridge: &Sender<(u64, BridgeEvent)>, cmd: &str) {
    use commands::{format_shell_block, format_shell_note, run_shell};
    let cwd = app.repo_root.clone();
    let result = run_shell(cmd, &cwd);
    // Echo into the transcript (what the user sees).
    app.push_system(format_shell_block(&result));
    // Seed the agent context (what the model gets) — no turn spent. Routed to the
    // ACTIVE session's child (lazy-spawned).
    let note = format_shell_note(&result);
    send_active(
        app,
        tx_bridge,
        UiToCore::Command {
            name: "shell".to_string(),
            args: note,
        },
    );
}

/// Copy `text` via the clean logical-source path (P2): OSC 52 → native. RETURNS
/// the [`CopyResult`](render::copy::CopyResult) so callers can surface a visible
/// "Copied N bytes" notice — the result used to be swallowed, so a copy gave no
/// feedback at all (the root of the "can't copy" complaint).
fn copy_text(text: &str) -> render::copy::CopyResult {
    use render::copy::copy_to_clipboard;
    use render::{CopyCaps, Selection};
    use std::io::IsTerminal;
    let has_tty = std::io::stdout().is_terminal();
    copy_to_clipboard(text, Selection::Clipboard, CopyCaps::from_env(), has_tty)
}

/// Push a localized "Copied N bytes" (or an HONEST failure) notice for a finished
/// copy. `label` is an already-localized noun for what was copied ("selection",
/// "last reply", …). Every copy outcome is surfaced so the user always sees it
/// worked (N1: never silent — the missing feedback was the actual bug).
fn notice_copy(app: &mut AppState, res: &render::copy::CopyResult, label: &str) {
    use render::copy::CopyReason;
    if res.ok {
        app.push_notice(format!(
            "{} {label} · {} {}",
            i18n::t(app.lang, "copy.ok"),
            res.bytes,
            i18n::t(app.lang, "unit.bytes"),
        ));
    } else {
        let why = match res.reason {
            Some(CopyReason::TooLarge) => i18n::t(app.lang, "copy.fail.too_large"),
            Some(CopyReason::NoNativeRemote) | Some(CopyReason::NoTty) => {
                i18n::t(app.lang, "copy.fail.no_tty")
            }
            _ => i18n::t(app.lang, "copy.fail.generic"),
        };
        app.push_notice(format!("{} {label}: {why}", i18n::t(app.lang, "copy.fail")));
    }
}

/// Toggle terminal mouse capture at runtime (Ctrl+Shift+M / `/mouse`). ON = wheel
/// scroll + click-to-dashboard; OFF lets the terminal's OWN drag-select work so
/// the user can select + copy transcript/input text natively (the portable,
/// Windows-safe answer to inline copy — Codex's model). Best-effort.
fn set_mouse_capture(on: bool) {
    let mut out = io::stdout();
    let _ = if on {
        execute!(out, EnableMouseCapture)
    } else {
        execute!(out, DisableMouseCapture)
    };
}

/// Read the native clipboard for Ctrl+V (best-effort; `None` on any error).
fn read_clipboard() -> Option<String> {
    arboard::Clipboard::new().ok()?.get_text().ok()
}

// ---------------------------------------------------------------------------
// Threads feeding the unified channel.
// ---------------------------------------------------------------------------

/// Poll crossterm for input on a dedicated thread and emit a `Tick` every 0.1s.
/// This keeps `terminal.draw` + `rx.recv` on the main thread non-blocking.
fn spawn_input_thread(tx: Sender<AppEvent>) {
    std::thread::spawn(move || {
        let tick = Duration::from_millis(100);
        let mut last_tick = Instant::now();
        loop {
            // Wait up to the remaining tick budget for an input event.
            let timeout = tick
                .checked_sub(last_tick.elapsed())
                .unwrap_or(Duration::ZERO);
            match event::poll(timeout) {
                Ok(true) => match event::read() {
                    Ok(ev) => {
                        if tx.send(AppEvent::Term(ev)).is_err() {
                            break; // main loop gone.
                        }
                    }
                    Err(_) => break,
                },
                Ok(false) => {}
                Err(_) => break,
            }
            if last_tick.elapsed() >= tick {
                if tx.send(AppEvent::Tick).is_err() {
                    break;
                }
                last_tick = Instant::now();
            }
        }
    });
}

/// Forward TAGGED bridge events `(session_id, ev)` onto the unified channel.
fn spawn_bridge_forwarder(bridge_rx: Receiver<(u64, BridgeEvent)>, tx: Sender<AppEvent>) {
    std::thread::spawn(move || {
        while let Ok((sid, ev)) = bridge_rx.recv() {
            if tx.send(AppEvent::Bridge(sid, ev)).is_err() {
                break;
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Terminal lifecycle — alt screen + raw mode, ALWAYS restored.
// ---------------------------------------------------------------------------

fn setup_terminal() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    // Alt-screen (own the scroll region, P1) + mouse capture (wheel scroll).
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;
    Ok(terminal)
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    // Disable raw mode + mouse capture + leave the alternate screen + show the
    // cursor. Best effort on each step so a partial failure still restores as
    // much as it can — and mouse mode is ALWAYS turned off (else the user's
    // terminal stays in a broken mouse-reporting state, a cousin of the
    // alt-screen-not-restored bug).
    let _ = disable_raw_mode();
    let _ = execute!(terminal.backend_mut(), DisableMouseCapture, LeaveAlternateScreen);
    let _ = terminal.show_cursor();
    Ok(())
}

/// Install a panic hook that restores the terminal BEFORE printing the panic, so
/// a crash never leaves the user in a raw, alt-screen, no-cursor state. Chains
/// the previous hook so the backtrace/message still prints.
fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = io::stdout().execute(DisableMouseCapture);
        let _ = io::stdout().execute(LeaveAlternateScreen);
        let _ = io::stdout().execute(crossterm::cursor::Show);
        previous(info);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use bridge::protocol::CoreToUi;

    /// THE deliverable test: the `/rewind` truncation COUNT.
    ///
    /// The `/rewind` picker offers the last ~20 USER turns; its selected row id is
    /// the FULL-transcript index of that user message. Truncating the transcript at
    /// that index drops that turn and everything after it; the number of REAL (user)
    /// turns dropped — what the UI reports and what the `Rewind{n}` frame tells the
    /// bridge to cut from `backend.history` — is computed by `rewind_real_turns_from`.
    /// This pins that count against a realistic interleaved transcript AND confirms
    /// the actual `transcript.truncate(id)` removes exactly that many user turns.
    #[test]
    fn rewind_truncation_count() {
        // Build a 4-turn conversation: U0 A0 U1 A1 U2 A2 U3 A3 (+ a trailing notice
        // that must NOT be counted as a turn). Indices: U0=0,A0=1,U1=2,A1=3,U2=4,
        // A2=5,U3=6,A3=7,notice=8.
        let mut app = AppState::new();
        for i in 0..4 {
            app.push_user(format!("question {i}"));
            // Stream an assistant reply for this turn.
            let mid = format!("m{i}");
            app.apply_bridge_event(
                BridgeEvent::Frame(CoreToUi::MessageBegin { mid: mid.clone(), role: "assistant".into() }),
                (i as u64) * 10,
            );
            app.apply_bridge_event(
                BridgeEvent::Frame(CoreToUi::MessageDelta { mid: mid.clone(), text: format!("answer {i}") }),
                (i as u64) * 10,
            );
            app.apply_bridge_event(
                BridgeEvent::Frame(CoreToUi::MessageEnd { mid, reason: "stop".into() }),
                (i as u64) * 10,
            );
        }
        app.push_notice("a system notice (not a turn)".into());

        // Locate each user block's full-transcript index (the picker row ids).
        let user_idxs: Vec<usize> = app
            .transcript
            .iter()
            .enumerate()
            .filter(|(_, b)| matches!(b.role, app::Role::User))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(user_idxs, vec![0, 2, 4, 6], "four user turns at the expected indices");

        // Rewinding to the LAST user turn (index 6) drops exactly 1 real turn
        // (U3 + A3); the trailing notice after A3 is NOT a turn but is still cut.
        assert_eq!(rewind_real_turns_from(&app.transcript, 6), 1);
        // Rewinding to U2 (index 4) drops 2 real turns (U2,A2,U3,A3).
        assert_eq!(rewind_real_turns_from(&app.transcript, 4), 2);
        // Rewinding to U1 drops 3; to U0 (the start) drops all 4.
        assert_eq!(rewind_real_turns_from(&app.transcript, 2), 3);
        assert_eq!(rewind_real_turns_from(&app.transcript, 0), 4);
        // A truncate point with no user turns at/after it still reports ≥1 (picking
        // a turn always drops at least that turn) — e.g. at the trailing notice.
        assert_eq!(rewind_real_turns_from(&app.transcript, 8), 1);
        assert_eq!(rewind_real_turns_from(&app.transcript, app.transcript.len()), 1);

        // The COUNT matches what `transcript.truncate(id)` actually removes: cutting
        // at U2 (index 4) leaves U0 A0 U1 A1 = 2 user turns, i.e. it dropped 2.
        let before_turns = user_idxs.len();
        let dropped = rewind_real_turns_from(&app.transcript, 4);
        app.transcript.truncate(4);
        let after_turns = app
            .transcript
            .iter()
            .filter(|b| matches!(b.role, app::Role::User))
            .count();
        assert_eq!(before_turns - after_turns, dropped, "the count equals the turns actually removed");
        assert_eq!(after_turns, 2);
    }

    /// A fresh transcript (no turns) yields nothing to rewind, and the count is the
    /// clamped minimum (never 0) for any index — so the `Rewind{n}` never sends a
    /// nonsensical 0 when a row was somehow selected.
    #[test]
    fn rewind_count_clamps_on_empty() {
        let app = AppState::new();
        assert!(app.transcript.is_empty());
        // Any index on an empty transcript → the clamped minimum of 1.
        assert_eq!(rewind_real_turns_from(&app.transcript, 0), 1);
        assert_eq!(rewind_real_turns_from(&app.transcript, 5), 1);
    }

    /// THE `/effort` forwarding deliverable (redesign_cc.md §3): applying a level
    /// forwards a `Command{name:"session.reasoning_effort=<backend>", args:""}` whose
    /// reconstruction over the bridge's EXACT rule (`"/" + name + (" "+args if args)`,
    /// ga_bridge.py:822-825) is the `/session.reasoning_effort=<backend>` line the GA
    /// core hot-reloads — with the slider `max` forwarding the backend `xhigh`. The
    /// frame is built by the pure `effort_command_frame` so this pins the wire
    /// contract WITHOUT spawning a child.
    #[test]
    fn effort_forwards_session_command() {
        use app::effort::ReasoningEffort;

        // The bridge's reconstruction rule for a Command{name, args} (mirrors
        // ga_bridge.py handle(): it submits "/" + name + (" " + args if args)).
        let reconstruct = |f: &UiToCore| -> String {
            match f {
                UiToCore::Command { name, args } => {
                    if args.is_empty() {
                        format!("/{name}")
                    } else {
                        format!("/{name} {args}")
                    }
                }
                other => panic!("expected a Command frame, got {other:?}"),
            }
        };

        // Every stop forwards a Command that reconstructs to its session command.
        for &lvl in &ReasoningEffort::LEVELS {
            let frame = effort_command_frame(lvl);
            // The forwarded args are empty (the whole line lives in `name`).
            match &frame {
                UiToCore::Command { args, .. } => assert!(args.is_empty(), "args are empty for {lvl:?}"),
                other => panic!("expected Command, got {other:?}"),
            }
            assert_eq!(
                reconstruct(&frame),
                lvl.session_command(),
                "the forwarded frame reconstructs to the session command for {lvl:?}"
            );
        }

        // The load-bearing mapping: the slider `max` forwards the backend `xhigh`.
        assert_eq!(
            reconstruct(&effort_command_frame(ReasoningEffort::Max)),
            "/session.reasoning_effort=xhigh",
            "max forwards xhigh"
        );
        assert_eq!(
            reconstruct(&effort_command_frame(ReasoningEffort::High)),
            "/session.reasoning_effort=high"
        );

        // The frame serializes to the JSONL the bridge dispatches on (a Command).
        let line = effort_command_frame(ReasoningEffort::Max).to_line();
        assert!(line.contains(r#""type":"Command""#));
        assert!(line.contains("session.reasoning_effort=xhigh"));

        // Applying a level (the slider-Enter / `/effort <level>` shared path) also
        // remembers it for the spinner suffix.
        let mut app = AppState::new();
        assert_eq!(app.effort_label(), None, "no effort suffix until set");
        app.set_reasoning_effort(ReasoningEffort::Max);
        assert_eq!(app.effort_label(), Some("max"), "the spinner shows the slider label");
        app.set_reasoning_effort(ReasoningEffort::High);
        assert_eq!(app.effort_label(), Some("high"));
    }

    /// THE mouse deliverable (redesign_cc.md §4): a RIGHT-click in the Dashboard
    /// returns to the cockpit (the mouse mirror of `Esc`), a right-click in the
    /// cockpit is a no-op (already at the root view), and a LEFT-click on the header
    /// band still opens the dashboard. Drives the real `handle_mouse_event` path so
    /// the `MouseButton::Right` arm is exercised, not just asserted in prose.
    #[test]
    fn right_click_returns_from_dashboard() {
        use crossterm::event::{KeyModifiers, MouseButton, MouseEvent, MouseEventKind};

        let down = |kind: MouseEventKind, col: u16, row: u16| MouseEvent {
            kind,
            column: col,
            row,
            modifiers: KeyModifiers::NONE,
        };
        let right = |col, row| down(MouseEventKind::Down(MouseButton::Right), col, row);
        let left = |col, row| down(MouseEventKind::Down(MouseButton::Left), col, row);

        let mut app = AppState::new();
        // Give the dashboard a sane click-geometry (set on render in the live app).
        app.set_term_size(100, 30);
        assert_eq!(app.view, View::Cockpit, "starts in the cockpit");

        // A LEFT-click on the header band (row 0) opens the dashboard (§4/§6).
        handle_mouse_event(left(10, 0), &mut app);
        assert_eq!(app.view, View::Dashboard, "left-click on the header opens the dashboard");

        // A RIGHT-click anywhere in the dashboard goes BACK to the cockpit.
        handle_mouse_event(right(42, 7), &mut app);
        assert_eq!(app.view, View::Cockpit, "right-click in the dashboard returns to the cockpit");

        // A RIGHT-click in the cockpit is a NO-OP (stays at the root view).
        handle_mouse_event(right(42, 7), &mut app);
        assert_eq!(app.view, View::Cockpit, "right-click in the cockpit is a no-op");

        // The broadened open-zone: a left-click on the separator row (row 1) also
        // opens the dashboard (not just the 1-row header sliver).
        handle_mouse_event(left(3, 1), &mut app);
        assert_eq!(app.view, View::Dashboard, "left-click on the separator row also opens it");

        // And right-click closes it again — round-trips cleanly.
        handle_mouse_event(right(0, 0), &mut app);
        assert_eq!(app.view, View::Cockpit, "right-click round-trips back to the cockpit");
    }
}
