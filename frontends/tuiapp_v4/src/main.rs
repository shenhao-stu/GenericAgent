//! tui_v4 — GenericAgent's terminal UI, rebuilt in Rust + ratatui.
//!
//! This binary root is just the entry + the event loop + the `--smoke`/
//! `--dump-frame` harness (ARCH Fix B). Key handling lives in `input/`, command
//! dispatch in `commands/dispatch.rs`, the terminal lifecycle in `term.rs`, and
//! the copy surface in `clipboard.rs`. The loop is the ONE place the bridge
//! transport (`tx_bridge`) lives: handlers emit [`AppEvent`] intents (ARCH Fix A)
//! that `perform_actions` drains and performs here.

mod app;
mod app_event;
mod bridge;
mod clipboard;
mod commands;
mod components;
mod effects;
mod flavor;
mod i18n;
mod input;
mod markdown;
mod render;
mod term;
mod theme;
mod util;
mod workflow;

use std::io::Stdout;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::backend::{Backend, CrosstermBackend, TestBackend};
use ratatui::layout::Rect;
use ratatui::Terminal;

use app::{AppState, View};
use app_event::AppEvent;
use bridge::{BridgeEvent, BridgeOptions};
use theme::Theme;

const VERSION: &str = "0.3.0";

/// The full terminal area (origin `(0,0)`, current size) — what `frame.area()`
/// reports for a fullscreen viewport. Read BEFORE `terminal.draw` so the loop can
/// run `AppState::prepare_frame` (the hoisted per-frame state writes, F7) ahead of
/// the now-pure render.
fn terminal_area<B: Backend>(terminal: &Terminal<B>) -> Result<Rect>
where
    B::Error: std::error::Error + Send + Sync + 'static,
{
    let size = terminal.size()?;
    Ok(Rect::new(0, 0, size.width, size.height))
}

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
        // print it as TEXT ROWS, so the LAYOUT is inspectable headlessly.
        // Scenario: normal | shell | busy | effort | effort-high | cost | done | keybindings.
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
    // markdown render plane (headings, bold, inline code, `$…$` math).
    app.apply_bridge_event(
        BridgeEvent::Frame(bridge::protocol::CoreToUi::MessageDelta {
            mid: "m1".into(),
            text: "## Result\n\nThe angle $\\alpha$ is **small**; see `main.rs`.".into(),
        }),
        0,
    );
    let theme = Theme::default_theme();

    // Render is PURE (F7): hoist the per-frame state writes into `prepare_frame`
    // BEFORE the draw, then draw from `&app` (immutable).
    let area = terminal_area(&terminal)?;
    app.prepare_frame(area, &theme);
    terminal.draw(|f| components::render(f, &app, &theme, 100))?;

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

/// Resolve the GA repo root for the `--dump-frame scheduler` seed: walk up from the
/// cwd to the first ancestor that holds a `reflect/` dir (cargo runs us inside
/// `frontends/tuiapp_v4`). Falls back to the cwd when no such ancestor exists (the
/// scheduler then degrades to `default_tasks`).
fn find_ga_root() -> std::path::PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let mut dir = cwd.as_path();
    for _ in 0..10 {
        if dir.join("reflect").is_dir() {
            return dir.to_path_buf();
        }
        match dir.parent() {
            Some(p) => dir = p,
            None => break,
        }
    }
    cwd
}

/// `--dump-frame [normal|shell|busy|effort|effort-high|cost|done|keybindings|
/// scheduler|continue]`: render the cockpit with representative seeded state into a
/// 100x30 in-memory backend and print every row as text, so the LAYOUT can be
/// eyeballed without a TTY.
fn run_dump_frame(scenario: &str) -> Result<()> {
    let (w, h) = (100u16, 30u16);
    let backend = TestBackend::new(w, h);
    let mut terminal = Terminal::new(backend)?;

    let mut app = AppState::new();
    app.lang = i18n::detect_system_lang();
    // The REAL MixinSession model shape: `get_llm_name()` returns the whole
    // fallback chain; the header/footer must truncate to the primary segment.
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
    // The assistant stream uses the REAL GA markers so the dump verifies they
    // render clean (Turn boundary, <summary>, compact tool call, [Info], !!!Error).
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
    // Seed a token/context snapshot so the dump shows the spinner readout + /cost.
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
    // A failover retry diagnostic on STDERR must be suppressed (no `[bridge]` row).
    app.apply_bridge_event(
        BridgeEvent::Stderr {
            line: "[MixinSession] codex-pro overloaded, retry 1/10 (2.0s→4.0s)".into(),
        },
        0,
    );
    match scenario {
        "busy" => {
            // The spinner frame: leave the turn open (busy=true). Stream a half-built
            // block math into the VOLATILE tail (Q3): `safe_commit_pos` holds it back
            // and the tail renderer trims the in-flight `$$…` so the active region
            // NEVER flashes a raw `$$` (C1-F3 + Fix D tail render).
            app.apply_bridge_event(
                BridgeEvent::Frame(bridge::protocol::CoreToUi::MessageDelta {
                    mid: "m1".into(),
                    text: "\n\nand the closed form is $$\\frac{a}{".into(),
                }),
                0,
            );
        }
        "effort" => {
            // The `/effort` slider over a busy cockpit: live level + applied `●`.
            app.set_reasoning_effort(app::effort::ReasoningEffort::Medium);
            app.open_effort_slider();
        }
        "effort-high" => {
            // Same slider, marker nudged right (→ → from medium = xhigh).
            app.set_reasoning_effort(app::effort::ReasoningEffort::Medium);
            app.open_effort_slider();
            if let Some(app::Overlay::EffortSlider(s)) = app.overlay.as_mut() {
                s.move_marker(2);
            }
        }
        "shell" => {
            // Finalize the turn (idle) so shell mode shows WITHOUT the spinner band.
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
            // Finalize the turn, then open /cost (token usage from the seeded Status).
            app.apply_bridge_event(
                BridgeEvent::Frame(bridge::protocol::CoreToUi::MessageEnd {
                    mid: "m1".into(),
                    reason: "stop".into(),
                }),
                0,
            );
            app.open_overlay(app::Overlay::Cost);
        }
        "keybindings" => {
            // Finalize the turn, then open /keybindings (the chord→action pairs
            // table + magic-prefix line — Q7).
            app.apply_bridge_event(
                BridgeEvent::Frame(bridge::protocol::CoreToUi::MessageEnd {
                    mid: "m1".into(),
                    reason: "stop".into(),
                }),
                0,
            );
            app.open_overlay(app::Overlay::Keybindings);
        }
        "done" => {
            // The above-composer FROZEN done-line (Q7): finalize the turn at a later
            // monotonic time so `last_turn_ms` reads a realistic `1m 46s` (the turn
            // began at now_ms=0 in the seed above). The `↑in/↓out` come from the
            // seeded Status (1234/340 → 1.2k/340).
            app.apply_bridge_event(
                BridgeEvent::Frame(bridge::protocol::CoreToUi::MessageEnd {
                    mid: "m1".into(),
                    reason: "stop".into(),
                }),
                106_000,
            );
        }
        "scheduler" => {
            // The `/scheduler` overlay seeded with the REAL discovery (Q10): reflect
            // modes (reflect/*.py, minus scheduler.py) + cron tasks (sche_tasks/*.json)
            // with their legal `repeat` cadences — never a fake HH:MM. Resolve the GA
            // repo root by walking up from cwd (cargo runs us inside frontends/...).
            app.apply_bridge_event(
                BridgeEvent::Frame(bridge::protocol::CoreToUi::MessageEnd {
                    mid: "m1".into(),
                    reason: "stop".into(),
                }),
                0,
            );
            app.repo_root = find_ga_root();
            let mut tasks = components::scheduler::discover_tasks(&app.repo_root);
            if tasks.is_empty() {
                tasks = components::scheduler::default_tasks();
            }
            app.open_overlay(app::Overlay::Scheduler(components::scheduler::Scheduler::new(tasks)));
        }
        "continue" => {
            // The de-iconified `/continue` restore banner (Q10): the bridge strips the
            // leading ✅/⚠️/❌ before streaming it as a system line, so the transcript
            // shows the informative text WITHOUT an icon.
            app.apply_bridge_event(
                BridgeEvent::Frame(bridge::protocol::CoreToUi::MessageEnd {
                    mid: "m1".into(),
                    reason: "stop".into(),
                }),
                0,
            );
            app.push_system(
                "已恢复 148 轮完整对话（model_responses_335438.txt）\n\
                 (已写入 backend.history，可直接继续)"
                    .into(),
            );
        }
        _ => {
            // `normal`: a TRUE IDLE cockpit. Finalize the turn so busy=false.
            app.apply_bridge_event(
                BridgeEvent::Frame(bridge::protocol::CoreToUi::MessageEnd {
                    mid: "m1".into(),
                    reason: "stop".into(),
                }),
                0,
            );
        }
    }

    let theme = Theme::default_theme();
    // Render is PURE (F7): prepare state writes BEFORE the draw, then draw from `&app`.
    let area = terminal_area(&terminal)?;
    app.prepare_frame(area, &theme);
    terminal.draw(|f| components::render(f, &app, &theme, 100))?;

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

/// A unified event for the main loop, multiplexing the three sources. (Distinct
/// from [`AppEvent`], the UI→app INTENT bus — this is the loop's input channel.)
enum LoopEvent {
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
    term::install_panic_hook();

    let mut terminal = term::setup()?;

    // The single channel all three sources feed (bridge frames, ticks, input).
    // Input + tick run on their own thread so the loop never blocks.
    let (tx, rx): (Sender<LoopEvent>, Receiver<LoopEvent>) = mpsc::channel();
    spawn_input_thread(tx.clone());

    // The TAGGED bridge-event channel: every session's child forwards
    // `(session_id, BridgeEvent)` here; a forwarder lifts them onto the unified
    // `LoopEvent` channel. This is the §6 multiplexer — N children, routed by id.
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
    term::restore(&mut terminal)?;
    app.sessions.shutdown_all();

    result
}

/// The main event loop. Returns when the user quits.
fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    app: &mut AppState,
    tx_bridge: &Sender<(u64, BridgeEvent)>,
    rx: &Receiver<LoopEvent>,
    start: Instant,
) -> Result<()> {
    loop {
        let now_ms = start.elapsed().as_millis() as u64;
        // While the /workflows panel is open, pull the watcher's latest merged
        // snapshot into the app before drawing (the watcher polls off-thread; this
        // is a cheap lock+clone that never blocks — §3 "never blocks chat").
        if app.view == View::Workflows {
            app.refresh_workflow_snapshot();
        }
        // The theme is owned by the app (so `/theme` live-preview can swap it); the
        // render functions still take it by reference, so clone it for this frame.
        let theme = app.theme.clone();
        // Render is PURE (F7): the per-frame state writes (`set_term_size` +
        // transcript wrap-cache/viewport sync) are hoisted into `prepare_frame`,
        // called HERE before the draw; the draw then reads `&app` (immutable, P11).
        let area = terminal_area(terminal)?;
        app.prepare_frame(area, &theme);
        terminal.draw(|f| components::render(f, app, &theme, now_ms))?;

        // Emit the OSC0 title + OSC-21337 tab status when they change (§9). Done
        // AFTER the draw so the out-of-band escapes never interleave mid-frame.
        app.sync_terminal_chrome();

        // Block for the next event from ANY source (input thread, tick thread,
        // bridge forwarder). A short recv timeout keeps the elapsed clock /
        // spinner ticking even if no event arrives.
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(LoopEvent::Term(ev)) => handle_term_event(ev, app, now_ms),
            Ok(LoopEvent::Tick) => app.tick(),
            Ok(LoopEvent::Bridge(sid, be)) => {
                let now_ms = start.elapsed().as_millis() as u64;
                // Route the tagged event to its session (active → live reducer;
                // others → their own record so the dashboard preview updates).
                app.apply_tagged_event(sid, be, now_ms);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => app.tick(),
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }

        // Drain + perform the intents the handlers emitted this iteration (ARCH
        // Fix A). This is the ONE place the bridge transport lives for UI intents;
        // the handlers `app.emit(AppEvent::ToActive(…))` instead of threading
        // `tx_bridge`.
        perform_actions(app, tx_bridge);

        if app.should_quit {
            break;
        }
    }
    Ok(())
}

/// Fold one terminal event into state. Routing forks on the active VIEW: the
/// dashboard (§6) owns its own keys/clicks; the cockpit handles composer + scroll.
fn handle_term_event(ev: Event, app: &mut AppState, now_ms: u64) {
    match ev {
        Event::Key(key) => {
            // A MODAL overlay (picker / ask-user / help / cost / scheduler /
            // continue) consumes keys FIRST — the topmost layer of the §3 overlay
            // stack. A NON-modal overlay (the `/btw` toast) only intercepts Esc
            // (dismiss); every other key flows to the cockpit so chat stays usable
            // while the side question is in flight (§7 `/btw` non-blocking).
            let modal = app.overlay.as_ref().map(|o| o.is_modal());
            match modal {
                Some(true) => input::views::handle_overlay_key(key, app),
                Some(false) => {
                    if key.code == KeyCode::Esc && key.kind != KeyEventKind::Release {
                        app.close_overlay(); // dismiss the /btw card (no history pollution).
                    } else {
                        input::views::route_view_key(key, app, now_ms);
                    }
                }
                None => input::views::route_view_key(key, app, now_ms),
            }
        }
        Event::Mouse(me) => input::mouse::mouse(me, app),
        // Resize is handled implicitly: the next `render` re-syncs the wrap cache
        // + viewport to the new geometry and re-derives the window from the same
        // logical anchor (P1). ratatui repaints from state, so nothing to do here.
        _ => {}
    }
}

/// Drain the intents the handlers queued this loop iteration and perform each one
/// (ARCH Fix A). This is the SINGLE site the bridge transport is applied for UI
/// intents: a handler emits `AppEvent::ToActive(frame)` instead of closing over
/// `tx_bridge`.
fn perform_actions(app: &mut AppState, tx_bridge: &Sender<(u64, BridgeEvent)>) {
    for ev in app.drain_actions() {
        match ev {
            AppEvent::ToActive(frame) => {
                let active = app.sessions.active;
                if !app.sessions.send_to(active, frame, tx_bridge) {
                    app.push_notice(i18n::tf(app.lang, "notice.bridge.not_connected"));
                }
            }
            AppEvent::ToSession(id, frame) => {
                app.sessions.send_to(id, frame, tx_bridge);
            }
            AppEvent::Copy { text, label } => clipboard::perform_copy(app, &text, label),
            AppEvent::SetMouseCapture(on) => {
                app.mouse_capture = on;
                term::set_mouse_capture(on);
            }
            AppEvent::OpenWorkflows => app.open_workflows(),
            AppEvent::OpenDashboard => app.open_dashboard(),
            AppEvent::CloseView => app.close_dashboard(),
            AppEvent::Quit => app.should_quit = true,
        }
    }
}

// ---------------------------------------------------------------------------
// Threads feeding the unified channel.
// ---------------------------------------------------------------------------

/// Poll crossterm for input on a dedicated thread and emit a `Tick` every 0.1s.
/// This keeps `terminal.draw` + `rx.recv` on the main thread non-blocking.
fn spawn_input_thread(tx: Sender<LoopEvent>) {
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
                        if tx.send(LoopEvent::Term(ev)).is_err() {
                            break; // main loop gone.
                        }
                    }
                    Err(_) => break,
                },
                Ok(false) => {}
                Err(_) => break,
            }
            if last_tick.elapsed() >= tick {
                if tx.send(LoopEvent::Tick).is_err() {
                    break;
                }
                last_tick = Instant::now();
            }
        }
    });
}

/// Forward TAGGED bridge events `(session_id, ev)` onto the unified channel.
fn spawn_bridge_forwarder(bridge_rx: Receiver<(u64, BridgeEvent)>, tx: Sender<LoopEvent>) {
    std::thread::spawn(move || {
        while let Ok((sid, ev)) = bridge_rx.recv() {
            if tx.send(LoopEvent::Bridge(sid, ev)).is_err() {
                break;
            }
        }
    });
}
