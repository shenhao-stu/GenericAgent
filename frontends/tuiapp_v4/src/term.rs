//! term.rs — terminal lifecycle: alt-screen + raw mode, ALWAYS restored.
//!
//! Owns the crossterm setup/teardown + the panic hook so a crash never leaves the
//! user in a raw, alt-screen, no-cursor state, plus the runtime mouse-capture
//! toggle (`Ctrl+Shift+M` / `/mouse`). Split out of `main.rs` so the binary root
//! is just entry + loop + harness (ARCH Fix B).

use std::io::{self, Stdout};

use anyhow::Result;
use crossterm::execute;
use crossterm::event::{DisableMouseCapture, EnableMouseCapture};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

pub fn setup() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    // Alt-screen (own the scroll region, P1). Mouse capture starts OFF so the
    // terminal owns drag-select for native inline copy (Q2); wheel-scroll is the
    // opt-in (`Ctrl+Shift+M` flips capture ON). `AppState::new()` defaults
    // `mouse_capture=false` to match this — the field is the single source of truth.
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;
    Ok(terminal)
}

pub fn restore(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
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
pub fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = io::stdout().execute(DisableMouseCapture);
        let _ = io::stdout().execute(LeaveAlternateScreen);
        let _ = io::stdout().execute(crossterm::cursor::Show);
        previous(info);
    }));
}

/// Toggle terminal mouse capture at runtime (Ctrl+Shift+M / `/mouse`). ON = wheel
/// scroll + click-to-dashboard; OFF lets the terminal's OWN drag-select work so
/// the user can select + copy transcript/input text natively (the portable,
/// Windows-safe answer to inline copy — Codex's model). Best-effort.
pub fn set_mouse_capture(on: bool) {
    let mut out = io::stdout();
    let _ = if on {
        execute!(out, EnableMouseCapture)
    } else {
        execute!(out, DisableMouseCapture)
    };
}
