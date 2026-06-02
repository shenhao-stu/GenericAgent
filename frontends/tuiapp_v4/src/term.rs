//! term.rs — terminal lifecycle: alt-screen + raw mode, ALWAYS restored.
//!
//! Owns the crossterm setup/teardown + the panic hook so a crash never leaves the
//! user in a raw, alt-screen, no-cursor state, plus the runtime mouse-capture
//! toggle (`Ctrl+Shift+M` / `/mouse`). Split out of `main.rs` so the binary root
//! is just entry + loop + harness (ARCH Fix B).
//!
//! MOUSE TOGGLE MODEL (S1):
//!   DEFAULT = NATIVE mode: no `EnableMouseCapture`, but `EnableAlternateScroll`
//!   (`\x1b[?1007h`, Codex model) so the wheel translates to arrow keys and the
//!   terminal does native OS drag-select + copy.
//!   TOGGLE (Ctrl+Shift+M / `/mouse`) → INTERACTIVE mode: `EnableMouseCapture` ON
//!   so click `▸/▾` fold + wheel `ScrollUp/Down`.  Native selection is suppressed
//!   in this mode; toggle back to native to copy.

use std::fmt;
use std::io::{self, Stdout};

use anyhow::Result;
use crossterm::execute;
use crossterm::event::{DisableMouseCapture, EnableMouseCapture};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::{Command, ExecutableCommand};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

// ---------------------------------------------------------------------------
// AlternateScroll commands — copied from Codex tui/src/tui.rs:173-204.
// `?1007h` makes the terminal translate wheel events into cursor-up/down key
// events WITHOUT stealing native mouse selection (the Codex model). This is
// belt-and-suspenders alongside `EnableMouseCapture` when in INTERACTIVE mode,
// and the ONLY scroll mechanism in NATIVE mode (no capture).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EnableAlternateScroll;

impl Command for EnableAlternateScroll {
    fn write_ansi(&self, f: &mut impl fmt::Write) -> fmt::Result {
        write!(f, "\x1b[?1007h")
    }

    #[cfg(windows)]
    fn execute_winapi(&self) -> std::io::Result<()> {
        Err(std::io::Error::other(
            "tried to execute EnableAlternateScroll using WinAPI; use ANSI instead",
        ))
    }

    #[cfg(windows)]
    fn is_ansi_code_supported(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DisableAlternateScroll;

impl Command for DisableAlternateScroll {
    fn write_ansi(&self, f: &mut impl fmt::Write) -> fmt::Result {
        write!(f, "\x1b[?1007l")
    }

    #[cfg(windows)]
    fn execute_winapi(&self) -> std::io::Result<()> {
        Err(std::io::Error::other(
            "tried to execute DisableAlternateScroll using WinAPI; use ANSI instead",
        ))
    }

    #[cfg(windows)]
    fn is_ansi_code_supported(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Terminal lifecycle
// ---------------------------------------------------------------------------

pub fn setup() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    // NATIVE mode by default: alt-screen + EnableAlternateScroll (wheel → arrow keys)
    // but NO EnableMouseCapture, so the terminal keeps native OS drag-select. The user
    // toggles to INTERACTIVE mode (EnableMouseCapture) via Ctrl+Shift+M / /mouse when
    // they want click-to-fold. `AppState::default()` sets `mouse_capture=false` to
    // match this; the field is the single source of truth.
    execute!(stdout, EnterAlternateScreen, EnableAlternateScroll)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;
    Ok(terminal)
}

pub fn restore(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    // Disable raw mode + alt-scroll + mouse capture (best effort for both so a
    // partially-interactive session is always cleaned up) + leave alt screen + cursor.
    let _ = disable_raw_mode();
    let _ = execute!(
        terminal.backend_mut(),
        DisableAlternateScroll,
        DisableMouseCapture,
        LeaveAlternateScreen
    );
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
        let _ = io::stdout().execute(DisableAlternateScroll);
        let _ = io::stdout().execute(DisableMouseCapture);
        let _ = io::stdout().execute(LeaveAlternateScreen);
        let _ = io::stdout().execute(crossterm::cursor::Show);
        previous(info);
    }));
}

/// Toggle terminal mouse capture at runtime (Ctrl+Shift+M / `/mouse`).
///
/// `on=true` → INTERACTIVE mode: `EnableMouseCapture` so clicks reach crossterm
/// (fold ▸/▾, wheel `ScrollUp/Down`). Native selection is suppressed.
/// `on=false` → NATIVE mode: `DisableMouseCapture`; alt-scroll stays on always
/// so the wheel keeps working as arrow keys while native selection is restored.
///
/// Best-effort (terminal errors are swallowed so a mid-session failure can't crash).
pub fn set_mouse_capture(on: bool) {
    let mut out = io::stdout();
    let _ = if on {
        execute!(out, EnableMouseCapture)
    } else {
        execute!(out, DisableMouseCapture)
    };
    // Alt-scroll stays ON in both modes (wheel → arrow keys is always desired).
    let _ = execute!(out, EnableAlternateScroll);
}
