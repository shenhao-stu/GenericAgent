//! commands/ — magic-prefix execution: the `!cmd` host shell (checklist §4 / §10
//! "`!cmd` seeds agent context without spending a turn") and the `@path` inline
//! expansion glue. The slash-command registry/dispatch lands in Phase 3; this
//! module ships the two MAGIC prefixes the cockpit needs now.
//!
//! `!cmd` behavior (tui_v3 `_run_shell`): run the command body in the host shell
//! with a 30s timeout, capture stdout+stderr, and produce BOTH:
//!   * a transcript SYSTEM block echoing `! cmd` then `└ output` (what the user
//!     sees), and
//!   * a `Command{name:"shell", args:<formatted note>}` frame for the bridge so
//!     the AGENT gets the exchange as context (ga_bridge.py `handle_shell_note`
//!     stashes it into `_intervene`) — WITHOUT spending a turn.
//!
//! The OUTPUT FORMATTERS (`format_shell_block`, `format_shell_note`) are PURE +
//! unit-tested; `run_shell` is the thin effectful wrapper that spawns the child
//! with a timeout (decoded `from_utf8_lossy` for Chinese-Windows safety).

pub mod registry;

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use registry::{resolve, split_command, CommandKind};

/// The classified OUTCOME of a typed `/command args` line — the pure routing
/// decision the app then executes (checklist §4 dispatch). Separating the decision
/// from the effect keeps the routing unit-testable (the dispatcher in `main.rs`
/// matches on this and performs the I/O: open an overlay, send a frame, quit, …).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlashOutcome {
    /// Open a dedicated UI panel/picker for a §4 **UI** command. `name` is the
    /// resolved command name; `args` the (trimmed) remainder.
    OpenUi { name: String, args: String },
    /// Core-forward a §4 **fwd** command as a `Command{name,args}` frame.
    Forward { name: String, args: String },
    /// Handle a §4 **app** command in-app. `name` is the resolved command.
    App { name: String, args: String },
    /// The leading-`/` text did not resolve to a known command. `suggestion` is the
    /// closest command name for a "did you mean /x?" breadcrumb (if any).
    Unknown { typed: String, suggestion: Option<String> },
}

/// Classify a SUBMITTED line that begins with `/` into a [`SlashOutcome`] (PURE).
/// The caller passes the whole submitted text; a non-slash line is treated as
/// `Unknown` (the caller should only call this for `/`-prefixed input). Resolution
/// is via the registry, so EVERY §4 name routes; an unknown name yields a
/// did-you-mean suggestion.
pub fn classify_slash(line: &str) -> SlashOutcome {
    let (name, args) = split_command(line);
    match resolve(&name) {
        Some(cmd) => {
            let name = cmd.name.to_string();
            match cmd.kind {
                CommandKind::Ui => SlashOutcome::OpenUi { name, args },
                CommandKind::Fwd => SlashOutcome::Forward { name, args },
                CommandKind::App => SlashOutcome::App { name, args },
            }
        }
        None => SlashOutcome::Unknown {
            typed: name.clone(),
            suggestion: registry::did_you_mean(&name).map(str::to_string),
        },
    }
}

/// The `!cmd` shell timeout (checklist §4: "30 s timeout").
pub const SHELL_TIMEOUT: Duration = Duration::from_secs(30);

/// Max bytes of combined output we keep (bound a runaway command).
pub const SHELL_MAX_OUTPUT: usize = 64 * 1024;

/// The captured result of a `!cmd` run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellResult {
    /// The command that ran (bang already stripped).
    pub cmd: String,
    /// Combined stdout+stderr (decoded lossy, bounded, trailing ws trimmed).
    pub output: String,
    /// Process exit code if it exited normally; `None` on timeout/spawn error.
    pub code: Option<i32>,
    /// True if we killed it for exceeding [`SHELL_TIMEOUT`].
    pub timed_out: bool,
}

/// Run a `!cmd` line in the host shell with a 30s timeout. Effectful. On any
/// failure (spawn error / timeout) it still returns a `ShellResult` describing
/// what happened (never a panic, never a silent hang).
///
/// `cwd` is the directory the command runs in (the GA repo root). The shell is
/// `cmd /C` on Windows and `sh -c` elsewhere.
pub fn run_shell(cmd: &str, cwd: &std::path::Path) -> ShellResult {
    let mut command = shell_command(cmd);
    command
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = match command.spawn() {
        Ok(c) => c,
        Err(e) => {
            return ShellResult {
                cmd: cmd.to_string(),
                output: format!("[shell spawn failed] {e}"),
                code: None,
                timed_out: false,
            };
        }
    };

    // Drain stdout+stderr on threads so a chatty command can't deadlock on a
    // full pipe while we poll for exit.
    let mut out_handle = child.stdout.take();
    let mut err_handle = child.stderr.take();
    let out_thread = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(h) = out_handle.as_mut() {
            let _ = h.take(SHELL_MAX_OUTPUT as u64).read_to_end(&mut buf);
        }
        buf
    });
    let err_thread = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(h) = err_handle.as_mut() {
            let _ = h.take(SHELL_MAX_OUTPUT as u64).read_to_end(&mut buf);
        }
        buf
    });

    // Poll for exit up to the timeout.
    let start = Instant::now();
    let mut timed_out = false;
    let code = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status.code(),
            Ok(None) => {
                if start.elapsed() >= SHELL_TIMEOUT {
                    let _ = child.kill();
                    let _ = child.wait();
                    timed_out = true;
                    break None;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(_) => break None,
        }
    };

    let out_bytes = out_thread.join().unwrap_or_default();
    let err_bytes = err_thread.join().unwrap_or_default();
    let mut output = String::new();
    output.push_str(&String::from_utf8_lossy(&out_bytes));
    if !err_bytes.is_empty() {
        if !output.is_empty() && !output.ends_with('\n') {
            output.push('\n');
        }
        output.push_str(&String::from_utf8_lossy(&err_bytes));
    }
    if timed_out {
        if !output.is_empty() && !output.ends_with('\n') {
            output.push('\n');
        }
        output.push_str("[timed out after 30s]");
    }
    // Bound + trim.
    if output.len() > SHELL_MAX_OUTPUT {
        output.truncate(SHELL_MAX_OUTPUT);
        output.push_str("\n…[truncated]");
    }
    let output = output.trim_end().to_string();

    ShellResult {
        cmd: cmd.to_string(),
        output,
        code,
        timed_out,
    }
}

/// Build the platform shell command for `cmd` (`cmd /C` on Windows, `sh -c`
/// elsewhere). Separated so the host-shell selection is explicit.
fn shell_command(cmd: &str) -> Command {
    if cfg!(windows) {
        let mut c = Command::new("cmd");
        c.arg("/C").arg(cmd);
        c
    } else {
        let mut c = Command::new("sh");
        c.arg("-c").arg(cmd);
        c
    }
}

/// Format the TRANSCRIPT system block for a shell run (what the user sees):
/// the `! cmd` line then each output line prefixed `└ ` (tui_v3 echo format).
/// PURE.
pub fn format_shell_block(res: &ShellResult) -> String {
    let mut out = format!("! {}", res.cmd);
    if res.output.is_empty() {
        out.push_str("\n└ (no output)");
    } else {
        for line in res.output.lines() {
            out.push_str("\n└ ");
            out.push_str(line);
        }
    }
    out
}

/// Format the AGENT-CONTEXT note for the `Command{name:"shell"}` frame (what the
/// bridge stashes into `_intervene` so the agent can reference the exchange). A
/// compact, labeled block. PURE.
pub fn format_shell_note(res: &ShellResult) -> String {
    let status = match (res.timed_out, res.code) {
        (true, _) => " (timed out)".to_string(),
        (false, Some(0)) => String::new(),
        (false, Some(c)) => format!(" (exit {c})"),
        (false, None) => " (unknown exit)".to_string(),
    };
    format!(
        "[!shell]{status} $ {}\n{}",
        res.cmd,
        if res.output.is_empty() {
            "(no output)"
        } else {
            &res.output
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_shell_block_echoes_command_and_output() {
        let res = ShellResult {
            cmd: "echo hi".into(),
            output: "hi\nthere".into(),
            code: Some(0),
            timed_out: false,
        };
        let block = format_shell_block(&res);
        assert_eq!(block, "! echo hi\n└ hi\n└ there");

        // No output → a clear marker.
        let empty = ShellResult {
            cmd: "true".into(),
            output: String::new(),
            code: Some(0),
            timed_out: false,
        };
        assert_eq!(format_shell_block(&empty), "! true\n└ (no output)");
    }

    #[test]
    fn format_shell_note_labels_status_for_agent() {
        let ok = ShellResult {
            cmd: "ls".into(),
            output: "a\nb".into(),
            code: Some(0),
            timed_out: false,
        };
        assert_eq!(format_shell_note(&ok), "[!shell] $ ls\na\nb");

        let failed = ShellResult {
            cmd: "false".into(),
            output: String::new(),
            code: Some(1),
            timed_out: false,
        };
        assert_eq!(format_shell_note(&failed), "[!shell] (exit 1) $ false\n(no output)");

        let slow = ShellResult {
            cmd: "sleep 99".into(),
            output: "[timed out after 30s]".into(),
            code: None,
            timed_out: true,
        };
        assert!(format_shell_note(&slow).starts_with("[!shell] (timed out) $ sleep 99"));
    }

    #[test]
    fn classify_slash_routes_every_kind() {
        use SlashOutcome::*;
        // UI command → OpenUi with the resolved name + args.
        assert_eq!(
            classify_slash("/llm 2"),
            OpenUi { name: "llm".into(), args: "2".into() }
        );
        assert_eq!(
            classify_slash("/theme"),
            OpenUi { name: "theme".into(), args: "".into() }
        );
        // fwd command → Forward.
        assert_eq!(
            classify_slash("/review fix the bug"),
            Forward { name: "review".into(), args: "fix the bug".into() }
        );
        assert_eq!(
            classify_slash("/conductor build X"),
            Forward { name: "conductor".into(), args: "build X".into() }
        );
        // app command → App.
        assert_eq!(classify_slash("/clear"), App { name: "clear".into(), args: "".into() });
        assert_eq!(classify_slash("/quit"), App { name: "quit".into(), args: "".into() });
        // Unknown → did-you-mean suggestion.
        match classify_slash("/halp") {
            Unknown { suggestion, .. } => assert_eq!(suggestion.as_deref(), Some("help")),
            other => panic!("expected Unknown, got {other:?}"),
        }
        // Aliases resolve to their own name (tools/trace are UI).
        assert_eq!(
            classify_slash("/tools"),
            OpenUi { name: "tools".into(), args: "".into() }
        );
    }

    #[test]
    fn run_shell_executes_and_captures_output() {
        // A trivial, fast, cross-platform echo.
        let cwd = std::env::temp_dir();
        let res = run_shell("echo tui_v4_shell_probe", &cwd);
        assert!(res.output.contains("tui_v4_shell_probe"), "got: {:?}", res.output);
        assert_eq!(res.code, Some(0));
        assert!(!res.timed_out);
    }
}
