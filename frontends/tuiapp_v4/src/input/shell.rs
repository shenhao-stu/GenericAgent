//! input/shell.rs — the `!cmd` magic-prefix detection (checklist §4 magic
//! prefixes; tui_v3 §C-45). A composer line whose FIRST non-space character is
//! `!` is a host-shell command: on submit it runs the rest in the host shell
//! (30s timeout) instead of going to the model, and the composer border tints
//! hot-pink while the bang is present. All PURE + unit-tested.

/// True when `text` is a shell-mode line (its first non-space char is `!`). A
/// bare `!` (or `!` + only spaces) is NOT a command (nothing to run) but still
/// counts as shell MODE so the border tints while the user is typing the bang.
pub fn is_shell_mode(text: &str) -> bool {
    text.trim_start().starts_with('!')
}

/// True when `text` is a shell line with an actual command body (a non-empty
/// command after the `!`). This is the gate for "run it on submit".
pub fn is_shell_line(text: &str) -> bool {
    matches!(strip_bang_opt(text), Some(cmd) if !cmd.trim().is_empty())
}

/// Strip the leading `!` (and the spaces around it) from a shell line, returning
/// the command body. Returns `None` when `text` is not shell-mode. The body is
/// returned verbatim (trailing newlines preserved so a multi-line `!` here-doc
/// survives), only the leading `!` and the single space after it are removed.
pub fn strip_bang_opt(text: &str) -> Option<String> {
    let trimmed = text.trim_start();
    let rest = trimmed.strip_prefix('!')?;
    // Drop a single leading space after the bang (`! ls` → `ls`), but keep the
    // rest exactly (so indentation inside the command is preserved).
    let body = rest.strip_prefix(' ').unwrap_or(rest);
    Some(body.to_string())
}

/// Strip the bang, defaulting to the original text when not shell-mode (the
/// convenience form the composer's submit path uses after `is_shell_line`).
pub fn strip_bang(text: &str) -> String {
    strip_bang_opt(text).unwrap_or_else(|| text.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_shell_mode_and_line() {
        // Mode = leading bang (even with no body yet → border tints while typing).
        assert!(is_shell_mode("!"));
        assert!(is_shell_mode("  !ls -la"));
        assert!(is_shell_mode("!git status"));
        assert!(!is_shell_mode("hello"));
        assert!(!is_shell_mode("a ! b")); // bang not first.

        // Line (run-on-submit) needs a real command body.
        assert!(!is_shell_line("!"));
        assert!(!is_shell_line("!   "));
        assert!(is_shell_line("!ls"));
        assert!(is_shell_line("  ! git status "));
        assert!(!is_shell_line("plain text"));
    }

    #[test]
    fn strips_bang_preserving_command_body() {
        assert_eq!(strip_bang("!ls -la"), "ls -la");
        assert_eq!(strip_bang("! git status"), "git status");
        assert_eq!(strip_bang("  !echo hi"), "echo hi");
        // Only ONE space after the bang is dropped; further indentation kept.
        assert_eq!(strip_bang("!  spaced"), " spaced");
        // Not shell-mode → unchanged.
        assert_eq!(strip_bang("plain"), "plain");
        assert_eq!(strip_bang_opt("plain"), None);
    }
}
