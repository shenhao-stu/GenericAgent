//! clipboard.rs — the copy/paste surface (OSC-52 → native), split out of `main.rs`
//! (ARCH Fix B). `perform_copy` is the single sink the `AppEvent::Copy` intent
//! resolves to; `export_action` performs the `/export` row choices. Every copy
//! outcome is surfaced as a notice (N1: never silent — the missing feedback was
//! the actual "can't copy" bug).

use crate::app::AppState;
use crate::i18n;
use crate::render;

/// Copy `text` via the clean logical-source path (P2): OSC 52 → native. RETURNS
/// the [`CopyResult`](render::copy::CopyResult) so callers can surface a visible
/// "Copied N bytes" notice — the result used to be swallowed, so a copy gave no
/// feedback at all (the root of the "can't copy" complaint).
pub(crate) fn copy_text(text: &str) -> render::copy::CopyResult {
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
pub(crate) fn notice_copy(app: &mut AppState, res: &render::copy::CopyResult, label: &str) {
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

/// Copy `text` and surface the outcome under `label` — the single sink the
/// `AppEvent::Copy` intent resolves to (the loop's `perform_actions` calls this).
pub(crate) fn perform_copy(app: &mut AppState, text: &str, label: &'static str) {
    let res = copy_text(text);
    notice_copy(app, &res, label);
}

/// Read the native clipboard for Ctrl+V (best-effort; `None` on any error).
pub(crate) fn read_clipboard() -> Option<String> {
    arboard::Clipboard::new().ok()?.get_text().ok()
}

/// Perform an `/export` action by row id: 0 = clip (last reply via OSC52), 1 = all
/// (whole transcript to clipboard), 2 = file (last reply to a cwd file).
pub(crate) fn export_action(app: &mut AppState, id: usize) {
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
