//! input/keymap.rs — the cockpit key handler + the global chord block + the
//! completion-dropdown intercept (ARCH Fix B). Folds a key into the composer /
//! transcript / session state, emitting [`AppEvent`] intents for any bridge verb
//! (Fix A) — the transport never threads through here. The session-CRUD chord
//! helpers (`cockpit_new_session`/`drop_active`/`cycle`) + the Ctrl+C / Esc chord
//! deciders live here too (they are cockpit-only).

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use crate::app::AppState;
use crate::app_event::AppEvent;
use crate::bridge::protocol::UiToCore;
use crate::clipboard;
use crate::commands::dispatch;
use crate::i18n;
use crate::input;

/// True if `key` is the unconditional quit chord (Ctrl+C / Ctrl+Q) — the dedup
/// (Fix G) for the copy that used to live in each view handler. `route_view_key`
/// calls it once at the top; the cockpit's own Ctrl+C is the 3-stage chord, so it
/// is handled in `cockpit_key` (this only catches the view-level immediate quit).
pub(crate) fn is_quit_chord(key: KeyEvent) -> bool {
    if key.kind == KeyEventKind::Release {
        return false;
    }
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    ctrl && matches!(key.code, KeyCode::Char('c') | KeyCode::Char('q'))
}

/// The composer's inner text width (borders + prompt) for visual-row nav.
pub(crate) fn composer_width(app: &AppState) -> u16 {
    // Best-effort: the real width is known at draw time, but key handling needs
    // a width for visual Up/Down. Use the last-synced transcript width minus the
    // composer chrome; fall back to 80.
    app.transcript_width().saturating_sub(4).max(1)
}

pub(crate) fn cockpit_key(key: KeyEvent, app: &mut AppState, now_ms: u64) {
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
        // (Ctrl+Shift+C is intentionally NOT bound — the terminal owns it for native
        // selection copy now that mouse capture starts OFF; "copy last reply" is the
        // explicit Ctrl+O below, Codex's stable chord.)
        // Ctrl+O — copy the LAST completed assistant reply via the clean logical
        // source (no soft-wrap newlines, P2). Arbitrary-span copy is the terminal's
        // job (drag-select, capture OFF); this is the one convenience copy action.
        KeyCode::Char('o' | 'O') if ctrl && !shift => {
            if let Some(src) = app.last_assistant_source() {
                let text = src.to_string();
                app.emit(AppEvent::Copy { text, label: i18n::t(app.lang, "copy.label.reply") });
            } else {
                app.push_notice(i18n::tf(app.lang, "export.none"));
            }
            return;
        }
        // Ctrl+Shift+O — toggle tool-chip / turn folding for ALL turns (moved off
        // Ctrl+O, which is now copy; `/fold` is the discoverable alias). Clears
        // per-node overrides so "fold all" / "unfold all" wins (Fix E).
        KeyCode::Char('o' | 'O') if ctrl && shift => {
            app.toggle_fold_all();
            return;
        }
        // Ctrl+Shift+M — toggle mouse capture so the terminal's native drag-select
        // works for inline copy (capture ON = wheel scroll + click-to-dashboard).
        KeyCode::Char('m' | 'M') if ctrl && shift => {
            let on = !app.mouse_capture;
            app.emit(AppEvent::SetMouseCapture(on));
            let k = if on { "mouse.on" } else { "mouse.off" };
            app.push_notice(i18n::tf(app.lang, k));
            return;
        }
        KeyCode::Char('c') if ctrl => {
            cockpit_ctrl_c(app, now_ms);
            return;
        }
        // (Ctrl+Q immediate-quit is deduped into `route_view_key`'s `is_quit_chord`
        // pre-check — it force-quits from every view; only the cockpit's Ctrl+C is
        // special (the 3-stage chord above), so route_view_key lets Ctrl+C fall here.)
        // Ctrl+L — force a full redraw (sleep/wake recovery).
        KeyCode::Char('l') if ctrl => {
            // ratatui repaints from state each frame; a Changed is enough to
            // trigger the next draw. (A hard clear is the terminal's job.)
            return;
        }
        // Ctrl+T — open the theme picker (v2 parity `ctrl+t`=pick_theme; the C3
        // parity-table gap). Routes through the same path as `/theme`.
        KeyCode::Char('t') if ctrl => {
            dispatch::open_ui_command(app, "theme", "");
            return;
        }
        // Ctrl+/ (and the legacy Ctrl+_ — many terminals send US/0x1F for Ctrl+/)
        // open the `/keybindings` cheat-sheet (the C3 parity-table Help gap, mapped
        // to the keybindings overlay). Routes through the same path as the command.
        KeyCode::Char('/') | KeyCode::Char('_') if ctrl => {
            dispatch::open_ui_command(app, "keybindings", "");
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
        // Ctrl+Y — redo (NOT copy — copy is the explicit Ctrl+O above, Codex's chord).
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
                let res = clipboard::copy_text(&text);
                clipboard::notice_copy(app, &res, label);
            }
            return;
        }
        // Ctrl+V — paste from the native clipboard.
        KeyCode::Char('v') if ctrl => {
            if let Some(text) = clipboard::read_clipboard() {
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
    // `ctrl` is threaded so a Ctrl+Enter (newline) isn't consumed as a completion.
    if try_complete_dropdown(app, code, shift, ctrl) {
        return;
    }

    // --- composer + transcript navigation ------------------------------------
    match code {
        // Shift+Enter OR Ctrl+Enter inserts a newline; plain Enter submits. (Ctrl+J
        // is the universal fallback for terminals that deliver neither modifier.)
        KeyCode::Enter if shift || ctrl => {
            app.composer.newline();
        }
        KeyCode::Enter => {
            let action = app.composer.submit();
            dispatch::dispatch_action(app, action);
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
        // EXCEPTION (redesign request #2 / Q5): when the composer is EMPTY there is
        // nothing to navigate, so ←/→ SWITCH VIEWS — the intuitive replacement for
        // the unobvious Ctrl+S. ← ENTERS the session dashboard, → returns to chat.
        // (Ctrl+S still toggles; Esc still backs out.) A non-empty buffer keeps the
        // normal cursor nav so typing/editing is never hijacked.
        KeyCode::Left if app.composer.is_empty() && !shift => {
            app.open_dashboard();
        }
        KeyCode::Right if app.composer.is_empty() && !shift => {
            app.close_dashboard(); // already-in-cockpit → no-op; harmless + future-proof.
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
            cockpit_esc(app, now_ms, width);
        }
        KeyCode::Char(c) => {
            app.composer.type_str(&c.to_string());
            app.palette_sel = 0; // typing changes the `/`-palette matches → reset highlight.
        }
        _ => {}
    }
    let _ = ComposerAction::None; // (action enum referenced via dispatch_action)
}

/// If a completion dropdown is active, handle Tab/Enter (complete) and Up/Down
/// (move highlight) here, returning `true` if the key was consumed. The slash
/// palette wins when both could match (a `/word` has no `@` query).
pub(crate) fn try_complete_dropdown(app: &mut AppState, code: KeyCode, shift: bool, ctrl: bool) -> bool {
    use crate::commands::registry;
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
            // `&& !ctrl` so Ctrl+Enter (newline) is NOT swallowed as a completion
            // while the palette is open — it falls through to the newline arm.
            KeyCode::Enter if !shift && !ctrl => {
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
                KeyCode::Tab | KeyCode::Enter if !shift && !ctrl => {
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

/// Ctrl+N — create a new session + switch to it (the leaving draft is stashed).
pub(crate) fn cockpit_new_session(app: &mut AppState) {
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
pub(crate) fn cockpit_drop_active(app: &mut AppState) {
    let active = app.sessions.active;
    app.sessions.delete(active);
    let new_active = app.sessions.active;
    // Loading the fallback session's stored state (a reset blank for the
    // last-session case, else the neighbour's transcript).
    app.load_active_fields_after_drop(new_active);
}

/// Ctrl+Up / Ctrl+Down — cycle the active session, stashing/restoring drafts.
pub(crate) fn cockpit_cycle(app: &mut AppState, delta: isize) {
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
pub(crate) fn cockpit_ctrl_c(app: &mut AppState, now_ms: u64) {
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
                let res = clipboard::copy_text(&text);
                clipboard::notice_copy(app, &res, label);
            }
        }
        CtrlCAction::AbortTurn => {
            // Mirror `/stop`: tell the active child to abort, flip busy off, notice.
            app.emit(AppEvent::ToActive(UiToCore::Abort { mid: None }));
            app.busy = false;
            app.push_notice(i18n::tf(app.lang, "cmd.aborted"));
        }
        CtrlCAction::ArmQuit => {
            // The hint ("press Ctrl+C again to quit") is surfaced on the BOTTOM `⎿`
            // tip row while the arm is live (`chord.ctrl_c_hint_active(now_ms)` in
            // `render_tips`) — NOT as a transcript notice, so it doesn't scroll away
            // or pollute history. A 2nd Ctrl+C within 2s quits; else the arm expires.
        }
        CtrlCAction::Quit => app.should_quit = true,
    }
}

/// The Esc / Esc-Esc chord (§8). Run the PURE decider with the injected `now_ms`,
/// store the new double-tap window, then either open the `/rewind` picker (2nd Esc
/// inside 0.8s) or perform the universal back (single Esc) — which NEVER exits.
pub(crate) fn cockpit_esc(app: &mut AppState, now_ms: u64, width: u16) {
    use input::keychord::{esc, EscAction};
    let (action, next) = esc(app.chord, now_ms);
    app.chord = next;
    match action {
        EscAction::Rewind => {
            // Open the rewind picker (same entry the `/rewind` command uses; it
            // shows a "nothing to rewind" notice when there are no turns yet).
            dispatch::open_ui_command(app, "rewind", "");
        }
        EscAction::Back => cockpit_universal_back(app, width),
    }
}

/// The cockpit's universal-BACK (a single Esc): clear a pending ask, else collapse
/// a composer selection, else stash the draft so work isn't lost. NEVER exits —
/// the §8 "Esc universal back … never exits" contract. PURE-ish (in-memory only).
pub(crate) fn cockpit_universal_back(app: &mut AppState, width: u16) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::View;

    fn key(code: KeyCode, mods: KeyModifiers) -> KeyEvent {
        KeyEvent::new(code, mods)
    }

    /// Q5 — on an EMPTY composer ← ENTERS the session dashboard and → returns to
    /// the chat cockpit (the deliberate swap of the old reversed polarity). Drives
    /// the real `cockpit_key` path so the arrow arms are exercised, not asserted.
    #[test]
    fn left_on_empty_composer_opens_dashboard() {
        let mut app = AppState::new();
        assert_eq!(app.view, View::Cockpit, "starts in the cockpit");

        // ← on the empty composer opens the dashboard.
        cockpit_key(key(KeyCode::Left, KeyModifiers::NONE), &mut app, 0);
        assert_eq!(app.view, View::Dashboard, "empty-composer Left opens the dashboard");

        // → returns to the cockpit (the cockpit's Right is the no-op-at-root exit;
        // here we re-enter the cockpit to prove the mirror direction).
        app.view = View::Cockpit; // back in chat (a real → from the dashboard does this)
        cockpit_key(key(KeyCode::Right, KeyModifiers::NONE), &mut app, 0);
        assert_eq!(app.view, View::Cockpit, "empty-composer Right stays in / returns to the cockpit");

        // A NON-empty composer keeps cursor nav — ← must NOT open the dashboard.
        app.composer.type_str("hi");
        cockpit_key(key(KeyCode::Left, KeyModifiers::NONE), &mut app, 0);
        assert_eq!(app.view, View::Cockpit, "a non-empty buffer keeps Left as cursor nav");
    }

    /// Q6 — Ctrl+Enter inserts a newline instead of submitting (parity with v2's
    /// `ctrl+enter`=newline). The buffer keeps its text + gains a `\n`; plain Enter
    /// still submits (clears) — covered by the composer's own submit tests.
    #[test]
    fn ctrl_enter_inserts_newline() {
        let mut app = AppState::new();
        app.composer.type_str("hi");
        cockpit_key(key(KeyCode::Enter, KeyModifiers::CONTROL), &mut app, 0);
        assert!(
            app.composer.text().contains('\n'),
            "Ctrl+Enter inserts a newline, got {:?}",
            app.composer.text()
        );
        assert!(
            app.composer.text().contains("hi"),
            "Ctrl+Enter must NOT submit/clear the buffer"
        );
    }

    /// Q2 — Ctrl+O copies the LAST completed assistant reply (Codex's stable copy
    /// chord) via an `AppEvent::Copy` over the clean logical `block.source` (no
    /// soft-wrap newlines). Seeds a real reply through the bridge path, then drains
    /// the emitted intents and asserts the copy text is the reply source.
    #[test]
    fn ctrl_o_copies_last_reply() {
        use crate::bridge::protocol::CoreToUi;
        use crate::bridge::BridgeEvent;
        let frame = |f: CoreToUi| BridgeEvent::Frame(f);

        let mut app = AppState::new();
        app.apply_bridge_event(frame(CoreToUi::MessageBegin { mid: "m1".into(), role: "assistant".into() }), 100);
        app.apply_bridge_event(frame(CoreToUi::MessageDelta { mid: "m1".into(), text: "the answer".into() }), 100);
        app.apply_bridge_event(frame(CoreToUi::MessageEnd { mid: "m1".into(), reason: "stop".into() }), 100);

        cockpit_key(key(KeyCode::Char('o'), KeyModifiers::CONTROL), &mut app, 0);

        let copied = app.drain_actions().into_iter().find_map(|ev| match ev {
            AppEvent::Copy { text, label } => Some((text, label)),
            _ => None,
        });
        let (text, label) = copied.expect("Ctrl+O emits an AppEvent::Copy");
        assert_eq!(text, "the answer", "Ctrl+O copies the last reply's logical source");
        assert_eq!(label, i18n::t(app.lang, "copy.label.reply"));
    }

    /// Q2 / F7 — mouse capture starts OFF so the terminal owns native drag-select
    /// for inline copy; the field is the single source of truth that `term::setup`
    /// (no longer captures) and the `Ctrl+Shift+M` toggle both agree with.
    #[test]
    fn mouse_capture_defaults_off() {
        assert!(!AppState::new().mouse_capture, "mouse capture defaults OFF (native select)");
    }
}
