//! input/views.rs — the per-VIEW key router + the full-screen panel handlers
//! (dashboard §6 / workflows §7) + the modal-overlay key handler (§3 overlay
//! stack) + the picker-preview helpers (ARCH Fix B). `route_view_key` forks on the
//! active view; the quit chord is deduped through `keymap::is_quit_chord` (Fix G).
//! Bridge verbs leave as [`AppEvent`] intents (Fix A).

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use crate::app::{self, AppState, View};
use crate::app_event::AppEvent;
use crate::bridge::protocol::UiToCore;
use crate::components;
use crate::commands::dispatch;
use crate::i18n;
use crate::input::keymap;
use crate::theme;
use crate::workflow;

/// Route a key to the active VIEW's handler (cockpit / dashboard / workflows).
/// Centralizes the per-view fork so the modal-overlay branches above don't each
/// re-list the views. Fix G: the quit chord is deduped through `is_quit_chord` —
/// Ctrl+Q force-quits from EVERY view; the dashboard/workflows panels also quit on
/// Ctrl+C, but the cockpit's Ctrl+C is the 3-stage chord, so it falls through to
/// `cockpit_key` there.
pub(crate) fn route_view_key(key: KeyEvent, app: &mut AppState, now_ms: u64) {
    if keymap::is_quit_chord(key) {
        let ctrl_c = matches!(key.code, KeyCode::Char('c'));
        if app.view != View::Cockpit || !ctrl_c {
            app.should_quit = true;
            return;
        }
    }
    match app.view {
        View::Dashboard => handle_dashboard_key(key, app),
        View::Workflows => handle_workflows_key(key, app),
        View::Cockpit => keymap::cockpit_key(key, app, now_ms),
    }
}

/// The full-screen `/workflows` PANEL key handler (§7). Navigates the workflow
/// tree, toggles the render style, opens the per-node detail overlay, and fires
/// node action verbs. When the detail overlay is open the keys drive its action
/// menu instead. `Esc` closes the detail overlay (if open) else returns to the
/// cockpit (never exits the app). The watcher keeps refreshing the snapshot in the
/// background — these keys only move the cursor / fire one-shot actions.
fn handle_workflows_key(key: KeyEvent, app: &mut AppState) {
    let KeyEvent { code, modifiers, kind, .. } = key;
    if kind == KeyEventKind::Release {
        return;
    }
    let ctrl = modifiers.contains(KeyModifiers::CONTROL);

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
fn handle_dashboard_key(key: KeyEvent, app: &mut AppState) {
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
            // Tree gesture, mirroring the cockpit (Q5): ← DRILLS — collapse an
            // EXPANDED category; otherwise stay (→ is the exit now). The cockpit's
            // empty-composer ← opened the dashboard, so inside it ← keeps going "in".
            match app.sessions.selected_category() {
                Some(cat) if !app.sessions.is_collapsed(cat) => app.sessions.toggle_category(cat),
                _ => { /* stay; Right exits */ }
            }
        }
        KeyCode::Right => {
            // → expands a COLLAPSED category; otherwise returns to the chat cockpit
            // (the mirror of the cockpit's empty-composer → that closes the dashboard).
            match app.sessions.selected_category() {
                Some(cat) if app.sessions.is_collapsed(cat) => app.sessions.toggle_category(cat),
                _ => app.close_dashboard(),
            }
        }

        // Enter: if the bottom new-session input has text → create a new session
        // seeded with it; else open/switch into the selected session. (A header
        // row with no input toggles its collapse, for a sensible Enter.)
        KeyCode::Enter => {
            let seed = app.sessions.new_session_input.trim().to_string();
            if !seed.is_empty() {
                dashboard_new_session(app, seed);
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
                    dashboard_quick_reply(app, id, seed);
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
            dashboard_new_session(app, seed);
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
/// live-preview theme); Enter applies the selection. Ctrl+C / Ctrl+Q still quit.
pub(crate) fn handle_overlay_key(key: KeyEvent, app: &mut AppState) {
    use app::Overlay;
    let KeyEvent { code, modifiers, kind, .. } = key;
    if kind == KeyEventKind::Release {
        return;
    }
    let ctrl = modifiers.contains(KeyModifiers::CONTROL);
    // Ctrl+C / Ctrl+Q quit even from an overlay (Fix G: shared `is_quit_chord`).
    if keymap::is_quit_chord(key) {
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
                KeyCode::Enter => dispatch::apply_picker(app, picker),
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
            KeyCode::Enter => dispatch::apply_ask_user(app, ask),
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
                    dispatch::apply_scheduler(app, &mut sched);
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
            // Editing applies the cheap META filter at once; the lazy content grep
            // is deferred to `picker.tick()` (driven by `app.tick()`) after typing
            // pauses, so the spinner_tick is the debounce clock.
            let now_tick = app.spinner_tick;
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
                    picker.backspace(now_tick);
                    app.overlay = Some(Overlay::Continue(picker));
                }
                KeyCode::Enter => {
                    if let Some(path) = picker.selected_path() {
                        // Restore via the existing restore path: the bridge's
                        // handle_restore loads the log into backend.history.
                        app.emit(AppEvent::ToActive(
                            UiToCore::Command { name: "restore".into(), args: path },
                        ));
                        app.push_notice(i18n::tf(app.lang, "continue.restoring"));
                        // overlay taken → closed.
                    } else {
                        app.overlay = Some(Overlay::Continue(picker));
                    }
                }
                KeyCode::Char(c) if !ctrl => {
                    picker.type_char(c, now_tick);
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
                dispatch::apply_effort(app, slider.selected());
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
            if let Some(t) = theme::by_name(&dispatch::strip_theme_label(&item.label)) {
                app.theme = t;
            }
        }
        PickerKind::Emoji => dispatch::apply_emoji_choice(app, item.id),
        PickerKind::Language => {
            // Live-preview the language: a full repaint happens on the next frame.
            app.set_language(dispatch::lang_for_picker_id(item.id));
        }
        _ => {}
    }
}

/// Create a new session from the dashboard, optionally seeded with a prompt that
/// is submitted to its (lazy-spawned) child, then switch into it. Clears the
/// bottom input.
fn dashboard_new_session(app: &mut AppState, seed: String) {
    use crate::commands::registry::split_command;
    app.snapshot_active_into_map();
    app.sessions.new_session_input.clear();
    let new_id = app.sessions.new_session(None);
    // Switch the live fields onto the new (empty) session — it is now `active`, so
    // the seed routes to it via `AppEvent::ToActive`.
    app.load_active_session_after_structural_change(new_id);
    if !seed.trim().is_empty() {
        // Echo the seed as the user's first message + submit to the child.
        app.push_user(seed.clone());
        let expanded = app.expand_at_paths(&seed);
        let frame = if expanded.trim_start().starts_with('/') {
            // Reuse the registry's `/name args` split (Fix G: kills the inline parse).
            let (name, args) = split_command(&expanded);
            UiToCore::Command { name, args }
        } else {
            UiToCore::Submit { text: expanded, images: None }
        };
        app.emit(AppEvent::ToActive(frame));
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
fn dashboard_quick_reply(app: &mut AppState, id: u64, text: String) {
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
    app.emit(AppEvent::ToSession(id, UiToCore::Submit { text, images: None }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::continue_picker::{ContinuePicker, ContinueSession, GREP_DEBOUNCE_TICKS};
    use std::path::PathBuf;

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    /// Slice-9 honest check on the LIVE key path (`handle_overlay_key`, the same fn
    /// the event loop calls): driving the `/continue` overlay with real key events
    /// (1) filters the META immediately while the content grep only contributes after
    /// the debounce TICK, and (2) Enter still routes `Command{restore}` into the
    /// emitted intent the bus performs (→ `ga_bridge.handle_restore`).
    #[test]
    fn continue_overlay_two_stage_and_restore_routing() {
        let session = ContinueSession {
            path: PathBuf::from("/temp/model_responses/model_responses_777.txt"),
            mtime: 100,
            preview: "fix the login bug".into(),
            rounds: 7,
        };
        let mut app = AppState::new();
        app.overlay = Some(app::Overlay::Continue(ContinuePicker::new(vec![session])));

        // Type a term that is NOT in the META (basename/preview) — only the live
        // read_head_window grep over the real (absent) file could match it, and it
        // can't, but the point is the STAGE-1 result must be computed with no grep.
        for c in "zzqq".chars() {
            handle_overlay_key(press(KeyCode::Char(c)), &mut app);
        }
        let pick = match app.overlay.as_ref() {
            Some(app::Overlay::Continue(p)) => p,
            _ => panic!("overlay should still be the Continue picker"),
        };
        // STAGE 1: META-only filtered it out immediately AND a grep is armed (pending),
        // i.e. typing did not run the content grep inline.
        assert_eq!(pick.matches(), 0, "META filter applied immediately");
        assert!(pick.grep_pending(), "content grep is deferred, not run on the keystroke");

        // STAGE 2: ticking past the debounce window runs the grep (here it reads the
        // real — missing — file, so the match set stays 0, but the grep disarms,
        // proving the two-stage hand-off fired off the keystroke path).
        for _ in 0..=GREP_DEBOUNCE_TICKS {
            app.tick();
        }
        let pick = match app.overlay.as_ref() {
            Some(app::Overlay::Continue(p)) => p,
            _ => panic!("overlay still open"),
        };
        assert!(!pick.grep_pending(), "the debounced grep fired on a later tick");

        // Now clear the query so the single session matches again, then Enter.
        for _ in 0..4 {
            handle_overlay_key(press(KeyCode::Backspace), &mut app);
        }
        assert!(app.overlay.is_some(), "overlay open before Enter");
        handle_overlay_key(press(KeyCode::Enter), &mut app);

        // Enter consumed (closed) the overlay AND emitted the restore intent.
        assert!(app.overlay.is_none(), "Enter closed the picker");
        let restored = app.drain_actions().into_iter().any(|ev| {
            matches!(ev, AppEvent::ToActive(UiToCore::Command { name, args })
                if name == "restore"
                    && args == "/temp/model_responses/model_responses_777.txt")
        });
        assert!(restored, "Enter routes Command{{restore}} → handle_restore path");
    }
}
