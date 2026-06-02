//! commands/dispatch.rs — the `/command` + composer-Enter dispatch layer (ARCH Fix
//! B). Classifies a submitted line via the registry and performs the effect:
//! opening a dedicated overlay/picker, emitting a core-forward `Command` intent,
//! running an in-app action, or surfacing a "did you mean /x?" breadcrumb. Every
//! bridge verb leaves as an `AppEvent` (Fix A) — no transport is threaded here.

use crate::app::{self, AppState};
use crate::app_event::AppEvent;
use crate::bridge::protocol::UiToCore;
use crate::clipboard;
use crate::components;
use crate::flavor;
use crate::i18n;
use crate::input;
use crate::theme;

/// Act on a [`input::ComposerAction`] produced by Enter: submit to the ACTIVE
/// session's bridge (spawning it lazily, §6), run a `!shell` line, etc.
pub(crate) fn dispatch_action(app: &mut AppState, action: input::ComposerAction) {
    use input::ComposerAction;
    match action {
        ComposerAction::None | ComposerAction::Changed | ComposerAction::Redraw
        | ComposerAction::ToggleFold | ComposerAction::Escape => {}
        ComposerAction::Submit { text } => {
            let expanded = app.expand_at_paths(&text);
            // A leading `/` is a slash command — classify + route it through the
            // registry (every §4 name resolves; unknowns get a did-you-mean). A
            // plain message echoes into the transcript + submits to the bridge; a
            // dead child surfaces `notice.bridge.not_connected` from the bus.
            if expanded.trim_start().starts_with('/') {
                dispatch_slash(app, &expanded);
            } else {
                app.push_user(text.clone());
                app.emit(AppEvent::ToActive(UiToCore::Submit { text: expanded, images: None }));
            }
        }
        ComposerAction::Shell { cmd } => run_shell_line(app, &cmd),
    }
}

/// Route a submitted `/command args` line (checklist §4). Classifies it via the
/// registry into a [`commands::SlashOutcome`] and performs the effect: open a
/// dedicated UI overlay, core-forward a `Command` frame, run an in-app action, or
/// surface a "did you mean /x?" breadcrumb for an unknown command.
pub(crate) fn dispatch_slash(app: &mut AppState, line: &str) {
    use crate::commands::SlashOutcome;
    match crate::commands::classify_slash(line) {
        SlashOutcome::OpenUi { name, args } => open_ui_command(app, &name, &args),
        SlashOutcome::Forward { name, args } => {
            // Echo the user's command in the transcript, then forward it as a
            // Command frame (the GA core intercepts the leading-slash itself). A
            // dead/unspawnable child surfaces `notice.bridge.not_connected` from
            // the bus when the `ToActive` intent is performed.
            app.push_user(format!("/{name}{}", if args.is_empty() { String::new() } else { format!(" {args}") }));
            // §4: /goal /hive /conductor are "fwd + a /workflows tile" — forwarding
            // kicks off the orchestration AND the panel shows its live progress. So
            // after forwarding one of these, open the /workflows panel so the user
            // watches the tree the watcher will populate (the others just forward).
            let opens_panel = matches!(name.as_str(), "goal" | "hive" | "conductor");
            app.emit(AppEvent::ToActive(UiToCore::Command { name, args }));
            if opens_panel {
                app.open_workflows();
            }
        }
        SlashOutcome::App { name, args } => app_command(app, &name, &args),
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
pub(crate) fn open_ui_command(app: &mut AppState, name: &str, args: &str) {
    use components::picker::{PickItem, Picker, PickerKind};
    match name {
        "help" => app.open_overlay(app::Overlay::Help),
        "keybindings" => app.open_overlay(app::Overlay::Keybindings),
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
                    app.emit(AppEvent::ToActive(UiToCore::SwitchLlm { n: idx + 1 }));
                    app.push_notice(i18n::tf(app.lang, "llm.switching"));
                    return;
                }
            }
            // Open the picker with a "querying…" placeholder, then ask the bridge
            // for the model list; the LlmList frame fills the rows in place (N3).
            let placeholder = vec![PickItem::new(0, i18n::t(app.lang, "llm.querying"))];
            app.open_picker(Picker::new(PickerKind::Llm, placeholder), None);
            app.emit(AppEvent::ToActive(UiToCore::ListLlms));
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
        "pets" => {
            // Pet-style picker (preview on move). Rows = the 5 pet styles + Off, each
            // on a stable id; spinner style is no longer user-configurable here (Slice 6).
            let items = emoji_picker_items(app);
            let backup = app.theme.clone(); // pet previews don't touch theme; backup harmless.
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
                Some(level) if !args.trim().is_empty() => apply_effort(app, level),
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
                "clip" => return clipboard::export_action(app, 0),
                "all" => return clipboard::export_action(app, 1),
                "file" => return clipboard::export_action(app, 2),
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
            // `/continue N` is the non-interactive form (v2 `tuiapp_v2.py:3897-3903`):
            // a bare 1-based integer restores the N-th most-recent session directly,
            // skipping the picker.
            let sessions = components::continue_picker::list_sessions(&app.repo_root, None);
            if sessions.is_empty() {
                app.push_notice(i18n::tf(app.lang, "continue.empty"));
                return;
            }
            if let Ok(n) = args.trim().parse::<usize>() {
                match n.checked_sub(1).and_then(|i| sessions.get(i)) {
                    Some(s) => {
                        let path = s.path.to_string_lossy().into_owned();
                        app.emit(AppEvent::ToActive(UiToCore::Command {
                            name: "restore".into(),
                            args: path,
                        }));
                        app.push_notice(i18n::tf(app.lang, "continue.restoring"));
                    }
                    None => app.push_notice(i18n::tf(app.lang, "continue.no_match")),
                }
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
            // overlay whose step advances on Enter. Reads the REAL reflect modes
            // (reflect/*.py) + cron tasks (sche_tasks/*.json); falls back to a
            // representative default set only when that tree is empty (§7 Q10).
            let mut tasks = components::scheduler::discover_tasks(&app.repo_root);
            if tasks.is_empty() {
                tasks = components::scheduler::default_tasks();
            }
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
            app.emit(AppEvent::ToActive(UiToCore::BtwAsk { ask_id, text: q }));
        }
        _ => app.push_notice(format!("/{name} {}", i18n::t(app.lang, "cmd.not_wired"))),
    }
}

/// Run a §4 **app** command (handled entirely in-app).
pub(crate) fn app_command(app: &mut AppState, name: &str, args: &str) {
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
            app.emit(AppEvent::ToActive(UiToCore::Abort { mid: None }));
            app.busy = false;
            app.push_notice(i18n::tf(app.lang, "cmd.aborted"));
        }
        "cost" => app.open_overlay(app::Overlay::Cost),
        "fold" => {
            // Fold / unfold ALL completed tool chips + turns (clears per-node
            // overrides — "fold all" wins, Fix E). The discoverable alias for the
            // Ctrl+Shift+O chord (Ctrl+O is now the clean-copy chord, Q2).
            app.toggle_fold_all();
        }
        "mouse" => {
            // S1 toggle: flip native ↔ interactive mouse mode.
            let on = !app.mouse_capture;
            app.emit(AppEvent::SetMouseCapture(on));
            let notice = if on {
                "mouse: click (expand/collapse ▸/▾)"
            } else {
                "mouse: select (drag to copy)"
            };
            app.push_notice(notice.to_string());
        }
        "new" => {
            let seed = args.trim();
            input::keymap::cockpit_new_session(app);
            if !seed.is_empty() {
                app.sessions.rename(app.sessions.active, seed.to_string());
            }
        }
        "close" => input::keymap::cockpit_drop_active(app),
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
            app.emit(AppEvent::ToActive(UiToCore::Command { name: "restore".into(), args: args.to_string() }));
            app.push_notice(i18n::tf(app.lang, "cmd.restoring"));
        }
        "reload-keys" => {
            app.emit(AppEvent::ToActive(UiToCore::Command { name: "reload-keys".into(), args: String::new() }));
            app.push_notice(i18n::tf(app.lang, "cmd.reloading_keys"));
        }
        _ => app.push_notice(format!("/{name} {}", i18n::t(app.lang, "cmd.not_handled"))),
    }
}

/// Apply the picker's selection on Enter (the overlay was already taken; this
/// performs the §4 action for its [`PickerKind`] then closes the overlay).
pub(crate) fn apply_picker(app: &mut AppState, picker: components::picker::Picker) {
    use components::picker::PickerKind;
    match picker.kind {
        PickerKind::Llm => {
            if let Some(idx) = picker.selected_id() {
                // The picker id is the 0-based LLM index; protocol SwitchLlm is
                // 1-based (the `llm_picker_maps_index` contract).
                app.emit(AppEvent::ToActive(UiToCore::SwitchLlm { n: idx as u32 + 1 }));
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
                clipboard::export_action(app, id);
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
                app.emit(AppEvent::ToActive(UiToCore::Rewind { n: back as u32 }));
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
                    app.emit(AppEvent::ToActive(UiToCore::Command { name: "continue".into(), args: id.to_string() }));
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
pub(crate) fn apply_scheduler(app: &mut AppState, sched: &mut components::scheduler::Scheduler) {
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
    app.emit(AppEvent::ToActive(UiToCore::Command { name: "scheduler".into(), args: arg }));
    app.push_notice(i18n::tf(app.lang, "scheduler.applied"));
}

/// Apply the ask_user card on Enter: resolve the answer text and send it back as an
/// `Answer` frame, clearing the pending ask + the overlay.
pub(crate) fn apply_ask_user(app: &mut AppState, ask: components::picker::AskUserPicker) {
    match ask.resolve_answer() {
        Some(text) => {
            let frame = UiToCore::Answer {
                ask_id: ask.ask_id.clone(),
                option_id: None,
                text: Some(text.clone()),
            };
            app.emit(AppEvent::ToActive(frame));
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

/// APPLY a chosen reasoning-effort level (redesign_cc.md §3): remember it for the
/// spinner suffix AND forward `/session.reasoning_effort=<backend>` (max→xhigh) to
/// the active bridge via the existing slash-forward `Command` path so the GA core
/// hot-reloads it live (`setattr(backend, "reasoning_effort", v)`, takes effect next
/// turn — no restart). Shared by `/effort <level>` (direct) and the slider's Enter.
pub(crate) fn apply_effort(app: &mut AppState, level: app::effort::ReasoningEffort) {
    app.set_reasoning_effort(level);
    // Forward as Command{name:"session.reasoning_effort=<v>", args:""} — the bridge
    // reconstructs "/session.reasoning_effort=<v>" and the core intercepts it. A
    // dead/unspawnable child surfaces `notice.bridge.not_connected` from the bus.
    app.emit(AppEvent::ToActive(effort_command_frame(level)));
    // Confirm in the transcript (mirrors CC: a "thinking <level>" confirmation line).
    // Show the slider LABEL (so `max` reads "max"); note the backend value if it
    // differs (max→xhigh) so the user sees what the backend actually got.
    let note = if level.label() == level.backend_value() {
        format!("effort · {} (next turn)", level.label())
    } else {
        format!("effort · {} → {} (next turn)", level.label(), level.backend_value())
    };
    app.push_notice(note);
}

/// Build the bridge frame that forwards a `/effort` level (redesign_cc.md §3): a
/// `Command{name:"session.reasoning_effort=<backend>", args:""}` — the bridge
/// reconstructs `"/" + name` → `/session.reasoning_effort=<backend>` and the GA core
/// hot-reloads it via `setattr(backend, "reasoning_effort", v)`. PURE (the routing
/// contract the `effort_forwards_session_command` test pins, without spawning a
/// child).
pub(crate) fn effort_command_frame(level: app::effort::ReasoningEffort) -> UiToCore {
    UiToCore::Command { name: level.command_name(), args: String::new() }
}

/// A fresh, short, unique-enough id for a `/btw` side-question card (ties the
/// `BtwAsk` request to the `BtwAnswer` reply). Derived from a monotonic process
/// clock + a counter so it is collision-free within a session without a uuid dep.
pub(crate) fn new_ask_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    format!("btw-{t:x}-{n:x}")
}

/// Map a `/language` picker row id onto a [`i18n::Lang`] (the picker rows are built
/// in `i18n::Lang::all()` order, so the id is the ordinal). PURE.
pub(crate) fn lang_for_picker_id(id: usize) -> i18n::Lang {
    i18n::Lang::all().get(id).copied().unwrap_or(i18n::Lang::En)
}

/// The theme NAME in a `/theme` picker row label (the label IS the bare name).
pub(crate) fn strip_theme_label(label: &str) -> String {
    label.trim().to_string()
}

/// Apply an `/emoji` picker choice by row id: spinner glyphs (braille/arc/pulse at
/// ids 100/101/102) OR pet faces (bear/cat/dot/unicode/fox at ids 0..=4, Off = 5).
/// ONE mutually-exclusive companion — drives the spinner lead AND the animated tab.
pub(crate) fn apply_emoji_choice(app: &mut AppState, id: usize) {
    use flavor::{CompanionKind, PetStyle, SpinnerStyle};
    app.companion = match id {
        100 => CompanionKind::Spinner(SpinnerStyle::Braille),
        101 => CompanionKind::Spinner(SpinnerStyle::Arc),
        102 => CompanionKind::Spinner(SpinnerStyle::Pulse),
        0..=4 => CompanionKind::Pet(PetStyle::all()[id]),
        5 => CompanionKind::Pet(PetStyle::Off),
        _ => return,
    };
}

/// Build the `/emoji` picker rows: 3 spinner glyphs (braille/arc/pulse, ids
/// 100..=102, braille first/default) + 5 pet faces (bear/cat/dot/unicode/fox, ids
/// 0..=4) + Off (id 5). ONE mutually-exclusive companion; each row marks `current`
/// against `app.companion`. PURE-ish.
pub(crate) fn emoji_picker_items(app: &AppState) -> Vec<components::picker::PickItem> {
    use components::picker::PickItem;
    use flavor::{CompanionKind, PetStyle, SpinnerStyle};
    let spin = i18n::t(app.lang, "emoji.spinner");
    let pet = i18n::t(app.lang, "emoji.pet");
    let off = i18n::t(app.lang, "emoji.off");
    let mut items: Vec<PickItem> = Vec::new();
    // Spinner glyphs first (braille is the default companion).
    for (i, style) in [SpinnerStyle::Braille, SpinnerStyle::Arc, SpinnerStyle::Pulse]
        .iter()
        .enumerate()
    {
        items.push(
            PickItem::new(100 + i, format!("{spin} · {}", style.name()))
                .with_detail(style.glyph(0).to_string())
                .current(app.companion == CompanionKind::Spinner(*style)),
        );
    }
    // Pet faces.
    for (i, style) in PetStyle::all().iter().enumerate() {
        items.push(
            PickItem::new(i, format!("{pet} · {}", style.name()))
                .with_detail(flavor::pet_face(*style, 0, 0).to_string())
                .current(app.companion == CompanionKind::Pet(*style)),
        );
    }
    items.push(
        PickItem::new(5, format!("{pet} · {off}"))
            .current(app.companion == CompanionKind::Pet(PetStyle::Off)),
    );
    items
}

/// Build the `/rewind` picker rows: the last ~20 USER turns, newest first. Each
/// row's `id` is the FULL-transcript index of that user message (so truncating the
/// transcript at `id` removes that turn and everything after — the rewind point);
/// the label shows the turn ordinal + how many turns back. PURE-ish.
pub(crate) fn rewind_picker_items(app: &AppState) -> Vec<components::picker::PickItem> {
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

/// The number of REAL (user) turns dropped by rewinding the transcript to
/// full-index `truncate_at` (the `/rewind` picker's row id). It's the count of
/// USER messages at or after `truncate_at` — i.e. the turns that `transcript
/// .truncate(truncate_at)` removes — clamped to ≥1 (picking a turn always drops at
/// least that turn). PURE — the `rewind_truncation_count` deliverable pins it.
pub(crate) fn rewind_real_turns_from(transcript: &[app::Block], truncate_at: usize) -> usize {
    transcript
        .iter()
        .skip(truncate_at)
        .filter(|b| matches!(b.role, app::Role::User))
        .count()
        .max(1)
}

/// Run a `!cmd` host-shell line: execute (30s timeout), echo the output into the
/// transcript as a SYSTEM block, and seed the agent's context via a
/// `Command{name:"shell"}` frame (ga_bridge.py stashes it into `_intervene` —
/// WITHOUT spending a turn). N1: a spawn failure still surfaces as a system line.
pub(crate) fn run_shell_line(app: &mut AppState, cmd: &str) {
    use crate::commands::{format_shell_block, format_shell_note, run_shell};
    let cwd = app.repo_root.clone();
    let result = run_shell(cmd, &cwd);
    // Echo into the transcript (what the user sees).
    app.push_system(format_shell_block(&result));
    // Seed the agent context (what the model gets) — no turn spent. Routed to the
    // ACTIVE session's child (lazy-spawned).
    let note = format_shell_note(&result);
    app.emit(AppEvent::ToActive(UiToCore::Command {
        name: "shell".to_string(),
        args: note,
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::protocol::CoreToUi;
    use crate::bridge::BridgeEvent;

    /// ROUND-5 HONEST CHECK: the UNIFIED `/emoji` companion picker (spinner ⊕ pet,
    /// pick-one) + the animated tab title. `/pets` is GONE; `/emoji` is the PRIMARY
    /// command; the picker offers 3 spinner glyphs (ids 100..=102) + 5 pet faces
    /// (0..=4) + Off (5); a pet selection drives `companion`; and `terminal_title()`
    /// ANIMATES across ticks (never "NativeClaude").
    #[test]
    fn emoji_unified_picker_and_animated_title() {
        use crate::flavor::{CompanionKind, PetStyle, SpinnerStyle, PET_TICKS_PER_FRAME};

        // /pets removed; /emoji is the PRIMARY command (not an alias).
        assert!(crate::commands::resolve("pets").is_none(), "/pets must no longer resolve");
        let emoji = crate::commands::resolve("emoji").expect("/emoji resolves");
        assert!(emoji.alias_of.is_none(), "/emoji must be a primary command, not an alias");

        let mut app = AppState::new();
        assert_eq!(app.companion, CompanionKind::Pet(PetStyle::Bear), "default companion = bear");

        // Picker: 3 spinner rows (ids 100/101/102) + 5 pet rows (0-4) + Off (5) = 9.
        let items = emoji_picker_items(&app);
        assert_eq!(
            items.len(),
            9,
            "3 spinner + 5 pet + off = 9: {:?}",
            items.iter().map(|i| (i.id, i.label.clone())).collect::<Vec<_>>()
        );
        assert_eq!(items.iter().filter(|i| i.id >= 100).count(), 3, "3 spinner rows (id >= 100)");
        for name in ["braille", "arc", "pulse"] {
            assert!(
                items.iter().any(|i| i.label.to_ascii_lowercase().contains(name)),
                "spinner row {name} present: {:?}",
                items.iter().map(|i| i.label.clone()).collect::<Vec<_>>()
            );
        }

        // Selecting a pet (id 0 = Bear) then a spinner (id 100 = Braille) flips companion.
        apply_emoji_choice(&mut app, 0);
        assert_eq!(app.companion, CompanionKind::Pet(PetStyle::all()[0]));
        apply_emoji_choice(&mut app, 100);
        assert_eq!(app.companion, CompanionKind::Spinner(SpinnerStyle::Braille));

        // The tab title ANIMATES: a pet companion's face advances across ticks.
        apply_emoji_choice(&mut app, 0); // Bear
        app.spinner_tick = 0;
        let t0 = app.terminal_title();
        app.spinner_tick = PET_TICKS_PER_FRAME; // one pet-frame later
        let t1 = app.terminal_title();
        assert_ne!(t0, t1, "tab title must animate (tick 0 vs {PET_TICKS_PER_FRAME}): {t0:?} vs {t1:?}");
        assert!(t0.contains("GenericAgent"), "title has GenericAgent: {t0:?}");
        assert!(!t0.contains("NativeClaude"), "no NativeClaude: {t0:?}");

        // A braille companion's title leads with the braille glyph, not a pet face.
        apply_emoji_choice(&mut app, 100);
        app.spinner_tick = 0;
        let tb = app.terminal_title();
        assert!(
            tb.starts_with(SpinnerStyle::Braille.glyph(0)),
            "braille companion title leads with the braille glyph: {tb:?}"
        );
    }

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
}
