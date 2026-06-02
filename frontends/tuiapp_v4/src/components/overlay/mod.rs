//! components/overlay/ — the MODAL overlay renderer (§3 overlay stack / §7).
//!
//! Draws whichever [`Overlay`] is active ON TOP of the cockpit/dashboard. This
//! module owns only the dispatch ([`render`]) + the shared card chrome
//! ([`centered`], [`titled_block`]); each card paints in a focused submodule:
//! [`effort`] (the `/effort` slider), [`picker`] (list picker + ask_user),
//! [`info`] (`/help` `/status` `/cost` `/verbose` `/btw`), [`effects`]
//! (`/effects demo`). Full-screen overlays cover the whole area; compact cards
//! draw a CENTERED bordered card. No hardcoded colors — every style is a [`Token`].

mod effects;
mod effort;
mod info;
mod picker;

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::Span;
use ratatui::widgets::{Block, Borders, Clear};
use ratatui::Frame;

use crate::app::{AppState, Overlay};
use crate::theme::{Theme, Token};

/// Draw the active overlay. `area` is the full frame; the overlay sizes its own
/// region within it. A full-screen overlay clears the whole area first; compact
/// cards clear only their own centered rect (done per-renderer).
pub fn render(frame: &mut Frame, area: Rect, ov: &Overlay, app: &AppState, theme: &Theme, now_ms: u64) {
    if ov.is_fullscreen() {
        frame.render_widget(Clear, area);
    }
    let lang = app.lang;
    match ov {
        Overlay::Picker { picker, .. } => picker::render_picker(frame, area, picker, theme, lang),
        Overlay::AskUser(ask) => picker::render_ask_user(frame, area, ask, theme, lang),
        Overlay::Help => info::render_help(frame, area, theme, lang),
        Overlay::Keybindings => info::render_keybindings(frame, area, theme, lang),
        Overlay::Status => info::render_status(frame, area, app, theme, now_ms),
        Overlay::Cost => info::render_cost(frame, area, app, theme),
        Overlay::Verbose => info::render_verbose(frame, area, app, theme),
        Overlay::Btw { question, answer, .. } => {
            info::render_btw(frame, area, question, answer.as_deref(), theme, lang)
        }
        Overlay::Scheduler(sched) => super::scheduler::render(frame, area, sched, theme, lang),
        Overlay::Continue(picker) => {
            let now_secs = super::continue_picker::wall_secs();
            super::continue_picker::render(frame, area, picker, theme, lang, now_secs)
        }
        Overlay::Effects => effects::render_effects_demo(frame, area, app, theme),
        Overlay::EffortSlider(slider) => effort::render_effort_slider(frame, area, slider, theme),
    }
}

/// A centered card `Rect` of `w`×`h` (clamped to `area`); the caller adds `Clear`
/// so it covers the view underneath.
pub(super) fn centered(area: Rect, w: u16, h: u16) -> Rect {
    let w = w.min(area.width);
    let h = h.min(area.height);
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(h)) / 2;
    Rect { x, y, width: w, height: h }
}

/// Bordered block with a title in the Claude accent.
pub(super) fn titled_block(title: &str, theme: &Theme) -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.color(Token::Claude)))
        .title(Span::styled(
            format!(" {title} "),
            Style::default().fg(theme.color(Token::Claude)).add_modifier(Modifier::BOLD),
        ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::effort::EffortSlider;
    use crate::components::picker::{AskUserPicker, PickItem, Picker, PickerKind};
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    /// Render each overlay to an in-memory backend and confirm its headline chrome
    /// paints (a render-level guard that the modal surface draws; the load-bearing
    /// selection logic is tested in `components::picker`).
    #[test]
    fn overlays_render_their_chrome() {
        let theme = Theme::default_theme();
        let render_with = |ov: Overlay, needle: &str| {
            let mut app = AppState::new();
            app.cost.input = 100;
            app.cost.output = 250;
            app.push_tool_audit("Read(src/main.rs)".into());
            app.overlay = Some(ov);
            let backend = TestBackend::new(80, 24);
            let mut terminal = Terminal::new(backend).unwrap();
            terminal
                .draw(|f| {
                    let area = f.area();
                    render(f, area, app.overlay.as_ref().unwrap(), &app, &theme, 1000);
                })
                .unwrap();
            let buf = terminal.backend().buffer();
            let text: String = buf.content().iter().map(|c| c.symbol()).collect();
            assert!(text.contains(needle), "overlay must paint {needle:?}; got chrome only");
        };

        render_with(Overlay::Help, "Commands");
        render_with(Overlay::Keybindings, "Keyboard shortcuts");
        render_with(Overlay::Status, "model");
        render_with(Overlay::Cost, "Token usage");
        render_with(Overlay::Verbose, "audit");
        render_with(
            Overlay::Picker {
                picker: Picker::new(
                    PickerKind::Llm,
                    vec![PickItem::new(0, "OpenAI/gpt").current(true)],
                ),
                theme_backup: None,
            },
            "Switch model",
        );
        render_with(
            Overlay::AskUser(AskUserPicker::new("a1", "pick one?", vec!["yes".into(), "no".into()], true)),
            "pick one?",
        );
        render_with(
            Overlay::Btw { ask_id: "b1".into(), question: "what is 2+2?".into(), answer: None },
            "querying",
        );
        render_with(
            Overlay::EffortSlider(EffortSlider::new(crate::app::effort::ReasoningEffort::High)),
            "Reasoning effort",
        );
        render_with(
            Overlay::EffortSlider(EffortSlider::new(crate::app::effort::ReasoningEffort::High)),
            "Smarter",
        );
        render_with(
            Overlay::EffortSlider(EffortSlider::new(crate::app::effort::ReasoningEffort::High)),
            "▲",
        );
        render_with(
            Overlay::EffortSlider(EffortSlider::new(crate::app::effort::ReasoningEffort::High)),
            "Enter to confirm",
        );
    }

    /// Render an overlay to a tall in-memory backend and return its flattened text
    /// (for the alias-presentation + keybindings-table render assertions).
    fn overlay_text(ov: Overlay) -> String {
        let theme = Theme::default_theme();
        let mut app = AppState::new();
        app.overlay = Some(ov);
        let backend = TestBackend::new(100, 44);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| {
                let area = f.area();
                render(f, area, app.overlay.as_ref().unwrap(), &app, &theme, 1000);
            })
            .unwrap();
        terminal.backend().buffer().content().iter().map(|c| c.symbol()).collect()
    }

    /// Q6 — `/help` lists an alias as a dim "alias of /primary" line under its
    /// primary, NOT as a peer command row. The primary (`/verbose`) is present, and
    /// the alias (`/tools`) appears with the "alias of" phrasing.
    #[test]
    fn help_lists_aliases_dimmed_not_as_peers() {
        let text = overlay_text(Overlay::Help);
        assert!(text.contains("/verbose"), "the primary /verbose is a peer row");
        // The alias renders as "/tools — alias of /verbose" (the i18n 'alias of').
        let alias_label = crate::i18n::t(crate::i18n::Lang::En, "help.alias_of");
        assert_eq!(alias_label, "alias of");
        assert!(text.contains("alias of"), "aliases are shown with the 'alias of' phrasing");
        assert!(text.contains("/tools"), "the /tools alias is listed (under its primary)");
        // S1: /mouse is back in the registry as a discoverable App command.
        assert!(text.contains("/mouse"), "/mouse is present in /help (S1 toggle model)");
    }

    /// Q7 — the `/keybindings` overlay paints the chord→action pairs table incl. the
    /// C3 parity-gap chords it added (Ctrl+T theme, Ctrl+/ help, Ctrl+Enter newline)
    /// and the magic-prefix line.
    #[test]
    fn keybindings_overlay_shows_pairs_table() {
        let text = overlay_text(Overlay::Keybindings);
        for chord in ["Ctrl+O", "Ctrl+T", "Ctrl+/", "Ctrl+Enter", "/fold"] {
            assert!(text.contains(chord), "keybindings table shows {chord:?}");
        }
        assert!(text.contains("Magic prefixes"), "the magic-prefix line is shown");
    }
}
