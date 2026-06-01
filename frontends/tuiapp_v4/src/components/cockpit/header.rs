//! cockpit/header.rs — the single-line cockpit HEADER widget.

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::app::AppState;
use crate::components::text::{compact_cwd, llm_channel, truncate_model, MODEL_LABEL_CAP};
use crate::theme::{Theme, Token};

/// The product slogan that opens the header (Q7). `❯❯` (double fast-forward caret)
/// echoes the composer + history `❯ ` prompt, so the chrome rhymes top-to-bottom;
/// mono, no emoji, one accent color.
const SLOGAN: &str = "❯❯ GenericAgent";

/// HEADER: ONE clean line, left-aligned (Q7) — the slogan followed by the live
/// `llm · model · dir · session` fields. Each field is a dim KEY label + an
/// accented VALUE so the eye lands on the values; no tip / shortcut hints here
/// (those live on the bottom rows / in `/keybindings`).
pub(crate) fn render_header(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme, _now_ms: u64) {
    // The model string is a MixinSession pipe-list; `llm_channel` gives the routing
    // channel (the `SessionType/` prefix) and `truncate_model` the primary segment,
    // so the header shows "how it's routed" + "which model" without the full chain.
    let llm = llm_channel(app.model.as_deref());
    let model = truncate_model(app.model.as_deref().unwrap_or("—"), MODEL_LABEL_CAP);
    // The cwd is the most compressible field — keep it short so the trailing
    // `session …` field stays on-screen at a narrow (≈100-col) terminal.
    let cwd = compact_cwd(&app.cwd, 16);
    let session = app.sessions.active_name().to_string();

    let dim = Style::default().fg(theme.color(Token::Dim));
    let key = |s: &'static str| Span::styled(s, dim);
    let val = |s: String, tok: Token| Span::styled(s, Style::default().fg(theme.color(tok)));

    let spans = vec![
        Span::styled(
            SLOGAN,
            Style::default()
                .fg(theme.color(Token::Claude))
                .add_modifier(Modifier::BOLD),
        ),
        key("  llm "),
        val(llm.to_string(), Token::Suggestion),
        key("  model "),
        val(model, Token::Claude),
        key("  dir "),
        val(cwd, Token::Text),
        key("  session "),
        val(session, Token::Suggestion),
    ];
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

#[cfg(test)]
mod tests {
    use crate::app::{AppState, ConnStatus};
    use crate::components::render;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    /// THE Q7 header deliverable: the top row carries the slogan `❯❯ GenericAgent`
    /// plus the `llm`/`model`/`dir`/`session` field labels, left-aligned.
    #[test]
    fn header_has_slogan_llm_model_dir_session() {
        let (w, h) = (120u16, 16u16);
        let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
        let mut app = AppState::new();
        let model = "MixinSession/codex-pro|gpt-5.2|kiro";
        app.conn = ConnStatus::Connected { model: Some(model.into()) };
        app.model = Some(model.into());
        let theme = crate::theme::Theme::default_theme();
        app.prepare_frame(ratatui::layout::Rect::new(0, 0, w, h), &theme);
        term.draw(|f| render(f, &app, &theme, 0)).unwrap();

        // The header is the TOP row (y=0). Read it back as text.
        let buf = term.backend().buffer();
        let mut row0 = String::new();
        for x in 0..w as usize {
            row0.push_str(buf.content()[x].symbol());
        }
        assert!(row0.starts_with("❯❯ GenericAgent"), "slogan leads the header: {row0:?}");
        assert!(row0.contains("llm "), "llm field label present");
        assert!(row0.contains("model "), "model field label present");
        assert!(row0.contains("dir "), "dir field label present");
        assert!(row0.contains("session "), "session field label present");
        // The simplified llm channel + the primary model segment both render.
        assert!(row0.contains("MixinSession"), "the llm channel / model shows");
        // It is left-aligned: the very first cell is the slogan glyph, not padding.
        assert_eq!(buf.content()[0].symbol(), "❯");
    }
}
