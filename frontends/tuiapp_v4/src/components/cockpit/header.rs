//! cockpit/header.rs — the multi-line rounded HEADER box (tui_v3 banner parity).
//!
//! A `BorderType::Rounded` block (lavender border) whose inner area holds 5 Lines:
//! `>_ GenericAgent` (only `>_` accent) / blank / `model: <llm> · <model>  /llm
//! switch` / `directory: <cwd>` / `session: <name> · scrollback`. The model row
//! uses the ROUND-4 wire identity (`app.llm_name` / `app.model_real`, e.g.
//! `codex-pro` / `gpt-5.5`), NOT the full MixinSession pipe-chain — it falls back to
//! the footer's `llm_channel`/`truncate_model` over `app.model` only when the new
//! fields are absent (a stale bridge). See `recon/round4/R1_tui_v3_render.md` ITEM 1.

use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph};
use ratatui::Frame;

use crate::app::AppState;
use crate::components::text::{llm_channel, truncate_model, MODEL_LABEL_CAP};
use crate::i18n::t;
use crate::theme::{Theme, Token};

/// The product slogan that opens the header box. Literally `>_ GenericAgent`
/// (tui_v3 `_make_banner_lines`); only the `>_` is accent-colored, the rest default.
const SLOGAN_PROMPT: &str = ">_";
const SLOGAN_NAME: &str = " GenericAgent";

/// Header rows the box needs: 1 blank top-margin + rounded box (top border + 5
/// interior Lines + bottom border = 7). The cockpit layout (`split_cockpit`) and
/// the `prepare_frame` geometry both read this so the box never clips.
pub(crate) const HEADER_ROWS: u16 = 8;

/// Render the ROUND-4 model identity for the `model:` row: `<llm> · <model>`
/// (e.g. `codex-pro · gpt-5.5`). When the additive wire fields are absent (an old
/// bridge), degrade to the footer shape — `llm_channel · truncate_model(model)` —
/// so the row is never blank.
fn model_identity(app: &AppState) -> String {
    match (app.llm_name.as_deref(), app.model_real.as_deref()) {
        (Some(llm), Some(model)) if !llm.is_empty() && !model.is_empty() => {
            format!("{llm} · {model}")
        }
        _ => {
            let llm = llm_channel(app.model.as_deref());
            let model = truncate_model(app.model.as_deref().unwrap_or("—"), MODEL_LABEL_CAP);
            format!("{llm} · {model}")
        }
    }
}

/// The cwd with `$HOME` collapsed to `~` (tui_v3 `os.getcwd()` with `$HOME`→`~`).
/// Keeps the FULL path (the box is full terminal width); only the home prefix is
/// abbreviated. PURE over `(cwd, home)`.
fn home_cwd(cwd: &str, home: Option<&str>) -> String {
    if let Some(home) = home.filter(|h| !h.is_empty()) {
        if let Some(rest) = cwd.strip_prefix(home) {
            if rest.is_empty() {
                return "~".to_string();
            }
            if rest.starts_with(['/', '\\']) {
                return format!("~{rest}");
            }
        }
    }
    cwd.to_string()
}

/// HEADER: a rounded, lavender-bordered box (tui_v3 banner) holding the slogan +
/// the live `model` / `directory` / `session` rows. The box is drawn into rows
/// `1..HEADER_ROWS` so a one-row blank margin sits above it (the v3 leading blank);
/// it gracefully shrinks if the caller hands a shorter `area`.
pub(crate) fn render_header(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme, _now_ms: u64) {
    if area.height == 0 || area.width == 0 {
        return;
    }
    let dim = Style::default().fg(theme.color(Token::Dim));
    let accent = Style::default().fg(theme.color(Token::Claude));
    let text = Style::default().fg(theme.color(Token::Text));

    // Left label column (dim) + value (default fg). The 3-wide gap after each label
    // left-aligns the values into a column, matching tui_v3's baked-in spacing.
    let label = |key: &str| Span::styled(format!("{}   ", t(app.lang, key)), dim);

    let model_row = Line::from(vec![
        label("banner.model"),
        Span::styled(model_identity(app), text),
        Span::styled(format!("   {}", t(app.lang, "banner.llm_hint")), dim),
    ]);
    let dir_row = Line::from(vec![
        label("banner.directory"),
        Span::styled(home_cwd(&app.cwd, dirs_home().as_deref()), text),
    ]);
    let session_row = Line::from(vec![
        label("banner.session"),
        Span::styled(
            format!("{} · {}", app.sessions.active_name(), t(app.lang, "banner.scrollback")),
            text,
        ),
    ]);
    let rows = vec![
        Line::from(vec![Span::styled(SLOGAN_PROMPT, accent), Span::styled(SLOGAN_NAME, text)]),
        Line::default(),
        model_row,
        dir_row,
        session_row,
    ];

    // A one-row top margin (the v3 leading blank), then the rounded box fills the
    // rest. When `area` is too short to seat the margin, draw the box flush.
    let box_area = if area.height > HEADER_ROWS.saturating_sub(1) {
        Rect { x: area.x, y: area.y + 1, width: area.width, height: area.height - 1 }
    } else {
        area
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme.color(Token::Border)));
    let inner = block.inner(box_area);
    frame.render_widget(block, box_area);
    // Inset one column on the left so every interior row sits a space inside the
    // border (tui_v3 `│ >_ GenericAgent`, not flush `│>_ GenericAgent`).
    let padded = Rect {
        x: inner.x + 1,
        y: inner.y,
        width: inner.width.saturating_sub(1),
        height: inner.height,
    };
    frame.render_widget(Paragraph::new(rows), padded);
}

/// The user's home directory as a string (`$HOME` / `%USERPROFILE%`), for the
/// `directory:` row's `~` collapse. Effectful (env read) and isolated here so
/// [`home_cwd`] stays pure + unit-tested.
fn dirs_home() -> Option<String> {
    std::env::var("HOME")
        .ok()
        .or_else(|| std::env::var("USERPROFILE").ok())
        .filter(|h| !h.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{AppState, ConnStatus};
    use crate::components::render;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    /// Render the cockpit into a `w×h` TestBackend and return the rows as
    /// trailing-trimmed strings (the bytes the terminal would show).
    fn rows(app: &mut AppState, w: u16, h: u16) -> Vec<String> {
        let theme = Theme::default_theme();
        let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
        app.prepare_frame(ratatui::layout::Rect::new(0, 0, w, h), &theme);
        term.draw(|f| render(f, app, &theme, 0)).unwrap();
        let buf = term.backend().buffer();
        (0..h as usize)
            .map(|y| {
                let mut row = String::new();
                for x in 0..w as usize {
                    row.push_str(buf.content()[y * w as usize + x].symbol());
                }
                row.trim_end().to_string()
            })
            .collect()
    }

    /// THE Slice-2 deliverable: the header renders a rounded, multi-ROW box — a
    /// `╭`/`╰` border, a `>_ GenericAgent` slogan line, and the model identity from
    /// `app.llm_name` / `app.model_real` (`codex-pro` and `gpt-5.5`) on its OWN row
    /// with the `/llm switch` hint, directory + session each on their own row. The
    /// full pipe-chain in `app.model` must NOT appear (the round-4 identity overrides
    /// the v3 full-chain banner).
    #[test]
    fn header_box_uses_round4_identity_rows() {
        let mut app = AppState::new();
        let chain = "MixinSession/codex-pro|getoken_20x|kiro";
        app.conn = ConnStatus::Connected { model: Some(chain.into()) };
        app.model = Some(chain.into());
        app.llm_name = Some("codex-pro".into());
        app.model_real = Some("gpt-5.5".into());

        let r = rows(&mut app, 100, 30);
        let screen = r.join("\n");

        // Rounded border glyphs (U+256D top-left, U+2570 bottom-left).
        assert!(screen.contains('╭'), "rounded top-left corner present:\n{screen}");
        assert!(screen.contains('╰'), "rounded bottom-left corner present:\n{screen}");

        // The slogan line begins `>_ GenericAgent` (after the `│ ` box gutter).
        let slogan = r
            .iter()
            .find(|l| l.contains(">_ GenericAgent"))
            .expect("a `>_ GenericAgent` slogan row inside the box");
        assert!(slogan.trim_start_matches(['│', ' ']).starts_with(">_ GenericAgent"));

        // The model row carries BOTH the llm and the real model, plus the hint.
        let model_row = r
            .iter()
            .find(|l| l.contains("codex-pro") && l.contains("gpt-5.5"))
            .expect("a model row with codex-pro · gpt-5.5");
        assert!(model_row.contains("/llm switch"), "the /llm switch hint trails the model: {model_row:?}");

        // directory + session each have their own labelled row.
        assert!(r.iter().any(|l| l.contains("directory:")), "a directory row:\n{screen}");
        assert!(r.iter().any(|l| l.contains("session:")), "a session row:\n{screen}");
        assert!(screen.contains("scrollback"), "the session value carries `scrollback`");

        // The FULL pipe-chain must not leak into the header (round-4 identity wins).
        assert!(
            !screen.contains("getoken_20x|kiro"),
            "the full MixinSession pipe-chain must not appear in the header:\n{screen}"
        );
    }

    /// The model row degrades to the footer shape (`llm_channel · truncate_model`)
    /// when the additive wire identity is absent (a stale/old bridge) — never blank.
    #[test]
    fn header_model_row_falls_back_without_identity() {
        let mut app = AppState::new();
        let chain = "MixinSession/codex-pro|gpt-5.2|kiro";
        app.model = Some(chain.into());
        // llm_name / model_real intentionally None (old bridge).
        assert!(app.llm_name.is_none() && app.model_real.is_none());

        let id = model_identity(&app);
        assert!(id.contains("codex-pro"), "fallback shows the primary segment: {id:?}");
        assert!(!id.is_empty());
        // And it renders on the model row of the live box.
        let r = rows(&mut app, 100, 30);
        assert!(
            r.iter().any(|l| l.contains("model:") && l.contains("codex-pro")),
            "the fallback model identity renders on the labelled row:\n{}",
            r.join("\n")
        );
    }

    /// `home_cwd` collapses the home prefix to `~` (both `/` and `\\` separators)
    /// and leaves a non-home path untouched. PURE — no env read.
    #[test]
    fn home_cwd_collapses_home_prefix() {
        assert_eq!(home_cwd("/home/u/proj", Some("/home/u")), "~/proj");
        assert_eq!(home_cwd("C:\\Users\\me\\proj", Some("C:\\Users\\me")), "~\\proj");
        assert_eq!(home_cwd("/home/u", Some("/home/u")), "~");
        assert_eq!(home_cwd("/var/log", Some("/home/u")), "/var/log");
        assert_eq!(home_cwd("/var/log", None), "/var/log");
    }
}
