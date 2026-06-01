//! cockpit/footer.rs — the bottom chrome rows: the two below-composer rows (row1
//! runtime session info, row2 `⎿ Tips`), the rainbow separator, the busy spinner
//! band, and the frozen above-composer done-line.

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::app::{AppState, ConnStatus};
use crate::components::text::{
    clip_to, fmt_dur, human_count, llm_channel, truncate_model, MODEL_LABEL_CAP,
};
use crate::flavor::{self, heat_bold, heat_token, PetStyle};
use crate::theme::{Theme, Token};

/// SEPARATOR: the full-width rainbow 7-stop line (§5 / §9). Upgraded from the old flat
/// `▓` rule to the effects-engine separator: a smoothly INTERPOLATED ROYGBIV gradient
/// that honors the capability gate (a plain dim line at mono / NO_COLOR) and animates a
/// raised-cosine SHIMMER sweep while the running indicator is active (subtle/full/demo).
pub(crate) fn render_separator(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
    let width = area.width as usize;
    // Animate the shimmer sweep only when the running indicator is active; otherwise the
    // separator is the static gradient (so an idle cockpit doesn't churn cells).
    let shimmer_phase = if app.effects.running_indicator_active() {
        Some(app.effects.shimmer.phase())
    } else {
        None
    };
    let spans = crate::theme::rainbow::separator_spans(theme, &app.effects.caps, width, shimmer_phase);
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// SPINNER band (only when busy): <pet> <spinner> <Gerund>… <elapsed>s · ctx bar.
/// Q9: the spinner glyph is a STATIC `⠿` (braille all-dots, U+283F) — the motion
/// comes from the pet blink + the rotating gerund + the heat-color ramp, not a
/// per-tick glyph cycle (so we no longer index `spinner_style`).
pub(crate) fn render_spinner(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme, now_ms: u64) {
    let elapsed = app.turn_elapsed_ms(now_ms);
    let tick = (now_ms / 100) as u64;
    let glyph = '⠿';
    let tok = heat_token(elapsed);
    let mut heat_style = Style::default().fg(theme.color(tok));
    if heat_bold(elapsed) {
        heat_style = heat_style.add_modifier(Modifier::BOLD);
    }
    let secs = elapsed as f64 / 1000.0;
    let dim = Style::default().fg(theme.color(Token::Dim));
    let text = Style::default().fg(theme.color(Token::Text));

    let mut spans: Vec<Span> = Vec::new();
    // LEAD (unchanged GA signature): pet face (heat-colored, ~0.5s blink) + spinner
    // glyph + gerund. The pet/glyph carry the heat ramp; the gerund is plain text.
    if app.pet_style != PetStyle::Off {
        let face = flavor::pet(app.pet_style, elapsed, tick);
        if !face.is_empty() {
            spans.push(Span::styled(format!("{face} "), heat_style));
        }
    }
    spans.push(Span::styled(format!("{glyph} "), heat_style));
    spans.push(Span::styled(format!("{}…", flavor::gerund(app.lang, tick)), text));

    // STATUS group — a CC-style ` (elapsed · ↑in ↓out · ctx ▰▱ pct% · effort)`
    // (SpinnerAnimationRow.tsx): dim chrome + parens, with only the live token
    // NUMBERS (Text) and the effort level (Claude) brightened so the eye lands on
    // them. `↑`/`↓` are the per-call input/output sizes (tui_v3 readout + CC's
    // arrows); they appear once the first LLM response of the turn lands.
    spans.push(Span::styled(" (".to_string(), dim));
    spans.push(Span::styled(format!("{secs:.1}s"), dim));
    if app.tok_in.is_some() || app.tok_out.is_some() {
        spans.push(Span::styled(" · ↑".to_string(), dim));
        spans.push(Span::styled(human_count(app.tok_in.unwrap_or(0)), text));
        spans.push(Span::styled(" ↓".to_string(), dim));
        spans.push(Span::styled(human_count(app.tok_out.unwrap_or(0)), text));
    }
    if let Some(pct) = app.context_percent {
        spans.push(Span::styled(format!(" · ctx {}", ctx_bar(pct)), dim));
    }
    if let Some(effort) = app.effort_label() {
        spans.push(Span::styled(" · ".to_string(), dim));
        spans.push(Span::styled(
            effort.to_string(),
            Style::default().fg(theme.color(Token::Claude)),
        ));
    }
    spans.push(Span::styled(")".to_string(), dim));

    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// DONE-LINE (above the composer, only when idle right after a turn — Q7): the
/// settled `⠿ <gerund> for <fmt_dur> · ↑ <in> · ↓ <out>` summary. FROZEN — the `⠿`
/// is a static mint glyph and the duration/tokens don't animate (the turn is over).
/// Mirrors the spinner's `(…)`-less readout but left-aligned with bright numbers.
/// The gerund index is derived from the (frozen) elapsed seconds so it stays stable
/// while idle and reads as "what just happened".
pub(crate) fn render_done_line(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
    let ms = app.last_turn_ms.unwrap_or(0);
    let secs = ms / 1000;
    let dim = Style::default().fg(theme.color(Token::Dim));
    let text = Style::default().fg(theme.color(Token::Text));
    let mut spans: Vec<Span> = vec![
        // Q9: `⠿` only, mint = done (a frozen completion mark, not the busy spinner).
        Span::styled("⠿ ".to_string(), Style::default().fg(theme.color(Token::Success))),
        Span::styled(format!("{} ", flavor::gerund_at(app.lang, secs)), text),
        Span::styled(format!("for {}", fmt_dur(secs)), dim),
    ];
    if app.tok_in.is_some() || app.tok_out.is_some() {
        spans.push(Span::styled(" · ↑ ".to_string(), dim));
        spans.push(Span::styled(human_count(app.tok_in.unwrap_or(0)), text));
        spans.push(Span::styled(" · ↓ ".to_string(), dim));
        spans.push(Span::styled(human_count(app.tok_out.unwrap_or(0)), text));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// ROW 1 (below the composer, Q7): runtime SESSION INFO — `llm · model · effort ·
/// ctx · branch`, left-aligned with dim ` · ` separators. The connection chip is
/// folded onto the TAIL when not connected (N1 "never a silent disconnect" — this
/// row replaced the old footer that carried the chip). The chip turns `Token::Error`
/// on a disconnect / `Token::Warning` while connecting.
pub(crate) fn render_session_info(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
    let dim = Style::default().fg(theme.color(Token::Dim));
    let sep = || Span::styled("  ·  ".to_string(), dim);
    let llm = llm_channel(app.model.as_deref());
    let model = truncate_model(app.model.as_deref().unwrap_or("—"), MODEL_LABEL_CAP);
    let effort = app.effort_label().unwrap_or("—");
    let ctx = match app.context_percent {
        Some(p) => format!("ctx {p:.0}%"),
        None => "ctx —".to_string(),
    };
    let branch = app.git_branch.as_deref().unwrap_or("—");
    let mut spans: Vec<Span> = vec![
        Span::styled(llm.to_string(), Style::default().fg(theme.color(Token::Suggestion))),
        sep(),
        Span::styled(model, Style::default().fg(theme.color(Token::Claude))),
        sep(),
        Span::styled(effort.to_string(), Style::default().fg(theme.color(Token::PlanMode))),
        sep(),
        Span::styled(ctx, dim),
        sep(),
        Span::styled(branch.to_string(), Style::default().fg(theme.color(Token::Suggestion))),
    ];
    // The connection chip lives on this row's tail (it owns the bottom chrome now
    // that the footer is gone) so a failed bridge stays visible (N1).
    if !matches!(app.conn, ConnStatus::Connected { .. }) {
        let conn_tok = match &app.conn {
            ConnStatus::Connecting => Token::Warning,
            ConnStatus::Disconnected { .. } => Token::Error,
            ConnStatus::Connected { .. } => Token::Success,
        };
        spans.push(sep());
        spans.push(Span::styled(app.conn.label(), Style::default().fg(theme.color(conn_tok))));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// ROW 2 (below the composer, Q7): `⎿ Tips` — the rotating tip (deterministic by
/// tick) under a `⎿` leader glyph (the rounded `└`, restored from v2/v3). While the
/// 3-stage Ctrl+C is ARMED (§8) the row instead shows the "press Ctrl+C again to
/// quit" hint in `Token::Warning` (a transient override that vanishes when the 2s
/// arm expires) — never a transcript notice.
pub(crate) fn render_tips(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme, now_ms: u64) {
    let dim = Style::default().fg(theme.color(Token::Dim));
    let armed = app.chord.ctrl_c_hint_active(now_ms);
    let (body_src, body_style) = if armed {
        (crate::i18n::t(app.lang, "ctrlc.arm"), Style::default().fg(theme.color(Token::Warning)))
    } else {
        (flavor::tip(app.lang, now_ms / 100), dim)
    };
    let body = clip_to(body_src, (area.width as usize).saturating_sub(2));
    let spans = vec![
        Span::styled("⎿ ".to_string(), dim),
        Span::styled(body, body_style),
    ];
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// A tiny context bar `▰▰▰▱▱ 50%` for a percent. PURE-ish helper.
pub(crate) fn ctx_bar(pct: f64) -> String {
    let filled = ((pct / 100.0) * 5.0).round().clamp(0.0, 5.0) as usize;
    let bar: String = (0..5).map(|i| if i < filled { '▰' } else { '▱' }).collect();
    format!("{bar} {pct:.0}%")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::AppState;
    use crate::bridge::protocol::CoreToUi;
    use crate::bridge::BridgeEvent;
    use crate::components::render;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    #[test]
    fn ctx_bar_fills_proportionally() {
        assert_eq!(ctx_bar(0.0), "▱▱▱▱▱ 0%");
        assert_eq!(ctx_bar(100.0), "▰▰▰▰▰ 100%");
        assert!(ctx_bar(50.0).contains("50%"));
    }

    /// Render the whole cockpit for `app` into a `w×h` TestBackend and return its
    /// rows as trimmed strings (the headless layout probe the dump scenarios use).
    fn cockpit_rows(app: &mut AppState, w: u16, h: u16) -> Vec<String> {
        let theme = crate::theme::Theme::default_theme();
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

    /// THE Q7 done-line deliverable: an idle cockpit right AFTER a turn shows the
    /// frozen `⠿ … for <dur> · ↑ <in> · ↓ <out>` summary above the composer.
    #[test]
    fn done_line_shows_elapsed_and_tokens_when_idle() {
        let mut app = AppState::new();
        app.conn = crate::app::ConnStatus::Connected { model: Some("m".into()) };
        app.model = Some("m".into());
        // A turn runs (begin at 0) then ends 106s later → last_turn_ms = 1m 46s.
        app.apply_bridge_event(
            BridgeEvent::Frame(CoreToUi::MessageBegin { mid: "m1".into(), role: "assistant".into() }),
            0,
        );
        app.apply_bridge_event(
            BridgeEvent::Frame(CoreToUi::Status {
                model: None,
                context_percent: Some(48.0),
                tokens: Some(1574),
                input_tokens: Some(1234),
                output_tokens: Some(340),
                cache_tokens: Some(96),
                last_input: Some(1234),
                last_output: Some(340),
                text: None,
            }),
            0,
        );
        app.apply_bridge_event(
            BridgeEvent::Frame(CoreToUi::MessageEnd { mid: "m1".into(), reason: "stop".into() }),
            106_000,
        );
        assert!(!app.busy, "idle after the turn");
        assert_eq!(app.last_turn_ms, Some(106_000));

        let rows = cockpit_rows(&mut app, 100, 30);
        let done = rows
            .iter()
            .find(|r| r.starts_with("⠿ "))
            .expect("the done-line is rendered when idle after a turn");
        assert!(done.contains("for 1m 46s"), "frozen elapsed: {done:?}");
        assert!(done.contains("↑ 1.2k"), "input tokens: {done:?}");
        assert!(done.contains("↓ 340"), "output tokens: {done:?}");

        // It does NOT show while busy (the spinner band owns that slot instead).
        // NB: the BUSY spinner now ALSO leads with `⠿` (Q9), so the done-line is
        // discriminated by its unique `… for <dur>` summary, which the spinner
        // (which reads `(<secs>s …)`) never carries.
        let mut busy = AppState::new();
        busy.apply_bridge_event(
            BridgeEvent::Frame(CoreToUi::MessageBegin { mid: "m1".into(), role: "assistant".into() }),
            0,
        );
        let busy_rows = cockpit_rows(&mut busy, 100, 30);
        assert!(
            !busy_rows.iter().any(|r| r.starts_with("⠿ ") && r.contains(" for ")),
            "no frozen done-line while a turn is running"
        );
    }

    /// Drive a BUSY cockpit (turn open) with a seeded token snapshot and return
    /// its rows — the shared setup for the Q9 spinner-glyph and Q4 live-token tests.
    fn busy_spinner_rows(tok_in: u64, tok_out: u64) -> Vec<String> {
        let mut app = AppState::new();
        app.conn = crate::app::ConnStatus::Connected { model: Some("m".into()) };
        app.model = Some("m".into());
        app.apply_bridge_event(
            BridgeEvent::Frame(CoreToUi::MessageBegin { mid: "m1".into(), role: "assistant".into() }),
            0,
        );
        app.apply_bridge_event(
            BridgeEvent::Frame(CoreToUi::Status {
                model: None,
                context_percent: Some(48.0),
                tokens: Some(tok_in + tok_out),
                input_tokens: Some(tok_in),
                output_tokens: Some(tok_out),
                cache_tokens: Some(96),
                last_input: Some(tok_in),
                last_output: Some(tok_out),
                text: None,
            }),
            0,
        );
        assert!(app.busy, "the turn is still running");
        cockpit_rows(&mut app, 100, 30)
    }

    /// Q9: the BUSY spinner glyph is the braille all-dots `⠿` (U+283F) — a static
    /// completion-style mark, NOT the old arc set (no `◜◠◝◞◡◟`).
    #[test]
    fn spinner_emits_braille_all_dots() {
        let rows = busy_spinner_rows(1234, 340);
        let spinner = rows
            .iter()
            .find(|r| r.contains('⠿'))
            .expect("the busy spinner row carries the `⠿` glyph");
        assert_eq!('⠿' as u32, 0x283F, "the glyph is U+283F braille all-dots");
        // The old arc frames must NOT appear anywhere in the busy chrome.
        for arc in ['◜', '◠', '◝', '◞', '◡', '◟'] {
            assert!(
                !rows.iter().any(|r| r.contains(arc)),
                "no arc glyph {arc:?} in the spinner: {spinner:?}"
            );
        }
    }

    /// Q4: the spinner's `↑in ↓out` readout reflects the LIVE token counts — a
    /// later `Status` frame with different numbers changes what the row shows
    /// (the spinner reads `app.tok_in/tok_out` directly, it does not cache them).
    #[test]
    fn spinner_token_readout_reflects_live_counts() {
        // Seeded 1234/340 → human-compacted to `1.2k`/`340` in the live readout.
        let rows = busy_spinner_rows(1234, 340);
        let spinner = rows
            .iter()
            .find(|r| r.contains('⠿'))
            .expect("busy spinner row");
        assert!(spinner.contains("↑1.2k"), "input tokens live: {spinner:?}");
        assert!(spinner.contains("↓340"), "output tokens live: {spinner:?}");

        // A DIFFERENT Status snapshot yields a DIFFERENT readout (not cached).
        let rows2 = busy_spinner_rows(48_300, 12_900);
        let spinner2 = rows2
            .iter()
            .find(|r| r.contains('⠿'))
            .expect("busy spinner row");
        assert!(spinner2.contains("↑48.3k"), "live input updates: {spinner2:?}");
        assert!(spinner2.contains("↓12.9k"), "live output updates: {spinner2:?}");
        assert_ne!(spinner, spinner2, "the readout tracks the live counts");
    }

    /// THE Q7 below-composer deliverable: exactly TWO rows under the composer — row1
    /// runtime session info, row2 `⎿ <tip>` — and NO `❯ chat` anywhere.
    #[test]
    fn below_composer_has_two_rows() {
        let mut app = AppState::new();
        app.conn = crate::app::ConnStatus::Connected { model: Some("m".into()) };
        app.model = Some("m".into());
        let (w, h) = (100u16, 30u16);
        let rows = cockpit_rows(&mut app, w, h);

        // The composer's bottom border is the `╰` row; the two rows AFTER it are the
        // session info (row1) and the `⎿` tip (row2) — and they are the LAST two rows.
        let bottom = rows
            .iter()
            .rposition(|r| r.starts_with('╰'))
            .expect("the composer has a bottom border");
        assert_eq!(bottom + 3, rows.len(), "exactly two rows follow the composer");
        let row2 = &rows[bottom + 2];
        assert!(row2.starts_with("⎿ "), "row2 is the `⎿ ` tip line: {row2:?}");
        // The old `❯ chat` footer eyesore is gone (Q7).
        assert!(
            !rows.iter().any(|r| r.contains("❯ chat")),
            "no `❯ chat` anywhere in the chrome"
        );
    }

    /// N1 survives the footer removal: when the bridge is DISCONNECTED the
    /// connection chip is still visible — folded onto row1's tail (it owns the
    /// bottom chrome now). A failed bridge is never silent.
    #[test]
    fn connection_chip_survives_footer_removal() {
        let mut app = AppState::new();
        app.apply_bridge_event(BridgeEvent::ChildExited { code: Some(1) }, 0);
        assert!(matches!(app.conn, crate::app::ConnStatus::Disconnected { .. }));

        let (w, h) = (100u16, 30u16);
        let rows = cockpit_rows(&mut app, w, h);
        // Row1 (just above the `⎿` tip row, i.e. the last-but-one row) carries the
        // disconnect chip — the reason text from the child exit is shown (N1).
        let last = rows.len() - 1;
        let row1 = &rows[last - 1];
        assert!(
            row1.contains("disconnected") || row1.contains("code 1"),
            "the disconnect chip is folded onto row1's tail: {row1:?}"
        );
    }
}
