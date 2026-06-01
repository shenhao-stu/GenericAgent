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
use crate::flavor::{self, heat_bold, heat_token};
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

/// SPINNER status line (only when busy), CC `SpinnerAnimationRow` composition:
/// `<glyph> <Gerund>… (<elapsed> · ↓ <tokens> tokens · thinking <effort>)`. The leading
/// glyph ANIMATES — it cycles `app.spinner_style`'s frames at the 0.1s `tick`
/// (default braille `⠋⠙⠹…`); the done-line (`render_done_line`) keeps the static
/// `⠿`, and busy/done are mutually exclusive so the spinner settles on `⠿` when the
/// turn ends. NO emoji pet here — the pet lives in the tab title; the spinner is
/// braille-only. The `(…)` group is dim chrome with parens; only the live token
/// NUMBER (Text) and the effort phrase (Claude) brighten so the eye lands on them.
/// `↓` is CC's per-turn token arrow, trailed by the i18n `tokens.unit` word. The effort reads as a "thinking …" phrase
/// (i18n `spinner.thinking` + level → "thinking with max effort"); with no effort
/// it shows the non-thinking label. Progressive width-gating (R5): when the full
/// line overruns `area.width` the tokens part is dropped FIRST so the gerund +
/// elapsed + thinking phrase stay legible on a narrow terminal.
pub(crate) fn render_spinner(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme, now_ms: u64) {
    let elapsed = app.turn_elapsed_ms(now_ms);
    let tick = (now_ms / 100) as u64;
    let glyph = app.spinner_style.glyph(tick);
    let mut heat_style = Style::default().fg(theme.color(heat_token(elapsed)));
    if heat_bold(elapsed) {
        heat_style = heat_style.add_modifier(Modifier::BOLD);
    }
    let secs = elapsed as f64 / 1000.0;
    let dim = Style::default().fg(theme.color(Token::Dim));
    let text = Style::default().fg(theme.color(Token::Text));

    let lead = vec![
        Span::styled(format!("{glyph} "), heat_style),
        Span::styled(format!("{}…", flavor::gerund(app.lang, tick)), text),
    ];
    let thinking = crate::i18n::thinking_phrase(app.lang, app.effort_label());

    // Build the `(…)` group with the tokens part, then drop ONLY the tokens part if
    // the whole line overruns the width (R5 progressive gating). Elapsed + the
    // thinking phrase always stay.
    let group = |with_tokens: bool| -> Vec<Span<'static>> {
        let mut g = vec![
            Span::styled(" (".to_string(), dim),
            Span::styled(format!("{secs:.1}s"), dim),
        ];
        if with_tokens {
            if let Some(tokens) = app.tokens {
                g.push(Span::styled(" · ↓ ".to_string(), dim));
                g.push(Span::styled(human_count(tokens), text));
                g.push(Span::styled(format!(" {}", crate::i18n::t(app.lang, "tokens.unit")), dim));
            }
        }
        g.push(Span::styled(" · ".to_string(), dim));
        g.push(Span::styled(thinking.clone(), Style::default().fg(theme.color(Token::Claude))));
        g.push(Span::styled(")".to_string(), dim));
        g
    };

    let line_width = |g: &[Span]| -> usize {
        use unicode_width::UnicodeWidthStr;
        lead.iter().chain(g).map(|s| UnicodeWidthStr::width(s.content.as_ref())).sum()
    };

    let mut group_spans = group(true);
    if line_width(&group_spans) > area.width as usize {
        group_spans = group(false);
    }

    let mut spans = lead.clone();
    spans.extend(group_spans);
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// DONE-LINE (above the composer, only when idle right after a turn — Q7): the
/// settled `⠿ <gerund> for <fmt_dur> · ↑ <in> · ↓ <out>` summary. FROZEN — the `⠿`
/// is a static mint glyph and the duration/tokens don't animate (the turn is over).
/// Mirrors the spinner's `(…)`-less readout but left-aligned with bright numbers.
/// The gerund index derives from the frozen elapsed seconds so it stays stable.
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
    // FOOTER identity (Slice 4) = the SAME wire identity the header uses (Slice 1/2):
    // prefer `llm_name` (codex-pro) / `model_real` (gpt-5.5), falling back to the
    // routing channel / truncated `model` only when the bridge omitted them.
    let llm = match app.llm_name.as_deref() {
        Some(name) if !name.is_empty() => name.to_string(),
        _ => llm_channel(app.model.as_deref()).to_string(),
    };
    let model = match app.model_real.as_deref() {
        Some(real) if !real.is_empty() => truncate_model(real, MODEL_LABEL_CAP),
        _ => truncate_model(app.model.as_deref().unwrap_or("—"), MODEL_LABEL_CAP),
    };
    let effort = app
        .effort_label()
        .map(str::to_string)
        .unwrap_or_else(|| crate::i18n::t(app.lang, "effort.none").to_string());
    let ctx = match app.context_percent {
        Some(p) => format!("ctx {p:.0}%"),
        None => "ctx —".to_string(),
    };
    let branch = app.git_branch.as_deref().unwrap_or("—");
    let mut spans: Vec<Span> = vec![
        Span::styled(llm, Style::default().fg(theme.color(Token::Suggestion))),
        sep(),
        Span::styled(model, Style::default().fg(theme.color(Token::Claude))),
        sep(),
        Span::styled(effort, Style::default().fg(theme.color(Token::PlanMode))),
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

#[cfg(test)]
mod tests {
    use crate::app::AppState;
    use crate::bridge::protocol::CoreToUi;
    use crate::bridge::BridgeEvent;
    use crate::components::render;
    use crate::flavor::{PetStyle, SpinnerStyle};
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    /// Render the whole cockpit for `app` at `now_ms` into a `w×h` TestBackend and
    /// return its rows as trimmed strings (the headless layout probe the dump
    /// scenarios use). `now_ms` drives the animation clock (spinner frame, pet blink).
    fn cockpit_rows_at(app: &mut AppState, w: u16, h: u16, now_ms: u64) -> Vec<String> {
        let theme = crate::theme::Theme::default_theme();
        let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
        app.prepare_frame(ratatui::layout::Rect::new(0, 0, w, h), &theme);
        term.draw(|f| render(f, app, &theme, now_ms)).unwrap();
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

    /// `cockpit_rows_at` at `now_ms=0` (the static-frame probe most tests want).
    fn cockpit_rows(app: &mut AppState, w: u16, h: u16) -> Vec<String> {
        cockpit_rows_at(app, w, h, 0)
    }

    /// The busy spinner status row — found by its `(` readout group + the gerund's
    /// trailing `…` (stable across animated glyph frames; the done-line has neither).
    fn find_spinner_row(rows: &[String]) -> &String {
        rows.iter()
            .find(|r| r.contains('…') && r.contains('('))
            .expect("the busy spinner status row")
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
                llm: None, model_real: None,
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

    /// Drive a BUSY cockpit (turn open) at `now_ms` with a seeded token snapshot and
    /// return its rows — the shared setup for the spinner-animation and live-token
    /// tests. The pet is turned OFF so the spinner LEAD cell is the bare glyph.
    fn busy_spinner_rows_at(tok_in: u64, tok_out: u64, now_ms: u64) -> Vec<String> {
        let mut app = AppState::new();
        app.pet_style = PetStyle::Off;
        app.conn = crate::app::ConnStatus::Connected { model: Some("m".into()) };
        app.model = Some("m".into());
        app.apply_bridge_event(
            BridgeEvent::Frame(CoreToUi::MessageBegin { mid: "m1".into(), role: "assistant".into() }),
            0,
        );
        app.apply_bridge_event(
            BridgeEvent::Frame(CoreToUi::Status {
                model: None,
                llm: None, model_real: None,
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
        cockpit_rows_at(&mut app, 100, 30, now_ms)
    }

    fn busy_spinner_rows(tok_in: u64, tok_out: u64) -> Vec<String> {
        busy_spinner_rows_at(tok_in, tok_out, 0)
    }

    /// Slice 4: the BUSY spinner glyph ANIMATES — it cycles `spinner_style`'s frames
    /// at the 0.1s tick, so the glyph at now_ms=0 (frame 0) DIFFERS from now_ms=300
    /// (frame 3). The default style is braille (the tui_v3 soul), never the old arc.
    #[test]
    fn spinner_animates_braille_frames() {
        let frames = SpinnerStyle::default().frames();
        // The lead glyph is the first char of the gerund status row (pet is Off).
        let glyph_at = |now_ms: u64| -> char {
            let rows = busy_spinner_rows_at(1234, 340, now_ms);
            find_spinner_row(&rows).chars().next().expect("a lead glyph")
        };
        let g0 = glyph_at(0);
        let g3 = glyph_at(300);
        assert!(frames.contains(&g0), "frame-0 glyph {g0:?} is from the style set");
        assert!(frames.contains(&g3), "frame-3 glyph {g3:?} is from the style set");
        assert_ne!(g0, g3, "the busy spinner glyph animates across ticks");
        assert_eq!(g0, frames[0], "tick 0 → frame 0");
        assert_eq!(g3, frames[3 % frames.len()], "tick 3 → frame 3");
        // The old arc frames must NOT appear anywhere in the busy chrome.
        let rows = busy_spinner_rows_at(1234, 340, 0);
        for arc in ['◜', '◠', '◝', '◞', '◡', '◟'] {
            assert!(!rows.iter().any(|r| r.contains(arc)), "no arc glyph {arc:?} in the spinner");
        }
    }

    /// Slice 5: the spinner's `↓ <tokens> tokens` readout (CC's per-turn token arrow)
    /// reflects the LIVE count — a later `Status` frame with a different total
    /// changes what the row shows (the spinner reads `app.tokens` directly, it does
    /// not cache it). The `↑in` / `ctx` parts are no longer on the spinner line.
    #[test]
    fn spinner_token_readout_reflects_live_counts() {
        // Seeded total 1234+340=1574 → human-compacted to `1.6k` after the `↓ `, with
        // the trailing `tokens` word (Slice 11 wording).
        let rows = busy_spinner_rows(1234, 340);
        let spinner = find_spinner_row(&rows);
        assert!(spinner.contains("↓ 1.6k tokens"), "live token total: {spinner:?}");
        assert!(!spinner.contains('↑'), "no `↑in` on the spinner line: {spinner:?}");

        // A DIFFERENT Status snapshot yields a DIFFERENT readout (not cached).
        let rows2 = busy_spinner_rows(48_300, 12_900);
        let spinner2 = find_spinner_row(&rows2);
        assert!(spinner2.contains("↓ 61.2k tokens"), "live token total updates: {spinner2:?}");
        assert_ne!(spinner, spinner2, "the readout tracks the live count");
    }

    /// Slice 4 session-info deliverable: with NO effort set the row shows the
    /// non-thinking label (`非思考模式` in zh), `ctx 48%`, and the SAME wire identity
    /// the header uses (`codex-pro` / `gpt-5.5`) — never the router name `MixinSession`.
    #[test]
    fn session_info_shows_non_thinking_ctx_and_wire_identity() {
        let mut app = AppState::new();
        app.lang = crate::i18n::Lang::Zh;
        app.conn = crate::app::ConnStatus::Connected { model: Some("MixinSession/codex-pro|gpt-5.2|kiro".into()) };
        // The wire identity arrives via Ready/Status (Slice 1): llm=codex-pro,
        // model_real=gpt-5.5; the router `model` stays the long MixinSession chain.
        app.apply_bridge_event(
            BridgeEvent::Frame(CoreToUi::Ready {
                version: None,
                model: Some("MixinSession/codex-pro|gpt-5.2|kiro".into()),
                llm: Some("codex-pro".into()),
                model_real: Some("gpt-5.5".into()),
            }),
            0,
        );
        app.apply_bridge_event(
            BridgeEvent::Frame(CoreToUi::Status {
                model: None,
                llm: None, model_real: None,
                context_percent: Some(48.0),
                tokens: Some(100),
                input_tokens: Some(60),
                output_tokens: Some(40),
                cache_tokens: None,
                last_input: Some(60),
                last_output: Some(40),
                text: None,
            }),
            0,
        );
        assert!(app.effort_label().is_none(), "no effort set");

        let rows = cockpit_rows(&mut app, 100, 30);
        let info = rows
            .iter()
            .find(|r| r.contains("ctx 48%"))
            .expect("the session-info row carries ctx 48%");
        // The TestBackend grid renders each wide CJK glyph into 2 cells, so the
        // cell-by-cell row join interleaves spaces (`非 思 考…`); compare the zh
        // label against a space-stripped form. ASCII tokens stay contiguous.
        let dense: String = info.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(dense.contains("非思考模式"), "non-thinking label (zh): {info:?}");
        assert!(info.contains("codex-pro"), "wire llm identity: {info:?}");
        assert!(info.contains("gpt-5.5"), "wire model identity: {info:?}");
        assert!(!info.contains("MixinSession"), "router name is NOT shown: {info:?}");
    }

    /// Slice 5 layout INVERSION (was `below_composer_has_two_rows`): the `⎿ Tip`
    /// hangs as a corner-continuation directly UNDER the busy spinner status line
    /// (above the composer), so below the composer there is now exactly ONE row
    /// (session info). IDLE keeps the historical two-row below-composer layout.
    #[test]
    fn busy_tip_under_spinner_one_row_below_composer() {
        // BUSY: spinner band on, the `⎿` Tip is its very next row (above composer).
        let rows = busy_spinner_rows(1234, 340);
        let spinner_idx = rows
            .iter()
            .position(|r| r.contains('…') && r.contains('('))
            .expect("the spinner status row");
        let tip = &rows[spinner_idx + 1];
        assert!(tip.starts_with("⎿ "), "the row under the spinner is the `⎿ ` tip: {tip:?}");

        // The Tip is ABOVE the composer (its bottom `╰` border comes after it), and
        // below the composer there is exactly ONE row (session info).
        let bottom = rows
            .iter()
            .rposition(|r| r.starts_with('╰'))
            .expect("the composer has a bottom border");
        assert!(spinner_idx + 1 < bottom, "the `⎿` tip sits above the composer");
        assert_eq!(bottom + 2, rows.len(), "exactly ONE row follows the composer while busy");
        // The sole below-composer row is the session info, NOT a second `⎿` tip.
        assert!(!rows[bottom + 1].starts_with("⎿ "), "no detached below-composer tip while busy");
        assert!(
            !rows.iter().any(|r| r.contains("❯ chat")),
            "no `❯ chat` anywhere in the chrome"
        );

        // IDLE: the Tip returns to its row2 below-composer slot (two rows below).
        let mut idle = AppState::new();
        idle.conn = crate::app::ConnStatus::Connected { model: Some("m".into()) };
        idle.model = Some("m".into());
        let idle_rows = cockpit_rows(&mut idle, 100, 30);
        let idle_bottom = idle_rows
            .iter()
            .rposition(|r| r.starts_with('╰'))
            .expect("the composer has a bottom border");
        assert_eq!(idle_bottom + 3, idle_rows.len(), "two rows follow the composer when idle");
        assert!(idle_rows[idle_bottom + 2].starts_with("⎿ "), "row2 is the `⎿ ` tip when idle");
    }

    /// Slice 5 HONEST CHECK: the busy spinner status row matches CC's
    /// `<glyph> <Gerund>… (<elapsed> · ↓<tokens> · thinking <effort>)` shape, with
    /// the effort rendered as a "thinking …" phrase. With `max` effort set it reads
    /// "thinking with max effort"; the down-arrow tokens and elapsed are present.
    #[test]
    fn spinner_status_line_shape_with_thinking_effort() {
        use crate::app::effort::ReasoningEffort;
        let mut app = AppState::new();
        app.pet_style = PetStyle::Off;
        app.conn = crate::app::ConnStatus::Connected { model: Some("m".into()) };
        app.model = Some("m".into());
        app.set_reasoning_effort(ReasoningEffort::Max);
        app.apply_bridge_event(
            BridgeEvent::Frame(CoreToUi::MessageBegin { mid: "m1".into(), role: "assistant".into() }),
            0,
        );
        app.apply_bridge_event(
            BridgeEvent::Frame(CoreToUi::Status {
                model: None,
                llm: None, model_real: None,
                context_percent: Some(48.0),
                tokens: Some(1574),
                input_tokens: Some(1234),
                output_tokens: Some(340),
                cache_tokens: None,
                last_input: Some(1234),
                last_output: Some(340),
                text: None,
            }),
            0,
        );
        let rows = cockpit_rows(&mut app, 100, 30);
        let spinner = find_spinner_row(&rows);

        // Shape: an open paren, a `·`-separated `↓<tokens>`, and the "thinking …"
        // phrase, all inside the `(…)` group.
        assert!(spinner.contains('('), "status group opens with `(`: {spinner:?}");
        assert!(spinner.contains("0.0s"), "elapsed is present: {spinner:?}");
        assert!(spinner.contains("· ↓ 1.6k tokens"), "down-arrow token total: {spinner:?}");
        assert!(spinner.contains("· thinking with max effort"), "thinking phrase: {spinner:?}");
        assert!(spinner.trim_end().ends_with(')'), "status group closes with `)`: {spinner:?}");
        // No ctx bar / `↑in` on the spinner line (moved off per Slice 5).
        assert!(!spinner.contains('↑'), "no `↑in`: {spinner:?}");
        assert!(!spinner.contains('▰') && !spinner.contains('▱'), "no ctx bar: {spinner:?}");

        // The next row is the hanging `⎿ ` tip continuation.
        let idx = rows.iter().position(|r| r.contains('…') && r.contains('(')).unwrap();
        assert!(rows[idx + 1].starts_with("⎿ "), "next row is the `⎿ ` tip: {:?}", rows[idx + 1]);
    }

    /// Slice 5 width-gating (R5): on a NARROW terminal the tokens part drops FIRST,
    /// but the gerund + elapsed + "thinking …" phrase stay legible.
    #[test]
    fn spinner_width_gating_drops_tokens_first() {
        use crate::app::effort::ReasoningEffort;
        let mut app = AppState::new();
        app.pet_style = PetStyle::Off;
        app.conn = crate::app::ConnStatus::Connected { model: Some("m".into()) };
        app.model = Some("m".into());
        app.set_reasoning_effort(ReasoningEffort::Max);
        app.apply_bridge_event(
            BridgeEvent::Frame(CoreToUi::MessageBegin { mid: "m1".into(), role: "assistant".into() }),
            0,
        );
        app.apply_bridge_event(
            BridgeEvent::Frame(CoreToUi::Status {
                model: None,
                llm: None, model_real: None,
                context_percent: Some(48.0),
                tokens: Some(1574),
                input_tokens: Some(1234),
                output_tokens: Some(340),
                cache_tokens: None,
                last_input: Some(1234),
                last_output: Some(340),
                text: None,
            }),
            0,
        );
        // 36-wide: too narrow for `… (0.0s · ↓1.6k · thinking with max effort)` in
        // full, so the tokens part is dropped but the thinking phrase remains.
        let rows = cockpit_rows(&mut app, 36, 12);
        let spinner = find_spinner_row(&rows);
        assert!(!spinner.contains("↓"), "tokens part dropped on narrow: {spinner:?}");
        assert!(spinner.contains("0.0s"), "elapsed kept on narrow: {spinner:?}");
        assert!(spinner.contains("thinking"), "thinking phrase kept on narrow: {spinner:?}");
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
