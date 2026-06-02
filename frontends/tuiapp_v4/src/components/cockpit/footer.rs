//! cockpit/footer.rs — the bottom chrome rows: the two below-composer rows (row1
//! runtime session info, row2 `└ Tips`), the rainbow separator, the busy spinner
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
/// glyph ANIMATES — it cycles `app.companion`'s frames at the 0.1s `tick`
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
    // The lead is the unified /emoji companion: a spinner glyph (braille/arc/pulse)
    // OR a pet face (bear/cat/…), animated at `tick`. Returns a String either way.
    let glyph = app.companion.spinner_lead(elapsed, tick);
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
            let has_split = app.tok_in.is_some() || app.tok_out.is_some();
            if has_split {
                // Use eased display values (smoothly animated toward the live targets).
                // Fall back to the live tok_in/tok_out if display is None (first tick).
                let di = app.display_tok_in.unwrap_or_else(|| app.tok_in.unwrap_or(0));
                let dout = app.display_tok_out.unwrap_or_else(|| app.tok_out.unwrap_or(0));
                g.push(Span::styled(" · ↑ ".to_string(), dim));
                g.push(Span::styled(human_count(di), text));
                g.push(Span::styled(" · ↓ ".to_string(), dim));
                g.push(Span::styled(human_count(dout), text));
            } else if let Some(tokens) = app.tokens {
                // Legacy fallback: bridge sent only a total, no split.
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

/// The ctx progress-bar width (S4) — a `[████░░░░░░░░░░░░]` fill 16 cells wide.
const CTX_BAR_W: usize = 16;

/// Build the `ctx` field's spans (S4): a `[████…░░░░]` bar filled proportional to
/// `context_percent`, then a ` {used_k}/{limit_k} ({P}%)` readout (k = chars/1000,
/// rounded; the HONEST GA trim metric is CHARS). `with_bar=false` drops the bar (the
/// narrow-terminal fallback) and keeps just the readout. When no context data has
/// arrived the bar is all-empty and the readout is a bare `—`.
fn ctx_field_spans(app: &AppState, theme: &Theme, with_bar: bool) -> Vec<Span<'static>> {
    let dim = Style::default().fg(theme.color(Token::Dim));
    let mut out: Vec<Span<'static>> = Vec::new();
    let pct = app.context_percent;
    if with_bar {
        // Fill `filled` of `CTX_BAR_W` cells; clamp the percent to [0,100] so a
        // stray over-100 reading can't overrun the bar (bounded by `.min`).
        let filled = pct
            .map(|p| ((p.clamp(0.0, 100.0) / 100.0) * CTX_BAR_W as f64).round() as usize)
            .unwrap_or(0)
            .min(CTX_BAR_W);
        // The fill brightens (Suggestion) so the eye reads the gauge; the empty
        // track + brackets stay dim chrome. `█`×filled + `░`×(W-filled), bounded by
        // the constant width (no unbounded loop).
        out.push(Span::styled("[".to_string(), dim));
        out.push(Span::styled(
            "█".repeat(filled),
            Style::default().fg(theme.color(Token::Suggestion)),
        ));
        out.push(Span::styled("░".repeat(CTX_BAR_W - filled), dim));
        out.push(Span::styled("] ".to_string(), dim));
    }
    match (app.context_used, app.context_limit, pct) {
        (Some(used), Some(limit), p) => {
            let used_k = (used as f64 / 1000.0).round() as u64;
            let limit_k = (limit as f64 / 1000.0).round() as u64;
            let pr = p.unwrap_or(0.0).round();
            out.push(Span::styled(
                format!("{used_k}k/{limit_k}k ({pr:.0}%)"),
                Style::default().fg(theme.color(Token::Text)),
            ));
        }
        // Char counts absent but a percent exists (an older bridge): show the %.
        (_, _, Some(p)) => {
            out.push(Span::styled(format!("({:.0}%)", p.round()), dim));
        }
        // No context data at all → a bare em-dash (mirrors the old `ctx —`).
        _ => out.push(Span::styled("—".to_string(), dim)),
    }
    out
}

/// ROW 1 (below the composer, Q7): runtime SESSION INFO — LABELED fields joined by
/// dim ` | ` separators (S4): `Channel: <llm> | Model: <model> | Effort: <effort> |
/// ctx: [████░░░░] Nk/Mk (P%) | branch: <branch> | mouse: select`. The connection
/// chip is folded onto the TAIL when not connected (N1 "never a silent disconnect" —
/// this row replaced the old footer that carried the chip). The chip turns
/// `Token::Error` on a disconnect / `Token::Warning` while connecting. On a NARROW
/// terminal the ctx bar is dropped FIRST (the readout + all labels stay legible),
/// mirroring the spinner's progressive width-gating.
pub(crate) fn render_session_info(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
    let dim = Style::default().fg(theme.color(Token::Dim));
    let sep = || Span::styled(" | ".to_string(), dim);
    let label = |key: &str| Span::styled(crate::i18n::t(app.lang, key).to_string(), dim);
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
    let branch = app.git_branch.as_deref().unwrap_or("—");
    // S1: Mouse mode indicator — dim, so it's discoverable but not distracting.
    // "mouse: select" in native mode (drag to copy); "mouse: click" in interactive.
    let mouse_label = if app.mouse_capture { "mouse: click" } else { "mouse: select" };

    // Build the row. The ctx BAR is the headline of this row (task 4), so it is the
    // LAST thing dropped: progressive width-gating sheds the trailing, least-important
    // fields FIRST (`mouse:` then `branch:`), keeping the bar; only if the core STILL
    // overruns do we drop the bar itself. The connection chip always stays (N1).
    let build = |with_bar: bool, with_branch: bool, with_mouse: bool| -> Vec<Span<'static>> {
        let mut spans: Vec<Span<'static>> = vec![
            label("footer.channel"),
            Span::styled(llm.clone(), Style::default().fg(theme.color(Token::Suggestion))),
            sep(),
            label("footer.model"),
            Span::styled(model.clone(), Style::default().fg(theme.color(Token::Claude))),
            sep(),
            label("footer.effort"),
            Span::styled(effort.clone(), Style::default().fg(theme.color(Token::PlanMode))),
            sep(),
            label("footer.ctx_label"),
        ];
        spans.extend(ctx_field_spans(app, theme, with_bar));
        if with_branch {
            spans.push(sep());
            spans.push(label("footer.branch"));
            spans.push(Span::styled(branch.to_string(), Style::default().fg(theme.color(Token::Suggestion))));
        }
        if with_mouse {
            spans.push(sep());
            spans.push(Span::styled(mouse_label.to_string(), dim));
        }
        // The connection chip lives on this row's tail (it owns the bottom chrome now
        // that the footer is gone) so a failed bridge stays visible (N1) — never gated.
        if !matches!(app.conn, ConnStatus::Connected { .. }) {
            let conn_tok = match &app.conn {
                ConnStatus::Connecting => Token::Warning,
                ConnStatus::Disconnected { .. } => Token::Error,
                ConnStatus::Connected { .. } => Token::Success,
            };
            spans.push(sep());
            spans.push(Span::styled(app.conn.label(), Style::default().fg(theme.color(conn_tok))));
        }
        spans
    };

    let row_width = |spans: &[Span]| -> usize {
        use unicode_width::UnicodeWidthStr;
        spans.iter().map(|s| UnicodeWidthStr::width(s.content.as_ref())).sum()
    };

    // Tiered fallback (widest → narrowest): full → drop mouse → drop branch → (last
    // resort) drop the bar. Pick the first tier that fits; the narrowest is the floor.
    let w = area.width as usize;
    let tiers = [
        (true, true, true),
        (true, true, false),  // drop `mouse:` first
        (true, false, false), // then `branch:` — bar still kept
        (false, false, false),// last resort: drop the bar
    ];
    let mut spans = build(true, true, true);
    for (bar, br, mo) in tiers {
        let candidate = build(bar, br, mo);
        if row_width(&candidate) <= w {
            spans = candidate;
            break;
        }
        spans = candidate; // keep the narrowest as the floor even if it still overruns
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// ROW 2 (below the composer, Q7): `└ Tips` — the rotating tip (deterministic by
/// tick) under a `└` leader glyph (U+2514, BOX DRAWINGS LIGHT UP AND RIGHT). While the
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
        Span::styled("└ ".to_string(), dim),
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
    use crate::flavor::{CompanionKind, SpinnerStyle};
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
                context_used: None,
                context_limit: None,
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
        app.companion = CompanionKind::Spinner(SpinnerStyle::Braille);
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
                context_used: None,
                context_limit: None,
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

    /// Slice 4: the spinner shows BOTH `↑ <in>` and `↓ <out>` arrows when the
    /// bridge sends split tok_in/tok_out, reflecting the live per-call counts.
    /// A different Status snapshot yields a different readout (not cached).
    #[test]
    fn spinner_token_readout_reflects_live_counts() {
        // Seeded tok_in=1234 (→ "1.2k") and tok_out=340.
        // display_tok_* start at None → first render falls back to tok_in/tok_out directly.
        let rows = busy_spinner_rows(1234, 340);
        let spinner = find_spinner_row(&rows);
        assert!(spinner.contains('↑'), "spinner must contain ↑ input arrow: {spinner:?}");
        assert!(spinner.contains('↓'), "spinner must contain ↓ output arrow: {spinner:?}");
        assert!(spinner.contains("↑ 1.2k"), "input token count 1234→1.2k: {spinner:?}");
        assert!(spinner.contains("↓ 340"), "output token count 340: {spinner:?}");

        // A DIFFERENT Status snapshot yields a DIFFERENT readout (not cached).
        let rows2 = busy_spinner_rows(48_300, 12_900);
        let spinner2 = find_spinner_row(&rows2);
        assert!(spinner2.contains("↑ 48.3k"), "input count 48300→48.3k: {spinner2:?}");
        assert!(spinner2.contains("↓ 12.9k"), "output count 12900→12.9k: {spinner2:?}");
        assert_ne!(spinner, spinner2, "the readout tracks the live count");
    }

    /// Slice 4 session-info deliverable: with NO effort set the row shows the
    /// non-thinking label (`非思考模式` in zh), the labeled `ctx:` field with its
    /// `(48%)` readout, and the SAME wire identity the header uses (`codex-pro` /
    /// `gpt-5.5`) — never the router name `MixinSession`. Rendered at 130-wide so the
    /// full labeled row (incl. the 16-cell ctx bar) fits the zh label without clipping.
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
                // Seed the raw char counts (S4) so the footer can draw the ctx fill
                // bar + `96k/200k (48%)` readout: 96000/200000 ≈ 48%.
                context_used: Some(96_000),
                context_limit: Some(200_000),
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

        let rows = cockpit_rows(&mut app, 130, 30);
        let info = rows
            .iter()
            .find(|r| r.contains("Channel:"))
            .expect("the session-info row carries the labeled Channel: field");
        // The TestBackend grid renders each wide CJK glyph into 2 cells, so the
        // cell-by-cell row join interleaves spaces (`非 思 考…`); compare the zh
        // label against a space-stripped form. ASCII tokens stay contiguous.
        let dense: String = info.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(dense.contains("非思考模式"), "non-thinking label (zh): {info:?}");
        assert!(info.contains("ctx:"), "the labeled ctx field: {info:?}");
        assert!(info.contains("(48%)"), "the ctx percent readout: {info:?}");
        assert!(info.contains("codex-pro"), "wire llm identity: {info:?}");
        assert!(info.contains("gpt-5.5"), "wire model identity: {info:?}");
        assert!(!info.contains("MixinSession"), "router name is NOT shown: {info:?}");
    }

    /// S4 deliverable (LIVE styled path): the session-info row renders LABELED
    /// fields joined by ` | ` separators, with a `[████░░░░]` ctx bar + a
    /// `96k/200k (48%)` readout. Rendered at 150-wide so the full row (incl. the
    /// 16-cell bar) fits comfortably; the wire identity (codex-pro/gpt-5.5) holds.
    #[test]
    fn session_info_labeled_pipe_fields_and_ctx_bar() {
        let mut app = AppState::new();
        app.conn = crate::app::ConnStatus::Connected {
            model: Some("MixinSession/codex-pro|gpt-5.2|kiro".into()),
        };
        app.git_branch = Some("feat/tui-v4".into());
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
                llm: None,
                model_real: None,
                context_percent: Some(48.0),
                context_used: Some(96_000),
                context_limit: Some(200_000),
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
        assert!(app.effort_label().is_none(), "no effort set → non-thinking");

        let rows = cockpit_rows(&mut app, 150, 30);
        let info = rows
            .iter()
            .find(|r| r.contains("Channel:"))
            .expect("the labeled session-info row");
        // Labeled fields.
        assert!(info.contains("Channel:"), "Channel label: {info:?}");
        assert!(info.contains("Model:"), "Model label: {info:?}");
        assert!(info.contains("Effort:"), "Effort label: {info:?}");
        assert!(info.contains("ctx:"), "ctx label: {info:?}");
        assert!(info.contains("branch:"), "branch label: {info:?}");
        // The ctx bar: an opening `[`, the filled glyph `█`, an empty `░`, a closing
        // `]`, and the `96k/200k (48%)` readout right after it.
        assert!(info.contains('['), "ctx bar opens with [: {info:?}");
        assert!(info.contains(']'), "ctx bar closes with ]: {info:?}");
        assert!(info.contains('█'), "ctx bar has a filled cell: {info:?}");
        assert!(info.contains('░'), "ctx bar has an empty cell: {info:?}");
        assert!(info.contains("96k/200k (48%)"), "ctx readout: {info:?}");
        // ` | ` pipe separators (was ` · `): at least the Channel|Model boundary.
        assert!(info.contains(" | "), "pipe separators present: {info:?}");
        assert!(!info.contains(" · "), "no old middot separators: {info:?}");
        // Non-thinking label shows (en) with no effort set.
        assert!(info.contains("non-thinking"), "non-thinking label (en): {info:?}");
        // Wire identity (Slice 1/2) holds — NEVER the router MixinSession chain.
        assert!(info.contains("codex-pro"), "wire llm: {info:?}");
        assert!(info.contains("gpt-5.5"), "wire model: {info:?}");
        assert!(!info.contains("MixinSession"), "router name hidden: {info:?}");
        // The branch value renders.
        assert!(info.contains("feat/tui-v4"), "branch value: {info:?}");
    }

    /// S4 width-gating (task-4 priority: the ctx BAR is the headline, dropped LAST).
    /// On a terminal too narrow for the full row, the TRAILING fields (`mouse:` then
    /// `branch:`) drop FIRST so the bar survives; only below the bar-core width does
    /// the bar itself drop. 112-wide fits the bar-core (~98) but not the full row
    /// (~136) → bar KEPT, `mouse:`/`branch:` gone. 88-wide can't fit the bar-core →
    /// the bar finally drops but the labels + `Nk/Mk (P%)` readout stay legible.
    #[test]
    fn session_info_width_gate_keeps_bar_drops_trailing_first() {
        let mut app = AppState::new();
        app.conn = crate::app::ConnStatus::Connected { model: Some("m".into()) };
        app.git_branch = Some("feat/tui-v4".into());
        app.apply_bridge_event(
            BridgeEvent::Frame(CoreToUi::Ready {
                version: None,
                model: Some("MixinSession/codex-pro".into()),
                llm: Some("codex-pro".into()),
                model_real: Some("gpt-5.5".into()),
            }),
            0,
        );
        app.apply_bridge_event(
            BridgeEvent::Frame(CoreToUi::Status {
                model: None,
                llm: None,
                model_real: None,
                context_percent: Some(48.0),
                context_used: Some(96_000),
                context_limit: Some(200_000),
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
        // 112: the bar-core (~98) fits, the full row (~136) does not → bar KEPT,
        // trailing fields dropped to make room.
        let rows = cockpit_rows(&mut app, 112, 30);
        let info = rows
            .iter()
            .find(|r| r.contains("Channel:"))
            .expect("the labeled session-info row");
        assert!(info.contains('█'), "the headline bar is KEPT on a mid-width term: {info:?}");
        assert!(info.contains('░'), "bar track kept: {info:?}");
        assert!(info.contains("96k/200k (48%)"), "ctx readout kept: {info:?}");
        assert!(info.contains("Channel:") && info.contains("ctx:"), "labels kept: {info:?}");
        // The trailing, least-important fields are shed FIRST to make room for the bar.
        assert!(!info.contains("mouse:"), "mouse field dropped first: {info:?}");
        assert!(!info.contains("branch:"), "branch field dropped before the bar: {info:?}");

        // 88: too narrow even for the bar-core (~98) → the bar finally drops (last
        // resort), but the identity labels + the readout (~80) still survive.
        let rows_narrow = cockpit_rows(&mut app, 88, 30);
        let info_n = rows_narrow
            .iter()
            .find(|r| r.contains("Channel:"))
            .expect("the labeled session-info row (narrow)");
        assert!(!info_n.contains('█'), "bar dropped as last resort on a very narrow term: {info_n:?}");
        assert!(!info_n.contains('['), "bar bracket gone when bar dropped: {info_n:?}");
        assert!(info_n.contains("Channel:"), "Channel label kept even narrow: {info_n:?}");
        assert!(info_n.contains("96k/200k (48%)"), "ctx readout kept even narrow: {info_n:?}");
    }

    /// S4: with NO context data the ctx field falls back to the `[░░░░…] —` stub
    /// (an all-empty bar + an em-dash readout), never a panic or a stray percent.
    #[test]
    fn session_info_ctx_absent_shows_dash() {
        let mut app = AppState::new();
        app.conn = crate::app::ConnStatus::Connected { model: Some("m".into()) };
        app.model = Some("m".into());
        // No Status frame → context_percent / used / limit all None.
        assert!(app.context_percent.is_none());
        assert!(app.context_used.is_none() && app.context_limit.is_none());
        let rows = cockpit_rows(&mut app, 150, 30);
        let info = rows
            .iter()
            .find(|r| r.contains("ctx:"))
            .expect("the labeled ctx field renders even with no data");
        assert!(info.contains('░'), "the absent bar is all-empty track: {info:?}");
        assert!(!info.contains('█'), "no fill when ctx is unknown: {info:?}");
        assert!(info.contains("—"), "the readout is a bare em-dash: {info:?}");
        assert!(!info.contains('%'), "no percent when ctx is unknown: {info:?}");
    }

    /// Slice 5 layout INVERSION (was `below_composer_has_two_rows`): the `└ Tip`
    /// hangs as a corner-continuation directly UNDER the busy spinner status line
    /// (above the composer), so below the composer there is now exactly ONE row
    /// (session info). IDLE keeps the historical two-row below-composer layout.
    #[test]
    fn busy_tip_under_spinner_one_row_below_composer() {
        // BUSY: spinner band on, the `└` Tip is its very next row (above composer).
        let rows = busy_spinner_rows(1234, 340);
        let spinner_idx = rows
            .iter()
            .position(|r| r.contains('…') && r.contains('('))
            .expect("the spinner status row");
        let tip = &rows[spinner_idx + 1];
        assert!(tip.starts_with("└ "), "the row under the spinner is the `└ ` tip: {tip:?}");

        // The Tip is ABOVE the composer (its bottom `╰` border comes after it), and
        // below the composer there is exactly ONE row (session info).
        let bottom = rows
            .iter()
            .rposition(|r| r.starts_with('╰'))
            .expect("the composer has a bottom border");
        assert!(spinner_idx + 1 < bottom, "the `└` tip sits above the composer");
        // Below the composer: exactly ONE non-blank row (session info), then the
        // hug-top blank spacer fills the screen bottom (R6 S1 — the composer now hugs
        // the content at the TOP instead of being pinned at `rows.len()`).
        assert!(!rows[bottom + 1].trim().is_empty(), "the session-info row follows the composer");
        assert!(
            rows[bottom + 2..].iter().all(|r| r.trim().is_empty()),
            "only the blank hug-top spacer follows the session-info row while busy"
        );
        // The sole below-composer row is the session info, NOT a second `└` tip.
        assert!(!rows[bottom + 1].starts_with("└ "), "no detached below-composer tip while busy");
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
        // Idle: session-info (row1) then the `└` tip (row2) follow the composer, then
        // the hug-top blank spacer (R6 S1 — no longer pinned at `idle_rows.len()`).
        assert!(idle_rows[idle_bottom + 2].starts_with("└ "), "row2 is the `└ ` tip when idle");
        assert!(
            idle_rows[idle_bottom + 3..].iter().all(|r| r.trim().is_empty()),
            "only the blank hug-top spacer follows the idle row2 tip"
        );
    }

    /// Slice 5 HONEST CHECK: the busy spinner status row matches CC's
    /// `<glyph> <Gerund>… (<elapsed> · ↓<tokens> · thinking <effort>)` shape, with
    /// the effort rendered as a "thinking …" phrase. With `max` effort set it reads
    /// "thinking with max effort"; the down-arrow tokens and elapsed are present.
    #[test]
    fn spinner_status_line_shape_with_thinking_effort() {
        use crate::app::effort::ReasoningEffort;
        let mut app = AppState::new();
        app.companion = CompanionKind::Spinner(SpinnerStyle::Braille);
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
                context_used: None,
                context_limit: None,
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

        // Shape: an open paren, `↑ <in> · ↓ <out>` split arrows, and the "thinking …"
        // phrase, all inside the `(…)` group.
        assert!(spinner.contains('('), "status group opens with `(`: {spinner:?}");
        assert!(spinner.contains("0.0s"), "elapsed is present: {spinner:?}");
        // New S4 shape: split ↑/↓ arrows instead of single ↓ total.
        assert!(spinner.contains('↑'), "spinner must have ↑ input arrow: {spinner:?}");
        assert!(spinner.contains('↓'), "spinner must have ↓ output arrow: {spinner:?}");
        assert!(spinner.contains("↑ 1.2k"), "input token count 1234→1.2k: {spinner:?}");
        assert!(spinner.contains("↓ 340"), "output token count 340: {spinner:?}");
        assert!(spinner.contains("· thinking with max effort"), "thinking phrase: {spinner:?}");
        assert!(spinner.trim_end().ends_with(')'), "status group closes with `)`: {spinner:?}");
        assert!(!spinner.contains('▰') && !spinner.contains('▱'), "no ctx bar: {spinner:?}");

        // The next row is the hanging `└ ` tip continuation.
        let idx = rows.iter().position(|r| r.contains('…') && r.contains('(')).unwrap();
        assert!(rows[idx + 1].starts_with("└ "), "next row is the `└ ` tip: {:?}", rows[idx + 1]);
    }

    /// Slice 5 width-gating (R5): on a NARROW terminal the tokens part drops FIRST,
    /// but the gerund + elapsed + "thinking …" phrase stay legible.
    #[test]
    fn spinner_width_gating_drops_tokens_first() {
        use crate::app::effort::ReasoningEffort;
        let mut app = AppState::new();
        app.companion = CompanionKind::Spinner(SpinnerStyle::Braille);
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
                context_used: None,
                context_limit: None,
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
        // 36-wide: too narrow for `… (0.0s · ↑ 1.2k · ↓ 340 · thinking with max effort)`
        // in full, so the whole tokens part (both ↑ and ↓) is dropped first but the
        // elapsed + thinking phrase stay legible.
        let rows = cockpit_rows(&mut app, 36, 12);
        let spinner = find_spinner_row(&rows);
        assert!(!spinner.contains('↑'), "↑ tokens dropped on narrow: {spinner:?}");
        assert!(!spinner.contains('↓'), "↓ tokens dropped on narrow: {spinner:?}");
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

        // 160-wide so the labeled row + the ctx bar + the disconnect chip on the tail
        // all fit (the S4 labeled fields are longer than the old ` · ` row, so a
        // narrow 100-wide frame would clip the tail chip — N1 needs it VISIBLE).
        let (w, h) = (160u16, 30u16);
        let rows = cockpit_rows(&mut app, w, h);
        // Row1 (just above the `└` tip row, i.e. the last-but-one row) carries the
        // disconnect chip — the reason text from the child exit is shown (N1).
        let last = rows.len() - 1;
        let row1 = &rows[last - 1];
        assert!(
            row1.contains("disconnected") || row1.contains("code 1"),
            "the disconnect chip is folded onto row1's tail: {row1:?}"
        );
    }

    // ---- S4 HONEST CHECKS (LIVE/STYLED path) --------------------------------

    /// HC-1: Spinner shows both `↑ <in>` and `↓ <out>` arrows (LIVE styled path).
    /// The seeded tok_in=1234 (→ "1.2k") and tok_out=340 must both be visible with
    /// their respective arrows. display_tok_* start None so the first render falls
    /// back to tok_in/tok_out directly (no easing yet until tick() is called).
    #[test]
    fn spinner_shows_both_up_and_down_arrows() {
        let rows = busy_spinner_rows(1234, 340);
        let spinner = find_spinner_row(&rows);
        // BOTH arrows present.
        assert!(spinner.contains('↑'), "spinner must contain ↑ input arrow: {spinner:?}");
        assert!(spinner.contains('↓'), "spinner must contain ↓ output arrow: {spinner:?}");
        // ↑ shows the input count (1234 → "1.2k").
        assert!(spinner.contains("↑ 1.2k"), "input token count: {spinner:?}");
        // ↓ shows the output count (340).
        assert!(spinner.contains("↓ 340"), "output token count: {spinner:?}");
    }

    /// HC-2: Tip rows use `└ ` leader, never `⎿` (LIVE styled path). Both the
    /// busy hanging tip (directly under the spinner) AND the idle row-2 tip below
    /// the composer must use the U+2514 corner glyph.
    #[test]
    fn tip_rows_use_floor_corner_glyph() {
        // Busy: hanging tip under spinner.
        let rows = busy_spinner_rows(1234, 340);
        let spinner_idx = rows
            .iter()
            .position(|r| r.contains('…') && r.contains('('))
            .expect("the busy spinner status row");
        let tip = &rows[spinner_idx + 1];
        assert!(tip.starts_with("└ "), "busy hanging tip must use └: {tip:?}");
        assert!(!tip.starts_with('⎿'), "must NOT use ⎿: {tip:?}");

        // Idle: row2 tip below composer.
        let mut idle = AppState::new();
        idle.conn = crate::app::ConnStatus::Connected { model: Some("m".into()) };
        idle.model = Some("m".into());
        let idle_rows = cockpit_rows(&mut idle, 100, 30);
        let last = idle_rows[idle_rows.len() - 1].clone();
        assert!(last.starts_with("└ "), "idle tip row2 must use └: {last:?}");
        assert!(!last.starts_with('⎿'), "must NOT use ⎿: {last:?}");
    }

    /// HC-3: Eased display values advance toward target on each tick() call (no
    /// loop — each tick is a single bounded step). After many ticks they converge.
    #[test]
    fn display_tok_eases_toward_target_on_tick() {
        let mut app = AppState::new();
        // Seed tok_in/tok_out targets.
        app.tok_in = Some(5000);
        app.tok_out = Some(1000);
        // display_tok* start as None.
        assert!(app.display_tok_in.is_none());
        assert!(app.display_tok_out.is_none());
        // First tick: initializes display (gap=5000 → step=50; gap=1000 → step=50).
        app.tick();
        let d_in = app.display_tok_in.expect("display_tok_in set after first tick");
        let d_out = app.display_tok_out.expect("display_tok_out set after first tick");
        assert!(d_in > 0 && d_in < 5000, "display_tok_in started easing: {d_in}");
        assert!(d_out > 0 && d_out < 1000, "display_tok_out started easing: {d_out}");
        // After enough ticks, converges to target.
        for _ in 0..200 {
            app.tick();
        }
        assert_eq!(app.display_tok_in, Some(5000), "converges to tok_in");
        assert_eq!(app.display_tok_out, Some(1000), "converges to tok_out");
    }
}
