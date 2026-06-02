//! cockpit/transcript.rs — the flex transcript region + its per-role row builders
//! (the full-width user band, the speaker-gutter fallback).

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::app::AppState;
use crate::components::text::clip_to;
use crate::render::BlockRole;
use crate::theme::{Theme, Token};

/// TRANSCRIPT: the flex region (P1). Renders the viewport's VISIBLE WINDOW.
pub(crate) fn render_transcript(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
    if app.transcript.is_empty() {
        let hint = Line::from(Span::styled(
            crate::i18n::t(app.lang, "transcript.empty"),
            Style::default().fg(theme.color(Token::Dim)),
        ));
        frame.render_widget(Paragraph::new(hint), area);
        return;
    }

    let block_of = |id: u64| -> Option<&crate::app::Block> {
        app.transcript.iter().find(|b| b.id == id)
    };

    // Assistant styled lines come from the BLOCK-OWNED memo (C1-F4): each block
    // renders its cockpit pass once per `(rev, width, fold_all, fold overrides)`, so
    // the per-frame `md_cache` HashMap is gone — and the styled lines drawn here are
    // the very same ones `to_render_block` projected for the wrap cache (1:1, the P1
    // parity). The per-node fold overrides (Fix E / Q8) ride in via `BlockFolds`.
    let fold_all = app.fold_all;

    let window = app.viewport.visible(&app.wrap_cache);
    let mut lines: Vec<Line> = Vec::with_capacity(window.len());
    for vl in &window {
        let block = block_of(vl.block_id);
        let role = block.map(|b| b.render_role()).unwrap_or(BlockRole::Assistant);

        if role == BlockRole::Assistant {
            // The block memo is keyed by the SAME width `prepare_frame` synced the
            // wrap cache at (`area.width`), so the indexed row matches the wrap
            // cache's row count; on a miss (out-of-range) fall back to the plain row.
            let line = block
                .and_then(|b| {
                    let folds = crate::render::BlockFolds {
                        block_id: b.id,
                        fold_all,
                        overrides: Some(&app.folds),
                    };
                    b.cockpit_line(theme, &folds, area.width, vl.intra)
                })
                .map(|l| crate::markdown::strip_atomic_line(&l))
                .unwrap_or_else(|| {
                    Line::from(Span::styled(
                        vl.text.clone(),
                        Style::default().fg(theme.color(Token::Text)),
                    ))
                });
            lines.push(line);
            continue;
        }

        // USER message → a full-width inverse BAND (Q1): bg `userMessageBackground`
        // rgb(58,58,58), white text, a `❯ ` prompt inside the band, every row padded
        // to the full width so the band spans edge-to-edge.
        if role == BlockRole::User {
            lines.push(user_band_line(&vl.text, area.width, theme));
            continue;
        }

        // Other non-assistant roles: speaker gutter + single token, hanging indent.
        let (gutter, gutter_tok, body_tok) = gutter_for(role);
        let mut spans: Vec<Span> = Vec::new();
        if !gutter.is_empty() {
            if vl.is_block_start {
                spans.push(Span::styled(
                    gutter,
                    Style::default()
                        .fg(theme.color(gutter_tok))
                        .add_modifier(Modifier::BOLD),
                ));
            } else {
                spans.push(Span::raw(" ".repeat(gutter.chars().count())));
            }
        }
        spans.push(Span::styled(
            vl.text.clone(),
            Style::default().fg(theme.color(body_tok)),
        ));
        lines.push(Line::from(spans));
    }

    if !app.following() && !lines.is_empty() {
        let hint = Span::styled(
            crate::i18n::t(app.lang, "transcript.more_below"),
            Style::default().fg(theme.color(Token::Suggestion)),
        );
        if let Some(last) = lines.last_mut() {
            *last = Line::from(hint);
        }
    }

    // SLICE S1 — HUG THE TOP: the transcript is rendered top-aligned into the given
    // `area` (a `Paragraph` draws top-down). When the content is SHORTER than the
    // viewport, `split_cockpit` has already sized this `area` to exactly
    // `total_visual_lines` rows and parked the blank in a trailing spacer at the very
    // bottom of the screen — so content + spinner + composer + footer all hug the top
    // (the Claude-Code "grow from top" look). When the content OVERFLOWS, `area` is the
    // full flex region and the viewport scrolls internally, exactly as before. The
    // round-5 bottom-anchor sub-rect (which pushed content DOWN behind a blank gap) is
    // gone — geometry now lives wholly in `split_cockpit` (one source of truth, P1).
    frame.render_widget(Paragraph::new(lines), area);
}

/// Build one full-width USER band row (Q1): the row `text` (one soft-wrapped
/// visual line of the user message) rendered with bg `UserBand` rgb(58,58,58) +
/// white `Text`, led by a visible `❯ ` prompt INSIDE the band and right-padded so
/// the band spans the FULL terminal width. PURE-ish (themed `Line`). The whole row
/// carries the bg (prompt included) so it reads as a solid inverse band with a
/// shell-native prefix, echoing the composer's `❯ ` mark.
pub(crate) fn user_band_line<'a>(text: &str, width: u16, theme: &Theme) -> Line<'a> {
    let band = Style::default()
        .bg(theme.color(Token::UserBand))
        .fg(theme.color(Token::Text));
    let w = width as usize;
    // `❯ ` (2 cells) + text (clipped so prompt+text never exceeds the width), then
    // right-pad with spaces to the full width (CJK-correct) so the band spans edge
    // to edge. The prompt shares the band bg → still one solid charcoal band.
    let prompt = "❯ ";
    let body = clip_to(text, w.saturating_sub(2));
    let used = 2 + unicode_width::UnicodeWidthStr::width(body.as_str());
    let pad = w.saturating_sub(used);
    Line::from(vec![
        Span::styled(prompt.to_string(), band),
        Span::styled(body, band),
        Span::styled(" ".repeat(pad), band),
    ])
}

/// STICKY HEADER (R6 Part A): a pinned 1-row breadcrumb at the TOP of the transcript
/// while scrolled away from the tail — `↑ <first line of the last user prompt…>` on
/// the `UserBand` charcoal bg, DIM fg (so it reads as chrome, not a live message),
/// clipped + width-padded edge-to-edge. The caller only invokes this when
/// `!app.following()` AND a user prompt exists (so the row never appears at the tail).
pub(crate) fn render_sticky_header(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
    let Some(text) = app.last_user_source_first_line() else {
        return;
    };
    let band = Style::default()
        .bg(theme.color(Token::UserBand))
        .fg(theme.color(Token::Dim));
    let w = area.width as usize;
    let prefix = "↑ ";
    let body = clip_to(text, w.saturating_sub(2));
    let used = 2 + unicode_width::UnicodeWidthStr::width(body.as_str());
    let pad = w.saturating_sub(used);
    let line = Line::from(vec![
        Span::styled(prefix.to_string(), band),
        Span::styled(body, band),
        Span::styled(" ".repeat(pad), band),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

/// The speaker gutter glyph + accent token + body token for a role.
pub(crate) fn gutter_for(role: BlockRole) -> (&'static str, Token, Token) {
    match role {
        BlockRole::User => ("❯ ", Token::Claude, Token::Text),
        BlockRole::Assistant => ("", Token::Text, Token::Text),
        BlockRole::System => ("» ", Token::Suggestion, Token::Dim),
        BlockRole::Tool => ("⚙ ", Token::Warning, Token::Dim),
        BlockRole::Notice => ("• ", Token::Warning, Token::Dim),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single USER band row carries the `UserBand` bg across its FULL width with
    /// white `Text` fg AND leads with the `❯ ` prompt (Q1) — the helper that builds
    /// the band.
    #[test]
    fn user_band_line_spans_width_with_band_bg() {
        use ratatui::style::Color;
        let theme = Theme::default_theme();
        let line = user_band_line("hello", 40, &theme);
        // Every span carries the band bg + white fg.
        for span in &line.spans {
            assert_eq!(span.style.bg, Some(theme.color(Token::UserBand)));
            assert_eq!(span.style.fg, Some(theme.color(Token::Text)));
        }
        // The bg is the exact CC userMessageBackground rgb(58,58,58).
        assert_eq!(theme.color(Token::UserBand), Color::Rgb(58, 58, 58));
        // The band spans the full 40 cells (prompt + text + right pad).
        let total: usize = line
            .spans
            .iter()
            .map(|s| unicode_width::UnicodeWidthStr::width(s.content.as_ref()))
            .sum();
        assert_eq!(total, 40, "the band fills the whole width");
        // Q1: the row leads with `❯ ` and the text sits in the band right after it.
        let joined: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(joined.starts_with("❯ hello"));
    }

    /// THE deliverable test (Q1): a rendered cockpit frame draws the user message
    /// as a full-width band — the buffer cells on the user row carry bg
    /// rgb(58,58,58) edge-to-edge (with the `❯ ` prompt living inside the band).
    #[test]
    fn user_row_has_band_bg() {
        use crate::app::{AppState, ConnStatus};
        use crate::components::render;
        use ratatui::backend::TestBackend;
        use ratatui::style::Color;
        use ratatui::Terminal;

        let (w, h) = (60u16, 16u16);
        let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
        let mut app = AppState::new();
        app.conn = ConnStatus::Connected { model: Some("m".into()) };
        app.push_user("hello world".into());
        let theme = Theme::default_theme();
        // Render is PURE (F7): the wrap-cache/viewport sync is hoisted into
        // `prepare_frame`, which the loop/harness call before the draw. Drive it the
        // same way here so the transcript region is synced before we read the buffer.
        app.prepare_frame(ratatui::layout::Rect::new(0, 0, w, h), &theme);
        term.draw(|f| render(f, &app, &theme, 0)).unwrap();

        let buf = term.backend().buffer();
        let band = Color::Rgb(58, 58, 58);
        // Find a row that contains the user text "hello world".
        let mut band_row: Option<usize> = None;
        for y in 0..h as usize {
            let mut row = String::new();
            for x in 0..w as usize {
                row.push_str(buf.content()[y * w as usize + x].symbol());
            }
            if row.contains("hello world") {
                band_row = Some(y);
                break;
            }
        }
        let y = band_row.expect("the user message row is rendered");
        // The whole row is the band: every cell on it has bg rgb(58,58,58) — the
        // `❯ ` prompt is INSIDE the band, so the bg stays solid edge-to-edge (Q1).
        for x in 0..w as usize {
            let cell = &buf.content()[y * w as usize + x];
            assert_eq!(
                cell.bg, band,
                "user-band cell ({x},{y}) must carry the band bg, got {:?}",
                cell.bg
            );
        }
    }

    /// SLICE S1 HONEST-CHECK — content_hugs_top_blank_at_bottom.
    ///
    /// Builds a styled TestBackend frame at 80×40 with a transcript SHORTER than its
    /// available height (a short assistant turn + user message + 2 notices). After the
    /// hug-top fix:
    ///   * the FIRST non-blank content row sits at the TOP of the transcript region
    ///     (NOT pushed down behind a blank gap),
    ///   * the just-above-composer status row (the frozen done-line `⠿ … for 0s`)
    ///     IMMEDIATELY follows the last content row (gap == 0),
    ///   * the blank is absorbed at the screen BOTTOM — the rows below the tips row are
    ///     blank.
    /// FAILS before the fix (round-5 bottom-anchor pushed content DOWN, leaving the
    /// blank gap ABOVE content: the first content row was far below `transcript.y` and
    /// the bottom rows were NOT blank). PASSES after.
    ///
    /// Also asserts the tool box ╰─╯ bottom border is present (the "visually not
    /// closing" appearance was an optical illusion from the old blank gap, not a
    /// missing border).
    #[test]
    fn content_hugs_top_blank_at_bottom() {
        use crate::app::{AppState, ConnStatus};
        use crate::bridge::{BridgeEvent, CoreToUi};
        use crate::components::cockpit::split_cockpit;
        use crate::components::render;
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let (w, h) = (80u16, 40u16);
        let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
        let theme = Theme::default_theme();
        let mut app = AppState::new();
        app.conn = ConnStatus::Connected { model: Some("gpt-5".into()) };

        // Short transcript: a brief assistant turn (with a tool box) + user + 2 notices.
        let ev = BridgeEvent::Frame(CoreToUi::MessageBegin {
            mid: "m1".into(),
            role: "assistant".into(),
        });
        app.apply_bridge_event(ev, 0);
        let text = "Let me check something.\n\u{1F6E0}\u{FE0F} code_run({\"script\": \"cargo build\"})\n[Info] build succeeded\nDone.";
        let ev2 = BridgeEvent::Frame(CoreToUi::MessageDelta {
            mid: "m1".into(),
            text: text.into(),
        });
        app.apply_bridge_event(ev2, 0);
        let ev3 = BridgeEvent::Frame(CoreToUi::MessageEnd {
            mid: "m1".into(),
            reason: "stop".into(),
        });
        app.apply_bridge_event(ev3, 0);
        app.push_user("hi".into());
        app.push_notice("已中止运行中的任务".into());
        app.push_notice("样式已更新".into());

        // Render via the LIVE path: prepare_frame then draw.
        let area = ratatui::layout::Rect::new(0, 0, w, h);
        app.prepare_frame(area, &theme);
        // The transcript region (top + height) from the SAME split render draws into —
        // and a precondition that the content actually fits (hug-top is engaged).
        let layout = split_cockpit(&app, area);
        let tx_top = layout.transcript.y as usize;
        assert!(
            app.wrap_cache.total_visual_lines() <= layout.transcript.height as usize,
            "precondition: the short transcript fits its region (hug-top engaged): \
             total={} region_h={}",
            app.wrap_cache.total_visual_lines(),
            layout.transcript.height
        );
        term.draw(|f| render(f, &app, &theme, 0)).unwrap();
        let buf = term.backend().buffer();

        let row_text = |y: usize| -> String {
            (0..w as usize)
                .map(|x| buf.content()[y * w as usize + x].symbol().chars().next().unwrap_or(' '))
                .collect()
        };

        // (1) The frozen done-line (`⠿ … for 0s`) row + the last content row above it.
        // (idle after a turn → the done-line takes the just-above-composer slot.)
        let mut done_row: Option<usize> = None;
        let mut last_content_row: Option<usize> = None;
        let mut first_content_row: Option<usize> = None;
        for y in 0..h as usize {
            let row = row_text(y);
            if row.contains('⠿') || row.contains("for 0s") {
                done_row = Some(y);
            } else if !row.trim().is_empty() {
                last_content_row = Some(y);
                if first_content_row.is_none() {
                    first_content_row = Some(y);
                }
            }
        }
        let done_y = done_row.expect("the frozen done-line row must be rendered");
        let last_y = last_content_row.expect("at least one content row must be rendered");

        // (1a) HUG TOP: the FIRST non-blank content row is the header — but the first
        // content row INSIDE the transcript region sits at its very top (not pushed
        // down). Find the first non-blank row at/below the transcript top.
        let first_tx_content = (tx_top..h as usize)
            .find(|&y| !row_text(y).trim().is_empty())
            .expect("the transcript region must contain content");
        assert_eq!(
            first_tx_content, tx_top,
            "content must HUG the top of the transcript region (row {tx_top}), \
             but the first non-blank transcript row is {first_tx_content} \
             (before the fix it was pushed down behind a blank gap)"
        );

        // (2) The done-line IMMEDIATELY follows the last content row (gap == 0). Before
        // the fix the content was bottom-anchored with the blank gap ABOVE it, so the
        // last content row still abutted the done-line — but (1a)+(3) pin the gap to the
        // bottom; together they fail the old layout. Keep gap==0 as the contiguity check.
        let gap = done_y.saturating_sub(last_y + 1);
        assert_eq!(
            gap, 0,
            "the done-line (row {done_y}) must immediately follow the last content row \
             (row {last_y}); {gap} blank rows between them"
        );

        // (3) BLANK ABSORBED AT THE BOTTOM: the final rows of the buffer (below the
        // tips row) are blank. Before the fix the block was bottom-anchored, so the
        // bottom rows were FULL (composer/info/tips) with the blank gap up top — this
        // asserts the inversion. The last 2 rows are info+tips (non-blank); the very
        // last row of the screen must be blank padding (the hug-top spacer).
        assert!(
            row_text(h as usize - 1).trim().is_empty(),
            "the very bottom row must be blank padding (the hug-top spacer absorbs the \
             gap at the bottom):\n{:?}",
            row_text(h as usize - 1)
        );

        // The tool box ╰─╯ bottom border must be visible (not cut off by viewport).
        let has_bottom_border = (0..h as usize).any(|y| {
            let row = row_text(y);
            row.contains('╰') && row.contains('╯')
        });
        assert!(
            has_bottom_border,
            "tool box bottom border ╰─╯ must be visible in the transcript"
        );
    }

    /// SLICE S6 HONEST-CHECK (R6 Part A): the sticky last-user-prompt breadcrumb is
    /// pinned at the TOP of the transcript ONLY while scrolled away from the tail.
    /// Following → absent; scrolled up → `↑ <prompt>` present. LIVE styled path.
    #[test]
    fn sticky_header_shows_when_scrolled_up_absent_when_following() {
        use crate::app::{AppState, ConnStatus};
        use crate::components::render;
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let (w, h) = (60u16, 24u16);
        let theme = Theme::default_theme();
        let area = ratatui::layout::Rect::new(0, 0, w, h);
        let mut app = AppState::new();
        app.conn = ConnStatus::Connected { model: Some("m".into()) };
        app.push_user("what is the meaning of life?".into());
        // Overflow the viewport so we can scroll up off the tail.
        for i in 0..40 {
            app.push_notice(format!("assistant detail line {i}"));
        }

        let frame_text = |app: &AppState| -> String {
            let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
            term.draw(|f| render(f, app, &theme, 0)).unwrap();
            let buf = term.backend().buffer();
            (0..h as usize)
                .map(|y| (0..w as usize).map(|x| buf.content()[y * w as usize + x].symbol()).collect::<String>())
                .collect::<Vec<_>>()
                .join("\n")
        };

        // FOLLOWING (at the tail): NO sticky breadcrumb (the spinner/done-line are not
        // shown here either, so `↑` is unique to the sticky row).
        app.prepare_frame(area, &theme);
        assert!(app.following(), "precondition: at the tail");
        let screen = frame_text(&app);
        assert!(!screen.contains('↑'), "no sticky `↑` breadcrumb while following:\n{screen}");

        // SCROLL UP: the sticky breadcrumb appears carrying the last user prompt.
        app.scroll_lines(-8);
        assert!(!app.following(), "precondition: scrolled away from the tail");
        app.prepare_frame(area, &theme);
        let screen2 = frame_text(&app);
        assert!(screen2.contains('↑'), "sticky `↑` breadcrumb appears when scrolled up:\n{screen2}");
        assert!(
            screen2.contains("what is the meaning of life"),
            "the sticky row shows the last user prompt:\n{screen2}"
        );
    }
}
