//! components/mod.rs — the FILLED ratatui cockpit layout (§5 / N5 / N8).
//!
//! One immediate-mode `render(frame, app, theme, now_ms)` draws the whole cockpit
//! each frame. The vertical layout (§5):
//!
//!   HEADER (1)      ◆ GenericAgent · tui_v4  <model> · <cwd> · <session>   <tip>
//!   SEPARATOR (1)   ▓▓▓ rainbow 7-stop, full width ▓▓▓
//!   TRANSCRIPT (Min 0 — FLEXES to fill all remaining height)
//!   SPINNER (1, only when busy)  <pet> <spinner> <Gerund>… 3.2s · ctx ▰▰▱
//!   COMPOSER (flex 1..6, bordered, multi-line; hot-pink border in shell mode)
//!   FOOTER (1)      ❯ <mode pill>   │   <model> │ ctx% (used/cap) │ $cost │ git
//!
//! The transcript uses `Constraint::Min(0)` so it fills. Assistant blocks are
//! rendered through the COCKPIT markdown layer (per-turn folds → `▸ summary`,
//! tool calls → boxed chips). No hardcoded colors: every style goes through theme
//! tokens. The composer renders the logical buffer with an inverse-cell cursor
//! and the slash-palette / `@`-file-picker dropdown.

pub mod continue_picker;
pub mod dashboard;
pub mod effects_paint;
pub mod overlay;
pub mod picker;
pub mod scheduler;

use std::collections::HashMap;

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph};
use ratatui::Frame;

use crate::app::{AppState, ConnStatus};
use crate::commands::registry::{self, SlashCommand};
use crate::flavor::{self, heat_bold, heat_token, PetStyle};
use crate::input::file_expand;
use crate::render::BlockRole;
use crate::theme::{Theme, Token};

/// Max composer height (rows) before it stops growing and scrolls internally.
const COMPOSER_MAX_ROWS: u16 = 8;

/// Draw the active full-screen VIEW for one frame: the dashboard (§6 / N2) when
/// `app.view == View::Dashboard`, else the chat cockpit. Takes `&mut AppState`
/// because the cockpit's transcript region must sync the wrap cache + viewport to
/// its on-screen geometry (P1) before the visible window can be derived.
pub fn render(frame: &mut Frame, app: &mut AppState, theme: &Theme, now_ms: u64) {
    let area = frame.area();
    // Record the full terminal size for dashboard click-mapping geometry.
    app.set_term_size(area.width, area.height);
    match app.view {
        crate::app::View::Dashboard => dashboard::render(frame, area, app, theme, now_ms),
        crate::app::View::Workflows => crate::workflow::panel::render(
            frame,
            area,
            &mut app.workflow_panel,
            &app.workflow_snapshot,
            theme,
            app.lang,
            now_ms,
        ),
        crate::app::View::Cockpit => render_cockpit(frame, app, theme, now_ms),
    }
    // A modal OVERLAY (picker / ask-user / help / cost / verbose / btw) draws on
    // TOP of the current view (§3 overlay stack). Painted last so it covers the
    // cockpit/dashboard underneath.
    if let Some(ov) = app.overlay.as_ref() {
        overlay::render(frame, area, ov, app, theme, now_ms);
    }
}

/// Draw the chat cockpit (header / transcript / composer / footer) — the normal
/// view. Split out from [`render`] so the dashboard view can short-circuit.
fn render_cockpit(frame: &mut Frame, app: &mut AppState, theme: &Theme, now_ms: u64) {
    let area = frame.area();

    // The composer flexes from 3 rows up to COMPOSER_MAX_ROWS as the buffer grows.
    let composer_inner_w = area.width.saturating_sub(4).max(1); // borders + prompt
    let composer_rows = composer_height(app, composer_inner_w);

    let show_spinner = app.busy;
    let dropdown_rows = dropdown_height(app, area.width);

    // §5 vertical split. Transcript = Min(0) so it flexes to fill.
    let mut constraints: Vec<Constraint> = vec![
        Constraint::Length(1), // header
        Constraint::Length(1), // rainbow separator
        Constraint::Min(0),    // transcript (FLEX)
    ];
    if show_spinner {
        constraints.push(Constraint::Length(1)); // spinner band
    }
    if dropdown_rows > 0 {
        constraints.push(Constraint::Length(dropdown_rows)); // palette / file picker
    }
    constraints.push(Constraint::Length(composer_rows)); // composer (bordered)
    constraints.push(Constraint::Length(1)); // status footer
    constraints.push(Constraint::Length(1)); // hint + rotating tip (bottom)

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    // Index the chunks in order.
    let mut i = 0;
    let header = chunks[i];
    i += 1;
    let sep = chunks[i];
    i += 1;
    let transcript = chunks[i];
    i += 1;
    let spinner = if show_spinner {
        let c = chunks[i];
        i += 1;
        Some(c)
    } else {
        None
    };
    let dropdown = if dropdown_rows > 0 {
        let c = chunks[i];
        i += 1;
        Some(c)
    } else {
        None
    };
    let composer = chunks[i];
    i += 1;
    let footer = chunks[i];
    i += 1;
    let hints = chunks[i];

    // Sync the wrap cache + viewport to the transcript region's real geometry
    // BEFORE rendering (the resize-reflow / P1 happens here).
    app.sync_transcript(transcript.width, transcript.height as usize, theme);

    render_header(frame, header, app, theme, now_ms);
    render_separator(frame, sep, app, theme);
    render_transcript(frame, transcript, app, theme);
    // Effects are now BORDER-BOUND (a flowing-rainbow composer border + a few
    // drifting particles), NOT a full-background fire/snow over the transcript —
    // the terminal background stays clean (user feedback). The `/effects demo`
    // splash still uses its own overlay; draw_ambient remains for that.
    if let Some(spin) = spinner {
        render_spinner(frame, spin, app, theme, now_ms);
    }
    if let Some(dd) = dropdown {
        render_dropdown(frame, dd, app, theme);
    }
    render_composer(frame, composer, app, theme);
    // Flowing-rainbow composer border + border-bound particles — GATED (redesign
    // request #4): the input box is PLAIN by default; the effects only light up when
    // the composer holds one of the orchestration commands (/hive /goal /conductor
    // /morphling) — making those feel special without an always-on distraction.
    // Shell mode keeps its hot-pink border (the fx fn early-returns there).
    if fx_command_active(app.composer.text()) {
        effects_paint::draw_composer_border_fx(frame, app, composer, now_ms);
    }
    render_footer(frame, footer, app, theme);
    render_hints(frame, hints, app, theme, now_ms);
}

/// HEADER: ONE clean line — `◆ GenericAgent · tui_v4 · <model> · <cwd>`. No tip,
/// no shortcut hints, no session counter; those live at the BOTTOM hint line / in
/// the dashboard, so the top stays uncluttered (mainstream-TUI alignment).
fn render_header(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme, _now_ms: u64) {
    // Truncate the model name (redesign_cc.md §2.5): a MixinSession's `get_llm_name`
    // is a long pipe-list — show only the primary segment, never the full chain.
    let model = truncate_model(app.model.as_deref().unwrap_or("—"), MODEL_LABEL_CAP);
    let cwd = compact_cwd(&app.cwd, 34);
    let spans = vec![
        Span::styled(
            "◆ ",
            Style::default()
                .fg(theme.color(Token::Claude))
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            "GenericAgent · tui_v4",
            Style::default()
                .fg(theme.color(Token::Text))
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("   ·   ", Style::default().fg(theme.color(Token::Dim))),
        Span::styled(model.to_string(), Style::default().fg(theme.color(Token::Claude))),
        Span::styled("   ·   ", Style::default().fg(theme.color(Token::Dim))),
        Span::styled(cwd, Style::default().fg(theme.color(Token::Dim))),
    ];
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// HINT LINE (very bottom): compact keybinding hints (left) + the rotating tip
/// (right). Mainstream TUIs keep tips/shortcuts at the BOTTOM, not crammed into
/// the header — this is the row that does it.
fn render_hints(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme, now_ms: u64) {
    let dim = Style::default().fg(theme.color(Token::Dim));
    let key = Style::default().fg(theme.color(Token::Suggestion));
    // OS-aware key labels (request #2). `←→` advertises the new view switch; the
    // Ctrl chords spell out per platform. Curated to the headline bindings so the
    // row stays within one terminal line (the right-tip drops first if it's tight).
    let ctrl = ctrl_key_label();
    let pairs: Vec<(String, String)> = vec![
        ("⏎".to_string(), "send".to_string()),
        ("←→".to_string(), "switch".to_string()),
        ("/".to_string(), "cmds".to_string()),
        ("!".to_string(), "shell".to_string()),
        ("@".to_string(), "file".to_string()),
        (format!("{ctrl}⇧C"), "copy".to_string()),
        (format!("{ctrl}C"), "quit".to_string()),
    ];
    let mut left: Vec<Span> = Vec::new();
    for (i, (k, a)) in pairs.iter().enumerate() {
        if i > 0 {
            left.push(Span::styled("  ", dim));
        }
        left.push(Span::styled(k.clone(), key));
        left.push(Span::styled(format!(" {a}"), dim));
    }
    let left_w: usize = left
        .iter()
        .map(|s| unicode_width::UnicodeWidthStr::width(s.content.as_ref()))
        .sum();
    // The RIGHT side normally shows the rotating tip (deterministic by tick, ~12s).
    // But while the 3-stage Ctrl+C is ARMED (§8), it shows the "press Ctrl+C again
    // to quit" hint in the WARNING color instead — a transient, non-polluting hint
    // (no transcript notice) that vanishes when the 2s arm expires.
    let armed = app.chord.ctrl_c_hint_active(now_ms);
    let (right_text_src, right_style) = if armed {
        (crate::i18n::t(app.lang, "ctrlc.arm"), Style::default().fg(theme.color(Token::Warning)))
    } else {
        (flavor::tip(app.lang, now_ms / 100), dim)
    };
    let avail = (area.width as usize).saturating_sub(left_w + 3);
    let mut spans = left;
    if avail > 16 {
        // No emoji icon (user: lightbulb-style icons look cheap). Just text — the
        // tip already reads "Tip: …"; the armed hint reads "press Ctrl+C again to quit".
        let right_text = clip_to(right_text_src, avail);
        let pad = (area.width as usize)
            .saturating_sub(left_w + unicode_width::UnicodeWidthStr::width(right_text.as_str()));
        spans.push(Span::raw(" ".repeat(pad)));
        spans.push(Span::styled(right_text, right_style));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// SEPARATOR: the full-width rainbow 7-stop line (§5 / §9). Upgraded from the old flat
/// `▓` rule to the effects-engine separator: a smoothly INTERPOLATED ROYGBIV gradient
/// that honors the capability gate (a plain dim line at mono / NO_COLOR) and animates a
/// raised-cosine SHIMMER sweep while the running indicator is active (subtle/full/demo).
fn render_separator(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
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

/// TRANSCRIPT: the flex region (P1). Renders the viewport's VISIBLE WINDOW.
fn render_transcript(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
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

    // Memoize each assistant block's styled cockpit lines for THIS frame.
    let mut md_cache: HashMap<u64, Vec<Line<'static>>> = HashMap::new();
    let fold_all = app.fold_all;

    let window = app.viewport.visible(&app.wrap_cache);
    let mut lines: Vec<Line> = Vec::with_capacity(window.len());
    for vl in &window {
        let block = block_of(vl.block_id);
        let role = block.map(|b| b.render_role()).unwrap_or(BlockRole::Assistant);

        if role == BlockRole::Assistant {
            let styled = md_cache.entry(vl.block_id).or_insert_with(|| {
                let src = block.map(|b| b.source.as_str()).unwrap_or("");
                let streaming = block.map(|b| !b.finalized).unwrap_or(false);
                crate::markdown::render_assistant_cockpit_streaming(
                    src, theme, fold_all, area.width, streaming,
                )
            });
            let line = styled.get(vl.intra).cloned().unwrap_or_else(|| {
                Line::from(Span::styled(
                    vl.text.clone(),
                    Style::default().fg(theme.color(Token::Text)),
                ))
            });
            lines.push(line);
            continue;
        }

        // USER message → a full-width inverse BAND (redesign_cc.md §2.1): bg
        // `userMessageBackground` rgb(58,58,58), white text, the prompt sitting in
        // the band (left-padded one cell), every row padded to the full width so
        // the band spans edge-to-edge. This replaces the bare `❯ hello` gutter.
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

    frame.render_widget(Paragraph::new(lines), area);
}

/// Build one full-width USER band row (redesign_cc.md §2.1): the row `text` (one
/// soft-wrapped visual line of the user message) rendered with bg `UserBand`
/// rgb(58,58,58) + white `Text`, left-padded one cell and right-padded so the band
/// spans the FULL terminal width. PURE-ish (themed `Line`). The whole row carries
/// the bg so it reads as a solid inverse band, like CC's `UserPromptMessage`.
fn user_band_line<'a>(text: &str, width: u16, theme: &Theme) -> Line<'a> {
    let band = Style::default()
        .bg(theme.color(Token::UserBand))
        .fg(theme.color(Token::Text));
    let w = width as usize;
    // " " + text (clipped so lead+text never exceeds the width), then right-pad
    // with spaces to the full width (CJK-correct) so the band spans edge-to-edge.
    let lead = " ";
    let body = clip_to(text, w.saturating_sub(1));
    let used = 1 + unicode_width::UnicodeWidthStr::width(body.as_str());
    let pad = w.saturating_sub(used);
    Line::from(vec![
        Span::styled(lead.to_string(), band),
        Span::styled(body, band),
        Span::styled(" ".repeat(pad), band),
    ])
}

/// The speaker gutter glyph + accent token + body token for a role.
fn gutter_for(role: BlockRole) -> (&'static str, Token, Token) {
    match role {
        BlockRole::User => ("❯ ", Token::Claude, Token::Text),
        BlockRole::Assistant => ("", Token::Text, Token::Text),
        BlockRole::System => ("» ", Token::Suggestion, Token::Dim),
        BlockRole::Tool => ("⚙ ", Token::Warning, Token::Dim),
        BlockRole::Notice => ("• ", Token::Warning, Token::Dim),
    }
}

/// SPINNER band (only when busy): <pet> <spinner> <Gerund>… <elapsed>s · ctx bar.
fn render_spinner(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme, now_ms: u64) {
    let elapsed = app.turn_elapsed_ms(now_ms);
    let tick = (now_ms / 100) as u64;
    let glyph = app.spinner_style.glyph(tick);
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
    spans.push(Span::styled(format!("{}…", flavor::gerund(tick)), text));

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

/// COMPOSER: the bordered multi-line input with an inverse-cell cursor. The
/// border tints HOT-PINK (AutoAccept coral token) in shell mode (`!cmd`).
fn render_composer(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
    let shell = app.composer.is_shell_mode();
    // Shell mode → hot-pink border + mark; else the normal border / claude mark.
    let border_tok = if shell {
        Token::ShellAccent
    } else if app.busy {
        Token::Dim
    } else {
        Token::Border
    };
    let mark_tok = if shell { Token::ShellAccent } else { Token::Claude };
    // Rounded corners to match the modern CC/Codex composer aesthetic
    // (clip_20260531_160036). When the flowing-rainbow border is active (the
    // /hive…/morphling gate) the effects painter overwrites these corner cells
    // with its own glyphs, so a single border type here is fine for both states.
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme.color(border_tok)));

    let inner_w = area.width.saturating_sub(4).max(1); // borders + "❯ "
    // Shell mode: the buffer ALREADY starts with the typed `!`, which serves as
    // the prompt — so we must NOT also prepend a `! ` mark, or it renders `! !ls`
    // (the double-`!` bug). The hot-pink border + bottom-hint signal shell mode.
    let mark = if shell { "" } else { "❯ " };

    let lines: Vec<Line> = if app.composer.is_empty() {
        let placeholder = if shell {
            crate::i18n::t(app.lang, "composer.placeholder.shell")
        } else {
            crate::i18n::t(app.lang, "composer.placeholder")
        };
        vec![Line::from(vec![
            Span::styled(mark, Style::default().fg(theme.color(mark_tok))),
            // The cursor cell on an empty buffer.
            Span::styled(" ", Style::default().add_modifier(Modifier::REVERSED)),
            Span::styled(
                placeholder,
                Style::default().fg(theme.color(Token::Dim)),
            ),
        ])]
    } else {
        composer_lines(app, theme, inner_w, mark, mark_tok, shell)
    };

    frame.render_widget(Paragraph::new(lines).block(block), area);
}

/// Build the composer's styled rows with the inverse-cell cursor placed at its
/// visual (row, col). The buffer is rendered as its logical lines (the visual
/// rows from the composer's wrap geometry); the cursor cell on the active row is
/// inverted (robust across terminals while we own the screen).
fn composer_lines<'a>(
    app: &'a AppState,
    theme: &Theme,
    inner_w: u16,
    mark: &'a str,
    mark_tok: Token,
    shell: bool,
) -> Vec<Line<'a>> {
    let rows = app.composer.visual_rows(inner_w);
    let (cur_row, cur_col) = app.composer.cursor_rc(inner_w);
    let text = app.composer.text();
    let text_style = Style::default().fg(theme.color(Token::Text));
    let shell_style = Style::default().fg(theme.color(Token::ShellAccent));

    let mut out: Vec<Line> = Vec::with_capacity(rows.len());
    for (ri, r) in rows.iter().enumerate() {
        let row_text = &text[r.start..r.end.min(text.len())];
        let mut spans: Vec<Span> = Vec::new();
        // The prompt mark only on the first row; a 2-space hanging indent after.
        if ri == 0 {
            spans.push(Span::styled(mark, Style::default().fg(theme.color(mark_tok))));
        } else {
            spans.push(Span::raw("  "));
        }
        // Shell mode: the buffer's OWN leading `!` is the prompt sigil — paint it in
        // hot-pink ShellAccent (matching the pink border + footer `!`), exactly like
        // Claude Code's bash `!` (bashBorder). Only on row 0, and only when the
        // cursor isn't sitting on that `!` cell (col 0) — in that case fall through
        // so the `!` still gets the inverse cursor cell.
        let bang_pink = shell
            && ri == 0
            && row_text.starts_with('!')
            && !(ri == cur_row && cur_col == 0);
        let (lead_pink, row_text) = if bang_pink {
            (true, &row_text[1..])
        } else {
            (false, row_text)
        };
        if lead_pink {
            // `! ` (with a trailing space) so the command reads `! ls -la`, not
            // `!ls -la` — a clear gap after the bash sigil (user feedback).
            spans.push(Span::styled("! ", shell_style));
        }
        if ri == cur_row {
            // The cursor column is measured from the row start; the peeled `!`
            // occupied visual col 0, so shift the split column left by 1.
            let split_col = if lead_pink { cur_col.saturating_sub(1) } else { cur_col };
            // Split the row around the cursor column and invert one cell.
            let (before, at, after) = split_at_col(row_text, split_col);
            spans.push(Span::styled(before.to_string(), text_style));
            spans.push(Span::styled(
                at.to_string(),
                Style::default().add_modifier(Modifier::REVERSED),
            ));
            spans.push(Span::styled(after.to_string(), text_style));
        } else {
            spans.push(Span::styled(row_text.to_string(), text_style));
        }
        out.push(Line::from(spans));
    }
    if out.is_empty() {
        out.push(Line::from(Span::styled(mark, Style::default().fg(theme.color(mark_tok)))));
    }
    out
}

/// Split a row's text at visual column `col` into (before, cursor-cell, after).
/// The cursor cell is the grapheme at `col` (a space if past end-of-line).
fn split_at_col(row: &str, col: usize) -> (String, String, String) {
    use unicode_segmentation::UnicodeSegmentation;
    use unicode_width::UnicodeWidthStr;
    let mut acc = 0usize;
    let mut before = String::new();
    let mut at = String::new();
    let mut after = String::new();
    let mut placed = false;
    for g in row.graphemes(true) {
        let gw = UnicodeWidthStr::width(g);
        if !placed && acc >= col {
            at.push_str(g);
            placed = true;
        } else if !placed {
            before.push_str(g);
        } else {
            after.push_str(g);
        }
        acc += gw;
    }
    if !placed {
        // Cursor at/after end-of-line → a blank cursor cell.
        at.push(' ');
    }
    (before, at, after)
}

/// FOOTER: prompt char + mode pill (left); model │ ctx% (used/cap) │ $cost │ git
/// (right). Codex/CC fusion (§5).
fn render_footer(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
    let shell = app.composer.is_shell_mode();
    // Left: prompt char + mode pill.
    // LEFT: a CC/Codex-style mode indicator — a DIM prompt caret + the lowercase
    // mode word in a foreground accent color. Neither CC nor Codex paints a
    // background pill here; the old inverse `❯ chat ` chip (a bg fill) was the ugly
    // part the user called out. Now it's plain colored text on the default bg.
    let (mark, mark_tok) = if shell {
        ("! ", Token::ShellAccent)
    } else {
        ("❯ ", Token::Dim)
    };
    let (mode_word, mode_tok) = mode_indicator(app);
    let mut left: Vec<Span> = vec![
        Span::styled(mark.to_string(), Style::default().fg(theme.color(mark_tok))),
        Span::styled(mode_word.to_string(), Style::default().fg(theme.color(mode_tok))),
    ];
    // A connection chip when not connected (so a failure is always visible, N1),
    // joined by a dim middot (CC's ` · ` footer spacing) — no background.
    if !matches!(app.conn, ConnStatus::Connected { .. }) {
        let conn_tok = match &app.conn {
            ConnStatus::Connecting => Token::Warning,
            ConnStatus::Disconnected { .. } => Token::Error,
            ConnStatus::Connected { .. } => Token::Success,
        };
        left.push(Span::styled(" · ", Style::default().fg(theme.color(Token::Dim))));
        left.push(Span::styled(
            app.conn.label(),
            Style::default().fg(theme.color(conn_tok)),
        ));
    }

    // Right: <model> · ctx <pct> · $<cost> · <git> (redesign_cc.md §2.5). The model
    // name is TRUNCATED to its primary segment (never the MixinSession pipe-list);
    // segments are joined by a dim middot ` · ` — CC's clean footer spacing, not the
    // old heavy `│` rules.
    let model = truncate_model(app.model.as_deref().unwrap_or("—"), MODEL_LABEL_CAP);
    let ctx = match app.context_percent {
        Some(p) => format!("ctx {p:.0}%"),
        None => "ctx —".to_string(),
    };
    // GA has no per-token price table, so a $0.00 would be misleading. When the
    // cost is unknown, show the REAL cumulative token total instead (what the user
    // actually wants to see) — `$X.XX` only once a real cost is known.
    let cost = if app.cost_usd > 0.0 {
        format!("${:.2}", app.cost_usd)
    } else {
        match app.tokens {
            Some(tk) => format!("{} tok", human_count(tk)),
            None => "—".to_string(),
        }
    };
    let git = app.git_branch.as_deref().unwrap_or("—");
    // Measure the right block width from the same pieces we render (so the
    // space-between padding is exact): `model · ctx · cost · git`, sep = ` · `.
    let sep = " · ";
    let right_w = unicode_width::UnicodeWidthStr::width(model.as_str())
        + unicode_width::UnicodeWidthStr::width(ctx.as_str())
        + unicode_width::UnicodeWidthStr::width(cost.as_str())
        + unicode_width::UnicodeWidthStr::width(git)
        + sep.chars().count() * 3;

    let left_w: usize = left
        .iter()
        .map(|s| unicode_width::UnicodeWidthStr::width(s.content.as_ref()))
        .sum();
    let pad = (area.width as usize).saturating_sub(left_w + right_w);

    let sep_style = Style::default().fg(theme.color(Token::Dim));
    let mut spans = left;
    spans.push(Span::raw(" ".repeat(pad)));
    spans.push(Span::styled(model, Style::default().fg(theme.color(Token::Text))));
    spans.push(Span::styled(sep, sep_style));
    spans.push(Span::styled(ctx, Style::default().fg(theme.color(Token::Dim))));
    spans.push(Span::styled(sep, sep_style));
    spans.push(Span::styled(cost, Style::default().fg(theme.color(Token::Success))));
    spans.push(Span::styled(sep, sep_style));
    spans.push(Span::styled(git.to_string(), Style::default().fg(theme.color(Token::Suggestion))));

    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// The mode-indicator word (localized) + its FOREGROUND token (no background —
/// CC/Codex render the mode as colored text, not an inverse pill). Shell mode wins
/// the word so it stays consistent with the `! ` caret; else busy → running.
fn mode_indicator(app: &AppState) -> (&'static str, Token) {
    if app.composer.is_shell_mode() {
        (crate::i18n::t(app.lang, "footer.mode.bash"), Token::ShellAccent)
    } else if app.busy {
        (crate::i18n::t(app.lang, "footer.mode.running"), Token::Warning)
    } else {
        (crate::i18n::t(app.lang, "footer.mode.chat"), Token::Suggestion)
    }
}

/// A tiny context bar `▰▰▰▱▱ 50%` for a percent. PURE-ish helper.
fn ctx_bar(pct: f64) -> String {
    let filled = ((pct / 100.0) * 5.0).round().clamp(0.0, 5.0) as usize;
    let bar: String = (0..5).map(|i| if i < filled { '▰' } else { '▱' }).collect();
    format!("{bar} {pct:.0}%")
}

// ---- the dropdown (slash palette / @ file picker) --------------------------

/// How many rows the active completion dropdown needs (0 = none).
fn dropdown_height(app: &AppState, _width: u16) -> u16 {
    if app.composer.is_empty() {
        return 0;
    }
    let text = app.composer.text();
    let matches = registry::palette_matches(text);
    if registry::palette_visible(text, &matches) {
        return (matches.len().min(registry::PALETTE_ROWS) as u16) + 1; // +1 hint row.
    }
    if let Some(q) = app.composer.at_query() {
        let files = app.list_project_files();
        let ranked = file_expand::rank_files(&q.partial, &files);
        if !ranked.is_empty() {
            return (ranked.len().min(file_expand::MAX_PICKER_ROWS) as u16) + 1;
        }
    }
    0
}

/// Render the active completion dropdown (palette OR file picker).
fn render_dropdown(frame: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
    let text = app.composer.text();
    let matches = registry::palette_matches(text);
    if registry::palette_visible(text, &matches) {
        render_palette(frame, area, &matches, app, theme);
        return;
    }
    if let Some(q) = app.composer.at_query() {
        let files = app.list_project_files();
        let ranked = file_expand::rank_files(&q.partial, &files);
        render_file_picker(frame, area, &ranked, app.composer.file_sel, theme, app.lang);
    }
}

/// The slash-command palette dropdown.
fn render_palette(
    frame: &mut Frame,
    area: Rect,
    matches: &[SlashCommand],
    app: &AppState,
    theme: &Theme,
) {
    // The highlighted row (↑/↓ move `app.palette_sel`; Tab/Enter complete it),
    // clamped to the live match count. Previously hardcoded to 0 so the highlight
    // was stuck on the top row regardless of arrow keys.
    let sel = app.palette_sel.min(matches.len().saturating_sub(1));
    let (start, slice) = registry::palette_window(matches, sel);
    let mut lines: Vec<Line> = Vec::new();
    for (i, cmd) in slice.iter().enumerate() {
        let active = start + i == sel;
        let (cursor, name_tok) = if active {
            ("❯ ", Token::Suggestion)
        } else {
            ("  ", Token::Text)
        };
        lines.push(Line::from(vec![
            Span::styled(cursor, Style::default().fg(theme.color(Token::Suggestion))),
            Span::styled(
                format!("/{}", cmd.name),
                Style::default()
                    .fg(theme.color(name_tok))
                    .add_modifier(if active { Modifier::BOLD } else { Modifier::empty() }),
            ),
            Span::styled("   ", Style::default()),
            Span::styled(cmd.desc.to_string(), Style::default().fg(theme.color(Token::Dim))),
        ]));
    }
    lines.push(Line::from(Span::styled(
        crate::i18n::t(app.lang, "palette.hint"),
        Style::default().fg(theme.color(Token::Dim)),
    )));
    frame.render_widget(Paragraph::new(lines), area);
}

/// The `@`-file-picker dropdown.
fn render_file_picker(
    frame: &mut Frame,
    area: Rect,
    files: &[String],
    sel: usize,
    theme: &Theme,
    lang: crate::i18n::Lang,
) {
    let sel = if files.is_empty() { 0 } else { sel % files.len() };
    let mut lines: Vec<Line> = Vec::new();
    for (i, f) in files.iter().take(file_expand::MAX_PICKER_ROWS).enumerate() {
        let active = i == sel;
        lines.push(Line::from(vec![
            Span::styled(
                if active { "❯ " } else { "  " },
                Style::default().fg(theme.color(Token::Suggestion)),
            ),
            Span::styled(
                f.clone(),
                Style::default()
                    .fg(theme.color(if active { Token::Suggestion } else { Token::Text }))
                    .add_modifier(if active { Modifier::BOLD } else { Modifier::empty() }),
            ),
        ]));
    }
    lines.push(Line::from(Span::styled(
        crate::i18n::t(lang, "filepicker.hint"),
        Style::default().fg(theme.color(Token::Dim)),
    )));
    frame.render_widget(Paragraph::new(lines), area);
}

// ---- pure layout helpers (unit-tested) -------------------------------------

/// The composer's rendered height (rows + 2 for the border), clamped to
/// `[3, COMPOSER_MAX_ROWS+2]`. PURE over the buffer geometry.
fn composer_height(app: &AppState, inner_w: u16) -> u16 {
    let rows = app.composer.visual_rows(inner_w).len() as u16;
    (rows.clamp(1, COMPOSER_MAX_ROWS)) + 2
}

/// Shorten a long cwd so the header never overflows. Keeps the tail (most
/// informative) with a leading ellipsis. PURE + unit-tested.
pub fn compact_cwd(cwd: &str, max: usize) -> String {
    let max = max.max(8);
    if cwd.chars().count() <= max {
        return cwd.to_string();
    }
    let tail: String = cwd
        .chars()
        .rev()
        .take(max - 1)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("…{tail}")
}

/// Default cap (display cells) for the footer/header model label (redesign_cc.md
/// §2.5: "cap ~22 cells").
pub const MODEL_LABEL_CAP: usize = 22;

/// Shorten the model label for the footer/header (redesign_cc.md §2.5). GA's
/// `get_llm_name()` for a MixinSession is a long pipe-list of the whole fallback
/// chain — e.g. `codex-pro|gpt-5|claude-opus|…|kiro` or `MixinSession/codex-pro|…`.
/// We show ONLY the PRIMARY segment (the first pipe member, the model that's
/// actually driving the turn): `codex-pro`, or `MixinSession·codex-pro` when a
/// `SessionType/` prefix is present. Never the full pipe-list. Capped to `cap`
/// display cells with a trailing `…`. PURE + unit-tested (`truncate_model_*`).
pub fn truncate_model(raw: &str, cap: usize) -> String {
    let raw = raw.trim();
    if raw.is_empty() {
        return "—".to_string();
    }
    // Peel an optional `SessionType/rest` prefix (GA emits `SessionType/name`).
    // Use the FIRST `/` so a model name that itself contains `/` keeps its tail in
    // the pipe-list step below.
    let (prefix, rest) = match raw.split_once('/') {
        Some((p, r)) if !p.is_empty() && !r.is_empty() => (Some(p.trim()), r.trim()),
        _ => (None, raw),
    };
    // The PRIMARY segment = the first pipe-separated member of the chain.
    let primary = rest.split('|').next().unwrap_or(rest).trim();
    let primary = if primary.is_empty() { rest } else { primary };
    // `SessionType·primary` (middot join) when a prefix was present; else bare.
    let label = match prefix {
        Some(p) => format!("{p}·{primary}"),
        None => primary.to_string(),
    };
    // Cap to `cap` cells; if it overflows, clip to `cap-1` and append `…`.
    use unicode_width::UnicodeWidthStr;
    if UnicodeWidthStr::width(label.as_str()) <= cap {
        return label;
    }
    // Prefer keeping the bare primary if the `SessionType·` join is what blew the
    // cap (the model name is the load-bearing part) — but only if it then fits.
    if prefix.is_some() && UnicodeWidthStr::width(primary) <= cap {
        return primary.to_string();
    }
    let body = clip_to(&label, cap.saturating_sub(1));
    format!("{body}…")
}

/// Whether the composer currently holds one of the ORCHESTRATION commands that
/// light up the input-box effects (redesign #4): `/hive` `/goal` `/conductor`
/// `/morphling`. Matches the command WORD at the start, so `/hive do x` counts but
/// `/hivemind` does not, and a plain message never triggers it. PURE + unit-tested.
pub fn fx_command_active(text: &str) -> bool {
    let Some(rest) = text.trim_start().strip_prefix('/') else {
        return false;
    };
    let word = rest.split(|c: char| c.is_whitespace()).next().unwrap_or("");
    matches!(word, "hive" | "goal" | "conductor" | "morphling")
}

/// The platform's Ctrl-modifier label for the hint row (request #2: "detect the
/// system, show different bindings"). macOS uses the compact `⌃` symbol; other
/// platforms spell `Ctrl-` (the `⌃` glyph reads as noise off-mac). Compile-time
/// `cfg!` equals the running OS for a native binary, so each release artifact (the
/// win .exe, the mac .dmg, the linux build) shows its own convention. PURE.
pub fn ctrl_key_label() -> &'static str {
    if cfg!(target_os = "macos") {
        "⌃"
    } else {
        "Ctrl-"
    }
}

/// Compact token count for the spinner readout: `950 → "950"`, `1234 → "1.2k"`,
/// `2_300_000 → "2.3m"` (CC's `formatNumber` / tui_v3's `_human`). PURE.
pub fn human_count(n: u64) -> String {
    if n < 1000 {
        n.to_string()
    } else if n < 1_000_000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        format!("{:.1}m", n as f64 / 1_000_000.0)
    }
}

/// Clip a string to at most `max` display cells (no ellipsis). PURE. Shared with
/// the dashboard component (name/preview truncation).
pub fn clip_to(s: &str, max: usize) -> String {
    use unicode_segmentation::UnicodeSegmentation;
    use unicode_width::UnicodeWidthStr;
    if UnicodeWidthStr::width(s) <= max {
        return s.to_string();
    }
    let mut out = String::new();
    let mut acc = 0usize;
    for g in s.graphemes(true) {
        let gw = UnicodeWidthStr::width(g);
        if acc + gw > max {
            break;
        }
        out.push_str(g);
        acc += gw;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_cwd_keeps_tail() {
        let long = "/very/long/path/to/some/deeply/nested/project/dir";
        let out = compact_cwd(long, 20);
        assert!(out.chars().count() <= 20);
        assert!(out.starts_with('…'));
        assert!(out.ends_with("project/dir"));
        assert_eq!(compact_cwd("/a/b", 40), "/a/b");
    }

    #[test]
    fn ctx_bar_fills_proportionally() {
        assert_eq!(ctx_bar(0.0), "▱▱▱▱▱ 0%");
        assert_eq!(ctx_bar(100.0), "▰▰▰▰▰ 100%");
        assert!(ctx_bar(50.0).contains("50%"));
    }

    #[test]
    fn split_at_col_inverts_one_cell() {
        // Cursor at col 2 of "abcd" → before "ab", at "c", after "d".
        let (b, at, a) = split_at_col("abcd", 2);
        assert_eq!((b.as_str(), at.as_str(), a.as_str()), ("ab", "c", "d"));
        // Cursor past end → a blank cursor cell.
        let (b, at, a) = split_at_col("ab", 5);
        assert_eq!((b.as_str(), at.as_str(), a.as_str()), ("ab", " ", ""));
        // CJK: col counts cells; cursor at col 2 lands after one wide glyph.
        let (b, at, _a) = split_at_col("你好", 2);
        assert_eq!(b, "你");
        assert_eq!(at, "好");
    }

    #[test]
    fn clip_to_respects_cells() {
        assert_eq!(clip_to("hello", 3), "hel");
        assert_eq!(clip_to("你好世界", 4), "你好"); // 2 wide glyphs = 4 cells.
        assert_eq!(clip_to("hi", 10), "hi");
    }

    /// The compact token formatter (spinner `↑in ↓out`): thousands → `k`, millions
    /// → `m`; small counts stay literal.
    #[test]
    fn human_count_compacts_thousands_and_millions() {
        assert_eq!(human_count(0), "0");
        assert_eq!(human_count(950), "950");
        assert_eq!(human_count(1234), "1.2k");
        assert_eq!(human_count(340), "340");
        assert_eq!(human_count(2_300_000), "2.3m");
    }

    /// The input-box effects (redesign #4) light up ONLY for the orchestration
    /// commands — matched on the command word at the start, not a substring.
    #[test]
    fn fx_command_active_only_for_orchestration() {
        assert!(fx_command_active("/hive"));
        assert!(fx_command_active("/goal build the thing"));
        assert!(fx_command_active("  /conductor"));
        assert!(fx_command_active("/morphling absorb a skill"));
        // NOT a longer word that merely starts with one, a different command,
        // a mid-line slash, or a plain message.
        assert!(!fx_command_active("/hivemind"));
        assert!(!fx_command_active("/goalkeeper"));
        assert!(!fx_command_active("/help"));
        assert!(!fx_command_active("hello /hive"));
        assert!(!fx_command_active(""));
        assert!(!fx_command_active("just a normal message"));
    }

    /// `truncate_model` shows ONLY the primary segment of a MixinSession's pipe-list
    /// (redesign_cc.md §2.5) — `MixinSession·codex-pro`, never `…|kiro`, capped to
    /// ~22 cells.
    #[test]
    fn truncate_model_primary_segment() {
        use unicode_width::UnicodeWidthStr;
        let cap = MODEL_LABEL_CAP;

        // The real MixinSession shape: `SessionType/primary|b|c|…|kiro`.
        let raw = "MixinSession/codex-pro|gpt-5.2|claude-opus-4|gemini-2.5-pro|grok-4|kiro";
        let out = truncate_model(raw, cap);
        assert_eq!(out, "MixinSession·codex-pro", "primary segment with the session prefix");
        assert!(UnicodeWidthStr::width(out.as_str()) <= cap, "within the ~22-cell cap");
        // The full pipe-list NEVER survives.
        assert!(!out.contains('|'), "no pipe-list");
        assert!(!out.contains("kiro"), "no trailing chain member");
        assert!(!out.contains("gpt-5.2"), "no secondary segments");

        // A bare pipe-list (no `SessionType/`) → just the primary, no middot prefix.
        assert_eq!(truncate_model("codex-pro|gpt-5|claude|kiro", cap), "codex-pro");

        // A plain single model passes through unchanged.
        assert_eq!(truncate_model("gpt-5.2-mini", cap), "gpt-5.2-mini");
        // The simple `SessionType/name` (no pipes) → `SessionType·name`.
        assert_eq!(truncate_model("MixinSession/codex-pro", cap), "MixinSession·codex-pro");

        // Empty / blank → the em-dash placeholder.
        assert_eq!(truncate_model("", cap), "—");
        assert_eq!(truncate_model("   ", cap), "—");

        // Over-cap: a long bare primary is clipped with a trailing `…` to the cap.
        let long = "supercalifragilistic-model-name|fallback";
        let out = truncate_model(long, cap);
        assert!(UnicodeWidthStr::width(out.as_str()) <= cap);
        assert!(out.ends_with('…'));
        assert!(!out.contains("fallback"));

        // Over-cap where the `SessionType·` join blows the budget but the bare
        // primary fits → drop the prefix, keep the (load-bearing) model name whole.
        let out = truncate_model("VeryLongSessionTypeName/codex-pro|x|y", cap);
        assert_eq!(out, "codex-pro");
        assert!(UnicodeWidthStr::width(out.as_str()) <= cap);
    }

    /// A single USER band row carries the `UserBand` bg across its FULL width with
    /// white `Text` fg (redesign_cc.md §2.1) — the helper that builds the band.
    #[test]
    fn user_band_line_spans_width_with_band_bg() {
        use ratatui::style::Color;
        let theme = Theme::ga_default();
        let line = user_band_line("hello", 40, &theme);
        // Every span carries the band bg + white fg.
        for span in &line.spans {
            assert_eq!(span.style.bg, Some(theme.color(Token::UserBand)));
            assert_eq!(span.style.fg, Some(theme.color(Token::Text)));
        }
        // The bg is the exact CC userMessageBackground rgb(58,58,58).
        assert_eq!(theme.color(Token::UserBand), Color::Rgb(58, 58, 58));
        // The band spans the full 40 cells (lead + text + right pad).
        let total: usize = line
            .spans
            .iter()
            .map(|s| unicode_width::UnicodeWidthStr::width(s.content.as_ref()))
            .sum();
        assert_eq!(total, 40, "the band fills the whole width");
        // The text sits in the band (after a one-cell lead).
        let joined: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(joined.starts_with(" hello"));
    }

    /// THE deliverable test (§2.1): a rendered cockpit frame draws the user message
    /// as a full-width band — the buffer cells on the user row carry bg
    /// rgb(58,58,58) edge-to-edge, NOT a bare `❯ hello`.
    #[test]
    fn user_row_has_band_bg() {
        use crate::app::{AppState, ConnStatus};
        use ratatui::backend::TestBackend;
        use ratatui::style::Color;
        use ratatui::Terminal;

        let (w, h) = (60u16, 16u16);
        let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
        let mut app = AppState::new();
        app.conn = ConnStatus::Connected { model: Some("m".into()) };
        app.push_user("hello world".into());
        let theme = Theme::ga_default();
        term.draw(|f| render(f, &mut app, &theme, 0)).unwrap();

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
        // The whole row is the band: every cell on it has bg rgb(58,58,58).
        for x in 0..w as usize {
            let cell = &buf.content()[y * w as usize + x];
            assert_eq!(
                cell.bg, band,
                "user-band cell ({x},{y}) must carry the band bg, got {:?}",
                cell.bg
            );
        }
        // It is NOT a bare `❯ ` gutter prompt.
        let mut row = String::new();
        for x in 0..w as usize {
            row.push_str(buf.content()[y * w as usize + x].symbol());
        }
        assert!(!row.trim_start().starts_with('❯'), "no bare `❯` prompt in the band row");
    }
}
