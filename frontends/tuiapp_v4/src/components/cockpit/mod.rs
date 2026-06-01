//! components/cockpit/ — the FILLED ratatui cockpit layout (§5 / N5 / N8).
//!
//! One immediate-mode `render(frame, app, theme, now_ms)` draws the whole cockpit
//! each frame. The vertical layout (§5 / Q7):
//!
//!   HEADER (8)      ╭ rounded box: `>_ GenericAgent` / model / directory / session ╮
//!   SEPARATOR (1)   ▓▓▓ rainbow 7-stop, full width ▓▓▓
//!   TRANSCRIPT (Min 0 — FLEXES to fill all remaining height)
//!   SPINNER (1, only when busy)  ⠙ <Gerund>… (3.2s · ↓ 1.6k · thinking max effort)
//!   SPINNER-TIP (1, only when busy)  ⎿ <rotating tip>   (corner-continuation)
//!     ─ or, when idle right after a turn ─
//!   DONE-LINE (1)   ⠿ <Gerund> for <dur> · ↑ <in> · ↓ <out>   (frozen)
//!   COMPOSER (flex 1..8, bordered, multi-line; hot-pink border in shell mode)
//!   ROW1 (1)        <llm> · <model> · <effort> · <ctx> · <branch>  [conn chip]
//!   ROW2 (1, idle only)  ⎿ <rotating tip>   (moves UNDER the spinner when busy)
//!
//! The transcript uses `Constraint::Min(0)` so it fills. Assistant blocks are
//! rendered through the COCKPIT markdown layer (per-turn folds → `▸ summary`,
//! tool calls → boxed chips). No hardcoded colors: every style goes through theme
//! tokens. The composer renders the logical buffer with an inverse-cell cursor
//! and the slash-palette / `@`-file-picker dropdown. The per-widget bodies live in
//! the sibling submodules (header / transcript / composer / footer / dropdown);
//! this module owns only the layout dispatch.

mod composer;
mod dropdown;
mod footer;
mod header;
mod transcript;

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::Frame;

use crate::app::AppState;
use crate::theme::Theme;

use composer::render_composer;
use dropdown::{composer_height, dropdown_height, render_dropdown};
use footer::{
    render_done_line, render_separator, render_session_info, render_spinner, render_tips,
};
use header::{render_header, HEADER_ROWS};
use transcript::render_transcript;

/// Max composer height (rows) before it stops growing and scrolls internally.
const COMPOSER_MAX_ROWS: u16 = 8;

/// Draw the active full-screen VIEW for one frame: the dashboard (§6 / N2) when
/// `app.view == View::Dashboard`, else the chat cockpit. Takes `&AppState`
/// (IMMUTABLE — P11): render is now `y = f(x)`. The per-frame state writes
/// (`set_term_size` + transcript wrap-cache/viewport sync) are hoisted into
/// [`AppState::prepare_frame`], which the event loop calls BEFORE `terminal.draw`.
pub fn render(frame: &mut Frame, app: &AppState, theme: &Theme, now_ms: u64) {
    let area = frame.area();
    match app.view {
        crate::app::View::Dashboard => super::dashboard::render(frame, area, app, theme, now_ms),
        crate::app::View::Workflows => crate::workflow::panel::render(
            frame,
            area,
            &app.workflow_panel,
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
        super::overlay::render(frame, area, ov, app, theme, now_ms);
    }
}

/// The cockpit's named region Rects for one frame, derived purely from
/// `(app, area)`. Single source of geometry truth so [`AppState::prepare_frame`]
/// (which syncs the wrap cache to `transcript`) and [`render_cockpit`] (which draws
/// into the SAME Rects) can never drift — the sync is hoisted out of render (P11).
pub(crate) struct CockpitLayout {
    pub header: Rect,
    pub sep: Rect,
    pub transcript: Rect,
    /// `Some` only when the spinner band is shown (`app.busy`).
    pub spinner: Option<Rect>,
    /// `Some` only when the spinner is shown: the `⎿ Tip` row DIRECTLY UNDER the
    /// spinner status line (above the composer), so the corner reads as a
    /// continuation of the status line (Slice 5 / R4 item 5). Mirrors the
    /// `spinner` Option wiring.
    pub spinner_tip: Option<Rect>,
    /// `Some` only when the frozen done-line is shown (idle + a turn just ran).
    /// Mutually exclusive with `spinner` (busy XOR done) — they share the
    /// just-above-composer slot.
    pub done: Option<Rect>,
    /// `Some` only when the completion dropdown has rows.
    pub dropdown: Option<Rect>,
    pub composer: Rect,
    /// ROW 1 below the composer: runtime session info (Q7) — always present.
    pub info: Rect,
    /// ROW 2 below the composer: `⎿ Tips`. `Some` only when IDLE; while busy the
    /// Tip moves to `spinner_tip` (under the spinner) and this slot is gone, so
    /// the sole below-composer row is `info`.
    pub tips: Option<Rect>,
}

/// Compute the §5 vertical split. PURE over `(app, area)` (no state writes) so it
/// can run BEFORE the draw (in `prepare_frame`) and again during the draw with an
/// identical result.
pub(crate) fn split_cockpit(app: &AppState, area: Rect) -> CockpitLayout {
    // The composer flexes from 3 rows up to COMPOSER_MAX_ROWS as the buffer grows.
    let composer_inner_w = area.width.saturating_sub(4).max(1); // borders + prompt
    let composer_rows = composer_height(app, composer_inner_w);

    let show_spinner = app.busy;
    // The frozen done-line takes the just-above-composer slot when the cockpit is
    // idle AND a turn has run (Q7). Busy XOR done — they never both show.
    let show_done = !show_spinner && app.last_turn_ms.is_some();
    let dropdown_rows = dropdown_height(app, area.width);

    // §5 vertical split. Transcript = Min(0) so it flexes to fill. The `⎿ Tip`
    // travels with the spinner state (Slice 5): while BUSY it sits as a
    // corner-continuation row right under the spinner status line (above the
    // composer), and the sole below-composer row is row1 session info. While IDLE
    // it returns to its row2 below-composer slot. Row1 always carries the conn chip.
    let mut constraints: Vec<Constraint> = vec![
        Constraint::Length(HEADER_ROWS), // header (multi-row rounded box, Slice 2)
        Constraint::Length(1),           // rainbow separator
        Constraint::Min(0),              // transcript (FLEX, shrinks for the taller header)
    ];
    if show_spinner {
        constraints.push(Constraint::Length(1)); // spinner status line
        constraints.push(Constraint::Length(1)); // ⎿ tip (under the spinner)
    } else if show_done {
        constraints.push(Constraint::Length(1)); // frozen done-line
    }
    if dropdown_rows > 0 {
        constraints.push(Constraint::Length(dropdown_rows)); // palette / file picker
    }
    constraints.push(Constraint::Length(composer_rows)); // composer (bordered)
    constraints.push(Constraint::Length(1)); // row1: runtime session info
    if !show_spinner {
        constraints.push(Constraint::Length(1)); // row2: ⎿ Tips (idle only)
    }

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    // Index the chunks in order.
    let mut i = 0;
    let mut next = || {
        let c = chunks[i];
        i += 1;
        c
    };
    let header = next();
    let sep = next();
    let transcript = next();
    let spinner = show_spinner.then(&mut next);
    let spinner_tip = show_spinner.then(&mut next);
    let done = show_done.then(&mut next);
    let dropdown = (dropdown_rows > 0).then(&mut next);
    let composer = next();
    let info = next();
    let tips = (!show_spinner).then(&mut next);

    CockpitLayout {
        header,
        sep,
        transcript,
        spinner,
        spinner_tip,
        done,
        dropdown,
        composer,
        info,
        tips,
    }
}

/// Draw the chat cockpit (header / transcript / composer / 2 below-composer rows)
/// — the normal view. Split out from [`render`] so the dashboard view can
/// short-circuit. Takes `&AppState` (immutable — P11): the transcript wrap cache is
/// already synced by [`AppState::prepare_frame`] (called from the loop before the
/// draw).
fn render_cockpit(frame: &mut Frame, app: &AppState, theme: &Theme, now_ms: u64) {
    let area = frame.area();
    let CockpitLayout {
        header,
        sep,
        transcript,
        spinner,
        spinner_tip,
        done,
        dropdown,
        composer,
        info,
        tips,
    } = split_cockpit(app, area);

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
    // The `⎿ Tip` hangs as a corner-continuation directly under the spinner status
    // line while busy (Slice 5) — drawn here, right after the spinner.
    if let Some(tip) = spinner_tip {
        render_tips(frame, tip, app, theme, now_ms);
    }
    if let Some(dl) = done {
        render_done_line(frame, dl, app, theme);
    }
    if let Some(dd) = dropdown {
        render_dropdown(frame, dd, app, theme);
    }
    render_composer(frame, composer, app, theme);
    // Per-command composer border FX — GATED (Q11b): the input box is PLAIN by
    // default; the effects only light up when the composer holds one of the
    // orchestration commands, and EACH (/goal /hive /conductor /morphling) gets its
    // own border identity + corner glyph. Shell mode keeps its hot-pink border (the
    // fx fn early-returns there).
    if let Some(cmd) = super::text::fx_command(app.composer.text()) {
        super::effects_paint::draw_composer_border_fx(frame, app, composer, now_ms, cmd);
    }
    render_session_info(frame, info, app, theme);
    // The below-composer `⎿ Tip` row exists only when IDLE; while busy the Tip has
    // moved up under the spinner (`spinner_tip`) and `info` is the sole row here.
    if let Some(tip) = tips {
        render_tips(frame, tip, app, theme, now_ms);
    }
}
