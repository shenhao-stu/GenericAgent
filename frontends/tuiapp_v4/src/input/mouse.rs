//! input/mouse.rs — pointer routing (ARCH Fix B). The wheel scrolls the transcript;
//! a left-click on the header band opens the dashboard; a right-click is the
//! universal "go back" (dashboard → cockpit). A click while a modal overlay is up
//! dismisses it (click-outside). Split out of `main.rs`.

use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
use ratatui::layout::Rect;

use crate::app::{AppState, View};
use crate::components;

/// How many transcript rows one mouse-wheel notch scrolls (recon §6.7).
const WHEEL_STEP: isize = 3;

/// Mouse: in the cockpit the wheel scrolls the transcript and a left-click on the
/// header/footer "sessions area" opens the dashboard (§6 "left-click on the
/// sessions area"); in the dashboard a left-click on a row switches into it.
pub(crate) fn mouse(me: MouseEvent, app: &mut AppState) {
    // A click anywhere while a MODAL overlay is up dismisses it (a click-outside
    // affordance); the overlay's own keys handle precise selection. A NON-modal
    // overlay (the `/btw` toast) lets the wheel keep scrolling the cockpit beneath
    // (chat stays usable) — only a left-click dismisses it.
    if let Some(modal) = app.overlay.as_ref().map(|o| o.is_modal()) {
        let is_left = matches!(me.kind, MouseEventKind::Down(MouseButton::Left));
        if modal {
            if is_left {
                app.close_overlay();
            }
            return;
        }
        // Non-modal: a left-click dismisses; otherwise fall through to cockpit
        // scroll handling below.
        if is_left {
            app.close_overlay();
            return;
        }
    }
    match (app.view, me.kind) {
        // -- cockpit --
        (View::Cockpit, MouseEventKind::ScrollUp) => app.scroll_lines(-WHEEL_STEP),
        (View::Cockpit, MouseEventKind::ScrollDown) => app.scroll_lines(WHEEL_STEP),
        (View::Cockpit, MouseEventKind::Down(MouseButton::Left)) => {
            // The header + rainbow separator (rows 0–1) are the "sessions area" — a
            // left-click there opens the full-screen dashboard (the §4/§6 left-click
            // entry point). Broadened from row 0 to the whole header band so the
            // click target isn't a 1-row sliver (redesign_cc.md §4 "broaden … a bit").
            if me.row <= 1 {
                app.open_dashboard();
                return;
            }
            // A click on a fold node's triangle/bullet column (the first cells of the
            // transcript) toggles that node's fold (Fix E / Q8): a `▸` turn header
            // expands, a tool bullet expands/collapses its result. The transcript
            // region top is derived from the SAME layout `prepare_frame`/render use,
            // so the row→node map matches what's drawn. A click elsewhere in the body
            // falls through (native selection owns it when capture is off).
            let area = Rect {
                x: 0,
                y: 0,
                width: app.last_term_width(),
                height: app.last_term_height(),
            };
            let transcript_top = components::cockpit::split_cockpit(app, area).transcript.y;
            app.click_fold_at(me.column, me.row, transcript_top);
        }
        // Right-click in the cockpit → no-op (redesign_cc.md §4: right-click is the
        // universal "go back", and the cockpit is already the root view). Matched
        // explicitly so it's a deliberate no-op, not an accidental fall-through.
        (View::Cockpit, MouseEventKind::Down(MouseButton::Right)) => { /* no-op (already at root) */ }
        // -- dashboard: right-click anywhere → go BACK to the cockpit (§4). This is
        // the missing `MouseButton::Right` handler — the mouse mirror of `Esc`.
        (View::Dashboard, MouseEventKind::Down(MouseButton::Right)) => app.close_dashboard(),
        // -- dashboard: left-click a session row → switch into it --
        (View::Dashboard, MouseEventKind::Down(MouseButton::Left)) => {
            let area = Rect {
                x: 0,
                y: 0,
                width: app.last_term_width(),
                height: app.last_term_height(),
            };
            let rows = app.sessions.dashboard_rows();
            if let Some(idx) =
                components::dashboard::click_to_row_index(me.column, me.row, area, rows.len(), app.sessions.dash_sel)
            {
                app.sessions.dash_sel = idx;
                if let Some(id) = app.sessions.selected_session_id() {
                    app.switch_session(id);
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE mouse deliverable (redesign_cc.md §4): a RIGHT-click in the Dashboard
    /// returns to the cockpit (the mouse mirror of `Esc`), a right-click in the
    /// cockpit is a no-op (already at the root view), and a LEFT-click on the header
    /// band still opens the dashboard. Drives the real `mouse` path so the
    /// `MouseButton::Right` arm is exercised, not just asserted in prose.
    #[test]
    fn right_click_returns_from_dashboard() {
        use crossterm::event::{KeyModifiers, MouseButton, MouseEvent, MouseEventKind};

        let down = |kind: MouseEventKind, col: u16, row: u16| MouseEvent {
            kind,
            column: col,
            row,
            modifiers: KeyModifiers::NONE,
        };
        let right = |col, row| down(MouseEventKind::Down(MouseButton::Right), col, row);
        let left = |col, row| down(MouseEventKind::Down(MouseButton::Left), col, row);

        let mut app = AppState::new();
        // Give the dashboard a sane click-geometry (set on render in the live app).
        app.set_term_size(100, 30);
        assert_eq!(app.view, View::Cockpit, "starts in the cockpit");

        // A LEFT-click on the header band (row 0) opens the dashboard (§4/§6).
        mouse(left(10, 0), &mut app);
        assert_eq!(app.view, View::Dashboard, "left-click on the header opens the dashboard");

        // A RIGHT-click anywhere in the dashboard goes BACK to the cockpit.
        mouse(right(42, 7), &mut app);
        assert_eq!(app.view, View::Cockpit, "right-click in the dashboard returns to the cockpit");

        // A RIGHT-click in the cockpit is a NO-OP (stays at the root view).
        mouse(right(42, 7), &mut app);
        assert_eq!(app.view, View::Cockpit, "right-click in the cockpit is a no-op");

        // The broadened open-zone: a left-click on the separator row (row 1) also
        // opens the dashboard (not just the 1-row header sliver).
        mouse(left(3, 1), &mut app);
        assert_eq!(app.view, View::Dashboard, "left-click on the separator row also opens it");

        // And right-click closes it again — round-trips cleanly.
        mouse(right(0, 0), &mut app);
        assert_eq!(app.view, View::Cockpit, "right-click round-trips back to the cockpit");
    }
}
