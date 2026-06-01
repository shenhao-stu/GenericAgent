//! app_event.rs — the UI→app intent bus (ARCH Fix A; Codex `app_event.rs` peer).
//!
//! The missing indirection that severs F1+F6: instead of every key/command
//! handler closing over the bridge `Sender` and performing effects inline, it
//! `app.emit(AppEvent::…)`s an intent. The event loop drains the queue AFTER
//! `handle_term_event` and is the ONE place the transport (`tx_bridge`) lives.
//! Keeping intents in a data enum is what lets later slices move the handlers
//! out of `main.rs` without dragging the sender along.

/// An intent the UI plane emits for the app layer to perform. The only bridge
/// verb the UI speaks is "send this wire frame" (to the active or a named
/// session); the rest are view/clipboard/lifecycle actions handled in the loop.
//
// `allow(dead_code)`: every variant is matched by `perform_actions`, but the
// handlers that CONSTRUCT the view/clipboard/lifecycle variants
// (Open*/CloseView/Copy/SetMouseCapture/Quit) land in later sub-steps (0c+).
// 0b wires the `ToActive` emitters; the rest stay un-constructed until then, so
// the allow remains until the bus is fully populated. Scoped to the enum only.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum AppEvent {
    /// Send a wire frame to the ACTIVE session (the common bridge verb).
    ToActive(crate::bridge::protocol::UiToCore),
    /// Send a wire frame to a specific session (dashboard quick-reply / bg work).
    ToSession(u64, crate::bridge::protocol::UiToCore),
    /// Open the full-screen `/workflows` panel.
    OpenWorkflows,
    /// Open the full-screen session dashboard.
    OpenDashboard,
    /// Return from a full-screen view to the cockpit.
    CloseView,
    /// Copy `text` to the clipboard, surfacing a notice labelled `label`.
    Copy { text: String, label: &'static str },
    /// Toggle terminal mouse capture (reconciles `app.mouse_capture` + the escape).
    SetMouseCapture(bool),
    /// Quit the app (the event loop breaks).
    Quit,
}
