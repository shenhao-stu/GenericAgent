//! app/multi.rs — the multi-session glue (§6 / N2) + the full-screen view
//! switches (dashboard / `/workflows`).
//!
//! The ACTIVE session's live state lives in the `AppState` fields (transcript /
//! conn / model / busy / pending_ask / …) — what the cockpit renders and what the
//! reducer + tests exercise. The [`SessionMap`](crate::app::session::SessionMap)
//! is the source of truth for the OTHER sessions. Two operations keep them
//! coherent: `snapshot_active_into_map` (push the live active state into its
//! record) and the swap inside `switch_session` (pull a target's stored state
//! into the live fields) — localizing the bookkeeping to a few methods instead
//! of threading a session id through the whole render plane.

use crate::app::{AppState, Role, View};
use crate::bridge::BridgeEvent;
use crate::render::{Viewport, WrapCache};

impl AppState {
    /// Mirror the live ACTIVE-session fields into its [`Session`] record so the
    /// dashboard reads an up-to-date snapshot. Cheap clone of the transcript (the
    /// call sites are rare — a frame arrival or a view switch, not per frame).
    pub fn snapshot_active_into_map(&mut self) {
        let active_id = self.sessions.active;
        let transcript = self.transcript.clone();
        let conn = self.conn.clone();
        let model = self.model.clone();
        let context_percent = self.context_percent;
        let busy = self.busy;
        let busy_since = self.turn_started_ms;
        let pending = self.pending_ask.clone();
        let had_reply = self
            .transcript
            .iter()
            .any(|b| b.role == Role::Assistant && !b.source.is_empty());
        if let Some(s) = self.sessions.session_mut(active_id) {
            s.transcript = transcript;
            s.conn = conn;
            s.model = model;
            s.context_percent = context_percent;
            s.busy = busy;
            s.busy_since_ms = busy_since;
            s.pending_ask = pending;
            s.had_reply = s.had_reply || had_reply;
        }
    }

    /// Load a [`Session`]'s stored state into the live ACTIVE-session fields and
    /// reset the render plane (wrap cache + viewport) so the cockpit re-derives
    /// from the incoming transcript at the next frame (P1). The composer buffer is
    /// the caller's concern.
    fn load_active_fields_from(&mut self, id: u64) {
        let Some(s) = self.sessions.session(id) else {
            return;
        };
        self.transcript = s.transcript.clone();
        self.conn = s.conn.clone();
        self.model = s.model.clone();
        self.context_percent = s.context_percent;
        self.busy = s.busy;
        self.turn_started_ms = s.busy_since_ms;
        self.pending_ask = s.pending_ask.clone();
        // Re-id the live next_block_id past the loaded transcript so new live
        // appends don't collide with the session's own ids.
        self.next_block_id = self
            .transcript
            .iter()
            .map(|b| b.id)
            .max()
            .map(|m| m.wrapping_add(1))
            .unwrap_or(1);
        // Reset the render plane so the next `sync_transcript` re-derives cleanly.
        self.wrap_cache = WrapCache::new(self.last_width.max(1));
        self.viewport = Viewport::new(1);
        self.last_width = 0; // force a full rewidth on the next frame.
        self.last_fold_all = self.fold_all;
        // Per-node fold overrides are keyed by block id; the incoming session re-mints
        // its ids, so the old overrides are meaningless here — clear them for a clean
        // per-session fold slate (Fix E). The global `fold_all` is preserved above.
        self.folds.clear();
        self.node_hit.clear();
        self.fold_epoch = self.fold_epoch.wrapping_add(1);
        self.last_fold_epoch = self.fold_epoch;
    }

    /// Switch the active session to `id`: snapshot the current active state into
    /// its record, stash the composer draft on it, load the target's state +
    /// restore its stashed draft, and return to the cockpit view.
    pub fn switch_session(&mut self, id: u64) {
        if id == self.sessions.active && self.view == View::Cockpit {
            return;
        }
        self.snapshot_active_into_map();
        let current_draft = self.composer.text().to_string();
        let incoming_draft = self.sessions.switch(id, current_draft);
        self.load_active_fields_from(self.sessions.active);
        self.composer.set_buffer(incoming_draft.clone(), incoming_draft.len());
        self.view = View::Cockpit;
    }

    /// Public wrapper for the cockpit's session-cycle path (Ctrl+Up/Down): the
    /// map's active already moved; pull its stored state into the live fields.
    pub fn load_active_fields_from_public(&mut self, id: u64) {
        self.load_active_fields_from(id);
        self.view = View::Cockpit;
    }

    /// Load the fallback active session after a drop (Ctrl+W/Ctrl+D): pull its
    /// stored state in and restore its stashed draft.
    pub fn load_active_fields_after_drop(&mut self, id: u64) {
        self.load_active_fields_from(id);
        let draft = self
            .sessions
            .session_mut(id)
            .map(|s| std::mem::take(&mut s.input_stash))
            .unwrap_or_default();
        self.composer.set_buffer(draft.clone(), draft.len());
        self.view = View::Cockpit;
    }

    /// After a structural change that already moved `self.sessions.active` (a
    /// `new_session` or `branch`), load the NEW active session's state into the
    /// live fields and reset the composer to its draft. The LEAVING session's live
    /// state must already be snapshotted (the caller does that first).
    pub fn load_active_session_after_structural_change(&mut self, new_active: u64) {
        debug_assert_eq!(self.sessions.active, new_active);
        self.load_active_fields_from(new_active);
        let draft = self
            .sessions
            .session_mut(new_active)
            .map(|s| std::mem::take(&mut s.input_stash))
            .unwrap_or_default();
        self.composer.set_buffer(draft.clone(), draft.len());
        self.view = View::Cockpit;
    }

    /// Open the full-screen session dashboard (Ctrl+S / left-click sessions area).
    /// Snapshots the active session, then seeds the selection on its row.
    pub fn open_dashboard(&mut self) {
        self.snapshot_active_into_map();
        self.view = View::Dashboard;
        self.rename = None;
        let rows = self.sessions.dashboard_rows();
        let active = self.sessions.active;
        if let Some(idx) = rows.iter().position(|r| {
            matches!(r, crate::app::session::DashRow::Session { id, .. } if *id == active)
        }) {
            self.sessions.dash_sel = idx;
        }
    }

    /// Close the dashboard, returning to the cockpit (Esc).
    pub fn close_dashboard(&mut self) {
        self.view = View::Cockpit;
        self.rename = None;
    }

    /// Open the full-screen `/workflows` panel. Lazily STARTS the singleton
    /// watcher on first open (its own poll thread), then ACTIVATES it (it parks
    /// on close → zero idle traffic). The panel renders the latest snapshot.
    pub fn open_workflows(&mut self) {
        if self.workflow_watcher.is_none() {
            // The change receiver is dropped: the event loop already redraws on its
            // 100ms tick + bridge events and the panel reads `snapshot()` each frame.
            let (watcher, _change_rx) = crate::workflow::WorkflowWatcher::start(self.repo_root.clone());
            self.workflow_watcher = Some(watcher);
        }
        if let Some(w) = &self.workflow_watcher {
            w.set_active(true);
        }
        self.view = View::Workflows;
        self.refresh_workflow_snapshot();
    }

    /// Close the `/workflows` panel, returning to the cockpit (Esc). PARKS the
    /// watcher so it stops generating background traffic while nobody is looking.
    pub fn close_workflows(&mut self) {
        if let Some(w) = &self.workflow_watcher {
            w.set_active(false);
        }
        self.view = View::Cockpit;
    }

    /// Refresh the panel's snapshot from the watcher (each frame while open).
    /// Re-clamps the panel focus when the generation advanced. Never blocks (the
    /// watcher does the I/O off-thread).
    pub fn refresh_workflow_snapshot(&mut self) {
        if let Some(w) = &self.workflow_watcher {
            self.workflow_snapshot = w.snapshot();
            self.workflow_panel.clamp_focus(&self.workflow_snapshot);
        }
    }

    /// Apply a TAGGED bridge event `(session_id, ev)` from the multiplexer. The
    /// ACTIVE session folds through the live reducer (cockpit path) + mirrors back
    /// into its record; a background session folds into its own record only (so
    /// its dashboard preview updates without disturbing the active session). A
    /// frame for an unknown (already-dropped) session is discarded.
    pub fn apply_tagged_event(&mut self, session_id: u64, ev: BridgeEvent, now_ms: u64) {
        if session_id == self.sessions.active {
            self.apply_bridge_event(ev, now_ms);
            self.snapshot_active_into_map();
        } else if let Some(s) = self.sessions.session_mut(session_id) {
            match ev {
                BridgeEvent::Frame(frame) => s.apply_frame(frame, now_ms),
                other => s.apply_lifecycle(&other),
            }
        }
    }
}
