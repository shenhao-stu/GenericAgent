//! input/keychord.rs — the two MULTI-PRESS key state machines (checklist §8;
//! tui_v3 parity): the **3-stage Ctrl+C** and the **Esc-Esc → /rewind** chord.
//!
//! Both are written as PURE functions: the only "clock" is a `now_ms` passed in by
//! the caller (the event loop's monotonic `start.elapsed()`), never a wall-clock
//! read here. The arm/pending timestamp also lives in `AppState` and is threaded
//! in + returned out. That is what makes the transitions unit-testable headlessly
//! ([`ctrl_c_three_stage_transitions`], [`esc_esc_within_window_triggers_rewind`],
//! [`esc_esc_outside_window_is_two_backs`]) — no sleeps, no flaky timing.
//!
//! ## (A) 3-stage Ctrl+C  (`ctrl_c`)
//! One Ctrl+C does the FIRST applicable of:
//!   1. a selection exists → **Copy** it (OSC-52 via `render/copy.rs`);
//!   2. else a turn is running → **Abort** it;
//!   3. else **Arm** quit + show a hint ("press Ctrl+C again to quit"); a SECOND
//!      Ctrl+C within [`CTRL_C_ARM_MS`] → **Quit**. The arm EXPIRES after the same
//!      window, so a stale arm never silently quits a later lone press.
//!
//! ## (B) Esc-Esc  (`esc`)
//! A SECOND Esc within [`ESC_ESC_MS`] of the first → **Rewind** (open the rewind
//! picker). A single Esc (or one outside the window) → **Back** (the existing
//! universal-back: clear pending ask → collapse selection → stash draft, never
//! exits). Each Esc re-arms the window from `now_ms` so a slow double still counts
//! while two deliberate, spaced Escs are two independent backs.

/// The arm/repeat window for the 3-stage Ctrl+C (ms). A 2nd Ctrl+C within this of
/// the arming press quits; the arm expires after it. (§8 "within 2s".)
pub const CTRL_C_ARM_MS: u64 = 2_000;

/// The Esc-Esc window (ms): a 2nd Esc within this of the 1st triggers `/rewind`.
/// (§8 "Esc-Esc (<0.8 s) → /rewind".)
pub const ESC_ESC_MS: u64 = 800;

/// What a Ctrl+C press resolves to this time (the caller performs the effect).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CtrlCAction {
    /// A selection exists → copy it to the clipboard (OSC-52). Disarms.
    CopySelection,
    /// A turn is running → abort it. Disarms.
    AbortTurn,
    /// Nothing to copy/abort → ARM quit + show the "press Ctrl+C again" hint.
    ArmQuit,
    /// A 2nd Ctrl+C arrived inside the arm window → quit.
    Quit,
}

/// What an Esc press resolves to this time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscAction {
    /// First Esc (or one outside the double-tap window) → universal back.
    Back,
    /// 2nd Esc inside the window → open the /rewind picker.
    Rewind,
}

/// The transient state both chords need, carried in `AppState`. Holds the
/// "armed-until"/"last-press" monotonic-ms stamps so the deciders stay pure (no
/// wall-clock inside). `Default` = neither chord is mid-sequence.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ChordState {
    /// While `Some(t)`, a Ctrl+C-quit is ARMED until monotonic-ms `t`; a 2nd
    /// Ctrl+C at `now <= t` quits. Cleared on quit-arm-expiry / copy / abort.
    pub ctrl_c_armed_until: Option<u64>,
    /// The monotonic-ms of the last Esc that resolved to `Back`, for double-tap
    /// detection. `Some(t)` means "one Esc landed at `t`; a 2nd by `t+window`
    /// is Esc-Esc". Cleared after a rewind fires (so a 3rd Esc starts fresh).
    pub last_esc_ms: Option<u64>,
}

impl ChordState {
    /// True while the Ctrl+C-quit hint should be shown to the user (i.e. quit is
    /// armed and not yet expired at `now_ms`). The renderer / a notice can use this
    /// to surface "press Ctrl+C again to quit". PURE.
    pub fn ctrl_c_hint_active(&self, now_ms: u64) -> bool {
        matches!(self.ctrl_c_armed_until, Some(t) if now_ms <= t)
    }
}

/// Decide what a single Ctrl+C press does, given the current state + `now_ms`.
/// Returns the [`CtrlCAction`] AND the NEXT [`ChordState`] (the caller stores it).
/// PURE — the whole 3-stage machine, no I/O, `now_ms` injected.
///
/// Precedence (the FIRST that applies wins, matching tui_v3 / Claude Code):
///   1. `has_selection` → [`CtrlCAction::CopySelection`] (disarm).
///   2. `turn_running`  → [`CtrlCAction::AbortTurn`] (disarm).
///   3. armed AND `now_ms <= armed_until` → [`CtrlCAction::Quit`].
///   4. otherwise → [`CtrlCAction::ArmQuit`] (re-arm to `now_ms + CTRL_C_ARM_MS`).
///
/// A copy or an abort ALWAYS disarms (a fresh Ctrl+C with nothing to do then
/// starts the arm from scratch — you never "carry" an arm across a copy/abort).
/// An EXPIRED arm (`now_ms > armed_until`) re-arms rather than quitting, so a
/// stale arm from minutes ago can't make a lone press silently exit the app.
pub fn ctrl_c(state: ChordState, now_ms: u64, has_selection: bool, turn_running: bool) -> (CtrlCAction, ChordState) {
    // 1. Copy a selection — highest precedence; disarms.
    if has_selection {
        return (
            CtrlCAction::CopySelection,
            ChordState { ctrl_c_armed_until: None, ..state },
        );
    }
    // 2. Abort a running turn; disarms.
    if turn_running {
        return (
            CtrlCAction::AbortTurn,
            ChordState { ctrl_c_armed_until: None, ..state },
        );
    }
    // 3. A live (un-expired) arm → quit.
    if let Some(until) = state.ctrl_c_armed_until {
        if now_ms <= until {
            return (CtrlCAction::Quit, state);
        }
    }
    // 4. Nothing to do + no live arm → arm quit (show the hint) for the window.
    (
        CtrlCAction::ArmQuit,
        ChordState { ctrl_c_armed_until: Some(now_ms.saturating_add(CTRL_C_ARM_MS)), ..state },
    )
}

/// Decide what a single Esc press does, given the current state + `now_ms`.
/// Returns the [`EscAction`] AND the NEXT [`ChordState`]. PURE.
///
///   * If an earlier Esc landed at `t` AND `now_ms <= t + ESC_ESC_MS` → this is the
///     2nd tap → [`EscAction::Rewind`]; the window is CLEARED so a 3rd Esc starts a
///     fresh sequence (Esc-Esc-Esc is not Esc-Esc + Esc-Esc).
///   * Otherwise → [`EscAction::Back`]; record `now_ms` as the new last-Esc so a
///     quick follow-up Esc is detected as the double.
///
/// The window is measured from the FIRST tap (`t`), not refreshed by the 2nd, so a
/// double-tap is genuinely "two within `ESC_ESC_MS`". A lone Esc, or two Escs
/// spaced beyond the window, are each an independent `Back` (never a rewind).
pub fn esc(state: ChordState, now_ms: u64) -> (EscAction, ChordState) {
    if let Some(first) = state.last_esc_ms {
        if now_ms <= first.saturating_add(ESC_ESC_MS) {
            // 2nd tap inside the window → rewind; clear so a 3rd starts fresh.
            return (EscAction::Rewind, ChordState { last_esc_ms: None, ..state });
        }
    }
    // First tap (or one past the window) → back; arm the window from now.
    (EscAction::Back, ChordState { last_esc_ms: Some(now_ms), ..state })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE deliverable test (A): the 3-stage Ctrl+C transitions — copy → abort →
    /// arm-quit → quit, plus arm expiry + disarm-on-copy/abort. All pure: every
    /// "time" is an injected `now_ms`, so there are no sleeps and no flake.
    #[test]
    fn ctrl_c_three_stage_transitions() {
        let s0 = ChordState::default();

        // Stage 1 — a selection exists → COPY (regardless of busy), and it disarms.
        let (act, s) = ctrl_c(s0, 100, /*has_selection=*/ true, /*turn_running=*/ false);
        assert_eq!(act, CtrlCAction::CopySelection);
        assert_eq!(s.ctrl_c_armed_until, None, "copy disarms");
        // Even with a turn running, a selection still wins (copy precedence).
        let (act, _) = ctrl_c(s0, 100, true, true);
        assert_eq!(act, CtrlCAction::CopySelection, "selection beats abort");

        // Stage 2 — no selection but a turn is running → ABORT (disarms).
        let (act, s) = ctrl_c(s0, 200, false, true);
        assert_eq!(act, CtrlCAction::AbortTurn);
        assert_eq!(s.ctrl_c_armed_until, None);

        // Stage 3 — nothing to copy/abort → ARM quit + show the hint window.
        let (act, armed) = ctrl_c(s0, 1_000, false, false);
        assert_eq!(act, CtrlCAction::ArmQuit);
        assert_eq!(armed.ctrl_c_armed_until, Some(1_000 + CTRL_C_ARM_MS));
        assert!(armed.ctrl_c_hint_active(1_000), "the hint is shown while armed");
        assert!(armed.ctrl_c_hint_active(1_000 + CTRL_C_ARM_MS), "…right up to the boundary");
        assert!(!armed.ctrl_c_hint_active(1_000 + CTRL_C_ARM_MS + 1), "…and not after it expires");

        // A 2nd Ctrl+C WITHIN the window → QUIT (state unchanged; caller exits).
        let (act, _) = ctrl_c(armed, 1_000 + CTRL_C_ARM_MS, false, false);
        assert_eq!(act, CtrlCAction::Quit);
        // Exactly at the boundary still quits; one ms past it re-arms (no silent quit).
        let (act, rearmed) = ctrl_c(armed, 1_000 + CTRL_C_ARM_MS + 1, false, false);
        assert_eq!(act, CtrlCAction::ArmQuit, "an EXPIRED arm re-arms, never silently quits");
        assert_eq!(rearmed.ctrl_c_armed_until, Some(1_000 + CTRL_C_ARM_MS + 1 + CTRL_C_ARM_MS));

        // A copy/abort while ARMED disarms (you don't carry an arm across them): a
        // following lone Ctrl+C must ARM afresh, not quit.
        let (act, after_copy) = ctrl_c(armed, 1_500, true, false);
        assert_eq!(act, CtrlCAction::CopySelection);
        assert_eq!(after_copy.ctrl_c_armed_until, None);
        let (act, _) = ctrl_c(after_copy, 1_600, false, false);
        assert_eq!(act, CtrlCAction::ArmQuit, "post-copy lone Ctrl+C arms, never quits");
    }

    /// THE deliverable test (B1): two Escs INSIDE the window trigger /rewind; the
    /// first is a Back, the second a Rewind, and the window is then cleared.
    #[test]
    fn esc_esc_within_window_triggers_rewind() {
        let s0 = ChordState::default();

        // First Esc → Back, arms the double-tap window at t=500.
        let (act, s1) = esc(s0, 500);
        assert_eq!(act, EscAction::Back);
        assert_eq!(s1.last_esc_ms, Some(500));

        // Second Esc within 800ms → Rewind; the window is cleared afterwards.
        let (act, s2) = esc(s1, 500 + ESC_ESC_MS); // exactly at the boundary still counts
        assert_eq!(act, EscAction::Rewind);
        assert_eq!(s2.last_esc_ms, None, "a fired rewind clears the window");

        // A well-inside double also fires.
        let (_, s1) = esc(s0, 1_000);
        let (act, _) = esc(s1, 1_300);
        assert_eq!(act, EscAction::Rewind);

        // Esc-Esc-Esc is NOT Esc-Esc + a dangling rewind: the 3rd Esc (after the
        // clear) is a fresh Back, and a 4th within its window rewinds again.
        let (_, s1) = esc(s0, 0); // Back @0
        let (_, s2) = esc(s1, 100); // Rewind @100 (clears)
        let (act, s3) = esc(s2, 200); // 3rd → Back (fresh sequence)
        assert_eq!(act, EscAction::Back);
        assert_eq!(s3.last_esc_ms, Some(200));
        let (act, _) = esc(s3, 300); // 4th within window → Rewind
        assert_eq!(act, EscAction::Rewind);
    }

    /// THE deliverable test (B2): two Escs OUTSIDE the window are two independent
    /// universal-backs (never a rewind), and a lone Esc is always a Back.
    #[test]
    fn esc_esc_outside_window_is_two_backs() {
        let s0 = ChordState::default();

        // A single Esc → Back.
        let (act, s1) = esc(s0, 0);
        assert_eq!(act, EscAction::Back);

        // A second Esc JUST past the window (801ms later) → Back again, NOT rewind.
        let (act, s2) = esc(s1, ESC_ESC_MS + 1);
        assert_eq!(act, EscAction::Back, "past the window it is a plain back, not a rewind");
        assert_eq!(s2.last_esc_ms, Some(ESC_ESC_MS + 1), "…and it re-arms from the new press");

        // Two slow, deliberate Escs (seconds apart) are each a back.
        let (a1, s1) = esc(s0, 10_000);
        let (a2, _s2) = esc(s1, 20_000);
        assert_eq!(a1, EscAction::Back);
        assert_eq!(a2, EscAction::Back);
    }
}
