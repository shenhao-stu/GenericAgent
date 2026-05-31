//! render/copy.rs — clean clipboard copy of the logical SOURCE (checklist §3,
//! P2 — "Copy yields clean logical text"). The structural guarantee: we copy the
//! block's `source` string, NEVER the rendered/soft-wrapped rows, so a terminal
//! soft-wrap can never become an embedded `\n` in the paste buffer.
//!
//! Delivery is a fallback CHAIN, SSH-safe first:
//!   1. **OSC 52** — `ESC ] 52 ; c ; base64(utf8 source) BEL`, written to the
//!      controlling TTY. Because it transports the exact logical bytes there is
//!      no grid, no columns, no wrap — immune to the P2 bug by construction. It
//!      crosses SSH (the *local* terminal performs the clipboard write) where a
//!      native tool on the server cannot. tmux/screen passthrough is handled so
//!      it survives a multiplexer.
//!   2. **native `arboard`** — only when LOCAL (no `SSH_*`): sets the OS
//!      clipboard in-process. Also byte-exact (we hand it the logical string).
//!   3. **copy-mode overlay** (a stub here, [`CopyOverlay`]) — the universal
//!      escape hatch: print the raw logical text with NO injected breaks and let
//!      the terminal soft-wrap it, so OS drag-select comes out clean even with no
//!      OSC 52 and no native lib.
//!
//! The pure, load-bearing surface (`build_osc52`, `join_visual_rows`,
//! `copy_payload`) is unit-tested headlessly; the actual TTY/clipboard writes are
//! thin effectful wrappers around it.

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;

use crate::render::measure::VisualLine;

const ESC: char = '\x1b';
const BEL: char = '\x07';

/// Which OS selection an OSC 52 write targets. `Primary`/`Both` complete the
/// xterm selection model (X11 PRIMARY / middle-click); the cockpit copies to
/// `Clipboard`, the others are available for a `/export primary`-style action.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Selection {
    /// The OS clipboard (Ctrl/Cmd-V) — the default and what we almost always want.
    Clipboard,
    /// X11 PRIMARY (middle-click).
    Primary,
    /// Both clipboard + primary.
    Both,
}

impl Selection {
    fn token(self) -> &'static str {
        match self {
            Selection::Clipboard => "c",
            Selection::Primary => "p",
            Selection::Both => "cp",
        }
    }
}

/// Best-effort environment capabilities (OSC 52 success is unverifiable, so these
/// only steer the *order* of attempts). Injectable so the chain is unit-testable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CopyCaps {
    /// We are inside an SSH session (`SSH_TTY`/`SSH_CONNECTION`/`SSH_CLIENT`).
    /// When true, the native clipboard would set the *server's* clipboard (wrong
    /// machine) — so native is skipped and OSC 52 is the only path that reaches
    /// the user's local clipboard.
    pub is_remote: bool,
    /// We are inside tmux (`$TMUX`).
    pub inside_tmux: bool,
    /// We are inside GNU screen (`$TERM` starts with `screen`, not tmux).
    pub inside_screen: bool,
}

impl CopyCaps {
    /// Capture the real environment.
    pub fn from_env() -> Self {
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        let is_remote =
            env("SSH_TTY").is_some() || env("SSH_CONNECTION").is_some() || env("SSH_CLIENT").is_some();
        let inside_tmux = env("TMUX").is_some();
        let term = std::env::var("TERM").unwrap_or_default();
        let inside_screen = !inside_tmux && term.starts_with("screen");
        CopyCaps {
            is_remote,
            inside_tmux,
            inside_screen,
        }
    }
}

/// The outcome of a copy attempt, for the UI's transient status flash
/// ("Copied 1.2 KB to clipboard (osc52)").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CopyResult {
    pub ok: bool,
    /// Which method succeeded (or `None`).
    pub method: CopyMethod,
    /// Raw byte length of the logical text attempted.
    pub bytes: usize,
    /// Why it failed / fell through, if applicable.
    pub reason: Option<CopyReason>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyMethod {
    Osc52,
    Native,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyReason {
    /// Larger than the OSC 52 byte cap; the caller should open the copy-mode
    /// overlay (we NEVER silently truncate).
    TooLarge,
    /// Native clipboard refused because we are remote (would set wrong machine).
    NoNativeRemote,
    /// Native clipboard errored (no backend / headless).
    NativeError,
    /// No TTY to write the OSC 52 escape to.
    NoTty,
}

/// Default OSC 52 raw-byte cap. Over this, we refuse (return [`CopyReason::TooLarge`])
/// rather than truncate, because terminals/tmux silently drop over-long sequences
/// and a half-copied code block is a nasty footgun.
pub const DEFAULT_MAX_OSC52_BYTES: usize = 64 * 1024;

/// Build the raw OSC 52 escape for `text` (the LOGICAL source string). PURE +
/// deterministic — base64 is over the UTF-8 bytes of `text`, never over wrapped
/// rows, which is the structural reason this is wrap-immune (P2). When inside a
/// multiplexer the sequence is wrapped for passthrough.
pub fn build_osc52(text: &str, selection: Selection, caps: CopyCaps) -> String {
    let b64 = BASE64.encode(text.as_bytes());
    let sel = selection.token();
    // Bare forms. BEL terminator for the un-multiplexed case; ESC \ (ST) inside
    // DCS passthrough to avoid ambiguity.
    let bare_bel = format!("{ESC}]52;{sel};{b64}{BEL}");

    if caps.inside_tmux {
        // tmux DCS passthrough: ESC Ptmux; <ESC-doubled inner OSC, ST-terminated> ESC \
        // Requires `set -g allow-passthrough on` (tmux >= 3.3) OR the simpler
        // `set -g set-clipboard on` (then the bare sequence works too).
        let inner_st = format!("{ESC}]52;{sel};{b64}{ESC}\\");
        let doubled = inner_st.replace('\x1b', "\x1b\x1b");
        return format!("{ESC}Ptmux;{doubled}{ESC}\\");
    }
    if caps.inside_screen {
        // GNU screen DCS passthrough (shown un-chunked; screen also chunks long
        // DCS strings every ~256 bytes — acceptable for our capped payloads).
        let inner_st = format!("{ESC}]52;{sel};{b64}{ESC}\\");
        return format!("{ESC}P{inner_st}{ESC}\\");
    }
    bare_bel
}

/// Join a run of VISUAL (soft-wrapped) rows back into ONE logical string,
/// inserting `\n` ONLY at hard boundaries (block starts / author newlines), and
/// NEVER at a soft-wrap continuation. This is the function that PROVES P2 for the
/// "reconstruct from rows" direction: even if some path joins rendered rows, a
/// soft wrap does not become a newline. (The primary copy path bypasses this
/// entirely by reading `block.source`; this exists for copy-mode / row-range
/// selections and as the P2 regression guard.)
#[allow(dead_code)] // P2 row-reconstruction path (copy-mode / drag-select, Phase 3); tested.
pub fn join_visual_rows(rows: &[VisualLine]) -> String {
    let mut out = String::new();
    for (i, vl) in rows.iter().enumerate() {
        if i > 0 {
            if !vl.is_continuation {
                // A new HARD line (author newline / block start) → a real `\n`.
                out.push('\n');
            } else if vl.wrapped_at_word_boundary {
                // A soft wrap that fell at a WORD boundary consumed a space; the
                // trimmed-off space is restored as a single ' ' — NEVER a `\n`.
                out.push(' ');
            }
            // else: a mid-word hard cell-break → concatenate with nothing (and,
            // crucially, still no `\n`). This is the P2 guarantee for the
            // row-reconstruction path: a soft wrap never becomes a newline.
        }
        out.push_str(&vl.text);
    }
    out
}

/// Decide the copy method + payload WITHOUT performing any I/O. Returns the OSC
/// 52 escape to write (if OSC 52 is chosen) and the planned [`CopyResult`]. The
/// effectful [`copy_to_clipboard`] uses this; tests assert on it directly.
///
/// `has_tty` lets tests model the piped-stdout case.
pub fn plan_copy(
    text: &str,
    selection: Selection,
    caps: CopyCaps,
    max_osc52_bytes: usize,
    has_tty: bool,
) -> (Option<String>, CopyResult) {
    let bytes = text.len();

    // 1. OSC 52 first (SSH-safe), unless over the cap or no TTY to write to.
    if bytes <= max_osc52_bytes && has_tty {
        let seq = build_osc52(text, selection, caps);
        return (
            Some(seq),
            CopyResult {
                ok: true,
                method: CopyMethod::Osc52,
                bytes,
                reason: None,
            },
        );
    }

    // OSC 52 unavailable: decide why, then consider native (local only).
    let osc_reason = if bytes > max_osc52_bytes {
        CopyReason::TooLarge
    } else {
        CopyReason::NoTty
    };

    // 2. Native fallback — only when local. Over-cap remote → copy-mode overlay.
    if caps.is_remote {
        let reason = if osc_reason == CopyReason::TooLarge {
            CopyReason::TooLarge // caller opens copy-mode
        } else {
            CopyReason::NoNativeRemote
        };
        return (
            None,
            CopyResult {
                ok: false,
                method: CopyMethod::None,
                bytes,
                reason: Some(reason),
            },
        );
    }

    // Local: signal the caller to try native (it is effectful, done in
    // copy_to_clipboard). We surface the OSC reason so the UI can explain a
    // too-large fallback.
    (
        None,
        CopyResult {
            ok: false,
            method: CopyMethod::Native, // "attempt native" marker
            bytes,
            reason: Some(osc_reason),
        },
    )
}

/// Write `seq` (an OSC 52 escape) to the controlling terminal. Returns whether
/// the write succeeded. Kept tiny + effectful so the rest is testable.
fn write_to_tty(seq: &str) -> bool {
    use std::io::Write;
    let mut out = std::io::stdout();
    out.write_all(seq.as_bytes()).is_ok() && out.flush().is_ok()
}

/// Copy the LOGICAL `text` to the clipboard via the fallback chain. The caller
/// passes the block's `source` (or [`join_visual_rows`] output for a row-range),
/// NEVER raw rendered rows. On `Some(reason == TooLarge)` with `!ok`, the caller
/// should open the copy-mode overlay.
///
/// `has_tty` should be `std::io::stdout().is_terminal()` from the caller (kept a
/// parameter so this stays unit-testable without a real TTY).
pub fn copy_to_clipboard(
    text: &str,
    selection: Selection,
    caps: CopyCaps,
    has_tty: bool,
) -> CopyResult {
    let (seq, plan) = plan_copy(text, selection, caps, DEFAULT_MAX_OSC52_BYTES, has_tty);

    // OSC 52 chosen → write it (fire-and-forget; success unverifiable).
    if let Some(seq) = seq {
        if write_to_tty(&seq) {
            return plan; // ok / Osc52
        }
        // The TTY write itself failed — fall through to native if local.
        if caps.is_remote {
            return CopyResult {
                ok: false,
                method: CopyMethod::None,
                bytes: text.len(),
                reason: Some(CopyReason::NoTty),
            };
        }
        return native_copy(text);
    }

    // No OSC 52. If the plan marked "attempt native" (local), do it now.
    if plan.method == CopyMethod::Native && !plan.ok {
        // A too-large payload that we could still try natively (local). If native
        // also fails, the caller still sees a not-ok result and can open copy-mode.
        return native_copy(text);
    }

    plan
}

/// Set the OS clipboard via `arboard` (in-process, local only). Byte-exact (we
/// hand it the logical string). Any backend error degrades to a not-ok result so
/// the caller can open copy-mode — never a panic.
fn native_copy(text: &str) -> CopyResult {
    let bytes = text.len();
    match arboard::Clipboard::new().and_then(|mut c| c.set_text(text.to_string())) {
        Ok(()) => CopyResult {
            ok: true,
            method: CopyMethod::Native,
            bytes,
            reason: None,
        },
        Err(_) => CopyResult {
            ok: false,
            method: CopyMethod::None,
            bytes,
            reason: Some(CopyReason::NativeError),
        },
    }
}

/// A minimal copy-mode overlay model (STUB — the full alt-screen overlay lands in
/// Phase 3 §7). It holds the RAW logical text to be printed verbatim (no injected
/// breaks, no gutter, no border) so the terminal's own soft-wrap keeps its
/// "wrapped line" flag and OS drag-select reconstructs the original string with
/// no inserted `\n`. The overlay also offers a one-key OSC 52 copy of the same
/// text (best of both).
#[allow(dead_code)] // copy-mode overlay STUB (checklist §3 / §7 deliverable); tested.
#[derive(Debug, Clone)]
pub struct CopyOverlay {
    /// The exact logical text (a block source, a code block, or the transcript).
    pub text: String,
    /// A short label for the hint bar ("last reply", "code block", "transcript").
    pub label: String,
}

#[allow(dead_code)] // overlay API consumed when the Phase-3 copy-mode view lands.
impl CopyOverlay {
    pub fn new(text: impl Into<String>, label: impl Into<String>) -> Self {
        CopyOverlay {
            text: text.into(),
            label: label.into(),
        }
    }

    /// The hint bar shown at the bottom of the overlay.
    pub fn hint(&self) -> String {
        format!(
            "-- COPY MODE ({}) -- drag to select, then your terminal's copy · y = OSC52 copy all · q/Esc to exit",
            self.label
        )
    }

    /// The verbatim text the overlay prints (identity — explicitly NOT wrapped by
    /// us; the terminal wraps it). Centralized so the contract "we never inject
    /// breaks here" is visible and testable.
    pub fn raw(&self) -> &str {
        &self.text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::block::{Block, BlockRole};
    use crate::render::measure::reflow_block;

    // ---- THE HEADLINE TEST: copy across a soft wrap has NO embedded newline --

    #[test]
    fn copy_across_wrap_has_no_newline() {
        // A single long logical line with NO author newline. At a narrow width it
        // soft-wraps into several visual rows. Reconstructing the logical text
        // from those rows MUST NOT introduce any `\n`.
        let long = "the quick brown fox jumps over the lazy dog and keeps on running";
        let block = Block::finalized(1, BlockRole::Assistant, long);

        // It really does wrap (more than one visual row at width 20).
        let rows = reflow_block(&block, 20);
        assert!(rows.len() > 1, "precondition: the line must soft-wrap");

        // (a) Joining the VISUAL rows yields no embedded newline (soft wraps are
        //     continuations → concatenated, never `\n`-joined).
        let joined = join_visual_rows(&rows);
        assert!(
            !joined.contains('\n'),
            "joining soft-wrapped rows must NOT introduce a newline: {joined:?}"
        );
        // And it reconstructs the original words in order (whitespace can differ
        // at the break points, so compare the word sequence).
        let want: Vec<&str> = long.split_whitespace().collect();
        let got: Vec<&str> = joined.split_whitespace().collect();
        assert_eq!(got, want, "the logical words must survive the join in order");

        // (b) The PRIMARY copy path is even stronger: it copies block.source
        //     verbatim. The OSC 52 payload base64-decodes to EXACTLY the source,
        //     newline-free here.
        let seq = build_osc52(&block.source, Selection::Clipboard, CopyCaps {
            is_remote: false,
            inside_tmux: false,
            inside_screen: false,
        });
        let decoded = decode_osc52_payload(&seq);
        assert_eq!(decoded, long);
        assert!(!decoded.contains('\n'));
    }

    #[test]
    fn join_preserves_author_newlines_but_not_soft_wraps() {
        // Two author lines, the first long enough to soft-wrap. The join must put
        // back the ONE author newline and nothing else.
        let src = "alpha beta gamma delta epsilon zeta\nsecond line";
        let block = Block::finalized(2, BlockRole::Assistant, src);
        let rows = reflow_block(&block, 12);
        // First hard line wraps to >1 row; second hard line is its own row(s).
        let joined = join_visual_rows(&rows);
        // Exactly one '\n' (the author newline); the soft wraps add none.
        assert_eq!(joined.matches('\n').count(), 1, "only the author newline survives: {joined:?}");
        let parts: Vec<&str> = joined.split('\n').collect();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].split_whitespace().collect::<Vec<_>>(), vec!["alpha","beta","gamma","delta","epsilon","zeta"]);
        assert_eq!(parts[1], "second line");
    }

    // ---- OSC 52 golden bytes + base64 over the LOGICAL string ----------------

    #[test]
    fn osc52_golden_bytes() {
        let caps = CopyCaps {
            is_remote: false,
            inside_tmux: false,
            inside_screen: false,
        };
        // base64("hi") == "aGk=".
        let seq = build_osc52("hi", Selection::Clipboard, caps);
        assert_eq!(seq, "\x1b]52;c;aGk=\x07");

        // CJK round-trips through UTF-8 base64 (the payload is over bytes).
        let seq = build_osc52("你好", Selection::Clipboard, caps);
        assert_eq!(decode_osc52_payload(&seq), "你好");

        // Selection token wiring.
        assert!(build_osc52("x", Selection::Primary, caps).contains("]52;p;"));
        assert!(build_osc52("x", Selection::Both, caps).contains("]52;cp;"));
    }

    #[test]
    fn osc52_tmux_passthrough_doubles_esc_and_wraps() {
        let caps = CopyCaps {
            is_remote: false,
            inside_tmux: true,
            inside_screen: false,
        };
        let seq = build_osc52("hi", Selection::Clipboard, caps);
        // Wrapped in tmux DCS passthrough.
        assert!(seq.starts_with("\x1bPtmux;"));
        assert!(seq.ends_with("\x1b\\"));
        // Every inner ESC is doubled (no lone 0x1b inside the DCS body except the
        // wrapper's own). The inner OSC's leading ESC became ESC ESC.
        assert!(seq.contains("\x1b\x1b]52;c;aGk="));
    }

    // ---- the fallback chain decision (pure plan) -----------------------------

    #[test]
    fn plan_prefers_osc52_when_tty_and_in_size() {
        let caps = CopyCaps { is_remote: false, inside_tmux: false, inside_screen: false };
        let (seq, res) = plan_copy("small", Selection::Clipboard, caps, DEFAULT_MAX_OSC52_BYTES, true);
        assert!(seq.is_some());
        assert!(res.ok);
        assert_eq!(res.method, CopyMethod::Osc52);
    }

    #[test]
    fn plan_remote_oversize_signals_copy_mode_not_truncate() {
        let caps = CopyCaps { is_remote: true, inside_tmux: false, inside_screen: false };
        let big = "x".repeat(DEFAULT_MAX_OSC52_BYTES + 1);
        let (seq, res) = plan_copy(&big, Selection::Clipboard, caps, DEFAULT_MAX_OSC52_BYTES, true);
        assert!(seq.is_none(), "must NOT emit a truncated OSC 52");
        assert!(!res.ok);
        // Remote + too large → the caller opens copy-mode (reason TooLarge).
        assert_eq!(res.reason, Some(CopyReason::TooLarge));
    }

    #[test]
    fn plan_remote_no_tty_refuses_native() {
        // Remote with no TTY: native would set the wrong machine → refuse.
        let caps = CopyCaps { is_remote: true, inside_tmux: false, inside_screen: false };
        let (seq, res) = plan_copy("hello", Selection::Clipboard, caps, DEFAULT_MAX_OSC52_BYTES, false);
        assert!(seq.is_none());
        assert_eq!(res.reason, Some(CopyReason::NoNativeRemote));
    }

    #[test]
    fn plan_local_no_tty_attempts_native() {
        let caps = CopyCaps { is_remote: false, inside_tmux: false, inside_screen: false };
        let (seq, res) = plan_copy("hello", Selection::Clipboard, caps, DEFAULT_MAX_OSC52_BYTES, false);
        assert!(seq.is_none());
        // Local: marked to attempt native (effectful path does it).
        assert_eq!(res.method, CopyMethod::Native);
        assert!(!res.ok);
    }

    #[test]
    fn copy_overlay_is_verbatim_and_has_hint() {
        let ov = CopyOverlay::new("line one\nline two", "last reply");
        // The overlay text is the EXACT logical string — no wrapping by us.
        assert_eq!(ov.raw(), "line one\nline two");
        assert!(ov.hint().contains("COPY MODE"));
        assert!(ov.hint().contains("last reply"));
        assert!(ov.hint().contains("OSC52"));
    }

    // -- test helper: pull the base64 payload out of an OSC 52 seq and decode --
    fn decode_osc52_payload(seq: &str) -> String {
        // Find ";<sel>;" then read base64 until the terminator (BEL or ESC\).
        // For tmux/screen wrapping callers don't use this; bare seq only here.
        let after_sel = seq
            .split_once(";c;")
            .or_else(|| seq.split_once(";p;"))
            .or_else(|| seq.split_once(";cp;"))
            .map(|(_, rest)| rest)
            .unwrap_or("");
        let payload: String = after_sel
            .chars()
            .take_while(|&c| c != '\x07' && c != '\x1b')
            .collect();
        let bytes = BASE64.decode(payload.as_bytes()).unwrap_or_default();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
