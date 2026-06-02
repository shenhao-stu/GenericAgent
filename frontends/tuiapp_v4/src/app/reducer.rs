//! app/reducer.rs — the ONE protocol fold (ARCH Fix C / kills F8a). A `CoreToUi`
//! frame is folded through the shared [`apply_frame`] match skeleton, which
//! dispatches each arm to a [`FrameSink`] hook. BOTH the active-session
//! [`AppState`] and a background [`Session`] (`app::session`) implement the
//! trait, so a new frame variant is added in ONE place (the fold + one trait
//! method the compiler then forces both sinks to handle) instead of two
//! hand-maintained matches. The per-arm BEHAVIOR still differs by sink (the
//! active session drives overlays/cost/effects; a background session only
//! updates its own record) — that difference lives in the hook impls, the fold
//! is identical for both.
//!
//! TRANSCRIPT HYGIENE (redesign_cc.md §1/§c): the bridge's STDERR / parse-noise
//! is NEVER a transcript row. GA's failover diagnostics (`[MixinSession] …retry
//! N/M`) + core `print()` chatter route to a DEBUG-ONLY log; a FATAL
//! `SpawnFailed`/`ChildExited` becomes a CONNECTION STATUS (the footer chip — N1
//! "never a silent disconnect"), not a transcript notice that scrolls away.

use crate::app::session::Session;
use crate::app::{AppState, Block, ConnStatus, PendingAsk, Role};
use crate::bridge::protocol::{CoreToUi, LlmItem};
use crate::bridge::BridgeEvent;

/// The per-arm hooks the shared [`apply_frame`] fold dispatches to. Each method
/// is one `CoreToUi` arm; an impl carries the EXACT behavior that sink had
/// before the dedup (so the fold is behavior-identical for both). `now_ms` is
/// the caller's injected monotonic clock (keeps the fold pure/testable).
pub(in crate::app) trait FrameSink {
    fn on_ready(&mut self, model: Option<String>, llm: Option<String>, model_real: Option<String>);
    fn on_message_begin(&mut self, mid: String, role: String, now_ms: u64);
    fn on_message_delta(&mut self, mid: String, text: String);
    fn on_message_end(&mut self, mid: String, now_ms: u64);
    fn on_ask_user(&mut self, ask_id: String, question: String, options: Vec<crate::bridge::protocol::AskUserOption>, free_text: bool);
    #[allow(clippy::too_many_arguments)]
    fn on_status(
        &mut self,
        model: Option<String>,
        llm: Option<String>,
        model_real: Option<String>,
        context_percent: Option<f64>,
        context_used: Option<u64>,
        context_limit: Option<u64>,
        tokens: Option<u64>,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
        cache_tokens: Option<u64>,
        last_input: Option<u64>,
        last_output: Option<u64>,
    );
    fn on_llm_list(&mut self, items: Vec<LlmItem>);
    fn on_btw_answer(&mut self, ask_id: String, text: Option<String>, error: Option<String>);
    fn on_rewind_result(&mut self, dropped: u32, remaining: u32);
    fn on_error(&mut self, message: String, fatal: bool);
}

/// The ONE fold (F8a): dispatch a parsed `CoreToUi` frame to the sink's hooks.
/// `Pong` is liveness only — no state change for either sink.
pub(in crate::app) fn apply_frame<S: FrameSink>(sink: &mut S, frame: CoreToUi, now_ms: u64) {
    match frame {
        CoreToUi::Ready { model, llm, model_real, .. } => sink.on_ready(model, llm, model_real),
        CoreToUi::MessageBegin { mid, role } => sink.on_message_begin(mid, role, now_ms),
        CoreToUi::MessageDelta { mid, text } => sink.on_message_delta(mid, text),
        CoreToUi::MessageEnd { mid, .. } => sink.on_message_end(mid, now_ms),
        CoreToUi::AskUser { ask_id, question, options, free_text } => {
            sink.on_ask_user(ask_id, question, options, free_text)
        }
        CoreToUi::Status {
            model,
            llm,
            model_real,
            context_percent,
            context_used,
            context_limit,
            tokens,
            input_tokens,
            output_tokens,
            cache_tokens,
            last_input,
            last_output,
            ..
        } => sink.on_status(
            model,
            llm,
            model_real,
            context_percent,
            context_used,
            context_limit,
            tokens,
            input_tokens,
            output_tokens,
            cache_tokens,
            last_input,
            last_output,
        ),
        CoreToUi::Pong { .. } => {}
        CoreToUi::LlmList { items } => sink.on_llm_list(items),
        CoreToUi::BtwAnswer { ask_id, text, error } => sink.on_btw_answer(ask_id, text, error),
        CoreToUi::RewindResult { dropped, remaining } => sink.on_rewind_result(dropped, remaining),
        CoreToUi::Error { message, fatal, .. } => sink.on_error(message, fatal),
    }
}

/// Clip a tool result to a short PREVIEW for the `/verbose` inspector's `result`
/// field (the full text is kept separately as `raw`): the first
/// [`AUDIT_PREVIEW_LINES`] non-empty content lines, joined with `\n`, each capped
/// to [`AUDIT_PREVIEW_COLS`] chars (char-count cap — the styled pane re-wraps), an
/// `…` appended to a line that was cut and a final `…` line when lines were
/// dropped. Bounded by `.take()` (terminates on a finite slice; no manual loop).
/// PURE.
fn clip_audit_preview(result: &str) -> String {
    const AUDIT_PREVIEW_LINES: usize = 8;
    const AUDIT_PREVIEW_COLS: usize = 200;
    let trimmed = result.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let all: Vec<&str> = trimmed.lines().filter(|l| !l.trim().is_empty()).collect();
    let mut out: Vec<String> = all
        .iter()
        .take(AUDIT_PREVIEW_LINES)
        .map(|l| {
            if l.chars().count() > AUDIT_PREVIEW_COLS {
                let cut: String = l.chars().take(AUDIT_PREVIEW_COLS).collect();
                format!("{cut}…")
            } else {
                (*l).to_string()
            }
        })
        .collect();
    if all.len() > AUDIT_PREVIEW_LINES {
        out.push("…".to_string());
    }
    out.join("\n")
}

impl AppState {
    /// Fold one bridge event into the ACTIVE-session state. PURE w.r.t. I/O (only
    /// mutates self; `now_ms` injected). A `Frame` goes through the shared
    /// [`apply_frame`] fold; the non-frame lifecycle events surface a FATAL as the
    /// connection status (never a transcript row — §c).
    pub fn apply_bridge_event(&mut self, ev: BridgeEvent, now_ms: u64) {
        match ev {
            BridgeEvent::Frame(frame) => apply_frame(self, frame, now_ms),
            BridgeEvent::SpawnFailed { detail } => {
                self.conn = ConnStatus::Disconnected { reason: detail.clone() };
                self.busy = false;
                self.push_bridge_debug(format!("spawn failed: {detail}"));
                self.effects.flash_lightning();
            }
            BridgeEvent::ParseNoise { line } => {
                self.push_bridge_debug(format!("[unparsed] {line}"));
            }
            BridgeEvent::ChildExited { code } => {
                let reason = match code {
                    Some(c) => format!("bridge exited (code {c})"),
                    None => "bridge exited".to_string(),
                };
                self.conn = ConnStatus::Disconnected { reason: reason.clone() };
                self.busy = false;
                self.push_bridge_debug(reason);
                self.effects.flash_lightning();
            }
            BridgeEvent::Stderr { line } => {
                // ga_bridge.py routes core print()/tracebacks + failover retry
                // diagnostics here. SUPPRESS from the transcript (debug-only log,
                // §c); the disconnect path above surfaces FATAL failures via `conn`.
                self.push_bridge_debug(line);
            }
        }
    }

    /// Store the wire model identity (`llm` + `model_real`), shared by `Ready` and
    /// `Status` so a mid-turn failover live-updates the header. Only overwrites a
    /// field when the frame carried it (a stale bridge omitting them keeps the prior).
    fn store_identity(&mut self, llm: Option<String>, model_real: Option<String>) {
        if llm.is_some() {
            self.llm_name = llm;
        }
        if model_real.is_some() {
            self.model_real = model_real;
        }
    }

    /// Find the (in-flight) block for a given mid, newest first.
    fn block_for_mid_mut(&mut self, mid: &str) -> Option<&mut Block> {
        self.transcript
            .iter_mut()
            .rev()
            .find(|b| b.mid.as_deref() == Some(mid))
    }
}

impl FrameSink for AppState {
    fn on_ready(&mut self, model: Option<String>, llm: Option<String>, model_real: Option<String>) {
        self.model = model.clone();
        self.conn = ConnStatus::Connected { model };
        self.store_identity(llm, model_real);
    }

    fn on_message_begin(&mut self, mid: String, role: String, now_ms: u64) {
        self.busy = true;
        self.turn_started_ms = now_ms;
        // A new turn starts → retire the previous turn's frozen done-line (Q7) so it
        // only ever shows in the idle gap BETWEEN turns, never bleeding into one.
        self.last_turn_ms = None;
        // Reset eased display counters so each new turn ramps fresh from zero.
        self.display_tok_in = None;
        self.display_tok_out = None;
        let id = self.alloc_block_id();
        self.transcript
            .push(Block::new(id, Some(mid), Role::from_proto(&role), String::new(), false));
    }

    fn on_message_delta(&mut self, mid: String, text: String) {
        if let Some(block) = self.block_for_mid_mut(&mid) {
            block.source.push_str(&text);
            // Bump rev so the wrap cache reflows ONLY this block (P1).
            block.rev = block.rev.wrapping_add(1);
        } else {
            // A delta with no begin — create a block so text isn't lost.
            let id = self.alloc_block_id();
            self.transcript
                .push(Block::new(id, Some(mid), Role::Assistant, text, false));
            self.busy = true;
        }
    }

    fn on_message_end(&mut self, mid: String, now_ms: u64) {
        let mut finalized_source: Option<String> = None;
        if let Some(block) = self.block_for_mid_mut(&mid)
            && !block.finalized
        {
            block.finalized = true;
            block.rev = block.rev.wrapping_add(1);
            if block.role == Role::Assistant {
                finalized_source = Some(block.source.clone());
            }
        }
        // Harvest tool calls in the just-finalized assistant message into the
        // `/verbose` inspector trail as full records (S7; split-borrow: read the
        // source, then push). Each chip's `name/args/result/status` come straight
        // from `parse_tool_calls`; GA has NO distinct raw payload, so `raw` is the
        // full result text and `result` is a clipped preview (first lines) — the
        // inspector's Result vs Raw fields. `parse_tool_calls`' s per-message id is
        // ignored; `push_tool_audit` assigns the session-wide monotonic `t{id}`.
        if let Some(src) = finalized_source {
            for tc in crate::render::chip::parse_tool_calls(&src) {
                let raw = tc.result.clone();
                let result = clip_audit_preview(&tc.result);
                self.push_tool_audit(tc.name, tc.args, result, tc.status, raw);
            }
        }
        // Freeze this turn's duration for the above-composer done-line (Q7) BEFORE
        // clearing busy — `turn_elapsed_ms` reads 0 once idle, so capture it here.
        self.last_turn_ms = Some(now_ms.saturating_sub(self.turn_started_ms));
        self.busy = false;
        self.effects.burst_sparkle();
    }

    fn on_ask_user(&mut self, ask_id: String, question: String, options: Vec<crate::bridge::protocol::AskUserOption>, free_text: bool) {
        let preview = format!("? {question}");
        let ask = PendingAsk { ask_id, question, options, free_text };
        self.push_notice(preview);
        self.busy = false;
        if self.pending_ask.is_some() {
            // A parallel ask arrived while one is being answered → QUEUE it so it
            // surfaces after the current one (§7 queued asks).
            self.ask_queue.push_back(ask);
        } else {
            self.pending_ask = Some(ask);
            // Surface the unified ask_user CARD automatically (§7) — but never
            // clobber an overlay the user already has open.
            if self.overlay.is_none() {
                self.open_ask_user();
            }
        }
    }

    fn on_status(
        &mut self,
        model: Option<String>,
        llm: Option<String>,
        model_real: Option<String>,
        context_percent: Option<f64>,
        context_used: Option<u64>,
        context_limit: Option<u64>,
        tokens: Option<u64>,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
        cache_tokens: Option<u64>,
        last_input: Option<u64>,
        last_output: Option<u64>,
    ) {
        if let Some(m) = model {
            self.model = Some(m.clone());
            if let ConnStatus::Connected { model: cm } = &mut self.conn {
                *cm = Some(m);
            }
        }
        // A mid-turn failover re-emits Status with a NEW active member — live-update.
        self.store_identity(llm, model_real);
        if let Some(p) = context_percent {
            self.context_percent = Some(p);
            self.cost.context_percent = Some(p);
        }
        // Raw char counts (S4) — only set when present (an older bridge omits them;
        // keep the last-known so the readout doesn't flicker to `—`).
        if context_used.is_some() {
            self.context_used = context_used;
        }
        if context_limit.is_some() {
            self.context_limit = context_limit;
        }
        if let Some(tk) = tokens {
            self.tokens = Some(tk);
        }
        // Cumulative split totals are AUTHORITATIVE when the bridge sends them, so
        // SET them rather than the old `max`-guess. `cache_tokens` is cache reads.
        if let Some(i) = input_tokens {
            self.cost.input = i;
        }
        if let Some(o) = output_tokens {
            self.cost.output = o;
        }
        if let Some(c) = cache_tokens {
            self.cost.cache = c;
        }
        // Per-call snapshots drive the spinner's `↑in ↓out` live readout.
        if last_input.is_some() {
            self.tok_in = last_input;
        }
        if last_output.is_some() {
            self.tok_out = last_output;
        }
        // Legacy bridge (tokens only, no split) → keep a sane /cost total by
        // folding the running count into output; the new bridge sends the split.
        if input_tokens.is_none() && output_tokens.is_none() {
            if let Some(tk) = tokens {
                self.cost.output = self.cost.output.max(tk);
            }
        }
    }

    fn on_llm_list(&mut self, items: Vec<LlmItem>) {
        self.apply_llm_list(items)
    }

    fn on_btw_answer(&mut self, ask_id: String, text: Option<String>, error: Option<String>) {
        self.apply_btw_answer(ask_id, text, error)
    }

    fn on_rewind_result(&mut self, dropped: u32, remaining: u32) {
        // A NON-history acknowledgment: surface as a notice (the rewind already
        // truncated the local display when the user picked).
        self.push_notice(format!(
            "{} {} {} · {} {}",
            crate::i18n::t(self.lang, "rewind.done"),
            dropped,
            crate::i18n::t(self.lang, "rewind.turns_suffix"),
            remaining,
            crate::i18n::t(self.lang, "status.total"),
        ));
    }

    fn on_error(&mut self, message: String, fatal: bool) {
        self.push_notice(format!("error: {message}"));
        if fatal {
            self.conn = ConnStatus::Disconnected { reason: message };
            self.busy = false;
        }
    }
}

impl FrameSink for Session {
    fn on_ready(&mut self, model: Option<String>, _llm: Option<String>, _model_real: Option<String>) {
        self.model = model.clone();
        self.conn = ConnStatus::Connected { model };
    }

    fn on_message_begin(&mut self, mid: String, role: String, now_ms: u64) {
        self.busy = true;
        self.busy_since_ms = now_ms;
        self.pending_ask = None;
        let id = self.alloc_block_id();
        self.transcript
            .push(Block::new_external(id, Some(mid), Role::from_proto(&role), String::new(), false));
    }

    fn on_message_delta(&mut self, mid: String, text: String) {
        if let Some(block) = self.block_for_mid_mut(&mid) {
            block.source.push_str(&text);
            block.rev = block.rev.wrapping_add(1);
        } else {
            let id = self.alloc_block_id();
            self.transcript
                .push(Block::new_external(id, Some(mid), Role::Assistant, text, false));
            self.busy = true;
        }
        self.had_reply = true;
    }

    fn on_message_end(&mut self, mid: String, _now_ms: u64) {
        if let Some(block) = self.block_for_mid_mut(&mid)
            && !block.finalized
        {
            block.finalized = true;
            block.rev = block.rev.wrapping_add(1);
            if block.role == Role::Assistant {
                self.had_reply = true;
            }
        }
        self.busy = false;
    }

    fn on_ask_user(&mut self, ask_id: String, question: String, options: Vec<crate::bridge::protocol::AskUserOption>, free_text: bool) {
        self.push_notice(format!("? {question}"));
        self.pending_ask = Some(PendingAsk { ask_id, question, options, free_text });
        self.busy = false;
    }

    fn on_status(
        &mut self,
        model: Option<String>,
        _llm: Option<String>,
        _model_real: Option<String>,
        context_percent: Option<f64>,
        _context_used: Option<u64>,
        _context_limit: Option<u64>,
        _tokens: Option<u64>,
        _input_tokens: Option<u64>,
        _output_tokens: Option<u64>,
        _cache_tokens: Option<u64>,
        _last_input: Option<u64>,
        _last_output: Option<u64>,
    ) {
        if let Some(m) = model {
            self.model = Some(m.clone());
            if let ConnStatus::Connected { model: cm } = &mut self.conn {
                *cm = Some(m);
            }
        }
        // Store ctx per-session (mirrors the active reducer) so a session promoted
        // to active carries its fill percent instead of a blank `ctx —` (Slice 4).
        if let Some(p) = context_percent {
            self.context_percent = Some(p);
        }
    }

    // A background session never opens the `/llm` overlay / fills a `/btw` card /
    // surfaces a `/rewind` notice — those live on the ACTIVE session's reducer.
    fn on_llm_list(&mut self, _items: Vec<LlmItem>) {}
    fn on_btw_answer(&mut self, _ask_id: String, _text: Option<String>, _error: Option<String>) {}
    fn on_rewind_result(&mut self, _dropped: u32, _remaining: u32) {}

    fn on_error(&mut self, message: String, fatal: bool) {
        self.push_notice(format!("error: {message}"));
        if fatal {
            self.conn = ConnStatus::Disconnected { reason: message };
            self.busy = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::session::Session;

    /// Slice 4 ctx-blank root-cause: the BACKGROUND `on_status` reducer now STORES
    /// `context_percent` (it used to ignore it as `_context_percent`), so a session
    /// whose status folds through the background sink carries its fill percent
    /// instead of leaving the footer at a blank `ctx —` after promotion.
    #[test]
    fn background_on_status_stores_context_percent() {
        let mut s = Session::new(1, "bg");
        assert_eq!(s.context_percent, None, "no ctx before any status");
        apply_frame(
            &mut s,
            CoreToUi::Status {
                model: Some("glm".into()),
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
            },
            0,
        );
        assert_eq!(s.context_percent, Some(48.0), "background on_status stores ctx");
        assert_eq!(s.model.as_deref(), Some("glm"), "model still tracked too");
    }
}
