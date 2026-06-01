//! bridge/protocol.rs — serde enums for the CoreToUi / UiToCore JSONL frames
//! exchanged with the GA Python core over the child's stdio.
//!
//! AUTHORITATIVE SCHEMA: this mirrors `scripts/ga_bridge.py` (and the Ink
//! reference `tuiapp_v4_ink_backup/src/bridge/protocol.ts`) EXACTLY. Every frame
//! is one JSON object per line, discriminated by a `"type"` field
//! (`#[serde(tag = "type")]`).
//!
//!   UiToCore (stdin → core):
//!     Submit{text, images?}  Abort/Cancel{mid?}  Intervene{text}
//!     SwitchLlm{n}  ListLlms  Ping{nonce}  Command{name,args}  Answer{...}  Shutdown
//!   CoreToUi (core → stdout):
//!     Ready{version?,model?}  MessageBegin{mid,role}  MessageDelta{mid,text}
//!     MessageEnd{mid,reason}  AskUser{ask_id,question,options,free_text}
//!     Status{model?,...}  Pong{nonce}  LlmList{items}  Error{message,code?,fatal}
//!
//! Load-bearing logic (serde shape) is kept pure + unit-tested at the bottom.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// CoreToUi — frames the Python core sends up to the UI.
// ---------------------------------------------------------------------------

/// One frame from the GA core to the UI. `#[serde(tag = "type")]` makes the
/// JSON shape `{"type":"MessageDelta","mid":"m1","text":"hi"}` exactly mirror
/// `ga_bridge.py`'s `self.emit({"type": ...})` payloads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CoreToUi {
    /// Handshake — the bridge is up. `model` is present only when an LLM is
    /// actually configured (ga_bridge.py emits a bare `Ready` on a degraded
    /// start). Receiving this == "connected".
    Ready {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        version: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        /// Active config name (e.g. `codex-pro`) — additive (serde-default None so
        /// an older bridge that omits it still parses).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        llm: Option<String>,
        /// Real underlying model (e.g. `gpt-5.5`), `[...]` tag stripped — additive.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model_real: Option<String>,
    },
    /// A new assistant/tool/system message has started streaming.
    MessageBegin {
        mid: String,
        #[serde(default = "default_role")]
        role: String,
    },
    /// An incremental chunk of text for an in-flight message.
    MessageDelta { mid: String, text: String },
    /// A message finished. `reason` ∈ stop | length | abort | error | end.
    MessageEnd {
        mid: String,
        #[serde(default = "default_reason")]
        reason: String,
    },
    /// The core is asking the user a question (human-intervention sentinel).
    AskUser {
        ask_id: String,
        question: String,
        #[serde(default)]
        options: Vec<AskUserOption>,
        #[serde(default = "default_true")]
        free_text: bool,
    },
    /// A status update (current model / context / token snapshot / free-form
    /// line). The token fields are ADDITIVE + optional: an older bridge that emits
    /// only `{model}` still parses (serde default = None). `tokens` is the grand
    /// total (footer `~N tok`); `input_tokens`/`output_tokens`/`cache_tokens` are
    /// cumulative split totals (the `/cost` card); `last_input`/`last_output` are
    /// the per-call sizes that drive the spinner's `↑/↓` live counters.
    Status {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        /// Active config name (additive; mirrors `Ready.llm`). A mid-turn failover
        /// re-emits Status, so this live-updates the header.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        llm: Option<String>,
        /// Real underlying model (additive; mirrors `Ready.model_real`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model_real: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        context_percent: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tokens: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        input_tokens: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output_tokens: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_tokens: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        last_input: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        last_output: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
    },
    /// Reply to a `Ping{nonce}` — liveness echo.
    Pong { nonce: String },
    /// The configured LLM list, in reply to a `ListLlms` request (N3 / §4 `/llm`
    /// picker). `items` is `agent.list_llms()` mapped onto `(idx, name, current)`:
    /// `idx` is the 0-based position into the configured list, `name` is the
    /// `"SessionType/name"` label, `current` marks the active one (`●`). The picker
    /// shows rows `"● i. name"` and Enter sends `SwitchLlm{n}` (1-based, see below).
    LlmList { items: Vec<LlmItem> },
    /// The answer to a background `/btw` side-question (§7 / §4 `/btw`). The bridge
    /// runs the side-ask on a worker thread (the main turn is NOT blocked) and
    /// emits this when it completes; the UI routes it to the EPHEMERAL `/btw` card
    /// (the `querying…` → answer flow) — NOT into the transcript history, so a side
    /// question never pollutes the conversation. `error` is set instead of `text`
    /// on failure so the card can show a reason rather than hang. `ask_id` ties the
    /// answer to the card that asked (so a stale card can't show a new answer).
    BtwAnswer {
        ask_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Confirmation that the backend history was truncated by `n` real turns in
    /// reply to a `Rewind{n}` frame (§7 `/rewind`). `dropped` is the number of
    /// real (user) turns actually removed from `llmclient.backend.history`;
    /// `remaining` is the turn count left. The UI surfaces this as a notice (not a
    /// streamed message) so a rewind is acknowledged without a model turn.
    RewindResult {
        dropped: u32,
        #[serde(default)]
        remaining: u32,
    },
    /// An error surfaced by the bridge or the core. `fatal` => the child should
    /// be considered dead. NEVER silently swallowed (N1).
    Error {
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<String>,
        #[serde(default)]
        fatal: bool,
    },
}

/// One selectable option attached to an `AskUser` frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AskUserOption {
    pub id: String,
    pub label: String,
}

/// One configured LLM in a `LlmList` frame (N3). Mirrors a `(idx, name, current)`
/// triple from `agent.list_llms()`: `idx` is the 0-based list position, `name` the
/// `"SessionType/name"` label, `current` whether it is the active model (`●`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmItem {
    pub idx: u32,
    pub name: String,
    #[serde(default)]
    pub current: bool,
}

// ---------------------------------------------------------------------------
// UiToCore — frames the UI sends down to the Python core.
// ---------------------------------------------------------------------------

/// One frame from the UI down to the GA core. Serializes to the same tagged
/// JSON `ga_bridge.py.handle()` dispatches on (`frame.get("type")`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum UiToCore {
    /// Submit a user message (optionally with images). Maps to `put_task`.
    Submit {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        images: Option<Vec<SubmitImage>>,
    },
    /// Abort the running turn (alias of `Cancel`). `agent.abort()` + `_stop`.
    Abort {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mid: Option<String>,
    },
    /// Cancel — kept distinct so the Ink/ga_bridge `Cancel` shape round-trips.
    Cancel {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mid: Option<String>,
    },
    /// Inject text into a live turn at the next boundary (`_intervene` file).
    Intervene { text: String },
    /// Switch the active LLM. `n` is 1-based into the configured list.
    SwitchLlm { n: u32 },
    /// Ask the core for the configured LLM list — it replies `LlmList{items}`
    /// (N3 / §4 `/llm` picker). Additive; `ga_bridge.py` handles it by emitting
    /// `agent.list_llms()`.
    ListLlms,
    /// Rewind the conversation by `n` REAL (user) turns (§7 `/rewind`). The bridge's
    /// `handle_rewind` truncates `llmclient.backend.history` by `n` user turns (a
    /// user message + its assistant reply each), aborts any live turn, and replies
    /// `RewindResult{dropped, remaining}`. Additive; backward-compatible (an older
    /// bridge that doesn't know `Rewind` simply ignores it — the UI still truncates
    /// its own display, so the rewind is at worst display-only on a stale bridge).
    Rewind { n: u32 },
    /// Fire a background `/btw` side-question (§7 `/btw`). The bridge runs it on a
    /// WORKER thread so the main turn is never blocked, and replies `BtwAnswer`
    /// tagged with the same `ask_id`. Distinct from a `Command` so the bridge can
    /// fork a thread + isolate the side-ask from the main display queue (a side
    /// question must NOT stream into the transcript). `ask_id` ties the reply back
    /// to the card.
    BtwAsk { ask_id: String, text: String },
    /// Liveness probe — the core replies with `Pong{nonce}`.
    Ping { nonce: String },
    /// A structured slash command (e.g. `restore`, `shell`) or a forwarded one.
    Command {
        name: String,
        #[serde(default)]
        args: String,
    },
    /// Answer to a prior `AskUser`.
    Answer {
        ask_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        option_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
    },
    /// Ask the bridge to exit its read loop cleanly (never hangs).
    Shutdown,
}

/// An image payload attached to a `Submit` frame (base64 data or a path).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubmitImage {
    pub data: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime: Option<String>,
}

// ---------------------------------------------------------------------------
// serde defaults (mirror the zod `.default(...)` in protocol.ts).
// ---------------------------------------------------------------------------

fn default_role() -> String {
    "assistant".to_string()
}
fn default_reason() -> String {
    "stop".to_string()
}
fn default_true() -> bool {
    true
}

// ---------------------------------------------------------------------------
// Pure parse / serialize helpers (the load-bearing, testable surface).
// ---------------------------------------------------------------------------

impl CoreToUi {
    /// Parse one JSONL line from the core into a frame. Returns `None` on an
    /// empty line or a parse failure (the caller surfaces parse errors as a
    /// visible status — it must never silently hang, N1).
    pub fn parse_line(line: &str) -> Option<CoreToUi> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return None;
        }
        serde_json::from_str::<CoreToUi>(trimmed).ok()
    }
}

impl UiToCore {
    /// Serialize this frame to a single JSONL line (NO trailing newline — the
    /// writer appends `\n` + flush). Infallible for these plain enums.
    pub fn to_line(&self) -> String {
        serde_json::to_string(self).expect("UiToCore serializes")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip every CoreToUi + UiToCore variant through serde and confirm
    /// the tagged JSON shape matches `ga_bridge.py` (the wire contract). This is
    /// the load-bearing protocol test (deliverable: protocol round-trip).
    #[test]
    fn protocol_round_trip_serde() {
        // --- CoreToUi: parse the EXACT bytes ga_bridge.py emits. ---
        let ready = CoreToUi::parse_line(r#"{"type":"Ready","version":"1","model":"glm"}"#).unwrap();
        assert_eq!(
            ready,
            CoreToUi::Ready {
                version: Some("1".into()),
                model: Some("glm".into()),
                llm: None,
                model_real: None
            }
        );

        // New bridge: Ready carries the additive llm + model_real fields.
        let ready_id = CoreToUi::parse_line(
            r#"{"type":"Ready","version":"1","model":"MixinSession/codex-pro|getoken_20x","llm":"codex-pro","model_real":"gpt-5.5"}"#,
        )
        .unwrap();
        assert_eq!(
            ready_id,
            CoreToUi::Ready {
                version: Some("1".into()),
                model: Some("MixinSession/codex-pro|getoken_20x".into()),
                llm: Some("codex-pro".into()),
                model_real: Some("gpt-5.5".into())
            }
        );

        // Degraded Ready (no model) — ga_bridge.py emits this when no LLM.
        let bare = CoreToUi::parse_line(r#"{"type":"Ready","version":"1"}"#).unwrap();
        assert_eq!(
            bare,
            CoreToUi::Ready {
                version: Some("1".into()),
                model: None,
                llm: None,
                model_real: None
            }
        );

        let begin =
            CoreToUi::parse_line(r#"{"type":"MessageBegin","mid":"m1","role":"assistant"}"#).unwrap();
        assert_eq!(
            begin,
            CoreToUi::MessageBegin {
                mid: "m1".into(),
                role: "assistant".into()
            }
        );

        // role defaults to "assistant" when omitted.
        let begin_default = CoreToUi::parse_line(r#"{"type":"MessageBegin","mid":"m2"}"#).unwrap();
        assert_eq!(
            begin_default,
            CoreToUi::MessageBegin {
                mid: "m2".into(),
                role: "assistant".into()
            }
        );

        let delta =
            CoreToUi::parse_line(r#"{"type":"MessageDelta","mid":"m1","text":"你好"}"#).unwrap();
        assert_eq!(
            delta,
            CoreToUi::MessageDelta {
                mid: "m1".into(),
                text: "你好".into()
            }
        );

        let end = CoreToUi::parse_line(r#"{"type":"MessageEnd","mid":"m1","reason":"abort"}"#).unwrap();
        assert_eq!(
            end,
            CoreToUi::MessageEnd {
                mid: "m1".into(),
                reason: "abort".into()
            }
        );

        let ask = CoreToUi::parse_line(
            r#"{"type":"AskUser","ask_id":"a1","question":"pick?","options":[{"id":"0","label":"yes"}],"free_text":true}"#,
        )
        .unwrap();
        assert_eq!(
            ask,
            CoreToUi::AskUser {
                ask_id: "a1".into(),
                question: "pick?".into(),
                options: vec![AskUserOption {
                    id: "0".into(),
                    label: "yes".into()
                }],
                free_text: true,
            }
        );

        let status = CoreToUi::parse_line(r#"{"type":"Status","model":"glm"}"#).unwrap();
        assert_eq!(
            status,
            CoreToUi::Status {
                model: Some("glm".into()),
                llm: None,
                model_real: None,
                context_percent: None,
                tokens: None,
                input_tokens: None,
                output_tokens: None,
                cache_tokens: None,
                last_input: None,
                last_output: None,
                text: None
            }
        );

        // The full token snapshot the bridge emits mid-turn + before MessageEnd
        // (the /cost-wiring payload). Confirms the additive fields round-trip.
        let status_full = CoreToUi::parse_line(
            r#"{"type":"Status","model":"glm","llm":"codex-pro","model_real":"gpt-5.5","context_percent":42.5,"tokens":1550,"input_tokens":1200,"output_tokens":350,"cache_tokens":90,"last_input":800,"last_output":120}"#,
        )
        .unwrap();
        assert_eq!(
            status_full,
            CoreToUi::Status {
                model: Some("glm".into()),
                llm: Some("codex-pro".into()),
                model_real: Some("gpt-5.5".into()),
                context_percent: Some(42.5),
                tokens: Some(1550),
                input_tokens: Some(1200),
                output_tokens: Some(350),
                cache_tokens: Some(90),
                last_input: Some(800),
                last_output: Some(120),
                text: None,
            }
        );

        let pong = CoreToUi::parse_line(r#"{"type":"Pong","nonce":"xyz"}"#).unwrap();
        assert_eq!(pong, CoreToUi::Pong { nonce: "xyz".into() });

        // LlmList: ga_bridge.py emits agent.list_llms() as (idx,name,current).
        let llms = CoreToUi::parse_line(
            r#"{"type":"LlmList","items":[{"idx":0,"name":"OpenAI/gpt","current":true},{"idx":1,"name":"GLM/glm-4"}]}"#,
        )
        .unwrap();
        assert_eq!(
            llms,
            CoreToUi::LlmList {
                items: vec![
                    LlmItem { idx: 0, name: "OpenAI/gpt".into(), current: true },
                    LlmItem { idx: 1, name: "GLM/glm-4".into(), current: false },
                ]
            }
        );

        // BtwAnswer: the background side-question reply (routed to the /btw card).
        let btw = CoreToUi::parse_line(
            r#"{"type":"BtwAnswer","ask_id":"b1","text":"42"}"#,
        )
        .unwrap();
        assert_eq!(
            btw,
            CoreToUi::BtwAnswer { ask_id: "b1".into(), text: Some("42".into()), error: None }
        );
        // A BtwAnswer error path (no text).
        let btw_err = CoreToUi::parse_line(
            r#"{"type":"BtwAnswer","ask_id":"b1","error":"no llm"}"#,
        )
        .unwrap();
        assert_eq!(
            btw_err,
            CoreToUi::BtwAnswer { ask_id: "b1".into(), text: None, error: Some("no llm".into()) }
        );

        // RewindResult: the truncation confirmation (a notice, not a message).
        let rw = CoreToUi::parse_line(r#"{"type":"RewindResult","dropped":2,"remaining":3}"#).unwrap();
        assert_eq!(rw, CoreToUi::RewindResult { dropped: 2, remaining: 3 });
        // `remaining` defaults to 0 when omitted (backward tolerant).
        let rw0 = CoreToUi::parse_line(r#"{"type":"RewindResult","dropped":1}"#).unwrap();
        assert_eq!(rw0, CoreToUi::RewindResult { dropped: 1, remaining: 0 });

        let err =
            CoreToUi::parse_line(r#"{"type":"Error","message":"boom","code":"x","fatal":true}"#)
                .unwrap();
        assert_eq!(
            err,
            CoreToUi::Error {
                message: "boom".into(),
                code: Some("x".into()),
                fatal: true
            }
        );

        // Garbage / empty lines parse to None, never panic (N1: surface, don't hang).
        assert!(CoreToUi::parse_line("not json").is_none());
        assert!(CoreToUi::parse_line("").is_none());
        assert!(CoreToUi::parse_line(r#"{"type":"Nope"}"#).is_none());

        // --- UiToCore: serialize and confirm ga_bridge.py would dispatch it. ---
        let submit = UiToCore::Submit {
            text: "hi".into(),
            images: None,
        };
        assert_eq!(submit.to_line(), r#"{"type":"Submit","text":"hi"}"#);

        // Re-parse it back (full round trip).
        let reparsed: UiToCore = serde_json::from_str(&submit.to_line()).unwrap();
        assert_eq!(reparsed, submit);

        let ping = UiToCore::Ping {
            nonce: "n1".into(),
        };
        assert_eq!(ping.to_line(), r#"{"type":"Ping","nonce":"n1"}"#);

        let switch = UiToCore::SwitchLlm { n: 3 };
        assert_eq!(switch.to_line(), r#"{"type":"SwitchLlm","n":3}"#);

        // ListLlms is a bare request (no fields) the bridge replies to with LlmList.
        let list = UiToCore::ListLlms;
        assert_eq!(list.to_line(), r#"{"type":"ListLlms"}"#);
        let list_reparsed: UiToCore = serde_json::from_str(&list.to_line()).unwrap();
        assert_eq!(list_reparsed, UiToCore::ListLlms);

        // Rewind{n}: the /rewind truncation request (ga_bridge.handle_rewind).
        let rewind = UiToCore::Rewind { n: 2 };
        assert_eq!(rewind.to_line(), r#"{"type":"Rewind","n":2}"#);
        let rewind_reparsed: UiToCore = serde_json::from_str(&rewind.to_line()).unwrap();
        assert_eq!(rewind_reparsed, rewind);

        // BtwAsk: the background /btw side-question request.
        let btw = UiToCore::BtwAsk { ask_id: "b1".into(), text: "what is 6*7?".into() };
        let btw_reparsed: UiToCore = serde_json::from_str(&btw.to_line()).unwrap();
        assert_eq!(btw_reparsed, btw);
        assert!(btw.to_line().contains(r#""type":"BtwAsk""#));

        let cmd = UiToCore::Command {
            name: "restore".into(),
            args: "path/to/log".into(),
        };
        assert_eq!(
            cmd.to_line(),
            r#"{"type":"Command","name":"restore","args":"path/to/log"}"#
        );

        let shutdown = UiToCore::Shutdown;
        assert_eq!(shutdown.to_line(), r#"{"type":"Shutdown"}"#);

        let intervene = UiToCore::Intervene {
            text: "stop that".into(),
        };
        assert_eq!(
            intervene.to_line(),
            r#"{"type":"Intervene","text":"stop that"}"#
        );

        let answer = UiToCore::Answer {
            ask_id: "a1".into(),
            option_id: Some("0".into()),
            text: None,
        };
        let answer_reparsed: UiToCore = serde_json::from_str(&answer.to_line()).unwrap();
        assert_eq!(answer_reparsed, answer);
    }
}
