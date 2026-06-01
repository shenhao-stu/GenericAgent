# R2 — GA core ↔ tui_v4 bridge: text protocol + model identity

Recon round 4, deliverable 2. Documents (1) the EXACT JSONL frame sequence and
the text shapes carried inside the assistant stream during one live turn, and
(2) the model-identity bug (`llm=MixinSession`, `model=MixinSession·codex-pro`)
plus the precise fix on both the Python (`ga_bridge.py` / `agentmain` /
`llmcore.MixinSession`) and Rust (`components/text.rs` + `header.rs`) sides.

The boundary the TUI talks to is **`frontends/tuiapp_v4/scripts/ga_bridge.py`**
(NOT `scripts/ga_bridge.py`). It wraps `agentmain.GenericAgent`, isolates the
core's stdout onto stderr (`ga_bridge.py:51-89`), and speaks one JSON object per
line. The Rust serde mirror is `src/bridge/protocol.rs`; transport is
`src/bridge/mod.rs`.

> NOTE on logs: `temp/model_responses/*.txt` are **LLM I/O logs** (`=== Prompt
> ===` / `=== Response ===` with Python `dict` reprs of message blocks). They are
> NOT the render stream. The render stream is the JSONL `MessageDelta.text`
> chunks below; `continue_cmd.extract_ui_messages()` (`continue_cmd.py:514-553`)
> *reconstructs* the same string shape FROM those logs for `/continue` replay,
> which is why the shapes coincide — but the live source is `agent_loop.py`'s
> yields, not the log files.

---

## Part 1 — The text protocol for ONE live assistant turn

### 1.1 The frame envelope (what `_drain` emits, ga_bridge.py:534-596)

A `Submit{text}` (UiToCore) lands on `handle_submit` → `agent.put_task` → a
per-task `display_queue`, drained by `Bridge._drain(dq)`. The bridge sets
`agent.inc_out = True` and `agent.verbose = False` (`ga_bridge.py:332-333`), so
the core streams **incremental deltas** through the queue's `{'next': ...}` /
`{'done': ...}` dicts. `_drain` maps one queue to exactly this frame sequence:

```
MessageBegin{mid:"m7", role:"assistant"}      # ga_bridge.py:543 — one per turn
Status{model, tokens, input_tokens, ...}      # :546   running token snapshot (/cost)
MessageDelta{mid:"m7", text:"<chunk>"}         # :582   N of these, the live stream
Status{...}                                    # :571   throttled refresh, ≥1s apart
MessageDelta{mid:"m7", text:"<chunk>"}
... (repeat) ...
Status{...}                                    # :594   FINAL token totals, pre-End
MessageEnd{mid:"m7", reason:"stop"}            # :595   reason ∈ stop|abort|error
```

Key invariants:
- **`inc_out=True` means `'done'` is NOT re-emitted as a delta** (the deltas
  already carried the full text; `_drain` breaks on `'done'` — `ga_bridge.py:574-578`).
  This is the same anti-duplication rule as the ACP bridge.
- **`reason`** is `"abort"` if `agent.stop_sig` was set mid-turn (`:589`),
  `"error"` if `_drain` caught an exception or the 600 s idle budget elapsed
  (`:560-562`), else `"stop"`. (`length`/`end` exist in the serde enum but the
  bridge does not emit them.)
- Every text field is capped at `MAX_TEXT = 64 KiB` via `_bound` (`ga_bridge.py:161`).
- `mid` is `"m%d"` from a monotonic counter (`_new_mid`, `:308`).

### 1.2 The text SHAPE inside `MessageDelta.text` — what GA actually yields

The concatenation of all `MessageDelta.text` chunks for one turn reproduces the
**non-verbose** `agent_loop.agent_runner_loop` yield stream. With `verbose=False`
the load-bearing emitters are:

**(a) The Turn-N marker** — `agent_loop.py:60-65`. The turnstring is built as:
```python
turn += 1; turnstr = f'LLM Running (Turn {turn}) ...'   # line 61 (verbose form)
if handler.parent.task_dir: turnstr = f'Turn {turn} ...'  # line 62 (BARE form)
if verbose: turnstr = f'**{turnstr}**'                     # line 63 (NOT taken here)
...
yield f"\n\n{turnstr}\n\n"                                 # line 65
```
The bridge sets `agent.task_dir` (`ga_bridge.py:336`), so **line 62 fires** and
the wire carries the **bare** form. The exact bytes are:
```
\n\nTurn 1 ...\n\n
```
(`Turn `, the number, a space, three dots — at the START of a line.) The Rust
boundary scanner `find_turn_line` (`chip.rs:236-255`) matches BOTH this bare form
and the legacy bold `**Turn N ...**`, but ONLY at a line start (so "Turn left"
in prose is never a false boundary).

**(b) The `<summary>` tag block** — emitted by the model itself inside the delta
text (the THINKING_PROMPT instructs a one-line `<summary>...</summary>` snapshot;
`llmcore.py:990-998`). It is plain text in the stream; the Rust side treats a
`<summary>` open as a structural boundary (`chip.rs:218-230`,
`next_marker_boundary`). Shape: `<summary>new info + current intent</summary>`.

**(c) Compact tool calls** — `agent_loop.py:89` (the `verbose=False` branch):
```python
else: yield f"🛠️ {tool_name}({_compact_tool_args(tool_name, args)})\n\n\n"
```
So one tool call is **`🛠️ NAME(ARGS)`** on its own line, e.g.:
```
🛠️ web_scan({"tabs_only": true})
```
**Confirm the marker:** `src/render/chip.rs:30` pins
```rust
pub const TOOL_MARK: &str = "🛠️ ";   // U+1F6E0 U+FE0F + trailing space
```
This is the COMPACT form. The OLD verbose marker `🛠️ Tool: \`name\`` (the
`verbose=True` branch, `agent_loop.py:88`) is NOT what the bridge emits.
`parse_tool_calls` (`chip.rs:142-179`) splits the header at the first `(` and
takes the balanced-paren payload as `args` (`split_name_args`, tolerant of nested
`()` in JSON args).

**(d) Tool RESULT lines** — yielded by the handler's `dispatch` generator
(`agent_loop.py:91-97`); in non-verbose mode they stream raw (no ` ````` ` fences;
those are the verbose-only wrappers at `:95,:97`). GA tools emit status-prefixed
lines the Rust chip parser keys on (`chip.rs:89-134`, `tool_status`). Real shapes:
```
[Info] 3 tabs scanned · ok
[Action] ...
[Status] failed: bad args        # a leading [Status]/[Error]+fail/error/❌ → Error chip
```
The result body is "everything after the `🛠️` header line up to the next
structural boundary" (`🛠️` / `Turn N ...` / `<summary>` / EOT —
`chip.rs:157-160`). Status inference is MARKER-ONLY: a read result that merely
*contains* `❌` as content is still `Ok`; only a LEADING error marker, a
`[Status]/[Error] … fail|error|❌` line, or an inline `!!!Error:` flips it to
`Error` (`chip.rs:95-134`).

**(e) `[Info]` / `!!!Error` lines.** `[Info] ...` is an informational tool line
(→ `Ok`). **`!!!Error:`** is the GA stream/LLM error sentinel — `tool_status`
treats a leading or inline `!!!Error` as `ToolStatus::Error` (`chip.rs:98,111`),
and `MixinSession._raw_ask` uses the same `!!!Error:` / `[Error:` prefixes to
detect a failed member and fail over (`llmcore.py:961`). A failover prints
`[MixinSession] Using session (NAME)` / `[MixinSession] …retry N/M` to the core's
stdout → which `ga_bridge.py` has redirected to **stderr** → surfaced by Rust as
`BridgeEvent::Stderr` (`mod.rs:457-471`), routed to the `/verbose` audit, NOT the
transcript (`reducer.rs:13`, `app/session.rs:227`).

**(f) The next / done framing.** There is no separate "next" frame on the wire —
the turn-to-turn boundary is purely the in-band `Turn N ...` marker (a) inside
the delta stream. The END of the whole turn is the queue's `{'done': ...}` dict,
which `_drain` converts to the single **`MessageEnd{mid, reason}`** frame
(`ga_bridge.py:574-578, 595`). The agent pushes exactly one `done` per task
(agentmain put_task contract), so each `Submit` ⇒ one `MessageBegin` … one
`MessageEnd`.

### 1.3 Out-of-band frames during a turn
- `AskUser{ask_id, question, options[], free_text}` (`ga_bridge.py:489-502`) —
  the human-intervention sentinel, surfaced via the turn-end hook
  (`_on_turn_end` / `_extract_ask_user`, `:452-487`), drained between deltas by
  `_flush_ask_user` so an interrupt isn't stuck behind a long turn. NOT a delta.
- `Error{message, code?, fatal}` (`emit_error`, `:301-306`) — every error path
  emits one; `fatal:true` ⇒ the child is dead. Never a silent hang (req. (e)).
- `Pong`, `LlmList`, `BtwAnswer`, `RewindResult` — replies to the matching
  UiToCore requests (`Ping`, `ListLlms`, `BtwAsk`, `Rewind`); see protocol.rs.

### 1.4 Handshake (turn 0)
`start_agent` (`ga_bridge.py:314-385`) emits **`Ready{version:"1", model:<name>}`**
on success, or `Error{...}` + a **degraded `Ready{version}` with NO model** on a
missing-LLM / import / init failure (`:327,:376,:382`). The Rust reader flips
`connected=true` only on a `Ready` frame (`mod.rs:436-438`). `model` here is
`self.llm_name()` = `agent.get_llm_name()` — see Part 2.

---

## Part 2 — The model-identity problem and its fix

### 2.1 What is reported today (the bug)

The bridge reports the model in TWO places, both via **`Bridge.llm_name()`**
(`ga_bridge.py:399-403`) which calls **`agent.get_llm_name()` with no args**:

- `Ready{..., "model": model}` — `ga_bridge.py:379, 384` (`model = self.llm_name()`).
- `Status{"model": self.llm_name(), ...}` — `_status_payload`, `ga_bridge.py:417`.

`agentmain.GenericAgent.get_llm_name` (`agentmain.py:96-100`):
```python
def get_llm_name(self, b=None, model=False):
    b = self.llmclient if b is None else b
    if isinstance(b, dict): return 'BADCONFIG_MIXIN'
    if model: return b.backend.model.lower()                       # underlying model
    return f"{type(b.backend).__name__}/{b.backend.name}"          # <-- what the bridge sends
```
With the live `mixin_config` (`mykey.py:74`), the default branch returns (verified
empirically):
```
MixinSession/codex-pro|getoken_20x|getoken|anyrouter_chenyt|tabcode_claude|tabcode_kiro|kiro
```
i.e. **`<SessionType>/<pipe-joined member CONFIG NAMES>`** — because
`MixinSession.name` (`llmcore.py:938`) is `'|'.join(s.name for s in self._sessions)`,
and each member's `.name` is its mykey config `name` (`BaseSession.name`,
`llmcore.py:528` = `cfg.get('name', self.model)`), NOT its model.

The Rust header (`components/cockpit/header.rs:26-27`) then derives two fields
from that one string:
```rust
let llm   = llm_channel(app.model.as_deref());                       // text.rs:103
let model = truncate_model(app.model.as_deref().unwrap_or("—"), 22); // text.rs:34
```
- `llm_channel` returns the SessionType prefix verbatim → **`"MixinSession"`**
  (the ROUTER, text.rs:108-116). 
- `truncate_model` keeps `prefix·<first-pipe-member>` → **`"MixinSession·codex-pro"`**
  (text.rs:42-52). The "codex-pro" here is the first config NAME, NOT a model.

So the header shows `llm=MixinSession` + `model=MixinSession·codex-pro`. The REAL
underlying model (`gpt-5.5` for the active `codex-pro`, or `claude-opus-4-8[1m]`
for a claude member) is **never on the wire** — `get_llm_name(model=True)` is
never called by the bridge.

### 2.2 What MixinSession already exposes (verified, llmcore.py)

`MixinSession` tracks the ACTIVE member via `self._cur_idx` (failover index;
`llmcore.py:943,956-958,975-978`). Both wanted values are already directly
available on the active member:

| Want | Source | live value |
|---|---|---|
| active config NAME (`codex-pro`) | `mixin._sessions[mixin._cur_idx].name` | `codex-pro` |
| real underlying model (`gpt-5.5`) | `mixin.model` (property, `llmcore.py:952-955`) → `_sessions[_cur_idx].model` | `gpt-5.5` |

`mixin.model` is a `@property` returning `getattr(self._sessions[self._cur_idx],
'model', None)` — i.e. it ALREADY follows the active member after a failover.
`get_llm_name(model=True)` (`agentmain.py:99`) returns `b.backend.model.lower()`
= `mixin.model.lower()` = the active member's model. Empirically:
```
get_llm_name()            -> 'MixinSession/codex-pro|getoken_20x|getoken|...|kiro'
get_llm_name(model=True)  -> 'gpt-5.5'
mixin._cur_idx            -> 0
mixin._sessions[0].name   -> 'codex-pro'      # active config name
mixin._sessions[0].model  -> 'gpt-5.5'        # active underlying model
member names              -> ['codex-pro','getoken_20x','getoken','anyrouter_chenyt','tabcode_claude','tabcode_kiro','kiro']
member models             -> ['gpt-5.5','claude-opus-4-8[1m]','claude-opus-4-8[1m]','claude-opus-4-8[1m]','claude-opus-4-6','claude-opus-4-6','claude-opus-4.7']
```
For a NON-mixin backend, `backend.name`/`backend.model` are scalars (no pipes), so
the same two accessors degrade correctly (name==model when no `name` configured).

CAVEAT — the `[1m]` tag: `get_llm_name(model=True)` does NOT strip the
`claude-opus-4-8[1m]` context-window tag (it is `llmcore.py:672-673`, only the
OUTBOUND wire model is stripped). For a clean header the bridge should strip a
trailing `[...]` from the reported model (e.g. `claude-opus-4-8[1m]` →
`claude-opus-4-8`).

### 2.3 EXACTLY how the bridge should report

Add an active-name + underlying-model resolver to the bridge and put BOTH on the
wire as two NEW, additive fields. Do NOT change the existing `model` field's
meaning blindly — but since `truncate_model` already only shows the primary, the
cleanest move is:

**Python — `ga_bridge.py`.** Add a helper next to `llm_name()` (`:399`):
```python
def llm_identity(self):
    """(active config NAME, real underlying model) for the header.
    Falls back to ('?','?') so the wire never breaks."""
    agent = self._agent
    try:
        b = getattr(agent.llmclient, "backend", None)
        # MixinSession: active member by _cur_idx; scalar backend: itself.
        sub = b._sessions[b._cur_idx] if hasattr(b, "_sessions") else b
        name  = _bound(getattr(sub, "name", "") or "", 128)
        model = _bound(getattr(sub, "model", "") or "", 128)
        model = model.split("[", 1)[0].strip()      # drop the [1m] tag
        return (name or "?", model or name or "?")
    except Exception:
        return ("?", "?")
```
Then emit both in `Ready` and `Status` (the two report sites):
- `Ready` (`:384`): `{"type":"Ready","version":PROTOCOL_VERSION, "model": model,
  "llm": name, "model_real": real}` where `name, real = self.llm_identity()` and
  keep `model = self.llm_name()` for backward-compat (older Rust still parses).
- `Status` (`_status_payload`, `:417`): add `frame["llm"] = name;
  frame["model_real"] = real` alongside the existing `frame["model"]`.

Rationale: keeping `model` (the full `SessionType/chain` string) untouched means a
stale Rust binary keeps working; the NEW `llm`/`model_real` fields carry the exact
two values the user wants. (`get_llm_name(model=True)` is the alternative source
for `model_real`, but reading `_sessions[_cur_idx]` gives BOTH name and model from
the SAME member in one shot, immune to a `_cur_idx` race between two calls.)

### 2.4 EXACTLY how the Rust side should parse

**`src/bridge/protocol.rs`** — add two optional fields to BOTH `Ready` and
`Status` (additive, serde-default `None`, so old bridges still parse):
```rust
Ready {
    #[serde(default, skip_serializing_if = "Option::is_none")] version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")] model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")] llm: Option<String>,        // NEW: active config name
    #[serde(default, skip_serializing_if = "Option::is_none")] model_real: Option<String>, // NEW: underlying model
},
// ...same two fields added to Status { ... } (protocol.rs:69-88)
```

**`src/app/reducer.rs`** — store them. `on_ready` (`reducer.rs:136-139`) and the
`Status` handler currently only keep `model`. Carry `llm`/`model_real` onto new
`AppState` fields (e.g. `app.llm_name: Option<String>`, `app.model_real:
Option<String>`), updated on BOTH `Ready` and `Status` so a mid-turn failover
(which re-emits `Status`) live-updates the header. The `FrameSink::on_ready`
signature (`reducer.rs:28`) must widen to take the two extra `Option<String>`s.

**`src/components/cockpit/header.rs:26-27`** — when the new fields are present,
use them DIRECTLY instead of parsing the chain:
```rust
let llm = app.llm_name.as_deref()
    .filter(|s| !s.is_empty())
    .unwrap_or_else(|| llm_channel(app.model.as_deref()));  // fallback: old behavior
let model = match app.model_real.as_deref().filter(|s| !s.is_empty()) {
    Some(m) => truncate_model(m, MODEL_LABEL_CAP),          // a bare model → passes through
    None    => truncate_model(app.model.as_deref().unwrap_or("—"), MODEL_LABEL_CAP),
};
```
Result with the live config: **`llm codex-pro   model gpt-5.5`** (and after a
spring-back/failover to e.g. `getoken_20x`, the next `Status` flips it to `llm
getoken_20x  model claude-opus-4-8`). `truncate_model` on a bare model string
(no `/`, no `|`) returns it unchanged (text.rs:42-52, the "plain single model"
path), so `gpt-5.5` and `claude-opus-4-8` render as-is, capped to 22 cells.

Note: `llm_channel`/`truncate_model` and their tests (`text.rs:275-345`) stay as
the FALLBACK path for a stale bridge that only sends the chain — do not delete
them; they remain correct for the legacy single-`model` wire.

### 2.5 File:line index (the load-bearing cites)
- Bridge report sites: `ga_bridge.py:379,384` (Ready), `:417` (Status `_status_payload`), `:399-403` (`llm_name`).
- Name source: `agentmain.py:96-100` (`get_llm_name`), `llmcore.py:938` (`MixinSession.name` = pipe-join), `:528` (`BaseSession.name`).
- Underlying model: `llmcore.py:952-955` (`MixinSession.model` property → `_sessions[_cur_idx].model`), `:522` (`BaseSession.model`), `agentmain.py:99` (`model=True` branch). Active index: `llmcore.py:943,956-958`.
- Config: `mykey.py:74` (`mixin_config.llm_nos`), `:294` (`codex-pro` → `gpt-5.5`).
- Rust parse: `protocol.rs:34-39` (Ready), `:69-88` (Status); `reducer.rs:28,136-139` (on_ready); `header.rs:26-27`; `text.rs:34-66` (`truncate_model`), `:103-120` (`llm_channel`).
- Turn marker / tool marker (Part 1): `agent_loop.py:61-65,88-89`; `chip.rs:30` (`TOOL_MARK`), `:236-255` (`find_turn_line`), `:142-179` (`parse_tool_calls`), `:95-134` (`tool_status`).
