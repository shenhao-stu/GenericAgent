#!/usr/bin/env python3
"""scripts/ga_bridge.py — stdio JSONL bridge wrapping agentmain.GenericAgent.

This is the GA-core child for tui_v4's chat plane. It speaks the JSONL line
protocol defined in src/bridge/protocol.ts (the AUTHORITATIVE schema):

  UiToCore  (stdin, one JSON object per line):
    Submit{text, images?}   Abort/Cancel{mid?}   Intervene{text}
    SwitchLlm{n}   ListLlms   Ping{nonce}   Shutdown   Command{name,args}   Answer{...}
  CoreToUi  (stdout, one JSON object per line):
    Ready{version,model}   MessageBegin{mid,role}   MessageDelta{mid,text}
    MessageEnd{mid,reason}   AskUser{ask_id,question,options,free_text}
    Status{model,...}   Pong{nonce}   LlmList{items}   Error{message,code,fatal}

It mirrors the proven integration in frontends/tui_v3.py (AgentBridge) and
frontends/genericagent_acp_bridge.py:

  * put_task -> per-task display_queue; drain {'next'|'done'} dicts.
  * ask_user surfaces via the GenericAgentHandler turn-end hook (NOT the queue).
  * mid-run intervene / abort go through the file signals in `task_dir`
    (`_intervene` / `_stop`) the way ga.py consumes them at turn boundaries.
  * exit-boundary replay: a queued intervene consumed on a turn that EXITs is
    re-submitted via put_task so the user's words aren't streamed headlessly.

ROBUSTNESS (the requirements that made the prior build "incomplete"):
  (a) ISOLATE CORE STDOUT — the protocol owns the real stdout fd; agentmain's
      print()/logs are redirected to stderr so they can never corrupt the JSONL.
  (b) read/decode stdin with errors='replace' (Chinese-Windows GBK safety).
  (c) BOUND every text field we emit (MAX_TEXT).
  (d) never hang on Shutdown — the read loop exits and the process returns.
  (e) emit a frame on EVERY error path (Error{...}); never a silent hang.
  Also: a vendored `V=1` capabilities line on stderr at startup.

Usage:  python scripts/ga_bridge.py [--llm-no N]
"""

import io
import json
import os
import sys

# ---------------------------------------------------------------------------
# (a) STDOUT ISOLATION — do this BEFORE importing agentmain.
#
# agentmain reconfigures stdout at import time and its submodules may print()
# during init / tool calls.  We dup the real stdout fd for our exclusive
# protocol channel, mark it non-inheritable, then point the process's stdout
# (fd 1) at stderr so every stray write lands on stderr instead of the JSONL.
# This is the same technique genericagent_acp_bridge.py uses for its JSON-RPC.
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    import msvcrt

    _proto_fd = os.dup(sys.__stdout__.fileno())
    msvcrt.setmode(_proto_fd, os.O_BINARY)
    _PROTO_OUT = os.fdopen(_proto_fd, "wb", buffering=0)
    msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
    os.set_inheritable(_proto_fd, False)
    os.dup2(sys.stderr.fileno(), sys.__stdout__.fileno())
else:
    _proto_fd = os.dup(sys.__stdout__.fileno())
    os.set_inheritable(_proto_fd, False)
    _PROTO_OUT = os.fdopen(_proto_fd, "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.__stdout__.fileno())


class _StdoutToStderrRouter(io.TextIOBase):
    """Route any text-mode `print()` to stderr so it never hits the protocol."""

    def writable(self):
        return True

    def write(self, s):
        if s:
            try:
                sys.stderr.write(s)
                sys.stderr.flush()
            except Exception:
                pass
        return len(s) if s else 0

    def flush(self):
        try:
            sys.stderr.flush()
        except Exception:
            pass


sys.stdout = _StdoutToStderrRouter()

import argparse
import queue
import threading
import time
import traceback
import uuid

# Make the GA repo root importable. A fixed "four dirnames up" assumption
# BREAKS when this script is COPIED elsewhere (e.g. target/release/ga_bridge.py
# in a packaged build) — the depth no longer matches the scripts/ layout, so
# `import agentmain` fails with ModuleNotFoundError. Locate the root ROBUSTLY:
# honor GENERICAGENT_ROOT, else walk up from this file's dir (then the cwd) to
# the first ancestor that actually contains agentmain.py.
def _find_repo_root():
    cand = os.environ.get("GENERICAGENT_ROOT")
    if cand and os.path.isfile(os.path.join(cand, "agentmain.py")):
        return os.path.abspath(cand)
    for start in (os.path.dirname(os.path.abspath(__file__)), os.path.abspath(os.getcwd())):
        d = start
        for _ in range(10):
            if os.path.isfile(os.path.join(d, "agentmain.py")):
                return d
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    # Last resort: the legacy four-levels-up guess (canonical scripts/ layout).
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )


_REPO_ROOT = _find_repo_root()
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# continue_cmd lives in frontends/; put it on the path too so the restore handler
# (and continue_cmd's own `from chatapp_common import ...` degraded path) resolve.
_FRONTENDS_DIR = os.path.join(_REPO_ROOT, "frontends")
if _FRONTENDS_DIR not in sys.path:
    sys.path.insert(0, _FRONTENDS_DIR)


def _import_continue_cmd():
    """Import continue_cmd robustly across the two layouts in the tree
    (`import continue_cmd` when frontends/ is on the path, else
    `from frontends import continue_cmd`)."""
    try:
        import continue_cmd  # frontends/ is on sys.path (see above)

        return continue_cmd
    except Exception:
        from frontends import continue_cmd  # repo-root layout

        return continue_cmd

PROTOCOL_VERSION = "1"
# Bound every text field we emit so one runaway buffer can't blow up the UI or
# the line reader on the other side.
MAX_TEXT = 1 << 16  # 64 KiB

_HOOK_KEY = "_ga_bridge_ask_user"


def _eprint(*args):
    try:
        print(*args, file=sys.stderr, flush=True)
    except Exception:
        pass


def _bound(s, limit=MAX_TEXT):
    """(c) Coerce to str and cap length; keep a marker so truncation is visible."""
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    if len(s) > limit:
        return s[:limit] + "\n...[truncated %d chars]" % (len(s) - limit)
    return s


def _deicon_restore(msg, n):
    """De-iconify continue_cmd.restore's banner (Q10: "no ✅").

    continue_cmd returns strings like "✅ 已恢复 N 轮完整对话（name）\n(…)"; the user
    explicitly does not want the leading status glyph. Strip ONE leading
    ✅/⚠️/❌ (plus surrounding space) and keep the informative remainder. `n` (the
    replayed-bubble count) is accepted for call-site symmetry / future copy; the
    banner text already carries the round count, so it is not interpolated here.
    Bounded like every other emitted field. Never edits GA-core continue_cmd.py
    (shared by v2/v3/st/tg/dc/qt) — the strip lives in the bridge.
    """
    del n  # the banner already carries the count; kept for call symmetry.
    s = (msg or "").lstrip()
    for g in ("✅", "⚠️", "❌"):
        if s.startswith(g):
            s = s[len(g):].lstrip()
            break
    return _bound(s)


def _is_user_msg(msg):
    """True if a backend.history entry is a USER message (a "real turn" anchor).
    Tolerant of dict / object shapes (`.role` attr or `['role']` key)."""
    role = None
    if isinstance(msg, dict):
        role = msg.get("role")
    else:
        role = getattr(msg, "role", None)
    return role == "user"


def _rewind_cut_index(history, n):
    """Compute the truncation `(cut_index, dropped_turns)` for rewinding `history`
    by `n` REAL (user) turns. PURE — unit-testable, and the load-bearing rewind
    math: gather the indices of user messages; the cut is at the (n-th from last)
    user message so it and everything after it are dropped. `dropped` is how many
    user turns were actually removed (clamped to what exists). `n<=0` is a no-op.

    Examples (U=user, A=assistant):
      [U,A,U,A,U,A], n=1 -> cut=4, dropped=1   (drop the last U,A)
      [U,A,U,A,U,A], n=2 -> cut=2, dropped=2
      [U,A,U,A],     n=9 -> cut=0, dropped=2   (clamp to all turns)
    """
    if n <= 0:
        return len(history), 0
    user_positions = [i for i, m in enumerate(history) if _is_user_msg(m)]
    if not user_positions:
        return len(history), 0
    drop = min(n, len(user_positions))
    cut = user_positions[len(user_positions) - drop]
    return cut, drop


def _coerce_answer_text(out):
    """Best-effort extraction of answer text from a variety of backend return
    shapes (a plain str, a {'content': ...} / {'text': ...} dict, a list of text
    blocks, or an OpenAI-ish {'choices':[{'message':{'content':...}}]})."""
    if out is None:
        return ""
    if isinstance(out, str):
        return out.strip()
    if isinstance(out, dict):
        if isinstance(out.get("text"), str):
            return out["text"].strip()
        content = out.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for blk in content:
                if isinstance(blk, dict) and isinstance(blk.get("text"), str):
                    parts.append(blk["text"])
            if parts:
                return "\n".join(parts).strip()
        choices = out.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"].strip()
    if isinstance(out, list):
        parts = []
        for blk in out:
            if isinstance(blk, dict) and isinstance(blk.get("text"), str):
                parts.append(blk["text"])
        if parts:
            return "\n".join(parts).strip()
    return str(out).strip()


class Bridge:
    def __init__(self, llm_no=0):
        self.llm_no = llm_no
        self._proto_out = _PROTO_OUT
        self._write_lock = threading.Lock()
        self._mid_lock = threading.Lock()
        self._mid_counter = 0
        self._agent = None
        self._runner = None
        self._init_error = None
        # Token/cost capture (cost_tracker.install monkey-patches llmcore). Set in
        # start_agent; `None` until then / if the import fails (degrades cleanly).
        self._cost_tracker = None
        # Monotonic ms of the last Status emit, to throttle the mid-turn token
        # refresh (tokens only change at LLM-response boundaries, not per delta).
        self._last_status_emit = 0.0
        # Mirrors AgentBridge._intervene_pending / _replay_dq (tui_v3.py:1334).
        self._intervene_pending = []
        self._intervene_lk = threading.Lock()
        self._replay_dq = None
        self._ask_user_queue = queue.Queue()
        # Serialize turns: only one display_queue drained at a time, so deltas
        # for distinct mids never interleave on the wire.
        self._turn_lock = threading.Lock()

    # -- frame emission ----------------------------------------------------
    def emit(self, frame):
        """(a) Write ONE CoreToUi frame to the isolated protocol fd."""
        try:
            line = json.dumps(frame, ensure_ascii=False, separators=(",", ":"))
        except Exception as exc:
            line = json.dumps(
                {"type": "Error", "message": "serialize failed: %s" % exc, "fatal": False}
            )
        raw = (line + "\n").encode("utf-8", errors="replace")
        try:
            with self._write_lock:
                self._proto_out.write(raw)
                self._proto_out.flush()
        except Exception as exc:
            _eprint("[ga_bridge] WRITE FAILED: %s: %s" % (type(exc).__name__, exc))

    def emit_error(self, message, code=None, fatal=False):
        """(e) Every error path emits a frame — never a silent hang."""
        frame = {"type": "Error", "message": _bound(str(message), 4096), "fatal": bool(fatal)}
        if code:
            frame["code"] = str(code)
        self.emit(frame)

    def _new_mid(self):
        with self._mid_lock:
            self._mid_counter += 1
            return "m%d" % self._mid_counter

    # -- agent lifecycle ---------------------------------------------------
    def start_agent(self):
        """Construct the GA core, install the ask_user hook, start its thread.

        Emits Ready (with the model name if available) on success, or a
        non-fatal Error + a degraded Ready if the LLM isn't configured — so the
        UI always gets a startup frame (requirement (e)).
        """
        try:
            from agentmain import GenericAgent
        except Exception as exc:
            _eprint("[ga_bridge] import failed:\n" + traceback.format_exc())
            self._init_error = "%s: %s" % (type(exc).__name__, exc)
            self.emit_error("GA core import failed: " + self._init_error, code="import", fatal=True)
            self.emit({"type": "Ready", "version": PROTOCOL_VERSION})
            return False

        try:
            agent = GenericAgent()
            agent.inc_out = True       # 'next' carries incremental deltas (recon §1.1)
            agent.verbose = False      # keep the core quiet; deltas come via the queue
            # PID-scoped task dir so the `_intervene`/`_stop`/`_keyinfo` file
            # signals don't collide across concurrent bridges (tui_v3.py:1325).
            agent.task_dir = os.path.join(
                _REPO_ROOT, "temp", "_tui_v4_%d_%s" % (os.getpid(), uuid.uuid4().hex[:8])
            )
            try:
                os.makedirs(agent.task_dir, exist_ok=True)
            except Exception:
                pass
            if self.llm_no:
                try:
                    agent.next_llm(self.llm_no)
                except Exception as exc:
                    _eprint("[ga_bridge] next_llm(%s) failed: %s" % (self.llm_no, exc))
            self._agent = agent
            self._install_hook()
            # Token/cost capture: monkey-patch llmcore._record_usage + print so
            # every LLM call accrues into per-thread TokenStats. Idempotent
            # (_INSTALLED guard); we SUM all_trackers() at emit time because
            # _record_usage fires on the agent/mixin thread(s), not this one.
            # frontends/ is already on sys.path (the bridge inserts it at import).
            try:
                import cost_tracker
            except Exception:
                try:
                    from frontends import cost_tracker
                except Exception:
                    cost_tracker = None
            self._cost_tracker = cost_tracker
            if cost_tracker is not None:
                try:
                    cost_tracker.install()
                except Exception as exc:
                    _eprint("[ga_bridge] cost_tracker.install failed: %s" % exc)
            self._runner = threading.Thread(
                target=self._run_safe, name="ga-bridge-agent", daemon=True
            )
            self._runner.start()
        except Exception as exc:
            _eprint("[ga_bridge] construct failed:\n" + traceback.format_exc())
            self._init_error = "%s: %s" % (type(exc).__name__, exc)
            self.emit_error("GA core init failed: " + self._init_error, code="init", fatal=True)
            self.emit({"type": "Ready", "version": PROTOCOL_VERSION})
            return False

        model = self.llm_name()
        if not getattr(agent, "llmclient", None):
            self.emit_error("no LLM configured (check mykey.py)", code="no_llm", fatal=False)
            self.emit({"type": "Ready", "version": PROTOCOL_VERSION})
            return True
        self.emit({"type": "Ready", "version": PROTOCOL_VERSION, "model": model})
        return True

    def _run_safe(self):
        try:
            self._agent.run()
        except Exception:
            _eprint("[ga_bridge] agent.run() crashed:\n" + traceback.format_exc())
            self.emit_error("agent loop crashed (see stderr)", code="agent_crash", fatal=True)

    def _install_hook(self):
        if not hasattr(self._agent, "_turn_end_hooks"):
            self._agent._turn_end_hooks = {}
        self._agent._turn_end_hooks[_HOOK_KEY] = self._on_turn_end

    def llm_name(self):
        try:
            return _bound(self._agent.get_llm_name(), 256)
        except Exception:
            return "?"

    def _status_payload(self):
        """Build a Status frame: model + cumulative token totals + context %.

        Tokens are SUMMED across cost_tracker.all_trackers() (never keyed on a
        single thread — _record_usage fires on the agent/mixin thread(s), so the
        proven tui_v3 approach is to sum). `input_tokens` is the full input side
        (input + cache_create + cache_read); `cache_tokens` is the cache-read hits;
        `tokens` is the grand total. `last_input`/`last_output` are per-call sizes
        for the spinner's ↑/↓ readout, taken from the freshest tracker. Context %
        compares the backend's char-history against context_win*3 (the trim unit).
        Best-effort: any failure degrades to a model-only frame (wire never breaks).
        """
        frame = {"type": "Status", "model": self.llm_name()}
        ct = self._cost_tracker
        if ct is None:
            return frame
        try:
            tot_in = tot_out = tot_cc = tot_cr = 0
            last_in = last_out = 0
            best_ts = -1.0
            for t in ct.all_trackers().values():
                tot_in += t.input
                tot_out += t.output
                tot_cc += t.cache_create
                tot_cr += t.cache_read
                if (t.last_input or t.last_output) and t.started_at >= best_ts:
                    best_ts = t.started_at
                    last_in, last_out = t.last_input, t.last_output
            input_tokens = tot_in + tot_cc + tot_cr  # full input side (incl. cache)
            total = input_tokens + tot_out
            frame["input_tokens"] = int(input_tokens)
            frame["output_tokens"] = int(tot_out)
            frame["cache_tokens"] = int(tot_cr)  # cache-read hits
            frame["last_input"] = int(last_in)
            frame["last_output"] = int(last_out)
            frame["tokens"] = int(total)
            be = getattr(getattr(self._agent, "llmclient", None), "backend", None)
            if be is not None:
                cap = ct.context_window_chars(be)
                used = ct.current_input_chars(be)
                if cap > 0:
                    frame["context_percent"] = float(min(100.0, used * 100.0 / cap))
        except Exception as exc:
            _eprint("[ga_bridge] status payload failed: %s" % exc)
        return frame

    # -- turn-end hook: ask_user + exit-boundary replay --------------------
    def _on_turn_end(self, ctx):
        """Runs inside the agent thread at each turn boundary (ga.py:579).

        Mirrors AgentBridge._on_turn_end (tui_v3.py:1391):
          * detect the ask_user INTERRUPT sentinel from `exit_reason`.
          * replay queued intervene text if THIS boundary was an exit (else the
            file we wrote was consumed but next_prompt is discarded).
        """
        ask = self._extract_ask_user(ctx)
        if ask is not None:
            self._ask_user_queue.put(ask)
        with self._intervene_lk:
            if not self._intervene_pending:
                return
            if (ctx or {}).get("exit_reason"):
                combined = "\n\n".join(self._intervene_pending)
                self._intervene_pending = []
                try:
                    self._replay_dq = self._agent.put_task(combined, source="user")
                except Exception as exc:
                    _eprint("[ga_bridge] replay put_task failed: %s" % exc)
            else:
                self._intervene_pending = []

    @staticmethod
    def _extract_ask_user(ctx):
        """Return (question, candidates) if the turn exited on the ask_user
        INTERRUPT sentinel (ga.py:97-100), else None (tui_v3.py:1298)."""
        er = (ctx or {}).get("exit_reason") or {}
        if er.get("result") != "EXITED":
            return None
        payload = er.get("data") or {}
        if payload.get("status") != "INTERRUPT" or payload.get("intent") != "HUMAN_INTERVENTION":
            return None
        data = payload.get("data") or {}
        return (data.get("question", ""), list(data.get("candidates", []) or []))

    def _emit_ask_user(self, question, candidates):
        options = []
        for i, c in enumerate(candidates or []):
            label = _bound(c if isinstance(c, str) else str(c), 512)
            options.append({"id": str(i), "label": label})
        self.emit(
            {
                "type": "AskUser",
                "ask_id": uuid.uuid4().hex[:8],
                "question": _bound(question, 8192),
                "options": options,
                "free_text": True,
            }
        )

    # -- Submit: drain one task's display_queue ----------------------------
    def handle_submit(self, text, images=None):
        if self._agent is None:
            self.emit_error("agent not initialized", code="no_agent", fatal=False)
            return
        text = _bound(text, MAX_TEXT)
        if not text and not images:
            self.emit_error("Submit had empty text", code="empty", fatal=False)
            return

        def worker():
            with self._turn_lock:
                try:
                    dq = self._agent.put_task(text, source="user", images=images or None)
                except Exception as exc:
                    _eprint("[ga_bridge] put_task failed:\n" + traceback.format_exc())
                    self.emit_error("put_task failed: %s" % exc, code="submit", fatal=False)
                    return
                self._drain(dq)
                # Drain any exit-boundary replay handed off by the turn-end hook
                # (tui_v3.py:1406 take_replay_dq) so the follow-up isn't headless.
                while True:
                    with self._intervene_lk:
                        rdq, self._replay_dq = self._replay_dq, None
                    if rdq is None:
                        break
                    self._drain(rdq)

        threading.Thread(target=worker, name="ga-bridge-turn", daemon=True).start()

    def _drain(self, dq):
        """Map one display_queue's {'next'|'done'} dicts -> Message* frames.

        Emits exactly one MessageBegin, N MessageDelta, one MessageEnd. Drains
        any pending AskUser between chunks so an interrupt isn't stuck behind a
        long turn. Never blocks forever: a missing 'done' is bounded by a hard
        idle timeout so the turn can't hang the bridge (requirement (e)).
        """
        mid = self._new_mid()
        self.emit({"type": "MessageBegin", "mid": mid, "role": "assistant"})
        # Status with the model + the running token/context snapshot (the /cost
        # wiring): the bridge used to send model-only here, so /cost was always 0.
        self.emit(self._status_payload())
        self._last_status_emit = time.monotonic()

        reason = "stop"
        # Bound total wait between chunks. The agent pushes a 'done' for every
        # task (agentmain.py:175); this timeout only fires if the core wedged.
        idle_budget = 600.0  # seconds with zero queue activity before we bail
        last_activity = time.monotonic()
        try:
            while True:
                self._flush_ask_user()
                try:
                    item = dq.get(timeout=0.25)
                except queue.Empty:
                    if time.monotonic() - last_activity > idle_budget:
                        reason = "error"
                        self.emit_error("turn timed out (no output)", code="timeout", fatal=False)
                        break
                    continue
                last_activity = time.monotonic()
                # Live token/context refresh for the spinner — throttled to 1s so
                # we don't spam the wire on every delta (tokens only move at
                # LLM-response boundaries anyway).
                if last_activity - self._last_status_emit >= 1.0:
                    self._last_status_emit = last_activity
                    self.emit(self._status_payload())
                if not isinstance(item, dict):
                    continue
                if "done" in item:
                    # 'done' text is the full final buffer; with inc_out=True we
                    # already streamed the deltas, so we DON'T re-emit it as a
                    # delta (avoids duplicate text — same rule as acp_bridge.py).
                    break
                if "next" in item:
                    delta = item.get("next")
                    if isinstance(delta, str) and delta:
                        self.emit({"type": "MessageDelta", "mid": mid, "text": _bound(delta)})
        except Exception as exc:
            _eprint("[ga_bridge] drain error:\n" + traceback.format_exc())
            reason = "error"
            self.emit_error("drain error: %s" % exc, code="drain", fatal=False)
        finally:
            # If the agent was aborted mid-turn, report that as the reason.
            if getattr(self._agent, "stop_sig", False):
                reason = "abort"
            # Flush the FINAL token totals before MessageEnd so /cost + the footer
            # settle on the real numbers (SSE output_tokens lands late, via the
            # llmcore [Output] print that cost_tracker patches).
            self.emit(self._status_payload())
            self.emit({"type": "MessageEnd", "mid": mid, "reason": reason})
            self._flush_ask_user()

    def _flush_ask_user(self):
        while True:
            try:
                question, candidates = self._ask_user_queue.get_nowait()
            except queue.Empty:
                return
            self._emit_ask_user(question, candidates)

    # -- intervene / abort via file signals --------------------------------
    def inject_intervene(self, text, track=True):
        """Append `text` to <task_dir>/_intervene (ga.py:576 consumes it at the
        next turn boundary). Mirrors AgentBridge.inject_intervene (tui_v3.py:1351).
        Returns False if the agent is idle (no turn to inject into)."""
        agent = self._agent
        td = getattr(agent, "task_dir", None) if agent else None
        if not td or not getattr(agent, "is_running", False):
            return False
        try:
            os.makedirs(td, exist_ok=True)
        except Exception:
            return False
        fp = os.path.join(td, "_intervene")
        try:
            sep = ""
            try:
                if os.path.getsize(fp) > 0:
                    sep = "\n\n"
            except OSError:
                pass
            with open(fp, "a", encoding="utf-8") as f:
                f.write(sep + text)
            if track:
                with self._intervene_lk:
                    self._intervene_pending.append(text)
            return True
        except Exception as exc:
            _eprint("[ga_bridge] inject_intervene failed: %s" % exc)
            return False

    def handle_intervene(self, text):
        text = _bound(text, MAX_TEXT)
        if not text:
            return
        if not self.inject_intervene(text, track=True):
            # Idle agent: nothing to intervene into — treat as a fresh submit so
            # the user's words are not lost.
            self.handle_submit(text)

    def handle_shell_note(self, text):
        """Record a TUI `!shell` exchange as agent CONTEXT without running a turn.

        The UI already executed the host command and rendered its output; this
        only seeds the agent so a follow-up question has the exchange in scope.
        If a turn is live we inject into <task_dir>/_intervene (consumed at the
        next boundary, ga.py:576); if idle we still stage the note in that file
        so the user's NEXT submit carries it — but we never call put_task, so a
        bare `!cmd` can never spend a model turn on its own. Fully best-effort:
        a failure is silent (the UI's output block is the user-visible record).
        """
        text = _bound(text, MAX_TEXT)
        if not text:
            return
        # Live turn → the tracked intervene path (so it's drained mid-turn).
        if self.inject_intervene(text, track=True):
            return
        # Idle → stage into _intervene for the next turn boundary, untracked
        # (there is no live turn whose delta-stream it should be reconciled with).
        agent = self._agent
        td = getattr(agent, "task_dir", None) if agent else None
        if not td:
            return
        try:
            os.makedirs(td, exist_ok=True)
            fp = os.path.join(td, "_intervene")
            sep = ""
            try:
                if os.path.getsize(fp) > 0:
                    sep = "\n\n"
            except OSError:
                pass
            with open(fp, "a", encoding="utf-8") as f:
                f.write(sep + text)
        except Exception as exc:
            _eprint("[ga_bridge] handle_shell_note failed: %s" % exc)

    def handle_abort(self):
        agent = self._agent
        if agent is None:
            return
        try:
            agent.abort()
        except Exception as exc:
            _eprint("[ga_bridge] abort failed: %s" % exc)
        # Also drop the `_stop` file signal: agent.run() consumes it each loop
        # iteration (agentmain.py:162), covering the window where is_running
        # flipped between our check and abort().
        td = getattr(agent, "task_dir", None)
        if td:
            try:
                os.makedirs(td, exist_ok=True)
                with open(os.path.join(td, "_stop"), "w", encoding="utf-8") as f:
                    f.write("1")
            except Exception:
                pass

    def handle_switch_llm(self, n):
        agent = self._agent
        if agent is None:
            self.emit_error("agent not initialized", code="no_agent", fatal=False)
            return
        try:
            # protocol n is 1-based into the configured list; next_llm is 0-based.
            agent.next_llm(int(n) - 1)
        except Exception as exc:
            self.emit_error("switch llm failed: %s" % exc, code="switch_llm", fatal=False)
            return
        # Recompute context % against the NEW backend's context_win + carry tokens.
        self.emit(self._status_payload())

    def handle_list_llms(self):
        """Emit the configured LLM list as a `LlmList{items}` frame (N3 / `/llm`
        picker). Mirrors `agent.list_llms()` → `[(idx, "SessionType/name",
        is_current)]` (agentmain.py:93-95) onto `{idx, name, current}` objects.
        Additive + best-effort: a missing/degraded agent yields an empty list
        (never a hang) so the picker can render "no models" cleanly."""
        items = []
        agent = self._agent
        if agent is not None and getattr(agent, "llmclient", None):
            try:
                for triple in agent.list_llms():
                    # triple = (idx, "SessionType/name", is_current)
                    idx = int(triple[0]) if len(triple) > 0 else 0
                    name = _bound(str(triple[1]) if len(triple) > 1 else "?", 256)
                    cur = bool(triple[2]) if len(triple) > 2 else False
                    items.append({"idx": idx, "name": name, "current": cur})
            except Exception as exc:
                _eprint("[ga_bridge] list_llms failed: %s" % exc)
                self.emit_error("list_llms failed: %s" % exc, code="list_llms", fatal=False)
        self.emit({"type": "LlmList", "items": items})

    # -- restore: load a prior transcript into this child's history --------
    def handle_restore(self, path):
        """Restore a prior model_responses_*.txt transcript into the agent's
        backend history via continue_cmd.restore (recon §4.3, continue_cmd.py:
        375-395). Used by the multi-session UI's `branch`/`/continue` so a new
        bridge child remembers a prior conversation.

        Replays the FULL prior conversation into the visible transcript (Q10 / v3
        parity, tui_v3.py:4129): after continue_cmd.restore writes backend.history,
        continue_cmd.extract_ui_messages(path) parses the log into [{role, content}]
        bubbles and each becomes one MessageBegin/Delta/End triple, THEN the
        de-iconified restore banner (no ✅, via _deicon_restore) as the final system
        line. Never hangs: every path emits frames or an Error. A failed extract
        degrades to just the banner (the model still remembers via backend.history).
        """
        if self._agent is None:
            self.emit_error("agent not initialized", code="no_agent", fatal=False)
            return
        path = _bound(str(path or ""), 4096).strip()
        if not path:
            self.emit_error("restore: empty path", code="restore", fatal=False)
            return
        # Serialize against live turns so we don't replace history mid-drain.
        with self._turn_lock:
            try:
                continue_cmd = _import_continue_cmd()
                msg, _is_full = continue_cmd.restore(self._agent, path)
            except Exception as exc:
                _eprint("[ga_bridge] restore failed:\n" + traceback.format_exc())
                self.emit_error("restore failed: %s" % exc, code="restore", fatal=False)
                return
            bubbles = []
            try:
                bubbles = continue_cmd.extract_ui_messages(path) or []
            except Exception:
                _eprint("[ga_bridge] extract_ui_messages failed:\n" + traceback.format_exc())
        # Replay the prior conversation as ordinary frames (the UI renders them
        # through the existing transcript path — no Rust change for the replay).
        for b in bubbles:
            content = _bound(str((b or {}).get("content") or ""))
            if not content.strip():
                continue
            role = "user" if (b or {}).get("role") == "user" else "assistant"
            bmid = self._new_mid()
            self.emit({"type": "MessageBegin", "mid": bmid, "role": role})
            self.emit({"type": "MessageDelta", "mid": bmid, "text": content})
            self.emit({"type": "MessageEnd", "mid": bmid, "reason": "stop"})
        # Finally, the de-iconified restore banner as a one-shot system line.
        mid = self._new_mid()
        self.emit({"type": "MessageBegin", "mid": mid, "role": "system"})
        self.emit({"type": "MessageDelta", "mid": mid, "text": _deicon_restore(msg, len(bubbles))})
        self.emit({"type": "MessageEnd", "mid": mid, "reason": "stop"})

    @staticmethod
    def _restore_path_from_args(args):
        """Accept the restore path as either a raw string (TS Command.args is a
        string) or a structured {"path": ...} object — robust to both shapes the
        spec mentions (Command{name:'restore', args:{path}})."""
        if isinstance(args, dict):
            return str(args.get("path", "") or "")
        return str(args or "")

    # -- rewind: truncate backend.history by N real turns ------------------
    def handle_rewind(self, n):
        """Truncate `llmclient.backend.history` by `n` REAL (user) turns (§7
        `/rewind`). A "real turn" is a user message + everything the model produced
        in response to it; rewinding `n` turns drops the last `n` user messages and
        all entries that follow each. Aborts any live turn first (so we don't mutate
        history mid-drain), then replies `RewindResult{dropped, remaining}` — a
        NON-streamed acknowledgment, so a rewind never spends a model turn.

        Robust + best-effort: a missing backend / empty history / bad `n` still
        emits a RewindResult (never a silent hang, requirement (e)). The
        truncation index is computed purely so it is easy to reason about: find the
        positions of the last `n` user messages; cut the history at the earliest of
        them.
        """
        try:
            n = max(0, int(n))
        except Exception:
            n = 0
        agent = self._agent
        if agent is None:
            self.emit_error("agent not initialized", code="no_agent", fatal=False)
            self.emit({"type": "RewindResult", "dropped": 0, "remaining": 0})
            return
        # Abort any in-flight turn so we replace history at a clean boundary.
        try:
            agent.abort()
        except Exception:
            pass
        with self._turn_lock:
            backend = getattr(getattr(agent, "llmclient", None), "backend", None)
            history = list(getattr(backend, "history", []) or [])
            total_turns = sum(1 for m in history if _is_user_msg(m))
            cut, dropped = _rewind_cut_index(history, n)
            if backend is not None and hasattr(backend, "history"):
                try:
                    backend.history = history[:cut]
                except Exception as exc:
                    _eprint("[ga_bridge] rewind set history failed: %s" % exc)
            remaining = max(0, total_turns - dropped)
        self.emit({"type": "RewindResult", "dropped": int(dropped), "remaining": int(remaining)})

    # -- btw: a background side-question that never blocks the main turn ----
    def handle_btw(self, ask_id, text):
        """Answer a `/btw` side-question on a WORKER thread so the MAIN turn is
        never blocked (§7 `/btw` "background thread; non-blocking; chat stays
        usable"). Runs a ONE-SHOT LLM completion against a throwaway message list
        (it does NOT touch the agent's task queue or backend.history, so the side
        question never pollutes the conversation), and emits `BtwAnswer{ask_id,
        text|error}` when done — which the UI routes into the ephemeral /btw card.

        Best-effort + never hangs: any failure emits a BtwAnswer with `error` set so
        the card shows a reason instead of spinning forever (requirement (e)).
        """
        ask_id = _bound(str(ask_id or ""), 128)
        text = _bound(str(text or ""), MAX_TEXT)
        if not text:
            self.emit({"type": "BtwAnswer", "ask_id": ask_id, "error": "empty question"})
            return
        agent = self._agent
        if agent is None or not getattr(agent, "llmclient", None):
            self.emit({"type": "BtwAnswer", "ask_id": ask_id, "error": "no LLM configured"})
            return

        def worker():
            try:
                answer = self._side_ask(text)
            except Exception as exc:
                _eprint("[ga_bridge] btw failed:\n" + traceback.format_exc())
                self.emit({"type": "BtwAnswer", "ask_id": ask_id, "error": "%s" % exc})
                return
            self.emit({"type": "BtwAnswer", "ask_id": ask_id, "text": _bound(answer)})

        threading.Thread(target=worker, name="ga-bridge-btw", daemon=True).start()

    def _side_ask(self, question):
        """One-shot side completion for `/btw`, ISOLATED from the agent's history.

        Tries, in order of preference: a dedicated `agent.side_ask` / `agent.ask`
        hook if the core exposes one; else a direct, stateless call on the LLM
        client's backend with a throwaway message list. Returns the answer text.
        Raises on a hard failure (the caller turns it into a BtwAnswer error)."""
        agent = self._agent
        # 1. A first-class side-ask hook, if the core grew one.
        for hook in ("side_ask", "ask_once", "ask"):
            fn = getattr(agent, hook, None)
            if callable(fn):
                try:
                    out = fn(question)
                    if isinstance(out, str) and out.strip():
                        return out
                except Exception:
                    pass  # fall through to the stateless backend path.
        # 2. Stateless backend completion on a THROWAWAY message list (no history
        #    mutation). We probe a few common client/backend method shapes so this
        #    works across GA backends without coupling to one signature.
        client = getattr(agent, "llmclient", None)
        backend = getattr(client, "backend", None)
        messages = [{"role": "user", "content": question}]
        for obj in (client, backend):
            if obj is None:
                continue
            for meth in ("complete", "chat", "completion", "once", "respond"):
                fn = getattr(obj, meth, None)
                if callable(fn):
                    try:
                        out = fn(messages)
                    except TypeError:
                        try:
                            out = fn(question)
                        except Exception:
                            continue
                    except Exception:
                        continue
                    text = _coerce_answer_text(out)
                    if text:
                        return text
        raise RuntimeError("no usable side-ask path on this LLM client")

    # -- dispatch ----------------------------------------------------------
    def handle(self, frame):
        """Dispatch one parsed UiToCore frame. Returns False to stop the loop."""
        if not isinstance(frame, dict):
            self.emit_error("frame is not an object", code="bad_frame", fatal=False)
            return True
        ftype = frame.get("type")
        try:
            if ftype == "Submit":
                self.handle_submit(frame.get("text", ""), self._coerce_images(frame.get("images")))
            elif ftype == "Command":
                name = str(frame.get("name", "")).strip()
                if name == "restore":
                    # Structured command: load a prior transcript into history
                    # (does NOT go through the LLM). args = path (str or {path}).
                    self.handle_restore(self._restore_path_from_args(frame.get("args")))
                elif name == "shell":
                    # The TUI's `!cmd` host-shell feature: the UI already ran the
                    # command + rendered the output block; it forwards the
                    # '[!shell] …' note here so the agent has the exchange as
                    # CONTEXT. This must NOT trigger a model turn (idle case) — we
                    # stash it into <task_dir>/_intervene so the next real turn
                    # (or a live one) picks it up at the boundary. Best-effort.
                    self.handle_shell_note(str(frame.get("args", "")))
                else:
                    # The core intercepts leading-slash commands itself
                    # (agentmain.py:114); forward as a normal submit of "/name args".
                    args = str(frame.get("args", "")).strip()
                    self.handle_submit(("/" + name + (" " + args if args else "")).strip())
            elif ftype == "Answer":
                # Answer to a prior AskUser: feed the chosen text back in. If a
                # turn is live, inject; else submit fresh.
                ans = frame.get("text")
                if not ans and frame.get("option_id") is not None:
                    ans = str(frame.get("option_id"))
                self.handle_intervene(_bound(ans or "", MAX_TEXT))
            elif ftype in ("Abort", "Cancel"):
                self.handle_abort()
            elif ftype == "Intervene":
                self.handle_intervene(frame.get("text", ""))
            elif ftype == "SwitchLlm":
                self.handle_switch_llm(frame.get("n", 1))
            elif ftype == "ListLlms":
                self.handle_list_llms()
            elif ftype == "Rewind":
                self.handle_rewind(frame.get("n", 1))
            elif ftype == "BtwAsk":
                self.handle_btw(str(frame.get("ask_id", "")), str(frame.get("text", "")))
            elif ftype == "Ping":
                self.emit({"type": "Pong", "nonce": _bound(str(frame.get("nonce", "")), 256)})
            elif ftype == "Shutdown":
                return False  # (d) exit the loop; never hang.
            elif ftype is None:
                self.emit_error("frame missing 'type'", code="bad_frame", fatal=False)
            else:
                self.emit_error("unknown frame type: %s" % ftype, code="unknown", fatal=False)
        except Exception as exc:
            _eprint("[ga_bridge] handler error:\n" + traceback.format_exc())
            self.emit_error("handler error: %s" % exc, code="handler", fatal=False)
        return True

    @staticmethod
    def _coerce_images(images):
        if not isinstance(images, list):
            return None
        out = []
        for img in images:
            if isinstance(img, dict) and "data" in img:
                out.append(img)
        return out or None

    # -- main read loop ----------------------------------------------------
    def serve(self):
        # (b) decode stdin with errors='replace' for Chinese-Windows GBK safety.
        if hasattr(sys.stdin, "buffer"):
            stream = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
        else:  # pragma: no cover
            stream = sys.stdin
        for raw in stream:
            line = raw.strip()
            if not line:
                continue
            try:
                frame = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                self.emit_error("invalid json line", code="parse", fatal=False)
                continue
            if not self.handle(frame):
                break
        _eprint("[ga_bridge] stopped")


def main():
    parser = argparse.ArgumentParser(description="GenericAgent JSONL bridge over stdio (tui_v4)")
    parser.add_argument("--llm-no", type=int, default=0, help="LLM index for GenericAgent")
    args = parser.parse_args()

    # Vendored capabilities line on stderr (requirement: V=1 line).
    _eprint("V=1 caps=submit,abort,intervene,switchllm,ping,askuser")

    bridge = Bridge(llm_no=args.llm_no)
    bridge.start_agent()  # always emits a Ready (or fatal Error + degraded Ready)
    try:
        bridge.serve()
    except KeyboardInterrupt:  # (d) clean exit on Ctrl+C, never hang.
        pass
    except Exception:
        _eprint("[ga_bridge] serve crashed:\n" + traceback.format_exc())
        bridge.emit_error("bridge serve crashed (see stderr)", code="serve", fatal=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
