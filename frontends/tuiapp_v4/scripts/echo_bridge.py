#!/usr/bin/env python3
"""tests/echo_bridge.py — a tiny stdlib-only protocol stub for tui_v4.

Speaks the SAME stdio JSONL protocol as scripts/ga_bridge.py (see
src/bridge/protocol.ts for the authoritative frame schema) but with ZERO
dependencies and ZERO GA-core coupling, so protocol round-trip tests are
deterministic and fast.

Behaviour (a strict subset of ga_bridge.py):
  - On start: emit Ready{version, model:"echo"} and a `V=1` capabilities line
    on stderr (vendored, mirrors ga_bridge.py for parity).
  - Submit{text}  -> MessageBegin{mid,role:"assistant"}
                     MessageDelta{mid,text:"echo: <text>"}
                     MessageEnd{mid,reason:"end"}
  - Ping{nonce}   -> Pong{nonce}
  - Intervene{}   -> ignored (no active turn to inject into; echo is synchronous)
  - Abort/Cancel  -> ignored (echo turns complete instantly)
  - SwitchLlm{n}  -> Status{model:"echo"}
  - ListLlms      -> LlmList{items:[{idx,name,current}]} (a fixed 2-entry stub)
  - Command{name} -> echoed like Submit ("echo: /<name> <args>")
  - Answer{}      -> echoed like Submit
  - Shutdown      -> exit the read loop cleanly (never hang).
  - Unknown / malformed line -> Error{...} (never a silent hang).

Reads stdin as a binary stream decoded with errors='replace' (so a stray GBK
byte on Chinese Windows can't kill the loop), and writes ONLY protocol frames
to stdout. Anything diagnostic goes to stderr.

Run directly:  python tests/echo_bridge.py
"""

import io
import json
import os
import sys

PROTOCOL_VERSION = "1"
MAX_TEXT = 1 << 16  # bound every echoed text field (64 KiB), same as the real bridge.


def _eprint(*args):
    try:
        print(*args, file=sys.stderr, flush=True)
    except Exception:
        pass


# Bind stdout as a binary, line-buffered-by-us channel; we always append "\n"
# and flush so the consumer can read line-delimited JSON immediately.
try:
    _OUT = sys.stdout.buffer  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - stdout already binary/None
    _OUT = sys.stdout


def emit(frame):
    """Serialize one CoreToUi frame as a single JSONL line to stdout."""
    try:
        line = json.dumps(frame, ensure_ascii=False, separators=(",", ":"))
    except Exception as exc:  # pragma: no cover - frame is always plain dict
        line = json.dumps(
            {"type": "Error", "message": "serialize failed: %s" % exc, "fatal": False}
        )
    data = (line + "\n").encode("utf-8", errors="replace")
    try:
        if hasattr(_OUT, "write") and "b" in getattr(_OUT, "mode", "b"):
            _OUT.write(data)
        else:  # text-mode stdout fallback
            _OUT.write(line + "\n")
        _OUT.flush()
    except Exception as exc:  # pragma: no cover
        _eprint("[echo_bridge] write failed:", exc)


def _bound(s):
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    if len(s) > MAX_TEXT:
        return s[:MAX_TEXT] + "...[truncated]"
    return s


_mid_counter = 0


def _next_mid():
    global _mid_counter
    _mid_counter += 1
    return "m%d" % _mid_counter


def _echo_turn(text):
    mid = _next_mid()
    emit({"type": "MessageBegin", "mid": mid, "role": "assistant"})
    emit({"type": "MessageDelta", "mid": mid, "text": _bound("echo: " + text)})
    emit({"type": "MessageEnd", "mid": mid, "reason": "end"})


def handle(frame):
    """Dispatch one parsed UiToCore frame. Returns True to keep looping."""
    if not isinstance(frame, dict):
        emit({"type": "Error", "message": "frame is not an object", "fatal": False})
        return True
    ftype = frame.get("type")
    if ftype == "Submit":
        _echo_turn(_bound(frame.get("text", "")))
    elif ftype == "Command":
        name = _bound(frame.get("name", ""))
        args = _bound(frame.get("args", ""))
        _echo_turn(("/" + name + (" " + args if args else "")).strip())
    elif ftype == "Answer":
        _echo_turn(_bound(frame.get("text", "") or frame.get("option_id", "")))
    elif ftype == "Ping":
        emit({"type": "Pong", "nonce": _bound(frame.get("nonce", ""))})
    elif ftype == "SwitchLlm":
        emit({"type": "Status", "model": "echo"})
    elif ftype == "ListLlms":
        emit(
            {
                "type": "LlmList",
                "items": [
                    {"idx": 0, "name": "Echo/echo-0", "current": True},
                    {"idx": 1, "name": "Echo/echo-1", "current": False},
                ],
            }
        )
    elif ftype in ("Intervene", "Cancel"):
        # No active async turn in the echo stub; acknowledge by no-op.
        pass
    elif ftype == "Shutdown":
        return False
    elif ftype is None:
        emit({"type": "Error", "message": "frame missing 'type'", "fatal": False})
    else:
        emit({"type": "Error", "message": "unknown frame type: %s" % ftype, "fatal": False})
    return True


def main():
    # Vendored capabilities line (stderr) — parity with ga_bridge.py so a
    # supervisor can probe either bridge identically.
    _eprint("V=1 caps=echo,ping")
    emit({"type": "Ready", "version": PROTOCOL_VERSION, "model": "echo"})

    if hasattr(sys.stdin, "buffer"):
        stream = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
    else:  # pragma: no cover - stdin already text
        stream = sys.stdin

    for raw in stream:
        line = raw.strip()
        if not line:
            continue
        try:
            frame = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            emit({"type": "Error", "message": "invalid json line", "fatal": False})
            continue
        try:
            if not handle(frame):
                break
        except Exception as exc:  # never let one bad frame hang the loop
            emit({"type": "Error", "message": "handler error: %s" % exc, "fatal": False})

    _eprint("[echo_bridge] stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
