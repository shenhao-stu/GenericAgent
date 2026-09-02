"""A session must either answer or fail fast — never hang on an unusable model.

Covers the first-run path: fresh mykey with an empty default group, user adds one model,
starts a chat. Run: pytest frontends/tests/test_bridge_model_readiness.py -v
"""

from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

from frontends.tests.test_bridge_sessions import AgentManager, _make_session, _mod, manager, tmp_ga_root  # noqa: F401

EMPTY_GROUP_MYKEY = (
    "mixin_config = {\n    'llm_nos': [],\n    'max_retries': 10,\n    'base_delay': 0.5,\n}\n"
)
OAI_MODEL = {"protocol": "oai", "name": "GA_TP_auto", "model": "auto",
             "apibase": "https://relay.example/v1", "apikey": "sk-test"}


def _exec_mykey(path: Path) -> dict:
    namespace: dict = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)  # bridge-generated dict literals only
    return {k: v for k, v in namespace.items() if not k.startswith("__")}


@pytest.fixture
def profile_manager(manager: AgentManager, tmp_ga_root: Path, monkeypatch):
    """AgentManager whose mykey.py round-trips through a file-backed reload_mykeys."""
    mykey = tmp_ga_root / "mykey.py"
    mykey.write_text(EMPTY_GROUP_MYKEY, encoding="utf-8")
    monkeypatch.setattr(sys.modules["llmcore"], "reload_mykeys", lambda: (_exec_mykey(mykey), True), raising=False)
    monkeypatch.setattr(manager, "_reload_live_agents", lambda: None)
    return manager


def _group_members(profiles: list[dict]) -> list[str]:
    return next(p["members"] for p in profiles if p["kind"] == "mixin")


class TestDefaultGroupInvariant:
    def test_first_model_joins_the_empty_default_group(self, profile_manager: AgentManager):
        result = profile_manager.add_model_profile(dict(OAI_MODEL))

        assert _group_members(result["profiles"]) == ["GA_TP_auto"]
        assert profile_manager._default_group_is_empty() is False

    def test_later_models_leave_a_populated_group_alone(self, profile_manager: AgentManager):
        profile_manager.add_model_profile(dict(OAI_MODEL))
        second = profile_manager.add_model_profile({**OAI_MODEL, "name": "second", "model": "gpt-x"})

        assert _group_members(second["profiles"]) == ["GA_TP_auto"]

    def test_no_mixin_block_needs_no_repair(self, profile_manager: AgentManager, tmp_ga_root: Path):
        (tmp_ga_root / "mykey.py").write_text("", encoding="utf-8")

        result = profile_manager.add_model_profile(dict(OAI_MODEL))

        assert [p["kind"] for p in result["profiles"]] == ["native"]


class _FakeSession:
    """Stand-in for llmcore.Native*Session: records the cfg it was built with, replays scripted chunks."""
    instances: list = []
    chunks: list = ["pong"]
    raise_on_ask: Optional[Exception] = None

    def __init__(self, cfg):
        self.cfg = cfg
        type(self).instances.append(self)

    def raw_ask(self, messages):
        self.messages = messages
        if self.raise_on_ask:
            raise self.raise_on_ask
        yield from self.chunks
        return []


class _OAISession(_FakeSession):
    instances = []


class _ClaudeSession(_FakeSession):
    instances = []


class TestModelProbe:
    """The Add-Model form verifies key/base/model by sending the runtime's own request before anything is saved."""

    @pytest.fixture(autouse=True)
    def _runtime_contract(self, monkeypatch):
        llmcore = sys.modules["llmcore"]
        monkeypatch.setattr(llmcore, "NativeOAISession", _OAISession, raising=False)
        monkeypatch.setattr(llmcore, "NativeClaudeSession", _ClaudeSession, raising=False)
        for cls in (_OAISession, _ClaudeSession):
            cls.instances.clear(); cls.chunks = ["pong"]; cls.raise_on_ask = None

    def test_probe_uses_the_runtime_session_class_with_minimal_overrides(self):
        cfg = {"protocol": "oai", "apibase": "https://api.example.com", "model": "gpt-4o", "apikey": "sk-x", "proxy": "http://p:1"}
        result = _mod.probe_model_config(cfg)

        assert result == {"ok": True, "latencyMs": result["latencyMs"]} and result["latencyMs"] >= 0
        (sess,) = _OAISession.instances
        assert _ClaudeSession.instances == []
        assert sess.messages == [{"role": "user", "content": "ping"}]
        assert sess.cfg == {**cfg, "stream": False, "max_tokens": 1, "max_retries": 0, "timeout": 20.0, "read_timeout": 20.0}

    def test_claude_protocol_selects_the_claude_session(self):
        _mod.probe_model_config({"protocol": "claude", "apibase": "https://api.anthropic.com", "model": "claude-x", "apikey": "sk-ant-1"})
        assert len(_ClaudeSession.instances) == 1 and _OAISession.instances == []

    def test_http_error_from_the_runtime_is_decoded(self):
        _OAISession.chunks = ['!!!Error: HTTP 401: {"error": {"message": "Incorrect API key provided", "type": "invalid_request_error"}}']
        result = _mod.probe_model_config({"protocol": "oai", "apibase": "https://api.example.com", "model": "m", "apikey": "bad"})
        assert result == {"ok": False, "status": 401, "latencyMs": result["latencyMs"], "error": "Incorrect API key provided"}

    def test_plain_string_error_bodies_pass_through(self):
        _OAISession.chunks = ["!!!Error: HTTP 403: ", "this group only allows Claude Code clients"]
        result = _mod.probe_model_config({"protocol": "oai", "apibase": "https://relay.example", "model": "m", "apikey": "k"})
        assert result["status"] == 403 and result["error"] == "this group only allows Claude Code clients"

    def test_transport_failure_is_reported_not_raised(self):
        _OAISession.chunks = ["!!!Error: ConnectionError: refused"]
        result = _mod.probe_model_config({"protocol": "oai", "apibase": "http://127.0.0.1:1", "model": "m", "apikey": "k"})
        assert result == {"ok": False, "latencyMs": result["latencyMs"], "error": "ConnectionError: refused"}

    def test_client_side_exception_is_reported_not_raised(self):
        _OAISession.raise_on_ask = ValueError("bad config")
        result = _mod.probe_model_config({"protocol": "oai", "apibase": "http://x", "model": "m", "apikey": "k"})
        assert result == {"ok": False, "error": "ValueError: bad config"}

    def test_missing_fields_short_circuit_before_any_request(self):
        assert _mod.probe_model_config({"protocol": "oai", "apibase": "", "model": "m"})["ok"] is False
        assert _OAISession.instances == []


class TestSessionWorkDir:
    """A session bound to a folder makes the agent work there; the default keeps GA's own temp/."""

    def test_default_session_keeps_agent_default(self, manager: AgentManager):
        sess = manager.create_session()
        assert sess.cwd == manager.ga_root
        assert manager.session_work_dir(sess) is None

    def test_explicit_folder_is_bound(self, manager: AgentManager, tmp_path: Path):
        project = tmp_path / "my-project"
        project.mkdir()
        sess = manager.create_session(cwd=str(project))
        assert manager.session_work_dir(sess) == str(project)

    def test_missing_folder_is_rejected(self, manager: AgentManager, tmp_path: Path):
        with pytest.raises(ValueError):
            manager.create_session(cwd=str(tmp_path / "nope"))

    def test_deleted_folder_falls_back_to_default(self, manager: AgentManager, tmp_path: Path):
        gone = tmp_path / "gone"
        gone.mkdir()
        sess = manager.create_session(cwd=str(gone))
        gone.rmdir()
        assert manager.session_work_dir(sess) is None


class _Backend:
    history: list = []
    name = "model-a"


class _Client:
    backend = _Backend()


class _Agent:
    inc_out = True

    def __init__(self, llmclient):
        self.llmclient = llmclient
        self.llm_no = 0

    def next_llm(self, no):
        self.llm_no = no

    def put_task(self, _prompt, images=None):
        raise AssertionError("an unusable client must fail before put_task")


class TestUnusableClient:
    def test_reason_covers_missing_and_badmixin_clients(self):
        assert AgentManager.unusable_client_reason(_Agent(None)) == "model_unavailable: no model configured"
        assert AgentManager.unusable_client_reason(_Agent({"mixin_cfg": {}})).startswith("model_unavailable:")
        assert AgentManager.unusable_client_reason(_Agent(_Client())) is None

    def test_turn_fails_fast_instead_of_hanging(self, manager: AgentManager):
        sess = _make_session("sess-badmixin")
        sess.agent = _Agent({"mixin_cfg": {"llm_nos": []}})
        sess.active_turn_id = "t1"
        manager.sessions[sess.id] = sess

        manager.run_agent_turn(sess, "hello", turn_id="t1")

        assert sess.status == "error"
        assert sess.partial is None
        assert sess.messages[-1]["role"] == "error"
        assert sess.messages[-1]["content"].startswith("model_unavailable:")

    def test_dead_worker_is_rebuilt_before_the_turn(self, manager: AgentManager):
        sess = _make_session("sess-dead-worker")
        dead = threading.Thread(target=lambda: None)
        dead.start(); dead.join()
        sess.agent, sess.agent_thread = _Agent(_Client()), dead
        rebuilt = _Agent(_Client())

        def put_task(_prompt, images=None):
            q = queue.Queue()
            q.put({"done": "ok", "outputs": ["ok"]})
            return q

        rebuilt.put_task = put_task
        sess.active_turn_id = "t2"
        manager.sessions[sess.id] = sess
        plan_state = sys.modules["plan_state"]
        with patch.object(manager, "make_agent", return_value=rebuilt) as make_agent, \
                patch.object(plan_state, "sync_plan_path_from_text", lambda *args: None, create=True):
            manager.run_agent_turn(sess, "hello", turn_id="t2")

        make_agent.assert_called_once_with(sess)
        assert sess.agent is rebuilt
        assert sess.status == "idle"

    def test_worker_dying_mid_turn_surfaces_an_error(self, manager: AgentManager):
        sess = _make_session("sess-worker-crash")
        agent = _Agent(_Client())
        agent.put_task = lambda _prompt, images=None: queue.Queue()  # never answers
        sess.agent = agent
        sess.agent_thread = threading.Thread(target=lambda: None)
        sess.agent_thread.start(); sess.agent_thread.join()
        sess.active_turn_id = "t3"
        manager.sessions[sess.id] = sess

        with patch.object(manager, "make_agent", return_value=agent):
            manager.run_agent_turn(sess, "hello", turn_id="t3")

        assert sess.status == "error"
        assert sess.messages[-1]["content"].startswith("agent_crashed:")
