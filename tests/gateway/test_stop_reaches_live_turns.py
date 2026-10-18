"""/stop reaches every live turn thread of the session, not just the one in the slot.

Incident (Feishu, 2026-10-17): a /stop landed while a freshly built turn agent was not yet
promoted into the session slot ("Skipping stale agent promotion"). /stop hard-interrupts only
``state.turn.agent``, so that turn ran on unreachable: it took the durable session turn lease,
kept running foreground commands for 12 minutes, every later /stop hit only the turn waiting
behind it, and follow-up turns sat in the lease wait posting "Still waiting" every 15 s.

Contracts pinned here (live turns are bound through the production binding point,
``TurnRunner._wire_turn_agent_callbacks``, and run on the real turn worker wrapper):
- /stop (and /new, eviction: everything through ``_interrupt_running_turn``) hard-interrupts the
  slot's agent AND every live turn agent of the session, reaping each with its own baseline;
- a turn whose generation went stale before it reached ``run_conversation`` never runs it, so it
  never takes the durable lease;
- the agent cache never hands a live turn's instance to a second turn and never soft-releases it;
- end to end against a real SessionDB: the lease a detached turn holds is released within seconds
  of /stop, and the next turn's admission does not wait for it.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.session_state import SessionState
from gateway.turn_context import TurnContext
from tools.process_registry import process_registry

KEY = "agent:main:feishu:dm:ou_stop_live"


class _Agent:
    """Records hard interrupts the way ``request_hard_interrupt`` delivers them."""

    def __init__(self, name: str):
        self.name = name
        self.hard_interrupts: list = []
        self.released = 0

    def hard_interrupt(self, message=None, *, tool_reason=None):
        self.hard_interrupts.append(message)

    def release_clients(self):
        self.released += 1


def _runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner._sessions = {KEY: SessionState()}
    runner._service_tier = None
    return runner


def _turn_ctx(session_key: str = KEY, *, session_id: str = "sess-1", task_id: str = "",
              baseline: frozenset = frozenset()) -> TurnContext:
    return TurnContext(
        source=SessionSource(platform=Platform.LOCAL, chat_id="c", user_id="u"),
        session_key=session_key, session_id=session_id, user_config={},
        _hooks_ref=SimpleNamespace(loaded_hooks=False), process_task_id=task_id, process_baseline=baseline,
    )


def _bind_turn(runner, agent, session_key: str = KEY, **ctx_kwargs) -> None:
    """Bind ``agent`` the way a turn thread does once its agent is built."""
    TurnRunner(runner, _turn_ctx(session_key, **ctx_kwargs))._wire_turn_agent_callbacks(
        agent, {}, None, None, None, False)


def _reaps(monkeypatch) -> tuple[list, threading.Event]:
    calls: list = []
    done = threading.Event()

    def _record(task_id, baseline, *, source):
        calls.append((task_id, baseline, source))
        if len(calls) >= 2:
            done.set()
        return 0

    monkeypatch.setattr(process_registry, "kill_started_since", _record)
    return calls, done


def test_stop_reaches_a_live_turn_outside_the_slot(monkeypatch):
    """The slot holds W (the turn waiting for the lease); Z is the detached holder thread."""
    calls, both_reaped = _reaps(monkeypatch)
    runner = _runner()
    w, z = _Agent("W"), _Agent("Z")
    _bind_turn(runner, z, task_id="sess-1", baseline=frozenset({"proc-a"}))
    _bind_turn(runner, w, task_id="sess-1", baseline=frozenset({"proc-a", "proc-z"}))
    runner._sessions[KEY].turn.agent = w

    runner._interrupt_running_turn(KEY, interrupt_reason="Stop requested", invalidation_reason="stop_command")

    assert w.hard_interrupts == ["Stop requested"]
    assert z.hard_interrupts == ["Stop requested"], "the detached turn thread never heard the /stop"
    # Each turn is reaped with ITS OWN baseline: Z started proc-z after its baseline, and W's
    # baseline already holds it, so reaping with W's baseline alone would leave proc-z running.
    assert both_reaped.wait(5)
    assert sorted(calls) == sorted([
        ("sess-1", frozenset({"proc-a", "proc-z"}), "gateway_turn_interrupt"),
        ("sess-1", frozenset({"proc-a"}), "gateway_turn_interrupt"),
    ])


def test_stop_reaches_a_live_turn_while_the_slot_holds_the_pending_sentinel(monkeypatch):
    """A /stop landing before the new agent was promoted (slot = sentinel) still reaches it."""
    from gateway.run import _AGENT_PENDING_SENTINEL

    _reaps(monkeypatch)
    runner = _runner()
    z = _Agent("Z")
    _bind_turn(runner, z)
    runner._sessions[KEY].turn.agent = _AGENT_PENDING_SENTINEL

    runner._interrupt_running_turn(KEY, interrupt_reason="Stop requested", invalidation_reason="stop_command_pending")

    assert z.hard_interrupts == ["Stop requested"]


def test_a_live_turn_in_the_slot_is_interrupted_once(monkeypatch):
    _reaps(monkeypatch)
    runner = _runner()
    agent = _Agent("A")
    _bind_turn(runner, agent)
    runner._sessions[KEY].turn.agent = agent

    runner._interrupt_running_turn(KEY, interrupt_reason="Stop requested", invalidation_reason="stop_command")

    assert agent.hard_interrupts == ["Stop requested"]


def test_another_sessions_live_turn_is_left_alone(monkeypatch):
    _reaps(monkeypatch)
    runner = _runner()
    other = _Agent("other")
    _bind_turn(runner, other, "agent:main:feishu:dm:someone_else")

    runner._interrupt_running_turn(KEY, interrupt_reason="Stop requested", invalidation_reason="stop_command")

    assert other.hard_interrupts == []


@pytest.mark.asyncio
async def test_a_turn_thread_counts_as_running_until_it_exits(monkeypatch):
    """The live-turn entry lives exactly as long as the turn's executor thread."""
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "0")
    runner = _runner()
    agent = _Agent("live")
    ctx = _turn_ctx()
    seen_while_running: list = []

    def run_sync():
        TurnRunner(runner, ctx)._wire_turn_agent_callbacks(agent, {}, None, None, None, False)
        seen_while_running.append(id(agent) in runner._running_agent_ids())
        return {"final_response": "ok"}

    worker = runner._run_agent_start_turn_worker(ctx, run_sync)
    await worker.executor_task

    assert seen_while_running == [True]
    assert id(agent) not in runner._running_agent_ids()


# ── a turn that went stale before run_conversation never runs it ────────────────────────────


class _CountingAgent:
    run_calls = 0

    def __init__(self, **kwargs):
        self.model = kwargs["model"]
        self.session_id = kwargs.get("session_id")
        self.tools = []
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0, context_length=200_000)
        self.session_prompt_tokens = self.session_completion_tokens = 0

    def run_conversation(self, _message, **_kwargs):
        type(self).run_calls += 1
        return {"final_response": "ran anyway", "messages": [], "api_calls": 1}


def _turn_runner_host():
    runner = MagicMock()
    runner.config = SimpleNamespace(streaming=None)
    runner._provider_routing = {}
    runner._agent_cache_lock = None
    runner._agent_cache = {}
    runner._session_db = runner._prefill_messages = None
    runner._pending_model_notes = runner._pending_skills_reload_notes = {}
    runner.session_store._entries = {}
    runner._get_system_prompt_for_channel.return_value = None
    runner._resolve_session_agent_runtime.return_value = ("test-model", {})
    runner._resolve_session_reasoning_config.return_value = None
    runner._resolve_session_service_tier.return_value = None
    runner._resolve_turn_agent_config.return_value = {"model": "test-model", "runtime": {}}
    runner._agent_config_signature.return_value = ("sig",)
    runner._extract_cache_busting_config.return_value = {}
    runner._refresh_fallback_model.return_value = None
    runner._consume_pending_native_image_paths.return_value = []
    runner._consume_pending_turn_sidecar_notes.return_value = []
    for lane in ("_is_telegram_topic_lane", "_is_discord_auto_thread_lane", "_is_relay_discord_channel_lane"):
        getattr(runner, lane).return_value = False
    return runner


@pytest.mark.parametrize("still_current", [False, True], ids=["stale", "current"])
def test_a_stale_generation_turn_never_calls_run_conversation(still_current):
    """/stop bumped the generation while the turn was still being set up: the built agent is
    never promoted, so nothing could reach it — it must not start the conversation at all."""
    _CountingAgent.run_calls = 0
    ctx = TurnContext(
        source=SessionSource(platform=Platform.LOCAL, chat_id="c", user_id="u"),
        message="hi", history=[], session_id="sid", session_key=KEY, user_config={},
        AIAgent=_CountingAgent, resolve_display_setting=lambda *_a: False,
        _run_still_current=lambda: still_current, _hooks_ref=SimpleNamespace(loaded_hooks=False),
    )

    result = TurnRunner(_turn_runner_host(), ctx).run_sync()

    if still_current:
        assert _CountingAgent.run_calls == 1 and result["final_response"] == "ran anyway"
    else:
        assert _CountingAgent.run_calls == 0
        assert result["interrupted"] is True and result["api_calls"] == 0


# ── the cache never reuses or soft-releases a live turn's instance ──────────────────────────


def _cache_runner():
    runner = _runner()
    runner._agent_cache = OrderedDict()
    runner._agent_cache_lock = threading.Lock()
    runner._agent_config_signature = lambda *a, **k: "sig"
    runner._extract_cache_busting_config = lambda _cfg: {}
    runner._refresh_fallback_model = lambda: None
    runner._apply_fallback_chain_to_agent = lambda agent, chain: None
    runner._enforce_agent_cache_cap = lambda: None
    # Release inline so a release under a live turn is observable without waiting on a thread.
    runner._spawn_release_thread = lambda target, args, name, **kw: target(*args)
    return runner


def _next_turn_agent(runner, cached_agent, *, cached_count, live_count):
    """The agent the session's NEXT turn resolves while ``cached_agent`` sits in the cache."""
    runner._agent_cache[KEY] = (cached_agent, "sig", cached_count, "sid")
    turn = TurnRunner(runner, _turn_ctx(session_id="sid"))
    turn._build_fresh_agent = lambda *a, **k: _Agent("fresh")
    turn._current_message_count = lambda: live_count
    turn._cached_sid_is_dead = lambda lock, cache: ("sid", False)
    return turn._resolve_turn_agent({"model": "m", "runtime": {}}, "feishu", "", 10, None, {})


def test_a_live_turns_cached_instance_is_not_reused():
    runner = _cache_runner()
    live = _Agent("live")
    _bind_turn(runner, live)

    agent, reused = _next_turn_agent(runner, live, cached_count=4, live_count=4)

    assert reused is False and agent is not live, "two live turns would share one AIAgent"
    assert live.released == 0


def test_a_live_turns_cached_instance_is_not_soft_released_on_count_drift():
    """The holder's own tool rounds bump message_count, so the next lookup evicts its entry:
    that eviction must not release_clients()/wipe the transcript under the running thread."""
    runner = _cache_runner()
    live = _Agent("live")
    _bind_turn(runner, live)

    agent, reused = _next_turn_agent(runner, live, cached_count=4, live_count=9)

    assert reused is False and agent is not live
    assert live.released == 0, "a running turn's clients were released under it"


def test_an_idle_cached_instance_is_still_reused():
    runner = _cache_runner()
    idle = _Agent("idle")

    agent, reused = _next_turn_agent(runner, idle, cached_count=4, live_count=4)

    assert reused is True and agent is idle
    assert idle.released == 0


# ── a /stop on an idle slot still reaches a live turn thread ────────────────────────────────


@pytest.mark.asyncio
async def test_stop_on_an_idle_slot_reaches_a_live_turn():
    """The slot was released without stopping the turn (``/resume``, an inactivity abandon):
    /stop must stop it, not answer "no active task" while it keeps the session busy."""
    from gateway.config import GatewayConfig, PlatformConfig
    from gateway.platforms.event import MessageEvent, MessageType

    runner = GatewayRunner(config=GatewayConfig(platforms={Platform.FEISHU: PlatformConfig(enabled=True)}))
    runner.adapters = {}
    source = SessionSource(platform=Platform.FEISHU, chat_id="ou_idle", chat_type="dm", user_id="ou_idle")
    key = runner._session_key_for_source(source)
    live = _Agent("live")
    _bind_turn(runner, live, key)

    reply = await runner._handle_stop_command(MessageEvent(text="/stop", message_type=MessageType.TEXT, source=source))

    assert live.hard_interrupts, "the live turn kept running"
    assert "Stopped" in str(getattr(reply, "text", reply))


# ── E2E: real AIAgent + real SessionDB + the real terminal tool ───────────────────────────


def _lease_rows(db) -> list:
    with db._read_ctx() as conn:
        return conn.execute("SELECT holder FROM session_turn_leases").fetchall()


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
async def test_stop_releases_the_lease_a_detached_turn_holds(tmp_path, monkeypatch):
    """The incident end to end: a turn the slot no longer points at runs a long foreground terminal
    command and holds the durable lease; /stop must end the command, the turn and the lease."""
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tests.e2e.core.history._helpers import NO_BACKGROUND_REVIEW, OFFLINE_CONFIG
    from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home

    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "0")
    home = Path(os.environ["HERMES_HOME"])
    started_file = tmp_path / "started"
    db = SessionDB(tmp_path / "state.db")
    db.create_session("sess-live", source="cli")
    command = f"touch {started_file} && sleep 60"

    with FakeLLMServer([ToolCall("terminal", {"command": command, "timeout": 120}), Text("stopped")]) as srv:
        write_hermes_home(home, srv.base_url, extra_config=OFFLINE_CONFIG + NO_BACKGROUND_REVIEW
                          + "approvals:\n  mode: off\n")
        holder = AIAgent(provider="custom", base_url=srv.base_url, api_key="sk-fake", model="fake-model",
                         session_id="sess-live", session_db=db, quiet_mode=True, platform="cli",
                         skip_memory=True, skip_context_files=True, enabled_toolsets=["terminal"])
        runner = _runner()
        ctx = _turn_ctx(session_id="sess-live")

        def run_sync():
            # The gateway turn thread: bind the built agent, then run the conversation.
            TurnRunner(runner, ctx)._wire_turn_agent_callbacks(holder, {}, None, None, None, False)
            return holder.run_conversation("release please", conversation_history=[])

        worker = runner._run_agent_start_turn_worker(ctx, run_sync)
        try:
            from tests.waiting import async_wait_until
            await async_wait_until(started_file.exists, timeout=30, what="the holder starting its command")
            assert _lease_rows(db), "the holder turn should own the durable lease"

            # The slot holds a DIFFERENT agent (the incident's lease waiter), as when the holder
            # was never promoted; /stop must still reach the holder thread.
            runner._sessions[KEY].turn.agent = _Agent("waiter")
            started = time.monotonic()
            runner._interrupt_running_turn(KEY, interrupt_reason="Stop requested", invalidation_reason="stop_command")
            done, _ = await asyncio.wait({worker.executor_task}, timeout=15)

            assert done, "the detached turn kept running after /stop"
            assert time.monotonic() - started < 10
            assert _lease_rows(db) == [], "the durable lease outlived the stopped turn"
            # The next admission takes the lease at once instead of waiting behind the holder.
            assert db.try_acquire_session_turn_lease("sess-live", "next-turn", ttl_seconds=30) is True
        finally:
            if not worker.executor_task.done():
                holder.interrupt("test cleanup")
                await asyncio.wait({worker.executor_task}, timeout=30)
            holder.close()
            db.close()
