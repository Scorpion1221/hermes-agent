"""A steer is accepted only while the turn can still read it (2026-10-21, Feishu card click lost).

The turn's last read of the steer slot is the finalizer's leftover hand-off. The gateway keeps the
agent registered as the session's running turn for a while after that (streaming final edit,
cleanup), and ``AIAgent.steer()`` used to accept text with no liveness check: a busy-mode message or
card click landing in that window was "steered into the current run", nobody read it, and it was
never run. Contract: after the last read the window is closed, ``steer()`` refuses, and the caller
falls back to queueing the event as the next turn; a steer accepted before the close is in the
result's ``pending_steer`` exactly once.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _agent() -> AIAgent:
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._disable_streaming = True
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _reply(content="done"):
    msg = SimpleNamespace(content=content, tool_calls=None)
    response = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], model="test/model")
    response.usage = None
    return response


def _run(agent, text="hello", *, on_memory_sync=None):
    agent._interruptible_api_call = lambda _kw: _reply()
    patches = [
        patch.object(agent, "_flush_messages_to_session_db"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ]
    if on_memory_sync is not None:
        patches.append(patch.object(agent, "_sync_external_memory_for_turn", side_effect=on_memory_sync))
    for p in patches:
        p.start()
    try:
        return agent.run_conversation(text)
    finally:
        for p in patches:
            p.stop()


def test_steer_after_the_turn_returned_is_refused():
    """The incident's shape: the turn already returned (the gateway still lists it as running)."""
    agent = _agent()
    result = _run(agent)
    assert result["completed"] is True
    assert agent.steer('card:button {"swp": "approve"}') is False
    assert agent._drain_pending_steer() is None            # nothing parked on an agent no turn will read


def test_steer_during_the_finalizer_tail_is_refused_not_swallowed():
    """A steer after the finalizer's last read (here: during the post-turn memory sync) is refused, so
    the caller queues the original event; it is never accepted into a slot nobody drains."""
    agent = _agent()
    seen = {}

    def late_click(**_kw):
        seen["accepted"] = agent.steer("late click")

    result = _run(agent, on_memory_sync=late_click)
    assert seen["accepted"] is False
    assert result.get("pending_steer") is None
    assert agent._drain_pending_steer() is None


def test_steer_before_the_last_read_is_handed_back_once():
    """Accepted while the turn still runs, unread by any tool batch: the finalizer hands it back as
    the next turn (result.pending_steer), exactly once."""
    agent = _agent()

    def model(_kw):
        assert agent.steer("mid-turn note") is True
        return _reply()

    agent._interruptible_api_call = model
    with (
        patch.object(agent, "_flush_messages_to_session_db"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("start")
    assert result.get("pending_steer") == "mid-turn note"
    assert agent._drain_pending_steer() is None


def test_the_window_reopens_for_the_next_turn():
    agent = _agent()
    _run(agent)
    assert agent.steer("between turns") is False
    seen = {}

    def model(_kw):
        seen["accepted"] = agent.steer("second turn note")
        return _reply()

    agent._interruptible_api_call = model
    with (
        patch.object(agent, "_flush_messages_to_session_db"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("second")
    assert seen["accepted"] is True
    assert result.get("pending_steer") == "second turn note"


def test_close_is_atomic_with_the_last_read():
    """Under concurrent steers, every accepted text is in the closing read exactly once, every refused
    one is not, and the slot is empty afterwards."""
    agent = _agent()
    agent._open_steer_window()
    accepted, refused, stop = [], [], threading.Event()

    def writer():
        i = 0
        while not stop.is_set():
            text = f"n{i}"
            (accepted if agent.steer(text) else refused).append(text)
            i += 1

    thread = threading.Thread(target=writer)
    thread.start()
    while len(accepted) < 50:
        pass
    closed = agent._close_steer_window()
    while not refused:                     # the writer keeps going until the closed window refuses it
        pass
    stop.set()
    thread.join()
    lines = (closed or "").split("\n")
    assert sorted(lines) == sorted(accepted)
    assert refused and not set(refused) & set(lines)
    assert agent._drain_pending_steer() is None
