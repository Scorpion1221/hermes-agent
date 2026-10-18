"""A durable turn-lease wait posts ONE notice to a chat, not one per wait tick.

While a turn waits for the session turn lease, ``admit_durable_turn_lease`` emits a status on every
wait tick (every 15s, for up to 30 minutes). Chat adapters without ``send_or_update_status``
(Feishu, Discord, WeCom, relay, ...) turned each tick into a new message. The gateway chat filter
drops the repeats (ticks and the closing "session is free"): the first notice still says why the
turn has not started, and the agent's own status rail (CLI, TUI, api_server) keeps every tick.
"""

import asyncio
import itertools
import os
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import hermes_state_compression
from agent.status_output import StatusOutputMixin
from agent.turn_facade_lease import (
    LEASE_FREE_STATUS, LEASE_WAIT_STATUS, LEASE_WAIT_TICK_STATUS_TEMPLATE, admit_durable_turn_lease)
from gateway.config import Platform
from gateway.run import _prepare_gateway_status_message
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from hermes_state import SessionDB

_TICK_SECONDS = (15, 30, 1799)
_EVERY_TICK = {LEASE_WAIT_TICK_STATUS_TEMPLATE.format(seconds=s) for s in range(1, 1800)}


class _AppendOnlyChatAdapter:
    """Feishu-shaped: no ``send_or_update_status``, so every delivered status is a new message."""

    def __init__(self):
        self.sent = []

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SimpleNamespace(success=True, message_id=f"m{len(self.sent)}")


class _WaitingAgent(StatusOutputMixin):
    """The agent surface ``admit_durable_turn_lease`` touches; statuses go through the real mixin."""

    def __init__(self, db, status_callback):
        self._session_db = db
        self.session_id = "s1"
        self.status_callback = status_callback
        self.suppress_status_output = True
        self.log_prefix = ""
        self._persist_disabled = False
        self._interrupt_requested = False
        self._activity_lock = threading.Lock()

    def _liveness_activity_lock(self):
        return self._activity_lock


def test_feishu_gets_one_lease_wait_notice_while_the_status_rail_gets_every_tick(tmp_path, monkeypatch):
    # The store's real 15s notice cadence, on a clock that advances 5s per reading and never sleeps.
    readings = itertools.count(0.0, 5.0)
    monkeypatch.setattr(hermes_state_compression, "time", SimpleNamespace(
        time=time.time, monotonic=lambda: next(readings), sleep=lambda _seconds: None))
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="test")
    holder = f"pid={os.getpid()}:turn=leftover"  # a leftover turn in this same process
    assert db.try_acquire_session_turn_lease("s1", holder, ttl_seconds=600)
    adapter = _AppendOnlyChatAdapter()
    agent_statuses = []

    async def wait_for_the_lease():
        ctx = TurnContext(
            source=SessionSource(platform=Platform.FEISHU, chat_id="oc_chat", chat_type="group"),
            user_config={}, _run_still_current=lambda: True, _status_adapter=adapter,
            _status_chat_id="oc_chat", _loop_for_step=asyncio.get_running_loop())
        turn = TurnRunner(SimpleNamespace(), ctx)

        def status_callback(kind, message):
            agent_statuses.append(message)
            turn._status_callback_sync(kind, message)
            if len(agent_statuses) == 4:  # the first notice and three ticks: the holder finishes
                db.release_session_turn_lease("s1", holder)

        admission = await asyncio.to_thread(
            admit_durable_turn_lease, _WaitingAgent(db, status_callback), session_id="s1",
            relay_turn_id="s1:t:1", task_context={"session_id": "s1", "task_id": "t", "platform": "feishu"},
            conversation_history=[])
        # Let every status send already scheduled onto this loop finish.
        await asyncio.gather(*(task for task in asyncio.all_tasks() if task is not asyncio.current_task()))
        return admission

    try:
        admission = asyncio.run(wait_for_the_lease())
        assert admission.early_result is None and admission.lease is not None
        admission.lease.release()
    finally:
        db.close()

    assert agent_statuses[0] == LEASE_WAIT_STATUS
    assert all(status in _EVERY_TICK for status in agent_statuses[1:4])
    assert agent_statuses[4:] == [LEASE_FREE_STATUS]
    assert adapter.sent == [LEASE_WAIT_STATUS]


@pytest.mark.parametrize("progress_notices", [False, True])
@pytest.mark.parametrize("platform", [Platform.FEISHU, Platform.TELEGRAM, Platform.DISCORD])
def test_chat_platforms_drop_lease_wait_repeats_and_keep_the_first_notice(platform, progress_notices):
    # Independent of the compression.progress_notices opt-in: these are not compression progress.
    with patch("gateway.run._gateway_compression_progress_notices_enabled", return_value=progress_notices):
        for seconds in _TICK_SECONDS:
            tick = LEASE_WAIT_TICK_STATUS_TEMPLATE.format(seconds=seconds)
            assert _prepare_gateway_status_message(platform, "lifecycle", tick) is None
        assert _prepare_gateway_status_message(platform, "lifecycle", LEASE_FREE_STATUS) is None
        assert _prepare_gateway_status_message(platform, "lifecycle", LEASE_WAIT_STATUS) == LEASE_WAIT_STATUS


@pytest.mark.parametrize("platform", ["api_server", "local"])
def test_raw_text_surfaces_keep_every_lease_wait_status(platform):
    # Programmatic clients (and their idle watchdogs) read every tick from the status stream.
    ticks = [LEASE_WAIT_TICK_STATUS_TEMPLATE.format(seconds=s) for s in _TICK_SECONDS]
    for message in (LEASE_WAIT_STATUS, *ticks, LEASE_FREE_STATUS):
        assert _prepare_gateway_status_message(platform, "lifecycle", message) == message
