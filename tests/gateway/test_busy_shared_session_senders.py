"""In a shared group/topic session, another participant's busy message gets its own turn.

Steer, redirect and interrupt are corrections by the running turn's sender. Before, any participant's
message in steer mode was spliced into someone else's running turn, and in interrupt mode it cut that
turn short. Driven through the real GatewayRunner busy handler.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.platforms.base import Platform, SessionSource, build_session_key
from gateway.platforms.event import MessageEvent, MessageType


def _runner(mode: str, *, shared: bool = True):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running_agents, runner._running_agents_ts = {}, {}
    runner._pending_messages, runner._busy_ack_ts, runner._queued_events = {}, {}, {}
    runner._draining = False
    runner._busy_input_mode = mode
    runner._busy_text_mode = "interrupt"
    runner.adapters = {}
    runner.config = MagicMock()
    runner.config.group_sessions_per_user = not shared
    runner.config.thread_sessions_per_user = False
    runner.session_store = None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    return runner


def _adapter():
    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.platform = Platform.FEISHU
    adapter._text_debounce = {}
    return adapter


def _event(text: str, user: str, message_id: str) -> MessageEvent:
    source = SessionSource(platform=Platform.FEISHU, chat_id="oc_group", chat_type="group", user_id=user)
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source, message_id=message_id)


def _running(runner, session_key: str, owner: str):
    agent = MagicMock()
    agent._user_id, agent._user_id_alt = owner, None
    agent.steer.return_value = True
    agent.get_activity_summary.return_value = {"seconds_since_activity": 0.0}
    runner._running_agents[session_key] = agent
    runner._running_agents_ts[session_key] = time.time()
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["steer", "interrupt"])
async def test_another_participants_message_is_queued_not_spliced_or_interrupting(mode):
    runner = _runner(mode)
    adapter = _adapter()
    event = _event("unrelated question", "ou_bob", "om_b")
    session_key = build_session_key(event.source, group_sessions_per_user=False)
    runner.adapters[Platform.FEISHU] = adapter
    agent = _running(runner, session_key, owner="ou_alice")

    assert await runner._handle_active_session_busy_message(event, session_key) is True

    agent.steer.assert_not_called()
    agent.interrupt.assert_not_called()
    assert adapter._pending_messages[session_key] is event


@pytest.mark.asyncio
async def test_the_turn_owners_own_follow_up_still_steers():
    runner = _runner("steer")
    adapter = _adapter()
    event = _event("also check the logs", "ou_alice", "om_a2")
    session_key = build_session_key(event.source, group_sessions_per_user=False)
    runner.adapters[Platform.FEISHU] = adapter
    agent = _running(runner, session_key, owner="ou_alice")

    await runner._handle_active_session_busy_message(event, session_key)

    agent.steer.assert_called_once()
    assert session_key not in adapter._pending_messages


@pytest.mark.asyncio
async def test_a_per_user_group_session_keeps_steering_everyone_it_belongs_to():
    runner = _runner("steer", shared=False)
    adapter = _adapter()
    event = _event("more detail", "ou_bob", "om_b")
    session_key = build_session_key(event.source, group_sessions_per_user=True)
    runner.adapters[Platform.FEISHU] = adapter
    agent = _running(runner, session_key, owner="ou_alice")  # not shared: identities are not compared

    await runner._handle_active_session_busy_message(event, session_key)

    agent.steer.assert_called_once()


@pytest.mark.asyncio
async def test_a_typed_yes_is_refused_when_its_sender_may_not_approve(monkeypatch):
    runner = _runner("steer")
    adapter = _adapter()
    event = _event("yes", "ou_bob", "om_b")
    session_key = build_session_key(event.source, group_sessions_per_user=False)
    runner.adapters[Platform.FEISHU] = adapter
    _running(runner, session_key, owner="ou_alice")
    monkeypatch.setattr("tools.approval.has_blocking_approval", lambda key: True)
    runner._check_slash_access = lambda source, cmd: "⛔ only admins can /approve"
    approve = AsyncMock(return_value="approved")
    runner._handle_approve_command = approve

    assert await runner._handle_active_session_busy_message(event, session_key) is True

    approve.assert_not_called()
    sent = adapter._send_with_retry.await_args.kwargs["content"]
    assert "only admins" in sent


@pytest.mark.asyncio
async def test_a_steered_follow_up_is_marked_as_part_of_the_running_turn():
    from plugins.platforms.feishu.adapter import FeishuAdapter
    from gateway.config import PlatformConfig

    runner = _runner("steer")
    adapter = FeishuAdapter(PlatformConfig())
    marked: list[tuple[str, str]] = []

    async def _add(message_id, emoji):
        marked.append((message_id, emoji))
        return f"r_{emoji}"

    adapter._add_reaction = _add
    event = _event("also check the logs", "ou_alice", "om_a2")
    session_key = build_session_key(event.source, group_sessions_per_user=False)
    runner.adapters[Platform.FEISHU] = adapter
    runner._delivery_adapter_for = lambda source: adapter
    _running(runner, session_key, owner="ou_alice")

    await runner._handle_active_session_busy_message(event, session_key)

    assert ("om_a2", "Typing") in marked


@pytest.mark.asyncio
async def test_a_steered_follow_up_gets_no_unpaired_start_on_other_adapters():
    """Discord/Signal/Slack mark a message in on_processing_start and clear it on completion, keyed on
    that one message. A steered message never gets its own completion, so it must not get a start."""
    from gateway.config import PlatformConfig
    from gateway.platforms.base import BasePlatformAdapter, SendResult

    started: list[str] = []

    class _ReactingAdapter(BasePlatformAdapter):
        async def connect(self, **kwargs):
            return True

        async def disconnect(self):
            pass

        async def send(self, *args, **kwargs):
            return SendResult(success=True)

        async def get_chat_info(self, chat_id):
            return {}

        async def on_processing_start(self, event):
            started.append(event.message_id)

    runner = _runner("steer")
    adapter = _ReactingAdapter(PlatformConfig(), Platform.FEISHU)
    event = _event("also check the logs", "ou_alice", "om_a2")
    session_key = build_session_key(event.source, group_sessions_per_user=False)
    runner.adapters[Platform.FEISHU] = adapter
    runner._delivery_adapter_for = lambda source: adapter
    agent = _running(runner, session_key, owner="ou_alice")

    await runner._handle_active_session_busy_message(event, session_key)

    agent.steer.assert_called_once()
    assert started == []
