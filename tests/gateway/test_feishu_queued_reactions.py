"""Per-message status badges on Feishu inbound messages.

Each message shows exactly one badge: "OneSecond" while it waits behind a busy
turn, "Typing" while its own turn runs, nothing once answered (or dropped).
A burst merged inside the batch window marks every message and replies to
the first one.
"""
import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import ProcessingOutcome


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_owner")
    monkeypatch.setenv("FEISHU_BOT_OPEN_ID", "ou_bot")
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    for name in ("FEISHU_ALLOW_ALL_USERS", "GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS",
                 "FEISHU_GROUP_POLICY", "FEISHU_REACTIONS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda *a, **kw: [])


class Reactions:
    """Fake reaction API: tracks the badges currently shown per message."""

    def __init__(self, adapter):
        self.shown = {}
        self.log = []
        self._ids = 0

        async def add(message_id, emoji):
            self._ids += 1
            rid = f"r{self._ids}"
            self.shown.setdefault(message_id, {})[rid] = emoji
            self.log.append(("+", message_id, emoji))
            return rid

        async def remove(message_id, reaction_id):
            emoji = self.shown.get(message_id, {}).pop(reaction_id, None)
            self.log.append(("-", message_id, emoji))
            return True

        adapter._add_reaction = add
        adapter._remove_reaction = remove

    def badges(self, message_id):
        return sorted(self.shown.get(message_id, {}).values())


def _adapter(extra=None):
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = FeishuAdapter(PlatformConfig(enabled=True, extra=extra or {}))
    adapter._send_with_retry = AsyncMock()
    adapter.send = AsyncMock()
    return adapter, Reactions(adapter)


def _busy_gateway(mode="queue"):
    """Real FeishuAdapter + real runner busy routing, with a turn running."""
    from gateway.run import GatewayRunner
    from run_agent import AIAgent

    adapter, reactions = _adapter({"group_sessions_per_user": False})
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(group_sessions_per_user=False)
    runner.session_store = None
    runner._draining = False
    runner._busy_input_mode = mode
    runner._busy_text_mode = "interrupt"
    runner._startup_restore_in_progress = False
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner.pairing_store = None
    runner.adapters = {Platform.FEISHU: adapter}
    adapter.set_message_handler(runner._handle_message)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)

    owner = adapter.build_source(chat_id="oc_dm", chat_type="dm", user_id="ou_owner")
    key = runner._session_key_for_source(owner)
    agent = object.__new__(AIAgent)
    agent._pending_steer = None
    agent._pending_steer_lock = threading.Lock()
    agent._active_children = []
    agent._gateway_stream_consumer = None
    agent.interrupt = Mock()
    runner._session_state(key).turn.agent = agent
    adapter._active_sessions[key] = asyncio.Event()
    return runner, adapter, reactions, key


def _dm(text, message_id):
    message = SimpleNamespace(
        message_id=message_id, chat_id="oc_dm", chat_type="p2p", message_type="text",
        content=json.dumps({"text": text}), mentions=[],
        thread_id=None, parent_id=None, root_id=None, upper_message_id=None,
    )
    sender = SimpleNamespace(sender_id=SimpleNamespace(open_id="ou_owner", user_id=None, union_id=None), sender_type="user")
    return SimpleNamespace(event=SimpleNamespace(message=message, sender=sender))


async def _deliver(adapter, *messages, batch_delay=0.0):
    adapter._text_batch_delay_seconds = batch_delay
    for message in messages:
        await adapter._handle_message_event_data(message)
    for _ in range(100):
        if not adapter._pending_text_batch_tasks:
            break
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_messages_sent_while_busy_each_show_one_second():
    runner, adapter, reactions, key = _busy_gateway(mode="queue")

    await _deliver(adapter, _dm("第二条", "om_2"))
    await _deliver(adapter, _dm("第三条", "om_3"))

    assert reactions.badges("om_2") == ["OneSecond"]
    assert reactions.badges("om_3") == ["OneSecond"]


@pytest.mark.asyncio
async def test_queued_message_swaps_one_second_for_typing_when_its_turn_starts_then_clears():
    runner, adapter, reactions, key = _busy_gateway(mode="queue")
    await _deliver(adapter, _dm("第二条", "om_2"))
    queued = adapter._pending_messages[key]

    await adapter.on_processing_start(queued)
    assert reactions.badges("om_2") == ["Typing"]
    await adapter.on_processing_complete(queued, ProcessingOutcome.SUCCESS)
    assert reactions.badges("om_2") == []


@pytest.mark.asyncio
async def test_merged_burst_marks_every_message_and_replies_to_the_first():
    adapter, reactions = _adapter()
    dispatched = []

    async def _capture(event):
        dispatched.append(event)
        await adapter.on_processing_start(event)

    adapter._handle_message_with_guards = _capture
    await _deliver(adapter, _dm("第一条", "om_1"), _dm("第二条", "om_2"), batch_delay=0.05)

    assert len(dispatched) == 1
    event = dispatched[0]
    assert event.message_id == "om_1"  # the answer starts with the first message
    assert reactions.badges("om_1") == ["Typing"] and reactions.badges("om_2") == ["Typing"]
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    assert reactions.badges("om_1") == [] and reactions.badges("om_2") == []


@pytest.mark.asyncio
async def test_a_message_never_shows_two_badges_at_once():
    runner, adapter, reactions, key = _busy_gateway(mode="queue")
    await _deliver(adapter, _dm("第二条", "om_2"))
    queued = adapter._pending_messages[key]
    await adapter.mark_event_queued(queued)  # passing through another branch is a no-op
    await adapter.on_processing_start(queued)
    await adapter.on_processing_start(queued)

    assert reactions.badges("om_2") == ["Typing"]


@pytest.mark.asyncio
async def test_stop_clears_one_second_on_the_discarded_message():
    runner, adapter, reactions, key = _busy_gateway(mode="queue")
    await _deliver(adapter, _dm("第二条", "om_2"))
    assert reactions.badges("om_2") == ["OneSecond"]

    source = adapter._pending_messages[key].source
    runner._session_state(key).turn.agent = None  # the hard interrupt itself is out of scope here
    await runner._interrupt_and_clear_session(
        key, source, interrupt_reason="stop", invalidation_reason="test",
    )

    assert reactions.badges("om_2") == []


@pytest.mark.asyncio
async def test_failed_turn_marks_each_message_with_cross_mark():
    adapter, reactions = _adapter()
    event = SimpleNamespace(message_id="om_1", metadata={"feishu_message_ids": ["om_1", "om_2"]})

    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)

    assert reactions.badges("om_1") == ["CrossMark"] and reactions.badges("om_2") == ["CrossMark"]


@pytest.mark.asyncio
async def test_reactions_toggle_disables_every_badge(monkeypatch):
    monkeypatch.setenv("FEISHU_REACTIONS", "false")
    runner, adapter, reactions, key = _busy_gateway(mode="queue")

    await _deliver(adapter, _dm("第二条", "om_2"))
    await adapter.on_processing_start(adapter._pending_messages[key])

    assert reactions.log == []
