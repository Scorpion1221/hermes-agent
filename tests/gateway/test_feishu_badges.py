"""Every inbound Feishu message shows where its turn is: OneSecond while queued, Typing while its turn
runs, nothing when it ends (CrossMark on failure), and nothing once /stop or /new drops it.

The adapter's reaction calls are replaced by a recorder of each message's current badge; the busy path
is the real base adapter + gateway funnel.
"""

from __future__ import annotations

import asyncio
import itertools
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import SessionSource, build_session_key
from plugins.platforms.feishu.adapter import FeishuAdapter


class Badges:
    def __init__(self, adapter: FeishuAdapter, *, removable: bool = True):
        self.on: dict[str, list[str]] = {}
        self._ids = itertools.count(1)
        self._by_id: dict[str, tuple[str, str]] = {}
        self._removable = removable

        async def _add(message_id, emoji):
            rid = f"r{next(self._ids)}"
            self._by_id[rid] = (message_id, emoji)
            self.on.setdefault(message_id, []).append(emoji)
            return rid

        async def _remove(message_id, reaction_id):
            if not self._removable:
                return False
            _, emoji = self._by_id.pop(reaction_id)
            self.on[message_id].remove(emoji)
            return True

        adapter._add_reaction = _add
        adapter._remove_reaction = _remove

    def of(self, message_id: str) -> list[str]:
        return self.on.get(message_id, [])


def _adapter() -> tuple[FeishuAdapter, Badges]:
    adapter = FeishuAdapter(PlatformConfig())
    return adapter, Badges(adapter)


def _event(text: str, message_id: str, *, user: str = "ou_a", chat: str = "oc_dm", chat_type: str = "dm") -> MessageEvent:
    source = SessionSource(platform=Platform.FEISHU, chat_id=chat, chat_type=chat_type, user_id=user)
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source, message_id=message_id)


def test_a_running_turn_shows_typing_and_clears_it_when_done():
    adapter, badges = _adapter()
    event = _event("hi", "om_1")

    asyncio.run(adapter.on_processing_start(event))
    assert badges.of("om_1") == ["Typing"]
    asyncio.run(adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS))
    assert badges.of("om_1") == []


def test_a_failed_turn_ends_with_a_cross_mark():
    adapter, badges = _adapter()
    event = _event("hi", "om_1")

    asyncio.run(adapter.on_processing_start(event))
    asyncio.run(adapter.on_processing_complete(event, ProcessingOutcome.FAILURE))

    assert badges.of("om_1") == ["CrossMark"]


def test_a_message_queued_behind_a_busy_turn_waits_with_one_second_then_types():
    adapter, badges = _adapter()
    first, second = _event("one", "om_1"), _event("two", "om_2")

    async def _run():
        await adapter.on_processing_start(first)
        await adapter.mark_event_queued(second)
        assert badges.of("om_2") == ["OneSecond"]
        await adapter.on_processing_complete(first, ProcessingOutcome.SUCCESS)
        await adapter.on_processing_start(second)

    asyncio.run(_run())
    assert badges.of("om_1") == [] and badges.of("om_2") == ["Typing"]


def test_every_message_merged_into_one_turn_carries_the_badge():
    adapter, badges = _adapter()
    a, b, c = _event("a", "om_a"), _event("b", "om_b"), _event("c", "om_c")
    a.absorb_reply_expected(b)
    a.absorb_reply_expected(c)

    asyncio.run(adapter.on_processing_start(a))

    assert all(badges.of(m) == ["Typing"] for m in ("om_a", "om_b", "om_c"))
    asyncio.run(adapter.on_processing_complete(a, ProcessingOutcome.SUCCESS))
    assert all(badges.of(m) == [] for m in ("om_a", "om_b", "om_c"))


@pytest.mark.asyncio
async def test_a_busy_message_through_the_real_base_path_gets_one_second():
    adapter, badges = _adapter()
    adapter._busy_text_mode = "interrupt"
    adapter._message_handler = lambda event: asyncio.sleep(0)
    queued: list[MessageEvent] = []

    async def _busy(event, session_key):
        queued.append(event)
        event._gateway_accepted = True
        return True

    adapter._busy_session_handler = _busy
    follow = _event("follow up", "om_2")
    session_key = build_session_key(follow.source)
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter.handle_message(follow)

    assert queued == [follow]
    assert badges.of("om_2") == ["OneSecond"]


@pytest.mark.asyncio
async def test_stop_drops_queued_messages_and_clears_their_waiting_badge():
    from gateway.run import GatewayRunner

    adapter, badges = _adapter()
    queued = _event("waiting", "om_2")
    await adapter.mark_event_queued(queued)
    session_key = build_session_key(queued.source)
    adapter._pending_messages[session_key] = queued

    runner = object.__new__(GatewayRunner)
    runner._delivery_adapter_for = lambda source: adapter
    runner._peek_session_state = lambda key: None
    runner._overflow_queue = lambda key: []
    runner._interrupt_running_turn = lambda *a, **k: 0
    runner._drop_turn_slot = lambda *a, **k: None
    runner._thread_metadata_for_source = lambda source: None

    await GatewayRunner._interrupt_and_clear_session(
        runner, session_key, queued.source, interrupt_reason="stop", invalidation_reason="stop")

    assert badges.of("om_2") == []
    assert session_key not in adapter._pending_messages


def test_a_badge_that_cannot_be_removed_is_never_stacked_on():
    adapter = FeishuAdapter(PlatformConfig())
    badges = Badges(adapter, removable=False)
    event = _event("hi", "om_1")

    async def _run():
        await adapter.mark_event_queued(event)
        await adapter.on_processing_start(event)
        await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)

    asyncio.run(_run())
    assert badges.of("om_1") == ["OneSecond"]


def test_reactions_off_shows_no_badges(monkeypatch):
    monkeypatch.setenv("FEISHU_REACTIONS", "false")
    adapter, badges = _adapter()
    event = _event("hi", "om_1")

    async def _run():
        await adapter.mark_event_queued(event)
        await adapter.on_processing_start(event)

    asyncio.run(_run())
    assert badges.on == {}


def test_setting_the_same_badge_again_calls_feishu_once():
    adapter, badges = _adapter()
    calls = []
    add = adapter._add_reaction

    async def _counting_add(message_id, emoji):
        calls.append(emoji)
        return await add(message_id, emoji)

    adapter._add_reaction = _counting_add
    event = _event("hi", "om_1")

    async def _run():
        await adapter.on_processing_start(event)
        await adapter.on_processing_start(event)  # a runner-drained follow-up fires start again

    asyncio.run(_run())
    assert calls == ["Typing"] and badges.of("om_1") == ["Typing"]


def test_a_message_already_running_is_not_sent_back_to_waiting():
    adapter, badges = _adapter()
    event = _event("hi", "om_1")

    async def _run():
        await adapter.on_processing_start(event)
        await adapter.mark_event_queued(event)  # re-queued by a requeue path while its badge says running

    asyncio.run(_run())
    assert badges.of("om_1") == ["Typing"]
