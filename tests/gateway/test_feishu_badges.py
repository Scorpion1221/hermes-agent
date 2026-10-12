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


def _busy(adapter, outcome):
    """A busy session handler that settles each follow-up the way the runner would."""
    async def _handler(event, session_key):
        if outcome == "queue":
            adapter._pending_messages[session_key] = event
            event._gateway_accepted = True
            await adapter._run_processing_hook("mark_event_queued", event)
        elif outcome == "steer":
            await adapter._run_processing_hook("mark_event_running", event)
        return True  # handled: queued, steered, or dropped (unauthorized, approval answer, cap)
    return _handler


async def _arrive_busy(adapter, follow, outcome, *, text_mode="interrupt"):
    adapter._busy_text_mode = text_mode
    adapter._message_handler = lambda event: asyncio.sleep(0)
    adapter._busy_session_handler = _busy(adapter, outcome) if outcome else None
    session_key = build_session_key(follow.source)
    adapter._active_sessions[session_key] = asyncio.Event()
    await adapter.handle_message(follow)
    return session_key


@pytest.mark.asyncio
async def test_a_follow_up_queued_behind_a_busy_turn_waits_with_one_second():
    adapter, badges = _adapter()
    await _arrive_busy(adapter, _event("follow up", "om_2"), "queue")
    assert badges.of("om_2") == ["OneSecond"]


@pytest.mark.asyncio
async def test_a_follow_up_the_busy_session_did_not_queue_shows_no_waiting_badge():
    """An approval answer, an unauthorized sender, the bot-loop guard, a full queue: handled without a
    later turn, so nothing ever clears a "waiting" badge — none may be shown."""
    adapter, badges = _adapter()
    await _arrive_busy(adapter, _event("yes", "om_2"), "drop")
    assert badges.of("om_2") == []


@pytest.mark.asyncio
async def test_a_steered_follow_up_shows_typing_and_ends_with_the_turn():
    adapter, badges = _adapter()
    turn = _event("first", "om_1")
    follow = _event("also this", "om_2")
    await adapter.on_processing_start(turn)
    await _arrive_busy(adapter, follow, "steer")
    turn.absorb_reply_expected(follow)  # the runner folds a steered message into the running turn
    assert badges.of("om_2") == ["Typing"]

    await adapter.on_processing_complete(turn, ProcessingOutcome.SUCCESS)
    assert badges.of("om_1") == [] and badges.of("om_2") == []


@pytest.mark.asyncio
async def test_without_a_busy_handler_the_base_queue_badges_what_it_stored():
    adapter, badges = _adapter()
    session_key = await _arrive_busy(adapter, _event("follow up", "om_2"), None)
    assert adapter._pending_messages[session_key].message_id == "om_2"
    assert badges.of("om_2") == ["OneSecond"]


@pytest.mark.asyncio
async def test_stop_clears_the_waiting_badge_of_a_burst_still_in_the_text_debounce():
    adapter, badges = _adapter()
    adapter._busy_text_debounce_seconds = 5.0  # keep the burst buffered
    follow = _event("still typing", "om_2")
    session_key = await _arrive_busy(adapter, follow, None, text_mode="queue")
    assert badges.of("om_2") == ["OneSecond"]

    async def _dispatch(event, key, cmd):
        return None

    adapter._dispatch_active_session_command = _dispatch
    await adapter.handle_message(_event("/stop", "om_3"))

    assert session_key not in adapter._text_debounce
    assert badges.of("om_2") == []


def test_a_batched_event_merged_into_a_turn_badges_every_message_of_the_batch():
    """A text batch (om_a1 + om_a2) folded into a running turn: the turn's end clears both."""
    from plugins.platforms.feishu.feishu_badges import record_merged

    adapter, badges = _adapter()
    turn, batch = _event("turn", "om_t"), _event("a1", "om_a1")
    record_merged(batch, _event("a2", "om_a2"))
    batch.message_id = "om_a2"  # a batch advances to its newest message
    turn.absorb_reply_expected(batch)

    async def _run():
        await adapter.on_processing_start(turn)
        assert all(badges.of(m) == ["Typing"] for m in ("om_t", "om_a1", "om_a2"))
        await adapter.on_processing_complete(turn, ProcessingOutcome.SUCCESS)

    asyncio.run(_run())
    assert all(badges.of(m) == [] for m in ("om_t", "om_a1", "om_a2"))


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
