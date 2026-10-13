"""The name shown on a resolved Feishu approval / update-prompt card."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig


def _adapter():
    from plugins.platforms.feishu.adapter import FeishuAdapter

    return FeishuAdapter(PlatformConfig())


class _Response:
    def __init__(self):
        self.card = None


class _Card:
    def __init__(self):
        self.type = self.data = None


@pytest.fixture
def card_types(monkeypatch):
    import plugins.platforms.feishu.adapter as feishu_module

    monkeypatch.setattr(feishu_module, "P2CardActionTriggerResponse", _Response)
    monkeypatch.setattr(feishu_module, "CallBackCard", _Card)


async def _settle():
    """Return once every other task on this loop is done (including any those tasks start)."""
    me = asyncio.current_task()
    while others := asyncio.all_tasks() - {me}:
        await asyncio.wait(others)


class _LoopThread:
    """A real adapter loop on its own thread; the test thread plays the SDK callback thread."""

    def __enter__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        return self.loop

    def __exit__(self, *_exc):
        # Let work the callback scheduled (approval resolution) finish before the loop stops.
        asyncio.run_coroutine_threadsafe(_settle(), self.loop).result(timeout=5)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
        self.loop.close()


def _approval_click(adapter, *, open_id="ou_clicker", user_id=None):
    adapter._approval_state[1] = {"session_key": "sess", "message_id": "om_card", "chat_id": "oc_chat"}
    return SimpleNamespace(event=SimpleNamespace(
        token="tok", context=SimpleNamespace(open_chat_id="oc_chat"),
        operator=SimpleNamespace(open_id=open_id, user_id=user_id),
        action=SimpleNamespace(tag="button", value={"hermes_action": "approve_once", "approval_id": 1}),
    ))


def _card_text(response):
    return response.card.data["elements"][0]["content"]


def _resolve_without_running(adapter):
    async def _noop(*_args, **_kwargs):
        return None

    adapter._resolve_approval = _noop


def test_uncached_operator_name_is_looked_up_for_the_resolved_card(card_types):
    adapter = _adapter()
    _resolve_without_running(adapter)
    lookups = []

    async def _lookup(sender_id, *, is_bot=False):
        lookups.append((sender_id, threading.current_thread()))
        return "Charlie"

    adapter._resolve_sender_name_from_api = _lookup
    with _LoopThread() as loop:
        adapter._loop = loop
        response = adapter._on_card_action_trigger(_approval_click(adapter))

    assert "Charlie" in _card_text(response)
    assert "ou_clicker" not in _card_text(response)
    assert lookups and lookups[0][0] == "ou_clicker"
    assert lookups[0][1] is not threading.current_thread()  # ran on the adapter loop, not the SDK thread


def test_cached_name_under_the_operators_user_id_is_used_without_a_lookup(card_types):
    adapter = _adapter()
    _resolve_without_running(adapter)
    adapter._sender_name_cache["u_clicker"] = ("Dana", time.time() + 600)
    adapter._resolve_sender_name_from_api = AsyncMock(return_value="wrong")
    with _LoopThread() as loop:
        adapter._loop = loop
        response = adapter._on_card_action_trigger(_approval_click(adapter, user_id="u_clicker"))

    assert "Dana" in _card_text(response)
    adapter._resolve_sender_name_from_api.assert_not_awaited()


def test_slow_name_lookup_falls_back_within_the_card_deadline(card_types, monkeypatch):
    import plugins.platforms.feishu.feishu_card_operator as operator_module

    monkeypatch.setattr(operator_module, "_LOOKUP_BUDGET_SECONDS", 0.2)
    adapter = _adapter()
    _resolve_without_running(adapter)
    release, finished = asyncio.Event(), threading.Event()

    async def _hanging_lookup(sender_id, *, is_bot=False):
        # Overruns the budget: answers once the first card is back (or after 5 s, so a callback that
        # ignored its budget fails the elapsed check instead of hanging the test).
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(release.wait(), timeout=5)
        adapter._sender_name_cache[sender_id] = ("Erin", time.time() + 600)
        finished.set()
        return "Erin"

    adapter._resolve_sender_name_from_api = _hanging_lookup
    with _LoopThread() as loop:
        adapter._loop = loop
        started = time.monotonic()
        try:
            first = adapter._on_card_action_trigger(_approval_click(adapter))
            elapsed = time.monotonic() - started
        finally:
            loop.call_soon_threadsafe(release.set)
        assert finished.wait(timeout=5)  # the overrunning lookup still completes and warms the cache
        second = adapter._on_card_action_trigger(_approval_click(adapter))

    assert elapsed < 2.5, elapsed  # Feishu's card-callback deadline is ~3 s
    assert "ou_clicker" in _card_text(first)
    assert "Erin" in _card_text(second)


def test_failed_name_lookup_shows_the_open_id(card_types):
    adapter = _adapter()
    _resolve_without_running(adapter)
    adapter._resolve_sender_name_from_api = AsyncMock(return_value=None)
    with _LoopThread() as loop:
        adapter._loop = loop
        response = adapter._on_card_action_trigger(_approval_click(adapter))

    assert "ou_clicker" in _card_text(response)


@pytest.mark.asyncio
async def test_callback_on_the_adapter_loop_never_waits_for_a_lookup(card_types):
    """Webhook transport: the callback runs on the adapter loop; waiting there would deadlock."""
    adapter = _adapter()
    _resolve_without_running(adapter)
    adapter._loop = asyncio.get_running_loop()
    adapter._resolve_sender_name_from_api = AsyncMock(return_value="never awaited")

    started = time.monotonic()
    response = adapter._on_card_action_trigger(_approval_click(adapter))

    assert time.monotonic() - started < 1.0
    assert "ou_clicker" in _card_text(response)
    adapter._resolve_sender_name_from_api.assert_not_awaited()
    await asyncio.sleep(0)  # let the scheduled resolution run to completion


def test_update_prompt_card_uses_the_looked_up_name(card_types):
    adapter = _adapter()

    async def _noop(*_args, **_kwargs):
        return None

    adapter._resolve_update_prompt = _noop
    adapter._resolve_sender_name_from_api = AsyncMock(return_value="Frank")
    adapter._update_prompt_state[3] = {"session_key": "sess", "message_id": "om_card", "chat_id": "oc_chat"}
    data = SimpleNamespace(event=SimpleNamespace(
        token="tok", context=SimpleNamespace(open_chat_id="oc_chat"),
        operator=SimpleNamespace(open_id="ou_clicker", user_id=None),
        action=SimpleNamespace(tag="button", value={"hermes_update_prompt_action": "y", "update_prompt_id": 3}),
    ))
    with _LoopThread() as loop:
        adapter._loop = loop
        response = adapter._on_card_action_trigger(data)

    assert "Frank" in _card_text(response)
