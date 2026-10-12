"""Busy-text debounce for ``BasePlatformAdapter``: one buffered burst per session, flushed as one turn.

Moved out of ``gateway/platforms/base.py``. ``base`` helpers (``merge_pending_message_event``,
``_append_text``) are reached through the module at call time, so ``patch("gateway.platforms.base.X")``
keeps intercepting them and the import cycle stays lazy.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from typing import Optional

from gateway.platforms.event import MessageEvent, MessageType

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.platforms.base")


def _base():
    return sys.modules["gateway.platforms.base"]


@dataclass
class TextDebounceState:
    event: "MessageEvent"
    task: asyncio.Task | None
    first_ts: float
    last_ts: float

    def cancel_timer(self, *, unless: asyncio.Task | None = None) -> None:
        """Cancel the pending flush timer (if live and not ``unless``)."""
        if self.task is not None and self.task is not unless and not self.task.done():
            self.task.cancel()


class TextDebounceMixin:
    """Queue-mode busy text: buffer a sender's burst, flush it as one pending turn."""

    def _text_debounce_store(self) -> dict[str, TextDebounceState]:
        return _base()._lazy_attr(self, "_text_debounce", dict)

    def _is_queue_text_debounce_candidate(self, event: MessageEvent) -> bool:
        """Return True for normal text eligible for queue-mode debounce."""
        result = (
            getattr(self, "_busy_text_mode", "interrupt") == "queue"
            and event.message_type == MessageType.TEXT and not getattr(event, "internal", False)
            and not event.is_command() and bool((event.text or "").strip()))
        if result:
            logger.debug("[%s] Queue-text debounce candidate accepted: session=%s text_len=%d",
                         self.name, getattr(event, "session_key", "?"), len(event.text or ""))
        return result

    def _can_merge_text_debounce_events(self, existing: MessageEvent, event: MessageEvent) -> bool:
        """Return True when two text debounce events came from the same sender."""

        def _identity(candidate: MessageEvent) -> tuple[str, ...] | None:
            source = getattr(candidate, "source", None)
            if source is None:
                return None
            platform = _base()._platform_name(getattr(source, "platform", None))
            sender = getattr(source, "user_id_alt", None) or getattr(source, "user_id", None)
            if sender:
                return (platform, str(sender))
            if getattr(source, "chat_type", None) in {"dm", "private"} and getattr(source, "chat_id", None):
                return (platform, "dm", str(source.chat_id))
            return None
        existing_sender = _identity(existing)
        return existing_sender is not None and existing_sender == _identity(event)

    def _text_debounce_delay(self, session_key: str) -> float:
        """Return bounded busy-text debounce delay for ``session_key``."""
        state = self._text_debounce_store().get(session_key)
        if state is None:
            return 0.0
        deadline = min(state.last_ts + self._busy_text_debounce_seconds,
                       state.first_ts + self._busy_text_hard_cap_seconds)
        return max(0.0, deadline - time.monotonic())

    async def _queue_text_debounce(self, session_key: str, event: MessageEvent) -> None:
        """Buffer normal queue-mode busy text and schedule a bounded flush."""
        store = self._text_debounce_store()
        state = store.get(session_key)
        if state is not None and not self._can_merge_text_debounce_events(state.event, event):
            # Preserve sender attribution: flush the buffer as the next turn, new sender starts
            # fresh.
            await self._flush_text_debounce_now(session_key)
            state = store.get(session_key)
            if state is not None and not self._can_merge_text_debounce_events(state.event, event):
                existing_pending = self._pending_messages.get(session_key)
                if existing_pending is not None and self._can_merge_text_debounce_events(existing_pending, event):
                    _base().merge_pending_message_event(self._pending_messages, session_key, event, merge_text=True)
                return
        now = time.monotonic()
        if state is None:
            state = TextDebounceState(event=event, task=None, first_ts=now, last_ts=now)
            store[session_key] = state
        else:
            if event.text:
                state.event.text = _base()._append_text(state.event.text, event.text)
            state.event.absorb_reply_expected(event)
            latest_message_id = getattr(event, "message_id", None)
            latest_anchor = latest_message_id or getattr(event, "reply_to_message_id", None)
            if latest_message_id is not None:
                state.event.message_id = str(latest_message_id)
            if latest_anchor is not None and hasattr(state.event, "reply_to_message_id"):
                state.event.reply_to_message_id = str(latest_anchor)
            state.last_ts = now
        state.cancel_timer()
        delay = self._text_debounce_delay(session_key)
        state.task = asyncio.create_task(self._flush_text_debounce(session_key, delay))

    async def _flush_text_debounce(self, session_key: str, delay: float) -> None:
        """Timer task that flushes the debounced text buffer."""
        try:
            await asyncio.sleep(delay)
            await self._flush_text_debounce_now(session_key)
        except asyncio.CancelledError:
            return
        finally:
            current = asyncio.current_task()
            state = self._text_debounce_store().get(session_key)
            if state is not None and state.task is current:
                state.task = None

    async def _flush_text_debounce_now(self, session_key: str) -> bool:
        """Force-flush one debounced busy-text burst into the pending slot."""
        store = self._text_debounce_store()
        state = store.get(session_key)
        if state is None:
            return False
        state.cancel_timer(unless=asyncio.current_task())
        state.task = None
        pending = self._pending_messages.get(session_key)
        # Another sender's message holds the slot (shared group/topic session): the burst becomes its own
        # turn in the runner's FIFO, or a third sender would find no place and be dropped.
        enqueue = getattr(getattr(self, "gateway_runner", None), "_enqueue_fifo", None)
        foreign = pending is not None and not self._can_merge_text_debounce_events(pending, state.event)
        if foreign and not callable(enqueue):
            return False
        store.pop(session_key, None)
        if foreign:
            enqueue(session_key, state.event, self)
        else:
            _base().merge_pending_message_event(self._pending_messages, session_key, state.event, merge_text=True)
        return True

    def _discard_text_debounce(self, session_key: str) -> Optional[MessageEvent]:
        """Cancel and drop pending text debounce state for control commands; returns the dropped
        burst so an async caller can clear its "waiting" marker."""
        state = self._text_debounce_store().pop(session_key, None)
        if state is None:
            return None
        state.cancel_timer()
        return state.event
