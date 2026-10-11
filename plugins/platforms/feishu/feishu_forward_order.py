"""A merge-forward or a quoted reply reaches the agent before the messages its sender sends behind it.

Inbound events are processed concurrently. Expanding a forward awaits a message GET, a name lookup
and child-media downloads; a quoted reply awaits the quoted message, its reply chain and its media
(time-boxed, but seconds on a cold cache). The note Feishu sends after a forward ("please
summarize"), or a plain follow-up to a quoted reply, would otherwise overtake it and be answered on
its own, with the slow message arriving later as a second (interrupting) turn. While a sender's
forward or explicit quote is being processed, that sender's next messages in the same chat and topic
wait for it (bounded). A forward then opens the text batch and its note merges into it; the
transcript itself does not count toward ``text_batch_max_chars``, or the note would flush it.

A command is never held and holds nothing: ``/stop``, ``/approve`` and clarify-style control reach
the base adapter's busy guard at once. A message posted into a topic names the topic's root without
quoting it, so it holds nothing either.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, AsyncIterator

MAX_HOLD_SECONDS = 30.0  # a stuck forward must not hold the sender's later messages forever


@contextlib.asynccontextmanager
async def in_arrival_order(adapter: Any, message: Any, sender: Any) -> AsyncIterator[None]:
    """Process one inbound message after the same sender's in-flight forward or quoted reply in that chat and topic.

    Must be entered before the handler's first await: events start in arrival order, so the claim
    made here is what orders a slow message ahead of the messages that follow it.
    """
    normalized = adapter._normalize(
        getattr(message, "message_type", "") or "", getattr(message, "content", "") or "", getattr(message, "mentions", None),
    )
    if _is_command(normalized):
        yield
        return
    sender_id = getattr(sender, "sender_id", None)
    key = (
        str(getattr(message, "chat_id", "") or ""),
        str(getattr(message, "thread_id", "") or ""),
        str(getattr(sender_id, "open_id", None) or getattr(sender_id, "user_id", None) or ""),
    )
    in_flight: dict[tuple[str, str, str], asyncio.Future] = vars(adapter).setdefault("_slow_inbound_in_flight", {})
    prior = in_flight.get(key)
    mine = None
    if normalized.raw_type == "merge_forward" or _quotes_explicitly(message):
        mine = in_flight[key] = asyncio.get_running_loop().create_future()
    try:
        if prior is not None:
            await asyncio.wait({prior}, timeout=MAX_HOLD_SECONDS)
        yield
    finally:
        if mine is not None:
            mine.set_result(None)
            if in_flight.get(key) is mine:
                del in_flight[key]


def _is_command(normalized: Any) -> bool:
    """What ``_process_inbound_message`` dispatches as a command: text reading ``/…`` once self-mentions are stripped."""
    from .adapter import _strip_edge_self_mentions

    text = _strip_edge_self_mentions(normalized.text_content, normalized.mentions)
    return normalized.preferred_message_type == "text" and text.startswith("/")


def _quotes_explicitly(message: Any) -> bool:
    """A reply fetches the message it quotes; a topic post's ``root_id`` (or root ``parent_id``) is no quote."""
    from .feishu_quote import _quotes_explicitly

    return _quotes_explicitly(message)


def batch_chars(batch: Any, text: str) -> int:
    """Size of a pending text batch for the flush cap, minus the transcript of a forward that opened it.

    The first time a batch is measured nothing has been merged into it yet, so its text is still
    exactly the forward's own; that length is remembered on the batch for later merges.
    """
    message = getattr(getattr(batch.raw_message, "event", None), "message", None)
    if getattr(message, "message_type", "") != "merge_forward":
        return len(text)
    return len(text) - vars(batch).setdefault("_feishu_forward_chars", len(batch.text or ""))
