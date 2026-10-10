"""A merge-forward and the note sent right behind it reach the agent as one turn, forward first.

Inbound events are processed concurrently, and expanding a forward awaits a message GET, a name
lookup and child-media downloads. The note Feishu sends after a forward ("please summarize") would
otherwise overtake it and be answered on its own, with the transcript arriving later as a second
(interrupting) turn. While a sender's forward is being processed, that sender's next messages in the
same chat wait for it (bounded), so the forward opens the text batch and the note merges into it;
the transcript itself does not count toward ``text_batch_max_chars``, or the note would flush it.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, AsyncIterator

MAX_HOLD_SECONDS = 30.0  # a stuck forward must not hold the sender's later messages forever


@contextlib.asynccontextmanager
async def in_arrival_order(adapter: Any, message: Any, sender: Any) -> AsyncIterator[None]:
    """Process one inbound message after the same sender's in-flight forward in that chat.

    Must be entered before the handler's first await: events start in arrival order, so the claim
    made here is what orders a forward ahead of the messages that follow it.
    """
    sender_id = getattr(sender, "sender_id", None)
    key = (
        str(getattr(message, "chat_id", "") or ""),
        str(getattr(sender_id, "open_id", None) or getattr(sender_id, "user_id", None) or ""),
    )
    in_flight: dict[tuple[str, str], asyncio.Future] = vars(adapter).setdefault("_forwards_in_flight", {})
    prior = in_flight.get(key)
    mine = None
    if getattr(message, "message_type", "") == "merge_forward":
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


def batch_chars(batch: Any, text: str) -> int:
    """Size of a pending text batch for the flush cap, minus the transcript of a forward that opened it.

    The first time a batch is measured nothing has been merged into it yet, so its text is still
    exactly the forward's own; that length is remembered on the batch for later merges.
    """
    message = getattr(getattr(batch.raw_message, "event", None), "message", None)
    if getattr(message, "message_type", "") != "merge_forward":
        return len(text)
    return len(text) - vars(batch).setdefault("_feishu_forward_chars", len(batch.text or ""))
