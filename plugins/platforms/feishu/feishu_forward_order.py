"""A merge-forward, a card, a folder or a quoted reply reaches the agent before the messages its sender sends behind it.

Inbound events are processed concurrently. Expanding a forward awaits a message GET, a name lookup
and child-media downloads; a card awaits the GET that reads its json_card; a folder awaits its listing and file downloads; a quoted reply awaits the
quoted message, its reply chain and its media (time-boxed, but seconds on a cold cache). The note
Feishu sends after a forward or a folder ("please review these"), or a plain follow-up to a quoted
reply, would otherwise overtake it and be answered on its own, with the slow message arriving later
as a second (interrupting) turn. While a sender's forward, folder or explicit quote is being
processed, that sender's next messages in the same chat and topic wait for it (bounded; downloads
stop at ``DOWNLOAD_BUDGET_SECONDS``, so the slow message is normally dispatched well inside the
hold). A forward, card or folder then opens the text batch and its note merges into it; its own text
(the transcript, the card, the folder tree) does not count toward ``text_batch_max_chars``, or the
note would flush it.

A command is never held and holds nothing: ``/stop``, ``/approve`` and clarify-style control reach
the base adapter's busy guard at once. A message posted into a topic names the topic's root without
quoting it, so it holds nothing either.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, AsyncIterator

MAX_HOLD_SECONDS = 30.0  # a stuck forward must not hold the sender's later messages forever
DOWNLOAD_BUDGET_SECONDS = 20.0  # a forward's or folder's downloads stop by then: dispatched inside the hold


@contextlib.asynccontextmanager
async def in_arrival_order(adapter: Any, message: Any, sender: Any) -> AsyncIterator[None]:
    """Process one inbound message after the same sender's in-flight forward, folder or quote in that chat and topic.

    Must be entered before the handler's first await: events start in arrival order, so the claim
    made here is what orders a slow message ahead of the messages that follow it.
    """
    normalized = _normalize(adapter, message)
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
    if _expands(normalized) or _quotes_explicitly(message):
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


def _normalize(adapter: Any, message: Any) -> Any:
    return adapter._normalize(
        getattr(message, "message_type", "") or "", getattr(message, "content", "") or "", getattr(message, "mentions", None),
    )


def _expands(normalized: Any) -> bool:
    """A forward or a card is read by GET, a folder (native or in a post) listed and downloaded: all after arrival."""
    return normalized.raw_type in {"merge_forward", "interactive", "card"} or bool(normalized.folder_refs)


def _is_command(normalized: Any) -> bool:
    """What ``_process_inbound_message`` dispatches as a command: text reading ``/…`` once self-mentions are stripped."""
    from .adapter import _strip_edge_self_mentions

    text = _strip_edge_self_mentions(normalized.text_content, normalized.mentions)
    return normalized.preferred_message_type == "text" and text.startswith("/")


def _quotes_explicitly(message: Any) -> bool:
    """A reply fetches the message it quotes; a topic post's ``root_id`` (or root ``parent_id``) is no quote."""
    from .feishu_quote import _quotes_explicitly

    return _quotes_explicitly(message)


def batch_chars(adapter: Any, batch: Any, text: str) -> int:
    """Size of a pending text batch for the flush cap, minus the expansion of a forward, card or folder that opened it.

    The first time a batch is measured nothing has been merged into it yet, so its text is still
    exactly the opener's own; that length is remembered on the batch for later merges.
    """
    message = getattr(getattr(batch.raw_message, "event", None), "message", None)
    if message is None or not _expands(_normalize(adapter, message)):
        return len(text)
    return len(text) - vars(batch).setdefault("_feishu_expansion_chars", len(batch.text or ""))
