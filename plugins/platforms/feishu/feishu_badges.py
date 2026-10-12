"""Per-message status badges on Feishu: every inbound message shows where its turn is.

A message waiting behind a busy turn shows ``OneSecond``; when its turn runs it shows ``Typing``; when the
turn ends the badge goes (``CrossMark`` on failure); a message that will never be answered (/stop, /new)
loses its badge. A batched event (text burst, media album) stands for several Feishu messages, and each of
them carries the badge.

The gateway reaches these through the processing hooks (``on_processing_start`` / ``_complete``, also
fired for runner-drained follow-ups) and through the optional ``mark_event_queued`` hook it calls when a
busy session accepts a message for later.
"""

from __future__ import annotations

from typing import Any, Optional

QUEUED = "OneSecond"
RUNNING = "Typing"
FAILED = "CrossMark"
_MERGED_IDS_KEY = "feishu_message_ids"
_CACHE_SIZE = 1024


def message_ids(event: Any) -> list[str]:
    """Every Feishu message id ``event`` stands for, oldest first: the adapter's own text/media batches,
    then the gateway's busy-slot merges (``absorbed_message_ids``), then its current id."""
    from .feishu_reaction import feishu_message_id

    raw = [*((getattr(event, "metadata", None) or {}).get(_MERGED_IDS_KEY) or ()),
           *(getattr(event, "absorbed_message_ids", None) or ()), getattr(event, "message_id", None)]
    ids: list[str] = []
    for value in raw:
        message_id = feishu_message_id(value) if value else None
        if message_id and message_id not in ids:
            ids.append(message_id)
    return ids


def record_merged(existing: Any, incoming: Any) -> None:
    """Fold ``incoming``'s ids into a batched ``existing`` before it advances to the newest message."""
    ids = message_ids(existing)
    ids += [i for i in message_ids(incoming) if i not in ids]
    if existing.metadata is None:
        existing.metadata = {}
    existing.metadata[_MERGED_IDS_KEY] = ids


async def _set(adapter: Any, message_id: str, emoji: Optional[str]) -> bool:
    """Show ``emoji`` on ``message_id`` (None clears it). One badge per message; False when the previous
    badge could not be removed (it is kept rather than stacking a contradictory second one)."""
    cache = adapter._pending_processing_reactions
    current = cache.get(message_id)
    if current and current[0] == emoji:
        return True
    if current:
        if not await adapter._remove_reaction(message_id, current[1]):
            return False
        cache.pop(message_id, None)
    if emoji:
        reaction_id = await adapter._add_reaction(message_id, emoji)
        if reaction_id:
            cache[message_id] = (emoji, reaction_id)
            cache.move_to_end(message_id)
            while len(cache) > _CACHE_SIZE:
                cache.popitem(last=False)
    return True


async def mark(adapter: Any, event: Any, emoji: Optional[str], *, only_unbadged: bool = False) -> list[str]:
    """Badge every message of ``event``; returns the ids whose previous badge was cleared."""
    if not adapter._reactions_enabled():
        return []
    cleared = []
    for message_id in message_ids(event):
        if only_unbadged and message_id in adapter._pending_processing_reactions:
            continue
        if await _set(adapter, message_id, emoji):
            cleared.append(message_id)
    return cleared


async def finish(adapter: Any, event: Any, failed: bool) -> None:
    """The turn for ``event`` ended: clear its badges, and mark failure where one was cleared."""
    for message_id in await mark(adapter, event, None):
        if failed:
            await adapter._add_reaction(message_id, FAILED)
