"""Text of a quoted (replied-to) Feishu message, surfaced as ``MessageEvent.reply_to_text``.

The quoted message is fetched with card bodies as raw ``json_card`` so a quoted Card JSON 2.0
card (the bot's own replies) reads as its content, and a quoted merge-forward reads as the
forwarded transcript (text only; its media is not downloaded).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


async def fetch_message_text(adapter: Any, message_id: str) -> Optional[str]:
    """Readable text of ``message_id`` (LRU-cached on the adapter); None when it can't be fetched."""
    from plugins.platforms.feishu.adapter import _FEISHU_MESSAGE_TEXT_CACHE_SIZE
    from plugins.platforms.feishu.feishu_forward import display_text, fetch_message_items, render_merge_forward

    if not adapter._client or not message_id:
        return None
    cache = adapter._message_text_cache
    if message_id in cache:
        cache.move_to_end(message_id)
        return cache[message_id]
    try:
        items = await fetch_message_items(adapter, message_id)
        if items is None:
            return None
        parent = items[0] if items else None
        msg_type = str(getattr(parent, "msg_type", "") or "")
        transcript = await render_merge_forward(adapter, message_id, items) if msg_type == "merge_forward" else None
        if transcript is not None:
            text: Optional[str] = transcript.text
        else:
            raw_content = getattr(getattr(parent, "body", None), "content", "") or ""
            normalized = adapter._normalize(msg_type, raw_content, getattr(parent, "mentions", None))
            text = display_text(normalized) or None
    except Exception:
        logger.warning("[Feishu] Failed to fetch parent message %s", message_id, exc_info=True)
        return None
    cache[message_id] = text
    while len(cache) > _FEISHU_MESSAGE_TEXT_CACHE_SIZE:
        cache.popitem(last=False)
    return text
