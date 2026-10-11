"""User reactions on this bot's own Feishu messages, routed as synthetic replies to that message.

A reaction becomes the text ``reaction:<added|removed>:<EMOJI>``, quoting the reacted message, in that
message's topic. Its ``message_id`` names the reaction, not the reacted message: the gateway keys its
inbound transcript dedupe, input ownership and delivery ledger on that id, so a second reaction on one
message must not read as a replay of the first. Wherever the adapter hands an inbound id to Feishu (the
reply target of every send, the processing badge) it maps it back with :func:`feishu_message_id`.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from gateway.platforms.event import MessageType

logger = logging.getLogger(__name__)

_REACTION_ID_MARK = ":reaction:"  # Feishu message ids never contain ':'


def feishu_message_id(message_id: Optional[str]) -> Optional[str]:
    """The Feishu message an inbound id names: the reacted message for a reaction, else the id itself."""
    return message_id.partition(_REACTION_ID_MARK)[0] if message_id else message_id


async def handle_reaction_event(adapter: Any, event_type: str, data: Any) -> None:
    """Fetch the reacted-to message; if this bot sent it, emit a synthetic reply to it."""
    if not adapter._client:
        return
    event = getattr(data, "event", None)
    message_id = str(getattr(event, "message_id", "") or "")
    msg = await _own_message(adapter, message_id) if message_id else None
    chat_id = str(getattr(msg, "chat_id", "") or "")
    if not chat_id:
        return
    emoji_type = str(getattr(getattr(event, "reaction_type", None), "emoji_type", "") or "UNKNOWN")
    action = "added" if "created" in event_type else "removed"
    # The reacted message as a quote (its reply chain, cards as their text); a reaction attaches no media.
    reply_to_text = (await adapter._fetch_quoted_reply(message_id, media=False)).text
    chat_info = await adapter.get_chat_info(chat_id)  # cached; reused by _dispatch_synthetic_event
    logger.info("[Feishu] Routing reaction %s:%s on bot message %s as synthetic event", action, emoji_type, message_id)
    await adapter._dispatch_synthetic_event(
        text=f"reaction:{action}:{emoji_type}", message_type=MessageType.TEXT, chat_id=chat_id,
        sender_id=getattr(event, "user_id", None), event_chat_type=_event_chat_type(chat_info),
        raw_message=data, message_id=_reaction_id(message_id, data),
        thread_id=str(getattr(msg, "thread_id", "") or "") or None,
        reply_to_message_id=message_id, reply_to_text=reply_to_text, reply_to_is_own_message=True,
    )


async def _own_message(adapter: Any, message_id: str) -> Any:
    """The reacted-to message when this bot sent it, else None. Fetched once, as the quote reads it: the
    node is cached, so quoting the message below costs no second GET."""
    from .feishu_forward import fetch_message_items
    from .feishu_quote import cache_node

    try:
        items = await fetch_message_items(adapter, message_id)
    except Exception:
        logger.debug("[Feishu] Failed to fetch message for reaction routing", exc_info=True)
        return None
    msg = items[0] if items else None
    # GET im/v1/messages reports sender.id=app_id for bot messages — peer bots share sender_type="app"
    # with us but differ on app_id. Only route our own messages.
    if msg is None or str(getattr(getattr(msg, "sender", None), "id", "") or "") != adapter._app_id:
        return None
    try:
        await cache_node(adapter, message_id, items)
    except Exception:  # the quote then fetches it again
        logger.debug("[Feishu] Failed to read reacted message %s as a quote", message_id, exc_info=True)
    return msg


def _event_chat_type(chat_info: dict[str, Any]) -> str:
    """The ``chat_type`` an inbound message in this chat carries, so the reaction joins that chat's (topic's)
    session: the reacted message's GET item has no chat type, and chat.get's ``chat_type`` is a group's
    visibility (private/public), while its ``chat_mode`` is p2p, group or topic. Unknown reads as a DM."""
    mode = chat_info.get("chat_mode")
    return "group" if mode and mode != "p2p" else "p2p"


def _reaction_id(message_id: str, data: Any) -> str:
    """Inbound id of one reaction, built on Feishu's event id so a redelivered event keeps it."""
    event_id = str(getattr(getattr(data, "header", None), "event_id", "") or "")
    return f"{message_id}{_REACTION_ID_MARK}{event_id or uuid.uuid4().hex}"
