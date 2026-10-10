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
    reply_to_text = await adapter._fetch_message_text(message_id)
    logger.info("[Feishu] Routing reaction %s:%s on bot message %s as synthetic event", action, emoji_type, message_id)
    await adapter._dispatch_synthetic_event(
        text=f"reaction:{action}:{emoji_type}", message_type=MessageType.TEXT, chat_id=chat_id,
        sender_id=getattr(event, "user_id", None), event_chat_type=str(getattr(msg, "chat_type", "p2p") or "p2p"),
        raw_message=data, message_id=_reaction_id(message_id, data),
        thread_id=str(getattr(msg, "thread_id", "") or "") or None,
        reply_to_message_id=message_id, reply_to_text=reply_to_text, reply_to_is_own_message=True,
    )


async def _own_message(adapter: Any, message_id: str) -> Any:
    """The reacted-to message when this bot sent it, else None."""
    try:
        request = adapter._build_get_message_request(message_id)
        response = await adapter._run_blocking(adapter._client.im.v1.message.get, request)
    except Exception:
        logger.debug("[Feishu] Failed to fetch message for reaction routing", exc_info=True)
        return None
    items = getattr(getattr(response, "data", None), "items", None) if adapter._response_succeeded(response) else None
    msg = items[0] if items else None
    # GET im/v1/messages reports sender.id=app_id for bot messages — peer bots share sender_type="app"
    # with us but differ on app_id. Only route our own messages.
    if msg is None or str(getattr(getattr(msg, "sender", None), "id", "") or "") != adapter._app_id:
        return None
    return msg


def _reaction_id(message_id: str, data: Any) -> str:
    """Inbound id of one reaction, built on Feishu's event id so a redelivered event keeps it."""
    event_id = str(getattr(getattr(data, "header", None), "event_id", "") or "")
    return f"{message_id}{_REACTION_ID_MARK}{event_id or uuid.uuid4().hex}"
