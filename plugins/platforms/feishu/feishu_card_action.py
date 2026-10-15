"""Clicks on this bot's Feishu cards (other than its own approval / update prompts), routed to the agent.

A click becomes the text ``card:<tag> <json>`` (the action's value, form values or pick), from the clicker,
quoting the card, in the card's chat and topic, the way a reaction on the bot's message does. The callback
names the card (``open_message_id``) but not its topic, so the card is fetched once for its ``thread_id``;
when that fails the click still reaches the agent, outside any topic. Its ``message_id`` names the click,
``<card>:card:<token>``, for the same reason a reaction's does (see :mod:`feishu_reaction`).
"""

from __future__ import annotations

import json
import logging
import uuid
from types import SimpleNamespace
from typing import Any

from gateway.platforms.event import MessageType

from .feishu_reaction import _event_chat_type

logger = logging.getLogger(__name__)

_CARD_ID_MARK = ":card:"
# What the user did: a button's value, a submitted form's values, a select / picker choice, a typed input.
_ACTION_FIELDS = ("name", "value", "form_value", "option", "options", "input_value", "checked")


async def handle_card_action_event(adapter: Any, data: Any) -> None:
    event = getattr(data, "event", None)
    token = str(getattr(event, "token", "") or "")
    if token and adapter._is_card_action_duplicate(token):
        logger.debug("[Feishu] Dropping duplicate card action token: %s", token)
        return
    context = getattr(event, "context", None)
    chat_id = str(getattr(context, "open_chat_id", "") or "")
    operator = getattr(event, "operator", None)
    open_id = str(getattr(operator, "open_id", "") or "")
    if not chat_id or not open_id:
        logger.debug("[Feishu] Card action missing chat_id or operator open_id, dropping")
        return
    action = getattr(event, "action", None)
    tag = str(getattr(action, "tag", "") or "button")
    card_id = str(getattr(context, "open_message_id", "") or "")
    card = await _card_message(adapter, card_id)
    quote = await adapter._fetch_quoted_reply(card_id, media=False)  # cached by the fetch above
    chat_info = await adapter.get_chat_info(chat_id)  # cached; reused by _dispatch_synthetic_event
    logger.info("[Feishu] Routing card action %r from %s on card %s in %s", tag, open_id, card_id, chat_id)
    await adapter._dispatch_synthetic_event(
        text=_click_text(tag, action), message_type=MessageType.TEXT, chat_id=chat_id,
        sender_id=SimpleNamespace(open_id=open_id, user_id=getattr(operator, "user_id", None) or None,
                                  union_id=getattr(operator, "union_id", None) or None),
        event_chat_type=_event_chat_type(chat_info), raw_message=data,
        # No card id (a click outside a chat card) maps back to no reply target: the answer is a new message.
        message_id=f"{card_id}{_CARD_ID_MARK}{token or uuid.uuid4().hex}",
        thread_id=str(getattr(card, "thread_id", "") or "") or None,
        reply_to_message_id=card_id or None, reply_to_text=quote.text, reply_to_is_own_message=quote.is_own,
    )


async def _card_message(adapter: Any, card_id: str) -> Any:
    """The clicked card as ``GET im/v1/messages`` reports it (its topic), cached for the quote; None when unreadable."""
    from .feishu_forward import fetch_message_items
    from .feishu_quote import cache_node

    if not card_id or not adapter._client:
        return None
    try:
        items = await fetch_message_items(adapter, card_id)
    except Exception:
        logger.debug("[Feishu] Failed to fetch clicked card %s", card_id, exc_info=True)
        return None
    if not items:
        return None
    try:
        await cache_node(adapter, card_id, items)
    except Exception:  # the quote then fetches it again
        logger.debug("[Feishu] Failed to read clicked card %s as a quote", card_id, exc_info=True)
    return items[0]


def _click_text(tag: str, action: Any) -> str:
    fields = {name: value for name in _ACTION_FIELDS
              if (value := getattr(action, name, None)) not in (None, "", [], {})}
    try:
        payload = json.dumps(fields, ensure_ascii=False) if fields else ""
    except (TypeError, ValueError):
        payload = ""
    return f"card:{tag} {payload}".rstrip()
