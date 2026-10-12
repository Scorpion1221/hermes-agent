"""Feishu CardKit: the streaming card API a reply streams into (``cardkit/v1``).

A streaming reply is one card entity created with ``streaming_mode`` on and sent as an ``interactive``
message that references it. Text streams into its markdown element with ``card_element.content`` (each
call carries the whole text, so a skipped call is repaired by the next), and the card is sealed by a full
``card.update`` that turns ``streaming_mode`` off and replaces the body with the final text. Every write
carries a strictly increasing ``sequence``.

Feishu closes a card's streaming window after about ten minutes: the next element write fails with
200850, later ones with 300309. The card stays valid, so it can still be sealed with a full update.

Bodies are rendered through ``feishu_card_render`` at this boundary only; callers keep raw text.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

STREAMING_ELEMENT_ID = "streaming_content"
LOADING_ELEMENT_ID = "streaming_loading"
# The animated "working" icon under a live card (an image uploaded to Feishu by the fork in 2026).
LOADING_ICON_KEY = "img_v3_02vb_496bec09-4b43-4773-ad6b-0cdd103cd2bg"

RATE_LIMITED = 230020  # a skipped frame; the next one carries the whole text again
STREAMING_TIMEOUT = 200850  # the streaming window closed (about ten minutes after create)
STREAMING_CLOSED = 300309  # every write after the window closed
STREAM_EXPIRED_CODES = frozenset({STREAMING_TIMEOUT, STREAMING_CLOSED})
SEQUENCE_CONFLICT = 300317  # this sequence was already used
INVALID_USER_RESOURCE = 100290  # a mention names a user Feishu cannot resolve

# One markdown element holds at most this many characters in practice; the reply rolls over before it.
MAX_ELEMENT_CHARS = 30_000


@dataclass
class CallResult:
    ok: bool
    code: Optional[int] = None
    msg: str = ""

    @property
    def expired(self) -> bool:
        return self.code in STREAM_EXPIRED_CODES


def _result(response: Any) -> CallResult:
    if response is not None and getattr(response, "success", lambda: False)():
        return CallResult(ok=True)
    raw = getattr(response, "code", None)
    try:
        code = int(raw) if raw is not None else None
    except (TypeError, ValueError):
        code = None
    return CallResult(ok=False, code=code, msg=str(getattr(response, "msg", "") or ""))


def streaming_card_body() -> dict:
    """A new card: an empty markdown element the text streams into, and the loading icon below it."""
    return {
        "schema": "2.0",
        "config": {"streaming_mode": True, "summary": {"content": "..."}},
        "body": {"elements": [
            {"tag": "markdown", "content": "", "text_align": "left", "text_size": "normal_v2",
             "element_id": STREAMING_ELEMENT_ID},
            {"tag": "markdown", "content": " ", "element_id": LOADING_ELEMENT_ID,
             "icon": {"tag": "custom_icon", "img_key": LOADING_ICON_KEY, "size": "16px 16px"}},
        ]},
    }


def final_card_body(text: str, *, footer: str = "") -> dict:
    """A sealed card: the final text, a small footer line when given, no loading icon, streaming off."""
    from .feishu_card_render import render_markdown_for_card

    elements: list[dict] = [{"tag": "markdown", "content": render_markdown_for_card(text),
                             "text_align": "left", "text_size": "normal_v2"}]
    if footer:
        elements.append({"tag": "markdown", "content": footer, "text_size": "notation", "text_align": "left"})
    return {"schema": "2.0", "config": {"streaming_mode": False}, "body": {"elements": elements}}


def card_message_content(card_id: str) -> str:
    """The ``interactive`` message content that shows card entity ``card_id``."""
    return json.dumps({"type": "card", "data": {"card_id": card_id}}, ensure_ascii=False)


def _build(module: str, name: str, **fields: Any) -> Any:
    """``lark_oapi.api.cardkit.v1.model.<module>.<name>`` built with ``fields``; a SimpleNamespace when the
    SDK is not installed, like the facade's ``_sdk_build`` (the requests are only sent once it is)."""
    from . import adapter as facade

    try:
        import importlib

        request_cls = getattr(importlib.import_module(f"lark_oapi.api.cardkit.v1.model.{module}"), name)
    except ImportError:
        request_cls = None
    return facade._sdk_build(request_cls, **fields)


async def create_card(adapter: Any) -> Optional[str]:
    """Create a streaming card entity; its ``card_id``, or None when Feishu refused."""
    body = _build("create_card_request_body", "CreateCardRequestBody",
                  type="card_json", data=json.dumps(streaming_card_body(), ensure_ascii=False))
    request = _build("create_card_request", "CreateCardRequest", request_body=body)
    response = await adapter._run_blocking(adapter._client.cardkit.v1.card.create, request)
    result = _result(response)
    if not result.ok:
        logger.info("[Feishu] CardKit card creation failed: code=%s msg=%s", result.code, result.msg)
        return None
    return getattr(getattr(response, "data", None), "card_id", None) or None


async def stream_text(adapter: Any, card_id: str, text: str, sequence: int, *, plain_mentions: bool) -> CallResult:
    """Replace the streaming element's text (the whole text so far, rendered for the card)."""
    from .feishu_card_render import render_markdown_for_card, strip_card_mentions

    content = render_markdown_for_card(strip_card_mentions(text) if plain_mentions else text)
    body = _build("content_card_element_request_body", "ContentCardElementRequestBody",
                  content=content, sequence=sequence)
    request = _build("content_card_element_request", "ContentCardElementRequest",
                     card_id=card_id, element_id=STREAMING_ELEMENT_ID, request_body=body)
    result = _result(await adapter._run_blocking(adapter._client.cardkit.v1.card_element.content, request))
    if not result.ok and result.code != RATE_LIMITED:
        logger.info("[Feishu] CardKit stream to %s failed: code=%s msg=%s", card_id, result.code, result.msg)
    return result


async def replace_card(adapter: Any, card_id: str, body: dict, sequence: int) -> CallResult:
    """Replace the whole card (seals it when ``body`` has ``streaming_mode`` off)."""
    card = _build("card", "Card", type="card_json", data=json.dumps(body, ensure_ascii=False))
    request_body = _build("update_card_request_body", "UpdateCardRequestBody", card=card, sequence=sequence)
    request = _build("update_card_request", "UpdateCardRequest", card_id=card_id, request_body=request_body)
    result = _result(await adapter._run_blocking(adapter._client.cardkit.v1.card.update, request))
    if not result.ok:
        logger.info("[Feishu] CardKit update of %s failed: code=%s msg=%s", card_id, result.code, result.msg)
    return result


async def close_streaming(adapter: Any, card_id: str, sequence: int) -> CallResult:
    """Turn ``streaming_mode`` off without touching the body."""
    settings = json.dumps({"config": {"streaming_mode": False}}, ensure_ascii=False)
    body = _build("settings_card_request_body", "SettingsCardRequestBody", settings=settings, sequence=sequence)
    request = _build("settings_card_request", "SettingsCardRequest", card_id=card_id, request_body=body)
    return _result(await adapter._run_blocking(adapter._client.cardkit.v1.card.settings, request))
