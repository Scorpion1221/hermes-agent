"""An inbound interactive card (``interactive`` / ``card``) rendered as readable text (moved out of ``adapter.py``).

The event body of a card is its degraded form: a Card JSON 2.0 card (another bot's under
``allow_bots``, or a single card forwarded into the chat) arrives as a "please upgrade your client"
placeholder. As for quoted cards, the message is fetched again with ``raw_card_content`` and read from
its ``json_card``; a failed or unreadable fetch keeps the event body. A template card
(``{"type": "template", "data": {"template_variable": {...}}}``) reads as its variable values.

``adapter`` text helpers are reached through the module at call time: ``adapter`` imports this one.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)


def _adapter():
    return sys.modules[__package__ + ".adapter"]


async def hydrate_card(adapter: Any, message_id: str, normalized: Any) -> Any:
    """``normalized`` read from the fetched card when that reads as more than the placeholder."""
    from .feishu_forward import fetch_message_items

    if not adapter._client or not message_id:
        return normalized
    try:
        items = await fetch_message_items(adapter, message_id)
    except Exception:
        logger.warning("[Feishu] Failed to fetch card message %s", message_id, exc_info=True)
        return normalized
    if not items:
        return normalized
    item = items[0]
    hydrated = adapter._normalize(
        str(getattr(item, "msg_type", "") or normalized.raw_type),
        getattr(getattr(item, "body", None), "content", "") or "", None,
    )
    if hydrated.text_content in ("", _adapter().FALLBACK_INTERACTIVE_TEXT):
        return normalized
    return dataclasses.replace(hydrated, mentions=normalized.mentions)  # the event's mentions know the bot


def normalize_interactive_message(message_type: str, payload: dict[str, Any]) -> Any:
    from .feishu_card_text import render_json_card

    a = _adapter()
    if card_text := render_json_card(payload):  # GET with card_msg_content_type=raw_card_content
        return a.FeishuNormalizedMessage(raw_type=message_type, text_content=card_text, relation_kind="interactive")
    card_payload = payload.get("card") if isinstance(payload.get("card"), dict) else payload
    title = a._first_non_empty_text(
        a._find_header_title(card_payload), payload.get("title"),
        a._find_first_text(card_payload, keys=("title", "summary", "subtitle")),
    )
    actions = a._collect_action_labels(card_payload)
    body = a._collect_card_lines(card_payload) + _template_variable_lines(card_payload)
    lines = ([title] if title else []) + a._unique_lines([line for line in body if line != title])
    if actions:
        lines.append(f"Actions: {', '.join(actions)}")
    return a.FeishuNormalizedMessage(
        raw_type=message_type,
        text_content="\n".join(lines[:12]).strip() or a.FALLBACK_INTERACTIVE_TEXT,
        relation_kind="interactive", metadata={"title": title, "actions": actions},
    )


def _template_variable_lines(payload: Any) -> list[str]:
    """Text values of a template card's ``template_variable``; ids, URLs and other non-text keys are left out."""
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    candidates = [value for value in (
        data.get("template_variable") if isinstance(data, dict) else None, payload.get("template_variable"),
    ) if isinstance(value, (dict, list))]
    lines: list[str] = []
    for candidate in candidates:
        _walk_template_values(candidate, "", lines)
    return _adapter()._unique_lines(lines)


def _walk_template_values(value: Any, key: str, lines: list[str]) -> None:
    lowered = key.strip().lower()
    if isinstance(value, str):
        text = _adapter()._normalize_feishu_text(value)
        if text and lowered not in _adapter()._SKIP_TEXT_KEYS and not lowered.endswith(("_id", "_url")):
            lines.append(text)
    elif isinstance(value, list):
        for item in value:
            _walk_template_values(item, key, lines)
    elif isinstance(value, dict):
        for child_key, child in value.items():
            _walk_template_values(child, str(child_key), lines)
