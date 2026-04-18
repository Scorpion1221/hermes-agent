from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Optional

logger = logging.getLogger(__name__)

STREAMING_ELEMENT_ID = "streaming_content"
LOADING_ELEMENT_ID = "streaming_loading"


def _ns(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


@dataclass
class CardKitState:
    card_id: str
    message_id: str = ""
    sequence: int = 1
    element_id: str = STREAMING_ELEMENT_ID
    failed: bool = False
    started_at: float = 0.0
    last_content: str = ""


def build_streaming_card_body() -> dict:
    return {
        "schema": "2.0",
        "config": {
            "streaming_mode": True,
            "summary": {"content": "..."},
        },
        "body": {
            "elements": [
                {
                    "tag": "markdown",
                    "content": "",
                    "text_align": "left",
                    "text_size": "normal_v2",
                    "element_id": STREAMING_ELEMENT_ID,
                },
            ],
        },
    }


def build_final_card_body(content: str, *, elapsed_seconds: float = 0.0, stopped: bool = False) -> dict:
    elements: list[dict] = [
        {
            "tag": "markdown",
            "content": content,
            "text_align": "left",
            "text_size": "normal_v2",
        },
    ]
    if elapsed_seconds > 0:
        if elapsed_seconds >= 60:
            mins = int(elapsed_seconds // 60)
            secs = int(elapsed_seconds % 60)
            time_str = f"{mins}m {secs}s"
        else:
            time_str = f"{elapsed_seconds:.1f}s"
        status = "已停止" if stopped else "已完成"
        elements.append({
            "tag": "markdown",
            "content": f"{status} · 耗时 {time_str}",
            "text_size": "notation",
            "text_align": "left",
        })
    return {
        "schema": "2.0",
        "config": {
            "streaming_mode": False,
        },
        "body": {
            "elements": elements,
        },
    }


async def create_streaming_card(client: Any) -> Optional[str]:
    try:
        from lark_oapi.api.cardkit.v1.model.create_card_request import CreateCardRequest
        from lark_oapi.api.cardkit.v1.model.create_card_request_body import CreateCardRequestBody
        body = CreateCardRequestBody.builder().type("card_json").data(
            json.dumps(build_streaming_card_body(), ensure_ascii=False)
        ).build()
        request = CreateCardRequest.builder().request_body(body).build()
    except ImportError:
        card_data = json.dumps(build_streaming_card_body(), ensure_ascii=False)
        body = _ns(type="card_json", data=card_data)
        request = _ns(request_body=body, body=body)

    response = await asyncio.to_thread(client.cardkit.v1.card.create, request)
    if not response or not getattr(response, "success", lambda: False)():
        logger.info(
            "[CardKit] Card creation failed: code=%s msg=%s",
            getattr(response, "code", "?"), getattr(response, "msg", "?"),
        )
        return None
    card_id = getattr(getattr(response, "data", None), "card_id", None)
    if card_id:
        logger.info("[CardKit] Created streaming card: %s", card_id)
    return card_id


async def stream_card_element(
    client: Any, *, card_id: str, element_id: str, content: str, sequence: int,
) -> bool:
    try:
        from lark_oapi.api.cardkit.v1.model.content_card_element_request import ContentCardElementRequest
        from lark_oapi.api.cardkit.v1.model.content_card_element_request_body import ContentCardElementRequestBody
        body = ContentCardElementRequestBody.builder().content(content).sequence(sequence).build()
        request = ContentCardElementRequest.builder().card_id(card_id).element_id(element_id).request_body(body).build()
    except ImportError:
        body = _ns(content=content, sequence=sequence)
        request = _ns(card_id=card_id, element_id=element_id, request_body=body, body=body, paths={"card_id": card_id, "element_id": element_id})

    response = await asyncio.to_thread(client.cardkit.v1.card_element.content, request)
    if not response or not getattr(response, "success", lambda: False)():
        code = getattr(response, "code", 0)
        if code == 230020:
            return True
        logger.info(
            "[CardKit] Stream content failed: code=%s msg=%s",
            code, getattr(response, "msg", "?"),
        )
        return False
    return True


async def update_card(
    client: Any, *, card_id: str, card_body: dict, sequence: int,
) -> bool:
    try:
        from lark_oapi.api.cardkit.v1.model.update_card_request import UpdateCardRequest
        from lark_oapi.api.cardkit.v1.model.update_card_request_body import UpdateCardRequestBody
        from lark_oapi.api.cardkit.v1.model.card import Card
        card_json = json.dumps(card_body, ensure_ascii=False)
        card = Card.builder().type("card_json").data(card_json).build()
        body = UpdateCardRequestBody.builder().card(card).sequence(sequence).build()
        request = UpdateCardRequest.builder().card_id(card_id).request_body(body).build()
    except ImportError:
        card_json = json.dumps(card_body, ensure_ascii=False)
        card = _ns(type="card_json", data=card_json)
        body = _ns(card=card, sequence=sequence)
        request = _ns(card_id=card_id, request_body=body, body=body, paths={"card_id": card_id})

    response = await asyncio.to_thread(client.cardkit.v1.card.update, request)
    if not response or not getattr(response, "success", lambda: False)():
        logger.info(
            "[CardKit] Card update failed: code=%s msg=%s",
            getattr(response, "code", "?"), getattr(response, "msg", "?"),
        )
        return False
    return True


async def set_card_streaming_mode(
    client: Any, *, card_id: str, enabled: bool, sequence: int,
) -> bool:
    try:
        from lark_oapi.api.cardkit.v1.model.settings_card_request import SettingsCardRequest
        from lark_oapi.api.cardkit.v1.model.settings_card_request_body import SettingsCardRequestBody
        settings_json = json.dumps({"config": {"streaming_mode": enabled}}, ensure_ascii=False)
        body = SettingsCardRequestBody.builder().settings(settings_json).sequence(sequence).build()
        request = SettingsCardRequest.builder().card_id(card_id).request_body(body).build()
    except ImportError:
        settings_json = json.dumps({"config": {"streaming_mode": enabled}}, ensure_ascii=False)
        body = _ns(settings=settings_json, sequence=sequence)
        request = _ns(card_id=card_id, request_body=body, body=body, paths={"card_id": card_id})

    response = await asyncio.to_thread(client.cardkit.v1.card.settings, request)
    if not response or not getattr(response, "success", lambda: False)():
        logger.info(
            "[CardKit] Set streaming mode=%s failed: code=%s msg=%s",
            enabled, getattr(response, "code", "?"), getattr(response, "msg", "?"),
        )
        return False
    return True


def build_card_id_message_content(card_id: str) -> str:
    return json.dumps({"type": "card", "data": {"card_id": card_id}}, ensure_ascii=False)


__all__ = [
    "STREAMING_ELEMENT_ID",
    "LOADING_ELEMENT_ID",
    "CardKitState",
    "build_streaming_card_body",
    "build_final_card_body",
    "create_streaming_card",
    "stream_card_element",
    "update_card",
    "set_card_streaming_mode",
    "build_card_id_message_content",
]
