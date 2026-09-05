from __future__ import annotations

import asyncio
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Optional

logger = logging.getLogger(__name__)

STREAMING_ELEMENT_ID = "streaming_content"
LOADING_ELEMENT_ID = "streaming_loading"
_MARKDOWN_FENCE_OPEN_RE = re.compile(r"^[ ]{0,3}(?P<fence>`{3,}|~{3,}).*$")
_MARKDOWN_ATX_HEADING_RE = re.compile(
    r"^(?P<indent>[ ]{0,3})(?P<marks>#{1,6})(?P<rest>(?:[ \t]+.*)?)$"
)
_MARKDOWN_INLINE_CODE_RE = re.compile(
    r"(?<!\\)(?P<fence>`+)(?!`)[^\r\n]*?(?P=fence)(?!`)"
)
_CARDKIT_STRONG_BOUNDARY_RE = re.compile(
    r"(?<![*\\])\*\*(?!\*)"
    r"(?P<label>[^`\\*\r\n]*?)(?P<last>[^`\\*\s\r\n])\*\*"
    r"(?=[^\W])"
)


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
    stopped: bool = False
    reply_to_message_id: str = ""


def _split_line_ending(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    if line.endswith("\r"):
        return line[:-1], "\r"
    return line, ""


def _inside_url_token(line: str, position: int) -> bool:
    token_start = position
    while token_start > 0 and not line[token_start - 1].isspace():
        token_start -= 1
    token_prefix = line[token_start:position].lower()
    return "://" in token_prefix or token_prefix.startswith("www.")


def _space_cardkit_strong_boundaries(line: str) -> str:
    """Separate punctuation-ended ``**`` spans from an attached word.

    CommonMark and CardKit leave ``**Label:**body`` literal because the closing
    delimiter is preceded by punctuation and followed by a word.  Only repair
    that delimiter shape, outside inline/indented code and URL tokens.
    """
    if line.startswith(("    ", "\t")):
        return line

    code_spans = [match.span() for match in _MARKDOWN_INLINE_CODE_RE.finditer(line)]
    pieces: list[str] = []
    cursor = 0
    for match in _CARDKIT_STRONG_BOUNDARY_RE.finditer(line):
        if not unicodedata.category(match.group("last")).startswith("P"):
            continue
        if any(start <= match.start() < end for start, end in code_spans):
            continue
        if _inside_url_token(line, match.start()):
            continue
        pieces.append(line[cursor:match.end()])
        pieces.append(" ")
        cursor = match.end()

    if not pieces:
        return line
    pieces.append(line[cursor:])
    return "".join(pieces)


def render_markdown_for_card(content: str) -> str:
    """Render raw assistant Markdown for Feishu Card readability.

    Feishu Card 2.0 renders top-level Markdown headings very large. Hermes
    messages usually live inside an already-framed card, so render Markdown
    headings two levels lower (# -> ###, ## -> ####, capped at ######).
    It also leaves punctuation-ended strong spans such as ``**Label:**body`` as
    literal text, so separate that verified boundary. Code stays untouched.

    Keep this as a render-only boundary: callers should pass raw assistant
    text/session state and must not store this returned card-specific Markdown
    back into conversation or streaming state.
    """
    if not content or ("#" not in content and "**" not in content):
        return content

    lines: list[str] = []
    fence_char = ""
    fence_len = 0
    for raw_line in content.splitlines(keepends=True):
        line, ending = _split_line_ending(raw_line)
        if fence_char:
            lines.append(raw_line)
            if re.match(
                rf"^[ ]{{0,3}}{re.escape(fence_char)}{{{fence_len},}}[ \t]*$",
                line,
            ):
                fence_char = ""
                fence_len = 0
            continue

        fence_match = _MARKDOWN_FENCE_OPEN_RE.match(line)
        if fence_match:
            fence = fence_match.group("fence")
            fence_char = fence[0]
            fence_len = len(fence)
            lines.append(raw_line)
            continue

        heading_match = _MARKDOWN_ATX_HEADING_RE.match(line)
        if heading_match:
            marks = heading_match.group("marks")
            new_level = min(len(marks) + 2, 6)
            line = (
                f"{heading_match.group('indent')}"
                f"{'#' * new_level}"
                f"{heading_match.group('rest')}"
            )

        lines.append(_space_cardkit_strong_boundaries(line) + ending)

    return "".join(lines)


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
                {
                    "tag": "markdown",
                    "content": " ",
                    "icon": {
                        "tag": "custom_icon",
                        "img_key": "img_v3_02vb_496bec09-4b43-4773-ad6b-0cdd103cd2bg",
                        "size": "16px 16px",
                    },
                    "element_id": LOADING_ELEMENT_ID,
                },
            ],
        },
    }


def build_final_card_body(content: str, *, elapsed_seconds: float = 0.0, stopped: bool = False, status: str = "") -> dict:
    elements: list[dict] = [
        {
            "tag": "markdown",
            "content": render_markdown_for_card(content),
            "text_align": "left",
            "text_size": "normal_v2",
        },
    ]
    if elapsed_seconds > 0 or status:
        if elapsed_seconds >= 60:
            mins = int(elapsed_seconds // 60)
            secs = int(elapsed_seconds % 60)
            time_str = f"{mins}m {secs}s"
        else:
            time_str = f"{elapsed_seconds:.1f}s"
        status = status or ("已停止" if stopped else "已完成")
        elements.append({
            "tag": "markdown",
            "content": f"{status} · 耗时 {time_str}" if elapsed_seconds > 0 else status,
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


def build_cron_notification_card(content: str, notification: dict) -> dict:
    """Render scheduler-owned metadata, never infer success from model prose."""
    failed = notification.get("status") == "error"
    title = str(notification.get("name") or "定时任务")
    footer = []
    if notification.get("completed_at"):
        footer.append(f"通知时间：{notification['completed_at']}")
    elapsed = notification.get("elapsed_seconds")
    if isinstance(elapsed, (int, float)) and elapsed >= 0:
        footer.append(f"执行耗时：{int(elapsed) // 60}分 {int(elapsed) % 60}秒")
    if notification.get("job_id"):
        footer.append(f"任务 ID：{notification['job_id']}")
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "red" if failed else "blue",
            "title": {"tag": "plain_text", "content": f"{'⚠️' if failed else '⏰'} {title}"},
            "subtitle": {"tag": "plain_text", "content": "执行失败" if failed else "定时任务通知"},
        },
        "body": {"elements": [
            {"tag": "markdown", "content": render_markdown_for_card(content)},
            {"tag": "hr"},
            {"tag": "markdown", "text_size": "notation", "content": "\n".join(footer)},
            {"tag": "markdown", "text_size": "notation", "content": "需要调整或暂停？回复并说明任务名称即可。"},
        ]},
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
    # Render raw streaming text only at the Feishu CardKit API boundary.  Do not
    # write this card-specific Markdown back to CardKitState or conversation
    # state; doing so would make a later final render downshift headings again.
    content = render_markdown_for_card(content)
    # DEBUG: log the exact bytes shipped to CardKit so we can correlate
    # rendering bugs (e.g. inline code spans being swallowed) against the
    # raw markdown the server actually sees. Enable with `--log-level DEBUG`
    # or by setting LOG_LEVEL=DEBUG. Truncate at 4 KB to keep logs sane.
    if logger.isEnabledFor(logging.DEBUG):
        preview = content if len(content) <= 4096 else content[:4096] + f"...[+{len(content) - 4096} bytes]"
        logger.debug(
            "[CardKit] stream_card_element card_id=%s element_id=%s seq=%d bytes=%d content=%r",
            card_id, element_id, sequence, len(content), preview,
        )
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
    # DEBUG: log the final card body so we can post-mortem rendering bugs
    # without needing to re-fetch raw_card_content from the message-get API.
    if logger.isEnabledFor(logging.DEBUG):
        try:
            elements = card_body.get("body", {}).get("elements", [])
            md_contents = [
                e.get("content") for e in elements
                if isinstance(e, dict) and e.get("tag") == "markdown"
            ]
            for idx, md in enumerate(md_contents):
                if md is None:
                    continue
                preview = md if len(md) <= 4096 else md[:4096] + f"...[+{len(md) - 4096} bytes]"
                logger.debug(
                    "[CardKit] update_card card_id=%s seq=%d elem=%d bytes=%d markdown=%r",
                    card_id, sequence, idx, len(md), preview,
                )
        except Exception:
            logger.debug("[CardKit] update_card debug log failed", exc_info=True)
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
    "render_markdown_for_card",
    "create_streaming_card",
    "stream_card_element",
    "update_card",
    "set_card_streaming_mode",
    "build_card_id_message_content",
]
