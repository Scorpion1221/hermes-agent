"""Feishu outbound text: the payload for each chunk of a reply, its fallbacks, and edits.

A formatted reply is chunked at ``MAX_MESSAGE_LENGTH``. With ``render_mode: card`` (the default) every chunk
goes out as an ``interactive`` Card JSON 2.0 card with one ``markdown`` element (``feishu_card_render``),
which renders GFM tables, headings, lists and code the same way every time. ``render_mode: post`` keeps the
post/text choice: the markdown-vs-text decision is made once for the whole message (#26841) and each chunk is
a ``post`` with an ``md`` element, or ``text``.

Fallbacks: a card Feishu refuses for a mention it cannot resolve (100290) is resent once with every mention
as plain ``@name``; a card or post refused for its content is resent as plain text with the Markdown
stripped. Edits PATCH a card (the PUT update API cannot edit ``interactive`` messages) and PUT text/post.
A cron delivery becomes a notification card instead (``feishu_cron_card``).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from gateway.platforms.base import SendResult

logger = logging.getLogger(__name__)

_MARKDOWN_HINT_RE = re.compile(
    # Pipe table: any header line + separator line both starting with '|'.
    r"(^\|.*\|\s*\n\|[-:|\s]+\|)"
    # Headings, lists, code, bold/italic/strike/underline, links, blockquotes.
    r"|(^#{1,6}\s)"
    r"|(^\s*[-*]\s)"
    r"|(^\s*\d+\.\s)"
    r"|(^\s*---+\s*$)"
    r"|(```)"
    r"|(`[^`\n]+`)"
    r"|(\*\*[^*\n].+?\*\*)"
    r"|(~~[^~\n].+?~~)"
    r"|(<u>.+?</u>)"
    r"|(\*[^*\n]+\*)"
    r"|(\[[^\]]+\]\([^)]+\))"
    r"|(^>\s)",
    re.MULTILINE,
)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_MARKDOWN_FENCE_OPEN_RE = re.compile(r"^```([^\n`]*)\s*$")
_MARKDOWN_FENCE_CLOSE_RE = re.compile(r"^```\s*$")
_POST_CONTENT_INVALID_RE = re.compile(r"content format of the post type is incorrect", re.IGNORECASE)
# How Feishu words a card it refuses for its content (seen in production on the fork since April 2026).
_CARD_CONTENT_INVALID_RE = re.compile(
    r"(invalid.{0,40}(card|interactive|schema)|(card|interactive|schema).{0,40}invalid|card.{0,40}(format|content|schema))",
    re.IGNORECASE,
)
CARD_INVALID_USER_RESOURCE = 100290  # a mention in the card names a user Feishu cannot resolve


def _strip_markdown_to_plain_text(text: str) -> str:
    """Plain-text fallback: shared strip_markdown plus Feishu extras (blockquote, ~~, <u>, hr, CRLF)."""
    from gateway.platforms.helpers import strip_markdown
    plain = text.replace("\r\n", "\n")
    plain = _MARKDOWN_LINK_RE.sub(lambda m: f"{m.group(1)} ({m.group(2).strip()})", plain)
    plain = re.sub(r"^>\s?", "", plain, flags=re.MULTILINE)
    plain = re.sub(r"^\s*---+\s*$", "---", plain, flags=re.MULTILINE)
    plain = re.sub(r"~~([^~\n]+)~~", r"\1", plain)
    plain = re.sub(r"<u>([\s\S]*?)</u>", r"\1", plain)
    return strip_markdown(plain)


def _build_markdown_post_payload(content: str) -> str:
    rows = _build_markdown_post_rows(content)
    return json.dumps({"zh_cn": {"content": rows}}, ensure_ascii=False)


def _build_markdown_post_rows(content: str) -> list[list[dict[str, str]]]:
    """Build Feishu post rows, giving each fenced code block its own row.

    Feishu's `md` renderer can swallow trailing content when a fence sits inside one
    large element; splitting at real fence lines keeps surrounding prose visible.
    """
    if not content:
        return [[{"tag": "md", "text": ""}]]
    if "```" not in content:
        return [[{"tag": "md", "text": content}]]

    rows: list[list[dict[str, str]]] = []
    current: list[str] = []
    in_code_block = False

    def _flush_current() -> None:
        nonlocal current
        segment = "\n".join(current)
        if segment.strip():
            rows.append([{"tag": "md", "text": segment}])
        current = []

    for raw_line in content.splitlines():
        fence_re = _MARKDOWN_FENCE_CLOSE_RE if in_code_block else _MARKDOWN_FENCE_OPEN_RE
        is_fence = bool(fence_re.match(raw_line.strip()))
        if is_fence and not in_code_block:  # opening fence: prose before it becomes its own row
            _flush_current()
        current.append(raw_line)
        if is_fence:
            in_code_block = not in_code_block
            if not in_code_block:  # closing fence: the code block becomes its own row
                _flush_current()
    _flush_current()
    return rows or [[{"tag": "md", "text": content}]]


def post_mode_payload(content: str, *, prefer_post: bool = False) -> tuple[str, str]:
    """``post`` + ``md`` for Markdown (tables included: Feishu clients render them there), else ``text``.

    ``prefer_post`` keeps every chunk of a split Markdown reply a post even when a chunk alone reads as
    prose (#26841). A table no longer downgrades the message to ``text`` (#52786).
    """
    if prefer_post or _MARKDOWN_HINT_RE.search(content):
        return "post", _build_markdown_post_payload(content)
    return "text", json.dumps({"text": content}, ensure_ascii=False)


def content_rejected(msg_type: str, text: str) -> bool:
    """Whether Feishu refused a post or card for its content, so the plain-text payload can go instead."""
    if msg_type == "post":
        return bool(_POST_CONTENT_INVALID_RE.search(text))
    return msg_type == "interactive" and bool(_CARD_CONTENT_INVALID_RE.search(text))


def outbound_payload(adapter: Any, content: str, *, prefer_post: bool = False) -> tuple[str, str]:
    """``(msg_type, payload)`` for one chunk under the adapter's ``render_mode``."""
    if getattr(adapter, "_render_mode", "card") == "post":
        return post_mode_payload(content, prefer_post=prefer_post)
    from .feishu_card_render import markdown_card
    return "interactive", markdown_card(content)


def _mention_refused(msg_type: str, code: Any, content: str) -> bool:
    from .feishu_card_render import has_card_mentions
    return msg_type == "interactive" and str(code) == str(CARD_INVALID_USER_RESOURCE) and has_card_mentions(content)


def _patch_request(message_id: str, content: str) -> Any:
    from . import adapter as facade  # the lazily bound SDK names live on the facade
    body = facade._sdk_build(facade.PatchMessageRequestBody, content=content)
    return facade._sdk_build(facade.PatchMessageRequest, message_id=message_id, request_body=body)


def _without_mentions(content: str) -> str:
    from .feishu_card_render import markdown_card, strip_card_mentions
    return markdown_card(strip_card_mentions(content))


async def send_message(adapter: Any, chat_id: str, content: str, *, reply_to: Any, metadata: Any) -> SendResult:
    """``FeishuAdapter.send``: chunk the formatted reply and send each chunk, falling back to plain text."""
    if not adapter._client:
        return SendResult(success=False, error="Not connected")

    formatted = adapter.format_message(content)
    from .feishu_cron_card import parse_cron_delivery
    cron = parse_cron_delivery(formatted) if getattr(adapter, "_render_mode", "card") == "card" else None
    chunks = adapter.truncate_message(cron.body if cron else formatted, adapter.MAX_MESSAGE_LENGTH)
    # Decide markdown-vs-text once for the whole message: a chunk of a long markdown reply may be plain
    # prose that fails the per-chunk regex and would otherwise render as literal ``**bold`` / fences while
    # other chunks render. See #26841.
    prefer_post = bool(_MARKDOWN_HINT_RE.search(formatted))
    last_response = None

    async def _send(msg_type: str, payload: str) -> Any:
        return await adapter._feishu_send_with_retry(
            chat_id=chat_id, msg_type=msg_type, payload=payload, reply_to=reply_to, metadata=metadata,
        )

    async def _send_plain(chunk: str) -> Any:
        return await _send("text", json.dumps({"text": _strip_markdown_to_plain_text(chunk)}, ensure_ascii=False))

    try:
        if cron:
            from .feishu_cron_card import cron_card
        for index, chunk in enumerate(chunks):
            msg_type, payload = outbound_payload(adapter, chunk, prefer_post=prefer_post)
            if cron:
                payload = cron_card(cron, chunk, header=index == 0, footer=index == len(chunks) - 1)
            try:
                response = await _send(msg_type, payload)
            except Exception as exc:
                if not content_rejected(msg_type, str(exc)):
                    raise
                logger.warning("[Feishu] Invalid %s payload rejected by API; falling back to plain text", msg_type)
                response = await _send_plain(chunk)
                msg_type = "text"
            if not adapter._response_succeeded(response) and _mention_refused(
                msg_type, getattr(response, "code", None), chunk,
            ):
                logger.info("[Feishu] A mention in the card could not be resolved; sending mentions as text")
                response = await _send(msg_type, _without_mentions(chunk))
            if not adapter._response_succeeded(response) and content_rejected(
                msg_type, str(getattr(response, "msg", "") or ""),
            ):
                logger.warning("[Feishu] %s payload rejected by API response; falling back to plain text", msg_type)
                response = await _send_plain(chunk)
            last_response = response

        return adapter._finalize_send_result(last_response, "send failed")
    except Exception as exc:
        logger.error("[Feishu] Send error: %s", exc, exc_info=True)
        return SendResult(success=False, error=str(exc))


async def edit_message(adapter: Any, message_id: str, content: str) -> SendResult:
    """``FeishuAdapter.edit_message``: PATCH a card or PUT a text/post payload, falling back to plain text.

    The edit keeps the type the message was sent as: under ``render_mode: card`` every formatted message
    went out as a card, so the edit PATCHes a card; if that PATCH is refused because the message is not
    a card (a plain-text fallback, or one sent before the mode changed), the edit PUTs instead.
    """
    if not adapter._client:
        return SendResult(success=False, error="Not connected")

    content = adapter.format_message(content)

    async def _update(msg_type: str, payload: str) -> SendResult:
        body = adapter._build_update_message_body(msg_type=msg_type, content=payload)
        request = adapter._build_update_message_request(message_id=message_id, request_body=body)
        response = await adapter._run_blocking(adapter._client.im.v1.message.update, request)
        return adapter._finalize_send_result(response, "update failed")

    async def _patch(payload: str) -> tuple[SendResult, Any]:
        request = _patch_request(message_id, payload)
        response = await adapter._run_blocking(adapter._client.im.v1.message.patch, request)
        return adapter._finalize_send_result(response, "patch failed"), getattr(response, "code", None)

    async def _put_post_or_text() -> SendResult:
        msg_type, payload = post_mode_payload(content)
        result = await _update(msg_type, payload)
        if not result.success and content_rejected(msg_type, result.error or ""):
            logger.warning("[Feishu] Invalid post update payload rejected by API; falling back to plain text")
            result = await _update(
                "text", json.dumps({"text": _strip_markdown_to_plain_text(content)}, ensure_ascii=False),
            )
        return result

    try:
        msg_type, payload = outbound_payload(adapter, content)
        if msg_type == "interactive":
            result, code = await _patch(payload)
            if not result.success and _mention_refused(msg_type, code, content):
                logger.info("[Feishu] A mention in the card could not be resolved; editing with mentions as text")
                result, code = await _patch(_without_mentions(content))
            if not result.success:
                logger.debug("[Feishu] Card PATCH refused for %s (%s); trying PUT update", message_id, result.error)
                result = await _put_post_or_text()
        else:
            result = await _put_post_or_text()
        if result.success:
            result.message_id = message_id
        return result
    except Exception as exc:
        logger.error("[Feishu] Failed to edit message %s: %s", message_id, exc, exc_info=True)
        return SendResult(success=False, error=str(exc))
