"""Feishu outbound text: the payload for each chunk of a reply, its fallbacks, and edits.

A formatted reply is chunked at ``MAX_MESSAGE_LENGTH``; the markdown-vs-text decision is made once for the
whole message (#26841) and each chunk goes out as a ``post`` with an ``md`` element, or as ``text``. A post
Feishu rejects for its content is resent as plain text with the Markdown stripped. Edits PUT the same payload.
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
    """Whether Feishu refused a post for its content, so the plain-text payload can go instead."""
    return msg_type == "post" and bool(_POST_CONTENT_INVALID_RE.search(text))


async def send_message(adapter: Any, chat_id: str, content: str, *, reply_to: Any, metadata: Any) -> SendResult:
    """``FeishuAdapter.send``: chunk the formatted reply and send each chunk, falling back to plain text."""
    if not adapter._client:
        return SendResult(success=False, error="Not connected")

    formatted = adapter.format_message(content)
    chunks = adapter.truncate_message(formatted, adapter.MAX_MESSAGE_LENGTH)
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
        for chunk in chunks:
            msg_type, payload = post_mode_payload(chunk, prefer_post=prefer_post)
            try:
                response = await _send(msg_type, payload)
            except Exception as exc:
                if not content_rejected(msg_type, str(exc)):
                    raise
                logger.warning("[Feishu] Invalid post payload rejected by API; falling back to plain text")
                response = await _send_plain(chunk)
            if not adapter._response_succeeded(response) and content_rejected(
                msg_type, str(getattr(response, "msg", "") or ""),
            ):
                logger.warning("[Feishu] Post payload rejected by API response; falling back to plain text")
                response = await _send_plain(chunk)
            last_response = response

        return adapter._finalize_send_result(last_response, "send failed")
    except Exception as exc:
        logger.error("[Feishu] Send error: %s", exc, exc_info=True)
        return SendResult(success=False, error=str(exc))


async def edit_message(adapter: Any, message_id: str, content: str) -> SendResult:
    """``FeishuAdapter.edit_message``: PUT the new text/post payload, falling back to plain text."""
    if not adapter._client:
        return SendResult(success=False, error="Not connected")

    content = adapter.format_message(content)

    async def _update(msg_type: str, payload: str) -> SendResult:
        body = adapter._build_update_message_body(msg_type=msg_type, content=payload)
        request = adapter._build_update_message_request(message_id=message_id, request_body=body)
        response = await adapter._run_blocking(adapter._client.im.v1.message.update, request)
        return adapter._finalize_send_result(response, "update failed")

    try:
        msg_type, payload = post_mode_payload(content)
        result = await _update(msg_type, payload)
        if not result.success and content_rejected(msg_type, result.error or ""):
            logger.warning("[Feishu] Invalid post update payload rejected by API; falling back to plain text")
            result = await _update(
                "text", json.dumps({"text": _strip_markdown_to_plain_text(content)}, ensure_ascii=False),
            )
        if result.success:
            result.message_id = message_id
        return result
    except Exception as exc:
        logger.error("[Feishu] Failed to edit message %s: %s", message_id, exc, exc_info=True)
        return SendResult(success=False, error=str(exc))
