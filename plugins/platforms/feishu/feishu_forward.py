"""Feishu merge-forward ("合并转发") messages rendered as a readable transcript.

A ``merge_forward`` event body is only a placeholder ("Merged and Forwarded Message"). The
forwarded messages come from ``GET im/v1/messages/{id}``: ``items[0]`` is the forward itself,
followed by every descendant flattened, each naming its direct container in
``upper_message_id`` (a nested forward is followed by its own children). A forwarded image or
file downloads through the *outermost* forward's message id; the child's own id and a nested
forward's id answer 234003 "File not in msg".
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

MAX_FORWARD_ENTRIES = 100
MAX_FORWARD_CHARS = 10_000
MAX_ENTRY_CHARS = 2_000
MAX_FORWARD_MEDIA = 10
# The gateway expands ``@file:``/``@url:``/``@diff``... references (built-in and plugin prefixes) in event text.
_CONTEXT_REFERENCE = re.compile(r"(?<![\w/])@(?=[A-Za-z][\w-]*:\S|(?:diff|staged)\b)")


@dataclass
class ForwardTranscript:
    text: str
    image_keys: list[str] = field(default_factory=list)
    media_refs: list[Any] = field(default_factory=list)


async def fetch_message_items(adapter: Any, message_id: str) -> Optional[list[Any]]:
    """The message followed by any forwarded descendants, card bodies as raw ``json_card``.

    Without ``raw_card_content`` a Card JSON 2.0 card reads back as a "please upgrade" placeholder.
    """
    from . import adapter as facade

    request = facade._sdk_build(
        facade.GetMessageRequest, message_id=message_id, card_msg_content_type="raw_card_content",
    )
    response = await adapter._run_blocking(adapter._client.im.v1.message.get, request)
    if not adapter._response_succeeded(response):
        logger.warning(
            "[Feishu] Failed to fetch message %s: [%s] %s", message_id,
            getattr(response, "code", "unknown"), getattr(response, "msg", "message lookup failed"),
        )
        return None
    return list(getattr(getattr(response, "data", None), "items", None) or [])


def keep_references_literal(text: str) -> str:
    """Forwarded text is other people's: a reference in it must not read a file or fetch a URL on the
    forwarder's behalf, so its ``@`` becomes a full-width ``＠`` (the expander no longer matches it)."""
    return _CONTEXT_REFERENCE.sub("＠", text)


def display_text(normalized: Any) -> str:
    """Readable text of a normalized message: its text, its attachment placeholder, or ``[<type>]``."""
    from .adapter import FALLBACK_IMAGE_TEXT

    text = normalized.text_content or str(normalized.metadata.get("placeholder_text") or "").strip()
    if text or normalized.raw_type in ("", "text"):
        return text
    return FALLBACK_IMAGE_TEXT if normalized.raw_type == "image" else f"[{normalized.raw_type}]"


async def expand_merge_forward(adapter: Any, message_id: str, normalized: Any) -> Any:
    """An inbound forward as its transcript plus (bounded) child media; ``normalized`` on failure."""
    from .adapter import FeishuNormalizedMessage

    if not adapter._client or not message_id:
        return normalized
    try:
        items = await fetch_message_items(adapter, message_id)
        transcript = await render_merge_forward(adapter, message_id, items) if items else None
    except Exception:
        logger.warning("[Feishu] Failed to expand merge-forward %s", message_id, exc_info=True)
        return normalized
    if transcript is None:
        return normalized
    return FeishuNormalizedMessage(
        raw_type="merge_forward", text_content=transcript.text, image_keys=transcript.image_keys,
        media_refs=transcript.media_refs, relation_kind="merge_forward",
    )


async def render_merge_forward(adapter: Any, root_id: str, items: list[Any]) -> Optional[ForwardTranscript]:
    """Transcript of a forward's GET ``items``; None when they hold no forwarded message."""
    from .adapter import FALLBACK_FORWARD_TEXT
    from .feishu_names import prefetch_user_names

    children: dict[str, list[Any]] = defaultdict(list)
    for item in items:
        if str(getattr(item, "message_id", "") or "") != root_id:
            children[str(getattr(item, "upper_message_id", "") or root_id)].append(item)
    if not children:
        return None
    total = sum(len(siblings) for siblings in children.values())
    for siblings in children.values():
        siblings.sort(key=_create_time)
    entries: list[tuple[int, Any]] = []
    _flatten(children, root_id, 0, entries)
    senders = [getattr(item, "sender", None) for _depth, item in entries]
    await prefetch_user_names(adapter, [
        str(sender.id) for sender in senders
        if getattr(sender, "id_type", "") == "open_id" and getattr(sender, "id", None)
        and not getattr(sender, "sender_name", None)
    ])
    transcript = _Transcript(adapter)
    shown = sum(1 for depth, item in entries if transcript.add(depth, item))
    lines = [FALLBACK_FORWARD_TEXT, *transcript.lines]
    if total > shown:
        lines.append(f"- … {total - shown} more forwarded message(s) not shown")
    return ForwardTranscript(text="\n".join(lines), image_keys=transcript.image_keys, media_refs=transcript.media_refs)


def _create_time(item: Any) -> int:
    try:
        return int(getattr(item, "create_time", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _flatten(children: dict[str, list[Any]], container_id: str, depth: int, out: list[tuple[int, Any]]) -> None:
    """Depth-first ``(depth, item)`` order, capped at ``MAX_FORWARD_ENTRIES``."""
    for item in children.pop(container_id, []):  # popped: a malformed self-reference can't loop
        if len(out) >= MAX_FORWARD_ENTRIES:
            return
        out.append((depth, item))
        if getattr(item, "msg_type", "") == "merge_forward":
            _flatten(children, str(getattr(item, "message_id", "") or ""), depth + 1, out)


class _Transcript:
    """Transcript lines plus the child media they reference, within the char and media budgets."""

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter
        self._anonymous: dict[str, str] = {}
        self._chars = 0
        self.lines: list[str] = []
        self.image_keys: list[str] = []
        self.media_refs: list[Any] = []

    def add(self, depth: int, item: Any) -> bool:
        """Append one entry; False once the char budget is spent."""
        from .adapter import FALLBACK_FORWARD_TEXT

        if self._chars >= MAX_FORWARD_CHARS:
            return False
        msg_type = str(getattr(item, "msg_type", "") or "")
        if msg_type == "merge_forward":
            body = FALLBACK_FORWARD_TEXT
        else:
            normalized = self._adapter._normalize(
                msg_type, getattr(getattr(item, "body", None), "content", "") or "", getattr(item, "mentions", None),
            )
            body = display_text(normalized) or "[message]"
            room = max(MAX_FORWARD_MEDIA - len(self.image_keys) - len(self.media_refs), 0)
            self.image_keys.extend(normalized.image_keys[:room])
            room = max(MAX_FORWARD_MEDIA - len(self.image_keys) - len(self.media_refs), 0)
            self.media_refs.extend(normalized.media_refs[:room])
        body = body.strip()
        limit = min(MAX_ENTRY_CHARS, MAX_FORWARD_CHARS - self._chars)
        if len(body) > limit:
            body = body[:limit].rstrip() + "…"
        indent = "  " * depth
        first, *rest = body.split("\n")
        entry = [f"{indent}- {self._label(getattr(item, 'sender', None))}: {first}", *(f"{indent}  {line}" for line in rest)]
        self.lines.extend(entry)
        self._chars += sum(len(line) + 1 for line in entry)
        return True

    def _label(self, sender: Any) -> str:
        return sender_label(self._adapter, sender, self._anonymous)


def sender_label(adapter: Any, sender: Any, anonymous: dict[str, str]) -> str:
    """Display name, or a neutral label numbered through ``anonymous``: raw ou_/cli_ ids never reach the model."""
    sender_id = str(getattr(sender, "id", "") or "")
    embedded = str(getattr(sender, "sender_name", "") or "").strip()
    if embedded:
        return embedded
    if getattr(sender, "sender_type", "") == "app":
        return (adapter._bot_name if sender_id and sender_id == adapter._app_id else "") or "Bot"
    name = adapter._get_cached_sender_name(sender_id) if sender_id else None
    return name or anonymous.setdefault(sender_id, f"Participant {len(anonymous) + 1}")
