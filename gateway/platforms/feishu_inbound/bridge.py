from __future__ import annotations

"""Pure Feishu inbound bridge helpers.

This module owns the adapter-independent parts of converting hydrated Feishu
content into Hermes gateway ``MessageEvent`` objects.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional, Sequence

from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource

from .parse import (
    FALLBACK_ATTACHMENT_TEXT,
    FALLBACK_FORWARD_TEXT,
    FALLBACK_IMAGE_TEXT,
    FALLBACK_INTERACTIVE_TEXT,
    FALLBACK_POST_TEXT,
    FALLBACK_SHARE_CHAT_TEXT,
    FeishuNormalizedMessage,
    normalize_feishu_message,
)
from .types import FeishuMessageContext, FeishuQuotedContext

_MEDIA_ONLY_RAW_TYPES = frozenset({"image", "file", "audio", "media"})
_TEXT_INJECTABLE_MESSAGE_TYPES = frozenset(
    {MessageType.PHOTO, MessageType.VIDEO, MessageType.AUDIO, MessageType.DOCUMENT}
)
_TEXT_INJECTABLE_PREFERENCES = frozenset({"document", "audio"})


@dataclass(frozen=True)
class FeishuInboundContentBridge:
    text: str
    message_type: MessageType
    media_urls: tuple[str, ...] = ()
    media_types: tuple[str, ...] = ()
    raw_message_type: str = ""
    message_context: Optional[FeishuMessageContext] = None


@dataclass(frozen=True)
class FeishuReplyContextBridge:
    reply_to_message_id: Optional[str] = None
    reply_to_text: Optional[str] = None
    reply_to_media_urls: tuple[str, ...] = ()
    reply_to_media_types: tuple[str, ...] = ()
    quoted_context: Optional[FeishuQuotedContext] = None


# Backward-compatible aliases for main-thread wiring while extraction is in flight.
FeishuExtractedContent = FeishuInboundContentBridge
FeishuReplyContextData = FeishuReplyContextBridge


@dataclass(frozen=True)
class FeishuSenderProfile:
    user_id: Optional[str]
    user_name: Optional[str]
    user_id_alt: Optional[str]
    auth_user_ids: tuple[str, ...] = ()


def resolve_feishu_media_message_type(media_type: str, *, default: MessageType) -> MessageType:
    normalized = str(media_type or "").strip().lower()
    if normalized.startswith("image/"):
        return MessageType.PHOTO
    if normalized.startswith("audio/"):
        return MessageType.AUDIO
    if normalized.startswith("video/"):
        return MessageType.VIDEO
    return default


def _resolve_preferred_message_type(preferred_message_type: str, media_types: Sequence[str]) -> MessageType:
    preferred = str(preferred_message_type or "text").strip().lower()
    primary_media_type = str(media_types[0] if media_types else "")
    if preferred == "photo":
        return resolve_feishu_media_message_type(primary_media_type, default=MessageType.PHOTO)
    if preferred == "audio":
        return resolve_feishu_media_message_type(primary_media_type, default=MessageType.AUDIO)
    if preferred == "document":
        return resolve_feishu_media_message_type(primary_media_type, default=MessageType.DOCUMENT)
    return MessageType.TEXT


def resolve_feishu_normalized_message_type(
    normalized: FeishuNormalizedMessage,
    media_types: Sequence[str],
) -> MessageType:
    return _resolve_preferred_message_type(
        getattr(normalized, "preferred_message_type", "text"),
        media_types,
    )


def resolve_feishu_context_message_type(
    context: FeishuMessageContext,
    media_types: Sequence[str],
) -> MessageType:
    if (
        str(getattr(context, "content_type", "") or "").strip().lower() == "audio"
        and str(getattr(context, "preferred_message_type", "") or "").strip().lower()
        == "audio"
    ):
        # Feishu's native ``audio`` message is an in-app voice note. Uploaded
        # audio files arrive through ``file``/``media`` and keep the document
        # path, so only the native context should trigger auto-transcription.
        return MessageType.VOICE
    return _resolve_preferred_message_type(
        getattr(context, "preferred_message_type", "text"),
        media_types,
    )


def extract_text_from_raw_content(*, msg_type: str, raw_content: str) -> Optional[str]:
    normalized = normalize_feishu_message(message_type=msg_type, raw_content=raw_content)
    text = normalized.text_content if isinstance(normalized.text_content, str) else ""
    text = text.strip()
    if text:
        return text

    metadata = normalized.metadata if isinstance(normalized.metadata, dict) else {}
    placeholder = metadata.get("placeholder_text")
    if isinstance(placeholder, str):
        placeholder = placeholder.strip()
        if placeholder:
            return placeholder

    return _fallback_text_for_type(normalized.raw_type or msg_type)


def build_extracted_content(
    *,
    raw_message_type: str,
    context: FeishuMessageContext,
    media_urls: Sequence[str] = (),
    media_types: Sequence[str] = (),
    injected_text: str = "",
) -> FeishuExtractedContent:
    normalized_raw_type = str(raw_message_type or context.content_type or "").strip().lower()
    normalized_media_urls = tuple(str(url or "") for url in media_urls if str(url or ""))
    normalized_media_types = tuple(str(media_type or "") for media_type in media_types if str(media_type or ""))
    resolved_message_type = resolve_feishu_context_message_type(context, normalized_media_types)

    text = str(context.content or "")
    if normalized_raw_type in _MEDIA_ONLY_RAW_TYPES:
        text = ""

    if (
        injected_text
        and resolved_message_type in _TEXT_INJECTABLE_MESSAGE_TYPES
        and len(normalized_media_urls) == 1
        and str(getattr(context, "preferred_message_type", "") or "").strip().lower() in _TEXT_INJECTABLE_PREFERENCES
    ):
        text = injected_text

    return FeishuExtractedContent(
        text=text,
        message_type=resolved_message_type,
        media_urls=normalized_media_urls,
        media_types=normalized_media_types,
        raw_message_type=normalized_raw_type,
        message_context=context,
    )


def should_ignore_extracted_content(content: FeishuExtractedContent) -> bool:
    return content.message_type == MessageType.TEXT and not content.text and not content.media_urls


def resolve_reply_to_message_id(
    *,
    parent_id: Optional[str] = None,
    root_id: Optional[str] = None,
    upper_message_id: Optional[str] = None,
) -> Optional[str]:
    for candidate in (parent_id, root_id, upper_message_id):
        normalized = str(candidate or "").strip()
        if normalized:
            return normalized
    return None


def build_reply_context(
    *,
    parent_id: Optional[str] = None,
    root_id: Optional[str] = None,
    upper_message_id: Optional[str] = None,
    quoted_context: Optional[FeishuQuotedContext] = None,
) -> FeishuReplyContextData:
    reply_to_text = quoted_context.display_text or None if quoted_context else None
    reply_to_media_urls = tuple(str(url or "") for url in (quoted_context.media_urls if quoted_context else ()) if str(url or ""))
    reply_to_media_types = tuple(
        str(media_type or "")
        for media_type in (quoted_context.media_types if quoted_context else ())
        if str(media_type or "")
    )
    return FeishuReplyContextData(
        reply_to_message_id=resolve_reply_to_message_id(
            parent_id=parent_id,
            root_id=root_id,
            upper_message_id=upper_message_id,
        ),
        reply_to_text=reply_to_text,
        reply_to_media_urls=reply_to_media_urls,
        reply_to_media_types=reply_to_media_types,
        quoted_context=quoted_context,
    )


def resolve_feishu_source_chat_type(*, chat_info: Dict[str, Any], event_chat_type: str) -> str:
    resolved = str(chat_info.get("type") or "").strip().lower()
    if resolved in {"group", "forum"}:
        return resolved
    if event_chat_type == "p2p":
        return "dm"
    return "group"


async def build_feishu_sender_profile(
    *,
    sender_id: Any,
    resolve_display_name: Callable[[Optional[str]], Awaitable[Optional[str]]],
) -> FeishuSenderProfile:
    open_id = getattr(sender_id, "open_id", None) or None
    user_id = getattr(sender_id, "user_id", None) or None
    union_id = getattr(sender_id, "union_id", None) or None
    primary_id = open_id or user_id
    auth_user_ids = tuple(
        dict.fromkeys(
            str(value).strip()
            for value in (user_id, open_id, union_id)
            if str(value or "").strip()
        )
    )
    display_name = await resolve_display_name(primary_id or union_id)
    return FeishuSenderProfile(
        user_id=primary_id,
        user_name=display_name,
        user_id_alt=union_id,
        auth_user_ids=auth_user_ids,
    )


def build_message_event(
    *,
    extracted: FeishuExtractedContent,
    source: SessionSource,
    raw_message: Any,
    message_id: Optional[str],
    reply_context: Optional[FeishuReplyContextData] = None,
    auto_skill: Optional[str | list[str]] = None,
    channel_prompt: Optional[str] = None,
    internal: bool = False,
    platform_auth_passed: bool = False,
    timestamp: Optional[datetime] = None,
) -> MessageEvent:
    reply = reply_context or FeishuReplyContextData()
    message_type = coerce_command_message_type(text=extracted.text, message_type=extracted.message_type)
    return MessageEvent(
        text=extracted.text,
        message_type=message_type,
        source=source,
        raw_message=raw_message,
        message_id=str(message_id) if message_id else None,
        media_urls=list(extracted.media_urls),
        media_types=list(extracted.media_types),
        reply_to_message_id=reply.reply_to_message_id,
        reply_to_text=reply.reply_to_text,
        reply_to_media_urls=list(reply.reply_to_media_urls),
        reply_to_media_types=list(reply.reply_to_media_types),
        quoted_context=reply.quoted_context,
        auto_skill=auto_skill,
        channel_prompt=channel_prompt,
        internal=internal,
        platform_auth_passed=platform_auth_passed,
        timestamp=timestamp or datetime.now(),
    )


def coerce_command_message_type(*, text: str, message_type: MessageType) -> MessageType:
    return MessageType.COMMAND if message_type == MessageType.TEXT and str(text or "").startswith("/") else message_type


def build_feishu_inbound_content_bridge(
    *,
    message: Any,
    hydrated_items: Sequence[Any] = (),
    resolve_sender_name_sync: Optional[callable] = None,
    media_urls: Sequence[str] = (),
    media_types: Sequence[str] = (),
    injected_text: str = "",
    build_message_context=None,
) -> FeishuInboundContentBridge:
    if build_message_context is None:
        from .lookup import build_feishu_message_context as _build_message_context
        build_message_context = _build_message_context

    context = build_message_context(
        message_id=str(getattr(message, "message_id", "") or ""),
        message_type=str(getattr(message, "message_type", "") or ""),
        raw_content=str(getattr(message, "content", "") or ""),
        response_items=list(hydrated_items) if hydrated_items else None,
        chat_id=str(getattr(message, "chat_id", "") or ""),
        chat_type=str(getattr(message, "chat_type", "") or ""),
        root_id=getattr(message, "root_id", None) or None,
        parent_id=getattr(message, "parent_id", None) or None,
        thread_id=getattr(message, "thread_id", None) or None,
        resolve_sender_name_sync=resolve_sender_name_sync,
    )
    return build_extracted_content(
        raw_message_type=str(getattr(message, "message_type", "") or context.content_type or ""),
        context=context,
        media_urls=media_urls,
        media_types=media_types,
        injected_text=injected_text,
    )


def resolve_feishu_reply_to_message_id(message: Any) -> Optional[str]:
    return resolve_reply_to_message_id(
        parent_id=getattr(message, "parent_id", None),
        root_id=getattr(message, "root_id", None),
        upper_message_id=getattr(message, "upper_message_id", None),
    )


def build_feishu_reply_context_bridge(
    *,
    message: Any,
    quoted_context: Optional[FeishuQuotedContext] = None,
) -> FeishuReplyContextBridge:
    return build_reply_context(
        parent_id=getattr(message, "parent_id", None),
        root_id=getattr(message, "root_id", None),
        upper_message_id=getattr(message, "upper_message_id", None),
        quoted_context=quoted_context,
    )


def build_feishu_message_event(
    *,
    data: Any,
    message: Any,
    source: SessionSource,
    inbound_content: FeishuInboundContentBridge,
    reply_context: Optional[FeishuReplyContextBridge] = None,
    auto_skill: Optional[str | list[str]] = None,
    channel_prompt: Optional[str] = None,
    internal: bool = False,
    platform_auth_passed: bool = False,
    timestamp: Optional[datetime] = None,
) -> MessageEvent:
    return build_message_event(
        extracted=inbound_content,
        source=source,
        raw_message=data,
        message_id=str(getattr(message, "message_id", "") or "") or None,
        reply_context=reply_context,
        auto_skill=auto_skill,
        channel_prompt=channel_prompt,
        internal=internal,
        platform_auth_passed=platform_auth_passed,
        timestamp=timestamp,
    )


# Generic alias names for incremental wiring and package re-exports.
resolve_media_message_type = resolve_feishu_media_message_type
resolve_normalized_message_type = resolve_feishu_normalized_message_type
resolve_message_context_type = resolve_feishu_context_message_type
resolve_context_message_type = resolve_feishu_context_message_type
extract_message_content = build_feishu_inbound_content_bridge
build_inbound_message_event = build_feishu_message_event


def _fallback_text_for_type(message_type: str) -> Optional[str]:
    return {
        "image": FALLBACK_IMAGE_TEXT,
        "post": FALLBACK_POST_TEXT,
        "merge_forward": FALLBACK_FORWARD_TEXT,
        "share_chat": FALLBACK_SHARE_CHAT_TEXT,
        "interactive": FALLBACK_INTERACTIVE_TEXT,
        "card": FALLBACK_INTERACTIVE_TEXT,
        "file": FALLBACK_ATTACHMENT_TEXT,
        "audio": FALLBACK_ATTACHMENT_TEXT,
        "media": FALLBACK_ATTACHMENT_TEXT,
    }.get(str(message_type or "").strip().lower())


def _publish_package_aliases() -> None:
    import sys

    package = sys.modules.get(__package__)
    if package is None:
        return
    aliases = {
        "FeishuInboundContentBridge": FeishuInboundContentBridge,
        "FeishuReplyContextBridge": FeishuReplyContextBridge,
        "build_feishu_inbound_content_bridge": build_feishu_inbound_content_bridge,
        "build_feishu_message_event": build_feishu_message_event,
        "build_feishu_reply_context_bridge": build_feishu_reply_context_bridge,
        "resolve_feishu_context_message_type": resolve_feishu_context_message_type,
        "resolve_feishu_media_message_type": resolve_feishu_media_message_type,
        "resolve_feishu_normalized_message_type": resolve_feishu_normalized_message_type,
        "resolve_feishu_reply_to_message_id": resolve_feishu_reply_to_message_id,
    }
    for name, value in aliases.items():
        setattr(package, name, value)


_publish_package_aliases()


__all__ = [
    "FeishuInboundContentBridge",
    "FeishuExtractedContent",
    "FeishuReplyContextBridge",
    "FeishuReplyContextData",
    "build_extracted_content",
    "build_feishu_inbound_content_bridge",
    "build_feishu_message_event",
    "build_feishu_reply_context_bridge",
    "build_inbound_message_event",
    "build_message_event",
    "build_reply_context",
    "coerce_command_message_type",
    "extract_message_content",
    "extract_text_from_raw_content",
    "resolve_context_message_type",
    "resolve_feishu_context_message_type",
    "resolve_feishu_media_message_type",
    "resolve_feishu_normalized_message_type",
    "resolve_feishu_reply_to_message_id",
    "resolve_media_message_type",
    "resolve_message_context_type",
    "resolve_normalized_message_type",
    "resolve_reply_to_message_id",
    "should_ignore_extracted_content",
]
