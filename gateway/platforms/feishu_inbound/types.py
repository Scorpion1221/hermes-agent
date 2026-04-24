from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

FeishuResourceType = Literal["image", "file", "audio", "video", "sticker"]


@dataclass(frozen=True)
class FeishuResourceDescriptor:
    type: FeishuResourceType
    file_key: str
    file_name: str = ""
    duration: Optional[int] = None
    cover_image_key: str = ""


@dataclass(frozen=True)
class FeishuMessageContext:
    message_id: str
    content: str
    content_type: str
    preferred_message_type: str = "text"
    relation_kind: str = "plain"
    raw_content: str = ""
    chat_id: str = ""
    chat_type: str = ""
    root_id: Optional[str] = None
    parent_id: Optional[str] = None
    thread_id: Optional[str] = None
    sender_id: str = ""
    sender_name: str = ""
    resource_descriptors: tuple[FeishuResourceDescriptor, ...] = ()
    mentioned_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeishuQuotedContext:
    message_id: str
    kind: str
    text: str = ""
    summary: str = ""
    sender_name: str = ""
    media_urls: tuple[str, ...] = ()
    media_types: tuple[str, ...] = ()
    stable_ref: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def display_text(self) -> str:
        return (self.summary or self.text or "").strip()
