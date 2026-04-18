"""Feishu inbound normalization helpers.

Incremental migration layer inspired by openclaw-lark's parse/enrich/dispatch
split. These helpers let the legacy Hermes Feishu adapter keep its outbound and
streaming UX while moving inbound message normalization, lookup, and quoted
context hydration into dedicated modules.
"""

from .types import FeishuMessageContext, FeishuQuotedContext, FeishuResourceDescriptor
from .parse import (
    FeishuNormalizedMessage,
    FeishuPostMediaRef,
    FeishuPostParseResult,
    normalize_feishu_message,
    parse_feishu_post_payload,
)
from .lookup import (
    build_feishu_message_context,
    build_feishu_quoted_context,
    build_resource_descriptors,
    extract_message_items,
)
from .bridge import (
    FeishuSenderProfile,
    build_inbound_message_event,
    build_feishu_sender_profile,
    extract_message_content,
    extract_text_from_raw_content,
    resolve_context_message_type,
    resolve_media_message_type,
    resolve_feishu_source_chat_type,
)
from .media_index import (
    FeishuMediaIndexEntry,
    make_media_index_key,
    get_feishu_media_index_entry,
    put_feishu_media_index_entry,
    remove_feishu_media_index_entry,
)
from .render import render_quoted_context_block
from .policy import (
    allow_feishu_group_message,
    feishu_message_mentions_bot,
    feishu_post_mentions_bot,
    should_accept_feishu_group_message,
)
from .comment_target import (
    FeishuCommentTarget,
    build_feishu_comment_target,
    parse_feishu_comment_target,
    is_feishu_comment_target,
)
from .comment_context import (
    FeishuDriveCommentEvent,
    FeishuDriveCommentTurn,
    build_drive_comment_prompt,
    parse_feishu_drive_comment_notice_event_payload,
    resolve_drive_comment_event_turn,
)
from .user_name_cache import (
    DEFAULT_FEISHU_SENDER_NAME_TTL_SECONDS,
    FeishuSenderNameCache,
    FeishuSenderNameCacheEntry,
    coerce_feishu_sender_display_name,
    resolve_feishu_sender_display_name,
    resolve_feishu_sender_display_names,
    resolve_feishu_sender_name,
    resolve_feishu_sender_names,
)

__all__ = [
    "FeishuMessageContext",
    "FeishuQuotedContext",
    "FeishuResourceDescriptor",
    "FeishuNormalizedMessage",
    "FeishuPostMediaRef",
    "FeishuPostParseResult",
    "normalize_feishu_message",
    "parse_feishu_post_payload",
    "build_feishu_message_context",
    "build_feishu_quoted_context",
    "build_resource_descriptors",
    "extract_message_items",
    "build_inbound_message_event",
    "FeishuSenderProfile",
    "build_feishu_sender_profile",
    "extract_message_content",
    "extract_text_from_raw_content",
    "resolve_context_message_type",
    "resolve_media_message_type",
    "resolve_feishu_source_chat_type",
    "FeishuMediaIndexEntry",
    "make_media_index_key",
    "get_feishu_media_index_entry",
    "put_feishu_media_index_entry",
    "remove_feishu_media_index_entry",
    "render_quoted_context_block",
    "DEFAULT_FEISHU_SENDER_NAME_TTL_SECONDS",
    "FeishuSenderNameCache",
    "FeishuSenderNameCacheEntry",
    "coerce_feishu_sender_display_name",
    "resolve_feishu_sender_display_name",
    "resolve_feishu_sender_display_names",
    "resolve_feishu_sender_name",
    "resolve_feishu_sender_names",
    "allow_feishu_group_message",
    "feishu_message_mentions_bot",
    "feishu_post_mentions_bot",
    "should_accept_feishu_group_message",
    "FeishuCommentTarget",
    "build_feishu_comment_target",
    "parse_feishu_comment_target",
    "is_feishu_comment_target",
    "FeishuDriveCommentEvent",
    "FeishuDriveCommentTurn",
    "build_drive_comment_prompt",
    "parse_feishu_drive_comment_notice_event_payload",
    "resolve_drive_comment_event_turn",
]
