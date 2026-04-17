"""Feishu inbound normalization helpers.

Incremental migration layer inspired by openclaw-lark's parse/enrich/dispatch
split. These helpers let the legacy Hermes Feishu adapter keep its outbound and
streaming UX while moving inbound message normalization, lookup, and quoted
context hydration into dedicated modules.
"""

from .types import FeishuMessageContext, FeishuQuotedContext, FeishuResourceDescriptor
from .lookup import (
    build_feishu_message_context,
    build_feishu_quoted_context,
    build_resource_descriptors,
    extract_message_items,
)
from .render import render_quoted_context_block

__all__ = [
    "FeishuMessageContext",
    "FeishuQuotedContext",
    "FeishuResourceDescriptor",
    "build_feishu_message_context",
    "build_feishu_quoted_context",
    "build_resource_descriptors",
    "extract_message_items",
    "render_quoted_context_block",
]
