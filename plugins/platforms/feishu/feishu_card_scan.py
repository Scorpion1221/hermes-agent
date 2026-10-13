"""Find a card's title and first text in a Feishu interactive-card payload (moved out of ``adapter.py``).

``adapter`` text helpers are reached through the module at call time: ``adapter`` imports this one.
"""

from __future__ import annotations

import sys
from typing import Any


def _adapter():
    return sys.modules[__package__ + ".adapter"]


def _find_header_title(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    header = payload.get("header")
    if not isinstance(header, dict):
        return ""
    title = header.get("title")
    if isinstance(title, dict):
        return _adapter()._first_non_empty_text(title.get("content"), title.get("text"), title.get("name"))
    return _adapter()._normalize_feishu_text(str(title or ""))


def _find_first_text(payload: Any, *, keys: tuple[str, ...]) -> str:
    for node in _walk_nodes(payload):
        if not isinstance(node, dict):
            continue
        for key in keys:
            value = node.get(key)
            if isinstance(value, str):
                normalized = _adapter()._normalize_feishu_text(value)
                if normalized:
                    return normalized
    return ""


def _walk_nodes(value: Any):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk_nodes(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_nodes(item)
