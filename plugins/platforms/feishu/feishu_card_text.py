"""Feishu card JSON (``json_card``) rendered as readable Markdown.

``GET im/v1/messages`` with ``card_msg_content_type=raw_card_content`` returns an interactive
message body as ``{"json_card": "<JSON string>", "card_schema": 1 | 2, ...}``. Without that query a
Card JSON 2.0 card (the bot's own streaming replies among them) comes back as a "please upgrade
your client" placeholder, so quoting or forwarding one surfaced nothing readable.

The decoded card is a tree of ``{"tag", "id", "property": {...}}`` nodes under
``body.property.elements`` (2.0; ``body.elements`` in 1.0). ``markdown`` containers mix inline
nodes (``plain_text``, ``code_span``, ``link``, ``br``, ``text_tag``, ``at``) with blocks (``heading``, ``table``,
``list``, ``blockquote``, ``code_block``, ``hr``). Unknown containers recurse into their
``elements`` / ``columns`` / ``actions`` so new tags degrade to their text instead of vanishing;
button labels are listed once at the end, as the legacy card walker does.
"""

from __future__ import annotations

import json
from typing import Any

MAX_CARD_TEXT_CHARS = 8000
_MAX_DEPTH = 32  # card trees are shallow; this only stops a pathological payload recursing away
_STYLE_MARKS = (("bold", "**"), ("italic", "*"), ("strikethrough", "~~"))


def render_json_card(payload: Any) -> str:
    """Readable text of a ``raw_card_content`` interactive body; "" when it carries no json_card."""
    raw = payload.get("json_card") if isinstance(payload, dict) else None
    if not isinstance(raw, str) or not raw:
        return ""
    try:
        card = json.loads(raw)
    except ValueError:
        return ""
    if not isinstance(card, dict):
        return ""
    header = _prop(card.get("header"))
    tags = (_text_tag(tag, 0) for tag in _list(header.get("textTagList")))
    title = " ".join(part for part in (_inline(header.get("title"), 0).strip(), *tags) if part)
    blocks = [title, _inline(header.get("subtitle"), 0).strip()]
    blocks += _render_blocks(_children(card.get("body")), 0)
    labels = _button_labels(card.get("body"), 0)
    if labels:
        blocks.append(f"Actions: {', '.join(labels)}")
    text = "\n\n".join(block for block in blocks if block).strip()
    return text if len(text) <= MAX_CARD_TEXT_CHARS else text[:MAX_CARD_TEXT_CHARS].rstrip() + "…"


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _prop(node: Any) -> dict[str, Any]:
    prop = node.get("property") if isinstance(node, dict) else None
    return prop if isinstance(prop, dict) else {}


def _children(node: Any) -> list[Any]:
    """Child nodes of a container (``elements``, ``column_set`` columns, ``action`` buttons).

    A 1.0 card's body keeps ``elements`` beside ``property`` rather than inside it.
    """
    if not isinstance(node, dict):
        return []
    prop = _prop(node)
    for key in ("elements", "columns", "actions"):
        if isinstance(prop.get(key), list):
            return prop[key]
    return _list(node.get("elements"))


def _button_labels(node: Any, depth: int) -> list[str]:
    labels: list[str] = []
    for child in _children(node) if depth <= _MAX_DEPTH else ():
        label = _inline(_prop(child).get("text"), depth + 1).strip() if _tag(child) == "button" else ""
        labels.extend([label] if label else _button_labels(child, depth + 1))
    return labels


def _tag(node: Any) -> str:
    return str(node.get("tag") or "").strip().lower() if isinstance(node, dict) else ""


def _render_blocks(nodes: list[Any], depth: int) -> list[str]:
    """A container's children: each run of inline nodes is one paragraph, blocks stand alone."""
    blocks: list[str] = []
    paragraph: list[str] = []

    def flush() -> None:
        text = "".join(paragraph).strip()
        if text:
            blocks.append(text)
        paragraph.clear()

    if depth > _MAX_DEPTH:
        return blocks
    for node in nodes:
        tag = _tag(node)
        if tag in _INLINE:
            paragraph.append(_INLINE[tag](node, depth))
            continue
        flush()
        if tag in _BLOCKS:
            blocks.append(_BLOCKS[tag](node, depth))
        elif tag != "button" and (children := _children(node)):
            blocks.extend(_render_blocks(children, depth + 1))
        else:
            blocks.append(_inline(node, depth).strip())
    flush()
    return [block for block in blocks if block]


def _inline(node: Any, depth: int) -> str:
    """Flatten any node to one run of text (table cells, list items, headings, header titles)."""
    if isinstance(node, str):
        return node
    if depth > _MAX_DEPTH or not isinstance(node, dict):
        return ""
    tag = _tag(node)
    if tag in _INLINE:
        return _INLINE[tag](node, depth)
    if tag in _BLOCKS:
        return " ".join(_BLOCKS[tag](node, depth).split("\n"))
    if tag == "button":
        return ""  # labels are listed once, in the trailing "Actions:" line
    children = _children(node)
    if children:
        return "".join(_inline(child, depth + 1) for child in children)
    prop = _prop(node)
    if isinstance(prop.get("text"), dict):  # div / note style leaves wrap a text node
        return _inline(prop["text"], depth + 1)
    return prop["content"] if isinstance(prop.get("content"), str) else ""


def _join_inline(nodes: list[Any], depth: int) -> str:
    return "".join(_inline(node, depth + 1) for node in nodes).strip()


# --- inline nodes ---

def _plain_text(node: Any, _depth: int) -> str:
    prop = _prop(node)
    text = str(prop.get("content") or "")
    style = prop.get("textStyle")
    attributes = _list(style.get("attributes")) if isinstance(style, dict) else []
    if text.strip():
        for attribute, mark in _STYLE_MARKS:
            if attribute in attributes:
                text = f"{mark}{text}{mark}"
    return text


def _code_span(node: Any, _depth: int) -> str:
    content = str(_prop(node).get("content") or "")
    return f"`{content}`" if content else ""


def _line_break(_node: Any, _depth: int) -> str:
    return "\n"


def _link(node: Any, depth: int) -> str:
    prop = _prop(node)
    url = prop.get("url")
    url = str((url.get("url") if isinstance(url, dict) else url) or "")
    text = str(prop.get("content") or "") or _join_inline(_children(node), depth)
    return f"[{text}]({url})" if text and url else text or url


def _text_tag(node: Any, depth: int) -> str:
    text = _inline(_prop(node).get("text"), depth + 1).strip()
    return f"[{text}]" if text else ""


def _mention(_node: Any, _depth: int) -> str:
    return "@user"  # the node carries only the user's id, which never reaches the model


# --- block nodes ---

def _heading(node: Any, depth: int) -> str:
    try:
        level = min(max(int(_prop(node).get("level") or 1), 1), 6)
    except (TypeError, ValueError):
        level = 1
    text = _join_inline(_children(node), depth)
    return f"{'#' * level} {text}" if text else ""


def _code_block(node: Any, _depth: int) -> str:
    prop = _prop(node)
    language = str(prop.get("language") or "").strip()
    if language.lower() in {"plain_text", "plaintext"}:
        language = ""
    lines = [
        "".join(str(part.get("content") or "") for part in _list(row.get("contents")) if isinstance(part, dict))
        for row in _list(prop.get("contents")) if isinstance(row, dict)
    ]
    return f"```{language}\n" + "\n".join(lines) + "\n```"


def _cell(value: Any, depth: int) -> str:
    if isinstance(value, dict):
        value = _inline(value.get("data") if "data" in value else value, depth + 1)
    text = value if isinstance(value, str) else "" if value is None else str(value)
    return " ".join(text.split()).replace("|", "\\|")


def _table(node: Any, depth: int) -> str:
    prop = _prop(node)
    columns = [column for column in _list(prop.get("columns")) if isinstance(column, dict)]
    if not columns:
        return ""
    keys = [str(column.get("name") or index) for index, column in enumerate(columns)]
    rows = [
        [_cell(row.get(key), depth) for key in keys]
        for row in _list(prop.get("rows")) if isinstance(row, dict)
    ]
    header = [_cell(column.get("displayName") or column.get("name"), depth) for column in columns]
    return "\n".join("| " + " | ".join(cells) + " |" for cells in [header, ["---"] * len(columns), *rows])


def _list_block(node: Any, depth: int) -> str:
    lines: list[str] = []
    counters: dict[int, int] = {}
    for item in _list(_prop(node).get("items")):
        if not isinstance(item, dict):
            continue
        level = item.get("level") if isinstance(item.get("level"), int) else 0
        level = min(max(level, 0), 8)
        counters = {lvl: count for lvl, count in counters.items() if lvl <= level}
        counters[level] = counters.get(level, 0) + 1
        marker = f"{counters[level]}." if item.get("type") == "ol" else "-"
        lines.append(f"{'  ' * level}{marker} {_join_inline(_list(item.get('elements')), depth)}")
    return "\n".join(lines)


def _blockquote(node: Any, depth: int) -> str:
    inner = "\n\n".join(_render_blocks(_children(node), depth + 1))
    return "\n".join(f"> {line}" if line else ">" for line in inner.split("\n")) if inner else ""


def _rule(_node: Any, _depth: int) -> str:
    return "---"


_INLINE = {
    "plain_text": _plain_text, "code_span": _code_span, "br": _line_break, "link": _link,
    "text_tag": _text_tag, "at": _mention,
}
_BLOCKS = {
    "heading": _heading, "code_block": _code_block, "table": _table, "list": _list_block,
    "blockquote": _blockquote, "hr": _rule,
}
