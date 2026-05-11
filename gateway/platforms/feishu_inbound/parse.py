from __future__ import annotations

"""Pure Feishu inbound parse and normalization helpers.

This module extracts the legacy inbound content parsing logic from
``gateway.platforms.feishu`` into a lightweight module that can be reused by
other Feishu inbound helpers without pulling in the full adapter.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

FALLBACK_POST_TEXT = "[Rich text message]"
FALLBACK_FORWARD_TEXT = "[Merged forward message]"
FALLBACK_SHARE_CHAT_TEXT = "[Shared chat]"
FALLBACK_INTERACTIVE_TEXT = "[Interactive message]"
FALLBACK_IMAGE_TEXT = "[Image]"
FALLBACK_ATTACHMENT_TEXT = "[Attachment]"

_PREFERRED_LOCALES = ("zh_cn", "en_us")
_MARKDOWN_SPECIAL_CHARS_RE = re.compile(r"([\\`*_{}\[\]()#+\-!|>~])")
_MENTION_PLACEHOLDER_RE = re.compile(r"@_user_\d+")
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_WHITESPACE_RE = re.compile(r"\s+")
_SUPPORTED_CARD_TEXT_KEYS = (
    "title",
    "text",
    "content",
    "label",
    "value",
    "name",
    "summary",
    "subtitle",
    "description",
    "placeholder",
    "hint",
)
_SKIP_TEXT_KEYS = {
    "tag",
    "type",
    "msg_type",
    "message_type",
    "chat_id",
    "open_chat_id",
    "share_chat_id",
    "file_key",
    "image_key",
    "user_id",
    "open_id",
    "union_id",
    "url",
    "href",
    "link",
    "token",
    "template",
    "locale",
    # CardKit json_card noise
    "id",
    "element_id",
    "textAlign",
    "textColor",
    "textSize",
}


@dataclass(frozen=True)
class FeishuPostMediaRef:
    file_key: str
    file_name: str = ""
    resource_type: str = "file"


@dataclass(frozen=True)
class FeishuMentionRef:
    name: str = ""
    open_id: str = ""
    is_all: bool = False
    is_self: bool = False


@dataclass(frozen=True)
class FeishuPostParseResult:
    text_content: str
    image_keys: list[str] = field(default_factory=list)
    media_refs: list[FeishuPostMediaRef] = field(default_factory=list)
    mentioned_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FeishuNormalizedMessage:
    raw_type: str
    text_content: str
    preferred_message_type: str = "text"
    image_keys: list[str] = field(default_factory=list)
    media_refs: list[FeishuPostMediaRef] = field(default_factory=list)
    mentions: list[FeishuMentionRef] = field(default_factory=list)
    relation_kind: str = "plain"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def mentioned_ids(self) -> list[str]:
        explicit = getattr(self, "_mentioned_ids", None)
        if explicit is not None:
            return list(explicit)
        return [ref.open_id for ref in self.mentions if ref.open_id]


def parse_feishu_post_payload(payload: Any) -> FeishuPostParseResult:
    resolved = _resolve_post_payload(payload)
    if not resolved:
        return FeishuPostParseResult(text_content=FALLBACK_POST_TEXT)

    image_keys: list[str] = []
    media_refs: list[FeishuPostMediaRef] = []
    mentioned_ids: list[str] = []
    parts: list[str] = []

    title = _normalize_feishu_text(str(resolved.get("title", "")).strip())
    if title:
        parts.append(title)

    for row in resolved.get("content", []) or []:
        if not isinstance(row, list):
            continue
        row_text = _normalize_feishu_text(
            "".join(_render_post_element(item, image_keys, media_refs, mentioned_ids) for item in row)
        )
        if row_text:
            parts.append(row_text)

    return FeishuPostParseResult(
        text_content="\n".join(parts).strip() or FALLBACK_POST_TEXT,
        image_keys=image_keys,
        media_refs=media_refs,
        mentioned_ids=mentioned_ids,
    )


def normalize_feishu_message(*, message_type: str, raw_content: str) -> FeishuNormalizedMessage:
    normalized_type = str(message_type or "").strip().lower()
    payload = _load_feishu_payload(raw_content)

    if normalized_type == "text":
        return FeishuNormalizedMessage(
            raw_type=normalized_type,
            text_content=_normalize_feishu_text(str(payload.get("text", "") or "")),
        )
    if normalized_type == "post":
        parsed_post = parse_feishu_post_payload(payload)
        normalized = FeishuNormalizedMessage(
            raw_type=normalized_type,
            text_content=parsed_post.text_content,
            image_keys=list(parsed_post.image_keys),
            media_refs=list(parsed_post.media_refs),
            relation_kind="post",
        )
        object.__setattr__(normalized, "_mentioned_ids", list(parsed_post.mentioned_ids))
        return normalized
    if normalized_type == "image":
        image_key = str(payload.get("image_key", "") or "").strip()
        alt_text = _normalize_feishu_text(
            str(payload.get("text", "") or "")
            or str(payload.get("alt", "") or "")
            or FALLBACK_IMAGE_TEXT
        )
        return FeishuNormalizedMessage(
            raw_type=normalized_type,
            text_content=alt_text if alt_text != FALLBACK_IMAGE_TEXT else "",
            preferred_message_type="photo",
            image_keys=[image_key] if image_key else [],
            relation_kind="image",
        )
    if normalized_type in {"file", "audio", "media"}:
        media_ref = _build_media_ref_from_payload(payload, resource_type=normalized_type)
        placeholder = _attachment_placeholder(media_ref.file_name)
        return FeishuNormalizedMessage(
            raw_type=normalized_type,
            text_content="",
            preferred_message_type="audio" if normalized_type == "audio" else "document",
            media_refs=[media_ref] if media_ref.file_key else [],
            relation_kind=normalized_type,
            metadata={"placeholder_text": placeholder},
        )
    if normalized_type == "merge_forward":
        return _normalize_merge_forward_message(payload)
    if normalized_type == "share_chat":
        return _normalize_share_chat_message(payload)
    if normalized_type in {"interactive", "card"}:
        return _normalize_interactive_message(normalized_type, payload)

    return FeishuNormalizedMessage(raw_type=normalized_type, text_content="")


def _resolve_post_payload(payload: Any) -> dict[str, Any]:
    direct = _to_post_payload(payload)
    if direct:
        return direct
    if not isinstance(payload, dict):
        return {}

    wrapped = payload.get("post")
    wrapped_direct = _resolve_locale_payload(wrapped)
    if wrapped_direct:
        return wrapped_direct
    return _resolve_locale_payload(payload)


def _resolve_locale_payload(payload: Any) -> dict[str, Any]:
    direct = _to_post_payload(payload)
    if direct:
        return direct
    if not isinstance(payload, dict):
        return {}

    for key in _PREFERRED_LOCALES:
        candidate = _to_post_payload(payload.get(key))
        if candidate:
            return candidate
    for value in payload.values():
        candidate = _to_post_payload(value)
        if candidate:
            return candidate
    return {}


def _to_post_payload(candidate: Any) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        return {}
    content = candidate.get("content")
    if not isinstance(content, list):
        return {}
    return {
        "title": str(candidate.get("title", "") or ""),
        "content": content,
    }


def _escape_markdown_text(text: str) -> str:
    return _MARKDOWN_SPECIAL_CHARS_RE.sub(r"\\\1", text)


def _to_boolean(value: Any) -> bool:
    return value is True or value == 1 or value == "true"


def _is_style_enabled(style: dict[str, Any] | None, key: str) -> bool:
    if not style:
        return False
    return _to_boolean(style.get(key))


def _wrap_inline_code(text: str) -> str:
    max_run = max([0, *[len(run) for run in re.findall(r"`+", text)]])
    fence = "`" * (max_run + 1)
    body = f" {text} " if text.startswith("`") or text.endswith("`") else text
    return f"{fence}{body}{fence}"


def _sanitize_fence_language(language: str) -> str:
    return language.strip().replace("\n", " ").replace("\r", " ")


def _render_text_element(element: dict[str, Any]) -> str:
    text = str(element.get("text", "") or "")
    style = element.get("style")
    style_dict = style if isinstance(style, dict) else None

    if _is_style_enabled(style_dict, "code"):
        return _wrap_inline_code(text)

    rendered = _escape_markdown_text(text)
    if not rendered:
        return ""
    if _is_style_enabled(style_dict, "bold"):
        rendered = f"**{rendered}**"
    if _is_style_enabled(style_dict, "italic"):
        rendered = f"*{rendered}*"
    if _is_style_enabled(style_dict, "underline"):
        rendered = f"<u>{rendered}</u>"
    if _is_style_enabled(style_dict, "strikethrough"):
        rendered = f"~~{rendered}~~"
    return rendered


def _render_code_block_element(element: dict[str, Any]) -> str:
    language = _sanitize_fence_language(
        str(element.get("language", "") or "") or str(element.get("lang", "") or "")
    )
    code = (
        str(element.get("text", "") or "") or str(element.get("content", "") or "")
    ).replace("\r\n", "\n")
    trailing_newline = "" if code.endswith("\n") else "\n"
    return f"```{language}\n{code}{trailing_newline}```"


def _render_post_element(
    element: Any,
    image_keys: list[str],
    media_refs: list[FeishuPostMediaRef],
    mentioned_ids: list[str],
) -> str:
    if isinstance(element, str):
        return element
    if not isinstance(element, dict):
        return ""

    tag = str(element.get("tag", "")).strip().lower()
    if tag == "text":
        return _render_text_element(element)
    if tag == "a":
        href = str(element.get("href", "")).strip()
        label = str(element.get("text", href) or "").strip()
        if not label:
            return ""
        escaped_label = _escape_markdown_text(label)
        return f"[{escaped_label}]({href})" if href else escaped_label
    if tag == "at":
        mentioned_id = (
            str(element.get("open_id", "")).strip()
            or str(element.get("user_id", "")).strip()
        )
        if mentioned_id and mentioned_id not in mentioned_ids:
            mentioned_ids.append(mentioned_id)
        display_name = (
            str(element.get("user_name", "")).strip()
            or str(element.get("name", "")).strip()
            or str(element.get("text", "")).strip()
            or mentioned_id
        )
        return f"@{_escape_markdown_text(display_name)}" if display_name else "@"
    if tag in {"img", "image"}:
        image_key = str(element.get("image_key", "")).strip()
        if image_key and image_key not in image_keys:
            image_keys.append(image_key)
        alt = str(element.get("text", "")).strip() or str(element.get("alt", "")).strip()
        return f"[Image: {alt}]" if alt else "[Image]"
    if tag in {"media", "file", "audio", "video"}:
        file_key = str(element.get("file_key", "")).strip()
        file_name = (
            str(element.get("file_name", "")).strip()
            or str(element.get("title", "")).strip()
            or str(element.get("text", "")).strip()
        )
        if file_key:
            media_refs.append(
                FeishuPostMediaRef(
                    file_key=file_key,
                    file_name=file_name,
                    resource_type=tag if tag in {"audio", "video"} else "file",
                )
            )
        return f"[Attachment: {file_name}]" if file_name else "[Attachment]"
    if tag in {"emotion", "emoji"}:
        label = str(element.get("text", "")).strip() or str(element.get("emoji_type", "")).strip()
        return f":{_escape_markdown_text(label)}:" if label else "[Emoji]"
    if tag == "br":
        return "\n"
    if tag in {"hr", "divider"}:
        return "\n\n---\n\n"
    if tag == "code":
        code = str(element.get("text", "") or "") or str(element.get("content", "") or "")
        return _wrap_inline_code(code) if code else ""
    if tag in {"code_block", "pre"}:
        return _render_code_block_element(element)

    nested_parts: list[str] = []
    for key in ("text", "title", "content", "children", "elements"):
        value = element.get(key)
        extracted = _render_nested_post(value, image_keys, media_refs, mentioned_ids)
        if extracted:
            nested_parts.append(extracted)
    return " ".join(part for part in nested_parts if part)


def _render_nested_post(
    value: Any,
    image_keys: list[str],
    media_refs: list[FeishuPostMediaRef],
    mentioned_ids: list[str],
) -> str:
    if isinstance(value, str):
        return _escape_markdown_text(value)
    if isinstance(value, list):
        return " ".join(
            part
            for item in value
            for part in [_render_nested_post(item, image_keys, media_refs, mentioned_ids)]
            if part
        )
    if isinstance(value, dict):
        direct = _render_post_element(value, image_keys, media_refs, mentioned_ids)
        if direct:
            return direct
        return " ".join(
            part
            for item in value.values()
            for part in [_render_nested_post(item, image_keys, media_refs, mentioned_ids)]
            if part
        )
    return ""


def _load_feishu_payload(raw_content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_content) if raw_content else {}
    except json.JSONDecodeError:
        return {"text": raw_content}
    return parsed if isinstance(parsed, dict) else {"content": parsed}


def _normalize_merge_forward_message(payload: dict[str, Any]) -> FeishuNormalizedMessage:
    title = _first_non_empty_text(
        payload.get("title"),
        payload.get("summary"),
        payload.get("preview"),
        _find_first_text(payload, keys=("title", "summary", "preview", "description")),
    )
    entries = _collect_forward_entries(payload)
    lines: list[str] = []
    if title:
        lines.append(title)
    lines.extend(entries[:8])
    text_content = "\n".join(lines).strip() or FALLBACK_FORWARD_TEXT
    return FeishuNormalizedMessage(
        raw_type="merge_forward",
        text_content=text_content,
        relation_kind="merge_forward",
        metadata={"entry_count": len(entries), "title": title},
    )


def _normalize_share_chat_message(payload: dict[str, Any]) -> FeishuNormalizedMessage:
    chat_name = _first_non_empty_text(
        payload.get("chat_name"),
        payload.get("name"),
        payload.get("title"),
        _find_first_text(payload, keys=("chat_name", "name", "title")),
    )
    share_id = _first_non_empty_text(
        payload.get("chat_id"),
        payload.get("open_chat_id"),
        payload.get("share_chat_id"),
    )
    lines = []
    if chat_name:
        lines.append(f"Shared chat: {chat_name}")
    else:
        lines.append(FALLBACK_SHARE_CHAT_TEXT)
    if share_id:
        lines.append(f"Chat ID: {share_id}")
    text_content = "\n".join(lines)
    return FeishuNormalizedMessage(
        raw_type="share_chat",
        text_content=text_content,
        relation_kind="share_chat",
        metadata={"chat_id": share_id, "chat_name": chat_name},
    )


def _normalize_interactive_message(message_type: str, payload: dict[str, Any]) -> FeishuNormalizedMessage:
    # CardKit-hydrated cards come back as {"json_card": "<escaped JSON>"}; unwrap
    # so the walkers below see the actual body/header/elements tree.
    json_card = payload.get("json_card") if isinstance(payload, dict) else None
    if isinstance(json_card, str) and json_card:
        try:
            parsed = json.loads(json_card)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            rendered = render_cardkit_body(parsed)
            if rendered:
                return FeishuNormalizedMessage(
                    raw_type=message_type,
                    text_content=rendered,
                    relation_kind="interactive",
                    metadata={"format": "cardkit_json_card"},
                )
            payload = parsed
    card_payload = payload.get("card") if isinstance(payload.get("card"), dict) else payload
    title = _first_non_empty_text(
        _find_header_title(card_payload),
        payload.get("title"),
        _find_first_text(card_payload, keys=("title", "summary", "subtitle")),
    )
    body_lines = _collect_card_lines(card_payload)
    template_lines = _collect_template_variable_lines(card_payload)
    actions = _collect_action_labels(card_payload)

    lines: list[str] = []
    if title:
        lines.append(title)
    for line in body_lines + template_lines:
        if line != title:
            lines.append(line)
    if actions:
        lines.append(f"Actions: {', '.join(actions)}")

    text_content = "\n".join(_unique_lines(lines)[:12]).strip() or FALLBACK_INTERACTIVE_TEXT
    return FeishuNormalizedMessage(
        raw_type=message_type,
        text_content=text_content,
        relation_kind="interactive",
        metadata={"title": title, "actions": actions},
    )


# ---------------------------------------------------------------------------
# CardKit json_card renderer
#
# When a CardKit card is fetched via im.v1.message.get with
# card_msg_content_type=raw_card_content, Feishu serializes the rendered
# element tree as {"json_card": "<escaped JSON>"}. The tree mirrors the
# Markdown + Feishu rich-text extensions spec documented here:
#   https://open.feishu.cn/document/feishu-cards/card-json-v2-structure
#   https://open.feishu.cn/document/uAjLw4CM/ukzMukzMukzM/feishu-cards/
#     card-json-v2-components/content-components/rich-text
# This renderer walks that tree and emits Markdown. Tag handlers are
# registered in dispatch tables (_CARDKIT_{BLOCK,INLINE}_RENDERERS); adding a
# new tag = one entry + one small function, no control-flow surgery.
# ---------------------------------------------------------------------------

_CARDKIT_MAX_RENDER_CHARS = 4000
_CARDKIT_BLOCK_RENDERERS: dict[str, Any] = {}
_CARDKIT_INLINE_RENDERERS: dict[str, Any] = {}


def render_cardkit_body(node: Any) -> str:
    if not isinstance(node, dict):
        return ""
    body = node.get("body") if isinstance(node.get("body"), dict) else node
    elements = (body.get("property") or {}).get("elements") if isinstance(body, dict) else None
    if not isinstance(elements, list):
        return ""
    blocks = _cardkit_render_blocks(elements)
    text = "\n\n".join(b for b in blocks if b).strip()
    if len(text) > _CARDKIT_MAX_RENDER_CHARS:
        text = text[:_CARDKIT_MAX_RENDER_CHARS].rstrip() + "…"
    return text


def _cardkit_render_blocks(elements: list[Any]) -> list[str]:
    blocks: list[str] = []
    inline_buffer: list[str] = []

    def flush() -> None:
        if not inline_buffer:
            return
        joined = "".join(inline_buffer).strip()
        if joined:
            blocks.append(joined)
        inline_buffer.clear()

    for el in elements or []:
        if not isinstance(el, dict):
            continue
        tag = str(el.get("tag", "")).strip().lower()
        if tag in _CARDKIT_INLINE_RENDERERS:
            inline_buffer.append(_CARDKIT_INLINE_RENDERERS[tag](el))
            continue
        if tag in _CARDKIT_BLOCK_RENDERERS:
            flush()
            rendered = _CARDKIT_BLOCK_RENDERERS[tag](el)
            if rendered:
                blocks.append(rendered)
            continue
        # Unknown or pure container (markdown/paragraph): children decide level.
        children = _cardkit_children(el)
        if isinstance(children, list):
            if _cardkit_children_have_blocks(children):
                nested = _cardkit_render_blocks(children)
                if nested:
                    flush()
                    blocks.extend(nested)
            else:
                inline_buffer.append("".join(_cardkit_render_inline(c) for c in children))
    flush()
    return [b for b in blocks if b]


def _cardkit_render_inline(el: Any) -> str:
    if isinstance(el, str):
        return el
    if not isinstance(el, dict):
        return ""
    tag = str(el.get("tag", "")).strip().lower()
    if tag in _CARDKIT_INLINE_RENDERERS:
        return _CARDKIT_INLINE_RENDERERS[tag](el)
    # Block tags in an inline context: collapse their output to a flat string.
    if tag in _CARDKIT_BLOCK_RENDERERS:
        rendered = _CARDKIT_BLOCK_RENDERERS[tag](el)
        return rendered.replace("\n", " ")
    children = _cardkit_children(el)
    if isinstance(children, list):
        return "".join(_cardkit_render_inline(c) for c in children)
    prop = el.get("property") if isinstance(el.get("property"), dict) else {}
    return str(prop.get("content", "") or "")


def _cardkit_children(el: Any) -> Optional[list[Any]]:
    if not isinstance(el, dict):
        return None
    prop = el.get("property")
    if not isinstance(prop, dict):
        return None
    children = prop.get("elements")
    return children if isinstance(children, list) else None


def _cardkit_children_have_blocks(children: list[Any]) -> bool:
    for c in children:
        if not isinstance(c, dict):
            continue
        t = str(c.get("tag", "")).strip().lower()
        if t in _CARDKIT_BLOCK_RENDERERS:
            return True
        if t not in _CARDKIT_INLINE_RENDERERS:
            sub = _cardkit_children(c)
            if isinstance(sub, list) and _cardkit_children_have_blocks(sub):
                return True
    return False


# --- inline renderers ------------------------------------------------------

def _cardkit_prop(el: Any) -> dict[str, Any]:
    prop = el.get("property") if isinstance(el, dict) else None
    return prop if isinstance(prop, dict) else {}


def _cardkit_inline_plain_text(el: Any) -> str:
    prop = _cardkit_prop(el)
    text = str(prop.get("content", "") or "")
    if not text:
        return ""
    style = prop.get("textStyle") if isinstance(prop.get("textStyle"), dict) else {}
    attrs = style.get("attributes") if isinstance(style.get("attributes"), list) else []
    if text.strip():
        if "bold" in attrs:
            text = f"**{text}**"
        if "italic" in attrs:
            text = f"*{text}*"
        if "strikethrough" in attrs:
            text = f"~~{text}~~"
    return text


def _cardkit_inline_code_span(el: Any) -> str:
    return f"`{_cardkit_prop(el).get('content', '') or ''}`"


def _cardkit_inline_br(_el: Any) -> str:
    return "\n"


def _cardkit_inline_link(el: Any) -> str:
    prop = _cardkit_prop(el)
    text = str(prop.get("content", "") or prop.get("text", "") or "")
    if not text:
        # Nested inline children can also form the visible text.
        children = _cardkit_children(el)
        if isinstance(children, list):
            text = "".join(_cardkit_render_inline(c) for c in children)
    href = str(prop.get("href", "") or prop.get("url", "") or "")
    if href and text:
        return f"[{text}]({href})"
    return text or href


def _cardkit_inline_at(el: Any) -> str:
    prop = _cardkit_prop(el)
    name = str(prop.get("name", "") or prop.get("user_name", "") or prop.get("content", "") or "").strip()
    uid = str(
        prop.get("id", "") or prop.get("user_id", "") or prop.get("open_id", "") or prop.get("union_id", "") or ""
    ).strip()
    if uid == "all" or str(el.get("id", "")).strip().lower() == "all":
        return "@全体成员"
    if name:
        return f"@{name}"
    if uid:
        return f"@{uid}"
    return "@"


def _cardkit_inline_text_tag(el: Any) -> str:
    prop = _cardkit_prop(el)
    text = prop.get("text") if isinstance(prop.get("text"), dict) else None
    if text:
        content = str(text.get("content", "") or "")
    else:
        content = str(prop.get("content", "") or "")
    return f"[{content}]" if content else ""


def _cardkit_inline_number_tag(el: Any) -> str:
    prop = _cardkit_prop(el)
    value = str(prop.get("content", "") or prop.get("text", "") or prop.get("value", "") or "").strip()
    return f"({value})" if value else ""


def _cardkit_inline_font(el: Any) -> str:
    # <font color="red">text</font> — color is semantic noise in LLM context;
    # surface the text verbatim.
    children = _cardkit_children(el)
    if isinstance(children, list):
        return "".join(_cardkit_render_inline(c) for c in children)
    return str(_cardkit_prop(el).get("content", "") or "")


def _cardkit_inline_local_datetime(el: Any) -> str:
    prop = _cardkit_prop(el)
    return str(prop.get("content", "") or prop.get("timestamp", "") or "")


def _cardkit_inline_audio(el: Any) -> str:
    prop = _cardkit_prop(el)
    name = str(prop.get("file_name", "") or prop.get("name", "") or "audio").strip()
    return f"[Audio: {name}]"


def _cardkit_inline_person(el: Any) -> str:
    prop = _cardkit_prop(el)
    name = str(prop.get("user_name", "") or prop.get("name", "") or prop.get("content", "") or "").strip()
    return f"@{name}" if name else "@"


def _cardkit_inline_image(el: Any) -> str:
    prop = _cardkit_prop(el)
    alt = str(prop.get("alt", "") or prop.get("hover_text", "") or prop.get("name", "") or "image").strip()
    img_key = str(prop.get("img_key", "") or prop.get("image_key", "") or "").strip()
    return f"![{alt}]({img_key})" if img_key else f"[{alt}]"


# --- block renderers -------------------------------------------------------

def _cardkit_block_heading(el: Any) -> str:
    prop = _cardkit_prop(el)
    try:
        level = int(prop.get("level") or 1)
    except (TypeError, ValueError):
        level = 1
    level = max(1, min(level, 6))
    children = _cardkit_children(el) or []
    inner = "".join(_cardkit_render_inline(c) for c in children).strip()
    return ("#" * level) + " " + inner if inner else ""


def _cardkit_block_code_block(el: Any) -> str:
    prop = _cardkit_prop(el)
    lang_raw = str(prop.get("language", "") or "").strip()
    lang = "" if lang_raw.lower() in {"", "plain_text", "plaintext"} else lang_raw
    rows = prop.get("contents") if isinstance(prop.get("contents"), list) else []
    lines: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_contents = row.get("contents") if isinstance(row.get("contents"), list) else []
        line = "".join(
            str(c.get("content", "") or "") for c in row_contents if isinstance(c, dict)
        )
        lines.append(line)
    body = "\n".join(lines)
    fence = f"```{lang}" if lang else "```"
    return f"{fence}\n{body}\n```"


def _cardkit_block_table(el: Any) -> str:
    prop = _cardkit_prop(el)
    columns = prop.get("columns") if isinstance(prop.get("columns"), list) else []
    rows = prop.get("rows") if isinstance(prop.get("rows"), list) else []
    if not columns:
        return ""
    headers: list[str] = []
    col_keys: list[str] = []
    for idx, col in enumerate(columns):
        if not isinstance(col, dict):
            headers.append("")
            col_keys.append(str(idx))
            continue
        headers.append(str(col.get("displayName") or col.get("name") or "").strip())
        col_keys.append(str(col.get("name") or idx))
    md = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        if not isinstance(row, dict):
            continue
        cells: list[str] = []
        for key in col_keys:
            cell = row.get(key)
            text = ""
            if isinstance(cell, dict):
                data = cell.get("data") if isinstance(cell.get("data"), dict) else None
                if data is not None:
                    text = _cardkit_render_inline(data)
            cells.append(text.replace("|", "\\|").replace("\n", " ").strip())
        md.append("| " + " | ".join(cells) + " |")
    return "\n".join(md)


def _cardkit_block_list_factory(ordered: bool):
    def render(el: Any) -> str:
        children = _cardkit_children(el) or []
        items: list[str] = []
        for idx, c in enumerate(children, 1):
            if not isinstance(c, dict):
                continue
            sub = _cardkit_children(c)
            if isinstance(sub, list):
                text = "".join(_cardkit_render_inline(x) for x in sub).strip()
            else:
                text = _cardkit_render_inline(c).strip()
            if text:
                prefix = f"{idx}. " if ordered else "- "
                items.append(prefix + text)
        return "\n".join(items)
    return render


def _cardkit_block_blockquote(el: Any) -> str:
    children = _cardkit_children(el) or []
    if _cardkit_children_have_blocks(children):
        inner = "\n\n".join(_cardkit_render_blocks(children))
    else:
        inner = "".join(_cardkit_render_inline(c) for c in children)
    inner = inner.strip()
    if not inner:
        return ""
    return "\n".join(f"> {line}" if line else ">" for line in inner.split("\n"))


def _cardkit_block_hr(_el: Any) -> str:
    return "---"


_CARDKIT_INLINE_RENDERERS.update({
    "plain_text": _cardkit_inline_plain_text,
    "code_span": _cardkit_inline_code_span,
    "br": _cardkit_inline_br,
    "a": _cardkit_inline_link,
    "link": _cardkit_inline_link,
    "at": _cardkit_inline_at,
    "mention": _cardkit_inline_at,
    "text_tag": _cardkit_inline_text_tag,
    "number_tag": _cardkit_inline_number_tag,
    "font": _cardkit_inline_font,
    "local_datetime": _cardkit_inline_local_datetime,
    "audio": _cardkit_inline_audio,
    "person": _cardkit_inline_person,
    "image": _cardkit_inline_image,
})

_CARDKIT_BLOCK_RENDERERS.update({
    "heading": _cardkit_block_heading,
    "code_block": _cardkit_block_code_block,
    "table": _cardkit_block_table,
    "list": _cardkit_block_list_factory(ordered=False),
    "bullet_list": _cardkit_block_list_factory(ordered=False),
    "ordered_list": _cardkit_block_list_factory(ordered=True),
    "blockquote": _cardkit_block_blockquote,
    "hr": _cardkit_block_hr,
})


def _collect_forward_entries(payload: dict[str, Any]) -> list[str]:
    candidates: list[Any] = []
    for key in ("messages", "items", "message_list", "records", "content"):
        value = payload.get(key)
        if isinstance(value, list):
            candidates.extend(value)
    entries: list[str] = []
    for item in candidates:
        if not isinstance(item, dict):
            text = _normalize_feishu_text(str(item or ""))
            if text:
                entries.append(f"- {text}")
            continue
        sender = _first_non_empty_text(
            item.get("sender_name"),
            item.get("user_name"),
            item.get("sender"),
            item.get("name"),
        )
        nested_type = str(item.get("message_type", "") or item.get("msg_type", "")).strip().lower()
        if nested_type == "post":
            body = parse_feishu_post_payload(item.get("content") or item).text_content
        else:
            body = _first_non_empty_text(
                item.get("text"),
                item.get("summary"),
                item.get("preview"),
                item.get("content"),
                _find_first_text(item, keys=("text", "content", "summary", "preview", "title")),
            )
        body = _normalize_feishu_text(body)
        if sender and body:
            entries.append(f"- {sender}: {body}")
        elif body:
            entries.append(f"- {body}")
    return _unique_lines(entries)


def _collect_card_lines(payload: Any) -> list[str]:
    lines = _collect_text_segments(payload, in_rich_block=False)
    normalized = [_normalize_feishu_text(line) for line in lines]
    return _unique_lines([line for line in normalized if line])


def _collect_action_labels(payload: Any) -> list[str]:
    labels: list[str] = []
    for item in _walk_nodes(payload):
        if not isinstance(item, dict):
            continue
        tag = str(item.get("tag", "") or item.get("type", "")).strip().lower()
        if tag not in {"button", "select_static", "overflow", "date_picker", "picker"}:
            continue
        label = _first_non_empty_text(
            item.get("text"),
            item.get("name"),
            item.get("value"),
            _find_first_text(item, keys=("text", "content", "name", "value")),
        )
        if label:
            labels.append(label)
    return _unique_lines(labels)


def _collect_template_variable_lines(payload: Any) -> list[str]:
    candidates: list[Any] = []
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict) and isinstance(data.get("template_variable"), (dict, list)):
            candidates.append(data.get("template_variable"))
        if isinstance(payload.get("template_variable"), (dict, list)):
            candidates.append(payload.get("template_variable"))
    if not candidates:
        return []

    lines: list[str] = []

    def _walk_template_values(value: Any, key: str = "") -> None:
        lowered = key.strip().lower()
        if isinstance(value, str):
            normalized = _normalize_feishu_text(value)
            if (
                normalized
                and lowered not in _SKIP_TEXT_KEYS
                and not lowered.endswith("_id")
                and not lowered.endswith("_url")
            ):
                lines.append(normalized)
            return
        if isinstance(value, list):
            for item in value:
                _walk_template_values(item, key)
            return
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                _walk_template_values(child_value, str(child_key))
            return

    for candidate in candidates:
        _walk_template_values(candidate)
    return _unique_lines(lines)


def _collect_text_segments(value: Any, *, in_rich_block: bool) -> list[str]:
    if isinstance(value, str):
        return [_normalize_feishu_text(value)] if in_rich_block else []
    if isinstance(value, list):
        segments: list[str] = []
        for item in value:
            segments.extend(_collect_text_segments(item, in_rich_block=in_rich_block))
        return segments
    if not isinstance(value, dict):
        return []

    tag = str(value.get("tag", "") or value.get("type", "")).strip().lower()
    next_in_rich_block = in_rich_block or tag in {
        "plain_text",
        "lark_md",
        "markdown",
        "note",
        "div",
        "column_set",
        "column",
        "action",
        "button",
        "select_static",
        "date_picker",
    }

    segments: list[str] = []
    for key in _SUPPORTED_CARD_TEXT_KEYS:
        item = value.get(key)
        if isinstance(item, str) and next_in_rich_block:
            normalized = _normalize_feishu_text(item)
            if normalized:
                segments.append(normalized)

    for key, item in value.items():
        if key in _SKIP_TEXT_KEYS:
            continue
        segments.extend(_collect_text_segments(item, in_rich_block=next_in_rich_block))
    return segments


def _build_media_ref_from_payload(payload: dict[str, Any], *, resource_type: str) -> FeishuPostMediaRef:
    file_key = str(payload.get("file_key", "") or "").strip()
    file_name = _first_non_empty_text(
        payload.get("file_name"),
        payload.get("title"),
        payload.get("text"),
    )
    effective_type = resource_type if resource_type in {"audio", "video"} else "file"
    return FeishuPostMediaRef(file_key=file_key, file_name=file_name, resource_type=effective_type)


def _attachment_placeholder(file_name: str) -> str:
    normalized_name = _normalize_feishu_text(file_name)
    return f"[Attachment: {normalized_name}]" if normalized_name else FALLBACK_ATTACHMENT_TEXT


def _find_header_title(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    header = payload.get("header")
    if not isinstance(header, dict):
        return ""
    title = header.get("title")
    if isinstance(title, dict):
        return _first_non_empty_text(title.get("content"), title.get("text"), title.get("name"))
    return _normalize_feishu_text(str(title or ""))


def _find_first_text(payload: Any, *, keys: tuple[str, ...]) -> str:
    for node in _walk_nodes(payload):
        if not isinstance(node, dict):
            continue
        for key in keys:
            value = node.get(key)
            if isinstance(value, str):
                normalized = _normalize_feishu_text(value)
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


def _first_non_empty_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str):
            normalized = _normalize_feishu_text(value)
            if normalized:
                return normalized
        elif value is not None and not isinstance(value, (dict, list)):
            normalized = _normalize_feishu_text(str(value))
            if normalized:
                return normalized
    return ""


def _normalize_feishu_text(text: str) -> str:
    cleaned = _MENTION_PLACEHOLDER_RE.sub(" ", text or "")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = "\n".join(_WHITESPACE_RE.sub(" ", line).strip() for line in cleaned.split("\n"))
    cleaned = "\n".join(line for line in cleaned.split("\n") if line)
    cleaned = _MULTISPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def _unique_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for line in lines:
        if not line or line in seen:
            continue
        seen.add(line)
        unique.append(line)
    return unique


__all__ = [
    "FALLBACK_ATTACHMENT_TEXT",
    "FALLBACK_FORWARD_TEXT",
    "FALLBACK_IMAGE_TEXT",
    "FALLBACK_INTERACTIVE_TEXT",
    "FALLBACK_POST_TEXT",
    "FALLBACK_SHARE_CHAT_TEXT",
    "FeishuNormalizedMessage",
    "FeishuPostMediaRef",
    "FeishuPostParseResult",
    "normalize_feishu_message",
    "parse_feishu_post_payload",
]
