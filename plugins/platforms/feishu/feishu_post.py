"""Feishu rich-text (``post``) messages parsed into text plus the images and files they carry.

A post body is ``{"title", "content": [[element, ...], ...], "files": [...]}``, either direct (as GET
returns it) or wrapped per locale (``{"zh_cn": {...}}``, optionally under ``"post"``). ``content`` rows
hold inline elements (text, links, mentions, images, files). ``files`` is the attachment zone the composer
sends beside the text (``{"file_key", "file_name", "is_folder"}``);
folder entries there are skipped.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .adapter import FeishuMentionRef, FeishuPostMediaRef

_PREFERRED_LOCALES = ("zh_cn", "en_us")
_MARKDOWN_SPECIAL_CHARS_RE = re.compile(r"([\\`*_{}\[\]()#+\-!|>~])")
_STATIC_POST_TAGS = {"br": "\n", "hr": "\n\n---\n\n", "divider": "\n\n---\n\n"}
_TEXT_STYLE_WRAPPERS = (("bold", "**", "**"), ("italic", "*", "*"), ("underline", "<u>", "</u>"), ("strikethrough", "~~", "~~"))


@dataclass(frozen=True)
class FeishuPostParseResult:
    text_content: str
    image_keys: list[str] = field(default_factory=list)
    media_refs: list[FeishuPostMediaRef] = field(default_factory=list)


def parse_feishu_post_payload(
    payload: Any, *, mentions_map: Optional[dict[str, FeishuMentionRef]] = None,
) -> FeishuPostParseResult:
    from .adapter import FALLBACK_POST_TEXT, _normalize_feishu_text, _to_boolean

    resolved = _resolve_post_payload(payload)
    if not resolved:
        return FeishuPostParseResult(text_content=FALLBACK_POST_TEXT)
    image_keys: list[str] = []
    media_refs: list[FeishuPostMediaRef] = []
    parts: list[str] = []
    title = _normalize_feishu_text(str(resolved.get("title", "")).strip())
    if title:
        parts.append(title)
    for row in resolved.get("content", []) or []:
        if not isinstance(row, list):
            continue
        row_text = _normalize_feishu_text(
            "".join(_render_post_element(item, image_keys, media_refs, mentions_map) for item in row)
        )
        if row_text:
            parts.append(row_text)
    for entry in resolved.get("files", []) or []:
        if (
            not isinstance(entry, dict)
            or _to_boolean(entry.get("is_folder"))
            or not str(entry.get("file_key", "")).strip()
        ):
            continue
        placeholder = _render_post_element(
            {**entry, "tag": "file"}, image_keys, media_refs, mentions_map,
        )
        if placeholder:
            parts.append(placeholder)
    return FeishuPostParseResult(
        text_content="\n".join(parts).strip() or FALLBACK_POST_TEXT, image_keys=image_keys, media_refs=media_refs,
    )


def _resolve_post_payload(payload: Any) -> dict[str, Any]:
    direct = _to_post_payload(payload)
    if direct:
        return direct
    if not isinstance(payload, dict):
        return {}
    return _resolve_locale_payload(payload.get("post")) or _resolve_locale_payload(payload)


def _resolve_locale_payload(payload: Any) -> dict[str, Any]:
    direct = _to_post_payload(payload)
    if direct:
        return direct
    if not isinstance(payload, dict):
        return {}
    # Preferred locales first, then any locale that carries a content list.
    preferred = (payload.get(key) for key in _PREFERRED_LOCALES)
    for candidate in map(_to_post_payload, itertools.chain(preferred, payload.values())):
        if candidate:
            return candidate
    return {}


def _to_post_payload(candidate: Any) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        return {}
    content = candidate.get("content")
    if not isinstance(content, list):
        return {}
    files = candidate.get("files")
    return {
        "title": str(candidate.get("title", "") or ""),
        "content": content,
        "files": files if isinstance(files, list) else [],
    }


def _escape_markdown_text(text: str) -> str:
    return _MARKDOWN_SPECIAL_CHARS_RE.sub(r"\\\1", text)


def _is_style_enabled(style: dict[str, Any] | None, key: str) -> bool:
    from .adapter import _to_boolean

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
    # Post text elements carry raw text plus separate style flags; the style wrappers below
    # re-create the markdown. Escaping the text here put `\*\*bold\*\*` / `\`code\`` into the
    # model's context and those backslashes came straight back out in replies (#9816).
    rendered = text
    if not rendered:
        return ""
    for key, prefix, suffix in _TEXT_STYLE_WRAPPERS:  # order matters for nesting
        if _is_style_enabled(style_dict, key):
            rendered = f"{prefix}{rendered}{suffix}"
    return rendered


def _render_code_block_element(element: dict[str, Any]) -> str:
    language = _sanitize_fence_language(str(element.get("language", "") or "") or str(element.get("lang", "") or ""))
    code = (str(element.get("text", "") or "") or str(element.get("content", "") or "")).replace("\r\n", "\n")
    trailing_newline = "" if code.endswith("\n") else "\n"
    return f"```{language}\n{code}{trailing_newline}```"


def _render_post_element(
    element: Any, image_keys: list[str], media_refs: list[FeishuPostMediaRef],
    mentions_map: Optional[dict[str, FeishuMentionRef]] = None,
) -> str:
    from .adapter import FeishuMentionRef, FeishuPostMediaRef

    if isinstance(element, str):
        return element
    if not isinstance(element, dict):
        return ""

    tag = str(element.get("tag", "")).strip().lower()
    if tag in _STATIC_POST_TAGS:
        return _STATIC_POST_TAGS[tag]
    if tag == "text":
        return _render_text_element(element)
    if tag in {"code_block", "pre"}:
        return _render_code_block_element(element)
    if tag == "a":
        href = str(element.get("href", "")).strip()
        label = str(element.get("text", href) or "").strip()
        if not label:
            return ""
        escaped_label = _escape_markdown_text(label)
        return f"[{escaped_label}]({href})" if href else escaped_label
    if tag == "at":
        # <at>.user_id is a placeholder ("@_user_N" / "@_all"); mentions_map has the real ref.
        placeholder = str(element.get("user_id", "")).strip()
        if placeholder == "@_all":
            # The SDK sometimes omits @_all from top-level mentions; record it so callers see it.
            if mentions_map is not None and "@_all" not in mentions_map:
                mentions_map["@_all"] = FeishuMentionRef(is_all=True)
            return "@all"
        ref = (mentions_map or {}).get(placeholder)
        display_name = (ref.name or ref.open_id or "user") if ref is not None else (
            str(element.get("user_name", "")).strip() or "user"
        )
        return f"@{_escape_markdown_text(display_name)}"
    if tag in {"img", "image"}:
        image_key = str(element.get("image_key", "")).strip()
        if image_key and image_key not in image_keys:
            image_keys.append(image_key)
        alt = str(element.get("text", "")).strip() or str(element.get("alt", "")).strip()
        return f"[Image: {alt}]" if alt else "[Image]"
    if tag in {"media", "file", "audio", "video"}:
        file_key = str(element.get("file_key", "")).strip()
        names = (str(element.get(k, "")).strip() for k in ("file_name", "title", "text"))
        file_name = next((n for n in names if n), "")
        placeholder = f"[Attachment: {file_name}]" if file_name else "[Attachment]"
        if not file_key:
            return placeholder
        if not any(ref.file_key == file_key for ref in media_refs):
            media_refs.append(FeishuPostMediaRef(
                file_key=file_key, file_name=file_name, resource_type=tag if tag in {"audio", "video"} else "file",
            ))
            return placeholder
        return ""
    if tag in {"emotion", "emoji"}:
        label = str(element.get("text", "")).strip() or str(element.get("emoji_type", "")).strip()
        return f":{_escape_markdown_text(label)}:" if label else "[Emoji]"
    if tag == "code":
        code = str(element.get("text", "") or "") or str(element.get("content", "") or "")
        return _wrap_inline_code(code) if code else ""
    nested = (element.get(key) for key in ("text", "title", "content", "children", "elements"))
    return _join_nested_posts(nested, image_keys, media_refs, mentions_map)


def _join_nested_posts(values: Any, image_keys: Any, media_refs: Any, mentions_map: Any) -> str:
    parts = (_render_nested_post(item, image_keys, media_refs, mentions_map) for item in values)
    return " ".join(part for part in parts if part)


def _render_nested_post(
    value: Any, image_keys: list[str], media_refs: list[FeishuPostMediaRef],
    mentions_map: Optional[dict[str, FeishuMentionRef]] = None,
) -> str:
    if isinstance(value, str):
        return _escape_markdown_text(value)
    if isinstance(value, list):
        return _join_nested_posts(value, image_keys, media_refs, mentions_map)
    if isinstance(value, dict):
        direct = _render_post_element(value, image_keys, media_refs, mentions_map)
        return direct or _join_nested_posts(value.values(), image_keys, media_refs, mentions_map)
    return ""
