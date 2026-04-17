from __future__ import annotations

from collections import defaultdict
from typing import Any, Awaitable, Callable, Iterable, Optional

from .types import FeishuMessageContext, FeishuQuotedContext, FeishuResourceDescriptor


def _obj_get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _obj_path(value: Any, *keys: str) -> Any:
    cur = value
    for key in keys:
        if cur is None:
            return None
        cur = _obj_get(cur, key)
    return cur


def _normalize_sender_name(sender_name: str | None, content: str) -> str:
    sender = str(sender_name or "").strip()
    if not sender:
        return content.strip()
    body = content.strip()
    if not body:
        return sender
    return f"{sender}: {body}"


def extract_message_items(response: Any) -> list[Any]:
    data = _obj_get(response, "data")
    items = _obj_get(data, "items") if data is not None else None
    if isinstance(items, list):
        return items
    if items is None:
        return []
    return list(items)


def build_resource_descriptors(normalized: Any) -> tuple[FeishuResourceDescriptor, ...]:
    out: list[FeishuResourceDescriptor] = []
    for image_key in list(getattr(normalized, "image_keys", []) or []):
        if image_key:
            out.append(FeishuResourceDescriptor(type="image", file_key=str(image_key).strip()))
    for media_ref in list(getattr(normalized, "media_refs", []) or []):
        file_key = str(getattr(media_ref, "file_key", "") or "").strip()
        if not file_key:
            continue
        resource_type = str(getattr(media_ref, "resource_type", "file") or "file").strip().lower()
        mapped_type = resource_type if resource_type in {"image", "file", "audio", "video", "sticker"} else "file"
        out.append(
            FeishuResourceDescriptor(
                type=mapped_type,  # type: ignore[arg-type]
                file_key=file_key,
                file_name=str(getattr(media_ref, "file_name", "") or "").strip(),
            )
        )
    return tuple(out)


def display_text_from_normalized(normalized: Any, effective_type: str) -> str:
    normalized_text = str(getattr(normalized, "text_content", "") or "").strip()
    if normalized_text:
        return normalized_text
    metadata = dict(getattr(normalized, "metadata", {}) or {})
    placeholder = str(metadata.get("placeholder_text", "") or "").strip()
    if placeholder:
        return placeholder
    return {
        "image": "[Image]",
        "post": "[Rich text message]",
        "merge_forward": "[Merged forward message]",
        "share_chat": "[Shared chat]",
        "interactive": "[Interactive message]",
        "card": "[Interactive message]",
        "file": "[Attachment]",
        "audio": "[Attachment]",
        "media": "[Attachment]",
    }.get(effective_type, "")


def _select_primary_item(items: list[Any], message_id: str) -> Any:
    for item in items:
        if str(_obj_get(item, "message_id", "") or "") == message_id:
            return item
    return items[0] if items else None


def _format_merge_forward_items(
    *,
    items: list[Any],
    root_message_id: str,
    normalize_message: Callable[..., Any],
    resolve_sender_name_sync: Optional[Callable[[str], Optional[str]]] = None,
) -> tuple[str, tuple[FeishuResourceDescriptor, ...]]:
    if not items:
        return "", ()

    children_map: dict[str, list[Any]] = defaultdict(list)
    for item in items:
        item_id = str(_obj_get(item, "message_id", "") or "")
        upper_id = str(_obj_get(item, "upper_message_id", "") or "")
        if item_id == root_message_id and not upper_id:
            continue
        parent_id = upper_id or root_message_id
        children_map[parent_id].append(item)

    for child_items in children_map.values():
        child_items.sort(key=lambda item: str(_obj_get(item, "create_time", "") or ""))

    def _render(parent_id: str, depth: int = 0) -> tuple[list[str], list[FeishuResourceDescriptor]]:
        rendered: list[str] = []
        resources: list[FeishuResourceDescriptor] = []
        for item in children_map.get(parent_id, []):
            item_id = str(_obj_get(item, "message_id", "") or "")
            raw_type = str(_obj_get(item, "msg_type", "") or "text")
            raw_content = str(_obj_path(item, "body", "content") or "")
            normalized = normalize_message(message_type=raw_type, raw_content=raw_content)
            sender_id = str(_obj_path(item, "sender", "id") or "").strip()
            sender_name = resolve_sender_name_sync(sender_id) if resolve_sender_name_sync and sender_id else None
            sender_name = sender_name or sender_id or str(_obj_get(item, "sender_name", "") or "").strip()
            content = display_text_from_normalized(normalized, raw_type).strip()
            resources.extend(build_resource_descriptors(normalized))
            if raw_type == "merge_forward" and item_id:
                nested_lines, nested_resources = _render(item_id, depth + 1)
                nested = "\n".join(nested_lines).strip()
                resources.extend(nested_resources)
                content = nested or content
            if not content:
                continue
            prefix = "  " * depth + "- "
            rendered.append(prefix + _normalize_sender_name(sender_name, content))
        return rendered, resources

    lines, resources = _render(root_message_id)
    return "\n".join(lines).strip(), tuple(resources)


def build_feishu_message_context(
    *,
    message_id: str,
    message_type: str,
    raw_content: str,
    normalize_message: Callable[..., Any],
    response_items: Optional[list[Any]] = None,
    chat_id: str = "",
    chat_type: str = "",
    root_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    sender_id: str = "",
    sender_name: str = "",
    resolve_sender_name_sync: Optional[Callable[[str], Optional[str]]] = None,
) -> FeishuMessageContext:
    effective_type = str(message_type or "").strip().lower()
    effective_content = raw_content or ""
    primary_item = _select_primary_item(response_items or [], message_id)
    if primary_item is not None:
        primary_type = str(_obj_get(primary_item, "msg_type", "") or effective_type).strip().lower()
        primary_content = str(_obj_path(primary_item, "body", "content") or "")
        if primary_type:
            effective_type = primary_type
        if primary_content:
            effective_content = primary_content
        sender_id = sender_id or str(_obj_path(primary_item, "sender", "id") or "")

    if effective_type == "merge_forward" and response_items and len(response_items) > 1:
        expanded, expanded_resources = _format_merge_forward_items(
            items=response_items,
            root_message_id=message_id,
            normalize_message=normalize_message,
            resolve_sender_name_sync=resolve_sender_name_sync,
        )
        if expanded:
            return FeishuMessageContext(
                message_id=message_id,
                content=expanded,
                content_type=effective_type,
                preferred_message_type="text",
                relation_kind="merge_forward",
                raw_content=effective_content,
                chat_id=chat_id,
                chat_type=chat_type,
                root_id=root_id,
                parent_id=parent_id,
                thread_id=thread_id,
                sender_id=sender_id,
                sender_name=sender_name,
                resource_descriptors=expanded_resources,
                mentioned_ids=(),
                metadata={"expanded_from_items": True},
            )

    normalized = normalize_message(message_type=effective_type, raw_content=effective_content)
    normalized_text = display_text_from_normalized(normalized, effective_type)
    return FeishuMessageContext(
        message_id=message_id,
        content=normalized_text,
        content_type=effective_type,
        preferred_message_type=str(getattr(normalized, "preferred_message_type", "text") or "text"),
        relation_kind=str(getattr(normalized, "relation_kind", "plain") or "plain"),
        raw_content=effective_content,
        chat_id=chat_id,
        chat_type=chat_type,
        root_id=root_id,
        parent_id=parent_id,
        thread_id=thread_id,
        sender_id=sender_id,
        sender_name=sender_name,
        resource_descriptors=build_resource_descriptors(normalized),
        mentioned_ids=tuple(getattr(normalized, "mentioned_ids", []) or []),
        metadata=dict(getattr(normalized, "metadata", {}) or {}),
    )


async def build_feishu_quoted_context(
    *,
    message_id: str,
    response_items: list[Any],
    normalize_message: Callable[..., Any],
    download_resources: Callable[[str, Iterable[FeishuResourceDescriptor]], Awaitable[tuple[list[str], list[str]]]],
    resolve_sender_name: Optional[Callable[[str], Awaitable[Optional[str]]]] = None,
    resolve_sender_name_sync: Optional[Callable[[str], Optional[str]]] = None,
) -> FeishuQuotedContext:
    primary_item = _select_primary_item(response_items, message_id)
    if primary_item is None:
        return FeishuQuotedContext(message_id=message_id, kind="plain", stable_ref=f"feishu:{message_id}")

    sender_id = str(_obj_path(primary_item, "sender", "id") or "").strip()
    sender_name = ""
    if resolve_sender_name and sender_id:
        sender_name = str(await resolve_sender_name(sender_id) or "").strip()

    ctx = build_feishu_message_context(
        message_id=message_id,
        message_type=str(_obj_get(primary_item, "msg_type", "") or "text"),
        raw_content=str(_obj_path(primary_item, "body", "content") or ""),
        normalize_message=normalize_message,
        response_items=response_items,
        sender_id=sender_id,
        sender_name=sender_name,
        resolve_sender_name_sync=resolve_sender_name_sync,
    )

    media_urls, media_types = await download_resources(message_id, ctx.resource_descriptors)
    display_text = _normalize_sender_name(sender_name, ctx.content or "").strip()
    return FeishuQuotedContext(
        message_id=message_id,
        kind=ctx.relation_kind or ctx.content_type or "plain",
        text=ctx.content,
        summary=display_text or ctx.content,
        sender_name=sender_name,
        media_urls=tuple(media_urls),
        media_types=tuple(media_types),
        stable_ref=f"feishu:{message_id}",
        metadata=dict(ctx.metadata or {}),
    )
