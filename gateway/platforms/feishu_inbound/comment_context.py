from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional

from .comment_target import build_feishu_comment_target
from .lookup import _fallback_sender_label
from .user_name_cache import is_probably_feishu_opaque_user_id


def _obj_get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


@dataclass(frozen=True)
class FeishuDriveCommentEvent:
    file_token: str
    file_type: str
    comment_id: str
    reply_id: Optional[str] = None
    user_id: Any = None
    action_time: Optional[str] = None
    is_mention: Optional[bool] = None
    notice_meta: Any = None


@dataclass(frozen=True)
class FeishuDriveCommentTurn:
    prompt: str
    comment_target: str
    quoted_text: str = ''
    comment_text: str = ''
    reply_chain_context: str = ''
    is_whole_comment: bool = False


def parse_feishu_drive_comment_notice_event_payload(data: Any) -> Optional[FeishuDriveCommentEvent]:
    raw = _obj_get(data, 'event', data)
    if raw is None:
        return None
    notice_meta = _obj_get(raw, 'notice_meta', None) or _obj_get(data, 'notice_meta', None)
    file_token = _obj_get(notice_meta, 'file_token', None) or _obj_get(raw, 'file_token', None) or _obj_get(data, 'file_token', None)
    file_type = _obj_get(notice_meta, 'file_type', None) or _obj_get(raw, 'file_type', None) or _obj_get(data, 'file_type', None)
    comment_id = _obj_get(raw, 'comment_id', None) or _obj_get(data, 'comment_id', None)
    reply_id = _obj_get(raw, 'reply_id', None) or _obj_get(data, 'reply_id', None)
    user_id = _obj_get(notice_meta, 'from_user_id', None) or _obj_get(raw, 'user_id', None) or _obj_get(data, 'user_id', None)
    action_time = _obj_get(notice_meta, 'timestamp', None) or _obj_get(raw, 'action_time', None) or _obj_get(data, 'action_time', None)
    is_mention = _obj_get(notice_meta, 'is_mentioned', None)
    if is_mention is None:
        is_mention = _obj_get(raw, 'is_mention', None)
    if not file_token or not file_type or not comment_id:
        return None
    return FeishuDriveCommentEvent(
        file_token=str(file_token),
        file_type=str(file_type),
        comment_id=str(comment_id),
        reply_id=str(reply_id) if reply_id else None,
        user_id=user_id,
        action_time=str(action_time) if action_time else None,
        is_mention=bool(is_mention) if is_mention is not None else None,
        notice_meta=notice_meta,
    )


def _extract_reply_text(reply: Any) -> str:
    content = _obj_get(reply, 'content', None)
    elements = _obj_get(content, 'elements', None) or _obj_get(reply, 'reply_elements', None) or []
    parts = []
    for element in elements:
        text_run = _obj_get(element, 'text_run', None)
        text = _obj_get(text_run, 'text', None)
        if text:
            parts.append(str(text))
    return ''.join(parts).strip()


def build_drive_comment_prompt(*, quoted_text: str = '', comment_text: str = '', reply_chain_context: str = '', file_type: str = '', file_token: str = '', comment_id: str = '', reply_id: str | None = None) -> str:
    lines = [
        'This is a Feishu Drive comment-thread event, not a normal IM conversation.',
    ]
    if comment_text:
        lines.append(f'User comment: {comment_text}')
    if quoted_text:
        lines.append(f'Quoted content: {quoted_text}')
    if reply_chain_context:
        lines.append('Reply chain context:')
        lines.append(reply_chain_context)
    lines.append(f'file_type: {file_type}')
    lines.append(f'file_token: {file_token}')
    lines.append(f'comment_id: {comment_id}')
    if reply_id:
        lines.append(f'reply_id: {reply_id}')
    lines.append('Reply in the same language as the user.')
    return '\n'.join(lines)


async def resolve_drive_comment_event_turn(
    *,
    adapter: Any,
    event: FeishuDriveCommentEvent,
    prefetch_sender_names: Optional[Callable[[Sequence[str]], Awaitable[Any]]] = None,
    resolve_sender_name_sync: Optional[Callable[[str], Optional[str]]] = None,
) -> Optional[FeishuDriveCommentTurn]:
    client = getattr(adapter, '_client', None)
    if client is None:
        return None
    try:
        from lark_oapi.api.drive.v1.model.get_file_comment_request import GetFileCommentRequest
        from lark_oapi.api.drive.v1.model.list_file_comment_reply_request import ListFileCommentReplyRequest
    except Exception:
        GetFileCommentRequest = None
        ListFileCommentReplyRequest = None

    if GetFileCommentRequest is not None:
        comment_request = (
            GetFileCommentRequest.builder()
            .file_token(event.file_token)
            .comment_id(event.comment_id)
            .file_type(event.file_type)
            .user_id_type('open_id')
            .build()
        )
    else:
        comment_request = SimpleNamespace(file_token=event.file_token, comment_id=event.comment_id, file_type=event.file_type, user_id_type='open_id')
    comment_response = await asyncio.to_thread(client.drive.v1.file_comment.get, comment_request)
    if not comment_response or not comment_response.success():
        return None
    comment = _obj_get(_obj_get(comment_response, 'data', None), 'comment', None) or _obj_get(_obj_get(comment_response, 'data', None), 'file_comment', None) or _obj_get(comment_response, 'data', None)
    quoted_text = str(_obj_get(comment, 'quote', '') or '').strip()
    root_reply_list = _obj_get(comment, 'reply_list', None)
    root_replies = _obj_get(root_reply_list, 'replies', None) or []
    comment_text = _extract_reply_text(root_replies[0]) if root_replies else ''
    reply_chain_context = ''
    if event.reply_id:
        if ListFileCommentReplyRequest is not None:
            reply_request = (
                ListFileCommentReplyRequest.builder()
                .file_token(event.file_token)
                .comment_id(event.comment_id)
                .file_type(event.file_type)
                .user_id_type('open_id')
                .page_size(100)
                .build()
            )
        else:
            reply_request = SimpleNamespace(file_token=event.file_token, comment_id=event.comment_id, file_type=event.file_type, user_id_type='open_id', page_size=100)
        reply_response = await asyncio.to_thread(client.drive.v1.file_comment_reply.list, reply_request)
        reply_items = _obj_get(_obj_get(reply_response, 'data', None), 'items', None) or []

        all_sender_ids: list[str] = []
        seen_ids: set[str] = set()
        for reply in reply_items:
            sid = str(_obj_get(_obj_get(reply, 'user_id', None), 'open_id', None) or '').strip()
            if sid and sid not in seen_ids:
                seen_ids.add(sid)
                all_sender_ids.append(sid)
        if all_sender_ids and prefetch_sender_names is not None:
            try:
                await prefetch_sender_names(all_sender_ids)
            except Exception:
                pass

        opaque_sender_labels: dict[str, str] = {}
        prior_lines = []
        matched_text = comment_text
        for reply in reply_items:
            rid = str(_obj_get(reply, 'reply_id', '') or '')
            text = _extract_reply_text(reply)
            sender_id = str(_obj_get(_obj_get(reply, 'user_id', None), 'open_id', None) or '').strip()
            sender_label = ''
            if sender_id and resolve_sender_name_sync is not None:
                sender_label = resolve_sender_name_sync(sender_id) or ''
            if not sender_label and sender_id:
                if is_probably_feishu_opaque_user_id(sender_id):
                    sender_label = _fallback_sender_label(
                        sender_id=sender_id,
                        opaque_sender_labels=opaque_sender_labels,
                    )
                else:
                    sender_label = sender_id
            if not sender_label:
                sender_label = _fallback_sender_label(
                    sender_id=sender_id or 'unknown',
                    opaque_sender_labels=opaque_sender_labels,
                )
            if rid == event.reply_id:
                matched_text = text or matched_text
                continue
            if text:
                prior_lines.append(f'[{sender_label}]: {text}')
        reply_chain_context = '\n'.join(prior_lines)
        comment_text = matched_text
    is_whole_comment = not bool(quoted_text)
    comment_target = build_feishu_comment_target(
        file_type=event.file_type, file_token=event.file_token, comment_id=event.comment_id,
        delivery_mode='create_whole' if is_whole_comment else 'reply',
    )
    prompt = build_drive_comment_prompt(
        quoted_text=quoted_text,
        comment_text=comment_text,
        reply_chain_context=reply_chain_context,
        file_type=event.file_type,
        file_token=event.file_token,
        comment_id=event.comment_id,
        reply_id=event.reply_id,
    )
    return FeishuDriveCommentTurn(
        prompt=prompt,
        comment_target=comment_target,
        quoted_text=quoted_text,
        comment_text=comment_text,
        reply_chain_context=reply_chain_context,
        is_whole_comment=is_whole_comment,
    )


__all__ = [
    'FeishuDriveCommentEvent',
    'FeishuDriveCommentTurn',
    'parse_feishu_drive_comment_notice_event_payload',
    'resolve_drive_comment_event_turn',
    'build_drive_comment_prompt',
]
