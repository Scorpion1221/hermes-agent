from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

CommentFileType = Literal['doc', 'docx', 'file', 'sheet', 'slides']
CommentDeliveryMode = Literal['reply', 'create_whole']

_VALID_FILE_TYPES = {'doc', 'docx', 'file', 'sheet', 'slides'}
_VALID_DELIVERY_MODES = {'reply', 'create_whole'}
_COMMENT_PREFIX = 'comment:'


@dataclass(frozen=True)
class FeishuCommentTarget:
    delivery_mode: CommentDeliveryMode
    file_type: CommentFileType
    file_token: str
    comment_id: str


def build_feishu_comment_target(
    *,
    file_type: CommentFileType,
    file_token: str,
    comment_id: str,
    delivery_mode: CommentDeliveryMode = 'reply',
) -> str:
    if delivery_mode == 'reply':
        return f'{_COMMENT_PREFIX}{file_type}:{file_token}:{comment_id}'
    return f'{_COMMENT_PREFIX}{delivery_mode}:{file_type}:{file_token}:{comment_id}'


def parse_feishu_comment_target(target: str) -> Optional[FeishuCommentTarget]:
    normalized = str(target or '').strip()
    if not normalized.startswith(_COMMENT_PREFIX):
        return None
    rest = normalized[len(_COMMENT_PREFIX):]
    parts = rest.split(':')
    delivery_mode: CommentDeliveryMode = 'reply'
    if len(parts) == 3:
        file_type, file_token, comment_id = parts
    elif len(parts) == 4 and parts[0] in _VALID_DELIVERY_MODES:
        delivery_mode = parts[0]  # type: ignore[assignment]
        file_type, file_token, comment_id = parts[1:]
    else:
        return None
    if file_type not in _VALID_FILE_TYPES or not file_token or not comment_id:
        return None
    return FeishuCommentTarget(
        delivery_mode=delivery_mode,
        file_type=file_type,  # type: ignore[arg-type]
        file_token=file_token,
        comment_id=comment_id,
    )


def is_feishu_comment_target(target: str) -> bool:
    return parse_feishu_comment_target(target) is not None


__all__ = [
    'CommentFileType',
    'CommentDeliveryMode',
    'FeishuCommentTarget',
    'build_feishu_comment_target',
    'parse_feishu_comment_target',
    'is_feishu_comment_target',
]
