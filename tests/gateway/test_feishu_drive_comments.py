from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.platforms.base import MessageType
from gateway.platforms.feishu_inbound.comment_context import (
    FeishuDriveCommentEvent,
    FeishuDriveCommentTurn,
    build_drive_comment_prompt,
    parse_feishu_drive_comment_notice_event_payload,
)
from gateway.platforms.feishu_inbound.comment_target import (
    build_feishu_comment_target,
    is_feishu_comment_target,
    parse_feishu_comment_target,
)


def test_comment_target_round_trip_and_detection():
    target = build_feishu_comment_target(file_type='docx', file_token='file_tok', comment_id='c_123')
    assert target == 'comment:docx:file_tok:c_123'
    parsed = parse_feishu_comment_target(target)
    assert parsed is not None
    assert parsed.delivery_mode == 'reply'
    assert parsed.file_type == 'docx'
    assert parsed.file_token == 'file_tok'
    assert parsed.comment_id == 'c_123'
    assert is_feishu_comment_target(target) is True
    assert parse_feishu_comment_target('oc_xxx') is None


def test_parse_drive_comment_event_prefers_notice_meta_shape():
    payload = {
        'event': {
            'comment_id': 'comment_1',
            'reply_id': 'reply_1',
            'notice_meta': {
                'file_token': 'file_tok',
                'file_type': 'docx',
                'from_user_id': {'open_id': 'ou_sender', 'user_id': 'u_sender', 'union_id': 'on_sender'},
                'timestamp': '1712000000000',
                'is_mentioned': True,
            },
        }
    }
    parsed = parse_feishu_drive_comment_notice_event_payload(payload)
    assert parsed == FeishuDriveCommentEvent(
        file_token='file_tok',
        file_type='docx',
        comment_id='comment_1',
        reply_id='reply_1',
        user_id={'open_id': 'ou_sender', 'user_id': 'u_sender', 'union_id': 'on_sender'},
        action_time='1712000000000',
        is_mention=True,
        notice_meta=payload['event']['notice_meta'],
    )


def test_build_drive_comment_prompt_includes_quote_chain_and_ids():
    prompt = build_drive_comment_prompt(
        quoted_text='quoted text',
        comment_text='please update this',
        reply_chain_context='[ou_a]: prior',
        file_type='docx',
        file_token='file_tok',
        comment_id='comment_1',
        reply_id='reply_1',
    )
    assert 'Feishu Drive comment-thread event' in prompt
    assert 'User comment: please update this' in prompt
    assert 'Quoted content: quoted text' in prompt
    assert 'Reply chain context:' in prompt
    assert '[ou_a]: prior' in prompt
    assert 'file_token: file_tok' in prompt
    assert 'comment_id: comment_1' in prompt
    assert 'reply_id: reply_1' in prompt


@pytest.mark.asyncio
async def test_feishu_send_routes_comment_target_through_drive_api(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.feishu import FeishuAdapter

    adapter = FeishuAdapter(PlatformConfig())
    calls = []

    class _Client:
        def request(self, req):
            calls.append(req)
            return SimpleNamespace(code=0, msg='ok')

    adapter._client = _Client()
    result = await adapter.send('comment:docx:file_tok:comment_1', 'hello comment')
    assert result.success is True
    assert result.message_id == 'comment-reply:comment_1'
    assert calls[0]['url'].endswith('/files/file_tok/comments/comment_1/replies')


@pytest.mark.asyncio
async def test_drive_comment_webhook_dispatches_synthetic_message(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.feishu import FeishuAdapter

    adapter = FeishuAdapter(PlatformConfig())
    adapter._resolve_sender_profile = AsyncMock(return_value={'user_id': 'ou_sender', 'user_name': 'Alice', 'user_id_alt': 'on_sender'})
    adapter._handle_message_with_guards = AsyncMock()
    turn = FeishuDriveCommentTurn(
        prompt='This is a Feishu Drive comment-thread event.\nUser comment: please update this',
        comment_target='comment:docx:file_tok:comment_1',
        quoted_text='quoted text',
        comment_text='please update this',
        reply_chain_context='',
        is_whole_comment=False,
    )
    with patch('gateway.platforms.feishu.resolve_drive_comment_event_turn', AsyncMock(return_value=turn)):
        payload = SimpleNamespace(
            event=SimpleNamespace(
                comment_id='comment_1',
                reply_id='reply_1',
                notice_meta=SimpleNamespace(
                    file_token='file_tok',
                    file_type='docx',
                    from_user_id=SimpleNamespace(open_id='ou_sender', user_id='u_sender', union_id='on_sender'),
                    timestamp='1712000000000',
                    is_mentioned=True,
                ),
            )
        )
        await adapter._handle_drive_comment_event_data(payload)

    adapter._handle_message_with_guards.assert_awaited_once()
    event = adapter._handle_message_with_guards.await_args.args[0]
    assert event.message_type == MessageType.TEXT
    assert event.source.chat_id == 'comment:docx:file_tok:comment_1'
    assert event.source.user_name == 'Alice'
    assert 'please update this' in event.text
