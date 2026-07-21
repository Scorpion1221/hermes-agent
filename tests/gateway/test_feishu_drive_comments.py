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
    resolve_drive_comment_event_turn,
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
    from plugins.platforms.feishu.adapter import FeishuAdapter

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
    from plugins.platforms.feishu.adapter import FeishuAdapter

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
    with patch('plugins.platforms.feishu.adapter.resolve_drive_comment_event_turn', AsyncMock(return_value=turn)):
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


def _make_reply_item(reply_id, open_id, text):
    return SimpleNamespace(
        reply_id=reply_id,
        user_id=SimpleNamespace(open_id=open_id),
        content=SimpleNamespace(elements=[SimpleNamespace(text_run=SimpleNamespace(text=text))]),
    )


@pytest.mark.asyncio
async def test_resolve_drive_comment_reply_chain_uses_resolved_names():
    """Reply chain context should use resolved names, not raw open_ids."""
    replies = [
        _make_reply_item('r1', 'ou_alice', 'first comment'),
        _make_reply_item('r2', 'ou_bob', 'second comment'),
        _make_reply_item('r3', 'ou_alice', 'the trigger reply'),
    ]
    comment_obj = SimpleNamespace(
        quote='some quoted text',
        reply_list=SimpleNamespace(replies=[
            SimpleNamespace(content=SimpleNamespace(elements=[
                SimpleNamespace(text_run=SimpleNamespace(text='root comment'))
            ]))
        ]),
    )
    mock_comment_resp = SimpleNamespace(
        success=lambda: True,
        data=SimpleNamespace(comment=comment_obj),
    )
    mock_reply_resp = SimpleNamespace(
        success=lambda: True,
        data=SimpleNamespace(items=replies),
    )

    client = SimpleNamespace(
        drive=SimpleNamespace(v1=SimpleNamespace(
            file_comment=SimpleNamespace(get=lambda req: mock_comment_resp),
            file_comment_reply=SimpleNamespace(list=lambda req: mock_reply_resp),
        )),
    )
    adapter = SimpleNamespace(_client=client)

    prefetched_ids = []
    async def mock_prefetch(ids):
        prefetched_ids.extend(ids)

    name_map = {'ou_alice': 'Alice', 'ou_bob': 'Bob'}
    def mock_sync_resolve(sender_id):
        return name_map.get(sender_id)

    event = FeishuDriveCommentEvent(
        file_token='ft_123', file_type='docx', comment_id='c_1', reply_id='r3',
        user_id=SimpleNamespace(open_id='ou_alice'),
    )
    turn = await resolve_drive_comment_event_turn(
        adapter=adapter,
        event=event,
        prefetch_sender_names=mock_prefetch,
        resolve_sender_name_sync=mock_sync_resolve,
    )
    assert turn is not None
    assert 'ou_alice' not in turn.reply_chain_context
    assert 'ou_bob' not in turn.reply_chain_context
    assert '[Alice]: first comment' in turn.reply_chain_context
    assert '[Bob]: second comment' in turn.reply_chain_context
    assert set(prefetched_ids) == {'ou_alice', 'ou_bob'}


@pytest.mark.asyncio
async def test_resolve_drive_comment_reply_chain_fallback_labels_without_resolver():
    """Without resolvers, opaque IDs should become Participant N labels."""
    replies = [
        _make_reply_item('r1', 'ou_xxx', 'hello'),
        _make_reply_item('r2', 'ou_yyy', 'world'),
        _make_reply_item('r3', 'ou_xxx', 'trigger'),
    ]
    comment_obj = SimpleNamespace(
        quote='',
        reply_list=SimpleNamespace(replies=[
            SimpleNamespace(content=SimpleNamespace(elements=[
                SimpleNamespace(text_run=SimpleNamespace(text='root'))
            ]))
        ]),
    )
    mock_comment_resp = SimpleNamespace(
        success=lambda: True,
        data=SimpleNamespace(comment=comment_obj),
    )
    mock_reply_resp = SimpleNamespace(
        success=lambda: True,
        data=SimpleNamespace(items=replies),
    )
    client = SimpleNamespace(
        drive=SimpleNamespace(v1=SimpleNamespace(
            file_comment=SimpleNamespace(get=lambda req: mock_comment_resp),
            file_comment_reply=SimpleNamespace(list=lambda req: mock_reply_resp),
        )),
    )
    adapter = SimpleNamespace(_client=client)

    event = FeishuDriveCommentEvent(
        file_token='ft_123', file_type='docx', comment_id='c_1', reply_id='r3',
    )
    turn = await resolve_drive_comment_event_turn(adapter=adapter, event=event)
    assert turn is not None
    assert 'ou_xxx' not in turn.reply_chain_context
    assert 'ou_yyy' not in turn.reply_chain_context
    assert '[Participant 1]: hello' in turn.reply_chain_context
    assert '[Participant 2]: world' in turn.reply_chain_context
