"""A Feishu event carries the time its message was sent, not when Hermes finished reading it.

Attachments download before the event is built, so a file follow-up used to be stamped seconds (or
minutes) late — and the transcript, the message-timestamp render and the follow-up receipts all read
that stamp.
"""

import asyncio
import json
import time
from datetime import timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.feishu.feishu_quote import QuotedReply


def _adapter(download_seconds: float = 0.0):
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = FeishuAdapter.__new__(FeishuAdapter)
    adapter.config = PlatformConfig(extra={})
    adapter._bot_open_id, adapter._bot_user_id, adapter._bot_name = "ou_bot", "", "Hermes"

    async def _download(*args, **kwargs):
        await asyncio.sleep(download_seconds)
        return [], []

    adapter._download_feishu_message_resources = _download
    adapter._fetch_quoted_reply = AsyncMock(return_value=QuotedReply())
    adapter.get_chat_info = AsyncMock(return_value={"name": "Test Chat"})
    adapter._resolve_sender_profile = AsyncMock(return_value={"user_id": "u1", "user_name": "Alice", "user_id_alt": None})
    adapter._resolve_source_chat_type = Mock(return_value="dm")
    adapter.build_source = Mock(return_value=SimpleNamespace(thread_id=None))
    adapter._dispatch_inbound_event = AsyncMock()
    return adapter


def _receive(adapter, **message_fields):
    message = SimpleNamespace(
        content=json.dumps({"text": "here is the report"}), message_type="text", message_id="om_1",
        mentions=[], chat_id="oc_dm", parent_id=None, upper_message_id=None, thread_id=None, **message_fields)
    asyncio.run(adapter._process_inbound_message(
        data=message, message=message, sender_id=None, chat_type="p2p", message_id="om_1"))
    return adapter._dispatch_inbound_event.call_args.args[0]


def test_a_slow_attachment_keeps_the_time_its_message_was_sent():
    sent_ms = int(time.time() * 1000) - 90_000  # sent a minute and a half before Hermes finished reading it
    event = _receive(_adapter(download_seconds=0.05), create_time=str(sent_ms))

    assert event.timestamp.tzinfo is not None
    assert int(event.timestamp.timestamp() * 1000) == sent_ms


@pytest.mark.parametrize("create_time", [None, "", "not-a-number", "0"])
def test_without_a_usable_send_time_the_event_is_stamped_now_in_utc(create_time):
    before = time.time()
    fields = {} if create_time is None else {"create_time": create_time}
    event = _receive(_adapter(), **fields)

    assert event.timestamp.tzinfo is timezone.utc
    assert before - 1 <= event.timestamp.timestamp() <= time.time() + 1


def test_the_send_time_is_what_the_transcript_records():
    from gateway.message_timestamps import coerce_message_timestamp

    sent_ms = 1_760_000_000_000
    event = _receive(_adapter(), create_time=str(sent_ms))

    assert coerce_message_timestamp(event.timestamp) == sent_ms / 1000
