"""Feishu message time and CardKit handoff time use one explicit display zone."""

import asyncio
import os
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gateway.platforms.base import MessageType
from gateway.platforms.feishu_inbound import bridge
from gateway import stream_consumer


SENT_AT = datetime(2026, 9, 30, 15, 2, 20, 123000, tzinfo=timezone.utc)
HANDOFF_AT = SENT_AT + timedelta(seconds=100)


class FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return HANDOFF_AT.astimezone().replace(tzinfo=None)
        return HANDOFF_AT.astimezone(tz)


@pytest.fixture(params=["UTC", "America/Los_Angeles", "Asia/Tokyo"])
def server_timezone(request, monkeypatch):
    if not hasattr(time, "tzset"):
        pytest.skip("Changing the server time zone requires tzset")
    previous = os.environ.get("TZ")
    monkeypatch.setenv("TZ", request.param)
    time.tzset()
    monkeypatch.setattr(stream_consumer, "datetime", FixedDatetime)
    monkeypatch.setattr(bridge, "datetime", FixedDatetime)
    yield request.param
    if previous is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = previous
    time.tzset()


def inbound_event(create_time=None, *, timestamp=None):
    return bridge.build_feishu_message_event(
        data={},
        message=SimpleNamespace(message_id="followup", create_time=create_time),
        source=SimpleNamespace(thread_id=None),
        inbound_content=bridge.FeishuInboundContentBridge(
            text="Use this attachment", message_type=MessageType.DOCUMENT,
            media_urls=("/tmp/folder-manifest.json",), media_types=("application/json",),
        ),
        timestamp=timestamp,
    )


class CardTransport:
    MAX_MESSAGE_LENGTH = 30000

    def __init__(self):
        self.sent = []
        self.edits = []
        self.finalized = []
        self.first_send = asyncio.Event()

    async def send(self, **kwargs):
        self.sent.append(kwargs)
        self.first_send.set()
        return SimpleNamespace(success=True, message_id=f"card-{len(self.sent)}")

    async def edit_message(self, **kwargs):
        self.edits.append(kwargs)
        return SimpleNamespace(success=True, message_id=kwargs["message_id"])

    async def finalize_streaming_message(self, message_id, content, *, status="", stopped=False):
        self.finalized.append((message_id, content, status))
        return True

    @staticmethod
    def truncate_message(content, limit, **kwargs):
        return [content]


@pytest.mark.asyncio
@pytest.mark.parametrize("timestamp_type", [str, int])
async def test_delayed_message_keeps_feishu_time_and_handoff_uses_utc8(server_timezone, timestamp_type):
    # Downloading an attachment delays dispatch, but may not rewrite its send time.
    event = inbound_event(
        timestamp_type(int(SENT_AT.timestamp() * 1000)),
        timestamp=FixedDatetime.now(),
    )
    assert event.timestamp == SENT_AT
    assert event.timestamp.tzinfo is not None

    adapter = CardTransport()
    consumer = stream_consumer.GatewayStreamConsumer(
        adapter, "chat", stream_consumer.StreamConsumerConfig(cursor=""),
        metadata={"streaming": True}, initial_reply_to_id="original",
    )
    consumer.on_delta("OLD")
    task = asyncio.create_task(consumer.run())
    try:
        await asyncio.wait_for(adapter.first_send.wait(), 3)
        receipt = consumer.register_followup(
            event.text, event.message_id, {}, lambda: False, received_at=event.timestamp,
        )
        await consumer.acknowledge_followup(receipt)
        consumer.on_user_input_boundary(text=event.text)
        consumer.on_delta("NEW")
        consumer.finish("NEW")
        await asyncio.wait_for(task, 5)
        assert receipt.received_at == "23:02:20"
        assert "已收到补充 · 23:02:20" in adapter.sent[1]["content"]
        assert adapter.finalized[0][2] == "已转向后续消息 · 23:04:00"
        assert any(
            "收到 23:02:20 / 接入 23:04:00" in edit["content"]
            for edit in adapter.edits
        )
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("create_time", [None, "", "not-a-time", "0", "-1", "1e999", "9" * 30])
@pytest.mark.parametrize("fallback", ["aware", "naive", "none"])
def test_missing_or_invalid_message_time_uses_an_aware_fallback(server_timezone, create_time, fallback):
    supplied = {
        "aware": HANDOFF_AT,
        "naive": FixedDatetime.now(),
        "none": None,
    }[fallback]
    event = inbound_event(create_time, timestamp=supplied)
    assert event.timestamp == HANDOFF_AT
    assert event.timestamp.tzinfo is not None


def test_receipt_without_platform_time_still_uses_utc8(server_timezone):
    consumer = stream_consumer.GatewayStreamConsumer(
        CardTransport(), "chat", metadata={"streaming": True},
    )
    receipt = consumer.register_followup("update", "user-message", {}, lambda: False)
    assert receipt.received_at == "23:04:00"
