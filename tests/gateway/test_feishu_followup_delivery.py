"""Lossless CardKit handoff contracts; network ACKs, not timers, permit detach."""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


class CardTransport:
    MAX_MESSAGE_LENGTH = 30000

    def __init__(self, failures=(), block=False):
        self.sent = []
        self.edits = []
        self.finalized = []
        self.failures = list(failures)
        self.first_send = asyncio.Event()
        self.boundary_entered = asyncio.Event()
        self.release = asyncio.Event()
        if not block:
            self.release.set()

    async def send(self, **kwargs):
        message_id = f"card-{len(self.sent) + 1}"
        self.sent.append({**kwargs, "message_id": message_id})
        self.first_send.set()
        return SimpleNamespace(success=True, message_id=message_id)

    async def edit_message(self, **kwargs):
        self.edits.append(kwargs)
        return SimpleNamespace(success=True, message_id=kwargs["message_id"])

    async def finalize_streaming_message(self, message_id, content, *, status="", stopped=False):
        self.finalized.append((message_id, content, status))
        if message_id == "card-1":
            self.boundary_entered.set()
            await self.release.wait()
            if self.failures:
                result = self.failures.pop(0)
                if isinstance(result, Exception):
                    raise result
                return result
        return True

    @staticmethod
    def truncate_message(content, limit, **kwargs):
        return [content]


def consumer_for(adapter):
    return GatewayStreamConsumer(
        adapter, "chat", StreamConsumerConfig(edit_interval=999, buffer_threshold=99999, cursor=""),
        metadata={"streaming": True}, initial_reply_to_id="original-user-message",
    )


@pytest.mark.asyncio
async def test_handoff_waits_for_full_old_card_ack_before_editing_new_reply():
    adapter = CardTransport(block=True)
    consumer = consumer_for(adapter)
    consumer.on_delta("Already visible. ")
    task = asyncio.create_task(consumer.run())
    try:
        await asyncio.wait_for(adapter.first_send.wait(), 3)
        receipt = consumer.register_followup("Use correction", "new-user-message", {"thread_id": "topic"}, lambda: None)
        await consumer.acknowledge_followup(receipt)
        # All these bytes are still buffered, including code/unicode/progress.
        tail = "\n```python\nprint('完整尾部🫡')\n```\n" + "未刷新的内容。" * 700
        consumer.on_progress("\n> terminal: finished old step\n")
        consumer.on_delta(tail)
        done = consumer.on_user_input_boundary(text="Use correction")
        consumer.on_delta("NEW ANSWER")
        consumer.finish("NEW ANSWER VERIFIED")
        await asyncio.wait_for(adapter.boundary_entered.wait(), 3)
        assert not done.is_set()
        assert consumer.message_id == "card-1"
        assert adapter.finalized[0][1] == "Already visible. \n> terminal: finished old step\n" + tail
        assert not any("NEW ANSWER" in x["content"] for x in adapter.edits)
        adapter.release.set()
        await asyncio.wait_for(task, 5)
        assert done.is_set()
        assert len(adapter.sent) == 2  # receipt reused, no third response card
        old, new = adapter.finalized
        assert old[2].startswith("已转向")
        assert "NEW ANSWER" not in old[1]
        assert tail not in new[1]
        assert new[1].count("NEW ANSWER VERIFIED") == 1
        assert adapter.sent[1]["reply_to"] == "new-user-message"
        new_edits = [x for x in adapter.edits if x["message_id"] == "card-2"]
        assert all(x["metadata"]["reply_to_message_id"] == "new-user-message" for x in new_edits)
        assert all(x["metadata"]["thread_id"] == "topic" for x in new_edits)
    finally:
        adapter.release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, None, RuntimeError("network failure")])
async def test_unconfirmed_boundary_keeps_unflushed_old_tail(failure):
    adapter = CardTransport(failures=[failure, failure, failure])
    consumer = consumer_for(adapter)
    consumer.on_delta("OLD COMPLETE TAIL")
    consumer.on_user_input_boundary()
    consumer.on_delta("NEW ANSWER")
    consumer.finish("NEW ANSWER")
    await asyncio.wait_for(consumer.run(), 5)
    assert len(adapter.sent) == 1
    # A false/None/exception ACK may not reset the old buffer. The final
    # successful commit must carry BOTH sides, with an explicit fallback note.
    final = adapter.finalized[-1][1]
    assert "OLD COMPLETE TAIL" in final
    assert "NEW ANSWER" in final
    assert "暂在本卡继续" in final


@pytest.mark.asyncio
async def test_retry_reuses_identical_complete_snapshot_then_rotates():
    adapter = CardTransport(failures=[False, RuntimeError("retry")])
    consumer = consumer_for(adapter)
    consumer.on_delta("OLD TAIL")
    consumer.on_user_input_boundary()
    consumer.on_delta("NEW")
    consumer.finish("NEW")
    await asyncio.wait_for(consumer.run(), 5)
    assert [x[1] for x in adapter.finalized[:3]] == ["OLD TAIL"] * 3
    assert adapter.finalized[-1][:2] == ("card-2", "NEW")


@pytest.mark.asyncio
async def test_same_text_received_after_consumption_is_not_stolen_by_old_boundary():
    adapter = CardTransport()
    consumer = consumer_for(adapter)
    first = consumer.register_followup("again", "u1", {}, lambda: None)
    second = consumer.register_followup("again", "u2", {}, lambda: None)
    first.ready.set()
    second.ready.set()
    consumer.on_delta("A")
    consumer.on_user_input_boundary(text="again")
    consumer.on_delta("B")
    consumer.on_user_input_boundary(text="again")
    consumer.on_delta("C")
    consumer.finish("C")
    await asyncio.wait_for(consumer.run(), 5)
    assert [x["reply_to"] for x in adapter.sent] == ["original-user-message", "u1", "u2"]
    assert "A" in adapter.finalized[0][1]
    assert "C" not in adapter.finalized[0][1]


@pytest.mark.asyncio
async def test_accepted_but_unconsumed_followup_is_requeued_on_final_race():
    adapter = CardTransport()
    consumer = consumer_for(adapter)
    requeued = []
    receipt = consumer.register_followup("late", "u1", {}, lambda: (requeued.append("late"), True)[1])
    await consumer.acknowledge_followup(receipt)
    consumer.finish()
    await asyncio.wait_for(consumer.run(), 5)
    assert requeued == ["late"]
    assert adapter.finalized[-1][2] == "等待后续处理"


@pytest.mark.asyncio
async def test_rejected_followup_does_not_create_a_boundary_or_requeue_twice():
    adapter = CardTransport()
    consumer = consumer_for(adapter)
    requeued = []
    receipt = consumer.register_followup("late", "u1", {}, lambda: (requeued.append("late"), True)[1])
    consumer.discard_followup(receipt)
    consumer.on_delta("UNINTERRUPTED")
    consumer.finish()
    await consumer.run()
    assert len(adapter.sent) == 1
    assert requeued == []


@pytest.mark.asyncio
@pytest.mark.parametrize("requeue_result", [False, RuntimeError("queue unavailable")])
async def test_stopped_or_failed_followup_does_not_claim_it_was_queued(requeue_result):
    adapter = CardTransport()
    consumer = consumer_for(adapter)

    def requeue():
        if isinstance(requeue_result, Exception):
            raise requeue_result
        return requeue_result

    receipt = consumer.register_followup("late", "u1", {}, requeue)
    await consumer.acknowledge_followup(receipt)
    consumer.finish()
    await asyncio.wait_for(consumer.run(), 5)
    assert adapter.finalized[-1][2] != "等待后续处理"
    assert "将按后续消息处理" not in adapter.finalized[-1][1]


@pytest.mark.asyncio
async def test_cancelled_boundary_closes_receipt_without_replaying_consumed_input():
    adapter = CardTransport(block=True)
    consumer = consumer_for(adapter)
    requeued = []
    receipt = consumer.register_followup("correction", "u1", {}, lambda: requeued.append("bad"))
    consumer.on_delta("OLD COMPLETE TAIL")
    task = asyncio.create_task(consumer.run())
    await asyncio.wait_for(adapter.first_send.wait(), 3)
    await consumer.acknowledge_followup(receipt)
    done = consumer.on_user_input_boundary(text="correction")
    await asyncio.wait_for(adapter.boundary_entered.wait(), 3)
    task.cancel()
    adapter.release.set()  # The cancellation handler can now seal the old card.
    await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 5)
    assert done.is_set()
    assert requeued == []
    assert adapter.finalized[-1][2] == "回复已中断"
    assert "OLD COMPLETE TAIL" in consumer._accumulated


@pytest.mark.asyncio
async def test_batch_receipts_are_sealed_and_latest_anchor_owns_new_output():
    adapter = CardTransport()
    consumer = consumer_for(adapter)
    consumer.on_delta("OLD")
    task = asyncio.create_task(consumer.run())
    await asyncio.wait_for(adapter.first_send.wait(), 3)
    for text, anchor in [("one", "u1"), ("two", "u2")]:
        receipt = consumer.register_followup(text, anchor, {}, lambda: False)
        await consumer.acknowledge_followup(receipt)
    consumer.on_user_input_boundary(text="one\ntwo")
    consumer.on_delta("NEW")
    consumer.finish("NEW")
    await asyncio.wait_for(task, 5)
    assert len(adapter.sent) == 3
    assert adapter.finalized[1][0] == "card-2"
    assert adapter.finalized[1][2] == "已合并补充"
    assert adapter.finalized[-1][0] == "card-3"
    assert adapter.sent[-1]["reply_to"] == "u2"
    assert adapter.finalized[-1][1].count("NEW") == 1
