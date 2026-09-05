"""Fork CardKit ledger and upstream authoritative-final queue must compose."""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


class CardAdapter:
    MAX_MESSAGE_LENGTH = 30000

    def __init__(self):
        self.sent = []
        self.edits = []
        self.finalized = []
        self.authoritative_seen = asyncio.Event()

    async def send(self, **kwargs):
        self.sent.append(kwargs["content"])
        return SimpleNamespace(success=True, message_id=f"card_{len(self.sent)}")

    async def edit_message(self, **kwargs):
        self.edits.append(kwargs["content"])
        if "VERIFIED" in kwargs["content"]:
            self.authoritative_seen.set()
        return SimpleNamespace(success=True, message_id=kwargs["message_id"])

    async def finalize_streaming_message(self, message_id, content, **kwargs):
        self.finalized.append((message_id, content))
        return True

    @staticmethod
    def truncate_message(content, limit, **kwargs):
        return [content]


@pytest.mark.parametrize("split_queue", [False, True])
@pytest.mark.parametrize("final", ["Answer VERIFIED", "Rewritten VERIFIED answer"])
@pytest.mark.asyncio
async def test_authoritative_final_keeps_preamble_progress_and_one_card(split_queue, final):
    adapter = CardAdapter()
    consumer = GatewayStreamConsumer(
        adapter, "dm", StreamConsumerConfig(edit_interval=0.001, buffer_threshold=1, cursor=""),
        metadata={"streaming": True},
    )
    consumer.on_delta("Preamble")
    consumer.on_delta(None)
    consumer.on_progress("Searching sources")
    consumer.on_progress_boundary()
    consumer.on_delta("Answer")
    # Model and consumer are different threads. FINAL_TEXT can be drained
    # before finish() puts DONE; hold that second put until an actual edit.
    put = consumer._queue.put
    held = []
    if split_queue:
        calls = 0
        def stagger(item):
            nonlocal calls
            calls += 1
            if calls == 1:
                put(item)
            else:
                held.append(item)
        consumer._queue.put = stagger
    consumer.finish(final)
    task = asyncio.create_task(consumer.run())
    try:
        if split_queue:
            await asyncio.wait_for(adapter.authoritative_seen.wait(), 2)
            assert len(held) == 1
            put(held[0])
        await asyncio.wait_for(task, 2)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert len(adapter.sent) == 1
    assert len(adapter.finalized) == 1
    text = adapter.finalized[0][1]
    assert "Preamble" in text and "Searching sources" in text
    assert text.endswith(final)
    assert text.count(final) == 1
    if final.startswith("Rewritten"):
        assert "\nAnswer" not in text
    assert consumer.delivered_final_matches(final) is True
    assert consumer.final_response_sent is True


@pytest.mark.parametrize("boundaries", [1, 2])
@pytest.mark.parametrize("cumulative_final", [False, True])
@pytest.mark.asyncio
async def test_steer_sealed_prefix_never_reappears_in_the_new_card(boundaries, cumulative_final):
    adapter = CardAdapter()
    consumer = GatewayStreamConsumer(
        adapter, "dm", StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor=""),
        metadata={"streaming": True},
    )
    prefix = []
    for index in range(boundaries):
        prefix.append(f"Before steer {index}")
        consumer.on_delta(prefix[-1])
        consumer.on_user_input_boundary()
    consumer.on_delta("Answer")
    final = ("".join(prefix) if cumulative_final else "") + "Answer VERIFIED"
    consumer.finish(final)
    await asyncio.wait_for(consumer.run(), 3)
    assert [text for _, text in adapter.finalized] == [*prefix, "Answer VERIFIED"]
    assert consumer.delivered_final_matches(final) is True
