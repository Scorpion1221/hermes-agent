"""A follow-up folded into the running turn gets its answer in a new card below it.

Feishu streams a turn's reply into one CardKit card. When the user sends a follow-up mid-turn and it
is steered or redirected into the running turn, the rest of the answer used to keep growing the card
above the user's new message. Now the moment the agent puts the follow-up into its context
(``steer_consumed_callback``), the gateway seals the live card with what it shows and the following
frames open a fresh card, which replies below the follow-up.

The real adapter, consumer and turn wiring run against a fake lark client (see
test_feishu_card_stream).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gateway.run_turn_steer_boundary import make_steer_consumed_callback
from tests.gateway.test_feishu_card_stream import FakeFeishu, _adapter, _consumer, _final_text, _sent


class _TurnRunner:
    """The two TurnRunner seams the callback uses: the turn's consumer, and a hop onto the loop."""

    def __init__(self, consumer, loop):
        self._consumer, self._loop = consumer, loop

    def _stream_consumer(self):
        return self._consumer

    def _schedule(self, coro, _log):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)


@pytest.mark.asyncio
async def test_a_steer_consumed_mid_turn_seals_the_card_and_the_answer_continues_in_a_new_one():
    fake = FakeFeishu()
    adapter = _adapter(fake)
    consumer = _consumer(adapter)
    task = asyncio.create_task(consumer.run())
    callback = make_steer_consumed_callback(_TurnRunner(consumer, asyncio.get_running_loop()))

    await _sent(fake, lambda: consumer.on_delta("Checking the production logs."))
    # The agent drains the follow-up on its worker thread and notifies from there.
    await asyncio.to_thread(callback)
    await _sent(fake, lambda: consumer.on_delta(" Staging instead: all green."))
    consumer.finish("Staging instead: all green.")
    await task

    above, below = fake.final_bodies()
    assert _final_text(above) == "Checking the production logs."
    assert _final_text(below) == "Staging instead: all green."
    assert fake.names().count("card.create") == 2


@pytest.mark.asyncio
async def test_without_the_adapter_hook_the_reply_stays_one_stream():
    events = []
    consumer = SimpleNamespace(
        adapter=SimpleNamespace(), chat_id="c1", flush_pending_sync=lambda timeout: events.append("flush"),
    )
    runner = SimpleNamespace(_stream_consumer=lambda: consumer, _schedule=lambda *a: events.append("schedule"))

    make_steer_consumed_callback(runner)()

    assert events == []


def test_the_turn_wires_the_callback_onto_the_agent():
    from tests.gateway.test_display_null_turn_wiring import _wire

    agent = _wire({})
    assert callable(agent.steer_consumed_callback)
