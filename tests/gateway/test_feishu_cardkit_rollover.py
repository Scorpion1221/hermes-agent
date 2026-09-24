"""CardKit card rollover: long replies continue in fresh cards.

Feishu closes a card's streaming mode ~600 s after creation. The consumer
seals the live card (full-card update + footer) and streams the rest of the
reply into a new card, triggered by the gateway heartbeat, a card-age cap, or
an edit that reports the window already closed. Cards are cut at paragraph
boundaries; an unfinished paragraph/table/code block moves to the next card.
The final answer must land exactly once and the gateway must see it as
delivered.
"""
import asyncio
import time
from types import SimpleNamespace

import pytest

from gateway.stream_consumer import (
    _DONE,
    _FINAL_TEXT,
    _ROLLOVER,
    GatewayStreamConsumer,
    StreamConsumerConfig,
)


class RolloverCardTransport:
    """Fake Feishu CardKit adapter speaking the rollover contract."""

    MAX_MESSAGE_LENGTH = 30000

    def __init__(
        self, *, seal_failures=0, expire_after_first_send=False, edit_failures=False, send_failures=0,
    ):
        self.sent = []
        self.edits = []
        self.finalized = []
        self.sealed_updates = []
        self.completed = []
        self.seal_failures = seal_failures
        self.send_failures = send_failures
        self.expire_after_first_send = expire_after_first_send
        self.edit_failures = edit_failures
        self.expired = set()
        self.first_send = asyncio.Event()

    async def send(self, **kwargs):
        if self.send_failures and self.sent:
            self.send_failures -= 1
            return SimpleNamespace(success=False, message_id=None, error="timeout")
        message_id = f"card-{len(self.sent) + 1}"
        self.sent.append({**kwargs, "message_id": message_id})
        if self.expire_after_first_send and message_id == "card-1":
            self.expired.add(message_id)
        self.first_send.set()
        return SimpleNamespace(success=True, message_id=message_id)

    async def edit_message(self, **kwargs):
        self.edits.append(kwargs)
        message_id = kwargs["message_id"]
        if message_id in self.expired:
            return SimpleNamespace(
                success=False,
                message_id=None,
                error="CardKit streaming window closed",
                raw_response={"cardkit_stream_expired": True, "code": 200850},
            )
        if self.edit_failures:
            return SimpleNamespace(success=False, message_id=None, error="CardKit stream failed")
        return SimpleNamespace(success=True, message_id=message_id)

    async def finalize_streaming_message(
        self, message_id, final_text="", *, stopped=False, status="", footer=None,
    ):
        self.finalized.append(
            {"message_id": message_id, "content": final_text, "footer": footer, "stopped": stopped}
        )
        if footer is not None and self.seal_failures:
            self.seal_failures -= 1
            return False
        return True

    async def update_sealed_streaming_message(
        self, message_id, content="", *, footer=None, stopped=False, status="",
    ):
        self.sealed_updates.append(
            {"message_id": message_id, "content": content, "footer": footer, "stopped": stopped}
        )
        return True

    async def on_streaming_message_complete(self, message_id):
        self.completed.append(message_id)

    @staticmethod
    def truncate_message(content, limit, **kwargs):
        return [content]

    def contents_for(self, message_id):
        texts = [s["content"] for s in self.sent if s["message_id"] == message_id]
        texts += [e["content"] for e in self.edits if e["message_id"] == message_id]
        texts += [f["content"] for f in self.finalized if f["message_id"] == message_id]
        return texts

    def seals(self):
        return [f for f in self.finalized if f["footer"] is not None]


def consumer_for(adapter, *, edit_interval=0.01, min_age=0.0, min_chars=0):
    consumer = GatewayStreamConsumer(
        adapter,
        "chat",
        StreamConsumerConfig(edit_interval=edit_interval, buffer_threshold=1, cursor=""),
        metadata={"streaming": True},
        initial_reply_to_id="user-message",
    )
    consumer._cardkit_min_rollover_age = min_age
    consumer._cardkit_rollover_min_chars = min_chars
    return consumer


async def wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_heartbeat_rollover_seals_live_card_and_continues_below():
    adapter = RolloverCardTransport()
    consumer = consumer_for(adapter)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("I'll check the logs.")
    consumer.on_delta(None)  # tool boundary
    consumer.on_progress("\n> terminal: ls\n")
    await asyncio.wait_for(adapter.first_send.wait(), 3)

    verdict = await consumer.request_cardkit_rollover(iteration=3, max_iterations=90)

    assert verdict == "rolled"
    seal = adapter.finalized[0]
    assert seal["message_id"] == "card-1"
    assert seal["content"] == "I'll check the logs.\n> terminal: ls"
    assert "第 3/90 轮" in seal["footer"] and "继续 ↓" in seal["footer"]

    consumer.on_delta("All clear.")
    consumer.finish("All clear.")
    await asyncio.wait_for(task, 3)

    assert [s["message_id"] for s in adapter.sent] == ["card-1", "card-2"]
    assert all("check the logs" not in text for text in adapter.contents_for("card-2"))
    last = adapter.finalized[-1]
    assert last["message_id"] == "card-2" and last["footer"] is None
    assert last["content"] == "All clear."
    # Continuation cards report the whole reply's elapsed time.
    assert "cardkit_elapsed_origin" in adapter.sent[1]["metadata"]
    assert consumer.message_id == "card-2"
    assert consumer.final_response_sent is True
    assert consumer.delivered_final_matches("All clear.") is True
    assert adapter.completed == ["card-2"]


@pytest.mark.asyncio
async def test_rollover_mid_answer_cuts_at_a_paragraph_and_carries_the_rest():
    adapter = RolloverCardTransport()
    consumer = consumer_for(adapter)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("## Findings\n\nThe probe fails.\n\n| probe | port |\n|---|---|\n| a | 80")
    await asyncio.wait_for(adapter.first_send.wait(), 3)

    assert await consumer.request_cardkit_rollover() == "rolled"
    # The unfinished table moved to the next card whole.
    assert adapter.seals()[0]["content"] == "## Findings\n\nThe probe fails."
    await wait_for(lambda: len(adapter.sent) == 2)
    assert adapter.sent[1]["content"].startswith("| probe | port |\n|---|---|")

    consumer.on_delta(" |\n| b | 443 |\n\nDone.")
    final = "## Findings\n\nThe probe fails.\n\n| probe | port |\n|---|---|\n| a | 80 |\n| b | 443 |\n\nDone."
    consumer.finish(final)
    await asyncio.wait_for(task, 3)

    assert adapter.finalized[-1]["content"] == "| probe | port |\n|---|---|\n| a | 80 |\n| b | 443 |\n\nDone."
    assert consumer.delivered_final_matches(final) is True


@pytest.mark.asyncio
async def test_rollover_carries_an_unfinished_code_block_with_its_opener():
    adapter = RolloverCardTransport()
    consumer = consumer_for(adapter)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("Patch:\n\n```python\nprint(1)\n")
    await asyncio.wait_for(adapter.first_send.wait(), 3)

    assert await consumer.request_cardkit_rollover() == "rolled"
    consumer.on_delta("```\n\nDone.")
    final = "Patch:\n\n```python\nprint(1)\n```\n\nDone."
    consumer.finish(final)
    await asyncio.wait_for(task, 3)

    assert adapter.seals()[0]["content"] == "Patch:"
    assert adapter.finalized[-1]["content"] == "```python\nprint(1)\n```\n\nDone."
    assert consumer.delivered_final_matches(final) is True


@pytest.mark.asyncio
async def test_mid_paragraph_card_is_not_cut_by_heartbeat():
    adapter = RolloverCardTransport()
    consumer = consumer_for(adapter)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("A single paragraph that is still being")
    await asyncio.wait_for(adapter.first_send.wait(), 3)

    assert await consumer.request_cardkit_rollover() == "live"

    consumer.finish()
    await asyncio.wait_for(task, 3)
    assert adapter.seals() == []


@pytest.mark.parametrize("closing", ["progress", "trailing_boundary"])
@pytest.mark.asyncio
async def test_turn_ending_right_after_rollover_completes_the_sealed_card(closing):
    adapter = RolloverCardTransport()
    consumer = consumer_for(adapter)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("The whole answer.")
    if closing == "progress":
        consumer.on_progress("\n> memory: saved\n")
    else:
        # Runtimes that project the completed final as a trailing boundary.
        consumer.on_segment_break()
    await asyncio.wait_for(adapter.first_send.wait(), 3)
    assert await consumer.request_cardkit_rollover() == "rolled"

    consumer.finish("The whole answer.")
    await asyncio.wait_for(task, 3)

    # No second card with the answer; the sealed card is restatused complete.
    assert len(adapter.sent) == 1
    assert adapter.sealed_updates[-1]["message_id"] == "card-1"
    assert adapter.sealed_updates[-1]["footer"] is None
    assert consumer.message_id == "card-1"
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True
    assert consumer.delivered_final_matches("The whole answer.") is True
    assert adapter.completed == ["card-1"]


@pytest.mark.asyncio
async def test_authoritative_suffix_after_rollover_does_not_repeat_sealed_head():
    adapter = RolloverCardTransport()
    consumer = consumer_for(adapter)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("Head of the answer.")
    consumer.on_progress("\n> verify: 2 files\n")
    await asyncio.wait_for(adapter.first_send.wait(), 3)
    assert await consumer.request_cardkit_rollover() == "rolled"

    final = "Head of the answer.\n\n---\nVerified 2 files."
    consumer.finish(final)
    await asyncio.wait_for(task, 3)

    assert [s["message_id"] for s in adapter.sent] == ["card-1", "card-2"]
    assert adapter.finalized[-1]["content"] == "---\nVerified 2 files."
    assert all("Head of the answer" not in t for t in adapter.contents_for("card-2"))
    assert consumer.delivered_final_matches(final) is True


@pytest.mark.asyncio
async def test_trailing_segment_boundary_after_mid_answer_rollover_does_not_repeat_head():
    adapter = RolloverCardTransport()
    consumer = consumer_for(adapter)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("Alpha beta.\n\n")
    await asyncio.wait_for(adapter.first_send.wait(), 3)
    assert await consumer.request_cardkit_rollover() == "rolled"
    consumer.on_delta("Gamma delta.")
    await wait_for(lambda: len(adapter.sent) == 2)
    consumer.on_segment_break()
    consumer.finish("Alpha beta.\n\nGamma delta.")
    await asyncio.wait_for(task, 3)

    assert adapter.finalized[-1]["message_id"] == "card-2"
    assert adapter.finalized[-1]["content"] == "Gamma delta."
    assert consumer.delivered_final_matches("Alpha beta.\n\nGamma delta.") is True


@pytest.mark.asyncio
async def test_heartbeat_without_live_card_refreshes_last_sealed_footer():
    adapter = RolloverCardTransport()
    consumer = consumer_for(adapter)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("Working on it.")
    consumer.on_delta(None)
    await asyncio.wait_for(adapter.first_send.wait(), 3)
    assert await consumer.request_cardkit_rollover() == "rolled"

    # Silent period (model thinking): nothing new to stream.
    assert await consumer.request_cardkit_rollover(iteration=5, max_iterations=90) == "annotated"
    update = adapter.sealed_updates[-1]
    assert update["message_id"] == "card-1"
    assert update["content"] == "Working on it."
    # Same wording as the seal, so it stays true once content resumes below.
    assert "第 5/90 轮" in update["footer"] and "继续 ↓" in update["footer"]

    consumer.finish()
    await asyncio.wait_for(task, 3)


@pytest.mark.asyncio
async def test_heartbeat_before_first_card_falls_back_to_status_message():
    adapter = RolloverCardTransport()
    consumer = consumer_for(adapter)
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)

    assert await consumer.request_cardkit_rollover() == "no_card"
    assert adapter.finalized == []

    consumer.finish()
    await asyncio.wait_for(task, 3)


@pytest.mark.asyncio
async def test_heartbeat_falls_back_when_the_run_loop_never_started():
    adapter = RolloverCardTransport()
    consumer = consumer_for(adapter)

    # Nothing would ever drain a queued request.
    assert await consumer.request_cardkit_rollover(timeout=0.2) == "no_card"


@pytest.mark.asyncio
async def test_young_or_small_cards_are_left_alone():
    adapter = RolloverCardTransport()
    consumer = consumer_for(adapter, min_age=60.0)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("Just started.")
    consumer.on_delta(None)
    await asyncio.wait_for(adapter.first_send.wait(), 3)

    assert await consumer.request_cardkit_rollover() == "live"
    # Old enough, but still a one-liner: don't splinter the reply.
    consumer._cardkit_min_rollover_age = 0.0
    consumer._cardkit_rollover_min_chars = 300
    assert await consumer.request_cardkit_rollover() == "live"
    # A small card is still rolled once it has been up for a while.
    consumer._cardkit_small_card_max_age = 0.0
    assert await consumer.request_cardkit_rollover() == "rolled"

    consumer.finish()
    await asyncio.wait_for(task, 3)


@pytest.mark.asyncio
async def test_inline_fence_in_progress_line_does_not_block_rollover():
    adapter = RolloverCardTransport()
    consumer = consumer_for(adapter)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("Searching.")
    consumer.on_delta(None)
    consumer.on_progress("\n> terminal: \"grep -rn '```' docs/\"\n")
    await asyncio.wait_for(adapter.first_send.wait(), 3)

    assert await consumer.request_cardkit_rollover() == "rolled"

    consumer.finish()
    await asyncio.wait_for(task, 3)


@pytest.mark.asyncio
async def test_adapter_without_rollover_support_keeps_legacy_status_messages():
    class LegacyCardAdapter(RolloverCardTransport):
        async def finalize_streaming_message(self, message_id, content, **kwargs):
            self.finalized.append({"message_id": message_id, "content": content, "footer": None})
            return True

    adapter = LegacyCardAdapter()
    consumer = consumer_for(adapter)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("Hello.")
    await asyncio.wait_for(adapter.first_send.wait(), 3)

    assert await consumer.request_cardkit_rollover() == "unsupported"

    consumer.finish("Hello.")
    await asyncio.wait_for(task, 3)
    assert [s["message_id"] for s in adapter.sent] == ["card-1"]


@pytest.mark.asyncio
async def test_failed_seal_keeps_all_content_in_the_same_card():
    adapter = RolloverCardTransport(seal_failures=2)
    consumer = consumer_for(adapter)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("First.")
    consumer.on_delta(None)
    await asyncio.wait_for(adapter.first_send.wait(), 3)

    assert await consumer.request_cardkit_rollover() == "failed"
    consumer.on_delta(" Second.")
    consumer.finish("Second.")
    await asyncio.wait_for(task, 3)

    assert [s["message_id"] for s in adapter.sent] == ["card-1"]
    last = adapter.finalized[-1]
    assert last["message_id"] == "card-1" and last["footer"] is None
    assert last["content"] == "First. Second."
    assert consumer.delivered_final_matches("Second.") is True


@pytest.mark.asyncio
async def test_continuation_card_send_failure_on_the_last_tick_is_retried():
    adapter = RolloverCardTransport(send_failures=1)
    consumer = consumer_for(adapter)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("Step one done.")
    consumer.on_delta(None)
    await asyncio.wait_for(adapter.first_send.wait(), 3)
    assert await consumer.request_cardkit_rollover() == "rolled"

    consumer.on_delta("Answer.")
    consumer.finish("Answer.")
    await asyncio.wait_for(task, 3)

    assert [s["message_id"] for s in adapter.sent] == ["card-1", "card-2"]
    assert consumer.final_response_sent is True
    assert consumer.delivered_final_matches("Answer.") is True


@pytest.mark.asyncio
async def test_expired_streaming_window_rolls_over_instead_of_freezing():
    adapter = RolloverCardTransport(expire_after_first_send=True)
    consumer = consumer_for(adapter)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("Before the cutoff.")
    await asyncio.wait_for(adapter.first_send.wait(), 3)

    consumer.on_delta(None)
    consumer.on_progress("\n> terminal: make test\n")
    await wait_for(lambda: adapter.finalized)
    # The next card opens lazily with the next content.
    consumer.on_delta("After the cutoff.")
    consumer.finish("After the cutoff.")
    await asyncio.wait_for(task, 3)

    seal = adapter.finalized[0]
    assert seal["message_id"] == "card-1" and "继续 ↓" in seal["footer"]
    # Everything queued before the expiry was sealed into card-1.
    assert seal["content"] == "Before the cutoff.\n> terminal: make test"
    last = adapter.finalized[-1]
    assert last["message_id"] == "card-2"
    assert last["content"] == "After the cutoff."
    assert consumer.delivered_final_matches("After the cutoff.") is True


@pytest.mark.asyncio
async def test_card_age_cap_rolls_over_before_feishu_cutoff():
    adapter = RolloverCardTransport()
    consumer = consumer_for(adapter)
    consumer._cardkit_max_card_age = 0.2
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("Early.\n\n")
    await asyncio.wait_for(adapter.first_send.wait(), 3)
    await asyncio.sleep(0.3)
    consumer.on_delta("Late.")
    await wait_for(lambda: adapter.finalized)
    consumer.on_delta(" Later.")
    consumer.finish("Early.\n\nLate. Later.")
    await asyncio.wait_for(task, 3)

    assert adapter.finalized[0]["message_id"] == "card-1"
    assert adapter.finalized[0]["content"] == "Early."
    assert adapter.finalized[0]["footer"]
    assert adapter.finalized[-1]["message_id"] == "card-2"
    assert adapter.finalized[-1]["content"] == "Late. Later."
    assert consumer.delivered_final_matches("Early.\n\nLate. Later.") is True


@pytest.mark.asyncio
async def test_no_automatic_rollover_once_the_turn_is_finishing():
    adapter = RolloverCardTransport()
    consumer = consumer_for(adapter)
    consumer._cardkit_max_card_age = 0.0
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("Answer.")
    await asyncio.wait_for(adapter.first_send.wait(), 3)
    # finish() runs on the agent thread: the loop can drain FINAL_TEXT
    # before DONE is queued.
    consumer._finish_requested = True
    consumer._queue.put((_FINAL_TEXT, "Answer.\n\nVerified."))
    await asyncio.sleep(0.2)
    consumer._queue.put(_DONE)
    await asyncio.wait_for(task, 3)

    assert adapter.seals() == []
    assert adapter.finalized[-1]["content"] == "Answer.\n\nVerified."
    assert consumer.delivered_final_matches("Answer.\n\nVerified.") is True


@pytest.mark.asyncio
async def test_rollover_requests_during_or_after_finish_are_resolved():
    adapter = RolloverCardTransport()
    consumer = consumer_for(adapter)
    consumer.finish()
    assert await consumer.request_cardkit_rollover() == "finishing"

    # A marker that races in behind DONE is resolved when the loop exits.
    late = asyncio.get_running_loop().create_future()
    consumer._queue.put((_ROLLOVER, ({}, late)))
    await asyncio.wait_for(consumer.run(), 3)
    assert late.result() == "finishing"


@pytest.mark.asyncio
async def test_steer_prefix_after_rollover_is_not_repeated_in_the_new_card():
    adapter = RolloverCardTransport()
    consumer = consumer_for(adapter)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("Para one.\n\n")
    await asyncio.wait_for(adapter.first_send.wait(), 3)
    assert await consumer.request_cardkit_rollover() == "rolled"
    consumer.on_delta("Para two.")
    await wait_for(lambda: len(adapter.sent) == 2)
    done = consumer.on_user_input_boundary()
    await asyncio.get_running_loop().run_in_executor(None, done.wait, 3)
    consumer.on_delta("After steer.")
    # A provider that returns the whole turn as the final response.
    consumer.finish("Para one.\n\nPara two.\n\nAfter steer.")
    await asyncio.wait_for(task, 3)

    assert adapter.finalized[-1]["content"] == "After steer."


@pytest.mark.asyncio
async def test_finish_after_edit_failures_does_not_back_off_into_the_join_timeout():
    adapter = RolloverCardTransport(edit_failures=True)
    consumer = consumer_for(adapter, edit_interval=0.001)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("x")
    await asyncio.wait_for(adapter.first_send.wait(), 3)
    for i in range(40):
        consumer.on_delta(f" {i}")
        await asyncio.sleep(0.01)

    finished_at = time.monotonic()
    consumer.finish("x done")
    # The gateway only waits 5 s before cancelling into a stopped card.
    await asyncio.wait_for(task, 2)

    assert time.monotonic() - finished_at < 2
    assert adapter.finalized[-1]["stopped"] is False
    assert consumer.final_response_sent is True


@pytest.mark.asyncio
async def test_cardkit_edits_are_paced_by_edit_interval():
    adapter = RolloverCardTransport()
    consumer = consumer_for(adapter, edit_interval=0.5)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("a" * 40)
    await asyncio.wait_for(adapter.first_send.wait(), 3)
    started = time.monotonic()
    while time.monotonic() - started < 1.2:
        consumer.on_delta("b")
        await asyncio.sleep(0.02)
    edits = len(adapter.edits)
    consumer.finish()
    await asyncio.wait_for(task, 3)

    # Whole-card CardKit updates: roughly one per interval, not one per tick.
    assert edits <= 4


@pytest.mark.asyncio
async def test_commentary_is_shown_in_the_card_but_not_as_the_answer():
    adapter = RolloverCardTransport()
    consumer = consumer_for(adapter)
    task = asyncio.create_task(consumer.run())
    # GPT-style commentary arrives as a completed message, not deltas.
    consumer.on_commentary("I'll inspect the config first.")
    consumer.on_progress("\n> terminal: cat app.yaml\n")
    consumer.on_delta("The port is wrong.")
    consumer.finish("The port is wrong.")
    await asyncio.wait_for(task, 3)

    card = adapter.finalized[-1]["content"]
    assert card.startswith("I'll inspect the config first.\n")
    assert "> terminal: cat app.yaml" in card
    assert card.endswith("The port is wrong.")
    assert card.count("The port is wrong.") == 1
    assert consumer.delivered_final_matches("The port is wrong.") is True


@pytest.mark.asyncio
async def test_card_over_the_size_limit_rolls_over_at_a_paragraph():
    class SmallCardTransport(RolloverCardTransport):
        MAX_MESSAGE_LENGTH = 700  # -> 600-char safe limit per card

    adapter = SmallCardTransport()
    consumer = consumer_for(adapter)
    task = asyncio.create_task(consumer.run())
    paragraphs = [f"Paragraph {i}: " + "word " * 40 for i in range(6)]
    final = "\n\n".join(p.strip() for p in paragraphs)
    consumer.on_delta(paragraphs[0].strip())
    await asyncio.wait_for(adapter.first_send.wait(), 3)
    for p in paragraphs[1:]:
        consumer.on_delta("\n\n" + p.strip())
        await asyncio.sleep(0.05)
    consumer.finish(final)
    await asyncio.wait_for(task, 3)

    seals = adapter.seals()
    assert seals, "an oversized card must be sealed and continued"
    assert all(len(s["content"]) <= 600 for s in seals)
    # Every card is closed (no card left streaming) and the paragraphs are
    # split cleanly across them, each exactly once.
    card_ids = {s["message_id"] for s in adapter.sent}
    assert card_ids == {f["message_id"] for f in adapter.finalized}
    shown = "\n\n".join(
        [s["content"] for s in seals] + [adapter.finalized[-1]["content"]]
    )
    for p in paragraphs:
        assert shown.count(p.strip()) == 1
    assert consumer.delivered_final_matches(final) is True


@pytest.mark.asyncio
async def test_rate_limited_frame_is_resent_on_the_next_tick():
    class RateLimitedOnce(RolloverCardTransport):
        limited = False

        async def edit_message(self, **kwargs):
            if not self.limited and "second" in kwargs["content"]:
                self.limited = True
                self.edits.append(kwargs)
                return SimpleNamespace(
                    success=True, message_id=kwargs["message_id"],
                    raw_response={"cardkit_rate_limited": True},
                )
            return await super().edit_message(**kwargs)

    adapter = RateLimitedOnce()
    consumer = consumer_for(adapter)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("first ")
    await asyncio.wait_for(adapter.first_send.wait(), 3)
    consumer.on_delta("second")
    # No new content: only the skipped frame being retried can deliver it.
    await wait_for(lambda: sum("second" in e["content"] for e in adapter.edits) >= 2)

    consumer.finish()
    await asyncio.wait_for(task, 3)


@pytest.mark.asyncio
async def test_heading_stays_with_its_paragraph_across_a_rollover():
    adapter = RolloverCardTransport()
    consumer = consumer_for(adapter)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("Intro.\n\n## Fix\n\nSet the port")
    await asyncio.wait_for(adapter.first_send.wait(), 3)

    assert await consumer.request_cardkit_rollover() == "rolled"
    assert adapter.seals()[0]["content"] == "Intro."
    await wait_for(lambda: len(adapter.sent) == 2)
    assert adapter.sent[1]["content"].startswith("## Fix\n\nSet the port")

    consumer.finish()
    await asyncio.wait_for(task, 3)
