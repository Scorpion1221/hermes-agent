"""A streamed Feishu reply is one CardKit card: drafts stream into it and the turn's final seals it.

The real adapter and the real GatewayStreamConsumer run against a fake lark client that records every
CardKit / IM call, so the tests see exactly what Feishu would receive.
"""

from __future__ import annotations

import asyncio
import itertools
import json
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.feishu import feishu_stream
from plugins.platforms.feishu.adapter import FeishuAdapter


def _ok(**data):
    return SimpleNamespace(success=lambda: True, code=0, msg="ok", data=SimpleNamespace(**data))


def _refused(code: int, msg: str = "refused"):
    return SimpleNamespace(success=lambda: False, code=code, msg=msg, data=None)


class FakeFeishu:
    """``im.v1.message`` + ``cardkit.v1.card`` / ``card_element`` with a call log."""

    def __init__(self):
        self.calls: list[tuple[str, object]] = []
        self.refuse: dict[str, list] = {}
        self._cards = itertools.count(1)
        self._messages = itertools.count(1)
        outer = self

        class _API:
            def __init__(self, name):
                self._name = name

            def __getattr__(self, method):
                def _call(request):
                    key = f"{self._name}.{method}"
                    outer.calls.append((key, request))
                    queued = outer.refuse.get(key)
                    if queued:
                        return queued.pop(0)
                    if key == "card.create":
                        return _ok(card_id=f"card_{next(outer._cards)}")
                    if key in {"message.create", "message.reply"}:
                        return _ok(message_id=f"om_reply_{next(outer._messages)}")
                    return _ok()
                return _call

        self.client = SimpleNamespace(
            im=SimpleNamespace(v1=SimpleNamespace(message=_API("message"))),
            cardkit=SimpleNamespace(v1=SimpleNamespace(card=_API("card"), card_element=_API("element"))),
        )

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def streamed(self) -> list[str]:
        return [r.request_body.content for name, r in self.calls if name == "element.content"]

    def final_bodies(self) -> list[dict]:
        bodies = []
        for name, r in self.calls:
            if name == "card.update":
                bodies.append(json.loads(r.request_body.card.data))
        return bodies


def _adapter(fake: FakeFeishu, **extra) -> FeishuAdapter:
    adapter = FeishuAdapter(PlatformConfig(extra=extra))
    adapter._client = fake.client

    async def _direct(func, *args):
        return func(*args)

    adapter._run_blocking = _direct
    return adapter


def _final_text(body: dict) -> str:
    return body["body"]["elements"][0]["content"]


def _consumer(adapter: FeishuAdapter, reply_to: str = "om_user_1") -> GatewayStreamConsumer:
    cfg = StreamConsumerConfig(transport="auto", chat_type="dm", edit_interval=0.01, buffer_threshold=1, cursor="")
    return GatewayStreamConsumer(adapter, "oc_chat", cfg, initial_reply_to_id=reply_to)


async def _stream(adapter: FeishuAdapter, deltas: list[str], final: str, reply_to: str = "om_user_1"):
    consumer = _consumer(adapter, reply_to)
    task = asyncio.create_task(consumer.run())
    for delta in deltas:
        consumer.on_delta(delta)
        await asyncio.sleep(0.05)
    consumer.finish(final)
    await task
    return consumer


@pytest.mark.asyncio
async def test_a_streamed_reply_is_one_card_that_the_final_seals():
    fake = FakeFeishu()
    adapter = _adapter(fake)

    consumer = await _stream(adapter, ["Hello ", "world. ", "## Done"], "Hello world. ## Done")

    names = fake.names()
    assert names.count("card.create") == 1
    assert names.count("message.reply") + names.count("message.create") == 1
    assert fake.streamed(), "text should stream into the card"
    (final,) = fake.final_bodies()
    assert final["config"]["streaming_mode"] is False
    assert _final_text(final) == "Hello world. ## Done"
    assert consumer.final_response_sent and consumer.delivered_final_matches("Hello world. ## Done") is True


@pytest.mark.asyncio
async def test_the_final_text_rides_the_seal_when_post_stream_text_was_added():
    fake = FakeFeishu()
    adapter = _adapter(fake)
    final = "The answer.\n\n⚠️ verifier footer"

    consumer = await _stream(adapter, ["The answer."], final)

    (body,) = fake.final_bodies()
    assert _final_text(body) == final
    assert consumer.delivered_final_matches(final) is True


@pytest.mark.asyncio
async def test_an_interim_send_does_not_seal_the_live_card():
    fake = FakeFeishu()
    adapter = _adapter(fake)
    consumer = _consumer(adapter)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("working on it")
    await asyncio.sleep(0.05)

    result = await adapter.send("oc_chat", "⏳ still running", metadata={"_interim_send": True,
                                                                       "reply_to_message_id": "om_user_1"})

    assert result.success and not fake.final_bodies(), "an interim send must not seal the card"
    consumer.finish("working on it, done")
    await task
    (body,) = fake.final_bodies()
    assert _final_text(body) == "working on it, done"


@pytest.mark.asyncio
async def test_a_send_for_another_turn_never_seals_this_card():
    fake = FakeFeishu()
    adapter = _adapter(fake)
    await adapter.send_draft("oc_chat", 7, "turn A text", metadata={"reply_to_message_id": "om_a"})

    await adapter.send("oc_chat", "an ack with no turn id")
    await adapter.send("oc_chat", "turn B's reply", reply_to="om_b")

    assert not fake.final_bodies()


@pytest.mark.asyncio
async def test_stop_seals_the_card_with_what_is_on_screen():
    fake = FakeFeishu()
    adapter = _adapter(fake)
    md = {"reply_to_message_id": "om_user_1"}
    await adapter.send_draft("oc_chat", 3, "partial answer", metadata=md)

    result = await adapter.abandon_open_draft("oc_chat", "partial answer", metadata=md)

    assert result.success
    (body,) = fake.final_bodies()
    assert body["config"]["streaming_mode"] is False and _final_text(body) == "partial answer"
    # A straggler frame for the sealed turn opens nothing.
    await adapter.send_draft("oc_chat", 3, "partial answer and more", metadata=md)
    assert fake.names().count("card.create") == 1


@pytest.mark.asyncio
async def test_a_card_that_cannot_be_created_falls_back_to_a_plain_reply():
    fake = FakeFeishu()
    fake.refuse["card.create"] = [_refused(99991400, "no cardkit permission")] * 10
    adapter = _adapter(fake)

    consumer = await _stream(adapter, ["plain ", "answer"], "plain answer")

    sent = [r for name, r in fake.calls if name in {"message.reply", "message.create"}]
    assert sent and json.loads(sent[-1].request_body.content)["schema"] == "2.0"
    assert consumer.final_response_sent


@pytest.mark.asyncio
async def test_an_old_card_rolls_over_into_a_new_one_with_a_continued_footer(monkeypatch):
    fake = FakeFeishu()
    adapter = _adapter(fake)
    md = {"reply_to_message_id": "om_user_1"}
    await adapter.send_draft("oc_chat", 5, "first part", metadata=md)
    stream = adapter._card_streams.open["oc_chat:turn:om_user_1"]
    stream.card.created_at -= feishu_stream.MAX_CARD_AGE_SECONDS + 1

    await adapter.send_draft("oc_chat", 5, "first part, second part", metadata=md)
    await adapter.send("oc_chat", "first part, second part, end", reply_to="om_user_1")

    first, last = fake.final_bodies()
    assert _final_text(first) == "first part" and len(first["body"]["elements"]) == 2
    assert _final_text(last) == ", second part, end"
    assert fake.names().count("card.create") == 2


@pytest.mark.asyncio
async def test_an_expired_streaming_window_continues_in_a_fresh_card():
    fake = FakeFeishu()
    adapter = _adapter(fake)
    md = {"reply_to_message_id": "om_user_1"}
    await adapter.send_draft("oc_chat", 9, "head", metadata=md)
    fake.refuse["element.content"] = [_refused(200850, "card streaming timeout")]

    result = await adapter.send_draft("oc_chat", 9, "head and tail", metadata=md)
    await adapter.send("oc_chat", "head and tail.", reply_to="om_user_1")

    assert result.success
    sealed_head, final = fake.final_bodies()
    assert _final_text(sealed_head) == "head"
    assert _final_text(final) == "and tail."


@pytest.mark.asyncio
async def test_render_mode_post_or_edit_transport_does_not_stream_cards():
    for extra in ({"render_mode": "post"}, {"streaming_transport": "edit"}):
        assert _adapter(FakeFeishu(), **extra).supports_draft_streaming() is False


@pytest.mark.asyncio
async def test_a_reply_longer_than_one_card_streams_into_two_cards_through_the_consumer(monkeypatch):
    monkeypatch.setattr(feishu_stream, "MAX_CARD_CHARS", 40)
    fake = FakeFeishu()
    adapter = _adapter(fake)
    paragraphs = [f"paragraph {i} " + "x" * 10 + "\n\n" for i in range(6)]

    consumer = await _stream(adapter, paragraphs, "".join(paragraphs).rstrip())

    names = fake.names()
    assert names.count("card.create") >= 2
    assert names.count("message.reply") + names.count("message.create") == names.count("card.create")
    bodies = fake.final_bodies()
    assert all(len(b["body"]["elements"]) == 2 for b in bodies[:-1])  # every rolled card ends "continued"
    shown = "".join(_final_text(b) for b in bodies)
    assert all(f"paragraph {i}" in shown for i in range(6))
    assert consumer.final_response_sent


@pytest.mark.asyncio
async def test_a_send_naming_no_turn_never_seals_the_only_open_card():
    fake = FakeFeishu()
    adapter = _adapter(fake)
    await adapter.send_draft("oc_chat", 11, "the only open reply", metadata={"reply_to_message_id": "om_a"})

    await adapter.send("oc_chat", "a lifecycle notice with no turn id")

    assert not fake.final_bodies()
    assert adapter._card_streams.open, "the card must stay open for its own final"


@pytest.mark.asyncio
async def test_when_no_card_can_be_created_the_draft_reports_failure_so_the_consumer_falls_back():
    fake = FakeFeishu()
    fake.refuse["card.create"] = [_refused(99991400, "no cardkit permission")]
    adapter = _adapter(fake)

    result = await adapter.send_draft("oc_chat", 12, "text", metadata={"reply_to_message_id": "om_a"})

    assert not result.success
    assert not adapter._card_streams.open


def test_cardkit_streaming_tells_the_consumer_never_to_split_and_post_mode_does_not():
    assert _adapter(FakeFeishu()).streaming_overflow_limit() >= 10**6
    assert _adapter(FakeFeishu(), render_mode="post").streaming_overflow_limit() is None


@pytest.mark.asyncio
async def test_an_anchorless_turn_is_sealed_by_its_own_final():
    fake = FakeFeishu()
    adapter = _adapter(fake)

    consumer = await _stream(adapter, ["no anchor ", "here"], "no anchor here", reply_to=None)

    (body,) = fake.final_bodies()
    assert _final_text(body) == "no anchor here"
    assert fake.names().count("card.create") == 1 and consumer.final_response_sent


@pytest.mark.asyncio
async def test_the_sealed_card_keeps_the_preamble_streamed_before_a_tool_call():
    fake = FakeFeishu()
    adapter = _adapter(fake)
    consumer = _consumer(adapter)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("First I will check the logs.")
    await asyncio.sleep(0.05)
    consumer.on_segment_break()
    await asyncio.sleep(0.05)
    consumer.on_delta("Found it: the cache was stale.")
    await asyncio.sleep(0.05)
    final = "Found it: the cache was stale.\n\n⚠️ verifier footer"
    consumer.finish(final)
    await task

    (body,) = fake.final_bodies()
    assert _final_text(body) == "First I will check the logs.\n\nFound it: the cache was stale.\n\n⚠️ verifier footer"
    assert consumer.delivered_final_matches(final) is True


@pytest.mark.parametrize("shown, final, sealed", [
    ("", "answer", "answer"),
    ("answer", "answer", "answer"),
    ("answer so", "answer so far", "answer so far"),
    ("Pre.Seg two", "Seg two, done", "Pre.\n\nSeg two, done"),
    ("Pre.\n\nSeg two", "Unrelated final", "Pre.\n\nSeg two\n\nUnrelated final"),
    ("All shown here.", "", "All shown here."),
])
def test_the_final_completes_the_card_without_erasing_or_repeating(shown, final, sealed):
    assert feishu_stream.merge_final(shown, final) == sealed


@pytest.mark.asyncio
async def test_an_approval_prompt_splits_the_card_and_later_text_streams_below_it():
    from gateway.platforms.base import ExecApprovalPrompt

    fake = FakeFeishu()
    adapter = _adapter(fake)
    md = {"reply_to_message_id": "om_user_1"}
    await adapter.send_draft("oc_chat", 21, "I need to run a command.", metadata=md)

    await adapter._send_exec_approval_prompt(ExecApprovalPrompt(
        chat_id="oc_chat", session_key="s", text="rm -rf build", actions=[("OK", "once", "primary")],
        command="rm -rf build", description="dangerous", smart_denied=False))
    await adapter.send_draft("oc_chat", 21, "I need to run a command. It ran; build cleaned.", metadata=md)
    await adapter.send("oc_chat", "I need to run a command. It ran; build cleaned.", reply_to="om_user_1")

    above, below = fake.final_bodies()
    assert _final_text(above) == "I need to run a command." and len(above["body"]["elements"]) == 1
    assert _final_text(below) == "It ran; build cleaned."
    order = [name for name in fake.names() if name in {"card.update", "message.create", "message.reply"}]
    assert order.index("card.update") < [i for i, n in enumerate(order) if n in {"message.create", "message.reply"}][1]


@pytest.mark.asyncio
async def test_a_clarify_question_splits_the_card_and_never_seals_the_reply():
    fake = FakeFeishu()
    adapter = _adapter(fake)
    md = {"reply_to_message_id": "om_user_1"}
    await adapter.send_draft("oc_chat", 22, "Which file did you mean?", metadata=md)

    await adapter.send_clarify("oc_chat", "Which file?", None, "clar_1", "s", metadata=md)

    (sealed,) = fake.final_bodies()
    assert _final_text(sealed) == "Which file did you mean?"
    assert adapter._card_streams.open, "the turn's stream stays open for the answer after the question"


@pytest.mark.asyncio
async def test_a_quiet_card_is_sealed_before_its_streaming_window_closes():
    fake = FakeFeishu()
    adapter = _adapter(fake)
    md = {"reply_to_message_id": "om_user_1"}
    await adapter.send_draft("oc_chat", 23, "running a long tool…", metadata=md)
    stream = adapter._card_streams.open["oc_chat:turn:om_user_1"]

    await feishu_stream.seal_quiet_cards(adapter)
    assert not fake.final_bodies(), "a young card is left alone"
    stream.card.created_at -= feishu_stream.MAX_CARD_AGE_SECONDS + 1
    await feishu_stream.seal_quiet_cards(adapter)

    (body,) = fake.final_bodies()
    assert body["config"]["streaming_mode"] is False and len(body["body"]["elements"]) == 2
    assert stream.card is None and stream.sealed_prefix == "running a long tool…"


@pytest.mark.asyncio
async def test_tool_progress_lines_stay_in_the_card_between_the_text_and_repeats_collapse():
    fake = FakeFeishu()
    adapter = _adapter(fake)
    consumer = _consumer(adapter)
    assert consumer.accepts_tool_progress is False  # not before the draft transport is up
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("Let me check.")
    await asyncio.sleep(0.05)
    assert consumer.accepts_tool_progress is True
    consumer.on_tool_progress("💻 terminal: ls")
    consumer.on_tool_progress("💻 terminal: ls (×2)")
    consumer.on_tool_progress("📄 read_file: a.py")
    await asyncio.sleep(0.05)
    consumer.on_segment_break()
    consumer.on_delta("Done: a.py is fine.")
    await asyncio.sleep(0.05)
    consumer.finish("Done: a.py is fine.")
    await task

    (body,) = fake.final_bodies()
    assert _final_text(body) == (
        "Let me check.\n\n> 💻 terminal: ls (×2)\n\n> 📄 read_file: a.py\n\nDone: a.py is fine.")
    assert consumer.delivered_final_matches("Done: a.py is fine.") is True


@pytest.mark.asyncio
async def test_without_the_opt_in_a_draft_stream_does_not_take_tool_progress():
    fake = FakeFeishu()
    adapter = _adapter(fake)
    adapter_cls = type("NoInline", (type(adapter),), {"inline_tool_progress": False})
    adapter.__class__ = adapter_cls
    consumer = _consumer(adapter)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("text")
    await asyncio.sleep(0.05)

    assert consumer.accepts_tool_progress is False
    consumer.finish("text")
    await task
