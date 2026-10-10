"""A quoted Feishu reply as the agent sees it: the reply chain above it, who sent it, its media.

Feishu answers ``GET im/v1/messages/{id}`` with ``parent_id`` on every reply and reports a bot
message's sender as ``sender_type="app"`` with ``id`` = the bot's app_id. The client is faked at
that SDK boundary (shared harness in ``test_feishu_fetched_content``); the chain text is then run
through the gateway's real reply-pointer and the context compressor's real pointer match. All
names, ids and texts are synthetic.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import threading
from types import SimpleNamespace

import pytest

from tests.gateway.test_feishu_fetched_content import (  # noqa: F401  (autouse SDK fakes)
    ROOT,
    FakeFeishu,
    _fake_lark_requests,
    dispatched_event,
    forward_root,
    inbound_message,
    item,
    json_card,
    make_adapter,
    markdown,
    text,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
USERS = {"ou_alice": "Alice", "ou_bob": "Bob"}


class ChainFeishu(FakeFeishu):
    """``messages`` keyed by id; ``slow`` message ids and resource keys answer after ``slow_delay``
    seconds or once ``released`` is set, whichever is first; resources succeed."""

    def __init__(self, messages, users=USERS, *, slow=(), slow_delay=0.0, resource_types=None):
        super().__init__(messages, users)
        self.slow, self.slow_delay, self.released = set(slow), slow_delay, threading.Event()
        self.resource_types = resource_types or {}

    def _get_message(self, request):
        if request.message_id in self.slow:
            self.message_gets.append(request)
            self.released.wait(self.slow_delay)  # runs on the adapter's SDK executor thread
            return SimpleNamespace(success=lambda: True, data=SimpleNamespace(items=self.messages[request.message_id]))
        if request.message_id not in self.messages:
            self.message_gets.append(request)
            return SimpleNamespace(success=lambda: False, code=230011, msg="message withdrawn")
        return super()._get_message(request)

    def _get_resource(self, request):
        self.resource_gets.append(request)
        if request.file_key in self.slow:
            self.released.wait(self.slow_delay)
        content_type = self.resource_types.get(request.file_key, "image/png")
        body = PNG if content_type.startswith("image/") else b"%PDF-1.4 synthetic"
        return SimpleNamespace(
            success=lambda: True, file=io.BytesIO(body), file_name=f"{request.file_key}.bin",
            raw=SimpleNamespace(headers={"Content-Type": content_type}),
        )


def msg(message_id, content, *, parent=None, sender="ou_alice", sender_type="user", msg_type="text"):
    node = item(message_id, msg_type, content, upper=None, sender=sender, sender_type=sender_type)
    node.parent_id = parent or ""
    return node


def bot(message_id, content, *, parent=None, sender="cli_self"):
    return msg(message_id, content, parent=parent, sender=sender, sender_type="app", msg_type="interactive")


def card(words):
    return json_card([markdown(text(words))])


def alternating_chain(length):
    """om_1 (Alice, root) ← om_2 (this bot) ← … ← om_<length>, alternating user text and bot cards."""
    chain = {}
    for index in range(1, length + 1):
        parent = f"om_{index - 1}" if index > 1 else None
        if index % 2:
            chain[f"om_{index}"] = [msg(f"om_{index}", {"text": f"question {index}"}, parent=parent)]
        else:
            chain[f"om_{index}"] = [bot(f"om_{index}", card(f"answer {index}"), parent=parent)]
    return chain


def reply_to(message_id, *, extra=None, client):
    adapter = make_adapter(client)
    adapter._quote_chain_max_depth = (extra or {}).get("quote_chain_max_depth", adapter._quote_chain_max_depth)
    event = dispatched_event(adapter, inbound_message("om_new", "text", json.dumps({"text": "and now?"}), parent_id=message_id))
    return adapter, event


def model_text(event):
    """The user turn the agent receives: the gateway's own reply-pointer wrapping."""
    from gateway.run_inbound import GatewayInboundMixin

    return GatewayInboundMixin._prepend_inbound_reply_context(event, None, event.text)


# --- chain order, marking, depth ---

def test_a_reply_chain_reads_oldest_first_with_the_directly_quoted_message_last_and_in_full():
    long_answer = "line one\nline two\n" + "detail " * 200
    messages = alternating_chain(3)
    messages["om_4"] = [bot("om_4", card(long_answer), parent="om_3")]
    client = ChainFeishu(messages)

    _adapter, event = reply_to("om_4", client=client)

    assert event.reply_to_message_id == "om_4"
    lines = event.reply_to_text.split("\n")
    assert lines[:5] == [
        "Earlier messages in this reply chain, oldest first:",
        "- Alice (text): question 1",
        "- You (card): answer 2",
        "- Alice (text): question 3",
        "Directly quoted message (card, from You):",
    ]
    assert "\n".join(lines[5:]) == long_answer.strip()


def test_a_quote_of_a_message_that_is_not_a_reply_reads_as_that_message_alone():
    client = ChainFeishu({"om_1": [msg("om_1", {"text": "standalone"})]})

    _adapter, event = reply_to("om_1", client=client)

    assert event.reply_to_text == "standalone"
    assert len(client.message_gets) == 1


@pytest.mark.parametrize("depth", [0, 1])
def test_depth_zero_or_one_quotes_only_the_direct_message(depth):
    client = ChainFeishu(alternating_chain(5))

    _adapter, event = reply_to("om_5", extra={"quote_chain_max_depth": depth}, client=client)

    assert event.reply_to_text == "question 5"
    assert [r.message_id for r in client.message_gets] == ["om_5"]


def test_the_walk_stops_at_the_configured_depth_and_says_earlier_messages_exist():
    client = ChainFeishu(alternating_chain(11))

    _adapter, event = reply_to("om_11", extra={"quote_chain_max_depth": 4}, client=client)

    assert event.reply_to_text.split("\n")[:5] == [
        "Earlier messages in this reply chain, oldest first:",
        "- … earlier messages not shown",
        "- You (card): answer 8",
        "- Alice (text): question 9",
        "- You (card): answer 10",
    ]
    assert len(client.message_gets) == 4


def test_quote_chain_depth_comes_from_platform_extra():
    from gateway.config import PlatformConfig
    from plugins.platforms.feishu.adapter import FeishuAdapter

    assert FeishuAdapter(PlatformConfig(extra={"quote_chain_max_depth": 3}))._quote_chain_max_depth == 3
    assert FeishuAdapter(PlatformConfig(extra={"quote_chain_max_depth": "bogus"}))._quote_chain_max_depth == 20
    assert FeishuAdapter(PlatformConfig())._quote_chain_max_depth == 20


def test_ancestors_are_bounded_per_message_and_in_total_while_the_quoted_message_is_not():
    from plugins.platforms.feishu.feishu_quote import MAX_ANCESTOR_CHARS, MAX_ANCESTORS_CHARS

    huge = "x" * (MAX_ANCESTOR_CHARS * 3)
    messages = {
        f"om_{i}": [msg(f"om_{i}", {"text": huge}, parent=f"om_{i - 1}" if i > 1 else None)] for i in range(1, 20)
    }
    client = ChainFeishu(messages)

    _adapter, event = reply_to("om_19", client=client)

    head, quoted = event.reply_to_text.split("\nDirectly quoted message (text, from Alice):\n")
    ancestor_lines = head.split("\n")[1:]
    assert quoted == huge
    assert ancestor_lines[0] == "- … earlier messages not shown"
    assert all(len(line) <= MAX_ANCESTOR_CHARS + 20 for line in ancestor_lines)
    assert len(head) <= MAX_ANCESTORS_CHARS + 200


def test_a_withdrawn_ancestor_ends_the_chain_without_losing_the_rest():
    messages = alternating_chain(4)
    del messages["om_2"]
    client = ChainFeishu(messages)

    _adapter, event = reply_to("om_4", client=client)

    assert event.reply_to_text.split("\n")[:3] == [
        "Earlier messages in this reply chain, oldest first:",
        "- … earlier messages not shown",
        "- Alice (text): question 3",
    ]


def test_quoted_cards_and_forwards_in_the_chain_render_as_their_text():
    forward_items = [forward_root(), item("om_f1", "text", {"text": "the numbers"}, at=1)]
    forward_items[0].parent_id = ""
    quoted_card = bot("om_card", card("Summary: all green"), parent=ROOT)
    client = ChainFeishu({ROOT: forward_items, "om_card": [quoted_card]})

    _adapter, event = reply_to("om_card", client=client)

    assert event.reply_to_text.split("\n") == [
        "Earlier messages in this reply chain, oldest first:",
        "- Alice (merged forward): [Merged forward message] - Alice: the numbers",
        "Directly quoted message (card, from You):",
        "Summary: all green",
    ]
    assert client.resource_gets == []


def test_chain_senders_get_names_or_neutral_labels_never_raw_ids():
    messages = {
        "om_1": [msg("om_1", {"text": "from outside"}, sender="ou_external")],
        "om_2": [bot("om_2", card("peer bot answer"), parent="om_1", sender="cli_peer")],
        "om_3": [msg("om_3", {"text": "from bob"}, parent="om_2", sender="ou_bob")],
    }
    client = ChainFeishu(messages)

    _adapter, event = reply_to("om_3", client=client)

    assert event.reply_to_text.split("\n")[1:4] == [
        "- Participant 1 (text): from outside",
        "- Bot (card): peer bot answer",
        "Directly quoted message (text, from Bob):",
    ]
    assert "ou_" not in event.reply_to_text and "cli_" not in event.reply_to_text


def test_a_quoted_mention_of_this_bot_reads_as_its_name_from_the_history_id_shape():
    """Fetched messages carry mention ids as a string plus ``id_type``; the bot is still recognised by open_id."""
    mention = SimpleNamespace(key="@_user_1", id="ou_bot", id_type="open_id", name="Hermes")
    quoted = msg("om_q", {"text": "@_user_1 hi"})
    quoted.mentions = [mention]
    adapter = make_adapter(ChainFeishu({"om_q": [quoted]}))
    adapter._bot_open_id, adapter._bot_name = "ou_bot", ""

    from plugins.platforms.feishu.feishu_quote import fetch_quoted_reply

    assert asyncio.run(fetch_quoted_reply(adapter, "om_q")).text == "@Hermes hi"


# --- the gateway pointer stays intact ---

@pytest.mark.parametrize("parent", [None, "om_1"])
def test_the_compressor_strips_exactly_the_reply_pointer_even_when_the_quote_closes_like_one(parent):
    """A card block ending in `"]` before the next block would end the compressor's pointer match early."""
    from agent.context_compressor import _authored_request_text

    quoted_card = json_card([markdown(text('Use ["a", "b"]')), markdown(text("then deploy"))])
    messages = {"om_1": [msg("om_1", {"text": "which list?"})], "om_2": [bot("om_2", quoted_card, parent=parent)]}
    client = ChainFeishu(messages)

    _adapter, event = reply_to("om_2", client=client)
    turn = model_text(event)

    assert event.reply_to_text.endswith('Use ["a", "b"]\n \nthen deploy')
    assert _authored_request_text(turn) == "and now?"


# --- own-message flag ---

@pytest.mark.parametrize(
    ("quoted", "own"),
    [
        (bot("om_q", card("my earlier answer")), True),
        (bot("om_q", card("another bot's answer"), sender="cli_peer"), False),
        (msg("om_q", {"text": "a person's message"}), False),
    ],
)
def test_reply_to_is_own_message_marks_only_this_bots_messages(quoted, own):
    client = ChainFeishu({"om_q": [quoted]})

    _adapter, event = reply_to("om_q", client=client)

    assert event.reply_to_is_own_message is own
    assert model_text(event).startswith('[Replying to your previous message: "' if own else '[Replying to: "')


def test_own_message_flag_follows_the_directly_quoted_message_not_its_ancestors():
    client = ChainFeishu(alternating_chain(3))

    _adapter, event = reply_to("om_3", client=client)

    assert event.reply_to_is_own_message is False
    assert "- You (card): answer 2" in event.reply_to_text


# --- latency: time budget and cache ---

def test_a_slow_ancestor_costs_at_most_the_budget_and_yields_the_partial_chain(monkeypatch):
    import plugins.platforms.feishu.feishu_quote as quote

    monkeypatch.setattr(quote, "CHAIN_BUDGET_SECONDS", 0.3)
    client = ChainFeishu(alternating_chain(4), slow={"om_2"}, slow_delay=10)
    adapter = make_adapter(client)
    message = inbound_message("om_new", "text", json.dumps({"text": "and now?"}), parent_id="om_4")

    async def timed():
        loop = asyncio.get_running_loop()
        start = loop.time()
        event = await dispatched_event_async(adapter, message)
        elapsed = loop.time() - start
        client.released.set()  # let the abandoned GET finish now rather than at interpreter exit
        return event, elapsed

    event, elapsed = asyncio.run(timed())

    assert elapsed < 3  # stopped at the 0.3s budget, not after the 10s GET
    assert event.reply_to_text.split("\n")[:3] == [
        "Earlier messages in this reply chain, oldest first:",
        "- … earlier messages not shown",
        "- Alice (text): question 3",
    ]


def test_an_ancestor_fetch_the_budget_abandoned_completes_into_the_cache(monkeypatch):
    import plugins.platforms.feishu.feishu_quote as quote

    monkeypatch.setattr(quote, "CHAIN_BUDGET_SECONDS", 0.2)
    client = ChainFeishu(alternating_chain(3), slow={"om_2"}, slow_delay=0.5)
    adapter = make_adapter(client)

    async def two_replies():
        first = await quote.fetch_quoted_reply(adapter, "om_3")
        await asyncio.gather(*adapter._quote_background_fetches)
        return first, await quote.fetch_quoted_reply(adapter, "om_3")

    first, second = asyncio.run(two_replies())

    assert "- … earlier messages not shown" in first.text
    assert second.text.split("\n")[1:3] == ["- Alice (text): question 1", "- You (card): answer 2"]


def test_chain_messages_are_fetched_once_and_shared_by_later_replies():
    client = ChainFeishu(alternating_chain(6))
    adapter = make_adapter(client)

    from plugins.platforms.feishu.feishu_quote import fetch_quoted_reply

    asyncio.run(fetch_quoted_reply(adapter, "om_6"))
    asyncio.run(fetch_quoted_reply(adapter, "om_6"))
    asyncio.run(fetch_quoted_reply(adapter, "om_4"))

    assert sorted(r.message_id for r in client.message_gets) == [f"om_{i}" for i in range(1, 7)]


def test_the_chain_cache_is_bounded():
    from plugins.platforms.feishu.adapter import _FEISHU_MESSAGE_TEXT_CACHE_SIZE
    from plugins.platforms.feishu.feishu_quote import fetch_node

    count = _FEISHU_MESSAGE_TEXT_CACHE_SIZE + 10
    client = ChainFeishu({f"om_{i}": [msg(f"om_{i}", {"text": str(i)})] for i in range(count)})
    adapter = make_adapter(client)

    async def fetch_all():
        for i in range(count):
            await fetch_node(adapter, f"om_{i}")

    asyncio.run(fetch_all())

    assert len(adapter._quote_node_cache) == _FEISHU_MESSAGE_TEXT_CACHE_SIZE
    assert "om_0" not in adapter._quote_node_cache and f"om_{count - 1}" in adapter._quote_node_cache


# --- quoted media ---

def test_a_text_reply_to_an_image_attaches_the_image_and_routes_it_to_vision():
    from gateway.run_inbound import GatewayInboundMixin

    client = ChainFeishu({"om_img": [msg("om_img", {"image_key": "img_chart"}, msg_type="image")]})

    _adapter, event = reply_to("om_img", client=client)

    assert [(r.message_id, r.file_key, r.type) for r in client.resource_gets] == [("om_img", "img_chart", "image")]
    assert event.message_type.value == "text" and event.text == "and now?"
    assert event.media_types == ["image/png"] and os.path.isfile(event.media_urls[0])
    assert event.reply_to_text == "[Image]"
    images, voice, audio_files, videos = GatewayInboundMixin._classify_inbound_media(event, False)
    assert (images, voice, audio_files, videos) == (event.media_urls, [], [], [])


def test_quoted_media_is_capped_and_only_the_direct_message_contributes():
    from plugins.platforms.feishu.feishu_quote import MAX_QUOTED_MEDIA

    images = [{"tag": "img", "image_key": f"img_{i}"} for i in range(MAX_QUOTED_MEDIA + 3)]
    post = {"title": "", "content": [[{"tag": "text", "text": "charts"}, *images]]}
    messages = {
        "om_1": [msg("om_1", {"image_key": "img_ancestor"}, msg_type="image")],
        "om_2": [msg("om_2", post, parent="om_1", msg_type="post")],
    }
    client = ChainFeishu(messages)

    _adapter, event = reply_to("om_2", client=client)

    keys = [r.file_key for r in client.resource_gets]
    assert keys == [f"img_{i}" for i in range(MAX_QUOTED_MEDIA)]
    assert len(event.media_urls) == len(event.media_text_inlined) == MAX_QUOTED_MEDIA


def test_a_quoted_file_is_attached_as_a_document_and_a_quoted_voice_note_is_not():
    messages = {
        "om_file": [msg("om_file", {"file_key": "file_report", "file_name": "report.pdf"}, msg_type="file")],
        "om_voice": [msg("om_voice", {"file_key": "file_voice", "duration": 3000}, msg_type="audio")],
    }
    client = ChainFeishu(messages, resource_types={"file_report": "application/pdf", "file_voice": "audio/ogg"})

    _adapter, file_event = reply_to("om_file", client=client)
    _adapter, voice_event = reply_to("om_voice", client=client)

    assert file_event.media_types == ["application/pdf"] and file_event.reply_to_text == "[Attachment: report.pdf]"
    assert voice_event.media_urls == [] and voice_event.message_type.value == "text"
    assert [r.file_key for r in client.resource_gets] == ["file_report"]


def test_a_slow_quoted_download_costs_at_most_the_budget_and_the_next_reply_reuses_it(monkeypatch):
    import plugins.platforms.feishu.feishu_quote as quote

    monkeypatch.setattr(quote, "CHAIN_BUDGET_SECONDS", 0.3)
    report = msg("om_file", {"file_key": "file_report", "file_name": "report.pdf"}, msg_type="file")
    client = ChainFeishu(
        {"om_file": [report]}, slow={"file_report"}, slow_delay=10, resource_types={"file_report": "application/pdf"},
    )
    adapter = make_adapter(client)

    async def two_replies():
        loop = asyncio.get_running_loop()
        start = loop.time()
        first = await quote.fetch_quoted_reply(adapter, "om_file")
        elapsed = loop.time() - start
        client.released.set()  # the abandoned download now finishes in the background
        await asyncio.gather(*adapter._quote_background_fetches)
        return first, elapsed, await quote.fetch_quoted_reply(adapter, "om_file")

    first, elapsed, second = asyncio.run(two_replies())

    assert elapsed < 3  # stopped at the 0.3s budget, not after the 10s download
    assert (first.text, first.media_urls) == ("[Attachment: report.pdf]", [])
    assert second.media_types == ["application/pdf"] and os.path.isfile(second.media_urls[0])
    assert len(client.resource_gets) == 1


def test_replies_quoting_the_same_image_share_one_download_until_its_file_is_gone():
    from plugins.platforms.feishu.feishu_quote import fetch_quoted_reply

    client = ChainFeishu({"om_img": [msg("om_img", {"image_key": "img_chart"}, msg_type="image")]})
    adapter = make_adapter(client)

    first = asyncio.run(fetch_quoted_reply(adapter, "om_img"))
    second = asyncio.run(fetch_quoted_reply(adapter, "om_img"))
    assert second.media_urls == first.media_urls and len(client.resource_gets) == 1

    os.remove(first.media_urls[0])  # the gateway's media cache cleanup removed it
    third = asyncio.run(fetch_quoted_reply(adapter, "om_img"))

    assert len(client.resource_gets) == 2 and os.path.isfile(third.media_urls[0])


def test_quoted_media_follows_the_reply_into_a_text_batch():
    from unittest.mock import AsyncMock

    client = ChainFeishu({"om_img": [msg("om_img", {"image_key": "img_chart"}, msg_type="image")]})
    adapter = make_adapter(client)
    adapter.get_chat_info = AsyncMock(return_value={"name": "DM"})
    adapter.handle_message = AsyncMock()
    adapter._text_batch_delay_seconds = adapter._text_batch_split_delay_seconds = 600
    message = inbound_message("om_new", "text", json.dumps({"text": "what is this?"}), parent_id="om_img")
    message.chat_type = "p2p"
    sender = SimpleNamespace(sender_type="user", sender_id=SimpleNamespace(open_id="ou_alice", user_id=None, union_id=None))

    async def deliver():
        await adapter._handle_message_event_data(SimpleNamespace(event=SimpleNamespace(message=message, sender=sender)))
        for task in adapter._pending_text_batch_tasks.values():
            task.cancel()
        for key in list(adapter._pending_text_batches):
            await adapter._flush_text_batch_now(key)

    asyncio.run(deliver())

    (call,) = adapter.handle_message.await_args_list
    event = call.args[0]
    assert (event.text, event.media_types, event.reply_to_text) == ("what is this?", ["image/png"], "[Image]")


async def dispatched_event_async(adapter, message):
    from unittest.mock import AsyncMock, Mock

    adapter.get_chat_info = AsyncMock(return_value={"name": "DM"})
    adapter._resolve_sender_profile = AsyncMock(return_value={"user_id": "u1", "user_name": "Alice", "user_id_alt": None})
    adapter.build_source = Mock(return_value=SimpleNamespace(thread_id=None))
    adapter._dispatch_inbound_event = AsyncMock()
    await adapter._process_inbound_message(
        data={}, message=message, sender_id=SimpleNamespace(open_id="ou_alice", user_id=None, union_id=None),
        chat_type="p2p", message_id=message.message_id,
    )
    return adapter._dispatch_inbound_event.await_args.args[0]


# --- quoted media: only for an explicit quote, once per batch ---

def inbound(message_id, content, *, parent=None, root=None, thread=None, message_type="text"):
    message = inbound_message(message_id, message_type, json.dumps(content), parent_id=parent)
    message.root_id, message.thread_id, message.chat_type = root or parent, thread, "group"
    sender = SimpleNamespace(sender_type="user", sender_id=SimpleNamespace(open_id="ou_alice", user_id=None, union_id=None))
    return SimpleNamespace(event=SimpleNamespace(message=message, sender=sender))


def delivered_events(adapter, *events):
    """Deliver ``events`` concurrently, in arrival order, through the real inbound path; return the
    agent turns. Batch timers are parked and the pending batches flushed once all are in."""
    from unittest.mock import AsyncMock

    adapter._admit = lambda sender, message: None  # group admission is not under test here
    adapter.get_chat_info = AsyncMock(return_value={"name": "Group", "type": "group"})
    adapter.handle_message = AsyncMock()
    adapter._text_batch_delay_seconds = adapter._text_batch_split_delay_seconds = 600
    adapter._media_batch_delay_seconds = 600

    async def deliver():
        await asyncio.gather(*(adapter._handle_message_event_data(event) for event in events))
        for task in [*adapter._pending_text_batch_tasks.values(), *adapter._pending_media_batch_tasks.values()]:
            task.cancel()
        for key in list(adapter._pending_text_batches):
            await adapter._flush_text_batch_now(key)
        for key in list(adapter._pending_media_batches):
            await adapter._flush_media_batch_now(key)

    asyncio.run(deliver())
    return [call.args[0] for call in adapter.handle_message.await_args_list]


@pytest.mark.parametrize("parent", ["om_root", None], ids=["parent-is-topic-root", "root-id-only"])
def test_messages_posted_in_a_topic_do_not_attach_the_topics_root_media(parent):
    """A topic post names the topic's root as parent (or carries only ``root_id``) without quoting it."""
    client = ChainFeishu({"om_root": [msg("om_root", {"image_key": "img_root"}, msg_type="image")]})
    adapter = make_adapter(client)

    turns = delivered_events(
        adapter,
        inbound("om_t1", {"text": "first"}, parent=parent, root="om_root", thread="omt_topic"),
        inbound("om_t2", {"text": "second"}, parent=parent, root="om_root", thread="omt_topic"),
    )

    assert [(turn.text, turn.reply_to_text, turn.media_urls) for turn in turns] == [("first\nsecond", "[Image]", [])]
    assert client.resource_gets == []


def test_an_explicit_quote_inside_a_topic_still_attaches_the_quoted_media():
    messages = {
        "om_root": [msg("om_root", {"text": "topic"})],
        "om_img": [msg("om_img", {"image_key": "img_chart"}, parent="om_root", msg_type="image")],
    }
    client = ChainFeishu(messages)
    adapter = make_adapter(client)

    (turn,) = delivered_events(adapter, inbound("om_t1", {"text": "this one?"}, parent="om_img", root="om_root", thread="omt_topic"))

    assert turn.media_types == ["image/png"] and [r.file_key for r in client.resource_gets] == ["img_chart"]


@pytest.mark.parametrize(
    ("message_type", "content", "own_media"),
    [("text", {"text": "look"}, []), ("image", {"image_key": "img_own"}, ["image/png"])],
    ids=["text-batch", "photo-batch"],
)
def test_replies_batched_together_carry_the_quoted_media_once(message_type, content, own_media):
    client = ChainFeishu({"om_img": [msg("om_img", {"image_key": "img_chart"}, msg_type="image")]})
    adapter = make_adapter(client)

    (turn,) = delivered_events(
        adapter,
        inbound("om_r1", content, parent="om_img", message_type=message_type),
        inbound("om_r2", content, parent="om_img", message_type=message_type),
    )

    quoted = [url for url in turn.media_urls if turn.media_urls.count(url) > 1]
    assert quoted == [] and len(turn.media_urls) == len(turn.media_types) == len(turn.media_text_inlined)
    assert turn.media_types == own_media + ["image/png"] + own_media
    assert [r.file_key for r in client.resource_gets].count("img_chart") == 1
