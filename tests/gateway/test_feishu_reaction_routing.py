"""User reactions on the Feishu bot's own messages, routed as synthetic replies to that message."""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import _reply_anchor_for_event
from tests.gateway.test_feishu_fetched_content import json_card, markdown, text

_SELF_APP = "cli_self_app"
_BOT_CARD = json_card([markdown(text("Shall I book the "), text("9:00", "bold"), text(" slot?"))])


def _adapter():
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = FeishuAdapter(PlatformConfig())
    adapter._app_id = _SELF_APP
    return adapter


# chat.get as Feishu answers it (live-probed): ``chat_mode`` is p2p / group / topic, while ``chat_type`` is
# a group's visibility (private / public) and is absent for a DM.
GROUP_CHAT = {"chat_mode": "group", "chat_type": "private", "name": "Ops"}
DM_CHAT = {"chat_mode": "p2p"}


def _client(*, msg=None, reply=None, others=(), chat=GROUP_CHAT):
    """``GET im/v1/messages/{id}`` answers from ``msg`` and ``others`` by id; ``GET im/v1/chats/{id}`` with ``chat``."""
    messages = {m.message_id: m for m in (msg, *others) if m}

    def _get(request):
        found = messages.get(request.message_id)
        return SimpleNamespace(success=lambda: True, data=SimpleNamespace(items=[found] if found else []))

    def _get_chat(request):
        fields = {"chat_mode": None, "chat_type": None, "name": None, **chat}
        return SimpleNamespace(success=lambda: True, data=SimpleNamespace(**fields))

    reply = reply or Mock(return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_reply")))
    return SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(
        message=SimpleNamespace(get=Mock(side_effect=_get), reply=reply),
        message_resource=SimpleNamespace(get=Mock()),
        chat=SimpleNamespace(get=Mock(side_effect=_get_chat)),
    )))


def _bot_message(*, thread_id=None, sender_app=_SELF_APP, msg_type="interactive", content=_BOT_CARD, parent_id=""):
    """A message this bot sent (a Card JSON 2.0 reply by default), as ``GET im/v1/messages`` reports it: the
    item names its chat and topic but carries no chat type."""
    return SimpleNamespace(
        message_id="om_bot", chat_id="oc_chat", thread_id=thread_id, msg_type=msg_type,
        parent_id=parent_id, mentions=None, body=SimpleNamespace(content=json.dumps(content)),
        sender=SimpleNamespace(sender_type="app", id=sender_app, id_type="app_id"),
    )


def _reaction(event_id, *, emoji="THUMBSUP", message_id="om_bot"):
    return SimpleNamespace(
        header=SimpleNamespace(event_id=event_id),
        event=SimpleNamespace(
            message_id=message_id, operator_type="user", reaction_type=SimpleNamespace(emoji_type=emoji),
            user_id=SimpleNamespace(open_id="ou_human", user_id=None, union_id=None),
        ),
    )


def _wire_routing(adapter, msg, *, others=(), chat=GROUP_CHAT):
    adapter._client = _client(msg=msg, others=others, chat=chat)
    adapter._resolve_sender_profile = AsyncMock(
        return_value={"user_id": "ou_human", "user_name": "Human", "user_id_alt": None})
    routed = []

    async def _capture(event):
        routed.append(event)

    adapter._handle_message_with_guards = _capture
    return routed


def _route(adapter, data, event_type="im.message.reaction.created_v1"):
    asyncio.run(adapter._handle_reaction_event(event_type, data))


def test_reaction_on_own_message_replies_to_it_with_quote_and_topic():
    adapter = _adapter()
    routed = _wire_routing(adapter, _bot_message(thread_id="omt_topic"))

    _route(adapter, _reaction("evt_1", emoji="DONE"))

    [event] = routed
    assert event.text == "reaction:added:DONE"
    assert event.reply_to_message_id == "om_bot"
    assert event.reply_to_text == "Shall I book the **9:00** slot?"  # the card's content, not "please upgrade"
    assert event.reply_to_is_own_message is True
    assert event.source.thread_id == "omt_topic"  # the topic session, not the chat's main session
    quote_get = adapter._client.im.v1.message.get.call_args_list[-1].args[0]
    assert (quote_get.message_id, quote_get.card_msg_content_type) == ("om_bot", "raw_card_content")


def test_a_reaction_asks_for_no_answer():
    adapter = _adapter()
    routed = _wire_routing(adapter, _bot_message())

    _route(adapter, _reaction("evt_1", emoji="OK"))

    [event] = routed
    assert event.reply_expected is False


def test_a_silent_answer_to_a_reaction_stays_silent():
    from gateway.response_filters import silence_allowed

    adapter = _adapter()
    routed = _wire_routing(adapter, _bot_message())
    _route(adapter, _reaction("evt_1", emoji="OK"))

    [event] = routed
    assert silence_allowed(None, event.reply_expected)  # a human turn, yet no "only a silence marker" notice


def _topic_message_session_key(adapter, *, thread_id):
    """Session key of an ordinary inbound message from the same user in that chat and topic."""
    message = SimpleNamespace(
        message_id="om_user", message_type="text", content=json.dumps({"text": "hi"}), mentions=None,
        chat_id="oc_chat", chat_type="group", thread_id=thread_id, parent_id=None, upper_message_id=None,
        root_id=None,
    )
    captured = []

    async def _capture(event):
        captured.append(event)

    adapter._dispatch_inbound_event = _capture
    asyncio.run(adapter._process_inbound_message(
        data=None, message=message, sender_id=SimpleNamespace(open_id="ou_human", user_id=None, union_id=None),
        chat_type="group", message_id="om_user",
    ))
    return adapter._event_session_key(captured[0])


@pytest.mark.parametrize("thread_id", [None, "omt_topic"])
def test_reaction_in_a_group_joins_the_session_of_that_chat_or_topic(thread_id):
    adapter = _adapter()
    routed = _wire_routing(adapter, _bot_message(thread_id=thread_id))

    _route(adapter, _reaction("evt_1"))

    [event] = routed
    assert event.source.chat_type == "group"
    assert adapter._event_session_key(event) == _topic_message_session_key(adapter, thread_id=thread_id)


def test_reaction_in_a_dm_stays_in_the_dm_session():
    adapter = _adapter()
    routed = _wire_routing(adapter, _bot_message(), chat=DM_CHAT)

    _route(adapter, _reaction("evt_1"))

    [event] = routed
    assert event.source.chat_type == "dm"


def test_reaction_on_a_bot_reply_quotes_the_reply_chain_above_it():
    adapter = _adapter()
    adapter._sender_name_cache["ou_human"] = ("Human", time.time() + 600)
    question = SimpleNamespace(
        message_id="om_question", msg_type="text", parent_id="", mentions=None,
        body=SimpleNamespace(content=json.dumps({"text": "Can you find a room for Monday?"})),
        sender=SimpleNamespace(sender_type="user", id="ou_human", id_type="open_id"),
    )
    routed = _wire_routing(adapter, _bot_message(parent_id="om_question"), others=[question])

    _route(adapter, _reaction("evt_1"))

    [event] = routed
    assert event.reply_to_text.splitlines() == [
        "Earlier messages in this reply chain, oldest first:",
        "- Human (text): Can you find a room for Monday?",
        "Directly quoted message (card, from You):",
        "Shall I book the **9:00** slot?",
    ]


def test_reaction_on_a_bot_image_quotes_it_without_downloading_it():
    """The synthetic event carries no media, so the reacted message's files are not fetched."""
    adapter = _adapter()
    routed = _wire_routing(adapter, _bot_message(msg_type="image", content={"image_key": "img_chart"}))

    _route(adapter, _reaction("evt_1"))

    [event] = routed
    assert event.reply_to_text == "[Image]"
    assert event.media_urls == []
    adapter._client.im.v1.message_resource.get.assert_not_called()


def test_a_reaction_fetches_the_reacted_message_once():
    adapter = _adapter()
    routed = _wire_routing(adapter, _bot_message())

    _route(adapter, _reaction("evt_1"))

    [event] = routed
    assert event.reply_to_text == "Shall I book the **9:00** slot?"
    gets = [call.args[0] for call in adapter._client.im.v1.message.get.call_args_list]
    assert [(r.message_id, getattr(r, "card_msg_content_type", None)) for r in gets] == [("om_bot", "raw_card_content")]


def test_removed_reaction_routes_as_removed():
    adapter = _adapter()
    routed = _wire_routing(adapter, _bot_message())

    _route(adapter, _reaction("evt_1"), event_type="im.message.reaction.deleted_v1")

    assert [e.text for e in routed] == ["reaction:removed:THUMBSUP"]
    assert routed[0].source.thread_id is None


def test_reaction_on_peer_bot_message_is_not_routed():
    adapter = _adapter()
    routed = _wire_routing(adapter, _bot_message(sender_app="cli_peer_app"))

    _route(adapter, _reaction("evt_1"))

    assert routed == []
    assert adapter._client.im.v1.message.get.call_count == 1  # the sender check only; nothing quoted


def test_each_reaction_is_its_own_inbound_message_and_a_redelivery_is_not():
    """The gateway keys transcript dedupe, input ownership and the delivery ledger on ``message_id``:
    a second reaction on one message needs its own id, a redelivered event must keep its id."""
    adapter = _adapter()
    routed = _wire_routing(adapter, _bot_message())

    _route(adapter, _reaction("evt_1"))
    _route(adapter, _reaction("evt_2"))
    _route(adapter, _reaction("evt_1"))

    first, second, redelivered = (e.message_id for e in routed)
    assert first != second
    assert first == redelivered
    assert "om_bot" not in {first, second}


def test_second_reaction_is_not_skipped_by_transcript_platform_id_dedupe(tmp_path):
    """Real SessionDB: the gateway skips a user row whose platform id is already in the transcript."""
    from hermes_state import SessionDB

    adapter = _adapter()
    routed = _wire_routing(adapter, _bot_message())
    _route(adapter, _reaction("evt_1"))
    _route(adapter, _reaction("evt_2"))
    first, second = routed

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("sess", "feishu")
        db.append_message("sess", "user", first.text, platform_message_id=str(first.message_id))
        assert db.has_platform_message_id("sess", str(first.message_id))
        assert not db.has_platform_message_id("sess", str(second.message_id))
    finally:
        db.close()


@pytest.mark.parametrize("thread_id", [None, "omt_topic"])
def test_reaction_reply_is_sent_as_a_reply_to_the_reacted_message(thread_id):
    adapter = _adapter()
    routed = _wire_routing(adapter, _bot_message(thread_id=thread_id))
    _route(adapter, _reaction("evt_1"))
    [event] = routed
    reply = Mock(return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_answer")))
    adapter._client = _client(reply=reply)

    result = asyncio.run(adapter.send("oc_chat", "Booked.", reply_to=_reply_anchor_for_event(event),
                                      metadata={"thread_id": thread_id} if thread_id else None))

    assert result.success
    request = reply.call_args.args[0]
    assert request.message_id == "om_bot"
    assert request.request_body.reply_in_thread is bool(thread_id)


def test_busy_ack_anchored_on_a_reaction_id_replies_to_the_reacted_message():
    """Busy acks reply to ``event.message_id`` directly (run_busy._busy_reply_to)."""
    adapter = _adapter()
    routed = _wire_routing(adapter, _bot_message())
    _route(adapter, _reaction("evt_1"))
    reply = Mock(return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_ack")))
    adapter._client = _client(reply=reply)

    asyncio.run(adapter.send("oc_chat", "Still working on it", reply_to=routed[0].message_id))

    assert reply.call_args.args[0].message_id == "om_bot"


def test_processing_badge_goes_on_the_reacted_message():
    adapter = _adapter()
    routed = _wire_routing(adapter, _bot_message())
    _route(adapter, _reaction("evt_1"))
    adapter._add_reaction = AsyncMock(return_value="r_typing")
    adapter._remove_reaction = AsyncMock(return_value=True)

    asyncio.run(adapter.on_processing_start(routed[0]))
    asyncio.run(adapter.on_processing_complete(routed[0], outcome=None))

    adapter._add_reaction.assert_awaited_once()
    assert adapter._add_reaction.await_args.args[0] == "om_bot"
    adapter._remove_reaction.assert_awaited_once_with("om_bot", "r_typing")


@pytest.mark.asyncio
async def test_same_reply_to_two_reactions_gets_two_delivery_ledger_rows():
    """Real delivery ledger: the obligation id is keyed on the inbound id, so identical replies to two
    reactions on one message must not collapse into one row (a refused reply would read as delivered)."""
    adapter = _adapter()
    routed = _wire_routing(adapter, _bot_message())
    for event_id in ("evt_1", "evt_2"):
        await adapter._handle_reaction_event("im.message.reaction.created_v1", _reaction(event_id))
    session_key = adapter._event_session_key(routed[0])

    ids = [await adapter._record_delivery_obligation(event, session_key, "Noted.", adapter, False)
           for event in routed]

    assert None not in ids
    assert ids[0] != ids[1]


def test_ordinary_message_ids_pass_through_unchanged():
    from plugins.platforms.feishu.feishu_reaction import feishu_message_id

    assert feishu_message_id("om_plain") == "om_plain"
    assert feishu_message_id(None) is None
    assert feishu_message_id("om_card:card:tok") == "om_card"  # a card click names its card
    assert feishu_message_id(":card:tok") == ""  # no card: no reply target
