"""User reactions on the Feishu bot's own messages, routed as synthetic replies to that message."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import _reply_anchor_for_event
from gateway.platforms.event import MessageEvent, MessageType

_SELF_APP = "cli_self_app"


def _adapter():
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = FeishuAdapter(PlatformConfig())
    adapter._app_id = _SELF_APP
    return adapter


def _client(*, msg=None, reply=None):
    get = Mock(return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace(items=[msg] if msg else [])))
    reply = reply or Mock(return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_reply")))
    return SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(get=get, reply=reply))))


def _bot_message(*, thread_id=None, sender_app=_SELF_APP):
    return SimpleNamespace(
        message_id="om_bot", chat_id="oc_chat", chat_type="group", thread_id=thread_id,
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


def _wire_routing(adapter, msg):
    adapter._client = _client(msg=msg)
    adapter._fetch_message_text = AsyncMock(return_value="Shall I book the 9:00 slot?")
    adapter._resolve_sender_profile = AsyncMock(
        return_value={"user_id": "ou_human", "user_name": "Human", "user_id_alt": None})
    adapter.get_chat_info = AsyncMock(return_value={"name": "Ops", "type": "group"})
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
    assert event.reply_to_text == "Shall I book the 9:00 slot?"
    assert event.reply_to_is_own_message is True
    assert event.source.thread_id == "omt_topic"  # the topic session, not the chat's main session
    adapter._fetch_message_text.assert_awaited_once_with("om_bot")


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
    adapter._fetch_message_text.assert_not_awaited()


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


def test_synthetic_card_command_events_keep_their_shape():
    """Non-approval card clicks still route as top-level synthetic commands."""
    adapter = _adapter()
    routed = _wire_routing(adapter, _bot_message())
    data = SimpleNamespace(event=SimpleNamespace(
        token="tok_cmd", context=SimpleNamespace(open_chat_id="oc_chat"),
        operator=SimpleNamespace(open_id="ou_clicker"), action=SimpleNamespace(tag="button", value={"x": 1}),
    ))

    asyncio.run(adapter._handle_card_action_event(data))

    [event] = routed
    assert isinstance(event, MessageEvent) and event.message_type is MessageType.COMMAND
    assert event.source.thread_id is None and event.reply_to_message_id is None
