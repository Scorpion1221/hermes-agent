"""Clicks on the Feishu bot's own cards, routed to the agent as replies to the card, in its chat and topic."""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.platforms.base import _reply_anchor_for_event
from gateway.platforms.event import MessageEvent, MessageType
from tests.gateway.test_feishu_reaction_routing import DM_CHAT, GROUP_CHAT, _adapter, _bot_message, _client


def _wire(adapter, card=None, *, chat=GROUP_CHAT, get=None):
    adapter._client = _client(msg=card, chat=chat)
    if get is not None:
        adapter._client.im.v1.message.get = get
    adapter._sender_name_cache["ou_human"] = ("Human", time.time() + 600)
    routed = []

    async def _capture(event):
        routed.append(event)

    adapter._handle_message_with_guards = _capture
    return routed


def _click(token, *, card_id="om_bot", value=None, tag="button", **action_fields):
    context = SimpleNamespace(open_chat_id="oc_chat", open_message_id=card_id) if card_id else SimpleNamespace(
        open_chat_id="oc_chat")
    return SimpleNamespace(event=SimpleNamespace(
        token=token, context=context,
        operator=SimpleNamespace(open_id="ou_human", user_id=None, union_id="on_human"),
        action=SimpleNamespace(tag=tag, value={"choice": "9:00"} if value is None else value, **action_fields),
    ))


def _route(adapter, data):
    asyncio.run(adapter._handle_card_action_event(data))


def _typed_session_key(adapter, *, chat_type, thread_id):
    """Session key of a message the same user types in that chat (and topic)."""
    message = SimpleNamespace(
        message_id="om_user", message_type="text", content=json.dumps({"text": "hi"}), mentions=None,
        chat_id="oc_chat", chat_type=chat_type, thread_id=thread_id, parent_id=None, upper_message_id=None,
        root_id=None,
    )
    captured = []

    async def _capture(event):
        captured.append(event)

    adapter._dispatch_inbound_event = _capture
    asyncio.run(adapter._process_inbound_message(
        data=None, message=message, sender_id=SimpleNamespace(open_id="ou_human", user_id=None, union_id="on_human"),
        chat_type=chat_type, message_id="om_user",
    ))
    return adapter._event_session_key(captured[0])


def test_a_click_reaches_the_agent_as_a_reply_to_the_card():
    adapter = _adapter()
    routed = _wire(adapter, _bot_message(thread_id="omt_topic"))

    _route(adapter, _click("tok_1"))

    [event] = routed
    assert isinstance(event, MessageEvent) and event.message_type is MessageType.TEXT
    assert event.text == 'card:button {"value": {"choice": "9:00"}}'
    assert not event.is_command()  # never the unknown-command notice: the agent answers the click
    assert event.reply_to_message_id == "om_bot"
    assert event.reply_to_text == "Shall I book the **9:00** slot?"  # which card, and what it says
    assert event.reply_to_is_own_message is True
    assert event.source.thread_id == "omt_topic"


@pytest.mark.parametrize(("chat", "chat_type"), [(GROUP_CHAT, "group"), (DM_CHAT, "p2p")])
@pytest.mark.parametrize("thread_id", [None, "omt_topic"])
def test_a_click_joins_the_session_a_message_typed_there_joins(chat, chat_type, thread_id):
    adapter = _adapter()
    routed = _wire(adapter, _bot_message(thread_id=thread_id), chat=chat)

    _route(adapter, _click("tok_1"))

    [event] = routed
    assert adapter._event_session_key(event) == _typed_session_key(adapter, chat_type=chat_type, thread_id=thread_id)


def test_the_card_is_fetched_once_for_its_topic_and_its_quote():
    adapter = _adapter()
    _wire(adapter, _bot_message(thread_id="omt_topic"))

    _route(adapter, _click("tok_1"))

    gets = [call.args[0] for call in adapter._client.im.v1.message.get.call_args_list]
    assert [(r.message_id, r.card_msg_content_type) for r in gets] == [("om_bot", "raw_card_content")]


@pytest.mark.parametrize("thread_id", [None, "omt_topic"])
def test_the_answer_is_a_reply_to_the_card(thread_id):
    adapter = _adapter()
    routed = _wire(adapter, _bot_message(thread_id=thread_id))
    _route(adapter, _click("tok_1"))
    [event] = routed
    reply = Mock(return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_answer")))
    adapter._client = _client(reply=reply)

    result = asyncio.run(adapter.send("oc_chat", "Booked.", reply_to=_reply_anchor_for_event(event),
                                      metadata={"thread_id": thread_id} if thread_id else None))

    assert result.success
    request = reply.call_args.args[0]
    assert request.message_id == "om_bot"
    assert request.request_body.reply_in_thread is bool(thread_id)


def test_busy_ack_and_badge_go_on_the_card():
    """Busy acks reply to ``event.message_id`` and the processing badge marks it: both must reach the card."""
    adapter = _adapter()
    routed = _wire(adapter, _bot_message())
    _route(adapter, _click("tok_1"))
    reply = Mock(return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_ack")))
    adapter._client = _client(reply=reply)
    adapter._add_reaction = AsyncMock(return_value="r_typing")
    adapter._remove_reaction = AsyncMock(return_value=True)

    asyncio.run(adapter.send("oc_chat", "Still working on it", reply_to=routed[0].message_id))
    asyncio.run(adapter.on_processing_start(routed[0]))
    asyncio.run(adapter.on_processing_complete(routed[0], outcome=None))

    assert reply.call_args.args[0].message_id == "om_bot"
    assert adapter._add_reaction.await_args.args[0] == "om_bot"
    adapter._remove_reaction.assert_awaited_once_with("om_bot", "r_typing")


def test_each_click_is_its_own_inbound_message_and_a_redelivery_is_dropped(tmp_path):
    from hermes_state import SessionDB

    adapter = _adapter()
    routed = _wire(adapter, _bot_message())

    for token in ("tok_1", "tok_2", "tok_1"):
        _route(adapter, _click(token))

    first, second = routed
    assert first.message_id != second.message_id
    assert "om_bot" not in {first.message_id, second.message_id}
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("sess", "feishu")
        db.append_message("sess", "user", first.text, platform_message_id=str(first.message_id))
        assert not db.has_platform_message_id("sess", str(second.message_id))
    finally:
        db.close()


def test_what_was_picked_or_submitted_reaches_the_agent():
    adapter = _adapter()
    routed = _wire(adapter, _bot_message())

    _route(adapter, _click("tok_1", value={}, tag="select_static", option="tuesday"))
    _route(adapter, _click("tok_2", value={}, name="book", form_value={"room": "A1", "notes": "两人"}))

    assert [e.text for e in routed] == [
        'card:select_static {"option": "tuesday"}',
        'card:button {"name": "book", "form_value": {"room": "A1", "notes": "两人"}}',
    ]


@pytest.mark.parametrize("get", [
    Mock(return_value=SimpleNamespace(success=lambda: False, code=230002, msg="no permission")),
    Mock(side_effect=RuntimeError("network down")),
])
def test_an_unreadable_card_still_routes_the_click_as_a_reply_to_it(get):
    adapter = _adapter()
    routed = _wire(adapter, get=get)

    _route(adapter, _click("tok_1"))

    [event] = routed
    assert event.source.thread_id is None
    assert event.reply_to_message_id == "om_bot"


def test_a_click_without_a_card_id_is_answered_as_a_new_message():
    adapter = _adapter()
    routed = _wire(adapter, _bot_message())
    _route(adapter, _click("tok_1", card_id=""))
    [event] = routed
    create = Mock(return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_new")))
    adapter._client.im.v1.message.create = create

    asyncio.run(adapter.send("oc_chat", "Done.", reply_to=_reply_anchor_for_event(event)))

    adapter._client.im.v1.message.get.assert_not_called()
    assert event.reply_to_message_id is None and event.source.thread_id is None
    assert create.call_args.args[0].request_body.receive_id == "oc_chat"
    adapter._client.im.v1.message.reply.assert_not_called()
