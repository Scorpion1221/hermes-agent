"""Clicks on the Feishu bot's own cards, routed to the agent as replies to the card, in its chat and topic."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import _reply_anchor_for_event
from gateway.platforms.event import MessageEvent, MessageType
from tests.gateway.test_feishu_card_operator import _LoopThread
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


def test_each_click_is_its_own_inbound_message(tmp_path):
    """The gateway dedupes the transcript on ``message_id``: a click's id names the click, not the card."""
    from hermes_state import SessionDB

    adapter = _adapter()
    routed = _wire(adapter, _bot_message())

    for token in ("tok_1", "tok_2"):
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


# --- Who may answer a card, and the answered card ---


_DSL = {"schema": "2.0", "header": {"title": {"tag": "plain_text", "content": "Pick a slot"}}, "body": {"elements": [
    {"tag": "markdown", "content": "Which slot?"},
    {"tag": "column_set", "columns": [{"tag": "column", "elements": [
        {"tag": "button", "text": {"tag": "plain_text", "content": "9:00"}, "type": "primary_filled"},
        {"tag": "button", "text": {"tag": "plain_text", "content": "Docs"}, "url": "https://example.com"},
    ]}]},
    {"tag": "overflow", "options": []},
    {"tag": "select_static", "options": []},
]}}


class _Toast:
    def __init__(self):
        self.type = self.content = None


class _Response:
    def __init__(self):
        self.card = self.toast = None


@pytest.fixture
def callback_types(monkeypatch):
    import plugins.platforms.feishu.adapter as feishu_module

    monkeypatch.setattr(feishu_module, "P2CardActionTriggerResponse", _Response)
    monkeypatch.setattr(feishu_module, "CallBackToast", _Toast)


def _user_message(open_id="ou_asker"):
    return SimpleNamespace(
        message_id="om_question", msg_type="text", parent_id="", mentions=None,
        body=SimpleNamespace(content=json.dumps({"text": "Book me a room"})),
        sender=SimpleNamespace(sender_type="user", id=open_id, id_type="open_id"),
    )


def _clickable(*, admins=(), chat=GROUP_CHAT, parent=_user_message(), card=None, others=(), allowed=""):
    """An adapter whose card ``om_bot`` (sent just now) replies to ``parent``; PATCH and the agent hand-off are
    recorded. ``card`` replaces the card; ``others`` are more messages GET can read."""
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = FeishuAdapter(PlatformConfig(extra={"admins": list(admins)}))
    adapter._app_id = "cli_self_app"
    adapter._allowed_group_users = {allowed} if allowed else set()
    if card is None:
        card = _bot_message(parent_id="om_question" if parent else "")
    card.create_time = str(int(time.time() * 1000))
    adapter._client = _client(msg=card, others=[*([parent] if parent else ()), *others], chat=chat)
    get = adapter._client.im.v1.message.get.side_effect

    def _get(request):  # the answered-card redraw reads the card as its sender wrote it
        if getattr(request, "card_msg_content_type", None) == "user_card_content":
            return SimpleNamespace(success=lambda: True, data=SimpleNamespace(
                items=[SimpleNamespace(body=SimpleNamespace(content=json.dumps(_DSL)))]))
        return get(request)

    adapter._client.im.v1.message.get = Mock(side_effect=_get)
    adapter._client.im.v1.message.patch = Mock(return_value=SimpleNamespace(success=lambda: True))
    adapter._sender_name_cache["ou_asker"] = ("Asker", time.time() + 600)
    adapter._sender_name_cache["ou_human"] = ("Human", time.time() + 600)
    adapter.routed = []

    async def _capture(event):
        adapter.routed.append(event)

    adapter._handle_message_with_guards = _capture
    return adapter


def _turn(adapter, open_id, *, thread_id=None, chat_id="oc_chat"):
    """A turn for ``open_id``'s message in that chat (and topic) is running: what the processing hook records."""
    source = SimpleNamespace(chat_id=chat_id, thread_id=thread_id, user_id=open_id, user_id_alt=None, auth_aliases=())
    adapter._add_reaction = AsyncMock(return_value=None)  # the turn's badge
    asyncio.run(adapter.on_processing_start(SimpleNamespace(message_id=f"om_{open_id}", source=source)))


def _press(adapter, token="tok_1", open_id="ou_asker", value=None):
    """Click ``om_bot`` from the SDK thread against a live adapter loop; returns the callback response."""
    data = _click(token, value=value)
    data.event.operator = SimpleNamespace(open_id=open_id, user_id=None, union_id=None)
    with _LoopThread() as loop:
        adapter._loop = loop
        return adapter._on_card_action_trigger(data)


def _patched_card(adapter):
    [call] = adapter._client.im.v1.message.patch.call_args_list
    request = call.args[0]
    assert request.message_id == "om_bot"
    return json.loads(request.request_body.content)


def test_the_user_the_card_answers_gets_a_toast_and_the_card_is_marked_answered(callback_types):
    adapter = _clickable()

    response = _press(adapter, open_id="ou_asker")

    assert response.toast.type == "success"
    [event] = adapter.routed
    assert event.text == 'card:button {"value": {"choice": "9:00"}}'
    card = _patched_card(adapter)
    elements = card["body"]["elements"]
    buttons = elements[1]["columns"][0]["elements"]
    assert buttons[0]["disabled"] is True  # the callback button takes no second answer
    assert "disabled" not in buttons[1]  # a link button still opens its link
    assert not any(e.get("tag") == "overflow" for e in elements)  # overflow can't be disabled: it goes
    assert next(e for e in elements if e.get("tag") == "select_static")["disabled"] is True
    assert "Asker" in elements[-1]["content"]


def test_someone_else_in_the_group_is_refused_and_the_card_stays_as_it_was(callback_types):
    adapter = _clickable()

    response = _press(adapter, open_id="ou_stranger")

    assert response.toast.type == "error"
    assert adapter.routed == []
    adapter._client.im.v1.message.patch.assert_not_called()


def test_an_admin_may_answer_a_card_someone_else_asked_for(callback_types):
    adapter = _clickable(admins=["ou_boss"])

    response = _press(adapter, open_id="ou_boss")

    assert response.toast.type == "success"
    assert len(adapter.routed) == 1


_OPEN = {"swp": "approve", "hermes_card_click": "anyone"}


def test_anyone_in_the_group_may_click_a_control_opened_to_everyone(callback_types):
    adapter = _clickable(admins=["ou_boss"])

    response = _press(adapter, open_id="ou_stranger", value=_OPEN)

    assert response.toast.type == "success"
    [event] = adapter.routed
    assert event.source.user_id == "ou_stranger"  # the agent hears it from the clicker
    assert '"hermes_card_click": "anyone"' in event.text
    adapter._client.im.v1.message.patch.assert_called_once()


def test_a_card_opened_to_everyone_is_still_answered_once(callback_types):
    adapter = _clickable()

    _press(adapter, token="tok_1", open_id="ou_stranger", value=_OPEN)
    second = _press(adapter, token="tok_2", open_id="ou_someone_else", value=_OPEN)

    assert second.toast.type == "info"
    assert len(adapter.routed) == 1


def test_another_value_does_not_open_the_click(callback_types):
    adapter = _clickable()

    response = _press(adapter, open_id="ou_stranger", value={"hermes_card_click": "admins"})

    assert response.toast.type == "error"
    assert adapter.routed == []


def test_anyone_in_a_dm_may_answer(callback_types):
    adapter = _clickable(admins=["ou_boss"], chat=DM_CHAT, parent=None)

    response = _press(adapter, open_id="ou_human")

    assert response.toast.type == "success"
    assert len(adapter.routed) == 1


def test_a_card_that_answers_nobody_in_a_group_is_for_admins(callback_types):
    adapter = _clickable(admins=["ou_boss"], parent=None)

    assert _press(adapter, open_id="ou_asker").toast.type == "error"
    assert _press(adapter, token="tok_2", open_id="ou_boss").toast.type == "success"


def test_a_redelivered_click_is_dropped(callback_types):
    adapter = _clickable()

    _press(adapter, token="tok_1")
    redelivered = _press(adapter, token="tok_1")

    assert redelivered.toast is None
    assert len(adapter.routed) == 1


def test_a_card_is_answered_once(callback_types):
    adapter = _clickable()

    first = _press(adapter, token="tok_1", open_id="ou_asker")
    second = _press(adapter, token="tok_2", open_id="ou_asker")

    assert (first.toast.type, second.toast.type) == ("success", "info")
    assert len(adapter.routed) == 1
    assert adapter._client.im.v1.message.patch.call_count == 1


def test_the_decision_waits_on_the_adapter_loop_not_the_sdk_thread(callback_types):
    """Over the websocket the callback runs on the SDK's thread; the lookups run on the adapter loop."""
    adapter = _clickable()
    threads = []
    get = adapter._client.im.v1.message.get.side_effect

    def _get(request):
        threads.append(threading.current_thread())
        return get(request)

    adapter._client.im.v1.message.get = Mock(side_effect=_get)
    adapter._run_blocking = AsyncMock(side_effect=lambda fn, *a, **k: fn(*a, **k))

    _press(adapter)

    assert threads and threading.main_thread() not in threads


def test_answered_card_without_schema_2_drops_its_button_rows():
    from plugins.platforms.feishu.feishu_card_action import answered_card

    card = {"elements": [{"tag": "div", "text": {"tag": "plain_text", "content": "Q"}},
                         {"tag": "action", "actions": [{"tag": "button"}]}]}

    answered = answered_card(card, "✅ Handled by Human")

    assert [e["tag"] for e in answered["elements"]] == ["div", "markdown"]
    assert card["elements"][1]["tag"] == "action"  # the input is not mutated


def _topic_card(parent_id="om_root"):
    card = _bot_message(thread_id="omt_topic", parent_id=parent_id)
    card.root_id = "om_root"
    return card


def _topic_root(open_id="ou_founder"):
    root = _user_message(open_id)
    root.message_id = "om_root"
    return root


def test_in_a_topic_the_card_answers_whoever_asked_not_who_opened_the_topic(callback_types):
    """In a topic the card replies to the topic's root, whoever asked: the running turn says who did."""
    adapter = _clickable(admins=["ou_boss"], parent=_topic_root(), card=_topic_card())
    _turn(adapter, "ou_asker", thread_id="omt_topic")

    assert _press(adapter, open_id="ou_founder").toast.type == "error"
    assert _press(adapter, token="tok_2", open_id="ou_asker").toast.type == "success"


def test_a_topic_card_no_turn_explains_is_for_admins(callback_types):
    adapter = _clickable(admins=["ou_boss"], parent=_topic_root(), card=_topic_card())

    assert _press(adapter, open_id="ou_founder").toast.type == "error"
    assert _press(adapter, token="tok_2", open_id="ou_boss").toast.type == "success"


def test_another_users_turn_in_the_chat_does_not_own_the_card(callback_types):
    adapter = _clickable()
    _turn(adapter, "ou_other", thread_id="omt_elsewhere")

    assert _press(adapter, open_id="ou_asker").toast.type == "success"  # its reply chain still names the asker


def test_a_follow_up_card_answers_the_user_the_first_one_did(callback_types):
    """A card answering a click replies to the bot's earlier card: the asker is found above it."""
    first = _bot_message(parent_id="om_question")
    first.message_id = "om_first"
    adapter = _clickable(admins=["ou_boss"], card=_bot_message(parent_id="om_first"), others=[first])

    assert _press(adapter, open_id="ou_stranger").toast.type == "error"
    assert _press(adapter, token="tok_2", open_id="ou_asker").toast.type == "success"


def test_a_card_whose_chat_and_owner_cannot_be_read_asks_again_later_without_routing(callback_types):
    adapter = _clickable(admins=["ou_boss"])
    adapter._client.im.v1.message.get = Mock(side_effect=RuntimeError("network down"))
    adapter._client.im.v1.chat.get = Mock(side_effect=RuntimeError("network down"))

    first = _press(adapter, open_id="ou_asker")
    second = _press(adapter, token="tok_2", open_id="ou_asker")

    assert (first.toast.type, second.toast.type) == ("warning", "warning")
    assert adapter._client.im.v1.chat.get.call_count == 1  # the failure is not retried on every click
    assert adapter.routed == []


def test_over_the_webhook_a_refused_click_is_not_answered(callback_types):
    """The webhook transport calls back on the adapter loop: the decision is made there, afterwards."""
    adapter = _clickable()

    async def _press_on_loop(open_id, token):
        adapter._loop = asyncio.get_running_loop()
        data = _click(token)
        data.event.operator = SimpleNamespace(open_id=open_id, user_id=None, union_id=None)
        adapter._on_card_action_trigger(data)
        await asyncio.sleep(0)
        while tasks := {t for t in asyncio.all_tasks() if t is not asyncio.current_task()}:
            await asyncio.wait(tasks)

    asyncio.run(_press_on_loop("ou_stranger", "tok_1"))
    assert adapter.routed == []
    adapter._client.im.v1.message.patch.assert_not_called()

    asyncio.run(_press_on_loop("ou_asker", "tok_2"))
    assert len(adapter.routed) == 1
    assert adapter._client.im.v1.message.patch.call_count == 1

