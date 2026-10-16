"""Sends into a Feishu topic without a reply anchor still land in the topic.

Feishu's create API takes no ``thread_id`` receive_id_type (it answers 99992402), so a message reaches a
topic only as a reply to a message in it. Every send that carries just the topic (platform notices, /goal
status, stall warnings, command acks, a background job's completion, ``hermes send`` into a topic) must
therefore reply inside the topic instead of creating a message addressed to it.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import _reply_anchor_for_event
from gateway.session_context import clear_session_vars, set_session_vars


class _ListMessageRequest:
    """The SDK's builder shape, so the topic lookup runs whether or not lark-oapi is installed."""

    @staticmethod
    def builder():
        fields = {}

        class _Builder:
            def __getattr__(self, name):
                def _set(value):
                    fields[name] = value
                    return self
                return _set

            def build(self):
                return SimpleNamespace(**fields)
        return _Builder()


@pytest.fixture(autouse=True)
def _sdk_list_request(monkeypatch):
    monkeypatch.setitem(sys.modules, "lark_oapi.api.im.v1", SimpleNamespace(ListMessageRequest=_ListMessageRequest))


def _ok(message_id):
    return SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id=message_id))


def _client(*, topic_messages=("om_in_topic",)):
    """``im/v1/messages``: list answers ``topic_messages`` for a topic; reply and create succeed."""
    listed = SimpleNamespace(success=lambda: True,
                             data=SimpleNamespace(items=[SimpleNamespace(message_id=m) for m in topic_messages]))
    return SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(
        list=Mock(return_value=listed), reply=Mock(return_value=_ok("om_reply")),
        create=Mock(return_value=_ok("om_created")),
    ))))


def _adapter(client):
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = FeishuAdapter(PlatformConfig())
    adapter._client = client
    return adapter


def _assert_replied_in_topic(client, anchor):
    client.im.v1.message.create.assert_not_called()
    request = client.im.v1.message.reply.call_args.args[0]
    assert request.message_id == anchor
    assert request.request_body.reply_in_thread is True


def test_an_anchorless_send_in_a_topic_replies_in_the_topic():
    client = _client()
    adapter = _adapter(client)

    result = asyncio.run(adapter.send("oc_chat", "Build finished.", metadata={"thread_id": "omt_topic"}))

    assert result.success
    _assert_replied_in_topic(client, "om_in_topic")
    listed = client.im.v1.message.list.call_args.args[0]
    assert (listed.container_id_type, listed.container_id) == ("thread", "omt_topic")


def test_a_send_with_an_anchor_does_not_look_the_topic_up():
    client = _client()
    adapter = _adapter(client)

    asyncio.run(adapter.send("oc_chat", "Done.", reply_to="om_user", metadata={"thread_id": "omt_topic"}))

    client.im.v1.message.list.assert_not_called()
    _assert_replied_in_topic(client, "om_user")


def test_a_send_outside_a_topic_is_still_created_in_the_chat():
    client = _client()
    adapter = _adapter(client)

    asyncio.run(adapter.send("oc_chat", "Hello."))

    client.im.v1.message.list.assert_not_called()
    client.im.v1.message.reply.assert_not_called()
    request = client.im.v1.message.create.call_args.args[0]
    assert (request.receive_id_type, request.request_body.receive_id) == ("chat_id", "oc_chat")


@pytest.mark.asyncio
async def test_a_background_completion_in_a_topic_is_answered_in_the_topic(monkeypatch, tmp_path):
    """The synthetic completion carries no reply anchor (a stale trigger is not quoted, #52694); the
    gateway's reply to it still has to reach the topic the job was started from."""
    import gateway.run as gateway_run
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from tests.gateway.test_background_process_notifications import AdmittingHandler

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner = GatewayRunner(GatewayConfig())
    runner.adapters[Platform.FEISHU] = SimpleNamespace(send=AsyncMock(), handle_message=AdmittingHandler())
    session_key = "agent:main:feishu:group:oc_chat:omt_topic"
    runner.session_store._entries[session_key] = SimpleNamespace(origin=SessionSource(
        platform=Platform.FEISHU, chat_id="oc_chat", chat_type="group", thread_id="omt_topic",
        user_id="ou_human", user_name="Human", message_id="om_trigger",
    ))

    await runner._inject_watch_notification(
        "[SYSTEM: Background process finished]",
        {"session_id": "proc_build", "session_key": session_key, "message_id": "om_trigger"},
    )
    synth = runner.adapters[Platform.FEISHU].handle_message.await_args.args[0]
    anchor = _reply_anchor_for_event(synth)
    client = _client()

    result = await _adapter(client).send(
        synth.source.chat_id, "The build passed.", reply_to=anchor,
        metadata=runner._thread_metadata_for_source(synth.source, anchor),
    )

    assert result.success
    _assert_replied_in_topic(client, "om_in_topic")


def _standalone(monkeypatch, client, *, thread_id):
    from plugins.platforms.feishu import adapter as feishu

    monkeypatch.setattr(feishu, "_load_lark_oapi", lambda: True)
    monkeypatch.setattr(feishu.FeishuAdapter, "_build_lark_client", lambda self, domain: client)
    return asyncio.run(feishu._standalone_send(PlatformConfig(), "oc_chat", "Report attached.", thread_id=thread_id))


def _in_session(**fields):
    session = {"platform": "feishu", "chat_id": "oc_chat", "thread_id": "omt_topic", "message_id": "om_inbound"}
    return set_session_vars(**{**session, **fields})


def test_hermes_send_into_its_own_topic_replies_to_the_conversations_message(monkeypatch):
    """``hermes send`` in the agent's terminal sends standalone; inside its own topic it anchors on the
    message the session is answering, which needs no message-list permission."""
    client = _client(topic_messages=())
    tokens = _in_session()
    try:
        result = _standalone(monkeypatch, client, thread_id="omt_topic")
    finally:
        clear_session_vars(tokens)

    assert result["success"] is True
    client.im.v1.message.list.assert_not_called()
    _assert_replied_in_topic(client, "om_inbound")


def test_hermes_send_into_another_topic_never_borrows_the_conversations_message(monkeypatch):
    client = _client(topic_messages=("om_in_other_topic",))
    tokens = _in_session()
    try:
        _standalone(monkeypatch, client, thread_id="omt_other")
    finally:
        clear_session_vars(tokens)

    _assert_replied_in_topic(client, "om_in_other_topic")
