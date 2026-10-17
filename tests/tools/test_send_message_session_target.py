"""A bare ``platform`` target means the conversation the sender is in, when it is in one.

``hermes send --to feishu`` run from the agent's terminal inherits the turn's HERMES_SESSION_*; sending to
the home channel there drops the reply into another chat (usually the bot owner's DM). Outside a
conversation (cron, a plain shell) the bare form still means the home channel.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform
from gateway.session_context import clear_session_vars, set_session_vars
from tools.send_message_tool import send_message_tool


def _send(target, mirror=None, **session):
    config = SimpleNamespace(
        platforms={p: SimpleNamespace(enabled=True, token="t", extra={}) for p in (Platform.FEISHU, Platform.TELEGRAM)},
        get_home_channel=lambda platform: SimpleNamespace(chat_id=f"{platform.value}-home"),
    )
    tokens = set_session_vars(**session) if session else None
    try:
        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("tools.send_message_tool._send_to_platform",
                   new=AsyncMock(return_value={"success": True})) as sent, \
             patch("gateway.mirror.mirror_to_session", new=mirror or (lambda *a, **k: False)):
            result = json.loads(send_message_tool({"action": "send", "target": target, "message": "hi"}))
    finally:
        if tokens is not None:
            clear_session_vars(tokens)
    assert result["success"] is True
    [call] = sent.await_args_list
    return call.args[2], call.kwargs["thread_id"], result


def test_a_bare_platform_inside_its_conversation_goes_to_that_conversation():
    chat_id, thread_id, result = _send("feishu", platform="feishu", chat_id="oc_group", thread_id="omt_topic")

    assert (chat_id, thread_id) == ("oc_group", "omt_topic")
    assert "home channel" not in result["note"]


def test_outside_any_conversation_a_bare_platform_is_the_home_channel():
    chat_id, thread_id, result = _send("feishu")

    assert (chat_id, thread_id) == ("feishu-home", None)
    assert "home channel" in result["note"]


def test_a_conversation_on_another_platform_does_not_capture_the_target():
    chat_id, _, _ = _send("feishu", platform="telegram", chat_id="-100123")

    assert chat_id == "feishu-home"


@pytest.mark.parametrize("target", ["feishu:oc_other", "feishu:oc_other:omt_other"])
def test_an_explicit_chat_is_never_redirected_to_the_conversation(target):
    chat_id, thread_id, _ = _send(target, platform="feishu", chat_id="oc_group", thread_id="omt_topic")

    assert chat_id == "oc_other"
    assert thread_id == (target.split(":")[2] if target.count(":") == 2 else None)


def test_a_cron_run_keeps_the_home_channel():
    """Cron binds no conversation (its origin is delivery metadata, not a sender)."""
    chat_id, _, _ = _send("feishu", platform="", chat_id="", cron_session="1")

    assert chat_id == "feishu-home"


@pytest.mark.parametrize("target", ["feishu", "feishu:oc_group:omt_topic"])
def test_a_send_into_the_own_conversation_is_not_mirrored_into_it(target):
    """The live transcript has the send as this turn's tool call; a mirror row would split call and result."""
    from unittest.mock import Mock

    mirror = Mock(return_value=True)
    _, _, result = _send(target, mirror=mirror, platform="feishu", chat_id="oc_group", thread_id="omt_topic")

    mirror.assert_not_called()
    assert "mirrored" not in result


def test_a_send_to_another_chat_is_still_mirrored():
    from unittest.mock import Mock

    mirror = Mock(return_value=True)
    _, _, result = _send("feishu:oc_other", mirror=mirror, platform="feishu", chat_id="oc_group", thread_id="omt_topic")

    mirror.assert_called_once()
    assert result["mirrored"] is True
