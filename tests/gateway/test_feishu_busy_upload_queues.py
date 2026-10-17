"""A file, folder or archive sent to Feishu while a turn runs waits for the next turn.

Under the default ``busy_input_mode: interrupt`` an upload used to abort the running turn
(``running_agent.interrupt``): work in progress was cut off and the upload only ran as the next
turn anyway. An upload is material for the work, not a correction, so on an adapter that opts in
(``queue_busy_uploads``, Feishu) it is queued whole and the running turn continues. Both busy entry
points — the base adapter's busy callback and the runner's priority path — must agree.
"""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from plugins.platforms.feishu.adapter import FeishuAdapter


class _RunningTurn:
    _supports_active_turn_redirect = True

    def __init__(self):
        self._active_children = []
        self.interrupt = Mock()
        self.redirect = Mock(return_value=True)
        self.steer = Mock(return_value=True)


_UPLOADS = {
    "document": (MessageType.DOCUMENT, "", ["/cache/doc_a_b_report.pdf"], ["application/pdf"]),
    "zip-with-caption": (MessageType.TEXT, "use these sources", ["/cache/doc_a_b_src.zip"], ["application/zip"]),
    "folder": (
        MessageType.TEXT, "",
        ["/cache/doc_a_b_feishu-folder-manifest.json", "/cache/doc_a_b_notes.txt"],
        ["application/json", "text/plain"],
    ),
}


def _event(message_type, text, media_urls, media_types, message_id="om_upload"):
    source = SessionSource(platform=Platform.FEISHU, chat_id="oc_dm", chat_type="dm", user_id="ou_owner")
    return MessageEvent(
        text=text, message_type=message_type, source=source, message_id=message_id,
        media_urls=list(media_urls), media_types=list(media_types),
        media_text_inlined=[False] * len(media_urls),
    )


def _running_session(monkeypatch, event):
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    monkeypatch.setenv("FEISHU_REACTIONS", "false")
    runner = GatewayRunner(config=GatewayConfig())
    runner._busy_input_mode = "interrupt"
    runner._busy_text_mode = "interrupt"
    runner._is_user_authorized = lambda source: True
    adapter = FeishuAdapter(PlatformConfig())
    runner.adapters = {Platform.FEISHU: adapter}
    key = runner._session_key_for_source(event.source)
    turn = _RunningTurn()
    runner._session_state(key).turn.agent = turn
    adapter._active_sessions[key] = asyncio.Event()
    adapter._message_handler = runner._handle_message
    adapter._busy_session_handler = runner._handle_active_session_busy_message
    return runner, adapter, turn, key


def _queued(runner, adapter, key):
    head = adapter._pending_messages.get(key)
    return [head, *runner._session_state(key).conversation.queued_events] if head is not None else []


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["adapter", "priority"])
@pytest.mark.parametrize("kind", sorted(_UPLOADS))
async def test_upload_sent_mid_turn_waits_for_the_next_turn_without_interrupting(monkeypatch, entry, kind):
    message_type, text, media_urls, media_types = _UPLOADS[kind]
    event = _event(message_type, text, media_urls, media_types)
    runner, adapter, turn, key = _running_session(monkeypatch, event)

    if entry == "adapter":
        await adapter.handle_message(event)
    else:
        await runner._hm_handle_running_session_message(event, event.source, key)

    turn.interrupt.assert_not_called()
    turn.redirect.assert_not_called()
    turn.steer.assert_not_called()
    queued = _queued(runner, adapter, key)
    assert queued == [event], "the whole upload runs as the next turn"
    assert (queued[0].media_urls, queued[0].media_types, queued[0].text) == (media_urls, media_types, text)

