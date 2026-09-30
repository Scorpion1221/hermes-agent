"""Attachment follow-ups must release a clarify wait without losing the upload."""

import asyncio
import json
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource
from tools import clarify_gateway as cm


class _Adapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.SLACK):
        super().__init__(PlatformConfig(enabled=True, token="test"), platform)

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise AssertionError("attachment interception must not send a separate reply")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "im"}


class _FellThroughIntercept(Exception):
    """The complete attachment event reached routing after clarify interception."""


@pytest.fixture(autouse=True)
def _clear_clarify_state():
    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()
    yield
    with cm._lock:
        session_keys = list(cm._session_index)
    for session_key in session_keys:
        cm.clear_session(session_key)
    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


def _event(text="", message_type=MessageType.DOCUMENT):
    return MessageEvent(
        text=text,
        message_type=message_type,
        source=SessionSource(
            platform=Platform.SLACK,
            chat_id="D123",
            chat_type="dm",
            user_id="U1",
            thread_id="1111.2222",
        ),
        message_id="upload-1",
        media_urls=["/tmp/upload.pdf"],
        media_types=["application/pdf"],
        media_text_inlined=[False],
    )


def _runner(adapter):
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner._startup_restore_in_progress = False
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._is_user_authorized = lambda source: True
    runner._adapter_for_source = lambda source: adapter
    return runner


def _register(runner, event, prompt_kind="native"):
    entry = cm.register(
        "clarify-upload",
        runner._session_key_for_source(event.source),
        "Pick an option or send the relevant file",
        None if prompt_kind == "open" else ["A", "B"],
    )
    if prompt_kind == "other":
        assert cm.mark_awaiting_text(entry.clarify_id)
    return entry


async def _dispatch(runner, event):
    def _tripwire(_key):
        raise _FellThroughIntercept()

    with patch("hermes_cli.lifecycle.invoke_hook", return_value=[]), \
            patch("tools.slash_confirm.get_pending", side_effect=_tripwire):
        return await runner._handle_message(event)


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt_kind", ["native", "open", "other"])
@pytest.mark.parametrize(
    "text,message_type",
    [
        ("", MessageType.DOCUMENT),
        ("Use this file instead", MessageType.DOCUMENT),
        ("[File: upload.pdf]", MessageType.DOCUMENT),
        ("2", MessageType.TEXT),
        ("", MessageType.PHOTO),
    ],
    ids=["empty-document", "caption", "file-placeholder", "text-with-media", "photo"],
)
async def test_attachment_releases_clarify_and_falls_through_with_media_intact(
    prompt_kind, text, message_type,
):
    runner = _runner(_Adapter())
    event = _event(text, message_type)
    if message_type == MessageType.PHOTO:
        event.media_urls = ["/tmp/upload.png"]
        event.media_types = ["image/png"]
    original = (event.text, list(event.media_urls), list(event.media_types), list(event.media_text_inlined))
    entry = _register(runner, event, prompt_kind)

    with pytest.raises(_FellThroughIntercept):
        await _dispatch(runner, event)

    assert entry.event.is_set()
    assert entry.response == ""
    assert (event.text, event.media_urls, event.media_types, event.media_text_inlined) == original


@pytest.mark.asyncio
@pytest.mark.parametrize("message_type", [MessageType.TEXT, MessageType.VOICE])
async def test_mixed_voice_and_document_is_a_followup_not_a_transcribed_answer(message_type):
    runner = _runner(_Adapter())
    event = _event("", message_type)
    event.media_urls = ["/tmp/voice.ogg", "/tmp/upload.pdf"]
    event.media_types = ["audio/ogg", "application/pdf"]
    event.media_text_inlined = [False, False]
    entry = _register(runner, event, "open")
    runner._transcribe_pending_audio_event_once = AsyncMock(return_value=("A", ["A"]))

    with pytest.raises(_FellThroughIntercept):
        await _dispatch(runner, event)

    assert entry.event.is_set()
    assert entry.response == ""
    assert event.media_urls == ["/tmp/voice.ogg", "/tmp/upload.pdf"]
    assert event.media_types == ["audio/ogg", "application/pdf"]


@pytest.mark.asyncio
async def test_pure_voice_transcript_still_answers_clarify():
    runner = _runner(_Adapter())
    event = _event("", MessageType.VOICE)
    event.media_urls = ["/tmp/voice.ogg"]
    event.media_types = ["audio/ogg"]
    entry = _register(runner, event)
    runner._transcribe_pending_audio_event_once = AsyncMock(return_value=("2", ["2"]))

    assert await _dispatch(runner, event) == ""

    assert entry.event.is_set()
    assert entry.response == "B"


@pytest.mark.asyncio
async def test_failed_pure_voice_transcription_keeps_clarify_pending():
    runner = _runner(_Adapter())
    event = _event("", MessageType.VOICE)
    event.media_urls = ["/tmp/voice.ogg"]
    event.media_types = ["audio/ogg"]
    entry = _register(runner, event, "open")
    runner._transcribe_pending_audio_event_once = AsyncMock(return_value=("", []))

    assert await _dispatch(runner, event) == ""

    assert not entry.event.is_set()
    assert entry.response is None


@pytest.mark.asyncio
async def test_attachment_does_not_overwrite_a_button_response():
    runner = _runner(_Adapter())
    event = _event()
    entry = _register(runner, event)
    assert cm.resolve_gateway_clarify(entry.clarify_id, "B")

    with pytest.raises(_FellThroughIntercept):
        await _dispatch(runner, event)

    assert entry.response == "B"


@pytest.mark.asyncio
async def test_attachment_releases_only_the_current_clarify():
    runner = _runner(_Adapter())
    event = _event()
    entry = _register(runner, event, "open")
    later_entry = cm.register("later-clarify", entry.session_key, "A later question", None)

    with pytest.raises(_FellThroughIntercept):
        await _dispatch(runner, event)

    assert entry.event.is_set()
    assert entry.response == ""
    assert not later_entry.event.is_set()
    assert later_entry.response is None


@pytest.mark.asyncio
async def test_untrusted_attachment_cannot_release_clarify():
    runner = _runner(_Adapter())
    event = _event("A")
    event.allow_gateway_control = False
    entry = _register(runner, event, "open")

    with pytest.raises(_FellThroughIntercept):
        await _dispatch(runner, event)

    assert not entry.event.is_set()
    assert entry.response is None


@pytest.mark.asyncio
async def test_quoted_media_does_not_change_a_text_clarify_answer():
    runner = _runner(_Adapter())
    event = _event("2", MessageType.TEXT)
    event.reply_to_media_urls = event.media_urls
    event.reply_to_media_types = event.media_types
    event.media_urls = []
    event.media_types = []
    event.media_text_inlined = []
    entry = _register(runner, event)

    assert await _dispatch(runner, event) == ""

    assert entry.event.is_set()
    assert entry.response == "B"


@pytest.mark.asyncio
@pytest.mark.parametrize("different_route", ["chat", "thread", "sender", "profile"])
async def test_attachment_does_not_release_another_session(different_route):
    runner = _runner(_Adapter())
    original_event = _event()
    if different_route == "sender":
        original_event.source = replace(original_event.source, chat_type="group", thread_id=None)
        other_source = replace(original_event.source, user_id="U2")
    elif different_route == "profile":
        runner.config.multiplex_profiles = True
        original_event.source = replace(original_event.source, profile="ops")
        other_source = replace(original_event.source, profile="dev")
    elif different_route == "chat":
        other_source = replace(original_event.source, chat_id="D456")
    else:
        other_source = replace(original_event.source, thread_id="3333.4444")
    entry = _register(runner, original_event, "open")
    event = replace(original_event, source=other_source)
    assert runner._session_key_for_source(event.source) != entry.session_key

    with pytest.raises(_FellThroughIntercept):
        await _dispatch(runner, event)

    assert not entry.event.is_set()
    assert entry.response is None


@pytest.mark.asyncio
async def test_unauthorized_group_attachment_cannot_release_clarify():
    runner = _runner(_Adapter())
    event = _event()
    event.source = replace(event.source, chat_type="group", thread_id=None)
    entry = _register(runner, event, "open")
    runner._is_user_authorized = lambda source: False

    assert await _dispatch(runner, event) is None

    assert not entry.event.is_set()
    assert entry.response is None


@pytest.mark.asyncio
async def test_active_adapter_attachment_bypass_unblocks_real_clarify_waiter():
    adapter = _Adapter()
    runner = _runner(adapter)
    event = _event()
    entry = _register(runner, event, "open")
    adapter._active_sessions[entry.session_key] = asyncio.Event()
    adapter._busy_session_handler = AsyncMock(return_value=True)
    routed_events = []

    async def _message_handler(inbound):
        try:
            return await _dispatch(runner, inbound)
        except _FellThroughIntercept:
            routed_events.append(inbound)
            return ""

    adapter._message_handler = _message_handler
    waiter_started = threading.Event()
    responses = []

    def _wait():
        waiter_started.set()
        responses.append(cm.wait_for_response(entry.clarify_id, timeout=5))

    waiter = threading.Thread(target=_wait, daemon=True)
    waiter.start()
    try:
        assert waiter_started.wait(timeout=1)
        await adapter.handle_message(event)

        assert entry.event.is_set()
        await asyncio.to_thread(waiter.join, 3)
        assert not waiter.is_alive()
        assert responses == [""]
        assert routed_events == [event]
        assert routed_events[0].media_urls == ["/tmp/upload.pdf"]
        adapter._busy_session_handler.assert_not_awaited()
        assert adapter._pending_messages == {}
    finally:
        cm.clear_session(entry.session_key)
        await asyncio.to_thread(waiter.join, 3)


@pytest.mark.asyncio
async def test_feishu_numbered_text_fallback_still_accepts_free_prose():
    adapter = _Adapter(Platform.FEISHU)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="question-1"))
    runner = _runner(adapter)
    event = _event("Use the existing production data", MessageType.TEXT)
    event.source = replace(event.source, platform=Platform.FEISHU)
    event.media_urls = []
    event.media_types = []
    event.media_text_inlined = []
    entry = _register(runner, event)
    assert not entry.awaiting_text

    sent = await adapter.send_clarify(
        chat_id=event.source.chat_id,
        question=entry.question,
        choices=entry.choices,
        clarify_id=entry.clarify_id,
        session_key=entry.session_key,
    )

    assert sent.success
    assert entry.awaiting_text
    adapter.send.assert_awaited_once()
    content = adapter.send.await_args.kwargs["content"]
    assert "1. A" in content
    assert "2. B" in content
    assert await _dispatch(runner, event) == ""
    assert entry.event.is_set()
    assert entry.response == event.text


@pytest.mark.asyncio
async def test_zip_followup_aborts_real_legacy_batch_before_second_question():
    from gateway.run import _clarify_send_then_wait
    from tools.clarify_tool import TIMEOUT_RESPONSE, clarify_tool

    adapter = _Adapter(Platform.FEISHU)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="question-1"))
    runner = _runner(adapter)
    event = _event()
    event.source = replace(event.source, platform=Platform.FEISHU)
    event.media_urls = ["/tmp/source.zip"]
    event.media_types = ["application/zip"]
    session_key = runner._session_key_for_source(event.source)
    adapter._active_sessions[session_key] = asyncio.Event()
    adapter._busy_session_handler = AsyncMock(return_value=True)
    routed_events = []

    async def _message_handler(inbound):
        try:
            return await _dispatch(runner, inbound)
        except _FellThroughIntercept:
            routed_events.append(inbound)
            return ""

    adapter._message_handler = _message_handler
    loop = asyncio.get_running_loop()
    prompt_sent = threading.Event()
    registered_entries = []
    callback_questions = []
    callback_responses = []
    results = []
    failures = []

    def _legacy_callback(question, choices, multi_select=False):
        callback_questions.append(question)
        entry = cm.register(
            f"batch-{len(callback_questions)}",
            session_key,
            question,
            choices,
            multi_select=multi_select,
        )
        registered_entries.append(entry)
        sent = asyncio.run_coroutine_threadsafe(
            adapter.send_clarify(
                chat_id=event.source.chat_id,
                question=question,
                choices=choices,
                clarify_id=entry.clarify_id,
                session_key=session_key,
            ),
            loop,
        )
        assert sent.result(timeout=3).success
        prompt_sent.set()
        response = _clarify_send_then_wait(
            sent,
            clarify_id=entry.clarify_id,
            session_key=session_key,
            clarify_mod=cm,
        )
        callback_responses.append(response)
        return response

    questions = [
        {"question": "Where are the source files?", "choices": ["Already local", "Upload ZIP"]},
        {"question": "Which output format?", "choices": ["Markdown", "HTML"]},
    ]

    def _run_batch():
        try:
            results.append(json.loads(clarify_tool("", questions=questions, callback=_legacy_callback)))
        except Exception as exc:
            failures.append(exc)

    worker = threading.Thread(target=_run_batch, daemon=True)
    with patch.object(cm, "get_clarify_timeout", return_value=30):
        worker.start()
        try:
            assert await asyncio.to_thread(prompt_sent.wait, 3), (failures, results)
            assert registered_entries[0].awaiting_text
            await adapter.handle_message(event)

            assert registered_entries[0].event.is_set()
            await asyncio.to_thread(worker.join, 3)
            assert not worker.is_alive()
            assert failures == []
            assert callback_questions == [questions[0]["question"]]
            assert callback_responses == [TIMEOUT_RESPONSE]
            assert len(registered_entries) == 1
            assert len(results) == 1
            assert results[0]["timed_out"] is True
            assert [row["user_response"] for row in results[0]["responses"]] == ["", ""]
            assert cm.get_pending_for_session(session_key, include_choice_prompts=True) is None
            assert routed_events == [event]
            assert routed_events[0].text == ""
            assert routed_events[0].media_urls == ["/tmp/source.zip"]
            assert routed_events[0].media_types == ["application/zip"]
            adapter._busy_session_handler.assert_not_awaited()
            assert adapter._pending_messages == {}
            adapter.send.assert_awaited_once()
        finally:
            cm.clear_session(session_key)
            await asyncio.to_thread(worker.join, 3)
            # A broken legacy callback can reopen question two during cleanup.
            if worker.is_alive():
                cm.clear_session(session_key)
                await asyncio.to_thread(worker.join, 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", [Platform.FEISHU, Platform.SLACK])
@pytest.mark.parametrize("busy_text_mode", ["interrupt", "queue"])
@pytest.mark.parametrize(
    "text,message_type",
    [("", MessageType.DOCUMENT), ("Use these source files", MessageType.TEXT)],
    ids=["empty-zip", "caption-with-zip"],
)
async def test_real_priority_route_preserves_clarify_upload_before_interrupt(
    monkeypatch, platform, busy_text_mode, text, message_type,
):
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    adapter = _Adapter(platform)
    runner = _runner(adapter)
    runner.adapters = {platform: adapter}
    runner.session_store = None
    runner._draining = False
    runner._busy_input_mode = "interrupt"
    runner._busy_text_mode = busy_text_mode
    event = _event(text, message_type)
    event.source = replace(event.source, platform=platform)
    event.media_urls = ["/tmp/source.zip"]
    event.media_types = ["application/zip"]
    entry = _register(runner, event, "open")
    session_key = entry.session_key
    interrupt_observations = []

    def _interrupt(_message):
        pending = adapter._pending_messages.get(session_key)
        interrupt_observations.append((
            pending is event,
            list(getattr(pending, "media_urls", [])),
            list(getattr(pending, "media_types", [])),
        ))
        # The busy handler catches interrupt exceptions, so the recorded
        # observation below also makes an ordering assertion failure visible.
        assert pending is event
        assert pending.media_urls == ["/tmp/source.zip"]
        assert pending.media_types == ["application/zip"]
        assert entry.event.is_set()

    running_agent = SimpleNamespace(
        interrupt=Mock(side_effect=_interrupt),
        _active_children=[],
        _gateway_stream_consumer=None,
    )
    state = runner._session_state(session_key)
    state.turn.agent = running_agent
    adapter._active_sessions[session_key] = asyncio.Event()
    adapter._message_handler = runner._handle_message
    adapter._busy_session_handler = runner._handle_active_session_busy_message

    with patch("hermes_cli.lifecycle.invoke_hook", return_value=[]), \
            patch("tools.slash_confirm.get_pending", return_value=None):
        await adapter.handle_message(event)

    assert entry.event.is_set()
    assert entry.response == ""
    running_agent.interrupt.assert_called_once()
    assert interrupt_observations == [(True, ["/tmp/source.zip"], ["application/zip"])]
    assert adapter._pending_messages[session_key] is event
    assert event.text == text
    assert state.conversation.queued_events == []


@pytest.mark.asyncio
async def test_real_busy_steer_mixed_voice_and_zip_queues_the_complete_event(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    adapter = _Adapter(Platform.FEISHU)
    runner = _runner(adapter)
    runner.adapters = {Platform.FEISHU: adapter}
    runner.session_store = None
    runner._draining = False
    runner._busy_input_mode = "steer"
    runner._busy_text_mode = "interrupt"
    runner._prepare_busy_steer_text = AsyncMock(return_value="Use the uploaded source files")
    event = _event("", MessageType.VOICE)
    event.source = replace(event.source, platform=Platform.FEISHU)
    event.media_urls = ["/tmp/voice.ogg", "/tmp/source.zip"]
    event.media_types = ["audio/ogg", "application/zip"]
    event.media_text_inlined = [False, False]
    entry = _register(runner, event, "open")
    session_key = entry.session_key
    running_agent = SimpleNamespace(
        steer=Mock(return_value=True),
        interrupt=Mock(),
        _active_children=[],
        _gateway_stream_consumer=None,
    )
    state = runner._session_state(session_key)
    state.turn.agent = running_agent
    adapter._active_sessions[session_key] = asyncio.Event()
    adapter._message_handler = runner._handle_message
    adapter._busy_session_handler = runner._handle_active_session_busy_message

    with patch("hermes_cli.lifecycle.invoke_hook", return_value=[]), \
            patch("tools.slash_confirm.get_pending", return_value=None):
        await adapter.handle_message(event)

    assert entry.event.is_set()
    assert entry.response == ""
    runner._prepare_busy_steer_text.assert_awaited_once_with(event)
    running_agent.steer.assert_not_called()
    running_agent.interrupt.assert_not_called()
    assert adapter._pending_messages[session_key] is event
    assert event.text == ""
    assert event.media_urls == ["/tmp/voice.ogg", "/tmp/source.zip"]
    assert event.media_types == ["audio/ogg", "application/zip"]
    assert event.media_text_inlined == [False, False]
    assert state.conversation.queued_events == []
