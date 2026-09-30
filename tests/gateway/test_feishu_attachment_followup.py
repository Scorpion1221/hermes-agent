"""Feishu cached uploads join the active turn without dropping their resources."""

import asyncio
import copy
import json
import threading
import zipfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from hermes_constants import get_hermes_home
from tools import clarify_gateway as cm


class _Adapter(BasePlatformAdapter):
    MAX_MESSAGE_LENGTH = 30000

    def __init__(self, platform):
        super().__init__(PlatformConfig(enabled=True, token="test"), platform)
        self.sent = []
        self._send_with_retry = AsyncMock()
        self.finalize_streaming_message = AsyncMock(return_value=True)
        self.edit_message = AsyncMock(return_value=SendResult(success=True, message_id="edited"))

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        message_id = f"card-{len(self.sent) + 1}"
        self.sent.append({"content": content, "reply_to": reply_to, "metadata": metadata, "message_id": message_id})
        return SendResult(success=True, message_id=message_id)

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "im"}


@pytest.fixture(autouse=True)
def _isolate_gateway(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda *a, **kw: [])
    yield
    with cm._lock:
        keys = list(cm._session_index)
    for key in keys:
        cm.clear_session(key)


def _upload(kind, *, platform=Platform.FEISHU):
    cache = get_hermes_home() / "cache" / "documents"
    cache.mkdir(parents=True, exist_ok=True)
    text = "Use these uploaded source files instead"
    if kind == "txt":
        leaf = cache / "doc_test_source.txt"
        leaf.write_text("source content", encoding="utf-8")
        paths, types = [leaf], ["text/plain"]
        message_type = MessageType.DOCUMENT
    elif kind == "zip":
        archive = cache / "doc_test_source.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("nested/source.txt", "zipped source content")
        paths, types = [archive], ["application/zip"]
        message_type = MessageType.TEXT
    else:
        leaf = cache / "doc_leaf_source.txt"
        leaf.write_text("nested source content", encoding="utf-8")
        picture = cache / "doc_leaf_diagram.png"
        # Valid one-pixel PNG: the test never replaces image bytes with caption text.
        import base64
        picture.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aVf0AAAAASUVORK5CYII="))
        manifest = cache / "doc_folder_feishu-folder-manifest.json"
        manifest.write_text(json.dumps({"status": "complete", "files": [
            {"relative_path": "folder/nested/source.txt", "local_path": str(leaf)},
            {"relative_path": "folder/diagram.png", "local_path": str(picture)},
        ]}), encoding="utf-8")
        paths, types = [manifest, leaf, picture], ["application/json", "text/plain", "image/png"]
        message_type = MessageType.DOCUMENT
    return MessageEvent(
        text=text, message_type=message_type,
        source=SessionSource(platform=platform, chat_id="oc_upload", chat_type="dm", user_id="owner"),
        message_id="om_upload", media_urls=[str(p) for p in paths], media_types=types,
        media_text_inlined=[False] * len(paths),
    )


def _setup(event, mode="interrupt"):
    from gateway.run import GatewayRunner
    from run_agent import AIAgent

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = None
    runner._draining = False
    runner._busy_input_mode = mode
    runner._busy_text_mode = "interrupt"
    runner._startup_restore_in_progress = False
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._is_user_authorized = lambda source: True
    adapter = _Adapter(event.source.platform)
    runner.adapters = {event.source.platform: adapter}
    key = runner._session_key_for_source(event.source)
    consumer = GatewayStreamConsumer(adapter, event.source.chat_id, StreamConsumerConfig(), metadata={"streaming": True})
    agent = object.__new__(AIAgent)
    agent.api_mode = "chat_completions"
    agent._pending_steer = None
    agent._pending_steer_lock = threading.Lock()
    agent._pending_redirect = None
    agent._pending_redirect_lock = threading.Lock()
    agent._model_request_active = threading.Event()
    agent._executing_tools = True
    agent._supports_active_turn_redirect = True
    agent._strip_think_blocks = lambda content: content
    agent._execution_thread_id = None
    agent._interrupt_requested = False
    agent._active_children = []
    agent._gateway_stream_consumer = consumer
    agent.interrupt = Mock()
    agent.steer_consumed_callback = lambda text="": consumer.on_user_input_boundary(text=text)
    runner._session_state(key).turn.agent = agent
    adapter._active_sessions[key] = asyncio.Event()
    adapter._message_handler = runner._handle_message
    adapter._busy_session_handler = runner._handle_active_session_busy_message
    return runner, adapter, agent, consumer, key


async def _dispatch(entry, runner, adapter, event, key):
    if entry == "adapter":
        await adapter.handle_message(event)
    else:
        await runner._handle_message(event)


def _consume(agent):
    from agent.agent_runtime_helpers import apply_pending_steer_to_tool_results

    messages = [
        {"role": "system", "content": "unchanged cached system prefix"},
        {"role": "user", "content": "original task"},
        {"role": "assistant", "tool_calls": [{"id": "call1"}]},
        {"role": "tool", "tool_call_id": "call1", "content": "original tool result"},
    ]
    prefix = copy.deepcopy(messages[:-1])
    apply_pending_steer_to_tool_results(agent, messages, 1)
    assert messages[:-1] == prefix
    assert [item["role"] for item in messages] == ["system", "user", "assistant", "tool"]
    assert messages[-1]["content"].startswith("original tool result")
    return messages[-1]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["adapter", "priority"])
@pytest.mark.parametrize("mode", ["interrupt", "steer"])
@pytest.mark.parametrize("kind", ["txt", "zip", "folder"])
async def test_cached_upload_enters_real_agent_steer_and_cardkit_receipt(entry, mode, kind):
    event = _upload(kind)
    original = copy.deepcopy((event.text, event.media_urls, event.media_types, event.media_text_inlined))
    runner, adapter, agent, consumer, key = _setup(event, mode)

    await _dispatch(entry, runner, adapter, event, key)

    assert agent._pending_steer, "the entire upload must enter the active turn, not a new interrupted task"
    assert event.text in agent._pending_steer
    for path in event.media_urls:
        assert path in agent._pending_steer
    assert len(consumer._followups) == 1
    receipt = consumer._followups[0]
    assert receipt.text == agent._pending_steer
    assert adapter.sent[0]["reply_to"] == event.message_id
    assert key not in adapter._pending_messages
    assert not runner._session_state(key).conversation.queued_events
    assert not runner._session_state(key).persistent.native_image_paths
    agent.interrupt.assert_not_called()
    adapter._send_with_retry.assert_not_awaited()
    consumed = _consume(agent)
    assert receipt.text in consumed
    assert agent._pending_steer is None
    assert receipt.consumed
    consumer.finish("continued with uploaded files")
    await asyncio.wait_for(consumer.run(), 5)
    assert (event.text, event.media_urls, event.media_types, event.media_text_inlined) == original


@pytest.mark.asyncio
@pytest.mark.parametrize("backend,prefix", [("docker", "/root/.hermes/cache/documents/"), ("ssh", "~/.hermes/cache/documents/")])
async def test_remote_attachment_paths_are_agent_visible_without_native_image_buffer(monkeypatch, backend, prefix):
    monkeypatch.setenv("TERMINAL_ENV", backend)
    event = _upload("folder")
    runner, adapter, agent, consumer, key = _setup(event)
    await runner._handle_active_session_busy_message(event, key)
    assert agent._pending_steer
    for path in event.media_urls:
        from pathlib import Path
        assert prefix + Path(path).name in agent._pending_steer
        assert path not in agent._pending_steer
    assert not runner._session_state(key).persistent.native_image_paths
    agent.interrupt.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "directory", "remote-url", "extra-mime"])
@pytest.mark.parametrize("mode", ["interrupt", "steer"])
async def test_unrepresentable_upload_keeps_complete_original_event(failure, mode):
    from pathlib import Path

    event = _upload("folder")
    if failure == "missing":
        Path(event.media_urls[-1]).unlink()
    elif failure == "directory":
        event.media_urls[-1] = str(Path(event.media_urls[-1]).parent)
    elif failure == "remote-url":
        event.media_urls[-1] = "https://example.invalid/unfetched.png"
    else:
        event.media_types.append("text/plain")
    original = copy.deepcopy((event.text, event.media_urls, event.media_types))
    runner, adapter, agent, consumer, key = _setup(event, mode)
    await runner._handle_active_session_busy_message(event, key)
    assert agent._pending_steer is None
    assert not consumer._followups
    assert adapter._pending_messages[key] is event
    assert (event.text, event.media_urls, event.media_types) == original
    agent.interrupt.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["interrupt", "steer"])
async def test_non_feishu_upload_retains_existing_routing(mode):
    event = _upload("txt", platform=Platform.SLACK)
    runner, adapter, agent, consumer, key = _setup(event, mode)
    await runner._handle_active_session_busy_message(event, key)
    assert agent._pending_steer is None
    assert adapter._pending_messages[key] is event
    assert not consumer._followups
    if mode == "interrupt":
        agent.interrupt.assert_called_once()
    else:
        agent.interrupt.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["interrupt", "steer"])
async def test_rejected_attachment_steer_preserves_original_event(mode):
    event = _upload("zip")
    runner, adapter, agent, consumer, key = _setup(event, mode)
    agent.steer = Mock(return_value=False)
    await runner._handle_active_session_busy_message(event, key)
    agent.steer.assert_called_once()
    assert event.media_urls[0] in agent.steer.call_args.args[0]
    assert adapter._pending_messages[key] is event
    assert not consumer._followups
    assert not adapter.sent
    agent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_captionless_folder_upload_redirects_active_model_and_keeps_cached_prefix():
    from agent.conversation_loop import _apply_active_turn_redirect

    event = _upload("folder")
    event.text = ""
    runner, adapter, agent, consumer, key = _setup(event)
    agent._executing_tools = False
    agent._model_request_active.set()
    await runner._handle_active_session_busy_message(event, key)
    correction = agent._drain_pending_redirect()
    assert correction
    for path in event.media_urls:
        assert path in correction
    agent.interrupt.assert_not_called()
    assert agent._pending_steer is None
    messages = [
        {"role": "system", "content": "cached system"},
        {"role": "user", "content": "initial request"},
        {"role": "assistant", "content": "prior committed response"},
    ]
    prefix = copy.deepcopy(messages)
    _apply_active_turn_redirect(agent, messages, correction)
    assert messages[:-1] == prefix
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
    assert messages[-1]["content"] == correction
    assert consumer._followups[0].consumed
    consumer.finish("continued")
    await asyncio.wait_for(consumer.run(), 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("message_type", [MessageType.AUDIO, MessageType.VIDEO])
async def test_audio_and_video_files_remain_cached_files_not_voice_notes(message_type):
    event = _upload("txt")
    event.message_type = message_type
    event.media_types = ["audio/mpeg" if message_type == MessageType.AUDIO else "video/mp4"]
    runner, adapter, agent, consumer, key = _setup(event)
    runner._transcribe_and_echo_pending_voice = AsyncMock(side_effect=AssertionError("not a voice note"))
    await runner._handle_active_session_busy_message(event, key)
    assert event.media_urls[0] in agent._pending_steer
    assert event.media_types[0] in agent._pending_steer
    runner._transcribe_and_echo_pending_voice.assert_not_awaited()
    agent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_explicit_steer_command_with_attachment_preserves_args_and_paths():
    event = _upload("zip")
    event.message_type = MessageType.COMMAND
    event.text = "/steer Use this uploaded archive instead"
    runner, adapter, agent, consumer, key = _setup(event)

    assert await runner._handle_message(event) == ""

    assert "Use this uploaded archive instead" in agent._pending_steer
    assert "/steer" not in agent._pending_steer
    assert event.media_urls[0] in agent._pending_steer
    assert len(consumer._followups) == 1
    assert consumer._followups[0].text == agent._pending_steer
    assert key not in adapter._pending_messages
    agent.interrupt.assert_not_called()
    assert consumer._followups[0].text in _consume(agent)
    consumer.finish("continued")
    await asyncio.wait_for(consumer.run(), 5)


@pytest.mark.asyncio
async def test_stop_with_attachment_remains_explicit_stop_not_attachment_steering():
    event = _upload("zip")
    event.message_type = MessageType.COMMAND
    event.text = "/stop"
    runner, adapter, agent, consumer, key = _setup(event)
    runner._interrupt_and_clear_session = AsyncMock()

    assert await runner._prepare_busy_attachment_followup(event) is None
    await runner._handle_message(event)

    runner._interrupt_and_clear_session.assert_awaited_once()
    assert runner._interrupt_and_clear_session.await_args.args[:2] == (key, event.source)
    assert runner._interrupt_and_clear_session.await_args.kwargs["invalidation_reason"] == "stop_command"
    assert agent._pending_steer is None
    assert not consumer._followups
    assert key not in adapter._pending_messages


@pytest.mark.asyncio
async def test_file_steering_does_not_consume_an_unrelated_native_image_buffer():
    event = _upload("folder")
    runner, adapter, agent, consumer, key = _setup(event)
    runner._session_state(key).persistent.native_image_paths = ["previous-preparation-image.png"]
    await runner._handle_active_session_busy_message(event, key)
    assert agent._pending_steer
    assert runner._session_state(key).persistent.native_image_paths == ["previous-preparation-image.png"]


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["adapter", "priority"])
@pytest.mark.parametrize("mode", ["interrupt", "steer"])
@pytest.mark.parametrize("raw_type", ["file", "image"])
async def test_real_bridge_distinguishes_uploaded_image_file_from_native_photo(entry, mode, raw_type):
    from gateway.platforms.feishu_inbound.bridge import (
        build_feishu_inbound_content_bridge,
        build_feishu_message_event,
    )

    upload = _upload("folder")
    image_path = upload.media_urls[-1]
    raw = {"file_key": "file_image", "file_name": "diagram.png"} if raw_type == "file" else {"image_key": "img_native"}
    message = SimpleNamespace(
        message_id="om_image", message_type=raw_type, content=json.dumps(raw),
        chat_id=upload.source.chat_id, chat_type="p2p", create_time="1790773200000",
    )
    extracted = build_feishu_inbound_content_bridge(
        message=message, media_urls=[image_path], media_types=["image/png"],
    )
    event = build_feishu_message_event(
        data=message, message=message, source=upload.source, inbound_content=extracted,
    )
    assert event.message_type == MessageType.PHOTO
    assert event.metadata["feishu_message_type"] == raw_type
    runner, adapter, agent, consumer, key = _setup(event, mode)

    if raw_type == "image":
        assert await runner._prepare_busy_attachment_followup(event) is None
    await _dispatch(entry, runner, adapter, event, key)

    if raw_type == "file":
        assert image_path in agent._pending_steer
        assert "image/png" in agent._pending_steer
        assert len(consumer._followups) == 1
        assert consumer._followups[0].text == agent._pending_steer
        assert key not in adapter._pending_messages
        assert not runner._session_state(key).persistent.native_image_paths
        agent.interrupt.assert_not_called()
        assert consumer._followups[0].text in _consume(agent)
        consumer.finish("continued using image file")
        await asyncio.wait_for(consumer.run(), 5)
    else:
        assert agent._pending_steer is None
        assert adapter._pending_messages[key] is event
        assert not consumer._followups
        if entry == "adapter" and mode == "interrupt":
            agent.interrupt.assert_called_once()
        else:
            agent.interrupt.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_state", ["starting", "absent"])
@pytest.mark.parametrize("args", ["Use this archive", ""])
async def test_steer_upload_without_ready_agent_queues_all_fields_without_replaying_command(agent_state, args):
    from gateway.run import _AGENT_PENDING_SENTINEL

    event = _upload("folder")
    event.message_type = MessageType.COMMAND
    event.text = "/steer" + (" " + args if args else "")
    event.metadata = {"feishu_message_type": "post", "preserve": "metadata"}
    event.reply_to_message_id = "om_prior"
    event.reply_to_text = "existing user context"
    runner, adapter, agent, consumer, key = _setup(event)
    runner._session_state(key).turn.agent = _AGENT_PENDING_SENTINEL if agent_state == "starting" else None

    await runner._busy_steer_command(event, key, event.source)

    pending = adapter._pending_messages[key]
    assert pending.text == args
    assert pending.message_type == MessageType.TEXT
    assert pending.source is event.source
    for field in (
        "message_id", "media_urls", "media_types", "media_text_inlined", "metadata",
        "reply_to_message_id", "reply_to_text", "timestamp",
    ):
        assert getattr(pending, field) == getattr(event, field)
    assert event.text.startswith("/steer")
    assert not consumer._followups
    agent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_cold_captionless_steer_upload_reaches_agent_with_complete_attachment_event():
    event = _upload("folder")
    event.message_type = MessageType.COMMAND
    event.text = "/steer"
    event.metadata = {"feishu_message_type": "post"}
    original_media = copy.deepcopy((event.media_urls, event.media_types, event.media_text_inlined))
    runner, adapter, agent, consumer, key = _setup(event)
    runner._session_state(key).turn.agent = None
    adapter._active_sessions.clear()
    runner._external_drain_active = False
    # Keep this a dispatcher integration test: no durable runtime status or
    # model/provider work is needed to verify the event crossing this boundary.
    runner._claim_active_session_slot = lambda *args: (None, None)
    runner._persist_active_agents = lambda: None
    runner._begin_session_run_generation = lambda _key: 1
    runner._run_post_turn_hooks = AsyncMock()
    runner._clear_durable_active_turn = AsyncMock()
    captured = []

    async def capture_agent_turn(inbound, source, session_key, generation):
        captured.append(inbound)
        assert inbound is event
        assert inbound.text == ""
        assert inbound.get_command() is None
        assert (inbound.media_urls, inbound.media_types, inbound.media_text_inlined) == original_media
        assert inbound.metadata == {"feishu_message_type": "post"}
        assert source is event.source
        assert session_key == key
        return "attachment reached the new turn"

    runner._handle_message_with_agent = capture_agent_turn

    assert await runner._handle_message(event) == "attachment reached the new turn"
    assert captured == [event]
    assert not consumer._followups


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["interrupt", "steer"])
async def test_clarify_batch_upload_releases_wait_and_continues_same_agent_before_question_two(mode):
    from gateway.run import _clarify_send_then_wait
    from tools.clarify_tool import TIMEOUT_RESPONSE, clarify_tool

    event = _upload("folder")
    event.text = ""
    runner, adapter, agent, consumer, key = _setup(event, mode)
    loop = asyncio.get_running_loop()
    prompt_sent = threading.Event()
    entries, questions, responses, results, failures = [], [], [], [], []

    def clarify_callback(question, choices, multi_select=False):
        questions.append(question)
        entry = cm.register(f"batch-{len(questions)}", key, question, choices, multi_select=multi_select)
        entries.append(entry)
        sent = asyncio.run_coroutine_threadsafe(adapter.send_clarify(
            chat_id=event.source.chat_id, question=question, choices=choices,
            clarify_id=entry.clarify_id, session_key=key,
        ), loop)
        assert sent.result(timeout=3).success
        prompt_sent.set()
        response = _clarify_send_then_wait(sent, clarify_id=entry.clarify_id, session_key=key, clarify_mod=cm)
        responses.append(response)
        return response

    def ask_batch():
        try:
            results.append(json.loads(clarify_tool("", questions=[
                {"question": "How will you provide the source?", "choices": ["Upload", "Paste"]},
                {"question": "Which output format?", "choices": ["Markdown", "HTML"]},
            ], callback=clarify_callback)))
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=ask_batch, daemon=True)
    with patch.object(cm, "get_clarify_timeout", return_value=20):
        worker.start()
        try:
            assert await asyncio.to_thread(prompt_sent.wait, 3), failures
            assert entries[0].awaiting_text
            await adapter.handle_message(event)
            await asyncio.to_thread(worker.join, 3)
            assert not worker.is_alive()
            assert failures == []
            assert len(questions) == 1
            assert responses == [TIMEOUT_RESPONSE]
            assert len(results) == 1 and results[0]["timed_out"] is True
            assert agent._pending_steer
            for path in event.media_urls:
                assert path in agent._pending_steer
            assert len(consumer._followups) == 1
            assert key not in adapter._pending_messages
            assert cm.get_pending_for_session(key, include_choice_prompts=True) is None
            agent.interrupt.assert_not_called()
            assert consumer._followups[0].text in _consume(agent)
            assert consumer._followups[0].consumed
            consumer.finish("continued with uploaded folder")
            await asyncio.wait_for(consumer.run(), 5)
        finally:
            cm.clear_session(key)
            await asyncio.to_thread(worker.join, 3)
