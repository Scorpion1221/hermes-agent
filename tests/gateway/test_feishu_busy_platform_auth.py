"""Busy-path parity with the cold path for adapter-admitted Feishu group members.

The Feishu adapter admits group traffic through its own group ACL and marks the
event ``platform_auth_passed``. The cold path honours that marker; the busy path
must too, otherwise every admitted member outside FEISHU_ALLOWED_USERS is
silently dropped while the bot is answering someone else.
"""
import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_owner")
    monkeypatch.setenv("FEISHU_BOT_OPEN_ID", "ou_bot")
    monkeypatch.setenv("HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS", "0")
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    for name in ("FEISHU_ALLOW_ALL_USERS", "GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS", "FEISHU_GROUP_POLICY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda *a, **kw: [])


def _busy_gateway(extra):
    """Real FeishuAdapter + real GatewayRunner auth, with the owner's turn running."""
    from gateway.run import GatewayRunner
    from plugins.platforms.feishu.adapter import FeishuAdapter
    from run_agent import AIAgent

    adapter = FeishuAdapter(PlatformConfig(enabled=True, extra={**extra, "group_sessions_per_user": False}))
    adapter._send_with_retry = AsyncMock()
    adapter.send = AsyncMock()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(group_sessions_per_user=False)
    runner.session_store = None
    runner._draining = False
    runner._busy_input_mode = "queue"
    runner._busy_text_mode = "interrupt"
    runner._startup_restore_in_progress = False
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner.pairing_store = None
    runner.adapters = {Platform.FEISHU: adapter}
    adapter.set_message_handler(runner._handle_message)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)

    owner = adapter.build_source(chat_id="oc_group", chat_type="group", user_id="ou_owner")
    key = runner._session_key_for_source(owner)
    agent = object.__new__(AIAgent)
    agent._pending_steer = None
    agent._pending_steer_lock = threading.Lock()
    agent._active_children = []
    agent._gateway_stream_consumer = None
    agent.interrupt = Mock()
    runner._session_state(key).turn.agent = agent
    adapter._active_sessions[key] = asyncio.Event()
    return runner, adapter, key


def _group_message(sender, text, message_id, *, chat_type="group"):
    message = SimpleNamespace(
        message_id=message_id, chat_id="oc_group", chat_type=chat_type, message_type="text",
        content=json.dumps({"text": f"@_user_1 {text}"}),
        mentions=[SimpleNamespace(key="@_user_1", id=SimpleNamespace(open_id="ou_bot", user_id=None, union_id=None), name="Bot")],
        thread_id=None, parent_id=None, root_id=None, upper_message_id=None,
    )
    sender = SimpleNamespace(sender_id=SimpleNamespace(open_id=sender, user_id=None, union_id=None), sender_type="user")
    return SimpleNamespace(event=SimpleNamespace(message=message, sender=sender))


async def _deliver(adapter, data):
    await adapter._handle_message_event_data(data)
    for _ in range(20):
        if not adapter._pending_text_batch_tasks:
            break
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_group_member_admitted_by_group_rule_is_queued_while_bot_is_busy():
    rules = {"group_rules": {"oc_group": {"policy": "allowlist", "allowlist": ["ou_member"]}}}
    runner, adapter, key = _busy_gateway(rules)

    await _deliver(adapter, _group_message("ou_member", "1+1", "om_1"))

    assert adapter._pending_messages[key].text == "1+1"
    assert adapter._pending_messages[key].source.user_id == "ou_member"


@pytest.mark.asyncio
async def test_group_member_rejected_by_group_rule_still_never_reaches_the_session():
    rules = {"group_rules": {"oc_group": {"policy": "allowlist", "allowlist": ["ou_member"]}}}
    runner, adapter, key = _busy_gateway(rules)

    await _deliver(adapter, _group_message("ou_outsider", "inject", "om_2"))

    assert key not in adapter._pending_messages


@pytest.mark.asyncio
async def test_unmarked_event_from_non_allowlisted_sender_is_still_dropped_while_busy():
    """#17775 stays closed for events the adapter did not admit (DMs, synthetic events)."""
    runner, adapter, key = _busy_gateway({})
    source = SessionSource(platform=Platform.FEISHU, chat_id="oc_group", chat_type="group", user_id="ou_outsider")
    event = MessageEvent(text="inject", message_type=MessageType.TEXT, source=source, message_id="om_3")

    assert await runner._handle_active_session_busy_message(event, key) is True
    assert key not in adapter._pending_messages


@pytest.mark.asyncio
async def test_queue_command_from_admitted_member_survives_the_drain():
    runner, adapter, key = _busy_gateway({})
    source = SessionSource(platform=Platform.FEISHU, chat_id="oc_group", chat_type="group", user_id="ou_member")
    event = MessageEvent(text="/queue 排队", message_type=MessageType.COMMAND, source=source, message_id="om_4")
    event.platform_auth_passed = True
    reached = []

    async def _agent(ev, *_a, **_k):
        reached.append(ev.text)

    runner._handle_message_with_agent = _agent
    await runner._busy_queue_command(event, key, source)
    queued = adapter._pending_messages.pop(key)
    # /stop released the run; the base adapter drain then re-enters the cold path.
    runner._release_running_agent_state(key)
    adapter._active_sessions.pop(key, None)
    await runner._handle_message(queued)

    assert reached == ["排队"]


# ── Shared group sessions: another member's busy message is its own turn ───


def _redirectable_owner_turn(runner, adapter, key, owner_id="ou_owner"):
    """Swap in an owner agent mid model request, so redirect() would accept."""
    agent = runner._session_state(key).turn.agent
    agent._user_id, agent._user_id_alt = owner_id, None
    agent._supports_active_turn_redirect = True
    agent._model_request_active = threading.Event()
    agent._model_request_active.set()
    agent._pending_redirect = None
    agent._pending_redirect_lock = threading.Lock()
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._executing_tools = False
    agent._execution_thread_id = None
    agent.api_mode = "chat_completions"
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["interrupt", "steer"])
async def test_other_members_busy_message_is_queued_not_spliced_into_the_owners_turn(mode):
    """Regression: 郭宝琪's "改" during 何钧豪's turn was spliced in as 何钧豪's own
    correction, and the rest of 何钧豪's answer was re-anchored to 郭宝琪's message."""
    rules = {"group_rules": {"oc_group": {"policy": "allowlist", "allowlist": ["ou_owner", "ou_member"]}}}
    runner, adapter, key = _busy_gateway(rules)
    runner._busy_input_mode = mode
    agent = _redirectable_owner_turn(runner, adapter, key)

    await _deliver(adapter, _group_message("ou_member", "改", "om_follow"))

    assert agent._pending_redirect is None and agent._pending_steer is None
    agent.interrupt.assert_not_called()
    queued = adapter._pending_messages[key]
    assert (queued.text, queued.source.user_id, queued.message_id) == ("改", "ou_member", "om_follow")


@pytest.mark.asyncio
async def test_the_owners_own_correction_still_redirects_the_running_turn():
    rules = {"group_rules": {"oc_group": {"policy": "allowlist", "allowlist": ["ou_owner"]}}}
    runner, adapter, key = _busy_gateway(rules)
    agent = _redirectable_owner_turn(runner, adapter, key)
    runner._busy_input_mode = "interrupt"

    await _deliver(adapter, _group_message("ou_owner", "换成方案二", "om_fix"))

    assert agent._pending_redirect == "换成方案二"
    assert key not in adapter._pending_messages


@pytest.mark.asyncio
async def test_queued_cross_sender_message_is_attributed_to_its_sender():
    rules = {"group_rules": {"oc_group": {"policy": "allowlist", "allowlist": ["ou_owner", "ou_member"]}}}
    runner, adapter, key = _busy_gateway(rules)
    runner._busy_input_mode = "interrupt"
    _redirectable_owner_turn(runner, adapter, key)

    await _deliver(adapter, _group_message("ou_member", "改", "om_follow"))
    queued = adapter._pending_messages[key]
    queued.source.user_name = "郭宝琪"
    text = await runner._prepare_inbound_message_text(event=queued, source=queued.source, history=[])

    assert "[郭宝琪] 改" in text


@pytest.mark.asyncio
async def test_two_members_texting_at_once_are_never_folded_into_one_message():
    rules = {"group_rules": {"oc_group": {"policy": "allowlist", "allowlist": ["ou_owner", "ou_member"]}}}
    _runner, adapter, _key = _busy_gateway(rules)
    dispatched = []

    async def _capture(event):
        dispatched.append((event.source.user_id, event.text))

    adapter._handle_message_with_guards = _capture
    adapter._text_batch_delay_seconds = 0.05
    await adapter._handle_message_event_data(_group_message("ou_owner", "我是郭宝琪，我点头了", "om_a"))
    await adapter._handle_message_event_data(_group_message("ou_member", "改", "om_b"))
    for _ in range(50):
        if not adapter._pending_text_batch_tasks:
            break
        await asyncio.sleep(0.02)

    assert sorted(dispatched) == [("ou_member", "改"), ("ou_owner", "我是郭宝琪，我点头了")]
