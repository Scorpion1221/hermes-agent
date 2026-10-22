"""A busy-mode message landing after the turn's last read is queued as the next turn, never lost.

2026-10-21 (cloud bot, Feishu, busy_input_mode steer): a preview-card click arrived 0.7 s after the
turn's finalizer had made its last read of the steer slot, while the gateway still listed the turn as
running (it releases the agent only after the streaming final edit). The busy path "steered" the click
into the finished agent and fanned it out to the post-turn background-review fork; neither ever ran it,
the card still redrew 「已由 X 处理」, and the release waited for an approval nobody saw. The real
AIAgent now refuses a steer after its last read, and the gateway queues the event instead.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource

CLICK = 'card:button {"name": "swp_approve", "value": {"hermes_card_click": "anyone", "swp": "approve"}}'


def _finished_agent():
    """A real AIAgent that has run one turn to completion (the gateway may still list it as running)."""
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=True)
    agent.client = MagicMock()
    agent._disable_streaming = True
    agent._cached_system_prompt = "sys"
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    msg = SimpleNamespace(content="preview posted", tool_calls=None)
    reply = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], model="m", usage=None)
    agent._interruptible_api_call = lambda _kw: reply
    with (
        patch.object(agent, "_flush_messages_to_session_db"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        assert agent.run_conversation("release Web Client")["completed"] is True
    return agent


class _Fork:
    """A post-turn background-review fork still listed in the parent's _active_children."""

    def __init__(self):
        self.payload = None

    def steer(self, text):
        self.payload = text
        return True


def _click():
    source = SessionSource(platform=Platform.FEISHU, chat_id="oc_group", chat_type="group", user_id="ou_owner")
    return MessageEvent(text=CLICK, message_type=MessageType.TEXT, source=source, message_id="om_card:card:tok")


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["busy_steer_mode", "priority", "explicit_command"])
async def test_a_click_after_the_last_read_is_queued_not_swallowed(route, tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    runner = GatewayRunner(config=GatewayConfig())
    agent = _finished_agent()
    fork = _Fork()
    agent._active_children = [fork]
    agent._active_children_lock = threading.Lock()
    runner._session_state("key").turn.agent = agent            # not released yet (the 2026-10-21 window)
    queued = []
    monkeypatch.setattr(runner, "_queue_or_replace_pending_event", lambda key, ev: queued.append((key, ev)))
    monkeypatch.setattr(runner, "_delivery_adapter_for", lambda _source: object())
    monkeypatch.setattr(runner, "_enqueue_fifo", lambda key, ev, _adapter: queued.append((key, ev)))
    event = _click()
    if route == "explicit_command":
        event = MessageEvent(text="/steer " + CLICK, message_type=MessageType.TEXT, source=event.source,
                             message_id=event.message_id)

    if route == "busy_steer_mode":
        outcome = await runner._resolve_busy_steer_or_redirect(event, "key", "steer", agent)
        assert outcome.steered is False and outcome.effective_mode == "queue"   # the caller queues it
    elif route == "priority":
        runner._hm_busy_steer(event, agent, "key")
        assert queued and queued[0][1] is event
    else:
        await runner._busy_steer_command(event, "key", event.source)
        assert queued and queued[0][1].text == CLICK, "a refused /steer runs as the next turn"

    assert agent._drain_pending_steer() is None                # nothing parked on the finished agent
    assert fork.payload is None                                 # the review fork never got it
