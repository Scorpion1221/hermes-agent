"""The host learns the moment a mid-turn follow-up enters the model's context.

A gateway splits its live reply there (Feishu: the answer to the follow-up streams in a new card
below the user's message instead of growing the card above it). The callback must fire only once the
follow-up row is in ``messages`` — never on receipt and never when the steer is requeued — and must
not change a byte of what the model sees.
"""

from __future__ import annotations

import copy
import threading

from agent.prompt_builder import STEER_MARKER_OPEN
from run_agent import AIAgent


def _agent(events):
    agent = object.__new__(AIAgent)
    agent._pending_steer = None
    agent._pending_steer_lock = threading.Lock()
    agent._pending_redirect = None
    agent._pending_redirect_lock = threading.Lock()
    agent._model_request_active = threading.Event()
    agent._executing_tools = False
    agent._execution_thread_id = None
    agent._interrupt_thread_signal_pending = False
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent._tool_worker_threads = None
    agent._tool_worker_threads_lock = None
    agent._current_streamed_assistant_text = ""
    agent._stream_needs_break = False
    agent._strip_think_blocks = lambda content: content
    agent.quiet_mode = True
    agent.api_mode = "chat_completions"
    agent.steer_consumed_callback = lambda: events.append("consumed")
    return agent


def _tool_turn():
    return [
        {"role": "system", "content": "cached prefix"},
        {"role": "user", "content": "original task"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "tool output"},
    ]


def _assert_appended_once(before, after, events):
    assert after[: len(before)] == before, "earlier rows are byte-identical (prompt cache)"
    assert len(after) == len(before) + 1 and after[-1]["role"] == "user"
    assert STEER_MARKER_OPEN in after[-1]["content"]
    assert events == ["consumed"]


def test_steer_drained_after_a_tool_batch_notifies_once_the_row_is_appended():
    from agent.agent_runtime_helpers import apply_pending_steer_to_tool_results
    from agent.tool_executor import _finalize_tool_batch
    from tools.budget_config import DEFAULT_BUDGET

    events = []
    agent = _agent(events)
    seen_at_notify = []
    agent.steer_consumed_callback = lambda: (events.append("consumed"), seen_at_notify.append(len(messages)))
    agent._apply_pending_steer_to_tool_results = lambda msgs, n: apply_pending_steer_to_tool_results(agent, msgs, n)
    messages = _tool_turn()
    before = copy.deepcopy(messages)
    agent.steer("use the staging data")

    _finalize_tool_batch(agent, messages, "task", 1, DEFAULT_BUDGET)

    _assert_appended_once(before, messages, events)
    assert seen_at_notify == [len(before) + 1], "the host hears of it only once the row is in the context"


def test_steer_drained_before_the_next_api_call_notifies():
    from agent.turn_iteration_prep import _inject_steer_after_newest_tool_result

    events = []
    agent = _agent(events)
    messages = _tool_turn()
    before = copy.deepcopy(messages)
    agent.steer("use the staging data")

    _inject_steer_after_newest_tool_result(agent, messages, agent._drain_pending_steer())

    _assert_appended_once(before, messages, events)


def test_a_steer_with_no_tool_result_to_follow_is_requeued_without_notifying():
    from agent.turn_iteration_prep import _inject_steer_after_newest_tool_result

    events = []
    agent = _agent(events)
    messages = [{"role": "system", "content": "cached prefix"}, {"role": "user", "content": "task"}]
    agent.steer("later")

    _inject_steer_after_newest_tool_result(agent, messages, agent._drain_pending_steer())

    assert events == []
    assert agent._pending_steer == "later"


def test_a_redirect_applied_at_the_iteration_start_notifies():
    from types import SimpleNamespace

    from agent.turn_iteration_prep import begin_iteration

    events = []
    agent = _agent(events)
    agent._persist_session = lambda *a, **k: None
    agent._checkpoint_mgr = SimpleNamespace(new_turn=lambda: None)
    agent.iteration_budget = SimpleNamespace(consume=lambda: True, remaining=10)
    agent.max_iterations = 10
    agent._budget_grace_call = False
    messages = _tool_turn()
    before = copy.deepcopy(messages)
    agent._pending_redirect = "what about tomorrow?"

    begin_iteration(agent, messages=messages, conversation_history=[], original_user_message="task",
                    api_call_count=0, interrupted=False, _turn_exit_reason="unknown")

    assert messages[: len(before)] == before
    assert messages[-1]["role"] == "user" and messages[-1]["content"] == "what about tomorrow?"
    assert events == ["consumed"]


def test_a_failing_callback_never_breaks_the_turn():
    from agent.turn_iteration_prep import _inject_steer_after_newest_tool_result

    agent = _agent([])

    def _boom():
        raise RuntimeError("host is gone")

    agent.steer_consumed_callback = _boom
    messages = _tool_turn()
    agent.steer("note")

    _inject_steer_after_newest_tool_result(agent, messages, agent._drain_pending_steer())

    assert messages[-1]["role"] == "user"
