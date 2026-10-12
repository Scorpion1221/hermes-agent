"""Clarify waits must observe the existing per-agent thread interrupt signal."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tools import clarify_gateway as cm
from tools.interrupt import is_thread_interrupted, set_interrupt


@pytest.fixture(autouse=True)
def _clear_entries():
    yield
    with cm._lock:
        keys = list(cm._session_index)
    for key in keys:
        cm.clear_session(key)


@pytest.mark.parametrize("timeout", [3600, 0, -1])
def test_agent_interrupt_wakes_clarify_without_a_user_answer(timeout):
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.quiet_mode = True
    agent._active_children_lock = threading.Lock()
    agent._active_children = []
    entry = cm.register("interrupted", "session", "Which file?", None)
    started = threading.Event()

    def wait():
        agent._execution_thread_id = threading.get_ident()
        started.set()
        try:
            return cm.wait_for_response(entry.clarify_id, timeout)
        finally:
            set_interrupt(False)

    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(wait)
        assert started.wait(1)
        try:
            agent.interrupt("Use the uploaded ZIP instead")
            assert future.result(timeout=0.75) is None
            assert cm.get_pending_for_session("session", include_choice_prompts=True) is None
        finally:
            cm.clear_session("session")


def test_interrupt_does_not_cancel_another_agent_wait():
    first = cm.register("first", "first-session", "First?", None)
    second = cm.register("second", "second-session", "Second?", None)
    starts = [threading.Event(), threading.Event()]
    tids = [None, None]

    def wait(index, entry):
        tids[index] = threading.get_ident()
        starts[index].set()
        try:
            return cm.wait_for_response(entry.clarify_id, 0)
        finally:
            set_interrupt(False)

    with ThreadPoolExecutor(2) as pool:
        first_future = pool.submit(wait, 0, first)
        second_future = pool.submit(wait, 1, second)
        assert all(start.wait(1) for start in starts)
        try:
            set_interrupt(True, tids[0])
            assert first_future.result(timeout=0.75) is None
            assert not second_future.done()
            assert not is_thread_interrupted(tids[1])
            assert cm.get_pending_for_session("second-session") is second
            assert cm.resolve_gateway_clarify(second.clarify_id, "Second answer")
            assert second_future.result(timeout=1) == "Second answer"
        finally:
            cm.clear_session("first-session")
            cm.clear_session("second-session")


def test_already_resolved_answer_wins_over_interrupt():
    entry = cm.register("answered", "answered-session", "Pick", ["A", "B"])
    assert cm.resolve_gateway_clarify(entry.clarify_id, "B")
    set_interrupt(True)
    try:
        assert cm.wait_for_response(entry.clarify_id, 0) == "B"
        assert is_thread_interrupted(threading.get_ident())
    finally:
        set_interrupt(False)


def test_interrupting_a_gateway_batch_cancels_it_instead_of_asking_the_next_question():
    """/stop (or a new message) while question 1 waits ends the batch as cancelled: the user moved
    on, so question 2 is never posted and the result does not claim they went idle."""
    import asyncio
    from types import SimpleNamespace

    from gateway.platforms.base import SendResult
    from gateway.run_turn_runner import TurnRunner

    asked = []

    class _Adapter:
        def pause_typing_for_chat(self, chat_id):
            return None

        def resume_typing_for_chat(self, chat_id):
            return None

        async def send_clarify(self, **kwargs):
            asked.append(kwargs["question"].split("\n")[0])
            return SendResult(success=True, message_id="m1")

    class _Done:
        def __init__(self, value):
            self._value = value

        def result(self, timeout=None):
            return self._value

    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        _status_adapter=_Adapter(), _status_chat_id="C1", _status_thread_metadata={},
        session_key="batch-session", stream_consumer_holder=[None])
    runner._schedule = lambda coro, label: _Done(asyncio.run(coro))
    runner._close_native_stream_boundary = lambda *a, **k: None
    questions = [{"qid": f"q{i}", "question": q, "choices": ["a", "b"], "multi_select": False}
                 for i, q in enumerate(("Where are the files?", "Which format?"))]
    tid = []

    def ask():
        tid.append(threading.get_ident())
        try:
            return runner._clarify_callback_sync(questions)
        finally:
            set_interrupt(False)

    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(ask)
        try:
            deadline = time.monotonic() + 2
            while cm.get_pending_for_session("batch-session", include_choice_prompts=True) is None:
                assert time.monotonic() < deadline
                time.sleep(0.01)
            set_interrupt(True, tid[0])
            reply = future.result(timeout=2)
        finally:
            cm.clear_session("batch-session")

    assert reply == {"answers": {}, "outcome": "cancelled"}
    assert asked == ["Where are the files?"]
