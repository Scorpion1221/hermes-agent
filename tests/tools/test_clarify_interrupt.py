"""Clarify waits must observe the existing per-agent thread interrupt signal."""

import threading
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
