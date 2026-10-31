"""The fork tool whitelist (background review, /btw) must hold where tools actually run.

Both executors dispatch each call on a pool worker. The whitelist used to be a thread-local
set on the fork's thread, so the worker never saw it and every tool call went through.
"""

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.tool_executor import execute_tool_calls_concurrent, execute_tool_calls_sequential
from hermes_cli.plugins import (
    clear_thread_tool_whitelist,
    get_pre_tool_call_block_message,
    set_thread_tool_whitelist,
)
from run_agent import AIAgent
from tools.daemon_pool import DaemonThreadPoolExecutor
from tools.thread_context import propagate_context_to_thread


def test_whitelist_follows_propagated_context_but_not_bare_threads(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda hook_name, **kwargs: [])

    def check(name="write_file"):
        return get_pre_tool_call_block_message(name, {})

    bare: dict = {}
    executor = DaemonThreadPoolExecutor(max_workers=1)
    set_thread_tool_whitelist({"memory"}, deny_msg_fmt="denied {tool_name}")
    try:
        propagated = executor.submit(propagate_context_to_thread(check)).result(timeout=10)
        allowed = executor.submit(propagate_context_to_thread(lambda: check("memory"))).result(timeout=10)
        worker = threading.Thread(target=lambda: bare.setdefault("msg", check()))
        worker.start()
        worker.join(timeout=10)
    finally:
        clear_thread_tool_whitelist()
        executor.shutdown(wait=False)

    assert propagated == "denied write_file"
    assert allowed is None
    assert bare["msg"] is None  # a thread that never copied the context stays unrestricted
    assert check() is None  # clear took effect for the caller


def _make_agent(tmp_path: Path) -> AIAgent:
    with (
        patch(
            "model_tools.get_tool_definitions",
            return_value=[
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": "test tool",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
                for name in ("web_extract", "skills_list")
            ],
        ),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("run_agent._hermes_home", tmp_path),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._flush_messages_to_session_db = MagicMock(return_value=True)
    agent._append_guardrail_observation = MagicMock(side_effect=lambda _name, _args, result, **_kw: result)
    agent._record_file_mutation_result = MagicMock()
    agent._subdirectory_hints.check_tool_call = MagicMock(return_value="")
    agent._tool_result_content_for_active_model = MagicMock(side_effect=lambda _name, result: result)
    return agent


def _call(call_id: str, name: str):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments="{}"))


@pytest.mark.parametrize("executor", [execute_tool_calls_sequential, execute_tool_calls_concurrent])
def test_whitelist_blocks_on_the_tool_worker(tmp_path, executor):
    agent = _make_agent(tmp_path)
    dispatched: list[str] = []

    def _dispatch(name, _args, _task_id, *, tool_call_id, **_kwargs):
        dispatched.append(name)
        return "ran"

    calls = [_call("denied", "web_extract"), _call("allowed", "skills_list")]
    messages: list[dict] = []
    set_thread_tool_whitelist({"skills_list"}, deny_msg_fmt="Review denied {tool_name}")
    try:
        with patch("model_tools.handle_function_call", side_effect=_dispatch):
            executor(agent, SimpleNamespace(tool_calls=calls), messages, "task")
    finally:
        clear_thread_tool_whitelist()

    by_id = {m["tool_call_id"]: m["content"] for m in messages}
    assert dispatched == ["skills_list"]
    assert "Review denied web_extract" in by_id["denied"]
    assert by_id["allowed"] == "ran"
