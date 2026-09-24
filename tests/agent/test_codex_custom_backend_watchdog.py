"""Responses-API watchdogs on non-OpenAI backends (e.g. a Claude proxy).

A proxy translating the Responses API to Claude emits ``response.created``
and then nothing while the model thinks (thinking is redacted), often for
1-3 minutes. The idle watchdog must not treat that as a dead stream at the
Codex-tuned budget, a stream that is producing events must not be killed by
the wall-clock stale timeout sized for whole non-streaming calls, and a
watchdog abort must not trigger a hidden inline retry.
"""

from __future__ import annotations

import sys
import time
import types
from types import SimpleNamespace

import httpx
import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

CLAUDE_VIA_PROXY = "claude-opus-5-5-combos"
LARGE_INPUT = "x" * 440_000  # ~110k estimated tokens


def _make_agent(tmp_path, monkeypatch, *, provider, base_url, model):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    from run_agent import AIAgent

    agent = AIAgent(
        model=model,
        provider=provider,
        api_key="sk-dummy",
        base_url=base_url,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    agent.api_mode = "codex_responses"
    monkeypatch.setattr(agent, "_emit_status", lambda *a, **k: None)
    return agent


def _proxy_agent(tmp_path, monkeypatch):
    return _make_agent(
        tmp_path, monkeypatch,
        provider="custom", base_url="https://proxy.example/v1", model=CLAUDE_VIA_PROXY,
    )


def test_proxy_backend_gets_stream_detector_patience(tmp_path, monkeypatch):
    from agent import chat_completion_helpers as h

    agent = _proxy_agent(tmp_path, monkeypatch)
    api_kwargs = {"model": CLAUDE_VIA_PROXY, "input": LARGE_INPUT}
    assert not h._is_openai_codex_backend(agent)

    idle = h.codex_stream_idle_timeout(
        agent, api_kwargs, openai_codex_backend=False, default=180.0,
    )

    # Same budget the chat-completions/Anthropic stream detector would use.
    assert idle == h._derive_stream_stale_timeout(agent, api_kwargs)
    assert idle > 180.0


def test_openai_codex_backend_keeps_its_own_idle_budget(tmp_path, monkeypatch):
    from agent import chat_completion_helpers as h

    agent = _proxy_agent(tmp_path, monkeypatch)
    idle = h.codex_stream_idle_timeout(
        agent, {"model": "gpt-5.5", "input": LARGE_INPUT}, openai_codex_backend=True, default=180.0,
    )

    assert idle == 180.0


def test_explicit_idle_override_still_wins(tmp_path, monkeypatch):
    from agent import chat_completion_helpers as h

    agent = _proxy_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "42")

    idle = h.codex_stream_idle_timeout(
        agent, {"model": CLAUDE_VIA_PROXY, "input": LARGE_INPUT}, openai_codex_backend=False, default=180.0,
    )

    assert idle == 42.0


def _wire_fake_clients(agent, monkeypatch):
    closes: list = []
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: SimpleNamespace())
    monkeypatch.setattr(agent, "_abort_request_openai_client", lambda c, reason=None: closes.append(reason))
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda c, reason=None: closes.append(reason))
    return closes


def test_streaming_proxy_call_outlives_the_wall_clock_stale_timeout(tmp_path, monkeypatch):
    from agent import chat_completion_helpers as h

    agent = _proxy_agent(tmp_path, monkeypatch)
    # Wall-clock budget for a whole non-streaming call.
    monkeypatch.setattr(agent, "_compute_non_stream_stale_timeout", lambda *a, **k: 0.5)
    closes = _wire_fake_clients(agent, monkeypatch)
    sentinel = SimpleNamespace(ok=True)

    def generating_stream(api_kwargs, client=None, on_first_delta=None):
        # Events keep arriving for well past the wall-clock budget.
        deadline = time.time() + 1.5
        while time.time() < deadline:
            agent._codex_stream_last_event_ts = time.time()
            time.sleep(0.05)
        return sentinel

    monkeypatch.setattr(agent, "_run_codex_stream", generating_stream)

    assert h.interruptible_api_call(agent, {"model": CLAUDE_VIA_PROXY, "input": "hi"}) is sentinel
    assert "stale_call_kill" not in closes


def test_streaming_proxy_call_still_hits_the_hard_ceiling(tmp_path, monkeypatch):
    from agent import chat_completion_helpers as h

    agent = _proxy_agent(tmp_path, monkeypatch)
    monkeypatch.setattr(agent, "_compute_non_stream_stale_timeout", lambda *a, **k: 0.5)
    monkeypatch.setenv("HERMES_CODEX_HARD_TIMEOUT_SECONDS", "1.2")
    closes = _wire_fake_clients(agent, monkeypatch)
    stop = {"flag": False}

    def endless_keepalives(api_kwargs, client=None, on_first_delta=None):
        while not stop["flag"] and not agent._interrupt_requested:
            agent._codex_stream_last_event_ts = time.time()
            time.sleep(0.05)
        raise RuntimeError("connection closed")

    monkeypatch.setattr(agent, "_run_codex_stream", endless_keepalives)

    started = time.time()
    try:
        with pytest.raises(TimeoutError):
            h.interruptible_api_call(agent, {"model": CLAUDE_VIA_PROXY, "input": "hi"})
        assert time.time() - started < 10
        assert "stale_call_kill" in closes
    finally:
        stop["flag"] = True


class _FailingResponses:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        raise httpx.RemoteProtocolError("peer closed connection")


@pytest.mark.parametrize("aborted", [False, True])
def test_aborted_client_is_not_retried_inline(tmp_path, monkeypatch, aborted):
    from agent.codex_runtime import run_codex_stream

    agent = _proxy_agent(tmp_path, monkeypatch)
    agent._interrupt_requested = False
    responses = _FailingResponses()
    client = SimpleNamespace(responses=responses)
    if aborted:
        # Set by the watchdog/interrupt abort from another thread.
        client._hermes_abort_reason = "codex_stream_idle_kill"

    with pytest.raises(httpx.RemoteProtocolError):
        run_codex_stream(agent, {"model": CLAUDE_VIA_PROXY, "input": "hi"}, client=client)

    # A watchdog abort leaves recovery to the caller's retry loop; an
    # ordinary transport blip keeps its one inline retry.
    assert responses.calls == (1 if aborted else 2)


def test_abort_marks_the_request_client(tmp_path, monkeypatch):
    agent = _proxy_agent(tmp_path, monkeypatch)
    client = SimpleNamespace()
    monkeypatch.setattr(agent, "_force_close_tcp_sockets", lambda c: 1)

    agent._abort_request_openai_client(client, reason="codex_stream_idle_kill")

    assert client._hermes_abort_reason == "codex_stream_idle_kill"
