"""Responses-API watchdogs on hosted non-OpenAI backends (e.g. a relay translating to Claude).

Such a relay emits ``response.created`` and then nothing while the model thinks (thinking is
redacted), often for minutes. The idle watchdog must not treat that as a dead stream at the
Codex-tuned budget, and a stream that keeps producing events must not be killed by the
wall-clock stale timeout sized for a whole non-streaming call — only the hard ceiling bounds it.
The official Codex backend and local Responses servers keep their own policies.
"""

from __future__ import annotations

import sys
import time
import types
from types import SimpleNamespace

import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

RELAY_URL = "https://relay.example/v1"
SMALL_PROMPT = {"model": "claude-opus-5-5-combos", "input": "x" * 4_000}


def _make_agent(tmp_path, monkeypatch, *, provider="custom", base_url=RELAY_URL, model="claude-opus-5-5-combos"):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    for name in ("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "HERMES_CODEX_HARD_TIMEOUT_SECONDS",
                 "HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "HERMES_STREAM_STALE_TIMEOUT",
                 "HERMES_API_CALL_STALE_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    from run_agent import AIAgent

    agent = AIAgent(model=model, provider=provider, api_key="sk-dummy", base_url=base_url,
                    quiet_mode=True, skip_context_files=True, skip_memory=True, platform="cli")
    agent.api_mode = "codex_responses"
    monkeypatch.setattr(agent, "_emit_status", lambda *a, **k: None)
    return agent


def _install_event_stream(agent, monkeypatch, event_factory, closes):
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_kwargs: event_factory()))
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **_kwargs: client)
    monkeypatch.setattr(agent, "_abort_request_openai_client", lambda _c, reason=None: closes.append(reason))
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda _c, reason=None: closes.append(reason))


def _completed(resp_id="resp-1"):
    return SimpleNamespace(type="response.completed",
                           response=SimpleNamespace(status="completed", id=resp_id, usage=None))


def test_relay_backend_idle_gets_stream_detector_patience(tmp_path, monkeypatch):
    from agent import chat_completion_helpers as h

    agent = _make_agent(tmp_path, monkeypatch)
    assert not h._is_openai_codex_backend(agent)

    wd = h._resolve_nonstream_watchdogs(agent, SMALL_PROMPT)

    # Same silence budget the chat-completions stream detector would allow this request.
    assert wd.idle_enabled
    assert wd.idle_timeout >= h._cloud_stale_timeout_for(agent, SMALL_PROMPT)
    assert wd.idle_timeout > 12.0


def test_relay_backend_post_event_budget_is_the_hard_ceiling(tmp_path, monkeypatch):
    from agent import chat_completion_helpers as h

    agent = _make_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_HARD_TIMEOUT_SECONDS", "777")
    wd = h._resolve_nonstream_watchdogs(agent, SMALL_PROMPT)

    assert wd.stale_after(False) == wd.stale_timeout
    assert wd.stale_after(True) == 777.0


def test_relay_backend_post_event_budget_still_honours_run_budget(tmp_path, monkeypatch):
    from agent import chat_completion_helpers as h

    agent = _make_agent(tmp_path, monkeypatch)
    agent.run_budget_seconds = 200
    agent._run_budget_started_at = time.time() - 10

    wd = h._resolve_nonstream_watchdogs(agent, SMALL_PROMPT)

    assert wd.stale_after(True) <= 100.0


@pytest.mark.parametrize("provider,base_url", [
    ("openai-codex", "https://chatgpt.com/backend-api/codex"),
    ("openai-api", "https://api.openai.com/v1"),
    ("custom", "http://127.0.0.1:8080/v1"),
], ids=["openai-codex", "openai-api", "local"])
def test_codex_backend_and_local_servers_keep_their_policy(tmp_path, monkeypatch, provider, base_url):
    from agent import chat_completion_helpers as h

    agent = _make_agent(tmp_path, monkeypatch, provider=provider, base_url=base_url, model="gpt-5.5")
    wd = h._resolve_nonstream_watchdogs(agent, {"model": "gpt-5.5", "input": "x" * 4_000})

    assert wd.stale_after(True) == wd.stale_timeout
    assert wd.idle_timeout < h._cloud_stale_timeout_for(agent, {"model": "gpt-5.5", "input": "x" * 4_000})


def test_explicit_idle_override_still_wins(tmp_path, monkeypatch):
    from agent import chat_completion_helpers as h

    agent = _make_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "42")

    assert h._resolve_nonstream_watchdogs(agent, SMALL_PROMPT).idle_timeout == 42.0


def test_streaming_relay_call_outlives_the_wall_clock_stale_timeout(tmp_path, monkeypatch):
    from agent import chat_completion_helpers as h

    agent = _make_agent(tmp_path, monkeypatch)
    # Wall-clock budget for a whole non-streaming call.
    monkeypatch.setattr(agent, "_compute_non_stream_stale_timeout", lambda *a, **k: 0.5)
    closes: list = []

    def generating_stream():
        yield SimpleNamespace(type="response.created")
        deadline = time.time() + 1.5  # events keep arriving well past the wall-clock budget
        while time.time() < deadline:
            yield SimpleNamespace(type="response.output_text.delta", delta=".")
            time.sleep(0.05)
        yield _completed()

    _install_event_stream(agent, monkeypatch, generating_stream, closes)

    response = h.interruptible_api_call(agent, dict(SMALL_PROMPT))

    assert response.output_text.startswith(".")
    assert "stale_call_kill" not in closes


def test_streaming_relay_call_still_hits_the_hard_ceiling(tmp_path, monkeypatch):
    from agent import chat_completion_helpers as h

    agent = _make_agent(tmp_path, monkeypatch)
    monkeypatch.setattr(agent, "_compute_non_stream_stale_timeout", lambda *a, **k: 0.5)
    monkeypatch.setenv("HERMES_CODEX_HARD_TIMEOUT_SECONDS", "1.2")
    closes: list = []

    def endless_stream():
        yield SimpleNamespace(type="response.created")
        deadline = time.time() + 10
        while time.time() < deadline:
            yield SimpleNamespace(type="response.output_text.delta", delta=".")
            time.sleep(0.05)
        yield _completed()

    _install_event_stream(agent, monkeypatch, endless_stream, closes)

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="threshold: 1s"):
        h.interruptible_api_call(agent, dict(SMALL_PROMPT))
    assert time.monotonic() - started < 6
    assert "stale_call_kill" in closes


@pytest.mark.parametrize("provider,base_url,counted", [
    ("custom", RELAY_URL, True),
    ("openai-codex", "https://chatgpt.com/backend-api/codex", False),
], ids=["relay", "openai-codex"])
def test_relay_idle_kill_counts_toward_stale_giveup(tmp_path, monkeypatch, provider, base_url, counted):
    """With the longer patience, a relay that keeps dying after its first event must trip the
    consecutive-stale give-up instead of burning that budget every attempt."""
    from agent import chat_completion_helpers as h

    agent = _make_agent(tmp_path, monkeypatch, provider=provider, base_url=base_url)
    monkeypatch.setattr(agent, "_compute_non_stream_stale_timeout", lambda *a, **k: 30.0)
    original = h._resolve_nonstream_watchdogs

    def resolve(a, kw):
        wd = original(a, kw)
        wd.idle_timeout = 0.4
        wd.idle_requires_progress = False
        wd.progress_timeout = 0.0
        return wd

    monkeypatch.setattr(h, "_resolve_nonstream_watchdogs", resolve)
    closes: list = []

    def stalled_stream():
        yield SimpleNamespace(type="response.created")
        while getattr(agent, "_active_codex_stream_request_token", None) is not None:
            time.sleep(0.02)
        raise ConnectionError("retired stalled stream")

    _install_event_stream(agent, monkeypatch, stalled_stream, closes)
    agent._consecutive_stale_streams = 0

    with pytest.raises(TimeoutError, match="no SSE events"):
        h.interruptible_api_call(agent, dict(SMALL_PROMPT))

    assert "codex_stream_idle_kill" in closes
    assert (agent._consecutive_stale_streams == 1) is counted
